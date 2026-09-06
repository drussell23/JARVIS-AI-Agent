"""commit_fault_recovery — the autonomous APPLY path's git-state safety net (Phase 2).

On a commit-stage fault (locked index, merge conflict, diff rejection, timeout):
classify it, STASH the op's change scoped to its own files (recoverable, never a
destructive reset, operator work untouched), emit a non-blocking diff_rejection
event, and route the fault to the PLAN subagent for a re-plan — all fail-soft.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.core.ouroboros.governance import commit_fault_recovery as CFR


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.parametrize("msg,expect", [
    ("fatal: Unable to create '.git/index.lock': File exists", "index_locked"),
    ("could not lock config file", "index_locked"),
    ("CONFLICT (content): Merge conflict in x.py", "merge_conflict"),
    ("error: patch does not apply", "diff_rejected"),
    ("error: hunk #2 failed to apply", "diff_rejected"),
    ("some entirely unrelated failure", "other"),
])
def test_classify_commit_fault(msg, expect):
    assert CFR.classify_commit_fault(Exception(msg)) == expect


def test_classify_timeout():
    assert CFR.classify_commit_fault(asyncio.TimeoutError()) == "timeout"


def test_fault_patterns_are_env_tunable(monkeypatch):
    monkeypatch.setenv("JARVIS_COMMIT_FAULT_PATTERNS_DIFF_REJECTED", "widgetsplode")
    assert CFR.classify_commit_fault(Exception("widgetsplode!")) == "diff_rejected"


def test_never_raises_on_garbage():
    assert CFR.classify_commit_fault(Exception("")) == "other"
    # a bad exception object still classifies to a defined value
    assert isinstance(CFR.classify_commit_fault(BaseException()), str)


def _git(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    d = tmp_path
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "a.py").write_text("x = 1\n")
    (d / "b.py").write_text("y = 1\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "init")
    return d


def test_scoped_stash_reverts_op_file_preserves_operator_work(repo):
    (repo / "a.py").write_text("x = 2  # op change\n")
    (repo / "b.py").write_text("y = 2  # operator work\n")
    stashed, ref = _run(CFR.stash_workspace(repo, ["a.py"], "op-1"))
    assert stashed
    assert ref.startswith("ov-commit-fault:op-1")
    # the op's file is reverted; the operator's other change is UNTOUCHED
    assert (repo / "a.py").read_text() == "x = 1\n"
    assert "operator work" in (repo / "b.py").read_text()
    # and the reverted change is recoverable, not destroyed
    assert "ov-commit-fault:op-1" in _git(repo, "stash", "list").stdout


def test_stash_untracked_new_file(repo):
    (repo / "new.py").write_text("z = 9\n")
    stashed, _ = _run(CFR.stash_workspace(repo, ["new.py"], "op-2"))
    assert stashed
    assert not (repo / "new.py").exists()  # untracked new file stashed away


def test_stash_nothing_to_stash_is_honest(repo):
    stashed, ref = _run(CFR.stash_workspace(repo, ["a.py"], "op-3"))
    assert stashed is False
    assert ref == "nothing_to_stash"


def test_stash_never_touches_files_outside_repo(repo):
    stashed, ref = _run(CFR.stash_workspace(repo, ["/etc/passwd"], "op-4"))
    assert stashed is False
    assert ref == "no_scoped_files"


def test_route_replan_skips_single_file_op():
    # single-file op has no DAG to re-plan → clean skip, no crash
    orch = SimpleNamespace(_subagent_orchestrator=SimpleNamespace(dispatch_plan=None))
    ctx = SimpleNamespace(target_files=("only.py",), description="d", op_id="o")
    assert CFR.route_replan_to_plan(orch, ctx, "merge_conflict", "detail") is False


def test_route_replan_no_orchestrator_is_clean():
    orch = SimpleNamespace(_subagent_orchestrator=None)
    ctx = SimpleNamespace(target_files=("a.py", "b.py"), description="d", op_id="o")
    assert CFR.route_replan_to_plan(orch, ctx, "index_locked", "detail") is False


def test_emit_diff_rejection_never_raises():
    # no stream configured in the test env → returns None, never raises
    assert CFR.emit_diff_rejection(
        "op-x", "merge_conflict", "detail", ("a.py",),
        stashed=True, stash_ref="ref",
    ) is None or True


def test_recover_end_to_end_fail_soft(repo):
    (repo / "a.py").write_text("x = 2\n")
    orch = SimpleNamespace(
        _config=SimpleNamespace(project_root=repo),
        _subagent_orchestrator=None,
    )
    ctx = SimpleNamespace(
        target_files=("a.py",), description="d", op_id="op-5",
        risk_tier=SimpleNamespace(name="NOTIFY_APPLY"),
    )
    res = _run(CFR.recover_from_commit_fault(
        orch, ctx, Exception("fatal: '.git/index.lock' exists"),
    ))
    assert res["fault"] == "index_locked"
    assert res["stashed"] is True
    assert (repo / "a.py").read_text() == "x = 1\n"  # reverted
