"""precommit_ast_guard — the fail-closed structural no-op abort at the commit
boundary (Phase 3, Anti-Venom Vector 1 scaled to live APPLY).

The guard reports ``is_noop=True`` ONLY when it can PROVE every target file is a
pure whitespace/comment no-op vs HEAD. The direction of caution is the whole
point: a no-op verdict ABORTS a commit, so anything that could hide a real change
— a literal-value edit, a docstring addition, a new file, a syntax error, a
non-Python target — must keep ``is_noop=False`` and let the commit proceed.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from backend.core.ouroboros.governance import precommit_ast_guard as G


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _git(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    d = tmp_path
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "m.py").write_text(
        "def f(x):\n"
        "    # original comment\n"
        "    total = 0\n"
        "    return total + x\n"
    )
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "init")
    return d


def _check(repo, files):
    return _run(G.check_precommit_structural_noop(repo, files))


# ---- the no-op cases: PROVEN structural identity → abort ----

def test_byte_identical_is_noop(repo):
    v = _check(repo, ["m.py"])
    assert v.is_noop is True
    assert v.reason == "all_files_structural_noop"
    assert v.files_matched == 1


def test_whitespace_only_change_is_noop(repo):
    (repo / "m.py").write_text(
        "def f(x):\n"
        "    # original comment\n"
        "    total     =    0\n"          # extra spaces
        "    return  total   +   x\n"
    )
    assert _check(repo, ["m.py"]).is_noop is True


def test_comment_only_change_is_noop(repo):
    (repo / "m.py").write_text(
        "def f(x):\n"
        "    # a COMPLETELY different comment\n"   # comment churn only
        "    total = 0\n"
        "    return total + x\n"
    )
    assert _check(repo, ["m.py"]).is_noop is True


def test_blank_line_reflow_is_noop(repo):
    (repo / "m.py").write_text(
        "def f(x):\n"
        "\n"
        "    # original comment\n"
        "    total = 0\n"
        "\n"
        "    return total + x\n"
    )
    assert _check(repo, ["m.py"]).is_noop is True


# ---- the real-change cases: a commit MUST proceed ----

def test_literal_value_change_is_not_noop(repo):
    # THE critical case: a one-literal fix must never be judged a no-op.
    (repo / "m.py").write_text(
        "def f(x):\n"
        "    # original comment\n"
        "    total = 999\n"               # 0 -> 999 is real work
        "    return total + x\n"
    )
    v = _check(repo, ["m.py"])
    assert v.is_noop is False
    assert v.reason == "structural_change"


def test_string_literal_change_is_not_noop(repo):
    (repo / "m.py").write_text(
        'def f(x):\n'
        '    total = 0\n'
        '    return "changed"\n'
    )
    assert _check(repo, ["m.py"]).is_noop is False


def test_docstring_addition_is_not_noop(repo):
    # DocStaleness's entire yield is adding docstrings — never suppress it.
    (repo / "m.py").write_text(
        "def f(x):\n"
        '    """Now documented."""\n'
        "    # original comment\n"
        "    total = 0\n"
        "    return total + x\n"
    )
    assert _check(repo, ["m.py"]).is_noop is False


def test_added_statement_is_not_noop(repo):
    (repo / "m.py").write_text(
        "def f(x):\n"
        "    total = 0\n"
        "    total += 1\n"
        "    return total + x\n"
    )
    assert _check(repo, ["m.py"]).is_noop is False


# ---- cannot-prove cases: default to letting the commit proceed ----

def test_new_file_is_not_noop(repo):
    (repo / "brand_new.py").write_text("y = 1\n")
    v = _check(repo, ["brand_new.py"])
    assert v.is_noop is False
    assert v.reason == "new_file"


def test_syntax_error_is_not_noop(repo):
    (repo / "m.py").write_text("def f(x):\n    return (\n")  # unbalanced
    v = _check(repo, ["m.py"])
    assert v.is_noop is False
    assert v.reason == "syntax_error"


def test_non_python_target_is_not_noop(repo):
    (repo / "requirements.txt").write_text("flask==1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "reqs")
    (repo / "requirements.txt").write_text("flask==1\n\n")  # whitespace churn
    v = _check(repo, ["requirements.txt"])
    assert v.is_noop is False
    assert v.reason == "non_python_target"


def test_empty_target_list_is_not_noop(repo):
    v = _check(repo, [])
    assert v.is_noop is False
    assert v.reason == "no_target_files"


def test_absolute_path_inside_repo_resolves(repo):
    v = _check(repo, [str(repo / "m.py")])
    assert v.is_noop is True


def test_path_outside_repo_is_not_noop(repo):
    v = _check(repo, ["/etc/passwd"])
    assert v.is_noop is False


# ---- multi-file semantics: ALL must be no-op to abort ----

def test_multifile_all_noop_aborts(repo):
    (repo / "n.py").write_text("a = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add n")
    # both files: comment/whitespace churn only
    (repo / "m.py").write_text(
        "def f(x):\n    # churned\n    total = 0\n    return total + x\n"
    )
    (repo / "n.py").write_text("a   =   1   # churn\n")
    v = _check(repo, ["m.py", "n.py"])
    assert v.is_noop is True
    assert v.files_matched == 2


def test_multifile_one_real_change_proceeds(repo):
    (repo / "n.py").write_text("a = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add n")
    (repo / "m.py").write_text(
        "def f(x):\n    # churned\n    total = 0\n    return total + x\n"
    )
    (repo / "n.py").write_text("a = 2\n")  # real literal change
    v = _check(repo, ["m.py", "n.py"])
    assert v.is_noop is False
    assert v.reason == "structural_change"


# ---- master switch + resilience ----

def test_disabled_never_reports_noop(repo, monkeypatch):
    monkeypatch.setenv("JARVIS_PRECOMMIT_AST_GUARD_ENABLED", "false")
    v = _check(repo, ["m.py"])
    assert v.is_noop is False
    assert v.reason == "guard_disabled"


def test_never_raises_on_bad_repo_root():
    v = _run(G.check_precommit_structural_noop("/nonexistent/xyz", ["a.py"]))
    assert v.is_noop is False  # git show fails → treated as new/unprovable


def test_verdict_to_dict_roundtrips(repo):
    v = _check(repo, ["m.py"])
    d = v.to_dict()
    assert d["is_noop"] is True
    assert d["files_checked"] == 1
    assert isinstance(d["per_file"], list)
