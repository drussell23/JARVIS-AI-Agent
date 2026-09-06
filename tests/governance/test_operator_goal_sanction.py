"""operator_goal_sanction — author + sign + scoped-inject, verified against the
cage's OWN verifier.

Phase 1 of the /goal ingestion pivot: a cockpit-authored, cryptographically
signed, file-scoped goal must (a) write the ``target_files`` key the reader
parses — NOT the ``files`` key the pre-refactor sanction_goal.py wrote, which
the reader dropped so the cage refused it ``goal_has_no_scope`` — and (b) mint
a provenance claim the real ``delegated_provenance.verify_provenance_claim``
accepts for in-scope files and refuses for out-of-scope ones (scope-laundering
guard). Everything is exercised end to end with a test secret + a tmp roadmap;
no live daemon.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.ouroboros.governance import operator_goal_sanction as ogs


@pytest.fixture
def sanction_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    roadmap = tmp_path / "roadmap.yaml"
    monkeypatch.setenv("JARVIS_ROADMAP_READER_HMAC_SECRET", "unit-secret-xyz")
    monkeypatch.setenv("JARVIS_ROADMAP_PATH", str(roadmap))
    monkeypatch.setenv("JARVIS_ROADMAP_READER_PATH", str(roadmap))
    monkeypatch.setenv("JARVIS_DELEGATED_PROVENANCE_ENABLED", "true")
    monkeypatch.setenv("JARVIS_OPERATOR_ID", "tester")
    try:
        from backend.core.ouroboros.governance.delegated_provenance import (
            reset_provenance_cache_for_tests,
        )
        reset_provenance_cache_for_tests()
    except Exception:
        pass
    return roadmap


_TARGET = "backend/core/ouroboros/governance/op_recap.py"


def _spec(goal_id: str = "goal-1", files=(_TARGET,)) -> "ogs.GoalSpec":
    return ogs.GoalSpec(
        goal_id=goal_id, title="Fix the recap",
        description="Make the recap do X.",
        target_files=tuple(files), target_symbols=("compose_recap",),
        success_criteria="recap shows X", max_duration_s=1800, note="unit",
    )


# ── authoring ────────────────────────────────────────────────────────────


def test_author_writes_target_files_key_not_files(sanction_env):
    res = ogs.author_and_sign_goal(_spec(), path_override=sanction_env)
    assert res.ok, res.to_dict()
    text = sanction_env.read_text(encoding="utf-8")
    # The reader parses ``target_files``; the pre-refactor ``files`` key is the
    # bug this whole pivot exists to not reproduce.
    assert "target_files:" in text
    assert "\n  files:" not in text
    assert "signature:" in text


def test_author_refuses_unset_secret(sanction_env, monkeypatch):
    monkeypatch.delenv("JARVIS_ROADMAP_READER_HMAC_SECRET", raising=False)
    res = ogs.author_and_sign_goal(_spec(), path_override=sanction_env)
    assert not res.ok
    assert res.reason == "secret_unset"


def test_author_refuses_unscoped_goal(sanction_env):
    res = ogs.author_and_sign_goal(
        ogs.GoalSpec(goal_id="x", title="t", description="d", target_files=()),
        path_override=sanction_env,
    )
    assert not res.ok
    assert res.reason == "unscoped_goal"


def test_author_refuses_duplicate_id(sanction_env):
    assert ogs.author_and_sign_goal(_spec(), path_override=sanction_env).ok
    res2 = ogs.author_and_sign_goal(_spec(), path_override=sanction_env)
    assert not res2.ok
    assert res2.reason == "duplicate_id"


def test_dry_run_writes_nothing(sanction_env):
    res = ogs.author_and_sign_goal(
        _spec(), path_override=sanction_env, dry_run=True,
    )
    assert res.ok and res.reason == "dry_run"
    assert not sanction_env.exists()


def test_withdraw_resigns_and_verifies(sanction_env):
    assert ogs.author_and_sign_goal(_spec("g-a"), path_override=sanction_env).ok
    assert ogs.author_and_sign_goal(_spec("g-b", files=("README.md",)),
                                    path_override=sanction_env).ok
    res = ogs.withdraw_goal("g-a", path_override=sanction_env)
    assert res.ok and res.reason == "withdrawn"
    text = sanction_env.read_text(encoding="utf-8")
    assert "g-a" not in text and "g-b" in text
    # unknown id is refused, not silently ignored
    assert not ogs.withdraw_goal("nope", path_override=sanction_env).ok


# ── injection envelope + the cage's real verifier ──────────────────────────


def test_scoped_envelope_verifies_through_the_cage(sanction_env):
    assert ogs.author_and_sign_goal(_spec(), path_override=sanction_env).ok
    env = ogs.build_scoped_envelope(
        goal_id="goal-1", description="Make the recap do X.",
        target_files=(_TARGET,),
    )
    assert env is not None
    assert env.source == "roadmap"
    claim = (env.evidence or {}).get("provenance")
    assert isinstance(claim, dict) and claim.get("goal_id") == "goal-1"

    from backend.core.ouroboros.governance.delegated_provenance import (
        verify_provenance_claim,
    )
    ok = verify_provenance_claim(claim, source="roadmap", file_strs=[_TARGET])
    assert ok.valid, ok.reason


def test_scoped_envelope_scope_laundering_is_refused(sanction_env):
    assert ogs.author_and_sign_goal(_spec(), path_override=sanction_env).ok
    env = ogs.build_scoped_envelope(
        goal_id="goal-1", description="d", target_files=(_TARGET,),
    )
    claim = (env.evidence or {}).get("provenance")
    from backend.core.ouroboros.governance.delegated_provenance import (
        verify_provenance_claim,
    )
    # A file the signed goal never declared must be refused — the claim is a
    # pointer, not a grant.
    bad = verify_provenance_claim(
        claim, source="roadmap",
        file_strs=["backend/core/ouroboros/governance/orchestrator.py"],
    )
    assert not bad.valid
    assert bad.reason == "target_out_of_scope"


def test_envelope_none_when_goal_absent(sanction_env):
    # No goal authored → no claim to present → fail closed (None), never an op
    # the cage will silently block.
    assert ogs.build_scoped_envelope(
        goal_id="never-signed", description="d", target_files=(_TARGET,),
    ) is None


def test_normalize_target_files_dedupes_and_relativizes():
    out = ogs.normalize_target_files(["a/b.py", "a/b.py", "./c.py", ""])
    assert out == ("a/b.py", "./c.py")


# ── cockpit /goal verb wiring (parser + roadmap read-back + shared dispatch) ─


class TestGoalVerbWiring:
    def test_flag_parser_multi_single_and_trailing_description(self):
        from backend.core.ouroboros.battle_test.harness import BattleTestHarness
        flags, desc = BattleTestHarness._parse_goal_flags(
            '--files a.py b.py --symbol foo bar --id my-goal '
            '--success "done when green" fix the thing'
        )
        assert flags["files"] == ["a.py", "b.py"]
        assert flags["symbols"] == ["foo", "bar"]
        assert flags["id"] == "my-goal"
        assert flags["success"] == "done when green"
        assert desc == "fix the thing"

    def test_flag_parser_leading_description(self):
        from backend.core.ouroboros.battle_test.harness import BattleTestHarness
        # The natural shape: description FIRST, flags after. Unambiguous — a
        # description after --files would be indistinguishable from more paths.
        flags, desc = BattleTestHarness._parse_goal_flags(
            "fix the recap bug --files a.py --symbol compose_recap"
        )
        assert desc == "fix the recap bug"
        assert flags["files"] == ["a.py"]
        assert flags["symbols"] == ["compose_recap"]

    def test_flag_parser_unknown_flag_is_skipped_not_guessed(self):
        from backend.core.ouroboros.battle_test.harness import BattleTestHarness
        flags, desc = BattleTestHarness._parse_goal_flags(
            "just do it --bogus --files a.py"
        )
        assert flags["files"] == ["a.py"]
        assert desc == "just do it"

    def test_slug_from_desc_is_stable_and_bounded(self):
        from backend.core.ouroboros.battle_test.harness import _slug_from_desc
        assert _slug_from_desc("Fix the Recap Timer!!").startswith(
            "op-fix-the-recap"
        )
        assert _slug_from_desc("") == "op-goal"
        assert len(_slug_from_desc("x " * 200)) <= 64

    def test_roadmap_readback_helpers(self, sanction_env):
        from backend.core.ouroboros.battle_test.harness import (
            _goal_desc_from_roadmap,
            _goal_scope_from_roadmap,
        )
        res = ogs.author_and_sign_goal(
            ogs.GoalSpec(
                goal_id="scope-goal", title="t",
                description="do the scoped thing",
                target_files=("backend/x.py", "backend/y.py"),
            ),
            path_override=sanction_env,
        )
        assert res.ok
        assert set(_goal_scope_from_roadmap("scope-goal")) == {
            "backend/x.py", "backend/y.py",
        }
        assert _goal_desc_from_roadmap("scope-goal") == "do the scoped thing"
        # a goal that was never signed reads back as empty scope, never a guess
        assert _goal_scope_from_roadmap("nope") == ()

    def test_injection_uses_the_shared_dispatch_seam(self):
        # DRY: the sanctioned-goal lane and the typed-chat lane BOTH reach
        # intake through _dispatch_intake_envelope — no private manual-goal path.
        import inspect

        from backend.core.ouroboros.battle_test.harness import BattleTestHarness
        src = inspect.getsource(BattleTestHarness._inject_sanctioned_goal)
        assert "self._dispatch_intake_envelope(" in src
        assert "build_scoped_envelope(" in src
