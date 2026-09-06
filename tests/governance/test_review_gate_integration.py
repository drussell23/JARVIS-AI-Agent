"""_apply_review_gate — the single seam that folds the graduated REVIEW verdict
into the risk tier (Phase 1b), for both the extracted gate_runner and its twin.

Escalate on reject (safety veto); authorize a routine APPROVAL_REQUIRED down-level
on a clean approve — but NEVER for the two base-tier hard sources that no later
GATE gate re-clamps (the self-modification cage and the delegated-provenance
ceiling). Every other hard gate runs after this seam and re-clamps on its own.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.core.ouroboros.governance import shadow_enforce as SE
from backend.core.ouroboros.governance.orchestrator import (
    GovernedOrchestrator as GO,
)
from backend.core.ouroboros.governance.risk_engine import RiskTier

AR = RiskTier.APPROVAL_REQUIRED
NA = RiskTier.NOTIFY_APPLY
SA = RiskTier.SAFE_AUTO
BL = RiskTier.BLOCKED


@pytest.fixture(autouse=True)
def _graduated(monkeypatch):
    monkeypatch.setenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", "true")
    monkeypatch.setenv("JARVIS_SUBAGENT_APPLY_AUTHORIZE", "true")


def _agg(**kw):
    d = dict(
        aggregate=SE.AGG_APPROVE, files_reviewed=1, rejected=0,
        reservations=0, approved=1, failed=0, had_failure=False,
    )
    d.update(kw)
    return SE.ReviewAggregate(**d)


class _MockOrch:
    """Real methods, controlled review verdict."""

    def __init__(self, agg):
        self._agg = agg

    async def _run_review_shadow(self, ctx, best_candidate):
        return self._agg

    _review_downlevel_hard_blocked = GO._review_downlevel_hard_blocked
    _apply_review_gate = GO._apply_review_gate


def _ctx(files=("README.md",), ev=""):
    return SimpleNamespace(
        target_files=files, intake_evidence_json=ev, op_id="op-x",
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


_CAGE_FILE = "backend/core/ouroboros/governance/orchestrator.py"


def test_clean_approve_routine_downlevels():
    r = _run(_MockOrch(_agg())._apply_review_gate(_ctx(), None, AR))
    assert r == NA


def test_clean_approve_cage_is_blocked():
    r = _run(_MockOrch(_agg())._apply_review_gate(_ctx(files=(_CAGE_FILE,)), None, AR))
    assert r == AR


def test_clean_approve_provenance_ceiling_is_blocked():
    r = _run(_MockOrch(_agg())._apply_review_gate(
        _ctx(ev='{"provenance": {"goal_id": "g"}}'), None, AR,
    ))
    assert r == AR


def test_reject_escalates():
    r = _run(_MockOrch(_agg(aggregate=SE.AGG_REJECT, rejected=1))._apply_review_gate(
        _ctx(), None, SA,
    ))
    assert r == AR


def test_reservations_escalate_to_notify_apply():
    r = _run(_MockOrch(_agg(aggregate=SE.AGG_RESERVATIONS, reservations=1))._apply_review_gate(
        _ctx(), None, SA,
    ))
    assert r == NA


def test_blocked_never_touched_even_on_clean_approve():
    r = _run(_MockOrch(_agg())._apply_review_gate(_ctx(), None, BL))
    assert r == BL


def test_none_aggregate_is_fail_soft():
    r = _run(_MockOrch(None)._apply_review_gate(_ctx(), None, AR))
    assert r == AR


def test_enforce_off_leaves_reject_alone(monkeypatch):
    monkeypatch.setenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", "false")
    r = _run(_MockOrch(_agg(aggregate=SE.AGG_REJECT, rejected=1))._apply_review_gate(
        _ctx(), None, SA,
    ))
    assert r == SA  # no escalation when enforce is off


def test_hard_blocked_probe():
    m = _MockOrch(_agg())
    assert m._review_downlevel_hard_blocked(_ctx(files=("README.md",))) is False
    assert m._review_downlevel_hard_blocked(_ctx(files=(_CAGE_FILE,))) is True
    assert m._review_downlevel_hard_blocked(_ctx(ev='{"provenance":{}}')) is True
