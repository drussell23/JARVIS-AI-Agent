"""shadow_enforce — the REVIEW-approve APPLY down-level, fail-closed on every branch.

Option 1 (routine-risk only): a graduated REVIEW subagent's CLEAN approve may
down-level a ROUTINE ``APPROVAL_REQUIRED`` to ``NOTIFY_APPLY`` (auto-apply WITH a
diff notice), authorizing VERIFY->APPLY without a human — but ONLY when no hard
gate demanded the tier (the caller proves that via ``hard_gate_present``). This
pins that the primitive can never reason around a hard gate, a non-clean verdict,
a BLOCKED tier, or an already-auto tier — the Option-1 safety invariant.
"""
from __future__ import annotations

import pytest

from backend.core.ouroboros.governance import shadow_enforce as SE
from backend.core.ouroboros.governance.risk_engine import RiskTier


@pytest.fixture(autouse=True)
def _authorize_on(monkeypatch):
    monkeypatch.setenv("JARVIS_SUBAGENT_APPLY_AUTHORIZE", "true")


def _agg(**kw):
    d = dict(
        aggregate=SE.AGG_APPROVE, files_reviewed=1, rejected=0,
        reservations=0, approved=1, failed=0, had_failure=False,
    )
    d.update(kw)
    return SE.ReviewAggregate(**d)


AR = RiskTier.APPROVAL_REQUIRED
NA = RiskTier.NOTIFY_APPLY
SA = RiskTier.SAFE_AUTO
BL = RiskTier.BLOCKED


def test_clean_approve_routine_downlevels_to_notify_apply():
    assert SE.authorize_apply_downlevel(AR, _agg(), hard_gate_present=False) == NA


def test_hard_gate_present_never_downlevels():
    assert SE.authorize_apply_downlevel(AR, _agg(), hard_gate_present=True) == AR


@pytest.mark.parametrize("bad", [
    dict(aggregate=SE.AGG_RESERVATIONS, reservations=1),
    dict(aggregate=SE.AGG_REJECT, rejected=1),
    dict(had_failure=True, failed=1),
    dict(files_reviewed=0),
])
def test_non_clean_verdict_never_downlevels(bad):
    assert SE.authorize_apply_downlevel(AR, _agg(**bad), hard_gate_present=False) == AR


def test_blocked_is_never_touched():
    assert SE.authorize_apply_downlevel(BL, _agg(), hard_gate_present=False) == BL


def test_already_auto_tiers_untouched():
    assert SE.authorize_apply_downlevel(SA, _agg(), hard_gate_present=False) == SA
    assert SE.authorize_apply_downlevel(NA, _agg(), hard_gate_present=False) == NA


def test_none_aggregate_never_downlevels():
    assert SE.authorize_apply_downlevel(AR, None, hard_gate_present=False) == AR


def test_flag_off_never_downlevels(monkeypatch):
    monkeypatch.setenv("JARVIS_SUBAGENT_APPLY_AUTHORIZE", "false")
    assert SE.authorize_apply_downlevel(AR, _agg(), hard_gate_present=False) == AR


def test_review_is_clean_approve_shapes():
    assert SE.review_is_clean_approve(_agg()) is True
    assert SE.review_is_clean_approve(_agg(reservations=1)) is False
    assert SE.review_is_clean_approve(_agg(aggregate=SE.AGG_REJECT, rejected=1)) is False
    assert SE.review_is_clean_approve(_agg(had_failure=True, failed=1)) is False
    assert SE.review_is_clean_approve(None) is False


def test_enforce_flags_graduated_on(monkeypatch):
    monkeypatch.delenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", raising=False)
    monkeypatch.delenv("JARVIS_PLAN_SUBAGENT_ENFORCE", raising=False)
    assert SE.review_enforce_enabled() is True
    assert SE.plan_enforce_enabled() is True


def test_escalate_still_only_raises():
    # The veto direction is unchanged — a floor can only raise, never lower.
    assert SE.escalate_risk_tier(SA, "approval_required") == AR
    assert SE.escalate_risk_tier(AR, "notify_apply") == AR  # never lowered
