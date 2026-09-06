"""Regression spine -- promote REVIEW + PLAN subagents SHADOW -> ENFORCE.

Pins the gated-default-OFF, fail-CLOSED-on-verdict, fail-SOFT-on-subsystem
contract for the two enforce branches:

REVIEW enforce (``JARVIS_REVIEW_SUBAGENT_ENFORCE``, default false):
  * OFF -> verdict logged, FSM proceeds (byte-identical shadow): the
    decision helper returns no floor and the tier is unchanged.
  * ON + blocking (REJECT) verdict -> risk-tier escalated to
    APPROVAL_REQUIRED via the EXISTING stricter-wins escalation.
  * ON + ambiguous (per-file review FAILED) -> escalate (fail-CLOSED).
  * subsystem error -> _run_review_shadow returns None -> no gating
    (fail-SOFT; the op is never blocked on a telemetry failure).

PLAN enforce (``JARVIS_PLAN_SUBAGENT_ENFORCE``, default false):
  * OFF -> flat plan authoritative (byte-identical shadow): should_fanout
    is False regardless of the DAG.
  * ON + multi-node parallelizable DAG -> should_fanout True -> drives the
    EXISTING is_fanout_eligible / enforce_evaluate_fanout path (force-armed).
  * ON + single-node / empty / malformed DAG -> should_fanout False ->
    legacy flat plan (fail-CLOSED).
  * force=True bypasses ONLY the parallel_dispatch flag guards; all real
    eligibility clamps remain authoritative.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from backend.core.ouroboros.governance.risk_engine import RiskTier
from backend.core.ouroboros.governance import shadow_enforce as se


# ---------------------------------------------------------------------------
# Flag readers default-OFF (byte-identical shadow).
# ---------------------------------------------------------------------------


def test_review_enforce_default_on(monkeypatch: Any) -> None:
    # GRADUATED 2026-09-06: the REVIEW verdict is authoritative by default.
    monkeypatch.delenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", raising=False)
    assert se.review_enforce_enabled() is True
    # Explicit off reverts to observer-only shadow.
    monkeypatch.setenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", "false")
    assert se.review_enforce_enabled() is False


def test_plan_enforce_default_on(monkeypatch: Any) -> None:
    # GRADUATED 2026-09-06: the PLAN DAG is authoritative by default.
    monkeypatch.delenv("JARVIS_PLAN_SUBAGENT_ENFORCE", raising=False)
    assert se.plan_enforce_enabled() is True
    monkeypatch.setenv("JARVIS_PLAN_SUBAGENT_ENFORCE", "false")
    assert se.plan_enforce_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE"])
def test_review_enforce_on_values(monkeypatch: Any, val: str) -> None:
    monkeypatch.setenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", val)
    assert se.review_enforce_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", ""])
def test_review_enforce_off_values(monkeypatch: Any, val: str) -> None:
    monkeypatch.setenv("JARVIS_REVIEW_SUBAGENT_ENFORCE", val)
    assert se.review_enforce_enabled() is False


# ---------------------------------------------------------------------------
# REVIEW enforce -- aggregate -> tier floor (fail-CLOSED on the verdict).
# ---------------------------------------------------------------------------


def test_reject_maps_to_approval_required() -> None:
    agg = se.ReviewAggregate(aggregate=se.AGG_REJECT, rejected=1, files_reviewed=1)
    assert se.aggregate_to_tier_floor(agg) == "approval_required"


def test_reservations_maps_to_notify_apply() -> None:
    agg = se.ReviewAggregate(
        aggregate=se.AGG_RESERVATIONS, reservations=1, files_reviewed=1
    )
    assert se.aggregate_to_tier_floor(agg) == "notify_apply"


def test_clean_approve_no_escalation() -> None:
    agg = se.ReviewAggregate(aggregate=se.AGG_APPROVE, approved=1, files_reviewed=1)
    assert se.aggregate_to_tier_floor(agg) is None


def test_no_files_no_escalation() -> None:
    agg = se.ReviewAggregate(aggregate=se.AGG_NO_FILES)
    assert se.aggregate_to_tier_floor(agg) is None


def test_ambiguous_failed_review_escalates_failclosed() -> None:
    # APPROVE aggregate but a per-file review FAILED -> ambiguous -> escalate.
    agg = se.ReviewAggregate(
        aggregate=se.AGG_APPROVE, approved=1, failed=1, files_reviewed=2,
        had_failure=True,
    )
    assert se.aggregate_to_tier_floor(agg) == "approval_required"


# ---------------------------------------------------------------------------
# REVIEW enforce -- escalate_risk_tier reuses stricter-wins (SemanticGuardian
# pattern): only RAISE the tier, never lower it.
# ---------------------------------------------------------------------------


def test_escalate_raises_lower_tier() -> None:
    out = se.escalate_risk_tier(RiskTier.SAFE_AUTO, "approval_required")
    assert out == RiskTier.APPROVAL_REQUIRED


def test_escalate_never_lowers() -> None:
    # Current is already APPROVAL_REQUIRED; a notify_apply floor must NOT
    # lower it.
    out = se.escalate_risk_tier(RiskTier.APPROVAL_REQUIRED, "notify_apply")
    assert out == RiskTier.APPROVAL_REQUIRED


def test_escalate_none_floor_unchanged() -> None:
    out = se.escalate_risk_tier(RiskTier.SAFE_AUTO, None)
    assert out == RiskTier.SAFE_AUTO


def test_escalate_unknown_floor_unchanged() -> None:
    out = se.escalate_risk_tier(RiskTier.SAFE_AUTO, "garbage_floor")
    assert out == RiskTier.SAFE_AUTO


# ---------------------------------------------------------------------------
# PLAN enforce -- DAG parallelizability probe on the 2d.1 payload.
# ---------------------------------------------------------------------------


def _multinode_payload() -> tuple:
    """A 2d.1 tuple-of-tuple payload with 2 dependency-free roots + cl=2."""
    units = (
        (("unit_id", "u1"), ("dependency_ids", ()), ("owned_paths", ("a.py",))),
        (("unit_id", "u2"), ("dependency_ids", ()), ("owned_paths", ("b.py",))),
    )
    return (
        ("schema_version", "2d.1"),
        ("graph_id", "deadbeef"),
        ("concurrency_limit", 2),
        ("units", units),
    )


def _single_node_payload() -> tuple:
    units = (
        (("unit_id", "u1"), ("dependency_ids", ()), ("owned_paths", ("a.py",))),
    )
    return (
        ("schema_version", "2d.1"),
        ("concurrency_limit", 1),
        ("units", units),
    )


def _serial_chain_payload() -> tuple:
    # cl=2 but only 1 root (u2 depends on u1) -> not parallelizable.
    units = (
        (("unit_id", "u1"), ("dependency_ids", ()), ("owned_paths", ("a.py",))),
        (("unit_id", "u2"), ("dependency_ids", ("u1",)), ("owned_paths", ("b.py",))),
    )
    return (
        ("schema_version", "2d.1"),
        ("concurrency_limit", 2),
        ("units", units),
    )


def test_multinode_dag_is_parallelizable() -> None:
    assert se.plan_dag_is_multinode(_multinode_payload()) is True


def test_single_node_dag_not_parallelizable() -> None:
    assert se.plan_dag_is_multinode(_single_node_payload()) is False


def test_serial_chain_not_parallelizable() -> None:
    assert se.plan_dag_is_multinode(_serial_chain_payload()) is False


def test_none_dag_not_parallelizable() -> None:
    assert se.plan_dag_is_multinode(None) is False


def test_malformed_dag_failclosed() -> None:
    assert se.plan_dag_is_multinode("not a graph") is False
    assert se.plan_dag_is_multinode(()) is False


# ---------------------------------------------------------------------------
# PLAN enforce -- should_fanout gates on flag AND DAG.
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    execution_graph: Any = None


def test_plan_should_fanout_off_byte_identical(monkeypatch: Any) -> None:
    # Explicit OFF (post-graduation the default is on) -> no fan-out even with a
    # perfectly parallelizable DAG.
    monkeypatch.setenv("JARVIS_PLAN_SUBAGENT_ENFORCE", "false")
    ctx = _Ctx(execution_graph=_multinode_payload())
    assert se.plan_enforce_should_fanout(ctx) is False


def test_plan_should_fanout_on_multinode(monkeypatch: Any) -> None:
    monkeypatch.setenv("JARVIS_PLAN_SUBAGENT_ENFORCE", "true")
    ctx = _Ctx(execution_graph=_multinode_payload())
    assert se.plan_enforce_should_fanout(ctx) is True


def test_plan_should_fanout_on_single_node_failclosed(monkeypatch: Any) -> None:
    monkeypatch.setenv("JARVIS_PLAN_SUBAGENT_ENFORCE", "true")
    ctx = _Ctx(execution_graph=_single_node_payload())
    assert se.plan_enforce_should_fanout(ctx) is False


def test_plan_should_fanout_on_no_dag_failclosed(monkeypatch: Any) -> None:
    monkeypatch.setenv("JARVIS_PLAN_SUBAGENT_ENFORCE", "true")
    ctx = _Ctx(execution_graph=None)
    assert se.plan_enforce_should_fanout(ctx) is False


# ---------------------------------------------------------------------------
# PLAN enforce -- force=True drives is_fanout_eligible past the master flag,
# but real clamps stay authoritative.
# ---------------------------------------------------------------------------


def test_force_bypasses_master_flag(monkeypatch: Any) -> None:
    from backend.core.ouroboros.governance import parallel_dispatch as pd

    # Master flag OFF.
    monkeypatch.delenv("JARVIS_WAVE3_PARALLEL_DISPATCH_ENABLED", raising=False)

    # Neutral posture so eligibility isn't clamped by posture confidence.
    def _posture_fn():
        return (None, None)

    # Without force -> MASTER_OFF -> not allowed.
    res_off = pd.is_fanout_eligible(
        op_id="op1", n_candidate_files=2, posture_fn=_posture_fn,
        emit_log=False, force=False,
    )
    assert res_off.allowed is False
    assert res_off.reason_code == pd.ReasonCode.MASTER_OFF

    # With force -> master flag bypassed; 2 files -> eligible (memory gate
    # permitting). We only assert it is NOT short-circuited by MASTER_OFF.
    res_on = pd.is_fanout_eligible(
        op_id="op1", n_candidate_files=2, posture_fn=_posture_fn,
        emit_log=False, force=True,
    )
    assert res_on.reason_code != pd.ReasonCode.MASTER_OFF


def test_force_does_not_relax_single_file_clamp(monkeypatch: Any) -> None:
    from backend.core.ouroboros.governance import parallel_dispatch as pd

    monkeypatch.delenv("JARVIS_WAVE3_PARALLEL_DISPATCH_ENABLED", raising=False)

    def _posture_fn():
        return (None, None)

    # force=True but only 1 candidate file -> still SINGLE_FILE_OP (clamp
    # is authoritative, force only bypasses the master FLAG).
    res = pd.is_fanout_eligible(
        op_id="op1", n_candidate_files=1, posture_fn=_posture_fn,
        emit_log=False, force=True,
    )
    assert res.allowed is False
    assert res.reason_code == pd.ReasonCode.SINGLE_FILE_OP


def test_enforce_evaluate_fanout_force_bypasses_flag_guards(monkeypatch: Any) -> None:
    """force=True lets enforce_evaluate_fanout proceed past Guards 1+2 even
    when the parallel_dispatch flags are off (the PLAN-enforce driver path).
    With an unrecognized generation it reaches Guard 3 (unrecognized_shape),
    proving Guards 1+2 were bypassed (not master_off / enforce_off)."""
    from backend.core.ouroboros.governance import parallel_dispatch as pd

    monkeypatch.delenv("JARVIS_WAVE3_PARALLEL_DISPATCH_ENABLED", raising=False)
    monkeypatch.delenv("JARVIS_WAVE3_PARALLEL_DISPATCH_ENFORCE", raising=False)

    class _Scheduler:
        async def submit(self, graph):  # pragma: no cover - not reached
            return True

    res = asyncio.run(
        pd.enforce_evaluate_fanout(
            op_id="op-force", generation=object(), scheduler=_Scheduler(),
            force=True,
        )
    )
    # Bypassed flag guards -> fell through to candidate extraction.
    assert res.skip_reason == "unrecognized_shape"


def test_enforce_evaluate_fanout_no_force_master_off(monkeypatch: Any) -> None:
    """Legacy caller (force=False) with flags off -> master_off (byte-identical)."""
    from backend.core.ouroboros.governance import parallel_dispatch as pd

    monkeypatch.delenv("JARVIS_WAVE3_PARALLEL_DISPATCH_ENABLED", raising=False)

    class _Scheduler:
        async def submit(self, graph):  # pragma: no cover - not reached
            return True

    res = asyncio.run(
        pd.enforce_evaluate_fanout(
            op_id="op-nf", generation=object(), scheduler=_Scheduler(),
        )
    )
    assert res.skip_reason == "master_off"
