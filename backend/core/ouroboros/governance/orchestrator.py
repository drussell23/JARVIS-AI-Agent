"""
Governed Pipeline Orchestrator
===============================

Central coordinator for the governed self-programming pipeline.  Ties
together the risk engine, candidate generator, approval provider, change
engine, and operation ledger into a single deterministic pipeline:

.. code-block:: text

    CLASSIFY -> ROUTE -> [CONTEXT_EXPANSION] -> [PLAN] -> GENERATE -> VALIDATE -> GATE -> [APPROVE] -> APPLY -> VERIFY -> COMPLETE

The orchestrator owns **no domain logic** -- only phase transitions and
error handling.  Every code path ends in a terminal phase (COMPLETE,
CANCELLED, EXPIRED, or POSTMORTEM).

Key guarantees:
- All unhandled exceptions are caught and transition to POSTMORTEM
- Retries are bounded by ``OrchestratorConfig`` limits
- BLOCKED operations are short-circuited at CLASSIFY
- APPROVAL_REQUIRED operations pause at APPROVE and wait for human decision
- Ledger entries are recorded at every significant lifecycle event
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import dataclasses
from dataclasses import asdict as _dc_asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from backend.core.ouroboros.governance.multi_repo.registry import RepoRegistry

from backend.core.ouroboros.governance.ascii_strict_gate import (
    AsciiStrictGate,
    build_retry_feedback as _ascii_gate_retry_feedback,
)
from backend.core.ouroboros.governance.target_existence_guard import (
    guard_enabled as _target_guard_enabled,
    universal_guard_enabled as _target_guard_universal_enabled,
    find_missing_targets as _find_missing_targets,
    build_retry_feedback as _target_missing_retry_feedback,
    missing_target_error_message as _target_missing_error_message,
    should_insulate_prompt as _should_insulate_prompt,
    TARGET_MISSING_PREFIX as _TARGET_MISSING_PREFIX,
)
from backend.core.ouroboros.governance.test_runner import BlockedPathError
from backend.core.ouroboros.governance.context_expander import ContextExpander
from backend.core.ouroboros.governance.approval_provider import (
    ApprovalResult,
    ApprovalStatus,
)
from backend.core.ouroboros.governance.change_engine import (
    ChangePhase,
    ChangeRequest,
    ChangeResult,
)
from backend.core.ouroboros.governance.mutation_critical_section import (
    maybe_mutation_section,
)
from backend.core.ouroboros.governance.ledger import LedgerEntry, OperationState
from backend.core.ouroboros.governance.learning_bridge import OperationOutcome
from backend.core.ouroboros.governance.cost_governor import (
    CostGovernor,
    CostGovernorConfig,
    OpCostCapExceeded,
)
from backend.core.ouroboros.governance.forward_progress import (
    ForwardProgressConfig,
    ForwardProgressDetector,
    candidate_content_hash,
)
from backend.core.ouroboros.governance.productivity_detector import (
    ProductivityDetector,
    ProductivityDetectorConfig,
    productivity_content_hash,
)
from backend.core.ouroboros.governance.op_context import (
    GenerationResult,
    OperationContext,
    OperationPhase,
    ValidationResult,
)
from backend.core.ouroboros.governance.risk_engine import (
    ChangeType,
    OperationProfile,
    RiskClassification,
    RiskTier,
)
from backend.core.ouroboros.governance.policy_engine import PolicyEngine, PolicyDecision
from backend.core.ouroboros.governance.saga.saga_apply_strategy import SagaApplyStrategy
from backend.core.ouroboros.governance.saga.cross_repo_verifier import CrossRepoVerifier
from backend.core.ouroboros.governance.saga.saga_types import RepoPatch, SagaTerminalState
# patch_benchmarker is intentionally NOT imported at module level — see
# `_run_benchmark` for the deferred import. This makes `patch_benchmarker`
# safely hot-reloadable via ModuleHotReloader: a module-level
# `from X import Y` would capture a stale class reference at orchestrator
# import time and never re-bind on reload.
from backend.core.ouroboros.integration import PerformanceRecord, TaskDifficulty

# B5 -- BLOCK -> decompose -> re-inject seam. Leaf modules (no orchestrator
# back-edge), so module-level import is circular-safe. Bound here as module
# attributes so the seam is fully unit-testable via monkeypatch.
from backend.core.ouroboros.governance.goal_decomposition_planner import (
    DecomposedPlan,
    chunking_enabled,
    decompose_for_block,
    estimate_subgoal_payload_chars,
    shed_block_goal_to_fit,
)
from backend.core.ouroboros.governance.adaptive_recursion_governor import (
    recursion_budget,
)
from backend.core.ouroboros.governance.recursion_dedup import (
    get_attempt_ledger,
    is_duplicate,
    subgoal_hash,
)
from backend.core.ouroboros.governance.multi_step_orchestrator import (
    advance_orchestration,
)
# T3 -- Convergence Watchdog wiring. Leaf modules (no orchestrator back-edge).
# Bound at module level so the seam is monkeypatch-testable (same discipline
# as decompose_for_block above).  estimate_subgoal_payload_chars already
# imported from goal_decomposition_planner above.
from backend.core.ouroboros.governance.convergence_watchdog import (
    get_reduction_tracker,
    watchdog_enabled,
    emit_sovereign_yield,
    max_self_heal_hops,
)
from backend.core.ouroboros.governance.epistemic_shedder import shed_to_fit

logger = logging.getLogger("Ouroboros.Orchestrator")


@dataclass(frozen=True)
class _BlockGoal:
    """Duck-typed RoadmapGoal-like view of a BLOCKed op for B5 decomposition.

    ``decompose_for_block`` reads ``goal_id`` / ``title`` / ``description`` /
    ``target_files`` (fail-soft). This adapter projects an OperationContext
    onto that shape without importing the roadmap goal type.
    """

    goal_id: str
    title: str
    description: str
    target_files: Tuple[str, ...]


# ──────────────────────────────────────────────────────────────────────────
# Slice 12Q — SessionRecorder terminal hook
# ──────────────────────────────────────────────────────────────────────────
#
# Bridge from the orchestrator's _record_ledger terminal site into the
# harness-owned SessionRecorder via the canonical process-singleton
# accessor. Closes the bt-2026-05-23-042249 gap where
# summary.json.operations[] was empty despite multiple terminal ops:
# the existing OP_COMPLETED autonomy-event subscription path subscribed
# to a callback (gls.report_outcome) that nothing in the runtime
# actually invokes for failed/exhausted ops, so terminal_reason_class
# attribution was unreachable.
#
# Mapping ledger.OperationState → recorder.status string:
#   APPLIED     → "completed"
#   ROLLED_BACK → "rolled_back"
#   FAILED      → "failed"
#   BLOCKED     → "failed"   (blocked = unrecoverable failure)
#
# Pulls terminal_reason_code from (ledger data dict) → (ctx attribute)
# → empty string fallback. Slice 12P's classifier in record_operation
# maps the code to the closed TerminalReasonClass taxonomy.
#
# NEVER raises into _record_ledger (the caller wraps in try/except;
# this helper additionally swallows internally as belt-and-suspenders).

_SLICE12Q_LEDGER_TO_STATUS: Dict[str, str] = {
    "applied": "completed",
    "rolled_back": "rolled_back",
    "failed": "failed",
    "blocked": "failed",
}


_ENV_SUBGOAL_WRITEBACK = "JARVIS_SUBGOAL_COMPLETION_WRITEBACK_ENABLED"


def _slice_a1_subgoal_completion_writeback(ctx: Any, state: Any) -> None:
    """§51.11.34-ROADMAP A1 — close the sub-goal completion feedback loop.

    The multi_step orchestrator emits sub-goal envelopes (stamping
    ``sub_goal_id`` + ``parent_goal_id`` into the envelope evidence) and writes
    a ``PROPOSED`` row to the canonical goal_decomposition completion ledger at
    EMIT time — but historically NOTHING wrote the terminal
    ``COMPLETED``/``FAILED`` transition back. ``done_count`` (which counts
    ``completed`` rows) was therefore structurally pinned at 0: a roadmap
    sub-goal could dispatch and succeed any number of times and the roadmap
    would never advance.

    This writeback fires from the orchestrator terminal hook — the same
    fail-soft, recorder-independent seam the Slice-134 episodic synapse uses.
    When the terminal op carries roadmap sub-goal provenance via
    ``ctx.intake_evidence_json``, the terminal state is mapped to a
    CompletionStatus (``applied`` -> COMPLETED; any other terminal -> FAILED)
    and appended to the completion ledger, so the multi_step orchestrator's
    ``done_count`` advances and the roadmap can progress.

    Gated ``JARVIS_SUBGOAL_COMPLETION_WRITEBACK_ENABLED`` (default TRUE — this
    closes a structural gap; OFF is byte-identical to the legacy severed loop).
    NEVER raises.
    """
    try:
        raw = os.environ.get(_ENV_SUBGOAL_WRITEBACK, "true").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return
        # Cheap substring pre-check avoids a json.loads on the vast majority of
        # ops (sensor signals) that carry no sub_goal provenance.
        evidence_json = getattr(ctx, "intake_evidence_json", "") or ""
        if not evidence_json or "sub_goal_id" not in evidence_json:
            return
        try:
            evidence = json.loads(evidence_json)
        except Exception:  # noqa: BLE001
            return
        if not isinstance(evidence, dict):
            return
        sub_goal_id = str(evidence.get("sub_goal_id") or "").strip()
        parent_goal_id = str(evidence.get("parent_goal_id") or "").strip()
        if not sub_goal_id or not parent_goal_id:
            return
        state_value = getattr(state, "value", str(state)) or ""
        from backend.core.ouroboros.governance.goal_decomposition_planner import (  # noqa: E501
            CompletionStatus,
            mark_sub_goal_status,
        )
        status = (
            CompletionStatus.COMPLETED
            if state_value == "applied"
            else CompletionStatus.FAILED
        )
        mark_sub_goal_status(
            sub_goal_id=sub_goal_id,
            parent_goal_id=parent_goal_id,
            status=status,
            note=(
                "terminal:" + str(state_value) + " via orchestrator op "
                + str(getattr(ctx, "op_id", ""))
            )[:512],
        )
    except Exception:  # noqa: BLE001 — writeback never perturbs the FSM
        return


def _slice230_record_exploration_drift(op_id: Any, model_id: Any) -> None:
    """Slice 230 — feed an Iron-Gate exploration rejection back into model
    rotation. Records ``DriftType.EXPLORATION_INSUFFICIENT`` for
    (op_id, model_id) in the Slice-20C drift tracker, so the GENERATE_RETRY
    sentinel walk skips the model that just emitted a no-tool patch and
    rotates to the next ranked candidate (the agentic elites, per Slices
    228/229). Without this wire, a weak model that "succeeds" at transport
    level keeps winning the walk and the op dies 0/1 forever. Loud by
    operator preference. NEVER raises — the gate path must not be perturbed."""
    try:
        op = str(op_id or "").strip()
        model = str(model_id or "").strip()
        if not op or not model:
            return
        from backend.core.ouroboros.governance.schema_drift_tracker import (
            DriftType,
            get_default_tracker,
        )
        get_default_tracker().record(
            op_id=op, model_id=model,
            drift_type=DriftType.EXPLORATION_INSUFFICIENT,
        )
        logger.warning(
            "[Orchestrator] ⚡ GATE→ROTATION: model=%s emitted a no-tool patch "
            "(exploration_insufficient) — drift-marked for op=%s; the retry "
            "walk will rotate to the next ranked (agentic) candidate",
            model, op[:16],
        )
    except Exception:  # noqa: BLE001 — feedback never perturbs the gate
        return


def _slice12q_record_terminal(
    ctx: Any, state: Any, data: Dict[str, Any],
) -> None:
    """Record one terminal op into the active SessionRecorder.

    Composition only — reads the canonical singleton accessor,
    extracts the smallest correct payload from ctx + state + data,
    delegates idempotency + classification to SessionRecorder.
    NEVER raises."""
    # Slice 134 — FSM synapse (write-side). Fire-and-forget episodic record of
    # this terminal transition (START -> <terminal state>) with a context
    # snapshot — captures COMPLETE / BLOCKED / failed (incl. IRON_GATE +
    # REFUSED_SAFETY via reason_code) + the route the op took. Self-contained +
    # recorder-INDEPENDENT (episodic memory must work in production without a
    # battle-test SessionRecorder). Gated JARVIS_EPISODIC_CORE_ENABLED,
    # non-blocking (scheduled, never awaited), fail-soft.
    try:
        from backend.core.ouroboros.governance.episodic_core import (
            note_transition_nowait as _note_episode,
        )
        _state_value = getattr(state, "value", str(state)) or ""
        _reason = (
            getattr(ctx, "terminal_reason_code", "")
            or (isinstance(data, dict) and (data.get("reason", "") or data.get("error", "")))
            or ""
        )
        _op_id = str(getattr(ctx, "op_id", "") or "")
        if _op_id:
            _route = (
                getattr(ctx, "provider_route", "")
                or (isinstance(data, dict) and data.get("route", "")) or ""
            )
            _note_episode(
                op_id=_op_id, phase_from="START", phase_to=str(_state_value),
                summary=("op terminal " + str(_state_value)
                         + (f" [{_reason}]" if _reason else "")),
                context={
                    "terminal_reason_code": str(_reason),
                    "route": str(_route),
                },
            )
    except Exception:  # noqa: BLE001 — synapse never perturbs the FSM
        pass
    # §51.11.34-ROADMAP A1 — sub-goal completion writeback (the severed feedback
    # wire). Recorder-independent + fail-soft, exactly like the episodic synapse
    # above. Closes the roadmap progress loop: terminal op -> completion ledger.
    _slice_a1_subgoal_completion_writeback(ctx, state)
    try:
        from backend.core.ouroboros.battle_test.session_recorder import (
            get_active_recorder,
        )
    except Exception:  # noqa: BLE001
        return
    try:
        recorder = get_active_recorder()
    except Exception:  # noqa: BLE001
        return
    if recorder is None:
        return
    try:
        state_value = getattr(state, "value", str(state)) or ""
        status = _SLICE12Q_LEDGER_TO_STATUS.get(state_value, "failed")
        # Terminal reason — prefer the explicit ctx attribute (set by
        # CircuitBreaker / Iron Gate / cooldown paths), then the
        # ledger data dict's "reason"/"error" field, then empty.
        reason_code = (
            getattr(ctx, "terminal_reason_code", "") or
            (isinstance(data, dict) and (
                data.get("reason", "") or data.get("error", "")
            )) or ""
        )
        # Best-effort metadata extraction; all defaults are recorder-safe.
        op_id = str(getattr(ctx, "op_id", "") or "")
        if not op_id:
            return  # nothing useful to record without op_id
        sensor = ""
        if isinstance(data, dict):
            sensor = data.get("source", "") or data.get("sensor", "")
        sensor = sensor or getattr(ctx, "intake_source", "") or "unknown"
        provider = (
            getattr(ctx, "provider_route", "") or
            (isinstance(data, dict) and data.get("route", "")) or
            ""
        )
        cost_usd = 0.0
        if isinstance(data, dict):
            try:
                cost_usd = float(data.get("cost_usd", 0.0) or 0.0)
            except (TypeError, ValueError):
                cost_usd = 0.0
        elapsed_s = 0.0
        if isinstance(data, dict):
            try:
                elapsed_s = float(data.get("duration_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                elapsed_s = 0.0
        recorder.record_operation(
            op_id=op_id,
            status=status,
            sensor=str(sensor),
            technique="orchestrator_terminal",
            composite_score=0.0,
            elapsed_s=elapsed_s,
            provider=str(provider),
            cost_usd=cost_usd,
            terminal_reason_code=str(reason_code),
        )
    except Exception:  # noqa: BLE001
        # Belt-and-suspenders — caller's try/except also catches.
        logger.debug(
            "[Orchestrator] Slice 12Q _slice12q_record_terminal raised "
            "(swallowed)",
            exc_info=True,
        )

# Module-level buffer for LearningConsolidator periodic consolidation.
# Outcomes accumulate here; once the threshold is reached, consolidate()
# is called to generate new domain-level rules.
_CONSOLIDATION_BUFFER: list = []
_CONSOLIDATION_THRESHOLD: int = 10

# Grace period added to route-based _gen_timeout for the outer wait_for
# Iron Gate.  The generator may internally refresh the fallback budget to
# _FALLBACK_MIN_GUARANTEED_S (90s) even when the parent deadline is nearly
# exhausted — 5s was too tight and caused 129s Claude streams to be cut
# by the 125s outer gate (bt-2026-04-12-061609).  15s accommodates Tier 0
# overhead + asyncio cancellation propagation delay on streaming responses.
_OUTER_GATE_GRACE_S = float(os.environ.get("JARVIS_OUTER_GATE_GRACE_S", "15"))
_TRUTHY = frozenset({"1", "true", "yes", "on"})


# ──────────────────────────────────────────────────────────────────────────
# Anti-Venom Lock A — fail-CLOSED guardian sentinel
# ──────────────────────────────────────────────────────────────────────────
#
# When the SemanticGuardian invocation itself raises (import error, detector
# regression, OOM mid-batch …) the historical behavior was to fail OPEN:
# swallow the exception, leave ``_guardian_findings`` empty, and let the
# classifier's tier stand — so a SAFE_AUTO candidate auto-applied with the
# semantic safety net silently down. This sentinel is the fail-CLOSED
# replacement: on guardian crash we inject ONE hard finding and force
# APPROVAL_REQUIRED so the op parks at GATE for a human instead of
# auto-applying. ``severity="hard"`` makes ``recommend_tier_floor`` (and any
# other findings consumer) treat a guardian crash exactly like a fired hard
# pattern. Defined at module scope so it is a single, grep-able, immutable
# (frozen Detection) source of truth — never reconstructed per-op.
from backend.core.ouroboros.governance.semantic_guardian import (  # noqa: E402
    Detection as _GuardianDetection,
)

_SENTINEL_GUARDIAN_CRASH = _GuardianDetection(
    pattern="guardian_crashed",
    severity="hard",
    message="guardian crashed — fail-closed",
    file_path="",
    lines=(),
    snippet="",
)


# Slice 6 Task 5 — attribution scope gate.
#
# Run #16 root cause: a TestFailure op scoped to the failing TEST file
# blindly mutated that test file (Task 4 now stamps such
# unresolved-attribution ops with evidence ``attribution.status=
# "unresolved"`` in ``ctx.intake_evidence_json``). When the op's
# attribution is unresolved AND the candidate mutates ONLY test loci,
# auto-applying is exactly that blind class. This helper wires the pure,
# fully-unit-tested Task 2 predicate ``unattributed_test_scope_violation``
# into the post-VALIDATE risk decision and escalates to APPROVAL_REQUIRED
# — a HUMAN GATE, not a reject: the test itself may be the legitimate fix
# target, so that judgment needs eyes, not a retry loop.
#
# It mirrors the SemanticGuardian hard-finding escalation at the same
# site: stricter-wins (``risk_tier.value < APPROVAL_REQUIRED.value``) so
# it never downgrades an already-strict tier (APPROVAL_REQUIRED / BLOCKED)
# and composes with — never bypasses — the existing risk machinery.
# Fail-SOFT: any exception (import error, malformed evidence, predicate
# raise) yields no escalation — the gate is protective, never fatal.
def _attribution_scope_risk_floor(
    ctx: Any,
    candidate_file_paths: Sequence[str],
    risk_tier: RiskTier,
    *,
    repo_root: str = "",
) -> Tuple[RiskTier, Optional[str]]:
    """Return ``(possibly-escalated risk_tier, violation_message|None)``.

    ``violation_message`` is non-None whenever the Run-16 blind class is
    detected (unresolved attribution + test-only candidate), even if the
    tier was already strict enough that no escalation was applied — the
    caller logs it for operator visibility either way.

    ``repo_root`` (I2): passed through to the predicate so ABSOLUTE
    candidate paths (which the model may emit) are normalized to
    repo-relative before test-locus classification — an absolute
    ``…/tests/conftest.py`` must not slip the gate.
    """
    try:
        from backend.core.ouroboros.governance.intent.test_source_attribution import (  # noqa: E501
            unattributed_test_scope_violation,
        )
        violation = unattributed_test_scope_violation(
            getattr(ctx, "intake_evidence_json", "") or "",
            candidate_file_paths,
            repo_root=repo_root,
        )
    except Exception:  # noqa: BLE001 — gate is protective, never fatal
        return risk_tier, None
    if not violation:
        return risk_tier, None
    # Escalate exactly as a SemanticGuardian HARD finding does at this
    # site: floor at APPROVAL_REQUIRED, stricter-wins (never a downgrade).
    if risk_tier.value < RiskTier.APPROVAL_REQUIRED.value:
        risk_tier = RiskTier.APPROVAL_REQUIRED
    return risk_tier, violation


def _value_ceiling_risk_floor(
    ctx: Any,
    risk_tier: RiskTier,
) -> Tuple[RiskTier, Optional[str]]:
    """Slice 15 T4 mandate 4 — above the adaptive ceiling, HALT for the
    human exception-handler. An oracle-band op (verified failing-test
    attribution) spanning more target files than
    ``JARVIS_VALUE_CEILING_FILES`` is real, high-value, high-blast work —
    exactly the exception a human reviews. Escalates to APPROVAL_REQUIRED;
    NEVER demotes (the Orange flow is preserved absolutely). Sibling of
    :func:`_attribution_scope_risk_floor`, wired at the SAME two GATE call
    sites (gate_runner is the shipping path). Fail-soft: any internal
    fault returns the tier unchanged."""
    try:
        from backend.core.ouroboros.governance.signal_value import (
            score_ctx,
            signal_value_routing_enabled,
            value_ceiling_breached,
        )
        if not signal_value_routing_enabled():
            return risk_tier, None
        _n = len(getattr(ctx, "target_files", ()) or ())
        if value_ceiling_breached(score_ctx(ctx), _n):
            if risk_tier not in (
                RiskTier.APPROVAL_REQUIRED, RiskTier.BLOCKED,
            ):
                return (
                    RiskTier.APPROVAL_REQUIRED,
                    "value_ceiling: oracle-band op spans %d target files "
                    "(> ceiling) — queued for the human exception-handler"
                    % _n,
                )
    except Exception:  # noqa: BLE001 — fail-soft, never fatal at GATE
        pass
    return risk_tier, None


# Slice 8 companion to _attribution_scope_risk_floor above: Slice 7's
# subset waiver correctly lets a test-only candidate pass the coverage
# gate when attribution is RESOLVED (the test may genuinely BE the fix
# target), but that opened a residual lane the Slice-7 final review
# flagged — a green-tier assertion-weakening test edit now auto-applies
# and VERIFY passes by construction (the test agrees with the broken
# code it was meant to catch). This floors that lane at NOTIFY_APPLY
# (operator-visible diff + delay), never blocking (the lane is
# legitimate) and never downgrading an already-stricter tier.
def _attribution_test_only_notify_floor(
    ctx: Any,
    candidate_file_paths: Sequence[str],
    risk_tier: RiskTier,
    *,
    repo_root: str = "",
) -> Tuple[RiskTier, Optional[str]]:
    """Slice 8 companion to :func:`_attribution_scope_risk_floor`:
    RESOLVED attribution + test-only candidate → floor at NOTIFY_APPLY
    (operator-visible diff+delay; legitimate lane, so a notify, not an
    approval). Stricter-wins; fail-SOFT (any fault → no escalation)."""
    try:
        from backend.core.ouroboros.governance.intent.test_source_attribution import (  # noqa: E501
            resolved_test_only_scope,
        )
        advisory = resolved_test_only_scope(
            getattr(ctx, "intake_evidence_json", "") or "",
            candidate_file_paths,
            repo_root=repo_root,
        )
    except Exception:  # noqa: BLE001 — floor is protective, never fatal
        return risk_tier, None
    if not advisory:
        return risk_tier, None
    if risk_tier.value < RiskTier.NOTIFY_APPLY.value:
        risk_tier = RiskTier.NOTIFY_APPLY
    return risk_tier, advisory


# Targeted Locality Bounding / Epistemic Humility (2026-07-21) —
# companion to the two attribution floors above, wired at the SAME two
# GATE call sites (gate_runner is the shipping path). When the
# OperationAdvisor's blast-radius scan resolved to provenance=unknown
# at CLASSIFY (global budget exhausted on a cold cache AND the bounded
# localized fallback could not resolve — bt-2026-07-21-205755), the
# advisor recorded an escalation in the advisor_locality epistemic
# ledger instead of fabricating blast=50. This floor converts that
# recorded uncertainty into a NOTIFY_APPLY minimum: the op stays
# appealable (operator-visible diff + delay), never silently green,
# never blocked on data nobody collected. Stricter-wins; never
# downgrades; fail-soft.
def _advisor_epistemic_notify_floor(
    ctx: Any,
    risk_tier: RiskTier,
) -> Tuple[RiskTier, Optional[str]]:
    """Floor an epistemically-uncertain op at NOTIFY_APPLY.

    Reads the advisor_locality epistemic ledger (non-destructive so
    both GATE twins + GENERATE retries observe the same state). Gated
    by ``JARVIS_ADVISOR_EPISTEMIC_NOTIFY_ENABLED`` (default TRUE).
    Returns ``(possibly-escalated tier, note|None)`` — the note is
    non-None whenever the ledger carries an escalation, even if the
    tier was already strict enough (caller logs for visibility).
    """
    try:
        from backend.core.ouroboros.governance.advisor_locality import (
            ESCALATION_NOTIFY_APPLY,
            epistemic_notify_enabled,
            peek_blast_epistemics,
        )
        if not epistemic_notify_enabled():
            return risk_tier, None
        rec = peek_blast_epistemics(getattr(ctx, "op_id", "") or "")
    except Exception:  # noqa: BLE001 — floor is protective, never fatal
        return risk_tier, None
    if not rec or rec.get("escalation") != ESCALATION_NOTIFY_APPLY:
        return risk_tier, None
    if risk_tier.value < RiskTier.NOTIFY_APPLY.value:
        return (
            RiskTier.NOTIFY_APPLY,
            "advisor_epistemic: blast radius UNKNOWN at CLASSIFY "
            "(cold-cache scans unresolved) — floored at NOTIFY_APPLY "
            "for operator visibility",
        )
    return (
        risk_tier,
        "advisor_epistemic: blast radius UNKNOWN at CLASSIFY "
        "(tier already >= NOTIFY_APPLY; no change)",
    )


# Contiguous, grep-discoverable terminal reason marker (single
# source — referenced by the breaker's ctx.advance + ledger payload;
# never split across string-concat lines so log/forensic greps and
# the spine can match it verbatim).
_FAILFAST_CIRCUIT_OPEN_REASON = "all_providers_exhausted_circuit_open"


# ──────────────────────────────────────────────────────────────────────────
# Sovereign Epistemic Context Matrix — non-retryable terminal reason codes
# ──────────────────────────────────────────────────────────────────────────
#
# Terminal reason codes that MUST NOT be re-driven through GENERATE_RETRY.
# Historically the orchestrator decided retryability imperatively at each
# terminal site (infra-class escalates immediately, exhaustion parks the op,
# advisor/plan-rejection short-circuits, etc.) — there was no single
# grep-discoverable set. This frozenset gathers the codes whose terminal
# state is genuinely unrecoverable within the same op so a classifier exists
# (and is unit-testable). It is *descriptive* of the already-non-retryable
# sites plus the new ``deadlock_override_failed`` terminal (raised when LR3's
# one-shot governance-deadlock breaker fails mid-Venom): retrying a wedged
# governance deadlock just re-wedges, so it terminates non-retryably. This is
# the AUTHORITATIVE source of truth for the non-retry decision — the
# ``except GovernanceDeadlockError`` terminal path consults it via
# ``_is_nonretryable_terminal`` rather than deciding implicitly.
_NONRETRYABLE_TERMINAL_REASONS: "frozenset[str]" = frozenset({
    "deadlock_override_failed",      # LR3 governance deadlock breaker failed
    "advisor_blocked",               # OperationAdvisor pre-gen veto
    "plan_rejected",                 # human/plan-review rejected the plan
    "plan_required_unavailable",     # plan mandatory but generator unavailable
    "plan_review_unavailable",       # plan-review mandatory but unavailable
    "plan_approval_expired",         # approval window elapsed
    "swebp_repo_root_rejected",      # repo-root boundary rejection
    "validation_infra_failure",      # infra-class VALIDATE failure (non-retry)
    "validation_budget_exhausted",   # VALIDATE budget gone
    "op_cost_cap_exceeded",          # per-op cost cap blown
    "user_cancelled",                # cooperative cancellation
    "sovereign_route_sealed",        # Absolute Route Sealing: committed J-Prime
                                     # dispatch failed; DW cascade forbidden -> halt
    _FAILFAST_CIRCUIT_OPEN_REASON,   # fail-fast exhaustion breaker open
})


def _is_nonretryable_terminal(reason_code: str) -> bool:
    """Return True iff ``reason_code`` is a non-retryable terminal reason.

    This is the AUTHORITATIVE registry of terminal reason codes that must never
    be retried. It is consulted by the ``GovernanceDeadlockError`` terminal path
    (``except GovernanceDeadlockError`` in ``run``) as the single source of
    truth for the non-retry decision, and is available to future terminal sites
    that need the same classification.

    Pure, side-effect-free classifier over ``_NONRETRYABLE_TERMINAL_REASONS``.
    Always returns a ``bool`` and never raises (coerces non-str input to str).
    Matches an exact code OR a ``prefix:detail`` shape by its colon-prefix head
    (e.g. ``sovereign_route_sealed:gcp-jprime:LocalLatencyLockup`` -> matches
    ``sovereign_route_sealed``). Existing colon-free codes are unaffected.
    """
    try:
        code = str(reason_code)
        if code in _NONRETRYABLE_TERMINAL_REASONS:
            return True
        return code.split(":", 1)[0] in _NONRETRYABLE_TERMINAL_REASONS
    except Exception:  # noqa: BLE001 — classifier must never raise
        return False


def _failfast_cb_enabled() -> bool:
    """Fail-Fast Exhaustion Circuit Breaker master switch.

    ``JARVIS_FAILFAST_EXHAUSTION_CB_ENABLED`` — §33.1 **default
    TRUE**. When OFF the legacy retry/park behaviour is
    byte-identical (an op that exhausts all providers keeps cycling
    until the 1800s eval window / op deadline). When ON, an op that
    raises ``all_providers_exhausted`` for N consecutive attempts is
    flipped to a terminal ``failed`` state immediately — so the
    ``operation_terminal`` SSE fires and any awaiting subscriber
    (B.2.2 evaluate_problem) wakes in seconds, not 1800s. Read at
    call time."""
    return os.environ.get(
        "JARVIS_FAILFAST_EXHAUSTION_CB_ENABLED", "true",
    ).strip().lower() not in {"0", "false", "no", "off"}


def _failfast_cb_threshold() -> int:
    """``JARVIS_FAILFAST_EXHAUSTION_MAX_CONSECUTIVE`` (default ``2``
    — both generation attempts genuinely exhausted; prevents a
    twitchy false-trip on a single micro-blip while enforcing
    thermodynamic containment). Clamped ``>= 1``. Read at call
    time; never raises."""
    try:
        v = int(os.environ.get(
            "JARVIS_FAILFAST_EXHAUSTION_MAX_CONSECUTIVE", "2",
        ).strip())
        return v if v >= 1 else 1
    except (ValueError, TypeError):
        return 2


def _candidate_tree_enabled() -> bool:
    """Slice 9 (default ON): VALIDATE materializes a working-tree-faithful
    candidate tree (RepairSandbox + dirty overlay + candidate files applied)
    and anchors the LanguageRouter AT the tree — so validation exercises
    the CANDIDATE, not the still-broken real tree (Run #19 root cause).
    OFF restores the legacy side-sandbox path byte-identically."""
    return os.environ.get(
        "JARVIS_VALIDATE_CANDIDATE_TREE_ENABLED", "true",
    ).strip().lower() not in ("0", "false", "no", "off")


def _validate_tree_min_budget_s() -> float:
    """Slice 9 final review (Important): minimum remaining pipeline budget
    (seconds) required before VALIDATE will materialize the candidate-tree
    (RepairSandbox + working-tree overlay — measured ~14s setup on this
    repo). Below this floor, skip straight to the legacy side-sandbox path
    instead of spending most of the remaining budget on setup alone. Env:
    ``JARVIS_VALIDATE_TREE_MIN_BUDGET_S``, default ``30.0``. Read at call
    time; never raises."""
    try:
        v = float(os.environ.get(
            "JARVIS_VALIDATE_TREE_MIN_BUDGET_S", "30.0",
        ).strip())
        return v if v >= 0.0 else 30.0
    except (ValueError, TypeError):
        return 30.0


def _map_tree_run_exception(exc: Exception, t0: float) -> "ValidationResult":
    """Slice 9 review (Important): map a candidate-tree ``_tree_runner.run()``
    exception to the SAME ``ValidationResult`` shape the legacy side-sandbox
    path's ``except BlockedPathError`` / ``except Exception`` handlers
    produce (see ``_run_validation_core`` ~L12996-13016).

    A ``BlockedPathError`` out of the tree run is a genuine Slice-8 security
    rejection — it must classify ``failure_class="security"`` and RETURN
    directly, never fall through to a second, differently-anchored legacy
    VALIDATE run that may not reproduce it. Any other exception mirrors the
    legacy generic handler (``failure_class="infra"``). Callers use this
    ONLY for exceptions raised by ``_tree_runner.run()`` itself — faults
    during candidate-tree MATERIALIZATION (RepairSandbox entry,
    ``apply_full_content``) stay fail-soft and fall back to the legacy path
    unchanged (never routed through this helper)."""
    if isinstance(exc, BlockedPathError):
        return ValidationResult(
            passed=False,
            best_candidate=None,
            validation_duration_s=time.monotonic() - t0,
            error=str(exc),
            failure_class="security",
            short_summary=f"BlockedPathError: {str(exc)[:280]}",
            adapter_names_run=(),
        )
    return ValidationResult(
        passed=False,
        best_candidate=None,
        validation_duration_s=time.monotonic() - t0,
        error=str(exc),
        failure_class="infra",
        short_summary=f"runner exception: {str(exc)[:200]}",
        adapter_names_run=(),
    )


def _phase_runner_complete_extracted() -> bool:
    """Slice 1 of Wave 2 (5) — COMPLETE phase extraction gate.

    Reads ``JARVIS_PHASE_RUNNER_COMPLETE_EXTRACTED``, **default
    ``true`` as of 2026-04-22 graduation (3 clean soak sessions
    bt-2026-04-22-183425 / -185203 / -190730 + Slice 1 parity
    22/22 byte-identical vs inline). Explicit ``=false`` remains a
    runtime kill switch that reverts to the inline block.**

    When ``true``, ``_run_pipeline`` delegates the COMPLETE block at
    line ~7073 to
    :class:`backend.core.ouroboros.governance.phase_runners.complete_runner.COMPLETERunner`.
    When ``false``, the inline block runs unchanged. Parity tests
    (tests/governance/phase_runner/test_complete_runner_parity.py)
    pin byte-identical observable output across both paths.

    Graduation ledger: ``memory/project_wave2_graduation_matrix.md``.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_COMPLETE_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_route_extracted() -> bool:
    """Slice 3 of Wave 2 (5) — ROUTE phase extraction gate.

    **Default ``true`` as of 2026-04-22 atomic #3 graduation** (3 clean
    soak sessions bt-2026-04-22-214630 / -220234 / -222322, each with
    zero runner-attributed frames + zero shutdown race + 40 total
    ROUTE+CTX+PLAN delegation markers). Flipped together with
    ``_phase_runner_context_expansion_extracted`` and
    ``_phase_runner_plan_extracted`` since the combined gate
    ``_phase_runner_slice3_fully_extracted`` requires all three.
    Explicit ``=false`` on this helper alone remains a per-phase
    kill switch — operator can sever ROUTE without affecting CTX/PLAN.

    Graduation ledger: ``memory/project_wave2_graduation_matrix.md``.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_ROUTE_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_context_expansion_extracted() -> bool:
    """Slice 3 of Wave 2 (5) — CONTEXT_EXPANSION phase extraction gate.

    **Default ``true`` as of 2026-04-22** (atomic #3 graduation with
    ROUTE + PLAN; see ``_phase_runner_route_extracted`` docstring for
    soak evidence). Explicit ``=false`` kill switch remains.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_CONTEXT_EXPANSION_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_plan_extracted() -> bool:
    """Slice 3 of Wave 2 (5) — PLAN phase extraction gate.

    **Default ``true`` as of 2026-04-22** (atomic #3 graduation with
    ROUTE + CTX; see ``_phase_runner_route_extracted`` docstring).
    Explicit ``=false`` kill switch remains.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_PLAN_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_generate_extracted() -> bool:
    """Slice 5a/5b of Wave 2 (5) — GENERATE phase extraction gate.

    **Default ``true`` as of 2026-04-23 graduation** (3 clean sessions
    under post-Ticket-A1/B/C guards: bt-2026-04-23-062014 (14 markers)
    + bt-2026-04-23-203517 S2′ (12 markers, session_outcome=complete)
    + bt-2026-04-23-210943 S3 (13 markers, session_outcome=complete);
    all three idle_timeout stop, 0 runner-attributed frames, 0 JARVIS
    shutdown race, 0 POSTMORTEMs, 39 total [PhaseRunnerDelegate] GENERATE
    markers). Iron Gate live lines NOT observed across the cadence
    because Anthropic transport weather (canonical signature:
    anthropic/_base_client.py:1637 request → httpx/_transports/default.py:101
    map_httpcore_exceptions) prevented candidates from forming; §6
    depth is attested by the Slice 5a+5b parity oracle (36/36 tests
    green on HEAD 68954cc62d — 12 FSM-edge parity + 24 Iron Gate
    suite across Exploration-first / Exploration Ledger / ASCII strict
    / Dependency integrity / Multi-file coverage / Retry feedback).
    reachability_source=partial_live+parity under the path (B)
    contract documented in project_wave2_graduation_matrix.md.
    Explicit ``=false`` remains a runtime kill switch reverting to the
    ~1,611-line inline GENERATE block. A post-flip confirmation session
    is required per operator directive to capture Iron Gate telemetry
    if/when the transport weather clears — failure to observe Iron
    Gate lines post-flip does NOT auto-rollback unless runner-attributed
    regression or parity breaks.

    When ``true``, delegates the ~1,611-line GENERATE block (prelude +
    retry loop + CandidateGenerator dispatch + cost cap + forward-progress
    detector + productivity detector + Iron Gate suite + retry feedback)
    to :class:`GENERATERunner`. Cross-phase artifacts (``generation``,
    ``_episodic_memory``) threaded via ``PhaseResult.artifacts`` for
    VALIDATE to consume. Slice delivery: 5a = spine parity, 5b = Iron
    Gate suite parity depth (same runner module + flag).
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_GENERATE_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_slice4b_extracted() -> bool:
    """Slice 4b of Wave 2 (5) — APPROVE + APPLY + VERIFY combined gate.

    **Default ``true`` as of 2026-04-23 graduation** (harness-class 4-session
    cadence bt-2026-04-23-033530 / -040327 / -043017 / -045653 — each 0 PM /
    0 runner-attributed frames / 0 shutdown race; reachability observed in
    4/4 via `[PhaseRunnerDelegate] APPROVE+APPLY+VERIFY → Slice4bRunner`
    markers on live RuntimeHealthSensor IMMEDIATE ops walking
    CLASSIFY → ROUTE+CTX+PLAN → VALIDATE → GATE → SLICE4B with APPLY
    HEARTBEAT @ 80% on `requirements.txt`; reachability_source=opportunistic
    per operator-accepted bar "real op hit the runner under flag-on, not
    that our backlog seed won a race"). Explicit ``=false`` remains a
    runtime kill switch reverting to the ~1150-line inline APPROVE+APPLY+VERIFY
    block.

    When ``true``, delegates the ~1150-line APPROVE + APPLY (with 7.5
    INFRA) + VERIFY (with 8a scoped tests, 8b auto-commit, 8b2 hot-reload,
    8c self-critique, 8d visual VERIFY) block to :class:`Slice4bRunner`.
    Mirror of the Slice 3 combined-gate approach: the three phases are
    deeply interleaved (APPROVE tail runs on every path; APPLY consumes
    APPROVE locals; VERIFY consumes APPLY locals) so per-phase flags
    would require 6-way artifact threading. Per-phase decomposition
    arrives with Slice 6 dispatcher cutover. ``t_apply`` is threaded
    via ``PhaseResult.artifacts["t_apply"]`` for COMPLETERunner's
    canary latency calculation.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_SLICE4B_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _phase_runner_gate_extracted() -> bool:
    """Slice 4a.2 of Wave 2 (5) — GATE phase extraction gate.

    **Default ``true`` as of 2026-04-23 graduation** (3 clean soak
    sessions bt-2026-04-23-005127 / -010733 / -012329, each 0 PM /
    $0 / 0 runner-attributed frames / 0 shutdown race; reachability
    observed in 2/3 sessions via ``[PhaseRunnerDelegate] GATE`` +
    ``[SemanticGuard]`` lines — S3 terminated upstream of GATE per
    downstream-of-VALIDATE reachability profile). Explicit ``=false``
    remains a runtime kill switch reverting to the 600-line inline
    GATE block.

    When ``true``, delegates the 600-line GATE block (can_write +
    SecurityReviewer + SimilarityGate + frozen_tier + risk ceiling +
    SemanticGuardian + REVIEW shadow + MutationGate + MIN_RISK_TIER
    floor + 5a green preview + 5b NOTIFY_APPLY yellow) to GATERunner.
    The ``risk_tier`` local mutates at up to 6 sites in GATE and is
    threaded back via ``PhaseResult.artifacts["risk_tier"]``.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_GATE_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _discover_tests_for_gate_worker(
    tests_dir_str: str, stem: str,
) -> List[str]:
    """Module-level worker for :meth:`Orchestrator._discover_tests_for_gate`.

    Dispatched into the shared ``advisor-blast`` thread pool via
    ``cooperative_fs_io.offload`` (fs-hot-tier Batch 3, row 15). Lifted
    out to module level so the offload trampoline doesn't capture any
    caller-local state, mirroring the pattern used by
    ``cooperative_fs_io._read_text_worker``. Returns path strings
    (not ``Path`` objects) — pickle-agnostic and cheap over the
    thread-pool boundary.
    """
    tests_dir = Path(tests_dir_str)
    found: List[str] = []
    for candidate in tests_dir.rglob(f"test_{stem}*.py"):
        if candidate.is_file():
            found.append(str(candidate))
    return sorted(found)


def _phase_runner_validate_extracted() -> bool:
    """Slice 4a.1 of Wave 2 (5) — VALIDATE phase extraction gate.

    **Default ``true`` as of 2026-04-22 graduation** (3 clean soak
    sessions bt-2026-04-22-230147 / -232323 / -235808, each 0 PM /
    $0 / 0 runner-attributed frames / 0 shutdown race; reachability
    observed in 2/3 sessions via 2 ``[PhaseRunnerDelegate] VALIDATE``
    delegation markers + 6 ``[ValidateRetryFSM]`` FSM transition lines).
    Explicit ``=false`` remains a runtime kill switch reverting to
    the 762-line inline VALIDATE block.

    When ``true``, delegates the 762-line VALIDATE block (nested retry
    FSM + L2 dispatch + source-drift + shadow harness + entropy +
    read-only short-circuit) to VALIDATERunner. Parity tests at
    ``tests/governance/phase_runner/test_validate_runner_parity.py``
    pin observable output across both paths. The ``best_candidate``
    local leaks downstream to GATE (37 refs) and is threaded via
    ``PhaseResult.artifacts``.

    Graduation ledger: ``memory/project_wave2_graduation_matrix.md``.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_VALIDATE_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


def _swe_bench_test_advisory(
    signal_source: str,
    op_id: str,
    candidate: "Dict[str, Any]",
    result: "ValidationResult",
) -> "ValidationResult":
    """Slice 66 — for swe_bench_pro ops, a VALIDATE ``test`` failure is ADVISORY.

    The benchmark repo's test env lives ONLY in the per-problem Docker image
    (the bare local env can't import qutebrowser/PyQt etc.), AND running the
    held-out ``fail_to_pass`` tests inside VALIDATE would LEAK them into the L2
    repair loop — the model would iterate against the evaluation oracle and game
    the score. So a ``test``-class failure is promoted to passed: the candidate
    proceeds to APPLY (captured), and the ONE-SHOT container scoring (Slice 65,
    in the autoscore layer that holds the ProblemSpec) is the authoritative
    held-out judge — exactly once, no leakage.

    This is NOT a bypass: syntax / build / infra failures STILL block (those are
    valid LOCAL checks — a patch must be well-formed). Pure + gated; non-
    swe_bench ops, non-``test`` failures, and already-passed results are returned
    unchanged (byte-identical). NEVER raises."""
    try:
        if result.passed:
            return result
        if result.failure_class != "test":
            return result
        if (signal_source or "") != "swe_bench_pro":
            return result
        from dataclasses import replace
        logger.info(
            "[Orchestrator] Slice 66 — swe_bench_pro 'test' failure is ADVISORY "
            "for op=%s (local env can't run repo tests; running held-out tests "
            "here would leak them). Promoting candidate to APPLY; the one-shot "
            "container scoring is the authoritative held-out judge.",
            op_id,
        )
        return replace(
            result,
            passed=True,
            best_candidate=candidate,
            error=None,
            failure_class=None,
            short_summary=(
                "swe_bench_pro: local test-gate advisory; container scoring "
                "is authoritative (Slice 66)"
            ),
        )
    except Exception:  # noqa: BLE001 — advisory must never break validation
        return result


def _swe_bench_verify_advisory(
    signal_source: str, verify_error: "Optional[str]", op_id: str,
) -> "Optional[str]":
    """Slice 67 — for swe_bench_pro ops, the post-APPLY VERIFY regression gate
    is ADVISORY (companion to the Slice 66 VALIDATE gate).

    The scoped/benchmark tests the VERIFY gate runs live only in the per-problem
    container, so locally they always regress (pass_rate=0.00). Worse, the gate
    ROLLS THE PATCH BACK on regression — which would wipe the candidate before
    the autoscore layer can capture + score it. Clearing ``verify_error`` for a
    benchmark op keeps the patch APPLIED (op ends state=applied → eval=resolved)
    so the ONE-SHOT held-out container scoring (Slice 65) is the authoritative
    judge. Pure + gated: non-swe_bench ops keep the strict gate (byte-identical);
    syntax/build are already enforced upstream. NEVER raises."""
    try:
        if not verify_error:
            return verify_error
        if (signal_source or "") != "swe_bench_pro":
            return verify_error
        logger.info(
            "[Orchestrator] Slice 67 — swe_bench_pro VERIFY regression gate "
            "ADVISORY for op=%s (local tests can't run without the container "
            "env; the held-out container scoring is authoritative). Keeping "
            "patch applied for capture: %s",
            op_id, verify_error,
        )
        return None
    except Exception:  # noqa: BLE001 — advisory must never break VERIFY
        return verify_error


def _phase_runner_slice3_fully_extracted() -> bool:
    """All three Slice 3 flags set — routes ROUTE+CTX+PLAN through runners.

    The three phases are currently interleaved in the inline pipeline
    (ROUTE body → conditional CTX → PLAN body); wiring each flag
    independently would require splitting the interleaving. For now,
    the dispatcher demands ALL THREE flags before using runners.
    Per-phase flags remain visible for env-var discoverability and
    future per-phase independence once Slice 6 (dispatcher cutover)
    decouples them entirely.
    """
    return (
        _phase_runner_route_extracted()
        and _phase_runner_context_expansion_extracted()
        and _phase_runner_plan_extracted()
    )


def _phase_runner_classify_extracted() -> bool:
    """Slice 2 of Wave 2 (5) — CLASSIFY phase extraction gate.

    Reads ``JARVIS_PHASE_RUNNER_CLASSIFY_EXTRACTED``, **default
    ``true`` as of 2026-04-22 graduation (3 clean soak sessions
    bt-2026-04-22-200312 / -202123 / -203723 with 38 total
    ``[PhaseRunnerDelegate] CLASSIFY → runner`` reachability markers
    + Slice 2 parity 22/22 byte-identical vs inline).** Explicit
    ``=false`` remains a runtime kill switch that reverts to the
    inline block.

    When ``true``, ``_run_pipeline`` delegates the 760-line CLASSIFY
    block (emergency check + advisor + risk classification + 8 prompt
    injections + advance to ROUTE + narrator/dialogue start +
    ClassifyClarify) to
    :class:`backend.core.ouroboros.governance.phase_runners.classify_runner.CLASSIFYRunner`.
    The ``_advisory`` + ``_consciousness_bridge`` locals leak
    downstream (Tier 6 personality voice + VERIFY L2 retry fragile-
    file injection) and are threaded back through
    ``PhaseResult.artifacts`` to preserve the data flow.

    When ``false``, the inline block runs unchanged. Parity tests
    (tests/governance/phase_runner/test_classify_runner_parity.py)
    pin observable output across both paths.

    Graduation ledger: ``memory/project_wave2_graduation_matrix.md``.
    """
    return (
        os.environ.get("JARVIS_PHASE_RUNNER_CLASSIFY_EXTRACTED", "true")
        .strip().lower() in _TRUTHY
    )


async def _inject_last_session_summary_impl(
    project_root: Path,
    ctx: OperationContext,
) -> OperationContext:
    """Inject rendered LastSessionSummary into ``ctx.strategic_memory_prompt``.

    Extracted from ``_run_pipeline`` for testability. Zero behavioral
    change vs. the inline block: reads LSS with ``get_default_summary``,
    appends the rendered dense one-liner(s) to the existing strategic
    memory prompt via ``with_strategic_memory_context``, emits the §8
    observability contract INFO line on success, DEBUG when disabled,
    and swallows any injection failure (returns ``ctx`` unchanged).

    Authority invariant unchanged: this path touches ONLY the prompt
    surface the model reads at CONTEXT_EXPANSION — zero authority over
    Iron Gate, UrgencyRouter, risk tier, policy engine, FORBIDDEN_PATH,
    ToolExecutor protected-path checks, or approval gating.
    """
    try:
        from backend.core.ouroboros.governance.last_session_summary import (
            get_default_summary,
        )
        _lss = get_default_summary(project_root)
        _lss_enabled, _lss_n, _lss_sid, _lss_chars, _lss_hash8 = (
            await _lss.inject_metrics()
        )
        if _lss_enabled:
            _lss_prompt = await _lss.format_for_prompt()
            if _lss_prompt:
                _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                ctx = ctx.with_strategic_memory_context(
                    strategic_intent_id=ctx.strategic_intent_id or "last-session-v1",
                    strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                    strategic_memory_prompt=(
                        _existing + "\n\n" + _lss_prompt
                        if _existing else _lss_prompt
                    ),
                    strategic_memory_digest=ctx.strategic_memory_digest,
                )
            logger.info(
                "[LastSessionSummary] op=%s enabled=true n_sessions=%d "
                "latest_session_id=%s chars_out=%d "
                "inject_site=context_expansion hash8=%s source=summary_json",
                ctx.op_id, _lss_n, _lss_sid, _lss_chars, _lss_hash8,
            )
        else:
            logger.debug(
                "[LastSessionSummary] op=%s enabled=false "
                "inject_site=context_expansion",
                ctx.op_id,
            )
    except Exception:
        logger.debug(
            "[Orchestrator] LastSessionSummary injection skipped",
            exc_info=True,
        )
    return ctx


def _inject_prior_knowledge_impl(ctx: OperationContext) -> OperationContext:
    """Inject Prior Ephemeral Knowledge into ``ctx.strategic_memory_prompt``.

    Extracted at module scope (mirrors ``_inject_last_session_summary_impl`` /
    ``_inject_postmortem_recall_impl``). Reads the boot-hydrated
    ``PriorKnowledgeCache`` singleton and renders up to
    ``JARVIS_COGNITIVE_INJECT_TOP_K`` cross-session experiences into the
    prompt via ``format_for_prompt``. Footprint resolution is best-effort:
    ``OperationContext`` does not carry resolved model/context-window
    attributes at CONTEXT_EXPANSION time (generation hasn't run yet), so
    ``footprint`` stays ``None`` in the default path and ``select()``
    degrades to the cross-footprint global top-K — this is the designed
    degradation, not an error.

    Authority invariant per PRD §12.2: read-only, best-effort, never blocks
    the FSM. Fail-soft — any exception is swallowed with a DEBUG breadcrumb
    and ``ctx`` is returned unchanged.
    """
    try:
        from backend.core.ouroboros.governance import cognitive_persistence as _cogp
        cache = _cogp.get_prior_knowledge_cache()
        footprint = None
        try:
            _model = getattr(ctx, "resolved_model_name", None)
            _num_ctx = getattr(ctx, "resolved_num_ctx", None)
            if _model:
                footprint = _cogp.cognitive_footprint(_model, _num_ctx)
        except Exception:
            footprint = None
        section = _cogp.format_for_prompt(cache, footprint)
        if not section:
            logger.debug(
                "[CognitivePersistence] op=%s inject_site=context_expansion "
                "section=empty",
                ctx.op_id,
            )
            return ctx
        _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
        ctx = ctx.with_strategic_memory_context(
            strategic_intent_id=ctx.strategic_intent_id or "prior-knowledge-v1",
            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
            strategic_memory_prompt=(
                _existing + "\n\n" + section if _existing else section
            ),
            strategic_memory_digest=ctx.strategic_memory_digest,
        )
        logger.info(
            "[CognitivePersistence] op=%s injected prior knowledge: %d chars "
            "footprint=%s inject_site=context_expansion",
            ctx.op_id, len(section), footprint or "any",
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[CognitivePersistence] injection skipped (fail-soft): %s", e,
        )
    return ctx


class _PreloadedExplorationRecord:
    """Synthetic exploration record for files the lean prompt builder inlined.

    ``ExplorationLedger.from_records`` duck-types on ``tool_name`` /
    ``arguments_hash`` / ``output_bytes`` / ``status``. When the lean prompt
    builder inlines a target file region directly into the generation
    prompt, the model has effectively "read" that file — we synthesize a
    fake ``read_file`` record so the ledger grants comprehension credit
    matching the legacy counter's ``_preloaded_credit`` behavior.

    Keeping this class in ``orchestrator.py`` preserves
    ``exploration_engine``'s pure-module contract (no orchestrator-side
    concepts leak into it). The ``preloaded:`` prefix on
    ``arguments_hash`` guarantees stable dedup per normalized path and
    no collision with a real ``read_file`` tool call.
    """

    __slots__ = ("tool_name", "arguments_hash", "output_bytes", "status")

    def __init__(self, path: str) -> None:
        self.tool_name = "read_file"
        self.arguments_hash = f"preloaded:{path}"
        self.output_bytes = 0
        self.status = "success"


async def _inject_postmortem_recall_impl(
    ctx: OperationContext,
) -> OperationContext:
    """Inject prior-op POSTMORTEM lessons into ``ctx.strategic_memory_prompt``.

    Extracted from ``_run_pipeline`` for testability (mirrors the
    ``_inject_last_session_summary_impl`` pattern). Zero behavioral change
    vs. the inline block: looks up POSTMORTEMs from prior sessions whose
    op_signature is similar to the current op's signature (file paths +
    descriptive intent) and injects up to ``top_k`` lessons into the prompt
    as the "## Lessons from prior similar ops" section.

    Authority invariant per PRD §12.2: read-only, best-effort, never blocks
    the FSM. Master flag default-off (``JARVIS_POSTMORTEM_RECALL_ENABLED``).
    When off this is byte-for-byte pre-P0 behavior. ``PostmortemRecallService``
    itself returns ``[]`` cleanly on any failure path; this wrapper additionally
    swallows any exception and emits a DEBUG breadcrumb.

    Closes the rooted "system has perfect memory and zero recall" gap from
    PRD §4.2 Shallow #2 — P0 of PRD Phase 1.
    """
    try:
        from backend.core.ouroboros.governance.postmortem_recall import (
            get_default_service as _get_pm_recall,
            render_recall_section as _render_pm_recall,
        )
        _pm_svc = _get_pm_recall()
        if _pm_svc is None:
            # Master flag off: emit observability breadcrumb so live-cadence
            # graduation can distinguish "helper ran with master off" from
            # "helper never ran". Mirrors LSS / ConversationBridge / SemanticIndex
            # disabled-state breadcrumbs (uniform CONTEXT_EXPANSION audit).
            logger.debug(
                "[PostmortemRecall] op=%s enabled=false "
                "inject_site=context_expansion",
                ctx.op_id,
            )
        else:
            _pm_target_files = ", ".join(
                sorted((ctx.target_files or ()))[:5]
            )
            _pm_op_signature = (
                f"description={(ctx.description or '')[:200]} | "
                f"files={_pm_target_files}"
            )
            _pm_matches = await _pm_svc.recall_for_op(_pm_op_signature)
            _pm_section = _render_pm_recall(_pm_matches)
            if _pm_section:
                _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                ctx = ctx.with_strategic_memory_context(
                    strategic_intent_id=ctx.strategic_intent_id or "pm-recall-p0",
                    strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                    strategic_memory_prompt=(
                        _existing + "\n\n" + _pm_section
                        if _existing else _pm_section
                    ),
                    strategic_memory_digest=ctx.strategic_memory_digest,
                )
                logger.info(
                    "[PostmortemRecall] op=%s enabled=true matched=%d "
                    "inject_site=context_expansion (P0 — PRD Phase 1)",
                    ctx.op_id, len(_pm_matches),
                )
            else:
                logger.debug(
                    "[PostmortemRecall] op=%s enabled=true matched=0 "
                    "inject_site=context_expansion",
                    ctx.op_id,
                )
    except Exception:
        logger.debug(
            "[Orchestrator] PostmortemRecall injection skipped",
            exc_info=True,
        )
    return ctx


def _reflect_cognitive_metrics_post_apply_impl(
    ctx: OperationContext,
    applied_files: Sequence[Any],
) -> None:
    """Phase 4 P3 follow-on — vindication call site at APPLY-success.

    Best-effort observability: when ``JARVIS_COGNITIVE_METRICS_ENABLED``
    is on AND the singleton is wired (set by orchestrator.__init__) AND
    a pre-apply snapshot was captured at CONTEXT_EXPANSION (via
    ``score_pre_apply``), calls ``CognitiveMetricsService.auto_reflect_post_apply``
    which computes before/after deltas and persists a vindication
    ``CognitiveMetricRecord`` to the JSONL ledger.

    Authority invariant per PRD §12.2: read-only, never blocks the FSM.
    Any exception (oracle down, ledger write failed) emits a DEBUG
    breadcrumb and returns silently. Vindication score is NOT consumed
    by Iron Gate / risk_tier / approve gating in this slice — advisory
    signal only, recorded for future Phase 4 work to consume.
    """
    try:
        from backend.core.ouroboros.governance.cognitive_metrics import (
            get_default_service as _get_cm_svc,
            is_enabled as _cm_enabled,
        )
        if not _cm_enabled():
            return
        svc = _get_cm_svc()
        if svc is None:
            return
        # applied_files is a Sequence[Path] from the orchestrator call
        # site; normalize to List[str] for the service API.
        target_strs = [str(p) for p in (applied_files or ())]
        if not target_strs:
            return
        svc.auto_reflect_post_apply(
            op_id=ctx.op_id,
            target_files=target_strs,
        )
    except Exception:
        logger.debug(
            "[Orchestrator] CognitiveMetrics post-apply reflection skipped",
            exc_info=True,
        )


def _score_cognitive_metrics_pre_apply_impl(
    ctx: OperationContext,
) -> None:
    """Phase 4 P3 — pre-APPLY oracle pre-score for the current op.

    Best-effort observability: when ``JARVIS_COGNITIVE_METRICS_ENABLED``
    is on AND the singleton is wired (set by orchestrator.__init__) AND
    the candidate ``ctx.target_files`` is non-empty, calls the wrapped
    ``OraclePreScorer`` and persists a ``CognitiveMetricRecord`` to the
    JSONL ledger. The pre-score is NOT consumed by Iron Gate / risk-tier
    / approve gating in this slice — it's an advisory signal only.
    Future slices can weight downstream decisions on it.

    Authority invariant per PRD §12.2: read-only, never blocks the FSM.
    Any exception (oracle down, ledger write failed, complexity probe
    raised) emits a DEBUG breadcrumb and returns silently.
    """
    try:
        from backend.core.ouroboros.governance.cognitive_metrics import (
            get_default_service as _get_cm_svc,
            is_enabled as _cm_enabled,
        )
        if not _cm_enabled():
            return
        svc = _get_cm_svc()
        if svc is None:
            return
        target_files = list(ctx.target_files or ())
        if not target_files:
            return
        # The OraclePreScorer accepts max_complexity + has_tests as
        # optional probes; we pass the conservative defaults so the
        # signal is computable on every well-formed ctx. Future slices
        # can wire real complexity probes.
        svc.score_pre_apply(
            op_id=ctx.op_id,
            target_files=target_files,
            max_complexity=0,
            has_tests=True,
        )
    except Exception:
        logger.debug(
            "[Orchestrator] CognitiveMetrics pre-score skipped",
            exc_info=True,
        )


def _plan_review_required() -> bool:
    """Return True when the session requires pre-execution plan review."""
    return (
        os.environ.get("JARVIS_SHOW_PLAN_BEFORE_EXECUTE", "").strip().lower()
        in _TRUTHY
    )


def _human_is_watching() -> bool:
    """Detect whether a human is likely watching the terminal.

    Returns ``True`` when any of:
    - ``sys.stdout`` is attached to an interactive TTY.
    - ``JARVIS_DIFF_PREVIEW_ALL`` env var is set to a truthy value.
      (Explicit flag for CI / headless modes where TTY is absent but the
      human is tailing logs.)

    Used to decide whether SAFE_AUTO (Green) operations should show a
    diff preview before auto-applying.
    """
    explicit = os.environ.get("JARVIS_DIFF_PREVIEW_ALL", "").lower() in (
        "true", "1", "yes",
    )
    if explicit:
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorConfig:
    """Frozen configuration for the governed pipeline orchestrator.

    Parameters
    ----------
    project_root:
        Root directory of the project being modified (jarvis repo).
    repo_registry:
        Optional multi-repo registry. When set, cross-repo saga applies
        resolve each repo's local_path from the registry instead of using
        project_root for all repos. Defaults to None (single-repo mode).
    generation_timeout_s:
        Maximum seconds for candidate generation (per attempt).
    validation_timeout_s:
        Maximum seconds for candidate validation (per attempt).
    approval_timeout_s:
        Maximum seconds to wait for human approval.
    max_generate_retries:
        Number of additional generation attempts after the first failure.
    max_validate_retries:
        Number of additional validation attempts after the first failure.
        Env-tunable via ``JARVIS_MAX_VALIDATE_RETRIES`` (default ``2``).

        Set to ``0`` to bypass retries and dispatch failures straight to
        L2 Repair on the first critique. The original justification was
        latency: for a complex multi-file op, each validation pass costs
        ~7 minutes, so 2 retries consume ~21 minutes before L2 can
        dispatch — exceeding a typical 20-minute idle budget.

        Session U (``bt-2026-04-15-215858``, FSM-instrumented) revealed
        a stronger second justification: re-validation is **non-
        deterministic across iterations**. Same candidate, same test
        targets — iter=0 returned ``failure_class='test'`` (the real LSP
        defect that L2 should repair) but iter=1 returned
        ``failure_class='infra'`` (a sandbox/pytest transient). The
        ``'infra'`` class is non-retryable by design: it triggers the
        early-return branch at ``_early_return_ctx`` and advances ctx
        straight to POSTMORTEM, killing the op on a flake instead of
        giving L2 a chance to repair the legitimate critique iter=0
        identified. Setting ``max_validate_retries=0`` takes the loop
        out of this race entirely — iter=0 runs once, the
        ``validate_retries_remaining`` counter decrements to ``-1``, and
        the L2 dispatch branch at ``validate_retries_remaining < 0``
        fires on the real ``'test'`` critique.

        Battle-test override: ``JARVIS_MAX_VALIDATE_RETRIES=0``.
    """

    project_root: Path
    repo_registry: Optional["RepoRegistry"] = None  # Forward ref avoids circular import; resolved at type-check time
    generation_timeout_s: float = 180.0
    validation_timeout_s: float = 60.0
    approval_timeout_s: float = 600.0
    max_generate_retries: int = 1
    max_validate_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("JARVIS_MAX_VALIDATE_RETRIES", "2")
        )
    )
    context_expansion_enabled: bool = True
    context_expansion_timeout_s: float = 30.0

    # Saga message bus (passive observability — created by GLS at startup)
    message_bus: Optional[Any] = None

    # Benchmarking
    benchmark_enabled: bool = True
    benchmark_timeout_s: float = 60.0

    # Model attribution
    model_attribution_enabled: bool = True
    model_attribution_lookback_n: int = 20
    model_attribution_min_sample_size: int = 3

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_publish_interval_s: float = 3600.0
    curriculum_window_n: int = 50
    curriculum_top_k: int = 5
    curriculum_impact_weights: Dict[str, float] = field(default_factory=dict)

    # Reactor event polling
    reactor_event_poll_interval_s: float = 30.0

    # L2 self-repair engine (disabled by default)
    # Set by GovernedLoopService._build_components() when JARVIS_L2_ENABLED=true.
    repair_engine: Optional[Any] = None
    execution_graph_scheduler: Optional[Any] = None

    # Shadow harness — optional; set by GovernedLoopService when
    # JARVIS_SHADOW_HARNESS_ENABLED=true in .env
    shadow_harness: Optional[Any] = None

    @property
    def execution_root(self) -> Path:
        """The mutation/judgment tree (Slice 11 role split).

        Twin of ``GovernedLoopConfig.execution_root`` — OrchestratorConfig
        is a separate frozen dataclass, so the property exists on both;
        each is a thin delegate to the ONE canonical seam
        (``autonomous_workspace.effective_execution_root``). Resolved
        lazily at every read: the ledger-sovereignty bootloader exports
        ``JARVIS_AUTO_COMMIT_WORKSPACE`` after configs are constructed.
        ``project_root`` keeps the observation role (sensors/TestWatcher).
        """
        from backend.core.ouroboros.governance.autonomous_workspace import (
            effective_execution_root,
        )

        return effective_execution_root(self.project_root)

    def resolve_repo_roots(
        self,
        repo_scope: Tuple[str, ...],
        op_id: str,
    ) -> Dict[str, Path]:
        """Resolve per-repo filesystem roots from registry; fallback to project_root.

        Parameters
        ----------
        repo_scope:
            Tuple of repo names from OperationContext.
        op_id:
            Operation ID for structured warning on missing registry keys.

        Returns
        -------
        Dict mapping repo name -> absolute Path.
        Missing keys fall back to project_root with a warning (never raise).
        """
        roots: Dict[str, Path] = {}
        for repo in repo_scope:
            if self.repo_registry is not None:
                try:
                    roots[repo] = Path(self.repo_registry.get(repo).local_path)
                except (KeyError, AttributeError, TypeError):
                    # repo_registry may be a duck-typed substitute; catch all lookup failures
                    logger.warning(
                        "[OrchestratorConfig] repo=%s not in registry for op_id=%s; "
                        "falling back to project_root=%s",
                        repo, op_id, self.project_root,
                    )
                    roots[repo] = self.project_root
            else:
                roots[repo] = self.project_root
        return roots


class _LiveWorkGateResult(NamedTuple):
    """Outcome of ``_live_work_apply_gate`` (Slice 10 + review C1).

    ``active_hit`` — ``(file, reason)`` when the gate went terminal on a
    human-active file (wait master off, infinite horizon, or an
    unaffordable horizon); ``None`` when the scan cleared.
    ``waited_s`` — total seconds actually slept inside this invocation.
    ``drift_stale_files`` — non-``None`` iff the gate WAITED and the
    post-wait re-run of the stale-exploration drift check
    (``state_drift.should_block_apply`` — the SAME helper the pre-gate
    check uses) came back blocking: a human edit made mid-wait is
    invisible to the pre-gate hash snapshot (TOCTOU), so callers must
    route this to the SAME ``state_drift_unreconciled`` terminal shape.
    """

    active_hit: Optional[Tuple[str, str]]
    waited_s: float
    drift_stale_files: Optional[List[str]]


# ---------------------------------------------------------------------------
# GovernedOrchestrator
# ---------------------------------------------------------------------------


def _await_approval_with_operator(orch: Any, request_id: str, ctx: Any) -> Any:
    """Await the gate decision, letting an attached cockpit answer it.

    Degrades to the plain provider wait on ANY failure: the gate's behaviour
    with nobody attached must stay byte-identical, so this can only ADD a way
    to answer.
    """
    timeout_s = orch._config.approval_timeout_s
    try:
        from backend.core.ouroboros.governance.approval_narrator import (
            await_decision_with_operator,
        )

        def _emit(line: str) -> None:
            # Same late-bound mirror the subagent narrator uses: SerpentFlow
            # attaches after the stack is built, so a handle captured at
            # construction time would be None forever.
            flow = getattr(orch, "_serpent_flow", None) or getattr(
                getattr(orch, "_gls", None), "_serpent_flow", None,
            )
            mirror = getattr(flow, "_mirror_markup", None) if flow else None
            if mirror is not None:
                mirror(line)

        return await_decision_with_operator(
            orch._approval_provider, request_id, timeout_s,
            emit=_emit,
            risk=str(getattr(getattr(ctx, "risk_tier", None), "name", "") or ""),
            reason=str(getattr(ctx, "approval_reason", "") or ""),
        )
    except Exception:  # noqa: BLE001
        return orch._approval_provider.await_decision(request_id, timeout_s)


class GovernedOrchestrator:
    """Central coordinator for the governed self-programming pipeline.

    Delegates to existing governance components (risk_engine, change_engine,
    ledger, canary via can_write).  Owns NO domain logic -- only phase
    transitions and error handling.

    Parameters
    ----------
    stack:
        GovernanceStack providing risk_engine, ledger, comm, change_engine,
        and the can_write() gate.
    generator:
        CandidateGenerator for code generation (has generate(context, deadline)).
    approval_provider:
        Optional ApprovalProvider for human-in-the-loop gate (has request(),
        await_decision()).
    config:
        Orchestrator configuration.
    """

    def __init__(
        self,
        stack: Any,
        generator: Any,
        approval_provider: Any,
        config: OrchestratorConfig,
        validation_runner: Any = None,  # LanguageRouter | duck-typed for testing
    ) -> None:
        self._stack = stack
        self._generator = generator
        self._approval_provider = approval_provider
        self._config = config
        self._validation_runner = validation_runner
        # Fail-Fast Exhaustion Circuit Breaker: per-op consecutive
        # all_providers_exhausted count, keyed by stable op_id so it
        # survives Stage-1.6 park/resume re-dispatch. Pruned on
        # success and on terminal. Only consulted when
        # _failfast_cb_enabled() (§33.1 default-FALSE).
        self._failfast_exhaust_consec: "Dict[str, int]" = {}

        # Phase B REVIEW subagent — harness-attached post-construction via
        # set_subagent_orchestrator() so the constructor signature stays
        # stable. None until governed_loop_service wires it.
        self._subagent_orchestrator: Any = None

        # ── Phase 1 Step 3C: reload-hostile state hoisted to _governance_state ──
        # Every field that would otherwise get re-allocated on
        # ``importlib.reload(orchestrator)`` now lives on an
        # :class:`OrchestratorState` dataclass in the quarantined
        # ``_governance_state`` module. When
        # ``JARVIS_UNQUARANTINE_ORCHESTRATOR=true``, the state is a
        # process-wide singleton — the second-generation orchestrator
        # instance rebinds into the already-populated state without
        # losing the oracle update lock, cost governor, forward-
        # progress detector, session lessons, RSI trackers, hot-reload
        # subscription, or any of the seven harness-attached refs.
        # When the flag is false (default during rollout), each call
        # mints a fresh state via ``OrchestratorState.fresh(...)`` so
        # behavior is bit-for-bit identical to the pre-hoist code.
        from backend.core.ouroboros.governance._governance_state import (
            OrchestratorState,
            get_orchestrator_state,
            unquarantine_orchestrator_enabled,
        )

        if unquarantine_orchestrator_enabled():
            self._state = get_orchestrator_state(
                project_root=self._config.project_root,
            )
            logger.info(
                "[Orchestrator] Unquarantined state path engaged — "
                "reload-hostile roots sourced from process-wide "
                "OrchestratorState singleton",
            )
        else:
            self._state = OrchestratorState.fresh(
                project_root=self._config.project_root,
            )

        # Bind-once aliases for container-stable roots. These fields are
        # never reassigned after __init__ — the dataclass attribute
        # identity stays put, so an instance-level alias is safe and
        # minimizes call-site churn. Compare with the property/setter
        # pattern below, which is required for *rebindable* fields.
        self._oracle_update_lock: asyncio.Lock = self._state.oracle_update_lock
        self._cost_governor: CostGovernor = self._state.cost_governor
        # Register the cost_governor as the process-wide default so
        # pure helper modules (PLAN-EXPLOIT, etc.) can look it up
        # without taking it as a parameter through every call site.
        # Best-effort — never fails orchestrator construction.
        try:
            from backend.core.ouroboros.governance.cost_governor import (
                set_default_cost_governor as _set_default_cg,
            )
            _set_default_cg(self._cost_governor)
        except Exception:  # noqa: BLE001
            pass

        # Phase 4 P3 (2026-04-26) — un-strand the OraclePreScorer +
        # VindicationReflector via the CognitiveMetricsService wrapper.
        # Wires the singleton with the live Oracle off the stack so that
        # both the orchestrator helpers and the /cognitive REPL surface
        # share one service per process. Best-effort: any failure means
        # cognitive metrics are observed-only-via-repl rather than
        # auto-scored — never breaks orchestrator construction.
        try:
            from backend.core.ouroboros.governance.cognitive_metrics import (
                CognitiveMetricsService as _CMSvc,
                is_enabled as _cm_enabled,
                set_default_service as _set_default_cm,
            )
            if _cm_enabled():
                _oracle_for_cm = getattr(self._stack, "oracle", None)
                if _oracle_for_cm is not None:
                    _set_default_cm(_CMSvc(
                        oracle=_oracle_for_cm,
                        project_root=self._config.project_root,
                    ))
        except Exception:  # noqa: BLE001
            pass

        self._forward_progress: ForwardProgressDetector = self._state.forward_progress
        self._productivity_detector: ProductivityDetector = (
            self._state.productivity_detector
        )
        # Counter dataclass alias — see class-level note on why the
        # candidate_generator pattern (bind once, mutate attributes on
        # the stable dataclass) is safer than property/setter for int
        # read-modify-write patterns like ``x += 1``.
        self._counters = self._state.counters

        # Config-only ints. Not reload-hostile because they're derived
        # from env vars and re-read on construction — the post-reload
        # instance gets the same value without indirection.
        _max = int(os.environ.get("JARVIS_SESSION_LESSONS_MAX", "20"))
        self._session_lessons_max: int = max(5, _max)
        # Slice 1 concurrency remediation (2026-07-18): serializes the COMPOUND
        # lesson-buffer mutations (append+cap+rebind in _add_session_lesson; the
        # convergence-negative clear+counter-reset) across the 3 BackgroundAgent-
        # Pool workers sharing this singleton. threading.RLock (not asyncio.Lock)
        # deliberately: the mutation sites are SYNC methods called from async
        # code — an asyncio.Lock cannot be acquired there — and an RLock also
        # covers the codebase's real threads (wall-clock watchdog, embed pool),
        # which an asyncio.Lock would not. Mirrors in_flight_registry's RLock
        # precedent. `with` context manager guarantees release on exceptions.
        import threading as _threading
        self._session_lessons_lock = _threading.RLock()
        self._convergence_check_interval: int = int(
            os.environ.get("JARVIS_LESSON_CONVERGENCE_CHECK_INTERVAL", "10")
        )

        # Log whichever trackers are live on the bound state. Preserves
        # the legacy debug-level visibility without re-running the
        # optional-module try/except chain (that happens once inside
        # ``OrchestratorState.fresh()``).
        if self._state.rsi_score_function is None:
            logger.debug("RSI: CompositeScoreFunction not available")
        if self._state.rsi_convergence_tracker is None:
            logger.debug("RSI: ConvergenceTracker not available")
        if self._state.rsi_transition_tracker is None:
            logger.debug("RSI: TransitionProbabilityTracker not available")
        _hr = self._state.hot_reloader
        if _hr is not None:
            logger.info(
                "[Orchestrator] ModuleHotReloader armed (%d safe modules)",
                len(_hr.safe_modules),
            )

    # ─────────────────────────────────────────────────────────────────
    # Phase 1 Step 3C: property/setter pairs for rebindable state
    # ─────────────────────────────────────────────────────────────────
    #
    # Every field below is either slice-rebound (``xs = xs[-CAP:]``)
    # or set to ``None`` at construction and later reassigned via a
    # harness ``set_*()`` method. Both patterns would plant a *real*
    # instance attribute that shadows any plain descriptor on the
    # class, so the rebind would silently drift away from the
    # :class:`OrchestratorState` singleton on the next reload.
    #
    # The property/setter pair fixes this by routing every read *and*
    # write through ``self._state.<field>``. The instance never grows
    # an attribute that could shadow the class descriptor, so the
    # post-reload instance sees the already-populated state.
    #
    # In-place mutations (``session_lessons.append(x)``,
    # ``session_lessons.clear()``) are alias-safe — they operate on
    # the list identity held inside ``self._state``, not on a local
    # copy — so the existing call sites keep working unchanged.

    def _resolve_session_id(self) -> str:
        """Best-available session id for session-scoped memory quarantine (LR2).
        Falls back to a stable per-process token so a NEW process (a NEW soak) is
        a NEW session scope even if no explicit session id is plumbed. Never raises."""
        try:
            # Canonical session source used elsewhere in the orchestrator.
            from backend.core.ouroboros.governance.strategic_direction import (
                get_active_session_id,
            )
            _v = get_active_session_id()
            if _v:
                return str(_v)
        except Exception:  # noqa: BLE001
            pass
        try:
            for attr in ("_session_id", "session_id"):
                v = getattr(self, attr, None)
                if v:
                    return str(v)
            _sd = getattr(self, "_session_dir", None)
            if _sd:
                from pathlib import Path as _P
                return _P(str(_sd)).name
        except Exception:  # noqa: BLE001
            pass
        # stable per-process fallback (a process == a session for the daemon)
        import os as _os
        return f"pid-{_os.getpid()}"

    @property
    def _subagent_scheduler(self) -> Any:
        """Alias to ``_config.execution_graph_scheduler``.

        Added 2026-04-24 (S7 finding) to close the W3(6) Slice 4 wiring gap:
        ``phase_dispatcher.py`` reads ``orchestrator._subagent_scheduler``
        when deciding whether to run the post-GENERATE enforce-mode
        ``dispatch_fanout`` path; orchestrator stores the same handle as
        ``_config.execution_graph_scheduler`` (passed in via
        ``OrchestratorConfig`` from ``governed_loop_service``). Pre-fix
        ``getattr`` returned ``None`` and the enforce path always logged
        ``enforce_fanout skipped: orchestrator has no _subagent_scheduler
        reference``. The alias keeps the dispatcher's call shape stable
        while making the field reachable.
        """
        return self._config.execution_graph_scheduler

    @property
    def _cancel_token_registry(self) -> Any:
        """Forward to GovernedLoopService's :class:`CancelTokenRegistry`.

        W3(7) Slice 2 — gives the dispatcher a single attribute lookup to
        find the per-session registry. The registry lives on GLS (created
        in __init__); the orchestrator surfaces it via ``self._stack``.
        Returns ``None`` for unit-test orchestrators constructed without
        a stack — runners must handle ``pctx.cancel_token is None``
        cleanly (no race wrap, behavior identical to pre-W3(7)).
        """
        _gls = getattr(self._stack, "governed_loop_service", None)
        if _gls is None:
            return None
        return getattr(_gls, "_cancel_token_registry", None)

    @property
    def _session_lessons(self) -> list:
        return self._state.session_lessons

    @_session_lessons.setter
    def _session_lessons(self, value: list) -> None:
        self._state.session_lessons = value

    @property
    def _ops_before_lesson(self) -> int:
        return self._state.counters.ops_before_lesson

    @_ops_before_lesson.setter
    def _ops_before_lesson(self, value: int) -> None:
        self._state.counters.ops_before_lesson = value

    @property
    def _ops_before_lesson_success(self) -> int:
        return self._state.counters.ops_before_lesson_success

    @_ops_before_lesson_success.setter
    def _ops_before_lesson_success(self, value: int) -> None:
        self._state.counters.ops_before_lesson_success = value

    @property
    def _ops_after_lesson(self) -> int:
        return self._state.counters.ops_after_lesson

    @_ops_after_lesson.setter
    def _ops_after_lesson(self, value: int) -> None:
        self._state.counters.ops_after_lesson = value

    @property
    def _ops_after_lesson_success(self) -> int:
        return self._state.counters.ops_after_lesson_success

    @_ops_after_lesson_success.setter
    def _ops_after_lesson_success(self, value: int) -> None:
        self._state.counters.ops_after_lesson_success = value

    @property
    def _rsi_score_function(self) -> Optional[Any]:
        return self._state.rsi_score_function

    @_rsi_score_function.setter
    def _rsi_score_function(self, value: Optional[Any]) -> None:
        self._state.rsi_score_function = value

    @property
    def _rsi_score_history(self) -> Optional[Any]:
        return self._state.rsi_score_history

    @_rsi_score_history.setter
    def _rsi_score_history(self, value: Optional[Any]) -> None:
        self._state.rsi_score_history = value

    @property
    def _rsi_convergence_tracker(self) -> Optional[Any]:
        return self._state.rsi_convergence_tracker

    @_rsi_convergence_tracker.setter
    def _rsi_convergence_tracker(self, value: Optional[Any]) -> None:
        self._state.rsi_convergence_tracker = value

    @property
    def _rsi_transition_tracker(self) -> Optional[Any]:
        return self._state.rsi_transition_tracker

    @_rsi_transition_tracker.setter
    def _rsi_transition_tracker(self, value: Optional[Any]) -> None:
        self._state.rsi_transition_tracker = value

    @property
    def _hot_reloader(self) -> Optional[Any]:
        return self._state.hot_reloader

    @_hot_reloader.setter
    def _hot_reloader(self, value: Optional[Any]) -> None:
        self._state.hot_reloader = value

    # § 4 attached refs — all seven flow through ``self._state``.
    # Harness ``set_*()`` methods below assign through these setters,
    # so rebinding the orchestrator class does not require re-running
    # the harness wiring pass.

    @property
    def _reasoning_bridge(self) -> Optional[Any]:
        return self._state.reasoning_bridge

    @_reasoning_bridge.setter
    def _reasoning_bridge(self, value: Optional[Any]) -> None:
        self._state.reasoning_bridge = value

    @property
    def _infra_applicator(self) -> Optional[Any]:
        return self._state.infra_applicator

    @_infra_applicator.setter
    def _infra_applicator(self, value: Optional[Any]) -> None:
        self._state.infra_applicator = value

    @property
    def _reasoning_narrator(self) -> Optional[Any]:
        return self._state.reasoning_narrator

    @_reasoning_narrator.setter
    def _reasoning_narrator(self, value: Optional[Any]) -> None:
        self._state.reasoning_narrator = value

    @property
    def _dialogue_store(self) -> Optional[Any]:
        return self._state.dialogue_store

    @_dialogue_store.setter
    def _dialogue_store(self, value: Optional[Any]) -> None:
        self._state.dialogue_store = value

    @property
    def _pre_action_narrator(self) -> Optional[Any]:
        return self._state.pre_action_narrator

    @_pre_action_narrator.setter
    def _pre_action_narrator(self, value: Optional[Any]) -> None:
        self._state.pre_action_narrator = value

    @property
    def _exploration_fleet(self) -> Optional[Any]:
        return self._state.exploration_fleet

    @_exploration_fleet.setter
    def _exploration_fleet(self, value: Optional[Any]) -> None:
        self._state.exploration_fleet = value

    @property
    def _critique_engine(self) -> Optional[Any]:
        return self._state.critique_engine

    @_critique_engine.setter
    def _critique_engine(self, value: Optional[Any]) -> None:
        self._state.critique_engine = value

    def set_reasoning_bridge(self, bridge: Any) -> None:
        """Attach a ReasoningChainBridge for pre-CLASSIFY reasoning.

        Writes through the :attr:`_reasoning_bridge` setter, which
        routes into ``self._state.reasoning_bridge``. When the
        orchestrator class reloads, the new instance inherits the
        already-populated state and the harness does *not* need to
        re-run this setter.
        """
        self._reasoning_bridge = bridge

    def set_infra_applicator(self, applicator: Any) -> None:
        """Attach an InfrastructureApplicator for deterministic post-APPLY hooks."""
        self._infra_applicator = applicator

    def set_reasoning_narrator(self, narrator: Any) -> None:
        """Attach a ReasoningNarrator for WHY-not-WHAT explanations."""
        self._reasoning_narrator = narrator

    def set_dialogue_store(self, store: Any) -> None:
        """Attach an OperationDialogueStore for reasoning journal recording."""
        self._dialogue_store = store

    def set_pre_action_narrator(self, narrator: Any) -> None:
        """Attach a PreActionNarrator for real-time WHAT-is-about-to-happen voice."""
        self._pre_action_narrator = narrator

    def set_exploration_fleet(self, fleet: Any) -> None:
        """Attach an ExplorationFleet for parallel codebase exploration."""
        self._exploration_fleet = fleet

    def set_critique_engine(self, engine: Any) -> None:
        """Attach a self-critique engine (Phase 3a).

        The engine runs after successful VERIFY + auto-commit and before
        the COMPLETE transition. Passing ``None`` detaches it. See
        ``self_critique.CritiqueEngine`` for the expected shape.
        """
        self._critique_engine = engine

    def set_subagent_orchestrator(self, orch: Any) -> None:
        """Attach the Phase B ``SubagentOrchestrator`` for REVIEW shadow dispatch.

        The orchestrator is the single spawn point for ephemeral REVIEW
        subagents (see ``subagent_orchestrator.py:dispatch_review``). Passing
        ``None`` detaches it — the post-VALIDATE shadow hook then no-ops.
        """
        self._subagent_orchestrator = orch

    async def _run_review_shadow(self, ctx: Any, best_candidate: Any) -> Any:
        """Phase B — post-VALIDATE REVIEW subagent in OBSERVER MODE.

        Gated by ``JARVIS_REVIEW_SUBAGENT_SHADOW`` (default **``true``**,
        graduated 2026-04-20). When on, dispatches a REVIEW subagent per
        candidate file and emits the verdict to telemetry. **The FSM
        proceeds to GATE regardless of verdict** — no risk-tier change,
        no retry routing, no state mutation. The contract stays
        observer-only even post-graduation; promoting REVIEW into
        authority-carrying gate logic is a separate slice with its own
        graduation arc.

        Graduation evidence (2026-04-20):
          * 28-test regression spine green (test_review_subagent.py +
            test_review_subagent_correlation.py).
          * Session 1 live FSM integration: observer hook fired
            post-VALIDATE at 25ms latency, FSM continued without
            interruption, aggregate telemetry format proven stable.
          * Session 2 Path B synthetic reject-proof: aggregate=REJECT
            emitted correctly for a poisoned candidate carrying the
            credential_shape_introduced pattern, with findings[0]
            identifying the triggering pattern precisely. Surfaced and
            fixed a latent case-mismatch bug in the aggregation logic
            (pinned by two new regression tests).
          * Upstream intelligence (Claude sonnet-4-6) refused to generate
            the credential-shape poison on its own — a complementary
            safety layer; REVIEW is the net for cases Claude's RLHF
            guardrails don't catch, not a redundant check.

        Must not raise under any condition — the observer contract forbids
        the shadow from breaking the main generation loop.

        Returns a ``shadow_enforce.ReviewAggregate`` describing the worst-of-N
        verdict (or ``None`` when there was nothing to review / the shadow was
        skipped). The enforce branch at the call site consumes this to gate
        the FSM via the EXISTING risk-tier escalation; when REVIEW-enforce is
        OFF (default) the caller ignores the return and behavior is
        byte-identical to the legacy shadow. The method itself NEVER gates —
        it stays a pure observer; gating is the caller's job.
        """
        from backend.core.ouroboros.governance.shadow_enforce import ReviewAggregate

        if best_candidate is None:
            return None
        if self._subagent_orchestrator is None:
            return None
        # NOTE: the shadow flag must stay enabled for the REVIEW subagent to
        # run at all (it is the dispatch gate). REVIEW-enforce composes ON TOP
        # of the shadow dispatch — it consumes the verdict the shadow already
        # computes. When the shadow is off there is no verdict to enforce.
        if os.environ.get(
            "JARVIS_REVIEW_SUBAGENT_SHADOW", "true"
        ).lower() not in ("true", "1"):
            return None

        try:
            _files = best_candidate.get("files") if isinstance(
                best_candidate.get("files"), list,
            ) else None
            _iter = (
                [
                    (entry.get("file_path", ""), entry.get("full_content", ""))
                    for entry in _files
                    if isinstance(entry, dict)
                ]
                if _files
                else [(
                    best_candidate.get("file_path", ""),
                    best_candidate.get("full_content", ""),
                )]
            )

            _t0 = time.monotonic()
            _verdicts: list = []
            for _path, _new in _iter:
                if not _path or not isinstance(_new, str):
                    continue
                _old = ""
                try:
                    _abs = (
                        self._config.project_root / _path
                        if not Path(_path).is_absolute()
                        else Path(_path)
                    )
                    if _abs.is_file():
                        _old = _abs.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    _old = ""

                _result = await self._subagent_orchestrator.dispatch_review(
                    parent_ctx=ctx,
                    file_path=_path,
                    pre_apply_content=_old,
                    candidate_content=_new,
                    generation_intent=getattr(ctx, "description", "") or "(no description)",
                    timeout_s=30.0,
                )

                _verdict = "unknown"
                _score = 0.0
                for _k, _v in (_result.type_payload or ()):
                    if _k == "verdict":
                        _verdict = str(_v)
                    elif _k == "semantic_integrity_score":
                        try:
                            _score = float(_v)
                        except (TypeError, ValueError):
                            pass
                _status = (
                    _result.status.value
                    if hasattr(_result.status, "value")
                    else str(_result.status)
                )
                _verdicts.append((_path, _verdict, _score, _status))

            _duration_ms = int((time.monotonic() - _t0) * 1000)

            # Aggregate: worst-of across files. REJECT dominates,
            # APPROVE_WITH_RESERVATIONS dominates APPROVE.
            #
            # Verdict comparison uses the string constants from
            # subagent_contracts (values: "reject", "approve_with_reservations",
            # "approve") — NOT uppercase literals. A prior uppercase comparison
            # silently reclassified every REJECT as APPROVE in the aggregate
            # telemetry (caught 2026-04-20 via synthetic reject-proof harness).
            # The _aggregate output string stays uppercase for stable log
            # parsing ("aggregate=REJECT"); only the input comparison is
            # lowercase-matched against the verdict values on the wire.
            from backend.core.ouroboros.governance.subagent_contracts import (
                REVIEW_VERDICT_APPROVE_WITH_RESERVATIONS,
                REVIEW_VERDICT_REJECT,
            )
            _counts = {"approved": 0, "reservations": 0, "rejected": 0, "failed": 0}
            _aggregate = "APPROVE" if _verdicts else "NO_FILES"
            for _p, _v, _s, _st in _verdicts:
                if _st != "completed":
                    _counts["failed"] += 1
                    continue
                if _v == REVIEW_VERDICT_REJECT:
                    _counts["rejected"] += 1
                    _aggregate = "REJECT"
                elif _v == REVIEW_VERDICT_APPROVE_WITH_RESERVATIONS:
                    _counts["reservations"] += 1
                    if _aggregate != "REJECT":
                        _aggregate = "APPROVE_WITH_RESERVATIONS"
                else:
                    _counts["approved"] += 1

            # Stable structured line — key=value so a simple split("=") parser
            # can build rollup counters (aggregate-verdict distribution,
            # approve/reject rates, per-session verdict sanity) across the
            # graduation arc. Matches the [SemanticGuard] log convention.
            logger.info(
                "[REVIEW-SHADOW] op=%s aggregate=%s files_reviewed=%d "
                "approved=%d reservations=%d rejected=%d failed=%d "
                "duration_ms=%d (observer — FSM proceeds regardless)",
                getattr(ctx, "op_id", "?"),
                _aggregate,
                len(_verdicts),
                _counts["approved"],
                _counts["reservations"],
                _counts["rejected"],
                _counts["failed"],
                _duration_ms,
            )

            # Build the aggregate the enforce branch consumes. This is a pure
            # restatement of the telemetry counts already computed above — the
            # shadow log line and the returned aggregate are the SAME verdict.
            return ReviewAggregate(
                aggregate=_aggregate,
                files_reviewed=len(_verdicts),
                rejected=_counts["rejected"],
                reservations=_counts["reservations"],
                approved=_counts["approved"],
                failed=_counts["failed"],
                had_failure=_counts["failed"] > 0,
            )
        except Exception:
            # Observer contract: shadow must never break the FSM. A whole-
            # dispatch crash is a SUBSYSTEM failure -> fail-SOFT to the legacy
            # shadow behavior: return None so the enforce branch does NOT gate
            # (never block the op on a telemetry/subsystem failure). This is
            # distinct from an ambiguous *verdict* (a per-file review that
            # completes with FAILED status -> had_failure=True -> escalate):
            # fail-CLOSED is on the verdict, fail-SOFT is on the subsystem.
            logger.debug(
                "[Orchestrator] REVIEW shadow dispatch skipped",
                exc_info=True,
            )
            return None

    def _review_downlevel_hard_blocked(self, ctx: Any) -> bool:
        """The base-tier hard sources that NO later GATE gate re-clamps: the
        self-modification cage and the delegated-provenance (sanctioned-goal)
        ceiling. A subagent APPLY down-level must never relax an op held for
        either. Every OTHER hard gate (similarity, frozen, risk ceiling,
        SemanticGuardian, mutation, MIN_RISK_TIER floor) runs AFTER the review
        seam and re-clamps on its own, so this check is deliberately narrow.
        FAIL-CLOSED: any doubt returns True (block the relax). NEVER raises."""
        try:
            from backend.core.ouroboros.governance.trust_calibration import (
                _op_touches_cage,
            )
            _files = [str(f) for f in (getattr(ctx, "target_files", ()) or ())]
            if _op_touches_cage(_files):
                return True
            # A delegated-provenance op (operator /goal, verify_provenance_claim
            # -> "ceiling APPROVAL_REQUIRED, never auto-apply") carries an
            # EXPLICIT human-approval ceiling the operator chose; a subagent must
            # not reason around it. Presence of the claim in the intake evidence
            # is sufficient — a false positive fails CLOSED (keeps the human).
            _ev = str(getattr(ctx, "intake_evidence_json", "") or "")
            if _ev and '"provenance"' in _ev:
                return True
            return False
        except Exception:  # noqa: BLE001 — a doubt about provenance blocks it
            return True

    async def _apply_review_gate(
        self, ctx: Any, best_candidate: Any, risk_tier: Any,
    ) -> Any:
        """Graduate the REVIEW subagent into the risk gate — the SINGLE seam the
        extracted ``gate_runner`` and its inline twin both call, applied at GATE
        ENTRY so the relax-then-reclamp state machine does the rest.

        Dispatches the REVIEW shadow once and folds its verdict into the tier:

          * **REVIEW-enforce** (``JARVIS_REVIEW_SUBAGENT_ENFORCE``, default on):
            a REJECT / ambiguous / reservations verdict ESCALATES the tier via
            the same stricter-wins clamp SemanticGuardian uses — the safety veto.
          * **APPLY-authorize** (``JARVIS_SUBAGENT_APPLY_AUTHORIZE``, default on):
            a CLEAN, unanimous, failure-free approve DOWN-LEVELS a ROUTINE
            ``APPROVAL_REQUIRED`` to ``NOTIFY_APPLY`` (auto-apply WITH a diff
            notice), authorizing VERIFY->APPLY without a human -- but NEVER when
            the tier is owned by the cage or the provenance ceiling
            (:meth:`_review_downlevel_hard_blocked`), and never below
            ``NOTIFY_APPLY``. Because this runs BEFORE similarity / frozen /
            ceiling / SemanticGuardian / mutation / the MIN_RISK_TIER floor,
            each of those re-clamps a wrongly-relaxed tier on its own — the
            down-level cannot outrun a hard gate.

        Returns the (possibly-adjusted) ``risk_tier``. Fail-SOFT on a subsystem
        error (tier unchanged); the verdict-level decisions are fail-CLOSED
        inside ``shadow_enforce``. NEVER raises into GATE."""
        try:
            from backend.core.ouroboros.governance.shadow_enforce import (
                aggregate_to_tier_floor,
                authorize_apply_downlevel,
                escalate_risk_tier,
                review_enforce_enabled,
            )
            _agg = await self._run_review_shadow(ctx, best_candidate)
            if _agg is None:
                return risk_tier
            if review_enforce_enabled():
                risk_tier = escalate_risk_tier(
                    risk_tier, aggregate_to_tier_floor(_agg),
                )
            risk_tier = authorize_apply_downlevel(
                risk_tier, _agg,
                hard_gate_present=self._review_downlevel_hard_blocked(ctx),
            )
            return risk_tier
        except Exception:  # noqa: BLE001 — the review gate never breaks GATE
            logger.debug(
                "[Orchestrator] review gate skipped (fail-soft)", exc_info=True,
            )
            return risk_tier

    async def _run_plan_shadow(self, ctx: Any) -> Any:
        """Phase B PLAN-shadow — AgenticPlanSubagent dispatch running
        alongside the legacy ``PlanGenerator`` as an observer.

        Gated by ``JARVIS_PLAN_SUBAGENT_SHADOW`` (default **``true``**,
        graduated 2026-04-20). When on, this hook:
          * Dispatches the PLAN subagent with ctx.target_files + description
          * Receives an execution_graph 2d.1-shaped payload back
          * Stashes the payload into ``ctx.execution_graph`` **without
            touching ``ctx.implementation_plan``** (the legacy flat-list
            plan remains the authoritative input to GENERATE; the DAG is
            observer-only this slice)
          * Emits a stable ``[PLAN-SHADOW]`` telemetry line so the legacy
            flat list and the subagent DAG can be compared across ops

        Single-file ops and ops with no target files skip — there is no
        DAG to build. Dispatch failures bump DEBUG logs only; the FSM
        proceeds regardless. Returns the (possibly updated) context so
        the caller can chain ``ctx = await self._run_plan_shadow(ctx)``.
        """
        if self._subagent_orchestrator is None:
            return ctx
        if os.environ.get(
            "JARVIS_PLAN_SUBAGENT_SHADOW", "true",
        ).lower() not in ("true", "1"):
            return ctx

        target_files = tuple(
            t for t in (getattr(ctx, "target_files", ()) or ()) if t
        )
        if len(target_files) < 2:
            # Single-file or zero-file op → no DAG to build.
            return ctx

        try:
            _t0 = time.monotonic()
            _description = (
                getattr(ctx, "description", "")
                or getattr(ctx, "goal", "")
                or "(no description)"
            )
            _primary_repo = getattr(ctx, "primary_repo", "jarvis") or "jarvis"
            _risk_tier = str(getattr(ctx, "risk_tier", "") or "")

            _result = await self._subagent_orchestrator.dispatch_plan(
                parent_ctx=ctx,
                op_description=str(_description),
                target_files=target_files,
                primary_repo=str(_primary_repo),
                risk_tier=_risk_tier,
                timeout_s=30.0,
            )

            # Extract the stable metrics from type_payload. Any missing
            # key falls back to a neutral default — the shadow contract
            # guarantees no raise.
            _payload = dict(_result.type_payload or ())
            _unit_count = int(_payload.get("unit_count", 0) or 0)
            _edge_count = int(_payload.get("edge_count", 0) or 0)
            _root_count = int(_payload.get("root_count", 0) or 0)
            _parallel = _payload.get("parallel_branches", ()) or ()
            _parallel_pairs = len(_parallel)
            _validation_valid = bool(_payload.get("validation_valid", False))
            _validation_errors = _payload.get("validation_errors", ()) or ()
            _execution_graph = _payload.get("execution_graph")
            _graph_id = ""
            if _execution_graph:
                # execution_graph is a tuple-of-tuple; find ("graph_id", X).
                for _k, _v in _execution_graph:
                    if _k == "graph_id":
                        _graph_id = str(_v)
                        break

            # Stash the DAG on ctx WITHOUT overwriting implementation_plan.
            # Uses dataclasses.replace so the immutable-by-convention ctx
            # is respected; if the field doesn't exist (older ctx shape),
            # fall through silently.
            if _execution_graph is not None:
                try:
                    ctx = dataclasses.replace(
                        ctx, execution_graph=_execution_graph,
                    )
                except (TypeError, ValueError):
                    # Older ctx without execution_graph field — log and
                    # continue; the shadow telemetry still fires.
                    logger.debug(
                        "[Orchestrator] PLAN-shadow could not stash "
                        "execution_graph on ctx — field missing",
                    )

            _duration_ms = int((time.monotonic() - _t0) * 1000)
            _status = (
                _result.status.value
                if hasattr(_result.status, "value")
                else str(_result.status)
            )

            logger.info(
                "[PLAN-SHADOW] op=%s status=%s dag_units=%d edges=%d "
                "roots=%d parallel_pairs=%d validation_valid=%s "
                "graph_id=%s duration_ms=%d "
                "(observer — FSM proceeds regardless)",
                getattr(ctx, "op_id", "?"),
                _status,
                _unit_count,
                _edge_count,
                _root_count,
                _parallel_pairs,
                _validation_valid,
                _graph_id or "<none>",
                _duration_ms,
            )

            # If the DAG itself was invalid, surface at INFO so the
            # graduation-arc telemetry captures validator failures
            # alongside the shadow dispatch. Still observer-only — no
            # raise, no FSM mutation.
            if not _validation_valid and _validation_errors:
                logger.info(
                    "[PLAN-SHADOW] op=%s validation_errors=%s",
                    getattr(ctx, "op_id", "?"),
                    list(_validation_errors)[:5],
                )
        except Exception:
            # Observer contract: shadow must never break the FSM.
            logger.debug(
                "[Orchestrator] PLAN shadow dispatch skipped",
                exc_info=True,
            )

        return ctx

    def _is_cancel_requested(self, op_id: str) -> bool:
        """Check if REPL /cancel was requested for this operation."""
        _gls = getattr(self._stack, "governed_loop_service", None)
        if _gls is not None and hasattr(_gls, "is_cancel_requested"):
            return _gls.is_cancel_requested(op_id)
        return False

    def _add_session_lesson(
        self,
        lesson_type: str,
        lesson_text: str,
        op_id: str = "",
    ) -> None:
        """Append a lesson, cap the buffer, and emit a heartbeat.

        Centralises the 4+ scattered ``_session_lessons.append((...))``
        + ``if len(...) > max: rebind`` blocks and adds the SerpentFlow
        heartbeat so the operator sees "📖 applying N lessons".

        Parameters
        ----------
        lesson_type:
            ``"code"`` or ``"infra"``.
        lesson_text:
            Human-readable lesson text (will be truncated to ~200 chars
            by the heartbeat for transport safety).
        op_id:
            Originating operation — passed to SerpentFlow for block scoping.
        """
        # Slice 1: the append+cap+rebind must be one atomic unit — a concurrent
        # worker's clear/rebind interleaving here could resurrect dropped lessons
        # or exceed the cap. Lock guarantees release on exceptions.
        with self._session_lessons_lock:
            self._session_lessons.append((lesson_type, lesson_text))
            if len(self._session_lessons) > self._session_lessons_max:
                self._session_lessons = self._session_lessons[-self._session_lessons_max:]

        # Emit heartbeat to SerpentFlow / transports
        try:
            _payload = {
                "phase": "session_lessons",
                "lesson_count": len(self._session_lessons),
                "latest_lesson": lesson_text[:200],
                "lessons": list(self._session_lessons),
            }
            for _t in getattr(self._stack.comm, "_transports", []):
                try:
                    _msg = type("_Msg", (), {
                        "payload": _payload,
                        "op_id": op_id,
                        "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                    })()
                    # Transport.send() is async but we are in sync context here;
                    # schedule it without awaiting (fire-and-forget for non-critical UX).
                    import asyncio as _aio
                    try:
                        _loop = _aio.get_running_loop()
                        _loop.create_task(_t.send(_msg))
                    except RuntimeError:
                        pass  # No running loop — skip heartbeat
                except Exception:
                    pass
        except Exception:
            pass  # Heartbeat is non-critical UX

    async def _emit_route_cost_heartbeat(
        self,
        ctx: OperationContext,
        *,
        cost_usd: float,
        provider: str,
        route: str,
        cost_event: str,
    ) -> None:
        """Emit route-aware cost telemetry for dashboard transports."""
        delta = float(cost_usd or 0.0)
        if delta <= 0.0:
            return
        comm = getattr(self._stack, "comm", None)
        if comm is None:
            return
        try:
            await comm.emit_heartbeat(
                op_id=ctx.op_id,
                phase="cost",
                progress_pct=0.0,
                route=route or "unknown",
                provider=provider or "",
                cost_usd=delta,
                cost_event=cost_event,
                task_complexity=getattr(ctx, "task_complexity", "") or "",
            )
        except Exception:
            logger.debug(
                "[Orchestrator] Route cost heartbeat failed", exc_info=True,
            )

    async def _watchdog_self_heal(
        self,
        ctx: OperationContext,
        target_files: tuple,
        description: str,
        compression_target: int,
        ledger: Any,
    ) -> "OperationContext | None":
        """Funnel-inversion self-heal: give the watchdog a shed-and-CONTINUE
        chance on a DUPLICATE GOAL before the de-dup ledger hard-fails it to
        advisor_blocked (Sovereign Ledger-Watchdog Composition).

        Returns the advanced (terminal ``decomposed``) context on a successful
        self-heal, or ``None`` to fall through to the legacy advisor_blocked
        path. ``None`` is returned when:

        - the lineage has NOT stalled yet (de-dup hard-fails as before), OR
        - the lineage's cumulative self-heal hop count exceeds
          ``max_self_heal_hops()`` (the structural bound, see TERMINATION), OR
        - the deep shed could not produce a sub-goal (fail-soft), OR
        - the shed result is a FIXPOINT (its hash is already a duplicate -> the
          shed can no longer reduce the payload, so the de-dup ledger is the
          final mathematical backstop -> advisor_blocked), OR
        - the re-injection emitted zero sub-goals, OR
        - ANY exception (fail-soft -- NEVER crash a dispatch).

        TERMINATION: self-heal is capped at ``max_self_heal_hops()`` (env
        ``JARVIS_WATCHDOG_MAX_SELF_HEAL_HOPS``, default 3) re-injections per
        INVARIANT lineage; the de-dup ledger remains the final backstop.

        Why a hop counter and not the fixpoint guard alone: the multi-step
        ``_make_envelope`` re-injects the sub-goal as
        ``f"{title}\n\n{description}"``. Historically the shed sub-goal carried
        a ``title=description[:80]`` prefix, so the tier3 truncation window
        shifted ~82 chars EACH hop -- the shed text kept changing and the
        fixpoint (shed-hash duplicate) guard was not hit until
        ~``compression_target/82`` hops (~7,300 at the 600k ceiling), each a
        real re-injection -> a multi-minute storm. The bounded hop counter caps
        self-heals at a small constant REGARDLESS of the shed/envelope dynamics
        because ``_lineage = subgoal_hash(target_files, ())`` is invariant
        across re-injection. (Fix 2 additionally sets the shed ``title=""`` so
        the prefix is a constant ``"\n\n"`` and the natural fixpoint converges
        immediately -- defense in depth.) The fixpoint check still short-circuits
        the common irreducible case before the budget is even consumed.
        """
        try:
            # INVARIANT LINEAGE: stable across re-injection (files do not change
            # per re-injection; the op description does). This lets the tracker
            # accumulate consecutive stalls across distinct op_ids.
            _lineage = subgoal_hash(target_files, ())
            # An exact repeat of an already-seen GOAL is a FULL stall: there was
            # no reduction at all (ratio ~1.0). Feed parent==child so the
            # tracker registers a stall pass.
            _len = max(1, len(description))
            _verdict = get_reduction_tracker().record_pass(
                _lineage, _len, _len,
            )
            if not _verdict.stalled:
                # Not enough consecutive stalls yet -> de-dup hard-fails as
                # before (the watchdog only intervenes on a proven stall).
                return None

            # BOUNDED SELF-HEAL (the REAL termination bound): cap the cumulative
            # self-heal re-injections per INVARIANT lineage at a small env
            # constant. Because ``_lineage`` is stable across re-injection, this
            # holds REGARDLESS of the shed/envelope truncation dynamics (the
            # _make_envelope title-prefix shifts the tier3 window each hop, so
            # the fixpoint-only guard alone bounded at ~compression_target/82
            # hops -- a multi-minute storm). Over budget -> fall to the de-dup
            # final backstop (advisor_blocked).
            if get_reduction_tracker().record_self_heal_hop(_lineage) > max_self_heal_hops():
                logger.warning(
                    "[SOVEREIGN YIELD] op=%s lineage=%s self-heal hop budget "
                    "exhausted (max=%d) -> advisor_blocked backstop",
                    ctx.op_id, _lineage, max_self_heal_hops(),
                )
                return None

            _sub, _tier = shed_block_goal_to_fit(
                target_files, description, compression_target, ctx.op_id,
            )
            if _sub is None:
                return None

            # FIXPOINT GUARD (termination keystone): if the shed result's hash
            # is ALREADY a duplicate, the shed can no longer reduce the payload
            # -> the de-dup ledger is the final backstop -> advisor_blocked.
            _shed_h = subgoal_hash(_sub.target_files, _sub.description)
            if is_duplicate(_shed_h, ledger, frozenset()):
                return None

            # Self-heal: emit the ONE shed sub-goal DIRECTLY. We do NOT re-run
            # decompose_for_block -- it would re-read the full files and undo the
            # shed. The shed sub-goal carries the shed source inline.
            emit_sovereign_yield(
                ctx.op_id,
                lineage_id=_lineage,
                ratio=_verdict.ratio,
                consecutive_stalls=_verdict.consecutive_stalls,
                parent_chars=len(description),
                child_chars=len(_sub.description),
                tier=_tier,
            )
            plan = DecomposedPlan(
                parent_goal_id=ctx.op_id,
                sub_goals=(_sub,),
                dag_valid=True,
                dag_depth=1,
                topological_order=(_sub.sub_goal_id,),
                diagnostic="watchdog_self_heal_reinject",
            )
            router = getattr(
                getattr(self._stack, "governed_loop_service", None),
                "_intake_router",
                None,
            )
            report = await advance_orchestration(plan, router=router)
            if getattr(report, "emitted_count", 0) >= 1:
                # Mark the shed-hash so the next arrival of THIS irreducible
                # lineage finds it duplicate at the fixpoint guard -> bounded.
                ledger.mark(_shed_h)
                logger.warning(
                    "[SOVEREIGN YIELD] op=%s self-healed: shed-and-continue "
                    "(tier=%s) -> 1 fitting sub-goal",
                    ctx.op_id, _tier,
                )
                return ctx.advance(
                    OperationPhase.CANCELLED,
                    terminal_reason_code="decomposed",
                )
            return None
        except Exception:  # noqa: BLE001 -- fail-soft -> legacy advisor_blocked
            logger.debug(
                "[Orchestrator] watchdog self-heal fail-soft -> legacy "
                "advisor_blocked (op=%s)", ctx.op_id, exc_info=True,
            )
            return None

    async def _decompose_block_or_legacy(
        self,
        ctx: OperationContext,
        advisory: Any,
        *,
        compression_target: int | None = None,
    ) -> OperationContext:
        """B5 -- the BLOCK -> decompose -> re-inject seam.

        Sovereign Egress Interceptor Mesh (T3): ``compression_target`` (the
        egress interceptor's ``max_allowed_size``) is threaded into
        ``decompose_for_block`` so each re-injected sub-goal's estimated
        payload fits under the local egress ceiling. ``None`` (default) is
        byte-identical to the legacy BLOCK-decompose behavior.

        At the OperationAdvisor BLOCK site, attempt to decompose the GOAL
        into AST-symbol-scoped + test-first sub-goals and re-inject them via
        the dep-gated multi-step emitter (gated by the B3 adaptive governor +
        B4 de-dup ledger). On success the parent terminates ``decomposed``;
        the READY sub-goals carry the work forward.

        I1 fix: captures the OrchestrationReport and only returns
        ``decomposed`` when ``report.emitted_count >= 1``.  If
        ``advance_orchestration`` emits zero sub-goals (master OFF, router
        None, all ingests fail) we fall through to the EXACT legacy
        ``advisor_blocked`` terminal so the op is NEVER silently lost.

        Fail-soft is ABSOLUTE: chunking-off, governor-not-allowed, duplicate,
        or ANY exception falls through to the EXACT legacy termination
        (``advisor_blocked``). The op is NEVER lost.

        M2 fix: reads real queue_len / loop_blocked_ms / pressure_level via
        fail-soft accessors; falls back to 0 when the signal is unreachable.

        Mutates nothing; returns the advanced (terminal CANCELLED) context.
        """
        if chunking_enabled():
            try:
                target_files = tuple(getattr(ctx, "target_files", ()) or ())
                description = ctx.description or ""

                # B4 de-dup -- primary infinite-cycle guard.
                h = subgoal_hash(target_files, description)
                ledger = get_attempt_ledger()
                # FUNNEL INVERSION (Sovereign Ledger-Watchdog Composition):
                # the de-dup ledger previously HARD-FAILED a repeating GOAL to
                # advisor_blocked BEFORE the watchdog could run, so the
                # structural self-heal was inert live. Give the watchdog a
                # shed-and-CONTINUE chance on the egress re-chunk path (compression
                # _target set + watchdog enabled) before the legacy hard-fail. The
                # de-dup ledger REMAINS the final mathematical backstop: a stalled
                # lineage whose shed cannot reduce further (fixpoint) yields a
                # duplicate shed-hash -> self-heal returns None -> advisor_blocked.
                _dup = is_duplicate(h, ledger, frozenset())
                if (
                    _dup
                    and compression_target is not None
                    and watchdog_enabled()
                ):
                    _healed = await self._watchdog_self_heal(
                        ctx,
                        target_files,
                        description,
                        compression_target,
                        ledger,
                    )
                    if _healed is not None:
                        return _healed
                if not _dup:
                    # Recursion depth from intake evidence (fail-soft to 0).
                    depth = 0
                    try:
                        evidence_json = (
                            getattr(ctx, "intake_evidence_json", "") or ""
                        )
                        if evidence_json and "recursion_depth" in evidence_json:
                            import json as _json
                            evidence = _json.loads(evidence_json)
                            if isinstance(evidence, dict):
                                depth = int(
                                    evidence.get("recursion_depth", 0) or 0
                                )
                    except Exception:  # noqa: BLE001
                        depth = 0

                    # M2 -- real load signals, each read fail-soft (default 0).
                    # queue_len: intake priority-queue depth
                    _queue_len = 0
                    try:
                        _gls = getattr(
                            self._stack, "governed_loop_service", None,
                        )
                        _router = getattr(_gls, "_intake_router", None)
                        if _router is not None and hasattr(
                            _router, "intake_queue_depth"
                        ):
                            _queue_len = int(_router.intake_queue_depth())
                    except Exception:  # noqa: BLE001
                        pass

                    # loop_blocked_ms: latest max from loop_sink stats
                    _loop_ms = 0.0
                    try:
                        from backend.core.ouroboros.telemetry.loop_sink import (
                            get_stats as _ls_stats,
                        )
                        _ls = _ls_stats()
                        if _ls:
                            _loop_ms = max(
                                v.get("max_ms", 0.0) for v in _ls.values()
                            )
                    except Exception:  # noqa: BLE001
                        pass

                    # pressure_level: MemoryPressureGate rank
                    _pressure = 0
                    try:
                        from backend.core.ouroboros.governance.memory_pressure_gate import (
                            get_default_gate as _mpg_gate,
                            _LEVEL_RANK as _MPG_RANK,
                        )
                        _pressure = int(
                            _MPG_RANK.get(_mpg_gate().pressure(), 0)
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    budget = recursion_budget(
                        queue_len=_queue_len,
                        loop_blocked_ms=_loop_ms,
                        pressure_level=_pressure,
                        depth=depth,
                    )
                    if budget.allowed:
                        zero_cov = False
                        try:
                            zero_cov = float(
                                getattr(advisory, "test_coverage", 1.0)
                            ) <= 0.0
                        except Exception:  # noqa: BLE001
                            zero_cov = False
                        if not zero_cov:
                            zero_cov = any(
                                "coverage" in str(r).lower()
                                for r in getattr(advisory, "reasons", ())
                            )

                        goal = _BlockGoal(
                            goal_id=ctx.op_id,
                            title=description[:80],
                            description=description,
                            target_files=target_files,
                        )
                        # Fan-out budget caps the MUTATION sub-goals but must
                        # NEVER sever a dependency chain: a mutation that blocks
                        # on a test prerequisite is meaningless without it. Keep
                        # ALL prerequisites (no-dep sub-goals, e.g. the test-gen)
                        # + up to budget.max_fanout dependent sub-goals. (A bare
                        # slice would, under load with max_fanout=1, keep only
                        # the test and silently drop the actual fix.)
                        _all_subs = decompose_for_block(
                            goal, zero_coverage=zero_cov,
                            compression_target=compression_target,
                        )
                        # T3 -- Convergence Watchdog: detect stalled reduction
                        # trajectory on the egress re-chunk path and structurally
                        # shed weight instead of looping forever.
                        # Guard: only active when compression_target is set (i.e.
                        # the egress-overweight re-chunk path) and the watchdog is
                        # enabled.  Fail-soft: any exception falls through to the
                        # legacy slice path unchanged; the de-dup ledger remains
                        # the backstop against infinite cycles.
                        if compression_target is not None and watchdog_enabled():
                            try:
                                _parent_chars = estimate_subgoal_payload_chars(goal)
                                _max_child = max(
                                    (estimate_subgoal_payload_chars(s) for s in _all_subs),
                                    default=0,
                                )
                                # INVARIANT LINEAGE (Ledger-Watchdog
                                # Composition): the lineage id MUST be stable
                                # across re-injections so the tracker can
                                # accumulate consecutive stalls. ctx.op_id is
                                # per-op (each re-injection is a NEW op_id) so
                                # "2 consecutive stalls" was never reached --
                                # the watchdog was inert live. subgoal_hash over
                                # (target_files, ()) is invariant across the same
                                # GOAL's re-injections (the op description
                                # changes per re-injection but the files do not).
                                _lineage = subgoal_hash(target_files, ())
                                _verdict = get_reduction_tracker().record_pass(
                                    _lineage, _parent_chars, _max_child,
                                )
                                if _verdict.stalled:
                                    # Stalled: deep-payload structural shed.
                                    # Shedding the description alone is useless --
                                    # the payload is dominated by the scoped-symbol
                                    # SOURCE segments. shed_block_goal_to_fit reads
                                    # the real file source, sheds it, and inlines
                                    # the shed result with scoped_symbols CLEARED
                                    # so the next egress estimate measures
                                    # <= target. Replace the non-shrinking slice
                                    # with that ONE fitting sub-goal so the next
                                    # pass makes progress.
                                    _shed_sub, _tier = shed_block_goal_to_fit(
                                        target_files,
                                        goal.description,
                                        compression_target,
                                        ctx.op_id,
                                    )
                                    emit_sovereign_yield(
                                        ctx.op_id,
                                        lineage_id=_lineage,
                                        ratio=_verdict.ratio,
                                        consecutive_stalls=_verdict.consecutive_stalls,
                                        parent_chars=_parent_chars,
                                        child_chars=_max_child,
                                        tier=_tier,
                                    )
                                    if _shed_sub is not None:
                                        _all_subs = (_shed_sub,)
                            except Exception:  # noqa: BLE001
                                # Fail-soft: watchdog error -> legacy slice path.
                                pass
                        _prereq = [
                            s for s in _all_subs if not s.depends_on_sub_ids
                        ]
                        _dependent = [
                            s for s in _all_subs if s.depends_on_sub_ids
                        ]
                        subs = _prereq + _dependent[: max(1, budget.max_fanout)]
                        plan = DecomposedPlan(
                            parent_goal_id=ctx.op_id,
                            sub_goals=tuple(subs),
                            dag_valid=True,
                            dag_depth=depth + 1,
                            topological_order=tuple(
                                s.sub_goal_id for s in subs
                            ),
                            diagnostic="block_decompose_reinject",
                        )
                        _gls_ref = getattr(
                            self._stack, "governed_loop_service", None,
                        )
                        router = getattr(_gls_ref, "_intake_router", None)
                        # I1 -- capture the report and check FORWARD PROGRESS.
                        # Sovereign State-Propagation Bridge: gate on
                        # ``made_forward_progress`` (emitted_count >= 1 OR a
                        # sub-goal dispatched THIS tick), NOT the lagging
                        # ``emitted_count`` aggregate alone. ``emitted_count`` is
                        # computed over the pre-emit completion_status ledger, so
                        # it STRUCTURALLY reads 0 on a fresh decompose even when
                        # router.ingest succeeded (emitted_this_tick >= 1) --
                        # gating on it false-negatived every first emit and
                        # wrongly DLQ'd a dispatched GOAL.
                        report = await advance_orchestration(plan, router=router)
                        if report.made_forward_progress:
                            ledger.mark(h)
                            logger.warning(
                                "[Orchestrator] BLOCK decomposed into %d "
                                "sub-goals (test_first=%s, emitted=%d "
                                "dispatched_this_tick=%d) op=%s",
                                len(subs), zero_cov, report.emitted_count,
                                report.emitted_this_tick, ctx.op_id,
                            )
                            return ctx.advance(
                                OperationPhase.CANCELLED,
                                terminal_reason_code="decomposed",
                            )
                        else:
                            # advance_orchestration master OFF / router None /
                            # all ingests failed -- the op MUST NOT be silently
                            # lost. Fall through to legacy advisor_blocked.
                            logger.critical(
                                "[Chunking] decompose emitted 0 sub-goals "
                                "-- falling back to advisor_blocked (op=%s). "
                                "Diagnostic: %s",
                                ctx.op_id,
                                getattr(report, "diagnostic", "unknown"),
                            )
                            # I1 optional DLQ append -- best-effort (sync, fire-and-forget).
                            try:
                                from backend.core.ouroboros.governance.intake_dlq import (
                                    append_dlq as _dlq_append,
                                )
                                # Build a minimal envelope-like dict for the DLQ.
                                _dlq_envelope = {
                                    "goal_id": ctx.op_id,
                                    "reason": "decompose_emitted_zero",
                                    "description": description[:200],
                                    "depth": depth,
                                    "diagnostic": getattr(
                                        report, "diagnostic", ""
                                    ),
                                }
                                _dlq_append(
                                    _dlq_envelope,
                                    reason="decompose_emitted_zero",
                                )
                            except Exception:  # noqa: BLE001
                                pass
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[Orchestrator] chunking seam fail-soft -> legacy "
                    "advisor_blocked (op=%s)", ctx.op_id, exc_info=True,
                )

        # Legacy fall-through -- EXACT today's behavior. The op is never lost.
        return ctx.advance(
            OperationPhase.CANCELLED,
            terminal_reason_code="advisor_blocked",
        )

    async def _handle_l2_pivot(
        self,
        ctx: "OperationContext",
        signature_hash: str,
        stderr_tail: str,
    ) -> "OperationContext":
        """T3 — Graceful Semantic Pivot handler for an ``l2_pivot`` directive.

        The L2 repair engine declared THIS sub-goal's path UNRESOLVABLE (the
        same ``failure_signature_hash`` persisted after the temperature hit
        the floor — verdict from the real ``pivot_verdict``). Instead of
        dead-stopping, we yield gracefully:

          1. Emit a ``[SOVEREIGN YIELD: UNRESOLVABLE PATH]`` telemetry line
             (reusing :func:`emit_sovereign_yield`).
          2. ``decompose_for_block`` the op into AST-symbol-scoped sub-goals
             with a ``failure_hint`` so the scoper biases the failure locus
             (the symbol implicated in the stderr tail) FIRST, and re-inject
             them via the SAME ``advance_orchestration`` seam the
             BLOCK-decompose path uses. On forward progress the parent
             terminates ``decomposed`` (the pivot is PROGRESS, not a cancel).
          3. If decompose yields no further split (already atomic) → flag for
             human review via :func:`append_dlq` and soft-terminate.

        DAG-preserving: only THIS op pivots; sibling ops are NEVER touched.

        Fail-soft ABSOLUTE: ANY error → the EXACT legacy L2 escape terminal
        (CANCELLED with ``l2_unresolvable_pivot_failed``). The op is NEVER
        lost — same I1 guarantee as the BLOCK-decompose seam.
        """
        _entry_phase = ctx.phase
        _escape_terminal = self._l2_escape_terminal(_entry_phase)
        try:
            target_files = tuple(getattr(ctx, "target_files", ()) or ())
            description = ctx.description or ""

            # 1. YIELD telemetry — reuse the sovereign-yield emitter with the
            #    UNRESOLVABLE_PATH reason so the log carries the [SOVEREIGN
            #    YIELD: UNRESOLVABLE PATH] label.
            try:
                emit_sovereign_yield(
                    ctx.op_id,
                    lineage_id=(signature_hash or "")[:16],
                    ratio=0.0,
                    consecutive_stalls=0,
                    parent_chars=len(description),
                    child_chars=0,
                    tier="epistemic_pivot",
                    reason="UNRESOLVABLE_PATH",
                )
            except Exception:  # noqa: BLE001 — telemetry is advisory
                logger.warning(
                    "[SOVEREIGN YIELD: UNRESOLVABLE PATH] op=%s sig=%s "
                    "(emitter fail-soft)",
                    ctx.op_id, (signature_hash or "")[:12],
                )

            # 2. Decompose-further AT the failure locus. failure_hint biases
            #    the scoper to split the implicated symbol first.
            goal = _BlockGoal(
                goal_id=ctx.op_id,
                title=description[:80],
                description=description,
                target_files=target_files,
            )
            _failure_hint = {
                "signature_hash": signature_hash or "",
                "stderr_tail": stderr_tail or "",
            }
            sub_goals = decompose_for_block(
                goal,
                zero_coverage=False,
                failure_hint=_failure_hint,
            )

            # An ATOMIC op cannot be split further: decompose returns a single
            # whole-op fallback sub-goal whose id mirrors the parent and which
            # carries no narrower scope. Treat "no genuine further split" as
            # atomic → HITL DLQ.
            _is_atomic = self._pivot_is_atomic(sub_goals, ctx.op_id)

            if not _is_atomic and sub_goals:
                # Re-inject via the SAME multi-step seam the BLOCK-decompose
                # path uses. NO parallel re-inject path.
                plan = DecomposedPlan(
                    parent_goal_id=ctx.op_id,
                    sub_goals=tuple(sub_goals),
                    dag_valid=True,
                    dag_depth=1,
                    topological_order=tuple(
                        s.sub_goal_id for s in sub_goals
                    ),
                    diagnostic="l2_pivot_decompose_reinject",
                )
                _gls_ref = getattr(
                    self._stack, "governed_loop_service", None,
                )
                router = getattr(_gls_ref, "_intake_router", None)
                report = await advance_orchestration(plan, router=router)
                if report.made_forward_progress:
                    logger.warning(
                        "[Orchestrator] L2_PIVOT decomposed-further into %d "
                        "sub-goals (emitted=%d dispatched_this_tick=%d) "
                        "op=%s sig=%s — DAG-preserving pivot",
                        len(sub_goals), report.emitted_count,
                        report.emitted_this_tick, ctx.op_id,
                        (signature_hash or "")[:12],
                    )
                    return ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="decomposed",
                    )
                # Re-inject emitted zero — fall through to DLQ (op never lost).
                logger.critical(
                    "[Orchestrator] L2_PIVOT decompose emitted 0 sub-goals "
                    "op=%s — routing to DLQ (op never lost)", ctx.op_id,
                )

            # 3. Atomic OR re-inject made no progress → HITL DLQ.
            try:
                from backend.core.ouroboros.governance.intake_dlq import (
                    append_dlq as _dlq_append,
                )
                _dlq_envelope = {
                    "goal_id": ctx.op_id,
                    "reason": "l2_unresolvable_awaiting_human",
                    "description": description[:200],
                    "failure_signature_hash": signature_hash or "",
                    "stderr_tail": (stderr_tail or "")[:500],
                }
                _dlq_append(
                    _dlq_envelope,
                    reason="l2_unresolvable_awaiting_human",
                )
            except Exception:  # noqa: BLE001 — DLQ is best-effort
                logger.debug(
                    "[Orchestrator] L2_PIVOT DLQ append fail-soft op=%s",
                    ctx.op_id, exc_info=True,
                )
            ctx = ctx.advance(
                _escape_terminal,
                terminal_reason_code="l2_unresolvable_awaiting_human",
            )
            await self._record_ledger(ctx, OperationState.FAILED, {
                "reason": "l2_unresolvable_awaiting_human",
                "entry_phase": _entry_phase.name,
                "terminal": _escape_terminal.name,
                "failure_signature_hash": signature_hash or "",
            })
            return ctx

        except Exception as exc:  # noqa: BLE001 — fail-soft ABSOLUTE
            # ANY pivot error → exact legacy L2 escape; the op is NEVER lost.
            logger.debug(
                "[Orchestrator] L2_PIVOT handler fail-soft -> legacy escape "
                "op=%s err=%r", ctx.op_id, exc, exc_info=True,
            )
            ctx = ctx.advance(
                _escape_terminal,
                terminal_reason_code="l2_unresolvable_pivot_failed",
            )
            try:
                await self._record_ledger(ctx, OperationState.FAILED, {
                    "reason": "l2_unresolvable_pivot_failed",
                    "entry_phase": _entry_phase.name,
                })
            except Exception:  # noqa: BLE001
                pass
            return ctx

    @staticmethod
    def _pivot_is_atomic(sub_goals: tuple, parent_op_id: str) -> bool:
        """Return True iff ``decompose_for_block`` produced NO genuine further
        split — i.e. the op is already atomic and must route to HITL DLQ.

        Atomic markers (any one ⇒ atomic):
          • empty result,
          • a single sub-goal that mirrors the parent (whole-op fallback:
            its ``sub_goal_id`` starts with the parent op id and it carries
            no ``scoped_symbols``).
        A genuine further split (≥1 sub-goal carrying ``scoped_symbols``, or
        ≥2 sub-goals) is NOT atomic. Fail-soft → True (safer to DLQ for human
        review than to re-inject a non-split).
        """
        try:
            if not sub_goals:
                return True
            if len(sub_goals) >= 2:
                return False
            only = sub_goals[0]
            _scoped = tuple(getattr(only, "scoped_symbols", ()) or ())
            if _scoped:
                return False
            # Single, unscoped, whole-op fallback → atomic.
            return True
        except Exception:  # noqa: BLE001
            return True

    async def run(self, ctx: OperationContext) -> OperationContext:
        """Execute the full governed pipeline, returning the terminal context.

        Top-level try/except catches ALL unhandled exceptions and transitions
        to POSTMORTEM.  Every code path ends in a terminal phase (COMPLETE,
        CANCELLED, EXPIRED, or POSTMORTEM).

        Parameters
        ----------
        ctx:
            The initial OperationContext in CLASSIFY phase.

        Returns
        -------
        OperationContext
            The terminal context after pipeline completion or failure.
        """
        # Phase 9.5 Part B — Phase 8 producer wiring (op-level).
        # Each call NEVER raises and gates on its own substrate master
        # flag (default false). Cost is microseconds when off — the
        # imports are lazy inside the producer module.
        _phase8_op_t0 = time.monotonic()
        _phase8_terminal_ctx = ctx
        try:
            from backend.core.ouroboros.governance.observability.phase8_producers import (
                check_flag_changes_and_publish as _phase8_flag_scan,
            )
            _phase8_flag_scan()
        except Exception:
            logger.debug(
                "[Phase8Wiring] op-start flag scan failed", exc_info=True,
            )
        try:
            try:
                _phase8_terminal_ctx = await self._run_pipeline(ctx)
                return _phase8_terminal_ctx
            except Exception as exc:
                logger.error(
                    "Unhandled exception in pipeline for %s: %s",
                    ctx.op_id,
                    exc,
                    exc_info=True,
                )
                # Try to advance to POSTMORTEM from current phase.
                # If we can't (e.g. already terminal), just return ctx.
                try:
                    ctx = ctx.advance(
                        OperationPhase.POSTMORTEM,
                        terminal_reason_code="unhandled_pipeline_exception",
                    )
                except ValueError:
                    # POSTMORTEM not legal from this phase — fall back to CANCELLED
                    # (legal from all non-terminal phases except VERIFY).
                    try:
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="unhandled_pipeline_exception",
                        )
                    except ValueError:
                        pass  # Already terminal — safe to return as-is
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {"error": str(exc), "phase": ctx.phase.name},
                )
                _phase8_terminal_ctx = ctx
                return ctx
        finally:
            # Phase 9.5 Part B — terminal-phase Phase 8 producer hooks.
            # NEVER raises. Records (a) op-level latency for the
            # terminal phase, (b) one decision-trace row tagged
            # OP_TERMINAL with the final phase + reason. Substrate
            # master flags (default false) gate the writes; calls are
            # microseconds when off.
            try:
                from backend.core.ouroboros.governance.observability.phase8_producers import (
                    record_decision_async as _phase8_record_decision_async,
                    record_phase_latency as _phase8_record_latency,
                )
                _phase8_elapsed_s = max(
                    0.0, time.monotonic() - _phase8_op_t0,
                )
                _phase8_final_ctx = _phase8_terminal_ctx
                _phase8_final_phase_name = (
                    _phase8_final_ctx.phase.name
                    if hasattr(_phase8_final_ctx, "phase") else "UNKNOWN"
                )
                _phase8_record_latency(
                    "OP_TERMINAL", _phase8_elapsed_s,
                )
                await _phase8_record_decision_async(
                    op_id=getattr(_phase8_final_ctx, "op_id", ""),
                    phase="OP_TERMINAL",
                    decision=_phase8_final_phase_name,
                    factors={
                        "terminal_reason": (
                            getattr(
                                _phase8_final_ctx,
                                "terminal_reason_code", "",
                            ) or ""
                        ),
                        "elapsed_s": round(_phase8_elapsed_s, 3),
                    },
                    rationale="op terminal",
                )
            except Exception:
                logger.debug(
                    "[Phase8Wiring] terminal hooks failed", exc_info=True,
                )
            # ── Trajectory recorder — the verdict half of the pair ───
            # Separate try/except from the Phase 8 hooks above: those
            # are gated on their own master flags, and a failure there
            # must not silently swallow this emit (or the reverse).
            # The recorder joins this to the candidates by op_id.
            # Non-blocking, master-gated default-OFF, NEVER raises.
            try:
                from backend.core.ouroboros.governance.observability.trajectory_recorder import (  # noqa: E501
                    record_outcome as _traj_record_outcome,
                )
                _traj_final_ctx = _phase8_terminal_ctx
                _traj_record_outcome(
                    op_id=str(
                        getattr(_traj_final_ctx, "op_id", "") or ""
                    ),
                    terminal_phase=(
                        _traj_final_ctx.phase.name
                        if hasattr(_traj_final_ctx, "phase") else ""
                    ),
                    terminal_reason=str(
                        getattr(
                            _traj_final_ctx, "terminal_reason_code", "",
                        ) or ""
                    ),
                )
            except Exception:
                logger.debug(
                    "[TrajectoryRecorder] terminal emit degraded",
                    exc_info=True,
                )
            # Finalize the cost-governor entry no matter how the op ended.
            # This also logs the full summary (cap, cumulative, per-provider
            # breakdown) at DEBUG for postmortem analysis.
            try:
                _cost_final = self._cost_governor.finish(ctx.op_id)
                if _cost_final is not None:
                    logger.info(
                        "[Orchestrator] Cost summary op=%s phase=%s "
                        "spent=$%.4f / cap=$%.4f (%d calls)",
                        ctx.op_id,
                        ctx.phase.name,
                        _cost_final.get("cumulative_usd", 0.0),
                        _cost_final.get("cap_usd", 0.0),
                        _cost_final.get("call_count", 0),
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] CostGovernor.finish failed", exc_info=True,
                )
            # Close the per-op TaskBoard registry entry (Gap #5 Slice 3).
            # Idempotent + safe on ops that never touched a task tool
            # (returns False cleanly). Single canonical shutdown hook
            # per the Gap #5 Slice 2 authorization. Authority-free —
            # just a scratchpad cleanup.
            try:
                from backend.core.ouroboros.governance.task_tool import (
                    close_task_board,
                )
                close_task_board(
                    ctx.op_id,
                    reason="op terminal phase=" + ctx.phase.name,
                )
            except Exception:
                logger.debug(
                    "[Orchestrator] TaskBoard close failed",
                    exc_info=True,
                )
            # Finalize the forward-progress detector entry. Safe to call
            # whether or not the op actually observed anything.
            try:
                self._forward_progress.finish(ctx.op_id)
            except Exception:
                logger.debug(
                    "[Orchestrator] ForwardProgress.finish failed", exc_info=True,
                )
            # Finalize the productivity detector entry. Logs the summary
            # (cost_since_last_change, consecutive_stable, total_cost) at
            # DEBUG for postmortem productivity analysis.
            try:
                _pd_final = self._productivity_detector.finish(ctx.op_id)
                if _pd_final is not None:
                    logger.debug(
                        "[Orchestrator] Productivity summary op=%s "
                        "stable=%d burn=$%.4f total=$%.4f tripped=%s",
                        ctx.op_id,
                        _pd_final.get("consecutive_stable", 0),
                        _pd_final.get("cost_since_last_change_usd", 0.0),
                        _pd_final.get("total_cost_usd", 0.0),
                        _pd_final.get("tripped", False),
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] ProductivityDetector.finish failed",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def _run_pipeline(self, ctx: OperationContext) -> OperationContext:
        """Internal pipeline logic -- phases 1 through 8."""

        #: Bound for real at the VALIDATE seam far below, and READ above it by
        #: the re-plan block on a later retry pass. Initialised here because
        #: the read guarded itself with `if 'validation' in dir()` — correct
        #: at runtime and invisible to every static tool, which is how two
        #: genuine undefined names in this repo's front door survived ten days
        #: inside blanket handlers. A name that may be unbound is INITIALISED,
        #: never interrogated. Not reset per retry: the re-plan block wants
        #: the PREVIOUS pass's verdict, which is exactly what persists.
        validation = None

        # A1-T4 — hop 5/5 (accept): the GOAL enters the governed FSM at
        # CLASSIFY. The fifth + final breadcrumb; the five ordered [A1Trace]
        # lines in a soak's stdout are the A1 milestone proof.
        try:
            from backend.core.ouroboros.governance.a1_trace import (  # noqa: PLC0415
                a1trace as _a1trace,
            )
            _a1trace("accept", ctx.op_id, phase="CLASSIFY")
        except Exception:  # noqa: BLE001
            pass

        # ── Ouroboros Serpent: visual indicator that the pipeline is active ──
        _serpent = None
        try:
            from backend.core.ouroboros.governance.serpent_animation import get_serpent
            _serpent = get_serpent()
            await _serpent.start("CLASSIFY")
        except Exception:
            pass

        # Wave 2 (5) Slice 6a — Dispatcher short-circuit.
        # When JARVIS_PHASE_RUNNER_DISPATCHER_ENABLED=true, the phase
        # dispatcher runs every phase through the PhaseRunnerRegistry;
        # the legacy inline blocks below are never reached. When off
        # (default), fall through to the legacy path unchanged.
        from backend.core.ouroboros.governance.phase_dispatcher import (
            dispatcher_enabled as _dispatcher_enabled,
        )
        if _dispatcher_enabled():
            from backend.core.ouroboros.governance.phase_dispatcher import (
                dispatch_pipeline as _dispatch_pipeline,
            )
            logger.info("[PhaseRunnerDelegate] DISPATCHER → pipeline op=%s", ctx.op_id[:16])
            return await _dispatch_pipeline(self, _serpent, ctx)

        # Wave 2 (5) Slice 2 - CLASSIFYRunner delegation gate.
        # Flag JARVIS_PHASE_RUNNER_CLASSIFY_EXTRACTED (default false) routes
        # the 760-line CLASSIFY block through the extracted PhaseRunner.
        # Parity tests pin identical observable output across both paths.
        # _advisory is the only local that leaks downstream (line ~2779
        # Tier 6 personality voice line reads .chronic_entropy) - we
        # thread it through result.artifacts to preserve the data flow.
        if _phase_runner_classify_extracted():
            from backend.core.ouroboros.governance.phase_runners.classify_runner import (
                CLASSIFYRunner,
            )
            logger.info("[PhaseRunnerDelegate] CLASSIFY → runner op=%s", ctx.op_id[:16])
            # Emit the CLASSIFY FSM-phase SSE so the A1 auditor witnesses the
            # phase progression (publish_fsm_phase had zero call sites). Fail-soft.
            try:
                from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501,PLC0415
                    publish_fsm_phase_for_ctx,
                )
                publish_fsm_phase_for_ctx(ctx, "CLASSIFY")
            except Exception:  # noqa: BLE001
                pass
            _classify_runner = CLASSIFYRunner(self, _serpent)
            # Task #98 (2026-05-14) — universal phase-local sub-budget
            # via shared kernel.  Wraps runner.run() in asyncio.wait_for
            # with phase budget = min(op_remaining × fraction,
            # op_remaining - reserve).  Graceful degrade on timeout or
            # insufficient-budget returns PhaseResult(status="skip",
            # reason="phase_budget_exhausted:classify:...").  Master
            # switch JARVIS_UNIVERSAL_PHASE_BUDGET_ENABLED (default true)
            # gates this wrap; legacy pass-through when off.
            from backend.core.ouroboros.governance.phase_budget import (
                dispatch_phase_with_budget,
            )
            from backend.core.ouroboros.governance.op_context import (
                OperationPhase as _OpPhase,
            )
            _classify_result = await dispatch_phase_with_budget(
                _classify_runner,
                ctx,
                phase_name="CLASSIFY",
                op_deadline=getattr(ctx, "pipeline_deadline", None),
                fallback_next_phase=_OpPhase.ROUTE,
            )
            # Rebind CLASSIFY locals that downstream phases read:
            #  - _advisory at ~line 2819 (Tier 6 personality voice)
            #  - _consciousness_bridge at ~line 3030 and ~line 4513
            #    (fragile-file memory injection, both initial + L2 retry)
            _advisory = _classify_result.artifacts.get("advisory")
            _consciousness_bridge = _classify_result.artifacts.get(
                "consciousness_bridge",
            )
            if _classify_result.next_phase is None:
                return _classify_result.next_ctx
            ctx = _classify_result.next_ctx
            # `risk_tier` is carried as a function-scoped local across
            # phases (reassigned at ~5498, 5515, 5538, 5628, 5731, 5737,
            # 5809). advance(ROUTE, risk_tier=...) stamped it on ctx,
            # so we rebind from there to keep both paths identical.
            risk_tier = ctx.risk_tier
        else:
            # ── JARVIS Tier 2: Emergency Protocol Check ──────────────────────
            # If emergency level is ORANGE or higher, block autonomous operations
            try:
                from backend.core.ouroboros.governance.emergency_protocols import (
                    EmergencyProtocolEngine, AlertLevel,
                )
                _emergency = getattr(self._stack, "_emergency_engine", None)
                if _emergency is not None and not _emergency.can_proceed():
                    state = _emergency.get_state()
                    logger.warning(
                        "[Orchestrator] Emergency level %s — operation blocked (op=%s)",
                        state.level.name, ctx.op_id,
                    )
                    if _serpent:
                        await _serpent.stop(success=False)
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code=f"emergency_{state.level.name.lower()}",
                    )
                    return ctx
            except ImportError:
                pass
            except Exception:
                pass

            # ── JARVIS Tier 1: Operation Advisor ────────────────────────────
            # "Sir, I wouldn't recommend that."
            _advisory = None
            try:
                from backend.core.ouroboros.governance.operation_advisor import (
                    OperationAdvisor,
                    AdvisoryDecision,
                    infer_read_only_intent,
                    guard_envelope_repo_root,
                    EnvelopeRepoRootRejected,
                )
                # Stamp read-only intent onto the hash-chained context BEFORE
                # advising. The Advisor's bypass of blast_radius + test_coverage
                # is mathematically safe only because ctx.is_read_only is
                # enforced downstream by tool_executor (mutating tools refused)
                # and the orchestrator's APPLY short-circuit.
                if not ctx.is_read_only:
                    _inferred_ro = infer_read_only_intent(ctx.description)
                    if _inferred_ro:
                        ctx = ctx.with_read_only_intent(True)
                        logger.info(
                            "[Orchestrator] Read-only intent inferred op=%s "
                            "— Advisor blast/coverage bypass active; tool_executor "
                            "will refuse mutations; APPLY phase will short-circuit",
                            ctx.op_id,
                        )
                _advisor = OperationAdvisor(self._config.project_root)
                # B.2.0 — worktree-aware advisory: when the envelope carries a
                # trusted ``repo_root`` string in evidence AND the master flag
                # is ON, the advisor scans THAT tree's import graph instead of
                # the orchestrator's bound project_root. Source-agnostic by
                # design — the resolver validates a path, not an envelope
                # category. Returns None when the flag is off / evidence is
                # missing / the path fails the untrusted-input safety
                # validation; advise() then falls back byte-identically.
                # B2 fail-closed: a promised-but-anchor-rejected repo_root
                # raises EnvelopeRepoRootRejected (handled below) instead
                # of silently falling back to the shared project_root tree.
                _adv_repo_root = guard_envelope_repo_root(
                    ctx.intake_evidence_json,
                    project_root=self._config.project_root,
                )
                if _adv_repo_root is not None:
                    logger.info(
                        "[Orchestrator] Advisor scanning per-envelope "
                        "repo_root=%s for op=%s "
                        "(legacy project_root retained as fallback)",
                        _adv_repo_root, ctx.op_id,
                    )
                # Dispatch through the dedicated advisor-blast executor
                # (PR-B 2026-05-13) — NOT the default asyncio
                # ThreadPoolExecutor.  In the live harness the default
                # executor is contested by 16 sensors + Oracle + etc.;
                # advisor work would queue behind them and miss the
                # BG-pool 360s ceiling.  See operation_advisor
                # ``_get_advisor_blast_executor`` for the isolation
                # contract.
                _advisory = await _advisor.advise_async(
                    ctx.target_files,
                    ctx.description,
                    ctx.op_id,
                    is_read_only=ctx.is_read_only,
                    repo_root=_adv_repo_root,
                    # C1: feed the intake evidence so the Advisor can extract
                    # scoped_symbols (stamped by decompose_for_block on
                    # re-injected sub-goals) and compute call-graph blast radius
                    # instead of the whole-file import-graph heuristic. Gated by
                    # JARVIS_ADVISOR_CALLGRAPH_BLAST_ENABLED (default off) — no
                    # behavior change until enabled + a scoped sub-goal flows.
                    intake_evidence_json=getattr(
                        ctx, "intake_evidence_json", ""
                    ) or "",
                )

                if _advisory.decision == AdvisoryDecision.BLOCK:
                    logger.warning(
                        "[Orchestrator] Advisor BLOCKED operation: %s (op=%s)",
                        "; ".join(_advisory.reasons), ctx.op_id,
                    )
                    if _serpent:
                        await _serpent.stop(success=False)
                    # I1 / B5 -- attempt decompose before terminal cancel.
                    # Fail-soft: falls through to advisor_blocked on any error.
                    ctx = await self._decompose_block_or_legacy(ctx, _advisory)
                    return ctx

                if _advisory.decision != AdvisoryDecision.RECOMMEND:
                    # Inject advisory into context for generation awareness
                    _adv_prompt = _advisor.format_for_prompt(_advisory)
                    if _adv_prompt:
                        _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                            strategic_memory_prompt=_existing + "\n\n" + _adv_prompt,
                            strategic_memory_digest=ctx.strategic_memory_digest,
                        )

                    # Active Forensic Inoculation: on a CAUTION/BLOCK hot-spot, stamp a
                    # non-destructive forensic ref + run a deterministic characterization baseline
                    # (import/interface smoke). If the fragile component is already broken, LOCK the
                    # gate + inject an un-bypassable structural constraint. Gated
                    # (JARVIS_FORENSIC_INOCULATION_ENABLED, default OFF) + fail-soft.
                    try:
                        from backend.core.ouroboros.governance.forensic_inoculation import (
                            ForensicInoculationEngine, inoculation_enabled,
                        )
                        if inoculation_enabled():
                            _inoc_graph = None
                            try:
                                from backend.core.ouroboros.oracle import get_oracle
                                _inoc_graph = getattr(get_oracle(), "_graph", None)
                            except Exception:  # noqa: BLE001
                                _inoc_graph = None
                            _inoc = await ForensicInoculationEngine(
                                self._config.project_root, graph=_inoc_graph,
                            ).inoculate(_advisory, ctx.target_files, ctx.op_id)
                            if _inoc.triggered and _inoc.locked and _inoc.constraint_clause:
                                _existing2 = getattr(ctx, "strategic_memory_prompt", "") or ""
                                ctx = ctx.with_strategic_memory_context(
                                    strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                                    strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                                    strategic_memory_prompt=_existing2 + "\n\n" + _inoc.constraint_clause,
                                    strategic_memory_digest=ctx.strategic_memory_digest,
                                )
                                logger.warning(
                                    "[Orchestrator] Forensic inoculation LOCKED op=%s branch=%s — "
                                    "baseline failed; constraint injected", ctx.op_id,
                                    _inoc.forensic_branch,
                                )
                    except Exception:  # noqa: BLE001 — inoculation must never break the pipeline
                        logger.debug("[Orchestrator] forensic inoculation skipped", exc_info=True)

                    # Voice the warning
                    if _advisory.voice_message and self._reasoning_narrator is not None:
                        try:
                            self._reasoning_narrator.record_classify(
                                ctx.op_id, _advisory.decision.value,
                                _advisory.voice_message,
                            )
                        except Exception:
                            pass

                    logger.info(
                        "[Orchestrator] Advisor: %s (risk=%.2f) — %s",
                        _advisory.decision.value, _advisory.risk_score,
                        _advisory.reasons[0] if _advisory.reasons else "no specific reason",
                    )
            except EnvelopeRepoRootRejected as _rr_exc:
                # §1 Boundary / §6 Iron Gate: isolation was promised and
                # broken — terminate infra-FAILED, NEVER fall back to the
                # shared tree (the bt-2026-05-17-002318 contamination).
                logger.warning(
                    "[Orchestrator] FAIL-CLOSED op=%s: %s — advancing "
                    "POSTMORTEM (no shared-tree fallback)",
                    ctx.op_id, _rr_exc,
                )
                if _serpent:
                    await _serpent.stop(success=False)
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code="swebp_repo_root_rejected",
                )
                return ctx
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Advisor failed", exc_info=True)

            # ---- Phase 1: CLASSIFY ----
            profile = self._build_profile(ctx)
            classification = self._stack.risk_engine.classify(profile)
            risk_tier = classification.tier

            # ---- Complexity + Persistence classification (Assimilation Gate) ----
            _complexity_result = None
            try:
                from backend.core.ouroboros.governance.complexity_classifier import (
                    OperationComplexityClassifier,
                )
                _classifier = OperationComplexityClassifier(
                    topology=getattr(self._stack, "topology", None),
                    ledger=getattr(self._stack, "ledger", None),
                )
                _complexity_result = _classifier.classify(
                    description=ctx.description,
                    target_files=list(ctx.target_files),
                    source=getattr(ctx, "signal_source", "") or "",
                )
                # (Trace-1 probe removed: it sat on this orchestrator
                # inline CLASSIFY block, which is DEAD under the phase
                # dispatcher — production runs CLASSIFYRunner. The root
                # fix is CLASSIFY parity in classify_runner.py, not a
                # diagnostic here. soak bt-2026-05-18-010430.)
                # Stamp complexity on context for downstream routing decisions.
                # task_complexity is a declared field on OperationContext, so
                # object.__setattr__ values survive dataclasses.replace() in
                # advance() and all with_*() methods.
                #
                # NO-DOWNGRADE (SWE-bench op-isolation fix #2b): intake
                # stamps task_complexity="complex" for
                # _COMPLEX_FLOOR_SOURCES BEFORE route/budget. The
                # coalesce/BG-pool ctx path can leave signal_source
                # unreadable at THIS site (exactly how the floor was
                # silently lost in soak bt-2026-05-17-213727), so the
                # classifier here could otherwise clobber the pre-stamp
                # back to "simple". Take the STRONGER of (pre-stamped,
                # freshly-classified) — a general robustness property
                # (classification must never downgrade an already-higher
                # complexity), not a swe_bench special case.
                _CX_RANK = {
                    "trivial": 0, "simple": 1, "light": 1, "moderate": 2,
                    "heavy_code": 3, "complex": 4, "architectural": 5,
                }
                _prev_cx = (getattr(ctx, "task_complexity", "") or "").lower()
                _new_cx = _complexity_result.complexity.value
                _eff_cx = (
                    _prev_cx
                    if _CX_RANK.get(_prev_cx, -1) > _CX_RANK.get(_new_cx, -1)
                    else _new_cx
                )
                object.__setattr__(ctx, "task_complexity", _eff_cx)

                logger.info(
                    "[Orchestrator] \U0001f4ca Complexity: %s, Persistence: %s, auto_approve=%s, fast_path=%s [%s]",
                    _complexity_result.complexity.value,
                    _complexity_result.persistence.value,
                    _complexity_result.auto_approve_eligible,
                    _complexity_result.fast_path_eligible,
                    ctx.op_id,
                )
            except Exception:
                logger.debug("[Orchestrator] ComplexityClassifier not available", exc_info=True)

            # ---- Consciousness regression detection (ProphecyEngine + MemoryEngine) ----
            _consciousness_bridge = getattr(self._stack, "consciousness_bridge", None)
            if _consciousness_bridge is None:
                # Check if GLS has the bridge (wired by Zone 6.12)
                _gls = getattr(self._stack, "governed_loop_service", None)
                if _gls is not None:
                    _consciousness_bridge = getattr(_gls, "_consciousness_bridge", None)
            if _consciousness_bridge is not None:
                try:
                    _regression = await _consciousness_bridge.assess_regression_risk(
                        list(ctx.target_files)
                    )
                    if _regression and _regression.get("risk_level") in ("high", "critical"):
                        logger.warning(
                            "[Orchestrator] Consciousness regression alert: %s risk for %s — %s [%s]",
                            _regression["risk_level"],
                            ctx.target_files,
                            _regression.get("reasoning", ""),
                            ctx.op_id,
                        )
                except Exception:
                    logger.debug("[Orchestrator] Consciousness regression check failed", exc_info=True)

            # ---- Goal Memory injection (cross-session learning via ChromaDB) ----
            _goal_memory_bridge = None
            _gls_for_gmb = getattr(self._stack, "governed_loop_service", None)
            if _gls_for_gmb is not None:
                _goal_memory_bridge = getattr(_gls_for_gmb, "_goal_memory_bridge", None)
            if _goal_memory_bridge is not None:
                try:
                    _goal_ctx = await _goal_memory_bridge.get_relevant_context(
                        description=ctx.description,
                        target_files=ctx.target_files,
                    )
                    if _goal_ctx:
                        _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                            strategic_memory_prompt=_existing + "\n\n" + _goal_ctx,
                            strategic_memory_digest=ctx.strategic_memory_digest,
                        )
                except Exception:
                    logger.debug("[Orchestrator] Goal memory injection failed", exc_info=True)

            # ---- Strategic Direction injection (Manifesto + architecture docs) ----
            # Slice 72 Phase 3 — withhold host-framework strategic context from
            # benchmark ops (it biased the model toward host paths like
            # backend/core/...). INERT for every non-swe_bench op.
            _strategic_svc = None
            if _gls_for_gmb is not None:
                _strategic_svc = getattr(_gls_for_gmb, "_strategic_direction", None)
            if (
                _strategic_svc is not None
                and getattr(_strategic_svc, "is_loaded", False)
                and not _should_insulate_prompt(getattr(ctx, "signal_source", ""))
            ):
                try:
                    _strat_prompt = await _strategic_svc.format_for_prompt(
                        op_id=getattr(ctx, "op_id", None),
                    )
                    if _strat_prompt:
                        _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id="manifesto-v4",
                            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                            strategic_memory_prompt=_strat_prompt + "\n\n" + _existing,
                            strategic_memory_digest=(
                                ctx.strategic_memory_digest
                                or _strategic_svc.digest[:500]
                            ),
                        )
                        # INFO (not DEBUG): graduation greps + §7
                        # observability must not depend on DEBUG
                        # capture. One line per op that actually
                        # injects — low volume, structural proof the
                        # StrategicDirection path executed.
                        logger.info(
                            "[Orchestrator] Strategic direction injected "
                            "op=%s principles=%d chars=%d",
                            getattr(ctx, "op_id", None) or "",
                            len(_strategic_svc.principles),
                            len(_strat_prompt),
                        )
                except Exception:
                    logger.debug("[Orchestrator] Strategic direction injection failed", exc_info=True)

            # ---- ConversationBridge (v0.1): TUI dialogue as untrusted soft bias ----
            # Injects the user's recent TUI turns BETWEEN the trusted manifesto
            # block (above) and the trusted goals + user-preferences blocks
            # (below). Untrusted-in-the-middle ordering preserves attention-
            # mechanism dominance for FORBIDDEN_PATH / style prefs (which come
            # last) while still surfacing conversational intent to the model.
            #
            # Authority invariant (plan v0.1 §9): this block has zero authority
            # over Iron Gate, UrgencyRouter, risk tier, policy engine,
            # FORBIDDEN_PATH, tool protected-path checks, or approval gating.
            # Consumed ONLY by StrategicDirection at this injection site.
            try:
                from backend.core.ouroboros.governance.conversation_bridge import (
                    get_default_bridge,
                )
                _bridge = get_default_bridge()
                (
                    _bridge_enabled,
                    _n_turns,
                    _n_user,
                    _n_assistant,
                    _n_postmortem,
                    _chars_in,
                    _redacted,
                    _hash8,
                ) = _bridge.inject_metrics()
                if _bridge_enabled:
                    _conv_prompt = _bridge.format_for_prompt()
                    if _conv_prompt:
                        _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id=ctx.strategic_intent_id or "conv-bridge-v1",
                            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                            strategic_memory_prompt=(
                                _existing + "\n\n" + _conv_prompt
                                if _existing else _conv_prompt
                            ),
                            strategic_memory_digest=ctx.strategic_memory_digest,
                        )
                    # §8 one-line observability contract (v1.1 source breakdown).
                    # Logged whether or not there were turns to inject —
                    # operators need to see that the wiring fired.
                    logger.info(
                        "[ConversationBridge] op=%s enabled=true n_turns=%d "
                        "n_user=%d n_assistant=%d n_postmortem=%d chars_in=%d "
                        "inject_site=context_expansion redacted=%s hash8=%s",
                        ctx.op_id, _n_turns, _n_user, _n_assistant, _n_postmortem,
                        _chars_in, _redacted, _hash8,
                    )
                else:
                    # §8 §7-tweak: DEBUG line at inject site when master switch
                    # is off so "is wiring live?" is answerable without content.
                    logger.debug(
                        "[ConversationBridge] op=%s enabled=false "
                        "inject_site=context_expansion",
                        ctx.op_id,
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] ConversationBridge injection skipped",
                    exc_info=True,
                )

            # ---- P0 PostmortemRecall (PRD Phase 1): prior-op lessons ----
            # Helper extraction mirrors LSS pattern (testability per PRD §11
            # Layer 3 reachability supplement, W3(6) precedent). Body lives at
            # module scope as `_inject_postmortem_recall_impl`. ConversationBridge
            # → PostmortemRecall → SemanticIndex ordering preserved.
            ctx = await _inject_postmortem_recall_impl(ctx)

            # ---- Task 6 Prior Ephemeral Knowledge: cross-session experiences ----
            # Helper extraction mirrors LSS/PostmortemRecall pattern. Body lives
            # at module scope as `_inject_prior_knowledge_impl`. Trust ordering:
            # Strategic → ConversationBridge → PostmortemRecall → PriorKnowledge
            # → SemanticIndex → Goals → UserPreferences.
            ctx = _inject_prior_knowledge_impl(ctx)

            # ---- Phase 4 P3 Cognitive Metrics: Oracle pre-score ----
            # Best-effort observability — calls OraclePreScorer via the
            # CognitiveMetricsService singleton wired at orchestrator boot.
            # Persists a CognitiveMetricRecord to the JSONL ledger when
            # JARVIS_COGNITIVE_METRICS_ENABLED is on. Advisory only —
            # the existing Iron Gate / risk_tier_floor stack remains
            # authoritative. Helper body at module scope as
            # `_score_cognitive_metrics_pre_apply_impl`.
            _score_cognitive_metrics_pre_apply_impl(ctx)

            # ---- SemanticIndex v0.1: recency-weighted focus + closures ----
            # Soft semantic prior drawn from the recency-weighted centroid
            # over recent commits + active goals + recent conversation.
            # Injected BETWEEN the ConversationBridge block (above) and the
            # Goals block (below) so the ordering reads top-to-bottom as:
            # Strategic → Bridge (untrusted dialogue) → Semantic (untrusted
            # prior) → Goals (trusted) → UserPreferences (highest trust).
            #
            # Authority invariant: this block has **zero** authority over
            # Iron Gate, UrgencyRouter, risk tier, policy engine, FORBIDDEN_PATH,
            # or approval gating. It affects ONLY the prompt surface the model
            # reads at CONTEXT_EXPANSION — §4 (data sovereignty, local
            # embedder) + §8 (hashes + counts, no raw vectors in logs).
            try:
                from backend.core.ouroboros.governance.semantic_index import (
                    get_default_index,
                )
                _semi = get_default_index(self._config.project_root)
                # Q3 Slice 3 — non-blocking build trigger so CLASSIFY
                # never stalls on subprocess+embed. format_prompt_sections
                # operates against the currently-loaded centroid (empty
                # on cold start → returns None, callers handle that).
                _semi.build_async()
                _semi_prompt = _semi.format_prompt_sections()
                if _semi_prompt:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=ctx.strategic_intent_id or "semantic-v1",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=(
                            _existing + "\n\n" + _semi_prompt
                            if _existing else _semi_prompt
                        ),
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    _semi_stats = _semi.stats()
                    logger.info(
                        "[SemanticIndex] op=%s corpus_n=%d centroid_hash8=%s "
                        "inject_site=context_expansion prompt_chars=%d",
                        ctx.op_id, _semi_stats.corpus_n,
                        _semi_stats.centroid_hash8, len(_semi_prompt),
                    )
                else:
                    logger.debug(
                        "[SemanticIndex] op=%s no prompt section (disabled or empty)",
                        ctx.op_id,
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] SemanticIndex injection skipped",
                    exc_info=True,
                )

            # ---- TaskBoard advisory prompt injection (Gap #5 Slice 3) ----
            #
            # Read-only + authority-free. We do NOT lazily create a board
            # here — only render when the model has already touched a task
            # tool during this op (i.e. a board exists in the registry).
            # Avoids injecting an empty "Current tasks" section on every
            # op. Per authorization: NEVER gates Iron Gate / policy /
            # approval (Manifesto §1 + §6). Tier -1 sanitation inside
            # TaskBoard.render_prompt_section() handles model content
            # safety — we don't fight the sanitizer here.
            try:
                from backend.core.ouroboros.governance.task_tool import (
                    _BOARDS,
                )
                _tb = _BOARDS.get(ctx.op_id)
                if _tb is not None:
                    _tb_prompt = _tb.render_prompt_section()
                    if _tb_prompt:
                        _tb_existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id=(
                                ctx.strategic_intent_id or "task-board-v1"
                            ),
                            strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                            strategic_memory_prompt=(
                                _tb_existing + "\n\n" + _tb_prompt
                                if _tb_existing else _tb_prompt
                            ),
                            strategic_memory_digest=ctx.strategic_memory_digest,
                        )
                        logger.info(
                            "[TaskBoard] op=%s inject_site=context_expansion "
                            "prompt_chars=%d",
                            ctx.op_id, len(_tb_prompt),
                        )
            except Exception:
                logger.debug(
                    "[Orchestrator] TaskBoard injection skipped", exc_info=True,
                )

            # ---- TDD directive (Feature 1 V1 — prompt contract, NOT red-green) ----
            #
            # When the intent envelope carries evidence["tdd_mode"]=True,
            # prepend a prompt directive instructing the model to emit
            # tests + impl together (test file first in files: [...]).
            # Honest scope: this is a prompt contract, not a red-green
            # proof. True test-first orchestration (run tests → confirm
            # fail → generate impl → run tests → confirm pass) is a
            # separate multi-commit project scoped for V1.1. The V1
            # module ships the declarative layer so ops can be marked
            # TDD now; V1.1 flips the flag from "prompt hint" to
            # "pipeline sub-phase trigger" without client-side changes.
            try:
                from backend.core.ouroboros.governance.tdd_directive import (
                    is_tdd_op,
                    tdd_prompt_directive,
                )
                if is_tdd_op(ctx):
                    _tdd_text = tdd_prompt_directive()
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=ctx.strategic_intent_id or "tdd-v1",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=(
                            _existing + "\n\n" + _tdd_text
                            if _existing else _tdd_text
                        ),
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[TDDDirective] op=%s tdd_mode=true directive_chars=%d "
                        "scope=prompt_contract_not_red_green",
                        ctx.op_id, len(_tdd_text),
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] TDD directive injection skipped",
                    exc_info=True,
                )

            # ---- Goal inference — hypothesized direction from multi-signal cross-corr ----
            #
            # Closes the "read the room" gap: watch commits, REPL inputs,
            # memory, completed ops, file hotspots, and declared goals;
            # synthesize ranked hypotheses about where the operator is
            # headed. Injected as a clearly-labeled "Inferred Direction
            # (hypotheses — not declared goals)" section so the model
            # weights it BELOW explicit goals. Default OFF, fail-closed.
            #
            # Authority invariant: hypotheses inform prompt surface only.
            # They NEVER affect risk tier, route, guardian findings, gate
            # verdicts, or approval. Operator accepts/rejects via /infer.
            try:
                from backend.core.ouroboros.governance.goal_inference import (
                    GoalInferenceEngine,
                    get_default_engine,
                    inference_enabled,
                    render_prompt_section,
                )
                if inference_enabled():
                    _engine = get_default_engine(self._config.project_root)
                    if _engine is None:
                        _engine = GoalInferenceEngine(
                            repo_root=self._config.project_root,
                        )
                    # Slice 148/149 — the heavy GoalInferenceEngine.build (→
                    # SemanticIndex.build, fastembed inference) is synchronous and
                    # stalls the event loop ~8s (LoopSink). Use the single off-loop
                    # entry point so both boot call sites share one pattern.
                    _inf_result = await _engine.build_offloaded()
                    _inf_text = render_prompt_section(_inf_result)
                    if _inf_text:
                        _existing = getattr(
                            ctx, "strategic_memory_prompt", "",
                        ) or ""
                        ctx = ctx.with_strategic_memory_context(
                            strategic_intent_id=(
                                ctx.strategic_intent_id or "goal-inference-v1"
                            ),
                            strategic_memory_fact_ids=(
                                ctx.strategic_memory_fact_ids
                            ),
                            strategic_memory_prompt=(
                                _existing + "\n\n" + _inf_text
                                if _existing else _inf_text
                            ),
                            strategic_memory_digest=(
                                ctx.strategic_memory_digest
                            ),
                        )
                        logger.info(
                            "[GoalInference] op=%s injected hypotheses=%d "
                            "top_conf=%.2f chars=%d",
                            ctx.op_id,
                            min(
                                len(_inf_result.inferred),
                                # top_k applied inside render
                                5,
                            ),
                            (_inf_result.inferred[0].confidence
                             if _inf_result.inferred else 0.0),
                            len(_inf_text),
                        )
            except Exception:
                logger.debug(
                    "[Orchestrator] Goal inference injection skipped",
                    exc_info=True,
                )

            # ---- LastSessionSummary v0.1: session-to-session episodic continuity ----
            # Read-only structured summary of past session(s), rendered as
            # a dense untrusted block. Injected between SemanticIndex (above)
            # and Goals (below) so the untrusted stack stays contiguous:
            # Strategic → Bridge → Semantic → LastSession → Goals → UserPrefs.
            # Helper extracted for integration-test coverage of the composed
            # CONTEXT_EXPANSION prompt (see test_last_session_summary_composition).
            ctx = await _inject_last_session_summary_impl(self._config.project_root, ctx)

            # ---- P2.4 + Week 2: Goal-directed context injection ----
            # Append the *most relevant* active user goals to the strategic
            # memory prompt so the generation model aligns its decisions with
            # current priorities. Scoped by target_files + description so a
            # noisy goal tracker doesn't hijack unrelated ops.
            #
            # Increment 3: after prompt injection, compute the full activity
            # entry set (direct matches + descendant credits + optional
            # sibling bumps) and append to the GoalActivityLedger. Every op
            # that reaches CLASSIFY writes at least one row so the session-end
            # drift aggregator sees it as "reached CLASSIFY", even when no
            # goal scored.
            try:
                from backend.core.ouroboros.governance.strategic_direction import (
                    GoalActivityLedger,
                    GoalTracker,
                    get_active_session_id,
                )
                _goal_tracker = GoalTracker(self._config.project_root)
                _goal_prompt = _goal_tracker.format_for_prompt(
                    target_files=list(ctx.target_files),
                    description=ctx.description or "",
                )
                if _goal_prompt and not _should_insulate_prompt(
                    getattr(ctx, "signal_source", "")
                ):
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=ctx.strategic_intent_id or "goals-v1",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing + "\n\n" + _goal_prompt if _existing else _goal_prompt,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.debug(
                        "[Orchestrator] Goal context injected (%d active / scoped)",
                        len(_goal_tracker.active_goals),
                    )

                # Activity ledger append (Increment 3). Ledger-only — does
                # not feed intake priority math. Zero-match ops still get a
                # marker row so the drift denominator counts them.
                _session_id = get_active_session_id() or ""
                if _session_id:
                    try:
                        _activity_entries = _goal_tracker.compute_activity_entries(
                            description=ctx.description or "",
                            target_files=list(ctx.target_files),
                        )
                        GoalActivityLedger(self._config.project_root).append(
                            session_id=_session_id,
                            op_id=ctx.op_id,
                            entries=_activity_entries,
                        )
                        logger.debug(
                            "[Orchestrator] GoalActivity ledger: wrote %d entries for op=%s",
                            len(_activity_entries) or 1,  # 1 marker row on zero-match
                            ctx.op_id,
                        )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] GoalActivity ledger append failed",
                            exc_info=True,
                        )
            except Exception:
                logger.debug("[Orchestrator] Goal injection skipped", exc_info=True)

            # ---- Task #195: User Preference Memory injection ----
            # Append typed user-preference memories (facts about the user,
            # feedback rules, forbidden paths, style choices) scoped by
            # relevance to the current op. Zero model inference — pure
            # deterministic scoring. Empty when no memory matches the op
            # shape, so silent on fresh repos.
            try:
                from backend.core.ouroboros.governance.user_preference_memory import (
                    get_default_store,
                )
                _user_prefs = get_default_store(self._config.project_root)
                _pref_prompt = _user_prefs.format_for_prompt(
                    target_files=list(ctx.target_files),
                    description=ctx.description,
                    risk_tier=str(getattr(ctx, "risk_tier", "") or ""),
                )
                if _pref_prompt:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=ctx.strategic_intent_id or "user-prefs-v1",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=(
                            _existing + "\n\n" + _pref_prompt if _existing else _pref_prompt
                        ),
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.debug(
                        "[Orchestrator] User preferences injected (%d chars)",
                        len(_pref_prompt),
                    )
            except Exception:
                logger.debug("[Orchestrator] User preference injection skipped", exc_info=True)

            # ---- Policy engine check (declarative YAML rules) ----
            # Evaluated BEFORE the risk-engine BLOCKED short-circuit so that
            # explicit deny rules in policy files can override the risk engine.
            # Wrapped in hasattr + try/except so the pipeline is never broken
            # by a missing or misconfigured policy_engine attribute.
            if hasattr(self._stack, "policy_engine") and self._stack.policy_engine is not None:
                try:
                    _policy_engine: PolicyEngine = self._stack.policy_engine
                    for _tf in ctx.target_files:
                        _policy_decision = _policy_engine.classify(tool="edit", target=str(_tf))
                        if _policy_decision is PolicyDecision.BLOCKED:
                            logger.info(
                                "[Orchestrator] PolicyEngine BLOCKED op=%s target=%r",
                                ctx.op_id, _tf,
                            )
                            risk_tier = RiskTier.BLOCKED
                            break
                except Exception:
                    logger.warning(
                        "[Orchestrator] PolicyEngine raised during CLASSIFY for op=%s; continuing",
                        ctx.op_id, exc_info=True,
                    )

            if risk_tier is RiskTier.BLOCKED:
                ctx = ctx.advance(
                    OperationPhase.CANCELLED,
                    risk_tier=risk_tier,
                    terminal_reason_code=classification.reason_code,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.BLOCKED,
                    {
                        "reason_code": classification.reason_code,
                        "risk_tier": risk_tier.name,
                    },
                )
                return ctx

            # Announce operation start — VoiceNarrator fires here (INTENT type)
            try:
                await self._stack.comm.emit_intent(
                    op_id=ctx.op_id,
                    goal=ctx.description,
                    target_files=list(ctx.target_files),
                    risk_tier=risk_tier.name,
                    blast_radius=len(ctx.target_files),
                )
            except Exception:
                logger.debug("emit_intent failed for op=%s", ctx.op_id, exc_info=True)

            # ---- Reasoning chain classification (optional, pre-routing) ----
            reasoning_result = None
            if self._reasoning_bridge and self._reasoning_bridge.is_active:
                try:
                    reasoning_result = await self._reasoning_bridge.classify_with_reasoning(
                        command=ctx.description,
                        op_id=ctx.op_id,
                    )
                except Exception:
                    logger.debug("Reasoning chain bridge error", exc_info=True)

            # P3.1: Emit intent chain heartbeat — full reasoning chain for the
            # SerpentFlow display.  Deterministic: all data already computed.
            try:
                _chain_payload: Dict[str, Any] = {
                    "phase": "intent_chain",
                    "risk_tier": risk_tier.name,
                    "complexity": (
                        _complexity_result.complexity.value
                        if _complexity_result is not None else ""
                    ),
                    "auto_approve": (
                        _complexity_result.auto_approve_eligible
                        if _complexity_result is not None else False
                    ),
                    "fast_path": (
                        _complexity_result.fast_path_eligible
                        if _complexity_result is not None else False
                    ),
                }
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="intent_chain", progress_pct=10.0,
                    **_chain_payload,
                )
            except Exception:
                pass  # Intent chain visibility is best-effort

            # Advance to ROUTE with risk_tier set (and optional reasoning result)
            if _serpent: _serpent.update_phase("ROUTE")
            ctx = ctx.advance(
                OperationPhase.ROUTE,
                risk_tier=risk_tier,
                reasoning_chain_result=reasoning_result,
            )

            # ── P0 Wiring: Start ReasoningNarrator + OperationDialogue ──────
            if self._reasoning_narrator is not None:
                try:
                    self._reasoning_narrator.start_trace(ctx.op_id)
                    self._reasoning_narrator.record_classify(
                        ctx.op_id,
                        risk_tier.value if hasattr(risk_tier, "value") else str(risk_tier),
                        f"files={list(ctx.target_files)[:3]}, "
                        f"complexity={getattr(_complexity_result, 'complexity', 'unknown')}",
                    )
                except Exception:
                    pass

            if self._dialogue_store is not None:
                try:
                    from backend.core.ouroboros.governance.entropy_calculator import extract_domain_key
                    _dk = extract_domain_key(ctx.target_files, ctx.description)
                    self._dialogue_store.start_dialogue(
                        op_id=ctx.op_id,
                        domain_key=_dk,
                        description=ctx.description,
                        target_files=ctx.target_files,
                    )
                    _dialogue = self._dialogue_store.get_active(ctx.op_id)
                    if _dialogue:
                        _dialogue.add_entry(
                            "CLASSIFY",
                            f"Risk={risk_tier}, complexity={getattr(_complexity_result, 'complexity', 'unknown')}",
                        )
                except Exception:
                    pass

            # ---- ClassifyClarify: one operator question at the CLASSIFY→ROUTE boundary ----
            #
            # Closes the "intake description is ambiguous" gap. Narrow
            # ambiguity heuristic (short desc + no target files, or generic
            # target list, or no goal-keyword match). On trigger, ask the
            # operator ONE concise question with a bounded timeout. The
            # answer enriches ctx.description + evidence only — it has NO
            # authority over risk classification, routing law, SemanticGuardian
            # findings, or any deterministic engine input (Manifesto §1
            # Boundary Principle).
            #
            # Default OFF (JARVIS_CLASSIFY_CLARIFY_ENABLED=0). Opt-in means
            # no session is interrupted until the operator explicitly
            # enables the feature + the heuristic actually fires.
            try:
                from backend.core.ouroboros.governance.classify_clarify import (
                    ask_operator as _clarify_ask,
                    merge_into_context as _clarify_merge,
                    clarify_enabled as _clarify_enabled,
                )
                if _clarify_enabled():
                    # Extract goal keywords from the active GoalTracker so
                    # the heuristic can check "no goal keyword match".
                    _goal_keywords: tuple = ()
                    try:
                        from backend.core.ouroboros.governance.strategic_direction import (
                            GoalTracker,
                        )
                        _kws: list = []
                        for _g in GoalTracker(
                            self._config.project_root,
                        ).active_goals:
                            _kws.extend(getattr(_g, "keywords", ()) or ())
                        _goal_keywords = tuple(_kws)
                    except Exception:
                        _goal_keywords = ()
                    _clarify_response = await _clarify_ask(
                        op_id=ctx.op_id,
                        description=ctx.description or "",
                        target_files=tuple(ctx.target_files or ()),
                        goal_keywords=_goal_keywords,
                    )
                    if _clarify_response.outcome == "answered":
                        # Merge the sanitized answer into the description.
                        # The risk classifier has ALREADY run above — we do
                        # not re-classify. The clarification only affects
                        # downstream prompt content (description + evidence).
                        _new_desc, _patch = _clarify_merge(
                            original_description=ctx.description or "",
                            response=_clarify_response,
                        )
                        try:
                            import dataclasses as _dc
                            ctx = _dc.replace(ctx, description=_new_desc)
                        except Exception:
                            logger.debug(
                                "[Orchestrator] ClassifyClarify ctx merge skipped",
                                exc_info=True,
                            )
            except Exception:
                logger.debug(
                    "[Orchestrator] ClassifyClarify skipped",
                    exc_info=True,
                )

        # Wave 2 (5) Slice 3 - ROUTE+CTX+PLAN PhaseRunner delegation gate.
        # Quota Shield (Phases 1+2): proactively route trivial/localized ops to the
        # zero-cost local tier (preserving DW quota), unless host memory is CRITICAL.
        # Placed AFTER the CLASSIFY if/else so it runs on BOTH the extracted-runner
        # (default/production) and legacy inline paths (both set _advisory + ctx here).
        # Gated (default OFF) + fail-soft + advisory-None-guarded.
        try:
            from backend.core.ouroboros.governance.quota_shield import apply_quota_shield
            ctx = await apply_quota_shield(ctx, advisory=_advisory)
        except Exception:
            logger.debug("[Orchestrator] quota shield skipped", exc_info=True)

        # All three flags (JARVIS_PHASE_RUNNER_{ROUTE,CONTEXT_EXPANSION,PLAN}_EXTRACTED)
        # must be set to engage the runner chain. This all-or-nothing
        # gate simplifies wiring while the three phases remain
        # interleaved (ROUTE body -> conditional CTX -> PLAN body) in the
        # inline pipeline. Per-phase independence arrives with Slice 6
        # (dispatcher cutover).
        if _phase_runner_slice3_fully_extracted():
            from backend.core.ouroboros.governance.phase_runners import (
                ContextExpansionRunner,
                PLANRunner,
                ROUTERunner,
            )
            logger.info("[PhaseRunnerDelegate] ROUTE+CTX+PLAN → runners op=%s", ctx.op_id[:16])
            # Task #98 (2026-05-14) — universal phase-local sub-budget.
            # Same shared kernel as CLASSIFY hook above.  Each phase
            # gets its own fraction of op_remaining, asyncio.wait_for
            # bounds, graceful degrade on timeout / insufficient budget.
            from backend.core.ouroboros.governance.phase_budget import (
                dispatch_phase_with_budget,
            )
            # ROUTE: runs the routing body + either advance(CTX) or advance(PLAN)
            _route_result = await dispatch_phase_with_budget(
                ROUTERunner(self, _serpent),
                ctx,
                phase_name="ROUTE",
                op_deadline=getattr(ctx, "pipeline_deadline", None),
                fallback_next_phase=OperationPhase.CONTEXT_EXPANSION,
            )
            ctx = _route_result.next_ctx
            # CTX: runs only if ROUTERunner advanced to CONTEXT_EXPANSION
            if _route_result.next_phase is OperationPhase.CONTEXT_EXPANSION:
                _ctx_result = await dispatch_phase_with_budget(
                    ContextExpansionRunner(self, _serpent),
                    ctx,
                    phase_name="CONTEXT_EXPANSION",
                    op_deadline=getattr(ctx, "pipeline_deadline", None),
                    fallback_next_phase=OperationPhase.PLAN,
                )
                ctx = _ctx_result.next_ctx
            # PLAN: advisory artifact comes from CLASSIFY's result — carried
            # via the _advisory local established by the CLASSIFY hook.
            # Note: PlanRunner.run also has Task #97's *internal* phase
            # budget (via PlanGenerator.generate_plan) — this outer wrap
            # is defense-in-depth at the runner-dispatch boundary.
            _plan_result = await dispatch_phase_with_budget(
                PLANRunner(self, _serpent, advisory=_advisory),
                ctx,
                phase_name="PLAN",
                op_deadline=getattr(ctx, "pipeline_deadline", None),
                fallback_next_phase=OperationPhase.GENERATE,
            )
            if _plan_result.next_phase is None:
                # Terminal exit from PLAN (plan_rejected, plan_expired, etc.)
                return _plan_result.next_ctx
            ctx = _plan_result.next_ctx
        else:
            # ---- Phase 2: ROUTE ----

            # Telemetry host-binding enforcement for remote routes (split-brain guard)
            _routing = getattr(ctx, "routing", None)
            if _routing is not None and str(getattr(_routing, "name", "")).upper() in ("GCP_PRIME", "REMOTE"):
                try:
                    from backend.core.ouroboros.governance.telemetry_contextualizer import (
                        TelemetryContextualizer,
                    )
                    _tc = TelemetryContextualizer()
                    _exec_host = str(getattr(_routing, "endpoint", "local"))
                    _tel_host = str(getattr(ctx, "telemetry_host", _exec_host))
                    await _tc.assert_host_binding(
                        execution_host=_exec_host,
                        telemetry_host=_tel_host,
                    )
                except RuntimeError as _bind_err:
                    logger.warning(
                        "[Orchestrator] Telemetry host-binding violation: %s [%s]",
                        _bind_err, ctx.op_id,
                    )
                except Exception:
                    logger.debug("[Orchestrator] TelemetryContextualizer not available", exc_info=True)

            # ── Urgency-aware provider routing (Manifesto §5 Tier 0) ──
            # Deterministic routing based on signal_urgency + signal_source +
            # task_complexity. Stamps provider_route on context for
            # CandidateGenerator dispatch.
            try:
                from backend.core.ouroboros.governance.urgency_router import (
                    UrgencyRouter,
                )
                _urgency_router = UrgencyRouter()
                _provider_route, _route_reason = _urgency_router.classify(ctx)
                object.__setattr__(ctx, "provider_route", _provider_route.value)
                object.__setattr__(ctx, "provider_route_reason", _route_reason)
                logger.info(
                    "[Orchestrator] \U0001f6e4\ufe0f  Route: %s (%s) [%s]",
                    _provider_route.value, _route_reason, ctx.op_id,
                )
                # Emit route decision to CommProtocol for observability
                if hasattr(self._stack, "comm") and self._stack.comm is not None:
                    try:
                        from backend.core.ouroboros.governance.urgency_router import (
                            UrgencyRouter as _UR,
                        )
                        await self._stack.comm.emit_decision(
                            op_id=ctx.op_id,
                            outcome=_provider_route.value,
                            reason_code=f"urgency_route:{_route_reason}",
                            route=_provider_route.value,
                            route_reason=_route_reason,
                            budget_profile=_UR.context_budget_profile(_provider_route, ctx),
                            details={
                                "route": _provider_route.value,
                                "route_description": _UR.describe_route(_provider_route),
                                "signal_urgency": getattr(ctx, "signal_urgency", ""),
                                "signal_source": getattr(ctx, "signal_source", ""),
                                "task_complexity": getattr(ctx, "task_complexity", ""),
                                "budget_profile": _UR.context_budget_profile(_provider_route, ctx),
                            },
                        )
                    except Exception:
                        pass
            except Exception:
                logger.debug("[Orchestrator] UrgencyRouter not available", exc_info=True)

            # ── Start per-op cost governor ──
            # Called here (post-ROUTE) so the cap is derived from the actual
            # stamped route + task_complexity. If either field is empty the
            # governor uses safe "standard/light" defaults. Safe to call even
            # when governor is disabled — returns +inf cap.
            try:
                self._cost_governor.start(
                    op_id=ctx.op_id,
                    route=getattr(ctx, "provider_route", "") or "",
                    complexity=getattr(ctx, "task_complexity", "") or "",
                    is_read_only=bool(getattr(ctx, "is_read_only", False)),
                )
            except Exception:
                logger.debug("[Orchestrator] CostGovernor.start failed", exc_info=True)

            if self._config.context_expansion_enabled:
                # ── PreActionNarrator: voice WHAT before CONTEXT_EXPANSION ──
                if self._pre_action_narrator is not None:
                    try:
                        await self._pre_action_narrator.narrate_phase(
                            "CONTEXT_EXPANSION",
                            {"target_file": list(ctx.target_files)[0] if ctx.target_files else "unknown"},
                        )
                    except Exception:
                        pass
                if _serpent: _serpent.update_phase("CONTEXT_EXPANSION")
                ctx = ctx.advance(OperationPhase.CONTEXT_EXPANSION)

                # ---- Cross-Repo Scope Promoter (ignition) ----
                # Before GENERATE, consult the unified Oracle graph: if the fault cone crosses a repo
                # boundary (jarvis→reactor/prime), elevate repo_scope to span both — igniting the
                # native Saga apply path at APPLY — and force Orange-tier. Gated + fail-soft → no-op
                # when off (single-repo pipeline byte-identical). Elevating here (pre-GENERATE) lets
                # the 2c.1 multi-repo candidate schema produce the per-repo patch_map the saga needs.
                try:
                    from backend.core.ouroboros.governance.cross_repo_scope_promoter import (
                        CrossRepoScopePromoter, promoter_enabled,
                    )
                    if promoter_enabled() and not ctx.cross_repo:
                        _promoter = CrossRepoScopePromoter()
                        ctx, _promo_report = await _promoter.maybe_promote(ctx)
                        if _promo_report is not None and _promo_report.promoted:
                            logger.info(
                                "[CrossRepoPromoter] op=%s scope elevated → %s\n%s",
                                ctx.op_id, _promo_report.elevated_scope, _promo_report.render(),
                            )
                except Exception:  # noqa: BLE001 — promoter is additive; never break the pipeline
                    logger.debug("[CrossRepoPromoter] hook skipped", exc_info=True)

                # ---- Cross-Repo Mutator G1: blast-radius context (GENERATE) ----
                # If the promoter elevated the op to cross-repo AND the master
                # arming switch is ON, trace the Oracle cross-repo dependents,
                # emit the operator-visible ASCII blast tree, and inject the
                # rendered prompt block into the generation context. Gated +
                # fail-soft → "" when off / on any error (byte-identical; the
                # promoter + immutable-Orange floor already elevate the tier).
                try:
                    from backend.core.ouroboros.governance.cross_repo_master_flag import (
                        cross_repo_mutation_enabled,
                    )
                    if cross_repo_mutation_enabled() and ctx.cross_repo:
                        from backend.core.ouroboros.governance.multi_repo.cross_repo_wiring import (
                            build_blast_context_block,
                        )
                        _blast_block = await build_blast_context_block(
                            ctx, oracle=getattr(self._stack, "oracle", None),
                        )
                        if _blast_block:
                            ctx = dataclasses.replace(
                                ctx, cross_repo_blast_prompt=_blast_block
                            )
                except Exception:  # noqa: BLE001 — additive; never break the pipeline
                    logger.debug("[CrossRepoBlast] G1 hook skipped", exc_info=True)

                # ---- Phase 2b: CONTEXT_EXPANSION ----
                try:
                    expansion_deadline = datetime.now(tz=timezone.utc) + timedelta(
                        seconds=self._config.context_expansion_timeout_s
                    )
                    from backend.core.ouroboros.governance.skill_registry import SkillRegistry as _SkillRegistry
                    _skill_registry = _SkillRegistry(self._config.project_root)
                    # DocFetcher: bounded external doc retrieval (P3 — Boundary Principle)
                    _doc_fetcher = None
                    try:
                        from backend.core.ouroboros.governance.doc_fetcher import DocFetcher
                        _doc_fetcher = DocFetcher()
                    except ImportError:
                        pass

                    # WebSearchCapability: structured search with epistemic allowlist
                    _web_search = None
                    try:
                        from backend.core.ouroboros.governance.web_search import WebSearchCapability
                        _ws = WebSearchCapability()
                        if _ws.is_available:
                            _web_search = _ws
                            logger.debug(
                                "[Orchestrator] WebSearchCapability available (backend=%s)",
                                _ws.backend_name,
                            )
                    except ImportError:
                        pass

                    # VisualCodeComprehension: screenshot-based analysis
                    _visual = None
                    try:
                        from backend.core.ouroboros.governance.visual_comprehension import (
                            VisualCodeComprehension,
                        )
                        _vc = VisualCodeComprehension()
                        if _vc.is_available:
                            _visual = _vc
                    except ImportError:
                        pass

                    # CodeExplorationTool: sandboxed hypothesis testing
                    _explorer = None
                    try:
                        from backend.core.ouroboros.governance.code_exploration import CodeExplorationTool
                        _explorer = CodeExplorationTool(str(self._config.project_root))
                    except ImportError:
                        pass

                    expander = ContextExpander(
                        generator=self._generator,
                        repo_root=self._config.project_root,
                        oracle=getattr(self._stack, "oracle", None),
                        skill_registry=_skill_registry,
                        doc_fetcher=_doc_fetcher,
                        web_search=_web_search,
                        visual_comprehension=_visual,
                        code_explorer=_explorer,
                        dialogue_store=self._dialogue_store,
                    )
                    ctx = await asyncio.wait_for(
                        expander.expand(ctx, expansion_deadline),
                        timeout=self._config.context_expansion_timeout_s,
                    )

                    # ExplorationFleet: parallel codebase exploration across Trinity repos
                    if self._exploration_fleet is not None:
                        try:
                            _fleet_report = await asyncio.wait_for(
                                self._exploration_fleet.deploy(
                                    goal=ctx.description,
                                    max_agents=8,
                                ),
                                timeout=min(30.0, self._config.context_expansion_timeout_s / 2),
                            )
                            if _fleet_report.total_findings > 0:
                                _fleet_text = self._exploration_fleet.format_for_prompt(_fleet_report)
                                ctx = ctx.with_expanded_files(
                                    ctx.expanded_files + (f"[Fleet:{_fleet_report.total_findings}]",)
                                )
                                logger.info(
                                    "[Orchestrator] ExplorationFleet: %d agents, %d findings in %.1fs",
                                    _fleet_report.agents_completed,
                                    _fleet_report.total_findings,
                                    _fleet_report.duration_s,
                                )
                        except Exception as _fleet_exc:
                            logger.debug("[Orchestrator] ExplorationFleet skipped: %s", _fleet_exc)

                    # P2.1: Dependency-aware generation — inject Oracle graph summary
                    _oracle_ref = getattr(self._stack, "oracle", None)
                    if _oracle_ref is not None and ctx.target_files:
                        try:
                            _dep_summary = self._build_dependency_summary(
                                _oracle_ref, ctx.target_files,
                            )
                            if _dep_summary:
                                ctx = dataclasses.replace(ctx, dependency_summary=_dep_summary)
                                logger.info(
                                    "[Orchestrator] Dependency summary injected (%d chars, %d files)",
                                    len(_dep_summary), len(ctx.target_files),
                                )
                        except Exception as _dep_exc:
                            logger.debug("[Orchestrator] Dependency summary skipped: %s", _dep_exc)

                    # Sovereign Epistemic Context Matrix (spec 5.1): on a heavy GOAL, build a
                    # bounded, hash-validated candidate DAG from the oracle to seed Venom +
                    # the Information-Gain Governor. Fail-soft; never blocks GENERATE.
                    try:
                        from backend.core.ouroboros.governance.epistemic_prefetch import (
                            build_prefetch_manifest, is_heavy_goal,
                        )
                        if is_heavy_goal(ctx.target_files, int(getattr(ctx, "blast_radius", 0) or 0)):
                            _pf_oracle = _oracle_ref if _oracle_ref is not None else getattr(self._stack, "oracle", None)
                            _pf_timeout = float(os.environ.get("JARVIS_EPISTEMIC_PREFETCH_TIMEOUT_S", "8") or "8")
                            _manifest = await asyncio.wait_for(
                                build_prefetch_manifest(
                                    target_files=tuple(ctx.target_files or ()),
                                    root=str(self._config.project_root),
                                    oracle=_pf_oracle,
                                    goal_text=str(getattr(ctx, "goal", "") or getattr(ctx, "description", "") or ""),
                                    is_heavy=True,
                                ),
                                timeout=_pf_timeout,
                            )
                            if _manifest:
                                ctx = dataclasses.replace(ctx, prefetch_manifest=_manifest)
                                logger.info(
                                    "[Orchestrator] Epistemic prefetch manifest: %d entries [%s]",
                                    len(_manifest), ctx.op_id,
                                )
                    except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — never block GENERATE
                        pass
                except Exception as exc:
                    # exc_info is load-bearing, not decoration. This handler
                    # spans ~130 lines and eight injected capabilities, so
                    # the message alone ("'bool' object is not callable")
                    # names a TYPE and no site — and because the op then
                    # CONTINUES to GENERATE with degraded context, the
                    # failure is invisible in every downstream artifact.
                    # Observed live 3-for-3 in bt-2026-08-11-230412: every
                    # op ran without expansion and nothing surfaced it.
                    # A swallowed exception that changes behaviour must at
                    # minimum say where it came from.
                    logger.warning(
                        "[Orchestrator] Context expansion failed for op=%s: "
                        "%s: %s; continuing to GENERATE with UNEXPANDED "
                        "context",
                        ctx.op_id, type(exc).__name__, exc,
                        exc_info=True,
                    )

                # ---- ModuleContextRouter: architecture memory injection (MEM-2, inline path) ----
                # Parity with ContextExpansionRunner. Gated default-OFF; fail-soft.
                try:
                    from backend.core.ouroboros.governance.module_routing import (  # lazy
                        ModuleContextRouter as _MR,
                        routing_enabled as _mr_enabled,
                    )
                    if _mr_enabled():
                        _mr_router = _MR(self._config.project_root)
                        _mr_result = await _mr_router.route(
                            list(ctx.target_files),
                            ctx.description,
                            op_id=ctx.op_id,
                            consumer="main",
                        )
                        if _mr_result.section:
                            _mr_existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                            ctx = ctx.with_strategic_memory_context(
                                strategic_intent_id=ctx.strategic_intent_id or "module-routing-v1",
                                strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                                strategic_memory_prompt=(
                                    _mr_existing + "\n\n" + _mr_result.section
                                    if _mr_existing else _mr_result.section
                                ),
                                strategic_memory_digest=ctx.strategic_memory_digest,
                            )
                            _mr_rec = getattr(_mr_result, "record", None)
                            logger.info(
                                "[ModuleRouter] op=%s topics=%d inject_site=context_expansion_inline "
                                "prompt_chars=%d corpus=%s/%s",
                                ctx.op_id, len(_mr_result.topics), len(_mr_result.section),
                                getattr(_mr_rec, "corpus_size", "?"),
                                getattr(_mr_rec, "corpus_provenance", "unrecorded"),
                            )
                except Exception:
                    logger.debug("[ModuleRouter] inline injection skipped", exc_info=True)

            # ---- OperatorRules: path-scoped human rules ----
            # The operator's own rules, delivered only where they apply. The
            # store, the `paths:` field, and FORBIDDEN_PATH enforcement all
            # existed; the injection that lets a rule GUIDE a generation rather
            # than only BLOCK a write was documented in
            # `user_preference_memory`'s docstring and never built. Authority-
            # free and fail-soft, exactly like the router above.
            try:
                from backend.core.ouroboros.governance.operator_rules import (
                    compose_for_op as _rules_compose,
                )
                _rules_section = _rules_compose(
                    self._config.project_root,
                    list(ctx.target_files),
                    ctx.description,
                    op_id=ctx.op_id,
                    consumer="main",
                )
                if _rules_section:
                    _r_existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=ctx.strategic_intent_id or "operator-rules-v1",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=(
                            _r_existing + "\n\n" + _rules_section
                            if _r_existing else _rules_section
                        ),
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
            except Exception:
                logger.debug("[OperatorRules] injection skipped", exc_info=True)

                ctx = ctx.advance(OperationPhase.PLAN)
            else:
                # Expansion disabled: skip directly from ROUTE to PLAN
                ctx = ctx.advance(OperationPhase.PLAN)

            # ---- Phase 2c: PLAN — model-reasoned implementation planning ----
            # The model reasons about HOW to implement the change before writing
            # code. Planning failures are soft — the pipeline falls through to
            # GENERATE with an empty plan. Trivial ops skip planning entirely.
            if _serpent:
                _serpent.update_phase("PLAN")
            try:
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="plan", progress_pct=25.0,
                )
            except Exception:
                pass

            _plan_result: Optional[Any] = None
            _plan_review_required_now = _plan_review_required()
            try:
                from backend.core.ouroboros.governance.plan_generator import (
                    PlanGenerator, PLAN_TIMEOUT_S,
                )
                _plan_gen = PlanGenerator(
                    generator=self._generator,
                    repo_root=self._config.project_root,
                )
                _plan_deadline = datetime.now(tz=timezone.utc) + timedelta(
                    seconds=PLAN_TIMEOUT_S,
                )
                # Move 6.5 PLAN seam (v2.97, 2026-05-10) — same
                # shared helper plan_runner.py uses (no duplication
                # between Slice 3-extracted path + this inline
                # legacy path). Returns None on master-off or
                # non-actionable consensus; caller falls through
                # to single-shot.
                _plan_result = None
                try:
                    from backend.core.ouroboros.governance.verification.multi_prior_plan_seam import (  # noqa: E501
                        dispatch_plan_with_multi_prior,
                    )
                    _plan_result = await dispatch_plan_with_multi_prior(
                        ctx=ctx,
                        plan_generator=_plan_gen,
                        deadline=_plan_deadline,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[Orchestrator] multi_prior_plan_seam "
                        "swallowed exception; falling through",
                        exc_info=True,
                    )
                    _plan_result = None
                if _plan_result is None:
                    _plan_result = await asyncio.wait_for(
                        _plan_gen.generate_plan(ctx, _plan_deadline),
                        timeout=PLAN_TIMEOUT_S + 5.0,
                    )

                if not _plan_result.skipped:
                    # Store plan in context for injection into GENERATE prompt
                    ctx = dataclasses.replace(
                        ctx,
                        implementation_plan=_plan_result.plan_json,
                        previous_hash=ctx.context_hash,
                    )
                    # Emit plan result for SerpentFlow rendering
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="plan", progress_pct=28.0,
                            plan_complexity=_plan_result.complexity,
                            plan_changes=len(_plan_result.ordered_changes),
                        )
                    except Exception:
                        pass
                    # The plan is a CHECKLIST — register it so the cockpit
                    # can tick items as files land, instead of the operator
                    # having to run /show_plan mid-flight to see the shape of
                    # work already decided.
                    try:
                        from backend.core.ouroboros.battle_test.plan_checklist import (  # noqa: E501
                            register_plan,
                        )
                        register_plan(
                            ctx.op_id,
                            getattr(_plan_result, "ordered_changes", []) or [],
                        )
                    except Exception:  # noqa: BLE001 — a checklist is additive
                        pass
                    logger.info(
                        "[Orchestrator] PLAN complete for op=%s: complexity=%s, "
                        "%d ordered changes, %.1fs",
                        ctx.op_id, _plan_result.complexity,
                        len(_plan_result.ordered_changes),
                        _plan_result.planning_duration_s,
                    )
                else:
                    logger.debug(
                        "[Orchestrator] PLAN skipped for op=%s: %s",
                        ctx.op_id, _plan_result.skip_reason,
                    )
            except ImportError:
                logger.debug("[Orchestrator] PlanGenerator not available, skipping PLAN phase")
            except Exception as exc:
                logger.warning(
                    "[Orchestrator] PLAN phase failed for op=%s: %s; "
                    "continuing to GENERATE without plan",
                    ctx.op_id, exc,
                )

            # Phase B PLAN-shadow (Slice 1b) — observer-only DAG dispatch.
            # Runs AFTER the legacy PlanGenerator regardless of whether the
            # legacy plan succeeded or skipped. Gated by
            # JARVIS_PLAN_SUBAGENT_SHADOW (default false). The shadow never
            # raises and never blocks the FSM; its only side-effect is
            # setting ctx.execution_graph + emitting [PLAN-SHADOW] telemetry.
            try:
                ctx = await self._run_plan_shadow(ctx)
            except Exception:
                # Defense in depth — the hook itself is exception-safe but
                # an awaitable propagation through asyncio.wait_for etc.
                # could surface edge-case cancellations. Never propagate.
                logger.debug(
                    "[Orchestrator] PLAN-shadow wrapper swallowed exception",
                    exc_info=True,
                )

            if _plan_review_required_now and (
                _plan_result is None or getattr(_plan_result, "skipped", True)
            ):
                _skip_reason = getattr(_plan_result, "skip_reason", "") or "plan_not_available"
                logger.info(
                    "[Orchestrator] Plan review required for op=%s but no plan is "
                    "available: %s",
                    ctx.op_id,
                    _skip_reason,
                )
                ctx = ctx.advance(
                    OperationPhase.CANCELLED,
                    terminal_reason_code="plan_required_unavailable",
                )
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {
                        "reason": "plan_required_unavailable",
                        "detail": _skip_reason,
                    },
                )
                return ctx

            # ---- Phase 2d: Plan Approval Hard Gate (Phase 1b) ----
            # For COMPLEX / ARCHITECTURAL ops, pause BEFORE burning generation
            # tokens and get human sign-off on the approach. Rejection aborts
            # the op; approval proceeds to GENERATE. Manifesto §6 (Iron Gate):
            # "every autonomous decision is visible" + cost protection.
            #
            # Env-gated, fully override-able for battle tests and CI:
            #   JARVIS_PLAN_APPROVAL_ENABLED         (default true)
            #   JARVIS_PLAN_APPROVAL_ROUTES          (default "complex")
            #   JARVIS_PLAN_APPROVAL_COMPLEXITIES    (default "complex,heavy_code,architectural")
            #   JARVIS_PLAN_APPROVAL_TIMEOUT_S       (default 600.0)
            #   JARVIS_PLAN_APPROVAL_EXPIRE_GRACE    (default false — strict)
            _plan_gate_enabled = _plan_review_required_now or (
                os.environ.get("JARVIS_PLAN_APPROVAL_ENABLED", "true").lower()
                not in ("false", "0", "no", "off")
            )
            _plan_gate_applied = False
            if (
                _plan_gate_enabled
                and _plan_result is not None
                and not getattr(_plan_result, "skipped", True)
            ):
                _gate_routes = {
                    r.strip().lower()
                    for r in os.environ.get(
                        "JARVIS_PLAN_APPROVAL_ROUTES", "complex"
                    ).split(",")
                    if r.strip()
                }
                _gate_complexities = {
                    c.strip().lower()
                    for c in os.environ.get(
                        "JARVIS_PLAN_APPROVAL_COMPLEXITIES",
                        "complex,heavy_code,architectural",
                    ).split(",")
                    if c.strip()
                }
                _route = (getattr(ctx, "provider_route", "") or "").lower()
                _task_cx = (getattr(ctx, "task_complexity", "") or "").lower()
                _plan_cx = (getattr(_plan_result, "complexity", "") or "").lower()
                # OR-predicate: gate trips if ANY of (provider_route,
                # task_complexity, plan_result.complexity) matches the filters.
                # plan_result.complexity takes precedence because the model
                # has just reasoned about the actual scope during PLAN phase.
                # Problem #7 Slice 2: plan-mode force-review override.
                # When JARVIS_PLAN_APPROVAL_MODE=true (or ctx opt-in)
                # the operator has explicitly asked to halt EVERY op
                # for review, regardless of the complexity heuristic.
                # Late import keeps plan_approval optional — if the
                # module is unavailable for any reason, plan mode is
                # treated as off. Never raises.
                _plan_mode_force = False
                try:
                    from backend.core.ouroboros.governance.plan_approval import (
                        should_force_plan_review as _should_force_plan_review,
                    )
                    _plan_mode_force = _should_force_plan_review(ctx)
                except Exception:  # noqa: BLE001 — optional dep
                    _plan_mode_force = False
                _should_gate = (
                    _plan_review_required_now
                    or _plan_mode_force
                    or _route in _gate_routes
                    or _task_cx in _gate_complexities
                    or _plan_cx in _gate_complexities
                )
                _provider_supports_plan = (
                    self._approval_provider is not None
                    and hasattr(self._approval_provider, "request_plan")
                )
                if _should_gate and not _provider_supports_plan:
                    logger_msg = (
                        "[Orchestrator] Plan review required for op=%s but no "
                        "plan approval provider is available"
                        if _plan_review_required_now
                        else "[Orchestrator] Plan Gate skipped for op=%s: "
                        "provider=%s has_request_plan=%s"
                    )
                    if _plan_review_required_now:
                        logger.info(logger_msg, ctx.op_id)
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="plan_review_unavailable",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": "plan_review_unavailable",
                                "detail": "approval_provider_missing",
                            },
                        )
                        return ctx
                    logger.debug(
                        logger_msg,
                        ctx.op_id,
                        type(self._approval_provider).__name__
                        if self._approval_provider
                        else "None",
                        hasattr(self._approval_provider, "request_plan"),
                    )
                elif _should_gate:
                    _plan_gate_applied = True
                    _plan_gate_timeout = float(os.environ.get(
                        "JARVIS_PLAN_APPROVAL_TIMEOUT_S", "600.0"
                    ))
                    _expire_grace = os.environ.get(
                        "JARVIS_PLAN_APPROVAL_EXPIRE_GRACE", "false"
                    ).lower() in ("true", "1", "yes", "on")

                    # Render plan as markdown for human review. Fall back to
                    # raw JSON if to_prompt_section() is unavailable.
                    try:
                        _plan_markdown = _plan_result.to_prompt_section()
                    except Exception:
                        _plan_markdown = _plan_result.plan_json or "(no plan)"

                    logger.info(
                        "[Orchestrator] Plan Gate engaged for op=%s "
                        "(route=%r task_cx=%r plan_cx=%r) — awaiting human",
                        ctx.op_id, _route, _task_cx, _plan_cx,
                    )
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="plan", progress_pct=30.0,
                            plan_gate_engaged=True,
                        )
                    except Exception:
                        pass

                    # Problem #7 Slice 2: shadow-register this plan with
                    # the PlanApprovalController so REPL (/plan pending,
                    # /plan show) and IDE observability (/observability/
                    # plans, SSE plan_* events) surface it. The primary
                    # approval authority stays with self._approval_provider;
                    # this is a read-only mirror for operator visibility.
                    # Best-effort: any failure (module unavailable,
                    # duplicate op_id, etc.) silently no-ops — the actual
                    # approval path is unaffected.
                    _plan_mirror_registered = False
                    try:
                        from backend.core.ouroboros.governance.plan_approval import (
                            get_default_controller as _get_pa_controller,
                        )
                        _pa_controller = _get_pa_controller()
                        if _pa_controller.snapshot(ctx.op_id) is None:
                            _pa_controller.request_approval(
                                ctx.op_id,
                                {
                                    "markdown": _plan_markdown,
                                    "description": getattr(ctx, "description", ""),
                                    "target_files": list(
                                        getattr(ctx, "target_files", []) or [],
                                    ),
                                    "approach": getattr(
                                        _plan_result, "approach", "",
                                    ) or "",
                                    "complexity": getattr(
                                        _plan_result, "complexity", "",
                                    ) or "",
                                    "ordered_changes": list(
                                        getattr(
                                            _plan_result, "ordered_changes", [],
                                        ) or [],
                                    ),
                                    "risk_factors": list(
                                        getattr(
                                            _plan_result, "risk_factors", [],
                                        ) or [],
                                    ),
                                    "test_strategy": getattr(
                                        _plan_result, "test_strategy", "",
                                    ) or "",
                                },
                                timeout_s=_plan_gate_timeout,
                            )
                            _plan_mirror_registered = True
                    except Exception:  # noqa: BLE001 — best-effort mirror
                        logger.debug(
                            "[Orchestrator] PlanApproval mirror register "
                            "best-effort failed for op=%s", ctx.op_id,
                            exc_info=True,
                        )

                    try:
                        _plan_req_id = await self._approval_provider.request_plan(
                            ctx, _plan_markdown,
                        )
                        _plan_decision: ApprovalResult = await (
                            self._approval_provider.await_decision(
                                _plan_req_id, _plan_gate_timeout,
                            )
                        )
                    except Exception as _gate_exc:
                        if _plan_review_required_now:
                            logger.info(
                                "[Orchestrator] Plan review required for op=%s but "
                                "the plan gate failed: %s",
                                ctx.op_id,
                                _gate_exc,
                            )
                            ctx = ctx.advance(
                                OperationPhase.CANCELLED,
                                terminal_reason_code="plan_review_unavailable",
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": "plan_review_unavailable",
                                    "detail": str(_gate_exc)[:200],
                                },
                            )
                            return ctx
                        # Gate infrastructure failure — log and continue without
                        # gating rather than blocking the pipeline forever.
                        logger.warning(
                            "[Orchestrator] Plan Gate infra failure for op=%s: %s; "
                            "continuing to GENERATE without approval",
                            ctx.op_id, _gate_exc,
                        )
                        _plan_decision = None  # type: ignore[assignment]

                    if _plan_decision is not None:
                        # Problem #7 Slice 2: mirror the decision onto
                        # the PlanApprovalController shadow so REPL /
                        # IDE views see the terminal transition. Best-
                        # effort; never raises.
                        if _plan_mirror_registered:
                            try:
                                from backend.core.ouroboros.governance.plan_approval import (
                                    get_default_controller as _get_pa_ctrl,
                                    PlanApprovalStateError as _PAStateError,
                                )
                                _pa_mirror_ctrl = _get_pa_ctrl()
                                _mirror_approver = (
                                    getattr(_plan_decision, "approver", None)
                                    or "orchestrator"
                                )
                                try:
                                    if _plan_decision.status is ApprovalStatus.APPROVED:
                                        _pa_mirror_ctrl.approve(
                                            ctx.op_id, reviewer=_mirror_approver,
                                        )
                                    elif _plan_decision.status is ApprovalStatus.REJECTED:
                                        _pa_mirror_ctrl.reject(
                                            ctx.op_id,
                                            reason=getattr(
                                                _plan_decision, "reason", "",
                                            ) or "",
                                            reviewer=_mirror_approver,
                                        )
                                    # EXPIRED path: the controller's own
                                    # timeout already auto-rejects; no
                                    # additional mirror call needed.
                                except _PAStateError:
                                    # Already terminal — the controller's
                                    # timeout_task may have expired the
                                    # shadow first. Harmless; skip.
                                    pass
                            except Exception:  # noqa: BLE001 — best-effort
                                logger.debug(
                                    "[Orchestrator] PlanApproval mirror terminal "
                                    "propagation best-effort failed for op=%s",
                                    ctx.op_id, exc_info=True,
                                )
                        if _plan_decision.status is ApprovalStatus.REJECTED:
                            _reject_reason = (
                                getattr(_plan_decision, "reason", "") or ""
                            )
                            logger.info(
                                "[Orchestrator] Plan REJECTED for op=%s: %s",
                                ctx.op_id, _reject_reason,
                            )
                            ctx = ctx.advance(
                                OperationPhase.CANCELLED,
                                terminal_reason_code="plan_rejected",
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": "plan_rejected",
                                    "approver": _plan_decision.approver,
                                    "rejection_reason": _reject_reason,
                                    "plan_complexity": _plan_cx,
                                },
                            )
                            # Persist rejection so future similar plans learn from it.
                            if _reject_reason:
                                try:
                                    from backend.core.ouroboros.governance.user_preference_memory import (
                                        get_default_store,
                                    )
                                    get_default_store().record_approval_rejection(
                                        op_id=ctx.op_id,
                                        description=f"[PLAN] {ctx.description}",
                                        target_files=list(ctx.target_files),
                                        reason=_reject_reason,
                                        provenance=getattr(
                                            _plan_decision,
                                            "reason_provenance", "unstated"),
                                        approver=(
                                            getattr(_plan_decision, "approver", "human")
                                            or "human"
                                        ),
                                    )
                                except Exception:
                                    pass
                            # Session lesson for intra-session learning.
                            _files_short = ", ".join(
                                p.rsplit("/", 1)[-1] for p in ctx.target_files[:3]
                            )
                            self._add_session_lesson(
                                "code",
                                f"[PLAN REJECTED] {ctx.description[:60]} "
                                f"({_files_short}) — human rejected the approach: "
                                f"{_reject_reason[:80] or 'no reason given'}. "
                                f"Reconsider strategy before retry.",
                                op_id=ctx.op_id,
                            )
                            return ctx

                        if _plan_decision.status is ApprovalStatus.EXPIRED:
                            if _expire_grace and not _plan_review_required_now:
                                logger.warning(
                                    "[Orchestrator] Plan Gate expired for op=%s; "
                                    "grace mode — continuing to GENERATE",
                                    ctx.op_id,
                                )
                            else:
                                logger.info(
                                    "[Orchestrator] Plan Gate EXPIRED for op=%s — "
                                    "aborting (strict mode)",
                                    ctx.op_id,
                                )
                                ctx = ctx.advance(
                                    OperationPhase.EXPIRED,
                                    terminal_reason_code="plan_approval_expired",
                                )
                                await self._record_ledger(
                                    ctx,
                                    OperationState.FAILED,
                                    {"reason": "plan_approval_expired"},
                                )
                                return ctx

                        # APPROVED (or grace on EXPIRED) — continue to GENERATE
                        if _plan_decision.status is ApprovalStatus.APPROVED:
                            logger.info(
                                "[Orchestrator] Plan APPROVED for op=%s by %s",
                                ctx.op_id, _plan_decision.approver,
                            )
            ctx = ctx.advance(OperationPhase.GENERATE)

            # Cryptographic Truth Guard (spec 5.3.1 / LR2): re-validate the prefetch
            # manifest against live disk right before it is consumed. Stale entries
            # are dropped from the seed (Venom reads them fresh) and quarantined
            # (session-scoped) for sibling workers + teardown reconcile. Fail-soft.
            if getattr(ctx, "prefetch_manifest", ()):
                try:
                    from backend.core.ouroboros.governance.epistemic_prefetch import revalidate_manifest
                    from backend.core.ouroboros.governance.epistemic_quarantine import QuarantineLedger
                    _root = str(self._config.project_root)
                    _sid = self._resolve_session_id()
                    _led = QuarantineLedger(
                        path=os.path.join(_root, ".jarvis", "epistemic_quarantine.jsonl"),
                        session_id=_sid,
                    )
                    _validated = revalidate_manifest(ctx.prefetch_manifest, _root, ledger=_led)
                    if _validated != ctx.prefetch_manifest:
                        ctx = dataclasses.replace(ctx, prefetch_manifest=_validated)
                except Exception:  # noqa: BLE001 — never block GENERATE
                    pass

            # ── Option C: DW topology early-detection circuit breaker ──
            # Pre-GENERATE check: if route=BACKGROUND AND topology says
            # skip_and_queue AND op is NOT read-only, the op is
            # structurally doomed (CandidateGenerator will raise
            # background_dw_blocked_by_topology when invoked). Skip
            # the GENERATE phase entirely and go straight to the same
            # graceful-accept path the late-detection branch already
            # uses (CANCELLED + FAILED ledger). Outcome is byte-
            # identical to today's late-detection path; the difference
            # is "[CircuitBreaker] pre-GENERATE skip" log instead of
            # "BACKGROUND route: DW failed... accepting" after a
            # generation hot-path entry.
            #
            # Master flag JARVIS_DW_TOPOLOGY_EARLY_REJECT_ENABLED
            # (default false). When off, this block is a no-op and
            # the late-detection path runs exactly as before.
            try:
                from backend.core.ouroboros.governance.dw_topology_circuit_breaker import (  # noqa: E501
                    is_circuit_breaker_enabled as _cb_enabled,
                    ledger_reason_label as _cb_ledger_label,
                    should_circuit_break as _cb_should_break,
                    terminal_reason_code as _cb_terminal_code,
                )
                if _cb_enabled():
                    _cb_break, _cb_reason = _cb_should_break(
                        provider_route=getattr(
                            ctx, "provider_route", "",
                        ) or "",
                        is_read_only=bool(
                            getattr(ctx, "is_read_only", False),
                        ),
                    )
                    if _cb_break:
                        logger.info(
                            "[CircuitBreaker] pre-GENERATE skip: "
                            "route=%s reason=%s op=%s",
                            getattr(ctx, "provider_route", "?"),
                            _cb_reason[:120],
                            (ctx.op_id or "?")[:16],
                        )
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code=(
                                _cb_terminal_code(_cb_reason)
                            ),
                        )
                        await self._record_ledger(
                            ctx, OperationState.FAILED,
                            {
                                "reason": _cb_ledger_label(_cb_reason),
                                "topology_reason": _cb_reason[:200],
                                "route": getattr(
                                    ctx, "provider_route", "",
                                ),
                                "circuit_breaker_fired": True,
                            },
                        )
                        return ctx
            except Exception:  # noqa: BLE001 — never let circuit
                # breaker crash GENERATE entry. The late-detection
                # path remains the authoritative behavior.
                logger.debug(
                    "[CircuitBreaker] consultation raised — falling "
                    "through to late-detection path",
                    exc_info=True,
                )

            # ── PreActionNarrator: voice WHAT before GENERATE ──
            if self._pre_action_narrator is not None:
                try:
                    _provider_name = getattr(ctx, "routing_actual", None) or "unknown"
                    await self._pre_action_narrator.narrate_phase(
                        "GENERATE",
                        {"provider": str(_provider_name), "thinking_mode": "standard"},
                    )
                except Exception:
                    pass

            # ── P2: Adaptive Learning — inject consolidated rules + success patterns ──
            try:
                from backend.core.ouroboros.governance.adaptive_learning import (
                    LearningConsolidator, SuccessPatternStore,
                )
                from backend.core.ouroboros.governance.entropy_calculator import (
                    extract_domain_key as _extract_dk,
                )
                _domain = _extract_dk(ctx.target_files, ctx.description)

                _consolidator = LearningConsolidator()
                _rules_context = _consolidator.format_rules_for_prompt(_domain)

                _success_store = SuccessPatternStore()
                _success_context = _success_store.format_for_prompt(_domain, ctx.target_files)

                if _rules_context or _success_context:
                    _existing_mem = getattr(ctx, "strategic_memory_prompt", "") or ""
                    _learning_block = ""
                    if _rules_context:
                        _learning_block += f"\n\n{_rules_context}"
                    if _success_context:
                        _learning_block += f"\n\n{_success_context}"
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing_mem + _learning_block,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[Orchestrator] Adaptive learning: injected %d rules + %d success "
                        "patterns for domain=%s (op=%s)",
                        len(_consolidator.get_rules_for_domain(_domain)),
                        len(_success_store.get_similar_successes(_domain, ctx.target_files)),
                        _domain, ctx.op_id,
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Adaptive learning injection failed", exc_info=True)

            # ── P0: Test Coverage Enforcer (pre-GENERATE) ─────────────────────
            # Slice 239 (adaptive test-sharding, layer 9): when budget is tight
            # and >2 target files are uncovered, DECOUPLE the "generate tests"
            # requirement into a SEPARATE background intake op (reusing the
            # UnifiedIntakeRouter WAL queue) so the PRIMARY patch graduates the
            # Iron Gate cleanly instead of ballooning past its deadline. Otherwise
            # inline-inject the instruction (legacy). Fail-soft throughout: no
            # router / emit failure / ample budget → inline injection.
            try:
                from backend.core.ouroboros.governance.intelligence_hooks import (
                    TestCoverageEnforcer,
                    should_decouple_test_gen,
                    build_test_coverage_envelope,
                    test_sharding_enabled,
                    estimate_test_gen_tokens,
                    _shard_velocity_tok_s,
                )
                _coverage_enforcer = TestCoverageEnforcer(self._config.project_root)
                _uncovered = _coverage_enforcer.detect_uncovered(ctx.target_files)
                _decoupled = False
                if _uncovered:
                    _remaining_s = float("inf")
                    try:
                        _dl = getattr(ctx, "pipeline_deadline", None)
                        if _dl is not None:
                            _remaining_s = (
                                _dl - datetime.now(tz=timezone.utc)
                            ).total_seconds()
                    except Exception:  # noqa: BLE001
                        _remaining_s = float("inf")
                    # Slice 240 — dynamic cost-vs-bandwidth shard trigger (no
                    # hardcoded file-count gate): decouple iff the estimated tokens
                    # to generate the tests exceed velocity × remaining budget.
                    _est_test_tokens = estimate_test_gen_tokens(
                        uncovered_files=_uncovered,
                        repo_root=self._config.project_root,
                    )
                    if test_sharding_enabled() and should_decouple_test_gen(
                        est_test_tokens=_est_test_tokens,
                        velocity_tok_s=_shard_velocity_tok_s(),
                        remaining_s=_remaining_s,
                        enabled=True,
                    ):
                        _gls = getattr(self._stack, "governed_loop_service", None)
                        _router = getattr(_gls, "_intake_router", None) if _gls else None
                        if _router is not None:
                            try:
                                _env = build_test_coverage_envelope(
                                    uncovered_files=_uncovered,
                                    parent_op_id=getattr(ctx, "op_id", "") or "",
                                    repo=getattr(ctx, "repo", "") or "jarvis",
                                )
                                _ing = await _router.ingest(_env)
                                _decoupled = True
                                logger.info(
                                    "[Orchestrator] Slice239 test-sharding: DECOUPLED "
                                    "test-gen for %d uncovered file(s) → intake (%s); "
                                    "primary patch graduates clean (op=%s)",
                                    len(_uncovered), _ing, ctx.op_id,
                                )
                            except Exception:  # noqa: BLE001 — never break the primary op
                                logger.debug(
                                    "[Orchestrator] Slice239 decouple emit failed — "
                                    "falling back to inline inject", exc_info=True,
                                )
                                _decoupled = False
                    if not _decoupled:
                        # Legacy inline injection (ample budget / no router / emit failed).
                        _coverage_instruction = _coverage_enforcer.check_and_inject(
                            ctx.target_files, ctx.description,
                        )
                        if _coverage_instruction:
                            _existing_human = getattr(ctx, "human_instructions", "") or ""
                            ctx = dataclasses.replace(
                                ctx,
                                human_instructions=_existing_human + _coverage_instruction,
                                previous_hash=ctx.context_hash,
                            )
                            logger.info(
                                "[Orchestrator] TestCoverageEnforcer: injected test "
                                "generation instruction for %d uncovered file(s) (op=%s)",
                                len(_uncovered), ctx.op_id,
                            )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] TestCoverageEnforcer failed", exc_info=True)

            # ── JARVIS Tier 5: Cross-Domain Intelligence ──────────────────────
            try:
                from backend.core.ouroboros.governance.jarvis_intelligence import (
                    UnifiedIntelligenceLayer,
                )
                _intel = UnifiedIntelligenceLayer(self._config.project_root)
                _syntheses = _intel.analyze_all_domains()
                _intel_prompt = _intel.format_for_prompt(_syntheses)
                if _intel_prompt:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing + "\n\n" + _intel_prompt,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[Orchestrator] JARVIS Tier 5: %d cross-domain syntheses injected",
                        len(_syntheses),
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Tier 5 injection failed", exc_info=True)

            # ── JARVIS Tier 6: Personality voice line ─────────────────────────
            _gls = getattr(self._stack, "governed_loop_service", None)
            if _gls is not None:
                _pe = getattr(_gls, "_personality_engine", None)
                if _pe is not None:
                    try:
                        _chronic = getattr(_advisory, "chronic_entropy", 0.0) if _advisory else 0.0
                        _emerg = getattr(self._stack, "_emergency_engine", None)
                        _emerg_lvl = _emerg.current_level.value if _emerg else 0
                        _state = _pe.compute_state(
                            success_rate=_pe.success_rate,
                            chronic_entropy=_chronic,
                            emergency_level=_emerg_lvl,
                        )
                        if self._reasoning_narrator is not None:
                            _voice = _pe.get_voice_line(_state)
                            self._reasoning_narrator.record_classify(
                                ctx.op_id, f"personality:{_state.value}", _voice,
                            )
                    except Exception:
                        pass

            # ── Advanced Repair: hierarchical localization + slow/fast thinking + doc-augmented ──
            try:
                from backend.core.ouroboros.governance.advanced_repair import (
                    HierarchicalFaultLocalizer, SlowFastThinkingRouter, DocAugmentedRepair,
                )
                _apr_blocks: list = []

                # 1. Hierarchical fault localization (file → function → line)
                _localizer = HierarchicalFaultLocalizer(self._config.project_root)
                _error_msg = getattr(ctx, "error_pattern", "") or ctx.description
                _locations = _localizer.localize(ctx.target_files, _error_msg)
                _loc_prompt = _localizer.format_for_prompt(_locations)
                if _loc_prompt:
                    _apr_blocks.append(_loc_prompt)

                # 2. Slow/fast thinking router
                _thinking = SlowFastThinkingRouter.route(
                    ctx.description, ctx.target_files,
                )
                _think_prompt = SlowFastThinkingRouter.format_for_prompt(_thinking)
                if _think_prompt:
                    _apr_blocks.append(_think_prompt)

                # 3. Documentation-augmented repair context
                _doc_repair = DocAugmentedRepair(self._config.project_root)
                _doc_context = _doc_repair.generate_docs_for_repair(ctx.target_files)
                if _doc_context:
                    _apr_blocks.append(_doc_context)

                if _apr_blocks:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    _apr_combined = "\n\n".join(_apr_blocks)
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing + "\n\n" + _apr_combined,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[Orchestrator] Advanced repair: %d blocks (localization=%d locs, "
                        "thinking=%s, docs=%d chars) for op=%s",
                        len(_apr_blocks), len(_locations), _thinking.depth,
                        len(_doc_context), ctx.op_id,
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Advanced repair injection failed", exc_info=True)

            # ── Self-Evolution P0: Inject runtime prompt adaptations + negative constraints + code metrics ──
            try:
                from backend.core.ouroboros.governance.self_evolution import (
                    RuntimePromptAdapter, NegativeConstraintStore,
                    CodeMetricsAnalyzer, MultiVersionEvolutionTracker,
                )
                from backend.core.ouroboros.governance.entropy_calculator import extract_domain_key as _edk

                _se_domain = _edk(ctx.target_files, ctx.description)
                _se_blocks: List[str] = []

                # P0: Runtime prompt adaptation — learned instructions from outcomes
                _prompt_adapter = RuntimePromptAdapter()
                _adapted = _prompt_adapter.get_adapted_instructions(_se_domain)
                if _adapted:
                    _se_blocks.append(_adapted)

                # P0: Negative constraints — "never do X" rules
                _neg_store = NegativeConstraintStore()
                _neg_prompt = _neg_store.format_for_prompt(_se_domain)
                if _neg_prompt:
                    _se_blocks.append(_neg_prompt)

                # P1: Code metrics feedback — objective quality signals
                for _tf in ctx.target_files[:3]:
                    _tf_path = self._config.project_root / _tf
                    if _tf_path.is_dir() or not _tf_path.suffix:
                        continue  # Skip directories — only analyze files
                    _metrics = CodeMetricsAnalyzer.analyze(_tf_path)
                    if _metrics:
                        _mf = CodeMetricsAnalyzer.format_for_prompt(_metrics)
                        if _mf:
                            _se_blocks.append(_mf)

                if _se_blocks:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    _se_combined = "\n\n".join(_se_blocks)
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing + "\n\n" + _se_combined,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[Orchestrator] Self-evolution: injected %d blocks for domain=%s",
                        len(_se_blocks), _se_domain,
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Self-evolution injection failed", exc_info=True)

            # ── Self-Evolution P2: Module-level function analysis + auto-documentation gaps ──
            try:
                from backend.core.ouroboros.governance.self_evolution import (
                    ModuleLevelMutator, RepositoryAutoDocumentation,
                )
                _se2_blocks: List[str] = []

                # ModuleLevelMutator: show function-level breakdown of target files
                # so the generator can do surgical mutations instead of full rewrites
                for _tf in ctx.target_files[:3]:
                    _tf_path = self._config.project_root / _tf
                    if not _tf_path.is_file() or _tf_path.suffix != ".py":
                        continue
                    _funcs = ModuleLevelMutator.list_functions(_tf_path)
                    if _funcs:
                        _complex = [f for f in _funcs if f["complexity"] > 5]
                        if _complex:
                            _func_info = ", ".join(
                                f"{f['name']}(CC={f['complexity']}, L{f['start_line']}-{f['end_line']})"
                                for f in sorted(_complex, key=lambda x: x["complexity"], reverse=True)[:5]
                            )
                            _se2_blocks.append(
                                f"## Function-level analysis: {_tf}\n"
                                f"Complex functions (surgical mutation targets): {_func_info}\n"
                                f"Prefer modifying individual functions over full-file rewrites."
                            )

                # RepositoryAutoDocumentation: show doc gaps in target files
                _auto_doc = RepositoryAutoDocumentation()
                for _tf in ctx.target_files[:3]:
                    _tf_path = self._config.project_root / _tf
                    if _tf_path.is_file() and _tf_path.suffix == ".py":
                        _auto_doc.scan_file(_tf_path)
                _doc_prompt = _auto_doc.format_for_prompt(
                    [str(self._config.project_root / tf) for tf in ctx.target_files[:3]]
                )
                if _doc_prompt:
                    _se2_blocks.append(_doc_prompt)

                if _se2_blocks:
                    _existing = getattr(ctx, "strategic_memory_prompt", "") or ""
                    _se2_combined = "\n\n".join(_se2_blocks)
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=getattr(ctx, "strategic_intent_id", "") or "",
                        strategic_memory_fact_ids=ctx.strategic_memory_fact_ids,
                        strategic_memory_prompt=_existing + "\n\n" + _se2_combined,
                        strategic_memory_digest=ctx.strategic_memory_digest,
                    )
                    logger.info(
                        "[Orchestrator] Self-evolution P2: injected %d blocks "
                        "(module analysis + doc gaps)",
                        len(_se2_blocks),
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("[Orchestrator] Self-evolution P2 injection failed", exc_info=True)

            # ── Cooperative cancellation check (pre-GENERATE) ──
            if self._is_cancel_requested(ctx.op_id):
                ctx = ctx.advance(OperationPhase.CANCELLED, terminal_reason_code="user_cancelled")
                await self._record_ledger(ctx, OperationState.FAILED, {"reason": "user_cancelled"})
                return ctx

        # Wave 2 (5) Slice 5a/5b - GENERATERunner delegation gate.
        # Flag JARVIS_PHASE_RUNNER_GENERATE_EXTRACTED (default TRUE) routes
        # the 1611-line GENERATE block through the extracted PhaseRunner
        # (the LIVE path — the inline block below is the legacy fallback).
        # Cross-phase artifacts (generation, _episodic_memory) threaded
        # via artifacts for VALIDATE consumption.
        if _phase_runner_generate_extracted():
            from backend.core.ouroboros.governance.phase_runners.generate_runner import (
                GENERATERunner,
            )
            logger.info("[PhaseRunnerDelegate] GENERATE → runner op=%s", ctx.op_id[:16])
            _generate_runner = GENERATERunner(self, _serpent, _consciousness_bridge)
            _generate_result = await _generate_runner.run(ctx)
            generation = _generate_result.artifacts.get("generation")
            _episodic_memory = _generate_result.artifacts.get("episodic_memory")
            # generate_retries_remaining is consumed by VALIDATE's entropy
            # computation (orchestrator.py ~5402 retries_used=...).
            generate_retries_remaining = _generate_result.artifacts.get(
                "generate_retries_remaining",
                self._config.max_generate_retries,
            )
            if _generate_result.next_phase is None:
                # Terminal exit (cost cap / no_forward_progress / stalled /
                # l2 escape / iron gate failure / etc.)
                return _generate_result.next_ctx
            ctx = _generate_result.next_ctx
            # ── M9 Slice 5b — feed logprob-entropy producer ──
            # Convert ConfidenceMonitor's rolling-mean margin into an
            # entropy-shaped signal: high margin → confident model →
            # low entropy; low margin → uncertain → high entropy.
            # ``entropy_normalized = clamp(1.0 - margin, 0, 1)`` is the
            # cheap-and-honest mapping (not Shannon entropy, but
            # monotonic in the same direction). Feeds M9's
            # CuriosityCollector once per target_file via the
            # producer bridge — lazy-imported, master-flag-gated,
            # exception-isolated. NEVER raises out.
            try:
                from backend.core.ouroboros.governance.curiosity_producer_bridge import (  # noqa: E501
                    feed_logprob_entropy as _m9_feed_entropy,
                )
                _m9_monitor = (
                    ctx.artifacts.get("confidence_monitor")
                    if hasattr(ctx, "artifacts") else None
                )
                if _m9_monitor is not None:
                    _m9_margin = _m9_monitor.current_margin()
                    if _m9_margin is not None:
                        _m9_entropy = max(
                            0.0, min(1.0, 1.0 - float(_m9_margin)),
                        )
                        _m9_targets = (
                            tuple(ctx.target_files)
                            if getattr(ctx, "target_files", None)
                            else ("_global",)
                        )
                        for _m9_target in _m9_targets:
                            _m9_feed_entropy(
                                region_or_path=str(_m9_target),
                                entropy_normalized=_m9_entropy,
                                op_id=str(
                                    getattr(ctx, "op_id", ""),
                                ),
                            )
            except Exception:  # noqa: BLE001 — defensive
                pass
        else:
            if _serpent: _serpent.update_phase("GENERATE")
            # ---- Phase 3: GENERATE (with retry + episodic failure memory) ----
            generation: Optional[GenerationResult] = None
            generate_retries_remaining = self._config.max_generate_retries

            # Episodic failure memory — per-operation, injected into retries
            _episodic_memory = None
            try:
                from backend.core.ouroboros.governance.episodic_memory import EpisodicFailureMemory
                _episodic_memory = EpisodicFailureMemory(ctx.op_id)
            except ImportError:
                pass

            # ── Inject cumulative session lessons into context ──
            # Filter out infrastructure failures (timeouts, provider outages) to
            # avoid poisoning the model with environmentally-caused failures.
            if self._session_lessons:
                _code_lessons = [
                    text for (ltype, text) in self._session_lessons
                    if ltype == "code"
                ][-self._session_lessons_max:]
                if _code_lessons:
                    _lessons_text = "\n".join(f"- {lesson}" for lesson in _code_lessons)
                    ctx = dataclasses.replace(
                        ctx,
                        session_lessons=_lessons_text,
                    )

            # ── Consciousness: inject fragile-file memory into first generation ──
            # Manifesto §4: "The organism possesses episodic memory and metacognition"
            if _consciousness_bridge is not None:
                try:
                    _fragile_ctx = _consciousness_bridge.get_fragile_file_context(
                        ctx.target_files
                    )
                    if _fragile_ctx:
                        _existing_mem = getattr(ctx, "strategic_memory_prompt", "") or ""
                        ctx = dataclasses.replace(
                            ctx,
                            strategic_memory_prompt=(
                                f"{_existing_mem}\n\n{_fragile_ctx}" if _existing_mem else _fragile_ctx
                            ),
                        )
                        logger.info(
                            "[Orchestrator] Consciousness memory injected into GENERATE context "
                            "(%d chars) [%s]",
                            len(_fragile_ctx), ctx.op_id,
                        )
                except Exception:
                    logger.debug("[Orchestrator] Consciousness injection failed", exc_info=True)

            # ── Stale-exploration guard: snapshot file hashes at GENERATE time ──
            _gen_hashes: list = []
            for _tf in ctx.target_files:
                _tf_path = self._config.project_root / _tf
                try:
                    _tf_bytes = _tf_path.read_bytes()
                    _gen_hashes.append((_tf, hashlib.sha256(_tf_bytes).hexdigest()))
                except (OSError, IOError):
                    _gen_hashes.append((_tf, ""))  # new file — no hash
            if _gen_hashes:
                ctx = dataclasses.replace(ctx, generate_file_hashes=tuple(_gen_hashes))

            # Cumulative exploration credit across the GENERATE retry loop. When a
            # prior attempt satisfied the floor but failed downstream gates (ASCII,
            # dependency integrity, etc.), the retry feedback embeds the rejected
            # file content — re-reading via read_file is wasteful, so the credit
            # carries forward instead of forcing the model to spend tool rounds on
            # the same file twice (bt-2026-04-11-204228 / op-019d7e4c).
            _op_explore_credit = 0
            # Ledger-path counterpart to _op_explore_credit (#103).
            # When JARVIS_EXPLORATION_LEDGER_ENABLED is true the Iron Gate consults
            # ExplorationLedger.from_records(_op_explore_records) instead of the
            # int counter. Records accumulate across retries so the ledger sees
            # the union of every tool call the model has made for this op, then
            # dedup-by-(tool, arguments_hash) happens inside diversity_score().
            _op_explore_records: List[Any] = []

            # Sovereign Epistemic Context Matrix (LR3): bind the governance
            # deadlock exception so the generation retry loop's dedicated catch
            # can resolve it. Lazy import (no module-level coupling to
            # tool_executor); fail-soft to a sentinel that can never be raised
            # so the generic Exception handler still covers everything.
            try:
                from backend.core.ouroboros.governance.tool_executor import (
                    GovernanceDeadlockError,
                )
            except Exception:  # noqa: BLE001 — never block GENERATE on import
                class GovernanceDeadlockError(RuntimeError):  # type: ignore[no-redef]
                    """Unreachable sentinel — import failed; never raised."""

            for attempt in range(1 + self._config.max_generate_retries):
                # ── Per-op cost cap check (Manifesto §5/§7) ──
                # If the cumulative spend across previous attempts has already
                # exceeded the dynamic cap, refuse to initiate another provider
                # call. Routes through the phase-aware terminal picker.
                if self._cost_governor.is_exceeded(ctx.op_id):
                    _cost_summary = self._cost_governor.summary(ctx.op_id) or {}
                    logger.warning(
                        "[Orchestrator] Per-op cost cap exceeded before attempt %d: "
                        "cumulative=$%.4f cap=$%.4f route=%s complexity=%s [%s]",
                        attempt + 1,
                        _cost_summary.get("cumulative_usd", 0.0),
                        _cost_summary.get("cap_usd", 0.0),
                        _cost_summary.get("route", "?"),
                        _cost_summary.get("complexity", "?"),
                        ctx.op_id,
                    )
                    _terminal = self._l2_escape_terminal(ctx.phase)
                    ctx = ctx.advance(
                        _terminal,
                        terminal_reason_code="op_cost_cap_exceeded",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {
                            "reason": "op_cost_cap_exceeded",
                            "cost_summary": dict(_cost_summary),
                            "entry_phase": "GENERATE",
                        },
                    )
                    self._cost_governor.finish(ctx.op_id)
                    return ctx

                try:
                    # Heartbeat: GENERATE phase starting (Manifesto §7: Absolute Observability)
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="generate",
                            progress_pct=30.0 + attempt * 5.0,
                        )
                    except Exception:
                        pass

                    # Route-aware generation timeout (Manifesto §5):
                    #   IMMEDIATE: 120s — fast reflex, but must accommodate Venom
                    #              tool-round + full_content patch generation.
                    #              Raised from 60s after bt-2026-04-11-085020
                    #              diagnosed tool_round=yes calls needing 63s+ to
                    #              stream 23KB+ full_content patches (first token
                    #              at 1.1s, healthy ~365 bytes/s). The 60s cap was
                    #              cutting legitimate streams off mid-output.
                    #   STANDARD:  220s — DW primary then Claude fallback in
                    #              series. Bumped from 120s after
                    #              bt-2026-04-11-211131 diagnosed every standard
                    #              op as `all_providers_exhausted`: DW eats 60-90s
                    #              of the budget then Claude only has 30-40s left
                    #              — too short for legitimate doc-gen / patch
                    #              streams (which need 60-100s). 220s = ~90s DW
                    #              cap + ~120s fallback hard cap + grace.
                    #   COMPLEX:   240s — Claude planning + DW execution serial
                    #   BACKGROUND/SPECULATIVE: 180s — no urgency
                    _route = getattr(ctx, "provider_route", "") or "standard"
                    # Per-route generation window — env-tunable so battle-test
                    # harnesses (e.g. live_fire_exploration_gate.py) can widen
                    # the architectural COMPLEX window without patching code.
                    # Defaults preserve the 2026-04-12 calibration.
                    from backend.core.ouroboros.governance.route_budgets import (  # noqa: PLC0415,E501
                        route_generation_budgets as _route_budgets,
                    )
                    _route_timeouts = _route_budgets()
                    # Slice 15 T4 — value-band adaptive allocation (parity
                    # twin of the live generate_runner seam). Fail-soft.
                    try:
                        from backend.core.ouroboros.governance.signal_value import (  # noqa: E501
                            adaptive_generation_scale as _vb_scale,
                            score_ctx as _vb_score,
                            signal_value_routing_enabled as _vb_on,
                        )
                        if _vb_on():
                            _vb_tm, _ = _vb_scale(_vb_score(ctx))
                            if _vb_tm > 1.0:
                                _route_timeouts = {
                                    k: v * _vb_tm
                                    for k, v in _route_timeouts.items()
                                }
                    except Exception:  # noqa: BLE001 — never crash routing
                        pass
                    _gen_timeout = _route_timeouts.get(
                        _route, self._config.generation_timeout_s
                    )
                    # ── Slice 231: Telemetry-Driven Budget Synthesis ──────────
                    # The per-route generation deadline is the REAL dispatch
                    # lever (budget_profile is observability-only). When the
                    # premium Claude fallback lane is economically down, an
                    # IMMEDIATE op that must drive the Iron-Gate tool loop is
                    # lifted from its reflex window (120s) to the COMPLEX-class
                    # window (240s) so the DW reroute (Slice 127 P2.1) isn't
                    # severed mid-tool-loop → kills deadline_exhausted_pre_fallback
                    # at its source. Fail-soft: any sensing fault keeps the base.
                    try:
                        from backend.core.ouroboros.governance.urgency_router import (
                            budget_synthesis_enabled as _bs_enabled,
                            synthesize_generation_timeout as _bs_gen_timeout,
                        )
                        if _bs_enabled():
                            from backend.core.ouroboros.governance.provider_availability import (
                                collect_provider_availability as _bs_collect,
                            )
                            from backend.core.ouroboros.governance.exploration_engine import (
                                exploration_gate_demands_tools as _bs_demands,
                            )
                            _bs_snap = _bs_collect()
                            _bs_tld = _bs_demands(
                                str(getattr(ctx, "task_complexity", "")),
                            )
                            _lifted = _bs_gen_timeout(
                                _route, _gen_timeout, _bs_snap,
                                tool_loop_demanded=_bs_tld,
                                elevated_timeout_s=_route_timeouts.get("complex"),
                            )
                            if _lifted > _gen_timeout:
                                logger.warning(
                                    "[BudgetSynth] route=%s claude=down:%s "
                                    "tool_loop=%s → gen_timeout lifted %.0fs→%.0fs "
                                    "(DW reroute funded) op=%s",
                                    _route, getattr(_bs_snap, "claude_reason", "?"),
                                    _bs_tld, _gen_timeout, _lifted, ctx.op_id,
                                )
                                _gen_timeout = _lifted
                    except Exception:  # noqa: BLE001 — never crash routing
                        pass
                    # Read-only BG/SPEC subagent fan-out override (Session 6,
                    # Derek 2026-04-17). The outer asyncio.wait_for at line
                    # below enforces this timeout absolutely — when the op
                    # is read-only and routed BG/SPEC, three parallel
                    # subagents can consume MAX_PARALLEL_SCOPES *
                    # PRIMARY_PROVIDER_TIMEOUT_S seconds of wall-clock before
                    # the parent Claude begins synthesis. 180s is the
                    # Session-5/6 killer. The cap-extension in candidate_
                    # generator._call_fallback is necessary but insufficient
                    # — this outer gate must also widen.
                    if (
                        bool(getattr(ctx, "is_read_only", False))
                        and _route in ("background", "speculative")
                    ):
                        try:
                            from backend.core.ouroboros.governance.subagent_contracts import (
                                MAX_PARALLEL_SCOPES,
                                PRIMARY_PROVIDER_TIMEOUT_S,
                            )
                            _fanout_budget_s = (
                                MAX_PARALLEL_SCOPES * PRIMARY_PROVIDER_TIMEOUT_S
                            )
                        except Exception:
                            _fanout_budget_s = 3 * 90  # Phase 1 Defaults
                        # Default 180s matches candidate_generator
                        # _BG_READONLY_SYNTHESIS_RESERVE_S — the two must
                        # stay aligned so the inner fallback cap and the
                        # outer orchestrator wait_for use the same reserve
                        # assumption. Session 12 empirically sized this.
                        _synthesis_reserve_s = float(os.environ.get(
                            "JARVIS_GEN_TIMEOUT_READONLY_SYNTHESIS_RESERVE_S",
                            "180",
                        ))
                        _gen_timeout_readonly = _gen_timeout + _fanout_budget_s + _synthesis_reserve_s
                        # Allow operator override via dedicated env var.
                        _gen_timeout_readonly = float(os.environ.get(
                            "JARVIS_GEN_TIMEOUT_BACKGROUND_READONLY_S",
                            str(_gen_timeout_readonly),
                        ))
                        logger.info(
                            "[Orchestrator] Read-only %s route: extending "
                            "gen_timeout %.0fs → %.0fs (fanout_budget=%.0fs, "
                            "synthesis_reserve=%.0fs) op=%s",
                            _route, _gen_timeout, _gen_timeout_readonly,
                            _fanout_budget_s, _synthesis_reserve_s, ctx.op_id,
                        )
                        _gen_timeout = _gen_timeout_readonly
                    # ── Phase R1: outer/inner timeout coherence ──────
                    # Parity with generate_runner (the LIVE phase-
                    # dispatcher path). Soak bt-2026-05-18-015317:
                    # COMPLEX outer 240s + 15s grace = 255s killed
                    # GENERATE before the inner 360s thinking window.
                    # Consume the SAME shared predicate + cap the inner
                    # fallback uses so outer >= inner by construction.
                    try:
                        from backend.core.ouroboros.governance.candidate_generator import (  # noqa: E501
                            gen_call_likely_thinking,
                            fallback_thinking_cap_s,
                        )
                        if gen_call_likely_thinking(
                            _route,
                            getattr(ctx, "task_complexity", "") or "",
                        ):
                            _r1_cap = fallback_thinking_cap_s()
                            if _r1_cap > _gen_timeout:
                                logger.info(
                                    "[Orchestrator] R1 thinking-cap "
                                    "floor: gen_timeout %.0fs → %.0fs "
                                    "route=%s op=%s", _gen_timeout,
                                    _r1_cap, _route,
                                    getattr(ctx, "op_id", "?"),
                                )
                            _gen_timeout = max(_gen_timeout, _r1_cap)
                    except Exception:  # noqa: BLE001 — fail-open
                        logger.debug(
                            "[Orchestrator] R1 thinking-cap floor "
                            "skipped (fail-open to route base)",
                            exc_info=True,
                        )
                    # ── Slice 50 Phase 2: force-batch deadline floor ──
                    # When this op will be dispatched through the DW BATCH
                    # lane (Slice 36/41 FORCE_BATCH), the provider's async
                    # batch poll legitimately runs up to
                    # JARVIS_DW_BATCH_TIMEOUT_S (Slice 43, default 300s). If
                    # the route-base GENERATE deadline is shorter than that
                    # lease (e.g. standard=220s for a trivial op the R1 floor
                    # skips), the OUTER deadline severs the async batch poll
                    # before its own lease expires — v45 probe
                    # bt-2026-06-01-034745: op-...e944 (standard/trivial) got
                    # remaining=220s, min(220,300)=220, batch killed at 220s
                    # with 300s lease runway unused (TimeoutError elapsed=220s,
                    # 0 APPLY). Floor the GENERATE window to batch_cap +
                    # overhead so the inner _call_primary hold gets the full
                    # batch lease. Mirror of the R1 thinking-cap floor above;
                    # outer >= inner by construction. Safe: Slice 36
                    # force-batch only engages when Claude is disabled (pure-DW
                    # mode), so no Claude-cascade calibration is regressed.
                    # Fail-open to route base on any error.
                    try:
                        from backend.core.ouroboros.governance.doubleword_provider import (  # noqa: E501
                            _slice36_should_force_batch,
                        )
                        from backend.core.ouroboros.governance.candidate_generator import (  # noqa: E501
                            apply_force_batch_deadline_floor,
                        )
                        if _slice36_should_force_batch(ctx):
                            _fb_floored = apply_force_batch_deadline_floor(
                                _gen_timeout, force_batch=True,
                            )
                            if _fb_floored > _gen_timeout:
                                logger.info(
                                    "[Orchestrator] Slice 50 force-batch "
                                    "deadline floor: gen_timeout %.0fs → "
                                    "%.0fs route=%s op=%s — batch lease no "
                                    "longer severed by outer deadline",
                                    _gen_timeout, _fb_floored, _route,
                                    getattr(ctx, "op_id", "?"),
                                )
                            _gen_timeout = _fb_floored
                    except Exception:  # noqa: BLE001 — fail-open
                        logger.debug(
                            "[Orchestrator] force-batch deadline floor "
                            "skipped (fail-open to route base)",
                            exc_info=True,
                        )
                    # Slice 2 — payload-adaptive GENERATE budget.
                    # Scales the FINAL route-base _gen_timeout by
                    # deterministic op-context geometry so a heavy
                    # real benchmark repo gets the headroom the
                    # trivial fixture never needed. Injected at the
                    # single highest-enforcement seam: the deadline,
                    # the outer Iron-Gate wait_for, AND the downstream
                    # tool-loop BudgetPlan all derive from this one
                    # value, so scaling here propagates coherently
                    # (no per-layer workaround). Floor = route base
                    # (zero regression), ceiling = session wall cap.
                    # Master flag default-FALSE; fail-open to base.
                    try:
                        from backend.core.ouroboros.governance.adaptive_gen_budget import (  # noqa: E501
                            scale_gen_timeout,
                        )
                        _adaptive_gt = scale_gen_timeout(
                            _gen_timeout, ctx,
                        )
                        if _adaptive_gt > _gen_timeout:
                            logger.info(
                                "[Orchestrator] adaptive gen budget: "
                                "%.0fs → %.0fs route=%s op=%s",
                                _gen_timeout, _adaptive_gt, _route,
                                getattr(ctx, "op_id", "?"),
                            )
                        _gen_timeout = _adaptive_gt
                    except Exception:  # noqa: BLE001 — fail-open
                        logger.debug(
                            "[Orchestrator] adaptive gen budget "
                            "skipped (fail-open to route base)",
                            exc_info=True,
                        )

                    # VALIDATION RESERVE. The same seam, the opposite
                    # question: not "how much does generation need" but
                    # "how much must it NOT take". VALIDATE was funded by
                    # the residue and got 0.2-16.7s against a ~12s import
                    # tax, so pytest was cut off before any verdict and
                    # every cut-off was recorded as a candidate FAILURE.
                    # Separate flag from the scaler above because that
                    # one's invariant is "never below route base" and a
                    # reserve necessarily lowers it.
                    try:
                        from backend.core.ouroboros.governance.adaptive_gen_budget import (  # noqa: E501
                            apply_validation_reserve,
                        )
                        _reserved_gt = apply_validation_reserve(
                            _gen_timeout,
                            total_budget_s=_gen_timeout,
                            route=str(_route or ""),
                            inflight=self._config.local_sibling_candidates
                            if hasattr(
                                self._config, "local_sibling_candidates",
                            ) else 1,
                        )
                        if _reserved_gt < _gen_timeout:
                            logger.info(
                                "[Orchestrator] validation reserve: gen "
                                "%.0fs → %.0fs (withheld %.0fs) route=%s "
                                "op=%s",
                                _gen_timeout, _reserved_gt,
                                _gen_timeout - _reserved_gt, _route,
                                getattr(ctx, "op_id", "?"),
                            )
                        _gen_timeout = _reserved_gt
                    except Exception:  # noqa: BLE001 — fail-open
                        logger.debug(
                            "[Orchestrator] validation reserve skipped "
                            "(fail-open to unreserved budget)",
                            exc_info=True,
                        )
                    deadline = datetime.now(tz=timezone.utc) + timedelta(
                        seconds=_gen_timeout
                    )
                    # Emit streaming=start so SerpentFlow can render the
                    # "synthesizing" header before tokens begin flowing.
                    # Provider is unknown at this point (chosen during adaptive failback).
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="generate", progress_pct=31.0,
                            streaming="start", provider="",
                        )
                    except Exception:
                        pass
                    # Operator-visible token streaming (UX Priority 2 — closes
                    # the "spinner for 2 minutes" gap). Gated on (1) the
                    # JARVIS_UI_STREAMING_ENABLED env flag (checked inside the
                    # renderer), and (2) the route: only IMMEDIATE / STANDARD /
                    # COMPLEX are operator-visible. BACKGROUND and SPECULATIVE
                    # skip — no operator is watching, and streaming serialization
                    # would waste CPU that should go to inference.
                    _stream_renderer = None
                    if _route not in ("background", "speculative"):
                        try:
                            from backend.core.ouroboros.battle_test.stream_renderer import (
                                get_stream_renderer,
                            )
                            _stream_renderer = get_stream_renderer()
                            if _stream_renderer is not None:
                                # Provider name is unknown at this point
                                # (adaptive failback chooses mid-generate).
                                # Pass empty string; the renderer's INFO line
                                # will show provider="" rather than mislabeling
                                # with task_complexity.
                                _stream_renderer.start(
                                    op_id=ctx.op_id,
                                    provider="",
                                )
                        except Exception:
                            logger.debug(
                                "[Orchestrator] stream renderer start failed",
                                exc_info=True,
                            )
                            _stream_renderer = None
                    # Hard timeout — the deadline is advisory to the generator,
                    # but asyncio.wait_for is the Iron Gate (Manifesto §6).
                    try:
                        # Phase B parallel-edge exploitation (Manifesto §2 + §3).
                        # Attempt DAG-driven fan-out first; on ANY fallback
                        # condition (flag off, no DAG, invalid DAG, edges>0,
                        # single-unit, BG route, read-only, per-unit error /
                        # timeout / noop) returns None — legacy single-stream
                        # path runs byte-identically below.
                        _parallel_gen = None
                        try:
                            from backend.core.ouroboros.governance.plan_exploit import (
                                try_parallel_generate,
                            )
                            _parallel_gen = await try_parallel_generate(
                                ctx,
                                deadline,
                                _gen_timeout,
                                self._generator,
                                outer_grace_s=_OUTER_GATE_GRACE_S,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            # Observer contract: the exploit hook must NEVER
                            # break the FSM. Any unexpected failure routes
                            # straight to the legacy path.
                            logger.debug(
                                "[Orchestrator] plan_exploit fan-out raised — "
                                "falling back to legacy generate",
                                exc_info=True,
                            )
                            _parallel_gen = None

                        if _parallel_gen is not None:
                            generation = _parallel_gen
                        else:
                            generation = await asyncio.wait_for(
                                self._generator.generate(ctx, deadline),
                                timeout=_gen_timeout + _OUTER_GATE_GRACE_S,
                            )
                    finally:
                        # End the stream regardless of success / failure so the
                        # Live widget closes and the observability INFO line
                        # emits TTFT + TPS even when generation times out.
                        if _stream_renderer is not None:
                            try:
                                _stream_renderer.end()
                            except Exception:
                                logger.debug(
                                    "[Orchestrator] stream renderer end failed",
                                    exc_info=True,
                                )
                    # Charge the CostGovernor with the actual generation cost.
                    # Non-positive costs (cache hits, fallback stubs) are a no-op.
                    try:
                        _cost_this_call = float(getattr(generation, "cost_usd", 0.0) or 0.0)
                        _prov_name = getattr(generation, "provider_name", "") or ""
                        if _cost_this_call > 0.0:
                            # Slice 2 of Per-Phase Cost Drill-Down arc:
                            # tag charge with current phase so the operator
                            # can answer "why did this op cost $X" per-phase.
                            _phase_tag = getattr(
                                getattr(ctx, "phase", None), "name", "",
                            ) or ""
                            self._cost_governor.charge(
                                ctx.op_id, _cost_this_call, _prov_name,
                                phase=_phase_tag,
                            )
                            await self._emit_route_cost_heartbeat(
                                ctx,
                                cost_usd=_cost_this_call,
                                provider=_prov_name,
                                route=getattr(ctx, "provider_route", "") or "standard",
                                cost_event="generation_attempt",
                            )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] CostGovernor.charge failed", exc_info=True,
                        )
                    # Emit streaming=end to close the streaming block
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="generate", progress_pct=49.0,
                            streaming="end",
                        )
                    except Exception:
                        pass

                    # is_noop=True means the model signalled the change is already present.
                    # Empty candidates is correct in this case — do not treat as a failure.
                    if generation is not None and generation.is_noop:
                        break

                    if generation is None or len(generation.candidates) == 0:
                        generation = None
                        raise RuntimeError("no_candidates_returned")

                    # ── Forward-progress detector ──
                    # Hash the first candidate's content and flag if the
                    # retry loop is producing the same candidate repeatedly.
                    # A trip means we're burning retries without any actual
                    # change — escape the loop via the phase-aware terminal.
                    try:
                        _fp_hash = candidate_content_hash(generation.candidates[0])
                        if _fp_hash and self._forward_progress.observe(
                            ctx.op_id, _fp_hash,
                        ):
                            _fp_summary = self._forward_progress.summary(ctx.op_id) or {}
                            logger.warning(
                                "[Orchestrator] Forward-progress trip: op=%s "
                                "stuck after %d repeats — escaping retry loop",
                                ctx.op_id,
                                _fp_summary.get("repeat_count", 0),
                            )
                            _terminal = self._l2_escape_terminal(ctx.phase)
                            ctx = ctx.advance(
                                _terminal,
                                terminal_reason_code="no_forward_progress",
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": "no_forward_progress",
                                    "progress_summary": dict(_fp_summary),
                                    "entry_phase": "GENERATE",
                                },
                            )
                            self._forward_progress.finish(ctx.op_id)
                            return ctx
                    except Exception:
                        logger.debug(
                            "[Orchestrator] ForwardProgress.observe failed",
                            exc_info=True,
                        )

                    # ── Productivity-ratio detector (EC9) ──
                    # Complements EC8: EC8 catches byte-identical repetition;
                    # EC9 catches *semantic* stagnation — candidates whose
                    # normalized form (AST dump / canonical JSON / whitespace-
                    # stripped) hasn't changed while the model keeps charging
                    # us for retries. Trip = $ burned since last semantic
                    # change exceeded the threshold AND we've seen enough
                    # stable observations. Escape via phase-aware terminal.
                    try:
                        _pd_hash = productivity_content_hash(
                            generation.candidates[0],
                            level=self._productivity_detector.level,
                        )
                        if _pd_hash and self._productivity_detector.observe(
                            ctx.op_id, _cost_this_call, _pd_hash,
                        ):
                            _pd_summary = self._productivity_detector.summary(ctx.op_id) or {}
                            logger.warning(
                                "[Orchestrator] Productivity stall: op=%s "
                                "burned=$%.4f stable=%d level=%s — escaping retry loop",
                                ctx.op_id,
                                _pd_summary.get("cost_since_last_change_usd", 0.0),
                                _pd_summary.get("consecutive_stable", 0),
                                _pd_summary.get("config", {}).get("normalization_level", "?"),
                            )
                            _terminal = self._l2_escape_terminal(ctx.phase)
                            ctx = ctx.advance(
                                _terminal,
                                terminal_reason_code="stalled_productivity",
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": "stalled_productivity",
                                    "productivity_summary": dict(_pd_summary),
                                    "entry_phase": "GENERATE",
                                },
                            )
                            self._productivity_detector.finish(ctx.op_id)
                            return ctx
                    except Exception:
                        logger.debug(
                            "[Orchestrator] ProductivityDetector.observe failed",
                            exc_info=True,
                        )

                    # ── Iron Gate: deterministic post-generation quality checks ──
                    # Manifesto §6: agentic intelligence proposes, deterministic
                    # code validates. These checks hard-fail BEFORE validation
                    # adapters run, routing back through the GENERATE retry loop
                    # with explicit error feedback so the model learns in-flight.
                    #
                    # Gate 1 — Exploration-first enforcement (no patch without
                    # reading the codebase). Trivial ops bypass (small-surface
                    # rewrites don't need the floor).
                    #
                    # Complexity-scaled threshold (bt-2026-04-11-090651 root cause):
                    # simple ops (single target file, mechanical change) need only
                    # 1 exploration call — one read_file IS reading the codebase.
                    # moderate/complex ops keep the 2-call floor because they
                    # touch multiple surfaces. Claude-sonnet-4-6 reliably refused
                    # retry feedback on simple ops ("1/2 → 0/2") because the
                    # exploration demand didn't match the task size; scaling by
                    # complexity restores intent-alignment while preserving the
                    # gate's purpose.
                    _task_complexity = getattr(ctx, "task_complexity", "") or ""
                    _EXPLORATION_TOOLS = frozenset({
                        "read_file", "search_code", "get_callers", "list_symbols",
                        "glob_files", "list_dir",
                    })
                    _env_min = os.environ.get("JARVIS_MIN_EXPLORATION_CALLS")
                    if _env_min is not None:
                        _min_explore = int(_env_min)
                    elif _task_complexity == "simple":
                        _min_explore = 1
                    else:
                        _min_explore = 2
                    # Slice 12P Phase 1 — envelope-aware exploration
                    # discipline. SWE-Bench-Pro wiring-validation
                    # fixtures (those with gold_patch="" AND
                    # metadata.real_benchmark=False) are structurally
                    # designed to score PASS with a no-op patch, so
                    # forcing 2+ exploration calls before they can
                    # propose that no-op patch creates the
                    # exploration_insufficient ↔ no-op-patch
                    # deadlock that killed bt-2026-05-23-030130.
                    # Drop the floor to 0 ONLY for fixtures — real
                    # benchmark problems (gold_patch != "") still
                    # require the full exploration discipline. Pure
                    # envelope-metadata composition; no hardcoded
                    # instance_ids.
                    try:
                        from backend.core.ouroboros.governance.envelope_metadata import (  # noqa: E501
                            is_wiring_validation_envelope as _slice12p_is_fixture,
                        )
                        if _slice12p_is_fixture(ctx):
                            logger.info(
                                "[Orchestrator] Iron Gate — Slice 12P "
                                "envelope-aware override: wiring-validation "
                                "fixture detected (swe_bench_pro=true, "
                                "gold_patch_empty=true, real_benchmark=false) "
                                "— exploration floor 0 for op=%s",
                                ctx.op_id[:12],
                            )
                            _min_explore = 0
                    except Exception:  # noqa: BLE001 — defensive
                        logger.debug(
                            "[Orchestrator] Slice 12P envelope-aware check "
                            "raised — falling through to pre-Slice-12P floor",
                            exc_info=True,
                        )
                    _explore_gate_enabled = (
                        os.environ.get("JARVIS_EXPLORATION_GATE", "true").lower() == "true"
                        and _task_complexity != "trivial"
                        # Slice 12P — when the envelope-aware override
                        # drops the floor to 0, the gate itself becomes
                        # a no-op (nothing to enforce). Skip the gate
                        # path entirely to avoid emitting confusing
                        # "Iron Gate" log lines for fixtures that have
                        # no enforcement to apply.
                        and _min_explore > 0
                    )
                    if _explore_gate_enabled:
                        _explore_count = sum(
                            1 for _rec in (generation.tool_execution_records or ())
                            if getattr(_rec, "tool_name", "") in _EXPLORATION_TOOLS
                        )
                        # Preloaded-prompt credit: when the lean prompt builder
                        # inlines target regions directly into the generation
                        # prompt, the model has already "seen" those files without
                        # needing a read_file tool call — semantically equivalent
                        # exploration. Gives DW BACKGROUND route (no tool loop)
                        # and simple/trivial ops a fair path through the gate.
                        _preloaded_credit = len(
                            getattr(generation, "prompt_preloaded_files", ()) or ()
                        )
                        # Roll the per-attempt count into the per-op credit BEFORE
                        # comparing — a prior attempt that already satisfied the
                        # floor lets a no-tool retry pass (the rejected file is
                        # already in the retry-feedback prompt).
                        _op_explore_credit += _explore_count + _preloaded_credit

                        # Accumulate ledger records across retry attempts (#103).
                        # Cumulative semantics mirror _op_explore_credit — the
                        # ledger sees every tool call the model has made for this
                        # op, then dedup-by-(tool, arguments_hash) happens inside
                        # diversity_score(). Preloaded files become synthetic
                        # read_file records so the ledger grants comprehension
                        # credit matching the legacy counter's preload behavior.
                        _op_explore_records.extend(
                            generation.tool_execution_records or ()
                        )
                        for _pf in (
                            getattr(generation, "prompt_preloaded_files", ()) or ()
                        ):
                            _op_explore_records.append(
                                _PreloadedExplorationRecord(str(_pf))
                            )

                        from backend.core.ouroboros.governance.exploration_engine import (  # noqa: E501
                            ExplorationFloors,
                            ExplorationInsufficientError,
                            ExplorationLedger,
                            evaluate_exploration,
                            is_ledger_enabled,
                        )

                        if is_ledger_enabled():
                            # ── DECISION path (#103) ──
                            # Ledger is authoritative. Legacy int-counter gate is
                            # skipped entirely. Emit ``(decision)`` log tag — kept
                            # distinct from ``(shadow)`` so ops can grep either
                            # mode without ambiguity.
                            try:
                                _ledger = ExplorationLedger.from_records(
                                    _op_explore_records
                                )
                                _floors = ExplorationFloors.from_env_with_adapted(_task_complexity)
                                _verdict = evaluate_exploration(_ledger, _floors)
                            except Exception:
                                # If the ledger itself blows up, fall through to
                                # the legacy counter gate so we never leave the op
                                # ungated. Log once so the failure is visible.
                                logger.exception(
                                    "[Orchestrator] ExplorationLedger(decision) "
                                    "evaluation failed — falling back to counter"
                                )
                                _verdict = None
                            if _verdict is not None:
                                _covered_names = sorted(
                                    c.value for c in _verdict.categories_covered
                                )
                                logger.info(
                                    "[Orchestrator] ExplorationLedger(decision) "
                                    "op=%s complexity=%s score=%.2f min_score=%.2f "
                                    "unique=%d categories=%s would_pass=%s",
                                    ctx.op_id[:12],
                                    _task_complexity or "unknown",
                                    _verdict.score,
                                    _floors.min_score,
                                    _ledger.unique_call_count(),
                                    ",".join(_covered_names) or "-",
                                    _verdict.sufficient,
                                )
                                if _verdict.insufficient:
                                    _missing = sorted(
                                        c.value for c in _verdict.missing_categories
                                    )
                                    _decision_msg = (
                                        f"exploration_insufficient: "
                                        f"score={_verdict.score:.1f}/"
                                        f"{_floors.min_score:.1f} "
                                        f"categories={len(_verdict.categories_covered)}/"
                                        f"{_floors.min_categories} "
                                        f"missing={','.join(_missing) or '-'}"
                                    )
                                    logger.warning(
                                        "[Orchestrator] Iron Gate — "
                                        "ExplorationLedger(decision) insufficient "
                                        "op=%s %s (attempt=%d)",
                                        ctx.op_id[:12],
                                        _decision_msg,
                                        attempt + 1,
                                    )
                                    # Slice 21 Fix A2 — same structural facts
                                    # on the ledger-path exception. The
                                    # ledger's unique_call_count includes
                                    # synthetic preload records, so 0 means
                                    # ZERO credit of any kind.
                                    _ledger_exc = ExplorationInsufficientError(
                                        _decision_msg,
                                        verdict=_verdict,
                                        floors=_floors,
                                    )
                                    _ledger_exc.structural_credit = int(  # type: ignore[attr-defined]
                                        _ledger.unique_call_count()
                                    )
                                    _ledger_exc.rejected_model_id = str(  # type: ignore[attr-defined]
                                        getattr(generation, "model_id", "") or ""
                                    )
                                    generation = None
                                    raise _ledger_exc
                                # Ledger PASSED — skip legacy counter gate
                                # entirely. Jump to the ASCII gate below.
                            else:
                                # Ledger eval crashed → fall through to legacy gate
                                pass

                        # ── LEGACY path (flag off) or ledger-eval fallback ──
                        # Shadow log + int-counter gate. Shadow log is suppressed
                        # when enforcement is on (the decision log above covers
                        # that path) so operators don't see duplicate lines.
                        if not is_ledger_enabled():
                            _shadow_on = (
                                os.environ.get(
                                    "JARVIS_EXPLORATION_SHADOW_LOG", "",
                                ).strip().lower() in _TRUTHY
                            )
                            if _shadow_on:
                                try:
                                    _sledger = ExplorationLedger.from_records(
                                        _op_explore_records
                                    )
                                    _sfloors = ExplorationFloors.from_env_with_adapted(
                                        _task_complexity
                                    )
                                    _sverdict = evaluate_exploration(
                                        _sledger, _sfloors
                                    )
                                    _scovered = sorted(
                                        c.value for c in _sverdict.categories_covered
                                    )
                                    logger.info(
                                        "[Orchestrator] ExplorationLedger(shadow) "
                                        "op=%s complexity=%s legacy_credit=%d "
                                        "score=%.2f min_score=%.2f unique=%d "
                                        "categories=%s would_pass=%s",
                                        ctx.op_id[:12],
                                        _task_complexity or "unknown",
                                        _op_explore_credit,
                                        _sverdict.score,
                                        _sfloors.min_score,
                                        _sledger.unique_call_count(),
                                        ",".join(_scovered) or "-",
                                        _sverdict.sufficient,
                                    )
                                except Exception:
                                    logger.debug(
                                        "[Orchestrator] ExplorationLedger shadow "
                                        "log error",
                                        exc_info=True,
                                    )

                        if (
                            not is_ledger_enabled()
                            and _op_explore_credit < _min_explore
                        ):
                            _explore_err = (
                                f"exploration_insufficient: {_op_explore_credit}/{_min_explore} "
                                f"exploration tool calls (expected >= {_min_explore}). "
                                f"You MUST call read_file/search_code/get_callers at least "
                                f"{_min_explore} times BEFORE proposing any patch. "
                                f"Use the tool loop to read the target file and grep for "
                                f"callers, then return your patch."
                            )
                            logger.warning(
                                # Slice 11: FULL op id — this line is an
                                # audit-keyed REJECT marker; [:12] cuts at
                                # the UUIDv7 same-millisecond boundary and
                                # poisoned 3 flags on an ambiguous prefix
                                # (Run-21 false-red class).
                                "[Orchestrator] Iron Gate — exploration_insufficient: "
                                "%d/%d (attempt=%d cumulative, preloaded=%d) for op=%s",
                                _op_explore_credit, _min_explore, attempt + 1,
                                _preloaded_credit, ctx.op_id,
                            )
                            # Slice 230 — feed the rejection back into model
                            # rotation: drift-mark the model that produced this
                            # no-tool candidate so the GENERATE_RETRY walk skips
                            # it and rotates to the next ranked (agentic) model.
                            _slice230_record_exploration_drift(
                                ctx.op_id,
                                getattr(generation, "model_id", ""),
                            )
                            # Slice 21 Fix A2 — carry the structural facts the
                            # retry handler's capability check needs: total
                            # exploration credit (tools + preload, cumulative)
                            # and the model that produced the rejected attempt.
                            _explore_exc = RuntimeError(_explore_err)
                            _explore_exc.structural_credit = int(_op_explore_credit)  # type: ignore[attr-defined]
                            _explore_exc.rejected_model_id = str(  # type: ignore[attr-defined]
                                getattr(generation, "model_id", "") or ""
                            )
                            generation = None
                            raise _explore_exc

                    # Gate 2 — ASCII/Unicode strictness (prevent rapidفuzz-class
                    # typos where model emits non-ASCII code points in identifier
                    # positions). Deterministic scan; O(n) on candidate size.
                    # Delegates to AsciiStrictGate which:
                    #   1) auto-repairs common punctuation drift (em-dash →
                    #      hyphen, curly quotes → straight, ellipsis → ...,
                    #      nbsp → space, zero-width strip) IN-PLACE on the
                    #      candidate dict — healing the deterministic training-
                    #      data artifact where Claude always inserts U+2014 at
                    #      the same byte offset of requirements.txt.
                    #   2) hard-rejects any residue (Unicode letters in
                    #      identifier positions, unlisted symbols) per the
                    #      original Iron Gate contract.
                    _ascii_gate = AsciiStrictGate()
                    if _ascii_gate.enabled:
                        for _cand in generation.candidates:
                            _ok, _ascii_err, _bad_list = _ascii_gate.check(_cand)
                            _repairs = _cand.get("_ascii_repair_count", 0) if isinstance(_cand, dict) else 0
                            if _repairs:
                                logger.info(
                                    "[Orchestrator] Iron Gate — ascii_auto_repaired: "
                                    "%d codepoint(s) healed file=%s op=%s",
                                    _repairs,
                                    _cand.get("file_path", "?") if isinstance(_cand, dict) else "?",
                                    ctx.op_id[:12],
                                )
                            if not _ok:
                                _samples_str = ", ".join(
                                    bc.format_sample() for bc in _bad_list
                                )
                                logger.warning(
                                    "[Orchestrator] Iron Gate — ascii_corruption: "
                                    "%d offender(s) [%s] op=%s",
                                    len(_bad_list), _samples_str, ctx.op_id[:12],
                                )
                                # Stash the rejected content + offenders on the
                                # exception so the retry feedback builder can
                                # extract the specific offending lines and show
                                # them back to the model in context. Without
                                # this, the model only sees "U+0641 at L106:C6"
                                # which isn't enough to locate the bad identifier
                                # in a 200-line file.
                                _rejected_content = ""
                                if isinstance(_cand, dict):
                                    _rejected_content = (
                                        _cand.get("full_content", "")
                                        or _cand.get("raw_content", "")
                                        or ""
                                    )
                                    if not _rejected_content and isinstance(_cand.get("files"), list):
                                        # Multi-file shape — grab the first file matching an offender
                                        _bad_path = _bad_list[0].file_path if _bad_list else ""
                                        for _entry in _cand["files"]:
                                            if isinstance(_entry, dict) and _entry.get("file_path") == _bad_path:
                                                _rejected_content = _entry.get("full_content", "") or ""
                                                break
                                generation = None
                                _ascii_exc = RuntimeError(_ascii_err or "ascii_corruption")
                                # Private attributes — read back in the retry feedback builder.
                                _ascii_exc._ascii_bad_codepoints = _bad_list  # type: ignore[attr-defined]
                                _ascii_exc._ascii_rejected_content = _rejected_content  # type: ignore[attr-defined]
                                raise _ascii_exc

                    # Gate 4 — Target existence (Slice 72). For a benchmark op
                    # the candidate MUST target a file that already exists in the
                    # prepared worktree. Catches the bt-2026-06-03 generation-
                    # steering failure where the Claude fallback, after a rushed
                    # single exploration round, emitted a HOST path
                    # (backend/core/process_manager.py) for a qutebrowser repo →
                    # APPLY ENOENT. Routes the miss back to GENERATE_RETRY with
                    # self-correcting feedback instead of crashing APPLY. INERT
                    # for every non-swe_bench op (host self-dev legitimately
                    # creates new files) and when write_root can't be resolved.
                    # Universal mode (2026-07-21, soak bt-2026-07-21-230753):
                    # the benchmark-only gating left host self-dev ops
                    # unguarded, so a write-root-DOUBLED candidate path
                    # sailed to APPLY and hard-ENOENT'd in the ChangeEngine.
                    # Universal mode gates ALL ops with the new-file lane
                    # preserved (allow_new_files: a missing target is only a
                    # steering error when its PARENT dir is also missing).
                    # Write root: benchmark keeps _swe_bench_write_root;
                    # host ops use the canonical Slice 11 execution-root seam
                    # (the SAME root ChangeEngine._effective_write_root
                    # resolves, so gate and engine provably agree). The
                    # existence stats run OFF-LOOP via asyncio.to_thread —
                    # a cold-FS stat must never starve the event loop.
                    _tg_is_benchmark = (
                        getattr(ctx, "signal_source", "") == "swe_bench_pro"
                    )
                    _tg_missing = []
                    if (
                        (_tg_is_benchmark and _target_guard_enabled())
                        or (not _tg_is_benchmark
                            and _target_guard_universal_enabled())
                    ):
                        # Fail-SOFT infrastructure resolution (mirrors the
                        # SHIPPING generate_runner twin): a host that
                        # cannot resolve a write root skips the check —
                        # the deliberate raise stays OUTSIDE the try.
                        try:
                            _tg_write_root = (
                                self._swe_bench_write_root(ctx)
                                if _tg_is_benchmark
                                else self._config.execution_root
                            )
                            _tg_missing = await asyncio.to_thread(
                                _find_missing_targets,
                                generation.candidates,
                                _tg_write_root,
                                allow_new_files=not _tg_is_benchmark,
                            )
                        except Exception:  # noqa: BLE001 — protective gate
                            logger.debug(
                                "[Orchestrator] Iron Gate — target-"
                                "existence infrastructure unresolvable; "
                                "gate skipped op=%s",
                                ctx.op_id[:12], exc_info=True,
                            )
                            _tg_missing = []
                            _tg_write_root = None
                        if _tg_missing:
                            logger.warning(
                                "[Orchestrator] Iron Gate — target_file_missing: "
                                "%s not in worktree %s op=%s (attempt=%d, "
                                "lane=%s)",
                                ",".join(_tg_missing), _tg_write_root,
                                ctx.op_id[:12], attempt + 1,
                                "benchmark" if _tg_is_benchmark else "universal",
                            )
                            generation = None
                            raise RuntimeError(
                                _target_missing_error_message(_tg_missing)
                            )

                    # Gate 4.5 — Source-domain purity (Slice 78). For a swe_bench
                    # op, a candidate that modifies ONLY test files is the "cheat
                    # the held-out suite" failure mode. SCORE_REJECT_TEST_MODS
                    # rejects it post-hoc — but only AFTER the op burned its full
                    # GENERATE + APPLY + scoring budget. Catch it HERE and route
                    # to GENERATE_RETRY telling the model to fix the SOURCE defect
                    # instead. INERT for non-swe_bench ops (host self-dev
                    # legitimately authors tests) and when the guard is disabled.
                    # Naming-heuristic only (no ctx.target_files) so a source file
                    # is never misclassified as a test.
                    if getattr(ctx, "signal_source", "") == "swe_bench_pro":
                        try:
                            from backend.core.ouroboros.governance.patch_domain_guard import (  # noqa: E501
                                PatchDomainGuard as _PatchDomainGuard,
                            )
                        except ImportError:
                            _PatchDomainGuard = None  # type: ignore[assignment]
                        if _PatchDomainGuard is not None:
                            _domain_guard = _PatchDomainGuard()
                            if _domain_guard.enabled:
                                for _cand in generation.candidates:
                                    _mod_paths: List[str] = []
                                    if isinstance(_cand, dict):
                                        if _cand.get("file_path"):
                                            _mod_paths.append(str(_cand["file_path"]))
                                        for _entry in (_cand.get("files") or []):
                                            if (
                                                isinstance(_entry, dict)
                                                and _entry.get("file_path")
                                            ):
                                                _mod_paths.append(
                                                    str(_entry["file_path"])
                                                )
                                    _dg_ok, _dg_reason, _dg_verdict = (
                                        _domain_guard.check(_mod_paths, None)
                                    )
                                    if not _dg_ok:
                                        logger.warning(
                                            "[Orchestrator] Iron Gate — "
                                            "patch_domain_test_only: candidate "
                                            "modifies only tests [%s] op=%s "
                                            "(attempt=%d)",
                                            ", ".join(_dg_verdict.test_files),
                                            ctx.op_id[:12], attempt + 1,
                                        )
                                        generation = None
                                        raise RuntimeError(
                                            _dg_reason or "patch_domain_test_only"
                                        )

                    # Gate 3 — Dependency file integrity. Catches hallucinated
                    # package-name renames/truncations in requirements.txt (and
                    # future: package.json, Cargo.toml, etc.). Engineered in
                    # response to bt-2026-04-10-184157, where Claude emitted a
                    # requirements.txt patch renaming ``anthropic`` →
                    # ``anthropichttp`` and ``rapidfuzz`` → ``rapidfu`` — two
                    # pure-ASCII corruptions that slipped past every other gate.
                    try:
                        from backend.core.ouroboros.governance.dependency_file_gate import (
                            check_candidate as _dep_check,
                        )
                    except ImportError:
                        _dep_check = None  # type: ignore[assignment]
                    if _dep_check is not None:
                        for _cand in generation.candidates:
                            _dep_result = _dep_check(_cand, self._config.project_root)
                            if _dep_result is None:
                                continue
                            _dep_reason, _dep_offenders = _dep_result
                            logger.warning(
                                "[Orchestrator] Iron Gate — dependency_file_integrity: "
                                "%d offender(s) [%s] op=%s",
                                len(_dep_offenders),
                                ", ".join(_dep_offenders[:5]),
                                ctx.op_id[:12],
                            )
                            # Extract the rejected content for retry feedback.
                            _rejected_content = ""
                            if isinstance(_cand, dict):
                                _rejected_content = _cand.get("full_content", "") or ""
                                if not _rejected_content and isinstance(_cand.get("files"), list):
                                    for _entry in _cand["files"]:
                                        if not isinstance(_entry, dict):
                                            continue
                                        _ep = _entry.get("file_path", "") or ""
                                        from backend.core.ouroboros.governance.dependency_file_gate import is_dependency_file
                                        if is_dependency_file(_ep):
                                            _rejected_content = _entry.get("full_content", "") or ""
                                            break
                            generation = None
                            _dep_exc = RuntimeError(_dep_reason)
                            # Private attributes — retry feedback builder reads these.
                            _dep_exc._dep_file_offenders = _dep_offenders  # type: ignore[attr-defined]
                            _dep_exc._dep_file_rejected_content = _rejected_content  # type: ignore[attr-defined]
                            raise _dep_exc

                    # Gate 4 — Docstring multi-line collapse detection. Catches
                    # the regression where Claude rewrites a multi-line module
                    # or function docstring as a single-line literal containing
                    # ``\n`` escape sequences (bt-2026-04-11-211131,
                    # headless_cli.py). Valid Python that breaks every reader.
                    try:
                        from backend.core.ouroboros.governance.docstring_collapse_gate import (
                            check_candidate as _docstring_check,
                        )
                    except ImportError:
                        _docstring_check = None  # type: ignore[assignment]
                    if _docstring_check is not None:
                        for _cand in generation.candidates:
                            _ds_result = _docstring_check(_cand, self._config.project_root)
                            if _ds_result is None:
                                continue
                            _ds_reason, _ds_offenders = _ds_result
                            logger.warning(
                                "[Orchestrator] Iron Gate — docstring_collapse: "
                                "%d offender(s) [%s] op=%s",
                                len(_ds_offenders),
                                ", ".join(_ds_offenders[:5]),
                                ctx.op_id[:12],
                            )
                            _rejected_content = ""
                            if isinstance(_cand, dict):
                                _rejected_content = _cand.get("full_content", "") or ""
                                if not _rejected_content and isinstance(_cand.get("files"), list):
                                    for _entry in _cand["files"]:
                                        if isinstance(_entry, dict) and (
                                            _entry.get("file_path", "") or ""
                                        ).endswith(".py"):
                                            _rejected_content = _entry.get("full_content", "") or ""
                                            break
                            generation = None
                            _ds_exc = RuntimeError(_ds_reason)
                            _ds_exc._docstring_collapse_offenders = _ds_offenders  # type: ignore[attr-defined]
                            _ds_exc._docstring_collapse_rejected_content = _rejected_content  # type: ignore[attr-defined]
                            raise _ds_exc

                    # Gate 5 — Multi-file coverage. Session O (bt-2026-04-15-
                    # 175547) closed the full governed APPLY arc but only 1
                    # of 4 target files landed on disk because the winning
                    # candidate returned legacy {file_path, full_content}
                    # instead of {files: [...]}, so _apply_multi_file_candidate
                    # was never invoked. This gate rejects any multi-target op
                    # whose candidate fails to cover every path in
                    # context.target_files via a populated files: [...] list.
                    # The retry-feedback builder names the missing paths and
                    # reiterates the multi-file contract. Master switch:
                    # JARVIS_MULTI_FILE_ENFORCEMENT (default true).
                    try:
                        from backend.core.ouroboros.governance.multi_file_coverage_gate import (
                            check_candidate as _mf_check,
                        )
                    except ImportError:
                        _mf_check = None  # type: ignore[assignment]
                    if _mf_check is not None:
                        for _cand in generation.candidates:
                            _mf_result = _mf_check(
                                _cand,
                                ctx.target_files,
                                self._config.project_root,
                                intake_evidence_json=getattr(
                                    ctx, "intake_evidence_json", "",
                                ) or "",
                            )
                            if _mf_result is None:
                                continue
                            _mf_reason, _mf_missing = _mf_result
                            logger.warning(
                                "[Orchestrator] Iron Gate — multi_file_coverage: "
                                "missing %d/%d [%s] op=%s",
                                len(_mf_missing),
                                len(ctx.target_files),
                                ", ".join(_mf_missing[:5]),
                                ctx.op_id[:12],
                            )
                            generation = None
                            _mf_exc = RuntimeError(_mf_reason)
                            # Private attributes — retry feedback builder reads these.
                            _mf_exc._mf_missing_paths = _mf_missing  # type: ignore[attr-defined]
                            _mf_exc._mf_target_files = tuple(ctx.target_files)  # type: ignore[attr-defined]
                            raise _mf_exc

                    # Heartbeat: generation succeeded with candidates
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="generate",
                            progress_pct=50.0,
                        )
                        # Also emit rich payload for BattleDiffTransport
                        _gen_msg = type(
                            "_Msg", (), {
                                "payload": {
                                    "phase": "generate",
                                    "candidates_count": len(generation.candidates),
                                    "provider": generation.provider_name,
                                    "model_id": getattr(generation, "model_id", ""),
                                    "generation_duration_s": generation.generation_duration_s,
                                    "tool_records": len(getattr(generation, "tool_execution_records", ()) or ()),
                                    "total_input_tokens": getattr(generation, "total_input_tokens", 0),
                                    "total_output_tokens": getattr(generation, "total_output_tokens", 0),
                                    "cost_usd": getattr(generation, "cost_usd", 0.0),
                                    # Include candidate file paths and preview for TUI display
                                    "candidate_files": [
                                        getattr(c, "file_path", "") for c in generation.candidates[:3]
                                    ],
                                    "candidate_rationales": [
                                        (c.get("rationale", "") or "")[:80]
                                        for c in generation.candidates[:3]
                                    ],
                                    "candidate_preview": (
                                        getattr(generation.candidates[0], "raw_content", "")[:500]
                                        if generation.candidates else ""
                                    ),
                                },
                                "op_id": ctx.op_id,
                                "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                            },
                        )()
                        for _t in getattr(self._stack.comm, "_transports", []):
                            try:
                                await _t.send(_gen_msg)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Success -- record reasoning trace + dialogue
                    if self._reasoning_narrator is not None:
                        try:
                            self._reasoning_narrator.record_generate(
                                ctx.op_id, generation.provider_name,
                                len(generation.candidates), generation.generation_duration_s,
                            )
                        except Exception:
                            pass
                    if self._dialogue_store is not None:
                        try:
                            _d = self._dialogue_store.get_active(ctx.op_id)
                            if _d:
                                _d.add_entry(
                                    "GENERATE",
                                    f"{generation.provider_name} produced {len(generation.candidates)} "
                                    f"candidates in {generation.generation_duration_s:.1f}s",
                                )
                        except Exception:
                            pass

                    # Success -- break out of retry loop
                    break

                except GovernanceDeadlockError as _dl_exc:
                    # Sovereign Epistemic Context Matrix (LR3): the one-shot
                    # governance-deadlock breaker raised mid-Venom and could
                    # not recover. This is a NON-RETRYABLE terminal — retrying
                    # a wedged governance deadlock just re-wedges. Terminate
                    # the op with deadlock_override_failed; do NOT fall through
                    # to the generic retry/demotion path below.
                    logger.warning(
                        "[Orchestrator] Governance deadlock override failed "
                        "for %s: %s — terminating (non-retryable)",
                        ctx.op_id, _dl_exc,
                    )
                    _reason = "deadlock_override_failed"
                    # Authoritative non-retry registry (single source of truth).
                    if _is_nonretryable_terminal(_reason):
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code=_reason,
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": _reason,
                                "error": str(_dl_exc)[:200],
                                "nonretryable": True,
                            },
                        )
                        return ctx

                except Exception as exc:
                    _err_msg = str(exc)
                    _route = getattr(ctx, "provider_route", "")

                    # ── Fail-Fast Exhaustion Circuit Breaker ──
                    # §33.1 default-FALSE. When ON: an op that raises
                    # all_providers_exhausted for N consecutive
                    # attempts (counter keyed by stable op_id so it
                    # survives Stage-1.6 park/resume re-dispatch) is
                    # flipped to a terminal `failed` state HERE —
                    # preempting the GENERATE_RETRY/park cycle below.
                    # _record_ledger(FAILED) is a TERMINAL_OPERATION_
                    # STATE → publish_operation_terminal fires → an
                    # awaiting B.2.2 subscriber wakes in seconds
                    # instead of burning the full 1800s eval window.
                    # `failed` (not `blocked`) keeps it natively
                    # retryable in a future run (the exhaustion is an
                    # environmental transient, not a policy block).
                    # Mirrors the existing background-failure
                    # fast-terminal precedent (advance → _record_ledger
                    # → return ctx). When OFF: zero counter mutation,
                    # byte-identical legacy behaviour.
                    if (
                        _failfast_cb_enabled()
                        and "all_providers_exhausted" in _err_msg
                    ):
                        _ff_key = str(getattr(ctx, "op_id", "") or "")
                        _ff_n = self._failfast_exhaust_consec.get(
                            _ff_key, 0,
                        ) + 1
                        self._failfast_exhaust_consec[_ff_key] = _ff_n
                        if _ff_n >= _failfast_cb_threshold():
                            logger.error(
                                "[Orchestrator] Fail-Fast circuit OPEN "
                                "for %s: %d consecutive "
                                "all_providers_exhausted ≥ threshold "
                                "%d — terminal failed (instant, no "
                                "30-min thrash) [%s]",
                                _ff_key, _ff_n,
                                _failfast_cb_threshold(), _ff_key,
                            )
                            self._failfast_exhaust_consec.pop(
                                _ff_key, None,
                            )
                            ctx = ctx.advance(
                                OperationPhase.POSTMORTEM,
                                terminal_reason_code=(
                                    _FAILFAST_CIRCUIT_OPEN_REASON
                                ),
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": (
                                        _FAILFAST_CIRCUIT_OPEN_REASON
                                    ),
                                    "failure_class": "infra",
                                    "consecutive_exhaustions": _ff_n,
                                    "error": _err_msg[:200],
                                },
                            )
                            return ctx

                    # ── Partial shadow log (widened) ──
                    # Fire the ExplorationLedger shadow pass for EVERY
                    # generation failure, regardless of route/cause. The
                    # original BG-DW-only branch missed failure modes like
                    # doubleword_schema_invalid, all_providers_exhausted,
                    # APITimeout. We classify the cause from _err_msg so the
                    # log line still tells you what killed the attempt, and
                    # we pull whatever tool_execution_records are reachable
                    # off the exception (may be empty). No-op when shadow
                    # logging is off so this stays free in production.
                    _shadow_on_partial = (
                        os.environ.get(
                            "JARVIS_EXPLORATION_SHADOW_LOG", "",
                        ).strip().lower() in {"1", "true", "yes", "on"}
                    )
                    if _shadow_on_partial:
                        try:
                            from backend.core.ouroboros.governance.exploration_engine import (  # noqa: E501
                                ExplorationFloors,
                                ExplorationLedger,
                                evaluate_exploration,
                            )
                            _partial_records = getattr(
                                exc, "tool_execution_records", ()
                            ) or ()
                            _pledger = ExplorationLedger.from_records(_partial_records)
                            _ptask_complexity = getattr(
                                ctx, "task_complexity", "",
                            ) or ""
                            _pfloors = ExplorationFloors.from_env_with_adapted(_ptask_complexity)
                            _pverdict = evaluate_exploration(_pledger, _pfloors)
                            _pcovered = sorted(
                                c.value for c in _pverdict.categories_covered
                            )
                            # Classify cause from error string — cheap
                            # substring match, no regex. Order matters:
                            # most specific first.
                            if "background_dw_" in _err_msg:
                                _pcause = "bg_dw_failure"
                            elif "doubleword_schema_invalid" in _err_msg:
                                _pcause = "dw_schema_invalid"
                            elif "all_providers_exhausted" in _err_msg:
                                _pcause = "all_providers_exhausted"
                            elif "APITimeout" in _err_msg or "timeout" in _err_msg.lower():
                                _pcause = "provider_timeout"
                            else:
                                _pcause = "generic_gen_failure"
                            logger.info(
                                "[Orchestrator] ExplorationLedger(shadow,partial) "
                                "op=%s complexity=%s route=%s cause=%s "
                                "records=%d score=%.2f min_score=%.2f unique=%d "
                                "categories=%s would_pass=%s",
                                ctx.op_id[:12],
                                _ptask_complexity or "unknown",
                                _route or "unknown",
                                _pcause,
                                len(_partial_records),
                                _pverdict.score,
                                _pfloors.min_score,
                                _pledger.unique_call_count(),
                                ",".join(_pcovered) or "-",
                                _pverdict.sufficient,
                            )
                        except Exception:
                            logger.debug(
                                "[Orchestrator] ExplorationLedger partial shadow log error",
                                exc_info=True,
                            )

                    # ── BACKGROUND / SPECULATIVE route failures ──
                    # These routes intentionally avoid Claude. Don't retry
                    # with expensive providers — accept failure gracefully.
                    if _route == "speculative" and "speculative_deferred" in _err_msg:
                        # Speculative ops are fire-and-forget — not a failure.
                        logger.info(
                            "[Orchestrator] SPECULATIVE op deferred (DW background) [%s]",
                            ctx.op_id,
                        )
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="speculative_deferred",
                        )
                        await self._record_ledger(
                            ctx, OperationState.COMPLETED,
                            {"reason": "speculative_deferred", "route": "speculative"},
                        )
                        return ctx

                    if _route == "background" and (
                        "background_dw_" in _err_msg
                        or "background_fallback_failed" in _err_msg
                    ):
                        # Background failure — accept gracefully, don't
                        # hammer the retry loop. Covers both the legacy
                        # DW-only failure mode ("background_dw_*") and the
                        # new cascade failure mode
                        # ("background_fallback_failed:...") introduced when
                        # JARVIS_BACKGROUND_ALLOW_FALLBACK=true and the
                        # Claude cascade itself also fails. In either case,
                        # the sensor will re-detect if the underlying work
                        # is still relevant.
                        _is_cascade_failure = "background_fallback_failed" in _err_msg
                        logger.info(
                            "[Orchestrator] BACKGROUND route: %s failed (%s), "
                            "accepting [%s]",
                            "DW+Claude cascade" if _is_cascade_failure else "DW",
                            _err_msg[:120], ctx.op_id,
                        )
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code=f"background_accepted:{_err_msg[:80]}",
                        )
                        await self._record_ledger(
                            ctx, OperationState.FAILED,
                            {
                                "reason": (
                                    "background_cascade_failure"
                                    if _is_cascade_failure else "background_dw_failure"
                                ),
                                "error": _err_msg[:200],
                                "route": "background",
                            },
                        )
                        return ctx

                    logger.warning(
                        "Generation attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        1 + self._config.max_generate_retries,
                        ctx.op_id,
                        exc,
                    )
                    generate_retries_remaining -= 1
                    if generate_retries_remaining < 0:
                        # ── IMMEDIATE → STANDARD demotion ──
                        # If IMMEDIATE exhausted Claude retries, demote to
                        # STANDARD (DW primary → Claude fallback) for one
                        # last attempt.  Direct call — don't rely on the
                        # exhausted for-loop range.
                        if _route == "immediate":
                            logger.info(
                                "[Orchestrator] IMMEDIATE exhausted — demoting "
                                "to STANDARD route for DW attempt [%s]",
                                ctx.op_id,
                            )
                            object.__setattr__(ctx, "provider_route", "standard")
                            object.__setattr__(
                                ctx, "provider_route_reason",
                                f"demotion:immediate_exhausted:{_err_msg[:60]}",
                            )
                            try:
                                await self._stack.comm.emit_decision(
                                    op_id=ctx.op_id,
                                    outcome="standard",
                                    reason_code="route_demoted:immediate_exhausted",
                                    details={
                                        "route": "standard",
                                        "previous_route": "immediate",
                                        "route_description": "Demoted to STANDARD after IMMEDIATE exhaustion",
                                        "budget_profile": "220s fallback budget",
                                        "route_reason": getattr(ctx, "provider_route_reason", ""),
                                    },
                                )
                            except Exception:
                                pass
                            _route = "standard"  # update local for timeout calc
                            # Refresh the cost-governor cap for the new route so
                            # the demotion gets a proportional budget headroom.
                            try:
                                self._cost_governor.start(
                                    op_id=ctx.op_id,
                                    route="standard",
                                    complexity=getattr(ctx, "task_complexity", "") or "",
                                    is_read_only=bool(getattr(ctx, "is_read_only", False)),
                                )
                            except Exception:
                                pass
                            # Guard the demotion call itself: if cumulative spend
                            # already blew past the new cap, skip the demotion.
                            if self._cost_governor.is_exceeded(ctx.op_id):
                                logger.warning(
                                    "[Orchestrator] Skipping STANDARD demotion — "
                                    "cost cap already exceeded [%s]",
                                    ctx.op_id,
                                )
                            else:
                                try:
                                    _dem_deadline = datetime.now(tz=timezone.utc) + timedelta(seconds=220.0)
                                    generation = await asyncio.wait_for(
                                        self._generator.generate(ctx, _dem_deadline),
                                        timeout=220.0 + _OUTER_GATE_GRACE_S,
                                    )
                                    # Charge demotion call cost (may be zero).
                                    try:
                                        _dem_cost = float(getattr(generation, "cost_usd", 0.0) or 0.0)
                                        _dem_prov = getattr(generation, "provider_name", "") or ""
                                        if _dem_cost > 0.0:
                                            _dem_phase = getattr(
                                                getattr(ctx, "phase", None),
                                                "name", "",
                                            ) or ""
                                            self._cost_governor.charge(
                                                ctx.op_id, _dem_cost, _dem_prov,
                                                phase=_dem_phase,
                                            )
                                            await self._emit_route_cost_heartbeat(
                                                ctx,
                                                cost_usd=_dem_cost,
                                                provider=_dem_prov,
                                                route="standard",
                                                cost_event="demotion_attempt",
                                            )
                                    except Exception:
                                        pass
                                    if generation is not None and len(generation.candidates) > 0:
                                        break  # success — continue pipeline
                                    generation = None
                                except Exception as dem_exc:
                                    logger.warning(
                                        "[Orchestrator] STANDARD demotion also failed: %s [%s]",
                                        dem_exc, ctx.op_id,
                                    )

                        # All retries truly exhausted
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="generation_failed",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {"reason": "generation_failed", "error": str(exc)},
                        )
                        return ctx
                    # P2: Dynamic Re-Planning — suggest alternative strategy on failure.
                    # Two-stage cascade:
                    #   (1) PlanFalsificationDetector (Slice 4 bridge) — proactive,
                    #       structural, evidence-typed. Preempts when plan steps
                    #       are falsified by filesystem probe + typed validation
                    #       evidence.
                    #   (2) DynamicRePlanner (legacy reactive) — backstop when
                    #       structural detector returns NO_FALSIFICATION /
                    #       INSUFFICIENT_EVIDENCE / DISABLED / FAILED.
                    _replan_text = ""
                    try:
                        # ONE READER, shared with `generate_runner` — the
                        # two call sites have already drifted once, which is
                        # how the shipping path spent months re-planning on
                        # empty inputs while this twin read a real verdict.
                        # Prefers the context (which VALIDATE publishes and
                        # `advance()` carries forward) and falls back to this
                        # scope's local, so neither source can go stale.
                        from backend.core.ouroboros.governance.op_context import (  # noqa: E501,PLC0415
                            replan_inputs as _replan_inputs,
                        )
                        _fc, _em = _replan_inputs(
                            getattr(ctx, "validation", None) or validation
                        )
                        _attempt_num = self._config.max_generate_retries - generate_retries_remaining + 1
                        # Stage 1 — structural falsification (proactive)
                        try:
                            from backend.core.ouroboros.governance.plan_falsification_orchestrator_bridge import (  # noqa: E501
                                bridge_to_replan as _falsification_bridge,
                            )
                            _fals_verdict, _fals_text = await _falsification_bridge(
                                plan_json=getattr(ctx, "implementation_plan", "") or "",
                                validation_failure_class=_fc,
                                validation_short_summary=_em,
                                target_files=tuple(getattr(ctx, "target_files", ()) or ()),
                                project_root=self._config.project_root,
                                op_id=ctx.op_id,
                            )
                            if _fals_text:
                                _replan_text = _fals_text
                                logger.info(
                                    "[Orchestrator] Falsification re-plan: "
                                    "step=%s kinds=%s (attempt %d) [%s]",
                                    _fals_verdict.falsified_step_index,
                                    ",".join(_fals_verdict.falsifying_evidence_kinds),
                                    _attempt_num, ctx.op_id,
                                )
                        except Exception as _fb_exc:
                            logger.debug(
                                "[Orchestrator] Falsification bridge degraded: %s",
                                _fb_exc,
                            )
                        # Stage 2 — legacy reactive (backstop, only if Stage 1 silent)
                        if not _replan_text:
                            from backend.core.ouroboros.governance.self_evolution import DynamicRePlanner
                            _replan = DynamicRePlanner.suggest_replan(_fc, _em, _attempt_num)
                            if _replan:
                                _replan_text = DynamicRePlanner.format_for_prompt(_replan)
                                logger.info(
                                    "[Orchestrator] Dynamic re-plan: %s (attempt %d)",
                                    _replan.trigger[:50], _attempt_num,
                                )
                    except Exception:
                        _replan_text = ""
                        pass

                    # Slice 100 — FSM Sentinel: signal repeated LLM-vs-validator
                    # contradiction to the ambiguity sensor mesh (Slice 99).
                    # Best-effort; NEVER affects the FSM. Lazy-import is INSIDE
                    # the try so even an ImportError can't break generation.
                    try:
                        from backend.core.ouroboros.governance.ambiguity_sensor_mesh import (  # noqa: E501
                            observe_generate_retry as _sentinel_observe,
                        )
                        _sentinel_attempt = (
                            self._config.max_generate_retries
                            - generate_retries_remaining + 1
                        )
                        _sentinel_observe(
                            ctx.op_id, _sentinel_attempt, detail=str(exc)[:120],
                        )
                    except Exception:
                        pass

                    # Retry: advance to GENERATE_RETRY with episodic memory context
                    _retry_ctx_kwargs = {}

                    # Inject direct error feedback so the model knows what went wrong
                    _err_str = str(exc)

                    # ── Iron Gate failures get targeted, in-flight instructions ──
                    if _err_str.startswith("exploration_insufficient"):
                        # ── Slice 21 Fix A2 — capability-aware halt ─────────
                        # The retry feedback below demands tool calls. When
                        # this op STRUCTURALLY cannot make them (the Venom
                        # loop is suppressed for its route/complexity — per
                        # the provider's OWN Slice-226 predicate) AND it
                        # earned ZERO exploration credit of any kind (no tool
                        # records, no preloaded-prompt credit), the retry is
                        # deterministically unresolvable: identical
                        # capability + identical prompt shape ⇒ identical
                        # rejection. bt-2026-07-15-063421 burned a full
                        # second generation + an EC8 forward-progress trip on
                        # 11 ops discovering this the hard way. Halt cleanly
                        # NOW: precise terminal reason, exact context frame,
                        # no artificial retry. Fail-open on any fault in the
                        # check itself (legacy retry proceeds).
                        _cap_halt = False
                        try:
                            _s21_credit = getattr(exc, "structural_credit", None)
                            if _s21_credit == 0:
                                from backend.core.ouroboros.governance.dw_terminal_worker_policy import (  # noqa: E501
                                    background_is_terminal_worker as _s21_bg_tw,
                                )
                                from backend.core.ouroboros.governance.exploration_engine import (  # noqa: E501
                                    compute_tool_loop_suppressed as _s21_suppressed,
                                )
                                _s21_route = str(
                                    getattr(ctx, "provider_route", "") or ""
                                )
                                _cap_halt = _s21_suppressed(
                                    complexity=str(
                                        getattr(ctx, "task_complexity", "") or ""
                                    ),
                                    route=_s21_route,
                                    is_bg_terminal_worker=_s21_bg_tw(_s21_route),
                                    has_repair_context=False,
                                    is_read_only=bool(
                                        getattr(ctx, "is_read_only", False)
                                    ),
                                )
                        except Exception:  # noqa: BLE001 — check fault ⇒ legacy retry
                            _cap_halt = False
                        if _cap_halt:
                            # Mandate 4 — the exact context frame, one
                            # grep-stable line, FULL op id (audit-keyed).
                            logger.warning(
                                "[Slice21CapabilityHalt] op=%s route=%s "
                                "complexity=%s model=%s attempt=%d "
                                "targets=%d first_target=%s credit=0 "
                                "preloaded=0 — exploration structurally "
                                "impossible (tool loop suppressed for this "
                                "route AND nothing preloaded); halting "
                                "instead of an unresolvable retry",
                                ctx.op_id,
                                getattr(ctx, "provider_route", "") or "?",
                                getattr(ctx, "task_complexity", "") or "?",
                                getattr(exc, "rejected_model_id", "") or "?",
                                self._config.max_generate_retries
                                - generate_retries_remaining + 1,
                                len(getattr(ctx, "target_files", ()) or ()),
                                (list(getattr(ctx, "target_files", ()) or ())
                                 or ["?"])[0],
                            )
                            _terminal = self._l2_escape_terminal(ctx.phase)
                            ctx = ctx.advance(
                                _terminal,
                                terminal_reason_code=(
                                    "exploration_impossible_no_capability"
                                ),
                            )
                            await self._record_ledger(
                                ctx,
                                OperationState.FAILED,
                                {
                                    "reason": (
                                        "exploration_impossible_no_capability"
                                    ),
                                    "route": getattr(
                                        ctx, "provider_route", "",
                                    ) or "",
                                    "complexity": getattr(
                                        ctx, "task_complexity", "",
                                    ) or "",
                                    "rejected_model_id": getattr(
                                        exc, "rejected_model_id", "",
                                    ) or "",
                                    "structural_credit": 0,
                                    "entry_phase": "GENERATE",
                                },
                            )
                            try:
                                self._forward_progress.finish(ctx.op_id)
                            except Exception:  # noqa: BLE001
                                pass
                            return ctx
                        # Ledger path (#103): when the exception carries a
                        # verdict + floors, render a category-aware feedback
                        # block so the model sees *which* categories are missing
                        # rather than the generic "call more tools" boilerplate.
                        # Legacy counter path has neither attribute and falls
                        # through to the hand-written block below.
                        _exc_verdict = getattr(exc, "verdict", None)
                        _exc_floors = getattr(exc, "floors", None)
                        if _exc_verdict is not None and _exc_floors is not None:
                            try:
                                from backend.core.ouroboros.governance.exploration_engine import (  # noqa: E501
                                    render_retry_feedback,
                                )
                                _ledger_feedback = render_retry_feedback(
                                    _exc_verdict, _exc_floors,
                                )
                            except Exception:
                                _ledger_feedback = ""
                        else:
                            _ledger_feedback = ""
                        if _ledger_feedback:
                            # ── CRITICAL_SYSTEM_OVERRIDE escalation ──
                            # Live-fire botyivw5b proved the feedback was
                            # landing in the prompt but the model was
                            # attending to the front-loaded task description
                            # and tool boilerplate instead of the retry
                            # directive. This is an attention-mechanism
                            # interference problem, not an injection
                            # problem. The three-pronged fix (this block is
                            # prong 2):
                            #
                            #   1. recency bias — _build_lean_codegen_prompt
                            #      appends strategic_memory as the ABSOLUTE
                            #      LAST section (after output schema), so
                            #      the model reads it last.
                            #   2. XML structural override — frontier models
                            #      are fine-tuned to obey
                            #      ``<CRITICAL_SYSTEM_OVERRIDE>`` tags at
                            #      higher priority than general prompt text.
                            #      "Mathematically required" language raises
                            #      perceived authority.
                            #   3. simulated assistant prefill — the lean
                            #      builder appends a model-voice commitment
                            #      stub after this block (persona
                            #      continuation kill switch; literal API
                            #      prefill is incompatible with the JSON
                            #      contract + tool_use response type on
                            #      sonnet-4-6 stream).
                            #
                            # Derive the specific tool names from the missing
                            # categories so the override preempts ambiguity
                            # about what "call_graph" means.
                            _cat_to_tools = {
                                "call_graph": "get_callers",
                                "history": "git_blame or git_log",
                                "discovery": "search_code or glob_files",
                                "structure": "list_symbols",
                                "comprehension": "read_file",
                            }
                            try:
                                _missing_cats = sorted(
                                    c.value for c in _exc_verdict.missing_categories
                                )
                            except Exception:
                                _missing_cats = []
                            _required_tools = [
                                _cat_to_tools.get(c, c) for c in _missing_cats
                            ]
                            _cat_list = ", ".join(_missing_cats) or "diverse"
                            _tool_list = ", ".join(_required_tools) or "get_callers"
                            _error_feedback = (
                                "<CRITICAL_SYSTEM_OVERRIDE>\n"
                                "Previous attempt failed the Iron Gate exploration "
                                "ledger. You are mathematically required to invoke "
                                f"tools from the following missing categories: "
                                f"[{_cat_list}].\n"
                                f"You MUST invoke {_tool_list} before emitting any "
                                "patch.\n"
                                "The ExplorationLedger dedups by (tool, "
                                "arguments_hash) — repeating the same read_file on "
                                "the same path earns ZERO new credit.\n"
                                "Your next action MUST be one of the required tool "
                                "calls listed above. Do NOT emit a patch. Do NOT "
                                "call read_file again on files you already read.\n"
                                "</CRITICAL_SYSTEM_OVERRIDE>\n\n"
                                "## PREVIOUS GENERATION REJECTED — EXPLORATION GATE\n\n"
                                f"{_ledger_feedback}\n\n"
                                "INSTRUCTIONS FOR RETRY:\n"
                                "- Call the missing-category tools listed above BEFORE\n"
                                "  emitting any patch. The ledger dedups by (tool,\n"
                                "  arguments_hash) so repeating the same read_file on\n"
                                "  the same path adds no credit.\n"
                                "- Prefer get_callers, list_symbols, and git_blame over\n"
                                "  repeated read_file calls — diversity beats volume.\n"
                                "- Exploration is NOT optional. Patches without context\n"
                                "  corrupt code.\n"
                            )
                        else:
                            _error_feedback = (
                                "## PREVIOUS GENERATION REJECTED — NO EXPLORATION\n\n"
                                f"{_err_str[:400]}\n\n"
                                "INSTRUCTIONS FOR RETRY:\n"
                                "- BEFORE writing any patch, call read_file on the target file(s).\n"
                                "- Call search_code or get_callers for any function/symbol you are\n"
                                "  about to modify so you understand its callers and tests.\n"
                                "- Only after you have at least 2 exploration tool calls in your\n"
                                "  tool_execution_records may you emit the final patch.\n"
                                "- Exploration is NOT optional. Patches without context corrupt code.\n"
                            )

                        # ── Slice 12P Phase 3 — Reflexive healing prepend ──
                        # Compose a structured <DEVELOPER_FEEDBACK> block
                        # (closed taxonomy class + canonical remediation
                        # actions) and prepend it to the existing feedback
                        # so the model sees:
                        #   1. NEW Slice 12P structured signal (priority
                        #      = CRITICAL_SYSTEM_OVERRIDE per the existing
                        #      attention-mechanism discipline at line ~5285)
                        #   2. EXISTING ExplorationLedger-aware deep detail
                        # Pure composition; format helper returns None for
                        # non-structural rejections so this is a no-op for
                        # provider exhaustion / wall cap / cancelled
                        # shutdown classes. NEVER raises.
                        try:
                            from backend.core.ouroboros.governance.reflexive_healing import (  # noqa: E501
                                format_structural_rejection_feedback as _slice12p_format,
                            )
                            _slice12p_block = _slice12p_format(
                                _err_str,
                                rejection_detail=_err_str[:300],
                                attempt_number=attempt + 1,
                                max_attempts=1 + self._config.max_generate_retries,
                            )
                            if _slice12p_block:
                                _error_feedback = (
                                    _slice12p_block + "\n\n" + _error_feedback
                                )
                                logger.debug(
                                    "[Orchestrator] Slice 12P reflexive "
                                    "healing prepend added to retry feedback "
                                    "for op=%s",
                                    ctx.op_id[:12],
                                )
                        except Exception:  # noqa: BLE001 — defensive
                            logger.debug(
                                "[Orchestrator] Slice 12P reflexive healing "
                                "formatter raised — falling through to "
                                "pre-Slice-12P feedback shape",
                                exc_info=True,
                            )
                    elif _err_str.startswith(_TARGET_MISSING_PREFIX):
                        # Slice 72 — target file doesn't exist in the worktree.
                        # Surface the host-vs-worktree steering correction so the
                        # model re-explores and targets a real repo file.
                        # Universal mode (2026-07-21): lane-correct wording —
                        # host self-dev ops get path-hygiene steering (doubled
                        # prefix / phantom parent), not third-party-repo text.
                        _tg_paths = [
                            p.strip()
                            for p in _err_str[len(_TARGET_MISSING_PREFIX):].split(",")
                            if p.strip()
                        ]
                        _error_feedback = _target_missing_retry_feedback(
                            _tg_paths,
                            benchmark=(
                                getattr(ctx, "signal_source", "")
                                == "swe_bench_pro"
                            ),
                        )
                    elif _err_str.startswith("ascii_corruption"):
                        # Extract the specific offending lines from the rejected
                        # candidate so the model sees its own bad code in context
                        # (not just "U+0641 at L106:C6"). The orchestrator stashed
                        # the full_content + BadCodepoint list on the exception
                        # just before raising, so we can reconstruct the exact
                        # lines that tripped the gate and show ASCII-only
                        # corrections alongside them.
                        _rejected = getattr(exc, "_ascii_rejected_content", "") or ""
                        _bad_cps = getattr(exc, "_ascii_bad_codepoints", None) or []
                        _offending_block = ""
                        if _rejected and _bad_cps:
                            _lines = _rejected.split("\n")
                            _seen_lines: set = set()
                            _line_samples = []
                            for _bc in _bad_cps[:5]:
                                _ln = getattr(_bc, "line", 0)
                                if _ln <= 0 or _ln in _seen_lines or _ln > len(_lines):
                                    continue
                                _seen_lines.add(_ln)
                                _raw_line = _lines[_ln - 1]
                                # Build an ASCII-only "what-to-write-instead" hint
                                # by stripping every non-ASCII codepoint. For
                                # letters this produces a visible "hole" that
                                # shows where the model must make a deliberate
                                # spelling decision (e.g. rapidفuzz → rapiduzz,
                                # which makes the corruption obvious).
                                _stripped = "".join(
                                    ch if ord(ch) < 128 else "·" for ch in _raw_line
                                )
                                _cp_hex = f"U+{getattr(_bc, 'codepoint', 0):04X}"
                                _char = getattr(_bc, "char", "?")
                                _line_samples.append(
                                    f"  line {_ln} contains {_cp_hex} '{_char}':\n"
                                    f"      WRONG: {_raw_line}\n"
                                    f"      (·=non-ASCII): {_stripped}"
                                )
                            if _line_samples:
                                _offending_block = (
                                    "\nSPECIFIC OFFENDING LINES FROM YOUR LAST OUTPUT:\n"
                                    + "\n".join(_line_samples) + "\n"
                                )

                        _error_feedback = (
                            "## PREVIOUS GENERATION REJECTED — UNICODE CORRUPTION\n\n"
                            f"{_err_str[:400]}\n"
                            f"{_offending_block}\n"
                            "INSTRUCTIONS FOR RETRY:\n"
                            "- The lines above contain Unicode LETTERS that look like\n"
                            "  ASCII letters but aren't. These are HARD FAILURES — the\n"
                            "  Iron Gate auto-heals punctuation (em-dash, curly quotes,\n"
                            "  ellipsis, nbsp) but it will NEVER auto-heal letters\n"
                            "  because changing a letter changes the identity of a\n"
                            "  package, function, or variable.\n"
                            "- Re-emit the ENTIRE file using only 7-bit ASCII (0x20–0x7E)\n"
                            "  for every character. Every byte must satisfy ord(c) < 128.\n"
                            "- Common culprits in package manifests (requirements.txt,\n"
                            "  package.json, pyproject.toml, Pipfile):\n"
                            "    * U+0641 Arabic FEH 'ف' looks like ASCII 'f'\n"
                            "    * U+0430 Cyrillic 'а' looks like ASCII 'a'\n"
                            "    * U+0435 Cyrillic 'е' looks like ASCII 'e'\n"
                            "    * U+03BF Greek omicron 'ο' looks like ASCII 'o'\n"
                            "  If you're about to write 'rapidfuzz', type r-a-p-i-d-f-u-z-z\n"
                            "  using ONLY characters from the ASCII table. Do not rely on\n"
                            "  memory of what the package name 'looks like'.\n"
                            "- Sanity check: every single character in your output must\n"
                            "  be in the range 0x20–0x7E or \\n (0x0A). No exceptions.\n"
                        )

                        # ── Slice 12R Phase 3 — Reflexive ASCII healing ──
                        # Compose the structured Slice 12P
                        # <DEVELOPER_FEEDBACK> block ahead of the
                        # hand-written ASCII feedback above. The
                        # block surfaces the rejection class
                        # (ascii_gate_failed) + canonical
                        # remediation actions so the model's
                        # attention mechanism gives it priority
                        # over front-loaded task text. Composes
                        # the same pattern that Slice 12P already
                        # wired for exploration_insufficient at
                        # ~line 5395. Pure prepend; None return
                        # leaves legacy feedback byte-identical.
                        # NEVER raises into the retry loop.
                        try:
                            from backend.core.ouroboros.governance.reflexive_healing import (  # noqa: E501
                                format_structural_rejection_feedback as _slice12r_format,
                            )
                            # Reflexive healing's classifier matches
                            # "ascii_gate_failed" substring — pass an
                            # equivalent canonical code so the
                            # classifier picks the right action list.
                            _slice12r_block = _slice12r_format(
                                "ascii_gate_failed: " + _err_str[:200],
                                rejection_detail=_err_str[:300],
                                attempt_number=attempt + 1,
                                max_attempts=1 + self._config.max_generate_retries,
                            )
                            if _slice12r_block:
                                _error_feedback = (
                                    _slice12r_block + "\n\n" + _error_feedback
                                )
                                logger.debug(
                                    "[Orchestrator] Slice 12R reflexive "
                                    "ASCII healing prepend added to retry "
                                    "feedback for op=%s",
                                    ctx.op_id[:12],
                                )
                        except Exception:  # noqa: BLE001 — defensive
                            logger.debug(
                                "[Orchestrator] Slice 12R reflexive ASCII "
                                "healing formatter raised — falling "
                                "through to pre-Slice-12R feedback shape",
                                exc_info=True,
                            )
                    elif _err_str.startswith("multi_file_coverage_insufficient"):
                        # Gate 5 rejection — name the missing target paths and
                        # reiterate the files: [...] shape. The model saw the
                        # single-file schema example in its prompt; here we
                        # hand it the multi-file example plus the exact list
                        # of paths it failed to cover.
                        _mf_missing = getattr(exc, "_mf_missing_paths", None) or []
                        _mf_targets = getattr(exc, "_mf_target_files", None) or tuple(
                            ctx.target_files
                        )
                        try:
                            from backend.core.ouroboros.governance.multi_file_coverage_gate import (
                                render_missing_block as _mf_render,
                            )
                            _missing_block = _mf_render(_mf_missing, _mf_targets)
                        except Exception:  # noqa: BLE001
                            _missing_block = (
                                "\nMISSING TARGET FILES:\n"
                                + "\n".join(f"  - {p}" for p in list(_mf_missing)[:16])
                                + "\n"
                            )
                        _target_count = len(_mf_targets)
                        _error_feedback = (
                            "## PREVIOUS GENERATION REJECTED — "
                            "MULTI-FILE COVERAGE INSUFFICIENT\n\n"
                            f"{_err_str[:400]}\n"
                            f"{_missing_block}\n"
                            "INSTRUCTIONS FOR RETRY:\n"
                            f"- This operation targets {_target_count} files. "
                            "You MUST return the multi-file shape: a `files` "
                            "list with one entry per target file.\n"
                            "- Do NOT use the legacy single-file schema "
                            "(`file_path` + `full_content` at the top level of "
                            "the candidate). That shape can only express ONE "
                            "file and will be rejected again.\n"
                            "- Use this structure for each candidate:\n\n"
                            "    {\n"
                            "      \"candidate_id\": \"c1\",\n"
                            "      \"files\": [\n"
                            "        {\n"
                            "          \"file_path\": \"<target path 1>\",\n"
                            "          \"full_content\": \"<complete file 1 content>\",\n"
                            "          \"rationale\": \"<why file 1 changes>\"\n"
                            "        },\n"
                            "        {\n"
                            "          \"file_path\": \"<target path 2>\",\n"
                            "          \"full_content\": \"<complete file 2 content>\",\n"
                            "          \"rationale\": \"<why file 2 changes>\"\n"
                            "        }\n"
                            "      ],\n"
                            "      \"rationale\": \"<one-sentence summary of the change set>\"\n"
                            "    }\n\n"
                            f"- Every one of the {_target_count} target paths above "
                            "must appear as a `file_path` entry in the `files` "
                            "list. Do not omit any.\n"
                            "- `full_content` in each entry must be the COMPLETE "
                            "file (not a diff, not a patch, not just the changed "
                            "lines).\n"
                            "- Python files must be syntactically valid "
                            "(`ast.parse()`-clean) per file.\n"
                        )
                    elif _err_str.startswith("Dependency file rename/truncation suspected"):
                        # Gate 3 rejection — show the offender pairs and a clear
                        # rule: you are NOT allowed to rename/shorten an existing
                        # package name, only add new ones or bump versions.
                        _dep_offenders = getattr(exc, "_dep_file_offenders", None) or []
                        _dep_rejected = getattr(exc, "_dep_file_rejected_content", "") or ""
                        _offender_block = ""
                        if _dep_offenders:
                            _offender_lines = "\n".join(
                                f"  {i + 1}. {pair}" for i, pair in enumerate(_dep_offenders[:10])
                            )
                            _offender_block = (
                                "\nSUSPICIOUS RENAMES DETECTED:\n"
                                f"{_offender_lines}\n"
                            )
                        _error_feedback = (
                            "## PREVIOUS GENERATION REJECTED — DEPENDENCY FILE CORRUPTION\n\n"
                            f"{_err_str[:400]}\n"
                            f"{_offender_block}\n"
                            "INSTRUCTIONS FOR RETRY:\n"
                            "- You deleted existing package(s) and added a near-identical\n"
                            "  new name. This is almost always a typo or hallucination —\n"
                            "  real upgrades change only the VERSION, not the package name.\n"
                            "- If the goal is to UPGRADE a package: keep the name identical\n"
                            "  (e.g. `anthropic==0.75.0` → `anthropic==0.80.0`). NEVER change\n"
                            "  the letters of the package name.\n"
                            "- If you truly need to REPLACE a package with a different one,\n"
                            "  the new name must be clearly distinct (not a substring or\n"
                            "  truncation of the old name) AND the reason must be in the\n"
                            "  `rationale` field of your candidate.\n"
                            "- Common hallucination patterns to avoid:\n"
                            "    * truncation: `rapidfuzz` → `rapidfu` (WRONG)\n"
                            "    * suffix append: `anthropic` → `anthropichttp` (WRONG)\n"
                            "    * single-char typo: `requests` → `reqest` (WRONG)\n"
                            "- Before emitting, compare each package name against the\n"
                            "  source file character-by-character. Every name that was\n"
                            "  there must still be there with the exact same spelling.\n"
                        )
                    else:
                        _error_feedback = (
                            "## PREVIOUS GENERATION FAILED\n\n"
                            f"Error: {_err_str[:300]}\n\n"
                            "INSTRUCTIONS FOR RETRY:\n"
                            "- Return schema_version '2b.1' with 'full_content' containing the COMPLETE file\n"
                            "- Do NOT return unified diffs or patches\n"
                            "- Ensure the JSON is valid (no trailing commas, no unquoted keys)\n"
                            "- full_content must be the entire file, not a summary or placeholder\n"
                        )
                    # Truncation-retry (gated): output truncation/elision ->
                    # retry with a changed output shape (diff) + token headroom
                    # instead of re-yelling. Reuses build_truncation_retry_directive.
                    try:
                        from backend.core.ouroboros.governance.truncation_retry import (
                            truncation_retry_enabled, is_truncation_failure,
                            build_truncation_retry_directive, stamp_retry_directive,
                        )
                        if truncation_retry_enabled() and is_truncation_failure(_err_msg):
                            _diff_capable = False
                            _ri = getattr(getattr(ctx, "telemetry", None), "routing_intent", None)
                            if _ri is not None:
                                _diff_capable = getattr(_ri, "schema_capability", "") == "full_content_and_diff"
                            _cur_max = int(getattr(ctx, "retry_max_tokens_override", 0) or 8192)
                            _tr_directive = build_truncation_retry_directive(
                                diff_capable=_diff_capable, current_max_tokens=_cur_max)
                            ctx = stamp_retry_directive(ctx, _tr_directive)
                            _error_feedback = (
                                _error_feedback + "\n\n" + _tr_directive.feedback
                                if _error_feedback else _tr_directive.feedback
                            )
                            logger.info(
                                "[TruncationRetry] op=%s force_diff=%s max_tokens=%d -> GENERATE_RETRY",
                                getattr(ctx, "op_id", "?"), _tr_directive.force_diff,
                                _tr_directive.new_max_tokens,
                            )
                    except Exception:
                        logger.debug("[TruncationRetry] skip", exc_info=True)

                    _retry_ctx_kwargs["strategic_memory_prompt"] = _error_feedback

                    # Record generation failure in episodic memory for downstream use
                    if _episodic_memory is not None:
                        _gen_failure_class = "content"
                        if "exploration_insufficient" in _err_str:
                            _gen_failure_class = "exploration"
                        elif _err_str.startswith(_TARGET_MISSING_PREFIX):
                            _gen_failure_class = "target_missing"
                        elif "ascii_corruption" in _err_str:
                            _gen_failure_class = "ascii"
                        elif _err_str.startswith("multi_file_coverage_insufficient"):
                            _gen_failure_class = "multi_file_coverage"
                        elif _err_str.startswith("Dependency file rename/truncation"):
                            _gen_failure_class = "dep_file_rename"
                        elif "json_parse_error" in _err_str:
                            _gen_failure_class = "json_parse"
                        elif "diff_apply_failed" in _err_str:
                            _gen_failure_class = "diff_apply"
                        elif "schema_invalid" in _err_str:
                            _gen_failure_class = "schema"
                        try:
                            _episodic_memory.record(
                                file_path=list(ctx.target_files)[0] if ctx.target_files else "unknown",
                                attempt=attempt + 1,
                                failure_class=_gen_failure_class,
                                error_summary=_err_str[:500],
                                specific_errors=[_err_str[:200]],
                                line_numbers=[],
                            )
                        except Exception:
                            pass

                    # ── Slice 235 fail-safe degradation governor ──────────────
                    # A 2b.1-diff candidate failed to apply (stale context /
                    # coordinate drift). Degrade THIS op's retry to full_content
                    # so a botched diff doesn't re-emit a diff that fails again.
                    # Set the per-op override the providers honor (op_context
                    # .force_full_content_override) + tell the model to emit the
                    # complete file. NEVER crash — this is recovery, not failure.
                    _diff_failed = (
                        "diff_apply_failed" in _err_str
                        or "stale_diff" in _err_str
                        or "StaleDiff" in _err_str
                        or "validate_diff" in _err_str
                    )
                    if _diff_failed:
                        _retry_ctx_kwargs["force_full_content_override"] = True
                        _df_msg = (
                            "\n\n[SYSTEM] Your unified-diff patch failed to apply "
                            "(stale/ambiguous context). For this retry, emit the "
                            "COMPLETE file content (full_content schema 2b.1), not "
                            "a diff."
                        )
                        _df_existing = _retry_ctx_kwargs.get("strategic_memory_prompt", "") or ""
                        _retry_ctx_kwargs["strategic_memory_prompt"] = (
                            f"{_df_existing}{_df_msg}" if _df_existing else _df_msg.strip()
                        )
                        logger.warning(
                            "[Orchestrator] Slice235 fail-safe: diff apply failed "
                            "(%s) → degrading op=%s retry to full_content",
                            _err_str[:80],
                            getattr(ctx, "op_id", "?"),
                        )

                    # Inject re-plan if available (appends to error feedback)
                    if _replan_text:
                        _existing = _retry_ctx_kwargs.get("strategic_memory_prompt", "")
                        _retry_ctx_kwargs["strategic_memory_prompt"] = (
                            f"{_existing}\n\n{_replan_text}" if _existing else _replan_text
                        )

                    if _episodic_memory is not None and _episodic_memory.has_failures():
                        _failure_context = _episodic_memory.format_for_prompt()
                        if _failure_context:
                            # Preserve iron-gate feedback already staged for retry
                            # (ExplorationInsufficientError etc). Reading from ctx
                            # here would silently drop _error_feedback — the
                            # severed nervous system bug that hid category-aware
                            # retry instructions from the model on every
                            # post-Iron-Gate retry.
                            _existing = _retry_ctx_kwargs.get("strategic_memory_prompt", "") or ""
                            _retry_ctx_kwargs["strategic_memory_prompt"] = (
                                f"{_existing}\n\n{_failure_context}" if _existing else _failure_context
                            )
                            logger.info(
                                "[Orchestrator] Injecting %d episodic failure(s) into retry context [%s]",
                                _episodic_memory.total_episodes, ctx.op_id,
                            )
                    # Inject consciousness fragile-file memory into retry context
                    if _consciousness_bridge is not None:
                        try:
                            _fragile_ctx = _consciousness_bridge.get_fragile_file_context(
                                ctx.target_files
                            )
                            if _fragile_ctx:
                                _existing_mem = _retry_ctx_kwargs.get("strategic_memory_prompt", "")
                                _retry_ctx_kwargs["strategic_memory_prompt"] = (
                                    f"{_existing_mem}\n\n{_fragile_ctx}" if _existing_mem else _fragile_ctx
                                )
                        except Exception:
                            pass
                    ctx = ctx.advance(OperationPhase.GENERATE_RETRY, **_retry_ctx_kwargs)

            assert generation is not None  # guaranteed by loop logic

            # L1: emit tool execution audit records to ledger stream.
            # This runs BEFORE the noop guard so that tool records are always
            # persisted regardless of whether the response was a noop.
            for _rec in generation.tool_execution_records:
                try:
                    _entry = LedgerEntry(
                        op_id=ctx.op_id,
                        state=OperationState.SANDBOXING,
                        data={"kind": "tool_exec.v1", **_dc_asdict(_rec)},
                        entry_id=_rec.call_id,
                    )
                    await self._stack.ledger.append(_entry)
                except asyncio.CancelledError:
                    raise
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(
                        "tool_exec ledger emit failed op=%s record=%s: %s",
                        ctx.op_id, getattr(_rec, "call_id", "?"), _exc,
                    )  # ledger failure must never abort governance pipeline

            # Short-circuit: model signalled the change is already present.
            #
            # Read-only discipline (Session 10, Derek 2026-04-17 Manifesto §8):
            # when ctx.is_read_only=True the noop short-circuit represents the
            # structurally expected terminal state (findings delivered via
            # subagent rollup, no code change by contract). Emit a POSTMORTEM
            # event with root_cause="read_only_complete" so the Synthetic Soul
            # has a clean audit trail and post-hoc analysis can distinguish
            # cartography completions from "model said no-op" completions.
            # Terminal reason code + ledger reason are aligned to the same
            # value so log, ledger, and comm-protocol all agree.
            if generation.is_noop:
                # Anti-Venom S2 — noop + in-loop-write guard. If Venom wrote
                # files DURING generation (edit_file/write_file landed on disk)
                # and THEN the model reports a no-op, those mutations never
                # passed the SemanticGuardian / GATE / risk-tier floor (the noop
                # fast-path skips APPLY). That is a guardian-bypass: code is on
                # disk that no gate ever saw. Fail-CLOSED by CANCELLING the op
                # so the operator sees a terminal failure (the on-disk writes
                # remain for inspection / next-op reconciliation) rather than a
                # silent COMPLETE that hides the unreviewed mutation.
                _inloop = getattr(generation, "venom_edit_history", ()) or ()
                if _inloop:
                    logger.warning(
                        "[Orchestrator] op=%s noop+%d in-loop writes never "
                        "passed guardian — cancelling",
                        ctx.op_id,
                        len(_inloop),
                    )
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="noop_inloop_write_guard",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "noop_inloop_write_guard"},
                    )
                    return ctx
                _is_read_only_terminal = bool(
                    getattr(ctx, "is_read_only", False)
                )
                _terminal_reason = (
                    "read_only_complete"
                    if _is_read_only_terminal
                    else "noop"
                )
                logger.info(
                    "[Orchestrator] op=%s is_noop=True (provider=%s) "
                    "terminal_reason_code=%s — skipping APPLY",
                    ctx.op_id,
                    generation.provider_name,
                    _terminal_reason,
                )
                # POSTMORTEM emission for read-only ops (Manifesto §8).
                # Emitted BEFORE ctx.advance so the audit trail matches
                # the lifecycle: GENERATE → (synthesis produced findings)
                # → POSTMORTEM → COMPLETE. Non-read-only noop ops retain
                # the legacy silent-complete semantics (no POSTMORTEM) to
                # preserve backward compatibility with existing analytics
                # that treat noop as a null event.
                if _is_read_only_terminal:
                    try:
                        await self._stack.comm.emit_postmortem(
                            op_id=ctx.op_id,
                            root_cause="read_only_complete",
                            failed_phase=None,
                            next_safe_action="none",
                        )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] read-only POSTMORTEM emit failed",
                            exc_info=True,
                        )
                ctx = ctx.advance(
                    OperationPhase.COMPLETE,
                    generation=generation,
                    terminal_reason_code=_terminal_reason,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.APPLIED,
                    {
                        "reason": _terminal_reason,
                        "provider": generation.provider_name,
                    },
                )
                return ctx

            # ---- Slice 13/15: semantic value gate (post-GENERATE, pre-
            # VALIDATE). Slice 15 hoisted the body into the shared
            # _maybe_complete_cosmetic_candidate helper because THIS legacy
            # inline seam is unreached on the dispatcher route (Run-24: zero
            # [ValueGate] lines — 'the legacy inline blocks below are never
            # reached'); the live call site is dispatch_pipeline's
            # GENERATE→VALIDATE transition, runtime-reachability-pinned.
            _vg_terminal = await self._maybe_complete_cosmetic_candidate(
                ctx, generation,
            )
            if _vg_terminal is not None:
                return _vg_terminal

        # Wave 2 (5) Slice 4a.1 - VALIDATERunner delegation gate.
        # Flag JARVIS_PHASE_RUNNER_VALIDATE_EXTRACTED (default false) routes
        # the 762-line VALIDATE block (nested retry FSM + L2 dispatch +
        # source-drift + shadow harness + entropy + read-only short-circuit)
        # through the extracted PhaseRunner. The best_candidate local leaks
        # downstream to GATE (37 refs); threaded via PhaseResult.artifacts.
        if _phase_runner_validate_extracted():
            from backend.core.ouroboros.governance.phase_runners.validate_runner import (
                VALIDATERunner,
            )
            logger.info("[PhaseRunnerDelegate] VALIDATE → runner op=%s", ctx.op_id[:16])
            _validate_runner = VALIDATERunner(
                self, _serpent, generation,
                generate_retries_remaining, _episodic_memory,
            )
            _validate_result = await _validate_runner.run(ctx)
            best_candidate = _validate_result.artifacts.get("best_candidate")
            best_validation = _validate_result.artifacts.get("best_validation")
            if _validate_result.next_phase is None:
                # Terminal exit (budget / infra / no_candidate / drift / cancel / read_only)
                return _validate_result.next_ctx
            ctx = _validate_result.next_ctx
        else:
            # Store generation result in context
            if _serpent: _serpent.update_phase("VALIDATE")
            ctx = ctx.advance(OperationPhase.VALIDATE, generation=generation)

            # Heartbeat: VALIDATE phase starting (Manifesto §7)
            try:
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="validate", progress_pct=55.0,
                )
            except Exception:
                pass

            # ── PreActionNarrator: voice WHAT before VALIDATE ──
            if self._pre_action_narrator is not None:
                try:
                    await self._pre_action_narrator.narrate_phase(
                        "VALIDATE", {"test_count": str(len(ctx.target_files))},
                    )
                except Exception:
                    pass

            # ---- Phase 4: VALIDATE ----

            # ── LSP Type Check (fast, incremental) ──
            _lsp_result = None
            try:
                from backend.core.ouroboros.governance.lsp_checker import LSPTypeChecker
                _lsp = LSPTypeChecker(project_root=self._config.project_root)
                if _lsp.detect_checker_sync():
                    _changed = [str(self._config.project_root / f) for f in ctx.target_files]
                    _lsp_result = await asyncio.get_event_loop().run_in_executor(
                        None, _lsp.check_incremental, _changed,
                    )
                    if not _lsp_result.passed:
                        logger.info(
                            "[Orchestrator] LSP found %d type errors in %s",
                            _lsp_result.error_count, list(ctx.target_files)[:3],
                        )
            except Exception:
                logger.debug("[Orchestrator] LSP check skipped", exc_info=True)

            # ── Exploration-first enforcement ──
            # Verify the model explored (read_file, search_code, get_callers)
            # before proposing writes.  Soft gate: warn + flag, don't reject.
            _EXPLORATION_TOOLS = frozenset({"read_file", "search_code", "get_callers"})
            _min_explore = int(os.environ.get("JARVIS_MIN_EXPLORATION_CALLS", "2"))
            _exploration_count = 0
            _exploration_first_ok = True
            if generation.tool_execution_records:
                for _rec in generation.tool_execution_records:
                    _tname = getattr(_rec, "tool_name", "")
                    if _tname in _EXPLORATION_TOOLS:
                        _exploration_count += 1
                if _exploration_count < _min_explore:
                    _exploration_first_ok = False
                    logger.warning(
                        "[Orchestrator] Exploration-first violation: %d/%d exploration calls "
                        "(expected >= %d) for op %s — candidate may lack codebase context",
                        _exploration_count, len(generation.tool_execution_records),
                        _min_explore, ctx.op_id[:12],
                    )

            best_candidate: Optional[Dict[str, Any]] = None
            best_validation: Optional[ValidationResult] = None
            validate_retries_remaining = self._config.max_validate_retries

            # ── [ValidateRetryFSM] instrumentation (Follow-up A, Session T aftermath) ──
            # Session T (bt-2026-04-15-211616) showed the op producing 1m40s of dead
            # air between `InteractiveRepair disabled — falling through to
            # VALIDATE_RETRY/L2` and cost_governor.finish, with phase=CLASSIFY in
            # the finalize line (ctx-reference mismatch). L2 never dispatched even
            # once. The retry loop has ~15 entry/exit/transition points and none
            # of them were logged at INFO level, so we couldn't tell which branch
            # the op took. This helper + tagged log lines make every transition
            # auditable so the next session's log pinpoints the silent-exit line.
            # Manifesto §8 (Absolute Observability): a path that ends cost
            # accounting without naming the terminal branch is a first-class bug.
            def _fsm_log(state: str, extra: str = "") -> None:
                _fc = (
                    getattr(best_validation, "failure_class", None)
                    if best_validation is not None else None
                )
                logger.info(
                    "[ValidateRetryFSM] %s op=%s ctx_id=%x phase=%s "
                    "retries_remaining=%d best_fc=%r n_cands=%d%s",
                    state,
                    ctx.op_id[:16],
                    id(ctx),
                    ctx.phase.name,
                    validate_retries_remaining,
                    _fc,
                    len(generation.candidates),
                    f" {extra}" if extra else "",
                )

            for _iter_idx in range(1 + self._config.max_validate_retries):
                _fsm_log("iter_start", f"iter={_iter_idx}")
                # Compute remaining budget from pipeline_deadline
                if ctx.pipeline_deadline is not None:
                    remaining_s = (
                        ctx.pipeline_deadline - datetime.now(tz=timezone.utc)
                    ).total_seconds()
                else:
                    remaining_s = self._config.validation_timeout_s  # fallback

                if remaining_s <= 0.0:
                    _fsm_log("budget_exhausted_pre", f"remaining_s={remaining_s:.1f}")
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="validation_budget_exhausted",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "validation_budget_exhausted"},
                    )
                    _fsm_log("budget_exhausted_return")
                    return ctx

                # Try all candidates in parallel; pick first that passes
                async def _validate_one(cand: Dict[str, Any]) -> Tuple[Dict[str, Any], "ValidationResult", float]:
                    _t0 = time.monotonic()
                    _val = await self._run_validation(ctx, cand, remaining_s)
                    return (cand, _val, time.monotonic() - _t0)

                _validation_tasks = [_validate_one(c) for c in generation.candidates]
                _validation_results = await asyncio.gather(*_validation_tasks, return_exceptions=True)

                # Process results in candidate order — preserves priority
                _early_return_ctx: Optional[OperationContext] = None
                for _vr in _validation_results:
                    if isinstance(_vr, BaseException):
                        logger.debug("[Orchestrator] Candidate validation raised: %s", _vr)
                        continue
                    candidate, validation, _validate_duration_s = _vr

                    await self._publish_candidate_verdict(
                        ctx,
                        candidate=candidate,
                        validation=validation,
                        duration_s=_validate_duration_s,
                        generation=generation,
                        exploration_first_ok=_exploration_first_ok,
                        exploration_count=_exploration_count,
                    )

                    # Heartbeat: validation result for TUI (Manifesto §7)
                    try:
                        # Phase 2 root-cause fix: this heartbeat read
                        # ``test_count`` / ``failure_count`` / ``output_preview``
                        # — none of which exist on ``ValidationResult`` — so
                        # every VALIDATE heartbeat reached the cockpit with empty
                        # detail (only ``failure_class`` survived). Read the REAL
                        # fields, now carrying the specific assertion the
                        # candidate died on.
                        _val_msg = type("_Msg", (), {
                            "payload": {
                                "phase": "validate",
                                "test_passed": validation.passed,
                                "test_count": getattr(validation, "test_total", 0),
                                "test_failures": getattr(validation, "test_failed", 0),
                                "failure_class": validation.failure_class or "",
                                "failure_detail": str(
                                    getattr(validation, "failure_detail", "") or "",
                                )[:600],
                                "failed_tests": list(
                                    getattr(validation, "failed_tests", ()) or (),
                                )[:6],
                                "validation_output": str(
                                    getattr(validation, "short_summary", "") or "",
                                )[:300],
                            },
                            "op_id": ctx.op_id,
                            "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                        })()
                        for _t in getattr(self._stack.comm, "_transports", []):
                            try:
                                await _t.send(_val_msg)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Emit gate event for duplication blocks
                    if validation.failure_class == "duplication":
                        try:
                            await self._stack.comm.emit_decision(
                                op_id=ctx.op_id,
                                outcome="blocked",
                                reason_code="duplication",
                                target_files=list(ctx.target_files),
                            )
                        except Exception:
                            pass

                    if validation.passed and best_candidate is None:
                        best_candidate = candidate
                        best_validation = validation
                        continue  # still record ledger for remaining, but winner is chosen

                    # Infra failure: non-retryable — escalate immediately
                    if validation.failure_class == "infra" and _early_return_ctx is None:
                        ctx = ctx.advance(
                            OperationPhase.POSTMORTEM,
                            validation=validation,
                            terminal_reason_code="validation_infra_failure",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": "validation_infra_failure",
                                "failure_class": "infra",
                                "adapter_names_run": list(validation.adapter_names_run),
                                "validation_duration_s": validation.validation_duration_s,
                                "short_summary": validation.short_summary,
                            },
                        )
                        _early_return_ctx = ctx
                        _fsm_log("infra_early_return_set")

                    # Budget failure: non-retryable
                    if validation.failure_class == "budget" and _early_return_ctx is None:
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            validation=validation,
                            terminal_reason_code="validation_budget_exhausted",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {"reason": "validation_budget_exhausted"},
                        )
                        _early_return_ctx = ctx
                        _fsm_log("budget_early_return_set")

                    if not validation.passed:
                        # test/build failure: track for ledger; try next candidate
                        best_validation = validation

                        # ---- Record failure in episodic memory + build structured critique ----
                        if _episodic_memory is not None and validation.failure_class in ("test", "build"):
                            try:
                                from backend.core.ouroboros.governance.structured_critique import CritiqueBuilder
                                critique_report = CritiqueBuilder.from_validation_output(
                                    file_path=candidate.get("file_path", "unknown"),
                                    failure_class=validation.failure_class or "test",
                                    error_text=validation.error or "",
                                    test_output=validation.short_summary or "",
                                )
                                _episodic_memory.record(
                                    file_path=candidate.get("file_path", "unknown"),
                                    attempt=self._config.max_validate_retries - validate_retries_remaining + 1,
                                    failure_class=validation.failure_class or "test",
                                    error_summary=critique_report.summary,
                                    specific_errors=[c.what_failed for c in critique_report.critiques],
                                    line_numbers=[c.line_number for c in critique_report.critiques if c.line_number],
                                )
                                logger.info(
                                    "[Orchestrator] Episodic memory recorded: %s — %s [%s]",
                                    candidate.get("file_path", "?"),
                                    critique_report.summary,
                                    ctx.op_id,
                                )
                            except Exception:
                                logger.debug("[Orchestrator] Episodic/critique recording failed", exc_info=True)

                # If a non-retryable failure was found and no candidate passed, return immediately
                if _early_return_ctx is not None and best_candidate is None:
                    _fsm_log("early_return")
                    return _early_return_ctx

                if best_candidate is not None:
                    _fsm_log("candidate_passed_break")
                    break  # at least one candidate passed

                # All candidates failed this attempt
                # Short-circuit: if no tests were discovered, retrying is pointless —
                # the same candidates will produce the same 0-test result every time.
                if best_validation is not None and getattr(best_validation, "test_count", -1) == 0:
                    logger.info(
                        "[Orchestrator] Skipping retries — no tests discovered for op=%s",
                        ctx.op_id,
                    )
                    _fsm_log("no_tests_short_circuit")
                    validate_retries_remaining = -1  # fall through to L2 / cancel

                validate_retries_remaining -= 1
                if validate_retries_remaining < 0:
                    # ── L2 self-repair dispatch ───────────────────────────────────
                    if self._config.repair_engine is not None and best_validation is not None:
                        # ──────────────────────────────────────────────────
                        # Slice 6 — bounded L2 re-dispatch loop for SOFT stops
                        # (bt-2026-05-25-174218 root: L2 used 14s of 120s
                        # before iter 2 generation returned no candidate;
                        # pre-Slice-6 hook returned 'cancel' on any L2_STOPPED,
                        # killing the op despite 106s of unused L2 budget).
                        #
                        # JARVIS_L2_DISPATCH_RETRIES (default 1) = number of
                        # ADDITIONAL L2 dispatches after the first. With
                        # default=1: up to 2 total L2 dispatches per op, each
                        # getting a fresh 120s timebox (so ~240s total L2 wall
                        # time per op in the worst case). The orchestrator's
                        # CLASSIFY cost cap and the harness wall-clock watchdog
                        # both cap session-level wall time independently, so
                        # this never violates a global safety invariant.
                        # ──────────────────────────────────────────────────
                        _l2_max_dispatches = int(
                            os.environ.get("JARVIS_L2_DISPATCH_RETRIES", "1"),
                        ) + 1
                        _l2_dispatch_idx = 0
                        _l2_soft_stop_history: list = []
                        _l2_break_directive = None
                    else:
                        _l2_max_dispatches = 0
                        _l2_dispatch_idx = 0
                        _l2_soft_stop_history = []
                        _l2_break_directive = None
                    while (
                        self._config.repair_engine is not None
                        and best_validation is not None
                        and _l2_dispatch_idx < _l2_max_dispatches
                    ):
                        _l2_dispatch_idx += 1
                        # ── L2 deadline reconciliation (Session V fix) ─────────
                        # Manifesto §8 (Absolute Observability): an env var named
                        # ``JARVIS_L2_TIMEBOX_S`` must mean **the wall time
                        # reserved for L2 from the moment of dispatch** — not
                        # "silently clamped to whatever the pipeline clock has
                        # left." The prior behavior passed ``ctx.pipeline_
                        # deadline`` through as L2's effective deadline, so the
                        # hidden ``min(L2 timebox, pipeline_deadline - now)``
                        # won silently whenever the pipeline clock was depleted
                        # by CLASSIFY → PLAN → GENERATE → VALIDATE.
                        #
                        # Session V (``bt-2026-04-15-223631``, ``op-019d934a``)
                        # proved it live: ``JARVIS_L2_TIMEBOX_S=600`` was set,
                        # but L2 reported ``Iteration 1/8 starting (0s elapsed,
                        # 120s remaining)`` because VALIDATE drained the
                        # pipeline clock over ~5 minutes before L2 saw it. One
                        # L2 iteration ran, returned ``directive='cancel'``,
                        # the op died. The env var name lied.
                        #
                        # Fix: compute L2's deadline fresh at dispatch as
                        # ``now + JARVIS_L2_TIMEBOX_S`` and reconcile
                        # ``ctx.pipeline_deadline`` via
                        # ``with_pipeline_deadline()`` so downstream phases
                        # (GATE, APPLY, VERIFY, POSTMORTEM) see a consistent
                        # op-level clock — preserving the "one notion of 'op
                        # must end by'" invariant without masking the L2 budget
                        # decision. If the pipeline_deadline is already LARGER
                        # than the L2 fresh budget (operator set a generous
                        # global cap), we keep the larger value: L2 must never
                        # shrink an op's envelope. Either way, both clocks
                        # and the winning cap are logged at INFO so operators
                        # can audit the decision without reading source.
                        _l2_timebox_s = float(
                            os.environ.get("JARVIS_L2_TIMEBOX_S", "120.0")
                        )
                        _now_dt = datetime.now(timezone.utc)
                        _l2_fresh_deadline = _now_dt + timedelta(
                            seconds=_l2_timebox_s
                        )
                        _orig_pl_deadline = ctx.pipeline_deadline
                        _orig_remaining_s = (
                            (_orig_pl_deadline - _now_dt).total_seconds()
                            if _orig_pl_deadline is not None else 0.0
                        )
                        if (
                            _orig_pl_deadline is None
                            or _orig_pl_deadline < _l2_fresh_deadline
                        ):
                            _l2_deadline = _l2_fresh_deadline
                            _winning_cap = "l2_timebox_fresh"
                            # Reconcile the op-level clock. `pipeline_deadline`
                            # is a cooperative budget; `cost_governor` and the
                            # harness idle watcher maintain their own wall
                            # clocks, so extending here does not violate any
                            # global safety invariant — it merely tells
                            # downstream phases that L2 has legitimately
                            # reserved additional time beyond the original
                            # envelope.
                            ctx = ctx.with_pipeline_deadline(_l2_fresh_deadline)
                        else:
                            _l2_deadline = _orig_pl_deadline
                            _winning_cap = "pipeline_deadline_inherited"
                        logger.info(
                            "[Orchestrator] L2 deadline reconciliation: "
                            "pipeline_remaining=%.1fs l2_timebox_env=%.1fs "
                            "effective=%.1fs winning_cap=%s op=%s",
                            _orig_remaining_s,
                            _l2_timebox_s,
                            (_l2_deadline - _now_dt).total_seconds(),
                            _winning_cap,
                            ctx.op_id[:16],
                        )
                        _fsm_log(
                            "l2_dispatch_pre",
                            f"effective_s={(_l2_deadline - _now_dt).total_seconds():.0f} "
                            f"cap={_winning_cap} l2_timebox_env={_l2_timebox_s:.0f}",
                        )
                        directive = await self._l2_hook(
                            ctx, best_validation, _l2_deadline,
                        )
                        _fsm_log(
                            "l2_dispatch_post",
                            f"directive={directive[0]!r} attempt={_l2_dispatch_idx}/{_l2_max_dispatches}",
                        )
                        if directive[0] == "break":
                            # ── Slice 6 — capture for post-loop handling ──
                            _l2_break_directive = directive
                            break  # inner Slice 6 retry loop
                        elif directive[0] == "l2_retry":
                            # ── Slice 6 — re-dispatch L2 with fresh budget ──
                            # _l2_hook left ctx unadvanced; we record the soft
                            # stop reason and loop back for another dispatch
                            # (if budget allows).
                            _l2_soft_stop_history.append(
                                directive[2] if len(directive) > 2 else "unknown"
                            )
                            if _l2_dispatch_idx >= _l2_max_dispatches:
                                # Out of retries — convert soft stop to a
                                # genuine cancel terminal. ctx is still
                                # unadvanced; advance it now to CANCELLED
                                # so downstream phases see consistent state.
                                logger.info(
                                    "[Orchestrator] L2 soft-stop retries exhausted "
                                    "op=%s attempts=%d/%d stop_history=%s — "
                                    "converting to cancel terminal",
                                    ctx.op_id,
                                    _l2_dispatch_idx,
                                    _l2_max_dispatches,
                                    _l2_soft_stop_history,
                                )
                                ctx = ctx.advance(
                                    OperationPhase.CANCELLED,
                                    terminal_reason_code=(
                                        f"l2_soft_stop_retries_exhausted:"
                                        f"{_l2_dispatch_idx}"
                                    ),
                                )
                                await self._record_ledger(
                                    ctx,
                                    OperationState.FAILED,
                                    {
                                        "reason": "l2_soft_stop_retries_exhausted",
                                        "attempts": _l2_dispatch_idx,
                                        "soft_stop_history": _l2_soft_stop_history,
                                    },
                                )
                                _fsm_log(
                                    "l2_soft_retries_exhausted",
                                    f"attempts={_l2_dispatch_idx} history={_l2_soft_stop_history}",
                                )
                                return ctx
                            # Otherwise: keep looping for another L2 dispatch.
                            logger.info(
                                "[Orchestrator] L2 soft-stop re-dispatch op=%s "
                                "attempt=%d/%d prev_stop_reason=%s",
                                ctx.op_id,
                                _l2_dispatch_idx + 1,
                                _l2_max_dispatches,
                                _l2_soft_stop_history[-1] if _l2_soft_stop_history else "?",
                            )
                            _fsm_log(
                                "l2_redispatch",
                                f"attempt={_l2_dispatch_idx + 1}/{_l2_max_dispatches} "
                                f"prev={_l2_soft_stop_history[-1] if _l2_soft_stop_history else '?'}",
                            )
                            continue  # next iteration of the inner Slice 6 loop
                        elif directive[0] == "l2_pivot":
                            # T3 — Graceful Semantic Pivot. _l2_hook left ctx
                            # UNADVANCED; the pivot handler owns the terminal
                            # (decompose-further at the failure locus, or HITL
                            # DLQ if atomic). DAG-preserving — siblings untouched.
                            _fsm_log("l2_pivot_return")
                            _pivot_ctx = directive[1]
                            _pivot_sig = directive[2] if len(directive) > 2 else ""
                            _pivot_tail = directive[3] if len(directive) > 3 else ""
                            return await self._handle_l2_pivot(
                                _pivot_ctx, _pivot_sig, _pivot_tail,
                            )
                        elif directive[0] in ("cancel", "fatal"):
                            _fsm_log("l2_escape_return", f"directive={directive[0]!r}")
                            return directive[1]  # ctx was advanced inside _l2_hook
                    # ── Post-Slice-6-inner-loop: handle break-out case ──
                    if _l2_break_directive is not None:
                        best_candidate = _l2_break_directive[1]
                        best_validation = _l2_break_directive[2]
                        logger.info(
                            "[Orchestrator] L2 broke VALIDATE_RETRY loop for op=%s — "
                            "proceeding to source-drift / shadow / entropy / GATE "
                            "(candidate_id=%s, file=%s, source_hash=%s)",
                            ctx.op_id,
                            best_candidate.get("candidate_id", "?"),
                            best_candidate.get("file_path", "?"),
                            (best_candidate.get("source_hash") or "")[:12],
                        )
                        _fsm_log("l2_converged_break")
                        break  # outer VALIDATE_RETRY while loop → GATE
                    if (
                        self._config.repair_engine is None
                        or best_validation is None
                    ):
                        _fsm_log(
                            "l2_skipped",
                            f"repair_engine={self._config.repair_engine is not None} "
                            f"best_validation={best_validation is not None}",
                        )
                    # ── end L2 dispatch ───────────────────────────────────────────

                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="no_candidate_valid",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {
                            "reason_code": "no_candidate_valid",
                            "candidates_tried": [
                                c.get("candidate_id", "?") for c in generation.candidates
                            ],
                            "failure_class": best_validation.failure_class if best_validation else "test",
                            "adapter_names_run": list(best_validation.adapter_names_run) if best_validation else [],
                            "validation_duration_s": best_validation.validation_duration_s if best_validation else 0.0,
                            "short_summary": best_validation.short_summary if best_validation else "",
                        },
                    )
                    _fsm_log("no_candidate_valid_return")
                    return ctx

                # ── Micro-Fix: try InteractiveRepair before expensive VALIDATE_RETRY ──
                _fsm_log("micro_fix_pre")
                if self._pre_action_narrator is not None:
                    try:
                        await self._pre_action_narrator.narrate_phase(
                            "MICRO_FIX", {"target": list(ctx.target_files)[:1]},
                        )
                    except Exception:
                        pass
                try:
                    from backend.core.ouroboros.governance.interactive_repair import InteractiveRepairLoop
                    _repair = InteractiveRepairLoop(
                        provider=self._generator,
                        project_root=self._config.project_root,
                    )
                    _repair_target = list(ctx.target_files)[0] if ctx.target_files else None
                    if _repair_target:
                        _repair_abs = self._config.project_root / _repair_target
                        if _repair_abs.is_file():
                            _repair_content = _repair_abs.read_text(errors="replace")
                            _test_argv = ["python3", "-m", "pytest", "-x", "-q"]
                            _repair_result = await asyncio.wait_for(
                                _repair.repair(
                                    file_path=str(_repair_target),
                                    file_content=_repair_content,
                                    test_argv=_test_argv,
                                    op_id=ctx.op_id,
                                ),
                                timeout=90.0,
                            )
                            _fsm_log(
                                "micro_fix_returned",
                                f"fixed={_repair_result.fixed} "
                                f"iterations={_repair_result.iterations_used}",
                            )
                            if _repair_result.fixed:
                                logger.info(
                                    "[Orchestrator] Micro-fix succeeded in %d iterations for op=%s",
                                    _repair_result.iterations_used, ctx.op_id,
                                )
                                # Skip full regeneration — advance to GATE
                                ctx = ctx.advance(OperationPhase.GATE, validation=best_validation)
                                _fsm_log("micro_fix_succeeded_break")
                                break
                        else:
                            _fsm_log(
                                "micro_fix_skipped_new_file",
                                f"target={_repair_target!r}",
                            )
                    else:
                        _fsm_log("micro_fix_skipped_no_target")
                except asyncio.CancelledError:
                    _fsm_log("micro_fix_cancelled")
                    raise
                except Exception as _repair_exc:
                    # §8 (Absolute Observability): a swallowed exception on this
                    # path is not acceptable. Upgrade from DEBUG to WARNING, stamp
                    # the exc class and message, keep exc_info for the traceback.
                    # The retry loop is designed to continue after this exception
                    # (the subsequent ctx.advance(VALIDATE_RETRY) runs below), so
                    # we do NOT re-raise — but we DO name the terminal branch.
                    logger.warning(
                        "[Orchestrator] Micro-fix failed (exc_class=%s): %s",
                        type(_repair_exc).__name__,
                        _repair_exc,
                        exc_info=True,
                    )
                    _fsm_log(
                        "micro_fix_exception_swallowed",
                        f"exc_class={type(_repair_exc).__name__}",
                    )

                # Retry: advance to VALIDATE_RETRY with episodic memory context
                _vr_kwargs = {}
                if _episodic_memory is not None and _episodic_memory.has_failures():
                    _vr_context = _episodic_memory.format_for_prompt()
                    if _vr_context:
                        _existing_vr = getattr(ctx, "strategic_memory_prompt", "") or ""
                        _vr_kwargs["strategic_memory_prompt"] = (
                            f"{_existing_vr}\n\n{_vr_context}" if _existing_vr else _vr_context
                        )
                _fsm_log("retry_advance_pre")
                _pre_ctx_id = id(ctx)
                ctx = ctx.advance(OperationPhase.VALIDATE_RETRY, **_vr_kwargs)
                # After ctx.advance: log the NEW ctx identity so the next session's
                # log lets us verify ctx actually rebound (Session T hypothesis:
                # the outer finally saw a stale ctx, which is only possible if
                # this rebind happened in a scope that didn't propagate).
                _fsm_log(
                    "retry_advance_post",
                    f"old_ctx_id={_pre_ctx_id:x} new_ctx_id={id(ctx):x}",
                )

            _fsm_log(
                "loop_exit_normal",
                f"best_candidate_present={best_candidate is not None}",
            )
            assert best_candidate is not None  # guaranteed by loop logic
            assert best_validation is not None

            # Source-drift check: file must not have changed since generation
            drift_hash = self._check_source_drift(best_candidate, self._config.project_root)
            if drift_hash is not None:
                logger.info(
                    "[Orchestrator] Source drift detected for op=%s file=%s "
                    "(expected=%s, actual=%s) — advancing to CANCELLED",
                    ctx.op_id,
                    best_candidate.get("file_path", "?"),
                    (best_candidate.get("source_hash") or "")[:12],
                    (drift_hash or "")[:12],
                )
                ctx = ctx.advance(
                    OperationPhase.CANCELLED,
                    terminal_reason_code="source_drift_detected",
                )
                await self._record_ledger(ctx, OperationState.FAILED, {
                    "reason_code": "source_drift_detected",
                    "file_path": best_candidate.get("file_path"),
                    "expected_source_hash": best_candidate.get("source_hash"),
                    "actual_source_hash": drift_hash,
                })
                return ctx
            logger.info(
                "[Orchestrator] Source-drift check passed for op=%s — "
                "proceeding to shadow harness + entropy + GATE",
                ctx.op_id,
            )

            # Winner traceability ledger entry
            await self._record_ledger(ctx, OperationState.GATING, {
                "event": "validation_complete",
                "winning_candidate_id": best_candidate.get("candidate_id"),
                "winning_candidate_hash": best_candidate.get("candidate_hash"),
                "winning_file_path": best_candidate.get("file_path"),
                "source_hash": best_candidate.get("source_hash"),
                "source_path": best_candidate.get("source_path"),
                "provider": generation.provider_name,
                "model": getattr(generation, "model_id", ""),
                "total_candidates_tried": len(generation.candidates),
            })

            # ── Shadow harness check (soft advisory — never hard-blocks GATE) ──────
            # Evaluates candidate structural integrity before GATE. Uses AST comparison
            # between the candidate's proposed content and itself (firewall-only path).
            # If the harness is disqualified, logs a warning — GATE still proceeds.
            if self._config.shadow_harness is not None and best_candidate is not None:
                import time as _sh_time
                from backend.core.ouroboros.governance.shadow_harness import (
                    OutputComparator,
                    SideEffectFirewall,
                    CompareMode,
                )
                from backend.core.ouroboros.governance.op_context import ShadowResult
                _sh_start = _sh_time.monotonic()
                _violations: list = []
                _confidence = 0.0
                try:
                    _content = (
                        best_candidate.get("full_content")
                        or best_candidate.get("unified_diff")
                        or ""
                    )
                    with SideEffectFirewall():
                        _confidence = OutputComparator().compare(
                            _content, _content, CompareMode.AST
                        )
                except Exception as _sh_exc:
                    _violations.append(str(_sh_exc))
                    _confidence = 0.0
                _sh_dur = _sh_time.monotonic() - _sh_start
                self._config.shadow_harness.record_run(_confidence)
                _shadow_result = ShadowResult(
                    confidence=_confidence,
                    comparison_mode="ast",
                    violations=tuple(_violations),
                    shadow_duration_s=_sh_dur,
                    production_match=(_confidence >= 0.7),
                    disqualified=self._config.shadow_harness.is_disqualified,
                )
                ctx = ctx.with_shadow_result(_shadow_result)
                if self._config.shadow_harness.is_disqualified:
                    logger.warning(
                        "[Orchestrator] ShadowHarness disqualified for op=%s "
                        "(confidence=%.2f, violations=%d) — proceeding to GATE with advisory",
                        ctx.op_id,
                        _confidence,
                        len(_violations),
                    )

            # ── Entropy measurement (Pillar 4: Synthetic Soul) ──────────────────
            # Compute CompositeEntropySignal from acute (this generation) +
            # chronic (historical domain) signals. Pure deterministic math.
            try:
                from backend.core.ouroboros.governance.entropy_calculator import (
                    compute_acute_signal,
                    compute_chronic_signal,
                    compute_systemic_entropy,
                    emit_entropy_capability_gap,
                    extract_domain_key,
                    EntropyQuadrant,
                )

                # Acute signal: from validation + shadow + retry data
                _shadow_conf = 1.0
                if ctx.shadow is not None:
                    _shadow_conf = getattr(ctx.shadow, "confidence", 1.0)

                _critique_errors = 0
                _critique_warnings = 0
                _critique_infos = 0
                if _episodic_memory is not None:
                    try:
                        for ep in getattr(_episodic_memory, "_episodes", []):
                            _critique_errors += getattr(ep, "error_count", 0)
                            _critique_warnings += getattr(ep, "warning_count", 0)
                            _critique_infos += getattr(ep, "info_count", 0)
                    except Exception:
                        pass

                _acute = compute_acute_signal(
                    validation_passed=best_validation.passed,
                    critique_errors=_critique_errors,
                    critique_warnings=_critique_warnings,
                    critique_infos=_critique_infos,
                    shadow_confidence=_shadow_conf,
                    retries_used=(self._config.max_generate_retries - generate_retries_remaining),
                    max_retries=self._config.max_generate_retries,
                )

                # Chronic signal: from LearningBridge history
                _domain_key = extract_domain_key(ctx.target_files, ctx.description)
                _chronic_outcomes: list = []
                if hasattr(self._stack, "learning_bridge") and self._stack.learning_bridge is not None:
                    try:
                        _history = await self._stack.learning_bridge.get_domain_history(
                            _domain_key
                        )
                        _chronic_outcomes = _history if _history else []
                    except Exception:
                        pass  # No history available — chronic signal stays neutral

                _chronic = compute_chronic_signal(_domain_key, _chronic_outcomes)

                # Fuse into systemic entropy
                _composite = compute_systemic_entropy(_acute, _chronic)

                # Log for observability (Pillar 7)
                logger.info(
                    "[Orchestrator] Entropy: acute=%.3f chronic=%.3f systemic=%.3f "
                    "quadrant=%s trigger=%s domain=%s (op=%s)",
                    _acute.normalized_score, _chronic.normalized_score,
                    _composite.systemic_score, _composite.quadrant.value,
                    _composite.should_trigger, _domain_key, ctx.op_id,
                )

                # Record in ledger
                await self._record_ledger(ctx, OperationState.GATING, {
                    "event": "entropy_measured",
                    "acute_score": round(_acute.normalized_score, 4),
                    "chronic_score": round(_chronic.normalized_score, 4),
                    "systemic_score": round(_composite.systemic_score, 4),
                    "quadrant": _composite.quadrant.value,
                    "domain_key": _domain_key,
                    "should_trigger": _composite.should_trigger,
                })

                # Act on quadrant
                if _composite.quadrant == EntropyQuadrant.IMMEDIATE_TRIGGER:
                    # Fold the entropy signal into a CapabilityGapEvent and emit
                    # it onto the GapSignalBus the CapabilityGapSensor consumes
                    # (Pillar 6 neuroplasticity). Previously severed: the old
                    # block called a non-existent singleton accessor on the bus
                    # class, swallowed as an AttributeError, so this signal
                    # reached the consumer NEVER. Now routed through the single
                    # safe seam (imported with the entropy block above).
                    emit_entropy_capability_gap(
                        op_id=ctx.op_id,
                        domain_key=_domain_key,
                        composite=_composite,
                        description=ctx.description or "",
                    )

                elif _composite.quadrant == EntropyQuadrant.FALSE_CONFIDENCE:
                    # Force sandbox validation even though validation passed
                    logger.warning(
                        "[Orchestrator] FALSE_CONFIDENCE: domain=%s has high chronic "
                        "failure rate (%.3f) despite passing validation. "
                        "Recommend sandbox re-verification. (op=%s)",
                        _domain_key, _chronic.failure_rate, ctx.op_id,
                    )

            except ImportError:
                pass  # entropy_calculator not available — degrade gracefully
            except Exception:
                logger.debug("[Orchestrator] Entropy computation failed", exc_info=True)

            # Read-only APPLY short-circuit (Manifesto §1 Boundary Principle).
            # When ctx.is_read_only is True the op is a cartography/analysis task
            # — the model's tool-round findings (including any dispatch_subagent
            # rollups) are the deliverable. GATE/APPLY/VERIFY have no semantic
            # meaning because nothing is being written. Skip straight to COMPLETE
            # with a structural terminal reason. This is the second half of the
            # cryptographic guarantee the Advisor's blast/coverage bypass rests
            # on: tool_executor refuses mutating tool calls, the orchestrator
            # refuses the APPLY transition.
            if ctx.is_read_only:
                logger.info(
                    "[Orchestrator] Read-only APPLY short-circuit op=%s — "
                    "skipping GATE/APPLY/VERIFY (no-mutation contract). "
                    "Findings are delivered via POSTMORTEM + ledger.",
                    ctx.op_id,
                )
                try:
                    await self._stack.comm.emit_decision(
                        op_id=ctx.op_id,
                        outcome="read_only_complete",
                        reason_code="read_only_complete",
                        diff_summary="",
                    )
                except Exception:
                    pass
                ctx = ctx.advance(
                    OperationPhase.COMPLETE,
                    terminal_reason_code="read_only_complete",
                    validation=best_validation,
                )
                if _serpent:
                    await _serpent.stop(success=True)
                return ctx

            # Store compact validation result in context; full output is in ledger
            ctx = ctx.advance(OperationPhase.GATE, validation=best_validation)
            logger.info(
                "[Orchestrator] Entered GATE phase for op=%s — invoking "
                "can_write policy check on target_files=%s",
                ctx.op_id,
                list(ctx.target_files)[:3],
            )

            # Heartbeat: GATE phase (Manifesto §7)
            try:
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="gate", progress_pct=75.0,
                )
            except Exception:
                pass

        # Wave 2 (5) Slice 4a.2 - GATERunner delegation gate.
        # Flag JARVIS_PHASE_RUNNER_GATE_EXTRACTED (default true — graduated;
        # the extracted runner is the shipping GATE path) routes
        # the 600-line GATE block (can_write + SecurityReviewer +
        # SimilarityGate + frozen_tier + risk ceiling + SemanticGuardian
        # + REVIEW shadow + MutationGate + MIN_RISK_TIER floor + 5a green
        # preview + 5b NOTIFY_APPLY yellow) through the extracted runner.
        # risk_tier mutates at up to 6 sites inside GATE and is threaded
        # back via PhaseResult.artifacts["risk_tier"] so APPROVE inline
        # code downstream sees the final (possibly escalated) value.
        if _phase_runner_gate_extracted():
            from backend.core.ouroboros.governance.phase_runners.gate_runner import (
                GATERunner,
            )
            logger.info("[PhaseRunnerDelegate] GATE → runner op=%s", ctx.op_id[:16])
            _gate_runner = GATERunner(self, _serpent, best_candidate, risk_tier)
            _gate_result = await _gate_runner.run(ctx)
            # Rebind risk_tier (GATE mutates it). best_candidate unchanged
            # but pass through for symmetry with other slices.
            risk_tier = _gate_result.artifacts.get("risk_tier", risk_tier)
            best_candidate = _gate_result.artifacts.get("best_candidate", best_candidate)
            if _gate_result.next_phase is None:
                # Terminal exit (gate_blocked / security_review_blocked /
                # user_rejected_safe_auto_preview / user_rejected_notify_apply)
                return _gate_result.next_ctx
            ctx = _gate_result.next_ctx
        else:
            if _serpent: _serpent.update_phase("GATE")
            # ---- Phase 5: GATE ----
            # Autonomy Gap 4 — earned-trust WIDENING, inline-GATE twin of the
            # extracted gate_runner hook. Applied at GATE ENTRY before the floor
            # stack so the immutable cage/governance floors re-clamp afterwards.
            # DEFAULT-INERT (double opt-in); wrapped so trust can never break GATE.
            try:
                from backend.core.ouroboros.governance.trust_calibration import (
                    relax_tier_for_op as _trust_relax,
                )
                risk_tier, _trust_why = _trust_relax(risk_tier, ctx)
                if _trust_why and _trust_why != "cage_excluded":
                    logger.info("[TrustCalibration] GATE %s", _trust_why)
            except Exception:  # noqa: BLE001 — trust widening must never break GATE
                pass

            # ---- REVIEW subagent → risk gate (graduated, Phase 1b) ----
            # Inline-twin parity with gate_runner: the SAME seam, at GATE entry,
            # so every hard gate below re-clamps a relaxed tier. See
            # _apply_review_gate. The legacy per-verdict REVIEW-enforce block
            # further down is retired in favour of this single call.
            risk_tier = await self._apply_review_gate(ctx, best_candidate, risk_tier)

            allowed, reason = self._stack.can_write(
                {"files": list(ctx.target_files)}
            )
            logger.info(
                "[Orchestrator] GATE can_write decision for op=%s: "
                "allowed=%s reason=%s",
                ctx.op_id, allowed, reason,
            )
            if not allowed:
                logger.warning(
                    "[Orchestrator] GATE BLOCKED: can_write=%s for op=%s files=%s",
                    reason, ctx.op_id, list(ctx.target_files)[:3],
                )
                ctx = ctx.advance(
                    OperationPhase.CANCELLED,
                    terminal_reason_code=f"gate_blocked:{reason}",
                )
                await self._record_ledger(
                    ctx,
                    OperationState.BLOCKED,
                    {"reason": f"gate_blocked:{reason}"},
                )
                return ctx

            # ---- Security Review (LLM-as-a-Judge) before APPROVE gate ----
            try:
                from backend.core.ouroboros.governance.security_reviewer import SecurityReviewer, SecurityVerdict
                # Only wire SecurityReviewer with a genuine PrimeClient — the
                # former fallback passed CandidateGenerator / provider objects
                # whose generate(context, deadline) signature crashes SecurityReviewer
                # (TypeError: generate() got an unexpected keyword argument 'prompt').
                # See orchestrator battle test bt-2026-04-10-184157 postmortem.
                _sec_client = getattr(self._stack, "prime_client", None)
                _sec_reviewer = SecurityReviewer(prime_client=_sec_client)
                if _sec_reviewer.is_enabled and best_candidate is not None:
                    _sec_result = await _sec_reviewer.review(
                        candidate=best_candidate,
                        target_files=list(ctx.target_files),
                        description=ctx.description,
                    )
                    if _sec_result.verdict == SecurityVerdict.BLOCK:
                        logger.warning(
                            "[Orchestrator] Security review BLOCKED: %s [%s]",
                            _sec_result.summary, ctx.op_id,
                        )
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="security_review_blocked",
                        )
                        await self._record_ledger(
                            ctx, OperationState.BLOCKED,
                            {"reason": "security_review_blocked", "summary": _sec_result.summary},
                        )
                        return ctx
                    elif _sec_result.verdict == SecurityVerdict.WARN:
                        logger.info(
                            "[Orchestrator] Security review WARN: %s [%s]",
                            _sec_result.summary, ctx.op_id,
                        )
                        # Emit proactive alert for security warnings (Manifesto §7)
                        try:
                            _warn_msg = type("_Msg", (), {
                                "payload": {
                                    "proactive_alert": True,
                                    "alert_title": "Security Review Warning",
                                    "alert_body": _sec_result.summary or "Potential security concern detected.",
                                    "alert_severity": "warning",
                                    "alert_source": "SecurityReviewer",
                                },
                                "op_id": ctx.op_id,
                                "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                            })()
                            for _t in getattr(self._stack.comm, "_transports", []):
                                try:
                                    await _t.send(_warn_msg)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                logger.debug("[Orchestrator] SecurityReviewer not available", exc_info=True)

            # ---- Diff-Aware Similarity Gate (Sub-project C) ----
            if best_candidate is not None:
                try:
                    from backend.core.ouroboros.governance.similarity_gate import check_similarity
                    _src_content = ""
                    if ctx.target_files:
                        _src_path = self._config.project_root / ctx.target_files[0]
                        if _src_path.exists():
                            _src_content = _src_path.read_text(encoding="utf-8", errors="replace")
                    # Extract candidate content as a string. best_candidate is a dict with
                    # either top-level `full_content` (legacy single-file) or a `files` list
                    # (multi-file). Passing the raw dict would crash similarity_gate with
                    # AttributeError: 'dict' object has no attribute 'splitlines'.
                    _cand_content = ""
                    if isinstance(best_candidate, dict):
                        _cand_content = best_candidate.get("full_content", "") or ""
                        if not _cand_content and isinstance(best_candidate.get("files"), list):
                            _target0 = ctx.target_files[0] if ctx.target_files else None
                            for _entry in best_candidate["files"]:
                                if not isinstance(_entry, dict):
                                    continue
                                if _target0 is None or _entry.get("file_path") == _target0:
                                    _cand_content = _entry.get("full_content", "") or ""
                                    if _cand_content:
                                        break
                    if _src_content and _cand_content:
                        _sim_reason = check_similarity(_cand_content, _src_content)
                        if _sim_reason is not None:
                            logger.info(
                                "[Orchestrator] GATE similarity escalation: %s [%s]",
                                _sim_reason, ctx.op_id,
                            )
                            if risk_tier is not RiskTier.APPROVAL_REQUIRED:
                                risk_tier = RiskTier.APPROVAL_REQUIRED
                            # Emit gate event for VoiceNarrator
                            try:
                                await self._stack.comm.emit_decision(
                                    op_id=ctx.op_id,
                                    outcome="escalated",
                                    reason_code="similarity_escalation",
                                    target_files=list(ctx.target_files),
                                )
                            except Exception:
                                pass
                except Exception:
                    logger.debug("[Orchestrator] Similarity gate skipped", exc_info=True)

            # Autonomy tier gate: frozen at submit() to prevent TrustGraduator race.
            # "observe" → force APPROVAL_REQUIRED regardless of risk_tier.
            _frozen_tier = getattr(ctx, "frozen_autonomy_tier", "governed")
            if _frozen_tier == "observe" and risk_tier is not RiskTier.APPROVAL_REQUIRED:
                risk_tier = RiskTier.APPROVAL_REQUIRED
                logger.info(
                    "[Orchestrator] GATE: frozen_tier=observe → APPROVAL_REQUIRED; op=%s",
                    ctx.op_id,
                )

            # Slice 119 — Bounded M10 synapse (Order-2 RSI, human-gated). On high
            # domain entropy / repeated algorithmic failures, the orchestrator
            # routes to M10 for a STRUCTURAL cognitive-upgrade proposal and FORCES
            # APPROVAL_REQUIRED — the proposal is queued PENDING the operator's
            # signature; nothing self-applies (§1 Zero-Order Doll). The Slice-104
            # recursion-depth gate bounds it independently. Gated, §33.1
            # default-FALSE → byte-identical when off; this is M10's live-path ref.
            try:
                from backend.core.ouroboros.governance.m10_synapse import (
                    evaluate_m10_routing,
                    m10_synapse_enabled,
                )
                if m10_synapse_enabled():
                    _m10_entropy = None
                    try:
                        from backend.core.ouroboros.governance.domain_entropy_engine import (
                            compute_domain_entropy,
                            domain_entropy_engine_enabled,
                        )
                        if domain_entropy_engine_enabled():
                            _m10_entropy = getattr(compute_domain_entropy(), "normalized_entropy", None)
                    except Exception:  # noqa: BLE001
                        _m10_entropy = None
                    _m10_fails = int(getattr(ctx, "recent_algorithmic_failures", 0) or 0)
                    _m10 = evaluate_m10_routing(
                        shannon_entropy=_m10_entropy, recent_algorithmic_failures=_m10_fails,
                    )
                    if _m10.route_to_m10 and risk_tier is not RiskTier.APPROVAL_REQUIRED:
                        risk_tier = RiskTier.APPROVAL_REQUIRED
                        logger.warning(
                            "[Orchestrator] M10 SYNAPSE: %s → APPROVAL_REQUIRED "
                            "(structural upgrade PENDING operator signature; "
                            "self-apply blocked); op=%s",
                            _m10.reason, ctx.op_id,
                        )
            except Exception:  # noqa: BLE001 — synapse must never break the FSM
                logger.debug("[Orchestrator] M10 synapse hook skipped", exc_info=True)

            # ---- Slice 120: Sovereign Layer-4 Roadmap Authority (escalation-only) ----
            # In unattended evidence-clock mode, the operator-signed roadmap may
            # suppress the human prompt for SAFE, explicitly-authorized scopes.
            # Here in the orchestrator we wire only the FAIL-CLOSED direction:
            # for any op the roadmap does NOT authorize for suppression — every
            # safety op (Order-2/M10, recursion-breach, governance-touch,
            # APPROVAL_REQUIRED/BLOCKED tier) and every out-of-scope op — we
            # RE-ASSERT APPROVAL_REQUIRED. The un-signable floor (§1) cannot be
            # reasoned around: no signature suppresses approval on a safety op.
            # Default-off (JARVIS_LAYER4_ROADMAP_ENABLED) → byte-identical.
            try:
                from backend.core.ouroboros.governance import layer4_roadmap_authority as _L4

                if _L4.layer4_enabled() and risk_tier is not RiskTier.APPROVAL_REQUIRED:
                    _l4_auth = _L4.load_and_verify_roadmap(now=int(time.time()))
                    _l4_scope = str(getattr(ctx, "scope", "") or getattr(ctx, "category", "") or "")
                    _l4_is_m10 = bool(getattr(ctx, "is_order2_rsi", False) or getattr(ctx, "m10_routed", False))
                    if not _L4.may_suppress_approval(
                        _l4_auth,
                        op_scope=_l4_scope,
                        risk_tier=risk_tier.name,
                        is_order2_rsi=_l4_is_m10,
                    ):
                        # Op is NOT cleared for unattended auto-resolution → the
                        # human still owns it.
                        risk_tier = RiskTier.APPROVAL_REQUIRED
                        logger.info(
                            "[Orchestrator] LAYER4: %s → APPROVAL_REQUIRED; op=%s",
                            _L4.degrade_reason(_l4_auth), ctx.op_id,
                        )
            except Exception:
                logger.debug("[Orchestrator] Layer-4 authority hook skipped", exc_info=True)

            # ---- Risk floor override (REPL /risk command) ----
            # JARVIS_RISK_CEILING env var sets the minimum risk tier floor.
            # E.g. /risk notify_apply → everything is at least NOTIFY_APPLY.
            _risk_floor_str = os.environ.get("JARVIS_RISK_CEILING", "")
            if _risk_floor_str:
                _floor_map = {
                    "SAFE_AUTO": RiskTier.SAFE_AUTO,
                    "NOTIFY_APPLY": RiskTier.NOTIFY_APPLY,
                    "APPROVAL_REQUIRED": RiskTier.APPROVAL_REQUIRED,
                }
                _floor = _floor_map.get(_risk_floor_str.upper())
                if _floor is not None and risk_tier.value < _floor.value:
                    logger.info(
                        "[Orchestrator] GATE: risk floor %s → escalating %s to %s; op=%s",
                        _risk_floor_str, risk_tier.name, _floor.name, ctx.op_id,
                    )
                    risk_tier = _floor

            # ---- SemanticGuardian: deterministic pre-APPLY pattern check ----
            #
            # Closes the SAFE_AUTO blast-radius gap (Priority 3 audit):
            # risk_engine.py classifies on size (blast radius / file count /
            # test confidence) only — a syntactically-valid but semantically-
            # inverted candidate (flipped boolean, removed import, collapsed
            # body, hardcoded credential, inverted test assertion, loosened
            # perms …) lands as SAFE_AUTO and auto-applies while the operator
            # is asleep. The guardian runs 10 deterministic AST/regex
            # patterns on (pre-apply on-disk content) vs (candidate content)
            # and, if any fire, upgrades the tier:
            #
            #   hard detection → APPROVAL_REQUIRED (force human gate)
            #   soft detection → NOTIFY_APPLY      (force 5s preview window)
            #
            # Pure-deterministic, no LLM, ~10ms per candidate. Master switch
            # JARVIS_SEMANTIC_GUARD_ENABLED (default on).
            _guardian_findings: list = []
            if best_candidate is not None:
                try:
                    from backend.core.ouroboros.governance.semantic_guardian import (
                        SemanticGuardian,
                        recommend_tier_floor,
                    )
                    _guardian = SemanticGuardian()
                    # Build (path, old, new) triples from the candidate. For
                    # multi-file candidates the orchestrator already has
                    # _iter_candidate_files; we replicate its unpacking here
                    # so we don't need to thread ctx through.
                    _pairs: list = []
                    _candidate_files = best_candidate.get("files") if isinstance(
                        best_candidate.get("files"), list,
                    ) else None
                    if _candidate_files:
                        _iter = [
                            (entry.get("file_path", ""), entry.get("full_content", ""))
                            for entry in _candidate_files
                            if isinstance(entry, dict)
                        ]
                    else:
                        _iter = [(
                            best_candidate.get("file_path", ""),
                            best_candidate.get("full_content", ""),
                        )]
                    # Anti-Venom S2 — guardian git-HEAD baseline. Venom may have
                    # already written files DURING generation (in-loop edits via
                    # edit_file/write_file). For those paths the on-disk content
                    # is the *post-write* state, so an on-disk read would compare
                    # candidate→candidate and the guardian would see no change.
                    # For any path Venom touched in this op we read the ORIGINAL
                    # from `git show HEAD:<path>` so the guardian compares
                    # original→candidate. Fail-SOFT: a new file (not in HEAD) or
                    # any git error yields _old="" (treated as creation).
                    _venom_paths: set = set()
                    try:
                        _vhist = getattr(generation, "venom_edit_history", ()) or ()
                        _venom_paths = {
                            e.get("path", "")
                            for e in _vhist
                            if isinstance(e, dict) and e.get("path")
                        }
                    except Exception:
                        _venom_paths = set()

                    for _path, _new in _iter:
                        if not _path or not isinstance(_new, str):
                            continue
                        _old = ""
                        # Repo-relative form for venom-history matching + git show.
                        _rel = _path
                        try:
                            _pp = Path(_path)
                            if _pp.is_absolute():
                                _rel = str(_pp.relative_to(self._config.project_root))
                        except Exception:
                            _rel = _path
                        if _rel in _venom_paths or _path in _venom_paths:
                            # In-loop write landed → baseline from git HEAD.
                            try:
                                _proc = subprocess.run(
                                    ["git", "show", f"HEAD:{_rel}"],
                                    cwd=str(self._config.project_root),
                                    capture_output=True,
                                    text=True,
                                    timeout=10,
                                )
                                _old = _proc.stdout if _proc.returncode == 0 else ""
                            except Exception:
                                _old = ""
                        else:
                            try:
                                _abs = (
                                    self._config.project_root / _path
                                    if not Path(_path).is_absolute()
                                    else Path(_path)
                                )
                                if _abs.is_file():
                                    _old = _abs.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                _old = ""
                        _pairs.append((_path, _old, _new))

                    # Time the whole batch so operators can detect a pattern
                    # detector regressing into a slow path (Track A telemetry).
                    _sg_t0 = time.monotonic()
                    _guardian_findings = _guardian.inspect_batch(_pairs)
                    _sg_duration_ms = int((time.monotonic() - _sg_t0) * 1000)

                    # Compute structured telemetry fields BEFORE any tier
                    # upgrade so ``risk_before`` reflects the classifier's
                    # verdict pre-guardian. The single INFO contract below
                    # fires on every op (hit OR clean) so downstream grep /
                    # aggregation pipelines have a stable one-line record.
                    _hard_count = sum(
                        1 for f in _guardian_findings if f.severity == "hard"
                    )
                    _soft_count = sum(
                        1 for f in _guardian_findings if f.severity == "soft"
                    )
                    _risk_before_name = risk_tier.name

                    _floor_name = recommend_tier_floor(_guardian_findings)
                    _upgrade: Optional[RiskTier] = None
                    if _floor_name is not None:
                        _upgrade_map = {
                            "notify_apply": RiskTier.NOTIFY_APPLY,
                            "approval_required": RiskTier.APPROVAL_REQUIRED,
                        }
                        _upgrade = _upgrade_map.get(_floor_name)
                        if _upgrade is not None and risk_tier.value < _upgrade.value:
                            risk_tier = _upgrade
                        else:
                            _upgrade = None  # floor wasn't stricter; no upgrade

                    # Slice 6 Task 5 — attribution scope gate. Reuses ``_pairs``
                    # (the EXACT filtered file list the guardian just batch-
                    # inspected) so gate and guardian provably agree on scope.
                    # An unresolved-attribution op whose candidate mutates ONLY
                    # test loci is the Run-16 blind class — escalate to human
                    # approval (NOT reject: the test itself may be the legitimate
                    # fix target). Mirrors the guardian's stricter-wins hard-
                    # finding escalation above; the resulting tier is captured in
                    # the [SemanticGuard] risk_after telemetry below. Helper is
                    # fail-soft (never fatal). NOTE: the SHIPPING path is the
                    # extracted GATERunner (gate_runner.py) — this inline twin
                    # covers JARVIS_PHASE_RUNNER_GATE_EXTRACTED=false.
                    risk_tier, _attr_violation = _attribution_scope_risk_floor(
                        ctx, [_p for (_p, _o, _n) in _pairs], risk_tier,
                        repo_root=str(self._config.project_root),
                    )
                    if _attr_violation:
                        logger.warning(
                            "[Attribution] gate: %s op=%s",
                            _attr_violation, ctx.op_id,
                        )
                    # Slice 15 T4 — adaptive-ceiling halt (inline twin).
                    risk_tier, _vc_note = _value_ceiling_risk_floor(
                        ctx, risk_tier,
                    )
                    if _vc_note:
                        logger.warning(
                            "[SignalValue] gate: %s op=%s", _vc_note, ctx.op_id,
                        )

                    # Slice 8 — test-only NOTIFY_APPLY floor. Companion to
                    # the Slice 6 Task 5 escalation above: a RESOLVED-
                    # attribution op whose candidate mutates ONLY test loci
                    # is a legitimate lane (Slice 7's subset waiver) but a
                    # sensitive one (assertion-weakening test edits
                    # auto-apply green) — floor at NOTIFY_APPLY, never
                    # blocking, never downgrading. Same ``_pairs``-derived
                    # scope + repo_root as the escalation above so both
                    # gates provably agree on scope. Fail-soft.
                    risk_tier, _attr_test_only = _attribution_test_only_notify_floor(
                        ctx, [_p for (_p, _o, _n) in _pairs], risk_tier,
                        repo_root=str(self._config.project_root),
                    )
                    if _attr_test_only:
                        logger.info(
                            "[Attribution] notify: %s op=%s",
                            _attr_test_only, ctx.op_id,
                        )

                    # Epistemic Humility floor (inline twin; shipping
                    # path is gate_runner). An op whose blast radius
                    # resolved to provenance=unknown at CLASSIFY is
                    # floored at NOTIFY_APPLY — uncertainty surfaces
                    # to the operator, never a silent green. Fail-soft.
                    risk_tier, _epi_note = _advisor_epistemic_notify_floor(
                        ctx, risk_tier,
                    )
                    if _epi_note:
                        logger.info(
                            "[Advisor] gate: %s op=%s", _epi_note, ctx.op_id,
                        )

                    # Stable structured line — always emitted. Fields are
                    # intentionally key=value so a simple split("=") parser
                    # can build rollup counters (top patterns, top files,
                    # FP rate estimate). Track A observability contract.
                    _pattern_names = (
                        ",".join(sorted({f.pattern for f in _guardian_findings}))
                        if _guardian_findings else "none"
                    )
                    logger.info(
                        "[SemanticGuard] op=%s findings=%d hard=%d soft=%d "
                        "patterns=[%s] risk_before=%s risk_after=%s "
                        "duration_ms=%d files_scanned=%d",
                        ctx.op_id,
                        len(_guardian_findings),
                        _hard_count, _soft_count,
                        _pattern_names,
                        _risk_before_name, risk_tier.name,
                        _sg_duration_ms,
                        len(_pairs),
                    )
                except Exception:
                    # Anti-Venom Lock A — FAIL CLOSED. A guardian crash used to
                    # be swallowed at DEBUG (fail-OPEN): empty findings → no
                    # tier floor → a SAFE_AUTO candidate auto-applied with the
                    # semantic net silently down. Now: inject one hard sentinel
                    # finding and force APPROVAL_REQUIRED so the op parks at GATE
                    # for a human (reuses the existing approval branch — no new
                    # FSM phase). risk_tier is set directly (the authoritative
                    # arm); _guardian_findings carries the sentinel for
                    # observability + any downstream findings consumer.
                    logger.warning(
                        "[Orchestrator] SemanticGuardian raised — FAILING "
                        "CLOSED; APPROVAL_REQUIRED op=%s",
                        ctx.op_id,
                        exc_info=True,
                    )
                    risk_tier = RiskTier.APPROVAL_REQUIRED
                    _guardian_findings = [_SENTINEL_GUARDIAN_CRASH]

            # ---- REVIEW subagent → risk gate ----
            # Retired: the REVIEW verdict is now dispatched + folded into the
            # tier once, at GATE ENTRY, via self._apply_review_gate (escalate on
            # reject + authorize a routine approval down-level on a clean
            # approve), identically on this inline twin and the extracted
            # gate_runner. Re-running it here would double the review cost and
            # could re-lower a tier a hard gate between entry and here just
            # raised, so it is NOT repeated.

            # ---- MutationGate: APPLY-phase execution boundary (cached) ----
            #
            # Closes the "tests pass != tests test" gap empirically surfaced
            # by the Session W calibration (28.6% mutation score on green
            # test suite). For operator-allowlisted critical paths only:
            #
            #   * Enumerate deterministic AST mutants (cached by content hash).
            #   * For each mutant, run the scoped test suite; cache outcomes
            #     by (sut_hash, tests_hash) so repeat ops on unchanged files
            #     are near-free.
            #   * Map score → decision:
            #       score >= allow_threshold  (default 0.75) → no change
            #       score in [block, allow)                  → force APPROVAL_REQUIRED
            #       score <  block_threshold (default 0.40)  → force BLOCKED
            #
            # Authority split (Manifesto §1): the tester measures; this
            # module decides; the orchestrator enforces. The gate never
            # auto-improves tests, never short-circuits VALIDATE, never
            # runs on non-critical paths (cost would be prohibitive).
            #
            # Master switch JARVIS_MUTATION_GATE_ENABLED (default 0).
            if best_candidate is not None:
                try:
                    from backend.core.ouroboros.governance import mutation_gate as _mg
                    if _mg.gate_enabled():
                        _mg_allowlist = _mg.load_allowlist()
                        # Reuse the _iter already built for SemanticGuardian.
                        _candidate_pairs = []
                        _candidate_files_mg = best_candidate.get("files") if isinstance(
                            best_candidate.get("files"), list,
                        ) else None
                        if _candidate_files_mg:
                            _candidate_pairs = [
                                entry.get("file_path", "")
                                for entry in _candidate_files_mg
                                if isinstance(entry, dict)
                            ]
                        else:
                            _single = best_candidate.get("file_path", "")
                            if _single:
                                _candidate_pairs = [_single]
                        # Filter to critical-only.
                        _critical = [
                            Path(p) for p in _candidate_pairs
                            if _mg.is_path_critical(Path(p), allowlist=_mg_allowlist)
                        ]
                        if _critical:
                            _verdicts = []
                            for _sp in _critical:
                                _abs_sp = (
                                    self._config.project_root / _sp
                                    if not _sp.is_absolute() else _sp
                                )
                                # Caller supplies tests — a path-correlated
                                # discovery helper keeps the wiring minimal
                                # (Session W style: tests/test_<stem>*.py).
                                _tests = await self._discover_tests_for_gate(_sp)
                                _verdicts.append(
                                    _mg.evaluate_file(_abs_sp, _tests)
                                )
                            if _verdicts:
                                _merged = _mg.merge_verdicts(_verdicts)
                                _risk_before_mg = risk_tier.name
                                _mg_mode = _mg.gate_mode()
                                _enforced = (_mg_mode == _mg.MODE_ENFORCE)
                                _applied_change = ""
                                if _enforced:
                                    if _merged.decision == "block":
                                        risk_tier = RiskTier.BLOCKED
                                        _applied_change = (
                                            f"{_risk_before_mg}->BLOCKED"
                                        )
                                    elif _merged.decision == "upgrade_to_approval":
                                        if risk_tier.value < RiskTier.APPROVAL_REQUIRED.value:
                                            risk_tier = RiskTier.APPROVAL_REQUIRED
                                            _applied_change = (
                                                f"{_risk_before_mg}->APPROVAL_REQUIRED"
                                            )
                                # Ledger EVERY verdict regardless of mode so
                                # shadow-mode operators accumulate data for
                                # the enforce-mode flip decision.
                                try:
                                    _mg.append_ledger(
                                        op_id=ctx.op_id, verdict=_merged,
                                        mode=_mg_mode, enforced=_enforced,
                                        applied_tier_change=_applied_change,
                                    )
                                except Exception:
                                    logger.debug(
                                        "[MutationGate] ledger append skipped",
                                        exc_info=True,
                                    )
                                logger.info(
                                    "[MutationGate] op=%s mode=%s enforced=%s "
                                    "decision=%s score=%.2f grade=%s "
                                    "caught=%d/%d survivors=%d cache_hits=%d "
                                    "cache_misses=%d duration=%.1fs "
                                    "risk_before=%s risk_after=%s",
                                    ctx.op_id, _mg_mode, _enforced,
                                    _merged.decision, _merged.score,
                                    _merged.grade, _merged.caught,
                                    _merged.total_mutants, len(_merged.survivors),
                                    _merged.cache_hits, _merged.cache_misses,
                                    _merged.duration_s,
                                    _risk_before_mg, risk_tier.name,
                                )
                except Exception:
                    logger.debug(
                        "[Orchestrator] MutationGate skipped",
                        exc_info=True,
                    )

            # ---- MIN_RISK_TIER floor (paranoia mode + quiet hours) ----
            #
            # Separate from JARVIS_RISK_CEILING above — that knob is scoped
            # to the /risk REPL command. This floor composes THREE operator
            # signals into a single tier floor:
            #
            #   JARVIS_MIN_RISK_TIER=notify_apply  (explicit)
            #   JARVIS_PARANOIA_MODE=1              (shortcut for notify_apply)
            #   JARVIS_AUTO_APPLY_QUIET_HOURS=22-7 (time-of-day window)
            #
            # The strictest of the three applies. Flipping PARANOIA_MODE or
            # QUIET_HOURS before going to sleep guarantees zero SAFE_AUTO
            # auto-applies land overnight.
            try:
                from backend.core.ouroboros.governance.risk_tier_floor import (
                    apply_floor_to_name,
                    floor_reason,
                )
                _cur_name = risk_tier.name.lower()
                # §37 Tier 2 #13 Slice 3 (2026-05-07) — pass op_id
                # so the confidence-derived floor (master-flag-gated)
                # composes with the existing env/paranoia/quiet-hours
                # floors. Low-confidence per-tool observations
                # (UNKNOWN/LOW/MEDIUM band) clamp to NOTIFY_APPLY
                # before auto-apply — load-bearing Antivenom defense
                # against Move 9 single-roll Quine-class hallucinations.
                # §40 Wave 2 #5 (2026-05-10) — pass target_files so the
                # RRD §1 Boundary recursion-depth gate composes into
                # the strictest-wins ladder. Ops touching the canonical
                # governance directory force APPROVAL_REQUIRED — closes
                # the infinite-regress risk where an autonomous proposer
                # could modify the cage layer without operator review.
                _op_id = getattr(ctx, "op_id", "") or ""
                _target_files = getattr(ctx, "target_files", ()) or ()
                _effective, _applied = apply_floor_to_name(
                    _cur_name,
                    op_id=_op_id,
                    target_files=_target_files,
                )
                if _applied is not None:
                    _floor_tier_map = {
                        "safe_auto": RiskTier.SAFE_AUTO,
                        "notify_apply": RiskTier.NOTIFY_APPLY,
                        "approval_required": RiskTier.APPROVAL_REQUIRED,
                        "blocked": RiskTier.BLOCKED,
                    }
                    _tgt = _floor_tier_map.get(_effective)
                    if _tgt is not None and risk_tier.value < _tgt.value:
                        logger.info(
                            "[Orchestrator] GATE: MIN_RISK_TIER floor → %s→%s "
                            "op=%s reason=%s",
                            risk_tier.name, _tgt.name, ctx.op_id,
                            floor_reason(
                                op_id=_op_id,
                                target_files=_target_files,
                            ),
                        )
                        risk_tier = _tgt
            except Exception:
                logger.debug(
                    "[Orchestrator] MIN_RISK_TIER floor skipped",
                    exc_info=True,
                )

            # ---- RR Pass B Slice 2b: ORDER_2_GOVERNANCE floor ----
            # Single-source seam (Slice 1 drift repair, 2026-07-18): identical
            # call on BOTH GATE paths — the drift audit found this floor wired
            # ONLY in the extracted gate_runner, so this inline kill-switch path
            # silently dropped the Order-2 governance floor.
            try:
                from backend.core.ouroboros.governance.meta.order2_classifier import (  # noqa: E501
                    apply_order2_floor_safe as _apply_order2_floor_safe,
                )
                risk_tier = _apply_order2_floor_safe(
                    risk_tier, list(ctx.target_files), op_id=ctx.op_id,
                )
            except Exception:  # noqa: BLE001 — import fault must never break GATE
                logger.debug("[Orchestrator] ORDER_2 floor skipped", exc_info=True)

            # ── Slice 101 Phase 5: Proof Carrier (pre-APPLY proof artifact) ──
            # Single-source seam (Slice 1 drift repair): the shared
            # emit_gate_proof_carrier helper is called identically on BOTH GATE
            # paths. Master JARVIS_PROOF_CARRIER_ENABLED §33.1 default-FALSE →
            # DISABLED (zero cost) when off. NEVER raises.
            try:
                from backend.core.ouroboros.governance.proof_carrier_transport import (  # noqa: E501
                    emit_gate_proof_carrier as _emit_gate_proof_carrier,
                )
                _emit_gate_proof_carrier(
                    getattr(ctx, "op_id", "") or "",
                    list(getattr(ctx, "target_files", ()) or ()),
                )
            except Exception:  # noqa: BLE001 — proof artifact must never touch GATE
                pass

            # ---- Phase 5a-green: SAFE_AUTO diff preview (Green — when human is watching) ----
            # Mythos §7.4 UX: when a human is watching (TTY or explicit flag),
            # show a brief diff preview even for Green ops so the operator can
            # /reject if they spot something wrong. The delay is shorter than
            # NOTIFY_APPLY because Green is inherently lower risk.
            if risk_tier is RiskTier.SAFE_AUTO and _human_is_watching():
                _green_delay_s = float(
                    os.environ.get("JARVIS_SAFE_AUTO_PREVIEW_DELAY_S", "2")
                )
                if best_candidate is not None and _green_delay_s > 0:
                    _diff_preview = (
                        best_candidate.get("unified_diff")
                        or best_candidate.get("full_content", "")
                    )
                    if _diff_preview:
                        try:
                            for _t in getattr(self._stack.comm, "_transports", []):
                                try:
                                    _preview_msg = type("_Msg", (), {
                                        "payload": {
                                            "phase": "safe_auto_diff_preview",
                                            "diff_preview": str(_diff_preview)[:4000],
                                            "delay_s": _green_delay_s,
                                            "target_files": list(ctx.target_files),
                                        },
                                        "op_id": ctx.op_id,
                                        "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                                    })()
                                    await _t.send(_preview_msg)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        logger.info(
                            "[Orchestrator] SAFE_AUTO diff preview shown (human watching), "
                            "waiting %.0fs for /reject; op=%s",
                            _green_delay_s, ctx.op_id,
                        )
                        await asyncio.sleep(_green_delay_s)
                        # Check if user cancelled during the preview window
                        if self._is_cancel_requested(ctx.op_id):
                            ctx = ctx.advance(
                                OperationPhase.CANCELLED,
                                terminal_reason_code="user_rejected_safe_auto_preview",
                            )
                            await self._record_ledger(
                                ctx, OperationState.FAILED,
                                {"reason": "user_rejected_safe_auto_preview"},
                            )
                            return ctx

            # ---- Phase 5b: NOTIFY_APPLY (Yellow — auto-apply with prominent CLI notice + diff preview) ----
            if risk_tier is RiskTier.NOTIFY_APPLY:
                _reason = getattr(ctx, "risk_reason_code", "notify_apply")
                logger.info(
                    "[Orchestrator] GATE: NOTIFY_APPLY (Yellow) — auto-applying with notice; op=%s reason=%s",
                    ctx.op_id, _reason,
                )
                try:
                    await self._stack.comm.emit_decision(
                        op_id=ctx.op_id,
                        outcome="notify_apply",
                        reason_code=_reason,
                        target_files=list(ctx.target_files),
                    )
                except Exception:
                    pass

                # Render diff preview in CLI before auto-apply.
                #
                # V1 rich preview: file tree + per-file panels + status
                # badges + live countdown + cancel polling. Safe fallback
                # to the legacy plain-sleep path on TTY-absent / env-off /
                # any render failure — NOTIFY_APPLY behavior is preserved
                # exactly in those cases. See diff_preview.py for the
                # authority / kill-switch / dump-path contract.
                _notify_delay_s = float(os.environ.get("JARVIS_NOTIFY_APPLY_DELAY_S", "5"))
                if best_candidate is not None and _notify_delay_s > 0:
                    _changes: list = []
                    try:
                        from backend.core.ouroboros.battle_test.diff_preview import (
                            build_changes_from_candidate,
                        )
                        _changes = build_changes_from_candidate(
                            best_candidate, self._config.project_root,
                        )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] build_changes_from_candidate failed; "
                            "using legacy plain preview",
                            exc_info=True,
                        )
                        _changes = []

                    # Resolve the SerpentFlow instance from the stack. When
                    # absent (headless / non-battle-test harness), take the
                    # plain asyncio.sleep path — behavior identical to legacy.
                    _serpent = getattr(self._stack, "serpent_flow", None)
                    _cancel_check = lambda: self._is_cancel_requested(ctx.op_id)
                    _cancelled = False

                    # ── Gap #4 Slice 3: review-branch coordinator ──
                    # Master-flag-gated short-circuit. When enabled, route
                    # the candidate through DiffArchive + ReviewBranchManager
                    # for IDE-native diff review (VS Code source control
                    # compares HEAD vs ouroboros/preview/{op-id}). Operator
                    # decision (accept/reject/timeout) drives the cancellation
                    # flag the same way the legacy preview's _cancel_check
                    # does — so downstream APPLY routing is unchanged.
                    # SKIPPED / FAILED outcomes fall through to the legacy
                    # rich-preview / plain-sleep path below.
                    _review_handled = False
                    try:
                        from backend.core.ouroboros.governance.review_coordinator import (
                            get_default_coordinator,
                            is_master_flag_enabled as _review_flag_on,
                            ReviewDecision,
                        )
                        from backend.core.ouroboros.governance.review_branch_manager import (
                            ReviewBranchManager,
                        )
                        if _review_flag_on() and _changes:
                            _coordinator = get_default_coordinator()
                            if _coordinator.branch_manager is None:
                                _coordinator.attach_branch_manager(
                                    ReviewBranchManager(self._config.project_root),
                                )
                            _files_for_review = [
                                (c.path, c.new_content)
                                for c in _changes
                                if getattr(c, "status", "") != "deleted"
                            ]
                            if _files_for_review:
                                _review = await _coordinator.coordinate_review(
                                    ctx.op_id, _files_for_review,
                                    risk_tier="notify_apply",
                                    summary=_reason or "",
                                    cancel_check=_cancel_check,
                                )
                                if _review.decision in (
                                    ReviewDecision.ACCEPTED,
                                    ReviewDecision.REJECTED,
                                    ReviewDecision.EXPIRED,
                                ):
                                    _cancelled = not _review.decision.implies_apply
                                    _review_handled = True
                                    logger.info(
                                        "[Orchestrator] review-branch decision "
                                        "op=%s decision=%s elapsed=%.1fs",
                                        ctx.op_id, _review.decision.value,
                                        _review.elapsed_s,
                                    )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] review_coordinator hook failed; "
                            "falling through to legacy preview",
                            exc_info=True,
                        )
                    # When _review_handled, the legacy preview branches
                    # below short-circuit — see the elif chain. _cancelled
                    # carries the operator decision either way.

                    if _review_handled:
                        # Review-branch flow already produced the
                        # _cancelled decision — skip both legacy preview
                        # paths.
                        pass
                    elif _serpent is not None and hasattr(_serpent, "show_notify_apply_preview"):
                        logger.info(
                            "[Orchestrator] NOTIFY_APPLY rich preview — op=%s "
                            "files=%d delay=%.1fs",
                            ctx.op_id, len(_changes), _notify_delay_s,
                        )
                        try:
                            _cancelled = await _serpent.show_notify_apply_preview(
                                op_id=ctx.op_id,
                                reason=_reason,
                                changes=_changes,
                                delay_s=_notify_delay_s,
                                cancel_check=_cancel_check,
                            )
                        except Exception:
                            logger.debug(
                                "[Orchestrator] rich NOTIFY_APPLY preview raised; "
                                "plain-sleep fallback",
                                exc_info=True,
                            )
                            await asyncio.sleep(_notify_delay_s)
                            _cancelled = _cancel_check()
                    else:
                        # Legacy path preserved: emit heartbeat + sleep +
                        # post-sleep cancel check.
                        _diff_preview = (
                            best_candidate.get("unified_diff")
                            or best_candidate.get("full_content", "")
                        )
                        if _diff_preview:
                            try:
                                for _t in getattr(self._stack.comm, "_transports", []):
                                    try:
                                        _preview_msg = type("_Msg", (), {
                                            "payload": {
                                                "phase": "notify_apply_diff",
                                                "diff_preview": str(_diff_preview)[:4000],
                                                "delay_s": _notify_delay_s,
                                                "target_files": list(ctx.target_files),
                                            },
                                            "op_id": ctx.op_id,
                                            "msg_type": type("_T", (), {"value": "HEARTBEAT"})(),
                                        })()
                                        await _t.send(_preview_msg)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        logger.info(
                            "[Orchestrator] NOTIFY_APPLY diff preview shown, "
                            "waiting %.0fs for /reject",
                            _notify_delay_s,
                        )
                        await asyncio.sleep(_notify_delay_s)
                        _cancelled = _cancel_check()

                    if _cancelled:
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="user_rejected_notify_apply",
                        )
                        await self._record_ledger(
                            ctx, OperationState.FAILED,
                            {"reason": "user_rejected_notify_apply"},
                        )
                        return ctx

        # Wave 2 (5) Slice 4b - combined APPROVE+APPLY+VERIFY delegation gate.
        # Flag JARVIS_PHASE_RUNNER_SLICE4B_EXTRACTED (default true; explicit
        # =false is the kill switch reverting to this inline block) routes
        # the ~1150-line APPROVE+APPLY+VERIFY block (including 7.5 INFRA +
        # 8a scoped tests + 8b auto-commit + 8b2 hot-reload + 8c self-critique
        # + 8d visual VERIFY) through Slice4bRunner. Single combined gate
        # because the three phases are deeply interleaved. t_apply is
        # threaded via artifacts for COMPLETERunner's canary latency.
        if _phase_runner_slice4b_extracted():
            from backend.core.ouroboros.governance.phase_runners.slice4b_runner import (
                Slice4bRunner,
            )
            logger.info("[PhaseRunnerDelegate] APPROVE+APPLY+VERIFY → Slice4bRunner op=%s", ctx.op_id[:16])
            # Emit the APPLY FSM-phase SSE (auditor's CLASSIFY->APPLY witness). Fail-soft.
            try:
                from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501,PLC0415
                    publish_fsm_phase_for_ctx,
                )
                publish_fsm_phase_for_ctx(ctx, "APPLY")
            except Exception:  # noqa: BLE001
                pass
            _slice4b_runner = Slice4bRunner(self, _serpent, best_candidate, risk_tier)
            _slice4b_result = await _slice4b_runner.run(ctx)
            # Rebind _t_apply (consumed by COMPLETERunner downstream).
            _t_apply = _slice4b_result.artifacts.get("t_apply", 0.0)
            if _slice4b_result.next_phase is None:
                # Terminal exit from APPROVE/APPLY/VERIFY (one of ~14 paths)
                return _slice4b_result.next_ctx
            ctx = _slice4b_result.next_ctx
        else:
            # ---- Phase 6: APPROVE (conditional) ----
            if risk_tier is RiskTier.APPROVAL_REQUIRED:
                # New: async PR review path. Opt-in via JARVIS_ORANGE_PR_ENABLED.
                # When enabled, we file a GitHub PR on a review branch instead of
                # blocking the loop. On any failure, we fall back to the existing
                # CLI approval provider path.
                try:
                    from backend.core.ouroboros.governance.orange_pr_reviewer import (
                        OrangePRReviewer,
                        is_orange_pr_enabled,
                    )
                    _orange_pr_on = is_orange_pr_enabled()
                except Exception:
                    _orange_pr_on = False

                if _orange_pr_on:
                    try:
                        _files_for_pr = self._iter_candidate_files(best_candidate)
                        # Iron Triad (Task 13b): when the enforcer is armed, run
                        # Gate (1) + Gate (2) inside an ISOLATED worktree so this
                        # ONE op assembles the branch-bound chain that Gate (3) +
                        # create_review_pr then verify. Default-OFF byte-
                        # identical: all pipeline machinery imports inside the
                        # guard, tokens come from ctx (None on the Orange path)
                        # exactly as today.
                        _pr_chain = getattr(ctx, "proof_chain", None)
                        _expected_branch = None
                        _sbx_tok = getattr(ctx, "sandbox_token", None)
                        _blast_tok = getattr(ctx, "blast_token", None)
                        if os.environ.get("JARVIS_A1_TOKEN_ENFORCER_ENABLED", "false").strip().lower() in ("1", "true", "yes"):
                            from backend.core.ouroboros.governance.autonomous_pr_pipeline import (
                                run_pr_gate_pipeline,
                                PRGatePipelineError,
                            )
                            from backend.core.ouroboros.governance.dag_capability_token import (
                                DAGProofChain,
                            )
                            _pr_chain = _pr_chain or DAGProofChain()
                            try:
                                _pipe = await run_pr_gate_pipeline(
                                    op_id=ctx.op_id,
                                    candidate_files=list(_files_for_pr),
                                    repo_root=str(self._config.project_root),
                                    chain=_pr_chain,
                                )
                                _sbx_tok, _blast_tok, _expected_branch = (
                                    _pipe.sandbox_token,
                                    _pipe.blast_token,
                                    _pipe.branch_context,
                                )
                            except PRGatePipelineError as _pe:
                                logger.warning(
                                    "[A1-PR] op=%s gate pipeline rejected "
                                    "candidate: %s -> no PR",
                                    ctx.op_id, _pe,
                                )
                                ctx = ctx.advance(
                                    OperationPhase.POSTMORTEM,
                                    terminal_reason_code="pr_gate_rejected",
                                )
                                await self._record_ledger(
                                    ctx,
                                    OperationState.FAILED,
                                    {"reason": "pr_gate_rejected"},
                                )
                                return ctx
                        _reviewer = OrangePRReviewer(self._config.project_root)
                        _pr_result = await _reviewer.create_review_pr(
                            op_id=ctx.op_id,
                            description=ctx.description,
                            files=_files_for_pr,
                            evidence={
                                "risk_tier": risk_tier.name,
                                "target_files": list(ctx.target_files),
                                "file_count": len(_files_for_pr),
                            },
                            risk_tier_name=risk_tier.name,
                            chain=_pr_chain,
                            sandbox_token=_sbx_tok,
                            blast_token=_blast_tok,
                            expected_branch_context=_expected_branch,
                        )
                    except Exception:
                        logger.exception(
                            "[Orchestrator] Orange PR reviewer raised for op=%s; "
                            "falling back to CLI approval",
                            ctx.op_id,
                        )
                        _pr_result = None

                    if _pr_result is not None:
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="pending_pr_review",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.GATING,
                            {
                                "event": "orange_pr_created",
                                "pr_url": _pr_result.url,
                                "branch": _pr_result.branch,
                                "base_branch": _pr_result.base_branch,
                                "risk_tier": risk_tier.name,
                            },
                        )
                        logger.info(
                            "[Orchestrator] op=%s handed off to async PR review: %s",
                            ctx.op_id, _pr_result.url,
                        )
                        return ctx
                    # Fall through to the CLI approval path on PR creation failure.
                    logger.warning(
                        "[Orchestrator] op=%s Orange PR creation failed; "
                        "using CLI approval fallback",
                        ctx.op_id,
                    )

                if self._approval_provider is None:
                    # No approval provider available -> CANCELLED
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="approval_required_but_no_provider",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "approval_required_but_no_provider"},
                    )
                    return ctx

                ctx = ctx.advance(OperationPhase.APPROVE)
                await self._record_ledger(
                    ctx,
                    OperationState.GATING,
                    {"waiting_approval": True, "risk_tier": risk_tier.name},
                )

                # Notify via comm channel (TUI + voice will receive this)
                try:
                    await self._stack.comm.emit_heartbeat(
                        op_id=ctx.op_id,
                        phase="approve",
                        progress_pct=0.0,
                    )
                except Exception:
                    logger.debug(
                        "Comm heartbeat failed for op=%s", ctx.op_id, exc_info=True
                    )

                request_id = await self._approval_provider.request(ctx)
                # The gate ASKS now. `await_decision` remains the authority on
                # the outcome — it owns the timeout, the EXPIRED stamp and the
                # ledger semantics — and the operator path is raced alongside
                # it as a faster route to the SAME decision: answering `y`
                # calls the provider's own approve(), which sets the event
                # await_decision is already waiting on.
                #
                # Before this, the gate emitted a comm heartbeat nothing
                # rendered and then sat silently until it expired. Nothing
                # hung; nobody was ever asked.
                decision: ApprovalResult = await _await_approval_with_operator(
                    self, request_id, ctx,
                )

                if decision.status is ApprovalStatus.EXPIRED:
                    ctx = ctx.advance(
                        OperationPhase.EXPIRED,
                        terminal_reason_code="approval_expired",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "approval_expired"},
                    )
                    return ctx

                if decision.status is ApprovalStatus.REJECTED:
                    _reject_reason = getattr(decision, "reason", "") or ""
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code="approval_rejected",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {
                            "reason": "approval_rejected",
                            "approver": decision.approver,
                            "rejection_reason": _reject_reason,
                        },
                    )

                    # P2.2: Capture rejection as a session lesson so the model
                    # learns what the human doesn't want within this session.
                    _files_short = ", ".join(
                        p.rsplit("/", 1)[-1] for p in ctx.target_files[:3]
                    )
                    _reason_tag = _reject_reason[:80] if _reject_reason else "no reason given"
                    self._add_session_lesson(
                        "code",
                        f"[REJECTED] {ctx.description[:60]} ({_files_short}) "
                        f"— human rejected: {_reason_tag}. "
                        f"Avoid this approach in future operations.",
                        op_id=ctx.op_id,
                    )

                    # P2.2: Feed rejection into NegativeConstraintStore for
                    # cross-session learning (prompt adaptation on similar ops).
                    if _reject_reason:
                        try:
                            from backend.core.ouroboros.governance.self_evolution import (
                                NegativeConstraintStore,
                            )
                            from backend.core.ouroboros.governance.entropy_calculator import (
                                extract_domain_key as _rej_edk,
                            )
                            _rej_domain = _rej_edk(ctx.target_files, ctx.description)
                            _ns = NegativeConstraintStore()
                            _ns.add_constraint(
                                _rej_domain,
                                f"Human rejected: {_reject_reason[:120]}",
                                f"Op {ctx.op_id} on {_files_short} was rejected at Iron Gate",
                                source_op_id=ctx.op_id,
                                severity="hard",
                            )
                        except Exception:
                            pass  # Constraint recording is best-effort

                    # Task #195: Persist rejection to UserPreferenceStore as a
                    # typed FEEDBACK memory. NegativeConstraintStore is domain-
                    # keyed (cross-session prompt adaptation); the user-pref
                    # memory is human-readable and surfaces in the User
                    # Preferences prompt section on any similarly-shaped op.
                    if _reject_reason:
                        try:
                            from backend.core.ouroboros.governance.user_preference_memory import (
                                get_default_store,
                            )
                            get_default_store().record_approval_rejection(
                                op_id=ctx.op_id,
                                description=ctx.description,
                                target_files=list(ctx.target_files),
                                reason=_reject_reason,
                                provenance=getattr(
                                    decision, "reason_provenance", "unstated"),
                                approver=getattr(decision, "approver", "human") or "human",
                            )
                        except Exception:
                            pass  # Postmortem persistence is best-effort

                    return ctx

                # APPROVED -- continue to APPLY

            # ── PreActionNarrator: voice WHAT before APPLY ──
            if self._pre_action_narrator is not None:
                try:
                    _tf = list(ctx.target_files)[0] if ctx.target_files else "unknown"
                    await self._pre_action_narrator.narrate_phase("APPLY", {"target_file": _tf})
                except Exception:
                    pass

            # ── Cooperative cancellation check (pre-APPLY) ──
            if self._is_cancel_requested(ctx.op_id):
                ctx = ctx.advance(OperationPhase.CANCELLED, terminal_reason_code="user_cancelled")
                await self._record_ledger(ctx, OperationState.FAILED, {"reason": "user_cancelled"})
                return ctx

            # ── Session-scoped dry-run gate (Priority 4 /plan dry-run) ──
            #
            # Flipped by ``/plan dry-run`` in the REPL; ops pass every gate
            # up to here (CLASSIFY → PLAN → GENERATE → VALIDATE → SECURITY
            # → GATE → APPROVE → pre-APPLY checks), then short-circuit just
            # before disk writes begin. Operators get full observability
            # into "what the model wanted to do" without any filesystem /
            # git side effects. This is distinct from JARVIS_SHOW_PLAN_BEFORE_EXECUTE
            # (which gates at the PLAN→GENERATE boundary) — the dry-run
            # flag is the hard "no side effects this session" kill switch.
            if os.environ.get("JARVIS_DRY_RUN", "").strip().lower() in _TRUTHY:
                logger.info(
                    "[Orchestrator] DRY_RUN: op=%s would APPLY %d file(s) — "
                    "skipping disk writes (set JARVIS_DRY_RUN=0 or /plan off)",
                    ctx.op_id,
                    len(ctx.target_files) if ctx.target_files else 0,
                )
                ctx = ctx.advance(
                    OperationPhase.CANCELLED,
                    terminal_reason_code="dry_run_session",
                )
                await self._record_ledger(
                    ctx, OperationState.FAILED,
                    {"reason": "dry_run_session"},
                )
                return ctx

            # ── Predictive Phase-Aware Checkpoint (pre-APPLY) ──
            # Mirror of the extracted Slice4bRunner gate: project the irreversible
            # APPLY→VERIFY tail against the remaining wall runway (decoupled read).
            # If it won't fit, gracefully self-suspend (signed checkpoint + atomic
            # dirty-tree stash) so the op resumes next ignition instead of a
            # hard-kill mid-APPLY. Fail-open: only terminate if the checkpoint
            # persisted; otherwise fall through into APPLY.
            try:
                from backend.core.ouroboros.governance import phase_runway_gate as _prg
                _runway_verdict = _prg.evaluate(ctx, _prg.PRE_APPLY_TAIL_PHASES)
                if _runway_verdict.should_suspend:
                    _ckpt_path = _prg.predictive_suspend(ctx, "APPLY", _runway_verdict)
                    if _ckpt_path:
                        ctx = ctx.advance(
                            OperationPhase.CANCELLED,
                            terminal_reason_code="predictive_suspend",
                        )
                        await self._record_ledger(
                            ctx, OperationState.FAILED,
                            {"reason": "predictive_suspend",
                             "runway": _runway_verdict.as_telemetry()},
                        )
                        return ctx
            except Exception:  # noqa: BLE001 — fail open into APPLY
                logger.debug(
                    "[Orchestrator] predictive checkpoint gate skipped", exc_info=True,
                )

            # ---- Phase 7: APPLY ----
            ctx = ctx.advance(OperationPhase.APPLY)

            # ── Pre-APPLY git checkpoint (Manifesto §6: Iron Gate) ──
            _checkpoint = None
            _ckpt_mgr = None
            try:
                from backend.core.ouroboros.governance.workspace_checkpoint import WorkspaceCheckpointManager
                _ckpt_mgr = WorkspaceCheckpointManager(self._config.project_root)
                _checkpoint = await _ckpt_mgr.create_checkpoint(
                    ctx.op_id, f"pre-apply: {ctx.description[:80]}"
                )
            except Exception:
                logger.debug("[Orchestrator] Pre-APPLY checkpoint skipped", exc_info=True)

            # Heartbeat: APPLY phase starting (Manifesto §7)
            try:
                _apply_target = list(ctx.target_files)[0] if ctx.target_files else ""
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="APPLY", progress_pct=80.0,
                    target_file=_apply_target,
                )
            except Exception:
                pass

            # Deploy gate: canary preflight before applying changes
            try:
                from backend.core.ouroboros.governance.deploy_gate import DeployGate
                _canary = getattr(self._stack, "canary_controller", None)
                if _canary is not None:
                    _gate = DeployGate(canary=_canary)
                    _preflight = _gate.preflight(
                        service=ctx.primary_repo,
                        target_files=list(ctx.target_files),
                    )
                    if not _preflight.passed:
                        logger.warning(
                            "[Orchestrator] DeployGate preflight FAILED: %s [%s]",
                            _preflight.reason, ctx.op_id,
                        )
                        # Don't block — log warning. Gate is advisory until graduation gate passes.
            except Exception:
                logger.debug("[Orchestrator] DeployGate not available", exc_info=True)

            # ── Lifecycle Hook PRE_APPLY gate (Slice 4, 2026-05-02) ──
            # Operator-defined hooks fire here BEFORE any file write.
            # BLOCK aggregate routes the op to CANCELLED via the
            # established ctx.advance(CANCELLED, terminal_reason_code=...)
            # pattern (mirrors emergency-cancel at line 1820+).
            # WARN/CONTINUE proceed normally. Master-flag-gated by
            # JARVIS_LIFECYCLE_HOOKS_ENABLED (default false through
            # Slices 1-4; Slice 5 graduates). NEVER raises out of the
            # bridge — fail-open on any bridge-side error (a broken
            # hook substrate cannot block the autonomous loop).
            try:
                from backend.core.ouroboros.governance.lifecycle_hook_orchestrator_bridge import (
                    gate_pre_apply,
                )
                _lh_gate = await gate_pre_apply(
                    ctx.op_id,
                    target_files=tuple(ctx.target_files or ()),
                    diff_summary=str(ctx.description or "")[:1000],
                    risk_tier=str(getattr(ctx, "risk_tier", "") or ""),
                )
                if not _lh_gate.passed:
                    _block_names = ",".join(_lh_gate.blocking_hooks)[:64]
                    logger.warning(
                        "[Orchestrator] Lifecycle hook PRE_APPLY "
                        "BLOCKED op=%s blocking_hooks=%s",
                        ctx.op_id, list(_lh_gate.blocking_hooks),
                    )
                    if _serpent:
                        await _serpent.stop(success=False)
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code=(
                            f"lifecycle_hook_blocked:{_block_names}"
                        ),
                    )
                    return ctx
                if _lh_gate.should_warn:
                    logger.info(
                        "[Orchestrator] Lifecycle hook PRE_APPLY "
                        "WARNED op=%s warning_hooks=%s",
                        ctx.op_id, list(_lh_gate.warning_hooks),
                    )
            except Exception:
                logger.debug(
                    "[Orchestrator] Lifecycle hook bridge not "
                    "available",
                    exc_info=True,
                )

            # Cross-repo saga path
            if ctx.cross_repo:
                if "execution_graph" in best_candidate:
                    ctx, best_candidate = await self._materialize_execution_graph_candidate(
                        ctx,
                        best_candidate,
                    )
                return await self._execute_saga_apply(ctx, best_candidate)

            # ── Stale-exploration guard: check hashes before APPLY ──
            # If a target file was modified by a concurrent operation since GENERATE,
            # the candidate is stale.  Log a warning (soft gate) — the apply proceeds
            # but the ledger records the staleness for future convergence analysis.
            _stale_files: list = []
            if ctx.generate_file_hashes:
                # Slice 247/248 — verification pass. Re-hash the targets and
                # decide whether to BLOCK: a candidate whose GENERATE baseline no
                # longer matches disk is provably stale, and applying it corrupts
                # the file (full-content overwrite = data loss; diff = line drift).
                # Single source of truth with the GENERATE-entry re-alignment.
                from backend.core.ouroboros.governance.state_drift import (
                    should_block_apply as _should_block_apply,
                    STATE_DRIFT_UNRECONCILED as _STATE_DRIFT_UNRECONCILED,
                )
                _block_apply, _stale_files = _should_block_apply(
                    ctx.generate_file_hashes, self._config.project_root,
                )
                if _stale_files:
                    logger.warning(
                        "[Orchestrator] Stale-exploration: %d file(s) changed between GENERATE and APPLY: %s [%s]",
                        len(_stale_files), _stale_files[:3], ctx.op_id[:12],
                    )
                    await self._record_ledger(ctx, OperationState.APPLYING, {
                        "event": "stale_exploration_detected",
                        "stale_files": _stale_files,
                    })
                if _block_apply:
                    # Slice 248 — VERIFICATION FAILED. The GENERATE-entry
                    # re-alignment (Slice 247) did not resolve the drift (the
                    # model ignored the re-read, or the disk drifted again during
                    # regeneration). Refuse the apply — fail safe to POSTMORTEM
                    # rather than corrupt the file. The op re-runs fresh on its
                    # next sensor trigger, regenerating against current disk (an
                    # eventual re-alignment). Mirrors the LiveWorkSensor abort.
                    logger.warning(
                        "[Orchestrator] STATE DRIFT UNRECONCILED — blocking APPLY "
                        "of stale candidate on %s — failing safe (no corruption) "
                        "[%s]", _stale_files[:3], ctx.op_id[:12],
                    )
                    await self._record_ledger(ctx, OperationState.FAILED, {
                        "reason": _STATE_DRIFT_UNRECONCILED,
                        "stale_files": _stale_files,
                    })
                    ctx = ctx.advance(
                        OperationPhase.POSTMORTEM,
                        terminal_reason_code=_STATE_DRIFT_UNRECONCILED,
                    )
                    await self._publish_outcome(
                        ctx, OperationState.FAILED, _STATE_DRIFT_UNRECONCILED,
                    )
                    return ctx

            # ── LiveWorkSensor: don't stomp on human-active files ──
            # If the human is actively editing a target file, defer the autonomous
            # apply. Slice 10: "defer" now means a bounded sensor-derived wait
            # (_live_work_apply_gate) — the gate only returns a hit when the wait
            # is infeasible (IDE lock, exhausted budget) or the wait master is
            # off. Green/Yellow tiers abort with `human_active`; Orange tier
            # (APPROVAL_REQUIRED) proceeds because the human already approved.
            try:
                from backend.core.ouroboros.governance.live_work_sensor import (
                    is_enabled as _lws_enabled,
                )
                if _lws_enabled() and ctx.risk_tier is not RiskTier.APPROVAL_REQUIRED:
                    _lw_gate = await self._live_work_apply_gate(ctx, best_candidate)
                    if _lw_gate.active_hit is not None:
                        _hit_file, _hit_reason = _lw_gate.active_hit
                        await self._record_ledger(ctx, OperationState.FAILED, {
                            "reason": "human_active_on_target",
                            "file": _hit_file,
                            "signal": _hit_reason,
                        })
                        ctx = ctx.advance(
                            OperationPhase.POSTMORTEM,
                            terminal_reason_code="human_active_on_target",
                        )
                        await self._publish_outcome(ctx, OperationState.FAILED, "human_active_on_target")
                        return ctx
                    if _lw_gate.drift_stale_files is not None:
                        # Review C1 (TOCTOU) — the Slice 248 drift check
                        # above ran BEFORE the gate's wait; the gate re-ran
                        # the SAME helper post-wait and it came back
                        # blocking. SAME terminal shape as the pre-gate
                        # block (state_drift_unreconciled).
                        from backend.core.ouroboros.governance.state_drift import (
                            STATE_DRIFT_UNRECONCILED as _SD_UNRECONCILED,
                        )
                        logger.warning(
                            "[Orchestrator] STATE DRIFT UNRECONCILED post-LiveWork-wait "
                            "(%.0fs) — blocking APPLY of stale candidate on %s — "
                            "failing safe (no corruption) [%s]",
                            _lw_gate.waited_s,
                            _lw_gate.drift_stale_files[:3], ctx.op_id[:12],
                        )
                        await self._record_ledger(ctx, OperationState.FAILED, {
                            "reason": _SD_UNRECONCILED,
                            "stale_files": _lw_gate.drift_stale_files,
                        })
                        ctx = ctx.advance(
                            OperationPhase.POSTMORTEM,
                            terminal_reason_code=_SD_UNRECONCILED,
                        )
                        await self._publish_outcome(
                            ctx, OperationState.FAILED, _SD_UNRECONCILED,
                        )
                        return ctx
            except Exception:
                logger.debug("[Orchestrator] LiveWorkSensor check skipped", exc_info=True)

            # Capture pre-apply snapshots for complexity baseline + multi-file rollback.
            # Include ctx.target_files AND every file the candidate proposes — for a
            # multi-file candidate the secondary files may not be in ctx.target_files
            # and we need their pre-state to restore them if any file in the batch
            # fails its apply.
            snapshots: Dict[str, str] = {}
            _snapshot_targets: set[str] = {str(f) for f in ctx.target_files}
            for _cf, _ in self._iter_candidate_files(best_candidate):
                if _cf:
                    _snapshot_targets.add(_cf)
            # Slice 11 review C3: capture from the EXECUTION root — the tree
            # APPLY writes and verify-gate rollback restores. Capturing the
            # observation tree while restoring into the workspace corrupted
            # the workspace baseline whenever the two trees diverged.
            _snap_root = Path(self._config.execution_root)
            for f in _snapshot_targets:
                fpath = Path(f) if Path(f).is_absolute() else _snap_root / f
                if fpath.exists():
                    try:
                        snapshots[str(f)] = fpath.read_text(errors="replace")
                    except OSError:
                        pass
            if snapshots:
                ctx = ctx.with_pre_apply_snapshots(snapshots)

            # Multi-file candidates go through a batch apply helper with
            # all-or-nothing rollback semantics. Single-file candidates still
            # use the legacy single ChangeRequest path (zero change for them).
            _candidate_files = self._iter_candidate_files(best_candidate)

            # Session O (bt-2026-04-15-175547) APPLY-path observability:
            # log the multi-file decision at a single INFO line so logs
            # prove single- vs multi-file flow without reading the raw
            # candidate JSON. Session O's 4-file backlog probe wrote only
            # dedup.py because the winning candidate returned a single
            # (file_path, full_content) pair instead of a ``files`` list —
            # the multi-file coordinated path (_apply_multi_file_candidate)
            # is gated behind len(_candidate_files) > 1, which requires the
            # candidate to include a populated ``files: [...]`` array.
            # Without this log line, it took cross-referencing disk state
            # against diff_summary text to confirm the single-file path
            # was taken. This line makes that one grep.
            _files_field = best_candidate.get("files") if isinstance(
                best_candidate, dict
            ) else None
            _has_files_key = isinstance(_files_field, list) and len(_files_field) > 0
            _multi_enabled = (
                os.environ.get("JARVIS_MULTI_FILE_GEN_ENABLED", "true").lower()
                not in ("false", "0", "no", "off")
            )
            _apply_mode = "multi" if len(_candidate_files) > 1 else "single"
            _file_basenames = [
                (fp.rsplit("/", 1)[-1] if "/" in fp else fp)
                for fp, _ in _candidate_files
            ]
            logger.info(
                "[Orchestrator] APPLY mode=%s candidate_files=%d "
                "files_list_present=%s multi_enabled=%s targets=[%s] op=%s",
                _apply_mode,
                len(_candidate_files),
                _has_files_key,
                _multi_enabled,
                ",".join(_file_basenames),
                ctx.op_id[:16],
            )

            if len(_candidate_files) > 1:
                _t_apply = time.monotonic()
                try:
                    # LR-B (spec 5.4): mark the on-disk multi-file apply as a
                    # critical mutation so the operator-yield drains before it
                    # can park the op mid-batch (no-op when the yield is off).
                    async with maybe_mutation_section(ctx.op_id):
                        # Anti-Venom C2 — shield the apply. If the op task is
                        # cancelled mid-write the inner coroutine still runs to
                        # completion (file-write + per-file APPLIED ledger commit
                        # are atomic), closing the mutated-on-disk-but-ledger-
                        # stuck-APPLYING split-brain. CancelledError still
                        # propagates here at the shield boundary, so the stop is
                        # honored once the write has finished.
                        change_result = await asyncio.shield(
                            self._apply_multi_file_candidate(
                                ctx, best_candidate, _candidate_files, snapshots,
                            )
                        )
                except Exception as exc:
                    logger.error(
                        "Multi-file change engine raised for %s: %s", ctx.op_id, exc
                    )
                    ctx = ctx.advance(
                        OperationPhase.POSTMORTEM,
                        terminal_reason_code="change_engine_error",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "change_engine_error", "error": str(exc), "multi_file": True},
                    )
                    self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_apply)
                    await self._publish_outcome(ctx, OperationState.FAILED, "change_engine_error")
                    return ctx
                # Single-file fall-through path (change_result is already set).
                change_request = None  # type: ignore[assignment]
            else:
                change_request = self._build_change_request(ctx, best_candidate)
                _t_apply = time.monotonic()
                # Priority F2 (Slice 1 drift repair, 2026-07-18): snapshot
                # target_files content BEFORE the change_engine writes — the
                # pre-state half of the diff_text evidence. The drift audit
                # found this stamped ONLY in the extracted slice4b_runner, so
                # this inline kill-switch path silently degraded
                # no_new_credential_shapes to INSUFFICIENT. Mirrors
                # slice4b_runner verbatim; best-effort, never raises.
                try:
                    from backend.core.ouroboros.governance.verification.evidence_capture import (  # noqa: E501
                        stamp_target_files_pre_async,
                    )
                    await stamp_target_files_pre_async(ctx)
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[Orchestrator] stamp_target_files_pre failed",
                        exc_info=True,
                    )
                try:
                    # LR-B (spec 5.4): mark the on-disk single-file apply as a
                    # critical mutation so the operator-yield drains before it
                    # can park the op mid-write (no-op when the yield is off).
                    async with maybe_mutation_section(ctx.op_id):
                        # Anti-Venom C2 — shield the apply (see multi-file path
                        # above): cancellation mid-write cannot leave the file
                        # written but the APPLIED ledger commit skipped.
                        # CancelledError still propagates at the shield boundary.
                        change_result = await asyncio.shield(
                            self._stack.change_engine.execute(change_request)
                        )
                except Exception as exc:
                    logger.error(
                        "Change engine raised for %s: %s", ctx.op_id, exc
                    )
                    ctx = ctx.advance(
                        OperationPhase.POSTMORTEM,
                        terminal_reason_code="change_engine_error",
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "change_engine_error", "error": str(exc)},
                    )
                    self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_apply)
                    await self._publish_outcome(ctx, OperationState.FAILED, "change_engine_error")
                    return ctx

            if not change_result.success:
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code="change_engine_failed",
                    rollback_occurred=change_result.rolled_back,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {
                        "reason": "change_engine_failed",
                        "rolled_back": change_result.rolled_back,
                    },
                )
                self._record_canary_for_ctx(
                    ctx, False, time.monotonic() - _t_apply,
                    rolled_back=change_result.rolled_back,
                )
                await self._publish_outcome(ctx, OperationState.FAILED, "change_engine_failed")
                return ctx

            # Priority F2 (Slice 1 drift repair): APPLY succeeded — capture full
            # post-state evidence (target_files_post / test_files_post /
            # diff_text) so the F1 gatherers find rich pre-stamped data. The
            # drift audit found this ONLY in the extracted slice4b_runner.
            # Mirrors it verbatim; best-effort, never raises.
            try:
                from backend.core.ouroboros.governance.verification.evidence_capture import (  # noqa: E501
                    stamp_apply_evidence_post_async,
                )
                await stamp_apply_evidence_post_async(ctx)
            except Exception:  # noqa: BLE001 — defensive
                logger.debug(
                    "[Orchestrator] stamp_apply_evidence_post failed",
                    exc_info=True,
                )

            # ---- Phase 7.5: INFRASTRUCTURE (deterministic post-APPLY hook) ----
            # Boundary Principle: the agentic layer wrote the file (e.g., requirements.txt).
            # This hook executes the KNOWN consequence (pip install). No inference.
            if self._infra_applicator is not None and self._infra_applicator.is_enabled:
                infra_results = await self._infra_applicator.execute_post_apply(
                    modified_files=ctx.target_files,
                    op_id=ctx.op_id,
                )
                if infra_results and not self._infra_applicator.all_succeeded(infra_results):
                    _failed = [r for r in infra_results if not r.success]
                    from backend.core.ouroboros.governance.infrastructure_applicator import (
                        infra_fail_soft_enabled,
                        summarize_infra_failures,
                    )
                    if not infra_fail_soft_enabled():
                        # Legacy terminal FAILED (operator opt-out via fail-soft=0): the
                        # file change is correct but the environment didn't accept it.
                        logger.error(
                            "[Orchestrator] Infrastructure hook failed for %s: %s",
                            ctx.op_id,
                            "; ".join(f"{r.file_trigger}: exit={r.exit_code}" for r in _failed),
                        )
                        ctx = ctx.advance(
                            OperationPhase.POSTMORTEM,
                            terminal_reason_code="infrastructure_failed",
                        )
                        await self._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": "infrastructure_failed",
                                "infra_results": [
                                    {
                                        "file": r.file_trigger,
                                        "command": r.command,
                                        "exit_code": r.exit_code,
                                        "stderr": r.stderr_tail[:500],
                                    }
                                    for r in _failed
                                ],
                            },
                        )
                        self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_apply)
                        await self._publish_outcome(ctx, OperationState.FAILED, "infrastructure_failed")
                        return ctx
                    # Slice 160 fail-soft: infra failure is NOT fatal — flag
                    # INFRA_WARNING + continue to VERIFY/COMPLETE (op survives; the
                    # operator sees the warning via telemetry / Discord).
                    _warn = summarize_infra_failures(_failed)
                    logger.warning(
                        "[Orchestrator] INFRA_WARNING (fail-soft) op=%s — continuing: %s",
                        ctx.op_id, _warn,
                    )
                    try:
                        ctx = ctx.with_infra_warning(_warn)
                    except Exception:  # noqa: BLE001
                        pass

                # Log successful infra operations for observability
                for r in infra_results:
                    logger.info(
                        "[Orchestrator] Infrastructure: %s completed in %.1fs (op=%s)",
                        r.file_trigger, r.duration_s, ctx.op_id,
                    )

            if _serpent: _serpent.update_phase("APPLY")

            # OpsDigestObserver v1.1a — APPLY milestone (best-effort telemetry).
            # Reaching this point means ChangeEngine succeeded (failed paths
            # returned early). Derive mode from target-files count so we
            # don't rely on outer-scope local variables remaining in scope.
            try:
                from backend.core.ouroboros.governance.ops_digest_observer import (
                    APPLY_MODE_MULTI,
                    APPLY_MODE_SINGLE,
                    get_ops_digest_observer,
                )
                _apply_file_count = len(ctx.target_files or ())
                _apply_mode_tag = (
                    APPLY_MODE_MULTI if _apply_file_count > 1 else APPLY_MODE_SINGLE
                )
                get_ops_digest_observer().on_apply_succeeded(
                    op_id=ctx.op_id,
                    mode=_apply_mode_tag,
                    files=_apply_file_count,
                )
            except Exception:
                logger.debug(
                    "[Orchestrator] on_apply_succeeded observer call failed",
                    exc_info=True,
                )

            # ---- Phase 8: VERIFY ----
            if _serpent: _serpent.update_phase("VERIFY")
            ctx = ctx.advance(OperationPhase.VERIFY)

            # Heartbeat: VERIFY phase starting (Manifesto §7)
            try:
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id, phase="verify", progress_pct=92.0,
                )
            except Exception:
                pass

            await self._record_ledger(
                ctx,
                OperationState.APPLIED,
                {"op_id": ctx.op_id},
            )

            # ---- Phase 8a: Scoped post-apply test run ----
            # Run tests scoped to the files that were just modified.  This catches
            # regressions *before* the broader benchmark gate and can route failures
            # into L2 repair instead of immediate rollback.
            _verify_test_passed = True
            _verify_test_total = 0
            _verify_test_failures = 0
            _verify_timed_out = False
            _verify_failed_names: Tuple[str, ...] = ()
            # Slice 11: resolve the judgment tree ONCE per VERIFY pass —
            # the tree APPLY wrote (workspace when Ledger Sovereignty is
            # armed, else the observation root). Every VERIFY-side consumer
            # below (scoped tests, containment probe, rollback) anchors
            # this local; none may re-read the observation root.
            _exec_root = Path(self._config.execution_root)

            if self._validation_runner is not None and ctx.target_files:
                _changed = tuple(
                    _exec_root / f for f in ctx.target_files
                )
                _files_str = ", ".join(str(f) for f in list(ctx.target_files)[:3])

                # Heartbeat: scoped verify starting (drives ⏺ Verify block in CLI)
                try:
                    await self._stack.comm.emit_heartbeat(
                        op_id=ctx.op_id, phase="verify",
                        verify_test_starting=True,
                        verify_target_files=list(ctx.target_files),
                    )
                except Exception:
                    pass

                _verify_budget_s = min(
                    60.0,
                    float(os.environ.get("JARVIS_VERIFY_TIMEOUT_S", "60")),
                )
                _v_runner, _v_sandbox = self._scoped_verify_runner(_exec_root)
                try:
                    _multi = await asyncio.wait_for(
                        _v_runner.run(
                            changed_files=_changed,
                            sandbox_dir=_v_sandbox,
                            timeout_budget_s=_verify_budget_s,
                            op_id=ctx.op_id,
                        ),
                        timeout=_verify_budget_s + 5.0,
                    )
                    _verify_test_passed = _multi.passed
                    for _ar in _multi.adapter_results:
                        _verify_test_total += _ar.test_result.total
                        _verify_test_failures += _ar.test_result.failed
                        _verify_failed_names += _ar.test_result.failed_tests
                    # 0/0 → N/A, not failure. When no test adapter has any tests
                    # for the changed files (deps-only changes, docs, configs),
                    # treat verify as a no-op rather than routing to L2 repair.
                    # Manifesto §6: only real signals trigger neuroplasticity.
                    if _verify_test_total == 0 and _verify_test_failures == 0:
                        _verify_test_passed = True
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("[Orchestrator] Verify scoped test timed out [%s]", ctx.op_id)
                    _verify_test_passed = False
                    _verify_test_failures = 1
                    _verify_timed_out = True
                except BlockedPathError:
                    pass  # security gate — skip scoped verify, let benchmark handle
                except Exception as exc:
                    logger.debug("[Orchestrator] Verify scoped test error: %s", exc)

                # Heartbeat: scoped verify result (drives ⏺ Verify result in CLI)
                try:
                    await self._stack.comm.emit_heartbeat(
                        op_id=ctx.op_id, phase="verify",
                        verify_test_passed=_verify_test_passed,
                        verify_test_total=_verify_test_total,
                        verify_test_failures=_verify_test_failures,
                        verify_target_files=list(ctx.target_files),
                    )
                except Exception:
                    pass

                # ---- Slice 106 — Runtime containment gate (gated, additive, fail-safe) ----
                # When the OS-level Docker sandbox is enabled AND a containment
                # probe is configured, run the candidate's runtime validation inside
                # the locked-down Linux container (network none / read-only rootfs /
                # cap-drop ALL / pids-limit). A ContainmentBreach (the candidate tried
                # to hang, crash, exfiltrate, or escape the FS at runtime) is a hard
                # VERIFY failure: mark verify failed (reusing the EXISTING failure →
                # L2/GENERATE_RETRY routing below) + record a FALSIFYING quarantine
                # belief so the learning loop steers away. The sandbox breach NEVER
                # crashes the FSM. Gated default-FALSE (JARVIS_RUNTIME_SANDBOX_ENABLED)
                # → byte-identical legacy. The probe + project image are operator-
                # configured for the production (Linux/Docker) deployment.
                try:
                    from backend.core.ouroboros.governance.container_sandbox import (  # noqa: E501
                        containerized_sandbox_enabled,
                        record_containment_breach_belief,
                        run_payload_contained,
                    )
                    from backend.core.ouroboros.governance.runtime_sandbox import (  # noqa: E501
                        ContainmentBreach as _CB106,
                    )
                    _probe106 = os.environ.get(
                        "JARVIS_RUNTIME_SANDBOX_VERIFY_PROBE", "",
                    ).strip()
                    if containerized_sandbox_enabled() and _probe106:
                        _cres106 = await run_payload_contained(
                            _probe106, worktree=str(_exec_root),
                        )
                        if (
                            _cres106 is not None and not _cres106.ok
                            and _cres106.breach in (
                                _CB106.TIMEOUT, _CB106.SIGNAL_KILLED, _CB106.NONZERO_EXIT,
                            )
                        ):
                            logger.warning(
                                "[Orchestrator] CONTAINMENT BREACH at VERIFY op=%s "
                                "breach=%s — quarantining candidate (→ GENERATE_RETRY)",
                                ctx.op_id, _cres106.breach.value,
                            )
                            _verify_test_passed = False
                            _verify_test_failures += 1
                            record_containment_breach_belief(
                                ctx.op_id, _cres106, ctx.target_files,
                            )
                except Exception:  # noqa: BLE001 — containment gate must NEVER crash VERIFY
                    logger.debug(
                        "[Orchestrator] containment gate skipped (non-fatal)",
                        exc_info=True,
                    )

                # OpsDigestObserver v1.1a — VERIFY milestone. Plan tightening
                # #1: ``scoped_to_applied_op=True`` because this branch only
                # runs when ``ctx.target_files`` was applied (it's the scoped
                # post-apply test run, not a repo-wide health check).
                try:
                    from backend.core.ouroboros.governance.ops_digest_observer import (
                        get_ops_digest_observer,
                    )
                    _verify_passed_count = max(
                        0, _verify_test_total - _verify_test_failures,
                    )
                    get_ops_digest_observer().on_verify_completed(
                        op_id=ctx.op_id,
                        passed=_verify_passed_count,
                        total=_verify_test_total,
                        scoped_to_applied_op=True,
                    )
                except Exception:
                    logger.debug(
                        "[Orchestrator] on_verify_completed observer call failed",
                        exc_info=True,
                    )

                # ClusterIntelligence-CrossSession Slice 4 -- post-
                # verify cascade: persist cluster_coverage explorations
                # into DomainMap so the next session sees prior context.
                # NEVER raises into the orchestrator. Master flag default-
                # off until Slice 5 graduation; observer short-circuits
                # cleanly when the op wasn't a cluster_coverage envelope.
                try:
                    from backend.core.ouroboros.governance.cluster_exploration_cascade_observer import (  # noqa: E501
                        observe_cluster_coverage_completion as _cascade_observe,
                    )
                    await _cascade_observe(
                        op_id=ctx.op_id,
                        intake_evidence_json=getattr(
                            ctx, "intake_evidence_json", "",
                        ) or "",
                        touched_files=tuple(
                            getattr(ctx, "target_files", ()) or (),
                        ),
                        verify_passed=bool(_verify_test_passed),
                        project_root=self._config.project_root,
                    )
                except Exception:
                    logger.debug(
                        "[Orchestrator] cluster cascade observer call failed",
                        exc_info=True,
                    )

                # ── M9 Slice 5b — feed prophecy-error producer ──
                # Read ProphecyEngine's cached per-file risk scores
                # (set at CLASSIFY via consciousness_bridge.assess_-
                # regression_risk) and feed (predicted_risk,
                # verify_passed) tuples to M9 via the producer
                # bridge. Bridge computes ``error_magnitude =
                # abs(predicted_risk - actual_outcome_indicator)`` —
                # high when Prophecy was wrong → high curiosity for
                # that file's cluster. Lazy-imported, master-flag-
                # gated, exception-isolated. NEVER raises out.
                try:
                    from backend.core.ouroboros.governance.curiosity_producer_bridge import (  # noqa: E501
                        feed_prophecy_error as _m9_feed_prophecy,
                    )
                    _m9_prophecy_engine = None
                    _m9_gls = getattr(
                        self._stack, "governed_loop_service", None,
                    )
                    if _m9_gls is not None:
                        _m9_cb = getattr(
                            _m9_gls, "_consciousness_bridge", None,
                        )
                        if _m9_cb is not None:
                            _m9_prophecy_engine = getattr(
                                _m9_cb, "_prophecy_engine", None,
                            ) or getattr(
                                _m9_cb, "prophecy_engine", None,
                            )
                    if _m9_prophecy_engine is not None:
                        _m9_risk_scores = (
                            _m9_prophecy_engine.get_risk_scores()
                        )
                        for _m9_path, _m9_risk in (
                            _m9_risk_scores or {}
                        ).items():
                            _m9_feed_prophecy(
                                region_or_path=str(_m9_path),
                                predicted_risk=float(_m9_risk),
                                verify_passed=bool(
                                    _verify_test_passed,
                                ),
                                op_id=str(ctx.op_id),
                            )
                except Exception:
                    logger.debug(
                        "[Orchestrator] M9 prophecy-error feed failed",
                        exc_info=True,
                    )

                # Tier 2 #6 follow-up Arc 1 (2026-05-03) — VERIFY hook
                # for auto_action_router + Production Oracle. Wires the
                # advisory framework that's been built but unused at
                # the orchestrator level. Reads the most-recent oracle
                # observation + recent postmortem outcomes + confidence
                # verdicts; proposes an AdvisoryAction; logs + emits
                # SSE event. ADVISORY ONLY -- never blocks COMPLETE,
                # never mutates Iron Gate / risk / route. Master flag
                # JARVIS_AUTO_ACTION_VERIFY_HOOK_ENABLED graduated
                # default-true; operators flip explicit "false" to
                # silence the loop.
                try:
                    if os.environ.get(
                        "JARVIS_AUTO_ACTION_VERIFY_HOOK_ENABLED", "",
                    ).strip().lower() not in ("0", "false", "no", "off"):
                        from backend.core.ouroboros.governance.auto_action_router import (  # noqa: E501
                            gather_context as _aa_gather,
                            propose_advisory_action as _aa_propose,
                            AdvisoryActionType,
                        )
                        # current_op_family/risk/route fields ride on
                        # ctx; defensive getattr in case ctx schema
                        # ever drifts.
                        # Event-loop unblocking (2026-07-22, soak
                        # bt-2026-07-22-022146): gather_context reads
                        # the postmortem ledger line-by-line from disk
                        # (list_recent_postmortems) — a 5.0s cold-FS
                        # STUCK_FRAME when run on the loop. Dispatch
                        # the whole sync gather to the dedicated
                        # advisor-blast executor (Task #88f isolation
                        # pool — DRY, no new executor).
                        from backend.core.ouroboros.governance.operation_advisor import (  # noqa: E501
                            _get_advisor_blast_executor as _aa_pool,
                        )
                        _aa_of = str(getattr(ctx, "op_family", "") or "")
                        _aa_rt = str(getattr(ctx, "risk_tier", "") or "")
                        _aa_route = str(getattr(
                            ctx, "provider_route", "",
                        ) or "")

                        def _aa_gather_offloaded(
                            _of=_aa_of, _rt=_aa_rt, _route=_aa_route,
                        ):
                            return _aa_gather(
                                current_op_family=_of,
                                current_risk_tier=_rt,
                                current_route=_route,
                                posture="",
                                include_oracle=True,
                            )

                        _aa_ctx = await asyncio.get_running_loop(
                        ).run_in_executor(
                            _aa_pool(), _aa_gather_offloaded,
                        )
                        _aa_action = _aa_propose(_aa_ctx)
                        if (
                            _aa_action.action_type
                            is not AdvisoryActionType.NO_ACTION
                        ):
                            logger.info(
                                "[Orchestrator] auto_action proposal "
                                "op=%s action=%s reason=%s",
                                ctx.op_id[:16],
                                _aa_action.action_type.value,
                                _aa_action.reason_code,
                            )
                            try:
                                from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501
                                    publish_auto_action_proposal,
                                )
                                publish_auto_action_proposal(
                                    op_id=ctx.op_id,
                                    action_type=(
                                        _aa_action.action_type.value
                                    ),
                                    reason_code=_aa_action.reason_code,
                                    target_op_family=(
                                        _aa_action.target_op_family
                                    ),
                                    proposed_risk_tier=(
                                        _aa_action.proposed_risk_tier
                                    ),
                                    evidence=_aa_action.evidence,
                                )
                            except Exception:
                                pass
                except Exception:
                    logger.debug(
                        "[Orchestrator] auto_action_router VERIFY "
                        "hook failed", exc_info=True,
                    )

                # On failure: attempt L2 repair before rollback
                if not _verify_test_passed and self._config.repair_engine is not None:
                    logger.info(
                        "[Orchestrator] VERIFY test failed (%d/%d) — routing to L2 repair [%s]",
                        _verify_test_failures, _verify_test_total, ctx.op_id,
                    )
                    _pl_deadline = ctx.pipeline_deadline or (
                        datetime.now(timezone.utc) + timedelta(seconds=60)
                    )
                    # Build a synthetic ValidationResult for L2
                    _synth_val = ValidationResult(
                        passed=False,
                        best_candidate=best_candidate,
                        validation_duration_s=0.0,
                        error=f"post-apply verify: {_verify_test_failures}/{_verify_test_total} failing",
                        failure_class="test",
                        short_summary=f"verify: {', '.join(_verify_failed_names[:3])}",
                        adapter_names_run=(),
                    )
                    try:
                        directive = await self._l2_hook(ctx, _synth_val, _pl_deadline)
                        if directive[0] == "break":
                            # L2 converged — apply the repair candidate to real files,
                            # then mark verify as passed.  Without this step, the L2
                            # candidate is validated in sandbox but never written to disk.
                            _l2_candidate = directive[1]
                            _l2_change = self._build_change_request(ctx, _l2_candidate)
                            try:
                                _l2_result = await self._stack.change_engine.execute(_l2_change)
                                if _l2_result.success:
                                    _verify_test_passed = True
                                    _verify_test_failures = 0
                                    logger.info(
                                        "[Orchestrator] L2 repair applied in VERIFY phase [%s]",
                                        ctx.op_id,
                                    )
                                else:
                                    logger.warning(
                                        "[Orchestrator] L2 repair candidate failed to apply [%s]",
                                        ctx.op_id,
                                    )
                            except Exception as _apply_exc:
                                logger.debug("[Orchestrator] L2 repair apply error: %s", _apply_exc)
                        elif directive[0] == "l2_pivot":
                            # T3 — Graceful Semantic Pivot in the VERIFY phase.
                            # Route through the shared pivot handler (decompose-
                            # further at the failure locus or HITL DLQ) and
                            # return its terminal ctx. DAG-preserving.
                            _pivot_sig = directive[2] if len(directive) > 2 else ""
                            _pivot_tail = directive[3] if len(directive) > 3 else ""
                            ctx = await self._handle_l2_pivot(
                                directive[1], _pivot_sig, _pivot_tail,
                            )
                            return ctx
                        elif directive[0] in ("cancel", "fatal"):
                            # L2 decided to escape. _l2_hook has already advanced
                            # ctx to the phase-appropriate terminal (POSTMORTEM
                            # from VERIFY per _l2_escape_terminal) and recorded a
                            # ledger entry. Capture the terminal ctx and return
                            # immediately — continuing VERIFY logic (benchmark,
                            # verify gate, rollback) on a terminal ctx would
                            # violate the FSM and produce spurious transitions.
                            ctx = directive[1]
                            logger.info(
                                "[Orchestrator] L2 escaped VERIFY phase — "
                                "op ctx advanced to %s [%s]",
                                ctx.phase.name, ctx.op_id,
                            )
                            return ctx
                    except Exception as _l2_exc:
                        # Log the failure as a one-liner instead of a full traceback;
                        # the exception path is handled inside _l2_hook which already
                        # advances ctx to POSTMORTEM.
                        logger.debug(
                            "[Orchestrator] L2 repair in VERIFY failed: %s: %s",
                            type(_l2_exc).__name__, _l2_exc,
                        )

            ctx = await self._run_benchmark(ctx, [])

            # ---- Verify Gate: enforce regression thresholds (Sub-project C) ----
            _verify_error = None
            try:
                from backend.core.ouroboros.governance.verify_gate import (
                    enforce_verify_thresholds,
                    rollback_files,
                )
                _br = getattr(ctx, "benchmark_result", None)
                if _br is not None:
                    _baseline_cov = None
                    _snapshots = getattr(ctx, "pre_apply_snapshots", {})
                    if isinstance(_snapshots, dict):
                        _baseline_cov = _snapshots.get("_coverage_baseline")
                    _verify_error = enforce_verify_thresholds(_br, baseline_coverage=_baseline_cov)
            except Exception as exc:
                logger.debug("[Orchestrator] Verify gate skipped: %s", exc)

            # Combine scoped-test failure with benchmark regression
            if _verify_error is None and not _verify_test_passed:
                if _verify_timed_out:
                    # Slice 11 rider (§7 honesty): the timeout sentinel
                    # counters rendered as "1/0 tests failing" and read as
                    # a denominator bug across Runs #20/#21.
                    _verify_error = (
                        "scoped verify timed out after "
                        f"{_verify_budget_s:.0f}s"
                    )
                else:
                    _verify_error = f"scoped verify: {_verify_test_failures}/{_verify_test_total} tests failing"

            # Slice 67 — swe_bench_pro VERIFY regression gate is advisory (the
            # repo tests can't run locally; the held-out container scoring is
            # authoritative). Clearing the error keeps the patch applied (no
            # rollback) so the autoscore layer captures + scores it.
            _verify_error = _swe_bench_verify_advisory(
                getattr(ctx, "signal_source", "") or "", _verify_error, ctx.op_id,
            )

            if _verify_error is not None:
                logger.warning(
                    "[Orchestrator] VERIFY regression gate fired: %s [%s]",
                    _verify_error, ctx.op_id,
                )
                # Emit gate event for VoiceNarrator
                try:
                    await self._stack.comm.emit_postmortem(
                        op_id=ctx.op_id,
                        root_cause=f"verify_regression: {_verify_error}",
                        failed_phase="VERIFY",
                        target_files=list(ctx.target_files),
                    )
                except Exception:
                    pass
                # Rollback files
                try:
                    _snapshots = getattr(ctx, "pre_apply_snapshots", {})
                    if _snapshots:
                        rollback_files(
                            pre_apply_snapshots=_snapshots,
                            target_files=list(ctx.target_files),
                            repo_root=_exec_root,
                        )
                except Exception as exc:
                    logger.error("[Orchestrator] Verify rollback failed: %s", exc)

                # Git checkpoint restore as safety net (Manifesto §6: Iron Gate)
                if _checkpoint is not None and _ckpt_mgr is not None:
                    try:
                        await _ckpt_mgr.restore_checkpoint(_checkpoint.checkpoint_id)
                        logger.info(
                            "[Orchestrator] Git checkpoint restored: %s [%s]",
                            _checkpoint.checkpoint_id, ctx.op_id,
                        )
                    except Exception:
                        logger.debug("[Orchestrator] Checkpoint restore failed", exc_info=True)

                if _serpent: _serpent.update_phase("POSTMORTEM")
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code="verify_regression",
                    rollback_occurred=True,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {"reason": "verify_regression", "detail": _verify_error, "rollback_occurred": True},
                )
                self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_apply, rolled_back=True)
                await self._publish_outcome(ctx, OperationState.FAILED, "verify_regression")
                return ctx

            # ---- Phase 8b: Auto-commit (Gap #6 — autonomy loop closer) ----
            # After successful APPLY+VERIFY, commit with structured O+V signature.
            # Commit failures are non-fatal — the change is already applied on disk.
            _committed_hash: Optional[str] = None  # captured for Phase 3a critique below
            _commit_skip_reason: Optional[str] = None
            try:
                from backend.core.ouroboros.governance.auto_committer import AutoCommitter
                _committer = AutoCommitter(repo_root=self._config.project_root)
                _gen = ctx.generation
                _provider = getattr(_gen, "provider_name", "") if _gen else ""
                _cost = 0.0
                if _gen:
                    _in_tok = getattr(_gen, "total_input_tokens", 0) or 0
                    _out_tok = getattr(_gen, "total_output_tokens", 0) or 0
                    _cost = (_in_tok * 0.0000001 + _out_tok * 0.0000004)  # rough estimate
                # LR-B (spec 5.4): the git commit is a critical mutation —
                # an in-progress commit must not be parked by the
                # operator-yield (no-op when the yield is off).
                async with maybe_mutation_section(ctx.op_id):
                    _commit_result = await asyncio.wait_for(
                        _committer.commit(
                            op_id=ctx.op_id,
                            description=ctx.description,
                            target_files=ctx.target_files,
                            risk_tier=ctx.risk_tier,
                            provider_name=_provider,
                            generation_cost=_cost,
                            # Mythos §7.4: originating signal + rationale for
                            # zero-context reviewers.
                            signal_source=getattr(ctx, "signal_source", ""),
                            signal_urgency=getattr(ctx, "signal_urgency", ""),
                            rationale=ctx.description,
                        ),
                        timeout=30.0,
                    )
                if _commit_result.committed:
                    _committed_hash = _commit_result.commit_hash
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id, phase="commit",
                            progress_pct=98.0,
                            commit_hash=_commit_result.commit_hash,
                            commit_pushed=_commit_result.pushed,
                            commit_branch=_commit_result.push_branch,
                        )
                    except Exception:
                        pass
                    logger.info(
                        "[Orchestrator] Auto-committed %s for op=%s",
                        _commit_result.commit_hash, ctx.op_id,
                    )

                    # OpsDigestObserver v1.1a — commit milestone. Hash shape
                    # validation happens in the observer implementer; this
                    # call site just forwards AutoCommitter's reported value.
                    try:
                        from backend.core.ouroboros.governance.ops_digest_observer import (
                            get_ops_digest_observer,
                        )
                        get_ops_digest_observer().on_commit_succeeded(
                            op_id=ctx.op_id,
                            commit_hash=_commit_result.commit_hash or "",
                        )
                    except Exception:
                        logger.debug(
                            "[Orchestrator] on_commit_succeeded observer call failed",
                            exc_info=True,
                        )
                elif _commit_result.skipped_reason:
                    _commit_skip_reason = _commit_result.skipped_reason
                    # A skip that is actually a git-STATE fault (the committer
                    # reports a lock / conflict / rejection as a skip rather than
                    # raising) gets the same Phase 2 recovery as an exception; a
                    # DELIBERATE skip (protected branch, no changes, gitignore
                    # breach) classifies as "other" and is left exactly as before.
                    try:
                        from backend.core.ouroboros.governance.commit_fault_recovery import (  # noqa: E501
                            classify_commit_fault,
                            recover_from_commit_fault,
                        )
                        _skip_fault = classify_commit_fault(
                            Exception(str(_commit_result.skipped_reason)),
                        )
                        if _skip_fault != "other":
                            _recovery = await recover_from_commit_fault(
                                self, ctx,
                                Exception(str(_commit_result.skipped_reason)),
                            )
                            _commit_skip_reason = f"commit_fault:{_skip_fault}"
                            await self._record_ledger(
                                ctx, OperationState.APPLYING,
                                {"event": "commit_fault_recovered", **_recovery},
                            )
                        else:
                            logger.debug(
                                "[Orchestrator] Auto-commit skipped: %s",
                                _commit_result.skipped_reason,
                            )
                    except Exception:  # noqa: BLE001 — recovery never breaks APPLY
                        logger.debug(
                            "[Orchestrator] Auto-commit skipped: %s",
                            _commit_result.skipped_reason,
                        )
            except ImportError:
                logger.debug("[Orchestrator] AutoCommitter not available")
            except Exception as exc:
                # ---- Phase 2: commit-stage fault recovery ----
                # A locked index, merge conflict, or diff rejection (or a commit
                # timeout) leaves a VERIFIED change contesting the working tree.
                # Rather than log-and-leave it dangling: stash the change (scoped
                # to this op's files, recoverable — never a destructive reset),
                # emit a non-blocking diff_rejection event, and route the precise
                # fault to the PLAN subagent for a SURGICAL re-plan — all without
                # stalling the daemon. Fail-soft: recovery NEVER escalates a
                # non-fatal commit miss into a crashed op.
                try:
                    from backend.core.ouroboros.governance.commit_fault_recovery import (  # noqa: E501
                        recover_from_commit_fault,
                    )
                    _recovery = await recover_from_commit_fault(self, ctx, exc)
                    _commit_skip_reason = (
                        f"commit_fault:{_recovery.get('fault', 'other')}"
                    )
                    await self._record_ledger(
                        ctx, OperationState.APPLYING,
                        {"event": "commit_fault_recovered", **_recovery},
                    )
                except Exception:  # noqa: BLE001 — recovery must never break APPLY
                    logger.warning(
                        "[Orchestrator] Auto-commit failed for op=%s: %s; change "
                        "is applied but not committed (recovery degraded)",
                        ctx.op_id, exc,
                    )

            # ---- Phase 8b-p: Workspace promotion (Slice 11) ----
            # Inline twin of the Slice4bRunner hook (T5 lesson: BOTH paths).
            # Lands the verified workspace commit on the operator tree;
            # must precede 8b2 (hot-reload re-imports from the REAL tree).
            # Refusals are fail-closed: op -> POSTMORTEM, workspace branch
            # stays quarantined.
            from backend.core.ouroboros.governance.workspace_promoter import (
                run_workspace_promotion,
            )
            _promo = await run_workspace_promotion(
                self, ctx, _committed_hash, best_candidate,
                commit_skipped_reason=_commit_skip_reason,
            )
            # Slice 14: true durable-write probe after 8b, every branch.
            self._emit_terminal_durability_probe(ctx, _committed_hash, _promo)
            # In-Memory Object Surgery (2026-07-22, inline twin): a
            # PENDING outcome (parked on ouroboros/pending/<op>) is NOT
            # a failure — mirrors the slice4b_runner exemption.
            if (
                _promo.attempted and not _promo.promoted
                and not getattr(_promo, "pending", False)
            ):
                if _serpent: _serpent.update_phase("POSTMORTEM")
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code="promotion_failed",
                    rollback_occurred=False,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {"reason": "promotion_failed", "detail": _promo.state},
                )
                await self._publish_outcome(
                    ctx, OperationState.FAILED, "promotion_failed",
                )
                return ctx
            if _promo.promoted:
                try:
                    await self._stack.comm.emit_heartbeat(
                        op_id=ctx.op_id, phase="promotion",
                        progress_pct=98.5,
                        promoted_shas=list(_promo.shas),
                    )
                except Exception:
                    pass

            # ---- Phase 8b2: In-process hot-reload (Manifesto §6 RSI loop closer) ----
            # If this op modified one of our hot-reloadable governance modules,
            # reload it now so the next op uses the freshly-fixed code without
            # a process restart. Quarantined modules trigger a restart_pending
            # flag that the harness honors after the current op completes.
            # Fault-isolated — never raises, never alters terminal state.
            if self._hot_reloader is not None:
                try:
                    _hr_batch = self._hot_reloader.reload_for_op(
                        op_id=ctx.op_id,
                        target_files=ctx.target_files,
                    )
                    if _hr_batch.overall_status == "success":
                        _reloaded_names = [
                            o.module_name.rsplit(".", 1)[-1]
                            for o in _hr_batch.outcomes
                            if o.status == "reloaded"
                        ]
                        logger.info(
                            "[Orchestrator] Hot-reloaded %d module(s) for op=%s: %s",
                            len(_reloaded_names), ctx.op_id, _reloaded_names,
                        )
                        try:
                            await self._stack.comm.emit_heartbeat(
                                op_id=ctx.op_id, phase="hot_reload",
                                progress_pct=99.0,
                                reloaded_modules=_reloaded_names,
                                reload_count=self._hot_reloader.reload_count,
                            )
                        except Exception:
                            pass
                    elif _hr_batch.overall_status in ("reload_failed", "preflight_failed"):
                        logger.warning(
                            "[Orchestrator] Hot-reload failed for op=%s: %s; "
                            "restart will be queued",
                            ctx.op_id, _hr_batch.restart_reason,
                        )
                    elif _hr_batch.restart_required:
                        logger.info(
                            "[Orchestrator] Hot-reload deferred to restart for op=%s: %s",
                            ctx.op_id, _hr_batch.restart_reason,
                        )
                except Exception as exc:
                    logger.warning(
                        "[Orchestrator] Hot-reload hook raised for op=%s: %s",
                        ctx.op_id, exc,
                    )

            # ---- Phase 8c: Self-critique (Phase 3a — post-VERIFY quality signal) ----
            # Runs cheap DW critique over the applied diff against the original
            # goal. Poor ratings (≤2) persist as FEEDBACK memories for future
            # ops; excellent ratings (=5) reinforce file reputation. Fully
            # non-blocking — every failure mode is swallowed.
            if self._critique_engine is not None:
                try:
                    _test_summary = "(no test summary captured)"
                    _vr = ctx.validation
                    if _vr is not None:
                        _passed = getattr(_vr, "tests_passed", 0) or 0
                        _total = getattr(_vr, "tests_total", 0) or 0
                        if _total:
                            _test_summary = f"{_passed}/{_total} tests passed"
                        elif _passed:
                            _test_summary = f"{_passed} tests passed"
                    _critique_result = await asyncio.wait_for(
                        self._critique_engine.critique_op(
                            op_id=ctx.op_id,
                            description=ctx.description,
                            target_files=ctx.target_files,
                            risk_tier=ctx.risk_tier,
                            commit_hash=_committed_hash,
                            test_summary=_test_summary,
                        ),
                        timeout=float(os.environ.get("JARVIS_CRITIQUE_TIMEOUT_S", "30")) + 5.0,
                    )
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id,
                            phase="critique",
                            progress_pct=99.0,
                            critique_rating=int(getattr(_critique_result, "rating", 0)),
                            critique_matches_goal=bool(
                                getattr(_critique_result, "matches_goal", True)
                            ),
                            critique_rationale=str(
                                getattr(_critique_result, "rationale", "")
                            )[:200],
                            critique_provider=str(
                                getattr(_critique_result, "provider_name", "")
                            ),
                            critique_parse_ok=bool(
                                getattr(_critique_result, "parse_ok", True)
                            ),
                        )
                    except Exception:
                        pass
                    # Session lesson: record poor critiques intra-session so
                    # retries this session avoid repeating the pattern.
                    if (
                        getattr(_critique_result, "parse_ok", False)
                        and getattr(_critique_result, "is_poor", False)
                    ):
                        _files_short = ", ".join(
                            p.rsplit("/", 1)[-1] for p in ctx.target_files[:3]
                        )
                        self._add_session_lesson(
                            "code",
                            f"[CRITIQUE POOR {getattr(_critique_result, 'rating', '?')}/5] "
                            f"{ctx.description[:60]} ({_files_short}): "
                            f"{str(getattr(_critique_result, 'rationale', ''))[:120]}",
                            op_id=ctx.op_id,
                        )
                except asyncio.TimeoutError:
                    logger.info(
                        "[Orchestrator] Self-critique timed out for op=%s — "
                        "non-blocking, continuing to COMPLETE",
                        ctx.op_id,
                    )
                except Exception as exc:
                    logger.debug(
                        "[Orchestrator] Self-critique failed for op=%s: %s",
                        ctx.op_id, exc,
                    )

            # ---- Phase 8d: Visual VERIFY (Slices 3-4 — Task 22 handoff #4) ----
            # Runs deterministic UI-regression checks + model-assisted
            # advisory between VERIFY and COMPLETE. Master-switch-gated via
            # ``visual_verify_enabled()`` inside the driver; a disabled
            # sensor returns ``ran=False`` and we proceed to COMPLETE as
            # before (back-compat preserved).
            #
            # Routing per Manifesto §2 DAG:
            #   ran=False      → COMPLETE (unchanged back-compat path)
            #   result=pass    → COMPLETE (FSM transitions VERIFY → VISUAL_VERIFY → COMPLETE)
            #   result=fail OR l2_triggered=True → L2 Repair via ``_l2_hook``,
            #     same path VERIFY-red uses; on L2 convergence we re-apply
            #     the repair candidate and continue to COMPLETE; on L2 escape
            #     we inherit the terminal ctx L2 advanced to and return early.
            try:
                from backend.core.ouroboros.governance.visual_verify import (
                    run_post_verify,
                )
                _vv_outcome = run_post_verify(
                    target_files=ctx.target_files,
                    attachments=ctx.attachments,
                    op_id=ctx.op_id,
                    op_description=ctx.description,
                    plan_ui_affected=False,
                    test_targets_resolved=(
                        ctx.validation.adapter_names_run if ctx.validation else None
                    ),
                    risk_tier=(
                        ctx.risk_tier.name.lower() if ctx.risk_tier else ""
                    ),
                    # We only reach this block on the VERIFY-passed path, so
                    # the I4 clamp's "red" branch never fires here; passing
                    # "passed" explicitly makes the contract obvious.
                    test_runner_result="passed",
                )
                if _vv_outcome.ran:
                    _vv_verdict = (
                        _vv_outcome.result.verdict if _vv_outcome.result else "?"
                    )
                    logger.info(
                        "[Orchestrator] Visual VERIFY outcome=%s "
                        "l2_triggered=%s [%s] %s",
                        _vv_verdict, _vv_outcome.l2_triggered,
                        ctx.op_id, _vv_outcome.reasoning,
                    )
                    # Advance the FSM through VISUAL_VERIFY so the traversal
                    # is auditable in the hash-chained ledger.
                    try:
                        ctx = ctx.advance(OperationPhase.VISUAL_VERIFY)
                    except ValueError as _adv_exc:
                        # Should never happen on the happy VERIFY-passed path
                        # but guard against cancel / postmortem races that
                        # advanced ctx out from under us.
                        logger.debug(
                            "[Orchestrator] VISUAL_VERIFY advance rejected "
                            "(ctx at %s): %s", ctx.phase.name, _adv_exc,
                        )

                    _vv_fail = (
                        _vv_outcome.l2_triggered
                        or (
                            _vv_outcome.result is not None
                            and _vv_outcome.result.verdict == "fail"
                        )
                    )
                    if _vv_fail and self._config.repair_engine is not None:
                        logger.info(
                            "[Orchestrator] Visual VERIFY fail/advisory — "
                            "routing to L2 repair [%s]", ctx.op_id,
                        )
                        _vv_deadline = ctx.pipeline_deadline or (
                            datetime.now(timezone.utc) + timedelta(seconds=60)
                        )
                        _vv_synth_val = ValidationResult(
                            passed=False,
                            best_candidate=best_candidate,
                            validation_duration_s=0.0,
                            error=f"visual_verify: {_vv_outcome.reasoning}",
                            failure_class="test",
                            short_summary=(
                                f"visual_verify: "
                                f"{_vv_outcome.result.check if _vv_outcome.result else 'advisory'}"
                            ),
                            adapter_names_run=(),
                        )
                        try:
                            _vv_directive = await self._l2_hook(
                                ctx, _vv_synth_val, _vv_deadline,
                            )
                            if _vv_directive[0] == "break":
                                # L2 converged — apply the repair candidate.
                                _vv_l2_candidate = _vv_directive[1]
                                _vv_l2_change = self._build_change_request(
                                    ctx, _vv_l2_candidate,
                                )
                                try:
                                    _vv_l2_result = (
                                        await self._stack.change_engine.execute(
                                            _vv_l2_change
                                        )
                                    )
                                    if _vv_l2_result.success:
                                        logger.info(
                                            "[Orchestrator] Visual VERIFY L2 "
                                            "repair applied [%s]", ctx.op_id,
                                        )
                                    else:
                                        logger.warning(
                                            "[Orchestrator] Visual VERIFY L2 "
                                            "repair candidate failed to apply [%s]",
                                            ctx.op_id,
                                        )
                                except Exception as _vv_apply_exc:
                                    logger.debug(
                                        "[Orchestrator] Visual VERIFY L2 apply "
                                        "error: %s", _vv_apply_exc,
                                    )
                            elif _vv_directive[0] == "l2_pivot":
                                # T3 -- Graceful Semantic Pivot in the Visual
                                # VERIFY phase. Mirror the VERIFY consumer:
                                # route through the shared pivot handler
                                # (decompose-further at the failure locus or
                                # HITL DLQ) and return its terminal ctx so an
                                # unresolvable op is NOT mis-marked COMPLETE.
                                # DAG-preserving; OFF byte-identical (engine
                                # only emits L2_PIVOT when epistemic feedback
                                # is enabled).
                                _vv_pivot_sig = (
                                    _vv_directive[2]
                                    if len(_vv_directive) > 2 else ""
                                )
                                _vv_pivot_tail = (
                                    _vv_directive[3]
                                    if len(_vv_directive) > 3 else ""
                                )
                                ctx = await self._handle_l2_pivot(
                                    _vv_directive[1],
                                    _vv_pivot_sig, _vv_pivot_tail,
                                )
                                return ctx
                            elif _vv_directive[0] in ("cancel", "fatal"):
                                # L2 escaped — inherit the terminal ctx.
                                ctx = _vv_directive[1]
                                logger.info(
                                    "[Orchestrator] L2 escaped Visual VERIFY — "
                                    "op ctx advanced to %s [%s]",
                                    ctx.phase.name, ctx.op_id,
                                )
                                return ctx
                        except Exception as _vv_l2_exc:
                            logger.debug(
                                "[Orchestrator] Visual VERIFY L2 failed: "
                                "%s: %s",
                                type(_vv_l2_exc).__name__, _vv_l2_exc,
                            )
            except Exception as _vv_exc:
                # Visual VERIFY dispatch must never break the pipeline.
                # A bug in the driver drops us through to the normal
                # COMPLETE path.
                logger.debug(
                    "[Orchestrator] Visual VERIFY dispatch error: %s: %s",
                    type(_vv_exc).__name__, _vv_exc,
                )

        # Wave 2 (5) Slice 1 — COMPLETERunner delegation gate.
        # Flag JARVIS_PHASE_RUNNER_COMPLETE_EXTRACTED (default false) routes
        # the COMPLETE block through the extracted PhaseRunner. Parity
        # tests pin byte-identical observable output across both paths.
        if _phase_runner_complete_extracted():
            from backend.core.ouroboros.governance.phase_runners.complete_runner import (
                COMPLETERunner,
            )
            logger.info("[PhaseRunnerDelegate] COMPLETE → runner op=%s", ctx.op_id[:16])
            _complete_runner = COMPLETERunner(self, _serpent, _t_apply)
            _complete_result = await _complete_runner.run(ctx)
            return _complete_result.next_ctx

        if _serpent: _serpent.update_phase("COMPLETE")
        ctx = ctx.advance(OperationPhase.COMPLETE, terminal_reason_code="complete")

        # Heartbeat: COMPLETE (Manifesto §7)
        try:
            await self._stack.comm.emit_heartbeat(
                op_id=ctx.op_id, phase="complete", progress_pct=100.0,
            )
        except Exception:
            pass

        self._record_canary_for_ctx(ctx, True, time.monotonic() - _t_apply)
        await self._publish_outcome(ctx, OperationState.APPLIED)
        await self._persist_performance_record(ctx)
        applied_files = [Path(p).resolve() for p in ctx.target_files]
        await self._oracle_incremental_update(applied_files)

        # ---- Phase 4 P3 follow-on: Cognitive Metrics post-APPLY ----
        # Vindication call site — reads the pre-apply OracleSnapshot
        # captured at CONTEXT_EXPANSION (next to score_pre_apply) and
        # records a vindication CognitiveMetricRecord. Adjacent to
        # _oracle_incremental_update so the live oracle has the most
        # recent state when computing after-values. Best-effort: helper
        # body at module scope as
        # `_reflect_cognitive_metrics_post_apply_impl`.
        _reflect_cognitive_metrics_post_apply_impl(ctx, applied_files)

        # ── P0 Wiring: Complete ReasoningNarrator + OperationDialogue ────
        if self._reasoning_narrator is not None:
            try:
                self._reasoning_narrator.record_outcome(ctx.op_id, True, "Applied successfully")
                await self._reasoning_narrator.narrate_completion(ctx.op_id)
            except Exception:
                pass
        if self._dialogue_store is not None:
            try:
                _d = self._dialogue_store.get_active(ctx.op_id)
                if _d:
                    _d.add_entry("COMPLETE", "Applied successfully")
                self._dialogue_store.complete_dialogue(ctx.op_id, "success")
            except Exception:
                pass

        # ── RSI Convergence: compute composite score ──────────────────
        if self._rsi_score_function is not None:
            try:
                _score = self._rsi_score_function.compute(
                    op_id=ctx.op_id,
                    test_pass_rate_before=getattr(ctx, "test_pass_rate_before", 0.0),
                    test_pass_rate_after=1.0 if getattr(ctx, "validation_passed", False) else 0.0,
                    coverage_before=getattr(ctx, "coverage_before", 0.0),
                    coverage_after=getattr(ctx, "coverage_after", 0.0),
                    complexity_before=getattr(ctx, "complexity_before", 0.0),
                    complexity_after=getattr(ctx, "complexity_after", 0.0),
                    # RSI-score signature drift fix (2026-07-22 — inline
                    # twin of complete_runner; keyword names are
                    # lint_violations_*).
                    lint_violations_before=getattr(ctx, "lint_before", 0),
                    lint_violations_after=getattr(ctx, "lint_after", 0),
                    blast_radius_total=getattr(ctx, "blast_radius_total", 0),
                )
                logger.info("[RSI Score] op=%s composite=%.4f", ctx.op_id, _score.composite)
            except Exception:
                logger.debug("RSI score computation failed", exc_info=True)

        # ── Ouroboros Serpent: stop animation ──
        if _serpent:
            try:
                await _serpent.stop(success=True)
            except Exception:
                pass

        return ctx

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_canary_for_ctx(
        self,
        ctx: OperationContext,
        success: bool,
        latency_s: float,
        rolled_back: bool = False,
    ) -> None:
        """Record canary telemetry for every file in ctx.target_files."""
        for f in ctx.target_files:
            self._stack.canary.record_operation(
                file_path=str(f),
                success=success,
                latency_s=latency_s,
                rolled_back=rolled_back,
            )

    async def _publish_outcome(
        self,
        ctx: OperationContext,
        final_state: OperationState,
        error_pattern: Optional[str] = None,
    ) -> None:
        """Publish operation outcome to LearningBridge + SuccessPatternStore.

        Fault-isolated — never raises. Records both failures (LearningBridge)
        and successes (SuccessPatternStore) for the adaptive learning loop.
        """
        if self._stack.learning_bridge is None:
            return
        try:
            outcome = OperationOutcome(
                op_id=ctx.op_id,
                goal=ctx.description,
                target_files=list(ctx.target_files),
                final_state=final_state,
                error_pattern=error_pattern,
            )
            await self._stack.learning_bridge.publish(outcome)
        except Exception:
            logger.exception(
                "[Orchestrator] LearningBridge.publish failed for op %s; outcome not recorded",
                ctx.op_id,
            )

        # P2: Record success patterns for positive feedback loop
        if final_state in (OperationState.APPLIED,):
            try:
                from backend.core.ouroboros.governance.adaptive_learning import (
                    SuccessPatternStore,
                )
                from backend.core.ouroboros.governance.entropy_calculator import (
                    extract_domain_key as _extract_dk,
                )
                _domain = _extract_dk(ctx.target_files, ctx.description)
                _provider = ""
                if ctx.generation is not None:
                    _provider = ctx.generation.provider_name
                _store = SuccessPatternStore()
                _store.record_success(
                    domain_key=_domain,
                    description=ctx.description,
                    target_files=ctx.target_files,
                    provider=_provider,
                    approach_summary=f"Succeeded via {_provider} on {len(ctx.target_files)} files",
                )
                logger.debug(
                    "[Orchestrator] Success pattern recorded: domain=%s provider=%s (op=%s)",
                    _domain, _provider, ctx.op_id,
                )
            except Exception:
                pass  # Positive feedback is best-effort — never block

        # P2.3: Provider performance tracking — model-selection learning.
        # Records (provider, complexity, success, duration) so future routing
        # can prefer the provider that succeeds at this complexity class.
        try:
            from backend.core.ouroboros.governance.adaptive_learning import (
                ProviderPerformanceTracker,
            )
            _provider = ""
            _gen_duration = 0.0
            if ctx.generation is not None:
                _provider = ctx.generation.provider_name
                _gen_duration = ctx.generation.generation_duration_s
            if _provider:
                _complexity = getattr(ctx, "task_complexity", "unknown") or "unknown"
                _is_success = final_state in (OperationState.APPLIED,)
                _tracker = ProviderPerformanceTracker()
                _tracker.record(
                    provider=_provider,
                    complexity=_complexity,
                    success=_is_success,
                    generation_s=_gen_duration,
                )
                _tracker.persist()
                logger.debug(
                    "[Orchestrator] Provider performance: %s/%s/%s (%.1fs)",
                    _provider, _complexity,
                    "OK" if _is_success else "FAIL", _gen_duration,
                )
        except Exception:
            pass  # Provider tracking is best-effort

        # Self-evolution feedback: record outcome for prompt adaptation +
        # negative constraints + evolution tracking
        try:
            from backend.core.ouroboros.governance.self_evolution import (
                RuntimePromptAdapter, NegativeConstraintStore,
                MultiVersionEvolutionTracker,
            )
            from backend.core.ouroboros.governance.entropy_calculator import (
                extract_domain_key as _se_edk,
            )
            _se_domain = _se_edk(ctx.target_files, ctx.description)
            _is_success = final_state in (OperationState.APPLIED,)

            # P0: Record for runtime prompt adaptation
            _pa = RuntimePromptAdapter()
            _pa.record_outcome(
                _se_domain, ctx.op_id, _is_success,
                failure_class=error_pattern or "",
            )

            # P0: Add negative constraint on failure
            if not _is_success and error_pattern:
                _ns = NegativeConstraintStore()
                _ns.add_constraint(
                    _se_domain,
                    f'Avoid pattern that caused "{error_pattern}"',
                    f"Operation {ctx.op_id} failed: {error_pattern}",
                    source_op_id=ctx.op_id,
                    severity="soft",
                )

            # P2: Multi-version evolution tracking
            _evt = MultiVersionEvolutionTracker()
            _evt.record_operation(_is_success, len(ctx.target_files))

            # P2: LearningConsolidator — periodic consolidation of outcomes into rules
            # Accumulates outcomes and consolidates when enough data is available.
            try:
                from backend.core.ouroboros.governance.adaptive_learning import (
                    LearningConsolidator,
                )
                _lc = LearningConsolidator()
                _provider_name = ""
                if ctx.generation is not None:
                    _provider_name = ctx.generation.provider_name
                _outcome_dict = {
                    "domain_key": _se_domain,
                    "success": _is_success,
                    "error_pattern": error_pattern or "",
                    "provider": _provider_name,
                    "target_files": list(ctx.target_files),
                }
                # Buffer outcome in a module-level accumulator; consolidate
                # when the buffer reaches threshold (10 outcomes).
                _CONSOLIDATION_BUFFER.append(_outcome_dict)
                if len(_CONSOLIDATION_BUFFER) >= _CONSOLIDATION_THRESHOLD:
                    _new_rules = _lc.consolidate(list(_CONSOLIDATION_BUFFER))
                    _CONSOLIDATION_BUFFER.clear()
                    if _new_rules:
                        logger.info(
                            "[Orchestrator] LearningConsolidator: %d new rules from %d outcomes",
                            len(_new_rules), _CONSOLIDATION_THRESHOLD,
                        )
            except Exception:
                pass  # Consolidation is best-effort

        except Exception:
            pass  # Self-evolution feedback is best-effort

        # JARVIS Tier 6: Record operation in PersonalityEngine
        _gls = getattr(self._stack, "governed_loop_service", None)
        if _gls is not None:
            _pe = getattr(_gls, "_personality_engine", None)
            if _pe is not None:
                try:
                    _pe.record_operation(_is_success)
                except Exception:
                    pass

            # JARVIS Tier 2: Record alert in EmergencyEngine on failure
            if not _is_success:
                _ee = getattr(_gls, "_emergency_engine", None)
                if _ee is not None:
                    try:
                        from backend.core.ouroboros.governance.emergency_protocols import AlertType
                        _ee.record_alert(
                            AlertType.GENERATION_FAILURE,
                            f"Operation {ctx.op_id} failed: {error_pattern or 'unknown'}",
                            ctx.op_id,
                        )
                    except Exception:
                        pass

        # ── RSI Convergence: check convergence state ──────────────────
        if self._rsi_score_history is not None and self._rsi_convergence_tracker is not None:
            try:
                composites = self._rsi_score_history.get_composite_values()
                if len(composites) >= 5:
                    _report = self._rsi_convergence_tracker.analyze(composites)
                    logger.info(
                        "[RSI Convergence] state=%s slope=%.4f r2_log=%.2f recommendation=%s",
                        _report.state.value, _report.slope,
                        _report.r_squared_log, _report.recommendation,
                    )
            except Exception:
                logger.debug("RSI convergence check failed", exc_info=True)

        # ── RSI Convergence: record technique outcomes ────────────────
        if self._rsi_transition_tracker is not None:
            try:
                from backend.core.ouroboros.governance.transition_tracker import TechniqueOutcome
                _techniques = getattr(ctx, "techniques_applied", [])
                _domain = getattr(ctx, "domain", "unknown")
                _complexity = getattr(ctx, "task_complexity", "unknown")
                _composite = getattr(ctx, "composite_score", 0.5)
                for _tech in _techniques:
                    self._rsi_transition_tracker.record(TechniqueOutcome(
                        technique=_tech, domain=_domain, complexity=_complexity,
                        success=(final_state.value in ("applied", "complete")),
                        composite_score=_composite, op_id=ctx.op_id,
                    ))
            except Exception:
                logger.debug("RSI transition tracking failed", exc_info=True)

        # ── Session Intelligence: record ephemeral lesson ──────────────
        # Each lesson is a (type, text) tuple.  Type is "code" or "infra".
        # Infrastructure failures (timeouts, provider outages) are excluded
        # from generation prompts to avoid poisoning the model with
        # environmentally-caused failures that don't reflect code quality.
        _INFRA_PATTERNS = frozenset({
            "timeout", "connection_error", "budget", "all_providers_exhausted",
            "pypi_timeout", "change_engine_error", "infrastructure_failed",
            "deadline_exceeded", "provider_unavailable", "rate_limited",
        })
        try:
            _files_short = ", ".join(str(f).split("/")[-1] for f in list(ctx.target_files)[:2])
            _err = error_pattern or ""
            _is_infra = any(p in _err.lower() for p in _INFRA_PATTERNS)
            _lesson_type = "infra" if _is_infra else "code"
            if final_state in (OperationState.APPLIED,):
                _lesson_text = f"[OK] {ctx.description[:80]} ({_files_short})"
            else:
                # P1.3: Causal post-mortem — deterministic analysis of what
                # went wrong and what the model should do differently next time.
                _causal = self._causal_postmortem(_err, ctx)
                _lesson_text = (
                    f"[FAIL:{_err or 'unknown'}] {ctx.description[:60]} "
                    f"({_files_short}) — {_causal}"
                )
            self._add_session_lesson(_lesson_type, _lesson_text, op_id=ctx.op_id)

            # ── Convergence metric: track success rate before/after first lesson ──
            _has_lessons = len(self._session_lessons) > 1  # >1 = lessons exist from prior ops
            if _has_lessons:
                self._ops_after_lesson += 1
                if _is_success:
                    self._ops_after_lesson_success += 1
                # Periodic check: if post-lesson success rate is worse, clear lessons
                if (self._ops_after_lesson > 0
                        and self._ops_after_lesson % self._convergence_check_interval == 0):
                    _pre_rate = (
                        self._ops_before_lesson_success / max(1, self._ops_before_lesson)
                    )
                    _post_rate = (
                        self._ops_after_lesson_success / max(1, self._ops_after_lesson)
                    )
                    if _post_rate < _pre_rate and self._ops_before_lesson >= 3:
                        logger.warning(
                            "[Orchestrator] Session intelligence convergence NEGATIVE: "
                            "pre-lesson %.0f%% (%d/%d) > post-lesson %.0f%% (%d/%d) — clearing lesson buffer",
                            _pre_rate * 100, self._ops_before_lesson_success, self._ops_before_lesson,
                            _post_rate * 100, self._ops_after_lesson_success, self._ops_after_lesson,
                        )
                        # Slice 1: clear + counter-reset is one atomic unit
                        # under the lessons lock (a concurrent _add_session_
                        # lesson interleaving mid-reset would corrupt the
                        # convergence metric's epoch).
                        with self._session_lessons_lock:
                            self._session_lessons.clear()
                            # Reset counters so the metric starts fresh
                            self._ops_before_lesson = self._ops_after_lesson
                            self._ops_before_lesson_success = self._ops_after_lesson_success
                            self._ops_after_lesson = 0
                            self._ops_after_lesson_success = 0
                    else:
                        logger.info(
                            "[Orchestrator] Session intelligence convergence OK: "
                            "pre-lesson %.0f%% post-lesson %.0f%%",
                            _pre_rate * 100, _post_rate * 100,
                        )
            else:
                self._ops_before_lesson += 1
                if _is_success:
                    self._ops_before_lesson_success += 1
        except Exception:
            pass  # Session lessons are best-effort

    @staticmethod
    def _build_dependency_summary(
        oracle: Any,
        target_files: Sequence[str],
    ) -> str:
        """Build a ~200-token dependency summary from the Oracle graph.

        Queries direct dependents, transitive importers, and blast radius
        for each target file.  The summary is injected into the generation
        prompt so the model avoids breaking downstream consumers.

        Returns empty string if the Oracle is unavailable or target files
        have no dependents.
        """
        if oracle is None or not target_files:
            return ""

        lines: list = []
        seen_files: set = set()

        for raw_path in target_files[:3]:  # Cap at 3 files to stay within budget
            try:
                # Slice 113: this builder is SYNC, so reach the underlying
                # in-process Oracle directly via the adapter's ``.raw`` (avoids
                # an async cascade through every caller). In-process → identical
                # behavior; under process isolation the graph is in another
                # process so this sync path simply degrades (caught below).
                _raw_oracle = getattr(oracle, "raw", oracle)
                ctx_info = _raw_oracle.get_context_for_improvement(raw_path, max_depth=2)
            except Exception:
                continue

            if not ctx_info.get("found"):
                continue

            risk = ctx_info.get("risk_assessment", {})
            dependents = ctx_info.get("dependents", [])
            related = ctx_info.get("related_files", [])

            if not dependents and not related:
                continue

            # Direct dependents (files that import/call this target)
            dep_paths = []
            for d in dependents[:8]:
                fp = d.get("file_path", "") if isinstance(d, dict) else getattr(d, "file_path", "")
                if fp and fp not in seen_files:
                    dep_paths.append(fp)
                    seen_files.add(fp)

            risk_level = risk.get("risk_level", "low")
            total_affected = risk.get("total_affected", 0)

            file_line = f"**{raw_path}** — risk={risk_level}, {total_affected} affected"
            if dep_paths:
                file_line += f"\n  Dependents: {', '.join(dep_paths[:6])}"
                if len(dep_paths) > 6:
                    file_line += f" (+{len(dep_paths) - 6} more)"
            lines.append(file_line)

        if not lines:
            return ""

        return (
            "## Dependency Impact (from Oracle graph)\n\n"
            "These files import/call your targets. Ensure changes are "
            "backward-compatible or update dependents too.\n\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _causal_postmortem(error_pattern: str, ctx: "OperationContext") -> str:
        """Deterministic causal analysis of a failed operation.

        Maps failure reason codes to actionable lessons the model can use
        in subsequent generations.  No LLM call — pure pattern matching.
        Returns a short (<100 word) causal sentence.
        """
        _err = (error_pattern or "").lower()
        _n_files = len(ctx.target_files)

        # Generation failures
        if "generation_failed" in _err:
            return (
                "All generation attempts failed. The prompt may be too large or "
                "the task too ambiguous. Try: narrower scope, fewer target files, "
                "or split into smaller operations."
            )
        if "tool_loop_max_iterations" in _err:
            return (
                "Model exhausted the tool loop without producing a patch. "
                "It may be over-exploring. Try: more specific task description."
            )
        if "tool_loop_budget_exceeded" in _err:
            return (
                "Accumulated tool context exceeded the prompt budget. "
                "Too many large file reads. Use targeted line ranges instead."
            )

        # Validation failures
        if "no_candidate_valid" in _err:
            return (
                "Generated code failed validation (tests or type checks). "
                "Read the test file first and ensure the patch matches expected behavior."
            )
        if "source_drift" in _err:
            return (
                "Target file changed between generation and application. "
                "Another operation may have modified the same file. "
                "Re-read before patching."
            )
        if "schema_invalid" in _err or "validate_diff" in _err:
            return (
                "Generated output didn't match the expected JSON schema. "
                "Ensure the response contains a valid diff block with correct format."
            )

        # Apply failures
        if "change_engine" in _err:
            return (
                "Patch application failed. The diff likely targets lines that "
                "no longer exist. Use read_file to verify the exact current content "
                "before generating the diff."
            )
        if "stale_diff" in _err:
            return (
                "The diff references content that doesn't match the current file. "
                "Always read_file immediately before generating a patch."
            )

        # Verify failures
        if "verify_regression" in _err:
            return (
                "Post-apply tests regressed. The change broke existing behavior. "
                "Check dependents with search_code/get_callers before modifying "
                "shared functions."
            )

        # Security / gate failures
        if "security_review_blocked" in _err:
            return (
                "Security review blocked the change. Avoid patterns like: "
                "hardcoded secrets, command injection, unsafe deserialization."
            )
        if "gate_blocked" in _err:
            return "File write permission denied. Check file lock state."

        # Budget / provider failures (infra — less actionable but still logged)
        if "budget" in _err or "exhausted" in _err:
            return "Provider budget exhausted. Operation was too expensive."
        if "timeout" in _err or "deadline" in _err:
            return (
                "Operation timed out. Consider: simpler task scope, fewer files, "
                "or check if the provider is under load."
            )

        # Fallback
        if _n_files > 3:
            return (
                f"Failed on {_n_files}-file operation. Multi-file changes are "
                "harder — consider splitting into single-file operations."
            )
        return "Unknown failure. Read target files and check dependents before retrying."

    async def _maybe_complete_cosmetic_candidate(
        self, ctx: "OperationContext", generation: Any,
    ) -> Optional["OperationContext"]:
        """Slice 13/15 semantic value gate — SHARED by the legacy inline
        seam and dispatch_pipeline's GENERATE→VALIDATE transition (the live
        route; Run-24 proved the inline seam alone is never reached there).

        When EVERY file of EVERY candidate is mathematically cosmetic
        (candidate_value_gate: AST equality after docstring stripping /
        declared line-grammar normalization), terminates the op as the
        benign ``no_op_cosmetic`` completion and returns the terminal ctx —
        no candidate tree, no APPLY, no VERIFY, no commit, no promotion.
        Returns ``None`` (op proceeds untouched) for any substantive or
        indeterminate candidate, when the master is off, and on ANY
        internal error (fail-safe FORWARD, mandate 4). Verdict + per-file
        reasoning logged on every evaluation (Slice 14 — never silent).
        Master: ``JARVIS_CANDIDATE_VALUE_GATE_ENABLED`` (default true).
        """
        # ── PRD §30 slice 3: the `explore` rung's mutation veto ──────────
        #
        # Placed HERE, on the shared helper, for the reason this method's own
        # docstring records: Run-24 proved the inline seam alone is never
        # reached on the live route, so a gate wired only there is wired and
        # inert. Both callers — the legacy inline seam and dispatch_pipeline's
        # GENERATE→VALIDATE transition — pass through this one method, which
        # makes it the only placement that cannot silently miss the shipping
        # path.
        #
        # A veto is a benign TERMINAL, not a retry: `ExplorationInsufficient`
        # routes through GENERATE_RETRY because a model can fix insufficient
        # exploration by exploring more, and it cannot fix the operator's
        # dial. Retrying here would burn the retry budget on a condition no
        # generation can satisfy and then fail the op for something that was
        # never the model's fault.
        _veto = None
        try:
            from backend.core.ouroboros.governance.proactive_mode import (
                mutation_permitted as _pm_mutation_permitted,
            )
            _veto = _pm_mutation_permitted()
        except Exception:  # noqa: BLE001 — fail OPEN, never halt the organism
            _veto = None
        if _veto is not None and not _veto.permitted:
            _n_cands = len(getattr(generation, "candidates", None) or ())
            logger.info(
                "[ProactiveMode] op=%s vetoed %d candidate(s) — %s; "
                "completing as no_op_mode_veto, skipping "
                "VALIDATE/APPLY/VERIFY",
                ctx.op_id, _n_cands, _veto.reason,
            )
            try:
                await self._stack.comm.emit_postmortem(
                    op_id=ctx.op_id,
                    root_cause="no_op_mode_veto",
                    failed_phase=None,
                    next_safe_action="none",
                )
            except Exception:
                logger.debug(
                    "[ProactiveMode] postmortem emit failed", exc_info=True,
                )
            _v_ctx = ctx.advance(
                OperationPhase.COMPLETE,
                generation=generation,
                terminal_reason_code="no_op_mode_veto",
            )
            try:
                # Same ledger shape the cosmetic terminal writes, so an
                # operator reading the ledger sees one vocabulary for "the
                # op completed without mutating" rather than two.
                await self._record_ledger(
                    _v_ctx,
                    OperationState.APPLIED,
                    {
                        "reason": "no_op_mode_veto",
                        "position": _veto.position,
                        "candidates": _n_cands,
                    },
                )
            except Exception:
                logger.debug(
                    "[ProactiveMode] ledger record failed", exc_info=True,
                )
            return _v_ctx

        if os.environ.get(
            "JARVIS_CANDIDATE_VALUE_GATE_ENABLED", "true",
        ).strip().lower() not in ("1", "true", "yes", "on"):
            return None
        try:
            from backend.core.ouroboros.governance.candidate_value_gate import (  # noqa: E501
                COSMETIC as _VG_COSMETIC,
                evaluate_candidate_value,
            )
            _vg_root = Path(self._config.execution_root)
            _vg_cands = getattr(generation, "candidates", None) or []
            _vg_all_cosmetic = bool(_vg_cands)
            _vg_detail: list = []
            for _vg_cand in _vg_cands:
                _vg_files = self._iter_candidate_files(_vg_cand)
                _verdict, _d = evaluate_candidate_value(_vg_root, _vg_files)
                _vg_detail.extend(_d)
                if _verdict != _VG_COSMETIC:
                    _vg_all_cosmetic = False
                    break
            logger.debug(
                "[ValueGate] verdict op=%s all_cosmetic=%s files=%s",
                ctx.op_id, _vg_all_cosmetic,
                [(p, v) for p, v in _vg_detail],
            )
            if not _vg_all_cosmetic:
                return None
            logger.info(
                "[ValueGate] op=%s all %d candidate file(s) proven cosmetic "
                "(no executable-logic change) — completing as "
                "no_op_cosmetic, skipping VALIDATE/APPLY/VERIFY",
                ctx.op_id, len(_vg_detail),
            )
            try:
                await self._stack.comm.emit_postmortem(
                    op_id=ctx.op_id,
                    root_cause="no_op_cosmetic",
                    failed_phase=None,
                    next_safe_action="none",
                )
            except Exception:
                logger.debug(
                    "[ValueGate] postmortem emit failed", exc_info=True,
                )
            _t_ctx = ctx.advance(
                OperationPhase.COMPLETE,
                generation=generation,
                terminal_reason_code="no_op_cosmetic",
            )
            await self._record_ledger(
                _t_ctx,
                OperationState.APPLIED,
                {
                    "reason": "no_op_cosmetic",
                    "files": [p for p, _ in _vg_detail],
                },
            )
            return _t_ctx
        except Exception:  # noqa: BLE001 — gate must never kill the pipeline
            logger.debug(
                "[ValueGate] evaluation skipped (non-fatal)", exc_info=True,
            )
            return None

    @staticmethod
    def _terminal_durability(
        committed_hash: Optional[str], promo: Any,
    ) -> bool:
        """Slice 14 — the TRUE durable-write state after Phase 8b resolves.

        Durable iff a workspace commit exists AND promotion either
        succeeded or was legitimately NOT attempted (master off / same
        root / no net change — the commit itself is then the durable
        artifact). An ATTEMPTED-but-unpromoted outcome (refused, aborted,
        failed) is NOT durable — mandate 4, absolutely.
        """
        if not committed_hash:
            return False
        if promo is None:
            return True  # promotion subsystem absent — legacy commit posture
        attempted = bool(getattr(promo, "attempted", False))
        promoted = bool(getattr(promo, "promoted", False))
        if attempted and not promoted:
            return False
        return True

    @staticmethod
    def _render_durability_probe(op_id: str, durable: bool) -> str:
        """The EXACT Slice74Probe LEDGER_TERMINAL schema (mandate 3 — no
        new logging sequence; the auditor's own regex must match)."""
        return (
            "[Slice74Probe] LEDGER_TERMINAL op_id=%s state=applied written=%s"
            % (op_id, durable)
        )

    def _emit_terminal_durability_probe(
        self, ctx: Any, committed_hash: Optional[str], promo: Any,
    ) -> None:
        """Re-emit the terminal probe strictly AFTER Phase 8b (AutoCommit +
        WorkspacePromoter) fully resolves (Slice 14, Run-23 final red).

        The in-``_record_ledger`` probe reports the ledger DEDUP boolean and
        fires at terminal-record time — BEFORE 8b — so every mutating op
        stamped ``written=False`` while its repair demonstrably landed
        (abbddeec24). This re-emission carries the true asynchronous
        durable-write state; the auditor greps the literal either way.
        Never raises.
        """
        try:
            durable = self._terminal_durability(committed_hash, promo)
            logger.info(
                "%s", self._render_durability_probe(
                    getattr(ctx, "op_id", "?"), durable,
                ),
            )
        except Exception:  # noqa: BLE001 — probe must never break 8b
            logger.debug("[Slice74Probe] re-emit failed", exc_info=True)

    def _scoped_verify_runner(
        self, exec_root: Path,
    ) -> Tuple[Any, Optional[Path]]:
        """Select the scoped post-APPLY verify runner for ``exec_root``.

        Legacy (execution root == observation root): the boot-time
        ``_validation_runner`` with ``sandbox_dir=None`` — byte-identical.

        Redirected (Ledger-Sovereignty workspace active): a per-root
        ``LanguageRouter`` anchored at ``exec_root`` — the SAME discipline
        as Slice 9's candidate-tree VALIDATE. Anchoring MUST live in the
        router/adapters' ``repo_root``: ``PythonAdapter.run`` ignores the
        ``sandbox_dir`` kwarg (see the L2 lane note in this file), and a
        boot-runner rooted at the real tree would let pytest resolve
        ``backend.*`` imports from the UNPATCHED tree — silently judging
        the wrong source, the exact Run-21 class this seam kills.
        """
        _exec = Path(exec_root)
        if _exec == Path(self._config.project_root):
            return self._validation_runner, None
        from backend.core.ouroboros.governance.test_runner import (
            CppAdapter,
            LanguageRouter,
            PythonAdapter,
        )

        router = LanguageRouter(
            repo_root=_exec,
            adapters={
                "python": PythonAdapter(repo_root=_exec),
                "cpp": CppAdapter(repo_root=_exec),
            },
        )
        return router, _exec

    async def _run_benchmark(
        self,
        ctx: OperationContext,
        applied_files: Sequence[Path],
    ) -> OperationContext:
        """Run PatchBenchmarker. Fault-isolated — never raises, never alters terminal state."""
        if not self._config.benchmark_enabled:
            return ctx
        try:
            # Deferred import: re-binds on every call so ModuleHotReloader
            # changes to patch_benchmarker.py take effect on the next op
            # without a process restart.
            from backend.core.ouroboros.governance.patch_benchmarker import (
                PatchBenchmarker,
            )
            # Slice 11: the benchmarker is the pass_rate source for the
            # VERIFY regression gate — it must judge the tree APPLY wrote
            # (the execution root), never the observation root.
            benchmarker = PatchBenchmarker(
                project_root=self._config.execution_root,
                timeout_s=self._config.benchmark_timeout_s,
                pre_apply_snapshots=getattr(ctx, "pre_apply_snapshots", {}),
            )
            result = await asyncio.wait_for(
                benchmarker.benchmark(ctx),
                timeout=self._config.benchmark_timeout_s,
            )
            return ctx.with_benchmark_result(result)
        except asyncio.CancelledError:
            logger.debug(
                "[Orchestrator] Benchmark cancelled for op=%s; continuing without metrics",
                ctx.op_id,
            )
            return ctx
        except Exception as exc:
            logger.warning(
                "[Orchestrator] Benchmark failed for op=%s: %s; continuing without metrics",
                ctx.op_id, exc,
            )
            return ctx

    async def _persist_performance_record(self, ctx: OperationContext) -> None:
        """Write PerformanceRecord to persistence. Fault-isolated — never raises."""
        if self._stack.performance_persistence is None:
            return
        try:
            br = getattr(ctx, "benchmark_result", None)
            record = PerformanceRecord(
                model_id=getattr(ctx, "model_id", None) or "unknown",
                task_type=br.task_type if br else "code_improvement",
                difficulty=getattr(ctx, "difficulty", TaskDifficulty.MODERATE),
                success=ctx.phase == OperationPhase.COMPLETE,
                latency_ms=getattr(ctx, "elapsed_ms", 0.0),
                iterations_used=getattr(ctx, "iterations_used", 1),
                code_quality_score=br.quality_score if br else 0.0,
                op_id=ctx.op_id,
                patch_hash=br.patch_hash if br else "",
                pass_rate=br.pass_rate if br else 0.0,
                lint_violations=br.lint_violations if br else 0,
                coverage_pct=br.coverage_pct if br else 0.0,
                complexity_delta=br.complexity_delta if br else 0.0,
            )
            await self._stack.performance_persistence.save_record(record)
        except Exception as exc:
            logger.warning(
                "[Orchestrator] PerformanceRecord persist failed for op=%s: %s",
                ctx.op_id, exc,
            )

    async def _oracle_incremental_update(
        self,
        applied_files: Sequence[Path],
    ) -> None:
        """Notify Oracle of changed files after successful COMPLETE. Fault-isolated — never raises."""
        oracle = getattr(self._stack, "oracle", None)
        if oracle is None:
            return
        try:
            async with self._oracle_update_lock:
                # P1-6: shielded_wait_for — oracle index is a must-complete write.
                # Cancellation leaves the index partially stale; shielding lets the
                # update finish in the background while we surface TimeoutError.
                from backend.core.async_safety import shielded_wait_for as _shielded_wf
                await _shielded_wf(
                    oracle.incremental_update(applied_files),
                    timeout=30.0,
                    name="oracle.incremental_update",
                )
        except asyncio.TimeoutError:
            logger.warning(
                "[Orchestrator] Oracle incremental_update timed out (>30s); "
                "update continues in background"
            )
        except asyncio.CancelledError:
            pass  # swallow — oracle update is non-blocking; don't abort COMPLETE
        except Exception as exc:
            logger.warning(
                "[Orchestrator] Oracle incremental_update failed: %s", exc
            )

    def _build_profile(self, ctx: OperationContext) -> OperationProfile:
        """Build an OperationProfile from the context's target files.

        Uses conservative defaults for blast radius and security surface
        detection since the orchestrator doesn't have deep code analysis.
        Real implementations would enrich this via blast-radius adapters.
        """
        target_paths = [Path(f) for f in ctx.target_files]

        # Conservative heuristics for profile fields
        touches_supervisor = any(
            "supervisor" in str(p).lower() for p in target_paths
        )
        touches_security = any(
            any(kw in str(p).lower() for kw in ("auth", "secret", "cred", "token", "encrypt"))
            for p in target_paths
        )
        is_core = any(
            any(kw in str(p).lower() for kw in ("router", "controller", "engine", "orchestrator"))
            for p in target_paths
        )

        # ── Slice 20 — Delegated Provenance threading ──
        # Under the master flag, the profile carries the signal's REAL source
        # plus any provenance CLAIM riding the existing evidence pipe
        # (ctx.intake_evidence_json), so risk_engine's self-protection gate
        # can verify operator-signed roadmap authority. Flag OFF (default)
        # ⇒ source stays "" and provenance None — byte-identical
        # pre-Slice-20 classification. Fail-soft: any fault degrades to the
        # legacy (unsourced) profile, which classifies strictly harsher.
        _s20_source = ""
        _s20_provenance = None
        try:
            from backend.core.ouroboros.governance.delegated_provenance import (
                delegated_provenance_enabled,
                extract_claim_from_evidence_json,
            )
            if delegated_provenance_enabled():
                _s20_source = str(getattr(ctx, "signal_source", "") or "")
                _s20_provenance = extract_claim_from_evidence_json(
                    getattr(ctx, "intake_evidence_json", "") or "",
                )
        except Exception:  # noqa: BLE001 — never perturb classification
            _s20_source, _s20_provenance = "", None

        return OperationProfile(
            files_affected=target_paths,
            change_type=ChangeType.MODIFY,
            blast_radius=len(target_paths),
            crosses_repo_boundary=False,
            touches_security_surface=touches_security,
            touches_supervisor=touches_supervisor,
            test_scope_confidence=0.8,
            is_dependency_change=False,
            is_core_orchestration_path=is_core,
            source=_s20_source,
            provenance=_s20_provenance,
        )

    @staticmethod
    def _ast_preflight(content: str) -> Optional[str]:
        """Return a short error string if content fails ast.parse, else None.

        Parameters
        ----------
        content:
            Python source code to parse.

        Returns
        -------
        Optional[str]
            ``None`` if the content parses cleanly, or a human-readable error
            string (e.g. ``"SyntaxError: invalid syntax (<unknown>, line 1)"``).
        """
        try:
            ast.parse(content)
            return None
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

    @staticmethod
    def _check_source_drift(
        candidate: Dict[str, Any],
        project_root: Path,
    ) -> Optional[str]:
        """Return None if source unchanged; return current hash if drift detected.

        Compares candidate["source_hash"] (hash at generation time) against the
        current file content hash.  Returns None if no source_hash recorded
        (skip check) or file not found (let APPLY handle).

        Parameters
        ----------
        candidate:
            Candidate dict containing ``source_hash`` (hash at generation time)
            and ``file_path`` (relative path from project root).
        project_root:
            Root directory of the project being modified.

        Returns
        -------
        Optional[str]
            ``None`` if no drift (source unchanged or check skipped), or the
            current file's SHA-256 hex digest if drift was detected.
        """
        import hashlib as _hl
        source_hash = candidate.get("source_hash", "")
        if not source_hash:
            return None  # nothing to compare — skip
        file_path = project_root / candidate.get("file_path", "")
        try:
            current_content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None  # file not found — let APPLY handle
        current_hash = _hl.sha256(current_content.encode()).hexdigest()
        return current_hash if current_hash != source_hash else None

    async def _materialize_execution_graph_candidate(
        self,
        ctx: OperationContext,
        candidate: Dict[str, Any],
    ) -> Tuple[OperationContext, Dict[str, Any]]:
        """Execute an L3 execution graph and convert it into saga-ready patches."""
        graph = candidate.get("execution_graph")
        if graph is None:
            return ctx, candidate

        scheduler = self._config.execution_graph_scheduler
        if scheduler is None:
            raise RuntimeError("execution_graph_scheduler_unavailable")

        ctx = ctx.with_execution_graph_metadata(
            execution_graph_id=graph.graph_id,
            execution_plan_digest=graph.plan_digest,
            subagent_count=len(graph.units),
            parallelism_budget=graph.concurrency_limit,
            causal_trace_id=graph.causal_trace_id,
        )

        submitted = await scheduler.submit(graph)
        if not submitted and not scheduler.has_graph(graph.graph_id):
            raise RuntimeError(f"execution_graph_submit_rejected:{graph.graph_id}")

        if ctx.pipeline_deadline is not None:
            timeout_s = max(
                0.1,
                (ctx.pipeline_deadline - datetime.now(tz=timezone.utc)).total_seconds(),
            )
        else:
            timeout_s = max(sum(unit.timeout_s for unit in graph.units), 1.0)

        state = await scheduler.wait_for_graph(graph.graph_id, timeout_s=timeout_s)
        if state.phase.value != "completed":
            raise RuntimeError(
                f"execution_graph_terminal:{state.phase.value}:{state.last_error or 'unknown'}"
            )

        updated = dict(candidate)
        updated["patches"] = scheduler.get_merged_patches(graph.graph_id)
        return ctx, updated

    # Phases where code has already been written to disk. An L2 escape from
    # any of these is a *regression* (disk state diverged from baseline) and
    # must be recorded as POSTMORTEM so the forensic path runs. Escapes from
    # earlier phases have touched no files and can safely be CANCELLED
    # (graceful abort). This set is the single source of truth — any new
    # post-apply phase added to the FSM should be added here once.
    _POST_APPLY_PHASES: frozenset = frozenset({
        OperationPhase.APPLY,
        OperationPhase.VERIFY,
    })

    @classmethod
    def _l2_escape_terminal(cls, current_phase: OperationPhase) -> OperationPhase:
        """Return the appropriate terminal phase for an L2 escape.

        Principle: once code has touched disk (APPLY/VERIFY), an escape is a
        regression requiring forensics → POSTMORTEM. Before that, the op
        hasn't altered any files, so a graceful abort is a user-level
        cancellation → CANCELLED.

        Parameters
        ----------
        current_phase:
            The phase the ctx is in when L2 is invoked.

        Returns
        -------
        OperationPhase
            Either ``POSTMORTEM`` (post-apply) or ``CANCELLED`` (pre-apply).
        """
        if current_phase in cls._POST_APPLY_PHASES:
            return OperationPhase.POSTMORTEM
        return OperationPhase.CANCELLED

    async def _l2_hook(
        self,
        ctx: "OperationContext",
        best_validation: "ValidationResult",
        deadline: datetime,
    ) -> tuple:
        """Run the L2 repair engine; return a directive tuple to the caller.

        Returns:
            ("break", candidate, canonical_val)  → L2 converged; caller breaks to GATE
            ("cancel", ctx)                      → L2 stopped or canonical validate failed; ctx is advanced to the phase-appropriate terminal
            ("fatal", ctx)                       → non-CancelledError exception; ctx is advanced to POSTMORTEM
        Raises:
            asyncio.CancelledError — if engine.run() was cancelled (terminal recorded first)

        The terminal phase chosen for ``cancel``/``fatal`` respects the
        current ctx phase via :meth:`_l2_escape_terminal`:
          • From VALIDATE/VALIDATE_RETRY (pre-apply) → CANCELLED
          • From APPLY/VERIFY (post-apply) → POSTMORTEM
        ``fatal`` classification always routes to POSTMORTEM regardless of
        phase — an engine-level exception is always a forensic event.
        """
        # Snapshot the entry phase up front — we use it for every terminal
        # selection below, even if ctx is later reassigned.
        _entry_phase = ctx.phase
        _escape_terminal = self._l2_escape_terminal(_entry_phase)

        try:
            l2_result = await self._config.repair_engine.run(ctx, best_validation, deadline)
        except asyncio.CancelledError:
            # asyncio cancellation is a forensic event — always POSTMORTEM.
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code="l2_cancelled",
            )
            await self._record_ledger(
                ctx,
                OperationState.FAILED,
                {"reason": "l2_cancelled", "entry_phase": _entry_phase.name},
            )
            raise
        except Exception as exc:
            # Engine-level exceptions are always POSTMORTEM (forensic path).
            logger.error("[Orchestrator] L2 engine error: %s", exc, exc_info=True)
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code=f"l2_fatal:{type(exc).__name__}",
            )
            await self._record_ledger(
                ctx,
                OperationState.FAILED,
                {"reason": f"l2_fatal:{type(exc).__name__}", "entry_phase": _entry_phase.name},
            )
            return ("fatal", ctx)

        if l2_result.terminal == "L2_CONVERGED" and l2_result.candidate is not None:
            # Self-Correction & DPO Alignment Engine (Phase 1): emit the converged repair as a
            # provider-labeled DPO trajectory (rejected=failing candidate, chosen=converged fix) to
            # Reactor-Core for DW-stability training. Fire-and-forget, gated
            # (JARVIS_REPAIR_TRAJECTORY_EMIT_ENABLED, default OFF), fail-soft — never affects APPLY.
            try:
                from backend.core.ouroboros.governance.repair_trajectory_emitter import (
                    RepairTrajectoryEmitter, emitter_enabled, critic_learn_enabled,
                )
                if emitter_enabled() or critic_learn_enabled():
                    # emit() routes: feed the local M1 online critic (learn) and/or stream to Reactor.
                    RepairTrajectoryEmitter().emit(ctx, l2_result)
            except Exception:  # noqa: BLE001 — emission must never break the pipeline
                logger.debug("[Orchestrator] repair-trajectory emit skipped", exc_info=True)

            # Sovereign RSI Flywheel (Phase 4a): synthesize a token-dense DPO
            # preference pair from this converged epistemic-repair trajectory and
            # export it fire-and-forget to reactor-core's ingestor. Distinct from
            # the emitter above: applies the Epistemic Purity gate (drops infra-
            # caused rejections), the golden-ratio gate (both proven states), and
            # AST-symbol isolation (token-dense, not raw full_content). Gated
            # (JARVIS_DPO_SYNTHESIS_ENABLED, default ON), fail-soft — never
            # affects repair/APPLY. Only runs on a resolved repair (chosen present);
            # a yield/pivot has candidate is None and never reaches here.
            try:
                from backend.core.ouroboros.governance.dpo_synthesizer import (
                    RepairTrajectory, synthesize_and_emit,
                )
                _dpo_traj = RepairTrajectory.from_repair(ctx, l2_result)
                if _dpo_traj is not None:
                    synthesize_and_emit(_dpo_traj)
            except Exception:  # noqa: BLE001 — DPO synthesis must never break the pipeline
                logger.debug("[Orchestrator] DPO pair synthesis skipped", exc_info=True)

            # Post-L2 canonical validation is architecturally broken for
            # Python candidates: test_runner.PythonAdapter.run hard-codes
            # sandbox_dir=None (see its docstring at run() line 228-237) so
            # pytest always executes against the LIVE repo_root regardless
            # of the temp sandbox _run_validation wrote the candidate to.
            # For an L2-converged candidate that signal is always stale-
            # false, and the op escapes via l2_canonical_validate_failed
            # despite L2 having actually proven the fix in a real git
            # worktree with the candidate overlaid (RepairSandbox path).
            #
            # Skip the re-run: trust L2's full-repo sandbox convergence and
            # proceed straight to GATE → APPLY. Synthesize a passing
            # ValidationResult explicitly — do NOT reuse ``best_validation``
            # (that is the FAILED L1 validation which triggered L2 in the
            # first place).
            #
            # The architectural follow-up is teaching PythonAdapter to
            # honor sandbox_dir (via a full worktree overlay or
            # PYTHONPATH=repo_root + pytest paths under sandbox), which
            # also fixes the pre-L2 blind spot. Until then this skip is
            # gated by JARVIS_L2_SKIP_CANONICAL_AFTER_CONVERGE (default on)
            # so CI / operators can force the old double-validate path.
            _skip_canonical = os.environ.get(
                "JARVIS_L2_SKIP_CANONICAL_AFTER_CONVERGE", "true"
            ).strip().lower() in {"1", "true", "yes", "on"}

            if _skip_canonical:
                logger.info(
                    "[Orchestrator] L2_CONVERGED op=%s — skipping canonical "
                    "re-validation (PythonAdapter ignores sandbox_dir, L2 "
                    "already validated in git-worktree sandbox). Proceeding "
                    "to GATE → APPLY with L2's proven candidate.",
                    ctx.op_id,
                )
                canonical_val = ValidationResult(
                    passed=True,
                    best_candidate=l2_result.candidate,
                    validation_duration_s=0.0,
                    error=None,
                    failure_class=None,
                    short_summary=(
                        "L2 converged in sandbox; canonical re-run skipped "
                        "(PythonAdapter drops sandbox_dir, see "
                        "test_runner.py:227-237)"
                    ),
                    adapter_names_run=("l2-sandbox",),
                )
                await self._record_ledger(ctx, OperationState.SANDBOXING, {
                    "event": "l2_converged",
                    "iterations": len(l2_result.iterations),
                    "canonical_revalidation": "skipped",
                    "skip_reason": (
                        "PythonAdapter.run hard-codes sandbox_dir=None; "
                        "pytest cwd is always repo_root, ignoring the "
                        "temp sandbox _run_validation wrote the candidate "
                        "to. L2 used RepairSandbox (git worktree) which "
                        "honors the overlay — that signal is trusted."
                    ),
                    **l2_result.summary,
                })
                return ("break", l2_result.candidate, canonical_val)

            # Legacy path — run canonical validation anyway. Retained so
            # operators can force the old behavior via the env flag; will
            # almost always escape to CANCELLED for Python candidates
            # until PythonAdapter is fixed.
            _remaining_s = (deadline - datetime.now(timezone.utc)).total_seconds()
            canonical_val = await self._run_validation(ctx, l2_result.candidate, _remaining_s)
            if canonical_val.passed:
                await self._record_ledger(ctx, OperationState.SANDBOXING, {
                    "event": "l2_converged",
                    "iterations": len(l2_result.iterations),
                    "canonical_revalidation": "passed",
                    **l2_result.summary,
                })
                return ("break", l2_result.candidate, canonical_val)
            else:
                # Phase-aware escape: post-apply → POSTMORTEM, pre-apply → CANCELLED.
                ctx = ctx.advance(
                    _escape_terminal,
                    terminal_reason_code="l2_canonical_validate_failed",
                )
                await self._record_ledger(ctx, OperationState.FAILED, {
                    "reason": "l2_canonical_validate_failed",
                    "entry_phase": _entry_phase.name,
                    "terminal": _escape_terminal.name,
                    **l2_result.summary,
                })
                return ("cancel", ctx)

        elif l2_result.terminal == "L2_PIVOT":
            # ──────────────────────────────────────────────────────────
            # Adaptive Epistemic Feedback Matrix (T3) — Graceful Semantic
            # Pivot. The repair engine has declared this sub-goal's path
            # UNRESOLVABLE (same failure_signature_hash persisted after the
            # temperature degenerated to its floor — verdict from the real
            # ``pivot_verdict``). Rather than dead-stop, surface a
            # ``l2_pivot`` directive carrying the signature + stderr tail so
            # the caller can route to ``decompose_for_block`` at the failure
            # locus (or HITL DLQ if already atomic) WITHOUT touching sibling
            # DAG ops. ctx is left UNADVANCED — the pivot handler owns the
            # terminal. The engine ONLY emits L2_PIVOT when
            # ``epistemic_feedback_enabled()`` is True, so this branch is
            # unreachable (OFF byte-identical) with the feature disabled.
            # ──────────────────────────────────────────────────────────
            _pivot_sig = getattr(l2_result, "failure_signature_hash", "") or ""
            _pivot_tail = getattr(l2_result, "stderr_tail", "") or ""
            await self._record_ledger(ctx, OperationState.SANDBOXING, {
                "event": "l2_pivot",
                "reason": "unresolvable_path",
                "failure_signature_hash": _pivot_sig,
                "entry_phase": _entry_phase.name,
                **l2_result.summary,
            })
            return ("l2_pivot", ctx, _pivot_sig, _pivot_tail)

        elif l2_result.terminal == "L2_STOPPED":
            # ──────────────────────────────────────────────────────────
            # Slice 6 — dynamic budget reconciliation
            # (bt-2026-05-25-174218 root: L2 stopped at 14s of a fresh
            # 120s budget because iter 2 generation produced no candidate;
            # the orchestrator murdered the op with directive='cancel'
            # despite L2 having 106s of unused budget. Operator framing:
            # "ensuring the repair engine gets its full 120 seconds to
            # iterate.")
            #
            # Stop-reason taxonomy:
            #   HARD (genuinely exhausted — re-dispatch would gain nothing):
            #     - timebox_exhausted
            #     - max_iterations_exhausted
            #     - max_validation_runs_exhausted
            #     - deadline_budget_exhausted
            #
            #   SOFT (transient failure — fresh L2 dispatch could converge):
            #     - generate_error:<TypeName>  (single bad provider response)
            #     - empty_candidates           (provider returned no candidates)
            #     - consecutive_provider_timeouts_exhausted:N (provider flake)
            #
            # On SOFT stop, return ("l2_retry", ctx, l2_result.stop_reason)
            # — the caller (VALIDATE_RETRY loop) tracks attempt count
            # against JARVIS_L2_DISPATCH_RETRIES and re-runs the L2
            # dispatch block (which re-reconciles the budget so L2 gets
            # a fresh 120s window each pass). On HARD stop, preserve the
            # pre-Slice-6 cancel behavior verbatim.
            # ──────────────────────────────────────────────────────────
            _l2_hard_stop_prefixes = (
                "timebox_exhausted",
                "max_iterations_exhausted",
                "max_validation_runs_exhausted",
                "deadline_budget_exhausted",
                # a1-brain-20260705-233225 storm root cause: a per-class
                # retry exhaustion IS "genuinely exhausted" by this
                # taxonomy's own definition, but was absent here →
                # classified SOFT → futilely re-dispatched — 120/120
                # identical class_retries_exhausted:env re-dispatches,
                # each burning a fresh 120s timebox. The engine's
                # per-run counter re-derives the same deterministic
                # failure every dispatch.
                "class_retries_exhausted",
            )
            _stop_reason_str = l2_result.stop_reason or ""
            _is_hard_stop = any(
                _stop_reason_str == p or _stop_reason_str.startswith(p + ":")
                for p in _l2_hard_stop_prefixes
            )
            if not _is_hard_stop:
                # Soft stop — leave ctx unadvanced; caller may re-dispatch.
                logger.info(
                    "[Orchestrator] L2 soft stop op=%s stop_reason=%s "
                    "iterations_used=%d — eligible for re-dispatch "
                    "(Slice 6 l2_retry directive)",
                    ctx.op_id,
                    _stop_reason_str or "unknown",
                    len(l2_result.iterations),
                )
                await self._record_ledger(ctx, OperationState.SANDBOXING, {
                    "event": "l2_soft_stop",
                    "stop_reason": _stop_reason_str,
                    "iterations_used": len(l2_result.iterations),
                    "entry_phase": _entry_phase.name,
                    **l2_result.summary,
                })
                return ("l2_retry", ctx, _stop_reason_str)
            # Phase-aware escape: post-apply → POSTMORTEM, pre-apply → CANCELLED.
            ctx = ctx.advance(
                _escape_terminal,
                terminal_reason_code="l2_stopped",
            )
            await self._record_ledger(ctx, OperationState.FAILED, {
                "reason": "l2_stopped",
                "entry_phase": _entry_phase.name,
                "terminal": _escape_terminal.name,
                "stop_reason": l2_result.stop_reason,
                **l2_result.summary,
            })
            return ("cancel", ctx)

        else:  # L2_CONVERGED with no candidate (shouldn't happen in practice)
            # No candidate is an engine invariant violation → POSTMORTEM.
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code="l2_no_candidate",
            )
            await self._record_ledger(ctx, OperationState.FAILED, {
                "reason": "l2_no_candidate",
                "entry_phase": _entry_phase.name,
                **l2_result.summary,
            })
            return ("fatal", ctx)

    async def _live_work_apply_gate(
        self,
        ctx: OperationContext,
        best_candidate: Dict[str, Any],
        *,
        max_wait_override_s: Optional[float] = None,
        scan_root_override: Optional[Path] = None,
    ) -> _LiveWorkGateResult:
        """LiveWorkSensor APPLY gate with bounded defer-wait (Slice 10).

        Scans ``ctx.target_files`` ∪ every file the candidate proposes
        for human activity. Run #20 root cause: an active hit was
        treated as TERMINAL (``human_active_on_target``) even though the
        condition is recoverable — a chaos-dirtied file crosses the
        recency window on its own. When ``JARVIS_APPLY_LIVE_WORK_WAIT_
        ENABLED`` (default true) and the sensor's horizon fits the op's
        remaining pipeline budget (re-clocked from
        ``ctx.pipeline_deadline`` — the same source the VALIDATE retry
        loop re-clocks from), the gate waits exactly that horizon, drops
        the sensor's git cache, and re-runs the FULL scan (a different
        file may be active now). Loops while hits remain and horizons
        stay finite + affordable; a mid-wait re-edit yields a fresh
        horizon that either re-waits or exhausts the budget. NO fixed
        sleep constants — every wait is the sensor-derived horizon.

        Review-round hardening:
        - I1: ONE sensor evaluation per file per iteration
          (``LiveWorkSensor.evaluate``) — active/horizon can never
          disagree across two calls at the window boundary. The residual
          exact-boundary case (age == window is inclusive-active with
          horizon 0) gets ONE bounded immediate rescan; a second
          consecutive zero-horizon-active eval goes terminal.
        - I2: the deadline-less fallback budget is initialized once from
          ``validation_timeout_s`` and shrinks by every slept horizon —
          a constant re-read would never terminate.
        - I4: cumulative sleeps within one invocation are clamped by
          ``JARVIS_APPLY_LIVE_WORK_WAIT_MAX_S``. The seeded default
          ``0.0`` means DERIVE at call time from ``JARVIS_FILE_LOCK_TTL_S``
          (default 300, matching unified_intake_router): a wait may never
          outlive the file lock that serializes writers, or a second op
          can acquire the same file mid-wait; an explicit positive env
          value wins. Bounds worker-pool occupancy too — 3 concurrent
          deferred ops must not put the whole pool to sleep for the full
          pipeline deadline.
        - C1 (TOCTOU): the callers' stale-exploration drift hash check
          runs BEFORE this gate — any wait here makes that snapshot
          stale, so a human edit made mid-wait would be applied over.
          When the gate actually waited, it re-runs the SAME helper
          (``state_drift.should_block_apply``) before clearing; blocking
          → ``drift_stale_files`` for the callers' SAME
          ``state_drift_unreconciled`` terminal shape. No-wait ops skip
          the recheck — today's failure-class ordering is preserved.

        Returns a :class:`_LiveWorkGateResult`. Shared by BOTH APPLY
        paths — the inline orchestrator block and Slice4bRunner (the
        shipping default under ``JARVIS_PHASE_RUNNER_SLICE4B_EXTRACTED``).
        Callers keep the Orange-tier bypass, the ledger/terminal shapes,
        and the fail-open try/except (sensor malfunction must never
        block APPLY).
        """
        from backend.core.ouroboros.governance.live_work_sensor import (
            LiveWorkSensor,
        )
        from backend.core.ouroboros.governance.state_drift import (
            should_block_apply as _should_block_apply,
        )
        # scan_root_override (2026-07-22, soak bt-2026-07-22-050025):
        # "the target IS the tree it scans." Under the isolation-
        # collapsed posture config.project_root IS the sovereignty
        # worktree, so the PROMOTION consult saw the op's OWN 33s-old
        # APPLY write as human activity and refused every promotion.
        # The promoter now passes the git-topology-derived operator
        # root; APPLY-time call sites keep the config root unchanged.
        _lws_root = scan_root_override or self._config.project_root
        _lws = LiveWorkSensor(_lws_root)
        _scan_targets: set[str] = set(ctx.target_files)
        for _cf, _ in self._iter_candidate_files(best_candidate):
            if _cf:
                _scan_targets.add(_cf)
        _wait_enabled = (
            os.environ.get("JARVIS_APPLY_LIVE_WORK_WAIT_ENABLED", "true")
            .strip().lower() in _TRUTHY
        )
        # I4 — cumulative-wait clamp; 0.0/unset/garbage → derive from the
        # file-lock TTL (see docstring for the derivation rationale).
        try:
            _max_wait_s = float(
                os.environ.get("JARVIS_APPLY_LIVE_WORK_WAIT_MAX_S", "0") or 0.0
            )
        except ValueError:
            _max_wait_s = 0.0
        if _max_wait_s <= 0.0:
            try:
                _max_wait_s = float(os.environ.get("JARVIS_FILE_LOCK_TTL_S", "300"))
            except ValueError:
                _max_wait_s = 300.0
        # Slice 11 review P5: a caller-scoped budget (promotion-time consult)
        # decouples the wait from the op's nearly-spent pipeline deadline AND
        # caps the second bounded wait — a verified+committed op must neither
        # die on a momentary human edit (allowance≈0 at 98% progress) nor
        # hold a worker for the full FILE_LOCK_TTL again.
        if max_wait_override_s is not None:
            _max_wait_s = min(_max_wait_s, float(max_wait_override_s))
        # I2 — deadline-less fallback budget, initialized ONCE.
        _fallback_budget_s = self._config.validation_timeout_s
        _slept_total = 0.0
        _zero_horizon_strikes = 0
        while True:
            _active_eval = None
            _hit_file: Optional[str] = None
            for _tf in sorted(_scan_targets):
                _eval = await _lws.evaluate(str(_tf))
                if _eval.active:
                    _active_eval = _eval
                    _hit_file = str(_tf)
                    break
            if _active_eval is None:
                # Slice 11 rider: positive quiet-path evidence. Run-21's
                # audit could only prove "the gate ran" by the ABSENCE of
                # the fail-open DEBUG line — one line makes it affirmative.
                logger.debug(
                    "[LiveWork] quiet — APPLY gate cleared op=%s "
                    "scanned=%d waited=%.1fs",
                    ctx.op_id, len(_scan_targets), _slept_total,
                )
                # Scan is quiet — C1: if we waited, the pre-gate drift
                # snapshot is stale; re-run the SAME check before clearing.
                if _slept_total > 0.0 and getattr(ctx, "generate_file_hashes", None):
                    _block, _stale = _should_block_apply(
                        ctx.generate_file_hashes, self._config.project_root,
                    )
                    if _block:
                        return _LiveWorkGateResult(None, _slept_total, list(_stale))
                return _LiveWorkGateResult(None, _slept_total, None)
            _hit_reason = _active_eval.reason or "human active"
            _active_hit = (_hit_file, _hit_reason)
            if not _wait_enabled:
                # Legacy immediate-terminal (kill switch) — today's path,
                # including its WARNING text, unchanged.
                logger.warning(
                    "[Orchestrator] LiveWorkSensor: human is active on %s (%s) — deferring APPLY [%s]",
                    _hit_file, _hit_reason, ctx.op_id[:12],
                )
                return _LiveWorkGateResult(_active_hit, _slept_total, None)
            _horizon = _active_eval.horizon_s
            # I1 residual — exact window boundary (age == window is
            # inclusive-active with horizon 0): one bounded immediate
            # rescan; the 2-strike counter keeps this from ever busy-
            # looping — a second consecutive zero-horizon-active eval
            # falls through to the terminal check below.
            if _horizon <= 0.0:
                _zero_horizon_strikes += 1
                if _zero_horizon_strikes < 2:
                    _lws.invalidate_cache()
                    continue
            else:
                _zero_horizon_strikes = 0
            # Re-clock remaining budget from the op's pipeline deadline —
            # same source as the VALIDATE retry loop. Deadline-less ops
            # consume the shrinking fallback budget (I2).
            if max_wait_override_s is not None:
                # Caller-scoped budget (promotion consult): dedicated
                # allowance, independent of the exhausted pipeline deadline.
                _remaining_s = float(max_wait_override_s) - _slept_total
            elif ctx.pipeline_deadline is not None:
                _remaining_s = (
                    ctx.pipeline_deadline - datetime.now(tz=timezone.utc)
                ).total_seconds()
            else:
                _remaining_s = _fallback_budget_s - _slept_total
            _wait_allowance = min(_remaining_s, _max_wait_s - _slept_total)
            # inf fails the affordability check naturally (inf > any
            # finite allowance); a persisting zero horizon lands here too.
            if not (0.0 < _horizon <= _wait_allowance):
                logger.warning(
                    "[Orchestrator] LiveWorkSensor: human is active on %s (%s) "
                    "— wait infeasible (horizon=%.0fs remaining=%.0fs "
                    "max_wait=%.0fs slept=%.0fs), failing op [%s]",
                    _hit_file, _hit_reason, _horizon, _remaining_s,
                    _max_wait_s, _slept_total, ctx.op_id[:12],
                )
                return _LiveWorkGateResult(_active_hit, _slept_total, None)
            logger.info(
                "[Orchestrator] LiveWorkSensor: %s active (%s) — waiting %.1fs for quiet [%s]",
                _hit_file, _hit_reason, _horizon, ctx.op_id[:12],
            )
            await asyncio.sleep(_horizon)
            _slept_total += _horizon
            _lws.invalidate_cache()

    @staticmethod
    def _iter_candidate_files(
        candidate: Dict[str, Any],
    ) -> list[Tuple[str, str]]:
        """Return every (file_path, full_content) pair this candidate proposes.

        Multi-file support (Manifesto §6 — coordinated architectural changes):
        when a candidate has a ``files`` list, each entry represents one file
        to apply atomically with the others. Otherwise the primary
        ``file_path`` / ``full_content`` pair is the only one.

        The feature is gated by ``JARVIS_MULTI_FILE_GEN_ENABLED`` (default
        ``true``). When disabled, any ``files`` list is ignored and only the
        primary file is returned — the pipeline behaves exactly as before.

        Ordering:
          • Single-file candidates yield ``[(file_path, full_content)]``.
          • Multi-file candidates yield the entries in ``files`` in order,
            so the first entry is the primary / authoritative file and
            subsequent entries are its coordinated siblings.

        Returns
        -------
        list[tuple[str, str]]
            Non-empty list of ``(file_path, full_content)`` pairs. At minimum,
            contains the primary file.
        """
        primary_path = candidate.get("file_path", "") or ""
        primary_content = candidate.get("full_content", "") or ""

        multi_enabled = (
            os.environ.get("JARVIS_MULTI_FILE_GEN_ENABLED", "true").lower()
            not in ("false", "0", "no", "off")
        )
        files_field = candidate.get("files") if multi_enabled else None
        if isinstance(files_field, list) and files_field:
            pairs: list[Tuple[str, str]] = []
            seen: set[str] = set()
            for entry in files_field:
                if not isinstance(entry, dict):
                    continue
                fp = str(entry.get("file_path", "") or "")
                fc = entry.get("full_content", "") or ""
                if not fp or not isinstance(fc, str):
                    continue
                # De-duplicate — if the primary appears in the list, we
                # don't want to process it twice.
                if fp in seen:
                    continue
                seen.add(fp)
                pairs.append((fp, fc))
            if pairs:
                return pairs

        # Fallback: single-file candidate (legacy path).
        return [(primary_path, primary_content)]

    @staticmethod
    def _validate_config_file_format(
        file_path_str: str, content: str,
    ) -> Optional[str]:
        """Deterministic pre-APPLY format check for common config files.

        Manifesto §6 Iron Gate: deterministic perimeter around agentic
        generation. When the model emits requirements.txt, package.json,
        or similar, a single typo or Unicode corruption would otherwise
        only surface at APPLY (pip install, npm install, etc.). This
        check catches malformed configs BEFORE the change reaches disk.

        Returns ``None`` if the file looks well-formed, or a human-readable
        error string if it does not. Unknown file extensions pass through.

        Parameters
        ----------
        file_path_str : str
            Path or basename of the target file (used for extension dispatch).
        content : str
            Full proposed content.
        """
        if not isinstance(content, str):
            return "config_format: content is not a string"

        _name = Path(file_path_str).name.lower()
        _suffix = Path(file_path_str).suffix.lower()

        # requirements.txt family
        if _name.startswith("requirements") and _suffix == ".txt":
            for _lineno, _raw in enumerate(content.splitlines(), start=1):
                _line = _raw.strip()
                if not _line or _line.startswith("#"):
                    continue
                # Strip inline comments
                if " #" in _line:
                    _line = _line.split(" #", 1)[0].strip()
                # Skip directives (-r, -e, --index-url, etc.)
                if _line.startswith("-"):
                    continue
                # Skip URLs and VCS refs
                if "://" in _line or _line.startswith(("git+", "hg+", "bzr+", "svn+")):
                    continue
                # First token is the distribution name — must start with an
                # ASCII letter/digit and contain only PEP 503 normalizable
                # chars (letters, digits, dash, underscore, dot).
                _first = _line.split(";", 1)[0]  # drop environment marker
                # Split on any version/extras separator
                _pkg_name = ""
                for _ch in _first:
                    if _ch.isalnum() or _ch in "-_.":
                        _pkg_name += _ch
                    else:
                        break
                if not _pkg_name:
                    return (
                        f"requirements.txt line {_lineno}: could not parse "
                        f"package name from {_raw[:60]!r}"
                    )
                # Check for non-ASCII codepoints anywhere in the line (the
                # rapidفuzz class of typo). The global ASCII gate also
                # catches this earlier, but belt-and-suspenders is cheap.
                for _ch in _raw:
                    if ord(_ch) > 127:
                        return (
                            f"requirements.txt line {_lineno}: non-ASCII "
                            f"codepoint U+{ord(_ch):04X} — likely typo "
                            f"in package name {_raw[:60]!r}"
                        )
            return None

        # JSON family
        if _suffix in (".json",) or _name in (
            "package.json", "tsconfig.json", "composer.json",
        ):
            import json as _json
            try:
                _json.loads(content)
            except _json.JSONDecodeError as exc:
                return (
                    f"{_name}: invalid JSON at line {exc.lineno} "
                    f"col {exc.colno}: {exc.msg[:120]}"
                )
            return None

        # YAML (only if PyYAML is available; otherwise pass through)
        if _suffix in (".yml", ".yaml"):
            try:
                import yaml as _yaml  # type: ignore  # noqa: PLC0415
                try:
                    _yaml.safe_load(content)
                except _yaml.YAMLError as exc:  # type: ignore[attr-defined]
                    return f"{_name}: invalid YAML: {str(exc)[:180]}"
            except ImportError:
                pass  # yaml not installed — skip check
            return None

        # Unknown extension — pass through (no gate)
        return None

    async def _run_validation(
        self,
        ctx: OperationContext,
        candidate: Dict[str, Any],
        remaining_s: float,
    ) -> ValidationResult:
        """Validation seam — runs the core pipeline, then applies the Slice 66
        swe_bench_pro advisory test-gate. A ``test`` failure for a benchmark op
        is promoted to passed (the held-out container scoring is authoritative;
        running the held-out tests here would leak them); non-swe_bench ops and
        non-``test`` failures are byte-identical. Wrapping HERE covers all three
        callers — the inline VALIDATE block, the extracted VALIDATERunner, and
        L2 re-validation — so the advisory holds on every validation path."""
        result = await self._run_validation_core(ctx, candidate, remaining_s)
        # Slice 8 §7 — a failed validation MUST be operator-visible.
        # Run #18: a BlockedPathError → fc='security' rejection carried
        # its reason ONLY inside the ValidationResult; nothing logged it,
        # so the blocked path was unrecoverable post-hoc. Single seam:
        # both the inline FSM and the extracted validate_runner call
        # this wrapper.
        try:
            if result is not None and not result.passed:
                logger.warning(
                    "[Validation] FAILED op=%s fc=%s summary=%s error=%s adapters=%s",
                    str(getattr(ctx, "op_id", ""))[:16],
                    result.failure_class or "",
                    (result.short_summary or "")[:200],
                    (result.error or "")[:280],
                    ",".join(result.adapter_names_run or ()),
                )
        except Exception:  # noqa: BLE001 — logging must never perturb VALIDATE
            pass
        return _swe_bench_test_advisory(
            getattr(ctx, "signal_source", "") or "",
            getattr(ctx, "op_id", "") or "",
            candidate, result,
        )

    async def _run_validation_core(
        self,
        ctx: OperationContext,
        candidate: Dict[str, Any],
        remaining_s: float,
    ) -> ValidationResult:
        """Run the full validation pipeline for a single candidate.

        Steps:
          1. AST preflight (fast, no subprocess)
          2. Budget guard (remaining_s <= 0 → budget failure)
          3. Write candidate to temp sandbox dir
          4. validation_runner.run() with op_id continuity
          5. Map MultiAdapterResult → compact ValidationResult

        The full adapter stdout/stderr is recorded in the ledger separately;
        ValidationResult holds only a ≤300-char summary.

        Parameters
        ----------
        ctx:
            Current operation context (used for op_id tracing).
        candidate:
            Candidate dict with ``file`` and ``content`` keys.
        remaining_s:
            Remaining pipeline budget in seconds.

        Returns
        -------
        ValidationResult
            Compact, immutable result suitable for embedding in the context.
        """
        if "execution_graph" in candidate:
            return ValidationResult(
                passed=True,
                best_candidate=candidate,
                validation_duration_s=0.0,
                error=None,
                failure_class=None,
                short_summary="execution graph accepted for L3 scheduling",
                adapter_names_run=(),
            )

        content = candidate.get("full_content", "")
        target_file_str = candidate.get(
            "file_path",
            str(ctx.target_files[0]) if ctx.target_files else "unknown.py",
        )

        # ── Multi-file expansion ────────────────────────────────────────
        # If the candidate has a `files` list (Manifesto §6 coordinated
        # edits), iterate the AST + duplication + config-format gates
        # over every file, not just the primary. The primary remains the
        # anchor for the single-file legacy runner path.
        _all_files = self._iter_candidate_files(candidate)
        _is_multi_file = len(_all_files) > 1

        # Step 1: AST preflight — fast gate, no subprocess (Python files only).
        # Runs on EVERY file in a multi-file candidate, short-circuiting on the
        # first failure so the retry feedback names the offending file.
        for _fp, _fc in _all_files:
            if not _fp.endswith(".py"):
                continue
            _syntax_error = self._ast_preflight(_fc)
            if _syntax_error:
                _scoped = (
                    f"{_fp}: {_syntax_error}" if _is_multi_file else _syntax_error
                )
                return ValidationResult(
                    passed=False,
                    best_candidate=None,
                    validation_duration_s=0.0,
                    error=_scoped,
                    failure_class="build",
                    short_summary=_scoped[:300],
                    adapter_names_run=(),
                )

        # Step 1b: Duplication guard — check for structural duplication (Python only).
        # Runs on each Python file in a multi-file candidate; every file must be
        # clean for the batch to pass.
        for _fp, _fc in _all_files:
            if not _fp.endswith(".py"):
                continue
            try:
                from backend.core.ouroboros.governance.duplication_checker import check_duplication
                _source_content = ""
                _src_path = Path(_fp)
                if not _src_path.is_absolute():
                    _src_path = self._config.project_root / _src_path
                if _src_path.exists():
                    _source_content = _src_path.read_text(encoding="utf-8", errors="replace")
                if _source_content:
                    _dup_error = check_duplication(_fc, _source_content, _fp)
                    if _dup_error is not None:
                        _scoped = (
                            f"{_fp}: {_dup_error}" if _is_multi_file else _dup_error
                        )
                        return ValidationResult(
                            passed=False,
                            best_candidate=None,
                            validation_duration_s=0.0,
                            error=_scoped,
                            failure_class="duplication",
                            short_summary=_scoped[:300],
                            adapter_names_run=(),
                        )
            except Exception as exc:
                logger.debug("[Orchestrator] Duplication check skipped for %s: %s", _fp, exc)

        # Non-code files (docs, configs, etc.) need no test/syntax runner,
        # but structured config files get a format sanity check so that
        # generation-quality failures (malformed deps, bad JSON, etc.) are
        # caught at VALIDATE instead of blowing up post-APPLY. This is the
        # pre-APPLY deterministic gate described in Manifesto §6.
        _RUNNABLE_EXTENSIONS = {".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}

        # Config-format gate runs on every non-runnable file. In a multi-file
        # candidate we must catch a bad requirements.txt even when the primary
        # is a .py — otherwise the pip install would still fail post-APPLY.
        for _fp, _fc in _all_files:
            if Path(_fp).suffix in _RUNNABLE_EXTENSIONS:
                continue
            _cfg_err = self._validate_config_file_format(_fp, _fc)
            if _cfg_err is not None:
                logger.warning(
                    "[Orchestrator] config-format gate rejected %s: %s",
                    _fp, _cfg_err[:160],
                )
                return ValidationResult(
                    passed=False,
                    best_candidate=None,
                    validation_duration_s=0.0,
                    error=_cfg_err,
                    failure_class="build",
                    short_summary=f"config-format: {_cfg_err[:240]}",
                    adapter_names_run=(),
                )

        # If NO file in the candidate is code (.py/.cpp/etc.), there's
        # nothing for the runner to execute — pass through after the
        # format gates above. For mixed candidates (some code + some
        # config) we still run the runner on the code files below.
        if not any(Path(_fp).suffix in _RUNNABLE_EXTENSIONS for _fp, _ in _all_files):
            return ValidationResult(
                passed=True,
                best_candidate=candidate,
                validation_duration_s=0.0,
                error=None,
                failure_class=None,
                short_summary="validation skipped: non-code file (format-checked)",
                adapter_names_run=(),
            )

        # When no runner is configured, skip test execution (dry-run / test mode)
        if self._validation_runner is None:
            return ValidationResult(
                passed=True,
                best_candidate=candidate,
                validation_duration_s=0.0,
                error=None,
                failure_class=None,
                short_summary="validation skipped: no runner configured",
                adapter_names_run=(),
            )

        # Step 2: Budget guard
        if remaining_s <= 0.0:
            return ValidationResult(
                passed=False,
                best_candidate=None,
                validation_duration_s=0.0,
                error="pipeline budget exhausted before validation",
                failure_class="budget",
                short_summary="Budget exhausted",
                adapter_names_run=(),
            )

        # Step 3: Write to temp sandbox
        # For a multi-file candidate we write every file preserving its
        # relative path under the sandbox root, so cross-file imports can
        # resolve during the runner's syntax / test pass. Only code files
        # become `changed_files` for the runner — configs are already
        # validated by the format gate above and don't need execution.
        multi = None
        t0 = time.monotonic()

        # ── Slice 12AE: per-envelope repo_root resolution ──────────
        # Compose the canonical Slice 12AC seam to honour per-envelope
        # repo_root promises (SWE-Bench-Pro TMPDIR worktrees). For
        # NO_PROMISE envelopes (normal in-repo ops), behavior is byte-
        # identical to pre-Slice-12AE — falls back to project_root.
        # For RESOLVED, the per-envelope path becomes the anchor for
        # both original_paths mapping AND a per-op LanguageRouter
        # instance (so the adapter's repo_root is the TMPDIR worktree,
        # not the main JARVIS repo). For REJECTED (promised but
        # invalid/escaped), refuse silent fallback to the shared tree —
        # return ValidationResult with failure_class="infra" so the
        # caller terminates the op cleanly (NO test execution in the
        # wrong tree, NO misleading critiques from the main repo).
        # Closes the bt-2026-05-24-053214 wedge: wiring-validation
        # smoke fixture's VALIDATE ran TestRunner Strategy 3 against
        # /Users/.../JARVIS-AI-Agent (main repo) instead of the
        # promised TMPDIR worktree, producing 1 unrelated critique +
        # L2 cancel + insufficient claims. NEVER hardcodes /tmp.
        from backend.core.ouroboros.governance.operation_advisor import (
            RepoRootPromiseStatus,
            envelope_repo_root_status,
        )
        _ae_status, _ae_resolved_repo_root, _ae_raw_repo_root = (
            envelope_repo_root_status(
                getattr(ctx, "intake_evidence_json", "") or "",
                project_root=self._config.project_root,
            )
        )
        if _ae_status is RepoRootPromiseStatus.REJECTED:
            return ValidationResult(
                passed=False,
                best_candidate=None,
                validation_duration_s=time.monotonic() - t0,
                error=(
                    "validation_runner: envelope-promised repo_root "
                    f"{_ae_raw_repo_root!r} REJECTED by advisor — "
                    "refusing silent fallback to project_root (would "
                    "execute tests in the wrong tree)"
                ),
                failure_class="infra",
                short_summary=(
                    f"slice12ae_repo_root_rejected:{_ae_raw_repo_root[:80]}"
                ),
                adapter_names_run=(),
            )
        # NO_PROMISE → project_root (legacy byte-identical).
        # RESOLVED → the per-envelope (e.g. TMPDIR) path.
        _ae_effective_repo_root: Path = (
            _ae_resolved_repo_root
            if _ae_resolved_repo_root is not None
            else self._config.project_root
        )

        # Per-op LanguageRouter only when the effective root diverges
        # from the boot-time project_root. Composes the same adapter
        # classes (PythonAdapter / CppAdapter) — NO parallel runner
        # implementation. Lazy-imported here so the orchestrator
        # doesn't add a hard top-level dep on test_runner module
        # internals.
        _ae_effective_runner = self._validation_runner
        if _ae_effective_repo_root != self._config.project_root:
            try:
                from backend.core.ouroboros.governance.test_runner import (
                    CppAdapter,
                    LanguageRouter,
                    PythonAdapter,
                )
                _ae_effective_runner = LanguageRouter(
                    repo_root=_ae_effective_repo_root,
                    adapters={
                        "python": PythonAdapter(
                            repo_root=_ae_effective_repo_root,
                        ),
                        "cpp": CppAdapter(
                            repo_root=_ae_effective_repo_root,
                        ),
                    },
                )
                logger.info(
                    "[Slice12AE] per-op LanguageRouter constructed "
                    "for op=%s repo_root=%s (envelope-promised; "
                    "advisor-resolved)",
                    ctx.op_id[:12], _ae_effective_repo_root,
                )
            except Exception:  # noqa: BLE001 — defensive: never
                # break VALIDATE on the per-op router path; fall back
                # to the boot-time runner with the resolved root used
                # only for original_paths mapping below.
                logger.warning(
                    "[Slice12AE] per-op LanguageRouter construction "
                    "raised — falling through to boot-time runner",
                    exc_info=True,
                )

        # ── Slice 9: candidate-tree validation (default ON) ─────────────
        # Run #19 root cause: the legacy path writes candidates into an
        # EMPTY side-tempdir while pytest runs from the main repo root —
        # the candidate is never exercised, so a repair op whose failing
        # test stays red until APPLY structurally cannot pass VALIDATE.
        # Here: materialize a working-tree-faithful full tree
        # (RepairSandbox + Slice-9 dirty overlay), apply the candidate
        # files INTO it, and run a per-op LanguageRouter anchored AT the
        # tree root (the Slice-12AE per-op-router pattern) — discovery
        # finds the sibling tests in the tree and pytest runs from the
        # tree root with the tree's own pytest.ini. Fail-SOFT: any fault
        # falls through to the legacy path (today's behavior).
        # No runnable pre-check needed here: the non-code early return
        # above ("validation skipped: non-code file") fires whenever the
        # candidate has NO runnable file, so this block is only reachable
        # with at least one .py/.cpp/etc. file in _all_files.
        _tree_used = False
        _tree_gate = _candidate_tree_enabled() and _all_files
        # Slice 9 review F7: capture the wall-clock BEFORE tree
        # materialization — every runner grant below (tree AND legacy
        # fallback) is decremented from this anchor, so the ~14s
        # RepairSandbox setup never inflates the budget actually handed
        # to pytest. The pre-setup floor check below deliberately keeps
        # the un-decremented remaining_s (its semantics are "budget on
        # entry", not "budget after setup").
        _v_t0 = time.monotonic()
        if _tree_gate and remaining_s < _validate_tree_min_budget_s():
            # Slice 9 final review (Important): budget-aware skip — tree
            # materialization itself costs real wall-clock (worktree +
            # overlay); when too little budget remains, spending it on
            # setup instead of the actual test run is a net loss. Skip
            # straight to legacy.
            logger.info(
                "[Validation] candidate-tree skipped op=%s reason=low_budget "
                "remaining=%.1fs",
                ctx.op_id[:12], remaining_s,
            )
        elif _tree_gate:
            # Slice 9 review (Important): the fail-soft try below covers
            # ONLY candidate-tree MATERIALIZATION — RepairSandbox entry
            # and the apply_full_content loop (incl. the relative-path
            # resolution inside it). A materialization fault is a real
            # "the tree itself couldn't be built" problem, so falling
            # back to the legacy side-sandbox path is correct there.
            # _tree_runner.run() (actual test execution) is deliberately
            # OUTSIDE this except's reach — see the nested try below,
            # which RETURNS instead of falling back so a BlockedPathError
            # security rejection keeps its fc="security" classification
            # instead of being silently re-run (and possibly not
            # reproduced) by the differently-anchored legacy path.
            try:
                from backend.core.ouroboros.governance.repair_sandbox import (
                    RepairSandbox,
                    _working_tree_mirror_enabled,
                )
                from backend.core.ouroboros.governance.test_runner import (
                    CppAdapter,
                    LanguageRouter,
                    PythonAdapter,
                )
                async with RepairSandbox(
                    _ae_effective_repo_root, max(remaining_s, 30.0),
                ) as _tree:
                    # Slice 9 review F4: honest-baseline gate. When the
                    # working-tree mirror is ON but the sandbox reports a
                    # degraded baseline (overlay refused → "head", or a
                    # mid-copy chimera → "partial"), the tree would
                    # validate the candidate against a tree that is NOT
                    # what TestWatcher observed — treat it exactly like a
                    # materialization failure and fall back to legacy
                    # (_tree_used stays False; the tree is never run).
                    # Mirror OFF is the operator CHOOSING a HEAD baseline
                    # (fidelity=="head" by design) — respected, no
                    # fallback.
                    if (
                        _working_tree_mirror_enabled()
                        and _tree.baseline_fidelity != "working_tree"
                    ):
                        logger.warning(
                            "[Validation] candidate-tree baseline degraded "
                            "op=%s fidelity=%s — falling back to legacy",
                            ctx.op_id[:12], _tree.baseline_fidelity,
                        )
                    else:
                        _troot = _tree.sandbox_root
                        _tree_changed: list = []
                        for _fp, _fc in _all_files:
                            _rel = Path(_fp)
                            if _rel.is_absolute():
                                try:
                                    _rel = _rel.resolve().relative_to(
                                        Path(_ae_effective_repo_root).resolve()
                                    )
                                except ValueError:
                                    raise RuntimeError(
                                        f"candidate path outside repo: {_fp}"
                                    )
                            # Slice 9 — write-escape clamp (Slice-8 final
                            # review I1): a model-chosen ``..``-containing
                            # relative file_path must not write outside the
                            # candidate tree either. Resolve WITHOUT touching
                            # the filesystem (os.path.normpath — pure string
                            # normalization, no FS access, no symlink-
                            # following on not-yet-existing paths) and prove
                            # containment BEFORE apply_full_content lands any
                            # byte. A BlockedPathError raised here is caught
                            # by the materialization fail-soft except below
                            # (no write has landed in the tree) and falls
                            # back to the legacy path, whose own clamp
                            # (below) re-raises the same BlockedPathError for
                            # the same escaping candidate — this time
                            # enclosed by the fc="security" handler. Net
                            # effect on EITHER path: escaping candidate ->
                            # fc="security", no write lands anywhere.
                            _tf = _troot / _rel
                            _resolved_tf = Path(os.path.normpath(str(_tf)))
                            if not str(_resolved_tf).startswith(
                                os.path.normpath(str(_troot)) + os.sep
                            ):
                                raise BlockedPathError(
                                    f"candidate file_path {_fp!r} escapes the "
                                    "VALIDATE candidate tree — write refused "
                                    "(security gate)"
                                )
                            await _tree.apply_full_content(_fc, str(_rel))
                            if _tf.suffix in _RUNNABLE_EXTENSIONS:
                                _tree_changed.append(_tf)

                        if _tree_changed:
                            _tree_runner = LanguageRouter(
                                repo_root=_troot,
                                adapters={
                                    # map_root: the sandbox is a COPY of the
                                    # working tree at a fresh /tmp path, so
                                    # its test index is identical to the
                                    # base's. Without this the cache key is
                                    # unique per candidate and an identical
                                    # 14k-key index is rebuilt every time,
                                    # spending the budget pytest needed.
                                    "python": PythonAdapter(
                                        repo_root=_troot,
                                        map_root=_ae_effective_repo_root,
                                    ),
                                    "cpp": CppAdapter(repo_root=_troot),
                                },
                            )
                            # Slice 9 review F7: the grant must reflect the
                            # ~14s materialization just paid — freshly
                            # decrement from the pre-tree-block anchor
                            # instead of handing pytest the stale pre-setup
                            # remaining_s.
                            _rem_after = max(
                                0.0,
                                remaining_s - (time.monotonic() - _v_t0),
                            )
                            # NOT covered by the materialization fail-soft
                            # except below: a `return` here propagates past
                            # it without triggering that except clause (only
                            # raised exceptions are caught by `except`), so
                            # this is a genuine early return from
                            # _run_validation_core, not a fallback trigger.
                            # The `async with` above still exits normally
                            # (RepairSandbox __aexit__ runs) on the way out.
                            try:
                                multi = await _tree_runner.run(
                                    changed_files=tuple(_tree_changed),
                                    sandbox_dir=_troot,
                                    timeout_budget_s=_rem_after,
                                    op_id=ctx.op_id,
                                    original_paths={
                                        p: _troot / p.relative_to(_troot)
                                        for p in _tree_changed
                                    },
                                )
                                _tree_used = True
                                logger.info(
                                    "[Validation] candidate-tree run op=%s "
                                    "files=%d passed=%s",
                                    ctx.op_id[:12], len(_tree_changed),
                                    getattr(multi, "passed", None),
                                )
                            except BlockedPathError as exc:
                                return _map_tree_run_exception(exc, t0)
                            except Exception as exc:
                                return _map_tree_run_exception(exc, t0)
            except Exception as _tree_exc:  # noqa: BLE001 — fail-soft
                multi = None
                _tree_used = False
                logger.warning(
                    "[Validation] candidate-tree materialization failed "
                    "(%s) — falling back to legacy side-sandbox path op=%s",
                    _tree_exc, ctx.op_id[:12],
                )

        if not _tree_used:
            with tempfile.TemporaryDirectory(prefix="ouroboros_validate_") as sandbox_str:
                sandbox = Path(sandbox_str)
                runner_changed: list[Path] = []
                _original_paths: Dict[Path, Path] = {}
                # Slice 9 — write-escape clamp (Slice-8 final review I1),
                # re-scoped by review F6: the write loop gets its OWN
                # try/except for BlockedPathError ONLY, so the clamp's
                # rejection is caught at the write (before any byte
                # lands) with the same fc="security" classification —
                # while every OTHER write-side fault (disk-full OSError,
                # UnicodeEncodeError, ...) PROPAGATES exactly as it did
                # pre-branch, where the caller's
                # gather(return_exceptions=True) skips the candidate
                # instead of escalating a non-retryable fc="infra"
                # POSTMORTEM.
                try:
                    for _fp, _fc in _all_files:
                        _rel = Path(_fp)
                        if _rel.is_absolute():
                            _sandbox_file = sandbox / _rel.name
                        else:
                            _sandbox_file = sandbox / _rel
                        # Slice 9 — write-escape clamp (Slice-8 final
                        # review I1): a model-chosen ``..``-containing
                        # file_path must not write outside the sandbox.
                        # Resolve WITHOUT touching the filesystem
                        # (os.path.normpath — pure string normalization,
                        # no FS access, no symlink-following on
                        # not-yet-existing paths) and prove containment
                        # BEFORE mkdir/write_text lands any byte.
                        _resolved_target = Path(os.path.normpath(str(_sandbox_file)))
                        if not str(_resolved_target).startswith(
                            os.path.normpath(str(sandbox)) + os.sep
                        ):
                            raise BlockedPathError(
                                f"candidate file_path {_fp!r} escapes the "
                                "VALIDATE sandbox — write refused "
                                "(security gate)"
                            )
                        _sandbox_file.parent.mkdir(parents=True, exist_ok=True)
                        _sandbox_file.write_text(_fc, encoding="utf-8")
                        if _sandbox_file.suffix in _RUNNABLE_EXTENSIONS:
                            runner_changed.append(_sandbox_file)
                            # Slice 12AE: anchor on per-envelope effective
                            # repo_root (TMPDIR worktree for SWE-Bench-Pro
                            # fixtures; project_root for everything else).
                            _original_paths[_sandbox_file] = (
                                _ae_effective_repo_root / _rel
                                if not _rel.is_absolute()
                                else _rel
                            )
                except BlockedPathError as exc:
                    # Security gate rejection → failure_class="security" → CANCELLED (not POSTMORTEM)
                    return ValidationResult(
                        passed=False,
                        best_candidate=None,
                        validation_duration_s=time.monotonic() - t0,
                        error=str(exc),
                        failure_class="security",
                        short_summary=f"BlockedPathError: {str(exc)[:280]}",
                        adapter_names_run=(),
                    )

                if not runner_changed:
                    _primary_rel = Path(target_file_str)
                    _primary_file = sandbox / (_primary_rel.name if _primary_rel.is_absolute() else _primary_rel)
                    runner_changed = [_primary_file]
                    # Slice 12AE: same per-envelope anchor for the
                    # primary-file fallback path.
                    _original_paths[_primary_file] = (
                        _ae_effective_repo_root / _primary_rel
                        if not _primary_rel.is_absolute()
                        else _primary_rel
                    )

                # Step 4: Run LanguageRouter (or any duck-typed runner)
                # Slice 12AE: use the per-op runner when constructed (its
                # repo_root matches the per-envelope TMPDIR worktree);
                # otherwise the boot-time runner (project_root anchored).
                # Slice 9 review F7: freshly decremented grant — this
                # legacy block may follow a failed/degraded candidate-tree
                # materialization that already consumed real wall-clock,
                # so it must not be handed the stale pre-setup remaining_s.
                _rem_after = max(
                    0.0, remaining_s - (time.monotonic() - _v_t0),
                )
                try:
                    multi = await _ae_effective_runner.run(
                        changed_files=tuple(runner_changed),
                        sandbox_dir=sandbox,
                        timeout_budget_s=_rem_after,
                        op_id=ctx.op_id,
                        original_paths=_original_paths,
                    )
                except BlockedPathError as exc:
                    # Security gate rejection → failure_class="security" → CANCELLED (not POSTMORTEM)
                    return ValidationResult(
                        passed=False,
                        best_candidate=None,
                        validation_duration_s=time.monotonic() - t0,
                        error=str(exc),
                        failure_class="security",
                        short_summary=f"BlockedPathError: {str(exc)[:280]}",
                        adapter_names_run=(),
                    )
                except Exception as exc:
                    return ValidationResult(
                        passed=False,
                        best_candidate=None,
                        validation_duration_s=time.monotonic() - t0,
                        error=str(exc),
                        failure_class="infra",
                        short_summary=f"runner exception: {str(exc)[:200]}",
                        adapter_names_run=(),
                    )

        # Step 5: Map to compact ValidationResult (sandbox dir is now cleaned up)
        assert multi is not None
        duration = time.monotonic() - t0
        adapter_names = tuple(r.adapter for r in multi.adapter_results)

        # Phase 2 — high-resolution test-gate telemetry. On FAILURE, extract the
        # SPECIFIC assertion / AST error the candidate died on (the failing node
        # + the ``E ...`` line + the error class) instead of a blind 150-char
        # stdout tail — which for pytest is the "1 failed" epilogue, a count not
        # a cause. The same digest reaches the re-planner (next GENERATE prompt),
        # the cockpit VALIDATE heartbeat, and the GRPO corpus, so a test-gate
        # death now says WHY. On PASS the terse per-adapter tag suffices.
        _digest = None
        if not multi.passed:
            try:
                from backend.core.ouroboros.governance.test_failure_digest import (  # noqa: E501
                    digest_from_adapter_results,
                )
                _digest = digest_from_adapter_results(multi.adapter_results)
            except Exception:  # noqa: BLE001 — telemetry never fails a verdict
                _digest = None

        if _digest:
            short_summary = _digest.headline[:300]
            _failure_detail = _digest.detail
            _failed_tests = _digest.failed_tests
            _test_total = _digest.test_total
            _test_failed = _digest.test_failed
        else:
            summary_parts = []
            for r in multi.adapter_results:
                tail = (r.test_result.stdout or "")[-150:] if r.test_result else ""
                summary_parts.append(
                    f"[{r.adapter}:{'PASS' if r.passed else 'FAIL'}] {tail}"
                )
            short_summary = " | ".join(summary_parts)[:300]
            _failure_detail = ""
            _failed_tests = ()
            _test_total = _test_failed = 0

        return ValidationResult(
            passed=multi.passed,
            best_candidate=candidate if multi.passed else None,
            validation_duration_s=duration,
            error=None if multi.passed else f"validation failed: {multi.failure_class}",
            failure_class=None if multi.passed else multi.failure_class,
            short_summary=short_summary,
            adapter_names_run=adapter_names,
            failure_detail=_failure_detail,
            failed_tests=_failed_tests,
            test_total=_test_total,
            test_failed=_test_failed,
        )

    def _swe_bench_write_root(self, ctx: OperationContext) -> Optional[Path]:
        """Slice 64 — the APPLY write root for a swe_bench_pro op.

        Returns the VALIDATED envelope ``repo_root`` (the prepared per-problem
        worktree — a cloned benchmark repo) so APPLY writes there instead of the
        JARVIS ``JARVIS_AUTO_COMMIT_WORKSPACE`` (the bt-2026-06-02-081453 mis-
        route where ``src/Markdown.ts`` was rebased onto a JARVIS worktree and
        APPLY failed). Composes the SAME ``guard_envelope_repo_root`` anchor-
        validator the Advisor uses (operation_advisor) — no raw-evidence trust.
        Returns ``None`` for every non-swe_bench op (legacy write resolution
        unchanged) and on any error (never breaks APPLY)."""
        try:
            if getattr(ctx, "signal_source", "") != "swe_bench_pro":
                return None
            from backend.core.ouroboros.governance.operation_advisor import (
                guard_envelope_repo_root,
            )
            return guard_envelope_repo_root(
                ctx.intake_evidence_json,
                project_root=self._config.project_root,
            )
        except Exception:  # noqa: BLE001 — APPLY must never break on this
            logger.debug(
                "[Orchestrator] _swe_bench_write_root resolution raised "
                "for op=%s — falling back to legacy write root",
                getattr(ctx, "op_id", "?"), exc_info=True,
            )
            return None

    def _build_change_request(
        self, ctx: OperationContext, candidate: Dict[str, Any]
    ) -> ChangeRequest:
        """Build a ChangeRequest from the context and best candidate.

        Parameters
        ----------
        ctx:
            The current operation context.
        candidate:
            The validated candidate dict with ``file`` and ``content`` keys.
        """
        target_file = Path(
            candidate.get("file_path", str(ctx.target_files[0] if ctx.target_files else "unknown.py"))
        )
        proposed_content = candidate.get("full_content", "")

        profile = self._build_profile(ctx)

        return ChangeRequest(
            goal=ctx.description,
            target_file=target_file,
            proposed_content=proposed_content,
            profile=profile,
            op_id=ctx.op_id,
            write_root=self._swe_bench_write_root(ctx),
        )

    async def _discover_tests_for_gate(self, sut_path: Path) -> List[Path]:
        """Discover pytest files scoped to one SUT for the MutationGate.

        Matches Session-W style fan-out (``tests/test_<stem>*.py``) via
        rglob under the project root's ``tests/`` dir. Returned paths
        are absolute so the mutation runner sees stable targets even
        when the gate is called from a non-project cwd.

        fs-hot-tier Batch 3 (row 15): the rglob scan is dispatched off
        the asyncio loop via ``cooperative_fs_io.offload`` (thread pool
        — a scoped ``tests/`` rglob is IO-bound/syscall-dominated, not
        CPU-after-scan). Fail-soft: an ``OffloadError`` degrades to the
        same ``[]`` the prior synchronous fail-soft implicitly gave for
        a missing/unreadable tree — never raises into the GATE phase.
        """
        stem = sut_path.stem
        tests_dir = self._config.project_root / "tests"
        if not tests_dir.is_dir():
            return []
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            offload,
            is_offload_error,
        )
        result = await offload(
            _discover_tests_for_gate_worker, str(tests_dir), stem,
        )
        if is_offload_error(result):
            logger.debug(
                "[Orchestrator] _discover_tests_for_gate offload failed "
                "for stem=%s — degrading to empty list",
                stem,
            )
            return []
        return [Path(p) for p in result]

    async def _apply_multi_file_candidate(
        self,
        ctx: OperationContext,
        candidate: Dict[str, Any],
        files: list[Tuple[str, str]],
        snapshots: Dict[str, str],
    ) -> ChangeResult:
        """Apply a multi-file candidate atomically.

        Manifesto §6 boundary rule: the agentic layer emits the coordinated
        edits, the deterministic layer applies them. We iterate through the
        files, running each through the existing single-file ChangeEngine
        pipeline (which keeps every per-file gate: risk classification,
        governance lock, verify hook, rollback). If any file fails, every
        previously-applied file in the batch is restored from its pre-apply
        snapshot so the on-disk state matches the pre-batch state.

        This helper explicitly does NOT re-implement the 8-phase pipeline —
        it composes the existing engine so the pipeline's guarantees still
        hold for each file, and adds batch-level rollback on top.

        Parameters
        ----------
        ctx:
            Current operation context.
        candidate:
            The validated best candidate, must contain a non-empty ``files`` list.
        files:
            The ``(file_path, full_content)`` pairs produced by
            ``_iter_candidate_files``. At least one entry (guaranteed by caller).
        snapshots:
            Pre-apply snapshots keyed by file path, used to restore files
            if the batch fails partway through.

        Returns
        -------
        ChangeResult
            A single aggregated result. ``success=True`` only when every file
            applied cleanly. ``rolled_back`` reflects whether any restoration
            was attempted on failure.
        """
        profile = self._build_profile(ctx)
        applied: list[Tuple[str, Path]] = []   # (rel_path, abs_path) of successfully applied files
        last_phase_reached: ChangePhase = ChangePhase.PLAN
        last_risk_tier: Optional[RiskTier] = None
        last_error: Optional[str] = None

        # Bounded Fan-Out pre-write rejection (Transactional Op-Scoping,
        # 2026-07-22): under an armed flush-freeze, a transaction wider
        # than the fan-out ceiling is REJECTED before its first byte —
        # never started, never half-applied, never rollback-churned. The
        # per-file engine consult would deny file N+1 anyway; failing the
        # batch here keeps 2PC from beginning a structurally doomed
        # transaction. Fail-soft: any consult error skips the guard.
        try:
            from backend.core.ouroboros.governance import (
                apply_flush_freeze as _aff,
            )
            if (
                _aff.flush_freeze_enabled()
                and len(files) > _aff.max_files_per_op()
            ):
                logger.warning(
                    "[Orchestrator] Multi-file apply REJECTED pre-write: "
                    "%d files exceeds the op-scoped fan-out ceiling %d "
                    "(op=%s) — %s",
                    len(files), _aff.max_files_per_op(), ctx.op_id,
                    _aff.FANOUT_DENIAL_REASON,
                )
                return ChangeResult(
                    op_id=ctx.op_id,
                    success=False,
                    phase_reached=ChangePhase.PLAN,
                    rolled_back=False,
                    error=_aff.FANOUT_DENIAL_REASON,
                )
        except Exception:  # noqa: BLE001 — guard is protective, never fatal
            logger.debug(
                "[Orchestrator] fan-out pre-write guard skipped",
                exc_info=True,
            )

        for idx, (fp, fc) in enumerate(files):
            # Build an absolute target path anchored at the project root.
            _rel = Path(fp)
            _abs = _rel if _rel.is_absolute() else (self._config.project_root / _rel)

            _per_file_request = ChangeRequest(
                goal=f"{ctx.description} [multi-file {idx + 1}/{len(files)}: {fp}]",
                target_file=_abs,
                proposed_content=fc,
                profile=profile,
                op_id=f"{ctx.op_id}::{idx:02d}",
                write_root=self._swe_bench_write_root(ctx),  # Slice 64
            )

            try:
                per_result = await self._stack.change_engine.execute(_per_file_request)
            except Exception as exc:
                logger.error(
                    "[Orchestrator] Multi-file apply: file %d/%d (%s) raised: %s",
                    idx + 1, len(files), fp, exc,
                )
                per_result = ChangeResult(
                    op_id=_per_file_request.op_id or ctx.op_id,
                    success=False,
                    phase_reached=last_phase_reached,
                    rolled_back=False,
                    error=f"change_engine_raise: {exc}",
                )

            last_phase_reached = per_result.phase_reached
            if per_result.risk_tier is not None:
                last_risk_tier = per_result.risk_tier

            if per_result.success:
                applied.append((fp, _abs))
                continue

            # ── Failure — roll back every previously-applied file ──
            last_error = (
                f"multi_file_apply failed on {fp} "
                f"(file {idx + 1}/{len(files)}): {per_result.error or 'unknown'}"
            )
            logger.error("[Orchestrator] %s", last_error)
            rolled_back_any = False
            for done_fp, done_abs in applied:
                if done_fp in snapshots:
                    try:
                        done_abs.parent.mkdir(parents=True, exist_ok=True)
                        done_abs.write_text(snapshots[done_fp], encoding="utf-8")
                        rolled_back_any = True
                        logger.info(
                            "[Orchestrator] Multi-file rollback: restored %s", done_fp,
                        )
                    except OSError as _restore_exc:
                        logger.error(
                            "[Orchestrator] Multi-file rollback FAILED for %s: %s",
                            done_fp, _restore_exc,
                        )
                else:
                    # No snapshot = file was new in this batch; unlink to undo creation.
                    try:
                        if done_abs.exists():
                            done_abs.unlink()
                            rolled_back_any = True
                            logger.info(
                                "[Orchestrator] Multi-file rollback: removed new file %s", done_fp,
                            )
                    except OSError as _unlink_exc:
                        logger.error(
                            "[Orchestrator] Multi-file rollback unlink FAILED for %s: %s",
                            done_fp, _unlink_exc,
                        )

            await self._record_ledger(ctx, OperationState.APPLYING, {
                "event": "multi_file_rollback",
                "failed_file": fp,
                "failed_index": idx,
                "total_files": len(files),
                "rolled_back_count": len(applied),
                "rolled_back_any": rolled_back_any,
            })

            return ChangeResult(
                op_id=ctx.op_id,
                success=False,
                phase_reached=last_phase_reached,
                risk_tier=last_risk_tier,
                rolled_back=rolled_back_any or per_result.rolled_back,
                error=last_error,
            )

        # All files applied cleanly — return aggregated success.
        await self._record_ledger(ctx, OperationState.APPLYING, {
            "event": "multi_file_apply_complete",
            "file_count": len(files),
            "files": [fp for fp, _ in files],
        })
        return ChangeResult(
            op_id=ctx.op_id,
            success=True,
            phase_reached=last_phase_reached,
            risk_tier=last_risk_tier,
            rolled_back=False,
            error=None,
        )

    async def _execute_saga_apply(
        self,
        ctx: OperationContext,
        best_candidate: dict,
    ) -> OperationContext:
        """Execute multi-repo saga apply + three-tier verify.

        Selected when ctx.cross_repo is True. Single-repo path is unchanged.
        """
        # Build patch_map from best_candidate["patches"] or fall back to empty per-repo patches
        patch_map: Dict[str, RepoPatch] = {}
        if best_candidate and "patches" in best_candidate:
            patch_map = best_candidate["patches"]
        else:
            for repo in ctx.repo_scope:
                patch_map[repo] = RepoPatch(repo=repo, files=())

        # Resolve per-repo filesystem roots from registry (fallback to project_root)
        repo_roots = self._config.resolve_repo_roots(
            repo_scope=ctx.repo_scope,
            op_id=ctx.op_id,
        )

        strategy = SagaApplyStrategy(
            repo_roots=repo_roots,
            ledger=self._stack.ledger,
            message_bus=getattr(self._config, "message_bus", None),
            branch_isolation=os.environ.get(
                "JARVIS_SAGA_BRANCH_ISOLATION", "false"
            ).lower() in ("1", "true", "yes"),
            keep_failed_saga_branches=os.environ.get(
                "JARVIS_SAGA_KEEP_FORENSICS_BRANCHES", "true"
            ).lower() in ("1", "true", "yes"),
        )
        _t_saga = time.monotonic()
        apply_result = await strategy.execute(ctx, patch_map)

        if apply_result.terminal_state == SagaTerminalState.SAGA_ABORTED:
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code=apply_result.reason_code,
            )
            await self._record_ledger(
                ctx,
                OperationState.FAILED,
                {"reason": apply_result.reason_code, "saga_id": apply_result.saga_id},
            )
            self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga)
            await self._publish_outcome(ctx, OperationState.FAILED, apply_result.reason_code)
            return ctx

        if apply_result.terminal_state == SagaTerminalState.SAGA_APPLY_COMPLETED:
            verifier = CrossRepoVerifier(
                repo_roots=repo_roots,
            )
            verify_result = await verifier.verify(
                repo_scope=ctx.repo_scope,
                patch_map=patch_map,
                dependency_edges=ctx.dependency_edges,
            )

            if not verify_result.passed:
                comp_ok = await strategy.compensate_after_verify_failure(
                    saga_result=apply_result,
                    patch_map=patch_map,
                    op_id=ctx.op_id,
                    reason_code=verify_result.reason_code,
                )
                # Emit SAGA_FAILED to bus if available
                _bus = getattr(strategy, "_bus", None)
                if _bus is not None:
                    try:
                        from backend.core.ouroboros.governance.autonomy.saga_messages import (
                            SagaMessage, SagaMessageType, MessagePriority,
                        )
                        _bus.send(SagaMessage(
                            message_type=SagaMessageType.SAGA_FAILED,
                            saga_id=apply_result.saga_id,
                            correlation_id=apply_result.saga_id,
                            priority=MessagePriority.HIGH,
                            payload={
                                "schema_version": "1.0",
                                "op_id": ctx.op_id,
                                "saga_id": apply_result.saga_id,
                                "reason_code": "verify_failed",
                                "failed_phase": "VERIFY",
                            },
                        ))
                    except Exception:
                        pass
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code=verify_result.reason_code,
                    rollback_occurred=comp_ok,
                )
                await self._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {
                        "reason": verify_result.reason_code,
                        "saga_id": apply_result.saga_id,
                        "compensated": comp_ok,
                    },
                )
                self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga)
                await self._publish_outcome(ctx, OperationState.FAILED, verify_result.reason_code)
                return ctx

            # ---- Cross-Repo Mutator G2: air-gapped Trinity sandbox gate ----
            # AFTER the existing CrossRepoVerifier (structure/compilation/
            # integration) passes and BEFORE promoting the ephemeral branches,
            # run the air-gapped Trinity integration sandbox. A FRACTURE (broken
            # Body<->Mind<->Nerves handshake; the FRACTURE yield is emitted
            # inside the gate) routes into the SAME compensating rollback the
            # verify-failure path uses — the op is sealed/terminal, never
            # half-applied across repos. Gated + fail-CLOSED (gate returns a
            # FRACTURE on any uncertainty); no-op PASS when the master switch is
            # OFF (byte-identical legacy).
            try:
                from backend.core.ouroboros.governance.cross_repo_master_flag import (
                    cross_repo_mutation_enabled,
                )
                _g2_armed = cross_repo_mutation_enabled()
            except Exception:  # noqa: BLE001
                _g2_armed = False
            if _g2_armed:
                from backend.core.ouroboros.governance.multi_repo.cross_repo_wiring import (
                    run_apply_sandbox_gate,
                )
                _sandbox_root = str(
                    (repo_roots.get(ctx.primary_repo) if isinstance(repo_roots, dict) else None)
                    or self._config.project_root
                )
                _verdict = await run_apply_sandbox_gate(
                    ctx, candidate_root=_sandbox_root, op_id=ctx.op_id,
                )
                if getattr(_verdict, "fracture", False):
                    logger.warning(
                        "[Orchestrator] Cross-repo Trinity sandbox FRACTURE "
                        "op=%s reason=%s — routing to compensating rollback",
                        ctx.op_id, getattr(_verdict, "reason", "?"),
                    )
                    comp_ok = await strategy.compensate_after_verify_failure(
                        saga_result=apply_result,
                        patch_map=patch_map,
                        op_id=ctx.op_id,
                        reason_code="cross_repo_fracture",
                    )
                    ctx = ctx.advance(
                        OperationPhase.POSTMORTEM,
                        terminal_reason_code="cross_repo_fracture",
                        rollback_occurred=comp_ok,
                    )
                    await self._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {
                            "reason": "cross_repo_fracture",
                            "saga_id": apply_result.saga_id,
                            "compensated": comp_ok,
                            "sandbox_reason": getattr(_verdict, "reason", ""),
                        },
                    )
                    self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga)
                    await self._publish_outcome(
                        ctx, OperationState.FAILED, "cross_repo_fracture"
                    )
                    return ctx

            # B+ mode: promote ephemeral branches before declaring success
            promote_state, promoted_shas = await strategy.promote_all(
                apply_order=list(ctx.repo_scope),
                saga_id=apply_result.saga_id,
                op_id=ctx.op_id,
            )

            if promote_state == SagaTerminalState.SAGA_PARTIAL_PROMOTE:
                try:
                    await self._stack.comm.emit_postmortem(
                        op_id=ctx.op_id,
                        root_cause="saga_partial_promote",
                        failed_phase="PROMOTE",
                        next_safe_action="human_intervention_required",
                    )
                except Exception:
                    pass
                try:
                    await self._stack.controller.pause(scope="cross_repo_saga")
                except TypeError:
                    await self._stack.controller.pause()
                except Exception:
                    logger.exception(
                        "[Orchestrator] controller.pause() failed for partial promote %s",
                        ctx.op_id,
                    )
                ctx = ctx.advance(
                    OperationPhase.POSTMORTEM,
                    terminal_reason_code="saga_partial_promote",
                )
                await self._record_ledger(
                    ctx, OperationState.FAILED,
                    {"reason": "saga_partial_promote", "saga_id": apply_result.saga_id, "promoted_repos": promoted_shas},
                )
                self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga)
                await self._publish_outcome(ctx, OperationState.FAILED, "saga_partial_promote")
                return ctx

            # SAGA_SUCCEEDED
            ctx = ctx.advance(OperationPhase.VERIFY)
            await self._record_ledger(
                ctx,
                OperationState.APPLIED,
                {"saga_id": apply_result.saga_id},
            )
            ctx = await self._run_benchmark(ctx, [])
            ctx = ctx.advance(OperationPhase.COMPLETE, terminal_reason_code="complete")
            self._record_canary_for_ctx(ctx, True, time.monotonic() - _t_saga)
            await self._publish_outcome(ctx, OperationState.APPLIED)
            await self._persist_performance_record(ctx)
            try:
                saga_applied: Sequence[Path] = [
                    (Path(self._config.repo_registry.get(repo).local_path) / rel_path).resolve()
                    for repo, patch in patch_map.items()
                    for rel_path, _ in patch.new_content
                ] if self._config.repo_registry is not None else []
            except Exception:
                saga_applied = []
            await self._oracle_incremental_update(saga_applied)
            return ctx

        if apply_result.terminal_state == SagaTerminalState.SAGA_STUCK:
            # Compensation failed: data may be inconsistent — emit postmortem
            try:
                await self._stack.comm.emit_postmortem(
                    op_id=ctx.op_id,
                    root_cause="saga_stuck",
                    failed_phase="APPLY",
                    next_safe_action="human_intervention_required",
                )
            except Exception:
                pass
            # Halt intake: dirty state requires human review before next op
            try:
                await self._stack.controller.pause()
            except Exception:
                logger.exception(
                    "[Orchestrator] controller.pause() failed for stuck saga %s; "
                    "manual pause may be required",
                    ctx.op_id,
                )
            else:
                logger.warning(
                    "[Orchestrator] Safe pause triggered after SAGA_STUCK on %s",
                    ctx.op_id,
                )
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code="saga_stuck",
            )
            await self._record_ledger(
                ctx,
                OperationState.FAILED,
                {"reason": apply_result.reason_code, "saga_id": apply_result.saga_id},
            )
            self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga)
            await self._publish_outcome(ctx, OperationState.FAILED, "saga_stuck")
            return ctx

        # SAGA_ROLLED_BACK: clean rollback — change not applied, system is clean
        # Advance to CANCELLED so the returned context is terminal and explicit.
        ctx = ctx.advance(
            OperationPhase.CANCELLED,
            terminal_reason_code=apply_result.reason_code,
            rollback_occurred=True,
        )
        await self._record_ledger(
            ctx,
            OperationState.FAILED,
            {"reason": apply_result.reason_code, "saga_id": apply_result.saga_id, "rolled_back": True},
        )
        self._record_canary_for_ctx(ctx, False, time.monotonic() - _t_saga, rolled_back=True)
        await self._publish_outcome(ctx, OperationState.FAILED, apply_result.reason_code)
        return ctx

    async def _publish_candidate_verdict(
        self,
        ctx: Any,
        *,
        candidate: Dict[str, Any],
        validation: Any,
        duration_s: float,
        generation: Any,
        exploration_first_ok: Any = None,
        exploration_count: Any = None,
    ) -> None:
        """Publish ONE sibling's VALIDATE verdict to both consumers.

        There are two of them and they are not interchangeable: the LEDGER
        (audit, replay, the report) and the TRAJECTORY RECORDER (training
        corpus). They must not drift apart, because a verdict that reaches
        only the ledger is invisible to training and a verdict that reaches
        only the recorder is unauditable.

        This exists because they DID drift. The VALIDATE block was extracted
        into ``phase_runners/validate_runner.py`` as a near-verbatim copy,
        and the copy kept the ledger write and dropped the recorder call.
        It is dead today only because
        ``JARVIS_PHASE_RUNNER_VALIDATE_EXTRACTED`` defaults false — flip that
        flag and per-candidate verdicts stop reaching the corpus silently,
        with every row falling back to the op-level outcome and every
        sibling in a group scoring identically. One publisher, two call
        sites, so the extraction can never lose half of it again.

        Never raises: a telemetry write must not be able to fail a
        validation that already succeeded.
        """
        try:
            await self._record_ledger(ctx, OperationState.GATING, {
                "event": "candidate_validated",
                "candidate_id": candidate.get("candidate_id", "unknown"),
                "candidate_hash": candidate.get("candidate_hash", ""),
                "validation_outcome": "pass" if validation.passed else "fail",
                "failure_class": validation.failure_class,
                # Phase 2 — the SPECIFIC cause the candidate died on, so training
                # (GRPO) learns WHY a patch failed the gate, not merely that it
                # did. Bounded at the digest; empty on a pass.
                "failure_detail": str(
                    getattr(validation, "failure_detail", "") or "",
                ),
                "failed_tests": list(
                    getattr(validation, "failed_tests", ()) or [],
                ),
                "test_total": int(getattr(validation, "test_total", 0) or 0),
                "test_failed": int(getattr(validation, "test_failed", 0) or 0),
                "duration_s": round(float(duration_s), 3),
                "provider": getattr(generation, "provider_name", ""),
                "model": getattr(generation, "model_id", ""),
                "exploration_first_ok": exploration_first_ok,
                "exploration_count": exploration_count,
            })
        except Exception:  # noqa: BLE001 — audit write is best-effort
            logger.debug("[Orchestrator] candidate ledger write degraded",
                         exc_info=True)

        # Feed the measured duration back so the reserve LEARNS. Without
        # this the EWMA never moves and the "adaptive" floor is a constant
        # in disguise. Placed in the publisher because this is the one seam
        # every validated candidate passes through, on both call paths.
        try:
            from backend.core.ouroboros.governance.adaptive_gen_budget import (  # noqa: E501
                observe_validation_duration,
            )
            observe_validation_duration(
                str(getattr(ctx, "provider_route", "") or ""),
                float(duration_s),
            )
        except Exception:  # noqa: BLE001 — telemetry must never fail a verdict
            logger.debug("[ValidationReserve] observe skipped", exc_info=True)

        # The per-candidate verdict is the ONLY thing that separates siblings
        # of one prompt. Without it every sibling inherits the op's single
        # terminal verdict, scores identically, and the group yields zero
        # training pairs -- so n>1 generation would buy N x the wall clock
        # and no data. Emitting HERE rather than in a second validation pass
        # is why n>1 costs only generation: the validation already happened.
        try:
            from backend.core.ouroboros.governance.observability.trajectory_recorder import (  # noqa: E501
                record_candidate_verdict as _traj_cand_verdict,
            )
            _traj_cand_verdict(
                op_id=ctx.op_id,
                candidate_hash=str(candidate.get("candidate_hash", "") or ""),
                passed=bool(validation.passed),
                failure_class=str(validation.failure_class or ""),
                # Phase 2 — the specific assertion/AST cause into the GRPO
                # corpus alongside the failure_class category.
                failure_detail=str(
                    getattr(validation, "failure_detail", "") or "",
                ),
            )
        except Exception:  # noqa: BLE001 — fail-open
            logger.debug(
                "[TrajectoryRecorder] candidate verdict emit degraded",
                exc_info=True,
            )

    async def _record_ledger(
        self,
        ctx: OperationContext,
        state: OperationState,
        data: Dict[str, Any],
    ) -> None:
        """Append a ledger entry, logging errors without raising.

        Awaits the ledger append inline so that entries are committed
        before the pipeline continues.  Errors are logged but never
        propagate -- ledger failures must not crash the pipeline.

        B.2.0.5 — operation-FSM lifecycle SSE publish: after a
        successful ledger.append() AND when ``state`` is one of the
        closed TERMINAL_OPERATION_STATES, fan out an
        ``operation_terminal`` SSE event via the canonical broker.
        The publish is best-effort + bounded payload + NEVER raises
        into this function — per operator binding "publish_* cannot
        block ledger". Idempotency rides on the ledger's existing
        (op_id, state) dedup key: ledger.append() returns False on
        duplicate, which suppresses the publish. Master flag
        ``JARVIS_OP_LIFECYCLE_SSE_ENABLED`` (§33.1 default-FALSE) —
        when off, the publish is a no-op and behavior is byte-
        identical to pre-B.2.0.5.
        """
        entry = LedgerEntry(
            op_id=ctx.op_id,
            state=state,
            data=data,
        )
        written = False
        try:
            written = await self._stack.ledger.append(entry)
        except Exception as exc:
            logger.error(
                "Ledger append failed: op_id=%s state=%s error=%s",
                entry.op_id,
                entry.state.value,
                exc,
            )
        # B.2.0.5 single-seam SSE publish — composed via lazy import to
        # avoid an orchestrator → observability hard-dep at module load
        # (mirrors the same pattern Slice 2 `task_tool.py` uses). The
        # nested try/except is load-bearing: publish_operation_terminal
        # already documents NEVER-raise, but this defensive wrapper
        # honors the operator binding verbatim ("never raise into
        # _record_ledger — swallow + DEBUG").
        # Slice 74 probe — did the TERMINAL ledger write land (written=True →
        # publish fires) or get DEDUPED (written=False → publish skipped → the
        # eval never wakes, falls back to the 25-min ledger scan)? Captures the
        # `written` boolean for terminal states. Zero-risk; remove after diag.
        try:
            # Canonical terminal set is lowercase {applied, rolled_back, failed,
            # blocked} — 'applied' is the SUCCESS terminal (there is no
            # 'completed'). Match the real set so the success path is observed.
            _s74_sv = str(getattr(state, "value", state)).lower()
            if _s74_sv in ("applied", "rolled_back", "failed", "blocked"):
                logger.info(
                    "[Slice74Probe] LEDGER_TERMINAL op_id=%s state=%s written=%s",
                    getattr(ctx, "op_id", "?"), _s74_sv, written,
                )
        except Exception:  # noqa: BLE001
            pass
        # ── Task 4 — Cognitive Persistence terminal-time recorder ──
        # Distills this op's tool-execution failures (+ the terminal
        # failure reason, when any) into cross-session CognitiveExperience
        # rows via the write-side of the Bi-Directional Cognitive
        # Persistence arc. Piggybacks on the Slice74Probe terminal-state
        # classification (_s74_sv) above — this IS the single chokepoint
        # every terminal ledger write passes through. In practice this
        # fires on applied/failed/blocked: OperationState.ROLLED_BACK
        # never reaches _record_ledger (rollback rows are appended
        # directly in change_engine.py's own ledger.append call) — the
        # "rolled_back" branch below stays in the state guard purely as
        # a future-proofing no-op. Gated on `written` (computed above)
        # to match the SessionRecorder dedup semantics a few lines up —
        # a replayed/deduped ledger write must never double-count
        # experience occurrences. Fully self-contained try/except: NEVER
        # raises into _record_ledger, DEBUG-logs and continues.
        # Fire-and-forget background write, capped at 20 experiences/op
        # inside the recorder. Authority-free — never influences
        # GATE/APPLY. No-op when JARVIS_COGNITIVE_PERSISTENCE_ENABLED is
        # unset (default False).
        try:
            from backend.core.ouroboros.governance import (
                cognitive_persistence as _cogp,
            )
            if _cogp.is_enabled() and written:
                _cogp_sv = str(getattr(state, "value", state)).lower()
                if _cogp_sv in ("applied", "rolled_back", "failed", "blocked"):
                    _cogp_gen = getattr(ctx, "generation", None)
                    # No resolved model-config object (model_name + num_ctx)
                    # is threaded this far up the FSM — the closest in-scope
                    # signal is the provider's reported model_id off the
                    # op's own GenerationResult. Never hardcoded; falls back
                    # to the documented "unknown" footprint when absent.
                    _cogp_model = getattr(_cogp_gen, "model_id", "") or "unknown"
                    _cogp_footprint = _cogp.cognitive_footprint(_cogp_model, None)
                    _cogp_records = list(
                        getattr(_cogp_gen, "tool_execution_records", ()) or ()
                    )
                    _cogp_reason = None
                    if _cogp_sv != "applied":
                        _cogp_reason = (
                            str(getattr(ctx, "terminal_reason_code", "") or "")
                            or str((data or {}).get("reason", "") or "")
                            or str((data or {}).get("reason_code", "") or "")
                            or _cogp_sv
                        )
                    # Most terminal paths have already advanced ctx.phase to
                    # POSTMORTEM/CANCELLED by the time _record_ledger runs —
                    # ctx.phase.name mislabels the originating phase. The
                    # codebase-wide convention (10+ call sites) for the true
                    # originating phase is data["entry_phase"]; prefer it and
                    # only fall back to ctx.phase.name when it's absent.
                    _cogp_phase = ""
                    try:
                        _cogp_phase = (
                            str((data or {}).get("entry_phase") or "")
                            if isinstance(data, dict) else ""
                        )
                    except Exception:
                        _cogp_phase = ""
                    if not _cogp_phase:
                        _cogp_phase = (
                            getattr(getattr(ctx, "phase", None), "name", None)
                            or str(getattr(ctx, "phase", "") or "")
                        )
                    _cogp.record_terminal_experiences_fire_and_forget(
                        _cogp_records,
                        footprint=_cogp_footprint,
                        terminal_reason=_cogp_reason,
                        phase=_cogp_phase,
                        op_id=str(getattr(ctx, "op_id", "") or "?"),
                    )
        except Exception as _cogp_exc:  # noqa: BLE001 — never disturb the FSM
            logger.debug(
                "[CognitivePersistence] terminal hook skipped: %s", _cogp_exc,
            )
        # Slice 74 — Immutable Lifecycle Boundary: the terminal SSE broadcast is
        # DECOUPLED from the ledger's physical-write dedup. A definitive terminal
        # state MUST notify the system (the autoscore eval rendezvous + IDE
        # consumers) even when the ledger deduped the row (written=False) —
        # otherwise a deduped COMPLETED/FAILED write silently drops the wake
        # (the bt-2026-06-03-063919 25-min-lag class). publish_operation_terminal
        # is internally a no-op for non-terminal states and carries its own
        # (op_id, state) idempotency guard, so this is exactly-once regardless of
        # how many times _record_ledger fires for the state. NEVER raises here.
        try:
            from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501
                publish_operation_terminal as _s74_publish_terminal,
            )
            _s74_publish_terminal(ctx, state)
        except Exception:  # noqa: BLE001 — never raise into _record_ledger
            pass
        # ── P0.4 — learning plane write seam ──
        # Record this op's per-file outcome into the default reputation store
        # so intake can later bias toward historically-substantive (fragile /
        # high-churn / high-blast) files. Terminal states only (the same
        # (op_id, state) that fires the SSE above); the store is ledger-free
        # (uses ctx.target_files, no re-read). Gated (§33.1 default-FALSE),
        # sync (a dict update + debounced flush), NEVER raises here.
        try:
            from backend.core.ouroboros.consciousness.memory_engine import (
                get_default_memory_engine as _rep_engine,
                reputation_write_enabled as _rep_write_on,
            )
            if _rep_write_on():
                _rep_state = str(getattr(state, "value", "") or "")
                if _rep_state in ("applied", "rolled_back", "failed", "blocked"):
                    _rep_engine(
                        getattr(self._config, "project_root", "."),
                    ).record_file_outcome(
                        getattr(ctx, "target_files", ()) or (),
                        success=(_rep_state == "applied"),
                    )
        except Exception:  # noqa: BLE001 — learning never breaks _record_ledger
            pass
        # ── Slice 101 — Cognitive Integration Bus lifecycle fan-out ──
        # Mirror the terminal state onto the cognitive bus so dormant
        # substrates (belief revision, counterfactual rehearsal, ...) can
        # react in the BACKGROUND via async subscribers. Sync-safe, fire-
        # and-forget, NEVER raises into _record_ledger; inert unless
        # JARVIS_COGNITIVE_BUS_ENABLED (§33.1 default-FALSE). Decoupled from
        # `written` just like the SSE publish above — a deduped terminal row
        # must still wake the cognitive layer (the publisher is internally a
        # no-op when the bus isn't running, so this is exactly-once-safe).
        try:
            _cb_sv = str(getattr(state, "value", state)).lower()
            _cb_kind = None
            if _cb_sv == "applied":
                _cb_kind = "post_apply"
            elif _cb_sv in ("failed", "blocked", "rolled_back"):
                _cb_kind = "post_failure"
            if _cb_kind is not None:
                from backend.core.ouroboros.governance.cognitive_bus import (  # noqa: E501
                    publish_lifecycle_event as _cb_publish,
                )
                try:
                    _cb_tf = [
                        str(p)
                        for p in (getattr(ctx, "target_files", None) or [])
                    ][:32]
                except Exception:  # noqa: BLE001
                    _cb_tf = []
                # Slice 109 — enrich the lifecycle payload with the decision
                # context (confidence + risk tier) so the cognitive
                # observability subscriber can build a complete Why-Snapshot
                # (confidence_aura band) at the moment of the decision.
                try:
                    _cb_conf = getattr(ctx, "confidence", None)
                    _cb_conf = float(_cb_conf) if _cb_conf is not None else None
                except Exception:  # noqa: BLE001
                    _cb_conf = None
                try:
                    _cb_rt = getattr(ctx, "risk_tier", None)
                    _cb_rt = (
                        _cb_rt.name.lower()
                        if _cb_rt is not None and hasattr(_cb_rt, "name")
                        else (str(_cb_rt) if _cb_rt is not None else "")
                    )
                except Exception:  # noqa: BLE001
                    _cb_rt = ""
                _cb_publish(
                    _cb_kind,
                    {
                        "op_id": str(getattr(ctx, "op_id", "") or ""),
                        "state": _cb_sv,
                        "phase": str(getattr(ctx, "current_phase", "") or ""),
                        "target_files": _cb_tf,
                        "reason": str((data or {}).get("reason", "")),
                        "confidence": _cb_conf,
                        "risk_tier": _cb_rt,
                    },
                    correlation_id=str(getattr(ctx, "op_id", "") or "") or None,
                )
        except Exception:  # noqa: BLE001 — bus fan-out must never touch the FSM
            pass
        # ── Slice 104 — Operator-Independent Recursion-Depth tracker ──
        # On a successful APPLY, advance the self-modification chain counter: a
        # governance-touching apply increments it, any other apply resets it.
        # The recursion-depth floor reads this counter at GATE to halt a runaway
        # self-modification chain (RRD §23.5). Master-gated (default-TRUE);
        # NEVER raises into _record_ledger.
        try:
            if str(getattr(state, "value", state)).lower() == "applied":
                from backend.core.ouroboros.governance.recursion_depth_gate import (  # noqa: E501
                    note_apply as _rdg_note_apply,
                )
                _rdg_note_apply(getattr(ctx, "target_files", None))
        except Exception:  # noqa: BLE001 — recursion tracker must never touch the FSM
            pass
        if written:
            try:
                from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501
                    TERMINAL_OPERATION_STATES,
                )
                # Fail-Fast counter prune — single terminal chokepoint.
                # Composes the SAME canonical terminal set the SSE
                # publisher uses (no duplicated state list). Pruning
                # ONLY on a genuine terminal state keeps the breaker's
                # consecutive count intact across mid-op ledger
                # records while preventing unbounded dict growth and
                # honoring reset-on-op-completion.
                if getattr(state, "value", state) in (
                    TERMINAL_OPERATION_STATES
                ):
                    self._failfast_exhaust_consec.pop(
                        str(getattr(ctx, "op_id", "") or ""), None,
                    )
                    # ── Slice 12Q — SessionRecorder terminal wiring ──
                    # Direct call into the harness-owned SessionRecorder
                    # via the process-singleton accessor. This closes the
                    # bt-2026-05-23-042249 gap where summary.json.
                    # operations[] was empty because the existing
                    # OP_COMPLETED event handler subscribes to a path
                    # (gls.report_outcome) that nothing in the runtime
                    # actually calls. The recorder's own idempotency
                    # (self._recorded_op_ids) protects against any
                    # future duplicate if the OP_COMPLETED path is
                    # wired up later. NEVER raises into _record_ledger.
                    _slice12q_record_terminal(ctx, state, data)
                    # ── Terminal voice — the outcome reaches the transport ──
                    await self._emit_terminal_decision(ctx, state, data)
                    # ── Slice 12AA — Per-op reservation release ──
                    # Free the foreground op's reserved session
                    # runway so subsequent ops can use it. Lazy-
                    # acquired during provider preflight; released
                    # at the SAME single-seam terminal chokepoint
                    # where the SessionRecorder lands the op.
                    # Idempotent; NEVER raises into _record_ledger.
                    try:
                        from backend.core.ouroboros.governance.session_budget_authority import (  # noqa: E501
                            release_reservation as _sba_release,
                        )
                        _aa_op_id = str(
                            getattr(ctx, "op_id", "") or ""
                        )
                        if _aa_op_id:
                            _sba_release(_aa_op_id)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001 — observability is best-effort
                logger.debug(
                    "[Orchestrator] publish_operation_terminal raised "
                    "(swallowed): op_id=%s state=%s",
                    entry.op_id, entry.state.value,
                    exc_info=True,
                )

    async def _emit_terminal_decision(
        self,
        ctx: OperationContext,
        state: OperationState,
        data: Dict[str, Any],
    ) -> None:
        """Speak the op's terminal outcome on the comm wire.

        ``ChangeEngine`` emits a DECISION for the one outcome it owns — a
        real apply. Every other way an op ends (the model declining because
        the change is already present, a generation failure, a rollback, a
        gate holding it for a human) reached the ledger and the SSE broker
        and never the transport, so the transcript showed an op begin and
        never end: measured 2026-09-06, 116 ``2b.1-noop`` refusals and not
        one closing line. This is the ONE terminal chokepoint, so it is
        where the remaining outcomes are spoken — reading the signals the
        ledger already reads, adding no state. An apply with no terminal
        reason is left to ChangeEngine (a second DECISION for the same op
        would be dropped by the transport, but the wire should carry one).
        Best-effort; NEVER raises into ``_record_ledger``.
        """
        try:
            comm = getattr(self._stack, "comm", None)
            if comm is None:
                return
            state_value = str(getattr(state, "value", state) or "").lower()
            reason_code = str(
                getattr(ctx, "terminal_reason_code", "")
                or (isinstance(data, dict)
                    and (data.get("reason") or data.get("error")))
                or ""
            )
            generation = getattr(ctx, "generation", None)
            words = ""
            if state_value == "applied":
                if not reason_code:
                    return
                is_noop = bool(getattr(generation, "is_noop", False))
                normalised = reason_code.replace("_", "").lower()
                outcome = (
                    "noop" if is_noop or normalised.startswith("noop")
                    or "noop" in normalised else "completed"
                )
                words = str(getattr(generation, "noop_reason", "") or "")
            elif state_value == "failed":
                outcome = "failed"
            elif state_value == "rolled_back":
                outcome = "failed"
                reason_code = reason_code or "rolled_back"
            elif state_value == "blocked":
                outcome = "escalated"
            else:
                return
            if not words and isinstance(data, dict):
                words = str(data.get("detail") or "")
            # Recap counts, synthesised from THIS op's execution ledger, so
            # the transport can draw the "✻ Crunched for … · N tools · done …"
            # line without re-deriving them. Best-effort; a missing generation
            # simply yields zeros, which the recap drops rather than prints.
            _tools = _tokens = 0
            try:
                from backend.core.ouroboros.governance.op_recap import (
                    output_tokens as _rc_tokens,
                    tool_count as _rc_tools,
                )
                _tools = _rc_tools(generation)
                _tokens = _rc_tokens(generation)
            except Exception:  # noqa: BLE001
                pass
            # Authoritative op duration for the recap + outcome line. The
            # op's lifetime is created_at (stamped at CLASSIFY, carried
            # immutably through every advance) -> now. The transport measures
            # elapsed from when ITS cockpit client first observed the op, so
            # an op it meets only at its terminal state (a gate held it with
            # no INTENT, or a client attached mid-flight) reads 0.0s. Supplied
            # here, the lifecycle duration lets the transport render the true
            # time for every terminal op, not only the ones it watched from
            # INTENT. Best-effort: a missing/naive stamp yields 0.0, which the
            # transport reads as 'unknown' and falls back to its local clock.
            _dur_s = 0.0
            try:
                _created = getattr(ctx, "created_at", None)
                if _created is not None:
                    from datetime import datetime as _dt
                    _ref = _dt.now(getattr(_created, "tzinfo", None))
                    _dur_s = max(0.0, (_ref - _created).total_seconds())
            except Exception:  # noqa: BLE001
                _dur_s = 0.0
            await comm.emit_decision(
                op_id=str(getattr(ctx, "op_id", "") or ""),
                outcome=outcome,
                reason_code=reason_code or state_value,
                target_files=[
                    str(p) for p in (getattr(ctx, "target_files", ()) or ())
                ],
                reason=words,
                terminal_state=state_value,
                tools_used=_tools,
                tokens=_tokens,
                duration_s=_dur_s,
            )
        except Exception:  # noqa: BLE001 — the voice never touches the FSM
            logger.debug(
                "[Orchestrator] terminal DECISION emit failed (swallowed): "
                "op_id=%s", getattr(ctx, "op_id", "?"), exc_info=True,
            )


# Alias so tests can import `Orchestrator` as well as `GovernedOrchestrator`
Orchestrator = GovernedOrchestrator
