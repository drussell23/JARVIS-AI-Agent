"""
Governed Loop Service — Lifecycle Manager
==========================================

Thin lifecycle manager for the governed self-programming pipeline.
Owns provider wiring, orchestrator construction, and health probes.
No domain logic — just coordination.

The supervisor instantiates this in Zone 6.8 and calls start()/stop().
All triggers go through submit(), which delegates to the orchestrator.

Service States
--------------
INACTIVE -> STARTING -> ACTIVE/DEGRADED
ACTIVE/DEGRADED -> STOPPING -> INACTIVE
STARTING -> FAILED (on error)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from backend.core.ouroboros.governance.approval_provider import CLIApprovalProvider  # noqa: F401  (kept for back-compat reference; factory selects)
from backend.core.ouroboros.governance.inline_approval_provider import (
    build_approval_provider,
)
from backend.core.ouroboros.governance.candidate_generator import (
    CandidateGenerator,
    FailbackState,
)
from backend.core.ouroboros.governance.ledger import LedgerEntry, OperationState
from backend.core.ouroboros.governance.op_context import (
    HostTelemetry,
    OperationContext,
    OperationPhase,
    RoutingIntentTelemetry,
    TelemetryContext,
)
from backend.core.ouroboros.governance.resource_monitor import PressureLevel, ResourceSnapshot
# IntakeLayerService is started by the supervisor (Zone 6.9); GLS only stores
# the resolved RepoRegistry on self._repo_registry for Zone 6.9 to reuse.
from backend.core.ouroboros.governance.multi_repo.registry import RepoRegistry
from backend.core.ouroboros.governance.orchestrator import (
    GovernedOrchestrator,
    OrchestratorConfig,
)
from backend.core.ouroboros.governance.curriculum_publisher import CurriculumPublisher
from backend.core.ouroboros.governance.model_attribution_recorder import ModelAttributionRecorder
from backend.core.ouroboros.integration import get_performance_persistence
from backend.core.ouroboros.governance.preemption_fsm import (
    PreemptionFsmEngine,
    PreemptionFsmExecutor,
    build_transition_input,
)
from backend.core.ouroboros.governance.contracts.fsm_contract import (
    LoopEvent,
    LoopRuntimeContext,
    LoopState,
    RetryBudget,
)
from backend.core.ouroboros.governance.autonomy.command_bus import CommandBus
from backend.core.ouroboros.governance.autonomy.event_emitter import EventEmitter
from backend.core.ouroboros.governance.autonomy.feedback_engine import (
    AutonomyFeedbackEngine,
    FeedbackEngineConfig,
)
from backend.core.ouroboros.governance.autonomy.autonomy_types import (
    CommandType as AutonomyCommandType,
    EventEnvelope as AutonomyEventEnvelope,
    EventType as AutonomyEventType,
)
from backend.core.ouroboros.governance.autonomy.safety_net import (
    ProductionSafetyNet,
    SafetyNetConfig,
)
from backend.core.ouroboros.governance.user_signal_bus import UserSignalBus

try:
    from backend.core.ouroboros.oracle import TheOracle as TheOracle
except ImportError:
    TheOracle = None  # type: ignore[assignment,misc]

logger = logging.getLogger("Ouroboros.GovernedLoop")


# ---------------------------------------------------------------------------
# T4 — Sovereign Telemetry Boot-Guard
# ---------------------------------------------------------------------------
# Extracted helper so unit tests can verify the exact warning string without
# booting the full GovernedLoopService. Default-ON means this fires ONLY when
# an operator has explicitly set JARVIS_DW_EGRESS_INTERCEPTOR_ENABLED=false
# (or 0/no/off). Never raises — boot must never crash due to this guard.


def _warn_if_egress_guard_disabled(log: logging.Logger) -> bool:
    """Emit a loud [SOVEREIGN WARNING] when the egress interceptor is OFF.

    Returns True (warning emitted) when the guard is disabled, False otherwise.
    NEVER raises — the check is always fail-soft.
    """
    try:
        from backend.core.ouroboros.governance.dw_egress_interceptor import (
            egress_interceptor_enabled,
        )
        if not egress_interceptor_enabled():
            log.warning(
                "[SOVEREIGN WARNING] API Citizenship Guard Disabled: Egress Interceptor "
                "is OFF. Node is vulnerable to overweight payload dispatch."
            )
            return True
    except Exception:  # noqa: BLE001
        pass  # never let this guard crash boot
    return False


# ---------------------------------------------------------------------------
# Slice 2B-ii.1 — Aegis-aware provider construction gate
# ---------------------------------------------------------------------------
# Closes the Catch-22 surfaced by Aegis Detonation soak bt-2026-05-24-222008:
# Aegis preflight (Slice 1) intentionally scrubs ANTHROPIC_API_KEY +
# DOUBLEWORD_API_KEY from the harness env at T+~2s. The previous gates
# `if self._config.claude_api_key:` and `if _dw_api_key:` then refused
# to construct the providers — despite the fact that the credentials
# are now safely held in the Aegis daemon and would be injected
# upstream-side at every forwarded request. This helper composes the
# canonical `aegis.client.is_enabled()` predicate as an OR-fallback so
# the providers ARE constructed under the Aegis-enabled path; their
# downstream `_ensure_client()` calls route through the Slice 2B-ii
# provider bridge (`aegis_provider_bridge.make_async_anthropic_client`)
# which transparently uses the daemon's `/v1/*` forwarding surface.
#
# The predicate is read at call-time (not cached) so the test
# monkeypatch path works and so an operator can enable Aegis after
# the module has been imported.
def _provider_construction_gate(
    *,
    local_api_key: Optional[str],
    provider_name: str = "",
) -> bool:
    """Decide whether a provider (Claude / DoubleWord) should be
    constructed at GovernedLoopService boot.

    Returns True iff EITHER the local API key is truthy OR Aegis is
    enabled (key is in the daemon, not our process). Returns False
    only when both are absent — the unambiguous "this provider is
    not configured" case.

    Pure, side-effect-free, no I/O. Extracted as a callable so the
    AST pin can enforce its composition shape and so unit tests can
    monkeypatch ``aegis.client.is_enabled`` cleanly.

    # Slice 19a (2026-05-26) — pure provider isolation gate

    Operator binding: "i want to run the soak with only using DW's API's
    because i want to understand how external API works". To run a
    DW-only soak (no Claude fallback for empirical isolation), the
    operator sets ``JARVIS_PROVIDER_CLAUDE_DISABLED=true``. When set
    AND this call is gating ClaudeProvider construction (signalled by
    ``provider_name="claude"``), the gate short-circuits to False even
    if the API key is present and Aegis is enabled. The provider is
    NEVER constructed; ``self._fallback`` stays None; the
    candidate_generator's cascade naturally degrades to
    "all_providers_exhausted" on DW failures instead of cascading to a
    non-existent fallback.

    IMMEDIATE-routed ops (per Manifesto §5) fail VISIBLY in this mode
    because §5 specifies Claude-direct for human-reflex routing and
    there is no Claude. Per operator binding: "if an unrelated process
    tries to call an IMMEDIATE reflex action while Claude is intentionally
    pulled, it must fail visibly to maintain absolute system observability."

    SWE-Bench-Pro ops are unaffected because Slice 10A (PR #58161)
    downgrades them to STANDARD route which uses DW primary.

    The disable knob is ONLY honored when ``provider_name="claude"``;
    other providers (DoubleWord) ignore it. DW disable would need its
    own Slice (operator-bound; not authorized here).

    Defensive fail-closed: the env value is parsed as a strict
    truthy-string set (``true``/``1``/``yes``/``on`` case-insensitive);
    any other value (including empty, missing, mis-typed) preserves
    pre-Slice-19a behavior.
    """
    # Slice 19a — Claude-specific isolation gate
    if provider_name == "claude":
        _disable_raw = os.environ.get(
            "JARVIS_PROVIDER_CLAUDE_DISABLED", "",
        ).strip().lower()
        if _disable_raw in ("true", "1", "yes", "on"):
            return False
    from backend.core.ouroboros.aegis.client import is_enabled as _aegis_is_enabled
    return bool(local_api_key) or _aegis_is_enabled()


# ---------------------------------------------------------------------------
# Sandbox-safe state directory — redirect ~/.jarvis when not writable
# ---------------------------------------------------------------------------
# P2 Slice 3 — Universal Convergence registry wire-helpers
# ---------------------------------------------------------------------------
#
# Thin module-level adapters around the in_flight_registry's
# safe-wire helpers. Lazy-imported so the registry stays optional
# at module-load time and so any future changes to the substrate
# don't ripple into the live loop's hot path through fragile
# import edges. Each helper is master-gated + NEVER-raise by
# construction (the substrate's helpers already are).


# ---------------------------------------------------------------------------
# Phase-transition tracking wiring (2026-07-18)
#
# The in-flight registry recorded only the registration-time phase (CLASSIFY),
# so a checkpoint of an op that had advanced to GENERATE serialized a stale
# CLASSIFY and resume couldn't fast-forward. Rather than sprinkle
# update_phase_safely() across every orchestrator phase-runner, register ONE
# observer on the state machine's own transition method (OperationContext.advance)
# — every legal transition auto-mirrors into the registry.
#
# The wiring lives HERE, at the pipeline-orchestration layer (the GLS already
# imports BOTH op_context and in_flight_registry), and NOT inside
# in_flight_registry — an AST authority-asymmetry pin forbids the registry from
# importing op_context (the deliberate observability→state-machine no-cycle
# invariant). This composition direction (GLS → both) respects it.
# ---------------------------------------------------------------------------

_phase_tracking_wired = False


def _mirror_phase_transition(op_id: str, phase_name: str) -> None:
    """Observer body: mirror an advance() transition into the in-flight registry.
    Touches only ops already registered (update_phase no-ops for unknown ids)."""
    try:
        from backend.core.ouroboros.governance.in_flight_registry import (  # noqa: E501,PLC0415
            update_phase_safely,
        )
        update_phase_safely(op_id, phase_name=phase_name)
    except Exception:  # noqa: BLE001
        pass


def _wire_phase_transition_tracking() -> bool:
    """Idempotently register the phase-mirror observer on OperationContext. No
    lock needed — ``register_phase_transition_observer`` dedups per-callable, so
    a concurrent double-call registers exactly one observer. NEVER raises."""
    global _phase_tracking_wired
    if _phase_tracking_wired:
        return True
    try:
        from backend.core.ouroboros.governance.op_context import (  # noqa: E501,PLC0415
            register_phase_transition_observer,
        )
        register_phase_transition_observer(_mirror_phase_transition)
        _phase_tracking_wired = True
        return True
    except Exception:  # noqa: BLE001
        return False


def _reset_phase_tracking_for_tests() -> None:
    global _phase_tracking_wired
    _phase_tracking_wired = False


def _register_op_in_flight_safely(
    op_id: str,
    *,
    ctx_ref: Any = None,
    last_phase_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        from backend.core.ouroboros.governance.in_flight_registry import (  # noqa: E501
            register_op_safely,
        )
    except Exception:  # noqa: BLE001
        return False
    # Arm the advance()-transition observer (idempotent) so the registry tracks
    # the LIVE phase, not just this registration-time phase — otherwise
    # capture_inflight serializes a stale CLASSIFY. One bool check after op #1.
    _wire_phase_transition_tracking()
    return register_op_safely(
        op_id,
        ctx_ref=ctx_ref,
        last_phase_name=last_phase_name,
        metadata=metadata,
    )


def _unregister_op_in_flight_safely(op_id: str) -> bool:
    try:
        from backend.core.ouroboros.governance.in_flight_registry import (  # noqa: E501
            unregister_op_safely,
        )
    except Exception:  # noqa: BLE001
        return False
    return unregister_op_safely(op_id)


def _op_registry_metadata(ctx: Any) -> Dict[str, Any]:
    """Project the in-flight registry metadata dict from an
    ``OperationContext``. Pulls only safe, JSON-friendly fields
    — provider name, route, urgency, source. NEVER raises;
    returns ``{}`` on any failure."""
    try:
        return {
            "provider": str(getattr(ctx, "provider", "") or ""),
            "route": str(getattr(ctx, "route", "") or ""),
            "urgency": str(
                getattr(ctx, "urgency_level", "") or ""
            ),
            "source": str(
                getattr(ctx, "outcome_source", "") or ""
            ),
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Many subsystems write to ~/.jarvis/ouroboros/*.  In sandboxed environments
# (e.g. Claude Code, macOS sandbox, CI containers) that path may not be
# writable.  Detect once at import time and redirect to a repo-local
# fallback via JARVIS_STATE_DIR so all subsystems that resolve via
# Path.home() / ".jarvis" or JARVIS_SELF_EVOLUTION_DIR fall through cleanly.

def _ensure_writable_state_dir() -> None:
    """Set JARVIS_STATE_DIR if ~/.jarvis is not writable."""
    if os.environ.get("JARVIS_STATE_DIR"):
        return  # already explicitly set
    home_jarvis = Path.home() / ".jarvis"
    try:
        home_jarvis.mkdir(parents=True, exist_ok=True)
        # Write-test with a temp file
        _probe = home_jarvis / ".write_probe"
        _probe.write_text("ok")
        _probe.unlink()
    except OSError:
        # Not writable — fall back to repo-local .ouroboros/state/
        _fallback = Path.cwd() / ".ouroboros" / "state"
        try:
            _fallback.mkdir(parents=True, exist_ok=True)
        except OSError:
            return  # can't create fallback either — let individual modules handle it
        os.environ["JARVIS_STATE_DIR"] = str(_fallback)
        # Redirect subsystem env vars that default to ~/.jarvis paths
        _redirects = {
            "JARVIS_SELF_EVOLUTION_DIR": str(_fallback / "ouroboros" / "evolution"),
            "JARVIS_GOVERNED_L3_STATE_DIR": str(_fallback / "ouroboros" / "execution_graphs"),
            "JARVIS_GOVERNED_L4_STATE_DIR": str(_fallback / "ouroboros" / "advanced_coordination"),
            # OperationLedger (change_engine Phase 1 writes). If this is not
            # redirected, every op that reaches APPLY dies on the first
            # ledger.append() with PermissionError — see bt-2026-04-10-075150.
            "OUROBOROS_LEDGER_DIR": str(_fallback / "ouroboros" / "ledger"),
        }
        for key, val in _redirects.items():
            if not os.environ.get(key):
                os.environ[key] = val
        logger.info(
            "[GovernedLoop] ~/.jarvis not writable — redirected state to %s",
            _fallback,
        )

_ensure_writable_state_dir()

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MIN_GENERATION_BUDGET_S: float = float(
    os.getenv("JARVIS_MIN_GENERATION_BUDGET_S", "30.0")
)

# ---------------------------------------------------------------------------
# Compute-class admission constants and helpers
# ---------------------------------------------------------------------------

_COMPUTE_RANK: dict[str, int] = {
    "cpu": 0,
    "gpu_t4": 1,
    "gpu_l4": 2,
    "gpu_v100": 3,
    "gpu_a100": 4,
}


def _oracle_full_scan_suppressed_by_benchmark() -> bool:
    """Slice 86 — True when the periodic full-repo Oracle scan must be skipped
    because a benchmark run is in flight.

    Composes the canonical ``benchmark_isolation_mode()`` (Slice 63) with a
    dedicated kill-switch ``JARVIS_BENCHMARK_SUPPRESS_ORACLE_FULL_SCAN`` (default
    ON) so an operator can re-enable the scan inside isolation if ever needed.
    Pure + defensive: any import/parse error returns ``False`` (preserve legacy —
    never silently stop indexing on an unrelated fault)."""
    try:
        from backend.core.ouroboros.governance.intake.intake_layer_service import (
            benchmark_isolation_mode,
        )
        if not benchmark_isolation_mode():
            return False
        raw = os.environ.get(
            "JARVIS_BENCHMARK_SUPPRESS_ORACLE_FULL_SCAN", "true",
        ).strip().lower()
        return raw not in ("0", "false", "no", "off")
    except Exception:  # noqa: BLE001 — never block the index loop
        return False


class ComputeClassMismatch(RuntimeError):
    """Raised when VM compute_class is below the brain's min_compute_class."""


def _check_compute_admission(brain_cfg: dict, capability: dict) -> None:
    """Hard-fail if this host cannot carry the brain's memory requirement.

    Two authorities, in priority order, because the question being asked is
    about BYTES and only one of them can answer it in bytes.

    **Measured (preferred).** ``compute_topology`` probes the accelerator and
    reports capacity. The brain states a requirement as ``min_vram_gb`` or —
    for policy written before that field existed — as a legacy
    ``min_compute_class`` name, which ``bytes_for_requirement`` interprets as
    the capacity that name implies. Admission then compares capacity to
    requirement directly. A card nobody enumerated needs no new rung: a
    32 GiB consumer GPU outranks ``gpu_l4`` because 32 > 24, which is the
    fact the name-ranked table was standing in for all along.

    **Ordinal (fallback).** When the probe is disabled, or the host cannot be
    resolved, the legacy ``_COMPUTE_RANK`` comparison runs unchanged. This is
    not a degraded mode — it is byte-for-byte the pre-existing behaviour, and
    it is what an ``UNKNOWN`` topology is *for*: an unresolved host may not
    have a capacity claim invented on its behalf in either direction.

    **Locality is proven, never assumed.** ``capability`` is fetched over
    HTTP from a J-Prime endpoint that may be a GCP VM on another continent.
    A local accelerator reading describes only this machine, so the measured
    path engages solely when ``describes_this_host`` positively matches the
    payload's host against this machine. Anything else — a remote brain, an
    absent host field, a malformed payload — takes the ordinal path.
    Authorizing a remote route with a local GPU reading would be the same
    wrong-resource error this work exists to remove, wearing new clothes.
    """
    vm_class = capability.get("compute_class", "cpu")
    min_class = brain_cfg.get("min_compute_class", "cpu")
    min_vram_gb = brain_cfg.get("min_vram_gb")

    try:
        from backend.core.ouroboros.governance import compute_topology as _ct

        required = _ct.bytes_for_requirement(
            min_compute_class=min_class, min_vram_gb=min_vram_gb,
        )
        local = _ct.describes_this_host(capability)
        reading = _ct.resolve_sync() if local else None
        if reading is not None and reading.measured and required > 0:
            if reading.usable_bytes >= required:
                return
            raise ComputeClassMismatch(
                f"host {reading.resolved_class!r} "
                f"({reading.usable_bytes / (1024 ** 3):.1f} GiB usable, "
                f"topology={reading.topology.value}, source={reading.source}) "
                f"cannot carry brain requirement "
                f"{required / (1024 ** 3):.1f} GiB "
                f"(min_vram_gb={min_vram_gb!r} min_compute_class={min_class!r}). "
                f"Route is denied. Select a lower-tier brain or a larger host."
            )
    except ComputeClassMismatch:
        raise
    except Exception as exc:  # noqa: BLE001 — measurement never blocks admission
        logger.debug(
            "[ComputeAdmission] measured path unavailable (%s); "
            "falling back to ordinal comparison", exc,
        )

    vm_rank = _COMPUTE_RANK.get(vm_class, 0)
    min_rank = _COMPUTE_RANK.get(min_class, 0)
    if vm_rank < min_rank:
        raise ComputeClassMismatch(
            f"VM compute_class={vm_class!r} (rank {vm_rank}) is below "
            f"brain min_compute_class={min_class!r} (rank {min_rank}). "
            f"Route to J-Prime is denied. Upgrade VM GPU or select a lower-tier brain."
        )


class ModelArtifactMismatch(RuntimeError):
    """Raised when VM model_artifact doesn't match policy model_artifact."""


def _check_artifact_integrity(brain_cfg: dict, capability: dict) -> None:
    """Hard-fail if model loaded on VM doesn't match policy's expected artifact.

    Comparison is case-insensitive to handle filesystem conventions.
    If either artifact is unknown/empty, skips the check.

    Raises:
        ModelArtifactMismatch: if filenames don't match (case-insensitive)
    """
    policy_artifact = brain_cfg.get("model_artifact", "")
    vm_artifact = capability.get("model_artifact", "")
    if not policy_artifact or not vm_artifact:
        return  # can't check — skip
    if policy_artifact.lower() != vm_artifact.lower():
        raise ModelArtifactMismatch(
            f"Model artifact mismatch: policy expects {policy_artifact!r} "
            f"but VM reports {vm_artifact!r}. "
            f"Update policy or reload correct model on VM."
        )


class HostBindingViolation(RuntimeError):
    """Raised when telemetry_host, selector_host, and execution_host don't all match."""


def _check_host_binding(
    telemetry_host: str,
    selector_host: str,
    execution_host: str,
) -> None:
    """Enforce the invariant: all three host references must be identical.

    This prevents scenarios where routing selects VM-A but execution reaches VM-B,
    or where local psutil data is incorrectly used for a remote route.

    Raises:
        HostBindingViolation: if any host differs from the others
    """
    hosts = {telemetry_host, selector_host, execution_host}
    if len(hosts) > 1:
        raise HostBindingViolation(
            f"Host-binding invariant violated: "
            f"telemetry_host={telemetry_host!r}, "
            f"selector_host={selector_host!r}, "
            f"execution_host={execution_host!r}. "
            f"All three must be identical."
        )


# ---------------------------------------------------------------------------
# Phase 4: FSM infrastructure adapters
# ---------------------------------------------------------------------------


class _FsmLedgerAdapter:
    """Adapts OperationLedger to the FSM Ledger protocol.

    Converts FSM checkpoint appends to OperationLedger LedgerEntry writes.
    Idempotency guard uses an in-memory set (resets on restart, which is
    acceptable because each LoopRuntimeContext begins from RUNNING on startup).
    """

    def __init__(self, ledger: Any) -> None:
        self._ledger = ledger
        self._seen: Set[Tuple[str, int]] = set()

    async def checkpoint_exists(self, *, op_id: str, checkpoint_seq: int) -> bool:
        return (op_id, checkpoint_seq) in self._seen

    async def append_checkpoint(
        self,
        *,
        op_id: str,
        checkpoint_seq: int,
        state: Any,
        event: Any,
        reason_code: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        from backend.core.ouroboros.governance.ledger import LedgerEntry, OperationState

        self._seen.add((op_id, checkpoint_seq))
        try:
            await self._ledger.append(
                LedgerEntry(
                    op_id=op_id,
                    state=OperationState.BLOCKED,
                    data={
                        "type": "preemption_fsm_checkpoint",
                        "loop_state": state.value,
                        "loop_event": event.value,
                        "reason_code": reason_code,
                        "checkpoint_seq": checkpoint_seq,
                        **payload,
                    },
                )
            )
        except Exception:
            pass  # ledger failure must never block an FSM transition


class _CommTelemetrySink:
    """Wraps CommProtocol to satisfy the FSM TelemetrySink protocol."""

    def __init__(self, comm: Any) -> None:
        self._comm = comm

    async def emit_transition(self, decision: Any, payload: Dict[str, Any]) -> None:
        op_id = payload.get("op_id", "unknown")
        try:
            await self._comm.emit_heartbeat(
                op_id=op_id,
                phase=f"preemption_fsm:{decision.to_state.value}",
                progress_pct=0.0,
            )
        except Exception:
            pass  # telemetry is best-effort


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _record_ledger(
    ctx: "OperationContext",
    ledger: Any,
    state: "OperationState",
    data: Dict[str, Any],
) -> None:
    """Append a ledger entry, logging errors without raising.

    Standalone helper used by GovernedLoopService._preflight_check() so that
    ledger writes can happen before the orchestrator is involved.
    """
    from backend.core.ouroboros.governance.ledger import LedgerEntry

    entry = LedgerEntry(
        op_id=ctx.op_id,
        state=state,
        data=data,
    )
    try:
        await ledger.append(entry)
    except Exception as exc:
        logger.error(
            "Ledger append failed: op_id=%s state=%s error=%s",
            entry.op_id,
            entry.state.value,
            exc,
        )


def _expected_provider_from_pressure(snap: ResourceSnapshot, active_ops: int = 0) -> str:
    """DEPRECATED — retained for backward compat only. Do not use for routing.

    Use _expected_provider_from_brain() instead, which derives expected_provider
    from the BrainSelectionResult, not from local Mac resource pressure.
    """
    # Phase 1 P0: local Mac pressure must not influence GCP routing telemetry.
    # This function is kept so callers that haven't been migrated don't break at
    # import time; all call sites inside GLS now use _expected_provider_from_brain.
    if snap.pressure_for_load(active_ops) >= PressureLevel.CRITICAL:
        return "LOCAL_CLAUDE"
    return "GCP_PRIME_SPOT"


def _expected_provider_from_brain(brain: "BrainSelectionResult") -> str:  # type: ignore[name-defined]
    """Derive expected_provider from the BrainSelectionResult, NOT from local psutil.

    Respects the host-binding invariant: routing-authority fields in telemetry
    must reflect the actual brain selection outcome, not local Mac resource state.
    """
    tier = getattr(brain, "provider_tier", "gcp_prime").upper()
    # Normalise known tiers to a canonical form
    if tier.startswith("GCP"):
        return "GCP_PRIME_SPOT"
    if tier.startswith("CLAUDE") or tier == "CLAUDE_API":
        return "CLAUDE_API"
    if tier == "QUEUED":
        return "QUEUED"
    return tier


def _policy_reason_from_brain(brain: "BrainSelectionResult") -> str:  # type: ignore[name-defined]
    """Return the causal routing_reason from BrainSelectionResult.

    Replaces the pattern of using snap.pressure_for_load().name as policy_reason,
    which incorrectly stamped LOCAL Mac pressure as the routing policy authority.
    """
    return getattr(brain, "routing_reason", "unknown")


def _infer_canary_slice(target_files: tuple) -> str:
    """Derive the most restrictive canary slice from target file paths.

    Checks all files and returns the most constrained slice:
    - "tests/" and "docs/" → GOVERNED (lowest restriction)
    - "backend/core/" → OBSERVE
    - "" (root default) → OBSERVE

    When files span multiple slices, returns the most restrictive.
    """
    # Ordered from most restrictive to least restrictive
    _SLICE_ORDER = ["backend/core/", "", "tests/", "docs/"]
    found: set = set()
    for fp in target_files:
        fp_norm = fp.replace("\\", "/").lstrip("./")
        if fp_norm.startswith("tests/"):
            found.add("tests/")
        elif fp_norm.startswith("docs/"):
            found.add("docs/")
        elif fp_norm.startswith("backend/core/"):
            found.add("backend/core/")
        else:
            found.add("")
    if not found:
        return ""
    # Return most restrictive: OBSERVE slices (backend/core/, "") beat GOVERNED slices
    for s in _SLICE_ORDER:
        if s in found:
            return s
    return ""


# ---------------------------------------------------------------------------
# Terminal classification helpers
# ---------------------------------------------------------------------------


def _classify_terminal(
    terminal_phase: "OperationPhase",
    provider_used: "str | None",
    reason_code: str,
    is_noop: bool,
) -> str:
    """Classify operation outcome into the terminal taxonomy.

    Returns one of: PRIMARY_SUCCESS, FALLBACK_SUCCESS, DEGRADED, TIMEOUT, NOOP
    """
    from backend.core.ouroboros.governance.op_context import OperationPhase
    if is_noop:
        return "NOOP"
    if terminal_phase == OperationPhase.COMPLETE:
        if provider_used and "prime" in provider_used.lower():
            return "PRIMARY_SUCCESS"
        elif provider_used:
            return "FALLBACK_SUCCESS"
        return "PRIMARY_SUCCESS"  # default for COMPLETE with no provider info
    if "timeout" in reason_code.lower() or "deadline" in reason_code.lower():
        return "TIMEOUT"
    return "DEGRADED"


def _classify_failure_signal_class(
    reason_code: str,
    *,
    rollback_occurred: bool = False,
) -> str:
    """Map a terminal reason into a coarse failure class for event consumers."""
    if rollback_occurred:
        return "rollback"
    reason = (reason_code or "").lower()
    if not reason:
        return "unknown"
    if any(token in reason for token in ("timeout", "deadline", "expired")):
        return "timeout"
    if any(token in reason for token in ("syntax", "indent")):
        return "syntax"
    if any(token in reason for token in ("validation", "verify", "test", "candidate", "source_drift")):
        return "validation"
    if any(token in reason for token in ("gate_blocked", "approval", "brain_not_admitted", "busy", "duplicate", "file_in_flight", "cost_gate")):
        return "policy"
    if any(token in reason for token in ("saga", "promote", "drift_detected")):
        return "saga"
    if any(token in reason for token in ("provider", "compute", "artifact", "capability", "host_binding", "dependency", "permission", "disk", "env", "unavailable")):
        return "env"
    if "change_engine" in reason or "apply" in reason:
        return "apply"
    if "l2_" in reason:
        return "repair"
    return "unknown"


def _build_proof_artifact(
    op_id: str,
    terminal_phase: "OperationPhase",
    terminal_class: str,
    provider_used: "str | None",
    model_id: "str | None",
    compute_class: "str | None",
    execution_host: "str | None",
    fallback_active: bool,
    phase_trail: "list[str]",
    generation_duration_s: float,
    total_duration_s: float,
) -> dict:
    """Build a structured proof artifact for a completed operation.

    This is written to the ledger and consumed by the observability layer.
    """
    return {
        "op_id": op_id,
        "terminal_phase": terminal_phase.name if hasattr(terminal_phase, "name") else str(terminal_phase),
        "terminal_class": terminal_class,
        "provider_used": provider_used,
        "model_id": model_id,
        "compute_class": compute_class,
        "execution_host": execution_host,
        "fallback_active": fallback_active,
        "phase_trail": phase_trail,
        "generation_duration_s": round(generation_duration_s, 3),
        "total_duration_s": round(total_duration_s, 3),
        "proof_ts_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# ServiceState
# ---------------------------------------------------------------------------


class ServiceState(Enum):
    """Lifecycle state of the GovernedLoopService."""

    INACTIVE = auto()
    STARTING = auto()
    ACTIVE = auto()
    DEGRADED = auto()
    STOPPING = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# OperationResult
# ---------------------------------------------------------------------------


MAX_TOOL_ROUNDS_ENV = "JARVIS_GOVERNED_TOOL_MAX_ROUNDS"

#: The tool-loop safety ceiling, in ONE place.
#:
#: This was resolved independently by the engine (default 15) and by the boot
#: panel that reports it to the operator (default 10). With the variable unset
#: — the default case — the panel announced a ceiling of 10 while the loop
#: allowed 15, so the number an operator read was not the number that governed
#: them. A display that re-derives a limit instead of reading it is a second
#: authority, and the operator only ever sees the second one.
_MAX_TOOL_ROUNDS_DEFAULT = 15


def configured_max_tool_rounds() -> int:
    """The live tool-round ceiling. NEVER raises; malformed input defaults."""
    raw = str(os.environ.get(MAX_TOOL_ROUNDS_ENV, "")).strip()
    if not raw:
        return _MAX_TOOL_ROUNDS_DEFAULT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _MAX_TOOL_ROUNDS_DEFAULT
    return value if value > 0 else _MAX_TOOL_ROUNDS_DEFAULT


@dataclass(frozen=True)
class OperationResult:
    """Stable result contract returned by submit().

    The full OperationContext stays internal/ledgered.  External callers
    see only this summary.
    """

    op_id: str
    terminal_phase: OperationPhase
    provider_used: Optional[str] = None
    generation_duration_s: Optional[float] = None
    total_duration_s: float = 0.0
    reason_code: str = ""
    trigger_source: str = "unknown"
    routing_reason: str = ""  # BrainSelectionResult.routing_reason; empty before brain selection
    terminal_class: str = "UNKNOWN"  # PRIMARY_SUCCESS | FALLBACK_SUCCESS | DEGRADED | TIMEOUT | NOOP


# ---------------------------------------------------------------------------
# ReadyToCommitPayload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadyToCommitPayload:
    """Terminal payload emitted when a governed op completes successfully.

    Contains all information needed for the human to decide whether to commit.
    """

    op_id: str
    changed_files: Tuple[str, ...]
    provider_id: str
    model_id: str
    routing_reason: str
    verification_summary: str
    rollback_status: str  # "clean" | "rolled_back" | "rollback_failed"
    suggested_commit_message: str


# ---------------------------------------------------------------------------
# Lazy helpers for optional L2 types
# ---------------------------------------------------------------------------


def _lazy_repair_budget_from_env() -> Any:
    """Lazily import RepairBudget and build it from environment variables.

    Using a module-level function (not a lambda) allows ``field(default_factory=...)``
    to reference it by name, satisfying frozen-dataclass requirements while
    avoiding a circular import at module load time.
    """
    from backend.core.ouroboros.governance.repair_engine import RepairBudget  # noqa: PLC0415
    return RepairBudget.from_env()


# ---------------------------------------------------------------------------
# GovernedLoopConfig
# ---------------------------------------------------------------------------


def _default_project_root() -> Path:
    """Authoritative repo root: ``.git``-anchored and cwd-independent, falling
    back to cwd only when no ``.git`` is found. SOURCE fix for the run-#14 path
    bug: ``os.getcwd()`` on the Linux node != the cloned repo root, so every
    consumer reading ``project_root`` (8 files, 45 live rejections) normalized
    scoped-test paths against the wrong root -> ``outside repo root`` -> the
    chaos test was never detected. Routing the DEFAULT through the resolver
    fixes all consumers at the source."""
    try:
        from backend.core.ouroboros.governance.workspace_resolver import (
            resolve_repo_root,
        )

        return resolve_repo_root()
    except Exception:  # noqa: BLE001 -- never break config construction
        return Path(os.getcwd())


@dataclass(frozen=True)
class GovernedLoopConfig:
    """Frozen configuration for the governed loop service."""

    project_root: Path = field(default_factory=_default_project_root)
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_cost_per_op: float = 0.50
    claude_daily_budget: float = 10.00
    generation_timeout_s: float = 180.0
    context_expansion_timeout_s: float = 30.0
    approval_timeout_s: float = 600.0
    health_probe_interval_s: float = 30.0
    max_concurrent_ops: int = 2
    initial_canary_slices: Tuple[str, ...] = ("tests/", "docs/")
    cold_start_grace_s: float = 300.0   # ops younger than this are not cancelled on boot
    approval_ttl_s: float = 1800.0      # stale approval expiry timeout
    pipeline_timeout_s: float = 600.0   # total wall-clock budget per submit(); env: JARVIS_PIPELINE_TIMEOUT_S

    # Curriculum + reactor event background task settings
    curriculum_enabled: bool = True
    curriculum_publish_interval_s: float = 3600.0
    curriculum_window_n: int = 50
    curriculum_top_k: int = 5
    curriculum_impact_weights: Dict[str, float] = field(default_factory=dict)
    model_attribution_lookback_n: int = 20
    model_attribution_min_sample_size: int = 3
    reactor_event_poll_interval_s: float = 30.0
    oracle_enabled: bool = True
    oracle_incremental_poll_interval_s: float = 300.0

    # L1 tool-use settings (Manifesto §6: tools enabled by default under
    # governance — Iron Gate + risk engine + approval gates are the safety net)
    tool_use_enabled: bool = True
    max_tool_rounds: int = _MAX_TOOL_ROUNDS_DEFAULT
    tool_timeout_s: float = 30.0
    max_concurrent_tools: int = 2
    # Budget-derived tool-loop timing (fixes bt-2026-04-10-045911 where
    # max_tool_rounds × tool_timeout_s could exceed the generation budget
    # and every IMMEDIATE op died with tool_loop_deadline_exceeded).
    # ``None`` means "use BudgetPlan.build defaults" which derive from
    # the actual generation budget at run() time.
    tool_min_per_round_s: Optional[float] = None
    tool_final_write_reserve_s: Optional[float] = None

    # L2 self-repair settings (RepairBudget drives the repair loop)
    repair_budget: Any = field(default_factory=_lazy_repair_budget_from_env)
    l3_enabled: bool = True  # Gap #5: worktree isolation enabled by default (Manifesto §6)
    max_concurrent_execution_graphs: int = 2
    execution_graph_state_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "ouroboros" / "execution_graphs"
    )
    l4_enabled: bool = False
    l4_state_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "ouroboros" / "advanced_coordination"
    )

    @property
    def execution_root(self) -> Path:
        """The mutation/judgment tree (Slice 11 role split).

        ``project_root`` is the OBSERVATION root (sensors, TestWatcher,
        intake — always the operator's real tree). ``execution_root`` is
        where APPLY writes and therefore where VERIFY/benchmark/rollback
        MUST judge — resolved lazily at every read through the canonical
        ``autonomous_workspace.effective_execution_root`` seam, because
        the ledger-sovereignty bootloader exports
        ``JARVIS_AUTO_COMMIT_WORKSPACE`` AFTER this frozen config is
        constructed (harness boot ordering). Never cache this value.
        """
        from backend.core.ouroboros.governance.autonomous_workspace import (
            effective_execution_root,
        )

        return effective_execution_root(self.project_root)

    @classmethod
    def from_env(cls, args: Any = None, project_root: Optional[Path] = None) -> GovernedLoopConfig:
        """Build config from environment variables with safe defaults.

        Resolution order (highest priority wins):
          1. Environment variables
          2. <repo_root>/.jarvis/governance.local.yaml
          3. <repo_root>/.jarvis/governance.yaml
          4. ~/.jarvis/governance.yaml  (global defaults)
          5. Hard-coded defaults below
        """
        import os
        from backend.core.ouroboros.governance.config_loader import load_layered_config

        resolved_root = (
            project_root if project_root is not None
            else Path(os.environ["JARVIS_PROJECT_ROOT"])
            if os.getenv("JARVIS_PROJECT_ROOT")
            else _default_project_root()
        )
        _yaml_cfg = load_layered_config(global_root=Path.home(), repo_root=resolved_root)

        def _cfg(key: str, env_var: str, default: str) -> str:
            env_val = os.environ.get(env_var)
            if env_val is not None:
                return env_val
            yaml_val = _yaml_cfg.get(key)
            if yaml_val is not None:
                return str(yaml_val)
            return default

        return cls(
            project_root=resolved_root,
            claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
            claude_model=os.getenv(
                "JARVIS_GOVERNED_CLAUDE_MODEL", "claude-sonnet-4-20250514"
            ),
            claude_max_cost_per_op=float(
                os.getenv("JARVIS_GOVERNED_CLAUDE_MAX_COST_PER_OP", "0.50")
            ),
            claude_daily_budget=float(
                os.getenv("JARVIS_GOVERNED_CLAUDE_DAILY_BUDGET", "10.00")
            ),
            generation_timeout_s=float(
                _cfg("generation_timeout_s", "JARVIS_GENERATION_TIMEOUT_S", "180")
            ),
            context_expansion_timeout_s=float(
                os.getenv("JARVIS_GOVERNED_EXPANSION_TIMEOUT", "30.0")
            ),
            approval_timeout_s=float(
                _cfg("approval_timeout_s", "JARVIS_APPROVAL_TIMEOUT_S", "600")
            ),
            health_probe_interval_s=float(
                os.getenv("JARVIS_GOVERNED_HEALTH_PROBE_INTERVAL", "30.0")
            ),
            max_concurrent_ops=int(
                _cfg("max_concurrent_ops", "JARVIS_GOVERNED_MAX_CONCURRENT_OPS", "2")
            ),
            cold_start_grace_s=float(os.environ.get("JARVIS_COLD_START_GRACE_S", "300")),
            approval_ttl_s=float(os.environ.get("JARVIS_APPROVAL_TTL_S", "1800")),
            pipeline_timeout_s=float(
                _cfg("pipeline_timeout_s", "JARVIS_PIPELINE_TIMEOUT_S", "600.0")
            ),
            tool_use_enabled=os.environ.get("JARVIS_GOVERNED_TOOL_USE_ENABLED", "true").lower() == "true",
            max_tool_rounds=configured_max_tool_rounds(),
            tool_timeout_s=float(os.environ.get("JARVIS_GOVERNED_TOOL_TIMEOUT_S", "30")),
            max_concurrent_tools=int(os.environ.get("JARVIS_GOVERNED_TOOL_MAX_CONCURRENT", "2")),
            tool_min_per_round_s=(
                float(os.environ["JARVIS_TOOL_LOOP_MIN_PER_ROUND_S"])
                if "JARVIS_TOOL_LOOP_MIN_PER_ROUND_S" in os.environ else None
            ),
            tool_final_write_reserve_s=(
                float(os.environ["JARVIS_TOOL_LOOP_FINAL_WRITE_RESERVE_S"])
                if "JARVIS_TOOL_LOOP_FINAL_WRITE_RESERVE_S" in os.environ else None
            ),
            repair_budget=_lazy_repair_budget_from_env(),
            l3_enabled=os.environ.get("JARVIS_GOVERNED_L3_ENABLED", "true").lower() == "true",
            # Bisection knob (B1, operator-bound 2026-05-14) — when "false",
            # the Oracle indexer background task is NOT spawned in start()
            # (see ``if self._config.oracle_enabled:`` ~1228).  Single-knob
            # falsification gate for "is the 29k-file scan the event-loop
            # offender behind Claude stream first_token=NEVER?"  Default
            # "true" preserves production byte-identically.  No new
            # substrate — composes the existing ``oracle_enabled`` dataclass
            # field that previously had no env path.
            oracle_enabled=os.environ.get("JARVIS_GOVERNED_ORACLE_INDEXER_ENABLED", "true").lower() == "true",
            max_concurrent_execution_graphs=int(
                os.environ.get("JARVIS_GOVERNED_L3_MAX_CONCURRENT_GRAPHS", "2")
            ),
            execution_graph_state_dir=Path(
                os.environ.get(
                    "JARVIS_GOVERNED_L3_STATE_DIR",
                    str(Path.home() / ".jarvis" / "ouroboros" / "execution_graphs"),
                )
            ),
            l4_enabled=os.environ.get("JARVIS_GOVERNED_L4_ENABLED", "false").lower() == "true",
            l4_state_dir=Path(
                os.environ.get(
                    "JARVIS_GOVERNED_L4_STATE_DIR",
                    str(Path.home() / ".jarvis" / "ouroboros" / "advanced_coordination"),
                )
            ),
        )


# ---------------------------------------------------------------------------
# GovernedLoopService
# ---------------------------------------------------------------------------


def _wrap_subagent_narration(gls: Any, inner: Any) -> Any:
    """Add cockpit narration to a CommSink. Returns `inner` unchanged on any
    failure — a missing narrator must never cost observability.
    """
    try:
        from backend.core.ouroboros.governance.subagent_narrator import (
            SubagentNarrationSink, narration_enabled,
        )
        if not narration_enabled():
            return inner

        def _emit(line: str) -> None:
            # Resolved PER EVENT: SerpentFlow attaches to the service after
            # this stack is constructed, so a handle captured at build time
            # is None forever. The same late-binding the alert emitter above
            # already uses.
            flow = getattr(gls, "_serpent_flow", None)
            mirror = getattr(flow, "_mirror_markup", None) if flow else None
            if mirror is not None:
                mirror(line)

        return SubagentNarrationSink(inner, _emit)
    except Exception:  # noqa: BLE001
        return inner



async def _maybe_await(value: Any) -> Any:
    """Await *value* only if it is awaitable. NEVER assumes.

    Oracle has TWO interchangeable implementations and they disagree about
    async-ness: `oracle.Oracle.get_metrics` is a plain `def` returning a dict,
    while `oracle_adapter`'s is `async def`. `self._oracle` may be either, so a
    hardcoded `await` is correct for one and fatal for the other — which is
    exactly what happened:

        [GovernedLoop] Oracle initialization failed:
          object dict can't be used in 'await' expression; codebase graph unavailable

    The whole codebase graph was discarded on every boot because a call site
    guessed which implementation it had. Deleting the `await` would only move
    the breakage to the other implementation; asking the VALUE whether it is
    awaitable works for both, and keeps working if a third arrives.
    """
    return await value if inspect.isawaitable(value) else value


class GovernedLoopService:
    """Lifecycle manager for the governed self-programming pipeline.

    No side effects in constructor. All async initialization in start().
    """

    @staticmethod
    def _build_hibernation_observability_hooks(
        stack: Any,
    ) -> Tuple[
        Callable[..., Awaitable[None]],
        Callable[..., Awaitable[None]],
    ]:
        """Factory for the step-7 hibernation observability hook pair.

        Produces two async hooks that share a closure-captured cycle
        counter and in-flight ``op_id``. Every enter/wake pair forms a
        single logical "hibernation cycle" in the CommProtocol message
        stream:

            HEARTBEAT(hibernation_enter, proactive_alert)
                → DECISION(hibernation_entered)
                → HEARTBEAT(hibernation_wake, proactive_alert)
                → DECISION(hibernation_wake)
                → POSTMORTEM

        Every message is fanned out through ``stack.comm._transports``,
        so SerpentFlow (via SerpentTransport) renders a proactive alert
        Panel, LogTransport writes to debug.log, and any dashboard
        transport picks up the event feed — all from a single code
        path. No subsystem is imported here; the indirection through
        CommProtocol keeps this layer decoupled from the battle-test
        harness.

        Parameters
        ----------
        stack:
            The governance stack (typically ``self._stack``). Must
            expose a ``.comm`` attribute that is a :class:`CommProtocol`
            — or ``None`` at hook-fire time, in which case the hook
            becomes a debug-logged no-op.

        Returns
        -------
        Tuple of ``(on_hibernate, on_wake)`` async callables. Each
        accepts a single keyword argument ``reason``. Exceptions from
        CommProtocol are logged and swallowed — observability must
        never block a lifecycle transition.
        """
        _stack_ref = stack
        # Mutable one-element lists so both closures share the cycle
        # counter and current op_id without needing a nonlocal block.
        _obs_cycle: List[int] = [0]
        _obs_current_op: List[str] = [""]

        async def _hibernate_obs(*, reason: str) -> None:
            if _stack_ref is None:
                return
            comm = getattr(_stack_ref, "comm", None)
            if comm is None:
                logger.debug(
                    "[GLS] hibernation obs: no comm on stack — skipping"
                )
                return
            _obs_cycle[0] += 1
            op_id = f"hibernation-{_obs_cycle[0]:03d}-{int(time.time())}"
            _obs_current_op[0] = op_id
            reason_text = reason or "unspecified"
            try:
                await comm.emit_heartbeat(
                    op_id=op_id,
                    phase="hibernation_enter",
                    progress_pct=0.0,
                    proactive_alert=True,
                    alert_title="HIBERNATING",
                    alert_body=(
                        f"Organism entering hibernation — {reason_text}"
                    ),
                    alert_severity="critical",
                    alert_source="provider_exhaustion",
                    hibernation_cycle=_obs_cycle[0],
                )
                await comm.emit_decision(
                    op_id=op_id,
                    outcome="hibernation_entered",
                    reason_code="provider_exhaustion",
                    diff_summary=reason_text,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[GLS] hibernation enter observability hook failed"
                )

        async def _wake_obs(*, reason: str) -> None:
            if _stack_ref is None:
                return
            comm = getattr(_stack_ref, "comm", None)
            if comm is None:
                logger.debug(
                    "[GLS] hibernation obs: no comm on stack — skipping"
                )
                return
            # Reuse the in-flight op_id so enter/wake share a seq space.
            # If wake fires without a prior enter (e.g. emergency_stop
            # racing), synthesize a standalone id so the sequence is
            # still well-formed.
            op_id = _obs_current_op[0] or (
                f"hibernation-wake-{int(time.time())}"
            )
            reason_text = reason or "unspecified"
            try:
                await comm.emit_heartbeat(
                    op_id=op_id,
                    phase="hibernation_wake",
                    progress_pct=100.0,
                    proactive_alert=True,
                    alert_title="RECOVERED",
                    alert_body=(
                        f"Provider substrate back online — {reason_text}"
                    ),
                    alert_severity="info",
                    alert_source="provider_recovery",
                    hibernation_cycle=_obs_cycle[0],
                )
                await comm.emit_decision(
                    op_id=op_id,
                    outcome="hibernation_wake",
                    reason_code="provider_recovery",
                    diff_summary=reason_text,
                )
                await comm.emit_postmortem(
                    op_id=op_id,
                    root_cause=reason_text,
                    failed_phase=None,
                    next_safe_action="resume_governed_loop",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[GLS] hibernation wake observability hook failed"
                )
            finally:
                _obs_current_op[0] = ""

        return _hibernate_obs, _wake_obs

    def __init__(
        self,
        stack: Any = None,
        prime_client: Any = None,
        config: Optional[GovernedLoopConfig] = None,
        active_brain_set: FrozenSet[str] = frozenset(),
        say_fn: Optional[Any] = None,
    ) -> None:
        self._stack = stack
        self._prime_client = prime_client
        self._say_fn = say_fn
        self._config = config if config is not None else GovernedLoopConfig.from_env()
        self._state = ServiceState.INACTIVE
        self._started_at: Optional[float] = None
        self._failure_reason: Optional[str] = None

        # Phase 4: admitted active brain set (published by supervisor post-handshake)
        # Empty frozenset = gate disabled (backward-compatible default)
        self._active_brain_set: FrozenSet[str] = active_brain_set

        # Phase 4: preemption FSM — initialized after ledger in start()
        self._fsm_engine: Optional[PreemptionFsmEngine] = None
        self._fsm_executor: Optional[PreemptionFsmExecutor] = None
        self._fsm_contexts: Dict[str, LoopRuntimeContext] = {}
        self._fsm_checkpoint_seq: Dict[str, int] = {}

        # GAP 6: user-initiated stop signal bus (created in _build_components)
        self._user_signal_bus: Optional[UserSignalBus] = None

        # Built during start()
        self._orchestrator: Optional[GovernedOrchestrator] = None
        self._generator: Optional[CandidateGenerator] = None
        self._approval_provider: Optional[CLIApprovalProvider] = None
        self._validation_runner: Optional[Any] = None
        self._health_probe_task: Optional[asyncio.Task] = None
        # Failover (Omni-Soak #3 fix): the FailoverLifecycleController FSM was
        # built + armed but NEVER ticked during the soak -> J-Prime never awoke
        # when DW collapsed. This task ticks the controller alongside the other
        # background loops so the bidirectional DW->J-Prime->DW failover RUNS.
        self._failover_task: Optional[asyncio.Task] = None
        self._failover_controller: Any = None
        # Layer 1 (Hybrid soak fix): the DWHeartbeat deep-probe loop, started as a
        # peer of the failover tick loop so is_degrading() is actually fed.
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._dw_heartbeat: Any = None
        # Meta-Goal aggregator wiring (the built-but-no-caller fix). The
        # aggregator (meta_goal_aggregator.py) bundles N disjoint single-file
        # ops into ONE fan-outable Meta-Goal DAG, but had no live caller — so
        # this drain loop ticks it alongside the failover loop and routes each
        # ready Meta-Goal into the EXISTING swarm fan-out path. Gated on
        # JARVIS_META_GOAL_AGGREGATOR_ENABLED (default OFF -> no task, no
        # aggregator wired, byte-identical).
        self._meta_goal_aggregator: Any = None
        self._meta_goal_drain_task: Optional[asyncio.Task] = None
        self._exhaustion_watcher: Any = None
        self._hibernation_prober: Any = None
        self._hibernate_bridge: Any = None
        self._wake_bridge: Any = None
        self._hibernate_obs_hook: Any = None
        self._wake_obs_hook: Any = None
        self._ledger: Any = None  # set in _build_components from stack.ledger
        self._repo_registry: Optional[Any] = None  # set in _build_components; reused by supervisor Zone 6.9
        self._trust_graduator: Optional[Any] = None

        # Phase 4: Brain selector — CAI-intent-aware async router (wraps BrainSelector)
        from backend.core.ouroboros.governance.route_decision_service import RouteDecisionService
        self._brain_selector = RouteDecisionService()

        # Sliding-window cooldown: maps file_path -> deque of touch timestamps (monotonic)
        self._file_touch_cache: Dict[str, Any] = {}  # str -> collections.deque[float]

        # Background task handles (curriculum + reactor event loop)
        self._curriculum_task: Optional[asyncio.Task] = None
        self._reactor_event_task: Optional[asyncio.Task] = None
        self._curriculum_publisher: Optional[CurriculumPublisher] = None
        self._model_attribution_recorder: Optional[ModelAttributionRecorder] = None
        self._performance_persistence: Optional[Any] = None
        self._event_dir: Optional[Path] = None
        self._oracle_indexer_task: Optional[asyncio.Task] = None
        self._oracle: Optional[Any] = None

        # YM-T10 SEAM 1 — Sovereign Daemon Injection Protocol (Layer 2).
        # Strong ref to the operator-presence watcher daemon so it is not GC'd
        # while running; cancelled in stop(). Spawn + attach are fail-soft and
        # no-op when JARVIS_OPERATOR_YIELD_ENABLED is off (byte-identical).
        self._operator_presence_task: Optional[asyncio.Task] = None

        # C+ autonomy infrastructure
        self._command_bus: Optional[CommandBus] = None
        self._event_emitter: Optional[EventEmitter] = None
        self._feedback_engine: Optional[AutonomyFeedbackEngine] = None
        self._command_consumer_task: Optional[asyncio.Task] = None
        self._feedback_loop_task: Optional[asyncio.Task] = None
        # C2 -- TransportCircuitBreaker HALF-OPEN probe daemon.
        # Periodically calls run_probe_if_due for each DW transport lane
        # so OPEN lanes self-heal. Default-OFF (JARVIS_TRANSPORT_BREAKER_ENABLED).
        self._transport_breaker_probe_task: Optional[asyncio.Task] = None
        self._safety_net: Optional[ProductionSafetyNet] = None
        self._subagent_scheduler: Optional[Any] = None
        # Gap #3 Slice 5 — hoisted from inner block so the
        # EventChannelServer + IDE observability router can
        # project the worktree topology in the GET surface. None
        # when l3 worktree isolation is disabled.
        self._worktree_manager: Optional[Any] = None
        # Miner graph coalescer — wired alongside the L3 scheduler below.
        self._graph_coalescer: Optional[Any] = None
        self._advanced_autonomy: Optional[Any] = None
        self._mcp_client: Optional[Any] = None  # Phase A: GovernanceMCPClient, wired in start()
        self._docker_ready: Optional[bool] = None  # set in start() by docker_preflight; None = not yet run

        # Compute-class admission gate (set externally after fetching /v1/capability;
        # None = gate disabled — backward-compatible default)
        self._vm_capability: Optional[dict] = None

        # Concurrency & dedup
        self._active_ops: Set[str] = set()
        # PRD §27.5: the reader `start()` lends to `why_engine`, held so
        # teardown can take back exactly what this instance gave — see
        # `_release_why_live_source`. Assigned here, before `start()`, so a
        # `/why` racing a failed boot finds an attribute rather than a
        # traceback.
        self._why_live_reader: Optional[Callable[[], Dict[str, Any]]] = None
        # PRD §28: the background audit-watchdog sweep, held so teardown can
        # cancel it. A scan outliving the service it reports on would keep
        # parsing thousands of files against a tree nobody is serving.
        self._audit_watchdog_task: Optional[Any] = None
        self._active_file_ops: Set[str] = set()  # canonical file paths currently in-flight
        self._completed_ops: Dict[str, OperationResult] = {}
        # Cooperative cancellation: op_ids requested for cancel via REPL /cancel
        self._cancel_requested: Set[str] = set()
        # W3(7) Slice 2 — per-op CancelToken registry. Slice 1 added the
        # primitive + Class D REPL emitter; Slice 2 attaches the registry
        # so the dispatcher / candidate_generator / tool_loop can look up
        # the in-flight token for an op. Master-flag-off: tokens are still
        # created (cheap) but never have ``set()`` called on them →
        # ``race()`` always returns the wrapped coro result → byte-for-byte
        # pre-W3(7) behavior. The REPL handler in serpent_flow.py looks
        # up this attribute (``_cancel_token_registry``) by name.
        from backend.core.ouroboros.governance.cancel_token import (
            CancelTokenRegistry as _CancelTokenRegistry,
        )
        self._cancel_token_registry = _CancelTokenRegistry()

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def active_brain_set(self) -> FrozenSet[str]:
        """Immutable snapshot of the supervisor-admitted brain set."""
        return self._active_brain_set

    @property
    def oracle(self):
        """TheOracle instance (None until oracle_index_loop completes initialization)."""
        return self._oracle

    @property
    def exploration_fleet(self):
        """ExplorationFleet reference (None if not yet wired or import failed)."""
        return getattr(self, "_exploration_fleet_ref", None)

    @property
    def background_pool(self):
        """BackgroundAgentPool reference (None if not yet wired)."""
        return self._bg_pool

    @property
    def doubleword_provider(self):
        """DoublewordProvider reference (None if API key not set or build failed)."""
        return getattr(self, "_doubleword_ref", None)

    def _resolve_provider_for_subagent(self, name):
        """Provider-registry callable passed to GENERAL's LLM driver factory.

        Maps the canonical provider name (as stamped on
        ``ctx.primary_provider_name`` by ``SubagentOrchestrator.dispatch``)
        to a live provider instance held on GLS. When the name is
        unrecognized or the referenced provider isn't wired, falls back
        to Claude (the "prefrontal cortex" per topology config, the
        best-fit default for GENERAL's bounded-agentic workloads).

        Called at run time by ``general_driver.run_general_tool_loop``;
        never cached so a runtime provider swap (e.g. after pool
        recycle) is picked up on the next dispatch.
        """
        lname = (str(name) or "").strip().lower()
        if "claude" in lname:
            return getattr(self, "_claude_ref", None)
        if "doubleword" in lname or lname.startswith("dw") or "qwen" in lname:
            return getattr(self, "_doubleword_ref", None)
        # Default: Claude — most GENERAL ops land on NOTIFY_APPLY tier
        # which routes IMMEDIATE (Claude direct) per topology config.
        # A None return triggers the driver's ``no_provider_wired``
        # structured trace — caller handles gracefully.
        return getattr(self, "_claude_ref", None)

    def set_active_brain_set(self, brain_set: FrozenSet[str]) -> None:
        """Update the admitted active brain set.

        Called by unified_supervisor after a successful boot handshake.
        The frozenset assignment is atomic under the GIL.
        """
        old = self._active_brain_set
        self._active_brain_set = brain_set
        logger.info(
            "[GovernedLoop] ActiveBrainSet updated: %s → %s",
            sorted(old),
            sorted(brain_set),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize providers, orchestrator, and canary slices.

        Idempotent — second call is no-op if already ACTIVE/DEGRADED/STARTING.
        On failure, sets state to FAILED with structured reason.
        Re-entrancy guard: raises RuntimeError if called concurrently.
        """
        if self._state in (ServiceState.ACTIVE, ServiceState.DEGRADED):
            return
        if self._state == ServiceState.STARTING:
            raise RuntimeError(
                "GLS re-entrancy detected: start() called while already STARTING. "
                "This would corrupt the FSM — aborting second call."
            )

        self._state = ServiceState.STARTING

        # Slice 212 — RUNTIME ATTESTATION GATE. Deliberately OUTSIDE the
        # fail-soft boot block below: a strict deployment-integrity mismatch
        # (image stamped with a different commit than the operator pinned,
        # dirty-tree build, or unstamped image) must HALT the boot — state →
        # FAILED, DeploymentIntegrityMismatch propagates — never be
        # logged-and-continued. Gated default-FALSE; UNPINNED is warn-only.
        # Guards against the 2026-06-10 drift class: a stale rebuild running
        # Slice-208 code while believed to be Slice-211.
        try:
            from backend.core.ouroboros.governance.runtime_attestation import (
                DeploymentIntegrityMismatch,
                enforce as _attest_enforce,
            )
        except Exception:  # noqa: BLE001
            DeploymentIntegrityMismatch = None  # type: ignore[assignment]
            _attest_enforce = None
        if _attest_enforce is not None:
            try:
                _attest_enforce()
            except DeploymentIntegrityMismatch:
                self._state = ServiceState.FAILED
                self._failure_reason = "deployment_integrity_mismatch"
                raise
            except Exception as _ax:  # noqa: BLE001
                # Only the explicit mismatch halts; infrastructure errors in
                # the attestation path itself must not brick a legacy boot.
                logger.debug("[GovernedLoop] attestation infra swallowed: %r", _ax)

        try:
            # Task #5 — GRADUATION-OVERRIDE BOOT APPLIER (un-sever the autonomy
            # bootstrap's output). The autonomous_graduation_engine records
            # AUTO_FLIP decisions "applied at next boot", but nothing at boot
            # ever read the ledger — apply_overrides_to_environ had ZERO callers,
            # so graduation was write-only theater and every default-off flag was
            # frozen transitively by this one missing call. Wire it FIRST in the
            # boot block, before any substrate reads its flag, so a graduated
            # STANDARD-tier flag is in effect for the rest of boot.
            #
            # Safe-by-default via THREE independent gates the applier already
            # owns: apply_enabled() default-FALSE (operator opt-in required);
            # shadow_mode_enabled() default-TRUE (the engine writes shadow
            # receipts the applier never reads); SAFETY-tier flags structurally
            # absent from the override ledger; operator env-precedence always
            # wins. NEVER raises into boot.
            try:
                from backend.core.ouroboros.governance.graduation_override_ledger import (  # noqa: E501
                    apply_overrides_to_environ as _apply_grad_overrides,
                )
                _grad_applied = _apply_grad_overrides()
                if _grad_applied:
                    logger.warning(
                        "[GovernedLoop] graduation boot applier activated %d "
                        "evidence-graduated flag(s): %s",
                        len(_grad_applied), ", ".join(_grad_applied),
                    )
            except Exception as _grad_exc:  # noqa: BLE001 — boot must never fail
                logger.debug(
                    "[GovernedLoop] graduation boot applier swallowed: %r",
                    _grad_exc,
                )

            # Slice 185 Phase 4 — purge DW learned-state corrupted by the NameError phantom
            # (surface-health + calibration learned from internal faults mislabeled as vendor
            # ruptures). Opt-in (JARVIS_DW_LEDGER_WIPE_ON_BOOT); NEVER raises.
            try:
                from backend.core.ouroboros.governance.dw_ledger_wipe import (
                    wipe_corrupted_dw_ledgers,
                )
                _wipe = wipe_corrupted_dw_ledgers()
                if _wipe.get("wiped"):
                    logger.warning(
                        "[GovernedLoop] DW ledger wipe (Slice 185): purged %d corrupted "
                        "learned-state file(s) — relearning from CLEAN signals: %s",
                        len(_wipe["wiped"]), _wipe["wiped"],
                    )
            except Exception as _wipe_exc:  # noqa: BLE001
                logger.debug("[GovernedLoop] DW ledger wipe swallowed: %r", _wipe_exc)

            # Task 5 -- boot-time READ path for the bi-directional cognitive
            # persistence organ. Fail-soft; disabled -> no-op, no PIM touch.
            try:
                from backend.core.ouroboros.governance import cognitive_persistence as _cogp
                if _cogp.is_enabled():
                    await _cogp.hydrate_prior_knowledge()
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    "[GLS] cognitive prior-knowledge hydration skipped (fail-soft): %s", _e
                )

            await self._build_components()

            # Task 9 -- async Docker pre-flight (only when A1 is armed -> byte-identical OFF)
            if os.environ.get("JARVIS_A1_TOKEN_ENFORCER_ENABLED", "false").strip().lower() in ("1", "true", "yes"):
                try:
                    from .pre_apply_exec_lock import docker_preflight
                    self._docker_ready = await docker_preflight()
                except Exception:  # noqa: BLE001
                    self._docker_ready = None

            # Phase 4: initialize preemption FSM executor (ledger available after _build_components)
            self._fsm_engine = PreemptionFsmEngine()
            if self._ledger is not None:
                comm = getattr(self._stack, "comm", None) if self._stack else None
                _sink = _CommTelemetrySink(comm) if comm is not None else None
                self._fsm_executor = PreemptionFsmExecutor(
                    engine=self._fsm_engine,
                    ledger=_FsmLedgerAdapter(self._ledger),
                    telemetry=_sink,
                )
                logger.debug("[GovernedLoop] Preemption FSM executor initialized")

            # Phase-Aware Heartbeats (Move 2 v4): register a stream-tick
            # callback that providers can pulse during long GENERATE
            # streams to keep the harness ActivityMonitor's freshness
            # signal accurate. Lookup is O(1) on the in-memory dict;
            # we update last_activity_at_utc *only* — never the
            # phase-transition timestamp, since no phase actually
            # advanced. Best-effort: failures (missing op, etc.) are
            # swallowed so a misbehaving provider can't kill generation.
            try:
                from backend.core.ouroboros.governance.providers import (
                    set_stream_activity_callback,
                )

                def _on_stream_tick(op_id: str) -> None:
                    if not op_id:
                        return
                    ctx = self._fsm_contexts.get(op_id)
                    if ctx is None:
                        return
                    try:
                        ctx.last_activity_at_utc = datetime.now(timezone.utc)
                    except Exception:  # noqa: BLE001
                        pass

                set_stream_activity_callback(_on_stream_tick)
                logger.info(
                    "[GovernedLoop] Stream-tick activity hook registered "
                    "(Phase-Aware Heartbeats live)"
                )
            except Exception as _exc:  # noqa: BLE001
                logger.warning(
                    "[GovernedLoop] Failed to register stream-tick hook: %s",
                    _exc,
                )

            # Fetch and cache VM capability contract
            if self._prime_client is not None:
                try:
                    cap = await self._prime_client.fetch_capability()
                    self._vm_capability = cap
                    logger.info(
                        "[GLS] VM capability: compute_class=%s model=%s host=%s gpu_layers=%s tok_s=%s",
                        cap.get("compute_class"), cap.get("model_id"),
                        cap.get("host"), cap.get("gpu_layers"), cap.get("tok_s_estimate"),
                    )

                    # Boot-time hard-fail: verify VM compute_class satisfies the
                    # default (tier-1) brain's requirements before completing startup.
                    # Attribute path confirmed from per-op gate at ~line 1079:
                    #   self._brain_selector           -> RouteDecisionService
                    #   ._brain_selector               -> BrainSelector
                    #   ._policy                       -> dict loaded from brain_selection_policy.yaml
                    try:
                        _boot_policy = getattr(
                            getattr(self._brain_selector, "_brain_selector", None),
                            "_policy", {},
                        ) or {}
                        _tier1_brains = (
                            _boot_policy.get("routing", {})
                            .get("task_class_map", {})
                            .get("tier1", [])
                        )
                        _default_brain_id = _tier1_brains[0] if _tier1_brains else None
                        if _default_brain_id:
                            _all_entries = (
                                _boot_policy.get("brains", {}).get("required", [])
                                + _boot_policy.get("brains", {}).get("optional", [])
                            )
                            _boot_brain_cfg: dict = {}
                            for _e in _all_entries:
                                if isinstance(_e, dict):
                                    _bid = _e.get("brain_id") or _e.get("id")
                                    if _bid == _default_brain_id:
                                        _boot_brain_cfg = {k: v for k, v in _e.items() if k not in ("brain_id", "id")}
                                        break
                            if _boot_brain_cfg:
                                # Resolve this host's accelerator ONCE, here, before
                                # the first admission question is asked. Load-bearing,
                                # not an optimisation: _check_compute_admission runs
                                # inside this running loop, and compute_topology's
                                # sync facade refuses to block a loop (§3) — so with
                                # no cache the measured path would never engage and
                                # the whole gate would be wired-but-inert. Bounded and
                                # fail-soft: a wedged driver costs its budget, then
                                # admission falls back to the ordinal comparison.
                                try:
                                    from backend.core.ouroboros.governance import (
                                        compute_topology as _ct_boot,
                                    )
                                    if _ct_boot.is_enabled():
                                        _ct_reading = await _ct_boot.prewarm()
                                        logger.info(
                                            "[GLS] compute topology: %s %s "
                                            "(%.1f GiB usable, source=%s)",
                                            _ct_reading.topology.value,
                                            _ct_reading.resolved_class,
                                            _ct_reading.usable_bytes / (1024 ** 3),
                                            _ct_reading.source,
                                        )
                                except Exception:  # noqa: BLE001
                                    logger.debug(
                                        "[GLS] compute topology prewarm skipped",
                                        exc_info=True,
                                    )

                                # Boot-time: only gate on compute class (does VM have
                                # the minimum GPU tier?).  Artifact integrity is checked
                                # per-operation in _preflight_check() where we know
                                # exactly which brain is being routed to — validating
                                # the tier-1 default brain's artifact at boot would
                                # hard-fail whenever the VM has a different model loaded
                                # (e.g. GPU VM running qwen-7B while tier1 default is
                                # phi3-1B).
                                _check_compute_admission(_boot_brain_cfg, cap)
                                logger.info(
                                    "[GLS] Boot-time compute-class validation passed for brain=%s",
                                    _default_brain_id,
                                )
                    except ComputeClassMismatch as exc:
                        logger.error("[GLS] Boot-time compute-class validation FAILED: %s", exc)
                        raise  # hard fail — do not complete startup below minimum compute class

                except ComputeClassMismatch:
                    raise  # propagate hard-fail boot validation errors
                except Exception as exc:
                    logger.warning("[GLS] Could not fetch capability (non-fatal): %s", exc)
                    self._vm_capability = None

            await self._reconcile_on_boot()  # boot reconciliation
            self._register_canary_slices()
            self._seed_autonomy_policies()
            self._attach_to_stack()
            self._started_at = time.monotonic()

            # T4 — Sovereign Telemetry Boot-Guard: emit a loud [SOVEREIGN WARNING]
            # when the operator has explicitly disabled the egress interceptor.
            # Default-ON (guard is active), so this fires only on an opt-out.
            # Fail-soft — never raises, never blocks boot.
            _warn_if_egress_guard_disabled(logger)

            # Wire curriculum and reactor event background tasks
            if self._config.curriculum_enabled:
                event_dir = Path(os.environ.get(
                    "JARVIS_REACTOR_EVENT_DIR",
                    str(Path.home() / ".jarvis" / "reactor_events"),
                ))
                event_dir.mkdir(parents=True, exist_ok=True)
                self._event_dir = event_dir
                persistence = get_performance_persistence()
                self._performance_persistence = persistence
                self._curriculum_publisher = CurriculumPublisher(
                    persistence=persistence,
                    event_dir=event_dir,
                    window_n=self._config.curriculum_window_n,
                    top_k=self._config.curriculum_top_k,
                    impact_weights=self._config.curriculum_impact_weights,
                )
                self._model_attribution_recorder = ModelAttributionRecorder(
                    persistence=persistence,
                    lookback_n=self._config.model_attribution_lookback_n,
                    min_sample_size=self._config.model_attribution_min_sample_size,
                )
                self._curriculum_task = asyncio.create_task(
                    self._curriculum_loop(), name="curriculum_loop"
                )
                self._reactor_event_task = asyncio.create_task(
                    self._reactor_event_loop(), name="reactor_event_loop"
                )

            if self._config.oracle_enabled:
                self._oracle_indexer_task = asyncio.create_task(
                    self._oracle_index_loop(), name="oracle_index_loop"
                )

            # Start health probe background task
            self._health_probe_task = asyncio.create_task(
                self._health_probe_loop(), name="health_probe_loop"
            )

            # C2 -- TransportCircuitBreaker HALF-OPEN probe daemon.
            # Only started when the master gate is ON (default OFF, byte-identical
            # when disabled). The daemon calls run_probe_if_due periodically so
            # OPEN batch/realtime lanes self-heal instead of staying OPEN forever.
            try:
                from backend.core.ouroboros.governance.transport_circuit_breaker import (
                    breaker_enabled as _tcb_enabled,
                )
                if _tcb_enabled():
                    self._transport_breaker_probe_task = asyncio.create_task(
                        self._transport_breaker_probe_loop(),
                        name="transport_breaker_probe_loop",
                    )
                    logger.info(
                        "[GovernedLoop] C2 TransportCircuitBreaker probe "
                        "daemon started (JARVIS_TRANSPORT_BREAKER_ENABLED=true)",
                    )
            except Exception as _tcb_exc:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] C2 transport breaker probe daemon "
                    "skipped (import/gate error): %r", _tcb_exc,
                )

            # Omni-Soak #3 fix — start the FailoverLifecycleController tick
            # loop alongside the other background daemons. The FSM
            # (DORMANT->AWAKENING->SERVING->handback) was BUILT + armed but
            # NEVER ticked, so J-Prime never awoke when DW collapsed. This
            # makes the bidirectional DW->J-Prime->DW failover actually RUN.
            # Gated on JARVIS_FAILOVER_LIFECYCLE_ENABLED (OFF -> no task,
            # byte-identical). Fail-soft: never blocks/raises into boot.
            self._start_failover_loop()

            # Pre-Flight Init Barrier (Omni-Soak v5/v6 fix) — INGEST the
            # SHA256-validated oracle_prewarm.json and WARM the shared Oracle
            # handle BEFORE the drain/flush loops below are scheduled. This is
            # an ABSOLUTE BLOCKING boot step: the JIT pre-warm was wired into
            # the drain path (dispatch_ready_bundles -> prewarm_window) but ops
            # exit via the OTHER path (_flush_aged_ops), which ran its OWN COLD
            # disjointness check and flushed them to legacy ("no disjoint
            # sibling found") BEFORE the drain's pre-warm fired. Pre-warming is
            # an INITIALIZATION event, not a runtime-loop event -> warm the
            # Oracle here so BOTH _flush_aged_ops AND drain see DISJOINT on the
            # first tick. Gated on JARVIS_ORACLE_SELF_WARMING_ENABLED (OFF ->
            # no-op, byte-identical). Fail-soft: missing/mismatch payload logs
            # + boot proceeds (runtime JIT remains the cold-miss fallback).
            await self._prewarm_oracle_barrier()

            # Built-but-no-caller fix — start the Meta-Goal aggregator drain
            # loop alongside the failover loop. The aggregator bundles N
            # disjoint pooled single-file ops into ONE fan-outable Meta-Goal
            # DAG, but nothing fed it / dispatched its output, so isolated
            # single-file ops dispatched serially (single_file_op, no fan-out).
            # Gated on JARVIS_META_GOAL_AGGREGATOR_ENABLED (OFF -> no task,
            # byte-identical). Fail-soft: never blocks/raises into boot.
            self._start_meta_goal_drain_loop()

            # Slice 5 Arc A — start PostureObserver so SensorGovernor's
            # default_posture_fn sees live readings. Without this, posture
            # weights collapse to 1.0 and the governor becomes posture-
            # blind (captured as soak finding #1 on 2026-04-21).
            # Idempotent: get_default_observer() is a singleton. Failures
            # are logged but never raise — posture observation is advisory.
            try:
                from backend.core.ouroboros.governance.direction_inferrer import (
                    is_enabled as _di_enabled,
                )
                if _di_enabled():
                    from backend.core.ouroboros.governance.posture_observer import (
                        get_default_observer,
                    )
                    self._posture_observer = get_default_observer(
                        Path(os.getcwd()),
                    )
                    self._posture_observer.start()
                    logger.info(
                        "[GovernedLoop] PostureObserver started (Slice 5 Arc A)",
                    )
                    # Install SSE bridges for Wave 1 #3 observability —
                    # best-effort, never-raise pattern.
                    try:
                        from backend.core.ouroboros.governance.ide_observability_stream import (
                            bridge_governor_to_broker,
                            bridge_memory_pressure_to_broker,
                            bridge_posture_to_broker,
                        )
                        bridge_posture_to_broker(observer=self._posture_observer)
                        bridge_governor_to_broker()  # uses default singleton
                        bridge_memory_pressure_to_broker()  # uses default singleton
                        logger.info(
                            "[GovernedLoop] Wave 1 SSE bridges installed "
                            "(posture + governor + memory-pressure)",
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "[GovernedLoop] SSE bridge install failed "
                            "(non-fatal)", exc_info=True,
                        )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GovernedLoop] PostureObserver startup failed (non-fatal)",
                    exc_info=True,
                )
                self._posture_observer = None

            # Tier 0.5 batch 1 — boot-wire the 3 dormant Slice 5b observers
            # (InvariantDrift / Coherence / CIGW). Each has shipped graduated
            # default-True for weeks but never .start()ed in production —
            # caught by the codebase audit as ~5,000 LOC of inert substrate.
            # Master-flag-gated by each observer's own substrate flags;
            # fail-open per observer; never blocks the loop.
            await self._start_governance_observers()

            # YM-T10 SEAM 1 — non-blocking daemon boot of the Layer-2
            # operator-presence watcher + yield bridge. The pool (_bg_pool) is
            # built in _build_components() earlier in start(); the bus self-
            # resolves via get_event_bus_if_exists() (bus=None). Both the
            # watcher .run() and bridge .attach() are already no-op when
            # JARVIS_OPERATOR_YIELD_ENABLED is off, so this is byte-identical
            # when the flag is off. Hard requirement: the watcher/attach must
            # NEVER prevent or crash boot — the helper wraps both fail-soft.
            await self._start_operator_yield_layer()

            # Wave 3 (6) Slice 5a — register parallel-dispatch env flags into
            # the FlagRegistry so `/help flags --search parallel_dispatch`
            # surfaces all 5 knobs. Best-effort, never-raise (the env reads
            # work without the registry).
            try:
                from backend.core.ouroboros.governance.parallel_dispatch import (
                    ensure_flag_registry_seeded as _w3_seed,
                )
                _w3_seed()
                logger.info(
                    "[GovernedLoop] Wave 3 (6) FlagRegistry seed installed "
                    "(parallel-dispatch knobs discoverable via /help flags)",
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] Wave 3 (6) FlagRegistry seed failed "
                    "(non-fatal)", exc_info=True,
                )

            # A1 graduation-flag boot telemetry (DRY): one centralized hook emits
            # a structured [A1FlagAudit] block (CADENCE_POLICY flags + live state)
            # so the A1 auditor credits each flag observed_evaluated. Lands in the
            # session debug.log via the configured logger. Best-effort, never-raise.
            try:
                from backend.core.ouroboros.governance.flag_registry import (
                    emit_a1_graduation_telemetry as _a1_flag_telemetry,
                )
                _n = _a1_flag_telemetry()
                if _n:
                    logger.info(
                        "[GovernedLoop] A1 graduation-flag telemetry emitted "
                        "(%d flags attested for the audit)", _n,
                    )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] A1 graduation-flag telemetry failed "
                    "(non-fatal)", exc_info=True,
                )

            # C+ L2/L3: CommandBus + EventEmitter + optional subagent scheduler
            if self._command_bus is None:
                self._command_bus = CommandBus(maxsize=1000)
            if self._event_emitter is None:
                self._event_emitter = EventEmitter()
            fe_config = FeedbackEngineConfig(
                event_dir=self._event_dir or Path.home() / ".jarvis" / "reactor_events",
                state_dir=Path(os.environ.get(
                    "JARVIS_AUTONOMY_STATE_DIR",
                    str(Path.home() / ".jarvis" / "ouroboros" / "state"),
                )),
            )
            self._feedback_engine = AutonomyFeedbackEngine(
                command_bus=self._command_bus,
                config=fe_config,
                event_emitter=self._event_emitter,
            )
            self._feedback_engine.register_event_handlers(self._event_emitter)
            self._feedback_loop_task = asyncio.create_task(
                self._feedback_loop(), name="feedback_loop"
            )
            self._command_consumer_task = asyncio.create_task(
                self._command_consumer_loop(), name="command_consumer_loop"
            )

            # C+ L3: ProductionSafetyNet
            self._safety_net = ProductionSafetyNet(
                command_bus=self._command_bus,
                config=SafetyNetConfig(),
            )
            self._safety_net.register_event_handlers(self._event_emitter)
            if self._subagent_scheduler is not None:
                await self._subagent_scheduler.start()
                await self._subagent_scheduler.recover_inflight()

            # Determine state based on provider availability
            if self._generator is not None:
                fsm_state = self._generator.fsm.state
                if fsm_state is FailbackState.QUEUE_ONLY:
                    self._state = ServiceState.DEGRADED
                elif fsm_state is FailbackState.FALLBACK_ACTIVE:
                    # Intentional GCP-first fallback — not degraded
                    self._state = ServiceState.ACTIVE
                else:
                    self._state = ServiceState.ACTIVE
            else:
                self._state = ServiceState.DEGRADED

            logger.info(
                "[GovernedLoop] Started: state=%s, canary_slices=%s",
                self._state.name,
                self._config.initial_canary_slices,
            )

            # P2 Slice 3 — boot the universal Convergence Reaper.
            # Master-gated NEVER-raise; when off, this is a no-op
            # and the loop's start path stays byte-identical.
            # When on, the reaper's background task begins
            # sweeping the typed registry, force-converging any
            # op past its deadline / ceiling and emitting
            # ``operation_terminal`` SSE events so observers
            # never lose visibility on a hung op.
            try:
                from backend.core.ouroboros.governance.convergence_reaper import (  # noqa: E501
                    safe_start_default_reaper,
                )
                if safe_start_default_reaper():
                    logger.info(
                        "[GovernedLoop] Convergence Reaper "
                        "started (universal terminal "
                        "invariant active)",
                    )
            except Exception as _cr_exc:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] Convergence Reaper boot "
                    "swallowed: %r", _cr_exc,
                )

            # Slice 177 — autonomous workspace hygiene. Fire a DEFERRED, once-per-boot
            # artifact sweep (compress aging logs, prune ancient artifacts) on a WORKER
            # THREAD via asyncio.to_thread, so the blocking file I/O never touches the
            # event loop or contends the GIL on the hot path. Gated default-FALSE
            # (§33.1 — it deletes files); the active session dir is explicitly protected
            # (and the age-gate already shields recent/live files). NEVER raises.
            try:
                from backend.core.ouroboros.governance.artifact_janitor import (  # noqa: E501
                    artifact_janitor_enabled,
                    ArtifactJanitor,
                    emit_maintenance_eviction,
                )
                if artifact_janitor_enabled():
                    import asyncio as _aio_jan
                    _protect = []
                    _sess = getattr(self, "_session_dir", None) \
                        or os.environ.get("JARVIS_OUROBOROS_SESSION_DIR", "").strip()
                    if _sess:
                        _protect.append(str(_sess))

                    async def _janitor_boot_sweep() -> None:
                        try:
                            _rep = await _aio_jan.to_thread(
                                ArtifactJanitor(protect_paths=_protect).sweep
                            )
                            logger.info(
                                "[GovernedLoop] artifact janitor: usage=%.0f%% evicted=%s "
                                "compressed=%s deleted=%s freed=%.1fMB errors=%s",
                                _rep.get("usage_ratio", 0) * 100, _rep.get("evicted"),
                                _rep.get("compressed"), _rep.get("deleted"),
                                _rep.get("freed_bytes", 0) / 1e6, _rep.get("errors"),
                            )
                            # Slice 178 — on an actual eviction, push MAINTENANCE_EVICTION
                            # to the Discord spine (webhook; best-effort, off-thread).
                            if _rep.get("evicted") and _rep.get("freed_bytes", 0) > 0:
                                await _aio_jan.to_thread(
                                    emit_maintenance_eviction,
                                    _rep.get("usage_ratio", 0.0),
                                    _rep.get("freed_bytes", 0),
                                )
                        except Exception as _je:  # noqa: BLE001
                            logger.debug("[GovernedLoop] janitor sweep swallowed: %r", _je)

                    self._janitor_task = _aio_jan.create_task(_janitor_boot_sweep())
                    logger.info(
                        "[GovernedLoop] artifact janitor: deferred boot sweep scheduled "
                        "(off-thread; active session protected)",
                    )
            except Exception as _jx:  # noqa: BLE001
                logger.debug("[GovernedLoop] artifact janitor boot swallowed: %r", _jx)

            # Slice 200 — Genesis Proposal. A SINGLE-USE, gated (default-FALSE),
            # fail-soft boot trigger that deterministically proves the full
            # code-shipping highway exactly once: build an honest architecture
            # doc → taste-check → open ONE review PR (APPROVAL_REQUIRED /
            # DO-NOT-AUTO-MERGE) → write a durable sentinel so it never fires
            # again (the bind-mounted .jarvis carries the sentinel across every
            # restart, so restart:always can't spam PRs). Deferred create_task,
            # never blocks boot. Mirrors the janitor one-shot precedent above.
            try:
                from backend.core.ouroboros.governance.genesis_proposal import (
                    genesis_enabled,
                    genesis_already_shipped,
                    run_genesis_proposal,
                )
                if genesis_enabled() and not genesis_already_shipped():
                    import asyncio as _aio_gen

                    async def _genesis_boot_trigger() -> None:
                        try:
                            _res = await run_genesis_proposal()
                            if _res:
                                logger.warning(
                                    "[GovernedLoop] genesis milestone PR shipped: "
                                    "%s — single-use trigger dissolved",
                                    _res.get("pr_url"),
                                )
                        except Exception as _ge:  # noqa: BLE001
                            logger.debug(
                                "[GovernedLoop] genesis trigger swallowed: %r", _ge,
                            )

                    self._genesis_task = _aio_gen.create_task(
                        _genesis_boot_trigger(),
                    )
                    logger.info(
                        "[GovernedLoop] genesis proposal: single-use boot "
                        "trigger scheduled (gated, deferred, fail-soft)",
                    )
            except Exception as _gx:  # noqa: BLE001
                logger.debug("[GovernedLoop] genesis wiring swallowed: %r", _gx)

            # Slice 203 — Strategy Simulator. A gated (default-FALSE), fail-soft
            # boot trigger: read the registry telemetry → synthesize fitness-
            # ranked remediation goals → bundle into ONE operator-review PR
            # ([Ouroboros Strategic Proposal], DO-NOT-AUTO-MERGE). Deduped by
            # deficiency-set fingerprint (the bind-mounted marker persists), so
            # restart:always opens a new PR only when the proposal actually
            # changes. PROPOSE-don't-dispose: writes only the .draft file, never
            # signs, never writes the active roadmap — the operator elevates via
            # strategy_signer. Deferred create_task; never blocks boot.
            try:
                from backend.core.ouroboros.governance.strategy_simulator import (
                    simulator_enabled as _sim_enabled,
                    propose_via_pr as _sim_propose,
                )
                if _sim_enabled():
                    import asyncio as _aio_sim

                    async def _strategy_sim_boot_trigger() -> None:
                        try:
                            _snap = {}
                            try:
                                from backend.core.ouroboros.governance.observability_registry import (  # noqa: E501
                                    get_observability_registry as _sim_reg,
                                )
                                _snap = _sim_reg().snapshot()
                            except Exception:  # noqa: BLE001
                                _snap = {}
                            _res = await _sim_propose(snapshot=_snap)
                            if _res:
                                logger.warning(
                                    "[GovernedLoop] strategy proposal PR opened "
                                    "(%d goals): %s",
                                    _res.get("goals"), _res.get("pr_url"),
                                )
                        except Exception as _se:  # noqa: BLE001
                            logger.debug(
                                "[GovernedLoop] strategy sim swallowed: %r", _se,
                            )

                    self._strategy_sim_task = _aio_sim.create_task(
                        _strategy_sim_boot_trigger(),
                    )
                    logger.info(
                        "[GovernedLoop] strategy simulator: telemetry-driven "
                        "proposal trigger scheduled (gated, deferred, deduped)",
                    )
            except Exception as _sx:  # noqa: BLE001
                logger.debug("[GovernedLoop] strategy sim wiring swallowed: %r", _sx)

            # Slice 204 — Chronos Continuity Matrix. Decouples operational
            # history from volatile container memory: re-chain the disk-backed
            # ledger on boot (preserve evolutionary history; chain the strict
            # unsupervised interval only across an UNSUPERVISED recovery — a
            # supervised rebuild resets it), then a 60s heartbeat accrues true
            # running time (sleep-frozen wall-clock excluded). Gated, fail-soft.
            try:
                from backend.core.ouroboros.governance.chronos_ledger import (
                    chronos_enabled as _chronos_on,
                    get_chronos_ledger as _chronos_get,
                    heartbeat_interval_s as _chronos_hb_s,
                )
                if _chronos_on():
                    import asyncio as _aio_chr
                    import time as _t_chr
                    _image = os.environ.get("JARVIS_SOAK_IMAGE_ID", "").strip() \
                        or os.environ.get("HOSTNAME", "").strip() or "local"
                    _chr_led = _chronos_get()
                    _chr_snap = _chr_led.rechain_on_boot(
                        now_unix=_t_chr.time(), image_id=_image,
                    )
                    logger.warning(
                        "[GovernedLoop] Chronos re-chained: boot=%d event=%s "
                        "total_days=%.3f unsupervised_days=%.3f",
                        _chr_snap.get("boot_count"), _chr_snap.get("last_event"),
                        _chr_snap.get("total_operational_days", 0.0),
                        _chr_snap.get("unsupervised_interval_days", 0.0),
                    )

                    # Proactive Cross-Space Coordinator (2026-07-19):
                    # rides the SAME heartbeat cadence — no new poll
                    # loop. Master-gated OFF; a headless daemon (no
                    # windowserver) is a silent no-op. Fail-soft: a
                    # coordinator fault never touches the heartbeat.
                    _proactive_coord = None
                    try:
                        from backend.core.ouroboros.governance.comms.duplex.proactive_coordinator import (  # noqa: E501
                            ProactiveCrossSpaceCoordinator,
                            build_proactive_sink,
                            native_windows_by_space,
                            proactive_enabled as _proactive_on,
                        )
                        if _proactive_on():
                            # WIRE 2 — present sink → the EXISTING
                            # SerpentFlow [Y/n] alert surface; WIRE 3 —
                            # approved reconciliations graduate into the
                            # backlog via the unified intake router
                            # under SignalSource.CROSS_SPACE.
                            def _emit_alert(**kw: Any) -> None:
                                flow = getattr(self, "_serpent_flow", None)
                                if flow is not None and hasattr(
                                    flow, "emit_proactive_alert",
                                ):
                                    flow.emit_proactive_alert(**kw)

                            def _backlog_emit(
                                summary: str, spaces: list,
                            ) -> None:
                                try:
                                    from backend.core.ouroboros.governance.intent.signals import (  # noqa: E501
                                        SignalSource,
                                    )
                                    router = getattr(
                                        self, "_intake_router", None,
                                    )
                                    submit = getattr(
                                        router, "submit_backlog_intent", None,
                                    ) or getattr(router, "emit", None)
                                    if submit is not None:
                                        submit(
                                            summary,
                                            source=SignalSource.CROSS_SPACE,
                                        )
                                except Exception:  # noqa: BLE001
                                    pass

                            _proactive_coord = ProactiveCrossSpaceCoordinator(
                                windows_source=native_windows_by_space,
                                present_sink=build_proactive_sink(
                                    _emit_alert, backlog_emit=_backlog_emit,
                                ),
                            )
                            self._proactive_coordinator = _proactive_coord
                            logger.info(
                                "[GovernedLoop] proactive cross-space "
                                "coordinator armed on heartbeat cadence",
                            )
                    except Exception:  # noqa: BLE001
                        _proactive_coord = None

                    async def _chronos_heartbeat_loop() -> None:
                        _iv = _chronos_hb_s()
                        while True:
                            try:
                                await _aio_chr.sleep(_iv)
                                _chr_led.heartbeat(
                                    now_unix=_t_chr.time(),
                                    now_monotonic=_t_chr.monotonic(),
                                )
                                if _proactive_coord is not None:
                                    _proactive_coord.tick()
                            except _aio_chr.CancelledError:
                                break
                            except Exception:  # noqa: BLE001
                                pass

                    self._chronos_task = _aio_chr.create_task(
                        _chronos_heartbeat_loop(),
                    )

                    # Residency Telemetry (2026-07-19, pre-soak): the
                    # loop instruments ITSELF — RSS / UDS conns / loop
                    # lag every 5 min → bounded rotating JSONL. Own
                    # task on the Orchestrator lifecycle; a leak shows
                    # as monotone rss_delta over the 24h soak.
                    try:
                        from backend.core.ouroboros.governance.residency_telemetry import (  # noqa: E501
                            ResidencyTelemetry,
                        )

                        def _uds_conns() -> int:
                            n = 0
                            for _attr in (
                                "_cockpit_attach_bridge", "audio_ipc",
                            ):
                                _b = getattr(self, _attr, None)
                                n += int(getattr(_b, "client_count", 0) or 0)
                            return n

                        self._residency_telemetry = ResidencyTelemetry(
                            conn_source=_uds_conns,
                        )
                        self._residency_telemetry.start()
                    except Exception:  # noqa: BLE001
                        self._residency_telemetry = None
                    logger.info(
                        "[GovernedLoop] Chronos heartbeat scheduled (%.0fs; "
                        "non-volatile uptime ledger active)", _chronos_hb_s(),
                    )
            except Exception as _cx:  # noqa: BLE001
                logger.debug("[GovernedLoop] chronos wiring swallowed: %r", _cx)

            # Slice 206 — Boot-Warmup Lifecycle + proactive off-loop warmup.
            # Enter BOOT_WARMUP so the watchdog records the one-time heavy-init
            # lag as warmup_lag (visible, benign) instead of polluting the
            # steady-state starvation metric. Then PROACTIVELY pre-warm the
            # heavy builds via their EXISTING thread-offloaded paths
            # (build_offloaded / build_async) so they never block the loop on
            # first lazy use. On completion → STEADY_STATE + WARMUP_COMPLETE.
            # The init_lifecycle hard deadline guarantees warmup can't be
            # claimed forever to mask real starvation. Gated, deferred, fail-soft.
            try:
                from backend.core.ouroboros.governance.init_lifecycle import (
                    init_lifecycle_enabled as _il_on,
                    start_warmup as _il_start,
                    mark_warmup_complete as _il_done,
                )
                if _il_on():
                    import asyncio as _aio_il
                    _il_start()
                    logger.info(
                        "[InitializationGuard] BOOT_WARMUP entered — heavy init "
                        "lag recorded as warmup_lag (not steady-state starvation)",
                    )

                    async def _warmup_lifecycle_boot() -> None:
                        # Concurrently pre-warm via the pre-existing offloaded
                        # builds (each best-effort; a missing/erroring warmer
                        # never blocks the transition).
                        async def _warm_semantic() -> None:
                            try:
                                from backend.core.ouroboros.governance.goal_inference import (  # noqa: E501
                                    get_default_engine,
                                )
                                eng = get_default_engine()
                                if eng is not None and hasattr(eng, "build_offloaded"):
                                    await eng.build_offloaded(force=True)
                            except Exception as _we:  # noqa: BLE001
                                logger.debug("[InitGuard] semantic warm: %r", _we)

                        async def _warm_index() -> None:
                            try:
                                from backend.core.ouroboros.governance.semantic_index import (  # noqa: E501
                                    get_default_index,
                                )
                                idx = get_default_index()
                                if idx is not None and hasattr(idx, "build_async"):
                                    idx.build_async()  # schedules off-loop build
                            except Exception as _we:  # noqa: BLE001
                                logger.debug("[InitGuard] index warm: %r", _we)

                        try:
                            await _aio_il.gather(
                                _warm_semantic(), _warm_index(),
                                return_exceptions=True,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        _il_done()
                        logger.warning(
                            "[InitializationGuard] WARMUP_COMPLETE — STEADY_STATE; "
                            "heavy init pre-warmed off-loop, steady-state "
                            "starvation now authoritative",
                        )

                    self._warmup_task = _aio_il.create_task(
                        _warmup_lifecycle_boot(),
                    )
            except Exception as _ilx:  # noqa: BLE001
                logger.debug("[GovernedLoop] warmup lifecycle swallowed: %r", _ilx)

            # Slice 211 — STRATEGIC IGNITION MESH. The roadmap_orchestrator
            # (reads the operator-signed roadmap, emits goal envelopes into
            # intake) had ZERO callers in the live loop — the disconnected wire
            # found in the GOAL-001 autonomy test. Plug it into GLS as a
            # deferred daemon that drives execute_roadmap in single-poll bursts
            # on an ADAPTIVE, stress-aware cadence (recovers to base when the
            # vendor stabilizes — corrected from the cumulative formula that
            # never recovers). Every emitted goal still flows through the full
            # gated pipeline (Iron Gate + SemanticGuardian + boundary gate ->
            # APPROVAL_REQUIRED -> orange PR). Gated, fail-soft, never blocks
            # boot. Wired into GLS (the LIVE loop), NOT legacy engine.py.
            try:
                from backend.core.ouroboros.governance.roadmap_orchestrator import (  # noqa: E501
                    master_enabled as _rmo_enabled,
                    execute_roadmap as _rmo_execute,
                )
                from backend.core.ouroboros.governance.roadmap_cadence import (
                    AdaptiveRoadmapCadence as _RmCadence,
                )
                if _rmo_enabled():
                    import asyncio as _aio_rm

                    async def _roadmap_ignition_daemon() -> None:
                        cad = _RmCadence()
                        # A1-T2 — event-driven router-ready valve. Replaces the
                        # blind 20s settle (a poll-hack) with a race-free,
                        # bounded wait for the intake router's dispatch loop, so
                        # a strategic GOAL is never emitted into a void (the
                        # silent-drop race A1 exists to kill). No sleep-poll on
                        # the happy (bus-present) path. Bind the readiness API
                        # once here so the loop body is NameError-safe even if
                        # the import fails (fail-open to the legacy emit).
                        try:
                            from backend.core.ouroboros.governance.intake.unified_intake_router import (  # noqa: E501
                                await_router_ready as _a1_await_ready,
                                router_is_ready as _a1_router_is_ready,
                            )
                            from backend.core.trinity_event_bus import (
                                get_event_bus_if_exists as _a1_get_bus,
                            )
                        except Exception:  # noqa: BLE001
                            _a1_await_ready = None
                            _a1_router_is_ready = None
                            _a1_get_bus = None
                        if _a1_await_ready is not None:
                            # 600s, not 60s. This budget is not the router's
                            # own work — it is everything the router waits
                            # BEHIND during a cold boot: 24 sensors
                            # registering, the embedding service loading its
                            # ONNX weights, the Aegis daemon binding a port.
                            # Measured on the reference workstation (soak
                            # bt-2026-08-30-070418): the breaker fired at
                            # +2m12s while the wall-clock watchdog did not
                            # arm until +4m38s, and full boot ran 6-7 min. A
                            # 60s budget could therefore NEVER be met on that
                            # host, so the very first sanctioned roadmap goal
                            # aborted before the daemon emitted anything.
                            #
                            # The breaker itself is correct and stays: on a
                            # clean not-ready it raises rather than looping a
                            # goal into a void (the A1 run #15 silent-DLQ).
                            # Only its calibration was wrong — it was sized
                            # for a warm process, not a cold organism. A
                            # too-SHORT deadline on a boot path is not a
                            # safety property; it is an outage that reports
                            # itself as one.
                            try:
                                _a1_timeout = float(
                                    (os.environ.get(
                                        "JARVIS_A1_ROUTER_READY_TIMEOUT_S", ""
                                    ) or "600").strip()
                                )
                            except (TypeError, ValueError):
                                _a1_timeout = 600.0
                            # Task 2 — CIRCUIT BREAKER: capture the readiness
                            # result. Fail-OPEN only on an UNEXPECTED valve error
                            # (infra hiccup) so a transient bus glitch doesn't
                            # abort; but on a clean not-ready-within-timeout, fail
                            # LOUD (raise) instead of swallowing and looping a goal
                            # into a never-ready void (the A1 run #15 silent-DLQ).
                            try:
                                _a1_bus = _a1_get_bus() if _a1_get_bus else None
                                _a1_ready_ok = await _a1_await_ready(
                                    _a1_bus, _a1_timeout
                                )
                            except Exception as _a1_vex:  # noqa: BLE001
                                logger.debug(
                                    "[GovernedLoop] router-ready valve errored "
                                    "(fail-open to legacy emit): %r",
                                    _a1_vex,
                                )
                                _a1_ready_ok = True
                            if not _a1_ready_ok:
                                from backend.core.ouroboros.governance.intake.unified_intake_router import (  # noqa: E501
                                    RouterInitializationTimeoutError,
                                )

                                logger.critical(
                                    "[A1] roadmap daemon CIRCUIT BREAKER — intake "
                                    "router not ready within %.0fs; aborting "
                                    "roadmap ignition (fail-loud, no silent "
                                    "DLQ-loop). Streamed to GCS sidecar via "
                                    "debug.log.",
                                    _a1_timeout,
                                )
                                raise RouterInitializationTimeoutError(
                                    "intake router not ready within %.0fs"
                                    % (_a1_timeout,)
                                )
                        _a1_timeout_dlq_done = False
                        while True:
                            try:
                                # A1-T2 — never emit into a void: guard each
                                # emit on the authoritative readiness flag. If
                                # the valve timed out (router never attached),
                                # record the stall LOUD + persisted exactly once
                                # and skip the emit; retry on the next cadence
                                # tick (the router may come up late). Fail-open
                                # to the legacy emit if the probe is unavailable.
                                if _a1_router_is_ready is not None:
                                    try:
                                        _a1_ready = _a1_router_is_ready()
                                    except Exception:  # noqa: BLE001
                                        _a1_ready = True
                                else:
                                    _a1_ready = True
                                if not _a1_ready:
                                    if not _a1_timeout_dlq_done:
                                        try:
                                            from backend.core.ouroboros.governance import (  # noqa: E501
                                                intake_dlq as _a1_dlq,
                                            )
                                            _a1_dlq.append_dlq(
                                                {
                                                    "source": "roadmap_ignition",
                                                    "reason":
                                                        "router_ready_timeout",
                                                    "goal_id":
                                                        "roadmap_ignition_gate",
                                                    "note": "router not ready "
                                                            "within timeout",
                                                },
                                                reason="router_ready_timeout",
                                            )
                                        except Exception:  # noqa: BLE001
                                            pass
                                        logger.critical(
                                            "[A1] roadmap daemon: router not "
                                            "ready within timeout — emit "
                                            "skipped, marker DLQ'd (retrying "
                                            "on next tick)"
                                        )
                                        _a1_timeout_dlq_done = True
                                else:
                                    _router = getattr(
                                        self, "_intake_router", None
                                    )
                                    _rep = await _rmo_execute(
                                        router=_router,
                                        max_iterations_override=1,
                                    )
                                    try:
                                        from backend.core.ouroboros.governance.progress_ledger import (  # noqa: E501
                                            ledger_enabled as _pl_on,
                                            update_progress as _pl_update,
                                        )
                                        if _pl_on() and _rep is not None:
                                            _pl_update(
                                                completed=[],
                                                next_targets=[(
                                                    "GOAL-001",
                                                    "roadmap orchestrator "
                                                    "polling (strategic "
                                                    "ignition mesh live)",
                                                )],
                                            )
                                    except Exception:  # noqa: BLE001
                                        pass
                            except _aio_rm.CancelledError:
                                break
                            except Exception as _re:  # noqa: BLE001
                                logger.debug(
                                    "[GovernedLoop] roadmap poll swallowed: %r",
                                    _re,
                                )
                            try:
                                await _aio_rm.sleep(cad.next_interval_s())
                            except _aio_rm.CancelledError:
                                break

                    self._roadmap_task = _aio_rm.create_task(
                        _roadmap_ignition_daemon(),
                    )
                    logger.warning(
                        "[GovernedLoop] STRATEGIC IGNITION MESH live — roadmap "
                        "orchestrator wired into the control plane (adaptive "
                        "stress-aware cadence); organism now feeds on "
                        "operator-signed goals.",
                    )
            except Exception as _rmx:  # noqa: BLE001
                logger.debug("[GovernedLoop] roadmap mesh wiring swallowed: %r", _rmx)

        except Exception as exc:
            self._state = ServiceState.FAILED
            self._failure_reason = str(exc)
            logger.error(
                "[GovernedLoop] Start failed: %s", exc, exc_info=True
            )
            await self._teardown_partial()
            raise

    async def stop(self) -> None:
        """Graceful shutdown. Drains in-flight ops, cancels probes."""
        if self._state is ServiceState.INACTIVE:
            return

        self._state = ServiceState.STOPPING

        # Sovereign Epistemic Context Matrix LR2: reconcile this session's memory
        # quarantine vs live disk on teardown; refresh oracle for revalidated
        # nodes. Fail-soft — never blocks shutdown.
        try:
            from pathlib import Path as _Path
            from backend.core.ouroboros.governance.epistemic_quarantine import QuarantineLedger
            # derive repo root the same way the loop does elsewhere:
            _cfg = getattr(self, "_config", None)
            _root = str(getattr(_cfg, "project_root", "") or "") or "."
            # LR2 reader/writer agreement: resolve the session id the SAME way
            # the 6c WRITER (orchestrator._resolve_session_id) does -- via the
            # canonical get_active_session_id() -- with a pid-<getpid()> fallback.
            # The previous _session_dir-derived id was always "" on the service
            # (self._session_dir is only ever set on the harness), so the
            # ``if _sid:`` guard never fired and reconcile never ran. Worse, the
            # writer keys by get_active_session_id() while this read keyed by
            # _session_dir.name -- a latent mismatch. Now they agree.
            _sid = ""
            try:
                from backend.core.ouroboros.governance.strategic_direction import get_active_session_id
                _sid = str(get_active_session_id() or "")
            except Exception:  # noqa: BLE001
                _sid = ""
            if not _sid:
                import os as _os
                _sid = f"pid-{_os.getpid()}"
            if _sid:
                _led = QuarantineLedger(
                    str(_Path(_root) / ".jarvis" / "epistemic_quarantine.jsonl"),
                    _sid,
                )
                _rec = _led.reconcile(root=_root)
                _oracle = getattr(self, "_oracle", None)
                if _rec.get("revalidated") and _oracle is not None:
                    _upd = getattr(_oracle, "incremental_update", None)
                    if _upd is not None:
                        await _upd()
        except Exception:  # noqa: BLE001 — never block shutdown
            pass

        # GracefulTeardownMatrix (2026-06-20) — cancel the DW discovery/heavy-probe
        # background loops FIRST. Left pending they leak ("Task was destroyed but
        # it is pending!") and wedge teardown for minutes on their aiohttp
        # connector-timeout awaits (the live-soak post-summary 5-min hang). Bounded
        # + fail-soft; never blocks shutdown.
        try:
            from backend.core.ouroboros.governance.dw_discovery_runner import (
                shutdown_background_loops as _dw_shutdown_loops,
            )
            await _dw_shutdown_loops(timeout_s=5.0)
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass
        # Close the DW provider's aiohttp session so its connector doesn't linger
        # waiting on a TCP timeout during teardown (reuses force_session_reset).
        try:
            _dw_ref = getattr(self, "_doubleword_ref", None)
            if _dw_ref is not None and hasattr(_dw_ref, "force_session_reset"):
                await asyncio.wait_for(_dw_ref.force_session_reset(), timeout=5.0)
        except Exception:  # noqa: BLE001
            pass

        # Slice 101 — publish session_end onto the cognitive bus before the
        # drain so observational subscribers can react. Fire-and-forget,
        # NEVER raises, inert unless JARVIS_COGNITIVE_BUS_ENABLED.
        try:
            from backend.core.ouroboros.governance.cognitive_bus import (
                LIFECYCLE_SESSION_END,
                publish_lifecycle_event as _cb_session_end,
            )
            _cb_session_end(LIFECYCLE_SESSION_END, {"service": "GLS"})
        except Exception:  # noqa: BLE001
            pass

        # Slice 101 Phase 4 — Autonomous Graduation Engine session-end pass.
        # Runs SYNCHRONOUSLY here (NOT via the fire-and-forget bus) so it
        # completes before the drain/shutdown. evaluate + execute are sync;
        # execute records AUTO_FLIP overrides to the durable
        # graduation_override_ledger (applied at next boot) and emits SAFETY-
        # tier APPROVAL advisories — it never mutates os.environ live and
        # never auto-flips a SAFETY flag. Master
        # JARVIS_AUTONOMOUS_GRADUATION_ENGINE_ENABLED §33.1 default-FALSE.
        # NEVER blocks shutdown.
        try:
            from backend.core.ouroboros.governance.autonomous_graduation_engine import (  # noqa: E501
                autonomous_graduation_engine_enabled,
                evaluate_graduations,
                execute_graduations,
            )
            if autonomous_graduation_engine_enabled():
                _grad_report = evaluate_graduations()
                _grad_exec = execute_graduations(_grad_report)
                if _grad_exec.recorded_overrides or _grad_exec.advisories_emitted:
                    logger.info(
                        "[GovernedLoop] Autonomous graduation (session-end): "
                        "auto_flipped=%d advisories=%d",
                        len(_grad_exec.recorded_overrides),
                        len(_grad_exec.advisories_emitted),
                    )
        except Exception:  # noqa: BLE001 — never block shutdown
            pass

        # P2 Slice 3 — stop the Convergence Reaper first so its
        # background task doesn't race the drain. Master-gated
        # NEVER-raise; idempotent on an already-stopped reaper.
        try:
            from backend.core.ouroboros.governance.convergence_reaper import (  # noqa: E501
                safe_stop_default_reaper,
            )
            await safe_stop_default_reaper()
        except Exception as _cr_exc:  # noqa: BLE001
            logger.debug(
                "[GovernedLoop] Convergence Reaper stop "
                "swallowed: %r", _cr_exc,
            )

        # Cancel health probe loop
        if self._health_probe_task and not self._health_probe_task.done():
            self._health_probe_task.cancel()
            try:
                await self._health_probe_task
            except asyncio.CancelledError:
                pass

        # C2 -- cancel the TransportCircuitBreaker probe daemon if running.
        _tcb_task = getattr(self, "_transport_breaker_probe_task", None)
        if _tcb_task is not None and not _tcb_task.done():
            _tcb_task.cancel()
            try:
                await _tcb_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] transport_breaker_probe_task cancel swallowed",
                    exc_info=True,
                )

        # Omni-Soak #3 — cancel the failover tick loop cleanly (no orphan
        # task). Fail-soft; never blocks shutdown.
        await self._stop_failover_loop()

        # Built-but-no-caller fix — cancel the Meta-Goal aggregator drain loop
        # cleanly (no orphan task). Fail-soft; never blocks shutdown.
        await self._stop_meta_goal_drain_loop()

        # YM-T10 SEAM 1 — cancel the operator-presence watcher daemon
        # alongside the other background tasks. Fail-soft.
        _op_pres_task = getattr(self, "_operator_presence_task", None)
        if _op_pres_task is not None and not _op_pres_task.done():
            _op_pres_task.cancel()
            try:
                await _op_pres_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — never block shutdown
                logger.debug(
                    "[GovernedLoop] operator-presence task cancel swallowed",
                    exc_info=True,
                )

        # Slice 5 Arc A — stop PostureObserver cleanly (no orphan tasks)
        observer = getattr(self, "_posture_observer", None)
        if observer is not None:
            try:
                await observer.stop()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] PostureObserver stop failed (non-fatal)",
                    exc_info=True,
                )

        # Tier 0.5 batch 1 — stop the 3 dormant-now-live Slice 5b observers
        # (InvariantDrift / Coherence / CIGW). Mirrors the boot wire-up;
        # fail-open per observer.
        await self._stop_governance_observers()

        # Stop L3 scheduler before background loops so no unit outlives GLS
        if self._subagent_scheduler is not None:
            await self._subagent_scheduler.stop()

        # Cancel curriculum and reactor event background tasks.
        # _batch_reconcile_task (Slice 18) rides this list: an in-flight boot
        # reconcile makes network calls (GET /batches, POST /cancel on remote
        # jobs) and must not outlive the service — an interrupted sweep is
        # safe because deferred claims stay OPEN and the next boot retries.
        for task_attr in ("_curriculum_task", "_reactor_event_task", "_oracle_indexer_task",
                         "_feedback_loop_task", "_command_consumer_task",
                         "_batch_reconcile_task"):
            task: Optional[asyncio.Task] = getattr(self, task_attr, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Hard-cancel in-flight pool ops. Task #95: budget-exhaust shutdown
        # must not let background workers keep issuing Claude/DW calls that
        # bill *after* CostTracker.budget_event fires. The previous
        # ``asyncio.sleep(0)`` was a no-op drain — workers survived, calls
        # completed, spend overshot the --cost-cap. Manifesto §6 hard stop.
        _bg_pool = getattr(self, "_bg_pool", None)
        if _bg_pool is not None:
            try:
                active = _bg_pool.list_active()
            except Exception:
                active = []
            if active:
                logger.info(
                    "[GovernedLoop] Cancelling %d in-flight pool ops via _bg_pool.stop()",
                    len(active),
                )
            try:
                await _bg_pool.stop()
            except Exception as exc:
                logger.warning("[GovernedLoop] _bg_pool.stop() error: %s", exc)

        if self._active_ops:
            logger.info(
                "[GovernedLoop] %d FSM contexts remained at stop",
                len(self._active_ops),
            )

        # Stop EventChannelServer (DW 3-tier webhook receiver)
        _evt_ch = getattr(self, "_event_channel", None)
        if _evt_ch is not None:
            try:
                await _evt_ch.stop()
            except Exception:
                pass

        # J-Prime local tier (Phase 3) — release the injected LocalPrimeClient's
        # pooled aiohttp session so shutdown leaves zero hanging FDs. Best-effort;
        # inert unless a local client was injected (flag ON + no GCP client).
        _lid = getattr(self, "_local_inference_director", None)
        if _lid is not None:
            try:
                await _lid.stop()
            except Exception:
                logger.debug(
                    "[GovernedLoop] local inference director stop failed",
                    exc_info=True,
                )

        # Phase 3.4: hand the local daemon back to the host (flush weights; stop the
        # daemon only if the governor started it). Gated + ownership-safe + fail-soft.
        _gov = getattr(self, "_local_daemon_governor", None)
        if _gov is not None:
            try:
                await _gov.stop_if_idle()
            except Exception:
                logger.debug(
                    "[GovernedLoop] local daemon governor stop failed",
                    exc_info=True,
                )

        # Detach from stack. `/why` released LAST, after the drain above:
        # while ops are still finishing they genuinely are in flight, and
        # unbinding early would make `/why` claim disk-only about work that
        # is still running.
        self._cancel_audit_watchdogs()
        self._release_why_live_source()
        self._detach_from_stack()
        self._state = ServiceState.INACTIVE
        logger.info("[GovernedLoop] Stopped")

    # ------------------------------------------------------------------
    # YM-T10 SEAM 1 — Sovereign Daemon Injection Protocol (Layer 2)
    # ------------------------------------------------------------------

    async def _start_operator_yield_layer(self) -> None:
        """Spawn the operator-presence watcher daemon + attach the yield bridge.

        Layer-2 production activation (YM-T10 SEAM 1). Two independent,
        fully fail-soft steps:

          1. Spawn ``OperatorPresenceWatcher().run(bus=None)`` as a background
             daemon task (mirrors the oracle/health-probe ``create_task``
             pattern). A strong ref is stored on ``self`` so the task is not
             GC'd; it is cancelled in ``stop()``.
          2. ``await operator_yield_bridge.attach(bus=None, pool=self._bg_pool)``
             once during boot, after the pool exists.

        Both ``run()`` and ``attach()`` are already no-op when
        ``JARVIS_OPERATOR_YIELD_ENABLED`` is off, and they self-resolve the
        TrinityEventBus singleton when ``bus=None`` — so this method is
        byte-identical to pre-YM-T10 when the flag is off.

        HARD REQUIREMENT: nothing here may prevent or crash boot. Every step
        is wrapped so a failure logs + degrades; the main loop runs regardless.
        """
        pool = getattr(self, "_bg_pool", None)

        # Step 1 — spawn the watcher daemon (fail-soft).
        try:
            from backend.core.ouroboros.governance.operator_presence import (
                OperatorPresenceWatcher,
            )

            watcher = OperatorPresenceWatcher()
            self._operator_presence_task = asyncio.create_task(
                watcher.run(bus=None),
                name="operator_presence_watcher",
            )
            logger.info(
                "[GovernedLoop] Operator-presence watcher daemon spawned "
                "(YM-T10 SEAM 1; no-op when JARVIS_OPERATOR_YIELD_ENABLED off)",
            )
        except Exception:  # noqa: BLE001 — watcher must NEVER crash boot
            logger.warning(
                "[GovernedLoop] Operator-presence watcher spawn failed "
                "(non-fatal; loop continues)",
                exc_info=True,
            )
            self._operator_presence_task = None

        # Step 2 — attach the yield bridge (fail-soft, independent of step 1).
        try:
            from backend.core.ouroboros.governance import (
                operator_yield_bridge as _oyb,
            )

            await _oyb.attach(bus=None, pool=pool)
            logger.info(
                "[GovernedLoop] Operator-yield bridge attach() invoked "
                "(YM-T10 SEAM 1; no-op when yield off)",
            )
        except Exception:  # noqa: BLE001 — attach must NEVER crash boot
            logger.warning(
                "[GovernedLoop] Operator-yield bridge attach failed "
                "(non-fatal; loop continues)",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Tier 0.5 batch 1 — governance observer boot/shutdown helpers
    # ------------------------------------------------------------------

    async def _start_governance_observers(self) -> None:
        """Boot the dormant Slice 5b observers (Tier 0.5 batch 1).

        Wires three substrates that shipped graduated default-True but
        had zero production callers per the codebase audit:

          * ``InvariantDriftObserver`` — Move 4 Slice 5b deferred
          * ``CoherenceObserver`` — Priority #1 Slice 5b deferred
          * ``CIGWObserver`` (gradient watcher) — Priority #5 Slice 5b
            deferred

        Each is master-flag-gated by its own substrate's ``_enabled``
        accessor (no hardcoded defaults — every flag is operator-
        controllable). Each is wrapped in its own try/except so a
        single observer's failure does NOT prevent the others from
        booting; the loop continues regardless. References are stored
        on ``self._<observer>`` for orderly shutdown.
        """
        # FiringTelemetry helper — best-effort instrumentation. The
        # substrate's incr() is total (NEVER raises) so the helper
        # is just a convenience indirection so tests can mock it.
        # Importing inside the helper keeps the whole observer-boot
        # path import-error-resilient (instrumentation degrades to
        # a no-op if the module is unavailable, observers still
        # start normally).
        def _incr_observer_boot(name: str) -> None:
            try:
                from backend.core.ouroboros.governance.firing_telemetry import (  # noqa: E501
                    incr_fire_counter,
                )
                incr_fire_counter(f"observer.{name}.booted")
            except Exception:  # noqa: BLE001
                pass

        # InvariantDriftObserver (Move 4 Slice 5b)
        self._invariant_drift_observer = None
        try:
            from backend.core.ouroboros.governance.invariant_drift_auditor import (  # noqa: E501
                invariant_drift_auditor_enabled,
            )
            from backend.core.ouroboros.governance.invariant_drift_observer import (  # noqa: E501
                get_default_observer as get_drift_observer,
                observer_enabled as drift_observer_enabled,
            )
            if (
                invariant_drift_auditor_enabled()
                and drift_observer_enabled()
            ):
                self._invariant_drift_observer = get_drift_observer()
                self._invariant_drift_observer.start()
                _incr_observer_boot("invariant_drift")
                logger.info(
                    "[GovernedLoop] InvariantDriftObserver started "
                    "(Tier 0.5 batch 1)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] InvariantDriftObserver startup "
                "failed (non-fatal)",
                exc_info=True,
            )
            self._invariant_drift_observer = None

        # CoherenceObserver (Priority #1 Slice 5b)
        self._coherence_observer = None
        try:
            from backend.core.ouroboros.governance.verification.coherence_auditor import (  # noqa: E501
                coherence_auditor_enabled,
            )
            from backend.core.ouroboros.governance.verification.coherence_observer import (  # noqa: E501
                get_default_observer as get_coherence_observer,
                observer_enabled as coherence_observer_enabled,
            )
            if (
                coherence_auditor_enabled()
                and coherence_observer_enabled()
            ):
                self._coherence_observer = get_coherence_observer()
                self._coherence_observer.start()
                _incr_observer_boot("coherence")
                logger.info(
                    "[GovernedLoop] CoherenceObserver started "
                    "(Tier 0.5 batch 1)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] CoherenceObserver startup failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._coherence_observer = None

        # CIGWObserver / GradientObserver (Priority #5 Slice 5b)
        # Async start. No module-level singleton — construct
        # explicitly. The default-arg constructor reads all knobs
        # from env at every interval poll (no hardcoding).
        self._cigw_observer = None
        try:
            from backend.core.ouroboros.governance.verification.gradient_watcher import (  # noqa: E501
                cigw_enabled,
            )
            from backend.core.ouroboros.governance.verification.gradient_observer import (  # noqa: E501
                CIGWObserver,
                cigw_observer_enabled,
            )
            if cigw_enabled() and cigw_observer_enabled():
                self._cigw_observer = CIGWObserver()
                await self._cigw_observer.start()
                _incr_observer_boot("cigw")
                logger.info(
                    "[GovernedLoop] CIGWObserver started "
                    "(Tier 0.5 batch 1)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] CIGWObserver startup failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._cigw_observer = None

        # ----- Tier 0.5 batch 2 -----
        # Three more observers from the audit's dead-code list.
        # Same lazy-import + master/sub-flag-gated + per-observer
        # fail-open discipline as batch 1.

        # SBTObserver (Priority #4 Slice 5b — Speculative Branch Tree)
        # Async start. Explicit constructor; reads knobs from env.
        self._sbt_observer = None
        try:
            from backend.core.ouroboros.governance.verification.speculative_branch import (  # noqa: E501
                sbt_enabled,
            )
            from backend.core.ouroboros.governance.verification.speculative_branch_observer import (  # noqa: E501
                SBTObserver,
                sbt_observer_enabled,
            )
            if sbt_enabled() and sbt_observer_enabled():
                self._sbt_observer = SBTObserver()
                await self._sbt_observer.start()
                _incr_observer_boot("sbt")
                logger.info(
                    "[GovernedLoop] SBTObserver started "
                    "(Tier 0.5 batch 2)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] SBTObserver startup failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._sbt_observer = None

        # ReplayObserver (Priority #3 Slice 5b — Counterfactual Replay)
        # Async start. Explicit constructor.
        self._replay_observer = None
        try:
            from backend.core.ouroboros.governance.verification.counterfactual_replay import (  # noqa: E501
                counterfactual_replay_enabled,
            )
            from backend.core.ouroboros.governance.verification.counterfactual_replay_observer import (  # noqa: E501
                ReplayObserver,
                replay_observer_enabled,
            )
            if (
                counterfactual_replay_enabled()
                and replay_observer_enabled()
            ):
                self._replay_observer = ReplayObserver()
                await self._replay_observer.start()
                _incr_observer_boot("replay")
                logger.info(
                    "[GovernedLoop] ReplayObserver started "
                    "(Tier 0.5 batch 2)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] ReplayObserver startup failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._replay_observer = None

        # ClosureLoopObserver (Q4 P#2 Slice 4 follow-up).
        # Async start. Singleton via get_default_observer. Single
        # master flag — no separate observer sub-flag in this module.
        self._closure_loop_observer = None
        try:
            from backend.core.ouroboros.governance.verification.closure_loop_orchestrator import (  # noqa: E501
                closure_loop_orchestrator_enabled,
            )
            from backend.core.ouroboros.governance.verification.closure_loop_observer import (  # noqa: E501
                get_default_observer as get_closure_observer,
            )
            if closure_loop_orchestrator_enabled():
                self._closure_loop_observer = get_closure_observer()
                await self._closure_loop_observer.start()
                _incr_observer_boot("closure_loop")
                logger.info(
                    "[GovernedLoop] ClosureLoopObserver started "
                    "(Tier 0.5 batch 2)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] ClosureLoopObserver startup failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._closure_loop_observer = None

        # TrajectoryAuditorObserver (un-stranding 2026-05-04 —
        # PRD §24.10.2 + §1 long-horizon semantic stability gap).
        # Boot snapshot + 6h periodic tick (env-tunable). Pure
        # stdlib codebase walk; SSE published only on warning /
        # critical drift verdicts. Master flag default-true
        # post-graduation.
        self._trajectory_auditor_observer = None
        try:
            from backend.core.ouroboros.governance.observability.trajectory_auditor_observer import (  # noqa: E501
                get_default_trajectory_observer,
                trajectory_observer_enabled,
            )
            if trajectory_observer_enabled():
                self._trajectory_auditor_observer = (
                    get_default_trajectory_observer()
                )
                await self._trajectory_auditor_observer.start()
                _incr_observer_boot("trajectory_auditor")
                logger.info(
                    "[GovernedLoop] TrajectoryAuditorObserver "
                    "started (un-stranding)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] TrajectoryAuditorObserver startup "
                "failed (non-fatal)",
                exc_info=True,
            )
            self._trajectory_auditor_observer = None

        # Slice 101 — Cognitive Integration Bus subscriber registration.
        # Registers the cognitive subscribers (belief-revision learning loop,
        # ...) onto the production TrinityEventBus so they react to FSM
        # lifecycle events published from _record_ledger. Inert (returns [])
        # unless JARVIS_COGNITIVE_BUS_ENABLED (§33.1 default-FALSE). Each
        # subscriber self-gates on its own substrate master. NEVER fatal.
        self._cognitive_bus_subscription_ids = []
        try:
            from backend.core.ouroboros.governance.cognitive_bus import (
                register_cognitive_subscribers,
            )
            from backend.core.ouroboros.governance.cognitive_subscribers import (
                build_default_subscribers,
            )
            self._cognitive_bus_subscription_ids = (
                await register_cognitive_subscribers(build_default_subscribers())
            )
            # Slice 109 — God-Tier Observability Matrix. Bind the SSE
            # Why-Snapshot publisher + Karen's voice narrator onto the SAME
            # cognitive bus. Both self-gate (SSE on JARVIS_IDE_STREAM_ENABLED,
            # voice on JARVIS_KAREN_VOICE_ENABLED + mute state); neither holds
            # authority. Inert when the cognitive bus is off. NEVER fatal.
            try:
                from backend.core.ouroboros.governance.cognitive_observability import (
                    register_observability,
                )
                obs_ids = await register_observability()
                if obs_ids:
                    self._cognitive_bus_subscription_ids = list(
                        self._cognitive_bus_subscription_ids
                    ) + list(obs_ids)
                    logger.info(
                        "[GovernedLoop] Observability Matrix: %d subscriber(s) "
                        "bound to cognitive bus (Slice 109)",
                        len(obs_ids),
                    )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] observability subscriber registration "
                    "swallowed (non-fatal)",
                    exc_info=True,
                )
            # Slice 110 — Native Command Center. Bind the bus→WebSocket bridge so
            # cognitive lifecycle frames fan out to the React command center
            # (when this process also serves the FastAPI app). Inert when the
            # gateway is disabled or no WS clients are connected. NEVER fatal.
            try:
                from backend.api.observability_gateway import (
                    register_gateway_bridge,
                )
                gw_ids = await register_gateway_bridge()
                if gw_ids:
                    self._cognitive_bus_subscription_ids = list(
                        self._cognitive_bus_subscription_ids
                    ) + list(gw_ids)
                    logger.info(
                        "[GovernedLoop] Command-Center gateway: %d bridge "
                        "subscriber(s) bound to cognitive bus (Slice 110)",
                        len(gw_ids),
                    )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] gateway bridge registration swallowed "
                    "(non-fatal)",
                    exc_info=True,
                )
            if self._cognitive_bus_subscription_ids:
                _incr_observer_boot("cognitive_bus")
                logger.info(
                    "[GovernedLoop] Cognitive Integration Bus: %d subscriber(s) "
                    "registered (Slice 101)",
                    len(self._cognitive_bus_subscription_ids),
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] Cognitive bus subscriber registration failed "
                "(non-fatal)",
                exc_info=True,
            )
            self._cognitive_bus_subscription_ids = []

        # ── Wire #2 — StrategyOutcomeLogger on the production TrinityEventBus ──
        # Flushes big-file Agentic-Swarm extraction-strategy outcomes to SQLite
        # on every op.terminal.* (off-loop, via run_in_executor). Only attached
        # when swarm routing is enabled (no pending strategies exist otherwise,
        # so it would be a no-op subscriber + a stray DB file on every boot).
        # Fault-Tolerant Shield: a locked/unavailable SQLite DB logs an error
        # telemetry event and boot continues UNINTERRUPTED — telemetry failure
        # must never poison the orchestration boot.
        self._strategy_outcome_logger = None
        _swarm_on = os.environ.get(
            "JARVIS_SWARM_ROUTING_ENABLED", "false",
        ).strip().lower() in ("1", "true", "yes", "on")
        if _swarm_on:
            try:
                from backend.core.trinity_event_bus import get_event_bus_if_exists
                from backend.core.ouroboros.governance.chunked_generation_bridge import (
                    StrategyOutcomeLogger,
                )
                _sbus = get_event_bus_if_exists()
                if _sbus is not None:
                    import sqlite3 as _sqlite3
                    from pathlib import Path as _Path
                    _db_dir = _Path(".jarvis")
                    _db_dir.mkdir(parents=True, exist_ok=True)
                    _conn = _sqlite3.connect(
                        str(_db_dir / "chunk_strategy.db"), check_same_thread=False,
                    )
                    _sol = StrategyOutcomeLogger(_conn)
                    await _sol.attach_to_bus(_sbus)
                    self._strategy_outcome_logger = _sol
                    _incr_observer_boot("strategy_outcome_logger")
                    logger.info(
                        "[GovernedLoop] Wire #2: StrategyOutcomeLogger attached "
                        "to TrinityEventBus (swarm reinforcement → SQLite)",
                    )
            except Exception:  # noqa: BLE001 — telemetry attach NEVER fatal to boot
                logger.warning(
                    "[GovernedLoop] Wire #2: StrategyOutcomeLogger attach failed "
                    "(non-fatal, boot continues)",
                    exc_info=True,
                )
                self._strategy_outcome_logger = None

        # Slice 101 Phase 6 — Sleep Consolidation Daemon. Background async task
        # that runs the cross-session memory cascade (belief + postmortem-fusion
        # → consolidation → meta-prior calibration) OFF the hot path on an idle-
        # gated cadence. Inert (no task spawned) unless JARVIS_SLEEP_DAEMON_
        # ENABLED (§33.1 default-FALSE). Stored on self for graceful cancel.
        self._sleep_daemon_task = None
        try:
            from backend.core.ouroboros.governance.sleep_daemon import (
                run_sleep_daemon_loop,
                sleep_daemon_enabled,
            )
            if sleep_daemon_enabled():
                self._sleep_daemon_task = asyncio.create_task(
                    run_sleep_daemon_loop()
                )
                _incr_observer_boot("sleep_daemon")
                logger.info(
                    "[GovernedLoop] Sleep Consolidation Daemon started "
                    "(Slice 101 Phase 6)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] Sleep daemon startup failed (non-fatal)",
                exc_info=True,
            )
            self._sleep_daemon_task = None

        # Task #2 — boot-time spec-drift one-shot. The post-commit scheduler
        # (auto_committer) catches drift a FUTURE commit introduces; this catches
        # drift that ALREADY exists at boot (e.g. a docstring/CLAUDE.md default
        # that silently diverged from the registry) so the organism surfaces it
        # immediately, not only after its next commit happens to run. Runs in a
        # thread executor (reads CLAUDE.md + registry = file I/O; keep off the
        # loop). Master-gated + fail-soft; a drift audit must never fail boot.
        try:
            from backend.core.ouroboros.governance.mirror_self_spec_drift import (  # noqa: E501
                master_enabled as _spec_drift_enabled,
                run_spec_drift_audit,
            )
            if _spec_drift_enabled():
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, run_spec_drift_audit)
                _incr_observer_boot("spec_drift")
                logger.info(
                    "[GovernedLoop] spec-drift boot audit scheduled (Task #2)",
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GovernedLoop] spec-drift boot audit failed (non-fatal)",
                exc_info=True,
            )

    async def _stop_governance_observers(self) -> None:
        """Stop the Tier 0.5 (batches 1+2) observers gracefully.

        Mirrors the boot helper. Each observer's stop is awaited
        independently — one observer's stop failure does NOT prevent
        the others from stopping. NEVER raises.
        """
        # Slice 101 Phase 6 — cancel the Sleep Consolidation Daemon first so its
        # background cycle doesn't race the drain. Idempotent; NEVER raises.
        _sleep_task = getattr(self, "_sleep_daemon_task", None)
        if _sleep_task is not None and not _sleep_task.done():
            _sleep_task.cancel()
            try:
                await _sleep_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] sleep daemon stop failed (non-fatal)",
                    exc_info=True,
                )

        for attr_name in (
            "_invariant_drift_observer",
            "_coherence_observer",
            "_cigw_observer",
            "_sbt_observer",
            "_replay_observer",
            "_closure_loop_observer",
            "_trajectory_auditor_observer",
        ):
            observer = getattr(self, attr_name, None)
            if observer is None:
                continue
            try:
                await observer.stop()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] %s.stop failed (non-fatal)",
                    attr_name,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit_background(
        self,
        ctx: OperationContext,
        trigger_source: str = "unknown",
    ) -> str:
        """Submit an operation for non-blocking background execution.

        Returns the op_id immediately. Use get_background_result(op_id)
        to poll for completion.

        Falls back to synchronous submit() if BackgroundAgentPool is not available.
        """
        # Built-but-no-caller fix — route clean single-file ops through the
        # Meta-Goal aggregator's intake (when JARVIS_META_GOAL_AGGREGATOR_
        # ENABLED). The drain loop owns their dispatch: it bundles disjoint
        # siblings into ONE fan-outable Meta-Goal (swarm fan-out) or flushes
        # an aged-out single to _bg_pool.submit. OFF / non-single-file / any
        # error -> offer_ctx returns False and the legacy submit below runs,
        # byte-identical. The op is never lost.
        try:
            from backend.core.ouroboros.governance.meta_goal_wiring import (
                offer_ctx as _mg_offer_ctx,
            )
            if _mg_offer_ctx(self, ctx):
                _pooled_id = str(getattr(ctx, "op_id", "") or "")
                logger.info(
                    "[GovernedLoop] op %s pooled for Meta-Goal aggregation "
                    "(trigger=%s)", _pooled_id, trigger_source,
                )
                return _pooled_id
        except Exception:  # noqa: BLE001 -- never block the legacy dispatch
            logger.debug(
                "[GovernedLoop] meta-goal intake skipped (non-fatal)",
                exc_info=True,
            )

        if self._bg_pool is not None:
            # Import the capacity-rejection class locally to keep the module
            # import graph flat (background_agent_pool imports op_context; a
            # top-level import here risks a cycle).
            from backend.core.ouroboros.governance.background_agent_pool import (
                QueueFullError,
            )
            try:
                op_id = await self._bg_pool.submit(ctx)
                logger.info(
                    "[GovernedLoop] Background op submitted: %s (trigger=%s)",
                    op_id, trigger_source,
                )
                return op_id
            except QueueFullError:
                # NO-LOSS SUBMIT BOUNDARY. A capacity rejection is NOT a broken
                # pool — it is transient backpressure. The legacy behavior ran
                # ``await self.submit(ctx)`` here: a ~2-minute op executed
                # SYNCHRONOUSLY INLINE on the dispatch path, serializing the
                # UnifiedIntakeRouter's dequeue loop so low-priority write-intent
                # signals never drained for 60+ min (live bt-iso-1783144982, 45x).
                # Instead we RE-RAISE: the router's intake WAL already holds this
                # envelope as a ``status="pending"`` row, so declining to ack it
                # parks it durably. The router catches QueueFullError, keeps the
                # WAL row pending, and re-drains it via _replay_wal once the pool
                # reports free capacity. The dispatcher stays fully async.
                logger.info(
                    "[GovernedLoop] Background submit deferred (capacity) — "
                    "parking in intake WAL: op=%s (trigger=%s)",
                    getattr(ctx, "op_id", "") or "?", trigger_source,
                )
                raise
            except Exception as exc:
                # Genuinely broken pool (not-started, worker crash, etc.) — the
                # legacy sync fallback still applies. This is a real failure, not
                # backpressure, so serializing one op inline is acceptable.
                logger.warning(
                    "[GovernedLoop] Background submit failed (non-capacity), "
                    "falling back to sync: %s", exc
                )
        # Fallback: run synchronously (no pool, or a non-capacity pool fault).
        result = await self.submit(ctx, trigger_source)
        return result.op_id

    def has_background_capacity(self) -> bool:
        """True when the background pool can accept a submit without raising
        ``QueueFullError`` (or when there is no pool — nothing to gate on).

        Consumed by the intake router's capacity-deferred WAL drain: parked
        envelopes are replayed only when a slot is actually free. NEVER raises
        (fail-open to ``True``)."""
        pool = self._bg_pool
        if pool is None:
            return True
        try:
            probe = getattr(pool, "has_capacity", None)
            if callable(probe):
                return bool(probe())
        except Exception:  # noqa: BLE001
            pass
        return True

    def get_background_result(self, op_id: str) -> Any:
        """Poll for a background operation result. Returns None if not ready."""
        if self._bg_pool is not None:
            return self._bg_pool.get_result(op_id)
        return None

    def note_operator_op(self, op_id: str) -> None:
        """Remember an op the OPERATOR asked for. NEVER raises.

        Esc means "stop what I asked for", not "stop the organism". The
        distinction is load-bearing: a bare cancel that reached autonomous
        work would let one keystroke kill a soak, and an operator who learns
        that stops trusting the key.

        Bounded: this is a recency hint for interrupt targeting, not a
        ledger. The authoritative record of what ran lives in the op ledger.
        """
        try:
            if not op_id:
                return
            ring = getattr(self, "_operator_ops", None)
            if ring is None:
                from collections import deque
                ring = deque(maxlen=16)
                self._operator_ops = ring
            ring.append(str(op_id))
        except Exception:  # noqa: BLE001
            pass

    def operator_ops_active(self) -> list:
        """Operator-initiated ops still running, most recent FIRST.

        Intersected with `_active_ops` so a finished op is never offered as an
        interrupt target — cancelling something already done would report
        success and change nothing, which is worse than reporting nothing to
        cancel.
        """
        try:
            ring = getattr(self, "_operator_ops", None) or ()
            active = self._active_ops
            seen, out = set(), []
            for op_id in reversed(list(ring)):
                if op_id in active and op_id not in seen:
                    seen.add(op_id)
                    out.append(op_id)
            return out
        except Exception:  # noqa: BLE001
            return []

    def request_cancel(self, op_id: str) -> bool:
        """Request cooperative cancellation of an in-flight operation.

        The orchestrator checks ``is_cancel_requested()`` at phase transitions.
        Returns True if the op was found active and the cancel was registered.
        """
        # Match by prefix — REPL users may provide abbreviated op_ids
        matched = [k for k in self._active_ops if k.startswith(op_id)]
        if not matched:
            return False
        for m in matched:
            self._cancel_requested.add(m)
        logger.info("[GovernedLoop] Cancel requested for op(s): %s", matched)
        return True

    def request_cancel_all(self) -> list:
        """Request cooperative cancellation of EVERY in-flight op.

        Returns the op ids newly marked, sorted — the caller reports what it
        actually stopped rather than a count it assumed, and an empty list is
        the honest answer to "stop everything" when nothing was running.

        Deliberately broader than :meth:`operator_ops_active`, which backs the
        bare `Esc` / `/cancel` path and is narrow on purpose ("stop what I
        asked for", never "stop the organism" — one key that reached
        autonomous work could kill a soak). This is the other half of that
        pair, and the friction lives in the KEYSTROKE: Claude Code binds the
        equivalent to `Ctrl+X Ctrl+K` pressed twice within three seconds, so
        reaching it takes four deliberate presses and cannot be a mis-hit.

        Cooperative, like every other cancel here: the orchestrator observes
        `is_cancel_requested` at phase transitions, so this asks the ops to
        stop at their next boundary rather than severing them mid-APPLY. A
        hard kill would leave exactly the half-written trees the phase model
        exists to prevent.
        """
        try:
            already = self._cancel_requested
            newly = sorted(op for op in self._active_ops if op not in already)
            for op_id in newly:
                self._cancel_requested.add(op_id)
            if newly:
                logger.info(
                    "[GovernedLoop] Cancel-all requested for %d op(s): %s",
                    len(newly), newly,
                )
            return newly
        except Exception:  # noqa: BLE001
            logger.debug("[GovernedLoop] cancel-all degraded", exc_info=True)
            return []

    def is_cancel_requested(self, op_id: str) -> bool:
        """Check if cancellation was requested for this operation."""
        return op_id in self._cancel_requested

    def _try_claim_file_ops(self, target_files) -> "tuple[Optional[list], Optional[str]]":
        """Slice 1 TOCTOU remediation (2026-07-18): the file-scope in-flight
        check AND the claim as ONE atomic synchronous check-and-set.

        The legacy submit path checked ``_canonical in self._active_file_ops``
        ~40 lines BEFORE the add-loop that claimed the locks. Correct today only
        because no ``await`` sits between them under cooperative asyncio — any
        future await inserted in that window would let two workers both pass the
        check and double-apply the same file, defeating the split-brain guard.
        This method is deliberately SYNCHRONOUS (structurally cannot yield the
        event loop between evaluation and mutation) and all-or-nothing: on any
        conflict NOTHING is claimed.

        Returns ``(claimed_canonical_paths, None)`` on success or
        ``(None, conflicting_path)`` on conflict. The caller MUST release the
        claimed paths in its ``finally`` (unchanged legacy discard loop)."""
        import pathlib as _pl_claim
        _canonicals: list = []
        for _fp in target_files or ():
            _canonicals.append(str(_pl_claim.Path(_fp).resolve()))
        for _c in _canonicals:
            if _c in self._active_file_ops:
                return None, _c            # all-or-nothing: nothing claimed
        for _c in _canonicals:
            self._active_file_ops.add(_c)
        return _canonicals, None

    async def submit(
        self,
        ctx: OperationContext,
        trigger_source: str = "unknown",
    ) -> OperationResult:
        """Submit an operation for governed execution.

        THE single entrypoint for all triggers (CLI, API, etc.).
        """
        start_time = time.monotonic()

        # Gate: Slice 53 dual-lane total-outage breaker. When a verified total
        # vendor blackout has tripped the breaker (JARVIS_TOTAL_OUTAGE_THRESHOLD
        # consecutive ops exhausted BOTH the streaming and batch lanes with no
        # candidate), pause target allocations — refuse new ops so the loop
        # idles into a clean idle_timeout exit-0 rather than burning retry
        # tokens against a confirmed-dead provider. Single-lane degradation
        # never trips it (a working batch lane yields candidates that reset the
        # counter), so Slice 41 ACTIVE_BATCH_ONLY is preserved.
        try:
            from backend.core.ouroboros.governance.dual_lane_breaker import (
                get_dual_lane_breaker,
            )
            _dlb = get_dual_lane_breaker()
        except Exception:  # noqa: BLE001 — defensive; never block submit on import
            _dlb = None
        if _dlb is not None and _dlb.is_tripped():
            if not getattr(self, "_dual_lane_pause_logged", False):
                logger.error(
                    "[GovernedLoop] Dual-lane outage breaker TRIPPED (diag=%s) "
                    "— pausing target allocations; loop will idle to a clean "
                    "exit. Disable via JARVIS_DUAL_LANE_BREAKER_ENABLED=false.",
                    _dlb.snapshot().last_diagnostic,
                )
                self._dual_lane_pause_logged = True
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code="dual_lane_outage_paused",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result

        # Gate: service must be active
        if self._state not in (ServiceState.ACTIVE, ServiceState.DEGRADED):
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code=f"service_not_active:{self._state.name}",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result

        # Gate: concurrency limit
        if len(self._active_ops) >= self._config.max_concurrent_ops:
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code="busy",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result

        # Gate: dedup
        dedupe_key = ctx.op_id
        if dedupe_key in self._active_ops:
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code="duplicate:in_flight",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result
        if dedupe_key in self._completed_ops:
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code="duplicate:already_completed",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result

        # Gate: file-scope in-flight lock (before acquiring — prevents self-cancel).
        # Slice 1 TOCTOU remediation: check-and-claim is ONE atomic synchronous
        # operation (_try_claim_file_ops) — the event loop structurally cannot
        # yield between evaluation and mutation. All-or-nothing: on conflict
        # nothing is claimed; on success the finally below releases the claim.
        _locked_files, _lock_conflict = self._try_claim_file_ops(ctx.target_files)
        if _locked_files is None:
            logger.warning(
                "[GovernedLoop] File-scope lock: %r already in-flight — "
                "rejecting op %s to prevent split-brain apply",
                _lock_conflict,
                ctx.op_id,
            )
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=OperationPhase.CANCELLED,
                reason_code="file_in_flight",
                trigger_source=trigger_source,
                terminal_class="DEGRADED",
            )
            await self._emit_terminal_events(ctx=ctx, result=result)
            return result

        # Execute pipeline
        self._active_ops.add(dedupe_key)
        # P2 Slice 3 — Universal Convergence registry. Master-
        # gated, NEVER-raise. When ``JARVIS_IN_FLIGHT_REGISTRY_
        # ENABLED=true``, the typed registry mirrors the
        # ``_active_ops`` lifecycle so the
        # :class:`ConvergenceReaper` (Slice 2) has an enriched
        # view of in-flight ops. Off-master path is byte-
        # identical to legacy behavior.
        _register_op_in_flight_safely(
            ctx.op_id, ctx_ref=ctx,
            last_phase_name=getattr(
                getattr(ctx, "phase", None), "name", "",
            ),
            metadata=_op_registry_metadata(ctx),
        )
        # --- Proactive Drive telemetry hook (entry) ---
        # Slice 1: guarded — file locks are already claimed above, so nothing
        # between the claim and the releasing try/finally may raise (a raise
        # here would leak the claim for the session).
        try:
            _pds = getattr(self, "_proactive_drive_service", None)
            if _pds is not None:
                _pds.record_sample("jarvis", depth=len(self._active_ops), latency_ms=0.0)
        except Exception:  # noqa: BLE001 — best-effort telemetry
            logger.debug("[GovernedLoop] proactive-drive sample skipped", exc_info=True)
        # (file locks claimed atomically by _try_claim_file_ops above;
        #  _locked_files released in the finally below — legacy discard loop.)
        try:
            assert self._orchestrator is not None
            # Stamp pipeline_deadline exactly once — shared budget for all downstream phases
            ctx = ctx.with_pipeline_deadline(
                datetime.now(tz=timezone.utc) + timedelta(seconds=self._config.pipeline_timeout_s)
            )

            # Stamp TelemetryContext exactly once at intake
            snap = await self._stack.resource_monitor.snapshot()
            now_ns = time.monotonic_ns()
            host_tel = HostTelemetry(
                schema_version="1.0",
                arch=snap.platform_arch,
                cpu_percent=snap.cpu_percent,           # already quantized
                ram_available_gb=snap.ram_available_gb, # already quantized
                pressure=snap.pressure_for_load(len(self._active_ops)).name,
                sampled_at_utc=datetime.now(tz=timezone.utc).isoformat(),
                sampled_monotonic_ns=snap.sampled_monotonic_ns,
                collector_status=snap.collector_status,
                sample_age_ms=(now_ns - snap.sampled_monotonic_ns) // 1_000_000,
            )
            # Phase 4: 3-layer brain selection gate (task → resource → cost)
            brain = await self._brain_selector.select(
                description=ctx.description,
                target_files=ctx.target_files,
                snapshot=snap,
                blast_radius=len(ctx.target_files),
            )
            logger.info(
                "[GovernedLoop] Brain selected: %s (%s) reason=%s complexity=%s spend=$%.4f",
                brain.brain_id, brain.model_name, brain.routing_reason,
                brain.task_complexity, self._brain_selector.daily_spend,
            )

            # Phase 4: ActiveBrainSet gate — reject brains not admitted by supervisor
            if self._active_brain_set and brain.brain_id not in self._active_brain_set:
                logger.warning(
                    "[GovernedLoop] Brain %r not in admitted set %s — rejecting op %s",
                    brain.brain_id, sorted(self._active_brain_set), ctx.op_id,
                )
                result = OperationResult(
                    op_id=ctx.op_id,
                    terminal_phase=OperationPhase.CANCELLED,
                    reason_code="brain_not_admitted",
                    trigger_source=trigger_source,
                    terminal_class="DEGRADED",
                )
                await self._emit_terminal_events(
                    ctx=ctx,
                    result=result,
                    brain_id=brain.brain_id,
                    model_name=brain.model_name,
                )
                return result

            # Phase 4: create per-op FSM context (starts in RUNNING)
            _fsm_ctx = LoopRuntimeContext(op_id=ctx.op_id)
            self._fsm_contexts[ctx.op_id] = _fsm_ctx
            self._fsm_checkpoint_seq[ctx.op_id] = 0

            # Emit routing narration via CommProtocol
            try:
                await self._stack.comm.emit_heartbeat(
                    op_id=ctx.op_id,
                    phase="brain_routing",
                    progress_pct=3.0,
                )
                # Narrate to voice — uses VoiceNarrator transport if active
                narration = brain.narration()
                await self._stack.comm.emit_intent(
                    op_id=ctx.op_id,
                    goal=narration,
                    target_files=list(ctx.target_files),
                    risk_tier="routing",
                    blast_radius=len(ctx.target_files),
                )
            except Exception:
                pass  # narration is best-effort

            # Short-circuit: cost gate queued heavy task
            if brain.provider_tier == "queued":
                logger.warning(
                    "[GovernedLoop] Cost gate queued op %s (daily_spend=$%.4f)",
                    ctx.op_id, self._brain_selector.daily_spend,
                )
                result = OperationResult(
                    op_id=ctx.op_id,
                    terminal_phase=OperationPhase.CANCELLED,
                    reason_code="cost_gate_triggered_queue",
                    trigger_source=trigger_source,
                    routing_reason=brain.routing_reason,
                    terminal_class="DEGRADED",
                )
                await self._emit_terminal_events(
                    ctx=ctx,
                    result=result,
                    brain_id=brain.brain_id,
                    model_name=brain.model_name,
                )
                return result

            intent_tel = RoutingIntentTelemetry(
                # Phase 1 P0: use brain-derived fields, NOT local Mac pressure.
                # expected_provider and policy_reason now reflect the actual brain
                # selection outcome (host-binding invariant).
                expected_provider=_expected_provider_from_brain(brain),
                policy_reason=_policy_reason_from_brain(brain),
                brain_id=brain.brain_id,
                brain_model=brain.model_name,
                routing_reason=brain.routing_reason,
                task_complexity=brain.task_complexity,
                estimated_prompt_tokens=brain.estimated_prompt_tokens,
                daily_spend_usd=self._brain_selector.daily_spend,
                schema_capability=getattr(brain, "schema_capability", "full_content_only"),
            )
            tc = TelemetryContext(local_node=host_tel, routing_intent=intent_tel)
            ctx = ctx.with_telemetry(tc)

            # Freeze autonomy tier at submit time — GATE reads ctx.frozen_autonomy_tier
            # not live TrustGraduator (prevents promotion races under concurrent ops).
            _canary_slice = _infer_canary_slice(ctx.target_files)
            _frozen_tier = "governed"  # default: backward compat
            if self._trust_graduator is not None:
                _tier_cfg = self._trust_graduator.get_config(
                    trigger_source=trigger_source,
                    repo=ctx.primary_repo,
                    canary_slice=_canary_slice,
                )
                if _tier_cfg is not None:
                    _frozen_tier = _tier_cfg.current_tier.value.lower()
            ctx = ctx.with_frozen_autonomy_tier(_frozen_tier)

            if self._advanced_autonomy is not None:
                try:
                    memory_ctx = self._advanced_autonomy.build_strategic_memory_context(
                        goal=ctx.description,
                        target_files=ctx.target_files,
                    )
                    active_intent = self._advanced_autonomy.remember_user_intent(
                        op_id=ctx.op_id,
                        description=ctx.description,
                        target_files=ctx.target_files,
                        repo_scope=ctx.repo_scope,
                    )
                    ctx = ctx.with_strategic_memory_context(
                        strategic_intent_id=active_intent.intent_id,
                        strategic_memory_fact_ids=memory_ctx.fact_ids,
                        strategic_memory_prompt=memory_ctx.prompt_block,
                        strategic_memory_digest=memory_ctx.context_digest,
                    )
                except Exception as exc:
                    logger.warning(
                        "[GovernedLoop] L4 strategic memory unavailable for op=%s: %s",
                        ctx.op_id,
                        exc,
                    )

            # ── OUROBOROS.md human instruction injection ─────────────────────────
            # Load 3-tier instruction hierarchy and stamp onto ctx before pipeline.
            # Providers prepend this block to every generation prompt.
            try:
                from backend.core.ouroboros.governance.context_memory_loader import (
                    ContextMemoryLoader,
                )
                _instructions = ContextMemoryLoader(
                    project_root=self._config.project_root,
                ).load()
                if _instructions:
                    ctx = ctx.with_human_instructions(_instructions)
            except Exception as _cml_exc:
                logger.debug(
                    "[GovernedLoop] ContextMemoryLoader error (non-fatal): %s", _cml_exc
                )

            # ── Semantic Triage (DW 35B pre-analysis) ────────────────────────
            # Cheap LLM-powered pre-scan: detects no-ops, redirects, and enriches
            # context BEFORE the expensive generation pipeline runs.
            _semantic_triage = getattr(self, "_semantic_triage", None)
            if _semantic_triage is not None and getattr(_semantic_triage, "is_available", False):
                try:
                    from backend.core.ouroboros.governance.semantic_triage import (
                        TriageDecision,
                    )
                    _triage_result = await _semantic_triage.triage(ctx)

                    if _triage_result.decision == TriageDecision.NO_OP:
                        # Change already present — skip generation entirely
                        logger.info(
                            "[GovernedLoop] Semantic triage: NO_OP for op=%s — %s",
                            ctx.op_id[:12], _triage_result.no_op_reason,
                        )
                        duration = time.monotonic() - start_time
                        result = OperationResult(
                            op_id=ctx.op_id,
                            terminal_phase=OperationPhase.COMPLETE,
                            total_duration_s=duration,
                            reason_code="semantic_triage_no_op",
                            trigger_source=trigger_source,
                            routing_reason=brain.routing_reason,
                            terminal_class="NOOP",
                        )
                        self._completed_ops[dedupe_key] = result
                        await self._emit_terminal_events(
                            ctx=ctx, result=result,
                            brain_id=brain.brain_id, model_name=brain.model_name,
                        )
                        return result

                    elif _triage_result.decision == TriageDecision.REDIRECT:
                        # Real problem is in different files — log and inject as
                        # expanded context (OperationContext is immutable on target_files
                        # after creation, so we add redirect targets to expansion list).
                        if _triage_result.redirect_files:
                            logger.info(
                                "[GovernedLoop] Semantic triage: REDIRECT op=%s "
                                "from %s → also consider %s",
                                ctx.op_id[:12], ctx.target_files,
                                _triage_result.redirect_files,
                            )
                            # Add redirect targets as expanded files so the
                            # generation model receives their context too.
                            _expanded = tuple(ctx.expanded_context_files or ()) + tuple(
                                f for f in _triage_result.redirect_files
                                if f not in ctx.target_files
                            )
                            ctx = ctx.with_expanded_files(_expanded)

                    elif _triage_result.decision == TriageDecision.ENRICH:
                        # Inject triage insights into human_instructions
                        # (strategic_memory_context requires full L4 parameters;
                        # human_instructions is a clean append-only field).
                        _insights = _semantic_triage.format_for_prompt(_triage_result)
                        if _insights:
                            _existing = getattr(ctx, "human_instructions", "") or ""
                            ctx = ctx.with_human_instructions(
                                _existing + _insights,
                            )
                            logger.info(
                                "[GovernedLoop] Semantic triage: ENRICH op=%s "
                                "confidence=%.2f (%d chars injected)",
                                ctx.op_id[:12], _triage_result.confidence,
                                len(_insights),
                            )

                    # Emit triage heartbeat for ALL decisions so the dashboard
                    # can track triage statistics (NO_OP saves, PROCEED rate, etc.)
                    try:
                        await self._stack.comm.emit_heartbeat(
                            op_id=ctx.op_id,
                            phase="semantic_triage",
                            progress_pct=5.0,
                            triage_decision=_triage_result.decision.name,
                            triage_confidence=_triage_result.confidence,
                            triage_reason=getattr(_triage_result, "no_op_reason", ""),
                        )
                    except Exception:
                        pass

                except Exception as _triage_exc:
                    logger.debug(
                        "[GovernedLoop] Semantic triage error (non-fatal): %s", _triage_exc
                    )

            # Connectivity preflight (spends from deadline budget)
            if self._generator is not None and self._ledger is not None:
                early_exit = await self._preflight_check(ctx)
                if early_exit is not None:
                    duration = time.monotonic() - start_time
                    _reason = (
                        getattr(early_exit, "terminal_reason_code", "")
                        or early_exit.phase.name.lower()
                    )
                    _tc = _classify_terminal(early_exit.phase, None, _reason, is_noop=False)
                    result = OperationResult(
                        op_id=ctx.op_id,
                        terminal_phase=early_exit.phase,
                        total_duration_s=duration,
                        reason_code=_reason,
                        trigger_source=trigger_source,
                        routing_reason=brain.routing_reason,
                        terminal_class=_tc,
                    )
                    self._completed_ops[dedupe_key] = result
                    if self._ledger is not None:
                        _proof = _build_proof_artifact(
                            op_id=ctx.op_id,
                            terminal_phase=result.terminal_phase,
                            terminal_class=result.terminal_class,
                            provider_used=result.provider_used,
                            model_id=None,
                            compute_class=self._vm_capability.get("compute_class") if self._vm_capability else None,
                            execution_host=self._vm_capability.get("host") if self._vm_capability else None,
                            fallback_active=(result.terminal_class == "FALLBACK_SUCCESS"),
                            phase_trail=[p.name for p in getattr(ctx, "phase_trail", []) if hasattr(p, "name")],
                            generation_duration_s=result.generation_duration_s or 0.0,
                            total_duration_s=result.total_duration_s or 0.0,
                        )
                        await _record_ledger(ctx, self._ledger, OperationState.FAILED, _proof)
                    await self._emit_terminal_events(
                        ctx=ctx,
                        result=result,
                        brain_id=brain.brain_id,
                        model_name=brain.model_name,
                        rollback_occurred=bool(getattr(early_exit, "rollback_occurred", False)),
                        rollback_reason=_reason,
                    )
                    return result

            _pipeline_timeout = (
                self._config.pipeline_timeout_s + 60.0
            )  # +60s grace beyond deadline for post-COMPLETE bookkeeping
            try:
                # GAP 6: race orchestrator against user stop signal when bus is present.
                # The no-bus path uses shielded_wait_for so ledger writes survive timeout.
                #
                # Phase 1 Step 3C: read the live orchestrator from the § 4
                # bind contract on every dispatch so ``importlib.reload``
                # of ``orchestrator.py`` flips over atomically. The
                # ``orchestrator_ref`` fallback chain returns the
                # currently-bound instance, or falls back to the legacy
                # ``stack.orchestrator`` slot if the bind contract has
                # not been engaged.
                _orch = (
                    self._stack.orchestrator_ref
                    if self._stack is not None
                    and hasattr(self._stack, "orchestrator_ref")
                    else self._orchestrator
                )
                if self._user_signal_bus is not None:
                    _op_task = asyncio.create_task(
                        _orch.run(ctx),
                        name=f"orchestrator/{ctx.op_id}",
                    )
                    _stop_task = asyncio.create_task(
                        self._user_signal_bus.wait_for_stop(),
                        name=f"stop-signal/{ctx.op_id}",
                    )
                    try:
                        _done, _pending = await asyncio.wait(
                            [_op_task, _stop_task],
                            timeout=_pipeline_timeout,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not _stop_task.done():
                            _stop_task.cancel()

                    if _stop_task in _done:
                        # User stop: cancel orchestrator, fire EV_PREEMPT, return CANCELLED.
                        # _op_task.cancel() is fire-and-forget — the task will raise
                        # CancelledError at its next await point and unwind cleanly.
                        _op_task.cancel()
                        self._user_signal_bus.reset()
                        _fsm_ctx_now = self._fsm_contexts.get(ctx.op_id)
                        if self._fsm_executor is not None and _fsm_ctx_now is not None:
                            _preempt_seq = self._fsm_checkpoint_seq.get(ctx.op_id, 0) + 1
                            self._fsm_checkpoint_seq[ctx.op_id] = _preempt_seq
                            _preempt_ti = build_transition_input(
                                op_id=ctx.op_id,
                                phase="GENERATE",
                                event=LoopEvent.EV_PREEMPT,
                                ctx=_fsm_ctx_now,
                                checkpoint_seq=_preempt_seq,
                                metadata={"source": "user_signal_bus"},
                            )
                            try:
                                await self._fsm_executor.apply(_fsm_ctx_now, _preempt_ti)
                            except Exception as _exc:
                                logger.debug("[GovernedLoop] FSM EV_PREEMPT apply failed: %s", _exc)
                        duration = time.monotonic() - start_time
                        result = OperationResult(
                            op_id=ctx.op_id,
                            terminal_phase=OperationPhase.CANCELLED,
                            total_duration_s=duration,
                            reason_code="user_stop",
                            trigger_source=trigger_source,
                            routing_reason=brain.routing_reason,
                            terminal_class=_classify_terminal(
                                OperationPhase.CANCELLED, None, "user_stop", is_noop=False
                            ),
                        )
                        self._completed_ops[dedupe_key] = result
                        await self._emit_terminal_events(
                            ctx=ctx,
                            result=result,
                            brain_id=brain.brain_id,
                            model_name=brain.model_name,
                            rollback_reason="user_stop",
                        )
                        return result

                    elif not _done:
                        # Timeout: cancel op_task to stop the orphaned orchestrator run.
                        _op_task.cancel()
                        # Reset bus in case stop was signalled just as timeout fired.
                        self._user_signal_bus.reset()
                        # Timeout: neither finished — build CANCELLED result and return.
                        duration = time.monotonic() - start_time
                        result = OperationResult(
                            op_id=ctx.op_id,
                            terminal_phase=OperationPhase.CANCELLED,
                            total_duration_s=duration,
                            reason_code="pipeline_timeout",
                            trigger_source=trigger_source,
                            routing_reason=brain.routing_reason,
                            terminal_class=_classify_terminal(
                                OperationPhase.CANCELLED, None, "pipeline_timeout", is_noop=False
                            ),
                        )
                        self._completed_ops[dedupe_key] = result
                        if self._ledger is not None:
                            _proof = _build_proof_artifact(
                                op_id=ctx.op_id,
                                terminal_phase=result.terminal_phase,
                                terminal_class=result.terminal_class,
                                provider_used=result.provider_used,
                                model_id=None,
                                compute_class=self._vm_capability.get("compute_class") if self._vm_capability else None,
                                execution_host=self._vm_capability.get("host") if self._vm_capability else None,
                                fallback_active=False,
                                phase_trail=[p.name for p in getattr(ctx, "phase_trail", []) if hasattr(p, "name")],
                                generation_duration_s=0.0,
                                total_duration_s=result.total_duration_s or 0.0,
                            )
                            await _record_ledger(ctx, self._ledger, OperationState.FAILED, _proof)
                        await self._emit_terminal_events(
                            ctx=ctx,
                            result=result,
                            brain_id=brain.brain_id,
                            model_name=brain.model_name,
                            rollback_reason="pipeline_timeout",
                        )
                        logger.error(
                            "[GovernedLoop] orchestrator.run() exceeded %.0fs hard timeout for op=%s",
                            _pipeline_timeout, ctx.op_id,
                        )
                        return result

                    else:
                        # Op completed normally — retrieve result.
                        terminal_ctx = _op_task.result()

                else:
                    # No signal bus: existing shielded path (ledger writes survive timeout).
                    # Phase 1 Step 3C: reuse the rebind-safe ``_orch`` captured
                    # at the top of the try-block so both dispatch paths
                    # read through the § 4 bind contract.
                    from backend.core.async_safety import shielded_wait_for as _shielded_wf
                    terminal_ctx = await _shielded_wf(
                        _orch.run(ctx),
                        timeout=_pipeline_timeout,
                        name=f"orchestrator.run/{ctx.op_id}",
                    )

            except asyncio.TimeoutError:
                logger.error(
                    "[GovernedLoop] orchestrator.run() exceeded %.0fs hard timeout for op=%s"
                    " (pipeline continues in background to allow COMPLETE phase to finish)",
                    _pipeline_timeout, ctx.op_id,
                )
                duration = time.monotonic() - start_time
                result = OperationResult(
                    op_id=ctx.op_id,
                    terminal_phase=OperationPhase.CANCELLED,
                    total_duration_s=duration,
                    reason_code="pipeline_timeout",
                    trigger_source=trigger_source,
                    routing_reason=brain.routing_reason,
                    terminal_class=_classify_terminal(
                        OperationPhase.CANCELLED, None, "pipeline_timeout", is_noop=False
                    ),
                )
                self._completed_ops[dedupe_key] = result
                if self._ledger is not None:
                    _proof = _build_proof_artifact(
                        op_id=ctx.op_id,
                        terminal_phase=result.terminal_phase,
                        terminal_class=result.terminal_class,
                        provider_used=result.provider_used,
                        model_id=None,
                        compute_class=self._vm_capability.get("compute_class") if self._vm_capability else None,
                        execution_host=self._vm_capability.get("host") if self._vm_capability else None,
                        fallback_active=False,
                        phase_trail=[p.name for p in getattr(ctx, "phase_trail", []) if hasattr(p, "name")],
                        generation_duration_s=0.0,
                        total_duration_s=result.total_duration_s or 0.0,
                    )
                    await _record_ledger(ctx, self._ledger, OperationState.FAILED, _proof)
                await self._emit_terminal_events(
                    ctx=ctx,
                    result=result,
                    brain_id=brain.brain_id,
                    model_name=brain.model_name,
                    rollback_reason="pipeline_timeout",
                )
                return result

            # Phase 4: record actual generation cost for cost gate persistence
            if terminal_ctx.generation:
                _gen = terminal_ctx.generation
                _provider_name = getattr(_gen, "provider_name", "unknown")
                _cost = getattr(_gen, "cost_usd", 0.0) or 0.0
                if _cost > 0.0:
                    self._brain_selector.record_cost(_provider_name, _cost)

            duration = time.monotonic() - start_time
            _provider_used = (
                getattr(terminal_ctx.generation, "provider_name", None)
                if terminal_ctx.generation else None
            )
            _is_noop = bool(
                terminal_ctx.generation and getattr(terminal_ctx.generation, "is_noop", False)
            )
            _gen_duration = (
                getattr(terminal_ctx.generation, "generation_duration_s", None)
                if terminal_ctx.generation else None
            )
            _model_id = (
                getattr(terminal_ctx.generation, "model_id", None)
                if terminal_ctx.generation else None
            )
            _reason_code = (
                getattr(terminal_ctx, "terminal_reason_code", "")
                or terminal_ctx.phase.name.lower()
            )
            _rollback_occurred = bool(getattr(terminal_ctx, "rollback_occurred", False))
            _tc = _classify_terminal(terminal_ctx.phase, _provider_used, _reason_code, is_noop=_is_noop)
            result = OperationResult(
                op_id=ctx.op_id,
                terminal_phase=terminal_ctx.phase,
                provider_used=_provider_used,
                generation_duration_s=_gen_duration,
                total_duration_s=duration,
                reason_code=_reason_code,
                trigger_source=trigger_source,
                routing_reason=brain.routing_reason,  # Phase 1 P0: causal code in ledger
                terminal_class=_tc,
            )

            # ── Canary slice metrics ─────────────────────────────────────────────────
            # record_operation matches file paths against registered slice prefixes.
            # Must be called after duration and _rollback_occurred are computed.
            if self._stack is not None and self._stack.canary is not None:
                _canary_success = terminal_ctx.phase is OperationPhase.COMPLETE
                _canary_files = (
                    getattr(terminal_ctx, "target_files", None) or ctx.target_files
                )
                for _canary_fp in (_canary_files or ()):
                    try:
                        self._stack.canary.record_operation(
                            file_path=str(_canary_fp),
                            success=_canary_success,
                            latency_s=duration,
                            rolled_back=_rollback_occurred,
                        )
                    except Exception as _canary_exc:
                        logger.debug(
                            "[GovernedLoop] canary.record_operation error: %s", _canary_exc
                        )

            self._completed_ops[dedupe_key] = result
            if self._ledger is not None:
                _proof = _build_proof_artifact(
                    op_id=ctx.op_id,
                    terminal_phase=result.terminal_phase,
                    terminal_class=result.terminal_class,
                    provider_used=result.provider_used,
                    model_id=_model_id,
                    compute_class=self._vm_capability.get("compute_class") if self._vm_capability else None,
                    execution_host=self._vm_capability.get("host") if self._vm_capability else None,
                    fallback_active=(result.terminal_class == "FALLBACK_SUCCESS"),
                    phase_trail=[p.name for p in getattr(ctx, "phase_trail", []) if hasattr(p, "name")],
                    generation_duration_s=result.generation_duration_s or 0.0,
                    total_duration_s=result.total_duration_s or 0.0,
                )
                await _record_ledger(
                    ctx, self._ledger,
                    OperationState.APPLIED,
                    _proof,
                )

            if self._advanced_autonomy is not None and terminal_ctx.phase is OperationPhase.COMPLETE:
                try:
                    self._advanced_autonomy.record_verified_outcome(
                        op_id=terminal_ctx.op_id,
                        description=terminal_ctx.description,
                        target_files=terminal_ctx.target_files,
                        repo_scope=terminal_ctx.repo_scope,
                        strategic_intent_id=getattr(terminal_ctx, "strategic_intent_id", ""),
                        provider_used=_provider_used or "",
                        routing_reason=brain.routing_reason,
                        benchmark_result=getattr(terminal_ctx, "benchmark_result", None),
                        is_noop=_is_noop,
                    )
                except Exception as exc:
                    logger.warning(
                        "[GovernedLoop] L4 verified outcome write failed for op=%s: %s",
                        terminal_ctx.op_id,
                        exc,
                    )

            await self._emit_terminal_events(
                ctx=ctx,
                result=result,
                brain_id=brain.brain_id,
                model_name=brain.model_name,
                rollback_occurred=_rollback_occurred,
                rollback_reason=_reason_code,
            )

            # ---- MCP external tool hooks (P5, fire-and-forget) ----
            if self._mcp_client is not None:
                try:
                    if terminal_ctx.phase is OperationPhase.POSTMORTEM:
                        await asyncio.wait_for(
                            self._mcp_client.on_postmortem(terminal_ctx),
                            timeout=12.0,
                        )
                    elif terminal_ctx.phase is OperationPhase.COMPLETE:
                        _applied = list(terminal_ctx.target_files) if not _is_noop else []
                        await asyncio.wait_for(
                            self._mcp_client.on_complete(terminal_ctx, _applied),
                            timeout=12.0,
                        )
                except Exception as _mcp_exc:
                    logger.debug("[GovernedLoop] MCP hook error: %s", _mcp_exc)

            return result

        finally:
            self._active_ops.discard(dedupe_key)
            # P2 Slice 3 — Universal Convergence registry parity.
            # Mirrors the ``_active_ops`` removal so the reaper
            # never re-converges an op whose natural terminal
            # has fired. Master-gated NEVER-raise.
            _unregister_op_in_flight_safely(ctx.op_id)
            # --- Proactive Drive telemetry hook (completion) ---
            _pds = getattr(self, "_proactive_drive_service", None)
            if _pds is not None:
                _elapsed_ms = (time.monotonic() - start_time) * 1000.0
                _pds.record_sample("jarvis", depth=len(self._active_ops), latency_ms=_elapsed_ms)
            for _canonical in _locked_files:
                self._active_file_ops.discard(_canonical)
            # Release intake router file-overlap locks (re-ingest queued signals)
            _intake = getattr(self, "_intake_router", None)
            if _intake is not None:
                try:
                    await _intake.release_op(ctx.op_id)
                except Exception:
                    pass
            # Universal Terminal-State Lock Releaser: the GUARANTEED terminal
            # seam. Every op — promoted, failed, tombstoned, conflict-aborted —
            # exits through here, so bridging it to the central releaser revokes
            # the ingress-side target locks (TestFailureSensor ``_pending_target_keys``)
            # that ``release_op`` above does NOT clear. Closes the sensor-side
            # wedge (soak bt-2026-07-22-174240). Best-effort; never leaks.
            try:
                from backend.core.ouroboros.governance.terminal_lock_releaser import (  # noqa: E501
                    release_locks_for_op as _release_ingress_locks,
                )
                _release_ingress_locks(
                    ctx.op_id, getattr(ctx, "target_files", None),
                )
            except Exception:  # noqa: BLE001
                pass
            # Phase 4: clean up per-op FSM context
            self._fsm_contexts.pop(ctx.op_id, None)
            self._fsm_checkpoint_seq.pop(ctx.op_id, None)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def _emit_outcome_events(
        self,
        *,
        op_id: str,
        terminal_phase: OperationPhase,
        provider_used: str,
        duration_s: float,
        reason_code: str,
        rollback_occurred: bool = False,
        failure_class: str = "",
        affected_files: Sequence[str] = (),
        brain_id: str = "",
        model_name: str = "",
        outcome_source: str = "governed_loop",
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a normalized terminal outcome payload to the autonomy event bus."""
        emitter = getattr(self, "_event_emitter", None)
        if emitter is None:
            return

        resolved_failure_class = (
            failure_class
            or _classify_failure_signal_class(
                reason_code,
                rollback_occurred=rollback_occurred,
            )
        )
        success = (
            terminal_phase is OperationPhase.COMPLETE
            and not rollback_occurred
        )
        payload = {
            "op_id": op_id,
            "brain_id": brain_id,
            "model_name": model_name,
            "terminal_phase": terminal_phase.name,
            "provider": provider_used or "",
            "duration_s": duration_s or 0.0,
            "duration_ms": (duration_s or 0.0) * 1000.0,
            "rollback": rollback_occurred,
            "success": success,
            "error": "" if success else reason_code,
            "failure_class": resolved_failure_class,
            "affected_files": list(affected_files),
            "outcome_source": outcome_source,
        }
        if extra_payload:
            payload.update(extra_payload)

        try:
            await emitter.emit(AutonomyEventEnvelope(
                source_layer="L1",
                event_type=AutonomyEventType.OP_COMPLETED,
                payload=payload,
                op_id=op_id,
            ))
            if rollback_occurred:
                await emitter.emit(AutonomyEventEnvelope(
                    source_layer="L1",
                    event_type=AutonomyEventType.OP_ROLLED_BACK,
                    payload={
                        **payload,
                        "rollback_reason": reason_code,
                    },
                    op_id=op_id,
                ))
        except Exception:
            pass  # fault-isolated

    async def report_external_outcome(
        self,
        *,
        op_id: str,
        terminal_phase: OperationPhase,
        reason_code: str,
        rollback_occurred: bool = False,
        affected_files: Sequence[str] = (),
        provider_used: str = "",
        brain_id: str = "",
        model_name: str = "",
        duration_s: float = 0.0,
        failure_class: str = "",
        outcome_source: str = "external",
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report a terminal outcome that happened outside submit()/orchestrator.

        Used for boot recovery and supervisor/manual rollback flows so L4,
        SafetyNet, and Reactor attribution observe the same event contract.
        """
        await self._emit_outcome_events(
            op_id=op_id,
            terminal_phase=terminal_phase,
            provider_used=provider_used,
            duration_s=duration_s,
            reason_code=reason_code,
            rollback_occurred=rollback_occurred,
            failure_class=failure_class,
            affected_files=affected_files,
            brain_id=brain_id,
            model_name=model_name,
            outcome_source=outcome_source,
            extra_payload=extra_payload,
        )

    async def _emit_terminal_events(
        self,
        *,
        ctx: OperationContext,
        result: OperationResult,
        brain_id: str = "",
        model_name: str = "",
        rollback_occurred: bool = False,
        rollback_reason: str = "",
        failure_class: str = "",
    ) -> None:
        """Emit terminal outcome events to advisory layers with rollback fidelity."""
        reason_code = rollback_reason or result.reason_code
        await self._emit_outcome_events(
            op_id=ctx.op_id,
            terminal_phase=result.terminal_phase,
            provider_used=result.provider_used or "",
            duration_s=result.total_duration_s or 0.0,
            reason_code=reason_code,
            rollback_occurred=rollback_occurred,
            failure_class=failure_class,
            affected_files=ctx.target_files,
            brain_id=brain_id,
            model_name=model_name,
            outcome_source="governed_loop",
        )

        # ---- ConsciousnessBridge outcome hook (Hive Step 2 root-cause fix).
        # record_operation_outcome had ZERO callers — the MemoryEngine's sole
        # write path (file reputations) never ran in the live loop. This seam
        # is deliberately INSIDE _emit_terminal_events: every terminal path —
        # inline AND the background-pool dispatcher (the twin-path drift class
        # that left the first wiring silent on BG ops) — crosses here exactly
        # once. Bridge absent → clean no-op. Fire-and-forget with a bound.
        _cb = getattr(self, "_consciousness_bridge", None)
        if _cb is not None and getattr(_cb, "is_active", False):
            try:
                _tp = getattr(result.terminal_phase, "name",
                              str(result.terminal_phase))
                await asyncio.wait_for(
                    _cb.record_operation_outcome(
                        op_id=ctx.op_id,
                        files_changed=list(ctx.target_files or ()),
                        success=(_tp == "COMPLETE"),
                        failure_reason=failure_class or None,
                    ),
                    timeout=5.0,
                )
            except Exception as _cb_exc:
                logger.debug(
                    "[GovernedLoop] consciousness outcome hook error: %s",
                    _cb_exc)

        # CR4: Violent Ephemeral Teardown -- reap the GPU J-Prime node the instant
        # the A1 DAG reaches terminal (PR opened OR fail-closed). Zero idle GPU.
        # Double-gated (lifecycle + violent-teardown) + fail-soft; default-OFF
        # stays byte-identical because force_teardown is never reached.
        try:
            from .failover_lifecycle import (  # noqa: PLC0415
                lifecycle_enabled,
                violent_teardown_enabled,
                get_failover_controller,
            )
            if lifecycle_enabled() and violent_teardown_enabled():
                _terminal_phase = getattr(
                    getattr(result, "terminal_phase", None), "name",
                    getattr(result, "terminal_phase", "?"),
                )
                await get_failover_controller().force_teardown(
                    reason="a1_terminal:%s" % _terminal_phase,
                )
        except Exception:  # noqa: BLE001 -- teardown is best-effort; never break finalization
            pass

    def health(self) -> Dict[str, Any]:
        """Return structured health report."""
        uptime = (
            time.monotonic() - self._started_at
            if self._started_at
            else 0.0
        )
        return {
            "state": self._state.name,
            "active_ops": len(self._active_ops),
            "completed_ops": len(self._completed_ops),
            "canary_slices": list(self._config.initial_canary_slices),
            "uptime_s": round(uptime, 1),
            "failure_reason": self._failure_reason,
            "provider_fsm_state": (
                self._generator.fsm.state.name
                if self._generator
                else "no_generator"
            ),
            "execution_graph_scheduler": (
                self._subagent_scheduler.health()
                if self._subagent_scheduler is not None
                else {"running": False, "reason": "disabled"}
            ),
            "strategic_memory": (
                self._advanced_autonomy.memory_stats()
                if self._advanced_autonomy is not None
                else {"enabled": False, "reason": "disabled"}
            ),
            "orphan_saga_branches": self._detect_orphan_branches(),
            "saga_bus": self._saga_bus.to_dict() if getattr(self, "_saga_bus", None) else {},
        }

    def _detect_orphan_branches(self) -> List[str]:
        """Detect orphaned saga branches across registered repos."""
        try:
            from backend.core.ouroboros.governance.saga.repo_lock import RepoLockManager
            mgr = RepoLockManager()
            # Prefer live registry (self._repo_registry) over config
            registry = self._repo_registry or getattr(self._config, "repo_registry", None)
            if registry is not None:
                roots = {
                    rc.name: rc.local_path
                    for rc in registry.list_enabled()
                }
            else:
                roots = {"jarvis": self._config.project_root}
            return mgr.detect_orphan_branches(roots)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Private: Preflight
    # ------------------------------------------------------------------

    async def _preflight_check(
        self,
        ctx: OperationContext,
    ) -> Optional[OperationContext]:
        """Run connectivity preflight after deadline is stamped.

        Called from submit() immediately after pipeline_deadline is set on ctx.
        Checks remaining budget and probes the primary provider.

        Returns:
            None                 — preflight passed; caller should proceed.
            OperationContext     — early-exit ctx (CANCELLED); caller returns it.
        """
        # NOTE: File-scope in-flight lock is checked in submit() before the pipeline
        # starts — not here.  Checking here would cause self-cancellation because
        # submit() adds files to _active_file_ops before calling the orchestrator.

        # --- Cooldown guard: block if same file touched >3 times in 10 min ---
        import collections as _collections
        import time as _time
        import pathlib as _pathlib_cooldown
        _COOLDOWN_WINDOW_S = 600.0   # 10 minutes
        _COOLDOWN_MAX_HITS = 3
        _now = _time.monotonic()
        for _fp in ctx.target_files:
            _canonical_fp = str(_pathlib_cooldown.Path(_fp).resolve())
            if _canonical_fp not in self._file_touch_cache:
                self._file_touch_cache[_canonical_fp] = _collections.deque()
            _dq = self._file_touch_cache[_canonical_fp]
            # Evict timestamps older than the window
            while _dq and (_now - _dq[0]) > _COOLDOWN_WINDOW_S:
                _dq.popleft()
            _dq.append(_now)
            if len(_dq) > _COOLDOWN_MAX_HITS:
                logger.warning(
                    "[GovernedLoop] Cooldown triggered for file %r "
                    "(%d touches in %.0fs window) — blocking op %s",
                    _canonical_fp,
                    len(_dq),
                    _COOLDOWN_WINDOW_S,
                    ctx.op_id,
                )
                return ctx.advance(OperationPhase.CANCELLED)

        # ── Degradation mode gate ────────────────────────────────────────────────
        _deg_ctrl = getattr(getattr(self, "_stack", None), "degradation", None)
        if _deg_ctrl is not None:
            from backend.core.ouroboros.governance.degradation import DegradationMode
            _deg_mode = _deg_ctrl.mode
            if _deg_mode >= DegradationMode.READ_ONLY_PLANNING:
                logger.warning(
                    "[GovernedLoop] Preflight: degradation_mode=%s blocks op %s",
                    _deg_mode.name,
                    ctx.op_id,
                )
                return ctx.advance(OperationPhase.CANCELLED)
            if _deg_mode == DegradationMode.REDUCED_AUTONOMY:
                # Only SAFE_AUTO ops are allowed in reduced autonomy mode.
                # risk_tier is set during the CLASSIFY phase — it is None at preflight
                # time for the normal pipeline flow.  Treat None as non-SAFE_AUTO
                # (fail-safe): the risk has not been evaluated yet, so we cannot
                # confirm the op is safe to proceed without full autonomy.
                from backend.core.ouroboros.governance.risk_engine import RiskTier
                _risk_tier = ctx.risk_tier
                if _risk_tier not in (RiskTier.SAFE_AUTO, RiskTier.NOTIFY_APPLY):
                    logger.warning(
                        "[GovernedLoop] Preflight: REDUCED_AUTONOMY blocks non-SAFE_AUTO op %s "
                        "(risk_tier=%s — None means not yet classified; fail-safe block)",
                        ctx.op_id,
                        _risk_tier,
                    )
                    return ctx.advance(OperationPhase.CANCELLED)

        # ── Compute-class admission gate ──────────────────────────────────────
        if self._vm_capability is not None:
            _brain_id = (
                ctx.telemetry.routing_intent.brain_id
                if ctx.telemetry is not None and ctx.telemetry.routing_intent is not None
                else None
            )
            if _brain_id:
                # Policy is stored as a list under brains.required; build a lookup dict.
                _policy = getattr(
                    getattr(self._brain_selector, "_brain_selector", None), "_policy", {}
                ) or {}
                _all_brain_entries = (
                    _policy.get("brains", {}).get("required", [])
                    + _policy.get("brains", {}).get("optional", [])
                )
                _brain_cfg: dict = {}
                for _entry in _all_brain_entries:
                    if isinstance(_entry, dict) and _entry.get("brain_id") == _brain_id:
                        _brain_cfg = _entry
                        break
                try:
                    _check_compute_admission(_brain_cfg, self._vm_capability)
                except ComputeClassMismatch as exc:
                    logger.error(
                        "[GLS] Compute admission DENIED for op=%s: %s", ctx.op_id, exc
                    )
                    raise

                # ── Model artifact integrity check ───────────────────────────────────
                # Only enforce for brains that target J-Prime (GPU brains).
                # CPU brains (phi3_lightweight etc.) route to Claude fallback and
                # never consume the GPU VM's model — checking them against the VM's
                # loaded artifact would produce false mismatches.
                _brain_compute = _brain_cfg.get("compute_class", "cpu")
                if _brain_compute != "cpu":
                    try:
                        _check_artifact_integrity(_brain_cfg, self._vm_capability)
                    except ModelArtifactMismatch as exc:
                        logger.error(
                            "[GLS] Artifact integrity DENIED for op=%s: %s", ctx.op_id, exc
                        )
                        raise

        now = datetime.now(tz=timezone.utc)
        remaining_s = (
            (ctx.pipeline_deadline - now).total_seconds()
            if ctx.pipeline_deadline
            else 0.0
        )

        # Budget pre-check: cancel immediately if not enough time remains
        if remaining_s < MIN_GENERATION_BUDGET_S:
            cancelled = ctx.advance(OperationPhase.CANCELLED)
            await _record_ledger(
                cancelled,
                self._ledger,
                OperationState.FAILED,
                {"reason_code": "budget_exhausted_pre_generation", "remaining_s": remaining_s},
            )
            logger.warning(
                "[GovernedLoop] Preflight: budget exhausted before generation "
                "(remaining=%.1fs, min=%.1fs); op_id=%s",
                remaining_s,
                MIN_GENERATION_BUDGET_S,
                ctx.op_id,
            )
            return cancelled

        # Connectivity preflight: probe primary provider
        # CandidateGenerator stores the primary provider as _primary (private).
        # health_probe() takes no arguments; wrap with asyncio.wait_for for timeout.
        probe_timeout = min(5.0, remaining_s * 0.05)
        try:
            provider = getattr(self._generator, "_primary", None)
            if provider is None:
                raise RuntimeError("no_primary_provider")
            primary_ok = await asyncio.wait_for(
                provider.health_probe(), timeout=probe_timeout
            )
        except Exception as _probe_exc:
            logger.debug(
                "[GovernedLoop] Preflight: primary probe failed: %s",
                type(_probe_exc).__name__,
            )
            primary_ok = False

        if primary_ok:
            # Primary healthy — proceed normally
            return None

        # Phase 4: fire EV_CONNECTION_LOSS through preemption FSM for audit trail
        if self._fsm_executor is not None:
            _fsm_ctx = self._fsm_contexts.get(ctx.op_id)
            if _fsm_ctx is not None and _fsm_ctx.state == LoopState.RUNNING:
                _seq = self._fsm_checkpoint_seq.get(ctx.op_id, 0) + 1
                self._fsm_checkpoint_seq[ctx.op_id] = _seq
                _ti = build_transition_input(
                    op_id=ctx.op_id,
                    phase="PREFLIGHT",
                    event=LoopEvent.EV_CONNECTION_LOSS,
                    ctx=_fsm_ctx,
                    checkpoint_seq=_seq,
                    metadata={"source": "preflight_probe_failure"},
                )
                try:
                    await self._fsm_executor.apply(_fsm_ctx, _ti)
                    logger.info(
                        "[GovernedLoop] Preemption FSM: op=%s → %s (connection loss)",
                        ctx.op_id, _fsm_ctx.state.value,
                    )
                except Exception as _exc:
                    logger.debug("[GovernedLoop] FSM apply skipped: %s", _exc)

        # Primary unavailable: decide based on FSM state
        # CandidateGenerator.fsm is a FailbackStateMachine; .state is a FailbackState enum.
        fsm = getattr(self._generator, "fsm", None)
        fsm_state = getattr(fsm, "state", None) if fsm is not None else None

        if fsm_state is FailbackState.QUEUE_ONLY:
            # No fallback available — cancel
            cancelled = ctx.advance(OperationPhase.CANCELLED)
            await _record_ledger(
                cancelled,
                self._ledger,
                OperationState.FAILED,
                {"reason_code": "provider_unavailable"},
            )
            logger.warning(
                "[GovernedLoop] Preflight: QUEUE_ONLY + primary unhealthy → CANCELLED; op_id=%s",
                ctx.op_id,
            )
            return cancelled

        # Fallback is active — log informational entry and continue
        await _record_ledger(
            ctx,
            self._ledger,
            OperationState.BLOCKED,
            {"reason_code": "primary_unavailable_fallback_active"},
        )
        logger.info(
            "[GovernedLoop] Preflight: primary unavailable, fallback active; op_id=%s",
            ctx.op_id,
        )
        return None

    # ------------------------------------------------------------------
    # Private: Component Construction
    # ------------------------------------------------------------------

    async def _build_components(self) -> None:
        """Build providers, generator, approval provider, and orchestrator."""
        # Wire ledger from stack so _preflight_check can append without orchestrator
        if self._stack is not None:
            self._ledger = getattr(self._stack, "ledger", None)

        # Build RepoRegistry first so providers receive repo_roots at construction time.
        # RepoRegistry.from_env() is synchronous — no ordering dependency prevents this.
        repo_registry = RepoRegistry.from_env()
        enabled_repos = repo_registry.list_enabled()
        logger.info(
            "[GovernedLoop] RepoRegistry enabled repos: %s",
            [r.name for r in enabled_repos],
        )
        repo_roots_map: Dict[str, Path] = {r.name: r.local_path for r in enabled_repos}

        # Build ToolLoopCoordinator if tool-use is enabled via config
        _tool_coordinator = None
        if self._config.tool_use_enabled:
            from backend.core.ouroboros.governance.tool_executor import (
                AsyncProcessToolBackend as _AsyncBE,
                GoverningToolPolicy as _GTP,
                ToolLoopCoordinator as _TLC,
            )
            _rr = repo_roots_map if repo_roots_map else {"jarvis": Path.cwd()}
            _policy  = _GTP(repo_roots=_rr)
            _backend = _AsyncBE(semaphore=asyncio.Semaphore(self._config.max_concurrent_tools))
            # Stash the backend on self so late-bound deps (ExplorationFleet
            # for delegate_to_agent, MCP client, etc.) can be attached after
            # downstream components are constructed further below.
            self._tool_backend = _backend
            # Real-time tool call display callback (Manifesto §7: Absolute Observability)
            # Fires for every lifecycle event emitted by ToolLoopCoordinator:
            # start / success / error / timeout / cancelled / denied.
            # Routed via ToolNarrationChannel (backend.core.ouroboros.governance.tool_narration)
            # which builds a real CommMessage, schedules delivery on the
            # running loop, and fault-isolates transport failures.
            from backend.core.ouroboros.governance.tool_narration import (
                ToolNarrationChannel as _TNC,
            )

            # The channel needs a CommProtocol reference, but the governance
            # stack isn't built yet at this point — use a late-bound lookup
            # inside the callback so the reference resolves at fire time.
            class _LateCommProxy:
                __slots__ = ("_gls",)

                def __init__(self, gls: "GovernedLoopService") -> None:
                    self._gls = gls

                def _comm(self) -> Any:
                    """The CommProtocol this service actually holds, or None.

                    ONE resolver for both entry points, so they cannot name
                    two different attributes. They did: this proxy read
                    ``self._gls._governance_stack``, an attribute that has
                    never existed on GovernedLoopService — the stack is
                    ``self._stack`` (assigned in ``__init__``). ``getattr``
                    with a default made the mistake silent: every tool
                    narration message resolved to no comm, ``_emit``
                    returned before touching a transport, and nothing was
                    counted as a failure because nothing had failed. The
                    cockpit showed intent lines (emitted by the orchestrator
                    through the real comm) and never a single tool call.
                    Measured 2026-09-06: 8 tool rounds in a 240 s window,
                    0 tool blocks on the socket, 0 recorded failures.

                    Resolved late and by the REAL name, with no default that
                    could hide a rename: a missing stack raises here, is
                    caught by the channel's own fault isolation, and is
                    COUNTED — which is the difference between a defect the
                    next reader can see and one that hides for months.
                    """
                    stack = self._gls._stack
                    return getattr(stack, "comm", None) if stack is not None else None

                @property
                def _transports(self) -> Any:
                    _comm = self._comm()
                    return getattr(_comm, "_transports", []) if _comm is not None else []

                async def _emit(self, msg: Any) -> None:
                    _comm = self._comm()
                    if _comm is None:
                        return
                    _emit_fn = getattr(_comm, "_emit", None)
                    if _emit_fn is not None:
                        await _emit_fn(msg)
                        return
                    # Fall back to direct transport fan-out.
                    for t in getattr(_comm, "_transports", []) or []:
                        try:
                            await t.send(msg)
                        except Exception:
                            logger.debug(
                                "[GovernedLoop] tool-narration transport %r failed",
                                t, exc_info=True,
                            )

            self._tool_narration = _TNC(_LateCommProxy(self))

            def _on_tool_call_display(**kwargs: Any) -> None:
                self._tool_narration.emit(
                    op_id=kwargs.get("op_id", "") or "",
                    tool_name=kwargs.get("tool_name", "") or "",
                    round_index=int(kwargs.get("round_index", 0) or 0),
                    args_summary=kwargs.get("args_summary", "") or "",
                    result_preview=kwargs.get("result_preview", "") or "",
                    duration_ms=float(kwargs.get("duration_ms", 0.0) or 0.0),
                    status=kwargs.get("status", "") or "",
                    preamble=kwargs.get("preamble", "") or "",
                )

            _tool_coordinator = _TLC(
                backend=_backend, policy=_policy,
                max_rounds=self._config.max_tool_rounds,
                tool_timeout_s=self._config.tool_timeout_s,
                on_tool_call=_on_tool_call_display,
                min_per_round_s=self._config.tool_min_per_round_s,
                final_write_reserve_s=self._config.tool_final_write_reserve_s,
            )

            # Streaming token callback — pipes tokens through CommProtocol
            # so SerpentFlow can render live Markdown via rich.Live.
            # Uses asyncio.get_running_loop() (the non-deprecated API) and
            # logs delivery failures at DEBUG instead of swallowing them.
            def _on_streaming_token(token: str) -> None:
                try:
                    _gov = getattr(self, "_governance_stack", None)
                    _comm = getattr(_gov, "comm", None) if _gov is not None else None
                    if _comm is None:
                        return
                    try:
                        _loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running loop — tokens can't be delivered.
                        return
                    from backend.core.ouroboros.governance.tool_narration import (
                        _DuckMessage as _TokDuck,
                    )
                    _tok_msg = _TokDuck(
                        op_id="",
                        payload={"streaming": "token", "token": token},
                    )
                    for _t in getattr(_comm, "_transports", []) or []:
                        try:
                            _loop.create_task(_t.send(_tok_msg))
                        except Exception:
                            logger.debug(
                                "[GovernedLoop] streaming-token delivery failed",
                                exc_info=True,
                            )
                except Exception:
                    logger.debug(
                        "[GovernedLoop] streaming-token outer failure",
                        exc_info=True,
                    )
            _tool_coordinator.on_token = _on_streaming_token

            logger.info(
                "[GovernedLoop] ToolLoopCoordinator wired: max_rounds=%d, timeout=%.1fs, concurrency=%d, streaming=ON",
                self._config.max_tool_rounds,
                self._config.tool_timeout_s,
                self._config.max_concurrent_tools,
            )

        primary = None
        fallback = None

        # Build PrimeProvider if PrimeClient available
        _primary_probe_ok = False  # track for FSM sync after generator build
        # Phase 3.4: JIT-boot the local Ollama daemon before the local tier is wired,
        # so its health probe sees a live engine. Gated (JARVIS_LOCAL_DAEMON_GOVERNOR_ENABLED
        # default OFF) + fail-soft: a governor failure never blocks boot.
        try:
            from backend.core.ouroboros.governance.local_daemon_governor import (
                daemon_governor_enabled, LocalDaemonGovernor,
            )
            if daemon_governor_enabled():
                self._local_daemon_governor = LocalDaemonGovernor()
                await self._local_daemon_governor.start_if_enabled()
        except Exception:
            logger.debug("[GovernedLoop] local daemon governor start skipped", exc_info=True)

        # J-Prime tier wiring.
        # Phase 3.2 (tiered, default OFF -> heavy GCP wired but NOT activated):
        #   compose heavy (existing GCP prime_client) + light (governed local Ollama)
        #   into a TieredPrimeClient. Phase 3.1 (default path): local-only injection
        #   when no heavy client is configured. OFF for both -> byte-identical legacy.
        try:
            from backend.core.ouroboros.governance.local_inference_director import (
                build_local_prime_client,
                LocalConfig,
                LocalInferenceDirector,
            )
            from backend.core.ouroboros.governance.tiered_prime_client import (
                tiered_enabled,
                build_tiered_prime_client,
            )

            def _build_governed_local():
                _lp = build_local_prime_client()
                if _lp is not None:
                    _dir = LocalInferenceDirector(LocalConfig.from_env(), client=_lp)
                    _lp.attach_governor(_dir)
                    self._local_inference_director = _dir
                return _lp

            if tiered_enabled():
                _light = _build_governed_local()
                _composite = build_tiered_prime_client(
                    heavy=self._prime_client, light=_light
                )
                if _composite is not None and _composite is not self._prime_client:
                    self._prime_client = _composite
                    logger.info(
                        "[GovernedLoop] J-Prime tiered composite active "
                        "(heavy=%s, light=%s)",
                        self._prime_client is not None,
                        _light is not None,
                    )
            elif self._prime_client is None:
                _local_prime = _build_governed_local()
                if _local_prime is not None:
                    self._prime_client = _local_prime
                    logger.info(
                        "[GovernedLoop] J-Prime local tier: LocalPrimeClient injected (Ollama)"
                    )
        except Exception:
            logger.debug(
                "[GovernedLoop] J-Prime tier wiring skipped", exc_info=True
            )
        if self._prime_client is not None:
            try:
                from backend.core.ouroboros.governance.providers import (
                    PrimeProvider,
                )

                primary = PrimeProvider(
                    self._prime_client,
                    repo_root=self._config.project_root,
                    repo_roots=repo_roots_map,
                    tool_loop=_tool_coordinator,
                )
                try:
                    if await primary.health_probe():
                        logger.info("[GovernedLoop] PrimeProvider: healthy at startup")
                        _primary_probe_ok = True
                    else:
                        logger.warning(
                            "[GovernedLoop] PrimeProvider: unhealthy at startup; "
                            "retained for probe-based recovery"
                        )
                        # Do NOT set primary = None — circuit breaker handles retry
                except Exception as probe_exc:
                    logger.warning(
                        "[GovernedLoop] PrimeProvider: startup probe raised %s; "
                        "retained for probe-based recovery",
                        probe_exc,
                    )
                    # Probe failure (raise) is treated same as probe failure (False):
                    # retain the provider for circuit-breaker-based recovery
            except Exception as exc:
                logger.warning(
                    "[GovernedLoop] PrimeProvider build failed: %s", exc
                )
                primary = None

        # Build ClaudeProvider if API key available OR Aegis is enabled.
        # Slice 2B-ii.1 — Aegis-aware gate. When Aegis is on, the local
        # claude_api_key has been intentionally scrubbed by preflight
        # (the real key is now confiscated to the daemon). We still
        # construct the provider; its _ensure_client() routes through
        # the Aegis bridge (Slice 2B-ii). The api_key is coerced
        # None → "" so the provider's self._api_key is always a string
        # (downstream code in the bridge handles the empty case).
        #
        # Slice 19a (2026-05-26) — provider_name="claude" lets the gate
        # honor JARVIS_PROVIDER_CLAUDE_DISABLED for pure DW-only soaks.
        # When that env is true, ClaudeProvider is NEVER constructed
        # (self._fallback stays None); IMMEDIATE ops fail visibly per
        # Manifesto §5 transparency; DW carries all non-IMMEDIATE routes.
        if not _provider_construction_gate(
            local_api_key=self._config.claude_api_key,
            provider_name="claude",
        ):
            # Slice 19a — diagnose the skip cause so operators can
            # distinguish "no API key + no Aegis" (existing case) from
            # "explicitly disabled via JARVIS_PROVIDER_CLAUDE_DISABLED"
            # (new Slice 19a case).
            _claude_disable_env = os.environ.get(
                "JARVIS_PROVIDER_CLAUDE_DISABLED", "",
            ).strip().lower()
            if _claude_disable_env in ("true", "1", "yes", "on"):
                logger.info(
                    "[GovernedLoop] Slice 19a: ClaudeProvider construction "
                    "SKIPPED — JARVIS_PROVIDER_CLAUDE_DISABLED=true. "
                    "self._fallback stays None; IMMEDIATE-routed ops will "
                    "fail visibly (Manifesto §5 transparency). SWE-Bench "
                    "ops unaffected (Slice 10A → STANDARD route → DW)."
                )
        if _provider_construction_gate(
            local_api_key=self._config.claude_api_key,
            provider_name="claude",
        ):
            try:
                from backend.core.ouroboros.governance.providers import (
                    ClaudeProvider,
                )

                fallback = ClaudeProvider(
                    api_key=(self._config.claude_api_key or ""),
                    model=self._config.claude_model,
                    max_cost_per_op=self._config.claude_max_cost_per_op,
                    daily_budget=self._config.claude_daily_budget,
                    repo_root=self._config.project_root,
                    repo_roots=repo_roots_map,
                    tool_loop=_tool_coordinator,
                )
                # Phase 3a: hold a reference so get_provider_stats() can
                # surface prompt-cache telemetry (hit rate, $ saved).
                self._claude_ref = fallback
                _cache_stats = fallback.get_cache_stats()
                logger.info(
                    "[GovernedLoop] ClaudeProvider: configured "
                    "(prompt_cache=%s, min_chars=%d)",
                    "on" if _cache_stats["enabled"] else "off",
                    _cache_stats["min_chars"],
                )
            except Exception as exc:
                logger.warning(
                    "[GovernedLoop] ClaudeProvider build failed: %s", exc
                )
                fallback = None

        # Build DoublewordProvider (Tier 0 — batch 397B MoE) if API key set
        # OR Aegis is enabled. Slice 2B-ii.1 — Aegis-aware gate (same shape
        # as the ClaudeProvider gate above). Under Aegis the local
        # DOUBLEWORD_API_KEY env has been scrubbed; the empty string
        # propagates to DoublewordProvider.__init__ which is now safe
        # because is_available also composes the Aegis predicate.
        tier0 = None
        _dw_api_key = os.environ.get("DOUBLEWORD_API_KEY", "")
        if _provider_construction_gate(local_api_key=_dw_api_key):
            try:
                from backend.core.ouroboros.governance.doubleword_provider import (
                    DoublewordProvider,
                )
                from backend.core.ouroboros.governance.rate_limiter import (
                    RateLimitService,
                )
                _dw_rate_limiter = RateLimitService()
                # Real-time mode: use /v1/chat/completions with Venom tool loop
                # Real-time SSE mode: /v1/chat/completions with token streaming.
                # Zero polling (Manifesto §3). Battle-tested latency: 20-40s
                # (comparable to batch 16-22s) but with streaming + Venom.
                # Default: ON. Opt out via DOUBLEWORD_REALTIME_ENABLED=false.
                _dw_realtime = os.environ.get(
                    "DOUBLEWORD_REALTIME_ENABLED", "true"
                ).lower() != "false"
                # DW 3-tier architecture (Manifesto §3: Zero polling. Pure reflex.)
                # Tier 1 webhook: BatchFutureRegistry resolves futures via webhook
                # Tier 2 adaptive poll: exponential backoff fallback
                _batch_registry = None
                try:
                    from backend.core.ouroboros.governance.batch_future_registry import (
                        BatchFutureRegistry,
                    )
                    _batch_registry = BatchFutureRegistry()
                    self._batch_registry = _batch_registry
                    logger.info("[GovernedLoop] BatchFutureRegistry: wired (Tier 1 webhook)")
                    # Slice 19 — hand the registry to the soak circuit-breaker so
                    # a trip can cancel active batch queues (fail-soft; inert
                    # when the breaker is unarmed).
                    try:
                        from backend.core.ouroboros.governance.soak_circuit_breaker import (  # noqa: E501
                            get_soak_breaker,
                        )
                        get_soak_breaker().register_batch_registry(_batch_registry)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "[GovernedLoop] soak breaker registry wiring skipped",
                            exc_info=True,
                        )
                except Exception as _bfr_exc:
                    logger.debug("[GovernedLoop] BatchFutureRegistry skipped: %s", _bfr_exc)
                tier0 = DoublewordProvider(
                    api_key=_dw_api_key,
                    repo_root=self._config.project_root,
                    repo_roots=repo_roots_map,
                    rate_limiter=_dw_rate_limiter,
                    tool_loop=_tool_coordinator,  # Always wire Venom (RT uses it, batch ignores it)
                    realtime_enabled=_dw_realtime,
                    batch_registry=_batch_registry,
                )
                self._doubleword_ref = tier0
                _mode = "real-time + Venom" if _dw_realtime else "batch"
                logger.info(
                    "[GovernedLoop] DoublewordProvider: configured (model=%s, mode=%s)",
                    tier0._model, _mode,
                )

                # ============================================================
                # Slice 18 — orphaned-batch reconciliation (§2 Progressive
                # Awakening for the batch plane)
                # ============================================================
                # A session that died — SIGKILL, OOM, power loss, or an orderly
                # exit that ran out of shutdown budget — leaves batches DW is
                # still holding for us. Before Slice 18 they were not merely
                # un-cleaned; they were *unknowable*: BatchFutureRegistry is two
                # in-memory dicts, so every batch_id evaporated with the process
                # while the batch itself kept its slot on DW's queue. The only
                # entity that ever cleaned one up was a human at DoubleWord, who
                # on 2026-07-14 cancelled two of ours by hand and emailed us.
                #
                # The durable claim ledger makes them knowable; this sweep makes
                # them settled. Each orphan is either ADOPTED (a batch that
                # completed while we were down is a free result — we already paid
                # for it) or RELEASED (cancelled, returning the queue slot).
                #
                # Fire-and-forget: reconciliation talks to the network and must
                # never sit in the boot path. Same discipline as the eager
                # discovery arm below.
                try:
                    self._batch_reconcile_task = asyncio.create_task(
                        tier0.reconcile_orphan_batches(),
                        name="dw-batch-reconcile",
                    )
                    # Consume the result so a failed sweep never surfaces as an
                    # "exception was never retrieved" warning. The method is
                    # already fail-soft internally; this is belt-and-braces.
                    self._batch_reconcile_task.add_done_callback(
                        lambda t: (
                            None if t.cancelled() else t.exception()
                        ),
                    )
                except Exception as _rec_exc:  # noqa: BLE001 — never block boot
                    logger.debug(
                        "[GovernedLoop] batch reconciliation not armed: %s",
                        _rec_exc,
                    )

                # ============================================================
                # Phase 12.2 Slice F — Autonomic Pacemaker
                # ============================================================
                # Eradicate the lazy-boot deadlock: in an idle dev environment
                # the only sensor that fires is BacklogSensor (BG-only route).
                # BG never cascades to Claude (project_bg_spec_sealed.md). BG
                # depends on the dynamic catalog. Lazy-boot wires discovery to
                # fire on first DW dispatch — but the empty catalog short-
                # circuits dispatch BEFORE the boot hook runs, deadlocking the
                # entire Phase 12.2 cognitive substrate.
                #
                # The fix: arm discovery EAGERLY at orchestrator startup,
                # asynchronously, BEFORE pulling ops from any queue. Once
                # boot_discovery_once fires, it both:
                #   1. populates the dynamic catalog (one-shot inline cycle)
                #   2. spawns the periodic refresh task (30-min cadence per
                #      JARVIS_DW_CATALOG_REFRESH_S, default 1800s) — a true
                #      autonomic heartbeat independent of operator traffic.
                #
                # Fire-and-forget: never blocks the boot sequence. Worst case
                # (DW endpoint down) the catalog stays empty + retry refresh
                # task keeps trying every 30 min. Exception is swallowed at
                # this seam — orchestrator boot must NEVER fail because
                # discovery had a bad day.
                try:
                    from backend.core.ouroboros.governance.dw_catalog_client import (
                        discovery_enabled as _discovery_enabled,
                    )
                    from backend.core.ouroboros.governance.dw_discovery_runner import (
                        boot_discovery_once as _boot_discovery_once,
                    )
                    if (
                        _discovery_enabled()
                        and getattr(tier0, "is_available", True)
                    ):
                        _pacemaker_session = await tier0._get_session()
                        asyncio.create_task(
                            _boot_discovery_once(
                                session=_pacemaker_session,
                                base_url=tier0._base_url,
                                api_key=tier0._api_key,
                            ),
                            name="dw_autonomic_pacemaker",
                        )
                        logger.info(
                            "[GovernedLoop] Autonomic Pacemaker armed — "
                            "DW catalog discovery + 30-min refresh cadence "
                            "running asynchronously (Phase 12.2 Slice F)",
                        )
                    else:
                        logger.info(
                            "[GovernedLoop] Autonomic Pacemaker skipped "
                            "(discovery_enabled=%s, dw_available=%s)",
                            _discovery_enabled(),
                            getattr(tier0, "is_available", True),
                        )
                except Exception as _pacemaker_exc:  # noqa: BLE001
                    logger.warning(
                        "[GovernedLoop] Autonomic Pacemaker arm failed "
                        "(non-fatal): %s — DW dispatch will fall back to "
                        "lazy boot on first op", _pacemaker_exc,
                    )

                # Boot Semantic Triage Engine (DW 35B pre-analysis)
                try:
                    from backend.core.ouroboros.governance.semantic_triage import (
                        SemanticTriageEngine,
                    )
                    self._semantic_triage = SemanticTriageEngine(
                        dw_provider=tier0,
                        project_root=self._config.project_root,
                    )
                    # Verify triage model is available on DW API (non-blocking, non-fatal)
                    _model_ok = await asyncio.wait_for(
                        self._semantic_triage.verify_model(), timeout=15.0,
                    )
                    if _model_ok:
                        logger.info(
                            "[GovernedLoop] SemanticTriageEngine: booted (model=%s, verified=True)",
                            self._semantic_triage._effective_model,
                        )
                    else:
                        logger.warning(
                            "[GovernedLoop] SemanticTriageEngine: triage model unavailable — disabled",
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[GovernedLoop] SemanticTriageEngine: model verification timed out — "
                        "proceeding with unverified model %s",
                        os.environ.get("OUROBOROS_TRIAGE_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8"),
                    )
                except Exception as _triage_boot_exc:
                    logger.debug(
                        "[GovernedLoop] SemanticTriageEngine boot failed (non-fatal): %s",
                        _triage_boot_exc,
                    )

            except Exception as exc:
                logger.warning(
                    "[GovernedLoop] DoublewordProvider build failed: %s", exc
                )

        # Build CandidateGenerator (needs at least one provider)
        # When J-Prime (primary) is unhealthy at startup and Doubleword (tier0)
        # is available, promote Doubleword to PRIMARY so GENERATE doesn't waste
        # time on J-Prime timeouts. Doubleword.generate() blocks until the batch
        # completes — it's a valid synchronous provider.
        if tier0 is not None and not _primary_probe_ok:
            # Doubleword becomes primary; demote J-Prime to fallback (for recovery)
            _demoted = primary
            primary = tier0
            fallback = _demoted or fallback  # Keep J-Prime as fallback if it exists
            logger.info("[GovernedLoop] Promoting DoublewordProvider to PRIMARY (J-Prime unhealthy)")

        if primary is not None or fallback is not None:
            # If only one provider, use it as both (FSM still works).
            #
            # Slice 20A (2026-05-26) — self-fallback elimination
            #
            # Pre-Slice-20A: `effective_fallback = fallback or primary` blindly
            # promoted primary (e.g., DW) to fallback when no real Tier 1
            # provider existed. This violated Slice 19a's "Claude disabled
            # → no fallback" contract: the FSM ended up with primary=DW
            # AND fallback=DW (same object), and Slice 19b's `fallback is
            # None` guard never fired. Empirically observed in
            # bt-2026-05-26-184355 (v15 soak): cascade fired
            # `fallback_failed` with `fallback_name=doubleword-397b ==
            # primary_name` — DW was called twice for the same op and
            # collided on its own scheduler.
            #
            # Fix: when JARVIS_PROVIDER_CLAUDE_DISABLED=true AND no
            # separate physical secondary provider is supplied, keep
            # effective_fallback strictly None. Slice 19b's `fallback is
            # None` guard fires correctly. The FSM gracefully emits
            # `fallback_skipped:no_fallback_configured`; ExhaustionWatcher
            # filters it; hibernation is reserved for genuine distress.
            _claude_disabled = os.environ.get(
                "JARVIS_PROVIDER_CLAUDE_DISABLED", "",
            ).strip().lower() in ("true", "1", "yes", "on")
            effective_primary = primary or fallback
            if _claude_disabled and fallback is None:
                effective_fallback = None
                logger.info(
                    "[GovernedLoop] Slice 20A: self-fallback ELIMINATED "
                    "— Claude disabled AND no Tier 1 provider supplied; "
                    "effective_fallback=None (Slice 19b's fallback_skipped "
                    "path will engage on cascade)."
                )
            else:
                effective_fallback = fallback or primary
            assert effective_primary is not None
            # Slice 20A — effective_fallback may legitimately be None now;
            # the assert was a pre-Slice-19b invariant that no longer holds.

            _pool_size = int(os.environ.get("JARVIS_BG_POOL_SIZE", "3"))
            _fallback_concurrency = int(os.environ.get(
                "JARVIS_FALLBACK_CONCURRENCY",
                str(min(_pool_size, 4)),
            ))
            logger.info(
                "[GovernedLoop] fallback_concurrency=%d (pool_size=%d, cap=4)",
                _fallback_concurrency, _pool_size,
            )

            # HIBERNATION_MODE step 5: wire the ProviderExhaustionWatcher
            # onto the CandidateGenerator so a sustained provider outage
            # trips the SupervisorOuroborosController into HIBERNATION
            # instead of crashing the loop with all_providers_exhausted.
            _exhaustion_watcher: Any = None
            _controller = (
                getattr(self._stack, "controller", None)
                if self._stack is not None
                else None
            )
            if _controller is not None:
                try:
                    from backend.core.ouroboros.governance.provider_exhaustion_watcher import (  # noqa: PLC0415  # type: ignore[import-not-found]
                        ProviderExhaustionWatcher,
                    )
                    _watcher_instance: Any = ProviderExhaustionWatcher(
                        controller=_controller,
                    )
                    logger.info(
                        "[GovernedLoop] ProviderExhaustionWatcher wired "
                        "(threshold=%s)",
                        getattr(_watcher_instance, "threshold", "?"),
                    )
                    _exhaustion_watcher = _watcher_instance
                except Exception as _watcher_exc:
                    logger.warning(
                        "[GovernedLoop] ProviderExhaustionWatcher wiring "
                        "failed (non-fatal): %s",
                        _watcher_exc,
                    )

            # Phase 3 Scope α — wire the PrimeProvider handle through
            # explicitly so BACKGROUND/SPECULATIVE primacy works even
            # when DoubleWord is promoted to ``primary`` (the handle we
            # captured before demotion is still usable for the primacy
            # path because the cost-optimized routes don't care about
            # J-Prime's first-token latency — they care about the
            # $0.002 vs $0.005 delta). ``primary`` here refers to the
            # local PrimeProvider variable built earlier, which survives
            # the "Doubleword becomes primary" promotion because the
            # demoted reference is reassigned to ``fallback``, not
            # ``primary``. If PrimeProvider was never built (None), the
            # generator simply has no primacy path and the flag is a
            # no-op.
            self._generator = CandidateGenerator(
                primary=effective_primary,
                fallback=effective_fallback,
                fallback_concurrency=_fallback_concurrency,
                tier0=tier0,
                ledger=self._ledger,
                exhaustion_watcher=_exhaustion_watcher,
                jprime=primary,
            )
            self._exhaustion_watcher = _exhaustion_watcher

            # Phase 3.3: give the CandidateGenerator the Oracle graph backend so the
            # exhaustion interceptor can prune by topological centrality. None-safe:
            # the pruner falls back to size-ordering when no backend is available.
            # Oracle stores its SqliteLazyGraphBackend at self._oracle._backend.
            try:
                _gb = None
                _oracle = getattr(self, "_oracle", None)
                if _oracle is not None:
                    _gb = (getattr(_oracle, "_backend", None)
                           or getattr(_oracle, "graph_backend", None)
                           or getattr(_oracle, "graph", None))
                setattr(self._generator, "_graph_backend", _gb)
            except Exception:
                logger.debug("[GovernedLoop] graph backend wiring skipped", exc_info=True)

            # A1 deterministic-fixture DI overlay (Factory-level interception).
            # Default-OFF: when JARVIS_A1_FIXTURE_MODE is inactive this is a
            # byte-identical no-op. When active (the Fast-Forward harness) it
            # swaps self._generator for a FixtureGenerator returning a
            # deterministic AST-mutated candidate with ZERO provider calls, so
            # the REAL VALIDATE -> APPLY -> AutoCommitter path proves the
            # written=True git plumbing decoupled from DW. The production
            # CandidateGenerator constructed above is never modified, and
            # scripts/ is imported ONLY on the fixture path (never in prod).
            #
            # FAIL-CLOSED (A1 run #15 fix): NO try/except swallow here. Under
            # fixture mode, an import or activation failure MUST hard-crash the
            # orchestrator -- a prior `except Exception -> use real generator`
            # silently fell back to DW autarky when the node PYTHONPATH differed,
            # masking the bug as an inconclusive UNKNOWN. We instead normalize
            # sys.path (repo root + scripts/) so the import resolves on macOS or
            # /opt/trinity/jarvis, then apply_fixture_overlay_or_raise() crashes
            # loud if it cannot.
            if (os.environ.get("JARVIS_A1_FIXTURE_MODE", "") or "").strip().lower() in {
                "1", "true", "yes", "on",
            }:
                import sys as _sys
                from pathlib import Path as _Path

                _repo_root = _Path(__file__).resolve().parents[4]
                for _p in (str(_repo_root), str(_repo_root / "scripts")):
                    if _p not in _sys.path:
                        _sys.path.insert(0, _p)
                # No try/except: an import failure propagates = fail-closed crash.
                from a1_deterministic_fixture import (  # noqa: E501
                    apply_fixture_overlay_or_raise,
                )

                apply_fixture_overlay_or_raise(self)
                logger.warning(
                    "[GovernedLoop] A1 fixture generator overlay ACTIVE — "
                    "deterministic AST candidate, zero-DW (written=True proof path)"
                )

            # HIBERNATION_MODE step 6: construct a HibernationProber over the
            # real provider handles and attach it to the watcher so that
            # entering HIBERNATION automatically arms a wake loop. Sequencing:
            # the watcher must exist before the CandidateGenerator is built
            # (so the generator can call record_exhaustion/record_success),
            # while the prober needs the live provider objects — hence the
            # post-construction attach_prober() hook.
            if _exhaustion_watcher is not None and _controller is not None:
                try:
                    from backend.core.ouroboros.governance.hibernation_prober import (  # noqa: PLC0415  # type: ignore[import-not-found]
                        HibernationProber,
                    )
                    # De-dupe providers: effective_primary/effective_fallback
                    # alias to the same object when only one side exists.
                    _probe_targets: list[Any] = []
                    for _candidate in (tier0, effective_primary, effective_fallback):
                        if _candidate is None:
                            continue
                        if any(existing is _candidate for existing in _probe_targets):
                            continue
                        _probe_targets.append(_candidate)
                    _prober_instance: Any = HibernationProber(
                        controller=_controller,
                        providers=_probe_targets,
                    )
                    _attach = getattr(
                        _exhaustion_watcher, "attach_prober", None,
                    )
                    if _attach is not None:
                        _attach(_prober_instance)
                    self._hibernation_prober = _prober_instance
                    logger.info(
                        "[GovernedLoop] HibernationProber wired "
                        "(providers=%d)",
                        len(_probe_targets),
                    )
                except Exception as _prober_exc:
                    logger.warning(
                        "[GovernedLoop] HibernationProber wiring failed "
                        "(non-fatal): %s",
                        _prober_exc,
                    )

            # Sync FSM to reflect actual startup probe result.
            # Without this, the FSM stays at PRIMARY_READY even when the startup
            # probe failed, making the FALLBACK_ACTIVE branch in start() unreachable.
            # SKIP if Doubleword was promoted to primary — Doubleword didn't fail
            # the probe; the original PrimeProvider did. FSM should stay PRIMARY_READY.
            _doubleword_is_primary = (tier0 is not None and effective_primary is tier0)
            self._doubleword_is_primary = _doubleword_is_primary
            if primary is not None and not _primary_probe_ok and not _doubleword_is_primary and self._generator is not None:
                try:
                    self._generator.fsm.record_primary_failure()
                except Exception:
                    pass  # FSM transition error should not abort startup
        else:
            logger.warning(
                "[GovernedLoop] No providers available — QUEUE_ONLY mode"
            )
            self._generator = None

        # Wire L2 RepairEngine if enabled
        _repair_engine = None
        if getattr(self._config.repair_budget, "enabled", False):
            try:
                from backend.core.ouroboros.governance.repair_engine import RepairEngine  # noqa: PLC0415
                if primary is not None:
                    _repair_engine = RepairEngine(
                        budget=self._config.repair_budget,
                        prime_provider=primary,
                        repo_root=self._config.project_root,
                        ledger=self._ledger,
                    )
                    logger.info(
                        "[GovernedLoop] L2 RepairEngine wired: max_iterations=%d, timebox=%.1fs",
                        self._config.repair_budget.max_iterations,
                        self._config.repair_budget.timebox_s,
                    )
                else:
                    logger.warning("[GovernedLoop] L2 disabled: primary provider unavailable")
            except Exception as exc:
                logger.warning("[GovernedLoop] L2 RepairEngine build failed: %s", exc)

        # GAP 6: instantiate user signal bus (always present; silent until request_stop() called)
        self._user_signal_bus = UserSignalBus()

        # Build approval provider via Slice 4 factory: returns
        # InlineApprovalProvider when JARVIS_APPROVAL_UX_INLINE_ENABLED
        # is truthy (default true post-graduation), else legacy
        # CLIApprovalProvider. Single source of truth for selection.
        self._approval_provider = build_approval_provider(
            project_root=self._config.project_root,
        )

        # Wire approval_provider into the tool backend for ask_human tool (Gap #4).
        # The backend was created earlier without it; we inject it now.
        if _tool_coordinator is not None:
            _be = getattr(_tool_coordinator, "_backend", None)
            if _be is not None and hasattr(_be, "_approval_provider"):
                _be._approval_provider = self._approval_provider

        # Build ValidationRunner (LanguageRouter with Python + C++ adapters)
        from backend.core.ouroboros.governance.test_runner import (
            CppAdapter,
            LanguageRouter,
            PythonAdapter,
        )
        from backend.core.ouroboros.governance.workspace_resolver import (
            resolve_repo_root,
        )

        # Run-#13 fix: anchor the post-apply scoped-verify repo_root at the
        # authoritative ``.git`` root, NOT the bare ``self._config.project_root``
        # (which falls back to ``os.getcwd()`` -- see GovernedLoopConfig). On the
        # live node the process CWD did not match where the changed file resolved
        # (``/opt/trinity/jarvis``), so the LanguageRouter / PythonAdapter
        # rejected a perfectly valid ``tests/...py`` as "outside repo root"
        # (``BlockedPathError`` in ``_normalize`` / ``_is_safe_path``) and the
        # scoped-verify silently degraded. ``resolve_repo_root(start=...)`` walks
        # the project_root's parents to its real ``.git`` anchor -- linked-
        # worktree aware (isolation mode) and CWD-independent, the SAME single
        # source of truth the TestWatcher boot-hydration already uses. When the
        # configured project_root is already ``.git``-anchored, the resolved root
        # is identical -> behavior unchanged. Fail-soft: never raises.
        _validation_repo_root = resolve_repo_root(start=self._config.project_root)
        validation_runner = LanguageRouter(
            repo_root=_validation_repo_root,
            adapters={
                "python": PythonAdapter(repo_root=_validation_repo_root),
                "cpp": CppAdapter(repo_root=_validation_repo_root),
            },
        )
        self._validation_runner = validation_runner

        if self._command_bus is None:
            self._command_bus = CommandBus(maxsize=1000)
        if self._event_emitter is None:
            self._event_emitter = EventEmitter()
        if self._config.l4_enabled:
            from backend.core.ouroboros.governance.autonomy.advanced_coordination import (
                AdvancedAutonomyService,
                AdvancedCoordinationConfig,
            )

            self._advanced_autonomy = AdvancedAutonomyService(
                command_bus=self._command_bus,
                config=AdvancedCoordinationConfig(
                    state_dir=self._config.l4_state_dir,
                ),
            )
            if self._event_emitter is not None:
                self._advanced_autonomy.register_event_handlers(self._event_emitter)
            logger.info(
                "[GovernedLoop] L4 AdvancedAutonomyService wired: state_dir=%s",
                self._config.l4_state_dir,
            )
        else:
            self._advanced_autonomy = None

        if self._config.l3_enabled and self._generator is not None:
            from backend.core.ouroboros.governance.autonomy.execution_graph_progress import (
                ExecutionGraphProgressTracker,
                install_default_tracker,
            )
            from backend.core.ouroboros.governance.autonomy.execution_graph_store import (
                ExecutionGraphStore,
            )
            from backend.core.ouroboros.governance.autonomy.subagent_scheduler import (
                GenerationSubagentExecutor,
                SubagentScheduler,
            )
            from backend.core.ouroboros.governance.saga.merge_coordinator import (
                MergeCoordinator,
            )

            # --- Worktree isolation for parallel subagents ---
            _wt_manager = None
            if getattr(self._config, "l3_enable_worktree_isolation", True):
                try:
                    from backend.core.ouroboros.governance.worktree_manager import WorktreeManager
                    _wt_base = getattr(
                        self._config, "l3_worktree_base",
                        Path.home() / ".jarvis" / "ouroboros" / "worktrees",
                    )
                    _wt_manager = WorktreeManager(
                        repo_root=self._config.project_root,
                        worktree_base=_wt_base,
                    )
                    # Gap #3 Slice 5 — hoist to instance attribute so
                    # the EventChannelServer + ide_observability
                    # router can project worktree paths into the
                    # topology GET surface. Local _wt_manager is the
                    # canonical reference; instance attribute is the
                    # cross-block accessor for later wiring.
                    self._worktree_manager = _wt_manager
                    logger.info("[GovernedLoop] WorktreeManager wired: base=%s", _wt_base)

                    # §2 Progressive Awakening: reap orphan worktrees from any
                    # prior SIGKILL/OOM/power-loss. finally-block cleanup covers
                    # normal exits; this covers the rest. Same pattern as
                    # JARVIS_BATTLE_REAP_ZOMBIES in the battle-test harness.
                    if os.environ.get(
                        "JARVIS_WORKTREE_REAP_ORPHANS", "true"
                    ).lower() in ("true", "1"):
                        try:
                            _reaped = await _wt_manager.reap_orphans()
                            if _reaped > 0:
                                logger.info(
                                    "[GovernedLoop] WorktreeManager reaped %d orphan worktree(s) at boot",
                                    _reaped,
                                )
                        except Exception as _reap_exc:  # noqa: BLE001
                            # Reaper must never break boot.
                            logger.warning(
                                "[GovernedLoop] WorktreeManager.reap_orphans failed: %s",
                                _reap_exc,
                            )
                except ImportError:
                    logger.debug("[GovernedLoop] WorktreeManager not available — shared repo mode")

            # ExecutionGraph progress tracker — Phase 3b operational
            # visibility. Subscribes to the scheduler's event emitter
            # and exposes a per-graph snapshot for SerpentFlow.
            self._execution_graph_tracker = ExecutionGraphProgressTracker(
                self._event_emitter
            )
            install_default_tracker(self._execution_graph_tracker)
            logger.info(
                "[GovernedLoop] ExecutionGraphProgressTracker wired (enabled=%s)",
                self._execution_graph_tracker.stats().get("enabled"),
            )
            # Cockpit visibility (2026-07-24): start the tracker→broker
            # forwarder that was built "for the orchestrator's eventual
            # integration" and never called — the wired-but-inert trap.
            # With it running, L3 execution-graph lifecycle surfaces on
            # the canonical broker → mirrored breadcrumb router → every
            # attached ov cockpit. Fail-soft; master
            # JARVIS_EXEC_GRAPH_BRIDGE_ENABLED (default true).
            try:
                from backend.core.ouroboros.governance.execution_graph_progress_bridge import (  # noqa: E501
                    start_default_bridge,
                )
                _egb_task = start_default_bridge()
                logger.info(
                    "[GovernedLoop] ExecutionGraphProgressBridge %s",
                    "started" if _egb_task is not None else "not started (gated/no-loop)",
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GovernedLoop] ExecutionGraphProgressBridge start failed",
                    exc_info=True,
                )

            self._subagent_scheduler = SubagentScheduler(
                store=ExecutionGraphStore(self._config.execution_graph_state_dir),
                command_bus=self._command_bus,
                event_emitter=self._event_emitter,
                executor=GenerationSubagentExecutor(
                    generator=self._generator,
                    validation_runner=validation_runner,
                    repo_roots=repo_roots_map,
                    worktree_manager=_wt_manager,
                ),
                merge_coordinator=MergeCoordinator(),
                max_concurrent_graphs=self._config.max_concurrent_execution_graphs,
                progress_tracker=self._execution_graph_tracker,
            )
            logger.info(
                "[GovernedLoop] L3 SubagentScheduler wired: state_dir=%s max_graphs=%d",
                self._config.execution_graph_state_dir,
                self._config.max_concurrent_execution_graphs,
            )

            # Gap #3 Slice 5 — install the SSE bridge. Best-effort:
            # master-flag-off → no-op + returns None. The bridge
            # subscribes to autonomy EventEmitter events and
            # republishes on the IDE StreamEventBroker so the IDE
            # worktree topology panel can refresh without polling.
            try:
                from backend.core.ouroboros.governance.verification.worktree_topology_sse_bridge import (
                    install_default_bridge as _install_topology_bridge,
                )
                _install_topology_bridge(self._event_emitter)
            except Exception as _bridge_exc:
                logger.debug(
                    "[GovernedLoop] worktree topology SSE bridge "
                    "install raised (non-fatal): %s", _bridge_exc,
                )

            # Manifesto §3 parallel DAG: miner signal coalescer — collapses
            # N same-strategy miner candidates into a single ExecutionGraph
            # instead of N independent ops. Lights up the Phase 3b
            # ExecutionGraphProgressTracker with a real multi-op workload.
            try:
                from backend.core.ouroboros.governance.graph_coalescer import (
                    MinerGraphCoalescer,
                )
                self._graph_coalescer: Optional[Any] = MinerGraphCoalescer(
                    scheduler=self._subagent_scheduler,
                    repo="jarvis",
                )
                logger.info(
                    "[GovernedLoop] MinerGraphCoalescer wired "
                    "(enabled=%s, auto_submit=%s)",
                    self._graph_coalescer.enabled,
                    getattr(self._graph_coalescer, "_auto_submit", False),
                )
            except Exception as _coalescer_exc:
                logger.warning(
                    "[GovernedLoop] MinerGraphCoalescer wire failed: %s",
                    _coalescer_exc,
                )
                self._graph_coalescer = None
        else:
            self._subagent_scheduler = None
            self._graph_coalescer = None

        # Create SagaMessageBus for passive saga observability
        try:
            from backend.core.ouroboros.governance.autonomy.saga_messages import SagaMessageBus
            self._saga_bus = SagaMessageBus(max_messages=500)
            logger.info("[GovernedLoop] SagaMessageBus created (max_messages=500)")
        except ImportError:
            self._saga_bus = None
            logger.debug("[GovernedLoop] SagaMessageBus unavailable — saga_messages not found")

        # Shadow harness — enabled via JARVIS_SHADOW_HARNESS_ENABLED=true
        _shadow_harness = None
        if os.environ.get("JARVIS_SHADOW_HARNESS_ENABLED", "false").lower() in ("true", "1"):
            from backend.core.ouroboros.governance.shadow_harness import ShadowHarness
            _shadow_harness = ShadowHarness(
                confidence_threshold=float(os.environ.get("JARVIS_SHADOW_CONFIDENCE_THRESHOLD", "0.7")),
                disqualify_after=int(os.environ.get("JARVIS_SHADOW_DISQUALIFY_AFTER", "3")),
            )
            logger.info(
                "[GovernedLoop] ShadowHarness wired: confidence_threshold=%.2f, disqualify_after=%d",
                float(os.environ.get("JARVIS_SHADOW_CONFIDENCE_THRESHOLD", "0.7")),
                int(os.environ.get("JARVIS_SHADOW_DISQUALIFY_AFTER", "3")),
            )

        # Build orchestrator
        # When Doubleword is primary, disable context expansion — its batch API
        # is too slow for plan() calls (~2-4 min). Sub-project B provides full
        # file context + AST index, making expansion less critical.
        _dw_primary = getattr(self, "_doubleword_is_primary", False)
        _ctx_expansion = self._config.context_expansion_enabled if hasattr(self._config, "context_expansion_enabled") else True

        # Operator override wins over any inference below. NEVER raises.
        _ctx_override = os.environ.get(
            "JARVIS_CONTEXT_EXPANSION_ENABLED", "",
        ).strip().lower()
        if _ctx_override in ("0", "false", "no", "off"):
            _ctx_expansion = False
            logger.info("[GovernedLoop] Context expansion disabled "
                        "(JARVIS_CONTEXT_EXPANSION_ENABLED=0)")
        elif _ctx_override in ("1", "true", "yes", "on"):
            _ctx_expansion = True
            logger.info("[GovernedLoop] Context expansion forced ON "
                        "(JARVIS_CONTEXT_EXPANSION_ENABLED=1)")
        elif _dw_primary:
            # This used to disable expansion whenever DoubleWord was primary,
            # full stop, because DW was BATCH-ONLY and a plan() round-trip took
            # 2-4 minutes. That judgement was right for its time and its
            # PREMISE has since changed: DW's realtime plane (SSE + priority
            # service_tier) is the default now, with TTFT measured in seconds.
            #
            # The question the guard is really asking is "will an expansion
            # call come back fast enough to be worth making" — a LATENCY
            # question. Phrased as "is DoubleWord primary" it silently
            # answered a provider question instead, and kept answering the old
            # way after the latency changed.
            #
            # The cost of getting it wrong is not small: skipping
            # CONTEXT_EXPANSION skips the ENTIRE architecture-memory arc
            # (ModuleContextRouter, the admission ledger, operator rules) plus
            # Oracle dependency injection. A 28-minute armed soak
            # (bt-2026-07-31-171143) ran 79 ops and logged zero memory
            # routing for exactly this reason.
            from backend.core.ouroboros.governance.doubleword_provider import (
                dw_realtime_plane_active,
            )
            if dw_realtime_plane_active():
                logger.info(
                    "[GovernedLoop] Context expansion KEPT — DoubleWord is "
                    "primary but on the REALTIME plane (SSE + service_tier); "
                    "the batch-latency rationale does not apply",
                )
            else:
                _ctx_expansion = False
                logger.info(
                    "[GovernedLoop] Context expansion disabled — DoubleWord "
                    "primary on the BATCH plane (async lane is too slow for "
                    "plan(); set JARVIS_CONTEXT_EXPANSION_ENABLED=1 to force)",
                )

        orch_config = OrchestratorConfig(
            project_root=self._config.project_root,
            repo_registry=repo_registry,
            generation_timeout_s=self._config.generation_timeout_s,
            context_expansion_timeout_s=self._config.context_expansion_timeout_s,
            context_expansion_enabled=_ctx_expansion,
            approval_timeout_s=self._config.approval_timeout_s,
            message_bus=self._saga_bus,
            repair_engine=_repair_engine,
            execution_graph_scheduler=self._subagent_scheduler,
            shadow_harness=_shadow_harness,
        )
        self._orchestrator = GovernedOrchestrator(
            stack=self._stack,
            generator=self._generator,
            approval_provider=self._approval_provider,
            config=orch_config,
            validation_runner=validation_runner,
        )

        # W3(7) Slice 3 — attach Class E cancel surface to CostGovernor.
        # CostGovernor needs the registry to emit a Class E:cost record on
        # cap-exceeded. Master + watchdog sub-flag both off (default) →
        # the emit path inside cost_governor returns None (byte-for-byte
        # pre-W3(7)). Attach is best-effort; if cost_governor lacks the
        # method (older test fixture), silently skip.
        try:
            _cost_gov = getattr(self._orchestrator, "_cost_governor", None)
            if _cost_gov is not None and hasattr(_cost_gov, "attach_cancel_surface"):
                _cost_gov.attach_cancel_surface(
                    registry=self._cancel_token_registry,
                    session_dir=getattr(self, "_session_dir", None),
                )
        except Exception as _attach_exc:  # noqa: BLE001 — best-effort
            logger.debug("[GLS] Class E cancel surface attach skipped: %s", _attach_exc)

        # ---- Wire ReasoningChainBridge (P1) ----
        try:
            from backend.core.ouroboros.governance.reasoning_chain_bridge import ReasoningChainBridge
            _reasoning_bridge = ReasoningChainBridge(comm=self._stack.comm)
            if _reasoning_bridge.is_active:
                self._orchestrator.set_reasoning_bridge(_reasoning_bridge)
                logger.info("[GLS] ReasoningChainBridge wired (phase=%s)",
                            getattr(getattr(_reasoning_bridge, '_orchestrator', None), '_config', None))
            else:
                logger.debug("[GLS] ReasoningChainBridge: chain not active (env flags not set)")
        except Exception as exc:
            logger.debug("[GLS] ReasoningChainBridge skipped: %s", exc)

        # ---- Wire InfrastructureApplicator (Boundary Principle: deterministic post-APPLY) ----
        try:
            from backend.core.ouroboros.governance.infrastructure_applicator import (
                InfrastructureApplicator,
            )
            _infra = InfrastructureApplicator(project_root=self._config.project_root)
            if _infra.is_enabled:
                self._orchestrator.set_infra_applicator(_infra)
                logger.info("[GLS] InfrastructureApplicator wired (deterministic post-APPLY)")
            else:
                logger.debug("[GLS] InfrastructureApplicator: disabled via env")
        except Exception as exc:
            logger.debug("[GLS] InfrastructureApplicator skipped: %s", exc)

        # ---- Wire GovernanceMCPClient (P5) ----
        self._mcp_client = None
        try:
            from backend.core.ouroboros.governance.mcp_tool_client import GovernanceMCPClient
            _mcp = GovernanceMCPClient()
            if _mcp.is_enabled:
                self._mcp_client = _mcp
                # Gap #7: inject MCP client into tool backend for MCP tool dispatch
                if _tool_coordinator is not None:
                    _be = getattr(_tool_coordinator, "_backend", None)
                    if _be is not None and hasattr(_be, "_mcp_client"):
                        _be._mcp_client = _mcp
                # Gap #7: inject MCP client into providers for tool section rendering
                if self._doubleword_ref is not None and hasattr(self._doubleword_ref, "_mcp_client"):
                    self._doubleword_ref._mcp_client = _mcp
                logger.info("[GLS] GovernanceMCPClient wired (tools forwarded to generation context)")
            else:
                logger.debug("[GLS] GovernanceMCPClient: no servers configured")
        except Exception as exc:
            logger.debug("[GLS] GovernanceMCPClient skipped: %s", exc)

        # ---- Wire EventChannelServer (DW 3-tier: webhook-driven batch) ----
        #
        # Phase B Slice 2 coexistence note:
        #   IntakeLayerService now owns the authoritative EventChannelServer
        #   activation (see ``IntakeLayerService._maybe_start_event_channel_server``).
        #   It holds the GitHubIssueSensor reference and starts the HTTP
        #   receiver when ``JARVIS_GITHUB_WEBHOOK_ENABLED=true``.
        #
        #   The GLS path below is kept as a fallback for deployments where
        #   ``self._intake_router`` is attached externally (legacy DI) — in
        #   those setups the IntakeLayer flow doesn't execute. If the
        #   IntakeLayer has already started a server on the same port, we
        #   short-circuit to avoid a port conflict and a duplicate HTTP
        #   listener. Detection: check whether ``_intake`` carries an
        #   ``_event_channel_server`` attribute already (IntakeLayer sets it).
        self._event_channel = None
        try:
            from backend.core.ouroboros.governance.event_channel import EventChannelServer
            _batch_reg = getattr(self, "_batch_registry", None)
            _intake = getattr(self, "_intake_router", None)
            _intake_layer = getattr(self, "_intake_layer", None)
            _intake_layer_server = getattr(
                _intake_layer, "_event_channel_server", None,
            ) if _intake_layer is not None else None
            if _intake_layer_server is not None:
                logger.info(
                    "[GLS] EventChannelServer skipped — IntakeLayer already "
                    "activated (authoritative owner for webhook receiver)",
                )
            elif _intake is not None:
                _evt_channel = EventChannelServer(
                    router=_intake,
                    batch_registry=_batch_reg,
                    # Gap #3 Slice 5 — pass scheduler + WM refs so
                    # the IDE observability worktree topology
                    # routes can project live state. Both default
                    # to None (degrades to 503 cleanly) when L3
                    # isolation isn't wired.
                    scheduler=self._subagent_scheduler,
                    worktree_manager=self._worktree_manager,
                )
                if _evt_channel.is_enabled:
                    await _evt_channel.start()
                    self._event_channel = _evt_channel
                    logger.info(
                        "[GLS] EventChannelServer started (batch_registry=%s)",
                        "wired" if _batch_reg is not None else "none",
                    )
                else:
                    logger.debug("[GLS] EventChannelServer: disabled via env")
        except Exception as exc:
            logger.debug("[GLS] EventChannelServer skipped: %s", exc)

        # ---- Wire ReasoningNarrator (P0 Wiring: WHY-not-WHAT explanations) ----
        try:
            from backend.core.ouroboros.governance.reasoning_narrator import ReasoningNarrator
            _say = getattr(self, "_say_fn", None)
            _narrator = ReasoningNarrator(say_fn=_say)
            self._orchestrator.set_reasoning_narrator(_narrator)
            logger.info("[GLS] ReasoningNarrator wired (explains WHY decisions were made)")
        except Exception as exc:
            logger.debug("[GLS] ReasoningNarrator skipped: %s", exc)

        # ---- Wire OperationDialogueStore (P0 Wiring: reasoning journal) ----
        try:
            from backend.core.ouroboros.governance.operation_dialogue import OperationDialogueStore
            _dialogue_store = OperationDialogueStore()
            self._orchestrator.set_dialogue_store(_dialogue_store)
            logger.info("[GLS] OperationDialogueStore wired (per-op reasoning journal)")
        except Exception as exc:
            logger.debug("[GLS] OperationDialogueStore skipped: %s", exc)

        # ---- Wire PreActionNarrator (real-time WHAT-before-action voice) ----
        try:
            from backend.core.ouroboros.governance.pre_action_narrator import PreActionNarrator
            _say = getattr(self, "_say_fn", None)
            _pan = PreActionNarrator(say_fn=_say)
            self._orchestrator.set_pre_action_narrator(_pan)
            logger.info("[GLS] PreActionNarrator wired (real-time WHAT before each phase)")
        except Exception as exc:
            logger.debug("[GLS] PreActionNarrator skipped: %s", exc)

        # ---- Wire ExplorationFleet (parallel codebase exploration) ----
        try:
            from backend.core.ouroboros.governance.exploration_fleet import ExplorationFleet
            _fleet = ExplorationFleet(jarvis_root=self._config.project_root)
            self._exploration_fleet_ref = _fleet
            self._orchestrator.set_exploration_fleet(_fleet)
            # Phase 2: wire the same fleet into the tool backend so
            # Venom's delegate_to_agent tool can spawn isolated sub-agents.
            _backend_ref = getattr(self, "_tool_backend", None)
            if _backend_ref is not None and hasattr(_backend_ref, "set_exploration_fleet"):
                _backend_ref.set_exploration_fleet(_fleet)
                logger.info(
                    "[GLS] ExplorationFleet wired (orchestrator CONTEXT_EXPANSION + Venom delegate_to_agent)"
                )
            else:
                logger.info(
                    "[GLS] ExplorationFleet wired (orchestrator CONTEXT_EXPANSION only — tool backend not present)"
                )
        except Exception as exc:
            logger.debug("[GLS] ExplorationFleet skipped: %s", exc)

        # ---- Wire Phase 1 SubagentOrchestrator (dispatch_subagent Venom tool) ----
        # Gated by JARVIS_SUBAGENT_DISPATCH_ENABLED (default true as of
        # 2026-04-18 graduation). Construction itself is side-effect-free;
        # the master switch only gates the dispatch path inside
        # SubagentOrchestrator.dispatch(). An unwired orchestrator reference
        # is safe — the tool backend handler returns EXEC_ERROR with a clear
        # message when dispatch_subagent is called without wiring.
        #
        # Step 5 observability: CommProtocolCommSink routes subagent spawn /
        # result events through the same CommProtocol heartbeat channel that
        # carries INTENT / PLAN / HEARTBEAT / DECISION / POSTMORTEM. Sink
        # resolves the live CommProtocol via late-bound lookup — if the
        # governance stack isn't yet constructed (or comm is None), the sink
        # silently no-ops rather than breaking dispatch.
        try:
            _backend_ref = getattr(self, "_tool_backend", None)
            if _backend_ref is not None and hasattr(_backend_ref, "set_subagent_orchestrator"):
                from backend.core.ouroboros.governance.subagent_orchestrator import (
                    SubagentOrchestrator,
                )
                from backend.core.ouroboros.governance.agentic_subagent import (
                    build_default_explore_factory,
                )
                # Phase B factories (REVIEW + PLAN + GENERAL) — registered
                # at boot alongside EXPLORE. Each factory is imported lazily
                # so a broken module doesn't cascade into the GLS boot
                # sequence; the try/except below catches any import-level
                # failure and degrades to EXPLORE-only operation with a
                # DEBUG log.
                from backend.core.ouroboros.governance.agentic_review_subagent import (
                    build_default_review_factory,
                )
                from backend.core.ouroboros.governance.agentic_plan_subagent import (
                    build_default_plan_factory,
                )
                # Phase C Slice 1b Step 0 — swap the default stub factory
                # for the LLM-driver factory. ``build_llm_general_factory``
                # itself falls back to the stub factory when
                # JARVIS_GENERAL_LLM_DRIVER_ENABLED=false, so attaching it
                # unconditionally here is safe (and byte-identical to the
                # old wire-in in the flag-off case).
                from backend.core.ouroboros.governance.agentic_general_subagent import (
                    build_llm_general_factory,
                )
                from backend.core.ouroboros.governance.subagent_comm_sink import (
                    build_comm_sink_from_gls,
                )
                from backend.core.ouroboros.governance.subagent_ledger_sink import (
                    build_ledger_sink_from_gls,
                )
                _sub_comm = build_comm_sink_from_gls(self)
                _sub_ledger = build_ledger_sink_from_gls(self)
                _sub_orch = SubagentOrchestrator(
                    explore_factory=build_default_explore_factory(
                        self._config.project_root
                    ),
                    review_factory=build_default_review_factory(
                        self._config.project_root
                    ),
                    plan_factory=build_default_plan_factory(
                        self._config.project_root
                    ),
                    general_factory=build_llm_general_factory(
                        self._config.project_root,
                        provider_registry=self._resolve_provider_for_subagent,
                    ),
                    # Narration WRAPS the CommProtocol sink rather than
                    # replacing it: the orchestrator takes one `comm`, and
                    # swapping it would cost the spine that carries these
                    # events to the ledger, the observability API and the SSE
                    # stream. Wrapped, the inner sink sees every event
                    # unchanged and the cockpit gains ⏺/⎿ chrome beside it.
                    #
                    # Emits through the SAME markup mirror every op-chrome
                    # line uses — resolved late, per event, because
                    # SerpentFlow attaches after this stack is built and a
                    # handle captured here would be permanently None.
                    comm=_wrap_subagent_narration(self, _sub_comm),
                    ledger=_sub_ledger,
                )
                self._subagent_orchestrator_ref = _sub_orch
                _backend_ref.set_subagent_orchestrator(_sub_orch)
                # Phase B Slice 1a: also attach to the governance orchestrator
                # so the post-VALIDATE REVIEW shadow hook has access. Observer
                # only — behavior is gated by JARVIS_REVIEW_SUBAGENT_SHADOW.
                if (
                    self._orchestrator is not None
                    and hasattr(self._orchestrator, "set_subagent_orchestrator")
                ):
                    self._orchestrator.set_subagent_orchestrator(_sub_orch)
                logger.info(
                    "[GLS] SubagentOrchestrator wired with Phase 1 EXPLORE + "
                    "Phase B REVIEW/PLAN/GENERAL factories "
                    "(Venom dispatch_subagent — default enabled after "
                    "Phase 1 graduation 2026-04-18; set "
                    "JARVIS_SUBAGENT_DISPATCH_ENABLED=false to disable; "
                    "observability via CommProtocol heartbeats + "
                    "OperationLedger SUBAGENT_DISPATCH records)"
                )
            else:
                logger.debug(
                    "[GLS] SubagentOrchestrator skipped — tool backend not present"
                )
        except Exception as exc:
            logger.debug("[GLS] SubagentOrchestrator skipped: %s", exc)

        # ---- Wire Self-Critique Engine (Phase 3a — post-VERIFY quality signal) ----
        # Cheap DW critique over the applied diff against the original goal.
        # Poor ratings become FEEDBACK memories; excellent ratings reinforce
        # file reputation. Fully non-blocking — all failures are swallowed.
        self._critique_engine = None
        try:
            _self_critique_enabled = (
                os.environ.get("JARVIS_SELF_CRITIQUE_ENABLED", "true").lower() == "true"
            )
            if _self_critique_enabled:
                from backend.core.ouroboros.governance.self_critique import (
                    CritiqueEngine,
                    DoublewordCritiqueProvider,
                    set_default_engine,
                )
                from backend.core.ouroboros.governance.user_preference_memory import (
                    get_default_store,
                )
                _dw_ref = getattr(self, "_doubleword_ref", None)
                if _dw_ref is not None:
                    _critique_provider = DoublewordCritiqueProvider(
                        dw_provider=_dw_ref,
                        max_tokens=int(
                            os.environ.get("JARVIS_CRITIQUE_MAX_TOKENS", "512")
                        ),
                    )
                    _user_store = None
                    try:
                        _user_store = get_default_store()
                    except Exception:
                        _user_store = None
                    _memory_engine = None
                    try:
                        _consciousness = getattr(self, "_consciousness", None)
                        if _consciousness is not None:
                            _memory_engine = getattr(
                                _consciousness, "memory_engine", None,
                            )
                    except Exception:
                        _memory_engine = None
                    self._critique_engine = CritiqueEngine(
                        provider=_critique_provider,
                        repo_root=self._config.project_root,
                        user_preference_store=_user_store,
                        memory_engine=_memory_engine,
                    )
                    self._orchestrator.set_critique_engine(self._critique_engine)
                    set_default_engine(self._critique_engine)
                    logger.info(
                        "[GLS] Self-critique engine wired (provider=doubleword, "
                        "user_prefs=%s, memory_engine=%s)",
                        "yes" if _user_store is not None else "no",
                        "yes" if _memory_engine is not None else "no",
                    )
                else:
                    logger.info(
                        "[GLS] Self-critique skipped: DoublewordProvider not available"
                    )
            else:
                logger.info("[GLS] Self-critique disabled via JARVIS_SELF_CRITIQUE_ENABLED")
        except Exception as exc:
            logger.debug("[GLS] Self-critique wiring skipped: %s", exc)

        # ---- Wire UnlimitedFleetOrchestrator (recursive agent spawning) ----
        self._unlimited_fleet = None
        try:
            from backend.core.ouroboros.governance.unlimited_agents import UnlimitedFleetOrchestrator
            self._unlimited_fleet = UnlimitedFleetOrchestrator(
                jarvis_root=self._config.project_root,
            )
            logger.info("[GLS] UnlimitedFleetOrchestrator wired (recursive agent spawning)")
        except Exception as exc:
            logger.debug("[GLS] UnlimitedFleetOrchestrator skipped: %s", exc)

        # ---- Wire HybridTeammateExecutor (coroutine/subprocess routing) ----
        self._hybrid_executor = None
        try:
            from backend.core.ouroboros.governance.hybrid_teammate_executor import (
                HybridTeammateExecutor,
            )
            self._hybrid_executor = HybridTeammateExecutor(
                project_root=self._config.project_root,
            )
            logger.info(
                "[GLS] HybridTeammateExecutor wired (cognitive=coroutine, mutation=subprocess)"
            )
        except Exception as exc:
            logger.debug("[GLS] HybridTeammateExecutor skipped: %s", exc)

        # ---- Wire BackgroundAgentPool (non-blocking operation submission) ----
        self._bg_pool = None
        try:
            from backend.core.ouroboros.governance.background_agent_pool import (
                BackgroundAgentPool,
            )

            # Move 2 v5 — Unified Observability hooks. BG ops register
            # into the same ``_active_ops`` set foreground ops use, AND
            # get a minimal LoopRuntimeContext in ``_fsm_contexts`` so the
            # harness ActivityMonitor's staleness check + Phase-Aware
            # Heartbeats apply to them too. The op_id passed here is the
            # *context*'s op_id — the same one used by everything
            # downstream (FSM, telemetry, ledgers).
            def _bg_register_active(op_id: str) -> None:
                if not op_id:
                    return
                self._active_ops.add(op_id)
                # P2 Slice 3 — paired registry parity. BG ops
                # are the Fix-A class (fire-and-forget); without
                # a ctx_ref here the reaper's ``_MinimalCtxShim``
                # still gets the SSE flowing. Master-gated
                # NEVER-raise.
                _register_op_in_flight_safely(
                    op_id,
                    metadata={"source": "bg_pool"},
                )
                # Create a minimal FSM context if one doesn't already
                # exist — gives the staleness check something to read,
                # and the stream-tick activity hook a target for
                # last_activity_at_utc updates.
                if op_id not in self._fsm_contexts:
                    self._fsm_contexts[op_id] = LoopRuntimeContext(
                        op_id=op_id,
                    )
                logger.debug(
                    "[GovernedLoop] BG op registered into _active_ops: %s",
                    op_id,
                )

            def _bg_unregister_active(op_id: str) -> None:
                if not op_id:
                    return
                self._active_ops.discard(op_id)
                self._fsm_contexts.pop(op_id, None)
                # P2 Slice 3 — registry parity (master-gated).
                _unregister_op_in_flight_safely(op_id)
                # ── ConsciousnessBridge outcome hook, BG seam (Hive Step 2).
                # BG-pool ops NEVER cross _emit_terminal_events (the twin-path
                # class), so the MemoryEngine ingest fires from this absolute
                # bottom too — same Slice 12R rationale as the telemetry seal
                # below. The engine reads the op ledger itself, so op_id alone
                # is the full contract; the bridge dedupes ops that crossed
                # the inline seam already. Fire-and-forget; NEVER raises.
                try:
                    _cb = getattr(self, "_consciousness_bridge", None)
                    if _cb is not None and getattr(_cb, "is_active", False):
                        asyncio.get_running_loop().create_task(
                            _cb.record_operation_outcome(
                                op_id=op_id, files_changed=[],
                                success=False, failure_reason=None),
                            name=f"consciousness-outcome-{op_id[:16]}",
                        )
                except Exception:  # noqa: BLE001 — cleanup path must not crash
                    pass
                # ── Slice 12R Phase 1 — Telemetry seal ──
                # This is the ABSOLUTE BOTTOM of the BG-op
                # lifecycle: every op that ever registered passes
                # through here on termination (normal, exhausted,
                # cancelled-shutdown, blocked, etc.). The Slice
                # 12Q orchestrator hook fires from _record_ledger
                # but bt-2026-05-23-063408 proved that path is
                # bypassed when shutdown-triggered cancellation
                # propagates through Slice 12O cooldown — the op
                # gets unregistered via this callback without ever
                # reaching _record_ledger.
                #
                # Slice 12R's fallback uses the SessionRecorder's
                # own idempotency (Slice 12Q _recorded_op_ids set):
                # if _record_ledger already recorded this op with
                # rich terminal_reason_code, the fallback below is
                # a no-op (first-write-wins). If _record_ledger
                # never ran (cancellation path), the fallback
                # writes a "cancelled_shutdown" attribution so the
                # op appears in summary.json.operations[] with the
                # correct terminal_reason_class.
                #
                # NEVER raises — telemetry must not crash the
                # critical cleanup path.
                try:
                    from backend.core.ouroboros.battle_test.session_recorder import (  # noqa: E501
                        get_active_recorder,
                    )
                    _recorder = get_active_recorder()
                    if _recorder is not None:
                        # ``cancelled_during_shutdown`` is the
                        # canonical code matched by Slice 12P's
                        # classifier rule at
                        # terminal_reason.py:80 — emitting it
                        # ensures the row carries
                        # ``terminal_reason_class=cancelled_shutdown``
                        # in summary.json. The bare string
                        # ``cancelled_shutdown`` would classify
                        # as OTHER (no substring match).
                        _recorder.record_operation(
                            op_id=op_id,
                            status="cancelled",
                            sensor="bg_pool",
                            technique="cancelled_during_shutdown",
                            composite_score=0.0,
                            elapsed_s=0.0,
                            terminal_reason_code=(
                                "cancelled_during_shutdown"
                            ),
                        )
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[GovernedLoop] Slice 12R telemetry seal "
                        "raised (swallowed) op=%s",
                        op_id, exc_info=True,
                    )
                logger.debug(
                    "[GovernedLoop] BG op unregistered from "
                    "_active_ops: %s",
                    op_id,
                )

            # ────────────────────────────────────────────────────────
            # Slice 26 — process-linked hardware power assertion
            # ────────────────────────────────────────────────────────
            # v19 forensic (bt-2026-05-27-003843): operator's Mac
            # suspended ~6 minutes after SWE-Bench-Pro injection.
            # LoopDeadman correctly fired os._exit(75) on the wedged
            # event loop, but soak-class continuity should not depend
            # on the operator manually wrapping the script with
            # `caffeinate`. Slice 26 spawns a process-linked
            # power-assertion subprocess at boot — the kernel
            # releases automatically when this Python process exits.
            #
            # Fired BEFORE Slice 25B preflight so the host can't sleep
            # during the 10s probe window. Defensive: assertion failure
            # is logged and boot continues (it's an enhancement, not a
            # correctness path).
            try:
                from backend.core.ouroboros.governance.power_supervisor import (
                    assert_power_lock,
                )
                _power_assertion = await assert_power_lock()
                if _power_assertion is not None:
                    logger.debug(
                        "[GLS] Slice 26 power assertion: platform=%s "
                        "parent_pid=%d subprocess_pid=%d binary=%s",
                        _power_assertion.platform,
                        _power_assertion.parent_pid,
                        _power_assertion.subprocess_pid,
                        _power_assertion.binary,
                    )
                # Store on self for visibility in /health observability
                # surfaces (optional; the kernel owns the lifecycle).
                self._power_assertion_ref = _power_assertion
            except Exception as _pwr_exc:  # noqa: BLE001
                # Defensive — power assertion failure must NOT block
                # boot. The assertion module already swallows internal
                # errors; this outer catch is belt-and-suspenders.
                logger.warning(
                    "[GLS] Slice 26 power assertion setup failure "
                    "(boot continues): %r",
                    _pwr_exc,
                )

            # ────────────────────────────────────────────────────────
            # Slice 25B Phase 2 — boot-eager preflight safety gate
            # ────────────────────────────────────────────────────────
            # Probe every trusted DW model in the PromotionLedger BEFORE
            # the BackgroundAgentPool unblocks worker allocations. This
            # is the operator-mandated fail-fast boundary: a network
            # blackout or fleet-wide entitlement failure halts boot
            # cleanly here rather than churning 30+ min of wall-clock
            # via the dispatcher's exhaustion loops (v18 forensic:
            # bt-2026-05-26-233010).
            #
            # Composition:
            #   * Slice 25B Phase 4 auto-activates when
            #     JARVIS_PROVIDER_CLAUDE_DISABLED=true (DW-only posture
            #     makes preflight a hard architectural requirement).
            #   * 403 entitlement → ledger.demote with origin=
            #     account_not_entitled (persisted; future boots inherit
            #     pre-filtered fleet).
            #   * 5xx / timeout → sentinel.report_failure (breaker trips
            #     toward OPEN).
            #   * All-fail → PreflightAllFailedError propagates to
            #     start()'s outer try/except → state=FAILED with
            #     structured diagnostic (clean safe-exit per directive).
            #
            # The probe runs concurrently via asyncio.gather with per-
            # probe asyncio.wait_for(10s) — worst-case wall is 10s
            # regardless of fleet size.
            try:
                from backend.core.ouroboros.governance.preflight_probe import (
                    run_boot_preflight,
                    PreflightAllFailedError as _PreflightAllFailedError,
                )
                _preflight_report = await run_boot_preflight(
                    dw_provider=self._doubleword_ref,
                )
                if _preflight_report is not None:
                    logger.info(
                        "[GLS] Slice 25B preflight: %s",
                        _preflight_report.summary_line(),
                    )
            except _PreflightAllFailedError as _pf_exc:
                # Output structured diagnostic to console (operator
                # directive: "output the structured network diagnostic
                # payload to the console") + re-raise for clean halt
                # via the outer try/except → state=FAILED transition.
                logger.error(
                    "[GLS] Slice 25B preflight FAIL-FAST — halting boot. "
                    "report=%s | per-model verdicts: %s",
                    _pf_exc.report.summary_line(),
                    ", ".join(
                        f"{r.model_id}={r.verdict.value}"
                        f"(status={r.status_code})"
                        for r in _pf_exc.report.results
                    ),
                )
                raise
            except Exception as _pf_setup_exc:  # noqa: BLE001
                # Defensive — preflight substrate failure must not
                # block boot. Only PreflightAllFailedError above is
                # the fail-fast contract; any OTHER exception (import
                # error, missing ledger, etc.) gets swallowed with a
                # warning so boot proceeds via legacy path.
                logger.warning(
                    "[GLS] Slice 25B preflight setup failure (boot continues "
                    "via legacy path): %r",
                    _pf_setup_exc,
                )

            # ---- Slice 40: boot-eager multi-surface DW transport-health ----
            # Probes /v1/files (Surface A) + streaming (Surface B) + Aegis
            # auth (Surface C) once, populates .jarvis/dw_surface_health.json,
            # and emits a SOFT topology-breaker signal if the streaming
            # surface is degraded — so the loop inherits the boot-time
            # health classification (e.g. done_before_content → upstream,
            # flush bypassed) instead of discovering it op-by-op. Self-gated
            # on JARVIS_DW_SURFACE_HEALTH_ENABLED (graduated default TRUE);
            # NEVER raises — must not block boot. Runs AFTER preflight,
            # BEFORE the worker fleet so the first dispatched op sees the
            # populated ledger.
            try:
                from backend.core.ouroboros.governance.preflight_probe import (
                    run_boot_surface_health_sweep,
                )
                _surf_snap = await run_boot_surface_health_sweep(
                    dw_provider=self._doubleword_ref,
                )
                if _surf_snap:
                    logger.info(
                        "[GLS] Slice 40 surface-health sweep: %s",
                        " ".join(
                            f"{k.value}={v.verdict.value}"
                            for k, v in _surf_snap.items()
                        ),
                    )
            except Exception as _surf_exc:  # noqa: BLE001 — never block boot
                logger.warning(
                    "[GLS] Slice 40 surface sweep skipped (boot continues): %r",
                    _surf_exc,
                )

            self._bg_pool = BackgroundAgentPool(
                orchestrator=self._orchestrator,
                on_op_active_register=_bg_register_active,
                on_op_active_unregister=_bg_unregister_active,
            )
            await self._bg_pool.start()

            # PRD §30: register the pool as the proactive-mode emission sink.
            #
            # The `watch` rung withholds INITIATIVE, and this pool is the only
            # thing that can grant or withhold it. Injected here rather than
            # imported there because a mode controller reaching into this
            # service for `_bg_pool` would invert the authority boundary every
            # governance module observes — the same reason `markup_mirror` and
            # `set_prompt_publisher` are injected at their own boot seams.
            #
            # Registered AFTER start(): a sink handed a pool that has not
            # started would pause something with no workers to pause, and
            # `watch` would report an initiative hold it had not achieved.
            #
            # Without this line the ladder still floors the AUTHORITY axis
            # correctly, and `watch` silently degrades to
            # `approval_required` — wired but inert on exactly one of its two
            # axes, which is the failure mode this codebase names most often.
            try:
                from backend.core.ouroboros.governance.proactive_mode import (  # noqa: PLC0415
                    set_emission_sink as _pm_set_sink,
                )
                _pm_set_sink(self._bg_pool)
                logger.debug("[GLS] proactive-mode emission sink registered")
            except Exception:  # noqa: BLE001 — never block boot on the dial
                logger.debug(
                    "[GLS] proactive-mode sink registration skipped",
                    exc_info=True,
                )

            # PRD §30.11 Q4: READ the dial this checkout persisted.
            #
            # `ProactiveModeStore.remember()` wrote `.jarvis/proactive_mode.
            # json` and NOTHING read it back — a write surface with no read
            # surface. The operator set a rung, the file recorded it, and the
            # controller kept answering with its in-memory default. Enabling
            # the ladder therefore resolved to `safe_auto` (grants mutation)
            # while the persisted judgement said `explore` (withholds it):
            # the dial appeared to work and inverted its own meaning.
            #
            # Hydrated HERE because this is where the organism acquires the
            # authority the dial governs. Registered as a standing voter
            # rather than assigned, so it composes through the same
            # strictest-wins path every cockpit uses — one composition rule,
            # not a privileged back door. To loosen it an operator sets the
            # dial, which persists, which moves this floor; the loop closes
            # through the surface that already exists.
            #
            # Before `_bg_pool` can emit: a rung that withholds INITIATIVE
            # must be in force before anything can take any.
            try:
                from backend.core.ouroboros.governance.proactive_mode_store import (  # noqa: E501,PLC0415
                    get_store as _pm_store,
                )
                from backend.core.ouroboros.governance.proactive_mode import (  # noqa: E501,PLC0415
                    PERSISTED_DIAL_VOTER_ID as _pm_voter,
                    get_controller as _pm_controller,
                )
                _rung = await _pm_store().hydrate()
                _eff = _pm_controller().request(_pm_voter, _rung)
                logger.info(
                    "[GLS] proactive dial hydrated: persisted=%s effective=%s",
                    _rung, getattr(_eff, "name", _eff))
            except Exception:  # noqa: BLE001 — a dial fault must not block boot
                # Degrades to the controller default, which is the behaviour
                # that shipped before this hydrate existed. Never to a LOOSER
                # rung than the file asked for by guessing.
                logger.debug(
                    "[GLS] proactive dial hydrate skipped", exc_info=True)

            # ---- The HUD's wire, after the process split (2026-08-15) ----
            #
            # This loop's activity used to reach JARVIS-Apple because the
            # loop and the EventStream shared a process. `ov` owns the loop
            # now, and TrinityEventBus cannot carry the gap: its receive
            # loop drops `event.source == self.local_repo`, and both
            # processes are RepoType.JARVIS — the transport is cross-REPO,
            # not cross-PROCESS. Nothing errors; the phone just goes quiet.
            #
            # Installed HERE because this is the moment the loop exists and
            # its bus is up. Self-gating (disabled by default, and a no-op
            # when the loop is not remote), so a supervisor-owned deployment
            # is byte-identical.
            try:
                from backend.api.governance_cross_process import (  # noqa: E501,PLC0415
                    install_governance_bus_producer as _gx_producer,
                )
                await _gx_producer()
            except Exception:  # noqa: BLE001 — telemetry never blocks a boot
                logger.debug("[GLS] governance bus producer skipped",
                             exc_info=True)

            # PRD §27.5: bind the in-flight op registry for `/why`.
            #
            # An operator asks about the op they are WATCHING, which is by
            # definition the one least likely to have reached disk. Without
            # this binding `/why o-12` answers "not found" for the thing on
            # screen — useless at exactly the moment it is most wanted.
            #
            # Injected, not imported: `why_engine` is a read-only projection
            # over the transcript, and a projection that reached into this
            # service for `_active_ops` would invert the same authority
            # boundary the sink above is placed to preserve. The engine holds
            # no registry and constructs nothing; it is handed a reader and
            # gives it back at teardown.
            #
            # `weak_live_source` holds this service WEAKLY. A GLS that is
            # replaced or crashes without reaching `stop()` would otherwise
            # stay alive inside a closure here and keep answering `/why` from
            # its frozen final state — a dangling pointer that reports live
            # data about a service that has stopped. Weak + explicit unbind
            # in `stop()`/`_teardown_partial()`: the unbind covers the clean
            # path immediately, the weakref covers the crash path without
            # waiting on anyone to remember.
            try:
                from backend.core.ouroboros.governance.why_engine import (  # noqa: PLC0415
                    set_live_source as _why_set_live,
                    weak_live_source as _why_weak,
                )
                # Held so teardown can release BY IDENTITY. Releasing by
                # `set_live_source(None)` would let a stopping instance
                # unbind its own successor during an overlapping restart.
                self._why_live_reader = _why_weak(self, "_active_ops")
                _why_set_live(self._why_live_reader)
                logger.debug("[GLS] /why live source bound (weak)")
            except Exception:  # noqa: BLE001 — `/why` degrades to disk-only
                logger.debug(
                    "[GLS] /why live-source binding skipped", exc_info=True)

            # PRD §28: run every declared audit watchdog once, in background.
            #
            # `reach_repl.run_watchdog` was written, tested, and had ZERO
            # production callers — the ratchet built to catch capabilities
            # that ship unreachable was itself unreachable. Wiring the two
            # instruments by name here would have re-created that defect for
            # the third, so the sweep asks `audit_ratchet` which verbs
            # DECLARE a ratchet: a new instrument mounts by declaration, and
            # this line never changes again.
            #
            # Fire-and-forget, and deliberately not awaited: the scans parse
            # thousands of files, and a watchdog that delayed boot by the
            # length of its own audit is a watchdog somebody deletes the
            # first time they profile startup — and then the instrument is
            # gone again. Each audit already runs off the loop.
            try:
                from backend.core.ouroboros.governance.audit_ratchet import (  # noqa: PLC0415
                    spawn_registered_watchdogs as _spawn_audits,
                )
                self._audit_watchdog_task = _spawn_audits()
            except Exception:  # noqa: BLE001 — a diagnostic never blocks boot
                logger.debug(
                    "[GLS] audit watchdog sweep not scheduled", exc_info=True)

            # Dynamic Fleet Registry Service Discovery: lanes TRACK the mesh.
            # While sovereign endpoints serve, the worker count locks to the
            # node count (one GPU = one lane; an N-node fleet = N lanes);
            # when the fleet empties, the configured size restores. The
            # Immutability Lock rejects env/manifest writers meanwhile.
            try:
                from backend.core.ouroboros.governance.fleet_registry import (  # noqa: PLC0415
                    get_fleet_registry,
                )
                _pool_ref = self._bg_pool
                _configured = int(self._bg_pool._pool_size)
                get_fleet_registry().subscribe(
                    lambda snap: _fleet_lane_sync(_pool_ref, _configured, snap)
                )
                # Reconcile against the CURRENT fleet too (a node may already
                # be serving when the pool boots -- resume/rebind paths).
                _fleet_lane_sync(
                    _pool_ref, _configured, get_fleet_registry().snapshot()
                )
            except Exception:  # noqa: BLE001 -- discovery is enhancement, never blocks boot
                logger.debug("[GLS] fleet lane sync wiring skipped", exc_info=True)
            logger.info(
                "[GLS] BackgroundAgentPool started (pool_size=%d, queue_size=%d)",
                self._bg_pool._pool_size, self._bg_pool._queue_size,
            )
        except Exception as exc:
            logger.debug("[GLS] BackgroundAgentPool skipped: %s", exc)

        # ---- HIBERNATION_MODE step 6.5: bridge controller transitions ----
        # Register hibernation hooks on the SupervisorOuroborosController so
        # that enter_hibernation() actually pauses the BG pool and freezes
        # the idle watchdog (and wake_from_hibernation() restores them).
        # Without this bridge the controller's mode flip is purely cosmetic
        # and the organism keeps burning work during a provider outage,
        # which is the exact failure HIBERNATION is supposed to prevent.
        #
        # Hooks receive a keyword ``reason`` and may be sync — pool.pause /
        # watchdog.freeze are already sync and idempotent. Pause-first,
        # resume-last ordering mirrors lock acquisition: shut the gate
        # before freezing the clock, unfreeze the clock before reopening
        # the gate, so observers never see "unfrozen & paused".
        _ctrl_for_hooks = (
            getattr(self._stack, "controller", None)
            if self._stack is not None
            else None
        )
        if (
            _ctrl_for_hooks is not None
            and hasattr(_ctrl_for_hooks, "register_hibernation_hooks")
        ):
            _bg_ref = self._bg_pool
            _watchdog_ref = getattr(self, "_idle_watchdog", None)

            def _hibernate_bridge(*, reason: str) -> None:
                # Pause the pool first so in-flight workers drain to the
                # unpaused gate before any later hook observes state.
                if _bg_ref is not None:
                    try:
                        _bg_ref.pause(reason=f"hibernation: {reason}")
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[GLS] bg_pool.pause failed under hibernation"
                        )
                if _watchdog_ref is not None:
                    try:
                        _watchdog_ref.freeze(reason=f"hibernation: {reason}")
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[GLS] idle_watchdog.freeze failed under hibernation"
                        )
                # Circadian Resilience (2026-07-18): persist the FSM state +
                # atomic workspace stash at hibernation ENTRY — the exact
                # capture_inflight primitive the wall-clock/predictive gates
                # use (DRY). Survives a process death during the dark window:
                # next ignition (or the wake hydration below) resumes from
                # signed checkpoints instead of losing in-flight work. Sync +
                # fail-soft by capture_inflight's own contract (NEVER raises).
                try:
                    from backend.core.ouroboros.governance.fsm_checkpoint import (  # noqa: PLC0415,E501
                        capture_inflight as _hib_capture,
                    )
                    _n = _hib_capture(reason=f"provider_hibernation:{reason}"[:80])
                    logger.info(
                        "[GLS] hibernation checkpoint: %d in-flight op(s) "
                        "suspended to signed checkpoints (+ workspace stash)", _n,
                    )
                except Exception:  # noqa: BLE001 — never break the transition
                    logger.exception(
                        "[GLS] hibernation capture_inflight failed"
                    )

            async def _wake_bridge(*, reason: str) -> None:
                # Unfreeze first so the watchdog resets its clock before
                # the pool starts dequeuing — avoids a spurious stale-fire
                # immediately on wake against stale _last_poke.
                if _watchdog_ref is not None:
                    try:
                        _watchdog_ref.unfreeze(reason=f"wake: {reason}")
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[GLS] idle_watchdog.unfreeze failed under wake"
                        )
                if _bg_ref is not None:
                    try:
                        _bg_ref.resume(reason=f"wake: {reason}")
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[GLS] bg_pool.resume failed under wake"
                        )
                    # Slice 245 — Absolute-Primacy Re-Ingest. Healthy QUEUED ops
                    # resumed in place above; the SURVIVORS (ops that failed with
                    # provider-exhaustion while the grid was dark) are terminal
                    # and must be re-ingested with Absolute-Max Primacy so they
                    # supersede everything that accumulated during the dark
                    # window. Re-submits the EXACT preserved OperationContext
                    # (no completed work re-computed). Gated + fail-soft.
                    if os.environ.get(
                        "JARVIS_RESURRECTION_REINGEST_ENABLED", "true",
                    ).strip().lower() in ("1", "true", "yes", "on"):
                        try:
                            survivors = _bg_ref.drain_exhaustion_failures()
                            for _ctx in survivors:
                                await _bg_ref.resubmit_resurrected(_ctx)
                            if survivors:
                                logger.info(
                                    "[GLS] wake re-ingested %d hibernation "
                                    "survivor(s) with absolute primacy",
                                    len(survivors),
                                )
                        except Exception:  # noqa: BLE001 — never break wake
                            logger.exception(
                                "[GLS] resurrection re-ingest failed under wake"
                            )
                # Circadian Resilience (2026-07-18): re-hydrate the signed FSM
                # checkpoints written at hibernation entry — the SAME
                # _hydrate_fsm_checkpoints seam the boot ignition uses (DRY),
                # so suspended ops fsm_resume mid-session the moment liquidity
                # returns instead of waiting for the next boot. Idempotent
                # (hydration consumes pending checkpoint files). Fail-soft.
                try:
                    _router_ref = getattr(self, "_intake_router", None)
                    _hydrate = getattr(
                        _router_ref, "_hydrate_fsm_checkpoints", None,
                    )
                    if _hydrate is not None:
                        await _hydrate()
                        logger.info(
                            "[GLS] wake hydration: pending FSM checkpoints "
                            "re-injected (fsm_resume on restored liquidity)",
                        )
                except Exception:  # noqa: BLE001 — never break wake
                    logger.exception("[GLS] wake checkpoint hydration failed")

            try:
                _ctrl_for_hooks.register_hibernation_hooks(
                    on_hibernate=_hibernate_bridge,
                    on_wake=_wake_bridge,
                    name="governed_loop_service",
                )
                self._hibernate_bridge = _hibernate_bridge
                self._wake_bridge = _wake_bridge
                logger.info(
                    "[GLS] hibernation hooks registered on controller "
                    "(pool=%s, watchdog=%s)",
                    "yes" if _bg_ref is not None else "no",
                    "yes" if _watchdog_ref is not None else "no",
                )
            except Exception as _hook_exc:
                logger.warning(
                    "[GLS] hibernation hook registration failed "
                    "(non-fatal): %s",
                    _hook_exc,
                )

        # ---- HIBERNATION_MODE step 7: observability (SerpentFlow + CommProtocol) ----
        # Emit structured CommProtocol messages around every hibernation
        # transition. CommProtocol fans the messages out to all registered
        # transports, which include:
        #
        #   - LogTransport       → debug.log (always)
        #   - SerpentTransport   → renders a proactive_alert Panel in the
        #                          flowing CLI (battle-test sessions)
        #
        # We do NOT import SerpentFlow here — it is owned by the
        # battle-test harness and may not exist in production. The
        # CommProtocol indirection keeps this layer clean.
        #
        # A single hibernation "cycle" shares one op_id across enter/wake so
        # the message sequence reads HEARTBEAT(enter) → DECISION(entered) →
        # HEARTBEAT(wake) → DECISION(wake) → POSTMORTEM in the transport.
        # The controller's hook layer awaits async hooks, so we can use the
        # plain async comm API directly.
        #
        # Master switch: JARVIS_HIBERNATION_OBS_ENABLED (default "true"). Set
        # to "0"/"false" in tests that don't want comm traffic.
        _obs_enabled = os.environ.get(
            "JARVIS_HIBERNATION_OBS_ENABLED", "true"
        ).strip().lower() not in ("0", "false", "no", "off")
        if (
            _obs_enabled
            and _ctrl_for_hooks is not None
            and hasattr(_ctrl_for_hooks, "register_hibernation_hooks")
        ):
            _hibernate_obs_hook, _wake_obs_hook = (
                self._build_hibernation_observability_hooks(self._stack)
            )
            try:
                _ctrl_for_hooks.register_hibernation_hooks(
                    on_hibernate=_hibernate_obs_hook,
                    on_wake=_wake_obs_hook,
                    name="governed_loop_service.observability",
                )
                self._hibernate_obs_hook = _hibernate_obs_hook
                self._wake_obs_hook = _wake_obs_hook
                logger.info(
                    "[GLS] hibernation observability hooks registered "
                    "(comm fan-out: %d transport(s))",
                    len(getattr(
                        getattr(self._stack, "comm", None),
                        "_transports",
                        [],
                    ) or []),
                )
            except Exception as _obs_exc:
                logger.warning(
                    "[GLS] hibernation observability registration failed "
                    "(non-fatal): %s",
                    _obs_exc,
                )

        # ---- Wire LifecycleHookEngine (P1: 15 lifecycle events) ----
        self._hook_engine = None
        try:
            from backend.core.ouroboros.governance.lifecycle_hooks import get_hook_engine, HookEvent
            self._hook_engine = get_hook_engine()
            asyncio.get_event_loop().create_task(
                self._hook_engine.fire(HookEvent.SESSION_START, {"service": "GLS"})
            )
            logger.info("[GLS] LifecycleHookEngine wired (15 event types)")
        except Exception as exc:
            logger.debug("[GLS] LifecycleHookEngine skipped: %s", exc)

        # ---- Wire ContextCompactor (P1: auto-compact large dialogues) ----
        self._compactor = None
        try:
            from backend.core.ouroboros.governance.context_compaction import ContextCompactor
            self._compactor = ContextCompactor()
            logger.info("[GLS] ContextCompactor wired (auto-compact at threshold)")
        except Exception as exc:
            logger.debug("[GLS] ContextCompactor skipped: %s", exc)

        # Phase 0 Functions-not-Agents: inject the compactor into the
        # ToolLoopCoordinator so Venom's live context auto-compaction
        # (tool_executor._compact_prompt) delegates through the hook-fired,
        # semantic-strategy-capable path. Without this injection, the
        # compactor is an inert singleton and the Phase 0 shadow telemetry
        # never exercises real production tool-loop prompts.
        if (
            self._compactor is not None
            and _tool_coordinator is not None
            and hasattr(_tool_coordinator, "set_compactor")
        ):
            try:
                _tool_coordinator.set_compactor(self._compactor)
                logger.info(
                    "[GLS] ContextCompactor attached to ToolLoopCoordinator (Phase 0 wire)",
                )
            except Exception as _attach_exc:
                logger.debug(
                    "[GLS] ToolLoopCoordinator compactor attach failed: %s",
                    _attach_exc,
                )

        # ---- Functions-not-Agents Phase 0: Gemma CompactionCaller ----
        # Wires a non-streaming Gemma semantic strategy into the
        # ContextCompactor above when JARVIS_COMPACTION_CALLER_ENABLED is
        # truthy. MUST run strictly after ContextCompactor is instantiated —
        # DAG boot-order dependency. Defaults OFF; shadow-mode first, live-mode
        # only after offline analysis of compaction_shadow.jsonl.
        try:
            if (
                self._compactor is not None
                and tier0 is not None
                and os.environ.get("JARVIS_COMPACTION_CALLER_ENABLED", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                from backend.core.ouroboros.governance.compaction_caller import (
                    CompactionCallerStrategy,
                )
                _session_dir_env = os.environ.get("JARVIS_OUROBOROS_SESSION_DIR", "").strip()
                _session_dir_path = Path(_session_dir_env) if _session_dir_env else None
                _strategy = CompactionCallerStrategy(
                    provider=tier0,
                    session_dir=_session_dir_path,
                )
                setattr(self._compactor, "_semantic_strategy", _strategy)
                logger.info(
                    "[GovernedLoop] CompactionCaller successfully attached to ContextCompactor (mode=%s, model=%s)",
                    _strategy.mode, _strategy._model or "<unresolved>",
                )
        except Exception as _compaction_boot_exc:
            logger.debug(
                "[GovernedLoop] CompactionCaller boot failed (non-fatal): %s",
                _compaction_boot_exc,
            )

        # ---- Wire AgentMemoryStore (P1: persistent per-agent memory) ----
        self._agent_memory_factory = None
        try:
            from backend.core.ouroboros.governance.agent_memory import get_agent_memory, MemoryScope
            self._agent_memory_factory = get_agent_memory
            logger.info("[GLS] AgentMemoryStore factory wired (USER/PROJECT/LOCAL scopes)")
        except Exception as exc:
            logger.debug("[GLS] AgentMemoryStore skipped: %s", exc)

        # ---- Wire PlanModeExecutor (P1: read-only dry-run) ----
        self._plan_executor = None
        try:
            from backend.core.ouroboros.governance.plan_mode import PlanModeExecutor
            self._plan_executor = PlanModeExecutor()
            logger.info("[GLS] PlanModeExecutor wired (read-only plan mode)")
        except Exception as exc:
            logger.debug("[GLS] PlanModeExecutor skipped: %s", exc)

        # ---- Wire ScopedToolGate (P1: per-agent tool restrictions) ----
        self._scoped_tool_gate = None
        try:
            from backend.core.ouroboros.governance.scoped_tool_access import get_scope_for_role
            self._scoped_tool_gate = get_scope_for_role
            logger.info("[GLS] ScopedToolGate wired (role-based tool restrictions)")
        except Exception as exc:
            logger.debug("[GLS] ScopedToolGate skipped: %s", exc)

        # ---- Wire DeferredToolRegistry (P2: lazy tool loading) ----
        self._tool_registry = None
        try:
            from backend.core.ouroboros.governance.deferred_tool_registry import get_tool_registry
            self._tool_registry = get_tool_registry()
            logger.info(
                "[GLS] DeferredToolRegistry wired (%d tools, lazy loading)",
                len(self._tool_registry.list_available()),
            )
        except Exception as exc:
            logger.debug("[GLS] DeferredToolRegistry skipped: %s", exc)

        # ---- Wire CheckpointManager (P2: interactive rewind) ----
        self._checkpoint_mgr = None
        try:
            from backend.core.ouroboros.governance.checkpoint_rewind import CheckpointManager
            self._checkpoint_mgr = CheckpointManager(project_root=self._config.project_root)
            logger.info("[GLS] CheckpointManager wired (git-based rewind)")
        except Exception as exc:
            logger.debug("[GLS] CheckpointManager skipped: %s", exc)

        # ---- Wire ScheduledAgentRunner (P2: cron-based recurring agents) ----
        self._scheduler = None
        try:
            from backend.core.ouroboros.governance.scheduled_agents import ScheduledAgentRunner
            self._scheduler = ScheduledAgentRunner(gls=self)
            asyncio.get_event_loop().create_task(self._scheduler.start())
            logger.info("[GLS] ScheduledAgentRunner started (cron-based agent scheduling)")
        except Exception as exc:
            logger.debug("[GLS] ScheduledAgentRunner skipped: %s", exc)

        # ---- Wire MultiFileRefactorEngine (P2: atomic cross-file changes) ----
        self._refactor_engine = None
        try:
            from backend.core.ouroboros.governance.multi_file_refactor import MultiFileRefactorEngine
            self._refactor_engine = MultiFileRefactorEngine(
                project_root=self._config.project_root,
            )
            logger.info("[GLS] MultiFileRefactorEngine wired (atomic cross-file refactoring)")
        except Exception as exc:
            logger.debug("[GLS] MultiFileRefactorEngine skipped: %s", exc)

        # ---- Wire BrowserBridge (P2: visual verification in pipeline) ----
        self._browser_bridge = None
        try:
            from backend.core.ouroboros.governance.browser_bridge import get_browser_bridge
            _bridge = get_browser_bridge()
            if _bridge.is_available:
                self._browser_bridge = _bridge
                logger.info(
                    "[GLS] BrowserBridge wired (backend=%s)", _bridge.backend_name
                )
            else:
                logger.debug("[GLS] BrowserBridge: no backend available")
        except Exception as exc:
            logger.debug("[GLS] BrowserBridge skipped: %s", exc)

        # ---- Wire PromptCache (token cost reduction via prefix caching) ----
        self._prompt_cache = None
        try:
            from backend.core.ouroboros.governance.prompt_cache import get_prompt_cache
            self._prompt_cache = get_prompt_cache()
            logger.info("[GLS] PromptCache wired (system prompt prefix caching)")
        except Exception as exc:
            logger.debug("[GLS] PromptCache skipped: %s", exc)

        # ---- Wire SessionManager (multi-turn operation resume) ----
        self._session_mgr = None
        try:
            from backend.core.ouroboros.governance.session_manager import get_session_manager
            self._session_mgr = get_session_manager()
            logger.info("[GLS] SessionManager wired (multi-turn session resume/fork)")
        except Exception as exc:
            logger.debug("[GLS] SessionManager skipped: %s", exc)

        # ---- Wire PermissionClassifier (ML-based auto-approve) ----
        self._permission_clf = None
        try:
            from backend.core.ouroboros.governance.permission_classifier import get_permission_classifier
            self._permission_clf = get_permission_classifier()
            logger.info(
                "[GLS] PermissionClassifier wired (%d rules, weighted voting)",
                len(self._permission_clf._rules),
            )
        except Exception as exc:
            logger.debug("[GLS] PermissionClassifier skipped: %s", exc)

        # ---- JARVIS Tier 3: Predictive Regression Engine (background task) ----
        self._predictive_engine = None
        try:
            from backend.core.ouroboros.governance.predictive_engine import PredictiveRegressionEngine
            self._predictive_engine = PredictiveRegressionEngine(self._config.project_root)
            asyncio.get_event_loop().create_task(self._predictive_engine.start())
            logger.info("[GLS] PredictiveRegressionEngine started (JARVIS Tier 3)")
        except Exception as exc:
            logger.debug("[GLS] PredictiveRegressionEngine skipped: %s", exc)

        # ---- JARVIS Tier 4: Distributed Resilience (heartbeat + sync) ----
        self._resilience_manager = None
        try:
            from backend.core.ouroboros.governance.distributed_resilience import DistributedResilienceManager
            self._resilience_manager = DistributedResilienceManager()
            asyncio.get_event_loop().create_task(self._resilience_manager.start())
            logger.info("[GLS] DistributedResilienceManager started (JARVIS Tier 4)")
        except Exception as exc:
            logger.debug("[GLS] DistributedResilienceManager skipped: %s", exc)

        # ---- JARVIS Tier 2: Emergency Protocol Engine ----
        self._emergency_engine = None
        try:
            from backend.core.ouroboros.governance.emergency_protocols import EmergencyProtocolEngine
            _say = getattr(self, "_say_fn", None)
            self._emergency_engine = EmergencyProtocolEngine(say_fn=_say)
            logger.info("[GLS] EmergencyProtocolEngine wired (JARVIS Tier 2)")
        except Exception as exc:
            logger.debug("[GLS] EmergencyProtocolEngine skipped: %s", exc)

        # ---- JARVIS Tier 6: Personality Engine ----
        self._personality_engine = None
        try:
            from backend.core.ouroboros.governance.jarvis_intelligence import PersonalityEngine
            self._personality_engine = PersonalityEngine()
            logger.info("[GLS] PersonalityEngine wired (JARVIS Tier 6)")
        except Exception as exc:
            logger.debug("[GLS] PersonalityEngine skipped: %s", exc)

        # ---- JARVIS Tier 7: Autonomous Judgment (daily review) ----
        self._judgment_framework = None
        try:
            from backend.core.ouroboros.governance.jarvis_intelligence import AutonomousJudgmentFramework
            self._judgment_framework = AutonomousJudgmentFramework()
            logger.info("[GLS] AutonomousJudgmentFramework wired (JARVIS Tier 7)")
        except Exception as exc:
            logger.debug("[GLS] AutonomousJudgmentFramework skipped: %s", exc)

        # NOTE: IntakeLayerService is started by the supervisor (Zone 6.9) which
        # injects say_fn and repo_registry.  GLS exposes _repo_registry so Zone 6.9
        # can reuse the already-resolved registry without a second from_env() call.
        self._repo_registry = repo_registry

    def _register_canary_slices(self) -> None:
        """Register initial canary slices and pre-activate them. Idempotent.

        Slices listed in ``initial_canary_slices`` are bootstrap-trusted — they
        are explicitly configured at startup, so promotion criteria (50 ops) are
        waived.  This avoids the chicken-and-egg problem where the first operation
        cannot run because no slice has accumulated the required track record yet.
        """
        from backend.core.ouroboros.governance.canary_controller import CanaryState
        for slice_prefix in self._config.initial_canary_slices:
            try:
                self._stack.canary.register_slice(slice_prefix)
                # Pre-activate: bootstrap slices are explicitly trusted from boot
                self._stack.canary._slices[slice_prefix].state = CanaryState.ACTIVE
            except Exception as exc:
                logger.warning(
                    "[GovernedLoop] Failed to register canary slice %r: %s",
                    slice_prefix,
                    exc,
                )

    def _seed_autonomy_policies(self) -> None:
        """Seed baseline SignalAutonomyConfig per repo x trigger_source x canary_slice.

        Default tiers:
          tests/            -> GOVERNED  (test-only changes run without human approval)
          docs/             -> GOVERNED  (doc patches run without human approval)
          backend/core/     -> OBSERVE   (infrastructure changes require voice confirmation)
          "" (root default) -> OBSERVE   (unclassified root-level changes default to safe)

        Tiers are seeded conservatively; TrustGraduator.promote() advances them
        automatically as operational track record accumulates.
        """
        from backend.core.ouroboros.governance.autonomy.graduator import TrustGraduator
        from backend.core.ouroboros.governance.autonomy.tiers import (
            AutonomyTier,
            GraduationMetrics,
            SignalAutonomyConfig,
            WorkContext,
            CognitiveLoad,
        )

        _TRIGGER_SOURCES = (
            "voice_command",
            "backlog",
            "test_failure",
            "opportunity_miner",
        )
        # canary_slice -> (tier, defer_during_work_context)
        _SLICE_POLICIES = {
            "tests/":        (AutonomyTier.GOVERNED, (WorkContext.MEETINGS,)),
            "docs/":         (AutonomyTier.GOVERNED, (WorkContext.MEETINGS,)),
            "backend/core/": (AutonomyTier.OBSERVE,  (WorkContext.MEETINGS, WorkContext.CODING)),
            "":              (AutonomyTier.OBSERVE,   (WorkContext.MEETINGS, WorkContext.CODING)),
        }

        graduator = TrustGraduator()
        repos = (
            [r.name for r in self._repo_registry.list_enabled()]
            if self._repo_registry is not None
            else ["jarvis"]
        )

        for repo in repos:
            for trigger_source in _TRIGGER_SOURCES:
                for canary_slice, (tier, defer_ctxs) in _SLICE_POLICIES.items():
                    config = SignalAutonomyConfig(
                        trigger_source=trigger_source,
                        repo=repo,
                        canary_slice=canary_slice,
                        current_tier=tier,
                        graduation_metrics=GraduationMetrics(),
                        defer_during_cognitive_load=CognitiveLoad.HIGH,
                        defer_during_work_context=tuple(defer_ctxs),
                        require_user_active=False,
                    )
                    graduator.register(config)

        self._trust_graduator = graduator
        logger.info(
            "[GovernedLoop] Autonomy policies seeded: %d configs across %d repos",
            len(graduator.all_configs()),
            len(repos),
        )

    def _attach_to_stack(self) -> None:
        """Attach governed loop components to GovernanceStack.

        Phase 1 Step 3C: routes the orchestrator assignment through
        :meth:`GovernanceStack.bind_orchestrator`, which writes both
        the legacy dataclass slot and the process-lifetime bind under
        ``_governance_state._bind_lock``. This makes the rebind atomic
        with respect to hot-path readers that use the new
        ``stack.orchestrator_ref`` property.
        """
        if self._stack is None:
            return
        if hasattr(self._stack, "bind_orchestrator"):
            self._stack.bind_orchestrator(self._orchestrator)
        else:
            # Legacy fallback for stacks that predate the bind contract.
            self._stack.orchestrator = self._orchestrator
        self._stack.generator = self._generator
        self._stack.approval_provider = self._approval_provider
        # Self-register on the stack so the production read path
        # (orchestrator._run_pipeline → getattr(stack,
        # "governed_loop_service")._strategic_direction →
        # format_for_prompt) resolves in ALL boot paths — not only
        # hud_governance_boot (the sole prior writer). Without this
        # the battle-test harness set gls._strategic_direction but
        # stack.governed_loop_service stayed None, so the ENTIRE
        # StrategicDirection injection (manifesto + dev-memory) was
        # dark (soak bt-2026-05-18-092457: injected=0, failed=0 —
        # block skipped, not raised). The GLS owns its own stack
        # registration: single source of truth, no second
        # StrategicDirection holder on the orchestrator.
        self._stack.governed_loop_service = self

    def _cancel_audit_watchdogs(self) -> None:
        """Stop the background audit sweep. NEVER raises. §28.

        Called on BOTH teardown paths. The sweep is fire-and-forget by
        design, which makes cancelling it a teardown OBLIGATION rather than
        an optimisation: an audit that outlives its service keeps parsing
        the tree and logs a regression about a repo nobody is watching.

        Cancellation only, never awaited — teardown must not block for the
        length of a scan it has just decided it no longer needs.
        """
        task = getattr(self, "_audit_watchdog_task", None)
        if task is None:
            return
        self._audit_watchdog_task = None
        try:
            if not task.done():
                task.cancel()
        except Exception:  # noqa: BLE001
            logger.debug("[GLS] audit watchdog cancel skipped", exc_info=True)

    def _release_why_live_source(self) -> None:
        """Take back the `/why` live reader this instance lent out. §27.5.

        Called on BOTH teardown paths — clean `stop()` and the partial
        teardown a failed boot runs — because a boot that dies after the
        binding and before the loop is exactly the case where a stale reader
        would answer `/why` with in-flight state for a service that never
        ran.

        Released by IDENTITY, so an instance that is stopping while its
        replacement has already bound leaves the successor's binding intact.
        Never raises: teardown must not fail on a diagnostic surface.
        """
        reader = getattr(self, "_why_live_reader", None)
        if reader is None:
            return
        self._why_live_reader = None
        try:
            from backend.core.ouroboros.governance.why_engine import (  # noqa: PLC0415
                release_live_source as _why_release,
            )
            if not _why_release(reader):
                logger.debug(
                    "[GLS] /why live source already rebound by a successor "
                    "— leaving it in place")
        except Exception:  # noqa: BLE001
            logger.debug("[GLS] /why live-source release skipped",
                         exc_info=True)

    def _detach_from_stack(self) -> None:
        """Detach governed loop components from GovernanceStack."""
        if self._stack is None:
            return
        if hasattr(self._stack, "bind_orchestrator"):
            self._stack.bind_orchestrator(None)
        else:
            self._stack.orchestrator = None
        self._stack.generator = None
        self._stack.approval_provider = None
        self._stack.governed_loop_service = None

    async def _reconcile_on_boot(self) -> None:
        """Scan ledger for orphaned APPLIED ops and reconcile.

        For each op with latest_state == APPLIED:
          - Check recovery_attempted marker (skip if present — idempotent)
          - Check file hash against expected post_apply_hash in ledger data
          - If hash matches: attempt rollback via RollbackArtifact
          - If hash drifted: emit manual_intervention_required, no rollback

        Also expires stale PENDING approvals and cancels stale PLANNED ops.
        """
        # ── Slice 19 — Soak circuit-breaker boot reconciliation ──
        # Reconstruct cumulative spend (durable Aegis spend-WAL) + live GCE
        # node runtime (GCP instances.list via the registered manager) BEFORE
        # the loop resumes, so a soak that already burned most of its budget
        # resumes at the right utilization (and trips immediately if already
        # over). Fail-soft + inert when the breaker is unarmed.
        try:
            from backend.core.ouroboros.governance.soak_circuit_breaker import (
                get_soak_breaker,
                soak_breaker_enabled,
            )
            if soak_breaker_enabled():
                _soak_summary = await get_soak_breaker().reconcile_on_boot()
                logger.info(
                    "[GovernedLoop] soak breaker boot reconcile: %s",
                    _soak_summary,
                )
        except Exception:  # noqa: BLE001 — boot reconcile is best-effort
            logger.debug(
                "[GovernedLoop] soak breaker boot reconcile skipped",
                exc_info=True,
            )

        if self._stack is None:
            return

        ledger = self._stack.ledger
        storage_dir = ledger._storage_dir

        TERMINAL = {
            OperationState.ROLLED_BACK, OperationState.FAILED,
            OperationState.BLOCKED,
        }

        # Scan all JSONL files in ledger storage
        for jsonl_file in storage_dir.glob("*.jsonl"):
            op_id = jsonl_file.stem  # sanitized op_id
            try:
                history = await ledger.get_history(op_id)
            except Exception:
                continue

            if not history:
                continue

            latest = history[-1]

            # ── Stale PLANNED cancellation ──────────────────────────────────
            if latest.state == OperationState.PLANNED:
                import time as _time
                stored_ts = latest.wall_time
                now_ts = _time.time()
                grace_s = getattr(self._config, "cold_start_grace_s", 300.0)
                skew_tol = 60.0
                age = now_ts - stored_ts
                if 0 < age < 604800 and age > grace_s + skew_tol:
                    await ledger.append(LedgerEntry(
                        op_id=op_id, state=OperationState.FAILED,
                        data={"reason": "stale_planned_on_boot", "age_s": age},
                    ))
                continue

            # ── Orphaned APPLIED reconciliation ─────────────────────────────
            if latest.state != OperationState.APPLIED:
                continue

            # Idempotency: skip if already attempted recovery
            if latest.data.get("recovery_attempted"):
                continue

            # Write recovery marker BEFORE doing any work
            import uuid as _uuid
            recovery_id = _uuid.uuid4().hex
            await ledger.append(LedgerEntry(
                op_id=op_id, state=OperationState.APPLIED,
                data={
                    **latest.data,
                    "recovery_attempted": True,
                    "recovery_attempt_id": recovery_id,
                },
            ))

            # Hash-guarded rollback
            target_path_str = latest.data.get("target_file")
            rollback_hash = latest.data.get("rollback_hash")  # pre-apply hash (set by ChangeEngine)

            if not target_path_str or not rollback_hash:
                # Insufficient provenance — cannot assess rollback, escalate.
                # Slice 123 Phase 1: ALSO sequester the unvouched payload to
                # .jarvis/quarantine/ so it stops re-clogging the boot intake on
                # every restart (escalation below is unchanged — auditability
                # preserved). Gated + best-effort; never breaks recovery.
                try:
                    from backend.core.ouroboros.governance.boot_recovery_quarantine import quarantine_op as _quarantine_op

                    _quarantine_op(
                        op_id,
                        {**latest.data, "recovery_attempt_id": recovery_id},
                        "boot_recovery_missing_provenance",
                    )
                except Exception:  # noqa: BLE001 - quarantine is best-effort
                    pass
                await ledger.append(LedgerEntry(
                    op_id=op_id, state=OperationState.FAILED,
                    data={"reason": "boot_recovery_missing_provenance",
                          "recovery_attempt_id": recovery_id,
                          "quarantined": True},
                ))
                await self._stack.comm.emit_decision(
                    op_id=op_id, outcome="manual_intervention_required",
                    reason_code="boot_recovery_missing_provenance",
                )
                await self.report_external_outcome(
                    op_id=op_id,
                    terminal_phase=OperationPhase.POSTMORTEM,
                    reason_code="boot_recovery_missing_provenance",
                    affected_files=((target_path_str,) if target_path_str else ()),
                    failure_class="env",
                    outcome_source="boot_recovery",
                    extra_payload={
                        "recovery_attempt_id": recovery_id,
                        "recovery_disposition": "manual_intervention_required",
                    },
                )
                continue

            import hashlib as _hashlib
            target = Path(target_path_str)
            if not target.exists():
                await ledger.append(LedgerEntry(
                    op_id=op_id, state=OperationState.FAILED,
                    data={"reason": "boot_recovery_file_missing",
                          "recovery_attempt_id": recovery_id},
                ))
                await self._stack.comm.emit_decision(
                    op_id=op_id, outcome="manual_intervention_required",
                    reason_code="boot_recovery_file_missing",
                )
                await self.report_external_outcome(
                    op_id=op_id,
                    terminal_phase=OperationPhase.POSTMORTEM,
                    reason_code="boot_recovery_file_missing",
                    affected_files=(target_path_str,),
                    failure_class="env",
                    outcome_source="boot_recovery",
                    extra_payload={
                        "recovery_attempt_id": recovery_id,
                        "recovery_disposition": "manual_intervention_required",
                    },
                )
                continue

            current_hash = _hashlib.sha256(target.read_bytes()).hexdigest()
            if current_hash == rollback_hash:
                # File already matches pre-apply content — change was undone externally
                await ledger.append(LedgerEntry(
                    op_id=op_id, state=OperationState.ROLLED_BACK,
                    data={"reason": "boot_recovery_already_reverted",
                          "recovery_attempt_id": recovery_id},
                ))
                await self.report_external_outcome(
                    op_id=op_id,
                    terminal_phase=OperationPhase.CANCELLED,
                    reason_code="boot_recovery_already_reverted",
                    rollback_occurred=True,
                    affected_files=(target_path_str,),
                    failure_class="rollback",
                    outcome_source="boot_recovery",
                    extra_payload={
                        "recovery_attempt_id": recovery_id,
                        "recovery_disposition": "already_reverted",
                    },
                )
                logger.info("[GovernedLoop] Boot recovery: op=%s already reverted externally", op_id)
                continue

            # File still has post-apply content; original bytes not stored — escalate
            await ledger.append(LedgerEntry(
                op_id=op_id, state=OperationState.FAILED,
                data={"reason": "boot_recovery_needs_manual_rollback",
                      "current_hash": current_hash,
                      "rollback_hash": rollback_hash,
                      "recovery_attempt_id": recovery_id},
            ))
            await self._stack.comm.emit_decision(
                op_id=op_id, outcome="manual_intervention_required",
                reason_code="boot_recovery_needs_manual_rollback",
            )
            await self.report_external_outcome(
                op_id=op_id,
                terminal_phase=OperationPhase.POSTMORTEM,
                reason_code="boot_recovery_needs_manual_rollback",
                affected_files=(target_path_str,),
                failure_class="env",
                outcome_source="boot_recovery",
                extra_payload={
                    "recovery_attempt_id": recovery_id,
                    "current_hash": current_hash,
                    "rollback_hash": rollback_hash,
                    "recovery_disposition": "manual_intervention_required",
                },
            )

        # Expire stale approvals — batch notify (no per-op comm storm)
        approval_store = getattr(self._stack, "approval_store", None)
        if approval_store is not None:
            ttl = getattr(self._config, "approval_ttl_s", 1800.0)
            expired = approval_store.expire_stale(timeout_seconds=ttl)
            if expired:
                await self._stack.comm.emit_decision(
                    op_id="boot_reconciliation",
                    outcome="approvals_expired_on_boot",
                    reason_code=f"expired_count={len(expired)}",
                    diff_summary=", ".join(expired[:10]),
                )
                logger.info("[GovernedLoop] Boot: expired %d stale approvals", len(expired))

    async def _teardown_partial(self) -> None:
        """Clean up partially constructed components on startup failure."""
        self._orchestrator = None
        self._generator = None
        self._approval_provider = None
        self._cancel_audit_watchdogs()
        self._release_why_live_source()
        self._detach_from_stack()

    # ------------------------------------------------------------------
    # Private: Background loops
    # ------------------------------------------------------------------

    async def _health_probe_loop(self) -> None:
        """Periodically probe provider health and update FSM state.

        Probe interval adapts based on recovery ETA: aggressive near
        predicted recovery, relaxed during deep backoff (Manifesto §5).
        """
        while True:
            try:
                # Adaptive interval: use FSM's recommendation, capped at 2x base
                base_interval = self._config.health_probe_interval_s
                if self._generator is not None:
                    adaptive = self._generator.fsm.recommended_probe_interval()
                    interval = max(5.0, min(adaptive, base_interval * 2))
                else:
                    interval = base_interval
                await asyncio.sleep(interval)
                if self._generator is not None:
                    provider = getattr(self._generator, "_primary", None)
                    if provider is not None:
                        ok = False  # default to failure
                        try:
                            ok = await asyncio.wait_for(
                                provider.health_probe(), timeout=5.0
                            )
                            if ok:
                                try:
                                    self._generator.fsm.record_probe_success()
                                except Exception:
                                    pass
                            else:
                                try:
                                    self._generator.fsm.record_probe_failure()
                                except Exception:
                                    pass
                        except Exception:
                            try:
                                self._generator.fsm.record_probe_failure()
                            except Exception:
                                pass
                        # C+ L1: Emit health probe result to SafetyNet (L3)
                        if self._event_emitter is not None:
                            try:
                                _fsm = self._generator.fsm
                                _fm = _fsm._failure_mode
                                _eta = max(0, _fsm.recovery_eta() - time.monotonic()) if not ok else 0
                                await self._event_emitter.emit(AutonomyEventEnvelope(
                                    source_layer="L1",
                                    event_type=AutonomyEventType.HEALTH_PROBE_RESULT,
                                    payload={
                                        "provider": getattr(provider, "provider_name", "unknown"),
                                        "success": ok,
                                        "latency_ms": 0,
                                        "consecutive_failures": _fsm._consecutive_failures,
                                        "failure_mode": _fm.name if _fm else None,
                                        "recovery_eta_s": round(_eta, 1),
                                    },
                                ))
                            except Exception:
                                pass  # fault-isolated
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] health_probe_loop error: %s", exc)

    async def _transport_breaker_probe_loop(self) -> None:
        """C2 -- Periodic HALF-OPEN probe driver for the TransportCircuitBreaker.

        Fired once per ``JARVIS_TRANSPORT_BREAKER_PROBE_INTERVAL_S`` (default 30s).
        For each lane (batch + realtime) calls ``run_probe_if_due`` so an OPEN
        lane's recovery deadline check fires and the lane self-heals via the
        cheap DW ping probe.

        Completely fail-soft: any import / probe error is caught and logged at
        DEBUG. CancelledError exits cleanly. The daemon is only started when
        JARVIS_TRANSPORT_BREAKER_ENABLED=true; default OFF = byte-identical.

        Probe function: attempts a lightweight DW catalog/models endpoint GET
        through the appropriate lane.  Reuses the DW provider's existing
        ``_dw_batch_lane_healthy`` check for batch, and the realtime streaming
        pre-flight for realtime. Falls back to a no-op True probe when neither
        is reachable (safe -- the breaker just transitions OPEN->CLOSED without
        real network evidence; cost ~0).
        """
        import time as _t

        _PROBE_INTERVAL_DEFAULT = 30.0

        async def _probe_fn(lane: str) -> bool:
            """Minimal bounded DW health probe for the given lane. Never raises."""
            try:
                if lane == "batch":
                    from backend.core.ouroboros.governance.doubleword_provider import (
                        _dw_batch_lane_healthy as _batch_ok,
                    )
                    return bool(_batch_ok())
                # realtime: reuse the existing preflight gate (side-effect-free).
                from backend.core.ouroboros.governance.doubleword_provider import (
                    _dw_streaming_warm_degraded as _rt_degraded,
                )
                return not bool(_rt_degraded())
            except Exception:  # noqa: BLE001 -- probe failure = ok=False
                return False

        while True:
            try:
                _interval = 30.0
                try:
                    _interval = float(
                        __import__("os").environ.get(
                            "JARVIS_TRANSPORT_BREAKER_PROBE_INTERVAL_S",
                            _PROBE_INTERVAL_DEFAULT,
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(max(5.0, _interval))
                try:
                    from backend.core.ouroboros.governance.transport_circuit_breaker import (
                        breaker_enabled as _tcb_on,
                        get_transport_breaker as _get_tb,
                        run_probe_if_due as _run_probe,
                    )
                    if not _tcb_on():
                        continue
                    _tb = _get_tb()
                    _now = _t.monotonic()
                    for _lane in ("batch", "realtime"):
                        await _run_probe(_tb, _lane, _probe_fn, now=_now)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[GovernedLoop] C2 transport breaker probe tick failed",
                        exc_info=True,
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "[GovernedLoop] transport_breaker_probe_loop error: %s", exc,
                )

    def _start_failover_loop(self) -> None:
        """Launch the FailoverLifecycleController tick loop as a peer background
        task (Omni-Soak #3 fix).

        Gated on ``JARVIS_FAILOVER_LIFECYCLE_ENABLED`` (default ON — graduated
        2026-06-23 after the Adversarial Cognitive Soak; hot-revert via
        ``export JARVIS_FAILOVER_LIFECYCLE_ENABLED=false``) --
        OFF -> NO task is created, byte-identical to before.  T4's any-route fix
        (``JARVIS_FAILOVER_ANY_ROUTE_OUTAGE_ENABLED`` default true) means a genuine
        route outage now triggers a real GCE awaken in baseline production; to run
        without GCE awaken risk also set ``JARVIS_FAILOVER_ANY_ROUTE_OUTAGE_ENABLED=false``.
        When ON, the
        process-wide controller singleton is resolved (it lazily wires its own
        real awaken / ready / delete boundaries + the ProviderHealthGradient +
        heartbeat + recovery_forecaster) and an async loop ticks it so the FSM
        can actually transition (DORMANT->AWAKENING on a real DW outage,
        SERVING->handback on recovery). Fail-soft: a resolution / scheduling
        error is logged and swallowed -- a failover bug NEVER blocks boot or
        crashes the main loop (the op still drops into the Cryo-DLQ as today).
        Idempotent -- a second call while a live task exists is a no-op.
        """
        try:
            from backend.core.ouroboros.governance.failover_lifecycle import (
                lifecycle_enabled as _failover_enabled,
                get_failover_controller as _get_failover_controller,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[GovernedLoop] failover wiring import skipped: %r", exc,
            )
            return
        if not _failover_enabled():
            # OFF byte-identical: no task started, controller stays DORMANT.
            return
        if self._failover_task is not None and not self._failover_task.done():
            return  # idempotent -- already running
        try:
            self._failover_controller = _get_failover_controller()
            self._failover_task = asyncio.create_task(
                self._failover_loop(), name="failover_lifecycle_loop",
            )
            logger.info(
                "[GovernedLoop] FailoverLifecycleController tick loop started "
                "(JARVIS_FAILOVER_LIFECYCLE_ENABLED=true) -- DW->J-Prime->DW "
                "failover is now LIVE",
            )
        except Exception as exc:  # noqa: BLE001 -- never block boot
            logger.warning(
                "[GovernedLoop] failover loop start failed (non-fatal): %r", exc,
            )
            self._failover_task = None
        # Layer 1 -- start the DWHeartbeat deep-probe loop as a PEER task so
        # is_degrading() is actually fed (the run-#13 wiring gap: the probe was
        # built + flag ON, but nothing started the loop -> Layer 1 was inert).
        try:
            from backend.core.ouroboros.governance.provider_heartbeat import (
                heartbeat_enabled as _hb_enabled,
                deep_probe_enabled as _deep_enabled,
                get_dw_heartbeat as _get_hb,
            )
            if _hb_enabled() and (
                self._heartbeat_task is None or self._heartbeat_task.done()
            ):
                self._dw_heartbeat = _get_hb()
                self._heartbeat_task = asyncio.create_task(
                    self._dw_heartbeat.run(), name="dw_heartbeat_loop",
                )
                logger.info(
                    "[GovernedLoop] DWHeartbeat probe loop started "
                    "(deep_probe=%s) -- Layer 1 (early-prewarm signal) is LIVE",
                    _deep_enabled(),
                )
        except Exception as exc:  # noqa: BLE001 -- never block boot
            logger.warning(
                "[GovernedLoop] heartbeat loop start failed (non-fatal): %r", exc,
            )
            self._heartbeat_task = None

    async def _failover_loop(self) -> None:
        """Tick the FailoverLifecycleController on a fixed interval until
        shutdown. Reuses the controller's own ``tick()`` (NO new FSM / awaken /
        delete logic) -- this is pure wiring of the existing FSM into the live
        boot so it actually RUNS.

        Cadence: ``JARVIS_FAILOVER_TICK_INTERVAL_S`` (default 25.0s; floor 1.0s).
        Fully fail-soft -- a tick exception is logged at DEBUG and the loop
        continues, so a failover bug never crashes the main loop. CancelledError
        exits cleanly (uses ``asyncio.sleep``, Python 3.9+ safe).
        """
        try:
            _interval_default = 25.0
            while True:
                try:
                    interval = _interval_default
                    try:
                        # Floor 0.01s: guards against a misconfigured busy-loop
                        # while still allowing fast ticks under test. Operators
                        # tune the default upward (~25s); production never goes
                        # sub-second.
                        interval = max(
                            0.01,
                            float(
                                os.environ.get(
                                    "JARVIS_FAILOVER_TICK_INTERVAL_S",
                                    _interval_default,
                                )
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        interval = _interval_default
                    ctrl = self._failover_controller
                    if ctrl is not None:
                        try:
                            await ctrl.tick()
                        except Exception as exc:  # noqa: BLE001 -- fail-soft
                            logger.debug(
                                "[GovernedLoop] failover tick fail-soft err=%r",
                                exc,
                            )
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- loop never dies
                    logger.debug(
                        "[GovernedLoop] failover loop iteration error: %r", exc,
                    )
                    try:
                        await asyncio.sleep(_interval_default)
                    except asyncio.CancelledError:
                        raise
        except asyncio.CancelledError:
            return

    async def _stop_failover_loop(self) -> None:
        """Cancel the failover tick loop cleanly on shutdown. Fail-soft;
        idempotent; never raises into the shutdown path."""
        task = getattr(self, "_failover_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 -- never block shutdown
                logger.debug(
                    "[GovernedLoop] failover loop cancel swallowed",
                    exc_info=True,
                )
        # Best-effort: tell the controller to stop its own internal run() loop
        # if it happens to be driving one (harmless when it is not).
        ctrl = getattr(self, "_failover_controller", None)
        if ctrl is not None:
            try:
                stop_fn = getattr(ctrl, "stop", None)
                if callable(stop_fn):
                    stop_fn()
            except Exception:  # noqa: BLE001
                pass
        self._failover_task = None
        # Layer 1 -- cancel the DWHeartbeat probe loop as a peer of the failover
        # task (stop() flips its internal flag; cancel() unblocks the sleep).
        hb = getattr(self, "_dw_heartbeat", None)
        if hb is not None:
            try:
                stop_fn = getattr(hb, "stop", None)
                if callable(stop_fn):
                    stop_fn()
            except Exception:  # noqa: BLE001
                pass
        hb_task = getattr(self, "_heartbeat_task", None)
        if hb_task is not None and not hb_task.done():
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 -- never block shutdown
                logger.debug(
                    "[GovernedLoop] heartbeat loop cancel swallowed", exc_info=True,
                )
        self._heartbeat_task = None

    async def _prewarm_oracle_barrier(self) -> int:
        """ABSOLUTE PRE-FLIGHT INITIALIZATION BARRIER (Omni-Soak v5/v6 fix).

        Ingest the SHA256-validated ``oracle_prewarm.json`` and warm the
        shared Oracle handle's ``_file_index`` for the known chaos targets as a
        **blocking** boot step BEFORE the Meta-Goal drain/flush loops are
        scheduled. Pure delegation to the wiring module's
        :func:`ingest_prewarm_barrier` (which REUSES the Oracle's existing
        ``ingest_prewarm_payload`` -- no new ingester). Gated on
        ``JARVIS_ORACLE_SELF_WARMING_ENABLED`` (OFF -> no-op, byte-identical).
        Fail-soft: a missing payload / mismatch / error logs and boot proceeds
        (the runtime JIT remains the cold-miss fallback). Returns warmed count.
        """
        try:
            from backend.core.ouroboros.governance.meta_goal_wiring import (
                ingest_prewarm_barrier as _barrier,
            )
            return await _barrier(self)
        except Exception as exc:  # noqa: BLE001 -- never block boot
            logger.debug(
                "[GovernedLoop] oracle pre-warm barrier skipped: %r", exc,
            )
            return 0

    def _start_meta_goal_drain_loop(self) -> None:
        """Start the Meta-Goal aggregator drain loop as a peer background task
        (the built-but-no-caller fix). Pure delegation to the wiring module —
        gated on ``JARVIS_META_GOAL_AGGREGATOR_ENABLED`` (default OFF ->
        byte-identical: no task, no aggregator). Fail-soft; never blocks boot.
        """
        try:
            from backend.core.ouroboros.governance.meta_goal_wiring import (
                start_meta_goal_drain_loop as _start,
            )
            _start(self)
        except Exception as exc:  # noqa: BLE001 -- never block boot
            logger.debug(
                "[GovernedLoop] meta-goal drain loop start skipped: %r", exc,
            )

    async def _stop_meta_goal_drain_loop(self) -> None:
        """Cancel the Meta-Goal drain loop cleanly on shutdown. Fail-soft;
        idempotent; never raises into the shutdown path."""
        try:
            from backend.core.ouroboros.governance.meta_goal_wiring import (
                stop_meta_goal_drain_loop as _stop,
            )
            await _stop(self)
        except Exception:  # noqa: BLE001 -- never block shutdown
            logger.debug(
                "[GovernedLoop] meta-goal drain loop stop swallowed",
                exc_info=True,
            )
            self._meta_goal_drain_task = None

    async def _curriculum_loop(self) -> None:
        """Publish curriculum signal every interval. Never crashes the service."""
        while True:
            try:
                await asyncio.sleep(self._config.curriculum_publish_interval_s)
                if self._curriculum_publisher:
                    await asyncio.wait_for(
                        self._curriculum_publisher.publish(),
                        timeout=30.0,
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] curriculum_loop error: %s", exc)

    async def _reactor_event_loop(self) -> None:
        """Poll event_dir for Reactor events. Never crashes the service."""
        seen: set[str] = set()
        while True:
            try:
                await asyncio.sleep(self._config.reactor_event_poll_interval_s)
                await self._handle_event_files(seen)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] reactor_event_loop error: %s", exc)

    async def _oracle_index_loop(self) -> None:
        """Index all repos into TheOracle graph on boot, then poll for incremental changes.

        Non-blocking: start() never awaits this. Fault-isolated: any exception in
        initialization sets self._oracle = None, logs a structured warning, and exits
        the task without impacting service state or any operation's terminal phase.

        If self._oracle is already set (injected by an external harness or the
        governance stack), skip initialization entirely — reuse the existing instance.
        This prevents double-initialization of ChromaDB's PersistentClient which
        causes a SQLite lock contention segfault (SIGSEGV at 0x0) when two clients
        target the same persistence directory concurrently.
        """
        try:
            # Reuse injected Oracle if already available (e.g. from battle test harness)
            if self._oracle is not None:
                logger.info(
                    "[GovernedLoop] Oracle already injected (%s nodes), skipping re-init",
                    (await _maybe_await(self._oracle.get_metrics())).get("total_nodes", "?"),
                )
            elif TheOracle is None:
                raise ImportError("TheOracle not available")
            else:
                oracle = TheOracle()
                await oracle.initialize()
                self._oracle = oracle
            if self._stack is not None:
                self._stack.oracle = self._oracle
            # PR-A 2026-05-13 — register Oracle with the advisor module
            # so blast-radius queries can compose Oracle's pre-built
            # CodeGraph BFS instead of the 29.5k-file rglob scan.
            # Master-flag-gated (JARVIS_ADVISOR_ORACLE_BLAST_ENABLED);
            # advisor falls back to legacy when off / cold-miss.
            try:
                from backend.core.ouroboros.governance.operation_advisor import (
                    set_active_oracle,
                )
                set_active_oracle(self._oracle)
                logger.info(
                    "[GovernedLoop] Oracle registered with OperationAdvisor "
                    "(PR-A blast path now available, gated by "
                    "JARVIS_ADVISOR_ORACLE_BLAST_ENABLED)",
                )
            except Exception:  # noqa: BLE001 — defensive: registration is best-effort
                logger.debug(
                    "[GovernedLoop] Oracle registration with advisor failed",
                    exc_info=True,
                )
            logger.info(
                "[GovernedLoop] Oracle indexed %s nodes across all repos",
                (await _maybe_await(self._oracle.get_metrics())).get("total_nodes", "?"),
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "[GovernedLoop] Oracle initialization failed: %s; codebase graph unavailable",
                exc,
            )
            self._oracle = None
            return

        # Incremental update loop — polls every oracle_incremental_poll_interval_s.
        #
        # Task #88f (2026-05-14) — cooperative yield to Advisor.
        # v14-rev10 graduation soak proved: Oracle's full-tree
        # incremental_update([]) contends with Advisor's blast-radius
        # rglob+read_text scans on disk I/O.  Advisor scans for a SWE
        # op's tiny worktree took 4m 46s when Oracle's main-tree poll
        # was concurrent (vs <2s when Oracle was quiet).
        #
        # When the Advisor is actively running blast scans
        # (``get_advisor_busy_count() > 0``), this loop SKIPS the
        # heavy ``incremental_update([])`` call for this cycle only —
        # it'll re-evaluate next poll interval.  Bounded by
        # ``JARVIS_ORACLE_YIELD_MAX_CONSECUTIVE_SKIPS`` (default 10):
        # after N consecutive skips, force a poll regardless to
        # prevent indefinite starvation.  Master-flag-gated by
        # ``JARVIS_ORACLE_YIELD_TO_ADVISOR`` (default ``true``) so
        # operators can revert to legacy behavior with one env flip
        # if any starvation regression appears.
        _yield_enabled = (
            os.environ.get("JARVIS_ORACLE_YIELD_TO_ADVISOR", "true")
            .strip().lower() in ("true", "1", "yes", "on")
        )
        try:
            _max_consec_skips = int(os.environ.get(
                "JARVIS_ORACLE_YIELD_MAX_CONSECUTIVE_SKIPS", "10",
            ))
        except (TypeError, ValueError):
            _max_consec_skips = 10
        _consec_skips = 0
        while True:
            try:
                await asyncio.sleep(self._config.oracle_incremental_poll_interval_s)
                # ── Slice 86 — benchmark-isolation event-loop hygiene ──
                # The periodic poll below calls ``incremental_update([])``; an
                # EMPTY list is falsy, so it falls to the else-branch FULL scan
                # of every repo — 48-72s of CPU-bound work that FREEZES the
                # event loop. During a benchmark run that freeze starves the DW
                # stream-reading coroutines mid-generation (bt-2026-06-04-041943:
                # first_token_ms=-1 / 175-242s "stalls" were ControlPlaneStarvation
                # lag up to 10s, NOT upstream delay — DW streams content in <=35s).
                # The consciousness Oracle index is not needed for a benchmark op
                # (search_code uses ripgrep, not the semantic graph; the targeted
                # post-APPLY reindex still runs via the orchestrator with the
                # actual applied_files), so skip ONLY the periodic full scan. The
                # loop stays alive so it resumes the moment isolation clears
                # (hot-flip), matching the master-flag short-circuit pattern.
                if _oracle_full_scan_suppressed_by_benchmark():
                    continue
                # Cooperative yield decision — Task #88f.  Read the
                # advisor busy count via the official public surface
                # (NOT executor._work_queue.qsize() which is private +
                # fragile across Python versions, per operator binding).
                if _yield_enabled and _consec_skips < _max_consec_skips:
                    try:
                        from backend.core.ouroboros.governance.operation_advisor import (
                            get_advisor_busy_count,
                        )
                        _busy = get_advisor_busy_count()
                    except Exception:  # noqa: BLE001 — defensive
                        _busy = 0
                    if _busy > 0:
                        _consec_skips += 1
                        logger.info(
                            "[GovernedLoop] Oracle yielding to advisor "
                            "(busy=%d, consec_skip=%d/%d) — Task #88f",
                            _busy, _consec_skips, _max_consec_skips,
                        )
                        continue
                # Either advisor idle, master flag off, or bounded-skip
                # ceiling reached — run the poll.
                if _consec_skips >= _max_consec_skips and _yield_enabled:
                    logger.info(
                        "[GovernedLoop] Oracle bounded-skip ceiling reached "
                        "(%d/%d) — forcing incremental_update to prevent "
                        "starvation",
                        _consec_skips, _max_consec_skips,
                    )
                _consec_skips = 0
                await _maybe_await(self._oracle.incremental_update([]))
            except asyncio.CancelledError:
                await _maybe_await(self._oracle.shutdown())
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] Oracle incremental update failed: %s", exc)

    async def _handle_event_files(self, seen: Set[str]) -> None:
        """Process new JSON files in event_dir. Extracted for testability."""
        if self._event_dir is None:
            return
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            is_offload_error,
            offload,
        )
        event_dir = self._event_dir
        paths_result = await offload(
            lambda: sorted(event_dir.glob("*.json")),
        )
        if is_offload_error(paths_result):
            logger.debug(
                "[GovernedLoop] _handle_event_files: offloaded glob "
                "failed (%s) — treating as empty this tick",
                paths_result.message,
            )
            return
        for path in paths_result:
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                data = json.loads(path.read_text())
                event_type = data.get("event_type", "")
                if event_type == "model_promoted":
                    await self._handle_model_promoted(data)
                elif event_type == "ouroboros_improvement":
                    pass  # consumed elsewhere
                else:
                    logger.debug(
                        "[GovernedLoop] Unknown event_type=%r in %s",
                        event_type, path.name,
                    )
            except Exception as exc:
                logger.warning(
                    "[GovernedLoop] reactor_event_loop: failed to process %s: %s",
                    path.name, exc,
                )

    async def _handle_model_promoted(self, data: dict) -> None:
        if self._model_attribution_recorder is None:
            return
        try:
            await asyncio.wait_for(
                self._model_attribution_recorder.record_model_transition(
                    new_model_id=data["model_id"],
                    previous_model_id=data["previous_model_id"],
                    training_batch_size=int(data["training_batch_size"]),
                    task_types=data.get("task_types"),
                ),
                timeout=30.0,
            )
        except Exception as exc:
            logger.warning("[GovernedLoop] _handle_model_promoted failed: %s", exc)

    # ------------------------------------------------------------------
    # C+ Autonomy: background loops
    # ------------------------------------------------------------------

    async def _feedback_loop(self) -> None:
        """Periodically run FeedbackEngine consumption loops."""
        while True:
            try:
                await asyncio.sleep(60.0)
                if self._feedback_engine:
                    await self._feedback_engine.consume_curriculum_once()
                    await self._feedback_engine.consume_reactor_events_once()
                    if self._performance_persistence is None:
                        self._performance_persistence = get_performance_persistence()
                    await self._feedback_engine.score_attribution_once(
                        self._performance_persistence,
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] feedback_loop error: %s", exc)

    async def _command_consumer_loop(self) -> None:
        """Consume commands from advisory layers and route to L1 handlers."""
        while True:
            try:
                if self._command_bus is None:
                    await asyncio.sleep(5.0)
                    continue
                cmd = await asyncio.wait_for(self._command_bus.get(), timeout=5.0)
                await self._handle_advisory_command(cmd)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[GovernedLoop] command_consumer error: %s", exc)

    async def _handle_advisory_command(self, cmd) -> None:
        """Route a command envelope to the appropriate L1 handler."""
        ct = cmd.command_type
        if ct == AutonomyCommandType.GENERATE_BACKLOG_ENTRY:
            logger.info("[GovernedLoop] L2 backlog: %s", cmd.payload.get("description", "")[:80])
        elif ct == AutonomyCommandType.ADJUST_BRAIN_HINT:
            logger.info("[GovernedLoop] L2 brain hint: brain=%s delta=%s",
                        cmd.payload.get("brain_id"), cmd.payload.get("weight_delta"))
        elif ct == AutonomyCommandType.REQUEST_MODE_SWITCH:
            logger.warning("[GovernedLoop] L3 mode switch: %s (reason: %s)",
                           cmd.payload.get("target_mode"), cmd.payload.get("reason"))
        elif ct == AutonomyCommandType.REPORT_ROLLBACK_CAUSE:
            logger.info("[GovernedLoop] L3 rollback analysis: op=%s cause=%s pattern=%s",
                        cmd.payload.get("op_id"), cmd.payload.get("root_cause_class"),
                        cmd.payload.get("pattern_match"))
        elif ct == AutonomyCommandType.SIGNAL_HUMAN_PRESENCE:
            logger.info("[GovernedLoop] L3 human presence: active=%s type=%s",
                        cmd.payload.get("is_active"), cmd.payload.get("activity_type"))
        elif ct == AutonomyCommandType.SUBMIT_EXECUTION_GRAPH:
            if self._subagent_scheduler is None:
                logger.warning("[GovernedLoop] L3 graph submit ignored: scheduler unavailable")
                return
            graph = cmd.payload.get("execution_graph")
            if graph is None:
                logger.warning("[GovernedLoop] L3 graph submit ignored: missing execution_graph")
                return
            accepted = await self._subagent_scheduler.submit(graph)
            logger.info(
                "[GovernedLoop] L3 graph submit: graph_id=%s accepted=%s",
                getattr(graph, "graph_id", "?"),
                accepted,
            )
        elif ct == AutonomyCommandType.REPORT_WORK_UNIT_RESULT:
            logger.info(
                "[GovernedLoop] L3 work unit result: graph=%s unit=%s repo=%s status=%s",
                cmd.payload.get("graph_id"),
                cmd.payload.get("unit_id"),
                cmd.payload.get("repo"),
                cmd.payload.get("status"),
            )
        elif ct == AutonomyCommandType.ABORT_EXECUTION_GRAPH:
            if self._subagent_scheduler is None:
                logger.warning("[GovernedLoop] L3 graph abort ignored: scheduler unavailable")
                return
            graph_id = str(cmd.payload.get("graph_id", ""))
            if not graph_id:
                logger.warning("[GovernedLoop] L3 graph abort ignored: missing graph_id")
                return
            aborted = await self._subagent_scheduler.abort(graph_id)
            logger.warning("[GovernedLoop] L3 graph abort: graph_id=%s aborted=%s", graph_id, aborted)
        else:
            logger.debug("[GovernedLoop] Unhandled command: %s", ct)


def _fleet_lane_sync(pool: Any, configured_size: int, snapshot: "Dict[str, str]") -> None:
    """Topology observer -> pool lane target (Dynamic Fleet Service Discovery).

    Non-empty fleet snapshot => lanes strictly equal the serving sovereign
    node count, LOCKED (hardware-derived truth overrides env/manifest).
    Empty fleet => the configured size restores, unlocked (DW-era regime).
    The snapshot dict is structurally extensible -- future golden-image
    topology payloads (VRAM, tiers) can weight lanes here without touching
    the observer channel. NEVER raises."""
    try:
        _nodes = len(snapshot or {})
        if _nodes > 0:
            pool.set_target_pool_size(_nodes, source="fleet_topology", lock=True)
        else:
            pool.set_target_pool_size(
                max(1, int(configured_size)), source="fleet_topology", lock=False,
            )
    except Exception:  # noqa: BLE001 -- lane sync must never break the fleet path
        logger.debug("[GLS] fleet lane sync fail-soft", exc_info=True)
