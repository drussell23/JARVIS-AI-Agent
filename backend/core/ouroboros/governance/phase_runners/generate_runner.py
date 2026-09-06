"""GENERATERunner — Slice 5a of Wave 2 item (5). The beast.

Extracts orchestrator.py lines ~3138-4748 (1611 lines: GENERATE phase
prelude + retry loop + CandidateGenerator dispatch + per-op cost cap +
forward-progress detector + productivity detector + Iron Gate suite
(exploration ledger, ASCII strict, config format, dependency-file
integrity, multi-file coverage) + retry feedback composition + L2
escape terminals) into a single :class:`PhaseRunner` behind
``JARVIS_PHASE_RUNNER_GENERATE_EXTRACTED`` (default ``true`` — graduated
2026-04-22/23; the inline twin remains as the kill-switch path).

**Zero behavior change per slice.** Verbatim transcription with
``self.`` → ``orch.`` substitutions. Scripted extraction via
``build_generate_runner.py`` keeps parity exact.

## Sub-slice delivery order

* **5a** (this commit): spine extraction + parity tests covering
  prelude, retry loop skeleton, CandidateGenerator dispatch, cost
  cap, happy path, bounded retry, L2 escape terminals.
* **5b** (next commit): Iron Gate suite parity depth —
  exploration ledger category-aware diversity scoring (§6 heart),
  ASCII strict gate, dependency-file integrity (hallucinated-rename
  catcher), multi-file coverage gate, retry-feedback composition.

Same runner module. Same flag. Split is *parity test depth*, not code.

## ~8 terminal exit paths (carry over from inline)

1. ``op_cost_cap_exceeded`` — per-op cost cap tripped pre-attempt
2. ``no_forward_progress`` — forward-progress detector EC8 trip
3. ``stalled_productivity`` — productivity detector EC9 trip
4. L2 ``cancel`` / ``fatal`` — from L2 escape on terminal retry
5. ``ascii_gate_violation`` — ASCII strict gate (§6 Iron Gate)
6. ``exploration_floor_not_met`` — exploration ledger insufficient
7. ``dependency_file_integrity_failed`` — hallucinated rename blocked
8. ``config_format_invalid`` — config format gate

## Success path

``next_phase = VALIDATE`` with ``generation`` stamped on result artifacts.
The orchestrator hook reads ``artifacts["generation"]`` and
``artifacts["episodic_memory"]`` for VALIDATE.

## Cross-phase artifacts

* ``generation`` — the GenerationResult (consumed by VALIDATERunner)
* ``episodic_memory`` — EpisodicFailureMemory (consumed by VALIDATERunner)

Both threaded via ``PhaseResult.artifacts``. Orchestrator hook rebinds
``generation`` + ``_episodic_memory`` locals before VALIDATE inline /
runner reads them.

## Dependencies injected via constructor

* ``orchestrator`` — reads many helpers (see verbatim block for full list)
* ``serpent`` — pipeline serpent handle (optional)
* ``consciousness_bridge`` — from CLASSIFY artifacts (fragile-file injection)

## Authority invariant

Runner imports: ledger, op_context, phase_runner, plus function-local
imports matching inline block. NO ``iron_gate`` module import — the
Iron Gate suite is inlined here same as orchestrator.py. No new
authority-widening; grep-pinned.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from backend.core.ouroboros.governance.ascii_strict_gate import (
    AsciiStrictGate,
    build_retry_feedback as _ascii_gate_retry_feedback,
)
from backend.core.ouroboros.governance.target_existence_guard import (
    TARGET_MISSING_PREFIX as _TARGET_MISSING_PREFIX,
    build_retry_feedback as _target_missing_retry_feedback,
    find_missing_targets as _find_missing_targets,
    guard_enabled as _target_guard_enabled,
    missing_target_error_message as _target_missing_error_message,
    universal_guard_enabled as _target_guard_universal_enabled,
)
from backend.core.ouroboros.governance.forward_progress import (
    candidate_content_hash,
)
from backend.core.ouroboros.governance.productivity_detector import (
    productivity_content_hash,
)
from dataclasses import asdict as _dc_asdict

from backend.core.ouroboros.governance.ledger import LedgerEntry, OperationState
from backend.core.ouroboros.governance.op_context import (
    GenerationResult,
    OperationContext,
    OperationPhase,
)
from backend.core.ouroboros.governance.phase_runner import (
    PhaseResult,
    PhaseRunner,
)

if TYPE_CHECKING:  # pragma: no cover
    from backend.core.ouroboros.governance.orchestrator import Orchestrator

# Match orchestrator's outer-gate grace constant (read at import time is
# fine — env won't change mid-op; inline does the same).
_OUTER_GATE_GRACE_S = float(os.environ.get("JARVIS_OUTER_GATE_GRACE_S", "15"))
_TRUTHY = frozenset({"1", "true", "yes", "on"})


logger = logging.getLogger("Ouroboros.Orchestrator")


# ──────────────────────────────────────────────────────────────────────────
# Task #11 — GENERATE replay adapter (closes GENERATE replay-blindness)
# ──────────────────────────────────────────────────────────────────────────
#
# The determinism substrate records/replays a phase's decision via an
# OutputAdapter that round-trips the phase output through JSON. Until now
# GENERATE captured only a provider-selection DIGEST (audit-only), so REPLAY
# re-invoked the model live — the single most expensive, least reproducible
# phase was the one blind to replay. This adapter serializes the full
# GenerationResult so REPLAY returns the recorded candidates and SKIPS the
# provider call entirely. Mirrors the ROUTE/CLASSIFY adapter-registration
# pattern (route_runner._register_route_adapter).
#
# ``candidates`` are already JSON-safe dicts; every scalar field round-trips.
# ``tool_execution_records`` (Tuple[Any, ...]) is LIVE-execution audit — no
# tools actually run during a replay, so it is (correctly) empty on the
# replayed result rather than reconstructed from opaque objects.
def _register_generate_adapter() -> None:
    try:
        from backend.core.ouroboros.governance.determinism.phase_capture import (
            OutputAdapter,
            register_adapter,
        )
        from backend.core.ouroboros.governance.op_context import (
            GenerationResult,
        )

        def _tool_rec_to_dict(rec: Any) -> Dict[str, Any]:
            """JSON-safe projection of a ToolExecutionRecord for the
            determinism ledger. The volatile wall-clock timings
            (started_at_ns / ended_at_ns / duration_ms) are DROPPED so the
            stored + hashed form is stable across runs (VERIFY-safe); every
            field the recap tool-count and the Iron Gate diversity scorer
            read -- tool_name, status, arguments_hash, output_bytes,
            round_index, identity -- is preserved. NEVER raises."""
            _st = getattr(rec, "status", None)
            return {
                "schema_version": str(getattr(rec, "schema_version", "") or ""),
                "op_id": str(getattr(rec, "op_id", "") or ""),
                "call_id": str(getattr(rec, "call_id", "") or ""),
                "round_index": int(getattr(rec, "round_index", 0) or 0),
                "tool_name": str(getattr(rec, "tool_name", "") or ""),
                "tool_version": str(getattr(rec, "tool_version", "") or ""),
                "arguments_hash": str(getattr(rec, "arguments_hash", "") or ""),
                "repo": str(getattr(rec, "repo", "") or ""),
                "policy_decision": str(getattr(rec, "policy_decision", "") or ""),
                "policy_reason_code": str(
                    getattr(rec, "policy_reason_code", "") or ""),
                "output_bytes": int(getattr(rec, "output_bytes", 0) or 0),
                "error_class": (
                    str(getattr(rec, "error_class", None))
                    if getattr(rec, "error_class", None) else None
                ),
                "status": (
                    getattr(_st, "value", None)
                    or (str(_st) if _st is not None else None)
                ),
            }

        def _tool_rec_from_dict(d: Any) -> Any:
            """Rebuild a ToolExecutionRecord from _tool_rec_to_dict's
            projection. Dropped timings rehydrate as None -- no consumer
            reads them and a fabricated timestamp would be a lie. An unknown
            status falls back to None (the diversity scorer treats None as
            succeeded). NEVER raises beyond a bad-shape guard."""
            from backend.core.ouroboros.governance.tool_executor import (
                ToolExecStatus,
                ToolExecutionRecord,
            )
            _raw = d.get("status") if isinstance(d, dict) else None
            try:
                _st = ToolExecStatus(_raw) if _raw is not None else None
            except Exception:  # noqa: BLE001
                _st = getattr(ToolExecStatus, str(_raw), None)
            return ToolExecutionRecord(
                schema_version=str(d.get("schema_version", "") or ""),
                op_id=str(d.get("op_id", "") or ""),
                call_id=str(d.get("call_id", "") or ""),
                round_index=int(d.get("round_index", 0) or 0),
                tool_name=str(d.get("tool_name", "") or ""),
                tool_version=str(d.get("tool_version", "") or ""),
                arguments_hash=str(d.get("arguments_hash", "") or ""),
                repo=str(d.get("repo", "") or ""),
                policy_decision=str(d.get("policy_decision", "") or ""),
                policy_reason_code=str(d.get("policy_reason_code", "") or ""),
                started_at_ns=None,
                ended_at_ns=None,
                duration_ms=None,
                output_bytes=int(d.get("output_bytes", 0) or 0),
                error_class=(d.get("error_class") or None),
                status=_st,
            )

        def _serialize(gen: Any) -> Any:
            if gen is None:
                return {"__none__": True}
            return {
                "candidates": [
                    dict(c) for c in (getattr(gen, "candidates", ()) or ())
                ],
                "provider_name": str(getattr(gen, "provider_name", "") or ""),
                "generation_duration_s": float(
                    getattr(gen, "generation_duration_s", 0.0) or 0.0,
                ),
                "model_id": str(getattr(gen, "model_id", "") or ""),
                "is_noop": bool(getattr(gen, "is_noop", False)),
                "tool_execution_records": [
                    _tool_rec_to_dict(r)
                    for r in (getattr(gen, "tool_execution_records", ()) or ())
                ],
                "venom_edit_history": [
                    dict(e)
                    for e in (getattr(gen, "venom_edit_history", ()) or ())
                ],
                "prompt_preloaded_files": [
                    str(f)
                    for f in (getattr(gen, "prompt_preloaded_files", ()) or ())
                ],
                "total_input_tokens": int(
                    getattr(gen, "total_input_tokens", 0) or 0,
                ),
                "total_output_tokens": int(
                    getattr(gen, "total_output_tokens", 0) or 0,
                ),
                "cost_usd": float(getattr(gen, "cost_usd", 0.0) or 0.0),
            }

        def _deserialize(stored: Any) -> Any:
            if not isinstance(stored, dict) or stored.get("__none__"):
                return None
            return GenerationResult(
                candidates=tuple(
                    dict(c) for c in (stored.get("candidates") or [])
                ),
                provider_name=str(stored.get("provider_name", "")),
                generation_duration_s=float(
                    stored.get("generation_duration_s", 0.0),
                ),
                model_id=str(stored.get("model_id", "")),
                is_noop=bool(stored.get("is_noop", False)),
                # Records ARE preserved across the round-trip (this adapter
                # runs in RECORD mode on the LIVE object, not only in REPLAY):
                # a determinism-stable projection was serialized above, so the
                # recap tool-count + Iron Gate exploration credit survive.
                tool_execution_records=tuple(
                    _tool_rec_from_dict(d)
                    for d in (stored.get("tool_execution_records") or [])
                ),
                venom_edit_history=tuple(
                    dict(e) for e in (stored.get("venom_edit_history") or [])
                ),
                prompt_preloaded_files=tuple(
                    str(f) for f in (stored.get("prompt_preloaded_files") or [])
                ),
                total_input_tokens=int(stored.get("total_input_tokens", 0)),
                total_output_tokens=int(stored.get("total_output_tokens", 0)),
                cost_usd=float(stored.get("cost_usd", 0.0)),
            )

        register_adapter(
            phase="GENERATE",
            kind="generate",
            adapter=OutputAdapter(
                serialize=_serialize,
                deserialize=_deserialize,
                name="generation_result_adapter",
            ),
        )
    except Exception:  # noqa: BLE001 — defensive (import-time)
        # Determinism module unavailable → capture_phase_decision
        # short-circuits to a pure passthrough. No import-time log spam.
        pass


_register_generate_adapter()


# ──────────────────────────────────────────────────────────────────────────
# Slice 12O — Foreground macro-cooldown helper
# ──────────────────────────────────────────────────────────────────────────
#
# Composes the canonical ForegroundCooldownPolicy + sleep_cooldown
# primitives. Module-level so the integration site (the
# Generation-attempt-failed branch ~L1420) stays single-line:
#
#     if await _slice12o_maybe_cooldown(orch, ctx, exc, route):
#         continue   # macro-retry without decrementing in-window counter
#
# Returns True iff cooldown was decided AND the sleep completed
# normally (op should re-attempt GENERATE). Returns False if the
# policy refused cooldown OR cooldown was decided but cancelled
# during shutdown (caller's existing terminal path handles).
# Re-raises CancelledError so the asyncio cascade can drain WAL +
# exit before the WallClockWatchdog Layer-3 hard-kill fires.

def _sovereign_physics_floor_s(base_s: float) -> float:
    """Sovereign-heavy physics floor for the route GENERATE budget.

    The route table is DW-API-sized; when the failover lifecycle is ENGAGED
    (AWAKENING/SERVING -- this op will route to the awakened heavy node whose
    single streaming round costs 200-400s), the outer wait_for it feeds
    severed EVERY multi-round attempt at ~375s (bt-iso-1782977669:
    'Generation attempt failed' with asyncio.TimeoutError's empty str at
    exactly the scaled budget). Floor the budget at
    ``expected_agentic_cycle_s()`` -- the SAME rounds x round-wall physics the
    BudgetPlan hint / Time-Dilated Deadline / arm-time walls share. DORMANT
    (normal DW ops) / master-off / any error -> base unchanged. Only ever
    RAISES (the route table stays the floor's floor). Master
    ``JARVIS_SOVEREIGN_GEN_PHYSICS_FLOOR_ENABLED`` (default true).
    NEVER raises."""
    try:
        if (os.environ.get("JARVIS_SOVEREIGN_GEN_PHYSICS_FLOOR_ENABLED", "true")
                or "").strip().lower() in ("0", "false", "no", "off"):
            return base_s
        from backend.core.ouroboros.governance import failover_lifecycle as _fl  # noqa: PLC0415
        if not _fl.lifecycle_enabled():
            return base_s
        if _fl.get_failover_controller().state == _fl.FailoverState.DORMANT:
            return base_s
        from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
            expected_agentic_cycle_s,
        )
        return max(float(base_s), float(expected_agentic_cycle_s()))
    except Exception:  # noqa: BLE001 -- a sizing floor must never break GENERATE
        return base_s


async def _slice12o_maybe_cooldown(
    *, orch: Any, ctx: Any, exc: BaseException, route: str,
) -> bool:
    """Slice 12O Phase 1+3 — check cooldown policy + execute
    cancellation-aware sleep if approved. NEVER eats CancelledError
    (caller catches + records terminal reason); returns False on
    any other failure path."""
    from backend.core.ouroboros.governance.foreground_cooldown import (
        get_default_policy as _slice12o_get_policy,
        sleep_cooldown as _slice12o_sleep_cooldown,
    )
    from backend.core.ouroboros.governance.circuit_breaker import (
        CircuitTripOrigin as _Slice12O_Origin,
    )

    # Resolve origin via the canonical ProviderRoute → origin map
    # (single source of truth, lives in candidate_generator). Empty
    # / unknown route → FOREGROUND (matches Slice 12N default).
    try:
        from backend.core.ouroboros.governance.candidate_generator import (
            _SLICE12N_ROUTE_TO_ORIGIN as _slice12n_map,
        )
        _origin = _slice12n_map.get(
            (route or "").strip().lower(),
            _Slice12O_Origin.FOREGROUND,
        )
    except Exception:  # noqa: BLE001 — be conservative
        _origin = _Slice12O_Origin.FOREGROUND

    _is_foreground = _origin == _Slice12O_Origin.FOREGROUND

    # Compose the canonical exception → reason-code shape used by
    # the breaker. We accept either an explicit code on ctx (set by
    # the breaker's TERMINATE_UNRESOLVED path) or fall back to the
    # exception's message.
    _terminal_reason_code = (
        getattr(ctx, "terminal_reason_code", None)
        or str(exc)
        or ""
    )

    # Read CostGovernor + WallClockWatchdog snapshots if exposed
    # on the orchestrator stack. Both are best-effort — the policy
    # treats None as "caller doesn't know" and skips the gate.
    _remaining_budget = None
    _remaining_wall = None
    try:
        _cost_gov = getattr(
            getattr(orch, "_stack", None), "cost_governor", None,
        )
        if _cost_gov is not None and hasattr(_cost_gov, "remaining_for_op"):
            _remaining_budget = _cost_gov.remaining_for_op(
                getattr(ctx, "op_id", "") or "",
            )
    except Exception:  # noqa: BLE001
        _remaining_budget = None
    try:
        _wd = getattr(
            getattr(orch, "_stack", None), "wall_clock_watchdog", None,
        )
        if _wd is not None and hasattr(_wd, "remaining_seconds"):
            _remaining_wall = _wd.remaining_seconds()
    except Exception:  # noqa: BLE001
        _remaining_wall = None

    _policy = _slice12o_get_policy()
    _decision = _policy.decide(
        op_id=getattr(ctx, "op_id", "") or "",
        origin_is_foreground=_is_foreground,
        terminal_reason_code=_terminal_reason_code,
        remaining_budget_usd=_remaining_budget,
        remaining_wall_s=_remaining_wall,
    )

    if not _decision.should_cooldown:
        logger.debug(
            "[ForegroundCooldown] op=%s refused reason=%s",
            getattr(ctx, "op_id", "?")[:16],
            _decision.refuse_reason or "?",
        )
        return False

    # Decision committed — record the attempt BEFORE sleeping so
    # the counter is observable in telemetry even if the sleep
    # gets cancelled mid-flight.
    _policy.record_attempt(getattr(ctx, "op_id", "") or "")
    logger.warning(
        "[ForegroundCooldown] op=%s parking for cooldown "
        "attempt=%d/%d reason=%s backoff_s=%.0f route=%s "
        "remaining_budget=%s remaining_wall_s=%s",
        getattr(ctx, "op_id", "?")[:16],
        _decision.attempt, _decision.max_attempts,
        (_decision.reason.value if _decision.reason else "?"),
        _decision.backoff_s, route,
        f"${_remaining_budget:.3f}" if _remaining_budget is not None else "?",
        f"{_remaining_wall:.0f}" if _remaining_wall is not None else "?",
    )

    # CancellationError propagates — caller catches + records
    # terminal reason "cooldown_cancelled_shutdown".
    await _slice12o_sleep_cooldown(
        _decision.backoff_s,
        op_id=getattr(ctx, "op_id", "") or "",
        label=f"attempt_{_decision.attempt}_of_{_decision.max_attempts}",
    )
    logger.info(
        "[ForegroundCooldown] op=%s cooldown complete attempt=%d/%d "
        "— re-attempting GENERATE",
        getattr(ctx, "op_id", "?")[:16],
        _decision.attempt, _decision.max_attempts,
    )
    return True


class GENERATERunner(PhaseRunner):
    """Verbatim transcription of orchestrator.py GENERATE block (~3138-4748)."""

    phase = OperationPhase.GENERATE

    def __init__(
        self,
        orchestrator: "Orchestrator",
        serpent: Optional[Any],
        consciousness_bridge: Optional[Any] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._serpent = serpent
        self._consciousness_bridge = consciousness_bridge

    async def run(self, ctx: OperationContext) -> PhaseResult:
        from backend.core.ouroboros.governance.op_context import (  # noqa: PLC0415
            replan_inputs as _replan_inputs,
        )
        orch = self._orchestrator
        _serpent = self._serpent
        _consciousness_bridge = self._consciousness_bridge

        # Resolve orchestrator module-level helpers/classes referenced by
        # the verbatim inline block. Late import to avoid circular deps.
        from backend.core.ouroboros.governance.orchestrator import (
            _PreloadedExplorationRecord,
        )

        # W2(4) Slice 2 — bind the per-op CuriosityBudget to the ambient
        # ContextVar so tool_executor Rule 14 can consult it during the
        # Venom tool loop. Master flag default-off → curiosity_enabled()
        # returns False → CuriosityBudget.try_charge() always denies →
        # tool_executor's curiosity widening short-circuits to the
        # legacy SAFE_AUTO reject. Byte-for-byte pre-W2(4) when master
        # is off. Best-effort: any exception here must not block GENERATE.
        try:
            from backend.core.ouroboros.governance.curiosity_engine import (
                CuriosityBudget as _CuriosityBudget,
                curiosity_budget_var as _curiosity_budget_var,
                curiosity_enabled as _curiosity_enabled,
            )
            if _curiosity_enabled():
                # Resolve current posture via Wave 1 #1 DirectionInferrer
                # observer. If unavailable (test orchestrator without a
                # PostureStore), default to "UNKNOWN" so the posture
                # allowlist gate denies cleanly.
                _posture_str = "UNKNOWN"
                try:
                    from backend.core.ouroboros.governance.posture_observer import (  # noqa: E501
                        get_default_store as _get_default_posture_store,
                    )
                    _store = _get_default_posture_store()
                    _reading = (
                        _store.current_reading() if _store is not None else None
                    )
                    if _reading is not None:
                        _posture_str = str(
                            getattr(_reading, "posture", _reading)
                        )
                        # Some posture types are Enum.NAME — strip the prefix
                        if "." in _posture_str:
                            _posture_str = _posture_str.split(".", 1)[1]
                except Exception:  # noqa: BLE001
                    pass
                # Resolve session_dir for the JSONL ledger (best-effort).
                _session_dir = None
                try:
                    _gls = getattr(orch._stack, "governed_loop_service", None)
                    if _gls is not None:
                        _sd = getattr(_gls, "_session_dir", None)
                        if _sd is not None:
                            from pathlib import Path as _Path
                            _session_dir = (
                                _sd if isinstance(_sd, _Path) else _Path(_sd)
                            )
                except Exception:  # noqa: BLE001
                    pass
                _curiosity_budget_var.set(_CuriosityBudget(
                    op_id=ctx.op_id,
                    posture_at_arm=_posture_str,
                    session_dir=_session_dir,
                ))
        except Exception:  # noqa: BLE001 — best-effort, never blocks GENERATE
            pass

        # ---- VERBATIM transcription of orchestrator.py 3138-4748 ----
        if _serpent: _serpent.update_phase("GENERATE")
        # ---- Phase 3: GENERATE (with retry + episodic failure memory) ----
        generation: Optional[GenerationResult] = None
        generate_retries_remaining = orch._config.max_generate_retries

        # ── Slice 12AH: synthetic GENERATE bypass for wiring-validation
        # fixtures ─────────────────────────────────────────────────────
        # Closes the bt-2026-05-24-080247 wedge structurally: a
        # wiring-validation fixture has ``target_files=()`` by canonical
        # SWE-Bench-Pro protocol (cheat-detection — test paths must not
        # be surfaced as the agent's target; gold_patch paths would leak
        # the solution). GENERATE has nothing to point at → produces
        # no candidate → L2 cancels with 3 insufficient claims. The
        # structurally CORRECT answer for any fixture with
        # ``gold_patch=""`` and a trivially-passing test is "no patch
        # needed" — exactly what the existing 2b.1-noop terminal path
        # at ``if generation.is_noop:`` (~line 2163) handles.
        #
        # This block composes:
        #   * ``envelope_metadata.is_route_wiring_validation_envelope``
        #     (Slice 12AD's 2-signal AND detector — operator-canonical
        #     ``fixture_purpose=="wiring_validation" AND
        #     real_benchmark is False``); REJECTS real benchmarks by
        #     defense-in-depth exact-False.
        #   * ``op_context.GenerationResult(is_noop=True)`` — the same
        #     dataclass the providers already emit on
        #     ``{"no_op": true}`` model responses.
        #   * The existing retry-loop short-circuit on
        #     ``generation is not None and generation.is_noop`` — no
        #     control-flow change beyond an early-break.
        #
        # Result: provider cascade NEVER invoked (zero Claude / DW
        # spend on the GENERATE phase), op flows directly to the
        # canonical noop terminal → ctx.advance(COMPLETE) →
        # operation_terminal SSE → fixture COMPLETE.
        #
        # NEVER raises; any defensive failure falls through to the
        # legacy retry loop (which then exhausts naturally — no worse
        # than pre-Slice-12AH behaviour for fixtures).
        try:
            from backend.core.ouroboros.governance.envelope_metadata import (  # noqa: E501
                is_route_wiring_validation_envelope as _slice12ah_is_fixture,
            )
            if _slice12ah_is_fixture(ctx):
                generation = GenerationResult(
                    candidates=(),
                    provider_name="slice_12ah_synthetic_noop",
                    generation_duration_s=0.0,
                    is_noop=True,
                )
                logger.info(
                    "[Slice12AH] wiring-validation fixture detected "
                    "— synthesizing 2b.1-noop, skipping provider "
                    "cascade for op=%s (fixture_purpose=wiring_"
                    "validation AND real_benchmark=False)",
                    ctx.op_id[:12],
                )
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[Slice12AH] wiring-validation detector raised — "
                "falling through to legacy provider cascade",
                exc_info=True,
            )

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
        if orch._session_lessons:
            _code_lessons = [
                text for (ltype, text) in orch._session_lessons
                if ltype == "code"
            ][-orch._session_lessons_max:]
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

        # ── Slice 247 — State-Drift Reconciliation (RE-ALIGNMENT micro-phase) ──
        # If this op already carries file hashes from a PRIOR generation (a
        # resumed / resurrected op re-entering GENERATE after preemption — Slices
        # 245/246), compare that PRESERVED baseline against the current disk
        # BEFORE the re-snapshot below erases it. On drift (a human override
        # patched a target during the suspension window), inject a re-alignment
        # instruction so the model re-reads the drifted files and regenerates
        # against the NEW state — never blind-patching a stale target. Zero-LLM,
        # gated, fail-soft; reuses the strategic_memory_prompt injection channel.
        try:
            from backend.core.ouroboros.governance.state_drift import (
                detect_drift as _detect_drift,
                build_realignment_feedback as _build_realignment_feedback,
                state_drift_reconcile_enabled as _drift_reconcile_enabled,
                STATE_CONTEXT_DRIFTED as _STATE_CONTEXT_DRIFTED,
            )
            if ctx.generate_file_hashes and _drift_reconcile_enabled():
                _drifted = _detect_drift(
                    ctx.generate_file_hashes, orch._config.project_root,
                )
                if _drifted:
                    _realign = _build_realignment_feedback(_drifted)
                    _existing_mem = getattr(ctx, "strategic_memory_prompt", "") or ""
                    ctx = dataclasses.replace(
                        ctx,
                        strategic_memory_prompt=(
                            f"{_existing_mem}\n\n{_realign}" if _existing_mem else _realign
                        ),
                    )
                    logger.warning(
                        "[Orchestrator] STATE=%s op=%s files=%s — injecting "
                        "re-alignment (re-read forced before patch)",
                        _STATE_CONTEXT_DRIFTED, ctx.op_id[:12], _drifted[:3],
                    )
        except Exception:  # noqa: BLE001 — drift reconcile must never crash GENERATE
            logger.debug("[Orchestrator] state-drift reconcile skipped", exc_info=True)

        # ── Stale-exploration guard: snapshot file hashes at GENERATE time ──
        _gen_hashes: list = []
        for _tf in ctx.target_files:
            _tf_path = orch._config.project_root / _tf
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

        for attempt in range(1 + orch._config.max_generate_retries):
            # Slice 12AH — synthetic-noop pre-set (above, for
            # wiring-validation fixtures) short-circuits the retry loop
            # on the FIRST iteration with zero provider-cascade work.
            # Without this break the loop body would call
            # ``orch._generator.generate(...)`` and overwrite the
            # synthetic ``is_noop=True`` with whatever the provider
            # returns. Placed at the very top of the loop body so even
            # the per-op cost-cap check below doesn't fire (the synthetic
            # noop costs $0 and shouldn't be subject to per-op caps).
            if generation is not None and generation.is_noop:
                break
            # ── Per-op cost cap check (Manifesto §5/§7) ──
            # If the cumulative spend across previous attempts has already
            # exceeded the dynamic cap, refuse to initiate another provider
            # call. Routes through the phase-aware terminal picker.
            if orch._cost_governor.is_exceeded(ctx.op_id):
                _cost_summary = orch._cost_governor.summary(ctx.op_id) or {}
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
                _terminal = orch._l2_escape_terminal(ctx.phase)
                ctx = ctx.advance(
                    _terminal,
                    terminal_reason_code="op_cost_cap_exceeded",
                )
                await orch._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {
                        "reason": "op_cost_cap_exceeded",
                        "cost_summary": dict(_cost_summary),
                        "entry_phase": "GENERATE",
                    },
                )
                orch._cost_governor.finish(ctx.op_id)
                return PhaseResult(

                    next_ctx=ctx, next_phase=None, status="fail",

                    reason=ctx.terminal_reason_code or "generate_terminal",

                    artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                )

            try:
                # Heartbeat: GENERATE phase starting (Manifesto §7: Absolute Observability)
                try:
                    await orch._stack.comm.emit_heartbeat(
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
                # Slice 15 T4 — value-band adaptive allocation: the REAL
                # dispatch lever scales as a function of verifiable
                # semantic weight (oracle-band repairs earn a wider
                # window; cosmetic/indeterminate never do). Fail-soft.
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
                    _route, orch._config.generation_timeout_s
                )
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
                # ── Phase R1: outer/inner timeout coherence ──────────
                # Soak bt-2026-05-18-015317: COMPLEX outer _gen_timeout
                # (240s) + _OUTER_GATE_GRACE_S (15s) = 255s killed
                # GENERATE with CancelledError BEFORE the inner fallback
                # widened its cap to the 360s thinking window — psf
                # never generated. The outer Iron-Gate window MUST be
                # >= the inner thinking cap for thinking-likely calls.
                # Consume the SAME shared predicate + cap the inner
                # fallback uses (candidate_generator) so the invariant
                # holds by construction — no duplicated rule, no
                # per-path drift. Lazy import mirrors the adaptive
                # block below (no module-load-order coupling).
                try:
                    from backend.core.ouroboros.governance.candidate_generator import (  # noqa: E501
                        gen_call_likely_thinking,
                        fallback_thinking_cap_s,
                    )
                    if gen_call_likely_thinking(
                        _route, getattr(ctx, "task_complexity", "") or "",
                    ):
                        _cap = fallback_thinking_cap_s()
                        if _cap > _gen_timeout:
                            logger.info(
                                "[Orchestrator] R1 thinking-cap floor: "
                                "gen_timeout %.0fs → %.0fs route=%s "
                                "op=%s (outer >= inner 360s window)",
                                _gen_timeout, _cap, _route,
                                getattr(ctx, "op_id", "?"),
                            )
                        _gen_timeout = max(_gen_timeout, _cap)
                except Exception:  # noqa: BLE001 — fail-open to route base
                    logger.debug(
                        "[Orchestrator] R1 thinking-cap floor skipped "
                        "(fail-open to route base)", exc_info=True,
                    )
                # ── Slice 50 Phase 2: force-batch deadline floor ──────
                # LIVE phase-dispatcher copy (mirror of orchestrator.py).
                # When this op force-batches (Slice 36/41), the DW async
                # batch poll runs up to JARVIS_DW_BATCH_TIMEOUT_S (Slice 43,
                # 300s). A route-base deadline shorter than that lease
                # (standard=220s for a trivial op the R1 floor skips) lets
                # the OUTER deadline sever the batch mid-poll — v45 probe
                # bt-2026-06-01-034745: op-...e944 remaining=220s,
                # min(220,300)=220, batch killed at 220s, 0 APPLY. Floor to
                # batch_cap + overhead so the inner hold gets the full lease.
                # Safe: force-batch only engages when Claude is disabled.
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
                                "deadline floor: gen_timeout %.0fs → %.0fs "
                                "route=%s op=%s — batch lease no longer "
                                "severed by outer deadline",
                                _gen_timeout, _fb_floored, _route,
                                getattr(ctx, "op_id", "?"),
                            )
                        _gen_timeout = _fb_floored
                except Exception:  # noqa: BLE001 — fail-open to route base
                    logger.debug(
                        "[Orchestrator] force-batch deadline floor skipped "
                        "(fail-open to route base)", exc_info=True,
                    )
                # Slice 2 parity — payload-adaptive GENERATE budget runs
                # AFTER the thinking-cap floor so one scaled value
                # propagates to deadline + outer wait_for + tool-loop
                # budget. Mirrors orchestrator.py verbatim. Master flag
                # default-FALSE; fail-open to (floored) route base.
                try:
                    from backend.core.ouroboros.governance.adaptive_gen_budget import (  # noqa: E501
                        scale_gen_timeout,
                    )
                    _adaptive_gt = scale_gen_timeout(_gen_timeout, ctx)
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
                        "[Orchestrator] adaptive gen budget skipped "
                        "(fail-open to route base)", exc_info=True,
                    )
                # Sovereign-heavy physics floor — the LAST static clock:
                # engaged lifecycle => budget floors at rounds x round-wall
                # so one floored value propagates to deadline + outer
                # wait_for + tool-loop budget.
                _physics_gt = _sovereign_physics_floor_s(_gen_timeout)
                if _physics_gt > _gen_timeout:
                    logger.info(
                        "[Orchestrator] sovereign physics floor: gen_timeout "
                        "%.0fs → %.0fs route=%s op=%s",
                        _gen_timeout, _physics_gt, _route,
                        getattr(ctx, "op_id", "?"),
                    )
                    _gen_timeout = _physics_gt
                deadline = datetime.now(tz=timezone.utc) + timedelta(
                    seconds=_gen_timeout
                )
                # Emit streaming=start so SerpentFlow can render the
                # "synthesizing" header before tokens begin flowing.
                # Provider is unknown at this point (chosen during adaptive failback).
                try:
                    await orch._stack.comm.emit_heartbeat(
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
                    # Task #11 — close GENERATE replay-blindness. The ENTIRE
                    # generation acquisition (parallel-edge fan-out + the
                    # park-aware single-stream seam) is wrapped in
                    # capture_phase_decision(kind="generate"). In REPLAY mode
                    # the recorded GenerationResult is returned via the
                    # module-registered adapter and the provider is NOT
                    # re-invoked — the model call is skipped entirely. In the
                    # default PASSTHROUGH mode this is bit-for-bit legacy
                    # (compute runs, nothing recorded); RECORD serializes the
                    # result. This REPLACES the old digest-only capture, which
                    # recorded a provider hash yet always re-ran the model.
                    async def _acquire_generation() -> Any:
                        # Phase B parallel-edge exploitation (Manifesto §2+§3).
                        # DAG-driven fan-out first; ANY fallback condition
                        # returns None → legacy single-stream path runs
                        # byte-identically below.
                        _parallel_gen = None
                        try:
                            from backend.core.ouroboros.governance.plan_exploit import (
                                try_parallel_generate,
                            )
                            _parallel_gen = await try_parallel_generate(
                                ctx,
                                deadline,
                                _gen_timeout,
                                orch._generator,
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
                            return _parallel_gen
                        # Stage 1.6 — park-aware GENERATE seam (RESUME /
                        # PARK-EMIT / LEGACY); byte-identical to the legacy
                        # direct-await when ``JARVIS_BG_PARK_ENABLED`` is off
                        # (default §33.1). On PARK-EMIT it raises
                        # ParkRequested — a BaseException, so it flies past
                        # the capture wrapper's Exception handlers AND the
                        # import fallback below, unwinding cleanly (the
                        # generator is never double-run). See
                        # generate_park_wrapper.py.
                        from backend.core.ouroboros.governance.generate_park_wrapper import (
                            maybe_park_or_resume,
                        )
                        return await maybe_park_or_resume(
                            orch=orch,
                            ctx=ctx,
                            deadline=deadline,
                            gen_timeout=_gen_timeout,
                            outer_grace_s=_OUTER_GATE_GRACE_S,
                        )

                    # Only an IMPORT failure of the determinism module is
                    # caught here → direct acquisition (compute has not run,
                    # so re-running is single-execution). A genuine GENERATE
                    # exception raised BY the generator propagates through
                    # capture_phase_decision exactly as the legacy direct
                    # call did (never re-run → no double generation).
                    try:
                        from backend.core.ouroboros.governance.determinism.phase_capture import (
                            capture_phase_decision as _capture_generate,
                        )
                    except Exception:  # noqa: BLE001 — determinism unavailable
                        _capture_generate = None
                    if _capture_generate is not None:
                        generation = await _capture_generate(
                            op_id=ctx.op_id,
                            phase="GENERATE",
                            kind="generate",
                            ctx=ctx,
                            compute=_acquire_generation,
                            extra_inputs={
                                "provider_route": str(
                                    getattr(ctx, "provider_route", "") or "",
                                ),
                            },
                        )
                    else:
                        generation = await _acquire_generation()
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
                        orch._cost_governor.charge(
                            ctx.op_id, _cost_this_call, _prov_name,
                            phase=_phase_tag,
                        )
                        await orch._emit_route_cost_heartbeat(
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
                    await orch._stack.comm.emit_heartbeat(
                        op_id=ctx.op_id, phase="generate", progress_pct=49.0,
                        streaming="end",
                    )
                except Exception:
                    pass

                # ── LEDGER-SEAM STATE BINDING (root-cause, DRY) ──
                # The generation carries its tool-execution records
                # (provider.with_tool_records). Every terminal path BELOW
                # (noop break, forward-progress, productivity, iron-gate,
                # exhaustion) records the ledger with `ctx`, and the
                # orchestrator's terminal seam then synthesises the op recap
                # from `ctx.generation`. Those advances did not carry the
                # generation, so `ctx.generation` reached the seam as None and
                # the recap's tool count was silently zero. Bind it to ctx
                # HERE, once, the moment it is finalised — the SAME late-bind
                # idiom the route already uses (`object.__setattr__(ctx,
                # "provider_route", …)`), so `dataclasses.replace` in every
                # subsequent advance carries it immutably to the seam. No
                # per-branch `generation=` to forget, no secondary cache.
                if generation is not None:
                    object.__setattr__(ctx, "generation", generation)
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
                    if _fp_hash and orch._forward_progress.observe(
                        ctx.op_id, _fp_hash,
                    ):
                        _fp_summary = orch._forward_progress.summary(ctx.op_id) or {}
                        logger.warning(
                            "[Orchestrator] Forward-progress trip: op=%s "
                            "stuck after %d repeats — escaping retry loop",
                            ctx.op_id,
                            _fp_summary.get("repeat_count", 0),
                        )
                        _terminal = orch._l2_escape_terminal(ctx.phase)
                        ctx = ctx.advance(
                            _terminal,
                            terminal_reason_code="no_forward_progress",
                        )
                        await orch._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": "no_forward_progress",
                                "progress_summary": dict(_fp_summary),
                                "entry_phase": "GENERATE",
                            },
                        )
                        orch._forward_progress.finish(ctx.op_id)
                        return PhaseResult(

                            next_ctx=ctx, next_phase=None, status="fail",

                            reason=ctx.terminal_reason_code or "generate_terminal",

                            artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                        )
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
                        level=orch._productivity_detector.level,
                    )
                    if _pd_hash and orch._productivity_detector.observe(
                        ctx.op_id, _cost_this_call, _pd_hash,
                    ):
                        _pd_summary = orch._productivity_detector.summary(ctx.op_id) or {}
                        logger.warning(
                            "[Orchestrator] Productivity stall: op=%s "
                            "burned=$%.4f stable=%d level=%s — escaping retry loop",
                            ctx.op_id,
                            _pd_summary.get("cost_since_last_change_usd", 0.0),
                            _pd_summary.get("consecutive_stable", 0),
                            _pd_summary.get("config", {}).get("normalization_level", "?"),
                        )
                        _terminal = orch._l2_escape_terminal(ctx.phase)
                        ctx = ctx.advance(
                            _terminal,
                            terminal_reason_code="stalled_productivity",
                        )
                        await orch._record_ledger(
                            ctx,
                            OperationState.FAILED,
                            {
                                "reason": "stalled_productivity",
                                "productivity_summary": dict(_pd_summary),
                                "entry_phase": "GENERATE",
                            },
                        )
                        orch._productivity_detector.finish(ctx.op_id)
                        return PhaseResult(

                            next_ctx=ctx, next_phase=None, status="fail",

                            reason=ctx.terminal_reason_code or "generate_terminal",

                            artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                        )
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
                # discipline (paired with orchestrator.py:4445).
                # SWE-Bench-Pro wiring-validation fixtures drop the
                # exploration floor to 0 so the no-op-passes
                # structural contract isn't deadlocked by the gate.
                # Pure envelope-metadata composition; no hardcoded
                # instance_ids.
                try:
                    from backend.core.ouroboros.governance.envelope_metadata import (  # noqa: E501
                        is_wiring_validation_envelope as _slice12p_is_fixture,
                    )
                    if _slice12p_is_fixture(ctx):
                        logger.info(
                            "[Orchestrator] Iron Gate — Slice 12P "
                            "envelope-aware override: wiring-validation "
                            "fixture detected — exploration floor 0 "
                            "for op=%s",
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
                    # Slice 12P — fixture override drops floor to 0;
                    # skip the gate path entirely in that case so we
                    # don't emit confusing "Iron Gate" log lines for
                    # an envelope with nothing to enforce.
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
                                generation = None
                                raise ExplorationInsufficientError(
                                    _decision_msg,
                                    verdict=_verdict,
                                    floors=_floors,
                                )
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
                            # Slice 12 (Run-22 catch): FULL op id — this is
                            # an audit-keyed REJECT marker, and THIS extracted
                            # runner is the live GENERATE path; its private
                            # [:12] copy survived the Slice-11 orchestrator
                            # fix (T5 lesson) and was excused at audit only
                            # by the ambiguity rule. Sweep test now covers
                            # phase_runners/.
                            "[Orchestrator] Iron Gate — exploration_insufficient: "
                            "%d/%d (attempt=%d cumulative, preloaded=%d) for op=%s",
                            _op_explore_credit, _min_explore, attempt + 1,
                            _preloaded_credit, ctx.op_id,
                        )
                        generation = None
                        raise RuntimeError(_explore_err)

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
                        _dep_result = _dep_check(_cand, orch._config.project_root)
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

                # Gate 3.5 — Target existence (Slice 72, UNIVERSALIZED +
                # ported to the SHIPPING path 2026-07-22). HISTORY: this
                # gate previously lived ONLY on the inline orchestrator
                # twin, which stopped being the shipping GENERATE path when
                # this runner graduated (2026-04-23) — the wired-but-inert
                # trap. Soak bt-2026-07-22-005943 proved it: a candidate
                # whose parent chain doesn't exist in the write tree sailed
                # ungated to APPLY and hard-ENOENT'd in the ChangeEngine.
                #
                # Synchronized Root Injection: the gate consults the EXACT
                # tree APPLY will write to — the same resolution the
                # ChangeRequest carries (_swe_bench_write_root for
                # benchmark ops; the Slice 11 execution_root seam
                # otherwise, which is ChangeEngine._effective_write_root's
                # fallback by construction). Existence stats run OFF-LOOP
                # via asyncio.to_thread. Host lane keeps legitimate
                # new-file creation (allow_new_files: a missing target is
                # only a steering error when its PARENT dir is also
                # missing). Benchmark semantics byte-identical strict.
                _tg_is_benchmark = (
                    getattr(ctx, "signal_source", "") == "swe_bench_pro"
                )
                _tg_missing: list = []
                if (
                    (_tg_is_benchmark and _target_guard_enabled())
                    or (not _tg_is_benchmark
                        and _target_guard_universal_enabled())
                ):
                    # Fail-SOFT infrastructure resolution: the gate is
                    # protective, never fatal — a host that cannot resolve
                    # a write root (parity-test fakes, degraded config)
                    # skips the check rather than failing the candidate.
                    # The deliberate target_file_missing raise below stays
                    # OUTSIDE this try so it is never swallowed.
                    try:
                        _tg_write_root = (
                            orch._swe_bench_write_root(ctx)
                            if _tg_is_benchmark
                            else orch._config.execution_root
                        )
                        _tg_missing = await asyncio.to_thread(
                            _find_missing_targets,
                            generation.candidates,
                            _tg_write_root,
                            allow_new_files=not _tg_is_benchmark,
                        )
                    except Exception:  # noqa: BLE001 — protective gate
                        logger.debug(
                            "[Orchestrator] Iron Gate — target-existence "
                            "infrastructure unresolvable; gate skipped "
                            "op=%s", ctx.op_id[:12], exc_info=True,
                        )
                        _tg_missing = []
                        _tg_write_root = None
                    if _tg_missing:
                        logger.warning(
                            "[Orchestrator] Iron Gate — target_file_missing: "
                            "%s not in write root %s op=%s (lane=%s)",
                            ",".join(_tg_missing), _tg_write_root,
                            ctx.op_id[:12],
                            "benchmark" if _tg_is_benchmark else "universal",
                        )
                        generation = None
                        raise RuntimeError(
                            _target_missing_error_message(_tg_missing)
                        )

                # Gate 4.5 — Source-domain purity (Slice 78, ported to the
                # SHIPPING path 2026-07-22 runner-parity audit — this gate
                # previously lived ONLY on the inline twin, so benchmark
                # ops on the shipping runner lacked the "cheat the
                # held-out suite" protection). For a swe_bench op, a
                # candidate that modifies ONLY test files is caught HERE
                # and routed to GENERATE_RETRY telling the model to fix
                # the SOURCE defect instead. INERT for non-swe_bench ops
                # (host self-dev legitimately authors tests; that lane is
                # governed by the attribution NOTIFY floor at GATE).
                # Naming-heuristic only so a source file is never
                # misclassified as a test.
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
                                        _mod_paths.append(
                                            str(_cand["file_path"])
                                        )
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
                                        "modifies only tests [%s] op=%s",
                                        ", ".join(_dg_verdict.test_files),
                                        ctx.op_id[:12],
                                    )
                                    generation = None
                                    raise RuntimeError(
                                        _dg_reason
                                        or "patch_domain_test_only"
                                    )

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
                        _ds_result = _docstring_check(_cand, orch._config.project_root)
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
                            orch._config.project_root,
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
                    await orch._stack.comm.emit_heartbeat(
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
                    for _t in getattr(orch._stack.comm, "_transports", []):
                        try:
                            await _t.send(_gen_msg)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Success -- record reasoning trace + dialogue
                if orch._reasoning_narrator is not None:
                    try:
                        orch._reasoning_narrator.record_generate(
                            ctx.op_id, generation.provider_name,
                            len(generation.candidates), generation.generation_duration_s,
                        )
                    except Exception:
                        pass
                if orch._dialogue_store is not None:
                    try:
                        _d = orch._dialogue_store.get_active(ctx.op_id)
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

            except Exception as exc:
                _err_msg = str(exc)
                _route = getattr(ctx, "provider_route", "")

                # ── PARTIAL-FLUSH (root-cause resilience) ──
                # A tool loop that RAISED or TIMED OUT attaches the records
                # it did run to the exception (tool_executor._attach_tool_
                # records). Bind a minimal generation carrying them to ctx
                # HERE, at the top of the one failure handler, so every
                # failure terminal below — and the recap the seam draws from
                # ctx.generation — reflects the PARTIAL execution instead of
                # dropping it to zero. Only when nothing real is bound yet
                # (never overwrite a genuine generation). Same late-bind idiom
                # as the success path. NEVER raises into the handler.
                try:
                    _partial_recs = getattr(exc, "tool_execution_records", ()) or ()
                    if _partial_recs and getattr(ctx, "generation", None) is None:
                        from backend.core.ouroboros.governance.op_context import (  # noqa: E501,PLC0415
                            GenerationResult as _PartialGen,
                        )
                        object.__setattr__(ctx, "generation", _PartialGen(
                            candidates=(),
                            provider_name="partial_tool_flush",
                            generation_duration_s=0.0,
                            tool_execution_records=tuple(_partial_recs),
                        ))
                except Exception:  # noqa: BLE001 — a flush must never break the handler
                    pass

                # ── Sovereign Egress Interceptor Mesh (T3) — route-back ──
                # The LIVE generate-failure handler. When OUR egress interceptor
                # blocked an over-ceiling DW body (LocalEgressOverweightError,
                # re-raised from candidate_generator carrying max_allowed_size),
                # this is NOT a vendor rupture and NOT a retryable generation
                # failure — retrying the SAME oversized body would just block
                # again. Instead route it BACK to context-aware chunking:
                # decompose the GOAL with compression_target=max_allowed_size so
                # each re-injected sub-goal fits under the ceiling, then terminate
                # this op ``decomposed`` (the sub-goals carry the work forward) —
                # exactly the BLOCK->decompose->re-inject seam. Chunking-eligible
                # gate: only when at least one target_file is present (a symbol-
                # scopable mutation). Fail-soft ABSOLUTE: any error falls through
                # to the legacy generation-failure path below (op is never lost).
                try:
                    from backend.core.ouroboros.governance.dw_fault_taxonomy import (  # noqa: E501
                        is_local_egress_overweight as _egress_overweight,
                    )
                    if _egress_overweight(exc):
                        _max_allowed = getattr(exc, "max_allowed_size", None)
                        _eligible = bool(
                            getattr(ctx, "target_files", ()) or ()
                        )
                        from backend.core.ouroboros.governance.goal_decomposition_planner import (  # noqa: E501
                            chunking_enabled as _chunking_enabled,
                        )
                        if (
                            _eligible
                            and _chunking_enabled()
                            and isinstance(_max_allowed, int)
                        ):
                            logger.warning(
                                "[Orchestrator] LOCAL_EGRESS_OVERWEIGHT op=%s — "
                                "egress interceptor blocked an over-ceiling body "
                                "(max_allowed=%d); routing BACK to context-aware "
                                "chunking (compression_target=%d) instead of a "
                                "doomed retry",
                                ctx.op_id, _max_allowed, _max_allowed,
                            )
                            _re_ctx = await orch._decompose_block_or_legacy(
                                ctx, None, compression_target=_max_allowed,
                            )
                            await orch._record_ledger(
                                _re_ctx,
                                OperationState.COMPLETED,
                                {
                                    "reason": "local_egress_overweight_rechunk",
                                    "max_allowed_size": _max_allowed,
                                    "terminal_reason_code": getattr(
                                        _re_ctx, "terminal_reason_code", "",
                                    ),
                                },
                            )
                            return PhaseResult(
                                next_ctx=_re_ctx,
                                next_phase=None,
                                status="fail",
                                reason=(
                                    getattr(_re_ctx, "terminal_reason_code", "")
                                    or "local_egress_overweight_rechunk"
                                ),
                                artifacts={
                                    "generation": generation,
                                    "episodic_memory": _episodic_memory,
                                },
                            )
                except Exception:  # noqa: BLE001 — fail-soft to legacy path
                    logger.debug(
                        "[Orchestrator] egress re-chunk seam fail-soft -> legacy "
                        "generation-failure path (op=%s)",
                        getattr(ctx, "op_id", "?"), exc_info=True,
                    )

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
                    await orch._record_ledger(
                        ctx, OperationState.COMPLETED,
                        {"reason": "speculative_deferred", "route": "speculative"},
                    )
                    return PhaseResult(

                        next_ctx=ctx, next_phase=None, status="fail",

                        reason=ctx.terminal_reason_code or "generate_terminal",

                        artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                    )

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
                    await orch._record_ledger(
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
                    return PhaseResult(

                        next_ctx=ctx, next_phase=None, status="fail",

                        reason=ctx.terminal_reason_code or "generate_terminal",

                        artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                    )

                # ── Absolute Route Sealing terminal ──
                # The router committed to the sovereign J-Prime provider and it
                # failed; the DW cascade is FORBIDDEN. This is a non-retryable
                # terminal (re-driving through GENERATE_RETRY would just re-hit the
                # same wedged/absent sovereign node) -- HALT the op cleanly rather
                # than letting it cascade or retry-storm.
                if "sovereign_route_sealed" in _err_msg:
                    logger.warning(
                        "[Orchestrator] ABSOLUTE ROUTE SEAL — op halted (sovereign "
                        "J-Prime committed + failed, DW cascade forbidden) [%s] %s",
                        ctx.op_id, _err_msg[:120],
                    )
                    ctx = ctx.advance(
                        OperationPhase.CANCELLED,
                        terminal_reason_code=f"sovereign_route_sealed:{_err_msg[:80]}",
                    )
                    await orch._record_ledger(
                        ctx, OperationState.FAILED,
                        {
                            "reason": "sovereign_route_sealed",
                            "error": _err_msg[:200],
                            "route": _route or "unknown",
                        },
                    )
                    return PhaseResult(
                        next_ctx=ctx, next_phase=None, status="fail",
                        reason=ctx.terminal_reason_code or "generate_terminal",
                        artifacts={"generation": generation, "episodic_memory": _episodic_memory},
                    )

                logger.warning(
                    "Generation attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    1 + orch._config.max_generate_retries,
                    ctx.op_id,
                    exc,
                )

                # ── Slice 12O — Foreground macro-cooldown ─────────
                # Before decrementing the retry counter or
                # transitioning to terminal, give a FOREGROUND op
                # one or more macro-retries with exponential
                # backoff IF the failure was provider-class
                # (terminal_structural / all_providers_exhausted
                # / stream_rupture). Pure decision; never raises.
                # Sleep is cancellation-aware (Phase 3) — Layer-2
                # graceful shutdown wakes the sleep immediately so
                # WAL drain can complete before Layer-3 SIGKILL.
                try:
                    _cooldown_decided = await _slice12o_maybe_cooldown(
                        orch=orch, ctx=ctx, exc=exc, route=_route,
                    )
                    if _cooldown_decided:
                        # Cooldown completed cleanly — re-enter the
                        # GENERATE attempt WITHOUT decrementing the
                        # in-window retry counter. This is the macro-
                        # retry layer ABOVE the per-window retries.
                        continue
                except asyncio.CancelledError:
                    # Phase 3 — graceful shutdown interrupted the
                    # cooldown sleep. Record a distinct terminal
                    # reason so operators can attribute the WAL
                    # drain to a cooperative cancellation rather
                    # than a wedge, then re-raise so the asyncio
                    # cancel cascade can complete.
                    try:
                        object.__setattr__(
                            ctx, "terminal_reason_code",
                            "cooldown_cancelled_shutdown",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    raise

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
                            await orch._stack.comm.emit_decision(
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
                            orch._cost_governor.start(
                                op_id=ctx.op_id,
                                route="standard",
                                complexity=getattr(ctx, "task_complexity", "") or "",
                                is_read_only=bool(getattr(ctx, "is_read_only", False)),
                            )
                        except Exception:
                            pass
                        # Guard the demotion call itself: if cumulative spend
                        # already blew past the new cap, skip the demotion.
                        if orch._cost_governor.is_exceeded(ctx.op_id):
                            logger.warning(
                                "[Orchestrator] Skipping STANDARD demotion — "
                                "cost cap already exceeded [%s]",
                                ctx.op_id,
                            )
                        else:
                            try:
                                _dem_deadline = datetime.now(tz=timezone.utc) + timedelta(seconds=220.0)
                                generation = await asyncio.wait_for(
                                    orch._generator.generate(ctx, _dem_deadline),
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
                                        orch._cost_governor.charge(
                                            ctx.op_id, _dem_cost, _dem_prov,
                                            phase=_dem_phase,
                                        )
                                        await orch._emit_route_cost_heartbeat(
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
                    await orch._record_ledger(
                        ctx,
                        OperationState.FAILED,
                        {"reason": "generation_failed", "error": str(exc)},
                    )
                    return PhaseResult(

                        next_ctx=ctx, next_phase=None, status="fail",

                        reason=ctx.terminal_reason_code or "generate_terminal",

                        artifacts={"generation": generation, "episodic_memory": _episodic_memory},

                    )
                # P2: Dynamic Re-Planning — suggest alternative strategy on failure
                try:
                    from backend.core.ouroboros.governance.self_evolution import DynamicRePlanner
                    _attempt_num = orch._config.max_generate_retries - generate_retries_remaining + 1
                    # THE VERDICT WAS ALWAYS REACHABLE — from `ctx`, not from
                    # a local. `validation` has no binding anywhere in
                    # `run()`; it was carried over from the inline
                    # orchestrator twin, where it IS bound at the VALIDATE
                    # seam. The `in dir()` guard therefore always evaluated
                    # False, and dynamic re-planning ran on empty failure
                    # context on every real op of the SHIPPING path.
                    #
                    # `validate_runner` publishes the verdict with
                    # `ctx.advance(GATE, validation=best_validation)`, and
                    # `advance()` uses `dataclasses.replace` — so every field
                    # not explicitly overridden carries forward, and a verdict
                    # set at VALIDATE survives into the next GENERATE attempt.
                    # That is exactly the "previous pass's verdict" this block
                    # wants, and it needed no new machinery to obtain.
                    #
                    # `replan_inputs` is TOTAL: a missing verdict (first
                    # attempt, or an evaluator that never produced one) yields
                    # ("", "") — the same honest no-evidence state this code
                    # already degraded to, now reached deliberately instead of
                    # by accident.
                    _fc, _em = _replan_inputs(getattr(ctx, "validation", None))
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

                # Retry: advance to GENERATE_RETRY with episodic memory context
                _retry_ctx_kwargs = {}

                # Inject direct error feedback so the model knows what went wrong
                _err_str = str(exc)

                # ── Iron Gate failures get targeted, in-flight instructions ──
                if _err_str.startswith("exploration_insufficient"):
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
                elif _err_str.startswith(_TARGET_MISSING_PREFIX):
                    # Gate 3.5 — target doesn't resolve in the write tree.
                    # Lane-correct steering (2026-07-22): benchmark ops get
                    # the third-party-repo text; host self-dev ops get
                    # path-hygiene steering (phantom parent / malformed
                    # prefix), never a wrong "you're outside the host" claim.
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
                _retry_ctx_kwargs["strategic_memory_prompt"] = _error_feedback

                # Record generation failure in episodic memory for downstream use
                if _episodic_memory is not None:
                    _gen_failure_class = "content"
                    if "exploration_insufficient" in _err_str:
                        _gen_failure_class = "exploration"
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

        # Slice 21 — Pipeline Supervisor Containment Boundary.
        #
        # Historical context: Slice 12AF Site 5 converted the bare
        # ``assert`` here into a structured ``RuntimeError`` so the
        # cascade surfaced a useful terminal_reason_code instead of
        # an uncaught ``AssertionError``. That was a step forward at
        # the time — but raising into the dispatcher path VIOLATES
        # the runner contract documented at ``phase_runner.py:103-104``:
        #
        #     "Never raise into the dispatcher path — catch
        #      exceptions, emit telemetry, and return
        #      PhaseResult(status='fail', ...)."
        #
        # v16 forensic (bt-2026-05-26-220930) showed the orchestrator
        # was already RESILIENT to the raise (a downstream handler
        # caught it and the BG worker correctly unregistered + picked
        # up the next op), but the failure mode produced repeated
        # traceback noise in debug.log and bypassed the structured
        # PhaseResult artifact channel that the dispatcher's
        # ``_fire_terminal_postmortem`` hook expects.
        #
        # Slice 21 brings the runner into compliance with its own
        # contract: return a structured ``PhaseResult(status='fail',
        # reason='generation_exhausted_unrepairable')`` with
        # ``next_phase=None`` (terminal). The dispatcher at
        # ``phase_dispatcher.py:1041`` recognizes ``next_phase is
        # None`` as terminal exit, fires the universal terminal
        # postmortem hook, and returns the terminated ctx to the
        # orchestrator. The orchestrator's BG worker loop archives
        # the op into session history, releases the worker lock, and
        # advances to the next queued task — all WITHOUT an
        # exception ever propagating through the dispatcher path.
        if generation is None:
            _exhaustion_reason = "generation_exhausted_unrepairable"
            logger.warning(
                "[GenerateRunner] Slice 21 supervisor containment: "
                "op=%s status=fail reason=%s — provider cascade "
                "exited without producing a candidate (see preceding "
                "EXHAUSTION events in debug.log for root cause)",
                ctx.op_id, _exhaustion_reason,
            )
            ctx = ctx.advance(
                OperationPhase.POSTMORTEM,
                terminal_reason_code=_exhaustion_reason,
            )
            return PhaseResult(
                next_ctx=ctx,
                # next_phase=None → dispatcher routes to terminal
                # postmortem hook (phase_dispatcher.py:1041+).
                next_phase=None,
                status="fail",
                reason=_exhaustion_reason,
                artifacts={
                    "generation_exhaustion": True,
                    "supervisor_containment_slice": "21",
                },
            )

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
                await orch._stack.ledger.append(_entry)
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
            # Anti-Venom S2 — noop + in-loop-write guard (live phase-runner
            # path). Mirror orchestrator.py: if Venom wrote files DURING
            # generation (edit_file/write_file landed on disk) and THEN the
            # model reports a no-op, those mutations never passed the
            # SemanticGuardian / GATE / risk-tier floor (the noop fast-path
            # skips APPLY). That is a guardian-bypass: code is on disk that no
            # gate ever saw. Fail-CLOSED by CANCELLING the op so the operator
            # sees a terminal failure (the on-disk writes remain for inspection
            # / next-op reconciliation) rather than a silent COMPLETE that hides
            # the unreviewed mutation.
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
                await orch._record_ledger(
                    ctx,
                    OperationState.FAILED,
                    {"reason": "noop_inloop_write_guard"},
                )
                return PhaseResult(
                    next_ctx=ctx, next_phase=None, status="fail",
                    reason="noop_inloop_write_guard",
                    artifacts={
                        "generation": generation,
                        "episodic_memory": _episodic_memory,
                    },
                )
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
                    await orch._stack.comm.emit_postmortem(
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
            await orch._record_ledger(
                ctx,
                OperationState.APPLIED,
                {
                    "reason": _terminal_reason,
                    "provider": generation.provider_name,
                },
            )
            return PhaseResult(

                next_ctx=ctx, next_phase=None, status="fail",

                reason=ctx.terminal_reason_code or "generate_terminal",

                artifacts={"generation": generation, "episodic_memory": _episodic_memory},

            )
        # ---- end verbatim transcription ----

        # Success path: generation produced candidates; advance to VALIDATE.
        # generation local is guaranteed non-None here (is_noop break or
        # candidates present). Thread it + episodic memory + retries count
        # via artifacts. VALIDATERunner reads generate_retries_remaining
        # for entropy computation (inline GENERATE mutated it across attempts).
        return PhaseResult(
            next_ctx=ctx,
            next_phase=OperationPhase.VALIDATE,
            status="ok",
            reason="generated",
            artifacts={
                "generation": generation,
                "episodic_memory": _episodic_memory,
                "generate_retries_remaining": generate_retries_remaining,
            },
        )


__all__ = ["GENERATERunner"]
