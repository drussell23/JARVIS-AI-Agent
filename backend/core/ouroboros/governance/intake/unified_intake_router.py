"""
Unified Intake Router — Phase 2C.1

Pipeline: schema_validate → normalize → dedup → priority_arbitration →
          rate_gate → conflict_detect → human_ack_gate →
          wal_enqueue → dispatch_queue

Dispatch loop runs as a background asyncio.Task.
File advisory lock prevents two router instances on the same project root.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import fcntl
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.core.ouroboros.governance.operation_id import generate_operation_id

from .intake_priority_queue import (
    IntakePriorityQueue,
    _intake_priority_scheduler_enabled,
)
from .intent_envelope import IntentEnvelope
from .intent_envelope import SOVEREIGN_SOURCES as _SOVEREIGN_SOURCES
from .wal import WAL, WALEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A1-T2 — event-driven router-ready valve (intake.router.ready)
# ---------------------------------------------------------------------------
# The roadmap-ignition daemon (GovernedLoopService) must never emit a
# strategic GOAL before the intake router is attached AND its dispatch loop
# is running, otherwise the envelope lands in a void (the silent-drop race
# A1 exists to kill). Readiness is a process-global signal because the
# IntakeLayerService (which marks ready) and the GLS daemon (which waits on
# it) are distinct services sharing one process + event loop.
#
# The signal has TWO surfaces, both required for a race-free handoff:
#   * ``_ROUTER_READY`` (threading.Event) — the checkable "already ready"
#     truth; safe to read synchronously from the async daemon. This is the
#     authoritative state even when no TrinityEventBus exists.
#   * ``EVENT_ROUTER_READY`` on the TrinityEventBus — the async wakeup so the
#     daemon doesn't have to poll. Best-effort: a missing bus degrades to a
#     bounded flag-poll, never to a blind sleep.
#
# ``await_router_ready`` implements subscribe-then-check: it subscribes to the
# topic FIRST, then checks the flag, so a "ready fired before I subscribed"
# cannot deadlock it. No sleep-poll on the happy (bus-present) path.
EVENT_ROUTER_READY = "intake.router.ready"

# Bus-absent degraded path: how often to re-check the readiness flag while
# bounded by the caller's timeout. Env-tunable; never a blind fixed settle.
_ENV_READY_POLL_S = "JARVIS_A1_ROUTER_READY_POLL_S"

_ROUTER_READY = threading.Event()


class RouterInitializationTimeoutError(RuntimeError):
    """Task 2 circuit breaker — the intake router did not signal readiness within
    the bounded window, so the roadmap daemon fails LOUD (and abandons emission)
    instead of silently DLQ-looping a goal into a never-ready void."""


def mark_router_ready() -> None:
    """Signal that the intake router is attached + its dispatch loop is live.

    Idempotent. Called by IntakeLayerService immediately after
    ``await router.start()`` succeeds. Sets the process-global flag that the
    roadmap daemon's readiness probe reads.
    """
    _ROUTER_READY.set()


def router_is_ready() -> bool:
    """Return True once :func:`mark_router_ready` has been called this process."""
    return _ROUTER_READY.is_set()


def _reset_router_ready_for_tests() -> None:
    """Test-only: clear the process-global readiness flag."""
    _ROUTER_READY.clear()


def _ready_poll_interval_s() -> float:
    """Bus-absent flag-poll cadence (env-tunable, fail-soft default 0.25s)."""
    try:
        val = float((os.environ.get(_ENV_READY_POLL_S, "") or "0.25").strip())
        return val if val > 0 else 0.25
    except (TypeError, ValueError):
        return 0.25


async def await_router_ready(bus: Any, timeout_s: float) -> bool:
    """Race-free wait for ``intake.router.ready``; never raises.

    Subscribe-then-check: subscribe to :data:`EVENT_ROUTER_READY` on *bus*
    FIRST, then check :func:`router_is_ready` — so a ready that fired before
    the subscription cannot deadlock the wait. Returns True if the router is
    ready within *timeout_s*, else False (the caller routes the stall to the
    DLQ rather than emitting into a void). No blind sleep on any path.

    Parameters
    ----------
    bus:
        A TrinityEventBus (or None). When None, falls back to a bounded
        flag-poll — still event-flagged + bounded, never a fixed settle.
    timeout_s:
        Maximum seconds to wait before returning False.
    """
    woke = asyncio.Event()
    if bus is not None:

        async def _on_ready(_ev: Any) -> None:
            woke.set()

        try:
            await bus.subscribe(EVENT_ROUTER_READY, _on_ready)
        except Exception as exc:  # noqa: BLE001 — bus is best-effort
            logger.debug("[A1] router-ready subscribe failed: %r", exc)
            bus = None  # degrade to flag-poll below

    # CHECK AFTER SUBSCRIBE — closes the ready-before-subscribe race.
    if router_is_ready():
        return True

    try:
        if bus is not None:
            await asyncio.wait_for(woke.wait(), timeout=timeout_s)
            return True
        # Degraded (no usable bus): bounded poll on the authoritative flag.
        interval = _ready_poll_interval_s()

        async def _until_ready() -> None:
            while not router_is_ready():
                await asyncio.sleep(interval)

        await asyncio.wait_for(_until_ready(), timeout=timeout_s)
        return True
    except asyncio.TimeoutError:
        # Re-check once in case readiness landed exactly at the deadline.
        return router_is_ready()


# ---------------------------------------------------------------------------
# A1-T3 — DAG-weight pre-flight tag
# ---------------------------------------------------------------------------
# A heavy multi-file strategic GOAL must route through the Epistemic Context
# Matrix prefetch (so its Venom exploration doesn't blow the generation
# deadline). The prefetch ALREADY recomputes heaviness at GENERATE — we do
# NOT duplicate it. This tag instead makes the intake-origin heaviness
# explicit + observable: it rides ``envelope.evidence`` (a mutable dict that
# the dispatch path already snapshots onto ``ctx.intake_evidence_json`` — the
# established "typed signal context" side-channel), so observers + the soak's
# breadcrumbs can see that intake classified the GOAL heavy. Pure, fail-soft,
# gated by the existing prefetch flag (no-op + byte-identical when off).
def stamp_dag_weight(envelope: Any) -> bool:
    """Stamp ``evidence["dag_weight"] = "heavy"`` on a heavy intake GOAL.

    Reuses :func:`epistemic_prefetch.is_heavy_goal` (multi-file OR high blast
    radius). Returns True iff the tag was stamped. No-op (returns False) when
    the prefetch flag is off, the GOAL is light, ``evidence`` is not a dict,
    or anything goes wrong. NEVER raises — intake must not fail on a tag.
    """
    try:
        from backend.core.ouroboros.governance.epistemic_prefetch import (
            is_heavy_goal,
            prefetch_enabled,
        )

        if not prefetch_enabled():
            return False
        target_files = getattr(envelope, "target_files", None)
        blast_radius = getattr(envelope, "blast_radius", 0)
        if not is_heavy_goal(target_files, blast_radius):
            return False
        evidence = getattr(envelope, "evidence", None)
        if not isinstance(evidence, dict):
            # Tag is best-effort observability; if there's no dict to ride,
            # the prefetch still recomputes heaviness independently.
            return False
        evidence["dag_weight"] = "heavy"
        logger.info(
            "[A1] dag_weight=heavy stamped goal=%s files=%d",
            getattr(envelope, "causal_id", "?"),
            len(target_files or ()),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — never fail intake on a tag
        logger.debug("[A1] dag_weight stamp skipped: %r", exc)
        return False


# ---------------------------------------------------------------------------
# F1 Slice 2 — shadow-mode flag for observational IntakePriorityQueue
# ---------------------------------------------------------------------------
# When ``JARVIS_INTAKE_PRIORITY_SCHEDULER_SHADOW=true`` AND the master flag
# is OFF, the router builds a parallel ``IntakePriorityQueue`` alongside the
# legacy ``asyncio.PriorityQueue``. Ingestion mirrors to both; dispatch
# dequeues from legacy but peeks at shadow to log a delta when the
# priority queue would have dequeued a different envelope. Enables live
# evidence-gathering without behavioral change.
#
# When the master flag (``JARVIS_INTAKE_PRIORITY_SCHEDULER_ENABLED``) is ON,
# shadow-mode is superseded — the priority queue becomes the source of
# truth for dequeue order, and legacy queue is drained as a tombstone.
def _intake_priority_scheduler_shadow_enabled() -> bool:
    """Re-read ``JARVIS_INTAKE_PRIORITY_SCHEDULER_SHADOW`` at call time.

    Shadow is inert when the master flag is on (primary mode dominates).
    Default off.
    """
    raw = (
        os.environ.get(
            "JARVIS_INTAKE_PRIORITY_SCHEDULER_SHADOW",
            "",
        )
        .strip()
        .lower()
    )
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Priority map — lower int = higher priority
# ---------------------------------------------------------------------------
_PRIORITY_MAP: Dict[str, int] = {
    "voice_human": 0,
    "test_failure": 1,
    "backlog": 2,
    # SWE-Bench-Pro evaluator envelopes (PRD §40.7.10-priority-map).  Tier
    # 2 (peer with `backlog`) is deliberate: this is queued benchmark
    # evaluation work — neither a runtime fire (would be `test_failure` @
    # 1) nor low-priority background fuzz (would land in the deferred
    # tier @ unmapped default 99).  Without this entry, the envelope
    # source falls through to `base = 99` and the `urgency="low"` default
    # subtracts another -1 → final priority 100, putting it strictly
    # behind every other in-flight signal.  Observed in stage-1 wiring
    # soak 2026-05-12 (session bt-2026-05-13-040242) as a 21-minute
    # dequeue lag while 16 other ops were dispatched ahead of ours.
    "swe_bench_pro": 2,
    "ai_miner": 3,
    "architecture": 3,
    "exploration": 4,
    "roadmap": 4,
    "capability_gap": 5,
    "cu_execution": 5,
    "runtime_health": 6,
}

# Documented technical debt: these sources in `_VALID_SOURCES` predate
# the `_PRIORITY_MAP`-completeness spine assertion and currently fall
# through to `base = 99` (starvation territory under any contended
# queue).  Each entry is a known prioritization gap — adding it to
# `_PRIORITY_MAP` with a deliberate tier closes the gap.  New sources
# MUST either join `_PRIORITY_MAP` directly OR opt into this set with
# an explicit comment justifying why the default is acceptable; the
# spine pin at
# `tests/governance/test_unified_intake_router_priority_map.py`
# catches drift.  Migration removes entries here as their tiers are
# assigned.
_PRIORITY_MAP_DEFERRED: frozenset = frozenset(
    {
        "auto_proposed",
        "cadence_synthetic",
        "cross_repo_drift",
        "doc_staleness",
        "github_issue",
        "intent_discovery",
        "meta_dormancy_alarm",
        "performance_regression",
        "security_advisory",
        "test_coverage",  # Slice 239 — decoupled background test-gen (deferred tier)
        "todo_scanner",
        "vision_sensor",
        "web_intelligence",
    }
)

# Urgency → priority boost (subtracted from base, so lower = higher priority)
_URGENCY_BOOST: Dict[str, int] = {
    "critical": 3,
    "high": 1,
    "normal": 0,
    "low": -1,
}

# Sources that bypass backpressure
_BACKPRESSURE_EXEMPT = frozenset({"voice_human", "test_failure"})


# ---------------------------------------------------------------------------
# Heap tie-break counter
# ---------------------------------------------------------------------------
#
# Every enqueue onto ``self._queue`` (asyncio.PriorityQueue) MUST carry a
# strictly-monotonic ``tie_seq`` so the heap NEVER falls through to
# IntentEnvelope comparison.  IntentEnvelope is a frozen dataclass with
# no ``__lt__``; when two items collide on the (priority, submitted_at)
# prefix, heapq's tuple comparison reaches the envelope and raises
# ``TypeError: '<' not supported between instances of 'IntentEnvelope'``.
# That exception bubbles up from inside ``await queue.put(...)``, leaves
# the heap in a partially-mutated state, and can quietly corrupt dequeue
# ordering — observed 2026-05-12 stage-1 wiring soak as a priority-2
# envelope sitting in the queue while priority-7+ envelopes dispatched
# ahead of it (session bt-2026-05-13-051420, swe_bench_pro envelope
# pending for 14+ min while 9 other ops drained).
#
# The fix is a totally-ordered prefix: heap items are
# ``(priority, submitted_at, tie_seq, envelope)``.  Heap compares only
# the first three fields (all numeric / never tie all at once because
# tie_seq is unique-per-enqueue).  ``envelope`` is inert for ordering.
# FIFO among genuine priority+timestamp ties is preserved via
# monotonic tie_seq.
#
# Discipline: ``next(_HEAP_TIE_SEQ)`` is the ONLY way to construct the
# tie-break field — both ingest-path enqueues and WAL-replay enqueues
# compose it.  AST-pinned in the spine.
_HEAP_TIE_SEQ: "itertools.count[int]" = itertools.count()

# ---------------------------------------------------------------------------
# Slice 5 Arc A — SensorGovernor consultation maps
# ---------------------------------------------------------------------------
# IntentEnvelope uses snake_case source strings (e.g. "test_failure") while
# the SensorGovernor seed registers CamelCase sensor names ("TestFailureSensor").
# Translate at the call site rather than renaming either side — both catalogs
# have existing test surface that would break on rename.
_SOURCE_TO_GOVERNOR_SENSOR: Dict[str, str] = {
    "test_failure": "TestFailureSensor",
    "backlog": "BacklogSensor",
    "voice_human": "VoiceCommandSensor",
    "ai_miner": "OpportunityMinerSensor",
    "capability_gap": "CapabilityGapSensor",
    "runtime_health": "RuntimeHealthSensor",
    "exploration": "ProactiveExplorationSensor",
    "intent_discovery": "IntentDiscoverySensor",
    "todo_scanner": "TodoScannerSensor",
    "doc_staleness": "DocStalenessSensor",
    "github_issue": "GitHubIssueSensor",
    "performance_regression": "PerformanceRegressionSensor",
    "cross_repo_drift": "CrossRepoDriftSensor",
    "web_intelligence": "WebIntelligenceSensor",
    "vision_sensor": "VisionSensor",
    # Unmapped (no governor spec): architecture, roadmap, cu_execution,
    # security_advisory — fall through to "governor.unregistered_sensor"
    # which always allows (safe default).
}

# Envelope urgency → Governor urgency. Envelope uses 4 values; Governor has 5
# (adds SPECULATIVE which isn't currently produced by sensors).
_URGENCY_STR_TO_GOVERNOR: Dict[str, str] = {
    "critical": "immediate",  # 2.0x multiplier
    "high": "standard",  # 1.0x multiplier
    "normal": "complex",  # 0.8x multiplier
    "low": "background",  # 0.5x multiplier
}


def _intake_governor_mode() -> str:
    """Shadow / enforce / off — default shadow for Slice 5 Arc A first drop.

    * ``off``      — skip governor consultation entirely (pre-Arc-A behavior)
    * ``shadow``   — consult + log/SSE any would-be denials, allow through
    * ``enforce``  — honor the decision; deny returns ``governor_throttled``
    """
    raw = os.environ.get("JARVIS_INTAKE_GOVERNOR_MODE", "shadow").strip().lower()
    if raw in ("off", "shadow", "enforce"):
        return raw
    return "shadow"


# Slice 101 Phase 4 — only the lowest-urgency (deferrable background /
# speculative) signals are eligible for cognitive load-shedding. "low" maps to
# the background provider route (0.5x). critical/high/normal are NEVER shed.
_SHEDDABLE_URGENCIES = frozenset({"low"})


def _intake_cognitive_shed_mode() -> str:
    """off / shadow / enforce — Slice 101 Phase 4 cognitive load-shed gate.

    Mirrors :func:`_intake_governor_mode`. Default ``shadow``: the
    ``cognitive_load_shedding`` substrate master
    (``JARVIS_COGNITIVE_LOAD_SHEDDING_ENABLED``, §33.1 default-FALSE) returns a
    DISABLED verdict until explicitly enabled, so ``shadow`` is inert by
    default. ``enforce`` sheds ONLY the lowest-urgency deferrable signals.
    """
    raw = (
        os.environ.get(
            "JARVIS_INTAKE_COGNITIVE_SHED_MODE",
            "shadow",
        )
        .strip()
        .lower()
    )
    if raw in ("off", "shadow", "enforce"):
        return raw
    return "shadow"


def _intake_backpressure_shed_fraction() -> float:
    """Slice 23 — live intake-queue-pressure fraction at/above which a burst
    escalates cognitive shedding regardless of the (lagging) host-stress
    forecast. Fraction of ``backpressure_threshold``; env-driven, no hardcoded
    depth. Default 0.75. Clamped to (0, 1]; out-of-range → default. A 72-signal
    burst fills the queue faster than host stress rises, so this real-time
    signal is what makes the shed BURST-responsive (mandate 2). NEVER raises."""
    try:
        raw = os.environ.get(
            "JARVIS_INTAKE_BACKPRESSURE_SHED_FRACTION", "",
        ).strip()
        v = float(raw) if raw else 0.75
        return v if 0.0 < v <= 1.0 else 0.75
    except (TypeError, ValueError):
        return 0.75


def _allow_log_mode() -> str:
    """Follow-up #1 — visibility for "governor allowed this op" decisions.

    * ``off``      — silent (default; preserves pre-follow-up quiet behavior)
    * ``summary``  — emit one structured INFO rollup line every N allows
                     (N from ``JARVIS_INTAKE_GOVERNOR_ALLOW_LOG_INTERVAL``,
                     default 100). Per-sensor counts + total included.
    * ``debug``    — DEBUG-level per-allow line (opt-in verbose)

    Operator constraint (binding from Slice 5 closure policy): default
    INFO noise is unacceptable. Default is ``off``; ``summary`` rate-limits
    to 1/N; ``debug`` requires explicit verbose opt-in.
    """
    raw = (
        os.environ.get(
            "JARVIS_INTAKE_GOVERNOR_ALLOW_LOG",
            "off",
        )
        .strip()
        .lower()
    )
    if raw in ("off", "summary", "debug"):
        return raw
    return "off"


def _allow_log_interval() -> int:
    """Allow-count threshold between summary rollup log lines. Default 100.

    Clamped to [1, 10000]. Lower = more frequent rollups (noisier);
    higher = longer aggregation window (less signal per line)."""
    raw = os.environ.get("JARVIS_INTAKE_GOVERNOR_ALLOW_LOG_INTERVAL", "100")
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 100
    return max(1, min(10000, v))


# P2.4: Module-level GoalTracker reference.  Set by UnifiedIntakeRouter on
# init so _compute_priority can apply goal-alignment boost without changing
# the function signature at every call site.
_active_goal_tracker: Optional[Any] = None

# Counters for the goal-alignment fault path — visible failure accounting
# replaces the old silent `except: pass`. The first failure logs at WARNING
# with full exc_info so operators notice; subsequent failures log at DEBUG
# so a broken tracker doesn't flood the logs. Aggregate counters are
# exposed via ``goal_alignment_failure_stats()`` for health endpoints.
_goal_alignment_failures: int = 0
_goal_alignment_warned: bool = False


def goal_alignment_failure_stats() -> Dict[str, int]:
    """Return cumulative failure counts for the goal-alignment scorer.

    Callers use this to surface broken strategic-direction integration on
    health dashboards or battle-test postmortems without parsing logs.
    """
    return {"failures": _goal_alignment_failures}


def _ingest_stratification_enabled() -> bool:
    """Slice 49 master switch (default ON). Off → byte-identical pre-Slice-49
    priority (no penalty, no file reads in the hot intake path)."""
    return os.environ.get(
        "JARVIS_INGEST_STRATIFICATION_ENABLED",
        "true",
    ).strip().lower() not in ("false", "0", "no", "off")


def _is_test_generation_intent(envelope: "IntentEnvelope") -> bool:
    """Slice 49 escape hatch — True when the op's intent is adding test
    coverage, so the blast-radius penalty is suppressed (lets the organism
    target and heal its own large uncovered core modules). Conservative:
    matches explicit coverage/test-generation phrasing or source tags."""
    src = (getattr(envelope, "source", "") or "").lower()
    if "coverage" in src or "test_gen" in src or "testgen" in src:
        return True
    desc = (getattr(envelope, "description", "") or "").lower()
    return (
        "add test" in desc
        or "test coverage" in desc
        or "coverage for" in desc
        or "write tests" in desc
    )


def _resurrection_primacy_margin() -> int:
    """Slice 245 — margin below the highest normal primacy for a resurrected op.
    Env-tunable, NEVER raises (floors at 1)."""
    try:
        v = int(float(os.getenv("JARVIS_RESURRECTION_PRIMACY_MARGIN", "").strip() or 100))
        return v if v >= 1 else 100
    except (TypeError, ValueError):
        return 100


def _resurrection_intake_priority() -> int:
    """Absolute-max intake primacy — dynamically below the highest normal source
    (NOT a hardcoded 0). Derived from the live _PRIORITY_MAP so it stays correct
    if the tiers change."""
    try:
        floor = min(_PRIORITY_MAP.values())
    except (ValueError, TypeError):
        floor = 0
    return floor - _resurrection_primacy_margin()


def _sovereign_primacy_margin() -> int:
    """Slice 246 — margin by which a human-origin intent outranks resurrection.
    Env-tunable, NEVER raises (floors at 1)."""
    try:
        v = int(float(os.getenv("JARVIS_SOVEREIGN_PRIMACY_MARGIN", "").strip() or 100))
        return v if v >= 1 else 100
    except (TypeError, ValueError):
        return 100


def _sovereign_human_priority() -> int:
    """Sovereign human primacy — strictly below resurrection (Human > Resurrected
    > Normal), so a live human intent can never be starved by an autonomous
    survivor. Dynamic; the host always keeps ultimate control."""
    return _resurrection_intake_priority() - _sovereign_primacy_margin()


# ---------------------------------------------------------------------------
# P0.1 — value-derived priority re-anchoring (kills the source-only queue)
# ---------------------------------------------------------------------------
#
# ROOT CAUSE of "O+V can't see real work": the queue's dominant term is
# ``base = _PRIORITY_MAP.get(envelope.source, 99)`` — priority derived from
# the SOURCE LABEL only. A substantive-but-deferred source (security_advisory
# CVE, performance_regression, github_issue, intent_discovery) sits at base 99
# and is buried ~90 points behind an ``ai_miner`` trivia signal at base 3; the
# appetite layer that already KNOWS the difference (signal_value, Slice 15) is
# consulted at ROUTE/GATE but never here. This composes that SAME band into
# the base with the SAME escalate/clamp semantics the ROUTE layer applies:
#
#   * BAND_ORACLE (a resolved failing-test attribution — a machine-checkable
#     real defect) OR an explicit high/critical urgency (the source DECLARED
#     it substantive — e.g. a critical CVE, whose requirements.txt change is
#     cosmetic-SHAPE but critical WORK) → ESCALATE the base up to the
#     "queued real work" anchor, un-starving it. min() only pulls up; it
#     never demotes an already-high signal.
#   * BAND_COSMETIC_CLASS (targets are only requirements/cfg/ini or a
#     docstring-only module — the Run-22 noise class) with NO explicit
#     urgency → CLAMP the base down to the deferred floor so trivia can
#     never outrank substance. max() only pushes down; never promotes.
#   * BAND_EXECUTABLE / BAND_INDETERMINATE → the source tier stands (the
#     band cannot tell real work from churn on a live module — that
#     discrimination is the sensor's job, P0.3 — so we defer to the
#     class-urgency the source encodes).
#
# No new hardcoded tier: both anchors are the map's OWN landmarks
# (``backlog`` = queued real work; the deferred base 99), env-tunable.
# Shadow-first rollout (mirrors mcp_output_scanner, Slice 18 #6): master
# default-TRUE COMPUTES the band + stashes it to evidence for the P0.5 soak;
# shadow default-TRUE means it is NOT yet applied (queue order byte-identical
# to today). Graduation = flip shadow off → the re-anchoring becomes
# authoritative. Composes the global appetite master so a repo-wide
# ``JARVIS_SIGNAL_VALUE_ROUTING_ENABLED=false`` disables it here too (DRY).

_VALUE_PRIORITY_MASTER_ENV = "JARVIS_INTAKE_VALUE_PRIORITY_ENABLED"
_VALUE_PRIORITY_SHADOW_ENV = "JARVIS_INTAKE_VALUE_PRIORITY_SHADOW"
_VALUE_SUBSTANTIVE_ANCHOR_ENV = "JARVIS_INTAKE_VALUE_SUBSTANTIVE_ANCHOR"
_VALUE_COSMETIC_FLOOR_ENV = "JARVIS_INTAKE_VALUE_COSMETIC_FLOOR"
_VALUE_URGENT_URGENCIES = frozenset({"high", "critical"})


def _value_priority_master_enabled() -> bool:
    """Compute-and-observe gate. Default TRUE, but composes the global
    appetite master so the whole layer honors one repo-wide off switch."""
    if os.environ.get(
        _VALUE_PRIORITY_MASTER_ENV, "true",
    ).strip().lower() not in ("1", "true", "yes", "on"):
        return False
    try:
        from backend.core.ouroboros.governance.signal_value import (
            signal_value_routing_enabled,
        )
        return signal_value_routing_enabled()
    except Exception:  # noqa: BLE001 — appetite layer unavailable → observe off
        return False


def _value_priority_shadow_enabled() -> bool:
    """Shadow (compute, don't apply).

    GRADUATED 2026-09-06 — default flipped to ENFORCE (shadow OFF): substance,
    not source VOLUME, now sets queue order. This is the root fix for the
    autonomous-work-selection pathology where documentation churn (the highest-
    VOLUME source) buried genuinely valuable work — a signal's value band, once
    it correctly reflects work-nature (see ``signal_value.score_signal``), sinks
    cosmetic work to the starvation floor and floats an oracle-band failing-test
    op to the front. Re-anchoring is PURE queue re-ordering — never fed to Iron
    Gate / risk tier / policy / approval, and never drops a signal — so enforcing
    changes only WHICH ready op runs first. Re-shadow explicitly with
    ``JARVIS_INTAKE_VALUE_PRIORITY_SHADOW=true``; the master
    (``JARVIS_INTAKE_VALUE_PRIORITY_ENABLED=false``) still disables the layer
    entirely."""
    return os.environ.get(
        _VALUE_PRIORITY_SHADOW_ENV, "false",
    ).strip().lower() not in ("0", "false", "no", "off")


def _value_priority_enforce() -> bool:
    """Re-anchoring is authoritative only when the master is on AND shadow is
    explicitly off."""
    return _value_priority_master_enabled() and not _value_priority_shadow_enabled()


def _value_substantive_anchor() -> int:
    """The tier a proven/declared-substantive signal escalates TO. Default =
    the map's own ``backlog`` (queued-real-work) landmark, NOT a magic number.
    Env-tunable; floored at 0 (can't out-rank sovereign/resurrection primacy,
    which short-circuit before this)."""
    raw = os.environ.get(_VALUE_SUBSTANTIVE_ANCHOR_ENV, "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return _PRIORITY_MAP.get("backlog", 2)


def _value_cosmetic_floor() -> int:
    """The tier un-urgent cosmetic trivia is clamped DOWN to. Default = the
    deferred-source base (the map's own starvation floor). Env-tunable."""
    raw = os.environ.get(_VALUE_COSMETIC_FLOOR_ENV, "").strip()
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return 99


def _reanchor_base_on_value(
    envelope: "IntentEnvelope", base: int, repo_root: "Optional[Path]",
) -> Tuple[int, Dict[str, Any]]:
    """Compose the signal_value band into the source-tier ``base`` so SUBSTANCE
    sets queue priority. Returns ``(proposed_base, evidence_delta)``. Pure
    re-ordering — never fed to Iron Gate / risk tier / policy / approval, and
    never drops a signal. Fail-soft: any scoring fault → the source tier stands.
    """
    if not _value_priority_master_enabled():
        return base, {}
    try:
        import json as _vj
        from backend.core.ouroboros.governance.signal_value import (
            BAND_COSMETIC_CLASS,
            BAND_ORACLE,
            score_signal,
        )
        _ev_json = (
            _vj.dumps(envelope.evidence)
            if isinstance(getattr(envelope, "evidence", None), dict)
            else ""
        )
        band = score_signal(
            str(getattr(envelope, "source", "") or ""),
            envelope.target_files or (),
            _ev_json,
            repo_root,
        )
    except Exception:  # noqa: BLE001 — substance scoring down → source tier stands
        return base, {}

    urgent = str(getattr(envelope, "urgency", "")).strip().lower() in (
        _VALUE_URGENT_URGENCIES
    )
    if band >= BAND_ORACLE or urgent:
        proposed = min(base, _value_substantive_anchor())  # escalate; never demote
        reason = "oracle" if band >= BAND_ORACLE else "explicit_urgency"
    elif band <= BAND_COSMETIC_CLASS:
        proposed = max(base, _value_cosmetic_floor())  # clamp trivia; never promote
        reason = "cosmetic_clamp"
    else:
        proposed = base  # executable / indeterminate → source tier stands
        reason = "source_tier"

    delta: Dict[str, Any] = {
        "value_band": int(band),
        "value_priority_reason": reason,
        "value_source_base": int(base),
        "value_reanchored_base": int(proposed),
        "value_priority_enforced": bool(_value_priority_enforce()),
    }
    return proposed, delta


# ---------------------------------------------------------------------------
# WAL Signal Liveness Tombstoning (2026-07-22)
# ---------------------------------------------------------------------------

_WAL_TARGET_LIVENESS_ENV = "JARVIS_WAL_TARGET_LIVENESS_ENABLED"


def _wal_target_liveness_enabled() -> bool:
    """Master flag — default TRUE. Falsey restores unconditional replay
    (ghost signals wake the orchestrator as before)."""
    raw = os.environ.get(_WAL_TARGET_LIVENESS_ENV, "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _all_targets_missing(
    target_files: Any, observation_root: Any,
) -> bool:
    """True iff the signal is TARGETED (>=1 target file) AND every target
    is absent from the observation root.

    Conservative by design: an untargeted signal (empty/None — e.g.
    SWE-bench envelopes carry no target_files by contract) is NEVER a
    ghost; one live target keeps a multi-target signal actionable; any
    path/stat error counts the target as PRESENT (never lose a real
    signal to an I/O hiccup). Sync — callers dispatch via
    ``asyncio.to_thread``. NEVER raises.
    """
    try:
        targets = [str(t).strip() for t in (target_files or ()) if str(t).strip()]
        if not targets:
            return False
        root = Path(observation_root)
        for rel in targets:
            try:
                p = Path(rel)
                candidate = p if p.is_absolute() else root / rel
                if candidate.exists():
                    return False
            except OSError:
                return False  # unknowable → treat as present
        return True
    except Exception:  # noqa: BLE001 — fail-soft: replay normally
        return False


def _compute_priority(
    envelope: "IntentEnvelope",
    dependency_credit: int = 0,
    *,
    repo_root: "Optional[Path]" = None,
    resurrected: bool = False,
    semantic_boost_override: "Optional[int]" = None,
    stratification_penalty_override: "Optional[int]" = None,
) -> Tuple[int, Optional[Any]]:
    """Compute cost-aware priority score + rich goal alignment for an envelope.

    Returns ``(priority_int, goal_alignment_or_none)``. Lower int = higher
    priority. ``goal_alignment`` is a :class:`GoalAlignment` when a tracker
    is installed and the scorer ran successfully (even on a no-match), and
    ``None`` when no tracker is present or the scorer raised — callers can
    branch on ``is None`` to tell "scoring didn't run" apart from
    "scoring ran and found nothing".

    Factors:
    1. Base priority from source type
    2. Urgency boost (critical/high get promoted)
    3. Cost-awareness: operations touching many files are penalized
       (they consume more generation tokens for less focused impact)
    4. Dependency credit: ops that unblock queued signals get priority boost
    5. Goal alignment: signals that match active user goals get boosted —
       boost magnitude now scales with raw relevance score, not just
       match/no-match, so a signal that hits three goals wins over one
       that hits a single low-confidence goal at the same source tier.

    Fault isolation: a broken or misconfigured GoalTracker MUST NOT break
    the intake router. Exceptions are logged (warn-once, debug-after) and
    counted so operators can tell "goal scoring is down" from a status
    endpoint rather than wondering why prioritization looks flat.
    """
    global _goal_alignment_failures, _goal_alignment_warned

    # Slice 246 — sovereign human primacy. A direct human-origin intent outranks
    # EVERYTHING, including a resurrected survivor, so the autonomous organism can
    # never starve a live human. Checked first; the host keeps ultimate control.
    if getattr(envelope, "source", "") in _SOVEREIGN_SOURCES:
        return _sovereign_human_priority(), None

    # Slice 245 — a hibernation survivor gets Absolute-Max Primacy, short-
    # circuiting all normal factors so nothing can demote it below the work that
    # accumulated during the dark window. Dynamic floor (not a hardcoded 0).
    if resurrected:
        return _resurrection_intake_priority(), None

    base = _PRIORITY_MAP.get(envelope.source, 99)
    # P0.1 — value-derived re-anchoring. Compose the appetite band (Slice 15)
    # into the source-tier base so SUBSTANCE, not source label, sets queue
    # order. Always computed + stashed to evidence when the master is on (the
    # P0.5 soak reads it); only APPLIED when shadow is off (enforce). Fail-soft
    # — a scoring fault leaves the source tier untouched.
    _reanchored_base, _value_delta = _reanchor_base_on_value(
        envelope, base, repo_root,
    )
    if _value_delta and isinstance(envelope.evidence, dict):
        try:
            envelope.evidence.update(_value_delta)
        except Exception:  # noqa: BLE001 — observability must never break intake
            pass
    if _value_priority_enforce():
        base = _reanchored_base
    urgency = _URGENCY_BOOST.get(envelope.urgency, 0)
    # Cost penalty: 0 for 1 file, 1 for 2-4 files, 2 for 5+ files
    file_count = len(envelope.target_files) if envelope.target_files else 1
    cost_penalty = 0 if file_count <= 1 else (1 if file_count <= 4 else 2)
    # Confidence discount: high-confidence signals get slight priority
    confidence_bonus = 1 if envelope.confidence >= 0.9 else 0
    # Dependency credit: ops that would unblock queued signals get boosted
    # Capped at 3 to prevent runaway priority from large queues
    dep_bonus = min(dependency_credit, 3)

    # P2.4 / Item 3: Goal alignment — visible failure, rich result.
    goal_boost = 0
    alignment: Optional[Any] = None
    if _active_goal_tracker is not None:
        try:
            alignment = _active_goal_tracker.alignment_context(
                envelope.description,
                envelope.target_files,
            )
            goal_boost = int(getattr(alignment, "boost", 0) or 0)
        except Exception as exc:
            _goal_alignment_failures += 1
            if not _goal_alignment_warned:
                _goal_alignment_warned = True
                logger.warning(
                    "[Router] goal alignment scorer failed (first occurrence, "
                    "subsequent failures will log at DEBUG): %s",
                    exc,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "[Router] goal alignment scorer failed (total=%d): %s",
                    _goal_alignment_failures,
                    exc,
                )

    # SemanticIndex v0.1: soft semantic prior capped at BOOST_MAX (default 1)
    # so it remains strictly subordinate to goal_alignment_boost (=2).
    # Master off → no import, no disk I/O, boost=0. Performance: one embed
    # + one cosine against a precomputed centroid per signal (beef #4).
    # Authority invariant: this boost ONLY affects priority ordering — it
    # is NEVER fed into UrgencyRouter, Iron Gate, risk tier, policy engine,
    # FORBIDDEN_PATH, or approval gating.
    semantic_boost = 0
    if semantic_boost_override is not None:
        # Tier-2b — the caller (``_ingest_impl``) already computed the
        # semantic boost OFF the asyncio loop via
        # ``SemanticIndex.boost_and_score_offloaded`` (a single fastembed
        # encode dispatched through ``cooperative_fs_io.offload``). Consume
        # its result here; do NOT run the inline on-loop embed. The evidence
        # stash (``semantic_alignment`` / ``semantic_boost``) is done by the
        # caller alongside the offloaded score.
        semantic_boost = max(0, int(semantic_boost_override))
    else:
        try:
            from backend.core.ouroboros.governance.semantic_index import (
                get_default_index,
            )

            _si = get_default_index()
            # Q3 Slice 3 — non-blocking build trigger. The hot intake path
            # must not stall on git-log subprocesses + corpus assembly +
            # bulk-embed. ``build_async`` returns immediately; ``boost_for``
            # below scores against whichever centroid is currently loaded
            # (empty on cold start → boost=0, no harm done).
            _si.build_async()
            semantic_boost = _si.boost_for(envelope.description or "")
            if semantic_boost > 0 or _si.stats().built_at > 0:
                # Stash in envelope evidence for observability. Score itself
                # (the raw cosine) is useful for operators inspecting "why
                # did this signal get boosted?" without exposing raw vectors.
                try:
                    _sim_raw = _si.score(envelope.description or "")
                    if isinstance(envelope.evidence, dict):
                        envelope.evidence["semantic_alignment"] = round(float(_sim_raw), 4)
                        envelope.evidence["semantic_boost"] = int(semantic_boost)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("[Router] semantic alignment scorer failed: %s", exc)

    # MissionInferrer Slice B — soft inferred-direction priority boost.
    # Reads the cached InferenceResult via get_current() (cache-only, no
    # rebuild trigger from intake — CONTEXT_EXPANSION already triggers
    # build() per-op, keeping the cache fresh). Cap is enforced inside
    # priority_boost_for_signal via priority_boost_max() env knob,
    # default 0.5, hard ceiling 1.0 — strictly below declared-goal boost.
    #
    # Authority invariant: priority ordering ONLY. NEVER fed to
    # UrgencyRouter, Iron Gate, risk tier, policy engine, FORBIDDEN_PATH,
    # or approval gating. Cost-contract: cannot escalate BG/SPEC routes
    # (route is a separate decision in UrgencyRouter, untouched here).
    inferred_direction_boost = 0
    try:
        from backend.core.ouroboros.governance.goal_inference import (
            get_default_engine,
            inference_enabled,
            priority_boost_for_signal,
        )

        if inference_enabled():
            _engine = get_default_engine()
            if _engine is not None:
                _cached = _engine.get_current()
                if _cached is not None and _cached.inferred:
                    _raw = priority_boost_for_signal(
                        signal_description=envelope.description or "",
                        signal_target_files=envelope.target_files or (),
                        result=_cached,
                    )
                    # int projection: any positive raw boost lands at
                    # least one priority point. Banker's rounding would
                    # silently drop the natural single-match score
                    # (0.5) to 0, neutralizing the wire-up. ceil()
                    # preserves the discrimination signal; the float
                    # cap (priority_boost_max) still bounds the raw
                    # value above so operators set the FLOAT ceiling
                    # and the INT projection follows naturally.
                    import math as _math

                    inferred_direction_boost = int(_math.ceil(_raw)) if _raw > 0.0 else 0
                    if inferred_direction_boost > 0 and isinstance(
                        envelope.evidence,
                        dict,
                    ):
                        envelope.evidence["inferred_direction_boost"] = inferred_direction_boost
                        envelope.evidence["inferred_direction_raw"] = round(float(_raw), 3)
    except Exception as exc:  # noqa: BLE001 -- defensive fail-soft
        logger.debug(
            "[Router] inferred-direction boost skipped: %s",
            exc,
        )

    # Slice 49 — universal ingestion stratification (SOFT). Add a bounded
    # penalty for ops targeting large uncovered files so they are processed
    # last fleet-wide (across -miner AND -cau tracks). This ONLY reorders the
    # queue; OperationAdvisor.advise() remains the hard blast-radius gate, and
    # the penalty is fully reachable (no drop) with a test-generation escape.
    # Authority invariant: priority ordering ONLY — never fed to UrgencyRouter,
    # Iron Gate, risk tier, policy engine, FORBIDDEN_PATH, or approval gating.
    stratification_penalty = 0
    if stratification_penalty_override is not None:
        # Tier-4 — the caller (``_ingest_impl``) already computed (or
        # deliberately degraded) the stratification penalty OFF the asyncio
        # loop. The inline path below runs file_has_test_coverage's
        # tests-tree rglob + cold AST-map build synchronously — the traced
        # 41.7 s on-loop block (bt-iso-1783102490) — so the hot intake path
        # must NEVER reach it. Evidence stash is done by the caller.
        stratification_penalty = max(0, int(stratification_penalty_override))
    elif repo_root is not None and _ingest_stratification_enabled():
        try:
            from backend.core.ouroboros.governance.target_stratification import (
                ingest_priority_penalty,
            )

            stratification_penalty = ingest_priority_penalty(
                envelope.target_files or (),
                repo_root,
                suppress=_is_test_generation_intent(envelope),
            )
            if stratification_penalty > 0 and isinstance(envelope.evidence, dict):
                envelope.evidence["stratification_penalty"] = stratification_penalty
        except Exception as exc:  # noqa: BLE001 -- fail-soft, never break intake
            logger.debug("[Router] ingest stratification skipped: %s", exc)

    # P0.4 — reputation bias (the learning plane's READ side). A signal
    # targeting a historically-substantive file (high fragility = frequent
    # failures / churn / blast, learned from prior op outcomes) earns a bounded
    # priority boost, so O+V adaptively prioritizes WHERE the real work has
    # lived. Read-only, capped, fail-soft; requires BOTH reputation gates
    # (write master + bias sub-gate) so it is inert until learning accumulated.
    # Authority invariant: priority ordering ONLY — never fed to UrgencyRouter,
    # Iron Gate, risk tier, policy engine, FORBIDDEN_PATH, or approval gating.
    reputation_boost = 0
    if repo_root is not None:
        try:
            from backend.core.ouroboros.consciousness.memory_engine import (
                get_default_memory_engine,
                reputation_bias_enabled,
                reputation_boost_max,
            )

            if reputation_bias_enabled():
                _eng = get_default_memory_engine(repo_root)
                _frag = 0.0
                for _f in (envelope.target_files or ()):
                    _r = _eng.get_file_reputation(str(_f)).fragility_score
                    if _r > _frag:
                        _frag = _r
                reputation_boost = int(round(_frag * reputation_boost_max()))
                if reputation_boost > 0 and isinstance(envelope.evidence, dict):
                    envelope.evidence["reputation_fragility"] = round(_frag, 3)
                    envelope.evidence["reputation_boost"] = reputation_boost
        except Exception as exc:  # noqa: BLE001 — learning never breaks intake
            logger.debug("[Router] reputation bias skipped: %s", exc)

    priority = (
        base
        - urgency
        + cost_penalty
        - confidence_bonus
        - dep_bonus
        - goal_boost
        - semantic_boost
        - inferred_direction_boost
        - reputation_boost
        + stratification_penalty
    )
    return priority, alignment


#: Size at which `_prune_dedup` sweeps expired entries. Not a cap — the dict
#: may legitimately exceed it between sweeps; it is the point where an O(n)
#: pass is cheap relative to the leak it prevents.
_DEDUP_PRUNE_THRESHOLD: int = 512


@dataclass(frozen=True)
class DedupCollision:
    """Why an envelope deduplicated, in facts the registry already holds.

    Deliberately does NOT claim the colliding op is still running. The
    registry stores one thing — the monotonic time a dedup_key was last
    accepted — so an op that has since COMPLETED or FAILED still collides
    inside the window. "Already in flight" would be a state nothing here
    measures; `age_s` and `retry_after_s` are measured, and a renderer can
    say something true from them.
    """

    dedup_key: str
    age_s: float
    window_s: float

    @property
    def short_hash(self) -> str:
        """The distinguishing head of the sha256. The key is already a
        digest, so its PREFIX distinguishes (unlike a UUIDv7 op id, whose
        tail does)."""
        return str(self.dedup_key)[:12]

    @property
    def retry_after_s(self) -> float:
        """Seconds until this key is accepted again. Never negative."""
        return max(0.0, float(self.window_s) - float(self.age_s))


@dataclass(frozen=True)
class IntakeRouterConfig:
    project_root: Path
    wal_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    max_retries: int = 3
    backpressure_threshold: int = 10
    dedup_window_s: float = 600.0
    voice_dedup_window_s: float = 300.0
    max_queue_size: int = 100
    dispatch_timeout_s: float = 300.0

    @property
    def _state_root(self) -> Path:
        """Writable root for intake state (.jarvis WAL/lock/...).

        Virtualizes the storage boundary so the code never depends on a
        hardcoded, possibly-unprivileged absolute path. ``JARVIS_TRINITY_ROOT``
        (when set) is the authoritative writable root -- the isomorphic local
        soak injects a temp dir there so the literal ``/opt/trinity`` production
        path is never required off the node. Unset => ``project_root`` (prod is
        byte-identical: nothing reads the env var on the real node).
        """
        env_root = os.environ.get("JARVIS_TRINITY_ROOT", "").strip()
        return Path(env_root) if env_root else self.project_root

    @property
    def resolved_wal_path(self) -> Path:
        return self.wal_path or (self._state_root / ".jarvis" / "intake_wal.jsonl")

    @property
    def resolved_lock_path(self) -> Path:
        return self.lock_path or (self._state_root / ".jarvis" / "intake_router.lock")


class PendingAckStore:
    def __init__(self) -> None:
        self._store: Dict[str, IntentEnvelope] = {}

    def park(self, envelope: IntentEnvelope) -> None:
        self._store[envelope.idempotency_key] = envelope

    def acknowledge(self, idempotency_key: str) -> Optional[IntentEnvelope]:
        return self._store.pop(idempotency_key, None)

    def count(self) -> int:
        return len(self._store)


class RouterAlreadyRunningError(RuntimeError):
    pass


class UnifiedIntakeRouter:
    """Central routing hub for all Ouroboros intake signals.

    Implements an async, priority-ordered dispatch pipeline with:
    - Deduplication within configurable time windows
    - Human acknowledgement gating
    - Backpressure signalling for low-priority sources
    - Write-ahead log for crash recovery
    - Advisory file lock preventing duplicate router instances
    - Dead-letter queue after max retries are exhausted
    """

    def __init__(
        self, gls: Any, config: IntakeRouterConfig, runtime_orchestrator: Any = None
    ) -> None:
        global _active_goal_tracker
        self._gls = gls
        self._runtime_orchestrator = runtime_orchestrator
        self._config = config
        self._wal = WAL(config.resolved_wal_path)
        # Heap tuple shape: (priority, submitted_at, tie_seq, envelope).
        # The tie_seq third element is REQUIRED so heapq never falls
        # through to envelope comparison — see _HEAP_TIE_SEQ module
        # docstring + the dedicated regression spine at
        # tests/governance/intake/test_unified_intake_router_heap_tiebreak.py
        self._queue: asyncio.PriorityQueue[Tuple[int, float, int, IntentEnvelope]] = (
            asyncio.PriorityQueue(maxsize=config.max_queue_size)
        )
        self._dedup: Dict[str, float] = {}
        self._retry_count: Dict[str, int] = {}
        # No-loss submit boundary: set True whenever a dispatch is deferred by
        # a background-pool QueueFullError (the envelope stays a pending WAL
        # row — durably parked). The dispatch loop's idle branch consults this
        # cheap flag to decide whether to re-drain the parked rows via
        # _replay_wal once the pool reports capacity. Self-clearing after a
        # drain; re-set by any subsequent capacity deferral.
        self._capacity_deferred_pending: bool = False
        # Single-flight guard for _drain_capacity_deferred: the drain's WAL
        # read is offloaded through cooperative_fs_io (a real suspension
        # point), which makes reentry possible if a second caller reaches the
        # drain while the first is parked on the thread hop. True while a
        # drain cycle is in flight; overlapping callers return immediately
        # (the in-flight cycle covers them).
        self._capacity_drain_in_flight: bool = False
        self._pending_ack = PendingAckStore()
        self._dead_letter: List[IntentEnvelope] = []
        self._dispatch_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock_fd: Optional[int] = None
        # Optional post-ingest hook (A-narrator). Called with envelope on "enqueued" only.
        # Assign a coroutine callable to enable; None disables.
        self._on_ingest_hook: Optional[Callable[..., Any]] = None

        # Slice 5 Arc A follow-up #1 — governor "allow" log visibility.
        # Counters reset to zero on every rollup emit in summary mode.
        self._gov_allow_total: int = 0
        self._gov_allow_by_sensor: Dict[str, int] = {}

        # P2.4: Initialize GoalTracker for goal-directed prioritization.
        # Sets the module-level reference so _compute_priority can use it.
        try:
            from backend.core.ouroboros.governance.strategic_direction import GoalTracker

            self._goal_tracker = GoalTracker(config.project_root)
            _active_goal_tracker = self._goal_tracker
        except Exception:
            self._goal_tracker = None

        # ── Operation dependency tracking (DAG-based signal merging) ──
        # Maps file paths to (op_id, registered_at_monotonic).
        # Used to detect when a new signal targets files already under active
        # modification — prevents conflicting concurrent patches.
        # TTL prevents starvation: stale locks are force-released.
        #
        # Q3 Slice 1 — ``_active_file_ops_lock`` closes the
        # TTL-detect/register-overwrite race:
        #   T1: _find_file_conflict reads entry (op_X, t_old)
        #       and decides it's stale (now - t_old > TTL).
        #   T2: register_active_op writes (op_Y, t_now) — fresh.
        #   T1: del self._active_file_ops[fpath] — silently
        #       deletes T2's fresh registration; the next
        #       conflicting envelope dispatches concurrently with
        #       op_Y, exactly the file conflict the lock was
        #       meant to prevent.
        #
        # Fix: every read/write/delete of _active_file_ops occurs
        # under _active_file_ops_lock (threading.Lock — works in
        # async + sync contexts since the critical section is
        # pure dict mutation, no I/O). The stale-release branch
        # uses a CAS pattern: capture (op, t) outside the lock
        # for the test, then under the lock re-verify the entry
        # is *identity-equivalent* before deletion. A concurrent
        # write that overwrote the entry between capture and
        # re-verify causes the CAS to abort the delete.
        self._active_file_ops: Dict[str, Tuple[str, float]] = (
            {}
        )  # file_path -> (op_id, time.monotonic())
        self._active_file_ops_lock: threading.Lock = threading.Lock()
        self._queued_behind: Dict[str, List[IntentEnvelope]] = {}  # op_id -> [envelopes]
        self._file_lock_ttl_s: float = float(os.environ.get("JARVIS_FILE_LOCK_TTL_S", "300"))

        # ── Signal coalescing buffer ──
        # Envelopes targeting overlapping files within a window are merged into
        # a single multi-goal operation before dispatch (reduces cost by N×).
        # HIGH urgency signals bypass coalescing and dispatch immediately.
        self._coalesce_window_s: float = float(os.environ.get("JARVIS_COALESCE_WINDOW_S", "30"))
        # Maps frozenset(target_files) key -> (first_arrival_monotonic, [envelopes])
        self._coalesce_buffer: Dict[str, List[IntentEnvelope]] = {}
        self._coalesce_timestamps: Dict[str, float] = {}  # key -> first arrival

        # ── F1 Slice 2 — parallel IntakePriorityQueue ──
        # Built when either the master flag (primary-mode: priority queue
        # backs dequeue) OR the shadow flag (observational: priority queue
        # tracks what WOULD have been dequeued, legacy queue stays primary)
        # is on. Flags re-read at __init__ time and cached — per-session
        # lifecycle, consistent with governor mode capture pattern.
        self._f1_master_on: bool = _intake_priority_scheduler_enabled()
        self._f1_shadow_on: bool = _intake_priority_scheduler_shadow_enabled()
        self._priority_queue: Optional[IntakePriorityQueue] = None
        self._f1_shadow_delta_count: int = 0
        self._f1_shadow_agree_count: int = 0
        if self._f1_master_on or self._f1_shadow_on:
            # Caller-side telemetry sink logs at INFO for visibility when
            # operator is tracing intake decisions. Queue's own debug lines
            # live at DEBUG via the caller's logger — we only promote a
            # few high-signal events here.
            def _priority_telemetry_sink(event_type: str, payload: Dict[str, Any]) -> None:
                if event_type == "priority_inversion":
                    logger.warning(
                        "[IntakePriority] priority_inversion urgency=%s source=%s "
                        "waited_s=%.2f deadline_s=%s",
                        payload.get("urgency"),
                        payload.get("source"),
                        payload.get("waited_s", 0.0),
                        payload.get("deadline_s"),
                    )
                elif event_type == "backpressure_applied":
                    logger.warning(
                        "[IntakePriority] backpressure_applied source=%s "
                        "urgency=%s retry_after_s=%.2f queue_depth=%d",
                        payload.get("source"),
                        payload.get("urgency"),
                        payload.get("retry_after_s", 0.0),
                        payload.get("queue_depth_total", 0),
                    )

            self._priority_queue = IntakePriorityQueue(
                telemetry_sink=_priority_telemetry_sink,
            )
            logger.info(
                "[IntakePriority] wired mode=%s (master=%s shadow=%s)",
                "primary" if self._f1_master_on else "shadow",
                self._f1_master_on,
                self._f1_shadow_on,
            )

        # PRD §11 (S2) — auto-register this instance as the
        # process-wide default for S2's head-of-queue peek. NEVER
        # raises; failure to register degrades S2 to "no high-prio
        # signal" (fail-open). Last-write-wins matches the cost_governor
        # singleton pattern.
        try:
            set_default_intake_router(self)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[UnifiedIntakeRouter] auto-register-default " "degraded: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Acquire advisory lock, start dispatch loop, and replay pending WAL entries."""
        if self._running:
            return
        self._acquire_lock()
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="intake_dispatch")
        await self._replay_wal()
        await self._hydrate_fsm_checkpoints()

    async def _hydrate_fsm_checkpoints(self) -> None:
        """Autonomous startup resume: re-inject HMAC-VERIFIED suspended ops (from a
        prior window's wall-clock cap / Spot preemption) with their preserved
        exploration context, so the DAG fast-forwards instead of re-exploring.
        Gated by ``JARVIS_FSM_RESUME_ENABLED`` (default on). Fully fail-soft --
        NEVER blocks boot. Rejected (unverified) checkpoints fall back to clean boot."""
        if os.environ.get("JARVIS_FSM_RESUME_ENABLED", "true").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        try:
            from backend.core.ouroboros.governance import fsm_checkpoint as _ckpt  # noqa: PLC0415
            from backend.core.ouroboros.governance.intake.intent_envelope import (  # noqa: PLC0415
                make_envelope as _make_env,
            )

            async def _reinject(env: "Dict[str, Any]") -> None:
                # Distributed Trace Lineage: the kwargs reuse the ORIGINAL
                # causal_id + carry the verified lineage; the partial thought
                # + resume markers ride intake_evidence_json so the prefill
                # re-ignition reaches the LLM (model continues the exact char).
                _kw = _resume_envelope_kwargs(env)
                _lin = _kw["evidence"].get("trace_lineage") or {}
                if _lin.get("emit_source"):
                    # Re-establish the window-1 emit hop from HMAC-verified
                    # lineage BEFORE ingest, so the chain stays ordered in
                    # THIS window's ledger + log (span continuation).
                    try:
                        from backend.core.ouroboros.governance import (  # noqa: PLC0415
                            a1_trace as _a1t,
                        )

                        _a1t.restore_emit_record(
                            _kw["causal_id"],
                            source=str(_lin["emit_source"]),
                            original_emit_wall=_lin.get("emit_wall_ts"),
                        )
                    except Exception:  # noqa: BLE001 -- lineage is observability
                        pass
                envelope = _make_env(**_kw)
                await self.ingest(envelope)

            # hydrate_pending_checkpoints wants a sync ingest_fn; bridge to async by
            # scheduling each re-inject and consuming the checkpoint on success.
            _pending = _ckpt.list_pending()
            # Atomic Workspace Checkpoint restore (before any re-inject): re-apply
            # each distinct dirty-tree stash ONCE, so an op suspended mid-mutation
            # resumes against its own uncommitted delta even if the tree was
            # cleaned across the boundary. Fail-soft (already-present/conflict is
            # fine — the op resumes regardless).
            _applied_stashes: set = set()
            for _cp in _pending:
                _ws_ref = getattr(_cp, "workspace_stash_ref", "") or ""
                if _ws_ref and _ws_ref not in _applied_stashes:
                    _applied_stashes.add(_ws_ref)
                    try:
                        _ckpt.restore_workspace_stash(_ws_ref)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "Router: workspace stash restore faulted ref=%s",
                            _ws_ref[:12], exc_info=True,
                        )
            for _cp in _pending:
                # Poison-Pill guard (DETERMINISTIC, not a swallow): persist the
                # incremented hydration count BEFORE attempting the resume, so a
                # checkpoint that crashes the pipeline on resume can't loop
                # forever. Past the ceiling it is shunted to the quarantine DLQ
                # and the loop proceeds to the next op — pipeline survival.
                try:
                    _attempt = _ckpt.record_hydration_attempt(_cp)
                    if _attempt > _ckpt.hydration_max_attempts():
                        _ckpt.quarantine_checkpoint(
                            _cp, reason="hydration_attempts_exhausted",
                        )
                        logger.warning(
                            "Router: QUARANTINED poison-pill op=%s after %d "
                            "resume attempts -> DLQ; proceeding to next op",
                            _cp.op_id, _attempt - 1,
                        )
                        continue
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Router: hydration-attempt guard faulted op=%s",
                        getattr(_cp, "op_id", "?"), exc_info=True,
                    )
                try:
                    _env = _ckpt.build_resume_envelope(_cp)
                    await _reinject(_env)
                    _ckpt.mark_resumed(_cp.op_id)
                    if _env.get("drifted"):
                        # Git-drift: HEAD moved while suspended → demoted to
                        # re-PLAN, stale exploration dropped (context-corruption
                        # guard), rather than fast-forwarding into dead code.
                        logger.warning(
                            "Router: RESUMED suspended op=%s DEMOTED %s->%s "
                            "(repo drift: checkpoint sha=%s != live HEAD; stale "
                            "exploration dropped, re-planning against current source)",
                            _cp.op_id, _cp.phase, _env.get("resume_phase"),
                            str(_cp.repo_sha)[:10],
                        )
                    else:
                        logger.info(
                            "Router: RESUMED suspended op=%s phase=%s (%d exploration "
                            "records preserved -> Venom fast-forward)",
                            _cp.op_id, _env.get("resume_phase"),
                            len(_cp.exploration_records),
                        )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Router: FSM resume re-inject failed op=%s -- left pending",
                        getattr(_cp, "op_id", "?"),
                    )

            # Task 5 -- checkpoint-resume re-hydration seam: a resumed window
            # re-loads prior cognitive experience alongside the FSM checkpoint
            # it just replayed. Idempotent (module cache is just refreshed).
            try:
                from backend.core.ouroboros.governance import cognitive_persistence as _cogp

                if _cogp.is_enabled() and _cogp._env_bool(
                    "JARVIS_COGNITIVE_HYDRATE_ON_CHECKPOINT_RESUME", True
                ):
                    await _cogp.hydrate_prior_knowledge()
            except Exception as _e:  # noqa: BLE001
                logger.debug("[IntakeRouter] cognitive re-hydration skipped (fail-soft): %s", _e)
        except Exception:  # noqa: BLE001
            logger.debug("Router: FSM checkpoint hydration skipped (fail-soft)", exc_info=True)

    async def stop(self) -> None:
        """Gracefully stop the dispatch loop and release the advisory lock."""
        self._running = False
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except (asyncio.CancelledError, Exception):
                pass
        self._release_lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def ingest(self, envelope: IntentEnvelope) -> str:
        """Route an envelope through the intake pipeline.

        Returns one of:
        - ``"enqueued"``       — accepted and placed on the priority queue
        - ``"deduplicated"``   — duplicate within the dedup window, dropped
        - ``"pending_ack"``    — parked awaiting human acknowledgement
        - ``"backpressure"``   — queue is full; non-exempt source rejected
        """
        # Slice 33 Arc 1+ widening — intake routing is on the hot path
        # for every sensor signal; if it's slow, the loop notices.
        from backend.core.ouroboros.telemetry.loop_sink import (
            sink_async as _ls_sink_async,
        )

        async with _ls_sink_async("intake.UnifiedIntakeRouter.ingest"):
            return await self._ingest_impl(envelope)

    async def _ingest_impl(self, envelope: IntentEnvelope) -> str:
        # A1-T4 — hop 2/5 (ingest): the live router accepts the envelope.
        try:
            from backend.core.ouroboros.governance.a1_trace import (  # noqa: PLC0415
                a1trace as _a1trace,
                probe_ingest_order as _probe_ingest_order,
            )

            _a1trace("ingest", envelope.causal_id, router="attached")
            # Deep emit-hop telemetry (Run #17): emit the order-assertion vs
            # this goal's prior emit (MISSING if none -- the Run-#17 mode).
            # Observe-only, fail-soft.
            _probe_ingest_order(envelope.causal_id)
        except Exception:  # noqa: BLE001
            pass
        # Unified Provenance Ledger -- stamp a tamper-evident hash-chained
        # ProvenanceRecord{op_id, origin=SignalSource, ...} at THE ingestion
        # point so the GraduationAuditor can validate the origin-correct
        # pipeline (sensor ops legitimately have NO emit hop -- the Run-#17
        # fix). Gated JARVIS_PROVENANCE_LEDGER_ENABLED (default on), fail-soft:
        # a ledger error NEVER blocks ingestion; OFF is byte-identical.
        try:
            from backend.core.ouroboros.governance.provenance_ledger import (  # noqa: PLC0415
                stamp_provenance as _stamp_provenance,
            )

            # Trace Lineage: a resumed op stamps its ORIGINAL origin (from
            # HMAC-verified checkpoint lineage), not the resume transport.
            _stamp_provenance(envelope.causal_id, _provenance_origin_for(envelope))
        except Exception:  # noqa: BLE001
            pass
        # 0-qos. QoS Escalation pump — the intake loop's own activity IS
        # the aging clock (no tickers): each ingest tick opportunistically
        # re-admits ONE starved envelope whose weight breached threshold
        # under healthy liquidity. Fire-and-forget + reentrancy-guarded;
        # a pump failure never touches this envelope's path.
        self._maybe_pump_qos_starvation()

        # 1. Dedup check
        if self._is_duplicate(envelope):
            return "deduplicated"

        # 2. Human ack gate
        if envelope.requires_human_ack:
            self._pending_ack.park(envelope)
            return "pending_ack"

        # 3. File-overlap conflict detection (DAG-based signal merging)
        # If another op is already active on any of this envelope's target files,
        # queue behind it instead of spawning a conflicting concurrent patch.
        if envelope.target_files:
            blocking_op = self._find_file_conflict(envelope)
            if blocking_op is not None:
                self._queued_behind.setdefault(blocking_op, []).append(envelope)
                logger.info(
                    "[Router] Signal queued behind active op %s (file overlap: %s)",
                    blocking_op[:12],
                    ", ".join(envelope.target_files[:3]),
                )
                return "queued_behind"

        # 4. Backpressure check
        if (
            envelope.source not in _BACKPRESSURE_EXEMPT
            and self.intake_queue_depth() >= self._config.backpressure_threshold
        ):
            return "backpressure"

        # 4b. Slice 5 Arc A — SensorGovernor advisory consultation.
        # Shadow mode (default for Arc A first drop): log any would-be deny +
        # let the SSE bridge publish, but pass through. Enforce mode: honor.
        # Off mode: skip entirely (pre-Arc-A behavior).
        gov_mode = _intake_governor_mode()
        if gov_mode != "off":
            # QoS Escalation (Aging) — one-shot grant pre-empts the
            # weighted cap for a starved envelope whose aged weight
            # breached the threshold under healthy liquidity. The grant
            # bypasses ONLY this consult; dedup / backpressure /
            # file-conflict / risk cage all still applied above/below.
            _qos_granted = False
            try:
                from backend.core.ouroboros.governance.qos_starvation import (  # noqa: PLC0415, E501
                    get_default_ledger as _qos_ledger,
                    qos_escalation_enabled as _qos_enabled,
                )
                if _qos_enabled() and _qos_ledger().consume_grant(
                    envelope.causal_id,
                ):
                    _qos_granted = True
                    logger.info(
                        "[Router] governor consult PRE-EMPTED by QoS "
                        "escalation grant: envelope=%s source=%s",
                        envelope.causal_id[:16], envelope.source,
                    )
            except Exception:  # noqa: BLE001 — QoS must never break intake
                pass
            gov_decision = (
                None if _qos_granted else self._consult_governor(envelope)
            )
            if gov_decision is not None and not gov_decision.allowed:
                if gov_mode == "enforce":
                    logger.info(
                        "[Router] governor ENFORCE deny: "
                        "sensor=%s urgency=%s reason=%s cap=%d count=%d",
                        envelope.source,
                        envelope.urgency,
                        gov_decision.reason_code,
                        gov_decision.weighted_cap,
                        gov_decision.current_count,
                    )
                    # Starvation shunt (mandate 2): a throttled
                    # low-urgency envelope parks with a timestamp and
                    # ages toward escalation instead of vanishing.
                    try:
                        from backend.core.ouroboros.governance.qos_starvation import (  # noqa: PLC0415, E501
                            get_default_ledger as _qos_ledger2,
                        )
                        _qos_ledger2().shunt(
                            envelope,
                            reason=str(gov_decision.reason_code),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    return "governor_throttled"
                # shadow: would have denied but allow through
                logger.info(
                    "[Router] governor SHADOW deny (would have thrown): "
                    "sensor=%s urgency=%s reason=%s cap=%d count=%d",
                    envelope.source,
                    envelope.urgency,
                    gov_decision.reason_code,
                    gov_decision.weighted_cap,
                    gov_decision.current_count,
                )
            elif gov_decision is not None and gov_decision.allowed:
                # Follow-up #1 — visibility into "governor allowed this"
                self._note_governor_allow(envelope, gov_decision)

        # 4b-ii. Slice 101 Phase 4 — Cognitive load-shedding gate. Composes
        # cognitive_load_shedding.evaluate_cognitive_load() (which reads the
        # anti-fragility + predictive-postmortem forecast) to shed ONLY the
        # lowest-urgency deferrable signals when overload is forecast. Mirrors
        # the SensorGovernor shadow/enforce discipline. Bounded: only evaluates
        # for sheddable urgencies, so the forecast ledger I/O never fires on
        # critical/high/normal intake. The substrate master keeps it inert by
        # default. NEVER raises into intake.
        shed_mode = _intake_cognitive_shed_mode()
        if (
            shed_mode != "off"
            and str(
                getattr(envelope, "urgency", ""),
            )
            in _SHEDDABLE_URGENCIES
        ):
            try:
                from backend.core.ouroboros.governance.cognitive_load_shedding import (  # noqa: E501
                    LoadVerdict,
                    ShedKind,
                    evaluate_cognitive_load,
                )

                # Slice 22 — evaluate_cognitive_load reads the forecast ledger
                # + flock-appends its own ledger (disk I/O). Off the loop via
                # the unified offload substrate so a burst of sheddable-urgency
                # signals can't serialize that I/O on the event-loop thread.
                # Fail-soft: an offload fault degrades to the inline call
                # (byte-identical), never breaks intake.
                from backend.core.ouroboros.governance.cooperative_fs_io import (  # noqa: E501
                    is_offload_error as _is_offload_error_cl,
                    offload as _offload_cl,
                )
                _load = await _offload_cl(evaluate_cognitive_load, cpu_bound=False)
                if _is_offload_error_cl(_load):
                    _load = evaluate_cognitive_load()
                # Slice 23 — real-time backpressure escalation. The host-stress
                # forecast lags a signal BURST (the queue fills faster than
                # memory/CPU stress rises), so compose the LIVE intake queue
                # pressure: at/above the env fraction of backpressure_threshold,
                # a sheddable (already low-urgency = low-value) signal is shed
                # even if the forecast verdict is still NORMAL. This is what
                # throttles the 72-signal burst at the source before it can
                # saturate the offload pools and starve the loop.
                try:
                    _qcap = max(1, int(self._config.backpressure_threshold))
                    _queue_pressure = self.intake_queue_depth() / _qcap
                except Exception:  # noqa: BLE001
                    _queue_pressure = 0.0
                _under_backpressure = (
                    _queue_pressure >= _intake_backpressure_shed_fraction()
                )
                _forecast_shed = _load.verdict in (
                    LoadVerdict.ELEVATED,
                    LoadVerdict.OVERLOADED,
                ) and _load.shed_kind in (
                    ShedKind.SPECULATIVE_SHED,
                    ShedKind.BACKGROUND_SHED,
                    ShedKind.FULL_SHED,
                )
                # Only signals of a sheddable urgency reached this gate (the
                # _SHEDDABLE_URGENCIES guard above), so a backpressure shed is
                # by construction a LOW-VALUE shed — high-urgency/high-value
                # work is never eligible here.
                _should_shed = _forecast_shed or _under_backpressure
                if _should_shed:
                    _shed_reason = (
                        "backpressure" if _under_backpressure and not _forecast_shed
                        else "forecast"
                    )
                    if shed_mode == "enforce":
                        logger.info(
                            "[Router] cognitive-shed ENFORCE (%s): source=%s "
                            "urgency=%s verdict=%s shed=%s load=%.3f qp=%.2f",
                            _shed_reason, envelope.source, envelope.urgency,
                            _load.verdict.value, _load.shed_kind.value,
                            _load.load_score, _queue_pressure,
                        )
                        return "cognitive_shed"
                    logger.info(
                        "[Router] cognitive-shed SHADOW (%s, would shed): "
                        "source=%s urgency=%s verdict=%s shed=%s load=%.3f "
                        "qp=%.2f",
                        _shed_reason, envelope.source, envelope.urgency,
                        _load.verdict.value, _load.shed_kind.value,
                        _load.load_score, _queue_pressure,
                    )
            except Exception:  # noqa: BLE001 — never break intake
                pass

        # Slice 22 — cooperative yield before the heavy (WAL + offload) phase.
        # Under a signal BURST (72 ingests in ~30s wedged the loop in
        # bt-2026-07-15-154242), admitted signals would otherwise run their
        # heavy phase back-to-back; one sleep(0) here hands the loop a
        # scheduling slot per admitted signal so the deadman heartbeat keeps
        # ticking. No-op when event-loop governance is disabled. NEVER raises.
        try:
            from backend.core.ouroboros.governance.event_loop_governance import (  # noqa: E501
                cooperative_yield as _coop_yield,
            )
            await _coop_yield()
        except Exception:  # noqa: BLE001 — a yield fault never breaks intake
            pass

        # 4. WAL enqueue — durable before placing on in-memory queue
        lease_id = generate_operation_id("lse")
        envelope = envelope.with_lease(lease_id)
        await self._wal.append_async(
            WALEntry(
                lease_id=lease_id,
                envelope_dict=envelope.to_dict(),
                status="pending",
                ts_monotonic=time.monotonic(),
                ts_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

        # 5. Register dedup key now so subsequent duplicates are caught
        self._register_dedup(envelope)

        # 6. Place on priority queue (lower int = higher priority)
        # Cost-aware: factors urgency, file count, confidence, dependency credit.
        # Dependency credit: count how many signals are queued behind files
        # this op would touch — completing it unblocks them.
        _dep_credit = 0
        for _fpath in envelope.target_files or ():
            _blocking_entry = self._active_file_ops.get(_fpath)
            if _blocking_entry is not None:
                _blocking_id, _ = _blocking_entry
                _dep_credit += len(self._queued_behind.get(_blocking_id, []))
        # Tier-2b — compute the SemanticIndex prior OFF the asyncio loop.
        # The inline path (``_compute_priority``) previously ran two fastembed
        # encodes (``boost_for`` + ``score``) synchronously on the loop
        # (~3.5s / signal — the 2nd starvation tier). Here we fire the
        # non-blocking build trigger and dispatch a single combined encode
        # through ``cooperative_fs_io.offload`` (thread — fastembed releases
        # the GIL), then hand the result to ``_compute_priority`` as an
        # override so it skips the on-loop embed. Fail-soft: any fault →
        # override stays None → legacy inline path (still correct).
        _semantic_override: Optional[int] = None
        try:
            from backend.core.ouroboros.governance.semantic_index import (
                get_default_index,
            )

            # Per-repo index keyed on the router's configured project_root
            # (matches orchestrator / classify_runner CONTEXT_EXPANSION) — in
            # production this resolves to the same singleton the legacy
            # no-arg lookup did (cwd == project_root).
            _si = get_default_index(self._config.project_root)
            _si.build_async()  # non-blocking; schedules off-loop rebuild
            _sem_boost, _sem_score = await _si.boost_and_score_offloaded(
                envelope.description or "",
            )
            _semantic_override = int(_sem_boost)
            if (_sem_boost > 0 or _si.stats().built_at > 0) and isinstance(
                envelope.evidence,
                dict,
            ):
                envelope.evidence["semantic_alignment"] = round(
                    float(_sem_score),
                    4,
                )
                envelope.evidence["semantic_boost"] = int(_sem_boost)
        except Exception as _sem_exc:  # noqa: BLE001 — never break intake
            logger.debug(
                "[Router] offloaded semantic scorer skipped: %s",
                _sem_exc,
            )
            _semantic_override = None
        # Tier-4 — stratification coverage penalty OFF the asyncio loop.
        # The inline path (``_compute_priority`` → ``ingest_priority_penalty``
        # → ``file_has_test_coverage``) rglob-walks the ENTIRE tests/ tree
        # (~2,839 files) per signal and cold-builds the AST import map
        # (read + ast.parse of ~963K test lines) on first miss — the traced
        # 41,713 ms single on-loop block of bt-iso-1783102490 (plus the
        # 1–16 s recurring blocks). Here we branch on index warmth:
        #   * warm  → penalty via cheap in-memory lookups, dispatched
        #     through ``cooperative_fs_io.offload`` (thread hop covers the
        #     residual ``_count_lines`` file read);
        #   * cold  → PROCEED DEGRADED (override 0 — no penalty, no
        #     evidence) and fire the single-flight off-loop index build.
        #     The prior is advisory priority-ordering only, so ops flowing
        #     unbiased while the index warms mirrors the SemanticIndex
        #     Tier-2b fail-soft convention; intake never waits ~40 s.
        # Fail-soft: ANY fault degrades to 0 — never None, because None
        # would route ``_compute_priority`` back onto the inline on-loop
        # scan (the bug this block eradicates).
        _strat_override: Optional[int] = None
        if envelope.target_files and _ingest_stratification_enabled():
            _strat_override = 0  # degraded-proceed default
            try:
                from backend.core.ouroboros.governance.target_stratification import (  # noqa: E501
                    coverage_index_ready,
                    ingest_priority_penalty,
                    trigger_coverage_index_build,
                )

                _idx_ready = coverage_index_ready(self._config.project_root)
                # Always fire the trigger — it self-gates (single-flight
                # "building" set, TTL freshness, failure cooldown) and
                # returns immediately; also covers TTL refresh when warm.
                trigger_coverage_index_build(self._config.project_root)
                if _idx_ready:
                    from backend.core.ouroboros.governance.cooperative_fs_io import (  # noqa: E501
                        is_offload_error,
                        offload,
                    )

                    _pen = await offload(
                        ingest_priority_penalty,
                        envelope.target_files,
                        self._config.project_root,
                        suppress=_is_test_generation_intent(envelope),
                        cpu_bound=False,
                    )
                    if not is_offload_error(_pen):
                        _strat_override = max(0, int(_pen))
                        if _strat_override > 0 and isinstance(
                            envelope.evidence,
                            dict,
                        ):
                            envelope.evidence["stratification_penalty"] = _strat_override
            except Exception as _strat_exc:  # noqa: BLE001 — never break intake
                logger.debug(
                    "[Router] offloaded stratification skipped (degraded): %s",
                    _strat_exc,
                )
        priority, alignment = _compute_priority(
            envelope,
            dependency_credit=_dep_credit,
            repo_root=self._config.project_root,
            semantic_boost_override=_semantic_override,
            stratification_penalty_override=_strat_override,
        )
        # Stash goal-alignment diagnostics on the envelope so downstream
        # phases (orchestrator, SerpentFlow postmortems, dead-letter audit)
        # can trace why this signal landed where it did. Mutating evidence
        # in place is safe: frozen=True protects top-level fields but the
        # dict reference itself is writable (matches intent_envelope.py:166).
        if alignment is not None and alignment.is_match:
            try:
                envelope.evidence.update(alignment.as_evidence())
            except Exception as _stash_exc:  # pragma: no cover — defence in depth
                logger.debug("[Router] evidence stash failed: %s", _stash_exc)
        await self._queue.put(
            (priority, envelope.submitted_at, next(_HEAP_TIE_SEQ), envelope),
        )

        # Announce AFTER the put, never before: a line claiming a signal was
        # queued must not appear if the put itself raised. The operator's view
        # follows the queue's truth rather than predicting it.
        _announce_enqueued(envelope)

        # F1 Slice 2 — mirror to IntakePriorityQueue when wired.
        # Primary mode (master on): this queue becomes the source of truth
        # for dispatch; legacy _queue still receives puts for WAL/back-compat
        # but dispatch reads from priority queue instead.
        # Shadow mode (shadow on, master off): purely observational —
        # legacy _queue remains primary; priority queue lets us log
        # "what would have been dequeued next" for diagnostic delta.
        # Capacity-limit refusal: when back-pressure fires, the enqueue
        # is dropped from the priority queue but the legacy queue still
        # accepted it, preserving flag-off byte-parity.
        if self._priority_queue is not None:
            self._priority_queue.enqueue(envelope)

        # Slice 5 Arc A — record emission so rolling-window counters update.
        # Only fires if governor mode was not "off". Never raises.
        if _intake_governor_mode() != "off":
            self._record_governor_emission(envelope)

        # Fire A-narrator hook — non-critical; failures logged only
        if self._on_ingest_hook is not None:
            try:
                await self._on_ingest_hook(envelope)
            except Exception as _hook_exc:
                logger.debug("[Router] on_ingest_hook error: %s", _hook_exc)

        return "enqueued"

    # ------------------------------------------------------------------
    # Slice 5 Arc A — SensorGovernor consultation helpers
    # ------------------------------------------------------------------

    def _maybe_pump_qos_starvation(self) -> None:
        """Schedule ONE starved-envelope re-ingest when escalation
        conditions hold (aged weight ≥ threshold, liquidity healthy,
        pledge ratio unspent). Reentrancy-guarded so the pumped ingest's
        own tick cannot cascade. NEVER raises."""
        try:
            from backend.core.ouroboros.governance.qos_starvation import (
                get_default_ledger,
                qos_escalation_enabled,
            )
            if not qos_escalation_enabled():
                return
            if getattr(self, "_qos_pump_inflight", False):
                return
            ledger = get_default_ledger()
            entry = ledger.try_escalate()
            if entry is None:
                return
            self._qos_pump_inflight = True

            async def _pump() -> None:
                try:
                    result = await self.ingest(entry.envelope)
                    logger.info(
                        "[Router] QoS escalation re-ingest: envelope=%s "
                        "result=%s",
                        entry.causal_id[:16], result,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[Router] QoS escalation re-ingest degraded",
                        exc_info=True,
                    )
                finally:
                    self._qos_pump_inflight = False

            import asyncio as _asyncio
            _asyncio.get_running_loop().create_task(_pump())
        except RuntimeError:
            # No running loop (sync test context) — clear the guard so a
            # later async tick can pump.
            self._qos_pump_inflight = False
        except Exception:  # noqa: BLE001
            self._qos_pump_inflight = False
            logger.debug("[Router] QoS pump degraded", exc_info=True)

    def _consult_governor(self, envelope: IntentEnvelope) -> Optional[Any]:
        """Return a BudgetDecision (or None on any failure).

        Translates envelope source + urgency to governor vocabulary and
        calls ``request_budget()``. Never raises into the ingest path —
        governor outage must not break intake.
        """
        try:
            from backend.core.ouroboros.governance.sensor_governor import (
                Urgency as GovernorUrgency,
                ensure_seeded,
            )

            sensor_name = _SOURCE_TO_GOVERNOR_SENSOR.get(
                envelope.source,
                envelope.source,
            )
            urgency_str = _URGENCY_STR_TO_GOVERNOR.get(
                envelope.urgency,
                "standard",
            )
            urgency = GovernorUrgency(urgency_str)
            governor = ensure_seeded()
            return governor.request_budget(sensor_name, urgency)
        except Exception:  # noqa: BLE001 — governor must never break intake
            logger.debug(
                "[Router] governor consultation failed (non-fatal)",
                exc_info=True,
            )
            return None

    def _note_governor_allow(
        self,
        envelope: IntentEnvelope,
        decision: Any,
    ) -> None:
        """Follow-up #1 — rate-limited / opt-in visibility for governor allows.

        Behavior driven by ``JARVIS_INTAKE_GOVERNOR_ALLOW_LOG``:
          * ``off``     → no-op (default; matches pre-follow-up quiet)
          * ``summary`` → increment per-sensor counter; every N allows
                          emit ONE structured INFO rollup line + reset
          * ``debug``   → emit one DEBUG line per allow (opt-in verbose)

        Never raises. Counter state is per-router-instance; resets at
        every rollup emit so the N-allow window starts fresh.
        """
        mode = _allow_log_mode()
        if mode == "off":
            return
        try:
            sensor = envelope.source
            if mode == "debug":
                logger.debug(
                    "[Router] governor allow: sensor=%s urgency=%s " "cap=%d count=%d remaining=%d",
                    sensor,
                    envelope.urgency,
                    decision.weighted_cap,
                    decision.current_count,
                    decision.remaining,
                )
                return
            # summary: accumulate and emit one structured line per N allows
            self._gov_allow_total += 1
            self._gov_allow_by_sensor[sensor] = self._gov_allow_by_sensor.get(sensor, 0) + 1
            interval = _allow_log_interval()
            if self._gov_allow_total >= interval:
                top5 = sorted(
                    self._gov_allow_by_sensor.items(),
                    key=lambda kv: -kv[1],
                )[:5]
                pairs = " ".join(f"{k}={v}" for k, v in top5)
                logger.info(
                    "[Router] governor allow rollup: total=%d window=%d " "top_sensors=[%s]",
                    self._gov_allow_total,
                    interval,
                    pairs,
                )
                self._gov_allow_total = 0
                self._gov_allow_by_sensor.clear()
        except Exception:  # noqa: BLE001 — allow-log must never break intake
            logger.debug(
                "[Router] governor allow-log accounting failed (non-fatal)",
                exc_info=True,
            )

    def _record_governor_emission(self, envelope: IntentEnvelope) -> None:
        """Record emission in the rolling-window counter. Never raises."""
        try:
            from backend.core.ouroboros.governance.sensor_governor import (
                Urgency as GovernorUrgency,
                ensure_seeded,
            )

            sensor_name = _SOURCE_TO_GOVERNOR_SENSOR.get(
                envelope.source,
                envelope.source,
            )
            urgency_str = _URGENCY_STR_TO_GOVERNOR.get(
                envelope.urgency,
                "standard",
            )
            urgency = GovernorUrgency(urgency_str)
            ensure_seeded().record_emission(sensor_name, urgency)
        except Exception:  # noqa: BLE001
            logger.debug(
                "[Router] governor record_emission failed (non-fatal)",
                exc_info=True,
            )

    def intake_queue_depth(self) -> int:
        """Current number of items waiting in the dispatch queue."""
        return self._queue.qsize()

    # ----------------------------------------------------------------------
    # PRD §11 (S2) additive composition surface — head-of-queue inspector
    # ----------------------------------------------------------------------
    # S2's preemption-signal predicate needs to know the urgency of the
    # next-to-be-dispatched envelope. Rather than build a parallel
    # queue inspector (PRD §3 forbids), this read-only accessor
    # composes the existing IntakePriorityQueue's heap (already
    # maintained for dispatch ordering). Does NOT pop. NEVER raises.

    def peek_top_urgency(self) -> Optional[str]:
        """Return the urgency string (``'critical'`` / ``'high'`` /
        ``'normal'`` / ``'low'``) of the next-to-be-dispatched
        envelope in the priority queue, or ``None`` if the queue is
        empty, the priority scheduler is not active, or any
        introspection fault occurs.

        Read-only — does NOT pop. Composes ``_priority_queue._heap``
        (existing dispatch-ordering heap). No parallel queue. NEVER
        raises (PRD §11.4 fail-open contract).
        """
        try:
            pq = self._priority_queue
            if pq is None:
                return None
            heap = getattr(pq, "_heap", None)
            if not heap:
                return None
            top_entry = heap[0]
            rank = getattr(top_entry, "urgency_rank", None)
            if rank is None:
                return None
            # Reverse URGENCY_RANK lookup (small dict; O(4)).
            from .intake_priority_queue import URGENCY_RANK as _RANK

            for urgency_str, r in _RANK.items():
                if r == rank:
                    return urgency_str
            return None
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.debug(
                "[UnifiedIntakeRouter] peek_top_urgency degraded: %s",
                exc,
            )
            return None

    def dead_letter_count(self) -> int:
        """Number of envelopes that exhausted all retries."""
        return len(self._dead_letter)

    def pending_ack_count(self) -> int:
        """Number of envelopes parked awaiting human acknowledgement."""
        return self._pending_ack.count()

    async def acknowledge(
        self,
        idempotency_key: str,
        *,
        extra_evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Release a parked envelope back into the pipeline.

        Returns ``True`` if the envelope was found and successfully re-ingested.

        ``extra_evidence`` is merged into the released envelope's evidence
        dict before re-ingest. Used by the OpportunityMiner auto-ack lane to
        stamp ``auto_acked``/``auto_ack_reason`` on the queued envelope so
        downstream phases (Orange PR review, postmortem) can see the lane
        was used. The merge is shallow — top-level keys overwrite.
        """
        envelope = self._pending_ack.acknowledge(idempotency_key)
        if envelope is None:
            return False
        from .intent_envelope import make_envelope

        merged_evidence = dict(envelope.evidence)
        if extra_evidence:
            merged_evidence.update(extra_evidence)
        unblocked = make_envelope(
            source=envelope.source,
            description=envelope.description,
            target_files=envelope.target_files,
            repo=envelope.repo,
            confidence=envelope.confidence,
            urgency=envelope.urgency,
            evidence=merged_evidence,
            requires_human_ack=False,
            causal_id=envelope.causal_id,
            signal_id=envelope.signal_id,
        )
        result = await self.ingest(unblocked)
        return result == "enqueued"

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    def _coalesce_key(self, envelope: IntentEnvelope) -> str:
        """Key for grouping envelopes that target overlapping files.

        Empty ``target_files`` (localize-from-issue sources —
        ``swe_bench_pro`` / ``vision_sensor``; see intent_envelope
        ``_EMPTY_TARGET_FILES_EXEMPT_SOURCES``) must NOT collapse to the
        shared ``""`` key. Soak bt-2026-05-17-213727 proved that fused
        DISTINCT problems (psf__requests-3362 + django__django-16255)
        into one Frankenstein op (``" | ".join(descs)``). Fall back to
        the envelope's EXISTING provenance signature — B.2.1 sets
        ``evidence["signature"] = problem.instance_id`` expressly for
        router-side identity — so distinct problems stay strictly
        distinct ops. Absent a signature, key on the unique
        ``idempotency_key`` (each envelope its own op; never coalesces).
        No new key invented; reuses existing provenance.
        """
        if envelope.target_files:
            return "|".join(sorted(envelope.target_files))
        _sig = ""
        try:
            _sig = str((envelope.evidence or {}).get("signature", "")).strip()
        except Exception:  # noqa: BLE001 — never raise in dispatch hot path
            _sig = ""
        if _sig:
            return f"sig:{envelope.source}:{_sig}"
        return f"uniq:{envelope.idempotency_key}"

    def _flush_coalesced(self, key: str) -> Optional[IntentEnvelope]:
        """Merge buffered envelopes for *key* into a single multi-goal envelope.

        Returns the merged envelope, or None if the buffer is empty.
        """
        envelopes = self._coalesce_buffer.pop(key, [])
        self._coalesce_timestamps.pop(key, None)
        if not envelopes:
            return None
        if len(envelopes) == 1:
            return envelopes[0]
        # Merge: union target_files, combine descriptions, keep highest urgency
        _all_files: list = []
        _descs: list = []
        _urgency_rank = {"high": 0, "medium": 1, "low": 2}
        _best_urgency = "low"
        for env in envelopes:
            _all_files.extend(env.target_files)
            _descs.append(env.description)
            if _urgency_rank.get(env.urgency, 2) < _urgency_rank.get(_best_urgency, 2):
                _best_urgency = env.urgency
        _merged_files = tuple(dict.fromkeys(_all_files))  # dedup, preserve order
        _merged_desc = " | ".join(_descs)
        logger.info(
            "[Router] Coalesced %d signals targeting %s into single operation",
            len(envelopes),
            list(_merged_files)[:3],
        )
        # Use the first envelope as base, replace merged fields
        base = envelopes[0]
        for _absorbed in envelopes[1:]:
            try:
                self._wal.update_status(_absorbed.lease_id, "acked")
            except Exception:  # noqa: BLE001 — ack is best-effort; never block the flush
                logger.warning(
                    "[Intake] absorbed-lease ack failed lease=%s",
                    getattr(_absorbed, "lease_id", "?"),
                )
        return IntentEnvelope(
            schema_version=base.schema_version,
            source=base.source,
            description=_merged_desc,
            target_files=_merged_files,
            repo=base.repo,
            confidence=max(e.confidence for e in envelopes),
            urgency=_best_urgency,
            dedup_key=base.dedup_key,
            causal_id=base.causal_id,
            signal_id=base.signal_id,
            idempotency_key=base.idempotency_key,
            lease_id=base.lease_id,
            evidence=base.evidence,
            requires_human_ack=any(e.requires_human_ack for e in envelopes),
            submitted_at=base.submitted_at,
        )

    async def _dispatch_loop(self) -> None:
        """Background task: drain the priority queue and call GLS.submit().

        Applies a coalescing window: envelopes targeting overlapping files
        are buffered for up to ``_coalesce_window_s`` before dispatch.
        HIGH urgency signals bypass coalescing.

        F1 Slice 2: when the master flag is on (``_f1_master_on``), the
        ``IntakePriorityQueue`` is the source of truth for dequeue order.
        The legacy ``_queue`` still receives puts (for WAL/back-compat)
        but is drained as a tombstone after each priority-queue pop.
        When only shadow flag is on, legacy is primary and we log a delta
        if the priority queue would have popped a different envelope.
        """
        while self._running:
            envelope: Optional[IntentEnvelope] = None
            # F1 Slice 2: track whether legacy _queue.task_done() still owed
            # after we finish processing this envelope. In primary-mode we
            # drain the legacy queue AT dequeue time and mark done there,
            # so downstream task_done() calls would unbalance the counter.
            _legacy_task_done_owed: bool = False
            if self._f1_master_on and self._priority_queue is not None:
                # F1 primary-mode: priority queue is source of truth.
                decision = self._priority_queue.dequeue()
                if decision is None:
                    # Priority queue empty — sleep briefly + flush coalesce
                    # then loop. Symmetric to the legacy TimeoutError path.
                    try:
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        break
                    await self._flush_expired_coalesce_buffers()
                    # No-loss submit boundary: re-drain capacity-parked WAL rows
                    # now the queue is idle and (maybe) the pool freed a slot.
                    await self._drain_capacity_deferred()
                    continue
                envelope = decision.envelope
                # Drain the matching entry from the legacy queue so
                # qsize() stays honest. Best-effort: if the head doesn't
                # match, skip (legacy queue is tombstone in primary-mode).
                # The drain consumes task_done() balance here — downstream
                # stays balanced because _legacy_task_done_owed stays False.
                try:
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                logger.debug(
                    "[IntakePriority] primary dequeue urgency=%s source=%s "
                    "waited_s=%.2f mode=%s depth=%d",
                    decision.urgency,
                    decision.source,
                    decision.waited_s,
                    decision.dequeue_mode,
                    len(self._priority_queue),
                )
            else:
                # Legacy path (byte-identical to pre-F1).
                try:
                    priority, ts, _tie_seq, envelope = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Flush any expired coalescing buffers
                    await self._flush_expired_coalesce_buffers()
                    # No-loss submit boundary: re-drain capacity-parked WAL rows
                    # now the queue is idle and (maybe) the pool freed a slot.
                    await self._drain_capacity_deferred()
                    continue
                except asyncio.CancelledError:
                    break
                _legacy_task_done_owed = True

                # F1 shadow-mode delta: consume one from the priority queue
                # (kept in sync via mirror-ingest) and compare. Only logs
                # when the priority queue would have popped something else.
                if self._f1_shadow_on and self._priority_queue is not None:
                    shadow_decision = self._priority_queue.dequeue()
                    if shadow_decision is not None:
                        if shadow_decision.envelope is envelope:
                            self._f1_shadow_agree_count += 1
                        else:
                            self._f1_shadow_delta_count += 1
                            logger.info(
                                "[IntakePriority shadow_delta] "
                                "legacy_popped=%s:%s shadow_would_pop=%s:%s "
                                "(mode=%s waited_s=%.2f)",
                                envelope.source,
                                envelope.urgency,
                                shadow_decision.source,
                                shadow_decision.urgency,
                                shadow_decision.dequeue_mode,
                                shadow_decision.waited_s,
                            )

            # Defensive: both branches must set envelope. The `None` path
            # returns via `continue` above, so mypy/pyright can't infer;
            # an explicit guard here makes the invariant readable.
            if envelope is None:
                continue

            # HIGH urgency: bypass coalescing, dispatch immediately
            if envelope.urgency == "high" or self._coalesce_window_s <= 0:
                try:
                    await self._dispatch_one(envelope)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("Router: dispatch error for lease_id=%s", envelope.lease_id)
                finally:
                    if _legacy_task_done_owed:
                        self._queue.task_done()
                continue

            # Buffer for coalescing
            _key = self._coalesce_key(envelope)
            if _key not in self._coalesce_buffer:
                self._coalesce_buffer[_key] = []
                self._coalesce_timestamps[_key] = time.monotonic()
            self._coalesce_buffer[_key].append(envelope)
            if _legacy_task_done_owed:
                self._queue.task_done()

            # Flush if window expired
            await self._flush_expired_coalesce_buffers()

    async def _flush_expired_coalesce_buffers(self) -> None:
        """Dispatch any coalescing buffers whose window has expired."""
        _now = time.monotonic()
        _expired_keys = [
            k for k, ts in self._coalesce_timestamps.items() if _now - ts >= self._coalesce_window_s
        ]
        for _key in _expired_keys:
            merged = self._flush_coalesced(_key)
            if merged is not None:
                try:
                    await self._dispatch_one(merged)
                except Exception:
                    logger.exception("Router: dispatch error for coalesced key=%s", _key[:50])

    @staticmethod
    def _is_runtime_task(envelope: IntentEnvelope) -> bool:
        """Classify an envelope as a runtime task (NOT a code change).

        Runtime tasks: browse, search, email, play, open app, schedule, etc.
        Code changes: fix bug, implement, refactor, add feature, etc.

        Uses description analysis — no hardcoded mapping. The classification
        is based on the ABSENCE of code-change signals, not the presence of
        runtime signals (open-world assumption: anything not obviously code
        is treated as a runtime task if there are no target files).
        """
        desc = envelope.description.lower()
        # Code change indicators — if present, route to GLS
        _CODE_SIGNALS = (
            "fix bug",
            "implement",
            "refactor",
            "add feature",
            "update code",
            "write function",
            "create module",
            "modify file",
            "change the code",
            "add test",
            "fix the",
            "patch",
            "debug",
            "commit",
            "merge",
        )
        if any(signal in desc for signal in _CODE_SIGNALS):
            return False
        # If there are target files, it's likely a code operation
        if envelope.target_files:
            return False
        # Everything else is a runtime task
        return True

    async def _dispatch_one(self, envelope: IntentEnvelope) -> None:
        """Route an envelope to either RuntimeTaskOrchestrator or GLS.

        Decision: If the envelope describes a runtime task (no target files,
        no code-change signals), dispatch to RuntimeTaskOrchestrator.
        Otherwise, dispatch to GovernedLoopService for code changes.
        """
        # No-loss submit boundary — the background pool's capacity-rejection
        # class. Handled distinctly from genuine dispatch failures below (park
        # durably in the WAL, never dead-letter). Local import keeps the
        # module import graph flat.
        from backend.core.ouroboros.governance.background_agent_pool import (
            QueueFullError,
        )

        ikey = envelope.idempotency_key

        # A1-T4 — hop 3/5 (dequeue): the dispatch loop has pulled this
        # envelope off the priority queue and is handing it to dispatch.
        try:
            from backend.core.ouroboros.governance.a1_trace import (  # noqa: PLC0415
                a1trace as _a1trace,
            )

            _a1trace("dequeue", envelope.causal_id)
        except Exception:  # noqa: BLE001
            pass

        # P0.5 — substance telemetry. Classify this dispatched signal from the
        # evidence the perception layers stamped (value_band / work_order /
        # deep_analysis_category / reputation_boost) into the running substance
        # ratio, so "everything is annotation-grade" (Run #25) becomes a
        # measurable KPI. Authority-free, gated, fail-soft — never perturbs
        # dispatch.
        try:
            from backend.core.ouroboros.governance.substance_ledger import (
                record_dispatch as _record_substance,
            )

            _record_substance(getattr(envelope, "evidence", None))
        except Exception:  # noqa: BLE001 — telemetry never blocks dispatch
            pass

        # --- Route to RuntimeTaskOrchestrator for runtime tasks ---
        if self._runtime_orchestrator is not None and self._is_runtime_task(envelope):
            try:
                result = await asyncio.wait_for(
                    self._runtime_orchestrator.execute(
                        query=envelope.description,
                        context={
                            "source": envelope.source,
                            "envelope_id": envelope.causal_id,
                            "repo": getattr(envelope, "repo", "jarvis"),
                        },
                    ),
                    timeout=self._config.dispatch_timeout_s,
                )
                await self._wal.update_status_async(envelope.lease_id, "acked")
                self._retry_count.pop(ikey, None)
                logger.info(
                    "[Router] Runtime task dispatched: %s -> %s (%d steps)",
                    envelope.description[:50],
                    "SUCCESS" if result.success else "PARTIAL",
                    len(result.steps),
                )
                return
            except Exception as exc:
                logger.warning(
                    "[Router] Runtime dispatch failed, falling back to GLS: %s",
                    exc,
                )
                # Fall through to GLS as fallback

        # --- Route to GLS for code changes ---
        # Use submit_background() for parallel operation execution via
        # BackgroundAgentPool. Falls back to synchronous submit() if pool
        # is unavailable. This enables the organism to work on multiple
        # operations concurrently (Manifesto §3: disciplined concurrency).
        from backend.core.ouroboros.governance.op_context import OperationContext
        from backend.core.ouroboros.governance.operation_advisor import (
            infer_read_only_intent,
        )

        # Stamp is_read_only at intake (NOT in orchestrator — too late).
        # Session 8 (bt-2026-04-18-044640) exposed the ordering bug:
        # BackgroundAgentPool worker picks up op.context BEFORE orchestrator
        # runs, so a later orchestrator-side stamp never propagates back to
        # the pool's per-worker ceiling selection. Stamping at intake means
        # op.context.is_read_only is already True by the time the pool sees
        # the op, so the 900s read-only ceiling branch at
        # background_agent_pool.py:~691 fires correctly.
        _is_read_only_at_intake = bool(infer_read_only_intent(envelope.description or ""))
        # F2 Slice 2: when the envelope carries a non-empty
        # routing_override (set by a sensor that emitted a valid
        # routing_hint under the F2 master flag), stamp ctx.provider_route
        # at creation so UrgencyRouter can honor it via the
        # envelope_routing_override path. When unset, behavior is
        # byte-identical to pre-F2 (ROUTE phase computes the route
        # normally via UrgencyRouter.classify source-type mapping).
        _env_routing = getattr(envelope, "routing_override", "") or ""
        _pre_route = _env_routing if _env_routing else ""
        _pre_route_reason = f"envelope_routing_override:{_env_routing}" if _env_routing else ""
        # A1-T3 — stamp intake-origin DAG weight onto evidence BEFORE the
        # snapshot below, so a heavy multi-file GOAL's heaviness rides the
        # existing intake_evidence_json side-channel onto ctx (observable to
        # the prefetch + soak breadcrumbs). Reuses is_heavy_goal; never
        # duplicates the prefetch. Pure, fail-soft, gated by the prefetch flag.
        stamp_dag_weight(envelope)

        # ClusterIntelligence-CrossSession Slice 4: snapshot
        # envelope.evidence as JSON onto ctx so post-verify
        # cascade observers (and future arcs needing typed signal
        # context) can extract structured tags without a separate
        # side-channel. Defensive serialize -- if evidence isn't
        # JSON-encodable (shouldn't happen but be safe), stamp
        # empty string so downstream parsers see "no signal" not
        # "corrupt signal".
        try:
            import json as _json_local

            _intake_evidence_json = (
                _json_local.dumps(
                    dict(envelope.evidence or {}),
                    sort_keys=True,
                )
                if envelope.evidence
                else ""
            )
        except (TypeError, ValueError):
            _intake_evidence_json = ""
        ctx = OperationContext.create(
            target_files=envelope.target_files,
            description=envelope.description,
            op_id=envelope.causal_id,
            signal_urgency=envelope.urgency,
            signal_source=envelope.source,
            intake_evidence_json=_intake_evidence_json,
            is_read_only=_is_read_only_at_intake,
            provider_route=_pre_route,
            provider_route_reason=_pre_route_reason,
        )
        # Time-Dilated Hydration: a resumed op is REBORN here -- stamp a
        # fresh pipeline wall so no downstream reader (tool-loop envelope,
        # L2 repair's direct pipeline_deadline read) inherits window-1's
        # spent clock.
        ctx = _stamp_resume_pipeline_deadline(ctx, envelope.source)

        # SWE-bench op-isolation + routing fix (#2): stamp the
        # complexity floor at the EARLIEST point — here, before the op
        # is submitted and the provider route + budget are computed.
        # Sources in _COMPLEX_FLOOR_SOURCES are inherently COMPLEX
        # (localize + multi-file source fix from an issue, no
        # target_files); the downstream file-count heuristic would
        # mislabel a no-target envelope "simple" and starve PLAN +
        # budget (soaks bt-2026-05-17-194855 / -213727). create()
        # hardcodes task_complexity="" so we set it post-construction.
        # envelope.source is definitionally present here — robust to
        # the coalesce/BG-pool ctx path that left signal_source
        # unreadable at the orchestrator classify site. Composes the
        # existing closed set; no new key, no flag. NEVER fails intake.
        try:
            from backend.core.ouroboros.governance.complexity_classifier import (  # noqa: E501
                _COMPLEX_FLOOR_SOURCES,
            )

            if envelope.source in _COMPLEX_FLOOR_SOURCES:
                object.__setattr__(ctx, "task_complexity", "complex")
        except Exception:  # noqa: BLE001 — floor is best-effort; never block intake
            pass

        # Manifesto §1 — the Tri-Partite Microkernel must bridge Senses→Mind.
        # When a vision-originated envelope carries a frame_path, hoist it
        # onto ctx.attachments so the GENERATE phase can perceive the actual
        # pixels (not just the text evidence verdict). This is the ONLY site
        # in the CLASSIFY path authorized to populate ctx.attachments from
        # sensor evidence — per I7, all other readers of ctx.attachments are
        # limited to the VisionSensor / visual_verify pair.
        #
        # Two ingress shapes are recognized here:
        #
        #   1. evidence["vision_signal"]["frame_path"]
        #        Autonomous path. VisionSensor-emitted envelopes carry a
        #        single frame captured from Ferrari. kind="sensor_frame",
        #        optional app_id from the sensor's Quartz inspection.
        #
        #   2. evidence["user_attachments"] = [{"path": ...}, ...]
        #        Human-initiated path. SerpentFlow /attach REPL command
        #        builds the envelope with one-or-more operator-supplied
        #        files. kind="user_provided", no app_id. Accepts the full
        #        _VALID_ATTACHMENT_MIMES set (images + PDFs).
        #
        # Both paths converge on ctx.with_attachments() → the GENERATE
        # phase sees a uniform ctx.attachments surface regardless of
        # origin. That's exactly the §1 Unified Organism invariant.
        try:
            from backend.core.ouroboros.governance.op_context import (
                Attachment,
            )

            _hoisted: List[Any] = []

            # (1) VisionSensor autonomous path.
            _vis_sig = (envelope.evidence or {}).get("vision_signal")
            if isinstance(_vis_sig, dict):
                _frame_path = _vis_sig.get("frame_path")
                _app_id = _vis_sig.get("app_id")
                if isinstance(_frame_path, str) and _frame_path:
                    _att = Attachment.from_file(
                        _frame_path,
                        kind="sensor_frame",
                        app_id=_app_id if isinstance(_app_id, str) else None,
                    )
                    _hoisted.append(_att)
                    logger.info(
                        "[IntakeRouter] attachments_hoisted op=%s kind=sensor_frame "
                        "hash8=%s mime=%s app_id=%s source=%s",
                        envelope.causal_id,
                        _att.hash8,
                        _att.mime_type,
                        (_app_id or "-"),
                        envelope.source,
                    )

            # (2) Operator-initiated /attach path.
            _user_atts = (envelope.evidence or {}).get("user_attachments")
            if isinstance(_user_atts, (list, tuple)):
                for _entry in _user_atts:
                    if not isinstance(_entry, dict):
                        continue
                    _p = _entry.get("path")
                    if not isinstance(_p, str) or not _p:
                        continue
                    _att = Attachment.from_file(_p, kind="user_provided")
                    _hoisted.append(_att)
                    logger.info(
                        "[IntakeRouter] attachments_hoisted op=%s kind=user_provided "
                        "hash8=%s mime=%s basename=%s source=%s",
                        envelope.causal_id,
                        _att.hash8,
                        _att.mime_type,
                        os.path.basename(_p),
                        envelope.source,
                    )

            if _hoisted:
                ctx = ctx.with_attachments(tuple(_hoisted))
        except Exception as _exc:
            # Never fail intake on attachment issues — the op can still run
            # text-only. Log at DEBUG so a stale frame_path doesn't spam.
            logger.debug(
                "[IntakeRouter] attachment hoist skipped op=%s: %s",
                envelope.causal_id,
                _exc,
            )
        # A1-T4 — hop 4/5 (submit): the envelope's OperationContext is handed
        # to the GovernedLoopService (the FSM entry). goal id == ctx.op_id.
        try:
            from backend.core.ouroboros.governance.a1_trace import (  # noqa: PLC0415
                a1trace as _a1trace,
            )

            _a1trace("submit", ctx.op_id, target="GLS")
        except Exception:  # noqa: BLE001
            pass
        try:
            _submit_fn = getattr(self._gls, "submit_background", None)
            if _submit_fn is not None:
                await asyncio.wait_for(
                    _submit_fn(ctx, trigger_source=envelope.source),
                    timeout=self._config.dispatch_timeout_s,
                )
            else:
                await asyncio.wait_for(
                    self._gls.submit(ctx, trigger_source=envelope.source),
                    timeout=self._config.dispatch_timeout_s,
                )
            await self._wal.update_status_async(envelope.lease_id, "acked")
            self._retry_count.pop(ikey, None)
        except QueueFullError:
            # NO-LOSS SUBMIT BOUNDARY. The background pool is at capacity. This
            # is NOT a dispatch failure — do NOT increment the retry counter and
            # do NOT dead-letter (either would eventually LOSE the op). The WAL
            # already holds this envelope as a status="pending" row (durable
            # park); we simply decline to ack it. It is re-drained by
            # _drain_capacity_deferred() -> _replay_wal() once the pool reports
            # free capacity. The dispatcher stays fully async — no sync-inline
            # execution serializes the queue (the live bt-iso-1783144982 defect).
            self._capacity_deferred_pending = True
            logger.info(
                "[Router] bg_submit_deferred lease_id=%s source=%s "
                "reason=pool_capacity_full wal=pending (durably parked, no loss)",
                envelope.lease_id,
                envelope.source,
            )
            return
        except Exception as exc:
            retries = self._retry_count.get(ikey, 0) + 1
            self._retry_count[ikey] = retries
            logger.warning(
                "Router: dispatch failed (attempt %d/%d) for lease_id=%s: %s",
                retries,
                self._config.max_retries,
                envelope.lease_id,
                exc,
            )
            if retries >= self._config.max_retries:
                logger.error(
                    "Router: dead-lettering envelope lease_id=%s after %d retries",
                    envelope.lease_id,
                    retries,
                )
                await self._wal.update_status_async(envelope.lease_id, "dead_letter")
                self._dead_letter.append(envelope)
                self._retry_count.pop(ikey, None)
            else:
                # Re-enqueue for retry at the same priority.
                # Use put_nowait() to avoid blocking the dispatch loop (self-deadlock).
                # If the queue is full, dead-letter immediately rather than stall.
                priority, _alignment = _compute_priority(envelope)
                try:
                    self._queue.put_nowait(
                        (
                            priority,
                            envelope.submitted_at,
                            next(_HEAP_TIE_SEQ),
                            envelope,
                        )
                    )
                except asyncio.QueueFull:
                    logger.error(
                        "Router: queue full during retry — dead-lettering lease_id=%s",
                        envelope.lease_id,
                    )
                    await self._wal.update_status_async(envelope.lease_id, "dead_letter")
                    self._dead_letter.append(envelope)
                    self._retry_count.pop(ikey, None)

    # ------------------------------------------------------------------
    # No-loss submit boundary — capacity-deferred WAL drain
    # ------------------------------------------------------------------

    async def _drain_capacity_deferred(self) -> None:
        """Re-drain envelopes parked by a background-pool ``QueueFullError``.

        The no-loss submit boundary keeps a capacity-rejected envelope as a
        ``status="pending"`` WAL row (durable park) instead of running it
        synchronously inline. This method re-enqueues those parked rows for
        another dispatch attempt — reusing the EXISTING replay machinery
        (:meth:`_replay_wal`); no new queue, no custom retry loop.

        Called from the dispatch loop's idle branch, so it is the natural
        drain cadence with zero extra tasks. Guarded three ways to guarantee
        no double-dispatch and no hot spin:

        * ``_capacity_deferred_pending`` — cheap flag; skip the full WAL read
          entirely when nothing is parked.
        * ``_capacity_drain_in_flight`` — single-flight: the offloaded WAL
          read is a real suspension point, so a concurrent second drain call
          is possible; it returns immediately (the in-flight cycle covers it).
        * pool capacity — only replay when the pool reports a free slot
          (``has_background_capacity``); otherwise the parked rows sit idle
          in the WAL (no re-submit spin against a saturated pool).
        * in-memory queue + coalesce buffers empty — checked BEFORE the
          offloaded read and RE-CHECKED after it resolves (an ``ingest()``
          on another coroutine may have enqueued a fresh — and WAL-pending —
          envelope during the thread hop; replaying then would double-enqueue
          it). The re-enqueue itself runs on-loop with no true suspension
          point after the re-check, so it is atomic w.r.t. other coroutines.

        The WAL read (whole-file read + JSON parse) is routed through the
        ``cooperative_fs_io.offload`` substrate — this method runs on the
        dispatch loop's idle branch repeatedly, and a synchronous
        ``pending_entries()`` here would be a new instance of the
        sync-FS-on-loop class. Fail-soft: an OffloadError skips this cycle
        (flag stays set → retried next idle tick).

        NEVER raises.
        """
        if not self._capacity_deferred_pending:
            return
        if self._capacity_drain_in_flight:
            return
        # Capacity gate — fail-open (drain) if the GLS can't be probed.
        try:
            probe = getattr(self._gls, "has_background_capacity", None)
            if callable(probe) and not probe():
                return
        except Exception:  # noqa: BLE001
            pass
        # Only drain when nothing pending is already in flight (see docstring).
        if self._queue.qsize() > 0 or self._coalesce_buffer:
            return
        self._capacity_drain_in_flight = True
        try:
            # Sync-FS-on-loop guard: the whole-WAL read + JSON parse happens
            # on the shared cooperative_fs_io thread pool, not the loop.
            from backend.core.ouroboros.governance.cooperative_fs_io import (
                is_offload_error,
                offload,
            )

            pending = await offload(self._wal.pending_entries, cpu_bound=False)
            if is_offload_error(pending):
                # Flag stays set — retried on the next idle tick.
                logger.debug(
                    "[Router] capacity-deferred drain skipped (OffloadError): %r",
                    pending,
                )
                return
            # Re-check the idle gate: ingest() may have enqueued a fresh
            # (WAL-pending) envelope while we were parked on the thread hop.
            # Replaying now would double-enqueue it — defer to the next tick.
            if self._queue.qsize() > 0 or self._coalesce_buffer:
                return
            # Cleared optimistically; any row that hits QueueFullError again
            # on the re-dispatch re-sets the flag, so a partial drain
            # self-continues.
            self._capacity_deferred_pending = False
            if pending:
                # Reuse the existing replay machinery for the re-enqueue
                # (on-loop, cheap — the expensive read already happened).
                await self._replay_wal(pending=pending)
        except Exception:  # noqa: BLE001 — the drain must never kill the loop
            logger.exception("[Router] capacity-deferred drain failed")
        finally:
            self._capacity_drain_in_flight = False

    # ------------------------------------------------------------------
    # WAL crash recovery
    # ------------------------------------------------------------------

    async def _replay_wal(
        self, pending: Optional[List[WALEntry]] = None
    ) -> None:
        """Re-enqueue all pending WAL entries from a previous run.

        ``pending`` (no-loss submit boundary): pre-fetched entries from an
        already-offloaded ``pending_entries()`` read — the capacity-deferred
        drain passes these so the whole-WAL file read + JSON parse never runs
        synchronously on the dispatch loop. ``None`` (the once-at-``start()``
        crash-recovery call) preserves the legacy inline read.
        """
        if pending is None:
            pending = self._wal.pending_entries()
        if not pending:
            return
        logger.info("Router: replaying %d pending WAL entries", len(pending))
        from .intent_envelope import IntentEnvelope as IE

        for entry in pending:
            try:
                envelope = IE.from_dict(entry.envelope_dict)
                # ── WAL Signal Liveness Tombstoning (2026-07-22) ──
                # A replayed signal was recorded in a PREVIOUS session; by
                # replay time its target files may have vanished (e.g. the
                # preemption shield stashes UNTRACKED files at boot — soak
                # bt-2026-07-22-005943 chased backend/soak_probes/
                # soak_probe_math.py through a full GENERATE round when it
                # existed in NO visible tree). When every target of a
                # targeted signal is missing from the observation root,
                # the entry is cleanly tombstoned (status update — the
                # WAL's native exclusion mechanism, no bus rewrite) and
                # never enqueued: the orchestrator is not woken for a
                # ghost. Signals with no target_files (e.g. SWE-bench
                # envelopes carry none by design) are never touched.
                # Liveness stats run OFF-LOOP. Fail-soft: any check error
                # replays the entry normally (never lose a real signal).
                if _wal_target_liveness_enabled():
                    try:
                        _ghost = await asyncio.to_thread(
                            _all_targets_missing,
                            envelope.target_files,
                            self._config.project_root,
                        )
                    except Exception:  # noqa: BLE001 — never lose a signal
                        _ghost = False
                    if _ghost:
                        # WAL status vocabulary is CLOSED ('acked' |
                        # 'dead_letter'); a benign ghost is RESOLVED, not
                        # poison — 'acked' is the semantically correct
                        # tombstone, and this log line carries the reason.
                        self._wal.update_status(entry.lease_id, "acked")
                        logger.warning(
                            "Router: WAL TOMBSTONE (ghost target) — "
                            "lease_id=%s source=%s targets=%s absent from "
                            "observation root; signal resolved without "
                            "waking the orchestrator",
                            entry.lease_id, envelope.source,
                            ",".join(envelope.target_files or ()),
                        )
                        continue
                # WAL replay preserves whatever alignment metadata was
                # already stashed in envelope.evidence from the original
                # ingest — no need to re-score and pollute the replay path
                # with a second round of scorer failures if the tracker
                # happens to be broken at replay time.
                priority, _alignment = _compute_priority(envelope)
                await self._queue.put(
                    (
                        priority,
                        envelope.submitted_at,
                        next(_HEAP_TIE_SEQ),
                        envelope,
                    )
                )
                # F1 mirror — same guard as the ingest site: in primary mode
                # the IntakePriorityQueue is the dispatch source of truth, so
                # a replayed row that only lands on the legacy _queue would
                # never dequeue. Mirror-enqueue exactly like ingest step 6.
                # None when both F1 flags are off → default-off behavior is
                # byte-identical.
                if self._priority_queue is not None:
                    self._priority_queue.enqueue(envelope)
                logger.debug(
                    "Router: replayed lease_id=%s source=%s",
                    entry.lease_id,
                    envelope.source,
                )
            except Exception:
                logger.exception("Router: WAL replay failed for lease_id=%s", entry.lease_id)

    # ------------------------------------------------------------------
    # Operation dependency tracking (DAG-based signal merging)
    # ------------------------------------------------------------------

    def _find_file_conflict(self, envelope: IntentEnvelope) -> Optional[str]:
        """Return the op_id of an active operation that overlaps this envelope's files.

        Returns None if no conflict exists (safe to dispatch concurrently).
        Stale locks (older than ``_file_lock_ttl_s``) are force-released
        via a CAS pattern (Q3 Slice 1) — re-verifying the captured
        entry under ``_active_file_ops_lock`` before delete so a
        concurrent ``register_active_op`` overwriting the same key
        is never silently clobbered.
        """
        _now = time.monotonic()
        for fpath in envelope.target_files or []:
            with self._active_file_ops_lock:
                entry = self._active_file_ops.get(fpath)
                if entry is None:
                    continue
                _op_id, _registered_at = entry
                age = _now - _registered_at
                if age > self._file_lock_ttl_s:
                    # CAS: confirm the entry is still the stale tuple
                    # before deleting. If a concurrent register_active_op
                    # has already written a fresh tuple for this fpath,
                    # the identity check fails and we abort the delete.
                    current = self._active_file_ops.get(fpath)
                    if current is not None and (
                        current[0] == _op_id and current[1] == _registered_at
                    ):
                        logger.warning(
                            "[Router] Force-releasing stale file lock: %s held by %s for %.0fs (TTL %ds)",
                            fpath,
                            _op_id[:12],
                            age,
                            self._file_lock_ttl_s,
                        )
                        del self._active_file_ops[fpath]
                    else:
                        logger.debug(
                            "[Router] CAS aborted stale-release on %s: "
                            "entry mutated under us (was %s, now %s)",
                            fpath,
                            _op_id[:12],
                            current[0][:12] if current else "(absent)",
                        )
                    continue
                return _op_id
        return None

    def register_active_op(self, op_id: str, target_files: List[str]) -> None:
        """Mark files as actively being modified by an operation.

        Called by GLS/orchestrator when an operation enters the GENERATE phase.
        Q3 Slice 1: holds ``_active_file_ops_lock`` for the whole batch
        so a concurrent ``release_op`` can't iterate a mid-write view.
        """
        _now = time.monotonic()
        with self._active_file_ops_lock:
            for fpath in target_files:
                self._active_file_ops[fpath] = (op_id, _now)

    async def release_op(self, op_id: str) -> None:
        """Release file locks for a completed/failed operation.

        Any envelopes that were queued behind this op are re-ingested
        into the pipeline, now that the conflicting files are free.

        Q3 Slice 1: scan + delete sequence is atomic under
        ``_active_file_ops_lock`` so a concurrent ``register_active_op``
        for the SAME op_id (e.g., a retry path that re-registers) can't
        be clobbered between scan and delete.
        """
        # Clear file reservations atomically — also filters by op_id
        # under the lock so we never delete a key that was already
        # rewritten to a different op_id by a concurrent registrant.
        with self._active_file_ops_lock:
            stale_keys = [k for k, v in self._active_file_ops.items() if v[0] == op_id]
            for k in stale_keys:
                # Re-verify identity under the same lock — defends
                # against a concurrent register_active_op that
                # rewrote this exact key between the scan and the
                # delete (would change v[0] to a different op_id).
                current = self._active_file_ops.get(k)
                if current is not None and current[0] == op_id:
                    del self._active_file_ops[k]

        # Re-ingest queued signals (outside the lock — ingest is async + I/O)
        queued = self._queued_behind.pop(op_id, [])
        for envelope in queued:
            logger.info(
                "[Router] Re-ingesting signal queued behind completed op %s: %s",
                op_id[:12],
                envelope.description[:50],
            )
            await self.ingest(envelope)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedup_window_s(self, envelope: IntentEnvelope) -> float:
        """The window that governs THIS envelope. One definition, two readers
        (`_is_duplicate` decides; `describe_dedup_collision` explains) — so a
        verdict and its explanation can never disagree about the rule."""
        return (
            self._config.voice_dedup_window_s
            if envelope.source == "voice_human"
            else self._config.dedup_window_s
        )

    def _is_duplicate(self, envelope: IntentEnvelope) -> bool:
        """Return True if the envelope's dedup_key was seen within its window."""
        window = self._dedup_window_s(envelope)
        # Window of 0.0 effectively disables dedup
        if window <= 0.0:
            return False
        last = self._dedup.get(envelope.dedup_key)
        if last is None:
            return False
        return (time.monotonic() - last) < window

    def describe_dedup_collision(
        self, envelope: IntentEnvelope,
    ) -> Optional["DedupCollision"]:
        """Read-only: WHY this envelope deduplicated. ``None`` if it did not.

        The verdict `ingest` returns is the bare string ``"deduplicated"``,
        which tells a caller that the goal was dropped but nothing a human
        could act on — so a cockpit reporting it had to either stay silent or
        invent a reason. This exposes the facts the registry ALREADY holds
        (the key, and when it was last seen) without adding a second tracker,
        a cache, or any writable surface.

        NOT an authority: it reads `self._dedup` and returns; it never
        registers, evicts, or influences a verdict. Callers must treat a
        ``None`` return as "no collision recorded", never as an error.
        """
        try:
            window = self._dedup_window_s(envelope)
            if window <= 0.0:
                return None
            last = self._dedup.get(envelope.dedup_key)
            if last is None:
                return None
            age = max(0.0, time.monotonic() - last)
            if age >= window:
                return None            # expired — not the reason it dropped
            return DedupCollision(
                dedup_key=str(envelope.dedup_key),
                age_s=age,
                window_s=float(window),
            )
        except Exception:  # noqa: BLE001 — explanation is never load-bearing
            return None

    def _register_dedup(self, envelope: IntentEnvelope) -> None:
        """Record the current monotonic time for the envelope's dedup_key."""
        self._dedup[envelope.dedup_key] = time.monotonic()
        self._prune_dedup()

    def _prune_dedup(self) -> None:
        """Drop entries no window can still consider a duplicate.

        `self._dedup` was append-only: one entry per distinct dedup_key, kept
        for the life of the process. Every distinct signature an operator ever
        types, every distinct sensor signature, forever — unbounded growth on
        the intake hot path.

        Provably semantics-preserving: an entry is evicted only once it is
        older than the LONGEST configured window, at which point
        `_is_duplicate` would already have returned False for it under any
        source. So pruning can never turn a duplicate into an accepted
        envelope.

        Amortized: the O(n) sweep runs only when the dict crosses a
        threshold, so steady-state ingest stays O(1). NEVER raises — a
        housekeeping failure must not drop a signal.
        """
        try:
            if len(self._dedup) < _DEDUP_PRUNE_THRESHOLD:
                return
            horizon = max(
                float(self._config.dedup_window_s),
                float(self._config.voice_dedup_window_s),
                0.0,
            )
            now = time.monotonic()
            stale = [k for k, t in self._dedup.items() if (now - t) >= horizon]
            for k in stale:
                self._dedup.pop(k, None)
            if stale:
                logger.debug(
                    "[IntakeRouter] dedup prune evicted=%d remaining=%d",
                    len(stale), len(self._dedup),
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Advisory file lock
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> None:
        """Acquire an exclusive non-blocking flock on the lock file.

        Writes PID + timestamp metadata so stale locks from crashed processes
        can be detected and cleaned automatically on next startup.

        Raises RouterAlreadyRunningError if another *live* process holds the lock.
        """
        lock_path = self._config.resolved_lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            # Check if the holder is still alive before raising
            if self._cleanup_stale_lock(lock_path):
                # Stale lock removed — retry once
                fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise RouterAlreadyRunningError(
                        f"Another router instance holds the lock at {lock_path}"
                    )
            else:
                raise RouterAlreadyRunningError(
                    f"Another router instance holds the lock at {lock_path}"
                )
        # Write PID metadata for stale-lock detection.
        #
        # Visibility note: the flock auto-releases when the holding
        # process dies, so the common "dead process left behind stale
        # metadata" case never hits the reaper at ``_cleanup_stale_lock``
        # (that path only fires when flock itself is still held — e.g.
        # inherited by a live child process). Instead, stale metadata
        # is overwritten *silently* here. When we detect prior metadata
        # from a non-self PID, log a one-line INFO so operators can
        # tell a stale-lock overwrite from a fresh-first-boot write.
        self._lock_fd = fd
        try:
            import json as _json

            _prior_pid: Optional[int] = None
            _prior_age_s: Optional[float] = None
            try:
                _prior_raw = os.read(fd, 4096)
                if _prior_raw:
                    _prior_json = _json.loads(_prior_raw.decode(errors="replace"))
                    _prior_pid = int(_prior_json.get("pid", 0)) or None
                    _prior_ts = float(_prior_json.get("ts", 0.0)) or None
                    if _prior_ts:
                        _prior_age_s = time.time() - _prior_ts
            except (ValueError, OSError, KeyError, TypeError):
                _prior_pid = None
                _prior_age_s = None

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            # Harness Epic Slice 2 — additive schema upgrade. New fields:
            #   * monotonic_ts — for stale-TTL detection independent of
            #     wall-clock skew
            #   * wall_iso — human-readable timestamp for log audits
            #   * session_id — links lock to the session dir on disk
            # Old readers continue to work (they read pid + ts only).
            from datetime import datetime, timezone

            _session_id = os.environ.get("OUROBOROS_SESSION_ID", "")
            _meta = _json.dumps(
                {
                    "pid": os.getpid(),
                    "ts": time.time(),
                    "monotonic_ts": time.monotonic(),
                    "wall_iso": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "session_id": _session_id,
                }
            )
            os.write(fd, _meta.encode())
            os.fsync(fd)

            if _prior_pid and _prior_pid != os.getpid():
                _age_str = f"{_prior_age_s:.0f}s" if _prior_age_s is not None else "?s"
                logger.info(
                    "[IntakeRouter] Overwrote stale lock metadata "
                    "(prior_pid=%d prior_age=%s new_pid=%d)",
                    _prior_pid,
                    _age_str,
                    os.getpid(),
                )
        except OSError:
            pass  # Lock is held — metadata is advisory

    @staticmethod
    def _cleanup_stale_lock(lock_path: Path) -> bool:
        """Check if the process holding the lock is dead OR the lock is too old.

        Returns True if a stale lock was removed.

        Two staleness predicates (Harness Epic Slice 2):
          1. **Dead-PID stale** (pre-Slice-2): PID in lock metadata is
             not running → remove lock.
          2. **Wedged-but-alive stale** (NEW Slice 2): PID is running BUT
             lock's wall ``ts`` is older than ``JARVIS_INTAKE_LOCK_STALE_TTL_S``
             (default 7200s = 2h) → treat as wedged zombie, remove lock.
             Closes the 14-incident class where Py_FinalizeEx-deadlocked
             zombies held the lock for hours while still being "alive".
        """
        try:
            import json as _json

            data = _json.loads(lock_path.read_text())
            pid = data.get("pid", 0)
            ts = float(data.get("ts", 0.0)) if data.get("ts") else 0.0
            if pid and pid != os.getpid():
                # (1) Dead-PID staleness check
                try:
                    os.kill(pid, 0)  # signal 0 = existence check
                except ProcessLookupError:
                    lock_path.unlink(missing_ok=True)
                    logger.warning(
                        "[IntakeRouter] Removed stale lock (dead PID %d)",
                        pid,
                    )
                    return True
                except PermissionError:
                    pass  # PID alive, different user — fall through to TTL check
                # (2) Wedged-but-alive TTL check (Slice 2)
                _stale_ttl_raw = os.environ.get(
                    "JARVIS_INTAKE_LOCK_STALE_TTL_S",
                    "7200",
                )
                try:
                    _stale_ttl = float(_stale_ttl_raw)
                except (TypeError, ValueError):
                    _stale_ttl = 7200.0
                _age_s = time.time() - ts if ts > 0 else 0.0
                if ts > 0 and _age_s > _stale_ttl:
                    lock_path.unlink(missing_ok=True)
                    logger.warning(
                        "[IntakeRouter] Removed wedged-but-alive stale lock "
                        "(PID=%d alive, age=%.0fs > TTL=%.0fs — treating as "
                        "Py_FinalizeEx-class zombie)",
                        pid,
                        _age_s,
                        _stale_ttl,
                    )
                    return True
        except (ValueError, OSError, KeyError):
            # Corrupt or empty lock file — remove it
            lock_path.unlink(missing_ok=True)
            logger.warning("[IntakeRouter] Removed corrupt lock file")
            return True
        return False

    def _release_lock(self) -> None:
        """Unlock and close the advisory lock file descriptor."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None


# ---------------------------------------------------------------------------
# Process-wide default singleton (PRD §11 S2 wiring)
# ---------------------------------------------------------------------------
# S2's preemption-signal predicate (s2_predictive_budget.py) consumes
# the head-of-queue urgency via ``peek_top_urgency()``. To avoid
# threading the router instance through every provider call site,
# this module exposes a thread-safe singleton getter/setter. The
# setter is called from ``UnifiedIntakeRouter.__init__`` so any
# instantiation auto-registers; last-write-wins (matches the
# cost_governor singleton pattern).
#
# When no router is registered (e.g., S2 is exercised in unit tests
# that don't construct an IntakeLayer), ``get_default_intake_router``
# returns ``None`` — S2's ``_peek_high_prio_queued`` treats this as
# "no signal" and emits nothing. NEVER raises.

# ---------------------------------------------------------------------------
# Intake announcement — what the organism FOUND, before it acts on it
# ---------------------------------------------------------------------------
# The cockpit's deck renders ops. An op is the LAST thing to happen: a sensor
# fires, a signal is enqueued, and only later does an FSM context exist. On a
# real boot four genuine test failures sat in this queue for two minutes while
# the deck stayed blank and the status line read IDLE — the organism was doing
# exactly its job and the operator could not tell it apart from a dead one.
#
# A sink, not a bus subscription: `set_operator_dispatcher`,
# `set_degradation_sink` and `markup_mirror` are all wired this way already, and
# a sink has no dependency on a TrinityEventBus existing — which it need not,
# per this module's own docstring.
#
# ZERO AUTHORITY. The sink is told; it is never asked. Nothing about dispatch,
# priority or admission consults it, so a slow or broken cockpit cannot change
# what the organism does — only what an operator can see.
_ANNOUNCE_SINK: Optional[Callable[[Dict[str, Any]], None]] = None


def set_intake_announce_sink(
    sink: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    """Register the surface that renders newly-enqueued signals.

    Idempotent, last-write-wins, ``None`` to detach. NEVER raises.
    """
    global _ANNOUNCE_SINK
    _ANNOUNCE_SINK = sink


def _announce_enqueued(envelope: Any) -> None:
    """Tell the operator surface what was just found. NEVER raises, never
    blocks, and never touches the envelope.

    Called on the ingest path only. Requeues and retries deliberately do not
    announce: the operator cares that a failure was FOUND, not that it was
    moved between queues, and a line per internal transition would bury the
    one that mattered.
    """
    sink = _ANNOUNCE_SINK
    if sink is None:
        return
    try:
        targets = tuple(getattr(envelope, "target_files", ()) or ())
        sink({
            "source": str(getattr(envelope, "source", "") or ""),
            "description": str(getattr(envelope, "description", "") or ""),
            "target_files": targets,
            "urgency": str(getattr(envelope, "urgency", "") or ""),
            "signal_id": str(getattr(envelope, "signal_id", "") or ""),
        })
    except Exception:  # noqa: BLE001 — a display fault must never reach intake
        logger.debug("[Router] announce sink raised", exc_info=True)


_DEFAULT_INTAKE_ROUTER: Optional["UnifiedIntakeRouter"] = None
_DEFAULT_INTAKE_ROUTER_LOCK = threading.Lock()


def set_default_intake_router(router: "UnifiedIntakeRouter") -> None:
    """Register a router as the process-wide default for S2's
    head-of-queue inspection. Idempotent; last-write-wins. NEVER
    raises."""
    global _DEFAULT_INTAKE_ROUTER
    try:
        with _DEFAULT_INTAKE_ROUTER_LOCK:
            _DEFAULT_INTAKE_ROUTER = router
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug(
            "[UnifiedIntakeRouter] set_default_intake_router " "degraded: %s",
            exc,
        )


def get_default_intake_router() -> Optional["UnifiedIntakeRouter"]:
    """Return the registered default router, or ``None`` if no router
    has been registered in this process. NEVER raises."""
    try:
        with _DEFAULT_INTAKE_ROUTER_LOCK:
            return _DEFAULT_INTAKE_ROUTER
    except Exception:  # noqa: BLE001 — defensive
        return None


def reset_default_intake_router_for_tests() -> None:
    """Test helper — drops the registered default. NEVER raises."""
    global _DEFAULT_INTAKE_ROUTER
    try:
        with _DEFAULT_INTAKE_ROUTER_LOCK:
            _DEFAULT_INTAKE_ROUTER = None
    except Exception:  # noqa: BLE001 — defensive
        pass


def _stamp_resume_pipeline_deadline(ctx: Any, source: str) -> Any:
    """Time-Dilated Hydration: a resumed op must NEVER inherit a spent
    window-1 pipeline wall (the L2 repair path reads ``pipeline_deadline``
    DIRECTLY, no phase-deadline floor -- a stale value collapses its budget
    to seconds). Stamp a FRESH envelope at rebirth from the operator's
    ``JARVIS_PIPELINE_TIMEOUT_S``. Non-resume sources pass through untouched.
    NEVER raises."""
    try:
        if str(source or "") != "fsm_resume":
            return ctx
        from datetime import datetime, timedelta, timezone  # noqa: PLC0415

        _pt_s = max(1.0, float(os.environ.get("JARVIS_PIPELINE_TIMEOUT_S", "600") or 600))
        return ctx.with_pipeline_deadline(datetime.now(tz=timezone.utc) + timedelta(seconds=_pt_s))
    except Exception:  # noqa: BLE001 -- hydration hygiene, never fatal
        return ctx


def _resume_envelope_kwargs(env: "Dict[str, Any]") -> "Dict[str, Any]":
    """Build the ``make_envelope`` kwargs for an FSM-resume re-injection.

    Distributed Trace Lineage: reuse the ORIGINAL causal_id (span continuation
    -- the resumed op IS the same goal, so the A1Trace chain stays ONE span
    across windows instead of minting an orphan id the auditor can never
    verify) and carry the HMAC-verified lineage in the evidence so a further
    suspension preserves window-1's identity."""
    import json as _rj  # noqa: PLC0415

    _lineage = dict(env.get("trace_lineage") or {})
    _ev_json = _rj.dumps(
        {
            "resume": True,
            "resume_phase": env.get("resume_phase", ""),
            "partial_completion": env.get("partial_completion", ""),
            "resumed_op_id": env.get("op_id", ""),
            "trace_lineage": _lineage,
        }
    )
    _ev = {
        "resume": True,
        "resume_phase": env.get("resume_phase", ""),
        "partial_completion": env.get("partial_completion", ""),
        "resumed_op_id": env.get("op_id", ""),
        "tool_history": env.get("tool_history") or [],
        "exploration_records": env.get("exploration_records") or [],
        "trace_lineage": _lineage,
        "intake_evidence_json": _ev_json,
        "signature": "fsm_resume:%s" % (env.get("op_id", "")),
    }
    return {
        "source": "fsm_resume",
        "description": env.get("description") or "resume suspended op",
        "target_files": tuple(env.get("target_files") or ()),
        "repo": os.environ.get("JARVIS_REPO_ROOT", "."),
        "confidence": 1.0,
        "urgency": "high",
        "evidence": _ev,
        "requires_human_ack": False,
        "causal_id": str(_lineage.get("origin_goal_id") or env.get("op_id") or ""),
        "routing_override": (env.get("provider_route") or ""),
    }


def _provenance_origin_for(envelope: Any) -> str:
    """Resolve the provenance-stamp origin for an envelope.

    A resumed op's provenance class is a property of the SPAN, not of the
    resume transport: when HMAC-verified checkpoint lineage carries the
    original origin (e.g. ``roadmap``), stamp THAT -- so the auditor's
    origin-driven ``required_hops`` sees the same pipeline it saw in window 1.
    Falls back to the envelope source. NEVER raises."""
    try:
        _lin = (getattr(envelope, "evidence", None) or {}).get("trace_lineage") or {}
        _orig = str(_lin.get("origin_source") or "").strip()
        return _orig or str(getattr(envelope, "source", "") or "")
    except Exception:  # noqa: BLE001
        return str(getattr(envelope, "source", "") or "")
