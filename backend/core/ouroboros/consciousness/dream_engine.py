"""backend/core/ouroboros/consciousness/dream_engine.py

DreamEngine — idle GPU speculative analysis for the Trinity Consciousness layer.

Design:
    - Runs a background asyncio loop that waits for *all five* readiness gates
      before sending speculative analysis prompts to the inference cascade.
    - Inference goes through the ONE gate cascade every synchronous gate
      uses (``rt_gate.gate_completion_detailed``): DW-RT first for light
      dreaming, Claude-RT next, the locally served model last where its
      lane is enabled. The hand-rolled DW → Claude → legacy J-Prime HTTP
      cascade this replaced (2026-09-06) could not reach the local lane,
      so on a host with no cloud key every dream exhausted while a $0
      lane sat idle; and its J-Prime tier was a second transport to the
      same weights with its own timeout and parser. Idle-timer semantics
      (TC29) are unchanged: no PrimeRouter/PrimeClient is touched.
    - Between every HTTP call the preemption flag is checked; if set the
      job is abandoned immediately and partial state is saved for
      resume.  (TC17, TC30)
    - After preemption, re-entry into dream mode is blocked for
      ``config.dream_reentry_cooldown_s`` seconds (flap damping — TC18).
    - Job idempotency is enforced via ``compute_job_key()``: a completed
      key whose blueprint is still fresh is never recomputed.  (TC12)
    - Prompts are hard-capped at ``DREAM_MAX_PROMPT_CHARS`` characters.  (TC23)
    - When J-Prime is unavailable, DREAM_DORMANT is emitted via CommProtocol
      and no local heuristic fallback is attempted.  (TC24)
    - All state (blueprints, completed keys) is persisted to JSON on disk
      and restored on start.  (TC30)

Thread-safety:
    All mutable state is only touched inside the single asyncio event loop.
"""

from __future__ import annotations

import asyncio
import threading
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.core.ouroboros.consciousness.dream_metrics import DreamMetricsTracker
from backend.core.ouroboros.consciousness.types import (
    ConsciousnessConfig,
    ImprovementBlueprint,
    UserActivityMonitor,
    compute_blueprint_id,
    compute_job_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DREAM_MAX_PROMPT_CHARS: int = 2048

# Output-token budget for a dream completion — DECOUPLED from the prompt-char
# cap above (2026-07-17). The two were conflated: DREAM_MAX_PROMPT_CHARS (an
# INPUT character limit, TC23) was passed as the provider's ``max_tokens``
# (an OUTPUT token budget) — a type error with teeth. bt-2026-07-17-033933:
# the only entitled DW model (Qwen3.5-397B) carries a per-model effort FLOOR of
# "low", so it ALWAYS reasons; 2048 output tokens were consumed entirely by
# chain-of-thought, yielding 0 chars of content in 30.25s. The default leaves
# room for the think phase AND the answer; env-tunable per deployment.
_DREAM_MAX_OUTPUT_TOKENS_DEFAULT: int = 8192


def _dw_completion_degrade_streak() -> int:
    """Consecutive DIRECT_COMPLETION failures before the DW-RT tier is bypassed
    (``JARVIS_DW_COMPLETION_DEGRADE_STREAK``, default 2 — mirrors the
    heartbeat's own degrade-streak convention). Floor of 1: a bypass must
    always require at least one observed failure, never fire on a clean
    surface. Never raises."""
    try:
        return max(1, int(os.environ.get("JARVIS_DW_COMPLETION_DEGRADE_STREAK", "2")))
    except (TypeError, ValueError):
        return 2


def dream_max_output_tokens() -> int:
    """``JARVIS_DREAM_MAX_OUTPUT_TOKENS`` (default 8192). Floor of 1024 keeps a
    misconfiguration from re-creating the truncation class. Never raises."""
    try:
        return max(1024, int(os.environ.get(
            "JARVIS_DREAM_MAX_OUTPUT_TOKENS", _DREAM_MAX_OUTPUT_TOKENS_DEFAULT,
        )))
    except (TypeError, ValueError):
        return _DREAM_MAX_OUTPUT_TOKENS_DEFAULT

# Candidate hydration vocabulary (2026-07-17). The KEYS are the MemoryEngine's
# own closed insight-category vocabulary (types.MemoryInsight.category) — this
# maps that existing taxonomy onto a dream focus rather than inventing a second
# one. An unmapped category passes through verbatim, so the MemoryEngine can
# grow new categories without this dict silently swallowing them.
_PROMPT_FAMILY_BY_INSIGHT: Dict[str, str] = {
    "failure_pattern": "failure_repair",
    "file_fragility": "fragility_hardening",
    "success_pattern": "success_extension",
}
# Honest "nothing learned yet" focus for a cold memory — NOT a fabricated one.
_PROMPT_FAMILY_NEUTRAL: str = "general_improvement"
# No provider wired: the job is unkeyable to a serving model class.
_MODEL_CLASS_UNKNOWN: str = "unwired"


class DreamProviderExhaustedError(RuntimeError):
    """All RT inference tiers (DW → Claude → local) failed for a dream job.

    Typed routing exception (RT migration, 2026-07-16): per-tier failures are
    raised and caught at each tier boundary to drive the cascade; when the
    whole cascade is exhausted THIS is raised so the caller makes an explicit
    routing decision (emit DREAM_DORMANT + skip the cycle) instead of
    interpreting an ambiguous ``None``."""
"""Hard cap on dream prompt text length (TC23)."""

_DREAM_LOOP_INTERVAL_S: float = 30.0


def _exhaustion_backoff_base_s() -> float:
    """First cooldown after a full-cascade exhaustion (doubles per streak)."""
    try:
        return max(5.0, float(os.environ.get(
            "JARVIS_DREAM_EXHAUSTION_BACKOFF_BASE_S", "60")))
    except (TypeError, ValueError):
        return 60.0


def _exhaustion_backoff_max_s() -> float:
    try:
        return max(60.0, float(os.environ.get(
            "JARVIS_DREAM_EXHAUSTION_BACKOFF_MAX_S", "1800")))
    except (TypeError, ValueError):
        return 1800.0
"""Seconds between dream-loop ticks when not actively computing."""

#: The gate's tier names, as the dream ledger has always recorded them.
_TIER_PROVIDER_LABELS: Dict[str, str] = {
    "dw": "doubleword", "claude": "claude", "local": "local",
}

_DREAM_SYSTEM_PROMPT: str = (
    "You are a senior AI reasoning engine for the JARVIS Trinity ecosystem. "
    "Return well-structured JSON output."
)


# ---------------------------------------------------------------------------
# Persistence helpers (pure functions)
# ---------------------------------------------------------------------------


def _blueprint_to_dict(bp: ImprovementBlueprint) -> Dict[str, Any]:
    """Serialize an ImprovementBlueprint to a JSON-safe dict."""
    return {
        "blueprint_id": bp.blueprint_id,
        "title": bp.title,
        "description": bp.description,
        "category": bp.category,
        "priority_score": bp.priority_score,
        "target_files": list(bp.target_files),
        "estimated_effort": bp.estimated_effort,
        "estimated_cost_usd": bp.estimated_cost_usd,
        "repo": bp.repo,
        "repo_sha": bp.repo_sha,
        "computed_at_utc": bp.computed_at_utc,
        "ttl_hours": bp.ttl_hours,
        "model_used": bp.model_used,
        "policy_hash": bp.policy_hash,
        "oracle_neighborhood": bp.oracle_neighborhood,
        "suggested_approach": bp.suggested_approach,
        "risk_assessment": bp.risk_assessment,
    }


def _blueprint_from_dict(d: Dict[str, Any]) -> ImprovementBlueprint:
    """Deserialize a dict back to an ImprovementBlueprint."""
    return ImprovementBlueprint(
        blueprint_id=d["blueprint_id"],
        title=d["title"],
        description=d["description"],
        category=d["category"],
        priority_score=float(d["priority_score"]),
        target_files=tuple(d.get("target_files", ())),
        estimated_effort=d["estimated_effort"],
        estimated_cost_usd=float(d["estimated_cost_usd"]),
        repo=d["repo"],
        repo_sha=d["repo_sha"],
        computed_at_utc=d["computed_at_utc"],
        ttl_hours=float(d["ttl_hours"]),
        model_used=d["model_used"],
        policy_hash=d["policy_hash"],
        oracle_neighborhood=d.get("oracle_neighborhood", {}),
        suggested_approach=d.get("suggested_approach", ""),
        risk_assessment=d.get("risk_assessment", ""),
    )


# ---------------------------------------------------------------------------
# DreamEngine
# ---------------------------------------------------------------------------


class DreamEngine:
    """Idle GPU speculative analysis engine for Trinity Consciousness.

    Monitors five readiness gates and, when all pass, sends speculative
    code-improvement prompts to J-Prime via direct HTTP.  Results are
    stored as :class:`ImprovementBlueprint` objects on disk for later
    consumption by the governance pipeline.

    Parameters
    ----------
    health_cortex:
        HealthCortex instance with ``get_snapshot() -> TrinityHealthSnapshot``.
    memory_engine:
        MemoryEngine instance with ``get_file_reputation(path) -> FileReputation``.
    activity_monitor:
        Any object implementing ``last_activity_s() -> float``.
    resource_governor:
        ResourceGovernor with ``async should_yield() -> bool``.
    metrics_tracker:
        DreamMetricsTracker for recording compute time, preemptions, etc.
    config:
        ConsciousnessConfig with dream_* parameters.
    jprime_url:
        Base URL for J-Prime HTTP API (e.g. ``http://136.113.252.164:8000``).
        Used directly via aiohttp — NOT PrimeRouter/PrimeClient (TC29).
    persistence_dir:
        Directory for storing blueprints and job keys on disk.
    comm:
        Optional CommProtocol instance for emitting DREAM_DORMANT (TC24).
    """

    def __init__(
        self,
        health_cortex: Any,
        memory_engine: Any,
        activity_monitor: UserActivityMonitor,
        resource_governor: Any,
        metrics_tracker: DreamMetricsTracker,
        config: ConsciousnessConfig,
        jprime_url: str = "",
        persistence_dir: Optional[Path] = None,
        comm: Any = None,
        dw_provider: Any = None,
        claude_provider: Any = None,
        repo_path: Optional[str] = None,
    ) -> None:
        self._health_cortex = health_cortex
        self._memory_engine = memory_engine
        self._activity_monitor = activity_monitor
        self._resource_governor = resource_governor
        self._metrics_tracker = metrics_tracker
        self._config = config
        self._jprime_url: str = jprime_url
        self._comm = comm
        # Repo root for truthful repo-state hydration (git HEAD + policy hash).
        # None → _get_git_head falls back to cwd (correct when the process runs
        # from the repo root, e.g. the battle test).
        self._repo_path: Optional[str] = repo_path

        # DW + Claude providers (preferred over raw J-Prime HTTP)
        self._dw_provider = dw_provider
        self._claude_provider = claude_provider

        # Persistence
        default_dir = (
            Path.home()
            / ".jarvis"
            / "ouroboros"
            / "consciousness"
            / "dreams"
        )
        self._persistence_dir: Path = persistence_dir or default_dir
        self._persistence_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._blueprints: Dict[str, ImprovementBlueprint] = {}
        self._completed_keys: Set[str] = set()
        self._interrupted_jobs: Dict[str, Dict[str, Any]] = {}

        # Preemption (TC17)
        self._preempted: asyncio.Event = asyncio.Event()
        # Exhaustion backoff (2026-07-23): when the WHOLE RT cascade is
        # down, retrying every loop tick is a per-30s warning storm that
        # rides the log pipeline while providers are provably dead.
        # Escalating cooldown; any successful inference resets it.
        self._exhaustion_streak: int = 0
        self._exhaustion_until: float = 0.0

        # Flap damping (TC18) — monotonic time of last user return
        self._last_user_return: float = 0.0

        # Current repo state — updated by callers or internal polling
        self._current_head: str = ""
        self._current_policy_hash: str = ""

        # Dream loop task
        self._loop_task: Optional[asyncio.Task[None]] = None

        # Blueprint-computed observers (Gap 3 conception bridge) — additive,
        # default empty. Fired after a blueprint is stored so downstream
        # consumers (e.g. the conception proposal bridge) can react to
        # production without polling. No observer registered => no behavior
        # change.
        self._blueprint_observers: List[Callable[[Any], Any]] = []
        # TOCTOU barrier (2026-07-17): the observer list AND the blueprint store
        # are read/written under this one lock so that register-and-drain vs
        # store-and-notify can never interleave in a way that drops a payload.
        # bt-2026-07-17-085445 dropped the run's only blueprint: the dream fired
        # at 01:55:39, the ConceptionBridge armed its observer at 01:55:40 — the
        # blueprint was produced into a world with no listener.
        self._observer_lock = threading.Lock()
        self._intake_barrier_awaited: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_blueprint_observer(
        self, observer: Callable[[Any], Any],
    ) -> None:
        """Register a blueprint observer AND atomically reconcile the store.

        TOCTOU-free (Mandate 3). Under ``_observer_lock`` — a single
        thread-safe synchronous block — the observer is appended and the
        CURRENT blueprint store is snapshotted together. This closes the gap a
        naive ``get_blueprints()`` pull would leave open:

          * a blueprint stored BEFORE this call is in the snapshot → drained to
            the new observer here;
          * a blueprint stored AFTER this call sees the observer already in the
            list → delivered by ``_notify_blueprint_observers``.

        The lock serializes append-vs-snapshot against notify's observer
        snapshot, so a payload lands via exactly one path (or both — the bridge
        dedups by blueprint_id, so a double is harmless; a drop is impossible).
        Idempotent per identity. Delivery of the drained backlog happens OUTSIDE
        the lock (it awaits) on the running loop; NEVER raises."""
        with self._observer_lock:
            if observer in self._blueprint_observers:
                return
            self._blueprint_observers.append(observer)
            pending = list(self._blueprints.values())   # atomic w/ the append
        if pending:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._drain_backlog_to(observer, pending))
                logger.info(
                    "[DreamEngine] observer registered + %d historical "
                    "blueprint(s) reconciled (boot-race closed)", len(pending),
                )
            except RuntimeError:
                # No running loop (rare: sync test / pre-loop wiring). The
                # observer is registered; future blueprints still reach it.
                logger.debug("[DreamEngine] backlog drain skipped: no running loop")

    async def _drain_backlog_to(self, observer: Callable[[Any], Any],
                                pending: List[Any]) -> None:
        """Deliver the historical blueprint backlog to a freshly-registered
        observer (fail-soft, per-item isolated)."""
        for bp in pending:
            try:
                res = observer(bp)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001 — best-effort, never disturb boot
                logger.debug("[DreamEngine] backlog drain observer faulted",
                             exc_info=True)

    async def _notify_blueprint_observers(self, blueprint: Any) -> None:
        """Await each registered observer, isolating faults — an observer
        error never disturbs the dream loop. The observer snapshot is taken
        under ``_observer_lock`` so it is atomic against register-and-drain."""
        with self._observer_lock:
            observers = list(self._blueprint_observers)
        for obs in observers:
            try:
                res = obs(blueprint)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001 — observers are best-effort
                logger.debug(
                    "[DreamEngine] blueprint observer faulted", exc_info=True,
                )

    async def start(self) -> None:
        """Load persisted state from disk and start the background dream loop."""
        self._load_state()
        # Hydrate truthful repo-state at boot so the FIRST idle cycle can
        # already derive a candidate (no wait for a later refresh).
        self._hydrate_repo_state()
        self._loop_task = asyncio.create_task(
            self._dream_loop(), name="dream_engine_loop",
        )
        logger.info(
            "[DreamEngine] Started (idle_threshold=%.0fs, max_min/day=%.0f, "
            "blueprints=%d, keys=%d, head=%s, policy=%s)",
            self._config.dream_idle_threshold_s,
            self._config.dream_max_minutes_per_day,
            len(self._blueprints),
            len(self._completed_keys),
            (self._current_head or "unset")[:10],
            (self._current_policy_hash or "unset"),
        )

    async def stop(self) -> None:
        """Cancel the dream loop, set preemption, and persist state to disk."""
        self._preempted.set()
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        self._loop_task = None
        self._persist_state()
        logger.info("[DreamEngine] Stopped, state persisted.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_blueprints(self, top_n: int = 5) -> List[ImprovementBlueprint]:
        """Return up to *top_n* non-stale blueprints sorted by priority desc."""
        fresh: List[ImprovementBlueprint] = []
        for bp in self._blueprints.values():
            if not bp.is_stale(self._current_head, self._current_policy_hash):
                fresh.append(bp)
        fresh.sort(key=lambda b: b.priority_score, reverse=True)
        return fresh[:top_n]

    def get_blueprint(self, blueprint_id: str) -> Optional[ImprovementBlueprint]:
        """Return a specific blueprint by ID, or None if not found."""
        return self._blueprints.get(blueprint_id)

    def discard_stale(self) -> int:
        """Remove stale blueprints from the store.  Return count removed."""
        stale_keys: List[str] = []
        for key, bp in self._blueprints.items():
            if bp.is_stale(self._current_head, self._current_policy_hash):
                stale_keys.append(key)
        for key in stale_keys:
            del self._blueprints[key]
            self._completed_keys.discard(key)
            self._metrics_tracker.record_blueprint_discarded()
        if stale_keys:
            logger.info(
                "[DreamEngine] Discarded %d stale blueprints", len(stale_keys),
            )
        return len(stale_keys)

    # ------------------------------------------------------------------
    # Readiness gates
    # ------------------------------------------------------------------

    async def _can_dream(self) -> Tuple[bool, str]:
        """Check all five readiness gates.  Returns (can_dream, reason).

        Gates (checked in order):
            1. J-Prime healthy + model loaded  (TC09)
            2. User idle >= threshold           (TC10)
            3. VM warm from user traffic        (TC11)
            4. ResourceGovernor not yielding
            5. Daily dream-minutes budget
            + Flap damping cooldown             (TC18)
        """
        # Gate 0: Flap damping (TC18)
        if self._last_user_return > 0.0:
            elapsed = time.monotonic() - self._last_user_return
            if elapsed < self._config.dream_reentry_cooldown_s:
                remaining = self._config.dream_reentry_cooldown_s - elapsed
                return False, (
                    f"Flap damping cooldown: {remaining:.0f}s remaining "
                    f"(threshold {self._config.dream_reentry_cooldown_s:.0f}s)"
                )

        # Gate 1: Inference backend available
        # Original gate required J-Prime healthy + model loaded (TC09).
        # With DW 35B and Claude as inference backends, J-Prime is no longer
        # required — skip the health check if an alternative is available.
        _has_alt_backend = (
            self._dw_provider is not None or self._claude_provider is not None
        )
        if not _has_alt_backend:
            # Legacy path: require J-Prime health
            snapshot = self._health_cortex.get_snapshot()
            if snapshot is None:
                return False, "No health snapshot and no alternative inference backend"
            prime = snapshot.prime
            if prime.status != "healthy":
                return False, f"Prime not healthy: status={prime.status}"
            if not prime.details.get("model_loaded"):
                return False, "Prime model not loaded"

        # Gate 2: User idle (TC10)
        idle_s = self._activity_monitor.last_activity_s()
        if idle_s < self._config.dream_idle_threshold_s:
            return False, (
                f"User active: idle {idle_s:.0f}s < "
                f"threshold {self._config.dream_idle_threshold_s:.0f}s"
            )

        # Gate 3: VM warm from user traffic (TC11)
        # When using DW/Claude (cloud-based), VM warmth is irrelevant —
        # inference doesn't run on the local VM.
        if not _has_alt_backend:
            snapshot = self._health_cortex.get_snapshot()
            prime = snapshot.prime if snapshot else None
            uptime_s = prime.details.get("uptime_s", 0) if prime else 0
            if uptime_s < self._config.dream_idle_threshold_s:
                return False, (
                    f"VM uptime too short: {uptime_s:.0f}s < "
                    f"threshold {self._config.dream_idle_threshold_s:.0f}s "
                    "(VM may have been woken for dream, not by user warm traffic)"
                )

        # Gate 4: Resource governor
        should_yield = await self._resource_governor.should_yield()
        if should_yield:
            return False, "ResourceGovernor says yield — system under pressure"

        # Gate 5: Daily budget (TC23 budget)
        metrics = self._metrics_tracker.get_metrics()
        if metrics.opportunistic_compute_minutes >= self._config.dream_max_minutes_per_day:
            return False, (
                f"Dream minutes budget exhausted: "
                f"{metrics.opportunistic_compute_minutes:.1f} >= "
                f"{self._config.dream_max_minutes_per_day:.1f}"
            )

        return True, "all_gates_passed"

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def _check_preempted(self) -> bool:
        """Return True if the preemption event has been set (TC17)."""
        return self._preempted.is_set()

    def _save_interrupted(self, job_key: str, candidate_info: Dict[str, Any]) -> None:
        """Save interrupted job info for potential resume (TC30)."""
        self._interrupted_jobs[job_key] = candidate_info
        logger.debug(
            "[DreamEngine] Saved interrupted job %s for resume", job_key[:16],
        )

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def _is_job_completed(
        self,
        job_key: str,
        current_head: str,
        current_policy_hash: str,
    ) -> bool:
        """Return True if the job key has been completed and its blueprint is fresh."""
        if job_key not in self._completed_keys:
            return False
        bp = self._blueprints.get(job_key)
        if bp is None:
            # Key exists but blueprint was removed — not completed
            self._completed_keys.discard(job_key)
            return False
        if bp.is_stale(current_head, current_policy_hash):
            # Blueprint is stale — needs recomputation
            self._completed_keys.discard(job_key)
            return False
        return True

    # ------------------------------------------------------------------
    # Token budget
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_prompt(text: str) -> str:
        """Truncate prompt text to DREAM_MAX_PROMPT_CHARS (TC23)."""
        if len(text) <= DREAM_MAX_PROMPT_CHARS:
            return text
        return text[:DREAM_MAX_PROMPT_CHARS]

    # ------------------------------------------------------------------
    # CommProtocol emission
    # ------------------------------------------------------------------

    async def _emit_dormant(self, reason: str) -> None:
        """Emit DREAM_DORMANT via CommProtocol (TC24).

        When J-Prime is unavailable, we emit this reason code and do NOT
        substitute local heuristics.
        """
        if self._comm is None:
            logger.debug(
                "[DreamEngine] DREAM_DORMANT (%s) — no comm, skipping emit",
                reason,
            )
            return
        try:
            await self._comm.emit_heartbeat(
                op_id="dream_engine",
                phase=f"DREAM_DORMANT:{reason}",
                progress_pct=0.0,
            )
            logger.info("[DreamEngine] Emitted DREAM_DORMANT: %s", reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "[DreamEngine] Failed to emit DREAM_DORMANT", exc_info=True,
            )

    # ------------------------------------------------------------------
    # Dream loop
    # ------------------------------------------------------------------

    async def _await_intake_barrier(self) -> None:
        """Lifecycle barrier (Mandate 2): block the FIRST dream until the
        conception bridge's observer is armed, so no inference compute is burned
        producing a blueprint nothing can route. Deterministic — an
        ``asyncio.Event`` (``observers_armed``), not a sleep/retry.

        Fail-OPEN: when the bridge is disabled there is no observer to wait for,
        so the barrier releases immediately; a bounded timeout guarantees a
        never-arming intake can never wedge dreaming (the atomic register-drain
        would still reconcile any blueprint produced meanwhile). Runs once."""
        if self._intake_barrier_awaited:
            return
        self._intake_barrier_awaited = True
        try:
            from backend.core.ouroboros.governance.conception_proposal_bridge import (
                await_observers_armed,
                master_enabled as _bridge_enabled,
            )

            if not _bridge_enabled():
                return  # no observer will ever arm — do not block dreaming
            armed = await await_observers_armed(self._intake_barrier_timeout_s())
            logger.info(
                "[DreamEngine] intake barrier %s — first dream may proceed",
                "released (observers armed)" if armed
                else "timed out (proceeding; atomic drain covers late arm)",
            )
        except Exception:  # noqa: BLE001 — the barrier is an optimization, not a gate
            logger.debug("[DreamEngine] intake barrier skipped", exc_info=True)

    @staticmethod
    def _intake_barrier_timeout_s() -> float:
        """``JARVIS_DREAM_INTAKE_BARRIER_TIMEOUT_S`` (default 120s — a full boot
        of the 6-layer stack). Bounded so a never-arming intake never wedges the
        dream loop. Never raises."""
        try:
            return max(1.0, float(os.environ.get(
                "JARVIS_DREAM_INTAKE_BARRIER_TIMEOUT_S", "120")))
        except (TypeError, ValueError):
            return 120.0

    async def _dream_loop(self) -> None:
        """Background loop: check gates, pick candidate, compute blueprint."""
        # Barrier: wait for the routing infrastructure (conception bridge
        # observer) before the first candidate/inference cycle. Boot-race fix
        # (bt-2026-07-17-085445). Bounded + fail-open — never wedges the loop.
        await self._await_intake_barrier()
        while True:
            try:
                # Reset preemption at start of each cycle
                self._preempted.clear()

                # Exhaustion cooldown gate: while the whole cascade was
                # down moments ago, do NOT re-hammer it every tick —
                # sleep out the escalating cooldown quietly (debug, not
                # warning — the storm class this kills).
                remaining = self._exhaustion_until - time.monotonic()
                if remaining > 0:
                    logger.debug(
                        "[DreamEngine] exhaustion cooldown — %.0fs until "
                        "next cascade attempt (streak=%d)",
                        remaining, self._exhaustion_streak,
                    )
                    await asyncio.sleep(min(remaining, _DREAM_LOOP_INTERVAL_S))
                    continue

                can, reason = await self._can_dream()
                if not can:
                    logger.debug("[DreamEngine] Cannot dream: %s", reason)
                    # If prime is not available, emit dormant (TC24)
                    if "prime" in reason.lower() and "healthy" not in reason.lower():
                        await self._emit_dormant(reason)
                    await asyncio.sleep(_DREAM_LOOP_INTERVAL_S)
                    continue

                # Attempt to compute a blueprint
                await self._run_dream_job()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[DreamEngine] Unexpected error in dream loop")

            try:
                await asyncio.sleep(_DREAM_LOOP_INTERVAL_S)
            except asyncio.CancelledError:
                raise

    async def _run_dream_job(self) -> Optional[ImprovementBlueprint]:
        """Execute a single dream job: generate an improvement blueprint.

        Returns the computed blueprint or None if preempted / skipped.
        """
        start_mono = time.monotonic()

        # Adaptive refresh: re-hydrate truthful repo-state before selecting a
        # candidate so a HEAD advanced since boot is honored (and stale
        # blueprints from the prior HEAD are correctly discarded downstream).
        self._hydrate_repo_state()

        # Recovery probe: if DW's completion surface is bypassed, this is the
        # ONLY traffic that can observe it healing (a bypass silences organic
        # calls). Fires only while degraded — a healthy DW costs nothing.
        await self._trace_dw_recovery_if_bypassed()

        # Build candidate info (in a real implementation this would
        # come from the oracle/memory engine analysis of the repo)
        candidate = self._pick_candidate()
        if candidate is None:
            logger.debug("[DreamEngine] No candidate to dream about")
            return None

        job_key = compute_job_key(
            candidate["repo_sha"],
            candidate["policy_hash"],
            candidate["prompt_family"],
            candidate["model_class"],
        )

        # Idempotency check
        if self._is_job_completed(
            job_key, candidate["repo_sha"], candidate["policy_hash"],
        ):
            self._metrics_tracker.record_dedup()
            logger.debug("[DreamEngine] Job %s already completed, skipping", job_key[:16])
            return None

        # Check preemption before HTTP (TC17)
        if self._check_preempted():
            self._save_interrupted(job_key, candidate)
            self._metrics_tracker.record_preemption()
            self._last_user_return = time.monotonic()
            return None

        # Semantic Context Engineering (2026-07-18): hydrate the FRONTIER block
        # async (sha-cached — zero git calls on repeat cycles) so the sync
        # prompt builder can attach it. Fail-soft: "" → dream proceeds without
        # frontier direction.
        try:
            from backend.core.ouroboros.governance.frontier_context import (  # noqa: PLC0415
                frontier_context_async,
            )
            candidate["frontier_digest"] = await frontier_context_async(
                repo_sha=str(candidate.get("repo_sha", "") or ""),
            )
        except Exception:  # noqa: BLE001 — hydration must never break a dream
            candidate["frontier_digest"] = ""

        # Build prompt (TC23: capped)
        prompt = self._build_dream_prompt(candidate)
        prompt = self._truncate_prompt(prompt)

        # Send via the RT cascade: DW-RT → Claude-RT → J-Prime (legacy).
        # Exhaustion is a TYPED routing exception, not an ambiguous None.
        try:
            result = await self._call_inference(prompt)
            self._exhaustion_streak = 0
            self._exhaustion_until = 0.0
        except DreamProviderExhaustedError as exc:
            self._exhaustion_streak += 1
            cooldown = min(
                _exhaustion_backoff_max_s(),
                _exhaustion_backoff_base_s()
                * (2 ** min(self._exhaustion_streak - 1, 16)),
            )
            self._exhaustion_until = time.monotonic() + cooldown
            # ONE line per escalation (WARNING the first, INFO after):
            # a dead cascade is one fact, not a per-tick warning storm.
            log = logger.warning if self._exhaustion_streak == 1 else logger.info
            log(
                "[DreamEngine] inference cascade exhausted: %s — "
                "backing off %.0fs (streak=%d)",
                exc, cooldown, self._exhaustion_streak,
            )
            await self._emit_dormant("provider_cascade_exhausted")
            return None

        # Check preemption after HTTP (TC17)
        if self._check_preempted():
            self._save_interrupted(job_key, candidate)
            self._metrics_tracker.record_preemption()
            self._last_user_return = time.monotonic()
            return None

        if result is None:
            # Preempted mid-tier — no dormant emit, just skip the cycle.
            return None

        # Build blueprint from result
        blueprint_id = compute_blueprint_id(
            candidate["repo_sha"],
            candidate["policy_hash"],
            candidate["prompt_family"],
            candidate["model_class"],
        )
        blueprint = self._parse_blueprint_result(
            blueprint_id, candidate, result,
        )
        if blueprint is not None:
            self._blueprints[blueprint_id] = blueprint
            self._completed_keys.add(job_key)
            self._metrics_tracker.record_blueprint_computed()
            # Event source for the conception proposal bridge (Gap 3) — fires
            # after the blueprint is durably stored. No-op when no observer.
            await self._notify_blueprint_observers(blueprint)

        # Record compute time
        elapsed_min = (time.monotonic() - start_mono) / 60.0
        self._metrics_tracker.record_compute_time(elapsed_min)

        return blueprint

    def _hydrate_repo_state(self) -> None:
        """Hydrate ``_current_head`` + ``_current_policy_hash`` from truthful
        repository state so ``_pick_candidate`` can derive a candidate.

        Dynamic + adaptive: called at boot and before each dream job, so a HEAD
        that advances between cycles is picked up and older blueprints correctly
        go stale. **Fail-safe** (Mandate 2): reuses ``memory_engine._get_git_head``
        (5s-timeout subprocess that returns None on a locked/absent/erroring
        index — DRY, Mandate 3) and only OVERWRITES the fields on a truthful
        read. A None read leaves the prior value intact; on first boot that
        keeps ``_current_head`` empty and ``_pick_candidate`` returns None (no
        candidate, no crash) rather than dreaming about a phantom SHA. NEVER
        raises — the dream loop and the primary governed loop are never
        disturbed by a git hiccup."""
        try:
            from backend.core.ouroboros.consciousness.memory_engine import (
                _get_git_head,
            )

            head = _get_git_head(self._repo_path)
            if head:
                self._current_head = head
            policy_hash = self._compute_policy_hash()
            if policy_hash:
                self._current_policy_hash = policy_hash
        except Exception:  # noqa: BLE001 — hydration is best-effort, fail-safe
            logger.debug(
                "[DreamEngine] repo-state hydration skipped", exc_info=True,
            )

    def _compute_policy_hash(self) -> str:
        """Truthful, stable fingerprint of the active brain-selection policy.

        sha256[:16] of ``brain_selection_policy.yaml`` (the canonical model
        policy per CLAUDE.md), searched by env override then repo-relative
        candidates. Returns the stable sentinel ``"nopolicy"`` when the file is
        unreadable/absent — non-empty so the candidate guard passes, and stable
        so staleness detection stays consistent. NEVER raises."""
        try:
            candidates = []
            env_p = os.environ.get("JARVIS_BRAIN_POLICY_PATH", "").strip()
            if env_p:
                candidates.append(Path(env_p))
            root = Path(self._repo_path) if self._repo_path else Path.cwd()
            candidates += [
                root / "brain_selection_policy.yaml",
                root / "config" / "brain_selection_policy.yaml",
                root / "backend" / "core" / "ouroboros" / "governance"
                / "brain_selection_policy.yaml",
            ]
            for p in candidates:
                try:
                    if p.is_file():
                        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                except OSError:
                    continue
        except Exception:  # noqa: BLE001
            pass
        return "nopolicy"

    def _pick_candidate(self) -> Optional[Dict[str, Any]]:
        """Select the next candidate for speculative analysis.

        In a full implementation, this would consult the oracle for
        high-fragility files and the memory engine for recent failures.
        For now returns a skeleton candidate derived from current state.
        """
        # Check for interrupted jobs first (TC30: resume)
        if self._interrupted_jobs:
            key, info = next(iter(self._interrupted_jobs.items()))
            del self._interrupted_jobs[key]
            logger.info("[DreamEngine] Resuming interrupted job %s", key[:16])
            return info

        if not self._current_head or not self._current_policy_hash:
            return None

        # Dynamic candidate hydration (2026-07-17) — replaces the hardcoded
        # ``prompt_family="general_improvement"`` / ``model_class="qwen2.5-7b"``
        # stub (the latter was also a lie: no soak has ever served a dream on
        # qwen2.5-7b). BOTH axes are now derived organically from live runtime
        # telemetry, which is what gives ``compute_job_key`` genuine identity
        # diversity WITHOUT touching the hash or the dedup registry: as the
        # organism's memory shifts (a new failure family emerges) or its
        # provider topology changes, the key changes with it and a fresh dream
        # is legitimately warranted.
        return {
            "repo": self._repo_name(),
            "repo_sha": self._current_head,
            "policy_hash": self._current_policy_hash,
            "prompt_family": self._derive_prompt_family(),
            "model_class": self._derive_model_class(),
        }

    def _repo_name(self) -> str:
        """Repo identity from the live repo path (no hardcoded slug)."""
        try:
            if self._repo_path:
                return Path(self._repo_path).name or "jarvis"
        except Exception:  # noqa: BLE001
            pass
        return "jarvis"

    def _derive_prompt_family(self) -> str:
        """The dream's focus, derived from live MemoryEngine telemetry.

        Composes the existing ``get_pattern_summary()`` API (no duplicated
        telemetry parsing): the organism's dominant NON-EXPIRED insight
        category — ranked by the engine's own evidence_count ordering — names
        what it should speculate about. ``failure_pattern`` → repair the
        recurring break; ``file_fragility`` → harden the brittle surface;
        ``success_pattern`` → extend what works.

        The category vocabulary is the MemoryEngine's own (types.MemoryInsight);
        we map it to a family rather than inventing a taxonomy. A cold memory
        (fresh organism, no ingested outcomes) has no insights and yields the
        neutral family — an honest "nothing learned yet", not a fabricated
        focus. NEVER raises."""
        try:
            summary = self._memory_engine.get_pattern_summary()
            now_iso = datetime.now(timezone.utc).isoformat()
            for insight in getattr(summary, "top_patterns", ()) or ():
                try:
                    if insight.is_expired(now_iso):
                        continue
                except Exception:  # noqa: BLE001 — malformed insight is not a focus
                    continue
                cat = str(getattr(insight, "category", "") or "").strip().lower()
                if cat:
                    return _PROMPT_FAMILY_BY_INSIGHT.get(cat, cat)
        except Exception:  # noqa: BLE001 — telemetry is advisory, never fatal
            logger.debug("[DreamEngine] prompt_family telemetry cold", exc_info=True)
        return _PROMPT_FAMILY_NEUTRAL

    def _derive_model_class(self) -> str:
        """The model class that will actually serve, from the live provider
        topology — mirroring ``_call_inference``'s real tier order (DW-RT
        first, Claude-RT fallback). Reads each provider's own ``_model`` rather
        than restating a slug, so a topology change (entitlement shift, model
        pin, provider outage) organically re-keys the job. NEVER raises."""
        for provider, tier in (
            (self._dw_provider, "dw"),
            (self._claude_provider, "claude"),
        ):
            if provider is None:
                continue
            try:
                model = str(getattr(provider, "_model", "") or "").strip()
            except Exception:  # noqa: BLE001
                model = ""
            if model:
                return f"{tier}:{model}"
            return tier
        return _MODEL_CLASS_UNKNOWN

    def _build_dream_prompt(self, candidate: Dict[str, Any]) -> str:
        """Build the speculative analysis prompt for J-Prime.

        Semantic Context Engineering (2026-07-18): the prompt now hydrates two
        bounded, deterministic, cached context blocks BEFORE the directive —

          * ARCHITECTURAL INTENT (``north_star_context``): the PRD §6 A-level
            "definition of DONE" table + the §-header outlines of the PRD and
            the North Star Galaxy, so speculative hypotheses intentionally
            ALIGN with feature progression (the Apple/GCP/DW/Claude alliance
            roadmap) instead of merely hunting code rot.
          * RECENT HUMAN FRONTIER (``frontier_context``, hydrated async into
            ``candidate['frontier_digest']`` by the job runner): the modules
            the human was actively building — dream attention picks up where
            that work ended.

        Both fail-soft to "" (byte-identical legacy directive when dark) and
        are master-gated + char-capped; ``_truncate_prompt`` (TC23) remains the
        downstream hard cap. NOT a RAG pipeline — plain deterministic extracts.
        """
        _context_blocks = []
        try:
            from backend.core.ouroboros.governance.north_star_context import (  # noqa: PLC0415
                north_star_context,
            )
            _intent = north_star_context()
            if _intent:
                _context_blocks.append(_intent)
        except Exception:  # noqa: BLE001 — hydration must never break a dream
            pass
        _frontier = str(candidate.get("frontier_digest", "") or "")
        if _frontier:
            _context_blocks.append(_frontier)
        _context = ("\n\n".join(_context_blocks) + "\n\n") if _context_blocks else ""
        _alignment = (
            "Prefer an improvement that ADVANCES the architectural intent and "
            "CONTINUES the recent human frontier above.  " if _context else ""
        )
        return (
            f"{_context}"
            f"Analyze the repository at SHA {candidate['repo_sha']} "
            f"for potential improvements in the '{candidate['prompt_family']}' "
            f"category.  {_alignment}"
            f"Suggest one concrete, small improvement with "
            f"estimated effort, target files, risk assessment, and approach.  "
            f"Return JSON with keys: title, description, category, "
            f"priority_score, target_files, estimated_effort, "
            f"estimated_cost_usd, suggested_approach, risk_assessment."
        )

    async def _call_inference(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send the prompt through the shared RT gate cascade.

        DW-RT first (light dreaming buys tokens, not time — a per-call
        preference, never a global policy flip), Claude-RT next, the local
        lane last where enabled. Every tier is a real-time call with a hard
        per-tier timeout and fail-fast semantics inside the gate; a tier
        whose text is not a JSON object is rejected by ``accept`` and the
        cascade moves on, as it always did. Returns the parsed blueprint or
        ``None`` on preemption; raises :class:`DreamProviderExhaustedError`
        when the whole cascade is exhausted.
        """
        # Check preemption before any call (TC17)
        if self._check_preempted():
            return None
        from backend.core.ouroboros.governance.rt_gate import (  # noqa: PLC0415
            GateProviderExhaustedError,
            gate_completion_detailed,
        )
        rt_timeout = self._dream_rt_timeout_s()
        # DW only through its stream-free RT primitive, and only when the
        # heartbeat does not already know it is down (no timeout tax).
        dw = self._dw_provider
        if dw is not None and (
            not hasattr(dw, "complete_sync") or self._dw_health_bypass()
        ):
            dw = None
        # Model: env override, else the provider's own default (the
        # account's entitled model) — no hardcoded model names.
        _dw_model = os.environ.get("JARVIS_DREAM_DW_MODEL", "").strip() or None
        try:
            raw, tier = await gate_completion_detailed(
                prompt,
                caller_id="dream_engine",
                system_prompt=_DREAM_SYSTEM_PROMPT,
                max_tokens=dream_max_output_tokens(),
                response_format={"type": "json_object"},
                timeout_s=rt_timeout,
                claude_provider=self._claude_provider,
                dw_provider=dw,
                dw_model=_dw_model,
                prefer="dw",
                accept=lambda text: self._parse_json_response(text) is not None,
            )
        except GateProviderExhaustedError as exc:
            raise DreamProviderExhaustedError(
                f"all RT inference tiers exhausted ({exc})"
            ) from exc
        if self._check_preempted():
            return None
        result = self._parse_json_response(raw)
        if result is None:
            # `accept` admitted it and the parser is deterministic; this is
            # a contract violation, not a provider outage, and says so.
            raise DreamProviderExhaustedError(
                "accepted response failed to parse on re-read"
            )
        result["_inference_provider"] = _TIER_PROVIDER_LABELS.get(tier, tier)
        logger.info("[DreamEngine] RT inference succeeded via %s tier", tier)
        return result

    @staticmethod
    def _dw_health_bypass() -> bool:
        """True when the DW-RT tier should be SKIPPED because the heartbeat
        already knows DW is unhealthy — eliminating the timeout tax.

        SURFACE-AWARE (2026-07-17). Reads the EXISTING SurfaceHealthLedger
        verdict for ``DIRECT_COMPLETION`` — the stream-free
        /v1/chat/completions surface that ``complete_sync`` (this tier's ONLY
        transport) actually uses — never the SSE surface.

        This decoupling is load-bearing. The first version of this bypass read
        ``DWHeartbeat.is_degrading()``, which probes ``DIRECT_STREAMING``, and
        bt-2026-07-17-080507 showed the cost: the tier was bypassed on
        ``consecutive_failures=254`` from a surface it was PURPOSE-BUILT to
        avoid (complete_sync exists precisely because DW's SSE endpoint stalls
        post-accept — bt-2026-04-14-182446). Cross-contaminated telemetry
        neutralizes the opportunistic tier: SSE can be broken forever while
        completions serve fine.

        Bypasses on either failure dimension of ITS OWN surface — transport
        (502/timeout/auth) or inference (HTTP 200 with zero generated content,
        the reasoning-budget exhaustion class) — once the streak reaches the
        degrade threshold.

        Fail-soft: any fault → False → attempt DW normally (a telemetry bug
        must never strand the cheap tier)."""
        try:
            from backend.core.ouroboros.governance.dw_surface_health import (
                SurfaceKind,
                SurfaceHealthLedger,
                SurfaceVerdict,
            )

            rec = SurfaceHealthLedger().verdict_for(SurfaceKind.DIRECT_COMPLETION)
            if rec is None:
                return False  # never probed → attempt DW (no verdict is not a verdict)
            if rec.verdict is SurfaceVerdict.HEALTHY:
                return False
            if int(rec.consecutive_failures) >= _dw_completion_degrade_streak():
                logger.info(
                    "[DreamEngine] DW-RT tier BYPASSED — DIRECT_COMPLETION "
                    "surface %s (consecutive_failures=%d, diag=%s); routing "
                    "straight to Claude (no timeout tax)",
                    rec.verdict.value, rec.consecutive_failures,
                    (rec.diagnostic or "")[:80],
                )
                return True
        except Exception:  # noqa: BLE001 — health telemetry is advisory
            logger.debug("[DreamEngine] DW health bypass probe cold", exc_info=True)
        return False

    async def _trace_dw_recovery_if_bypassed(self) -> None:
        """Close the ONE-WAY DOOR: probe a bypassed DW surface for recovery.

        A bypass suppresses organic ``complete_sync`` traffic, so the surface
        would never earn a fresh HEALTHY record and the bypass would be
        permanent. This fires the Synthetic Tracer ONLY while the surface is
        degraded — a healthy DW is never probed, so the cost is strictly
        proportional to the outage — and the tracer's generated-content
        assertion re-opens the tier the moment DW can truly serve again.

        Driven from the existing dream loop (no new daemon). NEVER raises."""
        try:
            if not self._dw_health_bypass():
                return  # healthy (or unprobed) → no tracer cost
            from backend.core.ouroboros.governance.dw_capacity_probe import (
                trace_direct_completion,
            )

            verdict = await trace_direct_completion(
                self._dw_provider,
                model=os.environ.get("JARVIS_DREAM_DW_MODEL", "").strip() or None,
            )
            logger.info(
                "[DreamEngine] DW recovery tracer → %s (bypass lifts "
                "automatically once the surface reports healthy)", verdict,
            )
        except Exception:  # noqa: BLE001 — recovery probing is advisory
            logger.debug("[DreamEngine] DW recovery tracer skipped", exc_info=True)

    @staticmethod
    def _dream_rt_timeout_s() -> float:
        """Per-tier RT budget (``JARVIS_DREAM_RT_TIMEOUT_S``, default 90s —
        sized to DW RT's measured ~67s p50 TTFT with headroom). Never raises."""
        try:
            return max(5.0, float(os.environ.get("JARVIS_DREAM_RT_TIMEOUT_S", "90")))
        except (TypeError, ValueError):
            return 90.0

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response from any inference provider (the Tier-0/1/2
        cascade return boundary).

        Structural integrity FIRST: a payload severed mid-transmission — the
        edge case the 600s Aegis read ceiling opened — is rejected as truncation
        and routed to the dream's no-blueprint failure path with DISTINCT
        telemetry, never silently brace-closed into a corrupt blueprint that
        would flow into the governed loop. Reuses the shared truncation-aware
        validator + the existing deterministic repair (DRY) in place of the
        ad-hoc fence-strip / brace-extraction this replaces.
        """
        from backend.core.ouroboros.governance.payload_integrity import (
            PayloadTruncationError, validate_json_payload,
        )
        try:
            return validate_json_payload(raw)
        except PayloadTruncationError as exc:
            logger.warning(
                "[DreamEngine] inference payload TRUNCATED (severed mid-stream) "
                "— routed to failure lifecycle, NOT parsed: %s", exc.detail,
            )
            return None
        except Exception as exc:  # noqa: BLE001 — JSONDecodeError / non-object / etc.
            logger.debug("[DreamEngine] inference payload unparseable: %s", exc)
            return None

    def _parse_blueprint_result(
        self,
        blueprint_id: str,
        candidate: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Optional[ImprovementBlueprint]:
        """Parse the J-Prime response into an ImprovementBlueprint."""
        try:
            return ImprovementBlueprint(
                blueprint_id=blueprint_id,
                title=result.get("title", "Untitled improvement"),
                description=result.get("description", ""),
                category=result.get("category", candidate.get("prompt_family", "general")),
                priority_score=float(result.get("priority_score", 0.5)),
                target_files=tuple(result.get("target_files", ())),
                estimated_effort=result.get("estimated_effort", "small"),
                estimated_cost_usd=float(result.get("estimated_cost_usd", 0.01)),
                repo=candidate.get("repo", "jarvis"),
                repo_sha=candidate["repo_sha"],
                computed_at_utc=datetime.now(timezone.utc).isoformat(),
                ttl_hours=self._config.dream_blueprint_ttl_hours,
                model_used=result.get("_inference_model", candidate.get("model_class", "unknown")),
                policy_hash=candidate["policy_hash"],
                oracle_neighborhood=result.get("oracle_neighborhood", {}),
                suggested_approach=result.get("suggested_approach", ""),
                risk_assessment=result.get("risk_assessment", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "[DreamEngine] Failed to parse blueprint from J-Prime result: %s",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Write all blueprints and completed keys to disk."""
        try:
            # Persist each blueprint individually
            for bp_id, bp in self._blueprints.items():
                bp_path = self._persistence_dir / f"blueprint_{bp_id}.json"
                bp_path.write_text(
                    json.dumps(_blueprint_to_dict(bp), indent=2),
                    encoding="utf-8",
                )

            # Persist completed keys
            keys_path = self._persistence_dir / "job_keys.json"
            keys_path.write_text(
                json.dumps(sorted(self._completed_keys), indent=2),
                encoding="utf-8",
            )

            # Persist metrics via tracker
            metrics_path = self._persistence_dir / "metrics.json"
            self._metrics_tracker.persist(metrics_path)

            logger.debug(
                "[DreamEngine] Persisted %d blueprints, %d keys",
                len(self._blueprints),
                len(self._completed_keys),
            )
        except OSError as exc:
            logger.error("[DreamEngine] Failed to persist state: %s", exc)

    def _load_state(self) -> None:
        """Restore blueprints and completed keys from disk."""
        # Load blueprints
        try:
            for bp_file in self._persistence_dir.glob("blueprint_*.json"):
                try:
                    data = json.loads(bp_file.read_text(encoding="utf-8"))
                    bp = _blueprint_from_dict(data)
                    self._blueprints[bp.blueprint_id] = bp
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning(
                        "[DreamEngine] Skipping corrupt blueprint file %s: %s",
                        bp_file.name,
                        exc,
                    )
        except OSError as exc:
            logger.warning("[DreamEngine] Failed to scan blueprint files: %s", exc)

        # Load completed keys
        keys_path = self._persistence_dir / "job_keys.json"
        if keys_path.exists():
            try:
                data = json.loads(keys_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._completed_keys = set(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[DreamEngine] Failed to load job keys: %s", exc,
                )

        # Load metrics if available
        metrics_path = self._persistence_dir / "metrics.json"
        if metrics_path.exists():
            try:
                restored = DreamMetricsTracker.load(metrics_path)
                # Merge counters into the active tracker
                restored_metrics = restored.get_metrics()
                self._metrics_tracker.record_compute_time(
                    restored_metrics.opportunistic_compute_minutes,
                )
            except Exception as exc:
                logger.debug(
                    "[DreamEngine] Failed to load metrics (non-fatal): %s", exc,
                )

        logger.info(
            "[DreamEngine] Loaded %d blueprints, %d completed keys from disk",
            len(self._blueprints),
            len(self._completed_keys),
        )
