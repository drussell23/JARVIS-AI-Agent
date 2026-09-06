"""Conception Proposal Bridge — ranked blueprints → intake router (Gap 3).

The final unification path. The conception value model (``conception_value_model``)
ranks self-conceived work by expected value; DreamEngine produces the conceived
work as ``ImprovementBlueprint``s. Until now nothing carried a *highly ranked*
blueprint into active execution — the ranked scores modulated one sensor's
confidence but no proposal reached the intake queue on its own. This bridge is
that missing translation layer, and NOTHING MORE:

  * **Event-driven** — it reacts to blueprint production (a DreamEngine
    ``register_blueprint_observer`` callback), never polls. On each event it
    evaluates the *current* top-N batch so the threshold is batch-relative.
  * **Dynamic threshold** — a proposal routes iff its EV clears
    ``max(ev_floor, batch_mean_EV)``. Distribution-relative; the only absolute is
    an env floor (default 0.5 = the neutral prior), so a COLD organism — whose
    every EV sits at neutral×cost < floor — routes nothing until real signal
    lifts a proposal above its peers AND above neutral. No hardcoded cutoff.
  * **DRY** — it consumes the value model's EMITTED ``ExpectedValue`` verbatim
    (never recomputes weights), builds the envelope with the existing
    ``make_envelope`` schema, submits through the existing
    ``UnifiedIntakeRouter.ingest``, and records to the existing
    ``proactive_proposal_surface`` ledger. It owns no queue.
  * **Bulletproof dedup** — a durable, TTL+capacity-bounded registry keyed by
    ``blueprint_id`` is the authoritative "already routed / in execution" guard
    (survives across events, unlike the router's 300s window); the envelope's
    ``dedup_key`` (via ``evidence['signature'] = blueprint_id``) is a second
    structural guard inside the router. A blueprint already routed is dropped.

Discipline: authority-free (no gate/policy/orchestrator imports), never raises,
default-OFF master flag (a new autonomous routing capability is opt-in).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CONCEPTION_BRIDGE_SCHEMA_VERSION = "conception_bridge.1"

_TRUTHY = ("1", "true", "yes", "on")

# Router ingest results that mean "the proposal is in the queue" — all of them
# imply the blueprint is now routed and must be deduped on subsequent events.
_ROUTED_RESULTS = frozenset({"enqueued", "pending_ack", "deduplicated"})


# ---------------------------------------------------------------------------
# Env surface
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# observers_armed lifecycle barrier (2026-07-17)
#
# Mirrors the A1-T2 router-ready valve (unified_intake_router.mark_router_ready
# / await_router_ready): a process-global asyncio.Event the DreamEngine awaits
# before its first dream, so no inference compute is burned producing a
# blueprint that nothing can route yet. Set by IntakeLayerService once the
# conception bridge's blueprint observer is registered (or on any arm-skip path,
# so a disabled/absent bridge releases the barrier instead of wedging dreaming).
# ---------------------------------------------------------------------------

_observers_armed_event: Optional["asyncio.Event"] = None
_observers_armed_lock = threading.Lock()


def _get_observers_armed_event() -> "asyncio.Event":
    """Lazily create the barrier Event on the running loop (asyncio.Event binds
    to the loop at construction; created on first touch inside the async boot)."""
    global _observers_armed_event
    with _observers_armed_lock:
        if _observers_armed_event is None:
            import asyncio as _asyncio
            _observers_armed_event = _asyncio.Event()
        return _observers_armed_event


def mark_observers_armed() -> None:
    """Signal that the conception observer is armed (or that arming was skipped
    — a disabled bridge has no observer, so the barrier must still release).
    Idempotent; NEVER raises."""
    try:
        _get_observers_armed_event().set()
    except Exception:  # noqa: BLE001
        logger.debug("[ConceptionBridge] mark_observers_armed skipped", exc_info=True)


def observers_are_armed() -> bool:
    try:
        return _get_observers_armed_event().is_set()
    except Exception:  # noqa: BLE001
        return False


async def await_observers_armed(timeout_s: float) -> bool:
    """Await the observers-armed barrier, bounded by ``timeout_s``. Returns True
    if armed within the window, False on timeout (caller proceeds — the atomic
    register-and-drain still reconciles anything produced meanwhile). NEVER
    raises."""
    import asyncio as _asyncio
    try:
        await _asyncio.wait_for(_get_observers_armed_event().wait(), timeout=timeout_s)
        return True
    except _asyncio.TimeoutError:
        return False
    except Exception:  # noqa: BLE001
        return False


def _reset_observers_armed_for_tests() -> None:
    global _observers_armed_event
    with _observers_armed_lock:
        _observers_armed_event = None


def master_enabled() -> bool:
    """``JARVIS_CONCEPTION_BRIDGE_ENABLED`` (default FALSE — opt-in)."""
    return os.environ.get(
        "JARVIS_CONCEPTION_BRIDGE_ENABLED", "false",
    ).strip().lower() in _TRUTHY


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _ev_floor() -> float:
    """Absolute EV floor below which nothing routes (default 0.5 = neutral)."""
    return min(1.0, max(0.0, _envf("JARVIS_CONCEPTION_BRIDGE_EV_FLOOR", 0.5)))


def _max_per_route() -> int:
    return max(1, _envi("JARVIS_CONCEPTION_BRIDGE_MAX_PER_ROUTE", 3))


def _dedup_ttl_s() -> float:
    return max(60.0, _envf("JARVIS_CONCEPTION_BRIDGE_DEDUP_TTL_S", 3600.0))


def _dedup_capacity() -> int:
    return max(16, _envi("JARVIS_CONCEPTION_BRIDGE_DEDUP_CAPACITY", 512))


def _top_n() -> int:
    return max(1, _envi("JARVIS_CONCEPTION_BRIDGE_TOP_N", 5))


def _incubator_max() -> int:
    """IncubationStore capacity — bounded so a static repo can't grow the
    sub-threshold backlog without limit (``JARVIS_CONCEPTION_INCUBATOR_SIZE``,
    default 50)."""
    return max(1, _envi("JARVIS_CONCEPTION_INCUBATOR_SIZE", 50))


def _incubator_max_attempts() -> int:
    """Re-evaluation attempts before a chronically sub-threshold blueprint is
    retired from incubation (``JARVIS_CONCEPTION_INCUBATOR_MAX_ATTEMPTS``,
    default 20). Prevents an unroutable idea from re-scoring forever."""
    return max(1, _envi("JARVIS_CONCEPTION_INCUBATOR_MAX_ATTEMPTS", 20))


def _decision_ring_size() -> int:
    return max(1, _envi("JARVIS_CONCEPTION_DECISION_RING_SIZE", 50))


# ---------------------------------------------------------------------------
# Outcome artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingOutcome:
    """One blueprint's disposition through the bridge — the structured
    telemetry artifact (Mandate 1). Carries the full EV matrix so an operator
    reads WHY a proposal was routed or declined, never reconstructs it."""

    blueprint_id: str
    ev: float
    scope: str
    threshold: float
    routed: bool
    reason: str            # "routed" | "below_threshold" | "incubated" | "duplicate" | "ingest_<result>" | "error"
    ingest_result: str = ""
    # EV matrix breakdown (the axes conception_value_model composed).
    substance: float = 0.0
    feasibility: float = 0.0
    alignment: float = 0.0
    incubation_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class _IncubationRecord:
    """A sub-threshold blueprint preserved for re-evaluation (Mandate 2). Not a
    new datastore — it references the existing blueprint artifact and tags it
    with an incubating state vector (ev + temporal metadata)."""

    blueprint: Any
    ev: Any
    first_incubated_unix: float
    attempts: int = 1


# ---------------------------------------------------------------------------
# Durable dedup registry — authoritative "already routed / in execution" guard
# ---------------------------------------------------------------------------


class _RoutedRegistry:
    """Bounded, TTL-evicted set of routed ``blueprint_id``s. Thread-safe."""

    def __init__(self) -> None:
        self._items: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        ttl = _dedup_ttl_s()
        cap = _dedup_capacity()
        # TTL eviction (oldest-first; insertion order == time order).
        while self._items:
            bid, ts = next(iter(self._items.items()))
            if now - ts > ttl:
                self._items.pop(bid, None)
            else:
                break
        # Capacity eviction.
        while len(self._items) > cap:
            self._items.popitem(last=False)

    def is_routed(self, blueprint_id: str, now: float) -> bool:
        with self._lock:
            self._evict(now)
            return blueprint_id in self._items

    def mark(self, blueprint_id: str, now: float) -> None:
        with self._lock:
            self._items.pop(blueprint_id, None)   # refresh insertion order
            self._items[blueprint_id] = now
            self._evict(now)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def reset(self) -> None:
        with self._lock:
            self._items.clear()


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


class ConceptionProposalBridge:
    """Event-driven translation layer: ranked blueprints → intake router."""

    def __init__(
        self,
        *,
        value_scorer: Optional[Callable[[Any], Any]] = None,
        ledger_emit: Optional[Callable[..., Any]] = None,
        registry: Optional[_RoutedRegistry] = None,
    ) -> None:
        self._score = value_scorer          # default resolved lazily (DRY)
        self._ledger_emit = ledger_emit     # default resolved lazily
        self._registry = registry or _RoutedRegistry()
        # observability counters
        self._routed_total = 0
        self._dropped_duplicate = 0
        self._below_threshold = 0
        self._incubated_total = 0
        self._graduated_total = 0
        self._errors = 0
        self._events = 0
        # IncubationStore (Mandate 2): sub-threshold blueprints preserved for
        # re-evaluation as the repo state / EV inputs evolve. Bounded ring —
        # drop-oldest at capacity, no unbounded growth (Mandate 4).
        self._incubating: "OrderedDict[str, _IncubationRecord]" = OrderedDict()
        # Structured decision telemetry ring (Mandate 1) — the last N routing
        # decisions with full EV matrix, surfaced via snapshot() → the existing
        # /observability/conception endpoint (Mandate 3, no new endpoint).
        self._decisions: "Deque[Dict[str, Any]]" = deque(maxlen=_decision_ring_size())

    # -- axis resolution (lazy, DRY) --------------------------------------

    async def _score_blueprint_offloaded(self, bp: Any) -> Any:
        """Score on a worker thread. The value model's alignment axis embeds
        the description (a multi-second ONNX run); ``route`` is a coroutine
        and the index refuses to embed on the loop thread, so the scoring
        must leave the loop to keep its alignment axis. Same substrate the
        index's own offloaded scorers use; bare thread when it is absent."""
        try:
            from backend.core.ouroboros.governance.cooperative_fs_io import (  # noqa: E501,PLC0415
                offload,
                is_offload_error,
            )
            res = await offload(self._score_blueprint, bp, cpu_bound=False)
            if is_offload_error(res):
                raise RuntimeError(f"offload failed: {res!r}")
            return res
        except ImportError:
            import asyncio  # noqa: PLC0415
            return await asyncio.to_thread(self._score_blueprint, bp)

    def _score_blueprint(self, bp: Any) -> Any:
        if self._score is not None:
            return self._score(bp)
        from backend.core.ouroboros.governance.conception_value_model import (
            score_blueprint,
        )
        return score_blueprint(bp)

    def _emit_to_ledger(self, bp: Any, ev: Any) -> None:
        try:
            emit = self._ledger_emit
            if emit is None:
                from backend.core.ouroboros.governance.proactive_proposal_surface import (
                    emit_proposal,
                )
                emit = emit_proposal
            emit(
                kind="opportunity",
                signal_source="auto_proposed",
                summary=str(getattr(bp, "title", "") or getattr(bp, "description", "")),
                rationale=str(getattr(ev, "rationale", "")),
                priority_hint=float(getattr(ev, "ev", 0.5)),
            )
        except Exception:  # noqa: BLE001 — ledger record is best-effort audit
            logger.debug("[ConceptionBridge] ledger emit skipped", exc_info=True)

    # -- structured telemetry + incubation --------------------------------

    def _outcome(
        self, bp: Any, ev: Any, *, threshold: float, routed: bool, reason: str,
        ingest_result: str = "", attempts: int = 0,
    ) -> RoutingOutcome:
        """Build the structured RoutingOutcome (full EV matrix) AND emit it as
        deterministic telemetry — a structured record into the decision ring
        plus one structured log line. Mandate 1: no forensic reconstruction."""
        oc = RoutingOutcome(
            blueprint_id=str(getattr(bp, "blueprint_id", "") or ""),
            ev=float(getattr(ev, "ev", 0.0)),
            scope=str(getattr(ev, "scope", "")),
            threshold=float(threshold), routed=routed, reason=reason,
            ingest_result=ingest_result,
            substance=float(getattr(ev, "substance", 0.0)),
            feasibility=float(getattr(ev, "feasibility", 0.0)),
            alignment=float(getattr(ev, "alignment", 0.0)),
            incubation_attempts=int(attempts),
        )
        self._decisions.append(oc.to_dict())
        logger.info(
            "[ConceptionBridge] decision blueprint=%s disposition=%s "
            "ev=%.4f threshold=%.4f alignment=%.3f substance=%.3f "
            "feasibility=%.3f attempts=%d%s",
            oc.blueprint_id[:12], reason, oc.ev, oc.threshold, oc.alignment,
            oc.substance, oc.feasibility, oc.incubation_attempts,
            f" ingest={ingest_result}" if ingest_result else "",
        )
        return oc

    def _incubate(self, bp: Any, ev: Any, now: float) -> int:
        """Preserve a sub-threshold blueprint for re-evaluation. Returns the
        attempt count. Bounded ring (drop-oldest); retires a blueprint that has
        stayed unroutable past the attempt ceiling."""
        bid = str(getattr(bp, "blueprint_id", "") or "")
        if not bid:
            return 0
        rec = self._incubating.get(bid)
        if rec is None:
            self._incubated_total += 1
            self._incubating[bid] = _IncubationRecord(
                blueprint=bp, ev=ev, first_incubated_unix=now, attempts=1)
            attempts = 1
        else:
            rec.attempts += 1
            rec.ev = ev            # refresh with the latest score
            rec.blueprint = bp
            self._incubating.move_to_end(bid)
            attempts = rec.attempts
            if attempts >= _incubator_max_attempts():
                self._incubating.pop(bid, None)   # retire the chronically-cold
        # capacity bound
        while len(self._incubating) > _incubator_max():
            self._incubating.popitem(last=False)
        return attempts

    def _graduate(self, bid: str) -> None:
        """A blueprint cleared the threshold on a later pass — remove it from
        incubation."""
        if self._incubating.pop(bid, None) is not None:
            self._graduated_total += 1

    # -- core reactor -----------------------------------------------------

    async def route(
        self,
        blueprints: Sequence[Any],
        router: Any,
        *,
        now: Optional[float] = None,
    ) -> List[RoutingOutcome]:
        """Route the batch's above-threshold, non-duplicate blueprints into the
        intake router. Event-reactor — call this on blueprint production, not on
        a timer. Never raises; returns per-blueprint outcomes."""
        if not master_enabled() or router is None:
            return []
        now = now if now is not None else time.time()
        self._events += 1
        try:
            # 1. Assemble the evaluation batch = freshly-produced blueprints
            #    UNION the incubating ones (Mandate 2: temporally-misaligned
            #    proposals get re-scored as repo state evolves — "truly adaptive"
            #    memory). Incoming duplicates of an incubating id take precedence.
            candidates: "OrderedDict[str, Any]" = OrderedDict()
            for bp in (blueprints or ()):
                bid = str(getattr(bp, "blueprint_id", "") or "")
                tfs = tuple(getattr(bp, "target_files", ()) or ())
                if not bid or not tfs:
                    continue
                candidates[bid] = bp
            for bid, rec in list(self._incubating.items()):
                candidates.setdefault(bid, rec.blueprint)

            # 2. Score (consume the value model's EMITTED EV; never recompute).
            scored: List[Tuple[Any, Any]] = []
            for bp in candidates.values():
                try:
                    ev = await self._score_blueprint_offloaded(bp)
                except Exception:  # noqa: BLE001
                    self._errors += 1
                    continue
                scored.append((bp, ev))

            # 3. Drop already-routed / in-execution (mandate 4 — durable guard).
            fresh: List[Tuple[Any, Any]] = []
            outcomes: List[RoutingOutcome] = []
            for (bp, ev) in scored:
                if self._registry.is_routed(str(bp.blueprint_id), now):
                    self._dropped_duplicate += 1
                    self._graduate(str(bp.blueprint_id))  # already in-flight: stop incubating
                    outcomes.append(self._outcome(
                        bp, ev, threshold=0.0, routed=False, reason="duplicate"))
                else:
                    fresh.append((bp, ev))
            if not fresh:
                return outcomes

            # 4. Dynamic, batch-relative threshold. No hardcoded cutoff — the
            #    quality bar rises with the batch mean and never drops below the
            #    neutral floor.
            evs = [float(ev.ev) for (_bp, ev) in fresh]
            threshold = max(_ev_floor(), sum(evs) / len(evs))

            survivors = sorted(
                [(bp, ev) for (bp, ev) in fresh if float(ev.ev) >= threshold],
                key=lambda pe: float(pe[1].ev), reverse=True,
            )[:_max_per_route()]
            survivor_ids = {str(bp.blueprint_id) for (bp, _ev) in survivors}

            # 5. Below-threshold blueprints INCUBATE (Mandate 2) rather than
            #    being discarded — preserved for re-evaluation on a later event.
            for (bp, ev) in fresh:
                if str(bp.blueprint_id) not in survivor_ids:
                    self._below_threshold += 1
                    attempts = self._incubate(bp, ev, now)
                    outcomes.append(self._outcome(
                        bp, ev, threshold=threshold, routed=False,
                        reason="incubated", attempts=attempts))

            # 6. Translate + ingest each survivor (graduates it out of incubation).
            for (bp, ev) in survivors:
                outcomes.append(await self._route_one(bp, ev, router, threshold, now))
            return outcomes
        except Exception:  # noqa: BLE001 — reactor never leaks into the loop
            self._errors += 1
            logger.debug("[ConceptionBridge] route pass faulted", exc_info=True)
            return []

    async def _route_one(
        self, bp: Any, ev: Any, router: Any, threshold: float, now: float,
    ) -> RoutingOutcome:
        bid = str(bp.blueprint_id)
        try:
            from backend.core.ouroboros.governance.intake.intent_envelope import (
                make_envelope,
            )

            envelope = make_envelope(
                source="auto_proposed",
                description=str(getattr(bp, "description", "") or getattr(bp, "title", "")),
                target_files=tuple(str(t) for t in bp.target_files),
                repo=str(getattr(bp, "repo", "") or ""),
                confidence=max(0.0, min(1.0, float(ev.ev))),
                urgency="low",
                evidence={
                    "sensor": "conception_bridge",
                    # `signature` drives the router's own dedup_key — deterministic
                    # per blueprint, a second structural guard against duplicates.
                    "signature": bid,
                    "blueprint_id": bid,
                    "category": str(getattr(bp, "category", "")),
                    "ev": round(float(ev.ev), 4),
                    "threshold": round(float(threshold), 4),
                    "conception_value": ev.to_dict() if hasattr(ev, "to_dict") else {},
                },
                requires_human_ack=True,  # proactive conception surfaces for ack
            )
            result = str(await router.ingest(envelope))
            # Mark routed on ANY in-queue result (incl. router-side dedup).
            if result in _ROUTED_RESULTS:
                self._registry.mark(bid, now)
                self._routed_total += 1
                self._graduate(bid)   # cleared the bar — leave incubation
                self._emit_to_ledger(bp, ev)
                return self._outcome(
                    bp, ev, threshold=threshold, routed=True, reason="routed",
                    ingest_result=result)
            return self._outcome(
                bp, ev, threshold=threshold, routed=False,
                reason=f"ingest_{result}", ingest_result=result)
        except Exception:  # noqa: BLE001
            self._errors += 1
            logger.debug("[ConceptionBridge] ingest faulted for %s", bid, exc_info=True)
            return self._outcome(
                bp, ev, threshold=threshold, routed=False, reason="error")

    # -- event wiring -----------------------------------------------------

    def make_observer(
        self, router: Any, blueprint_source: Callable[[], Sequence[Any]],
    ) -> Callable[[Any], Any]:
        """Return an async DreamEngine blueprint observer. On each
        blueprint-computed event it pulls the CURRENT batch (so the threshold is
        batch-relative) and routes it. Fail-soft."""
        async def _observer(_blueprint: Any) -> None:
            try:
                batch = list(blueprint_source() or ())
            except Exception:  # noqa: BLE001
                batch = []
            if batch:
                await self.route(batch, router)
        return _observer

    # -- observability ----------------------------------------------------

    def snapshot(self) -> dict:
        # Incubation projection — the tagged 'incubating' state vectors (Mandate
        # 2), surfaced through the EXISTING /observability/conception emission
        # (Mandate 3), never a bespoke endpoint.
        incubating = [
            {
                "blueprint_id": bid,
                "ev": round(float(getattr(rec.ev, "ev", 0.0)), 4),
                "alignment": round(float(getattr(rec.ev, "alignment", 0.0)), 3),
                "substance": round(float(getattr(rec.ev, "substance", 0.0)), 3),
                "feasibility": round(float(getattr(rec.ev, "feasibility", 0.0)), 3),
                "attempts": int(rec.attempts),
                "first_incubated_unix": round(float(rec.first_incubated_unix), 2),
            }
            for bid, rec in self._incubating.items()
        ]
        return {
            "schema_version": CONCEPTION_BRIDGE_SCHEMA_VERSION,
            "enabled": master_enabled(),
            "ev_floor": _ev_floor(),
            "max_per_route": _max_per_route(),
            "dedup_ttl_s": _dedup_ttl_s(),
            "dedup_registry_size": len(self._registry),
            "incubator_size": len(self._incubating),
            "incubator_capacity": _incubator_max(),
            "counters": {
                "events": self._events,
                "routed_total": self._routed_total,
                "dropped_duplicate": self._dropped_duplicate,
                "below_threshold": self._below_threshold,
                "incubated_total": self._incubated_total,
                "graduated_total": self._graduated_total,
                "errors": self._errors,
            },
            # Structured decision telemetry (Mandate 1) — newest last.
            "recent_decisions": list(self._decisions),
            "incubating": incubating,
        }

    def reset_for_tests(self) -> None:
        self._registry.reset()
        self._incubating.clear()
        self._decisions.clear()
        self._routed_total = self._dropped_duplicate = 0
        self._below_threshold = self._errors = self._events = 0
        self._incubated_total = self._graduated_total = 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_default_bridge: Optional[ConceptionProposalBridge] = None
_singleton_lock = threading.Lock()


def get_default_bridge() -> ConceptionProposalBridge:
    global _default_bridge
    with _singleton_lock:
        if _default_bridge is None:
            _default_bridge = ConceptionProposalBridge()
        return _default_bridge


def reset_bridge_for_tests() -> None:
    global _default_bridge
    with _singleton_lock:
        if _default_bridge is not None:
            _default_bridge.reset_for_tests()
        _default_bridge = None


def snapshot() -> dict:
    """Module-level observability projection (safe when disabled)."""
    try:
        return get_default_bridge().snapshot()
    except Exception:  # noqa: BLE001
        return {"schema_version": CONCEPTION_BRIDGE_SCHEMA_VERSION, "enabled": False}
