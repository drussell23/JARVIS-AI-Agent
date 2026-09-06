"""NarrativeChannel — the model's voice surface for proactive ops.
==================================================================

Slice 1 of the **Gap #6 closure arc** (proactive narrative).

Root problem
------------

Claude Code's UX feels alive because the assistant *narrates* its work
in continuous prose interleaved with tool calls. O+V is **proactive**
(sensors fire ops without operator input) so this need is even more
acute — the operator has no prior question to anchor context against.

Today the orchestrator captures the model's prose into structured
fields (rationales, plan blocks, postmortem analysis) but only streams
output during the GENERATE phase. PLAN reasoning, inter-tool-round
narrative, L2 repair planning text, and postmortem analysis prose all
get parsed into structured slots and stored — never flowing onto the
operator's screen.

This module supplies the **storage + streaming substrate** for a
"narrative channel": a parallel surface that captures the model's
voice across the whole op lifecycle and exposes it via:

  * Live streaming during emission (Slice 3 wires the renderer)
  * Bounded recovery via ``/expand n-N`` REPL verb (Slice 4)
  * SSE event for IDE consumption (Slice 4)

Substrate scope
---------------

* :class:`NarrativeFrame` frozen dataclass — one buffered narrative
  emission with phase + kind metadata
* Closed 7-value :class:`NarrativeKind` taxonomy (extended 2026-05-08
  §38.11-D adds :data:`DREAM` for DreamEngine prose)
* Closed 3-value :class:`FrameState` lifecycle (BUFFERING / COMMITTED
  / DISCARDED)
* :class:`NarrativeChannel` thread-safe FIFO ring with monotonic
  ``n-N`` refs (NEVER reused — mirrors :class:`BoundedBodyStore` /
  :class:`DiffArchive` / :class:`OpBlockBuffer` safety contracts)
* Streaming API: ``start_frame`` → ``append_token`` → ``commit`` /
  ``discard`` — same shape as token streaming so providers can reuse
  the existing fan-out callback (Slice 2 wiring).

Architectural reuse — zero duplication
---------------------------------------

* Same monotonic-ref + drop-oldest + active-index-pruning pattern from
  :class:`OpBlockBuffer` (Gap #3 Slice 2). The structural differences
  are in the FIELDS (phase + kind metadata) and the POLICY (terminal
  via ``commit`` vs ``discard``, no expand-state).
* :class:`StreamRenderer`'s ``on_token`` lifecycle (Slice 2 wiring
  forwards provider tokens to ``append_token`` so the existing
  16ms-batched Live widget can drive narrative rendering with no
  parallel rendering surface).
* House style: frozen dataclass + closed enum + ``schema_version`` +
  module-owned ``register_flags`` / ``register_shipped_invariants``
  for auto-discovery.

Authority boundary
------------------

* §1 deterministic — pure container; no LLM, no I/O on the hot path
* §7 fail-closed — every public method has documented degradation
  (unknown frame ref → ``None``, append on terminal frame → no-op);
  NEVER raises into the streaming hot path
* §8 observable — :class:`ChannelSnapshot` projection for
  observability surfaces
"""
from __future__ import annotations

import enum
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

logger = logging.getLogger("Ouroboros.NarrativeChannel")


# ===========================================================================
# Schema + env vocabulary
# ===========================================================================


NARRATIVE_CHANNEL_SCHEMA_VERSION: str = "narrative_channel.v1"


BUFFER_SIZE_ENV_VAR: str = "JARVIS_NARRATIVE_BUFFER_SIZE"


_DEFAULT_BUFFER_SIZE: int = 200
_MIN_BUFFER_SIZE: int = 1
_MAX_BUFFER_SIZE: int = 10_000


REF_PREFIX: str = "n-"


# ===========================================================================
# Closed taxonomies
# ===========================================================================


class NarrativeKind(str, enum.Enum):
    """Closed 7-value vocabulary for what the model is saying.

    Each kind maps to a distinct phase/context where the model emits
    natural-language prose. Closed taxonomy — adding a new kind
    requires a slice (the renderer in Slice 3 dispatches on this
    enum to choose a glyph/style).

    Extended to 7 values 2026-05-08 (§38.11-D Introspective Voice,
    merged with §39 #9): added :data:`DREAM` for DreamEngine
    speculative-improvement prose emitted during idle GPU.
    """

    INTENT = "intent"                       # op_started: "I'm going to fix X by..."
    PLAN_PROSE = "plan_prose"               # PLAN phase reasoning
    TOOL_PREAMBLE = "tool_preamble"         # "I'll read X first to understand Y"
    THINKING = "thinking"                   # extended-thinking REASONING_TOKEN content
    L2_REPAIR_PROSE = "l2_repair_prose"     # repair iteration narrative
    POSTMORTEM_PROSE = "postmortem_prose"   # failed-op analysis
    DREAM = "dream"                         # DreamEngine idle speculative blueprint prose

    @classmethod
    def coerce(cls, raw: object) -> "NarrativeKind":
        """Lenient parse — anything not recognized becomes
        :data:`THINKING` (the most generic). NEVER raises."""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            s = raw.strip().lower()
            for member in cls:
                if member.value == s:
                    return member
        return cls.THINKING


class FrameState(str, enum.Enum):
    """Closed 3-value frame lifecycle.

    ``BUFFERING`` is the only mutable state. ``COMMITTED`` means the
    model finished emitting; ``DISCARDED`` means the frame was
    abandoned (provider error / op cancelled / model returned no prose).
    """

    BUFFERING = "buffering"
    COMMITTED = "committed"
    DISCARDED = "discarded"

    @classmethod
    def coerce(cls, raw: object) -> "FrameState":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            s = raw.strip().lower()
            for member in cls:
                if member.value == s:
                    return member
        return cls.BUFFERING

    @property
    def is_terminal(self) -> bool:
        return self is not FrameState.BUFFERING


# ===========================================================================
# Frozen records
# ===========================================================================


@dataclass(frozen=True)
class NarrativeFrame:
    """One narrative emission.

    Frozen + hashable. Mutations replace-in-place via
    :func:`dataclasses.replace` from inside :meth:`NarrativeChannel.*`
    methods.

    Fields
    ------
    * ``ref`` — opaque ``n-N`` handle.
    * ``op_id`` — orchestrator op id (empty string for system-level
      narrative; rare).
    * ``phase`` — orchestrator phase string (e.g. ``"PLAN"``,
      ``"GENERATE"``, ``"L2_REPAIR"``). Stored as string for
      forward-compat with the orchestrator's enum.
    * ``kind`` — :class:`NarrativeKind` member.
    * ``provider`` — model provider id (``"claude"`` / ``"doubleword"``
      / ``"gcp-jprime"``). Empty for synthesized fallback preambles.
    * ``prose`` — accumulated text content (built up by ``append_token``
      while BUFFERING; finalized at ``commit``).
    * ``state`` — :class:`FrameState`.
    * ``started_at`` / ``terminal_at`` — ``time.monotonic()`` timestamps.
    """

    ref: str
    op_id: str
    phase: str
    kind: NarrativeKind
    provider: str
    prose: str
    state: FrameState
    started_at: float
    terminal_at: float
    schema_version: str = NARRATIVE_CHANNEL_SCHEMA_VERSION

    @property
    def char_count(self) -> int:
        return len(self.prose)

    @property
    def duration_s(self) -> float:
        if self.terminal_at <= 0.0:
            return 0.0
        return max(0.0, self.terminal_at - self.started_at)

    def to_dict(self, *, include_prose: bool = False) -> Dict[str, object]:
        d: Dict[str, object] = {
            "ref": self.ref,
            "op_id": self.op_id,
            "phase": self.phase,
            "kind": self.kind.value,
            "provider": self.provider,
            "state": self.state.value,
            "started_at": self.started_at,
            "terminal_at": self.terminal_at,
            "duration_s": self.duration_s,
            "char_count": self.char_count,
            "schema_version": self.schema_version,
        }
        if include_prose:
            d["prose"] = self.prose
        return d


@dataclass(frozen=True)
class ChannelSnapshot:
    """Read-only projection of the channel's state."""

    capacity: int
    size: int
    next_seq: int
    buffering_count: int
    committed_count: int
    discarded_count: int
    schema_version: str = NARRATIVE_CHANNEL_SCHEMA_VERSION

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, self.size / self.capacity)


# ===========================================================================
# Helpers
# ===========================================================================


def _read_capacity_from_env() -> int:
    raw = os.environ.get(BUFFER_SIZE_ENV_VAR, "").strip()
    if not raw:
        return _DEFAULT_BUFFER_SIZE
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BUFFER_SIZE
    if parsed < _MIN_BUFFER_SIZE:
        return _MIN_BUFFER_SIZE
    if parsed > _MAX_BUFFER_SIZE:
        return _MAX_BUFFER_SIZE
    return parsed


def _safe_str(raw: object) -> str:
    if raw is None:
        return ""
    try:
        return str(raw)
    except Exception:  # noqa: BLE001
        return ""


# ===========================================================================
# NarrativeChannel — the load-bearing class
# ===========================================================================


class NarrativeChannel:
    """Thread-safe bounded FIFO of :class:`NarrativeFrame` records.

    Active-frame index
    ------------------
    For O(1) ``append_token`` lookup, an index ``self._active_keys ->
    ref`` maps the composite key ``(op_id, phase, kind)`` to the
    current BUFFERING ref. Multiple parallel frames per op are
    supported as long as their (op, phase, kind) tuples differ —
    matches the real-world case where Claude emits THINKING tokens
    interleaved with PLAN_PROSE.

    Eviction
    --------
    Drop-oldest at capacity. If an evicted frame was still BUFFERING,
    the active-key entry is also pruned (preventing concurrent
    ``append_token`` from misrouting to a different op's frame).

    Thread safety
    -------------
    Single :class:`threading.RLock` serializes all operations.
    """

    def __init__(self, *, capacity: Optional[int] = None) -> None:
        cap = (
            _read_capacity_from_env()
            if capacity is None
            else max(_MIN_BUFFER_SIZE, min(_MAX_BUFFER_SIZE, int(capacity)))
        )
        self._capacity: int = cap
        self._items: "OrderedDict[str, NarrativeFrame]" = OrderedDict()
        # composite key (op_id, phase, kind) → ref
        self._active_keys: Dict[Tuple[str, str, str], str] = {}
        self._next_seq: int = 1
        self._lock = threading.RLock()
        # Commit subscribers. The channel is a RING the operator pages back
        # through; a surface that wants to show a frame AS IT LANDS needs a
        # signal, and until this existed each surface had to find and
        # render frames by hand -- which is why the default transport
        # rendered none of them. Notified OUTSIDE the lock, one listener's
        # fault isolated from the next. Deduplicated by EQUALITY: a bound
        # method is a new object on every attribute access, so identity
        # would let the same subscriber in twice and never let it out.
        self._commit_listeners: list = []

    # ---- commit subscribers --------------------------------------------

    def add_commit_listener(self, fn: object) -> bool:
        """Subscribe ``fn(frame)`` to every frame that COMMITS (including
        ``emit_complete``). Returns ``True`` when newly added. NEVER raises."""
        if not callable(fn):
            return False
        with self._lock:
            if any(existing == fn for existing in self._commit_listeners):
                return False
            self._commit_listeners.append(fn)
        return True

    def remove_commit_listener(self, fn: object) -> bool:
        """Unsubscribe; ``True`` when it was subscribed. NEVER raises."""
        with self._lock:
            for i, existing in enumerate(self._commit_listeners):
                if existing == fn:
                    del self._commit_listeners[i]
                    return True
        return False

    def _notify_commit(self, frame: NarrativeFrame) -> None:
        with self._lock:
            listeners = tuple(self._commit_listeners)
        for fn in listeners:
            try:
                fn(frame)
            except Exception:  # noqa: BLE001 -- one listener never silences another
                logger.debug(
                    "[NarrativeChannel] commit listener raised", exc_info=True,
                )

    # ---- introspection -----------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> ChannelSnapshot:
        with self._lock:
            buf = comm = disc = 0
            for entry in self._items.values():
                if entry.state is FrameState.BUFFERING:
                    buf += 1
                elif entry.state is FrameState.COMMITTED:
                    comm += 1
                elif entry.state is FrameState.DISCARDED:
                    disc += 1
            return ChannelSnapshot(
                capacity=self._capacity,
                size=len(self._items),
                next_seq=self._next_seq,
                buffering_count=buf,
                committed_count=comm,
                discarded_count=disc,
            )

    # ---- mutating API -------------------------------------------------

    def start_frame(
        self,
        *,
        op_id: object,
        phase: object,
        kind: object,
        provider: object = "",
    ) -> Optional[NarrativeFrame]:
        """Begin buffering a new narrative frame. Returns the new
        :class:`NarrativeFrame` (state=BUFFERING) or the existing
        frame if one is already active for ``(op_id, phase, kind)``
        — start_frame is idempotent for in-flight emissions.

        Empty / non-string ``op_id`` / ``phase`` is allowed (system-
        level narrative). NEVER raises.
        """
        op_safe = _safe_str(op_id)
        phase_safe = _safe_str(phase)
        kind_enum = NarrativeKind.coerce(kind)
        provider_safe = _safe_str(provider)
        composite = (op_safe, phase_safe, kind_enum.value)

        with self._lock:
            existing_ref = self._active_keys.get(composite)
            if existing_ref is not None:
                existing = self._items.get(existing_ref)
                if existing is not None and existing.state is FrameState.BUFFERING:
                    return existing
            ref = f"{REF_PREFIX}{self._next_seq}"
            # Transcript spine: one ordered sequence across all four
            # namespaces, so "what happened next" has an answer and eviction
            # is uniform. Fail-soft — never blocks admission.
            try:
                from backend.core.ouroboros.battle_test.transcript_spine import (
                    record_event,
                )
                record_event("narrative", ref)
            except Exception:  # noqa: BLE001
                pass
            self._next_seq += 1
            frame = NarrativeFrame(
                ref=ref,
                op_id=op_safe,
                phase=phase_safe,
                kind=kind_enum,
                provider=provider_safe,
                prose="",
                state=FrameState.BUFFERING,
                started_at=time.monotonic(),
                terminal_at=0.0,
            )
            self._items[ref] = frame
            self._active_keys[composite] = ref
            self._evict_if_needed()
            return frame

    def append_token(
        self,
        *,
        op_id: object,
        phase: object,
        kind: object,
        token: object,
    ) -> bool:
        """Append a token chunk to the active frame for
        ``(op_id, phase, kind)``. Returns ``True`` on success,
        ``False`` if no active frame exists (e.g. terminal already)
        or the inputs are non-string. NEVER raises.

        Concurrent append from multiple producers is safe — the
        composite-key lookup serializes through the RLock and the
        replace-in-place semantics preserve insertion order.
        """
        op_safe = _safe_str(op_id)
        phase_safe = _safe_str(phase)
        kind_enum = NarrativeKind.coerce(kind)
        token_safe = _safe_str(token)
        if not token_safe:
            return False
        composite = (op_safe, phase_safe, kind_enum.value)
        with self._lock:
            ref = self._active_keys.get(composite)
            if ref is None:
                return False
            current = self._items.get(ref)
            if current is None or current.state is not FrameState.BUFFERING:
                return False
            updated = replace(current, prose=current.prose + token_safe)
            self._items[ref] = updated
            return True

    def commit(
        self,
        *,
        op_id: object,
        phase: object,
        kind: object,
    ) -> Optional[NarrativeFrame]:
        """Mark the active frame for ``(op_id, phase, kind)`` as
        COMMITTED. Returns the committed frame, or ``None`` for
        unknown/already-terminal. NEVER raises."""
        op_safe = _safe_str(op_id)
        phase_safe = _safe_str(phase)
        kind_enum = NarrativeKind.coerce(kind)
        composite = (op_safe, phase_safe, kind_enum.value)
        with self._lock:
            ref = self._active_keys.pop(composite, None)
            if ref is None:
                return None
            current = self._items.get(ref)
            if current is None:
                return None
            if current.state is not FrameState.BUFFERING:
                return current
            updated = replace(
                current,
                state=FrameState.COMMITTED,
                terminal_at=time.monotonic(),
            )
            self._items[ref] = updated
        self._notify_commit(updated)
        return updated

    def discard(
        self,
        *,
        op_id: object,
        phase: object,
        kind: object,
    ) -> Optional[NarrativeFrame]:
        """Mark the active frame DISCARDED (provider error / cancel /
        empty model output). Returns the discarded frame or ``None``."""
        op_safe = _safe_str(op_id)
        phase_safe = _safe_str(phase)
        kind_enum = NarrativeKind.coerce(kind)
        composite = (op_safe, phase_safe, kind_enum.value)
        with self._lock:
            ref = self._active_keys.pop(composite, None)
            if ref is None:
                return None
            current = self._items.get(ref)
            if current is None:
                return None
            if current.state is not FrameState.BUFFERING:
                return current
            updated = replace(
                current,
                state=FrameState.DISCARDED,
                terminal_at=time.monotonic(),
            )
            self._items[ref] = updated
            return updated

    def emit_complete(
        self,
        *,
        op_id: object,
        phase: object,
        kind: object,
        prose: object,
        provider: object = "",
    ) -> Optional[NarrativeFrame]:
        """One-shot helper: start_frame + single append + commit.

        Used by Slice 2's deterministic preamble synthesis where the
        complete prose is known up front (no streaming). Returns the
        committed frame or ``None`` on error.
        """
        prose_safe = _safe_str(prose)
        if not prose_safe:
            return None
        frame = self.start_frame(
            op_id=op_id, phase=phase, kind=kind, provider=provider,
        )
        if frame is None:
            return None
        self.append_token(op_id=op_id, phase=phase, kind=kind, token=prose_safe)
        return self.commit(op_id=op_id, phase=phase, kind=kind)

    def clear(self) -> None:
        """Drop all frames. Counter is NOT reset."""
        with self._lock:
            self._items.clear()
            self._active_keys.clear()

    # ---- query API ---------------------------------------------------

    def lookup(self, ref: object) -> Optional[NarrativeFrame]:
        if not isinstance(ref, str):
            return None
        with self._lock:
            return self._items.get(ref)

    def list_recent(self, limit: int = 10) -> Tuple[NarrativeFrame, ...]:
        """Newest → oldest, capped by ``limit``."""
        if not isinstance(limit, int) or limit <= 0:
            return ()
        with self._lock:
            entries = list(self._items.values())
        entries.reverse()
        return tuple(entries[:limit])

    def find_by_op_id(self, op_id: object) -> Tuple[NarrativeFrame, ...]:
        """All frames (oldest → newest) for ``op_id``."""
        if not isinstance(op_id, str) or not op_id:
            return ()
        with self._lock:
            return tuple(
                e for e in self._items.values() if e.op_id == op_id
            )

    def find_by_kind(self, kind: object) -> Tuple[NarrativeFrame, ...]:
        kind_enum = NarrativeKind.coerce(kind)
        with self._lock:
            return tuple(
                e for e in self._items.values() if e.kind is kind_enum
            )

    def all_refs(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._items.keys())

    def active_keys(self) -> Tuple[Tuple[str, str, str], ...]:
        """Currently-buffering composite keys."""
        with self._lock:
            return tuple(self._active_keys.keys())

    # ---- internals ----------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Drop oldest entries until back within capacity. Prune the
        active-key index when the evicted frame was still BUFFERING."""
        while len(self._items) > self._capacity:
            ref, evicted = self._items.popitem(last=False)
            if evicted.state is FrameState.BUFFERING:
                composite = (evicted.op_id, evicted.phase, evicted.kind.value)
                if self._active_keys.get(composite) == ref:
                    self._active_keys.pop(composite, None)

    # ---- read-only query API (Phase 2 read-API extension, 2026-05-07) ----

    def frames_by_op_kind(
        self,
        *,
        op_id: str,
        kind: NarrativeKind,
        states: Optional[Tuple[FrameState, ...]] = None,
    ) -> Tuple["NarrativeFrame", ...]:
        """Return frames matching ``(op_id, kind)`` in registration
        order. Pure read; NEVER raises.

        Phase 2 (PRD §37 v2.54→v2.55, 2026-05-07): canonical
        operator-facing read API for the ThinkingProgressObserver
        aggregator. Eliminates the need for downstream consumers
        to reach into ``self._items`` private state.

        ``states`` filter — when provided, only frames whose state
        is in the tuple are returned. Default ``None`` returns
        every state (BUFFERING + COMMITTED + DISCARDED).

        Empty / invalid ``op_id`` returns empty tuple. Pure read."""
        try:
            op_safe = _safe_str(op_id)
            kind_enum = NarrativeKind.coerce(kind)
        except Exception:  # noqa: BLE001 — defensive
            return ()
        with self._lock:
            out = []
            for frame in self._items.values():
                if frame.op_id != op_safe:
                    continue
                if frame.kind is not kind_enum:
                    continue
                if (
                    states is not None
                    and frame.state not in states
                ):
                    continue
                out.append(frame)
            return tuple(out)

    def active_thinking_frame(
        self, *, op_id: str,
    ) -> Optional["NarrativeFrame"]:
        """Return the BUFFERING THINKING frame for ``op_id`` if
        one exists, else None. O(1) via composite-key lookup;
        canonical aggregator-facing accessor.

        Pure read; NEVER raises. Used by
        :class:`ThinkingProgressObserver` to bind verb-phrase to
        an op without iterating all frames."""
        try:
            op_safe = _safe_str(op_id)
        except Exception:  # noqa: BLE001 — defensive
            return None
        with self._lock:
            # Composite key is (op_id, phase, kind); we don't
            # know phase but THINKING is typically GENERATE.
            # Walk the active-keys index for any (op_safe, *,
            # THINKING) match.
            target_kind = NarrativeKind.THINKING.value
            for (k_op, _phase, k_kind), ref in (
                self._active_keys.items()
            ):
                if k_op == op_safe and k_kind == target_kind:
                    frame = self._items.get(ref)
                    if (
                        frame is not None
                        and frame.state is FrameState.BUFFERING
                    ):
                        return frame
            return None


# ===========================================================================
# Module singleton
# ===========================================================================


_default_channel: Optional[NarrativeChannel] = None
_singleton_lock = threading.Lock()


def get_default_channel() -> NarrativeChannel:
    global _default_channel
    with _singleton_lock:
        if _default_channel is None:
            _default_channel = NarrativeChannel()
        return _default_channel


def reset_default_channel_for_tests() -> None:
    global _default_channel
    with _singleton_lock:
        _default_channel = None


__all__ = [
    "BUFFER_SIZE_ENV_VAR",
    "ChannelSnapshot",
    "FrameState",
    "NARRATIVE_CHANNEL_SCHEMA_VERSION",
    "NarrativeChannel",
    "NarrativeFrame",
    "NarrativeKind",
    "REF_PREFIX",
    "get_default_channel",
    "register_flags",
    "register_shipped_invariants",
    "reset_default_channel_for_tests",
]


# ===========================================================================
# Slice 5 — FlagRegistry self-registration
# ===========================================================================


def register_flags(registry) -> int:
    """Module-owned FlagRegistry registration for the Gap #6 arc.
    Returns count of FlagSpecs added. NEVER raises."""
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category, FlagSpec, FlagType,
        )
    except ImportError:
        return 0

    specs = [
        FlagSpec(
            name="JARVIS_NARRATIVE_INTENT_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for the op_started intent prompt "
                "(Gap #6 Slice 2). Brief async LLM call (Tier 0 DW, "
                "50-token cap, 5s timeout) that asks the model to "
                "state its intent in 1 sentence. Default TRUE post "
                "graduation 2026-05-04. Operators set =false to "
                "disable the micro-LLM call (saves ~$0.0002 per op)."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/governance/intent_prompter.py"
            ),
            example="true",
            since="Gap #6 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_TOOL_PREAMBLE_FALLBACK_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for the synthesized tool-preamble "
                "fallback (Gap #6 Slice 2). When the model omits a "
                "preamble for a tool call, synthesize one "
                "deterministically from tool_name + args via the "
                "per-tool template registry. Tool Transparency: every "
                "tool call gets a 🗣 line. Default TRUE post graduation."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/governance/tool_preamble_synthesizer.py"
            ),
            example="true",
            since="Gap #6 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_NARRATIVE_BUFFER_SIZE",
            type=FlagType.INT,
            default=200,
            description=(
                "Capacity of the NarrativeChannel ring (Gap #6 Slice 1). "
                "Drop-oldest eviction; clamped to [1, 10_000]. Backs the "
                "/expand n-N REPL recovery."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/narrative_channel.py"
            ),
            example="200",
            since="Gap #6 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_NARRATIVE_INTENT_TIMEOUT_S",
            type=FlagType.FLOAT,
            default=5.0,
            description=(
                "Wall-clock timeout for the op_started intent prompt "
                "LLM call. Clamped to [0.5, 30.0]. Hard cap to keep "
                "op_started fast."
            ),
            category=Category.TIMING,
            source_file=(
                "backend/core/ouroboros/governance/intent_prompter.py"
            ),
            example="5.0",
            since="Gap #6 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_NARRATIVE_INTENT_MAX_TOKENS",
            type=FlagType.INT,
            default=50,
            description=(
                "Max output tokens for the intent prompt. Clamped "
                "[10, 200]. Cost-bound: structurally caps the per-op "
                "intent micro-spend."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/governance/intent_prompter.py"
            ),
            example="50",
            since="Gap #6 Slice 5 (2026-05-04)",
        ),
    ]

    count = 0
    for spec in specs:
        try:
            registry.register(spec)
            count += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "[NarrativeChannel] flag registration failed for %s",
                getattr(spec, "name", "?"), exc_info=True,
            )
    return count


# ===========================================================================
# Slice 5 — shipped_code_invariants self-registration
# ===========================================================================


def register_shipped_invariants() -> list:
    """Module-owned shipped-code AST pins for the Gap #6 arc.

    Four structural invariants:

      1. ``narrative_kind_taxonomy_frozen`` — closed 7-value enum
         is the source of truth for the renderer's dispatch table;
         losing a value silently disables a model-voice channel.
         Extended 2026-05-08 (§38.11-D) to require ``DREAM``.
      2. ``narrative_renderer_visual_hierarchy`` — the renderer's
         style table MUST cover every NarrativeKind explicitly
         (Constraint 1: visual hierarchy).
      3. ``op_tool_start_synthesizer_wired`` — the preamble
         synthesizer fallback MUST be invoked from op_tool_start
         (Constraint 2: tool transparency regression pin).
      4. ``op_started_intent_prompt_wired`` — _maybe_fire_intent_prompt
         MUST be called from op_started.

    NEVER raises (returns ``[]`` on import failure)."""
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except ImportError:
        return []

    import ast as _ast

    def _validate_narrative_kind_frozen(tree, _source) -> tuple:
        del _source
        seen: set = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == "NarrativeKind":
                for stmt in node.body:
                    if isinstance(stmt, _ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, _ast.Name):
                                seen.add(target.id)
        required = {
            "INTENT", "PLAN_PROSE", "TOOL_PREAMBLE",
            "THINKING", "L2_REPAIR_PROSE", "POSTMORTEM_PROSE",
            "DREAM",  # §38.11-D 2026-05-08 (Introspective Voice)
        }
        missing = required - seen
        if missing:
            return (
                f"NarrativeKind lost values: {sorted(missing)} — "
                "the closed taxonomy is frozen by Gap #6 Slice 5 "
                "+ §38.11-D (DREAM)",
            )
        return ()

    def _validate_renderer_covers_all_kinds(_tree, source) -> tuple:
        """The renderer's _KIND_STYLES table MUST include every
        NarrativeKind member literal. Pure source-grep — fast and
        robust to AST shape changes."""
        del _tree
        violations = []
        for kind in (
            "INTENT", "PLAN_PROSE", "TOOL_PREAMBLE",
            "THINKING", "L2_REPAIR_PROSE", "POSTMORTEM_PROSE",
            "DREAM",  # §38.11-D 2026-05-08
        ):
            if f"NarrativeKind.{kind}" not in source:
                violations.append(
                    f"narrative_renderer.py missing style for "
                    f"NarrativeKind.{kind} — visual-hierarchy "
                    "constraint requires explicit per-kind style"
                )
        return tuple(violations)

    def _validate_op_tool_start_synthesizer(tree, _source) -> tuple:
        del _source
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if node.name == "op_tool_start":
                    body = _ast.unparse(node)
                    if "synthesize_preamble" not in body:
                        return (
                            "op_tool_start missing synthesize_preamble "
                            "call — Tool Transparency constraint broken",
                        )
                    return ()
        return ("op_tool_start method not found in serpent_flow.py",)

    def _validate_op_started_intent_prompt(tree, _source) -> tuple:
        del _source
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if node.name == "op_started":
                    body = _ast.unparse(node)
                    if "_maybe_fire_intent_prompt" not in body:
                        return (
                            "op_started missing _maybe_fire_intent_prompt "
                            "call — intent prompt regressed",
                        )
                    return ()
        return ("op_started method not found",)

    return [
        ShippedCodeInvariant(
            invariant_name="narrative_kind_taxonomy_frozen",
            target_file=(
                "backend/core/ouroboros/battle_test/narrative_channel.py"
            ),
            description=(
                "NarrativeKind's 7-value closed taxonomy must remain "
                "intact (extended 2026-05-08 §38.11-D adds DREAM for "
                "DreamEngine prose). Losing a kind silently disables "
                "a model-voice channel."
            ),
            validate=_validate_narrative_kind_frozen,
        ),
        ShippedCodeInvariant(
            invariant_name="narrative_renderer_visual_hierarchy",
            target_file=(
                "backend/core/ouroboros/battle_test/narrative_renderer.py"
            ),
            description=(
                "Constraint 1 (visual hierarchy): the renderer's style "
                "table must include every NarrativeKind explicitly. "
                "Operators rely on stable per-kind glyph + tint."
            ),
            validate=_validate_renderer_covers_all_kinds,
        ),
        ShippedCodeInvariant(
            invariant_name="op_tool_start_synthesizer_wired",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "Constraint 2 (Tool Transparency) BUG-FIX REGRESSION "
                "PIN: op_tool_start must call synthesize_preamble so "
                "every tool call gets a 🗣 line."
            ),
            validate=_validate_op_tool_start_synthesizer,
        ),
        ShippedCodeInvariant(
            invariant_name="op_started_intent_prompt_wired",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "BUG-FIX REGRESSION PIN: op_started must call "
                "_maybe_fire_intent_prompt to surface the model's "
                "intent at op start."
            ),
            validate=_validate_op_started_intent_prompt,
        ),
    ]


# ---------------------------------------------------------------------------
# Narrative-density voices — one per kind, derived from the enum
# ---------------------------------------------------------------------------

def _kind_density_table() -> Dict[str, tuple]:
    """(min_density, legacy_flag) per kind value. NEVER raises.

    Keyed by the ENUM member's value, not by a parallel list of literals,
    so a renamed kind cannot silently lose its threshold — the same
    discipline as `_validate_renderer_covers_all_kinds`, which exists
    because a kind without a style renders as nothing at all.

    A kind absent from this table resolves to ON rather than being dropped:
    the failure mode of a forgotten entry must be "slightly too talkative",
    never "invisible", because only one of those is noticeable.
    """
    return {
        # The preamble IS the `preambles` level — it is what that rung of
        # the ladder was named after.
        NarrativeKind.TOOL_PREAMBLE.value: (
            "preambles", "JARVIS_TOOL_PREAMBLE_FALLBACK_ENABLED"),
        NarrativeKind.INTENT.value: ("on", "JARVIS_NARRATIVE_INTENT_ENABLED"),
        NarrativeKind.PLAN_PROSE.value: ("on", ""),
        NarrativeKind.L2_REPAIR_PROSE.value: ("on", ""),
        NarrativeKind.DREAM.value: ("verbose", ""),
        # Extended thinking is the only thing `verbose` ever promised, and
        # the flag it wrote had no reader. This is that promise acquiring
        # a consumer.
        NarrativeKind.THINKING.value: (
            "verbose", "JARVIS_NARRATIVE_THINKING_VERBOSE"),
        # A failure report is audible at every level. Not `exempt`: an
        # operator who explicitly silences postmortems has made a decision,
        # and only alarms outrank an explicit decision.
        NarrativeKind.POSTMORTEM_PROSE.value: ("off", ""),
    }


def register_voices(registry) -> int:
    """Declare one voice per narrative kind. NEVER raises.

    Iterating the enum rather than a hand-written list means a kind added
    later gets a voice automatically — the property whose absence is the
    whole defect: `/narrate` hardcoded four flags and was never updated
    when the organism learned to speak in new ways.
    """
    try:
        table = _kind_density_table()
        count = 0
        for kind in NarrativeKind:
            min_density, legacy = table.get(kind.value, ("on", ""))
            if registry.register(
                f"narrative.{kind.value}", min_density,
                legacy_flag=legacy, owner=__name__,
                describe=f"NarrativeKind.{kind.name} frames",
            ) is not None:
                count += 1
        return count
    except Exception:  # noqa: BLE001
        return 0


try:  # Self-register at import — a voice must be known BEFORE it speaks.
    from backend.core.ouroboros.ui.narrative_density import (  # noqa: E402
        default_registry as _default_voice_registry,
    )
    register_voices(_default_voice_registry())
except Exception:  # noqa: BLE001
    pass
