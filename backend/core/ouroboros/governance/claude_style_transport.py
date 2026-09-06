"""Claude-style CommProtocol transport — terse one-line-per-op rendering.

Closes the "O+V's flowing-CLI metaphor doesn't match CC's tool-call
metaphor" UX gap. CC shows discrete actions as single-line blocks
with bullet markers (●/✓/✗); each tool call is one visible unit
that fades to the scroll area when complete. O+V's :class:`Serpent-
Transport` builds multi-line per-op blocks (`┌ ... │ sensed │ route
│ planning │ routing │ synthesizing`) which scroll quickly into
unreadable cascades.

This transport is a **parallel implementation** of the same
:class:`CommProtocol` transport contract that :class:`Serpent-
Transport` satisfies. Operator picks via ``JARVIS_RENDER_MODE``
(``claude`` | ``serpent``). Both transports remain available; the
default flips to ``claude`` for the cleaner look.

Architectural pillars:

  1. **Same contract, different idiom** — implements ``async send(msg)``
     consuming :class:`CommMessage` events (INTENT / HEARTBEAT /
     DECISION / POSTMORTEM). No new message types needed; no
     producer-side changes required. The transport IS the rendering
     surface.
  2. **One line per op transition** — INTENT prints a single
     ``⏺ <Sensor> · <summary>`` line; DECISION prints
     ``✓ <Sensor> · <summary> · Xs`` (success), ``✗ … · shed: <reason>``
     (failure) or ``⎿ … · no change`` with the model's own reason
     beneath. No per-phase emoji cascade; no multi-line block per op;
     no truncated ids (OV_DESIGN_LANGUAGE §3 — a hash earns its place
     only as an ``/expand``-able ref, and these are not).
  3. **Status marks from a closed taxonomy** — :class:`OpStatusGlyph`,
     six STATUSES whose marks are NAMES in the design-language ration
     (``theme.mark``): one glyph table for every surface, one ASCII
     degradation. AST-pinned on the member names.
  3b. **The organism narrates** (2026-09-06) — this transport is the
     default renderer, and until now it was the one surface with no
     voice: the 💭 intent SerpentFlow requests on ``op_started`` and the
     🗣 preamble it prints on ``op_tool_start`` are SerpentFlow methods,
     which the default mode never calls. The transport now uses the SAME
     producers (``intent_prompter``, ``tool_preamble_synthesizer``) and
     renders every frame that commits to the ``NarrativeChannel`` through
     the same renderer, so plan prose, repair prose and postmortems
     arrive too — one subscription, one seam (``_safe_print``), mirrored
     to every attached cockpit.
  4. **Boot-recovery suppression preserved** — same logic as
     SerpentTransport: the first ``boot_recovery_*`` reason starts
     a counter; subsequent ones increment; on first non-recovery
     INTENT, flush a single summary line ``boot recovery │ N stale
     entries reconciled``. Operator sees ONE line for all 75 stale
     entries instead of 75 individual ones.
  5. **No hardcoded colors at the print site** — every color tag
     resolves through the existing :class:`ColorRole` + theme
     substrate. Operators flip themes via
     ``JARVIS_RENDER_CONDUCTOR_PALETTE_OVERRIDE`` and the per-op
     bullet/state colors update accordingly.
  6. **Defensive everywhere** — every send/render method swallows
     exceptions. A misbehaving message cannot crash the comm-
     protocol pipeline. Mirrors SerpentTransport's never-raise
     contract.

Authority invariants (AST-pinned via
``register_shipped_invariants``):

  * No imports of ``rich`` / ``rich.*`` at module top (lazy import
    inside render methods is allowed — Rich is a hard dep of the
    underlying console regardless).
  * No imports of orchestrator / policy / iron_gate / risk_tier /
    change_engine / candidate_generator / gate / semantic_guardian /
    semantic_firewall / providers / doubleword_provider /
    urgency_router / cancel_token / conversation_bridge.
  * :class:`OpStatusGlyph` enum members match the documented closed
    set.
  * :class:`RenderMode` enum members match the documented closed
    set.
  * ``register_flags`` + ``register_shipped_invariants`` symbols
    present (auto-discovery contract).

Kill switches:

  * ``JARVIS_RENDER_MODE`` — ``claude`` (default) or ``serpent``
    (legacy SerpentFlow per-op blocks). Hot-revert via env preserved.
  * ``JARVIS_CLAUDE_STYLE_SHOW_HEARTBEATS`` — bool, default ``false``.
    When true, HEARTBEAT messages emit phase ticks. Default is
    deliberately silent (the active line carries enough state).
"""
from __future__ import annotations

import asyncio
import enum
import logging
import textwrap
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from backend.core.ouroboros.ui import theme as _theme
from backend.core.ouroboros.ui.semantic_tokens import (  # noqa: E402
    role_palette as _role_palette,
)

#: Semantic colour roles — the SAME name and access pattern as every
#: other module. One vocabulary, one spelling, one owner.
_SEM = _role_palette()

logger = logging.getLogger(__name__)


CLAUDE_STYLE_TRANSPORT_SCHEMA_VERSION: str = "claude_style_transport.1"


_FLAG_RENDER_MODE = "JARVIS_RENDER_MODE"
_FLAG_CLAUDE_SHOW_HEARTBEATS = "JARVIS_CLAUDE_STYLE_SHOW_HEARTBEATS"
#: Characters of goal summary on an op line, and the wrap width of the
#: organism's narration beneath it. One budget, so the block reads as one.
_FLAG_LINE_CHARS = "JARVIS_CLAUDE_STYLE_LINE_CHARS"
_DEFAULT_LINE_CHARS = 72
_MIN_LINE_CHARS = 24
#: Lines of the model's own words shown under an outcome before the rest
#: is elided (OV_DESIGN_LANGUAGE §5: past ~8 lines it wants an /expand).
_FLAG_DETAIL_LINES = "JARVIS_CLAUDE_STYLE_DETAIL_LINES"
_DEFAULT_DETAIL_LINES = 3
#: Bound on remembered (op, round) preamble keys — a memory bound, not a
#: behaviour; the same figure SerpentFlow keeps for the same dedup.
_PREAMBLE_MEMORY = 512
#: Coalescing window for in-flight stream frames: tokens arriving inside
#: it ride one frame. The figure the stream renderer batches at.
_FLAG_STREAM_FLUSH_MS = "JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS"
_DEFAULT_STREAM_FLUSH_MS = 16
#: Characters of a generation kept in memory per op while it streams — a
#: memory bound (the wire frame carries only the renderer's tail cap).
_STREAM_BUFFER_CHARS = 4096


# ---------------------------------------------------------------------------
# Closed taxonomies — AST-pinned
# ---------------------------------------------------------------------------


class RenderMode(str, enum.Enum):
    """Closed taxonomy of operator-selectable render modes.

    CLAUDE: terse one-line-per-op rendering (this transport). The
    default — graduated immediately because the user explicitly asked
    for the cleaner look.

    SERPENT: legacy SerpentTransport with multi-line per-op blocks
    (``┌ … │ sensed │ route │ planning │ routing │ synthesizing``).
    Hot-revert escape hatch for operators who prefer the verbose
    flow."""

    CLAUDE = "CLAUDE"
    SERPENT = "SERPENT"


class OpStatusGlyph(enum.Enum):
    """Closed taxonomy of per-op status marks — six STATUSES.

    Each value is ``(mark, colour role, status)``. The mark is a NAME in
    the design-language ration (``theme.mark``), so this transport draws
    from the same six-glyph table as every other surface and degrades to
    ASCII with it; the colour role is a name in the semantic palette. Two
    statuses may share a mark — outcomes are told apart by colour and copy
    (OV_DESIGN_LANGUAGE §3.4), never by inventing a seventh glyph. Before
    this the transport kept bullets of its own (``· ● ◌ ⏭``): the middle
    dot is the language's SEPARATOR, and the other three are outside the
    ration, so the presentation router scrubbed them and the op lines read
    as noise on the very surface that is the default."""

    ACTIVE = ("action", "neural", "active")       # ⏺ an op begins
    RUNNING = ("detail", "heal", "running")       # ⎿ a mid-op transition
    DONE = ("check", "success", "done")           # ✓ typographic outcome
    FAILED = ("cross", "death", "failed")         # ✗ typographic outcome
    CANCELLED = ("detail", "dim", "cancelled")    # ⎿ ended without effect
    NOOP = ("detail", "dim", "noop")              # ⎿ the model declined

    @property
    def mark(self) -> str:
        return self.value[0]

    @property
    def role(self) -> str:
        return self.value[1]

    @property
    def glyph(self) -> str:
        """The rendered glyph, from the one theme table, for the terminal
        that will display it."""
        return _theme.mark(self.mark)


# ---------------------------------------------------------------------------
# Flag accessors
# ---------------------------------------------------------------------------


def _get_registry() -> Any:
    try:
        from backend.core.ouroboros.governance import flag_registry as _fr
        return _fr.ensure_seeded()
    except Exception:  # noqa: BLE001 — defensive
        return None


def resolve_render_mode() -> RenderMode:
    """Read the operator-selected render mode. Default ``CLAUDE``.
    Unknown values fall back to ``CLAUDE`` (operator typo doesn't
    accidentally restore the noisy legacy)."""
    reg = _get_registry()
    if reg is None:
        return RenderMode.CLAUDE
    raw = reg.get_str(_FLAG_RENDER_MODE, default="CLAUDE").strip().upper()
    try:
        return RenderMode(raw)
    except ValueError:
        logger.debug(
            "[claude_style_transport] unknown render mode %r, "
            "falling back to CLAUDE", raw,
        )
        return RenderMode.CLAUDE


def show_heartbeats() -> bool:
    """Whether HEARTBEAT messages emit phase ticks. Default false —
    operators get the cleaner one-line-per-op view; the active line
    carries enough state without per-tick chatter."""
    reg = _get_registry()
    if reg is None:
        return False
    return reg.get_bool(_FLAG_CLAUDE_SHOW_HEARTBEATS, default=False)


def _read_int_flag(name: str, default: int, floor: int) -> int:
    reg = _get_registry()
    if reg is None:
        return default
    try:
        return max(floor, int(str(reg.get_str(name, default=str(default))).strip()))
    except Exception:  # noqa: BLE001 — a typo in a knob never blanks a line
        return default


def line_chars() -> int:
    """Goal-summary budget and narration wrap width (``JARVIS_CLAUDE_STYLE_LINE_CHARS``)."""
    return _read_int_flag(_FLAG_LINE_CHARS, _DEFAULT_LINE_CHARS, _MIN_LINE_CHARS)


def detail_lines() -> int:
    """Lines of the model's words under an outcome (``JARVIS_CLAUDE_STYLE_DETAIL_LINES``)."""
    return _read_int_flag(_FLAG_DETAIL_LINES, _DEFAULT_DETAIL_LINES, 1)


def stream_flush_ms() -> int:
    """Coalescing window for in-flight frames (``JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS``)."""
    return _read_int_flag(_FLAG_STREAM_FLUSH_MS, _DEFAULT_STREAM_FLUSH_MS, 1)


# ---------------------------------------------------------------------------
# Per-op state
# ---------------------------------------------------------------------------


@dataclass
class _OpState:
    """In-flight op state. The transport tracks one per op_id; clears
    on DECISION/POSTMORTEM."""

    op_id: str
    short_id: str
    sensor: str            # "TestFailure" / "Operation" / "GitHub Issue" / etc.
    summary: str           # the goal text from INTENT
    started_monotonic: float = field(default_factory=time.monotonic)
    risk_tier: str = ""
    target_files: tuple = ()
    goal: str = ""         # the full goal — the intent prompt wants all of it


@dataclass
class _StreamState:
    """One generation being WRITTEN, as far as the cockpit has seen it."""

    op_id: str
    provider: str = ""
    text: str = ""
    last_flush: float = 0.0
    flush_handle: Any = None
    tokens: int = 0


def _short_id(op_id: str) -> str:
    """Produce a 6-char short id mirroring SerpentFlow's
    convention. Defensive — empty input yields ``"......"``."""
    if not isinstance(op_id, str) or not op_id:
        return "......"
    if "-" in op_id:
        head, _, rest = op_id.partition("-")
        return (head + rest)[:6] or "......"
    return op_id[:6] or "......"


def _format_elapsed(started_monotonic: float) -> str:
    """Format an elapsed time relative to start_monotonic. Defensive
    — clamps negative or absurdly large values to a 24h ceiling."""
    if started_monotonic <= 0.0:
        return "0.0s"
    elapsed = max(0.0, min(time.monotonic() - started_monotonic, 86400.0))
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    if elapsed < 3600:
        return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
    return f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"


def _escape(s: object) -> str:
    """Neutralise model-controlled text for Rich markup without importing
    ``rich`` (an authority invariant of this module, AST-pinned). Escaping
    every "[" is a strict superset of what ``rich.markup.escape`` does, so
    no text can open a tag."""
    return str(s).replace("[", "\\[")


def _clip_words(text: object, limit: int) -> str:
    """Cut prose to ``limit`` at a WORD boundary with the theme's ellipsis.
    A cut mid-word ("graduati...") reads as a defect; a cut at a word with a
    real ellipsis reads as a summary (OV_DESIGN_LANGUAGE §3)."""
    s = " ".join(str(text or "").split())
    if limit <= 0 or len(s) <= limit:
        return s
    ell = _theme.mark("ellipsis")
    room = max(1, limit - len(ell))
    cut = s[:room]
    space = cut.rfind(" ")
    if space >= room // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + ell


def _short_path(path: object, budget: int) -> str:
    """``…/parent/file.py`` once a path outgrows its budget."""
    p = str(path or "")
    if len(p) <= budget:
        return p
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if len(parts) < 2:
        return p
    return f"{_theme.mark('ellipsis')}/{'/'.join(parts[-2:])}"


def _humanise(code: object) -> str:
    """A reason CODE as words: ``background_dw_blocked`` → ``background dw
    blocked``. The code is information; its spelling is a field name, and
    the operator never sees a field name (§3.1)."""
    return " ".join(str(code or "").strip().split("_")).strip()


# ---------------------------------------------------------------------------
# ClaudeStyleTransport — terse per-op rendering
# ---------------------------------------------------------------------------


class ClaudeStyleTransport:
    """Drop-in alternative to :class:`SerpentTransport`. Renders
    CommProtocol messages in Claude Code's idiom: one line per op
    transition, bullet markers from a closed taxonomy, restrained
    palette via the existing theme.

    Wire into ``CommProtocol._transports`` exactly like
    SerpentTransport. Both transports cannot be active simultaneously
    on the same console — the harness picks one based on
    ``JARVIS_RENDER_MODE``.
    """

    def __init__(self, console: Any) -> None:
        """``console`` is a Rich Console (or duck-typed equivalent
        with a ``print`` method). The transport prints directly to
        it; never calls SerpentFlow methods. Bottom-toolbar/REPL
        state remains owned by SerpentFlow regardless of transport
        choice."""
        self._console = console
        self._op_state: Dict[str, _OpState] = {}
        self._boot_recovery_count: int = 0
        self._boot_recovery_flushed: bool = False
        # THE COCKPIT MIRROR — same name and contract as
        # ``SerpentFlow.markup_mirror``, so the harness wires both with one
        # idiom (``transport.markup_mirror = bridge.publish_markup``).
        #
        # Why it exists, measured 2026-09-06: this transport is the DEFAULT
        # (``JARVIS_RENDER_MODE=CLAUDE``) and every line it rendered went to
        # ``self._console`` alone. On a headless daemon that console is the
        # log file. An attached cockpit therefore showed a live prompt, a
        # live status line and an EMPTY transcript while the organism ran
        # forty-seven generations — the operator was watching a process
        # that had no channel to them. SerpentFlow had a mirror; the
        # transport that replaced it as default did not.
        #
        # ``None`` until wired. Every rendered line passes through
        # ``_safe_print``, so that one seam is where the mirror fires.
        self.markup_mirror: Optional[Callable[[str], None]] = None
        # THE COCKPIT TELEMETRY MIRROR — a DIRECT ``bridge.publish_telemetry``
        # reference, wired at the same attach-arming seam as ``markup_mirror``.
        # The in-flight token stream rides the telemetry lane; it must NOT go
        # through the module-global ``publish_telemetry_global``, whose
        # ``_ACTIVE_BRIDGE`` is cleared on some mount paths while the live
        # bridge keeps its clients (measured 2026-09-06: 4641 stream frames
        # saw cockpits=0 through the global while a cockpit was attached and
        # receiving heartbeats over the very same bridge). ``None`` until
        # wired; then it is the sink the stream publishes through.
        self.telemetry_mirror: Optional[Callable[[dict], None]] = None
        # CC2.1 — running counters for TASK_LIST composer field
        self._done_count: int = 0
        self._failed_count: int = 0
        # Narration. Intent requests run as tasks off the render path (a
        # transport must never await a model); the set keeps them
        # referenced until done. Preamble keys dedupe a parallel tool batch
        # to one 🗣 line per (op, round), exactly as SerpentFlow does.
        self._narration_tasks: Set[Any] = set()
        self._preamble_keys: "OrderedDict[tuple, None]" = OrderedDict()
        # Token streams in flight, by op. Fed by the render conductor's
        # PHASE_BEGIN / REASONING_TOKEN / PHASE_END triplet; carried to
        # cockpits as `stream_inflight` frames through the ONE producer in
        # stream_renderer, coalesced to `stream_flush_ms()`.
        self._streams: Dict[str, _StreamState] = {}
        self._subscribe_narrative()

    # -- composer feed (CC2.1) --------------------------------------

    def _feed_composer(self) -> None:
        """Push current state into the StatusLineComposer (D5).
        Sets ACTIVE_OP + TASK_LIST fields. NEVER raises — composer
        unavailable is a no-op."""
        try:
            from backend.core.ouroboros.governance.status_line_composer import (  # noqa: E501
                StatusField,
                update_field,
            )
        except Exception:  # noqa: BLE001 — defensive
            return
        try:
            # ACTIVE_OP = most recent INTENT (newest started)
            active_label = ""
            if self._op_state:
                latest = max(
                    self._op_state.values(),
                    key=lambda s: s.started_monotonic,
                )
                active_label = f"{latest.sensor}({latest.short_id})"
            update_field(StatusField.ACTIVE_OP, active_label)
            # TASK_LIST = compact counts
            update_field(StatusField.TASK_LIST, {
                "active": len(self._op_state),
                "queued": 0,  # populated by TaskListObserver (CC2.3)
                "done": self._done_count + self._failed_count,
            })
        except Exception:  # noqa: BLE001 — defensive
            pass

    # -- transport contract -----------------------------------------

    async def send(self, msg: Any) -> None:
        """Handle one :class:`CommMessage`. NEVER raises — defensive
        everywhere; misbehaving messages don't crash the comm pipeline.
        """
        try:
            payload = getattr(msg, "payload", {}) or {}
            op_id = getattr(msg, "op_id", "") or ""
            msg_type = ""
            mt = getattr(msg, "msg_type", None)
            if mt is not None:
                msg_type = (
                    mt.value if hasattr(mt, "value") else str(mt)
                )
            if msg_type == "INTENT":
                self._handle_intent(op_id, payload)
            elif msg_type == "HEARTBEAT":
                # Not pre-gated here. Two kinds of message share this type
                # -- phase ticks and tool calls -- and they have different
                # gates. Deciding on `show_heartbeats()` at the dispatch
                # silenced tool activity along with the ticks; the handler
                # reads the payload and applies the gate that fits it.
                self._handle_heartbeat(op_id, payload)
            elif msg_type == "DECISION":
                self._handle_decision(op_id, payload)
            elif msg_type == "POSTMORTEM":
                self._handle_postmortem(op_id, payload)
            # Unknown msg_types silently dropped — comm protocol
            # may add new types and this transport degrades cleanly.
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[claude_style_transport] send raised for msg_type=%s",
                getattr(getattr(msg, "msg_type", None), "value", "?"),
                exc_info=True,
            )

    # -- handlers ----------------------------------------------------

    def _handle_intent(
        self, op_id: str, payload: Dict[str, Any],
    ) -> None:
        """INTENT message — op begins. Render one line:
        ``· <Sensor>(<short_id>) <summary>``"""
        # Boot-recovery suppression (mirrors SerpentTransport).
        reason_code = str(payload.get("reason_code", "") or "")
        risk_tier = str(payload.get("risk_tier", "") or "")
        sep = _theme.mark("dot")
        if reason_code.startswith("boot_recovery_"):
            self._boot_recovery_count += 1
            if self._boot_recovery_count == 1:
                self._safe_print(
                    f"[dim]{_theme.mark('detail')} boot recovery {sep} "
                    f"reconciling stale ledger entries{_theme.mark('ellipsis')}[/dim]"
                )
            return
        if risk_tier == "routing":
            return  # internal routing decision, not an op start

        # Flush boot recovery summary on first real INTENT
        if self._boot_recovery_count > 0 and not self._boot_recovery_flushed:
            self._boot_recovery_flushed = True
            self._safe_print(
                f"[dim]{_theme.mark('detail')} boot recovery {sep} "
                f"{self._boot_recovery_count} stale entries reconciled[/dim]"
            )
            self._safe_print("")

        sensor = self._infer_sensor(payload)
        summary = self._summarize(payload)
        state = _OpState(
            op_id=op_id,
            short_id=_short_id(op_id),
            sensor=sensor,
            summary=summary,
            risk_tier=str(payload.get("risk_tier", "") or "").upper(),
            target_files=tuple(payload.get("target_files", []) or []),
            goal=str(payload.get("goal", "") or "").strip(),
        )
        self._op_state[op_id] = state
        # CC2.1 — feed composer with current ACTIVE_OP + TASK_LIST
        self._feed_composer()
        # Render: `⏺ <Sensor> · <summary> [· risk] [path]`
        tail = ""
        if state.risk_tier and state.risk_tier not in ("SAFE_AUTO", "LOW"):
            tone = (
                "yellow" if state.risk_tier in ("MEDIUM", "NOTIFY_APPLY")
                else "red"
            )
            tail += f" [{tone}]{sep} {_escape(_humanise(state.risk_tier.lower()))}[/{tone}]"
        if state.target_files:
            primary = _short_path(state.target_files[0], max(20, line_chars() // 2))
            tail += f" [dim]{_escape(primary)}[/dim]"
        self._safe_print(self._lead(OpStatusGlyph.ACTIVE, state, tail=tail))
        self._narrate_intent(state)

    def _handle_heartbeat(
        self, op_id: str, payload: Dict[str, Any],
    ) -> None:
        """HEARTBEAT — a phase tick, or a TOOL CALL riding the same type.

        Two different things arrive here and they were treated as one.
        ``ToolNarrationChannel`` emits every Venom tool call as a HEARTBEAT
        whose payload carries ``tool_name`` / ``tool_args_summary`` /
        ``result_preview``. This handler read only ``phase`` and printed
        ``└ generate`` — the tool name, its arguments and its result were
        discarded, and only when ``show_heartbeats()`` was on, which it is
        not by default. So the one channel that carries what the organism
        is DOING rendered nothing, on the default transport, to anyone.

        Now: a payload with ``tool_name`` renders a CC-style tool block
        through ``tool_render_view.compose_if_enabled`` — the SAME composer
        ``SerpentFlow.op_tool_call`` uses, so both transports draw one
        idiom — under the channel's own master gate
        (``cockpit_attach.tool_activity_enabled``, default ON), which is
        about tool activity and not about phase ticks. A plain phase tick
        keeps its old behaviour and its old gate.

        Only the COMPLETION renders as a tool block. The start event drives
        a spinner in SerpentFlow; a line stream has no spinner, and printing
        both would show every tool twice. But the start event is ALSO where
        the model's one-sentence WHY (``preamble``) travels — the narration
        the operator asked for — so the start narrates and the completion
        draws.
        """
        tool_name = str(payload.get("tool_name", "") or "").strip()
        if tool_name:
            if payload.get("tool_starting"):
                self._narrate_preamble(op_id, tool_name, payload)
                return
            self._render_tool_call(op_id, tool_name, payload)
            return
        if not show_heartbeats():
            return
        phase = str(payload.get("phase", "") or "").lower()
        if not phase or op_id not in self._op_state:
            return
        self._safe_print(f"  [dim]└ {phase}[/dim]")

    def _render_tool_call(
        self, op_id: str, tool_name: str, payload: Dict[str, Any],
    ) -> None:
        """One completed tool call, in the tool-activity idiom. NEVER raises.

        Composition is delegated so this transport owns no second opinion
        about how a tool block looks; when the registry composer is off, a
        minimal header in this transport's own palette is emitted rather
        than nothing, because an empty transcript is the defect.
        """
        try:
            from backend.core.ouroboros.battle_test.cockpit_attach import (
                tool_activity_enabled,
            )
            if not tool_activity_enabled():
                return
        except Exception:  # noqa: BLE001 — gate unavailable → channel on
            pass
        args = str(payload.get("tool_args_summary", "") or "")
        result = str(payload.get("result_preview", "") or "")
        status = str(payload.get("status", "") or "success")
        try:
            duration_ms = float(payload.get("duration_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration_ms = 0.0
        composed = None
        try:
            from backend.core.ouroboros.battle_test.tool_render_view import (
                compose_if_enabled, store_for_view,
            )
            composed = compose_if_enabled(
                tool_name, args, result,
                status=status, duration_ms=duration_ms, op_id=op_id,
                round_index=int(payload.get("round_index", 0) or 0),
                palette=_SEM, store=store_for_view(),
            )
        except Exception:  # noqa: BLE001 — composer fault → minimal line
            composed = None
        if composed is not None:
            for line in (
                composed.header_markup, composed.summary_markup,
                *composed.body_lines_markup, composed.expansion_hint,
            ):
                if line:
                    self._safe_print(f"  {line}")
            return
        tone = _SEM["death"] if status not in ("success", "ok") else _SEM["dim"]
        head = (
            f"  [{_SEM['dim']}]{_theme.mark('action')}[/] "
            f"[bold]{_escape(tool_name)}[/bold]"
        )
        if args:
            head += f"([dim]{_escape(_clip_words(args, line_chars()))}[/dim])"
        if status not in ("success", "ok"):
            head += f" [{tone}]{_escape(status)}[/]"
        self._safe_print(head)
        if result:
            self._safe_print(
                f"    [{_SEM['dim']}]{_theme.mark('detail')}[/]  "
                f"[dim]{_escape(_clip_words(result, line_chars() * 2))}[/dim]"
            )

    def _handle_decision(
        self, op_id: str, payload: Dict[str, Any],
    ) -> None:
        """DECISION — op outcome. Render the closing line."""
        outcome = str(payload.get("outcome", "") or "").lower()
        state = self._op_state.pop(op_id, None)
        if state is None:
            state = self._state_for_unannounced(op_id, payload)
            if state is None:
                # Decision without prior INTENT and without a terminal
                # stamp — boot-time orphan reconciliation. Suppress.
                return
        elapsed = _format_elapsed(state.started_monotonic)
        sep = _theme.mark("dot")
        code = _humanise(payload.get("reason_code", ""))
        # The model's own words about the outcome (a no-op's reason, a
        # gate's explanation) ride `reason`; the code is the label.
        words = str(payload.get("reason", "") or "")
        path_budget = max(20, line_chars() // 2)

        if outcome in ("completed", "applied", "auto_approved"):
            files = payload.get("files_changed") or payload.get(
                "affected_files",
            ) or []
            tail = f" [dim]{sep} {elapsed}"
            if files:
                tail += f" {sep} {_escape(_short_path(files[0], path_budget))}"
                if len(files) > 1:
                    tail += f" +{len(files) - 1}"
            tail += "[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.DONE, state, tail=tail))
            self._print_detail(words)
            self._done_count += 1
            self._feed_composer()
            return

        if outcome in ("failed", "postmortem"):
            tail = ""
            if code:
                tail += f" [{_SEM['death']}]{sep} shed: {_escape(code)}[/]"
            tail += f" [dim]{sep} {elapsed}[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.FAILED, state, tail=tail))
            self._print_detail(words)
            self._failed_count += 1
            self._feed_composer()
            return

        if outcome == "noop":
            tail = f" [dim]{sep} no change {sep} {elapsed}[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.NOOP, state, tail=tail))
            # Prefer the model's sentence; a bare "noop" code says nothing
            # the lead line did not.
            self._print_detail(words or (code if code != "noop" else ""))
            return

        if outcome == "cancelled":
            tail = f" [dim]{sep} cancelled"
            if code:
                tail += f" {sep} {_escape(code)}"
            tail += f" {sep} {elapsed}[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.CANCELLED, state, tail=tail))
            self._print_detail(words)
            return

        if outcome == "notify_apply":
            files = payload.get("target_files", []) or []
            tail = f" [{_SEM['heal']}]{sep} auto-applying[/]"
            if files:
                tail += f" [{_SEM['heal']}]{_escape(_short_path(files[0], path_budget))}[/]"
                if len(files) > 1:
                    tail += f" [dim]+{len(files) - 1}[/dim]"
            tail += f" [dim]{sep} {elapsed}[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.RUNNING, state, tail=tail))
            return

        if outcome == "escalated":
            tail = f" [{_SEM['heal']}]{sep} held for review[/]"
            if code:
                tail += f" [dim]{sep} {_escape(code)}[/dim]"
            tail += f" [dim]{sep} {elapsed}[/dim]"
            self._safe_print(self._lead(OpStatusGlyph.RUNNING, state, tail=tail))
            self._print_detail(words)
            return

        # Unknown outcome — record state cleared but emit nothing
        # (decision is incomplete; next message will resolve).
        # Re-add the state so a follow-up DECISION can find it.
        self._op_state[op_id] = state

    def _handle_postmortem(
        self, op_id: str, payload: Dict[str, Any],
    ) -> None:
        """POSTMORTEM — explicit failure annotation. Symmetric to a
        failed DECISION; some pipelines emit POSTMORTEM separately."""
        state = self._op_state.pop(op_id, None)
        elapsed = (
            _format_elapsed(state.started_monotonic)
            if state else "?"
        )
        if state is None:
            state = _OpState(
                op_id=op_id, short_id=_short_id(op_id),
                sensor="Operation", summary="",
            )
        sep = _theme.mark("dot")
        reason = _clip_words(
            payload.get("root_cause", "unknown") or "unknown", line_chars(),
        )
        tail = (
            f" [{_SEM['death']}]{sep} postmortem: {_escape(reason)}[/]"
            f" [dim]{sep} {elapsed}[/dim]"
        )
        self._safe_print(self._lead(OpStatusGlyph.FAILED, state, tail=tail))

    # -- helpers -----------------------------------------------------

    def _infer_sensor(self, payload: Dict[str, Any]) -> str:
        """Sensor classification — uses outcome_source first, falls
        back to keyword detection in the goal."""
        sensor = str(payload.get("outcome_source", "") or "")
        if sensor:
            return sensor
        sensor = str(payload.get("sensor", "") or "")
        if sensor:
            return sensor
        goal = str(payload.get("goal", "") or "").lower()
        if "test" in goal:
            return "TestFailure"
        if "todo" in goal:
            return "TODO"
        if "github" in goal or "issue" in goal:
            return "GitHubIssue"
        if "explor" in goal:
            return "Exploration"
        if "doc" in goal:
            return "Documentation"
        if "gap" in goal:
            return "CapabilityGap"
        return "Operation"

    def _summarize(self, payload: Dict[str, Any]) -> str:
        """One line of the op's goal, cut at a word to the line budget."""
        return _clip_words(payload.get("goal", ""), line_chars())

    def _state_for_unannounced(
        self, op_id: str, payload: Dict[str, Any],
    ) -> Optional[_OpState]:
        """An op the gate held BEFORE it ever announced itself.

        Measured 2026-09-06 (bt-2026-09-06-074921): sixteen ops reached
        the ledger as ``blocked`` — ``touches_kernel`` on
        ``unified_supervisor.py``, ``self_modification_unsanctioned_source``
        — with no INTENT ever emitted, so the terminal DECISION arrived
        here with no state and was dropped as a boot orphan. An op held
        for a human that the human never sees is the one line this
        transport must not lose. The ledger's terminal emit stamps
        ``terminal_state``; that stamp, and not a boot-recovery reason,
        is what tells a held op from a reconciled ghost."""
        if not payload.get("terminal_state"):
            return None
        if str(payload.get("reason_code", "") or "").startswith("boot_recovery_"):
            return None
        files = tuple(payload.get("target_files", []) or [])
        summary = (
            _short_path(files[0], max(20, line_chars() // 2)) if files else ""
        )
        return _OpState(
            op_id=op_id, short_id=_short_id(op_id),
            sensor=self._infer_sensor(payload), summary=str(summary),
            started_monotonic=0.0, target_files=files,
        )

    # -- the line shape --------------------------------------------------

    def _lead(self, glyph: OpStatusGlyph, state: _OpState, *, tail: str = "") -> str:
        """``<glyph> <Sensor> · <summary><tail>`` — every op line, one shape.
        No id: the sensor and the summary are what the operator recognises
        an op by, and a six-character hash they cannot ``/expand`` is noise."""
        sep = _theme.mark("dot")
        head = f"[{_SEM[glyph.role]}]{glyph.glyph}[/] [bold]{_escape(state.sensor)}[/bold]"
        if state.summary:
            head += f" {sep} {_escape(state.summary)}"
        return head + tail

    def _print_detail(self, text: object) -> None:
        """The model's own words beneath an outcome line: wrapped to the
        line budget, bounded in height, and marked as a continuation."""
        prose = " ".join(str(text or "").split())
        if not prose:
            return
        width = max(16, line_chars() - 4)
        lines = textwrap.wrap(prose, width=width)
        cap = detail_lines()
        if len(lines) > cap:
            lines = lines[:cap]
            lines[-1] = lines[-1].rstrip(" ,;:.") + _theme.mark("ellipsis")
        glyph = _theme.mark("detail")
        pad = " " * len(glyph)
        for i, line in enumerate(lines):
            lead = glyph if i == 0 else pad
            self._safe_print(
                f"    [{_SEM['dim']} italic]{lead} {_escape(line)}[/{_SEM['dim']} italic]"
            )

    # -- narration ---------------------------------------------------------

    def _narrate_intent(self, state: _OpState) -> None:
        """Ask the model, off the render path, WHY this op — the 💭 line.
        The same producer SerpentFlow calls from ``op_started``; the frame
        it records comes back through the channel's commit signal and is
        rendered by :meth:`_on_narrative_commit`. Never awaited here: a
        transport that awaits a model stalls every other line behind it."""
        try:
            from backend.core.ouroboros.governance.intent_prompter import (
                IntentRequest,
                is_master_flag_enabled,
                request_intent_and_emit,
            )
            if not is_master_flag_enabled():
                return
            loop = asyncio.get_running_loop()
        except Exception:  # noqa: BLE001 — no voice is not a broken render
            return
        req = IntentRequest(
            op_id=state.op_id,
            goal=state.goal or state.summary,
            risk_tier=state.risk_tier,
            target_files=tuple(state.target_files[:5]),
        )
        try:
            task = loop.create_task(
                request_intent_and_emit(req, phase="OP_STARTED"),
            )
        except Exception:  # noqa: BLE001
            logger.debug("[claude_style_transport] intent task refused",
                         exc_info=True)
            return
        self._narration_tasks.add(task)
        task.add_done_callback(self._narration_tasks.discard)

    def _narrate_preamble(
        self, op_id: str, tool_name: str, payload: Dict[str, Any],
    ) -> None:
        """One 🗣 line per (op, round) saying WHY the tool runs — the model's
        own preamble when it wrote one, the deterministic template when it
        did not (declared synthetic so the reader can tell). Recorded on the
        NarrativeChannel like every other frame; rendering is the commit
        listener's, so the line is subject to the operator's ``/narrate``
        density and reaches every cockpit through the one seam."""
        try:
            round_index = int(payload.get("round_index", 0) or 0)
        except (TypeError, ValueError):
            round_index = 0
        key = (op_id, round_index)
        if key in self._preamble_keys:
            return
        preamble = str(payload.get("preamble", "") or "").strip()
        provider = "model"
        if not preamble:
            try:
                from backend.core.ouroboros.ui.narrative_density import audible
                if not audible("narrative.tool_preamble"):
                    return
                from backend.core.ouroboros.governance.tool_preamble_synthesizer import (  # noqa: E501
                    synthesize_preamble,
                )
                preamble = synthesize_preamble(
                    tool_name, payload.get("tool_args_summary", ""),
                    existing_preamble="", fallback_only=True,
                )
                provider = "synthetic"
            except Exception:  # noqa: BLE001
                return
        if not preamble:
            return
        self._preamble_keys[key] = None
        while len(self._preamble_keys) > _PREAMBLE_MEMORY:
            self._preamble_keys.popitem(last=False)
        try:
            from backend.core.ouroboros.battle_test.narrative_channel import (
                NarrativeKind,
                get_default_channel,
            )
            get_default_channel().emit_complete(
                op_id=op_id, phase=f"generate:{round_index}",
                kind=NarrativeKind.TOOL_PREAMBLE, prose=preamble,
                provider=provider,
            )
        except Exception:  # noqa: BLE001
            logger.debug("[claude_style_transport] preamble emit failed",
                         exc_info=True)

    def _subscribe_narrative(self) -> None:
        """Hear every frame that commits to the NarrativeChannel — intent,
        preamble, plan prose, repair prose, postmortem, dream — and render
        it. One subscription is what made the default surface silent: the
        producers wrote frames nobody on this transport was listening for."""
        try:
            from backend.core.ouroboros.battle_test.narrative_channel import (
                get_default_channel,
            )
            get_default_channel().add_commit_listener(self._on_narrative_commit)
        except Exception:  # noqa: BLE001
            logger.debug("[claude_style_transport] narrative subscribe failed",
                         exc_info=True)

    def _on_narrative_commit(self, frame: Any) -> None:
        """Render one committed frame through the shared renderer (density
        gate, provenance footing, glyph and tint all live there) into this
        transport's one seam. NEVER raises into the channel."""
        try:
            from backend.core.ouroboros.battle_test.narrative_renderer import (
                render_to_printer,
            )
            render_to_printer(
                frame, self._print_narrative,
                op_active=False, max_chars_per_line=line_chars(),
            )
        except Exception:  # noqa: BLE001
            logger.debug("[claude_style_transport] narrative render failed",
                         exc_info=True)

    def _print_narrative(self, markup: str, **_kw: Any) -> None:
        """The renderer's print sink: one wire frame per line, so a wrapped
        paragraph lands on the cockpit canvas row by row."""
        for line in str(markup).split("\n"):
            if line.strip():
                self._safe_print(line)

    def _safe_print(self, text: str) -> None:
        """Console.print with defensive try/except. Falls back to
        logger DEBUG if console is missing or print raises.

        ONE seam for both surfaces. The cockpit mirror fires here, before
        the local print and independently of it, so a daemon whose console
        is a log file still reaches every attached terminal, and a mirror
        fault can never cost the local render. The raw markup travels; each
        client fits it to its own canvas.
        """
        mirror = self.markup_mirror
        if mirror is not None:
            try:
                mirror(text)
            except Exception:  # noqa: BLE001 — a mirror never breaks the render
                logger.debug("[claude_style_transport] mirror degraded",
                             exc_info=True)
        try:
            console = self._console
            if console is not None and hasattr(console, "print"):
                console.print(text, highlight=False)
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        logger.debug("[claude_style_transport] %s", text)

    # -- RenderBackend Protocol (CC2.2) -----------------------------
    # ClaudeStyleTransport doubles as a RenderBackend so it can
    # consume FILE_REF events from the conductor and render them as
    # Claude-style "Update(<path>) | Added N, removed M" blocks.
    # This is purely additive — the transport's primary surface
    # remains the CommProtocol send() above.

    name: str = "claude_style"

    _HANDLED_KINDS: frozenset = frozenset({
        "FILE_REF", "PHASE_BEGIN", "REASONING_TOKEN", "PHASE_END",
    })
    _NO_OP_KINDS: frozenset = frozenset({
        "STATUS_TICK", "MODAL_PROMPT", "MODAL_DISMISS",
        "THREAD_TURN", "BACKEND_RESET",
    })

    def notify(self, event: Any) -> None:
        """RenderBackend Protocol — receive RenderEvents from the
        conductor. FILE_REF renders an Update block; the stream triplet
        (PHASE_BEGIN / REASONING_TOKEN / PHASE_END) carries the model's
        generation to attached cockpits as it is written — the surface
        that had no producer on a headless daemon, because the only
        producer was the TTY-gated stream renderer (2026-09-06). Other
        kinds are documented no-ops (CommProtocol handles ops via send())."""
        if event is None:
            return
        try:
            kind = getattr(event, "kind", None)
            kind_value = (
                getattr(kind, "value", None) or str(kind or "")
            )
            if kind_value == "FILE_REF":
                self._handle_file_ref(event)
            elif kind_value == "PHASE_BEGIN":
                metadata = getattr(event, "metadata", None) or {}
                self._stream_begin(
                    str(getattr(event, "op_id", "") or ""),
                    str(metadata.get("provider", "") or ""),
                )
            elif kind_value == "REASONING_TOKEN":
                self._stream_token(
                    str(getattr(event, "op_id", "") or ""),
                    str(getattr(event, "content", "") or ""),
                )
            elif kind_value == "PHASE_END":
                self._stream_end(str(getattr(event, "op_id", "") or ""))
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[claude_style_transport] notify failed", exc_info=True,
            )

    # -- the token stream ---------------------------------------------

    def _stream_begin(self, op_id: str, provider: str) -> None:
        prior = self._streams.pop(op_id, None)
        if prior is not None:
            self._cancel_flush(prior)
        self._streams[op_id] = _StreamState(op_id=op_id, provider=provider)
        logger.info(
            "[claude_style_transport] stream begin op=%s provider=%s "
            "(carrying the token tail to cockpits)", op_id, provider,
        )

    def _stream_token(self, op_id: str, content: str) -> None:
        """Append one token and carry the tail to cockpits, coalesced.

        The first token of a stream goes out at once — the operator sees
        the model start writing the instant it does. Tokens inside the
        flush window ride one frame; a trailing flush is scheduled on the
        running loop so the last tokens before a pause never wait for the
        next token to arrive. A stream that never announced itself is
        opened here (a late-registered backend must not drop the op)."""
        if not content:
            return
        state = self._streams.get(op_id)
        if state is None:
            state = _StreamState(op_id=op_id)
            self._streams[op_id] = state
        state.text = (state.text + content)[-_STREAM_BUFFER_CHARS:]
        state.tokens += 1
        window = stream_flush_ms() / 1000.0
        elapsed = time.monotonic() - state.last_flush
        if elapsed >= window:
            self._stream_flush(op_id)
            return
        if state.flush_handle is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._stream_flush(op_id)       # no loop to defer on
                return
            state.flush_handle = loop.call_later(
                max(0.0, window - elapsed), self._stream_flush, op_id,
            )

    def _stream_flush(self, op_id: str, *, done: bool = False) -> None:
        state = self._streams.get(op_id)
        if state is None:
            return
        self._cancel_flush(state)
        state.last_flush = time.monotonic()
        try:
            from backend.core.ouroboros.battle_test.stream_renderer import (
                publish_inflight_tail,
            )
            publish_inflight_tail(
                op_id, state.text, done=done, sink=self.telemetry_mirror,
            )
        except Exception:  # noqa: BLE001 — a dropped frame is a frame of smoothness
            logger.debug("[claude_style_transport] inflight publish degraded",
                         exc_info=True)

    def _stream_end(self, op_id: str) -> None:
        """The stream is complete: one final frame clears the tail (the
        outcome lines that follow are the record; the tail was the view)."""
        state = self._streams.get(op_id)
        if state is None:
            return
        self._stream_flush(op_id, done=True)
        self._streams.pop(op_id, None)

    @staticmethod
    def _cancel_flush(state: _StreamState) -> None:
        handle, state.flush_handle = state.flush_handle, None
        if handle is not None:
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001
                pass

    def flush(self) -> None:
        """RenderBackend Protocol — no-op for this transport."""
        return

    def shutdown(self) -> None:
        """RenderBackend Protocol — release the narrative subscription and
        let in-flight narration go. A transport that outlives its console
        must not keep rendering into it."""
        try:
            from backend.core.ouroboros.battle_test.narrative_channel import (
                get_default_channel,
            )
            get_default_channel().remove_commit_listener(self._on_narrative_commit)
        except Exception:  # noqa: BLE001
            pass
        for task in tuple(self._narration_tasks):
            try:
                task.cancel()
            except Exception:  # noqa: BLE001
                pass
        for state in tuple(self._streams.values()):
            self._cancel_flush(state)
        self._streams.clear()

    def _handle_file_ref(self, event: Any) -> None:
        """FILE_REF → render as Claude-style Update(<path>) block.

        Format::

          Update(<path>)
            Added N lines, removed M lines
            [diff hunks first ~5 lines, dim]
        """
        try:
            metadata = getattr(event, "metadata", None) or {}
            path = str(metadata.get("path", "") or "")
            if not path:
                return
            line = metadata.get("line")
            line_repr = f":{line}" if line else ""
            # Diff stats from metadata if present (added/removed)
            added = metadata.get("added_lines")
            removed = metadata.get("removed_lines")
            diff_text = str(metadata.get("diff_text", "") or "")
            self._safe_print(
                f"  [bold]Update[/bold]("
                f"[{_SEM['neural']}]{path}{line_repr}[/])"
            )
            if added is not None or removed is not None:
                stats = []
                if added is not None:
                    stats.append(f"Added {added} lines")
                if removed is not None:
                    stats.append(f"removed {removed} lines")
                if stats:
                    self._safe_print(
                        f"  [dim]{', '.join(stats)}[/dim]"
                    )
            elif diff_text:
                # Fallback: count + and - lines from diff_text
                added_n = sum(
                    1 for ln in diff_text.splitlines()
                    if ln.startswith("+") and not ln.startswith("+++")
                )
                removed_n = sum(
                    1 for ln in diff_text.splitlines()
                    if ln.startswith("-") and not ln.startswith("---")
                )
                if added_n or removed_n:
                    self._safe_print(
                        f"  [dim]Added {added_n} lines, "
                        f"removed {removed_n} lines[/dim]"
                    )
                # First 5 lines of diff for context
                preview_lines = diff_text.splitlines()[:5]
                for prev in preview_lines:
                    color = (
                        "green" if prev.startswith("+")
                        else "red" if prev.startswith("-")
                        else "dim"
                    )
                    self._safe_print(f"    [{color}]{prev[:80]}[/{color}]")
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[claude_style_transport] _handle_file_ref failed",
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# FlagRegistry registration — auto-discovered
# ---------------------------------------------------------------------------


def register_flags(registry: Any) -> int:
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category,
            FlagSpec,
            FlagType,
            Relevance,
        )
    except Exception:  # noqa: BLE001 — defensive
        return 0
    all_postures_relevant = {
        "EXPLORE": Relevance.RELEVANT,
        "CONSOLIDATE": Relevance.RELEVANT,
        "HARDEN": Relevance.RELEVANT,
        "MAINTAIN": Relevance.RELEVANT,
    }
    specs = [
        FlagSpec(
            name=_FLAG_RENDER_MODE,
            type=FlagType.STR,
            default="CLAUDE",
            description=(
                "Operator-selected per-op rendering mode. Closed "
                "taxonomy: 'CLAUDE' (default — terse one-line-per-op "
                "Claude Code idiom) or 'SERPENT' (legacy SerpentFlow "
                "multi-line per-op blocks). Unknown values fall back "
                "to CLAUDE — operator typo doesn't restore noise. "
                "Hot-revert via env."
            ),
            category=Category.OBSERVABILITY,
            source_file=(
                "backend/core/ouroboros/governance/"
                "claude_style_transport.py"
            ),
            example="CLAUDE",
            since="v1.0",
            posture_relevance=all_postures_relevant,
        ),
        FlagSpec(
            name=_FLAG_CLAUDE_SHOW_HEARTBEATS,
            type=FlagType.BOOL,
            default=False,
            description=(
                "Whether HEARTBEAT messages emit per-phase ticks "
                "under Claude-style rendering. Default false — the "
                "active line carries enough state without per-tick "
                "chatter. Operators flip true for FULL-debug "
                "visibility on phase transitions."
            ),
            category=Category.OBSERVABILITY,
            source_file=(
                "backend/core/ouroboros/governance/"
                "claude_style_transport.py"
            ),
            example="false",
            since="v1.0",
        ),
        FlagSpec(
            name=_FLAG_LINE_CHARS,
            type=FlagType.INT,
            default=_DEFAULT_LINE_CHARS,
            description=(
                "Characters of goal summary on a Claude-style op line, and "
                "the wrap width of the organism's narration beneath it. "
                "Summaries are cut at a word boundary with an ellipsis."
            ),
            category=Category.OBSERVABILITY,
            source_file=(
                "backend/core/ouroboros/governance/"
                "claude_style_transport.py"
            ),
            example=str(_DEFAULT_LINE_CHARS),
            since="v1.1",
        ),
        FlagSpec(
            name=_FLAG_DETAIL_LINES,
            type=FlagType.INT,
            default=_DEFAULT_DETAIL_LINES,
            description=(
                "Lines of the model's own reason shown beneath an outcome "
                "line (a no-op's explanation, a gate's) before the rest is "
                "elided."
            ),
            category=Category.OBSERVABILITY,
            source_file=(
                "backend/core/ouroboros/governance/"
                "claude_style_transport.py"
            ),
            example=str(_DEFAULT_DETAIL_LINES),
            since="v1.1",
        ),
        FlagSpec(
            name=_FLAG_STREAM_FLUSH_MS,
            type=FlagType.INT,
            default=_DEFAULT_STREAM_FLUSH_MS,
            description=(
                "Coalescing window, in milliseconds, for the in-flight token "
                "stream the Claude-style transport carries to attached "
                "cockpits: tokens arriving inside it ride one frame. The "
                "first token of a stream always goes out at once."
            ),
            category=Category.OBSERVABILITY,
            source_file=(
                "backend/core/ouroboros/governance/"
                "claude_style_transport.py"
            ),
            example=str(_DEFAULT_STREAM_FLUSH_MS),
            since="v1.1",
        ),
    ]
    registry.bulk_register(specs, override=True)
    return len(specs)


# ---------------------------------------------------------------------------
# AST invariants — auto-discovered
# ---------------------------------------------------------------------------


_FORBIDDEN_RICH_PREFIX: tuple = ("rich",)
_FORBIDDEN_AUTHORITY_MODULES: tuple = (
    "backend.core.ouroboros.governance.orchestrator",
    "backend.core.ouroboros.governance.policy",
    "backend.core.ouroboros.governance.iron_gate",
    "backend.core.ouroboros.governance.risk_tier",
    "backend.core.ouroboros.governance.risk_tier_floor",
    "backend.core.ouroboros.governance.change_engine",
    "backend.core.ouroboros.governance.candidate_generator",
    "backend.core.ouroboros.governance.gate",
    "backend.core.ouroboros.governance.semantic_guardian",
    "backend.core.ouroboros.governance.semantic_firewall",
    "backend.core.ouroboros.governance.providers",
    "backend.core.ouroboros.governance.doubleword_provider",
    "backend.core.ouroboros.governance.urgency_router",
    "backend.core.ouroboros.governance.cancel_token",
    "backend.core.ouroboros.governance.conversation_bridge",
)


_EXPECTED_RENDER_MODE = frozenset({"CLAUDE", "SERPENT"})
_EXPECTED_OP_STATUS_GLYPH = frozenset({
    "ACTIVE", "RUNNING", "DONE", "FAILED", "CANCELLED", "NOOP",
})


def _imported_modules(tree: Any) -> List:
    import ast
    out: List = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                out.append((node.lineno, mod))
    return out


def _enum_member_names(tree: Any, class_name: str) -> List[str]:
    import ast
    out: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id.isupper():
                        out.append(tgt.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name,
            ):
                if stmt.target.id.isupper():
                    out.append(stmt.target.id)
    return out


def _validate_no_rich_import(tree: Any, source: str) -> tuple:
    del source
    violations: List[str] = []
    for lineno, mod in _imported_modules(tree):
        for forbidden in _FORBIDDEN_RICH_PREFIX:
            if mod == forbidden or mod.startswith(forbidden + "."):
                violations.append(
                    f"line {lineno}: forbidden rich import: {mod!r}"
                )
    return tuple(violations)


def _validate_no_authority_imports(tree: Any, source: str) -> tuple:
    del source
    violations: List[str] = []
    for lineno, mod in _imported_modules(tree):
        if mod in _FORBIDDEN_AUTHORITY_MODULES:
            violations.append(
                f"line {lineno}: forbidden authority import: {mod!r}"
            )
    return tuple(violations)


def _validate_render_mode_closed(tree: Any, source: str) -> tuple:
    del source
    found = set(_enum_member_names(tree, "RenderMode"))
    if found != _EXPECTED_RENDER_MODE:
        return (
            f"RenderMode members {sorted(found)} != expected "
            f"{sorted(_EXPECTED_RENDER_MODE)}",
        )
    return ()


def _validate_op_status_glyph_closed(tree: Any, source: str) -> tuple:
    del source
    found = set(_enum_member_names(tree, "OpStatusGlyph"))
    if found != _EXPECTED_OP_STATUS_GLYPH:
        return (
            f"OpStatusGlyph members {sorted(found)} != expected "
            f"{sorted(_EXPECTED_OP_STATUS_GLYPH)}",
        )
    return ()


def _validate_discovery_symbols_present(
    tree: Any, source: str,
) -> tuple:
    del source
    import ast
    needed = {"register_flags", "register_shipped_invariants"}
    found: set = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in needed:
                found.add(node.name)
    missing = needed - found
    if missing:
        return (f"missing discovery symbols: {sorted(missing)}",)
    return ()


_TARGET_FILE = (
    "backend/core/ouroboros/governance/claude_style_transport.py"
)


def register_shipped_invariants() -> List:
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except Exception:  # noqa: BLE001 — defensive
        return []
    return [
        ShippedCodeInvariant(
            invariant_name="claude_style_transport_no_rich_import",
            target_file=_TARGET_FILE,
            description=(
                "claude_style_transport.py MUST NOT import rich.* "
                "at module top — Rich is consumed via the duck-typed "
                "console reference passed in at construction. Lazy "
                "imports inside methods are allowed."
            ),
            validate=_validate_no_rich_import,
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "claude_style_transport_no_authority_imports"
            ),
            target_file=_TARGET_FILE,
            description=(
                "Transport must NOT import authority modules. "
                "Same descriptive-only contract as render_conductor."
            ),
            validate=_validate_no_authority_imports,
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "claude_style_transport_render_mode_closed"
            ),
            target_file=_TARGET_FILE,
            description=(
                "RenderMode enum members must exactly match the "
                "documented closed set (CLAUDE, SERPENT). Adding a "
                "mode requires coordinated harness-wire update."
            ),
            validate=_validate_render_mode_closed,
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "claude_style_transport_op_status_glyph_closed"
            ),
            target_file=_TARGET_FILE,
            description=(
                "OpStatusGlyph enum members must exactly match the "
                "documented 6-value closed set (ACTIVE, RUNNING, "
                "DONE, FAILED, CANCELLED, NOOP). Adding a glyph "
                "requires coordinated handler update."
            ),
            validate=_validate_op_status_glyph_closed,
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "claude_style_transport_discovery_symbols_present"
            ),
            target_file=_TARGET_FILE,
            description=(
                "register_flags + register_shipped_invariants must "
                "be module-level so dynamic discovery picks them up."
            ),
            validate=_validate_discovery_symbols_present,
        ),
    ]


__all__ = [
    "CLAUDE_STYLE_TRANSPORT_SCHEMA_VERSION",
    "ClaudeStyleTransport",
    "OpStatusGlyph",
    "RenderMode",
    "register_flags",
    "register_shipped_invariants",
    "resolve_render_mode",
    "show_heartbeats",
]
