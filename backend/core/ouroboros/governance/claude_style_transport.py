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
     ``· <Sensor>(<short_id>) <summary>`` line; DECISION prints
     ``✓ <Sensor>(<short_id>) done in Xs`` (success) or
     ``✗ <Sensor>(<short_id>) shed: <reason> (Xs)`` (failure). No
     per-phase emoji cascade; no multi-line block per op.
  3. **Bullet markers from a closed taxonomy** —
     :class:`OpStatusGlyph`. ``·`` (active), ``●`` (running with
     work), ``✓`` (done), ``✗`` (failed), ``◌`` (cancelled),
     ``⏭`` (no-op). AST-pinned. Adding a glyph requires coordinated
     update.
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

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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


class OpStatusGlyph(str, enum.Enum):
    """Closed taxonomy of per-op status bullets. Matches Claude
    Code's restraint: 6 bullets total, each with semantic meaning."""

    ACTIVE = "·"        # op in flight (just sensed; not yet routing)
    RUNNING = "●"       # op actively running (synthesizing / verifying)
    DONE = "✓"          # op completed successfully
    FAILED = "✗"        # op shed (failed)
    CANCELLED = "◌"     # op cancelled mid-flight
    NOOP = "⏭"          # triage NO_OP — op was unnecessary


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
        # CC2.1 — running counters for TASK_LIST composer field
        self._done_count: int = 0
        self._failed_count: int = 0

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
        if reason_code.startswith("boot_recovery_"):
            self._boot_recovery_count += 1
            if self._boot_recovery_count == 1:
                self._safe_print(
                    "[dim]· boot recovery │ "
                    "reconciling stale ledger entries...[/dim]"
                )
            return
        if risk_tier == "routing":
            return  # internal routing decision, not an op start

        # Flush boot recovery summary on first real INTENT
        if self._boot_recovery_count > 0 and not self._boot_recovery_flushed:
            self._boot_recovery_flushed = True
            self._safe_print(
                f"[dim]· boot recovery │ "
                f"{self._boot_recovery_count} stale entries reconciled"
                f"[/dim]"
            )
            self._safe_print("")

        sensor = self._infer_sensor(payload)
        summary = self._summarize(payload)
        short = _short_id(op_id)
        state = _OpState(
            op_id=op_id,
            short_id=short,
            sensor=sensor,
            summary=summary,
            risk_tier=str(payload.get("risk_tier", "") or "").upper(),
            target_files=tuple(payload.get("target_files", []) or []),
        )
        self._op_state[op_id] = state
        # CC2.1 — feed composer with current ACTIVE_OP + TASK_LIST
        self._feed_composer()
        # Render: `· <Sensor>(<short_id>) <summary>`
        target_repr = ""
        if state.target_files:
            primary = state.target_files[0]
            if isinstance(primary, str) and len(primary) > 50:
                parts = primary.split("/")
                primary = ".../" + "/".join(parts[-2:])
            target_repr = f" [dim]{primary}[/dim]"
        risk_repr = ""
        if state.risk_tier and state.risk_tier not in ("SAFE_AUTO", "LOW"):
            color = (
                "yellow" if state.risk_tier == "MEDIUM"
                else "red"
            )
            risk_repr = f" [[{color}]{state.risk_tier}[/{color}]]"
        self._safe_print(
            f"{OpStatusGlyph.ACTIVE.value} "
            f"[bold]{sensor}[/bold]([dim]{short}[/dim])"
            f"{risk_repr} {summary}"
            f"{target_repr}"
        )

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

        Only the COMPLETION renders. The start event drives a spinner in
        SerpentFlow; a line stream has no spinner, and printing both would
        show every tool twice.
        """
        tool_name = str(payload.get("tool_name", "") or "").strip()
        if tool_name:
            if payload.get("tool_starting"):
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
        # No `rich` import here — an authority invariant of this module,
        # AST-pinned. Escaping every "[" is a strict superset of what
        # rich.markup.escape neutralises, so model-controlled text can
        # never open a tag.
        def _escape(s: str) -> str:
            return str(s).replace("[", "\\[")

        tone = _SEM["death"] if status not in ("success", "ok") else _SEM["dim"]
        head = f"  [{_SEM['dim']}]⏺[/] [bold]{_escape(tool_name)}[/bold]"
        if args:
            head += f"([dim]{_escape(args[:60])}[/dim])"
        if status not in ("success", "ok"):
            head += f" [{tone}]{_escape(status)}[/]"
        self._safe_print(head)
        if result:
            self._safe_print(
                f"    [{_SEM['dim']}]⎿[/]  [dim]{_escape(result[:120])}[/dim]"
            )

    def _handle_decision(
        self, op_id: str, payload: Dict[str, Any],
    ) -> None:
        """DECISION — op outcome. Render the closing line."""
        outcome = str(payload.get("outcome", "") or "").lower()
        state = self._op_state.pop(op_id, None)
        if state is None:
            # Decision without prior INTENT — common at boot for
            # orphan reconciliation. Suppress.
            return
        elapsed = _format_elapsed(state.started_monotonic)

        if outcome in ("completed", "applied", "auto_approved"):
            files = payload.get("files_changed") or payload.get(
                "affected_files",
            ) or []
            files_repr = ""
            if files:
                first = str(files[0])
                if len(first) > 40:
                    parts = first.split("/")
                    first = ".../" + "/".join(parts[-2:])
                files_repr = f" [dim]{first}[/dim]"
                if len(files) > 1:
                    files_repr += f" [dim]+{len(files) - 1}[/dim]"
            self._safe_print(
                f"[{_SEM['success']}]{OpStatusGlyph.DONE.value}[/] "
                f"[bold]{state.sensor}[/bold]([dim]{state.short_id}[/dim])"
                f" done [dim]({elapsed})[/dim]{files_repr}"
            )
            self._done_count += 1
            self._feed_composer()
            return

        if outcome in ("failed", "postmortem"):
            reason = str(payload.get("reason_code", "") or "")[:60]
            self._safe_print(
                f"[{_SEM['death']}]{OpStatusGlyph.FAILED.value}[/] "
                f"[bold]{state.sensor}[/bold]([dim]{state.short_id}[/dim])"
                f" shed: [{_SEM['death']}]{reason}[/] [dim]({elapsed})[/dim]"
            )
            self._failed_count += 1
            self._feed_composer()
            return

        if outcome == "noop":
            reason = str(payload.get("reason_code", "") or "")[:50]
            reason_repr = f" [dim]{reason}[/dim]" if reason else ""
            self._safe_print(
                f"[dim]{OpStatusGlyph.NOOP.value}[/dim] "
                f"[bold]{state.sensor}[/bold]([dim]{state.short_id}[/dim])"
                f" no-op{reason_repr} [dim]({elapsed})[/dim]"
            )
            return

        if outcome == "notify_apply":
            files = payload.get("target_files", []) or []
            files_repr = ""
            if files:
                first = str(files[0])[:40]
                files_repr = f" [{_SEM['heal']}]{first}[/]"
                if len(files) > 1:
                    files_repr += f" [dim]+{len(files) - 1}[/dim]"
            self._safe_print(
                f"[{_SEM['heal']}]{OpStatusGlyph.RUNNING.value}[/] "
                f"[bold]{state.sensor}[/bold]([dim]{state.short_id}[/dim])"
                f" [{_SEM['heal']}]NOTIFY[/] auto-applying"
                f"{files_repr} [dim]({elapsed})[/dim]"
            )
            return

        if outcome == "escalated":
            reason = str(payload.get("reason_code", "") or "")[:50]
            self._safe_print(
                f"[{_SEM['heal']}]{OpStatusGlyph.RUNNING.value}[/] "
                f"[bold]{state.sensor}[/bold]([dim]{state.short_id}[/dim])"
                f" [{_SEM['heal']}]escalated[/] [dim]{reason}"
                f" ({elapsed})[/dim]"
            )
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
        sensor = state.sensor if state else "Operation"
        short = state.short_id if state else _short_id(op_id)
        reason = str(payload.get("root_cause", "unknown") or "unknown")[:60]
        self._safe_print(
            f"[{_SEM['death']}]{OpStatusGlyph.FAILED.value}[/] "
            f"[bold]{sensor}[/bold]([dim]{short}[/dim])"
            f" postmortem: [{_SEM['death']}]{reason}[/] [dim]({elapsed})[/dim]"
        )

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
        """One-line summary of the op's goal. Truncated to 70 chars
        for terminal width hygiene."""
        goal = str(payload.get("goal", "") or "").strip()
        if len(goal) > 70:
            goal = goal[:67] + "..."
        return goal

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

    _HANDLED_KINDS: frozenset = frozenset({"FILE_REF"})
    _NO_OP_KINDS: frozenset = frozenset({
        "PHASE_BEGIN", "PHASE_END", "REASONING_TOKEN",
        "STATUS_TICK", "MODAL_PROMPT", "MODAL_DISMISS",
        "THREAD_TURN", "BACKEND_RESET",
    })

    def notify(self, event: Any) -> None:
        """RenderBackend Protocol — receive RenderEvents from the
        conductor. ClaudeStyleTransport only handles FILE_REF;
        everything else is a documented no-op (CommProtocol surface
        handles ops via send())."""
        if event is None:
            return
        try:
            kind = getattr(event, "kind", None)
            kind_value = (
                getattr(kind, "value", None) or str(kind or "")
            )
            if kind_value == "FILE_REF":
                self._handle_file_ref(event)
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[claude_style_transport] notify failed", exc_info=True,
            )

    def flush(self) -> None:
        """RenderBackend Protocol — no-op for this transport."""
        return

    def shutdown(self) -> None:
        """RenderBackend Protocol — no-op for this transport."""
        return

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
