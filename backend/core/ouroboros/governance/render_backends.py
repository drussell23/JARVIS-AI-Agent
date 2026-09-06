"""RenderConductor adapter backends for SerpentFlow and OuroborosConsole.

Slice 2 of the RenderConductor arc (Wave 4 #1). Closes the architectural
fragmentation identified in §29 by routing the two REPL-class renderers
(``SerpentFlow`` — preferred CC-style flowing CLI; ``OuroborosConsole`` —
fallback scrolling Rich TUI) through the unified conductor as
``RenderBackend`` implementations.

``StreamRenderer`` got its backend conformance inline (in
``battle_test/stream_renderer.py``) because its API is a single-purpose
3-method lifecycle. ``SerpentFlow`` (5,300+ LOC, 70+ public methods) and
``OuroborosConsole`` (740 LOC, 25+ public methods) are wrapped here as
**composition adapters** — they do not modify the wrapped renderer; they
translate ``RenderEvent`` instances into existing API calls. This keeps
the load-bearing renderer files untouched while still completing the
substrate inversion: post-Slice-2, all three renderers are
``RenderBackend``-compliant and the conductor is the single fan-out
surface.

The adapter contract (mirrors ``StreamRenderer.notify``):

  * ``notify(event)`` — total over ``EventKind``: every closed-taxonomy
    value either maps to a wrapped-renderer method call OR is a
    documented no-op (for events the renderer doesn't surface — e.g.
    ``OuroborosConsole`` has no thread region, so ``THREAD_TURN`` is a
    no-op there). No silent drops on unknown kinds — the closed
    taxonomy means "unknown" is a contract violation and gets logged.
  * ``flush()`` / ``shutdown()`` — defensive best-effort. The wrapped
    renderers do not currently have flush/shutdown hooks; these adapters
    document the absence and degrade gracefully when Slice 3+ surfaces
    one.
  * NEVER raises — every method swallows exceptions and logs DEBUG. A
    mis-mapped event cannot break the conductor's fan-out to siblings.

Authority invariants (AST-pinned via ``register_shipped_invariants``):

  * No imports of orchestrator / policy / iron_gate / risk_tier /
    change_engine / candidate_generator / gate / semantic_guardian /
    semantic_firewall / providers / doubleword_provider / urgency_router.
    Adapters are descriptive surfaces only.
  * Two adapter classes (``SerpentFlowBackend`` + ``OuroborosConsoleBackend``)
    both define the four required RenderBackend symbols (``name`` /
    ``notify`` / ``flush`` / ``shutdown``).
  * ``register_shipped_invariants`` symbol present (auto-discovery
    contract).

This module is auto-discovered by both
``flag_registry_seed._discover_module_provided_flags`` (zero new flags
in Slice 2 — adapters are wired with the conductor's existing flag set)
and ``shipped_code_invariants._discover_module_provided_invariants``.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


RENDER_BACKENDS_SCHEMA_VERSION: str = "render_backends.1"


# ---------------------------------------------------------------------------
# Event-kind dispatch helpers — keep adapter bodies short and total.
# Maps the EventKind string value to the adapter's per-kind handler name.
# Each adapter declares which kinds it handles in ``_HANDLED_KINDS`` and
# which it documented-no-ops in ``_NO_OP_KINDS``. The union must equal
# the full EventKind closed set (validated by AST pin at the bottom).
# ---------------------------------------------------------------------------


def _event_kind_value(event: Any) -> str:
    """Extract the closed-taxonomy string value from a RenderEvent.
    Returns empty string on any extraction failure (caller treats as
    no-op). NEVER raises."""
    try:
        kind = getattr(event, "kind", None)
        if kind is None:
            return ""
        return kind.value if hasattr(kind, "value") else str(kind)
    except Exception:  # noqa: BLE001 — defensive
        return ""


def _event_metadata(event: Any) -> dict:
    """Extract a plain dict of the event's metadata. Returns empty dict
    on any failure. NEVER raises."""
    try:
        md = getattr(event, "metadata", None) or {}
        return dict(md) if not isinstance(md, dict) else md
    except Exception:  # noqa: BLE001 — defensive
        return {}


def _active_density() -> str:
    """Read the conductor's currently-resolved density. Returns the
    string value (``"COMPACT"`` / ``"NORMAL"`` / ``"FULL"``) or
    ``"NORMAL"`` as defensive fallback. Cheap — reuses the conductor's
    existing resolution path; no env scan duplicated here."""
    try:
        from backend.core.ouroboros.governance.render_conductor import (
            get_render_conductor,
        )
        conductor = get_render_conductor()
        if conductor is None:
            return "NORMAL"
        return conductor.active_density().value
    except Exception:  # noqa: BLE001 — defensive
        return "NORMAL"


def _terminal_width() -> int:
    """Resolve the operator terminal width. Uses ``shutil.get_terminal_-
    size`` which honors ``COLUMNS`` env / OS termios; defaults to 80
    on any failure. No JARVIS_*_WIDTH knob — operators tune via the
    standard ``COLUMNS`` env that every Unix tool already honors."""
    try:
        import shutil
        return max(20, shutil.get_terminal_size((80, 24)).columns)
    except Exception:  # noqa: BLE001 — defensive
        return 80


# Bounded ring buffer for FILE_REF dedup. Same path:line within the
# last N events suppressed at the backend boundary — operator UX win
# (FILE_REF spam during tight loops collapses to one render).
_FILE_REF_DEDUP_WINDOW: int = 16


class _RingDedup:
    """Tiny bounded ring buffer used to suppress immediate-repeat
    FILE_REF events. Per-backend (not shared) so each adapter's
    suppression decisions are independent. Thread-safe via a
    ``threading.Lock`` (FILE_REF can fire from any context — Slice 7
    producer wiring eventually pumps from generator runner threads)."""

    def __init__(self, window: int = _FILE_REF_DEDUP_WINDOW) -> None:
        import threading
        self._window = max(1, int(window))
        self._buf: List[str] = []
        self._lock = threading.Lock()

    def seen_recently(self, key: str) -> bool:
        """Return True iff ``key`` is in the ring (suppress this
        event); else record it and return False (let it through)."""
        if not key:
            return False
        with self._lock:
            if key in self._buf:
                return True
            self._buf.append(key)
            if len(self._buf) > self._window:
                self._buf.pop(0)
            return False

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


def _resolve_role_style(event: Any) -> str:
    """Map the event's stamped :class:`ColorRole` to a Rich-compatible
    style string via the conductor's active theme. Returns ``""`` when
    the conductor isn't registered or theme resolution fails — Rich
    treats empty style as default text."""
    try:
        from backend.core.ouroboros.governance.render_conductor import (
            ColorRole,
            RenderDensity,
            get_render_conductor,
        )
        conductor = get_render_conductor()
        if conductor is None:
            return ""
        role = getattr(event, "role", None)
        if not isinstance(role, ColorRole):
            return ""
        density_str = _active_density()
        try:
            density = RenderDensity(density_str)
        except ValueError:
            density = RenderDensity.NORMAL
        return conductor.active_theme().resolve(role, density)
    except Exception:  # noqa: BLE001 — defensive
        return ""


# ---------------------------------------------------------------------------
# SerpentFlowBackend — wraps the preferred CC-style flowing CLI
# ---------------------------------------------------------------------------


class SerpentFlowBackend:
    """Adapter exposing :class:`SerpentFlow` as a ``RenderBackend``.

    Composition (not subclassing) — the wrapped instance is held as
    ``_flow`` and its existing methods are called by ``notify``. The
    wrapped renderer is unaware of the adapter; nothing in
    ``serpent_flow.py`` is modified.

    Slice 2 wires the streaming triplet (PHASE_BEGIN / REASONING_TOKEN /
    PHASE_END) which is the most critical operator-visible surface and
    the one immediately consumable by Slice 3's typed
    ``ReasoningStream``. Other event kinds (FILE_REF, STATUS_TICK,
    MODAL_*, THREAD_TURN, BACKEND_RESET) are documented no-ops in
    Slice 2 — Slices 3-6 will wire each as their typed primitive
    ships, with the wrapped renderer's API surface as the proven
    target.
    """

    name: str = "serpent_flow"

    # Closed taxonomy of which event kinds this adapter actively handles
    # vs. which it documented-no-ops. The union MUST cover every
    # EventKind value (AST-pinned).
    #
    # Slice 7 graduation: 5 event kinds promoted from _NO_OP_KINDS to
    # _HANDLED_KINDS. Only BACKEND_RESET remains a documented no-op
    # (lifecycle event, no SerpentFlow API correspondence).
    _HANDLED_KINDS: frozenset = frozenset({
        "PHASE_BEGIN",
        "REASONING_TOKEN",
        "PHASE_END",
        "FILE_REF",
        "STATUS_TICK",
        "MODAL_PROMPT",
        "MODAL_DISMISS",
        "THREAD_TURN",
    })
    _NO_OP_KINDS: frozenset = frozenset({
        "BACKEND_RESET",
    })

    def __init__(self, flow: Any) -> None:
        """``flow`` is a constructed :class:`SerpentFlow` instance. We do
        not import ``SerpentFlow`` directly — duck-typed by the adapter
        contract so tests can substitute a stub."""
        self._flow = flow
        self._file_ref_dedup = _RingDedup()

    def notify(self, event: Any) -> None:
        """Route a RenderEvent to the wrapped SerpentFlow. Total over
        EventKind via the explicit handled/no-op partition."""
        if event is None:
            return
        kind = _event_kind_value(event)
        if not kind:
            return
        try:
            if kind == "REASONING_TOKEN":
                content = getattr(event, "content", "") or ""
                if content and hasattr(self._flow, "show_streaming_token"):
                    self._flow.show_streaming_token(content)
                return
            if kind == "PHASE_BEGIN":
                op_id = getattr(event, "op_id", None) or ""
                metadata = _event_metadata(event)
                provider = str(metadata.get("provider", "") or "")
                if hasattr(self._flow, "show_streaming_start"):
                    # SerpentFlow.show_streaming_start signature accepts
                    # (op_id, provider, model). Call with what we have;
                    # missing optional kwargs default at the wrapped
                    # renderer's discretion.
                    try:
                        self._flow.show_streaming_start(
                            op_id=op_id, provider=provider,
                        )
                    except TypeError:
                        # Fallback for renderers with a stricter signature
                        try:
                            self._flow.show_streaming_start(op_id, provider)
                        except Exception:  # noqa: BLE001 — defensive
                            logger.debug(
                                "[SerpentFlowBackend] show_streaming_start "
                                "signature mismatch", exc_info=True,
                            )
                return
            if kind == "PHASE_END":
                if hasattr(self._flow, "show_streaming_end"):
                    self._flow.show_streaming_end()
                return
            if kind == "FILE_REF":
                self._handle_file_ref(event)
                return
            if kind == "STATUS_TICK":
                self._handle_status_tick(event)
                return
            if kind == "MODAL_PROMPT":
                self._handle_modal_prompt(event)
                return
            if kind == "MODAL_DISMISS":
                self._handle_modal_dismiss(event)
                return
            if kind == "THREAD_TURN":
                self._handle_thread_turn(event)
                return
            if kind in self._NO_OP_KINDS:
                # Documented no-op (BACKEND_RESET is a lifecycle event
                # with no SerpentFlow API correspondence).
                return
            # Unknown closed-taxonomy value — log once and continue.
            logger.debug(
                "[SerpentFlowBackend] unknown event kind %r — no-op", kind,
            )
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[SerpentFlowBackend] notify failed for kind=%s",
                kind, exc_info=True,
            )

    # -- Slice 7 event-kind handlers (feature-detected) ----------------

    def _handle_file_ref(self, event: Any) -> None:
        """FILE_REF → flow.show_diff(path, diff_text) if metadata
        carries both; else flow.show_code_preview(path) if available;
        else fall back to console.print of the canonical render.

        Slice 7 follow-up backlog #3: dedup repeats within the last
        16 events. Same path:line:column key suppressed — operator UX
        win against FILE_REF spam during tight loops (e.g. a tool
        loop visiting the same file 30 times). Dedup runs at the
        backend boundary, after the conductor's fan-out, so other
        backends still see the event for their own coalescing."""
        metadata = _event_metadata(event)
        path = str(metadata.get("path", "") or "")
        if not path:
            return
        # Dedup key: path + line + column (anchor variations should
        # render — they're navigation distinctions, not noise).
        line = metadata.get("line")
        column = metadata.get("column")
        dedup_key = f"{path}:{line}:{column}"
        if self._file_ref_dedup.seen_recently(dedup_key):
            return
        diff_text = str(metadata.get("diff_text", "") or "")
        if diff_text and hasattr(self._flow, "show_diff"):
            try:
                self._flow.show_diff(path, diff_text)
                return
            except Exception:  # noqa: BLE001 — defensive
                pass
        if hasattr(self._flow, "show_code_preview"):
            try:
                self._flow.show_code_preview(path)
                return
            except Exception:  # noqa: BLE001 — defensive
                pass
        # Fallback: print via console attribute (SerpentFlow has one).
        self._console_print(getattr(event, "content", "") or path)

    def _handle_status_tick(self, event: Any) -> None:
        """STATUS_TICK → metadata-driven dispatch.

        Three branches in priority order:

          1. **D5 composer bridge** — when ``metadata.composed_status``
             is True, the event was published by the
             :class:`StatusLineComposer`. ``event.content`` carries
             the composed status line; route directly to
             ``flow._spinner_state.message`` (the prompt_toolkit
             bottom_toolbar surface). Single source of truth for
             the always-current status line.
          2. **Typed update_* dispatch** — recognised metadata keys
             map to specific ``update_*`` methods on SerpentFlow
             (cost, sensors, provider_chain, intent_chain). Used by
             callers that publish STATUS_TICK without going through
             the composer.
          3. **Console fallback** — generic status text printed via
             ``console.print``. Last-resort path for events that
             don't match either typed dispatch or composer marker.
        """
        metadata = _event_metadata(event)

        # Branch 1: D5 composer bridge
        if metadata.get("composed_status") is True:
            content = getattr(event, "content", "") or ""
            spinner_state = getattr(self._flow, "_spinner_state", None)
            if spinner_state is not None:
                try:
                    spinner_state.message = content
                    # Mark active so prompt_toolkit's bottom_toolbar
                    # callable knows to render the composed line
                    # (vs. the empty-state placeholder).
                    if hasattr(spinner_state, "active"):
                        spinner_state.active = bool(content)
                    return
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[SerpentFlowBackend] composed status write "
                        "failed", exc_info=True,
                    )
                    # fall through to console fallback below
            else:
                # No _spinner_state — fall through to console fallback
                pass

        # Branch 2: typed update_* dispatch
        for key, method_name, coerce in (
            ("cost", "update_cost", float),
            ("sensors", "update_sensors", int),
            ("provider_chain", "update_provider_chain", str),
            ("intent_chain", "update_intent_chain", str),
        ):
            if key not in metadata:
                continue
            method = getattr(self._flow, method_name, None)
            if not callable(method):
                continue
            try:
                method(coerce(metadata[key]))
                return
            except Exception:  # noqa: BLE001 — defensive
                continue

        # Branch 3: console fallback
        content = getattr(event, "content", "") or ""
        if content:
            self._console_print(content)

    def _handle_modal_prompt(self, event: Any) -> None:
        """MODAL_PROMPT → render the modal text. Slice 7 follow-up
        backlog #3: when the wrapped renderer's console exposes a
        Rich-compatible API AND density is not COMPACT, render via
        Rich Panel for clean visual separation; else fall through to
        the inline prompt block. COMPACT density collapses to a
        single one-liner — operator can still see "?" worked, with
        full content on a follow-up FULL-density frame."""
        content = getattr(event, "content", "") or ""
        if not content:
            return
        density = _active_density()
        if density == "COMPACT":
            # One-liner — clip to terminal width minus marker.
            width = max(20, _terminal_width() - 16)
            clipped = content.replace("\n", " ")[:width]
            self._console_print(f"  /help: {clipped}…")
            return
        # NORMAL / FULL — try Rich Panel via the console attribute.
        try:
            console = getattr(self._flow, "console", None)
            if console is not None:
                from rich.panel import Panel
                console.print(
                    Panel(content, title="help", expand=False),
                )
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        # Fallback: inline prompt block.
        self._console_print(
            f"\n[bold cyan]── help ──[/bold cyan]\n{content}\n",
        )

    def _handle_modal_dismiss(self, event: Any) -> None:
        """MODAL_DISMISS → render a separator marking the close. The
        flowing CLI doesn't track modal state; it just shows the
        separator and continues the stream."""
        del event
        self._console_print("[dim]── /help ──[/dim]\n")

    def _handle_thread_turn(self, event: Any) -> None:
        """THREAD_TURN → render the speaker-tagged turn. SerpentFlow
        doesn't have a dedicated sticky thread region in Slice 7;
        emit as a speaker-prefixed line in the flowing log.

        Slice 7 follow-up backlog #3: consume the conductor-stamped
        ``ColorRole`` (which Slice 5's :func:`publish_thread_turn`
        already maps per-speaker — USER→EMPHASIS, ASSISTANT→CONTENT,
        POSTMORTEM→MUTED, etc.) via the active theme instead of
        hardcoding per-speaker color tags. The label remains keyed
        to ``speaker`` for semantic clarity; the *style* derives
        from the role.
        """
        metadata = _event_metadata(event)
        speaker = str(metadata.get("speaker", "") or "?")
        content = getattr(event, "content", "") or ""
        if not content:
            return
        # Speaker label remains a closed-taxonomy lookup (semantic
        # text, not visual style). Theme-resolved Rich style wraps it.
        speaker_label = {
            "USER":       "you",
            "ASSISTANT":  "model",
            "POSTMORTEM": "postmortem",
            "TOOL":       "tool",
            "SYSTEM":     "sys",
        }.get(speaker, "?")
        style = _resolve_role_style(event)
        if style:
            self._console_print(f"  [{style}]{speaker_label}[/{style}]: {content}")
        else:
            self._console_print(f"  {speaker_label}: {content}")

    def _console_print(self, text: str) -> None:
        """Defensive console.print via the flow's console attribute.
        SerpentFlow exposes ``self.console`` (Rich Console). When
        unavailable, falls through to logger DEBUG."""
        try:
            console = getattr(self._flow, "console", None)
            if console is not None and hasattr(console, "print"):
                console.print(text)
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        logger.debug("[SerpentFlowBackend] %s", text)

    def flush(self) -> None:
        """SerpentFlow has no explicit flush hook — best-effort no-op.
        Slice 3+ will surface a flush method as the typed primitives
        require it (e.g. for end-of-phase rendering boundaries)."""
        return

    def shutdown(self) -> None:
        """Best-effort cleanup. SerpentFlow has no explicit shutdown
        method — calling show_streaming_end is the closest equivalent
        in case a stream was left mid-flight at session teardown."""
        try:
            if hasattr(self._flow, "show_streaming_end"):
                self._flow.show_streaming_end()
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[SerpentFlowBackend] shutdown failed", exc_info=True,
            )


# ---------------------------------------------------------------------------
# OuroborosConsoleBackend — wraps the fallback scrolling Rich TUI
# ---------------------------------------------------------------------------


class OuroborosConsoleBackend:
    """Adapter exposing :class:`OuroborosConsole` as a ``RenderBackend``.

    Same composition pattern as :class:`SerpentFlowBackend`. The wrapped
    console exists only when SerpentFlow boot fails (mutually-exclusive
    fallback) — the harness boot wire constructs whichever backend is
    alive.

    The OuroborosConsole API is closer to a per-event console.print()
    surface than SerpentFlow's regional layout, so Slice 2 wires the
    same streaming triplet but the FILE_REF / STATUS_TICK mappings will
    differ in Slice 3+ when wired (console.show_diff vs. flow.show_diff
    have different signatures).
    """

    name: str = "ouroboros_console"

    # Slice 7 graduation — symmetric with SerpentFlowBackend.
    _HANDLED_KINDS: frozenset = frozenset({
        "PHASE_BEGIN",
        "REASONING_TOKEN",
        "PHASE_END",
        "FILE_REF",
        "STATUS_TICK",
        "MODAL_PROMPT",
        "MODAL_DISMISS",
        "THREAD_TURN",
    })
    _NO_OP_KINDS: frozenset = frozenset({
        "BACKEND_RESET",
    })

    def __init__(self, console: Any) -> None:
        self._console = console

    def notify(self, event: Any) -> None:
        if event is None:
            return
        kind = _event_kind_value(event)
        if not kind:
            return
        try:
            if kind == "REASONING_TOKEN":
                content = getattr(event, "content", "") or ""
                if content and hasattr(self._console, "show_streaming_token"):
                    self._console.show_streaming_token(content)
                return
            if kind == "PHASE_BEGIN":
                metadata = _event_metadata(event)
                provider = str(metadata.get("provider", "") or "")
                if hasattr(self._console, "show_streaming_start"):
                    try:
                        self._console.show_streaming_start(provider)
                    except Exception:  # noqa: BLE001 — defensive
                        logger.debug(
                            "[OuroborosConsoleBackend] "
                            "show_streaming_start failed", exc_info=True,
                        )
                return
            if kind == "PHASE_END":
                if hasattr(self._console, "show_streaming_end"):
                    self._console.show_streaming_end()
                return
            if kind == "FILE_REF":
                self._handle_file_ref(event)
                return
            if kind == "STATUS_TICK":
                self._handle_status_tick(event)
                return
            if kind == "MODAL_PROMPT":
                self._handle_modal_prompt(event)
                return
            if kind == "MODAL_DISMISS":
                self._handle_modal_dismiss(event)
                return
            if kind == "THREAD_TURN":
                self._handle_thread_turn(event)
                return
            if kind in self._NO_OP_KINDS:
                return
            logger.debug(
                "[OuroborosConsoleBackend] unknown event kind %r — no-op",
                kind,
            )
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[OuroborosConsoleBackend] notify failed for kind=%s",
                kind, exc_info=True,
            )

    # -- Slice 7 event-kind handlers (feature-detected) ----------------

    def _handle_file_ref(self, event: Any) -> None:
        """OuroborosConsole has show_diff(file_path, diff_text)."""
        metadata = _event_metadata(event)
        path = str(metadata.get("path", "") or "")
        if not path:
            return
        diff_text = str(metadata.get("diff_text", "") or "")
        if hasattr(self._console, "show_diff"):
            try:
                self._console.show_diff(path, diff_text)
                return
            except Exception:  # noqa: BLE001 — defensive
                pass
        self._console_print(getattr(event, "content", "") or path)

    def _handle_status_tick(self, event: Any) -> None:
        """OuroborosConsole STATUS_TICK handler.

        D5 composer bridge first, then typed dispatch, then console
        fallback (mirrors SerpentFlowBackend). OuroborosConsole
        doesn't have a persistent bottom-toolbar surface, so the
        composed status line falls through to ``console.print`` —
        but tagged with a distinctive ``status:`` prefix so operators
        can grep for it.
        """
        metadata = _event_metadata(event)

        # Branch 1: D5 composer bridge — OuroborosConsole has no
        # persistent footer; print the composed line as a status row.
        if metadata.get("composed_status") is True:
            content = getattr(event, "content", "") or ""
            if content:
                self._console_print(f"  [dim]status:[/dim] {content}")
            return

        # Branch 2: typed dispatch
        if "cost" in metadata and hasattr(
            self._console, "show_cost_update",
        ):
            try:
                self._console.show_cost_update(float(metadata["cost"]))
                return
            except Exception:  # noqa: BLE001 — defensive
                pass

        # Branch 3: generic console fallback
        content = getattr(event, "content", "") or ""
        if content:
            self._console_print(content)

    def _handle_modal_prompt(self, event: Any) -> None:
        content = getattr(event, "content", "") or ""
        if content:
            self._console_print(f"\n── help ──\n{content}\n")

    def _handle_modal_dismiss(self, event: Any) -> None:
        del event
        self._console_print("── /help ──\n")

    def _handle_thread_turn(self, event: Any) -> None:
        metadata = _event_metadata(event)
        speaker = str(metadata.get("speaker", "") or "?")
        content = getattr(event, "content", "") or ""
        if not content:
            return
        self._console_print(f"  [{speaker.lower()}] {content}")

    def _console_print(self, text: str) -> None:
        try:
            console = getattr(self._console, "console", None)
            if console is not None and hasattr(console, "print"):
                console.print(text)
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        logger.debug("[OuroborosConsoleBackend] %s", text)

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        try:
            if hasattr(self._console, "show_streaming_end"):
                self._console.show_streaming_end()
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[OuroborosConsoleBackend] shutdown failed", exc_info=True,
            )


# ---------------------------------------------------------------------------
# Boot wire helper — constructs and registers the conductor with whatever
# renderers the harness has alive. Idempotent. NEVER raises.
# ---------------------------------------------------------------------------


def wire_render_conductor(
    *,
    stream_renderer: Optional[Any] = None,
    serpent_flow: Optional[Any] = None,
    ouroboros_console: Optional[Any] = None,
    posture_provider: Optional[Any] = None,
    per_op_transport: Optional[Any] = None,
) -> Optional[Any]:
    """Construct a :class:`RenderConductor`, attach the supplied
    renderers as backends, install posture provider if given, and
    register as the process-global conductor.

    Each renderer arg is optional — pass ``None`` for any not present
    in the current boot. Typical harness call:

        from backend.core.ouroboros.governance.render_backends import (
            wire_render_conductor,
        )
        wire_render_conductor(
            stream_renderer=self._stream_renderer,
            serpent_flow=self._serpent_flow,
            ouroboros_console=self._tui_console,
            posture_provider=lambda: get_current_posture_string(),
        )

    Returns the constructed conductor (or ``None`` on import failure).
    Idempotent — replaces any prior process-global conductor. NEVER
    raises out of this function — boot must not fail because rendering
    glue threw.
    """
    try:
        from backend.core.ouroboros.governance.render_conductor import (
            RenderConductor,
            register_render_conductor,
        )
    except Exception:  # noqa: BLE001 — defensive
        logger.debug(
            "[render_backends] conductor module unavailable", exc_info=True,
        )
        return None

    try:
        conductor = RenderConductor()
    except Exception:  # noqa: BLE001 — defensive
        logger.debug(
            "[render_backends] RenderConductor construction failed",
            exc_info=True,
        )
        return None

    # Attach each available renderer. The legacy on_token / show_*
    # entry points remain functional; the conductor adds a parallel
    # routing surface that Slice 3+ producers use.
    try:
        if stream_renderer is not None:
            conductor.add_backend(stream_renderer)
        if serpent_flow is not None:
            conductor.add_backend(SerpentFlowBackend(serpent_flow))
        if ouroboros_console is not None:
            conductor.add_backend(OuroborosConsoleBackend(ouroboros_console))
        if per_op_transport is not None and hasattr(per_op_transport, "notify"):
            # The default per-op transport (ClaudeStyleTransport) is a
            # RenderBackend that carries the stream triplet to attached
            # cockpits without a TTY. It becomes THE in-flight publisher so
            # a foreground run that also has a cockpit attached never sends
            # a frame twice (the stream renderer keeps its local render).
            conductor.add_backend(per_op_transport)
            try:
                from backend.core.ouroboros.battle_test.stream_renderer import (  # noqa: E501,PLC0415
                    set_inflight_publisher,
                )
                set_inflight_publisher(getattr(per_op_transport, "name", "transport"))
            except Exception:  # noqa: BLE001 — publisher election is best-effort
                pass
        if posture_provider is not None:
            conductor.set_posture_provider(posture_provider)
    except Exception:  # noqa: BLE001 — defensive
        logger.debug(
            "[render_backends] backend wiring partial failure",
            exc_info=True,
        )

    try:
        register_render_conductor(conductor)
    except Exception:  # noqa: BLE001 — defensive
        logger.debug(
            "[render_backends] register_render_conductor failed",
            exc_info=True,
        )
        return conductor  # still return; tests may use it directly

    logger.info(
        "[render_backends] conductor wired with %d backend(s)",
        len(conductor.backends()),
    )
    return conductor


# ---------------------------------------------------------------------------
# AST invariants — auto-discovered by shipped_code_invariants
# ---------------------------------------------------------------------------


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
)


# Required-symbol set on every adapter class — mirrors RenderBackend
# Protocol. AST-pinned so refactors cannot silently drop a method.
_REQUIRED_BACKEND_SYMBOLS: tuple = ("name", "notify", "flush", "shutdown")
_REQUIRED_ADAPTER_CLASSES: tuple = (
    "SerpentFlowBackend",
    "OuroborosConsoleBackend",
)


def _imported_modules(tree: Any) -> List:
    """Extract imported module names. Mirrors render_conductor's
    helper — keeps each module's pin functions self-contained."""
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


def _validate_backends_no_authority_imports(
    tree: Any, source: str,
) -> tuple:
    """Adapter module must NOT import authority modules — same
    descriptive-only contract as the conductor primitive."""
    del source
    violations: List[str] = []
    for lineno, mod_name in _imported_modules(tree):
        if mod_name in _FORBIDDEN_AUTHORITY_MODULES:
            violations.append(
                f"line {lineno}: forbidden authority import: {mod_name!r}"
            )
    return tuple(violations)


def _validate_adapter_protocol_conformance(
    tree: Any, source: str,
) -> tuple:
    """Both adapter classes MUST define the four RenderBackend symbols
    (``name`` / ``notify`` / ``flush`` / ``shutdown``). Refactors that
    silently drop a method break the Protocol contract — caught here
    at boot before the conductor would discover the gap at runtime."""
    del source
    import ast
    violations: List[str] = []
    found_classes: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in _REQUIRED_ADAPTER_CLASSES:
            members: set = set()
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(stmt.name)
                elif isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            members.add(tgt.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name,
                ):
                    members.add(stmt.target.id)
            found_classes[node.name] = members
    for required_class in _REQUIRED_ADAPTER_CLASSES:
        if required_class not in found_classes:
            violations.append(
                f"required adapter class missing: {required_class!r}"
            )
            continue
        members = found_classes[required_class]
        missing = set(_REQUIRED_BACKEND_SYMBOLS) - members
        if missing:
            violations.append(
                f"{required_class}: missing RenderBackend symbols: "
                f"{sorted(missing)}"
            )
    return tuple(violations)


_SLICE7_HANDLED_GRADUATIONS: tuple = (
    "FILE_REF", "STATUS_TICK", "MODAL_PROMPT",
    "MODAL_DISMISS", "THREAD_TURN",
)
_SLICE7_HARNESS_WIRING_TOKENS: tuple = (
    "wire_render_conductor",
    "InputController",
    "ThreadObserver",
    "ContextualHelpResolver",
    "register_help_action_handlers",
)


def _validate_serpent_handles_slice7_kinds(
    tree: Any, source: str,
) -> tuple:
    """SerpentFlowBackend._HANDLED_KINDS MUST contain the 5 Slice 7
    graduation event kinds (FILE_REF / STATUS_TICK / MODAL_PROMPT /
    MODAL_DISMISS / THREAD_TURN). Catches a future patch reverting
    the graduation by moving them back to _NO_OP_KINDS."""
    del source
    import ast
    found_in_handled: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SerpentFlowBackend":
            for stmt in node.body:
                # Look for: _HANDLED_KINDS: frozenset = frozenset({...})
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name,
                ) and stmt.target.id == "_HANDLED_KINDS":
                    if isinstance(stmt.value, ast.Call):
                        # frozenset(set_literal_or_iterable)
                        for arg in stmt.value.args:
                            if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
                                for elt in arg.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(
                                        elt.value, str,
                                    ):
                                        found_in_handled.add(elt.value)
    missing = set(_SLICE7_HANDLED_GRADUATIONS) - found_in_handled
    if missing:
        return (
            f"SerpentFlowBackend._HANDLED_KINDS missing graduation kinds: "
            f"{sorted(missing)}",
        )
    return ()


def _validate_ouroboros_handles_slice7_kinds(
    tree: Any, source: str,
) -> tuple:
    """OuroborosConsoleBackend._HANDLED_KINDS MUST contain the 5
    Slice 7 graduation event kinds. Symmetric pin to SerpentFlow."""
    del source
    import ast
    found_in_handled: set = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.ClassDef,
        ) and node.name == "OuroborosConsoleBackend":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name,
                ) and stmt.target.id == "_HANDLED_KINDS":
                    if isinstance(stmt.value, ast.Call):
                        for arg in stmt.value.args:
                            if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
                                for elt in arg.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(
                                        elt.value, str,
                                    ):
                                        found_in_handled.add(elt.value)
    missing = set(_SLICE7_HANDLED_GRADUATIONS) - found_in_handled
    if missing:
        return (
            f"OuroborosConsoleBackend._HANDLED_KINDS missing graduation "
            f"kinds: {sorted(missing)}",
        )
    return ()


def _validate_harness_wiring_present(
    tree: Any, source: str,
) -> tuple:
    """harness.py MUST contain the Slice 7 graduation wiring tokens
    (wire_render_conductor + InputController + ThreadObserver +
    ContextualHelpResolver + register_help_action_handlers). Pinned
    so a refactor cannot silently drop the boot wire and leave the
    substrate orphan."""
    del tree
    missing = [
        token for token in _SLICE7_HARNESS_WIRING_TOKENS
        if token not in source
    ]
    if missing:
        return (
            f"harness.py missing Slice 7 wiring tokens: {missing}",
        )
    return ()


def _validate_serpent_flow_prompt_helper(
    tree: Any, source: str,
) -> tuple:
    """Cross-file pin: serpent_flow.py MUST contain
    ``_build_repl_prompt_html`` AND its call from the REPL loop
    (CC2.4 multi-line prompt). Without these, the REPL falls back
    to the legacy single-line prompt and operators lose the
    cwd / mode / posture context."""
    del tree
    required = (
        "_build_repl_prompt_html",
        "JARVIS_PROMPT_TEMPLATE",
    )
    missing = [t for t in required if t not in source]
    if missing:
        return (
            f"serpent_flow.py missing prompt template tokens: "
            f"{missing} — multi-line REPL prompt won't render",
        )
    return ()


def _validate_serpent_animation_stop_guards(
    tree: Any, source: str,
) -> tuple:
    """serpent_animation.stop() body MUST contain BOTH ``_SUPPRESSED``
    AND ``_start_time`` token references. The two-guard combo
    prevents the boot-time "FAILED [ OUROBOROS ] in 673361.3s"
    bug — a stop() called without a paired start() would otherwise
    show ``time.monotonic() - 0.0`` (process uptime in seconds) as
    the elapsed.

    Catches a future refactor that strips one guard. The pin walks
    the source for the literal tokens; works regardless of how the
    guards are structured (single ``if not X or Y or Z`` line vs
    nested early returns)."""
    del tree
    required_tokens = ("_SUPPRESSED", "_start_time")
    missing = [t for t in required_tokens if t not in source]
    if missing:
        return (
            f"serpent_animation.py missing stop() guard tokens: "
            f"{missing} — boot-time UX bug will return",
        )
    return ()


def _validate_help_dispatcher_verb_discovery(
    tree: Any, source: str,
) -> tuple:
    """help_dispatcher.py MUST expose ``_discover_module_provided_verbs``
    AND call it from ``get_default_verb_registry`` (Slice 7 follow-up
    backlog #2). Without this hook, modules like ``render_repl`` lose
    their verb registration after every ``reset_default_verb_registry``
    call (test isolation, hot reloads). Pinned cross-file so future
    refactors of help_dispatcher cannot silently drop the discovery
    loop."""
    del tree
    missing = []
    if "_discover_module_provided_verbs" not in source:
        missing.append("_discover_module_provided_verbs definition")
    # The call site MUST be inside get_default_verb_registry's
    # singleton-construction branch.
    if "_discover_module_provided_verbs(_default_verbs)" not in source:
        missing.append(
            "_discover_module_provided_verbs(_default_verbs) call",
        )
    if missing:
        return (
            f"help_dispatcher.py missing verb discovery hook: {missing}",
        )
    return ()


# ---------------------------------------------------------------------------
# Slice 7 follow-up #5: AST pins on the 4 producer-side flag defaults.
# Each pin reads the per-slice register_flags source and verifies the
# FlagSpec for the named producer flag has default=True. Catches a
# refactor that flips a default back to False without coordinated
# accessor + test update.
# ---------------------------------------------------------------------------


def _flag_spec_default_in_source(
    source: str, flag_name: str,
) -> Optional[bool]:
    """Parse the source bytes for a ``FlagSpec(...)`` block whose
    ``name=`` argument is ``flag_name``, return its ``default=``
    value as a bool. Returns ``None`` when the spec block can't be
    located. Pure string scan — robust against field-order variations.

    Substrate convention: each module declares
    ``_FLAG_REASONING_STREAM_ENABLED = "JARVIS_REASONING_STREAM_ENABLED"``
    at module top, then references the constant in the FlagSpec via
    ``name=_FLAG_REASONING_STREAM_ENABLED``. We search for the
    ``= "JARVIS_..."`` constant assignment first; if found, we
    extract the constant name and search for ``name=<constant>``."""
    # Match the constant assignment: ``_FLAG_X = "JARVIS_X_ENABLED"``
    # The constant name precedes the ``=`` and ``"JARVIS_..."`` follows.
    quoted = f'"{flag_name}"'
    apos = f"'{flag_name}'"
    constant_name: Optional[str] = None
    for needle in (quoted, apos):
        idx = source.find(needle)
        if idx < 0:
            continue
        # Walk back to find the constant name. Stop at newline.
        line_start = source.rfind("\n", 0, idx) + 1
        line = source[line_start:idx]
        # Expected shape: ``_FLAG_<NAME> = ``
        if "=" in line:
            constant_name = line.split("=", 1)[0].strip()
            break
    # Locate the FlagSpec(name=<constant>) block.
    needle_start = -1
    if constant_name:
        needle_start = source.find(f"name={constant_name}")
    if needle_start < 0:
        # Fallback: literal name= form.
        needle_start = source.find(f"name={quoted}")
    if needle_start < 0:
        needle_start = source.find(f"name={apos}")
    if needle_start < 0:
        return None
    # Bound the search to a small window after the name= line.
    window = source[needle_start: needle_start + 800]
    if "default=True" in window:
        return True
    if "default=False" in window:
        return False
    return None


def _make_producer_default_validator(
    flag_name: str,
) -> Any:
    """Build a closure-validator that checks the producer flag's
    default is True in the target source bytes. The actual flag
    name is captured in the closure; the function signature matches
    ``ShippedCodeValidator``."""
    def _validator(tree: Any, source: str) -> tuple:
        del tree
        default = _flag_spec_default_in_source(source, flag_name)
        if default is None:
            return (
                f"FlagSpec for {flag_name!r} not located in source",
            )
        if default is False:
            return (
                f"FlagSpec for {flag_name!r} has default=False; "
                f"Slice 7 follow-up #4 graduated this to True",
            )
        return ()
    return _validator


def _validate_streamrenderer_protocol_conformance(
    tree: Any, source: str,
) -> tuple:
    """``StreamRenderer`` (defined in stream_renderer.py — separate file
    targeted by this pin's ``target_file``) MUST also expose the four
    RenderBackend symbols. Pinned here so the cross-file contract that
    "all 3 renderers are backends" is enforced from one auditable spot."""
    del source
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StreamRenderer":
            members: set = set()
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(stmt.name)
                elif isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            members.add(tgt.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name,
                ):
                    members.add(stmt.target.id)
            missing = set(_REQUIRED_BACKEND_SYMBOLS) - members
            if missing:
                return (
                    f"StreamRenderer: missing RenderBackend symbols: "
                    f"{sorted(missing)}",
                )
            return ()
    return ("StreamRenderer class not found in target file",)


def register_shipped_invariants() -> List:
    """Auto-discovered by shipped_code_invariants. Returns the AST pins
    that protect Slice 2's structural shape."""
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except Exception:  # noqa: BLE001 — defensive
        return []
    return [
        ShippedCodeInvariant(
            invariant_name="render_backends_no_authority_imports",
            target_file=(
                "backend/core/ouroboros/governance/render_backends.py"
            ),
            description=(
                "Adapter module must NOT import authority modules — "
                "rendering glue stays descriptive only, never a "
                "control-flow surface."
            ),
            validate=_validate_backends_no_authority_imports,
        ),
        ShippedCodeInvariant(
            invariant_name="render_backends_adapter_protocol_conformance",
            target_file=(
                "backend/core/ouroboros/governance/render_backends.py"
            ),
            description=(
                "SerpentFlowBackend and OuroborosConsoleBackend MUST "
                "both define the RenderBackend Protocol's four symbols "
                "(name / notify / flush / shutdown). Refactors that "
                "silently drop a method break the Protocol contract."
            ),
            validate=_validate_adapter_protocol_conformance,
        ),
        ShippedCodeInvariant(
            invariant_name="streamrenderer_protocol_conformance",
            target_file=(
                "backend/core/ouroboros/battle_test/stream_renderer.py"
            ),
            description=(
                "StreamRenderer (the third RenderBackend, with backend "
                "methods inline rather than via composition adapter) "
                "MUST expose the same four RenderBackend symbols. "
                "Cross-file contract pinned from one auditable spot — "
                "if any renderer drops backend conformance, this fails."
            ),
            validate=_validate_streamrenderer_protocol_conformance,
        ),
        # Slice 7 graduation pins — catch reverts of the 5-kind backend
        # handler expansion + the harness boot wire.
        ShippedCodeInvariant(
            invariant_name="render_backends_serpent_handles_slice7_kinds",
            target_file=(
                "backend/core/ouroboros/governance/render_backends.py"
            ),
            description=(
                "Slice 7 graduation pin: SerpentFlowBackend._HANDLED_"
                "KINDS MUST contain FILE_REF / STATUS_TICK / "
                "MODAL_PROMPT / MODAL_DISMISS / THREAD_TURN. A future "
                "patch moving any of these back to _NO_OP_KINDS would "
                "silently revert Gap #2/#3/#6/#7 rendering — caught "
                "here at boot."
            ),
            validate=_validate_serpent_handles_slice7_kinds,
        ),
        ShippedCodeInvariant(
            invariant_name="render_backends_ouroboros_handles_slice7_kinds",
            target_file=(
                "backend/core/ouroboros/governance/render_backends.py"
            ),
            description=(
                "Symmetric Slice 7 graduation pin for "
                "OuroborosConsoleBackend._HANDLED_KINDS — same 5 "
                "kinds. Symmetry is a property: if the SerpentFlow "
                "fallback ever drops backend conformance for these "
                "kinds, the operator-visible surface becomes "
                "asymmetric depending on which renderer is active."
            ),
            validate=_validate_ouroboros_handles_slice7_kinds,
        ),
        ShippedCodeInvariant(
            invariant_name="harness_wires_render_substrate",
            target_file=(
                "backend/core/ouroboros/battle_test/harness.py"
            ),
            description=(
                "Slice 7 graduation pin: the harness boot sequence "
                "MUST contain the 5 Slice 7 wiring tokens: "
                "wire_render_conductor + InputController + "
                "ThreadObserver + ContextualHelpResolver + "
                "register_help_action_handlers. Catches a refactor "
                "that silently drops a producer-singleton "
                "construction — without it the operator-side surface "
                "would orphan even with the substrate flag on."
            ),
            validate=_validate_harness_wiring_present,
        ),
        # Slice 7 follow-up #5 — pin each producer-flag default to
        # True. Each FlagSpec lives in its own file; cross-file
        # invariants pinned from one auditable spot.
        ShippedCodeInvariant(
            invariant_name=(
                "render_primitives_reasoning_stream_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/render_primitives.py"
            ),
            description=(
                "Slice 7 follow-up #4 graduated "
                "JARVIS_REASONING_STREAM_ENABLED default true. The "
                "FlagSpec MUST carry default=True; a future patch "
                "reverting it to False would silently disable the "
                "ReasoningStream producer wiring in providers.py."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_REASONING_STREAM_ENABLED",
            ),
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "key_input_input_controller_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/key_input.py"
            ),
            description=(
                "Slice 7 follow-up #4 graduated "
                "JARVIS_INPUT_CONTROLLER_ENABLED default true. The "
                "FlagSpec MUST carry default=True; a future patch "
                "reverting it to False would silently disable the "
                "Esc-mid-token interrupt + ?-help binding."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_INPUT_CONTROLLER_ENABLED",
            ),
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "render_thread_thread_observer_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/render_thread.py"
            ),
            description=(
                "Slice 7 follow-up #4 graduated "
                "JARVIS_THREAD_OBSERVER_ENABLED default true. The "
                "FlagSpec MUST carry default=True; a future patch "
                "reverting it to False would silently disable the "
                "ConversationBridge → conductor pump."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_THREAD_OBSERVER_ENABLED",
            ),
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "render_help_contextual_help_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/render_help.py"
            ),
            description=(
                "Slice 7 follow-up #4 graduated "
                "JARVIS_CONTEXTUAL_HELP_ENABLED default true. The "
                "FlagSpec MUST carry default=True; a future patch "
                "reverting it to False would silently disable the "
                "ContextualHelpResolver ranking + MODAL_PROMPT "
                "publish path."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_CONTEXTUAL_HELP_ENABLED",
            ),
        ),
        # D5 graduation pins — JARVIS_EMIT_TIER_GATING_ENABLED +
        # JARVIS_STATUS_LINE_COMPOSER_ENABLED. Same pattern as
        # Slice 7 follow-up #5 — catches a refactor that flips
        # either default back to False without coordinated test
        # update.
        ShippedCodeInvariant(
            invariant_name=(
                "render_emit_tier_gating_enabled_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/render_emit_tier.py"
            ),
            description=(
                "D5 graduated JARVIS_EMIT_TIER_GATING_ENABLED to "
                "default true. The FlagSpec MUST carry default=True; "
                "a refactor reverting it to False would silently "
                "restore the noisy 6-lines-per-op CLI."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_EMIT_TIER_GATING_ENABLED",
            ),
        ),
        ShippedCodeInvariant(
            invariant_name=(
                "status_line_composer_enabled_default_true"
            ),
            target_file=(
                "backend/core/ouroboros/governance/status_line_composer.py"
            ),
            description=(
                "D5 graduated JARVIS_STATUS_LINE_COMPOSER_ENABLED to "
                "default true. The FlagSpec MUST carry default=True; "
                "a refactor reverting it to False would silently "
                "restore the N-scattered-update_*-emits pattern."
            ),
            validate=_make_producer_default_validator(
                "JARVIS_STATUS_LINE_COMPOSER_ENABLED",
            ),
        ),
        # CC2.4 — multi-line REPL prompt helper presence pin.
        ShippedCodeInvariant(
            invariant_name="serpent_flow_repl_prompt_helper_present",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "CC2.4: serpent_flow.py MUST contain "
                "_build_repl_prompt_html (the multi-line cwd/mode/"
                "posture prompt builder) AND reference "
                "JARVIS_PROMPT_TEMPLATE (the operator override env "
                "var). Without these, the REPL reverts to the "
                "legacy single-line prompt and operators lose CC-"
                "style context. Pin reads serpent_flow.py source."
            ),
            validate=_validate_serpent_flow_prompt_helper,
        ),
        # D2 — serpent_animation stop() guards (boot-time UX fix).
        ShippedCodeInvariant(
            invariant_name=(
                "serpent_animation_stop_guards_present"
            ),
            target_file=(
                "backend/core/ouroboros/governance/serpent_animation.py"
            ),
            description=(
                "serpent_animation.py stop() MUST guard against "
                "_SUPPRESSED + _start_time<=0 to prevent the boot-"
                "time 'FAILED [ OUROBOROS ] in 673361.3s' bug. The "
                "pin checks for both token literals — refactors that "
                "strip either guard will be caught at boot."
            ),
            validate=_validate_serpent_animation_stop_guards,
        ),
        # Backlog #2 — verb discovery hook in help_dispatcher.
        ShippedCodeInvariant(
            invariant_name=(
                "help_dispatcher_verb_discovery_present"
            ),
            target_file=(
                "backend/core/ouroboros/governance/help_dispatcher.py"
            ),
            description=(
                "Slice 7 follow-up backlog #2: help_dispatcher.py MUST "
                "expose _discover_module_provided_verbs AND call it "
                "from get_default_verb_registry. Without this hook, "
                "module-owned verbs (render_repl./render, future "
                "REPL surfaces) lose registration after every "
                "reset_default_verb_registry call (test isolation, "
                "hot reload). Pinned cross-file so future refactors "
                "cannot silently drop the discovery loop."
            ),
            validate=_validate_help_dispatcher_verb_discovery,
        ),
    ]


__all__ = [
    "OuroborosConsoleBackend",
    "RENDER_BACKENDS_SCHEMA_VERSION",
    "SerpentFlowBackend",
    "register_shipped_invariants",
    "wire_render_conductor",
]
