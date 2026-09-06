"""Operator-visible token streaming renderer for Ouroboros GENERATE.

Closes the "spinner for 2 minutes while Claude generates" UX gap: tokens
arrive on the operator terminal in real-time via a Rich ``Live`` + ``Markdown``
widget, syntax-highlighted as they stream.

Architectural mandates (§ each enforced):

1. **Async isolation** — the token callback is O(1) non-blocking (enqueues
   via ``put_nowait``, drops on overflow rather than blocking the provider's
   stream). A dedicated consumer task batches at ~16ms cadence (60fps) and
   drives ``Live.update``, so terminal rendering lag cannot starve the
   inference I/O stream.
2. **Syntax-aware rendering** — the buffer is rendered via Rich's
   ``Markdown`` widget, which handles partial fenced code blocks gracefully
   (unclosed ```` ```python ```` renders as syntax-highlighted code-in-
   progress and seals cleanly when the closing fence arrives).
3. **Kill switch** — ``JARVIS_UI_STREAMING_ENABLED=1`` (default on). Flip to
   ``0`` for overnight batches where terminal UI overhead isn't wanted.
   When off, ``start()`` is a no-op and ``on_token()`` silently discards.
4. **Observability anchor** — on ``end()``, emits a single INFO line:
   ``[StreamRender] op=X provider=Y tokens=N dropped=D first_token_ms=T
   total_ms=M tps=P``. TTFT + TPS turn the UI widget into a provider-
   health telemetry sensor.

Authority invariant: this module mutates only the terminal presentation
layer. It never reads or writes ``ctx``, never touches Iron Gate,
UrgencyRouter, risk tier, policy engine, FORBIDDEN_PATH, ToolExecutor
protected-path checks, or approval gating.

Module-level singleton (``register_stream_renderer`` / ``get_stream_renderer``
/ ``reset_stream_renderer``) matches the pattern already used by
``OpsDigestObserver`` and ``LastSessionSummary``: the harness registers on
boot, providers look up at stream time, tests reset between cases.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, List, Optional

logger = logging.getLogger("Ouroboros.StreamRenderer")

_STREAMING_ENV_VAR = "JARVIS_UI_STREAMING_ENABLED"
_FLOW_MODE_ENV_VAR = "JARVIS_UI_STREAMING_FLOW_MODE_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def streaming_enabled() -> bool:
    """Env gate read. Default: ON (``1``). Flip to ``0`` for batch mode."""
    return os.environ.get(_STREAMING_ENV_VAR, "1").strip().lower() in _TRUTHY


def flow_mode_enabled() -> bool:
    """§41.3 #27 — progressive-streaming flow mode (unwrap the Live cage).

    When ON, completed markdown blocks are committed ABOVE the Live region
    into terminal scrollback as they stream (CC-style progressive flow);
    the Live cage holds only the current in-progress block. When OFF
    (default, §33.1), the legacy behavior is byte-identical: the whole
    stream lives inside the tail-truncated Live region and long
    generations scroll out of the cage without accumulating in scrollback.
    """
    return os.environ.get(_FLOW_MODE_ENV_VAR, "0").strip().lower() in _TRUTHY


#: Longest single line the deck will carry from a generation. A model can
#: emit a 40k-character line; the deck is a ring of terminal rows.
_MIRROR_MAX_LINE_CHARS = 2000

#: Longest in-flight tail carried per frame. Past this the model
#: has stopped writing a sentence and started writing a document.
_INFLIGHT_MAX_CHARS = 800


#: Rows an in-flight sentence may occupy before it is elided. It is one
#: thought, not a document; past this the strip would push the deck off
#: screen to show text that is about to become deck content.
INFLIGHT_MAX_ROWS = 4


def render_inflight(
    text: str, *, width: Optional[int] = None,
    max_rows: int = INFLIGHT_MAX_ROWS,
    preserve_lines: bool = False,
    keep_first: bool = False,
) -> List[str]:
    """Wrap an in-flight sentence into strip rows. Pure. NEVER raises.

    Lives HERE, beside the producer, and is called by every surface that
    draws in-flight text — the attach cockpit and the demo. The daemon
    cannot do this wrap itself (it may serve two cockpits of different
    widths, and only each client knows its own), so the wrap is necessarily
    client-side; what must NOT be client-side is a second opinion about what
    the shape is. That is the same split the roster and the status line use:
    state crosses the bridge, one renderer draws it.

    Indented as a continuation, because that is what it is — the line the
    deck is about to receive. A bottom-anchored deck puts its newest entry
    directly above this strip, so the sentence appears exactly where it will
    land, which is what makes it read as inline rather than as a widget.
    """
    try:
        import textwrap
        cols = int(width) if width and int(width) > 0 else 80
        room = max(20, cols - 4)
        rows = max(1, int(max_rows))
        if preserve_lines:
            # A COMMAND's output. Line breaks are content here — collapsing
            # them would run a pytest summary and its next test name into
            # one sentence. Each source line wraps independently so the
            # structure the command wrote survives the terminal's width.
            src = [ln for ln in str(text or "").splitlines()]
            if not any(ln.strip() for ln in src):
                return []
            wrapped = []
            for ln in src:
                wrapped.extend(textwrap.wrap(ln, width=room) or [""])
        else:
            # PROSE. The model streams a sentence whose newlines are an
            # artefact of token boundaries, not meaning.
            flat = " ".join(str(text or "").split())
            if not flat:
                return []
            wrapped = textwrap.wrap(flat, width=room) or [flat[:room]]
        if len(wrapped) > rows:
            # Keep the NEWEST rows: the tail is where the writing is
            # happening, and an elided head reads as "there is more above",
            # which is true and about to be in the transcript anyway.
            #
            # `keep_first` exempts row 0. A running command's first row is
            # its HEADER — `$ bash · 11s` — which is context rather than
            # content, and eliding it left a tail of test names with no
            # indication of what was running or for how long. Caught by
            # driving this from `ov demo live`, which is what the demo is
            # for. The caller declares which shape it has, the same way it
            # declares whether newlines are meaningful.
            if keep_first and rows > 2:
                wrapped = [wrapped[0], "…"] + wrapped[-(rows - 2):]
            elif rows > 1:
                wrapped = ["…"] + wrapped[-(rows - 1):]
            else:
                wrapped = ["…"]
        return [f"  {line}" for line in wrapped]
    except Exception:  # noqa: BLE001
        logger.debug("[StreamRender] inflight wrap degraded", exc_info=True)
        return []


def mirror_stream_enabled() -> bool:
    """Default ON. Off, the generation is invisible on an attached cockpit
    again — which is the state this exists to end. NEVER raises."""
    return os.environ.get(
        "JARVIS_STREAM_MIRROR_ENABLED", "1",
    ).strip().lower() not in ("0", "false", "no", "off")


#: Which backend carries the in-flight tail to attached cockpits. ``None``
#: = this renderer (the foreground TTY case, unchanged). The harness names
#: another backend when one is registered that publishes the same frame —
#: the default per-op transport — so a foreground run with a cockpit
#: attached does not send every frame twice.
_INFLIGHT_PUBLISHER: Optional[str] = None
_STREAM_RENDERER_PUBLISHER: str = "stream_renderer"


def set_inflight_publisher(name: Optional[str]) -> None:
    """Name the ONE backend that publishes ``stream_inflight`` frames."""
    global _INFLIGHT_PUBLISHER  # noqa: PLW0603
    _INFLIGHT_PUBLISHER = str(name) if name else None


def owns_inflight_publishing(name: str) -> bool:
    """Whether ``name`` is the backend that carries the tail right now."""
    return _INFLIGHT_PUBLISHER is None or _INFLIGHT_PUBLISHER == name


def publish_inflight_tail(op_id: str, tail: str, *, done: bool = False) -> bool:
    """The ONE producer of the ``stream_inflight`` frame.

    Carried on the telemetry lane rather than the markup lane because it is
    STATE, not a transcript entry: the last frame wins, and a dropped one
    costs a frame of smoothness rather than a lost line. The daemon's own
    in-flight registry is fed FIRST — `publish_telemetry_global` is a no-op
    with nobody attached, which is exactly when the daemon's own cockpit is
    the surface being looked at. Returns whether a cockpit received it.

    Extracted from the renderer (2026-09-06) because the renderer is
    TTY-gated: a headless daemon never constructs it, so the organism's
    generation was invisible on every attached cockpit — the exact state
    this frame exists to end. Any backend may now carry the tail.
    NEVER raises.
    """
    frame = {
        "kind": "stream_inflight",
        "op_id": str(op_id or ""),
        "text": "" if done else str(tail or "")[-_INFLIGHT_MAX_CHARS:],
        "done": bool(done),
    }
    try:
        from backend.core.ouroboros.battle_test.inflight_registry import (  # noqa: E501,PLC0415
            note_inflight_frame,
        )
        note_inflight_frame(frame)
    except Exception:  # noqa: BLE001
        pass
    try:
        from backend.core.ouroboros.battle_test.cockpit_attach import (  # noqa: PLC0415
            publish_telemetry_global,
        )
        return bool(publish_telemetry_global(frame))
    except Exception:  # noqa: BLE001
        logger.debug("[StreamRender] inflight publish degraded", exc_info=True)
        return False


def local_echo_enabled() -> bool:
    """Default ON. Whether a foreground run echoes the stream to its OWN
    terminal when no cockpit is attached over the bridge.

    Off restores the previous behaviour exactly: the generation is visible
    only to a remote `ov attach` client, and invisible in the in-process
    foreground run that is how `ov` actually boots the harness. That was the
    state this exists to end. NEVER raises.
    """
    return os.environ.get(
        "JARVIS_STREAM_LOCAL_ECHO_ENABLED", "1",
    ).strip().lower() not in ("0", "false", "no", "off")


def find_commit_boundary(text: str, start: int = 0) -> int:
    """Return the index up to which ``text`` can be safely committed to
    scrollback: just past the LAST complete blank line that sits OUTSIDE
    a fenced code block, at or after ``start``. Returns ``start`` when no
    safe boundary exists yet.

    Contract: ``start`` must be a previous return value of this function
    (or 0) — boundaries are only ever emitted at closed-fence points, so
    scanning ``text[start:]`` with fence-closed initial state is always
    correct and the scan cost is O(new content), not O(stream length).

    Pure, deterministic, NEVER raises. A trailing line without ``\\n`` is
    still in flight and never a boundary; blank lines inside an open
    ```` ``` ````/``~~~`` fence never split the block (a partial fenced
    code block must stay whole in the Live region until it seals).
    """
    try:
        fence_open = False
        boundary = start
        pos = start
        for line in text[start:].splitlines(keepends=True):
            end = pos + len(line)
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_open = not fence_open
            elif not fence_open and stripped == "" and line.endswith("\n"):
                boundary = end
            pos = end
        return boundary
    except Exception:  # noqa: BLE001 — presentation layer never raises
        return start


# ---------------------------------------------------------------------------
# Tunables — env-overridable without code churn
# ---------------------------------------------------------------------------

# Maximum tokens held in the producer→consumer queue. Overflow drops
# incoming tokens (dropped_count tracked) rather than blocking the provider.
# 256 sized for ~5s of buffering at 50 tok/s — plenty of headroom under
# normal render load, and the dropped_count is surfaced in the INFO line
# so tuning is empirical.
_QUEUE_MAX = int(os.environ.get("JARVIS_UI_STREAMING_QUEUE_MAX", "256"))

# Batch interval — target render cadence. 16ms ≈ 60fps.
_BATCH_INTERVAL_S = float(os.environ.get("JARVIS_UI_STREAMING_BATCH_MS", "16")) / 1000.0

# Rich Live refresh_per_second — internal widget refresh rate. Decoupled
# from our batch cadence; Rich handles its own render throttling on a
# background thread so .update() is essentially pointer-swap cheap.
_LIVE_REFRESH_HZ = int(os.environ.get("JARVIS_UI_STREAMING_LIVE_REFRESH_HZ", "30"))

# Sliding-window cap on the markdown re-parse buffer (Manifesto §3).
# Rich.Markdown re-parses the full string on every .update(); at the 16ms
# batch cadence over a 16k-token stream that's O(N²) work where N is
# accumulated chars. Slicing to the tail keeps the per-render cost O(1)
# in the stream length — Rich Live only displays the visible viewport
# anyway, so nothing above the slice would have rendered to the terminal.
_RENDER_TAIL_CHARS = int(os.environ.get("JARVIS_UI_STREAMING_RENDER_TAIL_CHARS", "4096"))


# ---------------------------------------------------------------------------
# StreamRenderer — per-session operator-visible token stream
# ---------------------------------------------------------------------------


class StreamRenderer:
    """Async-isolated token renderer for GENERATE phase.

    Lifecycle: ``start(op_id, provider)`` → many ``on_token(text)`` calls
    → ``end()``. Each lifecycle yields exactly one INFO line on ``end()``
    with TTFT + TPS metrics.

    Thread/coroutine model:
      - ``on_token`` runs on the provider's stream coroutine. Non-blocking
        enqueue via ``put_nowait``; on ``QueueFull`` it drops and
        increments ``dropped_count``. Never awaits.
      - ``start`` spawns a dedicated consumer task on the currently
        running loop; the consumer batches at ``_BATCH_INTERVAL_S`` and
        calls ``Live.update``. Rich's Live does the actual terminal
        render on its own thread, so ``update`` is cheap.
      - ``end`` cancels the consumer, flushes the final buffer, stops
        Live, and emits the observability INFO line.

    When ``streaming_enabled()`` is False at ``start`` time, the
    renderer becomes a no-op: no Live, no consumer, no INFO line (DEBUG
    line only). ``on_token`` calls fall through silently.

    RenderBackend conformance (Slice 2 of the RenderConductor arc): the
    ``name`` / ``notify`` / ``flush`` / ``shutdown`` methods below let
    this renderer plug into ``RenderConductor`` as a backend. Conductor
    events route to the same internal queue as the legacy ``on_token``
    entry point — no logic duplication. The legacy API stays functional
    for back-compat; both paths converge on the queue.
    """

    # RenderBackend Protocol — Slice 2 of the RenderConductor arc.
    name: str = "stream_renderer"

    def __init__(self, console: Optional[Any] = None) -> None:
        self._console = console
        self._queue: Optional[asyncio.Queue] = None
        self._buffer: str = ""
        self._live: Optional[Any] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._active: bool = False
        self._op_id: str = ""
        self._provider: str = ""
        self._start_mono: float = 0.0
        self._first_token_mono: Optional[float] = None
        self._token_count: int = 0
        self._dropped_count: int = 0
        # §41.3 #27 flow mode — offset of the buffer prefix already
        # committed to scrollback above the Live region. Snapshotted from
        # the env gate at start() so one op never splits across modes.
        self._flow_mode: bool = False
        self._committed_offset: int = 0
        #: How much of the buffer has been MIRRORED to attached cockpits.
        #:
        #: Separate from `_committed_offset`, which belongs to the local Live
        #: widget's scrollback commits. They advance on different triggers and
        #: for different audiences; sharing one cursor would make a local
        #: terminal's rendering decisions silently truncate a remote deck.
        self._mirrored_offset: int = 0
        #: Has this stream announced itself to the deck yet?
        self._mirror_opened: bool = False
        #: Last tail published, so an idle frame costs nothing.
        self._last_inflight: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, op_id: str, provider: str = "") -> None:
        """Begin a streaming session. Idempotent: safe to call mid-stream
        (ends any prior session cleanly first). No-op when the env gate
        is off.

        Safe to call from any coroutine on the asyncio loop; if no loop
        is running (e.g. unit test without loop), falls back to no-op
        gracefully — this preserves the "renderer optional" contract.
        """
        # Idempotency: if already active, close prior session first.
        if self._active:
            self.end()

        if not streaming_enabled():
            logger.debug(
                "[StreamRender] op=%s streaming disabled via %s — no-op",
                op_id, _STREAMING_ENV_VAR,
            )
            return

        # Reset per-session state.
        self._op_id = op_id
        self._provider = provider or ""
        self._buffer = ""
        self._token_count = 0
        self._dropped_count = 0
        self._first_token_mono = None
        self._start_mono = time.monotonic()
        self._flow_mode = flow_mode_enabled()
        self._committed_offset = 0

        # Obtain the running loop. The queue MUST be bound to the same
        # loop that will run the consumer, else put_nowait raises.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — renderer degrades to no-op. This is the
            # headless / non-async caller path; streaming wouldn't work
            # anyway since the provider is async.
            logger.debug(
                "[StreamRender] op=%s no running loop — renderer is no-op",
                op_id,
            )
            return

        self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)

        # Enforce the TTY contract (Manifesto §3). Headless / sandbox / CI
        # runs must bypass the Rich Markdown re-parse path entirely — a
        # cosmetic UI renderer cannot be permitted to block the async
        # event loop running the Claude stream. The consumer task still
        # drains the queue so token_count and the final INFO line stay
        # accurate; only the visible Live widget is skipped.
        #
        # REPL coordination (2026-05-03): the same log-only branch is
        # also taken when a SerpentREPL is active. Rich.Live writes via
        # direct cursor manipulation that bypasses ``patch_stdout`` and
        # clobbers the input prompt under concurrent output. Operators
        # retain the [StreamRender] INFO line at end-of-stream (token
        # count + duration + drops) so observability is preserved;
        # only the per-token visible widget goes away.
        try:
            from backend.core.ouroboros.battle_test.serpent_flow import (
                is_repl_active,
            )
            _repl_active = is_repl_active()
        except Exception:
            _repl_active = False
        if not sys.stdout.isatty() or _repl_active:
            _why = "non-TTY stdout" if not sys.stdout.isatty() else "REPL active"
            logger.debug(
                "[StreamRender] op=%s %s — Live skipped, log-only stream",
                op_id, _why,
            )
            self._live = None
        else:
            # Try to open a Rich Live widget. On any failure (no console, no
            # Rich, terminal misbehaves), degrade to log-only streaming — the
            # consumer still drains the queue and emits the INFO line at end.
            try:
                from rich.live import Live
                from rich.markdown import Markdown

                self._live = Live(
                    Markdown(""),
                    console=self._console,
                    transient=False,
                    refresh_per_second=_LIVE_REFRESH_HZ,
                )
                self._live.start()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[StreamRender] op=%s Rich.Live unavailable; log-only stream",
                    op_id, exc_info=True,
                )
                self._live = None

        self._consumer_task = loop.create_task(self._consume())
        self._active = True

    def on_token(self, text: str) -> None:
        """Non-blocking token ingress.

        Hot path: called once per token from the provider's stream
        coroutine. Must complete in O(1) — no awaits, no synchronous
        render, no I/O. Drops on queue overflow (rare) rather than
        blocking the producer. Overflow count surfaces in the INFO line.
        """
        if not self._active or not text:
            return
        if self._first_token_mono is None:
            self._first_token_mono = time.monotonic()
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(text)
        except asyncio.QueueFull:
            self._dropped_count += 1

    def end(self) -> None:
        """Finalize the stream: cancel the consumer, flush, stop Live,
        emit observability INFO line. Idempotent."""
        if not self._active:
            return
        self._active = False

        # Cancel consumer and wait for it to flush remaining batch.
        task = self._consumer_task
        self._consumer_task = None
        if task is not None and not task.done():
            task.cancel()
            # Best-effort: schedule a small drain. We're not awaiting
            # here (end is sync) — the task's CancelledError handler
            # does a final buffer flush before it exits.

        # Stop the Live widget after giving the consumer one last
        # synchronous drain opportunity.
        self._drain_remaining_sync()

        # FINAL mirror flush — the tail after the last newline would
        # otherwise never reach an attached cockpit, so every generation
        # would lose its closing sentence.
        self._mirror_completed_lines(final=True)
        # Clear the strip: everything is in the transcript now.
        self._publish_inflight_tail(done=True)

        if self._live is not None:
            try:
                from rich.markdown import Markdown
                if self._flow_mode:
                    # §41.3 #27 — final commit: land the uncommitted
                    # remainder in scrollback and clear the cage, so the
                    # ENTIRE stream persists in terminal history with no
                    # tail truncation.
                    remainder = self._buffer[self._committed_offset:]
                    if remainder.strip():
                        self._live.console.print(Markdown(remainder))  # type: ignore[attr-defined]
                    self._committed_offset = len(self._buffer)
                    self._live.update(Markdown(""))  # type: ignore[attr-defined]
                else:
                    # Final render: tail slice only. Rich Live's viewport
                    # shows at most the visible terminal area, so re-parsing
                    # the full buffer would pay O(N) cost to render content
                    # that never reaches pixels.
                    self._live.update(Markdown(self._buffer[-_RENDER_TAIL_CHARS:]))  # type: ignore[attr-defined]
                self._live.stop()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[StreamRender] op=%s Live.stop failed", self._op_id,
                    exc_info=True,
                )
            self._live = None

        # Observability INFO — single line, grep-able, metrics-first.
        total_s = time.monotonic() - self._start_mono
        ttft_ms = (
            int((self._first_token_mono - self._start_mono) * 1000)
            if self._first_token_mono is not None
            else -1
        )
        tps = (self._token_count / total_s) if total_s > 0.0 else 0.0
        logger.info(
            "[StreamRender] op=%s provider=%s tokens=%d dropped=%d "
            "first_token_ms=%d total_ms=%d tps=%.1f",
            self._op_id, self._provider, self._token_count,
            self._dropped_count, ttft_ms, int(total_s * 1000), tps,
        )

        # Clear state so the renderer instance is reusable across ops.
        self._queue = None
        self._buffer = ""
        self._committed_offset = 0
        self._mirrored_offset = 0
        self._mirror_opened = False
        self._last_inflight = ""
        self._flow_mode = False
        self._op_id = ""
        self._provider = ""
        self._first_token_mono = None
        self._token_count = 0
        self._dropped_count = 0

    # ------------------------------------------------------------------
    # RenderBackend Protocol — Slice 2 of RenderConductor arc.
    # Routes RenderEvents to the same internal pipeline as legacy
    # ``on_token`` / ``start`` / ``end``. Both legacy callers and
    # conductor-routed events converge on the queue — no duplication.
    # ------------------------------------------------------------------

    def notify(self, event: Any) -> None:
        """Consume a RenderEvent from the conductor.

        Maps event.kind → existing internal method:
          * REASONING_TOKEN  → on_token(content)
          * PHASE_BEGIN      → start(op_id, provider) (provider read from
                                event.metadata.provider, fallback to "")
          * PHASE_END        → end()
          * BACKEND_RESET    → end() (idempotent finalizer)
          * other kinds      → no-op (this renderer surfaces only the
                                token-stream lifecycle; other regions
                                are owned by SerpentFlow / OuroborosTUI)

        NEVER raises — defensive everywhere. Lazy import of EventKind
        keeps stream_renderer free of a hard import on the conductor
        primitive (the conductor module imports stream_renderer at boot
        time, not the other way around — preserves dependency direction).
        """
        if event is None:
            return
        try:
            kind = getattr(event, "kind", None)
            kind_value = getattr(kind, "value", None) or str(kind or "")
            if kind_value == "REASONING_TOKEN":
                content = getattr(event, "content", "") or ""
                if content:
                    self.on_token(content)
                return
            if kind_value == "PHASE_BEGIN":
                op_id = getattr(event, "op_id", None) or ""
                metadata = getattr(event, "metadata", None) or {}
                provider = ""
                try:
                    provider = str(metadata.get("provider", ""))
                except Exception:  # noqa: BLE001 — defensive
                    provider = ""
                if op_id:
                    self.start(op_id, provider)
                return
            if kind_value in ("PHASE_END", "BACKEND_RESET"):
                self.end()
                return
            # Other event kinds are not surfaced by this renderer.
        except Exception:  # noqa: BLE001 — defensive
            logger.debug(
                "[StreamRender] notify(event) failed", exc_info=True,
            )

    def flush(self) -> None:
        """Drain any pending tokens. Reuses the existing sync drain path."""
        try:
            self._drain_remaining_sync()
            self._render_buffer_safe()
        except Exception:  # noqa: BLE001 — defensive
            logger.debug("[StreamRender] flush failed", exc_info=True)

    def shutdown(self) -> None:
        """Tear down the active session if any. Idempotent — wraps end()."""
        try:
            self.end()
        except Exception:  # noqa: BLE001 — defensive
            logger.debug("[StreamRender] shutdown failed", exc_info=True)

    # ------------------------------------------------------------------
    # Introspection (for tests + debugging)
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def buffer(self) -> str:
        return self._buffer

    # ------------------------------------------------------------------
    # Consumer task — batches + renders
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Drain the queue at ~60fps cadence. Runs as a dedicated task.

        Pattern: wait up to ``_BATCH_INTERVAL_S`` for the next chunk,
        then drain anything else already queued (non-blocking), then
        flush the accumulated batch to the Live widget. This coalesces
        burst arrivals into a single render and never blocks if no
        tokens arrive — the timeout path just returns and loops.
        """
        pending: list = []
        last_render = time.monotonic()
        q = self._queue
        if q is None:
            return
        try:
            while True:
                # Compute a timeout that rounds down to the next render
                # boundary so we render at predictable cadence.
                elapsed = time.monotonic() - last_render
                timeout = max(0.001, _BATCH_INTERVAL_S - elapsed)
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=timeout)
                    pending.append(chunk)
                    self._token_count += 1
                except asyncio.TimeoutError:
                    pass

                # Opportunistic drain: pull anything already queued
                # without awaiting. Coalesces bursts into one render.
                while True:
                    try:
                        pending.append(q.get_nowait())
                        self._token_count += 1
                    except asyncio.QueueEmpty:
                        break

                # Flush if we have content and the batch interval elapsed.
                now = time.monotonic()
                if pending and (now - last_render) >= _BATCH_INTERVAL_S:
                    self._buffer += "".join(pending)
                    pending.clear()
                    self._render_buffer_safe()
                    # Same batch tick as the local widget, so the remote deck
                    # and a local terminal see the generation at one cadence.
                    self._mirror_completed_lines()
                    # The in-flight sentence, on the same tick — so the strip
                    # and the deck never disagree about where the boundary is.
                    self._publish_inflight_tail()
                    last_render = now
        except asyncio.CancelledError:
            # Final flush on cancellation (end() path). Any pending
            # chunks from the last interval land in the terminal before
            # Live is stopped.
            if pending:
                self._buffer += "".join(pending)
                pending.clear()
                self._render_buffer_safe()
            raise

    def _publish_inflight_tail(self, *, done: bool = False) -> None:
        """Send the sentence currently being WRITTEN to attached cockpits.

        WHAT THIS IS AND IS NOT
        ------------------------
        `_mirror_completed_lines` sends finished lines into the deck, where
        they stay. This sends the UNCOMMITTED remainder — the text after the
        last newline, which is the sentence the model is in the middle of.
        The two never overlap: everything before `_mirrored_offset` has
        already landed in the transcript, everything after it is in flight.

        WHY NOT MUTATE THE DECK'S TAIL
        -------------------------------
        The obvious shape is "replace the last line as it grows", and the
        deck's ring is APPEND-ONLY (`RegionBuffer.push`). Adding tail
        mutation would need an anchor per stream, a rule for what happens
        when another producer appends mid-sentence, and a re-wrap on every
        delta — a lot of new failure modes in the structure that holds the
        session's history.

        The cockpit already has a primitive for live, self-re-rendering
        state: the dynamic strip the agent view, status line and countdown
        all use. A strip sits directly under a bottom-anchored deck, so an
        in-flight sentence appears exactly where the next line will land —
        it reads as inline, and nothing has to mutate history to do it.

        Carried on the telemetry lane rather than the markup lane because it
        is STATE, not a transcript entry: the last frame wins, and a dropped
        one costs a frame of smoothness rather than a lost line.

        NEVER raises, and never blocks.
        """
        try:
            if not mirror_stream_enabled():
                return
            if not owns_inflight_publishing(_STREAM_RENDERER_PUBLISHER):
                return          # another backend carries the tail to cockpits
            tail = "" if done else self._buffer[self._mirrored_offset:]
            if tail == self._last_inflight and not done:
                return          # nothing new — do not spend a frame
            self._last_inflight = tail
            publish_inflight_tail(str(self._op_id or ""), tail, done=bool(done))
        except Exception:  # noqa: BLE001
            logger.debug(
                "[StreamRender] op=%s inflight degraded", self._op_id,
                exc_info=True,
            )

    def _mirror_completed_lines(self, *, final: bool = False) -> None:
        """Send completed LINES of the generation to attached cockpits.

        WHY THIS EXISTS
        ---------------
        The Rich `Live` widget is skipped whenever a SerpentREPL is active —
        it "writes via direct cursor manipulation that bypasses
        `patch_stdout` and clobbers the input prompt". In `ov` the REPL is
        always active, so the visible stream has never rendered there, and
        enabling it would corrupt the line the operator is typing on. The
        widget cannot be the answer; it is the thing that does not fit.

        What DOES fit is the channel the cockpit already has: a mirrored,
        line-oriented deck. So the generation arrives as it is written,
        a line at a time, next to the ⏺ blocks it belongs with.

        WHY NOT `find_commit_boundary`
        -------------------------------
        That helper commits at blank lines OUTSIDE fenced code blocks, so
        Rich can re-parse a complete Markdown block in scrollback. The deck
        does not re-parse — it draws escaped text. Borrowing that rule here
        would make a long paragraph or an open code fence show NOTHING until
        the stream ended, which is the silence this exists to fix. Different
        medium, different boundary: the last newline.

        NEVER raises, and never blocks — the bridge publish is a queue put.
        """
        try:
            if not mirror_stream_enabled():
                return
            end = len(self._buffer)
            if not final:
                nl = self._buffer.rfind("\n", self._mirrored_offset)
                if nl < 0:
                    return
                end = nl + 1
            chunk = self._buffer[self._mirrored_offset:end]
            if not chunk.strip():
                self._mirrored_offset = end
                return

            from rich.markup import escape

            lines = chunk.splitlines()
            sent_any = False
            for raw in lines:
                text = raw.rstrip()
                if not text and not sent_any:
                    continue
                # Model output is UNTRUSTED — escaped before it touches a
                # channel documented as styled chrome around inert data.
                body = escape(text[:_MIRROR_MAX_LINE_CHARS])
                if len(text) > _MIRROR_MAX_LINE_CHARS:
                    body += "…"
                if not self._mirror_opened:
                    # One ⏺ opens the block, as assistant prose does in the
                    # deck's grammar; continuations indent under it.
                    lead = f"⏺ {body}"
                    self._mirror_opened = True
                else:
                    lead = f"  {body}"
                if self._emit_deck_line(lead):
                    sent_any = True
            self._mirrored_offset = end
        except Exception:  # noqa: BLE001 — a mirror must not break a stream
            logger.debug(
                "[StreamRender] op=%s mirror degraded", self._op_id,
                exc_info=True,
            )

    def _emit_deck_line(self, lead: str) -> bool:
        """One composed deck line to whoever can actually see it.

        WHY THE STREAM WAS STILL INVISIBLE
        ----------------------------------
        `_mirror_completed_lines` correctly concluded that the Rich `Live`
        widget "is the thing that does not fit" and routed the generation
        into the cockpit's line-oriented deck instead. It then sent it with
        `publish_markup_global`, which returns False unless
        `attached_cockpits() > 0` — a client connected over the BRIDGE.

        `ov` boots the harness IN-PROCESS (`ov.py` → `battle_main`), with a
        SerpentREPL holding the operator's terminal. There is no bridge
        client in that shape, so every line was composed, escaped, offered,
        and dropped. The mirror published to everyone except the operator who
        was sitting in front of it.

        `cockpit_attach.operator_present()` already draws exactly this
        distinction, and already documents it: "a cockpit is ATTACHED over
        the bridge, or this process owns a real terminal (a foreground run
        with its own REPL, where the operator is looking straight at it)."
        It was written because Karen kept narrating to an empty room. This is
        the same defect in mirror image — text falling silent for the LOCAL
        operator for the same reason speech did for the remote one — and the
        concept did not need inventing, only consulting.

        WHY THIS IS SAFE WHERE `Live` WAS NOT
        --------------------------------------
        `print_fit` writes through the Rich console, which under
        prompt_toolkit's `patch_stdout` is coordinated with the prompt: the
        line lands in scrollback ABOVE the input and the prompt redraws
        below it. That is what `serpent_flow._emit_fit` has always done while
        the REPL is active. `Live` bypasses that with direct cursor
        manipulation, which is why it had to be skipped and why re-enabling
        it would still be wrong.

        Bridge first, local second, and never both for one line: a foreground
        run that ALSO has an attached client would otherwise print each line
        twice on the same terminal.
        """
        try:
            from backend.core.ouroboros.battle_test.cockpit_attach import (
                publish_markup_global,
            )
            if publish_markup_global(lead):
                return True
        except Exception:  # noqa: BLE001 — bridge unavailable; try local
            logger.debug("[StreamRender] bridge publish degraded",
                         exc_info=True)

        if not local_echo_enabled():
            return False
        try:
            from backend.core.ouroboros.battle_test.presentation_restraint import (
                print_fit,
                real_stdout_isatty,
            )
            # `real_stdout_isatty` reads `sys.__stdout__`, not `sys.stdout`.
            # Under `patch_stdout(raw=True)` the proxy reports False, and
            # testing the proxy is the load-bearing bug that made
            # `should_render()` blind in the presentation-restraint arc. The
            # same trap sits here: the ONE mode this fix exists for is the
            # one where the naive check is wrong.
            if not real_stdout_isatty():
                return False
            console = self._console
            if console is None:
                from rich.console import Console
                console = Console()
                self._console = console
            print_fit(console, lead)
            return True
        except Exception:  # noqa: BLE001 — a mirror must not break a stream
            logger.debug("[StreamRender] local echo degraded", exc_info=True)
            return False

    def _render_buffer_safe(self) -> None:
        """Swap the Markdown renderable on Live. Rich handles the
        actual terminal write on its background thread.

        Buffer is sliced to ``_RENDER_TAIL_CHARS`` to bound per-render
        parser work — prevents O(N²) event-loop pressure when a 16k-token
        stream re-parses a growing buffer on every 16ms batch tick.
        """
        if self._live is None:
            return
        try:
            from rich.markdown import Markdown
            if self._flow_mode:
                # §41.3 #27 — commit completed blocks ABOVE the Live region
                # into scrollback (Rich routes console.print above an active
                # Live display), then render only the in-progress tail in
                # the cage. Long generations accumulate in scrollback
                # instead of scrolling out of the tail-truncated widget.
                boundary = find_commit_boundary(
                    self._buffer, self._committed_offset,
                )
                if boundary > self._committed_offset:
                    chunk = self._buffer[self._committed_offset:boundary]
                    if chunk.strip():
                        self._live.console.print(Markdown(chunk))  # type: ignore[attr-defined]
                    self._committed_offset = boundary
                tail = self._buffer[self._committed_offset:]
                self._live.update(Markdown(tail[-_RENDER_TAIL_CHARS:]))  # type: ignore[attr-defined]
            else:
                self._live.update(Markdown(self._buffer[-_RENDER_TAIL_CHARS:]))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            logger.debug(
                "[StreamRender] Live.update failed", exc_info=True,
            )

    def _drain_remaining_sync(self) -> None:
        """Best-effort sync drain used in ``end()`` when the consumer
        task has been cancelled but hasn't fully run its except-block
        yet. Pulls anything still in the queue into the buffer so the
        final Live.update and the token_count metric are accurate.
        """
        q = self._queue
        if q is None:
            return
        while True:
            try:
                chunk = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            except Exception:  # noqa: BLE001
                break
            self._buffer += chunk
            self._token_count += 1


# ---------------------------------------------------------------------------
# Process-global singleton — matches OpsDigestObserver / LastSessionSummary
# ---------------------------------------------------------------------------

_DEFAULT_RENDERER: Optional[StreamRenderer] = None


def register_stream_renderer(renderer: Optional[StreamRenderer]) -> None:
    """Register the process-global renderer.

    Providers consult this on stream-start to know whether an operator
    terminal is watching. Called from the harness after SerpentFlow
    boots. Pass ``None`` to clear (also via ``reset_stream_renderer``).
    """
    global _DEFAULT_RENDERER
    _DEFAULT_RENDERER = renderer


def get_stream_renderer() -> Optional[StreamRenderer]:
    """Return the registered renderer or ``None`` if headless / not wired.

    Providers call this at the start of each streaming request. Return
    value is cached by the caller for the duration of one stream so
    late-registration doesn't split a single op across two modes.
    """
    return _DEFAULT_RENDERER


def reset_stream_renderer() -> None:
    """Clear the process-global singleton. Primarily for tests."""
    global _DEFAULT_RENDERER
    _DEFAULT_RENDERER = None
