"""A console whose output also reaches whoever asked for it.

The daemon runs detached. Its ~76 `dispatch_<verb>_command` handlers each call
``console.print(...)``, which renders onto a terminal nobody is watching — so
an operator who types `/posture status` in an attached cockpit sees the verb
dispatch, succeed, and produce nothing. Talking to an empty room.

Fixing that in the handlers would mean 76 edits and a 77th that forgets. The
presentation layer is the isolated thing, so the console is what changes: one
object, injected once, and every handler is mirrored without knowing it.

Three properties make it safe to put in front of a running daemon.

**It never blocks the event loop.** ``Console.print`` is synchronous and is
called from inside the loop. Doing UDS I/O there would stall every other
operation behind the slowest attached client. So print() renders to a string,
drops it on an ``asyncio.Queue`` with ``put_nowait``, and returns; a
background task drains to the bridge.

**It never grows without bound.** The queue is bounded. A client that stops
reading gets its backlog DROPPED, oldest first, with a counter — a detached
or wedged cockpit must not become a memory leak in the organism.

**It addresses, rather than broadcasts.** The session that ran the verb is
read from the ``ContextVar`` at print time, so `/posture status` typed in one
terminal does not paint in another. Ambient output — anything not inside a
session scope — still goes to everyone, which is correct: an autonomous
operation belongs to no one and is everyone's business.

The local render is unchanged. This ADDS a consumer; it does not redirect one.
A daemon started in the foreground still prints to its own terminal.
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger("Ouroboros.SpooledConsole")

__all__ = ["ConsoleSpooler", "make_spooled_console"]

#: Per-process ceiling on undrained output. Deliberately generous — a verb
#: that prints a large table is normal — but finite, because an attached
#: cockpit that stops reading must not be able to exhaust the daemon.
_QUEUE_MAX = 512


class ConsoleSpooler:
    """Buffers rendered output and drains it to a sink off the print path.

    The sink is called with ``(session_id, markup_text)``. ``session_id`` is
    None for ambient output, which the bridge broadcasts.
    """

    def __init__(self, sink: Callable[[Optional[str], str], Any],
                 maxsize: int = _QUEUE_MAX) -> None:
        self._sink = sink
        self._queue: "asyncio.Queue[Tuple[Optional[str], str]]" = (
            asyncio.Queue(maxsize=maxsize)
        )
        self._task: Optional["asyncio.Task[None]"] = None
        self.dropped = 0
        #: How many drops have already been announced, so the notice is
        #: coalesced per window instead of once per lost line.
        self._lag_reported = 0
        self.spooled = 0

    def offer(self, session_id: Optional[str], text: str) -> bool:
        """Enqueue without ever blocking. False if the payload was dropped.

        Drop-oldest rather than drop-newest: when a cockpit falls behind, the
        RECENT lines are the ones an operator is waiting for. Discarding what
        they have already conceptually missed is the lesser loss.
        """
        if not text:
            return False
        try:
            self._queue.put_nowait((session_id, text))
            self.spooled += 1
            return True
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()          # evict oldest
                self._queue.task_done()
                self._queue.put_nowait((session_id, text))
                self.dropped += 1
                return True
            except Exception:  # noqa: BLE001
                self.dropped += 1
                return False
        except Exception:  # noqa: BLE001 — output must never raise into a verb
            return False

    def _lag_notice(self) -> Optional[str]:
        """One line naming what this cockpit did NOT receive, or None.

        `self.dropped` has counted drop-oldest evictions since this spooler
        was built and nothing has ever read it. A bounded queue that silently
        discards is correct engineering and dishonest reporting: the operator
        sees a continuous transcript with holes in it and has no way to know.

        Coalesced, not per-drop — under load the notice would otherwise become
        the thing crowding the queue. Emitted once per window on the same
        principle as the SSE broker's single `stream_lag` per window, so the
        two backpressure surfaces say the same thing the same way.
        """
        try:
            from backend.core.ouroboros.battle_test.backpressure_notice import (
                coalesced_drop_notice,
            )
            notice, self._lag_reported = coalesced_drop_notice(
                self.dropped, self._lag_reported,
                unit="line",
                detail=("this cockpit fell behind; the daemon's log has the "
                        "full record"),
            )
            return notice
        except Exception:  # noqa: BLE001
            return None

    async def _drain(self) -> None:
        while True:
            try:
                session_id, text = await self._queue.get()
            except asyncio.CancelledError:
                return
            # Announce the hole BEFORE the frame that follows it, so the
            # notice appears where the gap actually is rather than after the
            # next unrelated block.
            notice = self._lag_notice()
            if notice is not None:
                try:
                    _r = self._sink(session_id, notice)
                    if asyncio.iscoroutine(_r):
                        await _r
                except asyncio.CancelledError:
                    return
                except Exception:  # noqa: BLE001
                    pass
            try:
                result = self._sink(session_id, text)
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 — one bad frame is not fatal
                logger.debug("[Spooler] sink failed", exc_info=True)
            finally:
                self._queue.task_done()

    def start(self) -> bool:
        """Begin draining. Requires a running loop; False if there is none."""
        if self._task is not None and not self._task.done():
            return True
        try:
            self._task = asyncio.get_running_loop().create_task(self._drain())
            return True
        except RuntimeError:
            # No loop yet — output still spools and drains once start() is
            # called from inside one. Silently dropping here would make a
            # boot-time print vanish for no visible reason.
            return False

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def flush(self, timeout: float = 1.0) -> None:
        """Wait for the backlog to drain — tests and clean shutdown."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass

    @property
    def pending(self) -> int:
        return self._queue.qsize()


def make_spooled_console(
    sink: Callable[[Optional[str], str], Any],
    *,
    base: Any = None,
    width: int = 120,
) -> Tuple[Any, ConsoleSpooler]:
    """Return ``(console, spooler)``. The console mirrors every print.

    Built by SUBCLASSING the live Console class rather than wrapping it: verb
    handlers call `console.print`, `console.rule`, `console.log` and reach for
    attributes like `console.width`, and a wrapper would have to reimplement
    each one it forgot. A subclass inherits the entire surface and overrides
    the single method that matters.
    """
    from rich.console import Console

    spooler = ConsoleSpooler(sink)

    class SpooledConsole(Console):  # type: ignore[misc]
        """A Console that also mirrors to whoever ran the command — at the
        width of the terminal that will actually display it.

        `size` is overridden rather than each consumer being converted, for
        the same reason this class exists at all: `print_fit` reads
        ``console.width``, Rich derives wrapping and table layout from
        ``console.size``, and the diff formatters ask the console how much
        room they have. Every one of them becomes capability-aware by
        changing the object they already hold — no call site moves.

        Before this, the daemon rendered every mirrored line at a literal 120
        columns no matter who was watching: an 80-column cockpit got wrapped
        diffs and a broken gutter, a 200-column one got two-thirds of a
        screen. `terminal_capabilities` knows the real answer per subscriber;
        this is where that answer is spent.
        """

        @property
        def size(self):  # type: ignore[override]
            """Live dimensions for THIS render. NEVER raises.

            Read per access, not cached: a SIGWINCH between two prints must
            take effect on the second one. Falls through to Rich's own
            detection when nothing has declared, so a foreground daemon on a
            real terminal keeps behaving exactly as it did.
            """
            try:
                from rich.console import ConsoleDimensions

                from backend.core.ouroboros.battle_test.terminal_capabilities import (  # noqa: E501
                    current_capabilities,
                )
                caps = current_capabilities()
                if caps is not None and caps.cols > 0:
                    return ConsoleDimensions(caps.cols, max(1, caps.rows))
            except Exception:  # noqa: BLE001 — never break a render path
                pass
            try:
                return super().size
            except Exception:  # noqa: BLE001
                from rich.console import ConsoleDimensions
                return ConsoleDimensions(width, 24)

        #: The seam a producer checks before passing ``mirror=False``. A
        #: plain Rich console has no such kwarg and would raise on it; the
        #: attribute, not the class name, is what a producer keys on, so a
        #: future relaying console needs only to declare it.
        relays_prints = True

        def print(self, *args: Any, **kwargs: Any) -> None:  # noqa: A003
            # LOCAL RENDER FIRST, and unconditionally. A daemon running in the
            # foreground must look exactly as it did; mirroring is additive.
            # `mirror=False`: print locally, do NOT relay. For a producer that
            # has ALREADY sent this line to the cockpit in its styled form
            # through `markup_mirror`. Without it every such line reached the
            # cockpit twice — once styled from the mirror, once plain from
            # this relay (measured 2026-09-06: each `⏺ X queued` arrived as a
            # pair). The relay stays the carrier for everything else,
            # addressed and ambient alike; the contract below is unchanged.
            mirror = bool(kwargs.pop("mirror", True))
            try:
                super().print(*args, **kwargs)
            except Exception:  # noqa: BLE001
                pass
            if not mirror:
                return
            try:
                # Rendered for the cockpit this line is ADDRESSED to —
                # `current_session()` is read below for routing, and the
                # width must come from the same subscriber or the text is
                # wrapped for a terminal that will never see it.
                text = _render_to_text(args, kwargs, width=_mirror_width(width))
                if text.strip():
                    from backend.core.ouroboros.battle_test.attach_session import (
                        current_session,
                    )
                    # Read the session HERE, on the calling task, so the
                    # answer belongs to whoever ran the verb. Reading it in
                    # the drain task would give the drainer's context, which
                    # is nobody's.
                    spooler.offer(current_session(), text.rstrip("\n"))
            except Exception:  # noqa: BLE001 — mirroring must never break a verb
                logger.debug("[SpooledConsole] mirror failed", exc_info=True)

    console = SpooledConsole(
        file=getattr(base, "file", None), width=width,
        force_terminal=getattr(base, "is_terminal", False) or None,
    )
    return console, spooler


def _mirror_width(fallback: int) -> int:
    """Columns to render a MIRRORED line at. NEVER raises.

    Resolves through `terminal_capabilities`, which answers per-subscriber
    for addressed output and with the minimum across live cockpits for
    ambient. `fallback` is the caller's literal and is reached only when no
    display has ever declared — a foreground daemon with no cockpit attached.
    """
    try:
        from backend.core.ouroboros.battle_test.terminal_capabilities import (
            effective_width,
        )
        return int(effective_width(fallback))
    except Exception:  # noqa: BLE001
        return int(fallback)


def _render_to_text(args: Any, kwargs: Any, *, width: int) -> str:
    """Render the same arguments to plain text, off to the side.

    A private Console writing to a StringIO, NOT ``self.capture()`` — capture
    redirects the console's own file, so the local render would be swallowed
    and a foreground daemon would go silent. Two consoles, one set of
    arguments, neither interfering with the other.
    """
    try:
        from rich.console import Console

        buffer = io.StringIO()
        scratch = Console(file=buffer, width=width, no_color=True,
                          highlight=False, soft_wrap=True)
        # The SAME theme the daemon's console carries. A renderable styled
        # with a semantic token (the boot summary's Panel, ``border_style=
        # "muted"``) renders locally and RAISED here, so it fell to the
        # ``str(a)`` fallback below and the cockpit's backlog replayed
        # ``<rich.panel.Panel object at 0x…>`` (measured 2026-09-06).
        try:
            from backend.core.ouroboros.ui import theme as _theme
            _theme.ensure_theme(scratch)
        except Exception:  # noqa: BLE001 — a theme fault must not lose the line
            pass
        safe = {k: v for k, v in kwargs.items()
                if k in ("sep", "end", "justify", "overflow", "markup",
                         "highlight", "emoji", "style")}
        scratch.print(*args, **safe)
        return buffer.getvalue()
    except Exception:  # noqa: BLE001
        try:
            return " ".join(str(a) for a in args)
        except Exception:  # noqa: BLE001
            return ""
