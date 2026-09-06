"""The boot crest must paint atomically at a fixed height, and one line
must reach the cockpit once.

## Flicker — two mechanisms in the animator, both measured

1. `rich.live.Live` (screen=False) repaints by moving the cursor up the
   height of the LAST frame and rewriting. While the six-line boot log
   filled, one row per event, and again each time the progress gauge
   appeared or cleared, the region's height changed under it: the terminal
   scrolled and re-laid out the whole block. Each was a visible jump.
   Fixed by padding the log window to its own capacity plus the gauge slot
   from the very first frame.

2. Nothing grouped the erase and the rewrite into one paint, so at fourteen
   frames a second the terminal could show the region half-drawn. DEC mode
   2026 (synchronized output) is the terminal feature for exactly this and
   Rich does not emit it. Every paint is now bracketed — decided once per
   playback, emitted only to a real VT terminal.

## Twin lines — one line, two publishers

The harness swaps SerpentFlow's console for a spooled console that relays
everything printed to it; that relay's contract includes ambient output,
and the pinned tests say so. SerpentFlow ALSO mirrored the same lines
explicitly. So each `⏺ X queued` reached the cockpit twice: styled from the
mirror, plain from the relay. Closed at the producer: a mirror-once helper,
and a `mirror=False` request the relaying console honours and a plain
console never sees.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.ui import theme  # noqa: E402
from backend.core.ouroboros.ui import crest_animator as ca  # noqa: E402


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def test_a_real_vt_terminal_gets_synchronized_output() -> None:
    assert theme.supports_synchronized_output({"TERM": "xterm-256color"}, is_tty=True)


def test_no_terminal_no_brackets() -> None:
    """Into a pipe or a log they are bytes of noise."""
    assert not theme.supports_synchronized_output({"TERM": "xterm"}, is_tty=False)


@pytest.mark.parametrize("term", ["dumb", ""])
def test_a_non_vt_terminal_is_excluded(term) -> None:
    assert not theme.supports_synchronized_output({"TERM": term}, is_tty=True)


def test_the_operator_can_force_it_either_way() -> None:
    assert theme.supports_synchronized_output(
        {"JARVIS_SYNC_OUTPUT": "1", "TERM": "dumb"}, is_tty=False)
    assert not theme.supports_synchronized_output(
        {"JARVIS_SYNC_OUTPUT": "off", "TERM": "xterm"}, is_tty=True)


def test_the_sequences_are_dec_mode_2026() -> None:
    assert theme.SYNC_BEGIN == "\x1b[?2026h" and theme.SYNC_END == "\x1b[?2026l"


# ---------------------------------------------------------------------------
# Fixed height
# ---------------------------------------------------------------------------


def _plain_rows(renderable) -> list:
    from rich.console import Console
    c = Console(file=io.StringIO(), width=120, force_terminal=False,
                color_system=None)
    c.print(renderable)
    return c.file.getvalue().rstrip("\n").split("\n")


def _anim():
    return ca.CrestAnimator(cols=40, rows=10, log_lines=4, frame_count=2)


def test_the_log_window_has_the_same_height_empty_and_full() -> None:
    a = _anim()
    empty = len(_plain_rows(a.logs_renderable()))
    for i in range(4):
        a.add_log(f"event {i}")
    full = len(_plain_rows(a.logs_renderable()))
    a.set_progress("⎿ 40% · sensors arming")
    with_gauge = len(_plain_rows(a.logs_renderable()))
    a.clear_progress()
    after = len(_plain_rows(a.logs_renderable()))
    assert empty == full == with_gauge == after == 4 + 1, (
        "capacity plus the gauge slot, from the first frame to the last")


def test_the_region_never_changes_height_across_a_boot() -> None:
    a = _anim()
    heights = set()
    heights.add(len(_plain_rows(a.render(0.0))))
    for i in range(6):                      # more events than slots
        a.add_log(f"event {i}")
        heights.add(len(_plain_rows(a.render(0.3))))
    a.set_progress("gauge")
    heights.add(len(_plain_rows(a.render(0.6))))
    heights.add(len(_plain_rows(a.render_resting())))
    assert len(heights) == 1, f"region height varied: {sorted(heights)}"


def test_the_newest_events_win_the_window() -> None:
    a = _anim()
    for i in range(6):
        a.add_log(f"event {i}")
    rows = _plain_rows(a.logs_renderable())
    assert "event 5" in "\n".join(rows) and "event 0" not in "\n".join(rows)


def test_the_gauge_is_always_last() -> None:
    a = _anim()
    a.add_log("one")
    a.set_progress("GAUGE")
    rows = _plain_rows(a.logs_renderable())
    assert rows[-1].strip() == "GAUGE"


# ---------------------------------------------------------------------------
# Atomic paints
# ---------------------------------------------------------------------------


class _TTYStream(io.StringIO):
    def isatty(self) -> bool:  # noqa: D401
        return True


def test_every_paint_is_bracketed_on_a_terminal(monkeypatch) -> None:
    from rich.console import Console
    import asyncio
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYStream()
    console = Console(file=stream, force_terminal=True, width=80)
    a = _anim()
    stop = asyncio.Event(); stop.set()

    async def fast(_s):
        await asyncio.sleep(0)

    asyncio.run(a.play(console, stop_event=stop, fps=30, sleep_fn=fast,
                       max_frames=5))
    out = stream.getvalue()
    begins, ends = out.count(theme.SYNC_BEGIN), out.count(theme.SYNC_END)
    assert begins == ends >= 2, f"begin={begins} end={ends}"
    # BEGIN precedes the first frame bytes and END follows the last.
    assert out.index(theme.SYNC_BEGIN) < out.index("▀")
    assert out.rindex(theme.SYNC_END) > out.rindex("▀")


def test_no_brackets_into_a_plain_stream(monkeypatch) -> None:
    """A test console, a pipe, a log: the bytes would be noise."""
    from rich.console import Console
    import asyncio
    monkeypatch.setenv("TERM", "xterm-256color")
    console = Console(file=io.StringIO(), force_terminal=False, width=80)
    a = _anim()
    stop = asyncio.Event(); stop.set()

    async def fast(_s):
        await asyncio.sleep(0)

    asyncio.run(a.play(console, stop_event=stop, fps=30, sleep_fn=fast,
                       max_frames=3))
    assert theme.SYNC_BEGIN not in console.file.getvalue()


def test_the_bracket_helper_never_raises_on_a_broken_stream() -> None:
    class _Bad:
        is_terminal = True

        @property
        def file(self):
            raise RuntimeError("no file")

    sync = ca._synchronized_paint(_Bad())
    with sync():
        pass


# ---------------------------------------------------------------------------
# Twin lines
# ---------------------------------------------------------------------------


class _Relaying:
    """A console that relays, as the spooled one does."""
    relays_prints = True

    def __init__(self) -> None:
        self.calls: list = []

    def print(self, *a, **k) -> None:
        self.calls.append((a, k))


class _Plain:
    def __init__(self) -> None:
        self.calls: list = []

    def print(self, *a, **k) -> None:
        if "mirror" in k:
            raise TypeError("unexpected keyword argument 'mirror'")
        self.calls.append((a, k))


def _flow_with(console):
    from backend.core.ouroboros.battle_test.serpent_flow import SerpentFlow
    f = SerpentFlow.__new__(SerpentFlow)
    f.console = console
    f.markup_mirror = None
    return f


def test_print_mirrored_mirrors_once_and_asks_the_relay_to_stand_down() -> None:
    f = _flow_with(_Relaying())
    mirrored = []
    f.markup_mirror = mirrored.append
    f._print_mirrored("⏺ X queued")
    assert mirrored == ["⏺ X queued"]
    assert f.console.calls[0][1].get("mirror") is False


def test_a_plain_console_is_never_handed_the_kwarg() -> None:
    f = _flow_with(_Plain())
    f.markup_mirror = lambda _l: None
    f._print_mirrored("⏺ X queued")          # must not raise TypeError
    assert f.console.calls and "mirror" not in f.console.calls[0][1]


def test_emit_fit_forwards_the_stand_down(monkeypatch) -> None:
    from backend.core.ouroboros.battle_test import presentation_restraint as pr
    seen = {}

    def _fit(console, markup, **kw):
        seen.update(kw)

    monkeypatch.setattr(pr, "print_fit", _fit)
    f = _flow_with(_Relaying())
    f._clean_markup = lambda m: m
    f._emit_fit("line", mirror=False)
    assert seen.get("mirror") is False


def test_emit_fit_default_still_relays() -> None:
    f = _flow_with(_Relaying())
    f._clean_markup = lambda m: m
    f._emit_fit("line")
    assert all(k.get("mirror", True) for _a, k in f.console.calls)


def test_print_fit_passes_the_request_through_both_paths(monkeypatch) -> None:
    from backend.core.ouroboros.battle_test import presentation_restraint as pr
    got = []

    class _C:
        width = 40

        def print(self, *a, **k):
            got.append(k)
            if len(got) == 1:
                raise RuntimeError("rich path failed")   # force the fallback

    pr.print_fit(_C(), "[bold]x[/bold]", mirror=False)
    assert all(k.get("mirror") is False for k in got) and len(got) == 2


@pytest.mark.asyncio
async def test_the_spooled_console_honours_mirror_false() -> None:
    from backend.core.ouroboros.battle_test.spooled_console import (
        make_spooled_console,
    )
    seen = []
    console, spooler = make_spooled_console(lambda s, t: seen.append(t))
    spooler.start()
    console.print("relayed")
    console.print("held back", mirror=False)
    await spooler.flush()
    await spooler.stop()
    assert seen == ["relayed"]


def test_the_spooled_console_declares_the_seam() -> None:
    from backend.core.ouroboros.battle_test.spooled_console import (
        make_spooled_console,
    )
    console, _ = make_spooled_console(lambda s, t: None)
    assert getattr(console, "relays_prints", False) is True


# ---------------------------------------------------------------------------
# The wire carries lines
# ---------------------------------------------------------------------------


def test_publish_markup_refuses_a_renderable() -> None:
    from backend.core.ouroboros.battle_test.cockpit_attach import (
        CockpitAttachBridge,
    )
    from rich.panel import Panel
    bridge = CockpitAttachBridge.__new__(CockpitAttachBridge)
    bridge._clients = {}
    bridge.stats = {}
    bridge._loop = None
    bridge._backlog = SimpleNamespace(retain=lambda m: (_ for _ in ()).throw(
        AssertionError("a repr must never be retained")))
    bridge.publish_markup(Panel("x"))       # refused, never retained, no raise
