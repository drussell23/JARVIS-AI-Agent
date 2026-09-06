"""The sentence being written, visible as it is written.

#70245 sent COMPLETED lines to the deck, which fixed the silence but not the
smoothness: an operator waited for a newline before anything appeared. CC
streams tokens into its transcript; this is the same effect through a
line-oriented bridge — completed lines land in the deck, and the uncommitted
remainder shows in a live strip directly beneath it.

The two never overlap. Everything before `_mirrored_offset` is transcript;
everything after it is in flight.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _legacy_raw_stream(monkeypatch):
    """This file pins the LEGACY raw in-flight stream rendering. The compact
    thinking indicator (default-on) is covered in test_cockpit_stream.py."""
    monkeypatch.setenv("JARVIS_COCKPIT_THINKING_INDICATOR_ENABLED", "false")
    yield

from backend.core.ouroboros.battle_test import cockpit_attach as ca
from backend.core.ouroboros.battle_test.stream_renderer import StreamRenderer
from backend.core.ouroboros.cli.ov import AttachUI


class _Bridge:
    def __init__(self) -> None:
        self._clients = {"c": object()}
        self.deck: list = []
        self.telemetry: list = []

    def publish_markup(self, text, session=None):
        self.deck.append(str(text))

    def publish_telemetry(self, payload):
        self.telemetry.append(payload)


@pytest.fixture
def bridge():
    b = _Bridge()
    ca.set_active_bridge(b)
    yield b
    ca.set_active_bridge(None)


def _renderer():
    r = StreamRenderer.__new__(StreamRenderer)
    r._buffer = ""
    r._mirrored_offset = 0
    r._mirror_opened = False
    r._op_id = "7759-86"
    r._last_inflight = ""
    return r


def _feed(r, ui, bridge, *chunks, done=False):
    for c in chunks:
        r._buffer += c
        r._mirror_completed_lines()
        r._publish_inflight_tail()
        if bridge.telemetry:
            ui.on_telemetry(bridge.telemetry[-1])
    if done:
        r._mirror_completed_lines(final=True)
        r._publish_inflight_tail(done=True)
        ui.on_telemetry(bridge.telemetry[-1])


class TestTheSentenceGrows:
    def test_text_appears_before_its_newline(self, bridge):
        """THE gap #70245 left. An operator waited for a newline before
        anything appeared at all."""
        ui = AttachUI()
        _feed(_renderer(), ui, bridge, "The vision floor ")
        assert bridge.deck == []            # nothing complete yet
        assert "The vision floor" in " ".join(ui._stream_rows())

    def test_it_grows_word_by_word(self, bridge):
        ui, r = AttachUI(), _renderer()
        seen = []
        for chunk in ("The vision ", "floor ", "raises"):
            _feed(r, ui, bridge, chunk)
            seen.append(" ".join(ui._stream_rows()).strip())
        assert seen[0] != seen[1] != seen[2]
        assert all(seen[i] in seen[i + 1] for i in range(2))

    def test_a_completed_line_MOVES_to_the_deck(self, bridge):
        ui, r = AttachUI(), _renderer()
        _feed(r, ui, bridge, "the caller swallows it.\n")
        assert any("swallows it." in d for d in bridge.deck)
        assert ui._stream_rows() == [], "it stayed in the strip too"

    def test_transcript_and_strip_never_overlap(self, bridge):
        """Everything before the commit offset is transcript; everything
        after is in flight. Showing a line in both would double it."""
        ui, r = AttachUI(), _renderer()
        _feed(r, ui, bridge, "first line\n", "second in progress")
        assert any("first line" in d for d in bridge.deck)
        strip = " ".join(ui._stream_rows())
        assert "first line" not in strip
        assert "second in progress" in strip

    def test_done_clears_the_strip(self, bridge):
        ui, r = AttachUI(), _renderer()
        _feed(r, ui, bridge, "trailing text", done=True)
        assert ui._stream_rows() == []
        assert any("trailing text" in d for d in bridge.deck)


class TestTheClientOwnsTheWrap:
    def test_it_wraps_to_THIS_terminal(self, bridge, monkeypatch):
        """The daemon serving two cockpits of different widths cannot
        pre-wrap for both, and the canvas draws with `wrap_lines=False` — an
        unwrapped sentence is clipped and appears to stop growing."""
        ui, r = AttachUI(), _renderer()
        monkeypatch.setattr(ui, "_terminal_size", lambda: (40, 30))
        _feed(r, ui, bridge, "word " * 30)
        rows = ui._stream_rows()
        assert len(rows) > 1
        assert all(len(x) <= 40 for x in rows)

    def test_a_wider_terminal_uses_fewer_rows(self, bridge, monkeypatch):
        ui, r = AttachUI(), _renderer()
        monkeypatch.setattr(ui, "_terminal_size", lambda: (40, 30))
        _feed(r, ui, bridge, "word " * 30)
        narrow = len(ui._stream_rows())
        monkeypatch.setattr(ui, "_terminal_size", lambda: (160, 30))
        assert len(ui._stream_rows()) < narrow

    def test_a_runaway_sentence_is_bounded(self, bridge):
        """An in-flight sentence is one thought, not a document. Past a few
        rows the strip would push the deck off screen to show text that is
        about to become deck content."""
        ui, r = AttachUI(), _renderer()
        _feed(r, ui, bridge, "word " * 800)
        rows = ui._stream_rows()
        assert len(rows) <= 4
        assert rows[0].strip() == "…"


class TestItIsStateNotTranscript:
    def test_the_LAST_frame_wins(self, bridge):
        """Carried on the telemetry lane because it is state. No
        accumulation to drift, and a dropped frame costs a tick of
        smoothness rather than a word."""
        ui = AttachUI()
        ui.on_telemetry({"kind": "stream_inflight", "text": "aaa"})
        ui.on_telemetry({"kind": "stream_inflight", "text": "bbb"})
        assert "bbb" in " ".join(ui._stream_rows())
        assert "aaa" not in " ".join(ui._stream_rows())

    def test_an_unchanged_tail_spends_no_frame(self, bridge):
        r = _renderer()
        r._buffer = "steady"
        r._publish_inflight_tail()
        n = len(bridge.telemetry)
        r._publish_inflight_tail()
        assert len(bridge.telemetry) == n

    def test_a_stale_frame_retires_the_strip(self, bridge):
        """A dead daemon must not leave half a sentence hanging."""
        import time
        ui = AttachUI()
        ui.on_telemetry({"kind": "stream_inflight", "text": "half a sentence"})
        assert ui._stream_rows()
        ui._stream_arrived = time.monotonic() - 10_000
        assert ui._stream_rows() == []

    def test_the_deck_is_never_mutated(self):
        """The obvious shape is "replace the last line as it grows", and the
        ring is APPEND-ONLY. Tail mutation would need an anchor per stream, a
        rule for a competing producer mid-sentence, and a re-wrap per delta —
        new failure modes in the structure holding the session's history."""
        import ast
        import inspect
        import textwrap
        # Checked as CALLS, not as text: the docstring names `push_raw` and
        # `replace` precisely to explain why they are not used, and a string
        # match cannot tell an explanation from an invocation.
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(StreamRenderer._publish_inflight_tail)))
        called = {
            getattr(n.func, "attr", getattr(n.func, "id", ""))
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        assert "push_raw" not in called
        assert not any("replace" in c for c in called)


class TestNeverBreaksTheStream:
    def test_no_cockpit_attached_is_silent_and_safe(self):
        ca.set_active_bridge(None)
        r = _renderer()
        r._buffer = "text"
        r._publish_inflight_tail()
        r._publish_inflight_tail(done=True)

    def test_a_raising_bridge_is_swallowed(self):
        class _Broken:
            _clients = {"c": 1}
            def publish_telemetry(self, *a, **k):
                raise RuntimeError("down")
        ca.set_active_bridge(_Broken())
        try:
            r = _renderer()
            r._buffer = "text"
            r._publish_inflight_tail()
        finally:
            ca.set_active_bridge(None)

    def test_the_master_flag_silences_it(self, bridge, monkeypatch):
        monkeypatch.setenv("JARVIS_STREAM_MIRROR_ENABLED", "0")
        r = _renderer()
        r._buffer = "text"
        r._publish_inflight_tail()
        assert bridge.telemetry == []

    def test_junk_frames_degrade(self):
        ui = AttachUI()
        for junk in (None, "x", {"kind": "stream_inflight"},
                     {"kind": "stream_inflight", "text": None}):
            ui.on_telemetry(junk)
            assert isinstance(ui._stream_rows(), list)


class TestTheDemoShowsIt:
    """`ov demo live` is where this gets WATCHED. A strip only the real
    organism can drive would be another surface built and never seen — the
    exact pattern that has cost this cockpit five modules."""

    def test_the_demo_reveals_a_sentence_progressively(self):
        from backend.core.ouroboros.cli import ov_demo as d
        seen = [" ".join(d._stream_rows(t)).strip()
                for t in (5.0, 5.6, 6.2, 6.8)]
        assert all(seen), "the generating window shows nothing"
        assert all(seen[i] in seen[i + 1] for i in range(3)), seen

    def test_it_vacates_when_the_line_becomes_transcript(self):
        """Showing it in both places would print the sentence twice."""
        from backend.core.ouroboros.cli import ov_demo as d
        lands = [t for t, line in d.compose_live_script()
                 if "vision floor" in str(line)]
        assert lands, "the voice beat vanished"
        assert d._stream_rows(lands[0]) == []
        assert d._stream_rows(lands[0] + 0.5) == []

    def test_the_reveal_finishes_BEFORE_the_line_lands(self):
        """Reasoning that arrives after the edit it justifies is not
        reasoning, it is a caption."""
        from backend.core.ouroboros.cli import ov_demo as d
        lands = min(t for t, line in d.compose_live_script()
                    if "vision floor" in str(line))
        assert d._stream_rows(lands - 0.1) != []

    def test_it_is_quiet_when_nothing_is_generating(self):
        """Idle means idle. Times chosen from the windows themselves, not
        written down — t=12.0 used to be idle and is now inside a running
        command, and a hardcoded moment silently stops testing what it
        was named for."""
        from backend.core.ouroboros.cli import ov_demo as d
        busy = list(d._GENERATING) + list(d._RUNNING)
        idle = [t for t in (0.5, 13.5, 14.5, 21.5)
                if not any(lo <= t <= hi for lo, hi in busy)]
        assert idle, "the script has no idle moment left to test"
        for t in idle:
            assert d._stream_rows(t) == [], t

    def test_a_RUNNING_command_shows_its_tail(self):
        """The 40-second black box, made watchable. Driven through the
        real LiveToolStream, so the demo exercises its coalescing,
        redaction and stderr tagging rather than a drawn picture."""
        from backend.core.ouroboros.cli import ov_demo as d
        (lo, hi) = list(d._RUNNING)[0]
        rows = d._stream_rows(lo + (hi - lo) * 0.8)
        assert rows, "a running command showed nothing"
        assert rows[0].strip().startswith("$"), rows[0]

    def test_the_running_header_survives_elision(self):
        """Row 0 says WHAT is running and for how long. The elision keeps
        newest rows, so without an exemption a long tail left test names
        with no subject."""
        from backend.core.ouroboros.cli import ov_demo as d
        (lo, hi) = list(d._RUNNING)[0]
        rows = d._stream_rows(hi - 0.2)
        assert rows and rows[0].strip().startswith("$"), rows

    def test_the_demo_calls_the_COCKPITS_renderer(self):
        """A second wrap here would keep agreeing with itself while the real
        one regressed — the rule stated at the top of `ov_demo`."""
        import inspect
        from backend.core.ouroboros.cli import ov_demo as d
        src = inspect.getsource(d._stream_rows)
        assert "render_inflight" in src
        assert "textwrap" not in src

    def test_the_strip_is_actually_MOUNTED(self):
        import inspect
        from backend.core.ouroboros.cli import ov_demo as d
        assert "stream_rows=" in inspect.getsource(d.scene_live)


class TestOneRendererTwoSurfaces:
    def test_the_cockpit_does_not_own_a_private_wrap(self):
        import inspect
        from backend.core.ouroboros.cli.ov import AttachUI
        src = inspect.getsource(AttachUI._stream_rows)
        assert "render_inflight" in src
        assert "textwrap" not in src

    def test_both_surfaces_render_identically(self):
        from backend.core.ouroboros.battle_test.stream_renderer import (
            render_inflight,
        )
        text = "the vision floor raises, and the caller swallows it entirely"
        ui = AttachUI()
        ui.on_telemetry({"kind": "stream_inflight", "text": text})
        ui._terminal_size = lambda: (72, 30)   # type: ignore[method-assign]
        assert ui._stream_rows() == render_inflight(text, width=72)

    def test_the_shape_survives_junk(self):
        from backend.core.ouroboros.battle_test.stream_renderer import (
            render_inflight,
        )
        # Nothing to say renders nothing. A non-string is a CALLER bug, and
        # stringifying it is the harmless degradation — type-checking here
        # would buy no safety the `except` does not already give.
        for empty in (None, "", "   ", "\n\t "):
            assert render_inflight(empty) == []        # type: ignore[arg-type]
        assert render_inflight(5) == ["  5"]           # type: ignore[arg-type]
        assert render_inflight("x", width=0) != []
        assert render_inflight("x", width=-40) != []
        assert render_inflight("x " * 400, max_rows=1) == ["  …"]
