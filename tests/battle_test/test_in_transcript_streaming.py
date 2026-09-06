"""Tokens land IN the transcript, not in a strip beneath it.

I reported for three status reports running that per-token streaming was
blocked by `RegionBuffer` being append-only, and that closing it would
need per-stream anchors, interleaving rules and a re-wrap per delta.

That was wrong. `as_text()` is `"\\n".join(self._lines)` — it COMPOSES on
every read. The ring is append-only for HISTORY, which is correct and
load-bearing, but nothing ever required the in-flight line to enter it.
Every complication I cited came from assuming mutation.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _legacy_raw_stream(monkeypatch):
    """This file pins the LEGACY raw in-flight stream rendering. The compact
    thinking indicator (default-on) is covered in test_cockpit_stream.py."""
    monkeypatch.setenv("JARVIS_COCKPIT_THINKING_INDICATOR_ENABLED", "false")
    yield

from backend.core.ouroboros.battle_test.split_layout import RegionBuffer


class TestTheTailIsComposedNotMutated:
    def test_it_renders_at_the_end_of_the_transcript(self):
        b = RegionBuffer(name="deck", maxlen=10)
        b.push("⏺ Signal(test_failure)")
        b.set_pending("  the vision floor raises")
        assert b.as_text().splitlines()[-1] == "  the vision floor raises"

    def test_a_competing_producer_cannot_displace_it(self):
        """The reason no anchor or ordering rule is needed: composition
        puts the tail last on every frame, whatever else arrived."""
        b = RegionBuffer(name="deck", maxlen=10)
        b.set_pending("  mid-sentence")
        b.push("⏺ Bash(pytest -q)")
        assert b.as_text().splitlines()[-1] == "  mid-sentence"

    def test_the_tail_is_NOT_history(self):
        """A pending line inside the ring would be evicted by maxlen
        mid-sentence, and would survive as history if the stream died
        before completing. It is a view of something happening, not a
        record of something that did."""
        b = RegionBuffer(name="deck", maxlen=2)
        b.set_pending("  in flight")
        for i in range(5):
            b.push(f"line {i}")
        assert b.snapshot() == ("line 3", "line 4")   # ring bounded
        assert b.as_text().splitlines()[-1] == "  in flight"  # survives

    def test_completing_a_line_leaves_no_duplicate(self):
        b = RegionBuffer(name="deck", maxlen=10)
        b.set_pending("  half a sen")
        b.set_pending("  half a sentence done")
        b.push("  half a sentence done")
        b.set_pending("")
        assert b.as_text().count("half a sentence done") == 1

    def test_an_empty_tail_changes_nothing(self):
        b = RegionBuffer(name="deck", maxlen=10)
        b.push("only line")
        before = b.as_text()
        b.set_pending("")
        assert b.as_text() == before

    def test_a_tail_with_no_history_still_renders(self):
        b = RegionBuffer(name="deck", maxlen=10)
        b.set_pending("  first words of the session")
        assert "first words" in b.as_text()

    def test_clear_takes_the_tail_with_it(self):
        b = RegionBuffer(name="deck", maxlen=10)
        b.push("x"); b.set_pending("  pending")
        b.clear()
        assert b.as_text() == ""

    @pytest.mark.parametrize("junk", [None, 42, object()])
    def test_junk_degrades(self, junk):
        b = RegionBuffer(name="deck", maxlen=10)
        b.set_pending(junk)          # type: ignore[arg-type]
        assert isinstance(b.as_text(), str)


class TestItIsActuallyREACHABLE:
    """The seam existing is not the property that matters — five inert
    features this session proved that."""

    def test_the_mux_exposes_the_seam(self):
        from backend.core.ouroboros.battle_test.bipartite_layout import (
            BipartiteLayout,
        )
        assert hasattr(BipartiteLayout, "set_streaming_tail")

    def test_the_repl_hands_its_mux_to_the_caller(self):
        """Without this the client holds no reference to the deck, and the
        seam is reachable in principle and inert in practice."""
        import inspect

        from backend.core.ouroboros.battle_test.bipartite_layout import (
            run_bipartite_repl,
        )
        assert "on_mux" in inspect.signature(run_bipartite_repl).parameters

    def test_the_client_captures_it_and_feeds_it(self):
        import inspect

        from backend.core.ouroboros.cli import ov
        src = inspect.getsource(ov)
        assert "on_mux=_capture_mux" in src
        assert "_push_tail_to_deck" in src

    def test_the_strip_stands_down_when_the_deck_carries_it(self):
        """Rendering the same text twice is worse than either placement
        alone."""
        from backend.core.ouroboros.cli.ov import AttachUI
        ui = AttachUI()
        ui._terminal_size = lambda: (80, 30)   # type: ignore[method-assign]
        ui.on_telemetry({"kind": "stream_inflight", "text": "hello there"})
        assert ui._stream_rows(), "no deck -> the strip must still stream"

        class _Mux:
            def set_streaming_tail(self, _t): pass

        ui._mux = _Mux()
        assert ui._stream_rows() == [], "both surfaces drew the same text"
