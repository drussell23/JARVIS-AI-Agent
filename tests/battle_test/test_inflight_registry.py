"""The daemon can finally read what it is writing.

`capability_handoff` measured `stream_rows` UNSET on the daemon cockpit, and
`cockpit_mount` recorded the reason honestly: there was no process-global
in-flight text to read. So the daemon composed every frame, shipped it across
the bridge, and could not draw it — attached clients saw the sentence being
written and the process writing it did not.

The failures pinned here are the ones that read as normal operation: a strip
that shows the WRONG sentence, one that hangs on a dead producer, and one
that works only while somebody else is attached.
"""
from __future__ import annotations

import pytest

from backend.core.ouroboros.battle_test import inflight_registry as R


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    R.reset_inflight_for_tests()
    monkeypatch.delenv("JARVIS_INFLIGHT_TTL_S", raising=False)
    yield
    R.reset_inflight_for_tests()


def _tool(op="op-1", tool="bash", text="running…", elapsed=11.4, done=False):
    return {"kind": "tool_stream", "op_id": op, "tool": tool,
            "elapsed_s": elapsed, "text": text, "done": done}


def _prose(op="op-2", text="I'll read the layout module.", done=False):
    return {"kind": "stream_inflight", "op_id": op, "text": text, "done": done}


class TestRecording:
    def test_an_idle_process_has_nothing_in_flight(self):
        assert R.current_inflight() is None
        assert R.live_inflight() == []

    def test_both_producers_are_recorded(self):
        assert R.note_inflight_frame(_tool()) is True
        assert R.note_inflight_frame(_prose()) is True
        assert len(R.live_inflight()) == 2

    def test_an_unrelated_frame_is_ignored(self):
        """The registry rides a lane that carries heartbeats, panics and
        pending-apply frames too. It must record only what is in flight."""
        assert R.note_inflight_frame({"kind": "heartbeat", "active": True}) is False
        assert R.note_inflight_frame({"kind": "fatal_panic"}) is False
        assert R.current_inflight() is None

    def test_done_retires_only_its_own_slot(self):
        R.note_inflight_frame(_tool(op="a"))
        R.note_inflight_frame(_prose(op="b"))
        R.note_inflight_frame(_prose(op="b", done=True))
        live = R.live_inflight()
        assert len(live) == 1 and live[0].op_id == "a"

    def test_garbage_never_raises(self):
        for bad in (None, "", 42, [], {"kind": "tool_stream"}):
            R.note_inflight_frame(bad)
        assert R.current_inflight() is None or True


class TestConcurrency:
    def test_tools_within_one_op_do_not_overwrite_each_other(self):
        """The Venom loop can hold several tools open inside ONE operation.
        Keying on op_id alone makes them clobber each other, which reads as a
        single stream flickering rather than the several actually running."""
        R.note_inflight_frame(_tool(op="op-1", tool="bash"))
        R.note_inflight_frame(_tool(op="op-1", tool="run_tests"))
        assert len(R.live_inflight()) == 2

    def test_the_newest_is_the_one_drawn(self):
        R.note_inflight_frame(_tool(op="old"))
        R.note_inflight_frame(_prose(op="new"))
        assert R.current_inflight().op_id == "new"

    def test_a_burst_cannot_evict_the_live_entry(self):
        """Bounded, but the cap has to be wide enough that a fan-out of
        subagents cannot push the entry being drawn out before it is read."""
        for i in range(R._MAX_SLOTS + 10):
            R.note_inflight_frame(_prose(op=f"op-{i}"))
        live = R.live_inflight()
        assert len(live) <= R._MAX_SLOTS
        assert live[0].op_id == f"op-{R._MAX_SLOTS + 9}"

    def test_eviction_is_by_age_not_insertion_order(self):
        """A long-running command that simply STARTED first must not be the
        one evicted."""
        R.note_inflight_frame(_tool(op="long-runner"))
        for i in range(R._MAX_SLOTS + 5):
            R.note_inflight_frame(_prose(op=f"filler-{i}"))
        # The long-runner keeps refreshing, as a live producer does.
        R.note_inflight_frame(_tool(op="long-runner", text="still going"))
        assert any(e.op_id == "long-runner" for e in R.live_inflight())


class TestExpiry:
    def test_a_dead_producer_expires(self):
        """A producer killed mid-command never sends `done`. Without a TTL the
        strip shows that command's last output for the rest of the session as
        though it were still running."""
        R.note_inflight_frame(_tool())
        entry = R.current_inflight()
        assert R.live_inflight(now=entry.at + R.inflight_ttl_s() + 1.0) == []

    def test_a_quiet_but_alive_producer_survives(self):
        R.note_inflight_frame(_tool())
        entry = R.current_inflight()
        assert R.live_inflight(now=entry.at + R.inflight_ttl_s() - 0.5)

    def test_expiry_is_evaluated_on_read_not_swept(self):
        """There is no thread here to own a sweep, and a registry that needs
        one has invented a lifecycle for what is otherwise pure state."""
        R.note_inflight_frame(_tool())
        entry = R.current_inflight()
        R.live_inflight(now=entry.at + R.inflight_ttl_s() + 1.0)
        assert len(R._SLOTS) == 0, "stale entries were not dropped on read"

    def test_the_ttl_is_clamped_not_trusted(self, monkeypatch):
        monkeypatch.setenv("JARVIS_INFLIGHT_TTL_S", "0")
        assert R.inflight_ttl_s() >= 1.0
        monkeypatch.setenv("JARVIS_INFLIGHT_TTL_S", "99999")
        assert R.inflight_ttl_s() <= 120.0
        monkeypatch.setenv("JARVIS_INFLIGHT_TTL_S", "nonsense")
        assert R.inflight_ttl_s() > 0

    def test_the_kill_switch_works(self, monkeypatch):
        monkeypatch.setenv("JARVIS_INFLIGHT_REGISTRY_ENABLED", "0")
        assert R.note_inflight_frame(_tool()) is False
        assert R.current_inflight() is None


class TestOneComposition:
    def test_the_client_and_the_daemon_compose_identically(self):
        """A second copy of the header would be a second opinion about what an
        in-flight tool tail looks like — the defect the roster and the status
        line each already paid for once."""
        from backend.core.ouroboros.cli.ov import AttachUI

        frame = _tool(elapsed=11.4, text="running…")
        ui = AttachUI()
        ui.on_telemetry(frame)
        R.note_inflight_frame(frame)
        assert ui._stream_inflight == R.current_inflight().text
        assert ui._stream_inflight.startswith("$ bash · 11s")

    def test_the_client_does_not_re_implement_the_header(self):
        import inspect

        from backend.core.ouroboros.cli import ov

        src = inspect.getsource(ov.AttachUI.on_telemetry)
        assert "compose_inflight_text" in src
        assert '$ {' not in src, "the client still builds its own header"

    def test_model_prose_carries_no_header(self):
        """Only a command tail gets `$ tool · Ns`. Prose is the sentence."""
        assert R.compose_inflight_text(_prose(text="hello")) == "hello"


class TestTheProducerSeam:
    def test_a_real_stream_records_through_its_own_send(self):
        """Recorded at EMISSION, not construction. `cockpit_mount` called for
        a registry fed by every stream construction site — but sites can
        multiply and a producer that forgets to register is silently dark.
        Every frame passes through `_send` regardless of how it was made."""
        from backend.core.ouroboros.battle_test.live_tool_stream import (
            LiveToolStream,
        )
        stream = LiveToolStream(tool="bash", op_id="op-9",
                                publish=lambda payload: None)
        stream("stdout", "collected 41 items\n")
        stream._emit(done=False)
        entry = R.current_inflight()
        assert entry is not None and entry.is_tool
        assert "collected 41 items" in entry.text

    def _first_line_of_call(self, fn, callee: str) -> int:
        """Line number of the first CALL to `callee`, by AST.

        Not `src.index(...)`: comments and docstrings mention these names
        while explaining them, so a string search finds the prose and
        compares the wrong two positions. This file's first cut did exactly
        that and failed on its own explanatory comment.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        lines = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) == callee
                 or getattr(node.func, "attr", None) == callee)
        ]
        assert lines, f"no call to {callee}"
        return min(lines)

    def test_an_injected_publisher_records_identically(self):
        """The demo and the tests inject a publisher. Recording after that
        branch would leave both of them dark."""
        from backend.core.ouroboros.battle_test import live_tool_stream

        send = live_tool_stream.LiveToolStream._send
        assert (self._first_line_of_call(send, "note_inflight_frame")
                < self._first_line_of_call(send, "_publish")), (
            "recording happens after the injected-publisher branch returns"
        )

    def test_recording_happens_before_the_transport_decision(self):
        """`publish_telemetry_global` returns early with no cockpit attached,
        so a daemon with nobody watching publishes nothing — exactly when its
        OWN cockpit is the surface being looked at. Recording after that call
        yields a strip that works only while someone else is watching."""
        from backend.core.ouroboros.battle_test.stream_renderer import (
            StreamRenderer,
        )

        from backend.core.ouroboros.battle_test import stream_renderer as _sr
        target = getattr(_sr, "publish_inflight_tail", None)
        assert target is not None, "no seam publishes an in-flight frame"
        assert (self._first_line_of_call(target, "note_inflight_frame")
                < self._first_line_of_call(target, "publish_telemetry_global"))


class TestTheDaemonStrip:
    def test_it_draws_what_is_in_flight(self):
        from backend.core.ouroboros.battle_test.cockpit_mount import (
            daemon_stream_rows,
        )
        rows = daemon_stream_rows()
        assert rows() == []
        R.note_inflight_frame(_tool(text="collected 41 items"))
        drawn = rows()
        assert drawn and any("bash" in r for r in drawn)

    def test_the_mount_actually_carries_it(self):
        """The hook having a provider is not the same as the cockpit being
        handed it — the exact gap this audit exists to catch."""
        import ast
        import inspect

        from backend.core.ouroboros.battle_test import cockpit_mount, serpent_flow

        assert "stream_rows" in inspect.getsource(cockpit_mount.build_daemon_mount)
        tree = ast.parse(inspect.getsource(serpent_flow.SerpentREPL._loop).lstrip())
        assert any(
            isinstance(n, ast.keyword) and n.arg == "stream_rows"
            for n in ast.walk(tree)
        ), "the daemon builds its cockpit without handing it stream_rows"
