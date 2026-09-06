"""The organism's generation is visible on the cockpit as it is written.

Claude Code streams the model's text token by token. O+V had every piece
of that — a render conductor with a PHASE_BEGIN / REASONING_TOKEN /
PHASE_END triplet, a ``stream_inflight`` wire frame the cockpit already
renders as a live tail in the deck, a local client that reads its stream
chunk by chunk — and no path between them on a headless daemon:

* the only producer of the ``stream_inflight`` frame was a method on the
  TTY-gated stream renderer, which a headless daemon never constructs;
* the default transport no-op'd the stream triplet;
* the local lane's chunks reached stdout and the inter-token watchdog and
  no caller, because ``complete()`` had no token callback and the
  dispatcher opened no reasoning stream for it.

Now the producer is a module seam any backend may use, the transport is a
conductor backend that carries the triplet with a coalescing flush, and
the local dispatcher opens a reasoning stream per generation exactly as
the Claude provider does.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, List

import pytest

from backend.core.ouroboros.battle_test import cockpit_attach as ca
from backend.core.ouroboros.battle_test import stream_renderer as sr
from backend.core.ouroboros.governance import claude_style_transport as cst
from backend.core.ouroboros.governance import render_backends as rb
from backend.core.ouroboros.governance import render_conductor as rc


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS", "JARVIS_STREAM_MIRROR_ENABLED",
                 "JARVIS_RENDER_MODE"):
        monkeypatch.delenv(name, raising=False)
    from backend.core.ouroboros.governance import flag_registry as fr
    from backend.core.ouroboros.governance import intent_prompter as ip
    from backend.core.ouroboros.battle_test import narrative_channel as nc
    fr.reset_default_registry()
    nc.reset_default_channel_for_tests()
    monkeypatch.setattr(ip, "is_master_flag_enabled", lambda: False)
    sr.set_inflight_publisher(None)
    rc.reset_render_conductor()
    yield
    rc.reset_render_conductor()
    sr.set_inflight_publisher(None)
    nc.reset_default_channel_for_tests()
    fr.reset_default_registry()


class _Console:
    def __init__(self) -> None:
        self.prints: List[str] = []

    def print(self, text: str, **kw: Any) -> None:
        self.prints.append(text)


@pytest.fixture
def wire(monkeypatch):
    """Capture every `stream_inflight` frame the daemon would send."""
    frames: list = []
    monkeypatch.setattr(ca, "publish_telemetry_global", lambda p: (frames.append(dict(p)) or True))
    return frames


# ---------------------------------------------------------------------------
# The producer is a seam
# ---------------------------------------------------------------------------


def test_the_inflight_frame_has_one_producer(wire):
    assert sr.publish_inflight_tail("op-1", "the model is writ", done=False) is True
    assert sr.publish_inflight_tail("op-1", "ignored", done=True) is True
    assert wire == [
        {"kind": "stream_inflight", "op_id": "op-1", "text": "the model is writ", "done": False},
        {"kind": "stream_inflight", "op_id": "op-1", "text": "", "done": True},
    ]


def test_the_tail_is_capped_to_the_renderer_budget(wire):
    sr.publish_inflight_tail("op", "x" * (sr._INFLIGHT_MAX_CHARS + 50))
    assert len(wire[0]["text"]) == sr._INFLIGHT_MAX_CHARS


def test_the_renderer_yields_the_wire_when_another_backend_owns_it():
    assert sr.owns_inflight_publishing("stream_renderer")
    sr.set_inflight_publisher("claude_style")
    assert not sr.owns_inflight_publishing("stream_renderer")
    assert sr.owns_inflight_publishing("claude_style")
    src = inspect.getsource(sr.StreamRenderer._publish_inflight_tail)
    assert "owns_inflight_publishing(_STREAM_RENDERER_PUBLISHER)" in src
    assert "publish_inflight_tail(" in src and '"kind": "stream_inflight"' not in src


# ---------------------------------------------------------------------------
# The transport carries the triplet
# ---------------------------------------------------------------------------


def _event(kind: str, op_id: str, content: str = "", provider: str = "") -> Any:
    return SimpleNamespace(
        kind=SimpleNamespace(value=kind), op_id=op_id, content=content,
        metadata={"provider": provider} if provider else {},
    )


@pytest.mark.asyncio
async def test_the_first_token_goes_out_at_once_and_the_rest_coalesce(wire, monkeypatch):
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS", "200")
    t = cst.ClaudeStyleTransport(console=_Console())
    t.telemetry_mirror = lambda frame: wire.append(dict(frame))
    t.notify(_event("PHASE_BEGIN", "op-1", provider="local"))
    t.notify(_event("REASONING_TOKEN", "op-1", "I "))
    assert [f["text"] for f in wire] == ["I "]            # instant
    t.notify(_event("REASONING_TOKEN", "op-1", "will "))
    t.notify(_event("REASONING_TOKEN", "op-1", "read "))
    assert len(wire) == 1                                  # inside the window
    await asyncio.sleep(0.3)                               # the trailing flush
    assert wire[-1]["text"] == "I will read " and len(wire) == 2
    t.notify(_event("PHASE_END", "op-1"))
    assert wire[-1] == {"kind": "stream_inflight", "op_id": "op-1", "text": "", "done": True}
    assert "op-1" not in t._streams


@pytest.mark.asyncio
async def test_the_transport_prefers_its_direct_mirror_over_the_global(monkeypatch):
    """The bug this fixes: the module-global publish_telemetry_global reads
    _ACTIVE_BRIDGE, cleared on some mount paths while the live bridge keeps
    its clients. The transport holds a DIRECT bridge ref, like markup_mirror,
    and the stream must ride it — never the global — when it is wired."""
    global_calls: list = []
    monkeypatch.setattr(ca, "publish_telemetry_global",
                        lambda p: (global_calls.append(p) or True))
    direct: list = []
    t = cst.ClaudeStyleTransport(console=_Console())
    t.telemetry_mirror = lambda frame: direct.append(dict(frame))
    t.notify(_event("REASONING_TOKEN", "op-1", "hello"))
    assert direct and direct[0]["text"] == "hello"
    assert global_calls == []                       # never the global when mirrored


@pytest.mark.asyncio
async def test_without_a_mirror_the_stream_falls_back_to_the_global(wire, monkeypatch):
    """The foreground/TTY path has no transport-held ref; the global is the
    documented fallback there."""
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS", "1")
    t = cst.ClaudeStyleTransport(console=_Console())
    assert t.telemetry_mirror is None
    t.notify(_event("REASONING_TOKEN", "op-1", "hi"))
    assert wire and wire[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_the_harness_arms_the_telemetry_mirror():
    """The arming seam wires telemetry_mirror beside markup_mirror."""
    import inspect
    from backend.core.ouroboros.battle_test import harness as h
    src = inspect.getsource(h)
    assert "_pot.telemetry_mirror = bridge.publish_telemetry" in src
    i = src.index("_pot.markup_mirror = bridge.publish_markup")
    j = src.index("_pot.telemetry_mirror = bridge.publish_telemetry")
    assert 0 < i < j                                 # armed together, mirror first


@pytest.mark.asyncio
async def test_a_stream_that_never_announced_itself_still_shows(wire):
    t = cst.ClaudeStyleTransport(console=_Console())
    t.notify(_event("REASONING_TOKEN", "late", "hello"))
    assert wire and wire[0]["op_id"] == "late" and wire[0]["text"] == "hello"


def test_the_transport_still_partitions_every_event_kind():
    union = cst.ClaudeStyleTransport._HANDLED_KINDS | cst.ClaudeStyleTransport._NO_OP_KINDS
    assert {m.value for m in rc.EventKind} <= union
    assert {"PHASE_BEGIN", "REASONING_TOKEN", "PHASE_END"} <= cst.ClaudeStyleTransport._HANDLED_KINDS


@pytest.mark.asyncio
async def test_shutdown_drops_in_flight_streams(wire, monkeypatch):
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS", "5000")
    t = cst.ClaudeStyleTransport(console=_Console())
    t.notify(_event("REASONING_TOKEN", "op", "a"))
    t.notify(_event("REASONING_TOKEN", "op", "b"))   # scheduled, not sent
    t.shutdown()
    assert not t._streams
    await asyncio.sleep(0.01)
    assert len(wire) == 1                            # the scheduled flush was cancelled


# ---------------------------------------------------------------------------
# Wired through the conductor, end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reasoning_stream_reaches_the_cockpit_through_the_transport(monkeypatch):
    """End to end via the DIRECT telemetry sink — the daemon's real path,
    NOT the module-global helper (whose _ACTIVE_BRIDGE is unreliable)."""
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_STREAM_FLUSH_MS", "1")
    # A dead global proves the direct sink is what carries the stream.
    monkeypatch.setattr(ca, "publish_telemetry_global", lambda p: False)
    seen: list = []
    t = cst.ClaudeStyleTransport(console=_Console())
    t.telemetry_mirror = lambda frame: seen.append(dict(frame))
    conductor = rb.wire_render_conductor(per_op_transport=t)
    assert conductor is not None and t in conductor.backends()
    assert sr.owns_inflight_publishing("claude_style") and not sr.owns_inflight_publishing("stream_renderer")
    from backend.core.ouroboros.governance.render_primitives import get_reasoning_stream_callback
    cb = get_reasoning_stream_callback("op-9", provider="local")
    assert cb is not None
    cb("Reading ")
    await asyncio.sleep(0.02)
    cb("the callers")
    await asyncio.sleep(0.02)
    cb.end_callback()
    texts = [f["text"] for f in seen]
    assert texts[0] == "Reading " and "Reading the callers" in texts
    assert seen[-1]["done"] is True and seen[-1]["op_id"] == "op-9"


# ---------------------------------------------------------------------------
# The local lane feeds it
# ---------------------------------------------------------------------------


def test_the_local_client_hands_every_chunk_to_the_caller():
    import backend.core.ouroboros.governance.local_inference_director as lid
    for fn in (lid.LocalPrimeClient.complete, lid.LocalPrimeClient.complete_guarded,
               lid.LocalPrimeClient.generate):
        assert "on_token" in inspect.signature(fn).parameters
    assert "on_token" in inspect.signature(lid.LocalPrimeClient._complete_streaming).parameters
    assert "on_token=on_token" in inspect.getsource(lid.LocalPrimeClient.complete)
    body = inspect.getsource(lid.LocalPrimeClient._complete_streaming)
    assert "_emit_stream_token(delta)" in body and "on_token(delta)" in body


def test_the_dispatcher_opens_a_stream_only_for_a_client_that_can_take_it(monkeypatch):
    from backend.core.ouroboros.governance import providers as pv

    class _Cloud:
        async def generate(self, prompt, system_prompt=None, max_tokens=4096):
            return None

    class _Local:
        async def generate(self, prompt, **kwargs):
            return None

    assert pv._local_stream_kw(_Cloud(), "op") == {}
    rc.reset_render_conductor()
    assert pv._local_stream_kw(_Local(), "op") == {}     # no conductor → nothing to stream to
    seen: list = []

    class _Backend:
        name = "probe"

        def notify(self, event):
            seen.append((event.kind.value, event.op_id))

        def flush(self):
            pass

        def shutdown(self):
            pass

    rb.wire_render_conductor(per_op_transport=_Backend())
    kw = pv._local_stream_kw(_Local(), "op-3")
    assert callable(kw.get("on_token"))
    kw["on_token"]("tok")
    pv._end_local_stream(kw)
    assert [k for k, _ in seen] == ["PHASE_BEGIN", "REASONING_TOKEN", "PHASE_END"]
    assert all(op == "op-3" for _, op in seen)
    pv._end_local_stream(kw)                             # idempotent
    pv._end_local_stream({})                             # nothing to end


def test_every_local_generation_closes_its_stream_on_every_path():
    from backend.core.ouroboros.governance import providers as pv
    src = inspect.getsource(pv.PrimeProvider._generate_impl)
    assert src.count("_local_stream_kw(") == 2
    assert src.count("_end_local_stream(_stream_kw)") == 3   # error path, success path, finally
