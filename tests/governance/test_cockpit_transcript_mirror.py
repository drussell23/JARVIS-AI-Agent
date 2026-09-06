"""What the organism does unasked must reach the operator's transcript.

## The measurement

`ov` attached to a healthy organism. The prompt was live, the status line
was live, and the transcript pane stayed EMPTY while the daemon ran 47
generations, 74 tool calls and 13 no-ops (session bt-2026-09-06-043834).
Every one of those events had a renderer. None of them had a path.

Three cuts, in order along the wire:

1. **The default per-op transport had no mirror.** `JARVIS_RENDER_MODE`
   defaults to CLAUDE, which selects `ClaudeStyleTransport`. Every line it
   rendered went to `self._console` -- on a headless daemon, the log file.
   `SerpentFlow` carried a `markup_mirror` to the attach bridge; the
   transport that replaced it as the default did not, and the harness
   never retained it, so nothing could have wired one.

2. **The tool-call channel was read for `phase` only.** `ToolNarrationChannel`
   emits every Venom tool call as a HEARTBEAT carrying `tool_name`,
   `tool_args_summary` and `result_preview`. The handler printed
   ``└ generate`` -- and only when `show_heartbeats()` was on, which it is
   not by default. The name, the arguments and the result were discarded.

3. **The cockpit routed autonomous work away from the transcript.** The
   client sink sent every frame the daemon did not address to a specific
   terminal to the deck -- four rows, sixty-second expiry -- and never to
   the canvas. Correct for a reactive tool where the transcript is what you
   asked for; inverted for a proactive one where what it does unasked IS
   the product.

## What these tests hold

Each cut, independently, plus the seams they share with existing code:
the composer SerpentFlow already uses, the deck's own severity classifier,
and the renderer addressed output already goes through.
"""
from __future__ import annotations

import ast
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.governance import claude_style_transport as cst  # noqa: E402

_HARNESS = _REPO / "backend" / "core" / "ouroboros" / "battle_test" / "harness.py"
_OV = _REPO / "backend" / "core" / "ouroboros" / "cli" / "ov.py"


class _Console:
    def __init__(self) -> None:
        self.prints: list = []

    def print(self, text, *a, **k) -> None:
        self.prints.append(str(text))


def _msg(msg_type: str, op_id: str = "op-01a0-test", **payload):
    return SimpleNamespace(
        msg_type=SimpleNamespace(value=msg_type), op_id=op_id, payload=payload,
    )


def _intent(t, op_id="op-01a0-test"):
    return t.send(_msg("INTENT", op_id, sensor="TodoScanner",
                       goal="fix it", target_files=["a.py"], risk_tier="SAFE_AUTO"))


# ---------------------------------------------------------------------------
# Cut 1 — the default transport mirrors every line it renders
# ---------------------------------------------------------------------------


def test_the_mirror_has_the_same_name_and_contract_as_serpentflow() -> None:
    """One harness idiom wires both: `x.markup_mirror = bridge.publish_markup`."""
    t = cst.ClaudeStyleTransport(console=_Console())
    assert hasattr(t, "markup_mirror") and t.markup_mirror is None


def test_every_rendered_line_reaches_the_mirror() -> None:
    console, mirrored = _Console(), []
    t = cst.ClaudeStyleTransport(console=console)
    t.markup_mirror = mirrored.append
    asyncio.run(_intent(t))
    asyncio.run(t.send(_msg("DECISION", outcome="completed", files_changed=["a.py"])))
    assert console.prints, "the local render still happens"
    assert mirrored == console.prints, "the cockpit sees exactly what the console sees"


def test_the_mirror_fires_even_when_the_console_is_gone() -> None:
    """A headless daemon's console can be None. The cockpit must not be."""
    mirrored = []
    t = cst.ClaudeStyleTransport(console=None)
    t.markup_mirror = mirrored.append
    asyncio.run(_intent(t))
    assert len(mirrored) == 1


def test_a_mirror_fault_never_costs_the_local_render() -> None:
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)

    def _boom(_line):
        raise RuntimeError("bridge is down")

    t.markup_mirror = _boom
    asyncio.run(_intent(t))
    assert len(console.prints) == 1


# ---------------------------------------------------------------------------
# Cut 2 — a tool call renders as a tool block, not a phase tick
# ---------------------------------------------------------------------------


def _tool_heartbeat(op_id="op-01a0-test", starting=False, **over):
    payload = dict(phase="generate", tool_name="read_file",
                   tool_args_summary="backend/x.py", round_index=0,
                   result_preview="212 lines", duration_ms=12.5,
                   status="success", tool_starting=starting)
    payload.update(over)
    return _msg("HEARTBEAT", op_id, **payload)


def test_a_completed_tool_call_renders_regardless_of_heartbeat_chatter(
        monkeypatch) -> None:
    """`show_heartbeats()` governs phase ticks. Tool activity is not a phase
    tick and has its own, already-existing master gate."""
    monkeypatch.setattr(cst, "show_heartbeats", lambda: False)
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(_intent(t))
    console.prints.clear()
    asyncio.run(t.send(_tool_heartbeat()))
    blob = "\n".join(console.prints)
    # The registry composer renders `read_file` in Claude Code's idiom as
    # `⏺ Read(...)` — the human label, not the internal id. Assert the
    # block, not the id: the action glyph, and the ARGUMENTS the model
    # chose, which is the part the operator actually reads.
    assert "⏺" in blob, "a tool block must be rendered"
    assert "backend/x.py" in blob, "with the tool's arguments"


def test_the_start_event_is_silent_so_nothing_renders_twice(monkeypatch) -> None:
    monkeypatch.setattr(cst, "show_heartbeats", lambda: False)
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(_intent(t))
    console.prints.clear()
    asyncio.run(t.send(_tool_heartbeat(starting=True)))
    assert console.prints == []


def test_a_tool_call_from_an_op_whose_intent_was_missed_still_renders(
        monkeypatch) -> None:
    """The old handler required `op_id in self._op_state`. A cockpit that
    attached mid-op must still see what that op is doing."""
    monkeypatch.setattr(cst, "show_heartbeats", lambda: False)
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(t.send(_tool_heartbeat(op_id="op-never-announced")))
    assert any("⏺" in p and "backend/x.py" in p for p in console.prints)


def test_a_plain_phase_tick_keeps_its_old_gate(monkeypatch) -> None:
    monkeypatch.setattr(cst, "show_heartbeats", lambda: False)
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(_intent(t))
    console.prints.clear()
    asyncio.run(t.send(_msg("HEARTBEAT", phase="generate")))
    assert console.prints == [], "no tool_name → a phase tick → silent by default"


def test_the_channel_master_gate_silences_tool_blocks(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_ATTACH_TOOL_ACTIVITY_ENABLED", "0")
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(t.send(_tool_heartbeat()))
    assert console.prints == []


def test_a_failed_tool_call_says_so() -> None:
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(t.send(_tool_heartbeat(status="error", result_preview="boom")))
    blob = "\n".join(console.prints)
    assert "⏺" in blob and "backend/x.py" in blob
    assert "✗" in blob or "error" in blob or "failed" in blob, (
        "a failed call must be visibly a failure, not a success block")


def test_model_controlled_text_cannot_open_a_markup_tag(monkeypatch) -> None:
    """The composer escapes model content; the minimal fallback must too."""
    from backend.core.ouroboros.battle_test import tool_render_view as trv
    monkeypatch.setattr(trv, "compose_if_enabled", lambda *a, **k: None)
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    asyncio.run(t.send(_tool_heartbeat(tool_args_summary="[bold red]x[/]")))
    blob = "\n".join(console.prints)
    assert "\\[bold red]" in blob


def test_the_module_still_imports_no_rich() -> None:
    """The authority invariant this module carries, re-pinned here because
    the fallback renderer needed an escape and must not have reached for
    rich to get one."""
    tree = ast.parse(Path(cst.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("rich")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("rich") for a in node.names)


# ---------------------------------------------------------------------------
# Cut 1, harness side — the transport is retained and wired
# ---------------------------------------------------------------------------


def test_the_harness_retains_the_chosen_transport() -> None:
    """A local that goes out of scope cannot be wired when the bridge is
    armed later. This is how the default ran with no cockpit path."""
    src = _HARNESS.read_text(encoding="utf-8")
    assert "self._per_op_transport = _chosen_transport" in src


def test_the_harness_wires_the_transport_mirror_where_it_wires_serpentflow() -> None:
    src = _HARNESS.read_text(encoding="utf-8")
    wire = src.index("_pot.markup_mirror = bridge.publish_markup")
    sf = src.index("sf.markup_mirror = bridge.publish_markup")
    assert abs(wire - sf) < 1200, "same arming seam, one idiom"


# ---------------------------------------------------------------------------
# Cut 3 — the cockpit puts autonomous work in the transcript
# ---------------------------------------------------------------------------


def _sink_with(monkeypatch, *, canvas_present: bool):
    """Build `_markup_sink` exactly as run_attach does, with the two
    destinations observable and the canvas presence controlled."""
    from backend.core.ouroboros.cli import ov as O
    from backend.core.ouroboros.battle_test import bipartite_layout as bl

    rendered, decked = [], []
    monkeypatch.setattr(O, "_render_markup_frame",
                        lambda text, console=None: rendered.append(text))
    monkeypatch.setattr(bl, "get_active_canvas",
                        lambda: object() if canvas_present else None)
    ui = SimpleNamespace(on_ambient=lambda t, **k: decked.append(t),
                         turn_spinner=None)
    src = _OV.read_text(encoding="utf-8")
    # Extract the sink body verbatim so the test exercises the shipped
    # routing rather than a re-typed copy of it.
    start = src.index("        def _markup_sink(text: str, addressed: bool = False) -> None:")
    end = src.index("        client = CockpitAttachClient(", start)
    body = "\n".join(l[8:] for l in src[start:end].splitlines())
    ns = {"ui": ui, "console": None, "_render_markup_frame": O._render_markup_frame,
          "_ambient_transcript_enabled": O._ambient_transcript_enabled}
    exec(body, ns)  # noqa: S102 — executing the repo's own source under test
    return ns["_markup_sink"], rendered, decked


def test_autonomous_work_lands_in_the_transcript_and_the_deck(monkeypatch) -> None:
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink("⏺ TodoScanner(verify_gate.py)  queued", False)
    assert rendered == ["⏺ TodoScanner(verify_gate.py)  queued"]
    assert decked == ["⏺ TodoScanner(verify_gate.py)  queued"]


def test_addressed_output_is_unchanged(monkeypatch) -> None:
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink("posture: EXPLORE", True)
    assert rendered == ["posture: EXPLORE"] and decked == []


def test_the_agora_stays_off_the_transcript(monkeypatch) -> None:
    """SOCIAL is the deck's compaction case; it must not flood the canvas."""
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink("🐍 @the-pit · celebration ⎿ stitched two grafts", False)
    assert rendered == [] and len(decked) == 1


def test_operational_alerts_still_reach_the_deck(monkeypatch) -> None:
    """The contract `test_ignition_skeleton` pins: a failover pins on the
    deck. It now also appears in the transcript; it must not vanish there."""
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink("DW provider failover", False)
    assert decked == ["DW provider failover"]
    assert rendered == ["DW provider failover"]


def test_without_a_canvas_the_legacy_pump_is_not_double_printed(monkeypatch) -> None:
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=False)
    sink("⏺ GithubIssue(backend/)  queued", False)
    assert rendered == [] and len(decked) == 1


def test_the_master_switch_restores_deck_only_routing(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_COCKPIT_AMBIENT_TRANSCRIPT", "0")
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink("⏺ TestCoverage(x.py)  queued", False)
    assert rendered == [] and len(decked) == 1


@pytest.mark.parametrize("line", [
    "   [cyan]⏺ Read[/cyan]([cyan underline]backend/x.py[/cyan underline])  [dim]11ms[/dim]",
    "   [cyan]⏺ Search[/cyan]([cyan underline]\"_check_api_keys_or_die\"[/cyan underline])  [dim]281ms[/dim]",
    "   [dim]  ⎿  212 lines read[/dim]",
    "   [cyan]⏺ Read[/cyan]([cyan underline]x.py[/cyan underline])  [dim]12ms[/dim]  [red]✗[/red]",
])
def test_a_composed_tool_block_lands_in_the_transcript(monkeypatch, line) -> None:
    """The exact styled shapes the daemon now broadcasts for a tool call
    (captured off the socket 2026-09-06). They must classify as WORK, not
    agora, and take the same route the intent lines take -- the canvas --
    or the daemon-side fix would be invisible at the one place it matters."""
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    sink(line, False)
    assert rendered == [line]
    assert decked == [line]


def test_a_routing_fault_never_breaks_attach(monkeypatch) -> None:
    from backend.core.ouroboros.battle_test import ambient_deck as ad
    sink, rendered, decked = _sink_with(monkeypatch, canvas_present=True)
    monkeypatch.setattr(ad, "classify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    sink("anything", False)          # must not raise
    assert len(decked) == 1
