"""The default transport narrates, and its lines obey the design language.

Measured 2026-09-06 on a live cockpit: tool blocks flowed, and nothing
else did. The organism never said WHY it opened an op, never said why it
ran a tool, and 116 no-op refusals closed without a line. Four cuts, all
on the default (``JARVIS_RENDER_MODE=CLAUDE``) surface:

1. The 💭 intent and the 🗣 preamble are produced by SerpentFlow methods
   (``op_started`` / ``op_tool_start``) that the default transport never
   calls. The transport now uses the same producers itself.
2. Every gate speaks through ``rt_gate.gate_completion``, whose tiers were
   Claude and DoubleWord only. On a host whose only lane is the local
   model, both fail at second zero and the gate raises — the voice could
   not speak here at all. A local tier now sits last in every order.
3. The local client forced the CANDIDATE JSON grammar on every completion.
   A gate asking for a sentence got an object. The format is now the
   call's decision.
4. The ledger's terminal states (no-op, failure, block, rollback) reached
   the SSE broker and never the comm wire. The one terminal chokepoint now
   emits the DECISION the transport already renders.

And the aesthetics: the transport rendered ``· Sensor(op7c17) goal...`` —
the middle dot is the language's SEPARATOR, the six-char hash is a
truncated id the operator cannot ``/expand``, and the cut was mid-word.
Its bullets ``● ◌ ⏭`` are outside the six-glyph ration. The lines now
draw from ``theme.mark`` like every other surface.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.ouroboros.battle_test import narrative_channel as nc
from backend.core.ouroboros.governance import claude_style_transport as cst
from backend.core.ouroboros.ui import theme


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "JARVIS_RENDER_MODE", "JARVIS_NARRATIVE_DENSITY",
        "JARVIS_NARRATIVE_INTENT_ENABLED", "JARVIS_TOOL_PREAMBLE_FALLBACK_ENABLED",
        "JARVIS_LOCAL_PRIME_ENABLED", "JARVIS_CLAUDE_STYLE_LINE_CHARS",
        "JARVIS_CLAUDE_STYLE_DETAIL_LINES",
    ):
        monkeypatch.delenv(name, raising=False)
    from backend.core.ouroboros.governance import flag_registry as fr
    fr.reset_default_registry()
    nc.reset_default_channel_for_tests()
    yield
    nc.reset_default_channel_for_tests()
    fr.reset_default_registry()


class _Console:
    def __init__(self) -> None:
        self.prints: List[str] = []

    def print(self, text: str, **kw: Any) -> None:
        self.prints.append(text)


def _msg(kind: str, op_id: str, payload: dict) -> Any:
    return SimpleNamespace(
        msg_type=SimpleNamespace(value=kind), op_id=op_id, payload=payload,
    )


def _transport():
    console = _Console()
    t = cst.ClaudeStyleTransport(console=console)
    mirrored: List[str] = []
    t.markup_mirror = mirrored.append
    return t, console, mirrored


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# The channel signals a commit
# ---------------------------------------------------------------------------


def test_commit_notifies_listeners_and_deduplicates_by_identity():
    ch = nc.NarrativeChannel(capacity=8)
    seen: list = []

    def on(frame):
        seen.append(frame)

    assert ch.add_commit_listener(on) is True
    assert ch.add_commit_listener(on) is False
    ch.emit_complete(op_id="op", phase="P", kind=nc.NarrativeKind.INTENT, prose="why")
    assert [f.prose for f in seen] == ["why"]
    assert ch.remove_commit_listener(on) is True
    ch.emit_complete(op_id="op", phase="Q", kind=nc.NarrativeKind.INTENT, prose="more")
    assert len(seen) == 1


def test_a_raising_listener_never_silences_the_next():
    ch = nc.NarrativeChannel(capacity=8)
    order: list = []

    def bad(_f):
        order.append("bad")
        raise RuntimeError("listener fault")

    def good(_f):
        order.append("good")

    ch.add_commit_listener(bad)
    ch.add_commit_listener(good)
    frame = ch.emit_complete(op_id="op", phase="P", kind=nc.NarrativeKind.PLAN_PROSE, prose="p")
    assert frame is not None and order == ["bad", "good"]


def test_a_discarded_frame_does_not_notify():
    ch = nc.NarrativeChannel(capacity=8)
    seen: list = []
    ch.add_commit_listener(seen.append)
    ch.start_frame(op_id="op", phase="P", kind=nc.NarrativeKind.INTENT)
    ch.discard(op_id="op", phase="P", kind=nc.NarrativeKind.INTENT)
    assert seen == []


# ---------------------------------------------------------------------------
# The renderer has a sink seam
# ---------------------------------------------------------------------------


def test_render_to_printer_uses_the_sink_and_keeps_the_voice_glyph():
    from backend.core.ouroboros.battle_test.narrative_renderer import render_to_printer
    ch = nc.NarrativeChannel(capacity=4)
    frame = ch.emit_complete(
        op_id="op", phase="P", kind=nc.NarrativeKind.TOOL_PREAMBLE,
        prose="Reading x.py to see the callers", provider="synthetic",
    )
    out: list = []
    assert render_to_printer(frame, lambda m, **k: out.append(m), op_active=False) is True
    assert len(out) == 1 and "Reading x.py" in out[0] and "🗣" in out[0]


# ---------------------------------------------------------------------------
# The transport narrates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_line_then_the_organism_says_why(monkeypatch):
    from backend.core.ouroboros.governance import intent_prompter as ip
    calls: list = []

    async def fake_request(req, *, phase="OP_STARTED", **_kw):
        calls.append((req.op_id, req.goal, phase, req.target_files))
        nc.get_default_channel().emit_complete(
            op_id=req.op_id, phase=phase, kind=nc.NarrativeKind.INTENT,
            prose="Fixing the flaky assertion so CI stops lying", provider="local",
        )
        return SimpleNamespace(succeeded=True, prose="x")

    monkeypatch.setattr(ip, "request_intent_and_emit", fake_request)
    monkeypatch.setattr(ip, "is_master_flag_enabled", lambda: True)
    t, console, mirrored = _transport()
    await t.send(_msg("INTENT", "op-1", {
        "goal": "Fix flaky test in tests/x.py", "outcome_source": "TestFailure",
        "target_files": ["tests/x.py"],
    }))
    for _ in range(4):
        await asyncio.sleep(0)             # let the narration task run
    assert calls == [("op-1", "Fix flaky test in tests/x.py", "OP_STARTED", ("tests/x.py",))]
    lead, voice = console.prints[0], console.prints[-1]
    assert theme.mark("action") in lead and "TestFailure" in lead
    assert "💭" in voice and "Fixing the flaky assertion" in voice
    assert mirrored == console.prints      # every line reached the cockpit


@pytest.mark.asyncio
async def test_the_intent_request_never_blocks_the_render(monkeypatch):
    from backend.core.ouroboros.governance import intent_prompter as ip
    gate = asyncio.Event()

    async def slow(req, **_kw):
        await gate.wait()
        return SimpleNamespace(succeeded=False, prose="")

    monkeypatch.setattr(ip, "request_intent_and_emit", slow)
    monkeypatch.setattr(ip, "is_master_flag_enabled", lambda: True)
    t, console, _ = _transport()
    await asyncio.wait_for(
        t.send(_msg("INTENT", "op-1", {"goal": "g", "outcome_source": "Operation"})),
        timeout=1.0,
    )
    assert len(console.prints) == 1 and len(t._narration_tasks) == 1
    t.shutdown()
    await asyncio.sleep(0)
    assert not t._narration_tasks or all(x.cancelled() or x.done() for x in t._narration_tasks)


@pytest.mark.asyncio
async def test_a_tool_start_narrates_once_per_round_and_the_completion_draws():
    t, console, _ = _transport()
    await t.send(_msg("INTENT", "op-1", {"goal": "x", "outcome_source": "TestFailure"}))
    start = {
        "tool_name": "read_file", "tool_args_summary": "backend/x.py",
        "round_index": 2, "tool_starting": True, "status": "start",
        "preamble": "Reading x.py to find the failing assertion",
    }
    await t.send(_msg("HEARTBEAT", "op-1", dict(start)))
    await t.send(_msg("HEARTBEAT", "op-1", dict(start)))     # parallel batch, same round
    voices = [p for p in console.prints if "🗣" in p]
    assert len(voices) == 1 and "Reading x.py to find" in voices[0]
    frames = nc.get_default_channel().find_by_kind(nc.NarrativeKind.TOOL_PREAMBLE)
    assert len(frames) == 1 and frames[0].provider == "model"
    before = len(console.prints)
    done = dict(start, tool_starting=False, status="success",
                duration_ms=11.0, result_preview="42 lines")
    await t.send(_msg("HEARTBEAT", "op-1", done))
    assert any(theme.mark("action") in p and "x.py" in p for p in console.prints[before:])


@pytest.mark.asyncio
async def test_without_a_model_preamble_the_template_speaks_and_says_so():
    t, console, _ = _transport()
    await t.send(_msg("INTENT", "op-1", {"goal": "x"}))
    await t.send(_msg("HEARTBEAT", "op-1", {
        "tool_name": "read_file", "tool_args_summary": "a/b.py", "round_index": 0,
        "tool_starting": True, "status": "start", "preamble": "",
    }))
    frames = nc.get_default_channel().find_by_kind(nc.NarrativeKind.TOOL_PREAMBLE)
    assert len(frames) == 1 and frames[0].provider == "synthetic" and frames[0].prose
    assert any("🗣" in p for p in console.prints)


@pytest.mark.asyncio
async def test_narrate_off_silences_the_voice_but_keeps_the_record(monkeypatch):
    from backend.core.ouroboros.ui.narrative_density import ensure_discovered
    ensure_discovered()
    monkeypatch.setenv("JARVIS_NARRATIVE_DENSITY", "off")
    t, console, _ = _transport()
    await t.send(_msg("INTENT", "op-1", {"goal": "x"}))
    base = {"tool_name": "read_file", "tool_args_summary": "a.py",
            "tool_starting": True, "status": "start"}
    await t.send(_msg("HEARTBEAT", "op-1", dict(base, round_index=0, preamble="")))
    assert not nc.get_default_channel().find_by_kind(nc.NarrativeKind.TOOL_PREAMBLE)
    await t.send(_msg("HEARTBEAT", "op-1", dict(base, round_index=1, preamble="the model's own words")))
    frames = nc.get_default_channel().find_by_kind(nc.NarrativeKind.TOOL_PREAMBLE)
    assert len(frames) == 1                          # the ring is not the dial's to edit
    assert not any("🗣" in p for p in console.prints)  # but nothing is shown at OFF


@pytest.mark.asyncio
async def test_every_committed_kind_reaches_the_transport():
    t, console, mirrored = _transport()
    ch = nc.get_default_channel()
    ch.emit_complete(op_id="op", phase="PLAN", kind=nc.NarrativeKind.PLAN_PROSE,
                     prose="Two changes: the fixture, then the assertion.")
    ch.emit_complete(op_id="op", phase="L2", kind=nc.NarrativeKind.L2_REPAIR_PROSE,
                     prose="Retrying with the import restored.")
    blob = "\n".join(console.prints)
    assert "Two changes" in blob and "import restored" in blob
    assert mirrored == console.prints


@pytest.mark.asyncio
async def test_a_wrapped_paragraph_lands_row_by_row(monkeypatch):
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_LINE_CHARS", "40")
    t, console, _ = _transport()
    nc.get_default_channel().emit_complete(
        op_id="op", phase="P", kind=nc.NarrativeKind.INTENT,
        prose="a sentence long enough that the renderer must wrap it onto several rows of the canvas",
    )
    assert len(console.prints) >= 2 and all("\n" not in p for p in console.prints)


@pytest.mark.asyncio
async def test_shutdown_releases_the_subscription():
    t, console, _ = _transport()
    t.shutdown()
    nc.get_default_channel().emit_complete(
        op_id="op", phase="P", kind=nc.NarrativeKind.INTENT, prose="late words",
    )
    assert console.prints == []


# ---------------------------------------------------------------------------
# The lines obey the design language
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_op_lines_carry_no_id_and_no_stray_bullets():
    t, console, _ = _transport()
    await t.send(_msg("INTENT", "op-7c17aa-x", {
        "goal": "Wave 3 graduation", "outcome_source": "TestFailure",
        "risk_tier": "NOTIFY_APPLY",
    }))
    await t.send(_msg("DECISION", "op-7c17aa-x", {
        "outcome": "noop", "reason_code": "noop",
        "reason": "The target already has comprehensive coverage for this path; nothing to add.",
    }))
    blob = "\n".join(console.prints)
    assert "op7c17" not in blob
    for stray in ("●", "◌", "⏭", "│"):
        assert stray not in blob, stray
    assert console.prints[0].startswith(f"[{cst._SEM['neural']}]{theme.mark('action')}")
    assert "notify apply" in console.prints[0] and "NOTIFY_APPLY" not in blob
    assert "no change" in console.prints[1] and theme.mark("detail") in console.prints[1]
    assert any("comprehensive coverage" in p for p in console.prints[2:])


@pytest.mark.asyncio
async def test_the_models_reason_is_bounded_in_height(monkeypatch):
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_DETAIL_LINES", "2")
    monkeypatch.setenv("JARVIS_CLAUDE_STYLE_LINE_CHARS", "30")
    t, console, _ = _transport()
    await t.send(_msg("INTENT", "op-1", {"goal": "g"}))
    await t.send(_msg("DECISION", "op-1", {
        "outcome": "noop", "reason_code": "noop", "reason": "word " * 60,
    }))
    detail = console.prints[2:]
    assert len(detail) == 2 and detail[-1].rstrip("[/dim italic]").rstrip().endswith(
        theme.mark("ellipsis") + f"[/{cst._SEM['dim']} italic]",
    ) or detail[-1].count(theme.mark("ellipsis")) == 1


@pytest.mark.asyncio
async def test_an_op_held_before_it_announced_itself_still_renders():
    """Sixteen ops blocked at the gate never emitted INTENT; their terminal
    DECISION arrived with no state and vanished as a boot orphan."""
    t, console, _ = _transport()
    await t.send(_msg("DECISION", "op-held", {
        "outcome": "escalated", "reason_code": "touches_kernel",
        "target_files": ["unified_supervisor.py"], "terminal_state": "blocked",
    }))
    assert len(console.prints) == 1
    line = console.prints[0]
    assert "held for review" in line and "touches kernel" in line
    assert "unified_supervisor.py" in line


@pytest.mark.asyncio
async def test_a_boot_orphan_decision_is_still_suppressed():
    t, console, _ = _transport()
    await t.send(_msg("DECISION", "op-ghost", {"outcome": "failed"}))
    await t.send(_msg("DECISION", "op-ghost2", {
        "outcome": "failed", "reason_code": "boot_recovery_orphan",
        "terminal_state": "failed",
    }))
    assert console.prints == []


def test_summaries_cut_at_a_word_with_the_theme_ellipsis():
    goal = "graduate the wave three sensors after the audit lands"
    s = cst._clip_words(goal, 30)
    ell = theme.mark("ellipsis")
    assert s.endswith(ell) and len(s) <= 30
    body = s[: -len(ell)]
    assert goal.startswith(body) and goal[len(body)] == " "   # cut AT a word
    assert cst._clip_words("short", 30) == "short"


def test_reason_codes_read_as_words():
    assert cst._humanise("background_dw_blocked_by_topology") == "background dw blocked by topology"
    assert cst._humanise("") == ""


def test_the_status_marks_come_from_the_design_ration():
    for m in cst.OpStatusGlyph:
        assert theme.mark(m.mark, unicode=True) and theme.mark(m.mark, unicode=False), m
        assert m.role in cst._SEM, m
    assert cst.OpStatusGlyph.ACTIVE.glyph == theme.mark("action")
    assert cst.OpStatusGlyph.DONE.glyph == theme.mark("check")


def test_the_second_glyph_table_is_gone():
    from backend.core.ouroboros.battle_test import presentation_restraint as pr
    assert pr.glyphs() == {"action": theme.mark("action"), "result": theme.mark("detail")}
    assert not hasattr(pr, "_GLYPHS_UTF8")


def test_tool_chrome_and_composer_degrade_with_the_theme(monkeypatch):
    monkeypatch.setattr(theme, "supports_unicode", lambda env=None: False)
    from backend.core.ouroboros.battle_test.serpent_flow import _tool_chrome_line
    assert _tool_chrome_line("read_file", "x.py").startswith("* ")
    from backend.core.ouroboros.battle_test.tool_render_view import (
        compose_if_enabled, store_for_view,
    )
    out = compose_if_enabled(
        "read_file", "x.py", "ok", status="success", duration_ms=1.0,
        op_id="op", round_index=0, palette=cst._SEM, store=store_for_view(),
    )
    assert out is not None and "* Read" in out.header_markup and "⏺" not in out.header_markup


# ---------------------------------------------------------------------------
# The gate can speak on a local-only host
# ---------------------------------------------------------------------------


class _FakeLocalClient:
    made: list = []

    def __init__(self, cfg, *a, **k):
        self.cfg = cfg
        self.closed = False
        self.calls: list = []
        _FakeLocalClient.made.append(self)

    async def generate(self, prompt, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content="local says why")

    async def aclose(self):
        self.closed = True


def _dead_cloud(monkeypatch):
    import backend.core.ouroboros.claude_fallback as cf
    monkeypatch.setattr(cf, "claude_inference",
                        AsyncMock(side_effect=RuntimeError("no key")))


@pytest.fixture
def local_lane(monkeypatch):
    import backend.core.ouroboros.governance.local_inference_director as lid
    monkeypatch.setattr(lid, "LocalPrimeClient", _FakeLocalClient)
    _FakeLocalClient.made.clear()
    yield lid
    _FakeLocalClient.made.clear()


def test_local_tier_speaks_when_every_cloud_tier_is_dead(monkeypatch, local_lane):
    import backend.core.ouroboros.governance.rt_gate as rtg
    _dead_cloud(monkeypatch)
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    out = _run(rtg.gate_completion("why?", caller_id="t", system_prompt="be brief"))
    assert out == "local says why"
    client = _FakeLocalClient.made[-1]
    assert client.closed                                   # session released
    assert client.calls[0]["response_format"] is None      # PROSE, not the candidate ladder
    assert client.calls[0]["system_prompt"] == "be brief"


def test_local_tier_is_absent_when_its_lane_is_off(monkeypatch, local_lane):
    import backend.core.ouroboros.governance.rt_gate as rtg
    _dead_cloud(monkeypatch)
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "false")
    with pytest.raises(rtg.GateProviderExhaustedError) as ei:
        _run(rtg.gate_completion("p", caller_id="t"))
    assert "local" not in str(ei.value) and not _FakeLocalClient.made


def test_local_tier_is_last_unless_preferred(monkeypatch, local_lane):
    import backend.core.ouroboros.governance.rt_gate as rtg
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value="claude says")
    assert _run(rtg.gate_completion("p", caller_id="t", claude_provider=claude)) == "claude says"
    assert not _FakeLocalClient.made                       # a paid host is byte-identical
    assert _run(rtg.gate_completion(
        "p", caller_id="t", claude_provider=claude, prefer="local")) == "local says why"
    claude.prompt_only.assert_called_once()


def test_a_failing_local_tier_still_closes_its_session(monkeypatch, local_lane):
    import backend.core.ouroboros.governance.rt_gate as rtg

    class _Boom(_FakeLocalClient):
        async def generate(self, *a, **k):
            raise RuntimeError("engine down")

    monkeypatch.setattr(local_lane, "LocalPrimeClient", _Boom)
    _dead_cloud(monkeypatch)
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    with pytest.raises(rtg.GateProviderExhaustedError) as ei:
        _run(rtg.gate_completion("p", caller_id="t"))
    assert "local" in str(ei.value) and _FakeLocalClient.made[-1].closed


# ---------------------------------------------------------------------------
# The local client's format is the call's decision
# ---------------------------------------------------------------------------


def test_response_format_is_a_per_call_decision():
    import backend.core.ouroboros.governance.local_inference_director as lid
    for fn in (lid.LocalPrimeClient.complete, lid.LocalPrimeClient.complete_guarded,
               lid.LocalPrimeClient.generate):
        assert inspect.signature(fn).parameters["response_format"].default is lid.RESPONSE_FORMAT_LADDER
    body = inspect.getsource(lid.LocalPrimeClient.complete)
    assert "if response_format is RESPONSE_FORMAT_LADDER:" in body
    assert "_apply_response_format(body, self._cfg)" in body


def test_an_explicit_shape_is_spelled_for_the_transport_and_none_means_prose():
    import backend.core.ouroboros.governance.local_inference_director as lid
    cfg = lid.LocalConfig.from_env()
    body: dict = {}
    assert lid._apply_explicit_response_format(body, cfg, {"type": "json_object"}) == "json_object"
    assert body.get("format") == "json" or body.get("response_format") == {"type": "json_object"}
    body = {}
    assert lid._apply_explicit_response_format(body, cfg, None) == "none" and body == {}


# ---------------------------------------------------------------------------
# The ledger's terminal seam speaks
# ---------------------------------------------------------------------------


class _Comm:
    def __init__(self) -> None:
        self.decisions: list = []

    async def emit_decision(self, **kw):
        self.decisions.append(kw)


def _orch(comm):
    from backend.core.ouroboros.governance.orchestrator import GovernedOrchestrator
    o = GovernedOrchestrator.__new__(GovernedOrchestrator)
    o._stack = SimpleNamespace(comm=comm)
    return o


def _ctx(**kw):
    base = dict(op_id="op-1", terminal_reason_code="", generation=None,
                target_files=("a.py",))
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_noop_termination_is_spoken_with_the_models_reason():
    comm = _Comm()
    ctx = _ctx(terminal_reason_code="noop",
               generation=SimpleNamespace(is_noop=True, noop_reason="already covered"))
    _run(_orch(comm)._emit_terminal_decision(ctx, SimpleNamespace(value="applied"), {}))
    d = comm.decisions[0]
    assert d["outcome"] == "noop" and d["reason"] == "already covered"
    assert d["reason_code"] == "noop" and d["target_files"] == ["a.py"]
    assert d["op_id"] == "op-1" and d["terminal_state"] == "applied"


def test_a_real_apply_is_left_to_the_change_engine():
    comm = _Comm()
    _run(_orch(comm)._emit_terminal_decision(_ctx(), SimpleNamespace(value="applied"), {}))
    assert comm.decisions == []


@pytest.mark.parametrize("state,data,outcome,code", [
    ("failed", {"reason": "no_candidates_returned"}, "failed", "no_candidates_returned"),
    ("blocked", {}, "escalated", "blocked"),
    ("rolled_back", {}, "failed", "rolled_back"),
])
def test_the_other_terminal_states_map_to_outcomes(state, data, outcome, code):
    comm = _Comm()
    _run(_orch(comm)._emit_terminal_decision(_ctx(), SimpleNamespace(value=state), data))
    d = comm.decisions[0]
    assert d["outcome"] == outcome and d["reason_code"] == code


def test_a_cosmetic_noop_code_is_a_noop_even_without_the_generation_flag():
    comm = _Comm()
    _run(_orch(comm)._emit_terminal_decision(
        _ctx(terminal_reason_code="no_op_cosmetic"), SimpleNamespace(value="applied"),
        {"detail": "only docstrings changed"},
    ))
    assert comm.decisions[0]["outcome"] == "noop"
    assert comm.decisions[0]["reason"] == "only docstrings changed"


def test_a_comm_fault_never_reaches_the_ledger():
    class _Broken:
        async def emit_decision(self, **kw):
            raise RuntimeError("wire down")

    _run(_orch(_Broken())._emit_terminal_decision(
        _ctx(terminal_reason_code="noop"), SimpleNamespace(value="applied"), {}))


def test_the_terminal_seam_is_wired_once_at_the_chokepoint():
    from backend.core.ouroboros.governance import orchestrator as om
    src = inspect.getsource(om.GovernedOrchestrator._record_ledger)
    assert src.count("await self._emit_terminal_decision(ctx, state, data)") == 1
    assert src.index("_slice12q_record_terminal(ctx, state, data)") < src.index(
        "await self._emit_terminal_decision(ctx, state, data)")
