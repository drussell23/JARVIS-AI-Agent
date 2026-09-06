"""A one-line op recap on completion — Claude Code's ``✻ Crunched for 2m 14s
· 3 tools used · done 11:40 PM`` — synthesised from the op's execution ledger,
and a graceful ``✻ Aborted after Xm Ys`` for a cancelled op.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from backend.core.ouroboros.governance import op_recap as r
from backend.core.ouroboros.governance import claude_style_transport as cst
from backend.core.ouroboros.ui import theme


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("JARVIS_OP_RECAP_ENABLED", "JARVIS_RECAP_VERBS"):
        monkeypatch.delenv(name, raising=False)
    from backend.core.ouroboros.governance import flag_registry as fr
    from backend.core.ouroboros.governance import intent_prompter as ip
    from backend.core.ouroboros.battle_test import narrative_channel as nc
    fr.reset_default_registry()
    nc.reset_default_channel_for_tests()
    monkeypatch.setattr(ip, "is_master_flag_enabled", lambda: False)
    yield
    nc.reset_default_channel_for_tests()
    fr.reset_default_registry()


# ---------------------------------------------------------------------------
# compose_recap — the line
# ---------------------------------------------------------------------------


def test_a_completed_recap_reads_like_claude_code():
    line = r.compose_recap(elapsed="2m 14s", verb="Crunched", tools=3,
                           tokens=210, done_at="11:40 PM")
    assert line == "✻ Crunched for 2m 14s · 3 tools used · ↑ 210 tokens · done 11:40 PM"


def test_zero_tools_and_zero_tokens_drop_their_segments():
    line = r.compose_recap(elapsed="8s", verb="Landed", tools=0, tokens=0,
                           done_at="9:01 AM")
    assert line == "✻ Landed for 8s · done 9:01 AM"
    assert "0 tool" not in line and "tokens" not in line


def test_one_tool_is_singular():
    assert "1 tool used" in r.compose_recap(elapsed="1s", verb="X", tools=1, done_at="")


def test_an_aborted_op_says_so_with_no_fabricated_counts():
    line = r.compose_recap(elapsed="2m 14s", verb="Crunched", tools=3, tokens=99,
                           aborted=True)
    assert line == "✻ Aborted after 2m 14s"
    assert "tool" not in line and "tokens" not in line


def test_a_failed_op_carries_the_duration_and_what_ran():
    line = r.compose_recap(elapsed="46s", failed=True, tools=2, done_at="3:00 PM")
    assert line == "✻ Failed after 46s · 2 tools used · done 3:00 PM"


def test_big_token_counts_use_the_k_suffix():
    assert "↑ 8.2k tokens" in r.compose_recap(elapsed="1m 0s", verb="X", tokens=8200, done_at="")


def test_the_mark_is_the_theme_glyph():
    assert r.compose_recap(elapsed="1s", verb="X", done_at="").startswith(theme.mark("recap"))


def test_compose_never_raises_on_junk():
    assert isinstance(r.compose_recap(elapsed=None, verb=None, tools="x", tokens=None), str)


def test_the_clock_is_twelve_hour_and_portable():
    import datetime
    assert r._clock(datetime.datetime(2026, 9, 6, 23, 40)) == "11:40 PM"
    assert r._clock(datetime.datetime(2026, 9, 6, 0, 5)) == "12:05 AM"
    assert r._clock(datetime.datetime(2026, 9, 6, 13, 0)) == "1:00 PM"


# ---------------------------------------------------------------------------
# counts pulled from the generation ledger
# ---------------------------------------------------------------------------


def test_tool_count_prefers_the_execution_record_then_edits():
    assert r.tool_count(SimpleNamespace(tool_execution_records=(1, 2, 3))) == 3
    assert r.tool_count(SimpleNamespace(tool_execution_records=(),
                                        venom_edit_history=({}, {}))) == 2
    assert r.tool_count(None) == 0


def test_output_tokens_reads_the_generation():
    assert r.output_tokens(SimpleNamespace(total_output_tokens=512)) == 512
    assert r.output_tokens(None) == 0


# ---------------------------------------------------------------------------
# the transport draws it beneath the outcome
# ---------------------------------------------------------------------------


class _Console:
    def __init__(self) -> None:
        self.prints: List[str] = []

    def print(self, text: str, **kw: Any) -> None:
        self.prints.append(text)


def _msg(kind: str, op_id: str, payload: dict) -> Any:
    return SimpleNamespace(msg_type=SimpleNamespace(value=kind), op_id=op_id, payload=payload)


async def _intent(t, op_id="op-1"):
    await t.send(_msg("INTENT", op_id, {"goal": "g", "outcome_source": "TestFailure"}))


@pytest.mark.asyncio
async def test_a_completed_decision_draws_a_recap_line():
    t = cst.ClaudeStyleTransport(console=_Console())
    await _intent(t)
    t._console.prints.clear()
    await t.send(_msg("DECISION", "op-1", {
        "outcome": "completed", "files_changed": ["x.py"],
        "tools_used": 3, "tokens": 210,
    }))
    blob = "\n".join(t._console.prints)
    assert theme.mark("recap") in blob and "3 tools used" in blob and "done" in blob
    assert len(t._console.prints) >= 2                      # outcome line + recap line


@pytest.mark.asyncio
async def test_a_cancelled_decision_draws_an_aborted_recap():
    t = cst.ClaudeStyleTransport(console=_Console())
    await _intent(t)
    t._console.prints.clear()
    await t.send(_msg("DECISION", "op-1", {"outcome": "cancelled", "tools_used": 2}))
    blob = "\n".join(t._console.prints)
    assert f"{theme.mark('recap')} Aborted after" in blob
    assert "tools used" not in blob                         # no fabricated counts


@pytest.mark.asyncio
async def test_the_recap_can_be_silenced(monkeypatch):
    monkeypatch.setenv("JARVIS_OP_RECAP_ENABLED", "false")
    t = cst.ClaudeStyleTransport(console=_Console())
    await _intent(t)
    t._console.prints.clear()
    await t.send(_msg("DECISION", "op-1", {"outcome": "completed", "tools_used": 3}))
    assert not any(theme.mark("recap") in p for p in t._console.prints)


def test_the_terminal_seam_supplies_the_recap_counts():
    import inspect
    from backend.core.ouroboros.governance import orchestrator as om
    src = inspect.getsource(om.GovernedOrchestrator._emit_terminal_decision)
    assert "tools_used=_tools" in src and "tokens=_tokens" in src
    assert "tool_count" in src and "output_tokens" in src


# ---------------------------------------------------------------------------
# Ledger-seam binding — the generation (with its tool records) reaches the
# terminal seam on EVERY path, not only a full apply.
# ---------------------------------------------------------------------------


def test_the_runner_binds_the_generation_to_ctx_before_the_terminal_ledger():
    """Root cause of the zero tool counts: terminal paths recorded the ledger
    with ctx.generation unbound. The runner now binds it the moment the
    generation is finalised, so dataclasses.replace carries its
    tool_execution_records through every advance to the recap seam."""
    import inspect
    from backend.core.ouroboros.governance.phase_runners import generate_runner as gr
    src = inspect.getsource(gr)
    assert 'object.__setattr__(ctx, "generation", generation)' in src
    # Bound right after the generation is finalised, and BEFORE the
    # post-generation noop break that follows it — so the real noop terminal
    # (the model exploring then declining) carries the records. (The FIRST
    # noop check in the file is the synthetic wiring-validation fixture path,
    # which legitimately ran zero tools.)
    i = src.index('object.__setattr__(ctx, "generation", generation)')
    j = src.index("if generation is not None and generation.is_noop:", i)
    assert 0 < i < j


def test_a_noop_generation_still_carries_its_tool_records_to_the_recap():
    """A noop op that explored (ran tools) must recap the exploration effort,
    not report zero — the desync this fixes."""
    from types import SimpleNamespace
    gen = SimpleNamespace(is_noop=True, noop_reason="already covered",
                          tool_execution_records=(1, 2, 3, 4),
                          total_output_tokens=147, venom_edit_history=())
    assert r.tool_count(gen) == 4 and r.output_tokens(gen) == 147
    line = r.compose_recap(elapsed="1m 3s", verb="Reviewed",
                           tools=r.tool_count(gen), tokens=r.output_tokens(gen),
                           done_at="2:15 PM")
    assert "4 tools used" in line and "↑ 147 tokens" in line
