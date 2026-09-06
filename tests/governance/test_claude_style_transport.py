"""ClaudeStyleTransport regression suite.

Pins the per-op single-line-per-message rendering substrate that
matches Claude Code's tool-call visual idiom.

Strict directives validated:

  * Closed-taxonomy RenderMode (CLAUDE/SERPENT) AST-pinned
  * Closed-taxonomy OpStatusGlyph (6 bullets) AST-pinned
  * No authority imports, no top-level Rich imports
  * Defensive everywhere — bad messages don't crash transport
  * Boot-recovery suppression — N orphans collapse to one summary
  * Render mode default CLAUDE — operator gets the cleaner look
    out of the box; hot-revert via SERPENT preserved

Covers:

  §A   RenderMode + OpStatusGlyph closed taxonomies
  §B   resolve_render_mode default CLAUDE + env override
  §C   Per-op INTENT renders single line with bullet
  §D   DECISION outcomes — completed / failed / noop / notify_apply /
       escalated each render distinct line
  §E   POSTMORTEM renders failure line
  §F   Boot-recovery suppression — N orphans → 1 summary
  §G   HEARTBEAT silenced by default; opt-in via flag
  §H   Defensive: bad msg shape doesn't crash send()
  §I   Sensor inference from goal text
  §J   AST pins clean + tampering caught
  §K   Auto-discovery integration
"""
from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from backend.core.ouroboros.governance import (
    claude_style_transport as cst,
)
from backend.core.ouroboros.ui import theme


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_flag_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "JARVIS_RENDER_MODE",
        "JARVIS_CLAUDE_STYLE_SHOW_HEARTBEATS",
        "JARVIS_CLAUDE_STYLE_LINE_CHARS",
        "JARVIS_CLAUDE_STYLE_DETAIL_LINES",
    ):
        monkeypatch.delenv(name, raising=False)
    # The transport now narrates: keep the model out of a unit test, and
    # give every test its own NarrativeChannel so subscriptions never
    # accumulate across transports. Narration itself is proven in
    # test_transport_narration_and_design.py.
    from backend.core.ouroboros.governance import intent_prompter as _ip
    monkeypatch.setattr(_ip, "is_master_flag_enabled", lambda: False)
    from backend.core.ouroboros.battle_test import narrative_channel as _nc
    _nc.reset_default_channel_for_tests()
    yield
    _nc.reset_default_channel_for_tests()


@pytest.fixture
def fresh_registry():
    from backend.core.ouroboros.governance import flag_registry as fr
    fr.reset_default_registry()
    reg = fr.ensure_seeded()
    yield reg
    fr.reset_default_registry()


class _RecordingConsole:
    def __init__(self) -> None:
        self.prints: List[str] = []

    def print(self, text: str, **kwargs: Any) -> None:
        self.prints.append(text)


def _msg(msg_type: str, op_id: str, payload: Dict[str, Any]) -> Any:
    return SimpleNamespace(
        msg_type=SimpleNamespace(value=msg_type),
        op_id=op_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# §A — Closed taxonomies
# ---------------------------------------------------------------------------


class TestRenderModeClosedTaxonomy:
    def test_exact_two_members(self):
        assert {m.value for m in cst.RenderMode} == {"CLAUDE", "SERPENT"}


class TestOpStatusGlyphClosedTaxonomy:
    def test_exact_six_members(self):
        assert {m.name for m in cst.OpStatusGlyph} == {
            "ACTIVE", "RUNNING", "DONE", "FAILED",
            "CANCELLED", "NOOP",
        }

    def test_glyph_values_distinct(self):
        # Six distinct STATUSES. Two may share a MARK — outcomes are told
        # apart by colour and copy (OV_DESIGN_LANGUAGE §3.4), never by a
        # glyph outside the six-glyph ration.
        values = [m.value for m in cst.OpStatusGlyph]
        assert len(set(values)) == len(values)

    def test_every_mark_is_in_the_design_ration(self):
        for m in cst.OpStatusGlyph:
            assert theme.mark(m.mark, unicode=True), m
            assert theme.mark(m.mark, unicode=False), m
            assert m.role in cst._SEM, m


# ---------------------------------------------------------------------------
# §B — resolve_render_mode
# ---------------------------------------------------------------------------


class TestResolveRenderMode:
    def test_default_claude(self, fresh_registry):
        assert cst.resolve_render_mode() is cst.RenderMode.CLAUDE

    def test_explicit_serpent(
        self, monkeypatch: pytest.MonkeyPatch, fresh_registry,
    ):
        monkeypatch.setenv("JARVIS_RENDER_MODE", "SERPENT")
        assert cst.resolve_render_mode() is cst.RenderMode.SERPENT

    def test_lowercase_normalized(
        self, monkeypatch: pytest.MonkeyPatch, fresh_registry,
    ):
        monkeypatch.setenv("JARVIS_RENDER_MODE", "claude")
        assert cst.resolve_render_mode() is cst.RenderMode.CLAUDE

    def test_unknown_falls_back_to_claude(
        self, monkeypatch: pytest.MonkeyPatch, fresh_registry,
    ):
        # Operator typo doesn't restore the noisy legacy
        monkeypatch.setenv("JARVIS_RENDER_MODE", "BOGUS_MODE")
        assert cst.resolve_render_mode() is cst.RenderMode.CLAUDE

    def test_show_heartbeats_default_false(self, fresh_registry):
        assert cst.show_heartbeats() is False


# ---------------------------------------------------------------------------
# §C — INTENT — single line with bullet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIntentRendering:
    async def test_basic_intent_emits_one_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-7c17aa-x", {
            "goal": "Wave 3 graduation",
            "outcome_source": "TestFailure",
        }))
        assert len(console.prints) == 1
        line = console.prints[0]
        assert theme.mark("action") in line          # an actor does something
        assert "TestFailure" in line
        assert "op7c17" not in line  # a hash the operator cannot /expand is noise
        assert "Wave 3 graduation" in line

    async def test_risk_tier_renders_as_words(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x",
            "outcome_source": "TestFailure",
            "risk_tier": "APPROVAL_REQUIRED",
        }))
        assert "approval required" in console.prints[0]   # never a raw enum
        assert "APPROVAL_REQUIRED" not in console.prints[0]

    async def test_safe_auto_risk_not_shown(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x",
            "outcome_source": "TestFailure",
            "risk_tier": "SAFE_AUTO",
        }))
        # SAFE_AUTO is the common case; not shown to reduce noise
        assert "SAFE_AUTO" not in console.prints[0]

    async def test_target_files_truncated(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        long_path = "a/" * 30 + "file.py"
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x",
            "outcome_source": "Operation",
            "target_files": [long_path],
        }))
        # Path shortened to …/parent/file.py with the theme's ellipsis
        assert theme.mark("ellipsis") in console.prints[0]
        assert "file.py" in console.prints[0]


# ---------------------------------------------------------------------------
# §D — DECISION outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDecisionOutcomes:
    async def _setup_op(self, t: Any, op_id: str = "op-1") -> None:
        await t.send(_msg("INTENT", op_id, {
            "goal": "test op", "outcome_source": "TestFailure",
        }))

    async def test_completed_renders_done_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "completed",
            "files_changed": ["x.py"],
        }))
        line = console.prints[0]
        assert cst.OpStatusGlyph.DONE.glyph in line
        assert "TestFailure" in line
        assert "x.py" in line

    async def test_failed_renders_shed_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "failed",
            "reason_code": "background_dw_blocked",
        }))
        line = console.prints[0]
        assert cst.OpStatusGlyph.FAILED.glyph in line
        assert "shed" in line
        assert "background dw blocked" in line      # the code, as words

    async def test_noop_renders_skip_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "noop",
            "reason_code": "duplicate_signal",
        }))
        line = console.prints[0]
        assert cst.OpStatusGlyph.NOOP.glyph in line
        assert "no change" in line

    async def test_notify_apply_renders_yellow_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "notify_apply",
            "target_files": ["x.py"],
        }))
        line = console.prints[0]
        assert "auto-applying" in line
        assert "x.py" in line
        assert cst._SEM["heal"] in line              # the tier is a colour, not a word

    async def test_escalated_renders_yellow_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "escalated",
            "reason_code": "iron_gate_block",
        }))
        line = console.prints[0]
        assert "held for review" in line
        assert "iron gate block" in line

    async def test_decision_without_intent_silently_dropped(
        self, fresh_registry,
    ):
        # Common at boot for orphan reconciliation
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("DECISION", "op-orphan", {
            "outcome": "failed",
        }))
        assert console.prints == []

    async def test_state_cleared_after_decision(self, fresh_registry):
        # After DECISION clears the state, a second INTENT for the
        # same op_id starts fresh.
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await self._setup_op(t)
        await t.send(_msg("DECISION", "op-1", {"outcome": "completed"}))
        assert "op-1" not in t._op_state


# ---------------------------------------------------------------------------
# §D2 — Authoritative elapsed (op lifetime), not the local observation clock
# ---------------------------------------------------------------------------


class TestElapsedResolution:
    def test_format_duration_units(self):
        assert cst._format_duration(5.0) == "5.0s"
        assert cst._format_duration(134.0) == "2m 14s"
        assert cst._format_duration(3725.0) == "1h 2m"
        # Defensive: negatives clamp to 0, non-numeric degrades not raises.
        assert cst._format_duration(-3.0) == "0.0s"
        assert cst._format_duration(None) == "0.0s"
        # Absurd values clamp to the 24h ceiling rather than overflow.
        assert cst._format_duration(10 ** 12) == "24h 0m"

    def test_resolve_elapsed_prefers_authoritative_duration(self):
        # An op the cockpit met only at its terminal state has no local
        # start (started_monotonic=0.0) -> the old path rendered 0.0s. The
        # terminal DECISION's authoritative op lifetime must win.
        t = cst.ClaudeStyleTransport(console=_RecordingConsole())
        state = cst._OpState(
            op_id="op-held", short_id="opheld",
            sensor="Operation", summary="", started_monotonic=0.0,
        )
        assert t._resolve_elapsed(state, {"duration_s": 134.0}) == "2m 14s"

    def test_resolve_elapsed_falls_back_to_local_when_absent(self):
        # No duration on the wire (older emitter / non-orchestrator
        # DECISION): fall back to the local clock, which is 0.0s for an
        # unobserved op -- honest 'unknown', never a fabricated number.
        t = cst.ClaudeStyleTransport(console=_RecordingConsole())
        state = cst._OpState(
            op_id="op-held", short_id="opheld",
            sensor="Operation", summary="", started_monotonic=0.0,
        )
        assert t._resolve_elapsed(state, {}) == "0.0s"
        # A malformed duration degrades to the fallback, never raises.
        assert t._resolve_elapsed(state, {"duration_s": "nan-ish"}) == "0.0s"

    @pytest.mark.asyncio
    async def test_decision_duration_renders_over_local_clock(
        self, fresh_registry,
    ):
        # Full path: an op announced via INTENT (local elapsed ~0.0s) whose
        # terminal DECISION carries the authoritative lifetime renders that
        # lifetime on both the outcome line and the recap beneath it.
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "test op", "outcome_source": "TestFailure",
        }))
        console.prints.clear()
        await t.send(_msg("DECISION", "op-1", {
            "outcome": "completed", "files_changed": ["x.py"],
            "duration_s": 134.0,
        }))
        blob = "\n".join(console.prints)
        assert "2m 14s" in blob
        assert "0.0s" not in blob


# ---------------------------------------------------------------------------
# §E — POSTMORTEM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPostmortem:
    async def test_postmortem_renders_failure_line(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x", "outcome_source": "Operation",
        }))
        console.prints.clear()
        await t.send(_msg("POSTMORTEM", "op-1", {
            "root_cause": "VERIFY phase exhausted retries",
        }))
        line = console.prints[0]
        assert cst.OpStatusGlyph.FAILED.glyph in line
        assert "postmortem" in line
        assert "VERIFY" in line


# ---------------------------------------------------------------------------
# §F — Boot recovery suppression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBootRecoverySuppression:
    async def test_orphans_collapse_to_one_summary(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        # 5 orphan reconciliation INTENTs with boot_recovery_* reason
        for i in range(5):
            await t.send(_msg("INTENT", f"op-stale-{i}", {
                "goal": "reconciliation",
                "reason_code": "boot_recovery_orphan",
            }))
        # Exactly ONE "reconciling" line emitted (on first orphan)
        assert sum(
            1 for p in console.prints if "reconciling" in p
        ) == 1
        # First real INTENT triggers the summary flush
        await t.send(_msg("INTENT", "op-real", {
            "goal": "real op", "outcome_source": "TestFailure",
        }))
        # Summary line shows the count (5)
        summary = [
            p for p in console.prints
            if "5 stale entries reconciled" in p
        ]
        assert len(summary) == 1


# ---------------------------------------------------------------------------
# §G — HEARTBEAT default-silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHeartbeatGate:
    async def test_heartbeat_silent_by_default(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x", "outcome_source": "TestFailure",
        }))
        intent_count = len(console.prints)
        await t.send(_msg("HEARTBEAT", "op-1", {"phase": "GENERATE"}))
        # No additional output
        assert len(console.prints) == intent_count

    async def test_heartbeat_renders_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch, fresh_registry,
    ):
        monkeypatch.setenv(
            "JARVIS_CLAUDE_STYLE_SHOW_HEARTBEATS", "true",
        )
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x", "outcome_source": "TestFailure",
        }))
        await t.send(_msg("HEARTBEAT", "op-1", {"phase": "GENERATE"}))
        assert any("generate" in p.lower() for p in console.prints)


# ---------------------------------------------------------------------------
# §H — Defensive paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDefensivePaths:
    async def test_malformed_msg_no_raise(self, fresh_registry):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        # No msg_type
        await t.send(SimpleNamespace(payload={}, op_id=""))
        # No payload
        await t.send(SimpleNamespace(
            msg_type=SimpleNamespace(value="INTENT"),
            op_id="op-1",
        ))
        # Reaches here without raising

    async def test_unknown_msg_type_silently_dropped(
        self, fresh_registry,
    ):
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("FUTURE_KIND", "op-1", {}))
        assert console.prints == []

    async def test_console_print_failure_doesnt_propagate(
        self, fresh_registry,
    ):
        class _BrokenConsole:
            def print(self, text: str, **kw: Any) -> None:
                raise RuntimeError("console boom")
        t = cst.ClaudeStyleTransport(console=_BrokenConsole())
        await t.send(_msg("INTENT", "op-1", {
            "goal": "x", "outcome_source": "TestFailure",
        }))
        # Reaches here without raising


# ---------------------------------------------------------------------------
# §I — Sensor inference from goal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSensorInference:
    async def _check_sensor(
        self, goal: str, expected: str,
    ) -> None:
        console = _RecordingConsole()
        t = cst.ClaudeStyleTransport(console=console)
        await t.send(_msg("INTENT", "op-1", {"goal": goal}))
        assert expected in console.prints[0]

    async def test_test_keyword(self, fresh_registry):
        await self._check_sensor("test failure in foo", "TestFailure")

    async def test_todo_keyword(self, fresh_registry):
        await self._check_sensor("TODO at file.py:1", "TODO")

    async def test_github_keyword(self, fresh_registry):
        await self._check_sensor(
            "GitHub issue #42", "GitHubIssue",
        )

    async def test_explor_keyword(self, fresh_registry):
        await self._check_sensor(
            "Proactive exploration", "Exploration",
        )

    async def test_doc_keyword(self, fresh_registry):
        await self._check_sensor(
            "Documentation drift", "Documentation",
        )

    async def test_fallback_operation(self, fresh_registry):
        await self._check_sensor("something else", "Operation")


# ---------------------------------------------------------------------------
# §J — AST pins
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cc1_pins() -> list:
    return list(cst.register_shipped_invariants())


class TestCC1ASTPinsClean:
    def test_five_pins_registered(self, cc1_pins):
        assert len(cc1_pins) == 5
        names = {i.invariant_name for i in cc1_pins}
        assert names == {
            "claude_style_transport_no_rich_import",
            "claude_style_transport_no_authority_imports",
            "claude_style_transport_render_mode_closed",
            "claude_style_transport_op_status_glyph_closed",
            "claude_style_transport_discovery_symbols_present",
        }

    @pytest.fixture(scope="class")
    def real_module_ast(self):
        import inspect
        src = inspect.getsource(cst)
        return ast.parse(src), src

    def test_no_rich_import_clean(self, cc1_pins, real_module_ast):
        tree, src = real_module_ast
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_no_rich_import")
        assert pin.validate(tree, src) == ()

    def test_no_authority_imports_clean(self, cc1_pins, real_module_ast):
        tree, src = real_module_ast
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_no_authority_imports")
        assert pin.validate(tree, src) == ()

    def test_render_mode_closed_clean(self, cc1_pins, real_module_ast):
        tree, src = real_module_ast
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_render_mode_closed")
        assert pin.validate(tree, src) == ()

    def test_op_status_glyph_closed_clean(self, cc1_pins, real_module_ast):
        tree, src = real_module_ast
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_op_status_glyph_closed")
        assert pin.validate(tree, src) == ()


class TestCC1ASTPinsCatchTampering:
    def test_authority_import_caught(self, cc1_pins):
        tampered = ast.parse(
            "from backend.core.ouroboros.governance.cancel_token "
            "import x\n"
        )
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_no_authority_imports")
        violations = pin.validate(tampered, "")
        assert any("cancel_token" in v for v in violations)

    def test_added_render_mode_caught(self, cc1_pins):
        tampered_src = (
            "class RenderMode:\n"
            "    CLAUDE = 'CLAUDE'\n"
            "    SERPENT = 'SERPENT'\n"
            "    HYBRID = 'HYBRID'\n"
        )
        tampered = ast.parse(tampered_src)
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_render_mode_closed")
        violations = pin.validate(tampered, tampered_src)
        assert violations

    def test_added_glyph_caught(self, cc1_pins):
        tampered_src = (
            "class OpStatusGlyph:\n"
            "    ACTIVE = '·'\n"
            "    RUNNING = '●'\n"
            "    DONE = '✓'\n"
            "    FAILED = '✗'\n"
            "    CANCELLED = '◌'\n"
            "    NOOP = '⏭'\n"
            "    NEW_GLYPH = '@'\n"
        )
        tampered = ast.parse(tampered_src)
        pin = next(p for p in cc1_pins
                   if p.invariant_name ==
                   "claude_style_transport_op_status_glyph_closed")
        violations = pin.validate(tampered, tampered_src)
        assert violations


# ---------------------------------------------------------------------------
# §K — Auto-discovery
# ---------------------------------------------------------------------------


class TestAutoDiscoveryIntegration:
    def test_flag_registry_picks_up_cc1(self, fresh_registry):
        names = {s.name for s in fresh_registry.list_all()}
        assert "JARVIS_RENDER_MODE" in names
        assert "JARVIS_CLAUDE_STYLE_SHOW_HEARTBEATS" in names

    def test_render_mode_default_is_claude(self, fresh_registry):
        from backend.core.ouroboros.governance import flag_registry as fr
        reg = fr.FlagRegistry()
        cst.register_flags(reg)
        spec = reg.get_spec("JARVIS_RENDER_MODE")
        assert spec is not None
        assert spec.default == "CLAUDE"

    def test_shipped_invariants_includes_cc1_pins(self):
        from backend.core.ouroboros.governance.meta import (
            shipped_code_invariants as sci,
        )
        for inv in cst.register_shipped_invariants():
            sci.register_shipped_code_invariant(inv)
        names = {
            i.invariant_name for i in sci.list_shipped_code_invariants()
        }
        for expected in (
            "claude_style_transport_no_rich_import",
            "claude_style_transport_no_authority_imports",
            "claude_style_transport_render_mode_closed",
            "claude_style_transport_op_status_glyph_closed",
            "claude_style_transport_discovery_symbols_present",
        ):
            assert expected in names

    def test_validate_all_no_cc1_violations(self):
        from backend.core.ouroboros.governance.meta import (
            shipped_code_invariants as sci,
        )
        for inv in cst.register_shipped_invariants():
            sci.register_shipped_code_invariant(inv)
        results = sci.validate_all()
        cc1_failures = [
            r for r in results
            if r.invariant_name.startswith("claude_style_transport_")
        ]
        assert cc1_failures == []
