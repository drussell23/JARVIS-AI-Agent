"""Parity tests for :class:`GATERunner` (Wave 2 (5) Slice 4a.2).

Verbatim transcription of orchestrator.py GATE block (~600 lines)
with explicit parity coverage on every sub-gate the operator named:

* can_write policy check (hard terminal)
* SecurityReviewer (LLM-as-a-Judge)
* SimilarityGate (risk escalation)
* SemanticGuardian (deterministic pre-APPLY pattern check)
* MutationGate (enforce mode risk upgrade)
* MIN_RISK_TIER floor (paranoia/quiet hours composed)
* frozen_autonomy_tier=observe → APPROVAL_REQUIRED
* JARVIS_RISK_CEILING env override
* Phase 5a green SAFE_AUTO preview + cancel window
* Phase 5b yellow NOTIFY_APPLY preview + cancel window

The `risk_tier` artifact threading is pinned as a first-class concern —
GATE mutates it at up to 6 sites and the downstream APPROVE phase
depends on the final value.

Branch ordering parity: SecurityReviewer fires BEFORE SimilarityGate
which fires BEFORE frozen_tier which fires BEFORE RISK_CEILING which
fires BEFORE SemanticGuardian. This ordering affects approval
semantics (e.g. hard SemanticGuardian finding can override a lower
RISK_CEILING). Tests enforce the ordering where applicable.

Authority invariant: no candidate_generator / iron_gate / change_engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.ouroboros.governance.ledger import OperationState
from backend.core.ouroboros.governance.op_context import (
    OperationContext,
    OperationPhase,
)
from backend.core.ouroboros.governance.phase_runner import (
    PhaseResult,
    PhaseRunner,
)
from backend.core.ouroboros.governance.phase_runners.gate_runner import (
    GATERunner,
)
from backend.core.ouroboros.governance.risk_engine import RiskTier


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSerpent:
    def __init__(self):
        self.updates: List[str] = []

    def update_phase(self, p: str):
        self.updates.append(p)


class _FakeComm:
    def __init__(self):
        self.heartbeats: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self._transports: List[Any] = []

    async def emit_heartbeat(self, **kwargs):
        self.heartbeats.append(kwargs)

    async def emit_decision(self, **kwargs):
        self.decisions.append(kwargs)


class _FakeStack:
    def __init__(self, can_write_allowed: bool = True, can_write_reason: str = "ok"):
        self.comm = _FakeComm()
        self._can_write_allowed = can_write_allowed
        self._can_write_reason = can_write_reason
        self.prime_client = None
        self.serpent_flow = None

    def can_write(self, req):
        return (self._can_write_allowed, self._can_write_reason)


@dataclass
class _FakeConfig:
    project_root: Path


@dataclass
class _FakeOrchestrator:
    _stack: _FakeStack
    _config: _FakeConfig
    _cancel_requested: bool = False
    ledger_records: List = field(default_factory=list)
    review_shadow_calls: int = 0

    async def _record_ledger(self, ctx, state, extra):
        self.ledger_records.append((ctx.phase, state, extra))

    async def _run_review_shadow(self, ctx, best_candidate):
        self.review_shadow_calls += 1

    async def _apply_review_gate(self, ctx, best_candidate, risk_tier):
        # Phase 1b seam: dispatch the review once and fold its verdict into the
        # tier. The fake review returns None (no aggregate), so this is a no-op
        # on the tier — the real fail-soft path — while preserving the
        # review-shadow call count the parity harness checks.
        await self._run_review_shadow(ctx, best_candidate)
        return risk_tier

    def _review_downlevel_hard_blocked(self, ctx) -> bool:
        return False

    def _is_cancel_requested(self, op_id: str) -> bool:
        return self._cancel_requested

    async def _discover_tests_for_gate(self, path):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gate_ctx(tmp_path: Path) -> OperationContext:
    (tmp_path / "a.py").write_text("pass\n")
    return (
        OperationContext.create(
            target_files=(str(tmp_path / "a.py"),),
            description="gate parity",
        )
        .advance(OperationPhase.ROUTE, risk_tier=RiskTier.SAFE_AUTO)
        .advance(OperationPhase.GENERATE)
        .advance(OperationPhase.VALIDATE)
        .advance(OperationPhase.GATE)
    )


@pytest.fixture
def ctx(tmp_path):
    return _gate_ctx(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "JARVIS_RISK_CEILING", "JARVIS_MIN_RISK_TIER",
        "JARVIS_PARANOIA_MODE", "JARVIS_AUTO_APPLY_QUIET_HOURS",
        "JARVIS_SEMANTIC_GUARD_ENABLED",
        "JARVIS_MUTATION_GATE_ENABLED",
        "JARVIS_SAFE_AUTO_PREVIEW_DELAY_S",
        "JARVIS_NOTIFY_APPLY_DELAY_S",
        "JARVIS_REVIEW_SUBAGENT_SHADOW",
        "JARVIS_DIFF_PREVIEW_ALL",
    ):
        monkeypatch.delenv(var, raising=False)
    # Set NOTIFY_APPLY_DELAY_S=0 so 5b tests don't sleep
    monkeypatch.setenv("JARVIS_NOTIFY_APPLY_DELAY_S", "0")
    monkeypatch.setenv("JARVIS_SAFE_AUTO_PREVIEW_DELAY_S", "0")
    yield


def _orch(tmp_path, **overrides):
    stack_kw = {}
    for k in ("can_write_allowed", "can_write_reason"):
        if k in overrides:
            stack_kw[k] = overrides.pop(k)
    return _FakeOrchestrator(
        _stack=_FakeStack(**stack_kw),
        _config=_FakeConfig(project_root=tmp_path),
        **overrides,
    )


def _candidate():
    return {
        "candidate_id": "c0",
        "file_path": "a.py",
        "full_content": "x = 1\n",
    }


# ---------------------------------------------------------------------------
# (1) Class wiring
# ---------------------------------------------------------------------------


def test_gate_runner_is_phase_runner():
    assert issubclass(GATERunner, PhaseRunner)
    assert GATERunner.phase is OperationPhase.GATE


# ---------------------------------------------------------------------------
# (2) can_write denied → gate_blocked terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_write_denied_terminates(ctx, tmp_path):
    orch = _orch(tmp_path, can_write_allowed=False, can_write_reason="forbidden_path")
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)

    assert result.status == "fail"
    assert result.reason == "gate_blocked:forbidden_path"
    assert result.next_phase is None
    assert result.next_ctx.phase is OperationPhase.CANCELLED
    assert result.next_ctx.terminal_reason_code == "gate_blocked:forbidden_path"
    # Ledger with BLOCKED state
    assert orch.ledger_records
    assert orch.ledger_records[0][1] is OperationState.BLOCKED


# ---------------------------------------------------------------------------
# (3) Happy path — SAFE_AUTO proceeds to APPROVE with unchanged tier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_safe_auto_proceeds_to_approve(ctx, tmp_path):
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)

    assert result.status == "ok"
    assert result.reason == "gated"
    assert result.next_phase is OperationPhase.APPROVE
    assert result.artifacts["risk_tier"] is RiskTier.SAFE_AUTO
    assert result.artifacts["best_candidate"] is not None
    # REVIEW shadow hook always fires
    assert orch.review_shadow_calls == 1


# ---------------------------------------------------------------------------
# (4) Frozen autonomy tier=observe → APPROVAL_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_frozen_tier_observe_escalates_to_approval(tmp_path):
    (tmp_path / "a.py").write_text("pass\n")
    ctx = (
        OperationContext.create(
            target_files=(str(tmp_path / "a.py"),), description="observe",
        )
        .advance(OperationPhase.ROUTE, risk_tier=RiskTier.SAFE_AUTO)
        .advance(OperationPhase.GENERATE)
        .advance(OperationPhase.VALIDATE)
        .advance(OperationPhase.GATE, frozen_autonomy_tier="observe")
    )
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)

    assert result.status == "ok"
    assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# (5) JARVIS_RISK_CEILING env floor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_ceiling_escalates_below_floor(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_RISK_CEILING", "NOTIFY_APPLY")
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert result.artifacts["risk_tier"] is RiskTier.NOTIFY_APPLY


@pytest.mark.asyncio
async def test_risk_ceiling_does_not_downgrade(ctx, tmp_path, monkeypatch):
    """A SAFE_AUTO ceiling does NOT downgrade APPROVAL_REQUIRED."""
    monkeypatch.setenv("JARVIS_RISK_CEILING", "SAFE_AUTO")
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.APPROVAL_REQUIRED).run(ctx)
    assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# (6) SemanticGuardian branch — emits structured log even on clean pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_guardian_clean_still_emits_log(ctx, tmp_path, caplog):
    """The `[SemanticGuard]` log line is a contract — fires on every op
    (hit OR clean) so aggregation pipelines have a stable record."""
    import logging
    caplog.set_level(logging.INFO, logger="Ouroboros.Orchestrator")
    orch = _orch(tmp_path)
    await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    sg_lines = [r for r in caplog.records if "[SemanticGuard]" in r.message]
    assert sg_lines, "expected SemanticGuard structured log"
    assert "findings=" in sg_lines[0].message


# ---------------------------------------------------------------------------
# (7) MutationGate enforce-mode upgrades to APPROVAL_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mutation_gate_enforce_mode_upgrades(ctx, tmp_path, monkeypatch):
    """When MutationGate returns decision=upgrade_to_approval in enforce
    mode, risk_tier must escalate from SAFE_AUTO to APPROVAL_REQUIRED."""
    monkeypatch.setenv("JARVIS_MUTATION_GATE_ENABLED", "1")

    _mock_verdict = MagicMock()
    _mock_verdict.decision = "upgrade_to_approval"
    _mock_verdict.score = 0.5
    _mock_verdict.grade = "C"
    _mock_verdict.caught = 5
    _mock_verdict.total_mutants = 10
    _mock_verdict.survivors = []
    _mock_verdict.cache_hits = 0
    _mock_verdict.cache_misses = 10
    _mock_verdict.duration_s = 1.0

    with patch("backend.core.ouroboros.governance.mutation_gate.gate_enabled", return_value=True), \
         patch("backend.core.ouroboros.governance.mutation_gate.load_allowlist", return_value={}), \
         patch("backend.core.ouroboros.governance.mutation_gate.is_path_critical", return_value=True), \
         patch("backend.core.ouroboros.governance.mutation_gate.evaluate_file", return_value=_mock_verdict), \
         patch("backend.core.ouroboros.governance.mutation_gate.merge_verdicts", return_value=_mock_verdict), \
         patch("backend.core.ouroboros.governance.mutation_gate.gate_mode", return_value="enforce"), \
         patch("backend.core.ouroboros.governance.mutation_gate.MODE_ENFORCE", "enforce"), \
         patch("backend.core.ouroboros.governance.mutation_gate.append_ledger"):
        orch = _orch(tmp_path)
        result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# (8) MIN_RISK_TIER floor composes with strictest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_risk_tier_floor_escalates(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_RISK_TIER", "notify_apply")
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert result.artifacts["risk_tier"] is RiskTier.NOTIFY_APPLY


@pytest.mark.asyncio
async def test_paranoia_mode_escalates(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PARANOIA_MODE", "1")
    orch = _orch(tmp_path)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    # PARANOIA_MODE is shortcut for notify_apply floor
    assert result.artifacts["risk_tier"] in (
        RiskTier.NOTIFY_APPLY, RiskTier.APPROVAL_REQUIRED,
    )


# ---------------------------------------------------------------------------
# (9) Phase 5a-green SAFE_AUTO preview — cancel during window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5a_green_preview_cancel_terminates(ctx, tmp_path, monkeypatch):
    """Human is watching (TTY OR JARVIS_DIFF_PREVIEW_ALL) + SAFE_AUTO +
    _is_cancel_requested returns True → user_rejected_safe_auto_preview."""
    monkeypatch.setenv("JARVIS_DIFF_PREVIEW_ALL", "1")  # Force _human_is_watching
    monkeypatch.setenv("JARVIS_SAFE_AUTO_PREVIEW_DELAY_S", "0.01")  # minimal

    orch = _orch(tmp_path, _cancel_requested=True)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)

    assert result.status == "fail"
    assert result.reason == "user_rejected_safe_auto_preview"
    assert result.next_ctx.phase is OperationPhase.CANCELLED
    assert result.next_ctx.terminal_reason_code == "user_rejected_safe_auto_preview"


@pytest.mark.asyncio
async def test_5a_green_preview_no_cancel_continues(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DIFF_PREVIEW_ALL", "1")
    monkeypatch.setenv("JARVIS_SAFE_AUTO_PREVIEW_DELAY_S", "0.01")

    orch = _orch(tmp_path, _cancel_requested=False)
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    # Preview shown but no cancel → proceeds to APPROVE
    assert result.status == "ok"
    assert result.next_phase is OperationPhase.APPROVE


# ---------------------------------------------------------------------------
# (10) Phase 5b NOTIFY_APPLY — cancel during preview window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5b_notify_apply_cancel_terminates(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_NOTIFY_APPLY_DELAY_S", "0.01")
    orch = _orch(tmp_path, _cancel_requested=True)
    result = await GATERunner(orch, None, _candidate(), RiskTier.NOTIFY_APPLY).run(ctx)

    assert result.status == "fail"
    assert result.reason == "user_rejected_notify_apply"
    assert result.next_ctx.phase is OperationPhase.CANCELLED


@pytest.mark.asyncio
async def test_5b_notify_apply_no_cancel_continues(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_NOTIFY_APPLY_DELAY_S", "0.01")
    orch = _orch(tmp_path, _cancel_requested=False)
    result = await GATERunner(orch, None, _candidate(), RiskTier.NOTIFY_APPLY).run(ctx)

    assert result.status == "ok"
    assert result.next_phase is OperationPhase.APPROVE
    assert result.artifacts["risk_tier"] is RiskTier.NOTIFY_APPLY


@pytest.mark.asyncio
async def test_5b_emits_notify_apply_decision(ctx, tmp_path, monkeypatch):
    """§8 logging parity: emit_decision fires with outcome=notify_apply."""
    monkeypatch.setenv("JARVIS_NOTIFY_APPLY_DELAY_S", "0")
    orch = _orch(tmp_path)
    await GATERunner(orch, None, _candidate(), RiskTier.NOTIFY_APPLY).run(ctx)
    assert any(
        d.get("outcome") == "notify_apply"
        for d in orch._stack.comm.decisions
    )


# ---------------------------------------------------------------------------
# (11) Similarity gate escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_similarity_gate_escalates_to_approval(ctx, tmp_path):
    with patch(
        "backend.core.ouroboros.governance.similarity_gate.check_similarity",
        return_value="too_similar_to_source",
    ):
        orch = _orch(tmp_path)
        result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# (12) Branch ordering — SemanticGuardian upgrade overrides RISK_CEILING NO-OP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_ordering_semantic_guardian_after_risk_ceiling(
    ctx, tmp_path, monkeypatch,
):
    """RISK_CEILING=NOTIFY_APPLY then SemanticGuardian hard finding →
    APPROVAL_REQUIRED. SemanticGuardian fires AFTER RISK_CEILING in the
    inline code; tier must reflect both gates composed (strictest wins)."""
    monkeypatch.setenv("JARVIS_RISK_CEILING", "NOTIFY_APPLY")

    _mock_finding = MagicMock()
    _mock_finding.severity = "hard"
    _mock_finding.pattern = "removed_import_still_referenced"

    with patch(
        "backend.core.ouroboros.governance.semantic_guardian.SemanticGuardian.inspect_batch",
        return_value=[_mock_finding],
    ), patch(
        "backend.core.ouroboros.governance.semantic_guardian.recommend_tier_floor",
        return_value="approval_required",
    ):
        orch = _orch(tmp_path)
        result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    # SemanticGuardian upgrade wins over RISK_CEILING floor
    assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# (13) REVIEW subagent shadow hook always fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_shadow_hook_always_fires(ctx, tmp_path):
    orch = _orch(tmp_path)
    await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert orch.review_shadow_calls == 1


# ---------------------------------------------------------------------------
# (14) Exception swallow — SecurityReviewer crash doesn't abort phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_security_reviewer_exception_swallowed(ctx, tmp_path):
    with patch(
        "backend.core.ouroboros.governance.security_reviewer.SecurityReviewer",
        side_effect=RuntimeError("sec boom"),
    ):
        orch = _orch(tmp_path)
        result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert result.status == "ok"
    assert result.next_phase is OperationPhase.APPROVE


# ---------------------------------------------------------------------------
# (15) Authority invariant
# ---------------------------------------------------------------------------


def test_gate_runner_bans_execution_authority_imports():
    import inspect
    from backend.core.ouroboros.governance.phase_runners import gate_runner

    src = inspect.getsource(gate_runner)
    for banned in ("candidate_generator", "iron_gate", "change_engine"):
        for line in src.splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")):
                assert banned not in s, (
                    f"gate_runner.py must not import {banned}: {s}"
                )


# ---------------------------------------------------------------------------
# (16) artifacts always expose risk_tier + best_candidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifacts_present_on_all_paths(ctx, tmp_path):
    orch = _orch(tmp_path, can_write_allowed=False, can_write_reason="x")
    result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)
    assert "risk_tier" in result.artifacts
    assert "best_candidate" in result.artifacts


# ---------------------------------------------------------------------------
# fs-hot-tier Batch 3 (row 15, 2026-07-03) — Orchestrator._discover_tests_for_gate
# routes MutationGate's ``tests_dir.rglob(...)`` scan through
# cooperative_fs_io.offload(cpu_bound=False) instead of running the rglob
# synchronously on the loop thread inside the GATE phase runner.
# ---------------------------------------------------------------------------


def _real_orchestrator_from_module():
    """Import the real Orchestrator class (not the parity fake) so the
    ``_discover_tests_for_gate`` method under test is the production
    implementation."""
    from backend.core.ouroboros.governance.orchestrator import Orchestrator
    return Orchestrator


class _MinimalRealConfig:
    def __init__(self, project_root):
        self.project_root = project_root


class TestDiscoverTestsForGateOffload:
    @pytest.mark.asyncio
    async def test_routes_through_offload_thread_pool(self, tmp_path, monkeypatch):
        """Spy on cooperative_fs_io.offload — the rglob scan must be
        dispatched with cpu_bound=False (thread pool: scoped tests/
        rglob is syscall-dominated, not CPU-after-scan)."""
        Orchestrator = _real_orchestrator_from_module()
        orch = Orchestrator.__new__(Orchestrator)
        orch._config = _MinimalRealConfig(project_root=tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_foo_widget.py").write_text("def test_x(): pass\n")

        from backend.core.ouroboros.governance import cooperative_fs_io
        calls = {"n": 0, "cpu_bound": None}
        real_offload = cooperative_fs_io.offload

        async def _spy_offload(fn, *args, cpu_bound=False, **kwargs):
            calls["n"] += 1
            calls["cpu_bound"] = cpu_bound
            return await real_offload(fn, *args, cpu_bound=cpu_bound, **kwargs)

        monkeypatch.setattr(cooperative_fs_io, "offload", _spy_offload)

        result = await orch._discover_tests_for_gate(Path("foo_widget.py"))

        assert calls["n"] == 1, "_discover_tests_for_gate must route through offload"
        assert calls["cpu_bound"] is False, (
            "scoped tests/ rglob is IO-bound (syscall-dominated) — must "
            "use the thread pool, not a process pool"
        )
        assert [p.name for p in result] == ["test_foo_widget.py"]

    @pytest.mark.asyncio
    async def test_parity_matches_sync_rglob_on_planted_tree(self, tmp_path):
        """The offloaded result must match what a direct synchronous
        rglob('test_<stem>*.py') would produce on a planted tree."""
        Orchestrator = _real_orchestrator_from_module()
        orch = Orchestrator.__new__(Orchestrator)
        orch._config = _MinimalRealConfig(project_root=tmp_path)

        tests_dir = tmp_path / "tests"
        (tests_dir / "sub").mkdir(parents=True)
        (tests_dir / "test_bar.py").write_text("pass\n")
        (tests_dir / "sub" / "test_bar_extra.py").write_text("pass\n")
        (tests_dir / "test_other.py").write_text("pass\n")

        expected = sorted(tests_dir.rglob("test_bar*.py"))
        result = await orch._discover_tests_for_gate(Path("bar.py"))
        assert result == expected

    @pytest.mark.asyncio
    async def test_missing_tests_dir_returns_empty(self, tmp_path):
        Orchestrator = _real_orchestrator_from_module()
        orch = Orchestrator.__new__(Orchestrator)
        orch._config = _MinimalRealConfig(project_root=tmp_path)
        result = await orch._discover_tests_for_gate(Path("nope.py"))
        assert result == []

    @pytest.mark.asyncio
    async def test_offload_error_degrades_to_empty_no_raise(self, tmp_path, monkeypatch):
        """Fail-soft: an OffloadError from the substrate must degrade to
        the same empty-list result the prior sync fail-soft gave —
        never raise into the GATE phase runner."""
        from backend.core.ouroboros.governance import cooperative_fs_io
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            OffloadError,
        )

        Orchestrator = _real_orchestrator_from_module()
        orch = Orchestrator.__new__(Orchestrator)
        orch._config = _MinimalRealConfig(project_root=tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_boom.py").write_text("pass\n")

        async def _boom_offload(fn, *args, **kwargs):
            return OffloadError(
                fn_name="_discover_tests_for_gate_worker",
                exc_type="OSError",
                message="synthetic offload-layer fault",
                cpu_bound=False,
            )

        monkeypatch.setattr(cooperative_fs_io, "offload", _boom_offload)
        result = await orch._discover_tests_for_gate(Path("boom.py"))
        assert result == []

    @pytest.mark.asyncio
    async def test_gate_runner_await_site_awaits_discover_tests(self, ctx, tmp_path, monkeypatch):
        """Await-guard: the GATERunner call site must actually await the
        (now-async) _discover_tests_for_gate — proven end-to-end through
        the MutationGate enforce-mode path with a real critical-path
        candidate, not just a unit call."""
        monkeypatch.setenv("JARVIS_MUTATION_GATE_ENABLED", "1")

        _mock_verdict = MagicMock()
        _mock_verdict.decision = "upgrade_to_approval"
        _mock_verdict.score = 0.5
        _mock_verdict.grade = "C"
        _mock_verdict.caught = 5
        _mock_verdict.total_mutants = 10
        _mock_verdict.survivors = []
        _mock_verdict.cache_hits = 0
        _mock_verdict.cache_misses = 10
        _mock_verdict.duration_s = 1.0

        discover_calls = {"n": 0}

        orch = _orch(tmp_path)

        async def _tracking_discover(path):
            discover_calls["n"] += 1
            return []

        orch._discover_tests_for_gate = _tracking_discover

        with patch("backend.core.ouroboros.governance.mutation_gate.gate_enabled", return_value=True), \
             patch("backend.core.ouroboros.governance.mutation_gate.load_allowlist", return_value={}), \
             patch("backend.core.ouroboros.governance.mutation_gate.is_path_critical", return_value=True), \
             patch("backend.core.ouroboros.governance.mutation_gate.evaluate_file", return_value=_mock_verdict), \
             patch("backend.core.ouroboros.governance.mutation_gate.merge_verdicts", return_value=_mock_verdict), \
             patch("backend.core.ouroboros.governance.mutation_gate.gate_mode", return_value="enforce"), \
             patch("backend.core.ouroboros.governance.mutation_gate.MODE_ENFORCE", "enforce"), \
             patch("backend.core.ouroboros.governance.mutation_gate.append_ledger"):
            result = await GATERunner(orch, None, _candidate(), RiskTier.SAFE_AUTO).run(ctx)

        assert discover_calls["n"] == 1, (
            "GATERunner must call orch._discover_tests_for_gate on the "
            "critical-path candidate"
        )
        assert result.artifacts["risk_tier"] is RiskTier.APPROVAL_REQUIRED


__all__ = []
