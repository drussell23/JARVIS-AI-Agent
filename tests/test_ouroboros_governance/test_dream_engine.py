"""tests/test_ouroboros_governance/test_dream_engine.py

TDD tests for DreamEngine — idle GPU speculative analysis.

Test cases:
    TC09:  Dream gate rejects when VM not ready
    TC10:  Dream gate rejects when user is active
    TC11:  Dream gate rejects when VM was woken by dream (not user)
    TC17:  Preemption on user activity abandons job
    TC18:  Flap damping prevents rapid re-entry
    TC23:  Dream prompts capped at 2048 tokens
    TC24:  Prime unavailable -> DREAM_DORMANT reason code
    TC29:  Direct HTTP used, NOT PrimeClient
    TC30:  Preemption saves partial state for resume
    Plus:  idempotent job key skip, sorted blueprints, stale discard,
           budget cap, resource governor yield, stop persists, start loads
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.ouroboros.consciousness.types import (
    BudgetHealth,
    ConsciousnessConfig,
    DreamMetrics,
    HealthTrend,
    ImprovementBlueprint,
    ResourceHealth,
    SubsystemHealth,
    TrinityHealthSnapshot,
    TrustHealth,
    UserActivityMonitor,
    compute_blueprint_id,
    compute_job_key,
)
from backend.core.ouroboros.consciousness.dream_metrics import DreamMetricsTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> ConsciousnessConfig:
    """Build a ConsciousnessConfig with sensible test defaults."""
    defaults = dict(
        enabled=True,
        health_poll_interval_s=30.0,
        dream_enabled=True,
        dream_idle_threshold_s=300.0,
        dream_reentry_cooldown_s=60.0,
        dream_max_minutes_per_day=120.0,
        dream_blueprint_ttl_hours=24.0,
        prophecy_enabled=True,
        memory_ttl_hours=168.0,
        briefing_on_startup=True,
    )
    defaults.update(overrides)
    return ConsciousnessConfig(**defaults)


def _make_prime_health(
    status: str = "healthy",
    model_loaded: bool = True,
    uptime_s: float = 600.0,
) -> SubsystemHealth:
    return SubsystemHealth(
        name="prime",
        status=status,
        score=1.0 if status == "healthy" else 0.0,
        details={"model_loaded": model_loaded, "uptime_s": uptime_s},
        polled_at_utc="2026-03-20T00:00:00+00:00",
    )


def _make_snapshot(
    prime_status: str = "healthy",
    model_loaded: bool = True,
    prime_uptime_s: float = 600.0,
) -> TrinityHealthSnapshot:
    prime = _make_prime_health(prime_status, model_loaded, prime_uptime_s)
    return TrinityHealthSnapshot(
        timestamp_utc="2026-03-20T00:00:00+00:00",
        overall_verdict="HEALTHY",
        overall_score=1.0,
        jarvis=SubsystemHealth(
            name="jarvis", status="healthy", score=1.0,
            details={}, polled_at_utc="2026-03-20T00:00:00+00:00",
        ),
        prime=prime,
        reactor=SubsystemHealth(
            name="reactor", status="healthy", score=1.0,
            details={}, polled_at_utc="2026-03-20T00:00:00+00:00",
        ),
        resources=ResourceHealth(
            cpu_percent=30.0, ram_percent=50.0, disk_percent=40.0,
            pressure="NORMAL",
        ),
        budget=BudgetHealth(
            daily_spend_usd=1.0, iteration_spend_usd=0.1, remaining_usd=9.0,
        ),
        trust=TrustHealth(current_tier="governed", graduation_progress=0.0),
    )


def _make_blueprint(
    blueprint_id: str = "test-bp-001",
    priority: float = 0.8,
    repo_sha: str = "abc123",
    policy_hash: str = "pol123",
) -> ImprovementBlueprint:
    return ImprovementBlueprint(
        blueprint_id=blueprint_id,
        title="Test Blueprint",
        description="A test improvement",
        category="test_coverage",
        priority_score=priority,
        target_files=("src/foo.py",),
        estimated_effort="small",
        estimated_cost_usd=0.01,
        repo="jarvis",
        repo_sha=repo_sha,
        computed_at_utc="2026-03-20T00:00:00+00:00",
        ttl_hours=24.0,
        model_used="qwen2.5-7b",
        policy_hash=policy_hash,
        oracle_neighborhood={},
        suggested_approach="Add unit test",
        risk_assessment="Low risk",
    )


class MockActivityMonitor:
    """Test double implementing UserActivityMonitor protocol."""

    def __init__(self, idle_seconds: float = 600.0) -> None:
        self._idle_seconds = idle_seconds

    def last_activity_s(self) -> float:
        return self._idle_seconds


@pytest.fixture
def tmp_dream_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dreams"
    d.mkdir()
    return d


@pytest.fixture
def healthy_cortex() -> MagicMock:
    cortex = MagicMock()
    cortex.get_snapshot.return_value = _make_snapshot()
    return cortex


@pytest.fixture
def memory_engine() -> MagicMock:
    engine = MagicMock()
    engine.get_file_reputation.return_value = MagicMock(fragility_score=0.1)
    return engine


@pytest.fixture
def idle_monitor() -> MockActivityMonitor:
    return MockActivityMonitor(idle_seconds=600.0)


@pytest.fixture
def active_monitor() -> MockActivityMonitor:
    return MockActivityMonitor(idle_seconds=10.0)


@pytest.fixture
def resource_governor() -> MagicMock:
    gov = MagicMock()
    gov.should_yield = AsyncMock(return_value=False)
    return gov


@pytest.fixture
def metrics_tracker() -> DreamMetricsTracker:
    return DreamMetricsTracker()


@pytest.fixture
def comm_protocol() -> MagicMock:
    comm = MagicMock()
    comm.emit_heartbeat = AsyncMock()
    return comm


@pytest.fixture
def config() -> ConsciousnessConfig:
    return _make_config()


def _build_engine(
    health_cortex: Any,
    memory_engine: Any,
    activity_monitor: Any,
    resource_governor: Any,
    metrics_tracker: DreamMetricsTracker,
    config: ConsciousnessConfig,
    persistence_dir: Path,
    jprime_url: str = "http://localhost:8000",
    comm: Any = None,
):
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    return DreamEngine(
        health_cortex=health_cortex,
        memory_engine=memory_engine,
        activity_monitor=activity_monitor,
        resource_governor=resource_governor,
        metrics_tracker=metrics_tracker,
        config=config,
        jprime_url=jprime_url,
        persistence_dir=persistence_dir,
        comm=comm,
    )


# ============================================================================
# TC09: Dream gate rejects VM not ready
# ============================================================================


@pytest.mark.asyncio
async def test_dream_gate_rejects_vm_not_ready(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC09: prime not healthy -> cannot dream."""
    # Case 1: prime status is not healthy
    healthy_cortex.get_snapshot.return_value = _make_snapshot(prime_status="offline")
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    can, reason = await engine._can_dream()
    assert can is False
    assert "prime" in reason.lower() or "health" in reason.lower()

    # Case 2: model not loaded
    healthy_cortex.get_snapshot.return_value = _make_snapshot(
        prime_status="healthy", model_loaded=False,
    )
    can2, reason2 = await engine._can_dream()
    assert can2 is False
    assert "model" in reason2.lower()


# ============================================================================
# TC10: Dream gate rejects user active
# ============================================================================


@pytest.mark.asyncio
async def test_dream_gate_rejects_user_active(
    healthy_cortex,
    memory_engine,
    active_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC10: last_activity < threshold -> cannot dream."""
    engine = _build_engine(
        healthy_cortex, memory_engine, active_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    can, reason = await engine._can_dream()
    assert can is False
    assert "user" in reason.lower() or "active" in reason.lower() or "idle" in reason.lower()


# ============================================================================
# TC11: Dream gate rejects VM woken by dream
# ============================================================================


@pytest.mark.asyncio
async def test_dream_gate_rejects_vm_woken_by_dream(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC11: VM uptime < idle threshold means VM was woken for dream, not by user."""
    healthy_cortex.get_snapshot.return_value = _make_snapshot(
        prime_uptime_s=10.0,  # VM just started, not warmed by user
    )
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    can, reason = await engine._can_dream()
    assert can is False
    assert "uptime" in reason.lower() or "warm" in reason.lower()


# ============================================================================
# TC17: Preemption on user activity
# ============================================================================


@pytest.mark.asyncio
async def test_dream_preemption_on_user_activity(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC17: setting preempted event -> job abandoned."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    # Simulate that preemption fires
    engine._preempted.set()

    # _check_preempted should return True
    assert engine._check_preempted() is True

    # Verify the metrics tracker can record preemption
    metrics_tracker.record_preemption()
    m = metrics_tracker.get_metrics()
    assert m.preemptions_count == 1


# ============================================================================
# TC18: Flap damping
# ============================================================================


@pytest.mark.asyncio
async def test_dream_flap_damping(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    tmp_dream_dir,
):
    """TC18: After preemption, cannot re-enter dream for cooldown_s period."""
    config = _make_config(dream_reentry_cooldown_s=60.0)
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )

    # Simulate a recent user return (just happened)
    engine._last_user_return = time.monotonic()

    can, reason = await engine._can_dream()
    assert can is False
    assert "cooldown" in reason.lower() or "flap" in reason.lower()


# ============================================================================
# TC23: Dream prompts capped at 2048 tokens
# ============================================================================


def test_separate_token_budgets(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC23: dream prompt text is capped at DREAM_MAX_PROMPT_CHARS."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    # Access the class constant
    from backend.core.ouroboros.consciousness.dream_engine import DREAM_MAX_PROMPT_CHARS
    assert DREAM_MAX_PROMPT_CHARS == 2048

    # Verify _truncate_prompt actually caps text
    long_text = "x" * 5000
    truncated = engine._truncate_prompt(long_text)
    assert len(truncated) <= DREAM_MAX_PROMPT_CHARS


# ============================================================================
# TC24: Dream dormant reason code
# ============================================================================


@pytest.mark.asyncio
async def test_dream_dormant_reason_code(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
    comm_protocol,
):
    """TC24: prime unavailable -> DREAM_DORMANT reason code via CommProtocol."""
    # Prime is offline
    healthy_cortex.get_snapshot.return_value = _make_snapshot(prime_status="offline")
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
        comm=comm_protocol,
    )

    # Call _emit_dormant directly
    await engine._emit_dormant("prime_unavailable")

    comm_protocol.emit_heartbeat.assert_called_once()
    call_kwargs = comm_protocol.emit_heartbeat.call_args
    # Should contain DREAM_DORMANT in phase
    assert "DREAM_DORMANT" in call_kwargs[1]["phase"] or "DREAM_DORMANT" in str(call_kwargs)


# ============================================================================
# TC29: Direct HTTP, NOT PrimeClient
# ============================================================================


def test_dream_http_direct(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC29: DreamEngine uses aiohttp directly, not PrimeClient or PrimeRouter."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
        jprime_url="http://136.113.252.164:8000",
    )
    # Verify engine stores the URL for direct HTTP
    assert engine._jprime_url == "http://136.113.252.164:8000"
    # Verify it has no PrimeClient/PrimeRouter reference
    assert not hasattr(engine, "_prime_client")
    assert not hasattr(engine, "_prime_router")


# ============================================================================
# TC30: Preemption saves partial state
# ============================================================================


@pytest.mark.asyncio
async def test_preemption_saves_partial(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """TC30: interrupted job info is preserved in _interrupted_jobs for resume."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )

    candidate_info = {
        "repo": "jarvis",
        "repo_sha": "abc123",
        "policy_hash": "pol123",
        "prompt_family": "test_coverage",
        "model_class": "qwen2.5-7b",
    }
    job_key = compute_job_key(
        candidate_info["repo_sha"],
        candidate_info["policy_hash"],
        candidate_info["prompt_family"],
        candidate_info["model_class"],
    )

    engine._save_interrupted(job_key, candidate_info)
    assert job_key in engine._interrupted_jobs
    assert engine._interrupted_jobs[job_key] == candidate_info


# ============================================================================
# Idempotent job key skip
# ============================================================================


@pytest.mark.asyncio
async def test_idempotent_job_key_skip(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """Same job key -> skip computation."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    key = compute_job_key("sha1", "pol1", "coverage", "qwen")
    engine._completed_keys.add(key)

    # Non-stale blueprint exists for this key
    bp = _make_blueprint(blueprint_id=key, repo_sha="sha1", policy_hash="pol1")
    engine._blueprints[key] = bp

    assert engine._is_job_completed(key, current_head="sha1", current_policy_hash="pol1") is True


# ============================================================================
# Blueprints sorted by priority
# ============================================================================


def test_get_blueprints_sorted_by_priority(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """Blueprints returned sorted by priority_score descending."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    bp_low = _make_blueprint(blueprint_id="low", priority=0.3, repo_sha="cur", policy_hash="pol")
    bp_high = _make_blueprint(blueprint_id="high", priority=0.9, repo_sha="cur", policy_hash="pol")
    bp_mid = _make_blueprint(blueprint_id="mid", priority=0.6, repo_sha="cur", policy_hash="pol")

    engine._blueprints["low"] = bp_low
    engine._blueprints["high"] = bp_high
    engine._blueprints["mid"] = bp_mid

    # Provide current head/policy that match so none are stale
    engine._current_head = "cur"
    engine._current_policy_hash = "pol"

    result = engine.get_blueprints(top_n=5)
    assert len(result) == 3
    assert result[0].priority_score == 0.9
    assert result[1].priority_score == 0.6
    assert result[2].priority_score == 0.3


# ============================================================================
# Discard stale removes expired
# ============================================================================


def test_discard_stale_removes_expired(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """discard_stale removes blueprints where repo_sha or policy_hash drifted."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    bp_fresh = _make_blueprint(
        blueprint_id="fresh", repo_sha="current", policy_hash="current_pol",
    )
    bp_stale = _make_blueprint(
        blueprint_id="stale", repo_sha="old_sha", policy_hash="current_pol",
    )

    engine._blueprints["fresh"] = bp_fresh
    engine._blueprints["stale"] = bp_stale
    engine._current_head = "current"
    engine._current_policy_hash = "current_pol"

    removed = engine.discard_stale()
    assert removed == 1
    assert "fresh" in engine._blueprints
    assert "stale" not in engine._blueprints


# ============================================================================
# Dream budget cap
# ============================================================================


@pytest.mark.asyncio
async def test_dream_budget_cap(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    tmp_dream_dir,
):
    """Max minutes exceeded -> cannot dream."""
    config = _make_config(dream_max_minutes_per_day=10.0)
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    # Simulate that we've already used 15 minutes today
    metrics_tracker.record_compute_time(15.0)

    can, reason = await engine._can_dream()
    assert can is False
    assert "budget" in reason.lower() or "minutes" in reason.lower()


# ============================================================================
# Resource governor yield
# ============================================================================


@pytest.mark.asyncio
async def test_resource_governor_yield(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """Resource governor says yield -> cannot dream."""
    gov = MagicMock()
    gov.should_yield = AsyncMock(return_value=True)

    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        gov, metrics_tracker, config, tmp_dream_dir,
    )
    can, reason = await engine._can_dream()
    assert can is False
    assert "resource" in reason.lower() or "yield" in reason.lower()


# ============================================================================
# Stop persists state
# ============================================================================


@pytest.mark.asyncio
async def test_stop_persists_state(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """stop() -> blueprints + completed keys saved to disk."""
    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    bp = _make_blueprint(blueprint_id="persist-test", repo_sha="sha1", policy_hash="pol1")
    engine._blueprints["persist-test"] = bp
    engine._completed_keys.add("key1")

    await engine.stop()

    # Check files were created
    bp_file = tmp_dream_dir / "blueprint_persist-test.json"
    keys_file = tmp_dream_dir / "job_keys.json"

    assert bp_file.exists()
    assert keys_file.exists()

    data = json.loads(bp_file.read_text())
    assert data["blueprint_id"] == "persist-test"

    keys_data = json.loads(keys_file.read_text())
    assert "key1" in keys_data


# ============================================================================
# Start loads state
# ============================================================================


@pytest.mark.asyncio
async def test_start_loads_state(
    healthy_cortex,
    memory_engine,
    idle_monitor,
    resource_governor,
    metrics_tracker,
    config,
    tmp_dream_dir,
):
    """start() -> blueprints + completed keys restored from disk."""
    # Pre-write a blueprint file and job_keys file
    bp = _make_blueprint(blueprint_id="loaded-bp", repo_sha="sha1", policy_hash="pol1")
    bp_data = {
        "blueprint_id": bp.blueprint_id,
        "title": bp.title,
        "description": bp.description,
        "category": bp.category,
        "priority_score": bp.priority_score,
        "target_files": list(bp.target_files),
        "estimated_effort": bp.estimated_effort,
        "estimated_cost_usd": bp.estimated_cost_usd,
        "repo": bp.repo,
        "repo_sha": bp.repo_sha,
        "computed_at_utc": bp.computed_at_utc,
        "ttl_hours": bp.ttl_hours,
        "model_used": bp.model_used,
        "policy_hash": bp.policy_hash,
        "oracle_neighborhood": bp.oracle_neighborhood,
        "suggested_approach": bp.suggested_approach,
        "risk_assessment": bp.risk_assessment,
    }
    bp_file = tmp_dream_dir / "blueprint_loaded-bp.json"
    bp_file.write_text(json.dumps(bp_data))

    keys_file = tmp_dream_dir / "job_keys.json"
    keys_file.write_text(json.dumps(["restored-key-1", "restored-key-2"]))

    engine = _build_engine(
        healthy_cortex, memory_engine, idle_monitor,
        resource_governor, metrics_tracker, config, tmp_dream_dir,
    )
    await engine.start()

    # Verify blueprints were loaded
    assert "loaded-bp" in engine._blueprints
    assert engine._blueprints["loaded-bp"].blueprint_id == "loaded-bp"

    # Verify completed keys were loaded
    assert "restored-key-1" in engine._completed_keys
    assert "restored-key-2" in engine._completed_keys

    # Clean up the dream loop task
    await engine.stop()


# ============================================================================
# Repo-state hydration → candidate picker (Gap 3 live-soak fix)
#
# The live soak proved _pick_candidate was structurally unreachable: it is
# gated on _current_head/_current_policy_hash, which nothing ever populated, so
# DreamEngine produced 0 blueprints and the conception bridge had no event to
# route. These pin the truthful hydration + the guard both ways.
# ============================================================================


def _hydration_engine(tmp_path: Path, repo_path: Any = None):
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    d = tmp_path / "dreams"
    d.mkdir(exist_ok=True)
    return DreamEngine(
        health_cortex=MagicMock(),
        memory_engine=MagicMock(),
        activity_monitor=MockActivityMonitor(idle_seconds=600.0),
        resource_governor=MagicMock(),
        metrics_tracker=DreamMetricsTracker(),
        config=_make_config(),
        persistence_dir=d,
        repo_path=repo_path,
    )


def test_pick_candidate_none_when_state_unhydrated(tmp_path):
    """Mandate 4: missing repo-state → no candidate (the guard clause holds)."""
    eng = _hydration_engine(tmp_path)
    assert eng._current_head == "" and eng._current_policy_hash == ""
    assert eng._pick_candidate() is None


def test_pick_candidate_returns_candidate_when_hydrated(tmp_path):
    """Mandate 4: hydrated repo-state → a native candidate derived from it."""
    eng = _hydration_engine(tmp_path)
    eng._current_head = "deadbeefcafe"
    eng._current_policy_hash = "abc123def4567890"
    cand = eng._pick_candidate()
    assert cand is not None
    assert cand["repo_sha"] == "deadbeefcafe"
    assert cand["policy_hash"] == "abc123def4567890"
    assert cand["repo"] == "jarvis"


def test_hydrate_populates_head_and_policy(tmp_path, monkeypatch):
    """Truthful hydration: real git HEAD via the reused _get_git_head util."""
    import backend.core.ouroboros.consciousness.memory_engine as me
    monkeypatch.setattr(me, "_get_git_head", lambda repo_path=None: "1234abcd5678")
    eng = _hydration_engine(tmp_path)
    eng._hydrate_repo_state()
    assert eng._current_head == "1234abcd5678"
    assert eng._current_policy_hash != ""       # a policy fingerprint was set
    assert eng._pick_candidate() is not None     # end-to-end: guard now passes


def test_hydrate_failsafe_when_git_unavailable(tmp_path, monkeypatch):
    """Mandate 2: git locked/absent (head=None) → prior state kept, no crash,
    candidate stays None (no phantom SHA dreamed about)."""
    import backend.core.ouroboros.consciousness.memory_engine as me
    monkeypatch.setattr(me, "_get_git_head", lambda repo_path=None: None)
    eng = _hydration_engine(tmp_path)
    eng._hydrate_repo_state()                     # must not raise
    assert eng._current_head == ""                # untouched
    assert eng._pick_candidate() is None          # still safe


def test_hydrate_never_raises_on_git_exception(tmp_path, monkeypatch):
    import backend.core.ouroboros.consciousness.memory_engine as me

    def _boom(repo_path=None):
        raise RuntimeError("git index locked")

    monkeypatch.setattr(me, "_get_git_head", _boom)
    eng = _hydration_engine(tmp_path)
    eng._hydrate_repo_state()                     # swallowed, fail-safe
    assert eng._current_head == ""


def test_compute_policy_hash_reads_file(tmp_path):
    (tmp_path / "brain_selection_policy.yaml").write_text("models:\n  a: b\n")
    eng = _hydration_engine(tmp_path, repo_path=str(tmp_path))
    h = eng._compute_policy_hash()
    assert h != "nopolicy" and len(h) == 16       # sha256[:16] of the file


def test_compute_policy_hash_sentinel_when_absent(tmp_path):
    empty = tmp_path / "empty_repo"
    empty.mkdir()                                     # exists but has no policy file
    eng = _hydration_engine(tmp_path, repo_path=str(empty))
    assert eng._compute_policy_hash() == "nopolicy"   # stable, non-empty


def test_compute_policy_hash_honors_env_override(tmp_path, monkeypatch):
    pol = tmp_path / "custom_policy.yaml"
    pol.write_text("x: 1\n")
    monkeypatch.setenv("JARVIS_BRAIN_POLICY_PATH", str(pol))
    eng = _hydration_engine(tmp_path)
    import hashlib as _h
    assert eng._compute_policy_hash() == _h.sha256(pol.read_bytes()).hexdigest()[:16]


def test_hydrate_adaptive_picks_up_advanced_head(tmp_path, monkeypatch):
    """Adaptive: a HEAD that advances between cycles is re-hydrated."""
    import backend.core.ouroboros.consciousness.memory_engine as me
    heads = iter(["head_one", "head_two"])
    monkeypatch.setattr(me, "_get_git_head", lambda repo_path=None: next(heads))
    eng = _hydration_engine(tmp_path)
    eng._hydrate_repo_state()
    assert eng._current_head == "head_one"
    eng._hydrate_repo_state()
    assert eng._current_head == "head_two"        # advanced HEAD honored


# ============================================================================
# RT migration + fail-fast cascade (Phase 1, 2026-07-16)
#
# Dreaming is latency-sensitive; the old Tier-1 used the DW 4-stage BATCH API
# (24h window) and swallowed failures as "" so the Claude tier never fired
# live (bt-2026-07-16-173540: 5 dream jobs, 0 completions). These pin:
# DW-RT (complete_sync) primary, per-tier fail-fast RAISING boundaries,
# Claude-RT fallback actually firing on primary timeout, and typed
# DreamProviderExhaustedError when the whole cascade is down.
# ============================================================================

from backend.core.ouroboros.consciousness.dream_engine import (
    DreamProviderExhaustedError,
)


class _SyncResult:
    def __init__(self, content):
        self.content = content
        self.model = "dw-model-x"
        self.latency_s = 0.5


_BP_JSON = (
    '{"title":"t","description":"d","category":"debt","priority_score":0.7,'
    '"target_files":["a.py"],"estimated_effort":"small","estimated_cost_usd":0.01,'
    '"suggested_approach":"x","risk_assessment":"low"}'
)


def _rt_engine(tmp_path, dw=None, claude=None, jprime_url="", dw_bypass=False):
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    d = tmp_path / "dreams"
    d.mkdir(exist_ok=True)
    eng = DreamEngine(
        health_cortex=MagicMock(),
        memory_engine=MagicMock(),
        activity_monitor=MockActivityMonitor(idle_seconds=600.0),
        resource_governor=MagicMock(),
        metrics_tracker=DreamMetricsTracker(),
        config=_make_config(),
        jprime_url=jprime_url,
        persistence_dir=d,
    )
    eng._dw_provider = dw
    eng._claude_provider = claude
    # The DW-RT health bypass reads the HOST's live surface ledger; a
    # daemon that has logged hundreds of 503s would bypass DW in every
    # test here. Host state is not this helper's input: the bypass is
    # pinned OFF unless a test installs its own surface (`_surface`) and
    # passes ``dw_bypass=None`` to exercise the real predicate.
    if dw_bypass is not None:
        eng._dw_health_bypass = lambda: bool(dw_bypass)
    return eng


async def test_dw_rt_primary_succeeds(tmp_path):
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    eng = _rt_engine(tmp_path, dw=dw, claude=MagicMock())
    result = await eng._call_inference("dream prompt")
    assert result["_inference_provider"] == "doubleword"
    dw.complete_sync.assert_awaited_once()          # RT primitive, not batch
    kwargs = dw.complete_sync.await_args.kwargs
    assert "timeout_s" in kwargs and kwargs["response_format"] == {"type": "json_object"}


async def test_primary_timeout_falls_back_to_claude_rt(tmp_path, monkeypatch):
    """THE mandate-4 test: primary provider timeout → cascade catches it →
    Claude RT tier fires and completes the dream."""
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "5")
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=asyncio.TimeoutError("dw rt timeout"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    result = await eng._call_inference("dream prompt")
    assert result["_inference_provider"] == "claude"    # fallback FIRED
    claude.prompt_only.assert_awaited_once()


async def test_primary_infra_error_falls_back_to_claude_rt(tmp_path):
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=RuntimeError("HTTP 403 entitlement"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    result = await eng._call_inference("dream prompt")
    assert result["_inference_provider"] == "claude"


async def test_hanging_primary_is_bounded_by_wait_for(tmp_path, monkeypatch):
    """A provider that HANGS (the batch-stall class) is cut by the outer
    wait_for and the cascade proceeds — a dream cycle can never wedge."""
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "5")   # floor-clamped to 5s

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    dw = MagicMock()
    dw.complete_sync = _hang
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    result = await asyncio.wait_for(eng._call_inference("p"), timeout=30)
    assert result["_inference_provider"] == "claude"


async def test_full_cascade_exhaustion_raises_typed_error(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "5")
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=RuntimeError("down"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(side_effect=RuntimeError("down too"))
    eng = _rt_engine(tmp_path, dw=dw, claude=claude, jprime_url="")  # no tier 3
    with pytest.raises(DreamProviderExhaustedError):
        await eng._call_inference("p")


async def test_non_json_primary_cascades_to_the_next_tier(tmp_path):
    """A tier that answers with prose is rejected by the gate's shape check
    and the cascade moves on — the behaviour the hand-rolled cascade had."""
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult("not a json object"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    result = await eng._call_inference("dream prompt")
    assert result["_inference_provider"] == "claude"
    dw.complete_sync.assert_awaited_once()


async def test_local_tier_completes_a_dream_when_cloud_is_dead(tmp_path, monkeypatch):
    """The reason the dream cascade moved onto rt_gate: on a host whose only
    lane is the local model, dreams exhausted forever (streak=4, cooldown
    after cooldown, 2026-09-06) while a $0 lane sat idle."""
    import backend.core.ouroboros.claude_fallback as cf
    import backend.core.ouroboros.governance.local_inference_director as lid
    monkeypatch.setattr(cf, "claude_inference",
                        AsyncMock(side_effect=RuntimeError("no key")))
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "5")

    class _Local:
        def __init__(self, cfg, *a, **k):
            self.closed = False

        async def generate(self, prompt, **kw):
            assert kw.get("response_format") == {"type": "json_object"}
            return SimpleNamespace(content=_BP_JSON)

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(lid, "LocalPrimeClient", _Local)
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=RuntimeError("down"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(side_effect=RuntimeError("down too"))
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    result = await eng._call_inference("dream prompt")
    assert result["_inference_provider"] == "local"
    assert result["title"] == "t"


def test_the_legacy_http_tier_is_gone():
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    assert not hasattr(DreamEngine, "_call_jprime_legacy")


async def test_run_dream_job_handles_exhaustion_with_dormant(tmp_path):
    """_run_dream_job catches the typed exhaustion, emits DREAM_DORMANT, and
    returns None — the dream loop survives a total provider outage."""
    comm = MagicMock()
    comm.emit_heartbeat = AsyncMock()
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=RuntimeError("down"))
    eng = _rt_engine(tmp_path, dw=dw, claude=None)
    eng._comm = comm
    eng._current_head = "h" * 12
    eng._current_policy_hash = "p" * 16
    result = await eng._run_dream_job()
    assert result is None
    comm.emit_heartbeat.assert_awaited()               # dormant surfaced


async def test_no_batch_api_on_dream_path(tmp_path):
    """Structural: the dream path never touches prompt_only/submit_batch on DW."""
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    eng = _rt_engine(tmp_path, dw=dw, claude=MagicMock())
    await eng._call_inference("p")
    dw.prompt_only.assert_not_called()
    dw.submit_batch.assert_not_called()


def test_rt_timeout_env_parsing(monkeypatch):
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    monkeypatch.delenv("JARVIS_DREAM_RT_TIMEOUT_S", raising=False)
    assert DreamEngine._dream_rt_timeout_s() == 90.0
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "45")
    assert DreamEngine._dream_rt_timeout_s() == 45.0
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "bogus")
    assert DreamEngine._dream_rt_timeout_s() == 90.0
    monkeypatch.setenv("JARVIS_DREAM_RT_TIMEOUT_S", "1")
    assert DreamEngine._dream_rt_timeout_s() == 5.0    # floor


# ============================================================================
# Dynamic candidate hydration (2026-07-17)
#
# The stub hardcoded prompt_family="general_improvement" / model_class=
# "qwen2.5-7b" (the latter a lie — no soak ever served a dream on it), so
# compute_job_key had exactly ONE possible identity per HEAD: dream once, then
# correctly skip forever ("Job ... already completed, skipping" x N). The hash
# and the dedup registry are SOUND and untouched; the fix is to give the key
# genuine, organically-derived diversity.
# ============================================================================

from backend.core.ouroboros.consciousness.dream_engine import (
    _MODEL_CLASS_UNKNOWN,
    _PROMPT_FAMILY_NEUTRAL,
)
from backend.core.ouroboros.consciousness.types import compute_job_key


def _insight(category, *, expired=False, evidence=5):
    i = MagicMock()
    i.category = category
    i.evidence_count = evidence
    i.is_expired = MagicMock(return_value=expired)
    return i


def _mem(*insights):
    m = MagicMock()
    s = MagicMock()
    s.top_patterns = tuple(insights)
    m.get_pattern_summary.return_value = s
    return m


def _cand_engine(tmp_path, *, memory=None, dw=None, claude=None, repo=None):
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    d = tmp_path / "dreams"
    d.mkdir(exist_ok=True)
    eng = DreamEngine(
        health_cortex=MagicMock(),
        memory_engine=memory if memory is not None else _mem(),
        activity_monitor=MockActivityMonitor(idle_seconds=600.0),
        resource_governor=MagicMock(),
        metrics_tracker=DreamMetricsTracker(),
        config=_make_config(),
        persistence_dir=d,
        repo_path=str(repo) if repo else None,
    )
    eng._dw_provider = dw
    eng._claude_provider = claude
    eng._current_head = "headsha1234"
    eng._current_policy_hash = "policyhash01"
    return eng


def _provider(model):
    p = MagicMock()
    p._model = model
    return p


# ---- prompt_family derives from live memory telemetry ----------------------


def test_prompt_family_tracks_dominant_failure_pattern(tmp_path):
    eng = _cand_engine(tmp_path, memory=_mem(_insight("failure_pattern")))
    assert eng._derive_prompt_family() == "failure_repair"


def test_prompt_family_tracks_fragility(tmp_path):
    eng = _cand_engine(tmp_path, memory=_mem(_insight("file_fragility")))
    assert eng._derive_prompt_family() == "fragility_hardening"


def test_prompt_family_tracks_success_pattern(tmp_path):
    eng = _cand_engine(tmp_path, memory=_mem(_insight("success_pattern")))
    assert eng._derive_prompt_family() == "success_extension"


def test_prompt_family_skips_expired_insights(tmp_path):
    """An expired insight is not a live focus — the engine's own TTL governs."""
    eng = _cand_engine(tmp_path, memory=_mem(
        _insight("failure_pattern", expired=True),
        _insight("file_fragility", expired=False),
    ))
    assert eng._derive_prompt_family() == "fragility_hardening"


def test_prompt_family_unknown_category_passes_through(tmp_path):
    """A NEW MemoryEngine category isn't silently swallowed by our map."""
    eng = _cand_engine(tmp_path, memory=_mem(_insight("perf_regression")))
    assert eng._derive_prompt_family() == "perf_regression"


def test_prompt_family_cold_memory_is_neutral(tmp_path):
    """Fresh organism: no insights → honest neutral focus, not a fabrication."""
    assert _cand_engine(tmp_path, memory=_mem())._derive_prompt_family() == _PROMPT_FAMILY_NEUTRAL


def test_prompt_family_never_raises_on_broken_memory(tmp_path):
    m = MagicMock()
    m.get_pattern_summary.side_effect = RuntimeError("memory down")
    assert _cand_engine(tmp_path, memory=m)._derive_prompt_family() == _PROMPT_FAMILY_NEUTRAL


# ---- model_class derives from the live provider topology -------------------


def test_model_class_reflects_dw_primary(tmp_path):
    eng = _cand_engine(tmp_path, dw=_provider("Qwen/Qwen3.5-397B-A17B-FP8"),
                       claude=_provider("claude-sonnet-4-6"))
    assert eng._derive_model_class() == "dw:Qwen/Qwen3.5-397B-A17B-FP8"  # mirrors tier order


def test_model_class_falls_to_claude_when_dw_absent(tmp_path):
    eng = _cand_engine(tmp_path, dw=None, claude=_provider("claude-sonnet-4-6"))
    assert eng._derive_model_class() == "claude:claude-sonnet-4-6"


def test_model_class_unwired_when_no_providers(tmp_path):
    assert _cand_engine(tmp_path)._derive_model_class() == _MODEL_CLASS_UNKNOWN


def test_model_class_never_hardcodes_the_stub_slug(tmp_path):
    """The stub's qwen2.5-7b lie must never reappear."""
    eng = _cand_engine(tmp_path, dw=_provider("Qwen/Qwen3.5-397B-A17B-FP8"))
    assert "qwen2.5-7b" not in eng._derive_model_class()


# ---- the payoff: DIVERSE job keys, hash untouched --------------------------


def test_candidate_carries_derived_axes(tmp_path):
    eng = _cand_engine(tmp_path, memory=_mem(_insight("failure_pattern")),
                       dw=_provider("dw-model-x"))
    c = eng._pick_candidate()
    assert c["prompt_family"] == "failure_repair"
    assert c["model_class"] == "dw:dw-model-x"
    assert c["repo_sha"] == "headsha1234" and c["policy_hash"] == "policyhash01"


def test_shifting_memory_yields_DISTINCT_job_keys_on_static_head(tmp_path):
    """THE fix: with HEAD frozen, a shift in live telemetry re-keys the job —
    so a genuinely new dream is warranted and the (correct) dedup skip no
    longer starves the DW-RT tier."""
    keys = set()
    for cat in ("failure_pattern", "file_fragility", "success_pattern"):
        eng = _cand_engine(tmp_path, memory=_mem(_insight(cat)), dw=_provider("m"))
        c = eng._pick_candidate()
        keys.add(compute_job_key(c["repo_sha"], c["policy_hash"],
                                 c["prompt_family"], c["model_class"]))
    assert len(keys) == 3          # same HEAD, three distinct dreamable jobs


def test_provider_topology_shift_rekeys_job(tmp_path):
    """A topology change (entitlement/pin/outage) legitimately re-keys."""
    a = _cand_engine(tmp_path, dw=_provider("model-a"))._pick_candidate()
    b = _cand_engine(tmp_path, dw=_provider("model-b"))._pick_candidate()
    ka = compute_job_key(a["repo_sha"], a["policy_hash"], a["prompt_family"], a["model_class"])
    kb = compute_job_key(b["repo_sha"], b["policy_hash"], b["prompt_family"], b["model_class"])
    assert ka != kb


def test_identical_state_is_STILL_deduped(tmp_path):
    """Bulletproof: unchanged state must remain idempotent — no runaway
    re-dreaming (the OOM/cost failure mode)."""
    k = []
    for _ in range(3):
        eng = _cand_engine(tmp_path, memory=_mem(_insight("failure_pattern")), dw=_provider("m"))
        c = eng._pick_candidate()
        k.append(compute_job_key(c["repo_sha"], c["policy_hash"],
                                 c["prompt_family"], c["model_class"]))
    assert len(set(k)) == 1        # stable identity — the cache still works


def test_repo_name_from_live_path_not_hardcoded(tmp_path):
    eng = _cand_engine(tmp_path, repo=tmp_path / "my-repo")
    assert eng._pick_candidate()["repo"] == "my-repo"


def test_pick_candidate_still_none_without_repo_state(tmp_path):
    eng = _cand_engine(tmp_path)
    eng._current_head = ""
    assert eng._pick_candidate() is None


# ============================================================================
# Output-budget decoupling + empty-tier observability (2026-07-17)
#
# bt-2026-07-17-033933: DW returned "0 chars, 30.25s, $0.00010". Two defects:
#   (1) DREAM_MAX_PROMPT_CHARS (an INPUT char cap, TC23) was passed as the
#       provider's max_tokens (an OUTPUT budget) — a type error. The only
#       entitled model (Qwen3.5-397B) has an effort FLOOR of "low" so it ALWAYS
#       reasons; 2048 output tokens went entirely to chain-of-thought.
#   (2) the empty response logged NOTHING and fell through silently, so the
#       cascade looked like DW was never attempted.
# ============================================================================

import logging

from backend.core.ouroboros.consciousness.dream_engine import (
    DREAM_MAX_PROMPT_CHARS,
    dream_max_output_tokens,
)


def test_output_budget_is_decoupled_from_prompt_char_cap(monkeypatch):
    """The type error must not regrow: the OUTPUT budget is its own knob."""
    monkeypatch.delenv("JARVIS_DREAM_MAX_OUTPUT_TOKENS", raising=False)
    assert dream_max_output_tokens() == 8192
    assert dream_max_output_tokens() != DREAM_MAX_PROMPT_CHARS   # never conflated
    assert DREAM_MAX_PROMPT_CHARS == 2048                        # input cap unchanged (TC23)


def test_output_budget_env_tunable_with_floor(monkeypatch):
    monkeypatch.setenv("JARVIS_DREAM_MAX_OUTPUT_TOKENS", "16384")
    assert dream_max_output_tokens() == 16384
    monkeypatch.setenv("JARVIS_DREAM_MAX_OUTPUT_TOKENS", "64")   # would re-create truncation
    assert dream_max_output_tokens() == 1024                     # floor protects
    monkeypatch.setenv("JARVIS_DREAM_MAX_OUTPUT_TOKENS", "junk")
    assert dream_max_output_tokens() == 8192


async def test_dw_tier_sends_the_output_budget_not_prompt_chars(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DREAM_MAX_OUTPUT_TOKENS", "8192")
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    eng = _rt_engine(tmp_path, dw=dw, claude=MagicMock())
    await eng._call_inference("p")
    sent = dw.complete_sync.await_args.kwargs["max_tokens"]
    assert sent == 8192
    assert sent != DREAM_MAX_PROMPT_CHARS      # the 2048 truncation is dead


async def test_empty_dw_response_logs_and_still_cascades(tmp_path, caplog):
    """Mandate 2: an empty tier response emits a DISTINCT diagnostic and the
    fallback still fires — no silent fall-through."""
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(""))   # the live defect
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw=dw, claude=claude)
    with caplog.at_level(logging.INFO):
        result = await eng._call_inference("p")
    assert result["_inference_provider"] == "claude"             # cascade fired
    assert any("exhausted/empty" in r.message or "exhausted/empty" in r.getMessage()
               for r in caplog.records), "empty DW response must log a distinct diagnostic"


async def test_empty_claude_response_also_logs(tmp_path, caplog):
    dw = MagicMock()
    dw.complete_sync = AsyncMock(side_effect=RuntimeError("dw down"))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value="")
    eng = _rt_engine(tmp_path, dw=dw, claude=claude, jprime_url="")
    with caplog.at_level(logging.INFO):
        with pytest.raises(DreamProviderExhaustedError):
            await eng._call_inference("p")
    assert any("exhausted/empty" in r.getMessage() for r in caplog.records)


# ============================================================================
# Heartbeat-driven DW bypass — the timeout-tax kill (2026-07-17)
#
# bt-2026-07-17-074626: DW answered HTTP 502 (upstream_unreachable) and EVERY
# dream paid ~53s of dead-tier latency before cascading to Claude — DW's
# latency AND Claude's price. The heartbeat (default-OFF) already knows DW's
# health; the cascade must READ that state, not re-probe.
# ============================================================================

from backend.core.ouroboros.governance.dw_surface_health import (
    SurfaceHealthLedger as _SHL,
    SurfaceKind as _SK,
    SurfaceVerdict as _SV,
)


def _surface(monkeypatch, tmp_path, verdict, *, times=1):
    """Drive the REAL ledger — the fake must mirror the real contract. The
    bypass reads DIRECT_COMPLETION, NOT the SSE heartbeat (which is chronically
    degraded by design and must never contaminate this tier)."""
    lg = _SHL(path=tmp_path / "surf.json", autosave=False)
    monkeypatch.setattr(
        "backend.core.ouroboros.governance.dw_surface_health.SurfaceHealthLedger",
        lambda *a, **k: lg,
    )
    for _ in range(times):
        lg.record(_SK.DIRECT_COMPLETION, verdict)
    return lg


async def test_degraded_dw_is_bypassed_instantly_no_timeout_tax(tmp_path, monkeypatch):
    """THE fix: DIRECT_COMPLETION degraded -> DW-RT never called -> Claude
    serves with ZERO dead-tier latency."""
    _surface(monkeypatch, tmp_path, _SV.UPSTREAM_DEGRADED, times=3)
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    claude = MagicMock()
    claude.prompt_only = AsyncMock(return_value=_BP_JSON)
    eng = _rt_engine(tmp_path, dw_bypass=None, dw=dw, claude=claude)
    result = await eng._call_inference("p")
    assert result["_inference_provider"] == "claude"
    dw.complete_sync.assert_not_called()          # the 53s tax is gone


async def test_healthy_completion_surface_still_attempts_dw(tmp_path, monkeypatch):
    """Bypass must not strand the cheap tier when the surface is healthy."""
    _surface(monkeypatch, tmp_path, _SV.HEALTHY)
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    eng = _rt_engine(tmp_path, dw_bypass=None, dw=dw, claude=MagicMock())
    result = await eng._call_inference("p")
    assert result["_inference_provider"] == "doubleword"
    dw.complete_sync.assert_awaited_once()


def test_sse_degradation_never_contaminates_the_completion_tier(tmp_path, monkeypatch):
    """The bug this replaced: 254 SSE failures bypassed a tier built to AVOID
    SSE. Surfaces are independent health domains."""
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    lg = _surface(monkeypatch, tmp_path, _SV.HEALTHY)
    for _ in range(254):
        lg.record(_SK.DIRECT_STREAMING, _SV.UPSTREAM_DEGRADED)
    assert DreamEngine._dw_health_bypass() is False


def test_bypass_fail_soft_never_strands_dw(tmp_path, monkeypatch):
    """A telemetry fault must not permanently disable the cheap tier."""
    monkeypatch.setattr(
        "backend.core.ouroboros.governance.dw_surface_health.SurfaceHealthLedger",
        MagicMock(side_effect=RuntimeError("ledger exploded")),
    )
    from backend.core.ouroboros.consciousness.dream_engine import DreamEngine
    assert DreamEngine._dw_health_bypass() is False   # attempt DW normally


async def test_recovery_tracer_fires_only_when_bypassed(tmp_path, monkeypatch):
    """Closes the ONE-WAY DOOR: a bypass silences organic traffic, so the
    tracer is the only thing that can observe DW healing. It must NOT probe a
    healthy surface (cost proportional to the outage)."""
    import backend.core.ouroboros.governance.dw_capacity_probe as _cap
    calls = []

    async def _fake_trace(provider, *, model=None):
        calls.append(model)
        return "healthy"

    monkeypatch.setattr(_cap, "trace_direct_completion", _fake_trace)

    _surface(monkeypatch, tmp_path, _SV.HEALTHY)
    eng = _rt_engine(tmp_path, dw_bypass=None, dw=MagicMock(), claude=MagicMock())
    await eng._trace_dw_recovery_if_bypassed()
    assert calls == []                              # healthy -> no probe cost

    _surface(monkeypatch, tmp_path, _SV.UPSTREAM_DEGRADED, times=3)
    await eng._trace_dw_recovery_if_bypassed()
    assert len(calls) == 1                          # degraded -> probe for recovery


# ============================================================================
# Exhaustion backoff (2026-07-23): a dead cascade is one fact, not a
# per-30s warning storm
# ============================================================================


def _exhaustion_job_engine(tmp_path, monkeypatch, streaks_raise=True):
    """A lean engine whose _run_dream_job reaches _call_inference
    directly (candidate/dedup/preemption seams pinned)."""
    from backend.core.ouroboros.consciousness.dream_engine import (
        DreamProviderExhaustedError,
    )
    eng = _rt_engine(tmp_path)
    monkeypatch.setattr(eng, "_hydrate_repo_state", lambda: None)
    monkeypatch.setattr(
        eng, "_trace_dw_recovery_if_bypassed", AsyncMock(),
    )
    monkeypatch.setattr(eng, "_pick_candidate", lambda: {
        "repo_sha": "a" * 40, "policy_hash": "b" * 8,
        "prompt_family": "f", "model_class": "m",
        "target": "backend/x.py", "reason": "test",
    })
    monkeypatch.setattr(
        eng, "_is_job_completed", lambda *a, **k: False,
    )
    monkeypatch.setattr(eng, "_check_preempted", lambda: False)
    monkeypatch.setattr(eng, "_build_dream_prompt", lambda c: "p", raising=False)
    if streaks_raise:
        monkeypatch.setattr(
            eng, "_call_inference",
            AsyncMock(side_effect=DreamProviderExhaustedError("all down")),
        )
    return eng


@pytest.mark.asyncio
async def test_exhaustion_backoff_escalates_and_quiets(
    tmp_path, monkeypatch, caplog,
):
    """Consecutive full-cascade exhaustions double the cooldown; only the
    FIRST is a WARNING (the rest are INFO — the log-storm kill)."""
    import logging
    monkeypatch.setenv("JARVIS_DREAM_EXHAUSTION_BACKOFF_BASE_S", "60")
    monkeypatch.setenv("JARVIS_DREAM_EXHAUSTION_BACKOFF_MAX_S", "1800")
    eng = _exhaustion_job_engine(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO):
        t0 = time.monotonic()
        assert await eng._run_dream_job() is None
        assert eng._exhaustion_streak == 1
        assert 55 <= eng._exhaustion_until - t0 <= 65      # ~60s
        assert await eng._run_dream_job() is None
        assert eng._exhaustion_streak == 2
        assert 115 <= eng._exhaustion_until - time.monotonic() <= 125  # ~120s
    warns = [r for r in caplog.records
             if "cascade exhausted" in r.message and r.levelname == "WARNING"]
    infos = [r for r in caplog.records
             if "cascade exhausted" in r.message and r.levelname == "INFO"]
    assert len(warns) == 1 and len(infos) == 1


@pytest.mark.asyncio
async def test_exhaustion_backoff_caps_and_resets_on_success(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("JARVIS_DREAM_EXHAUSTION_BACKOFF_BASE_S", "60")
    monkeypatch.setenv("JARVIS_DREAM_EXHAUSTION_BACKOFF_MAX_S", "300")
    eng = _exhaustion_job_engine(tmp_path, monkeypatch)
    for _ in range(6):
        await eng._run_dream_job()
    assert eng._exhaustion_until - time.monotonic() <= 305  # capped, not 1920s
    # A recovered cascade resets the streak instantly.
    dw = MagicMock()
    dw.complete_sync = AsyncMock(return_value=_SyncResult(_BP_JSON))
    eng2 = _rt_engine(tmp_path, dw=dw, claude=MagicMock())
    eng2._exhaustion_streak = 5
    eng2._exhaustion_until = time.monotonic() + 999
    monkeypatch.setattr(eng2, "_hydrate_repo_state", lambda: None)
    monkeypatch.setattr(eng2, "_trace_dw_recovery_if_bypassed", AsyncMock())
    monkeypatch.setattr(eng2, "_pick_candidate", lambda: {
        "repo_sha": "a" * 40, "policy_hash": "b" * 8,
        "prompt_family": "f", "model_class": "m",
        "target": "backend/x.py", "reason": "test",
    })
    monkeypatch.setattr(eng2, "_is_job_completed", lambda *a, **k: False)
    monkeypatch.setattr(eng2, "_check_preempted", lambda: False)
    await eng2._run_dream_job()
    assert eng2._exhaustion_streak == 0
    assert eng2._exhaustion_until == 0.0
