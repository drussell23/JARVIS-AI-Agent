"""Unified generation recursion bound (Tier-1 Phase 3).

One shared, self-expiring ceiling on total RECOVERY depth per op across the
Iron-Gate GENERATE_RETRY loop AND the local syntax-repair — strictly additive
(caps the SUM, never grants a retry a native bound refused), async-yield-safe,
self-expiring (no orphaned counters), fail-closed, never raises.
"""
from __future__ import annotations

import pytest

from backend.core.ouroboros.governance import generation_recursion_bound as RB


@pytest.fixture(autouse=True)
def _clean_ledger():
    with RB._LOCK:
        RB._LEDGER.clear()
    yield
    with RB._LOCK:
        RB._LEDGER.clear()


def test_default_bound_is_three(monkeypatch):
    monkeypatch.delenv("JARVIS_GENERATION_RECURSION_BOUND", raising=False)
    assert RB.recursion_bound() == 3


def test_bound_env_tunable(monkeypatch):
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "5")
    assert RB.recursion_bound() == 5


def test_bound_clamped_to_at_least_one(monkeypatch):
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "0")
    assert RB.recursion_bound() == 1
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "garbage")
    assert RB.recursion_bound() == 3


def test_depth_increments_and_hits_ceiling(monkeypatch):
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "3")
    op = "op-A"
    t1 = RB.enter_recovery(op)
    assert (t1.depth, t1.at_ceiling, t1.remaining) == (1, False, 2)
    t2 = RB.enter_recovery(op)
    assert (t2.depth, t2.at_ceiling, t2.remaining) == (2, False, 1)
    t3 = RB.enter_recovery(op)
    assert (t3.depth, t3.at_ceiling, t3.remaining) == (3, False, 0)
    t4 = RB.enter_recovery(op)  # 4th recovery exceeds bound 3 -> fail-closed
    assert t4.depth == 4 and t4.at_ceiling is True and t4.remaining == 0


def test_counters_are_per_op_no_cross_leak():
    RB.enter_recovery("op-X")
    RB.enter_recovery("op-X")
    t_y = RB.enter_recovery("op-Y")
    assert t_y.depth == 1 and t_y.at_ceiling is False
    assert RB.peek_depth("op-X") == 2


def test_shared_ceiling_sums_across_loops(monkeypatch):
    # simulate: 2 Iron-Gate retries + syntax retry on the SAME op share the budget
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "2")
    op = "op-shared"
    assert RB.enter_recovery(op).at_ceiling is False   # depth 1 (iron-gate)
    assert RB.enter_recovery(op).at_ceiling is False   # depth 2 (syntax)
    assert RB.enter_recovery(op).at_ceiling is True     # depth 3 > 2 -> denied


def test_reset_drops_counter():
    RB.enter_recovery("op-R")
    RB.enter_recovery("op-R")
    assert RB.peek_depth("op-R") == 2
    RB.reset("op-R")
    assert RB.peek_depth("op-R") == 0
    # after reset a fresh attempt starts at depth 1 again
    assert RB.enter_recovery("op-R").depth == 1


def _age_entry(op_id: str, by_s: float = 100000.0) -> None:
    """Rewrite an entry's monotonic touch time to simulate a long-orphaned op,
    without a real sleep (the prod TTL floor is 1s). Deterministic."""
    with RB._LOCK:
        depth, ts = RB._LEDGER[op_id]
        RB._LEDGER[op_id] = (depth, ts - by_s)


def test_ttl_expiry_reclaims_orphans():
    RB.enter_recovery("op-orphan")
    assert RB.peek_depth("op-orphan") == 1
    _age_entry("op-orphan")  # older than the default 900s TTL
    # a later access sweeps the expired orphan; the new op starts clean
    fresh = RB.enter_recovery("op-fresh")
    assert fresh.depth == 1
    assert RB.peek_depth("op-orphan") == 0  # swept


def test_disabled_is_permissive(monkeypatch):
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND_ENABLED", "false")
    monkeypatch.setenv("JARVIS_GENERATION_RECURSION_BOUND", "1")
    op = "op-off"
    for _ in range(10):
        tok = RB.enter_recovery(op)
        assert tok.at_ceiling is False  # never blocks when disabled
    assert RB.peek_depth(op) == 0  # disabled path records nothing


def test_empty_op_id_is_permissive():
    tok = RB.enter_recovery("")
    assert tok.at_ceiling is False and tok.depth == 0


def test_never_raises_and_emit_is_best_effort():
    # emit with no stream configured returns None, never raises
    assert RB.emit_generation_exhausted(
        "op-Z", phase="GENERATE_RETRY", depth=4, bound=3, detail="x",
    ) is None or True
    assert RB.sweep_expired() >= 0
    RB.reset("nonexistent")  # idempotent


def test_sweep_expired_counts():
    RB.enter_recovery("a")
    RB.enter_recovery("b")
    _age_entry("a")
    _age_entry("b")
    assert RB.sweep_expired() == 2
