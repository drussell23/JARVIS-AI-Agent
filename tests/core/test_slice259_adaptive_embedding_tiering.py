"""Slice 259 — Adaptive Model Tiering Engine (Sovereign Adaptive Memory Matrix).

The EmbeddingService runs on one of two tiers and adapts to host memory:
  * HIGH  — PyTorch SentenceTransformer (~800MB)
  * LITE  — fastembed / ONNX-CoreML (~200MB), dimension-compatible

Pins:
  §1  HIGH loads when the memory budget is granted
  §2  HIGH denial → graceful DEMOTION to the LITE tier (not "no embeddings")
  §3  torch absent → DEMOTION to LITE
  §4  both tiers unaffordable → full degradation (no model), tier stays NONE
  §5  PROMOTION LITE→HIGH once headroom returns, with hysteresis
  §6  no promotion without headroom / from HIGH / when disabled
  §7  observability surfaces the active tier
"""
from __future__ import annotations

import asyncio

import pytest

import backend.core.embedding_service as es
from backend.core.embedding_service import EmbeddingServiceConfig, EmbeddingTier


# ── helpers ─────────────────────────────────────────────────────────────
class _FakeModel:
    def __init__(self, tag: str):
        self.tag = tag


async def _granted(*_a, **_k):
    return True


async def _denied(*_a, **_k):
    return False


def _fresh(monkeypatch, **cfg_overrides):
    """A clean, singleton-reset EmbeddingService with the broker forced off
    (so the deterministic legacy budget gate runs) and promotion off by
    default (tests that exercise promotion re-enable it)."""
    import backend.core.memory_budget_broker as mbb
    monkeypatch.setattr(mbb, "get_memory_budget_broker", lambda: None, raising=False)
    cfg = EmbeddingServiceConfig.from_env()
    cfg.promotion_enabled = cfg_overrides.pop("promotion_enabled", False)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    es.EmbeddingService._instance = None
    return es.EmbeddingService(config=cfg)


# ── §1 HIGH loads when granted ──────────────────────────────────────────
def test_high_tier_loads_when_budget_granted(monkeypatch):
    svc = _fresh(monkeypatch)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)
    monkeypatch.setattr(svc, "_load_sentence_transformer", lambda: _FakeModel("high"))
    assert asyncio.run(svc._load_model()) is True
    assert svc.active_tier == EmbeddingTier.HIGH
    assert svc._model.tag == "high"


# ── §2 demotion on budget denial ────────────────────────────────────────
def test_demotes_to_lite_on_budget_denial(monkeypatch):
    svc = _fresh(monkeypatch)
    monkeypatch.setattr(svc, "_check_memory_budget", _denied)            # deny HIGH
    monkeypatch.setattr(svc, "_lite_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_make_fastembed_model", lambda factory=None: _FakeModel("lite"))
    assert asyncio.run(svc._load_model()) is True
    assert svc.active_tier == EmbeddingTier.LITE
    assert svc._model.tag == "lite"
    assert svc._tier_transitions == 1


# ── §3 torch absent → demote ────────────────────────────────────────────
def test_demotes_when_torch_absent(monkeypatch):
    svc = _fresh(monkeypatch)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)           # budget OK
    def _no_torch():
        raise ImportError("No module named 'sentence_transformers'")
    monkeypatch.setattr(svc, "_load_sentence_transformer", _no_torch)
    monkeypatch.setattr(svc, "_lite_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_make_fastembed_model", lambda factory=None: _FakeModel("lite"))
    assert asyncio.run(svc._load_model()) is True
    assert svc.active_tier == EmbeddingTier.LITE


# ── §4 full degradation ─────────────────────────────────────────────────
def test_full_degradation_when_no_tier_affordable(monkeypatch):
    svc = _fresh(monkeypatch)
    monkeypatch.setattr(svc, "_check_memory_budget", _denied)            # deny HIGH
    monkeypatch.setattr(svc, "_lite_headroom_available", lambda: False)  # deny LITE
    assert asyncio.run(svc._load_model()) is False
    assert svc.active_tier == EmbeddingTier.NONE
    assert svc._model is None


# ── §5 promotion with hysteresis ────────────────────────────────────────
def test_promotes_lite_to_high_after_stable_headroom(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True, promotion_stable_checks=2)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)
    monkeypatch.setattr(svc, "_load_sentence_transformer", lambda: _FakeModel("high"))

    # 1st observation: hysteresis not satisfied yet.
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert svc.active_tier == EmbeddingTier.LITE
    assert svc._promotion_stable_count == 1
    # 2nd observation: promote.
    assert asyncio.run(svc.maybe_promote_tier()) is True
    assert svc.active_tier == EmbeddingTier.HIGH
    assert svc._model.tag == "high"


def test_headroom_flap_resets_hysteresis(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True, promotion_stable_checks=2)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)
    monkeypatch.setattr(svc, "_load_sentence_transformer", lambda: _FakeModel("high"))

    headroom = {"v": True}
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: headroom["v"])
    assert asyncio.run(svc.maybe_promote_tier()) is False  # count 1
    headroom["v"] = False
    assert asyncio.run(svc.maybe_promote_tier()) is False  # resets to 0
    assert svc._promotion_stable_count == 0
    assert svc.active_tier == EmbeddingTier.LITE


# ── §6 promotion guards ─────────────────────────────────────────────────
def test_no_promotion_without_headroom(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: False)
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert svc.active_tier == EmbeddingTier.LITE


def test_no_promotion_when_already_high(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True)
    svc._model = _FakeModel("high")
    svc._active_tier = EmbeddingTier.HIGH
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert svc.active_tier == EmbeddingTier.HIGH


def test_promotion_respects_disable_flag(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=False)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: True)
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert svc.active_tier == EmbeddingTier.LITE


def test_promotion_high_load_failure_stays_on_lite(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True, promotion_stable_checks=1)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)
    def _boom():
        raise RuntimeError("OOM during load")
    monkeypatch.setattr(svc, "_load_sentence_transformer", _boom)
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert svc.active_tier == EmbeddingTier.LITE  # never lost the working LITE model
    assert svc._model.tag == "lite"
    assert svc._promotion_unavailable_reason is None   # transient: keep polling


def test_a_missing_package_trips_the_promotion_breaker(monkeypatch):
    """An ImportError is not a transient condition. Once the HIGH tier's
    package is known absent, promotion is disabled for the life of the
    process: no further loads, no further budget reservations, no further
    log lines — and the observability surface says why."""
    svc = _fresh(monkeypatch, promotion_enabled=True, promotion_stable_checks=1)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)
    calls = []

    def _absent():
        calls.append(1)
        raise ModuleNotFoundError("No module named 'sentence_transformers'")

    monkeypatch.setattr(svc, "_load_sentence_transformer", _absent)
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert "sentence_transformers" in (svc._promotion_unavailable_reason or "")
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert len(calls) == 1                                  # never retried
    assert svc.tier_status()["promotion_unavailable_reason"]
    assert svc.active_tier == EmbeddingTier.LITE and svc._model.tag == "lite"


def test_a_wrapped_import_fault_still_trips_the_breaker(monkeypatch):
    svc = _fresh(monkeypatch, promotion_enabled=True, promotion_stable_checks=1)
    svc._model = _FakeModel("lite")
    svc._active_tier = EmbeddingTier.LITE
    monkeypatch.setattr(svc, "_pytorch_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)

    def _wrapped():
        try:
            raise ImportError("No module named 'torch'")
        except ImportError as exc:
            raise RuntimeError("constructor failed") from exc

    monkeypatch.setattr(svc, "_load_sentence_transformer", _wrapped)
    assert asyncio.run(svc.maybe_promote_tier()) is False
    assert "torch" in (svc._promotion_unavailable_reason or "")


def test_torch_absent_at_boot_never_starts_the_poller(monkeypatch):
    """§3 demotion already trips the breaker, so the LITE→HIGH poller is
    never scheduled for a package that is not there."""
    svc = _fresh(monkeypatch, promotion_enabled=True)
    monkeypatch.setattr(svc, "_check_memory_budget", _granted)

    def _no_torch():
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(svc, "_load_sentence_transformer", _no_torch)
    monkeypatch.setattr(svc, "_lite_headroom_available", lambda: True)
    monkeypatch.setattr(svc, "_make_fastembed_model", lambda factory=None: _FakeModel("lite"))

    async def _go():
        await svc._load_model()
        svc._ensure_promotion_loop()
        return svc._promotion_task

    task = asyncio.run(_go())
    assert svc._promotion_unavailable_reason
    assert task is None


# ── §7 observability ────────────────────────────────────────────────────
def test_tier_status_and_stats_reflect_tier(monkeypatch):
    svc = _fresh(monkeypatch)
    svc._active_tier = EmbeddingTier.LITE
    status = svc.tier_status()
    assert status["active_tier"] == "LITE"
    assert "promotion_loop_running" in status
    assert svc.get_stats()["active_tier"] == "LITE"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
