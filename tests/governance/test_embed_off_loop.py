"""The event loop never runs the embedder.

Measured 2026-09-06 (bt-2026-09-06-074921): two 5 s ``STUCK_FRAME`` stalls
of the main thread inside ``onnxruntime … run`` — the semantic index's
synchronous ``score``/``boost_for``/``score_with_cluster`` called from
loop-side code. Every embed now passes one guarded seam that refuses on
the loop thread (counted, warned once per caller) and passes on a worker
thread, where the ``*_offloaded`` coroutines run it.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from backend.core.ouroboros.governance import semantic_index as si
from tests.governance.test_semantic_index import (  # noqa: F401  (fixture)
    _enable,
    _fake_vec,
    _new_index_with_fake_embedder,
    _reset_env_and_singletons,
)


def _warm(idx) -> None:
    with idx._lock:
        idx._centroid = _fake_vec("direction-A")
        idx._built_at = time.time()


@pytest.fixture(autouse=True)
def _forget_warnings():
    si._ON_LOOP_WARNED.clear()
    yield
    si._ON_LOOP_WARNED.clear()


def test_the_sync_api_refuses_to_embed_on_the_loop_thread(monkeypatch, tmp_path, caplog):
    _enable(monkeypatch)
    idx = _new_index_with_fake_embedder(tmp_path, monkeypatch)
    _warm(idx)

    async def on_loop():
        with caplog.at_level(logging.WARNING):
            return idx.score("direction-A"), idx.boost_for("direction-A"), idx.score("again")

    score, boost, again = asyncio.run(on_loop())
    assert (score, boost, again) == (0.0, 0, 0.0)
    assert idx._embedder.embed_calls == 0
    assert idx.stats().on_loop_refusals == 3
    warned = [r for r in caplog.records if "event-loop thread" in r.getMessage()]
    assert len(warned) == 2                      # once per caller: score, boost_for
    assert any("score()" in r.getMessage() for r in warned)


def test_the_same_call_on_a_worker_thread_embeds(monkeypatch, tmp_path):
    _enable(monkeypatch)
    idx = _new_index_with_fake_embedder(tmp_path, monkeypatch)
    _warm(idx)

    async def off_loop():
        return await asyncio.to_thread(idx.score, "direction-A")

    assert asyncio.run(off_loop()) > 0.99
    assert idx._embedder.embed_calls == 1
    assert idx.stats().on_loop_refusals == 0


def test_score_with_cluster_offloaded_runs_off_the_loop(monkeypatch, tmp_path):
    _enable(monkeypatch)
    idx = _new_index_with_fake_embedder(tmp_path, monkeypatch)
    _warm(idx)
    seen_threads = []
    real_embed = idx._embedder.embed

    def _embed(texts):
        seen_threads.append(threading.get_ident())
        return real_embed(texts)

    idx._embedder.embed = _embed
    loop_thread = threading.get_ident()
    detail = asyncio.run(idx.score_with_cluster_offloaded("direction-A"))
    assert detail is not None and detail["score"] > 0.99
    assert seen_threads and all(t != loop_thread for t in seen_threads)
    assert idx.stats().on_loop_refusals == 0


def test_boost_and_score_offloaded_never_embeds_inline(monkeypatch, tmp_path):
    _enable(monkeypatch)
    idx = _new_index_with_fake_embedder(tmp_path, monkeypatch)
    _warm(idx)
    boost, score = asyncio.run(idx.boost_and_score_offloaded("direction-A"))
    assert boost >= 0 and score > 0.99
    assert idx.stats().on_loop_refusals == 0


def test_every_embed_in_the_index_passes_the_guard():
    import inspect
    src = inspect.getsource(si.SemanticIndex)
    assert "self._embedder.embed(" in src.split("def _embed_guarded")[1].split("async def _run_off_loop")[0]
    body_without_seam = src.replace(
        src.split("def _embed_guarded")[1].split("async def _run_off_loop")[0], "")
    assert "self._embedder.embed(" not in body_without_seam


def test_the_bridge_scores_off_the_loop():
    import inspect
    from backend.core.ouroboros.governance import conception_proposal_bridge as b
    src = inspect.getsource(b)
    assert "await self._score_blueprint_offloaded(bp)" in src
    assert "ev = self._score_blueprint(bp)" not in src
