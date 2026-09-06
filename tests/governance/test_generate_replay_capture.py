"""Task #11 — close GENERATE replay-blindness.

GENERATE was the one phase blind to replay: the determinism substrate
captured only a provider-selection DIGEST (a hash), so REPLAY re-invoked the
model live — the most expensive, least reproducible phase couldn't be
replayed. This wraps the FULL generation acquisition in
capture_phase_decision with a GenerationResult adapter, so REPLAY returns the
recorded candidates and SKIPS the provider call entirely.

Proof obligations:
  * The GenerationResult adapter round-trips faithfully.
  * RECORD then REPLAY of the same op returns the recorded result WITHOUT
    invoking the (model) compute — the actual replay-blindness fix.
  * PASSTHROUGH (default) runs compute live — bit-for-bit legacy.
  * The wire is present + digest-only capture is gone (guard re-severing).
  * A park (BaseException) propagates through the capture, never swallowed.
"""
from __future__ import annotations

import inspect

import pytest

from backend.core.ouroboros.governance.op_context import GenerationResult
from backend.core.ouroboros.governance.tool_executor import (
    ToolExecStatus,
    ToolExecutionRecord,
)
# Importing the runner registers the GENERATE/generate adapter at module load.
import backend.core.ouroboros.governance.phase_runners.generate_runner as gr  # noqa: F401,E501
from backend.core.ouroboros.governance.determinism.phase_capture import (
    capture_phase_decision,
    get_adapter,
    iter_registered,
)
from backend.core.ouroboros.governance.determinism.decision_runtime import (
    reset_all_for_tests,
)
from backend.core.ouroboros.governance.park_signal import ParkRequested


def _gen(n: int = 2) -> GenerationResult:
    return GenerationResult(
        candidates=tuple(
            {"file_path": f"f{i}.py", "full_content": f"x = {i}\n"}
            for i in range(n)
        ),
        provider_name="dw",
        generation_duration_s=1.25,
        model_id="model-x",
        is_noop=False,
        venom_edit_history=({"tool": "edit_file", "path": "f0.py"},),
        prompt_preloaded_files=("f0.py",),
        tool_execution_records=(
            ToolExecutionRecord(
                schema_version="tool.exec.v1", op_id="op-gen-1",
                call_id="op-gen-1:r0:read_file", round_index=0,
                tool_name="read_file", tool_version="1",
                arguments_hash="abc123", repo="jarvis",
                policy_decision="ALLOW", policy_reason_code="ok",
                started_at_ns=111, ended_at_ns=222, duration_ms=1.5,
                output_bytes=42, error_class=None,
                status=ToolExecStatus.SUCCESS,
            ),
        ),
        total_input_tokens=100,
        total_output_tokens=42,
        cost_usd=0.01,
    )


class _Ctx:
    op_id = "op-gen-1"
    provider_route = "STANDARD"
    signal_urgency = ""
    signal_source = ""
    task_complexity = ""
    target_files = ()
    cross_repo = False
    is_read_only = False


@pytest.fixture
def det_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_DIR", str(tmp_path / "det"))
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_ENABLED", "true")
    monkeypatch.setenv("JARVIS_DETERMINISM_PHASE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("OUROBOROS_BATTLE_SESSION_ID", "gen-replay-test")
    monkeypatch.delenv("JARVIS_DETERMINISM_LEDGER_MODE", raising=False)
    reset_all_for_tests()
    yield
    reset_all_for_tests()


# ── adapter round-trip ───────────────────────────────────────────────

def test_generate_adapter_is_registered():
    assert ("GENERATE", "generate") in iter_registered()
    assert get_adapter(phase="GENERATE", kind="generate").name == (
        "generation_result_adapter"
    )


def test_adapter_roundtrips_generation_result():
    a = get_adapter(phase="GENERATE", kind="generate")
    g = _gen(3)
    back = a.deserialize(a.serialize(g))
    assert isinstance(back, GenerationResult)
    assert back.candidates == g.candidates
    assert back.provider_name == g.provider_name
    assert back.model_id == g.model_id
    assert back.total_output_tokens == g.total_output_tokens
    assert back.cost_usd == g.cost_usd
    assert back.venom_edit_history == g.venom_edit_history
    # tool_execution_records ARE preserved across the round-trip now: the
    # adapter runs in RECORD mode on the LIVE object (not only REPLAY), so
    # dropping them silently zeroed the recap tool-count on every op.
    assert len(back.tool_execution_records) == 1
    assert back.tool_execution_records[0].tool_name == "read_file"


def test_adapter_handles_none():
    a = get_adapter(phase="GENERATE", kind="generate")
    assert a.deserialize(a.serialize(None)) is None


# ── THE fix: RECORD then REPLAY skips the model ──────────────────────

@pytest.mark.asyncio
async def test_record_then_replay_skips_the_model(det_env, monkeypatch):
    calls = {"n": 0}

    async def compute_live():
        calls["n"] += 1
        return _gen(2)

    # RECORD — compute runs once, result serialized to the ledger.
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_MODE", "record")
    reset_all_for_tests()
    out1 = await capture_phase_decision(
        op_id="op-gen-1", phase="GENERATE", kind="generate",
        ctx=_Ctx(), compute=compute_live,
    )
    assert calls["n"] == 1
    assert isinstance(out1, GenerationResult)
    assert out1.candidates == _gen(2).candidates

    # REPLAY — the model MUST NOT be called. A fresh runtime reads the
    # recorded result from disk and returns it; compute would raise.
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_MODE", "replay")
    reset_all_for_tests()

    async def compute_boom():
        calls["n"] += 1
        raise AssertionError("the model was called during REPLAY!")

    out2 = await capture_phase_decision(
        op_id="op-gen-1", phase="GENERATE", kind="generate",
        ctx=_Ctx(), compute=compute_boom,
    )
    assert calls["n"] == 1  # compute_boom NEVER ran → blindness closed
    assert isinstance(out2, GenerationResult)
    assert out2.candidates == _gen(2).candidates
    assert out2.provider_name == "dw"
    assert out2.model_id == "model-x"


@pytest.mark.asyncio
async def test_passthrough_default_runs_compute_live(det_env, monkeypatch):
    """Default (no LEDGER_MODE = PASSTHROUGH): compute runs, nothing is
    recorded — bit-for-bit legacy."""
    monkeypatch.delenv("JARVIS_DETERMINISM_LEDGER_MODE", raising=False)
    reset_all_for_tests()
    calls = {"n": 0}

    async def compute_live():
        calls["n"] += 1
        return _gen(1)

    out = await capture_phase_decision(
        op_id="op-pt-1", phase="GENERATE", kind="generate",
        ctx=_Ctx(), compute=compute_live,
    )
    assert calls["n"] == 1
    assert isinstance(out, GenerationResult)


@pytest.mark.asyncio
async def test_park_propagates_through_capture(det_env, monkeypatch):
    """A PARK-EMIT (ParkRequested, a BaseException) raised by the generator
    must fly THROUGH the capture wrapper untouched — never swallowed, never
    recorded as a generation."""
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_MODE", "record")
    reset_all_for_tests()

    from types import SimpleNamespace
    _sig = SimpleNamespace(
        op_id="op-park-1", token="tok", attempt_seq=0,
        descriptor=SimpleNamespace(kind="generate"),
    )

    async def compute_park():
        raise ParkRequested(_sig)

    with pytest.raises(ParkRequested):
        await capture_phase_decision(
            op_id="op-park-1", phase="GENERATE", kind="generate",
            ctx=_Ctx(), compute=compute_park,
        )


# ── records survive the round-trip (the recap tool-count leak) ───────

def test_adapter_preserves_tool_execution_records():
    """The GENERATE adapter must carry tool_execution_records across the
    serialize->deserialize round-trip it performs in RECORD mode, not only
    REPLAY. The deserialize used to hardcode ``()`` so the terminal seam saw
    0 tools for every op even though the model explored. Volatile wall-clock
    timings are intentionally dropped so the stored form stays VERIFY-stable."""
    a = get_adapter(phase="GENERATE", kind="generate")
    g = _gen(1)
    back = a.deserialize(a.serialize(g))
    assert len(back.tool_execution_records) == 1
    r0 = back.tool_execution_records[0]
    o0 = g.tool_execution_records[0]
    assert r0.tool_name == o0.tool_name
    assert r0.arguments_hash == o0.arguments_hash
    assert r0.output_bytes == o0.output_bytes
    assert r0.round_index == o0.round_index
    assert getattr(r0.status, "value", r0.status) == "success"
    # Volatile timings dropped -> None (a fabricated timestamp would lie).
    assert r0.started_at_ns is None
    assert r0.ended_at_ns is None
    assert r0.duration_ms is None


@pytest.mark.asyncio
async def test_record_mode_preserves_records_through_capture(
    det_env, monkeypatch,
):
    """RECORD mode must return a result that still carries the live tool
    records and token counts. This is the actual leak: capture_phase_decision
    round-trips through the adapter in RECORD mode and the old deserialize
    dropped the records, so ctx.generation reached the seam with 0 tools."""
    monkeypatch.setenv("JARVIS_DETERMINISM_LEDGER_MODE", "record")
    reset_all_for_tests()

    async def compute_live():
        return _gen(1)

    out = await capture_phase_decision(
        op_id="op-gen-rec", phase="GENERATE", kind="generate",
        ctx=_Ctx(), compute=compute_live,
    )
    assert isinstance(out, GenerationResult)
    assert len(out.tool_execution_records) == 1
    assert out.tool_execution_records[0].tool_name == "read_file"
    assert out.total_output_tokens == 42


# ── reachability: the wire is present, digest-only capture is gone ───

def test_generate_runner_wraps_acquisition_not_digest():
    src = inspect.getsource(gr.GENERATERunner.run)
    # The full acquisition is captured under kind="generate"...
    assert 'kind="generate"' in src
    assert "_acquire_generation" in src
    assert "compute=_acquire_generation" in src
    # ...and the old digest-only provider_selection capture is gone.
    assert "provider_selection" not in src
    assert "_digest_compute" not in src
