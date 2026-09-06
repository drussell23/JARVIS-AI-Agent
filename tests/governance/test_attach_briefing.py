"""The organism greets an attaching cockpit in its own voice, and the banner
always names the model it is running on.

Two operator reports, one attach experience:

1. The banner showed no model. `set_active_model_tag` is set only by the boot
   gate, and only when the registry answered — a blip or an idle daemon left
   it blank. `resolve_display_model` falls back, daemon-side and
   config-driven, to the local lane's configured model, so the banner always
   names what will answer.

2. Attaching showed a blank transcript and a canned "I'm listening" line —
   no sign the organism was doing anything. `attach_briefing` generates, on
   the LOCAL model, a grounded one-or-two-sentence narration of what the
   organism is working on RIGHT NOW, emitted as the same INTENT voice the
   per-op 💭 uses, so the existing transport carries it to the cockpit.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.core.ouroboros.governance import attach_briefing as ab
from backend.core.ouroboros.governance import candidate_generator as cg
from backend.core.ouroboros.battle_test import narrative_channel as nc


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in (
        ab.MASTER_FLAG, ab.MIN_INTERVAL_FLAG, ab.TIMEOUT_FLAG, ab.MAX_TOKENS_FLAG,
        "JARVIS_LOCAL_PRIME_ENABLED", "JARVIS_LOCAL_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ab, "_last_emit_monotonic", 0.0, raising=False)
    monkeypatch.setattr(cg, "_ACTIVE_MODEL_TAG", "", raising=False)
    nc.reset_default_channel_for_tests()
    yield
    nc.reset_default_channel_for_tests()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# resolve_display_model — the banner always names the lane's model
# ---------------------------------------------------------------------------


def test_the_resolved_tag_wins_when_the_boot_gate_set_one(monkeypatch):
    monkeypatch.setattr(cg, "_ACTIVE_MODEL_TAG", "qwen3-coder-ov:30b")
    assert cg.resolve_display_model() == "qwen3-coder-ov:30b"


def test_an_empty_tag_falls_back_to_the_local_lanes_configured_model(monkeypatch):
    """The operator's exact case: idle daemon, no tag — still names the model."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_LOCAL_MODEL_NAME", "qwen3-coder-ov:30b")
    assert cg.resolve_display_model() == "qwen3-coder-ov:30b"


def test_no_tag_no_local_lane_and_no_pin_names_nothing(monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "false")
    monkeypatch.delenv("JARVIS_LOCAL_MODEL_NAME", raising=False)
    assert cg.resolve_display_model() == ""


def test_the_banner_helper_uses_the_resolver(monkeypatch):
    from backend.core.ouroboros.battle_test import cockpit_attach as ca
    monkeypatch.setattr(cg, "resolve_display_model", lambda: "the-local-model")
    assert ca.CockpitAttachBridge._active_model() == "the-local-model"


def test_the_banner_helper_never_raises(monkeypatch):
    from backend.core.ouroboros.battle_test import cockpit_attach as ca

    def _boom():
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(cg, "resolve_display_model", _boom)
    assert ca.CockpitAttachBridge._active_model() == ""


# ---------------------------------------------------------------------------
# The briefing is local-generated, grounded, and emitted as the INTENT voice
# ---------------------------------------------------------------------------


def _stub_gate(monkeypatch, capture=None, *, returns="I'm fixing the flaky "
              "assertion in tests/x.py so CI stops lying."):
    import backend.core.ouroboros.governance.rt_gate as rtg

    async def _fake(prompt, **kw):
        if capture is not None:
            capture["prompt"] = prompt
            capture["kw"] = kw
        return returns

    monkeypatch.setattr(rtg, "gate_completion", _fake)


def _stub_grounding(monkeypatch, text="100 signals queued; TestFailure on tests/x.py"):
    import backend.core.ouroboros.governance.side_channel as sc
    monkeypatch.setattr(sc, "compose_situation",
                        AsyncMock(return_value=SimpleNamespace(text=text)))
    monkeypatch.setattr(sc, "build_grounding",
                        lambda situation: f"CHARTER\n\n{situation.text}")


def test_a_briefing_is_generated_locally_and_emitted_as_intent(monkeypatch):
    cap: dict = {}
    _stub_grounding(monkeypatch)
    _stub_gate(monkeypatch, cap)
    ok = _run(ab.emit_for_attach(session="s1"))
    assert ok is True
    # It routed to the LOCAL lane, grounded on the live situation.
    assert cap["kw"].get("prefer") == "local"
    assert cap["kw"].get("caller_id") == "attach_briefing"
    assert "tests/x.py" in cap["prompt"] and "100 signals queued" in cap["prompt"]
    # It landed as the organism's INTENT voice, provider=local.
    frames = nc.get_default_channel().find_by_kind(nc.NarrativeKind.INTENT)
    assert len(frames) == 1
    assert "flaky assertion" in frames[0].prose and frames[0].provider == "local"


def test_the_master_flag_off_emits_nothing(monkeypatch):
    monkeypatch.setenv(ab.MASTER_FLAG, "false")
    _stub_grounding(monkeypatch)
    _stub_gate(monkeypatch)
    assert _run(ab.emit_for_attach()) is False
    assert nc.get_default_channel().find_by_kind(nc.NarrativeKind.INTENT) == ()


def test_rapid_reattaches_coalesce_into_one_briefing(monkeypatch):
    monkeypatch.setenv(ab.MIN_INTERVAL_FLAG, "3600")
    _stub_grounding(monkeypatch)
    _stub_gate(monkeypatch)
    assert _run(ab.emit_for_attach()) is True
    assert _run(ab.emit_for_attach()) is False        # coalesced
    assert len(nc.get_default_channel().find_by_kind(nc.NarrativeKind.INTENT)) == 1


def test_an_empty_generation_emits_nothing_and_frees_the_slot(monkeypatch):
    monkeypatch.setenv(ab.MIN_INTERVAL_FLAG, "3600")
    _stub_grounding(monkeypatch)
    _stub_gate(monkeypatch, returns="   ")            # model declined / cold
    assert _run(ab.emit_for_attach()) is False
    assert nc.get_default_channel().find_by_kind(nc.NarrativeKind.INTENT) == ()
    # The failed attempt did NOT burn the interval — the next attach may try.
    _stub_gate(monkeypatch, returns="Now I have something to say.")
    assert _run(ab.emit_for_attach()) is True


def test_a_gate_that_raises_never_breaks_the_attach(monkeypatch):
    import backend.core.ouroboros.governance.rt_gate as rtg
    _stub_grounding(monkeypatch)
    monkeypatch.setattr(rtg, "gate_completion",
                        AsyncMock(side_effect=RuntimeError("engine down")))
    assert _run(ab.emit_for_attach()) is False        # swallowed, no frame
    assert nc.get_default_channel().find_by_kind(nc.NarrativeKind.INTENT) == ()


def test_a_timeout_returns_no_briefing(monkeypatch):
    monkeypatch.setenv(ab.TIMEOUT_FLAG, "2")
    _stub_grounding(monkeypatch)
    import backend.core.ouroboros.governance.rt_gate as rtg

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(rtg, "gate_completion", _hang)
    assert _run(asyncio.wait_for(ab.emit_for_attach(), timeout=10)) is False


def test_grounding_failure_still_produces_a_leaner_briefing(monkeypatch):
    import backend.core.ouroboros.governance.side_channel as sc
    monkeypatch.setattr(sc, "compose_situation",
                        AsyncMock(side_effect=RuntimeError("digest down")))
    cap: dict = {}
    _stub_gate(monkeypatch, cap, returns="Watching for the next failing test.")
    assert _run(ab.emit_for_attach()) is True
    assert "no live activity is reported" in cap["prompt"]


# ---------------------------------------------------------------------------
# scheduling + the bridge hook
# ---------------------------------------------------------------------------


def test_schedule_returns_none_without_a_running_loop():
    assert ab.schedule_for_attach() is None            # no loop → nothing scheduled


@pytest.mark.asyncio
async def test_schedule_creates_a_task_on_the_running_loop(monkeypatch):
    _stub_grounding(monkeypatch)
    _stub_gate(monkeypatch)
    task = ab.schedule_for_attach(session="s")
    assert task is not None
    assert await asyncio.wait_for(task, timeout=5) is True


def test_the_bridge_schedules_a_briefing_when_a_cockpit_attaches():
    from backend.core.ouroboros.battle_test import cockpit_attach as ca
    src = inspect.getsource(ca.CockpitAttachBridge._on_client)
    assert "schedule_for_attach(" in src
    # After the client joins the live set, so the greeting reaches it.
    assert src.index("self._clients.add(writer)") < src.index("schedule_for_attach(")


def test_the_briefing_task_set_exists_on_the_bridge():
    from backend.core.ouroboros.battle_test import cockpit_attach as ca
    b = ca.CockpitAttachBridge.__new__(ca.CockpitAttachBridge)
    # The attribute is declared in __init__; a fresh instance sets it there.
    assert "_briefing_tasks" in inspect.getsource(ca.CockpitAttachBridge.__init__)
