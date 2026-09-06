"""A typed goal is a human asking, not a task-list entry.

`dispatch_backlog` wrote the goal to backlog.json, so it carried
`source="backlog"` — which sits in `_BACKGROUND_SOURCES` ("DW only, no Claude
fallback... cost-optimization-first") and is collected on a later sensor
sweep. Correct for the Backlog SENSOR mining a list. Wrong for someone who
just pressed Enter and is watching the screen.

The token described where the goal was STORED, not who ASKED for it — the same
class of defect `intent_envelope.py` already names: "Honest-source token:
decoupled test-coverage work MUST NOT masquerade as `test_failure`/`backlog`."
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pytest

from backend.core.ouroboros.governance.intake.intent_envelope import (
    SOVEREIGN_SOURCES,
    _VALID_SOURCES,
)
from backend.core.ouroboros.governance.intent.signals import SignalSource
from backend.core.ouroboros.governance.urgency_router import (
    _BACKGROUND_SOURCES,
    _IMMEDIATE_SOURCES,
)


# --------------------------------------------------------------------------
# 1. the honest token
# --------------------------------------------------------------------------

def test_a_typed_goal_has_its_own_source() -> None:
    assert SignalSource.OPERATOR_CHAT.value == "operator_chat"
    assert "operator_chat" in _VALID_SOURCES


def test_it_routes_IMMEDIATE_like_every_other_human_origin() -> None:
    """§5: human-originated signals route IMMEDIATE because a person is
    waiting on the answer. A typed goal is that, exactly as a spoken one is."""
    assert "operator_chat" in _IMMEDIATE_SOURCES
    assert "voice_human" in _IMMEDIATE_SOURCES


def test_it_is_no_longer_background() -> None:
    """THE defect: `backlog` is cost-optimized and polled. A watched goal
    must not inherit that."""
    assert "operator_chat" not in _BACKGROUND_SOURCES
    assert "backlog" in _BACKGROUND_SOURCES, "the sensor route must remain"


def test_it_holds_sovereign_primacy() -> None:
    """The host keeps ultimate control — a typed goal outranks a resurrected
    op for the same reason a spoken one does."""
    assert "operator_chat" in SOVEREIGN_SOURCES


def test_it_is_distinct_from_voice_and_from_backlog() -> None:
    """Attributable separately, so observability can tell a spoken goal from
    a typed one from a mined one."""
    assert len({"operator_chat", "voice_human", "backlog"}) == 3


def test_the_immediate_set_stays_tight() -> None:
    """The bt-2026-04-13 regression was seven autonomous sensors mislabelling
    themselves and firing unattended. This source can only be produced by a
    keystroke, but the set must still not sprawl."""
    assert len(_IMMEDIATE_SOURCES) <= 5


# --------------------------------------------------------------------------
# 2. the executor dispatches now, and never loses the goal
# --------------------------------------------------------------------------

class _Turn:
    turn_id = "chat-1dc4650228e7"
    session_id = "repl"


def _executor(tmp_path: Path, submit: Any = None) -> Any:
    from backend.core.ouroboros.governance.chat_repl_backlog_executor import (
        BacklogChatActionExecutor,
    )
    return BacklogChatActionExecutor(tmp_path, submit_now=submit)


def test_a_wired_submitter_dispatches_immediately(tmp_path: Path) -> None:
    seen: List[Any] = []

    def _submit(msg: str, turn: Any) -> str:
        seen.append((msg, turn.turn_id))
        return "op-019fa4d2-246e-7759-86"

    out = _executor(tmp_path, _submit).dispatch_backlog("fix the tests", _Turn())
    assert out.startswith("op-")
    assert seen and seen[0][0] == "fix the tests"
    assert not (tmp_path / "backlog.json").exists(), (
        "it filed the goal as well as running it — duplicate work"
    )


def test_without_a_submitter_the_old_path_is_untouched(tmp_path: Path) -> None:
    """Injected, not imported: this module must stay usable with no intake,
    no event loop and no daemon."""
    out = _executor(tmp_path).dispatch_backlog("fix the tests", _Turn())
    assert out == "chat:chat-1dc4650228e7"


def test_an_intake_fault_falls_back_rather_than_dropping(tmp_path: Path) -> None:
    """A queued goal is a degradation. A dropped one is a betrayal."""
    def _boom(_msg: str, _turn: Any) -> str:
        raise RuntimeError("intake unreachable")

    out = _executor(tmp_path, _boom).dispatch_backlog("fix the tests", _Turn())
    assert out == "chat:chat-1dc4650228e7", "the goal was lost"


def test_a_submitter_returning_nothing_falls_back(tmp_path: Path) -> None:
    """Intake declining is not an exception — it must still not lose the
    goal."""
    out = _executor(tmp_path, lambda _m, _t: None).dispatch_backlog("x", _Turn())
    assert out == "chat:chat-1dc4650228e7"


def test_an_empty_goal_is_never_dispatched(tmp_path: Path) -> None:
    """Refusing empty input predates this and must survive it — otherwise a
    stray Enter fires an IMMEDIATE Claude op."""
    calls: List[Any] = []
    out = _executor(tmp_path, lambda m, t: calls.append(m) or "op-x").dispatch_backlog(
        "   ", _Turn(),
    )
    assert calls == [], "an empty goal reached intake"
    assert out.startswith("error-empty-message")


# --------------------------------------------------------------------------
# 3. the seam is armed by the daemon, not assumed everywhere
# --------------------------------------------------------------------------

def test_the_registry_arms_and_disarms(tmp_path: Path) -> None:
    """ONE seam rather than a parameter threaded through three nested
    factories — an intermediate factory that forgets to forward produces an
    executor that silently files goals, which is the wired-but-inert failure
    this codebase keeps paying for."""
    from backend.core.ouroboros.governance.chat_repl_backlog_executor import (
        set_operator_dispatcher,
    )
    try:
        assert _executor(tmp_path).dispatch_backlog("x", _Turn()).startswith("chat:")
        set_operator_dispatcher(lambda _m, _t: "op-019fa4d2-246e-7759-86")
        assert _executor(tmp_path).dispatch_backlog("x", _Turn()).startswith("op-")
    finally:
        set_operator_dispatcher(None)
    assert _executor(tmp_path).dispatch_backlog("x", _Turn()).startswith("chat:")


def test_explicit_injection_outranks_the_registry(tmp_path: Path) -> None:
    """Tests and knowing callers must be able to override the daemon's
    standing answer."""
    from backend.core.ouroboros.governance.chat_repl_backlog_executor import (
        set_operator_dispatcher,
    )
    try:
        set_operator_dispatcher(lambda _m, _t: "op-from-registry")
        out = _executor(tmp_path, lambda _m, _t: "op-explicit").dispatch_backlog(
            "x", _Turn(),
        )
        assert out == "op-explicit"
    finally:
        set_operator_dispatcher(None)


def test_the_daemon_arms_it_at_boot() -> None:
    """Structural: an unarmed seam files every goal — correct, but not what
    was built."""
    src = (Path(__file__).resolve().parents[2]
           / "backend/core/ouroboros/battle_test/harness.py").read_text()
    assert "set_operator_dispatcher(self._submit_operator_goal)" in src


def test_the_submitter_uses_the_router_every_sensor_uses() -> None:
    """DRY, and governance: no private lane for operator input, so one set of
    dedup / WAL / priority rules covers autonomous and human work alike."""
    import inspect

    from backend.core.ouroboros.battle_test.harness import BattleTestHarness

    # The router resolution moved into the shared _dispatch_intake_envelope
    # seam (a typed chat goal AND a sanctioned /goal both reach intake through
    # it), so the submitter delegates rather than inlining the lane — the DRY
    # property this test guards, now stronger.
    submit_src = inspect.getsource(BattleTestHarness._submit_operator_goal)
    assert "self._dispatch_intake_envelope(" in submit_src
    assert 'source="operator_chat"' in submit_src
    dispatch_src = inspect.getsource(
        BattleTestHarness._dispatch_intake_envelope,
    )
    assert '"_intake_router", "intake_router", "_router"' in dispatch_src


def test_urgency_is_high_not_critical() -> None:
    """The operator is waiting, but a typed goal is not an emergency and must
    not outrank a runtime alarm. IMMEDIATE eligibility comes from the SOURCE
    being human-origin — inflating urgency to buy it would distort every
    priority queue the envelope passes through."""
    import inspect

    from backend.core.ouroboros.battle_test.harness import BattleTestHarness

    src = inspect.getsource(BattleTestHarness._submit_operator_goal)
    assert 'urgency="high"' in src
    assert 'urgency="critical"' not in src


def test_an_unreachable_intake_returns_none_rather_than_raising() -> None:
    """So the caller files the goal and says so."""
    from backend.core.ouroboros.battle_test.harness import BattleTestHarness

    bare = BattleTestHarness.__new__(BattleTestHarness)
    assert bare._submit_operator_goal("fix the tests", _Turn()) is None


# --------------------------------------------------------------------------
# 4. the submitter is CALLED, against the real envelope contract
# --------------------------------------------------------------------------
#
# Everything above this line either injects a fake submitter that returns a
# literal "op-…" or greps the SOURCE TEXT of `_submit_operator_goal`. Both
# pass against a function that cannot run — and for twelve days both did.
#
# `_submit_operator_goal` built its envelope with the raw `IntentEnvelope`
# dataclass constructor, passing 4 of its 15 required fields. Every call
# raised TypeError, was caught one frame down, logged at DEBUG, and returned
# None. The executor read None as "intake declined" and filed the goal. The
# cockpit then honestly reported it queued — the renderer was right; the path
# behind it was dead.
#
# So these tests CALL it, and let the REAL `make_envelope` validate what it
# built. A fake that accepts an envelope the real constructor would reject is
# the mirror of the bug it is supposed to catch.


class _RecordingRouter:
    """Shaped like UnifiedIntakeRouter: `ingest` is async and returns a
    VERDICT string, never an id. Getting that wrong is the second half of
    the original defect."""

    def __init__(self, verdict: str = "enqueued") -> None:
        self.verdict = verdict
        self.envelopes: List[Any] = []

    async def ingest(self, envelope: Any) -> str:
        self.envelopes.append(envelope)
        return self.verdict


class _GLS:
    def __init__(self, router: Any) -> None:
        self._intake_router = router
        self.operator_ops: List[str] = []

    def note_operator_op(self, op_id: str) -> None:
        self.operator_ops.append(op_id)


def _harness_with(router: Any, loop: Any) -> Any:
    from backend.core.ouroboros.battle_test.harness import BattleTestHarness

    h = BattleTestHarness.__new__(BattleTestHarness)
    h._governed_loop_service = _GLS(router)
    h._operator_goal_loop = loop
    h.repo = "jarvis"
    return h


@pytest.fixture()
def bg_loop():
    """A loop running on ANOTHER thread — because that is the real topology.

    `chat_text_bridge._run` dispatches through `asyncio.to_thread`, so the
    executor calls the submitter from a worker thread with no running loop of
    its own. A same-thread `asyncio.run(...)` test would hide that.
    """
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


def test_it_returns_a_real_op_id_when_intake_accepts(bg_loop) -> None:
    """THE regression. This is what every string assertion above believed."""
    router = _RecordingRouter("enqueued")
    out = _harness_with(router, bg_loop)._submit_operator_goal(
        "make the attach handshake honest", _Turn(),
    )
    assert out is not None, "the submitter returned None — the goal was FILED"
    assert out.startswith("op-"), (
        f"receipt {out!r} does not read as dispatched; chat_response_style."
        "_dispatched_now only recognises an `op-` prefix, so anything else "
        "renders as queued"
    )
    assert router.envelopes, "nothing ever reached intake"


def test_the_envelope_it_builds_passes_real_validation(bg_loop) -> None:
    """Not a shape-alike. The frozen dataclass validates in __post_init__, and
    it is that validation the original constructor call never survived."""
    from backend.core.ouroboros.governance.intake.intent_envelope import (
        SCHEMA_VERSION,
        IntentEnvelope,
    )

    router = _RecordingRouter("enqueued")
    _harness_with(router, bg_loop)._submit_operator_goal("fix the tests", _Turn())

    env = router.envelopes[0]
    assert isinstance(env, IntentEnvelope)
    assert env.source == "operator_chat"
    assert env.description == "fix the tests"
    assert env.urgency == "high"
    assert env.schema_version == SCHEMA_VERSION
    # causal == signal: the keystroke IS the origin event, as the utterance is
    # for VoiceCommandSensor.
    assert env.causal_id == env.signal_id
    assert not env.requires_human_ack, "the human IS the origin"
    assert env.evidence.get("origin") == "cockpit_chat"
    assert env.evidence.get("turn_id") == _Turn.turn_id


def test_a_typed_goal_needs_no_target_files() -> None:
    """A sentence names an intent, not a path. Localization is the Iron
    Gate's job — demanding target_files here would make the operator do it."""
    from backend.core.ouroboros.governance.intake.intent_envelope import (
        _EMPTY_TARGET_FILES_EXEMPT_SOURCES,
        make_envelope,
    )

    assert "operator_chat" in _EMPTY_TARGET_FILES_EXEMPT_SOURCES
    env = make_envelope(
        source="operator_chat", description="do the thing", target_files=(),
        repo="jarvis", confidence=1.0, urgency="high", evidence={},
        requires_human_ack=False,
    )
    assert env.target_files == ()


def test_the_receipt_is_recognised_as_dispatched_by_the_renderer(bg_loop) -> None:
    """End of the chain: the id this mints must make the cockpit say `on it`.
    Minting a correct id the renderer reads as queued would fix nothing the
    operator can see."""
    from backend.core.ouroboros.governance.chat_response_style import compose_reply

    out = _harness_with(_RecordingRouter("enqueued"), bg_loop)._submit_operator_goal(
        "fix the tests", _Turn(),
    )
    reply = compose_reply("backlog_dispatch", receipt=out or "")
    assert "on it" in reply
    assert "queued" not in reply
    assert "backlog" not in reply.lower()


@pytest.mark.parametrize("verdict", ["deduplicated", "backpressure", ""])
def test_a_refused_goal_is_filed_not_claimed(bg_loop, verdict: str) -> None:
    """`ingest` returns a verdict, never an id. `backpressure` means intake
    REFUSED it — reporting `op-` for that claims an acceptance we were denied,
    which is the exact dishonesty the receipt vocabulary exists to prevent."""
    out = _harness_with(_RecordingRouter(verdict), bg_loop)._submit_operator_goal(
        "fix the tests", _Turn(),
    )
    assert out is None, (
        f"verdict={verdict!r} was reported as dispatched; the goal must fall "
        "back to the backlog so the reply's 'queued' is true"
    )


@pytest.mark.parametrize("verdict", ["enqueued", "pending_ack"])
def test_both_accepting_verdicts_mint_a_receipt(bg_loop, verdict: str) -> None:
    """`pending_ack` is parked, but intake HOLDS it — it is not lost, and the
    backlog is not where it lives."""
    out = _harness_with(_RecordingRouter(verdict), bg_loop)._submit_operator_goal(
        "fix the tests", _Turn(),
    )
    assert out is not None and out.startswith("op-")


def test_esc_can_target_the_op_before_intake_answers(bg_loop) -> None:
    """Esc must be able to cancel what the operator just watched dispatch."""
    h = _harness_with(_RecordingRouter("enqueued"), bg_loop)
    out = h._submit_operator_goal("fix the tests", _Turn())
    assert h._governed_loop_service.operator_ops == [out]


def test_it_falls_back_to_the_process_wide_router(bg_loop) -> None:
    """The singleton the router registers on its own construction — the same
    handle the voice side uses. A deployment that never attached it to the GLS
    would otherwise file every goal with a live router one import away."""
    from backend.core.ouroboros.governance.intake import unified_intake_router as uir

    router = _RecordingRouter("enqueued")
    h = _harness_with(router, bg_loop)
    h._governed_loop_service = None          # nothing attached
    prev = uir.get_default_intake_router()
    try:
        uir.set_default_intake_router(router)  # type: ignore[arg-type]
        out = h._submit_operator_goal("fix the tests", _Turn())
    finally:
        uir.set_default_intake_router(prev)  # type: ignore[arg-type]
    assert out is not None and out.startswith("op-")


def test_an_intake_that_raises_files_rather_than_drops(bg_loop) -> None:
    """A queued goal is a degradation. A dropped one is a betrayal."""
    class _Boom:
        async def ingest(self, _envelope: Any) -> str:
            raise RuntimeError("intake unreachable")

    out = _harness_with(_Boom(), bg_loop)._submit_operator_goal("x", _Turn())
    assert out is None


def test_a_typed_goal_never_skips_PLAN() -> None:
    """The consequence of the exemption above, pinned deliberately.

    `PlanGenerator._should_skip` returns "" (do not skip) for every source in
    the exempt set, because an op with no target_files must reason about its
    OWN localization rather than be skipped for looking trivial. A typed goal
    is exactly that op — `n_files == 0` would otherwise read as "nothing to
    plan" and skip straight to GENERATE.
    """
    import inspect

    from backend.core.ouroboros.governance.plan_generator import PlanGenerator

    src = inspect.getsource(PlanGenerator._should_skip)
    assert "_EMPTY_TARGET_FILES_EXEMPT_SOURCES" in src


def test_a_typed_goal_is_not_floored_to_COMPLEX() -> None:
    """The exemption must NOT leak into the complexity floor. `operator_chat`
    covers everything from "fix a typo" to "rebuild the IPC layer"; forcing
    every typed sentence onto the COMPLEX route would buy Claude planning
    budget for "what time is it"."""
    from backend.core.ouroboros.governance.complexity_classifier import (
        _COMPLEX_FLOOR_SOURCES,
    )

    assert "operator_chat" not in _COMPLEX_FLOOR_SOURCES
    assert "swe_bench_pro" in _COMPLEX_FLOOR_SOURCES
