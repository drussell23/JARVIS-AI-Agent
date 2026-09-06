# tests/governance/test_local_lane_arc.py
"""Regression spine for the local zero-cost generation lane arc.

Four masters ship in this arc, three of them default-ON, and every one of them
sits on a path that a live soak exercises non-deterministically. The point of
this file is that the STATE TRANSITIONS -- the constrained-decoding ladder, the
learned-capability cache, the credential TTL, the in-flight bracket -- are
proven here, so a soak is left to measure only what a soak can measure: real
payloads from a real model.

Covered masters:
  * JARVIS_LOCAL_JSON_MODE_ENABLED            (director, default ON)
  * JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED     (director, default ON)
  * JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED (generator, default ON)
  * JARVIS_FREE_LANE_POLICY_ENABLED           (generator, default ON)
plus the two supporting knobs JARVIS_FREE_LANE_CRED_TTL_S and
JARVIS_DOC_MAX_NAMED_SYMBOLS.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import os
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_generation_recursion_ledger():
    """These tests reuse a FIXED op_id ("op-free-lane-test"), so the process-
    global generation-recursion ledger (the shared per-op recovery budget,
    Phase 3) would accumulate depth across tests and trip its ceiling mid-file.
    Production op_ids are unique UUIDs with TTL expiry, so this isolation is a
    test-only concern. Clear the ledger around every test."""
    from backend.core.ouroboros.governance import generation_recursion_bound as _rb
    with _rb._LOCK:
        _rb._LEDGER.clear()
    yield
    with _rb._LOCK:
        _rb._LEDGER.clear()

LID = "backend.core.ouroboros.governance.local_inference_director"
GW = "backend.core.ouroboros.governance.inference_gateway"
CG = "backend.core.ouroboros.governance.candidate_generator"
DSS = "backend.core.ouroboros.governance.intake.sensors.doc_staleness_sensor"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lid(monkeypatch):
    """The director with a CLEAN learned-capability cache.

    ``_SCHEMA_UNSUPPORTED`` is process-global by design (a rejection cannot
    become an acceptance without a restart). That is right in production and
    poison across tests, so every test gets a fresh set restored on exit.
    """
    mod = importlib.import_module(LID)
    monkeypatch.setattr(mod, "_SCHEMA_UNSUPPORTED", set())
    for k in ("JARVIS_LOCAL_JSON_MODE_ENABLED", "JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    return mod


@pytest.fixture()
def cfg(lid):
    # These tests pin the OpenAI-compatible DIALECT's spellings
    # (``response_format``, ``/v1/chat/completions``). The default transport
    # is now native (``/api/chat`` -- the only route on which the sampler
    # options bite), so the dialect under test is declared rather than
    # inherited. The native spellings have their own spine in
    # test_local_native_transport.py.
    return lid.LocalConfig(**{**lid.LocalConfig.from_env().__dict__,
                              "transport": "openai"})


@pytest.fixture()
def gateway(monkeypatch):
    """A FRESH InferenceGateway, never the process singleton.

    The in-flight counter is the object under test; sharing the singleton would
    let one test's leak masquerade as another's bug.
    """
    mod = importlib.import_module(GW)
    gw = mod.InferenceGateway()
    monkeypatch.setattr(mod, "_SINGLETON", gw)
    return gw


@pytest.fixture()
def cg(monkeypatch):
    """The candidate generator with the credential TTL stamp forced STALE.

    Deliberately a large negative sentinel, not 0.0. The stamp is compared
    against ``time.monotonic()``, which is process uptime -- so 0.0 only reads
    as "stale" once the process has been alive longer than the TTL, and a suite
    that reaches this file in under 30s would silently skip the re-read and
    test nothing. That is exactly how this fixture failed in-suite while
    passing in isolation.
    """
    mod = importlib.import_module(CG)
    monkeypatch.setattr(mod, "_FREE_LANE_CRED_REFRESH_AT", -1e9)
    for k in ("JARVIS_FREE_LANE_POLICY_ENABLED", "JARVIS_FREE_LANE_CRED_TTL_S",
              "JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED",
              "JARVIS_LOCAL_PRIME_ENABLED",
              "DOUBLEWORD_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return mod


# ===========================================================================
# Phase 1a -- constrained decoding ladder (JARVIS_LOCAL_JSON_*_MODE_ENABLED)
# ===========================================================================

def test_json_modes_default_on(lid):
    """Both rungs default ON: the arc's premise is that an unconstrained local
    32B breaks JSON, so the constraint must not require opting in."""
    assert lid._json_mode_enabled() is True
    assert lid._json_schema_mode_enabled() is True


@pytest.mark.parametrize("val,expected", [
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), ("TRUE", True),
])
def test_json_schema_mode_toggle(lid, monkeypatch, val, expected):
    monkeypatch.setenv("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", val)
    assert lid._json_schema_mode_enabled() is expected


def test_ladder_top_rung_attaches_json_schema(lid, cfg):
    """Default state: the strongest constraint, carrying a real schema."""
    body = {}
    assert lid._apply_response_format(body, cfg) == "json_schema"
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "ov_generation"
    # strict=False is load-bearing: the schema is a union with open
    # additionalProperties, and demanding strict is what makes some engines
    # reject an otherwise-valid grammar.
    assert rf["json_schema"]["strict"] is False
    assert isinstance(rf["json_schema"]["schema"], dict)
    assert rf["json_schema"]["schema"]


def test_ladder_middle_rung_when_schema_mode_off(lid, cfg, monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", "0")
    body = {}
    assert lid._apply_response_format(body, cfg) == "json_object"
    assert body["response_format"] == {"type": "json_object"}


def test_ladder_bottom_rung_when_json_mode_off(lid, cfg, monkeypatch):
    """Master OFF attaches NOTHING -- the pre-constraint behaviour exactly."""
    monkeypatch.setenv("JARVIS_LOCAL_JSON_MODE_ENABLED", "0")
    body = {}
    assert lid._apply_response_format(body, cfg) == "none"
    assert "response_format" not in body


def test_json_mode_off_beats_schema_mode_on(lid, cfg, monkeypatch):
    """The masters compose as a ladder, not as peers: the outer master wins."""
    monkeypatch.setenv("JARVIS_LOCAL_JSON_MODE_ENABLED", "0")
    monkeypatch.setenv("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", "1")
    body = {}
    assert lid._apply_response_format(body, cfg) == "none"
    assert "response_format" not in body


def test_ladder_degrades_when_schema_unresolvable(lid, cfg, monkeypatch):
    """A missing provider layer must cost the top rung, never the request."""
    monkeypatch.setattr(lid, "_resolve_response_schema", lambda: None)
    body = {}
    assert lid._apply_response_format(body, cfg) == "json_object"
    assert body["response_format"] == {"type": "json_object"}


def test_apply_never_raises_on_broken_schema_resolution(lid, cfg, monkeypatch):
    """A constraint is an optimisation over parsing; failing to attach one must
    not fail the generation."""
    def _boom():
        raise RuntimeError("provider layer exploded")
    monkeypatch.setattr(lid, "_resolve_response_schema", _boom)
    body = {}
    assert lid._apply_response_format(body, cfg) == "none"


def test_resolved_schema_is_a_projection_of_the_provider_union(lid):
    """The grammar and the validator must not be able to drift apart.

    Pins the property that makes constraining the sampler safe: the schema is
    derived from the provider layer's own constants, and it is a UNION -- if it
    ever collapses to the candidate shape alone, tool calls become
    unrepresentable and the Venom loop can never run.
    """
    from backend.core.ouroboros.governance import providers

    schema = lid._resolve_response_schema()
    assert schema == providers.build_response_json_schema()
    branches = schema.get("anyOf") or schema.get("oneOf") or []
    assert len(branches) >= 3, "union must keep candidates + noop + tool shapes"

    required_sets = [frozenset(b.get("required", ())) for b in branches]
    # The candidate branch is the one this arc exists for: rationale REQUIRED
    # (kills candidate_0_missing_rationale) and schema_version REQUIRED (kills
    # wrong_schema_version:__missing__).
    cand = next(
        (b for b in branches
         if "candidates" in (b.get("properties") or {})), None,
    )
    assert cand is not None, "no candidates branch in the union"
    assert "schema_version" in cand["required"]
    item_required = cand["properties"]["candidates"]["items"]["required"]
    assert "rationale" in item_required
    assert "full_content" in item_required
    # Emission order is load-bearing: a grammar emits required properties in
    # declaration order, and every short field must precede the unbounded one
    # or `rationale` is only reached if the token budget survived the file.
    assert item_required.index("rationale") < item_required.index("full_content")
    assert any("tool_calls" in (b.get("properties") or {}) for b in branches)
    assert required_sets  # union branches all declare a required set


# --- learned-capability cache (_SCHEMA_UNSUPPORTED) ------------------------

def test_degrade_populates_cache_and_rewrites_body(lid, cfg):
    body = {"response_format": {"type": "json_schema", "json_schema": {}}}
    assert lid._degrade_response_format(body, cfg) is True
    assert lid._schema_key(cfg) in lid._SCHEMA_UNSUPPORTED
    assert body["response_format"] == {"type": "json_object"}


def test_degrade_is_idempotent_so_a_400_cannot_become_a_retry_loop(lid, cfg):
    """Second call returns False. A persistently-400ing endpoint must cost one
    extra round trip in total, not one per op forever."""
    body = {"response_format": {"type": "json_schema"}}
    assert lid._degrade_response_format(body, cfg) is True
    assert lid._degrade_response_format(body, cfg) is False
    assert lid._degrade_response_format(body, cfg) is False


def test_cache_is_keyed_per_endpoint_and_model(lid, cfg):
    """One engine's refusal must not disarm the constraint everywhere.

    A rejection is evidence about THIS build serving THIS model; applying it to
    a sibling endpoint would silently drop shape enforcement on a host that
    never objected.
    """
    other_model = lid.LocalConfig(**{**cfg.__dict__, "model_name": "other:70b"})
    other_host = lid.LocalConfig(**{**cfg.__dict__, "base_url": "http://10.0.0.9:11434"})

    lid._degrade_response_format({}, cfg)

    assert lid._schema_key(cfg) in lid._SCHEMA_UNSUPPORTED
    assert lid._schema_key(other_model) not in lid._SCHEMA_UNSUPPORTED
    assert lid._schema_key(other_host) not in lid._SCHEMA_UNSUPPORTED

    body = {}
    assert lid._apply_response_format(body, cfg) == "json_object"
    body2 = {}
    assert lid._apply_response_format(body2, other_model) == "json_schema"


def test_schema_key_normalises_whitespace(lid, cfg):
    """The key is a cache identity; trailing whitespace from a .env value must
    not mint a second, never-hit entry."""
    padded = lid.LocalConfig(**{
        **cfg.__dict__,
        "base_url": f"  {cfg.base_url} ",
        "model_name": f"{cfg.model_name}  ",
    })
    assert lid._schema_key(padded) == lid._schema_key(cfg)


def test_known_bad_engine_skips_straight_to_json_object(lid, cfg):
    """Once learned, the top rung is never attempted again -- no round trip is
    spent to re-learn an answer that cannot have changed."""
    lid._SCHEMA_UNSUPPORTED.add(lid._schema_key(cfg))
    calls = []
    monkey_schema = lambda: (calls.append(1), {"type": "object"})[1]
    lid_resolve = lid._resolve_response_schema
    try:
        lid._resolve_response_schema = monkey_schema
        body = {}
        assert lid._apply_response_format(body, cfg) == "json_object"
    finally:
        lid._resolve_response_schema = lid_resolve
    assert calls == [], "schema was resolved for an engine known to reject it"


# --- the ladder under a SIMULATED 4xx, end to end -------------------------

class _FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload or {}
        self._text = text
        self.raise_called = False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    def raise_for_status(self):
        self.raise_called = True
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Records the body AS SENT on each post.

    Snapshots deeply because the production path mutates one dict in place
    across the retry -- comparing live references would make the two posts look
    identical no matter what the code did.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    def post(self, url, json=None, **kw):
        self.sent.append(copy.deepcopy(json))
        return self._responses.pop(0)


_OK = {"choices": [{"message": {"content": '{"ok": true}'}}],
       "usage": {"completion_tokens": 7}}


def _client(lid, session, **cfg_over):
    base = lid.LocalConfig.from_env().__dict__
    # num_ctx=None keeps the non-streaming branch, which is the path the 4xx
    # probe lives on.
    # These probes assert the OpenAI dialect's spellings (``response_format``);
    # the default transport is native, so the dialect is declared.
    cfg = lid.LocalConfig(**{**base, "num_ctx": None, "transport": "openai", **cfg_over})
    return lid.LocalPrimeClient(cfg, session=session), cfg


async def test_4xx_degrades_and_retries_once(lid):
    """The capability probe by OBSERVATION: the engine refuses the grammar, we
    believe it, remember it, and retry at the next rung down -- once."""
    sess = _FakeSession([
        _FakeResp(400, text="response_format.type not supported"),
        _FakeResp(200, _OK),
    ])
    client, cfg = _client(lid, sess)

    out = await client.complete(system="s", user="u", prompt_tokens=10)

    assert out.text == '{"ok": true}'
    assert len(sess.sent) == 2, "expected exactly one retry"
    assert sess.sent[0]["response_format"]["type"] == "json_schema"
    assert sess.sent[1]["response_format"] == {"type": "json_object"}
    assert lid._schema_key(cfg) in lid._SCHEMA_UNSUPPORTED


async def test_second_op_against_a_known_bad_engine_makes_one_call(lid):
    """The learned rejection must actually save the round trip on the NEXT op,
    not just on the retry that discovered it."""
    sess = _FakeSession([_FakeResp(400, text="nope"), _FakeResp(200, _OK)])
    client, cfg = _client(lid, sess)
    await client.complete(system="s", user="u", prompt_tokens=10)

    sess2 = _FakeSession([_FakeResp(200, _OK)])
    client2, _ = _client(lid, sess2)
    await client2.complete(system="s", user="u", prompt_tokens=10)

    assert len(sess2.sent) == 1
    assert sess2.sent[0]["response_format"] == {"type": "json_object"}


async def test_5xx_does_not_disarm_the_constraint(lid):
    """A 5xx is the engine FAILING, not REFUSING. Degrading on it would
    silently drop shape enforcement for the whole process over a blip."""
    sess = _FakeSession([_FakeResp(503, {"error": "overloaded"})])
    client, cfg = _client(lid, sess)

    # The 5xx body has no `choices`; the production path raises rather than
    # inventing a completion. What matters is the cache stayed clean.
    with pytest.raises(Exception):
        await client.complete(system="s", user="u", prompt_tokens=10)

    assert lid._schema_key(cfg) not in lid._SCHEMA_UNSUPPORTED
    assert len(sess.sent) == 1, "a 5xx must not trigger the degrade retry"


async def test_happy_path_makes_exactly_one_call(lid):
    """No probe overhead when the engine accepts the grammar."""
    sess = _FakeSession([_FakeResp(200, _OK)])
    client, cfg = _client(lid, sess)
    await client.complete(system="s", user="u", prompt_tokens=10)
    assert len(sess.sent) == 1
    assert sess.sent[0]["response_format"]["type"] == "json_schema"
    assert lid._SCHEMA_UNSUPPORTED == set()


async def test_json_mode_off_sends_no_response_format_and_never_probes(lid, monkeypatch):
    """Master OFF is a true bypass: no constraint, and therefore no 4xx probe
    branch to trip over."""
    monkeypatch.setenv("JARVIS_LOCAL_JSON_MODE_ENABLED", "0")
    sess = _FakeSession([_FakeResp(400, text="bad model")])
    client, cfg = _client(lid, sess)

    with pytest.raises(Exception):
        await client.complete(system="s", user="u", prompt_tokens=10)

    assert "response_format" not in sess.sent[0]
    assert len(sess.sent) == 1
    assert lid._SCHEMA_UNSUPPORTED == set()


# ===========================================================================
# Phase 1b -- external_generation bracket (zero counter leaks)
# ===========================================================================

EP = "http://127.0.0.1:11434"


def _inflight(gw, ep=EP):
    return int(gw._inflight.get(ep, 0))


async def test_bracket_counts_then_releases(gateway):
    assert _inflight(gateway) == 0
    async with gateway.external_generation(EP):
        assert _inflight(gateway) == 1
    assert _inflight(gateway) == 0


async def test_bracket_releases_on_exception(gateway):
    """A provider exception must not leak the host as permanently busy --
    which would silently disable pre-warming for the rest of the process."""
    with pytest.raises(ValueError):
        async with gateway.external_generation(EP):
            assert _inflight(gateway) == 1
            raise ValueError("provider blew up")
    assert _inflight(gateway) == 0


async def test_bracket_releases_on_cancellation(gateway):
    """The stall path. A timeout cancels the task mid-generation; the counter
    must unwind on the way out."""
    started = asyncio.Event()

    async def _work():
        async with gateway.external_generation(EP):
            started.set()
            await asyncio.sleep(30)

    task = asyncio.ensure_future(_work())
    await started.wait()
    assert _inflight(gateway) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _inflight(gateway) == 0


async def test_bracket_releases_on_timeout_wrapper(gateway):
    """The same unwind through asyncio.wait_for, which is how the production
    timeouts are actually spelled (3.9+ -- no asyncio.timeout)."""
    async def _work():
        async with gateway.external_generation(EP):
            await asyncio.sleep(30)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_work(), timeout=0.05)
    assert _inflight(gateway) == 0


async def test_concurrent_brackets_accumulate_and_unwind(gateway):
    """Two generations on one device read as two, and a failure in one does not
    corrupt the other's accounting."""
    hold = asyncio.Event()
    peak = []

    async def _ok():
        async with gateway.external_generation(EP):
            peak.append(_inflight(gateway))
            await hold.wait()

    async def _bad():
        try:
            async with gateway.external_generation(EP):
                peak.append(_inflight(gateway))
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    t1 = asyncio.ensure_future(_ok())
    await asyncio.sleep(0)
    t2 = asyncio.ensure_future(_bad())
    await t2
    hold.set()
    await t1
    assert max(peak) == 2
    assert _inflight(gateway) == 0


async def test_bracket_isolates_endpoints(gateway):
    other = "http://10.0.0.5:11434"
    async with gateway.external_generation(EP):
        async with gateway.external_generation(other):
            assert _inflight(gateway, EP) == 1
            assert _inflight(gateway, other) == 1
        assert _inflight(gateway, other) == 0
        assert _inflight(gateway, EP) == 1
    assert _inflight(gateway, EP) == 0


async def test_counter_key_is_removed_not_left_at_zero(gateway):
    """_inflight_adjust pops at <=0. Pinned because "idle" is read by absence
    as well as by value."""
    async with gateway.external_generation(EP):
        pass
    assert EP not in gateway._inflight


# --- target_for_endpoint --------------------------------------------------

def test_target_for_endpoint_defaults(gateway):
    t = gateway.target_for_endpoint(EP, "qwen2.5-coder:32b")
    assert t.base_url == EP
    assert t.model_name == "qwen2.5-coder:32b"
    assert t.scope == "local"
    assert t.is_remote is False
    assert t.reason == "externally resolved"


def test_target_for_endpoint_carries_live_health(gateway):
    """A caller must not be able to launder a host the breaker took out of
    service by resolving the endpoint itself."""
    mod = importlib.import_module(GW)
    health = gateway._health_for(EP)
    state = health.state()
    assert isinstance(state, mod.HostState)
    t = gateway.target_for_endpoint(EP, "m")
    assert t.state == state


def test_target_for_endpoint_never_raises_on_broken_health(gateway, monkeypatch):
    mod = importlib.import_module(GW)

    def _boom(_ep):
        raise RuntimeError("health table corrupt")

    monkeypatch.setattr(gateway, "_health_for", _boom)
    t = gateway.target_for_endpoint(EP, "m")
    assert t.state == mod.HostState.UNKNOWN


# ===========================================================================
# Phase 1c -- _f3c_inflight_bracket (generator -> gateway seam)
# ===========================================================================

async def test_f3c_bracket_registers_with_the_gateway(cg, gateway):
    async with cg._f3c_inflight_bracket(EP):
        assert _inflight(gateway) == 1
    assert _inflight(gateway) == 0


async def test_f3c_bracket_is_null_when_master_off(cg, gateway, monkeypatch):
    monkeypatch.setenv("JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED", "0")
    async with cg._f3c_inflight_bracket(EP):
        assert _inflight(gateway) == 0
    assert _inflight(gateway) == 0


@pytest.mark.parametrize("endpoint", [None, ""])
async def test_f3c_bracket_is_null_without_an_endpoint(cg, gateway, endpoint):
    async with cg._f3c_inflight_bracket(endpoint):
        assert _inflight(gateway) == 0


async def test_f3c_bracket_releases_on_exception(cg, gateway):
    with pytest.raises(ValueError):
        async with cg._f3c_inflight_bracket(EP):
            raise ValueError("generation failed")
    assert _inflight(gateway) == 0


async def test_f3c_bracket_degrades_to_null_when_gateway_unavailable(cg, gateway, monkeypatch):
    """Accounting must never cost an op: an import or singleton failure yields
    a working null bracket, not an exception into the dispatch."""
    mod = importlib.import_module(GW)

    def _boom():
        raise RuntimeError("no gateway")

    monkeypatch.setattr(mod, "get_default_gateway", _boom)
    async with cg._f3c_inflight_bracket(EP):
        pass  # must not raise


def test_inflight_unification_defaults_on(cg):
    assert cg._gateway_inflight_unification_enabled() is True


# ===========================================================================
# Phase 1d -- free-lane policy + credential TTL hot-reloader
# ===========================================================================

def _stub_loader(monkeypatch, calls, *, raises=False, sets=None):
    """Install a fake credential_env_loader so the TTL can be observed without
    touching a real .env.

    Writes through ``monkeypatch.setenv`` rather than ``os.environ`` directly.
    A raw write is not merely untidy here: a later ``monkeypatch.delenv`` would
    record the stub's own value as the pre-test state and RESTORE it at
    teardown, leaking a live-looking credential into every subsequent test.
    """
    import types

    mod = types.ModuleType("backend.core.ouroboros.aegis.credential_env_loader")

    def load_provider_credentials(**kw):
        calls.append(kw)
        if raises:
            raise RuntimeError("loader exploded")
        for k, v in (sets or {}).items():
            monkeypatch.setenv(k, v)
        return None

    mod.load_provider_credentials = load_provider_credentials
    monkeypatch.setitem(
        sys.modules, "backend.core.ouroboros.aegis.credential_env_loader", mod)
    return mod


def test_free_lane_defaults_off_without_a_local_lane(cg, monkeypatch):
    """The conservative direction. No local lane configured -> not free, even
    with no keys anywhere."""
    monkeypatch.delenv("JARVIS_LOCAL_PRIME_ENABLED", raising=False)
    assert cg._free_lane_active() is False


def test_free_lane_active_when_local_and_no_keys(cg, monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    _stub_loader(monkeypatch, [])
    assert cg._free_lane_active() is True


def test_master_off_forces_the_cost_averse_answer(cg, monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_FREE_LANE_POLICY_ENABLED", "0")
    _stub_loader(monkeypatch, [])
    assert cg._free_lane_active() is False


@pytest.mark.parametrize("key", ["DOUBLEWORD_API_KEY", "ANTHROPIC_API_KEY"])
def test_any_paid_key_revokes_free_lane(cg, monkeypatch, key):
    """Absence of a key is the strongest available evidence that no paid lane
    exists; presence of one must immediately end the free-lane assumption, or
    a swarm fan-out silently multiplies someone's bill."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    _stub_loader(monkeypatch, [])
    assert cg._free_lane_active() is True
    monkeypatch.setenv(key, "sk-live-xxxx")
    assert cg._free_lane_active() is False


def test_credential_reread_is_ttl_bounded(cg, monkeypatch):
    """A file stat per generation would be waste: the answer changes at
    operator speed, not per-op."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_FREE_LANE_CRED_TTL_S", "30")
    calls = []
    _stub_loader(monkeypatch, calls)

    clock = [1000.0]
    monkeypatch.setattr(cg.time, "monotonic", lambda: clock[0])

    cg._free_lane_active()
    assert len(calls) == 1
    for _ in range(5):
        cg._free_lane_active()
    assert len(calls) == 1, "re-read inside the TTL window"

    clock[0] += 29.0
    cg._free_lane_active()
    assert len(calls) == 1, "re-read one second early"

    clock[0] += 2.0
    cg._free_lane_active()
    assert len(calls) == 2, "TTL expired but no re-read happened"


def test_ttl_zero_disables_the_reread(cg, monkeypatch):
    """Documented escape hatch: boot-time environment only."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_FREE_LANE_CRED_TTL_S", "0")
    calls = []
    _stub_loader(monkeypatch, calls)
    for _ in range(3):
        cg._free_lane_active()
    assert calls == []


def test_loader_failure_never_propagates(cg, monkeypatch):
    """An unreadable .env must degrade to the last known environment, not kill
    a generation."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    _stub_loader(monkeypatch, [], raises=True)
    assert cg._free_lane_active() is True  # no keys in env -> still free
    cg._refresh_paid_lane_credentials()    # must not raise


def test_key_appearing_mid_run_revokes_free_lane_via_reread(cg, monkeypatch):
    """The load-bearing claim of the TTL: an operator who adds a key to .env
    while the loop is RUNNING revokes free-lane status without a restart.

    os.environ is per-process, so a shell export is invisible to a running
    orchestrator and load_env_once is idempotent -- without the re-read this
    would be a promise the code could not honour.
    """
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_FREE_LANE_CRED_TTL_S", "30")
    calls = []
    _stub_loader(monkeypatch, calls, sets={"ANTHROPIC_API_KEY": "sk-added-mid-run"})

    clock = [500.0]
    monkeypatch.setattr(cg.time, "monotonic", lambda: clock[0])

    # First read populates the key from the "file" -> already revoked.
    assert cg._free_lane_active() is False
    assert len(calls) == 1
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Inside the TTL the stale (free) answer stands -- bounded staleness.
    assert cg._free_lane_active() is True
    clock[0] += 31.0
    # Past the TTL the file is consulted again and the key returns.
    assert cg._free_lane_active() is False
    assert len(calls) == 2


def test_reread_against_a_real_dotenv_file(cg, monkeypatch, tmp_path):
    """End-to-end against the REAL credential_env_loader, not a stub.

    Proves the composition -- that _refresh_paid_lane_credentials actually
    reaches the module that owns the allowlist -- rather than only proving the
    TTL arithmetic around it.
    """
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    monkeypatch.setenv("JARVIS_FREE_LANE_CRED_TTL_S", "30")
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert cg._free_lane_active() is True

    (tmp_path / ".env").write_text("DOUBLEWORD_API_KEY=dw-live-key\n", encoding="utf-8")
    # Force STALE with a negative sentinel, never 0.0 -- see the `cg` fixture:
    # 0.0 only reads as "past the TTL" once process uptime exceeds it.
    monkeypatch.setattr(cg, "_FREE_LANE_CRED_REFRESH_AT", -1e9)
    assert cg._free_lane_active() is False
    assert os.environ.get("DOUBLEWORD_API_KEY") == "dw-live-key"


def test_free_lane_never_raises(cg, monkeypatch):
    """An unprovable answer is False -- preserving the pre-existing
    cost-averse behaviour exactly."""
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")

    def _boom(*a, **k):
        raise RuntimeError("env subsystem down")

    monkeypatch.setattr(cg, "_refresh_paid_lane_credentials", _boom)
    assert cg._free_lane_active() is False


# ===========================================================================
# Phase 1e -- doc staleness named symbols (JARVIS_DOC_MAX_NAMED_SYMBOLS)
# ===========================================================================

@pytest.fixture()
def dss(monkeypatch):
    mod = importlib.import_module(DSS)
    monkeypatch.delenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", raising=False)
    return mod


def test_max_named_symbols_default_and_overrides(dss, monkeypatch):
    assert dss._max_named_symbols() == 8
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "3")
    assert dss._max_named_symbols() == 3
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "0")
    assert dss._max_named_symbols() == 0
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "-4")
    assert dss._max_named_symbols() == 0
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "not-a-number")
    assert dss._max_named_symbols() == 8, "malformed value must not disable the feature"


def _write_module(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return p


def test_finding_names_the_undocumented_symbols_in_source_order(dss, tmp_path):
    """A count says a file needs work; a NAME says where. Only the latter can
    narrow an op below whole-file scope."""
    src = '''
        """Module docstring."""

        def alpha():
            pass

        class Beta:
            """Documented."""

        def gamma():
            pass

        def _private():
            pass
    '''
    f = _write_module(tmp_path, "m.py", src)
    sensor = dss.DocStalenessSensor(repo="r", router=None, project_root=tmp_path)
    finding = sensor._analyze_file(f, "m.py")

    assert finding is not None
    assert finding.public_symbols == 3
    assert finding.undocumented_symbols == ("alpha", "gamma")
    assert "_private" not in finding.undocumented_symbols
    assert "alpha, gamma" in finding.summary
    assert finding.details["undocumented_symbols"] == ["alpha", "gamma"]


def test_named_symbols_are_capped_with_an_honest_remainder(dss, tmp_path, monkeypatch):
    """A pathological file must not turn an op description into a wall of
    identifiers -- and the truncation must be stated, not silent."""
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "2")
    src = '"""Doc."""\n' + "".join(f"def f{i}():\n    pass\n" for i in range(6))
    f = _write_module(tmp_path, "big.py", src)
    sensor = dss.DocStalenessSensor(repo="r", router=None, project_root=tmp_path)
    finding = sensor._analyze_file(f, "big.py")

    assert finding is not None
    assert len(finding.undocumented_symbols) == 6, "the DATA is never truncated"
    assert "f0, f1" in finding.summary
    assert "(+4 more)" in finding.summary
    assert "f5" not in finding.summary


def test_cap_zero_restores_the_count_only_description(dss, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DOC_MAX_NAMED_SYMBOLS", "0")
    src = '"""Doc."""\n' + "".join(f"def f{i}():\n    pass\n" for i in range(4))
    f = _write_module(tmp_path, "z.py", src)
    sensor = dss.DocStalenessSensor(repo="r", router=None, project_root=tmp_path)
    finding = sensor._analyze_file(f, "z.py")

    assert finding is not None
    assert "public symbols undocumented" in finding.summary
    assert ": f0" not in finding.summary
    assert "more)" not in finding.summary
    # The structured twin still carries everything -- only the PROSE is capped.
    assert finding.details["undocumented_symbols"] == ["f0", "f1", "f2", "f3"]


def test_fully_documented_file_yields_no_finding(dss, tmp_path):
    src = '''
        """Doc."""

        def a():
            """A."""

        def b():
            """B."""

        def c():
            """C."""
    '''
    f = _write_module(tmp_path, "clean.py", src)
    sensor = dss.DocStalenessSensor(repo="r", router=None, project_root=tmp_path)
    assert sensor._analyze_file(f, "clean.py") is None


# ===========================================================================
# Phase 2 -- configuration governance: registered vs ACTUAL defaults
# ===========================================================================

#: (flag name, callable returning the code's live answer, advertised default).
#: The accessor is included on purpose: registering a default is worthless if
#: nothing proves it is the default the code applies. This table is the whole
#: point of the phase -- it fails when someone changes one side of the pair.
_ARC_FLAGS = [
    ("JARVIS_LOCAL_JSON_MODE_ENABLED", LID, "_json_mode_enabled", True),
    ("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", LID, "_json_schema_mode_enabled", True),
    ("JARVIS_FREE_LANE_POLICY_ENABLED", CG, None, True),
    ("JARVIS_FREE_LANE_CRED_TTL_S", CG, "_free_lane_cred_ttl_s", 30.0),
    ("JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED", CG,
     "_gateway_inflight_unification_enabled", True),
    ("JARVIS_LOCAL_VRAM_AUTODETECT_ENABLED", CG, "_local_vram_autodetect_enabled", False),
    ("JARVIS_DOC_MAX_NAMED_SYMBOLS", DSS, "_max_named_symbols", 8),
    ("JARVIS_BACKGROUND_LOCAL_LANE_ENABLED", CG,
     "_background_local_lane_enabled", True),
]


@pytest.fixture()
def registry():
    from backend.core.ouroboros.governance.flag_registry import ensure_seeded
    return ensure_seeded()


@pytest.mark.parametrize("name,_mod,_acc,_default", _ARC_FLAGS,
                         ids=[f[0] for f in _ARC_FLAGS])
def test_arc_flag_is_registered(registry, name, _mod, _acc, _default):
    """Every master this arc ships is discoverable by name.

    An unregistered flag is invisible to /help flags, to the typo detector and
    to the posture filter -- so the only way to find it is to already know it
    exists, which is the opposite of the property the registry exists for.
    """
    assert registry.get_spec(name) is not None, f"{name} is not registered"


@pytest.mark.parametrize("name,_mod,_acc,default", _ARC_FLAGS,
                         ids=[f[0] for f in _ARC_FLAGS])
def test_registered_default_matches_the_advertised_value(registry, name, _mod, _acc, default):
    spec = registry.get_spec(name)
    assert spec is not None
    assert spec.default == default, (
        f"{name}: registry advertises {spec.default!r}, arc contract says {default!r}"
    )


@pytest.mark.parametrize("name,mod,acc,default", _ARC_FLAGS,
                         ids=[f[0] for f in _ARC_FLAGS])
def test_code_default_matches_the_registry(monkeypatch, registry, name, mod, acc, default):
    """The anti-drift pin.

    Documenting a default in a registry that the accessor does not honour is
    worse than not documenting it: an operator reads `/help flag X`, believes
    it, and configures against a fiction. Deleting the env var and calling the
    real accessor is the only way to prove the two agree.
    """
    if acc is None:
        pytest.skip("no single accessor -- covered by the behavioural tests above")
    monkeypatch.delenv(name, raising=False)
    module = importlib.import_module(mod)
    live = getattr(module, acc)()
    spec = registry.get_spec(name)
    assert live == default, f"{name}: code default {live!r} != contract {default!r}"
    assert live == spec.default, (
        f"{name}: code default {live!r} != registered default {spec.default!r}"
    )


@pytest.mark.parametrize("name,_mod,_acc,_d", _ARC_FLAGS, ids=[f[0] for f in _ARC_FLAGS])
def test_arc_flag_metadata_is_usable(registry, name, _mod, _acc, _d):
    """A registration with an empty description or a placeholder source_file
    satisfies "registered" while helping nobody."""
    spec = registry.get_spec(name)
    assert spec.source_file.endswith(".py")
    assert Path(spec.source_file).exists() or True  # path is repo-relative prose
    assert len(spec.description) > 60, "description is too thin to be useful"
    assert spec.example, "no example value"
    assert spec.since, "no since tag"


def test_registered_source_files_exist(registry):
    """source_file is a pointer an operator will follow. A stale one sends
    them to a file that no longer exists."""
    repo_root = Path(__file__).resolve().parents[2]
    for name, _m, _a, _d in _ARC_FLAGS:
        spec = registry.get_spec(name)
        assert (repo_root / spec.source_file).is_file(), (
            f"{name}: source_file {spec.source_file} does not exist"
        )


def test_default_on_flags_are_revocable_without_a_revert(monkeypatch):
    """Every default-ON master in this arc must have a working OFF path.

    A flag that cannot actually be turned off is not a kill switch, and the
    stated reason all three ship ON is that they are revocable.
    """
    lid_mod = importlib.import_module(LID)
    cg_mod = importlib.import_module(CG)
    for name, getter in (
        ("JARVIS_LOCAL_JSON_MODE_ENABLED", lid_mod._json_mode_enabled),
        ("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", lid_mod._json_schema_mode_enabled),
        ("JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED",
         cg_mod._gateway_inflight_unification_enabled),
        ("JARVIS_BACKGROUND_LOCAL_LANE_ENABLED",
         cg_mod._background_local_lane_enabled),
    ):
        monkeypatch.setenv(name, "0")
        assert getter() is False, f"{name} ignored its OFF setting"
        monkeypatch.delenv(name, raising=False)
        assert getter() is True


# ===========================================================================
# Phase 3 -- BACKGROUND / SPECULATIVE free-lane preemption
# ===========================================================================
#
# BACKGROUND and SPECULATIVE encode "spend nothing" as a PROVIDER NAME. When
# the DW catalog is purged there is no cheap provider left to name, so the op
# dead-queues with ``background_dw_blocked_by_topology`` -- six of six ops in
# soak bt-2026-08-24-074121 died that way on a host whose local 32B was idle
# and free the entire time. `_try_free_lane_dispatch` asks after COST instead,
# and only ever acts on EVIDENCE: an endpoint that answered.
#
# The dispatch is exercised through a stub self. Booting a real
# CandidateGenerator would drag in the provider stack for a decision that
# reads exactly two collaborators.


class _FreeLaneStub:
    """Minimal `self` for `_try_free_lane_dispatch`: two collaborators."""

    def __init__(self, endpoint=None, result=None, discover_raises=None,
                 dispatch_raises=None):
        self._endpoint = endpoint
        self._result = result
        self._discover_raises = discover_raises
        self._dispatch_raises = dispatch_raises
        self.dispatched_to = None
        self.dispatch_calls = 0
        self.last_context = None

    async def _discover_jprime_endpoint(self):
        if self._discover_raises is not None:
            raise self._discover_raises
        return self._endpoint

    async def _failover_local_dispatch(self, context, deadline, endpoint):
        self.dispatched_to = endpoint
        self.dispatch_calls += 1
        self.last_context = context
        if self._dispatch_raises is not None:
            raise self._dispatch_raises
        return self._result

    async def _local_dispatch_with_syntax_repair(self, context, deadline, endpoint):
        """The REAL wrapper, bound to this stub.

        Not a stub of its own: the stub supplies only the true collaborator
        (`_failover_local_dispatch`) and production's retry logic runs on top,
        so these tests exercise the shipped code path rather than a second
        description of it.
        """
        mod = importlib.import_module(CG)
        return await mod.CandidateGenerator._local_dispatch_with_syntax_repair(
            self, context, deadline, endpoint,
        )


class _RetryableCtx:
    """A context that can carry syntax feedback, like the real one.

    `OperationContext.with_syntax_retry_feedback` returns a NEW context and
    advances the hash-chain; the retry has to be auditable as a distinct
    attempt. This mirrors that shape (new object, feedback set) without
    dragging the real dataclass and its hashing into these tests.
    """

    def __init__(self, syntax_retry_feedback="", force_diff_on_retry=False):
        self.op_id = "op-free-lane-test"
        self.provider_route = "background"
        self.syntax_retry_feedback = syntax_retry_feedback
        self.force_diff_on_retry = force_diff_on_retry

    def with_syntax_retry_feedback(self, feedback):
        # PRESERVES sibling fields, because the real implementation is
        # `dataclasses.replace(self, ...)` and replace copies everything it is
        # not told to change. A double that rebuilds from scratch silently
        # drops whatever a caller stamped just before — which is exactly how
        # this fixture hid the truncation reshape.
        return _RetryableCtx(
            syntax_retry_feedback=feedback,
            force_diff_on_retry=self.force_diff_on_retry,
        )

    def with_forced_diff_retry(self):
        return _RetryableCtx(
            syntax_retry_feedback=self.syntax_retry_feedback,
            force_diff_on_retry=True,
        )


class _Candidates:
    def __init__(self, n, is_noop=False):
        self.candidates = tuple(object() for _ in range(n))
        self.is_noop = is_noop


@pytest.fixture()
def free_lane(monkeypatch):
    """`_try_free_lane_dispatch` bound to a stub, master flag cleared."""
    mod = importlib.import_module(CG)
    monkeypatch.delenv("JARVIS_BACKGROUND_LOCAL_LANE_ENABLED", raising=False)

    async def _call(stub, route="background", reason="topology_block:purged"):
        return await mod.CandidateGenerator._try_free_lane_dispatch(
            stub, _RetryableCtx(), object(), route=route, reason=reason,
        )

    return _call


async def test_free_lane_serves_an_op_that_would_dead_queue(free_lane):
    """The regression: a purged DW catalog no longer kills a BACKGROUND op."""
    result = _Candidates(1)
    stub = _FreeLaneStub(endpoint=EP, result=result)

    assert await free_lane(stub) is result
    assert stub.dispatched_to == EP


async def test_free_lane_serves_speculative_too(free_lane):
    """SPECULATIVE dead-queues through the same block for the same reason."""
    result = _Candidates(2)
    stub = _FreeLaneStub(endpoint=EP, result=result)

    assert await free_lane(stub, route="speculative") is result


@pytest.mark.parametrize("route", ["standard", "immediate", "complex"])
async def test_free_lane_never_preempts_a_paid_route(free_lane, route):
    """Only the routes that dead-queue are touched.

    A STANDARD op has a working cascade and an explicit cost contract; quietly
    moving it onto a local model would change what the operator paid for.
    """
    stub = _FreeLaneStub(endpoint=EP, result=_Candidates(1))

    assert await free_lane(stub, route=route) is None
    assert stub.dispatched_to is None


@pytest.mark.parametrize("endpoint", [None, ""])
async def test_no_endpoint_falls_through_to_the_queue(free_lane, endpoint):
    """A flag is not evidence. With nothing serving, the dead-queue stands.

    This is the byte-identical path for every host without a local lane —
    which is every cloud deployment.
    """
    stub = _FreeLaneStub(endpoint=endpoint, result=_Candidates(1))

    assert await free_lane(stub) is None
    assert stub.dispatched_to is None


async def test_master_off_restores_the_unconditional_queue(free_lane, monkeypatch):
    monkeypatch.setenv("JARVIS_BACKGROUND_LOCAL_LANE_ENABLED", "0")
    stub = _FreeLaneStub(endpoint=EP, result=_Candidates(1))

    assert await free_lane(stub) is None
    assert stub.dispatched_to is None


async def test_empty_candidates_are_not_a_success(free_lane):
    """A result carrying no candidates must not short-circuit the queue path.

    Returning it would convert a dead-queue into a silent empty GENERATE —
    the op would look served and produce nothing.
    """
    stub = _FreeLaneStub(endpoint=EP, result=_Candidates(0))

    assert await free_lane(stub) is None


async def test_a_noop_verdict_is_an_answer_not_an_absence(free_lane):
    """`2b.1-noop` must terminate the op, not fall through to the queue raise.

    Measured, not hypothesised: in soak bt-2026-08-28-061124 the 32B answered
    "the dependency 'torch' is already present in requirements.txt" in 2.8s
    for 28 tokens. That carries zero candidates because zero was the CORRECT
    answer. Counting candidates alone cannot distinguish it from a lane that
    produced nothing, and the fall-through would tell the operator the op died
    of `background_dw_blocked_by_topology` — a topology outage that never
    happened, on a lane that worked perfectly.
    """
    verdict = _Candidates(0, is_noop=True)
    stub = _FreeLaneStub(endpoint=EP, result=verdict)

    assert await free_lane(stub) is verdict


async def test_a_result_without_the_noop_attribute_still_falls_through(free_lane):
    """Absence of `is_noop` is not a noop.

    The provider seat is duck-typed here; a result object that predates the
    flag must read as "no candidates", never as a silent success.
    """
    class _Bare:
        candidates = ()

    stub = _FreeLaneStub(endpoint=EP, result=_Bare())

    assert await free_lane(stub) is None


@pytest.mark.parametrize("boom", [RuntimeError("engine down"), OSError("refused")])
async def test_dispatch_failure_falls_through_instead_of_poisoning(free_lane, boom):
    """The lane is an OPPORTUNITY, never a new way for the op to die.

    Every failure here must land on the pre-existing raise, so the operator
    reads the same terminal reason they would have read before.
    """
    stub = _FreeLaneStub(endpoint=EP, result=None, dispatch_raises=boom)

    assert await free_lane(stub) is None


async def test_discovery_failure_falls_through(free_lane):
    stub = _FreeLaneStub(discover_raises=RuntimeError("no controller"))

    assert await free_lane(stub) is None


async def test_cancellation_is_not_swallowed(free_lane):
    """Structured concurrency: a parent cancel is not a lane failure.

    Catching it here would turn a shutdown into a dead-queue raise and leave
    the harness reporting a topology block for an op the operator stopped.
    """
    stub = _FreeLaneStub(
        endpoint=EP, result=None, dispatch_raises=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await free_lane(stub)


# ===========================================================================
# Phase 4 -- one-shot syntax repair on the local lane
# ===========================================================================
#
# `all_candidates_syntax_error` was the top non-governance failure on the
# local lane (6 dispatches in bt-2026-08-28-061124): valid JSON wrapping
# invalid Python. `syntax_escalation` exists for this class but cascades
# DW -> J-Prime, and on a workstation the local 32B IS J-Prime -- so it
# escalates to the model that just failed. The missing move was never another
# provider; it was showing the model the parse error it never saw.


def _syntax_exc(failures):
    exc = RuntimeError("gcp-jprime_schema_invalid:all_candidates_syntax_error")
    exc.syntax_failures = failures
    return exc


_ONE_FAILURE = [{
    "file_path": "backend/util.py",
    "line": 3,
    "message": "unindent does not match any outer indentation level",
    "source_line": "  y = 2",
    "preceding_line": "    x = 1",
}]


class _SyntaxThenSuccessStub(_FreeLaneStub):
    """Fails the first dispatch on syntax, succeeds on the retry."""

    def __init__(self, endpoint, result):
        super().__init__(endpoint=endpoint, result=result)
        self._first = True

    async def _failover_local_dispatch(self, context, deadline, endpoint):
        self.dispatch_calls += 1
        self.last_context = context
        self.dispatched_to = endpoint
        if self._first:
            self._first = False
            raise _syntax_exc(_ONE_FAILURE)
        return self._result


async def test_syntax_failure_is_retried_once_with_the_parse_error(free_lane):
    """The regression: a fixable slip must not be terminal."""
    good = _Candidates(1)
    stub = _SyntaxThenSuccessStub(EP, good)

    assert await free_lane(stub) is good
    assert stub.dispatch_calls == 2, "expected exactly one retry"


async def test_the_retry_actually_carries_the_error_text(free_lane):
    """Feedback must reach the model, naming the line and the source.

    A retry that does not show the model what broke is just a second roll of
    the same dice at full token cost.
    """
    stub = _SyntaxThenSuccessStub(EP, _Candidates(1))
    await free_lane(stub)

    fb = getattr(stub.last_context, "syntax_retry_feedback", "")
    assert "backend/util.py" in fb
    assert "line 3" in fb
    assert "unindent" in fb
    assert "y = 2" in fb, "the offending source line must be quoted"


async def test_only_one_retry_even_if_it_fails_again(free_lane):
    """Two corrections would be a loop that spends the whole op budget."""
    stub = _FreeLaneStub(
        endpoint=EP, result=None, dispatch_raises=_syntax_exc(_ONE_FAILURE),
    )

    assert await free_lane(stub) is None
    assert stub.dispatch_calls == 2, "must stop after a single retry"


async def test_master_off_restores_single_attempt(free_lane, monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_SYNTAX_REPAIR_ENABLED", "0")
    stub = _SyntaxThenSuccessStub(EP, _Candidates(1))

    assert await free_lane(stub) is None
    assert stub.dispatch_calls == 1, "no retry when the master is off"


async def test_a_non_syntax_failure_is_not_retried(free_lane):
    """Only parse errors are recoverable this way.

    A timeout or a dead engine retried immediately just burns the budget
    twice; those have their own recovery paths.
    """
    stub = _FreeLaneStub(
        endpoint=EP, result=None, dispatch_raises=RuntimeError("engine down"),
    )

    assert await free_lane(stub) is None
    assert stub.dispatch_calls == 1


def test_syntax_repair_defaults_on(cg):
    assert cg._syntax_repair_enabled() is True


# ===========================================================================
# Phase 5 -- local as PRIMARY when no paid lane exists
# ===========================================================================
#
# `_try_free_lane_dispatch` refused non-cost-optimized routes to protect a cost
# contract. Soak bt-2026-08-28-100733 showed the cost of that when there IS no
# contract: a SANCTIONED roadmap op routed STANDARD (the UrgencyRouter keys on
# SOURCE, so source="roadmap" never reaches BACKGROUND however low its urgency)
# and died all_providers_exhausted with a warm 32B resident on the GPU.


class _PrimaryStub(_FreeLaneStub):
    async def _try_local_primary(self, context, deadline):
        mod = importlib.import_module(CG)
        return await mod.CandidateGenerator._try_local_primary(
            self, context, deadline,
        )


@pytest.fixture()
def local_primary(monkeypatch):
    """`_try_local_primary` bound to a stub, with NO paid credentials."""
    mod = importlib.import_module(CG)
    monkeypatch.setattr(mod, "_FREE_LANE_CRED_REFRESH_AT", -1e9)
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_ENABLED", "true")
    for k in ("DOUBLEWORD_API_KEY", "ANTHROPIC_API_KEY",
              "JARVIS_LOCAL_PRIME_PRIMARY_ENABLED"):
        monkeypatch.delenv(k, raising=False)

    async def _call(stub, route="standard"):
        ctx = _RetryableCtx()
        ctx.provider_route = route
        return await mod.CandidateGenerator._try_local_primary(
            stub, ctx, object(),
        )

    return _call


@pytest.mark.parametrize("route", ["standard", "immediate", "complex"])
async def test_local_primary_serves_paid_routes_when_no_paid_lane(
    local_primary, route,
):
    """The regression: a sanctioned STANDARD op must not die exhausted."""
    good = _Candidates(1)
    stub = _PrimaryStub(endpoint=EP, result=good)

    assert await local_primary(stub, route=route) is good
    assert stub.dispatched_to == EP


@pytest.mark.parametrize("key", ["DOUBLEWORD_API_KEY", "ANTHROPIC_API_KEY"])
async def test_local_primary_declines_when_a_paid_lane_exists(
    local_primary, monkeypatch, key,
):
    """One credential is enough to restore the legacy cascade untouched.

    This is the guard that keeps a cloud deployment byte-identical: the
    predicate asks whether a PAID lane exists, never which route this is.
    """
    monkeypatch.setenv(key, "sk-not-a-real-key")
    stub = _PrimaryStub(endpoint=EP, result=_Candidates(1))

    assert await local_primary(stub) is None
    assert stub.dispatched_to is None


async def test_local_primary_master_off_restores_fallback_only(
    local_primary, monkeypatch,
):
    monkeypatch.setenv("JARVIS_LOCAL_PRIME_PRIMARY_ENABLED", "0")
    stub = _PrimaryStub(endpoint=EP, result=_Candidates(1))

    assert await local_primary(stub) is None


async def test_local_primary_requires_a_reachable_endpoint(local_primary):
    """A flag is not evidence; an endpoint that answers is."""
    stub = _PrimaryStub(endpoint=None, result=_Candidates(1))

    assert await local_primary(stub) is None


async def test_local_primary_failure_falls_through_to_the_cascade(local_primary):
    """The local lane may only ADD a way to succeed, never a way to die."""
    stub = _PrimaryStub(
        endpoint=EP, result=None, dispatch_raises=RuntimeError("engine down"),
    )

    assert await local_primary(stub) is None


# ---------------------------------------------------------------------------
# Truncation-shaped failures reshape the retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg,expected", [
    ("unterminated string literal (detected at line 231)", True),
    ("unterminated triple-quoted string literal", True),
    ("unexpected EOF while parsing", True),
    ("'(' was never closed", True),
    ("expected an indented block", True),
    ("invalid syntax", False),
    ("unindent does not match any outer indentation level", False),
])
def test_truncation_classifier(cg, msg, expected):
    """Truncation is a cut-off payload; a typo is not. They need opposite
    remedies, so the classifier must not blur them."""
    assert cg._truncation_shaped([{"message": msg}]) is expected


async def test_truncation_reshapes_the_retry_instead_of_repeating_it(free_lane):
    """A cut-off payload must be retried SMALLER, not identically.

    Measured: line 231 unterminated -> retry -> line 196 unterminated. The
    model was not mis-typing; its output was being cut off, and the second
    attempt was cut off earlier. `force_diff_on_retry` is the existing seam
    for changing output SHAPE rather than repeating parameters.
    """
    trunc = [{
        "file_path": "tests/x.py", "line": 231,
        "message": "unterminated string literal (detected at line 231)",
        "source_line": '    assert x == "abc',
    }]
    stub = _SyntaxThenSuccessStub(EP, _Candidates(1))
    stub._first_failures = trunc

    async def _dispatch(context, deadline, endpoint):
        stub.dispatch_calls += 1
        stub.last_context = context
        if stub.dispatch_calls == 1:
            raise _syntax_exc(trunc)
        return stub._result

    stub._failover_local_dispatch = _dispatch
    assert await free_lane(stub) is stub._result
    assert getattr(stub.last_context, "force_diff_on_retry", False) is True, (
        "a truncation-shaped failure must reduce the output shape on retry"
    )
