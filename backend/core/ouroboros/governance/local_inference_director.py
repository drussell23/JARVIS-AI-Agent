# backend/core/ouroboros/governance/local_inference_director.py
"""Local inference tier (J-Prime activation, Phase 3).

Three units (added across Phase 3 tasks): LatencyProfiler, LocalPrimeClient,
LocalInferenceDirector. Gated behind JARVIS_LOCAL_PRIME_ENABLED (default OFF ->
byte-identical legacy).
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

from .memory_pressure_gate import PressureLevel, get_default_gate, is_enabled as memory_gate_enabled

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def _envb(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in _TRUE


def local_prime_enabled() -> bool:
    """Master kill-switch. OFF means PrimeProvider gets no local client."""
    return _envb("JARVIS_LOCAL_PRIME_ENABLED", False)


def _json_schema_mode_enabled() -> bool:
    """Whether to constrain the sampler with a full JSON *Schema* rather than
    only ``json_object``.

    json_object guarantees the output PARSES; it says nothing about SHAPE, which
    is why ``candidate_0_missing_rationale`` and
    ``wrong_schema_version:__missing__`` survived it. A schema makes a missing
    required field unrepresentable. Default ON, with automatic per-model
    degradation below for engines that reject the field."""
    return _envb("JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED", True)


#: (base_url, model) pairs that have REJECTED a json_schema response_format.
#: Learned at run time from the engine's own 4xx rather than from a version
#: table: llama.cpp/ollama grammar support varies by build and by how exotic the
#: schema is, and a static capability list would be wrong the moment either
#: changes. One rejection is enough -- the answer cannot become "yes" without a
#: restart, and retrying a known-400 on every op would spend a round trip to
#: learn nothing.
_SCHEMA_UNSUPPORTED: "set" = set()


def _schema_key(cfg: "LocalConfig") -> "Tuple[str, str]":
    return ((cfg.base_url or "").strip(), (cfg.model_name or "").strip())


def _resolve_response_schema() -> "Optional[Dict[str, Any]]":
    """The sampler grammar, derived from the provider layer's own constants.

    Imported LAZILY: ``providers`` imports this module (PrimeProvider wraps
    LocalPrimeClient), so a module-level import here would be circular. Returns
    None if the provider layer is unavailable, which degrades to json_object
    rather than failing. NEVER raises."""
    try:
        from .providers import build_response_json_schema  # noqa: PLC0415
        # State-Driven Schema Constraint (tool_masking lever 3). While the op
        # is below the Iron Gate exploration floor this narrows the union to
        # the tool-call shape, making a premature patch -- or a premature
        # "already complete" -- UNREPRESENTABLE rather than merely rejected
        # after the fact. Returns None outside a tool loop, on every non-local
        # provider, and whenever the floor is met, so the default path is the
        # full union exactly as before.
        try:
            from .tool_masking import answer_shapes_allowed  # noqa: PLC0415
            _allow = answer_shapes_allowed()
        except Exception:  # noqa: BLE001 — narrowing is an optimisation
            _allow = None
        schema = build_response_json_schema(_allow)
        return schema if isinstance(schema, dict) and schema else None
    except Exception:  # noqa: BLE001
        return None


#: (base_url, model) pairs that have REJECTED a ``draft_num_predict``.
_DRAFT_UNSUPPORTED: "set" = set()


def _draft_num_predict() -> int:
    """Speculative tokens to draft per step. 0 disables.

    This is the REQUEST-TIME half of speculative decoding. Its deploy-time
    counterpart is the ``DRAFT`` Modelfile instruction, which binds a draft
    MODEL to a tag; this knob drives ollama's Multi-Token Prediction, where
    the model drafts its own continuations and needs no second model.

    Default 0 (off) deliberately. MTP requires an MTP-enabled build --
    ``qwen3.8:27b-mtp-q4_K_M`` rather than ``qwen3.8:27b`` -- and enabling
    it against a build without one is unmeasured behaviour on the hot path.
    Turn it on together with that tag, and measure.

    Draft-MODEL speculation additionally requires a shared tokenizer:
    qwen2.5-coder:7b can draft for qwen2.5-coder:32b (both pre=qwen2,
    BOS 151643) but NOT for qwen3.8:27b (pre=qwen35, BOS 248044).
    """
    try:
        return max(0, min(32, int(
            os.environ.get("JARVIS_LOCAL_DRAFT_NUM_PREDICT", "0")
        )))
    except (TypeError, ValueError):
        return 0


def _apply_draft_tokens(body: "Dict[str, Any]", cfg: "LocalConfig") -> int:
    """Attach ``draft_num_predict`` unless this engine has rejected it.

    NEVER raises: speculation is a throughput optimisation, and failing to
    request one must not fail the generation."""
    try:
        n = _draft_num_predict()
        if n <= 0 or _schema_key(cfg) in _DRAFT_UNSUPPORTED:
            return 0
        opts = body.setdefault("options", {})
        if isinstance(opts, dict):
            opts["draft_num_predict"] = n
            return n
        return 0
    except Exception:  # noqa: BLE001
        return 0


def _degrade_draft_tokens(body: "Dict[str, Any]", cfg: "LocalConfig") -> bool:
    """Record that this engine rejected speculation and drop it.

    Degrading to single-token generation is always safe: the OUTPUT is
    identical, only slower. That asymmetry is why this degrades on the
    first refusal without a second thought, where the schema ladder had to
    reason about what it was giving up."""
    try:
        key = _schema_key(cfg)
        if key in _DRAFT_UNSUPPORTED:
            return False
        _DRAFT_UNSUPPORTED.add(key)
        opts = body.get("options")
        if isinstance(opts, dict) and "draft_num_predict" in opts:
            opts.pop("draft_num_predict", None)
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _apply_sampling(body: "Dict[str, Any]", cfg: "LocalConfig") -> "Dict[str, Any]":
    """Attach the per-draw sampling point, if this config carries one.

    Written in BOTH spellings on purpose. The request targets an
    OpenAI-compatible ``/v1/chat/completions``, where ``top_p`` and
    ``seed`` are standard top-level fields; ``top_k`` and
    ``repeat_penalty`` have no OpenAI spelling and exist only as
    ollama-native ``options``. Writing each field where its engine looks
    for it is what makes the knob actually bite -- a value the engine
    silently ignores looks wired and changes nothing, which is precisely
    the failure this module exists to end (siblings were nominally
    "diversified by temperature" while every draw ran at 0.2).

    Unknown fields are ignored by both engines, so the dual spelling
    cannot break a request. Returns what it set, for logging.

    NEVER raises: sampling is an improvement to a draw, and failing to
    request one must not fail the generation."""
    applied: "Dict[str, Any]" = {}
    try:
        if cfg.top_p is not None:
            body["top_p"] = float(cfg.top_p)
            applied["top_p"] = float(cfg.top_p)
        if cfg.seed is not None:
            body["seed"] = int(cfg.seed)
            applied["seed"] = int(cfg.seed)
        native = {
            "top_p": None if cfg.top_p is None else float(cfg.top_p),
            "top_k": None if cfg.top_k is None else int(cfg.top_k),
            "repeat_penalty": (
                None if cfg.repeat_penalty is None else float(cfg.repeat_penalty)
            ),
            "seed": None if cfg.seed is None else int(cfg.seed),
        }
        native = {k: v for k, v in native.items() if v is not None}
        if native:
            opts = body.setdefault("options", {})
            if isinstance(opts, dict):
                opts.update(native)
                applied.update(native)
        return applied
    except Exception:  # noqa: BLE001
        return applied


def _sampling_overrides(sampling: "Optional[Any]") -> "Dict[str, Any]":
    """The ``LocalConfig`` fields a sampling point sets. ONE reader.

    Duck-typed on purpose: this module must not import ``sibling_entropy``
    (that module imports nothing from here, and the dependency would close
    a cycle through the governance package). Anything exposing
    ``config_overrides()`` works, and so does a plain mapping — which is
    what makes both the request seam and the timeout seam testable without
    constructing a ladder.

    Unknown keys are DROPPED rather than raising: a provider that learns a
    new sampling field before this dataclass does must degrade to the
    fields we understand, not fail the generation. NEVER raises.
    """
    if sampling is None:
        return {}
    try:
        raw = sampling
        getter = getattr(sampling, "config_overrides", None)
        if callable(getter):
            raw = getter()
        if not isinstance(raw, dict):
            return {}
        allowed = {"top_p", "top_k", "repeat_penalty", "seed"}
        return {k: v for k, v in raw.items() if k in allowed and v is not None}
    except Exception:  # noqa: BLE001
        logger.debug("[LocalPrimeClient] sampling override unreadable", exc_info=True)
        return {}


def entropy_latency_factor(
    temperature: "Optional[float]", sampling: "Optional[Any]",
) -> float:
    """How much LONGER a draw at this sampling point is expected to run.

    ## Why this scales output length, not the timeout

    Entropy does not make a token slower to produce — it makes the model
    produce MORE of them. A flatter distribution (higher temperature, a
    wider ``top_k``) lowers the probability mass on EOS at every step, and
    ``repeat_penalty`` actively discourages the repetition a model would
    otherwise use to wind a passage down. The cost lands on the token
    COUNT, and the adaptive timeout already prices tokens
    (``ttft + per_token * est_out``) from measured per-token latency.

    So this returns a multiplier for ``est_out``, and every statistical
    property of the existing profiler — warm/cold, the sigma margin, the
    EWMA escalation, the absolute breaker — continues to hold. A flat
    multiplier on the whole timeout would have inflated the fixed
    time-to-first-token as if entropy slowed the prefill, which it does
    not, and would have escaped the breaker's meaning.

    Soak 14 is why this exists: threading the real sampling point produced
    89 ``TimeoutError`` and a `session_exhausted` 25 minutes early, because
    the draws got genuinely longer and the budget was still sized for
    temperature 0.2.

    Returns exactly 1.0 for the legacy point (no sampling, or temperature
    at the baseline), so a non-sibling generation keeps a byte-identical
    budget. Bounded above so a runaway rung cannot inflate a budget without
    limit — the breaker stays the authority on a wedged model. NEVER raises.
    """
    if not _entropy_latency_enabled():
        return 1.0
    try:
        over = _sampling_overrides(sampling)
        t_base = _f_env("JARVIS_LOCAL_ENTROPY_TEMP_BASELINE", 0.2)
        k_base = max(1.0, _f_env("JARVIS_LOCAL_ENTROPY_TOPK_BASELINE", 40.0))

        factor = 1.0
        if temperature is not None:
            factor += _f_env("JARVIS_LOCAL_ENTROPY_TEMP_COEFF", 0.6) * max(
                0.0, float(temperature) - t_base)
        top_k = over.get("top_k")
        if top_k is not None and float(top_k) > k_base:
            # Logarithmic: doubling the candidate pool does not double the
            # length, it widens the tail the sampler can wander into.
            factor += _f_env("JARVIS_LOCAL_ENTROPY_TOPK_COEFF", 0.25) * math.log2(
                float(top_k) / k_base)
        rp = over.get("repeat_penalty")
        if rp is not None:
            factor += _f_env("JARVIS_LOCAL_ENTROPY_RP_COEFF", 0.5) * max(
                0.0, float(rp) - 1.0)

        ceiling = max(1.0, _f_env("JARVIS_LOCAL_ENTROPY_FACTOR_MAX", 2.5))
        return float(max(1.0, min(ceiling, factor)))
    except Exception:  # noqa: BLE001 — a budget hint must never fail a draw
        logger.debug("[LocalPrimeClient] entropy latency factor failed", exc_info=True)
        return 1.0


def _prefill_context_scale(prompt_tokens: "Optional[int]") -> float:
    """How much longer than baseline THIS prompt's prefill should be allowed.

    Prefill cost is close to linear in prompt length, and the first-token
    deadline is the one wait that measures prefill. The baseline is the
    same ``JARVIS_LOCAL_SEED_CTX_BASELINE`` that ``_cold_seed_ms`` scales the
    total budget by -- one notion of "a normal-sized prompt", not two.

    Bounded above (``JARVIS_LOCAL_PREFILL_SCALE_MAX``) so a pathological
    prompt cannot buy an unbounded first-token wait; the absolute ceiling
    still owns the wedged-model case. >= 1.0 always: a short prompt never
    TIGHTENS the deadline below the static value. NEVER raises.
    """
    try:
        if prompt_tokens is None or prompt_tokens <= 0:
            return 1.0
        baseline = max(1, _int_env("JARVIS_LOCAL_SEED_CTX_BASELINE", 8192))
        ceiling = max(1.0, _f_env("JARVIS_LOCAL_PREFILL_SCALE_MAX", 6.0))
        return float(max(1.0, min(ceiling, float(prompt_tokens) / baseline)))
    except Exception:  # noqa: BLE001
        return 1.0


def _entropy_latency_enabled() -> bool:
    raw = (os.environ.get("JARVIS_LOCAL_ENTROPY_LATENCY_ENABLED", "true") or "")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _config_for_draw(
    cfg: "LocalConfig", sampling: "Optional[Any]",
) -> "LocalConfig":
    """The config THIS ONE draw runs under. Never mutates ``cfg``.

    The sampling point is per-draw state; the client's ``_cfg`` is
    process-wide. Writing the point onto ``self._cfg`` would leak one
    sibling's temperature/seed into every later op on the same client --
    so the override is applied immutably here and the caller passes the
    result down, leaving the shared default untouched.

    ``sampling`` is duck-typed on purpose: this module must not import
    ``sibling_entropy`` (that module already imports nothing from here,
    and the dependency would close a cycle through the governance
    package). Anything exposing ``config_overrides()`` works, and so does
    a plain mapping of ``LocalConfig`` field names -- which is what makes
    the seam testable without constructing a ladder.

    Unknown keys are DROPPED rather than raising: a provider that learns a
    new sampling field before this dataclass does must degrade to the
    fields we understand, not fail the generation.

    NEVER raises -- on any fault the caller gets the unmodified config,
    i.e. exactly the pre-sampling behaviour.
    """
    try:
        overrides = _sampling_overrides(sampling)
        if not overrides:
            return cfg
        return _dc_replace(cfg, **overrides)
    except Exception:  # noqa: BLE001 — a draw without its point still generates
        logger.debug("[LocalPrimeClient] sampling override ignored", exc_info=True)
        return cfg


_TRANSPORT_NATIVE = "native"
_TRANSPORT_OPENAI = "openai"
_TRANSPORTS = (_TRANSPORT_NATIVE, _TRANSPORT_OPENAI)


def _normalise_transport(raw: "Optional[str]") -> str:
    """The configured wire dialect, or native. Unknown -> native, logged once.

    Native is the default because it is the only route on which the
    sampler options are known to bite (see ``LocalConfig.transport``). A
    typo cannot silently select the leaky dialect: it selects the working one
    and says so."""
    value = (raw or "").strip().lower()
    if value in _TRANSPORTS:
        return value
    if value:
        logger.warning(
            "[LocalPrimeClient] JARVIS_LOCAL_TRANSPORT=%r is not one of %s; "
            "using %s", raw, _TRANSPORTS, _TRANSPORT_NATIVE,
        )
    return _TRANSPORT_NATIVE


def transport_for(cfg: "LocalConfig") -> str:
    return _normalise_transport(getattr(cfg, "transport", _TRANSPORT_NATIVE))


def is_native_transport(cfg: "LocalConfig") -> bool:
    return transport_for(cfg) == _TRANSPORT_NATIVE


def chat_endpoint(cfg: "LocalConfig") -> str:
    """The ONE place the chat URL is spelled."""
    base = str(getattr(cfg, "base_url", "") or "").rstrip("/")
    return base + ("/api/chat" if is_native_transport(cfg) else "/v1/chat/completions")


def _spell_for_transport(body: "Dict[str, Any]", cfg: "LocalConfig") -> "Dict[str, Any]":
    """Move the OpenAI-spelled scalars into ``options`` for the native route.

    The request is BUILT once, in OpenAI spelling, by ``complete()``. This is
    the single translation seam: on the native route ``temperature`` and
    ``max_tokens`` have no top-level meaning and live in ``options`` as
    ``temperature`` / ``num_predict``; ``top_p`` and ``seed`` already have
    both spellings written by ``_apply_sampling`` and the top-level copies
    are simply dropped. Everything else (``model``, ``messages``,
    ``keep_alive``, ``stream``, ``options``) is shared by both dialects.

    On the OpenAI route this returns ``body`` untouched, so that path stays
    byte-identical to before the native transport existed. NEVER raises."""
    try:
        if not is_native_transport(cfg):
            return body
        opts = body.setdefault("options", {})
        if not isinstance(opts, dict):
            opts = body["options"] = {}
        if "temperature" in body:
            opts["temperature"] = float(body.pop("temperature"))
        if "max_tokens" in body:
            opts["num_predict"] = int(body.pop("max_tokens"))
        for k in ("top_p", "seed"):
            body.pop(k, None)             # ``options`` already carries them
        body.pop("stream_options", None)  # OpenAI-only accounting request
        # The native route STREAMS BY DEFAULT. A non-streaming caller must say
        # so or the engine answers application/x-ndjson and a JSON read
        # fails (measured live, first native call). The streaming path sets
        # this to True afterwards, so setdefault is the right verb.
        body.setdefault("stream", False)
        return body
    except Exception:  # noqa: BLE001 — a spelling fault must not drop the request
        return body


async def _read_json(resp: Any) -> Any:
    """Read a JSON body from ANY response object this client is handed.

    aiohttp refuses to decode a body whose mimetype is not application/json
    unless told ``content_type=None`` -- and ollama's native route labels a
    non-streaming reply ``application/x-ndjson``. Test doubles and other
    sessions expose a ``json()`` without that keyword. One reader, both
    shapes: ask leniently first, fall back to the bare call. NEVER hides a
    decode error -- only the keyword mismatch is absorbed."""
    try:
        return await resp.json(content_type=None)
    except TypeError:
        return await resp.json()


def _extract_completion(data: "Dict[str, Any]") -> "Tuple[str, int, int]":
    """``(text, completion_tokens, prompt_tokens)`` from EITHER dialect's
    non-streaming reply. Dispatches on SHAPE, not on config, so a proxy that
    answers in the other dialect is still read correctly. Missing counts are
    0 (the caller labels the estimate). NEVER raises on a well-formed reply of
    either shape; a malformed one raises KeyError like the old path did."""
    if isinstance(data.get("message"), dict):              # ollama native
        text = str(data["message"].get("content") or "")
        return (text, int(data.get("eval_count") or 0),
                int(data.get("prompt_eval_count") or 0))
    text = data["choices"][0]["message"]["content"]         # OpenAI-compat
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return (text, int(usage.get("completion_tokens", 0) or 0),
            int(usage.get("prompt_tokens", 0) or 0))


def _parse_ndjson_delta(line: "bytes") -> "Any":
    """Parse ONE line of an ollama native ``/api/chat`` stream.

    Clean sibling of ``_parse_sse_delta`` with the SAME contract, so the read
    loop is dialect-blind: the incremental content string, ``_SSE_DONE`` on
    the terminal frame, an ``_SSEUsage`` when that frame carries counts, or
    None for blanks / parse errors. The native stream frames a JSON object
    per line -- ``{"message": {"content": ...}, "done": false}`` -- and its
    LAST frame carries ``done: true`` plus ``eval_count`` /
    ``prompt_eval_count``. That final frame may also carry a trailing content
    fragment, which is returned first through the same channel so nothing is
    lost. Pure + fail-soft."""
    try:
        s = line.decode("utf-8", "ignore").strip() if isinstance(line, (bytes, bytearray)) else str(line).strip()
        if not s or not s.startswith("{"):
            return None
        import json as _json  # noqa: PLC0415
        obj = _json.loads(s)
        if not isinstance(obj, dict):
            return None
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        content = msg.get("content") or None
        if obj.get("done"):
            if content:
                # A last fragment AND the end: hand back the fragment; the
                # loop's next readline() hits EOF, which is also "done".
                return content
            ct = int(obj.get("eval_count") or 0)
            if ct > 0:
                return _SSEUsage(
                    prompt_tokens=int(obj.get("prompt_eval_count") or 0),
                    completion_tokens=ct,
                )
            return _SSE_DONE
        return content
    except Exception:  # noqa: BLE001
        return None


def _parse_stream_line(line: "bytes") -> "Any":
    """Dialect-blind line parser: native NDJSON frames begin with ``{``,
    OpenAI SSE frames begin with ``data:``. One loop, two wire formats, no
    per-transport branch in the loop itself."""
    try:
        head = line.lstrip()[:1] if isinstance(line, (bytes, bytearray)) else str(line).lstrip()[:1]
    except Exception:  # noqa: BLE001
        return None
    if head in (b"{", "{"):
        return _parse_ndjson_delta(line)
    return _parse_sse_delta(line)


def _apply_response_format(body: "Dict[str, Any]", cfg: "LocalConfig") -> str:
    """Attach the strongest response constraint this engine is known to accept.

    Ladder, strongest first: json_schema -> json_object -> nothing. Each rung is
    a strict superset of what the next allows, so degrading can only ever admit
    outputs the parser was already prepared to reject -- never break a request
    that would have worked. Returns the mode applied, for telemetry. NEVER
    raises: a constraint is an optimisation over parsing, and failing to attach
    one must not fail the generation."""
    try:
        if not _json_mode_enabled():
            return "none"
        if _json_schema_mode_enabled() and _schema_key(cfg) not in _SCHEMA_UNSUPPORTED:
            schema = _resolve_response_schema()
            if schema:
                if is_native_transport(cfg):
                    # Native spelling: ``format`` carries the schema object
                    # itself. Same grammar, same ladder, same rejection cache.
                    body["format"] = schema
                else:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ov_generation",
                            # strict=False: the schema is a UNION and leaves
                            # additionalProperties open by design (the parser
                            # diagnoses unknown keys far better than a grammar can).
                            # Demanding strict here is what makes some backends
                            # reject an otherwise-valid schema.
                            "strict": False,
                            "schema": schema,
                        },
                    }
                return "json_schema"
        if is_native_transport(cfg):
            body["format"] = "json"
        else:
            body["response_format"] = {"type": "json_object"}
        return "json_object"
    except Exception:  # noqa: BLE001
        return "none"


#: "Use the ladder" -- the per-call default for ``response_format``. A
#: SENTINEL rather than ``None`` because ``None`` has to mean something
#: too: *no constraint at all*, which is what a prose completion (a gate's
#: one-sentence intent, a narrated reason) needs. Until this existed every
#: completion through this client was forced into the CANDIDATE grammar,
#: so a prose caller got a JSON object back -- the ladder was a property of
#: the client when it is a property of the CALL.
RESPONSE_FORMAT_LADDER: Any = object()


def _apply_explicit_response_format(body: "Dict[str, Any]", cfg: "LocalConfig",
                                    fmt: "Dict[str, Any]") -> str:
    """Attach a caller-supplied OpenAI-shaped ``response_format``, spelled
    for the transport in use. ``{"type": "json_object"}`` is the only shape
    a gate asks for today; a ``json_schema`` shape carries its schema
    through. Returns the mode applied, for telemetry. NEVER raises."""
    try:
        kind = str(fmt.get("type", "") or "") if isinstance(fmt, dict) else ""
        if not kind:
            return "none"
        if is_native_transport(cfg):
            schema = None
            if kind == "json_schema":
                schema = (fmt.get("json_schema") or {}).get("schema")
            body["format"] = schema or "json"
        else:
            body["response_format"] = dict(fmt)
        return kind
    except Exception:  # noqa: BLE001
        return "none"


#: (base_url, model) pairs that have REJECTED a ``reasoning_effort`` field.
#: Kept separate from ``_SCHEMA_UNSUPPORTED`` deliberately: attributing a
#: reasoning_effort rejection to the schema would permanently disable
#: constrained decoding for the process over an unrelated field.
_REASONING_UNSUPPORTED: "set" = set()

_VALID_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh"}
)


def _reasoning_effort() -> str:
    """How much the engine should think before answering.

    Default ``none``. Measured on this host against qwen3.8:27b over
    ``/v1/chat/completions`` with the json_schema response_format attached:

        baseline (thinking on) : valid JSON, 629 chars reasoning, 6.8s
        reasoning_effort=none  : valid JSON,   0 chars reasoning, 1.5s
        reasoning_effort=low   : valid JSON, 531 chars reasoning, 2.3s

    Correctness is NOT the issue on this path -- ollama returns reasoning in
    a separate ``reasoning`` field, so the constrained ``content`` stays
    schema-valid either way. Cost is: reasoning is a 4.5x wall-clock tax on
    a lane whose soaks are wall-clock capped, and O+V judges a candidate by
    whether it parses and passes tests, not by the prose that preceded it.

    Set to empty to omit the field entirely (byte-identical to pre-flag
    behaviour); set higher to buy reasoning back if it proves to raise
    candidate quality.

    NOTE the two spellings that do NOT work here and must not be reached
    for: ollama's native ``think`` field and ``chat_template_kwargs
    {enable_thinking: false}`` are both silently IGNORED on
    /v1/chat/completions (measured: reasoning stayed at 629 chars for
    both). They belong to /api/chat.
    """
    raw = os.environ.get("JARVIS_LOCAL_REASONING_EFFORT", "none").strip().lower()
    return raw if raw in _VALID_REASONING_EFFORTS else ""


def _apply_reasoning_effort(body: "Dict[str, Any]", cfg: "LocalConfig") -> str:
    """Attach ``reasoning_effort`` unless this engine has rejected it.

    Returns the effort applied, for telemetry. NEVER raises: thinking
    budget is an optimisation, and failing to set one must not fail the
    generation."""
    try:
        effort = _reasoning_effort()
        if not effort or _schema_key(cfg) in _REASONING_UNSUPPORTED:
            return ""
        if is_native_transport(cfg):
            # Native spelling. The docstring on ``_reasoning_effort`` records
            # that ``think`` is silently IGNORED on /v1 -- it belongs to
            # /api/chat, which is now where this request goes. ``none``
            # maps to ``think: false``; any positive effort leaves thinking
            # at the engine's default, since the native field is boolean.
            body["think"] = effort != "none"
        else:
            body["reasoning_effort"] = effort
        return effort
    except Exception:  # noqa: BLE001
        return ""


def _degrade_reasoning_effort(body: "Dict[str, Any]", cfg: "LocalConfig") -> bool:
    """Record that this engine rejected ``reasoning_effort`` and drop it.

    Returns True when a retry is worth attempting. Idempotent, so a
    persistently-400ing endpoint cannot become a retry loop. NEVER raises."""
    try:
        key = _schema_key(cfg)
        if key in _REASONING_UNSUPPORTED:
            return False
        _REASONING_UNSUPPORTED.add(key)
        popped_openai = body.pop("reasoning_effort", None) is not None
        popped_native = body.pop("think", None) is not None
        return popped_openai or popped_native
    except Exception:  # noqa: BLE001
        return False


def _degrade_response_format(body: "Dict[str, Any]", cfg: "LocalConfig") -> bool:
    """Record that this engine rejected json_schema and downgrade the body.

    Returns True when a retry is worth attempting. Idempotent: a second call for
    the same engine returns False, so a persistently-400ing endpoint cannot
    become a retry loop. NEVER raises."""
    try:
        key = _schema_key(cfg)
        if key in _SCHEMA_UNSUPPORTED:
            return False
        _SCHEMA_UNSUPPORTED.add(key)
        if is_native_transport(cfg):
            body["format"] = "json"
        else:
            body["response_format"] = {"type": "json_object"}
        logger.warning(
            "[LocalPrimeClient] engine rejected json_schema response_format for "
            "model=%s at %s — degrading to json_object for the rest of this "
            "process. Shape errors become parser-diagnosed rather than "
            "sampler-prevented.",
            cfg.model_name, cfg.base_url,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def _json_mode_enabled() -> bool:
    """Whether to request constrained JSON decoding on local completions.

    Default ON: every schema this client speaks is JSON, so the constraint can
    only remove outputs that were going to be rejected anyway. Set
    ``JARVIS_LOCAL_JSON_MODE_ENABLED=0`` to restore free-form sampling -- worth
    doing only if a future schema stops being JSON, or to reproduce a parse
    failure that the constraint would otherwise mask."""
    return _envb("JARVIS_LOCAL_JSON_MODE_ENABLED", True)


@dataclass(frozen=True)
class LocalConfig:
    base_url: str
    model_name: str
    keep_alive_seconds: int
    timeout_seed_ms: int       # cold-start seed
    timeout_ceiling_ms: int    # absolute hard cap (adaptive never exceeds)
    timeout_floor_ms: int
    output_ratio: float        # est_output_tokens = prompt_tokens * ratio
    margin_sigma: float
    window_size: int
    min_samples: int
    max_concurrency: int
    pool_limit: int
    # Autonomous Context-Hardware Negotiator output: the VRAM-safe context window
    # injected as ollama ``options.num_ctx`` + used as the Cognitive Compression
    # budget. None -> legacy (no injection, no compression) = byte-identical.
    num_ctx: Optional[int] = None
    # Sibling entropy (see governance/sibling_entropy.py). All None -> the
    # engine's own defaults, i.e. byte-identical legacy sampling. Set per
    # DRAW by the candidate generator so siblings explore different regions
    # of sampling space instead of re-deriving one answer at temperature
    # 0.2. Threaded through ``dataclasses.replace`` like every other
    # override, so there is still exactly one request builder.
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    #: Which wire protocol the engine is spoken to over. ``native`` is
    #: ollama's own ``/api/chat``; ``openai`` is ``/v1/chat/completions``.
    #: Native is the default because the OpenAI-compatible layer DROPS the
    #: ``options`` block -- measured on ollama 0.33.2 with curl: the same
    #: ``seed`` produced two different completions and ``options.top_k=1``
    #: (greedy) still varied, while the native route reproduced a seed
    #: byte-for-byte and was greedy-identical twice. Every sampler field the
    #: entropy ladder sets (top_k, repeat_penalty, seed) rides ``options``,
    #: so on ``/v1`` the ladder was reaching the wire and not the sampler.
    #: ``openai`` stays selectable for engines that only speak that dialect.
    transport: str = "native"

    @classmethod
    def from_env(cls) -> "LocalConfig":
        def _i(n: str, d: int) -> int: return int(os.environ.get(n, str(d)))
        def _f(n: str, d: float) -> float: return float(os.environ.get(n, str(d)))
        ceiling = _i("JARVIS_LOCAL_INFERENCE_TIMEOUT_MS", 120_000)
        _nc = os.environ.get("JARVIS_LOCAL_NUM_CTX", "").strip()
        return cls(
            base_url=os.environ.get("JARVIS_LOCAL_MODEL_BASE_URL", "http://127.0.0.1:11434"),
            model_name=os.environ.get("JARVIS_LOCAL_MODEL_NAME", "qwen2.5-coder:3b"),
            keep_alive_seconds=_i("JARVIS_LOCAL_MODEL_KEEP_ALIVE_SECONDS", 300),
            timeout_seed_ms=_i("JARVIS_LOCAL_INFERENCE_TIMEOUT_SEED_MS", 30_000),
            timeout_ceiling_ms=ceiling,
            timeout_floor_ms=_i("JARVIS_LOCAL_INFERENCE_TIMEOUT_FLOOR_MS", 4_000),
            output_ratio=_f("JARVIS_LOCAL_OUTPUT_RATIO", 0.5),
            margin_sigma=_f("JARVIS_LOCAL_MARGIN_SIGMA", 2.0),
            window_size=_i("JARVIS_LOCAL_PROFILER_WINDOW", 20),
            min_samples=_i("JARVIS_LOCAL_PROFILER_MIN_SAMPLES", 5),
            max_concurrency=_i("JARVIS_LOCAL_MODEL_MAX_CONCURRENCY", 2),
            pool_limit=_i("JARVIS_LOCAL_POOL_LIMIT", 8),
            num_ctx=int(_nc) if _nc.isdigit() else None,
            transport=_normalise_transport(
                os.environ.get("JARVIS_LOCAL_TRANSPORT", "")),
        )


# ---------------------------------------------------------------------------
# Autonomous Context-Hardware Negotiator + Dynamic Cognitive Compression
# ---------------------------------------------------------------------------
#
# The warm 32B ServerDisconnects on an L4 because the KV cache for a large prompt
# overflows the VRAM left after the ~20GB model weights. We solve this in software:
# derive the max SAFE num_ctx from the MEASURED VRAM buffer (no static cap), then
# compress the payload to fit it (preserve the system rules + recent tool outputs).

# Accurate KV cache per token for a 32B GQA fp16 model: 2(K+V) * 64 layers *
# (8 kv_heads * 128 head_dim = 1024 kv_dim) * 2 bytes = 262144 = 256KB. (The prior
# 512KB was 2x too conservative -- it double-counted the kv_dim -- which crushed
# num_ctx and over-compressed the payload into empty responses.) Env-tunable.
_KV_BYTES_PER_TOKEN_DEFAULT = 262144
_CTX_OVERHEAD_BYTES_DEFAULT = 1_500_000_000  # CUDA/runtime/activation headroom
_NUM_CTX_FLOOR_DEFAULT = 2048
_NUM_CTX_CEILING_DEFAULT = 32768
# Tokens reserved for the model's OUTPUT inside num_ctx. The generation cap
# (max_tokens) is often 4096, but a patch is ~1-2K tokens; reserving the full cap
# halved the INPUT budget and over-compressed. Reserve a bounded, env-tunable
# output slice so the input window stays wide. Default 2048.
_OUTPUT_RESERVE_TOKENS_DEFAULT = 2048


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _f_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def estimate_tokens(text: Any) -> int:
    """Cheap deterministic token estimate (~4 chars/token). NEVER raises."""
    try:
        return max(0, len(text or "") // 4)
    except Exception:  # noqa: BLE001
        return 0


def expected_agentic_cycle_s(profiler: "Optional[LatencyProfiler]" = None,
                             *, num_ctx: "Optional[int]" = None) -> float:
    """THE shared time-physics formula: expected wall for ONE full multi-round
    agentic cycle = rounds x expected-single-round wall.

    EWMA-COUPLED when a calibrated ``LatencyProfiler`` is supplied
    (``adaptive_timeout_ms`` -- the GPU speeding up shrinks the cycle
    automatically); otherwise the cold-seed physics the profiler itself uses
    (seed x heavy-mult x ctx-factor). ``num_ctx`` defaults to the configured
    window, else the arm-time expectation ``JARVIS_HYBRID_MESH_EXPECTED_
    NUM_CTX`` (16384). Consumed by the BudgetPlan hint, the Time-Dilated
    Sovereign Deadline, the sovereign GENERATE physics floor, and the
    driver's arm-time walls -- ONE model, no magic numbers. NEVER raises."""
    rounds = 5
    try:
        rounds = max(1, int(float(os.environ.get(
            "JARVIS_A1_MAX_AGENTIC_ROUNDS", "5") or 5)))
        if profiler is not None:
            try:
                est_ms = float(profiler.adaptive_timeout_ms(
                    prompt_tokens=max(1, int(num_ctx or 4096) // 4)))
                return rounds * est_ms / 1000.0
            except Exception:  # noqa: BLE001 -- fall back to cold physics
                pass
        cfg = LocalConfig.from_env()
        seed_s = float(getattr(cfg, "timeout_seed_ms", 30_000) or 30_000) / 1000.0
        mult = max(1.0, float(os.environ.get(
            "JARVIS_JPRIME_HEAVY_COLDSTART_MULT", "4.0") or 4.0))
        baseline = max(1.0, float(os.environ.get(
            "JARVIS_LOCAL_SEED_CTX_BASELINE", "8192") or 8192))
        ctx = float(num_ctx or getattr(cfg, "num_ctx", 0) or 0)
        if ctx <= 0:
            ctx = float(os.environ.get(
                "JARVIS_HYBRID_MESH_EXPECTED_NUM_CTX", "16384") or 16384)
        per_round_s = seed_s * mult * max(1.0, ctx / baseline)
        # The Amnesia Cure: fold the cross-run physics ledger into the cold
        # path -- run N+1's very first floors/walls are sized from run N's
        # MEASURED per-round truth (the max persisted EWMA), never below the
        # seed formula. Fail-soft: empty/corrupt ledger -> formula only.
        try:
            if _physics_ledger_enabled():
                _persisted = [
                    float((v or {}).get("ewma_ms", 0.0) or 0.0) / 1000.0
                    for v in _physics_ledger_load().values()
                ]
                if _persisted:
                    per_round_s = max(per_round_s, max(_persisted))
        except Exception:  # noqa: BLE001
            pass
        return rounds * per_round_s
    except Exception:  # noqa: BLE001 -- physics sizing must never break a caller
        return rounds * 120.0


def derive_safe_num_ctx(
    *,
    vram_bytes: int,
    model_bytes: int,
    kv_bytes_per_token: "Optional[int]" = None,
    overhead_bytes: "Optional[int]" = None,
    floor: "Optional[int]" = None,
    ceiling: "Optional[int]" = None,
) -> int:
    """Mathematically derive the max SAFE context window from the MEASURED VRAM
    buffer: ``(vram - model - overhead) / kv_bytes_per_token``, floored to a 256
    multiple and clamped to [floor, ceiling]. NOT a static cap -- a bigger GPU or a
    smaller model widens the window automatically. Fail-soft -> floor on any
    non-positive buffer / bad input."""
    kvbpt = kv_bytes_per_token if kv_bytes_per_token is not None else _int_env(
        "JARVIS_KV_BYTES_PER_TOKEN", _KV_BYTES_PER_TOKEN_DEFAULT)
    ovh = overhead_bytes if overhead_bytes is not None else _int_env(
        "JARVIS_CTX_OVERHEAD_BYTES", _CTX_OVERHEAD_BYTES_DEFAULT)
    flr = floor if floor is not None else _int_env("JARVIS_NUM_CTX_FLOOR", _NUM_CTX_FLOOR_DEFAULT)
    ceil = ceiling if ceiling is not None else _int_env("JARVIS_NUM_CTX_CEILING", _NUM_CTX_CEILING_DEFAULT)
    try:
        if vram_bytes <= 0 or model_bytes <= 0 or kvbpt <= 0:
            return flr
        kv_buffer = int(vram_bytes) - int(model_bytes) - int(ovh)
        if kv_buffer <= 0:
            return flr
        nctx = (int(kv_buffer // kvbpt) // 256) * 256  # 256-multiple for engine friendliness
        return max(flr, min(ceil, nctx))
    except Exception:  # noqa: BLE001
        return flr


def fit_prompt_to_window(
    system: str,
    user: str,
    *,
    max_tokens: int,
    head_frac: float = 0.35,
    tail_frac: float = 0.5,
) -> "Tuple[str, str, bool]":
    """Dynamic Cognitive Compression (sliding window). Preserve the SYSTEM prompt
    (Iron Gate rules) IN FULL + a HEAD (task/plan) and TAIL (most recent tool
    outputs) of the user payload; compress the older intermediate middle into a
    deterministic marker. GUARANTEES ``estimate_tokens(system)+estimate_tokens(user)
    <= max_tokens`` (best-effort; the system is never cut, so if it alone exceeds
    the window the user is reduced to a stub). Returns (system, user, compressed?)."""
    system = system or ""
    user = user or ""
    if estimate_tokens(system) + estimate_tokens(user) <= max_tokens:
        return system, user, False
    user_budget_toks = max_tokens - estimate_tokens(system)
    if user_budget_toks <= 0:
        return system, "[context omitted: system prompt already fills the VRAM-safe window]", True
    char_budget = user_budget_toks * 4
    head_chars = max(0, int(char_budget * head_frac))
    tail_chars = max(0, int(char_budget * tail_frac))
    if len(user) <= head_chars + tail_chars or head_chars + tail_chars == 0:
        return system, user[:char_budget], True
    head = user[:head_chars]
    tail = user[-tail_chars:]
    dropped = len(user) - head_chars - tail_chars
    marker = (
        "\n\n[...cognitive compression: %d chars of older intermediate history "
        "elided to fit the %d-token VRAM-safe window; system rules + recent tool "
        "outputs preserved...]\n\n" % (dropped, max_tokens)
    )
    return system, head + marker + tail, True


def _physics_ledger_enabled() -> bool:
    return (os.environ.get("JARVIS_LATENCY_LEDGER_ENABLED", "true") or "").strip().lower() \
        not in ("0", "false", "no", "off")


def _physics_ledger_path() -> str:
    return os.environ.get(
        "JARVIS_LATENCY_LEDGER_PATH",
        os.path.join(".jarvis", "latency_physics.json"),
    )


def _physics_ledger_load() -> dict:
    """The Amnesia Cure ledger (bandit_router idiom): keyed physics dict at
    .jarvis/latency_physics.json. Fail-soft: corrupt/missing -> {}."""
    try:
        with open(_physics_ledger_path(), encoding="utf-8") as fh:
            data = json.loads(fh.read())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _physics_ledger_save(key: str, payload: dict) -> None:
    """Write-through one key (few hundred bytes). NEVER raises into dispatch."""
    try:
        path = _physics_ledger_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = _physics_ledger_load()
        data[str(key)] = payload
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def physics_key(cfg: "LocalConfig", *, endpoint: str = "") -> str:
    """The DURABLE physics identity: (hardware, model, ctx-bucket).

    NOT the endpoint -- node IPs change every run, and the physics belongs to
    the brain+window. That original reasoning is preserved and is why the
    address never appears here.

    But it was incomplete. (model, ctx) alone CONFLATES MACHINES: the same
    model name on a 16GB M1 and on an RTX 5090 wrote one ledger key, so
    whichever measured last governed both -- and once a Mac dispatches to a
    Windows inference host, the Mac's ~95ms/token would size the 5090's lane
    count. The ThroughputGovernor would then compute a wrong answer from
    perfectly honest data, which is the worst shape a defect can take.

    So the identity gains a hardware axis: a hashed, machine-scoped signature
    of the host SERVING this config (see `hardware_signature`), never of the
    host timing it. Two machines now keep strictly separate ledgers for the
    same model.

    MIGRATION: this changes the key shape, so entries written under the old
    shape become unreachable and each (hardware, model, ctx) triple
    cold-starts once. That is the intended direction -- the unreachable
    entries are exactly the conflated measurements this change exists to stop
    trusting. `JARVIS_HARDWARE_SIGNATURE_ENABLED=0` restores the old shape
    byte-for-byte.
    """
    try:
        model = str(getattr(cfg, "model_name", "") or "unknown")
        ctx = int(getattr(cfg, "num_ctx", 0) or 0)
        base = "%s@%s" % (model, ctx if ctx > 0 else "cpu")
    except Exception:  # noqa: BLE001
        return "unknown@cpu"
    try:
        from backend.core.ouroboros.governance.hardware_signature import (
            signature_for,
        )
        # *endpoint* wins over cfg.base_url when supplied. The failover path
        # holds one profiler per ENDPOINT while carrying a single cfg, so
        # resolving the signature from cfg there would stamp every failover
        # target with the base config's identity -- re-creating the exact
        # host conflation this axis exists to remove, at the one seam
        # (cross-node failover) where it matters most.
        target = str(endpoint or getattr(cfg, "base_url", "") or "")
        digest = signature_for(target).digest
        # TRANSPORT is a THIRD axis, distinct from hardware and from model.
        #
        # The same 5090 reached over a Tailscale direct link and over a DERP
        # relay is the same silicon on two different networks: identical
        # hardware signature, wildly different observed latency. Blending them
        # into one entry would have the ThroughputGovernor size lanes for a
        # hotel-wifi session from home-LAN measurements, and vice versa — the
        # exact cross-contamination the hardware axis exists to prevent, one
        # level down the stack.
        #
        # Empty when unmeasured: an unknown transport must not become a bucket
        # that later real measurements are mixed into.
        try:
            from backend.core.ouroboros.governance.transport_profile import (
                transport_class_for,
            )
            _tclass = transport_class_for(target)
        except Exception:  # noqa: BLE001
            _tclass = ""
        if digest and _tclass:
            return "%s@%s@%s" % (digest, _tclass, base)
        # An empty digest means the signature layer is off or declined to
        # guess. Fall back to the legacy shape rather than inventing a
        # placeholder segment -- a key containing "unknown" would silently
        # merge every unidentifiable host into one bucket, which is the very
        # conflation being removed.
        return "%s@%s" % (digest, base) if digest else base
    except Exception:  # noqa: BLE001 — identity must never break dispatch
        return base


class LatencyProfiler:
    """Thread-safe sliding window of (ttft_ms, per_token_ms) -> bounded adaptive timeout.

    Cold start uses the seed; the adaptive value is always clamped to
    [floor, ceiling]. The ceiling is the un-flexible hard cap that guarantees a
    wedged model still trips the breaker (watchdog-isolation invariant).
    """

    def __init__(self, cfg: "LocalConfig", ledger_key: str = "") -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._ttft: Deque[float] = deque(maxlen=cfg.window_size)
        self._per_tok: Deque[float] = deque(maxlen=cfg.window_size)
        self._total: Deque[float] = deque(maxlen=cfg.window_size)
        # The Amnesia Cure: warm-start from the cross-run physics ledger so a
        # fresh process inherits the MEASURED truth instead of the blind seed.
        self._ledger_key = str(ledger_key or "") if _physics_ledger_enabled() else ""
        if self._ledger_key:
            try:
                _prior = _physics_ledger_load().get(self._ledger_key) or {}
                for _v in (_prior.get("ttft") or [])[-cfg.window_size:]:
                    self._ttft.append(float(_v))
                for _v in (_prior.get("per_tok") or [])[-cfg.window_size:]:
                    self._per_tok.append(float(_v))
                for _v in (_prior.get("total") or [])[-cfg.window_size:]:
                    self._total.append(float(_v))
                self._ewma_prior = float(_prior.get("ewma_ms", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                self._ewma_prior = 0.0
        else:
            self._ewma_prior = 0.0
        # Asymmetric EWMA (ms): timeouts jump it UP (penalty), successes blend it
        # down toward the real latency. Acts as an escalating floor on the adaptive
        # timeout so a starved cold profiler still expands the window. 0 = no data.
        self._ewma_ms: float = float(getattr(self, '_ewma_prior', 0.0) or 0.0)
        # Async Calibration Mutex (Scout Lock): the FIRST cold coroutine calibrates
        # (Scout) while the concurrent herd waits on the lock; once calibrated, the
        # gate is open and dispatches run fully concurrently on the escalated EWMA.
        # Lazily bound to the running loop on first use.
        self._calibrated: bool = False
        self._scout_lock: "Optional[asyncio.Lock]" = None

    def is_calibrated(self) -> bool:
        return self._calibrated

    def mark_calibrated(self) -> None:
        self._calibrated = True

    def _get_scout_lock(self) -> "asyncio.Lock":
        if self._scout_lock is None:
            self._scout_lock = asyncio.Lock()
        return self._scout_lock

    async def run_calibrated(self, coro_factory: "Any") -> Any:
        """Async Calibration Mutex. If already calibrated -> run ``coro_factory()``
        immediately (no lock, full concurrency). Otherwise the FIRST caller acquires
        the scout lock and runs as the Scout; the concurrent herd awaits the lock.
        The Scout marks the profiler calibrated when it finishes (success OR
        timeout+escalate -- in ``finally``, so the herd is never stuck) and releases;
        the herd then runs CONCURRENTLY, reading the newly escalated EWMA seed.
        ``coro_factory`` is a zero-arg async callable (a fresh coroutine per call)."""
        if self.is_calibrated():
            return await coro_factory()
        lock = self._get_scout_lock()
        async with lock:
            if not self.is_calibrated():
                try:
                    return await coro_factory()          # Scout
                finally:
                    self.mark_calibrated()
        return await coro_factory()                       # herd (calibrated)

    def _cold_seed_ms(self) -> float:
        """Context-Aware Dynamic Seed. Survival/CPU (no num_ctx) -> plain base seed
        (byte-identical legacy). Heavy/GPU (negotiated num_ctx) -> the base seed
        scaled by JARVIS_JPRIME_HEAVY_COLDSTART_MULT AND the token payload
        (num_ctx / baseline) -- a 16k window inherently needs a longer first budget
        than 8k. Capped at half the absolute ceiling so escalation has room before
        the breaker. NEVER raises."""
        base = float(self._cfg.timeout_seed_ms)
        if not self._cfg.num_ctx:
            return base
        try:
            heavy_mult = max(1.0, _f_env("JARVIS_JPRIME_HEAVY_COLDSTART_MULT", 4.0))
            baseline = max(1, _int_env("JARVIS_LOCAL_SEED_CTX_BASELINE", 8192))
            ctx_factor = max(1.0, float(self._cfg.num_ctx) / baseline)
            seed = base * heavy_mult * ctx_factor
            return min(seed, _absolute_ceiling_ms() * 0.5)
        except Exception:  # noqa: BLE001
            return base

    def record(self, *, ttft_ms: float, total_ms: float, output_tokens: int) -> None:
        per_tok = (total_ms - ttft_ms) / max(1, output_tokens)
        with self._lock:
            self._ttft.append(float(ttft_ms))
            self._per_tok.append(max(0.0, per_tok))
            self._total.append(float(total_ms))
            # SUCCESS blends the EWMA DOWN toward the observed latency (asymmetric).
            if self._ewma_ms <= 0.0:
                self._ewma_ms = float(total_ms)
            else:
                a = _ewma_alpha()
                self._ewma_ms = a * float(total_ms) + (1.0 - a) * self._ewma_ms
        self._flush_physics()

    def record_timeout_penalty(self, timeout_ms: float) -> None:
        """Asymmetric penalty injection: a TIMEOUT jumps the EWMA UP to
        ``timeout_ms * escalation_factor`` so the very next dispatch expands the
        window aggressively (breaks the cold-profiler starvation). NEVER raises."""
        try:
            penalty = float(timeout_ms) * _timeout_escalation_factor()
            with self._lock:
                self._ewma_ms = max(self._ewma_ms, penalty)
            self._flush_physics()
        except Exception:  # noqa: BLE001
            pass

    def _flush_physics(self) -> None:
        """Write-through the durable physics (Amnesia Cure). NEVER raises."""
        if not self._ledger_key:
            return
        try:
            with self._lock:
                payload = {
                    "ewma_ms": float(self._ewma_ms),
                    "ttft": list(self._ttft),
                    "per_tok": list(self._per_tok),
                    "total": list(self._total),
                }
            _physics_ledger_save(self._ledger_key, payload)
        except Exception:  # noqa: BLE001
            pass

    def inter_token_budget_s(
        self, *,
        prompt_tokens: "Optional[int]" = None,
        temperature: "Optional[float]" = None,
        sampling: "Optional[Any]" = None,
    ) -> "Tuple[float, float]":
        """``(first_token_s, steady_token_s)`` for the streaming watchdog.

        TWO deadlines, because the two waits measure different physics and a
        single constant is wrong for both:

        * **first token** must cover PREFILL. A 64K-context prompt can
          legitimately take tens of seconds before its first byte, and the
          static 30s was applied to this wait too -- so a large prompt on a
          slow host could be declared "wedged" while it was working normally.
          This deadline is therefore derived from measured TTFT and may only
          ever be LOOSER than the legacy value, never tighter.
        * **steady state** is the gap BETWEEN chunks once generation is
          flowing. Here the static 30s is wildly loose: at 220 tok/s the
          normal gap is ~4.5ms, so a wedged peer would be given ~6,600 normal
          gaps before anyone noticed. This deadline is derived from measured
          per-token cost and may only ever be TIGHTER than the legacy value.

        ## The draw's own physics (2026-09-02)

        ``prompt_tokens``, ``temperature`` and ``sampling`` describe THIS
        draw. They exist because soak bt-2026-09-02-203607 died with 153
        streams armed at exactly ``30s`` and a profiler that never warmed:

        * **Prefill grows with the prompt**, and a cold profiler has no TTFT
          sample to say so. The first-token deadline is therefore scaled by
          the prompt's size against the same context baseline
          ``_cold_seed_ms`` already uses -- a 32K prompt on a 30B model
          legitimately needs longer than an 8K one before its first byte,
          warm or cold.
        * **A stall was already being recorded and never consulted.** The
          streaming path calls ``record_timeout_penalty`` on every wedge,
          which escalates ``_ewma_ms`` -- and this method never read it. So
          the very seam that exists to "break the cold-profiler starvation"
          had no effect on the deadline that was starving it. The first-token
          deadline now honours the escalated EWMA.
        * **Entropy widens the steady-state tolerance.** A wide ``top_k`` and
          a ``repeat_penalty`` applied across the whole ``repeat_last_n``
          window raise per-token sampler cost and its VARIANCE, so twelve
          "normal" gaps at temperature 0.2 is the wrong yardstick for a draw
          at T=1.10/top_k=140. The steady deadline's ceiling is scaled by the
          same ``entropy_latency_factor`` the total budget uses -- one
          definition of "how much wider is this draw".

        Every scale is >= 1.0 and every knob is env-derived; a draw with no
        description gets the pre-existing budget byte-for-byte. The absolute
        ceiling still bounds the first-token wait so a wedged model cannot
        buy unlimited time by being asked a long question.

        Cold profiler + no draw description -> the static value for both,
        i.e. byte-identical legacy behaviour. NEVER raises.
        """
        static = _inter_token_timeout_s()
        try:
            if not _inter_token_adaptive_enabled():
                return (static, static)
            with self._lock:
                warm = len(self._total) >= self._cfg.min_samples
                ttft_m = self._mean(self._ttft)
                ttft_sd = self._stddev(self._ttft)
                tok_m = self._mean(self._per_tok)
                tok_sd = self._stddev(self._per_tok)
                ewma_ms = self._ewma_ms

            ctx_scale = _prefill_context_scale(prompt_tokens)
            entropy = entropy_latency_factor(temperature, sampling)
            absolute_s = _absolute_ceiling_ms() / 1000.0
            sigma = float(getattr(self._cfg, "margin_sigma", 2.0) or 2.0)
            mult = _inter_token_stall_multiple()

            # First token: never below the static value scaled by the prompt
            # (prefill physics), and never below the timeout-escalated EWMA
            # (a stall the ledger already paid for). Warm adds the measured
            # TTFT + margin. Bounded by the absolute ceiling.
            first = static * ctx_scale
            if ewma_ms > 0.0:
                first = max(first, ewma_ms / 1000.0)
            if warm:
                first = max(first, mult * (ttft_m + sigma * ttft_sd) / 1000.0)
            first = min(first, absolute_s) if absolute_s > 0 else first

            if not warm:
                # Steady state has nothing measured to tighten against; the
                # entropy premium is all this draw knows about itself.
                return (first, static * entropy)

            # Steady state: measured per-token + margin, scaled, floored so
            # jitter is not mistaken for a wedge, and capped at the
            # entropy-scaled static value so this path can only ever DETECT
            # SOONER than that ceiling, never later.
            steady = mult * (tok_m + sigma * tok_sd) / 1000.0
            steady = max(_inter_token_floor_s(), min(steady, static * entropy))
            return (first, steady)
        except Exception:  # noqa: BLE001 — a watchdog must never fail to arm
            return (static, static)

    def is_warm(self) -> bool:
        with self._lock:
            return len(self._total) >= self._cfg.min_samples

    @staticmethod
    def _mean(xs: Deque[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    @classmethod
    def _stddev(cls, xs: Deque[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = cls._mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    def adaptive_timeout_ms(
        self, *, prompt_tokens: int,
        temperature: "Optional[float]" = None,
        sampling: "Optional[Any]" = None,
    ) -> float:
        """Budget for ONE draw, priced from measured latency AND its entropy.

        ``temperature``/``sampling`` describe this draw's point in sampling
        space. They widen the expected OUTPUT (see
        ``entropy_latency_factor``) rather than the timeout as a whole, so
        the profiler's own statistics keep their meaning: the sigma margin
        still covers variance, the EWMA still escalates on real timeouts,
        and the absolute breaker still owns the wedged-model case.

        Both default to None, which yields factor 1.0 — every caller that
        does not describe a sampling point gets a byte-identical budget.
        """
        cfg = self._cfg
        entropy = entropy_latency_factor(temperature, sampling)
        with self._lock:
            warm = len(self._total) >= cfg.min_samples
            ttft_m = self._mean(self._ttft)
            tok_m = self._mean(self._per_tok)
            tot_sd = self._stddev(self._total)
            ewma = self._ewma_ms

        # SURVIVAL / CPU path (no negotiated num_ctx): BYTE-IDENTICAL legacy at
        # entropy 1.0 -- no EWMA escalation, no absolute breaker, soft ceiling
        # is the cap.
        if not cfg.num_ctx:
            if not warm:
                seed = cfg.timeout_seed_ms * entropy
                return float(min(seed, cfg.timeout_ceiling_ms))
            est_out = max(1.0, prompt_tokens * cfg.output_ratio * entropy)
            flexed = ttft_m + tok_m * est_out + cfg.margin_sigma * tot_sd
            return float(max(cfg.timeout_floor_ms, min(flexed, cfg.timeout_ceiling_ms)))

        # HEAVY / GPU path: Context-Aware Dynamic Seed + asymmetric EWMA escalation
        # + Absolute Global Circuit Breaker.
        absolute = _absolute_ceiling_ms()
        if warm:
            est_out = max(1.0, prompt_tokens * cfg.output_ratio * entropy)
            value = ttft_m + tok_m * est_out + cfg.margin_sigma * tot_sd
        else:
            # A cold profiler has no per-token measurement to widen, so the
            # entropy premium rides the seed itself -- the first sibling of a
            # cold op is exactly the draw that must not be cut off.
            value = self._cold_seed_ms() * entropy
        # Never below the (timeout-escalated) EWMA -- a starved cold profiler still
        # expands the window on the next dispatch.
        if ewma > 0.0:
            value = max(value, ewma)
        # Runaway EWMA past the absolute ceiling kills the loop (no infinite
        # inflation / endless billing on a genuinely wedged model).
        if value >= absolute:
            raise UnrecoverableInferenceLatency(
                "adaptive inference timeout %.0fms >= absolute ceiling %.0fms "
                "(EWMA=%.0fms) -- wedged model, halting to prevent endless billing"
                % (value, absolute, ewma)
            )
        # The absolute ceiling is the cap so the dynamic seed / escalation is not
        # crushed by the (survival-sized) soft ceiling.
        return float(max(cfg.timeout_floor_ms, min(absolute, value)))

    def is_terminal_lag(self, *, elapsed_ms: float) -> bool:
        cfg = self._cfg
        if elapsed_ms > cfg.timeout_ceiling_ms:
            return True
        with self._lock:
            warm = len(self._total) >= cfg.min_samples
            if not warm:
                return False
            m = self._mean(self._total)
            sd = self._stddev(self._total)
        # Use a minimum stddev floor of 10% of mean so that a perfectly uniform
        # sample distribution still produces a meaningful 3-sigma band.
        sd_eff = max(sd, m * 0.1)
        return elapsed_ms > (m + 3.0 * sd_eff)


class LocalLatencyLockup(RuntimeError):
    """Raised when local inference breaches the adaptive/ceiling timeout.

    Consumed by candidate_generator's FailbackStateMachine to transition
    J-Prime to PRIMARY_DEGRADED and cascade the op upstream.
    """
    failure_class = "terminal_lag_lockup"


class UnrecoverableInferenceLatency(RuntimeError):
    """Absolute Global Circuit Breaker: the adaptive/EWMA timeout inflated past the
    absolute ceiling (default 20min). Raised to KILL the loop -- prevents infinite
    EWMA inflation + endless billing on a genuinely wedged model. Non-recoverable:
    the L7 auto-heal treats it as terminal (seal/halt), never retries."""
    failure_class = "unrecoverable_inference_latency"


def _absolute_ceiling_ms() -> float:
    """Hard absolute inference-timeout ceiling (ms). The EWMA escalation can grow
    the budget on timeouts; this is the un-inflatable kill line. Default 20min."""
    return max(1000.0, _int_env("JARVIS_LOCAL_INFERENCE_ABSOLUTE_CEILING_MS", 1_200_000))


def _timeout_escalation_factor() -> float:
    """Asymmetric penalty multiplier: a timeout injects timeout*factor into the
    EWMA so the next dispatch aggressively expands the window. Default 1.5."""
    try:
        f = float(os.environ.get("JARVIS_LOCAL_TIMEOUT_ESCALATION_FACTOR", "1.5"))
        return f if f > 1.0 else 1.5
    except (TypeError, ValueError):
        return 1.5


def _ewma_alpha() -> float:
    """EWMA blend weight for SUCCESS samples (decays an escalated budget back
    toward the real latency). Default 0.3. Timeouts jump UP (asymmetric max)."""
    try:
        a = float(os.environ.get("JARVIS_LOCAL_EWMA_ALPHA", "0.3"))
        return a if 0.0 < a <= 1.0 else 0.3
    except (TypeError, ValueError):
        return 0.3


class GracefulStreamInterruption(BaseException):
    """Cooperative freeze-mid-sentence (wall-clock cap / SIGTERM / Spot preemption).

    Strict Exception Hierarchy Elevation: inherits from ``BaseException`` (like
    ``asyncio.CancelledError`` / ``SystemExit``), NOT ``Exception`` -- so it PIERCES
    the Venom tool loop's ``except Exception`` per-round guards (the Earmuff Bypass)
    and propagates straight to the dispatch's explicit ``except GracefulStreamInterruption``
    checkpoint boundary, instead of being swallowed as a round failure (whack-a-mole).
    Distinct from a network drop / stall -- a DELIBERATE orderly suspend. Carries the
    exact buffered ``partial`` thought so the checkpointer preserves it and window-2
    resume prefills it. NON-recoverable (the op suspends -> checkpoint, never retry)."""
    failure_class = "graceful_stream_interruption"

    def __init__(self, message: str = "", *, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial


class InterTokenStall(RuntimeError):
    """Asynchronous Inter-Token Watchdog trip: the streamed generation went silent
    (no token chunk within the inter-token timeout). A stalled stream = a wedged
    worker; NON-recoverable (the L7 auto-heal seals/halts, never retries). A stream
    that keeps emitting is allowed to run indefinitely -- total duration is NOT a
    kill condition on the streaming (heavy) path."""
    failure_class = "inter_token_stall"


def _streaming_enabled() -> bool:
    """Master switch for the streaming inter-token watchdog on the heavy (num_ctx)
    generation path. Default TRUE. OFF -> legacy total-duration adaptive timeout."""
    return _envb("JARVIS_LOCAL_STREAMING_ENABLED", True) if os.environ.get(
        "JARVIS_LOCAL_STREAMING_ENABLED") is not None else True


def _stream_usage_enabled() -> bool:
    """Request the engine's own token accounting on the streaming path.

    Default TRUE. OFF -> no ``stream_options`` is sent and the chars/4
    estimate resumes (labelled ``tokens_estimated=True``), which is the
    byte-identical pre-slice request body for any engine that turns out to
    mind the field."""
    return _envb("JARVIS_LOCAL_STREAM_USAGE_ENABLED", True)


def _inter_token_adaptive_enabled() -> bool:
    """Derive the stall deadline from MEASURED physics instead of a constant.
    Default TRUE. OFF -> the static value for both deadlines (legacy)."""
    return _envb("JARVIS_LOCAL_INTER_TOKEN_ADAPTIVE_ENABLED", True)


def _inter_token_stall_multiple() -> float:
    """How many normal inter-token gaps constitute a STALL. Default 12.

    A stall is not "slow", it is "stopped". At 220 tok/s a normal gap is
    ~4.5ms, so twelve of them is ~54ms -- which is why the floor below
    exists and does the real work on fast hosts."""
    return max(2.0, _f_env("JARVIS_LOCAL_INTER_TOKEN_STALL_MULTIPLE", 12.0))


def _inter_token_floor_s() -> float:
    """Tightest steady-state deadline the adaptive path may produce.

    Default 2.0s. Below this, ordinary LAN jitter and server-side chunk
    batching would be indistinguishable from a wedged peer, and a false
    positive costs a real generation."""
    return max(0.25, _f_env("JARVIS_LOCAL_INTER_TOKEN_FLOOR_S", 2.0))


def _inter_token_timeout_s() -> float:
    """Max wall-time between streamed token chunks before the Stream Breaker trips.
    The model may run indefinitely as long as it emits within this gap. Default 30s."""
    return max(1.0, _f_env("JARVIS_LOCAL_INTER_TOKEN_TIMEOUT_S", 30.0))


_SSE_DONE = object()  # sentinel: the [DONE] terminator of an OpenAI-compat SSE stream


class _SSEUsage(NamedTuple):
    """The engine's OWN token accounting, carried on the stream's final frame.

    Distinct from a content delta so the read loop cannot mistake accounting
    for output: this never appends to the response buffer.
    """

    prompt_tokens: int
    completion_tokens: int


def _parse_sse_delta(line: bytes) -> "Any":
    """Parse ONE line of an ollama /v1/chat/completions SSE stream. Returns the
    incremental content string, the ``_SSE_DONE`` sentinel on ``data: [DONE]``,
    an :class:`_SSEUsage` on the terminal accounting frame, or None for
    keep-alives / non-data / parse errors. Pure + fail-soft.

    The accounting frame carries ``choices: []`` -- which is precisely the
    shape the old ``if not choices: return None`` discarded. That discard is
    why the streaming path had no token counts to report and fell back to a
    ``len(text) // 4`` guess.
    """
    try:
        s = line.decode("utf-8", "ignore").strip() if isinstance(line, (bytes, bytearray)) else str(line).strip()
        if not s or not s.startswith("data:"):
            return None
        payload = s[len("data:"):].strip()
        if payload == "[DONE]":
            return _SSE_DONE
        import json as _json  # noqa: PLC0415
        obj = _json.loads(payload)
        choices = obj.get("choices") or []
        if not choices:
            # No choices AND a usage object -> the accounting frame. No
            # choices and no usage -> a keep-alive, as before.
            usage = obj.get("usage")
            if isinstance(usage, dict):
                _ct = int(usage.get("completion_tokens", 0) or 0)
                if _ct > 0:
                    return _SSEUsage(
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=_ct,
                    )
            return None
        delta = (choices[0] or {}).get("delta") or {}
        return delta.get("content") or None
    except Exception:  # noqa: BLE001
        return None


def _emit_stream_token(text: str) -> None:
    """Yield a streamed chunk to stdout for real-time observability (constraint 2).
    Best-effort -- observability never breaks the generation. The 'wall yields to an
    active stream' behavior (constraint 3) is achieved structurally at the dispatch
    layer: the streaming path drops the outer op-deadline wait_for, so an
    actively-emitting call is bounded ONLY by the per-chunk inter-token watchdog +
    the STATIC hard wall-clock cap (kept blind per the Slice-47 Watchdog Isolation
    Invariant -- never coupled to stream state)."""
    try:
        import sys  # noqa: PLC0415
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    # Feed the streaming liveness heartbeat -> the IDLE/staleness watchdog stays
    # fresh while tokens flow (so a streaming op is never idle-killed). Does NOT
    # touch the wall-clock cap (Slice-47: that stays blind). Best-effort.
    try:
        from backend.core.ouroboros.governance import stream_heartbeat as _hb  # noqa: PLC0415
        _hb.pulse()
    except Exception:  # noqa: BLE001
        pass


class LocalMemoryCritical(RuntimeError):
    """Raised when host memory is CRITICAL at local-generate admission time.

    The local tier evicts the model and refuses the op so the cascade routes
    upstream to remote providers instead of OOM-ing the host. Consumed by
    classify_local_failure -> PRIMARY_DEGRADED.
    """
    failure_class = "local_memory_critical"


def render_structured_prompt(*, task: str, constraints: List[str], files: Dict[str, str]) -> str:
    """Structured-prompt discipline for the local 3B: rigid bounded tags, no loose NL."""
    parts = ["<task>", task, "</task>", "<constraints>"]
    parts += [f"- {c}" for c in constraints]
    parts += ["</constraints>", "<files>"]
    for path, body in files.items():
        parts += [f'<file path="{path}">', body, "</file>"]
    parts += ["</files>", "<output_format>full_content</output_format>"]
    return "\n".join(parts)


@dataclass
class LocalCompletion:
    text: str
    output_tokens: int
    ttft_ms: float
    total_ms: float
    # Split, and labelled. tok/s is completion_tokens over the generation
    # duration, so folding the prompt in makes throughput unrecoverable --
    # and an ESTIMATE reported as a measurement is worse than no number at
    # all, because a model A/B is decided on exactly this quantity. When
    # the engine reports its own usage, `tokens_estimated` is False and the
    # counts are the engine's; otherwise they are a chars/4 guess and the
    # flag says so, all the way out to the trajectory corpus.
    prompt_tokens: int = 0
    tokens_estimated: bool = True


class LocalPrimeClient:
    """aiohttp connection-pooled client -> Ollama OpenAI-compat endpoint.

    A persistent session (lazily built, or injected for tests) with a bounded
    TCPConnector + keep-alive eliminates per-call socket setup across L2 passes.
    """

    def __init__(self, cfg: LocalConfig, session: Optional[Any] = None,
                 profiler: "Optional[LatencyProfiler]" = None) -> None:
        self._cfg = cfg
        self._session = session
        # Stateful Latency Profiler: when injected (a session-scoped singleton kept
        # per-endpoint by the dispatcher), the EWMA/sample window SURVIVES across
        # ops + L7 retries -- so the client learns the 32B's real latency (incl. the
        # one-time ~109s load) and adapts its timeout up instead of resetting to the
        # cold seed on every fresh client (the "profiler amnesia").
        self.profiler = profiler if profiler is not None else LatencyProfiler(cfg)
        self._governor: Any = None
        # LLM Prefill Re-Ignition: when set (by the dispatch on a RESUMED op), the
        # next generation continues from this saved partial thought instead of
        # starting over. Consumed once per generate() call.
        self._resume_prefill: str = ""

    def attach_governor(self, governor: Any) -> None:
        """Attach a LocalInferenceDirector so generate() consults memory_guard()
        before each local inference (host-OOM protection). When unattached,
        behavior is byte-identical to the ungoverned path."""
        self._governor = governor

    async def _ensure_session(self) -> Any:
        if self._session is None:
            import aiohttp  # local import keeps module import cheap when OFF
            conn = aiohttp.TCPConnector(
                limit=self._cfg.pool_limit,
                limit_per_host=self._cfg.pool_limit,
                keepalive_timeout=max(30, self._cfg.keep_alive_seconds),
            )
            self._session = aiohttp.ClientSession(
                connector=conn, headers={"Connection": "keep-alive"},
            )
        return self._session

    async def complete(self, *, system: str, user: str, prompt_tokens: int,
                       temperature: float = 0.2,
                       max_tokens: "Optional[int]" = None,
                       stream: "Optional[bool]" = None,
                       prefill: str = "",
                       sampling: "Optional[Any]" = None,
                       response_format: Any = RESPONSE_FORMAT_LADDER,
                       on_token: "Optional[Any]" = None) -> LocalCompletion:
        sess = await self._ensure_session()
        url = chat_endpoint(self._cfg)
        # Dynamic Cognitive Compression + num_ctx injection (Context-Hardware
        # Negotiator). When a VRAM-safe num_ctx is configured, fit the payload to
        # the INPUT budget (num_ctx minus reserved output) so the KV cache can never
        # overflow VRAM -> no ServerDisconnect. The system prompt (Iron Gate rules)
        # is preserved in full; older intermediate history is compressed. num_ctx is
        # also declared to the engine so it never pre-allocates a fatal KV cache.
        if self._cfg.num_ctx:
            # Reserve only a BOUNDED output slice (not the full max_tokens cap) so
            # the input window stays wide -> less compression -> fewer empty results.
            _reserve_out = min(
                max_tokens or 0,
                _int_env("JARVIS_FAILOVER_OUTPUT_RESERVE_TOKENS", _OUTPUT_RESERVE_TOKENS_DEFAULT),
            ) if max_tokens else _int_env(
                "JARVIS_FAILOVER_OUTPUT_RESERVE_TOKENS", _OUTPUT_RESERVE_TOKENS_DEFAULT)
            _in_budget = max(256, int(self._cfg.num_ctx) - _reserve_out)
            system, user, _compressed = fit_prompt_to_window(
                system, user, max_tokens=_in_budget,
            )
            if _compressed:
                logger.info(
                    "[LocalPrimeClient] Cognitive Compression applied: fit payload "
                    "to %d-token input budget (num_ctx=%d, reserved_out=%d)",
                    _in_budget, int(self._cfg.num_ctx), _reserve_out,
                )
        # Explicit prefill arg wins; else the client-carried resume prefill (set by
        # the dispatch on a RESUMED op), consumed once.
        _eff_prefill = prefill or self._resume_prefill
        self._resume_prefill = ""
        _messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # LLM Prefill Re-Ignition: on RESUME, inject the saved partial thought as a
        # trailing assistant message so the model CONTINUES it (ollama/OpenAI treat a
        # trailing assistant message as a prefill to keep typing) instead of starting
        # the generation over. The returned text is prefill + continuation.
        if _eff_prefill:
            _messages.append({"role": "assistant", "content": _eff_prefill})
        body: Dict[str, Any] = {
            "model": self._cfg.model_name,
            "keep_alive": self._cfg.keep_alive_seconds,
            "temperature": temperature,
            "messages": _messages,
        }
        if self._cfg.num_ctx:
            # ollama-native option; harmless if a given engine ignores it (the
            # compression above is the hard guarantee, this is the declarative one).
            body["options"] = {"num_ctx": int(self._cfg.num_ctx)}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # Constrained decoding. Every schema this provider speaks (2b.1
        # candidates, 2b.2-tool calls) is JSON, and a local mid-size model
        # reliably breaks it in ONE specific way: asked to place a whole source
        # file into a JSON string, it reaches for Python triple-quotes --
        #     "full_content": """\"\"\"module docstring...
        # -- which is not representable in JSON and fails at the same offset
        # every time. Live soak: 6 of 6 candidate payloads, while the smaller
        # tool-call payloads parsed fine.
        #
        # Prompting harder is the wrong lever: it lowers the probability of
        # invalid output without bounding it. `response_format=json_object`
        # constrains the SAMPLER, so invalid JSON stops being representable at
        # all -- a guarantee rather than an improvement. This is the OpenAI-
        # compatible spelling because the request targets /v1/chat/completions;
        # ollama's native `format` field belongs to /api/chat and would be
        # silently ignored here.
        #
        # Advertised as opt-out (default ON) but harmless where unsupported: an
        # engine that does not know the field ignores it, which is exactly the
        # pre-existing behaviour.
        # The constraint is the CALL's, not the client's: the ladder for a
        # candidate generation (the default -- byte-identical), an explicit
        # shape when the caller has one, nothing at all for prose.
        if response_format is RESPONSE_FORMAT_LADDER:
            _apply_response_format(body, self._cfg)
        elif response_format:
            _apply_explicit_response_format(body, self._cfg, response_format)
        # Thinking budget. Same seam, same discipline: declared here once so
        # the streaming and non-streaming paths cannot disagree, and dropped
        # by observation if the engine refuses the field.
        _apply_reasoning_effort(body, self._cfg)
        _apply_draft_tokens(body, self._cfg)
        # Per-draw sampling point. Placed AFTER the num_ctx block, which
        # assigns `body["options"]` wholesale -- setting sampling first
        # would have it silently overwritten, the exact class of bug this
        # whole change is fixing.
        # Per-draw sampling point, resolved immutably so one sibling's seed
        # can never persist onto the shared client config.
        _sampling_applied = _apply_sampling(
            body, _config_for_draw(self._cfg, sampling),
        )
        if _sampling_applied:
            logger.info(
                "[LocalPrimeClient] per-draw sampling applied: %s",
                _sampling_applied,
            )
        # Built once in OpenAI spelling; translated once for the wire it is
        # about to travel. The native route is where ``options`` bites.
        _spell_for_transport(body, self._cfg)

        _use_stream = stream if stream is not None else (
            bool(self._cfg.num_ctx) and _streaming_enabled())
        if _use_stream:
            return await self._complete_streaming(
                sess, url, body, prefill=_eff_prefill,
                prompt_tokens=prompt_tokens, temperature=temperature,
                sampling=sampling, on_token=on_token,
            )

        t0 = time.monotonic()
        async with sess.post(url, json=body) as resp:
            # Capability probe by OBSERVATION, not by version table. A 4xx here
            # is the engine telling us it will not accept this grammar; the only
            # honest response is to believe it, remember it, and retry once at
            # the next rung down. Restricted to 4xx: a 5xx is the engine failing,
            # not refusing, and degrading on it would silently disable schema
            # enforcement for the whole process over a transient blip.
            _has_draft = isinstance(body.get("options"), dict) and (
                "draft_num_predict" in body["options"]
            )
            if 400 <= resp.status < 500 and (
                "response_format" in body
                or "format" in body
                or "reasoning_effort" in body
                or "think" in body
                or _has_draft
            ):
                _body_txt = await resp.text()
                # Attribute the refusal to the field the engine actually
                # named. Blaming the schema for a reasoning_effort rejection
                # would disable constrained decoding for the whole process
                # over an unrelated field -- and constrained decoding is what
                # made invalid JSON unrepresentable in the first place.
                if _has_draft and "draft" in _body_txt.lower():
                    # Cheapest thing to give up: dropping speculation
                    # changes throughput, never output.
                    _degraded = _degrade_draft_tokens(body, self._cfg)
                    _what = "draft_num_predict"
                elif (
                    ("reasoning_effort" in body and "reasoning_effort" in _body_txt)
                    or ("think" in body and "think" in _body_txt.lower())
                ):
                    _degraded = _degrade_reasoning_effort(body, self._cfg)
                    _what = "reasoning_effort"
                elif "response_format" in body or "format" in body:
                    _degraded = _degrade_response_format(body, self._cfg)
                    _what = "json_schema"
                else:
                    # Neither field is implicated: a 400 that names no
                    # constraint we attached is the engine refusing the
                    # REQUEST (bad model, bad auth), not a grammar. Blaming
                    # a constraint that was never sent would record a false
                    # capability and cost a retry for nothing.
                    _degraded = False
                    _what = ""
                if _degraded:
                    logger.info(
                        "[LocalPrimeClient] retrying once without %s "
                        "(engine said %s: %s)",
                        _what, resp.status, _body_txt[:160],
                    )
                    async with sess.post(url, json=body) as resp2:
                        data = await _read_json(resp2)
                else:
                    resp.raise_for_status()
                    data = await _read_json(resp)
            else:
                data = await _read_json(resp)
        total_ms = (time.monotonic() - t0) * 1000.0
        # Shape-dispatched: reads the native reply (``message.content`` +
        # ``eval_count``) or the OpenAI one (``choices[0]`` + ``usage``).
        text, _reported, _prompt_reported = _extract_completion(data)
        _usage = {"prompt_tokens": _prompt_reported}
        # Estimation is the LAST resort, and it is labelled when used. The
        # `or` chain that preceded this silently produced the same int for
        # a measurement and a guess, so nothing downstream could tell a
        # real tok/s from a fabricated one.
        out_toks = _reported or max(1, len(text) // 4)
        ttft_ms = min(total_ms, 0.1 * total_ms)
        self.profiler.record(ttft_ms=ttft_ms, total_ms=total_ms, output_tokens=out_toks)
        return LocalCompletion(
            text=text, output_tokens=out_toks, ttft_ms=ttft_ms, total_ms=total_ms,
            prompt_tokens=int(_usage.get("prompt_tokens", 0) or 0),
            tokens_estimated=(_reported <= 0),
        )

    async def _complete_streaming(self, sess: Any, url: str, body: Dict[str, Any],
                                  *, prefill: str = "",
                                  prompt_tokens: "Optional[int]" = None,
                                  temperature: "Optional[float]" = None,
                                  sampling: "Optional[Any]" = None,
                                  on_token: "Optional[Any]" = None) -> LocalCompletion:
        """Streaming generation with the Asynchronous Inter-Token Watchdog +
        cooperative shutdown. Reads the SSE stream chunk-by-chunk; each ``readline``
        is bounded by the inter-token timeout (NOT the total duration). Between
        chunks it polls the cooperative-shutdown signal and, if set, raises
        GracefulStreamInterruption carrying the exact buffered partial (freeze
        mid-sentence -> the loop never holds the graceful shutdown hostage). On
        RESUME the *prefill* seeds the buffer so the returned text is the prior
        partial + the continuation (the model resumes from the interrupted char).
        Records the REAL end-to-end latency on a clean finish."""
        from backend.core.ouroboros.governance import cooperative_shutdown as _coop  # noqa: PLC0415
        body = dict(body)
        body["stream"] = True
        # ASK for the accounting. A streamed response carries no usage object
        # unless the client requests one, which is why this path had nothing
        # to report and guessed `len(text) // 4` instead -- a guess that then
        # travelled into the trajectory corpus as if it were a measurement.
        # Verified against the engine this lane serves (ollama 0.33.1): the
        # terminal frame carries exact prompt/completion counts.
        #
        # Additive and free to ignore: an engine that does not know the field
        # returns the same stream it always did, `_stream_usage` stays None,
        # and the estimate resumes -- now labelled as one. There is nothing
        # to degrade, so this needs no capability-rejection cache like the
        # schema / reasoning_effort / draft ladders above.
        if _stream_usage_enabled() and not is_native_transport(self._cfg):
            # OpenAI-only accounting request. The native stream's terminal
            # frame carries eval_count/prompt_eval_count unasked.
            body["stream_options"] = {"include_usage": True}
        # TWO deadlines, not one. The wait before the first chunk measures
        # PREFILL (grows with prompt size); every wait after it measures the
        # gap between tokens (roughly constant). A single constant is loose
        # enough to hide a wedged peer for 6,600 normal gaps on a fast host,
        # AND tight enough to declare a large legitimate prefill "wedged".
        # See LatencyProfiler.inter_token_budget_s.
        _first_token_s, _steady_token_s = self.profiler.inter_token_budget_s(
            prompt_tokens=prompt_tokens, temperature=temperature,
            sampling=sampling,
        )
        # Say what was ARMED. The banner in `complete_guarded` prints the
        # static knob, and the only other place these deadlines appear is the
        # stall message -- so on a healthy run the derived budget was
        # unobservable, which is exactly how "wired but inert" hides. One
        # INFO line per stream is the proof a soak can grep for.
        logger.info(
            "[LocalPrimeClient] stream watchdog armed: first=%.1fs steady=%.1fs "
            "prompt_tokens=%s entropy=%.2f",
            _first_token_s, _steady_token_s,
            "?" if prompt_tokens is None else int(prompt_tokens),
            entropy_latency_factor(temperature, sampling),
        )
        # TRANSPORT FLOOR — added, never substituted.
        #
        # A stall deadline must cover the time the MODEL needs plus the time
        # the NETWORK takes to deliver it. The model term comes from the
        # physics ledger; this one comes from measured inter-arrival variance
        # and rises automatically when the path degrades mid-stream (Tailscale
        # can flip direct -> DERP without warning). Without it, the ~2.0s
        # steady deadline computed from a fast host would sever perfectly
        # healthy streams the moment the operator left the LAN.
        try:
            from backend.core.ouroboros.governance.transport_profile import (
                profile_for as _tprofile,
            )
            _tp = _tprofile(self._cfg.base_url)
            _steady_token_s = _steady_token_s + _tp.floor_s()
        except Exception:  # noqa: BLE001
            _tp = None
        inter_token_s = _first_token_s
        _last_chunk_at = time.monotonic()
        # Seed with the resume prefill so the assembled text continues the partial.
        parts: List[str] = [prefill] if prefill else []
        ttft_ms = 0.0
        t0 = time.monotonic()
        first = True
        _stream_usage: "Optional[_SSEUsage]" = None

        def _freeze() -> "GracefulStreamInterruption":
            return GracefulStreamInterruption(
                "cooperative shutdown (%s) mid-stream" % _coop.reason(),
                partial="".join(parts),
            )

        # Preemptive Asynchronous Race, phase 0 (response-begin): the server does
        # not return response HEADERS until the model prefill completes (1-4 min on
        # a heavy tier), so the ``post`` await is the LONGEST blocking window of a
        # streaming call. A cooperative shutdown during prefill must freeze NOW with
        # the prefill-seed partial -- not sit deaf until an outer cancel bypasses the
        # checkpoint path (live gap bt-iso-1782942507: SIGTERM at 59s into prefill,
        # tokens=0, no stall -> GSI never raised, 0 checkpoints).
        _shutdown_task = asyncio.ensure_future(_coop.wait_async())
        # Transport sovereignty: response-begin patience is owned by THIS
        # machinery (cooperative race + heartbeat + the pool/audit ceilings
        # above), NEVER by aiohttp's ambient 300s session default -- which
        # killed L4-queued sealed ops at exactly 300s while they legitimately
        # waited behind another op's generation (iso-a1-20260701-172436).
        # Connection ESTABLISHMENT stays bounded (a dead host fails fast).
        _req_kw: Dict[str, Any] = {"json": body}
        try:
            import aiohttp  # noqa: PLC0415 -- session is aiohttp when live
            _req_kw["timeout"] = aiohttp.ClientTimeout(
                total=None,
                connect=float(os.environ.get(
                    "JARVIS_STREAM_CONNECT_TIMEOUT_S", "30") or 30),
            )
        except Exception:  # noqa: BLE001 -- fakes/tests without aiohttp
            pass
        _req_cm = sess.post(url, **_req_kw)
        try:
            _enter_task = asyncio.ensure_future(_req_cm.__aenter__())
            # Prefill-aware heartbeat: while headers are withheld (model
            # prefill / L4 queue) the GPU is genuinely working -- pulse every
            # JARVIS_STREAM_REQUEST_PULSE_S so the idle watchdog + the audit
            # deferral probe see ACTIVE (tokens can't pulse yet by definition).
            _pulse_s = float(os.environ.get("JARVIS_STREAM_REQUEST_PULSE_S", "15") or 15)
            while True:
                _done_begin, _ = await asyncio.wait(
                    {_enter_task, _shutdown_task},
                    timeout=max(0.01, _pulse_s),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if _done_begin:
                    break
                _emit_stream_token("")  # heartbeat-only pulse (empty delta)
            if _enter_task not in _done_begin:
                # Shutdown fired while awaiting response headers (prefill) ->
                # drop the in-flight request and freeze this millisecond.
                _enter_task.cancel()
                raise _freeze()
            resp = _enter_task.result()  # re-raises genuine request errors faithfully
            try:
                reader = resp.content  # aiohttp StreamReader (line-iterable)
                # Phase 1 (chunk loop): the SAME long-lived waiter raced against
                # each readline. The instant the OS signal fires the event, the
                # readline task is dropped and we yield the partial to THIS
                # millisecond -- zero-latency, not up to the inter-token window.
                while True:
                    if _coop.is_requested():  # cheap fast-path (already requested)
                        raise _freeze()
                    _read_task = asyncio.ensure_future(reader.readline())
                    done, _pending = await asyncio.wait(
                        {_read_task, _shutdown_task},
                        timeout=inter_token_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if _read_task in done:
                        # Feed the ARRIVAL back into the transport profile
                        # before anything else. This is what makes the budget
                        # roll: each inter-chunk gap is a sample, so a path
                        # that degrades mid-stream widens the deadline within
                        # a chunk or two rather than after the stream dies.
                        if _tp is not None:
                            _now_chunk = time.monotonic()
                            _tp.observe((_now_chunk - _last_chunk_at) * 1000.0)
                            _last_chunk_at = _now_chunk
                        # Generation is flowing: every subsequent wait is a
                        # STEADY-STATE gap, so tighten from the prefill budget
                        # to the measured inter-token one. Reassigned on each
                        # chunk (not once) so a mid-stream physics refresh is
                        # picked up without restarting the stream.
                        inter_token_s = _steady_token_s
                        # A chunk (or EOF) is already in hand -- NEVER drop it, even if
                        # shutdown fired in the same tick. We buffer it here; the
                        # cooperative check at the top of the NEXT iteration freezes
                        # AFTER this chunk is appended, so the partial loses nothing.
                        line = _read_task.result()
                    elif _shutdown_task in done:
                        # Read still blocked -> drop the in-flight I/O + yield NOW
                        # (zero-latency: this millisecond, not after the read unblocks).
                        _read_task.cancel()
                        raise _freeze()
                    else:
                        # Neither fired within the window -> the stream is wedged.
                        _read_task.cancel()
                        # PENALISE THE LEDGER BEFORE RAISING.
                        #
                        # Without this the stall is invisible to the physics:
                        # record() only fires on a clean finish, so a host that
                        # wedges every op keeps its optimistic EWMA forever and
                        # the ThroughputGovernor keeps sizing lanes for a
                        # machine that is actively failing -- a wrong answer
                        # derived from stale-but-honest data, which is the
                        # hardest kind to notice. record_timeout_penalty is the
                        # EXISTING asymmetric-escalation seam (the non-streaming
                        # path already used it); this puts the streaming path,
                        # which is the one the LAN bridge forces, on the same
                        # footing. Fail-soft: telemetry must never mask the
                        # stall it is describing.
                        try:
                            self.profiler.record_timeout_penalty(
                                max(1.0, (time.monotonic() - t0) * 1000.0))
                        except Exception:  # noqa: BLE001
                            pass
                        raise InterTokenStall(
                            "inter-token stall: no chunk within %.1fs "
                            "(stream wedged; first_token_budget=%.1fs)"
                            % (inter_token_s, _first_token_s)
                        )
                    if not line:
                        break  # EOF -> stream complete
                    delta = _parse_stream_line(line)
                    if delta is _SSE_DONE:
                        break
                    if isinstance(delta, _SSEUsage):
                        # The accounting frame. NOT output: it must never
                        # reach `parts` or `_emit_stream_token`, and it is
                        # not a token arrival, so `first`/ttft stay put.
                        _stream_usage = delta
                        continue
                    if delta:
                        if first:
                            ttft_ms = (time.monotonic() - t0) * 1000.0
                            first = False
                        parts.append(delta)
                        _emit_stream_token(delta)
                        # The CALLER's view of the stream — the render
                        # conductor, when the dispatcher opened a reasoning
                        # stream for this generation. Until this existed the
                        # local lane's tokens reached stdout and the watchdog
                        # and nothing that could show an operator the model
                        # writing. A render fault never breaks the stream.
                        if on_token is not None:
                            try:
                                on_token(delta)
                            except Exception:  # noqa: BLE001
                                pass
            finally:
                try:
                    await _req_cm.__aexit__(None, None, None)
                except BaseException:  # noqa: BLE001 -- release never masks the freeze
                    pass
        finally:
            if not _shutdown_task.done():
                _shutdown_task.cancel()
        total_ms = (time.monotonic() - t0) * 1000.0
        text = "".join(parts)
        # Engine-reported when the stream carried an accounting frame; the
        # chars/4 guess only when it did not -- and then said to be a guess.
        # On a RESUME the count describes the continuation, not the seeded
        # prefill -- which is the right pairing, because `total_ms` is this
        # request's duration too, so the tok/s they form is this request's.
        _measured = _stream_usage is not None
        out_toks = _stream_usage.completion_tokens if _measured else max(1, len(text) // 4)
        # A completed stream is a REAL latency sample -> the EWMA learns + decays.
        self.profiler.record(ttft_ms=ttft_ms or min(total_ms, 0.1 * total_ms),
                             total_ms=total_ms, output_tokens=out_toks)
        return LocalCompletion(
            text=text, output_tokens=out_toks, ttft_ms=ttft_ms, total_ms=total_ms,
            prompt_tokens=(_stream_usage.prompt_tokens if _measured else 0),
            tokens_estimated=(not _measured),
        )

    async def complete_guarded(self, *, system: str, user: str, prompt_tokens: int,
                               temperature: float = 0.2,
                               max_tokens: "Optional[int]" = None,
                               sampling: "Optional[Any]" = None,
                               response_format: Any = RESPONSE_FORMAT_LADDER,
                               on_token: "Optional[Any]" = None) -> LocalCompletion:
        # HEAVY (num_ctx) STREAMING path: deprecate the total-duration timeout. The
        # Inter-Token Watchdog inside _complete_streaming is the sole guard -- a
        # model that keeps emitting tokens runs indefinitely; only a STALL trips it.
        # This is the mathematically-robust replacement for guessing total latency.
        if self._cfg.num_ctx and _streaming_enabled():
            logger.info(
                "[LocalPrimeClient] streaming generation (inter-token watchdog=%.0fs, "
                "no total-duration cap) num_ctx=%d",
                _inter_token_timeout_s(), int(self._cfg.num_ctx),
            )
            return await self.complete(
                system=system, user=user, prompt_tokens=prompt_tokens,
                temperature=temperature, max_tokens=max_tokens, stream=True,
                sampling=sampling, response_format=response_format,
                on_token=on_token,
            )

        # SURVIVAL / non-streaming path: legacy total-duration adaptive timeout.
        # May raise UnrecoverableInferenceLatency (absolute breaker) -> terminal.
        # The budget must know what it is buying. A high-entropy sibling
        # produces more tokens than the temperature-0.2 draw the profiler
        # was calibrated on, and sizing it as if it were that draw is what
        # ended soak 14 twenty-five minutes early (89 TimeoutError).
        timeout_ms = self.profiler.adaptive_timeout_ms(
            prompt_tokens=prompt_tokens, temperature=temperature,
            sampling=sampling,
        )
        try:
            return await asyncio.wait_for(
                self.complete(system=system, user=user, prompt_tokens=prompt_tokens,
                              temperature=temperature, max_tokens=max_tokens,
                              sampling=sampling, response_format=response_format,
                              on_token=on_token),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError as e:
            self.profiler.record_timeout_penalty(timeout_ms)
            raise LocalLatencyLockup(
                f"local_inference timeout: budget={timeout_ms:.0f}ms "
                f"warm={self.profiler.is_warm()}"
            ) from e

    async def generate(self, prompt: str, system_prompt: "Optional[str]" = None,
                       context: "Optional[Any]" = None, max_tokens: int = 4096,
                       temperature: float = 0.7, model_name: "Optional[str]" = None,
                       task_profile: "Optional[Any]" = None,
                       sampling: "Optional[Any]" = None,
                       response_format: Any = RESPONSE_FORMAT_LADDER,
                       on_token: "Optional[Any]" = None,
                       **kwargs: Any) -> Any:
        """Drop-in PrimeClient.generate adapter -> PrimeResponse (source=local_prime).

        context/task_profile are accepted for interface parity; the 3B path relies
        on the structured prompt + files already in `prompt` (documented v1 limit).

        ``sampling`` is this draw's point in sampling space (anything with
        ``config_overrides()``, or a plain mapping). It is NAMED rather than
        left to ride ``**kwargs`` because ``**kwargs`` here is a silent
        sink: the pre-existing signature already swallowed every extra a
        caller passed, which is how ``top_p``/``top_k``/``seed`` could be
        threaded this far and vanish without one error. A name is the only
        spelling a caller cannot get silently wrong.
        """
        if self._governor is not None:
            await self._governor.memory_guard()
        import uuid
        from backend.core.prime_client import PrimeResponse
        sys_txt = system_prompt or ""
        est_tokens = max(1, (len(prompt) + len(sys_txt)) // 4)
        lc = await self.complete_guarded(
            system=sys_txt, user=prompt, prompt_tokens=est_tokens,
            temperature=temperature, max_tokens=max_tokens,
            sampling=sampling, response_format=response_format,
            on_token=on_token,
        )
        return PrimeResponse(
            content=lc.text,
            request_id=uuid.uuid4().hex,
            model=model_name or self._cfg.model_name,
            source="local_prime",
            latency_ms=lc.total_ms,
            tokens_used=lc.output_tokens,
            # PrimeResponse has ONE token field and it is the completion
            # count, so the prompt half and the measured/estimated label
            # ride in `metadata` -- the only additive seam that does not
            # change a dataclass every provider in the tree constructs.
            # The trajectory recorder reads these; see providers.py.
            metadata={
                "prompt_tokens": lc.prompt_tokens,
                "completion_tokens": lc.output_tokens,
                "tokens_estimated": lc.tokens_estimated,
            },
        )

    async def warmup(self, *, timeout_s: float) -> bool:
        """Force model weights into VRAM via a minimal 1-token generation.

        Fires a lightweight dummy generation (prompt "warmup", num_predict/
        max_tokens 1, temperature 0.0) at the configured endpoint, bounded by
        asyncio.wait_for(timeout_s). Returns True on a successful completion,
        False on timeout or any error. Fail-soft -- never raises.

        This is the cold-load forcing call: awaiting it guarantees the model is
        resident in VRAM before the first real Sovereign generation clock starts.
        Reusable by the FSM AWAKENING gate and the soak harness.
        """
        async def _do_warmup() -> bool:
            sess = await self._ensure_session()
            url = chat_endpoint(self._cfg)
            body = _spell_for_transport({
                "model": self._cfg.model_name,
                "messages": [{"role": "user", "content": "warmup"}],
                "max_tokens": 1,
                "temperature": 0.0,
            }, self._cfg)
            # Dedicated Cold-Start HTTP Context: give the warmup POST its OWN total
            # timeout == the (heavy-mult-scaled) warmup budget, overriding aiohttp's
            # 300s session default. A 32B is ~20GB; the PCIe->VRAM cold-load can
            # exceed 300s, and without this the socket would be dropped mid-transfer
            # (min(720s wait_for, 300s default) = 300s). Fail-soft if aiohttp is
            # unavailable -- fall back to the session default.
            try:
                import aiohttp  # noqa: PLC0415
                _post_kw = {"timeout": aiohttp.ClientTimeout(total=max(1.0, timeout_s))}
            except Exception:  # noqa: BLE001
                _post_kw = {}
            async with sess.post(url, json=body, **_post_kw) as resp:
                await _read_json(resp)
            return True

        try:
            return await asyncio.wait_for(_do_warmup(), timeout=timeout_s)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 -- fail-soft
            return False

    async def _check_health(self) -> Any:
        """Drop-in PrimeClient._check_health -> PrimeStatus (AVAILABLE iff Ollama reachable)."""
        from backend.core.prime_client import PrimeStatus
        try:
            sess = await self._ensure_session()
            url = self._cfg.base_url.rstrip("/") + "/api/tags"
            async with sess.get(url) as resp:
                return PrimeStatus.AVAILABLE if resp.status == 200 else PrimeStatus.UNAVAILABLE
        except Exception:
            return PrimeStatus.UNAVAILABLE

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


async def flush_vram(endpoint: str, model_name: str, *, timeout_s: float = 10.0) -> bool:
    """Deterministic VRAM flush: POST ``keep_alive:0`` to the node so ollama
    immediately unloads the model from VRAM. Fired synchronously by the FSM's
    ``_reap_gpu_node`` BEFORE the GCP delete -- a violent, safe-termination flush so
    the node never lingers holding VRAM. Best-effort -> bool; NEVER raises."""
    if not endpoint or not model_name:
        return False
    try:
        import aiohttp  # noqa: PLC0415
        url = endpoint.rstrip("/") + "/api/generate"
        timeout = aiohttp.ClientTimeout(total=max(1.0, timeout_s))
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json={"model": model_name, "keep_alive": 0}) as resp:
                await resp.read()
        return True
    except Exception:  # noqa: BLE001 -- flush is best-effort; teardown proceeds regardless
        return False


def build_local_prime_client() -> "Optional[LocalPrimeClient]":
    """Factory honoring the master kill-switch. OFF -> None (legacy untouched)."""
    if not local_prime_enabled():
        return None
    return LocalPrimeClient(LocalConfig.from_env())


class LocalInferenceDirector:
    """Lifecycle + memory-aware governance for the local tier."""

    def __init__(self, cfg: LocalConfig, client: Any, gate: Any = None) -> None:
        self._cfg = cfg
        self._client = client
        self._gate = gate if gate is not None else get_default_gate()

    async def _evict_model(self) -> None:
        """Force immediate unload from unified memory via keep_alive:0."""
        try:
            sess = await self._client._ensure_session()
            url = self._cfg.base_url.rstrip("/") + "/api/generate"
            async with sess.post(url, json={"model": self._cfg.model_name, "keep_alive": 0}):
                pass
        except Exception:
            pass  # eviction is best-effort; never raise into the control path

    async def enforce_memory(self, level: PressureLevel) -> None:
        """At CRITICAL: un-bypassable atomic teardown."""
        if level is not PressureLevel.CRITICAL:
            return
        await self._evict_model()   # 1) API unload
        gc.collect()                # 2) dual-stage GC sweep
        gc.collect()
        await asyncio.sleep(0)      # 3) yield to host OS for RAM reclaim

    async def memory_guard(self) -> None:
        """Reuse the shared MemoryPressureGate before local inference.

        At CRITICAL host memory, evict the resident model and refuse the op by
        raising LocalMemoryCritical so the cascade routes upstream to remote
        providers instead of OOM-ing the host. Concurrency is NOT handled here
        (candidate_generator's _jprime_sem already caps local inference at 1).
        Pass-through when the gate master switch is OFF.
        """
        if not memory_gate_enabled():
            return
        level = self._gate.pressure()
        if level is PressureLevel.CRITICAL:
            await self.enforce_memory(level)  # evict + dual gc + yield (existing)
            raise LocalMemoryCritical(
                "host memory CRITICAL - local inference refused; cascading upstream"
            )

    async def stop(self) -> None:
        """Clean teardown: release the pooled session (zero hanging FDs)."""
        try:
            await self._client.aclose()
        except Exception:
            pass


def register_flags(registry: Any) -> int:
    """Module-owned FlagRegistry registration.  NEVER raises.

    Co-located with the consuming code deliberately: the default a caller sees
    and the default the registry advertises are read off the same lines, so
    ``/help flag JARVIS_LOCAL_JSON_MODE_ENABLED`` cannot drift from what
    :func:`_json_mode_enabled` actually does.
    """
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category,
            FlagSpec,
            FlagType,
        )
    except ImportError:
        return 0

    src = "backend/core/ouroboros/governance/local_inference_director.py"
    specs = [
        FlagSpec(
            name="JARVIS_LOCAL_REASONING_EFFORT",
            type=FlagType.STR,
            default="none",
            description=(
                "Thinking budget for reasoning-capable local models "
                "(none|low|medium|high|xhigh; empty omits the field). "
                "Measured on qwen3.8:27b over /v1/chat/completions WITH the "
                "json_schema response_format attached: thinking on -> valid "
                "JSON, 629 chars of reasoning, 6.8s; none -> valid JSON, 0 "
                "reasoning, 1.5s. Correctness is unaffected either way "
                "(ollama returns reasoning in a separate field, so the "
                "constrained content stays schema-valid) -- this is a 4.5x "
                "wall-clock tax on a wall-capped soak. Raise it to buy "
                "reasoning back if it proves to lift candidate quality. "
                "NOTE ollama's native `think` and `chat_template_kwargs "
                "{enable_thinking:false}` are silently IGNORED on this "
                "endpoint; this is the spelling that works."
            ),
            category=Category.TUNING,
            source_file=src,
            example="none",
            since="model A/B arc (2026-08-31)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_JSON_MODE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Outer rung of the constrained-decoding ladder: request JSON "
                "mode on local completions. Default ON because every schema "
                "this client speaks is JSON, so the constraint can only "
                "remove output the parser was going to reject anyway. Set "
                "falsey to restore free-form sampling -- worth doing only to "
                "reproduce a parse failure the constraint would mask."
            ),
            category=Category.SAFETY,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-24)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_JSON_SCHEMA_MODE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Inner rung: constrain the sampler with a full JSON *Schema* "
                "rather than only json_object. json_object guarantees the "
                "output PARSES and says nothing about SHAPE, which is how "
                "candidate_0_missing_rationale and "
                "wrong_schema_version:__missing__ survived it. Degrades "
                "automatically per-engine on a 4xx; requires "
                "JARVIS_LOCAL_JSON_MODE_ENABLED to be on."
            ),
            category=Category.SAFETY,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-24)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_STREAM_USAGE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Ask the engine for its own token accounting on the "
                "streaming path (stream_options.include_usage). Without it a "
                "stream carries no usage object and the path falls back to a "
                "len(text)//4 guess -- reported through the same field a "
                "measurement would use, so tok/s could not be trusted. The "
                "error is not uniform across models (measured: 31% low for "
                "qwen3-coder:30b, 10% for qwen2.5-coder:32b), so it biases a "
                "model A/B rather than cancelling out. Additive: an engine "
                "that ignores the field returns the same stream, and the "
                "estimate resumes -- now labelled tokens_estimated=true. Set "
                "falsey for the byte-identical previous request body."
            ),
            category=Category.OBSERVABILITY,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-31)",
        ),
    ]
    try:
        registry.bulk_register(specs, override=True)
    except Exception:  # noqa: BLE001 -- registration is descriptive, never load-bearing
        return 0
    return len(specs)
