"""RT Gate Router — Claude-RT-first completion for synchronous pipeline gates.

Phase 2 of the provider-positioning pivot (2026-07-16). The pipeline's
synchronous GATES (triage, critique, lint, heal, narrate, plan, review) are
LATENCY-SENSITIVE: an op the soak/human is waiting on blocks behind each one.
They were historically pointed at DoubleWord's token-priced queue — buying a
sub-cent discount with minutes of wall clock. This module is the ONE place
that encodes the correct positioning:

    Claude sells TIME  → gates buy time  → Claude-RT first.
    DW sells TOKENS    → kept as the OPPORTUNISTIC fallback (its stream-free
                         RT primitive ``complete_sync``), so a Claude outage
                         degrades a gate to slower-but-alive instead of dead.

Design contract (Mandate 3 — no duplicated fallback logic in gate classes):
  * ``gate_completion()`` is the single entry point. Gates pass their prompt
    and (optionally) their injected provider handles; ALL ordering, timeout,
    and fallback policy lives here.
  * Claude resolution is two-tier: an injected ``claude_provider`` (uses its
    resilient ``prompt_only``) else the provider-less, Aegis-wrapped
    ``claude_fallback.claude_inference`` — so gates that were constructed
    with only a DW handle still reach Claude without rewiring constructors.
  * Failures RAISE ``GateProviderExhaustedError`` (typed) only when every
    tier is exhausted; each gate keeps its own fail-open/fail-closed
    semantics around that exception — gate logic is NOT rewritten here
    (Mandate 1).
  * ``JARVIS_GATE_CLAUDE_FIRST_ENABLED`` (default TRUE) is the master; OFF
    restores DW-first ordering (one-flip rollback, byte-equivalent priority).

  * A LOCAL tier (2026-09-06) closes the gate on a host whose only lane
    is the locally served model: with no cloud key both cloud tiers fail
    at second zero and every gate raised, so nothing that speaks through
    a gate -- the narrated intent above an op, most visibly -- could ever
    speak on that host. Wherever its lane is enabled
    (``JARVIS_LOCAL_PRIME_ENABLED``) it is the PRIMARY tier: its marginal
    cost is zero and it answers a gate-sized prompt in well under a
    second, so putting a metered, possibly-dead cloud tier in front of it
    buys nothing and costs a failed round-trip per gate (measured: every
    intent paid a DW 503 before the local lane was asked). The cloud
    tiers remain the fallback, in the order the policy and ``prefer``
    decide. The tier asks the client for PROSE: the local client's JSON
    ladder is a per-call decision, and a gate that wants a sentence does
    not want a candidate object.
  * An EMPTY tier answer is logged distinctly (``exhausted/empty``) --
    never a silent fall-through. The DreamEngine burned a whole soak
    (bt-2026-07-17-033933) on a tier that returned 0 chars and logged
    nothing; the lesson lives here now, once, for every gate.

Env knobs:
  * ``JARVIS_GATE_CLAUDE_FIRST_ENABLED``  (default true)
  * ``JARVIS_GATE_RT_TIMEOUT_S``          (default 60; floor 5)
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")

_DEFAULT_SYSTEM = (
    "You are a senior AI reasoning engine for the JARVIS Trinity ecosystem. "
    "Think step by step and return well-structured output."
)


def _log_empty(tier: str, caller_id: str, raw: object, res: Any = None) -> None:
    """An EMPTY answer is a distinct diagnostic, never a silent fall-through:
    the tier was reached and produced nothing (the reasoning budget ate the
    output, or the model declined). Reported with what the response object
    can say about itself."""
    logger.info(
        "[RTGate] %s tier generation exhausted/empty for %s "
        "(model=%s, %.1fs, out_tokens=%s) — cascading",
        tier, caller_id,
        getattr(res, "model", "?"),
        float(getattr(res, "latency_s", -1.0) or -1.0),
        getattr(res, "output_tokens", "?"),
    )


class GateProviderExhaustedError(RuntimeError):
    """Every RT tier (Claude injected → Claude fallback → DW-RT) failed for a
    gate completion. Gates decide fail-open vs fail-closed on this."""


def claude_first_enabled() -> bool:
    """Master — ``JARVIS_GATE_CLAUDE_FIRST_ENABLED`` (default true)."""
    return os.environ.get(
        "JARVIS_GATE_CLAUDE_FIRST_ENABLED", "true",
    ).strip().lower() in _TRUTHY


def gate_rt_timeout_s() -> float:
    """Per-tier RT budget (``JARVIS_GATE_RT_TIMEOUT_S``, default 60s)."""
    try:
        return max(5.0, float(os.environ.get("JARVIS_GATE_RT_TIMEOUT_S", "60")))
    except (TypeError, ValueError):
        return 60.0


async def _try_claude(
    prompt: str,
    *,
    caller_id: str,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]],
    timeout_s: float,
    claude_provider: Any = None,
) -> Optional[str]:
    """Claude tier: injected provider's resilient ``prompt_only``, else the
    provider-less Aegis-wrapped ``claude_inference``. Returns text or None
    (this tier's failure is non-fatal — the caller cascades)."""
    # 1a. Injected ClaudeProvider (the resilient, budget-gated path).
    if claude_provider is not None and hasattr(claude_provider, "prompt_only"):
        try:
            raw = await asyncio.wait_for(
                claude_provider.prompt_only(
                    prompt=prompt,
                    caller_id=caller_id,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                ),
                timeout=timeout_s + 5.0,
            )
            if raw:
                return raw
            _log_empty("claude(injected)", caller_id, raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — tier boundary, cascade on
            logger.info(
                "[RTGate] claude(injected) tier failed for %s (%s: %s)",
                caller_id, type(exc).__name__, exc,
            )
    # 1b. Provider-less Claude (constructs its own Aegis-wrapped client).
    try:
        from backend.core.ouroboros.claude_fallback import claude_inference

        raw = await asyncio.wait_for(
            claude_inference(
                prompt,
                caller_id=caller_id,
                response_format=response_format,
                max_tokens=max_tokens,
            ),
            timeout=timeout_s + 5.0,
        )
        if raw:
            return raw
        _log_empty("claude(fallback)", caller_id, raw)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "[RTGate] claude(fallback) tier failed for %s (%s: %s)",
            caller_id, type(exc).__name__, exc,
        )
    return None


async def _try_dw_rt(
    prompt: str,
    *,
    caller_id: str,
    system_prompt: str,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]],
    timeout_s: float,
    dw_provider: Any = None,
    dw_model: Optional[str] = None,
) -> Optional[str]:
    """DW tier — the stream-free RT primitive ``complete_sync`` ONLY (never
    the 24h batch queue: a gate blocks an op, so a batch wait is forbidden
    here by design). Returns text or None."""
    if dw_provider is None or not hasattr(dw_provider, "complete_sync"):
        return None
    try:
        res = await asyncio.wait_for(
            dw_provider.complete_sync(
                prompt,
                system_prompt=system_prompt,
                caller_id=caller_id,
                model=dw_model,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                response_format=response_format,
            ),
            timeout=timeout_s + 5.0,
        )
        raw = getattr(res, "content", "") or ""
        if raw:
            return raw
        _log_empty("dw_rt", caller_id, raw, res)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "[RTGate] dw_rt tier failed for %s (%s: %s)",
            caller_id, type(exc).__name__, exc,
        )
    return None


@contextlib.asynccontextmanager
async def _local_generation_bracket(base_url: str):
    """Count the local call in the gateway's in-flight ledger, so an
    advisory pre-warm cannot read "idle" and evict the weights under it.
    The gateway's own seam for exactly this; absent gateway -> no bracket."""
    cm = None
    try:
        from backend.core.ouroboros.governance.inference_gateway import (
            get_default_gateway,
        )
        cm = get_default_gateway().external_generation(base_url)
    except Exception:  # noqa: BLE001 -- accounting must not block a gate
        cm = None
    if cm is None:
        yield
        return
    async with cm:
        yield


async def _try_local(
    prompt: str,
    *,
    caller_id: str,
    system_prompt: str,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]],
    timeout_s: float,
) -> Optional[str]:
    """Local tier -- the locally served model, one-shot, prose unless the
    caller asked for a shape. Off when the lane's master is off. Returns
    text or None; the client's session is closed on every path."""
    try:
        from backend.core.ouroboros.governance.local_inference_director import (
            LocalConfig,
            LocalPrimeClient,
            local_prime_enabled,
        )
    except Exception as exc:  # noqa: BLE001 -- lane absent on this host
        logger.debug("[RTGate] local tier unavailable: %s", exc)
        return None
    if not local_prime_enabled():
        return None
    client = None
    try:
        cfg = LocalConfig.from_env()
        client = LocalPrimeClient(cfg)
        async with _local_generation_bracket(cfg.base_url):
            res = await asyncio.wait_for(
                client.generate(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    response_format=response_format,
                ),
                timeout=timeout_s + 5.0,
            )
        raw = getattr(res, "content", "") or ""
        if raw:
            return raw
        _log_empty("local", caller_id, raw, res)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "[RTGate] local tier failed for %s (%s: %s)",
            caller_id, type(exc).__name__, exc,
        )
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
    return None


def local_tier_enabled() -> bool:
    """Whether the local tier is consulted -- the lane's own master."""
    try:
        from backend.core.ouroboros.governance.local_inference_director import (
            local_prime_enabled,
        )
        return bool(local_prime_enabled())
    except Exception:  # noqa: BLE001
        return False


def _accepts(accept: Optional[Callable[[str], bool]], raw: str) -> bool:
    """Whether a tier's non-empty text satisfies the caller's shape check.
    A predicate that raises REJECTS — the caller asked for a shape and the
    text could not be shown to have it."""
    if accept is None:
        return True
    try:
        return bool(accept(raw))
    except Exception:  # noqa: BLE001
        return False


async def gate_completion_detailed(
    prompt: str,
    *,
    caller_id: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    claude_provider: Any = None,
    dw_provider: Any = None,
    dw_model: Optional[str] = None,
    prefer: Optional[str] = None,
    accept: Optional[Callable[[str], bool]] = None,
) -> Tuple[str, str]:
    """:func:`gate_completion`, also reporting WHICH tier answered.

    Returns ``(text, tier)`` with ``tier`` in ``{"claude", "dw", "local"}``
    — for callers that stamp provenance on what they got back (the
    DreamEngine's ``_inference_provider``) without keeping a cascade of
    their own to know it. ``accept`` lets a caller that needs a SHAPE (a
    JSON object) reject a tier's text and cascade on, exactly as a failed
    tier would; a rejection is logged so it never reads as a tier that was
    not attempted. Raises :class:`GateProviderExhaustedError` when every
    tier fails or is rejected.
    """
    t = timeout_s if timeout_s is not None else gate_rt_timeout_s()
    sys_p = system_prompt or _DEFAULT_SYSTEM

    async def _claude() -> Optional[str]:
        return await _try_claude(
            prompt, caller_id=caller_id, max_tokens=max_tokens,
            response_format=response_format, timeout_s=t,
            claude_provider=claude_provider,
        )

    async def _dw() -> Optional[str]:
        return await _try_dw_rt(
            prompt, caller_id=caller_id, system_prompt=sys_p,
            max_tokens=max_tokens, response_format=response_format,
            timeout_s=t, dw_provider=dw_provider, dw_model=dw_model,
        )

    async def _local() -> Optional[str]:
        return await _try_local(
            prompt, caller_id=caller_id, system_prompt=sys_p,
            max_tokens=max_tokens, response_format=response_format,
            timeout_s=t,
        )

    # PER-CALL tier order. The local lane, where enabled, is PRIMARY:
    # zero marginal cost and sub-second for a gate-sized prompt, so a
    # metered cloud tier in front of it can only add a failed round-trip.
    # `prefer` orders the CLOUD fallback — it lets a caller that knows
    # something about THIS payload (the adaptive lane router weighs it)
    # put the right cloud tier first, so a lane choice biases cost/latency
    # and can never become a single point of failure. Unset → the global
    # policy decides. A host with no local lane sees the order it always saw.
    _pref = (prefer or "").strip().lower()
    _c, _d, _l = ("claude", _claude), ("dw", _dw), ("local", _local)
    _cloud = (_c, _d) if claude_first_enabled() else (_d, _c)
    if _pref == "dw":
        _cloud = (_d, _c)
    elif _pref == "claude":
        _cloud = (_c, _d)
    tiers = ((_l,) if local_tier_enabled() else ()) + _cloud
    for name, tier in tiers:
        raw = await tier()
        if not raw:
            continue
        if _accepts(accept, raw):
            return raw, name
        logger.info(
            "[RTGate] %s tier answered for %s but the caller rejected the "
            "shape (%d chars) — cascading", name, caller_id, len(raw),
        )
    raise GateProviderExhaustedError(
        f"gate_completion exhausted all RT tiers (caller={caller_id}, "
        f"order={','.join(name for name, _tier in tiers)})"
    )


async def gate_completion(
    prompt: str,
    *,
    caller_id: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    claude_provider: Any = None,
    dw_provider: Any = None,
    dw_model: Optional[str] = None,
    prefer: Optional[str] = None,
    accept: Optional[Callable[[str], bool]] = None,
) -> str:
    """Single-turn RT completion for a synchronous pipeline gate.

    Claude-RT first (buying time), DW-RT opportunistic fallback (availability),
    the local lane last where enabled, per :mod:`rt_gate` module contract.
    Raises :class:`GateProviderExhaustedError` when every tier fails — the
    caller's own fail-open/fail-closed semantics take over from there.
    Never returns an empty string.
    """
    raw, _tier = await gate_completion_detailed(
        prompt, caller_id=caller_id, system_prompt=system_prompt,
        max_tokens=max_tokens, response_format=response_format,
        timeout_s=timeout_s, claude_provider=claude_provider,
        dw_provider=dw_provider, dw_model=dw_model, prefer=prefer,
        accept=accept,
    )
    return raw
