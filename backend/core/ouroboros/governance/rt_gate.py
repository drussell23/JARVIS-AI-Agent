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
    speak on that host. It sits LAST in every order (a paid host is
    byte-identical) unless the caller prefers it, is gated by the lane's
    own master (``JARVIS_LOCAL_PRIME_ENABLED``), and asks the client for
    PROSE: the local client's JSON ladder is a per-call decision, and a
    gate that wants a sentence does not want a candidate object.

Env knobs:
  * ``JARVIS_GATE_CLAUDE_FIRST_ENABLED``  (default true)
  * ``JARVIS_GATE_RT_TIMEOUT_S``          (default 60; floor 5)
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")

_DEFAULT_SYSTEM = (
    "You are a senior AI reasoning engine for the JARVIS Trinity ecosystem. "
    "Think step by step and return well-structured output."
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
) -> str:
    """Single-turn RT completion for a synchronous pipeline gate.

    Claude-RT first (buying time), DW-RT opportunistic fallback (availability),
    per :mod:`rt_gate` module contract. Raises
    :class:`GateProviderExhaustedError` when every tier fails — the caller's
    own fail-open/fail-closed semantics take over from there. Never returns
    an empty string.
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

    # PER-CALL tier order. `prefer` lets a caller that knows something
    # about THIS payload (the adaptive lane router weighs it) put the
    # right tier first, while the others stay the fallback — so a lane
    # choice biases cost/latency and can never become a single point of
    # failure. Unset → the global policy decides, exactly as before. The
    # local tier joins the order only where its lane is enabled, LAST
    # unless preferred: a host with paid lanes sees the order it always saw.
    _pref = (prefer or "").strip().lower()
    _cloud = (_claude, _dw) if claude_first_enabled() else (_dw, _claude)
    if _pref == "dw":
        _cloud = (_dw, _claude)
    elif _pref == "claude":
        _cloud = (_claude, _dw)
    _local_on = local_tier_enabled()
    if _pref == "local" and _local_on:
        tiers = (_local,) + _cloud
    else:
        tiers = _cloud + ((_local,) if _local_on else ())
    for tier in tiers:
        raw = await tier()
        if raw:
            return raw
    _names = {_claude: "claude", _dw: "dw", _local: "local"}
    raise GateProviderExhaustedError(
        f"gate_completion exhausted all RT tiers (caller={caller_id}, "
        f"order={','.join(_names[tier] for tier in tiers)})"
    )
