"""IntentPrompter — brief LLM call at op_started for "I'm going to do X".
==========================================================================

Slice 2 of the **Gap #6 closure arc**.

Root problem
------------

For a proactive system like O+V (sensors fire ops without operator
input), the operator has no prior context for *why* an op is starting.
Claude Code gets this for free because the user's prompt provides
context. O+V needs to **manufacture** context — the model should
explicitly state, in 1-2 sentences, what it's about to do.

This module supplies a tightly-bounded async LLM call:

* **Cost-bound**: 50 output tokens max
* **Time-bound**: 5-second timeout (fails silently)
* **Provider**: cheapest available (Tier 0 = DoubleWord) — operator
  cost contract preserved by structurally never escalating
* **Fail-silent**: any error → returns ``None``, op proceeds without
  intent narrative (degraded but functional)

Architectural reuse
-------------------

* Existing :class:`DoublewordProvider` for the LLM call (no new
  provider, no parallel HTTP client)
* :class:`NarrativeChannel` (Slice 1) as the storage + streaming
  destination
* House style: frozen dataclass + ``schema_version`` + module-owned
  ``register_flags`` for graduation

Authority boundary
------------------

* §1 deterministic — pure orchestration; the LLM call IS bounded but
  is a model emission (consistent with Manifesto §6 Boundary
  Principle: deterministic perimeter, agentic content)
* §6 Iron Gate — refuses to issue intent prompts during BG/SPEC
  routes (cost contract preservation: BG ops must not incur extra
  Tier 0/1 spend)
* §7 fail-closed — every failure mode is silent. The op proceeds
  without intent narrative; never raises into the orchestrator
  hot path.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("Ouroboros.IntentPrompter")


# ===========================================================================
# Schema + env vocabulary
# ===========================================================================


INTENT_PROMPTER_SCHEMA_VERSION: str = "intent_prompter.v1"


MASTER_FLAG_ENV_VAR: str = "JARVIS_NARRATIVE_INTENT_ENABLED"
TIMEOUT_ENV_VAR: str = "JARVIS_NARRATIVE_INTENT_TIMEOUT_S"
MAX_TOKENS_ENV_VAR: str = "JARVIS_NARRATIVE_INTENT_MAX_TOKENS"


# Default 5s — fast enough that op_started doesn't wait forever, slow
# enough to give the cheapest provider a comfortable margin.
_DEFAULT_TIMEOUT_S: float = 5.0
_MIN_TIMEOUT_S: float = 0.5
_MAX_TIMEOUT_S: float = 30.0


# Default 50 tokens — plenty for "I'll do X by Y", structurally
# capped so a misbehaving model can't run away.
_DEFAULT_MAX_TOKENS: int = 50
_MIN_MAX_TOKENS: int = 10
_MAX_MAX_TOKENS: int = 200


# ===========================================================================
# Master flag + tunables
# ===========================================================================


def is_master_flag_enabled() -> bool:
    """Should the intent prompt's micro-LLM call happen at op_started?

    Resolved through `narrative_density`, which composes two answers this
    function used to know only half of: an explicitly set
    ``JARVIS_NARRATIVE_INTENT_ENABLED`` still wins (operator specificity
    beats a global dial), and otherwise the operator's `/narrate` level
    decides. Default true post Slice 5 graduation (2026-05-04) — preserved,
    because the default density is ON and intent is audible at ON.

    Checked here rather than at the renderer because this gate is a COST
    gate: refusing later would mean the model call was already spent to
    produce prose the operator asked not to see.

    Falls back to the original read if the dial is unavailable, so this
    module keeps working standalone. NEVER raises.
    """
    try:
        from backend.core.ouroboros.ui.narrative_density import audible
        return audible("narrative.intent")
    except Exception:  # noqa: BLE001
        raw = os.environ.get(MASTER_FLAG_ENV_VAR, "true")
        return raw.strip().lower() not in ("0", "false", "no", "off")


def read_timeout_s() -> float:
    raw = os.environ.get(TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, parsed))


def read_max_tokens() -> int:
    raw = os.environ.get(MAX_TOKENS_ENV_VAR, "").strip()
    if not raw:
        return _DEFAULT_MAX_TOKENS
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS
    return max(_MIN_MAX_TOKENS, min(_MAX_MAX_TOKENS, parsed))


# ===========================================================================
# Frozen records
# ===========================================================================


@dataclass(frozen=True)
class IntentRequest:
    """One intent-prompt invocation context.

    Fields
    ------
    * ``op_id`` — orchestrator op id.
    * ``goal`` — short description of the pending op (sensor-emitted
      goal text).
    * ``risk_tier`` — the tier as a string (``"safe_auto"`` / etc.)
      so the model can frame its intent appropriately.
    * ``target_files`` — first 5 paths the op expects to touch.
    """

    op_id: str
    goal: str
    risk_tier: str
    target_files: tuple
    schema_version: str = INTENT_PROMPTER_SCHEMA_VERSION


@dataclass(frozen=True)
class IntentResult:
    """Outcome of an intent-prompt call.

    Fields
    ------
    * ``prose`` — the model's 1-2 sentence intent (empty on failure).
    * ``provider`` — provider id that produced the prose
      (``"doubleword"`` typically).
    * ``elapsed_s`` — wall-clock duration of the call.
    * ``error`` — short reason on failure (empty on success).
    """

    prose: str
    provider: str
    elapsed_s: float
    error: str = ""
    schema_version: str = INTENT_PROMPTER_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        return bool(self.prose) and not self.error


# ===========================================================================
# Prompt construction — pure, no I/O
# ===========================================================================


_SYSTEM_PROMPT = (
    "You are O+V, a proactive autonomous coding agent. You are about "
    "to begin an operation. In 1 sentence (no preamble, no quotes), "
    "say what you intend to do and why. Use first-person present tense "
    "(e.g., \"I'm going to fix X by Y\"). Be specific and concrete. "
    "Never apologize. Never list multiple options. Maximum 30 words."
)


def build_user_prompt(req: IntentRequest) -> str:
    """Format the user-side prompt from an :class:`IntentRequest`.
    Pure function — no I/O. NEVER raises (safe-coerces all fields)."""
    goal = (req.goal or "").strip() or "(no goal text provided)"
    tier = (req.risk_tier or "").upper() or "UNKNOWN"
    files_part = ""
    if req.target_files:
        files_list = ", ".join(str(f) for f in list(req.target_files)[:5])
        files_part = f"\nTarget files: {files_list}"
    return (
        f"Op: {req.op_id}\n"
        f"Risk tier: {tier}\n"
        f"Goal: {goal}"
        f"{files_part}\n\n"
        "What's your one-sentence intent?"
    )


# ===========================================================================
# Async LLM call — bounded, fail-silent
# ===========================================================================


async def request_intent(
    req: IntentRequest,
    *,
    timeout_s: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> IntentResult:
    """Issue the intent prompt against the cheapest available provider.

    Returns an :class:`IntentResult`. On any failure (master flag off,
    provider unavailable, timeout, exception, BG/SPEC route guard),
    returns a result with empty ``prose`` and a diagnostic ``error``.
    NEVER raises.

    Implementation notes
    --------------------
    * Uses :class:`DoublewordProvider` directly — Tier 0 cost, the
      cheapest channel. No fan-out to Claude (would defeat the
      cost-contract preservation point).
    * The call is wrapped in :func:`asyncio.wait_for` with the
      configured timeout. A timeout returns ``error="timeout"``.
    * Provider construction is lazy + per-call to keep the substrate
      stateless (matches the existing one-shot LLM helper pattern).
    """
    import time
    started = time.monotonic()

    if not is_master_flag_enabled():
        return IntentResult(
            prose="", provider="", elapsed_s=0.0,
            error="master flag off",
        )

    eff_timeout = (
        float(timeout_s) if timeout_s is not None else read_timeout_s()
    )
    eff_max_tokens = (
        int(max_tokens) if max_tokens is not None else read_max_tokens()
    )

    try:
        # Lazy import — keeps the substrate import-cheap and lets
        # tests stub the provider without touching network code.
        from backend.core.ouroboros.governance.doubleword_provider import (
            DoublewordProvider,
        )
    except ImportError as exc:
        return IntentResult(
            prose="", provider="", elapsed_s=time.monotonic() - started,
            error=f"provider unavailable: {exc}",
        )

    user_prompt = build_user_prompt(req)

    async def _call() -> str:
        # Phase 2 RT repositioning (2026-07-16): narrative intent renders
        # inline while the operator watches — it buys TIME. Routed through
        # the unified Claude-RT-first gate router (DW-RT opportunistic,
        # constructed lazily as the fallback handle). Outer timeout and
        # IntentResult semantics unchanged.
        from backend.core.ouroboros.governance.rt_gate import (
            GateProviderExhaustedError,
            gate_completion,
        )

        _dw = None
        try:
            _dw = DoublewordProvider()
        except Exception:  # noqa: BLE001 — DW handle is optional fallback
            _dw = None

        try:
            return await gate_completion(
                user_prompt,
                caller_id="intent_prompter",
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=eff_max_tokens,
                dw_provider=_dw,
            )
        except GateProviderExhaustedError as exc:
            raise RuntimeError(f"rt gate exhausted: {exc}")

    # Failures are logged at INFO, not DEBUG: a session in which the
    # organism's voice went missing must be able to say WHY (2026-09-06:
    # two of five intents rendered nothing and the log could not tell a
    # timeout from an empty answer).
    try:
        prose = await asyncio.wait_for(_call(), timeout=eff_timeout)
    except asyncio.TimeoutError:
        logger.info(
            "[IntentPrompter] intent for op=%s timed out after %.1fs "
            "(budget %s=%.1fs)", req.op_id, time.monotonic() - started,
            TIMEOUT_ENV_VAR, eff_timeout,
        )
        return IntentResult(
            prose="", provider="doubleword",
            elapsed_s=time.monotonic() - started,
            error="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "[IntentPrompter] intent for op=%s failed after %.1fs: %s",
            req.op_id, time.monotonic() - started, exc,
        )
        return IntentResult(
            prose="", provider="doubleword",
            elapsed_s=time.monotonic() - started,
            error=f"call failed: {exc}",
        )

    prose_safe = (prose or "").strip()
    if not prose_safe:
        logger.info(
            "[IntentPrompter] intent for op=%s came back empty after %.1fs",
            req.op_id, time.monotonic() - started,
        )
        return IntentResult(
            prose="", provider="doubleword",
            elapsed_s=time.monotonic() - started,
            error="empty response",
        )

    return IntentResult(
        prose=prose_safe,
        provider="doubleword",
        elapsed_s=time.monotonic() - started,
    )


def _extract_text(result: object) -> str:
    """Defensive text extraction from provider response. NEVER raises."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    # Common shapes: dict with "text"/"content"/"output" keys
    if isinstance(result, dict):
        for k in ("text", "content", "output", "completion"):
            v = result.get(k)
            if isinstance(v, str) and v:
                return v.strip()
    # Object with .text / .content attribute
    for attr in ("text", "content", "output", "completion"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v:
            return v.strip()
    try:
        return str(result).strip()
    except Exception:  # noqa: BLE001
        return ""


# ===========================================================================
# Convenience: emit directly into NarrativeChannel
# ===========================================================================


async def request_intent_and_emit(
    req: IntentRequest,
    *,
    phase: str = "OP_STARTED",
    channel: Optional[object] = None,
    timeout_s: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> IntentResult:
    """One-shot helper that issues the intent prompt AND records the
    result into the :class:`NarrativeChannel` as kind=INTENT.

    Returns the :class:`IntentResult` for caller observation. The
    channel emission is best-effort — if it fails, the result is
    still returned (caller can log).

    Slice 3's renderer subscribes to the channel and surfaces the
    INTENT frame as a ``💭`` line above the active op block.
    """
    result = await request_intent(
        req, timeout_s=timeout_s, max_tokens=max_tokens,
    )
    if not result.succeeded:
        return result

    try:
        from backend.core.ouroboros.battle_test.narrative_channel import (
            get_default_channel,
            NarrativeKind,
        )
    except ImportError:
        return result

    target_channel = channel
    if target_channel is None:
        try:
            target_channel = get_default_channel()
        except Exception:  # noqa: BLE001
            return result

    try:
        # Use the one-shot helper since the prose is already complete.
        target_channel.emit_complete(  # type: ignore[union-attr]
            op_id=req.op_id,
            phase=phase,
            kind=NarrativeKind.INTENT,
            prose=result.prose,
            provider=result.provider,
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "[IntentPrompter] channel emit failed", exc_info=True,
        )

    return result


__all__ = [
    "INTENT_PROMPTER_SCHEMA_VERSION",
    "IntentRequest",
    "IntentResult",
    "MASTER_FLAG_ENV_VAR",
    "MAX_TOKENS_ENV_VAR",
    "TIMEOUT_ENV_VAR",
    "build_user_prompt",
    "is_master_flag_enabled",
    "read_max_tokens",
    "read_timeout_s",
    "request_intent",
    "request_intent_and_emit",
]
