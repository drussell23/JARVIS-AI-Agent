"""
Candidate Generator & Failback State Machine
=============================================

Routes code generation requests to a primary provider (GCP J-Prime) or a
fallback provider (local model).  The :class:`FailbackStateMachine` prevents
flapping by requiring N consecutive health probes over a dwell period before
restoring the primary provider.

Key Design Decisions
--------------------

- **Asymmetric transitions**: failover is immediate (one failure triggers
  switch to fallback), but failback requires ``required_probes`` consecutive
  health checks spanning at least ``dwell_time_s`` seconds.
- **Concurrency quotas**: separate :class:`asyncio.Semaphore` instances for
  primary and fallback, preventing thundering-herd overload.
- **Deadline propagation**: every call computes remaining time from the
  caller-supplied deadline and applies it as an ``asyncio.wait_for`` timeout.
- **QUEUE_ONLY**: when both providers are down, the generator raises
  immediately rather than blocking -- the orchestrator is expected to queue
  the operation for later retry.

State Diagram
-------------

.. code-block:: text

    PRIMARY_READY ---[primary_failure]---> FALLBACK_ACTIVE
         ^                                      |     |
         |                                      |     +--[fallback_failure]--> QUEUE_ONLY
         |                              [probe_success]
         |                                      |
         |                                      v
         +---[N probes + dwell met]--- PRIMARY_DEGRADED
                                          |
                                  [probe_failure]
                                          |
                                          v
                                   FALLBACK_ACTIVE

Components
----------

- :class:`CandidateProvider` -- runtime-checkable protocol for generation backends
- :class:`FailbackState` -- 4-state enum
- :class:`FailbackStateMachine` -- asymmetric failover/failback logic
- :class:`CandidateGenerator` -- orchestration layer with concurrency and deadline
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Mapping, NoReturn, Optional, Protocol, Sequence, Tuple,
    runtime_checkable,
)

from backend.core.ouroboros.governance.op_context import (
    GenerationResult,
    OperationContext,
)
from backend.core.ouroboros.governance.dw_latency_tracker import (
    DwLatencyTracker,
)
# Task T2 -- Autonomous Provider Quarantine Matrix. Top import is cycle-safe:
# provider_quarantine top-level imports only stdlib and lazy-imports its
# downstream deps (convergence_watchdog, intake_dlq), so it never re-enters
# candidate_generator at module-import time. Verified via
# `python3 -c "import ...candidate_generator"`.
from backend.core.ouroboros.governance.provider_quarantine import (
    get_provider_health_gradient,
    quarantine_enabled,
    quarantine_op,
)

# Phase 3c -- Sovereign Provider Failover Lifecycle: seamless DAG re-entry.
# Top import is cycle-safe -- failover_lifecycle imports only stdlib +
# intake_dlq (stdlib leaf) at module-import time; everything else (forecaster,
# quarantine gradient, intake router) is lazy. Verified via
# `python3 -c "import ...candidate_generator"`. Bound as module attributes so
# the seam is monkeypatchable in tests.
from backend.core.ouroboros.governance.failover_lifecycle import (
    get_failover_controller,
    lifecycle_enabled,
)

# LR3 terminal sentinel: the Information-Gain Governor's deadlock-override
# failure (raised inside the Venom tool loop). It MUST propagate through every
# broad-catch failback site below UNRECLASSIFIED so the orchestrator's
# ``except GovernanceDeadlockError`` terminal catch can stamp
# terminal_reason_code="deadlock_override_failed". Top-level import is
# cycle-safe (verified: tool_executor does not import candidate_generator); a
# fallback class keeps every ``except GovernanceDeadlockError`` clause
# evaluable even if the import is severed under an unusual load order
# (mirrors orchestrator.py's defensive fallback class).
try:
    from backend.core.ouroboros.governance.tool_executor import (
        GovernanceDeadlockError,
    )
except Exception:  # noqa: BLE001 -- defensive; keep except-clauses evaluable
    class GovernanceDeadlockError(RuntimeError):  # type: ignore[no-redef]
        """Fallback shim -- real class lives in tool_executor."""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deadline budget allocation — deterministic split (Manifesto §5)
# ---------------------------------------------------------------------------
# The skeleton (deterministic budget) decides how time is partitioned across
# tiers; the nervous system (agentic providers) works within its allocation.
# This prevents any single tier from starving downstream fallbacks.
#
# Tier 0 (DoubleWord batch): gets a capped fraction, MUST leave Tier 1 reserve.
# Tier 1 primary (J-Prime): gets a capped fraction, MUST leave fallback reserve.
# Tier 1 fallback (Claude): gets whatever remains — guaranteed minimum.

_TIER0_BUDGET_FRACTION = float(os.environ.get("OUROBOROS_TIER0_BUDGET_FRACTION", "0.65"))
_TIER0_MAX_WAIT_S = float(os.environ.get("OUROBOROS_TIER0_MAX_WAIT_S", "90"))
_TIER1_MIN_RESERVE_S = float(os.environ.get("OUROBOROS_TIER1_MIN_RESERVE_S", "25"))


# ---------------------------------------------------------------------------
# Slice 238 — cascade-to-dead-Claude guard (layer 8).
#
# The sentinel's ``fallback_tolerance=cascade_to_claude`` path invoked
# ``_call_fallback`` (the Claude lane) with NO breaker consult — so a DW
# transient hiccup poisoned the op via the credit-dead Claude lane
# (terminal_quota). The PRIMARY Claude lane already gates on the economic
# breaker; this makes the cascade read the SAME source-of-truth (the read-only
# ``doubleword_provider._claude_breaker_open`` predicate, no probe side-effect)
# so when Claude is economically/transport OPEN the cascade is suppressed and the
# op routes to the existing immortal DW-retry / clean-degrade branch instead.
# ---------------------------------------------------------------------------


def cascade_breaker_consult_enabled() -> bool:
    """Master switch for the Slice-238 cascade breaker consult. Default TRUE —
    failure-path-only (only changes behavior when the Claude breaker is OPEN,
    which is exactly when cascading to it is wrong); breaker-CLOSED is byte-
    identical to the legacy cascade. Kill switch is pure rollback. NEVER raises."""
    raw = (os.environ.get("JARVIS_CASCADE_BREAKER_CONSULT_ENABLED", "true") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _latency_quarantine_enabled() -> bool:
    """Master for the cold-storage latency quarantine at the DW selector seam.
    Default TRUE — failure-path-only: it only skips a model the TtftObserver has
    flagged as COLD_STORAGE (a real TTFT spike) AND only when another candidate
    remains. When the observer is absent/disabled it short-circuits, so this is a
    free no-op unless there's positive latency evidence. =0 reverts to the legacy
    (entitlement-breaker-only) selection. NEVER raises."""
    raw = (os.environ.get("JARVIS_DW_LATENCY_QUARANTINE_ENABLED", "true") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def autarky_backoff_wait_enabled() -> bool:
    """Master switch for the Sovereign Autarky Backoff-Wait (2026-06-20).

    Default TRUE — failure-path-only + autarky-only: it ONLY changes behavior
    when (a) the sole-provider primary is in transient backoff AND (b) there is
    NO fallback configured (DW-only mode). In every other state (fallback
    present, primary healthy) it is byte-identical. Without it, a transient DW
    TIMEOUT routes a STANDARD op to the absent Claude fallback and fails it with
    ``fallback_skipped`` despite ample remaining budget to wait out the short
    backoff and re-attempt the sole provider. Kill switch = pure rollback to the
    legacy degrade-immediately path. NEVER raises."""
    raw = (os.environ.get("JARVIS_AUTARKY_BACKOFF_WAIT_ENABLED", "true") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _autarky_backoff_max_wait_s() -> float:
    """Hard cap on a SINGLE autarky backoff-wait (don't sleep absurdly long even
    if budget allows). Env-tunable; defensive default 90s. NEVER raises."""
    raw = (os.environ.get("JARVIS_AUTARKY_BACKOFF_MAX_WAIT_S", "") or "").strip()
    try:
        v = float(raw) if raw else 90.0
        return v if v > 0 else 90.0
    except (TypeError, ValueError):
        return 90.0


def _autarky_retry_margin_s() -> float:
    """Budget that MUST remain AFTER the wait to attempt the primary call — so we
    never burn the whole budget sleeping and then have nothing left to generate.
    Env-tunable; defensive default 30s. NEVER raises."""
    raw = (os.environ.get("JARVIS_AUTARKY_RETRY_MARGIN_S", "") or "").strip()
    try:
        v = float(raw) if raw else 30.0
        return v if v > 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


def autarky_should_wait_and_retry(
    *,
    has_fallback: bool,
    enabled: bool,
    eta_s: float,
    remaining_s: float,
    max_wait_s: float,
    margin_s: float,
) -> "Optional[float]":
    """Pure decision: in DW-only autarky, should we WAIT out the sole provider's
    transient backoff and re-attempt the primary (instead of routing to an absent
    fallback)? Returns the bounded wait in seconds when yes, else ``None``.

    Yes IFF: enabled AND no fallback AND a real positive backoff exists AND the
    bounded wait + the post-wait call margin fit inside the remaining budget.
    Pure + total — NEVER raises; trivially unit-testable."""
    if not enabled or has_fallback:
        return None
    try:
        if eta_s <= 0 or remaining_s <= 0:
            return None
        wait = min(float(eta_s), float(max_wait_s))
        if wait <= 0:
            return None
        if wait + float(margin_s) < float(remaining_s):
            return wait
        return None
    except (TypeError, ValueError):
        return None


def should_cascade_to_claude(
    *, has_fallback: bool, claude_breaker_open: bool, enabled: bool,
    route_masked: bool = False,
) -> bool:
    """Pure decision: should the sentinel actually cascade to the Claude fallback
    after DW exhaustion? Cascade ONLY when a fallback is configured AND it is not
    suppressed by an OPEN economic breaker. When *enabled* and the breaker is OPEN
    (Claude known-dead), suppress the cascade (→ caller routes to the immortal
    DW-retry / degrade branch). When *enabled* is False (kill switch), legacy
    behavior: cascade iff a fallback exists. No env / breaker reads here — the
    caller injects both — so this stays deterministic + unit-testable. Pure.

    ``route_masked`` (Pre-emptive Route Masking, 2026-07-18): the caller
    evaluated the COST CONTRACT (cost_contract_assertion.
    classify_route_compatibility — the SAME policy the dispatch-time
    assert enforces) BEFORE building the pool. True = this route may not
    buy Claude (BG/SPEC non-read-only) — the cascade is omitted
    ENTIRELY, so the envelope exhausts its cheap pool natively instead
    of detonating a CostContractViolation mid-dispatch (the violent
    abort whose transport teardown bled onto the cockpit). Checked
    FIRST: a masked route never cascades, breaker state irrelevant."""
    if route_masked:
        return False
    if not has_fallback:
        return False
    if enabled and claude_breaker_open:
        return False  # Claude lane is dead — do not poison the op via it
    return True


def claude_route_masked(context: Any) -> bool:
    """Pre-emptive Route Masking predicate — pure composition over the
    cost contract's OWN classifier (zero duplicated budget rules).
    True iff dispatching *context* to Claude would violate the
    contract (BG/SPEC and not read-only). Consulted when BUILDING the
    fallback pool; the dispatch-time assert stays as defense-in-depth.
    NEVER raises (unknown metadata → not masked, the assert still
    guards)."""
    try:
        from backend.core.ouroboros.governance.cost_contract_assertion import (
            classify_route_compatibility,
            cost_contract_runtime_assert_enabled,
        )
        if not cost_contract_runtime_assert_enabled():
            return False           # contract off → legacy pool shape
        verdict = classify_route_compatibility(
            provider_route=getattr(context, "provider_route", ""),
            provider_tier="claude",
            is_read_only=getattr(context, "is_read_only", False),
        )
        return verdict == "violation"
    except Exception:  # noqa: BLE001
        return False

# Complexity-aware multipliers applied on top of _TIER0_BUDGET_FRACTION.
# Higher complexity => more time for DW 397B code generation.
_TIER0_COMPLEXITY_MULTIPLIER: Dict[str, float] = {
    "trivial": 0.31,           # 0.65 * 0.31 ≈ 0.20 → ~24s DW (one-file edits, RT SSE fast enough)
    "simple": 0.50,            # 0.65 * 0.50 ≈ 0.33 → ~39s DW, ~81s Claude
    "moderate": 1.077,         # 0.65 * 1.077 ≈ 0.70
    "standard": 1.077,         # alias for moderate
    "complex": 1.231,          # 0.65 * 1.231 ≈ 0.80
    "heavy_code": 1.231,       # alias for complex
}
_PRIMARY_BUDGET_FRACTION = float(os.environ.get("OUROBOROS_PRIMARY_BUDGET_FRACTION", "0.65"))
_FALLBACK_MIN_RESERVE_S = float(os.environ.get("OUROBOROS_FALLBACK_MIN_RESERVE_S", "30"))

# Tier 3 Reflex (Manifesto §5): aggressive hard cap on DoubleWord 397B
# across ALL cost-optimized call paths. If DW stalls on stream rendering
# or token generation for longer than this cap, the deterministic router
# severs the thread and cascades to the high-reliability frontier model
# (Claude). Single source of truth for the hard cap; both the Tier-0-first
# path (_compute_tier0_budget) and the Primary-first path
# (_compute_primary_budget) reference it via the aliases below.
#
# Problem this fixes — F1 Slice 4 S3 (bt-2026-04-24-204029) + S4
# (bt-2026-04-24-213248):
#   S3: primary held semaphore for up to 153.76s (DW SSE stream stall),
#       exceeding the then-current fraction-based cap of ~143s.
#   S4: patch landed but didn't fire — DW was promoted to BOTH Tier 0 AND
#       primary (J-Prime unhealthy), which routes via the Tier 0 fast path
#       (_compute_tier0_budget, max_wait=_TIER0_MAX_WAIT_S=90s), NOT via
#       _call_primary where the S3 patch lived. The _PRIMARY_MAX_TIMEOUT_S
#       cap was inert for this configuration because _call_primary was
#       never invoked. Same 153s DW semaphore hold pattern repeated.
#
# Manifesto §5 Tier 3 quote (verbatim):
#   "If a cost-optimized inference node (e.g., DW 397B) exhausts its
#    temporal budget without returning a valid execution plan, the
#    deterministic router autonomously severs the thread and triggers
#    an instant cascade to a high-reliability frontier model."
#
# This cap is a HARD TIME BOX applied at TWO sites:
#   1. _compute_primary_budget — for the "call primary first" path
#      (FSM PRIMARY_READY / PRIMARY_DEGRADED with J-Prime as primary).
#   2. _compute_tier0_budget — for the "Tier 0 fast-path first" (the
#      Manifesto §5 default cascade; DW-as-Tier-0 always tries here).
#
# Fraction + route-specific max_wait logic stays inside each function as
# inner floors; this cap is the strict outer ceiling that enforces the
# reflex regardless of route.
#
# Default 30s is calibrated from S3+S4 evidence: DW first_token_ms=1898
# was observed on a healthy call, but stream stall extended hold to
# 85-153s. A 30s cap forces a sever at any stall beyond the expected
# first-token-plus-generation window, which for docstring-expansion
# workloads should comfortably finish in <20s on a healthy DW endpoint.
#
# Env-tunable so operators can relax for legitimately slow workloads
# (e.g., extremely long CoT traces on architectural ops). The Claude
# fallback still sees its own `_FALLBACK_MAX_TIMEOUT_S` budget after
# the DW path is severed.
_TIER3_REFLEX_HARD_CAP_S = float(
    os.environ.get("OUROBOROS_TIER3_REFLEX_HARD_CAP_S", "30")
)

# ──────────────────────────────────────────────────────────────────────
# Slice 18c (2026-05-26) — route-aware Tier 0 RT budget cap
#
# Closes the cascade-to-Claude-on-premature-timeout pattern surfaced by
# soak bt-2026-05-26-070049 (FLEET v13): Slice 10A correctly routed
# SWE-Bench-Pro to STANDARD; Slice 10B-iii promoted Qwen 397B; Slice
# 10B-ii bridge unblocked the topology; candidate_generator dispatched
# DW Tier 0 RT — but the 30s default cap (above) clamped the budget
# below the 397B's actual TTFT envelope. Result: 8 EXHAUSTION events,
# each cascading to Claude which then refused on credit-balance.
#
# The 30s default was designed for IMMEDIATE-equivalent "reflex"
# semantics (per Manifesto §5 — speed permanently supersedes cost).
# Applying it to STANDARD + COMPLEX routes — which are explicitly
# cost-optimized (DW primary) and have no reflex-time SLA — is a
# category error.
#
# Fix: route-aware cap selector. STANDARD + COMPLEX use the new
# JARVIS_DW_TIER0_RT_BUDGET_S (default 90s — matches Qwen 397B + Kimi
# K2.6 TTFT envelope per §46.2). BG/SPEC + everything else keeps the
# 30s reflex cap (those are either cost-floored or DW-only routes
# where 30s is the right ceiling).
#
# Operator override per route via the env knob; future Slice 13B
# bandit (§45.7.2) can replace this static cap with per-shape
# empirical p95 envelope. Until then, 90s is the empirical floor
# observed on 397B cold-start cold-cache runs.
# ──────────────────────────────────────────────────────────────────────
_TIER0_RT_BUDGET_STANDARD_COMPLEX_S = float(
    os.environ.get("JARVIS_DW_TIER0_RT_BUDGET_S", "90"),
)


# ──────────────────────────────────────────────────────────────────────
# Slice 27 Phase 3 — Context-Aware Adaptive Timeboxing
# ──────────────────────────────────────────────────────────────────────
#
# v20 forensic (bt-2026-05-27-011121): 12 EXHAUSTION events, ALL with
# fsm_failure_mode=TIMEOUT, on a 3-model fleet (Qwen-397B + Qwen-35B +
# Kimi-K2.6). DW was reachable (cost recorded $0.0149) but every
# GENERATE call exceeded the static 90s Tier 0 budget. The model is
# given a fixed budget regardless of how heavy the prompt is or which
# model is processing it — defeating the purpose of having a
# multi-model fleet.
#
# Per operator directive: compute the streaming timeout window
# dynamically at dispatch time from (payload size, model tier).
#
#   base               = 60s
#   +15s per 5000 chars of input payload (step bonus)
#   × 1.5 scalar for heavy reasoning / long-context models
#                        (Qwen3.5-397B-A17B-FP8, Kimi-K2.6)
#   hard cap           = 240s (safe ceiling — no unbounded cost bleed)
#   non STANDARD/COMPLEX routes → preserve legacy 30s reflex cap
#
# Examples (STANDARD/COMPLEX route):
#   0 chars   + 397B  → 60.0 * 1.5  = 90.0s   (matches v18c default)
#   10000     + 397B  → (60+30)*1.5 = 135.0s  (50% more for 10KB SWE prompt)
#   30000     + 397B  → (60+90)*1.5 = 225.0s  (heavy prompt + heavy model)
#   50000     + 397B  → (60+150)*1.5 = 315 → capped 240.0s
#   0         + 35B   → 60.0s       (workhorse — no scalar)
#   10000     + 35B   → (60+30)     = 90.0s
#
# Hardcoding-free: every threshold reads from env at call time so
# operators can tune without code edits. Defaults match the operator's
# spec exactly.

_ADAPTIVE_BASE_S_DEFAULT = 60.0 # Base timeout in seconds for the adaptive formula when prompt_chars is zero. This is the starting point for the timeout calculation before adding the step bonus and applying the heavy model scalar. Default is 60s as per operator spec.
_ADAPTIVE_STEP_CHARS_DEFAULT = 5000 # Number of prompt characters that trigger each step bonus increment. For every multiple of this number of characters in the prompt, the step bonus is added to the base timeout. Default is 5000 chars as per operator spec.
_ADAPTIVE_STEP_BONUS_S_DEFAULT = 15.0 # Additional timeout in seconds added for each step of prompt_chars defined by _ADAPTIVE_STEP_CHARS_DEFAULT. For example, with a step_chars of 5000 and a step_bonus of 15s, a prompt of 10000 chars would add 30s to the base timeout. Default is 15s as per operator spec. 
_ADAPTIVE_HEAVY_SCALAR_DEFAULT = 1.5 # Scalar multiplier applied to the timeout when the model is identified as a heavy model (e.g., Qwen-397B or Kimi-K2.6). This accounts for the longer TTFT of heavy models. Default is 1.5x as per operator spec.
_ADAPTIVE_CAP_S_DEFAULT = 240.0 # Maximum timeout in seconds that can be returned by the adaptive formula, regardless of prompt size or model. This prevents unbounded timeouts for extremely large prompts. Default is 240s as a safe ceiling per operator spec.

# Slice 28 Phase 2 — Adaptive Streaming TTFT Horizon
# Heavy-reasoning / long-context models legitimately need more cold-start
# TTFT runway than the static 30s _PRIMARY_MAX_TIMEOUT_S allows. Scale
# _PRIMARY_MAX_TIMEOUT_S by this factor when the dispatched model is
# heavy (matched via _is_heavy_model — same Qwen-397B / Kimi-K2.6 markers
# Slice 27 Phase 3 uses). Hard ceiling at 240s prevents unbounded cost
# bleed. Per operator directive: base 30s × 2.5 = 75s for heavy models.
# v21 forensic (bt-2026-05-27-025855) showed 12 EXHAUSTION events on
# 397B all at elapsed=30.01s with remaining=329.86s — the static cap
# was the binding constraint, killing primary calls before the streaming
# layer's 120s TTFT could even fire on the wire.
_PRIMARY_HEAVY_TTFT_SCALAR_DEFAULT = 2.5 # Scalar multiplier for heavy models' TTFT horizon. When the dispatched model is identified as heavy (e.g., Qwen-397B or Kimi-K2.6), the primary timeout cap is scaled by this factor to allow for longer cold-start TTFT. This is applied on top of the existing _PRIMARY_MAX_TIMEOUT_S cap, which serves as a base for all models. Default is 2.5x as per operator directive, giving heavy models a 75s cap instead of 30s.
_PRIMARY_HEAVY_TTFT_CAP_S_DEFAULT = 240.0 # Maximum timeout in seconds for heavy models on the primary path. This serves as a hard ceiling to prevent unbounded timeouts even for heavy models. Default is 240s as a safe ceiling per operator directive, ensuring that even with the heavy scalar, the timeout does not exceed this limit.

# Slice 28 Phase 3 — Inline Fault Discriminator probe timeout.
# When the adaptive primary timeout fires on TimeoutError, this
# bounded probe (default 5s) discriminates context-lag vs
# infrastructure-outage. Short by design — the probe MUST NOT
# itself become a wedge.
_TTFT_PROBE_TIMEOUT_S_DEFAULT = 5.0
_TTFT_PROBE_PROMPT = "ping"
_TTFT_PROBE_MAX_TOKENS = 2


def _envb(name: str, default: bool = False) -> bool:
    """Stdlib-only truthy env reader. Lives alongside _envf/_envi helpers."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")

# Heavy-model substring matchers — checked case-insensitively against
# model_id. CSV-extensible via env var so operators can add new heavy
# variants without code edits (a Qwen3.5-512B-MoE release wouldn't need
# a code change to get the 1.5× scalar). Default set codifies operator's
# §46 fleet inventory: the 397B MoE workhorse + Kimi's 200K-context
# specialist (both warrant the heavy budget per §46 strengths).
_HEAVY_MODEL_DEFAULT_MARKERS = ("397B", "Kimi") # Default heavy model markers. 

# Defensive: this function is called on every Tier 0 dispatch, so we read and parse the env var once per call. The parsing logic is robust to empty/malformed env vars, falling back to the default marker set when necessary. The tuple of markers is returned for efficient substring checks in the hot path.
def _heavy_model_markers() -> Tuple[str, ...]:
    """CSV-tunable heavy-model match list. Default: ('397B', 'Kimi')."""
    raw = os.environ.get("JARVIS_ADAPTIVE_HEAVY_MODEL_MARKERS", "").strip() # Read the raw env var value as a string and strip whitespace. If the env var is not set or is empty after stripping, return the default heavy model markers. Defensive: if the raw value is empty, return the default immediately without trying to parse it. This handles both unset and explicitly empty env vars gracefully.
    if not raw: # Defensive: if the raw value is empty, return the default immediately without trying to parse it. This handles both unset and explicitly empty env vars gracefully.
        return _HEAVY_MODEL_DEFAULT_MARKERS # Return the default heavy model markers if the env var is not set or is empty. This ensures that we have a sensible default set of markers to identify heavy models without requiring operator configuration.
    return tuple(m.strip() for m in raw.split(",") if m.strip()) # Split the raw string by commas, strip whitespace from each marker, and return a tuple of non-empty markers. This allows operators to specify a custom list of heavy model markers via the env var, while ensuring that empty entries are ignored. 

# Float env vars are used for time thresholds to allow fractional seconds and to keep the env interface simple. Defensive: negative values are treated as zero.
def _envf_or_default(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip() # Read the raw env var value as a string and strip whitespace. If the env var is not set or is empty after stripping, return the default value.
    if not raw: # Defensive: if the raw value is empty, return the default immediately without trying to parse it. This handles both unset and explicitly empty env vars gracefully.
        return default # Return the default value if the env var is not set or is empty.
    try: # Try to parse the raw string as a float. If parsing fails (e.g., due to invalid format), catch the ValueError and return the default instead.
        return float(raw) # Convert the raw string to a float and return it. This allows for fractional seconds in time thresholds.
    except ValueError: # If the raw value cannot be parsed as a float, return the default. This ensures that invalid env var values don't cause crashes and instead fall back to safe defaults.
        return default # Return the default value if parsing fails due to invalid format.

# Integer env vars are used for char counts to avoid fractional chars and to keep the env interface simple. Defensive: negative values are treated as zero.
def _envi_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip() # Read the raw env var value as a string and strip whitespace. If the env var is not set or is empty after stripping, return the default value.
    if not raw: # Defensive: if the raw value is empty, return the default immediately without trying to parse it. This handles both unset and explicitly empty env vars gracefully.
        return default # Return the default value if the env var is not set or is empty.
    try: # Try to parse the raw string as an integer. If parsing fails (e.g., due to invalid format), catch the ValueError and return the default instead.
        return int(raw) # Convert the raw string to an integer and return it. This is used for char count thresholds where fractional chars don't make sense.
    except ValueError: # If the raw value cannot be parsed as an integer, return the default. This ensures that invalid env var values don't cause crashes and instead fall back to safe defaults.
        return default # Return the default value if parsing fails due to invalid format.

# Slice 84 — param-aware heavy threshold. The marker fast-path ("397B","Kimi")
# only covered two models; Slice 83 then ranked DeepSeek-V4-Pro (1000B) and
# GLM-5.1 (754B) FIRST, but they carried NO marker → got the bare 30s TTFT cap →
# killed at elapsed=30.01s before first token (the v44-v64 "DW down" mirage).
# Any model at/above this parameter count is treated as heavy (deserves the
# longer TTFT runway), so the whole frontier-coder fleet — present AND future —
# qualifies WITHOUT a per-model marker. 100B cleanly separates the 397B+/754B+
# workhorses (+ DeepSeek-V4-Flash 100B) from the cheap Qwen-35B fast-path model.
_HEAVY_MODEL_MIN_PARAMS_B_DEFAULT: float = 100.0


def _heavy_model_min_params_b() -> float:
    """Env-tunable parameter-count floor for param-aware heavy detection."""
    return _envf_or_default(
        "JARVIS_HEAVY_MODEL_MIN_PARAMS_B", _HEAVY_MODEL_MIN_PARAMS_B_DEFAULT,
    )


# Model ID matchers for heavy models. Two paths: (1) the curated/CSV marker
# fast-path, (2) Slice 84 param-aware fallback. Used to apply the heavy-model
# TTFT scalar in the adaptive primary-budget formula.
def _is_heavy_model(model_id: str) -> bool:
    """True iff ``model_id`` warrants the heavy-model TTFT runway.

    A model qualifies if EITHER it matches a curated/CSV marker
    (``397B``/``Kimi``, operator-extensible) OR — Slice 84 — its resolved
    parameter count is at/above ``JARVIS_HEAVY_MODEL_MIN_PARAMS_B`` (default
    100B). The param path reuses Slice 82's catalog resolver (curated map +
    ``\\d+B`` regex), so the strong DW coders (DeepSeek-V4-Pro 1000B, GLM-5.1
    754B, …) auto-qualify with no per-model hardcoding. Fail-soft: an
    unresolvable param count + no marker → not heavy. Pure; never raises."""
    if not model_id:  # Defensive: empty model_id is not heavy.
        return False
    mid_lower = model_id.lower()  # Lowercase once for efficiency.
    # (1) curated / CSV marker fast-path
    if any(m.lower() in mid_lower for m in _heavy_model_markers()):
        return True
    # (2) Slice 84 — param-aware fallback
    try:
        from backend.core.ouroboros.governance.dw_catalog_client import (
            parse_parameter_count,
        )
        pb = parse_parameter_count(model_id)
        if pb is not None and pb >= _heavy_model_min_params_b():
            return True
    except Exception:  # noqa: BLE001 — never block dispatch on a catalog hiccup
        pass
    return False

# Pure function for the adaptive Tier 0 timeout formula. Called by the route-aware cap selector (:func:`_tier0_rt_cap_for_route`) when the caller
def _compute_adaptive_tier0_timeout_s(
    *,
    prompt_chars: int, # Caller-provided prompt size in chars — used to compute the step bonus. Defensive: negative treated as zero.
    model_id: str, # Caller-provided model ID — used to determine if the heavy-model scalar applies. Defensive: empty treated as non-heavy.
    base_s: Optional[float] = None, # Optional override for the base timeout in seconds. If not provided, reads from env var JARVIS_ADAPTIVE_TIER0_BASE_S or defaults to _ADAPTIVE_BASE_S_DEFAULT.
    step_chars: Optional[int] = None, # Optional override for the number of chars per step in the adaptive formula. If not provided, reads from env var JARVIS_ADAPTIVE_TIER0_STEP_CHARS or defaults to _ADAPTIVE_STEP_CHARS_DEFAULT.
    step_bonus_s: Optional[float] = None, # Optional override for the step bonus in seconds. If not provided, reads from env var JARVIS_ADAPTIVE_TIER0_STEP_BONUS_S or defaults to _ADAPTIVE_STEP_BONUS_S_DEFAULT. 
    heavy_scalar: Optional[float] = None, # Optional override for the heavy model scalar. If not provided, reads from env var JARVIS_ADAPTIVE_TIER0_HEAVY_SCALAR or defaults to _ADAPTIVE_HEAVY_SCALAR_DEFAULT.
    cap_s: Optional[float] = None, # Optional override for the maximum timeout cap in seconds. If not provided, reads from env var JARVIS_ADAPTIVE_TIER0_CAP_S or defaults to _ADAPTIVE_CAP_S_DEFAULT.
) -> float:
    """Slice 27 Phase 3 — pure-function adaptive Tier 0 timeout.

    Operator's formula, fully env-tunable:

        timeout = (base + step_bonus × floor(prompt_chars / step_chars))
                  × (heavy_scalar if _is_heavy_model(model_id) else 1.0)
        timeout = min(timeout, cap)

    Caller-provided kwargs win over env defaults; env defaults win over
    code defaults. Pure function — no side effects, deterministic.
    """
    b = base_s if base_s is not None else _envf_or_default(
        "JARVIS_ADAPTIVE_TIER0_BASE_S", _ADAPTIVE_BASE_S_DEFAULT,
    )
    sc = step_chars if step_chars is not None else _envi_or_default(
        "JARVIS_ADAPTIVE_TIER0_STEP_CHARS", _ADAPTIVE_STEP_CHARS_DEFAULT,
    )
    sb = step_bonus_s if step_bonus_s is not None else _envf_or_default(
        "JARVIS_ADAPTIVE_TIER0_STEP_BONUS_S", _ADAPTIVE_STEP_BONUS_S_DEFAULT,
    )
    hs = heavy_scalar if heavy_scalar is not None else _envf_or_default(
        "JARVIS_ADAPTIVE_TIER0_HEAVY_SCALAR", _ADAPTIVE_HEAVY_SCALAR_DEFAULT,
    )
    cap = cap_s if cap_s is not None else _envf_or_default(
        "JARVIS_ADAPTIVE_TIER0_CAP_S", _ADAPTIVE_CAP_S_DEFAULT,
    )

    # Defensive — negative payload chars treated as zero
    safe_chars = max(0, int(prompt_chars or 0)) # Ensure prompt_chars is a non-negative integer. If prompt_chars is None or negative, treat it as zero. This prevents the formula from producing a smaller timeout due to negative char counts.
    steps = safe_chars // max(1, sc)  # avoid div-by-zero on misconfigured env
    timeout = b + sb * steps # Calculate the timeout based on the base, step bonus, and number of steps determined by the prompt size. The step bonus increases the timeout for larger prompts according to the operator's formula.
    if _is_heavy_model(model_id): # Apply the heavy model scalar if the model_id matches any of the heavy model markers. This accounts for the longer TTFT of heavy models.
        timeout *= hs # Scale the timeout by the heavy model scalar if applicable.
    return min(timeout, cap) # Apply the maximum cap to ensure the timeout does not exceed the specified limit, preventing unbounded timeouts for extremely large prompts or heavy models.


def _tier0_rt_cap_for_route(
    provider_route: str,
    *,
    model_id: str = "",
    prompt_chars: int = 0,
) -> float:
    """Tier 0 RT cap — adaptive when model_id/prompt_chars provided,
    legacy 90s/30s wall when not.

    Slice 18c semantics preserved for callers that don't pass the new
    kwargs (byte-identical to pre-Slice-27 behavior):
      STANDARD + COMPLEX → 90s default (env-tunable)
      everything else    → 30s reflex cap

    Slice 27 Phase 3 — when EITHER model_id or prompt_chars is provided,
    the STANDARD/COMPLEX path switches to the adaptive formula
    (:func:`_compute_adaptive_tier0_timeout_s`). The 30s reflex cap for
    other routes is preserved unconditionally (IMMEDIATE/BG/SPEC have
    cost-optimization semantics that should not pay the dispatch-time
    payload-sizing cost).
    """
    r = (provider_route or "").strip().lower()
    if r not in ("standard", "complex"):
        return _TIER3_REFLEX_HARD_CAP_S

    # Slice 27 Phase 3 — adaptive only when caller has context.
    # Legacy callers that pass only the route get the historical
    # 90s static cap (matches Slice 18c byte-identically).
    if not model_id and prompt_chars <= 0:
        return _TIER0_RT_BUDGET_STANDARD_COMPLEX_S
    return _compute_adaptive_tier0_timeout_s(
        prompt_chars=prompt_chars,
        model_id=model_id,
    )


# Legacy alias retained for downstream imports + existing test surface.
# Do not change to a different default without updating the test pins.
# Reads OUROBOROS_PRIMARY_MAX_TIMEOUT_S as a per-primary override — when
# set, wins over the shared Tier 3 cap for the _call_primary path only
# (the _compute_tier0_budget path continues to use _TIER3_REFLEX_HARD_CAP_S).
_PRIMARY_MAX_TIMEOUT_S = float(
    os.environ.get("OUROBOROS_PRIMARY_MAX_TIMEOUT_S", str(_TIER3_REFLEX_HARD_CAP_S))
)

# Minimum time worth attempting a fallback API call.  Below this threshold
# the call will almost certainly timeout before the model finishes; skip it
# and raise immediately to avoid burning network round-trip time.
#
# ONE AUTHORITY — `admission_gate.min_viable_call_s()`.
#
# This used to be an independent `OUROBOROS_MIN_VIABLE_FALLBACK_S` env read
# defaulting to 10s, while the admission gate answered the SAME question
# ("what is the least budget in which a Claude fallback can do useful work")
# with `JARVIS_ADMISSION_MIN_VIABLE_CALL_S`, default 25s. Two floors for one
# decision: the gate sheds first and is strictly tighter, so with the gate at
# its default the 10s constant was dead as a decision boundary — the classic
# "the tighter authority silently becomes the real budget for reasons no log
# explains". It was not merely redundant: the gate is disableable
# (`JARVIS_ADMISSION_GATE_ENABLED=0`), and with it off the 10s floor came back
# to life, so the effective minimum silently changed by 15s depending on an
# unrelated flag.
#
# The gate's number is authoritative because its rationale is the reasoned
# one: 25s is where a single Venom tool round with no thinking budget can
# land; below that we admit ops that time out at the API layer instead of at
# the gate, defeating the gate's purpose. Its clamp floor is 10.0 — exactly
# the old default — so the legacy value survives as the gate's own lower
# bound rather than as a second opinion.
#
# Read per call (not bound at import) so an operator's flip hot-reverts
# without a restart, matching the gate's own discipline.

#: Mirrors `admission_gate.min_viable_call_s()`'s documented default. Used
#: ONLY if that import fails — impossible in a healthy tree, since both
#: modules ship in the same package — and set to the conservative value
#: because admitting a doomed call is the worse failure direction.
_MIN_VIABLE_FALLBACK_FAILSOFT_S: float = 25.0


def _min_viable_fallback_s() -> float:
    """The least budget in which a fallback call is worth launching.

    Delegates to the admission gate so there is exactly one answer. NEVER
    raises.
    """
    try:
        from backend.core.ouroboros.governance.admission_gate import (
            min_viable_call_s,
        )
        return float(min_viable_call_s())
    except Exception:  # noqa: BLE001
        return _MIN_VIABLE_FALLBACK_FAILSOFT_S


def _warn_legacy_min_viable_env_once() -> None:
    """Tell an operator whose `OUROBOROS_MIN_VIABLE_FALLBACK_S` no longer does
    anything. Silently ignoring it would be the same class of lie this whole
    reconciliation removes — a number that looks like a control and is not.
    NEVER raises."""
    try:
        raw = (os.environ.get("OUROBOROS_MIN_VIABLE_FALLBACK_S", "") or "").strip()
        if raw:
            logger.warning(
                "[CandidateGenerator] OUROBOROS_MIN_VIABLE_FALLBACK_S=%s is "
                "SUPERSEDED and ignored — the minimum viable fallback budget "
                "is now JARVIS_ADMISSION_MIN_VIABLE_CALL_S (currently %.1fs), "
                "so the gate and the retry loop cannot disagree.",
                raw, _min_viable_fallback_s(),
            )
    except Exception:  # noqa: BLE001
        pass


_warn_legacy_min_viable_env_once()

# Guaranteed minimum window for the fallback (Claude) regardless of how much
# parent-deadline budget Tier 0 consumed before failing. When the parent
# deadline is depleted (e.g. DW timed out after 80s of a 120s window),
# `_call_fallback` REFRESHES its own deadline so Claude gets at least this
# many seconds — otherwise legitimate doc-gen / patch streams (60-100s)
# get cut off mid-flight and the whole op fails to `all_providers_exhausted`.
# Diagnosed in bt-2026-04-11-211131 (24x exhaustion, 0 commits).
# This OVERRIDES the parent wall-clock deadline; the orchestrator's outer
# `wait_for(_gen_timeout + _OUTER_GATE_GRACE_S)` is the absolute Iron Gate.
_FALLBACK_MIN_GUARANTEED_S = float(
    os.environ.get("OUROBOROS_FALLBACK_MIN_GUARANTEED_S", "90"),
)

# Tier 3 reflex cap for PLAN phase (item B from F1 Slice 4 S5 triage,
# bt-2026-04-24-220418). PLAN is soft-fail — callers (PlanGenerator) catch
# exceptions and fall through to GENERATE without plan, so an aggressive cap
# is even more appropriate here than at GENERATE. Two surfaces:
#   (a) primary path reuses the same `_TIER3_REFLEX_HARD_CAP_S` as the
#       GENERATE Tier-0 budget (default 30s) — see plan() below.
#   (b) fallback (Claude) path uses this PLAN-specific override (default 60s,
#       half the GENERATE fallback cap) because PLAN's structured plan.1 JSON
#       is short — Claude doesn't need the full 120s reserve.
# S5 surfaced the gap: CandidateGenerator.plan() at line ~2244 was passing
# raw `remaining` (≈parent deadline) to wait_for, so DW could stall up to
# 90s before failing and Claude could stall up to 120s, total 210s — eating
# the entire BG worker pool ceiling (360s) before GENERATE got to run.
_PLAN_FALLBACK_MAX_TIMEOUT_S = float(
    os.environ.get("OUROBOROS_PLAN_FALLBACK_MAX_TIMEOUT_S", "60"),
)

# ---------------------------------------------------------------------------
# Outer-retry budget (rooted-problem fix 2026-04-25)
# ---------------------------------------------------------------------------
#
# F1 Slice 4 cadence S1b (`bt-2026-04-25-054256`) surfaced the rooted
# problem behind W3(6) Slice 5b's `live_reachability=blocked_by_provider_exhaustion`:
#
#   * `_call_fallback` invokes the provider ONCE.
#   * The provider's internal `_call_with_backoff` does ~3 attempts with
#     exponential 2s/4s backoff, recycling the httpx pool between attempts.
#   * When TCP connect or stream-read fails (anyio cancel scope fires
#     before the API even responds), all 3 internal attempts can exhaust
#     in ~70-80s.
#   * `_call_fallback` then catches the propagated CancelledError and
#     fires `EXHAUSTION cause=fallback_failed` — even when 100+s of
#     parent budget remains.
#
# The budget JARVIS authorized at ROUTE goes unused. Network conditions
# may have recovered by the time those retries would have fired.
#
# Operator binding 2026-04-25 (Option B closure of S1b):
#   "Will not mask provider latency by modifying the seed (Option C) or
#    artificially inflating the timeout boundaries (Option D). The
#    internal architecture is mathematically sound."
#
# This fix adds NO new budget — it just CONSUMES the budget already
# authorized. Outer retry loop re-invokes the provider (head-of-queue
# preserved by holding `_fallback_sem`) on transient failures while
# remaining budget exceeds `_min_viable_fallback_s()` and the failure
# mode is in `_FALLBACK_TRANSIENT_MODES`. Cooperative cancel via
# `OperationCancelledError` (W3(7) cancel-token) is honored immediately
# — never retried.
# Read per call rather than bound at import, for the same reason the
# minimum-viable floor is: an import-bound env constant cannot be changed
# without re-executing the module, and `importlib.reload` is not a local
# operation. It replaces every class object the module owns while leaving the
# `sys.modules` key intact, so any module that already did
# ``from ... import FailureMode`` keeps a stale class and every subsequent
# ``mode is FailureMode.X`` silently becomes False — two enums with identical
# reprs, which is about as confusing as a failure gets.
#
# `test_outer_retry_cap_bounds_attempts` reloaded this module for exactly that
# reason and never restored it, so it poisoned every later test file in the
# same session. Per-call reads remove the need.
def _fallback_outer_retry_max() -> int:
    """Outer-retry attempt cap. NEVER raises."""
    try:
        return max(1, int(os.environ.get("JARVIS_FALLBACK_OUTER_RETRY_MAX", "3")))
    except (TypeError, ValueError):
        return 3


def _fallback_outer_retry_backoff_s() -> float:
    """Backoff between outer-retry attempts, seconds. NEVER raises."""
    try:
        return max(0.0, float(
            os.environ.get("JARVIS_FALLBACK_OUTER_RETRY_BACKOFF_S", "1.0")))
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Slice 12N — ProviderRoute → CircuitTripOrigin mapping
# ---------------------------------------------------------------------------
#
# Blast-radius isolation: only FOREGROUND-origin per-op breakers
# escalate structural trips to the global session_exhausted
# threshold. Background / speculative ops get their own per-op
# breaker but their structural trips are ISOLATED — they cannot
# assassinate a healthy in-flight foreground op (the wedge that
# killed the SWE-Bench-Pro fixture in bt-2026-05-23-015723).
#
# Lookup is by lowercased provider_route string so this map stays
# robust to either Enum-as-value or bare-string population of
# ``context.provider_route``. Unknown / empty routes default to
# FOREGROUND at the call site (safer — preserves legacy escalation).
#
# Lazy import inside the dict construction is unavoidable here
# because circuit_breaker would otherwise cycle through
# candidate_generator at module load. Resolved once at module
# import time.
def _slice12n_build_route_origin_map() -> Dict[str, Any]:
    from backend.core.ouroboros.governance.circuit_breaker import (
        CircuitTripOrigin,
    )
    return {
        "immediate":   CircuitTripOrigin.FOREGROUND,
        "standard":    CircuitTripOrigin.FOREGROUND,
        "complex":     CircuitTripOrigin.FOREGROUND,
        "background":  CircuitTripOrigin.BACKGROUND,
        "speculative": CircuitTripOrigin.SPECULATIVE,
    }


_SLICE12N_ROUTE_TO_ORIGIN: Dict[str, Any] = _slice12n_build_route_origin_map()

# Anthropic resilience pack 2026-04-25 — failure-rate-aware outer-retry.
# When the FailbackStateMachine has logged transient failures recently
# (a window of consecutive_failures > 0 within the past few cycles), bump
# the outer-retry cap from `_FALLBACK_OUTER_RETRY_MAX` to
# `_FALLBACK_OUTER_RETRY_MAX_DEGRADED` for that op only. Healthy ops
# never pay the extra retry cost.
#
# Observed live in F1 Slice 4 S4b: 6 Claude transient failures + 8 pool
# recycles in 30min. The seed's 1 outer-retry attempt wasn't enough to
# survive the full anthropic_transport instability window. Bumping to 5
# attempts during instability gives ~3× more headroom to catch a
# recovery window.
#
# Default = 5 (vs base 3). Set via JARVIS_FALLBACK_OUTER_RETRY_MAX_DEGRADED.
# Master-off via DEGRADED == base (no extra retries even when degraded).
_FALLBACK_OUTER_RETRY_MAX_DEGRADED = int(
    os.environ.get(
        "JARVIS_FALLBACK_OUTER_RETRY_MAX_DEGRADED", "5",
    )
)

# ---------------------------------------------------------------------------
# Nervous System Reflex — BACKGROUND cascade for read-only ops
# ---------------------------------------------------------------------------
#
# Manifesto §5: "Intelligence-driven routing", but survival and execution
# speed permanently supersede cost optimization. For read-only ops on the
# BACKGROUND route we sever the DW thread on a strict temporal budget and
# cascade to Claude — regardless of topology skip_and_queue flags or the
# JARVIS_BACKGROUND_ALLOW_FALLBACK gate that gates the same reflex for
# mutating BG ops.
#
# Cost safety is preserved by the upstream is_read_only contract:
#   * policy Rule 0d refuses every mutating tool under is_read_only=True
#   * orchestrator short-circuits APPLY on read-only ops
# so a Claude cascade under is_read_only carries no write risk; it only
# loses cost optimality, which the Nervous System Reflex explicitly
# trades against lockup avoidance.
_BG_READONLY_DW_STALL_BUDGET_S = float(
    os.environ.get("JARVIS_BG_DW_STALL_BUDGET_S", "60"),
)


def _attribute_cancel(
    exc: BaseException,
    *,
    label: str,
    op_id: str,
    elapsed_s: float,
    remaining_s: float,
) -> str:
    """Best-effort cancel-source attribution for telemetry.

    Pure-observation helper added 2026-04-24 (post-S6 / bt-2026-04-24-225137)
    to disambiguate three cancel classes seen in F1 Slice 4 graduation:

    - **A** — `_FALLBACK_MAX_TIMEOUT_S=120s` per-call cap (`TimeoutError`).
    - **B** — ToolLoop per-round budget (`TimeoutError` at the per-round mark).
    - **C** — external cooperative cancel (`CancelledError` with non-zero
      remaining budget) — sibling-task cancel / retry-harness deadline /
      mid-flight TopologyBlock reroute.

    Walks `asyncio.current_task()` to capture this task's `cancelling()`
    counter (>0 means we were cancelled by an outer task; ==0 means we
    timed out from our own `wait_for`). Walks `asyncio.all_tasks()` to
    surface a likely-canceller name (best-effort; no guarantee).

    Returns a single-line structured string suitable for logging.
    Never raises — attribution failure is logged as `attribution_error=...`.
    """
    err_class = type(exc).__name__

    def _safe_cancelling(task: Any) -> int:
        # Task.cancelling() is Python 3.11+. We target 3.9+, so always go
        # through getattr/lambda to keep typecheckers + 3.9 runtime happy.
        fn = getattr(task, "cancelling", None)
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:
            return 0

    try:
        current = asyncio.current_task()
        own_cancelling = _safe_cancelling(current)
        # Best-effort canceller search: any other live task with cancelling()>0
        # is a candidate. Walks at most 64 tasks to bound cost.
        canceller = "unknown"
        try:
            for t in list(asyncio.all_tasks())[:64]:
                if t is current:
                    continue
                if _safe_cancelling(t) > 0:
                    canceller = t.get_name()
                    break
        except RuntimeError:
            canceller = "no_running_loop"
        # Heuristic class assignment:
        #   - TimeoutError + own_cancelling==0 + remaining≈0 → Class A/B (own deadline)
        #   - CancelledError + own_cancelling>0              → Class C (external)
        #   - CancelledError + own_cancelling==0             → ambiguous (loop teardown?)
        if isinstance(exc, asyncio.TimeoutError):
            klass = "A_or_B_timeout"
        elif isinstance(exc, asyncio.CancelledError) and own_cancelling > 0:
            klass = "C_external_cancel"
        elif isinstance(exc, asyncio.CancelledError):
            klass = "C_ambiguous"
        else:
            klass = "non_cancel"
        return (
            f"label={label} op={op_id[:16]} class={klass} "
            f"err={err_class} elapsed={elapsed_s:.2f}s "
            f"remaining={remaining_s:.2f}s "
            f"own_cancelling={own_cancelling} "
            f"canceller_task={canceller}"
        )
    except Exception as e:
        return (
            f"label={label} op={op_id[:16]} class=attribution_failed "
            f"err={err_class} elapsed={elapsed_s:.2f}s "
            f"remaining={remaining_s:.2f}s "
            f"attribution_error={type(e).__name__}"
        )

# ---------------------------------------------------------------------------
# Route-scoped Claude fallback disable (isolation harnesses)
# ---------------------------------------------------------------------------
#
# ``JARVIS_DISABLE_CLAUDE_FALLBACK_ROUTES`` accepts a comma-separated list of
# route names. Any op whose ``provider_route`` matches will skip the Claude
# fallback entirely when Tier 0 fails, raising a clean
# ``fallback_disabled_by_env:{route}`` sentinel through the existing
# exhaustion path. Used by the Qwen 397B isolation benchmark to collect raw
# DW completion telemetry without Claude masking failures or burning tokens.
# Default unset → normal cascade behavior.
_DISABLE_FALLBACK_ROUTES_ENV = "JARVIS_DISABLE_CLAUDE_FALLBACK_ROUTES"


def _fallback_disabled_for_route(route: str) -> bool:
    raw = os.environ.get(_DISABLE_FALLBACK_ROUTES_ENV, "").strip()
    if not raw:
        return False
    disabled = {r.strip().lower() for r in raw.split(",") if r.strip()}
    return (route or "").strip().lower() in disabled


# ──────────────────────────────────────────────────────────────────────
# Slice 23 — Autonomous Registry-Driven Sentinel Activation
# ──────────────────────────────────────────────────────────────────────
#
# v16/v17 forensic exposed that locking dispatch to a single DW model
# when an entire trusted-seed fleet sits in the PromotionLedger is an
# architectural bottleneck. The fix is NOT a per-soak env flag — it is
# a structural decision the dispatcher makes at every call from the
# active registry state.
#
# Decision matrix (first-match-wins; closed and deterministic):
#
#   1. Operator explicit-on  (JARVIS_TOPOLOGY_SENTINEL_ENABLED=true)
#      → ACTIVATE (legacy explicit-on contract, preserved verbatim).
#
#   2. Operator explicit-off (JARVIS_TOPOLOGY_SENTINEL_ENABLED=false)
#      → DO NOT activate (operator rollback wins over every structural
#        condition — single-knob hot-revert preserved per §33).
#
#   3. Claude tier structurally absent  (JARVIS_PROVIDER_CLAUDE_DISABLED=true)
#      → ACTIVATE. Slice 19a declares "Claude removed → DW fleet IS
#        the only intelligence". Iterating the fleet is the architectural
#        contract that operator-binding implies. Composes with Slice 22
#        tier-decay (IMMEDIATE→STANDARD demotion when Claude absent).
#
#   4. Multi-model trusted fleet for this route  (≥2 promoted ledger
#      entries that pass the route's eligibility gate)
#      → ACTIVATE. A multi-model fleet exists precisely so dispatch can
#        rotate among them on failure. Locking to one when 2+ are
#        promoted defeats the PromotionLedger's purpose.
#
#   5. Default  (Claude enabled + single-model fleet + env unset)
#      → DO NOT activate. Phase 10 graduation contract preserved for
#        the Claude-enabled posture this contract was written about.
#
# The structural conditions (3, 4) compose `JARVIS_PROVIDER_CLAUDE_DISABLED`
# (Slice 19a) and `_trusted_seed_dw_models_for_route` (Slice 10B-ii) —
# both already-existing substrate. No new env knobs, no new state,
# no parallel ledgers. The PromotionLedger is the autonomous registry;
# the trusted-seed bridge already enforces per-route eligibility gates.
#
# The Phase 10 graduation contract AST pin
# (`phase10_graduation_contract.py`) asserts the master flag DEFAULT
# stays false — which it does. Slice 23 adds structural OVERRIDES on
# top of that default; the literal env-var default is unchanged.


_SENTINEL_ENABLED_ENV = "JARVIS_TOPOLOGY_SENTINEL_ENABLED"
_CLAUDE_DISABLED_ENV = "JARVIS_PROVIDER_CLAUDE_DISABLED"
_SLICE23_MIN_PROMOTED_FOR_AUTO = 2


def _claude_config_disabled() -> bool:
    """True when the Claude fallback tier is STRUCTURALLY disabled via
    ``JARVIS_PROVIDER_CLAUDE_DISABLED`` — the deadest possible fallback (the
    provider is never even constructed), distinct from a tripped circuit breaker.

    DW-autarky's full-runway grant (Slice 225) keys off ``_claude_breaker_open``,
    which reads only the breaker STATE — and a config-disabled Claude never trips
    the breaker, so it stays CLOSED and autarky NEVER engaged under
    ``JARVIS_PROVIDER_CLAUDE_DISABLED=true``. The sole-lane DW was then held to
    the 90s reflex cap and TIMED OUT on slow hosts (live container soak,
    2026-06-20). A config-disabled Claude is *more* dead than a breaker-open one;
    this predicate lets the autarky path treat it as such. Reuses the existing
    ``_CLAUDE_DISABLED_ENV`` constant (no new flag). NEVER raises."""
    try:
        return os.environ.get(_CLAUDE_DISABLED_ENV, "").strip().lower() in (
            "1", "true", "yes", "on",
        )
    except Exception:  # noqa: BLE001 — fail-closed to legacy cascade
        return False


def _slice23_should_activate_sentinel(provider_route: str) -> Tuple[bool, str]:
    """Slice 23 — autonomous registry-driven sentinel activation.

    Returns ``(activate, reason)`` where ``reason`` is a short
    classifier string suitable for logging (one of: ``env_explicit_on``,
    ``env_explicit_off``, ``claude_disabled``, ``multi_model_fleet``,
    ``default_off_phase10_contract``, ``trusted_seed_probe_failed``).

    Pure function over env + PromotionLedger snapshot. No side effects.
    Defensive against trusted-seed probe failures — falls through to
    default-off rather than raising into dispatch.
    """
    env_raw = os.environ.get(_SENTINEL_ENABLED_ENV, "").strip().lower()
    if env_raw in ("1", "true", "yes", "on"):
        return True, "env_explicit_on"
    if env_raw in ("0", "false", "no", "off"):
        return False, "env_explicit_off"

    claude_raw = os.environ.get(_CLAUDE_DISABLED_ENV, "").strip().lower()
    if claude_raw in ("1", "true", "yes", "on"):
        return True, "claude_disabled"

    # Multi-model fleet probe — lazy import keeps candidate_generator
    # bootable when provider_topology is unavailable (e.g., isolated
    # unit tests). Defensive try/except — bridge failure must NEVER
    # block dispatch; fall through to default-off if the probe raises.
    try:
        from backend.core.ouroboros.governance.provider_topology import (
            _trusted_seed_dw_models_for_route,
        )
        promoted_for_route = _trusted_seed_dw_models_for_route(
            provider_route or "standard",
        )
        if len(promoted_for_route) >= _SLICE23_MIN_PROMOTED_FOR_AUTO:
            return True, "multi_model_fleet"
    except Exception:  # noqa: BLE001 — defensive probe
        return False, "trusted_seed_probe_failed"

    return False, "default_off_phase10_contract"


def gen_call_likely_thinking(route: str, task_complexity: str) -> bool:
    """SINGLE SOURCE OF TRUTH: will this generation call have extended
    thinking enabled (per ``providers._resolve_thinking_budget``)?

    Conservative superset (matches the historical inline rule in
    ``_call_fallback``): any non-trivial ``task_complexity`` on a
    non-reflex (non-IMMEDIATE) route. IMMEDIATE intentionally skips
    thinking (reflex path).

    Phase R1 (soak bt-2026-05-18-015317): the INNER fallback widens
    its cap to ``fallback_thinking_cap_s()`` for thinking-likely
    calls; the OUTER Iron-Gate ``_gen_timeout`` (generate_runner /
    orchestrator) MUST floor to the SAME cap or it kills GENERATE at
    240+15s before the inner 360s window completes (CancelledError@
    255s, psf never generated). Both inner and outer consume THIS
    function so the invariant `outer >= inner` holds by construction
    — no duplicated predicate, no per-path drift.
    """
    _tc = (task_complexity or "").strip().lower()
    _r = (route or "").strip().lower()
    return _tc not in ("", "trivial") and _r not in ("immediate",)


def fallback_thinking_cap_s() -> float:
    """The thinking-enabled timeout cap (env-tunable, default 360s).
    Single resolver shared by the inner fallback cap and the outer
    Iron-Gate ``_gen_timeout`` floor (Phase R1 coherence invariant)."""
    try:
        return float(os.environ.get(
            "JARVIS_FALLBACK_MAX_TIMEOUT_THINKING_S", "360.0",
        ))
    except (TypeError, ValueError):
        return 360.0


def force_batch_gen_timeout_floor_s() -> float:
    """Slice 50 Phase 2 — minimum GENERATE-phase deadline for a force-batch op.

    The DW BATCH lane's async poll legitimately runs up to
    ``JARVIS_DW_BATCH_TIMEOUT_S`` (Slice 43, default 300s). The OUTER
    GENERATE deadline must STRICTLY exceed that lease so the batch poll is
    never severed by the outer ``wait_for`` at exactly its own expiry —
    add a small overhead (``JARVIS_FORCE_BATCH_GEN_OVERHEAD_S``, default
    30s) for the sentinel + Iron-Gate processing that follows the poll.

    Derived from the Slice 43 batch-timeout constant, NOT a second
    hardcoded value: change ``JARVIS_DW_BATCH_TIMEOUT_S`` and the floor
    tracks it. Mirror of :func:`fallback_thinking_cap_s` — a single shared
    resolver for the outer/inner deadline-coherence invariant.
    """
    batch_cap = _envf_or_default("JARVIS_DW_BATCH_TIMEOUT_S", 300.0)
    overhead = _envf_or_default("JARVIS_FORCE_BATCH_GEN_OVERHEAD_S", 30.0)
    return batch_cap + overhead


# ---------------------------------------------------------------------------
# Sovereign Infinite-Horizon Batch Matrix (2026-06-20)
# ---------------------------------------------------------------------------
# A PARKED ASYNC_BATCH_PAYLOAD op has had its worker slot freed — it costs ZERO
# CPU while DoubleWord's batch queue churns. Severing it at the 300s/330s legacy
# budget while DW is ACTIVELY processing the batch (validating / in_progress /
# finalizing) is pure waste: the live wedge is mode=TIMEOUT "Batch retrieval
# failed" (NOT a 403 — the retrieval HTTP path never returned >=300; the batch
# simply hadn't finished). The poll layer (_adaptive_poll_batch) is ALREADY
# lifecycle-aware — it returns ONLY on `completed` or terminal `failed/expired/
# cancelled`, and otherwise keeps polling up to DOUBLEWORD_MAX_WAIT_S. The only
# thing cutting it short is the OUTER budget. So: when (and ONLY when) the parked
# out-of-pool continuation is running a batch-bound op, lift the force-batch cap
# to the batch SLA horizon. Confined to the continuation via an async-task-local
# ContextVar so an IN-pool dispatch can NEVER inherit the long budget and wedge a
# live worker. Bounded by the session wall-clock cap in soaks.
import contextvars as _ctxvars  # noqa: E402

_PARKED_BATCH_HORIZON: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "jarvis_parked_batch_horizon_active", default=False,
)


def batch_sla_horizon_s() -> float:
    """The async-batch SLA horizon in seconds — how long a PARKED batch op may
    wait for DW while the worker slot is free. Default mirrors the poll horizon
    (``DOUBLEWORD_MAX_WAIT_S``, 3600s). Clamped [300s, 24h]. Env override:
    ``JARVIS_DW_BATCH_SLA_HORIZON_S``. NEVER raises."""
    default = _envf_or_default("DOUBLEWORD_MAX_WAIT_S", 3600.0)
    raw = _envf_or_default("JARVIS_DW_BATCH_SLA_HORIZON_S", default)
    return max(300.0, min(raw, 24 * 3600.0))


def set_parked_batch_horizon(active: bool):
    """Mark the current async task as a parked-batch continuation so
    ``_compute_primary_budget`` lifts the force-batch cap to the SLA horizon.
    Returns the ContextVar token; reset it in a finally. NEVER raises."""
    try:
        return _PARKED_BATCH_HORIZON.set(bool(active))
    except Exception:  # noqa: BLE001
        return None


def reset_parked_batch_horizon(token) -> None:
    """Reset the parked-batch-horizon ContextVar (paired with set_). NEVER raises."""
    try:
        if token is not None:
            _PARKED_BATCH_HORIZON.reset(token)
    except Exception:  # noqa: BLE001
        pass


def _parked_batch_horizon_active() -> bool:
    try:
        return bool(_PARKED_BATCH_HORIZON.get())
    except Exception:  # noqa: BLE001
        return False


def apply_force_batch_deadline_floor(
    gen_timeout_s: float, *, force_batch: bool
) -> float:
    """Floor a GENERATE-phase deadline so a force-batch op's outer window
    exceeds the DW batch lease (Slice 50 Phase 2).

    Forensic basis — v45 probe ``bt-2026-06-01-034745``: a
    ``route=standard, complexity=trivial`` op force-batched (Slice 36:
    Claude disabled + standard route) but its route-base GENERATE deadline
    was only ``JARVIS_GEN_TIMEOUT_STANDARD_S=220s`` — the R1 thinking-cap
    floor (-> 360s) does not fire for trivial ops. So
    ``_compute_primary_budget(remaining=220, force_batch=True) =
    min(220, 300) = 220`` and the async batch poll was severed at 220s
    while its own 300s lease still had runway (TimeoutError elapsed=220s).

    ``force_batch=False`` ops pass through unchanged (zero regression).
    The floor is a ``max()`` so an already-wide window (e.g. COMPLEX
    R1-floored to 360s) is preserved, never reduced. Safe by construction:
    Slice 36 force-batch only engages when Claude is disabled (pure-DW
    mode), so there is no Claude-cascade calibration to regress.
    """
    if not force_batch:
        return gen_timeout_s
    return max(gen_timeout_s, force_batch_gen_timeout_floor_s())


def structural_fast_cascade_enabled() -> bool:
    """Slice 73 master flag — default TRUE. When off, the dispatch loop tries
    every ranked DW model before cascading (byte-identical legacy behavior)."""
    raw = os.environ.get(
        "JARVIS_DW_STRUCTURAL_FAST_CASCADE_ENABLED", "true",
    ).strip().lower()
    return raw not in ("0", "false", "no", "off")


def should_sever_dw_lane(failure_source: Any) -> bool:
    """Slice 73 — True iff this failure is a STRUCTURAL transport break.

    A ``LIVE_TRANSPORT`` failure (socket/connection break, ``live_transport:
    RuntimeError``) means the transport to the DW endpoint is down — every
    ranked sibling model shares that dead transport, so trying the next one
    just burns another ~30s before the inevitable cascade. Sever the lane and
    hand Claude the full remaining budget.

    Model-SPECIFIC failures (429 rate-limit, 5xx, parse) are NOT severed — a
    sibling model may be healthy, so the loop still rotates to it. Pure;
    never raises (unknown source → don't sever).
    """
    try:
        from backend.core.ouroboros.governance.topology_sentinel import (
            FailureSource,
        )
        return failure_source is FailureSource.LIVE_TRANSPORT
    except Exception:  # noqa: BLE001 — never block dispatch
        return False


def _live_transport_sever_threshold() -> int:
    """Slice 83 Phase 2 — consecutive LIVE_TRANSPORT failures required before
    the whole DW lane is severed (Slice 73 behavior).

    Slice 73 severed the lane on the FIRST ``live_transport`` failure on the
    theory that all ranked siblings share one dead transport. But Slice 82/83
    made the ranked stack HETEROGENEOUS — DeepSeek-V4-Pro, Kimi-K2.6, GLM-5.1,
    Qwen397B, Qwen35B are distinct served endpoints. One model being briefly
    unavailable (deploy bounce, per-model 5xx surfacing as a transport break)
    is NOT a lane outage: the next coder may be perfectly healthy. So we now
    ROTATE to the next model on a single failure and only sever once
    ``threshold`` consecutive models have all failed with LIVE_TRANSPORT — the
    signature of a genuine endpoint-wide blackout. A success (or a non-transport
    failure on a reachable model) resets the streak. Default 3; floored at 1 so
    ``=1`` reproduces exact Slice 73 first-failure sever. Env-tunable."""
    try:
        raw = os.environ.get("JARVIS_DW_LIVE_TRANSPORT_SEVER_THRESHOLD", "3")
        return max(1, int(str(raw).strip()))
    except Exception:  # noqa: BLE001 — bad value → safe default
        return 3


def _note_dw_total_outage(diagnostic: str) -> None:
    """Slice 53 — record one GENERATE op that exhausted ALL DW models with no
    candidate from streaming OR batch (the total-vendor-blackout signature).

    Routed through the dual-lane breaker singleton. NEVER raises (defensive
    lazy import) — recording is best-effort observability + breaker state, it
    must not perturb the generation error path it sits on.
    """
    try:
        from backend.core.ouroboros.governance.dual_lane_breaker import (
            get_dual_lane_breaker,
        )
        get_dual_lane_breaker().record_total_outage(diagnostic or "all_models_open")
    except Exception:  # noqa: BLE001 — never perturb the error path
        pass


def _note_dw_candidate_success() -> None:
    """Slice 53 — record that some DW lane (or fallback) yielded a candidate,
    resetting the breaker's consecutive-outage counter. Preserves Slice 41
    single-lane resilience. NEVER raises."""
    try:
        from backend.core.ouroboros.governance.dual_lane_breaker import (
            get_dual_lane_breaker,
        )
        get_dual_lane_breaker().record_success()
    except Exception:  # noqa: BLE001
        pass
    # Slice 127 P3 — a DW completion succeeded → reset the dynamic-recovery
    # episode counter to 0 instantly so the next transient blip recovers at
    # ``base`` (gated, best-effort; never perturbs the success path).
    try:
        from backend.core.ouroboros.governance.dw_transport_recovery import (
            dw_dynamic_recovery_enabled as _s127_dyn_on,
            get_dw_transport_recovery as _s127_dwr,
        )
        if _s127_dyn_on():
            _s127_dwr().note_recovered()
    except Exception:  # noqa: BLE001
        pass


def _note_dw_live_transport_degraded(diagnostic: str = "", model_id: str = "") -> None:
    """Slice 77 — the millisecond a LIVE dispatch hits a transport break
    (``live_transport:RuntimeError`` / socket drop), stamp the
    ``dw_surface_health`` ledger ``DIRECT_STREAMING → TRANSPORT_DEGRADED`` so
    the NEXT op's Slice 76 P2 pre-flight gate (:func:`dw_transport_degraded_preflight`)
    fires and cascades straight to Claude with the full budget — instead of
    burning the next op's allowance on the same dead transport (the EVAL-2
    Phase-4 ``deadline_exhausted_pre_fallback`` failure, PRD §50.11).

    This converts the ledger from a one-shot BOOT probe into a live,
    event-driven status map. Recovery is automatic: once live generations stop
    failing, no further degraded records are written and the gate's freshness
    window lapses the stale verdict, re-enabling the DW lane. A fresh ledger
    instance per call reads-latest-then-saves, so this never clobbers a
    concurrent probe's record for another surface. NEVER raises (best-effort
    observability must not perturb the dispatch error path it sits on)."""
    try:
        from backend.core.ouroboros.governance.dw_surface_health import (
            SurfaceHealthLedger,
            SurfaceKind,
            SurfaceVerdict,
        )
        SurfaceHealthLedger(autosave=True).record(
            SurfaceKind.DIRECT_STREAMING,
            SurfaceVerdict.TRANSPORT_DEGRADED,
            diagnostic=(diagnostic or "live_transport")[:120],
        )
    except Exception:  # noqa: BLE001 — never perturb the dispatch error path
        pass
    # Slice 127 P3 — register a dynamic-recovery rupture episode (debounced by
    # ``base`` so a burst inside one outage = ONE episode). The dynamic window
    # grows the next probe interval exponentially for a chronically-rupturing
    # lane (gated, best-effort; never perturbs the dispatch error path).
    try:
        from backend.core.ouroboros.governance.dw_transport_recovery import (
            dw_dynamic_recovery_enabled as _s127_dyn_on,
            get_dw_transport_recovery as _s127_dwr,
        )
        if _s127_dyn_on():
            _s127_dwr().note_degraded()
    except Exception:  # noqa: BLE001 — never perturb the dispatch error path
        pass
    # Slice 172 — feed the predictive cortex the SAME rupture event (its own bounded
    # timestamp ring drives the recency-weighted Poisson forecast). Fire-and-forget,
    # lock-guarded append; never perturbs the dispatch error path. Record is
    # UNCONDITIONAL (the master flag gates *routing*, not data collection — so the
    # forecast is already warm the moment predictive routing is switched on).
    try:
        from backend.core.ouroboros.governance.dw_failure_predictor import (
            get_dw_failure_predictor as _s172_pred,
        )
        _s172_pred().record_rupture(model_id=model_id)  # Slice 175 — per-model ring
    except Exception:  # noqa: BLE001 — never perturb the dispatch error path
        pass


def _record_quota_outage_safely(provider: str, reason: str) -> None:
    """Flip the liquidity ledger's quota-outage state for *provider*
    (2026-07-21 council finding). Fire-and-forget: taxonomy accounting can
    never perturb the dispatch error path. NEVER raises."""
    try:
        from backend.core.ouroboros.governance.provider_liquidity_ledger import (
            record_quota_exhaustion,
        )
        record_quota_exhaustion(provider, reason=str(reason)[:160])
    except Exception:  # noqa: BLE001
        pass


def _record_dw_failure_signal(model_id: str, failure_source: Any) -> None:
    """Slice 176 — fuse a classified NON-transport DW FailureSource into the predictive
    cortex as a weighted failure vector (economic 429 / upstream 5xx+parse / stall), per
    model. Transport ruptures are already fed via _note_dw_live_transport_degraded; this
    covers the rest of the spectrum (Blindspot D). Fire-and-forget, never perturbs the
    dispatch error path. NEVER raises."""
    try:
        from backend.core.ouroboros.governance.topology_sentinel import FailureSource
        _kind = {
            FailureSource.LIVE_HTTP_429: "economic",   # quota / rate-limit — imminent lockdown
            FailureSource.LIVE_HTTP_4XX_QUOTA: "economic",  # wallet death (council finding)
            FailureSource.LIVE_HTTP_5XX: "upstream",    # server error — localized
            FailureSource.LIVE_PARSE_ERROR: "upstream",  # malformed/empty completion
            FailureSource.LIVE_STREAM_STALL: "transport",  # stalled stream — transport class
        }.get(failure_source)
        if _kind is None:
            return
        from backend.core.ouroboros.governance.dw_failure_predictor import (
            get_dw_failure_predictor as _s176_pred,
        )
        _s176_pred().record_failure(model_id=model_id, kind=_kind)
    except Exception:  # noqa: BLE001 — never perturb the dispatch error path
        pass


# ---------------------------------------------------------------------------
# A3 Transport Circuit Breaker -- lane-selection + outcome helpers
# ---------------------------------------------------------------------------

def _breaker_select_transport(preferred: str) -> str:
    """Return the actual transport lane to use for the current DW dispatch attempt.

    When the TransportCircuitBreaker is enabled and the preferred lane is OPEN,
    rotates traffic to the sibling lane (batch->realtime or realtime->batch).
    When the breaker is disabled (default), returns ``preferred`` unchanged --
    fully OFF byte-identical.

    Contract: NEVER raises -- any error inside the breaker returns ``preferred``.

    Env:
        JARVIS_TRANSPORT_BREAKER_ENABLED (bool, default false) -- master switch.
    """
    try:
        from backend.core.ouroboros.governance.transport_circuit_breaker import (
            breaker_enabled as _tb_enabled,
            get_transport_breaker as _get_tb,
        )
        if not _tb_enabled():
            return preferred
        import time as _t
        return _get_tb().select_lane(preferred, now=_t.monotonic())
    except Exception:  # noqa: BLE001 -- breaker must NEVER break dispatch
        return preferred


# I2 -- filter set: only LIVE transport signals are meaningful to the per-lane
# TransportCircuitBreaker. GENERATION_TIMEOUT / FSM_EXHAUSTED / HTTP status
# classification / auth terminals are OUR-side faults, NOT vendor transport
# ruptures. Recording them would spuriously trip the lane breaker on OUR bugs.
_BREAKER_RECORD_SOURCES: "frozenset[str]" = frozenset({
    "LIVE_TRANSPORT",
    "LIVE_HTTP_5XX",
    "LIVE_STREAM_STALL",
    "LIVE_HTTP_429",
})


def lane_escalation_enabled() -> bool:
    """Dynamic Lane Escalation (Part 2, T4) master flag -- default TRUE.

    When TRUE, ``_breaker_record_outcome`` re-arms the transport breaker's VISION
    for the ONE failure class the legacy ``_BREAKER_RECORD_SOURCES`` allowlist was
    blind to: a *batch-lane retrieval TIMEOUT* (the DW batch poll never completing
    inside its deadline). Such a failure is wrapped into an FSM_EXHAUSTED
    ``all_providers_exhausted`` and would otherwise be dropped -- so the armed
    batch->realtime breaker never tripped on the live C2 wedge.

    When FALSE, the breaker stays blind EXACTLY as before (byte-identical): the
    batch-TIMEOUT bypass is skipped and only the legacy LIVE_* allowlist records.
    NEVER raises.
    """
    raw = os.environ.get("JARVIS_LANE_ESCALATION_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _is_trippable_batch_lane_timeout(
    lane: str, exc: "Optional[BaseException]"
) -> bool:
    """True iff (gate ON) ``exc`` on ``lane`` is a DW batch-lane retrieval TIMEOUT
    that SHOULD be recorded as a trippable transport failure even though its
    surface classification is FSM_EXHAUSTED (excluded by the legacy allowlist).

    Delegates the load-bearing decision to the taxonomy predicate
    (``is_batch_lane_retrieval_timeout``) which keys on lane=batch +
    fsm_failure_mode=TIMEOUT + DoublewordInfraError "batch retrieval" -- and
    rejects generic exhaustion, tool-loop deadlines, and LOCAL_EGRESS_OVERWEIGHT.
    Fail-soft: any error -> False (record nothing, legacy behavior).
    """
    try:
        if exc is None or not lane_escalation_enabled():
            return False
        from backend.core.ouroboros.governance.dw_fault_taxonomy import (
            is_batch_lane_retrieval_timeout as _batch_timeout,
        )
        return bool(_batch_timeout(exc, lane=lane))
    except Exception:  # noqa: BLE001 -- classification error => record nothing
        return False


def _is_realtime_lane_collapse(
    lane: str, exc: "Optional[BaseException]"
) -> bool:
    """True iff (gate ON) ``exc`` on ``lane`` is a realtime-lane generation
    TIMEOUT -- i.e. LANE COLLAPSE: after T4 rotated this op off the wedged batch
    lane onto realtime, the realtime lane ALSO timed out (both transport lanes
    exhausted by timeout for this op).

    Delegates the load-bearing decision to the taxonomy predicate
    (``is_realtime_lane_timeout``) which keys on lane=realtime +
    fsm_failure_mode=TIMEOUT (or a bare asyncio TimeoutError) and rejects batch
    timeouts, tool-loop generation deadlines, and LOCAL_EGRESS_OVERWEIGHT.
    Fail-soft: any error -> False (no collapse declared, legacy behavior)."""
    try:
        if exc is None or not lane_escalation_enabled():
            return False
        from backend.core.ouroboros.governance.dw_fault_taxonomy import (
            is_realtime_lane_timeout as _rt_timeout,
        )
        return bool(_rt_timeout(exc, lane=lane))
    except Exception:  # noqa: BLE001 -- classification error => no collapse
        return False


def _record_lane_collapse_dilation(
    op_id: "Optional[str]",
    lane: str,
    exc: "Optional[BaseException]",
) -> int:
    """On detected LANE COLLAPSE (realtime-lane timeout after batch rotation):
    emit ``[SOVEREIGN YIELD: LANE COLLAPSE]`` and record ONE bounded per-op
    deadline-dilation hop. Returns the cumulative hop count for ``op_id`` (the
    number to use when computing the dilated deadline for the NEXT attempt), or
    ``0`` when no dilation should happen (gate off / not a collapse / over the
    hop cap / fail-soft).

    Bounded by ``JARVIS_LANE_DILATION_MAX_HOPS``: once the recorded hop count
    EXCEEDS the cap, returns 0 so the dispatcher STOPS dilating and falls
    through to the existing immortal-queue / DLQ backstop (the terminal). NEVER
    raises -> 0 (legacy deadline, op never lost)."""
    try:
        if not lane_escalation_enabled():
            return 0
        if not _is_realtime_lane_collapse(lane, exc):
            return 0
        _op = (op_id or "?")
        from backend.core.ouroboros.governance.convergence_watchdog import (
            get_lane_dilation_tracker as _get_dt,
            lane_dilation_max_hops as _max_hops,
            emit_sovereign_yield as _yield,
        )
        cap = _max_hops()
        if cap <= 0:
            return 0  # dilation disabled -> straight to backstop
        hops = _get_dt().record_dilation_hop(_op)
        if hops > cap:
            # Over budget: bounded -- no further dilation, fall to backstop.
            logger.warning(
                "[SOVEREIGN YIELD: LANE COLLAPSE] op=%s realtime+batch exhausted "
                "by TIMEOUT, dilation hops=%d > cap=%d -> immortal/DLQ backstop",
                _op[:16], hops, cap,
            )
            return 0
        # Within budget: emit the sovereign-yield telemetry + signal dilation.
        try:
            _yield(
                _op,
                lineage_id=_op,
                ratio=0.0,
                consecutive_stalls=hops,
                parent_chars=0,
                child_chars=0,
                tier="lane",
                reason="LANE COLLAPSE",
            )
        except Exception:  # noqa: BLE001 -- telemetry must never break dispatch
            pass
        return int(hops)
    except Exception:  # noqa: BLE001 -- never perturb the dispatch error path
        return 0


def _breaker_record_outcome(
    lane: str,
    *,
    ok: bool,
    failure_mode: "Optional[str]",
    exc: "Optional[BaseException]" = None,
) -> None:
    """Feed an attempt outcome into the TransportCircuitBreaker for ``lane``.

    Guarded by the master gate and fully fail-soft.  Callers do NOT need to
    check ``breaker_enabled()`` -- this function handles the guard internally.

    I2 filter: on ok=False, only records when ``failure_mode`` is a live
    transport signal (LIVE_TRANSPORT / LIVE_HTTP_5XX / LIVE_STREAM_STALL /
    LIVE_HTTP_429). Generation-side faults (GENERATION_TIMEOUT, FSM_EXHAUSTED)
    and auth terminals are silently dropped -- they are OUR fault, not the
    transport lane's.

    T4 batch-TIMEOUT vision bypass: the SOLE exception to the I2 filter. When
    ``JARVIS_LANE_ESCALATION_ENABLED`` is ON and ``exc`` is a *batch-lane
    retrieval TIMEOUT* (lane=batch + fsm_failure_mode=TIMEOUT +
    DoublewordInfraError "batch retrieval" -- the DW batch poll deadline, NOT a
    generic exhaustion, NOT an our-side LOCAL_* fault), the FSM_EXHAUSTED
    exclusion is bypassed for THAT failure only and it is recorded as a trippable
    batch-lane transport failure. The batch lane trips OPEN -> ``select_lane``
    rotates the op to realtime. Every other FSM_EXHAUSTED / GENERATION_TIMEOUT /
    LOCAL_EGRESS_OVERWEIGHT stays dropped.

    Args:
        lane:         The transport lane actually used ("batch" / "realtime").
        ok:           True on success, False on any failure.
        failure_mode: The FailureSource name (e.g. "LIVE_TRANSPORT"), or None on success.
        exc:          The originating exception (T4 batch-TIMEOUT classification);
                      optional and back-compatible -- omitting it preserves the
                      legacy allowlist-only behavior.
    """
    try:
        from backend.core.ouroboros.governance.transport_circuit_breaker import (
            breaker_enabled as _tb_enabled,
            get_transport_breaker as _get_tb,
        )
        if not _tb_enabled():
            return
        # I2: skip recording non-transport failures (they are OUR-side, not lane
        # faults) -- UNLESS T4's batch-lane retrieval TIMEOUT bypass applies. The
        # bypass is the single, surgically-scoped re-arming of the breaker's
        # vision; it cannot fire on a generic exhaustion or an our-side fault.
        if (
            not ok
            and failure_mode not in _BREAKER_RECORD_SOURCES
            and not _is_trippable_batch_lane_timeout(lane, exc)
        ):
            return
        import time as _t
        _get_tb().record(lane, ok=ok, failure_mode=failure_mode, now=_t.monotonic())
    except Exception:  # noqa: BLE001 -- record must NEVER break dispatch
        pass


def dw_preflight_gate_enabled() -> bool:
    """Slice 76 Phase 2 master flag -- default TRUE. When off, dispatch is
    byte-identical to the pre-Slice-76 path (no pre-flight short-circuit)."""
    raw = os.environ.get(
        "JARVIS_DW_PREFLIGHT_GATE_ENABLED", "true",
    ).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _dw_preflight_freshness_s() -> float:
    """Max age (seconds) of a TRANSPORT_DEGRADED surface verdict for the
    pre-flight gate to act on it. Stale evidence is ignored so the gate never
    starves DW on an old reading. Env-tunable; non-positive / invalid → 120s."""
    raw = os.environ.get("JARVIS_DW_PREFLIGHT_FRESHNESS_S", "120").strip()
    try:
        val = float(raw)
        return val if val > 0 else 120.0
    except (ValueError, TypeError):
        return 120.0


def dw_transport_degraded_preflight() -> bool:
    """Slice 76 Phase 2 — pre-flight DW transport health gate.

    Consults the EXISTING ``dw_surface_health`` ledger (kept fresh by the
    surface probes — NO new probe is issued here): returns True iff the
    ``DIRECT_STREAMING`` surface carries a FRESH ``TRANSPORT_DEGRADED`` verdict.
    That means the socket/TLS to the DW endpoint is down RIGHT NOW — every
    ranked sibling model shares that dead transport (cf.
    :func:`should_sever_dw_lane`), so the op should cascade to Claude with its
    full budget BEFORE the ``_primary_sem`` wait + per-model timeout cascade
    burns it (the EVAL-2 ``terminal_timeout``, PRD §50.11).

    Conservative by construction: unknown / stale / HEALTHY / UPSTREAM_DEGRADED
    (server responded — transport is up) all return False, so the DW lane
    proceeds normally and we never starve DW on thin evidence. NEVER raises
    (fail-open: a gate error must not block DW dispatch)."""
    if not dw_preflight_gate_enabled():
        return False
    try:
        from backend.core.ouroboros.governance.dw_surface_health import (
            SurfaceHealthLedger,
            SurfaceKind,
            SurfaceVerdict,
        )
        rec = SurfaceHealthLedger(autosave=False).verdict_for(
            SurfaceKind.DIRECT_STREAMING,
        )
        if rec is None or rec.verdict is not SurfaceVerdict.TRANSPORT_DEGRADED:
            return False
        age_s = time.time() - float(rec.last_probe_unix or 0.0)
        # Slice 127 P3 — the freshness window is how long the DW lane stays
        # severed before the next probe. When the dynamic-recovery master is ON,
        # use the full-jitter EXPONENTIAL window (widens for a chronically-
        # rupturing lane, resets on DW success) instead of the static default.
        # OFF → byte-identical to the pre-P3 fixed window. Fail-safe: a 0/invalid
        # dynamic window falls back to the static one (never starve DW).
        _window_s = _dw_preflight_freshness_s()
        try:
            from backend.core.ouroboros.governance.dw_transport_recovery import (
                dw_dynamic_recovery_enabled as _s127_dyn_on,
                get_dw_transport_recovery as _s127_dwr,
            )
            if _s127_dyn_on():
                _dyn = _s127_dwr().dynamic_recovery_window_s()
                if _dyn and _dyn > 0:
                    _window_s = _dyn
        except Exception:  # noqa: BLE001 — fail-open to the static window
            pass
        return 0.0 <= age_s <= _window_s
    except Exception:  # noqa: BLE001 — never block dispatch on a gate error
        return False


# ---------------------------------------------------------------------------
# Slice 127 P2.1 — fallback-skip gate (IMMEDIATE reroute to DW)
# ---------------------------------------------------------------------------
#
# The live soak proved P1+P2 (no terminal_config brick; economic reclassify +
# ECONOMIC TRIP). But `_generate_immediate` does "Claude direct, skip DW", so an
# IMMEDIATE op keeps grinding against a depleted Claude lane and exhausts instead
# of failing over to the funded DW lane — the existing should_allow_request gate
# only covers Claude-as-PRIMARY. This gate makes the Claude-direct path consult
# the Claude lane breaker first and reroute to the DW primary when it's OPEN.


def fallback_skip_gate_enabled() -> bool:
    """Slice 127 P2.1 master. Slice 146: graduated default-TRUE — when the Claude
    lane breaker is OPEN, IMMEDIATE ops skip the depleted fallback and reroute to
    funded DW (live-proven). Operator can still force-off with =0. NEVER raises."""
    try:
        return os.environ.get(
            "JARVIS_FALLBACK_SKIP_GATE_ENABLED", "true",
        ).strip().lower() in ("1", "true", "yes", "on")
    except Exception:  # noqa: BLE001
        return False


def _dw_autarky_enabled() -> bool:
    """Slice 225 Phase 2 master. Default-TRUE — when the Claude fallback breaker
    is OPEN/HALF_OPEN (terminal_quota / out-of-credits / transport), STANDARD and
    COMPLEX ops keep the DW primary on the full op budget instead of severing it
    at the 30s/75s reflex cap into a dead lane (the live GOAL-001::file-00
    generation_failed wedge). Sibling to the P2.1 IMMEDIATE-route gate above, for
    the STANDARD/COMPLEX primary-budget path. Operator force-off with =0. NEVER
    raises — fail-closed to legacy cascade."""
    try:
        return os.environ.get(
            "JARVIS_DW_AUTARKY_ENABLED", "true",
        ).strip().lower() in ("1", "true", "yes", "on")
    except Exception:  # noqa: BLE001
        return False


def _provider_quota_isolation_enabled() -> bool:
    """Sovereign State Isolation (2026-06-19) master. Default-TRUE — a
    provider's economic/quota death (e.g. Claude 402 'credit balance too
    low') is recorded on THAT provider's own lane breaker only, and is NOT
    allowed to trip the provider-NEUTRAL per-op circuit breaker into
    OPEN_TERMINAL. Without this, Claude's credit-death poisons the whole op
    so DW autarky can never carry it — the empirically-confirmed
    cross-provider contamination (terminal_quota 5->0 once isolated).
    Operator force-off with =0 -> byte-identical legacy. NEVER raises."""
    try:
        return os.environ.get(
            "JARVIS_PROVIDER_QUOTA_ISOLATION_ENABLED", "true",
        ).strip().lower() in ("1", "true", "yes", "on")
    except Exception:  # noqa: BLE001
        return False


def quota_isolation_skips_op_breaker(
    *, is_provider_economic_block: bool, isolation_enabled: bool,
) -> bool:
    """PURE predicate: should the per-op breaker trip be SKIPPED for this
    failure? True iff the failure is a provider economic block AND
    isolation is enabled — the provider's OWN lane breaker already owns the
    death, so tripping the op-neutral breaker would cross-contaminate the op
    for every other (still-viable) provider. NEVER raises."""
    return bool(is_provider_economic_block) and bool(isolation_enabled)


def immediate_reroute_to_dw(
    *,
    dw_is_primary: bool,
    gate_enabled: bool,
    claude_breaker_enabled: bool,
    claude_allows_request: bool,
) -> bool:
    """Pure decision: should an IMMEDIATE op reroute from Claude-direct to the
    DW primary? True iff DW is the primary lane, the gate is on, the Claude lane
    breaker is enabled, and the breaker is NOT allowing requests (OPEN within
    its window). When the breaker allows (CLOSED, or a HALF_OPEN probe), we keep
    Claude-direct so the lane self-heals. Pure — no I/O, no side effects."""
    return bool(
        dw_is_primary
        and gate_enabled
        and claude_breaker_enabled
        and not claude_allows_request
    )


# ---------------------------------------------------------------------------
# Content failure classification
# ---------------------------------------------------------------------------

# Keywords that identify content/model failures vs infrastructure failures.
# Content failures do NOT trigger FailbackFSM state transitions — the primary
# provider is still alive; it merely produced bad output (stale diff, invalid
# schema, etc.).  Infrastructure failures (timeout, connection error) DO
# trigger state transitions.
_CONTENT_FAILURE_PATTERNS: frozenset = frozenset({
    "diff_apply_failed",
    "stale_diff",
    "schema_invalid",
    "no_candidates",
    "validate_diff",
    "StaleDiffError",
})


# Defect #4 Slice A (2026-05-03) — task-leak prevention.
#
# Soak v5 (bt-2026-05-03-060330) recorded 4 "Task exception was never
# retrieved" asyncio errors. Root cause: ensure_future/create_task
# spawns of provider .generate() coroutines were wrapped in
# asyncio.shield(...) which prevents cancellation when the outer
# wait_for times out. The shielded task continues running; if it
# later raises (e.g., RuntimeError('all_providers_exhausted')) and
# nobody awaits the result, asyncio's default handler logs the
# unhandled exception.
#
# Fix: every ensure_future/create_task of .generate() (or background
# poll wrappers) gets _swallow_task_exception attached as a
# done_callback. The callback retrieves the exception, classifies
# it, and either logs at DEBUG (expected: all_providers_exhausted /
# CancelledError / TimeoutError) or WARNING (unexpected). The task
# exception is consumed either way.

_EXPECTED_BACKGROUND_EXC_PATTERNS = (
    "all_providers_exhausted",
    "deadline_exhausted_pre_fallback",
    "topology_block",
    "fallback_disabled_by_env",
    "queue_only_dispatch",
)


def _swallow_task_exception(task: "asyncio.Future") -> None:
    """Done-callback that retrieves + classifies + consumes a task
    exception so it never reaches asyncio's default handler.

    Attach to every ``asyncio.ensure_future(...)`` /
    ``asyncio.create_task(...)`` of provider .generate() or
    background poll coroutines that may outlive their primary
    awaiter (e.g., shielded tasks that survive outer wait_for
    timeouts).

    NEVER raises -- contract: even a misbehaving exception accessor
    must not propagate.
    """
    try:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        msg = str(exc) if exc else ""
        # Expected provider/orchestration exceptions: log at DEBUG.
        # The exception was already logged at the raise site; this
        # callback exists to CONSUME the exception, not re-log it.
        is_expected = (
            isinstance(exc, asyncio.CancelledError)
            or isinstance(exc, asyncio.TimeoutError)
            or any(p in msg for p in _EXPECTED_BACKGROUND_EXC_PATTERNS)
        )
        if is_expected:
            logger.debug(
                "[CandidateGenerator] background task expected exit: "
                "%s(%s)", type(exc).__name__, msg[:120],
            )
        else:
            logger.warning(
                "[CandidateGenerator] background task unhandled "
                "exception (consumed by _swallow_task_exception to "
                "prevent asyncio leak): %s(%s)",
                type(exc).__name__, msg[:200],
            )
    except Exception:  # noqa: BLE001 -- contract: never crash callback
        pass


def _is_content_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a content/model failure (not infrastructure).

    Content failures: wrong diff, stale context, invalid JSON schema.
    Infrastructure failures: timeout, connection refused, OOM.
    """
    msg = str(exc).lower()
    return any(pattern.lower() in msg for pattern in _CONTENT_FAILURE_PATTERNS)


# ---------------------------------------------------------------------------
# Exhaustion log helpers
# ---------------------------------------------------------------------------


def _trim_exc_msg(exc: BaseException, limit: int = 200) -> str:
    """Stringify *exc*, clip to *limit* chars, collapse whitespace."""
    msg = str(exc)
    if len(msg) > limit:
        msg = msg[:limit] + "..."
    return msg.replace("\n", "\\n").replace("\t", " ")


def _fmt_val(value: Any) -> str:
    """Format *value* for a ``key=value`` structured log line.

    Values with whitespace are underscored so grep-based audits can
    treat one log line as a flat sequence of ``key=value`` tokens.
    """
    s = str(value)
    return s.replace(" ", "_")


# ---------------------------------------------------------------------------
# Local-tier failure classifier (Phase 3 Task 7)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LocalFailureVerdict:
    degrade: bool
    cascade_upstream: bool
    target_state: Optional[str]


def classify_local_failure(exc: BaseException) -> LocalFailureVerdict:
    """Map a local-tier exception to an FSM transition verdict.

    A terminal_lag_lockup degrades J-Prime to PRIMARY_DEGRADED and cascades the
    op upstream (the FailbackStateMachine already passes context on cascade, so no
    L2 sandbox teardown). All other exceptions are ordinary provider failures.
    """
    _LOCAL_DEGRADE_CLASSES = ("terminal_lag_lockup", "local_memory_critical")
    if getattr(exc, "failure_class", None) in _LOCAL_DEGRADE_CLASSES:
        return LocalFailureVerdict(
            degrade=True, cascade_upstream=True, target_state="PRIMARY_DEGRADED"
        )
    return LocalFailureVerdict(degrade=False, cascade_upstream=False, target_state=None)


# ---------------------------------------------------------------------------
# CandidateProvider Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CandidateProvider(Protocol):
    """Runtime-checkable protocol for code generation backends.

    Any class that implements these methods can serve as a primary or
    fallback generation provider.
    """

    @property
    def provider_name(self) -> str:
        """Human-readable name of this provider (e.g. ``"gcp-jprime"``)."""
        ...  # pragma: no cover

    async def generate(
        self, context: OperationContext, deadline: datetime
    ) -> GenerationResult:
        """Generate candidate code changes for the given operation.

        Parameters
        ----------
        context:
            The operation context describing what needs to change.
        deadline:
            Absolute UTC deadline by which generation must complete.

        Returns
        -------
        GenerationResult
            The generated candidates with timing metadata.

        Raises
        ------
        Exception
            Any failure (timeout, OOM, network) should propagate as an exception.
        """
        ...  # pragma: no cover

    async def health_probe(self) -> bool:
        """Quick liveness check.

        Returns
        -------
        bool
            ``True`` if the provider is healthy and ready to serve requests.
        """
        ...  # pragma: no cover

    async def plan(self, prompt: str, deadline: datetime) -> str:
        """Send a lightweight planning prompt; return the raw string response.

        Used by ContextExpander. Planning failures are soft — callers tolerate
        exceptions and skip expansion rounds gracefully.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# FailbackState Enum
# ---------------------------------------------------------------------------


class FailbackState(Enum):
    """States in the failover/failback state machine."""

    PRIMARY_READY = auto()
    FALLBACK_ACTIVE = auto()
    PRIMARY_DEGRADED = auto()
    QUEUE_ONLY = auto()


class FailureMode(Enum):
    """Classification of provider failure for recovery prediction.

    Different failure modes have vastly different recovery profiles:
    rate limits clear in seconds, connection errors take minutes to hours.
    The FSM uses this to predict when the primary will be available again,
    minimizing expensive fallback spend (Manifesto §5 — deterministic routing).
    """

    RATE_LIMITED = auto()       # 429, CircuitBreakerOpen — seconds to recover
    TIMEOUT = auto()            # Request/connection timeout — minutes
    SERVER_ERROR = auto()       # 500/502/503 — minutes
    CONNECTION_ERROR = auto()   # Can't reach host — minutes to hours
    CONTENT_FAILURE = auto()    # Bad output, infra healthy — no penalty
    CONTEXT_OVERFLOW = auto()   # Tool loop prompt exceeded char limit — immediate fallback
    TRANSIENT_TRANSPORT = auto()  # HTTP/2 disconnect, premature stream close — seconds
    TEMPORAL_SHED = auto()      # Temporal Veto fast-fail — NOT a DW health fact;
    #                             zero primary penalty, zero retry, immediate cascade.
    #                             The op's budget was too tight for any DW lane; the
    #                             NEXT op with a normal budget must still use DW.
    LOCAL_DEFECT = auto()       # A bug in OUR code on the primary call path —
    #                             TypeError/AttributeError/NameError/ImportError.
    #                             NOT a provider fact at all. Every unrecognised
    #                             exception used to land on the TIMEOUT default,
    #                             so ONE signature drift locked DoubleWord out
    #                             (should_attempt_primary False after a single
    #                             occurrence) and silently billed every
    #                             subsequent op to Claude — with the logs
    #                             blaming an upstream that was never asked.
    #                             Zero primary penalty; still cascades so the
    #                             op is not dropped; logged at ERROR because
    #                             this one is ours to fix.


# Mode-specific recovery parameters for exponential backoff.
# base_s * 2^(consecutive_failures - 1), capped at max_s.
_RECOVERY_PARAMS: dict[FailureMode, dict[str, float]] = {
    FailureMode.RATE_LIMITED:    {"base_s": 15.0,  "max_s": 120.0},
    FailureMode.TIMEOUT:         {"base_s": 45.0,  "max_s": 300.0},
    FailureMode.SERVER_ERROR:    {"base_s": 60.0,  "max_s": 600.0},
    FailureMode.CONNECTION_ERROR: {"base_s": 120.0, "max_s": 900.0},
    FailureMode.CONTENT_FAILURE: {"base_s": 0.0,   "max_s": 0.0},
    # CONTEXT_OVERFLOW: Tool loop prompt exceeded char limit. The provider
    # infrastructure is healthy — the prompt was just too large. Immediate
    # fallback to Tier 1 with zero backoff penalty (same profile as
    # CONTENT_FAILURE). No timeout ETA penalty on the FSM.
    FailureMode.CONTEXT_OVERFLOW: {"base_s": 0.0,  "max_s": 0.0},
    # TEMPORAL_SHED: a routing refusal (deadline vs batch plane), not an
    # infra failure — no backoff, DW immediately eligible for the next op.
    FailureMode.TEMPORAL_SHED:   {"base_s": 0.0,  "max_s": 0.0},
    # TRANSIENT_TRANSPORT: HTTP/2 GOAWAY, RemoteProtocolError, ClosedResourceError.
    # The transport layer flapped (often a single dropped connection in a keep-alive
    # pool) but the upstream API is healthy. A 5s base backs off to 30s after 4
    # consecutive failures, then immediately retries — much shorter than TIMEOUT
    # (45s/300s) which it would otherwise be misclassified as. Diagnosed in
    # bt-2026-04-12-005521 where 9 consecutive ops died with all_providers_exhausted
    # because RemoteProtocolError fell through to the TIMEOUT default and the
    # CONNECTION_ERROR-only deep-backoff guard never engaged.
    FailureMode.TRANSIENT_TRANSPORT: {"base_s": 5.0, "max_s": 30.0},
    # LOCAL_DEFECT: our own bug, not the provider's. Zero backoff — there is
    # nothing upstream to wait for, and waiting would only hide the defect
    # behind an outage-shaped delay. Same zero profile as CONTENT_FAILURE.
    FailureMode.LOCAL_DEFECT:    {"base_s": 0.0,   "max_s": 0.0},
}


# Failure modes where the PRIMARY IS INNOCENT — the op still cascades, but the
# provider FSM must not be penalised for it. Named once and consulted at both
# classify-then-record sites (the Tier 0 RT path and
# ``_try_primary_then_fallback``), which previously kept two hand-maintained
# copies of this list and had already drifted apart by one member.
_PRIMARY_INNOCENT_MODES: frozenset = frozenset({
    FailureMode.CONTENT_FAILURE,   # model produced bad output; infra healthy
    FailureMode.TEMPORAL_SHED,     # OUR routing refusal, not a DW health fact
    FailureMode.LOCAL_DEFECT,      # OUR bug on the call path
})


# Exception TYPES that can only mean a defect in this codebase, never a
# provider condition. Matched by type — never by message — because a provider
# is perfectly capable of returning prose containing the word "attributeerror",
# and a string match would let an upstream error masquerade as our bug in
# exactly the direction that hides real outages.
#
# Deliberately EXCLUDES the ambiguous ones. ``KeyError`` / ``IndexError`` /
# ``ValueError`` are usually raised while parsing a provider's response, where
# the true fault is a malformed upstream payload as often as it is our
# indexing. Claiming those as local defects would suppress a real provider
# penalty, so they keep the conservative TIMEOUT default until something
# proves otherwise.
_LOCAL_DEFECT_TYPES: tuple = (
    TypeError,          # signature drift — the Slice 30 `model_id=` class
    AttributeError,     # a renamed/removed attribute
    NameError,          # includes UnboundLocalError
    ImportError,        # includes ModuleNotFoundError
    IndentationError,   # subclass of SyntaxError; listed for the reader
    SyntaxError,        # a generated/exec'd fragment that will never parse
)


def _local_defect_classification_enabled() -> bool:
    """Master gate. Default TRUE — failure-path-only: it changes nothing until
    an exception that can only be our bug reaches the primary handler. ``=0``
    restores the byte-identical legacy behaviour where such an exception was
    classified as a provider TIMEOUT. NEVER raises."""
    return (os.environ.get(
        "JARVIS_LOCAL_DEFECT_CLASSIFICATION_ENABLED", "true",
    ) or "").strip().lower() not in ("0", "false", "no", "off")


# Exception class names that indicate transient transport-layer flap rather than
# upstream API failure. Match by name (not isinstance) so we don't pull in httpx
# or anyio at module import time — the actual SDK may not be installed on hosts
# where the FSM is constructed (battle test harness, planner-only deployments).
_TRANSIENT_TRANSPORT_NAMES: frozenset = frozenset({
    "RemoteProtocolError",     # httpx — server disconnected without response
    "ClosedResourceError",     # anyio — stream got closed mid-read
    "ProtocolError",           # h11/h2 — generic protocol violation
    "LocalProtocolError",      # h11 — local-side protocol violation
    "IncompleteRead",          # http.client — short read
    "StreamConsumed",          # httpx — re-read of consumed stream
    "StreamClosed",            # httpx — read after close
    "ResponseNotRead",         # httpx — async stream race
})


# FailureMode set safe to retry from `_call_fallback`'s outer loop. Any
# mode in this set indicates a transient infrastructure condition where
# re-invoking the provider may succeed on a fresh TCP connection / fresh
# pool generation. Permanent failure modes (CONTENT_FAILURE,
# CONTEXT_OVERFLOW) MUST NOT be retried — they would just re-fail.
# Defined as a frozenset (not the FailureMode enum directly) to avoid
# import ordering with the FailureMode definition below; populated lazily
# by `_is_outer_retry_eligible_mode()`.
_FALLBACK_OUTER_RETRY_TRANSIENT_MODE_NAMES: frozenset = frozenset({
    "TIMEOUT",
    "CONNECTION_ERROR",
    "TRANSIENT_TRANSPORT",
    "SERVER_ERROR",
    "RATE_LIMITED",
})


def _is_outer_retry_eligible_mode(mode: "FailureMode") -> bool:
    """Return True iff ``mode`` indicates a transient failure worth
    retrying within the remaining fallback budget.

    Used by `_call_fallback`'s outer retry loop (rooted-problem fix
    2026-04-25). Defined as a free function so unit tests can pin the
    classification → retry decision without instantiating the full
    `CandidateGenerator`.
    """
    return mode.name in _FALLBACK_OUTER_RETRY_TRANSIENT_MODE_NAMES


def _walk_exception_chain(exc: BaseException, max_depth: int = 8) -> tuple:
    """Walk __cause__/__context__ chain returning a tuple of exceptions.

    Anthropic SDK wraps httpx exceptions in APIConnectionError; the inner
    httpx exception is the actual signal we need to classify. Walks both
    __cause__ (explicit `raise X from Y`) and __context__ (implicit during
    `except` handler), with cycle protection.

    Returns the chain ordered outermost-first.
    """
    chain: list = []
    seen: set = set()
    current: Optional[BaseException] = exc
    depth = 0
    while current is not None and depth < max_depth:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        # Prefer __cause__ (explicit) over __context__ (implicit).
        nxt = getattr(current, "__cause__", None)
        if nxt is None:
            nxt = getattr(current, "__context__", None)
        current = nxt
        depth += 1
    return tuple(chain)


# ---------------------------------------------------------------------------
# FailbackStateMachine
# ---------------------------------------------------------------------------


class FailbackStateMachine:
    """Asymmetric failover/failback state machine.

    Failover is immediate (one failure), but failback requires
    ``required_probes`` consecutive health probes spanning at least
    ``dwell_time_s`` seconds.

    Parameters
    ----------
    required_probes:
        Number of consecutive successful health probes needed before
        promoting from PRIMARY_DEGRADED to PRIMARY_READY.
    dwell_time_s:
        Minimum wall-clock seconds that must elapse between the first
        successful probe and the promotion to PRIMARY_READY.
    """

    def __init__(
        self,
        required_probes: int = 3,
        dwell_time_s: float = 45.0,
    ) -> None:
        self._state: FailbackState = FailbackState.PRIMARY_READY
        self._required_probes: int = required_probes
        self._dwell_time_s: float = dwell_time_s
        self._consecutive_probes: int = 0
        self._first_probe_at: Optional[float] = None  # monotonic timestamp
        self.content_failure_count: int = 0  # content/model failures (not infra)
        # Adaptive recovery tracking (Manifesto §5 — deterministic routing)
        self._failure_mode: Optional[FailureMode] = None
        self._consecutive_failures: int = 0
        self._last_failure_at: float = 0.0   # monotonic
        self._last_success_at: float = 0.0   # monotonic

    @property
    def state(self) -> FailbackState:
        """Current FSM state."""
        return self._state

    def record_primary_failure(
        self, mode: FailureMode = FailureMode.TIMEOUT,
    ) -> None:
        """Record a primary provider failure with failure mode classification.

        Transitions immediately to FALLBACK_ACTIVE from any non-QUEUE_ONLY state.
        Tracks failure mode for recovery prediction (Manifesto §5).

        Parameters
        ----------
        mode:
            Classification of the failure. Defaults to TIMEOUT for backward
            compatibility with existing callers.
        """
        if self._state is FailbackState.QUEUE_ONLY:
            return
        # Structural backstop for the modes where the primary is innocent.
        #
        # Their zero-valued `_RECOVERY_PARAMS` entries LOOK like the guarantee
        # ("no penalty"), and are not: this method sets FALLBACK_ACTIVE and
        # increments `_consecutive_failures` for any mode it is handed, so
        # `should_attempt_primary()` goes False after a single call even for
        # CONTENT_FAILURE. The exemption was only ever a property of each
        # caller remembering to branch — and one of the two callers had already
        # forgotten TEMPORAL_SHED.
        #
        # Enforced here so the guarantee lives with the invariant instead of
        # with everyone who calls it, and a third call site inherits it.
        if mode in _PRIMARY_INNOCENT_MODES:
            logger.debug(
                "[FailbackFSM] %s is not a primary-health fact — "
                "penalty refused, state unchanged", mode.name,
            )
            return
        if self._state in (
            FailbackState.PRIMARY_READY,
            FailbackState.FALLBACK_ACTIVE,
            FailbackState.PRIMARY_DEGRADED,
        ):
            self._state = FailbackState.FALLBACK_ACTIVE
            # Track failure mode for adaptive recovery — do NOT reset these
            # in _reset_probe_counters; they persist across probe cycles.
            self._failure_mode = mode
            self._consecutive_failures += 1
            self._last_failure_at = time.monotonic()
            self._reset_probe_counters()
            params = _RECOVERY_PARAMS.get(mode, _RECOVERY_PARAMS[FailureMode.TIMEOUT])
            # Phase 12.2 Slice C — full-jitter retrofit. Master-flag-off
            # preserves exact-exponential bit-for-bit. When enabled,
            # uniform jitter desynchronizes our probe waveform from
            # other JARVIS-class clients hammering the same DW endpoint
            # after recovery.
            try:
                from backend.core.ouroboros.governance.full_jitter import (
                    full_jitter_backoff_s,
                    full_jitter_enabled,
                )
                if full_jitter_enabled():
                    eta_s = full_jitter_backoff_s(
                        max(self._consecutive_failures - 1, 0),
                        base_s=params["base_s"],
                        cap_s=params["max_s"],
                    )
                else:
                    eta_s = min(
                        params["base_s"] * (2 ** max(self._consecutive_failures - 1, 0)),
                        params["max_s"],
                    )
            except Exception:  # noqa: BLE001 — defensive
                eta_s = min(
                    params["base_s"] * (2 ** max(self._consecutive_failures - 1, 0)),
                    params["max_s"],
                )
            logger.warning(
                "[FailbackFSM] Primary failure (mode=%s, consecutive=%d, "
                "recovery_eta=+%.0fs) -> FALLBACK_ACTIVE",
                mode.name, self._consecutive_failures, eta_s,
            )

    def record_fallback_failure(
        self, mode: FailureMode = FailureMode.TIMEOUT,
    ) -> None:
        """Record a fallback provider failure.

        FALLBACK_ACTIVE -> QUEUE_ONLY for permanent failures.
        For transient failures (TIMEOUT, RATE_LIMITED), stays in
        FALLBACK_ACTIVE so the system can retry on the next operation
        instead of permanently giving up.
        """
        if self._state is not FailbackState.FALLBACK_ACTIVE:
            return

        if mode in (FailureMode.TIMEOUT, FailureMode.RATE_LIMITED,
                    FailureMode.SERVER_ERROR, FailureMode.CONTEXT_OVERFLOW,
                    FailureMode.CONTENT_FAILURE):
            # Transient / non-infra: DON'T go to QUEUE_ONLY. The next
            # operation will re-evaluate should_attempt_primary() and may
            # succeed. CONTEXT_OVERFLOW is a prompt-size issue, not infra.
            logger.warning(
                "[FailbackFSM] Fallback transient failure (mode=%s) — "
                "staying FALLBACK_ACTIVE (recoverable)",
                mode.name,
            )
            return

        # Permanent failure (CONNECTION_ERROR, auth, unknown) → QUEUE_ONLY
        self._state = FailbackState.QUEUE_ONLY
        self._queue_only_at: float = time.monotonic()
        self._reset_probe_counters()
        logger.error(
            "[FailbackFSM] Fallback failure (mode=%s) -> QUEUE_ONLY "
            "(all providers exhausted)",
            mode.name,
        )

    def record_probe_success(self) -> None:
        """Record a successful health probe of the primary provider.

        FALLBACK_ACTIVE -> PRIMARY_DEGRADED (first probe).
        PRIMARY_DEGRADED stays until required_probes AND dwell_time_s met,
        then -> PRIMARY_READY.
        PRIMARY_READY -> no-op.
        QUEUE_ONLY -> FALLBACK_ACTIVE (auto-recovery: a successful probe
        means the primary is alive again, so we should exit the dead-end).
        """
        if self._state is FailbackState.PRIMARY_READY:
            return
        if self._state is FailbackState.QUEUE_ONLY:
            # Auto-recovery: primary is alive → exit dead-end
            self._state = FailbackState.FALLBACK_ACTIVE
            self._reset_probe_counters()
            elapsed = time.monotonic() - getattr(self, "_queue_only_at", 0.0)
            logger.info(
                "[FailbackFSM] QUEUE_ONLY auto-recovery: probe succeeded "
                "after %.1fs — transitioning to FALLBACK_ACTIVE",
                elapsed,
            )
            # Fall through to the FALLBACK_ACTIVE handler below
            # so the first probe is counted toward PRIMARY_DEGRADED.

        now = time.monotonic()

        if self._state is FailbackState.FALLBACK_ACTIVE:
            # First probe: transition to PRIMARY_DEGRADED
            self._state = FailbackState.PRIMARY_DEGRADED
            self._consecutive_probes = 1
            self._first_probe_at = now
            logger.info(
                "[FailbackFSM] First probe success -> PRIMARY_DEGRADED (1/%d)",
                self._required_probes,
            )
            self._maybe_promote(now)
            return

        if self._state is FailbackState.PRIMARY_DEGRADED:
            self._consecutive_probes += 1
            logger.info(
                "[FailbackFSM] Probe success (%d/%d)",
                self._consecutive_probes,
                self._required_probes,
            )
            self._maybe_promote(now)

    def record_probe_failure(self) -> None:
        """Record a failed health probe of the primary provider.

        PRIMARY_DEGRADED -> FALLBACK_ACTIVE (resets probe counters).
        Other states: no-op.
        """
        if self._state is FailbackState.PRIMARY_DEGRADED:
            self._state = FailbackState.FALLBACK_ACTIVE
            self._reset_probe_counters()
            logger.warning(
                "[FailbackFSM] Probe failure -> FALLBACK_ACTIVE (reset)"
            )

    def _maybe_promote(self, now: float) -> None:
        """Check if promotion criteria (probes + dwell) are met."""
        if self._state is not FailbackState.PRIMARY_DEGRADED:
            return
        if self._consecutive_probes < self._required_probes:
            return
        if self._first_probe_at is not None:
            elapsed = now - self._first_probe_at
            if elapsed < self._dwell_time_s:
                logger.info(
                    "[FailbackFSM] Probes met (%d/%d) but dwell not satisfied "
                    "(%.1fs / %.1fs)",
                    self._consecutive_probes,
                    self._required_probes,
                    elapsed,
                    self._dwell_time_s,
                )
                return
        # All criteria met
        self._state = FailbackState.PRIMARY_READY
        self._reset_probe_counters()
        self._reset_failure_tracking()
        logger.info("[FailbackFSM] Promoted -> PRIMARY_READY")

    def _reset_probe_counters(self) -> None:
        """Reset probe tracking state."""
        self._consecutive_probes = 0
        self._first_probe_at = None

    def _reset_failure_tracking(self) -> None:
        """Reset adaptive recovery state on successful recovery."""
        self._consecutive_failures = 0
        self._failure_mode = None
        self._last_success_at = time.monotonic()

    def record_primary_success(self) -> None:
        """Record a successful primary generation (explicit recovery signal).

        Called when the primary provider successfully generates candidates
        after a period of failure. Resets all failure tracking so subsequent
        failures start fresh with base-level backoff.
        """
        if self._consecutive_failures > 0:
            recovery_duration = time.monotonic() - self._last_failure_at
            logger.info(
                "[FailbackFSM] Primary recovered (was %s, %d consecutive failures, "
                "recovery took %.1fs)",
                self._failure_mode.name if self._failure_mode else "UNKNOWN",
                self._consecutive_failures,
                recovery_duration,
            )
        self._reset_failure_tracking()

    # ------------------------------------------------------------------
    # Recovery prediction (deterministic — Manifesto §5)
    # ------------------------------------------------------------------

    def recovery_eta(self) -> float:
        """Predicted monotonic timestamp when primary will be available.

        Uses mode-specific exponential backoff:
        ``last_failure_at + base_s * 2^(consecutive_failures - 1)``,
        capped at ``max_s``.

        Returns 0.0 if no failures recorded (primary is healthy).
        """
        if self._consecutive_failures == 0 or self._failure_mode is None:
            return 0.0
        if self._failure_mode is FailureMode.CONTENT_FAILURE:
            return time.monotonic()  # instant — no infra penalty
        params = _RECOVERY_PARAMS.get(
            self._failure_mode, _RECOVERY_PARAMS[FailureMode.TIMEOUT],
        )
        # Phase 12.2 Slice C — full-jitter retrofit (matches the sister
        # callsite in record_primary_failure). Master-flag-off preserves
        # exact-exponential bit-for-bit; on, uniform random delay
        # desynchronizes our probe schedule from the global herd.
        try:
            from backend.core.ouroboros.governance.full_jitter import (
                full_jitter_backoff_s,
                full_jitter_enabled,
            )
            if full_jitter_enabled():
                delay = full_jitter_backoff_s(
                    max(self._consecutive_failures - 1, 0),
                    base_s=params["base_s"],
                    cap_s=params["max_s"],
                )
            else:
                delay = min(
                    params["base_s"] * (2 ** max(self._consecutive_failures - 1, 0)),
                    params["max_s"],
                )
        except Exception:  # noqa: BLE001 — defensive
            delay = min(
                params["base_s"] * (2 ** max(self._consecutive_failures - 1, 0)),
                params["max_s"],
            )
        return self._last_failure_at + delay

    def should_attempt_primary(self) -> bool:
        """Should we try the primary (cheap) provider?

        Returns True if the primary is healthy or the predicted recovery
        window has elapsed. This enables cost-aware routing: always prefer
        the cheap provider when it's likely available.
        """
        if self._state is FailbackState.PRIMARY_READY:
            return True
        if self._consecutive_failures == 0:
            return True
        return time.monotonic() >= self.recovery_eta()

    def recommended_probe_interval(self) -> float:
        """Adaptive probe interval based on distance to recovery ETA.

        - Far from ETA (>60s away): 60s (relax — no point hammering)
        - Near ETA (<30s away): 10s (ramp up — catch recovery fast)
        - Past ETA: 5s (aggressive — recovery is imminent)
        - Primary healthy: 30s (normal cadence)

        Returns seconds to sleep before next health probe.
        """
        if self._state is FailbackState.PRIMARY_READY:
            return 30.0
        if self._consecutive_failures == 0:
            return 30.0

        eta = self.recovery_eta()
        distance = eta - time.monotonic()

        if distance > 60.0:
            return 60.0   # Deep backoff — relax probes
        elif distance > 30.0:
            return 20.0   # Approaching — moderate
        elif distance > 0.0:
            return 10.0   # Close — ramp up
        else:
            return 5.0    # Past ETA — aggressive probe

    @staticmethod
    def classify_exception(exc: BaseException) -> FailureMode:
        """Classify an exception into a failure mode for recovery prediction.

        Walks the ``__cause__`` / ``__context__`` chain because the Anthropic SDK
        (and other modern HTTP clients) wraps low-level transport errors in a
        higher-level wrapper class — e.g. ``APIConnectionError(cause=
        RemoteProtocolError("Server disconnected without sending a response."))``.
        Classifying only the outer wrapper would have us treat a 50ms HTTP/2
        keep-alive flap as a 120s CONNECTION_ERROR deep-backoff. Instead we walk
        every layer and let the most specific (transient transport) classification
        win.

        Uses string-based type checking to avoid hard dependency on httpx/anyio.
        """
        # Content failures first (don't penalize infra). Check the outermost
        # exception's full message — content failure markers are stamped on
        # the wrapper (e.g. RuntimeError("diff_apply_failed: ...")).
        if _is_content_failure(exc):
            return FailureMode.CONTENT_FAILURE

        # Stream Rupture Breaker: the typed exception carries a
        # provider_stream_rupture:... message. Classify as TRANSIENT_TRANSPORT
        # so the FSM uses the short 5s/30s recovery profile and cascades
        # to Tier 1 immediately.
        #
        # Slice 12F-B (2026-05-22) — StreamBudgetTooShortError is the
        # diagnostic sibling: not a network-side rupture, but a local
        # decision to refuse dispatch when wall_remaining < the
        # JARVIS_STREAM_MINIMUM_READ_BUDGET_S floor. Same classifier
        # mapping (TRANSIENT_TRANSPORT) — same Slice 7 fallback
        # behaviour — but the postmortem can tell the two apart.
        from backend.core.ouroboros.governance.stream_rupture import (
            StreamBudgetTooShortError,
            StreamRuptureError,
        )
        if isinstance(
            exc, (StreamRuptureError, StreamBudgetTooShortError),
        ):
            return FailureMode.TRANSIENT_TRANSPORT

        chain = _walk_exception_chain(exc)

        # First pass: any layer that names a known transient transport class
        # wins, regardless of how deep it is. This is the highest-priority
        # signal because the recovery profile (5s base / 30s max) is so much
        # cheaper than CONNECTION_ERROR (120s/900s).
        for layer in chain:
            if type(layer).__name__ in _TRANSIENT_TRANSPORT_NAMES:
                return FailureMode.TRANSIENT_TRANSPORT

        # Second pass: classic classification on the outermost exception.
        # Falls through layers using the existing rules.
        for layer in chain:
            mode = FailbackStateMachine._classify_single(layer)
            if mode is not FailureMode.TIMEOUT:
                # Anything more specific than the conservative TIMEOUT default
                # is preferred — e.g. an inner ConnectionError beats an outer
                # asyncio.TimeoutError because the connection layer is closer
                # to the truth.
                return mode

        # Third pass: is this our bug rather than a provider condition?
        #
        # Runs LAST, deliberately. Both passes above get first refusal, so an
        # SDK that wraps a transport flap in a TypeError still classifies as
        # TRANSIENT_TRANSPORT — the provider layer is closer to the truth
        # whenever it can speak at all. Only once every provider-shaped
        # reading has declined do we conclude the fault is ours.
        #
        # Before this existed, the conservative TIMEOUT default was applied to
        # exceptions that cannot possibly be a timeout, and the cost was not
        # cosmetic: `record_primary_failure(TIMEOUT)` flips
        # `should_attempt_primary()` to False after ONE occurrence, so a single
        # `TypeError` on the call path took the whole DoubleWord lane offline
        # and routed every subsequent op to Claude at ~10× the unit cost —
        # while the logs read "Primary failed (mode=TIMEOUT)", blaming an
        # upstream that had never been contacted.
        if _local_defect_classification_enabled():
            for layer in chain:
                if isinstance(layer, _LOCAL_DEFECT_TYPES):
                    return FailureMode.LOCAL_DEFECT

        # All layers landed on the conservative TIMEOUT default.
        return FailureMode.TIMEOUT

    @staticmethod
    def _classify_single(exc: BaseException) -> FailureMode:
        """Classify a single exception (no chain walking).

        Extracted from ``classify_exception`` so the chain walker can call
        it on each layer. Preserves the original classification rules
        verbatim minus the transient-transport handling (which is checked
        separately in the priority-1 pass).
        """
        exc_type = type(exc).__name__
        msg = str(exc).lower()

        # Temporal Veto fast-fail shed — checked FIRST (before the generic
        # DoublewordInfraError status walk: it subclasses that type with
        # status 0, which would otherwise fall through to the TIMEOUT
        # default and earn DW an undeserved penalty + retry).
        if exc_type == "TemporalBudgetShedError" or "temporal_load_shed" in msg:
            return FailureMode.TEMPORAL_SHED

        # DoublewordInfraError carries a status code
        if exc_type == "DoublewordInfraError":
            status = getattr(exc, "status_code", 0)
            if status == 429:
                return FailureMode.RATE_LIMITED
            if status in (500, 502, 503):
                return FailureMode.SERVER_ERROR
            # status 0 or other — fall through to message analysis

        # Rate limiting signals
        if exc_type == "CircuitBreakerOpen":
            return FailureMode.RATE_LIMITED
        if "429" in msg or "rate" in msg or "too many" in msg:
            return FailureMode.RATE_LIMITED

        # Context overflow — tool loop prompt exceeded char limit.
        # Must be checked before server errors because the char count
        # in the message (e.g. "155000") can contain "500".
        if "tool_loop_budget_exceeded" in msg or "tool_loop_context_overflow" in msg:
            return FailureMode.CONTEXT_OVERFLOW

        # Connection errors
        if isinstance(exc, ConnectionError):
            return FailureMode.CONNECTION_ERROR
        if any(kw in msg for kw in ("connection", "refused", "dns", "unreachable")):
            return FailureMode.CONNECTION_ERROR
        if exc_type in (
            "ClientConnectionError", "ServerDisconnectedError",
            "ClientConnectorError",
        ):
            return FailureMode.CONNECTION_ERROR

        # Server errors
        if any(code in msg for code in ("500", "502", "503")):
            return FailureMode.SERVER_ERROR
        if exc_type == "ClientResponseError":
            status = getattr(exc, "status", 0)
            if status in (500, 502, 503):
                return FailureMode.SERVER_ERROR
            if status == 429:
                return FailureMode.RATE_LIMITED

        # Timeouts
        if isinstance(exc, (asyncio.TimeoutError,)):
            return FailureMode.TIMEOUT
        if "timeout" in msg:
            return FailureMode.TIMEOUT
        if exc_type in ("ServerTimeoutError", "ConnectionTimeoutError"):
            return FailureMode.TIMEOUT

        # Conservative default
        return FailureMode.TIMEOUT


# ---------------------------------------------------------------------------
# Deterministic L7 model resolution (awakened J-Prime failover node)
# ---------------------------------------------------------------------------
#
# The Phase 3c dispatch must name the model the awakened node ACTUALLY serves
# (loaded in VRAM). The old path read the FSM's ``_active_model_label``, which
# lags -- empty when the endpoint is found by direct GCP query -- so it fell back
# to the survival 7B and the node returned an error object with no "choices"
# (KeyError('choices')). The race-free source of truth is the node's own ollama
# ``/api/tags``. We query it ONCE per endpoint and memoize per-endpoint: a new
# endpoint (node changed / re-awaken at a new IP) is a natural cache miss;
# ``_invalidate_jprime_model_cache()`` clears it on FSM->DORMANT. No per-dispatch
# network spam.

_JPRIME_SERVED_MODEL_CACHE: Dict[str, str] = {}


# Sibling cache: the served model's on-disk BYTES (from the SAME /api/tags), used
# by the Context-Hardware Negotiator to derive the VRAM-safe num_ctx.
_JPRIME_SERVED_BYTES_CACHE: Dict[str, int] = {}


def _invalidate_jprime_model_cache() -> None:
    """Clear the memoized per-endpoint served-model maps (name + bytes). Called on
    FSM->DORMANT (the node is gone); a fresh awaken re-queries /api/tags.
    Per-endpoint keying already self-invalidates on a node/IP change -- this covers
    same-IP reuse."""
    _JPRIME_SERVED_MODEL_CACHE.clear()
    _JPRIME_SERVED_BYTES_CACHE.clear()


def _sibling_candidate_count() -> int:
    """How many candidates to draw per op on the local lane. Default 3.

    A DPO preference pair needs at least TWO answers to one question, so
    1 is the value at which the trajectory corpus can never produce a pair
    however long a soak runs. 3 gives a pair even when one sibling is a
    duplicate or fails to parse. Clamped to [1, 8]; 1 restores the exact
    single-candidate behaviour this lane had.

    The upper clamp is not decoration: siblings are sequential on one GPU,
    so a fat-fingered 300 would spend the whole op budget generating and
    leave nothing for VALIDATE -- which is where the per-candidate verdict
    that makes siblings worth having actually comes from.
    """
    return max(1, min(8, _envi_or_default("JARVIS_LOCAL_SIBLING_CANDIDATES", 3)))


def _sibling_budget_margin() -> float:
    """Safety factor on "can the op still afford another sibling?".

    Default 1.5. The estimate is the PREVIOUS sibling's measured cost, and
    the next one can legitimately run longer (a larger patch, a tool
    round). Starting a sibling the budget cannot finish wastes the whole
    generation AND the slack, so the margin is deliberately generous:
    skipping a sibling costs one training pair, overrunning costs the op.
    """
    return max(1.0, _envf_or_default("JARVIS_LOCAL_SIBLING_BUDGET_MARGIN", 1.5))


def _model_pin() -> str:
    """Operator's explicit choice of local model, or empty for auto."""
    return os.environ.get("JARVIS_LOCAL_MODEL_NAME", "").strip()


def _entry_name(entry: Optional[Dict[str, Any]]) -> str:
    return ((entry or {}).get("name") or (entry or {}).get("model") or "").strip()


def _select_served_entry(
    tags: Optional[Dict[str, Any]],
    *,
    pin: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Choose ONE entry from an ollama ``/api/tags`` payload.

    Single source of selection: both the served-model NAME and its on-disk
    BYTES are projections of this one choice, so they cannot disagree. They
    were separately-implemented ``max(size)`` scans before, and the bytes
    feed the num_ctx negotiator -- a divergence there would size the
    context window from a different model than the one being asked to
    generate.

    Selection order:

    1. **An explicit pin that the node actually serves.**
       ``JARVIS_LOCAL_MODEL_NAME`` names the model the operator wants.
       Matched exactly, then by base tag (``qwen3.8`` matches
       ``qwen3.8:27b``). The "actually serves" clause is load-bearing and
       preserved from the original: naming a model the node lacks is how
       the request comes back ``KeyError('choices')``.
    2. **Largest by size**, the previous behaviour, when no pin is set or
       the pin is not served.

    Why the pin has to win: the size heuristic encodes "one big model plus
    small sidecars", which held when this host served a 32B and a 7B. It
    does not hold once several LARGE models are on disk -- a 32B (19.85GB),
    a 30B MoE (18.56GB) and a 27B (18GB) -- where "largest" is an arbitrary
    property with no relationship to which model the operator selected, and
    silently makes a model A/B compare a model against itself.

    Pure + fail-soft -> None on empty/malformed input.
    """
    try:
        models = [m for m in ((tags or {}).get("models") or []) if m]
        if not models:
            return None
        want = (pin if pin is not None else _model_pin()).strip()
        if want:
            matched = _match_pin(models, want)
            if matched is not None:
                return matched
            logger.warning(
                "[CandidateGenerator] pinned local model %r is not served "
                "(node offers %s) -- falling back to largest-by-size. The "
                "BOOT GATE (`resolve_active_model`) refuses this outright; "
                "reaching here means a pin changed under a running process.",
                want, ", ".join(sorted(_entry_name(m) for m in models)) or "none",
            )
        return max(models, key=lambda m: (m or {}).get("size", 0) or 0)
    except Exception:  # noqa: BLE001
        return None


def _match_pin(
    models: Sequence[Dict[str, Any]], want: str,
) -> Optional[Dict[str, Any]]:
    """The entry a pin names, or None. THE matching rule, defined once.

    Exact tag first, then base tag (``qwen3.8`` matches ``qwen3.8:27b``).
    Extracted so the SELECTOR and the boot-time VALIDATOR cannot disagree
    about whether a pin is honoured -- a validator that admitted a pin the
    selector would then ignore is worse than no validator, because it
    certifies the substitution it was built to prevent. Pure; NEVER raises.
    """
    try:
        target = str(want or "").strip()
        if not target:
            return None
        for entry in models:
            if _entry_name(entry) == target:
                return entry
        base = target.split(":")[0]
        for entry in models:
            if _entry_name(entry).split(":")[0] == base:
                return entry
        return None
    except Exception:  # noqa: BLE001
        return None


class ModelPinUnavailable(RuntimeError):
    """The operator pinned a model this node does not serve.

    Carries both halves of the evidence so the caller can render an alert
    that is actionable without a second lookup: what was asked for, and
    what the node actually offers.
    """

    def __init__(self, pin: str, served: Sequence[str]) -> None:
        self.pin = str(pin)
        self.served = tuple(str(s) for s in served)
        super().__init__(
            f"pinned model {self.pin!r} is not served by this node "
            f"(offers: {', '.join(self.served) or 'nothing'})"
        )


#: The tag this process WILL dispatch to, resolved once against a registry
#: that answered. Empty until the boot gate runs — an empty string means
#: "not yet resolved", never "no model", so a surface can tell the two
#: apart instead of rendering a confident blank.
_ACTIVE_MODEL_TAG = ""


def resolve_active_model(
    tags: Optional[Dict[str, Any]], *, pin: Optional[str] = None,
) -> str:
    """The tag that will actually be dispatched to. FAIL-CLOSED on a pin.

    The selector above is fail-soft by contract, because a grader on the
    hot path must never stop a running loop. That is the right policy
    there and the wrong one at boot: a pin the node does not serve means
    every subsequent generation silently answers from a DIFFERENT model,
    and a model A/B then compares a model against itself. Measured on this
    host: with no pin, "largest by size" selects ``qwen2.5-coder:32b``
    (19.85 GB) over the fine-tuned ``qwen3-coder-ov:30b`` (18.58 GB) --
    the substitution is not even in the same family.

    So this raises rather than substituting. The caller decides what to do
    with that; the daemon's boot gate declines to start.

    Note what is NOT a fault here: an EMPTY or unreadable registry. That is
    "we could not ask", not "the model is absent", and the two must not
    share a verdict — the lane preflight already dies loudly when the
    engine cannot serve, and a second opinion here would turn a transient
    blip into a self-kill. Only a registry that ANSWERED and does not
    contain the pin is evidence.
    """
    models = [m for m in ((tags or {}).get("models") or []) if m]
    if not models:
        # Nothing to prove either way. Report the pin as the intent it is.
        return (pin if pin is not None else _model_pin()).strip()
    want = (pin if pin is not None else _model_pin()).strip()
    if want and _match_pin(models, want) is None:
        raise ModelPinUnavailable(
            want, sorted(_entry_name(m) for m in models),
        )
    return _entry_name(_select_served_entry(tags, pin=want)) or ""


def set_active_model_tag(tag: str) -> None:
    """Record the resolved tag for every surface that reports it."""
    global _ACTIVE_MODEL_TAG  # noqa: PLW0603
    _ACTIVE_MODEL_TAG = str(tag or "").strip()


def active_model_tag() -> str:
    """The tag this process resolved at boot, or "" if it has not yet.

    THE one answer to "which model is answering". The cockpit banner used
    to derive this from the CLIENT's own environment, which is a different
    process with a different environment — so a correctly pinned daemon
    rendered no model at all, and a stale client export would have
    rendered the wrong one confidently.
    """
    return _ACTIVE_MODEL_TAG


def resolve_display_model() -> str:
    """The model to NAME as the one answering — for the cockpit banner.

    ``active_model_tag`` is the ground truth once the boot gate has run
    against a registry that answered; but a registry BLIP at boot (the gate
    treats "could not ask" as not-proven and declines to set a tag) or an
    idle daemon that has not resolved yet would then render no model at all --
    the confusing blank the operator reported. So this falls back, DAEMON-SIDE
    and config-driven, to the generation lane's CONFIGURED model (the local
    lane's ``JARVIS_LOCAL_MODEL_NAME`` when that lane is enabled), then to the
    pin. Never the client's environment, never a hardcoded name. Empty only
    when there is genuinely no lane to name. NEVER raises.
    """
    tag = active_model_tag()
    if tag:
        return tag
    try:
        from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415,E501
            LocalConfig,
            local_prime_enabled,
        )
        if local_prime_enabled():
            name = (LocalConfig.from_env().model_name or "").strip()
            if name:
                return name
    except Exception:  # noqa: BLE001 -- a banner never breaks on a cold import
        pass
    try:
        return _model_pin().strip()
    except Exception:  # noqa: BLE001
        return ""


def _parse_served_model(tags: Optional[Dict[str, Any]]) -> Optional[str]:
    """From an ollama ``/api/tags`` payload, the model to dispatch to --
    the operator's pin when the node serves it, else the largest by
    ``size``. See :func:`_select_served_entry`. Pure + fail-soft."""
    return _entry_name(_select_served_entry(tags)) or None


async def _fetch_served_model(endpoint: str, *, timeout_s: float = 8.0) -> Optional[str]:
    """GET ``<endpoint>/api/tags`` -> the model the node has loaded. Bounded by
    *timeout_s*. Fail-soft -> None (node unreachable / non-200 / bad JSON)."""
    try:
        import aiohttp
        url = endpoint.rstrip("/") + "/api/tags"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
        return _parse_served_model(data)
    except Exception:  # noqa: BLE001
        return None


def _absolute_route_sealing(context: "Any") -> bool:
    """Absolute Route Sealing predicate. When TRUE, a J-Prime dispatch that was
    COMMITTED to (endpoint discovered) and then failed/empty must RAISE terminal --
    NEVER cascade to the DW/adversary-stub lane (the hybrid-mesh cascade leak).

    Two triggers:
      * ``context.provider_override == "gcp-jprime"`` -- a Cryo-DLQ pin: the op was
        SEALED for J-Prime, so a fall-through to dead DW would violate the seal
        (ALWAYS absolute, independent of any flag);
      * ``JARVIS_FAILOVER_ABSOLUTE_ROUTE_SEALING`` env flag -- arms sealing globally
        for the hybrid-execution-mesh soak.

    Default OFF -> byte-identical legacy fail-soft cascade. Fail-soft -> False."""
    try:
        override = (getattr(context, "provider_override", "") or "").strip()
    except Exception:  # noqa: BLE001
        override = ""
    if override == "gcp-jprime":
        return True
    return os.environ.get(
        "JARVIS_FAILOVER_ABSOLUTE_ROUTE_SEALING", "false"
    ).strip().lower() in ("1", "true", "yes", "on")


async def _resolve_served_model(
    endpoint: Optional[str],
    *,
    fetcher: "Optional[Any]" = None,
) -> Optional[str]:
    """Memoized per-endpoint served-model lookup. Fetches ONCE per endpoint via
    *fetcher* (defaults to :func:`_fetch_served_model`) then serves from cache --
    no per-dispatch spam. A ``None`` fetch result is NOT cached (transient node
    unreachable -> retried next dispatch). *fetcher* is injectable for tests."""
    if not endpoint:
        return None
    cached = _JPRIME_SERVED_MODEL_CACHE.get(endpoint)
    if cached is not None:
        return cached
    fn = fetcher or _fetch_served_model
    model = await fn(endpoint)
    if model:
        _JPRIME_SERVED_MODEL_CACHE[endpoint] = model
    return model


# ---------------------------------------------------------------------------
# CandidateGenerator
# ---------------------------------------------------------------------------


def _parse_served_model_bytes(tags: Optional[Dict[str, Any]]) -> int:
    """On-disk BYTES of the model :func:`_parse_served_model` names -- the
    SAME :func:`_select_served_entry` choice, not a second scan. These feed
    the num_ctx negotiator, so a divergence would size the context window
    from a model other than the one generating. Pure, fail-soft -> 0."""
    try:
        entry = _select_served_entry(tags)
        return int((entry or {}).get("size", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


async def _fetch_served_model_bytes(endpoint: str, *, timeout_s: float = 8.0) -> int:
    """GET <endpoint>/api/tags -> the served model's on-disk size in bytes.
    Fail-soft -> 0."""
    try:
        import aiohttp
        url = endpoint.rstrip("/") + "/api/tags"
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json(content_type=None)
        return _parse_served_model_bytes(data)
    except Exception:  # noqa: BLE001
        return 0


async def _resolve_served_model_bytes(
    endpoint: Optional[str],
    *,
    fetcher: "Optional[Any]" = None,
) -> int:
    """Memoized per-endpoint served-model byte size (Context-Hardware Negotiator
    input). Fetched once; a 0 result is not cached (retried). NEVER raises."""
    if not endpoint:
        return 0
    cached = _JPRIME_SERVED_BYTES_CACHE.get(endpoint)
    if cached:
        return cached
    fn = fetcher or _fetch_served_model_bytes
    size = await fn(endpoint)
    if size:
        _JPRIME_SERVED_BYTES_CACHE[endpoint] = int(size)
    return int(size or 0)


async def _await_jprime_ready(
    endpoint: str,
    *,
    probe_fn: "Optional[Any]" = None,
    op_id: str = "",
) -> bool:
    """Pre-SERVING dispatch readiness gate (bt-iso-1782959216): a COMMITTED
    sovereign dispatch fired while the node was still BOOTING died on the 30s
    survival probe and was SEALED TERMINAL (20 of 23 dispatches that run).
    Wait for the node's own /api/tags to answer (the same L7 truth the driver
    gate and the model resolver use) before dispatching:

      * bounded by ``JARVIS_JPRIME_DISPATCH_READY_BUDGET_S`` (600s -- covers a
        cold awaken) polling every ``_POLL_S`` (10s);
      * each poll PULSES the stream heartbeat -- waiting for the sovereign
        node IS activity (idle watchdog + audit-deferral probe stay fresh);
      * cooperative shutdown FREEZES the wait (GracefulStreamInterruption ->
        the dispatch's checkpoint boundary) so a suspend never burns budget;
      * returns False on budget expiry -> caller proceeds with the legacy
        attempt (no new failure mode, only added patience);
      * master ``JARVIS_JPRIME_DISPATCH_READY_ENABLED`` (default true);
        ``=false`` = legacy immediate dispatch, byte-identical.
    """
    if (os.environ.get("JARVIS_JPRIME_DISPATCH_READY_ENABLED", "true") or "").strip().lower() \
            in ("0", "false", "no", "off"):
        return True
    from backend.core.ouroboros.governance import cooperative_shutdown as _coop  # noqa: PLC0415
    from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
        GracefulStreamInterruption as _GSI,
        _emit_stream_token as _hb_pulse,
    )
    _probe = probe_fn or _resolve_served_model_bytes
    budget_s = float(os.environ.get("JARVIS_JPRIME_DISPATCH_READY_BUDGET_S", "600") or 600)
    poll_s = float(os.environ.get("JARVIS_JPRIME_DISPATCH_READY_POLL_S", "10") or 10)
    deadline = time.monotonic() + max(0.0, budget_s)
    first = True
    while True:
        if _coop.is_requested():
            raise _GSI(
                "cooperative shutdown (%s) while awaiting sovereign node readiness"
                % _coop.reason(), partial="",
            )
        try:
            if await _probe(endpoint):
                return True
        except Exception:  # noqa: BLE001 -- a probe error is just "not ready yet"
            pass
        if time.monotonic() >= deadline:
            logger.warning(
                "[CandidateGenerator] sovereign node NOT ready after %.0fs "
                "(endpoint=%s op=%s) -> proceeding with legacy attempt",
                budget_s, endpoint, op_id,
            )
            return False
        if first:
            logger.info(
                "[CandidateGenerator] sovereign node not ready yet (endpoint=%s "
                "op=%s) -> waiting up to %.0fs (poll=%.0fs, heartbeat-pulsed)",
                endpoint, op_id, budget_s, poll_s,
            )
            first = False
        _hb_pulse("")  # waiting for the node IS activity
        await asyncio.sleep(max(0.01, poll_s))


def _dilate_sovereign_deadline(deadline: "datetime", profiler: Any,
                               num_ctx: int = 0) -> "datetime":
    """Time-Dilated Sovereign Deadline (bt-iso-1782973775: resumed / late-
    cascade dispatches reached the sealed 32B seam with 8-13s of a spent route
    budget while a single streaming round costs 200-400s).

    A COMMITTED sovereign dispatch derives its runway from the node's OWN
    measured physics: ``now + expected_rounds x profiler.adaptive_timeout_ms``
    (EWMA-floored; cold = the heavy-mult ctx-scaled seed -- the GPU speeding
    up SHRINKS the dilation automatically), clamped by the operator's op
    envelope ``JARVIS_PIPELINE_TIMEOUT_S``. Only ever EXTENDS (max) -- a
    healthy deadline passes through untouched. Master
    ``JARVIS_TIME_DILATION_ENABLED`` (default true). NEVER raises."""
    try:
        if (os.environ.get("JARVIS_TIME_DILATION_ENABLED", "true") or "").strip().lower() \
                in ("0", "false", "no", "off"):
            return deadline
        if profiler is None:
            return deadline
        rounds = max(1, int(float(os.environ.get(
            "JARVIS_A1_MAX_AGENTIC_ROUNDS", "5") or 5)))
        est_ms = float(profiler.adaptive_timeout_ms(
            prompt_tokens=max(1, int(num_ctx or 4096) // 4)))
        runway_s = (est_ms / 1000.0) * rounds
        envelope_s = max(1.0, float(os.environ.get(
            "JARVIS_PIPELINE_TIMEOUT_S", "600") or 600))
        runway_s = min(runway_s, envelope_s)
        now = datetime.now(tz=timezone.utc)
        dilated = now + timedelta(seconds=runway_s)
        if dilated <= deadline:
            return deadline
        logger.info(
            "[CandidateGenerator] Time-Dilated Sovereign Deadline: remaining "
            "%.0fs -> %.0fs (rounds=%d x est=%.0fs, envelope<=%.0fs)",
            max(0.0, (deadline - now).total_seconds()), runway_s, rounds,
            est_ms / 1000.0, envelope_s,
        )
        return dilated
    except Exception:  # noqa: BLE001 -- dilation is protective, never fatal
        return deadline


def _local_vram_autodetect_enabled() -> bool:
    """Master switch for preferring a MEASURED local VRAM reading over the GCP
    provisioning spec. Default OFF -> the legacy spec-derived path, byte-identical."""
    return _envb("JARVIS_LOCAL_VRAM_AUTODETECT_ENABLED", False)


def _gateway_inflight_unification_enabled() -> bool:
    """Master for registering Phase 3c generations with the InferenceGateway's
    in-flight counter. Default TRUE: this is a correctness fix, and OFF
    reinstates the blind spot where an advisory pre-warm cannot see a running
    local generation. Kept as a flag purely so it is revocable without a
    revert."""
    return _envb("JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED", True)


@contextlib.asynccontextmanager
async def _f3c_null_bracket() -> "AsyncIterator[None]":
    """An async no-op bracket. ``contextlib.nullcontext`` only grew async
    support in 3.10 and this module targets 3.9+, so the fallback is explicit
    rather than version-dependent."""
    yield


def _f3c_inflight_bracket(endpoint: "Optional[str]") -> Any:
    """Async context manager marking a Phase 3c generation as in-flight on the
    gateway, or a null bracket when that is unavailable.

    Pure accounting -- it never touches the generation itself. Its only job is
    to make the gateway's view of "is this device busy?" true for the path that
    actually generates, so the advisory pre-warm's in-flight guard stops being
    blind. Fail-soft to the null bracket: an accounting failure must never cost
    an op. NEVER raises."""
    try:
        if endpoint and _gateway_inflight_unification_enabled():
            from .inference_gateway import get_default_gateway  # noqa: PLC0415
            return get_default_gateway().external_generation(endpoint)
    except Exception:  # noqa: BLE001
        pass
    return _f3c_null_bracket()


#: Monotonic stamp of the last credential re-read. Module-level because the
#: question ("does this host have a paid lane?") is process-wide, not per-op.
_FREE_LANE_CRED_REFRESH_AT: float = 0.0


def _free_lane_cred_ttl_s() -> float:
    """How stale a credential reading may be before ``.env`` is consulted again.
    Default 30s. 0 disables refresh (boot-time environment only)."""
    return _envf_or_default("JARVIS_FREE_LANE_CRED_TTL_S", 30.0)


def _refresh_paid_lane_credentials() -> None:
    """Re-read allowlisted provider credentials from ``.env`` into the process.

    WHY THIS IS NEEDED AT ALL. Reading ``os.environ`` on every call already makes
    the interlock reactive to any IN-PROCESS change -- but an operator cannot
    reach a running process that way. ``os.environ`` is per-process, so a shell
    export is invisible to an orchestrator already running, and
    ``env_bootstrap.load_env_once`` is idempotent by design, so editing ``.env``
    mid-soak is never re-read either. Without this, "keys added mid-soak revoke
    the free lane" would be a claim the code could not honour: the loop would
    keep treating a now-metered host as free.

    Composes ``credential_env_loader.load_provider_credentials`` rather than
    parsing ``.env`` here -- that module already owns the allowlist (which is
    exactly DOUBLEWORD_API_KEY / ANTHROPIC_API_KEY plus HF tokens), the
    explicit-export-wins precedence, and the never-log-secrets contract. A second
    parser would be a second policy about credentials, which is the last thing
    this repo needs two of.

    TTL-bounded: the answer changes at operator speed, not per-op, so a file
    stat per generation would be waste. NEVER raises."""
    global _FREE_LANE_CRED_REFRESH_AT  # noqa: PLW0603
    try:
        ttl = _free_lane_cred_ttl_s()
        if ttl <= 0:
            return
        now = time.monotonic()
        if (now - _FREE_LANE_CRED_REFRESH_AT) < ttl:
            return
        _FREE_LANE_CRED_REFRESH_AT = now
        from backend.core.ouroboros.aegis.credential_env_loader import (  # noqa: PLC0415
            load_provider_credentials,
        )
        # Note the asymmetry this inherits, and it is the SAFE one: the loader
        # never overwrites an explicit export, so a key can be ADDED mid-run
        # (revoking free-lane status, the cautious direction) but a stale key
        # cannot be silently cleared into free-lane status by editing a file.
        load_provider_credentials()
    except Exception:  # noqa: BLE001
        pass


#: Parse-error messages that mean "the output stopped early", not "the model
#: mistyped". CPython's wording for each is stable and is the only signal
#: available at the parse seam — the generator cannot see its own token budget.
_TRUNCATION_SIGNATURES: Tuple[str, ...] = (
    "unterminated string literal",
    "unterminated triple-quoted string literal",
    "unexpected eof while parsing",
    "was never closed",
    "expected an indented block",
)


def _truncation_shaped(failures: Any) -> bool:
    """True when the parse failures look like a cut-off payload.

    Deliberately conservative: ANY failure bearing a truncation signature
    makes the whole retry truncation-shaped. Reshaping a retry that did not
    need it costs a smaller output; NOT reshaping one that did costs the op,
    because a whole-file retry hits the same ceiling. Asymmetric, so it errs
    toward the cheaper mistake. NEVER raises.
    """
    try:
        for fail in failures or ():
            msg = str((fail or {}).get("message", "")).lower()
            if any(sig in msg for sig in _TRUNCATION_SIGNATURES):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _truncation_reshape_enabled() -> bool:
    """Master for coupling truncation-shaped parse failures to a reduced
    output shape. Falsey retries whole-file with feedback only (the
    pre-coupling behaviour)."""
    return _envb("JARVIS_TRUNCATION_RESHAPE_ENABLED", True)


def agent_client_from(providers, *, injected=None):
    """The provider CLIENT an agent turn speaks to, or None.

    ``ProductionAgentTurnFn`` wants a *client* -- something with
    ``async generate(prompt=..., system_prompt=..., model_name=...,
    task_profile=...)``: a ``PrimeClient`` / ``LocalPrimeClient``. Callers
    hold PROVIDERS (``generate(context, deadline)``); the client lives one
    level down. This is the ONE place that knows where, so the GENERATE
    swarm and the L2 repair path can never disagree about it -- the swarm
    wire once read a ``_client`` no constructor set and took every
    generation down with it (2026-09-05).

    ``injected`` wins when it carries a callable ``generate`` (the test
    seam, kept honest by name). Then each provider in order, probing the
    known client attributes. None means "no agent brain here"; the caller
    DECLINES and its standard route runs byte-identical. Never raises.
    """
    if injected is not None and callable(getattr(injected, "generate", None)):
        return injected
    for provider in providers:
        if provider is None:
            continue
        for path in (("_client",), ("_state", "client"), ("_prime_client",), ("client",)):
            obj = provider
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and callable(getattr(obj, "generate", None)):
                return obj
    return None


def _declared_symbols_for(context: Any, file_path: str) -> Tuple[str, ...]:
    """Operator-declared repair targets for *file_path*, from the SIGNED goal.

    Re-derived from ground truth on every call — the op's evidence contributes
    only a POINTER (``goal_id``), exactly as `verify_provenance_claim` treats
    it. Reading the symbol list off the context instead would let a fabricated
    or hallucinated field name a target the operator never authorised, which
    is the forgery class Slice 20 exists to prevent.

    Scope is enforced twice over: the goal must be signature-valid, and the
    file must be inside that goal's own ``target_files``. A declaration cannot
    reach a file the mandate does not cover.

    Returns () for every op without a verified roadmap claim — which is almost
    all of them — so the resolver's inference cascade is unchanged for
    everything else. NEVER raises: a declaration that cannot be proven is
    simply absent, and absence degrades to inference.
    """
    try:
        claim = None
        evidence = getattr(context, "evidence", None)
        if isinstance(evidence, Mapping):
            claim = evidence.get("provenance")
        if not isinstance(claim, Mapping):
            return ()
        goal_id = str(claim.get("goal_id", "")).strip()
        if not goal_id:
            return ()

        from backend.core.ouroboros.governance.delegated_provenance import (
            _verified_roadmap,  # noqa: PLC0415 — REPORTS a verdict; see below
        )
        # (verdict, document) — NOT the other way round. Getting this
        # backwards yields a verdict object with no `.goals`, and the broad
        # except below would have swallowed the AttributeError into a silent
        # empty result: declared symbols would simply never work, with nothing
        # in the log to say why.
        _verdict, doc = _verified_roadmap()
        # It RETURNS THE DOCUMENT REGARDLESS OF VERDICT — it reports
        # verification, it does not enforce it. Gating on `doc is None` alone
        # would honour a roadmap whose signature is invalid, tampered or absent
        # and hand back its symbols at confidence 1.0 — precisely the forgery
        # this function's pointer-only contract exists to prevent, and it would
        # silently defeat an operator's JARVIS_ROADMAP_READER_REQUIRE_SIGNATURE.
        # Demand BOTH properties, the same pair `delegated_provenance` demands
        # of this same call: the cryptographic fact AND the verdict, because the
        # reader permits an unsigned dev-mode (REQUIRE_SIGNATURE=false) that
        # must never confer a target.
        if doc is None or not bool(getattr(doc, "signature_valid", False)):
            return ()
        if str(getattr(_verdict, "value", _verdict)) != "valid":
            return ()
        for goal in getattr(doc, "goals", ()) or ():
            if getattr(goal, "goal_id", "") != goal_id:
                continue
            targets = tuple(getattr(goal, "target_files", ()) or ())
            if targets and not any(
                str(file_path).replace("\\", "/").endswith(
                    str(t).replace("\\", "/").lstrip("./")
                )
                for t in targets
            ):
                return ()  # declaration does not cover this file
            return tuple(getattr(goal, "target_symbols", ()) or ())
    except Exception:  # noqa: BLE001 — unprovable ⇒ infer, never fabricate
        return ()
    return ()


def _local_primary_enabled() -> bool:
    """Master for serving ANY route on the local lane when no paid lane exists.

    Default ON, but the master alone changes nothing: `_try_local_primary`
    still requires `_free_lane_active()` (local configured AND no provider
    credentials) plus an endpoint that answers. A host with a paid key is
    byte-identical either way. Falsey pins the local lane back to a fallback
    for BACKGROUND/SPECULATIVE only.
    """
    return _envb("JARVIS_LOCAL_PRIME_PRIMARY_ENABLED", True)


def _syntax_repair_enabled() -> bool:
    """Master for the local lane's one-shot syntax-repair retry.

    Default ON: without it `all_candidates_syntax_error` is terminal on a
    topology whose escalation target is the failing model itself. Falsey
    restores the pre-existing single-attempt behaviour exactly.
    """
    return _envb("JARVIS_LOCAL_SYNTAX_REPAIR_ENABLED", True)


def _background_local_lane_enabled() -> bool:
    """Master for the cost-optimized routes' zero-cost lane preemption.

    Module-level, not a method, so the arc's flag table can bind it as the
    accessor that proves the registered default is the default the code
    applies. See :meth:`CandidateGenerator._try_free_lane_dispatch` for what
    it gates. NEVER raises -- an unreadable environment keeps the feature on,
    because the feature's own evidence gate (a reachable endpoint) is what
    actually decides, and it fails closed on its own.
    """
    return _envb("JARVIS_BACKGROUND_LOCAL_LANE_ENABLED", True)


def _free_lane_active() -> bool:
    """True when generation runs on a lane with ~zero marginal cost per op.

    Exists so cost-motivated gates can ask about COST rather than hardcode a
    route name. Several policies in this file trade quality away to save money
    -- correct against a metered provider, and pure loss on a locally-served
    model where the only per-op cost is electricity.

    Deliberately conservative: it requires the local lane to be the configured
    one AND both paid lanes to be unavailable. A host with credentials might
    still route to a paid provider, and treating that as free would silently
    multiply someone's bill. Absence of a key is the strongest available
    evidence that no paid lane exists.

    NEVER raises -- an unprovable answer is False, which preserves the
    pre-existing cost-averse behaviour exactly."""
    try:
        if not _envb("JARVIS_FREE_LANE_POLICY_ENABLED", True):
            return False
        from .local_inference_director import local_prime_enabled  # noqa: PLC0415
        if not local_prime_enabled():
            return False
        # Consult .env before answering, so a key added while the loop is
        # RUNNING revokes free-lane status without a restart. TTL-bounded.
        _refresh_paid_lane_credentials()
        if os.environ.get("DOUBLEWORD_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


async def _f3c_gateway_residency(
    endpoint: str, model: str, context: Any = None,
) -> "Optional[Dict[str, Any]]":
    """Run the InferenceGateway's residency handshake for a Phase 3c dispatch.

    Gives the live local path the gateway's swap mutex, capacity admission and
    warm-swap budget without moving the generation itself. The route is passed
    through so the gateway's down-route policy can decline to evict for
    eviction-averse work; a real dispatch is never advisory, so it will swap if
    it must.

    Returns the gateway's report for telemetry, or None when the handshake did
    not run. NEVER raises -- residency is an optimisation over Ollama's own
    load-on-first-use, so a failure here degrades to the pre-existing behaviour
    rather than costing an op."""
    try:
        if not _gateway_inflight_unification_enabled():
            return None
        from .inference_gateway import get_default_gateway  # noqa: PLC0415
        gw = get_default_gateway()
        target = gw.target_for_endpoint(endpoint, model)
        route = ""
        try:
            route = str(getattr(context, "provider_route", "") or "").lower()
        except Exception:  # noqa: BLE001
            route = ""
        report = await gw.ensure_model_resident(target, route=route or None)
        if report and report.get("swapped"):
            logger.info(
                "[CandidateGenerator] gateway warm-swap completed before Phase 3c "
                "dispatch: model=%s endpoint=%s (paid outside the op clock)",
                model, endpoint,
            )
        return report
    except Exception:  # noqa: BLE001
        return None


def _measured_local_vram_bytes() -> int:
    """VRAM (bytes) MEASURED on this host, via the existing compute-topology
    probe. 0 when the probe is disabled, unavailable, or returned an absence.

    Composes ``compute_topology.resolve_sync()`` -- the module that already owns
    the ``nvidia-smi`` cascade, its timeout discipline and its caching. A second
    probe here would be a second authority over the same physics. NEVER raises."""
    try:
        from . import compute_topology  # noqa: PLC0415
        if not compute_topology.is_enabled():
            return 0
        reading = compute_topology.resolve_sync()
        if not getattr(reading, "measured", False):
            return 0
        return max(0, int(getattr(reading, "total_bytes", 0) or 0))
    except Exception:  # noqa: BLE001 -- descriptive helper must never raise
        return 0


def _awakened_vram_bytes() -> int:
    """VRAM (bytes) of the serving GPU -- MEASURED on this host when possible,
    else the awakened tier, else the QUALITY provisioning spec. 0 if not a GPU
    tier / unknown (the negotiator then keeps the legacy path). NEVER raises.

    WHY MEASURED COMES FIRST. The spec path answers "what did we ASK GCP for",
    which is the right question for a provisioned failover node and the WRONG
    one for an operator workstation serving Ollama locally. There, no controller
    tier is awake, so resolution fell through to ``_quality_tier()`` whose
    accelerator default is ``nvidia-l4`` -- and a local 32 GiB card was sized as
    24 GiB. Not an absence the negotiator could floor on, but a confident wrong
    number that silently halved the derived context window. A reading of the
    actual hardware outranks a guess about it; the spec chain is retained
    verbatim beneath as the fallback for genuinely remote tiers."""
    if _local_vram_autodetect_enabled():
        measured = _measured_local_vram_bytes()
        if measured > 0:
            return measured
    accel = ""
    try:
        from .failover_lifecycle import get_failover_controller  # noqa: PLC0415
        _t = getattr(get_failover_controller(), "_awakened_tier", None)
        accel = (getattr(_t, "accelerator_type", "") or "").strip()
    except Exception:  # noqa: BLE001
        accel = ""
    if not accel:
        try:
            from .failover_tier import _quality_tier  # noqa: PLC0415
            accel = (getattr(_quality_tier(), "accelerator_type", "") or "").strip()
        except Exception:  # noqa: BLE001
            accel = ""
    try:
        from .failover_tier import accelerator_vram_bytes  # noqa: PLC0415
        return accelerator_vram_bytes(accel)
    except Exception:  # noqa: BLE001
        return 0


# --- Resilient L7 Recovery (auto-heal on connection drop) -------------------

def _l7_recovery_attempts() -> int:
    """Extra retries after a recoverable L7 failure (ServerDisconnect) before the
    dispatch raises (-> sentinel seam seals). Default 2. NEVER raises."""
    try:
        return max(0, int(os.environ.get("JARVIS_FAILOVER_L7_RECOVERY_ATTEMPTS", "2")))
    except (TypeError, ValueError):
        return 2


def _l7_tighten_factor() -> float:
    """num_ctx shrink factor applied on each auto-heal retry (more aggressive
    compression). Default 0.6. Clamped to (0,1). NEVER raises."""
    try:
        f = float(os.environ.get("JARVIS_FAILOVER_L7_TIGHTEN_FACTOR", "0.6"))
        return f if 0.0 < f < 1.0 else 0.6
    except (TypeError, ValueError):
        return 0.6


def _l7_rewarm_timeout_s() -> float:
    try:
        return max(1.0, float(os.environ.get("JARVIS_FAILOVER_L7_REWARM_TIMEOUT_S", "120")))
    except (TypeError, ValueError):
        return 120.0


def _failover_keep_alive_seconds() -> int:
    """Deterministic VRAM residency: the keep_alive every failover dispatch passes
    so ollama keeps the model RESIDENT while we're routing to the node (no ~109s
    reload between ops). Default -1 (keep forever while SERVING); the FSM's
    ``_reap_gpu_node`` fires the explicit ``keep_alive:0`` flush on teardown. Env
    ``JARVIS_FAILOVER_KEEP_ALIVE_SECONDS``. NEVER raises."""
    try:
        return int(os.environ.get("JARVIS_FAILOVER_KEEP_ALIVE_SECONDS", "-1"))
    except (TypeError, ValueError):
        return -1


_L7_RECOVERABLE_NAMES = frozenset({
    "ServerDisconnectedError", "ClientConnectionError", "ClientConnectionResetError",
    "ClientOSError", "ClientPayloadError", "ConnectionResetError",
    "ConnectionError", "ConnectionAbortedError", "ServerTimeoutError",
})


def _is_l7_recoverable(exc: BaseException) -> bool:
    """True iff *exc* is a transient connection-drop the auto-heal can retry (a
    warm worker that dropped mid-request -- e.g. a KV-cache OOM crash). A logic
    error (ValueError, KeyError) is NOT recoverable. The Absolute Global Circuit
    Breaker (``UnrecoverableInferenceLatency``) is explicitly NON-recoverable --
    retrying a wedged model just re-inflates + bills; it must seal/halt. NEVER
    raises."""
    try:
        name = type(exc).__name__
        if name == "UnrecoverableInferenceLatency":
            return False
        if isinstance(exc, (ConnectionError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if name in _L7_RECOVERABLE_NAMES:
            return True
        return "disconnect" in str(exc).lower()
    except Exception:  # noqa: BLE001
        return False


# ──────────────────────────────────────────────────────────────────────
# Dynamic 5xx Resiliency Matrix — DW transient-network absorb loop (2026-07-22)
# ──────────────────────────────────────────────────────────────────────
#
# A transient DoubleWord blip (``upstream_error`` in a 400 body, any 5xx,
# gateway timeout, or a 429 that carries ``Retry-After``) must be absorbed by a
# bounded exponential-backoff-with-jitter retry on the PRIMARY generate call —
# BEFORE the loop cascades to a (possibly dead) fallback and BEFORE the session
# breaker can trip terminally. The empirical foil is bt-2026-07-22-082657, where
# ONE transient ``upstream_error`` was mis-labeled ``terminal_quota`` and killed
# the whole soak. The classification lives in ``provider_retry_classifier`` and
# the jitter primitive is reused from ``circuit_breaker.full_jitter_delay``
# (DRY — no new jitter maths here).


def _dw_transient_max_retries() -> int:
    """Bounded absorb-loop budget for a DW TRANSIENT_NETWORK blip on the
    primary generate call (env ``JARVIS_DW_TRANSIENT_MAX_RETRIES``, default 2).
    The breaker's own window still caps the worst case. Fail-soft to 2."""
    try:
        return max(0, int(os.environ.get("JARVIS_DW_TRANSIENT_MAX_RETRIES", "2")))
    except (TypeError, ValueError):
        return 2


def _is_dw_transient_network(exc: Exception, http_status, retry_after_ts) -> bool:
    """True when *exc* classifies TRANSIENT_NETWORK per the Dynamic 5xx
    Resiliency Matrix (upstream_error / 5xx / gateway timeout / 429-with-
    Retry-After). Uses the canonical taxonomy — never raises."""
    try:
        from backend.core.ouroboros.governance.provider_retry_classifier import (
            classify, RetryDecision,
        )
        decision = classify(
            failure_class=type(exc).__name__,
            http_status=http_status,
            failure_message=str(exc),
            retry_after_present=retry_after_ts is not None,
        )
        return decision is RetryDecision.TRANSIENT_NETWORK
    except Exception:  # noqa: BLE001 — classification never blocks the caller
        return False


def _dw_transient_backoff_s(attempt: int, retry_after_ts, *, remaining_s: float) -> float:
    """Backoff before a DW transient retry. Prefers a server-directed
    ``Retry-After`` timestamp when present, else AWS full-jitter exponential
    backoff (reused from circuit_breaker). Clamped to a quarter of the
    remaining op budget and a hard 30s ceiling. Async-sleepable; never raises."""
    try:
        base_s = float(os.environ.get("JARVIS_DW_TRANSIENT_BACKOFF_BASE_S", "1.5"))
    except (TypeError, ValueError):
        base_s = 1.5
    try:
        cap_s = float(os.environ.get("JARVIS_DW_TRANSIENT_BACKOFF_CAP_S", "12.0"))
    except (TypeError, ValueError):
        cap_s = 12.0
    delay: Optional[float] = None
    try:
        if retry_after_ts is not None:
            wait = float(retry_after_ts) - time.time()
            if wait > 0:
                delay = wait
    except (TypeError, ValueError):
        delay = None
    if delay is None:
        try:
            from backend.core.ouroboros.governance.circuit_breaker import (
                full_jitter_delay,
            )
            delay = full_jitter_delay(attempt, base_s=base_s, cap_s=cap_s)
        except Exception:  # noqa: BLE001 — degrade to plain expo if helper absent
            delay = min(cap_s, base_s * (2 ** max(0, attempt)))
    budget_clamp = (
        max(0.1, remaining_s * 0.25) if remaining_s and remaining_s > 0 else cap_s
    )
    return max(0.1, min(delay, budget_clamp, 30.0))


try:
    from backend.core.ouroboros.governance.transient_absorb import (
        with_transient_absorb as _with_transient_absorb,
    )
except Exception:  # noqa: BLE001 — decorator is resilience; degrade to identity
    def _with_transient_absorb(**_kw):  # type: ignore[misc]
        def _identity(fn):
            return fn
        return _identity


class CandidateGenerator:
    """Orchestrates candidate generation with failover and concurrency control.

    Routes generation requests to the primary provider when healthy, falling
    back to the fallback provider on failure.  Each provider has its own
    :class:`asyncio.Semaphore` for concurrency limiting.

    Parameters
    ----------
    primary:
        The preferred (typically remote/powerful) generation provider.
    fallback:
        The backup (typically local/smaller) generation provider.
    primary_concurrency:
        Maximum concurrent calls to the primary provider.
    fallback_concurrency:
        Maximum concurrent calls to the fallback provider.
    """

    def __init__(
        self,
        primary: CandidateProvider,
        fallback: Optional[CandidateProvider] = None,
        primary_concurrency: int = 4,
        fallback_concurrency: int = 2,
        tier0: Optional[Any] = None,  # DoublewordProvider (batch, async)
        ledger: Optional[Any] = None,  # OperationLedger for batch traceability
        latency_tracker: Optional[DwLatencyTracker] = None,
        exhaustion_watcher: Optional[Any] = None,  # ProviderExhaustionWatcher
        jprime: Optional[Any] = None,  # PrimeProvider (Phase 3 Scope α primacy)
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._tier0 = tier0
        # Phase 3 Scope α — J-Prime primacy handle. Only consulted from
        # the BACKGROUND and SPECULATIVE dispatch paths, and only when
        # ``JARVIS_JPRIME_PRIMACY=true``. Can be ``None`` — primacy is
        # opt-in and test fixtures often don't build a PrimeProvider.
        # When the caller doesn't hand one in but ``self._primary`` is
        # already a PrimeProvider (the usual production wiring), we
        # detect and reuse it below to keep the API minimal for
        # existing call sites that don't know about Scope α yet.
        self._jprime = jprime
        if self._jprime is None and primary is not None and getattr(
            primary, "provider_name", ""
        ) == "gcp-jprime":
            self._jprime = primary
        self._ledger = ledger
        self._fallback_concurrency = fallback_concurrency
        # HIBERNATION_MODE step 5: optional watcher that counts
        # consecutive all_providers_exhausted raises and transitions
        # the SupervisorOuroborosController into HIBERNATION at the
        # configured threshold. Kept structural/optional so unit tests
        # of CandidateGenerator don't need to build a controller.
        self._exhaustion_watcher = exhaustion_watcher
        # Cap concurrent background polls to avoid connector exhaustion
        self._max_background_polls: int = 3
        # Per-op tier rotation config: belt-and-suspenders for the FSM
        # ETA-based skip. When the classifier mis-routes (e.g. an
        # unfamiliar wrapper exception falls through to TIMEOUT instead
        # of TRANSIENT_TRANSPORT) the FSM's `should_attempt_primary()`
        # keeps returning True and consecutive ops all hit the same
        # dead Tier 0. Once N failures land within W seconds, we
        # hard-skip Tier 0 for the next op regardless of FSM mode —
        # buying the human one cheap Claude success while DW recovers.
        self._tier0_skip_threshold: int = int(
            os.environ.get("OUROBOROS_TIER0_SKIP_THRESHOLD", "2")
        )
        self._tier0_skip_window_s: float = float(
            os.environ.get("OUROBOROS_TIER0_SKIP_WINDOW_S", "30")
        )

        # AdmissionGate Slice 2 — per-route rolling EWMA of
        # observed _fallback_sem wait times. Feeds the
        # admission gate's projected_wait_s input. Updated
        # post-acquire in _call_fallback after every successful
        # sem.acquire(). Master flag default-FALSE until Slice 3
        # graduation, so the gate is constructed but doesn't
        # change behavior — pre-Slice-2 path preserved when
        # disabled.
        try:
            from backend.core.ouroboros.governance.admission_estimator import (  # noqa: E501
                WaitTimeEstimator as _WaitTimeEstimator,
            )
            self._wait_estimator = _WaitTimeEstimator()
        except Exception:  # noqa: BLE001 — defensive
            self._wait_estimator = None

        # ── Phase 1 Step 3A: state hoist (un-quarantine blueprint) ──
        # Invariant: every mutable field that must survive
        # `importlib.reload(candidate_generator)` lives on `self._state`
        # (a ``GeneratorState``), not on ``self`` directly. The aliases
        # below are bound once in __init__ and share reference identity
        # with the state container — for dicts/FSM/sem that is enough;
        # the ``int``/``float`` counters live on ``self._counters`` (a
        # ``GeneratorCounters`` dataclass) so mutation-via-attribute
        # does not re-bind a local copy. Do NOT add new mutable fields
        # as ``self._*`` — extend ``GeneratorState`` instead.
        #
        # When ``JARVIS_UNQUARANTINE_GENERATOR`` is false (default), the
        # state is minted fresh per instance so today's tests and
        # production behavior stay bit-identical. Flipping the env to
        # true routes every new ``CandidateGenerator`` to the shared
        # singleton and retires the quarantine (follow-up PR).
        from ._governance_state import (
            GeneratorState,
            get_generator_state,
            unquarantine_generator_enabled,
        )
        if unquarantine_generator_enabled():
            self._state = get_generator_state(
                primary_concurrency=primary_concurrency,
                fallback_concurrency=fallback_concurrency,
                latency_tracker=latency_tracker,
            )
        else:
            self._state = GeneratorState.fresh(
                primary_concurrency=primary_concurrency,
                fallback_concurrency=fallback_concurrency,
                latency_tracker=latency_tracker,
            )
        # Aliases: all share reference identity with self._state so
        # reads AND writes via either name land on the same object.
        # Safe for Semaphores, FSM, dicts, trackers (objects / mutable
        # containers).
        self._primary_sem = self._state.primary_sem
        self._fallback_sem = self._state.fallback_sem
        self.fsm = self._state.fsm
        # Manifesto §5: rolling p95 DW RT latency → dynamic Tier 0 budget.
        # Cold endpoints get full ceiling, hot endpoints dial down aggressively.
        self._latency_tracker = self._state.latency_tracker
        # Async Tier 0 tracking: op_id → CompletedBatch (dict aliased).
        self._completed_batches: dict[str, Any] = self._state.completed_batches
        # Background polling tasks (kept to prevent GC; dict aliased).
        self._background_polls: dict[str, asyncio.Task[Any]] = (
            self._state.background_polls
        )
        # Counters container: lets ``self._counters.exhaustion_events +=
        # 1`` mutate the same dataclass instance stored on the state,
        # which a plain ``int`` alias could not. Do not rebind
        # ``self._counters`` — only mutate its fields.
        self._counters = self._state.counters

        # ── Phase 3 Scope α: J-Prime primacy state (process-lifetime) ──
        # The ``jprime_sem`` (Semaphore(1)) and ``model_stickiness``
        # placeholder MUST live on the hoisted ``JPrimeState`` even when
        # ``JARVIS_JPRIME_PRIMACY`` is off today — same binding
        # discipline as 3A/3B. Per Derek-locked middle path: never place
        # these roots on a hot ``CandidateGenerator`` instance, because
        # ``importlib.reload(candidate_generator)`` would silently reset
        # the client-side concurrency ceiling and let a burst hit the
        # 50-slot swap-transient queue at the server edge.
        #
        # ``get_jprime_state()`` is first-call-wins, so every generator
        # post-reload sees the same sem token and the same stickiness
        # dict. The alias here is reference-stable: the sem identity
        # never changes, so binding once is enough. The counters
        # container mutates in place for the same reason as
        # ``self._counters`` above.
        from ._governance_state import get_jprime_state
        self._jprime_state = get_jprime_state()
        self._jprime_sem = self._jprime_state.jprime_sem
        self._jprime_counters = self._jprime_state.counters

    def _raise_exhausted(
        self,
        cause: str,
        *,
        context: Optional[Any] = None,
        deadline: Optional[datetime] = None,
        primary_exc: Optional[BaseException] = None,
        fallback_exc: Optional[BaseException] = None,
        **breadcrumbs: Any,
    ) -> NoReturn:
        """Log a structured exhaustion breadcrumb line and raise RuntimeError.

        Never returns. Every raise of ``all_providers_exhausted`` from this
        class should go through this helper so the battle-test audit can
        grep a single log line and learn:

            * which cause fired (queue_only_dispatch, fallback_failed, ...)
            * the FailbackStateMachine state at that moment
            * the classified ``FailureMode`` of the most recent attempt
            * the route, op_id, complexity, and remaining deadline budget
            * the primary / fallback provider names
            * the underlying exception class + trimmed message (if any)
            * any cause-specific breadcrumbs passed as ``**breadcrumbs``

        The raised ``RuntimeError`` carries the full report dict as
        ``.exhaustion_report`` so downstream observers (orchestrator
        postmortem, ProviderExhaustionWatcher, ledger) can use it
        without re-parsing the log line.

        The exception message remains the stable ``"all_providers_exhausted"``
        prefix plus a ``:{cause}`` suffix, so every existing substring /
        regex match (``"all_providers_exhausted" in str(exc)``,
        ``pytest.raises(RuntimeError, match="all_providers_exhausted")``,
        and the orchestrator ``_INFRA_PATTERNS`` set) keeps working.
        """
        self._counters.exhaustion_events += 1
        try:
            # Slice 197 — durable charter counter: the graduation contract
            # reads provider exhaustions from the registry, not from logs.
            from backend.core.ouroboros.governance.observability_registry import (
                record_provider_exhaustion as _s197_record_exhaustion,
            )
            _s197_record_exhaustion()
        except Exception:  # noqa: BLE001
            pass

        fm = self.fsm._failure_mode
        report: Dict[str, Any] = {
            "event_n": self._counters.exhaustion_events,
            "cause": cause,
            "fsm_state": self.fsm.state.name,
            "fsm_failure_mode": fm.name if fm is not None else "NONE",
            "fsm_consecutive_failures": self.fsm._consecutive_failures,
            "tier0_consecutive_failures": self._counters.consecutive_tier0_failures,
            "primary_name": getattr(self._primary, "provider_name", "?"),
            "fallback_name": (
                getattr(self._fallback, "provider_name", "?")
                if self._fallback is not None else "none"
            ),
            "tier0_name": (
                getattr(self._tier0, "provider_name", "?")
                if self._tier0 is not None else "none"
            ),
        }
        if context is not None:
            report["op_id"] = (
                getattr(context, "op_id", None)
                or getattr(context, "operation_id", None)
                or "?"
            )
            report["route"] = getattr(context, "provider_route", "?") or "?"
            report["complexity"] = (
                getattr(context, "task_complexity", "?") or "?"
            )
        if deadline is not None:
            report["remaining_s"] = round(self._remaining_seconds(deadline), 2)
        if primary_exc is not None:
            report["primary_err_class"] = type(primary_exc).__name__
            report["primary_err_msg"] = _trim_exc_msg(primary_exc)
        if fallback_exc is not None:
            report["fallback_err_class"] = type(fallback_exc).__name__
            report["fallback_err_msg"] = _trim_exc_msg(fallback_exc)
        report.update(breadcrumbs)

        log_parts = " ".join(
            f"{k}={_fmt_val(v)}" for k, v in report.items()
        )
        logger.error(
            "[CandidateGenerator] EXHAUSTION %s", log_parts,
        )

        # Rehearsal tier: classify this exhaustion ONCE, here, where the
        # evidence already exists. Every op that follows can then read the
        # verdict instead of re-walking the cascade to rediscover it —
        # detection and consumption were disconnected, which is why
        # bt-2026-08-11-230412 paid the full chain eight times for an
        # outage the ledger recorded on the first.
        #
        # Deliberately does NOT alter control flow: this helper is typed
        # NoReturn and every caller depends on that. The verdict rides on
        # the report and the exception so downstream surfaces can act on
        # it without a second classifier.
        _rehearsal = None
        try:
            from backend.core.ouroboros.governance.rehearsal_tier import (
                get_rehearsal_tier as _get_rehearsal_tier,
            )
            _rehearsal = _get_rehearsal_tier().consult(
                str(getattr(context, "op_id", "") or ""),
                report=report,
                target_files=tuple(getattr(context, "target_files", ()) or ()),
                route=str(report.get("route", "") or ""),
            )
            report["rehearsal"] = _rehearsal.to_dict()
        except Exception:  # noqa: BLE001 — never mask the raise
            pass

        err = RuntimeError(f"all_providers_exhausted:{cause}")
        try:
            setattr(err, "exhaustion_report", report)
            if _rehearsal is not None:
                setattr(err, "rehearsal", _rehearsal)
        except Exception:
            pass  # attribute attachment is best-effort — never mask the raise
        if fallback_exc is not None:
            raise err from fallback_exc
        if primary_exc is not None:
            raise err from primary_exc
        raise err

    async def generate(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """Generate candidate code changes, with automatic failover.

        Thin wrapper around :meth:`_generate_dispatch` that notifies
        the optional :class:`ProviderExhaustionWatcher` on the way out
        so the watcher can flip the controller into HIBERNATION once
        exhaustion events cross the configured threshold.

        Parameters
        ----------
        context:
            The operation context describing what needs to change.
        deadline:
            Absolute UTC deadline by which generation must complete.

        Returns
        -------
        GenerationResult
            The generated candidates from whichever provider succeeded.

        Raises
        ------
        RuntimeError
            If all providers are exhausted (``"all_providers_exhausted"``).
        asyncio.TimeoutError
            If the deadline is already past and no provider can be tried.
        """
        try:
            result = await self._generate_dispatch(context, deadline)
        except RuntimeError as exc:
            if "all_providers_exhausted" in str(exc):
                if self._exhaustion_watcher is not None:
                    try:
                        await self._exhaustion_watcher.record_exhaustion(
                            reason=str(exc),
                            op_id=getattr(context, "op_id", None) or None,
                        )
                    except Exception:
                        logger.debug(
                            "[CandidateGenerator] exhaustion_watcher "
                            "record_exhaustion failed",
                            exc_info=True,
                        )
                # Feed github_issue-sourced exhaustions into the sensor-side
                # cooldown registry so chronic unresolvable issues (e.g.
                # #16501 "Unlock Test Suite Failed" observed re-exhausting
                # across bt-2026-04-15-012736 and bt-2026-04-15-013455)
                # don't re-emit on the next scan, re-enter generation, and
                # re-exhaust — each such re-exhaustion currently counts
                # toward ExhaustionWatcher's global hibernation threshold
                # even when the reflex path is healthy. The registry is
                # module-level in the sensor file; env gate
                # JARVIS_GITHUB_ISSUE_EXHAUSTION_COOLDOWN_S (default 900s,
                # set to 0 to disable). issue_key parsing is delegated to
                # issue_key_from_description so the returned key stays
                # byte-identical to the sensor's own dedup_key.
                if getattr(context, "signal_source", "") == "github_issue":
                    try:
                        from backend.core.ouroboros.governance.intake.sensors.github_issue_sensor import (
                            issue_key_from_description,
                            register_issue_exhaustion,
                        )
                        _desc = getattr(context, "description", "") or ""
                        _issue_key = issue_key_from_description(_desc)
                        if _issue_key is not None:
                            register_issue_exhaustion(
                                _issue_key, reason=str(exc)[:120]
                            )
                    except Exception:
                        logger.debug(
                            "[CandidateGenerator] github_issue cooldown "
                            "hook failed",
                            exc_info=True,
                        )
                # §3.6.2 vector #12 (2026-05-07) — Tier 3
                # deterministic fallback. When master flag on,
                # substitute a structured deferred GenerationResult
                # instead of re-raising. Prevents the organism
                # freeze when both Tier 0 + Tier 1 are out.
                # Master flag default-FALSE per §33.1 — when off,
                # byte-identical pre-slice behavior (re-raise).
                # NEVER raises into the dispatch path.
                try:
                    from backend.core.ouroboros.governance.tier3_deterministic_fallback import (  # noqa: E501
                        build_deferred_generation_result,
                        emit_substitution_telemetry,
                        should_intercept_exhaustion,
                    )
                    if should_intercept_exhaustion():
                        _deferred = build_deferred_generation_result(
                            op_id=getattr(
                                context, "op_id", "",
                            ) or "",
                            cause=str(exc)[:200],
                        )
                        if _deferred is not None:
                            emit_substitution_telemetry(
                                op_id=getattr(
                                    context, "op_id", "",
                                ) or "",
                                cause=str(exc)[:200],
                            )
                            return _deferred
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[CandidateGenerator] tier3 fallback "
                        "intercept failed (non-fatal); "
                        "re-raising original exhaustion",
                        exc_info=True,
                    )
                # Phase 3.3 Task 2 — J-Prime last-resort local handoff.
                # When JARVIS_JPRIME_LASTRESORT_ENABLED is true, route the op
                # to the local 3B tier with a topologically-pruned payload
                # instead of crashing the loop. Gate default OFF -> re-raise
                # (byte-identical legacy). Never masks the original error on
                # local failure or unhealthy probe.
                try:
                    from backend.core.ouroboros.governance.exhaustion_interceptor import (
                        should_intercept,
                        execute_local_last_resort,
                    )
                    if should_intercept(exc, jprime=self._jprime):
                        _ft = self._estimate_target_file_tokens(context)
                        _broker = self._resolve_sse_broker()
                        return await execute_local_last_resort(
                            jprime=self._jprime,
                            context=context,
                            deadline=deadline,
                            graph_backend=getattr(self, "_graph_backend", None),
                            broker=_broker,
                            file_tokens=_ft,
                            original_exc=exc,
                        )
                except RuntimeError:
                    # execute_local_last_resort re-raises original_exc on any
                    # local failure — let it propagate cleanly.
                    raise
                except Exception:  # noqa: BLE001 — defensive
                    logger.debug(
                        "[CandidateGenerator] jprime lastresort "
                        "intercept failed (non-fatal); "
                        "re-raising original exhaustion",
                        exc_info=True,
                    )
            raise
        else:
            if self._exhaustion_watcher is not None:
                try:
                    await self._exhaustion_watcher.record_success()
                except Exception:
                    logger.debug(
                        "[CandidateGenerator] exhaustion_watcher "
                        "record_success failed",
                        exc_info=True,
                    )
            # Antivenom Vector 1: BG/SPEC routes structurally skip
            # Quorum (cost-gated). A single-roll candidate that
            # claims to modify code but produces an AST fingerprint
            # identical to the original is a Quine-class hallucination.
            # Filter such candidates out — empty result is a
            # correctness win (orchestrator's accept-failure branch
            # handles it gracefully, no apply, no harm).
            try:
                result = await self._apply_bg_spec_structural_filter(
                    context=context, result=result,
                )
            except Exception:  # noqa: BLE001 — never break generate()
                logger.debug(
                    "[CandidateGenerator] bg_spec_structural_filter "
                    "raised; passing through unfiltered result",
                    exc_info=True,
                )
            return result

    def _estimate_target_file_tokens(self, context: Any) -> Dict[str, int]:
        """Best-effort per-file token estimate for the exhaustion interceptor.

        Reads each file in ``context.target_files`` relative to the repo root
        (``self._repo_root`` when set, else cwd) and estimates token count as
        ``len(text) // 4``. Missing or unreadable files contribute 0. Never
        raises -- any error is swallowed so the interceptor path stays safe.
        """
        result: Dict[str, int] = {}
        try:
            files = list(getattr(context, "target_files", ()) or ())
            repo_root = getattr(self, "_repo_root", None)
            for f in files:
                try:
                    import os as _os
                    path = (
                        _os.path.join(repo_root, f)
                        if repo_root and not _os.path.isabs(f)
                        else f
                    )
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        result[f] = len(fh.read()) // 4
                except Exception:  # noqa: BLE001
                    result[f] = 0
        except Exception:  # noqa: BLE001
            pass
        return result

    def _resolve_sse_broker(self) -> Any:
        """Return the default SSE broker for beacon publishing, or None on failure."""
        try:
            from backend.core.ouroboros.governance.ide_observability_stream import (
                get_default_broker,
            )
            return get_default_broker()
        except Exception:  # noqa: BLE001
            return None

    async def _apply_bg_spec_structural_filter(
        self,
        *,
        context: OperationContext,
        result: GenerationResult,
    ) -> GenerationResult:
        """Antivenom Vector 1: structural Quine-class guard for
        BG/SPEC routes.

        Routes BACKGROUND/SPECULATIVE structurally skip the
        Quorum gate (``COST_GATED_ROUTES`` in
        ``cost_contract_assertion``). That leaves single-roll
        generation with no consensus check — a hallucinated
        candidate whose AST equals the original (different text,
        same shape) can ship.

        This filter runs ``compute_bg_spec_structural_check`` on
        each candidate's ``(file_path, full_content)`` pair (or
        each entry in a multi-file candidate's ``files`` list)
        against the on-disk original. Anomaly → drop the
        candidate. New files (no on-disk original) are passed
        through (no AST to compare).

        Cost: zero LLM calls. AST signature compute is bounded
        by file size; runs in a thread to avoid blocking the
        event loop. Master gate
        ``JARVIS_BG_SPEC_STRUCTURAL_CHECK_ENABLED`` (default
        ``true``) lives on the primitive in
        ``generative_quorum_gate``."""
        try:
            route = (
                getattr(context, "provider_route", "") or ""
            ).strip().lower()
            if route not in ("background", "speculative"):
                return result
            if not result.candidates:
                return result

            # Lazy import: keep generative_quorum_gate out of the
            # hot import path for non-BG/SPEC ops.
            try:
                from backend.core.ouroboros.governance.verification.generative_quorum_gate import (
                    compute_bg_spec_structural_check,
                )
            except ImportError:
                return result

            change_desc = (
                getattr(context, "description", "") or ""
            )
            # No claimed change → no Quine vector. Skip.
            if not change_desc.strip():
                return result

            cwd = Path.cwd()

            def _check_one(file_path: str, candidate_src: str) -> Tuple[bool, str]:
                """Return ``(anomaly_detected, reason)``. Best-effort
                — any failure → no anomaly (defense in depth)."""
                try:
                    if not file_path or not isinstance(candidate_src, str):
                        return (False, "")
                    p = Path(file_path)
                    if not p.is_absolute():
                        p = cwd / p
                    if not p.exists() or not p.is_file():
                        # New file — no original to compare.
                        return (False, "")
                    try:
                        original_src = p.read_text(
                            encoding="utf-8", errors="replace",
                        )
                    except OSError:
                        return (False, "")
                    chk = compute_bg_spec_structural_check(
                        candidate_source=candidate_src,
                        original_source=original_src,
                        change_description=change_desc,
                    )
                    return (chk.anomaly_detected, chk.anomaly_reason)
                except Exception:  # noqa: BLE001 — defensive
                    return (False, "")

            def _candidate_anomalous(cand: Dict[str, Any]) -> Tuple[bool, str]:
                """Multi-file candidate: anomalous iff EVERY entry is
                anomalous (a partial mix may still be a real change).
                Single-file candidate: direct check."""
                files_list = cand.get("files")
                if isinstance(files_list, list) and files_list:
                    entries = [
                        e for e in files_list
                        if isinstance(e, dict)
                    ]
                    if not entries:
                        return (False, "")
                    flags: list = []
                    reasons: list = []
                    for entry in entries:
                        anom, reason = _check_one(
                            entry.get("file_path", ""),
                            entry.get("full_content", ""),
                        )
                        flags.append(anom)
                        if reason:
                            reasons.append(reason)
                    # All-or-nothing: only drop when every entry
                    # is structurally identical to its original.
                    if flags and all(flags):
                        return (
                            True,
                            f"multi_file_all_quine: {'; '.join(reasons)[:200]}",
                        )
                    return (False, "")
                # Single-file legacy shape.
                return _check_one(
                    cand.get("file_path", ""),
                    cand.get("full_content", ""),
                )

            # Run AST signature compute off the event loop. Each
            # check reads a file; bound the parallelism via the
            # default thread pool (no extra knobs to tune).
            anomaly_results: list = await asyncio.gather(
                *[
                    asyncio.to_thread(_candidate_anomalous, cand)
                    for cand in result.candidates
                ],
                return_exceptions=True,
            )

            kept: list = []
            dropped: int = 0
            for cand, outcome in zip(result.candidates, anomaly_results):
                if isinstance(outcome, BaseException):
                    kept.append(cand)
                    continue
                anom, reason = outcome
                if anom:
                    dropped += 1
                    op_id = (
                        getattr(context, "op_id", "") or "?"
                    )[:12]
                    logger.warning(
                        "[CandidateGenerator] bg_spec_quine_drop "
                        "op=%s route=%s reason=%s",
                        op_id, route, (reason or "")[:160],
                    )
                else:
                    kept.append(cand)

            if dropped == 0:
                return result

            # Replace the candidates tuple. Keep all other
            # GenerationResult fields intact (provider_name,
            # duration, tool records, token usage, cost — these
            # describe what the provider actually did, not the
            # filter outcome).
            return dataclasses.replace(
                result, candidates=tuple(kept),
            )
        except Exception:  # noqa: BLE001 — last-resort defensive
            return result

    async def _honor_provider_override(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> Optional[GenerationResult]:
        """Sovereign Failover Mesh Gap 3b — honor a Cryo-DLQ provider pin.

        When ``context.provider_override == "gcp-jprime"`` (stamped at Cryo-DLQ
        seal time on a DW global outage), route the op STRAIGHT to the awakened
        J-Prime ``PrimeProvider`` -- NOT through the dead DW cascade.

        Returns:
          * a ``GenerationResult`` when J-Prime produced candidates (honored);
          * ``None`` when there is no override (legacy dispatch continues);
          * raises a terminal ``RuntimeError`` (fail-CLOSED) when the override
            is set but J-Prime is unavailable / yielded nothing -- so the op
            STAYS SEALED in the DLQ and is NEVER re-routed to dead DW.

        Empty override -> None (byte-identical legacy). Fail-soft on the read
        side: an unreadable override is treated as no override.
        """
        try:
            override = (getattr(context, "provider_override", "") or "").strip()
        except Exception:  # noqa: BLE001
            override = ""
        if not override:
            return None  # legacy: no pin, normal cascade

        op_id_short = (getattr(context, "op_id", "") or "?")[:16]

        # Only "gcp-jprime" is a recognized pin today. An unknown override falls
        # through to the legacy cascade (forward-compatible; never a hard error
        # on an unrecognized value).
        if override != "gcp-jprime":
            logger.info(
                "[CandidateGenerator] provider_override=%s unrecognized -- "
                "falling through to legacy cascade [%s]",
                override, op_id_short,
            )
            return None

        # Dynamic state-driven routing: the failover node awakens at RUNTIME, so do
        # NOT require a boot-wired self._jprime. DISCOVER the awakened endpoint and
        # dispatch generation there (reusing the Phase 3c LocalPrimeClient seam).
        endpoint = await self._discover_jprime_endpoint()
        if endpoint:
            logger.info(
                "[CandidateGenerator] provider_override=gcp-jprime -> dynamically "
                "discovered awakened endpoint=%s, dispatching GENERATE to the 32B "
                "(DW lane bypassed) [%s]", endpoint, op_id_short,
            )
            result = await self._failover_local_dispatch(context, deadline, endpoint)
            if result is not None and len(
                getattr(result, "candidates", ()) or ()
            ) > 0:
                return result

        # Legacy fast path: a boot-wired static PrimeProvider (e.g. J-Prime primary).
        if self._jprime is not None and getattr(self._jprime, "provider_name", ""):
            logger.info(
                "[CandidateGenerator] provider_override=gcp-jprime -> static "
                "PrimeProvider primacy [%s]", op_id_short,
            )
            result = await self._try_jprime_primacy(
                context, deadline, route_label="failover_override", force=True,
            )
            if result is not None:
                return result

        # FAIL-CLOSED: no awakened endpoint discoverable AND no static PrimeProvider
        # yielded candidates -> do NOT cascade to the dead DW lane. Raise terminal
        # so the op stays sealed in the DLQ for a later replay (op is never lost).
        logger.warning(
            "[CandidateGenerator] provider_override=gcp-jprime but no awakened "
            "J-Prime endpoint is discoverable and no static PrimeProvider yielded "
            "candidates -- op STAYS SEALED (NOT routed to dead DW) [%s]", op_id_short,
        )
        raise RuntimeError(
            "provider_override_unavailable:gcp-jprime:no_endpoint"
        )

    def _swarm_agent_client(self) -> Optional[Any]:
        """The provider CLIENT the swarm's agent turn speaks to, or None.

        ``ProductionAgentTurnFn`` wants a *client* -- something with
        ``async generate(prompt=..., system_prompt=..., model_name=...,
        task_profile=...)``: a ``PrimeClient`` / ``LocalPrimeClient``. This
        generator holds PROVIDERS (``CandidateProvider.generate(context,
        deadline)``); the client lives one level down, inside
        ``PrimeProvider._state.client``. The swarm wire (#70029) read
        ``self._client``, which no constructor ever set, so with
        ``JARVIS_SWARM_ROUTING_ENABLED=true`` every generation died at
        ``AttributeError`` before any provider was asked -- the devtest
        baseline of 2026-09-05 could not reach GENERATE at all. Its tests
        passed because they injected ``g._client`` by hand.

        Resolution order: an explicitly injected ``_client`` (the test seam,
        kept honest by name), then the J-Prime seat, then primary, then
        fallback -- the first that carries a callable ``generate``. None
        means "no agent brain on this box"; the caller DECLINES the swarm
        route and the standard route runs byte-identical. Never raises.
        """
        return agent_client_from(
            (getattr(self, seat, None) for seat in ("_jprime", "_primary", "_fallback")),
            injected=getattr(self, "_client", None),
        )

    def _swarm_routing_enabled(self) -> bool:
        """Dynamic toggle (env ``JARVIS_SWARM_ROUTING_ENABLED``, default OFF) —
        WORKSPACE_PROMOTION-style, no code mutation to flip. Off → the
        short-circuit is a strict no-op and the standard generation route is
        byte-identical."""
        return os.environ.get(
            "JARVIS_SWARM_ROUTING_ENABLED", "false",
        ).strip().lower() in ("1", "true", "yes", "on")

    async def _swarm_or_none(
        self, context: OperationContext, deadline: datetime,
    ) -> Optional[GenerationResult]:
        """The swarm short-circuit with its documented contract enforced at
        the seam: any fault inside it -- not only the resolver's -- means
        "standard route", never a failed generation. The wire's own
        ``AttributeError`` escaped this way on 2026-09-05 and took BOTH
        generation attempts of every op with it. ``CancelledError`` still
        propagates (structured concurrency)."""
        try:
            return await self._maybe_swarm_short_circuit(context, deadline)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a swarm fault is not an op failure
            logger.warning(
                "[CandidateGenerator] swarm short-circuit raised %s: %s — "
                "standard route preserved",
                type(exc).__name__, exc,
            )
            return None

    def _read_source_for_swarm(self, rel_or_abs: str) -> Optional[str]:
        """Read a target file's current on-disk content (repo_root-joined).
        Returns None on any miss → the short-circuit declines. Never raises."""
        try:
            repo_root = getattr(self, "_repo_root", None)
            path = rel_or_abs
            if repo_root and not os.path.isabs(rel_or_abs):
                path = os.path.join(repo_root, rel_or_abs)
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:  # noqa: BLE001
            return None

    def _swarm_frames_from_ctx(self, context: OperationContext) -> Tuple[str, ...]:
        """Best-effort extraction of failing-test traceback frames from the
        op's intake evidence (feeds the resolver's deterministic pass). Empty
        on any miss — the resolver then falls to the goal-keyword pass."""
        raw = getattr(context, "intake_evidence_json", "") or ""
        if not raw:
            return ()
        try:
            import json as _json
            data = _json.loads(raw)
        except (ValueError, TypeError):
            return ()
        if not isinstance(data, dict):
            return ()
        for key in ("traceback_frames", "traceback", "frames"):
            v = data.get(key)
            if isinstance(v, (list, tuple)):
                return tuple(str(x) for x in v)
            if isinstance(v, str) and v.strip():
                return (v,)
        return ()

    async def _maybe_swarm_short_circuit(
        self, context: OperationContext, deadline: datetime,
    ) -> Optional[GenerationResult]:
        """Route a CONFIRMED big-file op through the Agentic Swarm interceptor,
        returning a ``GenerationResult`` that short-circuits normal generation.

        Returns ``None`` — falling through to the standard route BYTE-IDENTICAL —
        whenever the dynamic flag is off, the route is cost-optimized, the file
        is small, the swarm stack is unavailable, or the ``TargetSymbolResolver``
        FAILS CLOSED (no confident target). Structured concurrency: a parent
        cancellation/timeout is re-raised cleanly so the interceptor's swarm
        tasks + DW sockets (its awaited children) tear down with no ghost tasks.
        """
        if not self._swarm_routing_enabled():
            # The LAST unlogged decline path, and the one that cost the most to
            # find: with every other branch instrumented, an op that dispatched
            # while emitting no decline line could only have exited here. Left
            # silent, "the flag is off" and "the function was never called" look
            # identical from the log, and they demand opposite fixes.
            logger.info(
                "[CandidateGenerator] swarm decline: JARVIS_SWARM_ROUTING_ENABLED "
                "is not set (observed=%r)",
                os.environ.get("JARVIS_SWARM_ROUTING_ENABLED"))
            return None
        route = (getattr(context, "provider_route", "") or "standard").lower()
        if route in ("background", "speculative") and not _free_lane_active():
            logger.info(
                "[CandidateGenerator] swarm decline: route=%s is cost-optimized "
                "and this lane is metered", route)
            # Cost-optimized routes do not fan out a swarm -- because a swarm is
            # N generation calls and, on a metered provider, N times the bill.
            # That is a statement about MONEY, not about correctness, and it
            # stops being true on a lane whose marginal cost is zero.
            #
            # It also inverts on such a lane: the swarm exists to avoid handing a
            # model a whole large file, which is exactly the failure a local 32B
            # hits hardest (truncated full_content -> missing required fields and
            # syntax errors). Skipping chunking to save money we are not spending
            # buys nothing and costs the op. Note the local "swarm" is not even
            # parallel -- the VRAM mutex serializes it onto one device, so it is
            # sequential chunk processing, slower but free, which is precisely
            # the trade a background op should take.
            return None
        # DIAGNOSABILITY. This method declines at five separate points and, until
        # now, every one of them returned None silently. That is fine when the
        # feature is off and nobody is asking -- and actively obstructive the
        # moment someone enables it and it does nothing, because "no swarm
        # happened" and "the swarm declined for reason X" are indistinguishable
        # in the log. Three soaks were spent inferring which gate fired from the
        # SIZE OF THE MODEL'S OUTPUT. One line per decline turns that into a
        # reading. INFO, not DEBUG: a silent no-op the operator explicitly asked
        # for is exactly the thing worth saying out loud.
        target_files = tuple(getattr(context, "target_files", ()) or ())
        if not target_files:
            logger.info(
                "[CandidateGenerator] swarm decline: op carries no target_files")
            return None
        path = target_files[0]
        source = self._read_source_for_swarm(path)
        if source is None:
            logger.info(
                "[CandidateGenerator] swarm decline: source unreadable for %s "
                "(repo_root=%s)", path, getattr(self, "_repo_root", None))
            return None
        try:
            from backend.core.ouroboros.governance.chunked_generation import (
                is_big_file,
            )
            from backend.core.ouroboros.governance.target_symbol_resolver import (
                resolve_target_symbols,
            )
            from backend.core.ouroboros.governance.full_content_interceptor import (
                intercept_full_content,
            )
            from backend.core.ouroboros.governance.agent_turn_adapter import (
                ProductionAgentTurnFn,
            )
        except Exception as _swarm_imp_exc:  # noqa: BLE001 — stack absent → standard route
            logger.info(
                "[CandidateGenerator] swarm decline: stack unavailable (%s: %s)",
                type(_swarm_imp_exc).__name__, _swarm_imp_exc)
            return None
        if not is_big_file(source):
            logger.info(
                "[CandidateGenerator] swarm decline: %s is under the big-file "
                "threshold (%d lines)", path, source.count("\n"))
            return None

        res = resolve_target_symbols(
            source=source, file_path=path,
            traceback_frames=self._swarm_frames_from_ctx(context),
            source_loci=target_files,
            goal=getattr(context, "description", "") or "",
            # Operator-declared targets from the SIGNED roadmap goal this op
            # traces to. Re-derived from ground truth at read time rather than
            # taken off the op, so a fabricated context field cannot inject a
            # target the operator never authorised.
            declared_symbols=_declared_symbols_for(context, path),
        )
        if not res.resolved:
            logger.info(
                "[CandidateGenerator] swarm: resolver FAIL-CLOSED for %s — "
                "standard route preserved", path,
            )
            return None

        try:
            from backend.core.ouroboros.governance.providers import (
                _CODEGEN_SYSTEM_PROMPT as _sys_prompt,
            )
        except Exception:  # noqa: BLE001
            _sys_prompt = ""

        client = self._swarm_agent_client()
        if client is None:
            logger.info(
                "[CandidateGenerator] swarm decline for %s: no provider seat "
                "exposes an agent client — standard route preserved", path,
            )
            return None

        agent = ProductionAgentTurnFn(
            client=client,
            tool_backend=None,                       # pure-completion node repair (v1)
            repo_root=getattr(self, "_repo_root", "."),
            op_id=getattr(context, "op_id", ""),
            model_name=getattr(client, "_model", "") or "",  # client default, no hardcode
            system_prompt=_sys_prompt,
            parse_fn=lambda raw: None,               # single-shot node completion
            max_turns=1,
        )
        t0 = time.monotonic()
        try:
            result = await intercept_full_content(
                source, path, list(res.symbol_names), agent,
                op_id=getattr(context, "op_id", ""),
            )
        except asyncio.CancelledError:
            # Structured concurrency: awaiting the interceptor makes its swarm
            # tasks + DW sockets children of THIS task; cancellation has already
            # torn them down. Re-raise cleanly — NEVER swallow (no ghost tasks).
            logger.info(
                "[CandidateGenerator] swarm CANCELLED for %s — clean teardown", path,
            )
            raise
        except Exception as exc:  # noqa: BLE001 — any swarm fault → standard route
            logger.warning(
                "[CandidateGenerator] swarm error for %s: %s — standard route",
                path, exc,
            )
            return None

        if (
            getattr(result, "drifted", False)
            or not getattr(result, "stitched", False)
            or not getattr(result, "content", "")
        ):
            logger.info(
                "[CandidateGenerator] swarm no-stitch/drift for %s — standard route",
                path,
            )
            return None

        dur = time.monotonic() - t0
        rationale = (
            f"agentic swarm ({res.method}, conf={res.confidence:.2f}): "
            f"primary={list(res.primary)} cluster={list(res.cluster)} "
            f"converged={list(result.converged_nodes)} "
            f"rag_recovered={list(result.rag_recovered_nodes)}"
        )
        logger.info(
            "[CandidateGenerator] SWARM short-circuit LANDED for %s: %s [%.1fs]",
            path, rationale, dur,
        )
        return GenerationResult(
            candidates=(
                {
                    "candidate_id": f"swarm-{getattr(context, 'op_id', '')[:8]}",
                    "file_path": path,
                    "full_content": result.content,
                    "rationale": rationale,
                },
            ),
            provider_name="doubleword-agentic-swarm",
            generation_duration_s=dur,
            model_id=getattr(client, "_model", "") or "",
        )

    async def _generate_dispatch(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """Internal dispatch — the original body of :meth:`generate`.

        Route-based dispatch with Tier 0 → fallback cascade. This is
        the hot path; the public ``generate()`` above wraps it only to
        observe exhaustion and success signals.
        """
        # ── Sovereign Failover Mesh Gap 3b: provider_override honor-check ──
        # When an op was sealed into the Cryo-DLQ on a DW global outage it
        # carries provider_override="gcp-jprime" (the awakened J-Prime node).
        # On replay we MUST route it straight to J-Prime, NOT re-cascade through
        # the dead DW lane. Fail-CLOSED: if J-Prime is unavailable (no awakened
        # endpoint), raise a terminal sentinel so the op STAYS SEALED in the DLQ
        # (the orchestrator records a generation failure; the op is NOT lost and
        # is NEVER sent to dead DW). Master gate inside the helper keeps OFF
        # byte-identical (empty override -> no-op).
        _override_result = await self._honor_provider_override(context, deadline)
        if _override_result is not None:
            return _override_result


        # ── Big-file Agentic Swarm short-circuit (default-OFF; fail-closed) ──
        # When JARVIS_SWARM_ROUTING_ENABLED and the op targets a big file with a
        # deterministically-resolvable symbol, route through the swarm
        # interceptor instead of whole-file generation. Returns None (standard
        # route, byte-identical) on the flag being off / small file / resolver
        # fail-closed. CancelledError propagates (structured concurrency).
        _swarm_result = await self._swarm_or_none(context, deadline)
        if _swarm_result is not None:
            return _swarm_result

        # ── Local-primary: the free lane runs BEFORE the cascade ───────────
        # ORDER IS LOAD-BEARING, and getting it wrong is measured, not
        # theoretical. This block originally sat ABOVE the swarm short-circuit,
        # which meant that on a host with no paid credential it answered every
        # op first and the swarm interceptor was never reached — soak
        # bt-2026-08-28-111858 logged ZERO occurrences of "swarm" with
        # JARVIS_SWARM_ROUTING_ENABLED=true. Enabling the chunker and then
        # short-circuiting past it is worse than leaving it off, because the
        # telemetry says it is on.
        #
        # The two are answering different questions and must run in that
        # order: the swarm decides WHAT to generate (scope reduction — slice
        # the 75-line symbol out of a 10,923-line file), this decides WHO
        # generates it (provider selection). Reducing scope first is what
        # makes the local engine's 32,768-token ceiling sufficient; choosing
        # the engine first throws the reduction away.
        #
        # Still ahead of every route handler, which is the point: a lane that
        # runs only after the cascade has failed is a fallback, and on a host
        # with no paid credential the cascade does not fail usefully — it
        # exhausts (bt-2026-08-28-100733: a SANCTIONED op died
        # `all_providers_exhausted:circuit_breaker_tripped` on route=standard
        # while a warm 32B sat resident on the GPU).
        _local_first = await self._try_local_primary(context, deadline)
        if _local_first is not None:
            return _local_first

        # ── Route-based dispatch (Manifesto §5 Tier 0: deterministic) ──
        _provider_route = getattr(context, "provider_route", "") or "standard"

        # ── Phase 10 P10.3+P10.3.5 — AsyncTopologySentinel gate ────
        # Pre-Slice-23: env-only check (``JARVIS_TOPOLOGY_SENTINEL_ENABLED=true``).
        # Slice 23 (autonomous registry-driven): the gate now consults
        # ``_slice23_should_activate_sentinel`` which composes 5
        # decision conditions (env explicit on/off / Claude disabled /
        # multi-model trusted fleet / Phase 10 default-off). See helper
        # docstring for the full closed decision matrix. The Phase 10
        # graduation contract pin (env DEFAULT stays false) is preserved
        # — Slice 23 adds structural overrides on top of that default,
        # the literal default is unchanged.
        #
        # Pre-flight handshake (directive 2026-04-27): instead of a
        # silent try/except that swallows boundary-isolation defects
        # (which is what bit session bt-2026-04-27-194550), we run
        # ``preflight_check()`` at the gate. If the sentinel fails to
        # initialize inside this subprocess for ANY reason — module
        # import, topology load, missing dw_models — we raise
        # ``SentinelInitializationError`` so the operator sees the
        # defect at the point of decision, not minutes later in the
        # postmortem. Master-flag-off remains byte-identical legacy
        # behavior: this entire block is bypassed.
        _slice23_activate, _slice23_reason = _slice23_should_activate_sentinel(
            _provider_route,
        )
        if _slice23_activate:
            logger.info(
                "[CandidateGenerator] Slice 23 sentinel activation: "
                "route=%s reason=%s — walking ranked DW fleet "
                "(skips OPEN breakers + Slice 20C drifted models)",
                _provider_route, _slice23_reason,
            )
            try:
                from backend.core.ouroboros.governance.topology_sentinel import (
                    preflight_check as _sentinel_preflight,
                    SentinelInitializationError as _SentinelInitError,
                )
            except ImportError as _imp_exc:
                # Master flag explicitly true but the module is
                # unimportable — this is a deployment defect, NOT a
                # silent fall-through. Raise so the orchestrator's
                # existing accept-failure branch records it visibly.
                raise RuntimeError(
                    f"sentinel_module_import_failed:"
                    f"{type(_imp_exc).__name__}:"
                    f"{str(_imp_exc)[:120]}"
                ) from _imp_exc
            _preflight = _sentinel_preflight()
            if not _preflight.healthy:
                raise _SentinelInitError(
                    _preflight.failed_assertions,
                    _preflight.diagnostics,
                )
            logger.info(
                "[CandidateGenerator] Phase 10 sentinel preflight: "
                "healthy=True schema=%s routes_with_dw_models=%s "
                "monitor_config=%s event_loop_bound=%s diagnostics=%s",
                _preflight.schema_version,
                list(_preflight.routes_with_dw_models),
                _preflight.monitor_config_present,
                _preflight.event_loop_bound,
                list(_preflight.diagnostics),
            )
            _result = await self._dispatch_via_sentinel(
                context, deadline, _provider_route,
            )
            if _result is not None:
                _note_dw_candidate_success()  # Slice 53 — a lane yielded a candidate
                return _result
            # _dispatch_via_sentinel returns None to signal "fall
            # through to legacy path" (e.g. the route has empty
            # dw_models — IMMEDIATE by design — so the existing
            # _generate_immediate handler still runs below).

        # Brain Selection Topology — hard segmentation (Manifesto §5).
        # When ``doubleword_topology`` marks a route as DW-forbidden,
        # the ``block_mode`` field decides what to do next:
        #
        #   cascade_to_claude — IMMEDIATE/COMPLEX: route straight to
        #     Claude via ``_generate_immediate``. Live-fire bbpst3ebf
        #     (2026-04-14) proved BOTH DW 397B and Gemma 4 31B time out
        #     on the 120s Tier 0 RT budget for architectural COMPLEX
        #     GENERATE; Claude is the intended brain for these routes.
        #
        #   skip_and_queue — BACKGROUND/SPECULATIVE: raise a sentinel
        #     RuntimeError the orchestrator already handles gracefully
        #     (background_dw_* / speculative_deferred). Do NOT cascade
        #     to Claude. Alignment test bt-2026-04-14-182446 produced
        #     0/13 Gemma BG successes with a right-sized 2.8K-token
        #     envelope — root cause is provider-side SSE stream stall,
        #     not prompt size. Routing continuous background daemons
        #     to Claude violates the unit economics of scalable
        #     autonomy. The queue stays dormant until a viable,
        #     cost-effective inference endpoint is secured.
        from backend.core.ouroboros.governance.provider_topology import (
            get_topology as _get_topology,
        )
        _topology = _get_topology()
        # Phase 10 Slice 5a — unified deletion-side helper. Branches
        # on JARVIS_TOPOLOGY_SENTINEL_ENABLED internally so v1 yaml
        # fields can be deleted safely in Slice 5b after contract
        # green. block_mode preserved in v1 vocab — downstream
        # `== "skip_and_queue"` check unchanged.
        _is_blocked, _block_reason, _block_mode = (
            _topology.is_dw_blocked_for_route(_provider_route)
        )
        if _topology.enabled and _is_blocked:
            if _block_mode == "skip_and_queue":
                # Nervous System Reflex (Manifesto §5 — survival supersedes
                # cost optimization): read-only ops MUST NOT lock up on a
                # paused DW endpoint. When the topology has skipped DW on
                # BACKGROUND, cascade straight to Claude for the read-only
                # op instead of raising skip_and_queue. The is_read_only
                # contract (Rule 0d + APPLY short-circuit) makes this
                # structurally safe: no mutation can happen, so the cost
                # of the Claude call is bounded and observable.
                _is_read_only = bool(
                    getattr(context, "is_read_only", False)
                )
                if (
                    _is_read_only
                    and _provider_route == "background"
                    and self._fallback is not None
                ):
                    logger.info(
                        "[CandidateGenerator] Nervous-System Reflex: BG "
                        "topology skip_and_queue bypassed for read-only op "
                        "— cascading to Claude (reason=%s) [%s]",
                        _block_reason,
                        getattr(context, "op_id", "?")[:16],
                    )
                    try:
                        return await self._call_fallback(context, deadline)
                    except GovernanceDeadlockError:
                        raise  # LR3 terminal -- never wrap as fallback_failed
                    except Exception as exc:
                        raise RuntimeError(
                            f"background_fallback_failed:"
                            f"topology_skip_read_only_cascade:"
                            f"{type(exc).__name__}:{str(exc)[:80]}"
                        ) from exc
                logger.info(
                    "[CandidateGenerator] Topology block: route=%s "
                    "block_mode=skip_and_queue reason=%s — skipping "
                    "generation (no Claude cascade)",
                    _provider_route, _block_reason,
                )
                # Sentinel-Pacemaker handshake (2026-04-29) —
                # when the topology layer blocks BG/SPEC ops because
                # the catalog is purged/empty, ask the Pacemaker to
                # bypass its 30-min cadence sleep and probe DW now.
                # If DW is reachable, the next refresh cycle
                # repopulates the catalog and subsequent ops flow.
                # Best-effort, never raises. Rate-limited at the
                # trigger site so a block-storm doesn't thrash /models.
                _reason_lower = (_block_reason or "").lower()
                _is_catalog_purge = (
                    "catalog" in _reason_lower
                    and (
                        "purged" in _reason_lower
                        or "empty" in _reason_lower
                        or "static list" in _reason_lower
                    )
                )
                if _is_catalog_purge:
                    try:
                        from backend.core.ouroboros.governance.dw_discovery_runner import (
                            request_force_refresh,
                        )
                        request_force_refresh(
                            reason=(
                                f"topology_block:{_provider_route}:"
                                f"{_block_reason[:80]}"
                            ),
                        )
                    except Exception:  # noqa: BLE001 — never raise
                        logger.debug(
                            "[CandidateGenerator] force_refresh "
                            "request failed", exc_info=True,
                        )
                # Run-#13 UNMASK: this topology pre-block raises BEFORE the
                # model-walk loop's record_sweep, so the outage gradient never
                # saw the DW collapse and is_global_outage(route) never tripped
                # (J-Prime never awoke for a 20-min real outage). Feed the raw
                # failed sweep through the single chokepoint so the gradient
                # gets the truth -- the op still fails gracefully downstream.
                try:
                    from backend.core.ouroboros.governance.provider_quarantine import (
                        record_terminal_exhaustion as _rec_exhaustion,
                    )
                    _rec_exhaustion(
                        _provider_route,
                        reason=f"dw_severed_queued:topology_block:{_block_reason[:60]}",
                    )
                except Exception:  # noqa: BLE001 -- never block the raise path
                    pass
                # The dormant queue's own release condition, checked before
                # the queue claims it: "until a viable, cost-effective
                # inference endpoint is secured". A locally-served 32B IS
                # that endpoint, at $0.00/op. Returns None on every host that
                # does not have one -> the raises below are untouched.
                _free_lane = await self._try_free_lane_dispatch(
                    context, deadline,
                    route=_provider_route,
                    reason=f"topology_block:{_block_reason}",
                )
                if _free_lane is not None:
                    return _free_lane
                if _provider_route == "speculative":
                    raise RuntimeError(
                        f"speculative_deferred:blocked_by_topology:"
                        f"{_block_reason[:120]}"
                    )
                # BACKGROUND (and any future skip_and_queue route)
                # routes through the orchestrator's
                # "background_dw_*" graceful-accept branch.
                raise RuntimeError(
                    f"background_dw_blocked_by_topology:"
                    f"{_block_reason[:120]}"
                )
            logger.info(
                "[CandidateGenerator] Topology block: route=%s "
                "block_mode=cascade_to_claude reason=%s — routing "
                "direct to Claude",
                _provider_route, _block_reason,
            )
            return await self._generate_immediate(context, deadline)

        # Quota Shield: a shield-selected (prefer_local) op tries the zero-cost local
        # tier first via the existing primacy path, regardless of urgency route. On
        # local decline (sem saturation / timeout / error) it returns None and falls
        # through to the normal route below -> graceful, no behavior change when unset.
        if (
            getattr(context, "prefer_local", False)
            and self._jprime is not None
            and _provider_route not in ("background", "speculative")
        ):
            _ql = await self._try_jprime_primacy(context, deadline, route_label="quota_shield")
            if _ql is not None:
                return _ql

        if _provider_route == "immediate":
            return await self._generate_immediate(context, deadline)
        if _provider_route == "background":
            return await self._generate_background(context, deadline)
        if _provider_route == "speculative":
            return await self._generate_speculative(context, deadline)
        # "standard" and "complex" are both intercepted above by the
        # topology hard-block in the default yaml (standard sealed as of
        # bt-2026-04-14-203740, complex sealed per bbpst3ebf). They reach
        # this fall-through only if operators explicitly unseal them in
        # brain_selection_policy.yaml — at which point the full DW→Claude
        # cascade below applies. Not recommended without fresh empirical
        # evidence that DW's SSE endpoint has recovered.

        # ── Tier 0: DoubleWord 397B ──────────────────────────────
        #
        # Manifesto §3: "Zero polling. Pure reflex."
        # Manifesto §5: "Tier 0 (Deterministic Fast-Path)"
        #
        # Two modes based on DW real-time SSE availability:
        #   RT enabled  → tier0.generate() → _generate_realtime (SSE stream)
        #                 Zero polling. Token-by-token streaming. Pure reflex.
        #                 Internal RT→batch fallback on 429/503 (stay cheap).
        #   RT disabled → submit_batch() → background poll (legacy path)
        #                 Used only when DOUBLEWORD_REALTIME_ENABLED=false.
        #
        # On any Tier 0 failure → cascade to Claude fallback (Tier 1).

        _op_id = getattr(context, "operation_id", "")
        _dw_is_primary = (self._tier0 is not None and self._primary is self._tier0)
        _dw_is_fallback = (self._tier0 is not None and self._fallback is self._tier0)
        _tier0_attempted = False

        if self._tier0 is not None and getattr(self._tier0, "is_available", False):
            # Skip if DW is in any failure mode whose recovery ETA hasn't elapsed.
            # Previously this only fired on CONNECTION_ERROR — meaning a misclassified
            # TRANSIENT_TRANSPORT or TIMEOUT could keep hammering DW back-to-back
            # and exhaust every op until the human stopped the loop. Generalized
            # in bt-2026-04-12-005521 fix to honor whichever mode is active.
            _fsm_in_backoff = (
                self.fsm._failure_mode is not None
                and self.fsm._failure_mode is not FailureMode.CONTENT_FAILURE
                and not self.fsm.should_attempt_primary()
            )
            # Per-op rotation guard: even when the FSM says "go", if N consecutive
            # ops just died on Tier 0 within the rotation window, give DW a break.
            _rotation_skip = self._should_skip_tier0_for_op()

            if _rotation_skip:
                logger.info(
                    "[CandidateGenerator] Tier 0 skipped: per-op rotation "
                    "(consecutive_failures=%d threshold=%d window=%.0fs)",
                    self._counters.consecutive_tier0_failures,
                    self._tier0_skip_threshold,
                    self._tier0_skip_window_s,
                )
            elif _fsm_in_backoff:
                logger.info(
                    "[CandidateGenerator] Tier 0 skipped: DW in %s backoff "
                    "(failures=%d, ETA=%.0fs)",
                    self.fsm._failure_mode.name if self.fsm._failure_mode else "UNKNOWN",
                    self.fsm._consecutive_failures,
                    max(0, self.fsm.recovery_eta() - time.monotonic()),
                )

            elif getattr(self._tier0, "_realtime_enabled", False):
                # ── Real-time SSE path (Manifesto §3: zero polling) ──
                # Call tier0.generate() directly — hits _generate_realtime.
                # Budget-capped via asyncio.wait_for; on timeout or failure,
                # cascade to Claude fallback with guaranteed reserve time.
                _tier0_attempted = True
                remaining = self._remaining_seconds(deadline)
                _complexity = getattr(context, "task_complexity", "trivial")
                tier0_budget = self._compute_tier0_budget_dynamic(
                    remaining, _complexity, _provider_route,
                )
                tier1_reserve = remaining - tier0_budget
                _tracker_p95 = self._latency_tracker.p95() if self._latency_tracker else None

                logger.info(
                    "[CandidateGenerator] Tier 0 RT: budget=%.1fs of %.1fs "
                    "(Tier 1 reserve=%.1fs), complexity=%s, model=%s, p95=%s",
                    tier0_budget, remaining, tier1_reserve, _complexity,
                    getattr(self._tier0, "_model", "unknown"),
                    f"{_tracker_p95:.1f}s" if _tracker_p95 is not None else "cold",
                )

                if tier0_budget <= 0:
                    logger.info(
                        "[CandidateGenerator] Tier 0 skipped: zero budget "
                        "for complexity=%s. Cascading to Tier 1 (%.1fs)",
                        _complexity, remaining,
                    )
                    # Fall through to Claude cascade below

                if tier0_budget > 0:
                    # Stream-aware timeout: use asyncio.shield so we can
                    # grant a grace extension if DW is actively streaming
                    # tokens when the base budget expires (Manifesto §3).
                    _gen_task = asyncio.ensure_future(
                        self._tier0.generate(context, deadline),
                    )
                    # Defect #4 Slice A — leak-prevention callback.
                    # The shield above means _gen_task survives outer
                    # wait_for cancellation; if it later raises with
                    # nobody awaiting, asyncio's default handler logs
                    # "Task exception was never retrieved". The
                    # callback consumes the exception cleanly.
                    _gen_task.add_done_callback(_swallow_task_exception)
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(_gen_task), timeout=tier0_budget,
                        )
                        if result is not None and len(result.candidates) > 0:
                            # RT success — record recovery if coming back from failure
                            if self.fsm._consecutive_failures > 0:
                                self.fsm.record_primary_success()
                            self._record_tier0_success()
                            if self._latency_tracker is not None:
                                self._latency_tracker.record_success(
                                    result.generation_duration_s,
                                )
                            logger.info(
                                "[CandidateGenerator] Tier 0 RT: %d candidates in %.1fs "
                                "(zero polling)",
                                len(result.candidates), result.generation_duration_s,
                            )
                            return result
                        # Empty result — fall through to Claude
                        logger.info(
                            "[CandidateGenerator] Tier 0 RT: no candidates — "
                            "cascading to Tier 1 (%.1fs remaining)",
                            self._remaining_seconds(deadline),
                        )
                    except asyncio.TimeoutError:
                        # Check if DW is actively streaming SSE tokens.
                        # If so, grant up to 30s extension while preserving
                        # Tier 1 reserve — don't kill a productive stream.
                        _last_chunk = getattr(self._tier0, "_last_chunk_at", 0.0)
                        _streaming = _last_chunk > 0 and (time.monotonic() - _last_chunk) < 10.0
                        _ext_cap = self._remaining_seconds(deadline) - _TIER1_MIN_RESERVE_S
                        _extension = min(30.0, _ext_cap)

                        if _streaming and _extension > 5.0:
                            logger.info(
                                "[CandidateGenerator] Tier 0 RT: actively streaming, "
                                "granting +%.0fs extension (Tier 1 reserve preserved)",
                                _extension,
                            )
                            # Use asyncio.wait (not wait_for) so a timeout does
                            # NOT cancel the task — avoids the race where DW
                            # completes between timeout fire and cancel delivery.
                            _done, _ = await asyncio.wait(
                                {_gen_task}, timeout=_extension,
                            )
                            if _gen_task in _done:
                                try:
                                    result = _gen_task.result()
                                except Exception as ext_exc:
                                    _mode = FailbackStateMachine.classify_exception(ext_exc)
                                    logger.warning(
                                        "[CandidateGenerator] Tier 0 RT: grace-period "
                                        "content failed (mode=%s, %s). Cascading.",
                                        _mode.name, ext_exc,
                                    )
                                    result = None
                                if result is not None and len(result.candidates) > 0:
                                    if self.fsm._consecutive_failures > 0:
                                        self.fsm.record_primary_success()
                                    self._record_tier0_success()
                                    if self._latency_tracker is not None:
                                        self._latency_tracker.record_success(
                                            result.generation_duration_s,
                                        )
                                    logger.info(
                                        "[CandidateGenerator] Tier 0 RT: %d candidates "
                                        "in %.1fs (stream extension saved it)",
                                        len(result.candidates),
                                        result.generation_duration_s,
                                    )
                                    return result
                        # Task still pending or no extension granted — cancel it.
                        # Check done() first: task may have completed in the
                        # instant between timeout and here (shield race window).
                        if not _gen_task.done():
                            _gen_task.cancel()
                        elif not _gen_task.cancelled():
                            try:
                                _late = _gen_task.result()
                                if _late is not None and len(_late.candidates) > 0:
                                    if self.fsm._consecutive_failures > 0:
                                        self.fsm.record_primary_success()
                                    self._record_tier0_success()
                                    logger.info(
                                        "[CandidateGenerator] Tier 0 RT: %d candidates "
                                        "recovered from timeout race",
                                        len(_late.candidates),
                                    )
                                    return _late
                            except Exception:
                                pass

                        logger.warning(
                            "[CandidateGenerator] Tier 0 RT: budget exhausted "
                            "(%.1fs). Cascading to Tier 1 (%.1fs remaining)",
                            tier0_budget, self._remaining_seconds(deadline),
                        )
                        self.fsm.record_primary_failure(mode=FailureMode.TIMEOUT)
                        self._record_tier0_failure()
                        if self._latency_tracker is not None:
                            self._latency_tracker.record_failure()
                    except asyncio.CancelledError:
                        _gen_task.cancel()
                        raise
                    except Exception as rt_exc:
                        _gen_task.cancel()
                        mode = FailbackStateMachine.classify_exception(rt_exc)
                        logger.warning(
                            "[CandidateGenerator] Tier 0 RT failed (mode=%s, %s: %s). "
                            "Cascading to Tier 1 (%.1fs remaining)",
                            mode.name, type(rt_exc).__name__, rt_exc,
                            self._remaining_seconds(deadline),
                        )
                        # `_PRIMARY_INNOCENT_MODES` replaces the exemption list
                        # that used to be spelled out here: CONTENT_FAILURE
                        # (model produced bad output, infra healthy) and
                        # TEMPORAL_SHED (OUR routing refusal — penalizing DW
                        # would rotate the funded primary out because ONE op
                        # ran tight), now joined by LOCAL_DEFECT. Two hand-kept
                        # copies of one policy is how they drift.
                        if mode not in _PRIMARY_INNOCENT_MODES:
                            self.fsm.record_primary_failure(mode=mode)
                            self._record_tier0_failure()
                            if self._latency_tracker is not None:
                                self._latency_tracker.record_failure()

                        # ── Syntax-failure escalation recording ──
                        # When DW produces persistent AST failures, record
                        # the attempt in the SyntaxExhaustionEscalator so
                        # the J-Prime cascade can fire once the threshold
                        # is met. Fail-soft: never blocks the cascade.
                        _rt_err_msg = str(rt_exc)
                        if "all_candidates_syntax_error" in _rt_err_msg:
                            try:
                                from backend.core.ouroboros.governance.syntax_escalation import (  # noqa: E501
                                    get_escalator as _get_syntax_escalator,
                                )
                                _sx = _get_syntax_escalator()
                                _sx.record_attempt(
                                    op_id=getattr(context, "op_id", "") or "",
                                    error_msg=_rt_err_msg,
                                    candidate_preview=getattr(
                                        rt_exc, "candidate_preview", "",
                                    ) or "",
                                    target_file=getattr(
                                        rt_exc, "target_file", "",
                                    ) or "",
                                )
                            except Exception:  # noqa: BLE001
                                logger.debug(
                                    "[CandidateGenerator] syntax escalation "
                                    "recording failed (non-fatal)",
                                    exc_info=True,
                                )

            else:
                # ── Legacy batch path (DOUBLEWORD_REALTIME_ENABLED=false) ──
                # submit_batch() → background poll → await result.
                _TIER0_COMPLEXITY_CLASSES = frozenset({"heavy_code", "complex"})
                _complexity = ""
                if context.routing is not None:
                    _complexity = getattr(context.routing, "task_complexity", "")
                _is_cross_repo = getattr(context, "cross_repo", False)
                _qualifies = _complexity in _TIER0_COMPLEXITY_CLASSES or _is_cross_repo

                _dw_is_only_provider = (
                    self._primary is self._tier0 or self._fallback is self._tier0
                )
                if _dw_is_only_provider:
                    _qualifies = True

                if _qualifies:
                    _tier0_attempted = True
                    try:
                        # Slice 18 — CAPACITY BEFORE COMMITMENT.
                        #
                        # This prune + cap check used to sit AFTER submit_batch().
                        # That ordering was a guaranteed orphan generator: the
                        # batch was created on DW's queue, and only then did we
                        # discover the background-poll pool was full — at which
                        # point the `if` simply fell through with no `else`. No
                        # poller was ever created, the batch_id existed nowhere
                        # but in a local variable, and a live job sat on DW's
                        # queue that no code in this process knew about. It could
                        # only ever be cleaned up by a human at DoubleWord.
                        #
                        # Submitting work we have no capacity to collect is not a
                        # race we lost; it is an obligation we had no intention of
                        # honoring. So we check first, and simply don't submit.
                        self._background_polls = {
                            k: t for k, t in self._background_polls.items()
                            if not t.done()
                        }
                        _poll_capacity = (
                            len(self._background_polls) < self._max_background_polls
                        )
                        if not _poll_capacity:
                            logger.warning(
                                "[CandidateGenerator] Tier 0 batch NOT submitted "
                                "for op=%s: background-poll pool is full (%d/%d). "
                                "Submitting would create a batch on DW that "
                                "nothing in this process could ever collect.",
                                _op_id, len(self._background_polls),
                                self._max_background_polls,
                            )
                            pending = None
                        else:
                            pending = await self._tier0.submit_batch(context)
                        if pending is not None:
                            logger.info(
                                "[CandidateGenerator] Tier 0 batch %s submitted",
                                pending.batch_id,
                            )
                            await self._record_tier0_ledger(
                                _op_id, "pending_tier0", {
                                    "batch_id": pending.batch_id,
                                    "file_id": pending.file_id,
                                    "model": getattr(self._tier0, "_model", "unknown"),
                                },
                            )
                            task = asyncio.create_task(
                                self._background_poll_tier0(pending, context),
                                name=f"dw-poll-{pending.batch_id[:12]}",
                            )
                            # Defect #4 Slice A — defensive
                            # callback (background_poll_tier0
                            # already has try/except internally,
                            # but the callback ensures even an
                            # asyncio.CancelledError that bypasses
                            # the wrapper gets consumed).
                            task.add_done_callback(_swallow_task_exception)
                            self._background_polls[_op_id] = task
                    except asyncio.CancelledError:
                        raise
                    except Exception as t0_exc:
                        logger.warning(
                            "[CandidateGenerator] Tier 0 batch submit failed: %s",
                            t0_exc,
                        )

                # Await background poll if DW is primary/fallback
                _dw_poll_task = self._background_polls.get(_op_id)
                if _dw_poll_task is not None and (_dw_is_primary or _dw_is_fallback):
                    remaining = self._remaining_seconds(deadline)
                    _complexity = getattr(context, "task_complexity", "trivial")
                    tier0_budget = self._compute_tier0_budget(remaining, _complexity)

                    logger.info(
                        "[CandidateGenerator] Awaiting batch poll: "
                        "budget=%.1fs of %.1fs, complexity=%s",
                        tier0_budget, remaining, _complexity,
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(_dw_poll_task), timeout=tier0_budget,
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning(
                            "[CandidateGenerator] Batch poll budget exhausted "
                            "(%.1fs)", tier0_budget,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[CandidateGenerator] Batch poll error: %s", exc,
                        )

                    _completed = self._completed_batches.pop(_op_id, None)
                    if _completed is not None and _completed.result is not None:
                        _result = _completed.result
                        if len(_result.candidates) > 0:
                            logger.info(
                                "[CandidateGenerator] Batch result: %d candidates",
                                len(_result.candidates),
                            )
                            return _result

        # ── Syntax-failure J-Prime escalation (DW → J-Prime cascade) ──
        # When DW has produced persistent all_candidates_syntax_error
        # failures for this op, escalate to J-Prime with full failure
        # context (candidate previews + anti-hallucination directive).
        # This fires BEFORE the normal Tier 1 cascade so J-Prime gets
        # the enriched context. Fail-soft: on any error, falls through
        # to the normal cascade. Env-gated via
        # JARVIS_SYNTAX_ESCALATION_ENABLED (default true).
        if self._jprime is not None and _tier0_attempted:
            try:
                from backend.core.ouroboros.governance.syntax_escalation import (  # noqa: E501
                    get_escalator as _get_sx,
                    enrich_context_for_escalation as _enrich_sx,
                )
                _sx_op_id = getattr(context, "op_id", "") or ""
                _sx_esc = _get_sx()
                if _sx_esc.should_escalate(_sx_op_id):
                    _sx_ctx = _sx_esc.build_escalation_context(
                        _sx_op_id, op_context=context,
                    )
                    logger.warning(
                        "[SyntaxEscalator] ESCALATE op=%s "
                        "consecutive_syntax_failures=%d threshold_met=True "
                        "target=jprime target_file=%s — routing to J-Prime "
                        "with failure dossier (%.1fs remaining)",
                        _sx_op_id[:16],
                        _sx_ctx.consecutive_failures,
                        _sx_ctx.target_file,
                        self._remaining_seconds(deadline),
                    )
                    _sx_esc.mark_escalated(_sx_op_id)
                    _enriched_ctx = _enrich_sx(context, _sx_ctx)
                    try:
                        _sx_result = await self._jprime.generate(
                            _enriched_ctx, deadline,
                        )
                        if (
                            _sx_result is not None
                            and len(_sx_result.candidates) > 0
                        ):
                            logger.info(
                                "[SyntaxEscalator] J-Prime produced %d "
                                "candidates for op=%s — escalation "
                                "SUCCEEDED",
                                len(_sx_result.candidates),
                                _sx_op_id[:16],
                            )
                            _sx_esc.clear(_sx_op_id)
                            return _sx_result
                        logger.warning(
                            "[SyntaxEscalator] J-Prime returned no "
                            "candidates for op=%s — falling through "
                            "to normal Tier 1 cascade",
                            _sx_op_id[:16],
                        )
                    except Exception as _sx_jprime_exc:
                        logger.warning(
                            "[SyntaxEscalator] J-Prime escalation failed "
                            "for op=%s: %s — falling through to normal "
                            "Tier 1 cascade",
                            _sx_op_id[:16], _sx_jprime_exc,
                        )
            except Exception:  # noqa: BLE001 — never block the cascade
                logger.debug(
                    "[SyntaxEscalator] escalation check failed (non-fatal)",
                    exc_info=True,
                )

        # ── Tier 1: Primary → Fallback cascade ───────────────────
        #
        # If Tier 0 was attempted and DW IS the primary, skip redundant
        # primary.generate() call — go straight to Claude fallback.
        # (Manifesto §3: no wasteful retries)

        state = self.fsm.state

        if state is FailbackState.QUEUE_ONLY:
            self._raise_exhausted(
                "queue_only_dispatch",
                context=context,
                deadline=deadline,
                tier0_attempted=_tier0_attempted,
                dw_is_primary=_dw_is_primary,
            )

        if _tier0_attempted and _dw_is_primary:
            logger.info(
                "[CandidateGenerator] Tier 0 IS primary — routing directly "
                "to Claude fallback (%.1fs remaining)",
                self._remaining_seconds(deadline),
            )
            return await self._call_fallback(context, deadline)

        if state is FailbackState.PRIMARY_READY:
            # P2.3: Model-selection learning — check if historical data
            # recommends the fallback for this complexity class.  Only
            # applies when both providers are healthy (PRIMARY_READY);
            # infrastructure health always takes precedence.
            _complexity = getattr(context, "task_complexity", "") or "unknown"
            _recommended = self._query_provider_recommendation(_complexity)
            if (
                _recommended is not None
                and self._fallback is not None
                and getattr(self._fallback, "provider_name", "") == _recommended
            ):
                logger.info(
                    "[CandidateGenerator] Learning override: '%s' recommended "
                    "for complexity=%s — trying fallback first (%.1fs remaining)",
                    _recommended, _complexity, self._remaining_seconds(deadline),
                )
                try:
                    return await self._call_fallback(context, deadline)
                except Exception as _fb_exc:
                    logger.info(
                        "[CandidateGenerator] Learning-recommended fallback failed: %s "
                        "— falling back to primary",
                        type(_fb_exc).__name__,
                    )
                    return await self._call_primary(context, deadline)

            return await self._try_primary_then_fallback(context, deadline)

        # FALLBACK_ACTIVE or PRIMARY_DEGRADED: adaptive recovery routing.
        if self.fsm.should_attempt_primary():
            logger.info(
                "[CandidateGenerator] Recovery window elapsed (mode=%s, "
                "failures=%d), re-attempting primary "
                "(cost-save: $0.10/$0.40 vs $3.00/$15.00 per M)",
                self.fsm._failure_mode.name if self.fsm._failure_mode else "NONE",
                self.fsm._consecutive_failures,
            )
            return await self._try_primary_then_fallback(context, deadline)

        eta_s = max(0, self.fsm.recovery_eta() - time.monotonic())
        logger.info(
            "[CandidateGenerator] Primary in backoff (mode=%s, ETA=%.0fs), "
            "using fallback",
            self.fsm._failure_mode.name if self.fsm._failure_mode else "NONE",
            eta_s,
        )
        return await self._call_fallback(context, deadline)

    # ------------------------------------------------------------------
    # Route-specific generation strategies (Manifesto §5)
    # ------------------------------------------------------------------

    async def _local_jprime_endpoint(self) -> Optional[str]:
        """The ALWAYS-ON local J-Prime lane's endpoint, when it can serve.

        Sources 1-2 in :meth:`_discover_jprime_endpoint` both answer "was a GCP
        failover node awakened?". That was the same question as "is J-Prime
        available?" only while J-Prime was exclusively a cloud VM. An operator
        serving the 32B locally has a J-Prime endpoint permanently -- there is no
        node to awaken and no lifecycle FSM to enter SERVING, so discovery
        returned None and every dispatch fell to a DW lane that may not be
        funded at all.

        Enabled AND reachable, never enabled alone: an endpoint that does not
        answer would COMMIT the Phase 3c seam (``_committed = True``), and under
        Absolute Route Sealing a committed dispatch failure is TERMINAL -- so a
        flag pointing at a dead engine would convert a survivable DW attempt into
        an unrecoverable op. Reachability is the difference between a lane and a
        promise.

        Reads the endpoint from ``LocalConfig.from_env()`` and probes the same
        ``/api/tags`` readiness surface ``_resolve_dispatch_model_name`` uses, so
        this cannot disagree with the client that will serve the request. Bounded
        and memoized-by-nothing (residency changes; a stale yes is worse than a
        second probe). Fail-soft -- NEVER raises."""
        try:
            from .local_inference_director import LocalConfig, local_prime_enabled
            if not local_prime_enabled():
                return None
            base = (LocalConfig.from_env().base_url or "").strip()
            if not base:
                return None
        except Exception:  # noqa: BLE001
            return None
        try:
            import aiohttp  # noqa: PLC0415
            timeout_s = _envf_or_default("JARVIS_LOCAL_JPRIME_PROBE_S", 4.0)
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(base.rstrip("/") + "/api/tags") as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json(content_type=None)
            # A reachable engine serving zero models is not a lane.
            if not (data or {}).get("models"):
                return None
            return base
        except Exception:  # noqa: BLE001 -- unprovable lane == no lane
            return None

    async def _discover_jprime_endpoint(self) -> Optional[str]:
        """Dynamic state-driven discovery of a SERVING J-Prime endpoint.

        The failover node awakens at RUNTIME, so we do NOT rely on a boot-wired
        ``self._jprime``. Three dynamic sources, in order:
          (1) the failover controller's PUBLISHED endpoint when it is SERVING
              (fast path -- same as the Phase 3c seam);
          (2) a direct zone-aware GCP query (``get_node_endpoints``) -- works even
              when THIS process's controller is not in SERVING state (e.g. the node
              was awakened out-of-band by the ignition driver), composing
              ``http://<ip>:<port>``;
          (3) the ALWAYS-ON local lane (:meth:`_local_jprime_endpoint`).

        Source 3 is consulted LAST, deliberately: sources 1-2 are byte-identical
        to their previous behaviour, so a deployment with a real failover node
        keeps using it and nothing about GCP routing changes. Local only fills
        the case where discovery previously returned None -- which, on a host
        with no GCP failover at all, is every case.

        The GCP sources remain gated by ``lifecycle_enabled()``; the local lane
        is gated by ``local_prime_enabled()`` (also default-OFF) and must NOT
        inherit the failover gate, because a locally-served 32B has no failover
        lifecycle to enable. Returns a full URL or None. Fail-soft -- NEVER
        raises."""
        gcp_enabled = False
        try:
            from .failover_lifecycle import (
                lifecycle_enabled, get_failover_controller, _failover_port,
            )
            gcp_enabled = bool(lifecycle_enabled())
        except Exception:  # noqa: BLE001
            gcp_enabled = False
        if gcp_enabled:
            # (1) Controller-published endpoint (fast path).
            try:
                ctrl = get_failover_controller()
                if ctrl.is_jprime_serving():
                    ep = ctrl.jprime_endpoint()
                    if ep:
                        return ep
            except Exception:  # noqa: BLE001
                pass
            # (2) Direct zone-aware GCP discovery (controller-state-independent).
            try:
                from .gcp_compute_rest import get_compute_rest
                internal, external = await get_compute_rest().get_node_endpoints()
                ip = external or internal
                if ip:
                    return "http://%s:%d" % (ip, _failover_port())
            except Exception:  # noqa: BLE001
                pass
        # (3) The local lane -- no node to awaken, no FSM to enter SERVING.
        return await self._local_jprime_endpoint()

    async def _resolve_dispatch_model_name(self, endpoint: Optional[str]) -> Optional[str]:
        """The model the awakened J-Prime node ACTUALLY serves (loaded in VRAM) --
        the deterministic L7 source of truth, queried from the node's own
        ``/api/tags`` at *endpoint* and memoized per-endpoint.

        Deliberately abandons the lagging FSM ``_active_model_label``: when the
        endpoint is discovered by direct GCP query (controller not SERVING
        in-process) that label is empty, so the old path fell back to the survival
        7B and the node rejected the request with ``KeyError('choices')``. Asking
        the node itself is race-free.

        Memoized: fetched ONCE per endpoint, then served from cache -- no
        per-dispatch/-retry network spam. A new endpoint (node changed / re-awaken
        at a new IP) is a natural cache miss; ``_invalidate_jprime_model_cache()``
        clears it on FSM->DORMANT. None if undeterminable (caller keeps the env
        default). Fail-soft -- NEVER raises."""
        return await _resolve_served_model(endpoint)

    def _failover_profiler_for(self, endpoint: str, cfg: "Any") -> "Any":
        """Session-scoped LatencyProfiler singleton, one per failover endpoint,
        held on this generator so its EWMA/sample state persists across ops + L7
        retries (cures profiler amnesia). A new endpoint (re-awaken) gets a fresh
        profiler. Fail-soft -> a fresh profiler if the store is unavailable."""
        try:
            store = getattr(self, "_failover_profilers", None)
            if store is None:
                store = {}
                self._failover_profilers = store  # type: ignore[attr-defined]
            prof = store.get(endpoint)
            if prof is None:
                from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
                    LatencyProfiler,
                    physics_key,
                )
                # The Amnesia Cure: key the profiler by its DURABLE physics
                # identity so the EWMA warm-starts from the cross-run ledger.
                # That identity is (hardware, model, ctx) -- the ADDRESS is
                # still excluded, because IPs change every run, but the
                # MACHINE is not, because a 16GB Mac and a 5090 measuring the
                # same model are not the same physics. `endpoint` is passed
                # explicitly: this store holds one profiler per endpoint while
                # carrying one cfg, so deriving the machine from cfg would
                # stamp every failover target with the base config's identity.
                prof = LatencyProfiler(
                    cfg, ledger_key=physics_key(cfg, endpoint=endpoint))
                store[endpoint] = prof
            return prof
        except Exception:  # noqa: BLE001
            from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
                LatencyProfiler,
            )
            return LatencyProfiler(cfg)

    async def _negotiate_num_ctx(
        self, endpoint: Optional[str], *, model: Optional[str] = None,
    ) -> Optional[int]:
        """Autonomous Context-Hardware Negotiator: derive the VRAM-safe context
        window for *endpoint* from the MEASURED buffer -- node VRAM (awakened GPU
        tier) minus the served model's on-disk bytes (its own /api/tags) -- so the
        KV cache can never overflow VRAM (the warm-32B ServerDisconnect root cause).
        None when undeterminable (not a GPU tier / size unknown) -> caller keeps the
        legacy path. Fail-soft -- NEVER raises.

        When ``model`` is supplied AND model-physics autodetect is on, two inputs
        stop being global constants and become properties OF THAT MODEL:

          * ``kv_bytes_per_token`` -- architectural (layers x kv-heads x
            (key+value) x dtype), spanning 57,344..262,144 across the local
            fleet. One constant for all of them is necessarily wrong for most.
          * ``ceiling`` -- the STRICTER of the operator's configured cap and the
            model's NATIVE trained context. This is the decoupling that matters:
            VRAM says how much context fits, the model says how much it can
            actually use, and those are different limits. qwen2.5-coder is 32K
            native while qwen3-coder:30b is 262K, so a single global ceiling
            either starves the MoE or pushes the dense model past its training.

        Physics unavailable -> both fall back to the global env constants, which
        is exactly the pre-existing behaviour.

        ``model`` is an OPTIONAL injection point for tests and is NOT passed by
        the production caller. It is resolved internally instead, deliberately:
        this method is overridden by stubs and subclasses that implement the
        one-argument signature, so widening the CALL to pass a second argument
        would break them at runtime -- a contract change disguised as an
        enhancement. Resolution is memoized per-endpoint, so doing it here costs
        one cached lookup rather than a second round trip."""
        try:
            model_bytes = await _resolve_served_model_bytes(endpoint)
            vram_bytes = _awakened_vram_bytes()
            if not model_bytes or not vram_bytes:
                return None
            from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
                derive_safe_num_ctx,
            )
            kwargs: Dict[str, Any] = {}
            try:
                from . import local_inference_director as _lid  # noqa: PLC0415
                from .model_physics import (  # noqa: PLC0415
                    effective_ceiling,
                    physics_autodetect_enabled,
                    resolve_model_physics,
                )
                # Resolve the served model ONLY when physics is armed: gate off
                # means not even a cached lookup runs, so the legacy path stays
                # byte-identical in work performed, not merely in result.
                if model is None and physics_autodetect_enabled():
                    model = await self._resolve_dispatch_model_name(endpoint)
                physics = await resolve_model_physics(endpoint, model)
                if physics is not None:
                    kwargs["kv_bytes_per_token"] = physics.kv_bytes_per_token
                    # Read the configured ceiling through the SAME env name and
                    # the SAME default constant derive_safe_num_ctx itself uses,
                    # so the two can never disagree about what "unset" means.
                    configured = _lid._int_env(
                        "JARVIS_NUM_CTX_CEILING", _lid._NUM_CTX_CEILING_DEFAULT)
                    kwargs["ceiling"] = effective_ceiling(physics, configured)
            except Exception:  # noqa: BLE001 -- physics is advisory, never required
                kwargs = {}
            return derive_safe_num_ctx(
                vram_bytes=vram_bytes, model_bytes=model_bytes, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    async def _failover_local_dispatch(
        self,
        context: OperationContext,
        deadline: datetime,
        endpoint: str,
    ) -> Optional[GenerationResult]:
        """Phase 3c -- run generation through the LocalPrimeClient at *endpoint*.

        Builds a ``PrimeProvider`` wrapping a ``LocalPrimeClient`` pointed at the
        awakened J-Prime node's OpenAI-compatible endpoint and calls its
        ``.generate(context, deadline)``. The PrimeProvider seat produces the exact
        ``GenerationResult`` shape the dispatch returns, so APPLY/VERIFY downstream
        is byte-identical.

        Three intelligences layered here (all reuse existing seats -- no new
        provider, nothing hardcoded):
          * model name = the node's OWN /api/tags truth (deterministic L7), so the
            request never asks for a model the node lacks (the KeyError('choices'));
          * ``num_ctx`` = the Context-Hardware Negotiator's VRAM-safe window, which
            drives Dynamic Cognitive Compression inside ``LocalPrimeClient`` so the
            KV cache can never overflow VRAM;
          * Resilient L7 Recovery: on a transient connection drop (a warm worker
            that crashed mid-request -- e.g. a KV-cache OOM), AUTO-HEAL by
            re-warming + tightening num_ctx and retrying, up to N times, before
            raising (which lets the sentinel seam SEAL/halt -- never cascade).

        Bounded by the op's own remaining budget. Returns the ``GenerationResult``
        on success or ``None`` to fall through. Raises only when the auto-heal is
        exhausted on a recoverable fault (so sealing halts) or on a non-recoverable
        error (the caller's try/except is the final net).
        """
        # Do not generate from a model that is being replaced. Reactor's
        # gpu_lease stops a deploy landing while a leased job runs; this is
        # the other half, and without it `ollama create` can swap the blob
        # under a stream. Returning None PARKS the dispatch through the
        # existing fall-through, so the op re-queues rather than failing.
        try:
            from backend.core.ouroboros.governance.gpu_deployment_gate import (  # noqa: E501, PLC0415
                deployment_in_progress as _gpu_deploy_busy,
            )
            _blocked, _why = await _gpu_deploy_busy()
            if _blocked:
                logger.info(
                    "[CandidateGenerator] deferring local dispatch op=%s: %s",
                    getattr(context, "op_id", "?")[:16], _why,
                )
                return None
        except Exception:  # noqa: BLE001 -- fail OPEN, never block the loop
            logger.debug(
                "[CandidateGenerator] deployment gate probe failed",
                exc_info=True,
            )

        import dataclasses as _f3c_dc

        from backend.core.ouroboros.governance.local_inference_director import (
            GracefulStreamInterruption,
            LocalConfig as _F3cLocalConfig,
            LocalPrimeClient as _F3cLocalPrimeClient,
        )
        from backend.core.ouroboros.governance.providers import (
            PrimeProvider as _F3cPrimeProvider,
        )

        # Pre-SERVING readiness gate: a committed dispatch against a node that
        # is still booting dies on the 30s survival probe and gets SEALED
        # (bt-iso-1782959216: 20/23 dispatches). Wait (bounded, heartbeat-
        # pulsed, suspend-aware) for /api/tags before building the client --
        # the model resolver + num_ctx negotiator then see the REAL node.
        await _await_jprime_ready(
            endpoint, op_id=getattr(context, "op_id", "?")[:16],
        )
        _base_overrides: Dict[str, Any] = {
            "base_url": endpoint,
            # Deterministic VRAM residency: keep the model resident while we route
            # to this node (no ~109s reload between ops); the FSM reap flushes it.
            "keep_alive_seconds": _failover_keep_alive_seconds(),
        }
        _model = await self._resolve_dispatch_model_name(endpoint)
        if _model:
            _base_overrides["model_name"] = _model
        # Gateway residency handshake. The device can hold ONE large model, so
        # "which model is resident" is shared state that two concurrent
        # dispatches can race on: both read the same snapshot, both see a
        # mismatch, both request a load, and the loser is not a queued request
        # but a partial offload to system RAM for BOTH. The gateway already owns
        # the mutex, the capacity admission and the warm-swap budget for exactly
        # this -- but only for dispatches routed through it, and this seam was
        # not one of them. Running the handshake here puts the live local path
        # under the same lock as everything else instead of beside it.
        #
        # Advisory=False: this is a real op that NEEDS its model, unlike a
        # speculative pre-warm which must defer. Fail-soft and non-blocking on
        # error -- an unavailable gateway must never cost an op, so the dispatch
        # proceeds exactly as it did before this call existed.
        if _model:
            await _f3c_gateway_residency(endpoint, _model, context)
        _num_ctx = await self._negotiate_num_ctx(endpoint)
        _op = getattr(context, "op_id", "?")[:16]
        if _num_ctx:
            logger.info(
                "[CandidateGenerator] Context-Hardware Negotiator: VRAM-safe "
                "num_ctx=%d for the 32B at %s op=%s", _num_ctx, endpoint, _op,
            )

        # Stateful, session-scoped Latency Profiler kept PER ENDPOINT on this
        # generator (EWMA survives across ops + L7 retries -- cures profiler
        # amnesia). Built ONCE from the initial cfg; the L7 tighten rebuilds the
        # client but reuses this profiler. A new endpoint (re-awaken) -> fresh one.
        _init_overrides = dict(_base_overrides)
        if _num_ctx:
            _init_overrides["num_ctx"] = int(_num_ctx)
        _init_cfg = _f3c_dc.replace(_F3cLocalConfig.from_env(), **_init_overrides)
        _prof = self._failover_profiler_for(endpoint, _init_cfg)
        # Time-Dilated Sovereign Deadline: derive the committed dispatch's
        # runway from THIS node's measured round physics (never from the
        # scraps a spent route budget left behind).
        deadline = _dilate_sovereign_deadline(deadline, _prof, int(_num_ctx or 0))

        # LLM Prefill Re-Ignition: if this op is a RESUME, its checkpointed partial
        # thought rides in the intake evidence -> feed it to the client as a prefill
        # so the 32B continues from the interrupted character (no re-generation).
        _resume_prefill = ""
        try:
            import json as _rj  # noqa: PLC0415
            _ev_raw = getattr(context, "intake_evidence_json", "") or ""
            if _ev_raw:
                _ev = _rj.loads(_ev_raw)
                _resume_prefill = str((_ev or {}).get("partial_completion", "") or "")
        except Exception:  # noqa: BLE001
            _resume_prefill = ""
        if _resume_prefill:
            # Atomic Hydration Handshake (facet 2): the EXACT bytes + snippet of the
            # partial thought re-entering the LLM as the assistant prefill -> stdout.
            try:
                from backend.core.ouroboros.governance import (  # noqa: PLC0415
                    fsm_checkpoint as _hs_ckpt,
                )
                _hs_ckpt.emit_handshake(
                    _hs_ckpt.format_prefill_handshake(_op, _resume_prefill)
                )
            except Exception:  # noqa: BLE001
                logger.info(
                    "[CandidateGenerator] RESUME prefill: continuing a %d-char partial "
                    "thought op=%s (32B resumes typing, no re-generation)",
                    len(_resume_prefill), _op,
                )

        async def _attempts(sampling: Any = None) -> Optional[GenerationResult]:
            """One full generation attempt (with L7 auto-heal retries).

            ``sampling`` is an optional ``SiblingSampling`` naming this
            DRAW's point in sampling space. None -> the legacy point, so
            ``run_calibrated(_attempts)`` (which calls this with no
            arguments) is byte-identical to the pre-entropy path. The
            point rides the SAME override seam every other per-call
            setting uses, so no request-building logic is duplicated.
            """
            nonlocal _num_ctx
            _n = _l7_recovery_attempts()
            _last_exc: Optional[BaseException] = None
            for _try in range(_n + 1):
                _overrides = dict(_base_overrides)
                if _num_ctx:
                    _overrides["num_ctx"] = int(_num_ctx)
                if sampling is not None:
                    _overrides.update(sampling.config_overrides())
                _cfg = _f3c_dc.replace(_F3cLocalConfig.from_env(), **_overrides)
                _client = _F3cLocalPrimeClient(_cfg, profiler=_prof)
                if _resume_prefill:
                    _client._resume_prefill = _resume_prefill  # continue the partial
                try:
                    # Venom on the sovereign path (the 6/6 last boss,
                    # bt-iso-1782960801: 19 one-shot streams, 40 Iron Gate
                    # rejections, ZERO tool executions). Hand the primary
                    # provider's already-wired ToolLoopCoordinator (the SAME
                    # loop the DW path runs; governed_loop_service wires it
                    # into every provider seat) to the local PrimeProvider --
                    # _generate_impl then takes the multi-turn tool branch:
                    # tool advertisements, 2b.2-tool envelope parsing, REAL
                    # read_file/search_code executions, with_tool_records()
                    # crediting -> the exploration-first Iron Gate becomes
                    # satisfiable on the 32B. Master JARVIS_JPRIME_VENOM_ENABLED
                    # (default true); =false restores the one-shot legacy.
                    _venom_on = (os.environ.get(
                        "JARVIS_JPRIME_VENOM_ENABLED", "true") or "").strip().lower() \
                        not in ("0", "false", "no", "off")
                    _provider = _F3cPrimeProvider(
                        _client,
                        repo_root=self._repo_root if hasattr(self, "_repo_root") else None,
                        tool_loop=(getattr(getattr(self, "_primary", None),
                                           "_tool_loop", None)
                                   if _venom_on else None),
                        mcp_client=(getattr(getattr(self, "_primary", None),
                                            "_mcp_client", None)
                                    if _venom_on else None),
                    )
                    remaining = self._remaining_seconds(deadline)
                    if remaining <= 0.0:
                        return None
                    # Master Wall Yielding (constraint 3, invariant-safe): on the
                    # STREAMING path DROP the outer op-deadline wait_for -- an
                    # actively-emitting stream must run indefinitely, bounded ONLY by
                    # the per-chunk inter-token watchdog (inside the client) + the
                    # STATIC hard wall-clock cap (kept blind per Slice-47). The
                    # non-streaming survival path keeps the op-deadline cap.
                    try:
                        from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
                            _streaming_enabled as _f3c_stream_on,
                        )
                        _streaming = bool(_num_ctx) and _f3c_stream_on()
                    except Exception:  # noqa: BLE001
                        _streaming = False
                    # Gateway unification: bracket this generation in the
                    # InferenceGateway's in-flight counter. The generation stays
                    # HERE -- the PrimeProvider seat carries the Venom tool loop,
                    # which gateway.dispatch() (a single client.complete call)
                    # cannot express, so routing through it would either lose the
                    # tool loop or duplicate it. What the gateway must own is
                    # RESIDENCY STATE, not the call: its advisory pre-warm refuses
                    # to swap only while it believes weights are in use, and a
                    # generation it cannot see reads as idle. Unbracketed, a
                    # pre-warm could evict the 32B mid-stream -- exactly the
                    # failure the mutex exists to prevent, blind on the one path
                    # that actually generates. Fail-soft: no gateway -> a
                    # null bracket, and the op proceeds unchanged.
                    _bracket = _f3c_inflight_bracket(endpoint)
                    # The WHOLE sampling point crosses this seam, not just
                    # temperature.
                    #
                    # This used to forward `temperature` alone, because that
                    # was the one parameter `PrimeProvider.generate` already
                    # accepted (the T2 epistemic override). The ladder's
                    # top_p/top_k/repeat_penalty/seed were computed, printed
                    # by `describe()` into the sibling log line, and dropped
                    # right here -- so the log showed `T=1.10 top_p=0.90
                    # top_k=140 rp=1.15 seed=...` while the request carried a
                    # temperature and nothing else. Soak bt-2026-09-02-025257
                    # measured the result: structural similarity 1.0000
                    # between siblings drawn at different seeds, one group of
                    # 8 draws collapsing to a single structure_id, and a
                    # reward spread of 6e-05 where GRPO needs a real gap.
                    # `sibling_entropy`'s ladder comment had already named the
                    # cause: raising temperature alone re-weights a tail that
                    # top_k/top_p have truncated away.
                    #
                    # `config_overrides()` stays the single definition of
                    # which fields a point sets -- this seam does not
                    # re-derive it, it just carries the object.
                    _gen_kw: Dict[str, Any] = {}
                    if sampling is not None and not sampling.is_legacy:
                        _gen_kw["temperature"] = float(sampling.temperature)
                        _gen_kw["sampling"] = sampling
                    if _streaming:
                        async with _bracket:
                            return await _provider.generate(
                                context, deadline, **_gen_kw,
                            )
                    async with _bracket:
                        return await asyncio.wait_for(
                            _provider.generate(context, deadline, **_gen_kw),
                            timeout=remaining,
                        )
                except GracefulStreamInterruption as _gsi:
                    # Cooperative freeze-mid-sentence. GSI is a BaseException by design
                    # (the Earmuff Bypass) -- it pierced the Venom tool loop's
                    # `except Exception` and lands HERE at the checkpoint boundary.
                    # Capture the partial thought + write the checkpoint
                    # DETERMINISTICALLY (we hold both the ctx and the partial), avoiding
                    # the registry-unregister race, then re-raise so the op suspends
                    # (never retried -- resumes next ignition via prefill).
                    _last_exc = _gsi
                    try:
                        from backend.core.ouroboros.governance import (  # noqa: PLC0415
                            fsm_checkpoint as _gsi_ckpt,
                        )
                        _partial = getattr(_gsi, "partial", "") or ""
                        _gsi_ckpt.stash_partial(_op, _partial)
                        _cp = _gsi_ckpt.capture_from_context(
                            context, phase="GENERATE",
                            resume_reason="graceful_stream_interruption",
                        )
                        if _cp is not None:
                            _gsi_ckpt.write_checkpoint(_cp)
                            logger.warning(
                                "[CandidateGenerator] FROZE mid-stream op=%s -> "
                                "checkpointed %d-char partial thought; resumes "
                                "next ignition via prefill", _op, len(_partial),
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    raise
                except Exception as _exc:  # noqa: BLE001
                    _last_exc = _exc
                    # Non-recoverable OR out of retries -> propagate (sentinel seals).
                    if _try >= _n or not _is_l7_recoverable(_exc):
                        raise
                    # AUTO-HEAL: tighten the window + re-warm, then retry.
                    if _num_ctx:
                        _num_ctx = max(512, int(_num_ctx * _l7_tighten_factor()))
                    logger.warning(
                        "[CandidateGenerator] L7 AUTO-HEAL: %s on the 32B -> re-warm "
                        "+ tighten num_ctx=%s, retry %d/%d op=%s",
                        type(_exc).__name__, _num_ctx, _try + 1, _n, _op,
                    )
                    try:
                        await _client.warmup(timeout_s=_l7_rewarm_timeout_s())
                    except Exception:  # noqa: BLE001
                        pass
                finally:
                    try:
                        await _client.aclose()
                    except Exception:  # noqa: BLE001
                        pass
            if _last_exc is not None:
                raise _last_exc
            return None

        # Async Calibration Mutex (Scout Lock): on a COLD profiler the first
        # concurrent op scouts (calibrates the EWMA) while the herd awaits the lock;
        # once calibrated the herd runs CONCURRENTLY on the escalated seed -- the DAG
        # is never serialized, only the one-shot cold calibration is gated.
        _first = await _prof.run_calibrated(_attempts)
        return await self._extend_with_siblings(
            _first, _attempts, context, deadline,
            resume_prefill=_resume_prefill,
        )

    async def _extend_with_siblings(
        self,
        first: Optional[GenerationResult],
        attempt: Any,
        context: OperationContext,
        deadline: datetime,
        *,
        resume_prefill: str = "",
    ) -> Optional[GenerationResult]:
        """Draw additional candidates for the SAME op, sequentially.

        A preference pair needs two answers to one question. The local lane
        produced exactly one per op, so the DPO corpus had nothing to pair
        and every farming soak yielded zero pairs no matter how long it ran.

        SEQUENTIAL, not ``asyncio.gather``, on measurement rather than
        taste: on this host three concurrent generations finish in 2.6s
        against 2.7s sequential -- 1.04x -- because the engine serializes
        them onto one device anyway (the same reason the local "swarm" is
        sequential chunking). Concurrency here would buy 4% while putting
        three inter-token watchdogs, three streams and three num_ctx
        negotiations in flight against one GPU. The watchdog TTL "reset"
        this needs is structural and already true: each sibling builds its
        own client and its own stream, so the inter-token deadline starts
        fresh per sibling with no timer to reach into and reset.

        Siblings are STRICTLY ADDITIVE and best-effort. The first candidate
        is produced exactly as before; every sibling is attempted only if
        the op's own remaining budget can already pay for it, measured
        against what the previous one actually cost. The op deadline is
        never extended -- a sibling is a bonus drawn from slack, never a
        reason to overrun. So on a tight budget this degrades silently to
        today's single candidate, and a sibling that fails or stalls can
        never turn a working op into a failed one.

        ## Diversity is DRAWN, not hoped for

        This once read "diversity comes from sampling temperature, which
        the provider seat already applies (measured: 3/3 distinct outputs
        at temperature 0.7 and 1.0)". That experiment was never wired to
        this path: ``PrimeProvider._generate_impl`` computes
        ``_eff_temperature = 0.2 if temperature is None else ...`` and
        nothing here passed a temperature, so every sibling was drawn at
        0.2 -- a near-deterministic distribution sampled N times.

        Measured consequence on the live corpus (2026-09-01): 8 shipped
        sibling rows carried **3 structurally distinct answers**; all
        three groups collapsed to ONE fingerprint each, so not a single
        preference pair was constructible. Peak structural similarity
        between "distinct" siblings was 0.9987 -- they differed by
        docstring wording and one unused import.

        So each draw now gets its own point in sampling space
        (``sibling_entropy.sampling_for``), and a draw that adds no logic
        is REJECTED and re-taken at higher entropy rather than persisted.
        Draw 1 keeps the legacy point exactly, so the candidate an op
        would have produced anyway is byte-identical.

        Dedup is structural, not byte-wise. ``candidate_hash`` equality
        was the wrong predicate: two candidates differing in one docstring
        word have different hashes and identical logic, which is exactly
        the pair the corpus was full of. It is kept as a cheap first pass.

        ## An answer is not a dead lane

        Soak bt-2026-09-03-072129 recorded 36 ops with a primary and 15
        of them with no sibling row at all. Joining the corpus to the log
        accounted for every one:

          8  the sibling came back ``2b.1-noop`` ("already implemented").
             Zero candidates, so it read as EMPTY, and empty meant
             ``_stop`` -- the remaining slot was never attempted.
          6  redundant after the one re-draw allowed, at similarity
             1.0000 in 14 of the 15 drops session-wide: the re-draw was
             taken one rung up, INSIDE the region that had just collapsed.
          2  (session-wide) ``all_candidates_syntax_error`` -- an
             exception, so ``_stop`` again.

        A no-op is the model answering the prompt -- contradicting the
        primary it just produced -- and an unparseable draw is a bad
        answer. Neither is evidence the engine is gone. Both now escalate
        and re-draw within the slot like a redundant draw does, and when
        the slot's re-draw budget is spent the SLOT is dropped and the
        next one is tried. ``None`` and any other exception still stop:
        those are the lane, not the answer.

        Every slot that collapses (redundant / no-op / unparseable at the
        cap) extends a streak; a merged draw resets it. The streak is
        handed to ``sibling_entropy.sampling_for`` where, with the
        multiplier on, it jumps rungs exponentially instead of stepping.

        The exit line ``sibling_fulfillment`` is emitted for EVERY op with
        n_want > 1, singletons included. The summary that follows it only
        ever fired when something merged, which is why an op that lost
        every slot left no trace and the pathology hid in aggregates.
        """
        n_want = _sibling_candidate_count()
        if first is None or n_want <= 1 or not getattr(first, "candidates", None):
            return first
        # A RESUME is continuing one specific interrupted thought. Drawing
        # alternatives to it would be answering a different question.
        if resume_prefill:
            return first

        from backend.core.ouroboros.governance import (  # noqa: PLC0415
            sibling_entropy as _ent,
        )

        _op = (getattr(context, "op_id", "") or "?")[:16]
        merged = list(first.candidates)
        seen = {
            str((c or {}).get("candidate_hash", "") or "")
            for c in merged if isinstance(c, dict)
        }
        _dupes = 0
        _redundant = 0
        _resamples = 0
        # Structural fingerprints already accepted into this group. The
        # ACCEPTANCE test, where `seen` above is only the cheap byte-wise
        # pre-filter.
        # Fingerprint against the file each candidate proposes to REPLACE
        # (hunk-level) when the repo root is known; whole-file otherwise.
        # Resolved the way every other path in this class is resolved.
        _fp_root = getattr(self, "_repo_root", None) or None
        _seen_fps: List[str] = list(
            _ent.fingerprint_candidates(merged, _fp_root),
        )
        _max_resample = _ent.max_resample_attempts()
        # Seed the affordability estimate with what the FIRST candidate
        # actually cost on this model and this prompt. A static per-sibling
        # guess is the thing that would make this either skip siblings a
        # fast model could afford, or start one a slow model cannot finish.
        _last_cost = max(
            0.05, float(getattr(first, "generation_duration_s", 0.0) or 0.0),
        )

        _noops = 0
        _unparseable = 0
        _stop = False
        # Consecutive slots that COLLAPSED -- every draw they were allowed
        # came back redundant, a no-op, or unparseable. A merged draw
        # resets it: the region is not dead after all.
        _streak = 0
        # One entry per slot, every transition it went through. This is the
        # fulfillment ledger: it is what makes a lost slot a logged fact.
        _ledger: List[str] = []
        for _i in range(2, n_want + 1):
            if _stop:
                _ledger.append(f"{_i}:not_attempted")
                continue
            # One SLOT, possibly several draws: a redundant draw is re-taken
            # at higher entropy rather than persisted. Bounded by
            # `_max_resample`, and every re-draw still pays the same budget
            # test as a first draw -- a diversity retry can never overrun
            # the op any more than a sibling could.
            _escalation = 0
            _trail: List[str] = []
            while True:
                _t0 = time.monotonic()
                remaining = self._remaining_seconds(deadline)
                # Pay only out of slack we can already see. `_last_cost` is what
                # the PREVIOUS sibling actually took, so the estimate tracks the
                # real model on the real prompt instead of a static guess.
                _needed = _last_cost * _sibling_budget_margin()
                if remaining <= _needed:
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d skipped op=%s: "
                        "%.1fs budget left, previous sibling cost %.1fs",
                        _i, n_want, _op, remaining, _last_cost,
                    )
                    _trail.append("budget")
                    _stop = True
                    break
                _sampling = _ent.sampling_for(
                    _i, escalation=_escalation, op_id=_op,
                    collapse_streak=_streak,
                )
                try:
                    _sib = await self._profiler_for_siblings(attempt, _sampling)
                except Exception as _sib_exc:  # noqa: BLE001
                    _last_cost = max(0.05, time.monotonic() - _t0)
                    # Unparseable Python is a BAD ANSWER, not a dead engine:
                    # the provider parsed valid JSON and found invalid code
                    # inside it, and says so through the same
                    # `syntax_failures` attribute the primary path's repair
                    # keys on. Re-draw it like a redundant draw. Nothing to
                    # retract -- the provider raised before the recorder
                    # was ever told.
                    if getattr(_sib_exc, "syntax_failures", None):
                        _unparseable += 1
                        if _escalation < _max_resample:
                            _resamples += 1
                            _escalation += 1
                            logger.info(
                                "[CandidateGenerator] sibling %d/%d op=%s "
                                "unparseable at %s -- re-drawing at higher "
                                "entropy (%s)",
                                _i, n_want, _op, _sampling.describe(),
                                _ent.sampling_for(
                                    _i, escalation=_escalation, op_id=_op,
                                    collapse_streak=_streak,
                                ).describe(),
                            )
                            _trail.append("unparseable")
                            continue
                        logger.info(
                            "[CandidateGenerator] sibling %d/%d op=%s still "
                            "unparseable after %d re-draw(s) -- slot dropped",
                            _i, n_want, _op, _escalation,
                        )
                        _trail.append("unparseable(dropped)")
                        _streak += 1
                        break
                    # A sibling is a bonus. Losing one must never cost the op
                    # the candidate it already has -- so this swallows, where
                    # the FIRST attempt deliberately propagates.
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d failed op=%s (%s: %s)"
                        " -- keeping %d candidate(s)",
                        _i, n_want, _op, type(_sib_exc).__name__,
                        _trim_exc_msg(_sib_exc), len(merged),
                    )
                    _trail.append(f"failed:{type(_sib_exc).__name__}")
                    _stop = True
                    break
                _last_cost = max(0.05, time.monotonic() - _t0)
                _has_cands = bool(getattr(_sib, "candidates", None))
                if _sib is not None and not _has_cands and getattr(_sib, "is_noop", False):
                    # The model answered "already done" at this point in
                    # sampling space. That is an ANSWER to the prompt -- the
                    # one the primary just contradicted -- and the recorder
                    # never queued it (no candidates), so there is nothing to
                    # retract. Re-draw at higher entropy; at the cap, drop
                    # the SLOT, not the loop.
                    _noops += 1
                    if _escalation < _max_resample:
                        _resamples += 1
                        _escalation += 1
                        logger.info(
                            "[CandidateGenerator] sibling %d/%d op=%s "
                            "declined (2b.1-noop at %s) -- re-drawing at "
                            "higher entropy (%s)",
                            _i, n_want, _op, _sampling.describe(),
                            _ent.sampling_for(
                                _i, escalation=_escalation, op_id=_op,
                                collapse_streak=_streak,
                            ).describe(),
                        )
                        _trail.append("noop")
                        continue
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d op=%s still a "
                        "no-op after %d re-draw(s) -- slot dropped",
                        _i, n_want, _op, _escalation,
                    )
                    _trail.append("noop(dropped)")
                    _streak += 1
                    break
                if _sib is None or not _has_cands:
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d empty op=%s -- stopping",
                        _i, n_want, _op,
                    )
                    _trail.append("empty")
                    _stop = True
                    break

                _fresh = [_c for _c in _sib.candidates if isinstance(_c, dict)]
                _new_fps = _ent.fingerprint_candidates(_fresh, _fp_root)
                _is_red, _peak = _ent.is_structurally_redundant(
                    _new_fps, _seen_fps,
                    hunks=_ent.hunks_for_candidates(_fresh, _fp_root),
                )
                if _is_red and _escalation < _max_resample:
                    _redundant += 1
                    _resamples += 1
                    _escalation += 1
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d op=%s adds no "
                        "logic (structural similarity %.4f >= %.2f at %s) "
                        "-- re-drawing at higher entropy (%s)",
                        _i, n_want, _op, _peak, _ent.diversity_threshold(),
                        _sampling.describe(),
                        _ent.sampling_for(
                            _i, escalation=_escalation, op_id=_op,
                            collapse_streak=_streak,
                        ).describe(),
                    )
                    _trail.append(f"redundant:{_peak:.4f}")
                    # The provider already reported this draw to the
                    # recorder; without a retraction it lands in the corpus
                    # as a twin of the candidate it duplicates. Retract ONLY
                    # hashes not already accepted: a byte-identical twin
                    # carries the SAME candidate_hash as the candidate it
                    # duplicates, and retracting that hash would take the
                    # accepted generation with it. Measured in soak
                    # bt-2026-09-02-013719: one retract event removed two
                    # generations and the kept candidate's verdict orphaned.
                    _ent.retract_draw(
                        str(getattr(context, "op_id", "") or ""),
                        [_c for _c in _fresh
                         if str(_c.get("candidate_hash", "") or "") not in seen],
                        reason=f"redundant_redraw:{_peak:.4f}",
                    )
                    continue
                if _is_red:
                    # Entropy budget spent and it is still the same answer.
                    # Persisting it would write a row that cannot become
                    # half of a preference pair while looking like it could.
                    _redundant += 1
                    logger.info(
                        "[CandidateGenerator] sibling %d/%d op=%s still "
                        "redundant after %d re-draw(s) (similarity %.4f) "
                        "-- dropped",
                        _i, n_want, _op, _escalation, _peak,
                    )
                    _ent.retract_draw(
                        str(getattr(context, "op_id", "") or ""),
                        [_c for _c in _fresh
                         if str(_c.get("candidate_hash", "") or "") not in seen],
                        reason=f"redundant_dropped:{_peak:.4f}",
                    )
                    _trail.append(f"redundant(dropped):{_peak:.4f}")
                    _streak += 1
                    break

                for _c in _fresh:
                    _h = str(_c.get("candidate_hash", "") or "")
                    if _h and _h in seen:
                        _dupes += 1
                        continue
                    if _h:
                        seen.add(_h)
                    merged.append(_c)
                _seen_fps.extend(_new_fps)
                _trail.append("merged")
                _streak = 0
                break
            _ledger.append(f"{_i}:" + ">".join(_trail or ["?"]))

        # The fulfillment line fires for EVERY op that wanted siblings. It
        # is the one place a singleton is a fact rather than an absence.
        _got = len(merged) - len(first.candidates)
        logger.info(
            "[CandidateGenerator] sibling_fulfillment op=%s wanted=%d got=%d "
            "slots=[%s]%s%s%s",
            _op, n_want - 1, _got, ", ".join(_ledger),
            f" noops={_noops}" if _noops else "",
            f" unparseable={_unparseable}" if _unparseable else "",
            " -- SINGLETON" if _got == 0 else "",
        )
        if len(merged) == len(first.candidates):
            return first
        _distinct = _ent.distinct_structure_count(merged)
        logger.info(
            "[CandidateGenerator] op=%s drew %d candidate(s) from %d "
            "sibling generation(s), %d structurally distinct%s%s -- a "
            "preference pair needs two DIFFERENT answers to one question",
            _op, len(merged), n_want, _distinct,
            f" ({_dupes} identical, dropped)" if _dupes else "",
            f" ({_redundant} redundant, {_resamples} re-drawn)"
            if _redundant else "",
        )
        import dataclasses as _sib_dc
        return _sib_dc.replace(first, candidates=tuple(merged))

    async def _profiler_for_siblings(
        self, attempt: Any, sampling: Any = None,
    ) -> Optional[GenerationResult]:
        """Run one sibling attempt.

        Split out so the sibling loop reads as a loop. Deliberately NOT
        wrapped in ``run_calibrated``: the Scout Lock exists to serialize a
        COLD profiler's one-shot calibration, and by the time a sibling
        runs the profiler is warm by construction -- the first candidate
        just calibrated it. Re-entering the lock per sibling would
        serialize the herd for no reading.

        ``sampling`` names this draw's point in sampling space; None keeps
        the legacy point, which is what the calibration path passes.
        """
        return await attempt(sampling)

    async def _try_local_primary(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> Optional[GenerationResult]:
        """Serve ANY route on the local lane when no paid lane exists.

        `_try_free_lane_dispatch` made the free lane reachable for BACKGROUND
        and SPECULATIVE, and deliberately refused the rest: "a STANDARD op has
        a working cascade and an explicit cost contract; quietly moving it onto
        a local model would change what the operator paid for." That reasoning
        is sound while a paid lane exists. It is what stranded a SANCTIONED op
        in soak bt-2026-08-28-100733, which routed STANDARD by SOURCE (the
        UrgencyRouter keys on source, not urgency alone, so `source="roadmap"`
        never reaches BACKGROUND however low its urgency), then died
        `all_providers_exhausted:circuit_breaker_tripped` — with a warm 32B
        resident on the GPU the whole time.

        The gate is `_free_lane_active()`, and the choice of predicate is the
        whole design. It does not ask "which route is this" — route names are
        exactly the hardcoded mapping that produced the bug. It asks whether
        any PAID lane exists at all: it requires the local lane to be
        configured AND both provider credentials to be absent, re-reading
        `.env` on a TTL so a key added mid-run revokes this immediately. When
        no paid lane exists there is no cost contract to protect, so the
        objection above does not apply; when one does exist this returns None
        and every route cascades exactly as before, byte-identical.

        Evidence, not assertion: a reachable endpoint must answer before
        anything is dispatched. A configured-but-dead engine falls through to
        the legacy cascade rather than swallowing the op.

        Returns the result, or None to fall through — including on dispatch
        failure, so the local lane can only ever ADD a way for an op to
        succeed, never a new way for it to die.
        """
        if not _local_primary_enabled():
            return None
        if not _free_lane_active():
            # A paid lane exists (or the local lane is not configured). Do not
            # re-route work the operator is paying for.
            return None

        op_id_short = (getattr(context, "op_id", "") or "?")[:16]
        route = (getattr(context, "provider_route", "") or "standard").lower()
        try:
            endpoint = await self._discover_jprime_endpoint()
        except Exception:  # noqa: BLE001 — discovery is advisory
            return None
        if not endpoint:
            logger.debug(
                "[CandidateGenerator] local-primary declined: no endpoint "
                "(route=%s) [%s]", route, op_id_short,
            )
            return None

        logger.info(
            "[CandidateGenerator] local-primary: no paid lane is configured, "
            "serving route=%s on the local engine at %s BEFORE the cascade "
            "[%s]", route, endpoint, op_id_short,
        )
        try:
            result = await self._local_dispatch_with_syntax_repair(
                context, deadline, endpoint,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CandidateGenerator] local-primary dispatch failed "
                "(%s: %.120s) — falling through to the legacy cascade [%s]",
                type(exc).__name__, str(exc), op_id_short,
            )
            return None

        if result is None:
            return None
        if getattr(result, "is_noop", False):
            return result
        if len(getattr(result, "candidates", ()) or ()) > 0:
            return result
        return None

    async def _local_dispatch_with_syntax_repair(
        self,
        context: OperationContext,
        deadline: datetime,
        endpoint: str,
    ) -> Optional[GenerationResult]:
        """Local dispatch that shows the model its own parse errors once.

        `all_candidates_syntax_error` was the top non-governance failure on the
        local lane — 6 dispatches in soak bt-2026-08-28-061124, the model
        emitting VALID JSON wrapping INVALID Python. That is a recoverable
        slip, and it was being treated as terminal.

        `syntax_escalation` exists for this class, but it cascades DW →
        J-Prime, and on a workstation topology the local 32B *is* J-Prime — so
        it escalates to the model that just failed. The escalation is sound and
        simply has nowhere to go here; the missing move is not another provider
        but the feedback the first attempt never received.

        ONE retry, deliberately. Two would be a loop that spends the op's whole
        budget re-reading the same file; and if a model cannot fix a named
        `unexpected indent` on the second attempt, a third will not help. The
        retry is stamped through `with_syntax_retry_feedback`, which advances
        the hash-chain, so it is auditable as a distinct attempt rather than a
        silent re-run.

        Falls through to the original exception when the retry also fails, so
        the operator still reads the real terminal cause.
        """
        try:
            return await self._failover_local_dispatch(context, deadline, endpoint)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — inspected, then re-raised
            failures = getattr(exc, "syntax_failures", None)
            if not failures or not _syntax_repair_enabled():
                raise
            if getattr(context, "syntax_retry_feedback", ""):
                # Already the retry. A second correction round is the loop this
                # method exists to avoid.
                raise
            try:
                from backend.core.ouroboros.governance.providers import (
                    _format_syntax_feedback,  # noqa: PLC0415
                )
                feedback = _format_syntax_feedback(failures)
            except Exception:  # noqa: BLE001 — no feedback → no retry
                raise exc from None
            if not feedback:
                raise
            op_id_short = (getattr(context, "op_id", "") or "?")[:16]
            logger.info(
                "[CandidateGenerator] local lane returned unparseable Python "
                "(%d file(s), first: %s line %s) — retrying ONCE with the "
                "parse errors fed back [%s]",
                len(failures),
                (failures[0] or {}).get("file_path", "?"),
                (failures[0] or {}).get("line", "?"),
                op_id_short,
            )
            # Duck-typed contexts reach this path (the harness and several
            # call sites build minimal ones). A context that cannot CARRY the
            # feedback cannot benefit from the retry, and turning a diagnosed
            # syntax error into an AttributeError would replace a precise
            # terminal reason with a useless one.
            _stamp = getattr(context, "with_syntax_retry_feedback", None)
            if not callable(_stamp):
                logger.debug(
                    "[CandidateGenerator] context cannot carry syntax feedback "
                    "(%s) — not retrying [%s]",
                    type(context).__name__, op_id_short,
                )
                raise
            # Truncation is not a typo, and must not be retried as one.
            #
            # The first live firing (soak bt-2026-08-28-100733) showed the
            # distinction in one trace: line 231 "unterminated string literal"
            # → retry → line 196 "unterminated string literal". The model did
            # not mis-type a quote; its OUTPUT WAS CUT OFF, and the second
            # attempt was cut off earlier. Telling it "fix line 231" asks it to
            # repair a line it never finished writing, and a full-file retry
            # spends the same budget to hit the same ceiling.
            #
            # `force_diff_on_retry` already exists for exactly this: the
            # truncation-retry seam whose documented purpose is "change output
            # SHAPE on the next attempt instead of retrying with the same
            # parameters". A diff is a fraction of a whole file, so the payload
            # that overran the budget stops being the payload.
            if _truncation_shaped(failures) and _truncation_reshape_enabled():
                _reshape = getattr(context, "with_forced_diff_retry", None)
                logger.info(
                    "[CandidateGenerator] the parse failure is TRUNCATION-"
                    "shaped (%s) — retrying with a reduced output shape rather "
                    "than another whole-file attempt [%s]",
                    (failures[0] or {}).get("message", "?")[:60], op_id_short,
                )
                if callable(_reshape):
                    context = _reshape()
                else:
                    # No reshape helper on this context: dataclasses.replace is
                    # the same mechanism the field's own owner uses.
                    try:
                        import dataclasses as _dc  # noqa: PLC0415
                        context = _dc.replace(context, force_diff_on_retry=True)
                    except Exception:  # noqa: BLE001 — reshape is best-effort
                        pass
                _stamp = getattr(context, "with_syntax_retry_feedback", None)
                if not callable(_stamp):
                    raise

            retry_ctx = _stamp(feedback)
            if retry_ctx is context:
                # Stamping failed (degraded to self); retrying unchanged would
                # just repeat the failure at full cost.
                raise
            result = await self._failover_local_dispatch(
                retry_ctx, deadline, endpoint,
            )
            logger.info(
                "[CandidateGenerator] syntax-repair retry produced %d "
                "candidate(s) [%s]",
                len(getattr(result, "candidates", ()) or ()), op_id_short,
            )
            return result

    async def _try_free_lane_dispatch(
        self,
        context: OperationContext,
        deadline: datetime,
        *,
        route: str,
        reason: str,
    ) -> Optional[GenerationResult]:
        """Serve a dead-queued cost-optimized op on a zero-marginal-cost lane.

        BACKGROUND and SPECULATIVE exist to say "spend nothing you don't have
        to". Both encode that as a PROVIDER NAME — DoubleWord — and so, when
        the DW catalog is purged or every DW model is exhausted, the only move
        left is to dead-queue the op with ``background_dw_blocked_by_topology``.
        The dormant-queue comment at the block site names its own release
        condition exactly: *"the queue stays dormant until a viable,
        cost-effective inference endpoint is secured."* On a host serving the
        32B locally that endpoint is already secured, permanently, at $0.00 per
        operation — the gate simply cannot see it, because it is asking after a
        provider's NAME rather than after its COST. Six of six operations in
        soak ``bt-2026-08-24-074121`` died at that gate on a box whose GPU was
        idle the whole time; three of them with this exact code.

        This is the same correction :func:`_free_lane_active` made for the
        swarm-chunking skip, applied to the routing decision itself.

        Deliberately NOT conditioned on :func:`_free_lane_active`. That
        predicate answers "is the free lane the ONLY lane?", which is the right
        question for a cost *interlock* and the wrong one here: the local lane's
        marginal cost is zero whether or not a DW key also exists, so an
        operator who has both should still get the free lane for a route whose
        entire contract is frugality — rather than a dead queue — when DW is
        down. What IS required is EVIDENCE: a reachable endpoint that answered,
        never a flag asserting one should exist.

        Returns the ``GenerationResult`` on success, or ``None`` to fall
        through to the caller's existing raise. Every failure mode — flag off,
        wrong route, no endpoint, empty candidates, dispatch error — returns
        ``None``, so a topology with no local lane keeps its byte-identical
        dead-queue behaviour and the cloud deployment is untouched.
        """
        if not _background_local_lane_enabled():
            return None
        if route not in ("background", "speculative"):
            return None

        op_id_short = (getattr(context, "op_id", "") or "?")[:16]

        try:
            endpoint = await self._discover_jprime_endpoint()
        except Exception:  # noqa: BLE001 — discovery is advisory, never fatal
            logger.debug(
                "[CandidateGenerator] free-lane discovery failed [%s]",
                op_id_short, exc_info=True,
            )
            return None
        if not endpoint:
            # The honest, and common, case: no local lane on this host. Logged
            # at DEBUG because on a cloud node it is every cost-optimized op.
            logger.debug(
                "[CandidateGenerator] free-lane declined: no local J-Prime "
                "endpoint discoverable (route=%s, block=%s) [%s]",
                route, reason[:60], op_id_short,
            )
            return None

        logger.info(
            "[CandidateGenerator] free-lane preemption: route=%s would "
            "dead-queue (%s) but a zero-marginal-cost lane is SERVING at %s — "
            "dispatching GENERATE locally instead of queueing [%s]",
            route, reason[:80], endpoint, op_id_short,
        )
        try:
            result = await self._local_dispatch_with_syntax_repair(
                context, deadline, endpoint,
            )
        except asyncio.CancelledError:
            # Structured concurrency: a parent cancellation is NOT a lane
            # failure and must not be swallowed into a dead-queue raise.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CandidateGenerator] free-lane dispatch failed (%s: %.120s) — "
                "falling through to the queue path [%s]",
                type(exc).__name__, str(exc), op_id_short,
            )
            return None

        if result is not None:
            # A NO-OP is an ANSWER, not an absence. `2b.1-noop` means the model
            # read the request and concluded the work is already done -- soak
            # bt-2026-08-28-061124 returned exactly that ("the dependency
            # 'torch' is already present in requirements.txt") in 2.8s for 28
            # tokens. Counting candidates alone cannot tell that apart from a
            # lane that produced nothing, and falling through would report
            # `background_dw_blocked_by_topology` for an op that was never
            # blocked -- the operator would read a topology outage where the
            # truth is "correctly declined". The orchestrator already knows how
            # to terminate an `is_noop` result benignly; hand it over.
            if getattr(result, "is_noop", False):
                logger.info(
                    "[CandidateGenerator] free-lane returned a NO-OP verdict — "
                    "the op is answered, not blocked (route=%s) [%s]",
                    route, op_id_short,
                )
                return result
            if len(getattr(result, "candidates", ()) or ()) > 0:
                return result

        logger.info(
            "[CandidateGenerator] free-lane produced no candidates — falling "
            "through to the queue path (route=%s) [%s]", route, op_id_short,
        )
        return None

    async def _dispatch_via_sentinel(
        self,
        context: OperationContext,
        deadline: datetime,
        provider_route: str,
        *,
        _immortal_attempt: int = 0,
        _immortal_budget_deadline: Optional[float] = None,
    ) -> Optional[GenerationResult]:
        """Phase 10 P10.3 — sentinel-driven DW dispatch.

        Walks the route's ranked ``dw_models`` list (yaml v2). For each
        model whose breaker is not OPEN, stamps ``ctx._dw_model_override``
        and attempts DW. On per-model failure reports to the sentinel
        (with appropriate ``FailureSource`` weight) and continues to
        the next model. After exhausting all DW models, applies the
        route's ``fallback_tolerance``:

          * ``"cascade_to_claude"`` — invokes ``_call_fallback`` (Claude).
          * ``"queue"`` — raises the sentinel-already-known
            ``RuntimeError("dw_severed_queued:...")`` shape that the
            orchestrator's existing accept-failure branch handles.

        Returns:
          * ``GenerationResult`` on DW success or Claude cascade.
          * ``None`` to signal "fall through to legacy path" — used
            when the route has empty ``dw_models`` (e.g. IMMEDIATE,
            which is Claude-direct by Manifesto §5 design and is
            handled by the existing ``_generate_immediate`` dispatcher
            below).
        """
        # ── Phase 3c — Sovereign Failover DAG re-entry (THE seam) ──────────
        # When the Failover FSM is SERVING (J-Prime warm at its :11434
        # endpoint), generation re-enters through the LocalPrimeClient
        # (Tier-2 self-hosted), BYPASSING the DoubleWord sentinel entirely.
        # This is the single chokepoint every DW dispatch (Slice-23 sentinel
        # activation AND the immortal re-queue recursion) funnels through, so
        # one branch here re-routes all of them.
        #
        # Byte-identical when OFF: ``lifecycle_enabled()`` is false by default
        # -> ``is_jprime_serving()`` is always false -> this branch is never
        # taken -> the legacy DW path below runs unchanged.
        #
        # Fail-soft ABSOLUTE: if the local route errors (endpoint missing,
        # LocalPrimeClient raises, empty result), we log + fall through to the
        # normal DW path -- the op is NEVER lost.
        # Absolute Route Sealing: when the router has committed to the sovereign
        # J-Prime provider (Cryo-DLQ pin or the hybrid-mesh flag), a discovered-
        # and-dispatched 32B FAILURE must be TERMINAL -- never cascade to the dead
        # DW/adversary-stub lane. Only a COMMITTED dispatch (endpoint discovered)
        # can seal; a pre-dispatch miss (no endpoint) is not "a 32B failure during
        # generation" and falls through to the legacy path.
        _sealing = _absolute_route_sealing(context)
        _committed = False
        _seal_reason: Optional[str] = None
        try:
            # DRY universal router: EVERY DW dispatch (primary, GENERATE_RETRY,
            # critique, immortal re-queue) funnels through this chokepoint, so one
            # branch re-routes them all to the awakened 32B. Use the ROBUST
            # discovery (_discover_jprime_endpoint: controller-published endpoint
            # when SERVING, else a direct zone-aware GCP query) instead of only
            # is_jprime_serving() -- so a RETRY routes to the 32B even when this
            # process's controller FSM isn't itself in SERVING state.
            _ep = await self._discover_jprime_endpoint()
            if _ep:
                _committed = True
                _local_result = await self._failover_local_dispatch(
                    context, deadline, _ep,
                )
                if _local_result is not None and len(
                    getattr(_local_result, "candidates", ()) or ()
                ) > 0:
                    logger.info(
                        "[CandidateGenerator] Phase 3c DAG re-entry: routed "
                        "generation to the awakened 32B endpoint=%s (DW bypassed) "
                        "op=%s route=%s",
                        _ep, getattr(context, "op_id", "?")[:16], provider_route,
                    )
                    return _local_result
                _seal_reason = "empty_result"
                logger.info(
                    "[CandidateGenerator] Phase 3c DAG re-entry: J-Prime local "
                    "route empty/failed (op=%s route=%s)",
                    getattr(context, "op_id", "?")[:16], provider_route,
                )
        except Exception as _f3c_exc:  # noqa: BLE001 -- seam must never break the op
            if _committed:
                _seal_reason = type(_f3c_exc).__name__
            logger.warning(
                "[CandidateGenerator] Phase 3c failover seam err=%r%s", _f3c_exc,
                "" if _committed else " (pre-dispatch, not committed)",
            )

        # SEAL: committed sovereign dispatch failed AND sealing is armed -> HALT the
        # cognitive loop with a terminal reason (classified non-retryable by the
        # orchestrator). The DW fall-through below is NEVER reached -- no cascade to
        # the dead/adversary lane.
        if _sealing and _committed and _seal_reason:
            logger.warning(
                "[CandidateGenerator] ABSOLUTE ROUTE SEAL op=%s reason=%s -- "
                "HALTING cognitive loop (sovereign J-Prime committed + failed; DW "
                "cascade FORBIDDEN)",
                getattr(context, "op_id", "?")[:16], _seal_reason,
            )
            raise RuntimeError(
                "sovereign_route_sealed:gcp-jprime:%s" % _seal_reason
            )

        from backend.core.ouroboros.governance.provider_topology import (
            get_topology as _get_topology,
        )
        from backend.core.ouroboros.governance.topology_sentinel import (
            FailureSource,
            get_default_sentinel,
            reset_dw_model_override as _reset_override,
            set_dw_model_override as _set_override,
        )

        # Phase 12.2 Slice F — discovery is now armed eagerly by the
        # Autonomic Pacemaker in GovernedLoopService at orchestrator
        # boot, before any sensor signal is pulled. The dynamic catalog
        # is populated + the 30-min refresh task is heartbeating before
        # the dispatcher runs, so this code path never needs to bootstrap
        # discovery itself. Operator directive 2026-04-28 mandates a
        # single source of truth — the Pacemaker. If the Pacemaker fails
        # to arm, operators see the warning at boot rather than a silent
        # failure on first dispatch.

        topology = _get_topology()
        if not topology.enabled:
            return None
        ranked_models = topology.dw_models_for_route(provider_route)
        fallback_tolerance = topology.fallback_tolerance_for_route(
            provider_route,
        )

        # Slice 229 — exploration-floor driven route elevation. When this op
        # must satisfy the Iron Gate exploration floor (the SAME Slice-226
        # predicate that opens the tool loop + steers the hedge), prepend the
        # COMPLEX route's agentic-elite pool (active-param-ranked, family-
        # weighted) so tool-loop work is never starved onto low-active models
        # that cannot drive it. The live layer-5 wedge: Kimi/DeepSeek-V4-Pro/
        # GLM-5.1 all promoted=True yet UNREACHABLE from STANDARD — file-00's
        # 'simple' label kept it in a pool whose only capable member drifts.
        try:
            from backend.core.ouroboros.governance.exploration_engine import (
                exploration_gate_demands_tools as _s229_gate_demands,
            )
            from backend.core.ouroboros.governance.provider_topology import (
                elevate_pool_for_exploration as _s229_elevate,
            )
            _s229_demands = (
                provider_route not in ("background", "speculative")
                and _s229_gate_demands(
                    str(getattr(context, "task_complexity", "")),
                )
            )
            if _s229_demands and provider_route != "complex":
                _s229_elite = topology.dw_models_for_route("complex")
                _s229_pool = _s229_elevate(
                    tuple(ranked_models), tuple(_s229_elite),
                    demands_tools=True,
                )
                if tuple(_s229_pool) != tuple(ranked_models):
                    logger.warning(
                        "[CandidateGenerator] ⚡ ROUTE ELEVATION: op needs "
                        "Iron-Gate exploration — agentic-elite (COMPLEX) pool "
                        "prepended for route=%s: %s (op=%s)",
                        provider_route, list(_s229_pool)[:4],
                        getattr(context, "op_id", "?")[:16],
                    )
                    ranked_models = list(_s229_pool)
        except Exception:  # noqa: BLE001 — elevation is enhancement, never blocks
            pass

        # Slice 201 — Contextual Bandit Routing Advisor. ADVISORY-ONLY +
        # structurally fail-closed: the advisor reorders WITHIN ranked_models
        # (the brain_selection_policy active set for this route), so it can
        # only change the ORDER the sentinel tries policy-permitted models —
        # never select an out-of-policy arm. Gated (default OFF → no-op); any
        # error keeps the deterministic order. The hand-rolled router stays
        # authoritative.
        try:
            from backend.core.ouroboros.governance.bandit_router import (
                get_bandit_router as _s201_bandit,
            )
            _s201_order = _s201_bandit().advise(ranked_models)
            if _s201_order and set(_s201_order) == set(ranked_models):
                ranked_models = _s201_order
        except Exception:  # noqa: BLE001 — advisory, never blocks dispatch
            pass

        # Empty dw_models → fall through to legacy dispatch. IMMEDIATE
        # has empty models by design (Claude-direct); other routes
        # would fall here only if yaml is misconfigured.
        if not ranked_models:
            logger.debug(
                "[CandidateGenerator] Sentinel dispatch: route=%s "
                "has no dw_models — falling through to legacy",
                provider_route,
            )
            return None

        # Slice 76 Phase 2 — pre-flight DW transport gate. If the existing
        # dw_surface_health ledger shows the DIRECT_STREAMING surface FRESHLY
        # TRANSPORT_DEGRADED, the whole ranked list shares that dead transport
        # (cf. should_sever_dw_lane). Cascade to Claude with the FULL untouched
        # budget NOW — before the _primary_sem wait + per-model timeout cascade
        # burns it (the EVAL-2 terminal_timeout, PRD §50.11). Only when the
        # route already cascades to Claude (a "queue"-tolerance route keeps its
        # contract). Gated + fail-open; default-on.
        if (
            fallback_tolerance == "cascade_to_claude"
            and dw_transport_degraded_preflight()
            # Pre-emptive Route Masking (2026-07-18): the sever-cascade
            # is ALSO a Claude purchase — same contract consult as the
            # exhaustion cascade; a masked route falls through to the
            # DW path and exhausts cheaply instead.
            and not claude_route_masked(context)
        ):
            logger.info(
                "[CandidateGenerator] Slice 76 pre-flight: DW DIRECT_STREAMING "
                "TRANSPORT_DEGRADED (fresh) — severing DW lane pre-budget, "
                "cascading to Claude with full budget (op=%s route=%s)",
                getattr(context, "op_id", "?"), provider_route,
            )
            return await self._call_fallback(context, deadline)

        sentinel = get_default_sentinel()
        # Register every model in the ranked list (idempotent). The
        # sentinel needs to know about each model_id before it can
        # answer get_state.
        for model_id in ranked_models:
            sentinel.register_endpoint(model_id)

        op_id_short = (
            getattr(context, "op_id", "?")[:16]
            if hasattr(context, "op_id") else "?"
        )

        # Walk the ranked list. For each model not OPEN, attempt DW.
        attempts: List[str] = []
        last_failure: Optional[str] = None
        # Slice 4 T2 — LOCAL session-budget refusal tracking. A $0.00 session
        # budget makes EVERY ranked model refuse identically via
        # SessionBudgetPreflightRefused. That is a local config gate, NOT a
        # remote provider outage: it must NEVER poison vendor telemetry
        # (surface-health, the health gradient, dual-arm blacklist) or wake the
        # real-$ J-Prime GCE failover. Track it so a PURE budget exhaustion
        # (no genuine transport failure observed) fails FAST + visibly instead
        # of quarantining a phantom outage or immortal-re-queueing forever.
        _budget_refusal_exc: Optional[BaseException] = None
        _saw_non_refusal_failure: bool = False
        # Slice 83 Phase 2 — consecutive LIVE_TRANSPORT streak across the
        # heterogeneous coder stack. A single model's transport break rotates
        # to the next coder; only a `threshold`-long streak (genuine lane-wide
        # blackout) severs. Reset by any success / non-transport failure.
        _consecutive_lt: int = 0
        _lt_sever_threshold: int = _live_transport_sever_threshold()
        # Slice 182 — SENTINEL BATCH ENFORCEMENT (Gap 1). The per-model frozen context carries
        # an EMPTY provider_route, so the downstream _slice36_should_force_batch route gate
        # can't engage and every probe ruptured on RT (the v181 bleed). The sentinel KNOWS the
        # route + the risk — so if the stream is degraded / rupture-risk is high AND batch is
        # healthy, COMMAND every probe to batch at T=0 via the force-batch ContextVar.
        _s182_force_batch = False
        try:
            from backend.core.ouroboros.governance.doubleword_provider import (
                _dw_streaming_warm_degraded as _s182_warm,
                _dw_rupture_risk_high as _s182_risk,
                _dw_batch_lane_healthy as _s182_batch_ok,
                _dw_in_cold_start as _s184_cold,
                _dw_hedge_supersedes as _s192_supersedes,
            )
            # Slice 183 — LIVE TELEMETRY PROBE. Capture the EXACT boolean state of every
            # sub-gate AND the final computed decision, UNCONDITIONALLY (before the if), so the
            # live soak shows precisely why force-batch is False. Each gate is evaluated into
            # its own local — no short-circuit hiding which one fails.
            _g_route_ok = provider_route in ("standard", "complex")
            _g_batch = bool(_s182_batch_ok())
            _g_warm = bool(_s182_warm())
            _g_risk = bool(_s182_risk(""))
            # Slice 184 — cold-start is a degradation TRIGGER: at fresh boot the stream is
            # unproven, so the sentinel commands batch (fail-safe) even when warm/risk are blind.
            _g_cold = bool(_s184_cold())
            # Slice 192 — PROACTIVE HIERARCHY: the sentinel DEFERS to the hedge. When the hedge
            # supersedes (active + no storm), do NOT force batch here — let the op RACE. The
            # cold-start/warm-boot enforce only fires when the hedge is off or a storm is confirmed.
            _g_hedge = bool(_s192_supersedes(context, model_id))
            _s182_force_batch = (
                (not _g_hedge)
                and _g_route_ok and _g_batch and (_g_warm or _g_risk or _g_cold)
            )
            logger.warning(
                "[Slice183] dispatch-telemetry: op=%s route=%r route_ok=%s "
                "batch_lane_healthy=%s warm_degraded=%s rupture_risk=%s cold_start=%s → FORCE_BATCH=%s",
                op_id_short, provider_route, _g_route_ok, _g_batch, _g_warm, _g_risk, _g_cold,
                _s182_force_batch,
            )
            if _s182_force_batch:
                logger.warning(
                    "[Cortex] SENTINEL batch-enforce: stream degraded / rupture-risk high → "
                    "ALL probes via BATCH at T=0 (route=%s, op=%s) — RT bypass eradicated",
                    provider_route, op_id_short,
                )
        except Exception:  # noqa: BLE001 — enforcement is best-effort, never blocks dispatch
            # Slice 183 — DO NOT silently swallow. Log the full traceback so a hidden
            # ImportError / attribute error in the gate path is visible in the live soak.
            import traceback as _s183_tb
            logger.warning(
                "[Slice183] dispatch-telemetry EXCEPTION (NOT swallowed silently): %s",
                _s183_tb.format_exc(),
            )
            _s182_force_batch = False
        # A3 Transport Circuit Breaker -- lane-rotation seam.
        # When _s182_force_batch is True, the preferred transport is "batch".
        # If the breaker is enabled and that lane is OPEN, _breaker_select_transport
        # returns "realtime" -- we honour it by clearing the force-batch flag so
        # doubleword_provider stays on the SSE/realtime path.  When the breaker is
        # disabled (default) or the lane is CLOSED, this is a zero-cost no-op.
        if _s182_force_batch:
            _a3_chosen_lane = _breaker_select_transport("batch")
            if _a3_chosen_lane == "realtime":
                logger.info(
                    "[A3-Breaker] batch lane OPEN -- rotating to realtime for op=%s",
                    op_id_short,
                )
                _s182_force_batch = False
        for model_id in ranked_models:
            state = sentinel.get_state(model_id)
            # Phase 12 Slice H — TERMINAL_OPEN bypasses dispatch
            # entirely (deterministic ground-truth ban from a 4xx
            # modality or 401/403 auth failure; doesn't auto-recover
            # via probes, only via explicit reset / catalog refresh).
            # Treated indistinguishably from OPEN at the dispatch
            # gate — both are "do not attempt"; the difference is
            # purely in the recovery model (probe vs explicit reset).
            if state in ("OPEN", "TERMINAL_OPEN"):
                logger.info(
                    "[CandidateGenerator] Sentinel dispatch: route=%s "
                    "model=%s state=%s — skipping (op=%s)",
                    provider_route, model_id, state, op_id_short,
                )
                attempts.append(f"{model_id}:skipped_{state.lower()}")
                continue
            # Latency quarantine (2026-06-20): the entitlement breaker (above)
            # bans 403'd models; this bans models the TtftObserver has flagged as
            # COLD STORAGE (latest TTFT > mean + Nσ — weights evicted from VRAM →
            # the 180s-timeout black hole). Reuses the existing observer; only
            # skips when there's at least one OTHER candidate left to try (never
            # quarantines the sole remaining model into a no-op). Gated on the
            # observer's own master flag; fail-open (never blocks dispatch).
            if _latency_quarantine_enabled() and model_id != ranked_models[-1]:
                try:
                    from backend.core.ouroboros.governance.dw_ttft_observer import (
                        get_ttft_observer as _get_ttft_obs,
                    )
                    _obs = _get_ttft_obs()
                    if _obs is not None and _obs.is_cold_storage(model_id):
                        logger.info(
                            "[CandidateGenerator] Latency quarantine: route=%s "
                            "model=%s COLD_STORAGE (TTFT spike) — skipping to a "
                            "warmer candidate (op=%s)",
                            provider_route, model_id, op_id_short,
                        )
                        attempts.append(f"{model_id}:skipped_cold_storage")
                        continue
                except Exception:  # noqa: BLE001 — observer must never block dispatch
                    pass
            # Slice 20C — schema drift rotation. If this model has
            # produced a structurally-bad output earlier in this same
            # op (json_parse_error_after_heal / schema_id_hallucination
            # / zero_candidate_return), skip it indistinguishably from
            # a sentinel-OPEN breaker. Master-flag gated; when off, the
            # has_drifted() consultation short-circuits to False so the
            # check is a free no-op (byte-identical legacy behavior).
            try:
                from backend.core.ouroboros.governance.schema_drift_tracker import (
                    get_default_tracker as _get_drift_tracker,
                )
                _drift_tracker = _get_drift_tracker()
                _full_op_id_drift = getattr(context, "op_id", "") or ""
                if _drift_tracker.has_drifted(_full_op_id_drift, model_id):
                    logger.info(
                        "[CandidateGenerator] Sentinel dispatch: route=%s "
                        "model=%s drifted_on_op — rotating to sibling (op=%s)",
                        provider_route, model_id, op_id_short,
                    )
                    attempts.append(f"{model_id}:skipped_drift")
                    continue
            except Exception:  # noqa: BLE001 — rotation is enhancement, not gate
                # Tracker consultation must NEVER block dispatch. If
                # the tracker module is missing / unimportable / raises,
                # fall through to normal attempt (legacy behavior).
                pass
            # Slice 194 — race-triage rotation. If BOTH hedge arms died on
            # this model earlier in this same op (confirmed hard blockage),
            # skip the corpse — the next iteration IS the next-highest-ranked
            # catalog candidate. OWN master (JARVIS_RACE_TRIAGE_ENABLED,
            # default TRUE, failure-path-only) — deliberately independent of
            # the default-FALSE drift-rotation master above.
            try:
                from backend.core.ouroboros.governance.race_triage import (
                    is_blacklisted_for_op as _s194_is_blacklisted,
                )
                _s194_op_id = getattr(context, "op_id", "") or ""
                if _s194_is_blacklisted(_s194_op_id, model_id):
                    logger.warning(
                        "[RaceTriage] Sentinel dispatch: route=%s model=%s "
                        "dual-arm-blacklisted on op — rotating to next ranked "
                        "candidate (op=%s)",
                        provider_route, model_id, op_id_short,
                    )
                    attempts.append(f"{model_id}:skipped_dual_arm")
                    continue
            except Exception:  # noqa: BLE001 — rotation is enhancement, not gate
                pass
            attempts.append(f"{model_id}:attempted")
            # Stamp the per-attempt override via ContextVar (async-safe
            # per asyncio task, survives the frozen OperationContext
            # contract). The ContextVar is reset in the finally block
            # so the next iteration's set is a clean state, and so
            # cascade-to-Claude after exhaustion doesn't carry a stale
            # override into the fallback provider.
            _override_token = _set_override(model_id)
            # Slice 182 — alongside the model override, COMMAND batch for this probe when the
            # sentinel determined degradation (Gap 1). Reset in the same finally as the model
            # override, so neither leaks into the post-exhaustion cascade.
            _s182_fb_token = None
            if _s182_force_batch:
                try:
                    from backend.core.ouroboros.governance.doubleword_provider import (
                        set_sentinel_force_batch as _s182_set_fb,
                    )
                    _s182_fb_token = _s182_set_fb(True)
                except Exception:  # noqa: BLE001
                    _s182_fb_token = None
            logger.info(
                "[CandidateGenerator] Sentinel dispatch: route=%s "
                "attempting model=%s (state=%s, op=%s)",
                provider_route, model_id, state, op_id_short,
            )
            _attempt_result: Any = None
            _attempt_exc: Optional[BaseException] = None
            try:
                if provider_route == "background":
                    _attempt_result = await self._generate_background(
                        context, deadline,
                    )
                elif provider_route == "speculative":
                    _attempt_result = await self._generate_speculative(
                        context, deadline,
                    )
                else:
                    # Slice 23 — standard / complex / unknown route uses
                    # the primary-first cascade. The Slice 23 sentinel
                    # walker still stamps the ContextVar for the
                    # provider's INTERNAL routing (DoublewordProvider.
                    # _resolve_effective_model reads it to pick which
                    # model to actually call).
                    #
                    # Slice 30 — ALSO threads model_id explicitly through
                    # the orchestrator-side call chain so
                    # _compute_primary_budget's heavy-model 2.5× scalar
                    # (Slice 28 Phase 2) engages deterministically. The
                    # v23 wiring gap (ContextVar invisible across
                    # async/semaphore boundaries) is eliminated for the
                    # TIMEOUT decision; provider routing still uses the
                    # ContextVar (legitimate per-provider internal use).
                    _attempt_result = await self._try_primary_then_fallback(
                        context, deadline, model_id=model_id,
                    )
            except GovernanceDeadlockError:
                # LR3 terminal: the Information-Gain Governor's deadlock-override
                # failure (raised inside the Venom tool loop) MUST reach the
                # orchestrator's ``except GovernanceDeadlockError`` terminal catch
                # so it stamps terminal_reason_code="deadlock_override_failed".
                # If stored in _attempt_exc it gets reclassified by the per-model
                # rotation taxonomy (not internal_fault, not generation_timeout,
                # not fsm_exhausted) and re-driven as a transport failure /
                # all_providers_exhausted; the deadlock would never surface.
                raise
            except Exception as exc:
                _attempt_exc = exc
            finally:
                # Reset ContextVar before either success-return or
                # failure-continue so the next iteration starts with a
                # clean slate AND the post-loop cascade-to-Claude
                # doesn't carry a stale override into the fallback.
                _reset_override(_override_token)
                # Slice 182 — clear the force-batch command too (never leak into cascade).
                if _s182_fb_token is not None:
                    try:
                        from backend.core.ouroboros.governance.doubleword_provider import (
                            reset_sentinel_force_batch as _s182_reset_fb,
                        )
                        _s182_reset_fb(_s182_fb_token)
                    except Exception:  # noqa: BLE001
                        pass

            if _attempt_result is not None:
                # Success — let the sentinel know. Phase 10 P10.4
                # also wires report_failure at existing failure sites
                # so a stream-stall mid-generation also lands in the
                # sentinel; this report_success closes the
                # corresponding successful-stream signal.
                try:
                    sentinel.report_success(model_id)
                except Exception:
                    logger.debug(
                        "[CandidateGenerator] sentinel.report_success raised",
                        exc_info=True,
                    )
                try:
                    # Slice 201 — feed the bandit a SUCCESS reward for this arm.
                    from backend.core.ouroboros.governance.bandit_router import (
                        get_bandit_router as _s201_bandit_ok,
                    )
                    _s201_bandit_ok().record_outcome(model_id, success=True)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    # Override Matrix — clear the model-pin soft-lock streak on a
                    # real success (passive observed outcome; no active probe).
                    from backend.core.ouroboros.governance.model_pinning_heuristic import (
                        note_pin_outcome as _pin_ok,
                    )
                    _pin_ok(model_id, success=True)
                except Exception:  # noqa: BLE001
                    pass
                # Slice 20C — zero-candidate drift detection. The
                # parser succeeded (we're on the success branch) but
                # may have returned an empty candidates tuple while
                # NOT signaling no-op. That's the v15 "model judgment
                # flaw" — Venom exploration ran, model returned valid
                # JSON, but candidates=(). Record drift so the next
                # GENERATE_RETRY for this op_id rotates to a sibling.
                try:
                    _cands = getattr(_attempt_result, "candidates", None)
                    _is_noop = getattr(_attempt_result, "is_noop", False)
                    if (
                        _cands is not None
                        and len(_cands) == 0
                        and not _is_noop
                    ):
                        from backend.core.ouroboros.governance.schema_drift_tracker import (
                            DriftType,
                            get_default_tracker as _zc_tracker,
                        )
                        _full_op_zc = getattr(context, "op_id", "") or ""
                        if _full_op_zc:
                            _zc_tracker().record(
                                op_id=_full_op_zc,
                                model_id=model_id,
                                drift_type=DriftType.ZERO_CANDIDATE_RETURN,
                                raw_excerpt=(
                                    f"route={provider_route} "
                                    f"is_noop=False candidates=()"
                                ),
                            )
                            logger.info(
                                "[CandidateGenerator] Slice 20C zero-candidate "
                                "drift recorded: op=%s model=%s — next retry "
                                "will rotate to sibling",
                                op_id_short, model_id,
                            )
                except Exception:  # noqa: BLE001 — drift is enhancement
                    pass
                # A3 Transport Circuit Breaker -- record success outcome.
                # Lane: "batch" if force-batch was armed (_s182_fb_token set), else "realtime".
                _a3_success_lane = "batch" if _s182_fb_token is not None else "realtime"
                _breaker_record_outcome(_a3_success_lane, ok=True, failure_mode=None)
                # Task T2 -- provider health gradient RECOVERY. A real DW
                # candidate returned, so the per-route success window records a
                # success: this autonomously CLEARS a previously-deduced global
                # outage (is_global_outage flips False once one True lands in a
                # full all-False window). Fail-soft: gradient errors never
                # perturb the success return.
                try:
                    get_provider_health_gradient().record_sweep(
                        provider_route, success=True,
                    )
                except Exception:  # noqa: BLE001 -- gradient is advisory, never blocks
                    pass
                # Phase 1 Outage Ledger -- record DW recovery for the forecaster
                # + async Trinity export. Only runs if an outage was open (fail-soft).
                try:
                    from backend.core.ouroboros.governance.outage_ledger import (  # noqa: PLC0415
                        get_outage_ledger as _ol_get,
                        emit_outage_event as _ol_emit,
                    )
                    _ol_ledger = _ol_get()
                    if _ol_ledger.has_open_outage():
                        _ol_recent = _ol_ledger.recent(1)
                        if _ol_recent:
                            _ol_open = _ol_recent[-1]
                            if _ol_open.ended_ts is None:
                                _ol_ledger.close_outage(_ol_open.outage_id)
                                _ol_closed = _ol_ledger.recent(1)
                                if _ol_closed:
                                    _ol_emit("DW_OUTAGE_RECOVERED", _ol_closed[-1])
                except Exception:  # noqa: BLE001 -- outage ledger never blocks the op
                    pass
                return _attempt_result

            if _attempt_exc is not None:
                exc = _attempt_exc
                # Slice 4 T2 — classify a LOCAL session-budget refusal BEFORE any
                # vendor-fault taxonomy. It is not a transport rupture: do NOT
                # report it to the sentinel, the DW surface-health ledger, the
                # transport breaker, the bandit, or the health gradient. Record
                # it and rotate — every ranked model refuses identically at
                # $0.00, so the exhaustion path (below) fails fast instead of
                # quarantining a phantom provider outage or waking the GCE
                # failover (Run #14 failure-taxonomy fix).
                from backend.core.ouroboros.governance.session_budget_authority import (
                    is_budget_refusal as _s4_is_budget_refusal,
                )
                if _s4_is_budget_refusal(exc):
                    _budget_refusal_exc = exc
                    last_failure = (
                        f"{model_id}:budget_refusal:{type(exc).__name__}"
                    )
                    if attempts:
                        attempts[-1] = f"{model_id}:budget_refusal"
                    logger.warning(
                        "[CandidateGenerator] Sentinel dispatch: model=%s "
                        "REFUSED by LOCAL session-budget gate (not a provider "
                        "fault) — rotating without vendor-outage telemetry "
                        "(op=%s)", model_id, op_id_short,
                    )
                    continue
                _saw_non_refusal_failure = True
                # Slice 185 Phase 2 — STRICT-TYPE EXCEPTION SEGREGATION. A Python LOGICAL error
                # (NameError/TypeError/AttributeError/…) is OUR codebase bug, NOT a vendor
                # network rupture. It must bypass the vendor resilience path entirely: never be
                # classified as live_transport, never recorded to the DW surface-health ledger
                # (which corrupts the learned rupture rate), never silently degraded. Bubble it
                # up as an INTERNAL_FAULT and crash LOUDLY so we fix OUR bug — the AI must never
                # again blame the vendor for its own internal codebase flaws.
                from backend.core.ouroboros.governance.dw_fault_taxonomy import (
                    is_internal_fault as _s185_internal,
                    is_generation_timeout as _s241_gen_timeout,
                    is_fsm_exhaustion as _fsm_exhausted,
                    is_local_egress_overweight as _egress_overweight,
                )
                # Sovereign Egress Interceptor Mesh (T3) — OUR-side egress
                # interceptor blocked an over-ceiling body. DW never received the
                # request (good API citizenship); no socket failed. Classify
                # LOCAL_EGRESS_OVERWEIGHT (weight 0.0) so it NEVER trips the model/
                # topology breaker or corrupts surface-health, THEN re-raise the
                # ORIGINAL exception so its structured ``max_allowed_size`` survives
                # to the orchestrator's generate-failure path, which routes it BACK
                # to context-aware chunking (decompose_for_block(compression_target)).
                # Re-raising immediately is correct: every sibling model would
                # reject the SAME oversized body, so continuing the rotation only
                # burns the loop and discards the compression math. Fail-soft:
                # classification + report_failure are best-effort and never block
                # the re-raise.
                if _egress_overweight(exc):
                    try:
                        sentinel.report_failure(
                            model_id,
                            FailureSource.LOCAL_EGRESS_OVERWEIGHT,
                            f"{type(exc).__name__}:{str(exc)[:120]}",
                        )
                    except Exception:  # noqa: BLE001 — telemetry, never block
                        logger.debug(
                            "[CandidateGenerator] egress-overweight report_failure raised",
                            exc_info=True,
                        )
                    logger.warning(
                        "[CandidateGenerator] LOCAL_EGRESS_OVERWEIGHT (weight 0.0, "
                        "NOT a vendor rupture): our egress interceptor blocked an "
                        "over-ceiling body for model=%s — re-raising with "
                        "max_allowed_size=%s so the orchestrator re-chunks to fit "
                        "(op=%s)",
                        model_id, getattr(exc, "max_allowed_size", "?"),
                        op_id_short,
                    )
                    raise exc
                if _s185_internal(exc):
                    logger.error(
                        "[CandidateGenerator] INTERNAL_FAULT (%s) — NOT a vendor rupture; "
                        "bubbling up + crashing loud, NOT touching the DW vendor ledger "
                        "(op=%s, model=%s): %s",
                        type(exc).__name__, op_id_short, model_id, exc,
                        exc_info=True,
                    )
                    raise exc
                err_str = str(exc)
                err_lower = err_str.lower()

                # Phase 12 Slice F — Substrate Error Unmasking. When
                # the exception is a DoublewordInfraError (or any
                # structurally-unmasked equivalent that carries a
                # ``status_code`` attribute), classify FROM THE
                # STRUCTURED FIELD instead of regex on str(exc). This
                # is the substrate of Slice H's terminal-vs-transient
                # distinction — we MUST know the actual HTTP status to
                # decide TERMINAL_OPEN vs OPEN.
                _status_code = getattr(exc, "status_code", None)
                _response_body = getattr(exc, "response_body", "") or ""
                _is_modality = bool(
                    getattr(exc, "is_modality_error", lambda: False)()
                )
                _is_auth_terminal = bool(
                    getattr(exc, "is_terminal_auth_error", lambda: False)()
                )

                # Zero-Shot latency quarantine (2026-06-20): an explicit
                # generation TimeoutError (the 180s wall) OR a tool-loop deadline
                # is unambiguous evidence THIS model is unusable now. Flag it
                # cold-storage IMMEDIATELY (bypassing the n>=3 σ window that would
                # let it taint 2 more soaks) so the selector skips it next op. The
                # ban self-decays after a TTL (autonomic forgiveness). Fail-soft.
                if isinstance(exc, asyncio.TimeoutError) or _s241_gen_timeout(exc):
                    try:
                        from backend.core.ouroboros.governance.dw_discovery_runner import (
                            get_ttft_observer as _zs_get_obs,
                        )
                        _zs_obs = _zs_get_obs()
                        if _zs_obs is not None and model_id:
                            _zs_obs.record_timeout(model_id, op_id=op_id_short)
                    except Exception:  # noqa: BLE001 — never block dispatch
                        pass
                if _s241_gen_timeout(exc):
                    # Slice 241 — OUR op-level tool-loop budget exhaustion
                    # (tool_loop_deadline / max_rounds / starved), NOT a DW
                    # transport rupture. Classify GENERATION_TIMEOUT so the
                    # ==LIVE_TRANSPORT degrade/sever consumers ignore it and the
                    # topology breaker (weight 0.0) never trips on OUR budget.
                    # Stops blaming DoubleWord's network for our generation budget.
                    failure_source = FailureSource.GENERATION_TIMEOUT
                elif _fsm_exhausted(exc):
                    # Sovereign Exception Taxonomy (2026-06-20) — OUR-side FSM
                    # dispatch exhaustion (DW produced no candidate AND no Claude
                    # fallback configured under pure-DW autarky). NOT a vendor
                    # rupture: no socket failed, the vendor rejected nothing. The
                    # cloud soak proved a single
                    # ``all_providers_exhausted:fallback_skipped:no_fallback_configured``
                    # was mislabeled LIVE_TRANSPORT on all 16 models, severing the
                    # whole DW lane + corrupting surface-health. Classify
                    # FSM_EXHAUSTED (weight 0.0) so it fails ONLY this op without
                    # severing the lane or touching the vendor ledger.
                    failure_source = FailureSource.FSM_EXHAUSTED
                elif _is_modality or _is_auth_terminal:
                    # Slice H — terminal failure class. Even though we
                    # report it as LIVE_HTTP_5XX semantics here for
                    # back-compat, the breaker (Slice H wiring) will
                    # read the structured exception fields when
                    # available and flip the model's state to
                    # TERMINAL_OPEN. For now, classify with a body-
                    # accurate failure source so observers can audit
                    # the unmasked status.
                    failure_source = FailureSource.LIVE_TRANSPORT
                elif _status_code is not None:
                    # Structured HTTP status drives classification.
                    # QUOTA FIRST (the council's 2026-07-21 finding): a 4xx
                    # whose body is economic is a wallet state — before this
                    # branch it fell to LIVE_TRANSPORT and read as latency.
                    from backend.core.ouroboros.governance.economic_router import (  # noqa: E501
                        classify_http_failure_source as _econ_classify,
                        )
                    if _econ_classify(_status_code, err_str) is not None:
                        failure_source = FailureSource.LIVE_HTTP_4XX_QUOTA
                        _record_quota_outage_safely("doubleword", err_str)
                    elif _status_code == 429:
                        failure_source = FailureSource.LIVE_HTTP_429
                    elif _status_code in (500, 502, 503, 504):
                        failure_source = FailureSource.LIVE_HTTP_5XX
                    elif _status_code == 0:
                        # Non-HTTP failure: stream stall / DNS / TLS
                        if (
                            "stream" in err_lower
                            and ("stall" in err_lower or "timeout" in err_lower)
                        ) or "streamtimeouterror" in err_lower:
                            failure_source = FailureSource.LIVE_STREAM_STALL
                        else:
                            failure_source = FailureSource.LIVE_TRANSPORT
                    else:
                        failure_source = FailureSource.LIVE_TRANSPORT
                else:
                    # No status_code attribute → fall back to regex on
                    # str(exc) (legacy path for non-DW exceptions).
                    from backend.core.ouroboros.governance.economic_router import (  # noqa: E501
                        classify_http_failure_source as _econ_classify,
                        )
                    if (
                        "stream" in err_lower
                        and ("stall" in err_lower or "timeout" in err_lower)
                    ) or "streamtimeouterror" in err_lower:
                        failure_source = FailureSource.LIVE_STREAM_STALL
                    elif _econ_classify(None, err_str) is not None:
                        # Economic body with no structured status (e.g. an
                        # SDK BadRequestError repr) — same wallet taxonomy.
                        failure_source = FailureSource.LIVE_HTTP_4XX_QUOTA
                        _record_quota_outage_safely("doubleword", err_str)
                    elif "429" in err_str:
                        failure_source = FailureSource.LIVE_HTTP_429
                    elif "5" in err_str[:5] and (
                        "500" in err_str or "502" in err_str
                        or "503" in err_str or "504" in err_str
                    ):
                        failure_source = FailureSource.LIVE_HTTP_5XX
                    elif "parse" in err_lower or "json" in err_lower:
                        failure_source = FailureSource.LIVE_PARSE_ERROR
                    else:
                        failure_source = FailureSource.LIVE_TRANSPORT
                try:
                    # Pass structured fields to the sentinel so Slice H
                    # can use them for terminal-vs-transient decisions.
                    # Backward-compatible: legacy report_failure
                    # signature is preserved; structured fields are
                    # added via best-effort kwargs that the sentinel
                    # silently drops if it doesn't yet support them.
                    try:
                        sentinel.report_failure(
                            model_id, failure_source,
                            f"{type(exc).__name__}:{err_str[:120]}",
                            status_code=_status_code,
                            response_body=_response_body,
                            is_terminal=(_is_modality or _is_auth_terminal),
                        )
                    except TypeError:
                        # Sentinel doesn't accept new kwargs yet (pre-
                        # Slice-H sentinel) — fall back to legacy call
                        sentinel.report_failure(
                            model_id, failure_source,
                            f"{type(exc).__name__}:{err_str[:120]}",
                        )
                except Exception:
                    logger.debug(
                        "[CandidateGenerator] sentinel.report_failure raised",
                        exc_info=True,
                    )
                last_failure = (
                    f"{model_id}:{failure_source.value}:"
                    f"{type(exc).__name__}"
                )
                # Slice F — log the unmasked status_code + body excerpt
                # alongside the legacy WARNING line so operators see
                # ground truth in debug.log immediately.
                if _status_code is not None and _status_code > 0:
                    logger.warning(
                        "[CandidateGenerator] Sentinel dispatch: model=%s "
                        "FAILED (source=%s, http_%d, body=%r%s%s) — "
                        "trying next (op=%s)",
                        model_id, failure_source.value, _status_code,
                        _response_body[:160],
                        ", modality_terminal=true" if _is_modality else "",
                        ", auth_terminal=true" if _is_auth_terminal else "",
                        op_id_short,
                    )
                else:
                    logger.warning(
                        "[CandidateGenerator] Sentinel dispatch: model=%s "
                        "FAILED (source=%s, exc=%s) — trying next (op=%s)",
                        model_id, failure_source.value,
                        # Observability (2026-06-20): un-swallow the message. The
                        # prior log emitted only ``type(exc).__name__`` — which hid
                        # that a "live_transport RuntimeError" was actually an
                        # internal ``...:no_fallback_configured`` FSM exhaustion,
                        # costing two long blind diagnosis passes. Include the
                        # (bounded) message so the real cause is visible at WARNING.
                        f"{type(exc).__name__}: {err_str[:300]!r}", op_id_short,
                    )
                attempts[-1] = f"{model_id}:failed:{failure_source.value}"
                # A3 Transport Circuit Breaker -- record failure outcome (I2 filtered).
                # Only LIVE transport signals are meaningful to the per-lane breaker;
                # GENERATION_TIMEOUT / FSM_EXHAUSTED / HTTP classification (OUR-side faults)
                # are silently dropped by _breaker_record_outcome's _BREAKER_RECORD_SOURCES gate.
                # T4 (Dynamic Lane Escalation): ``exc`` is forwarded so the record helper
                # can re-arm the breaker's vision for the SINGLE batch-lane retrieval
                # TIMEOUT case (FSM_EXHAUSTED-wrapped DW batch-poll deadline) -> trip the
                # batch lane OPEN -> select_lane rotates the op to realtime.
                _a3_fail_lane = "batch" if _s182_fb_token is not None else "realtime"
                _breaker_record_outcome(
                    _a3_fail_lane,
                    ok=False,
                    failure_mode=failure_source.name,
                    exc=exc,
                )
                # T5 (Dynamic Lane Escalation) -- LANE COLLAPSE detection. After
                # T4 rotated this op off the wedged batch lane onto realtime, a
                # realtime-lane TIMEOUT here means BOTH transport lanes have now
                # failed by timeout for this op. Emit [SOVEREIGN YIELD: LANE
                # COLLAPSE] + record a BOUNDED per-op deadline-dilation hop so the
                # immortal queue's NEXT re-attempt runs at a dilated deadline
                # (DW under heavy global load). Bounded by
                # JARVIS_LANE_DILATION_MAX_HOPS -> falls to the existing
                # immortal/DLQ backstop once exhausted. Fail-soft; OFF byte-id.
                try:
                    _t5_full_op = getattr(context, "op_id", "") or getattr(
                        context, "operation_id", "",
                    ) or ""
                    _record_lane_collapse_dilation(_t5_full_op, _a3_fail_lane, exc)
                except Exception:  # noqa: BLE001 -- never perturb the error path
                    pass
                try:
                    # Slice 201 — feed the bandit a FAILURE reward for this arm
                    # so its posterior learns which models actually deliver.
                    #
                    # Slice 17 (mandate 3): classify the fault FIRST. A 403
                    # entitlement denial, an RT rupture, or a breaker timeout is
                    # a fact about the ENVIRONMENT, not about this model's
                    # generation quality — folding it into alpha/beta is how the
                    # posterior converged on garbage in Run-25c (it down-weighted
                    # models that generate fine and up-weighted an unreachable
                    # one). Infra faults are quarantined into ``infra_faults``;
                    # only genuine quality failures move the posterior.
                    from backend.core.ouroboros.governance.bandit_router import (
                        classify_fault as _s17_classify,
                        get_bandit_router as _s201_bandit_fail,
                    )
                    _s201_bandit_fail().record_outcome(
                        model_id,
                        success=False,
                        fault_class=_s17_classify(exc),
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    # Override Matrix — feed the model-pin soft-lock a real
                    # failure (429/500/live-transport). At threshold the pin
                    # enters cooldown and routing yields to the EWMA ranking.
                    from backend.core.ouroboros.governance.model_pinning_heuristic import (
                        note_pin_outcome as _pin_fail,
                    )
                    _pin_fail(model_id, success=False)
                except Exception:  # noqa: BLE001
                    pass
                # Slice 77 — dynamic transport telemetry. The moment a LIVE
                # generation confirms a transport break, feed it into the
                # dw_surface_health ledger so the NEXT op's Slice 76 P2
                # pre-flight gate fires and skips the dead DW lane (closes the
                # stale-boot-probe gap found in the EVAL-2 Phase-4 re-run,
                # §50.11). Only LIVE_TRANSPORT — 429/5xx/parse are model- or
                # request-specific, not a transport-wide break.
                if failure_source is FailureSource.LIVE_TRANSPORT:
                    _note_dw_live_transport_degraded(
                        f"{model_id}:{type(exc).__name__}",
                        model_id=model_id,  # Slice 175 — attribute the rupture to THIS model
                    )
                    _consecutive_lt += 1
                    # Slice 182 Gap 2 — HEDGE AT THE RUPTURE BOUNDARY. The first rupture is the
                    # absolute first line of defense: immediately COMMAND the remaining probes
                    # in THIS dispatch onto batch, so a fresh-session rupture (before Gap 1's
                    # persisted-degraded signal exists) doesn't walk all 6 models through RT.
                    if not _s182_force_batch:
                        try:
                            from backend.core.ouroboros.governance.doubleword_provider import (
                                dw_hedge_enabled as _s182_hedge_on,
                                _dw_batch_lane_healthy as _s182_bok,
                            )
                            if _s182_hedge_on() and _s182_bok():
                                _s182_force_batch = True
                                logger.warning(
                                    "[Immortal] rupture HEDGE at sentinel boundary: %s ruptured "
                                    "→ remaining probes switched to BATCH (op=%s)",
                                    model_id, op_id_short,
                                )
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    # Slice 83 Phase 2 — a non-transport failure (429/5xx/parse)
                    # proves THIS model's transport is reachable, so the prior
                    # transport breaks were per-model, not lane-wide. Reset the
                    # streak: a genuine blackout is N transport breaks in a row.
                    _consecutive_lt = 0
                    # Slice 176 — fuse the non-transport vector into the predictor
                    # (economic 429 / upstream 5xx+parse / stall), per-model + weighted.
                    _record_dw_failure_signal(model_id, failure_source)
                # Slice 73 + Slice 83 Phase 2 — structural transport short-circuit,
                # now streak-gated. A LIVE_TRANSPORT break MIGHT mean the whole DW
                # endpoint is down — but with the Slice 82/83 heterogeneous coder
                # stack (DeepSeek-V4-Pro / Kimi / GLM / Qwen are distinct served
                # endpoints) a single break may just be one model bouncing. So we
                # ROTATE to the next coder on the first break and only sever once
                # `threshold` consecutive models have ALL failed transport — the
                # signature of a real lane-wide blackout. Severing too early
                # starves the Claude fallback (bt-2026-06-03
                # deadline_exhausted_pre_fallback); rotating too long burns budget
                # on a dead lane. The streak threshold balances both. `=1`
                # reproduces exact Slice 73 first-failure sever.
                if (
                    structural_fast_cascade_enabled()
                    and should_sever_dw_lane(failure_source)
                    and _consecutive_lt >= _lt_sever_threshold
                ):
                    _severed = len(ranked_models) - len(attempts)
                    logger.warning(
                        "[CandidateGenerator] Slice 73/83 structural fast-cascade: "
                        "model=%s LIVE_TRANSPORT streak=%d>=%d — severing DW lane, "
                        "cascading to fallback with full budget (op=%s, skipped %d "
                        "sibling model(s))",
                        model_id, _consecutive_lt, _lt_sever_threshold,
                        op_id_short, max(0, _severed),
                    )
                    break
                if (
                    failure_source is FailureSource.LIVE_TRANSPORT
                    and structural_fast_cascade_enabled()
                ):
                    logger.info(
                        "[CandidateGenerator] Slice 83 granular isolation: "
                        "model=%s LIVE_TRANSPORT streak=%d<%d — rotating to next "
                        "coder, DW lane stays open (op=%s)",
                        model_id, _consecutive_lt, _lt_sever_threshold,
                        op_id_short,
                    )
                continue
        # All DW models exhausted (either OPEN or failed). The
        # per-attempt ContextVar was already reset by each loop
        # iteration's finally block (Slice 3.6) — no further cleanup
        # needed before cascade-to-Claude / queue.
        logger.warning(
            "[CandidateGenerator] Sentinel dispatch: route=%s exhausted "
            "all %d DW models [%s] — applying fallback_tolerance=%s "
            "(op=%s, last_failure=%s)",
            provider_route, len(ranked_models),
            ", ".join(attempts),
            fallback_tolerance, op_id_short, last_failure or "none",
        )
        # Task T2 -- provider health gradient FAILURE record. Every ranked DW
        # model failed for this route: that is ONE failed dispatch sweep. The
        # gradient DEDUCES a global outage from the RATE of these sweeps over a
        # bounded rolling window (NOT a hops count) -- a full all-False window
        # trips is_global_outage, consumed at the immortal re-queue intercept
        # below. Fail-soft: gradient errors never perturb the dispatch path.
        #
        # Slice 4 T2 — a PURE session-budget-refusal exhaustion (every model
        # refused on the local $0.00 gate, no genuine transport failure) is NOT
        # a provider-outage sweep. Recording success=False here would poison the
        # gradient's is_global_outage deduction across ops → spurious DW
        # quarantine (Run #14: a phantom 79-minute global outage from 30 budget
        # refusals). Skip the sweep in that case.
        if _budget_refusal_exc is not None and not _saw_non_refusal_failure:
            # NON-TRANSIENT local budget exhaustion: retrying, cascading,
            # quarantining, or waking the J-Prime failover (real GCE $) cannot
            # help — only an operator refund can. Fail FAST and VISIBLY with a
            # distinct terminal cause so the op surfaces the real class instead
            # of masquerading as a provider outage. This raise preempts the
            # cascade decision, the UPSTREAM QUARANTINE intercept, the immortal
            # re-queue loop, AND the budget-driven GCE failover awaken below.
            logger.error(
                "[Immortal] NON-TRANSIENT budget exhaustion — terminating "
                "retry loop (fail-fast, op fails visibly): op=%s last=%s",
                op_id_short, (last_failure or "?")[:80],
            )
            self._raise_exhausted(
                "fallback_skipped:budget_exhausted_non_transient",
                context=context,
                deadline=deadline,
                primary_exc=_budget_refusal_exc,
            )
        try:
            get_provider_health_gradient().record_sweep(
                provider_route, success=False,
            )
        except Exception:  # noqa: BLE001 -- gradient is advisory, never blocks
            pass
        if fallback_tolerance == "queue":
            # Zero-cost lane FIRST — before the read-only Claude cascade below.
            # Both branches exist to keep a cost-optimized op moving; this one
            # costs $0.00 and that one costs ~$0.005, so consulting Claude
            # first would be paying for the more expensive of two available
            # answers on the route whose entire contract is not to. Returns
            # None when no local lane is serving, leaving the cascade and the
            # queue raise exactly as they were.
            _free_lane = await self._try_free_lane_dispatch(
                context, deadline,
                route=provider_route,
                reason=f"dw_exhausted:{(last_failure or 'all_models_open')[:60]}",
            )
            if _free_lane is not None:
                return _free_lane
            # Defect #5 fix (2026-05-03) — Read-only cascade reflex.
            # Soak v5 (bt-2026-05-03-060330) had 17/19 BG ops terminal-
            # failing here with "background_dw_blocked_by_topology".
            # The legacy reflex in _generate_background() (line ~2806)
            # already turns this into a Claude cascade for read-only
            # ops, but THAT reflex is unreachable because we raise
            # BEFORE returning to _generate_background. Lift the same
            # logic here so it actually fires.
            #
            # Cost contract preserved: read-only ops are policy-safe
            # because Rule 0d (in policy_engine.py) refuses every
            # mutating tool under is_read_only=True. Cascading a
            # read-only op to Claude carries no write risk; only
            # synthesis cost (~$0.005/op).
            #
            # Mutating BG ops still respect JARVIS_BACKGROUND_ALLOW_
            # FALLBACK env knob — they fall through to the queue
            # raise below if the operator hasn't opted in.
            _is_read_only = bool(
                getattr(context, "is_read_only", False),
            )
            _allow_mutating_fallback = (
                provider_route == "background"
                and os.environ.get(
                    "JARVIS_BACKGROUND_ALLOW_FALLBACK", "",
                ).strip().lower() in {"1", "true", "yes", "on"}
            )
            _can_cascade = (
                self._fallback is not None
                and (_is_read_only or _allow_mutating_fallback)
            )
            # Slice 124 — Autonomous Economic Router. On a HARD economic block
            # (DW http_402 balance / 429 rate-limit), a small read-only (or
            # opt-in) BACKGROUND op should not dead-queue: cascade it to the
            # cheap Claude tier to preserve momentum, while MASSIVE ops stay
            # queued (don't pay Claude prices for a big background op). This
            # EXTENDS the read-only cascade with an economic size-gate; the
            # cheap model is resolved from JARVIS_ECONOMIC_FAILOVER_MODEL (no
            # hardcode). Gated + fail-open; default-off → byte-identical.
            try:
                from backend.core.ouroboros.governance import economic_router as _ER

                if _ER.economic_router_enabled() and self._fallback is not None:
                    _prompt_chars = len(str(getattr(context, "prompt", "") or "")) \
                        or len(str(getattr(context, "description", "") or ""))
                    _econ = _ER.decide(
                        route=provider_route,
                        error_text=last_failure or "",
                        prompt_chars=_prompt_chars,
                        is_read_only=_is_read_only,
                    )
                    if _econ.action is _ER.EconomicAction.CASCADE_CHEAP:
                        _can_cascade = True
                        logger.info(
                            "[CandidateGenerator] EconomicRouter: %s → cascade to "
                            "cheap tier '%s' (op=%s, %s)",
                            _econ.reason, _econ.model or "(default fallback)",
                            op_id_short, provider_route,
                        )
                    elif _econ.action is _ER.EconomicAction.QUEUE:
                        # Massive/unsafe op on a hard block — keep it queued
                        # (overrides a would-be read-only cascade for cost).
                        _can_cascade = False
                        logger.info(
                            "[CandidateGenerator] EconomicRouter: %s → staying "
                            "queued for cheap provider (op=%s)",
                            _econ.reason, op_id_short,
                        )
                    # Slice 136 — economic-router cognitive synapse. Fired from
                    # OUTSIDE the pure decide() (the AST-pinned classifier has no
                    # side effects), so the organism remembers its economic
                    # failover decisions. Coalesced per op; gated + non-blocking +
                    # fail-soft.
                    try:
                        from backend.core.ouroboros.governance.episodic_core import (
                            note_route_nowait as _note_route,
                        )
                        _note_route(
                            op_id=str(op_id_short or ""),
                            router="economic",
                            summary=(f"economic {_econ.action.value} → "
                                     f"{_econ.model or 'cheap-default'}"),
                            context={
                                "action": _econ.action.value,
                                "tier": _econ.model or "cheap_default",
                                "route": str(provider_route),
                                "reason": _econ.reason,
                            },
                        )
                    except Exception:  # noqa: BLE001 — synapse never perturbs routing
                        pass
            except Exception:  # noqa: BLE001 - economic routing is best-effort
                logger.debug("[CandidateGenerator] EconomicRouter consult skipped", exc_info=True)
            if _can_cascade:
                _cascade_reason = (
                    "read_only_cost_safe"
                    if _is_read_only
                    else "operator_allow_fallback_env"
                )
                logger.info(
                    "[CandidateGenerator] Sentinel queue tolerance "
                    "OVERRIDE: route=%s cascading to Claude (%s, "
                    "op=%s, fallback_tolerance=queue but is_read_only=%s "
                    "or allow_fallback_env=%s) — Defect #5 fix "
                    "2026-05-03 lifts the read-only reflex from "
                    "_generate_background where it was unreachable "
                    "after sentinel raise",
                    provider_route, _cascade_reason, op_id_short,
                    _is_read_only, _allow_mutating_fallback,
                )
                return await self._call_fallback(context, deadline)
            # Same exception shape the orchestrator's existing
            # accept-failure branch already handles for BG/SPEC.
            if provider_route == "speculative":
                _note_dw_total_outage(last_failure or "")  # Slice 53
                raise RuntimeError(
                    f"speculative_deferred:dw_severed_queued:"
                    f"{(last_failure or 'all_models_open')[:120]}"
                )
            _note_dw_total_outage(last_failure or "")  # Slice 53
            raise RuntimeError(
                f"background_dw_blocked_by_topology:"
                f"dw_severed_queued:"
                f"{(last_failure or 'all_models_open')[:120]}"
            )
        # cascade_to_claude — Claude is the explicit cost contract, BUT only when
        # the Claude lane is actually alive. Slice 238: consult the SAME economic
        # breaker the primary lane respects (read-only ``_claude_breaker_open`` —
        # no probe side-effect) before cascading. When it is OPEN (Claude
        # economically/transport dead) the cascade is suppressed and the op routes
        # to the immortal DW-retry / clean-degrade branch below instead of
        # poisoning the op via a known-dead lane (terminal_quota). Breaker CLOSED
        # → byte-identical legacy cascade (a funded Claude is used normally).
        _claude_lane_open = False
        try:
            from backend.core.ouroboros.governance.doubleword_provider import (
                _claude_breaker_open as _cascade_breaker_open,
            )
            _claude_lane_open = _cascade_breaker_open()
        except Exception:  # noqa: BLE001 — advisory; never block dispatch
            _claude_lane_open = False
        _route_masked = claude_route_masked(context)
        _do_cascade = should_cascade_to_claude(
            has_fallback=self._fallback is not None,
            claude_breaker_open=_claude_lane_open,
            enabled=cascade_breaker_consult_enabled(),
            route_masked=_route_masked,
        )
        if _route_masked and self._fallback is not None:
            logger.info(
                "[CandidateGenerator] route MASKED — claude omitted from "
                "fallback pool by cost contract (route=%s read_only=%s "
                "op=%s); exhausting cheap pool natively",
                getattr(context, "provider_route", "?"),
                getattr(context, "is_read_only", False), op_id_short,
            )
        if not _do_cascade and self._fallback is not None and _claude_lane_open:
            logger.warning(
                "[CandidateGenerator] Slice238 cascade-to-claude SUPPRESSED: "
                "Claude breaker OPEN (economic/transport) — not poisoning op via "
                "the known-dead lane (terminal_quota); routing to immortal "
                "DW-retry/degrade (op=%s, last=%s)",
                op_id_short, (last_failure or "?")[:60],
            )
        # cascade_to_claude — Claude is the explicit cost contract.
        if not _do_cascade:
            # Slice 180 — THE IMMORTAL EXECUTION LAYER. Raising here DELETES the op (the
            # soak's all_providers_exhausted bleed). With NO fallback configured, exhausting
            # is unacceptable. Instead → QUEUE_ONLY: exponential-backoff and RE-ATTEMPT the
            # full DW dispatch until the vendor recovers, bounded by the op's own deadline +
            # a capped attempt count. A transient TOTAL DW outage is survived (the warm-boot
            # + intra-DW failover route the recovered attempt to batch); a permanently-dead
            # DW still fails — but only after exhausting the queue budget, never instantly.
            # Task T2 -- UPSTREAM QUARANTINE intercept (the keystone). BEFORE the
            # immortal re-queue: if the provider health gradient has DEDUCED a
            # global DW outage (full rolling window of all-failed sweeps for this
            # route -- a RATE, never a hops count), the immortal queue would
            # otherwise re-queue this op forever (observed dilation hops=77,
            # hammering a degraded upstream). Instead, terminally seal the op in
            # the Cryo-DLQ via [SOVEREIGN YIELD: UPSTREAM QUARANTINE] and raise a
            # terminal error -- the op is NOT lost (it is replayable from the DLQ).
            # A TRANSIENT failure (window not yet all-False) leaves is_global_outage
            # False, so the EXISTING immortal retry below runs unchanged.
            #
            # Fail-soft ABSOLUTE: any gradient/quarantine error -> fall through to
            # the legacy immortal path (the I1 op-never-lost guarantee holds). When
            # quarantine_enabled() is false the whole block is skipped -> the legacy
            # immortal loop is byte-identical.
            try:
                if quarantine_enabled() and get_provider_health_gradient().is_global_outage(
                    provider_route
                ):
                    try:
                        from backend.core.ouroboros.governance.convergence_watchdog import (  # noqa: PLC0415
                            get_lane_dilation_tracker as _q_get_dt,
                        )
                        _q_hops = _q_get_dt().hops(
                            getattr(context, "op_id", "") or "",
                        )
                    except Exception:  # noqa: BLE001 -- telemetry best-effort
                        _q_hops = -1
                    _q_telemetry = {
                        "route": provider_route,
                        "fleet_exhausted": True,
                        "fleet_size": len(ranked_models),
                        "lanes": "batch+realtime",
                        "failure_mode": "TIMEOUT",
                        "dilation_hops": _q_hops,
                        "last_failure": (last_failure or "all_models_open")[:120],
                    }
                    if quarantine_op(
                        context, route=provider_route, telemetry=_q_telemetry,
                    ):
                        # Phase 1 Outage Ledger -- record DW global outage for the
                        # recovery forecaster + async Trinity export (fail-soft).
                        try:
                            from backend.core.ouroboros.governance.outage_ledger import (  # noqa: PLC0415
                                get_outage_ledger,
                                emit_outage_event,
                            )
                            _ol_rec_id = get_outage_ledger().open_outage(
                                failure_mode=_q_telemetry.get("failure_mode", "TIMEOUT"),
                                error_codes=[str(_q_telemetry.get("last_failure", ""))],
                                lane=str(_q_telemetry.get("lanes", "batch+realtime")),
                                model_ids=[
                                    m for m in [
                                        getattr(context, "model_id", None),
                                    ] if m
                                ],
                                dilation_hops=int(_q_hops) if _q_hops != -1 else 0,
                            )
                            _ol_rec = get_outage_ledger().recent(1)
                            if _ol_rec:
                                emit_outage_event("DW_OUTAGE_DETECTED", _ol_rec[-1])
                        except Exception:  # noqa: BLE001 -- outage ledger never blocks the op
                            pass
                        # Global outage DEDUCED -> terminal Cryo-DLQ seal. Do NOT
                        # immortal re-queue. The op is sealed (replayable), not lost.
                        logger.warning(
                            "[CandidateGenerator] UPSTREAM QUARANTINE: route=%s "
                            "global DW outage deduced (full-window all-failure) -- "
                            "op sealed in Cryo-DLQ, immortal re-queue SKIPPED "
                            "(op=%s, hops=%s)",
                            provider_route, op_id_short, _q_hops,
                        )
                        raise RuntimeError(
                            "upstream_quarantine:dw_global_outage"
                        )
                    # quarantine_op returned False (DLQ seal failed) -> fall through
                    # to the legacy immortal path so the op is never lost.
            except RuntimeError as _q_terminal:
                # The terminal quarantine signal must propagate (it is the op's
                # terminal outcome -- sealed in the DLQ, handled by the caller as a
                # generation failure, NOT a silent drop). Only re-raise OUR
                # sentinel; any other RuntimeError is unexpected and falls through.
                if str(_q_terminal).startswith("upstream_quarantine:"):
                    raise
            except Exception:  # noqa: BLE001 -- quarantine must never itself break the op
                logger.debug(
                    "[CandidateGenerator] UPSTREAM QUARANTINE intercept fail-soft "
                    "-> legacy immortal path", exc_info=True,
                )
            try:
                from backend.core.ouroboros.governance.dw_immortal import (
                    immortal_should_retry as _imm_should_retry,
                    immortal_backoff_s as _imm_backoff,
                    immortal_max_attempts as _imm_max,
                    immortal_max_wait_s as _imm_max_wait,
                    immortal_per_attempt_window_s as _imm_window,
                )
                import time as _imm_time
                from datetime import datetime as _imm_dt, timezone as _imm_tz, timedelta as _imm_td
                _imm_now = _imm_time.time()
                # Slice 182 Gap 3 — the immortal budget is DETACHED from the op's 120s generation
                # deadline: a separate, much-longer wall (default 1h) computed ONCE and threaded
                # across the retry recursion, so a sustained DW outage doesn't expire the op.
                _imm_budget = (
                    _immortal_budget_deadline if _immortal_budget_deadline is not None
                    else (_imm_now + _imm_max_wait())
                )
                if _imm_should_retry(
                    deadline=_imm_budget, now=_imm_now, claude_available=False,
                    attempt=_immortal_attempt, max_attempts=_imm_max(),
                ):
                    # CR2 (LIVE sentinel-cascade path): DW is exhausted with NO cloud
                    # fallback -> wake the J-Prime golden-image fallback. This Immortal
                    # re-attempt loop keeps the op alive (deadline-detached) until the
                    # failover FSM reaches SERVING, after which the recursive
                    # _dispatch_via_sentinel routes this very op to J-Prime via the
                    # Phase-3c seam. Double-gated + fail-soft; idempotent anchor.
                    try:
                        from .failover_lifecycle import (
                            lifecycle_enabled as _fo_on,
                            budget_awaken_enabled as _fo_budget,
                            get_failover_controller as _fo_ctrl,
                        )
                        if _fo_on() and _fo_budget():
                            _fo_ctrl().note_budget_exhausted()
                    except Exception:  # noqa: BLE001 -- never let the awaken signal break the op
                        pass
                    _imm_delay = _imm_backoff(_immortal_attempt)
                    logger.warning(
                        "[Immortal] DW exhausted + NO fallback → QUEUE_ONLY (deadline-detached): "
                        "backoff %.1fs then re-attempt #%d, budget %.0fs remaining (op NEVER lost; "
                        "op=%s, last=%s)",
                        _imm_delay, _immortal_attempt + 1, max(0.0, _imm_budget - _imm_now),
                        op_id_short, (last_failure or "?")[:60],
                    )
                    await asyncio.sleep(_imm_delay)
                    # FRESH generation window for the retry (not the original op's elapsed deadline)
                    _imm_fresh_deadline = _imm_dt.now(_imm_tz.utc) + _imm_td(seconds=_imm_window())
                    return await self._dispatch_via_sentinel(
                        context, _imm_fresh_deadline, provider_route,
                        _immortal_attempt=_immortal_attempt + 1,
                        _immortal_budget_deadline=_imm_budget,
                    )
            except Exception as _imm_exc:  # noqa: BLE001 — the immortal layer must never itself break the op
                logger.debug("[Immortal] queue-retry path swallowed: %r", _imm_exc)
            _note_dw_total_outage(last_failure or "")  # Slice 53
            raise RuntimeError(
                f"sentinel_dispatch_no_fallback:"
                f"{(last_failure or 'all_models_open')[:120]}"
            )
        return await self._call_fallback(context, deadline)

    async def _generate_immediate(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """IMMEDIATE route: Claude direct, skip DW entirely.

        For critical-urgency operations where every second counts:
        test failures, voice commands, runtime health critical.

        Cost: ~$0.03/op (Claude only)
        Latency: 15-30s (no DW overhead)
        """
        logger.info(
            "[CandidateGenerator] IMMEDIATE route: Claude direct "
            "(skip DW, urgency=%s, source=%s) [%.1fs remaining]",
            getattr(context, "signal_urgency", "?"),
            getattr(context, "signal_source", "?"),
            self._remaining_seconds(deadline),
        )

        # Try Claude as primary first, then fallback if available.
        # Skip the entire Tier 0 / DW path.
        state = self.fsm.state
        if state is FailbackState.QUEUE_ONLY:
            self._raise_exhausted(
                "queue_only_immediate",
                context=context,
                deadline=deadline,
            )

        # If DW IS the primary, go straight to Claude (the fallback).
        _dw_is_primary = (self._tier0 is not None and self._primary is self._tier0)
        if _dw_is_primary:
            # Slice 127 P2.1 — fallback-skip gate. Claude-direct would just
            # grind an IMMEDIATE op against a depleted Claude lane (the live
            # soak: terminal_quota x N, no completion). When the Claude lane
            # breaker is OPEN (economic/transport), reroute to the funded DW
            # primary instead. Slice 162 — read the breaker STATE (read-only) via the
            # Slice 161 predicate, NOT should_allow_request(): the latter flickers True
            # during a HALF_OPEN probe AND has a side effect (consumes the probe slot),
            # so an IMMEDIATE op kept hammering a dead-but-probing Claude and exhausted
            # before the gate. Now CLOSED → Claude-direct (self-heal); OPEN/HALF_OPEN →
            # reroute to funded DW. Gated default-FALSE → OFF is unchanged Claude-direct.
            if fallback_skip_gate_enabled():
                try:
                    from backend.core.ouroboros.governance.claude_circuit_breaker import (  # noqa: E501
                        get_claude_circuit_breaker as _p21_ccb,
                        is_enabled as _p21_ccb_enabled,
                    )
                    from backend.core.ouroboros.governance.doubleword_provider import (
                        _claude_breaker_open as _p21_breaker_open,
                    )
                    _p21_allows = not _p21_breaker_open(getter=_p21_ccb)
                    if immediate_reroute_to_dw(
                        dw_is_primary=True,
                        gate_enabled=True,
                        claude_breaker_enabled=_p21_ccb_enabled(),
                        claude_allows_request=_p21_allows,
                    ):
                        logger.warning(
                            "[CandidateGenerator] IMMEDIATE reroute → DW: "
                            "Claude lane breaker OPEN (economic/transport) — "
                            "bypassing depleted Claude, routing to funded DW "
                            "primary (op=%s)",
                            getattr(context, "op_id", "?"),
                        )
                        return await self._call_primary(context, deadline)
                except Exception:  # noqa: BLE001 — never block dispatch
                    pass
            return await self._call_fallback(context, deadline)

        # Otherwise try primary (Claude/J-Prime), then fallback.
        return await self._try_primary_then_fallback(context, deadline)

    async def _try_jprime_primacy(
        self,
        context: OperationContext,
        deadline: datetime,
        *,
        route_label: str,
        force: bool = False,
    ) -> Optional[GenerationResult]:
        """Phase 3 Scope α: try J-Prime first for BACKGROUND/SPECULATIVE.

        Returns the ``GenerationResult`` on success, or ``None`` to
        signal "fall through to DW". Never raises — all failure modes
        (flag off, no handle, sem saturated, generate error, empty
        result) are translated into a ``None`` return plus a counter
        bump so the caller can take the DW-only path unchanged.

        Parameters
        ----------
        context, deadline:
            The usual generation args, forwarded to J-Prime unchanged.
        route_label:
            ``"BACKGROUND"`` or ``"SPECULATIVE"``. Used only in log
            messages so operators can tell which route took which
            branch of the primacy path.

        Why a pre-check on ``self._jprime_sem.locked()``:
            ``asyncio.Semaphore(1)`` with overflow-fall-through has no
            clean primitive. We want "try to grab it right now, and if
            already held, don't queue — go to DW instead." The
            ``locked()`` check is a tiny race (a sibling op could take
            the token in the gap between the check and the acquire),
            but the worst case is that we serialize two ops for one
            J-Prime call, which is harmless. Using
            ``wait_for(acquire, timeout=0)`` would raise
            ``CancelledError`` on some asyncio versions and obscure
            the intent; ``locked()`` is clearer.
        """
        # Deferred import — ``jprime_primacy_enabled`` is a module-level
        # function in ``_governance_state``, and fetching it at call
        # time keeps the hot-path branch cheap when the flag is off.
        from ._governance_state import jprime_primacy_enabled

        # Quota Shield: prefer_local ops use the same local-first primacy path even
        # when jprime_primacy is otherwise off for this route.
        # Sovereign Failover Mesh Gap 3b: ``force=True`` (a Cryo-DLQ
        # provider_override replay) bypasses the primacy-enabled guard entirely
        # -- the op was explicitly pinned to J-Prime, so honor it regardless of
        # the primacy flag for this route.
        if not (
            force
            or jprime_primacy_enabled()
            or getattr(context, "prefer_local", False)
        ):
            return None
        if self._jprime is None or not getattr(
            self._jprime, "provider_name", ""
        ):
            return None

        # Sem saturation — a sibling op is already using the single
        # client-side slot. Don't queue; fall through to DW so the
        # background workload doesn't serialize behind one J-Prime call.
        if self._jprime_sem.locked():
            self._jprime_counters.jprime_sem_overflows += 1
            self._jprime_counters.fallthrough_to_dw += 1
            logger.info(
                "[CandidateGenerator] %s: J-Prime sem saturated (overflows=%d) "
                "— falling through to DW",
                route_label,
                self._jprime_counters.jprime_sem_overflows,
            )
            return None

        remaining = self._remaining_seconds(deadline)
        if remaining <= 0.0:
            # No budget left — fall through silently so the DW path
            # can emit its own deadline-exceeded diagnostic.
            return None

        async with self._jprime_sem:
            try:
                result = await asyncio.wait_for(
                    self._jprime.generate(context, deadline),
                    timeout=min(remaining, 180.0),
                )
            except asyncio.TimeoutError:
                self._jprime_counters.jprime_failures += 1
                self._jprime_counters.fallthrough_to_dw += 1
                logger.info(
                    "[CandidateGenerator] %s: J-Prime primacy timeout after "
                    "%.1fs — falling through to DW",
                    route_label,
                    remaining,
                )
                return None
            except Exception as exc:
                self._jprime_counters.jprime_failures += 1
                self._jprime_counters.fallthrough_to_dw += 1
                logger.info(
                    "[CandidateGenerator] %s: J-Prime primacy error "
                    "%s(%s) — falling through to DW",
                    route_label,
                    type(exc).__name__,
                    exc,
                )
                return None

        if result is None or len(getattr(result, "candidates", ()) or ()) == 0:
            self._jprime_counters.jprime_failures += 1
            self._jprime_counters.fallthrough_to_dw += 1
            logger.info(
                "[CandidateGenerator] %s: J-Prime primacy returned no "
                "candidates — falling through to DW",
                route_label,
            )
            return None

        self._jprime_counters.jprime_hits += 1
        logger.info(
            "[CandidateGenerator] %s: J-Prime primacy hit — %d candidates "
            "in %.1fs (hits=%d, overflows=%d, failures=%d)",
            route_label,
            len(result.candidates),
            getattr(result, "generation_duration_s", 0.0) or 0.0,
            self._jprime_counters.jprime_hits,
            self._jprime_counters.jprime_sem_overflows,
            self._jprime_counters.jprime_failures,
        )
        return result

    def _get_resilience_provider(self, route: str) -> Tuple[Optional[Any], str]:
        """Lazy, fail-soft resolver for the hosted resilience lane's armed
        provider on *route*. Returns ``(provider, reason)``; ``(None, ...)``
        on any disarm/import failure — a broken lane module can NEVER brick
        generator dispatch (Bulletproof mandate). All arm/disarm authority
        lives in hosted_resilience_lane.py, driven by policy shape only."""
        lane = getattr(self, "_hosted_resilience_lane", None)
        if lane is None:
            if getattr(self, "_hosted_resilience_lane_broken", False):
                return None, "lane_module_broken"
            try:
                from backend.core.ouroboros.governance.hosted_resilience_lane import (  # noqa: E501
                    HostedResilienceLane,
                )
                lane = HostedResilienceLane()
                self._hosted_resilience_lane = lane
            except Exception as exc:  # noqa: BLE001 — lane must never brick dispatch
                self._hosted_resilience_lane_broken = True
                logger.warning(
                    "[CandidateGenerator] hosted resilience lane import "
                    "failed (%s) — lane dark for this session", exc,
                )
                return None, "lane_import_failed"
        return lane.provider_for_route(route)

    async def _try_hosted_resilience_lane(
        self,
        context: OperationContext,
        deadline: datetime,
        *,
        route: str,
        tier0_error: Optional[str],
    ) -> Optional[GenerationResult]:
        """Consult the policy-driven hosted resilience lane after Tier-0
        exhaustion (LongCat stub Phase 1 — hosted_resilience_lane.py).

        GENERIC by design: candidates and their routes come exclusively
        from ``brain_selection_policy.yaml`` (``hosted_provider_candidates.
        *.resilience_lane``); no vendor name appears in this FSM. The lane
        is DOUBLE-DARK by default (policy ``enabled: false`` + master env
        default-false + Phase 0 verdict gate), so this consult is a single
        cached no-op in production today.

        Returns a GenerationResult on lane success, ``None`` on ANY
        disarm/failure condition (caller falls through to its legacy
        behavior unchanged — Bulletproof mandate: never an unhandled
        exception out of a dark lane), with ONE exception class:
        ``SessionBudgetPreflightRefused`` propagates. The lane provider is
        a real ClaudeProvider, so its internal wallet preflight raises the
        same exception DW/Claude raise — re-raising keeps a $0.00 session
        on the Slice 4 T2 ``is_budget_refusal`` axis (local gate, fail
        fast + visible) instead of masquerading as a lane fault or, worse,
        silently igniting the J-Prime GCE failover.
        """
        provider, reason = self._get_resilience_provider(route)
        if provider is None:
            return None
        logger.info(
            "[CandidateGenerator] %s: Tier-0 exhausted (%s) — trying hosted "
            "resilience lane (%s) [%s]",
            route.upper(), tier0_error or "unavailable", reason,
            getattr(context, "op_id", "?")[:16],
        )
        try:
            result = await provider.generate(context, deadline)
        except Exception as exc:  # noqa: BLE001 — classify, then fall through
            from backend.core.ouroboros.governance.session_budget_authority import (  # noqa: E501
                is_budget_refusal,
            )
            if is_budget_refusal(exc):
                raise  # T2 axis: local wallet gate, NOT a lane/provider fault
            logger.warning(
                "[CandidateGenerator] %s: resilience lane failed "
                "(%s: %s) — falling through to legacy path [%s]",
                route.upper(), type(exc).__name__, str(exc)[:120],
                getattr(context, "op_id", "?")[:16],
            )
            return None
        if result is not None and len(result.candidates) > 0:
            logger.info(
                "[CandidateGenerator] %s: resilience lane produced %d "
                "candidates in %.1fs ($%.4f) [%s]",
                route.upper(), len(result.candidates),
                getattr(result, "generation_duration_s", 0.0) or 0.0,
                getattr(result, "cost_usd", 0.0) or 0.0,
                getattr(context, "op_id", "?")[:16],
            )
            return result
        logger.info(
            "[CandidateGenerator] %s: resilience lane returned empty — "
            "falling through to legacy path [%s]",
            route.upper(), getattr(context, "op_id", "?")[:16],
        )
        return None

    async def _generate_background(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """BACKGROUND route: DW primary, optional Claude safety-net cascade.

        For low-urgency background sensors: opportunity mining,
        doc staleness, TODO scanning, backlog items.

        Default behavior (``JARVIS_BACKGROUND_ALLOW_FALLBACK`` unset): DW
        only, no Claude cascade. Cost ~$0.002/op. Raises
        ``RuntimeError("background_dw_*")`` on failure — the orchestrator
        accepts it gracefully and the sensor re-detects if still relevant.

        Nervous-system reflex (``JARVIS_BACKGROUND_ALLOW_FALLBACK=true``):
        when DW times out, is empty, or errors, cascade to Claude via
        :meth:`_call_fallback`. Diagnosed after bt-2026-04-14-041952
        showed **11/11 BACKGROUND ops dying on `background_dw_timeout:180s`**
        — every op exhausted its DW window, nothing reached the Iron
        Gate, and the cost-optimization invariant of the route became a
        100% failure mode. Staking survival of background cognition on a
        single latency-bound provider without a safety net violates
        Manifesto §5 (intelligence-driven routing) and §6
        (threshold-triggered neuroplasticity). On fallback failure, we
        raise ``RuntimeError("background_fallback_failed:...")`` so the
        orchestrator's existing BACKGROUND accept-failure branch still
        fires (no retry loop thrash).

        Bypass (``FORCE_CLAUDE_BACKGROUND=true``): skip DW entirely and
        call Claude directly. Used by the live-fire harness to unblock
        parity validation when DW 397B is degraded — hands BACKGROUND
        cognition straight to Claude so the generation actually reaches
        the tool loop and the Iron Gate can be exercised.

        Phase 3 Scope α (``JARVIS_JPRIME_PRIMACY``): when enabled and a
        PrimeProvider handle is wired, :meth:`_try_jprime_primacy` is
        consulted first. Sem saturation or any failure falls through to
        the DW path below unchanged.
        """
        _urgency = getattr(context, "signal_urgency", "?")
        _source = getattr(context, "signal_source", "?")
        _is_read_only = bool(getattr(context, "is_read_only", False))
        remaining = self._remaining_seconds(deadline)

        _force_claude = os.environ.get(
            "FORCE_CLAUDE_BACKGROUND", "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        _allow_fallback = os.environ.get(
            "JARVIS_BACKGROUND_ALLOW_FALLBACK", "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Nervous System Reflex: read-only ops ALWAYS get the Claude cascade
        # on DW failure, regardless of the env gate. Locking a read-only
        # cartography op onto a paused DW endpoint is the exact failure
        # mode that prompted this reflex (bt-2026-04-18-032820).
        if _is_read_only and self._fallback is not None:
            _allow_fallback = True

        # ── FORCE_CLAUDE_BACKGROUND bypass ─────────────────────────────
        # Skip DW entirely and route straight to Claude. No DW attempt,
        # no timeout, no cascade — used when DW is known-degraded and
        # we need BACKGROUND ops to actually reach the tool loop.
        if _force_claude:
            if self._fallback is None:
                raise RuntimeError(
                    "background_dw_unavailable:force_claude_set_but_no_fallback"
                )
            logger.info(
                "[CandidateGenerator] BACKGROUND: FORCE_CLAUDE_BACKGROUND=true "
                "— bypassing DW, calling Claude directly "
                "(urgency=%s, source=%s) [%.1fs budget]",
                _urgency, _source, remaining,
            )
            try:
                return await self._call_fallback(context, deadline)
            except GovernanceDeadlockError:
                raise  # LR3 terminal -- never wrap as fallback_failed
            except Exception as exc:
                raise RuntimeError(
                    f"background_fallback_failed:forced:"
                    f"{type(exc).__name__}:{str(exc)[:100]}"
                ) from exc

        logger.info(
            "[CandidateGenerator] BACKGROUND route: DW primary%s "
            "(urgency=%s, source=%s, read_only=%s) [%.1fs budget"
            "%s]",
            " + Claude cascade" if _allow_fallback else " (no Claude cascade)",
            _urgency, _source, _is_read_only, remaining,
            f", DW stall budget={_BG_READONLY_DW_STALL_BUDGET_S:.0f}s"
            if _is_read_only else "",
        )

        # Phase 3 Scope α — J-Prime primacy pre-check. Returns
        # ``None`` when the flag is off, no handle is wired, the sem is
        # saturated, or the J-Prime call failed. On ``None``, drop into
        # the existing DW path below.
        _primacy_result = await self._try_jprime_primacy(
            context, deadline, route_label="BACKGROUND",
        )
        if _primacy_result is not None:
            return _primacy_result

        if self._tier0 is None or not getattr(self._tier0, "is_available", False):
            # DW not configured — resilience lane, then Claude cascade if
            # allowed, else raise (lane is dark-by-default; see
            # _try_hosted_resilience_lane).
            _lane_result = await self._try_hosted_resilience_lane(
                context, deadline, route="background",
                tier0_error="background_dw_unavailable:tier0_not_configured",
            )
            if _lane_result is not None:
                return _lane_result
            if _allow_fallback and self._fallback is not None:
                logger.info(
                    "[CandidateGenerator] BACKGROUND: DW unavailable — "
                    "cascading to Claude fallback [%s]",
                    getattr(context, "op_id", "?")[:16],
                )
                try:
                    return await self._call_fallback(context, deadline)
                except GovernanceDeadlockError:
                    raise  # LR3 terminal -- never wrap as fallback_failed
                except Exception as exc:
                    raise RuntimeError(
                        f"background_fallback_failed:dw_unavailable:"
                        f"{type(exc).__name__}:{str(exc)[:100]}"
                    ) from exc
            raise RuntimeError(
                "background_dw_unavailable:tier0_not_configured"
            )

        # Reserve a slice of the BACKGROUND budget for Claude when
        # cascade is enabled so DW can't burn the entire window. The
        # DW cap here and the urgency_router's max_dw_wait_s for
        # BACKGROUND must agree — both tightened to 150s when fallback
        # is enabled.
        #
        # Nervous-System Reflex: read-only ops get a MUCH tighter DW
        # stall budget (default 60s via JARVIS_BG_DW_STALL_BUDGET_S)
        # so lockup is bounded. The Trinity cartography op is the
        # canonical case — it needs to reach the tool loop quickly so
        # dispatch_subagent can fan out; spending 150s on a stalled DW
        # stream is dead time the subagent fleet will never recover.
        if _is_read_only:
            _dw_cap = _BG_READONLY_DW_STALL_BUDGET_S
        else:
            _dw_cap = 150.0 if _allow_fallback else 180.0
        _dw_timeout = min(remaining, _dw_cap)
        _dw_error: Optional[str] = None

        # DW attempt — RT SSE preferred, batch fallback.
        # Phase 12 Slice F — Substrate Error Unmasking. Preserve the
        # underlying DoublewordInfraError on this attempt so the
        # sentinel-driven dispatcher can read its status_code +
        # response_body fields directly. The exception is still
        # caught here (so the legacy non-sentinel path can fall
        # through to Claude as before via _dw_error string), but
        # _structured_error captures the structured object for the
        # caller — when present, the caller re-raises it instead of
        # stringifying it through RuntimeError(_dw_error).
        _structured_error: Optional[Exception] = None
        if getattr(self._tier0, "_realtime_enabled", False):
            # Dynamic 5xx Resiliency Matrix (2026-07-22): absorb a transient
            # upstream/network blip on the DW primary with a bounded
            # exponential-backoff-with-jitter retry HERE — before cascading and
            # before any terminal breaker trip. Async sleep only; the ASGI
            # event loop is never dropped.
            _dw_transient_budget = _dw_transient_max_retries()
            _dw_attempt = 0
            while True:
                try:
                    result = await asyncio.wait_for(
                        self._tier0.generate(context, deadline),
                        timeout=_dw_timeout,
                    )
                    if result is not None and len(result.candidates) > 0:
                        logger.info(
                            "[CandidateGenerator] BACKGROUND: DW produced %d candidates "
                            "in %.1fs ($%.4f)",
                            len(result.candidates),
                            result.generation_duration_s,
                            getattr(result, "cost_usd", 0.0),
                        )
                        return result
                    _dw_error = "background_dw_empty_result"
                    break
                except asyncio.TimeoutError:
                    _dw_error = f"background_dw_timeout:{_dw_timeout:.0f}s"
                    break
                except Exception as exc:
                    _structured_error = exc  # Slice F preserves the object
                    # Build a richer _dw_error that surfaces status_code
                    # + a body excerpt when available (DoublewordInfraError),
                    # so legacy log-line consumers see ground truth too.
                    _status = getattr(exc, "status_code", None)
                    _body = getattr(exc, "response_body", "") or ""
                    _retry_after_ts = getattr(exc, "ratelimit_reset_ts", None)
                    # Transient network/upstream blip → absorb-and-retry (never
                    # cascade to a dead fallback, never terminally trip).
                    _budget_ok = (
                        self._remaining_seconds(deadline) > _dw_timeout * 0.5
                    )
                    if (
                        _is_dw_transient_network(exc, _status, _retry_after_ts)
                        and _dw_attempt < _dw_transient_budget
                        and _budget_ok
                    ):
                        _delay = _dw_transient_backoff_s(
                            _dw_attempt, _retry_after_ts,
                            remaining_s=self._remaining_seconds(deadline),
                        )
                        logger.warning(
                            "[CandidateGenerator] BACKGROUND: DW TRANSIENT_NETWORK "
                            "(%s http_%s) — full-jitter backoff %.1fs, retry %d/%d "
                            "[%s] (absorbed; no cascade, no terminal trip)",
                            type(exc).__name__, _status, _delay,
                            _dw_attempt + 1, _dw_transient_budget,
                            getattr(context, "op_id", "?")[:16],
                        )
                        await asyncio.sleep(_delay)
                        _dw_attempt += 1
                        continue
                    if _status is not None:
                        _dw_error = (
                            f"background_dw_error:{type(exc).__name__}:"
                            f"http_{_status}:{_body[:120]}"
                        )
                    else:
                        _dw_error = (
                            f"background_dw_error:{type(exc).__name__}:{exc}"
                        )
                    break
        else:
            # Legacy batch path
            try:
                pending = await self._tier0.submit_batch(context)
                if pending is None:
                    _dw_error = "background_dw_batch_submit_failed"
                else:
                    result = await asyncio.wait_for(
                        self._tier0.poll_and_retrieve(pending, context),
                        timeout=_dw_timeout,
                    )
                    if result is not None and len(result.candidates) > 0:
                        logger.info(
                            "[CandidateGenerator] BACKGROUND batch: DW produced "
                            "%d candidates",
                            len(result.candidates),
                        )
                        return result
                    _dw_error = "background_dw_batch_empty"
            except asyncio.TimeoutError:
                _dw_error = "background_dw_batch_timeout"
            except Exception as exc:
                _dw_error = (
                    f"background_dw_batch_error:{type(exc).__name__}"
                )

        # DW exhausted. Policy-driven hosted resilience lane first (LongCat
        # stub Phase 1): the cheap metered fallback for the route that
        # otherwise has NONE. Dark by default (double-gated in policy + env
        # + Phase 0 verdict); returns None on any disarm/failure so the
        # legacy cascade/raise below is byte-identical when dark. A
        # SessionBudgetPreflightRefused from the lane propagates (Slice 4
        # T2 axis — a $0 wallet is a local gate, not a lane fault).
        _lane_result = await self._try_hosted_resilience_lane(
            context, deadline, route="background", tier0_error=_dw_error,
        )
        if _lane_result is not None:
            return _lane_result

        # Either cascade to Claude or raise.
        if _allow_fallback and self._fallback is not None:
            _post_dw_remaining = self._remaining_seconds(deadline)
            logger.info(
                "[CandidateGenerator] BACKGROUND: DW failed (%s) — "
                "cascading to Claude fallback, %.1fs parent remaining [%s]",
                _dw_error, _post_dw_remaining, getattr(context, "op_id", "?")[:16],
            )
            try:
                return await self._call_fallback(context, deadline)
            except GovernanceDeadlockError:
                raise  # LR3 terminal -- never wrap as fallback_failed
            except Exception as exc:
                raise RuntimeError(
                    f"background_fallback_failed:dw={_dw_error[:80]}:"
                    f"{type(exc).__name__}:{str(exc)[:80]}"
                ) from exc

        # Phase 12 Slice F — Substrate Error Unmasking. When DW raised
        # a structured DoublewordInfraError (status_code + response_body
        # available), re-raise the ORIGINAL object so the sentinel
        # dispatch classifier can introspect status_code directly
        # without regex on str(exc). Falls through to RuntimeError when
        # the failure was a timeout / empty-result (no structured
        # exception to preserve).
        if _structured_error is not None:
            raise _structured_error
        raise RuntimeError(_dw_error)

    async def _generate_speculative(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """SPECULATIVE route: DW fire-and-forget pre-computation.

        For intent discovery, dream engine, proactive exploration.
        Submit to DW and don't block — store result for later use.

        Cost: ~$0.001/op (DW batch, tolerate high discard)
        Latency: N/A (async, result consumed later)

        Normally raises ``RuntimeError("speculative_deferred")`` — the
        orchestrator should mark this as a deferred operation, not a
        failure.

        Phase 3 Scope α (``JARVIS_JPRIME_PRIMACY``): when enabled and a
        PrimeProvider handle is wired, :meth:`_try_jprime_primacy` is
        consulted first. Because J-Prime on primacy runs synchronously
        inside the sem, a successful primacy hit *returns the result
        directly* instead of raising ``speculative_deferred`` — the
        caller gets a real synchronous result that can be used
        immediately, which is strictly better than a deferred batch.
        Sem saturation or any failure falls through to the existing
        DW fire-and-forget path below unchanged.
        """
        _source = getattr(context, "signal_source", "?")
        _op_id = getattr(context, "op_id", "unknown")

        logger.info(
            "[CandidateGenerator] SPECULATIVE route: DW fire-and-forget "
            "(source=%s, op=%s)",
            _source, _op_id,
        )

        # Phase 3 Scope α — J-Prime primacy pre-check. Synchronous hit
        # upgrades the op from deferred to completed; any miss falls
        # through to the legacy DW fire-and-forget path below.
        _primacy_result = await self._try_jprime_primacy(
            context, deadline, route_label="SPECULATIVE",
        )
        if _primacy_result is not None:
            return _primacy_result

        if self._tier0 is not None and getattr(self._tier0, "is_available", False):
            if getattr(self._tier0, "_realtime_enabled", False):
                # Use RT path but don't block — create background task.
                _gen_task = asyncio.ensure_future(
                    self._tier0.generate(context, deadline),
                )
                # Defect #4 Slice A — speculative pre-dispatch site.
                # Stored for later retrieval, but if op completes
                # without retrieving (timeout / route change /
                # demotion), the task continues. Callback consumes
                # exceptions silently.
                _gen_task.add_done_callback(_swallow_task_exception)
                # Store for later retrieval
                self._background_polls[_op_id] = _gen_task
                logger.info(
                    "[CandidateGenerator] SPECULATIVE: DW RT task dispatched "
                    "as background (op=%s)",
                    _op_id,
                )
            else:
                # Batch path — submit and background poll
                try:
                    pending = await self._tier0.submit_batch(context)
                    if pending is not None:
                        task = asyncio.create_task(
                            self._background_poll_tier0(pending, context),
                            name=f"speculative-{_op_id[:12]}",
                        )
                        # Defect #4 Slice A — defensive callback.
                        task.add_done_callback(_swallow_task_exception)
                        self._background_polls[_op_id] = task
                        logger.info(
                            "[CandidateGenerator] SPECULATIVE: DW batch submitted "
                            "(op=%s, batch=%s)",
                            _op_id, getattr(pending, "batch_id", "?"),
                        )
                except Exception as exc:
                    logger.debug(
                        "[CandidateGenerator] SPECULATIVE: batch submit failed: %s",
                        exc,
                    )

        else:
            # Tier-0 unavailable — the hosted resilience lane (dark by
            # default; policy-driven, see _get_resilience_provider) keeps
            # SPECULATIVE pre-computation alive through a DW outage.
            # Fire-and-forget mirror of the DW RT idiom above: dispatch as
            # a background task, store for later retrieval, swallow
            # exceptions (the route tolerates high discard by contract).
            _lane_provider, _lane_reason = self._get_resilience_provider(
                "speculative",
            )
            if _lane_provider is not None:
                _lane_task = asyncio.ensure_future(
                    _lane_provider.generate(context, deadline),
                )
                _lane_task.add_done_callback(_swallow_task_exception)
                self._background_polls[_op_id] = _lane_task
                logger.info(
                    "[CandidateGenerator] SPECULATIVE: Tier-0 unavailable — "
                    "resilience lane task dispatched as background "
                    "(%s, op=%s)", _lane_reason, _op_id,
                )

        # Always raise — speculative ops are deferred, not completed.
        raise RuntimeError("speculative_deferred")

    async def _background_poll_tier0(
        self, pending: Any, context: OperationContext,
    ) -> None:
        """Background task: poll Doubleword batch and store result when ready."""
        _op_id = pending.op_id
        try:
            assert self._tier0 is not None  # guaranteed by caller
            result = await self._tier0.poll_and_retrieve(pending, context)
            if result is not None and len(result.candidates) > 0:
                from backend.core.ouroboros.governance.doubleword_provider import (
                    CompletedBatch,
                )
                self._completed_batches[_op_id] = CompletedBatch(
                    op_id=_op_id,
                    batch_id=pending.batch_id,
                    result=result,
                    completed_at=time.monotonic(),
                )
                # Record TIER0_COMPLETE in ledger
                await self._record_tier0_ledger(
                    _op_id, "tier0_complete", {
                        "batch_id": pending.batch_id,
                        "candidates": len(result.candidates),
                        "provider": result.provider_name,
                        "duration_s": round(result.generation_duration_s, 1),
                    },
                )
                logger.info(
                    "[CandidateGenerator] Tier 0 background poll complete: "
                    "batch %s → %d candidates stored for op %s",
                    pending.batch_id, len(result.candidates), _op_id,
                )
            else:
                logger.info(
                    "[CandidateGenerator] Tier 0 background poll: "
                    "batch %s returned no usable candidates",
                    pending.batch_id,
                )
        except asyncio.CancelledError:
            logger.debug(
                "[CandidateGenerator] Tier 0 background poll cancelled: %s",
                pending.batch_id,
            )
        except Exception as _poll_exc:
            logger.warning(
                "[CandidateGenerator] Tier 0 background poll failed: %s",
                pending.batch_id,
                exc_info=True,
            )
            # ── Syntax-failure escalation recording (batch path) ──
            # The batch poll swallows the exception here. Record in the
            # escalator so the J-Prime cascade can fire on the NEXT
            # orchestrator retry. Mirrors the RT path recording.
            _poll_err = str(_poll_exc)
            if "all_candidates_syntax_error" in _poll_err:
                try:
                    from backend.core.ouroboros.governance.syntax_escalation import (  # noqa: E501
                        get_escalator as _get_sx_batch,
                    )
                    _sx_b = _get_sx_batch()
                    _sx_b.record_attempt(
                        op_id=getattr(context, "op_id", "") or _op_id or "",
                        error_msg=_poll_err,
                        candidate_preview=getattr(
                            _poll_exc, "candidate_preview", "",
                        ) or "",
                        target_file=getattr(
                            _poll_exc, "target_file", "",
                        ) or "",
                    )
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._background_polls.pop(_op_id, None)

    def get_completed_batch(self, op_id: str) -> Optional[Any]:
        """Check if a Tier 0 async result is available for the given op_id."""
        return self._completed_batches.get(op_id)

    def pop_completed_batch(self, op_id: str) -> Optional[Any]:
        """Retrieve and remove a completed Tier 0 result for the given op_id."""
        return self._completed_batches.pop(op_id, None)

    async def _record_tier0_ledger(
        self, op_id: str, state_name: str, data: dict[str, Any],
    ) -> None:
        """Record a Tier 0 batch event in the governance ledger.

        Fails silently — ledger writes must never crash the pipeline.
        """
        if self._ledger is None:
            return
        try:
            from backend.core.ouroboros.governance.ledger import (
                LedgerEntry,
                OperationState,
            )
            entry = LedgerEntry(
                op_id=op_id,
                state=OperationState(state_name),
                data=data,
                entry_id=data.get("batch_id"),
            )
            await self._ledger.append(entry)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "[CandidateGenerator] Ledger write failed for %s:%s",
                op_id, state_name, exc_info=True,
            )

    async def run_health_probe(self) -> bool:
        """Probe the primary provider and update the FSM.

        Returns
        -------
        bool
            ``True`` if the primary is healthy.
        """
        try:
            healthy = await self._primary.health_probe()
        except Exception:
            logger.warning(
                "[CandidateGenerator] Health probe raised exception, treating as failure",
                exc_info=True,
            )
            healthy = False

        if healthy:
            self.fsm.record_probe_success()
        else:
            self.fsm.record_probe_failure()

        return healthy

    async def plan(self, prompt: str, deadline: datetime) -> str:
        """Send a planning prompt to the active provider, with soft fallback.

        Does NOT update the failback state machine on failure — planning errors
        are non-fatal and the orchestrator continues to GENERATE regardless.

        Raises RuntimeError("all_providers_exhausted") only if QUEUE_ONLY.
        """
        state = self.fsm.state

        if state is FailbackState.QUEUE_ONLY:
            self._raise_exhausted(
                "queue_only_plan",
                deadline=deadline,
                phase="plan",
            )

        if state is FailbackState.PRIMARY_READY:
            try:
                async with self._primary_sem:
                    remaining = self._remaining_seconds(deadline)
                    primary_budget = min(remaining, _TIER3_REFLEX_HARD_CAP_S)
                    if (
                        remaining > _TIER3_REFLEX_HARD_CAP_S + 1.0
                        and primary_budget >= _TIER3_REFLEX_HARD_CAP_S - 0.01
                    ):
                        logger.info(
                            "[CandidateGenerator] Plan Tier3_cap_active: "
                            "primary_budget=%.1fs (hard cap _TIER3_REFLEX_HARD_CAP_S=%.1fs), "
                            "remaining=%.1fs — PLAN primary will sever at cap "
                            "for Manifesto §5 cascade",
                            primary_budget, _TIER3_REFLEX_HARD_CAP_S, remaining,
                        )
                    return await asyncio.wait_for(
                        self._primary.plan(prompt, deadline),
                        timeout=primary_budget,
                    )
            except (Exception, asyncio.CancelledError) as exc:
                logger.warning(
                    "[CandidateGenerator] Primary plan() failed (%s: %s), trying fallback",
                    type(exc).__name__,
                    exc,
                )

        # FALLBACK_ACTIVE, PRIMARY_DEGRADED, or primary plan() just failed
        _sem_t0 = time.monotonic()
        logger.debug(
            "[CandidateGenerator] Plan fallback sem acquire: slots_free=%d/%d",
            self._fallback_sem._value, self._fallback_concurrency,
        )
        # Slice 12F-A — priority-aware acquisition. The plan() entry
        # point doesn't carry context (the prompt was already
        # composed by the orchestrator), so we use the empty-route
        # default which falls through to DEFAULT_PRIORITY (STANDARD
        # bucket — FIFO-equivalent within the bucket). The dominant
        # starvation wedge is on the call() path which DOES have
        # _op_route in scope. plan() acquisitions are rare relative
        # to call() acquisitions in the soak; keeping them at default
        # priority is structurally safe.
        from backend.core.ouroboros.governance.priority_semaphore import (  # noqa: E501
            acquire_priority_aware as _slice12f_acquire,
        )
        async with _slice12f_acquire(self._fallback_sem, ""):
            _sem_wait_s = time.monotonic() - _sem_t0
            _parent_remaining = self._remaining_seconds(deadline)
            _budget_target = max(_parent_remaining, _FALLBACK_MIN_GUARANTEED_S)
            remaining = min(_budget_target, _PLAN_FALLBACK_MAX_TIMEOUT_S)
            if remaining > _parent_remaining + 1.0:
                deadline = datetime.now(tz=timezone.utc) + timedelta(
                    seconds=remaining,
                )
            if _sem_wait_s > 1.0:
                logger.info(
                    "[CandidateGenerator] Plan fallback sem_wait=%.1fs "
                    "(budget=%.1fs)", _sem_wait_s, remaining,
                )
            return await asyncio.wait_for(
                self._fallback.plan(prompt, deadline),
                timeout=remaining,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query_provider_recommendation(self, complexity: str) -> Optional[str]:
        """Query ProviderPerformanceTracker for a routing recommendation.

        Returns a provider name if learning data strongly recommends a
        non-default provider for this complexity class, else None.
        Fault-isolated — returns None on any error.
        """
        try:
            from backend.core.ouroboros.governance.adaptive_learning import (
                ProviderPerformanceTracker,
            )
            candidates = []
            if self._primary is not None:
                candidates.append(getattr(self._primary, "provider_name", "primary"))
            if self._fallback is not None:
                candidates.append(getattr(self._fallback, "provider_name", "fallback"))
            if len(candidates) < 2:
                return None
            tracker = ProviderPerformanceTracker()
            return tracker.recommend_provider(complexity, candidates)
        except Exception:
            return None

    async def _try_primary_then_fallback(
        self,
        context: OperationContext,
        deadline: datetime,
        *,
        model_id: str = "",
    ) -> GenerationResult:
        """Try primary, fall back on any failure.

        Slice 30 — Explicit Parameter Threading & Transport Determinism
        ─────────────────────────────────────────────────────────────────
        ``model_id`` is now an explicit keyword-only parameter threaded
        from the Slice 23 sentinel walker. Forwarded to ``_call_primary``
        so Slice 28 Phase 2's heavy-model 2.5× scalar engages
        deterministically. Legacy callers that don't have a specific
        model_id (pre-Slice-23 fallthroughs) pass nothing → empty
        string → legacy 30s cap path preserved byte-identically.


        Note: In Python 3.9, ``CancelledError`` is a ``BaseException`` (not
        ``Exception``), so we must catch it explicitly to handle
        ``asyncio.wait_for`` cancellation of the primary call.

        Move 2 v6 — Dynamic Provider Fallback. Before paying for a
        primary call we know is statistically likely to fail, consult
        the FSM. If it has classified the primary as in active backoff
        (consecutive transport failures + recovery ETA hasn't elapsed),
        route directly to fallback. The FSM bookkeeping already exists
        (``record_primary_failure`` + ``recovery_eta``); we just need to
        actually consult it at this critical dispatch site. Without
        this, repeated Claude transport failures used to keep retrying
        Claude — and the resulting cascade left ``_active_ops`` empty
        for an hour, idling the soak even after the v5 fixes.

        Cost-safe: dynamic fallback for IMMEDIATE/COMPLEX is Claude →
        DW (~30× cheaper). For STANDARD where DW is primary, fallback
        is Claude (more expensive, but the cost contract guard at
        ClaudeProvider boundary still rejects BG/SPEC).
        """
        # Move 2 v7 — Circuit Breaker pre-call gate. The Claude provider's
        # internal _call_with_backoff retry loop absorbs transport failures
        # within its 3-attempt window. The FSM only sees failures that
        # bubble through this dispatcher, missing exhaustions consumed by
        # the provider's internal retries (e.g. PLAN-phase calls). The
        # cross-cutting breaker sits at the provider boundary and trips
        # on consecutive transport-exhaustion events regardless of which
        # dispatch path triggered them. When OPEN, route directly to
        # fallback. The breaker only gates calls when the primary is the
        # Claude/Anthropic tier — it does not block DW/Tier-0 traffic.
        try:
            _is_claude_primary = (
                self._tier0 is None or self._primary is not self._tier0
            )
        except Exception:  # noqa: BLE001 — defensive
            _is_claude_primary = True
        if _is_claude_primary:
            try:
                from backend.core.ouroboros.governance.claude_circuit_breaker import (
                    get_claude_circuit_breaker,
                    is_enabled as _breaker_enabled,
                    CircuitState,
                )
                if _breaker_enabled():
                    _breaker = get_claude_circuit_breaker()
                    if not _breaker.should_allow_request():
                        _snap = _breaker.snapshot()
                        logger.warning(
                            "[CandidateGenerator] Circuit breaker OPEN — "
                            "routing %s op to fallback (state=%s, "
                            "consecutive_transport_failures=%d, "
                            "total_trips=%d)",
                            getattr(context, "provider_route", "?"),
                            _snap["state"],
                            _snap["consecutive_transport_failures"],
                            _snap["total_trips"],
                        )
                        return await self._call_fallback(context, deadline)
            except Exception as _exc:  # noqa: BLE001
                # Breaker failure must never block dispatch — treat as
                # CLOSED and fall through to normal flow.
                logger.debug(
                    "[CandidateGenerator] Circuit breaker check failed "
                    "(treating as CLOSED): %s", _exc,
                )

        # Dynamic fallback: skip a primary in active backoff.
        if not self.fsm.should_attempt_primary():
            _eta_s = max(0.0, self.fsm.recovery_eta() - time.monotonic())
            _mode_name = (
                self.fsm._failure_mode.name
                if self.fsm._failure_mode is not None else "UNKNOWN"
            )
            # Sovereign Autarky Backoff-Wait (2026-06-20): in DW-only mode there
            # is no fallback — routing to it is a guaranteed fallback_skipped
            # failure. If the sole provider's transient backoff clears within our
            # remaining budget, WAIT it out and re-attempt the primary instead of
            # failing the op. Bounded to ONE wait-and-retry per dispatch: if the
            # re-attempt also fails, control falls to the existing degrade path
            # (record_primary_failure → _call_fallback → clean fallback_skipped),
            # so a genuinely-dead provider self-limits and never loops.
            _autarky_wait = autarky_should_wait_and_retry(
                has_fallback=self._fallback is not None,
                enabled=autarky_backoff_wait_enabled(),
                eta_s=_eta_s,
                remaining_s=self._remaining_seconds(deadline),
                max_wait_s=_autarky_backoff_max_wait_s(),
                margin_s=_autarky_retry_margin_s(),
            )
            if _autarky_wait is not None:
                logger.info(
                    "[CandidateGenerator] Sovereign autarky backoff-wait — sole "
                    "provider %s in %s backoff (consecutive_failures=%d, "
                    "eta=+%.0fs); waiting %.0fs then RE-ATTEMPTING primary "
                    "(remaining_s=%.0f, route=%s) — no absent-fallback failure",
                    self.fsm.primary_name if hasattr(self.fsm, "primary_name")
                    else "primary",
                    _mode_name, self.fsm._consecutive_failures, _eta_s,
                    _autarky_wait, self._remaining_seconds(deadline),
                    getattr(context, "provider_route", "?"),
                )
                try:
                    await asyncio.sleep(_autarky_wait)
                except asyncio.CancelledError:
                    raise
                # Fall through to the primary attempt below (do NOT route to the
                # absent fallback). The backoff window has now elapsed, so
                # should_attempt_primary() is True on the next FSM read.
            else:
                logger.warning(
                    "[CandidateGenerator] Dynamic fallback engaged — primary "
                    "in %s backoff (consecutive_failures=%d, "
                    "recovery_eta=+%.0fs) — routing %s op to fallback "
                    "without re-attempting primary",
                    _mode_name,
                    self.fsm._consecutive_failures,
                    _eta_s,
                    getattr(context, "provider_route", "?"),
                )
                return await self._call_fallback(context, deadline)

        try:
            # Slice 30 — explicit model_id propagation (no ContextVar magic)
            result = await self._call_primary(
                context, deadline, model_id=model_id,
            )
            # Primary succeeded — record recovery if we were in a failure state
            if self.fsm._consecutive_failures > 0:
                self.fsm.record_primary_success()
            return result
        except GovernanceDeadlockError:
            # LR3 terminal: do NOT cascade a deadlock-override failure into the
            # Claude fallback (that would reclassify it as a transient primary
            # failure). Propagate to the orchestrator's terminal catch.
            raise
        except (Exception, asyncio.CancelledError) as exc:
            mode = FailbackStateMachine.classify_exception(exc)
            # Phase 3.1 observability: surface local-tier degradations (memory /
            # latency) distinctly in operator logs. Pure telemetry -- the FSM
            # transition above is authoritative; this never changes control flow.
            try:
                _lv = classify_local_failure(exc)
                if _lv.degrade:
                    logger.info(
                        "[LocalTier] degrade class=%s -> %s (cascading upstream)",
                        getattr(exc, "failure_class", "unknown"),
                        _lv.target_state,
                    )
            except Exception:
                pass
            if mode is not FailureMode.LOCAL_DEFECT:
                # LOCAL_DEFECT gets its own ERROR below, with the traceback.
                # Emitting this line too would file our bug under the same
                # "Primary failed … falling back" heading as every genuine
                # provider blip — which is precisely how it stayed invisible.
                logger.warning(
                    "[CandidateGenerator] Primary failed (mode=%s, %s: %s), "
                    "falling back",
                    mode.name, type(exc).__name__, exc,
                )
            if mode is FailureMode.LOCAL_DEFECT:
                # OUR bug, not the provider's. The op still cascades — the
                # operator's payload is never dropped for a defect on the
                # primary path — but the provider FSM stays untouched, because
                # blaming DoubleWord for a TypeError is how a five-character
                # signature drift becomes a lane-wide outage that bills to
                # Claude until someone reads the code.
                #
                # ERROR, with the traceback, deliberately: the WARNING this
                # replaces was indistinguishable from the ordinary provider
                # noise it sat among, which is why the class survived long
                # enough to be discovered by a unit test rather than by anyone
                # watching production.
                logger.error(
                    "[CandidateGenerator] LOCAL DEFECT on the primary call "
                    "path (%s: %s) — this is a bug in JARVIS, not a provider "
                    "failure. Cascading so the op survives; FSM UNCHANGED so "
                    "the primary lane is not falsely quarantined.",
                    type(exc).__name__, exc, exc_info=True,
                )
            elif mode is FailureMode.CONTENT_FAILURE:
                # Content failure: model produced bad output, but primary infra is healthy.
                # Do NOT penalise the FSM — only count for observability.
                self.fsm.content_failure_count += 1
                logger.info(
                    "[CandidateGenerator] Content failure (count=%d), FSM unchanged",
                    self.fsm.content_failure_count,
                )
            # ONE predicate, consulted here and at the Tier 0 RT site. This
            # used to be an `else` on the CONTENT_FAILURE branch, which meant
            # TEMPORAL_SHED was exempt at the Tier 0 site and penalised HERE —
            # so a temporal shed arriving on this path flipped the primary to
            # FALLBACK_ACTIVE in direct contradiction of its own documented
            # contract ("zero primary penalty ... the NEXT op with a normal
            # budget must still use DW"). Zero recovery params do not save it:
            # `record_primary_failure` increments `_consecutive_failures` and
            # sets FALLBACK_ACTIVE for ANY mode, so exemption has only ever
            # been a property of the callers remembering.
            if mode not in _PRIMARY_INNOCENT_MODES:
                self.fsm.record_primary_failure(mode=mode)
            # CR5 -- header-aware DW recovery. When the primary failed on a DW
            # rate-limit (429) that carried the provider's own Retry-After /
            # x-ratelimit-reset header, signal the failover controller so the
            # SERVING recovery probe can suspend until that exact reset deadline
            # instead of blind polling. Gated default-OFF + fail-soft: never let
            # the recovery signal perturb the existing fallback cascade.
            if mode is FailureMode.RATE_LIMITED:
                try:
                    from .failover_lifecycle import (
                        lifecycle_enabled,
                        header_aware_recovery_enabled,
                        get_failover_controller,
                    )
                    if lifecycle_enabled() and header_aware_recovery_enabled():
                        _reset = None
                        for _layer in _walk_exception_chain(exc):
                            _reset = getattr(_layer, "ratelimit_reset_ts", None)
                            if _reset is not None:
                                break
                        if _reset is not None:
                            get_failover_controller().note_rate_limited(_reset)
                except Exception:  # noqa: BLE001 -- signal must never break the op
                    pass
            return await self._call_fallback(context, deadline)

    async def _slice28_phase3_classify_ttft_failure(
        self,
        *,
        attempted_model_id: str,
        op_id: str,
        elapsed_s: float,
    ) -> None:
        """Slice 28 Phase 3 — Inline Fault Discriminator.

        Fires after a TimeoutError in ``_call_primary`` to classify
        the failure as either:

          * ``context_lag`` — endpoint is alive (probe returns fast);
            THIS prompt+model combo was just too slow. Sentinel walker
            will rotate to the next ranked model and may succeed there.
          * ``infrastructure_outage`` — endpoint is unresponsive (probe
            also times out). Sentinel rotation is unlikely to help
            because every model shares the same upstream tier.

        Pure-observability hook — NEVER raises into the caller, NEVER
        changes return values. The sentinel walker handles rotation
        structurally on the original raise (which still propagates
        normally after this returns). The classification just
        documents WHY the rotation is happening for postmortem
        attribution.

        Probe uses the Slice 27 Phase 2 Aegis-stabilized prompt_only
        lane with a 2-token cap + 5s wall budget. Fires only when
        ``JARVIS_TTFT_FAULT_DISCRIMINATOR_ENABLED=true``.
        """
        probe_timeout = _envf_or_default(
            "JARVIS_TTFT_PROBE_TIMEOUT_S", _TTFT_PROBE_TIMEOUT_S_DEFAULT,
        )
        # Resolve the prompt_only-capable primary surface. Not every
        # primary has prompt_only (e.g., Claude); skip cleanly if absent.
        prompt_only_fn = getattr(self._primary, "prompt_only", None)
        if prompt_only_fn is None:
            logger.info(
                "[Slice28.Phase3] op=%s elapsed=%.1fs model=%s — "
                "primary has no prompt_only lane; classification skipped",
                op_id[:16], elapsed_s, attempted_model_id,
            )
            return

        probe_start = time.monotonic()
        probe_ok = False
        probe_err = ""
        try:
            result_text = await asyncio.wait_for(
                prompt_only_fn(
                    _TTFT_PROBE_PROMPT,
                    model=attempted_model_id or None,
                    caller_id="ttft_fault_discriminator",
                    max_tokens=_TTFT_PROBE_MAX_TOKENS,
                ),
                timeout=probe_timeout,
            )
            probe_ok = bool(result_text and result_text.strip())
        except asyncio.TimeoutError:
            probe_err = f"probe_timeout_{probe_timeout}s"
        except Exception as exc:  # noqa: BLE001 — probe MUST NOT raise
            probe_err = f"probe_exception:{type(exc).__name__}"

        probe_elapsed = time.monotonic() - probe_start
        # Classification:
        #   probe_ok with fast latency → endpoint alive → context_lag
        #   probe failed → endpoint unresponsive → infrastructure_outage
        classification = (
            "context_lag" if probe_ok
            else "infrastructure_outage"
        )
        logger.warning(
            "[Slice28.Phase3] op=%s model=%s primary_elapsed=%.1fs "
            "probe_elapsed=%.2fs probe_ok=%s probe_err=%s "
            "classification=%s — sentinel walker will rotate to next "
            "ranked model (structural rotation already engaged by raise)",
            op_id[:16], attempted_model_id, elapsed_s,
            probe_elapsed, probe_ok, probe_err or "(none)",
            classification,
        )

    @_with_transient_absorb(
        remaining_s=lambda self, context, deadline, **_k: self._remaining_seconds(
            deadline
        ),
        label="_call_primary",
    )
    async def _call_primary(
        self,
        context: OperationContext,
        deadline: datetime,
        *,
        model_id: str = "",
    ) -> GenerationResult:
        """Call primary provider with concurrency and budget-capped deadline.

        Wrapped by ``@with_transient_absorb`` (2026-07-22): a transient DW round
        failure on the standard/immediate path — a watchdog fast-abort,
        ``upstream_error``, 5xx, or 429-with-Retry-After — is absorbed by an
        exponential-backoff retry of the whole primary attempt (budget-aware),
        so big-file ReAct rounds self-heal instead of failing the round. The
        decorator is a no-op when disabled / on a non-transient error.

        The primary gets at most ``_PRIMARY_BUDGET_FRACTION`` of the
        remaining time, guaranteeing ``_FALLBACK_MIN_RESERVE_S`` for the
        fallback provider if the primary hangs until timeout.

        Slice 30 — Explicit Parameter Threading & Transport Determinism
        ─────────────────────────────────────────────────────────────────
        ``model_id`` is now an explicit keyword-only parameter threaded
        from the Slice 23 sentinel walker → ``_try_primary_then_fallback``
        → here. This eliminates the v23 wiring gap where Slice 28's
        ContextVar-based model_id resolution silently returned empty
        across async/semaphore task boundaries, causing the heavy-model
        2.5× scalar to never engage in production (12 EXHAUSTION events
        across v20/v21/v23 all firing at the static 30s
        _PRIMARY_MAX_TIMEOUT_S cap instead of the adaptive 75s budget).

        Legacy callers that don't have a specific model_id (pre-Slice-23
        dispatch paths, IMMEDIATE route fallthrough, etc.) pass nothing
        → empty string → _compute_primary_budget skips the heavy scalar
        → legacy 30s cap behavior preserved byte-identically.
        """
        _primary_sem_t0 = time.monotonic()
        _primary_phase_hint = getattr(getattr(context, "phase", None), "name", "?")
        logger.info(
            "[CandidateGenerator] Primary sem acquire: slots_free=%d "
            "route=%s phase=%s op=%s model_id=%s",
            self._primary_sem._value,
            getattr(context, "provider_route", "?"),
            _primary_phase_hint,
            getattr(context, "op_id", "?")[:16],
            model_id or "(unspecified)",
        )
        async with self._primary_sem:
            _primary_sem_wait_s = time.monotonic() - _primary_sem_t0
            remaining = self._remaining_seconds(deadline)
            # Slice 30 — explicit model_id parameter (no ContextVar magic).
            # Slice 28 Phase 2's heavy-model 2.5× scalar now engages
            # deterministically when the sentinel walker passes a heavy
            # model_id. Empty model_id from legacy callers → legacy 30s
            # cap path (byte-identical to pre-Slice-28).
            # Slice 43 — if this op will force-batch (Slice 36/41), compute a
            # batch-appropriate budget so the outer wait_for doesn't sever the
            # async batch poll at the 30s RT reflex cap. The force-batch
            # decision is owned by the provider; we consult the same pure
            # predicate. NEVER raises → legacy budget on any failure.
            try:
                from backend.core.ouroboros.governance.doubleword_provider import (
                    _slice36_should_force_batch,
                )
                _force_batch = _slice36_should_force_batch(context)
            except Exception:  # noqa: BLE001 — defensive, legacy budget
                _force_batch = False
            # Sovereign Transport Profiler Matrix (2026-06-20) — learn-then-detach.
            # The transport-hedge (default-on) makes _slice36_should_force_batch
            # return False, so a batch-ONLY model (RT yields done_before_content)
            # gets the RT/autarky budget (180s) even though only the batch arm can
            # win → the batch poll is strangled mid-flight (the live-soak TIMEOUT
            # root). When the immortal profile knows this model is batch-only, force
            # the batch budget (reuses the EXISTING force_batch branch in
            # _compute_primary_budget → batch cap) AND stamp the op
            # ASYNC_BATCH_PAYLOAD so the Zero-Shot quarantine grants it immunity and
            # the park layer actively detaches it. Gated + fail-soft → legacy on any
            # failure (byte-identical when the profile is empty/off).
            try:
                if model_id:
                    from backend.core.ouroboros.governance.dw_transport_profile import (  # noqa: E501
                        get_transport_profile as _get_tp,
                    )
                    if _get_tp().is_batch_only(model_id):
                        _force_batch = True
                        # NB: OperationContext is FROZEN — we deliberately do NOT
                        # stamp a ctx tag (it would raise FrozenInstanceError, and
                        # consumers can't rely on it). Every consumer of "is this op
                        # batch-bound?" checks the immortal profile directly with the
                        # resolved model: the budget here (force_batch), the Zero-Shot
                        # ban-immunity seam below (is_batch_only(_dp_model)), and the
                        # park gate (generate_park_wrapper._resolve_async_batch_payload).
            except Exception:  # noqa: BLE001 — defensive, legacy budget
                pass
            # M3 — Transport Circuit Breaker at the primary dynamic transport-selection
            # point. A batch choice arriving via the dynamic router (_slice36 or
            # is_batch_only profile) is also rotated when the batch lane is OPEN.
            # When the breaker is disabled (default) or the lane is CLOSED this is a
            # zero-cost no-op (returns "batch" unchanged -> _force_batch stays True).
            if _force_batch:
                try:
                    _m3_chosen = _breaker_select_transport("batch")
                    if _m3_chosen == "realtime":
                        logger.info(
                            "[A3-Breaker/M3] batch lane OPEN at dynamic-select -- "
                            "rotating to realtime for op=%s model=%s",
                            getattr(context, "op_id", "?")[:16], model_id or "?",
                        )
                        _force_batch = False
                except Exception:  # noqa: BLE001 -- breaker consult never blocks dispatch
                    pass
            # Slice 225 Phase 2 — Sovereign DW Autarky. Read the Claude fallback
            # breaker (read-only, no probe side effect — same _claude_breaker_open
            # predicate the Slice 127 P2.1 IMMEDIATE reroute uses). When the
            # fallback lane is OPEN/HALF_OPEN (incl. terminal_quota / out-of-
            # credits), there's no live lane to sever DW into — give DW the full
            # runway instead of the 30s/75s reflex cap. Gated default-TRUE;
            # OFF (or breaker CLOSED) is the byte-identical legacy cascade.
            _fallback_dead = False
            _autarky_reason = ""  # "structural" (expected) | "breaker_open" (abnormal)
            # Is the seat about to be called the LOCAL engine? The reflex cap
            # is DW-shaped -- 30s tuned for a metered cloud API whose fallback
            # is a second cloud API. A locally-served model has neither
            # property: its prefill on a 32K prompt alone can exceed the cap,
            # and its budget is already governed by the streaming watchdog
            # and the profiler's absolute ceiling.
            try:
                _local_seat = (
                    self._jprime is not None and self._primary is self._jprime
                )
            except Exception:  # noqa: BLE001 — defensive
                _local_seat = False
            if _dw_autarky_enabled():
                # A CONFIG-disabled Claude (JARVIS_PROVIDER_CLAUDE_DISABLED) is the
                # deadest fallback of all — never constructed — yet it leaves the
                # circuit breaker CLOSED, so the breaker-state check below misses it.
                # Check it first so the sole-lane DW gets the full runway instead of
                # the reflex cap (the live-soak TIMEOUT root, 2026-06-20).
                #
                # The FREE LANE is the same class of dead: local engine
                # configured, no DW key, no Anthropic key -- nothing to
                # reserve runway for. Soak bt-2026-09-02-203607 ran exactly
                # that topology and still reserved 9,852s of a 9,882s budget
                # for a Claude call that `session_budget_preflight_refused`
                # and `anthropic_not_installed` could never make, severing
                # every local primary at 30s. `_free_lane_active` is the
                # existing predicate for "no paid lane exists".
                _fallback_dead = _claude_config_disabled() or _free_lane_active()
                if _fallback_dead:
                    _autarky_reason = "structural"
                else:
                    try:
                        from backend.core.ouroboros.governance.doubleword_provider import (
                            _claude_breaker_open as _autarky_breaker_open,
                        )
                        _fallback_dead = _autarky_breaker_open()
                        if _fallback_dead:
                            _autarky_reason = "breaker_open"
                    except Exception:  # noqa: BLE001 — fail-closed to legacy cascade
                        _fallback_dead = False
            primary_budget = self._compute_primary_budget(
                remaining, model_id=model_id, force_batch=_force_batch,
                fallback_dead=_fallback_dead, local_seat=_local_seat,
                # T5 -- per-op deadline dilation seam: a LANE-COLLAPSED op (both
                # transport lanes timed out) carries a recorded dilation hop, so
                # this re-attempt's budget is scaled up (bounded). No hops / gate
                # off -> byte-identical legacy budget.
                op_id=(
                    getattr(context, "op_id", "")
                    or getattr(context, "operation_id", "")
                    or ""
                ),
            )
            if _fallback_dead and primary_budget > _PRIMARY_MAX_TIMEOUT_S:
                # Severity matches reality: STRUCTURAL autarky (operator-attested
                # JARVIS_PROVIDER_CLAUDE_DISABLED) is the intended steady state →
                # INFO (observable, not alarming, no warning-count inflation). An
                # actual breaker OPEN (economic/transport failures) is a real
                # fallback-lane degradation → WARNING. The message now states the
                # ACCURATE reason instead of always claiming "breaker OPEN".
                _structural = _autarky_reason == "structural"
                _emit = logger.info if _structural else logger.warning
                _emit(
                    "[CandidateGenerator] ⚡ DW AUTARKY ENGAGED: Claude fallback "
                    "%s — granting DW the full %.1fs budget (vs %.1fs reflex cap), "
                    "no dead-lane handoff. route=%s op=%s model=%s",
                    "structurally disabled (autarky)" if _structural
                    else "breaker OPEN (degraded)",
                    primary_budget, _PRIMARY_MAX_TIMEOUT_S,
                    getattr(context, "provider_route", "?"),
                    getattr(context, "op_id", "?")[:16],
                    model_id or "(unspecified)",
                )
            # Slice 34 Phase 2 — dispatch profiler (default OFF; zero
            # overhead when disabled). Records the sem-wait + budget
            # stages into the per-op summary; STAGE_PROVIDER_GENERATE
            # below brackets the actual provider call so we can see
            # how much time is spent IN the orchestrator overhead vs
            # the provider's own dispatch path.
            from backend.core.ouroboros.telemetry import (
                dispatch_profiler as _dp_mod,
            )
            _dp_op_id = getattr(context, "op_id", "?") or "?"
            _dp_route = getattr(context, "provider_route", "?") or "?"
            _dp_model = model_id or "(unspecified)"
            if _dp_mod.is_enabled():
                _dp_key = _dp_mod._active_key(_dp_op_id, _dp_model)
                _dp_summary = _dp_mod.OpDispatchSummary(
                    op_id=_dp_op_id, model_id=_dp_model,
                    route=_dp_route, started_unix=time.time(),
                )
                with _dp_mod._active_ops_lock:
                    _dp_mod._active_ops[_dp_key] = _dp_summary
                # Record the already-measured sem-wait as a stage.
                _dp_summary.stages.append(_dp_mod.StageRecord(
                    stage_name="STAGE_SEM_WAIT",
                    duration_ms=_primary_sem_wait_s * 1000.0,
                ))
                # And a synthetic ~0 stage for the trivial budget
                # computation (kept for shape consistency in the
                # per-op summary — actual sub-ms math is recorded
                # below via the dispatch_stage wrap if anyone ever
                # makes _compute_primary_budget heavy).
                _dp_summary.stages.append(_dp_mod.StageRecord(
                    stage_name="STAGE_BUDGET_COMPUTATION",
                    duration_ms=0.0,
                ))
            # Tier 3 Reflex observability (Manifesto §5): log at INFO when
            # the hard cap is the binding constraint (not the fraction or
            # the fallback-reserve). Operators can grep for
            # "Tier3_cap_active" to see sessions where the aggressive
            # circuit breaker is forcing fast fallback cascades.
            _tier3_cap_active = (
                primary_budget >= _PRIMARY_MAX_TIMEOUT_S - 0.01
                and remaining > _PRIMARY_MAX_TIMEOUT_S + _FALLBACK_MIN_RESERVE_S
            )
            if _tier3_cap_active:
                logger.info(
                    "[CandidateGenerator] Tier3_cap_active: primary_budget=%.1fs "
                    "(hard cap _PRIMARY_MAX_TIMEOUT_S=%.1fs), remaining=%.1fs "
                    "fallback_reserve=%.1fs route=%s phase=%s op=%s — "
                    "primary will sever at budget expiry for Manifesto §5 cascade",
                    primary_budget, _PRIMARY_MAX_TIMEOUT_S, remaining,
                    remaining - primary_budget,
                    getattr(context, "provider_route", "?"),
                    _primary_phase_hint,
                    getattr(context, "op_id", "?")[:16],
                )
            else:
                logger.debug(
                    "[CandidateGenerator] Primary budget: %.1fs of %.1fs remaining "
                    "(fallback reserve: %.1fs)",
                    primary_budget, remaining, remaining - primary_budget,
                )
            try:
                # W3(7) Slice 2 — race against ambient cancel token (if any).
                # `current_cancel_token()` reads the ContextVar set by
                # `dispatch_pipeline`; None outside dispatcher (unit tests,
                # pre-W3(7) callers) → falls through to plain wait_for.
                from backend.core.ouroboros.governance.cancel_token import (
                    current_cancel_token as _curr_cancel_token,
                    race_or_wait_for as _race_or_wait_for,
                )
                # Slice 34 Phase 2 — Stage 3: STAGE_PROVIDER_GENERATE
                # brackets the entire provider call so we can see how
                # much wall-time is spent in the provider's own dispatch
                # path (Aegis auth + lease + HTTP POST + response parse).
                # Profiler is fail-closed default-OFF — zero overhead
                # when JARVIS_DISPATCH_PROFILER_ENABLED is unset.
                from backend.core.ouroboros.telemetry.dispatch_profiler import (
                    dispatch_stage as _dp_stage,
                )
                _dp_provider_outcome = "ok"
                _dp_provider_err = ""
                _dp_provider_t0 = time.monotonic()
                try:
                    _pri_result = await _race_or_wait_for(
                        self._primary.generate(context, deadline),
                        timeout=primary_budget,
                        cancel_token=_curr_cancel_token(),
                    )
                except asyncio.CancelledError:
                    _dp_provider_outcome = "cancelled"
                    raise
                except Exception as _dp_exc:  # noqa: BLE001
                    _dp_provider_outcome = "error"
                    _dp_provider_err = type(_dp_exc).__name__
                    # Zero-Shot timeout quarantine (Sovereign Zero-Shot & Decay
                    # Matrix). This is the ONLY seam where the raw, unwrapped
                    # provider exception is visible WITH the model_id in scope:
                    # the sentinel-dispatch FSM downstream re-wraps a primary
                    # TimeoutError into RuntimeError('all_providers_exhausted')
                    # before it reaches the outer dispatch handler, so a hook
                    # there never sees TimeoutError. A 180s provider timeout is
                    # unambiguous → 1-strike cold-storage (bypass the n>=3 σ
                    # window) so the model is skipped on the very next op rather
                    # than tainting two more soaks. Fail-soft, gated, never
                    # raises — must not perturb the cascade.
                    try:
                        from backend.core.ouroboros.governance.dw_fault_taxonomy import (  # noqa: E501
                            is_generation_timeout as _zs_is_timeout,
                        )
                        # Ban-immunity (Transport Profiler Matrix): a batch-only
                        # model times out only because the batch poll is async-slow,
                        # NOT because the model is dead — banning it would blacklist
                        # the fleet's best diff-capable models for transport latency.
                        # Checked DIRECTLY against the immortal profile with the
                        # resolved model (OperationContext is frozen — no ctx tag).
                        _zs_batch_immune = False
                        try:
                            from backend.core.ouroboros.governance.dw_transport_profile import (  # noqa: E501
                                get_transport_profile as _zs_get_tp,
                            )
                            _zs_batch_immune = _zs_get_tp().is_batch_only(_dp_model)
                        except Exception:  # noqa: BLE001
                            _zs_batch_immune = False
                        if (
                            not _zs_batch_immune
                            and (
                                isinstance(_dp_exc, asyncio.TimeoutError)
                                or _zs_is_timeout(_dp_exc)
                            )
                        ):
                            from backend.core.ouroboros.governance.dw_discovery_runner import (  # noqa: E501
                                get_ttft_observer as _zs_get_obs,
                            )
                            _zs_obs = _zs_get_obs()
                            if _zs_obs is not None:
                                _zs_obs.record_timeout(
                                    _dp_model, op_id=str(_dp_op_id)[:24],
                                )
                    except Exception:  # noqa: BLE001 — never raise from quarantine
                        pass
                    raise
                finally:
                    # Slice 34 Phase 2 — record STAGE_PROVIDER_GENERATE
                    # + emit the per-op summary. Fail-closed if profiler
                    # is disabled or accumulator was never created.
                    try:
                        _dp_provider_ms = (
                            time.monotonic() - _dp_provider_t0
                        ) * 1000.0
                        if _dp_mod.is_enabled():
                            _dp_key2 = _dp_mod._active_key(_dp_op_id, _dp_model)
                            with _dp_mod._active_ops_lock:
                                _dp_summary2 = _dp_mod._active_ops.pop(
                                    _dp_key2, None,
                                )
                            if _dp_summary2 is not None:
                                _dp_summary2.stages.append(
                                    _dp_mod.StageRecord(
                                        stage_name="STAGE_PROVIDER_GENERATE",
                                        duration_ms=_dp_provider_ms,
                                        outcome=_dp_provider_outcome,
                                        error_class=_dp_provider_err,
                                    )
                                )
                                _dp_summary2.total_duration_ms = sum(
                                    s.duration_ms for s in _dp_summary2.stages
                                )
                                _dp_summary2.outcome = _dp_provider_outcome
                                _dp_summary2.error_class = _dp_provider_err
                                with _dp_mod._recent_summaries_lock:
                                    _dp_mod._recent_summaries.append(_dp_summary2)
                                logger.info(
                                    "[DispatchProfiler] op_summary %s",
                                    _dp_summary2.to_log_kv(),
                                )
                    except Exception:  # noqa: BLE001 — never raise from profiler
                        pass
                logger.info(
                    "[CandidateGenerator] Primary sem release: "
                    "hold=%.1fs sem_wait=%.1fs route=%s phase=%s op=%s outcome=ok",
                    time.monotonic() - _primary_sem_t0, _primary_sem_wait_s,
                    getattr(context, "provider_route", "?"),
                    _primary_phase_hint,
                    getattr(context, "op_id", "?")[:16],
                )
                return _pri_result
            except (Exception, asyncio.CancelledError) as _exc:
                logger.info(
                    "[CandidateGenerator] Primary sem release: "
                    "hold=%.1fs sem_wait=%.1fs route=%s phase=%s op=%s outcome=fail",
                    time.monotonic() - _primary_sem_t0, _primary_sem_wait_s,
                    getattr(context, "provider_route", "?"),
                    _primary_phase_hint,
                    getattr(context, "op_id", "?")[:16],
                )
                logger.info(
                    "[CancelAttribution] %s",
                    _attribute_cancel(
                        _exc,
                        label="_call_primary",
                        op_id=getattr(context, "op_id", "?"),
                        elapsed_s=time.monotonic() - _primary_sem_t0,
                        remaining_s=self._remaining_seconds(deadline),
                    ),
                )
                # Slice 28 Phase 3 — Inline Fault Discriminator
                # ────────────────────────────────────────────────
                # On TimeoutError specifically, fire a lightweight
                # 2-token probe via the primary's prompt_only lane
                # (now Aegis-stabilized via Slice 27 Phase 2) to
                # discriminate between:
                #   * context_lag — endpoint alive, THIS prompt+model
                #     combination is just slow (probe completes fast)
                #   * infrastructure_outage — endpoint not responding
                #     (probe also times out)
                # The sentinel walker ALREADY rotates to the next
                # model in ranked_models on any raise from
                # _call_primary, so Phase 3 doesn't need to add
                # rotation — it adds the CLASSIFICATION SIGNAL so
                # postmortem analysis can attribute the rotation
                # reason structurally. Probe is bounded to 5s; on
                # outage, the cost is small and the diagnostic is
                # invaluable. Env-gated default-off; graduate after
                # v22 proves the probe yields actionable signal.
                if (
                    isinstance(_exc, asyncio.TimeoutError)
                    and _envb("JARVIS_TTFT_FAULT_DISCRIMINATOR_ENABLED", False)
                ):
                    # Slice 30 — use explicit model_id param (was ContextVar)
                    await self._slice28_phase3_classify_ttft_failure(
                        attempted_model_id=model_id,
                        op_id=getattr(context, "op_id", "?"),
                        elapsed_s=time.monotonic() - _primary_sem_t0,
                    )
                raise

    # Hard ceiling for fallback provider — fail fast when unreachable
    # rather than burning the entire pipeline budget (Manifesto §6: Iron Gate).
    # Raised from 60s to 120s after bt-2026-04-11-085020 diagnosed tool_round
    # full_content patches legitimately needing 60-90s of stream time. IMMEDIATE
    # route also funnels through this cap, and a 60s cap was cutting mid-stream
    # healthy generation (23KB received at 365 bytes/s — normal Claude rate).
    _FALLBACK_MAX_TIMEOUT_S: float = float(
        os.environ.get("JARVIS_FALLBACK_MAX_TIMEOUT_S", "120.0")
    )

    # Route-aware ceiling for complex-route generate. Session
    # bt-2026-04-15-065523 (Session F, 2026-04-14) diagnosed a
    # complex-route retry synthesis hitting the 120s cap by exactly 2
    # seconds: elapsed=122.1s, fallback_err_class=CancelledError,
    # all_providers_exhausted with 131s of nominal generation budget
    # still remaining. Complex ops under ledger enforcement legitimately
    # need wider synthesis windows because their tool-result prompts
    # exceed 40KB (44104 chars observed in Session F attempt 2) and
    # Claude needs 150-180s to produce a coherent multi-file patch.
    # 120s remains the default for all other routes; complex gets 180s.
    # Env-tunable so ops can tune without a code change — the default
    # 180.0 is the post-Session-F calibration.
    _FALLBACK_MAX_TIMEOUT_COMPLEX_S: float = float(
        os.environ.get("JARVIS_FALLBACK_MAX_TIMEOUT_COMPLEX_S", "180.0")
    )

    # ── Synthesis reserve for read-only BG subagent fan-out (Session 5) ──
    # Session 5 (bt-2026-04-18-035817) proved the graduation signal: three
    # parallel subagents dispatched, 80 findings returned, all with Iron
    # Gate diversity=3. But the parent Claude synthesis round died with
    # TimeoutError because the BG fallback cap (120s) was sized for a
    # single-shot Claude completion, not "Claude fans out → 3 subagents
    # consume 135s → Claude synthesizes the findings". Per Derek's
    # 2026-04-17 directive, the cap for read-only BG ops must dynamically
    # expand to account for subagent wall-clock PLUS a hard synthesis
    # reserve. The formula is:
    #
    #     _max_cap = base_cap  (=_FALLBACK_MAX_TIMEOUT_S, 120s default)
    #              + MAX_PARALLEL_SCOPES * PRIMARY_PROVIDER_TIMEOUT_S
    #              + _BG_READONLY_SYNTHESIS_RESERVE_S
    #
    # With Phase 1 constants (MAX_PARALLEL_SCOPES=3,
    # PRIMARY_PROVIDER_TIMEOUT_S=90) and the mandated 90s synthesis
    # reserve, this evaluates to 480s — about 4× the mutating-BG cap,
    # but strictly bounded by what the actual wall-clock needs for a
    # 3-subagent cartography op. Env-tunable so operators can retune
    # after graduation data accumulates.
    # Default sized from Session-12 empirical data (bt-2026-04-18-055042).
    # Session 11 synthesized 80 findings in 472s (8s under 480s cap).
    # Session 12 with 108 findings took 491.93s — 11.93s over. Subagent
    # finding counts are model-driven (exploration depth varies per
    # provider + cache state), so a fixed 90s reserve was too tight for
    # the high-yield end of the distribution. 180s absorbs another ~40%
    # finding-count drift before the cap bites.
    _BG_READONLY_SYNTHESIS_RESERVE_S: float = float(
        os.environ.get("JARVIS_BG_READONLY_SYNTHESIS_RESERVE_S", "180.0")
    )

    def _fallback_is_claude(self) -> bool:
        """Slice 238 — True iff the configured fallback is the Claude lane (so the
        Claude economic breaker is the right health signal to gate it). Reads the
        fallback's ``provider_name`` (e.g. ``claude-api``); a non-Claude fallback
        (e.g. Prime) returns False so the Claude breaker never suppresses it.
        NEVER raises → fail-soft to False (legacy: don't suppress)."""
        try:
            if self._fallback is None:
                return False
            name = (getattr(self._fallback, "provider_name", "") or "").strip().lower()
            return "claude" in name
        except Exception:  # noqa: BLE001
            return False

    async def _call_fallback(
        self,
        context: OperationContext,
        deadline: datetime,
    ) -> GenerationResult:
        """Call fallback provider with concurrency and deadline enforcement.

        Budget computation happens AFTER acquiring ``_fallback_sem`` so that
        time spent queued behind other ops doesn't silently zero out
        ``_parent_remaining``.  The post-acquire refresh guarantees at least
        ``_FALLBACK_MIN_GUARANTEED_S`` regardless of how long the wait was.

        The orchestrator's outer ``wait_for(_gen_timeout + _OUTER_GATE_GRACE_S)``
        is still the absolute Iron Gate — grace raised from 5s to 15s after
        bt-2026-04-12-061609 diagnosed 129s Claude streams cut by 125s gate.
        """
        # ──────────────────────────────────────────────────────────────
        # Slice 19b (2026-05-26) — fallback=None semantic correction
        #
        # Pre-Slice-19b: when self._fallback is None (e.g., Slice 19a
        # JARVIS_PROVIDER_CLAUDE_DISABLED=true), _call_fallback fell
        # through to the semaphore acquire + provider call. The
        # ``self._fallback.generate(...)`` would raise AttributeError,
        # the exception handler at line ~4731 classified it as
        # ``fallback_failed`` cause, and ExhaustionWatcher incremented
        # the consecutive counter. 3 consecutive ops with no-fallback
        # cascade → hibernation, even though DW (primary) was healthy.
        #
        # bt-2026-05-26-180129 (PURE-DW v14 soak) proved this:
        # DW completed a 265s, 23-tool-call, 76K-token Venom loop on
        # the SWE-Bench Ansible op and returned 0 candidates (model
        # judgment). The orchestrator wanted to retry via fallback,
        # fallback was None (Slice 19a intentional), instant
        # "fallback_failed", 3 consecutive → hibernation cycle 1.
        #
        # Fix: emit a DISTINCT cause prefix ``fallback_skipped:`` for
        # the "no fallback configured" case (vs ``fallback_failed:``
        # for genuine fallback failures). ExhaustionWatcher will
        # filter ``fallback_skipped:`` out of the consecutive count
        # (separate edit in provider_exhaustion_watcher.py). Hibernation
        # stays reserved for genuine provider distress, not for the
        # operator-attested DW-only mode.
        # ──────────────────────────────────────────────────────────────
        if self._fallback is None:
            logger.info(
                "[CandidateGenerator] Slice 19b: fallback=None "
                "(provider intentionally absent, e.g., Slice 19a "
                "JARVIS_PROVIDER_CLAUDE_DISABLED) — raising "
                "fallback_skipped sentinel (NOT counted toward "
                "ExhaustionWatcher hibernation threshold)"
            )
            # Multi-Vector Awaken (CR2): cloud primary exhausted with NO cloud
            # fallback -> wake the J-Prime golden-image fallback (budget/credit
            # exhaustion is a valid awaken vector, not just a data-plane outage).
            try:
                from .failover_lifecycle import (
                    lifecycle_enabled, budget_awaken_enabled, get_failover_controller,
                )
                if lifecycle_enabled() and budget_awaken_enabled():
                    get_failover_controller().note_budget_exhausted()
            except Exception:  # noqa: BLE001 -- never let the signal break the exhaustion path
                pass
            # Dynamic state-driven routing (the live A1-on-32B seam): cloud primary
            # exhausted + NO cloud fallback, but the failover FSM may have an awakened
            # J-Prime node SERVING. Discover its endpoint and route THIS op there
            # rather than dying -- so generation actually happens on the 32B instead
            # of failing pre-generation. Fail-soft: any miss falls through to the
            # terminal raise below (the op is never silently lost).
            try:
                _jp_ep = await self._discover_jprime_endpoint()
                if _jp_ep:
                    logger.info(
                        "[CandidateGenerator] DW exhausted + no fallback, but "
                        "J-Prime endpoint=%s discovered -> routing GENERATE to the "
                        "awakened 32B (op=%s)", _jp_ep,
                        (getattr(context, "op_id", "") or "?")[:16],
                    )
                    _jp_result = await self._failover_local_dispatch(
                        context, deadline, _jp_ep,
                    )
                    if _jp_result is not None and len(
                        getattr(_jp_result, "candidates", ()) or ()
                    ) > 0:
                        return _jp_result
            except Exception as _jp_exc:  # noqa: BLE001 -- never break the exhaustion path
                logger.debug(
                    "[CandidateGenerator] J-Prime exhaustion-reroute fail-soft "
                    "err=%r", _jp_exc,
                )
            self._raise_exhausted(
                "fallback_skipped:no_fallback_configured",
                context=context,
                deadline=deadline,
                fallback_state="absent_by_configuration",
            )

        # Isolation override: if the op's route is listed in
        # JARVIS_DISABLE_CLAUDE_FALLBACK_ROUTES, skip the fallback entirely
        # and raise through the existing exhaustion path. Used by the Qwen
        # 397B benchmark to collect raw DW telemetry without Claude masking.
        _op_route = (getattr(context, "provider_route", "") or "").strip().lower()
        if _fallback_disabled_for_route(_op_route):
            logger.info(
                "[CandidateGenerator] Fallback disabled by env for route=%s "
                "(%s) — raising fallback_disabled_by_env sentinel",
                _op_route, _DISABLE_FALLBACK_ROUTES_ENV,
            )
            self._raise_exhausted(
                f"fallback_disabled_by_env:{_op_route}",
                context=context,
                deadline=deadline,
                disabled_routes=os.environ.get(_DISABLE_FALLBACK_ROUTES_ENV, ""),
            )

        # Slice 238 — cascade breaker consult (CENTRAL seam). The s237 soak proved
        # the cascade-to-dead-Claude poison (BadRequestError 400 → terminal_quota →
        # cooldown cycle) reaches Claude from EVERY _call_fallback caller, not just
        # the sentinel cascade_to_claude path. Guard it here, where all callers
        # converge: when the fallback IS the Claude lane AND the economic breaker
        # is OPEN (read-only _claude_breaker_open — same source-of-truth the
        # primary lane respects, no probe side-effect), do NOT call the known-dead
        # lane. Raise the EXISTING fallback_skipped sentinel (Slice 19b — NOT
        # counted toward ExhaustionWatcher hibernation) so the op degrades cleanly
        # instead of burning a 400 and poisoning the consecutive-failure counter.
        # Breaker CLOSED → byte-identical (a funded Claude fallback is used).
        if cascade_breaker_consult_enabled() and self._fallback_is_claude():
            _claude_lane_open = False
            try:
                from backend.core.ouroboros.governance.doubleword_provider import (
                    _claude_breaker_open as _cf_breaker_open,
                )
                _claude_lane_open = _cf_breaker_open()
            except Exception:  # noqa: BLE001 — advisory; never block dispatch
                _claude_lane_open = False
            if _claude_lane_open:
                logger.warning(
                    "[CandidateGenerator] Slice238 fallback SUPPRESSED (central): "
                    "Claude lane breaker OPEN (economic/transport) — skipping the "
                    "known-dead Claude fallback (no terminal_quota poison); "
                    "raising fallback_skipped so the op degrades cleanly "
                    "(route=%s)", _op_route or "?",
                )
                self._raise_exhausted(
                    "fallback_skipped:claude_breaker_open",
                    context=context,
                    deadline=deadline,
                    fallback_state="economic_open",
                )

        _pre_sem_remaining = self._remaining_seconds(deadline)
        _sem_t0 = time.monotonic()
        _phase_hint = getattr(getattr(context, "phase", None), "name", "?")

        # Defect #4 Slice B (2026-05-03) — pre-fallback budget short-
        # circuit. Soak v5 saw 3 EXHAUSTION events with remaining_s=0.0
        # and fallback_err_class=CancelledError -- ops were entering
        # _call_fallback with insufficient budget, the call attempt
        # was cancelled mid-flight, and the resulting CancelledError
        # was relabeled as "fallback_failed". Wasted CPU + provider
        # call attempt + log noise + the unhandled-exception cascade.
        #
        # Fix: detect the "deadline already exhausted" pre-condition
        # and raise a clean cause sentinel instead of attempting an
        # invariably-doomed call. Env-tunable floor protects against
        # over-aggressive shedding (e.g., complex routes with legit
        # 4-5s remaining might still complete in fast paths).
        try:
            raw_min_viable = os.environ.get(
                "JARVIS_FALLBACK_MIN_VIABLE_BUDGET_S", "",
            ).strip()
            min_viable_s = max(
                1.0, min(60.0, float(raw_min_viable) if raw_min_viable else 5.0),
            )
        except (TypeError, ValueError):
            min_viable_s = 5.0
        if _pre_sem_remaining <= min_viable_s:
            logger.info(
                "[CandidateGenerator] Pre-fallback short-circuit: "
                "remaining=%.2fs <= min_viable=%.2fs route=%s "
                "(Defect #4 Slice B fix avoids attempting a doomed "
                "fallback call that would CancelledError mid-flight)",
                _pre_sem_remaining, min_viable_s, _op_route,
            )
            self._raise_exhausted(
                "deadline_exhausted_pre_fallback",
                context=context,
                deadline=deadline,
                pre_sem_remaining_s=round(_pre_sem_remaining, 2),
                min_viable_s=round(min_viable_s, 2),
                phase=_phase_hint,
                route=_op_route,
            )

        # Route-aware fallback ceiling: complex routes get a wider
        # synthesis window (180s) because their tool-result prompts are
        # legitimately larger and their multi-file patches take longer
        # to generate coherently. Non-complex routes keep the 120s cap.
        # _op_route was already computed earlier in this method for the
        # fallback-disabled-by-env check, so we reuse the lowercased value.
        #
        # Read-only BG subagent fan-out override (Session 5 graduation
        # directive, Derek 2026-04-17): when the op is read-only AND on
        # BG route, dynamically extend the cap to account for parallel
        # subagent wall-clock plus a 90s synthesis reserve. The three
        # parallel ExploreAgents consume up to
        # MAX_PARALLEL_SCOPES * PRIMARY_PROVIDER_TIMEOUT_S seconds of
        # wall-clock before the parent Claude begins synthesizing the
        # rolled-up findings — charging the parent's 120s cap for that
        # wait is the arithmetic that killed Session 5 at 134.56s.
        _is_read_only = bool(getattr(context, "is_read_only", False))
        if _is_read_only and _op_route == "background":
            # Lazy import to avoid a new top-level dependency on
            # subagent_contracts — this module is imported eagerly
            # at provider boot, subagent_contracts is imported by
            # the orchestrator later.
            try:
                from backend.core.ouroboros.governance.subagent_contracts import (
                    MAX_PARALLEL_SCOPES,
                    PRIMARY_PROVIDER_TIMEOUT_S,
                )
                _subagent_wallclock_budget_s = (
                    MAX_PARALLEL_SCOPES * PRIMARY_PROVIDER_TIMEOUT_S
                )
            except Exception:
                # Defensive fallback — hardcode the current Phase 1
                # constants so the cap still extends meaningfully if
                # the import fails for any reason.
                _subagent_wallclock_budget_s = 3 * 90  # = 270s
            _max_cap = (
                self._FALLBACK_MAX_TIMEOUT_S
                + _subagent_wallclock_budget_s
                + self._BG_READONLY_SYNTHESIS_RESERVE_S
            )
        elif _op_route == "complex":
            _max_cap = self._FALLBACK_MAX_TIMEOUT_COMPLEX_S
        else:
            _max_cap = self._FALLBACK_MAX_TIMEOUT_S

        # Task #88b — thinking-aware outer-budget widening (2026-05-13).
        #
        # v14-rev6 graduation soak proved: Task #88's inner rupture
        # widening (120s -> 360s for thinking-enabled calls) is correct
        # but insufficient — the OUTER asyncio.wait_for budget computed
        # from _max_cap fires FIRST and kills the Claude stream before
        # the inner rupture matters.  Log evidence:
        # ``elapsed=290.0s budget=218.7s first_token=NEVER thinking=on``.
        # Direct-host streaming probes confirmed Claude responds in
        # seconds with thinking events; the harness's outer budget was
        # the load-bearing constraint.
        #
        # Single policy with Task #88: outer >= inner for thinking-on.
        # When the op's task_complexity + route would produce a
        # thinking-enabled call (see providers.py:_resolve_thinking_*
        # rules), widen _max_cap to at least
        # JARVIS_FALLBACK_MAX_TIMEOUT_THINKING_S (default 360s, matches
        # Task #88's inner default).  Apply via ``max()`` so it never
        # SHRINKS the existing route-specific cap (e.g. COMPLEX's 180s,
        # read-only-BG's 480s+).
        #
        # The thinking-likelihood signal is conservative but correct
        # for the dominant case: any non-trivial task_complexity on a
        # non-reflex (non-IMMEDIATE) route will have thinking enabled
        # per the existing _resolve_thinking_budget rules.  We avoid
        # reaching back into providers._resolve_thinking_budget to keep
        # this module orchestration-free; the inline check matches the
        # decision rule structurally.
        # Single source of truth (Phase R1): the SAME predicate + cap
        # the OUTER Iron-Gate _gen_timeout uses, so outer >= inner by
        # construction (no duplicated rule, no 255-vs-360 drift).
        _likely_thinking = gen_call_likely_thinking(
            _op_route, getattr(context, "task_complexity", "") or "",
        )
        if _likely_thinking:
            _max_cap = max(_max_cap, fallback_thinking_cap_s())

        # Seed Arc Path 3 follow-up — PLAN-EXPLOIT per-stream override.
        # When ``plan_exploit_active_var`` is True (set by
        # ``try_parallel_generate`` before its gather), the per-stream
        # cap uses ``plan_exploit_per_stream_timeout_s()`` instead of
        # the default _FALLBACK_MAX_TIMEOUT_S. The 120s default was sized
        # for serial calls with retry rounds; applying it per-stream in
        # parallel mode artificially constrains streams doing legitimate
        # full-file generation when the parent has 220s+ remaining.
        # Outside PLAN-EXPLOIT context (the common case), behavior is
        # unchanged. The override clamps with max() against the existing
        # _max_cap so it never SHRINKS an already-larger cap (e.g. the
        # COMPLEX-route cap or the BG/SPEC subagent extension above).
        try:
            from backend.core.ouroboros.governance.plan_exploit import (
                plan_exploit_active_var as _plan_exploit_active,
                plan_exploit_per_stream_timeout_s as _plan_exploit_timeout,
            )
            if _plan_exploit_active.get(False):
                _max_cap = max(_max_cap, _plan_exploit_timeout())
        except Exception:  # noqa: BLE001 — override is best-effort
            pass

        # Promoted to INFO with phase label so traces distinguish first
        # GENERATE from GENERATE_RETRY contention on the shared fallback
        # semaphore — Session bt-2026-04-15-041413 (2026-04-14) saw a
        # retry wait 121.5s behind cohort ops with no visibility into
        # which acquisition phase was queuing. max_cap added after
        # Session F (bt-2026-04-15-065523) so the route-aware ceiling
        # that was actually applied is visible at acquire time.
        logger.info(
            "[CandidateGenerator] Fallback sem acquire: slots_free=%d/%d "
            "remaining=%.1fs route=%s phase=%s op=%s max_cap=%.0fs",
            self._fallback_sem._value, self._fallback_concurrency,
            _pre_sem_remaining,
            getattr(context, "provider_route", "?"),
            _phase_hint,
            getattr(context, "op_id", "?")[:16],
            _max_cap,
        )

        # AdmissionGate Slice 2 — pre-acquire viability check.
        # Refuses admission when projected wait + min-viable
        # call exceeds remaining budget, sheds load BEFORE
        # consuming a semaphore slot. Master flag default-FALSE
        # until Slice 3 — disabled gate degrades to ADMIT
        # (preserves pre-Slice-2 behavior). NEVER raises;
        # adopting a fail-open posture so a gate bug cannot
        # itself starve a legitimate op.
        try:
            from backend.core.ouroboros.governance.admission_gate import (  # noqa: E501
                AdmissionContext as _AdmissionContext,
                admission_gate_enabled as _admission_gate_enabled,
                compute_admission_decision as _compute_admission_decision,
            )
            _wait_est = getattr(self, "_wait_estimator", None)
            _projected_wait = (
                _wait_est.project_wait(_op_route)
                if _wait_est is not None else 0.0
            )
            # _fallback_sem._value is "slots free"; depth =
            # capacity − free.
            _live_depth = max(
                0,
                self._fallback_concurrency
                - int(getattr(self._fallback_sem, "_value", 0)),
            )
            _admission_ctx = _AdmissionContext(
                route=_op_route,
                remaining_s=_pre_sem_remaining,
                queue_depth=_live_depth,
                projected_wait_s=_projected_wait,
                op_id=str(getattr(context, "op_id", ""))[:48],
            )
            _admission = _compute_admission_decision(
                _admission_ctx,
                enabled=_admission_gate_enabled(),
                decided_at_ts=time.time(),
            )
            # Slice 3 — record EVERY decision (admit + shed) to
            # the bounded ring so the GET /observability/admission-
            # gate route shows recent admission patterns.
            # Best-effort, NEVER raises into the call path.
            try:
                from backend.core.ouroboros.governance.admission_estimator import (  # noqa: E501
                    get_default_history as _admit_history,
                )
                _admit_history().record(_admission.to_dict())
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.debug(
                    "[CandidateGenerator] admission history "
                    "record degraded: %s", exc,
                )
            if _admission.is_shed():
                logger.info(
                    "[CandidateGenerator] Pre-admission shed "
                    "decision=%s reason=%s route=%s "
                    "remaining=%.2fs projected_wait=%.2fs "
                    "queue_depth=%d required_budget=%.2fs",
                    _admission.decision.value,
                    _admission.reason, _admission.route,
                    _admission.remaining_s,
                    _admission.projected_wait_s,
                    _admission.queue_depth,
                    _admission.required_budget_s,
                )
                # Slice 3 — publish SSE event for IDE
                # consumers to surface saturation in real time.
                # Best-effort.
                try:
                    from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501
                        EVENT_TYPE_ADMISSION_DECISION_EMITTED,
                        get_default_broker,
                    )
                    _br = get_default_broker()
                    if _br is not None:
                        _br.publish(
                            event_type=(
                                EVENT_TYPE_ADMISSION_DECISION_EMITTED
                            ),
                            op_id=str(
                                getattr(context, "op_id", "")
                                or "",
                            )[:48],
                            payload=_admission.to_dict(),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[CandidateGenerator] admission SSE "
                        "publish degraded: %s", exc,
                    )
                self._raise_exhausted(
                    "pre_admission_shed",
                    context=context,
                    deadline=deadline,
                    sem_wait_total_s=0.0,
                    pre_sem_remaining_s=round(
                        _pre_sem_remaining, 2,
                    ),
                    admission_decision=(
                        _admission.decision.value
                    ),
                    admission_reason=_admission.reason,
                    projected_wait_s=round(
                        _admission.projected_wait_s, 2,
                    ),
                    queue_depth_at_check=(
                        _admission.queue_depth
                    ),
                    required_budget_s=round(
                        _admission.required_budget_s, 2,
                    ),
                )
        except RuntimeError:
            # _raise_exhausted raises RuntimeError — don't
            # swallow our own structural shed.
            raise
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.debug(
                "[CandidateGenerator] Admission gate "
                "degraded — proceeding to acquire: %s", exc,
            )

        try:
            # Slice 12F-A — priority-aware fallback sem acquire.
            # urgency=high SWE-Bench-Pro foreground ops (IMMEDIATE
            # route, priority=0) now preempt urgency=low /
            # BACKGROUND OpportunityMiner ops (priority=4) on slot
            # release. Hard concurrency cap preserved by the
            # underlying PrioritySemaphore counter. Hot-revert via
            # JARVIS_PRIORITY_SEM_ENABLED=false returns to FIFO.
            from backend.core.ouroboros.governance.priority_semaphore import (  # noqa: E501
                acquire_priority_aware as _slice12f_acquire,
            )
            async with _slice12f_acquire(self._fallback_sem, _op_route):
                _sem_wait_s = time.monotonic() - _sem_t0
                _parent_remaining = self._remaining_seconds(deadline)

                # D2 (Task #95, 2026-05-14) — sem-exhausted fast-fail.
                # Per operator binding: "after the semaphore wait is
                # charged, remaining_budget_for_network = max(0,
                # outer_remaining - sem_wait_total); if ≤ 0, fail fast
                # with a structured reason (sem_exhausted_zero_budget)
                # instead of still opening a stream that will always
                # violate outer wait_for."
                #
                # This sits BEFORE the post-acquire floor refresh (#88c
                # territory) by design: #88c's refresh is the explicit
                # op-envelope extension when budget is tight but
                # *nonzero* — honest enforcement.  D2 is the new
                # invariant for the *zero* case: do not pretend time
                # exists that does not.  When the entire pre-sem budget
                # was consumed waiting for the semaphore, the outer
                # asyncio.wait_for is already arithmetically violated;
                # opening a stream now guarantees a TimeoutError /
                # CancelledError 130s later (httpx connect+read
                # surrender latency).  Fast-fail here is observability +
                # cost win.
                #
                # Slice 12F-B (2026-05-22) — raise the D2 floor from
                # absolute-zero to JARVIS_STREAM_MINIMUM_READ_BUDGET_S
                # (default 10s). Phase 3A acceptance (bt-2026-05-22-
                # 184422) proved the gap: sem_wait_total=142.2s drained
                # the op's wall to ~0.01s — JUST above the 0.0 floor —
                # so D2 didn't fire and the stream opened with a
                # 0.01-second read budget. The subsequent inter-chunk
                # watchdog fired a misleading "no event for 0s" rupture.
                # That was a budget-too-short refusal masquerading as a
                # network rupture. Slice 12F-B raises the typed
                # StreamBudgetTooShortError BEFORE dispatch, which the
                # classifier maps to TRANSIENT_TRANSPORT →
                # RetryDecision.RETRY_TRANSIENT (NOT terminal
                # structural — Slice 7 fallback handles it as a
                # transient transport fault).
                from backend.core.ouroboros.governance.stream_rupture import (  # noqa: E501
                    StreamBudgetTooShortError,
                    stream_minimum_read_budget_s,
                )
                _min_read_budget_s = stream_minimum_read_budget_s()
                if _parent_remaining < _min_read_budget_s:
                    logger.info(
                        "[CandidateGenerator] Post-sem budget-floor "
                        "shed (Slice 12F-B): sem_wait=%.1fs drained "
                        "pre_sem_remaining=%.1fs → parent_remaining=%.2fs "
                        "below floor=%.1fs (route=%s). Refusing to "
                        "dispatch a stream that the wall budget cannot "
                        "honor; raising StreamBudgetTooShortError → "
                        "RETRY_TRANSIENT (Slice 7 fallback).",
                        _sem_wait_s, _pre_sem_remaining,
                        _parent_remaining, _min_read_budget_s, _op_route,
                    )
                    raise StreamBudgetTooShortError(
                        provider="claude-api",
                        op_id=str(getattr(context, "op_id", ""))[:48],
                        wall_remaining_s=round(_parent_remaining, 2),
                        minimum_required_s=_min_read_budget_s,
                        sem_wait_s=round(_sem_wait_s, 2),
                        route=str(_op_route or ""),
                    )

                # AdmissionGate Slice 2 — feed observed wait
                # back to the EWMA estimator so the next op's
                # projection reflects actual queue pressure.
                # NEVER raises into the call path.
                try:
                    _wait_est_post = getattr(
                        self, "_wait_estimator", None,
                    )
                    if _wait_est_post is not None:
                        _wait_est_post.update_observed(
                            _op_route, _sem_wait_s,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[CandidateGenerator] Estimator "
                        "update degraded: %s", exc,
                    )

                if _sem_wait_s > 1.0:
                    logger.info(
                        "[CandidateGenerator] Fallback sem_wait=%.1fs "
                        "(pre=%.1fs → post=%.1fs)",
                        _sem_wait_s, _pre_sem_remaining, _parent_remaining,
                    )

                # Post-acquire refresh: guarantee _FALLBACK_MIN_GUARANTEED_S
                # even when the parent deadline burned during sem wait or
                # Tier 0 consumed most of the window.  The orchestrator's
                # outer wait_for is the absolute Iron Gate. ``_max_cap`` is
                # the route-aware ceiling computed at acquire time (180s
                # for complex, 120s otherwise).
                #
                # Task #88c (2026-05-13) — thinking-aware floor reservation.
                # v14-rev7 proved the third budget layer: even with Task #88
                # (inner 360s) and #88b (outer _max_cap 360s) widened, the
                # actual Claude timeout was 90s because the DW cascade had
                # already consumed ~140s of the ~200s op deadline. The post-
                # acquire refresh's floor (_FALLBACK_MIN_GUARANTEED_S=90s)
                # was the binding constraint — and 90s is nowhere near the
                # 360s thinking-on inner/outer single-policy floor.
                #
                # Fix: when the call is likely-thinking (signal reused from
                # Task #88b, computed earlier in this method), promote the
                # floor to JARVIS_FALLBACK_MIN_GUARANTEED_THINKING_S (default
                # 360s, matching the #88/#88b inner+outer caps).  Same env-
                # tunable pattern.  Single-policy invariant the spine pins:
                # thinking floor >= max(inner, outer) for thinking-on calls.
                #
                # NOTE: the floor is OVERRIDDEN by ``_max_cap`` via the
                # subsequent ``min(..., _max_cap)``, so as long as #88b's
                # _max_cap=360 lands first, the math is:
                #   _budget_target = max(60.1s remaining, 360s floor) = 360s
                #   remaining = min(360s budget_target, 360s _max_cap) = 360s
                # Claude gets a guaranteed 360s, even when parent_remaining
                # was nearly exhausted by the DW cascade.  This is the
                # "Claude-floor reservation against op deadline" the operator
                # binding mandates — DW cannot force Claude below the floor.
                _min_guaranteed_s = (
                    float(os.environ.get(
                        "JARVIS_FALLBACK_MIN_GUARANTEED_THINKING_S", "360.0",
                    ))
                    if _likely_thinking
                    else _FALLBACK_MIN_GUARANTEED_S
                )
                _budget_target = max(_parent_remaining, _min_guaranteed_s)
                remaining = min(_budget_target, _max_cap)
                _refreshed = remaining > _parent_remaining + 1.0

                if remaining < _min_viable_fallback_s():
                    self._raise_exhausted(
                        "fallback_budget_starved",
                        context=context,
                        deadline=deadline,
                        sem_wait_s=round(_sem_wait_s, 2),
                        pre_sem_remaining_s=round(_pre_sem_remaining, 2),
                        parent_remaining_s=round(_parent_remaining, 2),
                        fallback_budget_s=round(remaining, 2),
                        min_viable_fallback_s=_min_viable_fallback_s(),
                    )

                if _refreshed:
                    logger.info(
                        "[CandidateGenerator] Fallback: budget=%.1fs REFRESHED "
                        "(parent=%.1fs, guaranteed_min=%.0fs, cap=%.0fs, "
                        "sem_wait=%.1fs, thinking=%s)",
                        remaining, _parent_remaining,
                        _min_guaranteed_s, _max_cap,
                        _sem_wait_s,
                        "yes" if _likely_thinking else "no",
                    )
                    deadline = datetime.now(tz=timezone.utc) + timedelta(
                        seconds=remaining,
                    )
                else:
                    logger.info(
                        "[CandidateGenerator] Fallback: budget=%.1fs "
                        "(cap=%.0fs, sem_wait=%.1fs)",
                        remaining, _max_cap, _sem_wait_s,
                    )

                # W3(7) Slice 2 — race against ambient cancel token (if any).
                # Outer-retry loop (rooted-problem fix 2026-04-25): re-invoke
                # the provider on transient failures while remaining budget
                # exceeds `_min_viable_fallback_s()`. Holds `_fallback_sem`
                # across attempts so head-of-queue position is preserved
                # (paying the wait fee twice would penalize the op for
                # provider flakiness — semantically incorrect).
                from backend.core.ouroboros.governance.cancel_token import (
                    OperationCancelledError as _OperationCancelledError,
                    current_cancel_token as _curr_cancel_token,
                    race_or_wait_for as _race_or_wait_for,
                )
                # Slice 7e (Provider Circuit Breaker) — wire the
                # state machine into the retry loop. Constructed once
                # per _call_fallback invocation (per op_id). Consumes
                # Slice 7a's classify() output; emits SSE telemetry.
                # On TERMINATE_UNRESOLVED short-circuits the loop +
                # fires _raise_exhausted with the breaker's reason
                # code — closing the empirical 35-min retry storm
                # from bt-2026-05-21-214521.
                #
                # Master flag ``JARVIS_PROVIDER_CIRCUIT_BREAKER_ENABLED``
                # default-FALSE. When off, ``breaker.evaluate()`` always
                # returns RETRY_OK → byte-identical to the pre-7e retry
                # loop (FailureMode / outer-retry cap / backoff constant
                # stay authoritative).
                #
                # Lazy imports avoid governance-package cycles
                # (mirrors the cancel_token import above).
                from backend.core.ouroboros.governance.circuit_breaker import (  # noqa: E501
                    CircuitBreaker as _Slice7e_CircuitBreaker,
                    CircuitScope as _Slice7e_CircuitScope,
                    CircuitTripOrigin as _Slice12N_CircuitTripOrigin,
                    VerdictAction as _Slice7e_VerdictAction,
                )
                from backend.core.ouroboros.governance.provider_retry_classifier import (  # noqa: E501
                    classify as _slice7e_classify,
                )
                # Slice 127 — economic reclassification gate (default-FALSE,
                # §33.1). Lives in economic_router so the PURE-DATA classifier
                # stays env-free (AST-pinned).
                from backend.core.ouroboros.governance.economic_router import (  # noqa: E501
                    economic_reclassify_enabled as _s127_econ_reclassify_enabled,
                )
                from backend.core.ouroboros.governance.ide_observability_stream import (  # noqa: E501
                    publish_provider_failure_classified as _slice7e_publish_classified,
                    publish_circuit_breaker_state_change as _slice7e_publish_state_change,
                    publish_circuit_breaker_tripped as _slice7e_publish_tripped,
                )

                _slice7e_op_id = str(
                    getattr(context, "op_id", "") or "",
                )
                # Slice 12N — blast-radius isolation. Map the op's
                # ProviderRoute to a CircuitTripOrigin so background /
                # speculative ops can trip their per-op breaker
                # WITHOUT escalating to the global session_exhausted
                # threshold. Foreground (IMMEDIATE / STANDARD /
                # COMPLEX) routes still escalate byte-identically to
                # pre-Slice-12N behavior. Unknown routes default to
                # FOREGROUND (safer — preserves legacy escalation).
                _slice12n_route = str(
                    getattr(context, "provider_route", "") or "",
                ).strip().lower()
                _slice12n_origin = _SLICE12N_ROUTE_TO_ORIGIN.get(
                    _slice12n_route,
                    _Slice12N_CircuitTripOrigin.FOREGROUND,
                )
                _slice7e_breaker = _Slice7e_CircuitBreaker(
                    op_id=_slice7e_op_id,
                    scope=_Slice7e_CircuitScope.PER_OP,
                    origin=_slice12n_origin,
                )
                _outer_attempt = 0
                # Anthropic resilience pack 2026-04-25 — failure-rate-aware
                # outer-retry max. When the FSM shows recent transient
                # failures (consecutive_failures > 0), bump the outer-retry
                # cap to give the op more headroom to catch a recovery
                # window during external instability. Healthy ops keep
                # the base cap (no extra cost).
                _fsm_consec_fails = getattr(
                    self.fsm, "_consecutive_failures", 0,
                )
                if _fsm_consec_fails > 0 and _FALLBACK_OUTER_RETRY_MAX_DEGRADED > _fallback_outer_retry_max():
                    _outer_max = _FALLBACK_OUTER_RETRY_MAX_DEGRADED
                    logger.info(
                        "[CandidateGenerator] Fallback outer-retry: degraded mode "
                        "detected (FSM consecutive_failures=%d) — bumping outer-retry "
                        "cap from %d to %d for op=%s (rooted-problem fix — failure-"
                        "rate-aware retry headroom)",
                        _fsm_consec_fails,
                        _fallback_outer_retry_max(), _outer_max,
                        getattr(context, "op_id", "?")[:16],
                    )
                else:
                    _outer_max = _fallback_outer_retry_max()
                _last_inner_exc: Optional[BaseException] = None
                # ── Slice 3C — outer-retry tool-record carryover ──
                # Iron Gate (post-GENERATE) inspects
                # ``GenerationResult.tool_execution_records`` to verify the
                # model met the exploration floor. Each provider attempt
                # carries only its OWN tool calls (the coordinator resets
                # ``_last_records`` at run start). If attempt N raises after
                # genuine exploration but attempt N+1 succeeds with NO tool
                # calls (the bt-2026-05-25-033000 cascade — direct patch
                # emit after retries), Iron Gate sees 0 records and rejects
                # even though the model explored the codebase across
                # attempts. Accumulator harvests records from each failed
                # attempt's exception (Slice-3C-stamped via
                # ``_attach_tool_records`` in tool_executor) and merges
                # into the winning attempt's GenerationResult below.
                _carryover_tool_records: List[Any] = []
                while True:
                    _outer_attempt += 1
                    _attempt_t0 = time.monotonic()
                    _attempt_remaining = self._remaining_seconds(deadline)
                    if _attempt_remaining < _min_viable_fallback_s():
                        # Budget exhausted — break to outer except handler
                        # which fires `fallback_budget_starved` if no prior
                        # exception, else fallback_failed with last exc.
                        if _last_inner_exc is not None:
                            raise _last_inner_exc
                        self._raise_exhausted(
                            "fallback_budget_starved",
                            context=context,
                            deadline=deadline,
                            sem_wait_s=round(_sem_wait_s, 2),
                            pre_sem_remaining_s=round(_pre_sem_remaining, 2),
                            parent_remaining_s=round(_attempt_remaining, 2),
                            fallback_budget_s=round(_attempt_remaining, 2),
                            min_viable_fallback_s=_min_viable_fallback_s(),
                        )
                    # Slice 89 — Build ExplorationManifest from DW's just-
                    # completed tool loop and stamp onto context BEFORE the
                    # Claude fallback generate() call.  NEVER raises
                    # (try/except guards the full block).  When
                    # JARVIS_EXPLORATION_MANIFEST_ENABLED is OFF (default
                    # §33.1), behavior is byte-identical to today.
                    #
                    # Gate: stamp only on the FIRST Claude attempt —
                    # DW's `_last_salient_args`/`_last_records` are only
                    # valid until DW's next run() resets them.  On outer
                    # attempt 1 _carryover_tool_records is always empty
                    # (populated only INSIDE the except block below), so
                    # the old `_carryover_tool_records and _outer_attempt==1`
                    # condition was mutually exclusive — dead code (C1 fix).
                    # We now harvest directly from the primary coordinator.
                    if _outer_attempt == 1:
                        try:
                            _s89_enabled = os.environ.get(
                                "JARVIS_EXPLORATION_MANIFEST_ENABLED", "false",
                            ).strip().lower() not in ("false", "0", "no", "off")
                            if _s89_enabled:
                                from backend.core.ouroboros.governance.op_context import (
                                    ExplorationManifest as _ExplorationManifest,
                                )
                                # Harvest BOTH records and salient_args from
                                # the primary provider's coordinator — these
                                # are the DW tool loop results just before
                                # DW's generate() failed/timed-out.  Using
                                # the same coordinator source keeps them
                                # length-aligned (C2 alignment preserved).
                                _coord = getattr(
                                    self._primary, "_tool_loop", None,
                                )
                                _s89_records = tuple(
                                    getattr(_coord, "_last_records", ()) or ()
                                ) if _coord is not None else ()
                                _s89_salient = list(
                                    getattr(_coord, "_last_salient_args", ()) or ()
                                ) if _coord is not None else []
                                _s89_manifest = _ExplorationManifest.from_telemetry(
                                    records=_s89_records,
                                    salient_args=_s89_salient,
                                    reason="dw_failure",
                                )
                                context = context.with_exploration_manifest(_s89_manifest)
                                logger.info(
                                    "[CandidateGenerator] Slice 89: stamped "
                                    "ExplorationManifest onto context "
                                    "(tool_calls=%d, target_files=%d, "
                                    "search_tokens=%d, failed_tests=%d) op=%s",
                                    _s89_manifest.tool_call_count,
                                    len(_s89_manifest.verified_target_files),
                                    len(_s89_manifest.high_signal_search_tokens),
                                    len(_s89_manifest.failed_test_commands),
                                    getattr(context, "op_id", "?")[:16],
                                )
                        except Exception:  # noqa: BLE001 — never break the cascade
                            pass
                    try:
                        _fb_result = await _race_or_wait_for(
                            self._fallback.generate(context, deadline),
                            timeout=_attempt_remaining,
                            cancel_token=_curr_cancel_token(),
                        )
                        if _outer_attempt > 1:
                            logger.info(
                                "[CandidateGenerator] Fallback outer-retry "
                                "succeeded on attempt %d/%d after %.1fs "
                                "(rooted-problem fix consumed %.1fs of "
                                "previously-unused budget)",
                                _outer_attempt, _outer_max,
                                time.monotonic() - _sem_t0,
                                time.monotonic() - _sem_t0 - (
                                    _attempt_t0 - _sem_t0
                                ),
                            )
                        logger.info(
                            "[CandidateGenerator] Fallback sem release: "
                            "hold=%.1fs sem_wait=%.1fs route=%s phase=%s op=%s outcome=ok",
                            time.monotonic() - _sem_t0, _sem_wait_s,
                            getattr(context, "provider_route", "?"),
                            _phase_hint,
                            getattr(context, "op_id", "?")[:16],
                        )
                        # Slice 3C — merge carryover records into the
                        # winning attempt's GenerationResult so Iron Gate
                        # sees the cumulative exploration across attempts.
                        # No-op when carryover is empty (single-attempt
                        # success path) — byte-identical legacy behavior.
                        if _carryover_tool_records:
                            try:
                                _winning_records = tuple(
                                    getattr(
                                        _fb_result,
                                        "tool_execution_records",
                                        (),
                                    ) or ()
                                )
                                _merged = (
                                    tuple(_carryover_tool_records)
                                    + _winning_records
                                )
                                _with_tool_records = getattr(
                                    _fb_result, "with_tool_records", None,
                                )
                                if callable(_with_tool_records):
                                    _fb_result = _with_tool_records(_merged)
                                    logger.info(
                                        "[CandidateGenerator] Slice 3C: "
                                        "merged %d carryover tool records "
                                        "from %d failed attempt(s) into "
                                        "winning GenerationResult "
                                        "(winning=%d, total=%d) op=%s",
                                        len(_carryover_tool_records),
                                        _outer_attempt - 1,
                                        len(_winning_records),
                                        len(_merged),
                                        getattr(
                                            context, "op_id", "?",
                                        )[:16],
                                    )
                            except Exception:  # noqa: BLE001 — defensive
                                # Carryover merge must NEVER break a
                                # successful generate. Log + fall through
                                # with the unmerged result.
                                logger.exception(
                                    "[CandidateGenerator] Slice 3C "
                                    "carryover merge degraded — returning "
                                    "winning-attempt GenerationResult "
                                    "unmodified for op=%s",
                                    getattr(context, "op_id", "?")[:16],
                                )
                        return _fb_result
                    except _OperationCancelledError:
                        # W3(7) cooperative cancel — operator/watchdog/signal.
                        # NEVER retry; honor the cancel immediately.
                        raise
                    except GovernanceDeadlockError:
                        # LR3 terminal: the fallback (Claude) tool loop hit the
                        # Information-Gain Governor deadlock-override failure.
                        # NEVER retry / reclassify; propagate to the orchestrator's
                        # terminal catch so it stamps deadlock_override_failed.
                        raise
                    except (Exception, asyncio.CancelledError) as inner_exc:
                        # Slice 3C — harvest tool-records from the failed
                        # attempt BEFORE any other handling. Best-effort:
                        # if the exception didn't carry records (legacy
                        # provider, untagged path), getattr returns (),
                        # extend is a no-op, behavior matches pre-Slice-3C.
                        #
                        # Slice 3D (2026-05-24) — coordinator-attribute
                        # fallback. The Slice 3C exception attachment only
                        # fires on raises through ToolLoopCoordinator's
                        # ``_attach_tool_records`` sites. When the outer
                        # ``_race_or_wait_for(... timeout=_attempt_remaining)``
                        # hits its deadline, it raises asyncio.TimeoutError
                        # / CancelledError that DOES NOT traverse the tool
                        # executor's raise sites — the records sit untouched
                        # in ``coordinator._last_records``. The
                        # bt-2026-05-25-041717 attempt 1: 244.2s tool loop
                        # made 13+ tool calls then the outer race timed
                        # out; Slice 3C harvested nothing because the
                        # TimeoutError was opaque; Iron Gate saw 0/2 on
                        # the final attempt and the cumulative exploration
                        # was lost.
                        #
                        # Fallback path: when the exception carries no
                        # records, read directly from the coordinator's
                        # instance attribute. ``_last_records`` is reset to
                        # ``[]`` at every ``run()`` start (tool_executor.py
                        # line 5250) and re-populated at each round
                        # boundary, so at except-block time it reflects
                        # exactly the just-failed attempt's records — no
                        # cross-attempt double-counting.
                        try:
                            _harvested = getattr(
                                inner_exc, "tool_execution_records", (),
                            ) or ()
                            if not _harvested:
                                # Slice 3D fallback — coordinator probe.
                                # Defensive getattr chain: any provider
                                # without ``_tool_loop`` (tools disabled
                                # config) or coordinator without
                                # ``_last_records`` (legacy/test stub)
                                # falls through to empty harvest.
                                _coord = getattr(
                                    self._fallback, "_tool_loop", None,
                                )
                                if _coord is not None:
                                    _harvested = tuple(
                                        getattr(
                                            _coord, "_last_records", (),
                                        ) or ()
                                    )
                                    if _harvested:
                                        logger.info(
                                            "[CandidateGenerator] Slice 3D: "
                                            "harvested %d records from "
                                            "coordinator._last_records "
                                            "(exception %s carried 0; "
                                            "fallback succeeded) op=%s",
                                            len(_harvested),
                                            type(inner_exc).__name__,
                                            getattr(
                                                context, "op_id", "?",
                                            )[:16],
                                        )
                            if _harvested:
                                _carryover_tool_records.extend(_harvested)
                        except Exception:  # noqa: BLE001 — never block retry
                            pass
                        # Pre-instrumented (e.g. fallback_budget_starved
                        # from a different code path) → propagate as-is.
                        if hasattr(inner_exc, "exhaustion_report"):
                            raise
                        _last_inner_exc = inner_exc
                        _inner_mode = (
                            FailbackStateMachine.classify_exception(inner_exc)
                        )
                        # Slice 7e — Consult the Circuit Breaker on
                        # every failure. When master flag is OFF,
                        # ``evaluate()`` returns RETRY_OK → byte-
                        # identical to the pre-7e path. When ON,
                        # TERMINAL_STRUCTURAL / TERMINAL_CONFIG short-
                        # circuit immediately (closing the 35-min
                        # retry storm); TERMINAL_QUOTA + repeated
                        # RETRY_TRANSIENT trigger Full-Jitter backoff;
                        # the FSM / outer-retry cap / existing
                        # eligibility check below remain as additional
                        # gates (defense in depth — no breaker bypass
                        # of the pre-existing semantics).
                        # Slice 127 — pass the raw message + economic gate so a
                        # "credit balance too low" 400 (class BadRequestError)
                        # reclassifies to recoverable TERMINAL_QUOTA instead of
                        # sticky TERMINAL_CONFIG. Gate default-FALSE → OFF is
                        # byte-identical to pre-127. (bt-2026-06-07-040933 root
                        # cause: 16 sticky terminal_config trips.)
                        try:
                            _s127_econ_on = _s127_econ_reclassify_enabled()
                        except Exception:  # noqa: BLE001 — failure-soft
                            _s127_econ_on = False
                        _slice7e_decision = _slice7e_classify(
                            failure_class=type(inner_exc).__name__,
                            failure_mode=_inner_mode.name,
                            failure_message=str(inner_exc),
                            economic_reclassify=_s127_econ_on,
                            # Dynamic 5xx Resiliency Matrix (2026-07-22): thread
                            # the provider HTTP status + Retry-After presence
                            # into the taxonomy (previously omitted). A 5xx /
                            # upstream_error / 429-with-Retry-After now
                            # classifies TRANSIENT_NETWORK instead of falling
                            # through to a terminal decision.
                            http_status=getattr(inner_exc, "status_code", None),
                            retry_after_present=(
                                getattr(inner_exc, "ratelimit_reset_ts", None)
                                is not None
                            ),
                        )
                        # Slice 127 Phase 2 — per-provider economic breaker.
                        # The fallback IS the Claude/Anthropic lane; when this
                        # failure is an economic block ("credit balance too
                        # low" / 402), trip the per-lane self-healing breaker so
                        # FUTURE ops route around the broke lane (existing
                        # should_allow_request gate) and it recovers after the
                        # window — no sticky session brick. Gated + defensive;
                        # detail is the redacted economic code, never a secret.
                        _claude_econ_block = False
                        try:
                            if _s127_econ_on:
                                from backend.core.ouroboros.governance.economic_router import (  # noqa: E501
                                    is_hard_economic_block as _s127_is_econ,
                                )
                                _s127_block = _s127_is_econ(str(inner_exc))
                                if _s127_block is not None:
                                    _claude_econ_block = True
                                    from backend.core.ouroboros.governance.claude_circuit_breaker import (  # noqa: E501
                                        get_claude_circuit_breaker as _s127_get_ccb,
                                    )
                                    _s127_get_ccb().record_economic_exhaustion(
                                        f"claude_lane_economic:{_s127_block}",
                                    )
                                    # Council finding (2026-07-21): the
                                    # economic state also lands in the
                                    # liquidity LEDGER so every runway
                                    # consumer sees the dead wallet.
                                    _record_quota_outage_safely(
                                        "anthropic", str(inner_exc)[:160])
                        except Exception:  # noqa: BLE001 — never block cascade
                            pass
                        # Telemetry — every classification is logged,
                        # regardless of breaker state. Best-effort.
                        _slice7e_publish_classified(
                            failure_class=type(inner_exc).__name__,
                            failure_mode=_inner_mode.name,
                            decision=_slice7e_decision.value,
                            provider="claude",
                            op_id=_slice7e_op_id,
                        )
                        # Sovereign State Isolation (2026-06-19): a confirmed
                        # Claude economic death is OWNED by Claude's lane breaker
                        # (recorded above). Do NOT let it trip the provider-
                        # NEUTRAL per-op breaker into the sticky OPEN_TERMINAL —
                        # that poisons the op for DW too (empirically-confirmed
                        # cross-provider contamination: terminal_quota 5->0 once
                        # isolated). Downgrade ONLY the decision the OP breaker
                        # sees to RETRY_TRANSIENT (non-poisoning) so the op stays
                        # viable for DW autarky; the Claude lane is already marked
                        # dead by its global breaker + the Slice238 suppression.
                        # Telemetry above already published the TRUE decision.
                        if quota_isolation_skips_op_breaker(
                            is_provider_economic_block=_claude_econ_block,
                            isolation_enabled=_provider_quota_isolation_enabled(),
                        ):
                            logger.warning(
                                "[CandidateGenerator] QUOTA ISOLATION: Claude "
                                "economic death contained to Claude lane breaker; "
                                "per-op breaker NOT terminal-tripped (op stays "
                                "viable for DW autarky) op=%s",
                                _slice7e_op_id,
                            )
                            _slice7e_decision = type(
                                _slice7e_decision
                            ).RETRY_TRANSIENT
                        _slice7e_prior_state = _slice7e_breaker.state.value
                        _slice7e_verdict = _slice7e_breaker.evaluate(
                            _slice7e_decision,
                        )
                        _slice7e_new_state = _slice7e_verdict.state_after \
                            and _slice7e_verdict.state_after.value or \
                            _slice7e_breaker.state.value
                        if _slice7e_prior_state != _slice7e_new_state:
                            # State change → SSE telemetry. Trip
                            # events use the more-specific publisher
                            # below; non-trip transitions go here.
                            if _slice7e_verdict.action != (
                                _Slice7e_VerdictAction.TERMINATE_UNRESOLVED
                            ):
                                _slice7e_publish_state_change(
                                    prior_state=_slice7e_prior_state,
                                    new_state=_slice7e_new_state,
                                    op_id=_slice7e_op_id,
                                    scope="per_op",
                                )
                        if _slice7e_verdict.action == (
                            _Slice7e_VerdictAction.TERMINATE_UNRESOLVED
                        ):
                            # Breaker trip — emit the trip SSE event
                            # + raise exhausted with the breaker's
                            # reason code. The orchestrator's existing
                            # exhaustion handler picks up the cause
                            # tag end-to-end; the parallel evaluator
                            # can subscribe to circuit_breaker_tripped
                            # for early-collapse instead of waiting
                            # on operation_terminal.
                            _slice7e_reason = (
                                _slice7e_verdict.terminal_reason_code
                                or "circuit_breaker_tripped:unknown"
                            )
                            _slice7e_publish_tripped(
                                terminal_reason_code=_slice7e_reason,
                                op_id=_slice7e_op_id,
                                scope="per_op",
                                backoff_attempt=(
                                    _slice7e_breaker.backoff_attempt
                                ),
                            )
                            self._raise_exhausted(
                                _slice7e_reason,
                                context=context,
                                deadline=deadline,
                                fallback_exc=inner_exc,
                                fallback_failure_mode=_inner_mode.name,
                                slice7e_decision=(
                                    _slice7e_decision.value
                                ),
                            )
                        # Permanent failures — never retry.
                        if not _is_outer_retry_eligible_mode(_inner_mode):
                            raise
                        # Hit the outer-retry cap.
                        if _outer_attempt >= _outer_max:
                            raise
                        # Budget check before backoff.
                        _attempt_elapsed = time.monotonic() - _attempt_t0
                        _budget_after = self._remaining_seconds(deadline)
                        if _budget_after < _min_viable_fallback_s():
                            raise
                        logger.info(
                            "[CandidateGenerator] Fallback outer-retry: "
                            "attempt %d/%d failed (%s/%s) after %.1fs; "
                            "%.1fs budget remains, retrying op=%s "
                            "(rooted-problem fix — consuming budget JARVIS "
                            "already authorized, not inflating)",
                            _outer_attempt, _outer_max,
                            type(inner_exc).__name__,
                            _inner_mode.name,
                            _attempt_elapsed, _budget_after,
                            getattr(context, "op_id", "?")[:16],
                        )
                        # Brief backoff between outer attempts. Capped at
                        # remaining-budget/4 so a 12s budget doesn't sleep
                        # 1s of it (which would risk underflow into the
                        # min_viable floor on the next attempt).
                        #
                        # Slice 7e — when the breaker returned
                        # RETRY_AFTER_BACKOFF with a non-None backoff_s,
                        # use the Full-Jitter delay (AWS algorithm)
                        # instead of the fixed constant. This is the
                        # anti-thundering-herd path. The budget/4 clamp
                        # still applies so a tight remaining budget
                        # doesn't oversleep.
                        if (
                            _slice7e_verdict.action == (
                                _Slice7e_VerdictAction.RETRY_AFTER_BACKOFF
                            )
                            and _slice7e_verdict.backoff_s is not None
                        ):
                            _backoff = min(
                                float(_slice7e_verdict.backoff_s),
                                max(0.1, _budget_after / 4.0),
                            )
                        else:
                            _backoff = min(
                                _fallback_outer_retry_backoff_s(),
                                max(0.1, _budget_after / 4.0),
                            )
                        await asyncio.sleep(_backoff)
                        continue
                # Unreachable — loop either returns or raises.
        except GovernanceDeadlockError:
            # LR3 terminal: defense-in-depth. The inner-retry handler already
            # re-raises this, but guard the outer catch too so a deadlock can
            # never be folded into the all_providers_exhausted taxonomy. Must
            # reach the orchestrator's deadlock_override_failed terminal catch.
            raise
        except (Exception, asyncio.CancelledError) as exc:
            # Cooperative cancel via W3(7) cancel-token — propagate
            # immediately (NEVER treat as exhaustion). The inner loop
            # raises OperationCancelledError; this outer handler must
            # not swallow it into the fallback_failed taxonomy or the
            # operator's cancel signal would be silently downgraded
            # into "another transport failure".
            from backend.core.ouroboros.governance.cancel_token import (
                OperationCancelledError as _OperationCancelledError_outer,
            )
            if isinstance(exc, _OperationCancelledError_outer):
                raise
            # If the exception is already instrumented (e.g. the inner
            # ``fallback_budget_starved`` raise), re-raise as-is so we
            # preserve the more-specific cause and don't double-count
            # the exhaustion event counter.
            if hasattr(exc, "exhaustion_report"):
                raise
            logger.info(
                "[CancelAttribution] %s",
                _attribute_cancel(
                    exc,
                    label="_call_fallback",
                    op_id=getattr(context, "op_id", "?"),
                    elapsed_s=time.monotonic() - _sem_t0,
                    remaining_s=self._remaining_seconds(deadline),
                ),
            )
            mode = FailbackStateMachine.classify_exception(exc)
            self.fsm.record_fallback_failure(mode=mode)
            # Distinct cause tag when the tool-loop pre-round viability
            # gate fired. This is NOT a transport/API failure — it's a
            # round-level budget exhaustion that the ToolLoopCoordinator
            # caught before a doomed sub-floor call. Keeping the cause
            # distinct in breadcrumbs lets grep audits see "round_starved"
            # vs generic "fallback_failed" without reading full messages.
            _cause = "fallback_failed"
            if "tool_loop_round_budget_starved" in str(exc):
                _cause = "fallback_round_starved"
            self._raise_exhausted(
                _cause,
                context=context,
                deadline=deadline,
                fallback_exc=exc,
                fallback_failure_mode=mode.name,
                sem_wait_total_s=round(time.monotonic() - _sem_t0, 2),
                pre_sem_remaining_s=round(_pre_sem_remaining, 2),
            )

    @staticmethod
    def _remaining_seconds(deadline: datetime) -> float:
        """Compute seconds remaining until *deadline*.

        Returns a non-negative float.  If the deadline has already passed,
        returns 0.0 (which will cause ``asyncio.wait_for`` to time out
        immediately).
        """
        now = datetime.now(tz=timezone.utc)
        remaining = (deadline - now).total_seconds()
        return max(remaining, 0.0)

    # ------------------------------------------------------------------
    # Per-op Tier 0 rotation (Manifesto §5 — defensive cost guard)
    # ------------------------------------------------------------------

    def _should_skip_tier0_for_op(self) -> bool:
        """Return True if Tier 0 should be skipped for the current op.

        Skips when ``_consecutive_tier0_failures`` reaches the threshold
        AND the most recent failure happened within ``_tier0_skip_window_s``
        seconds. Outside the window the counter resets implicitly because
        the elapsed-time check fails — equivalent to a stale-feed reset.

        This is independent of the FSM's mode-based ETA: even if the
        classifier mis-routes a transport flap to TIMEOUT (default), this
        guard still kicks in after N back-to-back failures.
        """
        if self._counters.consecutive_tier0_failures < self._tier0_skip_threshold:
            return False
        elapsed = time.monotonic() - self._counters.last_tier0_failure_at
        return elapsed < self._tier0_skip_window_s

    def _record_tier0_failure(self) -> None:
        """Increment the per-op rotation counter and stamp the timestamp."""
        self._counters.consecutive_tier0_failures += 1
        self._counters.last_tier0_failure_at = time.monotonic()

    def _record_tier0_success(self) -> None:
        """Reset the per-op rotation counter on Tier 0 success."""
        if self._counters.consecutive_tier0_failures > 0:
            logger.info(
                "[CandidateGenerator] Tier 0 rotation reset after %d "
                "consecutive failures",
                self._counters.consecutive_tier0_failures,
            )
        self._counters.consecutive_tier0_failures = 0
        self._counters.last_tier0_failure_at = 0.0

    # ------------------------------------------------------------------
    # Deadline budget allocation (deterministic — Manifesto §5)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tier0_budget(
        total_s: float,
        complexity: str = "trivial",
        provider_route: str = "standard",
    ) -> float:
        """Deterministic Tier 0 (DoubleWord) budget with Tier 1 reserve.

        Tier 0 is the preferred path (cheap, Manifesto §5 Tier 0 fast-path).
        It gets 65% of the total budget by default.  When the total budget is
        tight (< 90s), we log a warning — both tiers may starve.

        *complexity* scales the base fraction via ``_TIER0_COMPLEXITY_MULTIPLIER``
        so that complex operations receive proportionally more Tier 0 time
        (e.g. 80% instead of 65%).

        *provider_route* overrides the budget profile for non-standard routes:
          - "complex":     80% fraction, 120s max, 20s reserve (DW executes plan)
          - "background":  100% fraction, 180s max, 0s reserve (DW only)
          - "immediate":   0% (skip DW entirely)

        Invariants:
          - tier0_budget <= total_s * effective_fraction
          - tier0_budget <= max_wait_s
          - total_s - tier0_budget >= tier1_reserve (when possible)
        """
        if total_s <= 0:
            return 0.0

        # Route-aware budget profile overrides
        if provider_route == "immediate":
            return 0.0
        if provider_route == "background":
            # DW only — no Claude reserve needed
            return min(total_s, 180.0)
        if provider_route == "speculative":
            return min(total_s, 300.0)

        if total_s < 90.0 and provider_route == "standard":
            logger.warning(
                "[CandidateGenerator] Generation budget tight (%.0fs < 90s). "
                "Consider increasing JARVIS_GENERATION_TIMEOUT_S for reliable "
                "2-tier cascade.",
                total_s,
            )

        multiplier = _TIER0_COMPLEXITY_MULTIPLIER.get(complexity, 1.0)

        # COMPLEX route: DW gets more budget (Claude already planned)
        if provider_route == "complex":
            effective_fraction = min(_TIER0_BUDGET_FRACTION * max(multiplier, 1.231), 0.90)
            max_wait = 120.0
            min_reserve = 20.0
        else:
            effective_fraction = min(_TIER0_BUDGET_FRACTION * multiplier, 0.90)
            max_wait = _TIER0_MAX_WAIT_S
            min_reserve = _TIER1_MIN_RESERVE_S

        # Reserve Tier 1 budget first (defensive — Tier 1 must always get a chance)
        tier1_reserve = min(min_reserve, total_s * (1.0 - effective_fraction))
        # Tier 3 Reflex (Manifesto §5): absolute hard cap on DW calls.
        # Strictest of four constraints wins (fraction, route max_wait,
        # tier1 reserve, Tier 0 RT cap). Added 2026-04-24 after F1 Slice 4 S4
        # (bt-2026-04-24-213248) proved the previous patch (inside
        # _call_primary) was inert for the DW-is-Tier0-AND-Primary
        # configuration — this code path is where DW actually gets its
        # 90s max_wait in that configuration.
        #
        # Slice 18c (2026-05-26) — the 4th constraint is now route-aware.
        # STANDARD + COMPLEX get the new JARVIS_DW_TIER0_RT_BUDGET_S
        # (default 90s — matches 397B/Kimi TTFT envelope) instead of the
        # 30s reflex cap. Eliminates the FLEET-v13-soak premature-timeout
        # cascade pattern (8 EXHAUSTION events, each on a DW dispatch
        # that needed >30s to complete). IMMEDIATE/BG/SPEC preserved at
        # 30s for cost-optimization semantics.
        budget = min(
            total_s * effective_fraction,
            max_wait,
            total_s - tier1_reserve,
            _tier0_rt_cap_for_route(provider_route),
        )
        return max(budget, 0.0)

    def _compute_tier0_budget_dynamic(
        self,
        total_s: float,
        complexity: str = "trivial",
        provider_route: str = "standard",
    ) -> float:
        """Tier 0 budget with rolling p95 awareness (Manifesto §5).

        Computes the static deterministic budget first (preserving all Tier 1
        reserve invariants), then tightens it using the latency tracker's p95
        recommendation when the endpoint is hot. On cold start (few samples
        or recent failures), falls through to the static budget so the first
        calls get full runway.

        The tracker NEVER loosens beyond the static ceiling — it only dials
        down when DW RT has proven fast enough.
        """
        static_budget = self._compute_tier0_budget(total_s, complexity, provider_route)
        if static_budget <= 0:
            return 0.0

        # Routes that skip tracker scaling entirely.
        if provider_route in ("immediate", "background", "speculative"):
            return static_budget

        tracker = self._latency_tracker
        if tracker is None:
            return static_budget

        # Use the static budget as the caller-provided ceiling — the tracker
        # can only dial down from here, never above it. Tier 1 reserve is
        # already guaranteed by _compute_tier0_budget.
        complexity_mult = _TIER0_COMPLEXITY_MULTIPLIER.get(complexity, 1.0)
        recommended = tracker.recommended_budget(
            route_ceiling_s=static_budget,
            complexity_multiplier=complexity_mult,
        )
        final_budget = max(0.0, min(static_budget, recommended))

        if final_budget < static_budget - 0.5:
            logger.info(
                "[CandidateGenerator] DW dynamic budget: %.1fs → %.1fs "
                "(hot endpoint, p95=%.1fs)",
                static_budget, final_budget, tracker.p95() or 0.0,
            )
        return final_budget

    @staticmethod
    def _apply_lane_dilation(budget: float, op_id: str) -> float:
        """Dynamic Lane Escalation (Part 2, T5) -- per-op deadline dilation seam.

        When ``JARVIS_LANE_ESCALATION_ENABLED`` is ON and this op has recorded
        one or more LANE COLLAPSE dilation hops (realtime+batch both timed out),
        scale the computed primary ``budget`` by ``factor ** hops`` (capped at
        ``JARVIS_LANE_DILATION_MAX_S``) for THIS re-attempt. The hop count is the
        bounded per-op counter recorded by ``_record_lane_collapse_dilation`` --
        the immortal queue re-attempts the op with the SAME op_id, so each
        re-attempt picks up the recorded dilation.

        OFF / no hops recorded -> returns ``budget`` unchanged (byte-identical).
        Fail-soft: any error -> legacy ``budget`` (op never lost)."""
        try:
            if not op_id or not lane_escalation_enabled():
                return budget
            from backend.core.ouroboros.governance.convergence_watchdog import (
                get_lane_dilation_tracker as _get_dt,
                compute_dilated_deadline as _dilate,
            )
            hops = _get_dt().hops(op_id)
            if hops <= 0:
                return budget
            return _dilate(budget, hops)
        except Exception:  # noqa: BLE001 -- dilation must NEVER break dispatch
            return budget

    @staticmethod
    def _compute_primary_budget(
        total_s: float,
        *,
        model_id: str = "",
        force_batch: bool = False,
        fallback_dead: bool = False,
        local_seat: bool = False,
        op_id: str = "",
    ) -> float:
        """Deterministic Tier 1 primary budget with fallback reserve + Tier 3 cap.

        ``local_seat`` (2026-09-02): the primary about to be called is the
        locally-served engine. With ``fallback_dead`` it lifts the DW autarky
        constant in favour of the local profiler's own absolute ceiling --
        the number that module already treats as the un-inflatable kill
        line for a wedged model. A 180s constant tuned for a cloud API was
        wrong for a 30B model whose real draws run 250-600s; the physics it
        replaces is already measured and owned elsewhere, so it is read,
        not redeclared.

        Invariants (enforced via ``min()`` — strictest wins):
          - primary_budget <= total_s * _PRIMARY_BUDGET_FRACTION
          - total_s - primary_budget >= _FALLBACK_MIN_RESERVE_S (when possible)
          - primary_budget <= effective_max (Slice 28 adaptive Tier 3 cap)

        Tier 3 cap added 2026-04-24 after F1 Slice 4 S3 (bt-2026-04-24-204029)
        exposed a 153s DW primary hold that exhausted the session before
        Claude fallback could produce a candidate.

        Slice 28 Phase 2 — Adaptive Streaming TTFT Horizon
        ---------------------------------------------------
        v21 forensic (bt-2026-05-27-025855) revealed the actual wedge: 12
        EXHAUSTION events on the 397B model, all classified as TIMEOUT, all
        firing at elapsed=30.01s with remaining=329.86s. The static
        ``_PRIMARY_MAX_TIMEOUT_S`` (30s default) was killing primary calls
        long before the streaming layer's 120s TTFT could even fire on the
        wire. Cold-start TTFT for a 397B MoE on a contended endpoint
        legitimately exceeds 30s — per §46 fleet inventory the 397B is
        characterized as a heavy-reasoning workhorse whose TTFT envelope
        is materially larger than the 35B sibling.

        When ``model_id`` is a heavy-reasoning / long-context model
        (matched against the same marker set Slice 27 Phase 3 uses for the
        adaptive Tier 0 timeout), multiply ``_PRIMARY_MAX_TIMEOUT_S`` by
        a heavy scalar (default 2.5×) so the call has runway to receive
        the first token. Hard ceiling at 240s matches the Slice 27 Phase 3
        cap (no unbounded cost bleeding).

        Legacy callers that pass only ``total_s`` (no ``model_id``) get the
        byte-identical pre-Slice-28 behavior — the 30s cap is preserved as
        the binding constraint. The adaptive widening engages only when
        the dispatcher has stamped the per-attempt model_id via the
        topology ContextVar.
        """
        if total_s <= 0:
            return 0.0

        # Slice 43 — Async Batch Timeout Alignment.
        # When the op will be dispatched through the BATCH lane (Slice 36/41
        # FORCE_BATCH), the provider's internal poll_and_retrieve legitimately
        # runs for minutes — the batch_future_registry waits up to
        # _DW_MAX_WAIT_S (3600s). Wrapping that in the 30s RT reflex cap
        # (_PRIMARY_MAX_TIMEOUT_S) severs the async batch mid-flight (v37
        # bt-2026-05-28-235234: batch 7b7a7b52 submitted then abandoned at
        # 30s). Give batch ops a batch-appropriate budget instead, capped by
        # remaining session time. force_batch implies Claude is disabled
        # (Slice 36 precondition) → no fallback to reserve for, so the batch
        # gets the full remaining runway up to the batch cap.
        if force_batch:
            # Sovereign Infinite-Horizon Batch Matrix: a PARKED batch continuation
            # (worker freed, zero CPU) gets the full SLA horizon so an
            # actively-processing DW batch is never severed mid-flight. The
            # ContextVar is set ONLY by the out-of-pool park continuation, so an
            # in-pool dispatch keeps the bounded 300s cap (never wedges a live
            # worker). The poll itself is lifecycle-aware (gives up only on terminal
            # status), so this just stops the OUTER budget from cutting it short.
            if _parked_batch_horizon_active():
                batch_cap = batch_sla_horizon_s()
            else:
                batch_cap = _envf_or_default("JARVIS_DW_BATCH_TIMEOUT_S", 300.0)
            return CandidateGenerator._apply_lane_dilation(
                max(min(total_s, batch_cap), 0.0), op_id,
            )

        # Slice 225 Phase 2 — Sovereign DW Autarky. When the Claude fallback
        # lane is unreliable (breaker OPEN/HALF_OPEN — incl. the terminal_quota
        # / out-of-credits economic refusal), there is NO live fallback to hand
        # off to. Severing DW at the 30s/75s reflex cap only accelerates
        # exhaustion into a dead lane — the live-soak GOAL-001::file-00 wedge:
        # DW cut at 30s -> Claude 400 "credit balance too low" -> EXHAUSTION,
        # generation_failed, no patch ever produced. Give DW the full remaining
        # runway up to a cost-safety ceiling instead (default 180s = the COMPLEX
        # generation window). Mirrors the force_batch precedent directly above
        # ("Claude disabled -> no fallback to reserve -> full runway"). The
        # caller stamps fallback_dead from the read-only _claude_breaker_open
        # predicate; default False is byte-identical to the legacy cascade.
        if fallback_dead:
            if local_seat:
                # The local lane's kill line is the local profiler's absolute
                # ceiling (JARVIS_LOCAL_INFERENCE_ABSOLUTE_CEILING_MS) -- the
                # one number that module refuses to let the EWMA inflate past.
                # Below it, the streaming watchdog is the guard. Fail-soft to
                # the DW constant only if the director cannot be consulted.
                try:
                    from backend.core.ouroboros.governance.local_inference_director import (  # noqa: PLC0415
                        _absolute_ceiling_ms as _local_kill_line_ms,
                    )
                    local_cap = max(1.0, _local_kill_line_ms() / 1000.0)
                    return CandidateGenerator._apply_lane_dilation(
                        max(min(total_s, local_cap), 0.0), op_id,
                    )
                except Exception:  # noqa: BLE001 — never lose the op over a budget read
                    pass
            autarky_cap = _envf_or_default(
                "JARVIS_DW_AUTARKY_MAX_BUDGET_S", 180.0,
            )
            return CandidateGenerator._apply_lane_dilation(
                max(min(total_s, autarky_cap), 0.0), op_id,
            )

        fb_reserve = min(_FALLBACK_MIN_RESERVE_S, total_s * 0.35)

        # Slice 28 Phase 2 — adaptive Tier 3 cap for heavy models
        effective_max = _PRIMARY_MAX_TIMEOUT_S
        if model_id and _is_heavy_model(model_id):
            scalar = _envf_or_default(
                "JARVIS_PRIMARY_HEAVY_TTFT_SCALAR",
                _PRIMARY_HEAVY_TTFT_SCALAR_DEFAULT,
            )
            cap = _envf_or_default(
                "JARVIS_PRIMARY_HEAVY_TTFT_CAP_S",
                _PRIMARY_HEAVY_TTFT_CAP_S_DEFAULT,
            )
            effective_max = min(_PRIMARY_MAX_TIMEOUT_S * scalar, cap)

        budget = min(
            total_s * _PRIMARY_BUDGET_FRACTION,
            total_s - fb_reserve,
            effective_max,
        )
        return CandidateGenerator._apply_lane_dilation(max(budget, 0.0), op_id)


# ---------------------------------------------------------------------------
# Defect #4 fix (2026-05-03) — substrate AST pin
# ---------------------------------------------------------------------------


def register_shipped_invariants() -> list:
    """Defect #4 substrate pin. Pins:

      * ``_swallow_task_exception`` helper present (Slice A
        task-leak prevention).
      * ``deadline_exhausted_pre_fallback`` cause string present
        (Slice B pre-fallback budget short-circuit).
      * Every ``asyncio.ensure_future(...)`` / ``asyncio.create_task(...)``
        of provider .generate() OR background-poll coroutines has a
        paired ``add_done_callback(_swallow_task_exception)`` within
        the surrounding statements (catches the regression to
        unprotected task spawns).
      * No exec/eval/compile.
    """
    import ast as _ast
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except ImportError:
        return []

    REQUIRED_FUNCS = (
        "_swallow_task_exception",
        "register_shipped_invariants",
    )
    REQUIRED_LITERALS = (
        "deadline_exhausted_pre_fallback",
        "JARVIS_FALLBACK_MIN_VIABLE_BUDGET_S",
        "_EXPECTED_BACKGROUND_EXC_PATTERNS",
        # Defect #5 (2026-05-03) — read-only cascade reflex lifted
        # into _dispatch_via_sentinel queue branch. Pinned via the
        # cascade-reason marker so a regression that re-removes the
        # reflex (e.g., reverting to the unconditional raise that
        # killed 17/19 BG ops in soak v5) fires the AST pin.
        "Sentinel queue tolerance OVERRIDE",
        "read_only_cost_safe",
    )

    def _validate(
        tree: "_ast.Module", source: str,  # noqa: ARG001
    ) -> tuple:
        violations: list = []
        seen_funcs: set = set()
        # Compute the line range of _swallow_task_exception so we can
        # exclude its body (docstring mentions ensure_future/create_task
        # in documentation, would be false positives).
        helper_line_range: tuple = (-1, -1)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef):
                seen_funcs.add(node.name)
                if node.name == "_swallow_task_exception":
                    helper_line_range = (
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno + 80) or 0,
                    )
            elif isinstance(node, _ast.AsyncFunctionDef):
                seen_funcs.add(node.name)
            elif isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name):
                    if node.func.id in ("exec", "eval", "compile"):
                        violations.append(
                            f"line {getattr(node, 'lineno', '?')}: "
                            f"candidate_generator MUST NOT call "
                            f"{node.func.id}"
                        )
        for fn in REQUIRED_FUNCS:
            if fn not in seen_funcs:
                violations.append(f"missing function {fn!r}")
        for lit in REQUIRED_LITERALS:
            if lit not in source:
                violations.append(
                    f"missing string literal {lit!r}"
                )
        # Pairing pin: every ensure_future / create_task of a
        # provider .generate() must have add_done_callback in the
        # next ~10 source lines. Source-level heuristic (cheap +
        # robust to AST traversal noise from nested coroutines).
        # Skip lines inside _swallow_task_exception body (its
        # docstring mentions the spawn primitives as documentation).
        lines = source.splitlines()
        helper_lo, helper_hi = helper_line_range
        for idx, line in enumerate(lines):
            line_no = idx + 1
            if helper_lo <= line_no <= helper_hi:
                continue  # skip inside helper body
            stripped = line.strip()
            if (
                ("asyncio.ensure_future" in stripped
                 or "asyncio.create_task" in stripped)
                and (".generate(" in stripped
                     or "_background_poll_tier0" in stripped
                     or ("generate" in stripped and "self._tier0" in stripped))
            ):
                # Look for add_done_callback in next 10 source lines.
                window = lines[idx:idx + 10]
                if not any(
                    "add_done_callback" in w
                    and "_swallow_task_exception" in w
                    for w in window
                ):
                    violations.append(
                        f"line {line_no}: ensure_future/create_task "
                        "of .generate() / background-poll must have "
                        "paired add_done_callback(_swallow_task_"
                        "exception) within 10 lines (Defect #4 "
                        "Slice A task-leak prevention)"
                    )
        return tuple(violations)

    target = (
        "backend/core/ouroboros/governance/candidate_generator.py"
    )
    return [
        ShippedCodeInvariant(
            invariant_name="candidate_generator_defect4_substrate",
            target_file=target,
            description=(
                "Defect #4: _swallow_task_exception helper + paired "
                "add_done_callback for every ensure_future/create_task "
                "of provider .generate() / background-poll + "
                "deadline_exhausted_pre_fallback short-circuit cause; "
                "no dynamic-code calls."
            ),
            validate=_validate,
        ),
    ]


def register_flags(registry: Any) -> int:
    """Module-owned FlagRegistry registration.  NEVER raises.

    Covers the free-lane cost interlock and the gateway in-flight seam. Both
    are policy about MONEY and about device accounting respectively, so an
    operator needs to be able to find them by name without reading this file.
    """
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category,
            FlagSpec,
            FlagType,
        )
    except ImportError:
        return 0

    src = "backend/core/ouroboros/governance/candidate_generator.py"
    specs = [
        FlagSpec(
            name="JARVIS_FREE_LANE_POLICY_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Let cost-motivated gates ask about COST rather than hardcode "
                "a route name. When the local lane is the configured one AND "
                "no paid credential is present, per-op marginal cost is ~zero, "
                "so quality-for-money trades (notably skipping swarm chunking) "
                "stop being correct and are re-enabled. Deliberately "
                "conservative: any paid key present means NOT free. Set falsey "
                "to force the pre-existing cost-averse behaviour everywhere."
            ),
            category=Category.ROUTING,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-24)",
        ),
        FlagSpec(
            name="JARVIS_FREE_LANE_CRED_TTL_S",
            type=FlagType.FLOAT,
            default=30.0,
            description=(
                "How stale a credential reading may be before .env is "
                "consulted again, so a key added while the loop is RUNNING "
                "revokes free-lane status without a restart. os.environ is "
                "per-process and load_env_once is idempotent, so without this "
                "re-read a mid-soak key would never be seen. 0 disables the "
                "re-read entirely (boot-time environment only)."
            ),
            category=Category.TIMING,
            source_file=src,
            example="30",
            since="local-lane arc (2026-08-24)",
        ),
        FlagSpec(
            name="JARVIS_GATEWAY_INFLIGHT_UNIFICATION_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Register Phase 3c generations with the InferenceGateway's "
                "in-flight counter. Default TRUE: this is a correctness fix, "
                "and OFF reinstates the blind spot where an advisory pre-warm "
                "sees an idle host and evicts weights out from under a live "
                "stream. Kept as a flag purely so it is revocable without a "
                "revert."
            ),
            category=Category.SAFETY,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-24)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_VRAM_AUTODETECT_ENABLED",
            type=FlagType.BOOL,
            default=False,
            description=(
                "Prefer a MEASURED compute_topology VRAM reading over the GCP "
                "provisioning spec. The spec answers 'what did we ask GCP for', "
                "which is right for a provisioned failover node and wrong for a "
                "workstation serving Ollama. Default OFF -- byte-identical to "
                "the legacy spec-derived path."
            ),
            category=Category.SAFETY,
            source_file=src,
            example="true",
            since="local-inference 32B arc (2026-08-20)",
        ),
        FlagSpec(
            name="JARVIS_BACKGROUND_LOCAL_LANE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Let a BACKGROUND / SPECULATIVE op that would dead-queue on "
                "'background_dw_blocked_by_topology' run on a locally-served "
                "J-Prime instead. Those routes encode 'spend nothing' as a "
                "PROVIDER NAME (DoubleWord), so a purged DW catalog kills the "
                "op even when a $0.00/op lane is serving on the same host. "
                "Requires a reachable endpoint -- evidence, never a flag "
                "asserting one exists -- so a host without a local lane keeps "
                "its byte-identical dead-queue behaviour. Set falsey to "
                "restore the unconditional queue."
            ),
            category=Category.ROUTING,
            source_file=src,
            example="true",
            since="local-lane arc (2026-08-27)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_SIBLING_CANDIDATES",
            type=FlagType.INT,
            default=3,
            description=(
                "How many candidates to draw per op on the local lane. A DPO "
                "preference pair needs TWO answers to one question, so 1 is "
                "the value at which the trajectory corpus can never yield a "
                "pair however long a soak runs; 3 survives one duplicate or "
                "one parse failure. Drawn SEQUENTIALLY (measured on this "
                "host: concurrent n=3 is 1.04x sequential, because the engine "
                "serializes onto one device anyway) and only out of budget "
                "slack the op already has -- the deadline is never extended, "
                "so a tight budget degrades silently to one candidate. "
                "Clamped [1, 8]; 1 restores single-candidate behaviour."
            ),
            category=Category.ROUTING,
            source_file=src,
            example="3",
            since="local-lane arc (2026-08-31)",
        ),
        FlagSpec(
            name="JARVIS_LOCAL_SIBLING_BUDGET_MARGIN",
            type=FlagType.FLOAT,
            default=1.5,
            description=(
                "Safety factor on 'can the op still afford another sibling?'. "
                "The estimate is the PREVIOUS sibling's measured cost and the "
                "next may legitimately run longer, so the margin is generous "
                "on purpose: skipping a sibling costs one training pair, "
                "overrunning costs the op. Floored at 1.0."
            ),
            category=Category.ROUTING,
            source_file=src,
            example="1.5",
            since="local-lane arc (2026-08-31)",
        ),
    ]
    try:
        registry.bulk_register(specs, override=True)
    except Exception:  # noqa: BLE001 -- registration is descriptive, never load-bearing
        return 0
    return len(specs)
