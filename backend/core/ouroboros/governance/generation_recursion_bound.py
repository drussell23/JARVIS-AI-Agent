"""Unified generation recursion bound — one shared, self-expiring ceiling on the
total RECOVERY depth an op may spend across every generation-recovery loop (the
Iron-Gate ``GENERATE_RETRY`` loop AND the local syntax-repair), so the two
scattered per-loop budgets compose under a single tunable knob and can never sum
into an unbounded, event-loop-starving retry storm.

WHY A CEILING, NOT A TARGET (Root-Cause): the local syntax-repair is DELIBERATELY
one-shot (``candidate_generator._local_dispatch_with_syntax_repair`` — "Two would
be a loop that spends the op's whole budget re-reading the same file; and if a
model cannot fix a named ``unexpected indent`` on the second attempt, a third
will not help"), and the Iron-Gate default is 1 retry. This bound does NOT force
either loop to N — it caps their SUM. It is strictly ADDITIVE: it can only make
an op exhaust EARLIER, never grant a retry a native bound already refused. With
the default it is a pure safety net + observability, byte-identical on the happy
path (max_generate_retries=1 + one-shot syntax = depth 2 < bound 3).

ASYNC-YIELD-SAFE, SELF-EXPIRING STATE (the "no orphaned retry loops" mandate):
depth lives in a process-global, ``op_id``-keyed ledger guarded by a lock (the
same pattern the attribution module's ``_MAP_CACHE`` uses), so it survives the
many ``await`` yields inside one op's generation without leaking across ops.
Every entry carries a monotonic touch time and expires after a TTL — a paused,
cancelled, or crashed op leaves no orphaned counter to mis-fire a later attempt,
and a process restart starts clean. NEVER raises.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger("Ouroboros.GenerationRecursionBound")

_ENV_MASTER = "JARVIS_GENERATION_RECURSION_BOUND_ENABLED"
_ENV_BOUND = "JARVIS_GENERATION_RECURSION_BOUND"
_ENV_TTL = "JARVIS_GENERATION_RECURSION_TTL_S"
_EXHAUSTED_EVENT = "generation_exhausted"


def recursion_bound_enabled() -> bool:
    """Master (default ON). OFF -> ``enter_recovery`` is permissive (every caller
    is byte-identical to its legacy per-loop bound)."""
    return os.environ.get(_ENV_MASTER, "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def recursion_bound() -> int:
    """Max RECOVERY attempts (retries) an op may spend across ALL generation-
    recovery loops combined. Default 3; clamped >=1. Env
    ``JARVIS_GENERATION_RECURSION_BOUND``."""
    try:
        return max(1, int(os.environ.get(_ENV_BOUND, "3")))
    except (TypeError, ValueError):
        return 3


def _ttl_s() -> float:
    try:
        return max(1.0, float(os.environ.get(_ENV_TTL, "900")))
    except (TypeError, ValueError):
        return 900.0


@dataclass(frozen=True)
class RecoveryToken:
    """The result of recording one recovery attempt. Callers gate their retry on
    ``at_ceiling``: True means the op has spent its shared budget and this
    attempt must be refused (fail-closed)."""

    op_id: str
    depth: int          # 1-based count of recovery attempts recorded for this op
    bound: int
    at_ceiling: bool    # True => this attempt exceeds the budget; DENY it
    remaining: int


# op_id -> (depth, last_touch_monotonic). Process-global by design: it must
# survive await yields within one op, and a restart must clear it entirely.
_LEDGER: Dict[str, Tuple[int, float]] = {}
_LOCK = threading.Lock()


def _sweep_locked(now: float, ttl: float) -> None:
    if not _LEDGER:
        return
    dead = [k for k, (_d, ts) in _LEDGER.items() if now - ts > ttl]
    for k in dead:
        _LEDGER.pop(k, None)


def enter_recovery(op_id: str) -> RecoveryToken:
    """Record ONE recovery attempt for *op_id*; report whether it exceeds the
    shared budget.

    Permissive when disabled or ``op_id`` is empty (``at_ceiling=False``) so the
    caller stays byte-identical to its legacy per-loop bound. Fail-CLOSED on any
    internal error while ENABLED (``at_ceiling=True``) — an unbounded retry loop
    is the fatal case this module exists to prevent. NEVER raises."""
    bound = recursion_bound()
    try:
        if not recursion_bound_enabled() or not str(op_id or "").strip():
            return RecoveryToken(str(op_id or ""), 0, bound, False, bound)
        now = time.monotonic()
        ttl = _ttl_s()
        with _LOCK:
            _sweep_locked(now, ttl)
            depth = _LEDGER.get(op_id, (0, now))[0] + 1
            _LEDGER[op_id] = (depth, now)
        return RecoveryToken(
            op_id=op_id,
            depth=depth,
            bound=bound,
            at_ceiling=depth > bound,
            remaining=max(0, bound - depth),
        )
    except Exception:  # noqa: BLE001 — fail-closed: deny further recursion
        logger.debug(
            "[GenerationRecursionBound] enter_recovery degraded — fail-closed",
            exc_info=True,
        )
        return RecoveryToken(str(op_id or ""), bound + 1, bound, True, 0)


def peek_depth(op_id: str) -> int:
    """Current recorded recovery depth for *op_id* without incrementing. 0 when
    absent. NEVER raises."""
    try:
        with _LOCK:
            return _LEDGER.get(op_id, (0, 0.0))[0]
    except Exception:  # noqa: BLE001
        return 0


def reset(op_id: str) -> None:
    """Drop an op's recovery counter on any terminal outcome. Idempotent; NEVER
    raises. (TTL expiry is the backstop for ops that never call this.)"""
    try:
        with _LOCK:
            _LEDGER.pop(op_id, None)
    except Exception:  # noqa: BLE001
        pass


def sweep_expired() -> int:
    """Force a TTL sweep; returns how many orphaned counters were reclaimed.
    NEVER raises."""
    try:
        now = time.monotonic()
        ttl = _ttl_s()
        with _LOCK:
            before = len(_LEDGER)
            _sweep_locked(now, ttl)
            return before - len(_LEDGER)
    except Exception:  # noqa: BLE001
        return 0


def emit_generation_exhausted(
    op_id: str, *, phase: str, depth: int, bound: int, detail: str = "",
) -> Optional[str]:
    """Publish a ``generation_exhausted`` event on the canonical bus so the
    operator and any IDE stream see the fail-closed ceiling the moment it fires.
    Best-effort (a disabled stream / broker fault returns None); NEVER raises."""
    try:
        from backend.core.ouroboros.governance.ide_observability_stream import (
            publish_task_event,
        )
        return publish_task_event(
            _EXHAUSTED_EVENT,
            str(op_id or ""),
            {
                "phase": str(phase or ""),
                "depth": int(depth),
                "bound": int(bound),
                "detail": str(detail or "")[:600],
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "[GenerationRecursionBound] exhausted emit degraded", exc_info=True,
        )
        return None


__all__ = [
    "RecoveryToken",
    "recursion_bound_enabled",
    "recursion_bound",
    "enter_recovery",
    "peek_depth",
    "reset",
    "sweep_expired",
    "emit_generation_exhausted",
]
