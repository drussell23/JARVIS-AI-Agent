"""A Claude-Code-style one-line recap when an operation finishes.

CC closes a turn with a compact summary — ``✻ Crunched for 2m 14s · 3 tools
used · done 11:40 PM``. O+V is autonomous, so its unit is the OPERATION: when
one reaches a terminal state, the cockpit shows a recap synthesised from that
op's execution ledger — how long it took, how many tools it ran, how many
tokens it generated, and when it finished.

Pure composition. The transport draws the line; the orchestrator's terminal
seam supplies the counts (from ``ctx.generation``) and the state. NEVER
raises — a recap is a courtesy and must never touch the FSM or the render.

Resilience:
  * An ABORTED op (operator ``Ctrl+C`` / cancellation) renders
    ``✻ Aborted after 2m 14s`` — no verb, no fabricated counts.
  * A FAILED op renders ``✗ … Failed after 2m 14s`` (the outcome is the
    color; the recap carries the duration and whatever ran).
  * Every segment is CONDITIONAL: zero tools or zero tokens simply drop
    their segment rather than printing ``0 tools used``.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger("Ouroboros.OpRecap")

#: Past-tense verbs — the recap's whimsical word, O+V's analogue of Claude
#: Code's ``Crunched`` / ``Sautéed``. DATA (a vocabulary), not a threshold; a
#: comma list in ``JARVIS_RECAP_VERBS`` overrides it. One is chosen per op so
#: consecutive recaps read distinctly.
_RECAP_VERBS = (
    "Crunched", "Wrangled", "Threaded", "Forged", "Distilled",
    "Landed", "Composed", "Shaped",
)


def recap_verbs() -> tuple:
    raw = os.environ.get("JARVIS_RECAP_VERBS", "").strip()
    if raw:
        words = tuple(w.strip() for w in raw.split(",") if w.strip())
        if words:
            return words
    return _RECAP_VERBS


def recap_verb(op_id: object) -> str:
    """A stable-per-op past-tense verb. NEVER raises."""
    words = recap_verbs()
    try:
        h = int(hashlib.sha1(str(op_id or "op").encode()).hexdigest()[:8], 16)
        return words[h % len(words)]
    except Exception:  # noqa: BLE001
        return words[0]


def recap_enabled() -> bool:
    """Master — default ON. ``JARVIS_OP_RECAP_ENABLED=false`` silences it."""
    return os.environ.get(
        "JARVIS_OP_RECAP_ENABLED", "1",
    ).strip().lower() not in ("0", "false", "no", "off")


def _clock(now: Optional[datetime.datetime] = None) -> str:
    """``11:40 PM`` — 12-hour, portable (no ``%-I``). NEVER raises."""
    try:
        n = now or datetime.datetime.now()
        h = n.hour % 12 or 12
        ap = "AM" if n.hour < 12 else "PM"
        return f"{h}:{n.minute:02d} {ap}"
    except Exception:  # noqa: BLE001
        return ""


def _mark() -> str:
    try:
        from backend.core.ouroboros.ui import theme
        return theme.mark("recap") or "*"
    except Exception:  # noqa: BLE001
        return "*"


def _fmt_tokens(tokens: int) -> str:
    try:
        from backend.core.ouroboros.battle_test.stream_renderer import _fmt_tokens as _f
        return _f(tokens)
    except Exception:  # noqa: BLE001
        return str(max(0, int(tokens or 0)))


def compose_recap(*, elapsed: str, verb: str = "", tools: int = 0,
                  tokens: int = 0, done_at: Optional[str] = None,
                  aborted: bool = False, failed: bool = False) -> str:
    """The recap line. ``elapsed`` is the already-formatted duration (the
    transport owns that clock). Pure; NEVER raises."""
    try:
        mark = _mark()
        el = str(elapsed or "").strip() or "0s"
        if aborted:
            return f"{mark} Aborted after {el}"
        when = _clock() if done_at is None else str(done_at)
        lead = f"Failed after {el}" if failed else f"{verb or 'Done'} for {el}"
        parts = [lead]
        n = max(0, int(tools or 0))
        if n > 0:
            parts.append(f"{n} tool{'s' if n != 1 else ''} used")
        tk = max(0, int(tokens or 0))
        if tk > 0:
            parts.append(f"↑ {_fmt_tokens(tk)} tokens")
        if when:
            parts.append(f"done {when}")
        return f"{mark} " + " · ".join(parts)
    except Exception:  # noqa: BLE001
        logger.debug("[OpRecap] compose degraded", exc_info=True)
        return ""


def tool_count(generation: object) -> int:
    """How many tools this op ran — the length of the execution record, else
    the edit history. NEVER raises."""
    try:
        recs = getattr(generation, "tool_execution_records", ()) or ()
        if recs:
            return len(recs)
        return len(getattr(generation, "venom_edit_history", ()) or ())
    except Exception:  # noqa: BLE001
        return 0


def output_tokens(generation: object) -> int:
    try:
        return max(0, int(getattr(generation, "total_output_tokens", 0) or 0))
    except Exception:  # noqa: BLE001
        return 0


__all__ = [
    "compose_recap", "output_tokens", "recap_enabled", "recap_verb",
    "recap_verbs", "tool_count",
]
