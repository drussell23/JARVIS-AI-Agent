"""Commit-stage fault recovery — the autonomous APPLY path's git-state safety net.

When the AutoCommitter cannot land a verified change — a locked index, a merge
conflict, an unexpected diff rejection — leaving that change half-committed in a
contested working tree is the worst outcome. On such a fault this module:

  1. **Classifies** the fault from the exception (env-tunable patterns).
  2. **Stashes** the op's uncommitted change, SCOPED to its own target files
     (``git stash push -- <files>``) — preserving the work in a recoverable
     stash and cleaning the contested tree, never a destructive ``reset --hard``.
  3. **Emits** a non-blocking ``diff_rejection`` event onto the canonical event
     bus (``ide_observability_stream.publish_task_event``).
  4. **Routes** the precise error trace to the PLAN subagent for a SURGICAL
     re-plan, dispatched as a background task so the daemon's event loop is never
     stalled waiting on it.

Every step is fail-SOFT and NEVER raises into the FSM: a recovery that itself
fails must not turn a non-fatal commit miss into a crashed op.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("Ouroboros.CommitFaultRecovery")

# ---------------------------------------------------------------------------
# Fault taxonomy — DATA, not thresholds. Each maps a canonical fault name to a
# tuple of case-insensitive regexes; the operator can extend any class via
# ``JARVIS_COMMIT_FAULT_PATTERNS_<NAME>`` (comma-separated regexes) without a
# code change.
# ---------------------------------------------------------------------------
_DEFAULT_FAULT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "index_locked": (
        r"index\.lock", r"unable to create '?[^']*\.lock", r"another git process",
        r"could not lock", r"\.git/.*\.lock",
    ),
    "merge_conflict": (
        r"merge conflict", r"\bconflict\b", r"unmerged", r"needs merge",
        r"would be overwritten by merge",
    ),
    "diff_rejected": (
        r"patch does not apply", r"does not apply", r"patch failed",
        r"corrupt patch", r"\brejected\b", r"hunk .* failed",
    ),
    "timeout": (r"timed out", r"timeout", r"cancelled"),
}

_FAULT_OTHER = "other"


def _fault_patterns() -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    for name, defaults in _DEFAULT_FAULT_PATTERNS.items():
        raw = os.environ.get(
            f"JARVIS_COMMIT_FAULT_PATTERNS_{name.upper()}", "",
        ).strip()
        if raw:
            extra = tuple(p.strip() for p in raw.split(",") if p.strip())
            out[name] = defaults + extra
        else:
            out[name] = defaults
    return out


def classify_commit_fault(exc: BaseException) -> str:
    """Canonical fault name from an exception. Matches the exception's message
    (and, for a timeout, its type) against the pattern table, first-match by the
    table's declared order. Unknown -> ``"other"``. NEVER raises."""
    try:
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        text = f"{type(exc).__name__}: {exc}".lower()
        for name, patterns in _fault_patterns().items():
            for pat in patterns:
                try:
                    if re.search(pat, text, re.IGNORECASE):
                        return name
                except re.error:
                    continue
        return _FAULT_OTHER
    except Exception:  # noqa: BLE001
        return _FAULT_OTHER


# ---------------------------------------------------------------------------
# Scoped, fail-soft git stash (no shell — argument arrays, mirroring
# AutoCommitter's Iron-Gate discipline).
# ---------------------------------------------------------------------------


def _git_timeout_s() -> float:
    try:
        return max(1.0, float(os.environ.get("JARVIS_COMMIT_FAULT_GIT_TIMEOUT_S", "15")))
    except (TypeError, ValueError):
        return 15.0


async def _run_git(repo_root: Path, args: Sequence[str]) -> Tuple[int, str]:
    """Run one ``git`` invocation, argument-array only. Returns
    ``(returncode, combined_output)``; ``(-1, reason)`` when git could not even
    be launched or timed out. NEVER raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001 — git missing / spawn failure
        return -1, f"spawn_failed: {exc}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_git_timeout_s())
        return int(proc.returncode or 0), (out or b"").decode(errors="replace")
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return -1, "git_timeout"
    except Exception as exc:  # noqa: BLE001
        return -1, f"git_error: {exc}"


def _repo_relative(repo_root: Path, files: Sequence[str]) -> List[str]:
    """Repo-relative, existing, de-duplicated paths — a stash pathspec must not
    name a path outside the repo or one git will reject. NEVER raises."""
    out: List[str] = []
    seen = set()
    for f in files or ():
        try:
            s = str(f or "").strip()
            if not s:
                continue
            p = Path(s)
            if p.is_absolute():
                try:
                    s = str(p.resolve().relative_to(repo_root.resolve()))
                except (ValueError, OSError):
                    continue  # outside the repo — never stash it
            if s not in seen:
                seen.add(s)
                out.append(s.replace("\\", "/"))
        except Exception:  # noqa: BLE001
            continue
    return out


async def stash_workspace(
    repo_root: Path, target_files: Sequence[str], op_id: str,
) -> Tuple[bool, str]:
    """Stash the op's uncommitted change, SCOPED to its own target files, so the
    contested tree is cleaned without discarding the work (it lands in a named
    stash entry, recoverable with ``git stash list`` / ``pop``). Returns
    ``(stashed, detail)``. Fail-SOFT: a locked index (the very fault that may
    have caused this) makes the stash itself fail — that is reported, never
    raised, and the change is simply left in place. NEVER raises."""
    try:
        rel = _repo_relative(Path(repo_root), target_files)
        if not rel:
            return False, "no_scoped_files"
        _label = f"ov-commit-fault:{str(op_id)[:32]}:{int(time.time())}"
        rc, out = await _run_git(
            Path(repo_root),
            ["stash", "push", "--include-untracked", "-m", _label, "--", *rel],
        )
        if rc == 0:
            # "No local changes to save" is rc=0 with a message — report it
            # honestly rather than claim a stash that did not happen.
            if "no local changes" in out.lower():
                return False, "nothing_to_stash"
            return True, _label
        return False, f"stash_rc={rc}: {out.strip()[:200]}"
    except Exception:  # noqa: BLE001
        logger.debug("[CommitFaultRecovery] stash degraded", exc_info=True)
        return False, "stash_exception"


# ---------------------------------------------------------------------------
# Non-blocking DiffRejection event + async PLAN re-plan routing.
# ---------------------------------------------------------------------------

_DIFF_REJECTION_EVENT = "diff_rejection"


def emit_diff_rejection(
    op_id: str, fault: str, detail: str, target_files: Sequence[str],
    *, stashed: bool, stash_ref: str,
) -> Optional[str]:
    """Publish a ``diff_rejection`` event on the canonical bus — the operator and
    any IDE stream see the contested commit the moment it happens. Best-effort:
    a disabled stream or a broker fault returns None, never raises."""
    try:
        from backend.core.ouroboros.governance.ide_observability_stream import (
            publish_task_event,
        )
        return publish_task_event(
            _DIFF_REJECTION_EVENT,
            str(op_id or ""),
            {
                "fault": fault,
                "detail": str(detail or "")[:600],
                "target_files": [str(f) for f in (target_files or ())][:12],
                "stashed": bool(stashed),
                "stash_ref": str(stash_ref or ""),
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("[CommitFaultRecovery] diff_rejection emit degraded", exc_info=True)
        return None


def route_replan_to_plan(
    orch: Any, ctx: Any, fault: str, detail: str,
) -> bool:
    """Route the precise fault to the PLAN subagent for a SURGICAL re-plan,
    dispatched as a BACKGROUND task so the daemon's event loop is never stalled
    waiting on it. Returns True when the task was scheduled. A missing subagent
    orchestrator or a single-file op (no DAG to re-plan) is a clean skip.
    NEVER raises."""
    try:
        _subs = getattr(orch, "_subagent_orchestrator", None)
        if _subs is None or not hasattr(_subs, "dispatch_plan"):
            return False
        _targets = tuple(str(f) for f in (getattr(ctx, "target_files", ()) or ()))
        if len(_targets) < 2:
            # dispatch_plan is a DAG builder — a single-file op has nothing to
            # re-plan. The re-plan lane is for multi-file work.
            return False
        _desc = (
            f"{getattr(ctx, 'description', '') or ''}\n\n"
            f"[RE-PLAN after commit fault={fault}] The previous apply was "
            f"reverted because the commit could not land. Re-plan a diff that "
            f"avoids this fault. Fault detail: {str(detail or '')[:400]}"
        )
        _risk = str(getattr(getattr(ctx, "risk_tier", None), "name", "") or "")

        async def _do_replan() -> None:
            try:
                _res = await _subs.dispatch_plan(
                    ctx,
                    op_description=_desc,
                    target_files=_targets,
                    primary_repo=str(getattr(ctx, "primary_repo", "") or "jarvis"),
                    risk_tier=_risk,
                )
                _status = getattr(getattr(_res, "status", None), "value", _res)
                logger.info(
                    "[CommitFaultRecovery] PLAN re-plan dispatched op=%s "
                    "fault=%s status=%s (surgical re-plan, non-blocking)",
                    getattr(ctx, "op_id", "?"), fault, _status,
                )
                emit_diff_rejection(
                    getattr(ctx, "op_id", ""), fault,
                    f"replan_status={_status}", _targets,
                    stashed=False, stash_ref="",
                )
            except Exception:  # noqa: BLE001 — a background re-plan never crashes
                logger.debug(
                    "[CommitFaultRecovery] background re-plan degraded",
                    exc_info=True,
                )

        # Fire-and-forget on the running loop; keep a reference so it is not GC'd.
        _task = asyncio.ensure_future(_do_replan())
        _pending = getattr(orch, "_commit_replan_tasks", None)
        if _pending is None:
            _pending = set()
            try:
                setattr(orch, "_commit_replan_tasks", _pending)
            except Exception:  # noqa: BLE001
                _pending = None
        if _pending is not None:
            _pending.add(_task)
            _task.add_done_callback(_pending.discard)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("[CommitFaultRecovery] replan routing degraded", exc_info=True)
        return False


async def recover_from_commit_fault(
    orch: Any, ctx: Any, exc: BaseException,
) -> Dict[str, Any]:
    """The single entry point the orchestrator's commit handler calls. Classify
    -> stash -> emit -> route re-plan, each fail-soft. Returns a summary dict
    (fault / stashed / stash_ref / replanned) for the ledger. NEVER raises."""
    fault = classify_commit_fault(exc)
    detail = f"{type(exc).__name__}: {exc}"[:600]
    _targets = tuple(str(f) for f in (getattr(ctx, "target_files", ()) or ()))
    stashed, stash_ref = False, ""
    try:
        _root = getattr(getattr(orch, "_config", None), "project_root", None)
        if _root is not None and _targets:
            stashed, stash_ref = await stash_workspace(
                Path(_root), _targets, str(getattr(ctx, "op_id", "") or ""),
            )
    except Exception:  # noqa: BLE001
        logger.debug("[CommitFaultRecovery] stash step degraded", exc_info=True)
    emit_diff_rejection(
        getattr(ctx, "op_id", ""), fault, detail, _targets,
        stashed=stashed, stash_ref=stash_ref,
    )
    replanned = route_replan_to_plan(orch, ctx, fault, detail)
    logger.warning(
        "[CommitFaultRecovery] op=%s commit fault=%s stashed=%s replanned=%s "
        "(change reverted from the contested tree; daemon not stalled)",
        getattr(ctx, "op_id", "?"), fault, stashed, replanned,
    )
    return {
        "fault": fault, "stashed": stashed, "stash_ref": stash_ref,
        "replanned": replanned, "detail": detail,
    }


__all__ = [
    "classify_commit_fault",
    "stash_workspace",
    "emit_diff_rejection",
    "route_replan_to_plan",
    "recover_from_commit_fault",
]
