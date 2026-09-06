"""Pre-commit structural-AST guard — the last line against committing a no-op.

Anti-Venom Vector 1 (``generative_quorum_gate.compute_bg_spec_structural_check``)
catches a candidate whose AST fingerprint equals the original AT GENERATE. This
scales that same guard to the live APPLY stage, fail-closed, immediately before
the AutoCommitter runs: it compares the APPLIED working-tree content of every
target file against its committed (HEAD) content using the SAME canonical
signature (``verification.ast_canonical.compute_ast_signature`` — whitespace-,
comment-, and optionally docstring-insensitive), and reports a no-op ONLY when it
can PROVE that every target file is structurally identical — a Quine-class
hallucination or pure formatting churn that must never reach the tree.

The direction of caution is deliberate: a no-op verdict ABORTS a commit, so the
guard demands PROOF before it does. A new file, a structural change, a non-Python
target, an unreadable file, or a syntax error on either side all mean a real
change MAY exist — the commit proceeds. It can only ever suppress a change it has
proven adds nothing. NEVER raises.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("Ouroboros.PrecommitASTGuard")

_ENV_MASTER = "JARVIS_PRECOMMIT_AST_GUARD_ENABLED"
_ENV_GIT_TIMEOUT = "JARVIS_PRECOMMIT_AST_GIT_TIMEOUT_S"


def precommit_guard_enabled() -> bool:
    """``JARVIS_PRECOMMIT_AST_GUARD_ENABLED`` (default **on**). Off -> the guard
    always reports 'not a no-op' (byte-identical legacy commit path)."""
    return os.environ.get(_ENV_MASTER, "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _git_timeout_s() -> float:
    try:
        return max(1.0, float(os.environ.get(_ENV_GIT_TIMEOUT, "15")))
    except (TypeError, ValueError):
        return 15.0


@dataclass(frozen=True)
class PrecommitVerdict:
    """The guard's finding. ``is_noop`` gates the abort; the counts + per-file
    detail are for the ledger and the diff_rejection event."""

    is_noop: bool = False
    files_checked: int = 0
    files_matched: int = 0
    reason: str = ""
    per_file: Tuple[Tuple[str, str], ...] = ()  # (path, outcome)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_noop": self.is_noop,
            "files_checked": self.files_checked,
            "files_matched": self.files_matched,
            "reason": self.reason,
            "per_file": [list(p) for p in self.per_file],
        }


def _is_python(path: str) -> bool:
    return str(path).strip().lower().endswith(".py")


def _repo_rel(repo_root: Path, f: str) -> Optional[str]:
    try:
        s = str(f or "").strip()
        if not s:
            return None
        p = Path(s)
        if p.is_absolute():
            try:
                return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            except (ValueError, OSError):
                return None  # outside the repo — not ours to judge
        return s.replace("\\", "/")
    except Exception:  # noqa: BLE001
        return None


async def _git_show_head(repo_root: Path, rel: str) -> Optional[str]:
    """The file's committed content at HEAD, or ``None`` when it is not tracked
    at HEAD (a NEW file — never a no-op) or git could not answer. NEVER raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "show", f"HEAD:{rel}",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return None
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_git_timeout_s(),
        )
        if proc.returncode != 0:
            return None  # not at HEAD (new file) — the caller treats as changed
        return (out or b"").decode(errors="replace")
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception:  # noqa: BLE001
        return None


def _signature(source: str) -> str:
    """The canonical AST signature — the SAME ``compute_ast_signature`` Anti-Venom
    Vector 1 uses (DRY), but pinned to structural-ONLY equivalence:

      * ``normalize_literals=False`` — a literal IS content. The default collapses
        every constant to a type sentinel, so ``TIMEOUT = 30`` and ``TIMEOUT = 60``
        hash identically; that is right for an ADVISORY on BG/SPEC routes gated on
        a change_description, but a hard pre-commit ABORT must never suppress a
        real single-literal fix, so we keep literal values.
      * ``strip_docstrings=False`` — a docstring IS content (DocStaleness's whole
        yield). Adding one must count as a change, not a no-op.

    What remains ignored is exactly the mandate: whitespace and comments (erased by
    ``ast.parse`` before the dump). Empty for a syntax error / non-Python (the
    caller treats an empty signature as 'cannot prove a no-op'). NEVER raises."""
    try:
        from backend.core.ouroboros.governance.verification.ast_canonical import (
            compute_ast_signature,
        )
        return compute_ast_signature(
            source, normalize_literals=False, strip_docstrings=False,
        ) or ""
    except Exception:  # noqa: BLE001
        return ""


async def check_precommit_structural_noop(
    repo_root: Any, target_files: Sequence[str],
) -> PrecommitVerdict:
    """Prove whether the whole applied change is a structural no-op vs HEAD.

    ``is_noop`` is True ONLY when EVERY target file is a Python file that exists
    at HEAD, parses on both sides, and whose applied signature equals its HEAD
    signature. The FIRST target file that a real change could hide behind — a
    non-``.py`` target, a new file, an unreadable file, a syntax error, or a
    genuine structural change — sets ``is_noop=False`` and the commit proceeds.
    NEVER raises."""
    per_file: List[Tuple[str, str]] = []
    try:
        if not precommit_guard_enabled():
            return PrecommitVerdict(is_noop=False, reason="guard_disabled")
        _root = Path(repo_root)
        _files = list(target_files or ())
        if not _files:
            return PrecommitVerdict(is_noop=False, reason="no_target_files")

        matched = 0
        for f in _files:
            rel = _repo_rel(_root, str(f))
            if rel is None:
                per_file.append((str(f), "outside_repo"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="outside_repo",
                    per_file=tuple(per_file),
                )
            if not _is_python(rel):
                # A non-.py target could be a real change (config/data); we
                # cannot prove a no-op — the commit proceeds.
                per_file.append((rel, "non_python"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="non_python_target",
                    per_file=tuple(per_file),
                )
            abs_p = _root / rel
            try:
                applied = abs_p.read_text(encoding="utf-8", errors="replace") \
                    if abs_p.is_file() else None
            except OSError:
                applied = None
            if applied is None:
                per_file.append((rel, "applied_unreadable"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="applied_unreadable",
                    per_file=tuple(per_file),
                )
            original = await _git_show_head(_root, rel)
            if original is None:
                per_file.append((rel, "new_or_untracked"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="new_file",
                    per_file=tuple(per_file),
                )
            sig_a = _signature(applied)
            sig_o = _signature(original)
            if not sig_a or not sig_o:
                per_file.append((rel, "unparseable"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="syntax_error",
                    per_file=tuple(per_file),
                )
            if sig_a != sig_o:
                per_file.append((rel, "structural_change"))
                return PrecommitVerdict(
                    is_noop=False, files_checked=len(per_file),
                    files_matched=matched, reason="structural_change",
                    per_file=tuple(per_file),
                )
            per_file.append((rel, "noop_match"))
            matched += 1

        # Reached only when EVERY target file proved a structural no-op.
        return PrecommitVerdict(
            is_noop=matched > 0,
            files_checked=len(per_file),
            files_matched=matched,
            reason="all_files_structural_noop" if matched > 0 else "nothing_checked",
            per_file=tuple(per_file),
        )
    except Exception:  # noqa: BLE001 — a guard fault must NOT block a real commit
        logger.debug("[PrecommitASTGuard] check degraded — allowing commit",
                     exc_info=True)
        return PrecommitVerdict(
            is_noop=False, files_checked=len(per_file),
            reason="guard_fault", per_file=tuple(per_file),
        )


__all__ = [
    "PrecommitVerdict",
    "precommit_guard_enabled",
    "check_precommit_structural_noop",
]
