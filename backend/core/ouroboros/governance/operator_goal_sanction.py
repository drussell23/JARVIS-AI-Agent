"""Operator goal sanction + scoped injection — the cockpit's bridge to the
self-modification cage's cryptographic authorization.

The cage (``risk_engine._self_protection_verdict`` →
``delegated_provenance.verify_provenance_claim``) refuses any governance edit
that cannot prove it traces to an OPERATOR-signed roadmap goal. That control
is correct and this module does not weaken it: it makes the *legitimate* path
— author a signed, file-scoped goal, then inject a scoped op that references it
— reachable from the live cockpit instead of only from an out-of-band CLI.

## What this reuses (zero parallel paths)

* **Authoring** rides ``strategy_signer.sign_roadmap_doc`` (the ONE signer the
  operator CLI uses) and is VERIFIED back through ``roadmap_reader.read_roadmap``
  before the live document is replaced — a roadmap that does not verify is
  never written, because the cage would then refuse every op against it and the
  reason would be three layers from the cause. This is the exact discipline
  ``scripts/sanction_goal.py`` follows; that script now calls
  :func:`author_and_sign_goal` so there is one authoring path, not two.
* **The scope key is ``target_files``** — the key ``roadmap_reader.
  _parse_goal_entry`` actually parses. (The pre-refactor ``sanction_goal.py``
  wrote ``files``, which the reader dropped, so its goals parsed with empty
  scope and the cage refused them ``goal_has_no_scope``. Writing the parsed key
  is the root-cause fix.)
* **Injection** mints the claim pointer with ``delegated_provenance.
  claim_for_goal`` and builds the envelope with ``intent_envelope.make_envelope``
  under ``source="roadmap"`` — byte-identical in shape to the envelope
  ``WorkOrderSensor`` emits, so a cockpit-injected goal flows through the SAME
  classify → route → GENERATE → Iron Gate → GATE → VERIFY pipeline as every
  other signal, carrying no elevated authority. The claim grants nothing by
  itself; ``verify_provenance_claim`` re-derives signature, goal existence,
  freshness and per-file scope from ground truth at classify.

## Security invariants (bulletproof, fail-closed)

1. **Operator-initiated only.** Signing is reachable solely from the cockpit
   ``/goal`` verb (a human keystroke). The autonomous intake path has no route
   to it, so the organism can never sign its own authorizations.
2. **The secret is the authorization.** ``author_and_sign_goal`` refuses when
   ``JARVIS_ROADMAP_READER_HMAC_SECRET`` is unset — an unsigned roadmap is
   worse than none. In-cockpit signing is additionally gated by
   :func:`cockpit_signing_enabled` (``JARVIS_COCKPIT_GOAL_SIGNING_ENABLED``,
   default on) so an operator can disable it and fall back to the pure
   inject-a-pre-signed-goal path.
3. **A verified claim caps risk at APPROVAL_REQUIRED** (never auto-apply) — the
   sanctioned op runs the full pipeline and then WAITS for the operator. This
   module adds no bypass of that ceiling.
4. **NEVER raises** across any public entry point. Every fault is a typed
   refusal the caller surfaces, never an exception into the REPL or intake.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("Ouroboros.OperatorGoalSanction")

#: Env that carries the operator HMAC secret. Read through
#: ``roadmap_reader.hmac_secret`` so this module and the cage read the SAME
#: source (a second getattr chain is how the two would drift apart).
_ENV_SIGNING_MASTER = "JARVIS_COCKPIT_GOAL_SIGNING_ENABLED"
_ENV_OPERATOR_ID = "JARVIS_OPERATOR_ID"


def cockpit_signing_enabled() -> bool:
    """Whether the cockpit ``/goal`` verb may AUTHOR + sign a new roadmap goal
    in-process. Default ON — the operator asked for the sanction workflow to be
    reachable from the cockpit — but an operator who wants signing to happen
    only out-of-band (via ``scripts/sanction_goal.py``) sets this to ``false``
    and the verb still injects PRE-signed goals. NEVER raises."""
    raw = os.environ.get(_ENV_SIGNING_MASTER, "").strip().lower()
    if raw == "":
        return True
    return raw not in ("0", "false", "no", "off")


def _hmac_secret() -> str:
    """The operator secret, from the one place the cage reads it. Empty when
    unset — the caller REFUSES rather than emitting an unsigned roadmap."""
    try:
        from backend.core.ouroboros.governance.roadmap_reader import hmac_secret
        return (hmac_secret() or "").strip()
    except Exception:  # noqa: BLE001
        return (os.environ.get("JARVIS_ROADMAP_READER_HMAC_SECRET", "") or "").strip()


def _roadmap_path(path_override: Optional[Path] = None) -> Path:
    """The live roadmap the cage consults. Resolved the same way the reader
    resolves it (``JARVIS_ROADMAP_PATH`` then the ``.jarvis`` default) so the
    document this module writes is the document the cage reads."""
    if path_override is not None:
        return Path(path_override)
    raw = (os.environ.get("JARVIS_ROADMAP_PATH", "") or "").strip()
    if raw:
        return Path(raw)
    return _repo_root() / ".jarvis" / "roadmap.yaml"


def _repo_root() -> Path:
    # This module lives at backend/core/ouroboros/governance/ — five parents up
    # is the repo root. Resolved structurally, never hardcoded.
    return Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Result envelopes — typed, never exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SanctionResult:
    """Outcome of authoring + signing a goal. ``ok`` gates injection."""

    ok: bool
    goal_id: str = ""
    reason: str = ""          # machine code on refusal ("secret_unset", …)
    detail: str = ""          # human line (the reader's verdict on success)
    roadmap_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "goal_id": self.goal_id,
            "reason": self.reason, "detail": self.detail,
            "roadmap_path": self.roadmap_path,
        }


@dataclass(frozen=True)
class GoalSpec:
    """The operator's declaration of one unit of sanctioned work. Pure state —
    every field is validated at authoring; nothing here carries authority."""

    goal_id: str
    title: str
    description: str
    target_files: Tuple[str, ...]
    target_symbols: Tuple[str, ...] = ()
    priority: str = "high"
    success_criteria: str = ""
    max_duration_s: int = 0
    note: str = ""

    def to_entry(self) -> Dict[str, Any]:
        """The roadmap-goal dict, in the schema ``roadmap_reader.
        _parse_goal_entry`` parses — ``target_files`` (NOT ``files``), and
        ``target_symbol`` singular OR plural, both of which the reader
        accepts. Empty optionals are omitted so the document stays minimal."""
        entry: Dict[str, Any] = {
            "id": self.goal_id,
            "title": self.title,
            "description": self.description or self.title,
            "target_files": [str(f) for f in self.target_files],
        }
        if self.target_symbols:
            entry["target_symbols"] = [str(s) for s in self.target_symbols]
        if self.priority:
            entry["priority"] = self.priority
        if self.success_criteria:
            entry["success_criteria"] = self.success_criteria
        if self.max_duration_s and self.max_duration_s > 0:
            entry["max_duration_s"] = int(self.max_duration_s)
        return entry


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------


def normalize_target_files(
    raw: Sequence[str], *, repo_root: Optional[Path] = None,
) -> Tuple[str, ...]:
    """Repo-relative, de-duplicated, order-preserving target paths.

    Accepts absolute or relative, forward or back slashes, and paths inside or
    outside the repo root; anything that resolves under the root is returned
    relative to it (the shape the roadmap + the scope check both expect), and
    anything else is kept as the operator typed it (the scope check will refuse
    it loudly rather than silently mis-scope). NEVER raises."""
    root = (repo_root or _repo_root()).resolve()
    out: List[str] = []
    seen = set()
    for item in raw or ():
        try:
            s = str(item or "").strip().replace("\\", "/")
            if not s:
                continue
            p = Path(s)
            if p.is_absolute():
                try:
                    s = str(p.resolve().relative_to(root)).replace("\\", "/")
                except (ValueError, OSError):
                    s = str(p).replace("\\", "/")
            if s not in seen:
                seen.add(s)
                out.append(s)
        except Exception:  # noqa: BLE001 — a bad path never breaks the batch
            continue
    return tuple(out)


# ---------------------------------------------------------------------------
# Phase 1a — author + sign (the shared authoring path)
# ---------------------------------------------------------------------------


def author_and_sign_goal(
    spec: GoalSpec,
    *,
    path_override: Optional[Path] = None,
    operator_id: Optional[str] = None,
    dry_run: bool = False,
) -> SanctionResult:
    """Append ``spec`` to the operator roadmap, sign it, VERIFY it through the
    reader the cage consults, and only then replace the live document.

    Root-cause discipline (mirrors ``sanction_goal.py``, now shared):
      * refuse an unset secret — the signature IS the authorization;
      * refuse a duplicate id — an edit must never be mistaken for the original;
      * refuse an unscoped goal — a goal with no ``target_files`` authorizes
        every file, the one shape that makes the cage meaningless;
      * write to a sibling, verify, atomic-replace, keep a ``.prev`` backup.

    NEVER raises — every failure is a typed :class:`SanctionResult`."""
    try:
        if not spec.goal_id.strip() or not spec.title.strip():
            return SanctionResult(False, reason="missing_id_or_title")
        if not spec.target_files:
            return SanctionResult(
                False, goal_id=spec.goal_id, reason="unscoped_goal",
                detail="a goal with no target_files authorizes every file",
            )
        secret = _hmac_secret()
        if not secret:
            return SanctionResult(
                False, goal_id=spec.goal_id, reason="secret_unset",
                detail="JARVIS_ROADMAP_READER_HMAC_SECRET is unset; the "
                       "signature is the authorization",
            )

        path = _roadmap_path(path_override)
        doc = _load_roadmap(path)
        goals: List[Dict[str, Any]] = list(doc.get("goals") or [])
        if any(str(g.get("id")) == spec.goal_id for g in goals):
            return SanctionResult(
                False, goal_id=spec.goal_id, reason="duplicate_id",
                detail=f"goal {spec.goal_id!r} already exists — withdraw it "
                       f"first so an edit is never mistaken for the original",
            )
        goals.append(spec.to_entry())
        doc["goals"] = goals
        doc["signed_at"] = datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        )
        doc.setdefault("version", 1)
        doc.setdefault("authority", "operator_directed")
        doc.setdefault("source", "operator_directed_agent_signed")
        doc["operator_id"] = (
            (operator_id or os.environ.get(_ENV_OPERATOR_ID, "")).strip()
            or doc.get("operator_id")
            or "operator-unspecified"
        )
        if spec.note:
            doc["note"] = spec.note

        signed = _sign(doc, secret)
        if not signed.get("signature"):
            return SanctionResult(
                False, goal_id=spec.goal_id, reason="signing_failed",
                detail="signing produced no signature",
            )

        import yaml  # noqa: PLC0415
        rendered = yaml.safe_dump(signed, sort_keys=False, allow_unicode=True)
        if dry_run:
            return SanctionResult(
                True, goal_id=spec.goal_id, reason="dry_run",
                detail=rendered, roadmap_path=str(path),
            )

        ok, detail = _write_verified(path, rendered)
        if not ok:
            return SanctionResult(
                False, goal_id=spec.goal_id, reason="verify_failed",
                detail=detail, roadmap_path=str(path),
            )
        # Fresh authorization → drop the verifier's memo so the very next
        # classify sees this goal rather than a cached pre-write roadmap.
        _reset_provenance_cache()
        return SanctionResult(
            True, goal_id=spec.goal_id, reason="signed_and_verified",
            detail=detail, roadmap_path=str(path),
        )
    except Exception:  # noqa: BLE001 — a fault is a refusal, never a raise
        logger.debug("[OperatorGoalSanction] author degraded", exc_info=True)
        return SanctionResult(
            False, goal_id=getattr(spec, "goal_id", ""),
            reason="author_fault",
        )


def withdraw_goal(
    goal_id: str, *, path_override: Optional[Path] = None,
) -> SanctionResult:
    """Remove a goal by id and RE-SIGN + verify the remaining document — a
    withdrawal is an authoring act and leaves a signed roadmap, never a torn
    one. NEVER raises."""
    try:
        gid = str(goal_id or "").strip()
        if not gid:
            return SanctionResult(False, reason="missing_id")
        secret = _hmac_secret()
        if not secret:
            return SanctionResult(False, goal_id=gid, reason="secret_unset")
        path = _roadmap_path(path_override)
        doc = _load_roadmap(path)
        goals = list(doc.get("goals") or [])
        kept = [g for g in goals if str(g.get("id")) != gid]
        if len(kept) == len(goals):
            return SanctionResult(False, goal_id=gid, reason="goal_not_found")
        doc["goals"] = kept
        doc["signed_at"] = datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        )
        signed = _sign(doc, secret)
        import yaml  # noqa: PLC0415
        rendered = yaml.safe_dump(signed, sort_keys=False, allow_unicode=True)
        ok, detail = _write_verified(path, rendered)
        if not ok:
            return SanctionResult(
                False, goal_id=gid, reason="verify_failed", detail=detail,
            )
        _reset_provenance_cache()
        return SanctionResult(
            True, goal_id=gid, reason="withdrawn", detail=detail,
            roadmap_path=str(path),
        )
    except Exception:  # noqa: BLE001
        logger.debug("[OperatorGoalSanction] withdraw degraded", exc_info=True)
        return SanctionResult(False, goal_id=goal_id, reason="withdraw_fault")


# ---------------------------------------------------------------------------
# Phase 1b — scoped injection envelope (reuses claim + make_envelope)
# ---------------------------------------------------------------------------


def build_scoped_envelope(
    *,
    goal_id: str,
    description: str,
    target_files: Sequence[str],
    repo: str = "",
    urgency: str = "",
    confidence: float = 0.95,
) -> Optional[Any]:
    """The ``IntentEnvelope`` that injects a sanctioned goal as one scoped op.

    Byte-identical in shape to ``WorkOrderSensor``'s: ``source="roadmap"`` plus
    an ``evidence.provenance`` claim POINTER at the signed goal. Returns None
    (fail-closed) when the goal cannot be found in the VERIFIED roadmap — the
    injection then does not happen and the cockpit says why, rather than
    submitting an op the cage will silently block. NEVER raises."""
    try:
        gid = str(goal_id or "").strip()
        targets = normalize_target_files(target_files)
        if not gid or not targets:
            return None
        from backend.core.ouroboros.governance.delegated_provenance import (
            claim_for_goal,
            verify_provenance_claim,
        )
        from backend.core.ouroboros.governance.intake.intent_envelope import (
            make_envelope,
        )
        claim = claim_for_goal(gid)
        if claim is None:
            # Feature off or empty id — no pointer to present, so a governance
            # target would be refused. Fail closed.
            return None
        # Fail-closed at INJECTION, not silently at classify: only build the
        # envelope if the cage's OWN verifier accepts this claim for THESE
        # files. A goal that was never signed, expired, or whose scope does
        # not cover a target is rejected here so the cockpit can say why,
        # rather than dispatching an op that dies three phases downstream.
        _v = verify_provenance_claim(claim, source="roadmap", file_strs=targets)
        if not getattr(_v, "valid", False):
            logger.info(
                "[OperatorGoalSanction] inject refused goal=%s reason=%s",
                gid, getattr(_v, "reason", "?"),
            )
            return None
        evidence: Dict[str, Any] = {
            "operator_goal": True,
            "goal_id": gid,
            "provenance": claim,
        }
        return make_envelope(
            source="roadmap",
            description=str(description or "")[:2000],
            target_files=targets,
            repo=str(repo or _default_repo()),
            confidence=float(confidence),
            urgency=str(urgency or _default_urgency()),
            evidence=evidence,
            requires_human_ack=False,  # the operator already authorized it
        )
    except Exception:  # noqa: BLE001
        logger.debug("[OperatorGoalSanction] envelope degraded", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Internals — every one fail-soft
# ---------------------------------------------------------------------------


def _load_roadmap(path: Path) -> Dict[str, Any]:
    """Load the roadmap, or the empty-but-well-formed shell a first goal on a
    fresh box needs. Refuses a malformed document rather than appending into
    garbage."""
    import yaml  # noqa: PLC0415
    if not path.is_file():
        return {
            "version": 1,
            "authority": "operator_directed",
            "source": "operator_directed_agent_signed",
            "goals": [],
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    data.setdefault("goals", [])
    if not isinstance(data["goals"], list):
        raise ValueError(f"{path} has a non-list 'goals'")
    return data


def _sign(doc: Dict[str, Any], secret: str) -> Dict[str, Any]:
    from backend.core.ouroboros.governance.strategy_signer import (
        sign_roadmap_doc,
    )
    return sign_roadmap_doc(doc, secret)


def _write_verified(path: Path, rendered: str) -> Tuple[bool, str]:
    """Write to a sibling, VERIFY it through ``roadmap_reader`` (the component
    the cage actually consults, not our own assumptions), atomic-replace on
    success, keep a ``.prev`` backup. A document that does not verify never
    becomes the live one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(rendered, encoding="utf-8")
    ok, detail = _verify(tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        return False, detail
    if path.is_file():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".prev"))
        except OSError:
            pass
    os.replace(tmp, path)
    return True, detail


def _verify(path: Path) -> Tuple[bool, str]:
    """Read the candidate back through the reader — signing and verifying with
    the same code proves only self-consistency; this asks the gate. Passes the
    path KEYWORD-only (positional silently verifies the LIVE file, the one
    mistake that makes the check worthless while appearing to pass)."""
    try:
        from backend.core.ouroboros.governance import roadmap_reader as rr
        verdict, doc, diagnostic = rr.read_roadmap(path_override=path)
        valid = bool(getattr(doc, "signature_valid", False)) if doc else False
        return valid, f"verdict={verdict} signature_valid={valid} · {diagnostic}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _reset_provenance_cache() -> None:
    try:
        from backend.core.ouroboros.governance.delegated_provenance import (
            reset_provenance_cache_for_tests,
        )
        reset_provenance_cache_for_tests()
    except Exception:  # noqa: BLE001
        pass


def _default_repo() -> str:
    return (os.environ.get("JARVIS_PRIMARY_REPO", "") or "jarvis").strip() or "jarvis"


def _default_urgency() -> str:
    # Operator-authored work is due now; the router still applies its own
    # governance. Env-tunable, never a magic literal at the call site.
    return (os.environ.get("JARVIS_OPERATOR_GOAL_URGENCY", "") or "high").strip()


__all__ = [
    "GoalSpec",
    "SanctionResult",
    "author_and_sign_goal",
    "build_scoped_envelope",
    "cockpit_signing_enabled",
    "normalize_target_files",
    "withdraw_goal",
]
