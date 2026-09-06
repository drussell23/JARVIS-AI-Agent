#!/usr/bin/env python3
"""Authorize a piece of work, cryptographically, in one command.

## What this is NOT

It is not a bypass of the self-modification cage, and it does not weaken
it. `risk_engine._self_protection_verdict` still BLOCKS every governance
edit that cannot prove operator intent, `verify_provenance_claim` still
re-derives everything from ground truth on every classify, and the HMAC is
still the only thing that counts as proof. Nothing here touches any of it.

## What it is, and why it is the right fix

The cage refuses work that traces to no signed goal. That is correct: a
source label is a string any sensor can self-assert, and the module that
implements delegation says so directly -- a label carries "no proof a
directive was authored by the human operator rather than injected or
hallucinated by O+V in a prior cycle". The control was built after a soak
in which 70 operations were blocked, and the answer chosen then was
cryptographic delegation rather than a whitelist. That answer is still
right.

So the friction is not in the CHECK. It is in the AUTHORING: the signed
roadmap holds one goal scoped to one file, and adding another meant
hand-editing YAML, getting the schema right, and re-running the signer
with the secret. Every one of those steps is a chance to produce a
document that fails verification for a reason nobody can see.

This makes the legitimate path a single command, so that the easy thing to
do is also the correct one. When authorizing work is harder than
bypassing the check, the check gets bypassed.

## The signature is the whole point

The secret is read from the environment, never from an argument, so it
cannot land in shell history. The document is re-signed through
`strategy_signer.sign_roadmap_doc` -- the same function the operator CLI
uses, not a second implementation -- and then VERIFIED by re-reading it
through `roadmap_reader` before the old file is replaced. A roadmap that
does not verify is never written: an unverifiable authorization is worse
than none, because the cage will refuse it and the reason will be three
layers away.

Usage:
    sanction_goal.py --id fix-x --title "..." --files a.py b.py \\
                     [--description "..."] [--note "..."] [--dry-run]
    sanction_goal.py --list
    sanction_goal.py --remove <goal-id>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Where the signed roadmap lives, relative to the repo root.
ENV_ROADMAP = "JARVIS_ROADMAP_PATH"
DEFAULT_ROADMAP = ".jarvis/roadmap.yaml"
#: The operator secret. Read from the environment ONLY -- an argument would
#: put it in shell history, and a file path would invite committing it.
ENV_SECRET = "JARVIS_ROADMAP_READER_HMAC_SECRET"
#: Stamped into the document so a later reader can tell who authorized it.
ENV_OPERATOR_ID = "JARVIS_OPERATOR_ID"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def roadmap_path() -> Path:
    raw = (os.environ.get(ENV_ROADMAP, "") or "").strip()
    return Path(raw) if raw else _repo_root() / DEFAULT_ROADMAP


def _load(path: Path) -> Dict[str, Any]:
    import yaml  # noqa: PLC0415

    if not path.is_file():
        # A first goal on a fresh box is legitimate; the envelope is built
        # to the same shape the reader expects rather than left to chance.
        return {
            "version": 1,
            "authority": "operator_directed",
            "source": "operator_directed_agent_signed",
            "goals": [],
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"REFUSING: {path} is not a YAML mapping")
    data.setdefault("goals", [])
    if not isinstance(data["goals"], list):
        raise SystemExit(f"REFUSING: {path} has a non-list 'goals'")
    return data


def _verify(path: Path) -> tuple:
    """Read the document back through the READER, not through our own
    assumptions. Returns ``(ok, detail)``.

    Signing and verifying with the same code proves only that the code is
    self-consistent. This asks the component the cage actually consults.
    """
    try:
        sys.path.insert(0, str(_repo_root()))
        from backend.core.ouroboros.governance import roadmap_reader as rr  # noqa: PLC0415

        # `path_override` is KEYWORD-ONLY, and the reader otherwise takes its
        # path from JARVIS_ROADMAP_READER_PATH -- not from the variable this
        # script uses. Passing it positionally silently verified the LIVE
        # document instead of the candidate, which is the one mistake that
        # would make this check worthless while appearing to pass.
        verdict, doc, diagnostic = rr.read_roadmap(path_override=path)
        valid = bool(getattr(doc, "signature_valid", False)) if doc else False
        return valid, f"verdict={verdict} signature_valid={valid} · {diagnostic}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def cmd_list(doc: Dict[str, Any]) -> int:
    goals = doc.get("goals") or []
    print(f"{len(goals)} sanctioned goal(s) in {roadmap_path()}")
    print(f"  signed: {doc.get('signed')}   signed_at: {doc.get('signed_at')}")
    for g in goals:
        files = g.get("files") or g.get("target_files") or []
        print(f"\n  {g.get('id')}")
        print(f"    {g.get('title', '')[:100]}")
        for f in files:
            print(f"    · {f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sign a goal into the operator roadmap the cage verifies.",
    )
    ap.add_argument("--id", help="goal id, referenced by the provenance claim")
    ap.add_argument("--title", default="", help="one line: what is to be done")
    ap.add_argument("--description", default="",
                    help="the detail the model is given")
    ap.add_argument("--files", nargs="*", default=[],
                    help="repo-relative paths this goal MAY touch — the cage "
                         "refuses an op that strays outside them")
    ap.add_argument("--symbol", nargs="*", default=[],
                    help="function/class name(s) inside --files the model must "
                         "focus on — narrows generation + context slicing")
    ap.add_argument("--success", default="",
                    help="the operator's acceptance criterion for the goal")
    ap.add_argument("--note", default="", help="why this was authorized")
    ap.add_argument("--list", action="store_true", help="show current goals")
    ap.add_argument("--remove", default="", help="withdraw a goal by id")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the document; write nothing")
    args = ap.parse_args(argv)

    path = roadmap_path()
    doc = _load(path)

    if args.list:
        return cmd_list(doc)

    # DRY: the authoring, signing and verify-before-replace all live in ONE
    # place now — ``operator_goal_sanction`` — so the cockpit ``/goal`` verb
    # and this CLI cannot drift. The old inline body wrote the goal key as
    # ``files``; ``roadmap_reader._parse_goal_entry`` reads ``target_files``,
    # so those goals parsed with EMPTY scope and the cage refused them
    # ``goal_has_no_scope``. Delegating fixes that at the root.
    sys.path.insert(0, str(_repo_root()))
    from backend.core.ouroboros.governance.operator_goal_sanction import (  # noqa: E501,PLC0415
        GoalSpec,
        author_and_sign_goal,
        withdraw_goal,
    )

    if args.remove:
        res = withdraw_goal(args.remove, path_override=path)
        if not res.ok:
            print(f"REFUSING: {res.reason} — {res.detail}", file=sys.stderr)
            return 2
        print(f"withdrew {res.goal_id}\n  {res.detail}")
        return 0

    if not args.id:
        ap.error("--id is required (or use --list / --remove)")
    if not args.files:
        # A goal with no file scope authorizes everything, which is the one
        # shape that would make the cage meaningless.
        ap.error("--files is required: an unscoped goal authorizes every "
                 "file, which defeats the control it is asking to satisfy")

    spec = GoalSpec(
        goal_id=str(args.id),
        title=str(args.title or args.id),
        description=str(args.description or args.title or args.id),
        target_files=tuple(str(f) for f in args.files),
        target_symbols=tuple(str(s) for s in (args.symbol or [])),
        success_criteria=str(args.success or ""),
        note=str(args.note or ""),
    )
    print(f"sanctioning {spec.goal_id} over {len(spec.target_files)} file(s)")
    res = author_and_sign_goal(
        spec,
        path_override=path,
        operator_id=os.environ.get(ENV_OPERATOR_ID, "").strip() or None,
        dry_run=args.dry_run,
    )
    if args.dry_run and res.ok:
        print(res.detail)  # the rendered, would-be document
        return 0
    if not res.ok:
        print(f"REFUSING: {res.reason} — {res.detail}", file=sys.stderr)
        return 2
    print(f"  signed and verified -> {res.roadmap_path}")
    print(f"  {res.detail}")
    print("\n  The cage now accepts ops whose files fall inside this goal.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
