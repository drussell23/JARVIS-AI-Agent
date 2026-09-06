"""signal_value — a signal's DECLARED work-nature classifies its value, not the
target file's AST.

The rooted autonomous-work-selection pathology: doc-staleness (the highest-
VOLUME source) buried genuinely valuable work because ``score_signal`` judged
the target FILE (a real, executable Python module → BAND_EXECUTABLE) instead of
the WORK (a docstring rewrite → cosmetic). A signal that stamps
``evidence.work_nature = "documentation"`` is now scored COSMETIC regardless of
its target's AST, so the value layer (enforced by default) sinks it to the
starvation floor and floats an oracle-band failing-test op to the front.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.core.ouroboros.governance import signal_value as sv


@pytest.fixture
def executable_module(tmp_path: Path) -> str:
    p = tmp_path / "widget.py"
    p.write_text(
        "def add(a, b):\n    return a + b\n\n"
        "class W:\n    def m(self):\n        return 1\n",
        encoding="utf-8",
    )
    return str(p)


def _score(source, target, evidence_dict, root):
    return sv.score_signal(source, (target,), json.dumps(evidence_dict), root)


def test_executable_target_without_nature_scores_executable(
    executable_module, tmp_path,
):
    # Proves the file genuinely IS executable — so it is the work_nature, not
    # the file, that flips the band in the next test.
    assert _score("doc_staleness", executable_module, {}, str(tmp_path)) == (
        sv.BAND_EXECUTABLE
    )


def test_declared_documentation_nature_is_cosmetic_despite_executable_target(
    executable_module, tmp_path,
):
    band = _score(
        "doc_staleness", executable_module,
        {"work_nature": "documentation", "category": "undocumented_api"},
        str(tmp_path),
    )
    assert band == sv.BAND_COSMETIC_CLASS


def test_oracle_attribution_outranks_declared_cosmetic(
    executable_module, tmp_path,
):
    band = _score(
        "test_failure", executable_module,
        {"work_nature": "documentation", "attribution": {"status": "resolved"}},
        str(tmp_path),
    )
    assert band == sv.BAND_ORACLE


def test_cosmetic_natures_are_env_tunable(
    executable_module, tmp_path, monkeypatch,
):
    monkeypatch.setenv("JARVIS_SIGNAL_COSMETIC_WORK_NATURES", "onlythis")
    # "documentation" is no longer in the set → falls through to the AST census.
    assert _score(
        "x", executable_module, {"work_nature": "documentation"}, str(tmp_path),
    ) == sv.BAND_EXECUTABLE
    assert _score(
        "x", executable_module, {"work_nature": "onlythis"}, str(tmp_path),
    ) == sv.BAND_COSMETIC_CLASS


def test_malformed_or_missing_nature_never_raises(executable_module, tmp_path):
    # A bad marker is not authority — it must degrade to the AST census.
    assert _score(
        "x", executable_module, {"work_nature": 12345}, str(tmp_path),
    ) == sv.BAND_EXECUTABLE
    assert _score(
        "x", executable_module, {}, str(tmp_path),
    ) == sv.BAND_EXECUTABLE


# ── end-to-end: a documentation signal sinks below substance under enforce ──


def test_doc_signal_is_clamped_below_substance_under_enforce(
    executable_module, tmp_path, monkeypatch,
):
    from backend.core.ouroboros.governance.intake import (
        unified_intake_router as R,
    )
    from backend.core.ouroboros.governance.intake.intent_envelope import (
        make_envelope,
    )

    # Enforce is the graduated default; make it explicit for the test.
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_SHADOW", "false")
    monkeypatch.setenv("JARVIS_SIGNAL_VALUE_ROUTING_ENABLED", "true")

    doc = make_envelope(
        source="doc_staleness", description="add docstrings",
        target_files=(executable_module,), repo="jarvis", confidence=0.8,
        urgency="normal",
        evidence={"work_nature": "documentation", "category": "undocumented_api"},
        requires_human_ack=False,
    )
    bug = make_envelope(
        source="test_failure", description="fix the failing test",
        target_files=(executable_module,), repo="jarvis", confidence=0.95,
        urgency="high",
        evidence={"attribution": {"status": "resolved"}},
        requires_human_ack=False,
    )
    doc_prio, _ = R._compute_priority(doc, repo_root=tmp_path)
    bug_prio, _ = R._compute_priority(bug, repo_root=tmp_path)
    # lower int = higher priority — the failing-test op must win.
    assert bug_prio < doc_prio
