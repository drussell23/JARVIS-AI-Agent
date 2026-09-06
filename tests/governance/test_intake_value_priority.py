"""P0.1 — value-derived intake priority.

ROOT CAUSE of "O+V can't see real work": intake priority is
``base = _PRIORITY_MAP.get(envelope.source, 99)`` — the source LABEL only. A
substantive-but-deferred source (security_advisory CVE, performance_regression,
github_issue, intent_discovery) sits at base 99, buried ~90 points behind an
``ai_miner`` trivia signal at base 3; the appetite band (signal_value, Slice 15)
that already knows the difference is consulted at ROUTE/GATE but never here.

This composes that band into the base with the ROUTE layer's escalate/clamp
semantics: a proven-defect (oracle) OR explicit-urgent signal escalates out of
starvation; an un-urgent cosmetic signal clamps below substance; executable /
indeterminate leaves the source tier alone. GRADUATED 2026-09-06: master
default-ON computes + stashes the band, and shadow now defaults OFF so the
re-anchoring is APPLIED — substance sets queue order. Shadow mode stays
available as an explicit opt-in (the `shadow` fixture sets it) and a typo can
only fall back to shadow, never silently un-graduate into an unexpected state.
"""
from __future__ import annotations

import pytest

from backend.core.ouroboros.governance.intake.intent_envelope import (
    make_envelope,
)
from backend.core.ouroboros.governance.intake import (
    unified_intake_router as R,
)


def _env(source, *, target_files=(), urgency="normal", evidence=None):
    return make_envelope(
        source=source, description="do a thing",
        target_files=tuple(target_files), repo="jarvis",
        confidence=0.5, urgency=urgency,
        evidence=dict(evidence or {}), requires_human_ack=False,
    )


_ORACLE_EV = {"attribution": {"status": "resolved"}}


@pytest.fixture
def enforce(monkeypatch):
    """Master on, shadow OFF → re-anchoring is authoritative."""
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_SHADOW", "false")
    monkeypatch.setenv("JARVIS_SIGNAL_VALUE_ROUTING_ENABLED", "true")
    monkeypatch.delenv("JARVIS_INTAKE_VALUE_SUBSTANTIVE_ANCHOR", raising=False)
    monkeypatch.delenv("JARVIS_INTAKE_VALUE_COSMETIC_FLOOR", raising=False)


@pytest.fixture
def shadow(monkeypatch):
    """Explicit shadow: master on, shadow EXPLICITLY on → compute but don't
    apply. As of the 2026-09-06 graduation the layer ENFORCES by default, so
    shadow mode is now opt-in and this fixture must set it rather than rely on
    the default (which is now enforce)."""
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_SHADOW", "true")
    monkeypatch.setenv("JARVIS_SIGNAL_VALUE_ROUTING_ENABLED", "true")


# ── band → re-anchor logic (the core) ────────────────────────────────

def test_oracle_escalates_deferred_source(enforce, tmp_path):
    """A proven defect (resolved attribution) on a deferred-tier source
    (security_advisory, base 99) escalates to the substantive anchor."""
    e = _env("security_advisory", target_files=("requirements.txt",),
             urgency="low", evidence=_ORACLE_EV)
    base = R._PRIORITY_MAP.get("security_advisory", 99)
    new, delta = R._reanchor_base_on_value(e, base, tmp_path)
    assert delta["value_band"] == 3  # ORACLE beats the cosmetic requirements shape
    assert delta["value_priority_reason"] == "oracle"
    assert new == R._value_substantive_anchor()
    assert new < base  # un-starved


def test_explicit_urgency_escalates_cve(enforce, tmp_path):
    """A critical CVE targets requirements.txt (cosmetic SHAPE) but is
    critical WORK — its explicit urgency must escalate it, NOT clamp it."""
    e = _env("security_advisory", target_files=("requirements.txt",),
             urgency="critical", evidence={})
    base = 99
    new, delta = R._reanchor_base_on_value(e, base, tmp_path)
    assert delta["value_priority_reason"] == "explicit_urgency"
    assert new == R._value_substantive_anchor()
    assert new < base


def test_cosmetic_non_urgent_clamps(enforce, tmp_path):
    """An un-urgent requirements-comment op (the Run-22 noise class) clamps
    to the deferred floor so it can never outrank substance."""
    e = _env("ai_miner", target_files=("requirements.txt",),
             urgency="low", evidence={})
    base = R._PRIORITY_MAP.get("ai_miner", 3)  # 3 — a MAPPED trivia tier
    new, delta = R._reanchor_base_on_value(e, base, tmp_path)
    assert delta["value_band"] == 1  # COSMETIC
    assert delta["value_priority_reason"] == "cosmetic_clamp"
    assert new == R._value_cosmetic_floor()
    assert new > base  # pushed DOWN (larger int = lower priority)


def test_executable_target_leaves_source_tier(enforce, tmp_path):
    """A real .py target is executable-band — the value layer can't tell real
    work from churn at intake, so the source tier stands (that's P0.3)."""
    f = tmp_path / "mod.py"
    f.write_text("def go():\n    return compute() + 1\n")
    e = _env("ai_miner", target_files=(str(f),), urgency="normal", evidence={})
    base = R._PRIORITY_MAP.get("ai_miner", 3)
    new, delta = R._reanchor_base_on_value(e, base, tmp_path)
    assert delta["value_band"] == 2  # EXECUTABLE
    assert delta["value_priority_reason"] == "source_tier"
    assert new == base  # unchanged


def test_escalate_never_demotes(enforce, tmp_path):
    """min() escalation must never DEMOTE an already-high signal: a
    test_failure (base 1) with oracle evidence stays at 1, not raised to the
    anchor (2)."""
    e = _env("test_failure", target_files=("x.py",), urgency="high",
             evidence=_ORACLE_EV)
    base = R._PRIORITY_MAP.get("test_failure", 1)  # 1
    new, _ = R._reanchor_base_on_value(e, base, tmp_path)
    assert new == min(base, R._value_substantive_anchor()) == 1


def test_clamp_never_promotes(enforce, tmp_path):
    """max() clamp must never PROMOTE: a cosmetic op already at the floor
    stays there."""
    e = _env("doc_staleness", target_files=("setup.cfg",), urgency="low")
    base = 99
    new, _ = R._reanchor_base_on_value(e, base, tmp_path)
    assert new == 99


# ── shadow vs enforce (the rollout contract) ─────────────────────────

def test_shadow_computes_but_does_not_apply(shadow, tmp_path):
    """Graduated default: the band is computed + stashed to evidence, but the
    dispatched priority is byte-identical to the source-tier computation."""
    e = _env("security_advisory", target_files=("requirements.txt",),
             urgency="critical", evidence={})
    prio, _ = R._compute_priority(e, repo_root=tmp_path)
    # Evidence recorded (observability for the P0.5 soak)...
    assert e.evidence.get("value_band") is not None
    assert e.evidence.get("value_priority_enforced") is False
    # ...but the priority still used the SOURCE base (99), not the anchor.
    expected_source_base = 99 - R._URGENCY_BOOST["critical"]  # 99 - 3 = 96
    assert prio == expected_source_base


def test_enforce_substance_beats_trivia(enforce, tmp_path):
    """THE inversion fix: under enforce, an urgent CVE (was base 99) now
    out-prioritizes an ai_miner trivia signal (base 3)."""
    cve = _env("security_advisory", target_files=("requirements.txt",),
               urgency="critical", evidence={})
    trivia = _env("ai_miner", target_files=("requirements.txt",),
                  urgency="low", evidence={})
    cve_prio, _ = R._compute_priority(cve, repo_root=tmp_path)
    trivia_prio, _ = R._compute_priority(trivia, repo_root=tmp_path)
    assert cve_prio < trivia_prio  # lower int = higher priority → CVE wins


def test_master_off_is_inert(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "false")
    e = _env("security_advisory", target_files=("requirements.txt",),
             urgency="critical", evidence=_ORACLE_EV)
    new, delta = R._reanchor_base_on_value(e, 99, tmp_path)
    assert new == 99
    assert delta == {}


def test_composes_global_appetite_switch(monkeypatch, tmp_path):
    """A repo-wide JARVIS_SIGNAL_VALUE_ROUTING_ENABLED=false disables the
    intake layer too (one off switch for the whole appetite layer)."""
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("JARVIS_SIGNAL_VALUE_ROUTING_ENABLED", "false")
    assert R._value_priority_master_enabled() is False


@pytest.mark.parametrize("typo", ["garbage", " ", "maybe"])
def test_shadow_typo_stays_shadow(monkeypatch, typo):
    """An actuator must fail SAFE — a typo'd shadow value must NOT make value
    re-ranking authoritative."""
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_PRIORITY_SHADOW", typo)
    assert R._value_priority_shadow_enabled() is True
    assert R._value_priority_enforce() is False


# ── anchors: env-tunable + DRY-derived from the map ──────────────────

def test_anchors_default_to_map_landmarks(monkeypatch):
    monkeypatch.delenv("JARVIS_INTAKE_VALUE_SUBSTANTIVE_ANCHOR", raising=False)
    monkeypatch.delenv("JARVIS_INTAKE_VALUE_COSMETIC_FLOOR", raising=False)
    assert R._value_substantive_anchor() == R._PRIORITY_MAP["backlog"]
    assert R._value_cosmetic_floor() == 99


def test_anchors_env_tunable(monkeypatch):
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_SUBSTANTIVE_ANCHOR", "1")
    monkeypatch.setenv("JARVIS_INTAKE_VALUE_COSMETIC_FLOOR", "50")
    assert R._value_substantive_anchor() == 1
    assert R._value_cosmetic_floor() == 50


def test_scoring_fault_leaves_source_tier(enforce, monkeypatch, tmp_path):
    """Fail-soft: if the appetite scorer raises, the source tier stands and
    nothing is stashed — intake never breaks on a value-scoring fault."""
    import backend.core.ouroboros.governance.signal_value as sv
    monkeypatch.setattr(
        sv, "score_signal",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    e = _env("ai_miner", target_files=("x.py",), urgency="normal")
    new, delta = R._reanchor_base_on_value(e, 3, tmp_path)
    assert new == 3 and delta == {}
