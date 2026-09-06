"""Which model answers must be decided once, provably, and said out loud.

## Three defects, one subject

1. **The daemon read config too late.** `scripts/ouroboros_battle_test.py`
   called its own `.env` loader from inside `main()`, long after every
   `backend.*` module had been imported. Anything reading `os.environ` at
   MODULE scope captured a value from before the file was read.

2. **There were two `.env` parsers with different policies.**
   `env_bootstrap.load_env_once` is strictly `override=False`; the
   script's hand-rolled copy force-overrode the API keys. A variable's
   effective value depended on which loader touched it last.

3. **The cockpit banner guessed.** It resolved the model from the CLIENT's
   environment. The client is a different process that never loads
   `.env`, so a correctly pinned daemon showed NO model, and a stale
   client export would have shown the wrong one with full confidence.
   Measured 2026-09-06: the banner read `attached phase IDLE 22 sensors ·
   cost $0.00/$2.50` while the organism answered from a fine-tune.

## And the substitution underneath them

`_select_served_entry` falls back to "largest by size" when a pin is not
served. On this host that picks `qwen2.5-coder:32b` (19.85 GB) over the
fine-tuned `qwen3-coder-ov:30b` (18.58 GB) — a different model family,
chosen silently, leaving one WARNING in a log nobody tails. A fine-tune
A/B would have compared the base model against itself.

The fallback stays on the hot path, because a grader must never stop a
running loop. The BOOT GATE refuses it.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.governance import candidate_generator as cg  # noqa: E402

_SCRIPT = _REPO / "scripts" / "ouroboros_battle_test.py"

#: The real registry shape, with the two entries whose sizes make the
#: silent substitution possible.
TAGS = {"models": [
    {"name": "qwen3-coder-ov:30b", "size": 18583466060},
    {"name": "qwen2.5-coder:32b", "size": 19851349898},
    {"name": "qwen2.5-coder:7b", "size": 4683087561},
]}


# ---------------------------------------------------------------------------
# Phase 3 — fail closed on a pin the node does not serve
# ---------------------------------------------------------------------------


def test_a_served_pin_is_honoured() -> None:
    assert cg.resolve_active_model(TAGS, pin="qwen3-coder-ov:30b") == \
        "qwen3-coder-ov:30b"


def test_a_base_tag_pin_matches_its_full_tag() -> None:
    assert cg.resolve_active_model(TAGS, pin="qwen3-coder-ov") == \
        "qwen3-coder-ov:30b"


def test_no_pin_still_auto_selects() -> None:
    """Auto-selection is not the defect; SILENT SUBSTITUTION is."""
    assert cg.resolve_active_model(TAGS, pin="") == "qwen2.5-coder:32b"


def test_an_unserved_pin_RAISES_rather_than_substituting() -> None:
    with pytest.raises(cg.ModelPinUnavailable) as ei:
        cg.resolve_active_model(TAGS, pin="llama3:70b")
    assert ei.value.pin == "llama3:70b"
    assert "qwen3-coder-ov:30b" in ei.value.served


def test_the_error_carries_what_the_node_does_offer() -> None:
    """Actionable without a second lookup."""
    with pytest.raises(cg.ModelPinUnavailable) as ei:
        cg.resolve_active_model(TAGS, pin="nope:1b")
    msg = str(ei.value)
    assert "nope:1b" in msg and "qwen2.5-coder:32b" in msg


@pytest.mark.parametrize("registry", [None, {}, {"models": []}, {"models": None}])
def test_an_unreadable_registry_is_NOT_evidence_of_absence(registry) -> None:
    """"We could not ask" and "it is not there" must not share a verdict.

    The lane preflight already dies loudly when the engine cannot serve.
    A second opinion here would turn a transient blip into a self-kill.
    """
    assert cg.resolve_active_model(registry, pin="qwen3-coder-ov:30b") == \
        "qwen3-coder-ov:30b"


def test_the_hot_path_still_fails_soft() -> None:
    """`_select_served_entry` must keep substituting: a grader that raises
    stops a running loop, which is worse than a coarse answer."""
    entry = cg._select_served_entry(TAGS, pin="llama3:70b")
    assert cg._entry_name(entry) == "qwen2.5-coder:32b"


def test_one_matching_rule_serves_both() -> None:
    """A validator that admitted a pin the selector would then ignore
    would certify the substitution it exists to prevent."""
    for pin in ("qwen3-coder-ov:30b", "qwen3-coder-ov", "qwen2.5-coder:7b"):
        assert cg._entry_name(cg._select_served_entry(TAGS, pin=pin)) == \
            cg.resolve_active_model(TAGS, pin=pin)


# ---------------------------------------------------------------------------
# Phase 2 — one source of truth for the banner
# ---------------------------------------------------------------------------


def test_the_active_tag_is_empty_until_resolved(monkeypatch) -> None:
    monkeypatch.setattr(cg, "_ACTIVE_MODEL_TAG", "")
    assert cg.active_model_tag() == ""


def test_setting_the_tag_publishes_it(monkeypatch) -> None:
    monkeypatch.setattr(cg, "_ACTIVE_MODEL_TAG", "")
    cg.set_active_model_tag("  qwen3-coder-ov:30b  ")
    assert cg.active_model_tag() == "qwen3-coder-ov:30b"


def test_the_hydration_frame_carries_the_daemon_s_tag(monkeypatch) -> None:
    from backend.core.ouroboros.battle_test import cockpit_attach as ca
    monkeypatch.setattr(cg, "_ACTIVE_MODEL_TAG", "qwen3-coder-ov:30b")
    assert ca.CockpitAttachBridge._active_model() == "qwen3-coder-ov:30b"


def test_the_frame_reports_empty_rather_than_raising(monkeypatch) -> None:
    """A banner never breaks an attach."""
    from backend.core.ouroboros.battle_test import cockpit_attach as ca

    def _boom():
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(cg, "active_model_tag", _boom)
    assert ca.CockpitAttachBridge._active_model() == ""


def test_the_banner_reads_the_PAYLOAD_not_the_local_environment(
        monkeypatch) -> None:
    """The regression that matters. A client with a WRONG export and a
    payload naming the truth must render the truth."""
    monkeypatch.setenv("JARVIS_LOCAL_MODEL_NAME", "a-stale-client-export")
    printed = []

    class _Console:
        def print(self, obj, *a, **k):
            printed.append(getattr(obj, "plain", str(obj)))

    from backend.core.ouroboros.cli import ov as O
    O._render_hydration(_Console(), {
        "status": {"phase": "IDLE", "cost_spent_usd": 0.0,
                   "cost_budget_usd": 2.5},
        "ops": [], "liquidity": {}, "model": "qwen3-coder-ov:30b",
    })
    blob = "\n".join(printed)
    assert "qwen3-coder-ov:30b" in blob
    assert "a-stale-client-export" not in blob


def test_a_payload_without_a_model_omits_the_field_rather_than_guessing(
        monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_LOCAL_MODEL_NAME", "would-be-a-guess")
    printed = []

    class _Console:
        def print(self, obj, *a, **k):
            printed.append(getattr(obj, "plain", str(obj)))

    from backend.core.ouroboros.cli import ov as O
    O._render_hydration(_Console(), {
        "status": {"phase": "IDLE"}, "ops": [], "liquidity": {},
    })
    assert "would-be-a-guess" not in "\n".join(printed)


def test_the_banner_has_no_path_back_to_the_local_environment() -> None:
    """Pinned structurally: the guess must not be able to creep back as a
    fallback for an absent payload field."""
    src = (_REPO / "backend" / "core" / "ouroboros" / "cli" / "ov.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_render_hydration":
            body = ast.dump(node)
            assert "_model_pin" not in body, (
                "the banner must not resolve the model from this process")
            assert "JARVIS_LOCAL_MODEL_NAME" not in body
            return
    pytest.fail("_render_hydration not found")


# ---------------------------------------------------------------------------
# Phase 1 — the daemon ingests config before anything reads it
# ---------------------------------------------------------------------------


def test_the_daemon_loads_env_before_any_backend_import() -> None:
    """Module-scope `os.environ.get` in `backend.*` runs at IMPORT time.
    Loading config after those imports is loading it too late."""
    lines = _SCRIPT.read_text(encoding="utf-8").splitlines()
    call_at = next(i for i, l in enumerate(lines)
                   if l.strip() == "_load_env_once()")
    # The loader's OWN import does not count. `env_bootstrap` is a leaf
    # that reads no configuration, and it has to be imported before it can
    # be called; what must not precede the call is any module that could
    # capture a config value at import scope.
    first_consumer = next(
        i for i, l in enumerate(lines)
        if l.startswith("from backend.") and "env_bootstrap" not in l
    )
    assert call_at < first_consumer, (
        f"load_env_once() at line {call_at + 1} must precede the first "
        f"config-reading backend import at line {first_consumer + 1}")


def test_the_script_no_longer_hand_parses_dotenv() -> None:
    """Two parsers with different precedence made a variable's effective
    value depend on which loader touched it last."""
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_env_files":
            body = ast.dump(node)
            assert "partition" not in body, "hand-rolled parsing is back"
            assert "dotenv_values" in body, "must use the canonical library"
            assert "_load_env_once" in body, "must defer to the canonical loader"
            return
    pytest.fail("_load_env_files not found")


def test_the_env_file_pin_reaches_a_fresh_process(tmp_path) -> None:
    """The whole point, end to end: an interpreter with NOTHING exported
    must still resolve the operator's pin from the file."""
    env_file = tmp_path / ".env"
    env_file.write_text("JARVIS_LOCAL_MODEL_NAME=some-pinned-model:9b\n",
                        encoding="utf-8")
    env = dict(os.environ)
    env.pop("JARVIS_LOCAL_MODEL_NAME", None)
    env["JARVIS_ENV_FILE"] = str(env_file)
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from backend.core.env_bootstrap import load_env_once; load_env_once();"
        "from backend.core.ouroboros.governance.candidate_generator import "
        "_model_pin; print(_model_pin())" % _REPO
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=180)
    assert "some-pinned-model:9b" in out.stdout, out.stdout + out.stderr


def test_an_explicit_export_still_beats_the_file(tmp_path) -> None:
    """The cascade: environment first, file second. `override=False` is
    the loaded module's policy, not a second one invented here."""
    env_file = tmp_path / ".env"
    env_file.write_text("JARVIS_LOCAL_MODEL_NAME=from-the-file:1b\n",
                        encoding="utf-8")
    env = dict(os.environ)
    env["JARVIS_LOCAL_MODEL_NAME"] = "from-the-environment:2b"
    env["JARVIS_ENV_FILE"] = str(env_file)
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from backend.core.env_bootstrap import load_env_once; load_env_once();"
        "import os; print(os.environ['JARVIS_LOCAL_MODEL_NAME'])" % _REPO
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=180)
    assert "from-the-environment:2b" in out.stdout, out.stdout + out.stderr


def test_the_boot_gate_runs_after_the_lane_gate() -> None:
    """"The engine cannot serve" and "the engine serves, but not what you
    asked for" are different faults, and the first already has an owner."""
    src = _SCRIPT.read_text(encoding="utf-8")
    lane = src.rindex("_check_api_keys_or_die()")
    pin = src.rindex("_validate_model_pin_or_die()")
    assert lane < pin


def test_the_gate_has_its_own_exit_code() -> None:
    """So a client can say WHY the organism declined instead of "did not
    come up". 78 is EX_CONFIG; the repo already reads 75 as EX_TEMPFAIL."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "EXIT_MODEL_PIN_UNAVAILABLE = 78" in src
    assert "sys.exit(EXIT_MODEL_PIN_UNAVAILABLE)" in src


def test_the_client_and_the_daemon_agree_on_that_code() -> None:
    """Two declarations, because the client must not import the daemon
    script to read one integer. A code the client mis-reads would put a
    crash message over a configuration refusal."""
    from backend.core.ouroboros.cli import thin_client as tc
    src = _SCRIPT.read_text(encoding="utf-8")
    daemon_value = int(
        src.split("EXIT_MODEL_PIN_UNAVAILABLE = ", 1)[1].split("\n", 1)[0]
    )
    assert tc.EXIT_MODEL_PIN_UNAVAILABLE == daemon_value == 78


def test_the_client_does_not_retry_a_configuration_refusal() -> None:
    """Unlike EX_TEMPFAIL, nothing here resolves on its own."""
    tree = ast.parse(
        (_REPO / "backend" / "core" / "ouroboros" / "cli"
         / "thin_client.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "EXIT_MODEL_PIN_UNAVAILABLE" not in ast.dump(node.test):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        assert "_await_ignition_window" not in body, "must not retry"
        assert "Return" in body, "must return immediately"
        return
    pytest.fail("no branch handles the configuration refusal")
