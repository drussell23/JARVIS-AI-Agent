"""A missing dependency must not be reported as a missing terminal.

## What happened

`ov` attached to a healthy organism in 4.4 seconds and then showed a
frozen crest with no prompt, forever. The operator read that as "stuck on
the loading page". Everything except one thing was fine: the daemon
answered hydration in 1 ms and streamed telemetry at 1 Hz, and `py-spy`
found the client idle in `select` inside a live attach session.

`prompt_toolkit` was not installed. The interactive cockpit is a
full-screen `prompt_toolkit.Application`, and the gate that noticed said::

    def _can_run_split_plane() -> bool:
        try:
            if not sys.stdin.isatty():
                return False
            import prompt_toolkit
            return True
        except Exception:
            return False

Two structurally different facts, one `False`. A piped stdin SHOULD
degrade quietly. A missing package must not. And because the caller only
announces failures inside the branch that gate guards, returning False
skipped `mount_breaker.announce` -- the seam built to shout about exactly
this -- so the operator was told nothing at all. Measured on a real
terminal: zero output lines mentioned the failure.

`ov restart` reuses the same cockpit path, which is why it hung too.

## What these tests hold in place

That the probe answers with a REASON, that the reason survives being
piped, that the remedy names an installer which actually exists on this
machine, and that the announce seam is reached.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.cli import surface_probe as sp  # noqa: E402

_OV = _REPO / "backend" / "core" / "ouroboros" / "cli" / "ov.py"

#: A pair that can never be importable, for the missing-dependency cases.
ABSENT = (("definitely_not_a_real_module_xyz", "definitely-not-real"),)


# ---------------------------------------------------------------------------
# The distinction the whole module exists to make
# ---------------------------------------------------------------------------


def test_a_missing_package_is_an_INSTALLATION_fault() -> None:
    v = sp.probe_interactive_surface(stdin_isatty=True, required=ABSENT)
    assert v.kind == sp.INSTALLATION
    assert v.is_fault is True
    assert not v.ok
    assert "definitely_not_a_real_module_xyz" in v.reason


def test_a_piped_stdin_is_an_ENVIRONMENT_matter_not_a_fault() -> None:
    v = sp.probe_interactive_surface(stdin_isatty=False, required=())
    assert v.kind == sp.ENVIRONMENT
    assert v.is_fault is False, "a pipe is not a broken install"
    assert not v.ok


def test_a_real_terminal_with_its_deps_is_READY() -> None:
    v = sp.probe_interactive_surface(stdin_isatty=True, required=())
    assert v.ok and v.kind == sp.READY


def test_the_dependency_check_runs_BEFORE_the_tty_check() -> None:
    """`ov | tee log` on a box with no prompt_toolkit must still say what
    is actually wrong. Reporting 'not a TTY' to an operator whose venv is
    broken sends them to debug the one thing that is working."""
    v = sp.probe_interactive_surface(stdin_isatty=False, required=ABSENT)
    assert v.kind == sp.INSTALLATION


def test_the_fault_carries_a_real_exception_for_the_announce_seam() -> None:
    """`mount_breaker.announce` writes a traceback. A fabricated exception
    would name the wrong line."""
    v = sp.probe_interactive_surface(stdin_isatty=True, required=ABSENT)
    assert isinstance(v.error, ImportError)


def test_the_verdict_is_falsy_when_not_ready() -> None:
    assert not sp.probe_interactive_surface(stdin_isatty=True, required=ABSENT)
    assert sp.probe_interactive_surface(stdin_isatty=True, required=())


# ---------------------------------------------------------------------------
# The remedy has to actually work on THIS machine
# ---------------------------------------------------------------------------


def test_the_remedy_names_an_installer_and_the_requirement() -> None:
    v = sp.probe_interactive_surface(stdin_isatty=True, required=ABSENT)
    assert v.remedy
    assert "definitely-not-real" in v.remedy


def test_the_version_is_read_from_requirements_never_restated() -> None:
    line = sp.requirement_line("prompt_toolkit")
    assert line.startswith("prompt_toolkit")
    assert any(c in line for c in "<>=~!"), (
        "the repo declares a constraint; the remedy must carry it")


def test_a_dash_underscore_mismatch_still_matches(tmp_path, monkeypatch) -> None:
    """PEP 503 normalises them. The import is `prompt_toolkit`; a
    requirements file may spell it `prompt-toolkit`."""
    req = tmp_path / "r.txt"
    req.write_text("prompt-toolkit>=3.0.43  # comment\n", encoding="utf-8")
    monkeypatch.setenv(sp.ENV_REQUIREMENTS, str(req))
    assert sp.requirement_line("prompt_toolkit") == "prompt-toolkit>=3.0.43"


def test_an_unknown_package_yields_a_bare_name_not_an_invented_pin(
        tmp_path, monkeypatch) -> None:
    req = tmp_path / "r.txt"
    req.write_text("rich>=13\n", encoding="utf-8")
    monkeypatch.setenv(sp.ENV_REQUIREMENTS, str(req))
    assert sp.requirement_line("nowhere-pkg") == "nowhere-pkg"


def test_a_missing_requirements_file_still_gives_usable_advice(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(sp.ENV_REQUIREMENTS, str(tmp_path / "absent.txt"))
    assert sp.requirement_line("prompt_toolkit") == "prompt_toolkit"


def test_a_commented_out_requirement_is_not_matched(tmp_path, monkeypatch) -> None:
    req = tmp_path / "r.txt"
    req.write_text("# prompt_toolkit>=9.9.9\nrich>=13\n", encoding="utf-8")
    monkeypatch.setenv(sp.ENV_REQUIREMENTS, str(req))
    assert sp.requirement_line("prompt_toolkit") == "prompt_toolkit"


def test_the_installer_targets_THIS_interpreter(monkeypatch) -> None:
    """`pip install` at a shell installs into whatever venv is active,
    which on the box that produced this defect was not the one `ov`
    resolves through."""
    hint = sp._installer_hint()
    assert sys.executable in hint


def test_the_installer_falls_back_when_the_venv_has_no_pip(monkeypatch) -> None:
    """A uv-created venv has no pip. `python -m pip install` there fails
    with `No module named pip` -- precise, confident, and useless."""
    import importlib.util as _u
    real = _u.find_spec

    def _no_pip(name, *a, **k):
        return None if name == "pip" else real(name, *a, **k)

    monkeypatch.setattr(sp.importlib.util, "find_spec", _no_pip)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: "/opt/uv" if n == "uv" else None)
    hint = sp._installer_hint()
    assert hint.startswith("/opt/uv pip install --python "), hint
    assert sys.executable in hint


def test_with_neither_pip_nor_uv_it_still_says_where(monkeypatch) -> None:
    import importlib.util as _u
    real = _u.find_spec
    monkeypatch.setattr(sp.importlib.util, "find_spec",
                        lambda n, *a, **k: None if n == "pip" else real(n, *a, **k))
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    assert sys.executable in sp._installer_hint()


# ---------------------------------------------------------------------------
# It may never raise, and never fabricate
# ---------------------------------------------------------------------------


def test_a_broken_find_spec_reads_as_environment_not_as_a_broken_install(
        monkeypatch) -> None:
    """Claiming a broken installation on evidence we could not gather
    sends the operator to reinstall a package that is fine."""
    def _boom(*_a, **_k):
        raise RuntimeError("import system is on fire")

    monkeypatch.setattr(sp, "missing_imports", _boom)
    v = sp.probe_interactive_surface(stdin_isatty=True)
    assert v.kind == sp.ENVIRONMENT and not v.is_fault


def test_missing_imports_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(sp.importlib.util, "find_spec",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    assert sp.missing_imports((("anything", "anything"),)) == (("anything", "anything"),)


@pytest.mark.parametrize("bad", ["", None])
def test_requirement_line_tolerates_junk(bad) -> None:
    assert sp.requirement_line(bad) == ""


def test_the_probe_does_not_import_the_surface() -> None:
    """`find_spec`, not a real import: the cockpit's modules pull in a
    terminal stack, and a probe must not touch the screen it asks about."""
    before = set(sys.modules)
    sp.probe_interactive_surface(stdin_isatty=True)
    new = set(sys.modules) - before
    assert not any(m.startswith("prompt_toolkit") for m in new)


# ---------------------------------------------------------------------------
# The call site: the announce seam must be reached
# ---------------------------------------------------------------------------


def test_can_run_split_plane_is_still_a_bool() -> None:
    from backend.core.ouroboros.cli import ov as O
    assert isinstance(O._can_run_split_plane(), bool)


def test_the_gate_and_the_verdict_agree() -> None:
    from backend.core.ouroboros.cli import ov as O
    assert O._can_run_split_plane() == bool(O._split_plane_verdict())


def test_an_installation_fault_reaches_the_announce_seam() -> None:
    """The regression that matters. `announce` existed, was correct, and
    was unreachable for this cause -- the 'code right but unreachable'
    shape. Pinned structurally so a refactor cannot quietly re-orphan it.
    """
    tree = ast.parse(_OV.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "is_fault" not in test_src:
            continue
        body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        assert "announce" in body_src, (
            "the is_fault branch must announce the failure")
        assert "remedy" in body_src, (
            "and must tell the operator how to fix it")
        found = True
    assert found, "no branch keys off the verdict being a fault"


def test_the_gate_no_longer_swallows_the_import_error() -> None:
    """The old body caught ImportError and returned False. Nothing in the
    gate may do that again."""
    tree = ast.parse(_OV.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_can_run_split_plane":
            src = ast.dump(node)
            assert "prompt_toolkit" not in src, (
                "the dependency is declared in surface_probe, not here")
            assert "Try" not in src, (
                "a bare except here is what hid the missing package")
            return
    pytest.fail("_can_run_split_plane not found")
