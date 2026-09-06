"""Can the interactive cockpit actually run, and if not, WHOSE fault is it?

## The defect this closes

`ov` attached to a healthy organism in 4.5 seconds, rendered its banner,
and then showed the operator a frozen crest with no prompt, forever. The
daemon was fine -- hydration answered in 1 ms and telemetry streamed at
1 Hz. The client was fine too: `py-spy` found it idle in `select`, inside
a live attach session.

What was missing was `prompt_toolkit`. The interactive surface is a
full-screen `prompt_toolkit.Application`, the venv did not have it, and
the gate that noticed said only::

    def _can_run_split_plane() -> bool:
        if not sys.stdin.isatty():
            return False
        import prompt_toolkit
        return True
    except Exception:
        return False

Two structurally different facts collapse into one `False`:

  * **no TTY** -- a property of how the operator invoked it. Piped and
    scripted attaches SHOULD degrade quietly; nothing is broken.
  * **a package is not installed** -- a broken environment. Degrading
    quietly here is a lie, and the operator reads the frozen crest as
    "stuck on the loading page".

Worse, the caller only announces failures INSIDE the branch that gate
guards, so `mount_breaker.announce` -- the seam built precisely to shout
about software faults and write a traceback to disk -- was unreachable
for the one cause that most needed it. Measured on the real terminal:
zero lines of output mentioned the failure.

## What this module does

It answers with a REASON. `environment` degrades quietly, as it should.
`installation` is routed into the existing announce seam and told loudly,
with the exact command that fixes it.

## Nothing here is hardcoded twice

The import names are declared once, beside the surface that needs them.
The VERSION is never spelled here at all -- `remedy_for` reads the
constraint out of the repo's own `requirements.txt`, so the advice an
operator is given is the requirement the repo actually states, and a
version bump there cannot leave a stale instruction behind. If the
requirement cannot be read, the remedy names the distribution without a
constraint rather than inventing one.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("Ouroboros.SurfaceProbe")

#: Verdict kinds. Named, because the whole point is that the caller can
#: tell them apart.
READY = "ready"
#: The invocation cannot support the surface -- piped, redirected, no TTY.
#: Correct behaviour, quiet degradation.
ENVIRONMENT = "environment"
#: The surface's own dependencies are not installed. A broken install, and
#: the operator must be told.
INSTALLATION = "installation"

#: What the interactive surface imports at runtime, as (import name,
#: distribution name). Declared HERE, once, because this module is what
#: asks the question -- a second copy inside the gate is exactly how the
#: gate came to disagree with reality.
INTERACTIVE_IMPORTS: Tuple[Tuple[str, str], ...] = (
    ("prompt_toolkit", "prompt_toolkit"),
)

#: Where the repo states its own version constraints. Overridable so a
#: relocated checkout or a test can point at its own file.
ENV_REQUIREMENTS = "JARVIS_REQUIREMENTS_PATH"
_DEFAULT_REQUIREMENTS = "requirements.txt"


@dataclass(frozen=True)
class SurfaceVerdict:
    """Whether the interactive surface can run, and why not."""

    kind: str
    reason: str
    remedy: str = ""
    #: The real exception, when one was raised. `announce` needs it to
    #: write a traceback, and a fabricated one would name the wrong line.
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.kind == READY

    @property
    def is_fault(self) -> bool:
        """Is this a broken installation rather than a plain invocation?"""
        return self.kind == INSTALLATION

    def __bool__(self) -> bool:
        return self.ok


#: The leading name of a requirement line, before any extras, constraint
#: or marker. `prompt-toolkit[x]>=3.0.43; python_version>="3.9"` -> the name.
_REQ_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalise(name: str) -> str:
    """PEP 503 name normalisation.

    `prompt_toolkit` and `prompt-toolkit` are the SAME distribution. An
    import name uses the underscore and a requirements file may use either,
    so comparing the raw strings silently fails to find a requirement that
    is right there. Normalising both sides is the rule the packaging
    ecosystem already agreed on; matching by hand-built regex was how this
    got it wrong the first time.
    """
    return re.sub(r"[-_.]+", "-", str(name or "")).strip().lower()


def _repo_root() -> Path:
    # cli/ -> ouroboros/ -> core/ -> backend/ -> repo
    return Path(__file__).resolve().parents[4]


def requirements_path() -> Path:
    raw = (os.environ.get(ENV_REQUIREMENTS, "") or "").strip()
    return Path(raw) if raw else _repo_root() / _DEFAULT_REQUIREMENTS


def requirement_line(dist: str) -> str:
    """The repo's own constraint for *dist*, e.g. ``prompt_toolkit>=3.0.43``.

    Read rather than restated. A version pinned in two places drifts, and
    the copy that drifts is always the one in the error message nobody
    tests. Returns the bare distribution name when the file cannot be read
    or does not mention it -- advice without a constraint is still correct
    advice, whereas an invented constraint is not. NEVER raises.
    """
    name = str(dist or "").strip()
    if not name:
        return ""
    try:
        text = requirements_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return name
    want = _normalise(name)
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):      # -r / -e / --flag lines
            continue
        m = _REQ_NAME.match(line)
        if m and _normalise(m.group(1)) == want:
            # The requirement as the repo states it, minus any environment
            # marker: `;python_version<"3.11"` is a condition on whether the
            # line applies, not part of what to install.
            return line.split(";", 1)[0].strip()
    return name


def _installer_hint() -> str:
    """An installer command that will actually work HERE. NEVER raises.

    Two things make the obvious answer wrong.

    First, `pip install` typed at a shell installs into whatever venv
    happens to be active, which on the box that produced this defect was
    not the one the `ov` entry point resolves through. So the interpreter
    is always named explicitly: `sys.executable`, the one actually running.

    Second, that interpreter may have no pip at all. The venv here was
    built by `uv`, which does not install pip by default, so
    `python -m pip install ...` fails with `No module named pip` — advice
    that is precise, confident, and useless. The installer is therefore
    DERIVED from what this environment can do, in order of directness:
    pip inside the target interpreter, then `uv` if it is on PATH, then a
    plain instruction that at least names the interpreter.
    """
    exe = sys.executable or "python3"
    try:
        if importlib.util.find_spec("pip") is not None:
            return f"{exe} -m pip install"
    except Exception:  # noqa: BLE001
        pass
    try:
        import shutil  # noqa: PLC0415

        uv = shutil.which("uv")
        if uv:
            # `--python` targets the venv `ov` resolves through, which is
            # the whole point: uv's default target is the project venv,
            # not necessarily this one.
            return f"{uv} pip install --python {exe}"
    except Exception:  # noqa: BLE001
        pass
    return f"install into {exe}:"


def missing_imports(
    required: Tuple[Tuple[str, str], ...] = INTERACTIVE_IMPORTS,
) -> Tuple[Tuple[str, str], ...]:
    """Which declared imports are not importable. NEVER raises.

    Uses `find_spec`, not a real import: the surface's modules pull in a
    terminal stack, and a probe must not have side effects on the screen
    it is asking about.
    """
    out = []
    for mod, dist in required:
        try:
            if importlib.util.find_spec(mod) is None:
                out.append((mod, dist))
        except (ImportError, ValueError, AttributeError):
            # A namespace package with a broken parent raises rather than
            # returning None. Unimportable either way.
            out.append((mod, dist))
        except Exception:  # noqa: BLE001 — a probe may never raise
            logger.debug("find_spec(%s) faulted", mod, exc_info=True)
            out.append((mod, dist))
    return tuple(out)


def probe_interactive_surface(
    *,
    stdin_isatty: Optional[bool] = None,
    required: Tuple[Tuple[str, str], ...] = INTERACTIVE_IMPORTS,
) -> SurfaceVerdict:
    """Can the full-screen cockpit run here? NEVER raises.

    The dependency check runs FIRST, before the TTY check. A broken
    install is a fact about the machine and stays true whether or not this
    particular invocation had a terminal -- reporting "not a TTY" to an
    operator whose venv is missing a package sends them to debug the one
    thing that is working. `ov | tee log` on a box with no
    `prompt_toolkit` should still say what is actually wrong.

    `stdin_isatty` is injectable so the decision can be tested without a
    pty, and so a caller that already knows can avoid asking twice.
    """
    try:
        gone = missing_imports(required)
        if gone:
            names = ", ".join(m for m, _ in gone)
            reqs = " ".join(requirement_line(d) for _, d in gone)
            return SurfaceVerdict(
                kind=INSTALLATION,
                reason=f"the interactive cockpit needs {names}, "
                       f"which this interpreter cannot import",
                remedy=f"{_installer_hint()} {reqs}",
                error=ModuleNotFoundError(f"No module named {gone[0][0]!r}"),
            )
        tty = sys.stdin.isatty() if stdin_isatty is None else bool(stdin_isatty)
        if not tty:
            return SurfaceVerdict(
                kind=ENVIRONMENT,
                reason="stdin is not a terminal",
            )
        return SurfaceVerdict(kind=READY, reason="")
    except Exception as exc:  # noqa: BLE001 — a probe may never raise
        logger.debug("surface probe faulted", exc_info=True)
        # Unknown means NOT ready, and reported as an environment matter:
        # claiming a broken installation on evidence we could not gather
        # would send the operator to reinstall a package that is fine.
        return SurfaceVerdict(
            kind=ENVIRONMENT,
            reason=f"surface probe could not answer ({type(exc).__name__})",
            error=exc,
        )
