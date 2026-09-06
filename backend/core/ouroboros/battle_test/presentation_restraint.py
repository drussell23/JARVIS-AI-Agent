"""PresentationRestraint — CC-style minimal boot for O+V (with emojis kept).
============================================================================

Slice 1 of the **Gap #7 closure arc** (presentation restraint).

Root problem
------------

O+V's boot screen is a **dense dashboard** (~30 lines): preflight
checklist + shutdown-diagnostics startup logs + ASCII banner +
session metadata + 6-Layer Organism block + activity ribbon. Claude
Code's boot is **~5 lines** of essential context — the operator
pulls deeper info on demand via verbs (``/help``, ``/status``).

The dense dashboard isn't *wrong* — O+V is a more complex system
than CC and operators need that visibility. But pushing it ALL at
boot, on EVERY boot, is information overload that obscures the
prompt operators need to interact with. The fix is **pull-on-demand
restraint**: keep the rich content, surface it via REPL verbs
instead of forcing it into the boot stream.

Slice 1 scope
-------------

* :class:`MinimalWelcomePayload` frozen record (welcome text + the
  one-shot context line that follows it)
* :func:`render_minimal_welcome` — the CC-style panel renderer.
  Operator's emojis (``🐍``, ``⚙️ ``, etc.) are KEPT inside the panel
  — restraint is about **count and density**, not erasing identity.
* :func:`render_preflight` / :func:`render_organism` — the moved
  content. Same rendering used by ``/preflight`` / ``/organism`` REPL
  verbs.
* :func:`suppress_diagnostic_logs` — turns OFF propagation of
  ``jarvis.shutdown.diagnostics`` INFO logs to the root logger so
  the boot screen isn't littered with internal forensic startup
  noise (the file handler at DEBUG level still captures everything
  for post-hoc analysis).
* In-process **layer capture** — :func:`set_captured_layers` /
  :func:`get_captured_layers` so ``/organism`` re-renders the same
  data the harness computed at boot (avoids re-running expensive
  feature-detection logic).

Master flag :data:`MASTER_FLAG_ENV_VAR` defaults **false** during
this slice; Slice 5 graduation flips to true.

Architectural reuse — zero duplication
---------------------------------------

* ``Rich.Panel`` for the welcome panel (already used by `diff_preview`
  and `ouroboros_tui` — same primitive).
* :class:`StatusLineBuilder` snapshot for the context line below
  the panel (Gap #1+5 substrate).
* ``logging.Logger.propagate`` — stdlib mechanism for the diag log
  fix, not a parallel filter framework.
* House style: frozen dataclass, ``schema_version``, module-owned
  ``register_flags`` / ``register_shipped_invariants`` (Slice 5).

Authority boundary
------------------

* §1 deterministic — pure rendering + state read; no LLM, no I/O on
  the hot path
* §7 fail-closed — every helper degrades silently on bad input;
  rendering NEVER raises into the boot path
* §8 observable — captured-layers store is queryable for
  ``GET /observability/organism`` (Slice 5 follow-up)

What this module does NOT do
----------------------------

* Change palettes or emojis (Slice 2 owns palette discipline; emojis
  are kept by operator decision).
* Rewrite the legacy verbose render — kept verbatim under the
  master-flag-OFF branch for byte-identical rollback.
* Surface the new ``/preflight`` / ``/organism`` REPL handlers —
  those live in ``serpent_flow.py``'s :class:`SerpentREPL` and are
  added by the same slice.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from backend.core.ouroboros.ui import theme

logger = logging.getLogger("Ouroboros.PresentationRestraint")


# ===========================================================================
# Schema + master flag
# ===========================================================================


PRESENTATION_RESTRAINT_SCHEMA_VERSION: str = "presentation_restraint.v1"


MASTER_FLAG_ENV_VAR: str = "JARVIS_PRESENTATION_RESTRAINT_ENABLED"


def is_restraint_enabled() -> bool:
    """``JARVIS_PRESENTATION_RESTRAINT_ENABLED``. **Default true** post
    Slice 5 graduation (2026-05-04). Operators flip ``=false`` for
    instant rollback to the legacy verbose dashboard. NEVER raises."""
    raw = os.environ.get(MASTER_FLAG_ENV_VAR, "true")
    return raw.strip().lower() not in ("0", "false", "no", "off")


# ===========================================================================
# Frozen records
# ===========================================================================


@dataclass(frozen=True)
class MinimalWelcomePayload:
    """Frozen content of the minimal welcome surface.

    Two lines of essential context wrapped in one ``Rich.Panel``.
    The operator's emojis are preserved — restraint is about
    information density, not identity erasure.
    """

    title: str  # e.g. "🐍 OUROBOROS + VENOM"
    subtitle: str  # e.g. "autonomous coding organism"
    verb_hints: Tuple[Tuple[str, str], ...]
    # Each tuple is (verb, description), e.g. ("/preflight", "provider + venom + L2 status")
    cwd_str: str
    branch: str
    cost_cap_str: str  # e.g. "$0.50"
    idle_timeout_str: str  # e.g. "600s"
    mode_str: str
    schema_version: str = PRESENTATION_RESTRAINT_SCHEMA_VERSION


_DEFAULT_VERB_HINTS: Tuple[Tuple[str, str], ...] = (
    ("/help", "commands & verbs"),
    ("/preflight", "provider + venom + L2 status"),
    ("/organism", "6-layer trinity status"),
    ("/expand <ref>", "recover artifact (t-/d-/o-/n-)"),
)


# ===========================================================================
# In-process layer capture — backs /organism
# ===========================================================================


_captured_layers: Optional[Tuple[Tuple[str, str, bool, str], ...]] = None
_capture_lock = threading.Lock()


def set_captured_layers(
    layers: Optional[Sequence[Tuple[Any, ...]]],
) -> None:
    """Capture the harness-computed 6-layer status tuple so ``/organism``
    can re-render without re-running feature detection. Called by
    :func:`render_minimal_welcome` at boot. NEVER raises — bad input
    coerces to ``None`` (the verb then prints a graceful "no layers
    captured yet" hint)."""
    global _captured_layers
    if layers is None:
        with _capture_lock:
            _captured_layers = None
        return
    safe: List[Tuple[str, str, bool, str]] = []
    try:
        for entry in layers:
            if not isinstance(entry, (tuple, list)) or len(entry) < 4:
                continue
            icon = str(entry[0]) if entry[0] is not None else ""
            name = str(entry[1]) if entry[1] is not None else ""
            is_on = bool(entry[2])
            detail = str(entry[3]) if entry[3] is not None else ""
            safe.append((icon, name, is_on, detail))
    except (TypeError, ValueError):
        with _capture_lock:
            _captured_layers = None
        return
    with _capture_lock:
        _captured_layers = tuple(safe)


def get_captured_layers() -> Optional[Tuple[Tuple[str, str, bool, str], ...]]:
    """Return the most recently captured layers, or ``None`` if no
    boot has run yet (e.g. the REPL was started without a harness
    boot, like a smoke test)."""
    with _capture_lock:
        return _captured_layers


def clear_captured_layers_for_tests() -> None:
    """Test isolation hook."""
    global _captured_layers
    with _capture_lock:
        _captured_layers = None


# ===========================================================================
# Diagnostic-log suppression — boot-noise reduction
# ===========================================================================


_diag_propagation_originally: Optional[bool] = None


# ===========================================================================
# Boot-noise logger suppression — operator-visible boot quietness
# ===========================================================================
#
# These loggers fire DEBUG / INFO messages exclusively during boot:
# module discovery, kernel package init, graceful-shutdown setup,
# termination-hook registration. Operators don't need to see them on
# every run — the messages are forensic-only and matter only when
# debugging boot itself.


BOOT_NOISE_LOGGER_NAMES: Tuple[str, ...] = (
    "GracefulShutdown",
    "backend.kernel",  # parent — submodules inherit unless explicit
    "backend.core.ouroboros.battle_test.termination_hook_registry",
    "backend.core.ouroboros.governance.meta.module_discovery",
)


BOOT_NOISE_VERBOSE_ENV_VAR: str = "JARVIS_BOOT_NOISE_VERBOSE"


_boot_noise_levels_originally: dict = {}


def is_boot_noise_verbose() -> bool:
    """``JARVIS_BOOT_NOISE_VERBOSE``. Default false. Operators
    debugging boot issues set ``=true`` to keep the noisy loggers
    chatty even under restraint. NEVER raises."""
    raw = os.environ.get(BOOT_NOISE_VERBOSE_ENV_VAR, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def suppress_boot_noise_logs() -> int:
    """Raise the boot-noise loggers to ``WARNING`` so their DEBUG /
    INFO accounting doesn't litter the operator's boot screen.

    Their original level is captured per-logger so
    :func:`restore_boot_noise_logs_for_tests` can roll back. Skips
    the suppression entirely when
    :func:`is_boot_noise_verbose` is on (``JARVIS_BOOT_NOISE_VERBOSE``
    escape hatch for boot-debugging operators).

    Returns the count of loggers suppressed (0 when verbose mode is
    on or all calls failed). NEVER raises.
    """
    global _boot_noise_levels_originally
    if is_boot_noise_verbose():
        return 0
    suppressed = 0
    for name in BOOT_NOISE_LOGGER_NAMES:
        try:
            log = logging.getLogger(name)
            if name not in _boot_noise_levels_originally:
                _boot_noise_levels_originally[name] = log.level
            log.setLevel(logging.WARNING)
            suppressed += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "[PresentationRestraint] suppress_boot_noise(%s) failed",
                name, exc_info=True,
            )
    return suppressed


def restore_boot_noise_logs_for_tests() -> None:
    """Test isolation: restore captured original levels."""
    global _boot_noise_levels_originally
    for name, level in list(_boot_noise_levels_originally.items()):
        try:
            logging.getLogger(name).setLevel(level)
        except Exception:  # noqa: BLE001
            pass
    _boot_noise_levels_originally = {}


def suppress_diagnostic_logs() -> bool:
    """Stop the ``jarvis.shutdown.diagnostics`` logger from propagating
    INFO messages to the root logger (which has a default StreamHandler
    spilling forensic startup noise onto the operator's screen).

    The diagnostics module's own file handler keeps writing at DEBUG
    level — full forensics are preserved at
    ``~/.jarvis/trinity/shutdown_diagnostics.log``. Only the
    *operator-facing* INFO leak to stderr is suppressed.

    Returns ``True`` on success. Idempotent — calling twice is a no-op.
    NEVER raises.
    """
    global _diag_propagation_originally
    try:
        diag_logger = logging.getLogger("jarvis.shutdown.diagnostics")
        if _diag_propagation_originally is None:
            _diag_propagation_originally = diag_logger.propagate
        diag_logger.propagate = False
        return True
    except Exception:  # noqa: BLE001
        logger.debug(
            "[PresentationRestraint] suppress_diagnostic_logs failed",
            exc_info=True,
        )
        return False


def restore_diagnostic_logs_for_tests() -> None:
    """Test isolation: restore the original propagation state."""
    global _diag_propagation_originally
    if _diag_propagation_originally is None:
        return
    try:
        diag_logger = logging.getLogger("jarvis.shutdown.diagnostics")
        diag_logger.propagate = _diag_propagation_originally
    except Exception:  # noqa: BLE001
        pass
    _diag_propagation_originally = None


# ===========================================================================
# Render helpers — Rich-based, defensive
# ===========================================================================


def render_minimal_welcome(
    console: object,
    *,
    session_id: str = "",
    branch: str = "",
    cost_cap: float = 0.0,
    idle_timeout_s: float = 0.0,
    mode_str: str = "",
    cwd_str: str = "",
    verb_hints: Optional[Sequence[Tuple[str, str]]] = None,
    title: str = "OUROBOROS + VENOM",
    subtitle: str = "autonomous coding organism",
) -> bool:
    # ``session_id`` is accepted for caller compatibility (the harness
    # passes it) but deliberately NOT shown at boot — CC restraint:
    # operators retrieve it via ``/status``. Suppress unused-arg warnings.
    del session_id
    """Render the CC-style minimal welcome panel.

    Two output regions:

      * A bordered ``Rich.Panel`` containing title + subtitle +
        verb hints (3-4 lines).
      * A flat context line below the panel: ``cwd · branch ·
        budget · idle · mode``.

    NEVER raises. Returns ``True`` on success, ``False`` if the
    console / Rich primitives aren't available.
    """
    print_fn = getattr(console, "print", None)
    if not callable(print_fn):
        return False

    try:
        from rich.text import Text
    except ImportError:
        # Fallback: print plain lines without panel (still respects the
        # restraint contract — no multi-section dashboard).
        try:
            print_fn(f"\n  {title}")
            print_fn(f"  {subtitle}")
            for verb, desc in (verb_hints or _DEFAULT_VERB_HINTS):
                print_fn(f"    {verb}  {desc}")
            print_fn()
            print_fn(_format_context_line(
                cwd_str, branch, cost_cap, idle_timeout_s, mode_str,
            ))
            print_fn()
        except Exception:  # noqa: BLE001
            return False
        return True

    hints = verb_hints if verb_hints is not None else _DEFAULT_VERB_HINTS

    # Build the panel body. Rich accepts plain text or a Text instance.
    theme.ensure_theme(console)
    body = Text()
    body.append(f"{title}\n", style=theme.Token.HEADING.value)
    body.append(f"{subtitle}\n", style=theme.Token.MUTED.value)
    body.append("\n")
    # Verb hints — 2-space indent per CC's pattern.
    for verb, desc in hints:
        body.append(f"  {verb:<14s}", style=theme.Token.ACCENT.value)
        body.append(f"{desc}\n", style=theme.Token.MUTED.value)

    try:
        print_fn()
        # Single panel primitive — no hand-rolled Panel/box logic (DRY).
        theme.render_panel(console, body, token=theme.Token.MUTED)
        # Context line below the panel — matches CC's "cwd:" format.
        print_fn(_format_context_line(
            cwd_str, branch, cost_cap, idle_timeout_s, mode_str,
        ))
        print_fn()
        return True
    except Exception:  # noqa: BLE001
        logger.debug(
            "[PresentationRestraint] panel render failed",
            exc_info=True,
        )
        return False


def _format_context_line(
    cwd_str: str, branch: str,
    cost_cap: float, idle_timeout_s: float, mode_str: str,
) -> str:
    """One-line context summary. Plain Rich markup string."""
    parts: List[str] = []
    if cwd_str:
        parts.append(f"[muted]cwd:[/muted] {cwd_str}")
    if branch:
        parts.append(f"[muted]branch:[/muted] {branch}")
    parts.append(f"[muted]budget:[/muted] $0.00 / ${cost_cap:.2f}")
    if idle_timeout_s > 0:
        parts.append(f"[muted]idle:[/muted] 0s / {idle_timeout_s:.0f}s")
    if mode_str:
        parts.append(f"[muted]mode:[/muted] {mode_str}")
    sep = theme.mark("dot") or "-"
    return "  " + f"  [muted]{sep}[/muted]  ".join(parts)


def render_preflight(
    console: object,
    *,
    checks: Optional[Sequence[Mapping[str, Any]]] = None,
) -> bool:
    """Render the preflight checklist on demand. Used by ``/preflight``
    REPL verb.

    When ``checks`` is ``None``, the function re-runs the standard
    env-based feature detection (matches what
    ``scripts/ouroboros_battle_test.py:_print_preflight`` checks).

    Each check dict has keys: ``label`` (str), ``env_key`` (str),
    ``detail`` (str). The function reads ``os.environ`` per call so
    operators get a current snapshot, not a stale boot-time view.
    """
    print_fn = getattr(console, "print", None)
    if not callable(print_fn):
        return False

    if checks is None:
        checks = _default_preflight_checks()

    try:
        theme.ensure_theme(console)
        print_fn()
        print_fn("[heading]  Preflight Checklist[/heading]")
        theme.render_rule(console)
        for check in checks:
            label = str(check.get("label", "?"))
            env_key = str(check.get("env_key", ""))
            detail = str(check.get("detail", ""))
            is_on = bool(os.environ.get(env_key, ""))
            mark = (
                "[success]ON[/success]"
                if is_on else "[muted]OFF[/muted]"
            )
            print_fn(
                f"  [{mark}] {label:<30s} [muted]{detail}[/muted]"
            )
        # READ the ceiling, never re-derive it.
        #
        # This resolved the same variable with its own default of 10 while the
        # engine used 15, so with the variable unset — the default case — this
        # panel announced a ceiling that was not the one governing the loop.
        # A display that re-derives a limit is a second authority, and the
        # operator only ever sees the second one.
        #
        # If the authority cannot be reached, the line is omitted rather than
        # printed with a guess: an unmeasured ceiling is not a ceiling.
        try:
            from backend.core.ouroboros.governance.governed_loop_service import (
                configured_max_tool_rounds,
            )
            print_fn(
                f"\n[muted]  Tool rounds: {configured_max_tool_rounds()} "
                "(deadline-based, safety ceiling)[/muted]"
            )
        except Exception:  # noqa: BLE001 — a boot panel is never fatal
            pass
        # API-key warnings (matches the legacy script's behavior).
        has_dw = bool(os.environ.get("DOUBLEWORD_API_KEY"))
        has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not has_dw and not has_claude:
            print_fn(
                "  [danger]ERROR: No API keys set. "
                "Export DOUBLEWORD_API_KEY or ANTHROPIC_API_KEY.[/danger]"
            )
        elif not has_claude:
            print_fn(
                "  [warning]WARNING: ANTHROPIC_API_KEY not set "
                "— no Claude fallback.[/warning]"
            )
        elif not has_dw:
            print_fn(
                "  [warning]WARNING: DOUBLEWORD_API_KEY not set "
                "— Claude only (expensive).[/warning]"
            )
        print_fn()
        return True
    except Exception:  # noqa: BLE001
        logger.debug(
            "[PresentationRestraint] render_preflight failed",
            exc_info=True,
        )
        return False


def _default_preflight_checks() -> Tuple[Mapping[str, str], ...]:
    """Default preflight check tuple — matches the legacy
    ``_print_preflight`` content in the battle-test script.

    Single source of truth for the boot AND the ``/preflight`` REPL
    verb, so adding a new check requires editing one place.
    """
    l2_iters = os.environ.get("JARVIS_L2_MAX_ITERS", "5")
    l2_timebox = os.environ.get("JARVIS_L2_TIMEBOX_S", "120")
    return (
        {
            "label": "Provider: DoubleWord 397B",
            "env_key": "DOUBLEWORD_API_KEY",
            "detail": "$0.10/$0.40/M (Tier 0 PRIMARY)",
        },
        {
            "label": "Provider: Claude Sonnet",
            "env_key": "ANTHROPIC_API_KEY",
            "detail": "$3/$15/M (Tier 1 FALLBACK)",
        },
        {
            "label": "Venom: Tool Loop",
            "env_key": "JARVIS_GOVERNED_TOOL_USE_ENABLED",
            "detail": "read_file, search_code, get_callers, list_symbols",
        },
        {
            "label": "Venom: Bash (100+ cmds)",
            "env_key": "JARVIS_BASH_TOOL_ENABLED",
            "detail": "python, git, docker, curl, terraform...",
        },
        {
            "label": "Venom: Web Search",
            "env_key": "JARVIS_WEB_TOOL_ENABLED",
            "detail": "DuckDuckGo / Brave / Google CSE",
        },
        {
            "label": "Venom: Run Tests",
            "env_key": "JARVIS_TOOL_RUN_TESTS_ALLOWED",
            "detail": "pytest in sandbox during generation",
        },
        {
            "label": "L2 Repair Engine",
            "env_key": "JARVIS_L2_ENABLED",
            "detail": f"max {l2_iters} iters, {l2_timebox}s timebox",
        },
        {
            "label": "Trinity Consciousness",
            "env_key": "JARVIS_CONSCIOUSNESS_ENABLED",
            "detail": "Memory + Prophecy + Health",
        },
    )


def render_organism(
    console: object,
    *,
    layers: Optional[Sequence[Tuple[Any, ...]]] = None,
) -> bool:
    """Render the 6-layer organism block on demand. Used by
    ``/organism`` REPL verb.

    When ``layers`` is ``None``, queries :func:`get_captured_layers`
    for the harness-computed boot snapshot. If no snapshot is
    available, prints a graceful hint.
    """
    print_fn = getattr(console, "print", None)
    if not callable(print_fn):
        return False

    theme.ensure_theme(console)
    if layers is None:
        layers = get_captured_layers()

    if not layers:
        try:
            print_fn(
                "[muted]  No organism state captured yet — "
                "harness boot has not completed.[/muted]"
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    try:
        print_fn()
        theme.render_rule(console, label="6-Layer Organism")
        for entry in layers:
            if not isinstance(entry, (tuple, list)) or len(entry) < 4:
                continue
            # entry[0] is the legacy emoji icon — dropped under Restrained Mono.
            name = str(entry[1]) if entry[1] is not None else ""
            is_on = bool(entry[2])
            detail = str(entry[3]) if entry[3] is not None else ""
            mark = (
                "[success]ON[/success]"
                if is_on else "[muted]OFF[/muted]"
            )
            print_fn(
                f"  {name:<24s} {mark}  [muted]{detail}[/muted]"
            )
        print_fn()
        return True
    except Exception:  # noqa: BLE001
        logger.debug(
            "[PresentationRestraint] render_organism failed",
            exc_info=True,
        )
        return False


__all__ = [
    "MASTER_FLAG_ENV_VAR",
    "MinimalWelcomePayload",
    "PRESENTATION_RESTRAINT_SCHEMA_VERSION",
    "chrome_color",
    "clear_captured_layers_for_tests",
    "format_idle_breadcrumb",
    "get_captured_layers",
    "is_restraint_enabled",
    "real_stdout_isatty",
    "register_flags",
    "register_shipped_invariants",
    "render_minimal_welcome",
    "render_organism",
    "render_preflight",
    "restore_diagnostic_logs_for_tests",
    "set_captured_layers",
    "suppress_diagnostic_logs",
]


# ===========================================================================
# Slice 5 — FlagRegistry self-registration (umbrella for the Gap #7 arc)
# ===========================================================================


def register_flags(registry) -> int:
    """Module-owned FlagRegistry registration for the Gap #7 arc.
    Auto-discovered via the ``battle_test`` entry in
    ``_FLAG_PROVIDER_PACKAGES``. Returns count of FlagSpecs added.
    NEVER raises."""
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category, FlagSpec, FlagType,
        )
    except ImportError:
        return 0

    specs = [
        # ── Slice 1+2: presentation restraint umbrella ────────────
        FlagSpec(
            name=MASTER_FLAG_ENV_VAR,  # JARVIS_PRESENTATION_RESTRAINT_ENABLED
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for the CC-style minimal boot + "
                "color discipline (Gap #7 Slices 1+2). When false: "
                "verbose multi-section dashboard returns; preflight + "
                "shutdown-diagnostics + 6-layer block all render at "
                "boot; chrome uses bright_green. Default TRUE post "
                "graduation 2026-05-04."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/battle_test/presentation_restraint.py"
            ),
            example="true",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
        # ── Slice 3: REPL completion ──────────────────────────────
        FlagSpec(
            name="JARVIS_REPL_COMPLETION_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for the auto-discovered slash-"
                "command palette + tab completion (Gap #7 Slice 3). "
                "When false: PromptSession runs without a completer "
                "(operators must type full verb names from memory). "
                "Default TRUE post graduation. Verbs are discovered "
                "from SerpentREPL._handle_* methods at boot — "
                "structurally automatic, no hardcoded list."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/battle_test/repl_completion.py"
            ),
            example="true",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_REPL_HISTORY_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Persistent REPL history (↑/↓/Ctrl+R) across sessions. "
                "Default TRUE — conventional shell behavior. Operators "
                "set ``=false`` for confidentiality (no history file "
                "on disk)."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/repl_completion.py"
            ),
            example="true",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_REPL_HISTORY_FILE",
            type=FlagType.STR,
            default="",
            description=(
                "Override the REPL history file path. Empty (default) "
                "means ``.jarvis/repl_history`` relative to cwd "
                "(per-project history). Useful for shared history "
                "across multiple O+V projects via an absolute path."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/repl_completion.py"
            ),
            example="~/.jarvis/repl_history",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
        # ── Sovereign Terminal UI: borderless render + pulse ──────
        FlagSpec(
            name="JARVIS_OPBLOCK_BORDERLESS_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Borderless Claude-Code-clean op-block render: glyph "
                "action/result hierarchy (no box-drawing borders), "
                "grayscale chrome, vertical rhythm. Gated by the "
                "presentation-restraint master; when that master is off "
                "the legacy boxed renderer is byte-identical. Default TRUE."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/battle_test/presentation_restraint.py"
            ),
            example="true",
            since="Sovereign Terminal UI (2026-06-15)",
        ),
        FlagSpec(
            name="JARVIS_TUI_PULSE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Async pulse spinner on the active action line during "
                "synthesizing/validating awaits. TTY-gated (no-op "
                "headless/CI). Default TRUE under the restraint master."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/presentation_restraint.py"
            ),
            example="true",
            since="Sovereign Terminal UI (2026-06-15)",
        ),
        # ── Slice 4: input polish ─────────────────────────────────
        FlagSpec(
            name="JARVIS_REPL_INPUT_POLISH_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for input ergonomics (Gap #7 Slice "
                "4): @filepath mention extraction + Esc-to-cancel + "
                "terminal title updates. When false: operators must "
                "use explicit /attach + Ctrl+C; terminal title stays "
                "at the shell default. Default TRUE post graduation."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/battle_test/repl_input_polish.py"
            ),
            example="true",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_TERMINAL_TITLE_ENABLED",
            type=FlagType.BOOL,
            default=True,
            description=(
                "Title bar updates (OSC 0). Inherits from "
                "JARVIS_REPL_INPUT_POLISH_ENABLED when unset. "
                "Operators on tmux / terminal multiplexers may "
                "set ``=false`` to avoid title-fight contention."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/repl_input_polish.py"
            ),
            example="true",
            since="Gap #7 Slice 5 (2026-05-04)",
        ),
    ]

    count = 0
    for spec in specs:
        try:
            registry.register(spec)
            count += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "[PresentationRestraint] flag registration failed for %s",
                getattr(spec, "name", "?"), exc_info=True,
            )
    return count


# ===========================================================================
# Slice 5 — shipped_code_invariants self-registration
# ===========================================================================


def register_shipped_invariants() -> list:
    """Module-owned shipped-code AST pins for the Gap #7 arc.

    Five structural invariants:

      1. ``presentation_restraint_default_true`` — the master flag
         must default to ``"true"`` (graduated). Without this pin a
         future env-default-flip silently regresses operators back to
         the verbose dashboard.
      2. ``boot_banner_short_circuits_under_restraint`` — BUG-FIX
         REGRESSION PIN: ``serpent_flow.boot_banner`` must check
         ``is_restraint_enabled()`` and route to
         ``render_minimal_welcome``. Without this, the minimal-welcome
         path is dead code.
      3. ``repl_loop_wires_completion_and_polish`` — the
         ``SerpentREPL._loop`` method must invoke both
         ``build_completion_wiring`` and the input-polish helpers
         (extract_attachments, make_esc_cancel_binding). Single
         regression pin covers both Slice 3 and Slice 4 wiring.
      4. ``op_lifecycle_sets_terminal_title`` — op_started /
         op_completed / op_failed must each call
         ``_maybe_set_terminal_title``. Without this, terminal title
         silently regresses to the shell default mid-op.
      5. ``status_line_uses_real_stdout_isatty`` — the TTY gate
         must use ``real_stdout_isatty`` (not raw
         ``sys.stdout.isatty``) so the live status line surfaces
         under ``patch_stdout``. Slice 2 fix that this pin guards.

    NEVER raises (returns ``[]`` on import failure).
    """
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except ImportError:
        return []

    import ast as _ast

    def _validate_default_true(_tree, source) -> tuple:
        del _tree
        # Look for the env get with "true" default
        if 'os.environ.get(MASTER_FLAG_ENV_VAR, "true")' not in source:
            return (
                "is_restraint_enabled() must default to 'true' — "
                "the env get's second argument (the default) is the "
                "graduation marker. Reverting to '' or 'false' "
                "silently regresses operators to the verbose "
                "dashboard.",
            )
        return ()

    def _validate_boot_banner_short_circuit(tree, _source) -> tuple:
        del _source
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if node.name == "boot_banner":
                    body = _ast.unparse(node)
                    violations = []
                    if "is_restraint_enabled" not in body:
                        violations.append(
                            "boot_banner missing is_restraint_enabled() "
                            "check — minimal-welcome path is dead code"
                        )
                    if "render_minimal_welcome" not in body:
                        violations.append(
                            "boot_banner missing render_minimal_welcome "
                            "call — Gap #7 Slice 1 hook regressed"
                        )
                    return tuple(violations)
        return ("boot_banner method not found in serpent_flow.py",)

    def _validate_repl_loop_wires_polish(tree, _source) -> tuple:
        del _source
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if node.name == "_loop":
                    body = _ast.unparse(node)
                    violations = []
                    if "build_completion_wiring" not in body:
                        violations.append(
                            "_loop missing build_completion_wiring call "
                            "— Slice 3 palette + history regressed"
                        )
                    if "extract_attachments" not in body:
                        violations.append(
                            "_loop missing extract_attachments call — "
                            "Slice 4 @filepath mention regressed"
                        )
                    if "make_esc_cancel_binding" not in body:
                        violations.append(
                            "_loop missing make_esc_cancel_binding call "
                            "— Slice 4 Esc-to-cancel regressed"
                        )
                    return tuple(violations)
        return ("_loop method not found in SerpentREPL",)

    def _validate_op_lifecycle_title(tree, _source) -> tuple:
        del _source
        required = {"op_started", "op_completed", "op_failed"}
        violations = []
        seen: set = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if node.name in required:
                    seen.add(node.name)
                    body = _ast.unparse(node)
                    if "_maybe_set_terminal_title" not in body:
                        violations.append(
                            f"{node.name} missing "
                            "_maybe_set_terminal_title call — "
                            "Slice 4 terminal title regressed"
                        )
        missing_methods = required - seen
        if missing_methods:
            violations.append(
                f"missing methods: {sorted(missing_methods)}"
            )
        return tuple(violations)

    def _validate_status_line_uses_real_stdout(tree, _source) -> tuple:
        del _source
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == "should_render":
                body = _ast.unparse(node)
                if "real_stdout_isatty" not in body:
                    return (
                        "should_render() must use real_stdout_isatty "
                        "(not direct sys.stdout.isatty) — Slice 2 TTY "
                        "gate fix regressed; status line will silently "
                        "stop appearing under patch_stdout",
                    )
                return ()
        return ("should_render not found in status_line.py",)

    return [
        ShippedCodeInvariant(
            invariant_name="presentation_restraint_default_true",
            target_file=(
                "backend/core/ouroboros/battle_test/presentation_restraint.py"
            ),
            description=(
                "BUG-FIX REGRESSION PIN: the master flag's env-get "
                "default must remain 'true' post graduation. Reverting "
                "silently regresses operators to the verbose dashboard."
            ),
            validate=_validate_default_true,
        ),
        ShippedCodeInvariant(
            invariant_name="boot_banner_short_circuits_under_restraint",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "boot_banner must check is_restraint_enabled() and "
                "route to render_minimal_welcome — without this the "
                "Slice 1 minimal-welcome path is unreachable."
            ),
            validate=_validate_boot_banner_short_circuit,
        ),
        ShippedCodeInvariant(
            invariant_name="repl_loop_wires_completion_and_polish",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "SerpentREPL._loop must wire build_completion_wiring + "
                "extract_attachments + make_esc_cancel_binding — these "
                "are the operator-visible deliverables of Slices 3+4."
            ),
            validate=_validate_repl_loop_wires_polish,
        ),
        ShippedCodeInvariant(
            invariant_name="op_lifecycle_sets_terminal_title",
            target_file=(
                "backend/core/ouroboros/battle_test/serpent_flow.py"
            ),
            description=(
                "op_started / op_completed / op_failed must each call "
                "_maybe_set_terminal_title — Slice 4 terminal title "
                "regression pin."
            ),
            validate=_validate_op_lifecycle_title,
        ),
        ShippedCodeInvariant(
            invariant_name="status_line_uses_real_stdout_isatty",
            target_file=(
                "backend/core/ouroboros/battle_test/status_line.py"
            ),
            description=(
                "should_render() must use real_stdout_isatty — Slice 2 "
                "TTY gate fix. Without this, the live status line "
                "silently stops appearing under prompt_toolkit's "
                "patch_stdout proxy."
            ),
            validate=_validate_status_line_uses_real_stdout,
        ),
    ]


# ===========================================================================
# Slice 2 — chrome color discipline + idle status content + TTY gate fix
# ===========================================================================


def chrome_color(default: str = "bright_green") -> str:
    """Return ``"dim"`` when presentation restraint is enabled,
    otherwise ``default``.

    **Rationale (Constraint: green = success outcomes only).** The
    legacy palette uses ``bright_green`` (``_C['life']``) for both
    *outcomes* (✨ evolved, ✓ test passed, 📝 committed) AND *chrome*
    (event-stream activity ribbon, "🔋 Organism alive" line, section
    headers). Operators can't distinguish "this op succeeded" from
    "boot decoration" at a glance.

    This helper lets callers thread a master-flag-aware color choice
    into a single seam without changing the legacy palette dict (which
    is shared across many other rendering paths). When restraint is
    on: chrome turns dim; outcomes (which call sites pass their own
    explicit color, not via this helper) stay bright_green.

    NEVER raises.
    """
    if is_restraint_enabled():
        return "dim"
    return _theme_adapted(default)


def _theme_adapted(color: str) -> str:
    """Adjust a colour for the terminal that will actually display it.

    The daemon picked this palette against an assumed dark background, and
    until `terminal_capabilities` existed there was no way to know better.
    ``bright_*`` on a light terminal is the specific failure: bright green on
    white is close to unreadable, and it is the colour reserved for OUTCOMES —
    the one an operator most needs to see.

    Only ``bright_`` is demoted, and only on a terminal that DECLARED itself
    light. ``unknown`` changes nothing: a guess here would repaint every
    foreground daemon on no evidence, and "I was not told" is not "it is
    light". NEVER raises — a colour lookup on a render path must not be able
    to fail.
    """
    try:
        from backend.core.ouroboros.battle_test.terminal_capabilities import (
            effective_theme,
        )
        if effective_theme() != "light":
            return color
        c = str(color)
        return c.replace("bright_", "", 1) if c.startswith("bright_") else c
    except Exception:  # noqa: BLE001
        return color


def real_stdout_isatty() -> bool:
    """Check the **unpatched** stdout's TTY status.

    ``prompt_toolkit.patch_stdout`` replaces ``sys.stdout`` with a
    non-TTY proxy during the REPL's lifetime. Code that checks
    ``sys.stdout.isatty()`` during REPL operation gets ``False``
    even on real interactive terminals — that's why the live status
    line (Gap #1+5) never surfaced during normal use.

    This helper checks ``sys.__stdout__`` (Python's saved reference
    to the original stdout, untouched by patch_stdout). Falls back
    to the patched ``sys.stdout`` only if ``__stdout__`` is ``None``
    (rare: Windows pythonw, daemonized processes). NEVER raises.
    """
    import sys
    primary = getattr(sys, "__stdout__", None)
    if primary is not None:
        try:
            return bool(primary.isatty())
        except Exception:  # noqa: BLE001
            pass
    try:
        return bool(sys.stdout.isatty())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Encoding-aware glyph vocabulary (ASCII fallback for non-UTF-8 terminals)
# ---------------------------------------------------------------------------

#: The two marks this module hands out, by their design-language names.
#: The glyphs themselves live in ``theme._GLYPHS`` — ONE table, ONE
#: degradation. This module used to keep a second copy, with a different
#: ASCII form for the continuation mark, and a probe of its own.
_GLYPH_MARKS = {"action": "action", "result": "detail"}


def _stdout_supports_utf8() -> bool:
    """True only when stdout can encode our glyphs. Fail-safe to False."""
    try:
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        return "utf" in enc
    except Exception:  # noqa: BLE001
        return False


def glyphs() -> dict:
    """Glyph vocabulary from the design-language table, degraded for the
    terminal that will DISPLAY it. The theme asks an attached cockpit
    before the local locale; stdout's encoding is the wrong question for a
    daemon rendering for a terminal it does not own."""
    return {name: theme.mark(mark) for name, mark in _GLYPH_MARKS.items()}


def borderless_enabled() -> bool:
    """Borderless glyph op-block render. Default TRUE under the restraint master;
    the master gates it so master-off is byte-identical legacy boxed rendering."""
    if not is_restraint_enabled():
        return False
    raw = os.environ.get("JARVIS_OPBLOCK_BORDERLESS_ENABLED", "true")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def pulse_enabled() -> bool:
    """Async pulse spinner. Default TRUE under the restraint master."""
    if not is_restraint_enabled():
        return False
    raw = os.environ.get("JARVIS_TUI_PULSE_ENABLED", "true")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def spinner_name() -> str:
    """Rich spinner name: braille 'dots' on UTF-8, ASCII 'line' otherwise."""
    return "dots" if _stdout_supports_utf8() else "line"


def print_fit(console, markup: str, **print_kw: Any) -> None:
    """Print one op-block line, truncated to the live console width with an
    ellipsis -- never wraps (so the glyph column never moves). Width is read
    from the console per call (SIGWINCH-adaptive; Rich defaults to 80
    off-terminal). Fail-soft: on any Rich error, falls back to a plain crop
    print, and if that fails too, swallows the error rather than crash the
    render path.

    ``print_kw`` rides through to BOTH prints unchanged. A caller that has
    already mirrored the line to the cockpit passes ``mirror=False`` so a
    relaying console does not carry it twice; this function has no opinion
    about that and must not eat the request on the fallback path either."""
    try:
        from rich.text import Text
        console.print(
            Text.from_markup(markup),
            no_wrap=True, overflow="ellipsis", crop=True, soft_wrap=False,
            **print_kw,
        )
    except Exception:  # noqa: BLE001
        try:
            width = getattr(console, "width", 80) or 80
            console.print(str(markup)[: max(8, int(width) - 1)],
                          highlight=False, **print_kw)
        except Exception:  # noqa: BLE001
            pass


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def pulse(console, line: str, *, spinner: str = ""):
    """Non-blocking spinner on the active action line during awaited work.

    TTY-gated -- a no-op headless/CI/piped (no spinner art leaks into logs; the
    awaited body still runs). Cursor armor: a try/finally GUARANTEES the cursor
    is restored and the buffer flushed on ANY exception boundary (fatal LLM /
    network error mid-spin must never leave the operator's terminal with a
    hidden cursor). Leverages Rich ``console.status`` (background-thread spinner
    + cursor management); raw ``\\033[?25h`` is only the last-resort fallback."""
    if not real_stdout_isatty():
        yield
        return
    spin = spinner or spinner_name()
    status = None
    try:
        status = console.status(line, spinner=spin)
        status.start()
        yield
    finally:
        try:
            if status is not None:
                status.stop()          # clears spinner region, restores cursor
        except Exception:  # noqa: BLE001
            pass
        try:
            console.show_cursor(True)  # Rich-native cursor restore
        except Exception:  # noqa: BLE001
            try:
                sys.stdout.write("\033[?25h")   # last-resort raw ANSI show-cursor
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass


def format_idle_breadcrumb(
    *,
    branch: str = "",
    cost_spent: float = 0.0,
    cost_budget: float = 0.0,
    posture: str = "",
    op_id: str = "",
    detail: str = "",
    funding: str = "",
    funding_label: str = "",
) -> str:
    """One-line breadcrumb for the IDLE phase — replaces the silent
    empty status line that today leaves operators wondering whether
    O+V is alive.

    Returns a plain text string the caller wraps with their own
    Rich markup at the emission seam (matches ``status_line.py``'s
    library-agnostic policy — the substrate produces strings, the
    consumer applies styling).

    Format: ``IDLE · main · $0.04/$0.50 · EXPLORE`` — only includes
    fields the caller supplies. NEVER raises.
    """
    # "IDLE" plus the evidence for it, when the caller has any.
    #
    # This used to be the bare literal, and the literal was the bug: the
    # sampler upstream had already worked out that the organism was idle with
    # 17 sensors armed, or that 4 signals were queued and no op had claimed
    # them — and every word of that was discarded here, because this function
    # re-asserted the state instead of rendering the one it was handed.
    #
    # An operator watching a real boot therefore read `IDLE` for two minutes
    # while four genuine test failures sat in the queue, and could not tell it
    # apart from an organism with nothing to do. The detail IS the difference
    # between those two, which makes dropping it the whole defect.
    parts: List[str] = ["IDLE"]
    if isinstance(detail, str) and detail:
        parts[0] = f"IDLE · {detail}"
    if isinstance(branch, str) and branch:
        parts.append(branch)
    # The ceiling is a POLICY cap derived from spend HISTORY (p95 of prior
    # sessions x3), not a balance. Rendered bare it reads as headroom, and an
    # operator watching `$0.00/$0.71` over two empty accounts is reading a
    # number nothing can spend. `funding` says which kind of number it is;
    # the cap is QUALIFIED, never deleted, because it is still what stops a
    # runaway session the moment a lane is funded again.
    if cost_budget > 0.0:
        chip = f"${cost_spent:.2f}/${cost_budget:.2f}"
        if funding == "local":
            # Every paid lane is dry, but a sovereign tier is serving. This is
            # the case a flat "unfunded" gets exactly backwards: the organism
            # is WORKING, at zero marginal cost, and a dollar ceiling is no
            # longer the binding constraint. Show what is actually true — what
            # has been spent — and name the tier carrying the work.
            chip = f"${cost_spent:.2f} · local tier"
            if funding_label:
                chip += f" ({funding_label})"
        elif funding == "unfunded":
            # Nothing can spend against it and nothing else can serve.
            chip += " unfunded"
        elif funding == "partial":
            # FRACTIONAL — and the reason this is not folded into "unfunded".
            # One lane out does NOT make the ceiling inert: the funded lane can
            # still spend the whole of it, so the denominator stays honest and
            # only gains the name of the lane that is out. Collapsing this to a
            # blanket verdict would tell an operator with a working Anthropic
            # key that their organism is broke.
            chip += f" · ⚠ {funding_label} dry" if funding_label else " · ⚠ dry"
        parts.append(chip)
    elif cost_spent > 0.0:
        parts.append(f"${cost_spent:.2f}")
    if isinstance(posture, str) and posture:
        parts.append(posture)
    if isinstance(op_id, str) and op_id:
        # Last-completed op tail (8 chars) so operators see "previous
        # work" hint when scrolling away from the receipt
        tail = op_id.split("-")[-1][:8] if "-" in op_id else op_id[:8]
        parts.append(f"prev:{tail}")
    return " · ".join(parts)
