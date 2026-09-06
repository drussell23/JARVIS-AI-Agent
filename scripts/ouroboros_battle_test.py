#!/usr/bin/env python3
"""
Ouroboros Battle Test Runner
============================

Boots the full Ouroboros + Venom + Trinity Consciousness stack as a
headless daemon that autonomously detects, generates, validates, and
commits code improvements.

6-Layer Architecture:
  1. Strategic Direction  — Manifesto principles injected into every prompt
  2. Trinity Consciousness — Memory, prediction, cross-session learning
  3. Event Spine           — FileWatchGuard + TrinityEventBus, <1s detection
  4. Ouroboros Pipeline    — Governance, adaptive 3-tier routing, parallel ops
  5. Venom Agentic Loop    — read_file, bash, web_search, run_tests, L2 repair
  6. Thought Log           — Observable reasoning, signed commits

Usage::

    python3 scripts/ouroboros_battle_test.py [options]
    python3 scripts/ouroboros_battle_test.py --help

Examples::

    # Default: $0.50 budget, 600s idle timeout, verbose
    python3 scripts/ouroboros_battle_test.py -v

    # Extended session: $2.00 budget, 30 min idle
    python3 scripts/ouroboros_battle_test.py --cost-cap 2.00 --idle-timeout 1800 -v

    # Quick test: $0.10 budget, 2 min idle
    python3 scripts/ouroboros_battle_test.py --cost-cap 0.10 --idle-timeout 120 -v
"""
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Slice 12X Phase 1 — ABSOLUTE BOOT-TIME SUPREMACY                 ║
# ║                                                                  ║
# ║ This block MUST run before any other import — including          ║
# ║ ``backend.*`` and any third-party module that transitively       ║
# ║ imports chromadb / posthog. bt-2026-05-23-204519 proved Slice    ║
# ║ 12W's __init__-time exorcism fired TOO LATE: posthog's consumer  ║
# ║ thread + aiosqlite's per-connection workers had already spawned  ║
# ║ as non-daemon during module-load. By the time                    ║
# ║ ``BattleTestHarness.__init__`` ran ``exorcise_at_boot``, the     ║
# ║ rogue threads were alive and blocked ``threading._shutdown`` at  ║
# ║ teardown.                                                        ║
# ║                                                                  ║
# ║ Only stdlib (``os``, ``sys``) is imported here — these never     ║
# ║ trigger third-party module loads. The env vars are set via       ║
# ║ ``os.environ.setdefault`` so operator overrides survive (e.g.,   ║
# ║ a debugging session that explicitly enables telemetry). A boot   ║
# ║ marker line is written to ``sys.stderr`` so operators can see    ║
# ║ the exorcism executed before ``backend.*`` imports — visible     ║
# ║ in any tee/pipe redirection regardless of the file-logger        ║
# ║ wire-up state.                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
import os as _os_for_boot_exorcism
import sys as _sys_for_boot_exorcism

# Closed table of (env_var, value). Mirrors
# ``rogue_thread_exorcism._TELEMETRY_DEFAULTS`` exactly — duplication
# is intentional and load-bearing because this block runs BEFORE
# ``backend.*`` is importable. The dedicated test
# ``test_telemetry_defaults_match_rogue_thread_exorcism`` AST-pins
# the two tables to stay in sync.
_SLICE12X_TELEMETRY_DEFAULTS = (
    ("ANONYMIZED_TELEMETRY", "False"),
    ("POSTHOG_DISABLED", "True"),
    ("OTEL_SDK_DISABLED", "true"),
)
_slice12x_applied: list = []
for _env_var, _env_value in _SLICE12X_TELEMETRY_DEFAULTS:
    try:
        if _env_var not in _os_for_boot_exorcism.environ:
            _os_for_boot_exorcism.environ[_env_var] = _env_value
            _slice12x_applied.append(_env_var)
    except Exception:  # noqa: BLE001 — never raise during boot
        pass
# ov cockpit silence (Slice 2) — this pure-ceremony marker is gated on
# presentation mode. Only ``os``/``sys`` are importable this early
# (see the block comment above), so we can't import
# ``presentation_mode.resolve_presentation_mode`` yet; the raw env
# check below mirrors that function's exact resolution logic
# (stripped + lowercased comparison, fail-safe to SOAK/print).
_ov_mode_for_boot_exorcism = (
    _os_for_boot_exorcism.environ.get("JARVIS_OV_PRESENTATION") or ""
).strip().lower()
try:
    if _ov_mode_for_boot_exorcism != "cockpit":
        _sys_for_boot_exorcism.stderr.write(
            "[Slice12X.BootExorcism] script-top env hygiene applied — "
            "pre-import telemetry kill switches set: "
            + (",".join(_slice12x_applied) if _slice12x_applied
               else "<none-needed, all pre-set>")
            + " (operator overrides preserved via setdefault)\n"
        )
        _sys_for_boot_exorcism.stderr.flush()
except Exception:  # noqa: BLE001 — never raise during boot
    pass

import argparse
import asyncio
import atexit
import contextlib
import importlib.metadata as _metadata
import logging
import os
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any, Optional

# Python 3.9 compat: patch packages_distributions before any library touches it
if not hasattr(_metadata, "packages_distributions"):
    def _packages_distributions_fallback():  # type: ignore[misc]
        """Minimal fallback for packages_distributions on Python <3.11."""
        try:
            from importlib_metadata import packages_distributions  # type: ignore[import-untyped]
            return packages_distributions()
        except Exception:
            return {}
    _metadata.packages_distributions = _packages_distributions_fallback  # type: ignore[attr-defined]

# Suppress noisy warnings that leak to terminal (urllib3, google, etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

# REPL UX fix (2026-05-03) — silence the per-fork tokenizers warning at its
# source. The warning prints to OS-level stderr from forked subprocesses,
# which bypasses prompt_toolkit's patch_stdout entirely and clobbers the
# REPL prompt line on every sensor fork. The huggingface library itself
# tells you to set this env var; we honor any operator override via
# setdefault so it isn't hardcoded.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# macOS fork-safety fix (2026-05-03) — Apple's libdispatch / Objective-C
# runtime aborts (SIGABRT, "multi-threaded process forked / crashed on
# child side of fork pre-exec") when fork() is called after threads
# have started. The harness goes multi-threaded at boot (asyncio +
# sensors + embedding service), and downstream subprocess.Popen calls
# (git, pytest, model downloads) use fork+exec on macOS. Apple ships
# this env var EXPLICITLY for the fork+exec case where the child
# replaces its image immediately — it is the documented fix, not a
# workaround. setdefault honors any operator override.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# REPL UX fix (2026-05-03) — prompt_toolkit CPR (Cursor Position
# Request) bypass. Some terminals don't respond to CPR escape
# sequences; prompt_toolkit then prints "WARNING: your terminal
# doesn't support cursor position requests (CPR)." DIRECTLY to
# stderr (bypassing patch_stdout) AND falls into a degraded
# rendering codepath where input characters may not display.
# PROMPT_TOOLKIT_NO_CPR=1 is the library's documented escape
# hatch — it skips the CPR query entirely, uses static-size
# detection, and uses the safe non-CPR rendering path that always
# shows typed input. Reference:
#   prompt_toolkit/output/vt100.py:Vt100_Output.responds_to_cpr
# setdefault honors any operator override.
os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

# Ensure the project root is importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── THE DAEMON'S CONFIGURATION FLOOR ──────────────────────────────────────
#
# `env_bootstrap`'s own docstring says "every process boundary
# (unified_supervisor, backend.main, THE OV THIN-CLIENT DAEMON) calls
# load_env_once at its highest bootstrap point". This one did not, and the
# promise went unkept for as long as nobody read the two files together.
#
# What that cost, measured 2026-09-06: `spawn_daemon` copies the launching
# process's environment into the child, a login shell has no
# JARVIS_LOCAL_MODEL_NAME, and nothing downstream ever read `.env`. So a
# cold `ov` from a fresh terminal booted an organism with NO model pin,
# selection fell through to "largest by size", and the host answered from
# `qwen2.5-coder:32b` (19.85 GB) instead of the fine-tuned
# `qwen3-coder-ov:30b` (18.58 GB) -- a different model family, chosen
# silently, with the operator's own pin sitting unread in `.env`.
#
# It belongs HERE rather than in the harness: this is the first line after
# the repo becomes importable and before ANY `backend.*` module is loaded,
# so no module can capture a config value before the file that supplies it
# has been read. Later is not equivalent -- module-scope `os.environ.get`
# calls are evaluated at import time.
#
# Precedence is the loaded module's, not a second policy invented here:
# `override=False` means a real exported variable always wins and `.env`
# supplies DEFAULTS. That IS the resolution cascade -- explicit
# environment, then `.env`, then each reader's own documented default.
# Idempotent and never-raising by contract, so a missing python-dotenv or
# a malformed file degrades to "use the environment as-is" rather than
# blocking a boot.
from backend.core.env_bootstrap import load_env_once as _load_env_once  # noqa: E402

_load_env_once()

# ov awakening Task 1 — presentation-mode gate (COCKPIT vs SOAK). Leaf
# module: stdlib only, safe to import at top level now that _PROJECT_ROOT
# is on sys.path. See backend/core/ouroboros/ui/presentation_mode.py.
from backend.core.ouroboros.ui.presentation_mode import (  # noqa: E402
    PresentationMode, resolve_presentation_mode,
)

# ANSI color codes
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# API keys that .env should always override (stale shell exports are a
# common source of 401 errors during battle test).  Everything else uses
# setdefault so explicit `env VAR=val cmd` still works for non-secret config.
_FORCE_OVERRIDE_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "DOUBLEWORD_API_KEY",
})


def _load_env_files() -> None:
    """Apply the two overlays the canonical loader does not cover.

    ## This is no longer a parser

    It used to be a second, hand-rolled ``.env`` reader living alongside
    ``backend.core.env_bootstrap.load_env_once`` -- two implementations of
    one job, and they did not agree: the canonical loader is strictly
    ``override=False`` while this one force-overrode the API keys. A
    variable's effective value therefore depended on which loader last
    touched it, and nothing said so.

    Splitting a difference in policy from a difference in CODE is the
    whole fix. The canonical loader now runs at the top of this module,
    before any ``backend.*`` import, so a module reading ``os.environ`` at
    import scope sees a populated environment. What remains here is only
    what that loader deliberately does not do, each stated as a rule:

      1. ``backend/.env`` — a second file the canonical loader does not
         know about. Same precedence: the environment wins.
      2. The API keys — the ONE documented exception, where the file beats
         a stale shell export, because an expired key exported months ago
         produces a silent 401 rather than an error anyone can read.

    Parsing itself is `dotenv.dotenv_values`, the same library the
    canonical loader uses, so quoting, ``export`` prefixes and escapes are
    handled one way. The previous `.strip("'\\"")` mangled any value that
    legitimately contained a quote. NEVER raises.
    """
    try:
        _load_env_once()          # idempotent; already ran at module top
    except Exception:  # noqa: BLE001 — config must never block boot
        pass
    try:
        from dotenv import dotenv_values  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — dependency optional, as upstream
        return
    for env_path in (_PROJECT_ROOT / ".env", _PROJECT_ROOT / "backend" / ".env"):
        try:
            if not env_path.exists():
                continue
            for key, value in (dotenv_values(env_path) or {}).items():
                if not key or value is None:
                    continue
                if key in _FORCE_OVERRIDE_KEYS:
                    os.environ[key] = value      # the file wins for API keys
                else:
                    os.environ.setdefault(key, value)
        except Exception:  # noqa: BLE001 — a bad overlay is not fatal
            continue


def _check_env(key: str) -> str:
    """Check if an env var is set and return a status indicator."""
    val = os.environ.get(key, "")
    if val:
        return f"{_GREEN}ON{_RESET}"
    return f"{_DIM}OFF{_RESET}"


def _check_env_val(key: str, default: str = "") -> str:
    """Return the value of an env var or default."""
    return os.environ.get(key, default)


def _is_spawn_worker_cmdline(cmdline: list) -> bool:
    """A python interpreter running multiprocessing spawn bootstrap —
    the cmdline shape of every Oracle IPC / FS-pool worker:
    ``python -c "from multiprocessing.spawn import spawn_main; ..."
    --multiprocessing-fork``. Strict: both markers required."""
    if not cmdline:
        return False
    exe = Path(str(cmdline[0])).name.lower()
    if not exe.startswith("python"):
        return False
    has_spawn_main = any(
        "multiprocessing.spawn" in str(arg) and "spawn_main" in str(arg)
        for arg in cmdline[1:]
    )
    has_fork_flag = any(
        str(arg).startswith("--multiprocessing-fork") for arg in cmdline[1:]
    )
    return has_spawn_main and has_fork_flag


_ORPHAN_WORKER_ENV_MARKERS = ("JARVIS_ACTIVE_SESSION_LOG", "OUROBOROS_BATTLE_")


def _has_session_env_marker(environ: dict) -> bool:
    """True iff *environ* carries a battle-session fingerprint — the
    tie that scopes orphan reaping to OUR spawn workers and never
    another app's multiprocessing children."""
    for key in environ:
        if key == _ORPHAN_WORKER_ENV_MARKERS[0]:
            return True
        if key.startswith(_ORPHAN_WORKER_ENV_MARKERS[1]):
            return True
    return False


def _is_orphaned_session_spawn_worker(proc) -> bool:
    """2026-07-11 OOM RCA — the orphan class the path-tail match above
    can never see: multiprocessing spawn workers (Oracle IPC, FS pool)
    whose organism died non-gracefully. One was caught live 29h later
    at a 33.9 GB phys_footprint (7 MB rss — compressor-hidden). The
    in-worker lifeline (worker_lifeline.py) is the primary kill; this
    reaper is boot-time defense in depth for pre-fix stragglers.

    Reaps ONLY when ALL hold (fail-safe: any probe error → False):
      • spawn-bootstrap cmdline (``_is_spawn_worker_cmdline``)
      • ppid == 1 — already orphaned; live sessions' workers keep
        their organism as parent and are NEVER matched
      • environ carries a battle-session fingerprint (inherited from
        the organism) — scopes to us, not other apps' pools
    """
    try:
        if proc.ppid() != 1:
            return False
        if not _is_spawn_worker_cmdline(proc.cmdline()):
            return False
        return _has_session_env_marker(proc.environ())
    except Exception:  # noqa: BLE001 — NoSuchProcess/AccessDenied/zombie
        return False


def _read_lock_holder(
    project_root: "Optional[Path]" = None,
) -> "Optional[tuple]":
    """Read the single-flight lock and return ``(pid, age_s, alive)``,
    or ``None`` when absent/corrupt/self-owned.

    THE shared authority for "who holds the organism right now" —
    consumed by the zombie reaper (incumbent immunity), the single-flight
    preflight (conflict detection), AND the thin CLI's ensure_daemon
    (never-clean-a-live-organism's-socket + exit-75 attribution), so no
    two can disagree about legitimacy (the 2026-07-18 class: the reaper
    SIGTERM'd a healthy incumbent, then the preflight rejected the
    launcher against the dying incumbent's still-fresh lock — one ``ov``
    keystroke murdered the live soak AND locked the operator out).
    Implementation lives in ``battle_test.singleton_lock.read_lock_holder``
    (promoted 2026-07-23 so the thin client shares it without importing
    this heavy script). NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.singleton_lock import (
            read_lock_holder,
        )
        root = Path(project_root) if project_root is not None else _PROJECT_ROOT
        return read_lock_holder(root, exclude_pid=os.getpid())
    except Exception:  # noqa: BLE001
        return None


def _lock_stale_ttl_s() -> float:
    try:
        from backend.core.ouroboros.battle_test.singleton_lock import (
            lock_stale_ttl_s,
        )
        return lock_stale_ttl_s()
    except Exception:  # noqa: BLE001
        return 7200.0


def _live_incumbent_pid(
    project_root: "Optional[Path]" = None,
) -> "Optional[int]":
    """PID of a LIVE, FRESH single-flight lock holder — the legitimate
    incumbent organism — else None. NEVER raises."""
    holder = _read_lock_holder(project_root)
    if holder is None:
        return None
    pid, age_s, alive = holder
    if alive and age_s <= _lock_stale_ttl_s():
        return pid
    return None


def _reap_zombies(*, quiet: bool = False) -> "set[int]":
    """Detect and reap any lingering ouroboros_battle_test.py processes.

    A terminal disconnect or crashed session can leave the battle test
    running in the background, where it continues to burn API budget,
    compete for the intake router lock, and race this new session on
    git branches. This reaper scans for zombies at startup and kills
    them cleanly (SIGTERM, then SIGKILL after 3s) before we boot.

    Only reaps processes:
      • whose cmdline contains ``ouroboros_battle_test.py``, OR that
        match ``_is_orphaned_session_spawn_worker`` (ppid==1 spawn
        workers carrying our session env fingerprint — the 33.9 GB /
        29 h Oracle-worker orphan class, 2026-07-11 OOM RCA)
      • owned by the current UID
      • that are not this process

    Returns the set of PIDs that were targeted (and either terminated
    or SIGKILLed). The caller passes this to
    ``_cleanup_stale_router_lock`` so the lock cleanup trusts this
    authoritative knowledge instead of re-probing via ``os.kill(pid, 0)``
    — that probe is racy on macOS because the kernel can recycle a
    PID between SIGKILL and the probe, making a just-killed PID look
    "alive" again under a fresh unrelated occupant. Closes the boot
    coordination gap between reaper and lock cleanup (2026-05-03).

    ``quiet`` (ov awakening Task 1, Mandate 1) withholds the stdout
    banner ceremony only — the scan/terminate/kill side effects below
    are functional and run unconditionally in either mode. Callers in
    COCKPIT presentation mode pass ``quiet=True`` so lingering
    processes are still reaped without flooding the clean boot.
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return set()  # Silently skip; psutil is in requirements.txt but not hard-required

    my_pid = os.getpid()
    my_ppid = os.getppid() if hasattr(os, "getppid") else None
    my_uid = os.getuid() if hasattr(os, "getuid") else None

    def _is_battle_test_proc(cmdline: list) -> bool:
        """Strict match: a python interpreter running our script path.

        We require:
          • the first argv is a python-family executable (python/python3/pythonX),
          • and some argv ends with ``ouroboros_battle_test.py`` as a path segment.

        Substring matching is too loose — a shell or editor whose buffer
        contains the literal filename would otherwise be reaped.
        """
        if not cmdline:
            return False
        exe = Path(str(cmdline[0])).name.lower()
        if not exe.startswith("python"):
            return False
        for arg in cmdline[1:]:
            # Match on trailing path segment so `/abs/path/ouroboros_battle_test.py`
            # and `scripts/ouroboros_battle_test.py` both qualify, but `-c "... ouroboros_battle_test.py ..."`
            # embedded in a code string does NOT (that lives in a single argv together
            # with surrounding code, not as a clean path).
            tail = Path(str(arg)).name
            if tail == "ouroboros_battle_test.py":
                return True
        return False

    # Incumbent immunity (2026-07-18): a battle-test holding a LIVE,
    # FRESH single-flight lock is the legitimate organism, not a zombie.
    # Reaping it here and then bouncing off its lock in the preflight
    # was the one-keystroke murder/lockout class. True zombies (dead-PID
    # locks, lockless strays, wedged-TTL holders) still die below.
    incumbent = _live_incumbent_pid()

    victims: list = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "uids", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid == my_pid or pid == my_ppid:
                continue
            if incumbent is not None and pid == incumbent:
                if not quiet:
                    print(
                        f"  {_DIM}[reaper] incumbent organism excluded "
                        f"(PID {pid} holds a live single-flight lock)"
                        f"{_RESET}"
                    )
                continue
            cmdline = proc.info.get("cmdline") or []
            # Two victim classes: (1) lingering battle-test mains by
            # script path; (2) orphaned session spawn workers — cheap
            # cmdline+ppid prefilter first, environ() probe last.
            if not _is_battle_test_proc(cmdline) and not (
                _is_spawn_worker_cmdline(cmdline)
                and _is_orphaned_session_spawn_worker(proc)
            ):
                continue
            if my_uid is not None:
                uids = proc.info.get("uids")
                if uids is not None and getattr(uids, "real", my_uid) != my_uid:
                    continue
            victims.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not victims:
        return set()

    reaped_pids: "set[int]" = {p.pid for p in victims}

    # Mandate 1: gate the banner ceremony at the source, not the
    # scan/terminate/kill side effects below (those always run).
    _emit = (lambda *_a, **_k: None) if quiet else print

    _emit(f"\n{_BOLD}{_YELLOW}  Zombie Reaper{_RESET}")
    _emit(f"{_DIM}  {'─' * 52}{_RESET}")
    for p in victims:
        try:
            age_s = time.time() - p.create_time()
            m, s = int(age_s) // 60, int(age_s) % 60
            _emit(
                f"  {_YELLOW}→{_RESET} reaping PID {p.pid} "
                f"{_DIM}(age {m}m{s:02d}s){_RESET}"
            )
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            _emit(f"  {_DIM}  skipped PID {p.pid}: {type(exc).__name__}{_RESET}")

    # Wait up to 3s for graceful shutdown, then SIGKILL holdouts.
    try:
        alive = psutil.wait_procs(victims, timeout=3.0)[1]
    except Exception:
        alive = victims
    for p in alive:
        try:
            p.kill()
            _emit(f"  {_RED}→{_RESET} SIGKILL PID {p.pid} {_DIM}(ignored SIGTERM){_RESET}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    count = len(victims)
    plural = "s" if count != 1 else ""
    _emit(f"  {_GREEN}✓ reaped {count} zombie{plural}{_RESET}\n")
    return reaped_pids


def _cleanup_stale_router_lock(
    reaped_pids: "set[int] | None" = None,
) -> None:
    """Remove a stale ``.jarvis/intake_router.lock`` left by a crashed session.

    The lock file carries ``{"pid": ..., "ts": ...}`` metadata.

    If ``reaped_pids`` is supplied and the lock's PID is in that set,
    the lock is removed unconditionally — the reaper just killed that
    PID, so its claim is definitively void. This bypasses the
    ``os.kill(pid, 0)`` existence probe, which is racy on macOS:
    between SIGKILL and the probe, the kernel can recycle the PID to
    a fresh unrelated process (zsh, terminal subprocess, etc.),
    making the just-killed PID appear "alive" and letting the stale
    lock survive into the single-flight check (which then rejects
    the new launch). Trust the reaper's authoritative knowledge.

    Otherwise (or for locks held by PIDs the reaper didn't touch),
    fall back to the existence-probe path — the intake router would
    also clean a dead-PID lock at startup, but doing it here first
    avoids a noisy retry and keeps the reaper banner coherent.
    """
    lock_path = _PROJECT_ROOT / ".jarvis" / "intake_router.lock"
    if not lock_path.exists():
        return
    try:
        import json as _json
        data = _json.loads(lock_path.read_text() or "{}")
    except (ValueError, OSError):
        try:
            lock_path.unlink()
            print(f"  {_DIM}  cleaned corrupt intake_router.lock{_RESET}")
        except OSError:
            pass
        return
    pid = int(data.get("pid", 0) or 0)
    if pid <= 0:
        return
    if reaped_pids is not None and pid in reaped_pids:
        try:
            lock_path.unlink()
            print(
                f"  {_DIM}  cleaned stale intake_router.lock "
                f"(reaped PID {pid}){_RESET}"
            )
        except OSError:
            pass
        return
    try:
        os.kill(pid, 0)  # existence probe — no signal delivered
        # PID is alive; leave the lock alone (router will error loudly if it's us).
    except ProcessLookupError:
        try:
            lock_path.unlink()
            print(
                f"  {_DIM}  cleaned stale intake_router.lock "
                f"(dead PID {pid}){_RESET}"
            )
        except OSError:
            pass
    except PermissionError:
        pass  # Different user — leave it alone


def _reap_stale_jarvis_locks(
    jarvis_dir: "Path",
    max_age_s: float = 86400.0,
    now: "float | None" = None,
) -> int:
    """Slice 48 — purge stale ``.jarvis/**/*.lock`` debris at boot.

    flock auto-releases on process death, so an old ``.lock`` *file* left by a
    crashed session is inert crumbs — but it accumulates and pollutes the
    workspace (v43 flagged a ~53h-old ``metrics_history.jsonl.lock``). Any
    ``.lock`` whose mtime is older than ``max_age_s`` (default 24h) is removed.

    ``intake_router.lock`` is intentionally skipped — it has a dedicated
    PID-aware handler (:func:`_cleanup_stale_router_lock`) that must own the
    single-flight decision. Returns the number of lock files removed; never
    raises (best-effort hygiene).
    """
    if not jarvis_dir.exists():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_s
    reaped = 0
    try:
        candidates = list(jarvis_dir.rglob("*.lock"))
    except OSError:
        return 0
    for lock in candidates:
        if lock.name == "intake_router.lock":
            continue  # PID-aware handler owns this one
        try:
            if lock.stat().st_mtime >= cutoff:
                continue
            lock.unlink()
            reaped += 1
        except OSError:
            continue  # raced/removed/permission — leave it
    if reaped:
        print(
            f"  {_DIM}  reaped {reaped} stale .jarvis lock file(s) "
            f"(>{max_age_s / 3600:.0f}h old){_RESET}"
        )
    return reaped


def _reap_stale_cross_process_jsonl_locks(
    jarvis_dir: "Path",
    now: "float | None" = None,
    *,
    quiet: bool = False,
) -> int:
    """ov cockpit silence Slice 2 Task 5 (F3) — sweep stale
    ``.jarvis/**/*.jsonl.lock`` files at boot, ahead of (and at a
    tighter threshold than) the coarser 24h ``_reap_stale_jarvis_locks``
    debris sweep.

    A live battle-test session (bt-2026-07-08-013911) logged::

        WARNING [CrossProcessJSONL] stale_lock_detected
            path=.jarvis/coherence_window.jsonl.lock age_s=29261 threshold_s=300

    An 8h-stale lock from a dead session survives boot because
    ``_reap_stale_jarvis_locks``'s debris sweep uses a 24h default —
    far looser than the 300s (5 min) threshold the runtime module
    itself already treats as stale
    (:func:`cross_process_jsonl.stale_lock_age_s`, env
    ``JARVIS_STALE_LOCK_AGE_S``). This function reuses that SAME
    threshold rather than inventing a new one (Mandate 3, DRY).

    Liveness detection: these lock files carry no PID payload (unlike
    ``intake_router.lock``'s ``{"pid": ..., "ts": ...}``) — the file
    is a pure ``fcntl.flock`` target created by
    ``cross_process_jsonl._acquire_cross_process_lock``. The correct
    liveness proxy is therefore the SAME primitive the module uses
    internally: a non-blocking flock attempt via
    :func:`cross_process_jsonl.flock_critical_section`. If a live
    process currently holds the lock, the attempt fails immediately
    and the lock is left untouched — NEVER reaped, regardless of age
    (binding constraint: never touch a lock owned by a live PID).
    Only when BOTH conditions hold — the lock is uncontended right
    now AND its mtime exceeds the threshold — is it removed, deleting
    the file while still holding our own flock (the same TOCTOU-safe
    pattern ``_cleanup_stale_router_lock`` already uses: unlink on an
    open fd is safe on POSIX, and the kernel releases the flock when
    the fd closes on ``__exit__``).

    Fail-soft: any per-file or import error is swallowed and the file
    is left alone; the function itself NEVER raises. ``quiet``
    withholds only the summary print (Mandate 1 presentation gate) —
    the reap side effects always run. Returns the count reaped.
    """
    if not jarvis_dir.exists():
        return 0
    try:
        from backend.core.ouroboros.governance.cross_process_jsonl import (
            flock_critical_section,
            stale_lock_age_s,
        )
    except Exception:
        return 0  # substrate unavailable — nothing to reuse, skip safely

    cutoff_age = stale_lock_age_s()
    now_ts = time.time() if now is None else now
    reaped = 0
    try:
        candidates = list(jarvis_dir.rglob("*.jsonl.lock"))
    except OSError:
        return 0
    for lock in candidates:
        try:
            age = now_ts - lock.stat().st_mtime
        except OSError:
            continue  # raced/removed — leave it
        if age <= cutoff_age:
            continue  # fresh enough — a live session may be about to use it
        data_path = lock.with_suffix("")  # strip only the trailing ".lock"
        try:
            with flock_critical_section(data_path, timeout_s=0.05) as acquired:
                if not acquired:
                    continue  # live holder — NEVER touch (binding constraint)
                try:
                    lock.unlink()
                    reaped += 1
                except OSError:
                    continue  # raced/removed — leave it
        except Exception:
            continue  # fail-soft — reaper errors never block boot
    if reaped and not quiet:
        print(
            f"  {_DIM}  reaped {reaped} stale CrossProcessJSONL lock file(s) "
            f"(>{cutoff_age:.0f}s idle, threshold reused from "
            f"JARVIS_STALE_LOCK_AGE_S){_RESET}"
        )
    return reaped


def _single_flight_preflight(*, quiet: bool = False) -> None:
    """Harness Epic Slice 2 — single-flight launcher enforcement.

    Rejects concurrent battle-test runs at the process level. Two checks:

    1. **pgrep canonical**: ``pgrep -f "^python3? scripts/ouroboros_battle_test\\.py"``
       must return at most 1 PID (this process).

       The leading ``^`` anchor is load-bearing 2026-05-13 (operator
       runbook v12): when the harness is launched via a wrapper like
       ``caffeinate -dimsu python3 scripts/ouroboros_battle_test.py``,
       the wrapper's cmdline CONTAINS the python path as later argv
       elements, so an unanchored ``-f`` match would treat the
       wrapper as a "concurrent battle-test" and reject the launch.
       ``^`` forces the match against argv[0]+argv[1] only, so the
       wrapper (whose cmdline starts with "caffeinate") is correctly
       excluded.  This is structural — we leverage the existing
       wrapper-friendly pattern (``caffeinate`` is the canonical
       macOS App-Nap escape) rather than fighting it.

    2. **Lock-with-live-PID-and-non-stale-TTL**: if ``.jarvis/intake_router.lock``
       exists AND its PID is alive AND its ``ts`` is newer than the
       stale TTL (``JARVIS_INTAKE_LOCK_STALE_TTL_S``, default 7200s),
       another battle-test is genuinely running. Note: the zombie reaper
       and ``_cleanup_stale_router_lock`` already removed dead-PID and
       wedged-TTL locks before this preflight runs — so anything still
       present here is a true conflict.

    On conflict: print actionable info + ``sys.exit(75)``. Exit code 75
    (``EX_TEMPFAIL`` from BSD sysexits.h) signals "try again later" to
    wrappers — distinct from generic error code 1.

    Disable via ``JARVIS_BATTLE_SINGLE_FLIGHT_ENABLED=false`` (operator
    escape hatch — useful for diagnostics or recovery).

    ov awakening Task 1 (Mandate 1): this guard is FUNCTIONAL (prevents two
    sessions competing for budget/locks/branches) and runs in BOTH
    presentation modes. ``quiet`` withholds only the happy-path diagnostic
    chatter (the wedged-lock adoption note below, where the launch
    proceeds). The conflict-path REJECTED block is ERROR-class telemetry
    and prints unconditionally — no mode can suppress it.
    """
    import subprocess as _subprocess

    self_pid = os.getpid()
    violators: list = []

    # (1) pgrep canonical probe — anchored to avoid matching wrappers
    # whose cmdline contains the python path as a later argv element
    # (notably ``caffeinate -dimsu python3 scripts/ouroboros_battle_test.py``,
    # which is the canonical macOS App-Nap escape for foreground runs).
    try:
        result = _subprocess.run(
            ["pgrep", "-f", r"^python3? scripts/ouroboros_battle_test\.py"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if result.returncode == 0:
            pgrep_pids = [
                int(line.strip())
                for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
            others = [pid for pid in pgrep_pids if pid != self_pid]
            for pid in others:
                violators.append(("pgrep", pid))
    except (FileNotFoundError, _subprocess.TimeoutExpired, OSError):
        # pgrep unavailable / timed out — fall through; rely on lock check
        pass

    # (2) Lock-with-live-PID-and-non-stale-TTL check.
    # Anchor at the ``__file__``-derived repo root (cwd-independent) -- NOT
    # ``cwd``. On the Linux node ``cwd != /opt/trinity/jarvis`` so a cwd-rooted
    # lock path pointed at the wrong ``.jarvis`` dir (the run-#14 class of bug:
    # cwd-relative roots in the soak pipeline). ``_PROJECT_ROOT`` is the same
    # root every other lock-path site here uses (e.g. the stale-lock sweep).
    # DRY (2026-07-18): the SAME _read_lock_holder authority the zombie
    # reaper consults for incumbent immunity — the two surfaces can never
    # again disagree about who legitimately owns the organism.
    holder = _read_lock_holder(_PROJECT_ROOT)
    if holder is not None:
        holder_pid, age_s, alive = holder
        _stale_ttl = _lock_stale_ttl_s()
        if alive and age_s <= _stale_ttl:
            violators.append(("lock", holder_pid))
        elif alive and age_s > _stale_ttl:
            # Happy-path chatter (launch proceeds) — the only
            # print this guard gates on ``quiet``.
            if not quiet:
                print(
                    f"  {_DIM}[single-flight] adopting wedged lock "
                    f"(PID={holder_pid} alive, age={age_s:.0f}s > "
                    f"TTL={_stale_ttl:.0f}s — Py_FinalizeEx-class "
                    f"zombie pattern){_RESET}"
                )

    if violators:
        # COCKPIT collision surface (2026-07-18, design language §3):
        # an operator typing ``ov`` while the organism is already awake
        # is not an error condition — it is a status moment. Render the
        # incumbent + live session digest + the paths forward, instead
        # of a raw rejection wall. SOAK/headless keeps the terse
        # diagnostic (wrappers parse exit 75).
        _is_cockpit = False
        try:
            from backend.core.ouroboros.ui.presentation_mode import (
                is_cockpit as _is_cockpit_fn,
            )
            _is_cockpit = _is_cockpit_fn()
        except Exception:
            _is_cockpit = False
        if _is_cockpit:
            # The collision surface IS a presentation moment — hand the
            # TTY over cleanly (no dead-man dump; the buffered boot
            # chatter stays in boot.log where it belongs).
            try:
                from backend.core.ouroboros.ui.boot_mux import (
                    release_boot_mux,
                )
                release_boot_mux()
            except Exception:  # noqa: BLE001
                pass
            # THE EMBLEM ALWAYS GREETS `ov` (operator law 2026-07-18):
            # the static crest renders above the already-awake card too
            # — animation is the birth; the mark is the identity.
            try:
                from backend.core.ouroboros.ui.crest import (
                    print_static_crest,
                )
                from backend.core.ouroboros.ui.theme import build_console
                print_static_crest(build_console())
            except Exception:  # noqa: BLE001 — emblem never blocks the card
                pass
            _pid = violators[0][1]
            _held = ""
            _h = _read_lock_holder(_PROJECT_ROOT)
            if _h is not None and _h[1] not in (None, float("inf")):
                _held = f" · up ~{max(1, int(_h[1] / 60))}m"
            print()
            print(
                f"  {_BOLD}⏺ the organism is already awake{_RESET}"
                f"{_DIM} — pid {_pid}{_held}{_RESET}"
            )
            try:
                from backend.core.ouroboros.cli.ov import status_digest
                for _ln in status_digest().splitlines()[:3]:
                    print(f"  {_DIM}⎿ {_ln}{_RESET}")
            except Exception:
                pass
            print(
                f"  {_DIM}⎿ attach: ov attach (live view + input) · "
                f"stop: kill {_pid}{_RESET}"
            )
            print(
                f"  {_DIM}⎿ force a second organism: "
                f"JARVIS_BATTLE_SINGLE_FLIGHT_ENABLED=false (budget/locks "
                f"will contend){_RESET}"
            )
            print()
            sys.exit(75)
        print()
        print(
            f"  {_RED}[single-flight] REJECTED — concurrent battle-test detected{_RESET}"
        )
        for source, pid in violators:
            print(f"  {_DIM}  • {source}: PID {pid}{_RESET}")
        print(
            f"  {_DIM}  exit code 75 (EX_TEMPFAIL) — try again after the other run completes{_RESET}"
        )
        print(
            f"  {_DIM}  override: JARVIS_BATTLE_SINGLE_FLIGHT_ENABLED=false{_RESET}"
        )
        print()
        sys.exit(75)


def _local_generation_lane() -> "Optional[str]":
    """The local J-Prime lane's endpoint when it can actually serve, else None.

    Returns the endpoint only when the lane is BOTH enabled and REACHABLE. A
    flag alone is not evidence: ``JARVIS_LOCAL_PRIME_ENABLED=true`` against a
    stopped Ollama would let the preflight pass and the soak boot into a loop
    with no generation lane at all -- trading a loud death at second zero for a
    silent one an hour in, which is strictly worse.

    Reads the endpoint from ``LocalConfig.from_env()`` rather than a literal, so
    this and the inference director can never disagree about where local lives.
    Probes ``/api/tags`` -- the same readiness surface the model resolver and the
    driver gate already use. Bounded and fail-soft: any error, timeout or
    non-200 means "no lane", which routes back into the fatal path. NEVER
    raises."""
    try:
        from backend.core.ouroboros.governance.local_inference_director import (
            LocalConfig, local_prime_enabled,
        )
        if not local_prime_enabled():
            return None
        base = (LocalConfig.from_env().base_url or "").strip()
        if not base:
            return None
        import json as _json
        import urllib.request as _url
        try:
            timeout = float(os.environ.get("JARVIS_PREFLIGHT_LOCAL_PROBE_S", "4"))
        except (TypeError, ValueError):
            timeout = 4.0
        with _url.urlopen(base.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            payload = _json.loads(resp.read().decode("utf-8", "replace"))
        # A reachable engine serving ZERO models is not a lane either.
        if not (payload or {}).get("models"):
            return None
        return base
    except Exception:  # noqa: BLE001 -- unprovable lane == no lane
        return None


def _check_api_keys_or_die() -> None:
    """FATAL preflight: no usable GENERATION LANE -> die loudly. Deliberately
    OUTSIDE the presentation gate (Mandate 1): no mode can suppress this.

    The assertion is "a lane exists", not "a cloud key exists". Those were the
    same statement when every provider was remote; they stopped being the same
    once J-Prime could be served locally. A host running a local 32B has a
    lane -- refusing to boot it because no cloud key is exported would make the
    zero-marginal-cost configuration the one configuration that cannot run,
    which is precisely backwards.

    The fatality is unchanged. Mandate 1 is about no MODE suppressing the
    check; it was never about requiring a cloud vendor specifically."""
    if os.environ.get("DOUBLEWORD_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return
    local = _local_generation_lane()
    if local:
        print(f"  {_BOLD}Generation lane: LOCAL J-Prime at {local}{_RESET}")
        print("  (no cloud keys exported — running at zero marginal cost)")
        return
    print(f"\n  {_RED}{_BOLD}ERROR: No usable generation lane.{_RESET}")
    print(f"  {_RED}Export DOUBLEWORD_API_KEY or ANTHROPIC_API_KEY,{_RESET}")
    print(f"  {_RED}or enable the local lane: JARVIS_LOCAL_PRIME_ENABLED=true{_RESET}")
    print(f"  {_RED}with a reachable engine at JARVIS_LOCAL_MODEL_BASE_URL{_RESET}")
    print(f"  {_RED}(currently unreachable, or serving no models).{_RESET}\n")
    sys.exit(1)


#: POSIX ``sysexits.h`` EX_CONFIG. The repo already speaks this dialect --
#: `ensure_daemon` reads 75 (EX_TEMPFAIL) as a single-flight refusal -- so a
#: configuration fault gets its own code rather than joining the generic 1.
#: A client can then say WHY the organism declined instead of "did not come
#: up", which is the difference between a fix and a debugging session.
EXIT_MODEL_PIN_UNAVAILABLE = 78


def _validate_model_pin_or_die() -> None:
    """FATAL preflight: the pinned model must be one this node serves.

    ## Why this is fatal rather than a warning

    The selector is fail-soft by contract and substitutes "largest by
    size" when a pin is not served. On the hot path that is right -- a
    grader must never stop a running loop. At BOOT it is wrong: every
    subsequent generation answers from a model the operator did not
    choose, a fine-tune A/B silently compares the base against itself,
    and the only trace is one WARNING line in a log nobody tails.

    ## What counts as evidence

    Only a registry that ANSWERED and does not contain the pin. An
    unreachable or empty registry is "we could not ask", and
    `_check_api_keys_or_die` already dies loudly on exactly that -- so a
    second opinion here would turn a transient blip into a self-kill
    while adding nothing. This gate therefore runs AFTER that one and
    treats silence as not-proven rather than as absence.

    No pin means no promise to keep, and auto-selection stands.
    """
    try:
        from backend.core.ouroboros.governance.candidate_generator import (
            ModelPinUnavailable, resolve_active_model, set_active_model_tag,
        )
    except Exception as exc:  # noqa: BLE001 — never block boot on the gate
        print(f"  (model pin gate unavailable: {type(exc).__name__}: {exc})")
        return

    pin = (os.environ.get("JARVIS_LOCAL_MODEL_NAME") or "").strip()
    tags = _local_registry_tags()
    if tags is None:
        # Not proven either way. Say so plainly rather than implying the
        # pin was honoured.
        if pin:
            print(f"  model pin: {pin} (registry unreachable — unverified)")
        return
    try:
        resolved = resolve_active_model(tags, pin=pin)
    except ModelPinUnavailable as exc:
        served = "\n".join(f"      - {s}" for s in exc.served) or "      (none)"
        print(f"\n  {_RED}{_BOLD}FATAL: pinned model is not served.{_RESET}")
        print(f"  {_RED}component : model_selection{_RESET}")
        print(f"  {_RED}pinned    : {exc.pin}{_RESET}")
        print(f"  {_RED}source    : JARVIS_LOCAL_MODEL_NAME "
              f"(environment, else .env){_RESET}")
        print(f"  {_RED}served    :{_RESET}\n{_RED}{served}{_RESET}")
        print(f"  {_RED}action    : refusing to start. Falling back to another "
              f"model would answer{_RESET}")
        print(f"  {_RED}            every request from a model you did not "
              f"choose, silently.{_RESET}")
        print(f"  {_RED}fix       : `ollama pull {exc.pin}`, or correct "
              f"JARVIS_LOCAL_MODEL_NAME.{_RESET}\n")
        sys.exit(EXIT_MODEL_PIN_UNAVAILABLE)
    set_active_model_tag(resolved)
    if resolved:
        origin = "pinned" if pin else "auto-selected (no pin)"
        print(f"  Model: {_BOLD}{resolved}{_RESET} ({origin})")


def _local_registry_tags() -> "Optional[dict]":
    """The engine's ``/api/tags`` payload, or None when it cannot be read.

    None is "could not ask", never "serves nothing" — the caller's whole
    correctness rests on that distinction. Reuses `_local_generation_lane`
    for the endpoint so this and the lane preflight can never disagree
    about where local lives. NEVER raises.
    """
    try:
        base = _local_generation_lane()
        if not base:
            return None
        import json as _json
        import urllib.request as _url
        try:
            timeout = float(os.environ.get("JARVIS_PREFLIGHT_LOCAL_PROBE_S", "4"))
        except (TypeError, ValueError):
            timeout = 4.0
        with _url.urlopen(base.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return _json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — unreadable registry == not proven
        return None


def _print_preflight() -> None:
    """Print a preflight checklist showing what's enabled.

    Gap #7 Slice 1 — when ``JARVIS_PRESENTATION_RESTRAINT_ENABLED``
    is on, the verbose multi-line checklist is suppressed at boot.
    Operators retrieve the same content on demand via the ``/preflight``
    REPL verb.

    ov awakening Task 1 — this is now pure presentation. The API-key
    fail-fast lives in ``_check_api_keys_or_die()`` and is called
    unconditionally by ``main()`` before this function runs, so this
    function itself never exits the process.
    """
    # Restraint mode: skip the verbose render.
    try:
        from backend.core.ouroboros.battle_test.presentation_restraint import (
            is_restraint_enabled, suppress_diagnostic_logs,
        )
        _restraint_on = is_restraint_enabled()
    except Exception:
        _restraint_on = False

    if _restraint_on:
        # Suppress the shutdown_diagnostics INFO leak too — same boot
        # noise reduction.
        try:
            suppress_diagnostic_logs()
        except Exception:
            pass
        return

    print(f"\n{_BOLD}{_CYAN}  Preflight Checklist{_RESET}")
    print(f"{_DIM}  {'─' * 52}{_RESET}")

    checks = [
        ("Provider: DoubleWord 397B", "DOUBLEWORD_API_KEY",
         "$0.10/$0.40/M (Tier 0 PRIMARY)"),
        ("Provider: Claude Sonnet", "ANTHROPIC_API_KEY",
         "$3/$15/M (Tier 1 FALLBACK)"),
        ("Venom: Tool Loop", "JARVIS_GOVERNED_TOOL_USE_ENABLED",
         "read_file, search_code, get_callers, list_symbols"),
        ("Venom: Bash (100+ cmds)", "JARVIS_BASH_TOOL_ENABLED",
         "python, git, docker, curl, terraform..."),
        ("Venom: Web Search", "JARVIS_WEB_TOOL_ENABLED",
         "DuckDuckGo / Brave / Google CSE"),
        ("Venom: Run Tests", "JARVIS_TOOL_RUN_TESTS_ALLOWED",
         "pytest in sandbox during generation"),
        ("L2 Repair Engine", "JARVIS_L2_ENABLED",
         f"max {_check_env_val('JARVIS_L2_MAX_ITERS', '5')} iters, "
         f"{_check_env_val('JARVIS_L2_TIMEBOX_S', '120')}s timebox"),
        ("Trinity Consciousness", "JARVIS_CONSCIOUSNESS_ENABLED",
         "Memory + Prophecy + Health"),
    ]

    all_good = True
    for label, env_key, detail in checks:
        status = _check_env(env_key)
        is_on = bool(os.environ.get(env_key, ""))
        if not is_on and env_key in ("DOUBLEWORD_API_KEY", "ANTHROPIC_API_KEY"):
            all_good = False
        indicator = f"  [{status}]"
        print(f"{indicator} {label:<30s} {_DIM}{detail}{_RESET}")

    rounds = _check_env_val("JARVIS_GOVERNED_TOOL_MAX_ROUNDS", "10")
    print(f"\n{_DIM}  Tool rounds: {rounds} (deadline-based, safety ceiling){_RESET}")

    # Single-provider warnings (non-fatal; the "no keys at all" case is a
    # hard error and lives in _check_api_keys_or_die, called before this
    # function ever runs).
    has_dw = bool(os.environ.get("DOUBLEWORD_API_KEY"))
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_claude:
        print(f"\n  {_YELLOW}WARNING: ANTHROPIC_API_KEY not set — no Claude fallback.{_RESET}")
    if not has_dw:
        print(f"\n  {_YELLOW}WARNING: DOUBLEWORD_API_KEY not set — Claude only (expensive).{_RESET}")

    print()


def _run_gated_boot_banners(
    mode: PresentationMode, *, reap_enabled: bool,
) -> None:
    """Nominal boot banners, gated AT THE SOURCE (Mandate 1). COCKPIT
    withholds; detail stays reachable via the /preflight + /organism verbs.
    Fatal telemetry never routes through here.

    Only pure ceremony belongs in this helper. FUNCTIONAL boot steps —
    _check_api_keys_or_die, _single_flight_preflight (concurrent-launch
    guard), and the zombie reap in main() — run unconditionally in both
    modes at their own call sites; they must never be added back here.
    """
    if mode is PresentationMode.COCKPIT:
        return
    if reap_enabled:
        _reap_zombies()
    _print_preflight()


def _resolve_boot_log_level(mode: PresentationMode, *, verbose: bool = False) -> int:
    """COCKPIT quiets the INFO flood to WARNING. -v always wins. ERROR and
    CRITICAL pass at every level -- the gate lowers verbosity, filters nothing."""
    if verbose:
        return logging.DEBUG
    if mode is PresentationMode.COCKPIT:
        return logging.WARNING
    return logging.INFO


def _print_battle_test_defaults_banner(
    caps_result: Any, mode: PresentationMode,
) -> None:
    """``[BattleTestDefaults]`` boot line -- ov cockpit silence (Slice 2).

    The success line is pure ceremony, gated at the source: COCKPIT
    withholds, SOAK prints (unchanged). The WARNING/failure line stays
    UNCONDITIONAL in both modes (Mandate 1: fatal-adjacent telemetry
    never routes through a presentation gate). Extracted to a free
    function so it's spy-testable without booting the full ``main()``.
    """
    if caps_result.ok:
        if mode is not PresentationMode.COCKPIT:
            print(
                f"[BattleTestDefaults] session_cap_source={caps_result.session_cap_source} "
                f"hourly_burn_cap_source={caps_result.hourly_burn_cap_source} "
                f"({caps_result.detail})"
            )
    else:
        print(f"[BattleTestDefaults] WARNING: {caps_result.detail} — boot continues", file=sys.stderr)


def _print_ledger_hygiene_banner(
    hygiene_result: Any, mode: PresentationMode,
) -> None:
    """``[LedgerHygiene]`` boot line -- ov cockpit silence (Slice 2).

    The skipped/ok lines are pure ceremony, gated at the source:
    COCKPIT withholds, SOAK prints (unchanged). The WARNING/failure
    line stays UNCONDITIONAL in both modes (Mandate 1). Extracted to a
    free function so it's spy-testable without booting the full
    ``main()``.
    """
    if hygiene_result.skipped:
        if mode is not PresentationMode.COCKPIT:
            print(f"[LedgerHygiene] skipped (operator opt-out): {hygiene_result.detail}")
    elif hygiene_result.ok:
        if mode is not PresentationMode.COCKPIT:
            _bits = []
            if hygiene_result.rotated_path:
                _bits.append(f"rotated→{hygiene_result.rotated_path}")
            if hygiene_result.lock_removed:
                _bits.append("lock_removed")
            if hygiene_result.pruned_count:
                _bits.append(f"pruned={hygiene_result.pruned_count}")
            print(f"[LedgerHygiene] {' '.join(_bits) or 'no-op (fresh WAL)'}")
    else:
        print(f"[LedgerHygiene] WARNING: {hygiene_result.detail} — boot continues", file=sys.stderr)


def _print_aegis_daemon_ready(result: Any, mode: PresentationMode) -> None:
    """``[Aegis] daemon ready`` boot line -- ov cockpit silence (Slice 2 Task 2).

    Pure ceremony, gated at the source: COCKPIT withholds, SOAK prints
    (unchanged). The preflight-FAILURE print (a few lines above this
    call site in ``main()``) stays UNCONDITIONAL in both modes
    (Mandate 1 -- fatal-adjacent telemetry never routes through this
    gate); only the READY-path success line is ceremony. Extracted to a
    free function so it's spy-testable without booting the full
    ``main()``.
    """
    if mode is PresentationMode.COCKPIT:
        return
    print(
        f"[Aegis] daemon ready at {result.aegis_url} "
        f"(pid={result.subprocess_pid})"
    )


def _render_multi_op_and_exit(arg: str, *, color: bool) -> None:
    """Phase 8 Slice 3 — render a multi-op timeline view via the
    observability/multi_op_renderer module and print to stdout.

    NEVER boots the battle-test stack. Lazy-imports the renderer so
    this CLI path doesn't pay the substrate import cost when unused.
    """
    try:
        from backend.core.ouroboros.governance.observability.multi_op_renderer import (  # noqa: E501
            dispatch_cli_argument,
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  {_RED}multi-op renderer unavailable: {exc}{_RESET}")
        return
    text = dispatch_cli_argument(arg, color=color)
    print(text)


def _replay_session(session_ref: str) -> None:
    """Replay a previous battle test session timeline.

    Parameters
    ----------
    session_ref:
        Either a session ID (e.g. ``bt-2026-04-08-143022``), a direct path
        to ``summary.json``, or ``"list"`` to show available sessions.
    """
    import json

    sessions_root = _PROJECT_ROOT / ".ouroboros" / "sessions"

    # ── List mode ──
    if session_ref.lower() == "list":
        if not sessions_root.exists():
            print(f"  {_RED}No sessions found in {sessions_root}{_RESET}")
            return
        found = sorted(sessions_root.iterdir(), reverse=True)
        if not found:
            print(f"  {_RED}No sessions found.{_RESET}")
            return
        print(f"\n{_BOLD}{_CYAN}  Available Sessions{_RESET}")
        print(f"{_DIM}  {'─' * 52}{_RESET}")
        for d in found[:20]:
            summary_path = d / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text())
                    ops = len(data.get("operations", []))
                    cost = data.get("cost_total", 0.0)
                    dur = data.get("duration_s", 0.0)
                    m, s = int(dur) // 60, int(dur) % 60
                    stop = data.get("stop_reason", "?")
                    print(
                        f"  {_CYAN}{d.name}{_RESET}  "
                        f"{ops} ops  ${cost:.3f}  {m}m{s:02d}s  "
                        f"{_DIM}{stop}{_RESET}"
                    )
                except Exception:
                    print(f"  {_CYAN}{d.name}{_RESET}  {_DIM}(corrupt summary){_RESET}")
            else:
                print(f"  {_DIM}{d.name}  (no summary.json){_RESET}")
        print()
        return

    # ── Resolve summary.json path ──
    summary_path: Path
    if session_ref.endswith(".json") and Path(session_ref).exists():
        summary_path = Path(session_ref)
    else:
        # Try as session ID
        candidate = sessions_root / session_ref / "summary.json"
        if candidate.exists():
            summary_path = candidate
        else:
            # Try partial match
            matches = sorted(sessions_root.glob(f"*{session_ref}*"))
            if matches:
                summary_path = matches[-1] / "summary.json"
            else:
                print(f"  {_RED}Session not found: {session_ref}{_RESET}")
                print(f"  {_DIM}Use --replay list to see available sessions{_RESET}")
                return

    if not summary_path.exists():
        print(f"  {_RED}Summary not found: {summary_path}{_RESET}")
        return

    data = json.loads(summary_path.read_text())
    operations = data.get("operations", [])
    session_id = data.get("session_id", "unknown")
    duration_s = data.get("duration_s", 0.0)
    cost_total = data.get("cost_total", 0.0)
    stop_reason = data.get("stop_reason", "unknown")
    stats = data.get("stats", {})

    m, s = int(duration_s) // 60, int(duration_s) % 60

    # ── Header ──
    print(f"\n{'═' * 64}")
    print(f"  {_BOLD}{_CYAN}SESSION REPLAY{_RESET}  {session_id}")
    print(f"  {_DIM}Duration: {m}m {s:02d}s │ Cost: ${cost_total:.3f} │ Stop: {stop_reason}{_RESET}")
    print(f"  {_DIM}Attempted: {stats.get('attempted', '?')} │ "
          f"Completed: {stats.get('completed', '?')} │ "
          f"Failed: {stats.get('failed', '?')} │ "
          f"Queued: {stats.get('queued', '?')}{_RESET}")
    print(f"{'═' * 64}\n")

    if not operations:
        print(f"  {_DIM}No operations recorded in this session.{_RESET}\n")
        return

    # ── Sort by recorded_at for chronological timeline ──
    operations.sort(key=lambda o: o.get("recorded_at", 0.0))

    # Find session start time (earliest recorded_at - elapsed)
    first_ts = operations[0].get("recorded_at", 0.0)
    first_elapsed = operations[0].get("elapsed_s", 0.0)
    session_start = first_ts - first_elapsed if first_ts else 0.0

    # ── Timeline ──
    for i, op in enumerate(operations, 1):
        op_id = op.get("op_id", "?")
        short_id = op_id[:12] if len(op_id) > 12 else op_id
        status = op.get("status", "?")
        sensor = op.get("sensor", "?")
        provider = op.get("provider", "?")
        cost = op.get("cost_usd", 0.0)
        elapsed = op.get("elapsed_s", 0.0)
        technique = op.get("technique", "")
        tool_calls = op.get("tool_calls", 0)
        files_changed = op.get("files_changed", 0)
        recorded_at = op.get("recorded_at", 0.0)

        # Time offset from session start
        offset_s = recorded_at - session_start if session_start else 0.0
        om, os_ = int(offset_s) // 60, int(offset_s) % 60

        # Status icon + color
        if status == "completed":
            icon = f"{_GREEN}✅"
            status_color = _GREEN
        elif status == "failed":
            icon = f"{_RED}💀"
            status_color = _RED
        elif status == "queued":
            icon = f"{_YELLOW}⏳"
            status_color = _YELLOW
        elif status == "cancelled":
            icon = f"{_DIM}⏭️"
            status_color = _DIM
        else:
            icon = f"{_DIM}?"
            status_color = _DIM

        # Provider short name
        prov_map = {
            "doubleword-397b": "DW-397B", "doubleword": "DW-397B",
            "claude-api": "Claude", "claude": "Claude",
            "gcp-jprime": "J-Prime",
        }
        prov_short = prov_map.get(provider, provider[:10])

        print(
            f"  {_DIM}[{om:02d}:{os_:02d}]{_RESET} "
            f"{icon}{_RESET} "
            f"{_CYAN}{short_id}{_RESET}  "
            f"{status_color}{status:<10s}{_RESET}  "
            f"{sensor}"
        )

        detail_parts = []
        if prov_short:
            detail_parts.append(f"via {prov_short}")
        detail_parts.append(f"{elapsed:.1f}s")
        if cost > 0:
            detail_parts.append(f"${cost:.4f}")
        if tool_calls:
            detail_parts.append(f"{tool_calls} tools")
        if files_changed:
            detail_parts.append(f"{files_changed} files")
        if technique:
            detail_parts.append(technique)

        print(f"  {_DIM}         {'  │  '.join(detail_parts)}{_RESET}")

        # ── Check for ledger entries ──
        ledger_path = _PROJECT_ROOT / ".jarvis" / "ouroboros" / "ledger" / f"{op_id}.jsonl"
        if ledger_path.exists():
            try:
                ledger_lines = ledger_path.read_text().strip().splitlines()
                phases = []
                for ll in ledger_lines:
                    entry = json.loads(ll)
                    phase = entry.get("phase", entry.get("state", ""))
                    if phase:
                        phases.append(phase)
                if phases:
                    chain = " → ".join(phases)
                    print(f"  {_DIM}         {chain}{_RESET}")
            except Exception:
                pass

        print()

    # ── Footer ──
    top_sensors = data.get("top_sensors", [])
    if top_sensors:
        print(f"  {_BOLD}Top Sensors:{_RESET}")
        for name, count in top_sensors[:5]:
            print(f"    {name:<30s} {count} ops")
        print()

    convergence = data.get("convergence_state", "")
    if convergence:
        slope = data.get("convergence_slope", 0.0)
        r2 = data.get("convergence_r2", 0.0)
        print(
            f"  {_BOLD}Convergence:{_RESET} {convergence}  "
            f"{_DIM}(slope={slope:.4f}, R²={r2:.2f}){_RESET}"
        )
        print()

    print(f"{'═' * 64}\n")


def main(argv: "list[str] | None" = None) -> None:
    # ``argv`` defaults to ``None`` (reads ``sys.argv`` -- unchanged legacy
    # behavior for direct ``python3 scripts/...`` invocation). The ``ov``
    # console script passes a translated argv so both front-ends share this
    # single bootstrap (DRY -- spec 2026-07-06 §4.3).
    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="ouroboros_battle_test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
        {_BOLD}{_CYAN}Ouroboros Battle Test Runner{_RESET}
        {_DIM}Autonomous self-developing AI session{_RESET}

        Boots the full Ouroboros + Venom + Trinity Consciousness stack.
        The organism finds work, reads code, generates Manifesto-aligned
        fixes, runs tests, iteratively converges, commits with its
        signature, and learns from outcomes. Autonomously. In parallel.

        {_BOLD}6-Layer Architecture:{_RESET}
          {_CYAN}1.{_RESET} Strategic Direction  {_DIM}Manifesto principles → every prompt{_RESET}
          {_CYAN}2.{_RESET} Trinity Consciousness {_DIM}Memory + prediction + learning{_RESET}
          {_CYAN}3.{_RESET} Event Spine           {_DIM}FileWatchGuard → TrinityEventBus → sensors{_RESET}
          {_CYAN}4.{_RESET} Ouroboros Pipeline    {_DIM}Governance + routing + parallel ops{_RESET}
          {_CYAN}5.{_RESET} Venom Agentic Loop    {_DIM}bash, web_search, run_tests, L2 repair{_RESET}
          {_CYAN}6.{_RESET} Thought Log           {_DIM}Observable reasoning + signed commits{_RESET}
        """),
        epilog=textwrap.dedent(f"""\
        {_BOLD}Examples:{_RESET}
          %(prog)s -v                          {_DIM}# Default: $0.50, 600s idle{_RESET}
          %(prog)s --cost-cap 2.00 -v          {_DIM}# Extended: $2.00 budget{_RESET}
          %(prog)s --cost-cap 0.10 -v          {_DIM}# Quick test: $0.10 budget{_RESET}

        {_BOLD}Artifacts produced:{_RESET}
          {_DIM}ouroboros/battle-test/<timestamp>    Git branch with autonomous commits
          .jarvis/ouroboros_thoughts.jsonl     Reasoning thread
          .jarvis/test_results.json            Structured test results
          .ouroboros/sessions/bt-*/             Session summary + cost tracker{_RESET}

        {_BOLD}Commit signature:{_RESET}
          {_DIM}Author: JARVIS Ouroboros <ouroboros@jarvis.local>
          Generated-By: Ouroboros + Venom + Consciousness
          Signed-off-by: JARVIS Ouroboros <ouroboros@jarvis.local>{_RESET}
        """),
    )
    # Presentation-aware session budget default (2026-07-18 root cause:
    # bare `ov` died in 4 minutes — the $0.50 SOAK default collided with
    # the $0.50 worst-case per-op Claude reservation, so the FIRST op
    # was structurally unaffordable → 3 refusals → hibernation →
    # session_exhausted. A product cockpit must afford real work out of
    # the box; soak/CI keeps the conservative cap.
    _cockpit_boot = (os.environ.get(
        "JARVIS_OV_PRESENTATION", "",
    ).strip().lower() == "cockpit")
    _default_cap = (
        os.environ.get("JARVIS_COCKPIT_COST_CAP", "2.50")
        if _cockpit_boot
        else os.environ.get("OUROBOROS_BATTLE_COST_CAP", "0.50")
    )
    parser.add_argument(
        "--cost-cap",
        type=float,
        default=float(_default_cap),
        metavar="USD",
        help=(
            "Session budget in USD (cockpit default 2.50 via "
            "JARVIS_COCKPIT_COST_CAP; soak default 0.50 via "
            "OUROBOROS_BATTLE_COST_CAP)"
        ),
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.environ.get("OUROBOROS_BATTLE_IDLE_TIMEOUT", "600")),
        metavar="SEC",
        help="Inactivity timeout in seconds (env: OUROBOROS_BATTLE_IDLE_TIMEOUT, default: 600)",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=float(os.environ.get("OUROBOROS_BATTLE_MAX_WALL_SECONDS", "0")),
        metavar="SEC",
        help=(
            "Hard wall-clock ceiling on total session duration — fires stop_reason=wall_clock_cap "
            "when exceeded. 0 or unset = disabled (legacy behavior). Graduation soaks MUST set "
            "this (e.g. 2400 = 40 min) to guarantee deterministic termination when provider "
            "retry storms defeat --idle-timeout. Env: OUROBOROS_BATTLE_MAX_WALL_SECONDS."
        ),
    )
    parser.add_argument(
        "--production-soak",
        action="store_true",
        default=os.environ.get("OUROBOROS_PRODUCTION_SOAK", "").strip().lower() in ("1", "true", "yes", "on"),
        help=(
            "Slice 123 Phase 3: production T5-evidence profile. Overrides the "
            "battle-test defaults to scale up + remove leashes — cost-cap=25.00, "
            "idle-timeout=0 (no idle stop), max-wall-seconds=0 (no wall cap) — and "
            "enables process-isolated Oracle load (JARVIS_ORACLE_PROCESS_ISOLATION_ENABLED) "
            "+ boot-recovery quarantine (JARVIS_BOOT_RECOVERY_QUARANTINE_ENABLED). "
            "Explicit --cost-cap/--idle-timeout/--max-wall-seconds still win if also passed. "
            "Env: OUROBOROS_PRODUCTION_SOAK."
        ),
    )
    # Ticket C (2026-04-23): tri-state --headless. argparse doesn't
    # natively distinguish "flag absent" from "--no-headless" across
    # a single option — use a mutually-exclusive group so absence maps
    # to None (auto-detect via isatty in HarnessConfig.resolve_headless).
    _headless_group = parser.add_mutually_exclusive_group()
    _headless_group.add_argument(
        "--headless",
        dest="headless",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Skip the SerpentREPL input task — the TUI REPL is a no-op "
            "in headless runs and starting it against non-TTY stdin "
            "exits in ~16 log lines. When absent, auto-detects via "
            "``not sys.stdin.isatty()``. Env: OUROBOROS_BATTLE_HEADLESS."
        ),
    )
    _headless_group.add_argument(
        "--no-headless",
        dest="headless",
        action="store_const",
        const=False,
        help=(
            "Force the interactive REPL even when stdin isn't a TTY "
            "(rare; escape hatch). Overrides auto-detection."
        ),
    )
    parser.add_argument(
        "--branch-prefix",
        type=str,
        default=os.environ.get("OUROBOROS_BATTLE_BRANCH_PREFIX", "ouroboros/battle-test"),
        metavar="PREFIX",
        help="Git branch prefix (default: ouroboros/battle-test)",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=os.environ.get("JARVIS_REPO_PATH", str(_PROJECT_ROOT)),
        metavar="PATH",
        help="Repository root path (default: project root)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (shows thought process).",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help=(
            "Replay a previous session timeline instead of running live. "
            "Pass a session ID (e.g. bt-2026-04-08-143022) or a path to "
            "summary.json. Lists available sessions when set to 'list'."
        ),
    )
    # Phase 1 Slice 1.4 — deterministic re-execution.
    # Distinct from --replay (which is a read-only timeline display).
    # --rerun re-boots the harness in REPLAY mode against the recorded
    # decisions ledger so every captured decision is replayed without
    # calling its compute() function. The resulting session can then be
    # diffed against the original to prove determinism.
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help=(
            "Deterministic re-execution. Locates the session's persisted "
            "seed + decisions ledger under .jarvis/determinism/<id>/, "
            "applies the replay env vars, and runs the harness in "
            "REPLAY (or VERIFY) mode. Fails fast if the session has no "
            "recorded state. Pair with --rerun-mode for verify."
        ),
    )
    parser.add_argument(
        "--rerun-mode",
        type=str,
        default="replay",
        choices=("replay", "verify"),
        help=(
            "When --rerun is set: 'replay' (default) returns recorded "
            "decisions without calling compute(); 'verify' runs live AND "
            "asserts each decision matches the recorded output."
        ),
    )
    # Priority 2 Slice 5 — record-level fork replay.
    # Distinct from --rerun (which replays a full session). --rerun-from
    # loads the CausalityDAG, locates the target record, and forks
    # execution from that point. New decisions carry counterfactual_of.
    parser.add_argument(
        "--rerun-from",
        type=str,
        default=None,
        metavar="RECORD_ID|SESSION:PHASE",
        help=(
            "Fork replay from a specific decision record within the "
            "session specified by --rerun. Requires --rerun to identify "
            "the session. New decisions written during the forked run "
            "carry counterfactual_of=<original-record-id>. "
            "§37 Tier 2 #10: also accepts the form "
            "<session-id>:<phase> (e.g. bt-2026-05-05-120000:GENERATE) "
            "— the harness resolves the FIRST record in that phase via "
            "the canonical CausalityDAG and forks from it."
        ),
    )
    # Phase 8 surface wiring Slice 3 — multi-op timeline renderer.
    # Read-only over the decision-trace ledger; never boots the
    # battle-test stack. Default false until graduation; respects
    # JARVIS_PHASE8_MULTI_OP_RENDERER_ENABLED.
    parser.add_argument(
        "--multi-op",
        type=str,
        default=None,
        metavar="REF",
        help=(
            "Render a chronological multi-op timeline from the "
            "decision-trace ledger and exit. REF can be: 'list' to "
            "show recent op_ids; 'op-A,op-B,op-C' for a comma-list "
            "(<=16 ops); '@last:N' for the most-recent N ops; "
            "'session:bt-...' for ops in a battle-test session "
            "summary. Requires JARVIS_PHASE8_MULTI_OP_RENDERER_ENABLED."
        ),
    )
    parser.add_argument(
        "--multi-op-no-color",
        action="store_true",
        help=(
            "Disable ANSI color in --multi-op output (default: color "
            "ON when stdout is a TTY)."
        ),
    )
    # Phase 9 Slice 2 — synthetic workload injection for cadence soaks.
    # Default 0 = zero behavior change for non-cadence runs. Only the
    # cadence wrapper (run_live_fire_graduation_soak.sh + cron entry)
    # sets this to N >= 1. Operator binding 2026-05-05: composes
    # canonical UnifiedIntakeRouter pipeline + honest source token
    # ("cadence_synthetic") + transparent observability markers.
    # See docs/architecture/OUROBOROS_VENOM_PRD.md §36.5 priority #1.
    parser.add_argument(
        "--seed-intents",
        type=int,
        default=int(os.environ.get(
            "OUROBOROS_BATTLE_SEED_INTENTS", "0",
        )),
        metavar="N",
        help=(
            "Phase 9 cadence: inject N synthetic IntentEnvelopes "
            "via the canonical UnifiedIntakeRouter at boot to "
            "exercise the FSM (closes the headless-cadence "
            "zero-ops blocker). Default 0 = no injection. Hard-"
            "capped via JARVIS_PHASE9_SEED_INTENTS_MAX (default "
            "16, clamped [1, 64]). Headless-only — interactive "
            "sessions never inject. Envelopes carry "
            "source='cadence_synthetic' so operators can "
            "filter cadence load from real signal traffic. Env: "
            "OUROBOROS_BATTLE_SEED_INTENTS."
        ),
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Belt-and-suspenders repo-root anchor (run-#14 SOURCE fix).
    # Export the authoritative ``.git``-anchored root into the env BEFORE any
    # config is built, so ANY path that derived a root from cwd/'.' anywhere in
    # the pipeline (a site we may have missed) still resolves to the real repo
    # root rather than the process cwd. On the Linux node ``cwd != the cloned
    # repo`` -> a cwd-relative root reached ``_normalize`` -> 45 'outside repo
    # root' rejections -> the chaos test was never scoped-detected. ``setdefault``
    # preserves an operator-provided override. We set BOTH conventions:
    #   * ``JARVIS_REPO_PATH``    — read by ``resolve_repo_root`` / harness.
    #   * ``JARVIS_PROJECT_ROOT`` — read by ``GovernedLoopConfig.from_env``.
    try:
        from backend.core.ouroboros.governance.workspace_resolver import (
            resolve_repo_root as _resolve_repo_root,
        )

        _anchor = str(_resolve_repo_root())
    except Exception:  # noqa: BLE001 -- never block the soak on the anchor
        _anchor = str(_PROJECT_ROOT)
    os.environ.setdefault("JARVIS_REPO_PATH", _anchor)
    os.environ.setdefault("JARVIS_PROJECT_ROOT", _anchor)

    # ------------------------------------------------------------------
    # Slice 123 Phase 3 — --production-soak profile.
    # Scales the battle-test defaults up to a real long-term T5 evidence run
    # and enables the process-isolated Oracle load + boot-recovery quarantine.
    # Explicit flags still win: we only override a limit the operator did NOT
    # pass on the command line (detected via sys.argv, not the parsed value).
    # ------------------------------------------------------------------
    if getattr(args, "production_soak", False):
        import sys as _sys

        _passed = set(_sys.argv[1:])
        if "--cost-cap" not in _passed:
            args.cost_cap = 25.00
        if "--idle-timeout" not in _passed:
            args.idle_timeout = 0.0
        if "--max-wall-seconds" not in _passed:
            args.max_wall_seconds = 0.0
        # Enable the existing (Slice 112/113) process-isolated Oracle + the
        # Slice 123 quarantine — only if the operator hasn't set them otherwise.
        os.environ.setdefault("JARVIS_ORACLE_PROCESS_ISOLATION_ENABLED", "1")
        os.environ.setdefault("JARVIS_BOOT_RECOVERY_QUARANTINE_ENABLED", "1")
        print(
            "[production-soak] cost_cap=%.2f idle_timeout=%s max_wall=%s "
            "oracle_isolation=on quarantine=on"
            % (args.cost_cap, args.idle_timeout, args.max_wall_seconds)
        )

    # ------------------------------------------------------------------
    # Phase 1 Slice 1.4 — deterministic re-execution (--rerun)
    # ------------------------------------------------------------------
    # Resolve + apply replay env vars BEFORE any harness module is
    # imported / instantiated. The phase capture wrapper + decision
    # runtime read env at call time, so as long as we set env before
    # the first decide() call the harness boots in REPLAY mode
    # transparently. Fails fast on missing state — operator gets a
    # clear diagnostic instead of silent fall-through to fresh session.
    if args.rerun is not None:
        try:
            from backend.core.ouroboros.governance.determinism.session_replay import (
                render_plan_summary,
                setup_replay_from_cli,
            )
            _plan = setup_replay_from_cli(
                args.rerun, mode=args.rerun_mode, raise_on_failure=True,
            )
            print(render_plan_summary(_plan))
            print(
                f"  rerun_mode:     {args.rerun_mode}\n"
                f"  → harness will boot in {args.rerun_mode.upper()} "
                f"mode against the recorded ledger.\n"
            )
        except ValueError as exc:
            print(f"\n  {_RED}Replay setup failed:{_RESET}\n  {exc}\n")
            sys.exit(2)
        except Exception as exc:
            print(
                f"\n  {_RED}Replay subsystem unavailable:{_RESET} "
                f"{type(exc).__name__}: {exc}\n"
                "  Continuing with fresh-session boot.\n"
            )

    # ------------------------------------------------------------------
    # Priority 2 Slice 5 — record-level fork replay (--rerun-from)
    # §37 Tier 2 #10 — ALSO accepts <session>:<phase> form,
    # resolved via the canonical CausalityDAG before the existing
    # record_id codepath.
    # ------------------------------------------------------------------
    if args.rerun_from is not None:
        if args.rerun is None:
            print(
                f"\n  {_RED}--rerun-from requires --rerun <session-id>{_RESET}\n"
            )
            sys.exit(2)
        if ":" in args.rerun_from:
            try:
                _sess_part, _phase_part = args.rerun_from.split(":", 1)
                _sess_part = _sess_part.strip()
                _phase_part = _phase_part.strip()
                if _sess_part and _sess_part != args.rerun:
                    print(
                        f"\n  {_RED}--rerun-from session "
                        f"{_sess_part!r} disagrees with --rerun "
                        f"{args.rerun!r}{_RESET}\n"
                    )
                    sys.exit(2)
                from backend.core.ouroboros.governance.verification.causality_dag import (
                    build_dag,
                )
                _dag = build_dag(session_id=args.rerun)
                _record = _dag.first_record_in_phase(_phase_part)
                if _record is None:
                    _phases = _dag.distinct_phases()
                    print(
                        f"\n  {_RED}No records in phase "
                        f"{_phase_part!r} for session "
                        f"{args.rerun!r}{_RESET}\n"
                        f"  Available: "
                        f"{', '.join(_phases) if _phases else '(none)'}\n"
                    )
                    sys.exit(2)
                args.rerun_from = _record.record_id
                print(
                    f"\n  Resolved {_sess_part}:{_phase_part} → "
                    f"record_id {args.rerun_from}\n"
                )
            except SystemExit:
                raise
            except Exception as exc:
                print(
                    f"\n  {_RED}Phase-form resolution failed:"
                    f"{_RESET} {type(exc).__name__}: {exc}\n"
                )
                sys.exit(2)
        try:
            from backend.core.ouroboros.governance.verification.replay_from_record import (
                prepare_replay_from_record,
                apply_replay_from_record_env,
                render_replay_from_record_summary,
            )
            _fork_plan = prepare_replay_from_record(
                args.rerun, args.rerun_from, mode=args.rerun_mode,
            )
            print(render_replay_from_record_summary(_fork_plan))
            if _fork_plan.is_replayable:
                apply_replay_from_record_env(
                    _fork_plan, mode=args.rerun_mode,
                )
                print(
                    f"  → forking from record {args.rerun_from} "
                    f"in {args.rerun_mode.upper()} mode.\n"
                )
            else:
                print(
                    f"\n  {_RED}Fork setup failed: "
                    f"{_fork_plan.failure_reason}{_RESET}\n"
                )
                sys.exit(2)
        except Exception as exc:
            print(
                f"\n  {_RED}Fork subsystem unavailable:{_RESET} "
                f"{type(exc).__name__}: {exc}\n"
            )
            sys.exit(2)

    # ------------------------------------------------------------------
    # Replay mode — show a previous session timeline and exit
    # ------------------------------------------------------------------
    if args.replay is not None:
        _replay_session(args.replay)
        return

    # ------------------------------------------------------------------
    # Phase 8 Slice 3 — multi-op timeline render-and-exit
    # ------------------------------------------------------------------
    if args.multi_op is not None:
        _render_multi_op_and_exit(
            args.multi_op,
            color=(not args.multi_op_no_color) and sys.stdout.isatty(),
        )
        return

    # ------------------------------------------------------------------
    # Load environment
    # ------------------------------------------------------------------
    _load_env_files()
    os.environ.setdefault("JARVIS_GOVERNANCE_MODE", "governed")

    # ------------------------------------------------------------------
    # ov awakening Task 1 — resolve presentation mode + the structural
    # fatal bypass (Mandate 1). Both read env populated by _load_env_files
    # above, so this must come after it. _check_api_keys_or_die is
    # deliberately unconditional and outside every gate below — no mode
    # can suppress it.
    # ------------------------------------------------------------------
    _mode = resolve_presentation_mode()
    _check_api_keys_or_die()
    # AFTER the lane gate, never before: "the engine cannot serve" and "the
    # engine serves, but not what you asked for" are different faults, and
    # the first already has an owner. Running this first would report a
    # missing model when the truth is a stopped engine.
    _validate_model_pin_or_die()

    # ------------------------------------------------------------------
    # Zombie reaper — kill lingering battle tests from prior sessions
    # before they race us on API budget, git branches, and the intake
    # router lock. Opt-out with JARVIS_BATTLE_REAP_ZOMBIES=false.
    #
    # ov awakening Task 1: reaping + lock hygiene are FUNCTIONAL side
    # effects and always run in both presentation modes — only the
    # reaper's stdout banner is gated (quiet=True in COCKPIT). This is
    # handled here rather than inside _run_gated_boot_banners below
    # because _cleanup_stale_router_lock needs the reaped-PID set that
    # helper's contract doesn't expose.
    # ------------------------------------------------------------------
    _reap_enabled = os.environ.get("JARVIS_BATTLE_REAP_ZOMBIES", "true").lower() not in ("false", "0", "no", "off")
    if _reap_enabled:
        _reaped = _reap_zombies(quiet=(_mode is PresentationMode.COCKPIT))
        _cleanup_stale_router_lock(reaped_pids=_reaped)
        # Slice 48 — sweep multi-day stale .jarvis/*.lock debris (flock
        # auto-releases on death; these are inert crumbs that accumulate).
        with contextlib.suppress(Exception):
            _stale_lock_age_s = float(
                os.environ.get("JARVIS_STALE_LOCK_REAP_AGE_S", "86400") or "86400"
            )
            _reap_stale_jarvis_locks(
                _PROJECT_ROOT / ".jarvis", max_age_s=_stale_lock_age_s,
            )
            # ov cockpit silence Slice 2 Task 5 (F3) — tighter,
            # threshold-matched sweep for CrossProcessJSONL's own
            # *.jsonl.lock files (300s default vs. the 24h debris
            # default above). Runs after the coarse sweep so it never
            # duplicates work; unrelated .lock files are untouched.
            _reap_stale_cross_process_jsonl_locks(
                _PROJECT_ROOT / ".jarvis",
                quiet=(_mode is PresentationMode.COCKPIT),
            )

    # ------------------------------------------------------------------
    # P1 Slice 3 — Ledger Sovereignty B1.2 singleton lock.
    # Structural single-instance defense: composes the canonical
    # cross_process_jsonl.flock_critical_section primitive to take
    # a kernel-arbitrated LOCK_EX | LOCK_NB on a well-known path
    # under the repo root. Default-FALSE per §33.1 — when off, the
    # existing _single_flight_preflight pgrep diagnostic path is
    # unchanged. When on, runs BEFORE _single_flight_preflight so
    # the structural defense fires first; the pgrep layer remains
    # as a human-readable diagnostic.
    #
    # The flock fd is held by the ExitStack for the rest of main(),
    # released by the kernel on any process exit (clean, SIGKILL,
    # os._exit) — no leaked-lock failure mode.
    # ------------------------------------------------------------------
    _lock_stack = contextlib.ExitStack()
    try:
        from backend.core.ouroboros.battle_test.singleton_lock import (
            acquire_singleton,
            singleton_lock_enabled,
        )
    except Exception as _sl_imp_err:  # noqa: BLE001 — defensive
        # Substrate unavailable → fall through to pgrep preflight.
        logging.getLogger(__name__).debug(
            "[singleton_lock] substrate import failed: %r — "
            "falling through to pgrep preflight",
            _sl_imp_err,
        )
    else:
        if singleton_lock_enabled():
            _sl_result = _lock_stack.enter_context(
                acquire_singleton(
                    repo_root=Path(args.repo_path),
                ),
            )
            if not _sl_result.acquired:
                print(
                    f"{_RED}{_BOLD}✘ Another Ouroboros soak holds "
                    f"the singleton lock at "
                    f"{_sl_result.lock_path}{_RESET}",
                    file=sys.stderr,
                )
                print(
                    f"{_DIM}  Wait for the running soak to "
                    f"finish, or kill it.{_RESET}",
                    file=sys.stderr,
                )
                _lock_stack.close()
                sys.exit(75)  # EX_TEMPFAIL
    # Register stack close at process exit as belt-and-suspenders;
    # the with-block in main() drops it anyway, but atexit ensures
    # the lock is dropped even if a sys.exit path bypasses normal
    # unwind. NOTE: ExitStack.close() is idempotent.
    atexit.register(_lock_stack.close)

    # ------------------------------------------------------------------
    # Harness Epic Slice 2 — single-flight preflight. Reject concurrent
    # battle-test runs at the process level; the zombie reap above kills
    # DEAD lingering processes, this check rejects ALIVE concurrent
    # processes (operator launched twice by accident, etc.). Exit 75
    # (EX_TEMPFAIL) signals "try again later" to wrappers — distinct
    # from generic error code 1.
    # Master flag: JARVIS_BATTLE_SINGLE_FLIGHT_ENABLED (default true).
    #
    # ov awakening Task 1 fix — this guard is FUNCTIONAL (prevents two
    # sessions competing for budget), so it runs unconditionally in BOTH
    # presentation modes, structurally outside _run_gated_boot_banners.
    # COCKPIT gates only the happy-path chatter (quiet=True); the
    # conflict-path REJECTED block prints in every mode (same Mandate 1
    # rationale as _check_api_keys_or_die). It stays at this position —
    # after the reap (dead-PID locks must be cleaned first, or this
    # check false-positives on them) and after the singleton lock
    # (structural defense fires before the pgrep diagnostic).
    # ------------------------------------------------------------------
    if os.environ.get("JARVIS_BATTLE_SINGLE_FLIGHT_ENABLED", "true").lower() not in ("false", "0", "no", "off"):
        _single_flight_preflight(quiet=(_mode is PresentationMode.COCKPIT))

    # ------------------------------------------------------------------
    # Preflight checklist — pure ceremony, gated at the source: COCKPIT
    # withholds, SOAK calls through. Reap already ran (functionally)
    # above, so reap_enabled=False here.
    # ------------------------------------------------------------------
    _run_gated_boot_banners(_mode, reap_enabled=False)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level = _resolve_boot_log_level(_mode, verbose=args.verbose)
    logging.basicConfig(
        level=log_level,
        format=(
            f"{_DIM}%(asctime)s{_RESET} "
            f"[%(name)s] "
            f"%(levelname)s %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy loggers that flood DEBUG output with internal details
    for _noisy in (
        "fsevents", "watchdog", "watchdog.observers",  # file watcher internals
        "aiohttp.access", "urllib3", "urllib3.connectionpool",  # HTTP internals
        "chromadb", "chromadb.telemetry",  # vector store internals
        "anthropic._base_client", "anthropic._client",  # Anthropic SDK request/response dumps
        "httpcore", "httpx",  # HTTP transport internals
        "asyncio",  # event loop debug
        "aiosqlite",  # SQLite debug queries
        "markdown_it",  # rich.markdown transitive — per-token "entering fence/list/..." spam at DEBUG
        # Typing-responsiveness fix: prompt_toolkit logs at DEBUG on
        # every key event under -v mode. Each log line costs ~50us and
        # races with key handlers — operators perceived as typing
        # freeze. WARNING level keeps real warnings (e.g. terminal
        # capability missing) while silencing per-keystroke chatter.
        "prompt_toolkit",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Sovereign Telemetry (2026-06-20) — authoritative operator log-level override.
    # ``logging.basicConfig`` above is a NO-OP when an imported module already
    # installed a root handler (classic Python logging gotcha), so the root can end
    # up at WARNING and silence the dispatch_profiler's INFO op_summary trace. When
    # JARVIS_LOG_LEVEL is set we force it via explicit setLevel (which always wins),
    # on the root AND the dispatch surfaces the telemetry mesh needs. Unset =
    # legacy behavior (byte-identical).
    _ov_level = os.environ.get("JARVIS_LOG_LEVEL", "").strip().upper()
    if _ov_level:
        _ov_resolved = getattr(logging, _ov_level, None)
        if isinstance(_ov_resolved, int):
            logging.getLogger().setLevel(_ov_resolved)
            for _surface in (
                "backend.core.ouroboros.governance.candidate_generator",
                "backend.core.ouroboros.governance.doubleword_provider",
                "backend.core.ouroboros.telemetry.dispatch_profiler",
            ):
                logging.getLogger(_surface).setLevel(_ov_resolved)

    # Gap #7 follow-up: O+V's own boot-accounting loggers (module
    # discovery, kernel init, graceful-shutdown, termination-hook
    # registration) emit DEBUG/INFO during early boot that's pure
    # forensic noise for operators. Suppress under restraint;
    # operators debugging boot itself set JARVIS_BOOT_NOISE_VERBOSE=true
    # to bypass. Single source of truth in
    # ``presentation_restraint.BOOT_NOISE_LOGGER_NAMES``.
    try:
        from backend.core.ouroboros.battle_test.presentation_restraint import (
            is_restraint_enabled, suppress_boot_noise_logs,
        )
        if is_restraint_enabled():
            suppress_boot_noise_logs()
    except Exception:
        pass  # fail-closed: legacy verbose output if suppression fails

    # ------------------------------------------------------------------
    # Boot timing — instrument the heavy phases so optimization is
    # data-driven (not guesswork). Fires only when verbose mode is on
    # to keep non-debug runs silent.
    # ------------------------------------------------------------------
    try:
        from backend.core.ouroboros.battle_test.boot_timing import (
            get_default_timer,
        )
        _boot_timer = get_default_timer()
        _boot_timer.mark("script_logging_configured")
    except Exception:
        _boot_timer = None

    # ------------------------------------------------------------------
    # Aegis battle-test cap defaults (Slice 2B-iii.2)
    # ------------------------------------------------------------------
    # Structural fix for the $0.00 fail-closed defaults in
    # backend/core/ouroboros/aegis/flags.py — those production-safe
    # defaults refuse every lease, which is correct for production
    # but catastrophic for the battle-test soak. The helper installs
    # canonical battle-test caps ONLY if the operator hasn't already
    # set them (env-precedence preserved). Daemon-side defaults stay
    # strict per operator binding "ceiling should remain strict".
    # MUST be invoked before the Aegis preflight step spawns the
    # daemon (the daemon's BudgetCaps are read from env at boot).
    # NEVER raises — failure folds into CapsResult(ok=False).
    from backend.core.ouroboros.aegis.battle_test_defaults import (
        default_battle_test_caps as _default_battle_test_caps,
    )
    _caps_result = _default_battle_test_caps()
    _print_battle_test_defaults_banner(_caps_result, _mode)

    # ------------------------------------------------------------------
    # Aegis battle-test ledger hygiene (Slice 2B-iii.1)
    # ------------------------------------------------------------------
    # Rotates .jarvis/aegis/spend.jsonl + removes its .lock companion
    # BEFORE the Aegis preflight step spawns the daemon — so the
    # daemon's ImmutableBudgetStateMachine.replay_for_recovery() reads
    # a clean WAL and the new session boots with a fresh financial slate.
    # Closes the cost_ceiling_exceeded:session_cap_exceeded failure
    # mode surfaced by re-detonation soak bt-2026-05-24-225714 where
    # the daemon replayed a stale prior-session WAL and denied the
    # very first lease request. Gated by
    # JARVIS_AEGIS_BATTLE_TEST_HYGIENE_ENABLED (default TRUE).
    # NEVER raises — failure folds into HygieneResult(ok=False) and
    # the harness keeps booting (operator investigates via logs).
    # Production Aegis use NEVER touches the WAL; the helper is
    # battle-test-scoped.
    from backend.core.ouroboros.aegis.ledger_hygiene import (
        rotate_aegis_wal_for_battle_test as _rotate_aegis_wal_for_battle_test,
    )
    _hygiene_session_tag = f"pre-bt-{int(time.time())}"
    _hygiene_result = _rotate_aegis_wal_for_battle_test(
        session_tag=_hygiene_session_tag,
    )
    _print_ledger_hygiene_banner(_hygiene_result, _mode)

    # ------------------------------------------------------------------
    # Aegis preflight (Arc #1 — out-of-process egress + budget chokepoint)
    # ------------------------------------------------------------------
    # Gated on JARVIS_AEGIS_ENABLED. Slice 1 dark substrate, default
    # FALSE — returns SKIPPED_DISABLED with zero behavior change. When
    # operator opts in, spawns the Aegis subprocess, atomic-reads its
    # bootstrap payload, scrubs upstream credentials from this process,
    # and exposes JARVIS_AEGIS_URL + JARVIS_AEGIS_BOOTSTRAP_PSK for the
    # Slice 2 provider rewire. Runs BEFORE the harness import so any
    # provider module that captures creds at import time (e.g.,
    # doubleword_provider.py:43) sees a scrubbed env.
    from backend.core.ouroboros.aegis.preflight import (
        PreflightOutcome as _AegisPreflightOutcome,
        aegis_preflight as _aegis_preflight,
    )
    _aegis_result = asyncio.run(_aegis_preflight())
    if _aegis_result.outcome not in (
        _AegisPreflightOutcome.READY,
        _AegisPreflightOutcome.SKIPPED_DISABLED,
    ):
        print(
            f"[Aegis] preflight failed: {_aegis_result.outcome.value} — "
            f"{_aegis_result.detail}",
            file=sys.stderr,
        )
        sys.exit(1)
    if _aegis_result.outcome is _AegisPreflightOutcome.READY:
        _print_aegis_daemon_ready(_aegis_result, _mode)
        # Slice 24 — register the daemon in the LIVE (post-re-exec) battle-test
        # process so the child_reaper cascade targets it in the fast path. The
        # register in preflight._spawn_daemon runs in whatever process spawned
        # it; the battle-test os.execv re-exec (generation restart) clears that
        # in-memory registry, which is why the graceful-halt cascade saw
        # "0 children" (the daemon was still reaped by its own WorkerLifeline +
        # the fs-pool cleanup, so zero orphans held — this closes the fast-path
        # gap). Idempotent + fail-soft.
        try:
            from backend.core.ouroboros.governance.child_reaper import (
                register_child as _cr_register,
            )
            _pid = getattr(_aegis_result, "subprocess_pid", None)
            if _pid:
                _cr_register(int(_pid), role="aegis_daemon")
        except Exception:  # noqa: BLE001 — never block boot
            pass
        # Slice 125 — credential health probe. Prove the daemon injects a VALID
        # credential of the funded class BEFORE a multi-hour soak spends time.
        # Two arms (direct funded key vs Aegis-routed); a 402 through Aegis while
        # the direct key is 200 is an INJECTION FAILURE — fail loud rather than
        # silently 402 for hours and burn Claude. Gated; redacted logging only.
        try:
            from backend.core.ouroboros.aegis.credential_probe import (
                credential_probe_enabled as _probe_enabled,
                is_fatal as _probe_fatal,
                probe_dw_credential_health as _probe_health,
            )

            if _probe_enabled():
                _verdict = asyncio.run(_probe_health())
                print(f"[Aegis] credential probe: {_verdict.value}")
                if _probe_fatal(_verdict):
                    print(
                        f"[Aegis] FATAL: {_verdict.value} — the Aegis daemon is "
                        f"not injecting the funded provider credential. Refusing "
                        f"to start a long soak that would silently 402 / burn "
                        f"fallback credits. Fix the credential path and retry.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
        except SystemExit:
            raise
        except Exception as _probe_exc:  # noqa: BLE001 - probe must not block boot on its own bug
            print(f"[Aegis] credential probe skipped ({_probe_exc.__class__.__name__})")

    # ------------------------------------------------------------------
    # Build config + harness
    # ------------------------------------------------------------------
    if _boot_timer is not None:
        _boot_timer.begin("harness_module_import")
    from backend.core.ouroboros.battle_test.harness import BattleTestHarness, HarnessConfig
    if _boot_timer is not None:
        _boot_timer.end("harness_module_import")

    # Ticket C: when CLI did not specify --headless/--no-headless (args.headless
    # is None), fall back to the env var OUROBOROS_BATTLE_HEADLESS via
    # HarnessConfig.resolve_headless() which also does isatty auto-detect.
    # CLI wins over env when set — consistent with --cost-cap etc.
    _env_headless = os.environ.get("OUROBOROS_BATTLE_HEADLESS", "").strip().lower()
    if args.headless is None and _env_headless:
        if _env_headless in ("1", "true", "yes", "on"):
            args.headless = True
        elif _env_headless in ("0", "false", "no", "off"):
            args.headless = False

    config = HarnessConfig(
        repo_path=Path(args.repo_path),
        cost_cap_usd=args.cost_cap,
        idle_timeout_s=args.idle_timeout,
        max_wall_seconds_s=args.max_wall_seconds or None,
        headless=args.headless,
        branch_prefix=args.branch_prefix,
        seed_intents=int(args.seed_intents or 0),
    )

    # PRD §11 (S2) wiring B1 — bridge --cost-cap into S2's session
    # budget env so the CLI flag transparently flows through. Uses
    # setdefault: if operator explicitly set JARVIS_S2_SESSION_BUDGET_USD
    # (Tier 1 of the precedence chain), that wins; otherwise --cost-cap
    # populates the Tier-2 fallback by way of Tier-1 env.
    os.environ.setdefault(
        "JARVIS_S2_SESSION_BUDGET_USD", str(args.cost_cap),
    )

    if _boot_timer is not None:
        with _boot_timer.phase("harness_construct"):
            harness = BattleTestHarness(config)
    else:
        harness = BattleTestHarness(config)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        harness.register_signal_handlers(loop)
    except Exception:
        pass  # Windows or unsupported platform

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if _boot_timer is not None:
        _boot_timer.mark("harness_run_started")

    # Print boot-timing summary on console once subsystems have booted
    # (best effort — fires after the REPL banner is up). Verbose mode
    # only — non-verbose runs stay silent.
    if args.verbose and _boot_timer is not None:
        async def _emit_boot_timing_after_settle():
            # Wait briefly for subsystem boot to settle (the harness
            # boots most layers within ~2s). After that, print the
            # summary so operators see what was slow.
            await asyncio.sleep(2.5)
            try:
                from rich.console import Console as _C
                _boot_timer.emit_summary(
                    console=_C(force_terminal=True), threshold_ms=10.0,
                )
            except Exception:
                pass
        loop.create_task(_emit_boot_timing_after_settle())

    interrupted = False
    try:
        loop.run_until_complete(harness.run())
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{_YELLOW}Interrupted — shutting down gracefully...{_RESET}")
    finally:
        # Phase 9.1c (Fix A) — arm the BoundedShutdownWatchdog BEFORE the
        # post-asyncio teardown phase, regardless of stop_reason. The
        # signal-handler arming at harness.py:3276 only fires on
        # signal-induced shutdown — clean shutdowns (idle_timeout /
        # budget_exhausted / wall_clock_cap) skip the arm and have NO
        # escape hatch if shutdown_default_executor() wedges on a
        # non-daemon ThreadPoolExecutor worker. The Phase 9.1 once-run
        # (session bt-2026-04-27-085300) revealed exactly this hang:
        # process completed _generate_report cleanly at 02:09:25, then
        # sat for 1h 50m+ in the executor shutdown. Daemon-thread
        # watchdog → no Py_FinalizeEx interference; if shutdown wedges
        # past default_deadline_s (default 30s), os._exit(75) fires.
        #
        # arm() is first-wins-with-reset: if signal-handler already
        # armed-then-disarmed earlier, this arm() re-arms cleanly.
        # If everything below completes within deadline, the daemon
        # thread dies with the interpreter — no os._exit fires.
        try:
            from backend.core.ouroboros.battle_test.shutdown_watchdog import (  # noqa: E501
                default_deadline_s as _bsw_deadline_s,
            )
            _wdg = getattr(harness, "_shutdown_watchdog", None)
            if _wdg is not None:
                _wdg.arm(
                    reason="post_asyncio_teardown",
                    deadline_s=_bsw_deadline_s(),
                )
        except Exception:  # noqa: BLE001 — never let watchdog arm
            # crash the script's clean-exit path.
            pass

        # Shutdown hygiene (Python 3.9+): drain pending async generators
        # and thread-pool executor tasks before closing the loop. Without
        # this, background asyncio.to_thread / run_in_executor callbacks
        # can race loop.close() and raise "RuntimeError: Event loop is
        # closed" during otherwise-clean session exit. See
        # memory/project_async_shutdown_race_triage.md for the full
        # traceback + root cause analysis.
        # THE MISSING PHASE. `asyncio.run` does four things here and this
        # block did the last three; the first — cancel every remaining task
        # and let it finish — was absent. Without it any task the harness did
        # not explicitly own is still parked inside an async generator when
        # `shutdown_asyncgens()` calls `aclose()` on it, which is precisely
        # `RuntimeError: aclose(): asynchronous generator is already running`
        # (bt-2026-08-18-021438, on StreamEventBroker.stream_iter and
        # ExecutionGraphProgressTracker._drain_subscriber), and precisely the
        # "Task was destroyed but it is pending!" panic two seconds later.
        #
        # Bounded, unlike the stdlib's unbounded gather: a task that swallows
        # CancelledError must not hang teardown, and survivors are NAMED
        # rather than silently abandoned.
        try:
            from backend.core.ouroboros.battle_test.loop_teardown import (  # noqa: E501
                cancel_remaining_tasks as _cancel_remaining_tasks,
            )
            _td_report = loop.run_until_complete(_cancel_remaining_tasks())
            if not _td_report.skipped and _td_report.cancelled:
                _log = logging.getLogger("Ouroboros.LoopTeardown")
                (_log.warning if not _td_report.clean else _log.info)(
                    "%s", _td_report.render(),
                )
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass

        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        loop.close()

    # ------------------------------------------------------------------
    # Hot-reload restart respawn (Manifesto §6)
    # ------------------------------------------------------------------
    # If the harness's stop_reason starts with "restart_pending:", the
    # ModuleHotReloader queued a restart because O+V self-modified a
    # quarantined or unsafe-to-reload module. Re-exec this same script
    # with identical argv so the new code is loaded fresh from disk.
    #
    # JARVIS_RESTART_GENERATION (private env var) tracks the depth of the
    # respawn chain to prevent infinite loops if the same self-mod keeps
    # tripping. Capped at JARVIS_RESTART_MAX (default 5).
    if not interrupted and getattr(harness, "stop_reason", "").startswith("restart_pending:"):
        max_restarts = int(os.environ.get("JARVIS_RESTART_MAX", "5"))
        gen = int(os.environ.get("JARVIS_RESTART_GENERATION", "0"))
        if gen >= max_restarts:
            print(
                f"\n{_YELLOW}[respawn] restart cap reached "
                f"(JARVIS_RESTART_GENERATION={gen} >= JARVIS_RESTART_MAX={max_restarts}); "
                f"exiting normally instead of re-execing.{_RESET}"
            )
            sys.exit(0)
        print(
            f"\n{_YELLOW}[respawn] {harness.stop_reason} — "
            f"re-execing battle test (generation {gen + 1}/{max_restarts}){_RESET}"
        )
        os.environ["JARVIS_RESTART_GENERATION"] = str(gen + 1)
        # os.execv replaces this process — code after this line is unreachable.
        # argv[0] is the interpreter, argv[1] is this script, then the original CLI flags.
        os.execv(sys.executable, [sys.executable, *sys.argv])

    # Stateful KeepAlive Handoff (operator-authorized 2026-07-18): every
    # CLEAN completion — idle_timeout, budget_exhausted, wall_clock_cap,
    # operator shutdown — exits 0 EXPLICITLY. Under the resident
    # launchd agent (KeepAlive.SuccessfulExit=false) this lets the
    # organism SLEEP on intentional exit instead of entering a
    # CPU-burning restart loop against the host OS; the thin client's
    # cold-boot path revives it on the next operator touch. Crashes
    # (tracebacks, os._exit(75) wedges) exit nonzero and ARE revived.
    sys.exit(0)


if __name__ == "__main__":
    main()
