"""Thin-Client Split — bare ``ov`` becomes a sub-second presentation shell.

Operator authorization 2026-07-18. The root cause of slow ``ov`` was
architectural: the entry process imported the ENTIRE organism (the
battle-test bootstrap chain) before painting anything. This module is
the bifurcation: the presentation plane and the domain plane are now
separate EXECUTION BOUNDARIES, not lazily-interleaved imports.

Strict import isolation (mandate 2, AST-pinned by the test spine):
this module — and ``cli/ov.py`` — may import ONLY the standard
library, the TUI layer (``backend.core.ouroboros.ui.*``), and the IPC
client (``battle_test.cockpit_attach``). Zero ML, zero embeddings,
zero governance/domain modules. The organism lives in a DETACHED
process spawned via ``subprocess.Popen``.

Flow for bare ``ov``::

    crest (instant) → zero-trust socket probe
      live   → attach (sub-second warm path)
      stale  → unlink ghost socket → cold boot
      absent → cold boot
    cold boot: Popen detached daemon → waking breadcrumb →
               attach the moment the bridge binds → same split-plane
               TUI; boot lines arrive through the SAME PresentationRouter
               chokepoint the warm path uses (DRY — one conformance).

Zero-Trust probe (mandate 2): the socket FILE is never trusted — a
violently-killed daemon leaves a ghost inode that connects refuse. The
probe is a bounded real connection attempt; refusal classifies the
socket as STALE and the client unlinks it before cold-booting. No
traceback ever reaches the operator.

Resident Organism (``ov daemon --install``): generates a per-user
macOS launchd Agent plist (POSIX paths, no hardcoded machine state —
interpreter, repo root, and log paths resolved at install time),
piping raw stderr/stdout to dedicated logs while the IPC socket serves
thin clients. ``--uninstall`` reverses it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import plistlib
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

#: THE MODULE USED THIS AND NEVER DEFINED IT.
#:
#: `spawn_daemon` has called `logger.debug(...)` since 2026-08-08 (#70440)
#: while `logging` was not imported and no `logger` existed. Every ignition
#: therefore raised `NameError: name 'logger' is not defined`, its blanket
#: `except Exception: return None` swallowed it, and `ensure_daemon` reported
#: "⚠ ignition failed" -- so bare `ov` could not cold-boot the organism at
#: all for ten days. Stdlib only, per the import-isolation mandate above.
logger = logging.getLogger("Ouroboros.ThinClient")

_TRUTHY = ("1", "true", "yes", "on")

#: launchd label — stable identity for the resident agent.
AGENT_LABEL = "com.jarvis.ov.daemon"


def thin_client_enabled() -> bool:
    """Master gate — default ON (the split IS the product now);
    ``JARVIS_OV_THIN_CLIENT=false`` or ``ov --legacy-boot`` reverts to
    the in-process organism boot. NEVER raises."""
    return os.environ.get(
        "JARVIS_OV_THIN_CLIENT", "1",
    ).strip().lower() in _TRUTHY


def repo_root() -> Path:
    """The checkout root (…/JARVIS-AI-Agent*), resolved structurally
    from this file — never from cwd."""
    return Path(__file__).resolve().parents[4]


def _probe_timeout_s() -> float:
    try:
        raw = float(os.environ.get("JARVIS_OV_PROBE_TIMEOUT_S", "0.4"))
    except (TypeError, ValueError):
        raw = 0.4
    return max(0.05, min(3.0, raw))


#: How much longer than a typical boot we are willing to wait before saying
#: it did not come up. 3x, because the estimator's input is a MEDIAN: half
#: of all boots are slower than it by construction, and a cold cache, a
#: model load and 22 sensors arming make the tail long.
_BOOT_WAIT_TOLERANCE = 3.0
#: Never wait less than this, whatever the history claims. A ledger holding
#: only fast samples must not be able to produce a window too short for any
#: real boot to finish in.
_BOOT_WAIT_FLOOR_S = 120.0


def _boot_wait_s() -> float:
    """How long to wait for a cold boot, DERIVED from observed boots.

    This was the constant 120. Measured 2026-09-05: a cold boot was still
    wiring the cockpit at 114s and the client gave up at 120, printing "the
    organism did not come up" about an organism that came up fine seconds
    later — the operator sees a failure, the daemon is healthy, and nothing
    connects the two.

    A fixed number cannot be right on both a warm laptop and a box loading
    an 18 GB model, so it is derived from the operator's own boots — the
    same pattern `expected_boot_s` already uses for the progress estimate,
    and `session_economics.derived_cost_cap` uses for money. The explicit
    env var still wins, for a box that knows better than its history.
    """
    raw = (os.environ.get("JARVIS_OV_BOOT_WAIT_S", "") or "").strip()
    if raw:
        try:
            return max(5.0, min(900.0, float(raw)))
        except (TypeError, ValueError):
            pass
    observed = None
    try:
        from backend.core.ouroboros.cli import boot_progress as _bp  # noqa: PLC0415

        observed = _bp.expected_boot_s()
    except Exception:  # noqa: BLE001 — no history is not a failure
        observed = None
    if observed and observed > 0:
        return max(_BOOT_WAIT_FLOOR_S,
                   min(900.0, observed * _BOOT_WAIT_TOLERANCE))
    return _BOOT_WAIT_FLOOR_S


#: POSIX ``sysexits.h`` EX_CONFIG, as the daemon spends it: the pinned
#: model is not one this node serves. Declared here as well because this
#: module must not import the daemon script to read one integer, and a
#: test pins the two to stay equal — a code the client mis-reads would put
#: a crash message over a configuration refusal.
EXIT_MODEL_PIN_UNAVAILABLE = 78

#: How many stall windows a boot may take in TOTAL. The stall window says
#: how long silence is tolerated; this says how long the whole thing may
#: run even while it keeps talking. 4x.
_BOOT_CEILING_MULTIPLE = 4.0


def _boot_ceiling_s(stall_s: Optional[float] = None) -> float:
    """The absolute cap on a boot wait — deliberately BLIND to progress.

    The stall window below can be renewed by the daemon writing to its log,
    which is exactly what makes it honest about a slow boot. It is also
    what would make it unbounded if a boot could spin while logging: a
    wait that a live process can extend forever is not a bound.

    So the pair is: a renewable stall window that decides "is it still
    working", and this static ceiling that decides "has it had long
    enough", answered from the clock alone. Nothing the daemon does moves
    it. That is the same separation the harness keeps between its idle
    timeout and `--max-wall-seconds`, and for the same reason — a bound
    that shares a signal with the thing it bounds is not a bound.
    """
    raw = (os.environ.get("JARVIS_OV_BOOT_CEILING_S", "") or "").strip()
    if raw:
        try:
            return max(5.0, min(3600.0, float(raw)))
        except (TypeError, ValueError):
            pass
    base = stall_s if stall_s is not None else _boot_wait_s()
    return max(base, min(3600.0, base * _BOOT_CEILING_MULTIPLE))


def _boot_log_mark() -> int:
    """Bytes the daemon has written so far — a cheap, monotonic proof of
    work. Size ONLY: the wait must never parse or interpret what the boot
    says about itself, or a daemon that lies in its log could talk its way
    past its own deadline. -1 when unreadable, which reads as no progress
    and therefore never extends anything. NEVER raises."""
    try:
        from backend.core.ouroboros.cli import boot_progress as _bp  # noqa: PLC0415

        return int(_bp.log_size(str(daemon_log_path())))
    except Exception:  # noqa: BLE001
        return -1


# ---------------------------------------------------------------------------
# Zero-Trust socket probe
# ---------------------------------------------------------------------------


async def probe_socket(
    path: Path, timeout: Optional[float] = None, *, deep: bool = False,
) -> str:
    """Classify the attach socket with a REAL bounded connect:

    * ``"live"``    — something accepted (a daemon is home). With
      ``deep=True``, ONLY when the server actually *served* its first
      frame within the bound — the same application-level contract
      ``CockpitAttachClient.connect`` consumes.
    * ``"booting"`` — (deep only) the connection was accepted but no
      frame arrived within the bound. This is NOT a ghost: a UDS
      ``listen()`` backlog completes handshakes at the KERNEL level
      even while a boot-starved event loop has yet to run ``accept()``
      or write the hydration line. Callers must WAIT, never clean.
    * ``"stale"``   — inode exists but connection REFUSED/reset (ghost
      of a violently-killed daemon). A connect *timeout* is NOT stale:
      a starved-but-live organism's loop can miss the accept window
      while the kernel backlog still exists — that classifies as
      ``"booting"`` (wait, never clean). Only a kernel-level refusal
      proves nobody is home. (Root cause of the 2026-07-23 class: a
      starvation-lagged live soak was probed "stale", the CLI unlinked
      its LIVE socket, and the organism became permanently unattachable
      since the bridge binds once at boot.)
    * ``"absent"``  — no inode at all

    ``timeout`` overrides the env default for a STRICT live-validation
    bound (Phase 7 health handshake). Non-invasive: opens, confirms, and
    immediately closes the transport — no lingering connection on the
    daemon's multiplexer. NEVER raises; NEVER hangs."""
    try:
        if not path.exists():
            return "absent"
        bound = timeout if timeout is not None else _probe_timeout_s()
        try:
            r, w = await asyncio.wait_for(
                asyncio.open_unix_connection(path=str(path)),
                timeout=bound,
            )
        except asyncio.TimeoutError:
            # Timeout ≠ dead. A starved event loop misses the accept
            # window while the listener still exists — treat exactly
            # like an accepted-but-unserved handshake: WAIT, never clean.
            return "booting"
        except (ConnectionRefusedError, ConnectionResetError, OSError):
            return "stale"
        try:
            if not deep:
                return "live"
            # Handshake depth: connection-established is a kernel fact;
            # readiness is an APPLICATION fact.
            #
            # This demanded ONE BYTE, and one byte is a weaker claim than the
            # contract it vouches for. ``CockpitAttachClient._connect_once``
            # needs a COMPLETE LINE that PARSES as JSON — anything else lands
            # in its except-clause as "dead", which fails fast with no retry.
            # So a daemon mid-write (partial frame, split across the bound)
            # satisfied the probe and then failed the attach, producing the
            # contradiction the operator sees:
            #
            #     ⏺ organism live — attaching
            #     no organism awake — nothing to attach to.
            #
            # The probe now consumes exactly what attach consumes. The law
            # ``ov_doctor`` already states — no probe may be weaker than the
            # contract it vouches for — applied to the probe that ignition
            # itself depends on.
            try:
                line = await asyncio.wait_for(r.readline(), timeout=bound)
            except asyncio.TimeoutError:
                return "booting"
            except (ConnectionResetError, OSError):
                return "stale"
            if not line:
                return "booting"          # EOF mid-handshake: still coming up
            try:
                json.loads(line)
            except (ValueError, TypeError):
                # A complete but unparseable frame means the bridge protocol
                # is broken, not that nobody is home. "booting" (wait, bounded,
                # then an honest timeout) over "stale" DELIBERATELY: "stale"
                # authorises unlinking the socket, and the bridge binds ONCE at
                # boot, so cleaning a live-but-misbehaving organism's socket
                # makes it permanently unattachable — the 2026-07-23 class.
                # Waiting merely costs a bounded wait and a truthful message.
                return "booting"
            return "live"
        finally:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return "stale" if path.exists() else "absent"


async def probe_tcp(host: str, port: int, timeout: float = 2.0) -> str:
    """TCP companion to ``probe_socket`` — a REAL bounded, non-invasive
    connect to prove a listener's event loop is live:

    * ``"live"``    — the connection was accepted
    * ``"refused"`` — ConnectionRefusedError (nothing/dead bound the port)
    * ``"timeout"`` — no response within ``timeout`` (deadlocked loop)

    Opens, confirms, immediately closes — leaves no dangling connection.
    NEVER raises."""
    try:
        try:
            _r, w = await asyncio.wait_for(
                asyncio.open_connection(host=host, port=port),
                timeout=timeout,
            )
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
            return "live"
        except asyncio.TimeoutError:
            return "timeout"
        except (ConnectionRefusedError, ConnectionResetError):
            return "refused"
        except OSError:
            return "refused"
    except Exception:  # noqa: BLE001
        return "refused"


async def probe_http(
    host: str, port: int, timeout: float = 2.0,
    path: str = "/observability/health",
) -> str:
    """APPLICATION-level liveness probe. A bare TCP connect proves only
    that the KERNEL accepted the socket — a deadlocked event loop still
    completes the handshake at the kernel layer. So this sends a real
    HTTP request and awaits a response within ``timeout``:

    * ``"live"``    — an HTTP response came back (the loop is processing)
    * ``"timeout"`` — connected, but the app never answered (ZOMBIE: the
      event loop is wedged / never bound its handlers)
    * ``"refused"`` — nothing/dead bound the port

    Non-invasive: sends ``Connection: close`` and closes the transport —
    no lingering connection on the server. NEVER raises."""
    writer = None
    try:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host=host, port=port), timeout=timeout)
        except asyncio.TimeoutError:
            return "timeout"
        except (ConnectionRefusedError, ConnectionResetError, OSError):
            return "refused"
        try:
            req = (f"GET {path} HTTP/1.0\r\nHost: {host}\r\n"
                   "Connection: close\r\n\r\n")
            writer.write(req.encode())
            await asyncio.wait_for(writer.drain(), timeout=timeout)
            data = await asyncio.wait_for(reader.read(64), timeout=timeout)
            # Any bytes back = the app's event loop answered → live. Empty
            # read = connected but the loop produced nothing → wedged.
            return "live" if data else "timeout"
        except asyncio.TimeoutError:
            return "timeout"          # accepted but no app response → ZOMBIE
        except (ConnectionResetError, OSError):
            return "refused"
    except Exception:  # noqa: BLE001
        return "refused"
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass


def clean_stale_socket(path: Path) -> bool:
    """Unlink a ghost socket. True when the path is gone afterwards.
    NEVER raises."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        return not path.exists()
    except Exception:  # noqa: BLE001
        return False



def _cockpit_cost_cap() -> "tuple[str | None, str]":
    """The session ceiling and its basis, or ``(None, reason)``.

    ``None`` means "do not assert a ceiling" — the caller then leaves
    OUROBOROS_BATTLE_COST_CAP unset and the harness applies its own default.

    That is deliberate, and this pin caught the alternative: an earlier version
    fell back to a literal of its own, which made this a SECOND default for the
    same variable — the exact defect the estimator was written to end,
    reintroduced in the code that ended it. If we cannot compute a ceiling we
    do not invent one; we defer to the layer that already has an answer.
    """
    try:
        from backend.core.ouroboros.battle_test.session_economics import (
            cockpit_cost_cap,
        )
        return cockpit_cost_cap()
    except Exception:  # noqa: BLE001
        return (None, "not asserted — estimator unavailable, harness default "
                      "applies")

# ---------------------------------------------------------------------------
# Detached cold boot
# ---------------------------------------------------------------------------


def daemon_argv() -> list:
    """The detached organism's argv — the SAME bootstrap the operator
    would run by hand (one source of truth for flags)."""
    return [
        sys.executable,
        str(repo_root() / "scripts" / "ouroboros_battle_test.py"),
        "--headless",
    ]


def daemon_log_path() -> Path:
    return repo_root() / ".jarvis" / "logs" / "ov-daemon.log"


def _log_max_bytes() -> int:
    try:
        return max(1024, int(os.environ.get(
            "JARVIS_OV_DAEMON_LOG_MAX_BYTES", str(50 * 1024 * 1024),
        )))
    except (TypeError, ValueError):
        return 50 * 1024 * 1024


def _log_backups() -> int:
    try:
        return max(1, int(os.environ.get(
            "JARVIS_OV_DAEMON_LOG_BACKUPS", "3",
        )))
    except (TypeError, ValueError):
        return 3


def rollover_daemon_log(path: Optional[Path] = None) -> bool:
    """Circadian Log Janitor for the RAW daemon sinks (the streams
    launchd / Popen capture BELOW the logging layer). Uses the
    standard ``RotatingFileHandler`` machinery (``shouldRollover`` /
    ``doRollover`` — never manual byte manipulation) at each
    spawn/boot boundary: oversized logs shift to ``.1``/``.2``/``.3``
    and the oldest falls off. Caveat, stated honestly: a stream fd
    HELD OPEN by launchd keeps writing to the rotated inode until the
    next daemon start — the bound is therefore max_bytes × (backups+2)
    per sink, never unbounded. NEVER raises."""
    import logging as _logging
    import logging.handlers as _lh
    target = path or daemon_log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = _lh.RotatingFileHandler(
            str(target),
            maxBytes=_log_max_bytes(),
            backupCount=_log_backups(),
            encoding="utf-8",
            delay=True,                    # never touches a healthy file
        )
        probe = _logging.LogRecord(
            "ov.janitor", _logging.INFO, __file__, 0, "", (), None,
        )
        try:
            if handler.shouldRollover(probe):
                handler.doRollover()
                return True
            return False
        finally:
            handler.close()
    except Exception:  # noqa: BLE001
        return False


def spawn_daemon(
    *, spawner: Callable[..., Any] = subprocess.Popen,
) -> Optional[Any]:
    """Launch the organism DETACHED (new session — it survives this
    terminal closing) with stdout+stderr piped to the daemon log.
    Returns the child process HANDLE (``.pid`` / ``.poll()``), or None
    on spawn failure — the handle makes an instant single-flight
    rejection (exit 75) OBSERVABLE instead of a 120s blind socket
    vigil. NEVER raises."""
    try:
        log = daemon_log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        rollover_daemon_log(log)           # janitor at every ignition
        env = dict(os.environ)
        # The resident organism serves cockpits — give it the cockpit
        # session economics unless the operator overrode them.
        # DERIVED, not declared. This used to be a hardcoded dollar figure,
        # written here and again in the launchd agent — two copies of a
        # budget, which
        # means the effective one is whichever nobody remembered to change.
        # The ceiling now comes from the sessions that actually spent money,
        # and it carries the basis it rests on.
        _cap, _cap_basis = _cockpit_cost_cap()
        if _cap is not None:
            env.setdefault("OUROBOROS_BATTLE_COST_CAP", _cap)
        # THE BASIS TRAVELS WITH THE NUMBER.
        #
        # `derived_cost_cap` computes both and its docstring is explicit that
        # "a surface that shows the number without it repeats the exact defect
        # this replaces" — yet the basis went only to logger.debug here, so an
        # operator seeing `$0.00/$0.71` had no way to learn that 0.71 is the
        # p95 of their own 98 recorded sessions times a 3x headroom. It read
        # as an arbitrary constant.
        #
        # Exported rather than recomputed daemon-side ON PURPOSE: the basis
        # belongs to the DECISION that set this ceiling. Re-deriving it later
        # would sample a different set of sessions and could explain the
        # number with evidence that did not produce it.
        env.setdefault("OUROBOROS_BATTLE_COST_CAP_BASIS", _cap_basis or "")
        logger.debug("[ov] session ceiling %s — %s",
                     f"${_cap}" if _cap else "(not asserted)", _cap_basis)
        env.setdefault("OUROBOROS_BATTLE_IDLE_TIMEOUT", os.environ.get(
            "JARVIS_OV_DAEMON_IDLE_TIMEOUT_S", "86400",
        ))
        with open(log, "ab") as sink:
            proc = spawner(
                daemon_argv(),
                cwd=str(repo_root()),
                stdout=sink,
                stderr=sink,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        return proc if getattr(proc, "pid", None) is not None else None
    except Exception:  # noqa: BLE001 — ignition must never raise at the caller
        # exc_info is load-bearing, not decoration. Returning a bare None here
        # turns EVERY failure -- a missing binary, an unwritable log, a typo in
        # this function -- into the single operator-facing sentence "ignition
        # failed", which names no cause and cannot be acted on. That is how a
        # NameError on the line above survived ten days and every CI run: the
        # only symptom was a cockpit that would not start.
        logger.warning(
            "[ov] ignition failed before the daemon was spawned — see %s",
            daemon_log_path(), exc_info=True,
        )
        return None


def _backoff_min_s() -> float:
    try:
        raw = os.environ.get("JARVIS_OV_PROBE_BACKOFF_MIN_S", "")
        return float(raw) if raw else 0.1
    except (TypeError, ValueError):
        return 0.1


def _backoff_max_s() -> float:
    try:
        raw = os.environ.get("JARVIS_OV_PROBE_BACKOFF_MAX_S", "")
        return float(raw) if raw else 2.0
    except (TypeError, ValueError):
        return 2.0


async def await_socket(
    path: Path,
    *,
    on_tick: Optional[Callable[[float], None]] = None,
    deadline_s: Optional[float] = None,
    child_poll: Optional[Callable[[], Optional[int]]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    record_boot: bool = False,
) -> bool:
    """Wait (bounded) for the daemon's bridge to genuinely SERVE — each
    re-probe is the deep application handshake (``probe_socket(deep=True)``),
    never mere socket existence or a kernel-backlog accept, so ``True``
    here can never be a weaker claim than what attach itself consumes.

    Polling is a jittered exponential backoff (full jitter: each sleep is
    ``uniform(min, delay)`` with ``delay`` doubling to an env-tunable
    ceiling) — resource-frugal against a boot-starved daemon and never a
    thundering probe train. The PROBE BOUND escalates with the backoff
    (from the env default up to 3s): a live organism whose loop is lagged
    past the quick bound is still recognized instead of being waited on
    forever (the starved-organism misclassification class).

    ``child_poll`` (Popen.poll of a just-ignited daemon) makes death
    observable: the moment the child exits without the socket serving,
    the wait stops after ONE final escalated probe — never a blind
    full-deadline vigil over a corpse (the 117s-wait class; a
    single-flight rejection exits within ~2s of ignition).

    The wait ends on SILENCE, not on elapsed time. Measured 2026-09-05: a
    cold boot was still wiring the cockpit at 114s against a fixed 120s
    window, and the client was about to report "the organism did not come
    up" about an organism that came up fine. Raising the constant only
    moves the cliff, and the estimator cannot learn its way out — a boot
    slow enough to be abandoned records no duration, so the history keeps
    only the fast ones and keeps predicting them. The signal that
    distinguishes a slow boot from a dead one is not the clock, it is
    whether the daemon is still writing: `stall_s` of no growth in its log
    AND no live socket ends the wait, while growth renews it, up to a
    `ceiling_s` that no amount of progress can move.

    ``on_tick(elapsed)`` drives the waking breadcrumb. NEVER raises."""
    stall = deadline_s if deadline_s is not None else _boot_wait_s()
    hard_deadline = _boot_ceiling_s(stall)
    start = time.monotonic()
    last_advance = start
    mark = _boot_log_mark()
    delay = _backoff_min_s()
    ceiling = max(_backoff_max_s(), delay)
    try:
        while True:
            now = time.monotonic()
            if (now - start) >= hard_deadline:
                break
            moved = _boot_log_mark()
            if moved >= 0 and moved != mark:
                # It is still working. The silence clock restarts; the
                # ceiling above does not.
                #
                # ANY change, not just growth: a fresh boot may truncate or
                # rotate the previous session's log, and a shrink read as
                # "no progress" would strand a starting daemon behind the
                # byte count of the run before it.
                mark = moved
                last_advance = now
            if (now - last_advance) >= stall:
                break
            bound = min(3.0, max(_probe_timeout_s(), delay))
            if await probe_socket(path, timeout=bound, deep=True) == "live":
                # Closed HERE rather than at each of the four call sites —
                # one seam, so a future wait branch cannot forget to clear the
                # in-place line or to record the measurement.
                if on_tick is not None:
                    _finish_tick(time.monotonic() - start, live=True,
                                 on_progress=on_progress,
                                 record=record_boot)
                return True
            if child_poll is not None:
                try:
                    if child_poll() is not None:
                        # The ignited daemon is DEAD. One last escalated
                        # probe (an incumbent may serve this same path),
                        # then stop — the caller reads the exit code.
                        _dead_live = await probe_socket(
                            path, timeout=3.0, deep=True,
                        ) == "live"
                        if on_tick is not None:
                            _finish_tick(time.monotonic() - start,
                                         live=_dead_live,
                                         on_progress=on_progress,
                                         record=record_boot)
                        return _dead_live
                except Exception:  # noqa: BLE001
                    pass
            if on_tick is not None:
                try:
                    on_tick(time.monotonic() - start)
                except Exception:  # noqa: BLE001
                    pass
            # Sleep no further than the SOONER of the two bounds, so a
            # long backoff cannot overshoot either one.
            now = time.monotonic()
            remaining = min(hard_deadline - (now - start),
                            stall - (now - last_advance))
            if remaining <= 0:
                break
            sleep_for = min(random.uniform(_backoff_min_s(), delay), remaining)
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, ceiling)
        if on_tick is not None:
            # Deadline reached without a live socket. The line is cleared, but
            # NOTHING is recorded: an abandoned boot has no duration.
            _finish_tick(time.monotonic() - start, live=False,
                         on_progress=on_progress)
        return False
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        return False


def _live_incumbent() -> Optional[int]:
    """PID of a live, fresh single-flight lock holder — the shared
    reader in ``singleton_lock`` (one authority, zero disagreement
    with the reaper/preflight). NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.singleton_lock import (
            live_incumbent_pid,
        )
        return live_incumbent_pid(repo_root(), exclude_pid=os.getpid())
    except Exception:  # noqa: BLE001
        return None


def _visible_len(text: str) -> int:
    """How many COLUMNS the terminal will give *text*.

    ``len()`` counts code points, which is the wrong unit for a line that
    carries box-drawing and block glyphs: a string of 85 code points can
    occupy more or fewer than 85 columns, and a redraw that guesses wrong
    either wraps (tearing) or under-erases (debris).

    ``wcwidth`` is the correct answer and is optional here; without it the
    code-point count is a serviceable approximation for this line, whose
    glyphs are overwhelmingly single-width. Never raises.
    """
    try:
        from wcwidth import wcswidth  # noqa: PLC0415 — optional dependency

        measured = wcswidth(text)
        if isinstance(measured, int) and measured >= 0:
            return measured
    except Exception:  # noqa: BLE001 — absent or confused: fall back
        pass
    return len(text)


def _terminal_columns(stream: Any) -> int:
    """The terminal's current width, or 0 when it cannot be known.

    Asked fresh on every redraw rather than cached, so a window resized
    mid-boot re-fits on the next frame instead of tearing until restart.
    Returns 0 rather than a guess: a wrong width is worse than no clamp,
    because it truncates content that would have fitted.
    """
    try:
        import os as _os  # noqa: PLC0415

        fd = getattr(stream, "fileno", None)
        if callable(fd):
            return int(_os.get_terminal_size(fd()).columns)
    except Exception:  # noqa: BLE001 — not a tty, or no ioctl
        pass
    try:
        import shutil as _shutil  # noqa: PLC0415

        # COLUMNS honours an operator override; fallback=(0, 0) keeps the
        # "unknown" answer distinguishable from a real 80.
        return int(_shutil.get_terminal_size(fallback=(0, 0)).columns)
    except Exception:  # noqa: BLE001
        return 0


def _fit_to_width(text: str, columns: int) -> str:
    """Trim *text* so it cannot wrap, keeping the LEFT.

    The left is where the meaning is: the bar, the percentage and the stage
    label answer "is it moving"; the trailing detail is elaboration. An
    ellipsis marks the cut so a truncated line never reads as a complete
    one -- and it is only added when there is room for it to mean anything.
    """
    if columns <= 0 or _visible_len(text) <= columns:
        return text
    if columns <= 1:
        return text[:columns]
    ell = "…"
    budget = columns - _visible_len(ell)
    out = []
    used = 0
    for ch in text:
        w = _visible_len(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out) + ell


def _mk_tick(say: Callable[[str], None],
             on_progress: Optional[Callable[[str], None]] = None,
             ) -> Callable[[float], None]:
    """The waking indicator — one builder for every wait branch.

    Replaces a fresh `organism waking · Ns` line every five seconds, which
    said only that time was passing and read as a stall by virtue of
    repeating. The line now carries the STAGE the boot has actually reached
    (parsed from the daemon's own log, which is already being written and
    whose path this module already knows) and a completion estimate drawn
    from this machine's measured boot history.

    IN-PLACE ON A TTY, APPEND OTHERWISE. A carriage return redraws one line
    for a human; written into a pipe or a log file it produces a single
    unreadable smear, so a non-TTY keeps the old cadence and the old shape.
    The wait must not become less legible in the transcript to become prettier
    on screen.
    """
    last = [-1e9]
    started = [time.monotonic()]
    # RENDER THROUGH WHOEVER OWNS THE SCREEN.
    #
    # During ignition a Rich `Live` draws the animated crest and owns the
    # terminal. Writing a carriage-return line to stdout underneath it is
    # erased by the very next frame — six screenshots taken across six
    # seconds showed the bar absent from every one, which the operator
    # correctly described as flickering. `on_progress` hands the line to that
    # owner instead, as a VALUE it renders, and the Live picks it up on its
    # next repaint. Direct terminal writing is reserved for the case where
    # nothing else owns the screen.
    #
    # `real_stdout_isatty`, NOT `sys.stdout.isatty()`.
    #
    # prompt_toolkit's `patch_stdout` replaces sys.stdout with a non-TTY
    # proxy, so the naive check returns False on a real terminal and the
    # indicator falls back to APPENDING a line per tick — which is precisely
    # the duplicate-line behaviour this was written to remove. The codebase
    # already solved this once (the live status line never surfaced for the
    # same reason) and the helper exists for it; using it is the fix, and
    # writing a second TTY check would be the bug a third time.
    try:
        from backend.core.ouroboros.battle_test.presentation_restraint import (
            real_stdout_isatty,
        )
        interactive = bool(real_stdout_isatty())
    except Exception:  # noqa: BLE001
        try:
            interactive = bool(sys.stdout.isatty())
        except Exception:  # noqa: BLE001
            interactive = False
    try:
        from backend.core.ouroboros.cli import boot_progress as _bp
        enabled = _bp.boot_progress_enabled()
        prog = _bp.make_progress(str(daemon_log_path())) if enabled else None
    except Exception:  # noqa: BLE001
        _bp, enabled, prog = None, False, None

    # A TTY can be redrawn ~4x/s without flicker; a transcript should not gain
    # a line that often, so the two cadences differ on purpose. A screen owner
    # takes the fast cadence too — it is repainting anyway.
    live_owner = on_progress is not None
    cadence = 0.25 if ((interactive or live_owner) and enabled) else 5.0

    def _tick(elapsed: float) -> None:
        if elapsed - last[0] < cadence:
            return
        last[0] = elapsed
        if prog is None or _bp is None:
            say(f"⎿ organism waking · {int(elapsed)}s")
            return
        try:
            prog.observe_log(
                _bp.read_log_tail(prog.log_path, since=prog.log_origin),
                now=time.monotonic() - started[0])
            body = prog.render(elapsed)
        except Exception:  # noqa: BLE001
            body = f"waking · {int(elapsed)}s"
        line = f"⎿ {body}"
        if live_owner:
            try:
                on_progress(line)     # a value for the owner to render
                return
            except Exception:  # noqa: BLE001
                pass                  # fall through to the plain paths
        if not interactive:
            say(line)
            return
        try:
            # WRITE TO THE STREAM WE TESTED. `real_stdout_isatty` inspects
            # `sys.__stdout__`; drawing on `sys.stdout` would test one stream
            # and paint another, and under patch_stdout that proxy buffers by
            # line — which turns a carriage-return redraw back into the very
            # append this replaces.
            out = sys.__stdout__ or sys.stdout
            # A CARRIAGE RETURN CANNOT UNDO A WRAP.
            #
            # This line runs 85 columns at full extent ("[bar] 75%  session
            # open  108s  +98s over  waiting on cockpit wired"). On any
            # terminal narrower than that it WRAPS, and `\r` then returns to
            # the start of the last visual row only — so each redraw paints
            # over half of itself and leaves the other half standing. Four
            # times a second that reads as flicker, which is exactly what it
            # was: not an animation, a line fighting its own wrap.
            #
            # Clamped to the CURRENT width on every tick rather than once at
            # construction, so resizing the terminal mid-boot re-fits instead
            # of tearing until restart. `_visible_len` measures what the
            # terminal will actually allot, not how many code points there
            # are — the bar glyphs and the ⎿ are not all one column wide.
            cols = _terminal_columns(out)
            body = _fit_to_width(line, cols) if cols else line
            # `\033[K` erases from the cursor to end of line: the terminal's
            # own primitive for exactly this, and it replaces the manual
            # space-padding plus the `width[0]` it had to carry. One escape,
            # no state, and correct when the previous frame wrapped.
            out.write("\r" + body + "\033[K")
            out.flush()
        except Exception:  # noqa: BLE001
            say(line)

    return _tick


def _finish_tick(elapsed: float, *, live: bool,
                 on_progress: Optional[Callable[[str], None]] = None,
                 record: bool = False) -> None:
    """Close the progress surface, and record ONLY a genuine cold boot.

    `record` defaults False, and that default is the fix for a real defect.
    `await_socket` has four callers and three of them are "an organism is
    already up — confirm and attach", which completes in ~0.1s. Recording
    those as boot durations poisoned the estimator: the observed history read
    [0.10, 0.22, 0.10, 0.22, 0.10, 0.23, 72.05, 0.10, 0.22] with a median of
    0.22s, so any elapsed time instantly exceeded the expected total and the
    bar rendered 97% on its first frame — complete before the work began.

    Attach latency and boot duration are different quantities. Only the
    caller that actually spawned a daemon knows which it just measured, so
    only that caller asks for the sample to be kept.
    """
    # Clear the owner's gauge first: it renders inside the Live, so wiping
    # the raw terminal would not touch it.
    if on_progress is not None:
        try:
            on_progress("")
        except Exception:  # noqa: BLE001
            pass
    try:
        from backend.core.ouroboros.battle_test.presentation_restraint import (
            real_stdout_isatty as _rti,
        )
        _tty = _rti()
    except Exception:  # noqa: BLE001
        _tty = False
    try:
        if _tty:
            out = sys.__stdout__ or sys.stdout
            out.write("\r" + " " * 120 + "\r")
            out.flush()
    except Exception:  # noqa: BLE001
        pass
    if not (live and record):
        return
    try:
        from backend.core.ouroboros.cli import boot_progress as _bp
        _bp.record_boot_duration(elapsed)
    except Exception:  # noqa: BLE001
        pass


def _ignition_retry_budget_s() -> float:
    """How long to wait out a transient single-flight refusal.

    Sized to a SHUTDOWN, not a boot: the window being waited out is an
    organism draining after SIGTERM. Bounded so a genuinely wedged incumbent
    still returns the operator to a prompt."""
    try:
        raw = os.environ.get("JARVIS_OV_IGNITION_RETRY_S", "").strip()
        return max(0.0, float(raw)) if raw else 20.0
    except (TypeError, ValueError):
        return 20.0


async def _await_ignition_window(
    path: Any, *, say: Any,
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """Wait out a transient lock refusal. True iff an organism is now serving.

    Races the two ways the window closes — the incumbent starts serving, or it
    releases the lock and a fresh ignition succeeds — because waiting on only
    one of them would hang on the other. NEVER raises."""
    budget = _ignition_retry_budget_s()
    if budget <= 0.0:
        return False
    say("⎿ another organism is still shutting down — waiting for the lock")
    deadline = time.monotonic() + budget
    attempt = 0
    while time.monotonic() < deadline:
        try:
            # (a) the incumbent came up after all — nothing was ever wrong.
            #
            # `== "live"`, NOT truthiness: probe_socket returns a CLASSIFI-
            # CATION string, so `if await probe_socket(...)` is true for
            # "absent" and "stale" too — it would report a corpse as live.
            # Caught by the existing starved-organism suite.
            if await probe_socket(path, deep=True) == "live":
                say("⏺ organism live — attaching")
                return True
            # (b) it let go: the lock is free, so ignite again.
            if _live_incumbent() is None:
                proc = spawn_daemon()
                if proc is not None and await await_socket(
                    path, on_tick=_mk_tick(say, on_progress),
                    on_progress=on_progress,
                    child_poll=getattr(proc, "poll", None),
                ):
                    say("⏺ organism live — attaching")
                    return True
        except Exception:  # noqa: BLE001 — a failed probe is just another tick
            pass
        attempt += 1
        # Full jitter, same discipline as the audio reflex: several cockpits
        # started together must not all re-ignite on the same instant.
        delay = min(2.0, 0.25 * (2 ** min(attempt, 4)))
        await asyncio.sleep(random.uniform(0.0, delay))
    return False


async def ensure_daemon(
    *,
    on_status: Optional[Callable[[str], None]] = None,
    spawner: Callable[..., Any] = subprocess.Popen,
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """The full zero-trust route: probe → (clean ghost) → (cold boot)
    → wait-for-live. True when a live daemon is reachable. NEVER
    raises; NEVER hangs past the boot deadline; NEVER shows a
    traceback."""
    def _say(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:  # noqa: BLE001
                pass

    try:
        from backend.core.ouroboros.battle_test.cockpit_attach import (
            attach_socket_path,
        )
        path = attach_socket_path()
        state = await probe_socket(path, deep=True)
        if state == "live":
            return True
        if state == "booting":
            # "booting" is INFERRED from a socket that exists and does not
            # answer — which is exactly what a daemon killed mid-life leaves
            # behind. The two are indistinguishable from the socket alone, so
            # the claim is corroborated against a LIVE process before it is
            # believed.
            #
            # Without this, an ungraceful death wedges `ov` permanently: the
            # client waits for a corpse, and the stale-socket reaper lives
            # inside the harness the client is refusing to start. The cleanup
            # is trapped inside the thing that cannot boot, and only a human
            # deleting a file breaks the cycle — which no operator would know
            # to do. Observed 2026-07-27 after a routine `kill`.
            #
            # `_live_incumbent` is the SAME reader the reaper and preflight
            # use, so there is one authority on "is anyone home" and no way
            # for two components to disagree about it.
            if _live_incumbent() is None:
                _say(
                    "⎿ a socket is present but nobody owns it — "
                    "treating as a ghost and igniting",
                )
                state = "stale"          # fall through to the reap+spawn path
            else:
                # A daemon is home but boot-starved — its socket must NOT be
                # cleaned and no second ignition raced. Wait for it to serve.
                _say("⏺ organism already waking — waiting for it to serve")
                if await await_socket(path, on_tick=_mk_tick(_say, on_progress),
                                      on_progress=on_progress):
                    _say("⏺ organism live — attaching")
                    return True
                _say(
                    "⚠ the waking organism never served — "
                    "tail " + str(daemon_log_path()),
                )
                return False
        if state == "stale":
            # NEVER unlink while a live single-flight incumbent exists:
            # the bridge binds ONCE at boot, so cleaning a live-but-
            # refusing organism's socket makes it permanently
            # unattachable (the 2026-07-23 class). Only a dead/absent
            # holder proves a true ghost.
            incumbent = _live_incumbent()
            if incumbent is not None:
                _say(
                    f"⏺ organism (PID {incumbent}) is alive but not "
                    "serving — waiting, never cleaning a live socket"
                )
                if await await_socket(path, on_tick=_mk_tick(_say, on_progress),
                                      on_progress=on_progress):
                    _say("⏺ organism live — attaching")
                    return True
                _say(
                    f"⚠ organism PID {incumbent} is alive but its "
                    "control plane never served — its loop may be "
                    "starved; tail " + str(daemon_log_path()),
                )
                return False
            _say("⎿ ghost socket from a dead organism — cleaning")
            clean_stale_socket(path)
        _say("⏺ no organism awake — igniting one in the background")
        proc = spawn_daemon(spawner=spawner)
        if proc is None:
            _say("⚠ ignition failed — see " + str(daemon_log_path()))
            return False

        if await await_socket(
            path, on_tick=_mk_tick(_say, on_progress),
            child_poll=getattr(proc, "poll", None),
            on_progress=on_progress,
            record_boot=True,   # this branch SPAWNED — it is a real boot
        ):
            _say("⏺ organism live — attaching")
            return True
        rc = None
        try:
            rc = proc.poll()
        except Exception:  # noqa: BLE001
            pass
        if rc == EXIT_MODEL_PIN_UNAVAILABLE:
            # A CONFIGURATION REFUSAL, not a failure to start. The daemon
            # declined because the pinned model is not served, and it
            # already printed the full alert — pin, source, what the node
            # offers, and the fix. Repeating "the organism did not come
            # up" over that would describe a crash that did not happen and
            # send the operator to the wrong log.
            #
            # Retrying is pointless here, unlike EX_TEMPFAIL below: nothing
            # resolves on its own. So this returns immediately and names
            # the log that holds the alert.
            _say(
                "⚠ the organism declined to start: the pinned model is not "
                "served. The full alert is in " + str(daemon_log_path())
            )
            return False
        if rc == 75:                      # EX_TEMPFAIL — single-flight
            # A REFUSAL IS USUALLY TRANSIENT, so retrying is the cockpit's job
            # rather than the operator's.
            #
            # The message used to end at "retry shortly", which reads as an
            # instruction and is in fact a description of a race the cockpit
            # can wait out itself. `pkill` sends SIGTERM and the organism
            # drains for several seconds; throughout that drain the kernel
            # still holds its flock, so an ignition fired in that window is
            # refused for a condition that resolves on its own. The operator
            # then sees a hard failure for something nobody needed to fix.
            #
            # Two outcomes are watched together, because either ends the wait
            # legitimately: the incumbent finishes booting and starts SERVING
            # (attach to it — there was never anything wrong), or it finishes
            # DYING and releases the lock (ignite again).
            if await _await_ignition_window(path, say=_say,
                                            on_progress=on_progress):
                return True
            incumbent = _live_incumbent()
            who = f"PID {incumbent}" if incumbent else "another process"
            _say(
                f"⚠ ignition refused — {who} holds the single-flight "
                "organism lock and never served. Its loop may be starved; "
                "tail " + str(daemon_log_path()),
            )
            return False
        if rc is not None:
            _say(
                f"⚠ ignition died (exit {rc}) before serving — "
                "tail " + str(daemon_log_path()),
            )
            return False
        _say(
            "⚠ boot did not surface within the wait window — "
            "tail " + str(daemon_log_path()),
        )
        return False
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Resident Organism — launchd User Agent
# ---------------------------------------------------------------------------


def agent_plist_path(agents_dir: Optional[Path] = None) -> Path:
    base = agents_dir or (Path.home() / "Library" / "LaunchAgents")
    return base / f"{AGENT_LABEL}.plist"


def build_agent_plist() -> dict:
    """The launchd definition — every path resolved at INSTALL time
    (interpreter, repo, logs); nothing machine-hardcoded."""
    log_dir = repo_root() / ".jarvis" / "logs"
    return {
        "Label": AGENT_LABEL,
        "ProgramArguments": [str(a) for a in daemon_argv()],
        "WorkingDirectory": str(repo_root()),
        "RunAtLoad": True,
        # KeepAlive restarts a crashed organism; single-flight makes a
        # racing duplicate exit 75 harmlessly.
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "ov-daemon.out.log"),
        "StandardErrorPath": str(log_dir / "ov-daemon.err.log"),
        "EnvironmentVariables": {
            # Same single definition the detached spawn uses. An unresolvable
            # ceiling yields "" so the agent carries no assertion and the
            # harness default applies, rather than a competing literal.
            "OUROBOROS_BATTLE_COST_CAP": _cockpit_cost_cap()[0] or "",
            "OUROBOROS_BATTLE_IDLE_TIMEOUT": os.environ.get(
                "JARVIS_OV_DAEMON_IDLE_TIMEOUT_S", "86400",
            ),
        },
    }


def install_agent(
    *,
    agents_dir: Optional[Path] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Write the plist + bootstrap it into the user's launchd domain.
    Returns a one-line operator message. NEVER raises."""
    try:
        path = agent_plist_path(agents_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = repo_root() / ".jarvis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Janitor the launchd sinks at (re)install so an accumulated
        # history never rides into the resident era unbounded.
        rollover_daemon_log(log_dir / "ov-daemon.out.log")
        rollover_daemon_log(log_dir / "ov-daemon.err.log")
        with open(path, "wb") as fh:
            plistlib.dump(build_agent_plist(), fh)
        try:
            uid = os.getuid()
            runner(
                ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
                capture_output=True, timeout=10,
            )
        except Exception:  # noqa: BLE001
            return (
                f"⏺ resident agent written to {path} — load it with: "
                f"launchctl bootstrap gui/$UID {path}"
            )
        return f"⏺ resident organism installed ({AGENT_LABEL}) — {path}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠ install failed: {exc}"


def uninstall_agent(
    *,
    agents_dir: Optional[Path] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Boot the agent out of launchd + remove the plist. NEVER raises."""
    try:
        path = agent_plist_path(agents_dir)
        try:
            uid = os.getuid()
            runner(
                ["launchctl", "bootout", f"gui/{uid}/{AGENT_LABEL}"],
                capture_output=True, timeout=10,
            )
        except Exception:  # noqa: BLE001
            pass
        existed = path.exists()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return (
            f"⏺ resident organism uninstalled ({AGENT_LABEL})"
            if existed else "⎿ no resident agent was installed"
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠ uninstall failed: {exc}"


__all__ = [
    "AGENT_LABEL",
    "agent_plist_path",
    "await_socket",
    "build_agent_plist",
    "clean_stale_socket",
    "daemon_argv",
    "daemon_log_path",
    "ensure_daemon",
    "install_agent",
    "probe_socket",
    "probe_tcp",
    "probe_http",
    "repo_root",
    "spawn_daemon",
    "thin_client_enabled",
    "uninstall_agent",
]
