"""Tier 1 #3 — Cross-process JSONL append helper.

Closes §28.5.1 v9 brutal review's concrete data-loss race:

  ``auto_action_router.py:1110-1113`` and ``adaptation/ledger.py:648``
  use ``path.open("a")`` with ``threading.Lock()`` only. POSIX
  append-mode is line-atomic **within a single process** but **NOT
  across processes**. Two ``ouroboros_battle_test.py`` processes
  writing the same ``.jsonl`` concurrently can interleave partial
  writes — the second write can overwrite the tail of the first
  before the newline lands. ``ApprovalStore`` already uses
  ``fcntl.flock`` correctly elsewhere; the action ledgers don't.

This module is the single source of truth for cross-process JSONL
append safety. Three ledgers wire to it:

  * ``auto_action_router.AutoActionProposalLedger.append``
  * ``adaptation.ledger.AdaptationLedger.append``
  * ``invariant_drift_store.InvariantDriftStore.append_history``
  * ``invariant_drift_store.InvariantDriftStore.append_audit``

Design pillars:

  * **Asynchronous** — flock is sync-blocking but the cost is
    bounded (microseconds for a single append; the lock scope is
    open-write-flush-close, never wraps long operations). Pattern
    matches ``ApprovalStore.decide`` exactly.

  * **Dynamic** — POSIX uses ``fcntl.flock``; Windows falls through
    to a documented degraded mode (advisory threading.Lock only —
    a future ``msvcrt.locking`` wiring can land additively without
    touching call sites). Production target is POSIX.

  * **Adaptive** — degrades gracefully when ``fcntl`` is missing
    (extreme edge — embedded environments / Windows). Returns False
    on lock-acquire failure rather than raising.

  * **Intelligent** — distinguishes (a) lock-acquire failure
    (concurrent writer holds the lock past timeout, returns False),
    (b) write failure (OSError mid-write, returns False), (c)
    success (returns True). Caller can stat() the result and
    surface accordingly.

  * **Robust** — never raises. Lock is released on every exit path
    including exceptions. ``finally`` block guarantees release +
    fd close even if ``fcntl.flock`` itself raises.

  * **No hardcoding** — lock acquisition timeout is env-tunable;
    default 5.0s (long enough for any reasonable single append on
    a healthy disk; short enough that a deadlocked writer doesn't
    hang the entire ledger system).

Authority invariants (AST-pinned by companion tests):

  * Imports stdlib ONLY (``fcntl`` / ``msvcrt`` are stdlib-conditional;
    everything else is core stdlib).
  * NEVER imports any governance module — this is a pure-stdlib
    primitive consumed by ledgers; reverse-coupling would create
    a cycle. Exactly TWO audited, cycle-free, FUNCTION-LOCAL
    exceptions (both AST-allowlisted in
    ``tests/governance/test_cross_process_jsonl.py``):

      1. ``workspace_resolver`` — durable-path resolution; itself
         stdlib-only (a companion test self-seals its purity).
      2. ``cooperative_fs_io`` — the F8 async offload substrate,
         imported ONLY inside the async append wrappers (Slice 3
         Task 2) so the flock poll loop runs off the asyncio loop.
         ``cooperative_fs_io`` never imports this module back (no
         cycle), and the import is deferred to call time, so the
         module stays pure-stdlib at import scope.
  * Never raises out of any public method.
"""
from __future__ import annotations

import asyncio
import errno
import logging
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterable, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


CROSS_PROCESS_JSONL_SCHEMA_VERSION: str = "cross_process_jsonl.1"


# ---------------------------------------------------------------------------
# fcntl detection — degrade gracefully on Windows / embedded environments
# ---------------------------------------------------------------------------


try:
    import fcntl as _fcntl  # type: ignore[import]
    _HAS_FCNTL: bool = True
except ImportError:
    _fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


def fcntl_available() -> bool:
    """True iff fcntl is importable (POSIX). Public so callers can
    decide whether the cross-process guarantee is real or degraded
    to in-process-only on this platform."""
    return _HAS_FCNTL


# ---------------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------------


_DEFAULT_LOCK_TIMEOUT_S: float = 5.0
_LOCK_TIMEOUT_FLOOR_S: float = 0.1


def lock_timeout_s() -> float:
    """``JARVIS_CROSS_PROCESS_LOCK_TIMEOUT_S`` (default 5.0s, floor
    0.1s).

    Maximum wall-clock seconds to wait for an exclusive flock. Long
    enough for any reasonable single append on a healthy disk;
    short enough that a deadlocked writer doesn't hang the entire
    ledger system."""
    raw = os.environ.get(
        "JARVIS_CROSS_PROCESS_LOCK_TIMEOUT_S", "",
    ).strip()
    if not raw:
        return _DEFAULT_LOCK_TIMEOUT_S
    try:
        return max(_LOCK_TIMEOUT_FLOOR_S, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_LOCK_TIMEOUT_S


# --- Stale lock age knob (cascading state vector fix 2026-05-01) ---

_DEFAULT_STALE_LOCK_AGE_S: float = 300.0  # 5 minutes
_STALE_LOCK_AGE_FLOOR_S: float = 10.0
_STALE_LOCK_AGE_CEILING_S: float = 86400.0  # 24 hours


def stale_lock_age_s() -> float:
    """``JARVIS_STALE_LOCK_AGE_S`` (default 300s = 5 min, floor 10s,
    ceiling 86400s = 24h).

    When a ``.lock`` file's mtime is older than this threshold at
    the time of a successful flock acquire, a structured WARNING log
    is emitted: ``[CrossProcessJSONL] stale_lock_detected``.

    The log signal lets operators detect "something was SIGKILL'd
    mid-critical-section" and investigate the corresponding data
    file for possible corruption (partial write). The lock mechanism
    itself is functionally correct — the kernel releases flock on
    process death — so this is purely diagnostic. NEVER raises."""
    raw = os.environ.get(
        "JARVIS_STALE_LOCK_AGE_S", "",
    ).strip()
    if not raw:
        return _DEFAULT_STALE_LOCK_AGE_S
    try:
        val = float(raw)
        return max(_STALE_LOCK_AGE_FLOOR_S, min(
            _STALE_LOCK_AGE_CEILING_S, val,
        ))
    except (TypeError, ValueError):
        return _DEFAULT_STALE_LOCK_AGE_S


# ---------------------------------------------------------------------------
# In-process lock map — keyed by lock-file absolute path
# ---------------------------------------------------------------------------
#
# The threading.Lock layer is NOT redundant with fcntl.flock — it
# handles fast in-process serialization (avoiding a lock-file open()
# + flock syscall per append from the same process). The fcntl layer
# handles the cross-process serialization. Both compose: in-process
# threads serialize via threading.Lock; processes serialize via
# fcntl.flock; one writer at a time globally.


_in_process_locks: dict = {}
_in_process_locks_guard = threading.Lock()


def _get_inprocess_lock(lock_path: Path) -> threading.Lock:
    """Return (or create) the threading.Lock for a given lock-path.
    Per-path locks so different ledgers don't serialize against each
    other unnecessarily."""
    key = str(lock_path.resolve())
    with _in_process_locks_guard:
        existing = _in_process_locks.get(key)
        if existing is not None:
            return existing
        new_lock = threading.Lock()
        _in_process_locks[key] = new_lock
        return new_lock


def _reset_inprocess_locks_for_tests() -> None:
    """Test isolation helper. Drops the in-process lock map so each
    test starts fresh (matters for tests that assert lock identity)."""
    with _in_process_locks_guard:
        _in_process_locks.clear()


# ---------------------------------------------------------------------------
# Lock acquisition — context manager wrapping flock + threading.Lock
# ---------------------------------------------------------------------------


@contextmanager
def _acquire_cross_process_lock(
    lock_path: Path,
    *,
    timeout_s: Optional[float] = None,
) -> Iterator[bool]:
    """Acquire exclusive cross-process lock on ``lock_path``. Yields
    True on success, False on timeout / fcntl-unavailable / OSError.
    Always releases on exit. NEVER raises."""
    effective_timeout = (
        timeout_s if timeout_s is not None and timeout_s > 0
        else lock_timeout_s()
    )
    inprocess = _get_inprocess_lock(lock_path)
    # In-process serialize first (cheap)
    if not inprocess.acquire(timeout=effective_timeout):
        yield False
        return

    try:
        # Cross-process serialize via fcntl.flock with poll-loop
        # (POSIX flock has no native timeout; we use LOCK_NB +
        # exponential-backoff poll until the deadline).
        if not _HAS_FCNTL:
            # Degraded mode: in-process lock only. Document this
            # in stats; caller can detect via fcntl_available().
            yield True
            return

        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            yield False
            return

        try:
            lock_fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_RDWR,
                0o644,
            )
        except OSError as exc:
            logger.debug(
                "[CrossProcessJSONL] lock-file open failed: %s", exc,
            )
            yield False
            return

        deadline = time.monotonic() + effective_timeout
        backoff = 0.005
        max_backoff = 0.25
        acquired = False
        # Cascading state vector fix (2026-05-01): stat lock file
        # mtime BEFORE acquiring the flock. If the file pre-existed
        # with a stale mtime, a prior process may have been SIGKILL'd
        # mid-critical-section. The lock is functionally correct
        # (kernel released it), but the data file may be corrupt.
        lock_mtime_before: float = 0.0
        try:
            st = os.fstat(lock_fd)
            lock_mtime_before = st.st_mtime
        except OSError:
            pass
        try:
            while True:
                try:
                    _fcntl.flock(  # type: ignore[union-attr]
                        lock_fd,
                        _fcntl.LOCK_EX | _fcntl.LOCK_NB,  # type: ignore[union-attr]
                    )
                    acquired = True
                    break
                except (BlockingIOError, OSError) as exc:
                    if (
                        isinstance(exc, OSError)
                        and exc.errno not in (
                            errno.EWOULDBLOCK,
                            errno.EAGAIN,
                            errno.EACCES,
                        )
                    ):
                        # Non-contention OSError — bail
                        logger.debug(
                            "[CrossProcessJSONL] flock raised "
                            "unexpected OSError: %s", exc,
                        )
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 1.5)

            if not acquired:
                yield False
                return

            # Stale lock detection: if the file existed before we
            # opened it AND its mtime is older than the staleness
            # threshold, log a structured WARNING. The data file
            # associated with this lock may have been left in a
            # partially-written state.
            if lock_mtime_before > 0:
                age = time.time() - lock_mtime_before
                threshold = stale_lock_age_s()
                if age > threshold:
                    logger.warning(
                        "[CrossProcessJSONL] stale_lock_detected "
                        "path=%s age_s=%.1f threshold_s=%.1f",
                        lock_path, age, threshold,
                    )

            # Update mtime so the next acquirer can distinguish
            # "file from a clean session" (mtime recent) from
            # "file from a SIGKILL'd process" (mtime stale).
            try:
                os.utime(lock_fd)
            except OSError:
                pass  # best-effort; not critical

            yield True
        finally:
            if acquired:
                try:
                    _fcntl.flock(  # type: ignore[union-attr]
                        lock_fd,
                        _fcntl.LOCK_UN,  # type: ignore[union-attr]
                    )
                except OSError as exc:
                    logger.debug(
                        "[CrossProcessJSONL] flock UN raised: %s",
                        exc,
                    )
            try:
                os.close(lock_fd)
            except OSError:
                pass
    finally:
        try:
            inprocess.release()
        except RuntimeError:
            # Already released — should not happen but defensive
            pass


# ---------------------------------------------------------------------------
# Public append helpers
# ---------------------------------------------------------------------------


def flock_append_line(
    path: Path,
    line: str,
    *,
    timeout_s: Optional[float] = None,
) -> bool:
    """Append a single line (with trailing newline) to ``path``,
    serialized cross-process via ``fcntl.flock`` on a sibling
    ``.lock`` file. Returns True on success, False on any failure
    (lock timeout, write error, fcntl unavailable, etc.). NEVER
    raises.

    The line is written exactly as-given plus exactly one trailing
    ``\\n``. Caller is responsible for ensuring ``line`` does not
    already contain newlines (the JSONL contract — one record per
    line)."""
    from backend.core.ouroboros.governance.workspace_resolver import (
        resolve_durable_path,
    )
    path = resolve_durable_path(path)
    # Slice 33 Arc 0 — diagnostic only. v26 saw 12 CrossProcessJSONL
    # WARNs (stale-lock detection); under contention this can block.
    from backend.core.ouroboros.telemetry.loop_sink import (
        sink_sync as _ls_sink_sync,
    )
    with _ls_sink_sync("cross_process_jsonl.flock_append_line"):
        return flock_append_lines(
            path, (line,), timeout_s=timeout_s,
        )


@contextmanager
def flock_critical_section(
    path: Path,
    *,
    timeout_s: Optional[float] = None,
) -> Iterator[bool]:
    """Acquire exclusive cross-process lock on ``path``'s sibling
    ``.lock`` file for a custom critical section (e.g., a ring-
    buffer read-modify-write). Yields True on success, False on
    timeout / failure. Always releases on exit. NEVER raises.

    Use this when a single ``flock_append_line`` call doesn't fit —
    e.g., the InvariantDriftStore history-ring-buffer pattern that
    reads existing lines, appends + trims, then atomic-writes the
    truncated tail. Concurrent processes must not race the read-
    trim-write block.

    Caller is responsible for the actual file I/O inside the block.
    The context manager only provides the lock; if I/O fails
    inside the block, the caller handles it (defensive contract is
    caller-owned)."""
    try:
        from backend.core.ouroboros.governance.workspace_resolver import (
            resolve_durable_path,
        )
        path = resolve_durable_path(path)
        target = Path(path)
        lock_path = target.with_suffix(target.suffix + ".lock")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug(
                "[CrossProcessJSONL] critical-section parent mkdir "
                "failed: %s", exc,
            )
            yield False
            return
        with _acquire_cross_process_lock(
            lock_path, timeout_s=timeout_s,
        ) as acquired:
            yield acquired
    except Exception as exc:  # noqa: BLE001 — last-resort defensive
        logger.debug(
            "[CrossProcessJSONL] flock_critical_section raised: %s",
            exc,
        )
        yield False


def flock_append_lines(
    path: Path,
    lines: Iterable[str],
    *,
    timeout_s: Optional[float] = None,
) -> bool:
    """Append multiple lines atomically (all-or-nothing within the
    flock scope) to ``path``. Each line gets exactly one trailing
    ``\\n``. Returns True iff every line landed; False on any
    failure. NEVER raises.

    All lines write under one flock acquire — concurrent writers
    cannot interleave. Cheaper than calling ``flock_append_line``
    in a loop when batching."""
    try:
        from backend.core.ouroboros.governance.workspace_resolver import (
            resolve_durable_path,
        )
        path = resolve_durable_path(path)
        target = Path(path)
        lock_path = target.with_suffix(target.suffix + ".lock")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug(
                "[CrossProcessJSONL] parent mkdir failed: %s", exc,
            )
            return False

        with _acquire_cross_process_lock(
            lock_path, timeout_s=timeout_s,
        ) as acquired:
            if not acquired:
                logger.debug(
                    "[CrossProcessJSONL] lock acquisition failed "
                    "for %s", target,
                )
                return False
            try:
                with target.open("a", encoding="utf-8") as fh:
                    # HEAL A TORN TAIL BEFORE APPENDING.
                    #
                    # A killed process can leave a line whose payload
                    # landed and whose terminator did not — the recorder's
                    # rows reach 24 KB, far past any size a single write is
                    # atomic at. Appending onto that tail CONCATENATES the
                    # next row into it, so one interruption destroys TWO
                    # rows: the torn one and the good one that follows.
                    # Demonstrated: 3 rows written, 2 lines on disk, 1
                    # parseable.
                    #
                    # Terminating the tail turns that into one lost row,
                    # which the reader already counts and tolerates. It
                    # happens under the same flock as the append, so no
                    # other writer can be mid-line while we look, and the
                    # seek is one read of at most one byte.
                    _terminate_partial_tail(fh, target)
                    for raw_line in lines:
                        if not isinstance(raw_line, str):
                            # Coerce defensively rather than raise.
                            raw_line = str(raw_line)
                        # ONE write, so there is no window in which the
                        # payload is on its way to the file and its
                        # terminator is not yet queued behind it.
                        fh.write(raw_line + "\n")
                    fh.flush()
                return True
            except OSError as exc:
                logger.debug(
                    "[CrossProcessJSONL] append write failed at "
                    "%s: %s", target, exc,
                )
                return False
    except Exception as exc:  # noqa: BLE001 — last-resort defensive
        logger.debug(
            "[CrossProcessJSONL] flock_append_lines raised: %s",
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Async wrapper — async_flock_critical_section
# ---------------------------------------------------------------------------
#
# Rationale: AutoCommitter.commit() runs inside an async event loop
# and orchestrates ``asyncio.create_subprocess_exec`` git calls. Its
# critical section spans `_intent_token_exists()` → git commit →
# `_store_intent_token()` (TOCTOU race per §3.6.2 vector #10). The
# sync flock_critical_section above would block the event loop for
# the full git-commit duration. async_flock_critical_section enters
# / exits the underlying sync context via ``asyncio.to_thread`` so
# the lock acquisition + release execute in worker threads, while
# the caller's async block runs while the lock is held.
#
# Wave 3 hygiene 2026-05-05 — Item 5 (vector #10 closure).


from contextlib import ExitStack as _ExitStack  # noqa: E402


@asynccontextmanager
async def async_flock_critical_section(
    path: Path,
    *,
    timeout_s: Optional[float] = None,
) -> AsyncIterator[bool]:
    """Async-safe variant of :func:`flock_critical_section`.

    Acquires the cross-process flock in a worker thread (so the
    event loop is not blocked during the contention poll) and
    holds it across the async block. Releases identically on
    exit. NEVER raises.

    Yields True on success; False on timeout / fcntl-unavailable
    / OSError (caller branches; same contract as the sync variant).

    Composition: pure wrapper over :func:`flock_critical_section`
    using :class:`contextlib.AsyncExitStack` to keep the sync
    context manager's file descriptor open across the async block.
    No new lock-acquisition logic — single source of truth in the
    sync primitive.

    Use cases:
      * AutoCommitter.commit() critical section (PRD §3.6.2 #10
        — TOCTOU race between intent_token_exists / git commit /
        store_intent_token)
      * Any future async caller needing the §33.4 Per-Cluster
        flock'd JSONL Persistence pattern across awaitable
        operations.
    """
    from backend.core.ouroboros.governance.workspace_resolver import (
        resolve_durable_path,
    )
    path = resolve_durable_path(path)
    # Use a sync ExitStack — we drive it from worker threads
    # via asyncio.to_thread on enter + exit. The sync CM's fd
    # persists across the async yield because the stack holds
    # it open until we close().
    stack = _ExitStack()
    acquired_holder = [False]

    def _enter() -> bool:
        # Enter the sync context manager and store its yielded
        # bool. The CM's __enter__ runs in the worker thread;
        # __exit__ will run in another worker thread on async
        # exit. The fd persists across the async block because
        # the CM's state lives in the stack.
        cm = flock_critical_section(path, timeout_s=timeout_s)
        result = stack.enter_context(cm)
        acquired_holder[0] = bool(result)
        return acquired_holder[0]

    def _exit() -> None:
        try:
            stack.close()
        except Exception:  # noqa: BLE001 — defensive
            pass

    try:
        try:
            ok = await asyncio.to_thread(_enter)
        except Exception:  # noqa: BLE001 — defensive
            yield False
            return
        try:
            yield ok
        finally:
            try:
                await asyncio.to_thread(_exit)
            except Exception:  # noqa: BLE001 — defensive
                pass
    except Exception as exc:  # noqa: BLE001 — last-resort
        logger.debug(
            "[CrossProcessJSONL] async_flock_critical_section "
            "raised: %s", exc,
        )
        yield False


# ---------------------------------------------------------------------------
# Async append helpers — async_flock_append_line(s)
# ---------------------------------------------------------------------------
#
# Rationale (Slice 3 Task 2): `_acquire_cross_process_lock`'s LOCK_NB
# poll loop (`time.sleep` backoff, worst-case ~2x
# JARVIS_CROSS_PROCESS_LOCK_TIMEOUT_S) is sync-blocking. The sync
# `flock_append_line(s)` above is fine for genuinely sync callers, but
# an async caller invoking it directly runs the ENTIRE poll loop ON
# the asyncio event loop — the exact on-loop LoopSink hits observed
# in session bt-iso-1783574982 (81-300ms) via
# route_runner._classify_route -> phase8_producers.record_decision ->
# DecisionTraceLedger.record -> flock_append_line. These helpers route
# the whole lock-wait+write through cooperative_fs_io.offload so the
# poll loop runs on the F8 thread pool instead.


def _terminate_partial_tail(fh, target: Path) -> bool:
    """Give the file a trailing newline if it lacks one. NEVER raises.

    Called with the append handle open and the flock HELD, so the file
    cannot change underneath the check. Returns True if a terminator was
    written, which the caller does not need but a test does.

    Reads exactly one byte. The handle is opened in append mode, so the
    write below lands at the end regardless of where this seek leaves the
    read position — append-mode writes always go to the end on POSIX, and
    a separate read handle keeps the two positions from interacting at
    all.

    An empty or absent file needs nothing: there is no partial line to
    terminate, and writing a newline into an empty file would create a
    blank first line for every fresh log.
    """
    try:
        size = target.stat().st_size
        if size <= 0:
            return False
        with target.open("rb") as probe:
            probe.seek(-1, os.SEEK_END)
            if probe.read(1) == b"\n":
                return False
        fh.write("\n")
        logger.warning(
            "[CrossProcessJSONL] %s ended mid-line; terminated it before "
            "appending. One row was lost to an interrupted write — the "
            "rows around it are intact.", target,
        )
        return True
    except OSError as exc:
        # Fail OPEN. Refusing to append because the tail could not be
        # inspected would turn a recoverable torn line into total data
        # loss for everything after it.
        logger.debug("[CrossProcessJSONL] tail probe failed at %s: %s",
                     target, exc)
        return False


def _append_lines_with_mkdir(
    path: Path,
    lines: Tuple[str, ...],
    timeout_s: Optional[float],
) -> bool:
    """Offload body for the async append helpers — module-level so it
    is a plain picklable function (thread path doesn't require it,
    but keeps the substrate contract uniform). Ensures the parent
    dir exists (mirrors flock_critical_section's mkdir contract at
    :414-421), then delegates to the canonical sync append.
    NEVER raises."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug(
            "[CrossProcessJSONL] async-append parent mkdir failed: %s", exc,
        )
        return False
    return flock_append_lines(path, lines, timeout_s=timeout_s)


async def async_flock_append_lines(
    path: Path,
    lines: Iterable[str],
    *,
    timeout_s: Optional[float] = None,
) -> bool:
    """Async variant of flock_append_lines — the lock poll loop
    (LOCK_EX|LOCK_NB + time.sleep backoff, worst-case ~2x
    JARVIS_CROSS_PROCESS_LOCK_TIMEOUT_S) runs on the F8
    cooperative_fs_io pool, never on the asyncio loop (Slice 3;
    cures the on-loop decision-trace appends observed in session
    bt-iso-1783574982, 81-300ms LoopSink hits).

    ORDERING: awaited appends from one coroutine land in call
    order. Concurrent tasks appending the same path serialize on
    the per-path in-process lock + flock (all-or-nothing per call)
    but their relative order is scheduler-defined — same as today's
    cross-thread behavior.

    NON-REENTRANCY (inherited from the sync substrate): do NOT
    await this while holding flock_critical_section /
    async_flock_critical_section on the SAME path — the wait is
    BOUNDED (returns False at timeout_s) but always fails.
    NEVER raises."""
    # Durable-path resolution — the SINGLE async-path resolution point
    # (Slice 3 review): mirrors the sync funnel `flock_append_lines`
    # (:462) so the offload body's parent mkdir targets the DURABLE
    # parent, not an active overlay/worktree reroot source. Without
    # this, a direct call would mkdir a spurious empty overlay dir
    # while the write — resolved again, idempotently, inside
    # `flock_append_lines` — lands in the durable root. Both async
    # entry points funnel here, so `async_flock_append_line` delegates
    # without resolving itself (no double-resolve on the async path).
    from backend.core.ouroboros.governance.workspace_resolver import (
        resolve_durable_path,
    )
    try:
        path = resolve_durable_path(path)
    except Exception:  # noqa: BLE001 — mirror sync helper's fail-soft
        pass
    materialized = tuple(lines)  # never iterate a caller generator off-thread
    try:
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            offload,
            is_offload_error,
        )
    except Exception:  # noqa: BLE001 — substrate import fault
        try:
            return await asyncio.to_thread(
                _append_lines_with_mkdir, path, materialized, timeout_s,
            )
        except Exception:  # noqa: BLE001
            return False
    # NEVER-raises guard (Slice 3 T2 review #1): offload()'s thread
    # path acquires the shared executor + dispatches run_in_executor
    # without a top-level guard of its own — a RuntimeError from an
    # executor-shutdown race at teardown would otherwise propagate
    # out of this coroutine, contradicting the NEVER-raises contract.
    try:
        result = await offload(
            _append_lines_with_mkdir, path, materialized, timeout_s,
            cpu_bound=False,
        )
        if is_offload_error(result):
            logger.debug(
                "[CrossProcessJSONL] async append fail-soft: %r", result,
            )
            return False
        return bool(result)
    except Exception as exc:  # noqa: BLE001 — NEVER raise into caller
        logger.debug(
            "[CrossProcessJSONL] async append offload raised: %s", exc,
        )
        return False


async def async_flock_append_line(
    path: Path,
    line: str,
    *,
    timeout_s: Optional[float] = None,
) -> bool:
    """Async variant of flock_append_line. See
    async_flock_append_lines for the ordering / non-reentrancy
    contract. NEVER raises."""
    # Durable-path resolution is single-sourced in
    # `async_flock_append_lines` (the shared async funnel) — this
    # wrapper is a pure delegate so the async path resolves exactly
    # once (Slice 3 review: no double-resolve, symmetry with the
    # sync `flock_append_line` -> `flock_append_lines` funnel).
    return await async_flock_append_lines(path, (line,), timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "CROSS_PROCESS_JSONL_SCHEMA_VERSION",
    "async_flock_append_line",
    "async_flock_append_lines",
    "async_flock_critical_section",
    "fcntl_available",
    "flock_append_line",
    "flock_append_lines",
    "flock_critical_section",
    "lock_timeout_s",
    "stale_lock_age_s",
]
