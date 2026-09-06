"""
TestFailureSensor (Sensor B) — Adapter over existing TestWatcher.

Converts stable IntentSignal(source='intent:test_failure') objects into
IntentEnvelope(source='test_failure') objects and ingests them via the router.

Phase 2 Event Spine: also consumes ``.jarvis/test_results.json`` written by
the ouroboros_pytest_plugin, providing structured test results without
spawning a subprocess.

The existing TestWatcher (intent/test_watcher.py) handles pytest polling and
streak-based stability detection. This sensor wraps it as an adapter.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from backend.core.ouroboros.governance.intent.signals import IntentSignal
from backend.core.ouroboros.governance.intent.test_source_attribution import (
    attribute_strict_or_none,
    attribution_enabled,
    prewarm_module_map,
    strict_isolation_enabled,
)
from backend.core.ouroboros.governance.intent.test_watcher import TestFailure
from backend.core.ouroboros.governance.workspace_resolver import resolve_repo_root
from backend.core.ouroboros.governance.intake.intent_envelope import (
    IntentEnvelope,
    make_envelope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-flight dedup (bt-2026-04-15-010727 findings)
# ---------------------------------------------------------------------------
#
# TestFailureSensor polls every ``poll_interval_s`` (default 30s via
# JARVIS_INTENT_TEST_INTERVAL_S). When an op for a broken test file is
# already in flight, subsequent polls at t+30s / t+60s / … continue to
# observe the same broken test (the in-flight op hasn't APPLIED yet) and
# re-emit signals. Each re-emission is accepted by the router because:
#
#   (a) The router's ``register_active_op`` hook — which would populate
#       ``_active_file_ops`` and trigger ``_find_file_conflict`` → queued_behind
#       — is defined but NEVER called from any caller. Dead code as of this
#       fix. That path would require wiring in GLS and an orchestration
#       ordering guarantee (register before the next ingest arrives), which
#       has its own race window.
#   (b) GLS's *separate* ``_active_file_ops`` set (line 966 in
#       governed_loop_service.py) IS populated at dispatch time and rejects
#       duplicates with ``reason_code="file_in_flight"`` — but only *after*
#       the router has already accepted the envelope, burned a WAL entry,
#       created an op_id, and handed it to GLS. In v5 the test_failure
#       concurrency storm (3 ops × same file × 88s under 85s Claude first-
#       token) bypassed GLS's check entirely, probably because the three
#       workers raced past the check window.
#
# Sensor-side dedup is the narrow, race-free fix: reject the re-emission at
# the earliest possible point (before even calling ``router.ingest``) using
# an in-process dict keyed by target_file. TTL-based cleanup means a stuck
# op eventually releases the slot automatically — we don't need a completion
# callback from the orchestrator.
#
# Env gate: ``JARVIS_TEST_FAILURE_INFLIGHT_TTL_S`` (default 300s). Set to 0
# or negative to disable the dedup entirely.

_INFLIGHT_TTL_S: float = float(
    os.environ.get("JARVIS_TEST_FAILURE_INFLIGHT_TTL_S", "300")
)

# --- Gap #4 migration: FS-event primary mode (Slice 3) --------------------
#
# Manifesto §3 (Disciplined Concurrency): test failures are the highest-
# leverage self-healing signal in the organism. Polling pytest every 30s
# is thermodynamic waste — pytest runs even when no code has changed.
# When ``JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED=true``, the FileSystemEvent
# Bridge (``fs.changed.*`` on ``TrinityEventBus``) becomes the primary
# trigger and the legacy poll loop demotes to a
# ``JARVIS_TEST_FAILURE_FALLBACK_INTERVAL_S`` cadence (default 600s = 10min)
# whose only job is to catch missed FS events.
#
# Shadow pattern: flag defaults OFF so current production behavior is
# pure-poll (30s) with no FS subscription — no silent activation. Operators
# flip the flag to true, run a graduation arc, then the default flips in a
# follow-up commit. Matches the GitHubIssueSensor Slice 1/2 precedent.
_TEST_FAILURE_FALLBACK_INTERVAL_S: float = float(
    os.environ.get("JARVIS_TEST_FAILURE_FALLBACK_INTERVAL_S", "600")
)


# --- Slice 5 T2: set-based accumulator debounce (Run #15 L2, F2) ----------
#
# The single-slot debounce (``asyncio.create_task`` + cancel-on-every-event)
# was last-event-wins: a burst of FS events (e.g. a worktree checkout
# touching many files) cancelled the pending task on EVERY new event, so a
# task that never survives long enough to fire never runs the FIRST event's
# scoped targets. Run #15's leaf-change event was evicted this way inside a
# worktree burst. Fixed window semantics replace cancel-on-event: the first
# ``.py`` event opens a ``_debounce_window_s()`` window, every event during
# it ADDS to ``_pending_changed_paths`` (never evicts), and one scoped run
# covers the union. Not re-read into a module constant (env re-checked at
# call time), same pattern as the other gate helpers above.


def _debounce_window_s() -> float:
    return float(os.environ.get("JARVIS_TEST_FAILURE_DEBOUNCE_WINDOW_S", "2.0"))


def _debounce_max_paths() -> int:
    return int(os.environ.get("JARVIS_TEST_FAILURE_DEBOUNCE_MAX_PATHS", "32"))


def fs_events_enabled() -> bool:
    """Re-read ``JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED`` at call-time.

    Not cached — tests monkeypatch the env and the sensor's
    ``subscribe_to_bus`` re-checks on invocation, same pattern as
    ``github_issue_sensor.webhook_enabled``.
    """
    return os.environ.get(
        "JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED", "true",
    ).lower() in ("true", "1", "yes")


# --- Dynamic test scoping (2026-06-24) ------------------------------------
#
# The FS-driven pytest run used to call ``poll_once()`` blind — pytest ran
# the WHOLE ``tests/`` suite on every ``.py`` change. On a non-trivial repo
# that sweep exceeds the 180s pytest ceiling and is SIGKILLed mid-run, so a
# stable failure introduced by a single edit is NEVER detected. This is the
# exact foil that blocked O+V's chaos self-detection in the A1 soak.
#
# Fix (reuse-first, no new mapper): the changed file's repo-relative path is
# threaded to the EXISTING ``TestRunner.resolve_affected_tests`` — a 4-level
# deterministic mapper (name-convention -> recursive search -> package
# fallback -> repo fallback, capped at ``JARVIS_TEST_MAX_FILES``) — and the
# bounded result is passed to ``poll_once(target_paths=...)``. A one-line
# edit now runs ONLY that file's tests.
#
# Fail-safe ladder (never the whole ``tests/`` on a resolve failure):
#   1. resolve_affected_tests -> bounded scoped targets (primary).
#   2. resolver empty / errors -> nearest sibling ``tests/<mirror-dir>/``
#      (bounded to a single directory, not the repo root).
#   3. mirror dir unresolvable -> deep-background full-suite poll ONLY when
#      ``JARVIS_TEST_FULL_SUITE_FALLBACK`` is explicitly true (default
#      false); otherwise the run is skipped (a missed run is cheaper than a
#      180s SIGKILL that detects nothing).
#
# Master gate ``JARVIS_TEST_DYNAMIC_SCOPING_ENABLED`` (default true). OFF ->
# the FS path calls ``poll_once()`` with no target_paths == legacy
# whole-suite behavior, byte-identical.


def dynamic_scoping_enabled() -> bool:
    """Re-read ``JARVIS_TEST_DYNAMIC_SCOPING_ENABLED`` at call-time (default true).

    Not cached so tests can monkeypatch the env per-case (same pattern as
    ``fs_events_enabled``). OFF restores byte-identical legacy whole-suite
    behavior on the FS path.
    """
    return os.environ.get(
        "JARVIS_TEST_DYNAMIC_SCOPING_ENABLED", "true",
    ).lower() in ("true", "1", "yes")


def fs_confirm_enabled() -> bool:
    """Re-read ``JARVIS_TEST_FAILURE_FS_CONFIRM_ENABLED`` at call-time
    (default true). When on, an fs.changed-scoped run that observes a NEW
    failure re-runs the SAME scoped targets once, immediately — a
    deterministic failure reaches the 2-consecutive-runs stability gate in
    seconds instead of waiting for the 600s poll fallback (which the A1
    soak wall outlives: a1-brain-20260706-014931, where the vector's
    single scoped observation starved at streak 1 for the whole session).
    Bounded: at most ONE confirmation re-run per fs.changed event, scoped
    targets only, never the full suite."""
    return os.environ.get(
        "JARVIS_TEST_FAILURE_FS_CONFIRM_ENABLED", "true",
    ).strip().lower() in ("1", "true", "yes", "on")


def quiet_reconcile_enabled() -> bool:
    """Re-read ``JARVIS_TEST_FAILURE_QUIET_RECONCILE_ENABLED`` (default true).

    Slice 5 F5/F6 (Run #15 L4): the event-primary derate skips the legacy
    whole-suite poll indefinitely once armed, so a dropped ``fs.changed``
    event is dropped for the rest of the session. This flag gates the
    bounded recovery — a single git-dirty-scoped reconcile fired only when
    a whole fallback window observed ZERO fs events. Not cached so tests
    can monkeypatch per-case (same pattern as ``fs_events_enabled``).
    """
    return os.environ.get(
        "JARVIS_TEST_FAILURE_QUIET_RECONCILE_ENABLED", "true",
    ).strip().lower() in ("1", "true", "yes", "on")


def full_suite_fallback_enabled() -> bool:
    """Re-read ``JARVIS_TEST_FULL_SUITE_FALLBACK`` at call-time (default false).

    The last-resort escape hatch: when scoping is on but NOTHING resolves
    (no scoped targets, no mirror dir), only fall back to the whole-suite
    poll if an operator has explicitly opted in. Default false means a
    fully-unresolvable change skips the run rather than risking the 180s
    SIGKILL that detects nothing.
    """
    return os.environ.get(
        "JARVIS_TEST_FULL_SUITE_FALLBACK", "false",
    ).lower() in ("true", "1", "yes")


# --- Boot-Time Differential Hydration (offline-state blindspot fix) --------
#
# An event-driven watcher that boots AFTER a state mutation loses the
# ``fs.changed`` event forever. The A1 live soak proved this: the chaos bug is
# mutated BEFORE O+V boots, so no FS event ever fires and the only fallback
# (the full-suite poll) SIGKILLs at 180s without detecting it. The same hole
# opens on ANY crash/restart in prod.
#
# On boot the sensor reconstructs the missed change set from GROUND TRUTH (the
# working tree, via ``git diff --name-only HEAD``), resolves each changed file
# to its tests through the SAME ``resolve_affected_tests`` mapper the live FS
# path uses, and runs the localized SCOPED pytest immediately. De-dupe tracking
# (``_hydrated_keys``) suppresses a redundant live ``fs.changed`` run for a file
# that was just hydrated.
#
# Gated ``JARVIS_TESTWATCHER_BOOT_HYDRATION_ENABLED`` (default true). OFF ->
# the sensor never hydrates == legacy byte-identical behavior.
# ``JARVIS_TESTWATCHER_HYDRATION_DEDUP_TTL_S`` (default 120s) bounds the de-dupe
# window so a genuinely-recurring later edit still re-runs.


def boot_hydration_enabled() -> bool:
    """Re-read ``JARVIS_TESTWATCHER_BOOT_HYDRATION_ENABLED`` (default true).

    Not cached so tests can monkeypatch per-case (same pattern as
    ``fs_events_enabled``). OFF restores byte-identical legacy boot behavior.
    """
    return os.environ.get(
        "JARVIS_TESTWATCHER_BOOT_HYDRATION_ENABLED", "true",
    ).lower() in ("true", "1", "yes")


# --- Test-Cache-First boot hydration (state-hashing blind-spot fix) --------
#
# The working-tree/Merkle hydration above keys on CONTENT CHANGE: a red test
# whose file hash is unchanged since the last snapshot (``walk_changed=0``) is
# invisible to it — the failure predates the snapshot, so no ``fs.changed``
# fires and the git-diff finds a clean tree. That is a *persistent
# environmental defect*, and pytest already persists it in its own
# ``.pytest_cache/v/cache/lastfailed`` ledger (survives process restarts).
#
# This layer reads that ledger BEFORE the hash diff and seeds a repair for
# each unresolved red — independent of whether any file hash changed. It emits
# through the SAME ``process_failures`` -> ``handle_signals`` -> ``router.ingest``
# sink every other path uses (DRY), treating the cache as the FIRST observation
# of each failure so the 2-consecutive-runs stability contract is honored with
# the cache standing in for the prior run it recorded.


def cache_first_hydration_enabled() -> bool:
    """Re-read ``JARVIS_TEST_FAILURE_CACHE_FIRST_ENABLED`` (default true).

    Not cached so tests can monkeypatch per-case (same pattern as
    ``boot_hydration_enabled``). OFF -> byte-identical legacy boot (the
    git-diff working-tree hydration only, no pytest-cache consultation).
    """
    return os.environ.get(
        "JARVIS_TEST_FAILURE_CACHE_FIRST_ENABLED", "true",
    ).strip().lower() in ("1", "true", "yes", "on")


def _cache_first_max_files() -> int:
    """Bound the cache-first synthetic seed count (default 16) so a large
    persisted red suite cannot flood intake at boot. Fail-soft to 16."""
    try:
        return max(
            1, int(os.environ.get("JARVIS_TEST_FAILURE_CACHE_FIRST_MAX_FILES", "16"))
        )
    except (TypeError, ValueError):
        return 16


_HYDRATION_DEDUP_TTL_S: float = float(
    os.environ.get("JARVIS_TESTWATCHER_HYDRATION_DEDUP_TTL_S", "120")
)

# Boot-marker the Chaos Readiness Handshake (and operators) grep for to know
# the TestWatcher subscription is LIVE. Emitted once subscribe_to_bus succeeds.
TESTWATCHER_READY_MARKER = "[TestWatcher] READY subscribed=fs.changed.*"


# --- Slice 4 T3: off-loop scoped-target resolution ------------------------
#
# Run #14 tombstone: the main thread wedged 83 minutes inside
# ``_resolve_scoped_targets`` -> ``pathlib.resolve`` -> ``_joinrealpath``
# (test_failure_sensor.py:836 at the time). ``Path.resolve()`` walks the
# filesystem synchronously (symlink resolution, ``_joinrealpath``) and the
# downstream ``TestRunner.resolve_affected_tests`` call does its own
# directory walks -- both are blocking work that has no business running on
# the asyncio loop that ALSO needs to keep servicing the FS-event bridge,
# heartbeats, and every other sensor. ``_offload_fs`` routes that work
# through the unified ``cooperative_fs_io.offload`` substrate (thread pool),
# copying the sanctioned import-fault fallback idiom from
# ``posture_observer._offload_signal`` (posture_observer.py:204-229) -- the
# ONLY sanctioned fallback shape (Mandate 3).
async def _offload_fs(fn, /, *args, **kwargs):
    """Route blocking FS work through the cooperative substrate; fail-soft
    to the caller's neutral value at the call site (never raises past it).
    Copied import-fault idiom from posture_observer._offload_signal."""
    try:
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            is_offload_error,
            offload,
        )
    except ImportError:
        import asyncio as _aio
        return await _aio.get_event_loop().run_in_executor(
            None, lambda: fn(*args, **kwargs)
        )
    result = await offload(fn, *args, cpu_bound=False, **kwargs)
    if is_offload_error(result):
        raise RuntimeError(f"offload failed: {result!r}")
    return result


def _mirror_tests_dir_sync(
    changed_abs: Path, repo_root: Path
) -> Optional[Path]:
    """Module-level counterpart of ``TestFailureSensor._mirror_tests_dir``.

    Extracted (Slice 4 T3) so the off-loop sync resolver
    (:func:`_resolve_scoped_targets_sync`) can call it without an instance.
    Bounded to a single nearest sibling ``tests/`` dir, NOT the repo root.
    """
    try:
        from backend.core.ouroboros.governance.test_runner import (
            _find_sibling_tests_dir,
        )
    except Exception:
        return None
    try:
        sibling = _find_sibling_tests_dir(changed_abs)
    except Exception:
        return None
    if sibling is None:
        return None
    if TestFailureSensor._is_repo_test_root(sibling, repo_root):
        return None
    return sibling


def _fs_io_inline_mode() -> bool:
    """True when the ``cooperative_fs_io`` master switch is OFF — i.e.
    ``offload()`` would run the delegated fn INLINE on this loop thread
    (cooperative_fs_io.py's master-off byte-identical-rollback contract).

    Consulted by ``_resolve_scoped_targets`` to pick the primary-mapper
    EXECUTION strategy: master-off must await the primary natively on the
    loop (``asyncio.run`` inside an inlined worker raises ``RuntimeError:
    cannot be called from a running event loop`` and would silently kill
    the primary AST resolver — the Slice 4 T3 re-review Critical).
    Fail-soft: a consult fault assumes the offload path, which is safe —
    ``_offload_fs``'s own import-fault fallback still dispatches to a
    loop-free executor thread where ``asyncio.run`` works.
    """
    try:
        from backend.core.ouroboros.governance.cooperative_fs_io import (
            cooperative_fs_io_enabled,
        )
        return not cooperative_fs_io_enabled()
    except Exception:  # noqa: BLE001 — consult fault -> assume offload path
        return False


async def _resolve_primary(
    repo_root: Path, changed_abs: Path, changed_rel_path: str
) -> Optional[List[str]]:
    """Primary mapper step — the existing 4-level deterministic
    ``TestRunner.resolve_affected_tests`` + whole-suite-root filter.

    SINGLE-SOURCED (Mandate 3) with two execution strategies:
    * awaited NATIVELY by ``_resolve_scoped_targets`` when
      ``cooperative_fs_io`` is master-off (the on-loop inline degrade —
      same coroutine, same results, byte-identical legacy semantics);
    * driven via ``asyncio.run`` by ``_resolve_scoped_targets_sync``
      inside the offload worker thread (no running loop there).

    Best-effort (existing contract): any fault logs at debug and returns
    ``None`` so the caller falls to the shared fallback ladder.
    """
    try:
        from backend.core.ouroboros.governance.test_runner import TestRunner

        runner = TestRunner(repo_root)
        resolved = await runner.resolve_affected_tests((changed_abs,))
        targets = [
            str(p) for p in resolved
            if not TestFailureSensor._is_repo_test_root(p, repo_root)
        ]
        return targets or None
    except Exception as exc:  # noqa: BLE001 — resolver is best-effort
        logger.debug(
            "TestFailureSensor: resolve_affected_tests failed for %r: %s",
            changed_rel_path, exc,
        )
        return None


# Sentinel distinguishing "primary not yet executed" (offload-worker mode:
# drive it here via asyncio.run) from "primary already executed by the
# caller and returned None" (master-off mode: the async wrapper awaited it
# natively and injects the result).
_PRIMARY_UNSET: Any = object()


def _resolve_scoped_targets_sync(
    repo_root: Path,
    changed_rel_path: str,
    primary_targets: Any = _PRIMARY_UNSET,
) -> Optional[List[str]]:
    """Map *changed_rel_path* -> bounded scoped pytest targets.

    The ONE source of truth for the resolution SEQUENCE (primary mapper ->
    mirror-dir -> tier-3 package discovery), extracted from the pre-Slice-4
    ``TestFailureSensor._resolve_scoped_targets`` body. Only the
    primary-mapper EXECUTION strategy differs by mode:

    * Offload mode (default, ``primary_targets`` unset): this function
      runs inside the ``cooperative_fs_io`` thread-pool worker and drives
      the async :func:`_resolve_primary` via ``asyncio.run()`` — safe
      because a worker thread never has a running event loop.
      (``TestRunner.resolve_affected_tests`` has NO sync underlying
      mapper: every blocking step inside it — ``_get_ast_import_map``,
      ``_find_tests_suffix_aware``, ``_find_test_recursive`` — is itself
      ``loop.run_in_executor``-wrapped and requires a running loop;
      test_runner.py:1006-1025, :803-825, :828-862.)
    * Inline / master-off mode: the async wrapper has ALREADY awaited
      :func:`_resolve_primary` natively on the loop (``asyncio.run`` on a
      running loop raises — the Slice 4 T3 re-review Critical) and
      injects its result via *primary_targets*; only the pure-sync
      fallback ladder runs here.

    Returns
    -------
    * A non-empty ``List[str]`` of scoped test paths when the existing
      ``TestRunner.resolve_affected_tests`` (or the bounded mirror-dir
      fallback) yields targets.
    * ``None`` when nothing resolved — the caller then either skips the
      run or (opt-in) falls back to the whole suite. **Never** returns
      the whole ``tests/`` directory implicitly.

    Fail-safe by construction: any error in the resolver degrades to the
    mirror-dir fallback, and any error there degrades to ``None``.
    """
    if not changed_rel_path:
        return None

    changed_abs = (repo_root / changed_rel_path).resolve()

    if primary_targets is _PRIMARY_UNSET:
        # Worker-thread mode: drive the shared async primary here.
        # Defensive belt: if a future code path ever executes this
        # function INLINE on a loop thread without injecting the primary,
        # fail LOUD (warning + fallbacks) instead of the silent swallowed
        # RuntimeError that motivated this split.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop_running = False
        else:
            loop_running = True
        if loop_running:
            logger.warning(
                "TestFailureSensor: _resolve_scoped_targets_sync executed "
                "inline on a loop thread without a precomputed primary — "
                "primary mapper SKIPPED. Inline callers must await "
                "_resolve_primary natively and inject primary_targets."
            )
            primary_targets = None
        else:
            try:
                primary_targets = asyncio.run(
                    _resolve_primary(repo_root, changed_abs, changed_rel_path)
                )
            except Exception as exc:  # noqa: BLE001 — drive fault, not mapper fault
                logger.debug(
                    "TestFailureSensor: primary mapper drive failed for "
                    "%r: %s", changed_rel_path, exc,
                )
                primary_targets = None

    if primary_targets:
        return primary_targets

    # Fail-safe: nearest sibling mirror tests/ dir (bounded, single dir).
    mirror = _mirror_tests_dir_sync(changed_abs, repo_root)
    if mirror is not None:
        return [str(mirror)]

    # Tier 3: package-layout test discovery — search for test_<stem>.py
    # across known test dirs when both primary resolver and mirror-dir
    # fail. Prevents silent signal drop on AST analysis failures.
    try:
        stem = Path(changed_rel_path).stem
        if stem and not stem.startswith("test_"):
            pattern = "test_%s.py" % (stem,)
            pkg_targets: list = []
            for test_root_name in ("tests", "test"):
                test_root = repo_root / test_root_name
                if test_root.is_dir():
                    for hit in test_root.rglob(pattern):
                        pkg_targets.append(str(hit))
                        if len(pkg_targets) >= 3:  # bounded
                            break
                if len(pkg_targets) >= 3:
                    break
            if pkg_targets:
                logger.info(
                    "TestFailureSensor: package-layout fallback resolved "
                    "%d test target(s) for %r",
                    len(pkg_targets), changed_rel_path,
                )
                return pkg_targets
    except Exception:  # noqa: BLE001 — fail-soft
        logger.debug(
            "TestFailureSensor: package-layout fallback error for %r",
            changed_rel_path, exc_info=True,
        )
    return None


class TestFailureSensor:
    """Adapter that bridges TestWatcher → UnifiedIntakeRouter.

    Parameters
    ----------
    repo:
        Repository name (e.g. ``"jarvis"``).
    router:
        UnifiedIntakeRouter instance.
    test_watcher:
        Optional existing TestWatcher. If None, sensor operates in
        signal-push mode only (caller calls ``handle_signals()``).
    """

    def __init__(
        self,
        repo: str,
        router: Any,
        test_watcher: Any = None,
    ) -> None:
        self._repo = repo
        self._router = router
        self._watcher = test_watcher
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        # In-flight dedup: target_file_path -> monotonic submitted_at
        # See module docstring above ``_INFLIGHT_TTL_S`` for the rationale.
        self._pending_target_keys: Dict[str, float] = {}
        # Gap #4 migration — captured at __init__ time. When True, the poll
        # loop runs at the fallback cadence and the FS subscription is the
        # primary trigger. When False, preserves legacy pure-poll behavior.
        self._fs_events_mode: bool = fs_events_enabled()
        # Slice 4 T3: latch so the poll-loop derate logs once per
        # state-change (armed -> derated / disarmed -> resumed), not once
        # per cycle.
        self._poll_derate_logged: bool = False
        # Telemetry counters (exposed via health snapshots; useful for
        # convergence tracking during the graduation arc).
        self._fs_events_handled: int = 0
        self._fs_events_ignored: int = 0
        # Review fix (T5 Important): the quiet-lane reconcile (_poll_task)
        # and the debounce run (_debounce_task) both reach
        # ``self._watcher.poll_once`` — and _run_scoped_with_confirmation
        # snapshots + re-reads ``_failure_streak`` around a subprocess
        # await, so interleaved runs can corrupt the stability-gate
        # accounting. Every sensor-side poll_once invocation (plus its
        # streak read-back critical section) serializes through this lock.
        # NEVER held across the derate sleep — only around the run itself.
        self._poll_once_lock: asyncio.Lock = asyncio.Lock()
        # Boot hydration de-dupe: changed_rel_path -> monotonic hydrated_at.
        # A file hydrated on boot is suppressed from a redundant live
        # fs.changed run for ``_HYDRATION_DEDUP_TTL_S``.
        self._hydrated_keys: Dict[str, float] = {}
        self._boot_hydrated: bool = False
        # Universal Terminal-State Lock Releaser: register as an ingress-lock
        # surface so ANY terminal op-state (the TrinityEventBus terminal
        # observer, or a LeaseReaper re-queue) auto-revokes this sensor's target
        # locks via the existing ``release_target``. This closes the sensor-side
        # wedge (soak bt-2026-07-22-174240) — ``release_target`` finally has a
        # caller, driven by a central event bridge rather than scattered calls.
        try:
            from backend.core.ouroboros.governance.terminal_lock_releaser import (  # noqa: E501
                get_terminal_lock_releaser as _get_tlr,
            )
            _get_tlr().register_surface(self)
        except Exception:  # noqa: BLE001 — registration is resilience, never blocks init
            pass
        # Slice 5 T2 (Run #15 autopsy L2, F2): set-based debounce
        # accumulator. Events never cancel the pending window — paths
        # aggregate and one scoped run covers the union.
        self._pending_changed_paths: Set[str] = set()
        # Initialized here (not only in ``subscribe_to_bus``) so a caller
        # that drives ``_on_fs_event`` directly (e.g. tests, or a future
        # non-bus event source) never hits an AttributeError before the
        # bus subscription runs. ``subscribe_to_bus`` re-initializes this
        # to ``None`` too, which is safe/idempotent — it always runs
        # before any ``fs.changed`` event could reach ``_on_fs_event``.
        self._debounce_task: Optional[asyncio.Task] = None
        # Slice 5 F4 (Run #15 L3, results-file staleness gate): monotonic
        # timestamp of the last time a *fresh, parseable* plugin results
        # read armed the 10s suppression window. Sole other write site is
        # the fresh-parse path in ``_on_test_results_changed`` — absent/
        # stale/unparseable results files never touch this.
        self._last_plugin_ts: float = 0.0
        # Slice 5 F4: wall-clock boot time, used as the staleness floor's
        # fallback when no pytest run has been spawned yet by the watcher
        # (e.g. a results file left over from before this sensor booted).
        self._boot_walltime: float = time.time()

    def _prune_stale_pending(self) -> None:
        """Drop pending target entries that have exceeded their TTL.

        Bounds the dict size and ensures a stuck op (orchestrator crash,
        hibernation, forgotten release callback) eventually releases the
        slot so the next legitimate signal for the same file can flow.
        """
        if _INFLIGHT_TTL_S <= 0 or not self._pending_target_keys:
            return
        now = time.monotonic()
        stale = [
            k for k, ts in self._pending_target_keys.items()
            if now - ts > _INFLIGHT_TTL_S
        ]
        for k in stale:
            del self._pending_target_keys[k]

    def _in_flight_target(self, signal: IntentSignal) -> Optional[str]:
        """Return the first target_file from *signal* that is already
        marked in-flight (within TTL), or None if all targets are free.

        Called before ``router.ingest`` to short-circuit re-emission of
        a signal whose target file already has an op working on it.
        """
        if _INFLIGHT_TTL_S <= 0:
            return None
        self._prune_stale_pending()
        for target in (signal.target_files or ()):
            if target in self._pending_target_keys:
                return target
        return None

    def _mark_targets_in_flight(self, signal: IntentSignal) -> None:
        """Record the signal's target files as in-flight. Called only
        after a successful ``router.ingest`` with status ``"enqueued"`` —
        dropped / deduplicated / queued signals do NOT mark targets,
        because the router is going to re-ingest them later and that
        re-ingest should not be self-suppressed by sensor-side dedup.
        """
        if _INFLIGHT_TTL_S <= 0:
            return
        now = time.monotonic()
        for target in (signal.target_files or ()):
            self._pending_target_keys[target] = now

    def release_target(self, target_file: str) -> None:
        """Manually release an in-flight target slot. Public API for
        orchestrator / GLS completion hooks that want to unblock the
        next signal immediately instead of waiting for TTL expiry.
        Idempotent — no-op if the target was not tracked.
        """
        self._pending_target_keys.pop(target_file, None)

    async def _signal_to_envelope_and_ingest(
        self, signal: IntentSignal
    ) -> Optional[IntentEnvelope]:
        """Convert one IntentSignal to IntentEnvelope and ingest it.

        Returns the envelope if ingested, None if skipped.
        """
        if not signal.stable:
            return None

        # In-flight dedup: reject re-emission while an op is already
        # working on any of the signal's target files. This is the
        # earliest-possible short-circuit — before envelope creation,
        # before router.ingest, before any WAL / queue / op_id burn.
        in_flight_target = self._in_flight_target(signal)
        if in_flight_target is not None:
            logger.info(
                "TestFailureSensor: suppressing re-emission — target "
                "%s already in-flight (%.0fs ago): %s",
                in_flight_target,
                time.monotonic() - self._pending_target_keys[in_flight_target],
                signal.description[:80],
            )
            return None

        confidence = min(1.0, signal.confidence)
        envelope = make_envelope(
            source="test_failure",
            description=signal.description,
            target_files=signal.target_files,
            repo=self._repo,
            confidence=confidence,
            urgency="high",
            evidence=dict(signal.evidence),
            requires_human_ack=False,
            causal_id=signal.signal_id,  # signal_id becomes causal_id
            signal_id=signal.signal_id,
        )
        try:
            result = await self._router.ingest(envelope)
            if result == "enqueued":
                self._mark_targets_in_flight(signal)
                logger.info(
                    "TestFailureSensor: enqueued test failure: %s",
                    signal.description,
                )
            return envelope
        except Exception:
            logger.exception("TestFailureSensor: ingest failed: %s", signal.description)
            return None

    async def handle_signals(
        self, signals: List[IntentSignal]
    ) -> List[Optional[IntentEnvelope]]:
        """Process a batch of IntentSignals. Returns per-signal results."""
        results = []
        for sig in signals:
            result = await self._signal_to_envelope_and_ingest(sig)
            results.append(result)
        return results

    async def start(self) -> None:
        """Start background polling via TestWatcher (if provided)."""
        if self._watcher is None:
            return
        self._running = True
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="test_failure_sensor_poll",
        )
        if self._fs_events_mode:
            logger.info(
                "TestFailureSensor: FS-events primary mode — poll demoted to "
                "%ds fallback (JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED=true)",
                int(_TEST_FAILURE_FALLBACK_INTERVAL_S),
            )
        else:
            logger.debug(
                "TestFailureSensor: poll-primary mode (%.0fs interval) — "
                "FS events disabled (default)",
                self._watcher.poll_interval_s,
            )

    async def stop(self) -> None:
        """Cancel the poll task and stop the underlying watcher.

        Previously this method was sync and only set ``_running=False``
        + stopped the watcher — the poll task reference was never
        captured, so asyncio emitted "Task was destroyed but pending"
        on every teardown (battle test bt-2026-04-13-031119). Now
        async so callers can ``await`` clean drain; task handle is
        tracked from ``start()`` and cancelled deterministically.
        """
        self._running = False
        if self._watcher is not None:
            try:
                self._watcher.stop()
            except Exception:
                logger.debug("TestFailureSensor: watcher.stop() raised", exc_info=True)
        task = self._poll_task
        self._poll_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # ------------------------------------------------------------------
    # Boot-Time Differential Hydration (offline-state blindspot fix)
    # ------------------------------------------------------------------

    def _is_recently_hydrated(self, changed_rel_path: str) -> bool:
        """True if *changed_rel_path* was hydrated within the de-dupe TTL.

        Lets a live ``fs.changed`` for a just-hydrated file be suppressed so
        the same edit is not double-run (boot hydration + live event). A
        different file (or one whose TTL expired) is not suppressed.
        """
        if _HYDRATION_DEDUP_TTL_S <= 0 or not changed_rel_path:
            return False
        ts = self._hydrated_keys.get(changed_rel_path)
        if ts is None:
            return False
        if time.monotonic() - ts > _HYDRATION_DEDUP_TTL_S:
            # Expired -> drop the entry and allow a fresh run.
            self._hydrated_keys.pop(changed_rel_path, None)
            return False
        return True

    def _pytest_lastfailed_path(self) -> Path:
        """Path to pytest's persisted ``lastfailed`` ledger under the repo
        root — the SAME ``.git``-anchored root the git-diff + scoped resolver
        use (``_repo_root``), so the cache and the failure loci agree."""
        return self._repo_root() / ".pytest_cache" / "v" / "cache" / "lastfailed"

    def _read_pytest_lastfailed(self) -> List[str]:
        """Read pytest's ``lastfailed`` cache -> list of failing node-ids.

        The file is a JSON object mapping ``nodeid -> true`` for every test
        that failed in pytest's most recent run (pytest's own persisted
        ground truth). Missing / unreadable / malformed -> ``[]``. Only
        truthy, path-shaped keys are kept. Never raises.
        """
        path = self._pytest_lastfailed_path()
        try:
            if not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        return [str(k) for k, v in data.items() if v and ".py" in str(k)]

    async def hydrate_from_pytest_cache(self) -> int:
        """Test-Cache-First boot hydration — seed persistent reds the hash
        diff cannot see (``walk_changed=0``).

        An event-driven sensor keyed on file-hash change is blind to a red
        test whose source did NOT change since the last snapshot: the failure
        predates the snapshot, so no ``fs.changed`` fires and the git-diff
        hydration finds a clean tree. That is a persistent ENVIRONMENTAL
        defect — and pytest already records it. This layer reads pytest's own
        ``lastfailed`` ledger and constructs synthetic failure signals for each
        unresolved red, independent of any hash diff.

        DRY: emits through the SAME ``process_failures`` ->
        ``handle_signals`` -> ``router.ingest`` sink every other path uses.
        The cache is treated as the FIRST observation of each failure (streak
        seeded to 1, never downgrading a higher live streak), so a single
        ``process_failures`` pass promotes it past the 2-consecutive-runs
        stability threshold — the cache standing in for the prior run it
        persisted. Each seeded file is recorded in ``_hydrated_keys`` so a
        redundant live ``fs.changed`` for it is de-duped.

        Gated ``JARVIS_TEST_FAILURE_CACHE_FIRST_ENABLED`` (default true);
        OFF / no watcher / empty cache -> 0 with no side effects. Fail-soft:
        any error logs at DEBUG and returns the count so far.
        """
        if not cache_first_hydration_enabled() or self._watcher is None:
            return 0
        reds = self._read_pytest_lastfailed()
        if not reds:
            return 0
        cap = _cache_first_max_files()
        now = time.monotonic()
        failures: List[TestFailure] = []
        streaks = getattr(self._watcher, "_failure_streak", None)
        for nodeid in reds[:cap]:
            test_file = nodeid.split("::", 1)[0]
            # The cache IS the prior observation → seed streak to 1 so one
            # process_failures pass reaches the stable (>=2) threshold. Never
            # downgrade an existing (higher) live streak.
            if isinstance(streaks, dict):
                streaks[nodeid] = max(streaks.get(nodeid, 0), 1)
            failures.append(
                TestFailure(
                    test_id=nodeid,
                    file_path=test_file,
                    error_text=(
                        "pytest lastfailed cache: persistent unresolved failure"
                    ),
                )
            )
            self._hydrated_keys[test_file] = now
        if not failures:
            return 0
        try:
            if attribution_enabled():
                await prewarm_module_map(self._watcher.repo_path)
        except Exception:
            logger.debug("[CacheFirstHydration] prewarm failed", exc_info=True)
        # Anti-noise (soak bt-2026-09-06): a force-promoted lastfailed red has
        # no traceback, so attribution would spray every imported module. Keep
        # ONLY reds whose failing source is deterministically ISOLABLE; discard
        # the rest here — at the sensor — so the orchestrator queue is never
        # polluted with a 6-file import spray the generator can only no-op.
        # Gated by the strict-isolation master; OFF -> byte-identical (no
        # filtering). The module-map is already warm from the prewarm above, so
        # each check is a dict cache-hit, never an on-loop crawl.
        if attribution_enabled() and strict_isolation_enabled():
            _repo = self._watcher.repo_path
            _kept: List[TestFailure] = []
            _discarded = 0
            for _f in failures:
                try:
                    _attr = attribute_strict_or_none(
                        _f.file_path, repo_root=_repo, traceback_frames=(),
                    )
                except Exception:  # noqa: BLE001 — filter is best-effort
                    _attr = None
                if _attr is None:
                    _discarded += 1
                else:
                    _kept.append(_f)
            if _discarded:
                logger.info(
                    "[CacheFirstHydration] discarded %d unmappable stale red(s) "
                    "(no traceback, source not isolable) — not enqueuing an "
                    "import spray; %d isolable red(s) kept",
                    _discarded, len(_kept),
                )
            failures = _kept
            if not failures:
                return 0
        try:
            signals = self._watcher.process_failures(failures)
        except Exception:
            logger.debug(
                "[CacheFirstHydration] process_failures failed", exc_info=True
            )
            return 0
        ingested = 0
        if signals:
            results = await self.handle_signals(signals)
            ingested = sum(1 for r in results if r is not None)
            logger.info(
                "[CacheFirstHydration] %d persistent red(s) from pytest cache "
                "-> %d stable signal(s) ingested (merkle-diff-independent)",
                len(failures), len(signals),
            )
        return ingested

    async def hydrate_on_boot(self) -> int:
        """Reconstruct + scope-run tests for pre-boot working-tree changes.

        Ground-truth recovery of the offline-state blindspot: enumerate
        uncommitted ``.py`` changes via ``TestWatcher.diff_working_tree``
        (async ``git diff --name-only HEAD``), resolve each through the SAME
        ``resolve_affected_tests`` mapper the live FS path uses, run the
        localized SCOPED pytest (NEVER the whole ``tests/`` suite), and ingest
        any resulting stable signals. Each hydrated file is recorded so a
        later live ``fs.changed`` for it is de-duped.

        Returns the number of stable signals ingested. Gated
        ``JARVIS_TESTWATCHER_BOOT_HYDRATION_ENABLED`` (default true); OFF /
        no watcher / clean tree -> returns 0 with no side effects. Fail-soft:
        any error logs at DEBUG and returns the count so far -- boot is never
        crashed.
        """
        if self._watcher is None:
            return 0
        ingested = 0
        # Test-Cache-First layer (independent gate): seed persistent reds the
        # working-tree/Merkle hash diff cannot see (walk_changed=0), BEFORE the
        # git-diff hydration consults any file hash.
        try:
            ingested += await self.hydrate_from_pytest_cache()
        except Exception:
            logger.debug("[BootHydration] cache-first layer failed", exc_info=True)
        # Working-tree differential hydration (git diff HEAD) — its own gate.
        if not boot_hydration_enabled():
            self._boot_hydrated = True
            return ingested
        try:
            changed = await self._watcher.diff_working_tree()
        except Exception:
            logger.debug("[BootHydration] diff_working_tree failed", exc_info=True)
            return ingested
        if not changed:
            logger.debug("[BootHydration] clean working tree -- nothing to hydrate")
            self._boot_hydrated = True
            return ingested

        now = time.monotonic()
        for rel in changed:
            try:
                targets = await self._resolve_scoped_targets(rel)
            except Exception:
                logger.debug(
                    "[BootHydration] resolve failed for %r", rel, exc_info=True
                )
                continue
            if not targets:
                # No scoped targets -> skip (never the 180s whole-suite sweep).
                logger.debug(
                    "[BootHydration] no scoped targets for %r -- skipping", rel
                )
                continue
            # Record BEFORE running so a concurrent live event is de-duped.
            self._hydrated_keys[rel] = now
            try:
                async with self._poll_once_lock:
                    signals = await self._watcher.poll_once(
                        target_paths=targets
                    )
            except Exception:
                logger.debug(
                    "[BootHydration] scoped poll failed for %r", rel, exc_info=True
                )
                continue
            if signals:
                results = await self.handle_signals(signals)
                ingested += sum(1 for r in results if r is not None)
                logger.info(
                    "[BootHydration] %r -> %d scoped target(s), %d stable "
                    "signal(s) ingested (NO fs.changed event needed)",
                    rel, len(targets), len(signals),
                )
        self._boot_hydrated = True
        if ingested:
            logger.info(
                "[BootHydration] boot hydration complete: %d stable failure(s) "
                "recovered from working tree (%d changed .py file(s))",
                ingested, len(changed),
            )
        return ingested

    # ------------------------------------------------------------------
    # Event-driven path (Manifesto §3: zero polling, pure reflex)
    # ------------------------------------------------------------------

    async def subscribe_to_bus(self, event_bus: Any) -> None:
        """Subscribe to file-system events via ``TrinityEventBus``.

        Gated by ``JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED`` (default OFF).
        When the flag is off this method is a logged no-op so the legacy
        pure-poll behavior is preserved exactly (no silent regression when
        the graduation flip lands).

        When the flag is on, two event paths become active (Slice 3 gap
        #4 resolution):

        1. ``.jarvis/test_results.json`` change → structured consumption
           via the ouroboros_pytest_plugin (no subprocess spawn).
        2. ``*.py`` change → debounced (2s) subprocess pytest run
           reusing ``TestWatcher.poll_once``.

        Caller contract: ``IntakeLayerService`` unconditionally calls
        ``subscribe_to_bus`` on every sensor that exposes it. The flag
        check lives here so one sensor's decision doesn't require
        special-casing at the call site.
        """
        # Initialize unconditionally so the rest of the class can reference
        # it without AttributeError regardless of whether the flag flipped
        # subscription on. (``_last_plugin_ts`` is initialized in
        # ``__init__`` — see Slice 5 F4 note there; not re-initialized here
        # to keep the fresh-parse bump its sole other write site.)
        self._debounce_task: Optional[asyncio.Task] = None

        if not self._fs_events_mode:
            logger.debug(
                "TestFailureSensor: FS-event subscription skipped "
                "(JARVIS_TEST_FAILURE_FS_EVENTS_ENABLED=false). "
                "Poll-primary mode active — no gap #4 resolution.",
            )
            return

        try:
            await event_bus.subscribe("fs.changed.*", self._on_fs_event)
        except Exception as exc:
            # Subscription failure must not break intake boot. Falls back
            # to pure poll (at the demoted fallback interval, since flag
            # is still on — operator intent preserved).
            logger.warning(
                "TestFailureSensor: FS-event subscription failed: %s "
                "(poll-fallback at %ds continues)",
                exc, int(_TEST_FAILURE_FALLBACK_INTERVAL_S),
            )
            return

        logger.info(
            "TestFailureSensor: subscribed to fs.changed.* — "
            "FS events now PRIMARY (poll demoted to %ds fallback)",
            int(_TEST_FAILURE_FALLBACK_INTERVAL_S),
        )

        # Boot-marker -- the subscription is now LIVE. The Chaos Readiness
        # Handshake (and operators) grep this exact line to know the bus +
        # TestWatcher are listening before any mutation. Emitted to stdout
        # (flushed) so it lands in the soak log the external probe tails, and
        # mirrored to the INFO log. Fail-soft: a print failure never breaks boot.
        try:
            import sys as _sys

            # Cockpit silence (2026-07-18): the raw-stdout marker exists
            # for soak harness greps; on the operator's product surface
            # it is boot noise that stomps the awakening Live region.
            # The logger.info below is the cockpit-safe twin (the
            # ERROR-only console threshold absorbs it; session logs
            # keep it).
            from backend.core.ouroboros.ui.presentation_mode import (
                is_cockpit as _is_cockpit,
            )
            if not _is_cockpit():
                print(TESTWATCHER_READY_MARKER, file=_sys.stdout, flush=True)
        except Exception:  # noqa: BLE001 -- marker is best-effort
            pass
        logger.info("%s", TESTWATCHER_READY_MARKER)

        # Boot-Time Differential Hydration -- reconstruct any pre-boot working-
        # tree mutation from ground truth NOW that the subscription is live (so
        # a file later touched live is de-duped). Gated + fail-soft inside
        # ``hydrate_on_boot``; a missed FS event (chaos injected before boot,
        # crash/restart) is no longer lost.
        try:
            await self.hydrate_on_boot()
        except Exception:  # noqa: BLE001 -- hydration must never break intake boot
            logger.debug("TestFailureSensor: boot hydration error", exc_info=True)

    async def _on_fs_event(self, event: Any) -> None:
        """Route events: test_results.json → instant consume; .py → debounced pytest."""
        rel_path = event.payload.get("relative_path", "")

        # Phase 2: ouroboros_pytest_plugin results file
        if rel_path.endswith("test_results.json") and ".jarvis" in rel_path:
            self._fs_events_handled += 1
            await self._on_test_results_changed(event)
            return

        # Phase 1 fallback: .py changes → debounced subprocess
        if event.payload.get("extension") != ".py":
            self._fs_events_ignored += 1
            return
        self._fs_events_handled += 1
        self._pending_changed_paths.add(rel_path)
        if self._debounce_task is not None and not self._debounce_task.done():
            return  # window open — accumulate, NEVER cancel/evict (Slice 5 F2)
        self._debounce_task = asyncio.create_task(
            self._debounced_pytest_run(),
            name="test_failure_debounced_run",
        )

    # ------------------------------------------------------------------
    # Phase 2: Structured results from ouroboros_pytest_plugin
    # ------------------------------------------------------------------

    async def _on_test_results_changed(self, event: Any) -> None:
        """Consume .jarvis/test_results.json written by the pytest plugin.

        Slice 5 F4 (Run #15 L3): the sensor once consumed a DELETED/stale
        results file (parse of a missing file -> "0 failures") and still
        armed the 10s plugin-suppression window at exactly the wrong
        moment. This gate is structural: the file's ``st_mtime`` must be
        >= ``max(watcher.last_pytest_spawn_walltime, boot_walltime) - 1.0``
        (1s slack for coarse FS timestamps) or the event is ignored
        entirely — ``_last_plugin_ts`` is untouched.
        """
        path = event.payload.get("path", "")
        floor_ts = max(
            getattr(self._watcher, "last_pytest_spawn_walltime", 0.0),
            self._boot_walltime,
        ) - 1.0  # 1s slack for coarse FS timestamps
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            logger.debug(
                "TestFailureSensor: results file absent/unreadable (%r) — "
                "ignored, suppression NOT armed (Slice 5 F4)", path,
            )
            return
        if mtime < floor_ts:
            logger.debug(
                "TestFailureSensor: STALE plugin results ignored "
                "(mtime %.1fs before run/boot floor) — suppression NOT armed",
                floor_ts - mtime,
            )
            return

        failures = self._parse_results_file(path)
        if failures is None:
            # Slice 5 F4: mtime was fresh but the payload itself was
            # unparseable (truncated write, mid-write torn read, corrupt
            # JSON). Not a "fresh, parseable read" — suppression must NOT
            # arm on garbage.
            logger.debug(
                "TestFailureSensor: fresh results file unparseable (%r) — "
                "ignored, suppression NOT armed (Slice 5 F4)", path,
            )
            return

        if self._watcher is not None:
            # C1: process_failures runs a synchronous ~7s repo-wide rglob
            # (build_module_to_path) in-loop via the attribution path. This
            # is the ONLY site that calls process_failures directly (all
            # other sites route through poll_once, which pre-warms itself);
            # pre-warm the module-map cache OFF the loop here, red-cycles
            # only. Fail-soft.
            if failures and attribution_enabled():
                await prewarm_module_map(self._watcher.repo_path)
            signals = self._watcher.process_failures(failures)
            if signals:
                logger.info(
                    "TestFailureSensor: plugin results → %d stable signals",
                    len(signals),
                )
                await self.handle_signals(signals)
            else:
                logger.debug(
                    "TestFailureSensor: plugin results consumed "
                    "(%d failures, no stable signals yet)",
                    len(failures),
                )

        self._last_plugin_ts = time.monotonic()

    def _parse_results_file(self, path: str) -> Optional[List[TestFailure]]:
        """Parse the JSON results file into TestFailure objects.

        Returns ``None`` (distinct from ``[]``) when the payload itself
        could not be read/decoded — Slice 5 F4's staleness-gate bump must
        distinguish "fresh file, genuinely zero failures" from "fresh
        file, garbage payload" (the latter must NOT arm suppression).
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.debug("TestFailureSensor: failed to read results file: %s", exc)
            return None

        if data.get("schema_version") != 1:
            logger.debug(
                "TestFailureSensor: unknown schema_version %s",
                data.get("schema_version"),
            )
            return []

        # Staleness check — ignore results older than 60s
        ts = data.get("timestamp", 0)
        if time.time() - ts > 60.0:
            logger.debug("TestFailureSensor: stale results file (%.0fs old)", time.time() - ts)
            return []

        failures: List[TestFailure] = []
        for entry in data.get("failures", []):
            nodeid = entry.get("nodeid", "")
            file_path = entry.get("file_path", nodeid.split("::")[0])
            error_text = entry.get("error_text", "")
            failures.append(TestFailure(
                test_id=nodeid,
                file_path=file_path,
                error_text=error_text,
            ))

        return failures

    # ------------------------------------------------------------------
    # Phase 1 fallback: debounced subprocess pytest run
    # ------------------------------------------------------------------

    async def _debounced_pytest_run(self, changed_rel_path: str = "") -> None:
        """Fixed-window set-accumulator debounce (Slice 5 F2).

        The first .py event opens a ``_debounce_window_s()`` window; every
        event during it adds to ``_pending_changed_paths``. One scoped run
        covers the UNION of resolved targets. Paths arriving mid-run land
        in the (fresh) set and re-arm one follow-up window — nothing is
        evicted (Run #15 L2: last-event-wins cancel dropped the chaos leaf
        under a worktree burst). *changed_rel_path* kept for back-compat:
        a direct caller's path seeds the set.
        """
        cancelled = False
        try:
            if changed_rel_path:
                self._pending_changed_paths.add(changed_rel_path)
            await asyncio.sleep(_debounce_window_s())
            batch = sorted(self._pending_changed_paths)
            self._pending_changed_paths.clear()
            if not batch:
                return
            # Suppress if plugin results were consumed recently (Phase 2 active)
            if time.monotonic() - self._last_plugin_ts < 10.0:
                logger.debug(
                    "TestFailureSensor: deferring subprocess run — "
                    "plugin results consumed %.1fs ago",
                    time.monotonic() - self._last_plugin_ts,
                )
                # Review fix (Slice 5 T2 Critical): the batch was already
                # drained — re-seed it so the finally re-arm schedules a
                # follow-up window instead of silently losing the paths.
                # Bounded self-healing: the chain loops at window cadence
                # until the 10s suppression expires, then runs.
                self._pending_changed_paths.update(batch)
                return
            if self._watcher is None:
                # Same re-seed contract: never silently lose a drained batch.
                self._pending_changed_paths.update(batch)
                return

            # Boot-hydration de-dupe: a file just reconstructed from the
            # working tree on boot must not be re-run by the live event that
            # the same edit also triggers. The TTL window expires so a genuine
            # later edit still re-runs. Filtered paths are dropped BY DESIGN
            # (the hydration run already covered them) — but named, never
            # silent (review fix: restores the pre-F2 per-path debug log).
            hydration_filtered = [p for p in batch if self._is_recently_hydrated(p)]
            if hydration_filtered:
                logger.debug(
                    "TestFailureSensor: suppressing live run for %r -- "
                    "hydrated on boot within de-dupe window",
                    hydration_filtered,
                )
                batch = [p for p in batch if p not in hydration_filtered]
            if not batch:
                return

            # No-silent-cap: bound the batch, but name every dropped path.
            cap = _debounce_max_paths()
            if len(batch) > cap:
                logger.warning(
                    "TestFailureSensor: debounce batch %d paths > cap %d — "
                    "resolving first %d, dropped: %s",
                    len(batch), cap, cap, batch[cap:],
                )
                batch = batch[:cap]

            if not dynamic_scoping_enabled():
                # OFF -> byte-identical legacy whole-suite behavior.
                async with self._poll_once_lock:
                    signals = await self._watcher.poll_once()
                if signals:
                    await self.handle_signals(signals)
                return

            union = await self._resolve_union(batch)
            if union:
                logger.info(
                    "TestFailureSensor: scoped %d test target(s) for %d "
                    "changed path(s): %r",
                    len(union), len(batch), batch,
                )
                await self._run_scoped_with_confirmation(union)
                return
            if not full_suite_fallback_enabled():
                logger.debug(
                    "TestFailureSensor: no scoped targets for %r and "
                    "full-suite fallback disabled — skipping run", batch,
                )
                return
            logger.info(
                "TestFailureSensor: no scoped targets for %r — "
                "JARVIS_TEST_FULL_SUITE_FALLBACK on, running full suite",
                batch,
            )
            async with self._poll_once_lock:
                signals = await self._watcher.poll_once()
            if signals:
                await self.handle_signals(signals)
        except asyncio.CancelledError:
            cancelled = True  # sensor stopping — the finally below must NOT re-arm
            pass
        except Exception:
            logger.debug("TestFailureSensor: debounced run error", exc_info=True)
        finally:
            if not cancelled and self._pending_changed_paths and self._running:
                self._debounce_task = asyncio.create_task(
                    self._debounced_pytest_run(),
                    name="test_failure_debounced_run",
                )

    async def _resolve_union(self, rel_paths: Sequence[str]) -> list:
        """Resolve each path via the T3 scoped resolver; ordered dedup union."""
        union: list = []
        seen: Set[str] = set()
        for rel in rel_paths:
            targets = await self._resolve_scoped_targets(rel)
            for t in targets or ():
                if t not in seen:
                    seen.add(t)
                    union.append(t)
        return union

    async def _run_scoped_with_confirmation(self, targets: Any) -> None:
        """Scoped run + bounded immediate confirmation (fs.changed path).

        The stability gate needs two consecutive failing observations; an
        fs.changed detection used to provide only ONE, leaving emission to
        the 600s poll fallback — which the A1 soak wall outlives
        (a1-brain-20260706-014931: the vector starved at streak 1 all
        session). When the first scoped run observes a NEW failure without
        emitting, re-run the SAME scoped targets once, immediately: a
        deterministic RED reaches the gate in seconds; a flake that greens
        on the re-run is correctly absorbed (that is the gate's purpose).
        Bounded to one re-run; gated by fs_confirm_enabled().

        Review fix (T5 Important): the streak snapshot (``pre``), both
        poll_once calls, and the streak read-back are ONE critical section
        under ``_poll_once_lock`` — an interleaved run from the other
        trigger path (debounce vs quiet-lane reconcile) mutates
        ``_failure_streak`` mid-flight and corrupts the gate accounting.
        Callers must NOT already hold the lock (it is not reentrant)."""
        async with self._poll_once_lock:
            pre = dict(self._watcher._failure_streak)
            signals = await self._watcher.poll_once(target_paths=targets)
            if not signals and fs_confirm_enabled():
                streak = self._watcher._failure_streak
                observed_new_failure = any(
                    streak.get(test_id, 0) > pre.get(test_id, 0)
                    for test_id in streak
                )
                if observed_new_failure:
                    logger.info(
                        "TestFailureSensor: scoped RED below stability gate — "
                        "immediate confirmation re-run (%d target(s))",
                        len(targets),
                    )
                    signals = await self._watcher.poll_once(
                        target_paths=targets
                    )
        if signals:
            await self.handle_signals(signals)

    # ------------------------------------------------------------------
    # Dynamic test scoping — changed file -> scoped test targets
    # ------------------------------------------------------------------

    def _repo_root(self) -> Path:
        """Authoritative repo root for the resolver / mirror-dir fallback.

        Prefers the watcher's ``repo_path`` (which is itself now the
        ``.git``-anchored :func:`resolve_repo_root` value, not bare ``"."``),
        and falls back to the SAME canonical resolver — never to a raw ``"."``
        / CWD that could disagree with where the changed-file paths anchor.
        This single-source-of-truth is the run-#12 path-mismatch fix: the
        boot-hydration ``git diff`` cwd, the ``TestRunner`` repo_root, and the
        ``_normalize`` relative-to base are now one and the same anchor, so a
        valid ``tests/...py`` is never rejected as "outside repo root" and the
        scope never falls back to the 180s whole-suite sweep. Never raises.
        """
        candidate = getattr(self._watcher, "repo_path", None)
        try:
            if candidate:
                return Path(candidate).resolve()
        except Exception:  # noqa: BLE001 — fall through to the canonical resolver
            pass
        try:
            return resolve_repo_root()
        except Exception:  # noqa: BLE001 — defensive; resolver is fail-soft already
            return Path(".").resolve()

    async def _resolve_scoped_targets(
        self, changed_rel_path: str
    ) -> Optional[List[str]]:
        """Map *changed_rel_path* -> bounded scoped pytest targets.

        Slice 4 T3: the actual resolution logic now runs OFF the asyncio
        loop, in the ``cooperative_fs_io`` thread-pool worker (see
        :func:`_resolve_scoped_targets_sync`) — the Run #14 tombstone was
        this coroutine's ``(repo_root / changed_rel_path).resolve()``
        blocking the main thread for 83 minutes. Signature and return
        contract are unchanged.

        Master-off degrade (re-review Critical fix): when
        ``JARVIS_COOPERATIVE_FS_IO_ENABLED=false``, ``offload()`` runs the
        worker INLINE on this loop thread, where the worker's
        ``asyncio.run`` would raise and silently kill the primary AST
        resolver. So under master-off the primary is awaited NATIVELY here
        (same coroutine, on-loop — the byte-identical pre-slice
        semantics), and its result is injected into the shared sync
        fallback ladder, which runs inline as master-off intends.

        Returns
        -------
        * A non-empty ``List[str]`` of scoped test paths when the existing
          ``TestRunner.resolve_affected_tests`` (or the bounded mirror-dir
          fallback) yields targets.
        * ``None`` when nothing resolved — the caller then either skips the
          run or (opt-in) falls back to the whole suite. **Never** returns
          the whole ``tests/`` directory implicitly.

        Fail-safe by construction: any error in the resolver — including an
        ``_offload_fs`` substrate fault — degrades to ``None``, same as the
        sync path's own internal fail-safe ladder.
        """
        if not changed_rel_path:
            return None
        try:
            repo_root = self._repo_root()
            if _fs_io_inline_mode():
                # Master-off: byte-identical on-loop degrade. Await the
                # shared primary natively; run the shared fallback ladder
                # inline with the result injected (Mandate 3: one
                # resolution sequence, two primary execution strategies).
                changed_abs = (repo_root / changed_rel_path).resolve()
                primary = await _resolve_primary(
                    repo_root, changed_abs, changed_rel_path,
                )
                return _resolve_scoped_targets_sync(
                    repo_root, changed_rel_path, primary,
                )
            return await _offload_fs(
                _resolve_scoped_targets_sync, repo_root, changed_rel_path,
            )
        except Exception:  # noqa: BLE001 — resolver is best-effort (existing contract)
            return None

    @staticmethod
    def _is_repo_test_root(path: Path, repo_root: Path) -> bool:
        """True if *path* is a top-level repo test dir (the whole-suite root).

        The resolver's strategy-4/last-resort returns the repo-level
        ``tests/`` dir, which is exactly the whole-suite sweep we are
        avoiding. Treat it as "did not scope" so the caller can fall to the
        bounded mirror-dir or skip — never run it implicitly.

        Slice 4 T3: also called from the module-level off-loop resolver
        (:func:`_resolve_scoped_targets_sync`, :func:`_mirror_tests_dir_sync`)
        — kept ``@staticmethod`` so both the instance and the worker-thread
        sync path can call it without an instance.
        """
        try:
            from backend.core.ouroboros.governance.test_runner import (
                _TEST_DIR_NAMES,
            )
        except Exception:
            _TEST_DIR_NAMES = frozenset({"tests", "test"})
        try:
            rp = path.resolve()
        except Exception:
            rp = path
        return rp.parent == repo_root and rp.name in _TEST_DIR_NAMES

    # ------------------------------------------------------------------
    # Poll fallback (safety net when event spine is unavailable)
    # ------------------------------------------------------------------

    def _event_primary_derate(self) -> bool:
        """Consult the SINGLE-SOURCE derate decision each cycle (dynamic).

        Slice 4 T3 (production path): delegates to
        ``test_watcher.event_primary_derate()`` — event lane armed AND the
        ``JARVIS_INTENT_POLL_WHEN_EVENT_PRIMARY`` escape hatch unset. Never
        raises: any fault (import, env parse) degrades to ``False`` so the
        poll safety-net keeps running rather than silently going dark.
        """
        try:
            from backend.core.ouroboros.governance.intent.test_watcher import (
                event_primary_derate,
            )
            return event_primary_derate()
        except Exception:  # noqa: BLE001 — fail-open to legacy polling
            logger.debug(
                "TestFailureSensor: event_primary_derate consult failed — "
                "keeping legacy poll", exc_info=True,
            )
            return False

    async def _poll_loop(self) -> None:
        """Poll loop — primary when FS events are disabled, fallback when on.

        Uses the shorter ``TestWatcher.poll_interval_s`` (default 30s)
        when ``_fs_events_mode`` is False. When the flag is on, demotes
        to ``JARVIS_TEST_FAILURE_FALLBACK_INTERVAL_S`` (default 600s) so
        the FS subscription carries the hot path and this loop only
        catches missed FS events.

        Slice 4 T3 (Run #14 / Run #15 gate): when the event-primary lane
        is armed AND the operator has NOT set the
        ``JARVIS_INTENT_POLL_WHEN_EVENT_PRIMARY`` escape hatch, the
        whole-suite ``poll_once()`` sweep is fully SKIPPED for that cycle
        (not merely interval-demoted) — Run #14 had the 300s sweep SIGKILL
        at the 180s pytest ceiling 7/7 times, starving the box the event
        lane needed while producing zero signals. Consulted EACH cycle
        (dynamic, unlike the init-cached ``_fs_events_mode``) so operators
        can flip lanes live. The ``_fs_events_mode`` interval demotion is
        retained for the forced/legacy modes.
        """
        while self._running and self._watcher is not None:
            if self._event_primary_derate():
                # Log once per state-change, not per cycle.
                if not self._poll_derate_logged:
                    logger.debug(
                        "[TestFailureSensor] event-primary lane armed — "
                        "skipping legacy whole-suite poll each cycle "
                        "(JARVIS_INTENT_POLL_WHEN_EVENT_PRIMARY=true to "
                        "force)."
                    )
                    self._poll_derate_logged = True
                lane_counter = self._fs_events_handled + self._fs_events_ignored
                try:
                    await asyncio.sleep(_TEST_FAILURE_FALLBACK_INTERVAL_S)
                except asyncio.CancelledError:
                    break
                # Slice 5 F5 (Run #15 L4): the derate skip is only safe when
                # the event lane demonstrably delivers. Zero events across a
                # whole fallback window -> one bounded git-dirty-scoped
                # reconcile (NEVER the whole suite — T3 anti-storm holds).
                if (
                    quiet_reconcile_enabled()
                    and self._running
                    and (self._fs_events_handled + self._fs_events_ignored)
                    == lane_counter
                ):
                    logger.info(
                        "TestFailureSensor: event lane delivered ZERO events "
                        "in %.0fs window — quiet-lane reconcile (bounded, "
                        "git-dirty scoped)",
                        _TEST_FAILURE_FALLBACK_INTERVAL_S,
                    )
                    try:
                        await self._reconcile_quiet_lane()
                    except Exception:
                        logger.debug(
                            "TestFailureSensor: quiet-lane reconcile error",
                            exc_info=True,
                        )
                continue
            if self._poll_derate_logged:
                logger.debug(
                    "[TestFailureSensor] event-primary derate lifted — "
                    "legacy poll resumed."
                )
                self._poll_derate_logged = False
            try:
                async with self._poll_once_lock:
                    signals = await self._watcher.poll_once()
                if signals:
                    await self.handle_signals(signals)
            except Exception:
                logger.exception("TestFailureSensor: poll error")
            # Resolve the interval per-iteration so a mid-flight flag
            # flip (restart preferred, but runtime change is safe too)
            # takes effect on the next wait.
            effective_interval = (
                _TEST_FAILURE_FALLBACK_INTERVAL_S
                if self._fs_events_mode
                else self._watcher.poll_interval_s
            )
            try:
                await asyncio.sleep(effective_interval)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Quiet-lane reconcile (Slice 5 T5, Run #15 L4, F5/F6)
    # ------------------------------------------------------------------

    async def _git_dirty_py_paths(self) -> list:
        """Tracked-dirty ``.py`` files via async git — bounded, never raises.

        Uses ``git status --porcelain=v1 --untracked-files=no`` (not the
        boot-hydration path's ``git diff --name-only HEAD``) so staged AND
        unstaged tracked changes both surface — the reconcile's job is to
        catch whatever is dirty right now, not diff against a specific ref.
        Rename/copy lines (``R  old.py -> new.py``) are parsed by splitting
        on `` -> `` and keeping the NEW path — a naive ``line[3:]`` slice
        would yield the compound ``"old.py -> new.py"`` which itself ends
        in ``.py`` and would silently corrupt the path list rather than
        crash, so it must be split explicitly rather than trusted to fail
        safe. 10s bound; on timeout the subprocess is killed and reaped via
        ``wait()`` so no zombie is left behind.
        """
        proc = None
        try:
            # Review fix (T5 Critical): ``self._repo`` is a repo LABEL
            # ("jarvis"/"prime"/…) in production wiring, not a filesystem
            # path — anchoring git there raises and silently no-ops the
            # whole reconcile. ``_repo_root()`` is the file's authoritative
            # ``.git``-anchored path (same fix class as run #12's resolver
            # anchoring).
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain=v1", "--untracked-files=no",
                cwd=str(self._repo_root()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                # The timeout and a natural exit can land in the same tick:
                # `communicate()` was cancelled but the process had already
                # gone, and `kill()` on a reaped process raises
                # ProcessLookupError -- three tracebacks per soak for a scan
                # that had, in fact, finished. Only kill what is still alive,
                # and treat "already gone" as the success it is.
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await proc.wait()
                return []
        except Exception:
            logger.debug("TestFailureSensor: git dirty scan failed", exc_info=True)
            return []
        dirty: list = []
        for line in out.decode(errors="ignore").splitlines():
            p = line[3:].strip()
            if " -> " in p:
                # Rename/copy line — keep the NEW (post-rename) path, the
                # one that actually exists on disk right now.
                p = p.split(" -> ", 1)[1].strip()
            # git quotes porcelain paths containing spaces/special chars
            # (e.g. `"a b.py"`) — strip the surrounding double-quotes so
            # they aren't silently dropped by the .endswith(".py") check
            # below. Basic unescape only: a full C-style unescape (git's
            # \NNN octal + \\/\" escapes for core.quotePath) is NOT
            # attempted here — pathological filenames with embedded
            # backslash-escapes may still round-trip imperfectly.
            if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
                p = p[1:-1]
            if p.endswith(".py"):
                dirty.append(p)
        return dirty

    async def _reconcile_quiet_lane(self) -> None:
        """One bounded scoped run over git-dirty paths (Slice 5 F5).

        Fired from the derate branch of ``_poll_loop`` only when a whole
        fallback window observed zero fs events. Scoped to git-dirty ``.py``
        files — never the whole suite (T3 anti-storm invariant). A clean
        tree runs nothing.
        """
        dirty = await self._git_dirty_py_paths()
        if not dirty:
            logger.debug(
                "TestFailureSensor: quiet-lane reconcile — tree clean, "
                "nothing to reconcile",
            )
            return
        cap = _debounce_max_paths()
        if len(dirty) > cap:
            logger.warning(
                "TestFailureSensor: reconcile %d dirty paths > cap %d — "
                "first %d only, dropped: %s",
                len(dirty), cap, cap, dirty[cap:],
            )
            dirty = dirty[:cap]
        union = await self._resolve_union(dirty)
        if not union:
            logger.debug(
                "TestFailureSensor: quiet-lane reconcile — no scoped "
                "targets for %r", dirty,
            )
            return
        logger.info(
            "TestFailureSensor: quiet-lane reconcile scoped %d test "
            "target(s) for %d dirty path(s): %r", len(union), len(dirty), dirty,
        )
        await self._run_scoped_with_confirmation(union)
