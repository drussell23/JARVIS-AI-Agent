"""The cold-boot wait must end on SILENCE, not on the clock.

Two defects, measured on 2026-09-05, made `ov` report a failure about a
healthy organism.

1. The wait window was the constant 120s. A cold boot was still wiring the
   cockpit at 114s. The margin was four seconds of luck.

2. The estimator that would have raised the window is fed a ledger that
   counted ATTACHES as boots. The real file held
   ``[33.9, 0.10, 32.2, 0.10]`` -- half the samples were sub-second
   connections to an organism that was already up.

The trap in "just make the number bigger": the history can never learn its
way out of it. A boot slow enough to be abandoned records NO duration, so
the ledger keeps only the boots that finished quickly and keeps predicting
them. The clock is the wrong signal. Whether the daemon is still writing is
the right one -- bounded by a ceiling that no amount of writing can move.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.cli import boot_progress as bp  # noqa: E402
from backend.core.ouroboros.cli import thin_client as tc  # noqa: E402

#: The exact ledger found on disk. Not a shape -- the file.
CONTAMINATED = [33.90652549699985, 0.10131130999980087,
                32.21258383899976, 0.10107319700000517]

DEAD_SOCKET = Path("/nonexistent-ov-boot-wait-test.sock")


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "boot_durations.json"
    p.write_text(json.dumps({"schema_version": "1.0",
                             "durations": list(CONTAMINATED)}),
                 encoding="utf-8")
    monkeypatch.setenv(bp.HISTORY_ENV, str(p))
    return p


# ---------------------------------------------------------------------------
# An attach is not a boot
# ---------------------------------------------------------------------------


def test_the_real_ledger_is_half_attaches(ledger) -> None:
    assert bp.observed_boot_durations() == [CONTAMINATED[0], CONTAMINATED[2]]


def test_an_attach_is_refused_at_the_write(ledger) -> None:
    bp.record_boot_duration(0.101)
    assert json.loads(ledger.read_text())["durations"] == CONTAMINATED, (
        "the file must be left alone, not rewritten, when nothing is recorded")


def test_a_real_boot_is_recorded_and_launders_the_file(ledger) -> None:
    bp.record_boot_duration(31.5)
    rows = json.loads(ledger.read_text())["durations"]
    assert 31.5 in rows
    assert not any(r < bp.min_boot_s() for r in rows)


def test_the_floor_is_derived_not_hardcoded_at_the_guard(monkeypatch) -> None:
    monkeypatch.setenv(bp.MIN_BOOT_ENV, "10")
    assert bp.min_boot_s() == 10.0
    monkeypatch.setenv(bp.MIN_BOOT_ENV, "not-a-number")
    assert bp.min_boot_s() == 2.0


def test_attaches_cannot_evict_genuine_boots(tmp_path, monkeypatch) -> None:
    """Only the last `max_samples()` are kept. Without the floor, a run of
    attaches pushes every real measurement out of the retained window."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"durations": [30.0, 31.0, 32.0]}),
                 encoding="utf-8")
    monkeypatch.setenv(bp.HISTORY_ENV, str(p))
    monkeypatch.setenv(bp.MAX_SAMPLES_ENV, "3")
    for _ in range(20):
        bp.record_boot_duration(0.1)
    assert bp.observed_boot_durations() == [30.0, 31.0, 32.0]


def test_the_estimator_refuses_rather_than_inventing(ledger) -> None:
    """Two survivors is under the three-sample minimum, so there is no
    median to report. None, not a guess."""
    assert bp.expected_boot_s() is None


# ---------------------------------------------------------------------------
# The window is derived; the ceiling is static
# ---------------------------------------------------------------------------


def test_no_history_falls_to_the_floor_not_to_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(bp.HISTORY_ENV, str(tmp_path / "absent.json"))
    monkeypatch.delenv("JARVIS_OV_BOOT_WAIT_S", raising=False)
    assert tc._boot_wait_s() == tc._BOOT_WAIT_FLOOR_S


def test_a_slow_machine_widens_its_own_window(tmp_path, monkeypatch) -> None:
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"durations": [200.0, 210.0, 220.0]}),
                 encoding="utf-8")
    monkeypatch.setenv(bp.HISTORY_ENV, str(p))
    monkeypatch.delenv("JARVIS_OV_BOOT_WAIT_S", raising=False)
    assert tc._boot_wait_s() == pytest.approx(210.0 * tc._BOOT_WAIT_TOLERANCE)


def test_the_operator_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_OV_BOOT_WAIT_S", "45")
    assert tc._boot_wait_s() == 45.0


def test_the_ceiling_is_above_the_stall_window(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_OV_BOOT_CEILING_S", raising=False)
    assert tc._boot_ceiling_s(100.0) > 100.0


def test_the_ceiling_is_overridable(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_OV_BOOT_CEILING_S", "600")
    assert tc._boot_ceiling_s(100.0) == 600.0


# ---------------------------------------------------------------------------
# The wait itself
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_backoff(monkeypatch):
    """Shrink the poll cadence so a wait measured in tenths of a second
    still takes many iterations -- the loop logic is what is under test,
    not the sleeps."""
    async def _dead(*_a, **_k):
        return "dead"

    monkeypatch.setattr(tc, "_backoff_min_s", lambda: 0.001)
    monkeypatch.setattr(tc, "_backoff_max_s", lambda: 0.002)
    monkeypatch.setattr(tc, "_probe_timeout_s", lambda: 0.001)
    monkeypatch.setattr(tc, "probe_socket", _dead)


def test_a_silent_daemon_is_abandoned_at_the_stall_window(fast_backoff,
                                                          monkeypatch) -> None:
    """No log growth: the stall window must end it, not the far ceiling."""
    monkeypatch.setattr(tc, "_boot_log_mark", lambda: 4096)
    monkeypatch.setenv("JARVIS_OV_BOOT_CEILING_S", "30")
    start = time.monotonic()
    got = asyncio.run(tc.await_socket(DEAD_SOCKET, deadline_s=0.30))
    elapsed = time.monotonic() - start
    assert got is False
    assert elapsed < 5.0, "the stall window must end it, not the 30s ceiling"


def test_a_writing_daemon_is_waited_on_past_the_stall_window(
        fast_backoff, monkeypatch) -> None:
    """The 114s-boot class. The stall window is short, the daemon keeps
    writing, and the wait must outlive the window it would have died in."""
    marks = {"n": 0}

    def _mark():
        marks["n"] += 1
        return marks["n"] * 100          # always advancing

    monkeypatch.setattr(tc, "_boot_log_mark", _mark)
    monkeypatch.setenv("JARVIS_OV_BOOT_CEILING_S", "1.0")
    start = time.monotonic()
    got = asyncio.run(tc.await_socket(DEAD_SOCKET, deadline_s=0.05))
    elapsed = time.monotonic() - start
    assert got is False
    assert elapsed > 0.05 * 3, (
        "continuous progress must renew the silence clock past one window")


def test_the_ceiling_cannot_be_set_below_a_usable_boot(monkeypatch) -> None:
    """A five-second clamp, so a typo in the override cannot produce a
    ceiling no real boot could ever finish inside."""
    monkeypatch.setenv("JARVIS_OV_BOOT_CEILING_S", "0.6")
    assert tc._boot_ceiling_s(0.05) == 5.0


def test_progress_can_never_outlive_the_ceiling(fast_backoff,
                                                monkeypatch) -> None:
    """The bound a live process must not be able to move. The ceiling is
    stubbed rather than set through the env because the env path clamps at
    five seconds, and what is under test here is the LOOP, not the clamp."""
    marks = {"n": 0}

    def _mark():
        marks["n"] += 1
        return marks["n"] * 100

    monkeypatch.setattr(tc, "_boot_log_mark", _mark)
    monkeypatch.setattr(tc, "_boot_ceiling_s", lambda *_a, **_k: 0.6)
    start = time.monotonic()
    asyncio.run(tc.await_socket(DEAD_SOCKET, deadline_s=0.05))
    elapsed = time.monotonic() - start
    assert elapsed < 4.0, "an endlessly-logging boot must still be bounded"
    assert elapsed > 0.05 * 3, "and it must have been renewed while it lasted"


def test_a_rotated_log_reads_as_progress_not_as_silence(fast_backoff,
                                                        monkeypatch) -> None:
    """A fresh boot truncates the previous session's log.

    The mark drops from a large previous-session size to a few bytes and
    then climbs, never regaining the old value. Under a 'growth only' rule
    nothing after the drop counts, and the boot is abandoned one stall
    window later, stranded behind the byte count of the run before it.
    """
    t0 = time.monotonic()

    def _mark():
        elapsed = time.monotonic() - t0
        if elapsed < 0.02:
            return 50_000                    # the previous session's log
        return 12 + int(elapsed * 10_000)    # rotated, climbing, still smaller

    monkeypatch.setattr(tc, "_boot_log_mark", _mark)
    monkeypatch.setattr(tc, "_boot_ceiling_s", lambda *_a, **_k: 0.4)
    start = time.monotonic()
    asyncio.run(tc.await_socket(DEAD_SOCKET, deadline_s=0.05))
    assert time.monotonic() - start > 0.25, (
        "the rotated log must renew the wait; a growth-only rule ends it at "
        "roughly the 0.05s stall window instead of the 0.4s ceiling")


def test_an_unreadable_log_never_extends_anything(fast_backoff,
                                                  monkeypatch) -> None:
    monkeypatch.setattr(tc, "_boot_log_mark", lambda: -1)
    monkeypatch.setenv("JARVIS_OV_BOOT_CEILING_S", "30")
    start = time.monotonic()
    asyncio.run(tc.await_socket(DEAD_SOCKET, deadline_s=0.25))
    assert time.monotonic() - start < 5.0


def test_the_mark_never_raises(monkeypatch) -> None:
    def _boom():
        raise OSError("no log")

    monkeypatch.setattr(tc, "daemon_log_path", _boom)
    assert tc._boot_log_mark() == -1
