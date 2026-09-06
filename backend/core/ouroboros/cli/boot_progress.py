"""What the organism is actually doing while you wait for it.

THE DEFECT THIS REPLACES
------------------------
The cold-boot wait printed a fresh line every five seconds::

    ⎿ organism waking · 0s
    ⎿ organism waking · 5s
    ⎿ organism waking · 10s
    ⎿ organism waking · 15s
    ⎿ organism waking · 21s
    ⎿ organism waking · 27s

Six lines that say one thing: time is passing. They do not say what stage the
boot reached, whether it is progressing or wedged, or how much longer it will
take — and the repetition itself reads as a stall, which is the opposite of
the reassurance a wait indicator exists to give.

TWO SOURCES OF TRUTH, IN PRIORITY ORDER
---------------------------------------
1. **EVIDENCE — stages actually reached.** The daemon already writes its boot
   to `.jarvis/logs/ov-daemon.log`; the client already knows that path (the
   failure message points at it). Watching it costs one tail per poll and
   yields REAL progress: preflight passed, socket bound, session opened,
   sensors armed. This is measurement, not animation.

2. **ESTIMATE — how long this usually takes.** Boot durations are recorded to
   a small ledger and the median of past boots gives an expected duration.
   Exactly the pattern `session_economics.derived_cost_cap` uses for money:
   the operator's own history, not a constant somebody guessed.

Evidence outranks estimate. When both exist the bar interpolates smoothly
between confirmed stages using elapsed time, so it moves continuously without
ever claiming a stage that has not happened.

THREE HONESTY RULES
-------------------
* **Never regress.** A percentage that goes backwards destroys the one thing a
  progress bar is for. Monotonic by construction.
* **Never reach 100% before the socket is live.** The last few percent are
  reserved for the event that actually matters. A bar that sits at 100% while
  nothing happens is worse than no bar.
* **Degrade to elapsed-only rather than lie.** No history and no matched
  markers means no percentage — just the stage and the clock. A log format
  change therefore costs the bar, never its correctness.

Python 3.9+, stdlib only. Never raises: a progress indicator that can break a
boot is not worth having.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Ouroboros.BootProgress")

ENABLED_ENV = "JARVIS_OV_BOOT_PROGRESS_ENABLED"
HISTORY_ENV = "JARVIS_OV_BOOT_HISTORY_PATH"
MAX_SAMPLES_ENV = "JARVIS_OV_BOOT_HISTORY_MAX"
CEILING_ENV = "JARVIS_OV_BOOT_PROGRESS_CEILING"
#: How far past its own prediction a boot may run before the line says so.
OVERRUN_TOLERANCE_ENV = "JARVIS_OV_BOOT_OVERRUN_TOLERANCE"
OVERRUN_GRACE_ENV = "JARVIS_OV_BOOT_OVERRUN_GRACE_S"
#: The shortest duration that may be called a boot. Below it, the sample is
#: an ATTACH to an organism that was already up.
MIN_BOOT_ENV = "JARVIS_OV_MIN_BOOT_S"

_TRUTHY = ("1", "true", "yes", "on")
_DEFAULT_HISTORY = os.path.join(".jarvis", "boot_durations.json")


def _finite_seconds(value: Any) -> float:
    """A number of seconds that is safe to format. NEVER raises.

    `elapsed` arrives from a caller's clock arithmetic, and clock arithmetic
    produces NaN and infinities: a monotonic source read twice across a
    suspend, a subtraction of two unset timestamps. `int(nan)` raises
    ValueError, which is how the fallback branch — the one whose entire job is
    to guarantee this module never breaks a boot — became the only line in it
    that could. Non-finite and negative both degrade to 0.0; a wait cannot
    have run for a negative or unknowable time, and printing `0s` is honest
    where crashing is not.
    """
    try:
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")) or v < 0.0:
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def boot_progress_enabled() -> bool:
    """Master gate. Default ON. OFF restores the plain elapsed breadcrumb."""
    try:
        return os.environ.get(ENABLED_ENV, "1").strip().lower() in _TRUTHY
    except Exception:  # noqa: BLE001
        return True


def history_path() -> str:
    try:
        return os.environ.get(HISTORY_ENV, "") or _DEFAULT_HISTORY
    except Exception:  # noqa: BLE001
        return _DEFAULT_HISTORY


def max_samples() -> int:
    """Boot durations retained. Default 40 — enough for a stable median,
    short enough that a machine that got faster is reflected within days."""
    try:
        return max(3, int(os.environ.get(MAX_SAMPLES_ENV, "40")))
    except (TypeError, ValueError):
        return 40


def overrun_tolerance() -> float:
    """Fraction past the prediction before a boot is called overrun.

    PROPORTIONAL, not a fixed number of seconds: a 2s boot taking 3s is noise
    and a 90s boot taking 135s is a fact, and no single absolute grace can
    serve both. Default 0.25.
    """
    try:
        return max(0.0, float(os.environ.get(OVERRUN_TOLERANCE_ENV, "0.25")))
    except (TypeError, ValueError):
        return 0.25


def overrun_grace_s() -> float:
    """Floor under the proportional tolerance, in seconds.

    A very short prediction has a very short 25%, which would trip on ordinary
    scheduler jitter and cry overrun on a healthy boot. Default 3.0.
    """
    try:
        return max(0.0, float(os.environ.get(OVERRUN_GRACE_ENV, "3.0")))
    except (TypeError, ValueError):
        return 3.0


def progress_ceiling() -> float:
    """The highest fraction shown before the socket is confirmed live.

    Default 0.97. The remaining 3% belongs to the event that actually
    matters; a bar that sits at 100% while nothing happens has told the
    operator the boot finished when it has not."""
    try:
        v = float(os.environ.get(CEILING_ENV, "0.97"))
        return min(0.99, max(0.50, v))
    except (TypeError, ValueError):
        return 0.97


@dataclass(frozen=True)
class BootStage:
    """One observable milestone, and the log marker that proves it."""

    key: str
    label: str
    marker: str          # substring that appears in the daemon log
    weight: float = 1.0  # relative share of the boot this stage represents


#: The stages, in order. Weights are ROUGH SHARES, not durations: the ETA
#: comes from measured history, so these only decide how the bar distributes
#: itself between confirmed milestones.
#:
#: Markers are substrings of log lines the daemon already writes. That is a
#: coupling to log text, which is why an unmatched table degrades to
#: elapsed-only rather than to a wrong number — see `Progress.fraction`.
DEFAULT_STAGES: Tuple[BootStage, ...] = (
    BootStage("preflight", "preflight", "[AegisPreflight]", 1.0),
    BootStage("credentials", "credentials", "[CredentialBootstrap]", 0.5),
    BootStage("aegis", "aegis serving", "[AegisDaemon] serving on", 1.5),
    BootStage("ready", "aegis ready", "daemon ready at", 0.5),
    BootStage("session", "session open", "session=bt-", 1.0),
    BootStage("status", "cockpit wired", "StatusLineBuilder registered", 1.5),
    BootStage("sensors", "sensors arming", "IntakeLayer", 2.0),
)


def min_boot_s() -> float:
    """The shortest duration that can honestly be called a boot.

    A real boot imports the stack, binds a socket, opens a session and arms
    the sensor set; none of that happens in under a second on any hardware.
    A sub-second sample is therefore an ATTACH to an organism that was
    already running, and counting it as a boot teaches the estimator that
    booting is instant. Default 2.0 — an order of magnitude above the
    observed attach cost and an order below the observed boot cost, so the
    separation does not depend on tuning. Tunable because the floor is a
    claim about THIS machine, not a law.
    """
    try:
        v = float(os.environ.get(MIN_BOOT_ENV, "2.0"))
        return v if v > 0 else 2.0
    except (TypeError, ValueError):
        return 2.0


def observed_boot_durations(path: Optional[str] = None) -> List[float]:
    """Durations of past successful boots, seconds. NEVER raises.

    The floor is applied HERE as well as at the write, because a guard on
    the write only protects a ledger written after the guard shipped. The
    contaminated rows are already on disk, and every consumer reads through
    this function — so filtering here heals the existing history without
    rewriting the operator's file behind their back. A row below the floor
    is not deleted, it is simply not counted as a boot, and the next
    successful write drops it out of the retained window on its own.
    """
    try:
        p = path or history_path()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        floor = min_boot_s()
        out = [float(x) for x in (data.get("durations") or [])
               if isinstance(x, (int, float)) and floor < float(x) < 3600.0]
        return out
    except Exception:  # noqa: BLE001
        return []


def record_boot_duration(seconds: float, path: Optional[str] = None) -> None:
    """Append one SUCCESSFUL boot duration. NEVER raises.

    Only successes are recorded. A failed or abandoned boot has no duration —
    folding one in would teach the estimator that boots take as long as the
    operator's patience, which is the number it exists to replace.
    """
    try:
        if not (min_boot_s() < float(seconds) < 3600.0):
            # Below the floor this was an ATTACH, not a boot. Observed
            # 2026-09-05, the ledger held [33.9, 0.10, 32.2, 0.10]: half the
            # samples were sub-second connections to an organism that was
            # ALREADY running, recorded as though the whole boot had taken
            # a tenth of a second. They drag the median toward zero, and
            # because only the last `max_samples()` are kept, a few attaches
            # can evict every genuine boot from the history the estimator
            # and the wait window both read.
            return
        p = path or history_path()
        rows = observed_boot_durations(p)
        rows.append(float(seconds))
        rows = rows[-max_samples():]
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": "1.0", "durations": rows}, fh)
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001
        pass


def expected_boot_s(path: Optional[str] = None) -> Optional[float]:
    """Median of observed boots, or None when nothing has been measured.

    None rather than a default: an invented expectation produces a confident
    percentage with nothing behind it, and the operator cannot tell the
    difference. No history means no bar.
    """
    rows = sorted(observed_boot_durations(path))
    if len(rows) < 3:
        return None
    return rows[len(rows) // 2]


@dataclass
class Progress:
    """Mutable boot state. One instance per wait."""

    stages: Tuple[BootStage, ...] = DEFAULT_STAGES
    log_path: str = ""
    expected_s: Optional[float] = None
    #: Log size when this wait began. Everything before it belongs to a
    #: PREVIOUS boot and must not count toward this one.
    log_origin: int = 0
    _reached: int = 0
    _high_water: float = 0.0
    _last_stage_at: float = 0.0
    _seen: set = field(default_factory=set)
    #: Elapsed-at-arrival for each stage reached IN THIS BOOT. The basis for
    #: projecting the rest when no cross-boot history exists yet.
    _stage_times: List[float] = field(default_factory=list)

    def observe_log(self, text: str, *, now: float) -> None:
        """Fold in whatever the daemon has written so far. NEVER raises."""
        try:
            for i, st in enumerate(self.stages):
                if st.key in self._seen:
                    continue
                if st.marker and st.marker in text:
                    self._seen.add(st.key)
                    if i + 1 > self._reached:
                        self._reached = i + 1
                        self._last_stage_at = now
                        self._stage_times.append(float(now))
        except Exception:  # noqa: BLE001
            pass

    def _raw_projection(self) -> Optional[float]:
        """This boot's own projected total — UNCLAMPED. NEVER raises.

        Split out from :meth:`_projected_total_s` because that method answers
        an ETA question and therefore clamps to ``elapsed``: an ETA may never
        point into the past. The clamp is right for a countdown and fatal for
        an overrun test, because the moment a boot runs long the clamped value
        starts tracking ``elapsed`` and the two become equal by construction —
        the evidence that the estimate was exceeded is destroyed by the very
        function that computed it. One number cannot answer both "when will
        this finish" and "has this taken longer than predicted".
        """
        try:
            if self._reached <= 0 or not self._stage_times:
                return None
            last = float(self._stage_times[-1])
            if last <= 0:
                return None
            per_stage = last / float(self._reached)
            if per_stage <= 0:
                return None
            return per_stage * float(len(self.stages))
        except Exception:  # noqa: BLE001
            return None

    def _eta_horizon(self, elapsed: float) -> Optional[float]:
        """THE horizon an ETA counts down from — history first, else cadence.

        Extracted because :meth:`fraction` and :meth:`render` each spelled
        ``self.expected_s or self._projected_total_s(elapsed)`` independently.
        Two copies of a precedence rule is two rules, and the one that drifts
        is found by an operator watching a bar disagree with its own ETA.
        """
        horizon = self.expected_s
        if not horizon or horizon <= 0:
            horizon = self._projected_total_s(elapsed)
        return horizon if horizon and horizon > 0 else None

    def overrun_s(self, elapsed: float) -> Optional[float]:
        """Seconds this boot is past its own prediction, or ``None``.

        Uses the UNCLAMPED basis (:meth:`_raw_projection`) with the same
        history-first precedence an ETA uses, so the two never contradict.

        A tolerance stands between "late" and "overrun", and it is
        proportional rather than a fixed number of seconds: a 2s boot that
        takes 3s is noise, a 90s boot that takes 135s is a fact, and the same
        absolute grace cannot serve both. The floor keeps a very short
        prediction from tripping on jitter. Both are env-tunable
        (``JARVIS_OV_BOOT_OVERRUN_TOLERANCE`` / ``_GRACE_S``) because the
        right patience on a cold laptop is not the right patience in CI.

        Returns None when there is no prediction to exceed — an unmeasured
        boot cannot be late, and claiming otherwise would invent a deadline.
        """
        try:
            basis = self.expected_s
            if not basis or basis <= 0:
                basis = self._raw_projection()
            if not basis or basis <= 0:
                return None
            allowance = max(overrun_grace_s(), basis * overrun_tolerance())
            over = float(elapsed) - (float(basis) + allowance)
            return over if over > 0.0 else None
        except Exception:  # noqa: BLE001
            return None

    @property
    def awaiting_label(self) -> str:
        """The stage that has NOT arrived — what the wait is actually on.

        ``stage_label`` names the last milestone REACHED, which during an
        overrun is the least useful thing on screen: it reports what already
        succeeded while the operator is asking what is stuck.
        """
        try:
            if self._reached >= len(self.stages):
                return ""
            return self.stages[self._reached].label
        except Exception:  # noqa: BLE001
            return ""

    def _projected_total_s(self, elapsed: float) -> Optional[float]:
        """Expected total boot time, projected from THIS boot's own cadence.

        The first boots on a machine have no history, so `expected_s` is None
        and the bar could only move when a marker landed — it sat frozen at
        one percentage between stages, which is what the operator sees as a
        hang. But a boot in progress is already evidence about itself: if four
        stages arrived over twelve seconds, the remaining three will plausibly
        take about nine more.

        This is measurement, not a constant — it adapts to a slow disk, a cold
        page cache or a loaded machine, and it needs no prior run.

        Rate measured FROM THE START OF THE WAIT, not between arrivals. The
        interval form needed two distinct timestamps and returned nothing when
        several markers landed in the same poll — the common case on a fast
        boot, since the client reads the log tail every quarter second and a
        burst of stages can complete between two reads. Anchoring on t=0 makes
        a single arrival sufficient: two stages reached by second four is two
        seconds a stage, and that is a rate, not a point.

        The math lives in :meth:`_raw_projection`; this is that value made
        SAFE FOR A COUNTDOWN. Never project a finish already in the past: an
        ETA of "0s left" that keeps not arriving is worse than no ETA. Callers
        asking whether the estimate was EXCEEDED must use the raw projection —
        the clamp here makes overrun undetectable by design.
        """
        try:
            projected = self._raw_projection()
            if projected is None:
                return None
            return max(elapsed, projected)
        except Exception:  # noqa: BLE001
            return None

    @property
    def stage_label(self) -> str:
        if self._reached <= 0:
            return "igniting"
        return self.stages[min(self._reached, len(self.stages)) - 1].label

    def fraction(self, elapsed: float) -> Optional[float]:
        """Best honest estimate of completion, or None. NEVER decreases.

        Returns None when there is neither evidence nor history — the honest
        rendering of "I don't know how far along this is" is no number at all.
        """
        try:
            total_w = sum(s.weight for s in self.stages) or 1.0
            evidence = None
            if self._reached > 0:
                done = sum(s.weight for s in self.stages[:self._reached])
                evidence = done / total_w
            # Cross-boot history is the better estimator when it exists (more
            # samples, and it already knows how this machine behaves). Within
            # -boot projection is the fallback that makes a FIRST boot move.
            horizon = self._eta_horizon(elapsed)
            estimate = None
            if horizon and horizon > 0:
                estimate = elapsed / horizon

            if evidence is None and estimate is None:
                return None
            if evidence is None:
                value = estimate
            elif estimate is None:
                value = evidence
            else:
                # Evidence sets the floor; the estimate may only interpolate
                # ABOVE it, and never past the next unconfirmed stage. That is
                # what keeps the bar moving between milestones without ever
                # claiming one that has not happened.
                nxt = (sum(s.weight for s in self.stages[:self._reached + 1])
                       / total_w) if self._reached < len(self.stages) else 1.0
                value = max(evidence, min(estimate, nxt))

            value = min(progress_ceiling(), max(0.0, float(value)))
            # MONOTONIC. A bar that goes backwards destroys the only thing it
            # is for, and both inputs can legitimately fall (history reloads,
            # a marker arrives late).
            self._high_water = max(self._high_water, value)
            return self._high_water
        except Exception:  # noqa: BLE001
            return None

    def render(self, elapsed: float, *, width: int = 18) -> str:
        """One line. Bar + stage + clock + ETA when known. NEVER raises."""
        elapsed = _finite_seconds(elapsed)
        try:
            frac = self.fraction(elapsed)
            parts = []
            if frac is not None:
                filled = int(round(frac * width))
                parts.append("[" + "█" * filled + "·" * (width - filled) + "]")
                parts.append(f"{int(frac * 100):3d}%")
            parts.append(self.stage_label)
            parts.append(f"{int(elapsed)}s")
            # An overrun used to be reported by ABSENCE: `remaining` clamped
            # to 0, the "~Ns left" token simply stopped being appended, and an
            # operator cannot see a token that is not there. Paired with the
            # 97% ceiling that is the whole defect — a slow boot and a dead
            # one rendered identically (`97%  session open  89s`), which is
            # exactly the line that sent an operator here asking if it hung.
            over = self.overrun_s(elapsed)
            if over is not None:
                parts.append(f"+{int(over)}s over")
                # Name what has NOT arrived. `stage_label` reports the last
                # milestone reached, which during an overrun answers a
                # question nobody is asking.
                waiting = self.awaiting_label
                if waiting:
                    parts.append(f"waiting on {waiting}")
            else:
                horizon = self._eta_horizon(elapsed)
                if frac is not None and horizon and frac > 0.02:
                    remaining = max(0.0, horizon - elapsed)
                    if remaining >= 1.0:
                        parts.append(f"~{int(remaining)}s left")
            return "  ".join(parts)
        except Exception:  # noqa: BLE001
            # The fallback must not be able to fail: `int(nan)` raises
            # ValueError, so the branch that exists to guarantee "NEVER
            # raises" was itself the last thing that could break a boot.
            # Sanitised HERE and not only at entry, because this handler also
            # catches failures that happen before any sanitising would run.
            return f"waking · {_finite_seconds(elapsed):.0f}s"


def log_size(path: str) -> int:
    """Current size of the daemon log, or 0. NEVER raises."""
    try:
        return os.path.getsize(path) if path and os.path.exists(path) else 0
    except Exception:  # noqa: BLE001
        return 0


def read_log_tail(path: str, *, since: int = 0, max_bytes: int = 65536) -> str:
    """Log bytes written AFTER *since*, capped at *max_bytes*. NEVER raises.

    `since` is load-bearing, not an optimisation. The daemon log is APPEND-ONLY
    ACROSS RUNS, so its tail already contains every boot marker from every
    previous boot. Reading it unanchored made a fresh wait match all seven
    stages on its first poll and render 97% instantly — a bar that is
    complete before the work starts, which is worse than no bar at all.
    Anchoring at the size observed when the wait began means only THIS boot's
    output can advance it.

    Still bounded: the log reaches tens of megabytes, and reading it whole on
    every poll would make the progress indicator the slowest thing in the boot.
    """
    try:
        if not path or not os.path.exists(path):
            return ""
        size = os.path.getsize(path)
        start = max(0, int(since))
        if size <= start:
            return ""                      # nothing new since the wait began
        start = max(start, size - max_bytes)
        with open(path, "rb") as fh:
            fh.seek(start)
            return fh.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def make_progress(log_path: str = "") -> Progress:
    """A Progress primed with this machine's history AND anchored to now.

    The anchor is taken at construction, which is the moment the wait starts —
    any later and markers from the boot's own first milliseconds would be
    skipped; any earlier and the previous boot's tail would be counted.
    """
    return Progress(log_path=log_path, expected_s=expected_boot_s(),
                    log_origin=log_size(log_path))
