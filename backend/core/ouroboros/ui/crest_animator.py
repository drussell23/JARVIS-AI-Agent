"""Client-Side Boot Animator — the WHOLE snake chases the ``+`` around the "V".

Root cause of "the snake is not moving" (operator, 2026-07-23): the first
animator rotated only the coil's GRADIENT while the sculpted head (eye, open
mouth) stayed fixed at the gap — a stationary snake with shimmering skin. The
snake must PHYSICALLY travel. This version rotates the crest's ``gap_center``
per frame, so the entire anatomy — head, mouth, eye, tapered tail, gradient —
sweeps around the ring through the crest's OWN sampling pipeline (DRY: the same
``_sample_pixel`` / ``_pixel_color_and_delay`` / ``_prune_sparse_geometry`` that
draw the static emblem draw every animation frame; nothing re-derived). The
white ``+`` prey rides IN THE GAP — just ahead of the open mouth — so the snake
visibly chases it, mouth-first, around the "V".

Architecture (advanced / asynchronous / adaptive / zero hardcoding):

  * **Managed rendering context.** No raw cursor codes: the boot is a
    ``rich.live.Live`` + ``rich.console.Group`` partition — top the animating
    crest, bottom the async wake logs. ``Live`` owns every redraw atomically.
  * **Progressive background build.** A rotated frame costs ~100-300ms of pure
    sampling, so frames are built OFF the event loop (``asyncio.to_thread``),
    one by one; the animation starts the moment the first frames exist and gains
    smoothness as the ring fills. The boot is never blocked.
  * **Persistent frame cache.** Finished rings are cached on disk keyed by
    (cols, rows, frames, ss, geometry knobs) — the SECOND boot animates at full
    smoothness instantly. Corrupt/mismatched caches are ignored and rebuilt.
  * **Adaptive duration.** The play loop runs until BOTH the daemon is up AND a
    minimum visible run has shown (warm daemons return in milliseconds — the
    original freeze bug), then settles onto the TRUE full-fidelity static crest
    (the resting emblem is byte-identical to ``print_static_crest``'s raster —
    the "looks different" fix).

Leaf module: stdlib + Rich + ui.crest. Passes the ``ui/`` no-literal-styling
guard (rgb() notation, like crest.py). Fable never referenced. Never raises.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .crest import (
    _EYE_RGB,
    _GAP_CENTER,
    _sample_coil,
    _sample_eye,
    _sample_head,
    _sample_v,
    _V_BOT_RGB,
    _V_TOP_RGB,
    _REF_ASPECT,
    _Geometry,
    _ang_norm,
    _pixel_color_and_delay,
    _prune_sparse_geometry,
    _sample_pixel,
    generate_crest_pixels,
)

# rgb() functional notation (not a colour name) — passes the ui/ no-literal-
# styling guard exactly as crest.py's per-pixel gradient styles do.
_LOG_RGB = "rgb(94,224,106)"            # venom-green boot logs

_CACHE_VERSION = 6                      # v6: resolved-knob keys (stale spun-V purge)                      # bump to invalidate stale frame caches


# ---- env knobs (no hardcoding) --------------------------------------------


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return min(hi, max(lo, int(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _fps() -> int:
    return _env_int("JARVIS_CREST_ANIM_FPS", 14, 4, 30)


def _synchronized_paint(console: Any):
    """A context-manager factory that brackets ONE paint in synchronized
    output, or does nothing when the other end is not a real terminal.

    Decided ONCE per playback, not per frame: the answer is a property of
    the terminal, and asking fourteen times a second would be waste. The
    bytes go to the console's own stream, the same one `Live` writes to,
    so BEGIN, the frame and END arrive in order on a single file. Flushed
    after END so the terminal is released to paint immediately rather than
    whenever the buffer next fills. NEVER raises; a stream that refuses the
    write simply gets an unbracketed frame, which is what it got before.
    """
    from contextlib import contextmanager
    try:
        from backend.core.ouroboros.ui.theme import (
            SYNC_BEGIN, SYNC_END, supports_synchronized_output,
        )
        is_tty = bool(getattr(console, "is_terminal", False))
        enabled = supports_synchronized_output(is_tty=is_tty)
        stream = getattr(console, "file", None)
    except Exception:  # noqa: BLE001
        enabled, stream = False, None

    @contextmanager
    def _noop():
        yield

    @contextmanager
    def _bracketed():
        try:
            stream.write(SYNC_BEGIN)
        except Exception:  # noqa: BLE001
            pass
        try:
            yield
        finally:
            try:
                stream.write(SYNC_END)
                stream.flush()
            except Exception:  # noqa: BLE001
                pass

    return _bracketed if (enabled and stream is not None) else _noop


def _frame_count_env() -> int:
    """Frames per full lap — the ring resolution of the rotation."""
    return _env_int("JARVIS_CREST_ANIM_FRAMES", 24, 4, 96)


def _anim_ss() -> int:
    """Supersampling for ANIMATION frames (the resting emblem always uses the
    static crest's full quality; frames trade a little AA for build speed)."""
    return _env_int("JARVIS_CREST_ANIM_SS", 2, 1, 5)


def _min_intro_s() -> float:
    """Minimum visible animation regardless of wake speed (the freeze-fix)."""
    return _env_float("JARVIS_CREST_ANIM_MIN_S", 3.5, 0.0, 30.0)


def _min_laps() -> float:
    return _env_float("JARVIS_CREST_ANIM_MIN_LAPS", 1.0, 0.1, 10.0)


def _v_spin_mult() -> float:
    """V revolutions per snake lap (negative = counter-spin — the gear look).
    ``0`` disables the spin. Env ``JARVIS_CREST_ANIM_V_SPIN``."""
    return _env_float("JARVIS_CREST_ANIM_V_SPIN", 0.0, -4.0, 4.0)


def _plus_lead_deg_env() -> float:
    """Angular offset of the ``+`` from the gap centre (0 = dead-centre of the
    open mouth's path — the classic prey position)."""
    return _env_float("JARVIS_CREST_ANIM_PLUS_LEAD_DEG", 0.0, -90.0, 90.0)


def _cache_dir() -> str:
    return os.environ.get(
        "JARVIS_CREST_ANIM_CACHE_DIR",
        os.path.expanduser("~/.jarvis/crest_anim"),
    )


def animator_enabled() -> bool:
    """Kill-switch: ``JARVIS_CREST_ANIM_DISABLED=1`` reverts to the static crest.
    Default ON. Never raises."""
    return os.environ.get(
        "JARVIS_CREST_ANIM_DISABLED", "",
    ).strip().lower() not in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Rotated-geometry frame builder — pure, thread-safe, composes ui.crest
# ---------------------------------------------------------------------------


def _rotated_geometry(
    cols: int, rows: int, rot: float, v_rot: float = 0.0,
) -> _Geometry:
    """The crest's geometry with the WHOLE snake rotated by ``rot`` radians —
    gap (mouth opening), head and tail all derive from ``gap_center``, so one
    rotation moves the full anatomy — and the V spun by ``v_rot`` (the
    ``_sample_v`` seam reads ``geo.v_rot``). Pure."""
    geo = _Geometry.for_size(cols, rows)
    geo.gap_center = _ang_norm(_GAP_CENTER + rot)
    geo.tail_tip = _ang_norm(geo.gap_center + geo.gap_half)
    geo.head_theta = _ang_norm(geo.gap_center - geo.gap_half)
    geo.v_rot = v_rot
    return geo


def build_rotated_frame(
    cols: int, rows: int, rot: float, ss: int, v_rot: float = 0.0,
    alpha: str = "aa",
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """Sample ONE fully-rotated frame through the crest's own pipeline
    (`_sample_pixel` → `_pixel_color_and_delay` → coverage-alpha →
    `_prune_sparse_geometry`). Pure + thread-safe (called via to_thread).
    Returns {(x, py): rgb}. Never raises — returns {} on any fault."""
    try:
        geo = _rotated_geometry(cols, rows, rot, v_rot)
        v_span = max(1e-6, geo.v_top + geo.v_bot)
        px_rows = rows * 2
        pixels: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        for py in range(px_rows):
            py0 = float(py) * _REF_ASPECT
            for x in range(cols):
                kind, coverage, theta = _sample_pixel(float(x), py0, geo, ss)
                if kind is None or coverage < 0.06:
                    continue
                py_frac = (py * _REF_ASPECT - (geo.cy - geo.v_top)) / v_span
                base, _d = _pixel_color_and_delay(kind, theta, py_frac, geo)
                if alpha == "sharp" and coverage < 0.22:
                    continue
                if alpha == "crisp":
                    # Hard pixel art (the CC-logo look): at tiny scales the
                    # coverage-alpha AA dims nearly EVERY pixel (all edges) into
                    # mud. Binary threshold + full-intensity color = crystal.
                    if coverage < 0.34:
                        continue
                    pixels[(x, py)] = tuple(
                        max(0, min(255, round(c))) for c in base
                    )
                else:
                    a = coverage ** (1.35 if alpha == "sharp" else 0.75)
                    pixels[(x, py)] = tuple(
                        max(0, min(255, round(c * a))) for c in base
                    )
        return _prune_sparse_geometry(pixels)
    except Exception:  # noqa: BLE001
        return {}


def _prey_center_px(
    cols: int, rows: int, rot: float, lead_rad: float,
) -> Tuple[int, int, float]:
    """The prey's centre in PIXEL-raster coordinates ``(x, py, scale)`` — on the
    ring path at the gap centre + lead (just ahead of the open mouth). Pure."""
    geo = _rotated_geometry(cols, rows, rot)
    theta = _ang_norm(geo.gap_center + lead_rad)
    px = geo.cx + geo.r_mid * math.cos(theta)
    py_phys = geo.cy - geo.r_mid * math.sin(theta)
    x = max(0, min(cols - 1, round(px)))
    py = max(0, min(rows * 2 - 1, round(py_phys / _REF_ASPECT)))
    return (x, py, geo.scale)


def _prey_plus_hit(dx: float, dy: float, arm: float, thick: float) -> bool:
    """The prey's SHAPE law — ONE filled plus, shared by every renderer (big
    sprite AND mini rasterizer). ``arm``/``thick`` in the caller's units; the
    predicate is unit-agnostic. Pure."""
    return (abs(dy) <= thick and abs(dx) <= arm) or (abs(dx) <= thick and abs(dy) <= arm)


def _prey_rgb(dx: float, dy: float, arm: float, glow: float) -> Tuple[int, int, int]:
    """The prey's COLOR law — pale eye-colour core fading radially to the V's
    venom purple, brightness-scaled by ``glow``. Shared by every renderer so
    both logos carry the identical prey palette. Pure."""
    core, edge, far = _EYE_RGB, _V_TOP_RGB, _V_BOT_RGB
    t = min(1.0, (abs(dx) + abs(dy)) / max(1.0, float(arm)))
    mid = edge if t < 0.75 else far
    return tuple(  # type: ignore[return-value]
        max(0, min(255, round((core[i] + (mid[i] - core[i]) * t) * glow)))
        for i in range(3)
    )


def build_prey_sprite(
    cols: int, rows: int, rot: float, lead_rad: float, pulse: float,
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """The ``+`` prey as a PIXEL sprite in the SAME half-block raster as the
    snake (root cause of the old "small, off-theme" prey: a text character in a
    pixel-art medium). A filled plus, sized by the crest's own ``scale``
    (adaptive — grows with the emblem), coloured from the crest's OWN palette
    via the shared :func:`_prey_rgb` law, with a soft pulse (``pulse`` in
    [0,1]) so it reads alive. Pure; never raises."""
    try:
        x0, py0, scale = _prey_center_px(cols, rows, rot, lead_rad)
        arm = max(2, round(1.9 * scale))          # arm length (physical px)
        thick = max(1, round(0.55 * scale))       # bar half-thickness
        glow = 0.72 + 0.28 * pulse                # brightness pulse
        sprite: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        for dx in range(-arm, arm + 1):
            for dy in range(-arm, arm + 1):
                if not _prey_plus_hit(dx, dy, arm, thick):
                    continue
                x, py = x0 + dx, py0 + dy
                if not (0 <= x < cols and 0 <= py < rows * 2):
                    continue
                sprite[(x, py)] = _prey_rgb(dx, dy, arm, glow)
        return sprite
    except Exception:  # noqa: BLE001
        return {}


def _cache_key(cols: int, rows: int, n: int, ss: int) -> str:
    """Fingerprint the RESOLVED knob values, never raw env strings — a default
    change (e.g. V-spin -1.0 → 0.0) left the env string identical ("") so the
    STALE spun-V ring kept loading forever (the arrow-V the operator saw on the
    boot emblem). Resolved values make cache identity == rendered identity."""
    try:
        from .crest import _band_strength, _coverage_threshold, _min_component_px
        knobs = (
            _CACHE_VERSION, cols, rows, n, ss,
            round(_band_strength(), 4),
            round(_coverage_threshold(), 4),
            _min_component_px(),
            round(_v_spin_mult(), 4),          # RESOLVED — defaults included
        )
    except Exception:  # noqa: BLE001
        knobs = (_CACHE_VERSION, cols, rows, n, ss, "fallback")
    return hashlib.sha256(repr(knobs).encode()).hexdigest()[:16]


def _cache_path(cols: int, rows: int, n: int, ss: int) -> str:
    return os.path.join(
        _cache_dir(), f"ring-{cols}x{rows}-n{n}-ss{ss}-{_cache_key(cols, rows, n, ss)}.pkl",
    )


def _load_cached_ring(cols: int, rows: int, n: int, ss: int) -> Optional[List[dict]]:
    """Load a finished frame ring from disk (or None). Never raises."""
    try:
        path = _cache_path(cols, rows, n, ss)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        frames = data.get("frames")
        if isinstance(frames, list) and len(frames) == n and all(
            isinstance(f, dict) for f in frames
        ):
            return frames
        return None
    except Exception:  # noqa: BLE001
        return None


def _save_cached_ring(cols: int, rows: int, n: int, ss: int, frames: List[dict]) -> None:
    """Persist a finished ring (best-effort, atomic rename). Never raises."""
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        path = _cache_path(cols, rows, n, ss)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"frames": frames}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# The animator
# ---------------------------------------------------------------------------


class CrestAnimator:
    """Physically-rotating Snake-and-Plus animation + a partitioned bottom log
    region. Frames build progressively in the background (or load from the disk
    cache); the resting emblem is the TRUE static crest. Headless-testable."""

    def __init__(
        self,
        *,
        cols: int,
        rows: int,
        plus_lead_deg: Optional[float] = None,
        log_lines: int = 6,
        frame_count: Optional[int] = None,
        ss: Optional[int] = None,
        alpha_mode: str = "aa",
    ) -> None:
        # Size exactly like the static crest (same clamps + row fit).
        self._pf = generate_crest_pixels(cols, rows)
        self._cols = self._pf.cols if self._pf else 0
        self._rows = self._pf.rows if self._pf else 0
        self._n = frame_count if frame_count is not None else _frame_count_env()
        self._ss = ss if ss is not None else _anim_ss()
        self._alpha_mode = alpha_mode if alpha_mode in ("aa", "sharp", "crisp") else "aa"
        # mode-distinct cache namespace (ss+100 per non-aa mode) — zero schema change
        self._cache_ss = self._ss if self._alpha_mode == "aa" else self._ss + 100
        self._plus_lead = math.radians(
            plus_lead_deg if plus_lead_deg is not None else _plus_lead_deg_env()
        )
        self._logs: "deque[str]" = deque(maxlen=max(1, int(log_lines)))
        #: The single live progress line, rendered under the transcript. A
        #: value, not a history — see `set_progress`.
        self._progress: str = ""
        self._lock = threading.Lock()
        # The resting emblem — the static crest's OWN raster (full fidelity).
        self._base: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        self._frames: List[Optional[dict]] = [None] * max(1, self._n)
        self._built = 0
        self._builder_started = False
        self._save_thread: Optional[threading.Thread] = None
        self._cy_lo = 0
        self._cy_hi = 0
        if self._pf:
            self._base = {k: v[0] for k, v in self._pf.pixels.items()}
            occ = [py // 2 for (_x, py) in self._base]
            self._cy_lo = min(occ) if occ else 0
            self._cy_hi = max(occ) if occ else self._rows - 1
            self._frames[0] = dict(self._base)   # frame 0 ≈ resting pose
            self._built = 1
            cached = _load_cached_ring(self._cols, self._rows, self._n, self._ss)
            if cached:
                self._frames = list(cached)
                self._built = self._n

    @property
    def available(self) -> bool:
        return bool(self._pf and self._base)

    @property
    def frames_built(self) -> int:
        return self._built

    # -- background ring builder ----------------------------------------

    async def ensure_frames(self) -> None:
        """Build the missing rotated frames OFF the event loop, one at a time
        (progressive: the animation uses each the moment it lands). Saves the
        finished ring to the disk cache. Idempotent; never raises."""
        import asyncio
        if not self.available or self._builder_started:
            return
        self._builder_started = True
        try:
            for k in range(self._n):
                if self._frames[k] is not None:
                    continue
                rot = 2.0 * math.pi * k / self._n
                frame = await asyncio.to_thread(
                    build_rotated_frame, self._cols, self._rows, rot, self._ss,
                    _v_spin_mult() * rot, self._alpha_mode,
                )
                if frame:
                    self._frames[k] = frame
                    with self._lock:
                        self._built = sum(1 for f in self._frames if f is not None)
            if all(f is not None for f in self._frames):
                # Persist via a daemon THREAD, not the loop: the boot's short
                # asyncio.run() closes (cancelling loop tasks) right after the
                # play — a thread survives it, and the ov process lives on
                # through attach, so the write always completes. The handle is
                # kept so tests (and shutdown paths) can join it.
                self._save_thread = threading.Thread(
                    target=_save_cached_ring,
                    args=(self._cols, self._rows, self._n, self._cache_ss,
                          [f for f in self._frames if f is not None]),
                    daemon=True,
                )
                self._save_thread.start()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    def _frame_for(self, phase: float) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
        """The nearest BUILT frame for a phase — adaptive: falls back toward the
        resting pose while the ring is still filling. Never raises."""
        if not self._frames:
            return self._base
        n = len(self._frames)
        want = int((phase % 1.0) * n) % n
        for off in range(n):
            for idx in ((want - off) % n, (want + off) % n):
                f = self._frames[idx]
                if f is not None:
                    return f
        return self._base

    # -- the + prey (analytic, phase-shifted, rides the gap) -------------

    def plus_cell(self, phase: float) -> Optional[Tuple[int, int]]:
        """The prey's centre (x, cell_row) for this phase — on the ring path in
        the gap, just ahead of the open mouth. Never raises."""
        if not self.available:
            return None
        rot = 2.0 * math.pi * (phase % 1.0)
        x, py, _s = _prey_center_px(self._cols, self._rows, rot, self._plus_lead)
        return (x, py // 2)

    def prey_pixels(self, phase: float) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
        """The prey sprite overlay for this phase — rendered LIVE (not baked into
        cached frames) so its pulse animates even on a fully-cached ring. The
        pulse beats three times per lap. Never raises."""
        if not self.available:
            return {}
        ph = phase % 1.0
        rot = 2.0 * math.pi * ph
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * ph * 3.0)
        return build_prey_sprite(self._cols, self._rows, rot, self._plus_lead, pulse)

    # -- rendering (independent of logs) ---------------------------------

    def _pixels_to_text(
        self,
        pixels: Dict[Tuple[int, int], Tuple[int, int, int]],
        overlay: Optional[Dict[Tuple[int, int], Tuple[int, int, int]]] = None,
    ) -> Any:
        try:
            from rich.text import Text
        except Exception:  # noqa: BLE001
            return ""
        text = Text()
        ov = overlay or {}

        def _px(key):
            v = ov.get(key)
            return v if v is not None else pixels.get(key)

        for cy in range(self._cy_lo, self._cy_hi + 1):
            for x in range(self._cols):
                top = _px((x, cy * 2))
                bot = _px((x, cy * 2 + 1))
                if top is None and bot is None:
                    text.append(" ")
                elif top is not None and bot is not None:
                    tr, tg, tb = top
                    br, bg, bb = bot
                    text.append("▀", style=f"rgb({tr},{tg},{tb}) on rgb({br},{bg},{bb})")
                elif top is not None:
                    tr, tg, tb = top
                    text.append("▀", style=f"rgb({tr},{tg},{tb})")
                else:
                    br, bg, bb = bot  # type: ignore[misc]
                    text.append("▄", style=f"rgb({br},{bg},{bb})")
            if cy < self._cy_hi:
                text.append("\n")
        return text

    def crest_frame_text(self, phase: float) -> Any:
        """The animated crest for ``phase`` — the physically-rotated snake + the
        ``+`` prey. Independent of the log buffer. Never raises."""
        if not self.available:
            try:
                from rich.text import Text
                return Text("")
            except Exception:  # noqa: BLE001
                return ""
        return self._pixels_to_text(self._frame_for(phase), self.prey_pixels(phase))

    def resting_text(self) -> Any:
        """The TRUE static emblem (full-fidelity raster, no prey) — what the
        canvas freezes on. Byte-identical to the static crest's pixels."""
        return self._pixels_to_text(self._base)

    # -- the bottom log partition (thread-safe) --------------------------

    def set_progress(self, line: str) -> None:
        """Set the ONE live progress line under the boot log. Thread-safe.

        Distinct from `add_log` on purpose. The log deque is a TRANSCRIPT —
        each entry is an event that happened and must stay. Progress is a
        GAUGE: one value that supersedes its predecessor, and appending it
        produced the stack of six identical bars the operator saw.

        No repaint is triggered here. `Live` re-renders `render(phase)` on
        every frame anyway, so a mutated slot is picked up by the next one —
        which is precisely why the progress line must live INSIDE this
        renderable. Written to the terminal directly it lands underneath the
        Live and is erased by the following frame, which is what made it
        appear to flicker.
        """
        try:
            with self._lock:
                self._progress = str(line)
        except Exception:  # noqa: BLE001
            pass

    def clear_progress(self) -> None:
        """Drop the progress line — the boot is over and the gauge is spent."""
        try:
            with self._lock:
                self._progress = ""
        except Exception:  # noqa: BLE001
            pass

    def add_log(self, line: str) -> None:
        """Append one async boot log to the BOTTOM partition — never touches the
        crest matrix. Thread-safe; never raises."""
        try:
            with self._lock:
                self._logs.append(str(line))
        except Exception:  # noqa: BLE001
            pass

    def logs_renderable(self) -> Any:
        try:
            from rich.text import Text
            with self._lock:
                lines = list(self._logs)
            with self._lock:
                progress = self._progress
            # FIXED HEIGHT, from the first frame. The window is the deque's
            # own capacity plus the one gauge slot, padded to that height
            # whether or not anything has arrived yet.
            #
            # Why: `Live` (screen=False) repaints by moving the cursor up
            # the height of the LAST frame and rewriting. While the log
            # filled — one row per boot event — and again each time the
            # gauge appeared or cleared, the region's height changed under
            # it, so the terminal scrolled and re-laid out the whole block.
            # Every one of those was a visible jump. A region that never
            # changes height is repainted in place and never scrolls; the
            # deque's maxlen already fixes the ceiling, this makes the floor
            # match it.
            slots = int(self._logs.maxlen or max(1, len(lines)))
            rows = list(lines[-slots:])
            rows += [""] * (slots - len(rows))
            # LAST, and always last: the gauge sits beneath the transcript so
            # new log entries push it down rather than scrolling past it —
            # and it is always PRESENT, blank when spent, so its arrival and
            # departure cannot move the rows above it.
            rows.append(progress)
            t = Text()
            for i, ln in enumerate(rows):
                t.append(ln, style=_LOG_RGB)
                if i < len(rows) - 1:
                    t.append("\n")
            return t
        except Exception:  # noqa: BLE001
            return ""

    def render(self, phase: float) -> Any:
        try:
            from rich.console import Group
            from rich.text import Text
            return Group(self.crest_frame_text(phase), Text(""), self.logs_renderable())
        except Exception:  # noqa: BLE001
            return self.crest_frame_text(phase)

    def render_resting(self) -> Any:
        try:
            from rich.console import Group
            from rich.text import Text
            return Group(self.resting_text(), Text(""), self.logs_renderable())
        except Exception:  # noqa: BLE001
            return self.resting_text()

    # -- the Live playback loop (adaptive, decoupled duration) -----------

    async def play(
        self,
        console: Any,
        *,
        stop_event: Any,
        fps: Optional[int] = None,
        min_seconds: Optional[float] = None,
        min_laps: Optional[float] = None,
        sleep_fn=None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Animate inside a ``rich.live.Live`` until BOTH the daemon is up
        (``stop_event``) AND the minimum visible run has shown, then settle onto
        the true static emblem. Kicks the background frame builder itself.
        Never raises out."""
        import asyncio
        if not self.available:
            return
        builder = asyncio.ensure_future(self.ensure_frames())
        sleep = sleep_fn or asyncio.sleep
        rate = fps or _fps()
        n = max(1, len(self._frames))
        step = 1.0 / n
        floor_s = _min_intro_s() if min_seconds is None else max(0.0, float(min_seconds))
        floor_laps = _min_laps() if min_laps is None else max(0.0, float(min_laps))
        min_ticks = max(int(math.ceil(floor_s * rate)), int(math.ceil(floor_laps * n)))
        if max_frames is not None:
            min_ticks = min(min_ticks, max_frames)
        try:
            from rich.live import Live
        except Exception:  # noqa: BLE001
            return
        phase = 0.0
        ticks = 0
        # ONE PAINT PER FRAME, ATOMIC. `Live` repaints a region by moving the
        # cursor up and rewriting every line; nothing groups the erase and
        # the rewrite, so at fourteen frames a second the terminal can show
        # the region half-drawn. That is the boot-time flicker. DEC mode
        # 2026 makes the terminal hold everything between BEGIN and END and
        # paint once; every paint below — the first, each frame, the resting
        # emblem — is bracketed by it. Emitted only when a real VT terminal
        # is on the other end (`supports_synchronized_output`), which is
        # also why the Live is started by hand rather than by `with`: the
        # context manager's own first paint would land outside the bracket.
        sync = _synchronized_paint(console)
        live = Live(
            self.render(phase), console=console, refresh_per_second=rate,
            transient=False, auto_refresh=False, screen=False,
        )
        try:
            with sync():
                live.start(refresh=True)
            try:
                while True:
                    await sleep(1.0 / rate)          # cooperative yield
                    phase = (phase + step) % 1.0
                    with sync():
                        live.update(self.render(phase))
                        live.refresh()
                    ticks += 1
                    if max_frames is not None and ticks >= max_frames:
                        break
                    done = (stop_event is None or stop_event.is_set()) and ticks >= min_ticks
                    if done and phase < step:        # lap boundary → snap-free
                        break
                with sync():
                    live.update(self.render_resting())   # settle on the TRUE emblem
                    live.refresh()
            finally:
                # `stop()` paints once more as it releases the region and
                # restores the cursor. It is a paint like any other and gets
                # the same bracket — measured: without it, the very last
                # frame of every boot was the one frame the terminal could
                # show half-drawn.
                with sync():
                    live.stop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return
        finally:
            # The builder may finish + cache in the background; don't await it.
            if builder.done():
                try:
                    builder.result()
                except Exception:  # noqa: BLE001
                    pass


def build_animator(console: Any, *, plus_lead_deg: Optional[float] = None) -> Optional["CrestAnimator"]:
    """Construct an animator sized to the live console, or ``None`` when the
    crest can't render (disabled / non-TTY / tiny / NONE-tier). Never raises."""
    try:
        if not animator_enabled():
            return None
        import sys
        if not sys.stdout.isatty():
            return None
        size = console.size
        anim = CrestAnimator(cols=size.width, rows=size.height, plus_lead_deg=plus_lead_deg)
        return anim if anim.available else None
    except Exception:  # noqa: BLE001
        return None




# ---------------------------------------------------------------------------
# MiniCrest — the CC-style header logo (small, ambient, stateless-animated)
# ---------------------------------------------------------------------------




def _quant_palette() -> List[Tuple[int, int, int]]:
    """The definitive quantization palette — the crest's OWN pure colors (DRY:
    gradient stops + head + eye + V), never a hardcoded copy."""
    from .crest import _EYE_RGB, _HEAD_RGB, _STOPS, _V_BOT_RGB, _V_TOP_RGB
    pal = [tuple(c) for (_deg, c) in _STOPS]
    pal += [tuple(_HEAD_RGB), tuple(_EYE_RGB), tuple(_V_TOP_RGB), tuple(_V_BOT_RGB)]
    out: List[Tuple[int, int, int]] = []
    for c in pal:
        if c not in out:
            out.append(c)
    return out


def _v_family() -> set:
    from .crest import _V_BOT_RGB, _V_TOP_RGB
    return {tuple(_V_TOP_RGB), tuple(_V_BOT_RGB)}


def _snap(rgb: Tuple[float, float, float], pal: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    r, g, b = rgb
    best, bd = pal[0], 1e18
    for p in pal:
        d = (p[0] - r) ** 2 + (p[1] - g) ** 2 + (p[2] - b) ** 2
        if d < bd:
            bd, best = d, p
    return best


def build_scaled_frame(
    cells_w: int, rot: float, ss: int, v_rot: float = 0.0, alpha: str = "sharp",
    src_cols: int = 47, pack: str = "quad",
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """Aspect-TRUE mini frame for the quadrant medium. A terminal cell is 1:2,
    so quadrant subpixels are 0.5w x 1h — a symmetric 2x2 pack of half-block
    art renders OVAL (the squashed-logo bug). This samples the SAME quality-
    floor geometry (crest's own _sample_* + _pixel_color_and_delay — DRY) on an
    ANISOTROPIC grid: 2*cells_w subs across x, cells_w subs down y → after the
    2x2 quadrant pack the emblem is ROUND at cells_w x cells_w/2 cells. The +
    prey is RASTERIZED through the same window transform from the big logo's
    physical law (shared _prey_plus_hit/_prey_rgb — DRY). Pure; thread-safe;
    never raises — returns {} on any fault."""
    try:
        pf = generate_crest_pixels(src_cols, _Geometry.rows_needed(src_cols - 1) + 3)
        if pf is None:
            return {}
        geo = _rotated_geometry(pf.cols, pf.rows, rot, v_rot)
        # Icon-legibility exaggeration (the pixel-artist rule: at icon scale,
        # identity features must be OVERSIZED to read). Applied ONLY here — the
        # big emblem keeps true proportions. Env-tunable, derived not drawn.
        # Default 1.0 = EXACT big-logo proportions (operator mandate). Raising
        # the boost trades exactness for icon legibility; every factor scales
        # from (fb-1) so fb=1.0 is mathematically identity.
        fb = _env_float("JARVIS_CREST_MINI_FEATURE_BOOST", 1.0, 1.0, 2.5)
        if fb > 1.0:
            geo.gap_half = geo.gap_half * fb
            geo.head_len = geo.head_len * (1.0 + (fb - 1.0) * 0.6)
            geo.head_w = geo.head_w * (1.0 + (fb - 1.0) * 0.5)
            geo.eye_r = geo.eye_r * (1.0 + (fb - 1.0) * 1.5)
            geo.tail_tip = _ang_norm(geo.gap_center + geo.gap_half)
            geo.head_theta = _ang_norm(geo.gap_center - geo.gap_half)
        xs = [x for (x, _py) in pf.pixels]
        pys = [py for (_x, py) in pf.pixels]
        if not xs:
            return {}
        pad = 1.5
        x_lo, x_hi = min(xs) - pad, max(xs) + 1 + pad
        y_lo, y_hi = min(pys) * _REF_ASPECT - pad, (max(pys) + 1) * _REF_ASPECT + pad
        span = max(x_hi - x_lo, y_hi - y_lo)        # square physical window
        cx_p, cy_p = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        if pack == "half":
            # THE BIG LOGO'S OWN FINISH, smaller: halfblock pixels are square
            # on screen (1 cell wide x half a cell tall), so the grid is
            # ISOTROPIC — same aa feathering, same per-pixel color fidelity.
            sub_w, sub_h = cells_w, cells_w
        else:
            sub_w, sub_h = 2 * cells_w, cells_w      # 0.5w x 1h subs -> round
        px_pitch, py_pitch = span / sub_w, span / sub_h
        v_span = max(1e-6, geo.v_top + geo.v_bot)
        out: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        for qy in range(sub_h):
            y0 = cy_p - span / 2.0 + qy * py_pitch
            for qx in range(sub_w):
                x0 = cx_p - span / 2.0 + qx * px_pitch
                votes = {"eye": 0, "head": 0, "coil": 0, "v": 0}
                theta = None
                spy = y0
                for sy in range(ss):
                    spy = y0 + (sy + 0.5) / ss * py_pitch
                    for sx in range(ss):
                        spx = x0 + (sx + 0.5) / ss * px_pitch
                        if _sample_eye(spx, spy, geo):
                            votes["eye"] += 1
                        elif _sample_head(spx, spy, geo):
                            votes["head"] += 1
                        elif _sample_coil(spx, spy, geo):
                            votes["coil"] += 1
                            theta = math.atan2(-(spy - geo.cy), spx - geo.cx)
                        elif _sample_v(spx, spy, geo):
                            votes["v"] += 1
                inside = sum(votes.values())
                if inside == 0:
                    continue
                coverage = inside / (ss * ss)
                if coverage < (0.22 if alpha == "sharp" else 0.06):
                    continue
                kind = max(votes, key=votes.get)
                py_frac = (spy - (geo.cy - geo.v_top)) / v_span
                base, _d = _pixel_color_and_delay(kind, theta, py_frac, geo)
                a = coverage ** (1.35 if alpha == "sharp" else 0.75)
                out[(qx, qy)] = tuple(
                    max(0, min(255, round(c * a))) for c in base
                )
        out = _prune_sparse_geometry(out)
        # The + prey — the BIG logo's own PHYSICAL law (arm = 1.9·scale,
        # thick = 0.55·scale — build_prey_sprite's exact numbers, ~9% of the
        # emblem), rasterized through the SAME window transform + ss
        # supersampling + aa feathering as the body. The old stamp sized off
        # the CELL grid (cells_w // 8 → over HALF the emblem wide at 24 cells:
        # the giant-plus bug) — a wrong-medium law. A shape-preserving
        # legibility floor (arms ≥ 1.5 subs, bar ≥ 1 sub) keeps the plus
        # READING as a plus at icon scale and converges to exact proportion as
        # the mini grows. Pulse rides rot (3 beats/lap — prey_pixels' law) so
        # baked frames stay clock-stateless yet alive.
        pscale = _env_float("JARVIS_CREST_MINI_PREY_SCALE", 1.0, 0.0, 3.0)
        if pscale > 0.0:
            theta = _ang_norm(geo.gap_center)
            px_p = geo.cx + geo.r_mid * math.cos(theta)   # SAME radius as the big prey
            py_p = geo.cy - geo.r_mid * math.sin(theta)
            max_pitch = max(px_pitch, py_pitch)
            arm_p = max(1.9 * geo.scale * pscale, 1.5 * max_pitch)
            thick_p = max(0.55 * geo.scale * pscale, 0.51 * max_pitch)
            glow = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(3.0 * rot))
            cut = 0.22 if alpha == "sharp" else 0.06
            gamma = 1.35 if alpha == "sharp" else 0.75
            x_org, y_org = cx_p - span / 2.0, cy_p - span / 2.0
            qx_lo = max(0, int((px_p - arm_p - x_org) / px_pitch) - 1)
            qx_hi = min(sub_w - 1, int((px_p + arm_p - x_org) / px_pitch) + 1)
            qy_lo = max(0, int((py_p - arm_p - y_org) / py_pitch) - 1)
            qy_hi = min(sub_h - 1, int((py_p + arm_p - y_org) / py_pitch) + 1)
            best_k, best_d = None, None
            for qy in range(qy_lo, qy_hi + 1):
                y0 = y_org + qy * py_pitch
                for qx in range(qx_lo, qx_hi + 1):
                    x0 = x_org + qx * px_pitch
                    hits, rs, gs, bs = 0, 0, 0, 0
                    for sy in range(ss):
                        spy = y0 + (sy + 0.5) / ss * py_pitch
                        for sx in range(ss):
                            spx = x0 + (sx + 0.5) / ss * px_pitch
                            if _prey_plus_hit(spx - px_p, spy - py_p, arm_p, thick_p):
                                hits += 1
                                # integrate the COLOR FIELD over the hits — a
                                # point-sample at the sub centre misses the
                                # narrow pale core when the grid straddles it
                                # (the coarse-raster core-loss bug).
                                r_, g_, b_ = _prey_rgb(
                                    spx - px_p, spy - py_p, arm_p, glow,
                                )
                                rs, gs, bs = rs + r_, gs + g_, bs + b_
                    dxc = x0 + px_pitch / 2.0 - px_p
                    dyc = y0 + py_pitch / 2.0 - py_p
                    d = dxc * dxc + dyc * dyc
                    if best_d is None or d < best_d:
                        best_d, best_k = d, (qx, qy)
                    if not hits:
                        continue
                    coverage = hits / (ss * ss)
                    if coverage < cut:
                        continue
                    a = coverage ** gamma
                    out[(qx, qy)] = tuple(
                        max(0, min(255, round(c / hits * a)))
                        for c in (rs, gs, bs)
                    )
            if best_k is not None:
                # STRUCTURAL core guarantee — the big sprite's integer loop
                # always evaluates dx=dy=0 (the pale core); the mini's coarse
                # grid can straddle it, losing the core to the purple mid-band.
                # Mirror the guarantee: the sub nearest the true centre gets
                # t=0 through the SAME shared law. Doubles as the never-vanish
                # guard at degenerate sizes.
                out[best_k] = _prey_rgb(0.0, 0.0, arm_p, glow)
        return out
    except Exception:  # noqa: BLE001
        return {}


class MiniCrest:
    """The header logo — the REAL emblem, aspect-true and CC-sized. Frames come
    from :func:`build_scaled_frame` (the crest's own samplers on an anisotropic
    grid matched to the quadrant medium), packed 2x2 via the crest's _QUAD
    table. Round, sharp, physically rotating, prey included. The ring builds in
    ~1s off-loop (no disk cache needed). Clock-stateless animation."""

    def __init__(
        self,
        *,
        cols: Optional[int] = None,
        frame_count: Optional[int] = None,
        ss: Optional[int] = None,
        speed_laps_per_s: Optional[float] = None,
        source_cols: Optional[int] = None,     # API compat
        source_rows: Optional[int] = None,     # API compat
    ) -> None:
        self._speed = speed_laps_per_s if speed_laps_per_s is not None else _env_float(
            "JARVIS_CREST_MINI_SPEED", 0.10, 0.01, 2.0,
        )
        self._cells_w = cols if cols is not None else _env_int(
            "JARVIS_CREST_MINI_CELLS", 24, 10, 32,
        )
        self._ss_mini = ss if ss is not None else _env_int("JARVIS_CREST_MINI_SS", 5, 2, 6)
        # THE BIG LOGO'S AESTHETIC by default: halfblock + aa (soft gradient,
        # feathered edges, per-pixel color — the emblem's own finish, smaller).
        # JARVIS_CREST_MINI_PACK=quad restores the packed pixel-art mode.
        self._pack = os.environ.get("JARVIS_CREST_MINI_PACK", "half").strip().lower()
        if self._pack not in ("half", "quad"):
            self._pack = "half"
        edge = os.environ.get(
            "JARVIS_CREST_MINI_EDGE", "aa" if self._pack == "half" else "sharp",
        ).strip().lower()
        self._edge = edge if edge in ("aa", "sharp", "crisp") else "sharp"
        self._n = frame_count if frame_count is not None else _env_int(
            "JARVIS_CREST_MINI_FRAMES", 24, 4, 48,
        )
        self._frames: List[Optional[dict]] = [None] * max(1, self._n)
        f0 = build_scaled_frame(self._cells_w, 0.0, self._ss_mini, 0.0, self._edge,
                                pack=self._pack)
        if f0:
            self._frames[0] = f0
        self._built = 1 if f0 else 0
        self._builder_started = False

    @property
    def available(self) -> bool:
        return self._built > 0

    @property
    def rows(self) -> int:
        return max(1, self._cells_w // 2)

    @property
    def cols(self) -> int:
        return self._cells_w

    async def ensure_frames(self) -> None:
        """Fill the ring off-loop (~20-60ms/frame → ~1s total). Idempotent."""
        import asyncio
        if self._builder_started:
            return
        self._builder_started = True
        for k in range(self._n):
            if self._frames[k] is not None:
                continue
            rot = 2.0 * math.pi * k / self._n
            f = await asyncio.to_thread(
                build_scaled_frame, self._cells_w, rot, self._ss_mini,
                _v_spin_mult() * rot, self._edge, 47, self._pack,
            )
            if f:
                self._frames[k] = f
                self._built = sum(1 for x in self._frames if x)

    def _frame_now(self, now: Optional[float] = None) -> Optional[dict]:
        import time as _time
        if not self._frames:
            return None
        t = _time.monotonic() if now is None else float(now)
        n = len(self._frames)
        want = int(((t * self._speed) % 1.0) * n) % n
        for off in range(n):
            for idx in ((want - off) % n, (want + off) % n):
                if self._frames[idx] is not None:
                    return self._frames[idx]
        return None

    def row_texts(self, now: Optional[float] = None) -> List[Any]:
        """Quadrant-pack the aspect-true subgrid (crest's _QUAD table — DRY):
        every subpixel renders individually at 2x horizontal density. Never
        raises."""
        try:
            from rich.text import Text
            from .crest import _QUAD
        except Exception:  # noqa: BLE001
            return []
        pixels = self._frame_now(now)
        if not pixels:
            return []
        if self._pack == "half":
            # The big logo's own renderer: halfblock ▀ with independent fg/bg —
            # per-pixel color fidelity, feathered aa edges. Identical finish.
            rows: List[Any] = []
            W = self._cells_w
            for cy in range(W // 2):
                t = Text()
                for x in range(W):
                    top = pixels.get((x, cy * 2))
                    bot = pixels.get((x, cy * 2 + 1))
                    if top is None and bot is None:
                        t.append(" ")
                    elif top is not None and bot is not None:
                        tr, tg, tb = top
                        br, bg, bb = bot
                        t.append("▀", style=f"rgb({tr},{tg},{tb}) on rgb({br},{bg},{bb})")
                    elif top is not None:
                        tr, tg, tb = top
                        t.append("▀", style=f"rgb({tr},{tg},{tb})")
                    else:
                        br, bg, bb = bot
                        t.append("▄", style=f"rgb({br},{bg},{bb})")
                rows.append(t)
            return rows
        sub_w, sub_h = 2 * self._cells_w, self._cells_w
        rows: List[Any] = []
        for cy in range(sub_h // 2):
            t = Text()
            qy = cy * 2
            for qx in range(0, sub_w, 2):
                subs = [
                    pixels.get((qx, qy)),
                    pixels.get((qx + 1, qy)),
                    pixels.get((qx, qy + 1)),
                    pixels.get((qx + 1, qy + 1)),
                ]
                bits = 0
                lit = []
                for i, c in enumerate(subs):
                    if c is not None:
                        bits |= 1 << (3 - i)
                        lit.append(c)
                if not bits:
                    t.append(" ")
                    continue
                # Brightest lit sub wins the cell — averaging 4 subs dulled
                # edges into mush; max-luminance keeps the punch of the big one.
                r, g, b = max(lit, key=lambda c: c[0] + c[1] + c[2])
                t.append(_QUAD.get(bits, "█"), style=f"rgb({r},{g},{b})")
            rows.append(t)
        return rows


def render_cockpit_header(
    mini: Optional["MiniCrest"],
    lines: List[Any],
    width: int,
    *,
    now: Optional[float] = None,
    right_gutter: Optional[Any] = None,
    gutter_row: int = 0,
    gutter_align: str = "right",
) -> str:
    """Compose the CC-style header — mini crest left, identity/context/path
    lines beside it — to an ANSI string bounded to ``width``. ``lines`` are
    Rich renderable Texts (or plain strings). Degrades to text-only when the
    mini crest is unavailable (tiny terminal / NONE tier). Never raises.

    ``right_gutter`` fills the empty real estate to the RIGHT of the identity
    lines — the Braille audio oscilloscope lives there. It may be a Rich
    renderable, a plain/markup string, or a zero-arg callable returning either;
    a callable is re-evaluated per frame so the clock-driven repaint animates it
    with no extra task. It is RIGHT-ALIGNED against ``width``, which is why the
    alignment lives here rather than in the caller: this function is the only
    place that knows the terminal width. Rendered on ``gutter_row`` of the text
    block. Silently omitted if it does not fit, so a narrow terminal degrades to
    the plain header instead of wrapping into a broken layout.

    ``gutter_align`` chooses WHERE, because the same live surface reads very
    differently in the two spots:

      ``"right"``  — right-aligned on ``gutter_row``, hugging the terminal
                     edge. Out of the way; also easy to miss.
      ``"below"``  — its own row in the text column, directly beneath the
                     identity lines, so it reads as part of the organism's
                     nameplate rather than as chrome pinned to the margin.

    Placement is a parameter rather than a fork because everything else about
    the gutter — the per-frame callable, the markup handling, the fit check —
    is identical between the two."""
    try:
        from io import StringIO
        from rich.console import Console
        from rich.text import Text

        crest_rows = mini.row_texts(now) if mini is not None else []
        text_rows: List[Any] = [
            ln if not isinstance(ln, str) else Text(ln) for ln in lines
        ]
        # BOUND EACH IDENTITY ROW TO THE COLUMN IT LIVES IN.
        #
        # The rows are appended after the crest and a two-space gutter, and
        # were previously appended WHOLE. A row longer than the remaining
        # width therefore wrapped — and a wrapped continuation starts at
        # column ZERO, i.e. on top of the crest. The capability reason
        # ("doubleword is out of credit — add credits, or configure a local
        # lane to keep background work moving") is exactly long enough to do
        # it, so the nameplate collided with the artwork.
        #
        # This function's own docstring already states the principle — it is
        # "the only place that knows the terminal width", and it degrades
        # "instead of wrapping into a broken layout" — but applied it only to
        # the gutter. The same bound belongs on the rows themselves, using the
        # same arithmetic, which is why it is computed once here rather than
        # guessed by the caller.
        #
        # Truncation over wrapping is deliberate: the header is a NAMEPLATE,
        # a fixed number of rows beside a fixed-height crest. Letting it grow
        # would push the layout around every time a provider state changed,
        # and the full text is always available in the banner below.
        _text_avail = max(8, int(width) - (
            mini.cols if (mini is not None and crest_rows) else 0) - 2)
        _bounded: List[Any] = []
        for _ln in text_rows:
            try:
                # Copy before truncating: `lines` belongs to the caller and is
                # rebuilt per frame, but mutating a shared Text would make the
                # first narrow frame permanent.
                _c = _ln.copy() if hasattr(_ln, "copy") else _ln
                if hasattr(_c, "truncate"):
                    _c.truncate(_text_avail, overflow="ellipsis")
                _bounded.append(_c)
            except Exception:  # noqa: BLE001 — a bound never breaks the header
                _bounded.append(_ln)
        text_rows = _bounded
        # Top-align the text beside the crest (the CC layout) with a
        # 1-row optical inset. Computed from the ORIGINAL text rows, BEFORE a
        # "below" gutter is appended: otherwise arming the audio plane would
        # add a row, flip this inset, and shift every identity line up by one —
        # the header would visibly jump the moment the mic came alive.
        pad_top = 1 if len(crest_rows) > len(text_rows) else 0
        crest_w = mini.cols if (mini is not None and crest_rows) else 0

        if right_gutter is not None and str(gutter_align).lower() == "below":
            # The gutter becomes an ordinary text row. Nothing about the row
            # loop needs to know it is special, which is why this is an append
            # rather than a second rendering path.
            try:
                _g = right_gutter() if callable(right_gutter) else right_gutter
                _gt = _g if not isinstance(_g, str) else Text.from_markup(_g)
                # An empty gutter contributes NO row — an unarmed audio plane
                # must not leave a blank line under the path.
                if _gt is not None and len(_gt) > 0:
                    _avail = int(width) - crest_w - 2 - 1
                    if len(_gt) <= _avail:
                        text_rows.append(_gt)
            except Exception:  # noqa: BLE001 — a gutter fault never breaks the header
                pass

        n_rows = max(len(crest_rows), len(text_rows) + pad_top)
        if n_rows == 0:
            return ""
        out = Text()
        for i in range(n_rows):
            if i < len(crest_rows):
                out.append_text(crest_rows[i])
            else:
                out.append(" " * crest_w)
            ti = i - pad_top
            if 0 <= ti < len(text_rows):
                out.append("  ")
                _row = text_rows[ti] if not isinstance(text_rows[ti], str) else Text(text_rows[ti])
                out.append_text(_row)
                if (
                    ti == gutter_row
                    and right_gutter is not None
                    and str(gutter_align).lower() != "below"
                ):
                    try:
                        _g = right_gutter() if callable(right_gutter) else right_gutter
                        _gt = _g if not isinstance(_g, str) else Text.from_markup(_g)
                        if _gt is not None and len(_gt) > 0:
                            # Right-align: pad from where this row currently ends
                            # to (width - gutter_len). Omit rather than wrap when
                            # the terminal is too narrow.
                            _used = crest_w + 2 + len(_row)
                            _pad = max(0, int(width) - _used - len(_gt) - 1)
                            if _used + _pad + len(_gt) <= int(width):
                                out.append(" " * _pad)
                                out.append_text(_gt)
                    except Exception:  # noqa: BLE001 — a gutter fault never breaks the header
                        pass
            if i < n_rows - 1:
                out.append("\n")
        buf = StringIO()
        Console(
            file=buf, force_terminal=True, color_system="truecolor",
            width=max(20, width), highlight=False,
        ).print(out)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["CrestAnimator", "MiniCrest", "animator_enabled", "build_animator", "build_rotated_frame", "render_cockpit_header"]
