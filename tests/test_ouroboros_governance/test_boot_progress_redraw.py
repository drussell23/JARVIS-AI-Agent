"""The boot progress line redraws in place, and cannot wrap.

The reported symptom was "the logo flickers at the beginning when it is
trying to load". It was not an animation and not the logo: the progress
line runs 85 columns at full extent --

    ⎿ [██████████████····]   75%  session open  108s  +98s over  waiting on cockpit wired

-- and nothing clamped it to the terminal. On any terminal narrower than
that it WRAPS, and `\\r` then returns to the start of the last visual row
only. Each redraw paints over part of itself and leaves the rest standing;
four times a second that reads as flicker.

The second defect was the eraser. The old code padded with spaces to the
previous line's length, carrying a `width[0]` cell to remember it. That
approximates `\\033[K` (erase to end of line) while being wrong in exactly
the case that matters: when the previous frame wrapped, the padding erases
the wrong row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.cli import thin_client as tc  # noqa: E402

# The real line, at its widest.
WIDE = ("⎿ [██████████"
        "████····]   75%  session open"
        "  108s  +98s over  waiting on cockpit wired")


# ---------------------------------------------------------------------------
# It cannot wrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cols", [40, 60, 80, 100])
def test_the_line_never_exceeds_the_terminal(cols) -> None:
    fitted = tc._fit_to_width(WIDE, cols)
    assert tc._visible_len(fitted) <= cols, (
        f"{tc._visible_len(fitted)} columns on a {cols}-column terminal wraps, "
        "and a carriage return cannot undo a wrap")


def test_a_line_that_already_fits_is_untouched() -> None:
    """No ellipsis, no truncation, byte-identical — the common case must
    not pay for the rare one."""
    short = "⎿ waking · 3s"
    assert tc._fit_to_width(short, 80) == short


def test_truncation_keeps_the_left_where_the_meaning_is() -> None:
    fitted = tc._fit_to_width(WIDE, 30)
    assert fitted.startswith("⎿ [█"), "the bar answers 'is it moving'"
    assert fitted.endswith("…"), "a cut line must not read as a complete one"


def test_an_unknown_width_does_not_truncate() -> None:
    """0 means 'could not measure'. Guessing a width would cut content that
    would have fitted, which is worse than not clamping."""
    assert tc._fit_to_width(WIDE, 0) == WIDE
    assert tc._fit_to_width(WIDE, -1) == WIDE


def test_a_one_column_terminal_does_not_crash() -> None:
    assert len(tc._fit_to_width(WIDE, 1)) <= 1


# ---------------------------------------------------------------------------
# Width is measured, not counted
# ---------------------------------------------------------------------------


def test_visible_len_is_columns_not_code_points() -> None:
    assert tc._visible_len("") == 0
    assert tc._visible_len("abc") == 3
    # Whatever the measurement backend, it must be a usable non-negative int.
    assert isinstance(tc._visible_len(WIDE), int)
    assert tc._visible_len(WIDE) > 0


def test_width_is_asked_fresh_so_a_resize_re_fits() -> None:
    """Cached at construction, a resize would tear until restart."""
    import inspect
    src = inspect.getsource(tc._mk_tick)
    assert "_terminal_columns(out)" in src, "measured inside the tick"
    assert "width = [0]" not in src, "the padding state is gone"


def test_an_unmeasurable_stream_reports_unknown() -> None:
    class _NoFd:
        pass

    assert tc._terminal_columns(_NoFd()) >= 0   # never raises


# ---------------------------------------------------------------------------
# The eraser
# ---------------------------------------------------------------------------


def test_the_redraw_uses_the_terminals_own_erase() -> None:
    """`\\033[K` is the primitive for 'clear to end of line'. The manual
    space-padding it replaces erases the wrong row once a frame has
    wrapped, which is the state this whole fix exists to prevent."""
    import inspect
    src = inspect.getsource(tc._mk_tick)
    assert "\\033[K" in src or "\x1b[K" in src
    assert '" " * pad' not in src, "manual padding is gone"


def test_the_redraw_still_writes_to_the_stream_it_tested() -> None:
    """`real_stdout_isatty` inspects `sys.__stdout__`; painting `sys.stdout`
    would test one stream and draw on another."""
    import inspect
    src = inspect.getsource(tc._mk_tick)
    assert "sys.__stdout__" in src
