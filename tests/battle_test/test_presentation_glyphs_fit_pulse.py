"""Sovereign Terminal UI — glyph vocabulary / print_fit / pulse unit tests."""
import sys

import backend.core.ouroboros.battle_test.presentation_restraint as PR
from backend.core.ouroboros.ui import theme


class _FakeStdout:
    """Minimal stdout stand-in carrying a settable ``encoding`` (the real
    ``sys.stdout.encoding`` is a readonly C attribute, so we replace the whole
    object to exercise the genuine encoding-detection path)."""

    def __init__(self, encoding):
        self.encoding = encoding

    def write(self, *_a):
        return 0

    def flush(self):
        pass


# --------------------------------------------------------------------------- glyphs
# The glyphs come from the ONE design-language table (`theme._GLYPHS`) and
# degrade with it, for the terminal that will DISPLAY them — the theme asks
# an attached cockpit before the local locale. The spinner is local and
# still keys off stdout.
def test_glyphs_utf8(monkeypatch):
    monkeypatch.setattr(theme, "supports_unicode", lambda env=None: True)
    g = PR.glyphs()
    assert g["action"] == "⏺" and g["result"] == "⎿"
    monkeypatch.setattr(sys, "stdout", _FakeStdout("utf-8"))
    assert PR.spinner_name() == "dots"


def test_glyphs_ascii_fallback(monkeypatch):
    monkeypatch.setattr(theme, "supports_unicode", lambda env=None: False)
    g = PR.glyphs()
    assert g["action"] == "*" and g["result"] == "-"   # the table's ASCII forms
    monkeypatch.setattr(sys, "stdout", _FakeStdout("ascii"))
    assert PR.spinner_name() == "line"


def test_glyphs_are_the_theme_marks(monkeypatch):
    for want in (True, False):
        monkeypatch.setattr(theme, "supports_unicode", lambda env=None, _w=want: _w)
        assert PR.glyphs() == {
            "action": theme.mark("action", unicode=want),
            "result": theme.mark("detail", unicode=want),
        }


def test_spinner_none_encoding_is_safe(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(None))   # encoding can be None
    assert PR.spinner_name() == "line"


def test_spinner_missing_stdout_is_safe(monkeypatch):
    monkeypatch.setattr(sys, "stdout", object())  # no .encoding attr at all
    assert PR.spinner_name() == "line"            # fail-safe to ASCII


# --------------------------------------------------------------------------- print_fit
from io import StringIO  # noqa: E402
from rich.console import Console  # noqa: E402


def _con(width):
    return Console(file=StringIO(), width=width, force_terminal=False, color_system=None)


def test_print_fit_truncates_to_width():
    con = _con(40)
    PR.print_fit(con, "  ⎿ " + ("x/" * 80))            # far wider than 40
    out = con.file.getvalue().rstrip("\n")
    assert len(out) <= 40                               # never exceeds width
    assert out.endswith("…") or out.endswith("...")     # ellipsis applied


def test_print_fit_short_line_unchanged():
    con = _con(80)
    PR.print_fit(con, "⏺ applied")
    out = con.file.getvalue()
    assert "applied" in out and "…" not in out


def test_print_fit_never_wraps_multiline():
    con = _con(20)
    PR.print_fit(con, "⏺ " + ("verylongtokenwithoutspaces" * 4))
    assert con.file.getvalue().count("\n") == 1        # exactly one line, no wrap


def test_print_fit_failsoft_on_bad_console():
    class _Boom:
        width = 30
        def print(self, *a, **k):
            raise RuntimeError("nope")
    PR.print_fit(_Boom(), "anything")                  # must not raise


# --------------------------------------------------------------------------- pulse
import asyncio  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


def test_pulse_noop_when_not_tty(monkeypatch):
    monkeypatch.setattr(PR, "real_stdout_isatty", lambda: False)
    con = MagicMock()
    ran = {}

    async def go():
        async with PR.pulse(con, "⏺ synthesizing"):
            ran["body"] = True

    asyncio.run(go())
    assert ran["body"] is True
    con.status.assert_not_called()             # no spinner in headless


def test_pulse_starts_stops_and_restores_cursor(monkeypatch):
    monkeypatch.setattr(PR, "real_stdout_isatty", lambda: True)
    con = MagicMock()
    status = MagicMock()
    con.status.return_value = status

    async def go():
        async with PR.pulse(con, "⏺ synthesizing"):
            pass

    asyncio.run(go())
    status.start.assert_called_once()
    status.stop.assert_called_once()
    con.show_cursor.assert_called_with(True)   # cursor armor


def test_pulse_restores_cursor_on_exception(monkeypatch):
    monkeypatch.setattr(PR, "real_stdout_isatty", lambda: True)
    con = MagicMock()
    status = MagicMock()
    con.status.return_value = status

    async def go():
        async with PR.pulse(con, "⏺ x"):
            raise ValueError("boom")

    with pytest_raises(ValueError):
        asyncio.run(go())
    status.stop.assert_called_once()           # stopped despite exception
    con.show_cursor.assert_called_with(True)


import pytest as _pytest  # noqa: E402


def pytest_raises(exc):
    return _pytest.raises(exc)
