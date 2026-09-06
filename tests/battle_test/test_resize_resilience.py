"""A terminal resize (SIGWINCH) mid-stream reflows to the new width without
corrupting the prompt or orphaning lines.

The mechanism has two halves, and this pins both:
  * the CLIENT re-sends its caps on SIGWINCH and CHAINS to whatever handler
    already owns the signal (prompt_toolkit's, which reflows the prompt) —
    never clobbering it, and never stacking a second copy of its own;
  * the DAEMON reflows narration to the freshly-declared width, because the
    transport reads ``terminal_capabilities.effective_width`` at render time
    (Phase 1), so the next line uses the new geometry.
"""
from __future__ import annotations

import signal
import sys

import pytest

from backend.core.ouroboros.battle_test.cockpit_attach import CockpitAttachClient


needs_sigwinch = pytest.mark.skipif(
    not hasattr(signal, "SIGWINCH"), reason="no SIGWINCH on this platform")


def _client():
    c = CockpitAttachClient.__new__(CockpitAttachClient)
    c._caps_sent = 0

    def _send_caps():
        c._caps_sent += 1
        return True

    c.send_caps = _send_caps          # type: ignore[assignment]
    return c


@needs_sigwinch
def test_the_listener_chains_the_prior_handler_and_resends_caps(monkeypatch):
    prior_calls = []
    monkeypatch.setattr(signal, "getsignal", lambda s: (lambda *a: prior_calls.append(1)))
    installed = {}
    monkeypatch.setattr(signal, "signal",
                        lambda s, h: installed.__setitem__("h", h))
    c = _client()
    assert c.install_resize_listener() is True
    handler = installed["h"]
    handler(signal.SIGWINCH, None)                 # simulate a resize
    assert c._caps_sent == 1                        # daemon learns the new width
    assert prior_calls == [1]                       # prompt_toolkit still reflows


@needs_sigwinch
def test_reinstall_is_idempotent_and_does_not_recurse(monkeypatch):
    monkeypatch.setattr(signal, "getsignal", lambda s: None)
    installs = []
    monkeypatch.setattr(signal, "signal", lambda s, h: installs.append(h))
    c = _client()
    assert c.install_resize_listener() is True
    assert c.install_resize_listener() is True      # second call — no-op
    assert len(installs) == 1                        # our handler installed ONCE
    installs[0](signal.SIGWINCH, None)
    assert c._caps_sent == 1                         # exactly one re-send, no recursion


@needs_sigwinch
def test_off_the_main_thread_it_declines_without_raising(monkeypatch):
    monkeypatch.setattr(signal, "getsignal", lambda s: None)

    def _boom(s, h):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", _boom)
    c = _client()
    assert c.install_resize_listener() is False     # embedded cockpit — not an error


def test_the_daemon_reflows_to_the_new_width_on_the_next_line(monkeypatch):
    """After a resize re-declares the width, the transport's narration
    viewport reports the NEW width — so the next 💭 line reflows to it."""
    from backend.core.ouroboros.governance import claude_style_transport as cst
    from backend.core.ouroboros.battle_test import terminal_capabilities as tc
    monkeypatch.delenv("JARVIS_CLAUDE_STYLE_LINE_CHARS", raising=False)
    width = {"cols": 80}
    monkeypatch.setattr(tc, "effective_width", lambda default=None: width["cols"])
    assert cst.narration_viewport() == 80
    width["cols"] = 200                              # SIGWINCH → new caps declared
    assert cst.narration_viewport() == 200           # picked up at render time
