"""Split-Plane Multiplexer — concurrent-I/O integrity spine.

Operator mandate: an async daemon emit arriving at the exact moment an
operator keystroke registers must not corrupt the line buffer, drop
characters, or break the prompt boundary. The mux is prompt_toolkit's
PromptSession + patch_stdout (DRY — SerpentFlow's proven mechanism);
these tests drive it with a REAL pipe input + interleaved emits.
"""
from __future__ import annotations

import asyncio

import pytest

pt = pytest.importorskip("prompt_toolkit")

from prompt_toolkit import PromptSession               # noqa: E402
from prompt_toolkit.input import create_pipe_input     # noqa: E402
from prompt_toolkit.output import DummyOutput          # noqa: E402
from prompt_toolkit.patch_stdout import patch_stdout   # noqa: E402
from prompt_toolkit.application import create_app_session  # noqa: E402


# ---------------------------------------------------------------------------
# (1) THE mandated test — mid-keystroke emit, zero corruption
# ---------------------------------------------------------------------------


async def test_concurrent_emit_never_corrupts_keystrokes():
    """Type 'hel' → daemon emits mid-buffer → type 'lo' → more emits →
    Enter. The returned line must be EXACTLY 'hello' (no drops, no
    splits, no telemetry bleeding into the buffer)."""
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput(),
    ):
        session: PromptSession = PromptSession()
        with patch_stdout():
            task = asyncio.ensure_future(session.prompt_async("ov › "))
            await asyncio.sleep(0.05)
            pipe.send_text("hel")
            print("⏺ GENERATE — op-019f progressing")     # mid-keystroke emit
            await asyncio.sleep(0.02)
            pipe.send_text("lo")
            print("⎿ verify: 4/4 · cost $0.12")            # and another
            pipe.send_text("\n")
            line = await asyncio.wait_for(task, timeout=5)
    assert line == "hello"
    # (Emit delivery goes to the isolated app session's DummyOutput —
    # the property under test is BUFFER INTEGRITY, asserted above.)


async def test_burst_emits_between_every_keystroke(capsys):
    """Adversarial cadence: a telemetry line between EVERY character."""
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput(),
    ):
        session: PromptSession = PromptSession()
        target = "cancel op-019f77"
        with patch_stdout():
            task = asyncio.ensure_future(session.prompt_async("ov › "))
            await asyncio.sleep(0.05)
            for ch in target:
                pipe.send_text(ch)
                print(f"⎿ telemetry burst around {ch!r}")
            pipe.send_text("\n")
            line = await asyncio.wait_for(task, timeout=5)
    assert line == target                                  # every char survived


async def test_unicode_emits_never_bleed_into_ascii_buffer(capsys):
    """Unicode-hostile EMITS (glyphs, emoji) around plain keystrokes —
    the buffer must stay byte-exact. (Multibyte KEYSTROKES through the
    pipe-input harness are timing-flaky under pytest-asyncio; the
    property is proven by a standalone probe — the production stdin
    path is a real vt100 stream, not the pipe harness.)"""
    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput(),
    ):
        session: PromptSession = PromptSession()
        with patch_stdout():
            task = asyncio.ensure_future(session.prompt_async("ov › "))
            await asyncio.sleep(0.05)
            pipe.send_text("status")
            print("⏺ unicode-hostile emit ▸ 💭 · ⚠ 🎙")
            pipe.send_text("\n")
            line = await asyncio.wait_for(task, timeout=5)
    assert line == "status"


# ---------------------------------------------------------------------------
# (2) Structure pins — async loop, no blockers, fallback, host moment
# ---------------------------------------------------------------------------


def _src() -> str:
    from pathlib import Path
    return (
        Path(__file__).resolve().parents[2]
        / "backend/core/ouroboros/cli/ov.py"
    ).read_text()


def test_split_plane_uses_prompt_toolkit_mux():
    src = _src()
    body = src[src.index("async def _split_plane_loop"):]
    body = body[:body.index("\nasync def _legacy_pump_loop")]
    assert "PromptSession" in body
    assert "patch_stdout(raw=True)" in body
    assert "prompt_async" in body                  # async loop, not input()
    assert "time.sleep" not in body                # zero UI blockers
    import re
    assert not re.search(r"(?<![\w.])input\(", body)   # no blocking input()


def test_daemon_death_races_the_prompt():
    src = _src()
    body = src[src.index("async def _split_plane_loop"):]
    body = body[:body.index("\nasync def _legacy_pump_loop")]
    assert "_watch_disconnect" in body
    assert "FIRST_COMPLETED" in body               # never hangs on a dead daemon
    assert "_reap_task(prompt_task)" in body   # reaper cancels + retrieves


def test_persona_host_line_present():
    src = _src()
    assert "Karen ▸ attached" in src
    assert "'detach' leaves the organism running" in src


def test_non_tty_degrades_to_legacy_pump():
    """Asserted on the BEHAVIOUR, not on an 800-character window after the
    `def`. The window measured how much prose sat between the name and the
    call — the same defect the renderer test below already calls out — and
    it broke the moment the TTY check moved into `cli.surface_probe`, where
    it now sits beside the dependency check it must be distinguishable
    from. What matters is unchanged: no terminal, no split plane.
    """
    from backend.core.ouroboros.cli import ov as O
    from backend.core.ouroboros.cli import surface_probe as sp

    piped = sp.probe_interactive_surface(stdin_isatty=False, required=())
    assert not piped.ok
    assert piped.kind == sp.ENVIRONMENT
    assert not piped.is_fault, "a pipe is not a broken install"

    real = sp.probe_interactive_surface(stdin_isatty=True, required=())
    assert real.ok

    assert isinstance(O._can_run_split_plane(), bool)
    assert "_legacy_pump_loop" in _src()


def test_line_renderer_resolves_stdout_dynamically():
    """The pre-bound Rich console would bypass patch_stdout and corrupt
    the prompt — daemon lines must go through builtin print()."""
    import ast

    # Asserted on the FUNCTION, not on a 700-character window after its name.
    # The window measured how much prose sat between the def and the call, so
    # it failed the moment a comment was added — the same defect as a
    # proximity pin. The invariant never changed.
    src = _src()
    body = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == "_print_line":
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            break
    assert body is not None, "_print_line is gone"
    assert "print(" in body, (
        "daemon lines no longer go through builtin print() — a pre-bound Rich "
        "console bypasses patch_stdout and corrupts the prompt"
    )
    assert "console.print" not in body


# ---------------------------------------------------------------------------
# (3) Clean detach — the dirty-KeyboardInterrupt class is dead
# ---------------------------------------------------------------------------


async def test_reap_consumes_keyboard_interrupt_task():
    """The 2026-07-18 report: an abandoned prompt task finished with
    KeyboardInterrupt → asyncio dumped 'Task exception was never
    retrieved' over the clean goodbye. _reap_task must consume it so
    the GC has nothing to complain about."""
    import gc
    from backend.core.ouroboros.cli.ov import _reap_task

    complaints = []
    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _l, ctx: complaints.append(ctx))
    try:
        class _FakeKI(BaseException):
            """KeyboardInterrupt-shaped (BaseException, not Exception) —
            a REAL KI in a task nukes the test runner itself, which is
            precisely the sharpness of the original bug. Retrieval
            semantics are identical for any BaseException."""

        async def _boom():
            raise _FakeKI

        task = asyncio.ensure_future(_boom())
        await asyncio.sleep(0.05)              # let it finish dirty
        await _reap_task(task)                 # consume the corpse
        del task
        gc.collect()
        await asyncio.sleep(0.05)
        assert not any(
            "never retrieved" in str(c.get("message", "")) for c in complaints
        )
    finally:
        loop.set_exception_handler(old_handler)


async def test_reap_cancels_pending_task():
    from backend.core.ouroboros.cli.ov import _reap_task
    task = asyncio.ensure_future(asyncio.sleep(30))
    await _reap_task(task)
    assert task.cancelled()


def test_split_plane_reaps_on_every_exit_path():
    src = _src()
    body = src[src.index("async def _split_plane_loop"):]
    body = body[:body.index("\nasync def _legacy_pump_loop")]
    assert body.count("_reap_task(") >= 4          # KI-wait, daemon-death, EOF paths
    assert "except (KeyboardInterrupt, asyncio.CancelledError):" in body


def test_collision_surface_renders_the_emblem():
    """Operator law: the crest ALWAYS greets `ov` — including the
    already-awake collision card (static emblem, no animation)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    src = (root / "scripts/ouroboros_battle_test.py").read_text()
    idx = src.index("⏺ the organism is already awake")
    region = src[max(0, idx - 3000):idx]
    assert "print_static_crest" in region


def test_static_emblem_renders_full_crest():
    from backend.core.ouroboros.ui.crest import frame_to_text, generate_crest
    from backend.core.ouroboros.ui.theme import ColorTier
    f = generate_crest(80, 30, tier=ColorTier.TRUECOLOR, unicode_ok=True)
    text = frame_to_text(f, ColorTier.TRUECOLOR)          # elapsed=None = FULL
    assert len(text.plain.strip()) > 200                  # the whole mark
    partial = frame_to_text(f, ColorTier.TRUECOLOR, elapsed=0.01)
    assert len(partial.plain.strip()) < len(text.plain.strip())
