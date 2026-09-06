"""The tool-narration proxy must resolve the comm the service actually holds.

## The defect

`GovernedLoopService` stores its governance stack as ``self._stack``. The
late-bound comm proxy that carries every Venom tool call to the render
transports read ``getattr(self._gls, "_governance_stack", None)`` — an
attribute the service has never had. With a default of ``None``, the
lookup never raised: ``_transports`` resolved to ``[]``, ``_emit`` returned
before touching a transport, and the channel counted nothing as a failure
because nothing had failed.

So the orchestrator's intent lines (emitted through the real comm)
reached the cockpit, and not one tool call ever did. Measured
2026-09-06 on a live session: 8 tool rounds in a 240 s window, 0 tool
blocks on the socket, 0 recorded failures.

## What these tests hold

That the proxy resolves the SAME comm the service holds; that a rename of
the real attribute cannot fail silently again; and that a tool call fed
through the proxy end to end reaches a render transport's mirror.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.governance import governed_loop_service as gls_mod  # noqa: E402


def _proxy_class():
    """Extract `_LateCommProxy` from the method that defines it, without
    booting a service: exec the class statement in a namespace that
    supplies what its body references."""
    src = inspect.getsource(gls_mod)
    start = src.index("            class _LateCommProxy:")
    end = src.index("            self._tool_narration = _TNC(_LateCommProxy(self))", start)
    body = "\n".join(l[12:] for l in src[start:end].splitlines())
    ns = {"Any": object, "logger": gls_mod.logger}
    exec(body, ns)  # noqa: S102 — the repo's own source under test
    return ns["_LateCommProxy"]


def _service_with(comm):
    """A stand-in with exactly the attribute the real service assigns."""
    return SimpleNamespace(_stack=SimpleNamespace(comm=comm))


# ---------------------------------------------------------------------------
# The identity: one comm, the one the service holds
# ---------------------------------------------------------------------------


def test_the_service_stores_its_stack_as__stack() -> None:
    """The attribute the proxy must read. Pinned so a rename of either
    side fails here rather than silently in production."""
    src = inspect.getsource(gls_mod.GovernedLoopService.__init__)
    assert "self._stack = stack" in src


def test_the_proxy_names_no_attribute_the_service_lacks() -> None:
    src = inspect.getsource(gls_mod)
    # Anchor WITH the indent so the slice begins at column 0 of the line
    # and the dedent below strips whitespace, not the class keyword.
    start = src.index("            class _LateCommProxy:")
    end = src.index("            self._tool_narration = _TNC(", start)
    proxy_src = src[start:end]
    tree = ast.parse("\n".join(l[12:] for l in proxy_src.splitlines()))
    # CODE, not prose. The docstring is where the old name belongs — as
    # history — so the check walks attribute accesses and string constants
    # that are NOT a statement's docstring.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstrings.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "_governance_stack", (
                "the proxy read an attribute GovernedLoopService has never had")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            assert node.value != "_governance_stack", (
                "a getattr by that name is the same defect with a string")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            if len(node.args) >= 1 and ast.dump(node.args[0]).endswith("attr='_gls')"):
                pytest.fail("no getattr-with-default on the service: a missing "
                            "stack must raise into the channel's fault "
                            "isolation and be COUNTED, not hidden")


def test_the_proxy_resolves_the_service_s_transports() -> None:
    Proxy = _proxy_class()
    sentinel = object()
    comm = SimpleNamespace(_transports=[sentinel])
    proxy = Proxy(_service_with(comm))
    assert proxy._transports == [sentinel]


def test_a_service_with_no_stack_is_visible_not_silent() -> None:
    """The channel wraps delivery in fault isolation and counts a
    failure. The proxy must let that happen rather than returning an
    empty list that reads as 'delivered to nobody'."""
    Proxy = _proxy_class()
    proxy = Proxy(SimpleNamespace())          # no _stack at all
    with pytest.raises(AttributeError):
        _ = proxy._transports


def test_emit_prefers_the_comm_s_own_emit() -> None:
    Proxy = _proxy_class()
    seen = []

    async def _emit(msg):
        seen.append(msg)

    comm = SimpleNamespace(_emit=_emit, _transports=[])
    proxy = Proxy(_service_with(comm))
    asyncio.run(proxy._emit("m"))
    assert seen == ["m"]


def test_emit_falls_back_to_transport_fan_out() -> None:
    Proxy = _proxy_class()
    got = []

    class _T:
        async def send(self, msg):
            got.append(msg)

    comm = SimpleNamespace(_transports=[_T()])
    proxy = Proxy(_service_with(comm))
    asyncio.run(proxy._emit("m"))
    assert got == ["m"]


def test_a_failing_transport_does_not_stop_the_others() -> None:
    Proxy = _proxy_class()
    got = []

    class _Bad:
        async def send(self, msg):
            raise RuntimeError("boom")

    class _Good:
        async def send(self, msg):
            got.append(msg)

    comm = SimpleNamespace(_transports=[_Bad(), _Good()])
    proxy = Proxy(_service_with(comm))
    asyncio.run(proxy._emit("m"))
    assert got == ["m"]


# ---------------------------------------------------------------------------
# End to end: a tool call reaches a render transport's mirror
# ---------------------------------------------------------------------------


def test_a_tool_call_reaches_the_cockpit_mirror_through_the_proxy() -> None:
    from backend.core.ouroboros.governance.comm_protocol import CommProtocol
    from backend.core.ouroboros.governance.claude_style_transport import (
        ClaudeStyleTransport,
    )
    from backend.core.ouroboros.governance.tool_narration import (
        ToolNarrationChannel,
    )
    Proxy = _proxy_class()
    mirrored = []
    transport = ClaudeStyleTransport(console=None)
    transport.markup_mirror = mirrored.append
    comm = CommProtocol()
    comm._transports.append(transport)
    channel = ToolNarrationChannel(Proxy(_service_with(comm)))

    async def main():
        channel.emit(op_id="op-proxy", tool_name="search_code", round_index=0,
                     args_summary="pattern=foo", status="")
        channel.emit(op_id="op-proxy", tool_name="search_code", round_index=0,
                     args_summary="pattern=foo", result_preview="3 hits",
                     duration_ms=4.0, status="success")
        await asyncio.sleep(0.3)

    asyncio.run(main())
    assert channel.failure_count == 0
    assert any("⏺" in m and "pattern=foo" in m for m in mirrored), mirrored
