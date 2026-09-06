"""Phase 3 — symbol-centered AST slicing: protect the goal's target symbols.

When a target file exceeds the prefill budget, the slicer must skeletonize the
NON-focus functions first and keep the operator-declared target symbol's body
intact — instead of the size-greedy Slice 11.4.1 tiers dropping that symbol
precisely because it is large. A declared focus symbol engages the slicer even
when the general AST-slice flag is off (a sanctioned goal must not need the
global flag flipped), and a measured prefill budget overrides the route's
static estimate so the slice fits the actual KV cache.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.ouroboros.governance import providers as pv


def _focus_source() -> str:
    """A big NON-target helper + a smaller TARGET function ``sub``."""
    big = "\n".join(
        f"    x{i} = compute_{i}(x{i - 1}) + {i}  # filler {i}"
        for i in range(1, 400)
    )
    return (
        '"""module docstring."""\n\n'
        "def helper_big(x0):\n"
        '    """A large NON-target helper the size-greedy slicer drops first."""\n'
        + big
        + "\n    return x399\n\n\n"
        "def sub(a, b):\n"
        '    """The operator TARGET symbol — must survive slicing intact."""\n'
        "    result = a - b\n"
        '    assert result >= 0, "sub went negative"\n'
        "    return result\n"
    )


def test_chunk_is_focus_matches_name_and_qualified():
    class _C:
        def __init__(self, name, qual):
            self.name = name
            self.qualified_name = qual

    assert pv._chunk_is_focus(_C("sub", "m.Widget.sub"), {"sub"})
    assert pv._chunk_is_focus(_C("sub", "m.Widget.sub"), {"Widget.sub"})
    assert pv._chunk_is_focus(_C("add", "m.Widget.add"), {"m.Widget.add"})
    assert not pv._chunk_is_focus(_C("other", "m.other"), {"sub"})
    assert not pv._chunk_is_focus(_C("sub", "m.sub"), set())


def test_focus_symbol_survives_while_non_focus_is_skeletonized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARVIS_GEN_AST_SLICE_MIN_CHARS", "1000")
    out = pv._maybe_ast_outline(
        abs_path=Path("widget.py"), raw_path="widget.py",
        full_content=_focus_source(), op_id="t", provider_route="standard",
        focus_symbols=("sub",), prefill_budget_chars=1500,
    )
    assert out is not None
    # The TARGET symbol's body is intact...
    assert "sub went negative" in out
    # ...and the big non-target helper is skeletonized away.
    assert "filler 200" not in out
    # ...via a focus-aware tier.
    assert "_focus" in out


def test_focus_engages_slicer_even_with_master_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JARVIS_GEN_AST_SLICE_ENABLED", raising=False)
    monkeypatch.setenv("JARVIS_GEN_AST_SLICE_MIN_CHARS", "1000")
    with_focus = pv._maybe_ast_outline(
        abs_path=Path("widget.py"), raw_path="widget.py",
        full_content=_focus_source(), op_id="t", provider_route="standard",
        focus_symbols=("sub",), prefill_budget_chars=1500,
    )
    without = pv._maybe_ast_outline(
        abs_path=Path("widget.py"), raw_path="widget.py",
        full_content=_focus_source(), op_id="t", provider_route="standard",
        prefill_budget_chars=1500,
    )
    assert with_focus is not None       # focus engaged it
    assert without is None              # flag off + no focus = legacy fallthrough


def test_prefill_budget_protects_focus_under_a_tight_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARVIS_GEN_AST_SLICE_MIN_CHARS", "1000")
    tight = pv._maybe_ast_outline(
        abs_path=Path("widget.py"), raw_path="widget.py",
        full_content=_focus_source(), op_id="t", provider_route="standard",
        focus_symbols=("sub",), prefill_budget_chars=1200,
    )
    assert tight is not None
    assert "sub went negative" in tight
