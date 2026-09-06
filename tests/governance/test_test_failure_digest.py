"""test_failure_digest — the SPECIFIC assertion/AST cause, parsed deterministically.

Phase 2 of the /goal pivot: a candidate dying at ``fc='test'`` must tell the
re-planner, the cockpit, and the GRPO corpus WHY — the failing node + the
``E ...`` assertion + the error class — not a blind 150-char stdout tail.
"""
from __future__ import annotations

from backend.core.ouroboros.governance.test_failure_digest import (
    TestFailureDigest,
    digest_from_adapter_results,
    digest_from_text,
)

_PYTEST = """\
tests/test_widget.py::test_sub FAILED
=================================== FAILURES ===================================
___________________________________ test_sub ___________________________________
    def test_sub():
>       assert w.sub(5, 3) == 3
E       assert 2 == 3
E        +  where 2 = sub(5, 3)
tests/test_widget.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_widget.py::test_sub - assert 2 == 3
========================= 1 failed, 1 passed in 0.03s ==========================
"""


class _TR:
    def __init__(self, stdout="", failed_tests=(), total=0, failed=0):
        self.stdout = stdout
        self.failed_tests = failed_tests
        self.total = total
        self.failed = failed


class _AR:
    def __init__(self, adapter, passed, tr):
        self.adapter = adapter
        self.passed = passed
        self.test_result = tr


def test_pytest_assertion_is_extracted():
    d = digest_from_text(_PYTEST, failed_tests=("tests/test_widget.py::test_sub",))
    assert d.error_class == "AssertionError"
    assert "tests/test_widget.py::test_sub" in d.failed_tests
    assert any("2 == 3" in a for a in d.assertions)
    assert "tests/test_widget.py:12" in d.locations
    assert "AssertionError" in d.headline and "test_sub" in d.headline
    assert bool(d) is True


def test_syntax_error_is_classified():
    d = digest_from_text("backend/x.py:3: SyntaxError: invalid syntax")
    assert d.error_class == "SyntaxError"
    assert "SyntaxError" in d.headline


def test_adapter_aggregation_counts_and_scopes():
    ars = [
        _AR("python", False, _TR(_PYTEST, ("tests/test_widget.py::test_sub",), 2, 1)),
        _AR("cpp", True, _TR("", (), 3, 0)),
    ]
    d = digest_from_adapter_results(ars)
    assert d.test_total == 2 and d.test_failed == 1
    assert d.adapters_failed == ("python",)
    assert "AssertionError" in d.headline
    assert "failed: tests/test_widget.py::test_sub" in d.detail


def test_all_passing_is_falsy():
    d = digest_from_adapter_results([_AR("python", True, _TR("", (), 3, 0))])
    assert not d
    assert d.to_dict()["headline"] == ""


def test_never_raises_on_garbage():
    # bytes-ish, None, empty, weird — every one degrades to an empty digest.
    assert isinstance(digest_from_text(""), TestFailureDigest)
    assert isinstance(digest_from_text(None), TestFailureDigest)  # type: ignore[arg-type]
    assert isinstance(digest_from_adapter_results(None), TestFailureDigest)
    assert isinstance(digest_from_adapter_results([object()]), TestFailureDigest)


def test_bounds_are_env_tunable(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_DIGEST_MAX_ASSERTIONS", "1")
    many = "\n".join(f"E   assert {i} == {i + 1}" for i in range(10))
    d = digest_from_text(many)
    assert len(d.assertions) == 1
