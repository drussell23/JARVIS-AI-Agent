"""Structured, bounded digest of a validation failure — the SPECIFIC assertion
or AST/syntax error, not a blind stdout tail.

Candidates die at ``fc='test'`` and the operator (and the model's next GENERATE
attempt, and the GRPO corpus) never learned WHY: ``_run_validation_core``
summarised a failure as the last 150 characters of stdout, which for pytest is
the ``"1 failed, 3 passed"`` epilogue — a count, not a cause. This extracts the
cause deterministically from the adapter's OWN output: the failing node id, the
``E ...`` assertion line, the ``<file>:<line>: <ErrorClass>`` anchor, and the
error class — one high-signal :class:`TestFailureDigest` three consumers share
so they can never drift:

  * the re-planner (``op_context.replan_inputs`` → the next GENERATE prompt),
  * the cockpit VALIDATE heartbeat (the operator sees the assertion live), and
  * the execution ledger + GRPO trajectory recorder (training learns the cause).

No LLM. Every bound is env-tunable (no magic literals at a call site). NEVER
raises — a telemetry parser must not be able to fail a validation.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

# ── env-tunable bounds (no hardcoded values at the call sites) ──────────────
_ENV_MAX_ASSERTIONS = "JARVIS_TEST_DIGEST_MAX_ASSERTIONS"
_ENV_MAX_NODES = "JARVIS_TEST_DIGEST_MAX_NODES"
_ENV_HEADLINE_CHARS = "JARVIS_TEST_DIGEST_HEADLINE_CHARS"
_ENV_DETAIL_CHARS = "JARVIS_TEST_DIGEST_DETAIL_CHARS"
_ENV_ASSERTION_CHARS = "JARVIS_TEST_DIGEST_ASSERTION_CHARS"


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip() or default)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# Lines pytest prints for the actual failure, e.g. ``E   assert 3 == 5``.
_E_LINE = re.compile(r"^\s*E\s{2,}(?P<msg>\S.*)$")
# The traceback anchor: ``tests/x.py:42: AssertionError`` (class is optional).
_LOC_LINE = re.compile(
    r"^(?P<path>[^\s:]+\.[A-Za-z0-9_]+):(?P<line>\d+):\s*(?P<cls>[A-Za-z_][\w.]*)?\s*$"
)
# The short-summary line: ``FAILED tests/x.py::test_y - AssertionError: ...``.
_FAILED_LINE = re.compile(
    r"^(?:FAILED|ERROR)\s+(?P<node>\S+)(?:\s*-\s*(?P<rest>.*))?$"
)
# A bare ``SyntaxError: ...`` / ``ImportError: ...`` / ``TypeError: ...`` line.
_ERRCLASS_LINE = re.compile(
    r"(?P<cls>[A-Za-z_][\w.]*(?:Error|Exception|Failed|Warning)):\s*(?P<msg>.+)"
)


@dataclass(frozen=True)
class TestFailureDigest:
    """The parsed cause of a validation failure. Pure state; ``bool(digest)``
    is True only when something specific was actually extracted."""

    error_class: str = ""
    failed_tests: Tuple[str, ...] = ()
    assertions: Tuple[str, ...] = ()
    locations: Tuple[str, ...] = ()
    test_total: int = 0
    test_failed: int = 0
    headline: str = ""
    detail: str = ""
    adapters_failed: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.headline or self.assertions or self.failed_tests)

    def to_dict(self) -> Dict[str, Any]:
        """The shape the ledger stores + GRPO reads. Bounded by construction."""
        return {
            "error_class": self.error_class,
            "failed_tests": list(self.failed_tests),
            "assertions": list(self.assertions),
            "locations": list(self.locations),
            "test_total": self.test_total,
            "test_failed": self.test_failed,
            "headline": self.headline,
            "detail": self.detail,
            "adapters_failed": list(self.adapters_failed),
        }


def digest_from_text(
    stdout: str,
    *,
    failed_tests: Sequence[str] = (),
    error_class_hint: str = "",
) -> TestFailureDigest:
    """Parse one adapter's raw output into a digest. The core; ``NEVER raises``.

    ``failed_tests`` (the adapter's own parsed node ids, when it has them) is
    trusted over scraping and merged with any ``FAILED …`` lines found here.
    ``error_class_hint`` seeds the class for a shape the regexes miss (e.g. an
    AST pre-flight that already produced ``"SyntaxError: …"``)."""
    try:
        text = str(stdout or "")
        max_assert = _int_env(_ENV_MAX_ASSERTIONS, 6)
        max_nodes = _int_env(_ENV_MAX_NODES, 6)
        assert_chars = _int_env(_ENV_ASSERTION_CHARS, 200)

        assertions: List[str] = []
        nodes: List[str] = list(dict.fromkeys(str(t).strip() for t in failed_tests if str(t).strip()))
        locations: List[str] = []
        error_class = str(error_class_hint or "").strip()

        for raw in text.splitlines():
            m = _E_LINE.match(raw)
            if m:
                msg = m.group("msg").strip()
                if msg and msg not in assertions:
                    assertions.append(msg[:assert_chars])
                continue
            m = _FAILED_LINE.match(raw.strip())
            if m:
                node = m.group("node").strip()
                if node and node not in nodes:
                    nodes.append(node)
                rest = (m.group("rest") or "").strip()
                if rest:
                    ec = _ERRCLASS_LINE.match(rest)
                    if ec and not error_class:
                        error_class = ec.group("cls")
                    if rest not in assertions:
                        assertions.append(rest[:assert_chars])
                continue
            m = _LOC_LINE.match(raw.strip())
            if m:
                loc = f"{m.group('path')}:{m.group('line')}"
                if loc not in locations:
                    locations.append(loc)
                if m.group("cls") and not error_class:
                    error_class = m.group("cls")
                continue
            if not error_class:
                ec = _ERRCLASS_LINE.search(raw)
                if ec:
                    error_class = ec.group("cls")
                    cand = f"{ec.group('cls')}: {ec.group('msg').strip()}"
                    if cand not in assertions:
                        assertions.append(cand[:assert_chars])

        assertions = assertions[:max_assert]
        nodes = nodes[:max_nodes]
        locations = locations[:max_nodes]

        # Headline: the class + the first failing node + the first assertion —
        # the one line the re-planner and the operator read first.
        head_parts: List[str] = []
        if error_class:
            head_parts.append(error_class)
        if nodes:
            head_parts.append(nodes[0].rsplit("::", 1)[-1])
        if assertions:
            head_parts.append(assertions[0])
        headline = " · ".join(p for p in head_parts if p)[
            : _int_env(_ENV_HEADLINE_CHARS, 240)
        ]

        detail_lines: List[str] = []
        if nodes:
            detail_lines.append("failed: " + ", ".join(nodes))
        for a in assertions:
            detail_lines.append("E " + a)
        if locations:
            detail_lines.append("at " + ", ".join(locations))
        detail = "\n".join(detail_lines)[: _int_env(_ENV_DETAIL_CHARS, 1200)]

        return TestFailureDigest(
            error_class=error_class,
            failed_tests=tuple(nodes),
            assertions=tuple(assertions),
            locations=tuple(locations),
            headline=headline,
            detail=detail,
        )
    except Exception:  # noqa: BLE001 — a telemetry parser NEVER fails a verdict
        return TestFailureDigest()


def digest_from_adapter_results(adapter_results: Any) -> TestFailureDigest:
    """Aggregate the FAILED adapters of a MultiAdapter validation into one
    digest. Reuses each adapter's own ``test_result`` (failed_tests, stdout,
    total/failed counts) — never re-derives what the adapter already parsed.
    NEVER raises."""
    try:
        results = list(adapter_results or ())
        failed = [r for r in results if not getattr(r, "passed", True)]
        if not failed:
            return TestFailureDigest()

        total = failed_ct = 0
        all_nodes: List[str] = []
        all_assert: List[str] = []
        all_loc: List[str] = []
        error_class = ""
        adapters_failed: List[str] = []
        for r in failed:
            adapters_failed.append(str(getattr(r, "adapter", "") or "?"))
            tr = getattr(r, "test_result", None)
            stdout = str(getattr(tr, "stdout", "") or "")
            fnodes = tuple(getattr(tr, "failed_tests", ()) or ())
            total += int(getattr(tr, "total", 0) or 0)
            failed_ct += int(getattr(tr, "failed", 0) or 0)
            d = digest_from_text(stdout, failed_tests=fnodes)
            error_class = error_class or d.error_class
            for n in d.failed_tests:
                if n not in all_nodes:
                    all_nodes.append(n)
            for a in d.assertions:
                if a not in all_assert:
                    all_assert.append(a)
            for loc in d.locations:
                if loc not in all_loc:
                    all_loc.append(loc)

        max_assert = _int_env(_ENV_MAX_ASSERTIONS, 6)
        max_nodes = _int_env(_ENV_MAX_NODES, 6)
        all_assert = all_assert[:max_assert]
        all_nodes = all_nodes[:max_nodes]
        all_loc = all_loc[:max_nodes]

        head_parts: List[str] = []
        if error_class:
            head_parts.append(error_class)
        if all_nodes:
            head_parts.append(all_nodes[0].rsplit("::", 1)[-1])
        if all_assert:
            head_parts.append(all_assert[0])
        headline = " · ".join(p for p in head_parts if p)[
            : _int_env(_ENV_HEADLINE_CHARS, 240)
        ]

        detail_lines: List[str] = []
        if adapters_failed:
            detail_lines.append("adapters: " + ", ".join(adapters_failed))
        if all_nodes:
            detail_lines.append("failed: " + ", ".join(all_nodes))
        for a in all_assert:
            detail_lines.append("E " + a)
        if all_loc:
            detail_lines.append("at " + ", ".join(all_loc))
        detail = "\n".join(detail_lines)[: _int_env(_ENV_DETAIL_CHARS, 1200)]

        return TestFailureDigest(
            error_class=error_class,
            failed_tests=tuple(all_nodes),
            assertions=tuple(all_assert),
            locations=tuple(all_loc),
            test_total=total,
            test_failed=failed_ct,
            headline=headline,
            detail=detail,
            adapters_failed=tuple(adapters_failed),
        )
    except Exception:  # noqa: BLE001
        return TestFailureDigest()


__all__ = [
    "TestFailureDigest",
    "digest_from_adapter_results",
    "digest_from_text",
]
