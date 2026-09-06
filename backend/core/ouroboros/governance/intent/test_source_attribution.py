"""Slice 6 — deterministic AST test→source attribution bridge.

THE GAP (battle-test Run #16): a TestFailure signal's ``target_files``
was definitionally the failing TEST file (``test_id.split("::")[0]``),
so APPLY scope never contained the module under test, the
``file_scope_mismatch`` guard REJECTED correct source repairs, and
VERIFY died deterministically at pass_rate<1.0 while the source bug
survived. This module resolves the source loci a test exercises by
parsing the test module's AST and tracing its ACTUAL imports — never
path heuristics (mandate 1), never a new parser (mandate 3: composes
``reverse_dep_resolver``'s sanctioned extractor + the new inverse
module→path map), alias/relative/indirection-aware with typed fail-fast
(mandate 4). Traceback frames are a ranking TIE-BREAKER only: for the
Run-16 class (assertion failures) the deepest in-repo frame is the test
line itself, so imports must be primary.
"""
from __future__ import annotations

import ast
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from backend.core.ouroboros.governance.reverse_dep_resolver import (
    _is_test_module,
    _module_from_relpath,
    _relpath_under_root,
    build_module_to_path,
    extract_module_imports,
)

ATTRIBUTION_SCHEMA_VERSION = 1

# Evidence kinds, ranked: direct imports are the primary deterministic
# signal; patch-target strings recover mock-indirection (~17% of suite).
_KIND_DIRECT = "direct_import"
_KIND_PATCH = "patch_target"


class AttributionUnresolved(Exception):
    """Typed fail-fast (mandate 4): the source under test cannot be
    deterministically resolved. Carries a machine-readable ``reason`` so
    the signal evidence (and the scope gate) can act on it."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"test->source attribution unresolved: {reason}"
            + (f" ({detail})" if detail else "")
        )


@dataclass(frozen=True)
class Attribution:
    """Resolved loci. All paths repo-relative POSIX. ``source_loci`` is
    never empty (emptiness raises ``AttributionUnresolved`` instead).

    ``method`` is derived honestly from the evidence kinds actually
    present in ``evidence_kinds`` (never inferred from one kind's
    presence when another is absent) — valid values are
    ``"direct_import"``, ``"patch_target"``, or
    ``"direct_import+patch_target"``."""

    test_locus: str
    source_loci: Tuple[str, ...]
    method: str
    evidence_kinds: Tuple[str, ...]


def attribution_enabled() -> bool:
    return os.environ.get(
        "JARVIS_TEST_SOURCE_ATTRIBUTION_ENABLED", "true",
    ).strip().lower() not in ("0", "false", "no", "off")


def _max_source_files() -> int:
    try:
        val = int(os.environ.get("JARVIS_ATTRIBUTION_MAX_SOURCE_FILES", "8"))
        return max(1, val)
    except (TypeError, ValueError):
        return 8


def _module_map_ttl_s() -> float:
    try:
        return max(0.0, float(os.environ.get(
            "JARVIS_ATTRIBUTION_MODULE_MAP_TTL_S", "300",
        )))
    except (TypeError, ValueError):
        return 300.0


def _test_dir_names() -> frozenset:
    """Config-driven test-tree classification — reuses TestRunner's
    existing ``JARVIS_TEST_DIR_NAMES`` knob (mandate 1: no hardcoded
    directory assumptions; the default matches TestRunner's)."""
    raw = os.environ.get("JARVIS_TEST_DIR_NAMES", "tests").strip()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


# Bounded TTL cache for the module→path map (one rglob per repo per TTL,
# not per failing test). Keyed by repo_root; thread-safe.
_MAP_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_MAP_CACHE_LOCK = threading.Lock()


def _build_and_cache_module_map(repo_root: str) -> Dict[str, str]:
    """Single-flight builder for the module→path map.

    Holds ``_MAP_CACHE_LOCK`` across the ~7s ``build_module_to_path``
    rglob (double-checked inside the lock) so two concurrent expirers
    never both pay the crawl — the first to enter builds, the rest see
    the fresh entry and return it. When dispatched off-loop (via
    :func:`prewarm_module_map`'s ``cooperative_fs_io.offload``), the
    executor threads may briefly serialize on this lock — that is
    off-loop and correct; the event loop is never the thread waiting."""
    now = time.monotonic()
    with _MAP_CACHE_LOCK:
        hit = _MAP_CACHE.get(repo_root)
        if hit is not None and now - hit[0] < _module_map_ttl_s():
            return hit[1]
        mapping = build_module_to_path(repo_root)
        _MAP_CACHE[repo_root] = (time.monotonic(), mapping)
        return mapping


def _get_module_map(repo_root: str) -> Dict[str, str]:
    return _build_and_cache_module_map(repo_root)


async def prewarm_module_map(repo_root: str) -> None:
    """Off-loop pre-warm of the module→path cache (C1).

    ``build_module_to_path`` does a synchronous repo-wide ``rglob("*.py")``
    over ~63k files (~7s measured) — running it inside the in-loop
    ``_get_module_map`` (as ``process_failures`` does) reintroduces the
    repo's closed "sync-FS-on-loop" class every red poll. This helper
    routes that build through ``cooperative_fs_io.offload`` (the repo's
    canonical off-loop substrate) so the SAME ``_MAP_CACHE`` the sync
    path reads is populated in an executor thread; the subsequent in-loop
    ``_get_module_map(repo_root)`` is then a dict cache-hit, not a crawl.

    Callers must invoke this (``await``) before ``process_failures`` runs,
    and only on red cycles (failures present + attribution enabled) so
    green cycles never pay the crawl.

    Fail-soft: the substrate is imported lazily and any offload fault
    (import error, ``OffloadError``, non-dict result) leaves the cache
    untouched — the inline sync path in ``_get_module_map`` still works
    (just on-loop, which is exactly what this pre-warm avoids). The
    freshness probe itself is lock-free to avoid blocking the event loop."""
    now = time.monotonic()
    # Lock-free probe (Slice 7 fast-follow): an in-flight executor build
    # holds _MAP_CACHE_LOCK for the full ~7s crawl, so probing under the
    # lock would block the event loop for exactly that long. CPython
    # dict reads are atomic; a stale read is benign — the offload lands
    # in the single-flight builder, which dedups inside the lock.
    hit = _MAP_CACHE.get(repo_root)
    if hit is not None and now - hit[0] < _module_map_ttl_s():
        return  # already warm — no crawl, on- or off-loop
    try:
        from backend.core.ouroboros.governance import cooperative_fs_io
    except Exception:  # noqa: BLE001 — substrate optional; fall back to inline
        return
    try:
        result = await cooperative_fs_io.offload(
            _build_and_cache_module_map, repo_root
        )
    except Exception:  # noqa: BLE001 — offload must never break the poll
        return
    if cooperative_fs_io.is_offload_error(result) or not isinstance(result, dict):
        # Build faulted inside the worker — cache left as-is; the inline
        # sync path remains correct (this pre-warm is best-effort).
        return
    # ``_build_and_cache_module_map`` already stored the result in
    # ``_MAP_CACHE`` inside the executor thread — nothing further to do.


def _resolve_dotted_to_path(
    dotted: str, module_map: Dict[str, str],
) -> Optional[str]:
    """Longest-prefix resolution: ``x.y`` tries the submodule ``x.y``
    first, then the module ``x`` (``y`` was a symbol) — the exact-match-
    first discipline ``test_runner._find_tests_by_ast_import`` documents
    to avoid parent-package over-matching."""
    parts = dotted.split(".")
    while parts:
        hit = module_map.get(".".join(parts))
        if hit:
            return hit
        parts.pop()
    return None


_MOCK_PATCH_RECEIVERS = frozenset({"mock", "unittest.mock"})


def _dotted_receiver(node: ast.expr) -> str:
    """Render a callee *receiver* chain of ``Name``/``Attribute`` nodes to
    its dotted string (e.g. ``mock.patch`` -> receiver node ``mock`` ->
    ``"mock"``; ``unittest.mock.patch`` -> receiver ``unittest.mock`` ->
    ``"unittest.mock"``). Returns ``""`` for any other node shape (calls,
    subscripts, etc. are not a resolvable identity and are ignored)."""
    parts: list = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _extract_patch_targets(tree: ast.Module) -> Set[str]:
    """Dotted-string first arguments of ``mock.patch("x.y.z")`` /
    ``monkeypatch.setattr("x.y.z", ...)`` calls — deterministic AST
    literal extraction (string constants only; f-strings/variables are
    not resolvable and are correctly ignored). Receiver-identity checked
    (mandate 4 tightening): a REST client's ``client.patch(path)`` or a
    bare builtin ``setattr(obj, "x.y", val)`` must NOT match — only the
    canonical ``unittest.mock`` / ``monkeypatch`` call shapes do."""
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "patch":
            # Bare `patch(...)` — only valid via `from unittest.mock import patch`.
            pass
        elif isinstance(fn, ast.Attribute) and fn.attr == "patch":
            receiver = _dotted_receiver(fn.value)
            if receiver not in _MOCK_PATCH_RECEIVERS:
                continue
        elif (
            isinstance(fn, ast.Attribute)
            and fn.attr in ("setattr", "delattr")
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "monkeypatch"
        ):
            pass
        else:
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            val = arg0.value.strip()
            if "." in val:
                targets.add(val)
    return targets


def _is_test_infra(rel_path: str, dir_names: frozenset) -> bool:
    """True when *rel_path* lives in the configured test tree — it is a
    test-locus (the test itself, a helper, a conftest), never a
    source-locus. Config-driven via JARVIS_TEST_DIR_NAMES."""
    module = _module_from_relpath(rel_path)
    if not module:
        return True
    parts = module.split(".")
    if parts[0] in dir_names:
        return True
    return _is_test_module(module, dir_names)


def attribute_test_to_sources(
    test_file: str,
    *,
    repo_root: str,
    traceback_frames: Sequence[str] = (),
) -> Attribution:
    """Resolve the source file(s) *test_file* exercises. Deterministic:
    identical inputs yield identical output. Raises
    :class:`AttributionUnresolved` (typed reason) when no first-party
    source module is deterministically reachable — the caller must then
    fail-fast, never silently fall back to test-file mutation scope."""
    rel_test = _relpath_under_root(test_file, repo_root)
    if not rel_test:
        raise AttributionUnresolved("test_outside_root", test_file)
    abs_test = os.path.join(repo_root, rel_test)
    try:
        source = Path(abs_test).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AttributionUnresolved("test_file_missing", rel_test) from exc
    try:
        tree = ast.parse(source, filename=abs_test)
    except (SyntaxError, ValueError) as exc:
        raise AttributionUnresolved("parse_error", f"{rel_test}: {exc}") from exc

    module = _module_from_relpath(rel_test)
    is_init = rel_test == "__init__.py" or rel_test.endswith("/__init__.py")
    dir_names = _test_dir_names()
    module_map = _get_module_map(repo_root)

    # candidates: rel_path -> evidence kind (direct import wins over patch)
    candidates: Dict[str, str] = {}
    for dotted in sorted(extract_module_imports(tree, module, is_init)):
        rel = _resolve_dotted_to_path(dotted, module_map)
        if not rel or rel == rel_test or _is_test_infra(rel, dir_names):
            continue
        candidates.setdefault(rel, _KIND_DIRECT)
    for dotted in sorted(_extract_patch_targets(tree)):
        rel = _resolve_dotted_to_path(dotted, module_map)
        if not rel or rel == rel_test or _is_test_infra(rel, dir_names):
            continue
        candidates.setdefault(rel, _KIND_PATCH)

    if not candidates:
        raise AttributionUnresolved("no_first_party_source_imports", rel_test)

    tb_hits = {
        _relpath_under_root(f, repo_root) or f.replace("\\", "/")
        for f in traceback_frames
    }
    ranked = sorted(
        candidates.items(),
        key=lambda kv: (
            kv[0] not in tb_hits,          # traceback-implicated first
            kv[1] != _KIND_DIRECT,          # direct imports before patch targets
            kv[0],                          # lexical — total deterministic order
        ),
    )[: _max_source_files()]

    kinds = tuple(kind for _, kind in ranked)
    present_kinds = set(kinds)
    method = "+".join(
        kind for kind in (_KIND_DIRECT, _KIND_PATCH) if kind in present_kinds
    )
    return Attribution(
        test_locus=rel_test,
        source_loci=tuple(path for path, _ in ranked),
        method=method,
        evidence_kinds=kinds,
    )


# ---------------------------------------------------------------------------
# Strict source ISOLATION (anti-noise) — used by force-promoted signals that
# carry no fresh failure evidence (the pytest ``lastfailed`` cache-first path).
# ---------------------------------------------------------------------------
#
# THE NOISE CLASS (soak bt-2026-09-06-212249): the cache-first hydration
# force-promotes a persisted ``lastfailed`` node-id to a "stable" signal
# WITHOUT re-running the test, so it carries no traceback. With no traceback
# ``attribute_test_to_sources`` cannot narrow, and every first-party module the
# test imports becomes a target (one op scoped 6 files). The generator then
# correctly returns ``2b.1-noop`` for each — 30 no-ops, 0 commits in one soak.
# The root cause is that an un-reproduced cache id has no evidence of WHICH
# source is at fault; enqueuing a spray of imports is not actionable work.
#
# This predicate keeps a signal ONLY when the failing source is deterministically
# ISOLABLE, and returns ``None`` (the caller DISCARDS) otherwise — so the queue
# is fed real, narrowly-scoped work instead of import sprays. It is a pure
# refinement over ``attribute_test_to_sources`` (mandate 3: no new parser, no
# duplicated import tracing) and it NEVER raises.


def strict_isolation_enabled() -> bool:
    """Master for strict source isolation (default ON). OFF restores the
    pre-existing spray-all-imports behaviour byte-identically."""
    return os.environ.get(
        "JARVIS_ATTRIBUTION_STRICT_ISOLATION_ENABLED", "true",
    ).strip().lower() not in ("0", "false", "no", "off")


def _strict_max_loci() -> int:
    """Max first-party source files a NO-TRACEBACK signal may resolve to and
    still count as 'isolated'. Default 1 — an un-reproduced failure that maps
    to more than one module cannot be pinned, so it is discarded rather than
    sprayed. Env-tunable; clamped to >=1."""
    try:
        return max(1, int(os.environ.get(
            "JARVIS_ATTRIBUTION_STRICT_MAX_LOCI", "1",
        )))
    except (TypeError, ValueError):
        return 1


def _narrowed(attr: "Attribution", loci: Tuple[str, ...]) -> "Attribution":
    """Rebuild an ``Attribution`` restricted to *loci* (a subset of
    ``attr.source_loci``), carrying each locus's original evidence kind so
    ``method``/``evidence_kinds`` stay honest. Order follows *loci*."""
    kind_of = dict(zip(attr.source_loci, attr.evidence_kinds))
    kinds = tuple(kind_of.get(p, _KIND_DIRECT) for p in loci)
    present = set(kinds)
    method = "+".join(
        k for k in (_KIND_DIRECT, _KIND_PATCH) if k in present
    )
    return Attribution(
        test_locus=attr.test_locus,
        source_loci=loci,
        method=method,
        evidence_kinds=kinds,
    )


def attribute_strict_or_none(
    test_file: str,
    *,
    repo_root: str,
    traceback_frames: Sequence[str] = (),
) -> Optional[Attribution]:
    """Precision-first attribution for force-promoted / low-evidence signals.

    Returns a (possibly narrowed) :class:`Attribution` ONLY when the failing
    source is deterministically isolable; returns ``None`` when it is not — the
    caller then DISCARDS the signal instead of enqueuing an import spray.
    NEVER raises.

    Decision tree (the middle branch preserves the Run-16 assertion class):

      * ``attribute_test_to_sources`` raises ``AttributionUnresolved`` (no
        first-party source reachable) -> ``None``.
      * **traceback intersects >=1 source locus** -> narrow to the intersection.
        An exception-style failure's traceback IS the evidence of the faulting
        module; keep exactly those.
      * **traceback present but hits no source locus** (assertion failure whose
        deepest in-repo frame is the test line — the Run-16 class) -> keep the
        full import-based attribution. This is a real, freshly-reproduced
        failure and imports are the only signal; it is NOT discarded.
      * **no traceback at all** (cache-first force-promotion) -> keep ONLY when
        the import set already isolates to ``<= _strict_max_loci()`` files;
        otherwise ``None`` (discard the spray).

    Master ``JARVIS_ATTRIBUTION_STRICT_ISOLATION_ENABLED`` (default on); OFF
    behaves exactly like ``attribute_test_to_sources`` wrapped to return
    ``None`` on ``AttributionUnresolved`` (no discard-on-breadth)."""
    try:
        attr = attribute_test_to_sources(
            test_file, repo_root=repo_root, traceback_frames=traceback_frames,
        )
    except AttributionUnresolved:
        return None
    except Exception:  # noqa: BLE001 — isolation must never break a poll
        return None

    if not strict_isolation_enabled():
        return attr

    try:
        tb_hits = {
            _relpath_under_root(f, repo_root) or str(f).replace("\\", "/")
            for f in (traceback_frames or ())
        }
        if tb_hits:
            inter = tuple(p for p in attr.source_loci if p in tb_hits)
            if inter:
                return _narrowed(attr, inter)  # narrowed to the faulting frame
            return attr  # Run-16: assertion at the test line — keep imports
        # No traceback: keep only a genuinely isolated set.
        if len(attr.source_loci) <= _strict_max_loci():
            return attr
        return None
    except Exception:  # noqa: BLE001 — on any error, do not fabricate isolation
        return None


# ---------------------------------------------------------------------------
# Scope-gate predicate (Task 5 wires this at the orchestrator)
# ---------------------------------------------------------------------------


def scope_gate_enabled() -> bool:
    return os.environ.get(
        "JARVIS_ATTRIBUTION_SCOPE_GATE_ENABLED", "true",
    ).strip().lower() not in ("0", "false", "no", "off")


def _attribution_dict(intake_evidence_json: str) -> Dict[str, Any]:
    """Fail-soft parse of the Slice-6 evidence block: returns the
    ``attribution`` dict from an op's intake evidence JSON, or ``{}`` on
    absent / non-JSON / non-dict shapes. Never raises."""
    try:
        evidence = json.loads(intake_evidence_json or "{}")
        attribution = evidence.get("attribution") or {}
    except (ValueError, TypeError, AttributeError):
        return {}
    return attribution if isinstance(attribution, dict) else {}


def attribution_status(intake_evidence_json: str) -> str:
    """``attribution.status`` from an op's intake evidence JSON, ``""``
    when absent or malformed. The single evidence parser shared by the
    scope gate (Slice 6) and the coverage-gate subset waiver (Slice 7)."""
    return str(_attribution_dict(intake_evidence_json).get("status", ""))


def unattributed_test_scope_violation(
    intake_evidence_json: str,
    candidate_files: Sequence[str],
    *,
    repo_root: str = "",
) -> Optional[str]:
    """Mandate 4's enforcement predicate: when the op's attribution is
    ``unresolved`` and EVERY candidate file is a test-locus, mutating is
    exactly the Run-16 blind class — return a violation message (the
    orchestrator escalates to APPROVAL_REQUIRED). ``None`` = no
    violation. Strictly fail-soft on malformed evidence (absent /
    non-JSON / missing keys → None): this gate must never break ops that
    predate the schema.

    ``repo_root`` (I2): model candidates may carry ABSOLUTE paths
    (``/Users/x/repo/tests/conftest.py``). Without normalization the
    module derived from such a path becomes ``Users.x…`` — NOT
    test-classified — so ``_is_test_infra`` silently fails and the blind-
    mutation gate never fires. When ``repo_root`` is provided each
    candidate is normalized to a repo-relative POSIX path via
    ``_relpath_under_root`` (the same relativizer the attributor uses),
    falling back to the plain slash/``./`` normalization when the file is
    outside the root (relativizer returns "")."""
    if not scope_gate_enabled() or not candidate_files:
        return None
    attribution = _attribution_dict(intake_evidence_json)
    status = str(attribution.get("status", ""))
    if status != "unresolved":
        return None
    dir_names = _test_dir_names()
    test_locus = str(attribution.get("test_locus", ""))
    normalized = _normalize_candidate_paths(candidate_files, repo_root)
    if all(
        f == test_locus or _is_test_infra(f, dir_names) for f in normalized
    ):
        return (
            "attribution_unresolved_test_scope: op attribution is "
            f"unresolved ({attribution.get('reason', 'unknown')}) and the "
            f"candidate mutates only test loci {normalized} — blind "
            "test-file mutation is forbidden; requires human approval "
            "or source-locus exploration"
        )
    return None


# ---------------------------------------------------------------------------
# Slice 8 — NOTIFY_APPLY floor for RESOLVED-attribution test-only candidates
# ---------------------------------------------------------------------------
#
# Slice 7's subset waiver correctly lets a test-only candidate pass the
# coverage gate when attribution is RESOLVED (the test may genuinely BE
# the fix target). But that lane is sensitive: an assertion-weakening
# test edit auto-applies at SAFE_AUTO and VERIFY passes by construction
# (the test now agrees with the broken code it was supposed to catch).
# This predicate floors that lane at NOTIFY_APPLY — operator-visible
# diff + delay, never blocking (the lane is legitimate) and never
# downgrading a stricter tier.


def _normalize_candidate_paths(
    candidate_files: Sequence[str], repo_root: str,
) -> list:
    """Repo-relative POSIX normalization for candidate paths — shared by
    the unresolved scope gate (Slice 6) and the test-only NOTIFY floor
    (Slice 8). Absolute paths under *repo_root* relativize via
    ``_relpath_under_root``; everything else gets slash/``./`` cleanup."""
    normalized = []
    for f in candidate_files:
        _norm = str(f).replace("\\", "/")
        if repo_root:
            _rel = _relpath_under_root(str(f), repo_root)
            if _rel:
                normalized.append(_rel)
                continue
        if _norm.startswith("./"):
            _norm = _norm[2:]
        normalized.append(_norm)
    return normalized


def test_only_notify_floor_enabled() -> bool:
    return os.environ.get(
        "JARVIS_ATTRIBUTION_TEST_ONLY_NOTIFY_ENABLED", "true",
    ).strip().lower() not in ("0", "false", "no", "off")


def resolved_test_only_scope(
    intake_evidence_json: str,
    candidate_files: Sequence[str],
    *,
    repo_root: str = "",
) -> Optional[str]:
    """Slice 8: attribution RESOLVED + candidate mutates ONLY test loci.

    That lane is legitimate (the test may genuinely be the fix target —
    the whole point of the Slice-7 subset waiver) but sensitive: an
    assertion-weakening test edit auto-applies green and VERIFY passes by
    construction. Returns an advisory message — the caller floors risk at
    NOTIFY_APPLY (operator-visible diff, stricter-wins, never blocks,
    never downgrades) — or ``None``. Fail-soft on malformed evidence."""
    if not test_only_notify_floor_enabled() or not candidate_files:
        return None
    attribution = _attribution_dict(intake_evidence_json)
    if str(attribution.get("status", "")) != "resolved":
        return None
    dir_names = _test_dir_names()
    test_locus = str(attribution.get("test_locus", ""))
    normalized = _normalize_candidate_paths(candidate_files, repo_root)
    if all(
        f == test_locus or _is_test_infra(f, dir_names) for f in normalized
    ):
        return (
            "attribution_resolved_test_only_scope: attribution resolved "
            f"but the candidate mutates only test loci {normalized} — "
            "floored to NOTIFY_APPLY for operator visibility"
        )
    return None
