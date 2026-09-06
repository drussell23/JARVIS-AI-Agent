"""Strict source-isolation attribution (Phase 2 anti-noise).

``attribute_strict_or_none`` keeps a signal ONLY when the failing source is
deterministically isolable, and returns None (caller discards) otherwise —
killing the stale-``lastfailed`` import-spray class (soak bt-2026-09-06) while
leaving freshly-reproduced failures (incl. the Run-16 assertion class) intact.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.ouroboros.governance.intent import test_source_attribution as A


def _make_repo(root: Path, *, n_source_imports: int) -> str:
    """A repo with ``pkg/mod_a..`` source modules and a test importing the
    first ``n_source_imports`` of them. Returns the abs test-file path."""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "__init__.py").write_text("")
    names = [f"mod_{c}" for c in "abcde"][:max(1, n_source_imports)]
    for name in names:
        (root / "pkg" / f"{name}.py").write_text(f"def f_{name}():\n    return 1\n")
    imports = "\n".join(f"from pkg.{n} import f_{n}" for n in names)
    test = root / "tests" / "test_x.py"
    test.write_text(imports + "\n\ndef test_x():\n    assert f_mod_a() == 1\n")
    # unique repo_root per call → no module-map cache collision across tests
    return str(test)


def _clear_cache():
    with A._MAP_CACHE_LOCK:
        A._MAP_CACHE.clear()


def test_single_source_no_traceback_is_isolable(tmp_path):
    _clear_cache()
    test_file = _make_repo(tmp_path, n_source_imports=1)
    attr = A.attribute_strict_or_none(test_file, repo_root=str(tmp_path))
    assert attr is not None
    assert attr.source_loci == ("pkg/mod_a.py",)


def test_multi_source_no_traceback_is_discarded(tmp_path):
    _clear_cache()
    test_file = _make_repo(tmp_path, n_source_imports=3)
    # no traceback + 3 candidate modules → cannot isolate → discard
    attr = A.attribute_strict_or_none(test_file, repo_root=str(tmp_path))
    assert attr is None


def test_traceback_intersection_narrows(tmp_path):
    _clear_cache()
    test_file = _make_repo(tmp_path, n_source_imports=3)
    tb = [str(tmp_path / "pkg" / "mod_b.py")]  # the exception came from mod_b
    attr = A.attribute_strict_or_none(
        test_file, repo_root=str(tmp_path), traceback_frames=tb,
    )
    assert attr is not None
    assert attr.source_loci == ("pkg/mod_b.py",)  # narrowed to the faulting frame


def test_run16_assertion_traceback_at_test_line_keeps_imports(tmp_path):
    _clear_cache()
    test_file = _make_repo(tmp_path, n_source_imports=3)
    # assertion failure: deepest in-repo frame is the TEST file, not a source
    tb = [str(tmp_path / "tests" / "test_x.py")]
    attr = A.attribute_strict_or_none(
        test_file, repo_root=str(tmp_path), traceback_frames=tb,
    )
    assert attr is not None
    # NOT discarded and NOT narrowed away — imports remain the signal
    assert set(attr.source_loci) == {"pkg/mod_a.py", "pkg/mod_b.py", "pkg/mod_c.py"}


def test_no_first_party_source_is_none(tmp_path):
    _clear_cache()
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "__init__.py").write_text("")
    test = tmp_path / "tests" / "test_y.py"
    test.write_text("import os\nimport json\n\ndef test_y():\n    assert True\n")
    attr = A.attribute_strict_or_none(str(test), repo_root=str(tmp_path))
    assert attr is None


def test_disabled_keeps_spray(tmp_path, monkeypatch):
    _clear_cache()
    monkeypatch.setenv("JARVIS_ATTRIBUTION_STRICT_ISOLATION_ENABLED", "false")
    test_file = _make_repo(tmp_path, n_source_imports=3)
    attr = A.attribute_strict_or_none(test_file, repo_root=str(tmp_path))
    assert attr is not None  # OFF → no discard-on-breadth
    assert len(attr.source_loci) == 3


def test_max_loci_env_tunable(tmp_path, monkeypatch):
    _clear_cache()
    monkeypatch.setenv("JARVIS_ATTRIBUTION_STRICT_MAX_LOCI", "3")
    test_file = _make_repo(tmp_path, n_source_imports=3)
    attr = A.attribute_strict_or_none(test_file, repo_root=str(tmp_path))
    assert attr is not None  # 3 <= max_loci(3) → kept
    assert len(attr.source_loci) == 3
    _clear_cache()
    test_file4 = _make_repo(tmp_path / "r4", n_source_imports=4)
    attr4 = A.attribute_strict_or_none(test_file4, repo_root=str(tmp_path / "r4"))
    assert attr4 is None  # 4 > 3 → discard


def test_never_raises_on_garbage(tmp_path):
    _clear_cache()
    attr = A.attribute_strict_or_none(
        "/nonexistent/does_not_exist.py", repo_root=str(tmp_path),
    )
    assert attr is None
