"""An interrupted append must cost ONE row, not two.

The trajectory corpus is append-only JSONL and its rows reach 24 KB —
far past any size at which a single write is atomic. A process killed
mid-append therefore leaves a line whose payload landed and whose
terminator did not.

The next append then CONCATENATES its row onto that unterminated tail.
Demonstrated before the fix: three rows written by callers, two lines on
disk, one of them parseable. The interruption destroyed the torn row AND
the good row that followed it, and nothing said so.

The reader (`grpo_pipeline.iter_trajectory_rows`) already counts
undecodable lines and treats a single torn line as benign, because a
harvest running against a live soak can catch the row being appended.
That tolerance is exactly what the concatenation defeats: the damage
spreads to a row that was written correctly, after the writer had
recovered.

So the append terminates a partial tail before writing, under the flock
it already holds. One interruption, one lost row, and the rows around it
intact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.core.ouroboros.governance.cross_process_jsonl import (  # noqa: E402
    _terminate_partial_tail,
    flock_append_line,
    flock_append_lines,
)

ROW = json.dumps({"event_type": "interaction", "user_input": "A",
                  "assistant_output": "x = 1\n"})


def _tear(path: Path, text: str = ROW, keep: int = 40) -> None:
    """Exactly what a killed process leaves: payload, no terminator."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text[:keep])


def _rows(path: Path):
    ok, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ok.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return ok, bad


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_a_torn_write_does_not_take_the_next_row_with_it(tmp_path) -> None:
    p = tmp_path / "events.jsonl"
    flock_append_line(p, ROW)
    _tear(p)
    flock_append_line(p, ROW)

    ok, bad = _rows(p)
    assert len(ok) == 2, "both correctly-written rows must survive"
    assert bad == 1, "exactly the torn write is lost"


def test_the_batch_append_heals_the_tail_too(tmp_path) -> None:
    """`flock_append_lines` is the funnel both entry points reach."""
    p = tmp_path / "events.jsonl"
    flock_append_line(p, ROW)
    _tear(p)
    assert flock_append_lines(p, [ROW, ROW]) is True

    ok, bad = _rows(p)
    assert len(ok) == 3 and bad == 1


def test_repeated_interruptions_each_cost_only_themselves(tmp_path) -> None:
    p = tmp_path / "events.jsonl"
    for _ in range(5):
        flock_append_line(p, ROW)
        _tear(p)
    flock_append_line(p, ROW)

    ok, bad = _rows(p)
    assert len(ok) == 6 and bad == 5


def test_a_realistic_24kb_row_is_the_case_that_cannot_be_atomic(tmp_path) -> None:
    big = json.dumps({"event_type": "interaction", "user_input": "P" * 24_000,
                      "assistant_output": "x = 1\n"})
    p = tmp_path / "events.jsonl"
    flock_append_line(p, big)
    _tear(p, big, keep=12_000)
    flock_append_line(p, big)

    ok, bad = _rows(p)
    assert len(ok) == 2 and bad == 1
    assert all(len(r["user_input"]) == 24_000 for r in ok)


# ---------------------------------------------------------------------------
# It must not disturb the healthy path
# ---------------------------------------------------------------------------


def test_a_fresh_file_gets_no_leading_blank_line(tmp_path) -> None:
    p = tmp_path / "fresh.jsonl"
    flock_append_line(p, ROW)
    assert p.read_text(encoding="utf-8").splitlines()[0] != ""
    assert len(_rows(p)[0]) == 1


def test_an_intact_file_is_byte_identical_to_before(tmp_path) -> None:
    """The healing branch must be invisible when nothing is torn."""
    p = tmp_path / "a.jsonl"
    for _ in range(4):
        flock_append_line(p, ROW)
    assert p.read_text(encoding="utf-8") == (ROW + "\n") * 4


def test_every_line_still_ends_with_exactly_one_newline(tmp_path) -> None:
    p = tmp_path / "a.jsonl"
    flock_append_lines(p, [ROW, ROW, ROW])
    raw = p.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "\n\n" not in raw


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_the_probe_reports_whether_it_wrote(tmp_path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(ROW, encoding="utf-8")           # no terminator
    with p.open("a", encoding="utf-8") as fh:
        assert _terminate_partial_tail(fh, p) is True
    with p.open("a", encoding="utf-8") as fh:
        assert _terminate_partial_tail(fh, p) is False, "already terminated"


def test_an_empty_file_needs_no_terminator(tmp_path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with p.open("a", encoding="utf-8") as fh:
        assert _terminate_partial_tail(fh, p) is False


def test_an_unreadable_tail_fails_OPEN(tmp_path, monkeypatch) -> None:
    """Refusing to append because the tail could not be inspected would
    turn one recoverable torn line into total loss for everything after."""
    p = tmp_path / "a.jsonl"
    flock_append_line(p, ROW)

    real_open = Path.open

    def _boom(self, *a, **k):
        if a and "b" in str(a[0]):
            raise OSError("probe denied")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", _boom)
    assert flock_append_line(p, ROW) is True
    monkeypatch.undo()
    assert len(_rows(p)[0]) == 2


def test_the_probe_never_raises(tmp_path) -> None:
    missing = tmp_path / "gone.jsonl"
    with (tmp_path / "other.jsonl").open("a", encoding="utf-8") as fh:
        assert _terminate_partial_tail(fh, missing) is False


# ---------------------------------------------------------------------------
# What the corpus reader makes of it
# ---------------------------------------------------------------------------


def test_the_reader_keeps_the_surrounding_rows(tmp_path) -> None:
    """End to end against reactor's own reader, which is what decides
    whether a row reaches training."""
    reactor = _REPO.parent / "reactor"
    if not (reactor / "scripts" / "grpo_preflight.py").is_file():
        pytest.skip("reactor repo not present beside jarvis")
    sys.path.insert(0, str(reactor / "scripts"))
    try:
        import grpo_preflight as pf
        gp = pf._load("grpo_pipeline")
    finally:
        sys.path.pop(0)

    p = tmp_path / "events.jsonl"
    row = dict(json.loads(ROW))
    row["metadata"] = {"op_id": "op-1", "candidate_hash": "h1",
                       "draw_kind": "primary", "should_train": True}
    flock_append_line(p, json.dumps(row))
    _tear(p)
    row["metadata"] = dict(row["metadata"], candidate_hash="h2",
                           draw_kind="sibling")
    flock_append_line(p, json.dumps(row))

    got = list(gp.iter_trajectory_rows(tmp_path, trainable_only=True))
    assert len(got) == 2, "the torn write must not cost the row after it"
