"""Trajectory recorder — O+V generations as canonical experience events.

Phase 1 of the Reactor-Core data flywheel. Every local-lane generation
already produces the two halves of a preference pair:

  * the PROMPT + the CANDIDATE the model actually emitted, and
  * the VERDICT the pipeline reached on it (applied / syntax-error /
    caged by governance).

They were never written down together, so the corpus that would let
Reactor-Core improve the 32B did not exist. This module joins them and
appends ONE canonical ``ExperienceEvent`` line per candidate.

## Why this schema

The line format is Reactor-Core's canonical ``ExperienceEvent``
(``reactor_core/schemas/experience_schema.py``) — *and* those same field
names are the first-choice keys that
``reactor_core/training/dpo_pair_generator.py::_ingest_telemetry`` reads
(``user_input`` / ``assistant_output`` / ``model_id`` / ``outcome`` /
``confidence`` / ``latency_ms`` / ``timestamp`` / ``event_id``).

One stream, two consumers, zero translation code: the
``TrinityExperienceReceiver`` watches this directory already, and the DPO
generator reads it by pointing ``DPO_TELEMETRY_DIR`` at the same path.
A cross-repo *import* is impossible here (jarvis and reactor-core are
separate repos in separate venvs), so the shared contract is the schema.

``metadata.should_train`` carries the SAME exclusion policy as
``reactor_core/ingestion/autonomy_classifier.py``: a governance denial is
INFRASTRUCTURE, not a model-quality signal, and must never become a
"rejected" sample. Training on the cage would teach the model that
correctly-refused work is bad output.

## Design constraints (load-bearing)

  * **Never in the hot path.** Emission is ``Queue.put_nowait`` — on a
    full queue the event is DROPPED and counted, never awaited. No
    generation ever waits on disk I/O.
  * **Fail-open.** Every path is swallowed. Recording failure must never
    cost an op.
  * **Default-off** (§33.1): ``JARVIS_TRAJECTORY_RECORDER_ENABLED``.
    Off ⇒ both emit calls are a flag read and a return.
  * **Bounded** — queue depth, pending-join map, per-field char caps.
  * **Reuses the canonical substrates**: candidate projection via
    ``provider_response_cache._trajectory_from_generation_result``,
    prompt identity via its ``_prefix_key``, and the append via
    ``cross_process_jsonl.async_flock_append_line``. No second hasher,
    no second locking discipline, no second JSON writer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRAJECTORY_RECORDER_SCHEMA_VERSION = "1.0"

_TRUTHY = ("1", "true", "yes", "on")

_ENV_MASTER = "JARVIS_TRAJECTORY_RECORDER_ENABLED"
_ENV_DIR = "JARVIS_TRAJECTORY_RECORDER_DIR"
_ENV_QUEUE_MAX = "JARVIS_TRAJECTORY_RECORDER_QUEUE_MAX"
_ENV_PENDING_MAX = "JARVIS_TRAJECTORY_RECORDER_PENDING_MAX"
_ENV_PENDING_TTL_S = "JARVIS_TRAJECTORY_RECORDER_PENDING_TTL_S"
_ENV_TICK_S = "JARVIS_TRAJECTORY_RECORDER_TICK_S"
_ENV_MAX_PROMPT_CHARS = "JARVIS_TRAJECTORY_RECORDER_MAX_PROMPT_CHARS"
_ENV_MAX_OUTPUT_CHARS = "JARVIS_TRAJECTORY_RECORDER_MAX_OUTPUT_CHARS"

_DEFAULT_QUEUE_MAX = 512
_DEFAULT_PENDING_MAX = 256
_DEFAULT_PENDING_TTL_S = 900.0
_DEFAULT_TICK_S = 60.0
_DEFAULT_MAX_PROMPT_CHARS = 24_000
_DEFAULT_MAX_OUTPUT_CHARS = 24_000

_TRUNC_MARKER = "\n...[truncated by trajectory_recorder]"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def recorder_enabled() -> bool:
    """Master flag. Default FALSE per §33.1 (shadow-first)."""
    return _env_flag(_ENV_MASTER)


def _canonical_session_id() -> str:
    """Which soak a row belongs to, when the caller did not say.

    Every caller passes ``session_id=getattr(context, "session_id", "")``
    and ``OperationContext`` has NEVER defined that field, so the default
    won and every row on disk carried ``session_id: ""``. Partitioning a
    corpus by session then had to be done by clustering timestamps and
    guessing where one soak ended -- a heuristic that silently merges two
    runs whose gap is small, which is exactly what a reward comparison
    between consecutive soaks must never do.

    The process-wide identity is authoritative here BECAUSE a soak process
    runs exactly one session: the harness stamps
    ``JARVIS_OUROBOROS_SESSION_ID`` at boot and the worktree manager,
    auto-committer and workspace router already treat it as the session's
    name. Imported lazily so this observability module never sits on a
    governance import cycle, and fail-open: an unidentified session must
    not stop a row being written.
    """
    try:
        from backend.core.ouroboros.governance.autonomous_workspace import (  # noqa: PLC0415
            canonical_session_id,
        )
        return canonical_session_id()
    except Exception:  # noqa: BLE001 — a nameless row beats a lost row
        return ""


#: Schema the model itself uses to decline. The synthesised body below is
#: this exact envelope because reactor's `grpo_verifier.extract_sources`
#: already knows it: it returns `no_source_by_shape`, which `verify_static`
#: scores at the SYNTAX ceiling -- full credit for a well-formed answer,
#: zero credit for substance it never claimed. Storing the bare reason
#: PROSE instead would fail `json.loads`, fall through to `_grade_source`,
#: and be graded as broken Python: measured 0.250/tier-2 `syntax_error`
#: against 0.450/tier-1 for the envelope. That inversion would teach the
#: model that declining is no better than emitting garbage, which is the
#: precise failure the verifier's own reason-kind dispatch exists to stop.
_NOOP_SCHEMA_VERSION = "2b.1-noop"

#: Candidate id for a synthesised refusal row. Stable, so a consumer can
#: recognise one without parsing prose.
_NOOP_CANDIDATE_ID = "noop"

#: The structure class every refusal shares. Not a digest: a decline has
#: no AST to digest, and all declines are one answer-kind.
_NOOP_STRUCTURE_ID = "noop"


def noop_candidate(reason: str) -> Dict[str, Any]:
    """One candidate dict standing for a refusal. NEVER raises.

    The body is the decline ENVELOPE, not the reason text -- see
    `_NOOP_SCHEMA_VERSION`. The hash is derived from that body so the
    recorder's existing (op_id, candidate_hash) dedupe collapses a model
    that repeats itself while keeping two genuinely different refusals as
    two rows. No second hasher: `candidate_hash` is the same identity the
    retract seam and the per-candidate verdict join already key on.
    """
    body = json.dumps(
        {"schema_version": _NOOP_SCHEMA_VERSION, "reason": str(reason or "")},
        ensure_ascii=False, separators=(",", ":"),
    )
    digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:32]
    return {
        "candidate_id": _NOOP_CANDIDATE_ID,
        "candidate_hash": digest,
        "file_path": "",
        "full_content": body,
        # The deterministic tag downstream reads INSTEAD of sniffing the
        # body. A consumer must never have to string-match prose to learn
        # that a row is a refusal.
        "candidate_status": "noop",
    }


#: Candidate id for a synthesised unparseable row.
_PARSE_ERROR_CANDIDATE_ID = "parse_error"

#: The structure class every unparseable draw shares. A candidate that does
#: not parse has no AST to digest, and `_structure_stamps` would drop it —
#: leaving a {parse_error, patch} group reading as one answer.
_PARSE_ERROR_STRUCTURE_ID = "parse_error"


def parse_error_candidate(
    raw: str, failures: "Optional[List[Dict[str, Any]]]" = None,
) -> Dict[str, Any]:
    """One candidate dict standing for a draw that did not parse.

    The body is the model's RAW response, verbatim. That is the whole
    point: reactor's `extract_sources` reaches through the envelope to the
    candidate bodies and hands the invalid Python to `_grade_source`,
    which scores it by HOW FAR the parse got — measured 0.250 for a
    failure on line 1 of 3, 0.393 for line 6 of 7. Storing a summary, a
    truncated preview, or a synthetic marker instead would throw that
    gradient away and collapse every parse error onto one value.

    `candidate_preview` on the provider's exception is `raw[:800]`; using
    it here would fabricate a truncation the model never emitted and the
    grader would score the cut, not the code.

    The hash is the digest of the raw body, so the existing
    (op_id, candidate_hash) dedupe absorbs a model that fails identically
    twice while keeping two different failures as two rows. NEVER raises.
    """
    body = str(raw or "")
    digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:32]
    first = (failures or [{}])[0] if failures else {}
    return {
        "candidate_id": _PARSE_ERROR_CANDIDATE_ID,
        "candidate_hash": digest,
        "file_path": str((first or {}).get("file_path", "") or ""),
        "full_content": body,
        "candidate_status": "parse_error",
    }


def events_dir() -> Path:
    """Directory the Trinity experience receiver already watches."""
    raw = os.getenv(_ENV_DIR, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".jarvis" / "trinity" / "events"


# ---------------------------------------------------------------------------
# Outcome policy — mirrors reactor_core/ingestion/autonomy_classifier.py
# ---------------------------------------------------------------------------

# (canonical_outcome, autonomy_event_type, should_train)
_OutcomePolicy = Tuple[str, str, bool]

_SUCCESS: _OutcomePolicy = ("success", "committed", True)
_FAILURE: _OutcomePolicy = ("failure", "failed", True)
_NOOP: _OutcomePolicy = ("partial", "committed", True)
_CAGED: _OutcomePolicy = ("unknown", "policy_denied", False)
_INFRA: _OutcomePolicy = ("unknown", "no_journal_lease", False)
_UNKNOWN: _OutcomePolicy = ("unknown", "intent_written", False)

#: Op-level outcomes that a generation's OWN ``is_noop`` may override.
#:
#: A refusal is SELF-EVIDENCING. The model returned ``2b.1-noop`` — that IS
#: the answer, and unlike a patch it needs no external verdict to be known:
#: its diff is verifiably null BY CONSTRUCTION, because no candidate was
#: produced and the recorded body is the decline envelope itself. Running an
#: AST check over it would parse a JSON object that contains no code and
#: prove nothing the schema has not already stated.
#:
#: ``success`` was the original member: an op that ended well but whose
#: generation declined is ``partial``, not ``applied``.
#:
#: ``unknown`` is added because it was silently costing the corpus its
#: refusals. ``classify_terminal_reason`` falls through to ``_UNKNOWN``
#: (should_train FALSE) whenever an op terminates without a reason the
#: policy names — which is the common case for a declining op, since it
#: never reaches a verdict-bearing phase. Measured on soak
#: bt-2026-09-04-213313: 70 ``noop``/``primary`` rows were written and only
#: 24 survived to the trainer; the rest carried should_train=False from this
#: fall-through. The ``_UNKNOWN`` policy exists so an UNSEEN outcome is
#: never guessed — but a refusal's outcome is not unseen, it is declared by
#: the generation itself, so deferring to the op-level silence discards a
#: fact the row already carries.
#:
#: Deliberately NOT overridable: ``failure`` (the candidate was judged bad
#: on its own merits and that judgement outranks the declaration) and the
#: governance-cage outcomes, whose whole point is that model quality was
#: never the reason the op died.
#:
#: Matched on the POLICY TUPLE, never on the outcome string. ``_CAGED``,
#: ``_INFRA`` and ``_UNKNOWN`` ALL carry outcome ``"unknown"`` and are told
#: apart only by their ``autonomy_event_type``, so an outcome-string test
#: silently makes a governance-denied refusal trainable — the one thing this
#: must not do. The first version of this constant did exactly that and its
#: own test caught it.
_NOOP_OVERRIDABLE_POLICIES: frozenset = frozenset({_SUCCESS, _UNKNOWN})

#: Validation failure classes that mean "never judged", not "judged bad".
#: Mirrors reactor-core's autonomy_classifier policy -- infrastructure is
#: not quality -- and the op-level mapping that already sends
#: `validation_infra_failure` to a non-trainable `unknown`.
_UNASSESSED_FAILURE_CLASSES: frozenset = frozenset({"infra", "build", "security"})

# Terminal reason codes observed in the battle-test soaks. Governance
# denials and environmental faults are deliberately NOT trainable: the
# model's output was never the reason the op died.
_TERMINAL_REASON_POLICY: Dict[str, _OutcomePolicy] = {
    # --- model produced something the pipeline accepted ---
    "background_accepted": _SUCCESS,
    "applied": _SUCCESS,
    "completed": _SUCCESS,
    # --- a NO-OP verdict is an answer, not an absence ---
    "noop": _NOOP,
    "2b.1-noop": _NOOP,
    # --- the candidate itself was bad: the trainable failure ---
    "all_candidates_syntax_error": _FAILURE,
    "validation_failed": _FAILURE,
    "tests_failed": _FAILURE,
    # --- governance cage: correct refusals, never model-quality ---
    "self_modification_unsanctioned_source": _CAGED,
    "touches_kernel": _CAGED,
    "touches_supervisor": _CAGED,
    "touches_security": _CAGED,
    "target_out_of_scope": _CAGED,
    # --- environmental / lifecycle ---
    "l2_stopped": _INFRA,
    "session_exhausted": _INFRA,
    "unhandled_pipeline_exception": _INFRA,
    "wall_clock_cap": _INFRA,
}


def classify_terminal_reason(
    terminal_reason: str, terminal_phase: str = "",
) -> _OutcomePolicy:
    """Map an O+V terminal reason to ``(outcome, autonomy_type, train)``.

    Unknown reasons degrade to non-trainable UNKNOWN rather than being
    guessed into SUCCESS/FAILURE — a mislabelled pair is worse than a
    missing one.
    """
    reason = (terminal_reason or "").strip().lower()
    if reason in _TERMINAL_REASON_POLICY:
        return _TERMINAL_REASON_POLICY[reason]
    phase = (terminal_phase or "").strip().upper()
    if phase in ("COMPLETED", "APPLIED"):
        return _SUCCESS
    return _UNKNOWN


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNC_MARKER


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _structure_stamps(
    candidates: "Tuple[Dict[str, Any], ...]",
) -> "Tuple[Dict[str, str], int]":
    """``(candidate_hash -> structure_id, distinct_structure_count)``.

    The corpus must be able to say how many ANSWERS a generation holds,
    not how many rows it wrote. Measured before this existed: 8 sibling
    rows across 3 groups carried 3 structurally distinct answers, every
    group collapsing to one -- so ``n_candidates`` read 2 or 3 while the
    number that decides whether a preference pair is constructible was
    1. A consumer filtering on ``n_candidates >= 2`` was selecting groups
    that cannot train.

    ``structure_id`` is a short digest of the docstring-stripped AST, so
    two rows that differ only in prose share one id and a reader can
    group by it without re-parsing. Empty when the candidate does not
    parse -- unparseable answers are real and must not be silently
    folded together under one shared id.

    A REFUSAL is its own structure class, ``_NOOP_STRUCTURE_ID``. Its body
    is a decline envelope, not Python, so it fingerprints as unparseable
    and would otherwise contribute nothing -- making a {refusal, patch}
    group report ``n_distinct_structures=1`` and read as unpairable when
    it is in fact the widest-separated pair the corpus holds (0.3586 by
    the static grader). Two refusals still share the id, which is also
    right: declining twice is one answer given twice, so an all-refusal
    group correctly reports 1 and stays out of the pairable count.

    Describes; never decides. The recorder does not re-sample -- that
    belongs to the generation lane, which owns the budget and the
    provider seat.
    """
    ids: "Dict[str, str]" = {}
    distinct: set = set()
    try:
        from backend.core.ouroboros.governance import (  # noqa: PLC0415
            sibling_entropy as _ent,
        )
        for cand in candidates or ():
            if not isinstance(cand, dict):
                continue
            _status = str(cand.get("candidate_status", "") or "")
            if _status == "noop":
                ids[str(cand.get("candidate_hash", "") or "")] = (
                    _NOOP_STRUCTURE_ID
                )
                distinct.add(_NOOP_STRUCTURE_ID)
                continue
            if _status == "parse_error":
                # Unparseable answers are REAL and must not be folded
                # together with each other OR dropped: they are their own
                # answer class, distinct from a refusal and from any patch.
                ids[str(cand.get("candidate_hash", "") or "")] = (
                    _PARSE_ERROR_STRUCTURE_ID
                )
                distinct.add(_PARSE_ERROR_STRUCTURE_ID)
                continue
            fp = _ent.structural_fingerprint(_ent.candidate_source(cand))
            if fp is None:
                continue
            sid = hashlib.sha256(fp.encode("utf-8", "replace")).hexdigest()[:12]
            ids[str(cand.get("candidate_hash", "") or "")] = sid
            distinct.add(sid)
    except Exception:  # noqa: BLE001 — telemetry must never block a write
        logger.debug("[TrajectoryRecorder] structure stamp fault", exc_info=True)
    return ids, len(distinct)


# ---------------------------------------------------------------------------
# Queue payloads
# ---------------------------------------------------------------------------


@dataclass
class _PendingGeneration:
    """One generation awaiting its verdict."""

    op_id: str
    prompt: str
    prompt_key: str
    candidates: Tuple[Dict[str, Any], ...]
    model_id: str
    provider_name: str
    is_noop: bool
    latency_ms: float
    # Split, not summed. tok/s is completion_tokens over the generation
    # duration; a single `tokens_used` that folds the prompt in makes the
    # throughput term unrecoverable, and throughput is half of the
    # model A/B question (a smarter model that is 3x slower may lose).
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    task_type: str
    session_id: str
    # Whether the counts above were REPORTED by the engine or inferred from
    # response length. An A/B decided on tok/s must be able to exclude rows
    # whose tok/s was derived from a chars/4 guess -- a corpus that cannot
    # distinguish the two silently launders an estimate into a measurement.
    tokens_estimated: bool = True
    # candidate_hash -> (passed, failure_class), filled in as VALIDATE
    # judges each sibling. THE reason n>1 generation is worth doing: without
    # a per-candidate verdict every sibling of one op inherits that op's
    # single terminal outcome, scores identically in the DPO ranker, and
    # yields ZERO pairs (measured: 3 uniform siblings -> 0 pairs; the same
    # 3 with per-candidate outcomes -> 2 pairs, method=outcome_diff).
    candidate_verdicts: Dict[str, Tuple[bool, str]] = field(default_factory=dict)
    # candidate_hash -> the SPECIFIC failure digest (assertion / AST error) a
    # rejected sibling died on. Parallel to candidate_verdicts (additive — the
    # verdict tuple's arity is unchanged) so GRPO can rank on the CAUSE, not
    # only the failure_class category (Phase 2 test-gate telemetry).
    candidate_details: Dict[str, str] = field(default_factory=dict)
    # Where this generation sits in its op's lineage. A retry exists
    # BECAUSE the previous attempt was rejected, so attempt 0 and attempt 1
    # of one op are very often a genuine {rejected, chosen} pair on the same
    # prompt -- the corpus has to be able to see that ordering to use it.
    attempt_index: int = 0
    lineage_size: int = 1
    created_monotonic: float = field(default_factory=time.monotonic)
    created_iso: str = field(default_factory=_utc_now_iso)
    # WHAT KIND OF DRAW this was -- the discriminator soak 17 was missing.
    # `attempt_index = len(lineage)` made every provider call for an op the
    # next "attempt", so an L2 repair re-generation (same prompt, legacy
    # temperature, very often the accepted candidate's own structure) was
    # indistinguishable from an entropy-ladder sibling. Measured across 11
    # groups: attempt patterns like [0,1,2,1] / [0,1,1] / [0,1,2,2] where the
    # repeated index carried `l2_stopped` and an EARLIER attempt's
    # structure_id -- byte-identical "twins" that inflated groups with
    # untrainable rows (29 of 43) and dragged the reward spread.
    draw_kind: str = "unknown"
    temperature: Optional[float] = None
    sampling: Dict[str, Any] = field(default_factory=dict)


#: Draw kinds. ONE vocabulary, shared by the record seam, the persisted row,
#: the harvest and the reactor's ingestion filter.
DRAW_PRIMARY = "primary"    # an op's first draw at the legacy point
DRAW_SIBLING = "sibling"    # an entropy-ladder redraw (non-legacy sampling)
DRAW_REPAIR = "repair"      # an L2 repair iteration (repair_context present)
DRAW_RETRY = "retry"        # another legacy-point draw after a primary exists
DRAW_UNKNOWN = "unknown"    # rows written before the discriminator existed
#: The kinds a preference GROUP may contain. A repair or retry re-answers the
#: same prompt without exploring, so it can only ever add a twin.
GENUINE_DRAW_KINDS = frozenset({DRAW_PRIMARY, DRAW_SIBLING, DRAW_UNKNOWN})


def _sampling_overrides_of(sampling: Any) -> Dict[str, Any]:
    """The sampling point as a plain dict, or {}. Duck-typed on the SAME
    contract ``local_inference_director._sampling_overrides`` reads
    (``config_overrides()`` or a mapping) so the row carries exactly what
    the request carried. NEVER raises."""
    try:
        raw = sampling
        getter = getattr(sampling, "config_overrides", None)
        if callable(getter):
            raw = getter()
        if not isinstance(raw, dict):
            return {}
        return {str(k): v for k, v in raw.items() if v is not None}
    except Exception:  # noqa: BLE001
        return {}


def derive_draw_kind(
    *, is_repair: bool, sampling: Any, explicit: str = "",
) -> str:
    """Classify a draw from what the record seam knows.

    An explicit kind wins. Otherwise: a repair context makes it a repair
    regardless of sampling (L2 lowers temperature, and a lowered legacy
    point is still not exploration); a non-legacy sampling point makes it a
    sibling; anything else is a primary, which the lineage may later
    reclassify as a RETRY when a primary already exists for the op. Pure,
    NEVER raises."""
    try:
        if explicit:
            return str(explicit)
        if is_repair:
            return DRAW_REPAIR
        if sampling is not None and not bool(getattr(sampling, "is_legacy", False)):
            if _sampling_overrides_of(sampling) or getattr(sampling, "temperature", None) is not None:
                return DRAW_SIBLING
        return DRAW_PRIMARY
    except Exception:  # noqa: BLE001
        return DRAW_UNKNOWN


def is_genuine_draw(meta: Any) -> bool:
    """True when a persisted row may take part in a preference group.
    Reads ``metadata.draw_kind``; rows written before the field existed
    are kept (``unknown``) so the historical corpus is not silently
    emptied. NEVER raises."""
    try:
        kind = str((meta or {}).get("draw_kind") or DRAW_UNKNOWN)
    except Exception:  # noqa: BLE001
        return True
    return kind in GENUINE_DRAW_KINDS


@dataclass
class _OutcomeEvent:
    op_id: str
    terminal_phase: str
    terminal_reason: str


@dataclass
class _RetractEvent:
    """The generator DROPPED a draw it had already reported.

    ``record_generation`` fires per provider call, before the sibling loop
    has judged the draw, so a redundant sibling was reaching the corpus as
    a row with the same ``structure_id`` as its twin. This event travels
    the SAME queue as the generation it names, so ordering is a property
    of the queue, not of timing: it can only be processed after that
    generation was admitted and before the outcome that would write it.
    """

    op_id: str
    candidate_hashes: Tuple[str, ...]
    reason: str = ""


@dataclass
class _CandidateVerdictEvent:
    """One sibling's VALIDATE result, en route to its pending generation.

    Separate from :class:`_OutcomeEvent` because it arrives EARLIER and does
    not terminate anything: validation judges candidates while the op is
    still running, and only the op-level verdict triggers the write.
    """

    op_id: str
    candidate_hash: str
    passed: bool
    failure_class: str
    failure_detail: str = ""


class TrajectoryRecorder:
    """Async, bounded, fail-open recorder. One writer task per process."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path_override = path
        self._queue: Optional[asyncio.Queue] = None
        self._writer: Optional[asyncio.Task] = None
        # Wall-clock expiry watchdog. Separate from the drain loop because
        # expiry that only runs when a queue item arrives is not expiry at
        # all: on a sparse workload the pending generation waits for an
        # event that never comes, and its trajectory is never written. That
        # is exactly what produced 4 candidate sets and 0 recorded lines.
        self._watchdog: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None
        self._pending: "OrderedDict[str, List[_PendingGeneration]]" = OrderedDict()
        self._stats: Dict[str, int] = {
            "generations_queued": 0,
            "outcomes_queued": 0,
            "events_written": 0,
            "lineage_pruned": 0,
            "rows_deduped": 0,
            "dropped_queue_full": 0,
            "dropped_no_loop": 0,
            "pending_evicted": 0,
            "pending_expired": 0,
            "orphan_outcomes": 0,
            "write_failures": 0,
            "candidate_verdicts_queued": 0,
            "candidate_verdicts_joined": 0,
            "orphan_candidate_verdicts": 0,
        }

    # -- paths ------------------------------------------------------------
    @property
    def path(self) -> Path:
        if self._path_override is not None:
            return self._path_override
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return events_dir() / f"experience_{day}.jsonl"

    # -- lifecycle --------------------------------------------------------
    def _ensure_writer(self) -> bool:
        """Start the drain task if a loop is running. False => cannot record."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._stats["dropped_no_loop"] += 1
            return False
        if self._queue is None:
            self._queue = asyncio.Queue(
                maxsize=_env_int(
                    _ENV_QUEUE_MAX, _DEFAULT_QUEUE_MAX, 16, 65_536
                )
            )
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._writer is None or self._writer.done():
            self._writer = loop.create_task(
                self._drain_loop(), name="trajectory_recorder_drain"
            )
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = loop.create_task(
                self._expiry_watchdog(), name="trajectory_recorder_expiry"
            )
        return True

    async def _expiry_watchdog(self) -> None:
        """Flush overdue pendings on WALL-CLOCK time, not on queue traffic."""
        while True:
            tick = _env_float(_ENV_TICK_S, _DEFAULT_TICK_S, 1.0, 3_600.0)
            try:
                await asyncio.sleep(tick)
                await self._expire_pending()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- a watchdog must not die
                logger.debug(
                    "[TrajectoryRecorder] expiry tick failed", exc_info=True,
                )

    def _offer(self, item: Any) -> bool:
        if not recorder_enabled():
            return False
        try:
            if not self._ensure_writer():
                return False
            assert self._queue is not None
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            # Backpressure is a DROP, never a wait: the control loop must
            # not pay for a slow disk.
            self._stats["dropped_queue_full"] += 1
            return False
        except Exception:  # noqa: BLE001 — fail-open, always
            logger.debug("[TrajectoryRecorder] offer failed", exc_info=True)
            return False

    # -- public emit API --------------------------------------------------
    def record_generation(
        self,
        *,
        op_id: str,
        prompt: str,
        generation_result: Any,
        latency_ms: float = 0.0,
        task_type: str = "",
        session_id: str = "",
        route: str = "codegen",
        model_id_override: str = "",
        completion_tokens_override: int = -1,
        prompt_tokens_override: int = -1,
        tokens_estimated: bool = False,
        draw_kind: str = "",
        is_repair: bool = False,
        sampling: Any = None,
        temperature: Optional[float] = None,
    ) -> bool:
        """Queue one generation. Non-blocking. NEVER raises.

        ``is_repair`` / ``sampling`` / ``temperature`` are the draw's
        PROVENANCE: whether an L2 repair context was present, the sampling
        point the request carried, and its temperature. They resolve to a
        ``draw_kind`` (see ``derive_draw_kind``) that the row persists, so
        the harvest can group only genuine draws. ``draw_kind`` may be
        passed explicitly by a caller that knows better.

        The ``*_override`` arguments exist because a GenerationResult does
        not always carry the truth. On the local lane the result reports
        ``model_id="gpt-4"`` -- the OpenAI-compat client's default, not the
        model that ran -- and zero token counts. For a MODEL A/B the model
        identity is the entire experiment: three runs all labelled "gpt-4"
        are indistinguishable, and the corpus is worthless. Callers that
        know better pass what they know.
        """
        if not recorder_enabled() or not op_id:
            return False
        try:
            from backend.core.ouroboros.governance.provider_response_cache import (  # noqa: E501
                _prefix_key,
                _trajectory_from_generation_result,
            )
        except Exception:  # noqa: BLE001 — substrate optional
            return False

        model_id = str(getattr(generation_result, "model_id", "") or "")
        prompt_raw = str(prompt or "")
        prompt_text = _truncate(
            prompt_raw,
            _env_int(
                _ENV_MAX_PROMPT_CHARS,
                _DEFAULT_MAX_PROMPT_CHARS,
                256,
                2_000_000,
            ),
        )
        try:
            prompt_key = _prefix_key(prompt_raw, model_id, route)
        except Exception:  # noqa: BLE001
            prompt_key = ""

        # Candidate projection is the cache's already-hardened one — it
        # drops non-serializable tool objects and never raises.
        traj = _trajectory_from_generation_result(
            "", prompt_key, generation_result,
        )
        if traj is None:
            return False
        _cands = tuple(traj.candidates)
        if not _cands:
            # A REFUSAL IS AN ANSWER. The model was asked to change a file
            # and replied "already correct" -- a response to this exact
            # prompt, and the cleanest negative half of a preference pair
            # the corpus can hold: measured against a working patch the
            # static grader separates them by 0.3586, where the trainable
            # gate asks for 0.01.
            #
            # Dropping it here is what made 8 of soak 19's 15 singleton
            # ops singletons, and the drop was silent: this early return
            # fires BEFORE the "was NOT queued" log below, so a refused
            # draw left no row and no trace of having been refused.
            #
            # `_NOOP` in the outcome policy above has ALWAYS said
            # should_train=True ("a NO-OP verdict is an answer, not an
            # absence"). Only the row was missing.
            if not bool(getattr(traj, "is_noop", False)):
                return False
            _cands = (
                noop_candidate(str(getattr(traj, "noop_reason", "") or "")),
            )

        pending = _PendingGeneration(
            op_id=str(op_id),
            prompt=prompt_text,
            prompt_key=prompt_key,
            candidates=_cands,
            model_id=(str(model_id_override).strip() or traj.model_id),
            provider_name=traj.provider_name,
            is_noop=bool(traj.is_noop),
            latency_ms=max(0.0, float(latency_ms or 0.0)),
            prompt_tokens=(
                prompt_tokens_override if prompt_tokens_override >= 0
                else int(traj.total_input_tokens or 0)
            ),
            completion_tokens=(
                completion_tokens_override if completion_tokens_override >= 0
                else int(traj.total_output_tokens or 0)
            ),
            cost_usd=float(traj.original_cost_usd),
            task_type=str(task_type or ""),
            session_id=str(session_id or "") or _canonical_session_id(),
            tokens_estimated=bool(tokens_estimated),
            draw_kind=derive_draw_kind(
                is_repair=bool(is_repair), sampling=sampling,
                explicit=str(draw_kind or ""),
            ),
            temperature=(None if temperature is None else float(temperature)),
            sampling=_sampling_overrides_of(sampling),
        )
        if self._offer(pending):
            self._stats["generations_queued"] += 1
            # Speak on SUCCESS too. A recorder that only logs failures
            # cannot distinguish "the generation hook never fired" from
            # "it fired and the verdict never joined" -- and those have
            # opposite fixes. Both halves must be visible by op_id or the
            # join is undiagnosable from a log.
            logger.info(
                "[TrajectoryRecorder] queued generation op=%s "
                "candidates=%d model=%s (awaiting verdict)",
                pending.op_id, len(pending.candidates), pending.model_id,
            )
            return True
        logger.info(
            "[TrajectoryRecorder] generation for op=%s was NOT queued "
            "(enabled=%s candidates=%d) — no trajectory will be written",
            str(op_id), recorder_enabled(), len(traj.candidates),
        )
        return False

    def record_retraction(
        self,
        *,
        op_id: str,
        candidate_hashes: Any,
        reason: str = "",
    ) -> bool:
        """Retract a generation the generator has since rejected.

        Non-blocking, same queue as ``record_generation``. NEVER raises.
        The generator calls this in the one place it DROPS a draw -- the
        structural-redundancy branch of the sibling loop -- so a rejected
        twin stops reaching the corpus as a row that looks like half of a
        preference pair.
        """
        if not recorder_enabled() or not op_id:
            return False
        try:
            hashes = tuple(
                str(h or "") for h in (candidate_hashes or ()) if h
            )
        except TypeError:
            return False
        if not hashes:
            return False
        evt = _RetractEvent(
            op_id=str(op_id), candidate_hashes=hashes, reason=str(reason or ""),
        )
        if self._offer(evt):
            self._stats["retractions_queued"] = (
                self._stats.get("retractions_queued", 0) + 1
            )
            return True
        return False

    def record_candidate_verdict(
        self,
        *,
        op_id: str,
        candidate_hash: str,
        passed: bool,
        failure_class: str = "",
        failure_detail: str = "",
    ) -> bool:
        """Queue ONE sibling's VALIDATE verdict. Non-blocking. NEVER raises.

        Keyed by ``candidate_hash`` because that is what the written event
        already carries per candidate, so the join needs no new identity.
        A candidate with no hash cannot be joined and is dropped rather
        than guessed onto a sibling.
        """
        if not recorder_enabled() or not op_id or not candidate_hash:
            return False
        evt = _CandidateVerdictEvent(
            op_id=str(op_id),
            candidate_hash=str(candidate_hash),
            passed=bool(passed),
            failure_class=str(failure_class or ""),
            failure_detail=str(failure_detail or ""),
        )
        if self._offer(evt):
            self._stats["candidate_verdicts_queued"] += 1
            return True
        return False

    def record_outcome(
        self,
        *,
        op_id: str,
        terminal_phase: str = "",
        terminal_reason: str = "",
    ) -> bool:
        """Queue one op verdict. Non-blocking. NEVER raises."""
        if not recorder_enabled() or not op_id:
            return False
        evt = _OutcomeEvent(
            op_id=str(op_id),
            terminal_phase=str(terminal_phase or ""),
            terminal_reason=str(terminal_reason or ""),
        )
        if self._offer(evt):
            self._stats["outcomes_queued"] += 1
            return True
        return False

    # -- writer -----------------------------------------------------------
    async def _drain_loop(self) -> None:
        assert self._queue is not None
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                continue
            try:
                if isinstance(item, _PendingGeneration):
                    await self._admit_pending(item)
                elif isinstance(item, _RetractEvent):
                    await self._retract(item)
                elif isinstance(item, _CandidateVerdictEvent):
                    await self._attach_verdict(item)
                elif isinstance(item, _OutcomeEvent):
                    await self._resolve(item)
                # Opportunistic sweep on activity; the watchdog is what
                # guarantees expiry when there is none.
                await self._expire_pending()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[TrajectoryRecorder] drain step failed", exc_info=True,
                )
            finally:
                try:
                    self._queue.task_done()
                except Exception:  # noqa: BLE001
                    pass

    def _pending_count(self) -> int:
        """Total pending GENERATIONS, not ops. Caller holds the lock."""
        return sum(len(v) for v in self._pending.values())

    async def _admit_pending(self, gen: _PendingGeneration) -> None:
        """Append this generation to its op's lineage.

        APPEND, not assign. Keying by op_id alone made the map
        last-write-wins, so an op that generated more than once --
        GENERATE_RETRY, syntax repair, a sibling draw that itself retried
        -- kept only its final attempt and every earlier candidate set was
        discarded before it could be written. Measured on soak
        bt-2026-08-31-185439: 31 generations across 13 ops produced 14
        rows.

        Those discarded attempts are not noise, they are the most valuable
        rows in the corpus: a retry exists BECAUSE the first attempt was
        rejected, so an op's lineage is very often a genuine
        {rejected, chosen} pair on one prompt -- exactly what DPO needs and
        exactly what the flat key was throwing away.
        """
        cap = _env_int(_ENV_PENDING_MAX, _DEFAULT_PENDING_MAX, 8, 100_000)
        async with self._guard():
            lineage = self._pending.get(gen.op_id)
            if lineage is None:
                lineage = []
                self._pending[gen.op_id] = lineage
            gen.attempt_index = len(lineage)
            # A second legacy-point draw for an op that already has a
            # primary is a RETRY, not a sibling and not a primary: it
            # re-answers the same prompt without exploring. The record seam
            # cannot know this (it sees one call); the lineage can.
            if gen.draw_kind == DRAW_PRIMARY and any(
                g.draw_kind == DRAW_PRIMARY for g in lineage
            ):
                gen.draw_kind = DRAW_RETRY
            lineage.append(gen)
            self._pending.move_to_end(gen.op_id)
            # The cap counts GENERATIONS. Counting ops would let a single
            # retry-storming op hold unbounded memory while the map looked
            # one entry deep -- the leak this structure could otherwise
            # introduce.
            while self._pending_count() > cap and self._pending:
                oldest_op = next(iter(self._pending))
                victim = self._pending[oldest_op]
                # Drop the OLDEST attempt of the oldest op, not the whole
                # lineage: evicting a list would discard newer generations
                # to make room for one.
                victim.pop(0)
                if not victim:
                    self._pending.pop(oldest_op, None)
                self._stats["pending_evicted"] += 1
                logger.warning(
                    "[TrajectoryRecorder] evicted one pending generation of "
                    "op=%s (cap=%d generations) — that attempt is lost; "
                    "raise %s if this recurs",
                    oldest_op, cap, _ENV_PENDING_MAX,
                )

    async def _retract(self, evt: "_RetractEvent") -> None:
        """Drop a generation the generator has since REJECTED.

        Removes every pending generation of ``evt.op_id`` whose candidate
        set is entirely covered by ``evt.candidate_hashes`` -- a draw is one
        provider call, so its hashes are exactly one lineage entry. The
        survivors are re-indexed so ``attempt_index`` stays dense and
        ``lineage_size`` (stamped at write) stays honest.

        Transactional by construction: the removal happens under the same
        guard ``_admit_pending`` and ``_resolve`` take, and the event sits
        on the one queue those operations share, so a retract can never
        race the write it is meant to pre-empt. A retract naming a
        generation that was never admitted (or already written) is counted
        as an orphan and named, never guessed at -- the row it would have
        removed is either not there or already on disk, and this reader
        must not decide which.
        """
        wanted = {h for h in evt.candidate_hashes if h}
        if not wanted or not evt.op_id:
            return
        removed = 0
        async with self._guard():
            lineage = self._pending.get(evt.op_id)
            if lineage:
                # ONE retract names ONE draw, and a draw is the MOST RECENT
                # generation carrying those hashes. Scanning the whole
                # lineage removed two generations in soak
                # bt-2026-09-02-013719: a byte-identical twin carries the
                # same candidate_hash as the candidate it duplicates, so a
                # subset match also took the accepted generation and its
                # verdict orphaned. Walk from the newest and stop at the
                # first match.
                keep: List[_PendingGeneration] = list(lineage)
                for idx in range(len(keep) - 1, -1, -1):
                    gen = keep[idx]
                    hashes = {
                        str((c or {}).get("candidate_hash", "") or "")
                        for c in gen.candidates if isinstance(c, dict)
                    }
                    hashes.discard("")
                    if hashes and hashes <= wanted:
                        del keep[idx]
                        removed = 1
                        break
                if removed:
                    for idx, gen in enumerate(keep):
                        gen.attempt_index = idx
                    if keep:
                        self._pending[evt.op_id] = keep
                    else:
                        self._pending.pop(evt.op_id, None)
        if removed:
            self._stats["generations_retracted"] = (
                self._stats.get("generations_retracted", 0) + removed
            )
            logger.info(
                "[TrajectoryRecorder] retracted %d generation(s) for op=%s "
                "(%s) -- will not reach the corpus",
                removed, evt.op_id, evt.reason or "rejected",
            )
        else:
            self._stats["orphan_retractions"] = (
                self._stats.get("orphan_retractions", 0) + 1
            )
            logger.debug(
                "[TrajectoryRecorder] retraction for op=%s matched no "
                "pending generation (hashes=%s)",
                evt.op_id, [h[:12] for h in sorted(wanted)],
            )

    async def _attach_verdict(self, evt: "_CandidateVerdictEvent") -> None:
        """Record one sibling's VALIDATE result onto its pending generation.

        Ordering is sound by construction: the generation is queued when
        the provider returns, and VALIDATE cannot judge a candidate that
        has not been generated -- so the pending entry always exists first.
        A verdict that finds none is therefore a real anomaly, counted and
        named rather than silently dropped.
        """
        async with self._guard():
            lineage = self._pending.get(evt.op_id)
            if not lineage:
                self._stats["orphan_candidate_verdicts"] += 1
                logger.debug(
                    "[TrajectoryRecorder] candidate verdict for op=%s "
                    "cand=%s had no pending generation",
                    evt.op_id, evt.candidate_hash[:12],
                )
                return
            # Attach to the attempt that actually PRODUCED this candidate.
            # With a lineage, "the op's generation" is ambiguous and
            # guessing the newest would credit a retry's verdict to the
            # wrong attempt -- mislabelling exactly the row a pair is built
            # from. candidate_hash is unique per candidate, so it resolves
            # the attempt without needing a second identity.
            target = next(
                (
                    g for g in reversed(lineage)
                    if any(
                        str((c or {}).get("candidate_hash", "") or "")
                        == evt.candidate_hash
                        for c in g.candidates
                        if isinstance(c, dict)
                    )
                ),
                None,
            )
            if target is None:
                # A verdict whose candidate belongs to no recorded attempt
                # (its generation was evicted, say). Counted, never guessed
                # onto a neighbour.
                self._stats["orphan_candidate_verdicts"] += 1
                logger.debug(
                    "[TrajectoryRecorder] candidate verdict for op=%s "
                    "cand=%s matched no attempt in a %d-deep lineage",
                    evt.op_id, evt.candidate_hash[:12], len(lineage),
                )
                return
            target.candidate_verdicts[evt.candidate_hash] = (
                evt.passed, evt.failure_class,
            )
            _detail = getattr(evt, "failure_detail", "") or ""
            if _detail:
                target.candidate_details[evt.candidate_hash] = _detail
            self._stats["candidate_verdicts_joined"] += 1

    def _guard(self) -> Any:
        """The pending-map lock, created lazily with the loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _resolve(self, evt: _OutcomeEvent) -> None:
        async with self._guard():
            lineage = self._pending.pop(evt.op_id, None)
        gen = (lineage or [None])[0]
        if gen is None:
            # Two very different causes, and the distinction matters:
            #   * an op caged before GENERATE has no candidate text, so
            #     there is genuinely nothing to record; or
            #   * the op_id on the generation side does not MATCH the one
            #     on the terminal side, in which case the join is silently
            #     broken and every trajectory is being lost.
            # Only the log can tell them apart, so name the op.
            self._stats["orphan_outcomes"] += 1
            logger.info(
                "[TrajectoryRecorder] verdict for op=%s had no pending "
                "generation (reason=%s). Expected when the op was caged "
                "before GENERATE; if generations ARE happening, the "
                "generation/terminal op_ids disagree and the join is broken.",
                evt.op_id, evt.terminal_reason or "?",
            )
            return
        outcome, autonomy_type, should_train = classify_terminal_reason(
            evt.terminal_reason, evt.terminal_phase,
        )
        # EVERY attempt in the lineage is written, not just the survivor.
        # The op's verdict describes where the op ENDED, and that is the
        # honest label for each attempt that led there: a retry exists
        # because the previous attempt was rejected, so the earlier rows
        # carry the model's rejected work on the same prompt. Discarding
        # them was discarding the better half of the pair.
        lineage = self._validate_lineage(evt.op_id, list(lineage or []))
        for _n, _gen in enumerate(lineage or []):
            _outcome, _autonomy, _train = outcome, autonomy_type, should_train
            _outcome, _autonomy, _train = self._noop_override(
                _gen, (_outcome, _autonomy, _train),
            )
            _gen.lineage_size = len(lineage or [])
            await self._write(_gen, _outcome, _autonomy, _train, evt)
        # The op's lineage is on disk: release its dedupe set. The map is
        # bounded by construction, so a leak here would be a slow one, but
        # a set that outlives its op is still state nobody reads again.
        _seen = getattr(self, "_persisted", None)
        if _seen:
            _seen.pop(evt.op_id, None)

    @staticmethod
    def _self_evidencing_policy(
        gen: "_PendingGeneration",
    ) -> "Optional[_OutcomePolicy]":
        """The policy a generation ESTABLISHES about itself, or None.

        A draw is self-evidencing when its own content settles its outcome,
        with no downstream phase required:

        * a REFUSAL (``is_noop``) declared the outcome and its diff is null
          by construction -- no candidate was produced and the recorded body
          is the decline envelope itself;
        * an UNPARSEABLE draw failed at the AST parser. That is a
          deterministic fact established at generation time -- the parser IS
          the verdict -- and it is a stronger claim than a refusal's, since
          nothing downstream could have rescued code that does not parse.
          It maps to _FAILURE: the candidate was bad on its own merits,
          which is precisely the trainable failure.

        A PATCH is deliberately absent. Whether its code is CORRECT is
        genuinely unknown until VALIDATE runs, and guessing would put a
        fabricated label in the corpus.

        Pure; NEVER raises.
        """
        try:
            if gen.is_noop:
                return _NOOP
            cands = gen.candidates or ()
            if cands and all(
                isinstance(c, dict)
                and str(c.get("candidate_status", "") or "") == "parse_error"
                for c in cands
            ):
                return _FAILURE
        except Exception:  # noqa: BLE001 - a label rule never breaks a write
            return None
        return None

    @classmethod
    def _noop_override(
        cls, gen: "_PendingGeneration", policy: "_OutcomePolicy",
    ) -> "_OutcomePolicy":
        """Apply the self-evidencing rule. ONE place, both writers.

        The verdict-joined path and the pending-EXPIRY path each decide a
        row's policy, and the first version of this rule lived only in the
        former. The expiry path hardcoded (unknown, intent_written, False),
        so a refusal whose op never reported was still discarded -- 6 of
        soak 24's first 40 rows. Its comment ("an outcome we never saw is
        not a label") is right for a PATCH, whose correctness genuinely is
        unknown, and wrong for a draw that already settled its own outcome.

        Measured on soak bt-2026-09-05-010735: three ops carried a patch AND
        a parse_error/noop sibling -- the exact {good, bad} pair the corpus
        exists to produce -- and every one read as UNMIXED downstream because
        the contrasting half was dropped `not_train`.
        """
        established = cls._self_evidencing_policy(gen)
        if established is not None and policy in _NOOP_OVERRIDABLE_POLICIES:
            return established
        return policy

    def _validate_lineage(
        self, op_id: str, lineage: "List[_PendingGeneration]",
    ) -> "List[_PendingGeneration]":
        """The lineage validation guard: prune what cannot be a preference
        pair BEFORE it reaches the corpus, and re-index the survivors.

        Dropped, with a counter each:
          * a generation with no candidates (nothing to pair);
          * a REPAIR or RETRY whose every candidate hash duplicates one an
            earlier generation of this op already carries -- the L2 path
            re-recording the accepted candidate under an op-level verdict,
            the exact twin that inflated soak 17's groups (29 of 43 rows).
            A repair that produced something NEW is kept: it is a genuine
            second answer to the prompt.

        Survivors are re-indexed so ``attempt_index`` stays dense and
        ``lineage_size`` honest, mirroring what ``_retract`` already does.
        Pure over its input, NEVER raises -- on any fault the lineage is
        returned untouched, because a guard that drops rows on its own
        bug is worse than no guard."""
        try:
            seen_hashes: set = set()
            kept: List[_PendingGeneration] = []
            for gen in lineage:
                hashes = {
                    str((c or {}).get("candidate_hash", "") or "")
                    for c in (gen.candidates or ()) if isinstance(c, dict)
                }
                hashes.discard("")
                if not gen.candidates:
                    self._stats["lineage_pruned"] += 1
                    continue
                if gen.draw_kind in (DRAW_REPAIR, DRAW_RETRY) and hashes and hashes <= seen_hashes:
                    self._stats["lineage_pruned"] += 1
                    logger.info(
                        "[TrajectoryRecorder] pruned %s draw for op=%s: every "
                        "candidate duplicates an earlier attempt (%s)",
                        gen.draw_kind, op_id, sorted(h[:12] for h in hashes),
                    )
                    continue
                seen_hashes |= hashes
                kept.append(gen)
            for idx, gen in enumerate(kept):
                gen.attempt_index = idx
            return kept
        except Exception:  # noqa: BLE001 — a guard must never lose a lineage
            logger.debug("[TrajectoryRecorder] lineage guard degraded", exc_info=True)
            return list(lineage)

    async def _expire_pending(self) -> None:
        """Flush generations whose op never reported a verdict.

        Collects under the lock and writes OUTSIDE it: the write awaits a
        cross-process file lock, and holding the pending-map lock across
        that would block every concurrent emit for the duration of disk
        I/O.
        """
        ttl = _env_float(
            _ENV_PENDING_TTL_S, _DEFAULT_PENDING_TTL_S, 30.0, 86_400.0
        )
        now = time.monotonic()
        expired: list = []
        async with self._guard():
            # Age each ATTEMPT on its own clock. Expiring a whole lineage
            # on its oldest member would flush a retry that is seconds old
            # and still likely to get a real verdict; expiring on its
            # newest would pin stale attempts in memory for as long as an
            # op keeps retrying -- the leak this structure has to avoid.
            for op_id in list(self._pending.keys()):
                lineage = self._pending.get(op_id) or []
                fresh = [
                    g for g in lineage
                    if (now - g.created_monotonic) <= ttl
                ]
                if len(fresh) == len(lineage):
                    continue
                for g in lineage:
                    if (now - g.created_monotonic) > ttl:
                        expired.append((op_id, g))
                if fresh:
                    self._pending[op_id] = fresh
                else:
                    self._pending.pop(op_id, None)

        for op_id, gen in expired:
            self._stats["pending_expired"] += 1
            # The breakpoint, named: this generation produced candidates
            # and its op never reported a terminal reason. Recorded as
            # non-trainable, because an outcome we never saw is not a
            # label -- but recorded, because the candidate text is still
            # evidence about the model.
            logger.warning(
                "[TrajectoryRecorder] op=%s expired after %.0fs with %d "
                "candidate(s) and NO verdict — writing outcome=unknown, "
                "should_train=false. If this is common, ops are outliving "
                "the TTL (%s) or never reaching a terminal phase.",
                op_id, ttl, len(gen.candidates), _ENV_PENDING_TTL_S,
            )
            _exp_outcome, _exp_autonomy, _exp_train = self._noop_override(
                gen, _UNKNOWN,
            )
            await self._write(
                gen,
                _exp_outcome,
                _exp_autonomy,
                _exp_train,
                _OutcomeEvent(
                    op_id=op_id, terminal_phase="", terminal_reason="",
                ),
            )

    async def _write(
        self,
        gen: _PendingGeneration,
        outcome: str,
        autonomy_type: str,
        should_train: bool,
        evt: _OutcomeEvent,
    ) -> None:
        try:
            from backend.core.ouroboros.governance.cross_process_jsonl import (  # noqa: E501
                async_flock_append_line,
            )
        except Exception:  # noqa: BLE001
            self._stats["write_failures"] += 1
            return

        out_cap = _env_int(
            _ENV_MAX_OUTPUT_CHARS, _DEFAULT_MAX_OUTPUT_CHARS, 256, 2_000_000
        )
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._stats["write_failures"] += 1
            return

        n_cands = len(gen.candidates)
        # How many ANSWERS this generation holds, as opposed to how many
        # rows it is about to write. Computed once for the whole group.
        _structure_ids, _n_distinct = _structure_stamps(gen.candidates)

        for idx, cand in enumerate(gen.candidates):
            if not isinstance(cand, dict):
                continue
            body = str(cand.get("full_content", "") or "")
            if not body:
                continue
            cand_hash = str(cand.get("candidate_hash", "") or "")
            # DETERMINISTIC DEDUPE at the persistence layer, keyed by the
            # candidate's OWN hash (the one the retract seam already keys
            # on -- no second hasher). One (op_id, candidate_hash) reaches
            # the corpus once, whatever path produced it again.
            if cand_hash:
                _seen = getattr(self, "_persisted", None)
                if _seen is None:
                    _seen = self._persisted = {}
                _op_seen = _seen.setdefault(gen.op_id, set())
                if cand_hash in _op_seen:
                    self._stats["rows_deduped"] += 1
                    continue
                _op_seen.add(cand_hash)
                # Bounded: cap the per-op map so a flood of ops that never
                # report an outcome cannot grow it without limit.
                while len(_seen) > 512:
                    _seen.pop(next(iter(_seen)))
            # Per-candidate verdict beats the op-level one where it exists.
            #
            # A candidate VALIDATE rejected was bad on its own merits --
            # broken syntax, failing tests -- and that is model quality
            # regardless of how the op later ended, so it is trainable
            # even when the op died of something the model did not cause.
            # A candidate that PASSED has only earned "not broken"; what
            # happened after (GATE, APPLY, VERIFY) is the op-level verdict,
            # so it inherits that. Siblings of one op therefore end up with
            # DIFFERENT outcomes, which is the entire point: identical
            # outcomes score identically and the ranker emits no pair.
            _verdict = gen.candidate_verdicts.get(cand_hash)
            if _verdict is not None and not _verdict[0]:
                # ...but ONLY when the candidate was actually judged. A
                # verdict whose failure_class is infrastructure means the
                # tests never ran -- a timeout, an exhausted budget, a
                # harness fault. Labelling that a FAILURE trains the model
                # against code no one assessed, which is worse than
                # discarding the row: it is a confident wrong label.
                # Measured: 20 of 36 per-candidate rows carried
                # failure_class=infra with should_train=True.
                if _verdict[1] in _UNASSESSED_FAILURE_CLASSES:
                    c_outcome, c_autonomy, c_train = _INFRA
                else:
                    c_outcome, c_autonomy, c_train = _FAILURE
                c_reason = _verdict[1] or "validation_failed"
            else:
                c_outcome, c_autonomy, c_train = outcome, autonomy_type, should_train
                c_reason = evt.terminal_reason
            c_confidence = (
                1.0 if c_outcome == "success"
                else (0.0 if c_outcome == "failure" else 0.5)
            )
            # Phase 2 — the specific assertion/AST cause this sibling died on,
            # so the ranker can learn from the CAUSE, not only the category.
            c_detail = gen.candidate_details.get(cand_hash, "")
            line = json.dumps(
                {
                    "event_id": str(uuid.uuid4()),
                    "schema_version": TRAJECTORY_RECORDER_SCHEMA_VERSION,
                    "event_type": "interaction",
                    "source": "jarvis_body",
                    "timestamp": gen.created_iso,
                    "user_input": gen.prompt,
                    "assistant_output": _truncate(body, out_cap),
                    "system_context": "",
                    "outcome": c_outcome,
                    "confidence": c_confidence,
                    "model_id": gen.model_id,
                    "latency_ms": gen.latency_ms,
                    # Canonical ExperienceEvent field, kept for the reactor
                    # consumers that read it; the split values below are the
                    # ones throughput analysis uses.
                    "tokens_used": gen.prompt_tokens + gen.completion_tokens,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "tokens_per_second": (
                        round(
                            gen.completion_tokens / (gen.latency_ms / 1000.0),
                            2,
                        )
                        if gen.latency_ms > 0 and gen.completion_tokens
                        else 0.0
                    ),
                    # Provenance of the two counts above and the rate they
                    # form. False = the engine reported them; True = they
                    # were inferred from response length. Filter on this
                    # before ranking models by throughput.
                    "tokens_estimated": gen.tokens_estimated,
                    "session_id": gen.session_id,
                    "task_type": gen.task_type,
                    "metadata": {
                        "op_id": gen.op_id,
                        "prompt_key": gen.prompt_key,
                        "candidate_id": str(
                            cand.get("candidate_id", "") or ""
                        ),
                        "candidate_hash": cand_hash,
                        "candidate_index": idx,
                        "n_candidates": n_cands,
                        # The specific test-gate cause (Phase 2), empty on a
                        # pass or an unassessed/infra failure.
                        "failure_detail": c_detail,
                        # `n_candidates` counts ROWS; this counts ANSWERS.
                        # A consumer selecting trainable groups must filter
                        # on THIS -- three rows sharing one structure_id
                        # cannot yield a preference pair, and filtering on
                        # `n_candidates >= 2` was selecting exactly those.
                        "n_distinct_structures": _n_distinct,
                        # Docstring-stripped AST digest: rows differing only
                        # in prose share an id. Empty when the candidate does
                        # not parse -- unparseable answers are real and must
                        # not be folded together under one shared id.
                        "structure_id": _structure_ids.get(cand_hash, ""),
                        # Lineage position. Two rows of one op with
                        # different attempt_index are the SAME prompt
                        # answered twice, which is the shape a preference
                        # pair is built from.
                        "attempt_index": gen.attempt_index,
                        "lineage_size": gen.lineage_size,
                        # The draw-kind discriminator: harvest and Reactor
                        # ingestion pair only GENUINE draws (primary /
                        # sibling); an L2 repair iteration is a different
                        # prompt's answer and never a sibling of these.
                        "draw_kind": gen.draw_kind,
                        "temperature": gen.temperature,
                        "sampling": dict(gen.sampling),
                        # Deterministic discriminator: "noop" for a
                        # synthesised refusal, "patch" for a real
                        # candidate. Downstream selects on THIS rather
                        # than inferring from an empty file_path or by
                        # parsing the body.
                        "candidate_status": str(
                            cand.get("candidate_status", "") or "patch"
                        ),
                        "file_path": str(cand.get("file_path", "") or ""),
                        "source_path": str(cand.get("source_path", "") or ""),
                        "provider_name": gen.provider_name,
                        "is_noop": gen.is_noop,
                        "cost_usd": gen.cost_usd,
                        "terminal_phase": evt.terminal_phase,
                        "terminal_reason": c_reason,
                        # Same exclusion policy as reactor-core's
                        # autonomy_classifier: infrastructure != quality.
                        "autonomy_event_type": c_autonomy,
                        "should_train": c_train,
                        # Whether THIS row's outcome came from VALIDATE
                        # judging this candidate, or was inherited from the
                        # op. A sibling set that is all-inherited explains
                        # its own zero pair count without re-deriving it.
                        "verdict_source": (
                            "candidate" if _verdict is not None else "operation"
                        ),
                        "candidate_validated": (
                            None if _verdict is None else bool(_verdict[0])
                        ),
                        # Attempt-scoped. Without it, two attempts of one
                        # op collide whenever candidate_hash is absent and
                        # the key falls back to the index -- so a
                        # downstream dedup would drop the retry, which is
                        # the row this whole change exists to keep.
                        "idempotency_key": (
                            f"{gen.op_id}:{gen.attempt_index}:"
                            f"{cand_hash or idx}"
                        ),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            ok = await async_flock_append_line(path, line)
            if ok:
                self._stats["events_written"] += 1
            else:
                self._stats["write_failures"] += 1

    # -- shutdown / introspection ----------------------------------------
    async def drain(self, timeout_s: float = 5.0) -> bool:
        """Flush queued work. Returns True if fully drained in time."""
        if self._queue is None:
            return True
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self, timeout_s: float = 5.0) -> None:
        await self.drain(timeout_s)
        # Flush anything still awaiting a verdict BEFORE tearing down, or a
        # clean shutdown silently discards the trajectories of every op
        # that was still in flight.
        try:
            prev = os.environ.get(_ENV_PENDING_TTL_S)
            os.environ[_ENV_PENDING_TTL_S] = "30"
            # Values are LINEAGES now, not single generations. Iterating
            # them as objects would set an attribute on a list and raise
            # into the fail-open below -- turning the final flush back into
            # the silent no-op it was before it had a caller at all.
            for lineage in list(self._pending.values()):
                for gen in lineage:
                    gen.created_monotonic = 0.0
            await self._expire_pending()
            if prev is None:
                os.environ.pop(_ENV_PENDING_TTL_S, None)
            else:
                os.environ[_ENV_PENDING_TTL_S] = prev
        except Exception:  # noqa: BLE001
            logger.debug("[TrajectoryRecorder] final flush failed", exc_info=True)

        for task_attr in ("_watchdog", "_writer"):
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    pass
            setattr(self, task_attr, None)

    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "pending_open": len(self._pending),
            "enabled": recorder_enabled(),
            "path": str(self.path),
        }


# ---------------------------------------------------------------------------
# Module-level singleton + thin emit helpers (the call-site surface)
# ---------------------------------------------------------------------------

_default_recorder: Optional[TrajectoryRecorder] = None


def get_recorder() -> TrajectoryRecorder:
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = TrajectoryRecorder()
    return _default_recorder


def harvest_snapshot(
    *, max_rows: int = 20_000, events_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """What the flywheel has actually harvested. Read-only. NEVER raises.

    ONE definition of "how is the harvest going", so the REPL verb, any
    status chip and a soak postmortem cannot each answer it differently.
    It lives here because this module owns both halves: the live in-process
    counters AND the file those counters write to.

    ## The number that matters is not `rows`

    A sibling group is rows sharing an ``op_id`` with different
    ``attempt_index`` -- ``record_generation`` is called per PROVIDER CALL,
    so each draw writes its own row with ``n_candidates=1``. Counting rows
    therefore says nothing about trainability: measured 2026-09-01, 8
    sibling rows across 3 groups carried 3 structurally distinct answers
    and every group collapsed to one, so not one preference pair was
    constructible while the row count looked healthy.

    So this reports ``groups_pairable`` -- groups holding 2+ structurally
    distinct answers -- alongside ``groups``. A soak whose rows climb while
    ``groups_pairable`` stays 0 is producing nothing, and that is exactly
    the failure an operator watching a row counter would not see.

    Prefers each row's recorded ``structure_id`` and falls back to
    fingerprinting the text, so rows written before that field existed are
    still counted rather than silently dropped from the denominator.
    """
    out: Dict[str, Any] = {
        "enabled": False, "counters": {}, "path": "", "rows": 0,
        "rows_trainable": 0, "groups": 0, "groups_pairable": 0,
        "groups_collapsed": 0, "truncated": False, "error": "",
        # Lineage purification: rows excluded because they are not a genuine
        # primary/sibling draw (L2 repair / retry re-generations), and rows
        # dropped as an exact (op_id, candidate_hash) duplicate of one seen.
        "rows_repair": 0, "rows_deduped": 0,
    }
    try:
        out["enabled"] = recorder_enabled()
        try:
            out["counters"] = dict(get_recorder().stats())
        except Exception:  # noqa: BLE001 — a dead recorder still has a corpus
            out["counters"] = {}
        path = Path(events_path) if events_path is not None else events_dir()
        out["path"] = str(path)

        groups: Dict[str, List[Dict[str, Any]]] = {}
        seen_keys: set = set()
        rows = 0
        for f in sorted(path.glob("*.jsonl")) if path.is_dir() else []:
            try:
                fh = f.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if rows >= max_rows:
                        out["truncated"] = True
                        break
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if row.get("event_type") != "interaction":
                        continue
                    rows += 1
                    meta = row.get("metadata") or {}
                    # Purification, in the same order Reactor ingestion
                    # applies it: a repair/retry re-generation is not a
                    # sibling of the primary draw it re-answered, and one
                    # (op_id, candidate_hash) is one row however many
                    # paths re-recorded it.
                    if not is_genuine_draw(meta):
                        out["rows_repair"] += 1
                        continue
                    _dk = (str(meta.get("op_id", "") or ""),
                           str(meta.get("candidate_hash", "") or ""))
                    if _dk[1]:
                        if _dk in seen_keys:
                            out["rows_deduped"] += 1
                            continue
                        seen_keys.add(_dk)
                    if meta.get("should_train"):
                        out["rows_trainable"] += 1
                    key = str(
                        meta.get("op_id")
                        or meta.get("prompt_key")
                        or (row.get("user_input") or "")[:80]
                    )
                    groups.setdefault(key, []).append(row)
            if out["truncated"]:
                break
        out["rows"] = rows

        multi = [v for v in groups.values() if len(v) > 1]
        out["groups"] = len(multi)
        for members in multi:
            ids = set()
            for row in members:
                meta = row.get("metadata") or {}
                sid = str(meta.get("structure_id") or "")
                if not sid:
                    sid = _fingerprint_id(row.get("assistant_output") or "")
                if sid:
                    ids.add(sid)
            if len(ids) <= 1:
                continue                      # identical ids — definitely one answer
            # Distinct ids are NECESSARY but not sufficient. The acceptance
            # filter judges by SIMILARITY, so a group whose answers differ
            # by an unused import has two ids and one answer -- measured at
            # 0.9987 and 0.9999 on this corpus. Counting those as pairable
            # would make this surface disagree with the filter that decides
            # what gets kept, and the disagreement would always flatter the
            # harvest. Only groups that clear the cheap test pay for
            # fingerprinting, so the common (collapsed) case stays cheap.
            if _group_has_distinct_answers(members):
                out["groups_pairable"] += 1
        out["groups_collapsed"] = out["groups"] - out["groups_pairable"]
        return out
    except Exception as exc:  # noqa: BLE001 — an observability read is never fatal
        logger.debug("[TrajectoryRecorder] harvest_snapshot failed", exc_info=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _group_has_distinct_answers(members: "List[Dict[str, Any]]") -> bool:
    """True when two rows of this group differ by MORE than the threshold.

    Reuses ``sibling_entropy``'s own predicate rather than re-deriving one,
    so "distinct enough to train on" has a single definition shared by the
    filter that accepts a draw and the surface that reports the harvest.
    """
    try:
        from backend.core.ouroboros.governance import (  # noqa: PLC0415
            sibling_entropy as _ent,
        )
        seen: List[str] = []
        for row in members:
            fp = _ent.structural_fingerprint(row.get("assistant_output") or "")
            if fp is None:
                continue
            redundant, _ = _ent.is_structurally_redundant([fp], seen)
            if seen and not redundant:
                return True
            seen.append(fp)
        return False
    except Exception:  # noqa: BLE001
        return False


def _fingerprint_id(text: str) -> str:
    """Structure id for a row written before ``structure_id`` existed."""
    try:
        from backend.core.ouroboros.governance import (  # noqa: PLC0415
            sibling_entropy as _ent,
        )
        fp = _ent.structural_fingerprint(text)
        if fp is None:
            return ""
        return hashlib.sha256(fp.encode("utf-8", "replace")).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return ""


def reset_recorder_for_tests(
    path: Optional[Path] = None,
) -> TrajectoryRecorder:
    global _default_recorder
    _default_recorder = TrajectoryRecorder(path=path)
    return _default_recorder


def record_generation(**kwargs: Any) -> bool:
    """Fire-and-forget generation emit. NEVER raises."""
    try:
        return get_recorder().record_generation(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[TrajectoryRecorder] record_generation failed", exc_info=True,
        )
        return False


def record_outcome(**kwargs: Any) -> bool:
    """Fire-and-forget verdict emit. NEVER raises."""
    try:
        return get_recorder().record_outcome(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[TrajectoryRecorder] record_outcome failed", exc_info=True,
        )
        return False


def record_retraction(**kwargs: Any) -> bool:
    """Fire-and-forget retraction of a rejected draw. NEVER raises."""
    try:
        return get_recorder().record_retraction(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[TrajectoryRecorder] record_retraction failed", exc_info=True,
        )
        return False


def record_candidate_verdict(**kwargs: Any) -> bool:
    """Fire-and-forget per-candidate VALIDATE verdict. NEVER raises."""
    try:
        return get_recorder().record_candidate_verdict(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug(
            "[TrajectoryRecorder] record_candidate_verdict failed",
            exc_info=True,
        )
        return False


def register_flags(registry) -> int:  # noqa: ANN001
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category, FlagSpec, FlagType,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[TrajectoryRecorder] register_flags degraded: %s", exc,
        )
        return 0
    tgt = (
        "backend/core/ouroboros/governance/observability/"
        "trajectory_recorder.py"
    )
    specs = [
        FlagSpec(
            name=_ENV_MASTER, type=FlagType.BOOL, default=False,
            category=Category.OBSERVABILITY, source_file=tgt,
            example=f"{_ENV_MASTER}=true",
            description=(
                "Master for the O+V trajectory recorder. OFF (default, "
                "§33.1) => both emit calls are a flag read and a return. "
                "ON => one canonical ExperienceEvent line per candidate "
                "is appended to the Trinity events dir for Reactor-Core."
            ),
        ),
        FlagSpec(
            name=_ENV_DIR, type=FlagType.STR, default="",
            category=Category.OBSERVABILITY, source_file=tgt,
            example=f"{_ENV_DIR}=/home/jarvis_svc/.jarvis/trinity/events",
            description=(
                "Override the events directory. Empty => "
                "~/.jarvis/trinity/events, which the Trinity experience "
                "receiver already watches."
            ),
        ),
        FlagSpec(
            name=_ENV_QUEUE_MAX, type=FlagType.INT,
            default=_DEFAULT_QUEUE_MAX, category=Category.CAPACITY,
            source_file=tgt, example=f"{_ENV_QUEUE_MAX}=1024",
            description=(
                "Emit queue depth. A full queue DROPS the event (counted "
                "in stats) rather than blocking the control loop."
            ),
        ),
        FlagSpec(
            name=_ENV_PENDING_MAX, type=FlagType.INT,
            default=_DEFAULT_PENDING_MAX, category=Category.CAPACITY,
            source_file=tgt, example=f"{_ENV_PENDING_MAX}=512",
            description=(
                "Max generations held awaiting a verdict (LRU "
                "drop-oldest). Bounds memory when ops never terminate."
            ),
        ),
        FlagSpec(
            name=_ENV_PENDING_TTL_S, type=FlagType.FLOAT,
            default=_DEFAULT_PENDING_TTL_S, category=Category.TIMING,
            source_file=tgt, example=f"{_ENV_PENDING_TTL_S}=1800",
            description=(
                "Seconds a generation waits for its verdict before being "
                "flushed with outcome=unknown/should_train=false."
            ),
        ),
        FlagSpec(
            name=_ENV_TICK_S, type=FlagType.FLOAT,
            default=_DEFAULT_TICK_S, category=Category.TIMING,
            source_file=tgt, example=f"{_ENV_TICK_S}=30",
            description=(
                "Seconds between wall-clock expiry sweeps. A dedicated "
                "watchdog task owns this because expiry driven by queue "
                "activity is not expiry: on a sparse workload the pending "
                "generation waits for an event that never arrives and its "
                "trajectory is never written."
            ),
        ),
        FlagSpec(
            name=_ENV_MAX_PROMPT_CHARS, type=FlagType.INT,
            default=_DEFAULT_MAX_PROMPT_CHARS, category=Category.CAPACITY,
            source_file=tgt, example=f"{_ENV_MAX_PROMPT_CHARS}=48000",
            description="Per-event prompt char cap (truncated, marked).",
        ),
        FlagSpec(
            name=_ENV_MAX_OUTPUT_CHARS, type=FlagType.INT,
            default=_DEFAULT_MAX_OUTPUT_CHARS, category=Category.CAPACITY,
            source_file=tgt, example=f"{_ENV_MAX_OUTPUT_CHARS}=48000",
            description="Per-event candidate char cap (truncated, marked).",
        ),
    ]
    n = 0
    for s in specs:
        try:
            registry.register(s)
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[TrajectoryRecorder] seed %s skipped: %s", s.name, exc,
            )
    return n


__all__ = [
    "TRAJECTORY_RECORDER_SCHEMA_VERSION",
    "TrajectoryRecorder",
    "classify_terminal_reason",
    "events_dir",
    "noop_candidate",
    "parse_error_candidate",
    "get_recorder",
    "harvest_snapshot",
    "record_candidate_verdict",
    "record_generation",
    "record_outcome",
    "record_retraction",
    "recorder_enabled",
    "register_flags",
    "reset_recorder_for_tests",
]
