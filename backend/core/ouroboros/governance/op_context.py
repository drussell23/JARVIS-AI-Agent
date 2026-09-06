"""
Operation Context & Phase State Machine
========================================

Typed, frozen state object that flows through every Ouroboros pipeline phase.

``OperationContext`` is immutable -- all mutations produce a **new** instance
via :meth:`OperationContext.advance`, which enforces the phase state machine
and extends a SHA-256 hash chain so that every state transition is
cryptographically linked to the previous one.

Phase Transitions
-----------------

.. code-block:: text

    CLASSIFY -> ROUTE -> [CONTEXT_EXPANSION] -> [PLAN] -> GENERATE -> VALIDATE -> GATE -> [APPROVE] -> APPLY -> VERIFY -> COMPLETE
                                                           |              |        |       |           |          |
                                                           v              v        v       v           v          v
                                                      GEN_RETRY      VAL_RETRY         EXPIRED    POSTMORTEM  POSTMORTEM
                                                           |              |
                                                           v              v
                                                       VALIDATE         GATE

    (most non-terminal phases can also transition to CANCELLED)

Terminal phases: COMPLETE, CANCELLED, EXPIRED, POSTMORTEM
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

if TYPE_CHECKING:
    from backend.core.ouroboros.governance.patch_benchmarker import BenchmarkResult
    from backend.core.ouroboros.governance.epistemic_prefetch import PrefetchEntry

from backend.core.ouroboros.governance.operation_id import generate_operation_id
from backend.core.ouroboros.governance.risk_engine import RiskTier
from backend.core.ouroboros.governance.routing_policy import RoutingDecision

import logging as _logging

_logger = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase-transition lifecycle event (2026-07-18)
#
# ``advance()`` is the ONE method every legal FSM phase transition flows through
# (it validates PHASE_TRANSITIONS). It fires a lifecycle event here so observers
# can track the live phase WITHOUT sprinkling per-phase calls across the
# orchestrator. The in-flight registry registers an observer (via
# in_flight_registry.wire_phase_transition_tracking) that mirrors each transition
# into the registry — so fsm_checkpoint.capture_inflight serializes the EXACT
# current phase (not the stale registration-time phase).
#
# op_context stays PURE: it owns only the observer LIST and fires blindly; it
# imports nothing from the observability substrate, so the state-machine →
# observability direction the in_flight_registry deliberately avoids is never
# introduced here either. Registration is one-way (observer side imports us).
# ---------------------------------------------------------------------------

from typing import Callable, List

_PHASE_TRANSITION_OBSERVERS: "List[Callable[[str, str], None]]" = []


def register_phase_transition_observer(cb: "Callable[[str, str], None]") -> None:
    """Register a ``(op_id, phase_name) -> None`` callback fired on every
    ``advance()`` transition. Idempotent per-callable (a duplicate registration
    is ignored). NEVER raises."""
    try:
        if cb not in _PHASE_TRANSITION_OBSERVERS:
            _PHASE_TRANSITION_OBSERVERS.append(cb)
    except Exception:  # noqa: BLE001
        pass


def _reset_phase_transition_observers_for_tests() -> None:
    _PHASE_TRANSITION_OBSERVERS.clear()


def _notify_phase_transition(op_id: str, phase_name: str) -> None:
    """Fire all registered observers. Fully fail-soft — an observer fault must
    NEVER break the state-machine transition it is merely witnessing."""
    if not op_id or not _PHASE_TRANSITION_OBSERVERS:
        return
    for _cb in list(_PHASE_TRANSITION_OBSERVERS):
        try:
            _cb(op_id, phase_name)
        except Exception:  # noqa: BLE001
            _logger.debug("[op_context] phase-transition observer faulted", exc_info=True)


class ArchitecturalCycleError(ValueError):
    """Raised when dependency_edges contains a directed cycle.

    Detected at OperationContext construction via Kahn's algorithm.
    Prevents deadlock before the GENERATE phase.
    """


# ---------------------------------------------------------------------------
# Phase Enum
# ---------------------------------------------------------------------------


class OperationPhase(Enum):
    """Pipeline phase for an autonomous Ouroboros operation."""

    CLASSIFY = auto()
    ROUTE = auto()
    CONTEXT_EXPANSION = auto()
    PLAN = auto()           # Model-reasoned implementation planning (Manifesto §5)
    GENERATE = auto()
    GENERATE_RETRY = auto()
    VALIDATE = auto()
    VALIDATE_RETRY = auto()
    GATE = auto()
    APPROVE = auto()
    APPLY = auto()
    VERIFY = auto()
    VISUAL_VERIFY = auto()   # Post-APPLY UI regression check (Slices 3-4, Task 17)
    COMPLETE = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    POSTMORTEM = auto()


# ---------------------------------------------------------------------------
# Phase Transition Table
# ---------------------------------------------------------------------------

# Progress transitions only — terminal escapes (CANCELLED, POSTMORTEM,
# EXPIRED, COMPLETE-noop) are auto-injected below by
# _inject_terminal_reachability() so the table does not need to repeat them.
# Keep this table focused on the *forward flow* of the pipeline.
PHASE_TRANSITIONS: Dict[OperationPhase, Set[OperationPhase]] = {
    OperationPhase.CLASSIFY: {
        OperationPhase.ROUTE,
    },
    OperationPhase.ROUTE: {
        OperationPhase.CONTEXT_EXPANSION,
        OperationPhase.PLAN,            # fast-path: skip expansion, go directly to planning
        OperationPhase.GENERATE,
    },
    OperationPhase.CONTEXT_EXPANSION: {
        OperationPhase.PLAN,
        OperationPhase.GENERATE,       # direct-to-GENERATE for trivial ops (skip planning)
    },
    OperationPhase.PLAN: {
        OperationPhase.GENERATE,
    },
    OperationPhase.GENERATE: {
        OperationPhase.VALIDATE,
        OperationPhase.GENERATE_RETRY,
    },
    OperationPhase.GENERATE_RETRY: {
        OperationPhase.VALIDATE,
        OperationPhase.GENERATE_RETRY,
    },
    OperationPhase.VALIDATE: {
        OperationPhase.GATE,
        OperationPhase.VALIDATE_RETRY,
    },
    OperationPhase.VALIDATE_RETRY: {
        OperationPhase.GATE,
        OperationPhase.VALIDATE_RETRY,
    },
    OperationPhase.GATE: {
        OperationPhase.APPROVE,
        OperationPhase.APPLY,
    },
    OperationPhase.APPROVE: {
        OperationPhase.APPLY,
    },
    OperationPhase.APPLY: {
        OperationPhase.VERIFY,
    },
    OperationPhase.VERIFY: {
        # Optional — orchestrator chooses whether to invoke Visual
        # VERIFY based on env + trigger logic (spec §VERIFY Extension).
        # Back-compat: VERIFY can still terminate directly to COMPLETE
        # via the auto-injected terminal reachability, so existing
        # paths that don't know about Visual VERIFY keep working.
        OperationPhase.VISUAL_VERIFY,
    },
    OperationPhase.VISUAL_VERIFY: {
        # Visual VERIFY fail → L2 Repair via VALIDATE_RETRY (same
        # routing TestRunner-red uses). Pass → auto-injected COMPLETE.
        OperationPhase.VALIDATE_RETRY,
    },
    # Terminal phases -- no outgoing transitions
    OperationPhase.COMPLETE: set(),
    OperationPhase.CANCELLED: set(),
    OperationPhase.EXPIRED: set(),
    OperationPhase.POSTMORTEM: set(),
}

TERMINAL_PHASES: Set[OperationPhase] = {
    OperationPhase.COMPLETE,
    OperationPhase.CANCELLED,
    OperationPhase.EXPIRED,
    OperationPhase.POSTMORTEM,
}

# ---------------------------------------------------------------------------
# Dynamic Terminal Reachability Invariant
# ---------------------------------------------------------------------------
#
# Rule: every non-terminal phase can transition to every terminal phase.
#
# Why this is enforced here rather than maintained by hand per-phase:
#   • The hand-maintained table above only needs to declare *progress*
#     transitions (non-terminal → non-terminal) — terminal escapes
#     (CANCELLED / POSTMORTEM / EXPIRED / COMPLETE-noop) are auto-injected.
#   • Prevents the entire class of bugs where a new phase (e.g. VERIFY)
#     silently forbids an escape route and corrupts the FSM at runtime.
#     This was the root cause of the "Illegal phase transition:
#     VERIFY -> CANCELLED" incident the L2 repair path was hitting.
#   • COMPLETE is also terminal-reachable from any non-terminal phase to
#     preserve the noop fast-path semantics (model signals no change needed).
#
# Callers that want to restrict specific terminals (e.g. forbid CANCELLED
# from VERIFY for semantic reasons) should do so at the call site, not via
# the FSM. The FSM's job is to guarantee reachability; semantic choices
# belong to the orchestrator/hooks.
def _inject_terminal_reachability(
    transitions: Dict[OperationPhase, Set[OperationPhase]],
    terminals: Set[OperationPhase],
) -> None:
    """Auto-inject every terminal phase into every non-terminal phase's
    allowed-transition set. Idempotent and in-place.
    """
    for phase, allowed in transitions.items():
        if phase in terminals:
            continue  # terminals stay terminal (empty set)
        allowed.update(terminals)


def _verify_terminal_invariant(
    transitions: Dict[OperationPhase, Set[OperationPhase]],
    terminals: Set[OperationPhase],
) -> None:
    """Assert the terminal-reachability invariant at module load.

    Raises RuntimeError if any non-terminal phase is missing a terminal
    target — an explicit fast-fail instead of silently shipping a broken FSM.
    """
    for phase, allowed in transitions.items():
        if phase in terminals:
            if allowed:
                raise RuntimeError(
                    f"Terminal phase {phase.name} has outgoing transitions "
                    f"{sorted(p.name for p in allowed)} — terminals must be dead ends"
                )
            continue
        missing = terminals - allowed
        if missing:
            raise RuntimeError(
                f"Phase {phase.name} is missing terminal-escape routes: "
                f"{sorted(p.name for p in missing)}. "
                f"Every non-terminal phase must be able to reach every terminal phase."
            )


_inject_terminal_reachability(PHASE_TRANSITIONS, TERMINAL_PHASES)
_verify_terminal_invariant(PHASE_TRANSITIONS, TERMINAL_PHASES)


# ---------------------------------------------------------------------------
# Typed Sub-objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of the candidate generation phase.

    Parameters
    ----------
    candidates:
        Tuple of candidate dicts (each describing a proposed change).
    provider_name:
        Name of the model/provider that generated candidates.
    generation_duration_s:
        Wall-clock seconds spent generating candidates.
    """

    candidates: Tuple[Dict[str, Any], ...]
    provider_name: str
    generation_duration_s: float
    model_id: str = ""      # provider model identifier; empty = not reported
    is_noop: bool = False   # True when model signals change already present
    # The model's stated REASON for declining. Only meaningful when
    # `is_noop`. Carried because a refusal is an ANSWER: the trajectory
    # recorder persists it as the candidate body, and two refusals that
    # differ only in reasoning are two answers, not one.
    noop_reason: str = ""
    # L1: audit records from tool-use loop (empty when tools disabled)
    tool_execution_records: Tuple[Any, ...] = ()
    # Venom edit/write/delete audit trail captured from ToolExecutor at
    # tool_loop.run() exit. Each entry carries tool/path/action/before_hash/
    # after_hash/timestamp (see ToolExecutor._record_edit). Empty when no
    # mutating tool calls were issued.
    venom_edit_history: Tuple[Dict[str, Any], ...] = ()
    # Target files that the prompt builder embedded as file-content
    # regions *before* the tool loop ran. The lean prompt builder
    # in-lines ~100 lines of each target file into the initial prompt,
    # which is the semantic equivalent of the model having called
    # ``read_file`` on that path — the Iron Gate treats one entry here
    # as one unit of exploration credit so BACKGROUND-route DW ops
    # (which tend to emit patches directly without a tool round) are
    # not falsely tripped by ``exploration_insufficient``.
    prompt_preloaded_files: Tuple[str, ...] = ()
    # Token usage (0 = not reported by provider)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Cost in USD (0.0 = not reported by provider)
    cost_usd: float = 0.0

    def with_tool_records(self, records: Tuple[Any, ...]) -> "GenerationResult":
        """Return a new GenerationResult with tool_execution_records set (called by provider after tool loop)."""
        return dataclasses.replace(self, tool_execution_records=records)

    def with_venom_edits(self, edits: Tuple[Dict[str, Any], ...]) -> "GenerationResult":
        """Return a new GenerationResult carrying Venom's edit history.

        Called by providers right after ``tool_loop.run()`` completes, so
        the orchestrator ledger and SerpentFlow can surface every autonomous
        mutation Venom performed during generation.
        """
        return dataclasses.replace(self, venom_edit_history=edits)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of the validation phase.

    Parameters
    ----------
    passed:
        Whether validation passed.
    best_candidate:
        The winning candidate dict, or ``None`` if validation failed.
    validation_duration_s:
        Wall-clock seconds spent validating.
    error:
        Human-readable error string if validation failed.
    """

    passed: bool
    best_candidate: Optional[Dict[str, Any]]
    validation_duration_s: float
    error: Optional[str]
    # Phase 2A: compact provenance fields (full output goes to ledger, not here)
    failure_class: Optional[str] = None          # "test" | "build" | "infra" | "budget" | None
    short_summary: str = ""                      # ≤300 chars human-readable summary
    adapter_names_run: Tuple[str, ...] = ()      # e.g. ("python",) or ("python", "cpp")
    # High-resolution test-gate telemetry (Phase 2). The SPECIFIC assertion /
    # AST error the candidate died on — parsed by ``test_failure_digest`` from
    # the adapter's own output, not a blind stdout tail. Immutable fields so
    # ``ValidationResult`` stays hashable. Consumed by the cockpit VALIDATE
    # heartbeat (operator sees the cause live), the re-planner (the next
    # GENERATE prompt), and the execution ledger + GRPO recorder.
    failure_detail: str = ""                     # bounded structured digest
    failed_tests: Tuple[str, ...] = ()           # failing node ids
    test_total: int = 0                          # tests run
    test_failed: int = 0                         # tests failed


#: Cap on the summary handed to the re-planner. The field's own comment
#: promises "<=300 chars"; a promise nothing enforces is a promise, and an
#: unbounded string reaches a prompt.
REPLAN_SUMMARY_MAX_CHARS: int = 300


def replan_inputs(validation: Any) -> Tuple[str, str]:
    """``(failure_class, short_summary)`` for the dynamic re-planner.

    THE ONE READER, because there are two callers and they have already
    drifted once. `generate_runner.run()` — the SHIPPING path — referenced a
    bare `validation` that has no binding anywhere in its scope, guarded by
    `if 'validation' in dir()`. The guard therefore always evaluated False and
    re-planning ran on empty inputs on every real op, while the inline
    orchestrator twin read a real verdict. One helper, both call sites.

    NO NEW VERDICT TYPE. :class:`ValidationResult` is already frozen, already
    carries exactly these two fields, and is already carried on
    :class:`OperationContext` — `advance()` uses ``dataclasses.replace``, so a
    verdict set at VALIDATE survives into a GENERATE retry, which is precisely
    the "previous attempt's verdict" the re-planner wants. Defining a parallel
    dataclass would be a second spelling of a value that already exists.

    TOTAL BY CONSTRUCTION — every degenerate input maps to a defined state,
    because the re-planner must degrade rather than fly blind or raise:

    * ``None`` — no VALIDATE has run yet (the first GENERATE attempt, or an
      op that failed before validation). Returns ``("", "")``: the honest
      "no evidence" state, which `DynamicRePlanner` already tolerates.
    * **evaluator crashed / verdict never set** — indistinguishable from the
      above at this seam, and deliberately so. Both mean "no verdict
      exists"; inventing a third state would be claiming knowledge about WHY
      that this reader does not have.
    * **foreign or malformed object** — read by ``getattr`` with defaults, so
      a duck-typed stand-in (tests, a future verdict type) works and a
      missing attribute degrades instead of raising.
    * **non-string fields** — coerced with ``str()``. A failure_class that
      arrived as an enum or ``None`` must not put ``"None"`` into a prompt,
      so falsy values normalise to ``""``.
    * **oversized summary** — truncated to :data:`REPLAN_SUMMARY_MAX_CHARS`.

    NEVER raises."""
    if validation is None:
        return ("", "")
    try:
        raw_fc = getattr(validation, "failure_class", None)
        raw_em = getattr(validation, "short_summary", None)
    except Exception:  # noqa: BLE001 — a hostile property must not break GENERATE
        return ("", "")

    def _clean(value: Any, limit: Optional[int] = None) -> str:
        if not value:
            return ""
        try:
            out = str(value).strip()
        except Exception:  # noqa: BLE001
            return ""
        if limit is not None and len(out) > limit:
            out = out[:limit]
        return out

    return (_clean(raw_fc), _clean(raw_em, REPLAN_SUMMARY_MAX_CHARS))


@dataclass(frozen=True)
class ApprovalDecision:
    """Context-embedded approval decision.

    This is the version stored inside :class:`OperationContext`, separate
    from any provider-specific approval model.

    Parameters
    ----------
    status:
        One of ``"approved"``, ``"rejected"``, ``"pending"``, ``"expired"``.
    approver:
        Identifier of the human or system that made the decision.
    reason:
        Free-text justification.
    decided_at:
        Timestamp of the decision.
    request_id:
        Unique identifier for the approval request.
    """

    status: str
    approver: Optional[str]
    reason: Optional[str]
    decided_at: Optional[datetime]
    request_id: str


@dataclass(frozen=True)
class ShadowResult:
    """Outcome of a shadow-mode comparison run.

    Parameters
    ----------
    confidence:
        Float in ``[0, 1]`` representing structural match confidence.
    comparison_mode:
        Comparison strategy used (e.g. ``"structural"``, ``"exact"``).
    violations:
        Tuple of violation descriptions found during comparison.
    shadow_duration_s:
        Wall-clock seconds the shadow run took.
    production_match:
        Whether the shadow output matched the production output.
    disqualified:
        Whether the shadow candidate was disqualified from promotion.
    """

    confidence: float
    comparison_mode: str
    violations: Tuple[str, ...]
    shadow_duration_s: float
    production_match: bool
    disqualified: bool


# ---------------------------------------------------------------------------
# Attachment — bounded, redacted-by-hash image reference
# ---------------------------------------------------------------------------
#
# Substrate shared exclusively between VisionSensor and visual_verify.py per
# the I7 substrate export-ban invariant in
# docs/superpowers/specs/2026-04-18-vision-sensor-verify-design.md.
#
# Design invariants (all enforced at construction, all tested):
#   1. Frozen. No in-place mutation. Hash-chain safe.
#   2. kind / mime_type are hard whitelisted — no free-form strings.
#   3. hash8 is first 8 lowercase hex chars of sha256(bytes). Stable,
#      redaction-safe identifier that never leaks bytes to logs.
#   4. image_path validated absolute and length-capped.
#   5. __repr__ redacts the full path — only basename + hash8 escape.
#   6. from_file() is the canonical constructor; forbids forged
#      hash8/mime triples by computing both from the actual file.
#   7. read_bytes() enforces a size cap; read_bytes_verified() also
#      re-checks the hash8 → detects silent file rotation/corruption.
#   8. OperationContext caps at 8 attachments (MAX_PER_CTX).

_VALID_ATTACHMENT_KINDS: Tuple[str, ...] = (
    "pre_apply",     # visual_verify pre-APPLY capture
    "post_apply",    # visual_verify post-APPLY capture
    "sensor_frame",  # VisionSensor autonomous capture
    "user_provided", # Human-initiated /attach upload via SerpentFlow REPL (CC-parity)
)
_VALID_ATTACHMENT_MIMES: Tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",  # Anthropic Messages API document content block
)
_ATTACHMENT_HASH8_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_ATTACHMENT_MAX_IMAGE_BYTES_DEFAULT = 10 * 1024 * 1024   # 10 MiB per frame
_ATTACHMENT_MAX_PATH_LEN = 512
_ATTACHMENT_MAX_APP_ID_LEN = 256
_ATTACHMENT_MAX_PER_CTX = 8
_ATTACHMENT_EXT_TO_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True)
class Attachment:
    """Bounded, redacted-by-hash image reference on ``OperationContext``.

    Consumable only by ``VisionSensor`` and ``visual_verify.py`` per I7.
    Any other reader must go through a spec review (see
    ``tests/governance/test_attachment_export_ban.py``).

    Parameters
    ----------
    kind:
        One of ``{"pre_apply", "post_apply", "sensor_frame"}``.
        Hard whitelisted.
    image_path:
        Absolute local filesystem path. Validated at construction.
        Length-capped at ``_ATTACHMENT_MAX_PATH_LEN`` chars.
    mime_type:
        One of ``{"image/jpeg", "image/png", "image/webp"}``.
    hash8:
        First 8 lowercase hex chars of ``sha256(image_bytes)``. Redaction-
        safe identifier for deduplication, logging, and change detection.
    ts:
        Non-negative capture monotonic timestamp (``time.monotonic()``).
    app_id:
        Optional macOS bundle identifier (e.g. ``"com.apple.Terminal"``).
    """

    kind: str
    image_path: str
    mime_type: str
    hash8: str
    ts: float
    app_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_ATTACHMENT_KINDS:
            raise ValueError(
                f"Attachment.kind must be one of {_VALID_ATTACHMENT_KINDS}; "
                f"got {self.kind!r}"
            )
        if not isinstance(self.image_path, str) or not self.image_path:
            raise ValueError("Attachment.image_path must be a non-empty string")
        if not os.path.isabs(self.image_path):
            raise ValueError(
                f"Attachment.image_path must be absolute; got {self.image_path!r}"
            )
        if len(self.image_path) > _ATTACHMENT_MAX_PATH_LEN:
            raise ValueError(
                f"Attachment.image_path exceeds {_ATTACHMENT_MAX_PATH_LEN} chars "
                f"(got {len(self.image_path)})"
            )
        if self.mime_type not in _VALID_ATTACHMENT_MIMES:
            raise ValueError(
                f"Attachment.mime_type must be one of {_VALID_ATTACHMENT_MIMES}; "
                f"got {self.mime_type!r}"
            )
        if not isinstance(self.hash8, str) or not _ATTACHMENT_HASH8_PATTERN.match(self.hash8):
            raise ValueError(
                f"Attachment.hash8 must be exactly 8 lowercase hex chars; "
                f"got {self.hash8!r}"
            )
        if not isinstance(self.ts, (int, float)) or self.ts < 0:
            raise ValueError(f"Attachment.ts must be non-negative float; got {self.ts!r}")
        if self.app_id is not None:
            if not isinstance(self.app_id, str) or not self.app_id:
                raise ValueError("Attachment.app_id, if set, must be a non-empty string")
            if len(self.app_id) > _ATTACHMENT_MAX_APP_ID_LEN:
                raise ValueError(
                    f"Attachment.app_id exceeds {_ATTACHMENT_MAX_APP_ID_LEN} chars"
                )

    def __repr__(self) -> str:
        """Redaction-safe repr — never leaks the full ``image_path``."""
        basename = os.path.basename(self.image_path) if self.image_path else "<unset>"
        return (
            f"Attachment(kind={self.kind!r}, hash8={self.hash8!r}, "
            f"mime={self.mime_type!r}, app_id={self.app_id!r}, "
            f"path=<redacted:basename={basename!r}>)"
        )

    __str__ = __repr__

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        kind: str,
        app_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> "Attachment":
        """Canonical constructor — computes ``hash8`` and infers ``mime_type``.

        Callers cannot forge a mismatched ``(path, mime_type, hash8)``
        triple — both fields are derived from the actual file.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``path`` is not absolute, the extension is unsupported,
            or any ``Attachment`` invariant fails.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("Attachment.from_file requires a non-empty string path")
        if not os.path.isabs(path):
            raise ValueError(f"Attachment.from_file requires absolute path; got {path!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attachment image missing: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in _ATTACHMENT_EXT_TO_MIME:
            raise ValueError(
                f"Attachment.from_file unsupported extension {ext!r}; "
                f"must be one of {sorted(_ATTACHMENT_EXT_TO_MIME.keys())}"
            )
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        return cls(
            kind=kind,
            image_path=path,
            mime_type=_ATTACHMENT_EXT_TO_MIME[ext],
            hash8=digest[:8],
            ts=ts if ts is not None else time.monotonic(),
            app_id=app_id,
        )

    def read_bytes(self, *, max_bytes: Optional[int] = None) -> bytes:
        """Read the underlying image bytes with an enforced size cap.

        Raises
        ------
        FileNotFoundError
            If the image has been removed since construction.
        ValueError
            If the file size exceeds ``max_bytes``.
        """
        cap = max_bytes if max_bytes is not None else _ATTACHMENT_MAX_IMAGE_BYTES_DEFAULT
        if cap <= 0:
            raise ValueError(f"max_bytes must be positive; got {cap}")
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Attachment image disappeared: {self.image_path}")
        size = os.path.getsize(self.image_path)
        if size > cap:
            raise ValueError(
                f"Attachment {self.hash8} exceeds cap: size={size} > max_bytes={cap}"
            )
        with open(self.image_path, "rb") as fh:
            return fh.read()

    def read_bytes_verified(self, *, max_bytes: Optional[int] = None) -> bytes:
        """Read bytes and verify ``hash8`` still matches the on-disk content.

        Raises
        ------
        ValueError
            If on-disk bytes no longer match the captured ``hash8``.
        """
        data = self.read_bytes(max_bytes=max_bytes)
        actual = hashlib.sha256(data).hexdigest()[:8]
        if actual != self.hash8:
            raise ValueError(
                f"Attachment integrity check failed: expected hash8={self.hash8}, "
                f"on-disk hash8={actual}. File changed since construction."
            )
        return data


# ---------------------------------------------------------------------------
# Saga Types
# ---------------------------------------------------------------------------


class SagaStepStatus(str, Enum):
    """Per-repo lifecycle status inside a multi-repo saga."""

    PENDING = "pending"
    APPLYING = "applying"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


@dataclass(frozen=True)
class RepoSagaStatus:
    """Frozen per-repo status entry in a multi-repo saga."""

    repo: str
    status: SagaStepStatus
    attempt: int = 0
    last_error: str = ""
    reason_code: str = ""
    compensation_attempted: bool = False


# ---------------------------------------------------------------------------
# Telemetry Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostTelemetry:
    """Snapshot of local hardware state at operation intake."""

    schema_version: str           # "1.0"
    arch: str                     # platform.machine() → "arm64"
    cpu_percent: float            # quantized to 2dp
    ram_available_gb: float       # quantized to 2dp
    pressure: str                 # PressureLevel.name: "NORMAL"|"ELEVATED"|"CRITICAL"|"EMERGENCY"
    sampled_at_utc: str           # datetime.now(utc).isoformat()
    sampled_monotonic_ns: int     # time.monotonic_ns() at sample time
    collector_status: str         # "ok" | "partial" | "stale"
    sample_age_ms: int            # (now_ns - sampled_monotonic_ns) // 1_000_000


@dataclass(frozen=True)
class RoutingIntentTelemetry:
    """Routing decision EXPECTED at FSM intake (before any execution)."""

    expected_provider: str        # e.g. "GCP_PRIME_SPOT", "LOCAL_CLAUDE"
    policy_reason: str            # e.g. "PRIMARY_AVAILABLE", "NORMAL"
    # Brain selector causal fields (Phase 4 — default "" for backwards compat)
    brain_id: str = ""            # e.g. "qwen_coder_32b", "phi3_lightweight"
    brain_model: str = ""         # exact model name passed to j-prime
    routing_reason: str = ""      # causal code: "task_gate_trivial", "cost_gate_triggered_queue"
    task_complexity: str = ""     # "trivial" | "light" | "heavy_code" | "complex"
    estimated_prompt_tokens: int = 0
    daily_spend_usd: float = 0.0  # snapshot of daily spend at intake
    schema_capability: str = "full_content_only"  # "full_content_only" | "full_content_and_diff"
    # Urgency-aware provider routing (Phase 5)
    provider_route: str = ""      # "immediate" | "standard" | "complex" | "background" | "speculative"
    provider_route_reason: str = ""  # causal code from UrgencyRouter


@dataclass(frozen=True)
class RoutingActualTelemetry:
    """Routing outcome AFTER execution (stamped at COMPLETE or POSTMORTEM)."""

    provider_name: str
    endpoint_class: str           # "gcp_spot" | "local" | "cloud_api"
    fallback_chain: Tuple[str, ...]
    was_degraded: bool


@dataclass(frozen=True)
class TelemetryContext:
    """Root telemetry envelope stamped once at intake, updated once at completion."""

    local_node: HostTelemetry
    routing_intent: RoutingIntentTelemetry
    routing_actual: Optional[RoutingActualTelemetry] = None


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------


def _compute_hash(ctx_dict: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hex digest of *ctx_dict*.

    Keys are sorted and non-serialisable values are coerced to ``str``
    via ``json.dumps(..., sort_keys=True, default=str)``.

    Parameters
    ----------
    ctx_dict:
        Dictionary of context fields to hash.

    Returns
    -------
    str
        64-character lowercase hex string (SHA-256).
    """
    canonical = json.dumps(ctx_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_dag(edges: Tuple[Tuple[str, str], ...]) -> None:
    """Kahn's algorithm cycle detection. Raises ArchitecturalCycleError if cycle found."""
    if not edges:
        return
    graph: Dict[str, list] = collections.defaultdict(list)
    in_degree: Dict[str, int] = collections.defaultdict(int)
    nodes: set = set()
    for src, dst in edges:
        graph[src].append(dst)
        in_degree[dst] += 1
        nodes.add(src)
        nodes.add(dst)
    queue = collections.deque(n for n in nodes if in_degree[n] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if visited < len(nodes):
        cycle_nodes = [n for n in nodes if in_degree[n] > 0]
        raise ArchitecturalCycleError(
            f"Cycle detected in dependency_edges involving repos: {sorted(cycle_nodes)}"
        )


# ---------------------------------------------------------------------------
# Slice 89 — ExplorationManifest (DW→Claude neural handoff mesh)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplorationManifest:
    """Structured summary of DW's tool-loop exploration, carried forward to Claude.

    Built in the CandidateGenerator DW-failure window and stamped onto
    OperationContext via ``with_exploration_manifest()`` BEFORE the Claude
    fallback ``generate()`` call.  Claude's prompt builders inject this as
    an observation block ("## Prior Exploration (DoubleWord)") so Claude
    can compile the patch without repeating directory scans and initial
    searches that DW already completed.

    Authority-free — it is an observation block, not a hard constraint.
    Claude's tool autonomy is preserved.

    Fields
    ------
    verified_target_files:
        File paths that DW successfully read, edited, or wrote.  These have
        been pre-localized — Claude should analyse them directly.
    high_signal_search_tokens:
        Query/pattern strings from successful ``search_code`` calls.
        Represents what DW found relevant to the task.
    failed_test_commands:
        Commands from ``run_tests`` calls that FAILED.  Claude should know
        these tests were already attempted and broke.
    tool_call_count:
        Total number of tool calls DW made (``len(records)``).
    exploration_reason:
        Why the manifest was built (e.g. ``"dw_failure"``).
    schema_version:
        Manifest schema version — bump when fields are added/removed.

    Slice 89 markers are ``# Slice 89`` comments at call sites.
    """

    verified_target_files: Tuple[str, ...]
    high_signal_search_tokens: Tuple[str, ...]
    failed_test_commands: Tuple[str, ...]
    tool_call_count: int
    exploration_reason: str
    schema_version: str = "manifest.1"  # Slice 89

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation suitable for serialisation."""
        return {
            "schema_version": self.schema_version,
            "verified_target_files": self.verified_target_files,
            "high_signal_search_tokens": self.high_signal_search_tokens,
            "failed_test_commands": self.failed_test_commands,
            "tool_call_count": self.tool_call_count,
            "exploration_reason": self.exploration_reason,
        }

    @classmethod
    def from_telemetry(
        cls,
        records: Any,
        salient_args: Any,
        reason: str,
    ) -> "ExplorationManifest":
        """Build a manifest from ToolExecutionRecord list + salient-arg list.

        Never raises — any parsing failure returns a valid empty manifest.

        Parameters
        ----------
        records:
            List of ToolExecutionRecord objects from the DW tool loop.
        salient_args:
            Parallel list of ``(tool_name, salient_arg)`` tuples captured
            out-of-band by ToolLoopCoordinator at dispatch time.
        reason:
            Human-readable reason for building the manifest.

        Derivation rules
        ----------------
        * ``verified_target_files`` — salient_arg from SUCCESS read_file /
          edit_file / write_file calls (deduplicated, non-empty).
        * ``high_signal_search_tokens`` — salient_arg from SUCCESS search_code
          calls (deduplicated, non-empty).
        * ``failed_test_commands`` — salient_arg from non-SUCCESS run_tests
          calls (deduplicated, non-empty).
        """
        try:
            if records is None:
                records = []
            if salient_args is None:
                salient_args = []

            # Safely zip — use min length to avoid IndexError
            _records = list(records) if records else []
            _salient = list(salient_args) if salient_args else []
            pairs = list(zip(_records, _salient))

            _target_files: list = []
            _search_tokens: list = []
            _failed_tests: list = []

            _READ_WRITE_TOOLS = frozenset({"read_file", "edit_file", "write_file"})
            _SEARCH_TOOLS = frozenset({"search_code"})
            _TEST_TOOLS = frozenset({"run_tests"})

            for rec, (tool_name, salient_arg) in pairs:
                try:
                    status_val = getattr(rec, "status", None)
                    # Normalize status to string for comparison
                    if hasattr(status_val, "value"):
                        status_str = str(status_val.value)
                    else:
                        status_str = str(status_val) if status_val is not None else ""
                    is_success = (status_str == "success")
                    salient_arg = str(salient_arg) if salient_arg is not None else ""
                    if not salient_arg:
                        continue

                    if tool_name in _READ_WRITE_TOOLS and is_success:
                        if salient_arg not in _target_files:
                            _target_files.append(salient_arg)
                    elif tool_name in _SEARCH_TOOLS and is_success:
                        if salient_arg not in _search_tokens:
                            _search_tokens.append(salient_arg)
                    elif tool_name in _TEST_TOOLS and not is_success:
                        if salient_arg not in _failed_tests:
                            _failed_tests.append(salient_arg)
                except Exception:  # noqa: BLE001 — never break build
                    continue

            return cls(
                verified_target_files=tuple(_target_files),
                high_signal_search_tokens=tuple(_search_tokens),
                failed_test_commands=tuple(_failed_tests),
                tool_call_count=len(_records),
                exploration_reason=reason,
            )
        except Exception:  # noqa: BLE001 — Slice 89 never-raises contract
            return cls(
                verified_target_files=(),
                high_signal_search_tokens=(),
                failed_test_commands=(),
                tool_call_count=0,
                exploration_reason=reason,
            )


# ---------------------------------------------------------------------------
# OperationContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationContext:
    """Frozen, hash-chained state object for an Ouroboros pipeline run.

    All mutations go through :meth:`advance` which returns a **new** instance
    with an updated phase, timestamp, and cryptographic hash chain.

    Parameters
    ----------
    op_id:
        Globally unique, time-sortable operation identifier.
    created_at:
        Timestamp when the operation was first created.
    phase:
        Current pipeline phase.
    phase_entered_at:
        Timestamp when the current phase was entered.
    context_hash:
        SHA-256 hex of all fields (except ``context_hash`` itself).
    previous_hash:
        Hash of the predecessor context (``None`` for the initial state).
    target_files:
        Tuple of file paths this operation targets.
    risk_tier:
        Assigned risk tier (set after classification).
    description:
        Human-readable description of the operation.
    routing:
        Routing decision (set after routing phase).
    approval:
        Approval decision (set after approval phase).
    shadow:
        Shadow-mode comparison result.
    generation:
        Candidate generation result.
    validation:
        Validation result.
    policy_version:
        Version of the governance policy in effect.
    side_effects_blocked:
        Whether side effects (writes, network calls) are blocked.
    """

    op_id: str
    created_at: datetime
    phase: OperationPhase
    phase_entered_at: datetime
    context_hash: str
    previous_hash: Optional[str]
    target_files: Tuple[str, ...]
    # Slice 167 — IMMUTABLE targets declared at inception (intake/create). The
    # generation phase may overwrite ``target_files`` with the candidate's files; this
    # preserves the originally-declared targets so the governance floor can never be
    # blinded by a mid-pipeline overwrite. Set once in create(); carried by every
    # ``with_*`` (dataclasses.replace preserves it) and never re-derived.
    declared_targets: Tuple[str, ...] = ()
    risk_tier: Optional[RiskTier] = None
    description: str = ""
    # Slice 235 — per-op force-full override. Set by the orchestrator fail-safe
    # on a GENERATE_RETRY after a 2b.1-diff candidate failed to apply (stale /
    # ambiguous context): the retry forces full_content regardless of model
    # capability / file size, so a botched diff degrades cleanly instead of
    # re-emitting a diff that fails again. Default False = adaptive (Slice 235).
    force_full_content_override: bool = False
    routing: Optional[RoutingDecision] = None
    approval: Optional[ApprovalDecision] = None
    shadow: Optional[ShadowResult] = None
    generation: Optional[GenerationResult] = None
    validation: Optional[ValidationResult] = None
    policy_version: str = ""
    side_effects_blocked: bool = True
    pipeline_deadline: Optional[datetime] = None  # stamped once at submit(); phases compute remaining budget

    # ---- Phase 3: Multi-repo saga fields ----
    primary_repo: str = "jarvis"
    repo_scope: Tuple[str, ...] = ("jarvis",)
    cross_repo: bool = dataclasses.field(default=False, init=False)
    dependency_edges: Tuple[Tuple[str, str], ...] = ()
    apply_plan: Tuple[str, ...] = ()
    repo_snapshots: Tuple[Tuple[str, str], ...] = ()
    saga_id: str = ""
    saga_state: Tuple[RepoSagaStatus, ...] = ()
    schema_version: str = "3.0"
    expanded_context_files: Tuple[str, ...] = ()
    benchmark_result: Optional["BenchmarkResult"] = None
    pre_apply_snapshots: Dict[str, str] = field(default_factory=dict)
    execution_graph_id: str = ""
    execution_plan_digest: str = ""
    subagent_count: int = 0
    parallelism_budget: int = 0
    causal_trace_id: str = ""
    strategic_intent_id: str = ""
    strategic_memory_fact_ids: Tuple[str, ...] = ()
    strategic_memory_prompt: str = ""
    strategic_memory_digest: str = ""
    # Sovereign Cross-Repo Mutator G1 (2026-06-23): the Oracle-traced cross-repo
    # blast-radius prompt block, stamped at GENERATE when ctx.cross_repo and the
    # master arming switch is ON. Empty (default) -> byte-identical legacy.
    cross_repo_blast_prompt: str = ""
    terminal_reason_code: str = ""
    # Slice 160 — non-fatal infrastructure-hook warning (e.g. pip install failed under
    # fail-soft). Carried forward so the op reaches the governance floor / COMPLETE and
    # the operator sees it (Discord embed / telemetry) rather than the op dying.
    infra_warning: str = ""
    # Sovereign Transport Profiler Matrix (2026-06-20) — stamped True BEFORE the
    # budget layer when the resolved DW model is known batch-only (RT yields
    # done_before_content; learned + immortal in dw_transport_profile). Routes the
    # op to the extended batch budget, grants Zero-Shot quarantine immunity, and
    # triggers active park-detachment. Default False → byte-identical legacy.
    async_batch_payload: bool = False
    rollback_occurred: bool = False
    # P2-6: Runbook-grade observability — cross-operation correlation identifier.
    # For single-repo ops: defaults to op_id (self-referential).
    # For multi-repo sagas: all saga-member ops share the root op's correlation_id.
    correlation_id: str = ""

    # ---- Telemetry (stamped at intake and COMPLETE) ----
    telemetry: Optional[TelemetryContext] = None
    previous_op_hash_by_scope: Tuple[Tuple[str, str], ...] = ()
    # e.g. (("jarvis", "abc123..."), ("prime", "def456..."))
    # Frozen-safe representation of Dict[repo_name, last_context_hash]

    # ---- Autonomy tier frozen at submit() — gate reads this, never re-queries TrustGraduator ----
    frozen_autonomy_tier: str = "governed"  # "governed" | "observe"; default = backward compat

    # ---- Human-authored instructions from OUROBOROS.md hierarchy (injected at submit time) ----
    human_instructions: str = ""  # injected from OUROBOROS.md hierarchy at submit time

    # ---- Cumulative session intelligence (injected before GENERATE) ----
    session_lessons: str = ""  # ephemeral lessons from prior ops in this session

    # ---- Signal metadata (propagated from IntentEnvelope at intake) ----
    signal_urgency: str = ""   # "critical" | "high" | "normal" | "low"
    signal_source: str = ""    # "test_failure" | "voice_human" | "ai_miner" | etc.
    compute_context: str = ""  # "" | "Local_Open_Source" (free local Ollama) -- bypasses the financial preflight
    # JSON-encoded snapshot of the originating envelope's
    # ``evidence`` dict. Frozen-friendly (string, no Mapping
    # ordering issues). Default empty string when no envelope (op
    # constructed manually) or when envelope.evidence was empty.
    # Consumers (Slice 4 ClusterIntelligence-CrossSession cascade
    # observer; future arcs that need typed signal context) parse
    # via ``json.loads`` defensively. Bounded by intake-side
    # caller; orchestrator does not re-validate.
    intake_evidence_json: str = ""

    # ---- Complexity classification (stamped at CLASSIFY by ComplexityClassifier) ----
    task_complexity: str = ""  # "trivial" | "simple" | "light" | "heavy_code" | "complex"

    # ---- Provider routing (stamped at ROUTE by UrgencyRouter) ----
    # Determines which provider strategy CandidateGenerator uses.
    provider_route: str = ""   # "immediate" | "standard" | "complex" | "background" | "speculative"
    provider_route_reason: str = ""  # human-readable reason for telemetry
    prefer_local: bool = False    # Quota Shield: route this op to the local J-Prime tier
    # Sovereign Failover Mesh (Gap 3b): an explicit provider pin stamped when an
    # op is sealed into the Cryo-DLQ on a DW global outage. On replay,
    # CandidateGenerator HONORS this override and routes the op straight to the
    # named provider (e.g. "gcp-jprime", the awakened J-Prime node) instead of
    # re-cascading through the dead DW lane. Fail-CLOSED: if the named provider
    # is unavailable, the op stays sealed in the DLQ (never routed to dead DW).
    provider_override: str = ""   # "" = no override (legacy cascade); "gcp-jprime" = pin to J-Prime

    # ---- Truncation-retry shape flags (Task 2, feat/jprime-truncation-retry-diff-shape) ----
    # Stamped by the orchestrator GENERATE_RETRY path when a truncation/elision
    # failure is detected (JARVIS_TRUNCATION_RETRY_ENABLED=true). Consumed by
    # the provider tier to change output shape on the next attempt instead of
    # retrying with the same parameters.
    force_diff_on_retry: bool = False        # Truncation-retry: force 2b.1-diff output on this retry
    retry_max_tokens_override: int = 0        # Truncation-retry: >0 overrides provider max_tokens for this retry

    # Syntax-retry feedback. Same lifecycle as the two flags above — stamped
    # when the previous attempt's candidates were rejected by ``ast.parse``,
    # consumed by the provider tier to change the next attempt rather than
    # repeating it unchanged.
    #
    # It carries the actual parse errors (file, line, message, offending
    # source line). Before this the SyntaxError was logged and discarded, so
    # `all_candidates_syntax_error` was terminal on a lane with nowhere to
    # escalate to: the local 32B IS J-Prime, so `syntax_escalation`'s
    # DW → J-Prime cascade escalates to the model that just failed. A model
    # cannot correct an error it is never shown.
    syntax_retry_feedback: str = ""

    # ---- Slice 245 — hibernation resurrection flag ----
    # Set (via with_resurrection()) when an op that failed with provider
    # exhaustion during a dark window is re-ingested after the Grid Sentinel
    # confirms recovery. Consumed by _compute_priority (intake) and
    # BackgroundAgentPool.submit() (pool) to grant Absolute-Max Primacy so the
    # survivor supersedes everything that accumulated while the grid was dark.
    # The flag travels ON the preserved context — durable partial state (phase,
    # already-generated candidates, plan) rides along; no completed work is
    # re-computed.
    resurrected_from_hibernation: bool = False

    # ---- Dependency intelligence from Oracle graph (injected at CONTEXT_EXPANSION) ----
    # ~200-token summary: direct dependents, transitive importers, blast radius.
    # Prevents breaking downstream consumers that import the target files.
    dependency_summary: str = ""

    # ---- Stale-exploration guard: file hashes captured at GENERATE ----
    # Tuple of (filepath, sha256_hex) pairs snapshotted when GENERATE begins.
    # Compared at APPLY time — if any hash differs, the file was modified by
    # a concurrent operation and the candidate is stale.
    generate_file_hashes: Tuple[Tuple[str, str], ...] = ()

    # ---- Gate 1: Isomorphic Execution Lock (pre-write L4 proof chain) ----
    # Set at APPLY-start by the Gate 1 block in slice4b_runner when
    # JARVIS_A1_SANDBOX_LOCK_ENABLED=true.  Both fields are None (default-OFF)
    # on every path that does not activate the gate — zero behavioural change.
    proof_chain: object = None    # DAGProofChain (per-op accumulator)
    sandbox_token: object = None  # SandboxExecutionToken minted at Gate 1
    blast_token: object = None    # BlastRadiusClearedToken minted at Gate 2

    # ---- Model-reasoned implementation plan (stamped at PLAN phase) ----
    # Structured JSON plan produced by PlanGenerator before GENERATE.
    # Contains: approach, file_changes (ordered with dependencies), risk_factors,
    # test_strategy, complexity estimate. Injected into GENERATE prompt so the
    # model follows a coherent strategy instead of ad-hoc patching.
    implementation_plan: str = ""

    # ---- PLAN-subagent DAG output (stamped post-PlanGenerator when
    # JARVIS_PLAN_SUBAGENT_SHADOW=true) ----
    # execution_graph 2d.1-shaped payload (tuple-of-tuple form so the
    # dataclass stays hashable/comparable). Produced by
    # AgenticPlanSubagent and stashed here by
    # ``orchestrator._run_plan_shadow`` as an observer signal —
    # **does NOT overwrite implementation_plan**. Present so downstream
    # telemetry + future GENERATE hooks can compare the legacy flat-list
    # plan against the DAG without a separate lookup.
    #
    # Shape (when set):
    #   ((("schema_version", "2d.1"),
    #     ("graph_id", "<hash>"),
    #     ("planner_id", "AgenticPlanSubagent/deterministic"),
    #     ("concurrency_limit", N),
    #     ("units", (
    #         (("unit_id", ...), ("dependency_ids", (...)),
    #          ("owned_paths", (...)), ("acceptance_tests", (...)),
    #          ("barrier_id", "")),
    #         ...
    #     ))))
    # ``None`` = shadow hook was not invoked (flag off, single-file op,
    # or dispatch failed). Consumers must handle the None case.
    execution_graph: Optional[Any] = None

    # ---- Reasoning chain result (stamped at CLASSIFY if chain is active) ----
    reasoning_chain_result: Optional[Dict[str, Any]] = None

    # ---- Read-only intent (stamped pre-CLASSIFY by orchestrator) ----
    # When True, the op is a cartography/analysis task — no mutating tool
    # calls permitted and the APPLY phase is short-circuited. The flag is
    # trusted by OperationAdvisor to bypass blast_radius + test_coverage
    # blocks ONLY because tool_executor + orchestrator jointly enforce the
    # no-mutation contract (Manifesto §1 Boundary Principle: friction at
    # a threshold rests on deterministic enforcement, not a label).
    is_read_only: bool = False

    # ---- Image attachments (I7 substrate; Vision-phase only) ----
    # Consumable only by VisionSensor and visual_verify.py. All other
    # readers are forbidden by tests/governance/test_attachment_export_ban.py
    # and providers._serialize_attachments(purpose=) gate. Capped at
    # _ATTACHMENT_MAX_PER_CTX entries. Default empty tuple — non-vision
    # ops traverse the pipeline exactly as before.
    attachments: Tuple[Attachment, ...] = ()

    # ---- Slice 89: ExplorationManifest (DW→Claude handoff context) ----
    # Stamped by CandidateGenerator in the DW-failure window (before
    # _fallback.generate()) when JARVIS_EXPLORATION_MANIFEST_ENABLED=true.
    # Carries DW's pre-localized file list and search findings so Claude
    # skips redundant re-exploration. Authority-free (observation block).
    # Default None — non-handoff ops are byte-identical to today.
    exploration_manifest: Optional["ExplorationManifest"] = None  # Slice 89

    # ---- Sovereign Epistemological Context Matrix (spec 5.1): bounded, hash- ----
    # validated candidate DAG from the DAG Router; seeds Venom + the Governor's
    # round-0 baseline. Empty tuple = no prefetch (light op / oracle cold / off).
    prefetch_manifest: Tuple["PrefetchEntry", ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Post-init
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "cross_repo", len(self.repo_scope) > 1)
        _validate_dag(self.dependency_edges)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        target_files: Tuple[str, ...],
        description: str,
        op_id: Optional[str] = None,
        policy_version: str = "",
        pipeline_deadline: Optional[datetime] = None,
        _timestamp: Optional[datetime] = None,
        primary_repo: str = "jarvis",
        repo_scope: Optional[Tuple[str, ...]] = None,
        dependency_edges: Tuple[Tuple[str, str], ...] = (),
        apply_plan: Tuple[str, ...] = (),
        repo_snapshots: Tuple[Tuple[str, str], ...] = (),
        saga_id: str = "",
        saga_state: Tuple[RepoSagaStatus, ...] = (),
        schema_version: str = "3.0",
        previous_op_hash_by_scope: Tuple[Tuple[str, str], ...] = (),
        correlation_id: str = "",
        signal_urgency: str = "",
        signal_source: str = "",
        intake_evidence_json: str = "",
        is_read_only: bool = False,
        attachments: Tuple[Attachment, ...] = (),
        # F2 Slice 2 — optional pre-stamped provider route + reason.
        # Default "" = unset (normal ROUTE-phase decides). When the
        # UnifiedIntakeRouter observes a ``routing_override`` on the
        # inbound envelope, it stamps these fields on ctx at creation
        # time so UrgencyRouter can honor the hint. Values intentionally
        # unvalidated here — UrgencyRouter re-validates against its
        # ProviderRoute enum before consuming.
        provider_route: str = "",
        provider_route_reason: str = "",
    ) -> OperationContext:
        """Create an initial CLASSIFY-phase context.

        Parameters
        ----------
        target_files:
            Tuple of file paths this operation targets.
        description:
            Human-readable description of the operation.
        op_id:
            Optional explicit operation ID; generated if omitted.
        policy_version:
            Version of the governance policy in effect.
        _timestamp:
            Optional explicit timestamp for deterministic tests.

        Returns
        -------
        OperationContext
            A new context in the CLASSIFY phase with a computed hash.
        """
        now = _timestamp or datetime.now(tz=timezone.utc)
        resolved_op_id = op_id or generate_operation_id()
        resolved_repo_scope = repo_scope if repo_scope is not None else (primary_repo,)
        # P2-6: default correlation_id to op_id for single-repo ops
        resolved_correlation_id = correlation_id or resolved_op_id

        # Build a temporary dict of all fields (except context_hash) for hashing
        fields_for_hash: Dict[str, Any] = {
            "op_id": resolved_op_id,
            "created_at": now,
            "phase": OperationPhase.CLASSIFY.name,
            "phase_entered_at": now,
            "previous_hash": None,
            "target_files": target_files,
            "declared_targets": target_files,  # Slice 167 — immutable inception targets
            "risk_tier": None,
            "description": description,
            "routing": None,
            "approval": None,
            "shadow": None,
            "generation": None,
            "validation": None,
            "policy_version": policy_version,
            "side_effects_blocked": True,
            "pipeline_deadline": pipeline_deadline,
            "primary_repo": primary_repo,
            "repo_scope": resolved_repo_scope,
            "cross_repo": len(resolved_repo_scope) > 1,
            "dependency_edges": dependency_edges,
            "apply_plan": apply_plan,
            "repo_snapshots": repo_snapshots,
            "saga_id": saga_id,
            "saga_state": saga_state,
            "schema_version": schema_version,
            "expanded_context_files": (),
            "benchmark_result": None,
            "pre_apply_snapshots": {},
            "execution_graph_id": "",
            "execution_plan_digest": "",
            "subagent_count": 0,
            "parallelism_budget": 0,
            "causal_trace_id": "",
            "strategic_intent_id": "",
            "strategic_memory_fact_ids": (),
            "strategic_memory_prompt": "",
            "strategic_memory_digest": "",
            "terminal_reason_code": "",
            "rollback_occurred": False,
            "correlation_id": resolved_correlation_id,
            "telemetry": None,
            "previous_op_hash_by_scope": previous_op_hash_by_scope,
            "frozen_autonomy_tier": "governed",
            "reasoning_chain_result": None,
            "signal_urgency": signal_urgency,
            "signal_source": signal_source,
            "intake_evidence_json": intake_evidence_json,
            "task_complexity": "",
            "provider_route": provider_route,
            "provider_route_reason": provider_route_reason,
            "is_read_only": is_read_only,
            "attachments": attachments,
        }
        if len(attachments) > _ATTACHMENT_MAX_PER_CTX:
            raise ValueError(
                f"OperationContext.create: at most {_ATTACHMENT_MAX_PER_CTX} "
                f"attachments per context; got {len(attachments)}"
            )
        context_hash = _compute_hash(fields_for_hash)

        return cls(
            op_id=resolved_op_id,
            created_at=now,
            phase=OperationPhase.CLASSIFY,
            phase_entered_at=now,
            context_hash=context_hash,
            previous_hash=None,
            target_files=target_files,
            declared_targets=target_files,  # Slice 167 — immutable inception targets
            risk_tier=None,
            description=description,
            routing=None,
            approval=None,
            shadow=None,
            generation=None,
            validation=None,
            policy_version=policy_version,
            side_effects_blocked=True,
            pipeline_deadline=pipeline_deadline,
            primary_repo=primary_repo,
            repo_scope=resolved_repo_scope,
            dependency_edges=dependency_edges,
            apply_plan=apply_plan,
            repo_snapshots=repo_snapshots,
            saga_id=saga_id,
            saga_state=saga_state,
            schema_version=schema_version,
            strategic_intent_id="",
            strategic_memory_fact_ids=(),
            strategic_memory_prompt="",
            strategic_memory_digest="",
            terminal_reason_code="",
            rollback_occurred=False,
            correlation_id=resolved_correlation_id,
            previous_op_hash_by_scope=previous_op_hash_by_scope,
            frozen_autonomy_tier="governed",
            signal_urgency=signal_urgency,
            signal_source=signal_source,
            intake_evidence_json=intake_evidence_json,
            provider_route=provider_route,
            provider_route_reason=provider_route_reason,
            is_read_only=is_read_only,
            attachments=attachments,
        )

    # ------------------------------------------------------------------
    # State Machine Transition
    # ------------------------------------------------------------------

    def advance(
        self,
        new_phase: OperationPhase,
        _timestamp: Optional[datetime] = None,
        **updates: Any,
    ) -> OperationContext:
        """Transition to *new_phase*, returning a new context instance.

        Validates that the transition is legal according to
        :data:`PHASE_TRANSITIONS`, then produces a new frozen instance with:

        - ``phase`` set to *new_phase*
        - ``phase_entered_at`` set to now (or *_timestamp* for deterministic tests)
        - ``previous_hash`` set to ``self.context_hash``
        - ``context_hash`` recomputed over all fields
        - Any keyword arguments in *updates* applied via ``dataclasses.replace``

        Parameters
        ----------
        new_phase:
            The target phase.
        _timestamp:
            Optional explicit timestamp for deterministic tests.
        **updates:
            Additional field updates to apply (e.g. ``risk_tier=RiskTier.SAFE_AUTO``).

        Returns
        -------
        OperationContext
            A new context instance in *new_phase*.

        Raises
        ------
        ValueError
            If the transition from ``self.phase`` to *new_phase* is not allowed.
        """
        allowed = PHASE_TRANSITIONS.get(self.phase, set())
        if new_phase not in allowed:
            raise ValueError(
                f"Illegal phase transition: {self.phase.name} -> {new_phase.name}. "
                f"Allowed targets from {self.phase.name}: "
                f"{sorted(p.name for p in allowed) if allowed else '(terminal)'}"
            )

        now = _timestamp or datetime.now(tz=timezone.utc)

        # Build the replacement dict
        replacements: Dict[str, Any] = {
            "phase": new_phase,
            "phase_entered_at": now,
            "previous_hash": self.context_hash,
            **updates,
        }

        # Create intermediate instance without final hash
        # We need to compute hash over the new state, so build the dict first
        intermediate = dataclasses.replace(
            self,
            context_hash="",  # placeholder
            **replacements,
        )

        # Compute hash over all fields except context_hash
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)

        # Final instance with correct hash
        final = dataclasses.replace(intermediate, context_hash=new_hash)

        # Lifecycle event — fire AFTER the transition is fully built + validated,
        # so observers (the in-flight registry's phase mirror) see the committed
        # new phase. Fail-soft: never lets an observer break the transition.
        _notify_phase_transition(getattr(self, "op_id", "") or "", new_phase.name)

        # Predictive Phase-Aware Checkpointing — feed the just-COMPLETED phase's
        # wall-clock duration into the per-phase EWMA at the ONE canonical
        # transition choke (single source, all phases, no double-count). `self`
        # is the pre-transition instance, so `self.phase` is the phase being LEFT
        # and `self.phase_entered_at` its start. Lazy import + fully guarded so
        # this can never break a transition (Watchdog Isolation stays intact —
        # this only RECORDS latency; the suspend decision lives orchestrator-side).
        try:
            _entered = getattr(self, "phase_entered_at", None)
            if isinstance(_entered, datetime):
                _dur = (now - _entered).total_seconds()
                if _dur >= 0.0:
                    from backend.core.ouroboros.governance.phase_runway_gate import (  # noqa: PLC0415
                        record_phase_duration as _record_phase_duration,
                    )
                    _record_phase_duration(self.phase.name, _dur)
        except Exception:  # noqa: BLE001 — never let telemetry break a transition
            pass

        return final

    def with_compute_context(self, compute_context: str) -> "OperationContext":
        """Return a new context tagged with a compute_context.

        ``"Local_Open_Source"`` marks an op as free local open-source compute
        (e.g. local Ollama, $0.00), which bypasses the financial preflight in
        :func:`session_budget_authority.check_preflight`. Mirrors the existing
        ``with_*`` dataclasses.replace idiom (no hash-chain rewrite needed --
        this is a pre-pipeline tag set alongside signal metadata).
        """
        import dataclasses as _dc
        return _dc.replace(self, compute_context=compute_context)

    def with_pipeline_deadline(self, deadline: "datetime") -> "OperationContext":
        """Return a new context with pipeline_deadline set (no phase change).

        Uses the same hash-chain update as advance() but does not validate a
        phase transition. Called exactly once by GovernedLoopService.submit()
        before handing ctx to the orchestrator.
        """
        # Create intermediate with updated deadline and previous_hash chain
        intermediate = dataclasses.replace(
            self,
            pipeline_deadline=deadline,
            previous_hash=self.context_hash,
            context_hash="",  # placeholder — will be recomputed below
        )

        # Compute hash over all fields except context_hash
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)

        # Final instance with correct hash
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def effective_targets(self) -> Tuple[str, ...]:
        """Slice 167 — the UNION of the immutable declared targets (set at inception)
        and the current ``target_files`` (which the generation phase may have
        overwritten with the candidate's files), order-preserving + deduped.

        The governance floor evaluates THIS so a generated candidate can never mask a
        declared cage target to slip past the approval gate — declared-then-overwritten
        targets still reach the boundary check. NEVER raises."""
        try:
            merged = (*(self.declared_targets or ()), *(self.target_files or ()))
            return tuple(dict.fromkeys(str(t) for t in merged if str(t).strip()))
        except Exception:  # noqa: BLE001
            return tuple(self.target_files or ())

    def with_expanded_files(self, files: Tuple[str, ...]) -> "OperationContext":
        """Return a new context with expanded_context_files set (no phase change).

        Called by ContextExpander after expansion rounds complete.
        Uses the same hash-chain mechanics as with_pipeline_deadline().
        """
        intermediate = dataclasses.replace(
            self,
            expanded_context_files=files,
            previous_hash=self.context_hash,
            context_hash="",  # placeholder — recomputed below
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_resurrection(self) -> "OperationContext":
        """Slice 245 — return a new context flagged as a hibernation survivor
        (no phase change). Preserves all durable partial state (phase,
        generation candidates, plan, route) via the same hash-chain mechanics as
        with_expanded_files(). Consumed by the priority layers for Absolute-Max
        Primacy on re-ingest after grid recovery."""
        intermediate = dataclasses.replace(
            self,
            resurrected_from_hibernation=True,
            previous_hash=self.context_hash,
            context_hash="",  # placeholder — recomputed below
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_syntax_retry_feedback(self, feedback: str) -> "OperationContext":
        """Return a context carrying the previous attempt's parse errors.

        Hash-chain-safe by the same mechanics as :meth:`with_resurrection`:
        the retry is a genuinely different attempt and must be auditable as
        one, not a silent re-run wearing the first attempt's identity.

        ``generate_file_hashes`` is deliberately untouched — the retry targets
        the same files at the same disk state, and rewriting those hashes here
        would blind the Slice 248 file-drift guillotine to real drift.

        NEVER raises into the caller: a retry that cannot be stamped degrades
        to no retry, which is the pre-existing behaviour.
        """
        try:
            if not (feedback or "").strip():
                return self
            intermediate = dataclasses.replace(
                self,
                syntax_retry_feedback=str(feedback),
                previous_hash=self.context_hash,
                context_hash="",  # placeholder — recomputed below
            )
            fields_for_hash = _context_to_hash_dict(intermediate)
            new_hash = _compute_hash(fields_for_hash)
            return dataclasses.replace(intermediate, context_hash=new_hash)
        except Exception:  # noqa: BLE001 — degrade to no retry
            return self

    def with_steering_guidance(self, guidance: str) -> "OperationContext":
        """Slice 249 — fold live human steering into the context (durable/auditable
        path), appending to ``strategic_memory_prompt`` and advancing the
        hash-chain via the same mechanics as with_expanded_files(). This is the
        HASH-CHAIN-SAFE context update: it recomputes ``context_hash`` cleanly and
        NEVER touches ``generate_file_hashes`` — so the Slice 248 file-drift
        guillotine (which compares disk-file hashes, not the context hash) cannot
        mistake a steering command for malicious file drift. NEVER raises into the
        caller path (degrades to self on error)."""
        try:
            _existing = self.strategic_memory_prompt or ""
            _merged = f"{_existing}\n\n{guidance}" if _existing else str(guidance)
            intermediate = dataclasses.replace(
                self,
                strategic_memory_prompt=_merged,
                previous_hash=self.context_hash,
                context_hash="",  # placeholder — recomputed below
            )
            fields_for_hash = _context_to_hash_dict(intermediate)
            new_hash = _compute_hash(fields_for_hash)
            return dataclasses.replace(intermediate, context_hash=new_hash)
        except Exception:  # noqa: BLE001
            return self

    def with_infra_warning(self, warning: str) -> "OperationContext":
        """Slice 160 — return a new context carrying a non-fatal infra-hook warning
        (no phase change). Set on fail-soft so the op continues to the governance
        floor / COMPLETE and the operator sees it instead of the op dying."""
        intermediate = dataclasses.replace(
            self,
            infra_warning=str(warning or "")[:2000],
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_benchmark_result(self, result: "BenchmarkResult") -> "OperationContext":
        """Return a new context with benchmark_result set (no phase change)."""
        intermediate = dataclasses.replace(
            self,
            benchmark_result=result,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_pre_apply_snapshots(self, snapshots: Dict[str, str]) -> "OperationContext":
        """Return a new context with pre_apply_snapshots set (no phase change)."""
        intermediate = dataclasses.replace(
            self,
            pre_apply_snapshots=dict(snapshots),  # shallow copy for immutability
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_execution_graph_metadata(
        self,
        *,
        execution_graph_id: str,
        execution_plan_digest: str,
        subagent_count: int,
        parallelism_budget: int,
        causal_trace_id: str,
    ) -> "OperationContext":
        """Stamp execution-graph metadata onto the context (no phase change)."""
        intermediate = dataclasses.replace(
            self,
            execution_graph_id=execution_graph_id,
            execution_plan_digest=execution_plan_digest,
            subagent_count=subagent_count,
            parallelism_budget=parallelism_budget,
            causal_trace_id=causal_trace_id,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_cross_repo_promotion(
        self,
        *,
        repo_scope: Tuple[str, ...],
        dependency_edges: Tuple[Tuple[str, str], ...] = (),
        apply_plan: Tuple[str, ...] = (),
        risk_tier: Optional["RiskTier"] = None,
    ) -> "OperationContext":
        """Elevate this op to multi-repo scope (Cross-Repo Scope Promoter ignition).

        Setting ``repo_scope`` to >1 repo re-derives ``cross_repo=True`` in ``__post_init__`` (via
        ``dataclasses.replace``), routing APPLY through ``_execute_saga_apply``. Optionally forces the
        risk tier (Orange-tier APPROVAL_REQUIRED for autonomous multi-repo mutation). No phase change;
        hash chain preserved (mirrors the other ``with_*`` stampers)."""
        updates: Dict[str, Any] = {
            "repo_scope": tuple(repo_scope),
            "dependency_edges": tuple(dependency_edges),
            "apply_plan": tuple(apply_plan),
            "previous_hash": self.context_hash,
            "context_hash": "",
        }
        if risk_tier is not None:
            updates["risk_tier"] = risk_tier
        intermediate = dataclasses.replace(self, **updates)
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_strategic_memory_context(
        self,
        *,
        strategic_intent_id: str,
        strategic_memory_fact_ids: Tuple[str, ...],
        strategic_memory_prompt: str,
        strategic_memory_digest: str,
    ) -> "OperationContext":
        """Stamp L4 strategic-memory prompt metadata onto the context."""
        intermediate = dataclasses.replace(
            self,
            strategic_intent_id=strategic_intent_id,
            strategic_memory_fact_ids=tuple(strategic_memory_fact_ids),
            strategic_memory_prompt=strategic_memory_prompt,
            strategic_memory_digest=strategic_memory_digest,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_terminal_outcome(
        self,
        *,
        terminal_reason_code: str,
        rollback_occurred: bool = False,
    ) -> "OperationContext":
        """Stamp terminal outcome metadata onto the context."""
        intermediate = dataclasses.replace(
            self,
            terminal_reason_code=terminal_reason_code,
            rollback_occurred=rollback_occurred,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_telemetry(self, tc: TelemetryContext) -> "OperationContext":
        """Stamp TelemetryContext onto the context (no phase change).

        Called exactly once by GovernedLoopService.submit() at intake,
        after concurrency/dedup gates and pipeline_deadline stamping.
        Uses the same hash-chain mechanics as with_pipeline_deadline().
        """
        intermediate = dataclasses.replace(
            self,
            telemetry=tc,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_routing_actual(self, ra: RoutingActualTelemetry) -> "OperationContext":
        """Stamp actual routing outcome onto the existing TelemetryContext (no phase change).

        Called at COMPLETE or POSTMORTEM when the actual provider is known.

        Raises
        ------
        ValueError
            If ``telemetry`` has not been set yet (with_telemetry must precede this).
        """
        if self.telemetry is None:
            raise ValueError(
                "with_routing_actual() called before telemetry was set; "
                "call with_telemetry() first."
            )
        updated_tc = dataclasses.replace(self.telemetry, routing_actual=ra)
        intermediate = dataclasses.replace(
            self,
            telemetry=updated_tc,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_frozen_autonomy_tier(self, tier: str) -> "OperationContext":
        """Stamp autonomy tier onto context at submit time (no phase change).

        Called exactly once by GovernedLoopService.submit() before handing ctx
        to the orchestrator. Gate phase reads ctx.frozen_autonomy_tier instead
        of querying TrustGraduator live, preventing promotion races.

        Parameters
        ----------
        tier:
            ``"governed"`` (auto-proceed) or ``"observe"`` (requires approval).
        """
        intermediate = dataclasses.replace(
            self,
            frozen_autonomy_tier=tier,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_shadow_result(self, result: "ShadowResult") -> "OperationContext":
        """Attach shadow harness result to context (no phase change, hash updates)."""
        intermediate = dataclasses.replace(
            self,
            shadow=result,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_human_instructions(self, instructions: str) -> "OperationContext":
        """Stamp human-authored OUROBOROS.md instructions onto context."""
        intermediate = dataclasses.replace(
            self,
            human_instructions=instructions,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_read_only_intent(self, is_read_only: bool) -> "OperationContext":
        """Stamp the read-only intent flag onto the context (no phase change).

        Called pre-CLASSIFY by the orchestrator after deterministic intent
        inference. Downstream consumers trust the flag because tool_executor
        + orchestrator jointly enforce the no-mutation contract — the flag
        is not advisory metadata, it is part of the hash-chained state.
        """
        intermediate = dataclasses.replace(
            self,
            is_read_only=is_read_only,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def with_attachments(
        self,
        attachments: Tuple[Attachment, ...],
    ) -> "OperationContext":
        """Stamp image ``attachments`` onto the context (no phase change).

        Called by ``VisionSensor`` (``kind="sensor_frame"``) and
        ``visual_verify`` (``kind="pre_apply"`` / ``"post_apply"``). All
        other callers are forbidden by the I7 substrate export-ban
        (``tests/governance/test_attachment_export_ban.py``).

        Validates the per-context cap and element types; preserves the
        hash-chain exactly like the other ``with_*`` helpers.

        Raises
        ------
        ValueError
            If ``len(attachments) > _ATTACHMENT_MAX_PER_CTX``.
        TypeError
            If any element is not an ``Attachment``.
        """
        if not isinstance(attachments, tuple):
            attachments = tuple(attachments)
        if len(attachments) > _ATTACHMENT_MAX_PER_CTX:
            raise ValueError(
                f"OperationContext.with_attachments: at most "
                f"{_ATTACHMENT_MAX_PER_CTX} attachments per context; "
                f"got {len(attachments)}"
            )
        for i, att in enumerate(attachments):
            if not isinstance(att, Attachment):
                raise TypeError(
                    f"attachments[{i}] must be Attachment; "
                    f"got {type(att).__name__}"
                )
        intermediate = dataclasses.replace(
            self,
            attachments=attachments,
            previous_hash=self.context_hash,
            context_hash="",
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)

    def add_attachment(self, attachment: Attachment) -> "OperationContext":
        """Convenience: append a single ``Attachment`` to the existing tuple.

        Delegates to :meth:`with_attachments`, which enforces the cap and
        re-validates element types.
        """
        return self.with_attachments(self.attachments + (attachment,))

    def with_exploration_manifest(
        self,
        manifest: "Optional[ExplorationManifest]",
    ) -> "OperationContext":
        """Stamp a Slice 89 ExplorationManifest onto the context (no phase change).

        Called by CandidateGenerator in the DW-failure window before the
        Claude fallback generate() call.  Mirror of with_pipeline_deadline()
        hash-chain mechanics — previous_hash set, context_hash recomputed.
        """
        # Slice 89 — follow the exact with_*-helper hash pattern
        intermediate = dataclasses.replace(
            self,
            exploration_manifest=manifest,
            previous_hash=self.context_hash,
            context_hash="",  # placeholder — recomputed below
        )
        fields_for_hash = _context_to_hash_dict(intermediate)
        new_hash = _compute_hash(fields_for_hash)
        return dataclasses.replace(intermediate, context_hash=new_hash)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _context_to_hash_dict(ctx: OperationContext) -> Dict[str, Any]:
    """Extract all fields from *ctx* into a dict suitable for hashing.

    The ``context_hash`` field is excluded since it is the value being
    computed.  Enum values are serialized by name for stability. Tuples of
    frozen dataclasses (e.g. ``attachments``) are canonicalized element-
    wise so hashing is deterministic regardless of ``__repr__`` layout.
    """
    d: Dict[str, Any] = {}
    for f in dataclasses.fields(ctx):
        if f.name == "context_hash":
            continue
        value = getattr(ctx, f.name)
        # Serialize enums by name for cross-version stability
        if isinstance(value, Enum):
            value = value.name
        # Serialize frozen dataclass sub-objects to dict
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        # Tuple of dataclass elements (e.g. attachments) — canonicalize
        # each element so the hash is stable independent of repr layout.
        elif (
            isinstance(value, tuple)
            and value
            and dataclasses.is_dataclass(value[0])
            and not isinstance(value[0], type)
        ):
            value = tuple(dataclasses.asdict(elt) for elt in value)
        d[f.name] = value
    return d


# ---------------------------------------------------------------------------
# RepairContext  (L2 self-repair — typed seam between RepairEngine + providers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairContext:
    """Failure context injected into the correction prompt for L2 repair iterations.

    Passed from RepairEngine to PrimeProvider.generate() to _build_codegen_prompt()
    where it triggers the REPAIR MODE section.

    Parameters
    ----------
    iteration:
        1-based current repair iteration number.
    max_iterations:
        Budget ceiling from RepairBudget.max_iterations.
    failure_class:
        One of "syntax", "test", "env", "flake".
    failure_signature_hash:
        SHA-256 of sorted failing test IDs + failure_class (stable across retries).
    failing_tests:
        Top-5 failing test node IDs from the most recent sandbox run.
    failure_summary:
        300-char human-readable error excerpt for the correction prompt.
    current_candidate_content:
        Full text of the failing file as it exists in the sandbox after the
        last patch was applied. The model is asked to diff against this.
    current_candidate_file_path:
        Repo-relative path of the file being repaired.
    dependency_cone:
        Repair Context Bridge (Slice 2) — graph-derived dependency-cone boundary
        clause (downstream dependents / upstream dependencies / causal call-chain
        from the failing test) rendered for the REPAIR MODE prompt. ``None`` until
        the bridge populates it (gated ``JARVIS_REPAIR_CONTEXT_BRIDGE_ENABLED``);
        ``None`` leaves the repair prompt byte-identical to pre-bridge behavior.
    """

    iteration: int
    max_iterations: int
    failure_class: str
    failure_signature_hash: str
    failing_tests: Tuple[str, ...]
    failure_summary: str
    current_candidate_content: str
    current_candidate_file_path: str
    dependency_cone: Optional[str] = None
    # L2 completion (Phase 2): stochastic strategy-escalation directive injected when the repair loop
    # diverges (local minimum). Switches the generation paradigm (localized patch → full-method /
    # module-level rewrite). ``None`` outside a divergence → prompt byte-identical.
    escalation_directive: Optional[str] = None
    # Adaptive Epistemic Feedback Matrix (T2): hybrid epistemic diff between the prior STABLE
    # candidate and the current FAILING iteration, plus the FULL sandbox stderr trace (NOT the
    # 300-char ``failure_summary`` excerpt, which stays for backward-compat). Populated by the L2
    # repair loop only when ``epistemic_feedback_enabled()`` and the diff/trace are non-empty;
    # both default to ``""`` so the repair prompt is byte-identical when the feature is OFF or the
    # context could not be assembled (fail-soft). Rendered into the REPAIR MODE prompt block.
    prior_iteration_diff: str = ""
    failure_trace: str = ""
