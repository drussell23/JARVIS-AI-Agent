"""Cockpit Attach Bridge — `ov attach` over a state-hydrated UDS.

CLI wiring item #6 (operator-authorized 2026-07-18). The organism (the
battle-test harness process) owns a Unix-domain pub/sub socket; an
``ov attach`` terminal is a DUMB RENDERER of the serialized frames the
daemon emits — zero business logic client-side.

Protocol (newline-delimited JSON, schema ``cockpit_attach.v1``):

  * **Hydration payload** — the FIRST frame on accept. Contains the
    live FSM/status snapshot (phase, cost, idle, active op), the active
    LiveWork/op digest, and real-time provider liquidity — the operator
    NEVER stares at a blank screen waiting for the next FSM tick.
    Providers are injected callables (StatusLineBuilder snapshot, the
    liquidity ledger, GLS active-ops) so this module holds no authority
    and no direct state references.
  * **Downstream frames** — ``{"type":"line","text":...}``: every line
    the harness's ``_repl_print`` chokepoint emits (ALREADY conformed by
    the PresentationRouter — the attach surface inherits the design
    language for free). Clients render this frame ESCAPED (inert DATA).
  * **Styled frames (v2.2, Tool Activity)** — ``{"type":"markup",
    "text":...}``: DAEMON-COMPOSED Rich-markup lines mirrored from
    SerpentFlow's op-scoped render chokepoint (``markup_mirror``) — the
    CC-style ``⏺ Bash(...)`` / ``⏺ Update(path)`` + numbered-diff /
    ``⎿ result`` blocks. The composition layer (tool_render_view)
    escapes all MODEL-controlled content before wrapping in markup, so
    the frame is styled chrome around inert data; clients render it
    unescaped (with a validate-else-escape fail-soft). Master:
    ``JARVIS_ATTACH_TOOL_ACTIVITY_ENABLED`` (default on).
  * **Upstream frames** — ``{"type":"input","text":...}``: operator
    stdin from the attached terminal, routed into the harness's
    ``_handle_repl_command`` — the FULL verb set plus the bare-text
    chat bridge, not a chat-only side channel.
  * **Audio orchestration (v2, the Audio-Visual Synapse)** — upstream
    ``{"type":"audio","cmd":"wake"|"sleep"|"barge"}`` arms / disarms /
    interrupts the daemon's karen_duplex voice plane; downstream
    ``{"type":"audio_state","state":...}`` streams the audio FSM
    (``OFFLINE`` / ``UNAVAILABLE`` / ``LISTENING`` / ``HEARING`` /
    ``THINKING`` / ``SPEAKING``) so the attached TUI can morph its
    prompt in real time. The hydration payload carries the CURRENT
    audio state (late joiners never render a stale footer).

Bulletproof contract (mandate 4): the daemon-side publisher is strictly
non-blocking and per-client fail-drop — ``BrokenPipeError`` /
``ConnectionResetError`` / any write fault on a subscriber DROPS that
subscriber and the FSM loop never notices. A SIGKILL'd ``ov attach`` is
a dropped connection, nothing more.

Master ``JARVIS_ATTACH_IPC_ENABLED`` (default on — read/render plus the
same operator input surface the local REPL already exposes; socket is
0600 same-user). NEVER raises anywhere.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
)

logger = logging.getLogger("Ouroboros.CockpitAttach")

COCKPIT_ATTACH_SCHEMA_VERSION = "cockpit_attach.v2"

#: Closed taxonomy of daemon→TUI audio FSM states (the morphing
#: vocabulary). OFFLINE = voice plane not armed; UNAVAILABLE = wake
#: requested but no duplex handle mounted in this process.
AUDIO_STATES = (
    "OFFLINE", "UNAVAILABLE", "LISTENING", "HEARING", "THINKING",
    "SPEAKING", "HELD",
)

#: Closed taxonomy of TUI→daemon audio commands. v2.1: force_wake
#: preempts an incumbent terminal; ptt/ptt_stop bracket an ephemeral
#: push-to-talk hold; flush halts outbound audio (ducking).
AUDIO_CMDS = (
    "wake", "sleep", "barge", "force_wake", "ptt", "ptt_stop", "flush",
)

_TRUTHY = ("1", "true", "yes", "on")


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Does *fn* declare a keyword parameter called *name*?

    Keyword rather than a third positional: the answer's prompt_id is
    OPTIONAL context, and threading it positionally would make every existing
    two-argument sink's meaning depend on argument order it never agreed to.
    A sink that wants it says so by name.

    Defaults to False when the signature cannot be read — the sink is then
    called exactly as it was before.
    """
    try:
        import inspect
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            if p.name == name and p.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return True
        return False
    except (TypeError, ValueError):
        return False


def _accepts_two_positional(fn: Any) -> bool:
    """Can *fn* be called with (text, session)?

    Sinks predating session routing take one argument, and both shapes must
    keep working — this bridge is the only path an attached cockpit has to
    the REPL, so guessing wrong drops the operator's command. Determined by
    signature rather than by trial call: see the note at the assignment site
    for why a retry-on-TypeError is unsafe here.

    Defaults to False (the older, narrower shape) when the signature cannot
    be read at all — a C builtin or an exotic callable still gets called."""
    try:
        import inspect
        sig = inspect.signature(fn)
        positional = 0
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                return True
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional += 1
        return positional >= 2
    except (TypeError, ValueError):
        return False


#: The live bridge, for code that cannot be handed one.
#:
#: Same pattern as `bipartite_layout.set_active_canvas`: the producer
#: (the harness, which builds the bridge) and the consumer (the speech
#: primitive, three packages away) have no handle to each other.
_ACTIVE_BRIDGE: Any = None


def set_active_bridge(bridge: Any) -> None:
    """Register the live bridge, or clear it with None. NEVER raises."""
    global _ACTIVE_BRIDGE
    try:
        _ACTIVE_BRIDGE = bridge
    except Exception:  # noqa: BLE001
        pass


def attached_cockpits() -> int:
    """How many cockpits are listening right now. NEVER raises."""
    try:
        clients = getattr(_ACTIVE_BRIDGE, "_clients", None)
        return len(clients) if clients else 0
    except Exception:  # noqa: BLE001
        return 0


def operator_present() -> bool:
    """Is there a human who can perceive output right now?

    THE PROBLEM THIS ANSWERS
    -------------------------
    `ov` detaches: closing the terminal leaves the organism running, which is
    the intended behaviour and a good one. But Karen kept NARRATING to a
    machine whose operator had gone home — audio out of the speakers with no
    cockpit anywhere to hear it. Text already behaves correctly:
    `publish_markup` returns early when `_clients` is empty. Speech had no
    equivalent, so it addressed an empty room.

    Two ways a human can be present, and both count:

      * a cockpit is ATTACHED over the bridge (`ov attach`), or
      * this process owns a real terminal (a foreground run with its own
        REPL, where the operator is looking straight at it).

    The TTY arm is also the honest degradation: if the bridge cannot be
    consulted at all, a detached daemon has no terminal and falls silent
    while a foreground run keeps its voice — which is the correct answer in
    both cases, reached without guessing.
    """
    try:
        if attached_cockpits() > 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from backend.core.ouroboros.battle_test.presentation_restraint import (
            real_stdout_isatty,
        )
        return bool(real_stdout_isatty())
    except Exception:  # noqa: BLE001
        return False


def publish_markup_global(text: str, *, session: Optional[str] = None) -> bool:
    """Emit one composed line to every attached cockpit. Returns whether it went.

    For producers with no handle to the bridge — the token stream runs three
    packages away from the harness that owns it. Same registry the presence
    check uses, so "is anyone listening" and "send it to them" cannot
    disagree about which bridge is live.

    The caller owns ESCAPING. This channel is styled chrome around inert
    data; untrusted text must be escaped before it arrives, per
    `publish_markup`. NEVER raises.
    """
    try:
        bridge = _ACTIVE_BRIDGE
        if bridge is None or attached_cockpits() <= 0:
            return False
        bridge.publish_markup(str(text), session=session)
        return True
    except Exception:  # noqa: BLE001
        return False


def publish_telemetry_global(payload: dict) -> bool:
    """Emit one typed telemetry frame to attached cockpits. NEVER raises.

    The control-plane sibling of :func:`publish_markup_global`, for producers
    with no handle to the bridge. Used by live STATE — a frame dropped here
    costs one frame of smoothness, never a transcript line.
    """
    try:
        bridge = _ACTIVE_BRIDGE
        if bridge is None or attached_cockpits() <= 0:
            return False
        bridge.publish_telemetry(payload)
        return True
    except Exception:  # noqa: BLE001
        return False


def install_panic_broadcast() -> bool:
    """Route FATAL_PANIC frames onto the UDS telemetry lane. NEVER raises.

    Registered once; the arbiter dedups, so a cascade is one frame. Uses
    `publish_telemetry_global` — the same lane and envelope the heartbeat,
    the in-flight stream and the tool tail already ride, so no new
    transport, no new frame convention, and clients that predate this
    ignore an unknown `kind` exactly as they always have.
    """
    try:
        from backend.core.ouroboros.battle_test.panic_arbiter import (
            register_sink,
        )
        register_sink(publish_telemetry_global)
        return True
    except Exception:  # noqa: BLE001
        return False


def attach_enabled() -> bool:
    """Master gate — default ON. NEVER raises."""
    return os.environ.get(
        "JARVIS_ATTACH_IPC_ENABLED", "1",
    ).strip().lower() in _TRUTHY


def attach_socket_path() -> Path:
    return Path(os.environ.get(
        "JARVIS_ATTACH_IPC_SOCKET", ".jarvis/cockpit_attach.sock",
    ))


def _connect_patience_s() -> float:
    """Total attach patience across escalating retries (client side).
    A post-boot storm can lag hydration for seconds — the patience budget
    absorbs it; a REFUSED socket still fails instantly."""
    try:
        return max(0.5, min(120.0, float(os.environ.get(
            "JARVIS_ATTACH_CONNECT_PATIENCE_S", "12",
        ))))
    except (TypeError, ValueError):
        return 12.0


def _connect_timeout_s() -> float:
    try:
        return max(0.05, float(os.environ.get(
            "JARVIS_ATTACH_CONNECT_TIMEOUT_S", "0.5",
        )))
    except (TypeError, ValueError):
        return 0.5


def tool_activity_enabled() -> bool:
    """Master gate for the typed ``markup`` frame (CC-style tool blocks /
    diffs mirrored to attached cockpits). Default ON. NEVER raises."""
    return os.environ.get(
        "JARVIS_ATTACH_TOOL_ACTIVITY_ENABLED", "1",
    ).strip().lower() in _TRUTHY


def offline_backlog_enabled() -> bool:
    """Master gate for retaining render frames emitted with NO cockpit attached.

    Default ON. OFF is byte-identical to the pre-backlog behaviour: frames
    emitted to an empty audience are dropped exactly as before."""
    return os.environ.get(
        "JARVIS_ATTACH_OFFLINE_BACKLOG_ENABLED", "1",
    ).strip().lower() in _TRUTHY


def _backlog_size() -> int:
    """How many unheard frames to retain. Env ``JARVIS_ATTACH_BACKLOG_SIZE``.

    A ring, not a buffer: the organism is long-lived and may run unattended
    for hours, so this is sized to "what an operator can usefully read on
    reattach", never to "everything that happened". Overflow is COUNTED, per
    the no-silent-caps discipline -- a truncated backlog that presented itself
    as complete would be a receipt for work it did not show."""
    try:
        return max(1, min(10000, int(
            os.environ.get("JARVIS_ATTACH_BACKLOG_SIZE", "200"),
        )))
    except Exception:  # noqa: BLE001
        return 200


def _backlog_ttl_s() -> float:
    """Age past which a retained frame is no longer worth showing.

    Env ``JARVIS_ATTACH_BACKLOG_TTL_S``, default 1800 (30 min); ``0``
    disables ageing. Render frames describe what the organism was doing at a
    MOMENT. Replaying six-hour-old tool activity into a fresh cockpit would
    paint history as present tense -- the same class of dishonesty as a
    cached blast radius reported as a live measurement."""
    try:
        return max(0.0, float(
            os.environ.get("JARVIS_ATTACH_BACKLOG_TTL_S", "1800"),
        ))
    except Exception:  # noqa: BLE001
        return 1800.0


class _OfflineBacklog:
    """Render frames emitted while nobody was listening.

    THE DEFECT THIS CLOSES
    ----------------------
    ``publish_line`` / ``publish_markup`` both open with ``if not
    self._clients: return``. That is correct about the socket -- there is
    nobody to write to -- and wrong about the ORGANISM, which went on
    working. Every ``Update(path)`` block, every tool result, every diff
    produced while the operator was detached was composed, formatted, and
    discarded. Reattaching then showed an idle-looking cockpit for a session
    that had been busy, which is indistinguishable from the organism having
    done nothing.

    WHAT IS RETAINED, AND WHAT IS DELIBERATELY NOT
    ---------------------------------------------
    * **Only the two RENDER channels.** The other ~9 ``not self._clients``
      early returns in this module are control-plane (pending prompts, audio
      state, autonomy holds, telemetry). A prompt frame replayed after its
      gate resolved would offer the operator a decision that no longer
      exists; ``focus_shield`` owns that lifecycle and has its own queue.
      Retention here would be a second, disagreeing opinion about a live FSM.

    * **Only AMBIENT frames.** A frame emitted under a session ContextVar is
      one cockpit's private verb output. Its asker is gone, and session ids
      are per-run, so replaying it to whoever attaches next paints someone
      else's ``/posture status`` into a stranger's scrollback -- precisely
      what ``_broadcast``'s addressing exists to prevent, arriving through a
      new door. Ambient output has no owner and is everyone's business,
      which is exactly what makes it safe to replay.

    * **Only while the audience is EMPTY.** Retention stops the moment a
      cockpit attaches. A second cockpit attaching later is joining a live
      session, not recovering a gap, so it correctly sees nothing.

    TWO-PHASE DRAIN
    ---------------
    :meth:`peek` then :meth:`commit`. The flush happens mid-handshake, and a
    client whose socket breaks during it never saw a byte; clearing the ring
    on read would destroy the backlog on behalf of a cockpit that failed to
    receive it. So nothing is discarded until the write is known to have
    landed.

    Thread-safe: producers call from arbitrary threads (the publishers hop to
    the loop only AFTER this point), the drain runs on the loop.
    """

    __slots__ = ("_frames", "_lock", "retained", "overflowed",
                 "replayed", "dropped_stale")

    def __init__(self) -> None:
        self._frames: Deque[Tuple[float, Dict[str, Any]]] = deque(
            maxlen=_backlog_size(),
        )
        self._lock = threading.Lock()
        self.retained = 0
        self.overflowed = 0
        self.replayed = 0
        self.dropped_stale = 0

    def _evict_stale(self, now: float) -> None:
        """Drop aged frames from the front. Caller holds the lock."""
        ttl = _backlog_ttl_s()
        if ttl <= 0.0:
            return
        cutoff = now - ttl
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()
            self.dropped_stale += 1

    def retain(self, msg: Dict[str, Any]) -> None:
        """Record one unheard frame. NEVER raises."""
        try:
            now = time.time()
            with self._lock:
                # `maxlen` silently discards the oldest, so overflow is
                # detected by watching the ring sit at capacity rather than by
                # trusting append to tell us. Counted, never silent.
                if (self._frames.maxlen is not None
                        and len(self._frames) == self._frames.maxlen):
                    self.overflowed += 1
                self._frames.append((now, msg))
                self.retained += 1
                self._evict_stale(now)
        except Exception:  # noqa: BLE001 — retention must never break a publish
            pass

    def peek(self) -> List[Dict[str, Any]]:
        """Frames to replay, oldest first. Does NOT consume. NEVER raises."""
        try:
            now = time.time()
            with self._lock:
                self._evict_stale(now)
                out = []
                for ts, msg in self._frames:
                    frame = dict(msg)
                    # Honest provenance: this is history, and it carries the
                    # moment it happened rather than the moment it was
                    # replayed. A client that stamped `ts` at receipt would
                    # date an hour-old diff to now.
                    frame["backlogged"] = True
                    frame["ts"] = ts
                    frame["addressed"] = False
                    out.append(frame)
                return out
        except Exception:  # noqa: BLE001
            return []

    def commit(self, count: int) -> None:
        """Discard the *count* frames a client has now provably received."""
        try:
            with self._lock:
                for _ in range(max(0, int(count))):
                    if not self._frames:
                        break
                    self._frames.popleft()
                    self.replayed += 1
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> Dict[str, int]:
        """Counters for `/doctor` and the bridge stats. NEVER raises."""
        try:
            with self._lock:
                pending = len(self._frames)
        except Exception:  # noqa: BLE001
            pending = 0
        return {
            "backlog_pending": pending,
            "backlog_retained": self.retained,
            "backlog_replayed": self.replayed,
            "backlog_overflowed": self.overflowed,
            "backlog_dropped_stale": self.dropped_stale,
        }


def _sentinel_interval_s() -> float:
    """Socket self-heal cadence (0 disables). The bridge binds once at
    boot; if ANY confused peer unlinks the inode (a CLI misclassifying
    a starved organism as a ghost — the 2026-07-23 class — an operator
    ``rm``, a tmp janitor), the organism silently becomes permanently
    unattachable. The sentinel rebinds a vanished inode. NEVER raises."""
    try:
        return max(0.0, min(300.0, float(os.environ.get(
            "JARVIS_ATTACH_SENTINEL_S", "10",
        ))))
    except (TypeError, ValueError):
        return 10.0


# ---------------------------------------------------------------------------
# Daemon side — lives in the harness process
# ---------------------------------------------------------------------------


class CockpitAttachBridge:
    """The organism's attach server.

    ``status_provider`` / ``ops_provider`` / ``liquidity_provider`` are
    injected zero-authority callables returning JSON-serializable dicts;
    each is consulted fresh at every handshake (pull model — hydration
    is always current, never cached). ``on_input`` receives operator
    text from attached terminals.
    """

    def __init__(
        self,
        *,
        status_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        ops_provider: Optional[Callable[[], Any]] = None,
        liquidity_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        fabrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        replay_provider: Optional[Callable[[], Any]] = None,
        on_input: Optional[Callable[[str], None]] = None,
        on_audio: Optional[Callable[[str], None]] = None,
        on_autonomy: Optional[Callable[[str], None]] = None,
        rewind_provider: Optional[Callable[[int], Any]] = None,
        path: Optional[Path] = None,
    ) -> None:
        self._status = status_provider or (lambda: {})
        self._ops = ops_provider or (lambda: [])
        self._liquidity = liquidity_provider or (lambda: {})
        # ov doctor edge 4: hive-fabric attachment stats (trinity subs /
        # sse / emitter). Pull-model like every other provider — consulted
        # fresh at each handshake, zero authority.
        self._fabrics = fabrics_provider or (lambda: {})
        # Slice H: zero-authority pull of the TrinityEventBus replay buffer —
        # the chronological critical telemetry flushed to a client on connect
        # so a late attach reconciles the DAG history, not just the snapshot.
        self._replay = replay_provider or (lambda: [])
        self._on_input = on_input or (lambda _t: None)
        self._on_audio = on_audio or (lambda _c: None)
        #: The Transactional Viewport Lock's execution seam: called with
        #: "pause" on the FIRST hold and "resume" when the LAST hold
        #: releases. The harness wires this to the same intake pause the
        #: pause/resume verbs use — one implementation, two entrances.
        self._on_autonomy = on_autonomy or (lambda _a: None)
        #: rewind_provider(limit) → serializable candidate list for the
        #: Esc-Esc menu. Supplied by the harness from the EXISTING /undo
        #: planner — this bridge never touches git.
        self._rewind = rewind_provider or (lambda _n: [])
        #: session → monotonic deadline. A REFCOUNT, not a flag: two panes
        #: may hold rewind menus at once, and autonomy resumes only when
        #: the last hold releases. The TTL is the dead-client failsafe —
        #: an operator's crashed terminal must never freeze the organism
        #: forever; _drop() also releases synchronously on disconnect.
        self._autonomy_holds: Dict[str, float] = {}
        #: Does the input sink accept the originating session?
        #:
        #: Decided ONCE, by inspection. The obvious alternative — call with
        #: two arguments and retry on TypeError — cannot distinguish "wrong
        #: arity" from "the sink raised TypeError while doing its job", and
        #: in the second case the retry executes the operator's command a
        #: SECOND time. A dispatch surface must never guess by re-running the
        #: thing it is dispatching.
        self._input_takes_session = _accepts_two_positional(self._on_input)
        self._input_takes_prompt_id = _accepts_kwarg(self._on_input,
                                                     "prompt_id")
        self._path = Path(path) if path is not None else attach_socket_path()
        self._server: Optional[asyncio.AbstractServer] = None
        self._sentinel_task: Optional[asyncio.Task] = None
        self._clients: Set[asyncio.StreamWriter] = set()
        #: Render frames composed while `_clients` was empty. The organism
        #: does not stop working when the operator closes a terminal; before
        #: this, everything it rendered in that window was discarded.
        self._backlog = _OfflineBacklog()
        #: session_id → that cockpit's writer. Populated from the session
        #: field on inbound frames; entries are removed by _drop so a
        #: detached cockpit can never be addressed.
        self._sessions: Dict[str, asyncio.StreamWriter] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_state: str = "OFFLINE"
        self.stats: Dict[str, int] = {
            "connects": 0, "dropped": 0, "lines_published": 0,
            "inputs_received": 0, "audio_cmds": 0,
            "audio_states_published": 0,
        }

    # ---- lifecycle ----

    async def start(self) -> bool:
        """Bind (removing any stale socket). False on failure — the
        organism keeps running unattachable. NEVER raises."""
        try:
            if not attach_enabled():
                return False
            self._loop = asyncio.get_running_loop()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self._path.exists():
                    self._path.unlink()
            except OSError:
                pass
            self._server = await asyncio.start_unix_server(
                self._on_client, path=str(self._path),
            )
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
            logger.info("[CockpitAttach] bridge bound at %s", self._path)
            if _sentinel_interval_s() > 0:
                try:
                    self._sentinel_task = asyncio.get_running_loop(
                    ).create_task(self._socket_sentinel())
                except Exception:  # noqa: BLE001
                    self._sentinel_task = None
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CockpitAttach] bind failed: %s", exc)
            self._server = None
            return False

    async def stop(self) -> None:
        """Close server + subscribers, unlink socket. NEVER raises.

        Order is load-bearing on Python 3.12+: ``Server.wait_closed()``
        now genuinely waits for every in-flight client handler (the 3.9-3.11
        behavior returned immediately), so clients must be DROPPED FIRST —
        with an attached ``ov`` terminal, the old order hangs shutdown
        forever. Bounded as belt-and-braces: never-hangs is this module's
        law even if a handler wedges."""
        try:
            if self._sentinel_task is not None:
                try:
                    self._sentinel_task.cancel()
                except Exception:  # noqa: BLE001
                    pass
                self._sentinel_task = None
            for w in list(self._clients):
                self._drop(w)
            if self._server is not None:
                self._server.close()
                try:
                    await asyncio.wait_for(self._server.wait_closed(),
                                           timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
                self._server = None
            try:
                if self._path.exists():
                    self._path.unlink()
            except OSError:
                pass
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] stop degraded", exc_info=True)

    async def _socket_sentinel(self) -> None:
        """Self-heal watchdog: while the bridge is up, verify the socket
        inode still exists; if something unlinked it, REBIND at the same
        path so the organism never becomes silently unattachable. Reads
        only the filesystem + its own server handle (no app state — the
        watchdog-isolation principle). NEVER raises; ends with stop()."""
        try:
            while True:
                await asyncio.sleep(_sentinel_interval_s())
                if self._server is None:      # stopped — sentinel ends
                    return
                try:
                    if self._path.exists():
                        continue
                    logger.warning(
                        "[CockpitAttach] socket inode VANISHED at %s — "
                        "rebinding (self-heal)", self._path,
                    )
                    old = self._server
                    self._server = None
                    try:
                        old.close()
                        await asyncio.wait_for(
                            old.wait_closed(), timeout=2.0,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._server = await asyncio.start_unix_server(
                        self._on_client, path=str(self._path),
                    )
                    try:
                        os.chmod(self._path, 0o600)
                    except OSError:
                        pass
                    logger.info(
                        "[CockpitAttach] bridge REBOUND at %s", self._path,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[CockpitAttach] sentinel heal degraded",
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            return

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def backlog_stats(self) -> Dict[str, int]:
        """Counters for the offline render backlog. NEVER raises.

        Separate from ``self.stats`` because those are lifetime totals that
        several call sites increment in place, while these are owned entirely
        by the ring and include a LIVE gauge (`backlog_pending`) that would be
        wrong the moment it were copied into a totals dict."""
        try:
            return self._backlog.snapshot()
        except Exception:  # noqa: BLE001
            return {}

    def _publish_presence(self) -> None:
        """Republish operator attention from the ONE authority for it —
        ``_clients`` — to the process-wide ledger that gates operator
        clocks (the review window, principally).

        A COUNT, never a delta: a missed detach cannot leak a phantom
        attendee and a double-drop cannot underflow, because the ledger
        is told the current truth rather than asked to derive it. Called
        at exactly the two transitions of ``_clients`` — post-hydration
        add, and ``_drop`` — so a socket that connected but died before
        its hydration frame landed is correctly never counted as
        attention. Best-effort: a degraded ledger must never break the
        transport."""
        try:
            from backend.core.ouroboros.governance.attention_ledger import (
                SOURCE_COCKPIT,
                get_attention_ledger,
            )
            get_attention_ledger().set_count(
                SOURCE_COCKPIT, len(self._clients),
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "[CockpitAttach] presence publish degraded", exc_info=True,
            )

    # ---- hydration ----

    def _hydration_payload(self) -> Dict[str, Any]:
        def _safe(fn: Callable[[], Any], fallback: Any) -> Any:
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return fallback

        return {
            "type": "hydration",
            "schema_version": COCKPIT_ATTACH_SCHEMA_VERSION,
            "ts": time.time(),
            "status": _safe(self._status, {}),
            "ops": _safe(self._ops, []),
            "liquidity": _safe(self._liquidity, {}),
            "fabrics": _safe(self._fabrics, {}),
            "audio": {"state": self._audio_state},
            # WHICH MODEL IS ANSWERING — from the process that will answer.
            #
            # The cockpit banner used to derive this from the CLIENT's own
            # environment. The client is a different process with a
            # different environment: it never loads `.env`, so a correctly
            # pinned daemon rendered NO model at all, and a stale client
            # export would have rendered the wrong one confidently. The
            # operator had no way to tell a fine-tune from its base.
            #
            # Resolved at the daemon's boot gate against a registry that
            # answered, so this is what is loaded, not what was requested.
            # Empty means "not yet resolved" and the banner omits the
            # field — naming a model we are not certain of is worse than
            # naming none.
            "model": _safe(self._active_model, ""),
        }

    @staticmethod
    def _active_model() -> str:
        """The tag the generation lane resolved. NEVER raises."""
        try:
            from backend.core.ouroboros.governance.candidate_generator import (
                active_model_tag,
            )
            return active_model_tag()
        except Exception:  # noqa: BLE001 — a banner never breaks an attach
            return ""

    # ---- publish (the harness _repl_print mirror) ----

    def _retain_unheard(self, msg: Dict[str, Any]) -> None:
        """Capture one render frame nobody was there to receive.

        ONE seam for both render channels so the retention rule cannot drift
        between them -- the two publishers already differ in escaping policy
        and that is exactly the kind of divergence that produced a palette
        fixed on one surface and dead on the other.

        Ambience is resolved through ``attach_session.current_session()``, the
        SAME call ``_broadcast`` uses to address a frame. Reading the session
        a second way here would mean two opinions about who owns a line, and
        the one that decides retention would be invisible."""
        if not offline_backlog_enabled():
            return
        try:
            from backend.core.ouroboros.battle_test.attach_session import (
                current_session,
            )
            if current_session() is not None:
                # Addressed output: it belongs to a cockpit that has gone, and
                # session ids do not survive a reattach. Replaying it would
                # put one operator's verb answer in another's scrollback.
                return
        except Exception:  # noqa: BLE001 — unresolvable ownership -> do not retain
            return
        self._backlog.retain(msg)

    def publish_line(self, text: str) -> None:
        """Broadcast one (already router-conformed) line to every
        attached terminal. Strictly non-blocking; per-client fail-drop
        (BrokenPipe/ConnectionReset = subscriber gone, organism
        unbothered). Thread-safe. NEVER raises."""
        try:
            msg = {"type": "line", "text": str(text), "ts": time.time()}
            if not self._clients:
                # Composed, and nobody to receive it. Retained rather than
                # dropped -- see `_OfflineBacklog`.
                self._retain_unheard(msg)
                return
            self.stats["lines_published"] += 1
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._broadcast(msg)
            else:
                loop.call_soon_threadsafe(self._broadcast, msg)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] publish degraded", exc_info=True)

    def publish_markup(self, text: str, *,
                       session: Optional[str] = None) -> None:
        """Publish one DAEMON-COMPOSED styled line (Rich markup) to every
        attached terminal — the CC-style tool-activity channel (⏺ Bash /
        ⏺ Update diffs / ⎿ results). TYPED separately from ``line`` so the
        client can render it unescaped: the composition layer
        (tool_render_view / serpent_flow) escapes all MODEL-controlled
        content before wrapping it in markup, so the frame is styled chrome
        around inert data. Untrusted/raw text must NEVER travel here — use
        publish_line. Strictly non-blocking; thread-safe; NEVER raises.

        ``session`` addresses the frame to ONE cockpit — the one that ran the
        command. Verb output belongs to whoever asked for it: `/posture
        status` typed in one terminal must not paint in another. ``None``
        broadcasts, which is right for AMBIENT output, since an autonomous
        operation belongs to no one and is everyone's business."""
        try:
            if not tool_activity_enabled():
                # The CHANNEL is off. Not an empty audience -- retaining here
                # would replay frames the operator disabled, so a kill switch
                # would become a delay.
                return
            # A LINE STREAM CARRIES LINES. A Rich renderable handed here by a
            # producer that forgot to render it would cross the wire as its
            # repr — `<rich.panel.Panel object at 0x...>` reached a cockpit
            # this way (measured 2026-09-06, the boot masthead). That is
            # worse than dropping it: it is a line that looks like a bug and
            # says nothing. Refused here, at the one seam every markup frame
            # passes through, and said so at DEBUG with the producer's type,
            # so the next reader can find the caller rather than the symptom.
            if not isinstance(text, str):
                logger.debug(
                    "[CockpitAttach] markup publish refused a non-string "
                    "%s — render it before publishing", type(text).__name__,
                )
                return
            msg = {"type": "markup", "text": text, "ts": time.time()}
            if session:
                msg["session"] = session
            if not self._clients:
                if not session:
                    self._retain_unheard(msg)
                return
            self.stats["markup_published"] = (
                self.stats.get("markup_published", 0) + 1
            )
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._broadcast(msg)
            else:
                loop.call_soon_threadsafe(self._broadcast, msg)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] markup publish degraded", exc_info=True)

    # ---- the Transactional Viewport Lock (autonomy pause refcount) ----

    @staticmethod
    def _autonomy_ttl_s() -> float:
        """Dead-holder failsafe. A crashed terminal releases via _drop;
        this covers the one it can't — a client alive on the socket but
        wedged with its menu open. Env: JARVIS_AUTONOMY_LOCK_TTL_S."""
        try:
            return max(5.0, float(
                os.environ.get("JARVIS_AUTONOMY_LOCK_TTL_S", "120"),
            ))
        except (TypeError, ValueError):
            return 120.0

    def _sweep_autonomy_holds(self) -> None:
        now = time.monotonic()
        for sid, deadline in list(self._autonomy_holds.items()):
            if deadline <= now:
                del self._autonomy_holds[sid]

    def _autonomy_ctl(self, action: str, session: Optional[str]) -> None:
        """One pause/resume request from one cockpit. Refcounted: the
        daemon's execution intake pauses on the FIRST hold and resumes on
        the LAST release — pane B's open rewind menu keeps the world
        frozen even after pane A closes hers. Every transition (and every
        no-op re-assert) re-broadcasts autonomy_state so all panes agree.
        NEVER raises."""
        try:
            sid = str(session or "").strip() or "?anonymous"
            was_held = bool(self._autonomy_holds)
            self._sweep_autonomy_holds()
            if action == "pause":
                self._autonomy_holds[sid] = (
                    time.monotonic() + self._autonomy_ttl_s()
                )
            elif action == "resume":
                self._autonomy_holds.pop(sid, None)
            else:
                return
            now_held = bool(self._autonomy_holds)
            if now_held and not was_held:
                self.stats["autonomy_pauses"] = (
                    self.stats.get("autonomy_pauses", 0) + 1
                )
                try:
                    self._on_autonomy("pause")
                except Exception:  # noqa: BLE001
                    logger.debug("[CockpitAttach] pause sink degraded",
                                 exc_info=True)
            elif was_held and not now_held:
                try:
                    self._on_autonomy("resume")
                except Exception:  # noqa: BLE001
                    logger.debug("[CockpitAttach] resume sink degraded",
                                 exc_info=True)
            self.publish_autonomy_state()
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] autonomy ctl degraded",
                         exc_info=True)

    def publish_autonomy_state(self) -> None:
        """Broadcast the lock's truth to every cockpit — held or free, and
        by how many holders — so a pane that never touched Esc-Esc still
        shows the ⏸ freeze another pane caused. NEVER raises."""
        try:
            self._sweep_autonomy_holds()
            msg = {
                "type": "autonomy_state",
                "paused": bool(self._autonomy_holds),
                "holders": len(self._autonomy_holds),
                "ts": time.time(),
            }
            loop = self._loop
            if loop is None or loop.is_closed():
                return

            def _emit() -> None:
                from backend.core.ouroboros.battle_test.attach_session import (
                    session_scope,
                )
                with session_scope(None):
                    self._broadcast(msg)

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _emit()
            else:
                loop.call_soon_threadsafe(_emit)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] autonomy state degraded",
                         exc_info=True)

    def _serve_rewind_list(
        self, session: Optional[str], *, limit: Any = None,
    ) -> None:
        """Answer one Esc-Esc menu request with the LOCKED restore points.

        Called after the requester's pause hold landed, so the list is a
        static snapshot — nothing is mutating underneath it. The provider
        is the harness's adapter over the EXISTING /undo planner; this
        bridge never reads git. Addressed to the requester alone, like
        lane history. NEVER raises."""
        try:
            try:
                n = max(1, min(int(limit), 50)) if limit is not None else 10
            except (TypeError, ValueError):
                n = 10
            try:
                candidates = list(self._rewind(n) or [])
            except Exception:  # noqa: BLE001
                logger.debug("[CockpitAttach] rewind provider degraded",
                             exc_info=True)
                candidates = []
            payload = {
                "type": "rewind_list",
                "candidates": candidates,
                "locked": bool(self._autonomy_holds),
                "ts": time.time(),
            }
            from backend.core.ouroboros.battle_test.attach_session import (
                session_scope,
            )
            loop = self._loop
            if loop is None or loop.is_closed():
                return

            def _emit() -> None:
                with session_scope(session):
                    self._broadcast(payload)

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _emit()
            else:
                loop.call_soon_threadsafe(_emit)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] rewind list degraded",
                         exc_info=True)

    def publish_history_append(
        self, text: str, *, origin_session: Optional[str] = None,
    ) -> None:
        """Fan one appended history entry out to every attached cockpit
        EXCEPT the originator — it already has the line; echoing it back
        would double it in the one terminal guaranteed to have it.

        Server-side exclusion by the session→writer binding, not by a
        ContextVar: this is a route decision ("everyone but X"), which the
        addressing scope ("only X") cannot express. ``origin_session=None``
        means the DAEMON's own terminal typed it — every client receives.
        The frame still carries ``origin`` so a client can drop its own
        echo if a rebind ever mis-routes. Thread-safe; strictly
        non-blocking; NEVER raises."""
        try:
            text = str(text or "")
            if not text.strip() or not self._clients:
                return
            msg: Dict[str, Any] = {
                "type": "history_append", "text": text, "ts": time.time(),
            }
            if origin_session:
                msg["origin"] = origin_session
            self.stats["history_fanout"] = (
                self.stats.get("history_fanout", 0) + 1
            )
            loop = self._loop
            if loop is None or loop.is_closed():
                return

            def _emit() -> None:
                exclude = (
                    self._sessions.get(origin_session)
                    if origin_session else None
                )
                data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
                for w in list(self._clients):
                    if w is exclude:
                        continue
                    try:
                        if w.is_closing():
                            self._drop(w)
                            continue
                        w.write(data)
                    except Exception:  # noqa: BLE001 — writer fault = drop
                        self._drop(w)

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _emit()
            else:
                loop.call_soon_threadsafe(_emit)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] history fanout degraded",
                         exc_info=True)

    def _sync_history(self, text: str, origin_session: Optional[str]) -> None:
        """One operator line became history somewhere — propagate it.

        Fan out to the other cockpits AND inject memory-only into THIS
        process's singleton, so the daemon terminal's Up-arrow keeps pace
        with its clients. Gated + fail-soft; NEVER raises."""
        try:
            from backend.core.ouroboros.battle_test.history_sync import (
                inject_history_entry,
                is_history_sync_enabled,
            )
            if not is_history_sync_enabled():
                return
            self.publish_history_append(text, origin_session=origin_session)
            inject_history_entry(text)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] history sync degraded",
                         exc_info=True)

    def publish_prompt(
        self,
        prompt_id: str,
        text: str = "",
        *,
        risk: str = "",
        timeout_s: float = 0.0,
    ) -> None:
        """Announce an OPEN interactive gate as structured data.

        The question already travels as a mirrored ``markup`` line, and that
        stays — it is what puts the gate in the transcript. But a line is
        prose: a cockpit cannot tell it apart from any other ⏺ chrome, so it
        cannot know a question is pending, which one, or when it dies.

        With the id and the deadline on the wire, an attached cockpit can
        defer the gate until its operator is not mid-sentence and still
        answer the RIGHT op (`resolve` refuses a mismatched ``prompt_id``).
        Broadcast, never session-addressed: an autonomous gate belongs to no
        one terminal and every attached operator may answer it.

        Thread-safe; strictly non-blocking; NEVER raises.
        """
        try:
            prompt_id = str(prompt_id or "").strip()
            if not prompt_id or not self._clients:
                return
            msg = {
                "type": "prompt", "prompt_id": prompt_id,
                "text": str(text or ""), "risk": str(risk or ""),
                "timeout_s": float(timeout_s or 0.0), "ts": time.time(),
            }
            self.stats["prompts_published"] = (
                self.stats.get("prompts_published", 0) + 1
            )
            self._dispatch(msg)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] prompt publish degraded",
                         exc_info=True)

    def publish_transcript(
        self, utterance_id: str, text: str, *,
        final: bool = False, role: str = "user", seq: int = 0,
    ) -> None:
        """Stream one recognised chunk to attached cockpits.

        Carries the SAME shape `audio_state_ipc` already uses — utterance id,
        accumulated text, final flag — rather than a second transcript
        vocabulary that would have to be kept in step with it.

        The id and seq are what let a cockpit fold chunks safely: a socket
        does not promise ordering, and a partial applied after its own final
        would rewind the sentence in the operator's prompt.
        """
        try:
            utterance_id = str(utterance_id or "").strip()
            if not utterance_id or not self._clients:
                return
            self.stats["transcripts_published"] = (
                self.stats.get("transcripts_published", 0) + 1
            )
            self._dispatch({
                "type": "transcript", "utterance_id": utterance_id,
                "text": str(text or ""), "final": bool(final),
                "role": str(role or "user"), "seq": int(seq or 0),
                "ts": time.time(),
            })
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] transcript publish degraded",
                         exc_info=True)

    def publish_acoustic(self, payload: Dict[str, Any]) -> None:
        """Tell attached cockpits the microphone has gone bad.

        Carries the verdict WHOLE — diagnosis, crest, device — because an
        operator seeing "degraded" with no cause cannot tell a dropped
        headset from a noisy room, and those have different fixes.
        """
        try:
            if not self._clients:
                return
            self.stats["acoustic_published"] = (
                self.stats.get("acoustic_published", 0) + 1
            )
            self._dispatch({
                "type": "acoustic", "ts": time.time(),
                "diagnosis": str(payload.get("diagnosis", "") or ""),
                "crest_db": float(payload.get("crest_db", 0.0) or 0.0),
                "device": str(payload.get("device", "") or ""),
                "spoken": str(payload.get("spoken", "") or ""),
            })
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] acoustic publish degraded",
                         exc_info=True)

    def publish_prompt_resolved(self, prompt_id: str) -> None:
        """The gate is closed — answered here, elsewhere, or expired.

        Without this a deferred gate would sit in a cockpit's queue and be
        offered to the operator long after the organism stopped waiting. The
        queue drops it on pop by deadline too, but only this says *why* and
        says it immediately.
        """
        try:
            prompt_id = str(prompt_id or "").strip()
            if not prompt_id or not self._clients:
                return
            self._dispatch({
                "type": "prompt_resolved", "prompt_id": prompt_id,
                "ts": time.time(),
            })
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] prompt-resolved degraded",
                         exc_info=True)

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        """Hop to the server loop and broadcast. The cross-loop dance every
        publisher above repeats by hand — written once so a new frame type
        cannot get it subtly wrong."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._broadcast(msg)
        else:
            loop.call_soon_threadsafe(self._broadcast, msg)

    def publish_audio_state(self, state: str) -> None:
        """Stream one audio-FSM state to every attached terminal
        (edge-coalesced: republishing the current state is a no-op).
        The state is also retained for hydration so a late joiner's
        footer is never stale. Thread-safe; NEVER raises."""
        try:
            state = str(state or "").strip().upper()
            if state not in AUDIO_STATES or state == self._audio_state:
                return
            self._audio_state = state
            self.stats["audio_states_published"] += 1
            msg = {"type": "audio_state", "state": state, "ts": time.time()}
            loop = self._loop
            if loop is None or loop.is_closed() or not self._clients:
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._broadcast(msg)
            else:
                loop.call_soon_threadsafe(self._broadcast, msg)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] audio publish degraded", exc_info=True)

    def publish_thermal(self, state: str) -> None:
        """Sovereign Governor lane: thermal posture to every attached
        client (jarvis topology + ov). Thread-safe; NEVER raises."""
        try:
            msg = {"type": "thermal", "state": str(state), "ts": time.time()}
            loop = self._loop
            if loop is None or loop.is_closed() or not self._clients:
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._broadcast(msg)
            else:
                loop.call_soon_threadsafe(self._broadcast, msg)
        except Exception:  # noqa: BLE001
            pass

    def publish_telemetry(self, payload: Dict[str, Any]) -> None:
        """Control-plane lane: push ONE structured telemetry frame (DAG
        hydration / DoubleWord failover / Ouroboros Actor) to every attached
        client. Consumed by the ``ov system`` observability panel. Mirrors
        ``publish_thermal`` (thread-safe, cross-loop marshalled). NEVER raises."""
        try:
            if not isinstance(payload, dict):
                return
            msg: Dict[str, Any] = {"type": "telemetry", "ts": time.time()}
            msg.update(payload)
            loop = self._loop
            if loop is None or loop.is_closed() or not self._clients:
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._broadcast(msg)
            else:
                loop.call_soon_threadsafe(self._broadcast, msg)
        except Exception:  # noqa: BLE001
            pass

    def _broadcast(self, msg: Dict[str, Any]) -> None:
        """Deliver *msg* to the cockpit that asked, or to all of them.

        The addressing decision is made HERE rather than at each call site,
        because the ~15 render frames between a verb dispatch and this method
        have no business knowing about IPC sessions. They set a ContextVar on
        the way in; this reads it on the way out.

        An addressed message whose cockpit has since detached is DROPPED, not
        broadcast. Falling back to broadcast would mean a reconnecting
        operator's private verb output appears in someone else's scrollback
        precisely when the intended reader is gone — the failure mode this
        routing exists to prevent, arriving through its own error path."""
        target = None
        try:
            from backend.core.ouroboros.battle_test.attach_session import (
                current_session,
            )
            target = current_session()
        except Exception:  # noqa: BLE001 — routing must never eat output
            target = None

        if target is not None:
            w = self._sessions.get(target)
            if w is None:
                self.stats["addressed_undeliverable"] = (
                    self.stats.get("addressed_undeliverable", 0) + 1
                )
                return
            writers = [w]
            self.stats["addressed"] = self.stats.get("addressed", 0) + 1
        else:
            writers = list(self._clients)

        # Tell the client WHICH kind of output this is. It already knows here
        # — the ContextVar decided it one line up — and the client cannot
        # re-derive it, because "was this answering my command?" is not a
        # property of the text. Addressed output belongs in the scrollback the
        # operator asked for; ambient output belongs in the live deck that
        # ages out. Without the marker the client would have to guess, and
        # guessing puts provider failovers in the transcript and command
        # answers in a region that erases them.
        msg = dict(msg)
        msg["addressed"] = target is not None

        # LANE TAGGING — the same interceptor, one more ContextVar.
        #
        # Deliberately not a second interception layer: session and lane are
        # read from the same execution context at the same seam, because both
        # answer questions only the emitter's context can answer ("who asked?"
        # and "who is doing it?"). Splitting them would mean two places to
        # keep in step.
        #
        # The ring is written HERE rather than at the producer, so every
        # producer is covered without any producer knowing lanes exist —
        # including ones written later.
        try:
            from backend.core.ouroboros.battle_test.attach_session import (
                AMBIENT_LANE,
                current_lane,
            )
            lane = current_lane()
            if lane and lane != AMBIENT_LANE:
                msg["lane"] = lane
                body = msg.get("text")
                if isinstance(body, str) and body:
                    from backend.core.ouroboros.battle_test.lane_rings import (
                        get_lane_registry,
                    )
                    get_lane_registry().record(lane, body)
        except Exception:  # noqa: BLE001 — tagging must never eat output
            pass

        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        for w in writers:
            try:
                if w.is_closing():
                    self._drop(w)
                    continue
                w.write(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._drop(w)
            except Exception:  # noqa: BLE001 — ANY writer fault = drop
                self._drop(w)

    def _serve_lane_history(
        self, lane: str, session: Optional[str], *, limit: Any = None,
    ) -> None:
        """Answer one lane-history request. NEVER raises.

        A lane that no longer exists gets an EMPTY history with
        ``found: false`` rather than silence. The client can then say "that
        worker's output has aged out" instead of rendering a blank pane the
        operator reads as "it produced nothing" — the two are very different
        facts and the pane must not conflate them."""
        try:
            lane = str(lane or "").strip()
            if not lane:
                return
            from backend.core.ouroboros.battle_test.lane_rings import (
                get_lane_registry,
            )
            reg = get_lane_registry()
            try:
                n = int(limit) if limit is not None else None
            except (TypeError, ValueError):
                n = None
            lines = reg.history(lane, limit=n)
            payload = {
                "type": "lane_history",
                "lane": lane,
                "found": bool(lines) or reg.is_tombstoned(lane),
                "tombstoned": reg.is_tombstoned(lane),
                "dropped": reg.dropped(lane),
                "lines": [ln.text for ln in lines],
                "ts": time.time(),
            }
            # Addressed to the requester alone, reusing the session scope the
            # bridge already routes on — no second addressing mechanism.
            from backend.core.ouroboros.battle_test.attach_session import (
                session_scope,
            )
            loop = self._loop
            if loop is None or loop.is_closed():
                return

            def _emit() -> None:
                with session_scope(session):
                    self._broadcast(payload)

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _emit()
            else:
                loop.call_soon_threadsafe(_emit)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] lane history degraded", exc_info=True)

    def announce_lane_reaped(self, lane: str) -> None:
        """Tell every cockpit a lane no longer exists. NEVER raises.

        Broadcast, not addressed: any terminal could be focused on it. This
        is the one lane message that is genuinely everyone's business."""
        try:
            if not self._clients:
                return
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            msg = {"type": "lane_reaped", "lane": str(lane), "ts": time.time()}

            def _emit() -> None:
                from backend.core.ouroboros.battle_test.attach_session import (
                    session_scope,
                )
                with session_scope(None):     # ambient — reaches all cockpits
                    self._broadcast(msg)

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _emit()
            else:
                loop.call_soon_threadsafe(_emit)
        except Exception:  # noqa: BLE001
            logger.debug("[CockpitAttach] reap announce degraded", exc_info=True)

    def bind_session(self, session_id: str, w: asyncio.StreamWriter) -> None:
        """Associate a cockpit's declared identity with its socket.

        Called on the first frame that carries one. Idempotent, and a
        reconnecting client that reuses an id simply re-points it."""
        if not session_id:
            return
        self._sessions[session_id] = w
        self.stats["sessions_bound"] = self.stats.get("sessions_bound", 0) + 1

    def _drop(self, w: asyncio.StreamWriter) -> None:
        if w in self._clients:
            self.stats["dropped"] += 1
        self._clients.discard(w)
        self._publish_presence()
        for sid, sw in list(self._sessions.items()):
            if sw is w:
                del self._sessions[sid]
                # Safe resumption: a cockpit that dies with its rewind
                # menu open must release its autonomy hold NOW, not at
                # TTL expiry — a SIGKILL'd terminal cannot send resume.
                if sid in self._autonomy_holds:
                    self._autonomy_ctl("resume", sid)
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- per-client session ----

    async def _on_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            hydration = (
                json.dumps(self._hydration_payload(), separators=(",", ":"))
                + "\n"
            ).encode()
            writer.write(hydration)
            await writer.drain()
            # Slice H — Atomic State Flush: yield the buffered critical
            # telemetry history to THIS client BEFORE it joins the live
            # broadcast set, so historical always precedes live (no interleave).
            try:
                for frame in (self._replay() or []):
                    writer.write((json.dumps(frame, separators=(",", ":"))
                                  + "\n").encode())
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._drop(writer)
                return
            except Exception:  # noqa: BLE001
                pass
            # The render backlog rides the SAME pre-join flush, immediately
            # after the control-plane replay above. Two ordered groups rather
            # than one timestamp-merged stream: they are different planes, the
            # existing contract already promises telemetry-before-live, and
            # merging would have to invent an ordering for `_replay()` frames
            # that carry no comparable clock.
            #
            # Placement before `_clients.add(writer)` is what makes "history
            # always precedes live" true for this channel too -- a frame
            # published a microsecond later goes to the live set, and the
            # backlog is already on the wire ahead of it.
            _backlog_frames: List[Dict[str, Any]] = []
            try:
                _backlog_frames = self._backlog.peek()
                for frame in _backlog_frames:
                    writer.write((json.dumps(frame, separators=(",", ":"))
                                  + "\n").encode())
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The socket died mid-flush: this cockpit received nothing we
                # can rely on. NOT committed -- the next attach still gets the
                # backlog, because a client that failed to read it must not be
                # able to destroy it on everyone else's behalf.
                self._drop(writer)
                return
            except Exception:  # noqa: BLE001
                _backlog_frames = []
            else:
                # Provably on the wire -> discard exactly what was sent.
                # Counted by length rather than cleared wholesale, so a frame
                # retained DURING the flush survives to the next attach.
                if _backlog_frames:
                    self._backlog.commit(len(_backlog_frames))
                    logger.info(
                        "[CockpitAttach] replayed %d unheard render frame(s) "
                        "to a reattaching cockpit", len(_backlog_frames),
                    )
            self._clients.add(writer)
            # Attention begins where hydration LANDED, not where the
            # socket connected: a client that never received the payload
            # never saw the pending review, and must not be counted as
            # having looked at it.
            self._publish_presence()
            self.stats["connects"] += 1
            logger.info(
                "[CockpitAttach] terminal attached (subscribers=%d)",
                len(self._clients),
            )
            # Upstream read-loop: operator input frames. EOF/reset =
            # detach; a malformed frame never kills the session.
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                except (ValueError, TypeError):
                    continue
                ftype = frame.get("type")
                # Any frame may declare who is speaking. Bound before the
                # frame is acted on, so the very first command a cockpit
                # sends is already addressable when its output returns.
                _sid = str(frame.get("session", "")).strip()
                if _sid:
                    self.bind_session(_sid, writer)
                if ftype == "caps":
                    # The display describing itself. Handled BEFORE `input`
                    # because a cockpit declares at handshake and again on
                    # every SIGWINCH — the very first render addressed to it
                    # must already know its width, or the operator's opening
                    # frame is the one that wraps.
                    #
                    # Fail-soft and per-client: a malformed declaration is
                    # dropped and the composite continues to answer for this
                    # subscriber. One bad cockpit cannot narrow the others.
                    try:
                        from backend.core.ouroboros.battle_test.terminal_capabilities import (  # noqa: E501
                            TerminalCapabilities,
                            declare,
                        )
                        _caps = TerminalCapabilities.from_wire(frame)
                        if _caps is not None and _sid:
                            declare(_sid, _caps)
                    except Exception:  # noqa: BLE001
                        pass
                elif ftype == "input":
                    text = str(frame.get("text", "")).strip()
                    if text:
                        self.stats["inputs_received"] += 1
                        # Which gate this answers, when the cockpit says so.
                        # Only forwarded to a sink that asked for it by name.
                        _kw = {}
                        _pid = str(frame.get("prompt_id", "")).strip()
                        if _pid and self._input_takes_prompt_id:
                            _kw["prompt_id"] = _pid
                        try:
                            if self._input_takes_session:
                                self._on_input(text, _sid or None, **_kw)
                            else:
                                self._on_input(text, **_kw)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "[CockpitAttach] input sink degraded",
                                exc_info=True,
                            )
                        # Distributed history: the daemon already holds
                        # this line WITH its originating session — the
                        # fan-out is derived HERE rather than demanding a
                        # second frame for the same keystroke.
                        self._sync_history(text, _sid or None)
                elif ftype == "history_append":
                    # A line the CLIENT handled locally (/deck, /keys) —
                    # never dispatched, but every other terminal must
                    # still recall it.
                    text = str(frame.get("text", "")).strip()
                    if text:
                        self._sync_history(text, _sid or None)
                elif ftype == "autonomy":
                    # The Transactional Viewport Lock: pause on menu
                    # open, resume on close/rollback — refcounted across
                    # panes, TTL'd against wedged holders.
                    self._autonomy_ctl(
                        str(frame.get("action", "")).strip().lower(),
                        _sid or None,
                    )
                elif ftype == "rewind_list":
                    # Esc-Esc menu hydration — answered only to the
                    # asking cockpit, from the locked snapshot.
                    self._serve_rewind_list(
                        _sid or None, limit=frame.get("limit"),
                    )
                elif ftype == "lane":
                    # Hydration request: the client focused a lane and wants
                    # its history. Answered ONLY to the asking cockpit —
                    # another terminal's focus is not this one's business —
                    # by addressing the reply to the session on the frame.
                    self._serve_lane_history(
                        str(frame.get("lane", "")), _sid or None,
                        limit=frame.get("limit"),
                    )
                elif ftype == "audio":
                    cmd = str(frame.get("cmd", "")).strip().lower()
                    if cmd in AUDIO_CMDS:
                        self.stats["audio_cmds"] += 1
                        try:
                            self._on_audio(cmd)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "[CockpitAttach] audio sink degraded",
                                exc_info=True,
                            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                     # SIGKILL'd terminal — just a detach
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._drop(writer)
            # Retire this display's declaration. Ambient renders take the
            # MINIMUM width across live subscribers, so a departed 40-column
            # terminal would otherwise squeeze every future broadcast for the
            # life of the daemon — a dead cockpit constraining live ones.
            try:
                from backend.core.ouroboros.battle_test.terminal_capabilities import (  # noqa: E501
                    forget,
                )
                for _sid, _w in list(getattr(self, "_sessions", {}).items()):
                    if _w is writer:
                        forget(_sid)
            except Exception:  # noqa: BLE001
                pass
            logger.info(
                "[CockpitAttach] terminal detached (subscribers=%d)",
                len(self._clients),
            )


# ---------------------------------------------------------------------------
# Client side — the dumb terminal (ov attach)
# ---------------------------------------------------------------------------


class CockpitAttachClient:
    """Render-only subscriber + input pipe for ``ov attach``.

    ``connect()`` shares one bounded deadline across connect + hydration
    read; a missing/refused/dead socket returns False (the CLI renders
    "no organism awake"). NEVER raises."""

    def __init__(
        self,
        *,
        on_hydration: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_markup: Optional[Callable[[str], None]] = None,
        on_telemetry: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_prompt: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_transcript: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_acoustic: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_prompt_resolved: Optional[Callable[[str], None]] = None,
        on_audio_state: Optional[Callable[[str], None]] = None,
        on_thermal: Optional[Callable[[str], None]] = None,
        on_lane_history: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_lane_reaped: Optional[Callable[[str], None]] = None,
        on_history_append: Optional[Callable[[str], None]] = None,
        on_autonomy_state: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_rewind_list: Optional[Callable[[Dict[str, Any]], None]] = None,
        path: Optional[Path] = None,
    ) -> None:
        self._path = Path(path) if path is not None else attach_socket_path()
        self._on_hydration = on_hydration or (lambda _m: None)
        self._on_line = on_line or (lambda _t: None)
        # Typed styled channel: daemon-composed tool blocks / diffs. When
        # unset, markup frames degrade to the on_line callback (rendered
        # escaped by conservative clients — never dropped).
        self._on_markup = on_markup
        self._on_telemetry = on_telemetry or (lambda _m: None)
        self._on_prompt = on_prompt
        self._on_transcript = on_transcript
        self._on_acoustic = on_acoustic
        self._on_prompt_resolved = on_prompt_resolved
        self._on_audio_state = on_audio_state or (lambda _s: None)
        self._on_thermal = on_thermal or (lambda _s: None)
        #: Focused-lane hydration payloads (D3). Absent handler = the client
        #: does not use selection; the frame is simply not dispatched.
        self._on_lane_history = on_lane_history or (lambda _p: None)
        #: A lane stopped existing. The FSM cannot infer this from an absent
        #: heartbeat row — that is indistinguishable from a slow frame.
        self._on_lane_reaped = on_lane_reaped or (lambda _l: None)
        #: Another terminal's line, for Up-arrow parity. Absent handler =
        #: a pre-sync cockpit; the frame is simply not dispatched.
        self._on_history_append = on_history_append
        #: The viewport lock's truth (paused/holders) — every pane shows
        #: the freeze, whoever caused it.
        self._on_autonomy_state = on_autonomy_state
        #: The Esc-Esc menu's hydration payload (locked restore points).
        self._on_rewind_list = on_rewind_list
        #: This cockpit's identity for the life of the attachment. Declared
        #: on every outbound frame so the daemon can address answers back to
        #: the terminal that asked, instead of to all of them.
        try:
            from backend.core.ouroboros.battle_test.attach_session import (
                new_session_id,
            )
            self.session_id: str = new_session_id()
        except Exception:  # noqa: BLE001 — an unidentified client still works
            self.session_id = ""
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._read_task: Optional[asyncio.Task] = None
        self.connected: bool = False

    async def connect(self) -> bool:
        """Escalating-patience attach. A fresh organism's post-boot storm
        (sensor fan-out, executor warm-up) can lag hydration past a single
        0.5s shot — which misdiagnosed a LIVE organism as absent (the
        2026-07-23 attach-after-successful-boot failure). Timeout-class
        failures retry with doubling bounds up to a total patience budget;
        REFUSED/absent fails fast (waiting cannot conjure a dead listener).
        NEVER raises."""
        patience = _connect_patience_s()
        bound = _connect_timeout_s()
        spent = 0.0
        while True:
            t0 = time.monotonic()
            outcome = await self._connect_once(bound)
            if outcome == "ok":
                return True
            if outcome == "dead":
                return False
            spent += time.monotonic() - t0
            if spent >= patience:
                return False
            bound = min(bound * 2, max(0.5, patience - spent))

    async def _connect_once(self, timeout: float) -> str:
        """One bounded attach attempt: ``"ok"`` / ``"slow"`` (timeout-class
        — a starved loop, worth retrying) / ``"dead"`` (refused/absent —
        retrying cannot help). NEVER raises."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path=str(self._path)),
                timeout=timeout,
            )
            line = await asyncio.wait_for(
                self._reader.readline(), timeout=timeout,
            )
            if not line:
                await self.close()
                return "dead"
            frame = json.loads(line)
            if frame.get("type") == "hydration":
                self._safe_cb(self._on_hydration, frame)
            self._read_task = asyncio.get_running_loop().create_task(
                self._read_loop(),
            )
            self.connected = True
            # Declare this display IMMEDIATELY. The daemon renders for a
            # terminal it cannot see, so the very first frame it sends back
            # must already be sized correctly — a cockpit that declares late
            # has its opening render wrapped.
            try:
                self.send_caps()
                self.install_resize_listener()
            except Exception:  # noqa: BLE001
                pass
            return "ok"
        except asyncio.TimeoutError:
            await self.close()
            return "slow"
        except (FileNotFoundError, ConnectionRefusedError,
                ConnectionResetError):
            await self.close()
            return "dead"
        except Exception:  # noqa: BLE001
            await self.close()
            return "dead"

    @staticmethod
    def _wcwidth_available() -> bool:
        """True iff this client can reason about double-width glyphs.

        Presence of `wcwidth` is the honest proxy: with it, the client's
        renderer and the daemon's agree on how many columns an emoji occupies;
        without it, both are guessing and the safe report is narrow.
        """
        try:
            import wcwidth  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_caps(self) -> bool:
        """Tell the daemon what this display actually is. NEVER raises.

        The daemon renders every table, diff and gutter for a terminal it
        cannot see. This is the only frame that closes that gap, so it is sent
        at handshake AND on every resize — a cockpit that declares once and
        then gets dragged wider spends the rest of its life being rendered for
        a window that no longer exists.

        Measured, never assumed: width/height come from `os.get_terminal_size`
        on the real TTY. `COLORFTERM`/`TERM` answer colour depth. Theme comes
        from `COLORFGBG` when the terminal sets it and is reported "unknown"
        otherwise — a guess here paints grey on white for every light-terminal
        operator, and "unknown" routes the renderer to its theme-agnostic path
        instead.
        """
        try:
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False

            cols = rows = 0
            try:
                size = os.get_terminal_size()
                cols, rows = int(size.columns), int(size.lines)
            except Exception:  # noqa: BLE001 — not a TTY (piped `ov`)
                cols = rows = 0
            if cols <= 0:
                return False           # nothing measured → declare nothing

            # COLORFGBG is "fg;bg"; bg 0-6 and 8 are dark, 7 and 9-15 light.
            theme = "unknown"
            try:
                raw = os.environ.get("COLORFGBG", "")
                if ";" in raw:
                    bg = int(raw.rsplit(";", 1)[-1].strip())
                    theme = "dark" if (bg <= 6 or bg == 8) else "light"
            except (TypeError, ValueError):
                theme = "unknown"

            depth = 0
            _term = os.environ.get("TERM", "")
            if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
                depth = 16_777_216
            elif "256" in _term:
                depth = 256
            elif _term:
                depth = 8

            frame = {
                "type": "caps",
                "session": self.session_id,
                "cols": cols, "rows": rows,
                "theme": theme,
                # Terminals disagree about emoji/CJK advance width and only
                # the client can answer. `wcwidth` present → trust the
                # double-width prediction; absent → report narrow, because an
                # aligned ASCII gutter beats a misaligned pretty one.
                "wide_glyphs": self._wcwidth_available(),
                "color_depth": depth,
            }
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def install_resize_listener(self) -> bool:
        """Re-declare on SIGWINCH. NEVER raises; False where unsupported.

        Chained, not replaced: another handler may already own SIGWINCH
        (prompt_toolkit installs one to reflow its own layout), and clobbering
        it would fix the daemon's width while breaking the client's.
        """
        try:
            import signal
            if not hasattr(signal, "SIGWINCH"):
                return False            # Windows / no job control
            prior = signal.getsignal(signal.SIGWINCH)

            def _on_resize(signum, frame) -> None:
                try:
                    self.send_caps()
                except Exception:  # noqa: BLE001
                    pass
                if callable(prior):
                    try:
                        prior(signum, frame)
                    except Exception:  # noqa: BLE001
                        pass

            signal.signal(signal.SIGWINCH, _on_resize)
            return True
        except (ValueError, OSError, AttributeError):
            # `signal.signal` off the main thread raises ValueError — a real
            # scenario for an embedded cockpit, and not an error.
            return False

    def send_audio(self, cmd: str) -> bool:
        """Pipe one audio-orchestration command upstream (``wake`` /
        ``sleep`` / ``barge``). Non-blocking; False when detached or
        the command is outside the closed taxonomy. NEVER raises."""
        try:
            cmd = str(cmd or "").strip().lower()
            if cmd not in AUDIO_CMDS:
                return False
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False
            frame = {"type": "audio", "cmd": cmd}
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_lane(self, lane: str, limit: Optional[int] = None) -> bool:
        """Ask the daemon for one lane's history. NEVER raises."""
        try:
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False
            frame: Dict[str, Any] = {
                "type": "lane", "lane": str(lane),
                "session": self.session_id,
            }
            if limit is not None:
                frame["limit"] = int(limit)
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_autonomy(self, action: str) -> bool:
        """Acquire ("pause") or release ("resume") this cockpit's hold on
        the Transactional Viewport Lock. Non-blocking; False when
        detached or the action is outside the taxonomy. NEVER raises."""
        try:
            action = str(action or "").strip().lower()
            if action not in ("pause", "resume"):
                return False
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False
            frame = {"type": "autonomy", "action": action,
                     "session": self.session_id}
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_rewind_request(self, limit: int = 10) -> bool:
        """Ask the daemon for the locked restore-point list. NEVER
        raises."""
        try:
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False
            frame = {"type": "rewind_list", "limit": int(limit),
                     "session": self.session_id}
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_history(self, text: str) -> bool:
        """Report one CLIENT-HANDLED line upstream for history fan-out.

        Only for lines that never travel as ``input`` (client-local verbs
        like ``/deck``) — a line that crosses as input is fanned out from
        that seam, and sending both would double it everywhere else.
        Non-blocking; False when detached. NEVER raises."""
        try:
            entry = str(text or "").strip()
            if not entry:
                return False
            w = self._writer
            if not self.connected or w is None or w.is_closing():
                return False
            frame = {"type": "history_append", "text": entry,
                     "session": self.session_id}
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    def send_input(self, text: str, prompt_id: Optional[str] = None) -> bool:
        """Pipe one operator line upstream. Non-blocking; False when
        detached. NEVER raises."""
        try:
            w = self._writer
            # ``connected`` is the session truth (the read-loop flips it
            # on EOF/daemon-exit); the writer's own is_closing() lags a
            # dead peer — a detached pipe must refuse input immediately.
            if not self.connected or w is None or w.is_closing():
                return False
            # Declare who is asking, so the daemon can address the answer
            # back rather than broadcasting it to every attached cockpit.
            frame = {"type": "input", "text": str(text),
                     "session": self.session_id}
            if prompt_id:
                # Tag the answer with the gate it was written for. A deferred
                # verdict can outlive the slot it was meant for, and the
                # daemon refuses a mismatch rather than landing "y" on
                # whichever op happens to be armed now.
                frame["prompt_id"] = str(prompt_id)
            w.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            return True
        except Exception:  # noqa: BLE001
            return False

    async def _read_loop(self) -> None:
        reader = self._reader
        if reader is None:
            return
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                except (ValueError, TypeError):
                    continue
                ftype = frame.get("type")
                if ftype == "line":
                    self._safe_cb_text(self._on_line, str(frame.get("text", "")))
                elif ftype == "markup":
                    # Daemon-composed styled line. No on_markup handler →
                    # degrade to on_line (conservative clients escape it).
                    cb = self._on_markup or self._on_line
                    # Forward the addressed/ambient marker to sinks that want
                    # it, using the same arity inspection the input path uses
                    # rather than a second convention. A one-argument sink —
                    # every pre-deck client — is called exactly as before.
                    if _accepts_two_positional(cb):
                        try:
                            cb(str(frame.get("text", "")),
                               bool(frame.get("addressed", False)))
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        self._safe_cb_text(cb, str(frame.get("text", "")))
                elif ftype == "autonomy_state":
                    if self._on_autonomy_state is not None:
                        try:
                            self._on_autonomy_state(dict(frame))
                        except Exception:  # noqa: BLE001
                            pass
                elif ftype == "rewind_list":
                    if self._on_rewind_list is not None:
                        try:
                            self._on_rewind_list(dict(frame))
                        except Exception:  # noqa: BLE001
                            pass
                elif ftype == "history_append":
                    # Another terminal's line. Drop our own echo — the
                    # daemon excludes the originator by session binding,
                    # but a rebound id must not double an entry here.
                    if self._on_history_append is not None:
                        _origin = str(frame.get("origin", "") or "")
                        if not (_origin and _origin == self.session_id):
                            self._safe_cb_text(
                                self._on_history_append,
                                str(frame.get("text", "")),
                            )
                elif ftype == "prompt":
                    # An open gate, as data. A client with no handler is a
                    # pre-shield cockpit: it already received the same
                    # question as a markup line, so silence here is correct
                    # and it behaves exactly as it did before.
                    if self._on_prompt is not None:
                        try:
                            self._on_prompt(dict(frame))
                        except Exception:  # noqa: BLE001
                            pass
                elif ftype == "transcript":
                    # A cockpit with no handler is pre-dictation: it already
                    # gets the finished utterance the way it always did.
                    if self._on_transcript is not None:
                        try:
                            self._on_transcript(dict(frame))
                        except Exception:  # noqa: BLE001
                            pass
                elif ftype == "acoustic":
                    if self._on_acoustic is not None:
                        try:
                            self._on_acoustic(dict(frame))
                        except Exception:  # noqa: BLE001
                            pass
                elif ftype == "prompt_resolved":
                    if self._on_prompt_resolved is not None:
                        self._safe_cb_text(
                            self._on_prompt_resolved,
                            str(frame.get("prompt_id", "")),
                        )
                elif ftype == "telemetry":
                    try:
                        self._on_telemetry(dict(frame))
                    except Exception:  # noqa: BLE001
                        pass
                elif ftype == "audio_state":
                    self._safe_cb_text(
                        self._on_audio_state, str(frame.get("state", "")),
                    )
                elif ftype == "thermal":
                    self._safe_cb_text(
                        self._on_thermal, str(frame.get("state", "")),
                    )
                elif ftype == "lane_reaped":
                    self._safe_cb_text(
                        self._on_lane_reaped, str(frame.get("lane", "")),
                    )
                elif ftype == "lane_history":
                    try:
                        self._on_lane_history(dict(frame))
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.connected = False

    async def close(self) -> None:
        self.connected = False
        task = self._read_task
        self._read_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        w = self._writer
        self._writer = None
        self._reader = None
        if w is not None:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _safe_cb(cb: Callable[[Dict[str, Any]], None], m: Dict[str, Any]) -> None:
        try:
            cb(m)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _safe_cb_text(cb: Callable[[str], None], t: str) -> None:
        try:
            cb(t)
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "AUDIO_CMDS",
    "AUDIO_STATES",
    "COCKPIT_ATTACH_SCHEMA_VERSION",
    "CockpitAttachBridge",
    "CockpitAttachClient",
    "attach_enabled",
    "attach_socket_path",
]
