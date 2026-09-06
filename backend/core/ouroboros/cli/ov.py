"""backend/core/ouroboros/cli/ov.py -- the ``ov`` console entry point.

``ov`` is the packaged binary (PEP 621 ``[project.scripts]``). It is a thin
*dispatcher*: it translates subcommands into the legacy battle-test
bootstrap's argv and delegates, so it never re-parses the real flags -- the
single source of truth for arguments stays in
``scripts/ouroboros_battle_test.py`` (DRY, spec §4.3).

Subcommands::

    ov                 boot the organism + live cockpit (default)
    ov run [flags]     headless autonomous session  (-> --headless)
    ov daemon [flags]  alias for a headless run
    ov status          last-session digest (no boot)
    ov attach          attach to a running organism (hydrated live view + input)
    ov help            usage

Everything after the verb forwards verbatim to the bootstrap, e.g.
``ov run --cost-cap 2.00 -v`` -> ``main(["--headless", "--cost-cap", "2.00", "-v"])``.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from backend.core.ouroboros.ui.theme import build_console

_VERBS = {"cockpit", "run", "daemon", "status", "attach", "system", "hive",
          "link",
          "doctor", "demo", "restart", "version"}
_HELP_TOKENS = {"help", "--help", "-h"}
_VERSION_TOKENS = {"version", "--version", "-V"}

#: Milestone name — paired with the pyproject version at render time.
#: Minted per release; the number itself is NEVER duplicated here.
RELEASE_NAME = "unchained"


def resolve_version() -> str:
    """``0.1.0`` — dynamically from installed metadata, falling back to
    the repo's pyproject.toml (editable/dev checkouts), then to the
    honest ``0.0.0+unknown``. NEVER raises."""
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("ouroboros-ov")
    except Exception:
        pass
    try:
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parents[4]
        data = tomllib.loads((root / "pyproject.toml").read_text())
        v = str(data.get("project", {}).get("version", "")).strip()
        if v:
            return v
    except Exception:
        pass
    return "0.0.0+unknown"


from backend.core.ouroboros.ui.alt_screen import alternate_screen

from backend.core.ouroboros.ui.semantic_tokens import (  # noqa: E402
    role_palette as _role_palette,
)

#: Semantic colour roles — the SAME name and access pattern as every
#: other module. One vocabulary, one spelling, one owner.
_SEM = _role_palette()

#: `ov` referenced a bare `logger` in `_client_extra_bindings` while no such
#: name existed. The call raised NameError, the surrounding handler
#: referenced `logger` AGAIN while handling it, and the outer
#: `except Exception: return None` discarded the ENTIRE extra key-binding
#: set — confirm actions, the completion arbiter, paste collapse, rewind,
#: transcript hatches and mode. All built, all mounted, all thrown away one
#: line later. A comment at the audio scope already carried the diagnosis
#: ("`ov` has no module-level logger ... a bare `logger` here resolved to
#: nothing but a swallowed NameError") and worked around it locally rather
#: than declaring one. Stdlib only, per this module's import-isolation
#: mandate.
logger = logging.getLogger("Ouroboros.Ov")


def version_line() -> str:
    """``ov 0.1.0 “unchained” — ouroboros + venom``. NEVER raises."""
    try:
        return f"ov {resolve_version()} “{RELEASE_NAME}” — ouroboros + venom"
    except Exception:
        return "ov — ouroboros + venom"

#: Operator words the CLIENT handles itself, mapped to the audio command they
#: fire. Module scope on purpose: ``_route_operator_line`` dispatches from this
#: table and the slash palette ENUMERATES it, so a verb cannot exist in one and
#: be missing from the other. A second list for the menu is exactly how a
#: palette starts lying about what the CLI accepts.
AUDIO_VERBS = {
    "wake": "wake", "voice": "wake", "listen": "wake",
    "wake!": "force_wake", "force-wake": "force_wake",
    "force wake": "force_wake",
    "ptt": "ptt",
    "ptt stop": "ptt_stop", "ptt-stop": "ptt_stop", "ptt off": "ptt_stop",
    "flush": "flush", "shh": "flush", "hush": "flush",
    "mute": "sleep", "sleep": "sleep",
    "barge": "barge",
}

#: One line per client verb for the palette's ``display_meta``. Keys that are
#: absent fall back to a generated description, so adding to AUDIO_VERBS can
#: never break the menu — it just reads less well until documented here.
CLIENT_VERB_HELP = {
    "wake": "arm Karen's microphone",
    "sleep": "disarm the microphone",
    "barge": "interrupt Karen mid-sentence",
    "flush": "halt outbound audio now (ducking)",
    "ptt": "hold push-to-talk open",
    "ptt stop": "close the push-to-talk hold",
    "force-wake": "seize the mic from another terminal",
    "deck": "deck height — off | compact | full",
    "tasks": "show/hide the running-subagent roster",
    "keys": "the bindings this terminal answers to",
    "detach": "leave; the organism keeps running",
}


def _alias_help(verb: str) -> str:
    """Help for an audio verb with no entry of its own. NEVER raises.

    The old fallback rendered ``f"audio: {AUDIO_VERBS[verb]}"``, so six rows of
    the palette read "audio: force_wake", "audio: ptt_stop", "audio: flush" —
    an internal action identifier presented as a description. It answers
    nothing, and it is the one thing the palette must never do: look like help.

    But the map already carries the answer. ``AUDIO_VERBS`` is a SYNONYM table
    — ``wake!``, ``force-wake`` and ``force wake`` all resolve to
    ``force_wake`` — so a verb without help has siblings, and at least one of
    them is documented. Saying "alias of /force-wake" is both true and more
    useful than a fresh sentence would be, because it tells the operator these
    are the same word rather than three things to learn.

    Derived from the routing table itself, per this module's own rule that a
    verb "cannot exist in one and be missing from the other". A second
    transcription of which spellings are synonyms is exactly how a palette
    starts lying about what the CLI accepts.
    """
    try:
        action = AUDIO_VERBS.get(verb)
        if not action:
            return ""
        siblings = [v for v, a in AUDIO_VERBS.items()
                    if a == action and v != verb and CLIENT_VERB_HELP.get(v)]
        if not siblings:
            return ""
        # Canonical selection is `_canonical_of`'s job, not a second copy of
        # the same preference order. This text names a verb, and
        # `audio_alias_families` decides which verb the palette SHOWS — two
        # implementations that drifted would point the operator at a row that
        # had been folded away.
        canonical = _canonical_of(action, siblings)
        return f"alias of /{canonical} — {CLIENT_VERB_HELP[canonical]}"
    except Exception:  # noqa: BLE001 — a palette must never throw
        return ""


def audio_alias_families() -> "dict":
    """``{canonical verb: (synonyms, ...)}``, derived from AUDIO_VERBS.

    ``AUDIO_VERBS`` is a SYNONYM table — ``wake!``, ``force-wake`` and
    ``force wake`` all resolve to ``force_wake`` — and the palette enumerated
    its KEYS, so eleven rows carried four meanings::

        /flush      halt outbound audio now (ducking)
        /hush       alias of /flush — halt outbound audio now (ducking)
        /shh        alias of /flush — halt outbound audio now (ducking)

    Three rows for one capability, two of them spending their description
    saying so. `VerbDescriptor.aliases` exists precisely to express this and
    was empty on every row in the registry, while four consumers — typo
    suggestion, prefix matching, ``matches()`` and ``/verb --help`` — were
    already reading it.

    Sameness is DERIVED from the routing table rather than listed: two verbs
    are aliases exactly when they fire the same audio command. A hand-written
    list of families would be a fourth place to state a fact the table already
    holds, and the first one to go stale.

    Two subtleties that a naive "group by action" gets wrong:

    * ``voice`` and ``listen`` map to ``wake`` here AND have real daemon
      dispatchers with their own descriptions. They are separate verbs that
      happen to share an audio effect, so folding them would delete two
      capabilities from the palette. Deferred to `_daemon_owns`, the same
      precedence `_resolve_audio_verb` uses.
    * the canonical is chosen deterministically — the spelling that IS the
      action, else the first DOCUMENTED sibling in table order. An alias
      target that reshuffles between runs is worse than none.

    NEVER raises.
    """
    try:
        by_action: "dict" = {}
        for verb, action in AUDIO_VERBS.items():
            if _daemon_owns(verb):
                continue        # a real verb that merely shares an effect
            by_action.setdefault(action, []).append(verb)
        families: "dict" = {}
        for action, verbs in by_action.items():
            if len(verbs) < 2:
                continue
            canonical = _canonical_of(action, verbs)
            families[canonical] = tuple(v for v in verbs if v != canonical)
        return families
    except Exception:  # noqa: BLE001 — a palette must never throw
        return {}


def _canonical_of(action: str, verbs: "Sequence[str]") -> str:
    """Which spelling of a synonym group is the one to show. NEVER raises.

    ONE definition, shared with :func:`_alias_help` — the alias text says
    "alias of /X" and the palette shows row X, so if the two disagreed the
    menu would point at a verb it had also hidden.
    """
    try:
        exact = next((v for v in verbs if v == action), None)
        if exact:
            return exact
        documented = next((v for v in verbs if CLIENT_VERB_HELP.get(v)), None)
        return documented or verbs[0]
    except Exception:  # noqa: BLE001
        return verbs[0] if verbs else ""


def _daemon_owns(verb: str) -> bool:
    """Does the DAEMON dispatch this verb? NEVER raises.

    Fails CLOSED — an unreadable table answers True, so the client declines to
    intercept and the line goes where it goes today. A guard that guesses
    "mine" when it cannot tell would silently swallow verbs on a degraded
    boot, and swallowing is the harder failure to diagnose.
    """
    try:
        from backend.core.ouroboros.battle_test.repl_dispatch_registry import (
            _VERB_TO_DISPATCHER, registry_primed,
        )
        if not registry_primed():
            return True
        return verb in _VERB_TO_DISPATCHER
    except Exception:  # noqa: BLE001
        return True


def _resolve_audio_verb(low: str, audio_verbs: "dict") -> "Optional[str]":
    """The audio command for an operator line, slash form included.

    ``AUDIO_VERBS`` is keyed on BARE words — ``wake``, ``ptt stop``,
    ``force-wake`` — and the lookup was ``audio_verbs.get(low)`` where ``low``
    is whatever the operator typed. The palette enumerates the same table and
    renders every entry with a leading slash, so ``/wake`` was offered in the
    menu, missed the table on selection, and was relayed to a daemon that has
    no ``/wake`` dispatcher. The verb appeared in the palette and did nothing.

    The neighbouring branches show the shape of the bug: ``/deck``, ``/tasks``
    and ``/keys`` each test ``low == "deck" or low.startswith("/deck")`` by
    hand. Every client-handled verb was patched for slash forms one at a time,
    and the table lookup — which covers fifteen of them — never was.

    **Daemon wins on collision**, matching `registry_from_dispatch`, which
    adds a client verb only ``if slash not in known``. ``voice`` and ``listen``
    sit in ``AUDIO_VERBS`` *and* have real daemon dispatchers, so a blind
    strip would hijack ``/voice`` away from the verb the palette describes.
    Deferring to the same precedence means the menu and the router cannot
    disagree about who answers — which is the property this module already
    claims when it says a verb "cannot exist in one and be missing from the
    other". NEVER raises.
    """
    try:
        cmd = audio_verbs.get(low)
        if cmd is not None:
            return cmd               # bare form — unchanged
        if not low.startswith("/"):
            return None
        bare = low[1:].strip()
        if not bare or bare not in audio_verbs:
            return None
        return None if _daemon_owns(bare) else audio_verbs[bare]
    except Exception:  # noqa: BLE001
        return None


def client_verbs() -> "dict":
    """Verbs this cockpit answers WITHOUT the daemon. NEVER raises.

    Derived from the live dispatch tables rather than transcribed beside
    them: the audio words come from AUDIO_VERBS (the same object the router
    switches on) and the local-only verbs are listed once here."""
    out = {v: CLIENT_VERB_HELP.get(v) or _alias_help(v) for v in AUDIO_VERBS}
    # `keys` and `tasks` were routed in `_route_operator_line` but missing
    # here, so neither appeared in the `/` palette — routed and unreachable
    # unless you already knew the word. A verb the operator cannot discover
    # is a verb that does not exist.
    for v in ("deck", "tasks", "keys", "detach"):
        out[v] = CLIENT_VERB_HELP.get(v, "")
    return out


def _render_markup_frame(text: str, console: Any = None) -> None:
    """Render ONE daemon-composed styled line (the typed ``markup`` frame:
    CC-style ⏺/⎿ tool blocks + numbered diffs). Unlike the untyped ``line``
    frame (always escaped — inert DATA), markup frames carry daemon-authored
    styling whose MODEL-controlled content was escaped at composition
    (tool_render_view). Fail-soft: markup that does not parse renders
    ESCAPED rather than dropped or crashing the canvas. NEVER raises."""
    try:
        from rich.text import Text as _RichText
        from rich.markup import escape as _escape
        raw = str(text)
        try:
            _RichText.from_markup(raw)          # validate before trusting
            safe = raw
        except Exception:  # noqa: BLE001 — malformed → inert fallback
            safe = _escape(raw)
        from backend.core.ouroboros.battle_test.bipartite_layout import (
            get_active_canvas,
        )
        canvas = get_active_canvas()
        if canvas is not None:
            canvas.push_raw(safe)
            return
        # NON-DESTRUCTIVE INJECTION.
        #
        # A Rich Console binds sys.stdout at CONSTRUCTION. `patch_stdout`
        # swaps sys.stdout afterwards, so a console built before the prompt
        # started writes straight past the proxy and paints over the line the
        # operator is typing. `_print_line` already documents this — "a
        # pre-bound Rich console would bypass the patch and corrupt the input
        # line" — and then this branch did precisely that.
        #
        # It mattered little while markup carried only occasional op chrome.
        # It matters now: every Moltbook post and all 60 REPL verb results
        # arrive on this channel, unprompted, while the operator types.
        #
        # The fix is to bind LATE, not to build a redraw engine. A console
        # constructed against the CURRENT sys.stdout is the patched proxy, so
        # prompt_toolkit renders the line above the prompt and redraws the
        # input buffer intact — its own machinery, reused rather than
        # reimplemented. Width is inherited from the original console so the
        # proxy (not a tty) does not collapse to 80 columns.
        _emitted = False
        try:
            import sys as _sys

            from rich.console import Console as _Console
            _kw = {"file": _sys.stdout, "highlight": False}
            _width = getattr(console, "width", None)
            if isinstance(_width, int) and _width > 0:
                _kw["width"] = _width
            _late = _Console(**{k: v for k, v in _kw.items()
                                if k != "highlight"})
            _late.print(safe, highlight=False)
            _emitted = True
        except Exception:  # noqa: BLE001 — never lose the frame
            _emitted = False
        if not _emitted:
            if console is not None:
                console.print(safe, highlight=False)
            else:
                print(raw)
    except Exception:  # noqa: BLE001
        try:
            print(str(text))
        except Exception:  # noqa: BLE001
            pass


_NO_ORGANISM_MESSAGE = (
    "no organism awake — nothing to attach to. Start one with `ov` "
    "(cockpit) or `ov daemon` (headless)."
)

_HELP_TEXT = """ov -- Ouroboros + Venom, autonomous engineering organism

  ov                  instant cockpit — attach to the organism
                      (cold-boots one in the background if needed;
                      --legacy-boot forces the old in-process boot)
  ov run [flags]      headless autonomous session (foreground)
  ov daemon [flags]   alias for a headless run
  ov daemon --install    install the resident organism (launchd agent)
  ov daemon --uninstall  remove the resident organism
  ov status           last-session digest (no boot)
  ov doctor [--live]  8-edge connectivity matrix; --live fires the
                      trace-isolated synthetic tool probe end-to-end
  ov demo [scene]     watch the cockpit with synthetic events
  ov link ...         the Body/Engine bridge -- issue certs, serve, connect
                      (`ov link --help` for the full set)
  ov attach           attach this terminal to the running organism
  ov version          version + milestone
  ov help             this message

All flags after the verb forward to the battle-test bootstrap, e.g.
  ov run --cost-cap 2.00 -v
See `python3 scripts/ouroboros_battle_test.py --help` for the full flag set.
"""


@dataclass
class Invocation:
    """The resolved intent of an ``ov`` command line.

    ``action`` is one of ``cockpit`` / ``headless`` / ``status`` / ``attach``
    / ``help``. ``delegate_argv`` is the argv handed to the legacy bootstrap
    for the boot actions. ``message`` carries the notice for the attach stub.
    """

    action: str
    delegate_argv: List[str] = field(default_factory=list)
    message: str = ""


def resolve(argv: Optional[Sequence[str]] = None) -> Invocation:
    """Translate an ``ov`` argv into an :class:`Invocation`.

    Pure + side-effect free so the routing is fully unit-testable without
    booting the organism. Unknown leading flags (no verb) default to the
    cockpit with the flags forwarded verbatim.
    """
    tokens = list(argv or [])

    if tokens and tokens[0] in _HELP_TOKENS:
        return Invocation("help")

    if tokens and tokens[0] in _VERSION_TOKENS:
        return Invocation("version")

    if tokens and tokens[0] in _VERBS:
        verb, rest = tokens[0], list(tokens[1:])
    else:
        verb, rest = "cockpit", list(tokens)

    if verb == "daemon" and "--install" in rest:
        return Invocation("daemon_install")
    if verb == "daemon" and "--uninstall" in rest:
        return Invocation("daemon_uninstall")
    if verb in ("run", "daemon"):
        return Invocation("headless", ["--headless", *rest])
    if verb == "status":
        return Invocation("status")
    if verb == "attach":
        return Invocation("attach")
    if verb == "system":
        return Invocation("system")
    if verb == "hive":
        return Invocation("hive")
    if verb == "doctor":
        return Invocation("doctor", list(rest))
    if verb == "link":
        return Invocation("link", list(rest))
    if verb == "demo":
        return Invocation("demo", list(rest))
    if verb == "restart":
        return Invocation("restart", list(rest))
    # cockpit (explicit or defaulted)
    return Invocation("cockpit", rest)


# ---------------------------------------------------------------------------
# ov status
# ---------------------------------------------------------------------------


def _default_status_provider() -> Optional[str]:
    """Best-effort digest of recent sessions (authority-free, read-only).

    Reads :meth:`LastSessionSummary.operator_digest_sync` — the
    OPERATOR-plane surface, deliberately ungated by
    ``JARVIS_LAST_SESSION_SUMMARY_ENABLED`` (that flag governs the
    organism's prompt-injection authority, not a human's explicit query;
    routing status through the autonomy gate was the wired-but-inert
    root cause: sessions on disk, "no prior session found" on screen).
    Returns ``None`` when no parseable prior session exists -- callers
    render a friendly fallback. NEVER raises.
    """
    try:
        from backend.core.ouroboros.governance.last_session_summary import (
            get_default_summary,
        )

        text = get_default_summary().operator_digest_sync()
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except Exception:
        return None


def status_digest(provider: Optional[Callable[[], Optional[str]]] = None) -> str:
    """Return a one-line status string. NEVER raises.

    ``provider`` is injectable for tests; production uses
    :func:`_default_status_provider`.
    """
    try:
        p = provider or _default_status_provider
        line = p()
        if line:
            return line
        return "ov -- no prior session found"
    except Exception:
        return "ov -- status unavailable"


# ---------------------------------------------------------------------------
# ov attach — dumb terminal over the Cockpit Attach Bridge (CLI item #6)
# ---------------------------------------------------------------------------


#: The operator-plane glyphs, resolved ONCE through the design language.
#:
#: `ui/theme` ships six glyphs with an ASCII degradation each, and says why:
#: "so 16-color/none terminals keep identical geometry". This client had 75
#: hardcoded ⏺ / ⎿ / ⚠ / · and ZERO calls to `mark()`, so that degradation
#: never fired here. `supports_unicode()` is locale-driven — a non-UTF-8
#: `LANG` is ordinary over ssh, cron and CI — and on those terminals the
#: cockpit's whole glyph vocabulary rendered as mojibake or nothing.
#:
#: Resolved at call time rather than import time: a module-level constant
#: would freeze the locale of whichever process imported first, and the
#: attach client and daemon do not share one.
def _glyph(name: str, fallback: str) -> str:
    """One operator-plane glyph, ASCII-degraded when the locale demands.

    NEVER raises: a theme that will not import leaves the caller with the
    literal it already had, so this can only ever improve a terminal.
    """
    try:
        from backend.core.ouroboros.ui.theme import mark
        return mark(name) or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _tone(token: str, fallback: str = "") -> str:
    """A semantic style string for the CURRENT terminal tier.

    `ov.py` styles with raw ``rgb(...)`` literals, which `ui/theme` exists to
    replace — Token.MUTED resolves per tier, so a 16-color terminal gets a
    16-color answer instead of a truecolor sequence it renders as noise. The
    repo already ratchets on "raw colour literals only ever decrease".

    ``SUCCESS`` is deliberately NOT used for status here: the theme reserves
    it for OUTCOMES (apply/verify OK), and spending it on "attached" would
    make a connection look like an accomplishment.
    """
    try:
        from backend.core.ouroboros.ui.theme import Token, active_tier, style_for
        return style_for(Token(token), active_tier()) or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _render_hydration(console: Any, payload: dict) -> None:
    """Instant-state render — the operator NEVER stares at a blank
    screen waiting for the next FSM tick. Pure presentation; the daemon
    is the single source of truth. NEVER raises."""
    try:
        status = payload.get("status") or {}
        liq = payload.get("liquidity") or {}
        ops = payload.get("ops") or []
        phase = status.get("phase", "IDLE")
        detail = status.get("phase_detail", "")
        cost = status.get("cost_spent_usd", 0.0)
        budget = status.get("cost_budget_usd", 0.0)
        # VISUAL HIERARCHY, matching the boot banner three functions up.
        #
        # Every line here was `console.print(str, markup=False)` — flat, one
        # weight, no dim. The boot banner in this same file builds Rich `Text`
        # with per-span styles and its comment cites "the CC title grammar
        # ... exactly like Claude Code v2.1.218". Two standards, one screen:
        # the operator's first frame was the unstyled one.
        #
        # `Text.append(style=)` keeps `markup=False`'s guarantee — daemon
        # content is never parsed for `[...]` — while restoring the three
        # tiers the palette already defines: the ACTION reads first, its
        # metadata recedes, and the labels recede further than their values.
        from rich.text import Text as _T
        _dot = _glyph("dot", "·")
        _muted, _body = _tone("muted", "dim"), _tone("body", "")
        head = _T()
        head.append(f"{_glyph('action', '*')} ", style=_tone("accent", "cyan"))
        head.append("attached", style=_body)
        head.append("  phase ", style=_muted)
        head.append(str(phase), style=_body)
        if detail:
            head.append(f" {detail}", style=_muted)
        # WHICH MODEL IS GENERATING belongs on the first frame.
        #
        # This line gave a whole field to cost and none to the model. On a
        # local lane that is backwards: cost is structurally $0.00 against a
        # ceiling nothing can spend (see the two comments below, which
        # already fought to stop that number implying a balance), while the
        # model is the one thing that actually changes between sessions --
        # base vs fine-tuned adapter, and no way to tell them apart.
        #
        # Read through `_model_pin()`, the same accessor the generation lane
        # uses, so the banner cannot disagree with what actually answers.
        # Empty means no explicit pin (auto-select), and the field is then
        # omitted rather than guessed at: naming a model we are not certain
        # of is worse than naming none.
        try:
            from backend.core.ouroboros.governance.candidate_generator import (
                _model_pin as _pin,
            )
            _model = _pin()
        except Exception:  # noqa: BLE001 — a banner never breaks an attach
            _model = ""
        if _model:
            head.append(f"  {_dot}  model ", style=_muted)
            head.append(_model, style=_body)
        head.append(f"  {_dot}  cost ", style=_muted)
        head.append(f"${cost:.2f}", style=_body)
        head.append(f"/${budget:.2f}", style=_muted)
        console.print(head, highlight=False)
        # A CEILING WITHOUT ITS BASIS IS AN ARBITRARY CONSTANT.
        #
        # `$0.00/$0.71` prompted "where is the $0.71 coming from?" — a fair
        # question, because 0.71 is derived (p95 of the operator's own
        # recorded sessions x a headroom multiple) and nothing on screen said
        # so. The derivation already returns its basis; it was being dropped
        # between the spawner and here.
        # A CEILING IS NOT A BALANCE.
        #
        # `$0.00/$0.71` reads as "you have $0.71 to spend". It is not: 0.71 is
        # a POLICY CEILING derived from past sessions, and when every paid
        # lane is out of credit the spendable capacity behind it is zero. The
        # arithmetic was right and the meaning was wrong, which is why the
        # number "looked inaccurate" — it was describing a limit while the
        # operator read it as a balance.
        #
        # Neither vendor can settle this for us: Anthropic has no balance
        # endpoint (GET /v1/organizations/balance is a 404 and the feature is
        # an open request), and Doubleword's inference API answers 404 on
        # /balance, /credits, /account, /billing and /usage alike. So the
        # honest move is not to invent a balance but to stop implying one.
        try:
            from backend.core.ouroboros.governance.capability_state import (
                get_default_evaluator as _cap_eval2,
            )
            _unfunded = _cap_eval2().evaluate().is_funding_issue
        except Exception:  # noqa: BLE001
            _unfunded = False
        if _unfunded and budget:
            console.print(
                _T(f"{_glyph('warn', '!')} the ${budget:.2f} ceiling is a "
                   f"POLICY LIMIT, not a balance — no paid lane is funded, so "
                   f"nothing can be spent against it", style=_muted),
                highlight=False)
        _basis = str(status.get("cost_budget_basis") or "").strip()
        if _basis and budget:
            # `_T`, not `_Text`: this function aliases rich's Text as `_T`,
            # while `_Text` exists only inside the header builder. The wrong
            # name raised NameError, the "NEVER raises" wrapper swallowed it,
            # and every line BELOW this point — including the liquidity rows —
            # silently stopped rendering. A fail-soft contract turns a typo
            # into missing output rather than a crash, which is why this is
            # pinned by a test that asserts the lines are actually produced.
            console.print(
                # Parenthesised, not em-dashed: the basis string carries its
                # own punctuation and varies in shape ("observed — 98
                # sessions…", "clamped to the Aegis session cap…",
                # "unmeasured — …", "operator"). A parenthetical reads
                # correctly for all of them without parsing any of them.
                _T(f"{_glyph('detail', '-')} budget ${budget:.2f} "
                   f"({_basis})", style=_muted),
                highlight=False)
        if ops:
            console.print(_active_ops_line(ops), markup=False, highlight=False)
        for line in _liquidity_lines(liq.get("providers") or {},
                                     any_exhausted=liq.get("any_exhausted"),
                                     economic=liq.get("economic")):
            # A warning must NOT recede with the rest of the block — it is
            # the one line here an operator has to act on.
            _is_warn = line.lstrip().startswith(
                (_glyph("warn", "!"), "!", "⚠"))
            console.print(
                line, markup=False, highlight=False,
                style=(_tone("warning", "yellow") if _is_warn else _muted),
            )
        # The hint is entirely secondary — one tone, no competing emphasis,
        # and the literal backticks are gone. `markup=False` printed them as
        # characters, so the line advertised "`ov restart`" with the quoting
        # visible; a terminal shows a command by styling it, not by fencing it.
        hint = _T()
        hint.append(f"{_glyph('detail', '-')} ", style=_muted)
        hint.append("type verbs or plain text", style=_muted)
        hint.append(f"  {_dot}  ", style=_muted)
        hint.append("Ctrl+C", style=_body)
        hint.append(" detaches, the organism keeps running", style=_muted)
        hint.append(f"  {_dot}  ", style=_muted)
        hint.append("ov restart", style=_body)
        hint.append(" reloads it", style=_muted)
        console.print(hint, highlight=False)
    except Exception:
        pass


def _live_incumbent() -> Any:
    """PID of the live single-flight holder, or None.

    Delegates to `thin_client`, which owns the ONE reader the reaper,
    preflight and ghost-socket check all share — so `ov restart` cannot form
    a different opinion about whether a daemon exists than the code that
    decides whether to ignite one.
    """
    try:
        from backend.core.ouroboros.cli.thin_client import (
            _live_incumbent as _probe,
        )
        return _probe()
    except Exception:  # noqa: BLE001
        return None


def _restart_daemon(say: Any) -> int:
    """Stop the running organism and ignite a fresh one. Returns an exit code.

    This exists because the alternative was `ps`, reading a pid, `kill`, then
    `ov` — four steps and a number the operator had to transcribe correctly,
    performed most often when a daemon is stale and least often when they have
    patience for it.

    NOT bound to Ctrl+C, deliberately. The organism is PROACTIVE: it runs
    sensors and autonomous operations, and detaching is meant to leave it
    working. Killing on Ctrl+C would end a long soak every time someone
    glanced at it — the persistence is the feature, and this is the escape
    hatch, not a reversal of it.

    Graceful first: SIGTERM lets the harness write its summary, flush
    telemetry and release its lock, which is what makes the NEXT boot clean.
    SIGKILL only if it will not leave — and a killed daemon leaves the ghost
    socket this codebase now detects rather than wedges on.
    """
    import signal
    import time as _t

    pid = _live_incumbent()
    if pid is None:
        say(_glyph("detail", "-") + " no organism running — igniting a fresh one")
    else:
        say(_glyph("action", "*") + f" stopping organism (pid {pid}) — SIGTERM, letting it finish")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid = None
        except PermissionError:
            say(_glyph("warn", "!") + f" not permitted to stop pid {pid} — is it yours?")
            return 1
        except Exception as exc:  # noqa: BLE001
            say(_glyph("warn", "!") + f" could not signal pid {pid}: {type(exc).__name__}")
            return 1

    if pid is not None:
        # Wait for it to ACTUALLY go. Igniting while the old one still holds
        # the single-flight lock is how two organisms briefly race for one
        # socket — the class this codebase already paid for.
        deadline = _t.monotonic() + _restart_grace_s()
        while _t.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            _t.sleep(0.2)
        else:
            say(_glyph("detail", "-") + " it did not stop in time — escalating to SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
                _t.sleep(0.5)
            except Exception:  # noqa: BLE001
                pass
        say(_glyph("detail", "-") + " organism stopped")
    return 0


def _restart_grace_s() -> float:
    """How long a daemon gets to shut down cleanly before SIGKILL.

    Generous by default: the harness writes a summary and flushes telemetry
    on the way out, and cutting that short is what leaves the next boot
    dirty. Env ``JARVIS_OV_RESTART_GRACE_S``.
    """
    try:
        return max(1.0, float(
            os.environ.get("JARVIS_OV_RESTART_GRACE_S", "15") or 15))
    except (TypeError, ValueError):
        return 15.0


def _active_ops_line(ops: Any) -> str:
    """What the organism is working on, in a form a human can hold.

    This printed full UUIDv7s — ``op-019fa4d2-246e-7759-86,
    op-019fa4d2-2468-794f-bc, ...`` — three of which fill a line and none of
    which an operator can distinguish, because UUIDv7 is time-ordered and
    same-millisecond ops share their entire prefix. The identifying bytes are
    at the END, which is exactly where the truncation cut.

    So the SUFFIX is shown: it is the part that actually differs. The count
    leads, because "how many" is the question a glance is asking, and the
    total is stated when more are running than are listed — a silent
    truncation reads as "that is all of them".
    """
    try:
        items = [str(o) for o in (ops or []) if str(o).strip()]
    except Exception:  # noqa: BLE001
        return _glyph("detail", "-") + " active ops: (unreadable)"
    if not items:
        return _glyph("detail", "-") + " active ops: none"

    def _short(op_id: str) -> str:
        # Whole trailing SEGMENTS, never a raw character slice: cutting
        # mid-segment yields "-7759-86" with a leading dash that reads as a
        # typo. Two segments is enough to distinguish same-millisecond ops,
        # which share every earlier byte.
        parts = [p for p in op_id.split("-") if p]
        if len(parts) >= 3:
            return "-".join(parts[-2:])
        return parts[-1] if parts else op_id

    shown = items[:4]
    body = ", ".join(_short(o) for o in shown)
    more = len(items) - len(shown)
    suffix = f" (+{more} more)" if more > 0 else ""
    return _glyph("detail", "-") + (
        f" {len(items)} active op{'s' if len(items) != 1 else ''}: "
        f"{body}{suffix}")


def _economic_evidence(reason: str, limit: int = 88) -> str:
    """The human sentence out of a provider's error envelope. NEVER raises.

    Vendors wrap the one useful sentence in transport noise —
    ``Error code: 400 - {'type': 'error', 'error': {'type': ...,
    'message': 'Your credit balance is too low…'}}``. Truncating that at 88
    characters yields the JSON scaffolding and drops the sentence, so the
    operator reads a type name where the remedy should be. Pull the message
    out when it is there; fall back to the raw text when it is not.
    """
    try:
        text = str(reason or "").strip()
        if not text:
            return ""
        for marker in ("'message': '", '"message": "'):
            i = text.find(marker)
            if i == -1:
                continue
            rest = text[i + len(marker):]
            end = rest.find(marker[-1])
            msg = (rest[:end] if end > 0 else rest).strip()
            if msg:
                text = msg
                break
        return text if len(text) <= limit else text[:limit - 1] + "…"
    except Exception:  # noqa: BLE001
        return str(reason or "")[:limit]


def _liquidity_lines(providers: Any, *, any_exhausted: Any = None,
                     economic: Any = None) -> list:
    """Provider runways, ordered by what an operator needs to act on.

    Three defects this replaces, and they compounded:

      * ``⚠ a provider runway is dry`` named NOTHING. The per-provider rows
        that answer "which one" were already in hand, so the warning withheld
        an answer it was holding.
      * the rows were sliced ``[:3]`` in DICT ORDER — arbitrary, so an
        exhausted provider sitting fourth was never displayed, and the warning
        then referred to something invisible.
      * ``5,000,000 tokens`` beside "a runway is dry" reads as a contradiction
        until you know they describe different providers.

    So the list is ordered by URGENCY rather than by whatever order the dict
    happened to have: exhausted first, then the thinnest runway. Truncation
    now drops what matters least instead of whatever sorted last, and an
    exhausted provider can never be the row that gets cut.
    """
    lines: list = []
    try:
        rows = list((providers or {}).items())
    except Exception:  # noqa: BLE001
        return lines

    def _remaining(row: Any) -> Any:
        try:
            return row.get("tokens_remaining")
        except Exception:  # noqa: BLE001
            return None

    def _dry(row: Any) -> bool:
        try:
            if row.get("exhausted"):
                return True
        except Exception:  # noqa: BLE001
            return False
        left = _remaining(row)
        return isinstance(left, (int, float)) and left <= 0

    # Sort key: dry first, then thinnest. `None` (undeclared) sorts last —
    # an unknown runway is not evidence of a problem, and promoting it would
    # push a real one off the list.
    def _key(item: Any) -> Any:
        _name, row = item
        left = _remaining(row)
        known = isinstance(left, (int, float))
        return (0 if _dry(row) else 1, 0 if known else 1,
                left if known else 0)

    try:
        rows.sort(key=_key)
    except Exception:  # noqa: BLE001
        pass

    dry_names = [n for n, r in rows if _dry(r)]
    # Always show every dry provider, plus enough healthy ones for context.
    shown = max(3, len(dry_names))
    for name, row in rows[:shown]:
        left = _remaining(row)
        if isinstance(left, (int, float)):
            amount = f"{int(left):,} tokens"
        else:
            amount = "undeclared"
        mark = " ← dry" if _dry(row) else ""
        lines.append(
            f"{_glyph('detail', '-')} liquidity {name}: {amount}{mark}")

    # ECONOMIC DEATH, said FIRST and said plainly.
    #
    # A rate-limit bucket and an account balance are different axes, and the
    # cockpit only ever showed the first. During soak bt-2026-08-01-015739 it
    # displayed `liquidity anthropic: 5,000,000 tokens` for 20 hours while
    # every request returned 400 "Your credit balance is too low" — 23 ops, 0
    # completed, $0.00 spent. Maximum displayed health and total inability to
    # spend rendered identically.
    #
    # "Add credits" is also the one remedy in this whole banner the operator
    # can act on immediately, so it leads.
    for provider, detail in sorted((economic or {}).items()):
        # READ THE SHAPE `economic_view()` ACTUALLY RETURNS.
        #
        # This block filtered on `consecutive_economic_failures`, a key that
        # does not exist in the payload it is given. `economic_view(name)`
        # returns {state, reason, hard_open, expires_in_s, ...}, so the filter
        # was always zero and the branch never rendered — for ANY provider,
        # ever. Verified live: doubleword sat at `state='economic',
        # hard_open=True, reason='status 402'` while the banner showed only
        # `liquidity anthropic: 5,000,000 tokens` and said nothing about
        # credit at all.
        #
        # That is the precise defect the comment above this loop describes as
        # FIXED. The axis was computed, plumbed and then read with the wrong
        # key, so the cockpit knew the lane was economically dead and had no
        # way to say it — 20 hours of "maximum displayed health and total
        # inability to spend" was the symptom, and this was the cause.
        #
        # Both shapes are accepted: the counter form in case any caller does
        # supply it, and the state form the real provider emits.
        try:
            if not isinstance(detail, dict):
                continue
            failures = int(detail.get("consecutive_economic_failures") or 0)
            state = str(detail.get("state") or "").strip().lower()
            hard_open = bool(detail.get("hard_open"))
            reason = str(detail.get("reason") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        dead = failures > 0 or state == "economic" or hard_open
        if not dead:
            # A LAPSED ECONOMIC FLAG IS NOT AN ABSENCE OF ONE.
            #
            # `economic_view` degrades a stale verdict to state="unknown" once
            # its TTL expires without a re-probe — correct, since it must not
            # assert current knowledge it does not have. But the UI then
            # dropped the row entirely, which reports "nothing known" when
            # what is actually known is "last time we looked, this lane was
            # out of money, and that was N hours ago."
            #
            # Observed live: anthropic carried the literal string "Your credit
            # balance is too low" in `reason`, `stale_clock=True`, and
            # rendered NOTHING — while the row above it advertised 5,000,000
            # tokens. The classifier is left authoritative; this only stops a
            # stale-but-informative verdict from being displayed as silence.
            if not (detail.get("stale_clock") and reason):
                continue
            since = detail.get("unverified_since")
            try:
                import time as _t
                age = f"{max(0.0, (_t.time() - float(since))) / 3600.0:.1f}h"
            except Exception:  # noqa: BLE001
                age = "an unknown time"
            _ev = _economic_evidence(reason, limit=80)
            lines.append(
                f"{_glyph('warn', '!')} {provider}: last known OUT OF CREDIT, "
                f"unverified for {age} ({_ev}) — the flag lapsed on a timer, "
                f"not on a successful probe"
            )
            continue
        # Prefer the evidence the provider actually gave us over a count it
        # never reported: "status 402" is more use to an operator than "1
        # billing refusal(s)".
        if reason:
            evidence = _economic_evidence(reason)
        elif failures > 0:
            evidence = f"{failures} billing refusal(s)"
        else:
            evidence = state or "economically refused"
        lines.append(
            f"{_glyph('warn', '!')} {provider}: OUT OF CREDIT — the lane is "
            f"economically dead ({evidence}); the token count above is a "
            f"RATE LIMIT, not a balance. Add credits to restore it."
        )

    if dry_names:
        # NAME them. "a provider" is the one thing the operator cannot look up.
        lines.append(
            f"{_glyph('warn', '!')} runway exhausted: {', '.join(dry_names)} — "
            f"routing will fall through to the remaining providers"
        )
    elif any_exhausted and not economic:
        # The aggregate flag disagrees with every row we can see. Say that,
        # rather than repeating a claim nothing supports.
        lines.append(
            f"{_glyph('warn', '!')} a runway is reported dry but no provider "
            "row shows it — "
            "run /provider for the authoritative view"
        )
    return lines


def _can_run_split_plane() -> bool:
    """Split-plane needs a real TTY on stdin AND prompt_toolkit — piped
    / scripted attaches degrade to the legacy pump. NEVER raises."""
    try:
        if not sys.stdin.isatty():
            return False
        import prompt_toolkit  # noqa: F401
        return True
    except Exception:
        return False


async def _reap_task(task: Any) -> None:
    """Retrieve a task's outcome on EVERY exit path — the 2026-07-18
    dirty-detach class: an abandoned prompt task finishing with its own
    KeyboardInterrupt made asyncio dump 'Task exception was never
    retrieved' over the clean goodbye. Cancel if pending, then consume
    the result/exception so nothing is left for the GC to complain
    about. NEVER raises."""
    import asyncio
    try:
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 — incl. KeyboardInterrupt
            pass
        # Belt: mark a completed task's exception as retrieved.
        if task.done() and not task.cancelled():
            try:
                task.exception()
            except BaseException:  # noqa: BLE001
                pass
    except Exception:
        pass


class AttachUI:
    """The rigid Footer/Header state of the attached TUI.

    Owns the audio-FSM presentation binding (operator mandate: Dynamic
    UI Morphing). ``prompt()`` and ``toolbar()`` are the dynamic
    callables handed to the ONE persistent ``PromptSession`` — prompt
    duplication died with per-iteration prompt construction; the
    session re-evaluates these on every repaint, so a state change
    repaints the footer WITHOUT touching the active keystroke buffer.
    ``on_audio_state`` is loop-safe: it mutates state then invalidates
    the app so prompt_toolkit repaints on its own schedule.
    """

    #: The caret when no voice plane is in play — the base identity prompt.
    #: Named once because FOUR things resolve to it: three FSM states, and the
    #: unknown-state fallback, which must give the same answer as OFFLINE or an
    #: unrecognised frame would look like a state change. Spelling it four
    #: times made it four things to keep in step.
    _BASE_CARET = "ov › "
    #: The same identity in ASCII, for the append-only degradation. A terminal
    #: that cannot report a cursor position may equally be one that cannot
    #: render U+203A, so the degraded path does not borrow the glyph.
    _ASCII_CARET = "ov > "

    _PROMPTS = {
        "OFFLINE": _BASE_CARET,
        "UNAVAILABLE": _BASE_CARET,
        "HELD": _BASE_CARET,
        "LISTENING": "🎙 Karen › ",
        "HEARING": "🎙 Karen (hearing you) › ",
        "THINKING": "💭 Karen (thinking) › ",
        "SPEAKING": "🗣 Karen (speaking) › ",
    }

    _TOOLBAR_NOTES = {
        "HELD": "voice: held by another terminal ('wake!' to take it)",
        "UNAVAILABLE": "voice: unavailable (no audio plane)",
    }

    def __init__(self) -> None:
        self.audio_state: str = "OFFLINE"
        # A gate that arrives mid-sentence waits. Constructed with the sinks
        # it needs so the FSM never reaches for a live app: `_shield_show`
        # renders, `refresh` repaints the badge.
        self.shield = _build_shield(self)
        #: Owns the provisional span while speech is being recognised.
        self.composer = _build_composer()
        #: Last acoustic verdict: (monotonic_ts, diagnosis, device). The
        #: TIMESTAMP is the point — a microphone that recovers sends no
        #: "recovered" event, so a badge with no decay would accuse a
        #: perfectly good headset for the rest of the session.
        self.acoustic: Any = None
        #: Health, derived from the doctor's OWN verdicts on the hydration
        #: frame this cockpit already receives — not a second opinion.
        self.advisor = _build_advisor()
        #: Set once the attach client exists. LATE-BOUND for the same reason
        #: the approval narrator's emit is: the markup sink is built after the
        #: UI, so a handle captured here would be None forever.
        self.markup_sink: Any = None
        self._app_ref: Any = None
        # Latest heartbeat frame (the CC-style pulse) + arrival clock —
        # rendered by toolbar() with client-side elapsed advance and a
        # time-driven glyph (the pt refresh_interval animates it free).
        self._heartbeat: Any = None
        self._heartbeat_arrived: float = 0.0
        # The ambient deck: severity-ordered rows below the pulse. Ambient
        # frames land here instead of the scrollback, so a chatty agora
        # cannot bury the session it is commenting on.
        try:
            from backend.core.ouroboros.battle_test.ambient_deck import (
                DeckManager,
            )
            self.deck: Any = DeckManager()
        except Exception:  # noqa: BLE001 — a deckless cockpit still works
            self.deck = None
        # D3: selectable lanes. The list is whatever the last heartbeat
        # carried — never a client-side cache, so the cursor always resolves
        # against what the daemon currently holds.
        self._lanes: List[dict] = []
        #: The daemon's agent roster, as of the last heartbeat. A SNAPSHOT,
        #: not a local roster: this process dispatches nothing, so it has no
        #: business holding agent state of its own.
        self._agents: dict = {}
        #: The daemon's `StatusSnapshot`, as of the last heartbeat.
        self._status: dict = {}
        #: Applies waiting out their rejection window, as of the last frame.
        self._pending_apply: dict = {}
        #: Crashed-step confirmations from the daemon's heartbeat. None
        #: means the daemon has not said — distinct from [] meaning
        #: nothing is pending, which is why the strip can tell an idle
        #: organism from a silent bridge.
        self._forensics: Optional[list] = None
        #: Submitted-but-unprocessed operator lines, as of the last frame.
        self._input_queue: dict = {}
        #: The last FATAL_PANIC, until the operator dismisses it. Sticky
        #: on purpose: a crash notice that scrolls away was never seen.
        self._panic: dict = {}
        #: The sentence currently being written, if a generation is live.
        self._stream_inflight: str = ""
        self._stream_arrived: float = 0.0
        #: Is the strip currently showing COMMAND output rather than model
        #: prose? Decides whether line breaks are content or artefact.
        self._stream_is_tool: bool = False
        self._focus_lines: List[str] = []
        self._focus_note: str = ""
        self._flash_text: str = ""
        self._flash_until: float = 0.0
        self._deck_size: str = "full"
        try:
            from backend.core.ouroboros.battle_test.cockpit_fsm import (
                CockpitFSM,
            )
            self.fsm: Any = CockpitFSM(lanes_provider=lambda: self._lanes)
        except Exception:  # noqa: BLE001
            self.fsm = None

    def on_lane_reaped(self, lane: str) -> None:
        """The daemon garbage-collected a lane. Eject if we are in it.

        DRY: an auto-eject is the daemon pressing Esc for the operator, so it
        calls the SAME ``fsm.escape()`` the key binding calls — one transition
        path, one set of invariants. A parallel "force_flow" would be a second
        way to reach FLOW that could drift from the first.

        The flash matters as much as the transition. Silently yanking someone
        out of a pane looks like the UI glitched; naming it turns a mystery
        into an explanation. NEVER raises."""
        try:
            fsm = self.fsm
            if fsm is None or not lane:
                return
            if fsm.focused_lane != lane:
                return          # not our pane — nothing to eject from
            fsm.escape()
            self.flash(f"lane {lane} expired — returned to ambient view")
            self._focus_lines = []
            self._focus_note = ""
            self._invalidate()
        except Exception:  # noqa: BLE001
            pass

    def flash(self, message: str, seconds: float = 4.0) -> None:
        """Show a transient notice above the caret. NEVER raises."""
        try:
            import time as _t
            self._flash_text = str(message)
            self._flash_until = _t.monotonic() + max(0.5, float(seconds))
            self._invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _flash_line(self) -> Optional[str]:
        try:
            import time as _t
            if self._flash_text and _t.monotonic() < self._flash_until:
                return f"  [{_SEM['heal']}]![/] {self._flash_text}"
            self._flash_text = ""
        except Exception:  # noqa: BLE001
            pass
        return None

    def set_deck_size(self, mode: str) -> str:
        """``/deck off|compact|full`` — the operator's screen budget.

        Height is a CLIENT concern: two cockpits on different terminals want
        different amounts of screen, and the daemon has no business knowing
        how tall anyone's window is."""
        mode = str(mode or "").strip().lower()
        sizes = {"off": 0, "compact": 2, "full": 8}
        if mode not in sizes:
            return f"deck: {self._deck_size} (off | compact | full)"
        self._deck_size = mode
        self._invalidate()
        return f"deck: {mode}"

    def set_task_view(self, mode: str) -> str:
        """``/tasks [on|off]`` — show or hide the running-subagent roster.

        The same client concern `/deck` is, and for a sharper reason: the
        roster mounts BELOW the caret, so every row it takes is a row between
        the operator's cursor and the bottom of their screen. Three workers
        and a sentinel cost five of them on an idle session.

        Claude Code separates these two surfaces explicitly — the `Ctrl+T`
        checklist is ambient, and "to see running shells and subagents, use
        `/tasks`". This is that verb: the roster is data the daemon streams
        continuously and the operator asks to LOOK at, not a permanent
        fixture under the prompt.

        A bare `/tasks` toggles, because that is what an operator reaching for
        it wants nine times in ten. Explicit `on`/`off` exists so a keybinding
        or a script can be idempotent."""
        from backend.core.ouroboros.battle_test.agent_roster import (
            roster_visible, set_roster_visible, toggle_roster,
        )
        mode = str(mode or "").strip().lower()
        if mode in ("on", "show"):
            shown = set_roster_visible(True)
        elif mode in ("off", "hide"):
            shown = set_roster_visible(False)
        elif mode:
            state = "shown" if roster_visible() else "hidden"
            return f"tasks: {state} (on | off)"
        else:
            shown = toggle_roster()
        self._invalidate()
        if not shown:
            return "tasks: hidden"
        # The COUNT, not just the state. "shown" on an empty roster looks
        # identical to a verb that did nothing, and Claude Code hit the same
        # edge: "when Claude hasn't created any checklist items yet, the
        # toggle has no visible effect because there's nothing to display."
        # Saying how many there are is what distinguishes the two.
        return f"tasks: shown · {self._agent_count()} running"

    def _agent_count(self) -> int:
        """Running agents in the daemon's last snapshot. NEVER raises."""
        try:
            rows = (self._agents or {}).get("rows") or ()
            return sum(
                1 for r in rows
                if isinstance(r, dict)
                and str(r.get("state") or "running") == "running"
            )
        except Exception:  # noqa: BLE001
            return 0

    def refresh(self) -> None:
        """Repaint after a mode change. Alias of the invalidate seam so key
        handlers read as intent rather than mechanism."""
        self._invalidate()

    def on_lane_history(self, payload: dict) -> None:
        """Hydrate the focused pane from the daemon's ring. NEVER raises.

        ``found: false`` is rendered as an explicit notice rather than as an
        empty pane: "this worker's output has aged out" and "this worker
        produced nothing" are different facts and must not look alike."""
        self.note_upstream_activity()
        try:
            lane = str(payload.get("lane", ""))
            if self.fsm is not None and self.fsm.focused_lane != lane:
                return                      # a stale answer for a lane we left
            self._focus_lines = [str(x) for x in (payload.get("lines") or [])]
            if not payload.get("found"):
                self._focus_note = "no retained output — this lane has aged out"
            elif payload.get("tombstoned"):
                dropped = int(payload.get("dropped") or 0)
                more = f", {dropped} earlier line(s) dropped" if dropped else ""
                self._focus_note = f"finished — final output{more}"
            else:
                self._focus_note = "live"
            self._invalidate()
        except Exception:  # noqa: BLE001
            pass

    def bind_app(self, app: Any) -> None:
        self._app_ref = app

    def on_ambient(self, text: str, *, kind: str = "") -> None:
        """One ambient frame → the deck, then repaint. NEVER raises."""
        self.note_upstream_activity()
        try:
            if self.deck is None:
                return
            from backend.core.ouroboros.battle_test.ambient_deck import (
                classify,
            )
            severity, key = classify(text, kind=kind)
            author = ""
            if "@" in text:
                for tok in text.split():
                    if tok.startswith("@"):
                        author = tok.strip("[]")
                        break
            self.deck.push(
                text, severity=severity, key=key or None, author=author,
            )
            self._invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _invalidate(self) -> None:
        """Ask prompt_toolkit to repaint. Its own scheduler decides when, so
        this never touches the operator's keystroke buffer."""
        try:
            app = self._app_ref
            if app is not None:
                app.invalidate()
        except Exception:  # noqa: BLE001
            pass

    def _ignition_deadline(self) -> float:
        """Seconds to wait for first telemetry before flagging unreachable."""
        try:
            return max(1.0, float(
                os.environ.get("JARVIS_IGNITION_TIMEOUT_S", "10") or 10))
        except (TypeError, ValueError):
            return 10.0

    def note_upstream_activity(self) -> bool:
        """First byte from the daemon ends ignition. Returns True on the edge.

        Called from every inbound callback rather than from a poller: the
        bridge already delivers these, and a timer asking "has anything
        arrived yet" would be a second source of truth for a question the
        stream itself answers.

        Idempotent — the transition happens once, on the first payload of any
        kind. Which kind does not matter: a heartbeat, a lane registration and
        an ambient line all prove the same thing, that the far end is alive.
        """
        if not self._igniting:
            return False
        self._igniting = False
        return True

    @property
    def ignition_state(self) -> str:
        """``ignition`` until the first payload, then the FSM's own mode."""
        from backend.core.ouroboros.battle_test.cockpit_fsm import MODE_IGNITION
        return MODE_IGNITION if self._igniting else self.fsm.mode

    def _ignition_line(self) -> Optional[str]:
        """The skeleton row, or None once telemetry has arrived.

        A blank deck during a cold boot is indistinguishable from a healthy
        idle organism with nothing to say — and from a dead socket. Saying
        which one it is costs one line and removes the entire question.
        """
        if not self._igniting:
            return None
        if not self._ignition_started:
            # Lazily anchored to the FIRST RENDER, not to construction: the
            # clock that matters is how long the OPERATOR has been looking at
            # an empty deck. A zero default would read as an instant timeout.
            self._ignition_started = time.monotonic()
        waited = time.monotonic() - self._ignition_started
        if waited > self._ignition_deadline():
            # Still nothing. Do not keep implying progress that is not
            # happening; name the suspicion and let the operator act.
            return ("  " + _glyph("warn", "!")
                    + " daemon unreachable — no telemetry in "
                    + f"{int(waited)}s "
                    + _glyph("dot", "-") + " 'detach' to leave")
        return "  " + _glyph("action", "*") + " awaiting daemon telemetry…"

    #: Set by the surface that has nowhere else to draw the palette. Default
    #: False so the cockpit, which floats it, never double-renders.
    palette_in_toolbar: bool = False
    #: True until the first upstream payload of any kind arrives.
    _igniting: bool = True
    _ignition_started: float = 0.0

    def degrade_to_append_only(self) -> None:
        """Strip this UI to a single caret and nothing else.

        Entered when the terminal never reports its cursor position. Every
        region above the caret — pulse, deck, focused pane — is drawn by
        repainting a block whose position is only knowable from the cursor,
        so on a stream that cannot answer that question they do not degrade
        into something ugly, they degrade into corrupted output.
        """
        self._append_only = True

    @property
    def append_only(self) -> bool:
        return bool(getattr(self, "_append_only", False))

    def prompt(self) -> str:
        """The live region sits ABOVE the input line, then the caret.

        Operator correction, and it is the right shape: the pulse, the deck
        and a focused pane are all things the organism is DOING, and reading
        order runs downward — status, then the line you are typing. A live
        region below the caret makes the eye travel back up to read what just
        happened, and the caret drifts down the screen as the region grows.

        prompt_toolkit renders a multi-line ``message`` above the cursor and
        repaints the whole block on invalidate, so this is the same seam and
        the same redraw machinery — no second region, no manual cursor math.
        The bottom toolbar keeps only the static key hints, which genuinely
        do belong under the input."""
        caret = self.caret()
        if self.append_only:
            # A bare caret. No live region, because there is no way to
            # repaint one without knowing where it is.
            return caret
        try:
            block = self._live_region()
            return f"{block}\n{caret}" if block else caret
        except Exception:  # noqa: BLE001 — a caret always renders
            return caret

    def caret(self) -> str:
        """The input caret ALONE — the audio FSM's own observable surface.

        The FSM had no surface of its own: the caret was resolved inline
        inside :meth:`prompt`, so the only way to ask "what did the state
        change do?" was to read the whole composed block — pulse, flash,
        ignition skeleton, deck rows and all. Those move for reasons that have
        nothing to do with audio, so an FSM assertion written against
        ``prompt()`` fails whenever the DECK changes, which is exactly what
        happened when the live region moved above the input line: three
        ``TestFooterMorphing`` cases had been asserting ``prompt() == "ov › "``
        and went red for a layout decision they were not testing.

        This is the ONE resolution of state -> caret, and :meth:`prompt`
        composes it rather than repeating the lookup. That direction matters:
        a caret re-derived here in parallel with the one ``prompt`` renders
        would be a second authority, and the operator sees the one downstream.

        Always answers — an unknown state falls back to the base caret, which
        is deliberately the SAME string ``OFFLINE`` resolves to, so a garbled
        frame is indistinguishable from idle rather than looking like a new
        mode nobody can name.
        """
        if self.append_only:
            return self._ASCII_CARET
        return self._PROMPTS.get(self.audio_state, self._BASE_CARET)

    def _live_region(self) -> str:
        """Pulse + (deck | lanes | focused pane) — the block above the caret.

        One region, three states, exactly as before; only its position moved.
        Composed from the same pieces the toolbar used, so nothing about the
        deck, the FSM or the heartbeat formatter had to change."""
        try:
            pulse = self._pulse_line()
            flash = self._flash_line()
            mode_lines = self._mode_lines()
            if mode_lines is not None:
                return "\n".join([pulse] + ([flash] if flash else []) + mode_lines)
            from backend.core.ouroboros.battle_test.ambient_deck import (
                GLYPHS,
                deck_enabled,
            )
            head = [pulse] + ([flash] if flash else []) + self._agent_lines()
            cap = {"off": 0, "compact": 2, "full": 99}.get(self._deck_size, 99)
            if self.deck is None or not deck_enabled() or cap == 0:
                return "\n".join(head)
            rows = self.deck.rows()[:cap]
            if not rows:
                # The cold-boot gap: the cockpit paints before the daemon has
                # hydrated. A skeleton row here is the difference between
                # "waiting" and an empty screen that could equally mean idle,
                # wedged, or disconnected.
                skeleton = self._ignition_line()
                return "\n".join(head + ([skeleton] if skeleton else []))
            return "\n".join(
                head + [
                    f"  {GLYPHS.get(sev, '·')} {text}" for sev, text in rows
                ]
            )
        except Exception:  # noqa: BLE001
            return ""

    def _agent_lines(self) -> List[str]:
        """Who is working right now, from the daemon's last snapshot.

        Three properties this has to get right, and each of them is a way the
        surface could lie:

        **Staleness retires the roster, silence does not.** If the daemon dies
        mid-dispatch the last frame we hold says three agents are running, and
        it will say that forever. So the roster expires on the SAME window the
        pulse uses — one clock, one definition of "we have lost contact",
        rather than a second timeout to keep in sync.

        **Elapsed advances between frames.** Frames arrive at ~1 Hz. Without
        the age correction, every running agent's duration would freeze for a
        second and jump, which reads as a stalled system.

        **Width comes from here.** The client knows its terminal; the daemon
        that composed the snapshot does not. Passing it means the goal column
        is clipped to the screen the operator is actually looking at.

        **Rows are asked for, not assumed.** The roster mounts BELOW the
        caret, so each of its rows sits between the operator's cursor and the
        bottom of the screen — and Claude Code puts nothing standing there
        ("the input box stays fixed at the bottom of the screen"), keeping the
        running-subagent view behind `/tasks`. Hidden by default here for the
        same reason: three workers and a sentinel cost five rows under the
        cursor of an idle session. The daemon keeps streaming the snapshot
        while it is hidden, so `/tasks` answers from live data immediately.

        NEVER raises — an unrenderable roster costs its rows, not the cockpit.
        """
        try:
            from backend.core.ouroboros.battle_test.agent_roster import (
                render_roster, roster_line_budget, roster_visible,
            )
            if not roster_visible():
                return []
            age = self._heartbeat_age()
            if age is None:
                return []
            size = self._terminal_size()
            return render_roster(
                self._agents, age_s=age, width=size[0],
                max_lines=roster_line_budget(size[1]),
            )
        except Exception:  # noqa: BLE001
            return []

    def _status_rows(self) -> List[str]:
        """CC's status line, from the daemon's snapshot, at THIS terminal.

        Retires on the same staleness window as the pulse and the roster —
        one definition of "lost contact" across every surface fed by the
        heartbeat, rather than three that drift apart and leave a dead
        daemon's phase showing under an idle pulse.

        NEVER raises: a status line is chrome.
        """
        try:
            from backend.core.ouroboros.battle_test.status_line import (
                payload_to_snapshot, render_snapshot,
            )
            if self._heartbeat_age() is None:
                return []
            snap = payload_to_snapshot(self._status)
            if snap is None:
                return []
            line = render_snapshot(snap, width=self._terminal_size()[0])
            return [f"  {line}"] if line else []
        except Exception:  # noqa: BLE001
            return []

    def _heartbeat_age(self) -> Optional[float]:
        """Seconds since the daemon's last frame, or None when contact is lost.

        ONE definition of "lost contact", which is what three separate
        docstrings on this class were already asking for in prose — the
        roster's ("the roster expires on the SAME window the pulse uses — one
        clock, one definition"), the status line's ("rather than three that
        drift apart and leave a dead daemon's phase showing under an idle
        pulse") and the countdown's. All three then implemented it inline, and
        the serpent border was about to make it four.

        The failure that discipline prevents is specific and reads as normal
        operation: if the daemon dies mid-dispatch, the last frame this
        process holds says three agents are running, phase is GENERATE and the
        border should be moving — and it will say that forever. Every surface
        fed by this heartbeat has to retire on the same clock or the cockpit
        shows a confident, coherent, wrong picture of a dead organism.

        Returns the AGE rather than a bool because callers need it: running
        agents advance by it so seconds tick smoothly between 1 Hz frames, and
        the apply countdown subtracts it. A bool would force every caller to
        recompute the number the predicate already had.

        NEVER raises — a surface that cannot ask degrades to "lost", which is
        the safe direction: it stops drawing rather than inventing them.
        """
        try:
            import time as _time
            from backend.core.ouroboros.battle_test.attach_heartbeat import (
                heartbeat_stale_after_s,
            )
            arrived = float(self._heartbeat_arrived or 0.0)
            if not arrived:
                return None
            age = max(0.0, _time.monotonic() - arrived)
            return None if age > heartbeat_stale_after_s() else age
        except Exception:  # noqa: BLE001
            return None

    def diff_controller(self, client: Any = None) -> Any:
        """This cockpit's diff overlay, backed by the daemon's archive.

        NOT a second overlay. `DiffOverlayController` takes its archive as a
        constructor argument — it was built transport-agnostic and nobody had
        used that — so the client gets the same renderer, the same epoch
        guard, the same `Escape` arbitration and the same off-thread Pygments
        pass that keeps the loop unstalled. A regression in the daemon's diff
        surfaces here too, instead of in a parallel drawing that agrees with
        itself.

        Built lazily and once: the controller that the `/expand d-N` verb
        OPENS and the `diff_rows` hook that DRAWS must be the same object, or
        the verb fills a surface nothing renders.
        """
        existing = getattr(self, "_diff_controller", None)
        if existing is not None:
            return existing
        try:
            from backend.core.ouroboros.battle_test.diff_bridge import (
                RemoteDiffArchive,
            )
            from backend.core.ouroboros.battle_test.diff_overlay import (
                DiffOverlayController,
            )

            def _request(ref: str) -> None:
                # Issued from a RENDER path, so it must never block. The verb
                # travels on the ordinary input lane and the answer comes back
                # addressed on the telemetry lane.
                try:
                    if client is not None:
                        client.send_input(f"/diff-fetch {ref}")
                except Exception:  # noqa: BLE001
                    pass

            archive = RemoteDiffArchive(request=_request)
            controller = DiffOverlayController(
                archive=archive,
                invalidate=self._invalidate,
                width_fn=lambda: self._terminal_size()[0] or 100,
            )
            self._diff_archive = archive
            self._diff_controller = controller
            try:
                controller.register()
            except Exception:  # noqa: BLE001
                pass
            return controller
        except Exception:  # noqa: BLE001
            return None

    def _ingest_diff_catalog(self, rows: Any) -> None:
        """Absorb the heartbeat's diff catalog. NEVER raises."""
        try:
            archive = getattr(self, "_diff_archive", None)
            if archive is not None:
                archive.ingest_catalog(rows)
        except Exception:  # noqa: BLE001
            pass

    def _ingest_diff_payload(self, frame: Any) -> None:
        """Absorb a fetched diff and re-render if it is the one on screen.

        Re-OPENING is what turns a late arrival into a repaint: the
        controller's epoch guard already makes a stale render harmless, so
        asking it to open the same ref again is the whole mechanism — no
        second code path for "the body finally landed". NEVER raises.
        """
        try:
            archive = getattr(self, "_diff_archive", None)
            controller = getattr(self, "_diff_controller", None)
            if archive is None:
                return
            ref = archive.ingest_payload(frame)
            if not ref or controller is None:
                return
            # Only if THIS ref is the one the operator is looking at. A fetch
            # that lands after they moved on must not yank the overlay back.
            if controller.is_active() and getattr(
                    controller, "_ref", None) == ref:
                controller.open(ref)
        except Exception:  # noqa: BLE001
            pass

    def _serpent_active(self) -> bool:
        """Is the organism THINKING right now, as far as THIS terminal knows?

        Drives the serpent hairline that frames the caret. The daemon answers
        this from `build_heartbeat_payload` in-process; an attach client has
        no organism to ask, so it reads the `active` flag off the last frame
        that crossed the bridge. Same question, same field, two sources —
        which is the property that keeps the border, the toolbar verb and the
        token counter from disagreeing about whether work is happening.

        `capability_handoff` measured this hook UNSET on `ov`, so the
        animation ran in `ov demo live` and on the daemon's own terminal and
        was dead on the surface an operator actually attaches with. The border
        simply never moved, which is indistinguishable from an organism that
        is never busy.

        Staleness retires it, and that is the load-bearing half: a border that
        keeps animating after the daemon dies is the cockpit asserting work is
        in flight when nothing is running at all.
        """
        try:
            if self._heartbeat_age() is None:
                return False
            return bool((self._heartbeat or {}).get("active"))
        except Exception:  # noqa: BLE001
            return False

    def _pending_apply_rows(self) -> List[str]:
        """The rejection window, counting down. NEVER raises.

        Retires on the same staleness window as the pulse, the roster and
        the status line — a dead daemon must not leave a countdown ticking
        toward an apply that will never happen.
        """
        try:
            from backend.core.ouroboros.battle_test.pending_apply import render
            age = self._heartbeat_age()
            if age is None:
                return []
            return render(self._pending_apply, age_s=age,
                          width=self._terminal_size()[0])
        except Exception:  # noqa: BLE001
            return []

    def _forensic_rows(self) -> List[str]:
        """A crashed UI step's black box, as seen from the other process.

        Same renderer as the daemon's own strip — `forensic_delta.rows_for` —
        different source, which is the rule `cockpit_mount` states: neither
        surface learns where the other's state came from.

        Retires on the same staleness window as the countdown and the roster.
        A dead daemon must not leave a confirmation prompt on screen: the
        operator would answer a question about a process that is gone, and the
        answer would go nowhere.
        """
        try:
            from backend.core.ouroboros.battle_test.forensic_delta import rows_for
            if self._heartbeat_age() is None:
                return []
            return rows_for(self._forensics, width=self._terminal_size()[0])
        except Exception:  # noqa: BLE001
            return []

    def _push_tail_to_deck(self) -> None:
        """Compose the in-flight text into the transcript. NEVER raises.

        Wrapped by the deck's own renderer at its own width, so nothing
        here needs to know the terminal size — which is why this can live
        on the producer side at all.
        """
        try:
            mux = getattr(self, "_mux", None)
            if mux is None or not hasattr(mux, "set_streaming_tail"):
                return
            mux.set_streaming_tail(self._stream_inflight or "")
        except Exception:  # noqa: BLE001
            pass

    def _deck_carries_tail(self) -> bool:
        """Is the in-flight text already inside the transcript? NEVER raises.

        The strip predates the deck being able to hold it. Both exist so
        the fallback survives — a cockpit whose mux lacks the seam still
        streams, below the deck, exactly as before.
        """
        try:
            mux = getattr(self, "_mux", None)
            return mux is not None and hasattr(mux, "set_streaming_tail")
        except Exception:  # noqa: BLE001
            return False

    def _panic_rows(self) -> List[str]:
        """The crash overlay's content, or []. NEVER raises."""
        try:
            from backend.core.ouroboros.battle_test.panic_arbiter import (
                render_panic,
            )
            return render_panic(
                self._panic, width=self._terminal_size()[0])
        except Exception:  # noqa: BLE001
            return []

    def dismiss_panic(self) -> None:
        """Operator acknowledged the crash. NEVER raises."""
        self._panic = {}

    def _input_queue_rows(self) -> List[str]:
        """Lines typed but not yet reached. NEVER raises.

        Silent at depth 0 — a queue keeping up should be invisible. It
        earns a row only when the operator is ahead of the organism,
        which is exactly when a busy system is otherwise indistinguishable
        from a dropped keystroke.
        """
        try:
            from backend.core.ouroboros.battle_test.operator_input_queue import (  # noqa: E501
                render_queue,
            )
            return render_queue(
                self._input_queue, width=self._terminal_size()[0])
        except Exception:  # noqa: BLE001
            return []

    def _stream_rows(self) -> List[str]:
        """The in-flight sentence, wrapped to THIS terminal. NEVER raises.

        This half is STATE — what was last received, and when. The SHAPE is
        `stream_renderer.render_inflight`, shared with every other surface
        that draws in-flight text, so no two of them can disagree about it.
        The wrap is necessarily client-side (the daemon may serve two
        cockpits of different widths, and the canvas draws with
        `wrap_lines=False`, so an unwrapped sentence is clipped at the right
        edge and appears to stop growing) — but a second OPINION about the
        shape is exactly what the roster and status line already paid for.

        Retires on the same staleness window as every other heartbeat-fed
        surface: a dead daemon must not leave half a sentence hanging on
        screen as though it were still being written.
        """
        try:
            import time as _time
            from backend.core.ouroboros.battle_test.attach_heartbeat import (
                heartbeat_stale_after_s,
            )
            from backend.core.ouroboros.battle_test.stream_renderer import (
                render_inflight,
            )
            if not self._stream_inflight or not self._stream_arrived:
                return []
            age = max(0.0, _time.monotonic() - float(self._stream_arrived))
            if age > heartbeat_stale_after_s():
                return []
            # When the deck can carry the tail itself, the strip stands
            # down: rendering the same text twice is worse than either
            # placement alone. The deck is the better home — the words
            # appear where they will come to rest.
            if self._deck_carries_tail():
                return []
            return render_inflight(
                self._stream_inflight, width=self._terminal_size()[0],
                # Command output keeps its line structure; model prose does
                # not. The producer is known from the frame that set it.
                preserve_lines=self._stream_is_tool,
                # Row 0 of a tool tail is `$ bash · 11s` — what is running
                # and for how long. Eliding it leaves test names with no
                # subject.
                keep_first=self._stream_is_tool)
        except Exception:  # noqa: BLE001
            return []

    def _terminal_size(self) -> tuple:
        """``(columns, rows)``, or ``(None, None)`` when it cannot be known.

        None rather than a guess: both consumers treat an unknown dimension as
        "do not constrain on it", and two different fallbacks for the same
        unknown is how a client and its renderer end up disagreeing about how
        much room there is.
        """
        try:
            import shutil
            size = shutil.get_terminal_size()
            return max(20, int(size.columns)), max(4, int(size.lines))
        except Exception:  # noqa: BLE001
            return None, None

    def _pulse_line(self) -> str:
        """The CC-style working pulse, or the idle breadcrumb.

        A live TURN outranks the ambient pulse here. The fallback surface
        has no container to host a turn row, so its one region answers the
        more specific question first: what is happening to MY question,
        then what is happening to the organism."""
        try:
            spinner = getattr(self, "turn_spinner", None)
            if spinner is not None and spinner.active:
                from backend.core.ouroboros.battle_test.turn_spinner import (
                    _markup_to_ansi,
                )
                row = spinner.render()
                if row:
                    return "  " + _markup_to_ansi(row)
        except Exception:  # noqa: BLE001
            pass
        try:
            from backend.core.ouroboros.battle_test.attach_heartbeat import (
                format_heartbeat_line,
            )
            pulse = format_heartbeat_line(
                self._heartbeat, arrival_mono=self._heartbeat_arrived,
            )
            if pulse:
                return pulse
        except Exception:  # noqa: BLE001
            pass
        return "  ov attach — organism live"

    def toolbar(self) -> Any:
        """The slash palette while completing; static key hints otherwise.

        The region directly beneath the caret is where a command palette
        belongs — it describes the line being typed — and on this surface it
        is the only full-width region available. ``PromptSession`` builds its
        own layout and takes no extra containers, which is why the page-style
        palette shipped in #70123 reached the bipartite cockpit and never
        reached the surface ``ov`` actually attaches with: it was written as a
        container, and this surface has nowhere to put one.

        Rendering it as formatted text instead removes that constraint. Same
        ``layout_palette`` maths, same live ``complete_state`` — one palette,
        both surfaces, no second implementation to drift.

        Returns a fragment list while completing and a plain string otherwise;
        prompt_toolkit accepts either."""
        if self.append_only:
            # No toolbar: prompt_toolkit anchors it to the bottom of the
            # screen, which is an absolute position by definition.
            return ""
        if not self.palette_in_toolbar:
            # The cockpit floats the palette as a Z-index overlay, so drawing
            # it here too would render it twice. Only the PromptSession
            # surface — which has nowhere to put a container — opts in.
            return self._key_hints()
        try:
            from backend.core.ouroboros.battle_test.palette_render import (
                palette_fragments,
            )
            fragments = palette_fragments()
            if fragments:
                return fragments
        except Exception:  # noqa: BLE001 — hints are the safe fallback
            pass
        return self._key_hints()

    def acoustic_badge(self) -> str:
        """``🎙 mic: reverb (AirPods)`` while a verdict is fresh, else "".

        Decays on a timer rather than waiting for an all-clear, because the
        gate only fires on a RUN of bad utterances — silence afterwards is
        indistinguishable from "fixed" and from "stopped talking". Fading is
        the honest reading of that ambiguity; a sticky badge would assert
        something the telemetry never said.
        """
        try:
            snap = self.acoustic
            if not snap:
                return ""
            import time as _t
            seen, diagnosis, device = snap
            import os as _o
            ttl = float(_o.environ.get("JARVIS_ACOUSTIC_BADGE_TTL_S", "90"))
            if _t.monotonic() - seen > max(5.0, ttl):
                self.acoustic = None
                return ""
            where = f" ({device})" if device else ""
            return (_glyph("audio", "mic") + ": "
                    + f"{diagnosis or 'degraded'}{where}")
        except Exception:  # noqa: BLE001
            return ""

    def on_acoustic(self, frame: Any) -> None:
        """A degradation verdict arrived. NEVER raises."""
        try:
            import time as _t
            self.acoustic = (
                _t.monotonic(),
                str((frame or {}).get("diagnosis", "") or ""),
                str((frame or {}).get("device", "") or ""),
            )
            spoken = str((frame or {}).get("spoken", "") or "")
            if spoken:
                # Karen says this aloud too; showing it means an operator
                # who missed it — or has her muted — still learns why.
                self.flash(f"🎙 {spoken}", seconds=8.0)
            self.refresh()
        except Exception:  # noqa: BLE001
            pass

    def _key_hints(self) -> str:
        """The affordance list — what the line you are typing on can do."""
        note = self._TOOLBAR_NOTES.get(self.audio_state)
        if note is not None:
            audio = f" · {note}"
        elif self.audio_state == "OFFLINE":
            audio = " · voice: off ('wake')"
        else:
            audio = f" · voice: {self.audio_state.lower()}"
        try:
            from backend.core.ouroboros.battle_test.cockpit_fsm import (
                MODE_FLOW, selection_enabled,
            )
            keys = ""
            if selection_enabled() and self.fsm is not None:
                keys = (" · ^X ^L lanes" if self.fsm.mode == MODE_FLOW
                        else " · esc back")
        except Exception:  # noqa: BLE001
            keys = ""
        # A pending approval outranks every other hint here: it is the only
        # one that says the organism is BLOCKED waiting on this operator.
        # Placed first so it survives a narrow terminal truncating the tail.
        badge = ""
        try:
            badge = self.shield.badge()
        except Exception:  # noqa: BLE001
            badge = ""
        # The microphone outranks even a pending approval: every OTHER thing
        # on this line is still working. A degraded mic means the operator's
        # words are not arriving at all, and everything they try next will
        # fail for a reason they cannot see.
        health = ""
        try:
            if self.advisor is not None:
                health = self.advisor.render()
        except Exception:  # noqa: BLE001
            health = ""
        # The risk floor, when it is NOT the configured one. Silent while
        # following config: a permanent badge saying "normal" is chrome. The
        # moment it says anything, the operator changed something — which is
        # exactly when they need to see it.
        floor = ""
        try:
            from backend.core.ouroboros.governance.session_risk_floor import (
                session_floor_label,
            )
            floor = session_floor_label()
        except Exception:  # noqa: BLE001
            floor = ""
        mic = self.acoustic_badge()
        badge = f"{mic} · {badge}" if mic and badge else (mic or badge)
        badge = f"{floor} · {badge}" if floor and badge else (floor or badge)
        # Health LAST in the composed line, so it survives truncation least —
        # it is the least time-critical of the three and the only one with a
        # dedicated verb (`ov doctor`) that recovers the full detail.
        badge = f"{badge} · {health}" if badge and health else (badge or health)
        head = f"{badge} · " if badge else ""
        return f"{head}{audio.lstrip(' ·')}{keys} · 'detach' to leave"

    def _mode_lines(self) -> Optional[List[str]]:
        """SELECT / FOCUS rendering, or None to fall through to the deck.

        Both modes draw into the SAME bottom toolbar the ambient deck uses —
        one region, three states — so nothing competes for the terminal and
        the operator's input line is never disturbed."""
        try:
            from backend.core.ouroboros.battle_test.cockpit_fsm import (
                MODE_SELECT,
            )
            fsm = self.fsm
            if fsm is None:
                return None
            if fsm.mode == MODE_SELECT:
                rows = fsm.rows()
                if not rows:
                    return ["  (no lanes) · esc"]
                out = ["  [bold]lanes[/bold] [dim]↑↓ move · ⏎ focus · esc[/dim]"]
                for i, r in enumerate(rows[:8]):
                    cur = "▸" if i == fsm.cursor else " "
                    dead = " [dim](finished)[/dim]" if r.get("tombstoned") else ""
                    out.append(
                        f"  {cur} {r.get('lane','?')}{dead} "
                        f"[dim]{r.get('lines',0)} lines · {r.get('age_s',0)}s[/dim]"
                    )
                return out
            lane = fsm.focused_lane
            if lane:
                head = (
                    f"  [bold]{lane}[/bold] "
                    f"[dim]{self._focus_note} · esc to return[/dim]"
                )
                body = [f"  [dim]│[/dim] {ln}" for ln in self._focus_lines[-6:]]
                return [head] + (body or ["  [dim]│ (hydrating…)[/dim]"])
            return None
        except Exception:  # noqa: BLE001
            return None

    def _with_deck(self, pulse_line: str) -> str:
        """Pulse on top, ambient rows beneath.

        Rendered into the SAME bottom_toolbar prompt_toolkit already repaints
        — no second render loop, no Live region competing for the terminal.
        The toolbar simply grows to the number of lines returned."""
        try:
            from backend.core.ouroboros.battle_test.ambient_deck import (
                GLYPHS,
                deck_enabled,
            )
            # SELECT / FOCUS take the region when active; FLOW shows ambient.
            mode_lines = self._mode_lines()
            if mode_lines is not None:
                return "\n".join([pulse_line] + mode_lines)
            if self.deck is None or not deck_enabled():
                return pulse_line
            rows = self.deck.rows()
            if not rows:
                return pulse_line
            lines = [pulse_line]
            for severity, text in rows:
                lines.append(f"  {GLYPHS.get(severity, '·')} {text}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return pulse_line

    def on_telemetry(self, frame: Any) -> None:
        """Telemetry lane landing point — retain heartbeat frames for
        the toolbar pulse; other telemetry kinds pass untouched.
        NEVER raises."""
        self.note_upstream_activity()
        try:
            if isinstance(frame, dict) and frame.get("kind") == "heartbeat":
                import time as _time
                self._heartbeat = frame
                self._heartbeat_arrived = _time.monotonic()
                # Selectable lanes ride the heartbeat. Replaced wholesale
                # rather than merged: the daemon's summary IS the truth, and
                # a client-side merge would resurrect lanes the daemon has
                # already reaped.
                lanes = frame.get("lanes")
                if isinstance(lanes, list):
                    self._lanes = [x for x in lanes if isinstance(x, dict)]
                # The agent roster, same wholesale-replacement rule: the
                # daemon's snapshot IS the truth. Merging would resurrect
                # agents it has already reaped, and a resurrected agent
                # claims work is still happening.
                #
                # A frame WITHOUT the key leaves the last roster standing —
                # that is an older daemon, not an emptied roster, and the
                # staleness window below is what retires it. A frame with an
                # explicit empty snapshot clears it, because that daemon is
                # saying "nothing is running".
                agents = frame.get("agents")
                if isinstance(agents, dict):
                    self._agents = agents
                # The daemon's status snapshot. Same wholesale rule and the
                # same reason: this process's own builder is empty, so a
                # merge would blend real state with a blank.
                status = frame.get("status")
                if isinstance(status, dict):
                    self._status = status
                # A pending apply is CLEARED by absence, unlike the roster and
                # the status line. Those keep their last value because a
                # missing key means an older daemon; here it means the window
                # closed, and a countdown that outlives its op is telling the
                # operator they can still stop something that already ran.
                self._pending_apply = frame.get("pending_apply") or {}
                self._forensics = frame.get("forensics")
                # Lines the operator submitted that the organism has not
                # reached yet. Cleared by absence for the same reason the
                # countdown is: a stale backlog tells them work is
                # pending that already ran.
                self._input_queue = frame.get("input_queue") or {}
                # The diff CATALOG — refs and metadata, no bytes. Replaced
                # wholesale by the archive itself for the same reason the
                # countdown is cleared by absence: the archive is a RING, so
                # a ref that stopped being advertised was evicted, and a
                # merge would let it live forever in a client that saw it
                # once. `all_refs()` feeding "no such diff — available: …"
                # is only honest if this is current.
                if frame.get(_DIFF_CATALOG_KEY) is not None:
                    self._ingest_diff_catalog(frame.get(_DIFF_CATALOG_KEY))
            elif isinstance(frame, dict) and frame.get(
                "kind",
            ) == "stream_inflight":
                # The sentence the model is in the middle of. STATE, so the
                # last frame wins outright — no accumulation to drift, and a
                # dropped frame costs one tick of smoothness rather than a
                # word. `done` clears it: everything is in the deck by then.
                self._stream_inflight = (
                    "" if frame.get("done") else str(frame.get("text") or "")
                )
                self._stream_is_tool = False
                import time as _time
                self._stream_arrived = _time.monotonic()
                self._push_tail_to_deck()
            elif isinstance(frame, dict) and frame.get(
                    "kind") == _DIFF_PAYLOAD_KIND:
                # The bytes of a diff we asked for, addressed to this cockpit.
                self._ingest_diff_payload(frame)
            elif isinstance(frame, dict) and frame.get(
                    "kind") == "fatal_panic":
                # A background task died. STICKY — unlike every other live
                # state here, this is NOT cleared by absence: the organism
                # may be degraded, and a notice that vanishes on the next
                # heartbeat is a notice nobody read. `/dismiss` or esc
                # clears it.
                self._panic = dict(frame)
            elif isinstance(frame, dict) and frame.get("kind") == "tool_stream":
                # A running command's live tail. It shares the in-flight
                # strip with the model's sentence rather than owning a
                # second one: both answer "what is happening right now",
                # they never overlap in time within an op, and two strips
                # would compete for the same rows below the deck.
                #
                # The header carries the tool and elapsed so a long command
                # reads as WORKING rather than stalled — which is the whole
                # complaint a black box produces.
                import time as _time
                # Composed by the SHARED function rather than here. The header
                # `$ bash · 11s` is what makes a long command read as WORKING
                # rather than stalled, and the daemon now draws this strip at
                # its own terminal too — so a second copy of the composition
                # would be a second opinion about what an in-flight tool tail
                # looks like, which is the defect the roster and the status
                # line each already paid for once.
                from backend.core.ouroboros.battle_test.inflight_registry import (  # noqa: E501
                    compose_inflight_text,
                )
                self._stream_inflight = (
                    "" if frame.get("done") else compose_inflight_text(frame)
                )
                self._stream_is_tool = True
                self._stream_arrived = _time.monotonic()
                self._push_tail_to_deck()
        except Exception:
            pass

    def should_flush_on_input(self) -> bool:
        """Ducking predicate: the operator typed a NEW command while
        Karen is composing or speaking — outbound audio yields to the
        human instantly. NEVER raises."""
        return self.audio_state in ("THINKING", "SPEAKING")

    def on_audio_state(self, state: str) -> None:
        """The synapse landing point — morph + repaint. NEVER raises."""
        try:
            state = str(state or "").strip().upper()
            if not state or state == self.audio_state:
                return
            self.audio_state = state
            app = self._app_ref
            if app is not None:
                app.invalidate()
        except Exception:
            pass


def _maybe_summon_audio_plane(client: Any, cmd: str) -> None:
    """Ensure an audio plane exists, without blocking the input loop.

    Schedules the reflex on the running loop and re-sends the arming verb once
    the supervisor is listening — the first send raced the boot and was
    answered by nobody. A no-op when a supervisor is already up (the reflex
    probes before it spawns), and entirely inert with no running loop.
    NEVER raises: a cockpit that cannot summon audio is still a cockpit."""
    try:
        # Local imports: this module imports asyncio per-function (see the
        # existing pattern) and has no module-level logger.
        import asyncio as _aio
        import logging as _logging

        from backend.core.ouroboros.cli.audio_daemon_reflex import (
            ensure_audio_daemon, reflex_enabled,
        )
        _log = _logging.getLogger(__name__)
        if not reflex_enabled():
            return
        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            return

        async def _summon() -> None:
            try:
                available, reason = await ensure_audio_daemon()
                if available and reason == "spawned":
                    # Re-arm: the original verb was sent before anything was
                    # listening for it.
                    try:
                        client.send_audio(cmd)
                    except Exception:  # noqa: BLE001
                        pass
                    _log.info("[ov] audio plane summoned (%s)", reason)
                elif not available:
                    _log.info("[ov] audio plane unavailable (%s)", reason)
            except Exception:  # noqa: BLE001
                pass

        loop.create_task(_summon())
    except Exception:  # noqa: BLE001
        pass


def _extract_mentions(text: str) -> list:
    """The `@path` mentions in a line, via the ONE parser that defines them.

    Delegates to `repl_input_polish.extract_attachments` rather than matching
    `@\\S+` here: that module already decides what counts — `@here` is prose,
    a real path is a mention — and a second rule would eventually disagree
    with the daemon about which is which.
    """
    try:
        from backend.core.ouroboros.battle_test.repl_input_polish import (
            extract_attachments, is_polish_enabled,
        )
        if not is_polish_enabled():
            return []
        return list(extract_attachments(text).paths or ())
    except Exception:  # noqa: BLE001
        return []


def _short_path(path: str) -> str:
    """Filename plus its parent — enough to disambiguate, short enough to
    sit in a flash message."""
    parts = [p for p in str(path or "").split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")


#: The reply frame `RemoteDiffArchive` waits for. Read from the bridge module
#: rather than spelled again here — a kind string that disagrees between
#: producer and consumer is a frame silently dropped, which is exactly how
#: `_VALID_EVENT_TYPES` bit this codebase before.
try:
    from backend.core.ouroboros.battle_test.diff_bridge import (
        CATALOG_KEY as _DIFF_CATALOG_KEY,
        DIFF_PAYLOAD_KIND as _DIFF_PAYLOAD_KIND,
    )
except Exception:  # noqa: BLE001
    _DIFF_PAYLOAD_KIND = "diff_payload"
    _DIFF_CATALOG_KEY = "diffs"


def _route_operator_line(client: Any, ui: Any, line: Any) -> str:
    """THE one operator-line router — shared by the legacy split-plane loop AND
    the Bipartite cockpit (DRY: verbs behave identically on both surfaces).

    Returns ``"detach"`` (leave the loop), ``"handled"`` (audio verb routed on
    the audio lane), or ``"sent"``/``"empty"``. Audio Control Plane verbs never
    travel as chat text — the daemon synapse owns the duplex. A new operator
    command while Karen is composing flushes her outbound buffer FIRST (the
    human always owns the floor). Never raises."""
    try:
        text = (line or "").strip()
        low = text.lower()
        if low in ("detach", "exit", "quit"):
            return "detach"

        # The message anchor — every submitted line except shell mode,
        # which opens its own ⏺ chrome with the command in the header.
        if text and not text.startswith("!"):
            _echo_operator_line(ui, text)

        # `!` shell mode (CC parity): one command on THIS machine, output
        # into this operator's scrollback. Runs in an executor — a slow
        # command never blocks a keystroke. A bare "!" falls through as
        # ordinary text.
        if text.startswith("!") and text[1:].strip():
            import asyncio as _aio
            _aio.ensure_future(_run_client_shell(ui, client, text))
            return "handled"

        audio_verbs = AUDIO_VERBS
        # /deck sizing is a CLIENT concern — two cockpits on different
        # terminals want different amounts of screen, and the daemon has no
        # business knowing how tall anyone's window is. Handled here rather
        # than relayed.
        if low.startswith("/deck") or low == "deck" or low.startswith("deck "):
            arg = text.split(None, 1)[1].strip() if " " in text else ""
            if ui is not None and hasattr(ui, "set_deck_size"):
                ui.flash(ui.set_deck_size(arg))
            _report_local_history(client, text)
            return "handled"

        # /tasks is a CLIENT concern for the same reason /deck is — it spends
        # THIS terminal's rows — and routed here rather than relayed for a
        # second one: the daemon's own `agent_roster` visibility flag governs
        # the daemon's cockpit, not this one. Forwarding would toggle a
        # roster nobody attached is looking at. (The two-process trap: a verb
        # that runs on the wrong side of the socket reports success and
        # changes nothing the operator can see.)
        if low in ("/tasks", "tasks") or low.startswith(("/tasks ", "tasks ")):
            arg = text.split(None, 1)[1].strip() if " " in text else ""
            if ui is not None and hasattr(ui, "set_task_view"):
                ui.flash(ui.set_task_view(arg))
            _report_local_history(client, text)
            return "handled"

        # `/expand d-N` is a CLIENT concern for a reason the other refs are
        # not: the diff OVERLAY is drawn on this terminal. Relayed, it opened
        # the diff on the daemon's own cockpit and mirrored back a line saying
        # it had opened — the operator was told a diff was on screen and shown
        # nothing. Only `d-` is intercepted; `t-`/`o-`/`n-` refs keep
        # round-tripping and mirroring as markup, which already works.
        if low.startswith(("/expand d-", "expand d-")):
            arg = text.split(None, 1)[1].strip() if " " in text else ""
            controller = (ui.diff_controller(client)
                          if ui is not None and hasattr(ui, "diff_controller")
                          else None)
            if controller is not None and controller.open(arg):
                _report_local_history(client, text)
                return "handled"
            # No controller (no prompt_toolkit, degraded mount): fall through
            # and let the daemon answer in the transcript rather than
            # swallowing the operator's request.

        # /keys is likewise a CLIENT concern when attached: the bindings
        # that govern THIS terminal (deck selection, Esc-interrupt,
        # Ctrl+R) are mounted in THIS process's keymap — the daemon's
        # catalog describes a different surface. `/keys daemon [...]`
        # forwards to the daemon REPL's own table for that view.
        if low == "/keys" or low == "keys" or low.startswith(("/keys ", "keys ")):
            parts = text.split(None, 2)
            if len(parts) > 1 and parts[1].lower() == "daemon":
                forward = "/keys" + (f" {parts[2]}" if len(parts) > 2 else "")
                client.send_input(forward)
                return "sent"
            _render_client_keys(ui, text)
            _report_local_history(client, text)
            return "handled"

        cmd = _resolve_audio_verb(low, audio_verbs)
        if cmd is not None:
            # AUTO-SPAWN REFLEX. `ov` boots ouroboros_battle_test.py, which has
            # no audio pipeline; the mic lives in unified_supervisor.py. Arming
            # verbs therefore had nothing to arm unless a supervisor happened
            # to be running.
            #
            # `ov` stays a thin IPC relayer — it does NOT import the audio
            # pipeline and never touches CoreAudio. It just starts the process
            # that OWNS the hardware, then relays the verb over the existing
            # UDS. Fire-and-forget so the input loop never stalls behind a
            # 98K-line kernel boot; the verb is relayed either way, so a
            # supervisor that is already live behaves exactly as before.
            if cmd in ("wake", "force_wake"):
                _maybe_summon_audio_plane(client, cmd)
            client.send_audio(cmd)
            # Audio verbs travel on the audio lane, never as input —
            # report them so other panes still recall the keystroke.
            _report_local_history(client, text)
            return "handled"
        if text:
            # Splice collapsed pastes back in — the daemon receives what
            # was pasted; the operator's screen showed the chip.
            try:
                from backend.core.ouroboros.battle_test.paste_chips import (
                    expand_paste_chips,
                )
                text = expand_paste_chips(text)
            except Exception:  # noqa: BLE001
                pass
            # The turn opens HERE — the one place a line is proven to be
            # leaving for the daemon. Opening at keypress would spin for
            # client-local verbs (/deck, /keys) that no reply is coming
            # for; opening after send would miss a reply that beat us.
            try:
                spinner = getattr(ui, "turn_spinner", None)
                if spinner is not None:
                    spinner.open(text)
            except Exception:  # noqa: BLE001
                pass
            if ui is not None and ui.should_flush_on_input():
                client.send_audio("flush")
            # The daemon receives the GOAL, not the typing mechanics. A
            # trailing `\` told the PROMPT to keep the line open; upstream it
            # is noise the model would read as content. Fences and brackets
            # are content and travel untouched.
            try:
                from backend.core.ouroboros.battle_test.input_continuation import (  # noqa: E501
                    strip_continuations,
                )
                text = strip_continuations(text)
            except Exception:  # noqa: BLE001
                pass
            # @path mentions, acknowledged HERE rather than silently relayed.
            #
            # `repl_input_polish` has parsed these since it shipped — but it
            # was wired into SerpentFlow's own REPL, which lives on a headless
            # daemon nobody types into. The operator's actual input surface
            # never called it, so `@backend/auth.py` travelled upstream as
            # ordinary prose.
            #
            # The mention itself is relayed UNCHANGED. Stripping it here would
            # make the cockpit and the daemon disagree about what was said,
            # and the daemon is where attachment resolution belongs. This
            # confirms the operator was understood; it does not decide.
            # A gate is on screen and this line is a verdict → it answers
            # THAT gate, tagged with its id. The daemon refuses a mismatch,
            # so a verdict written for a gate that has since been superseded
            # is declined rather than landing on whichever op is armed now.
            #
            # Anything that is not a bare verdict falls straight through to
            # the REPL: the operator answering "later, first fix the tests"
            # is giving a goal, not a decision.
            try:
                shield = getattr(ui, "shield", None)
                showing = getattr(shield, "showing", None) if shield else None
                if showing is not None:
                    from backend.core.ouroboros.battle_test.operator_prompt_bridge import (  # noqa: E501
                        is_bare_verdict,
                    )
                    if is_bare_verdict(text) is not None:
                        client.send_input(text, prompt_id=showing.prompt_id)
                        shield.dismiss(showing.prompt_id)
                        return "sent"
            except Exception:  # noqa: BLE001
                pass
            _mentions = _extract_mentions(text)
            if _mentions and ui is not None:
                try:
                    ui.flash(
                        f"attached {len(_mentions)} file"
                        f"{'s' if len(_mentions) != 1 else ''}: "
                        + ", ".join(_short_path(m) for m in _mentions[:3]),
                        seconds=3.0,
                    )
                except Exception:  # noqa: BLE001
                    pass
            client.send_input(text)
            # The buffer is now empty, so a gate deferred while they were
            # typing surfaces in the lull that follows. This is the half the
            # operator never has to learn: they send their line and the
            # question they were shielded from appears.
            try:
                if getattr(ui, "shield", None) is not None:
                    ui.shield.note_buffer("")
            except Exception:  # noqa: BLE001
                pass
            return "sent"
        return "empty"
    except Exception:  # noqa: BLE001 — routing must never crash an input loop
        return "empty"


async def _split_plane_loop(
    client: Any, console: Any, ui: Optional["AttachUI"] = None,
) -> None:
    """The Split-Plane Multiplexer (operator mandate 2026-07-18).

    prompt_toolkit's ``PromptSession`` + ``patch_stdout`` IS the
    thread-safe split-plane mux (DRY — the same solved mechanism
    SerpentFlow's REPL trusts): the ``ov ›`` prompt permanently owns
    the bottom of the TTY on an ASYNC loop (no sleep-blockers, no
    blocking reads); a daemon telemetry line arriving MID-
    KEYSTROKE is intercepted by patch_stdout, the stdin buffer is
    hidden, the line renders on the scrolling plane above, and the
    active input buffer is restored on the bottom line — keystrokes
    can never be split or corrupted (pinned by the concurrent-I/O
    test). The prompt task races a connection watch so a daemon death
    mid-typing detaches instantly instead of hanging on the prompt.
    """
    import asyncio
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    # Persona-host moment: the persistent interactive surface opens
    # with Karen as the host — the visual seam between the scrolling
    # daemon history above and the command plane below.
    console.print(
        "\U0001f4ad Karen ▸ attached — I'm listening. verbs or plain "
        "words both work · 'wake' arms my voice · "
        "'detach' leaves the organism running",
        markup=False, highlight=False,
    )

    ui = ui or AttachUI()
    # Say what the organism is made of, BEFORE anything else is trusted. A
    # daemon outlives its terminal, so "attached" alone does not tell an
    # operator whether today's work is loaded — and a stale one answers every
    # verb with old behaviour while looking perfectly healthy.
    try:
        from backend.core.ouroboros.battle_test.daemon_provenance import (
            env_drift_line,
            staleness_line,
        )
        _stale = staleness_line()
        # The OTHER half of "why isn't this daemon doing what I asked".
        # Stale code and a stale environment are different failures with
        # the same symptom, and showing only one sent an operator chasing
        # a flag at a process that could never have seen it.
        _drift = env_drift_line()
        if _drift:
            _stale = f"{_stale}\n{_drift}" if _stale else _drift
        if _stale:
            console.print(_stale, markup=False, highlight=False)
    except Exception:  # noqa: BLE001 — provenance must never block an attach
        pass
    # Arm the out-of-band stack dump BEFORE the UI mounts. If the cockpit
    # wedges, `kill -USR1 <pid>` is the only way to ask it what it is doing —
    # Ctrl+C does nothing to a deadlocked thread and `kill -9` destroys the
    # frames. Armed here so it covers the mount itself, not just steady state.
    try:
        from backend.core.ouroboros.battle_test.oob_diagnostics import (
            install_oob_stack_dump, oob_hint,
        )
        if install_oob_stack_dump():
            hint = oob_hint()
            if hint and os.environ.get("JARVIS_OOB_HINT", "1") != "0":
                console.print(_glyph("detail", "-") + f" {hint}", markup=False, highlight=False)
    except Exception:  # noqa: BLE001 — a missing debugger must not stop a boot
        pass
    # …and the automatic half of the same instrument.
    #
    # `kill -USR1` needs a human present at the moment of the wedge, so it can
    # only ever catch a TOTAL freeze — the sub-second stalls that precede one
    # are invisible, and those are where the cause is still legible.
    #
    # The watchdog samples this loop's scheduling delay, builds a rolling
    # baseline, and re-arms a C-level dump deadline on every healthy tick. A
    # loop that keeps ticking keeps postponing its own autopsy; a loop that
    # wedges stops postponing and the stacks land in the SAME crash log the
    # signal writes to, without anyone being awake to ask.
    #
    # Deliberately after the mount hint and before the prompt loop, so it
    # covers the surface that actually hosts the reported freeze.
    _loop_watchdog = None
    try:
        from backend.core.ouroboros.battle_test.loop_watchdog import (
            install_loop_watchdog,
        )
        _loop_watchdog = install_loop_watchdog()
    except Exception:  # noqa: BLE001 — diagnostics never block an attach
        _loop_watchdog = None
    # This surface has no container to float the palette into, so it draws it
    # in the toolbar. The cockpit floats it and must NOT opt in, or it renders
    # twice.
    ui.palette_in_toolbar = True
    # Ctrl+R history search — the SAME gated-completer mechanism the
    # bipartite cockpit uses (history_search.py), merged ahead of the
    # slash palette so the two attach surfaces cannot diverge.
    _history = _build_prompt_history()
    _completer = _build_slash_completer()
    _hist_controller = None
    _kb = _build_selection_bindings(ui, client)
    try:
        from backend.core.ouroboros.battle_test.history_search import (
            build_history_search,
            install_history_search,
            merge_history_completer,
        )
        _hist_controller, _hc = build_history_search(_history)
        _completer = merge_history_completer(_completer, _hc)
        if _hist_controller is not None:
            if _kb is None:
                from prompt_toolkit.key_binding import KeyBindings
                _kb = KeyBindings()
            install_history_search(_kb, _hist_controller)
    except Exception:  # noqa: BLE001 — search is a bonus, typing is not
        _hist_controller = None
    # Rewind source + the client action set — the SAME composition the
    # bipartite surface mounts, so the two cannot diverge.
    try:
        from backend.core.ouroboros.battle_test.rewind_menu import (
            merge_rewind_completer,
        )
        _completer = merge_rewind_completer(
            _completer, getattr(ui, "rewind", None),
        )
    except Exception:  # noqa: BLE001
        pass
    _extra_kb = _client_extra_bindings(ui, client)
    if _extra_kb is not None:
        try:
            from prompt_toolkit.key_binding import merge_key_bindings
            _kb = (merge_key_bindings([_kb, _extra_kb])
                   if _kb is not None else _extra_kb)
        except Exception:  # noqa: BLE001
            pass
    # ONE persistent session, dynamic prompt + rigid footer toolbar:
    # both are callables re-evaluated on every repaint, so an
    # audio_state frame morphs the footer via app.invalidate() while
    # the active keystroke buffer stays untouched (mandate 4).
    try:
        from backend.core.ouroboros.battle_test.keymap import editing_mode
        _em = editing_mode()
    except Exception:  # noqa: BLE001
        _em = None
    session: Any = PromptSession(
        message=lambda: ui.prompt(),
        bottom_toolbar=lambda: ui.toolbar(),
        key_bindings=_kb,
        **({"editing_mode": _em} if _em is not None else {}),
        # Native `/` palette over the SAME 60-verb dispatch table the daemon
        # routes to — not a hand-kept list that can drift from it. Threaded:
        # priming the registry walks packages, and on the event loop that
        # would freeze the very keystroke that opened the menu.
        completer=_completer,
        complete_while_typing=True,
        # Persistent recall + history ghost-text — the same
        # .jarvis/repl_history every other surface reads, so a verb
        # typed at the daemon REPL suggests here and vice versa.
        history=_history,
        auto_suggest=_build_prompt_auto_suggest(),
        # Up PREFIX-FILTERS on what has already been typed instead of walking
        # the whole file. Without it a 2000-entry history is technically
        # recallable and practically unusable — the operator presses Up
        # eleven times looking for a goal they typed this morning.
        enable_history_search=True,
        # Ctrl+X Ctrl+E — readline's "finish this in $EDITOR". It matters
        # most for exactly the long multi-line goals the prompt now accepts,
        # where the terminal's one-line editing model runs out.
        enable_open_in_editor=True,
        # Multi-line, with the CONDITION applied to the buffer below — the
        # same two-step the cockpit uses, from the same module, so the two
        # surfaces cannot disagree about when Enter means "go".
        multiline=True,
        # The SAME Style the bipartite cockpit uses. Without it
        # prompt_toolkit paints its default filled light-grey listbox — the
        # loudest thing on a dark screen, and the exact look #70121 removed
        # from the other surface. Two surfaces, one palette (DRY): the brand
        # owns its colours in ui.theme, not per widget.
        style=_cockpit_style(),
        # The palette draws in the toolbar, so the native menu's reserved
        # strip would only open a gap between the caret and the entries.
        reserve_space_for_menu=0,
    )
    # Replace prompt_toolkit's floating menu rather than restyling it: they
    # are different LAYOUTS, and leaving the widget in place renders both at
    # once — a narrow floating column on top of the full-width page.
    _strip_native_menu(session.app)
    # Enter submits unless the text is visibly unfinished. `Buffer.multiline`
    # is prompt_toolkit's own seam — it holds a Filter that `is_multiline`
    # calls per keystroke — so no custom Enter binding has to fight the
    # library's. Bound to THIS buffer rather than `get_app().current_buffer`,
    # which would consult whatever has focus.
    try:
        from backend.core.ouroboros.battle_test.input_continuation import (
            continuation_filter,
        )
        _buf = session.default_buffer
        _buf.multiline = continuation_filter(lambda: _buf.text)
    except Exception:  # noqa: BLE001 — plain multiline still beats one line
        pass
    # The history-search auto-disarm rides the buffer's OWN
    # completions-changed event — subscribable only now that the
    # session (and its default buffer) exists.
    if _hist_controller is not None:
        try:
            _hist_controller.watch(session.default_buffer)
        except Exception:  # noqa: BLE001
            pass
    # Consume prompt_toolkit's OWN cursor-position timeout rather than probing
    # the terminal again — a second probe races the first for the same reply
    # bytes and misclassifies healthy terminals. See append_only.py.
    try:
        from backend.core.ouroboros.battle_test.append_only import (
            install_cpr_degradation,
        )
        install_cpr_degradation(session.app, ui.degrade_to_append_only)
    except Exception:  # noqa: BLE001 — a styled fallback still works
        pass
    ui.bind_app(session.app)
    # A dropped file becomes something the organism understands.
    #
    # Terminals inject an ABSOLUTE PATH on drag-and-drop. Wrapping
    # `Buffer.insert_text` rather than binding a paste key is deliberate: a
    # drop arrives as bracketed-paste text through the same insertion path as
    # typing, so there is no keystroke to bind, and wrapping the one method
    # every insertion goes through catches it however the terminal delivers
    # it.
    try:
        from backend.core.ouroboros.battle_test.drop_translate import (
            install_drop_translation,
        )
        install_drop_translation(session.app.current_buffer)
    except Exception:  # noqa: BLE001 — a paste must still paste
        pass

    async def _watch_disconnect() -> None:
        while client.connected:
            await asyncio.sleep(0.25)

    with patch_stdout(raw=True):
        while client.connected:
            prompt_task = asyncio.ensure_future(
                session.prompt_async(),
            )
            watch_task = asyncio.ensure_future(_watch_disconnect())
            try:
                done, _pending = await asyncio.wait(
                    {prompt_task, watch_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl+C landed in the WAIT itself — reap BOTH tasks
                # (retrieving any KeyboardInterrupt the prompt task
                # finished with) so the goodbye stays clean: no
                # 'Task exception was never retrieved' ever again.
                await _reap_task(prompt_task)
                await _reap_task(watch_task)
                break
            if watch_task in done and prompt_task not in done:
                # Daemon died mid-typing — never hang on the prompt.
                await _reap_task(prompt_task)
                await _reap_task(watch_task)
                break
            await _reap_task(watch_task)
            try:
                line = prompt_task.result()
            except (EOFError, KeyboardInterrupt):
                await _reap_task(prompt_task)
                break
            outcome = _route_operator_line(client, ui, line)
            if outcome == "detach":
                break



def _strip_native_menu(app: Any) -> int:
    """Drop prompt_toolkit's completions float. NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.palette_render import (
            strip_native_completion_menu,
        )
        return strip_native_completion_menu(app)
    except Exception:  # noqa: BLE001 — the native menu is a survivable fallback
        return 0


def _cockpit_style() -> Any:
    """The brand Style, or None if unavailable. NEVER raises."""
    try:
        from backend.core.ouroboros.ui.theme import cockpit_prompt_style
        return cockpit_prompt_style()
    except Exception:  # noqa: BLE001 — default styling still types fine
        return None


def _build_slash_completer() -> Any:
    """The cockpit's slash palette. None when unavailable. NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.repl_completion import (
            build_attach_completer,
        )
        return build_attach_completer()
    except Exception:  # noqa: BLE001 — a cockpit without a palette still works
        return None


def _build_prompt_history() -> Any:
    """Persistent prompt history for the attach surfaces — the SAME
    ``.jarvis/repl_history`` file the daemon REPL writes, so what the
    operator typed at one surface recalls at every other. None when
    disabled/unavailable. NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.repl_completion import (
            build_history,
        )
        return build_history()
    except Exception:  # noqa: BLE001 — a cockpit without recall still works
        return None


def _build_prompt_auto_suggest() -> Any:
    """History ghost-text for the attach surfaces. None when disabled.
    NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.repl_completion import (
            build_auto_suggest,
        )
        return build_auto_suggest()
    except Exception:  # noqa: BLE001
        return None


def _render_client_keys(ui: Any, line: str) -> None:
    """Render THIS process's keymap table into the operator's scrollback.

    Reuses keys_repl's renderer verbatim (one table format everywhere);
    only the delivery differs — the addressed markup sink every ⏺/⎿ line
    already takes, with rich-markup escaping because context headers
    like ``[Chat]`` are content here, not tags. NEVER raises."""
    try:
        from backend.core.ouroboros.governance.keys_repl import (
            dispatch_keys_command,
        )
        result = dispatch_keys_command(line)
        sink = getattr(ui, "markup_sink", None) if ui is not None else None
        if callable(sink):
            try:
                from rich.markup import escape as _escape
            except Exception:  # noqa: BLE001
                def _escape(s: str) -> str:
                    return s
            for ln in str(result.text or "").splitlines():
                sink(_escape(ln), True)
        elif ui is not None and hasattr(ui, "flash"):
            first = str(result.text or "").splitlines() or [""]
            ui.flash(f"{first[0]} — `/keys daemon` for the daemon view")
    except Exception:  # noqa: BLE001
        pass


def _echo_operator_line(ui: Any, text: str) -> None:
    """CC-style message anchor: what you typed lands in YOUR scrollback
    as a ❯ block the moment you press Enter, so the exchange has a
    visible head before anything answers it. Without this the only echo
    of a question was whatever fragment a downstream renderer chose to
    quote. Multi-line input indents its continuation under the glyph.
    Env: JARVIS_OPERATOR_ECHO_ENABLED (default true). NEVER raises."""
    try:
        if os.environ.get(
            "JARVIS_OPERATOR_ECHO_ENABLED", "true",
        ).strip().lower() in ("0", "false", "no", "off"):
            return
        sink = getattr(ui, "markup_sink", None) if ui is not None else None
        if not callable(sink):
            return
        try:
            from rich.markup import escape as _escape
        except Exception:  # noqa: BLE001
            def _escape(s: str) -> str:
                return s
        lines = str(text or "").splitlines() or [""]
        sink(f"[bold #5ee06a]❯[/bold #5ee06a] "
             f"[#dbe6e1]{_escape(lines[0])}[/#dbe6e1]", True)
        for ln in lines[1:]:
            sink(f"  [#dbe6e1]{_escape(ln)}[/#dbe6e1]", True)
    except Exception:  # noqa: BLE001
        pass


def _ring_gate_bell() -> None:
    """An Iron Gate opened while the operator may be in another pane —
    ring the terminal (BEL) and post an OSC-9 notification (iTerm2/kitty
    forward it to the OS). Writes to the REAL stdout, bypassing any
    patched proxy; a non-TTY gets nothing. Env: JARVIS_GATE_BELL_ENABLED
    (default true). NEVER raises."""
    try:
        if os.environ.get("JARVIS_GATE_BELL_ENABLED", "true").strip().lower() \
                in ("0", "false", "no", "off"):
            return
        out = sys.__stdout__
        if out is None or not out.isatty():
            return
        out.write("\x1b]9;O+V: Iron Gate awaiting approval\x07\a")
        out.flush()
    except Exception:  # noqa: BLE001
        pass


def _on_autonomy_state_frame(ui: Any, frame: Any) -> None:
    """The viewport lock's broadcast truth — every pane shows the freeze,
    whoever caused it. NEVER raises."""
    try:
        paused = bool(frame.get("paused"))
        holders = int(frame.get("holders", 0) or 0)
        try:
            ui.autonomy_paused = paused
        except Exception:  # noqa: BLE001
            pass
        if paused:
            ui.flash(f"⏸ autonomy held ({holders} viewer"
                     f"{'s' if holders != 1 else ''})", seconds=4.0)
        else:
            ui.flash("▶ autonomy flowing", seconds=2.0)
    except Exception:  # noqa: BLE001
        pass


async def _run_client_shell(ui: Any, client: Any, text: str) -> None:
    """``!`` shell mode — run one command on THIS terminal's machine,
    render its output into the operator's scrollback, report the line to
    distributed history. Off the event loop (executor) so a slow command
    never freezes a keystroke. NEVER raises."""
    import asyncio
    import subprocess

    cmd = text[1:].strip()

    def _run() -> str:
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            body = (proc.stdout or "") + (proc.stderr or "")
            tail = f"\n[exit {proc.returncode}]" if proc.returncode else ""
            return (body.rstrip() or "(no output)") + tail
        except subprocess.TimeoutExpired:
            return "(timed out after 30s)"
        except Exception as exc:  # noqa: BLE001
            return f"(shell failed: {exc})"

    try:
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(None, _run)
        sink = getattr(ui, "markup_sink", None) if ui is not None else None
        try:
            from rich.markup import escape as _escape
        except Exception:  # noqa: BLE001
            def _escape(s: str) -> str:
                return s
        if callable(sink):
            sink(_escape(f"⏺ ! {cmd}"), True)
            for ln in output.splitlines()[:200]:
                sink("  " + _escape(ln), True)
        elif ui is not None and hasattr(ui, "flash"):
            ui.flash(output.splitlines()[0] if output else "(done)")
        _report_local_history(client, text)
    except Exception:  # noqa: BLE001
        pass


def _transcript_search_rows() -> Any:
    """The `/` search bar renderer, or None when the hatches are unavailable.

    Resolved ONCE at mount rather than probed per repaint, and returns None
    rather than an empty lambda when the module is missing — a strip whose
    provider can never yield anything should not be in the layout at all.
    NEVER raises.
    """
    try:
        from backend.core.ouroboros.battle_test.transcript_hatches import (
            search_status,
        )
        return search_status
    except Exception:  # noqa: BLE001
        return None


def _client_extra_bindings(ui: Any, client: Any) -> Any:
    """The client-side action set both attach surfaces mount — every key
    remappable via keybindings.json, every action ALSO reachable as a
    typed verb so keystroke and verb are one code path. Returns a
    KeyBindings or None. NEVER raises."""
    try:
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings

        from backend.core.ouroboros.battle_test.keymap import bind_action

        kb = KeyBindings()

        @Condition
        def _empty_buffer() -> bool:
            try:
                from prompt_toolkit.application.current import get_app_or_none
                app = get_app_or_none()
                return app is not None and not app.current_buffer.text
            except Exception:  # noqa: BLE001
                return False

        def _cycle_trust(event: Any) -> None:
            try:
                client.send_input("/trust cycle")
                ui.flash("⛨ cycling trust dial…", seconds=2.0)
            except Exception:  # noqa: BLE001
                pass

        bind_action(
            kb, "app:cycleTrust", ("shift+tab",), _cycle_trust,
            context="Chat",
            description="cycle the autonomy trust dial (/trust cycle)",
        )

        @Condition
        def _panic_showing() -> bool:
            try:
                return bool(getattr(ui, "_panic", None))
            except Exception:  # noqa: BLE001
                return False

        def _dismiss_panic_now() -> None:
            """The dismissal itself, with no key event attached.

            The arbiter calls this; the legacy `bind_action` fallback wraps it.
            Split so the behaviour is written once — an overlay's dismiss is a
            fact about the overlay, not about the keystroke that asked for it.
            """
            try:
                ui.dismiss_panic()
                ui.flash("☠ panic dismissed — /status to check the organism",
                         seconds=3.0)
            except Exception:  # noqa: BLE001
                pass

        def _dismiss_panic(event: Any) -> None:
            _dismiss_panic_now()

        # The overlay SAYS "esc dismisses", and this is what makes the key mean
        # it. Routed through `overlay_arbiter` rather than bound here, because
        # `Escape` is over-subscribed: rewind owns `esc esc`, so a plain
        # (non-eager) binding needed the sequence to TIME OUT before the panic
        # would close — the overlay advertised a key that answered late — while
        # `eager=True` would have made `esc esc` unreachable forever.
        #
        # The arbiter binds ONE Escape whose `eager` is a FILTER, so eagerness is
        # decided per keystroke by whether anything is actually on screen. The
        # panic is REGISTERED rather than hardcoded into that filter: the Iron
        # Gate prompt and the diff preview are equally dismissable, and a list of
        # them inside the arbiter would need editing every time the cockpit grows
        # a surface.
        try:
            from backend.core.ouroboros.battle_test.overlay_arbiter import (
                Z_PANIC, install_escape_arbiter, register_overlay,
            )
            register_overlay(
                "panic", z=Z_PANIC,
                is_active=lambda: bool(getattr(ui, "_panic", None)),
                dismiss=_dismiss_panic_now,
            )
            # `rewind` is deliberately NOT passed: `install_rewind_binding` below
            # already owns the sequence together with its empty-prompt filter,
            # and re-binding it here would be a second opinion about when a
            # draft's double-Esc means "clear" instead of "rewind".
            install_escape_arbiter(kb)
        except Exception:  # noqa: BLE001 — a cockpit must boot without the arbiter
            bind_action(
                kb, "app:dismissPanic", ("escape",), _dismiss_panic,
                context="Chat", filter=_panic_showing,
                description="dismiss the FATAL panic overlay",
            )

        @Condition
        def _not_in_transcript() -> bool:
            """`?` means something else inside the transcript viewer.

            CC: "press `?` in the transcript viewer to see available
            shortcuts THERE" — a different table from the cockpit's. Both
            bindings' filters pass inside the viewer with an empty prompt, so
            without this the winner is decided by which was registered last.
            That happens to be correct today and would silently invert the
            day someone reorders two mounts.
            """
            try:
                from backend.core.ouroboros.battle_test.transcript_mode import (
                    is_transcript_mode,
                )
                return not is_transcript_mode()
            except Exception:  # noqa: BLE001
                return True

        def _show_help(event: Any) -> None:
            _render_client_keys(ui, "/keys")

        bind_action(
            kb, "app:help", ("?",), _show_help,
            context="Chat", filter=_empty_buffer & _not_in_transcript,
            description="show keyboard shortcuts (empty prompt only)",
        )

        def _external_editor(event: Any) -> None:
            _edit_in_external_editor(event)

        bind_action(
            kb, "chat:externalEditor", ("ctrl+g",), _external_editor,
            context="Chat",
            description="edit the prompt in $EDITOR",
        )

        rewind = getattr(ui, "rewind", None)
        if rewind is not None:
            try:
                from backend.core.ouroboros.battle_test.rewind_menu import (
                    install_rewind_binding,
                )
                install_rewind_binding(kb, rewind)
            except Exception:  # noqa: BLE001
                pass
        # The transcript escape hatches: [ v { } inside the viewer or while
        # scrolled, Ctrl+L repaint, Ctrl+X Ctrl+N narration toggle — the
        # "see what it is doing / what it did" cluster.
        try:
            from backend.core.ouroboros.battle_test.transcript_hatches import (
                install_transcript_hatches,
            )
            install_transcript_hatches(kb, ui, client)
        except Exception:  # noqa: BLE001
            pass
        # Ctrl+O and the less-style viewer table. The hatches are the keys;
        # this is the state that makes them unambiguous — and the reason the
        # whole j/k/g/G/Space table can be bound at all, since at the live
        # tail every one of those types as itself.
        try:
            from backend.core.ouroboros.battle_test.transcript_mode import (
                install_transcript_mode_bindings,
            )
            install_transcript_mode_bindings(
                kb,
                notify=lambda out: ui.flash(
                    out if isinstance(out, str) else "\n".join(out),
                    seconds=8.0 if not isinstance(out, str) else 2.5,
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        # Large pastes collapse to a chip; the full text splices back in
        # at submit (expand_paste_chips in _route_operator_line).
        try:
            from backend.core.ouroboros.battle_test.paste_chips import (
                install_paste_collapse,
            )
            install_paste_collapse(kb)
        except Exception:  # noqa: BLE001
            pass

        # Tab arbitration. The prompt carries TWO completion sources — the
        # verb/mention completer and history ghost-text — and prompt_toolkit
        # gives them different keys, so on prose like `wha` the completer
        # correctly offered nothing and Tab correctly did nothing while a
        # suggestion sat visible on screen. This makes Tab a dispatcher over
        # both. Mounted HERE because this builder is the one action set both
        # attach surfaces share; binding it at either call site would fix the
        # surface that was looked at and leave the other one dead.
        try:
            from backend.core.ouroboros.battle_test.completion_arbiter import (
                install_completion_arbiter,
            )
            install_completion_arbiter(kb)
        except Exception:  # noqa: BLE001
            pass

        # PRD §28 C2 — single-key gate answering, on the surface `ov attach`
        # actually mounts.
        #
        # `install_confirm_actions` has existed and worked since it shipped,
        # imported at exactly ONE site: bipartite_layout. The capability was
        # never missing; it was unreachable from this cockpit. Mounted at
        # THIS builder for the reason the arbiter above states in its own
        # comment — "this builder is the one action set both attach surfaces
        # share; binding it at either call site would fix the surface that
        # was looked at and leave the other one dead."
        #
        # `submit` routes through `_route_operator_line`, the same path a
        # typed `/accept` takes, so the risk tier, the audit trail and the
        # countdown clear identically. A key that bypassed the verb would be
        # a second approval path, and approval is the one surface where two
        # paths that disagree is a governance problem rather than a UI one.
        #
        # CONCURRENCY: no lock is added here, deliberately. The pending gate
        # is a fact about the ORGANISM, not about a screen — both cockpits
        # should see the same one — and `pending_apply` already guards its
        # state with a `threading.Lock` while `snapshot()` returns a copy, so
        # a reader never sees a torn row. Two cockpits answering at once are
        # separated where it matters: `send_input` tags every verdict with
        # its `prompt_id` and the daemon refuses a mismatch rather than
        # landing "y" on whichever op happens to be armed. A lock here would
        # serialise two reads that were never in conflict.
        try:
            from backend.core.ouroboros.battle_test.menu_bindings import (
                install_confirm_actions,
            )
            _n_confirm = install_confirm_actions(
                kb, submit=lambda _text: _route_operator_line(
                    client, ui, _text),
            )
            logger.debug(
                "[ov] mounted %d confirm action(s) on the attach surface",
                _n_confirm)
        except Exception:  # noqa: BLE001
            logger.debug("[ov] confirm actions unavailable", exc_info=True)
        return kb
    except Exception:  # noqa: BLE001
        return None


def _edit_in_external_editor(event: Any) -> None:
    """Ctrl+G — compose in $EDITOR; the buffer comes back as the prompt.
    Suspends the TUI for the editor's lifetime via prompt_toolkit's own
    seam. NEVER raises."""
    try:
        import subprocess
        import tempfile

        buf = event.app.current_buffer
        editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR")
                  or "vi")

        def _edit() -> None:
            try:
                with tempfile.NamedTemporaryFile(
                    "w+", suffix=".ov-prompt", delete=False,
                ) as fh:
                    fh.write(buf.text)
                    path = fh.name
                subprocess.run([*editor.split(), path], check=False)
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read().rstrip("\n")
                os.unlink(path)
                buf.text = content
                buf.cursor_position = len(content)
            except Exception:  # noqa: BLE001
                pass

        from prompt_toolkit.application import run_in_terminal
        run_in_terminal(_edit)
    except Exception:  # noqa: BLE001
        pass


def _build_history_injector() -> Any:
    """The client's ``on_history_append`` sink, or a no-op when the sync
    module is unavailable. NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.history_sync import (
            make_client_injector,
        )
        return make_client_injector()
    except Exception:  # noqa: BLE001
        return None


def _report_local_history(client: Any, text: str) -> None:
    """A line this CLIENT handled locally still belongs in every other
    terminal's recall — it never crosses as ``input``, so it travels as an
    explicit history_append frame. NEVER raises."""
    try:
        from backend.core.ouroboros.battle_test.history_sync import (
            send_local_append,
        )
        send_local_append(client, text)
    except Exception:  # noqa: BLE001
        pass


def _register_client_op_provider(ui: Any) -> None:
    """Arg-completion candidates for ``<op_id>`` positions, resolved from
    the op ids the bridge stream has ALREADY delivered to this client
    (``ui._active_ops``) — the same source the Esc-interrupt gate trusts.
    Completion never asks the daemon anything on a keystroke. NEVER
    raises."""
    try:
        from backend.core.ouroboros.battle_test.repl_completion import (
            register_arg_provider,
        )

        def _candidates(prefix: str) -> tuple:
            try:
                ops = getattr(ui, "_active_ops", None) or ()
                return tuple(
                    str(op) for op in ops if str(op).startswith(prefix or "")
                )
            except Exception:  # noqa: BLE001
                return ()

        register_arg_provider("op_id", _candidates)
    except Exception:  # noqa: BLE001
        pass


def _build_selection_bindings(ui: Any, client: Any) -> Any:
    """Arrow/Enter/Esc for the selectable deck. NEVER raises.

    Every binding is gated by a ``@Condition`` on the FSM mode rather than by
    an ``if`` inside the handler. The difference matters: a filtered binding
    is not registered for that keypress at all, so ``Up`` still edits history
    and ``Enter`` still submits the line while the cockpit is in FLOW. A
    handler that swallowed the key and then decided not to act would have
    stolen normal editing from the operator.

    Returns None when selection is disabled, which leaves the session exactly
    as it was before D3."""
    try:
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings

        from backend.core.ouroboros.battle_test.cockpit_fsm import (
            MODE_FLOW,
            MODE_SELECT,
            selection_enabled,
        )
        if not selection_enabled():
            return None

        kb = KeyBindings()
        fsm = ui.fsm

        in_select = Condition(lambda: fsm.mode == MODE_SELECT)
        not_flow = Condition(lambda: fsm.mode != MODE_FLOW)
        # Entering SELECT is only offered when the buffer is EMPTY. Ctrl-O on
        # a half-typed command would otherwise discard the operator's context
        # to open a picker they can only leave by escaping.
        can_open = Condition(
            lambda: fsm.mode == MODE_FLOW and not _buffer_text(),
        )

        @Condition
        def in_flow_working() -> bool:
            """FLOW mode with work in flight.

            Gated on work EXISTING so an idle Esc does nothing rather than
            emitting a verb that reports "nothing to cancel" — a key that
            answers when it was not asked trains the operator to ignore it.
            """
            try:
                from backend.core.ouroboros.battle_test.cockpit_fsm import (
                    MODE_FLOW,
                )
                if ui.fsm is None or ui.fsm.mode != MODE_FLOW:
                    return False
                return bool(getattr(ui, "_active_ops", None))
            except Exception:  # noqa: BLE001
                return False

        # Every selection key routes through the remappable keymap
        # (defaults unchanged) — `bind_action` resolves the operator's
        # keybindings.json over the declared default, registers the
        # action for `/keys`, and NEVER raises; when the config is
        # absent this is byte-for-byte the old `@kb.add(...)`.
        from backend.core.ouroboros.battle_test.keymap import bind_action

        def _open(event: Any) -> None:
            fsm.enter_select()
            ui.refresh()

        bind_action(
            # MOVED off Ctrl+O so the transcript viewer can have it, as in
            # Claude Code. Lanes keep a chord rather than losing a key: this
            # is the deck's doorway and there is no verb that opens it.
            kb, "deck:open", ("ctrl+x ctrl+l",), _open,
            context="Deck", filter=can_open,
            description="enter deck selection (empty buffer only)",
        )

        def _up(event: Any) -> None:
            fsm.move(-1)
            ui.refresh()

        bind_action(
            kb, "deck:previous", ("up",), _up,
            context="Deck", filter=in_select,
            description="previous deck entry",
        )

        def _down(event: Any) -> None:
            fsm.move(1)
            ui.refresh()

        bind_action(
            kb, "deck:next", ("down",), _down,
            context="Deck", filter=in_select,
            description="next deck entry",
        )

        def _enter(event: Any) -> None:
            lane = fsm.selected_lane()
            if fsm.focus_selected() and lane:
                # Hydration is REQUESTED, never assumed present. The daemon
                # answers from the lane's ring — including a tombstoned one.
                try:
                    client.send_lane(lane)
                except Exception:  # noqa: BLE001
                    pass
            ui.refresh()

        bind_action(
            kb, "deck:focus", ("enter",), _enter,
            context="Deck", filter=in_select,
            description="focus the selected lane",
        )

        # `eager` is a FILTER, not a flag. Eager means "fire now, do not wait
        # to see if this is the start of a longer sequence" — which is exactly
        # what would swallow the escape half of Alt+Enter. So Esc stays
        # instant while the buffer is EMPTY (the state an operator is in when
        # they want to interrupt: watching, not typing) and yields the
        # sequence while they are composing. Both meanings survive, and which
        # one applies is decided by what the operator is visibly doing.
        def _interrupt(event: Any) -> None:
            """Esc in FLOW interrupts the operator's own work.

            Layered under the existing Esc, which leaves SELECT/FOCUS — that
            binding is filtered to `not_flow`, so FLOW was free and no
            conflict had to be resolved. One key, two meanings, disambiguated
            by what is on screen rather than by a modifier the operator must
            remember.

            Sends the EXISTING `/cancel` verb rather than a new frame type:
            bare cancel already means "my work" on the daemon, so the
            keystroke and the typed verb resolve through one path and cannot
            drift apart.
            """
            try:
                client.send_input("/cancel")
                spinner = getattr(ui, "turn_spinner", None)
                if spinner is not None:
                    spinner.close(reason="interrupted")
                ui.flash("interrupting…", seconds=2.0)
            except Exception:  # noqa: BLE001
                pass

        bind_action(
            kb, "chat:interrupt", ("escape",), _interrupt,
            context="Chat", filter=in_flow_working, eager=_not_composing(),
            description="interrupt your own in-flight work (/cancel)",
        )

        # Ctrl+X Ctrl+K — CC's stop-all, the one keyboard control ov lacked
        # over the L3 subagents it actually runs. Bound through the shared
        # installer so this surface and the daemon's cannot disagree about
        # what the chord means; both just send `/stop-all`.
        try:
            from backend.core.ouroboros.battle_test.subagent_control import (
                install_stop_all_binding,
            )
            install_stop_all_binding(
                kb, client,
                notify=lambda msg: ui.flash(msg, seconds=3.5),
                running=(ui._agent_count if hasattr(ui, "_agent_count")
                         else None),
            )
        except Exception:  # noqa: BLE001
            pass

        def _esc(event: Any) -> None:
            fsm.escape()
            ui.refresh()

        bind_action(
            kb, "deck:escape", ("escape",), _esc,
            context="Deck", filter=not_flow, eager=_not_composing(),
            description="leave SELECT/FOCUS",
        )

        def _release_gate(event: Any) -> None:
            """Surface a deferred approval on demand.

            Ctrl+P was unbound — verified, not assumed — so this takes no
            key away from anyone. The binding exists even with an empty
            queue: a key that works only sometimes teaches the operator not
            to trust it, so an empty pop simply says so.
            """
            try:
                shield = getattr(ui, "shield", None)
                if shield is None:
                    return
                if shield.pop() is None:
                    ui.flash(
                        "no pending approvals" if shield.pending_count == 0
                        else "a gate is already on screen", seconds=2.0,
                    )
            except Exception:  # noqa: BLE001
                pass

        bind_action(
            kb, "gate:review", ("ctrl+p",), _release_gate,
            context="Chat",
            description="surface a deferred approval on demand",
        )

        try:
            from backend.core.ouroboros.battle_test.input_continuation import (
                install_newline_binding,
            )
            install_newline_binding(kb)
        except Exception:  # noqa: BLE001
            pass
        # Shift+Tab raises the risk floor for this session. It composes
        # into risk_tier_floor's strictest-wins resolution rather than
        # overriding it, so the keystroke can only ever ADD friction —
        # it cannot make the organism more permissive than the config
        # already allows, in any cycle position.
        try:
            from backend.core.ouroboros.governance.session_risk_floor import (
                cycle_session_floor,
            )
            kb.add("s-tab")(lambda event: cycle_session_floor())
        except Exception:  # noqa: BLE001
            pass
        # Ctrl+V pastes a SCREENSHOT. The most common way an operator has
        # an image is Cmd+Shift+Ctrl+4 — on the clipboard, never written
        # to disk — and a terminal pastes text, so it produced nothing.
        # Spilled to a file and handed to the EXISTING /attach verb, so
        # validation, the size cap and the multi-modal path are the ones
        # a dragged file already uses. Text pastes fall through unchanged.
        try:
            from backend.core.ouroboros.battle_test.clipboard_image import (
                install_image_paste_binding,
            )
            install_image_paste_binding(
                kb, lambda text: (_prompt_buffer().insert_text(text)
                                  if _prompt_buffer() is not None else None),
            )
        except Exception:  # noqa: BLE001
            pass
        # Ctrl+T collapses the plan checklist. A four-item plan is
        # orientation while work runs and clutter while reading a diff,
        # and which of those it is changes minute to minute — so it is a
        # keystroke, not a setting.
        try:
            from backend.core.ouroboros.battle_test.plan_checklist import (
                toggle_checklist,
            )
            kb.add("c-t")(lambda event: toggle_checklist())
        except Exception:  # noqa: BLE001
            pass
        # Ctrl+S parks a draft. Bound on BOTH surfaces from one definition,
        # because a feature wired to one while the operator types into the
        # other is the defect this codebase keeps finding.
        try:
            from backend.core.ouroboros.battle_test.draft_stash import (
                install_stash_binding,
            )
            install_stash_binding(kb, _prompt_buffer)
        except Exception:  # noqa: BLE001
            pass
        return kb
    except Exception:  # noqa: BLE001 — a cockpit without selection still works
        return None


def _not_composing() -> Any:
    """True while the input buffer is empty. NEVER raises.

    Defaults to True (eager) on any fault, preserving the instant Esc that
    shipped before multi-line existed.
    """
    try:
        from prompt_toolkit.filters import Condition

        @Condition
        def _cond() -> bool:
            try:
                return not _buffer_text().strip()
            except Exception:  # noqa: BLE001
                return True

        return _cond
    except Exception:  # noqa: BLE001
        return True


def _build_advisor() -> Any:
    """The health advisory line, or None if unavailable."""
    try:
        from backend.core.ouroboros.cli.health_advisory import HealthAdvisor
        return HealthAdvisor()
    except Exception:  # noqa: BLE001 — a cockpit without it still attaches
        return None


def _build_composer() -> Any:
    """The dictation span manager, or None if unavailable."""
    try:
        from backend.core.ouroboros.battle_test.transcript_composer import (
            TranscriptComposer,
        )
        return TranscriptComposer()
    except Exception:  # noqa: BLE001 — voice still works without dictation
        return None


def _build_shield(ui: Any) -> Any:
    """The client's deferral FSM, wired to this cockpit's surfaces."""
    try:
        from backend.core.ouroboros.battle_test.focus_shield import FocusShield

        return FocusShield(
            show=lambda prompt: _shield_show(ui, prompt),
            notify=lambda: ui.refresh(),
        )
    except Exception:  # noqa: BLE001 — a cockpit without deferral still runs
        return None


def _shield_show(ui: Any, prompt: Any) -> None:
    """Put a released gate on screen.

    Renders through the SAME markup path every ⏺/⎿ line already takes rather
    than inventing a second display: the gate then lands in the deck, in
    order, in the scrollback the operator can page back through — and on the
    bipartite surface it draws over the canvas Float the palette established,
    with the deck still visible underneath.
    """
    try:
        risk = f" · {prompt.risk}" if getattr(prompt, "risk", "") else ""
        left = prompt.seconds_left(time.monotonic())
        expiry = "" if left == float("inf") else f" · {int(left)}s left"
        body = str(prompt.text or "approve?").replace("\n", " ").strip()
        lines = [f"⏺ Iron Gate({prompt.ref}){risk}",
                 f"  ⎿ {body}  [y/n]{expiry}"]
        sink = getattr(ui, "markup_sink", None)
        if callable(sink):
            for line in lines:
                # ADDRESSED: a question put to this operator belongs in their
                # scrollback, not in the ambient deck where it would scroll
                # away behind autonomous chatter.
                sink(line, True)
        else:
            ui.flash(" ".join(lines), seconds=8.0)
        ui.refresh()
    except Exception:  # noqa: BLE001
        pass


def _on_transcript_frame(ui: Any, frame: Any) -> None:
    """Fold one recognised chunk into the prompt buffer. NEVER raises.

    The buffer is SHARED with the operator's typing, so the composer decides
    what may be replaced and this only carries it out. When it releases a
    span — because the utterance finished, or because they edited inside it —
    the words simply stay where they are and become ordinary typed text.
    """
    try:
        composer = getattr(ui, "composer", None)
        buf = _prompt_buffer()
        if composer is None or buf is None:
            return
        result = composer.on_chunk(frame, buf.text, buf.cursor_position)
        if not result.edits:
            return
        # Splice rather than insert: replacing our own previous partial is
        # what keeps the prompt showing ONE evolving sentence instead of
        # every revision the recogniser passed through.
        before, after = buf.text[:result.start], buf.text[result.end:]
        buf.text = before + result.text + after
        buf.cursor_position = result.start + len(result.text)
        ui.refresh()
    except Exception:  # noqa: BLE001
        pass


def _prompt_buffer() -> Any:
    """The live input buffer, or None when no application is running."""
    try:
        from prompt_toolkit.application.current import get_app
        return get_app().current_buffer
    except Exception:  # noqa: BLE001
        return None


def _on_prompt_frame(ui: Any, frame: Any) -> None:
    """A gate arrived. Show it only if the operator is not mid-sentence.

    The bell rings REGARDLESS of composing state — an operator in another
    tmux pane cannot see a deferred badge, and a gate that expires unseen
    auto-REJECTs. The Global Bell Arbiter's clear-side already exists:
    resolution in ANY terminal broadcasts `prompt_resolved`, and every
    client's shield dismisses the badge.

    `composing` is answered from the LIVE buffer at the moment of arrival —
    the one fact that decides everything here, and the reason the FSM takes
    it as an argument rather than reading a global it cannot be tested
    against.
    """
    try:
        _ring_gate_bell()
        shield = getattr(ui, "shield", None)
        if shield is None:
            return
        shield.offer(frame, composing=bool(_buffer_text().strip()))
    except Exception:  # noqa: BLE001
        pass


def _buffer_text() -> str:
    """Current input buffer, or "" when there is no live application."""
    try:
        from prompt_toolkit.application.current import get_app
        return get_app().current_buffer.text or ""
    except Exception:  # noqa: BLE001
        return ""


async def _attach_rms_stream(scope: Any) -> Any:
    """Feed the header scope from the SUPERVISOR's amplitude stream.

    The gap this closes. `ov` and `unified_supervisor` are separate processes
    — CoreAudio hands the microphone to exactly one of them, and that one is
    the supervisor. The scope was wired only to ``audio_broadcast_tap``, an
    IN-PROCESS zero-copy broadcast, so in the cockpit process nothing ever
    captured audio, the tap never fired, and the wave sat at its flat baseline
    however loudly anyone spoke.

    The supervisor has been publishing ``rms_level`` frames on the audio-state
    socket the whole time (``MicTelemetryBridge`` → ``publish_rms``). There was
    simply no consumer: a producer with no reader is indistinguishable from a
    silent room, which is exactly why it went unnoticed.

    Subscribed DIRECTLY rather than relayed through the daemon bridge. The
    frames are lossy-by-contract 20 FPS telemetry; hopping them through a
    second socket would add a queue that must then be given its own drop
    policy, to carry samples whose whole design is that dropping them is free.

    The in-process tap subscription stays alongside this — it is still correct
    when the cockpit itself owns audio. Whichever source produces data drives
    the wave; neither knows about the other.

    Returns the connected client (so the caller can close it), or None.
    NEVER raises: no amplitude stream is survivable, a broken cockpit is not.
    """
    try:
        from backend.core.ouroboros.governance.comms.duplex.audio_state_ipc import (
            MSG_RMS_LEVEL, AudioStateClient,
        )
        from backend.core.ouroboros.ui.audio_scope import AudioPlane

        # plane wire-value → the scope's colour lane. Karen speaking is venom
        # green, the operator cyan, so a glance answers "is that me or her?"
        planes = {
            "user": AudioPlane.USER,
            "mic": AudioPlane.USER,
            "system": AudioPlane.SYSTEM,
            "karen": AudioPlane.SYSTEM,
            "tts": AudioPlane.SYSTEM,
        }

        # Edge memory for state transitions, scoped to this subscription.
        _last = {"event": ""}

        def _on_frame(msg: dict) -> None:
            try:
                # TRANSCRIPTS — what the organism actually HEARD.
                #
                # The supervisor has published these on this very socket since
                # 2026-07-18 and the cockpit never read one. So an operator who
                # spoke and got no answer had no way to tell "it never heard
                # me" from "it heard me and could not reply" — two completely
                # different faults, and days were spent guessing between them.
                #
                # Showing the transcript makes the loop legible: you see your
                # own words land, or you see nothing and know the ears are the
                # problem, not the mouth.
                if msg.get("type") == "transcript":
                    _txt = str(msg.get("chunk") or msg.get("text") or "").strip()
                    if _txt and msg.get("final", True):
                        _role = str(msg.get("role", "user")).lower()
                        _who = "you" if _role == "user" else "Karen"
                        _style = "cyan" if _role == "user" else "rgb(94,224,106)"
                        _render_markup_frame(
                            f"[{_style}]🎙 {_who}:[/{_style}] "
                            + __import__("rich.markup", fromlist=["escape"]).escape(_txt)
                        )
                    return
                # LIVE STATE — so the operator is never left in the dark.
                #
                # The supervisor has published these transitions since the IPC
                # was written; the cockpit rendered none of them. Between
                # "Hello Karen" and her reply there are 3-5 seconds of STT,
                # LLM and synthesis during which the screen said nothing at
                # all, and silence is indistinguishable from a hang.
                #
                # Each transition is announced ONCE, on the edge: the state
                # machine upstream is already edge-coalesced, and re-printing
                # a steady state would turn a status line into a scroll.
                if msg.get("type") == "event":
                    _kind = str(msg.get("kind", ""))
                    _label = _AUDIO_STATE_LABELS.get(_kind)
                    # Closure-local, NOT the caller's _audio dict: that name
                    # belongs to a different function and referencing it here
                    # raised a NameError the surrounding except swallowed, so
                    # every state line vanished silently. Exactly the failure
                    # this indicator exists to make impossible.
                    if _label and _kind != _last["event"]:
                        _last["event"] = _kind
                        _render_markup_frame(_label)
                    return
                if msg.get("type") != MSG_RMS_LEVEL:
                    return
                plane = planes.get(str(msg.get("plane", "user")).lower())
                if plane is not None and plane != scope.plane:
                    scope.set_plane(plane)
                # Already normalized upstream: the RMS + adaptive scaling ran
                # on the producer side, next to the frames. Re-normalizing a
                # normalized value here would square the curve.
                scope.push(float(msg.get("level", 0.0)), normalized=True)
            except Exception:  # noqa: BLE001 — one bad frame is not an outage
                pass

        client = AudioStateClient(on_message=_on_frame)
        return client if await client.connect() else None
    except Exception:  # noqa: BLE001
        return None


#: Audio-state transition -> the one line the cockpit shows for it.
#: Karen's own voice grammar (💭 thinking, 🗣 speaking) rather than raw event
#: names, so the operator reads a conversation rather than a state machine.
#: Keys are the EVENT_KINDS values verbatim. Written from the module's own
#: tuple rather than from memory: guessing lowercase names produced a mapping
#: that matched nothing and rendered silently — the same shape of failure as
#: a publisher with no subscriber, which this feature exists to end.
_AUDIO_STATE_LABELS = {
    "VAD_ACTIVE": f"[{_SEM['neural']}]🎙 listening…[/]",
    "TTS_GENERATING": "[rgb(94,224,106)]💭 Karen is thinking…[/rgb(94,224,106)]",
    "AUDIO_PLAYING": "[rgb(94,224,106)]🗣 Karen is speaking…[/rgb(94,224,106)]",
    "AUDIO_IDLE": "[dim]· ready[/dim]",
    "SYSTEM_WARMING": "[dim]· audio plane warming…[/dim]",
    "SYSTEM_READY": "[dim]· audio plane ready[/dim]",
    "HW_FAULT": f"[{_SEM['death']}]⚠ audio hardware fault[/]",
    "SYS_TELEMETRY_DEGRADED": f"[{_SEM['heal']}]⚠ telemetry degraded[/]",
    "SYS_TELEMETRY_RECOVERED": "[dim]· telemetry recovered[/dim]",
}


async def _keep_rms_stream(scope: Any, state: dict) -> None:
    """Maintain the amplitude subscription for the cockpit's whole lifetime.

    The one-shot connect this replaces encoded a boot-order assumption that
    the operator's own workflow violates: `ov` first, `wake` second. The
    cockpit subscribed ONCE at boot; if the audio host wasn't serving at that
    exact instant — it usually isn't, since `wake` is what spawns it — the
    client was None forever and the wave could never move, no matter what
    came up afterwards.

    A subscription is not an event, it is a RELATIONSHIP: the host may start
    late, restart, re-bind after losing its address, or die and be respawned
    by the reflex. So the keeper loops for the cockpit's lifetime — connect
    when absent, notice disconnection, back off with full jitter (several
    cockpits must not stampede a booting host), and reconnect. Cheap when
    idle: one failed connect per backoff tick. NEVER raises."""
    # Local imports — this module imports asyncio per-function by convention,
    # and a bare module-level name here would be a NameError swallowed by the
    # task wrapper: the keeper would die instantly and silently, recreating
    # the exact one-shot behaviour it exists to replace.
    import asyncio
    import random as _random

    delay = 0.5
    while not state.get("closing"):
        client = state.get("rms_client")
        if client is not None and getattr(client, "connected", False):
            delay = 0.5                       # healthy — re-arm the backoff
            await asyncio.sleep(1.0)
            continue
        if client is not None:                # died — release before retrying
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
            state["rms_client"] = None
        try:
            state["rms_client"] = await _attach_rms_stream(scope)
        except Exception:  # noqa: BLE001
            state["rms_client"] = None
        if state.get("rms_client") is None:
            await asyncio.sleep(_random.uniform(0.2, delay))
            delay = min(5.0, delay * 2)       # capped: a host can appear any time
        else:
            delay = 0.5


async def _bipartite_attach_loop(client: Any, console: Any, ui: Any) -> None:
    """The Style-Guide §06 cockpit ON THE CLIENT: Zone 1 (the Proactive Canvas,
    state-reactive border) auto-scrolls the daemon's bridge stream; Zone 2 the
    anchored ``› `` prompt reusing THE SAME verb router as the legacy loop; the
    morphing AttachUI footer rides below. The connection watcher exits the app
    the instant the daemon dies (never hangs mid-typing). Never raises out."""
    from backend.core.ouroboros.battle_test.bipartite_layout import (
        run_bipartite_repl,
    )
    from backend.core.ouroboros.ui.theme import UIState, get_reactive_theme

    # Client-side reactive accent: attached = HEALTHY (cyan). The daemon-death
    # path flips DEGRADED (red) just before the app exits — an honest mapping of
    # the CLIENT's own connection state onto the Style-Guide state ladder.
    try:
        get_reactive_theme().set_state(UIState.HEALTHY)
    except Exception:  # noqa: BLE001
        pass

    def _on_accept(text: str) -> None:
        outcome = _route_operator_line(client, ui, text)
        if outcome == "detach":
            try:
                from prompt_toolkit.application import get_app
                get_app().exit()
            except Exception:  # noqa: BLE001
                pass

    # Dynamic op-id completion from the ops THIS client has already been
    # told about — no bridge round-trip on a keystroke. Inert until a
    # dispatch module declares an ``<op_id>`` position via __verb_args__.
    _register_client_op_provider(ui)

    def _alive() -> bool:
        ok = bool(client.connected)
        if not ok:
            try:
                get_reactive_theme().set_state(UIState.DEGRADED)
            except Exception:  # noqa: BLE001
                pass
        return ok

    # The CC-style identity header: the mini ANIMATED crest at top-left with
    # version · state · path beside it (the reactive accent lives in the status
    # dot now that the canvas is borderless). Stateless clock-driven animation;
    # the mini ring builds progressively off-loop. Degrades to text-only on
    # tiny/incapable terminals.
    mini = None
    header_height = 0
    header_render = None
    try:
        import asyncio as _aio
        import os as _os
        import time as _time
        from backend.core.ouroboros.ui.crest_animator import (
            MiniCrest,
            render_cockpit_header,
        )
        from rich.text import Text as _Text

        mini = MiniCrest()
        if mini.available:
            _aio.ensure_future(mini.ensure_frames())
            header_height = max(3, mini.rows)
        else:
            mini = None
            header_height = 3

        def _home_path() -> str:
            try:
                cwd = _os.getcwd()
                home = _os.path.expanduser("~")
                return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd
            except Exception:
                return ""

        _STATE_DOT = {
            "HEALTHY": "rgb(67,214,208)", "DEGRADED": "rgb(248,81,73)",
            "ARMED": "rgb(227,179,65)", "SOAKING": "rgb(94,224,106)",
            "DORMANT": "rgb(108,125,119)",
            # Capability states. BLOCKED and UNKNOWN share the warning hue:
            # "I cannot work" and "I cannot tell whether I can work" are both
            # things the operator must not read as green.
            "BLOCKED": "rgb(248,81,73)", "UNKNOWN": "rgb(227,179,65)",
            # Amber, not red: unfunded is a state the operator can clear in a
            # minute with a card, which is a different urgency from a broken
            # organism even though both stop dispatch.
            "UNFUNDED": "rgb(227,179,65)",
        }

        def _header_lines():
            t1 = _Text()
            # The CC title grammar: "O+V v0.1.0" (bold brand + bare version),
            # exactly like "Claude Code v2.1.218". DRY: resolve_version().
            t1.append("O+V", style="bold rgb(94,224,106)")
            t1.append(f" v{resolve_version()}", style="rgb(219,230,225)")
            t2 = _Text()
            # CAPABILITY, not presentation.
            #
            # This line used to read `get_reactive_theme().state`, which
            # answers "what colour should the dot be" -- and rendered that
            # answer as though it meant "am I able to work". It also defaulted
            # to HEALTHY and swallowed the lookup failure, so two optimistic
            # defaults stacked on a category error. The observed result: a
            # green `● healthy` while both provider lanes were at zero credit
            # and every op was failing.
            #
            # `capability_state` fuses the liquidity ledger (the same reading
            # that already renders "⚠ … dry" on the status line), the daemon
            # heartbeat and op telemetry, degrades deterministically, recovers
            # only on a verified success, and resolves UNKNOWN to blocked.
            state, _reason = "HEALTHY", ""
            try:
                from backend.core.ouroboros.governance.capability_state import (
                    get_default_evaluator as _cap_eval,
                )
                _cap = _cap_eval().evaluate()
                state = _cap.badge.upper()
                _reason = _cap.reason
            except Exception:
                # Even here the fallback is the theme's PRESENTATION state,
                # used only to pick a word we then do not trust to be green:
                # an unreadable capability is not evidence of health.
                state = "UNKNOWN"
            t2.append("● ", style=_STATE_DOT.get(state, "rgb(227,179,65)"))
            t2.append(state.lower(), style="rgb(174,188,182)")
            # The tagline is a claim too. When the organism cannot dispatch,
            # "the organism drives" is false, so the reason takes its place.
            if state == "HEALTHY":
                t2.append(" · ouroboros + venom · the organism drives",
                          style="rgb(108,125,119)")
            else:
                t2.append(" · ouroboros + venom · ", style="rgb(108,125,119)")
                t2.append(_reason or "cannot verify capability",
                          style="rgb(227,179,65)")
            t3 = _Text(_home_path(), style="rgb(108,125,119)")
            return [t1, t2, t3]

        _hdr_width = {"w": 0}

        # ── Audio plane: Braille oscilloscope + protocol-adaptive PTT ──────
        # The scope fills the empty header real estate; the pump owns it and is
        # driven by the zero-copy tap on the EXISTING mic stream (CoreAudio
        # refuses a second handle). Wholly fail-soft: any fault here leaves the
        # cockpit exactly as it was without the visualizer.
        # Columns the crest already owns, plus the 2-space gap the header puts
        # between crest and text. Read from the crest itself rather than
        # guessed, so a different crest tier cannot silently overlap the wave.
        try:
            _CREST_RESERVE = int(getattr(mini, "cols", 0) or 0) + 2
        except Exception:  # noqa: BLE001
            _CREST_RESERVE = 2
        _scope_align = "right"
        _audio = {"pump": None, "latch": None, "mode": None, "unsub": None}
        try:
            from backend.core.ouroboros.ui.audio_pump import (
                AudioLevelPump, default_publisher,
            )
            from backend.core.ouroboros.ui.audio_scope import (
                AudioPlane, BrailleScope, scope_enabled, scope_placement,
                scope_width_for,
            )
            import shutil as _shutil_boot
            from backend.core.ouroboros.ui.ptt_router import (
                PTTLatch, resolve_ptt_mode,
            )
            _scope_align = scope_placement()
            if scope_enabled() and _scope_align != "off":
                # Boot width from the terminal, not a constant. The crest
                # column plus its 2-space gap is reserved so the scope never
                # collides with the identity text it sits beneath.
                _cols0 = _shutil_boot.get_terminal_size(fallback=(100, 30)).columns
                _scope = BrailleScope(
                    width=scope_width_for(_cols0, reserved=_CREST_RESERVE),
                )
                _pump = AudioLevelPump(
                    scope=_scope, publish=default_publisher(),
                )
                # Probe the terminal ONCE at boot: hold-to-talk where the kitty
                # keyboard protocol answers, toggle+VAD everywhere else.
                _mode, _verdict, _tel = resolve_ptt_mode()
                _latch = PTTLatch(
                    mode=_mode,
                    on_open=lambda: (
                        _scope.set_plane(AudioPlane.USER),
                        _pump.publish_mic_state("open"),
                    ),
                    on_close=lambda why: (
                        _scope.set_plane(AudioPlane.IDLE),
                        _pump.publish_mic_state("closed", reason=why),
                    ),
                )
                # Subscribe the pump to the zero-copy tap. RMS runs HERE, on the
                # consumer side — never in the capture thread.
                try:
                    from backend.voice.audio_broadcast_tap import get_default_tap

                    def _on_chunk(view, sr) -> None:
                        lvl = _pump.feed_frames(view, plane=_scope.plane)
                        if lvl is not None:
                            _latch.note_level(lvl)

                    _audio["unsub"] = get_default_tap().subscribe(_on_chunk)
                except Exception:  # noqa: BLE001 — no voice stack: scope stays idle
                    pass
                # ...and to the supervisor's stream, which is where the mic
                # actually lives. A KEEPER task, not a one-shot connect: the
                # host usually starts AFTER the cockpit (wake spawns it), and
                # may restart at any point in the session.
                # `_aio` is this scope's already-imported alias for asyncio.
                # The bare name was never bound here and raised NameError, so
                # the RMS keeper task was never created.
                _audio["rms_task"] = _aio.get_running_loop().create_task(
                    _keep_rms_stream(_scope, _audio),
                )
                _audio.update(pump=_pump, latch=_latch, mode=_mode)
                # `ov` has no module-level logger — this scope is the only
                # record of which PTT paradigm the terminal probe chose, and a
                # bare `logger` here resolved to nothing but a swallowed
                # NameError, so the line never emitted.
                import logging as _lg
                _lg.getLogger(__name__).debug(
                    "[ov] audio scope armed mode=%s verdict=%s terminal=%s",
                    getattr(_mode, "value", "?"),
                    getattr(_verdict, "value", "?"), (_tel or {}).get("terminal"),
                )
        except Exception:  # noqa: BLE001
            _audio = {"pump": None, "latch": None, "mode": None, "unsub": None}

        def _gutter():
            """The live scope for the header — placed by ``gutter_align``.
            None when unarmed, so the header renders exactly as before."""
            _p = _audio.get("pump")
            if _p is None:
                return None
            try:
                # Follow the terminal. header_render() has already stamped the
                # live column count for this frame, so the scope re-widths on
                # a resize instead of staying pinned at its boot width.
                _w = _hdr_width.get("w") or 0
                if _w:
                    _p.scope.set_width(
                        scope_width_for(_w, reserved=_CREST_RESERVE)
                    )
                # Gravity BEFORE paint, on the render clock rather than the
                # audio clock. When heavy STT inference starves the telemetry
                # stream the wave must keep falling; a repaint that only ever
                # drew the last received frame would freeze the trace mid-spike
                # and report "loud right now" long after the sound stopped.
                _p.scope.tick()
                return _p.scope.render_rich()
            except Exception:  # noqa: BLE001
                return None

        def header_render() -> str:
            try:
                import shutil as _shutil
                w = _shutil.get_terminal_size(fallback=(100, 30)).columns
                _hdr_width["w"] = w
                return render_cockpit_header(
                    mini, _header_lines(), w, now=_time.monotonic(),
                    right_gutter=_gutter, gutter_align=_scope_align,
                )
            except Exception:
                return ""
    except Exception:
        mini, header_render, header_height = None, None, 0

    # Spacebar PTT: merged through the layout's EXISTING extra_key_bindings
    # seam, so no layout surgery. Inert unless the input buffer is empty.
    _ptt_kb = None
    try:
        _latch = _audio.get("latch") if isinstance(_audio, dict) else None
        if _latch is not None:
            from backend.core.ouroboros.ui.ptt_router import build_ptt_key_bindings
            # HOLD-TO-TALK. A TTY sends no key-release, but the OS repeats a
            # held key — so a hold is a RATE and a release is that rate
            # stopping. The detector is created here and owned by the binding
            # for the life of the cockpit, because its whole state is the
            # arrival times of previous keystrokes; a fresh one per press
            # would have nothing to compare against.
            _hold = None
            try:
                from backend.core.ouroboros.ui.hold_to_talk import (
                    HoldDetector, hold_to_talk_enabled,
                )
                if hold_to_talk_enabled():
                    _hold = HoldDetector()
            except Exception:  # noqa: BLE001 — toggle still works without it
                _hold = None
            _ptt_kb = build_ptt_key_bindings(_latch, detector=_hold)
    except Exception:  # noqa: BLE001 — no PTT is survivable; a broken app is not
        _ptt_kb = None

    def _toolbar_with_mode() -> str:
        """Existing toolbar plus the ACTIVE PTT paradigm. Stating the real mode
        matters: a cockpit claiming 'hold' on a terminal blind to key-release
        would leave the mic latched with no obvious way out."""
        base = ""
        try:
            base = str(ui.toolbar()) if ui is not None else ""
        except Exception:  # noqa: BLE001
            base = ""
        try:
            _m = _audio.get("mode") if isinstance(_audio, dict) else None
            _l = _audio.get("latch") if isinstance(_audio, dict) else None
            if _m is not None:
                live = " ● mic" if (_l is not None and _l.is_open) else ""
                return f"{base} · {_m.hint}{live}" if base else f"{_m.hint}{live}"
        except Exception:  # noqa: BLE001
            pass
        return base

    # Client action set (trust cycle, ?, Ctrl+G, Esc-Esc rewind) merged
    # with PTT through the layout's ONE extra-bindings seam.
    _extra_kb = _client_extra_bindings(ui, client)
    if _ptt_kb is not None and _extra_kb is not None:
        try:
            from prompt_toolkit.key_binding import merge_key_bindings
            _extra_kb = merge_key_bindings([_ptt_kb, _extra_kb])
        except Exception:  # noqa: BLE001
            _extra_kb = _ptt_kb
    elif _extra_kb is None:
        _extra_kb = _ptt_kb

    # The rewind menu rides the same palette the verb completions use.
    _completer = _build_slash_completer()
    try:
        from backend.core.ouroboros.battle_test.rewind_menu import (
            merge_rewind_completer,
        )
        _completer = merge_rewind_completer(
            _completer, getattr(ui, "rewind", None),
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        def _capture_mux(m: Any) -> None:
            # The deck's own buffer, so in-flight text can be composed
            # INTO the transcript rather than approximated by a strip
            # beneath it.
            try:
                setattr(ui, "_mux", m)
            except Exception:  # noqa: BLE001
                pass

        await run_bipartite_repl(
            on_accept=_on_accept,
            on_mux=_capture_mux,
            title="◇ O+V · proactive canvas",
            toolbar=_toolbar_with_mode if ui is not None else None,
            watch_alive=_alive,
            header=header_render,
            header_height=header_height,
            extra_key_bindings=_extra_kb,
            # The `/` palette. THIS is the surface the operator actually
            # types into — the bipartite Application, not the split-plane
            # PromptSession the completer was first wired to. (Composed
            # above with the gated rewind source.)
            completer=_completer,
            # Persistent recall + history ghost-text — the same history
            # file every surface shares, so Up-arrow survives a detach.
            history=_build_prompt_history(),
            auto_suggest=_build_prompt_auto_suggest(),
            turn_spinner=getattr(ui, "turn_spinner", None),
            # WHO is working. Bound to the AttachUI rather than to the roster
            # singleton: this process dispatches nothing, so its own roster is
            # permanently empty, and rendering it would show an organism that
            # never delegates. The UI holds the daemon's snapshot instead.
            agent_rows=(
                ui._agent_lines if ui is not None
                and hasattr(ui, "_agent_lines") else None
            ),
            # The archived-diff overlay, backed by the daemon's archive over
            # the bridge. Measured UNSET here: the daemon owns the archive, so
            # this was the surface that could not draw the one thing it exists
            # to review.
            diff_rows=(
                (lambda: ui.diff_controller(client).rows())
                if ui is not None and hasattr(ui, "diff_controller")
                else None
            ),
            # The serpent runs the hairlines while the organism is THINKING.
            # Measured UNSET here by `capability_handoff`, which meant the
            # animation ran in the demo and on the daemon's own terminal and
            # was dead on the surface an operator actually attaches with — a
            # border that never moves is indistinguishable from an organism
            # that is never busy.
            serpent_active=(
                ui._serpent_active if ui is not None
                and hasattr(ui, "_serpent_active") else None
            ),
            # The `/` search bar. Read from the hatches module rather than
            # held here: the search session belongs to the transcript key
            # cluster that drives it, and a copy of its state on the UI would
            # be a second opinion about what the operator is typing.
            search_rows=_transcript_search_rows(),
            # CC's status line — phase, cost, route, warnings — from the
            # daemon's snapshot rather than this process's empty builder.
            status_rows=(
                ui._status_rows if ui is not None
                and hasattr(ui, "_status_rows") else None
            ),
            # The rejection window, directly under the caret: while it is
            # open, `/reject` is the only thing the operator may want to
            # type, and the row is about the NEXT keystroke rather than the
            # session.
            pending_rows=(
                ui._pending_apply_rows if ui is not None
                and hasattr(ui, "_pending_apply_rows") else None
            ),
            # The crashed-step confirmation. Passed by NAME like every other
            # hook: `capability_handoff` reads a `**splat` as OPAQUE, so a
            # mount that spread itself would blind the audit that proves this
            # strip reached both surfaces.
            forensic_rows=(
                ui._forensic_rows if ui is not None
                and hasattr(ui, "_forensic_rows") else None
            ),
            # The sentence being written — directly under the deck it is
            # about to become part of.
            stream_rows=(
                ui._stream_rows if ui is not None
                and hasattr(ui, "_stream_rows") else None
            ),
            # The operator's own backlog, directly above the caret where
            # they are still typing.
            queue_rows=(
                ui._input_queue_rows if ui is not None
                and hasattr(ui, "_input_queue_rows") else None
            ),
            # The crash overlay — above everything, cleared only by the
            # operator.
            panic_rows=(
                ui._panic_rows if ui is not None
                and hasattr(ui, "_panic_rows") else None
            ),
            seed=[
                "[bold]💭 Karen ▸[/bold] attached — I'm listening. verbs or "
                f"plain words both work · [{_SEM['neural']}]wake[/] arms my voice · "
                f"[{_SEM['neural']}]detach[/] leaves the organism running",
                # Boot warnings (stale-binary sentinel, …) must survive
                # the alt-screen mount — a console print alone dies with
                # the primary buffer the moment the cockpit takes over.
                *(f"[bold #e3b341]{w}[/bold #e3b341]"
                  for w in getattr(ui, "boot_warnings", ()) or ()),
            ],
        )
    finally:
        # Release the amplitude subscriptions. The tap unsubscribe was already
        # unreleased before the RMS client joined it; that one leaked only
        # until process death, but a live socket plus its read task deserves
        # an explicit close on the way out.
        try:
            _u = _audio.get("unsub") if isinstance(_audio, dict) else None
            if callable(_u):
                _u()
        except Exception:  # noqa: BLE001
            pass
        try:
            if isinstance(_audio, dict):
                _audio["closing"] = True
                _t = _audio.get("rms_task")
                if _t is not None:
                    _t.cancel()
                    try:
                        await _t
                    except (_aio.CancelledError, Exception):  # noqa: BLE001
                        pass
                _c = _audio.get("rms_client")
                if _c is not None:
                    await _c.close()
        except Exception:  # noqa: BLE001
            pass


async def _legacy_pump_loop(client: Any) -> None:
    """Non-TTY fallback: the original blocking-read-off-loop pump."""
    import asyncio

    async def _stdin_pump() -> None:
        while client.connected:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except Exception:
                break
            if not line:                   # EOF — operator closed stdin
                break
            text = line.strip()
            if text and not client.send_input(text):
                break

    pump = asyncio.get_running_loop().create_task(_stdin_pump())
    try:
        while client.connected:
            await asyncio.sleep(0.25)
    finally:
        pump.cancel()
        try:
            await pump
        except (asyncio.CancelledError, Exception):
            pass


def run_attach(console: Any) -> int:
    """``ov attach`` — hydrate, stream, and pipe stdin upstream.

    The client is a DUMB terminal: every rendered line arrives already
    conformed by the daemon's PresentationRouter chokepoint. Detach
    (Ctrl+C / EOF / daemon exit) never touches the organism."""
    import asyncio

    async def _session() -> int:
        from backend.core.ouroboros.battle_test.cockpit_attach import (
            CockpitAttachClient,
        )

        def _print_line(text: str) -> None:
            # Cockpit mounted → the daemon's bridge stream auto-scrolls into
            # Zone 1 (the Proactive Canvas). Rich markup is escaped so a daemon
            # line can never inject styling into the canvas (inert DATA).
            try:
                from backend.core.ouroboros.battle_test.bipartite_layout import (
                    get_active_canvas,
                )
                canvas = get_active_canvas()
                if canvas is not None:
                    from rich.markup import escape
                    canvas.push_raw(escape(str(text)))
                    return
            except Exception:
                pass
            # Legacy split-plane: builtin print() resolves sys.stdout
            # DYNAMICALLY — under patch_stdout this routes daemon telemetry
            # above the pinned prompt; a pre-bound Rich console would bypass
            # the patch and corrupt the input line.
            try:
                print(text)
            except Exception:
                pass

        hydrated = asyncio.Event()
        ui = AttachUI()

        def _on_hydration(payload: dict) -> None:
            _render_hydration(console, payload)
            # Health, from the SAME frame — the doctor's own verdicts on the
            # bytes already in hand. No second probe, no second thresholds,
            # so the line and `ov doctor` cannot disagree.
            try:
                if ui.advisor is not None and ui.advisor.observe_hydration(
                    payload,
                ):
                    ui.refresh()
            except Exception:  # noqa: BLE001
                pass
            try:
                state = (payload.get("audio") or {}).get("state", "")
                if state:
                    ui.on_audio_state(str(state))
            except Exception:
                pass
            hydrated.set()

        def _markup_sink(text: str, addressed: bool = False) -> None:
            """THE ambient/addressed split, at the one place both arrive.

            Addressed — the daemon answering a command this cockpit sent —
            goes to the scrollback: the operator asked for it and expects to
            scroll back to it. Ambient — a worker spawning, a provider
            failing over, the agora talking — goes to the deck, which is
            severity-ordered and ages out.

            Routed on the daemon's marker rather than inferred from the text,
            because "was this answering my command?" is not a property of the
            characters. The daemon knows; it decided when it addressed the
            frame."""
            if addressed:
                # An addressed frame IS the answer to something this
                # cockpit asked — the honest close signal. Ambient frames
                # (autonomous work) deliberately do NOT close a turn.
                try:
                    spinner = getattr(ui, "turn_spinner", None)
                    if spinner is not None:
                        spinner.note_reply()
                except Exception:  # noqa: BLE001
                    pass
                _render_markup_frame(text, console)
                return
            ui.on_ambient(text)

        client = CockpitAttachClient(
            on_hydration=_on_hydration, on_line=_print_line,
            on_markup=_markup_sink,
            on_telemetry=ui.on_telemetry,
            # Gates as DATA — the id + deadline the shield needs to defer one
            # safely and still answer the right op.
            on_prompt=lambda frame: _on_prompt_frame(ui, frame),
            # Speech, into the prompt where it can be corrected before it is
            # sent — rather than an utterance leaving unseen.
            on_transcript=lambda frame: _on_transcript_frame(ui, frame),
            on_acoustic=ui.on_acoustic,
            on_prompt_resolved=lambda pid: (
                ui.shield.dismiss(pid) if ui.shield is not None else None
            ),
            on_audio_state=ui.on_audio_state,
            on_lane_history=ui.on_lane_history,
            on_lane_reaped=ui.on_lane_reaped,
            # Another terminal's line → memory-only injection into THIS
            # process's history singleton, so Up in this pane recalls
            # what was just typed in that one (tmux split parity).
            on_history_append=_build_history_injector(),
            # The viewport lock's broadcast truth — a freeze any pane
            # causes is visible in this one.
            on_autonomy_state=lambda f: _on_autonomy_state_frame(ui, f),
            # Esc-Esc menu hydration, delivered to the controller bound
            # just below (late-bound: the controller needs the client).
            on_rewind_list=lambda f: (
                getattr(ui, "rewind", None) is not None
                and ui.rewind.deliver(f)
            ),
        )
        # The turn spinner — the live row bound to the operator's own
        # question. Reads the heartbeat this UI already retains (pure
        # pull, zero new state) and writes its tombstone through the
        # SAME addressed markup sink every ⏺/⎿ line takes.
        try:
            from backend.core.ouroboros.battle_test.turn_spinner import (
                TurnSpinner,
            )
            ui.turn_spinner = TurnSpinner(
                heartbeat_fn=lambda: getattr(ui, "_heartbeat", None),
                emit_fn=lambda line: _markup_sink(line, True),
            )
        except Exception:  # noqa: BLE001
            ui.turn_spinner = None
        # The Transactional Viewport Lock's client half. Lives on the ui
        # so both attach surfaces reach it without new plumbing.
        try:
            from backend.core.ouroboros.battle_test.rewind_menu import (
                RewindController,
            )
            ui.rewind = RewindController(
                client, notify=lambda m: ui.flash(m, seconds=3.0),
            )
        except Exception:  # noqa: BLE001
            ui.rewind = None
        # A bell that fires into tmux's void is worse than none — the
        # operator TRUSTS it. Probe once at attach and say so plainly.
        try:
            from backend.core.ouroboros.battle_test.transcript_hatches import (
                tmux_bell_warning,
            )
            _bell_warn = tmux_bell_warning()
            if _bell_warn:
                console.print(_bell_warn, markup=False, highlight=False)
        except Exception:  # noqa: BLE001
            pass
        # The stale-shim sentinel: a pyenv/pip copy shadowing the editable
        # install shows a GHOST of an older interface — old renderers, old
        # bugs, hours lost re-fixing the fixed. Said loudly, at attach, on
        # every surface (console now; cockpit seed via ui.boot_warnings).
        try:
            from backend.core.ouroboros.battle_test.daemon_provenance import (
                client_binary_warning,
            )
            _stale_bin = client_binary_warning()
            if _stale_bin:
                console.print(_stale_bin, markup=False, highlight=False)
                try:
                    ui.boot_warnings = [*getattr(ui, "boot_warnings", ()),
                                        _stale_bin]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        # The shield renders released gates through the SAME markup path
        # every other ⏺/⎿ line takes — one display, one ordering, and the
        # gate lands in scrollback the operator can page back to.
        ui.markup_sink = _markup_sink
        if not await client.connect():
            console.print(_NO_ORGANISM_MESSAGE, markup=False, highlight=False)
            return 1

        try:
            _ran_cockpit = False
            if _can_run_split_plane():
                # Style-Guide cockpit is the default interactive surface; ANY
                # failure falls through to the proven split-plane loop (the
                # cockpit can never brick the attach). Kill-switch:
                # JARVIS_BIPARTITE_LAYOUT_DISABLED=1.
                _why = ""
                _crash: Optional[BaseException] = None
                try:
                    from backend.core.ouroboros.battle_test.bipartite_layout import (
                        bipartite_enabled,
                        should_run_bipartite,
                    )
                    if should_run_bipartite():
                        await _bipartite_attach_loop(client, console, ui)
                        _ran_cockpit = True
                    elif not bipartite_enabled():
                        _why = "kill-switch (JARVIS_BIPARTITE_LAYOUT_DISABLED)"
                    else:
                        _why = "stdout is not a real TTY"
                except Exception as _exc:
                    # A software crash, NOT a hardware downgrade. Keep the
                    # exception itself: the old code truncated it to 80
                    # characters, discarded the traceback, and printed the
                    # same routine line a missing TTY produces — which is how
                    # a cockpit bug reaches an operator disguised as normal
                    # behaviour and never gets reported.
                    _ran_cockpit = False
                    _crash = _exc
                    _why = f"{type(_exc).__name__}: {str(_exc)[:80]}"
                if not _ran_cockpit:
                    # ONE seam for both causes (DRY): quiet for hardware, a
                    # banner plus a full traceback on disk for software. Both
                    # then land in the same degraded session below.
                    try:
                        from backend.core.ouroboros.battle_test.mount_breaker import (
                            announce,
                        )
                        _kind, _ = announce(
                            _crash, _why,
                            emit=lambda text: console.print(
                                text, markup=False, highlight=False,
                            ),
                        )
                        if _kind == "software":
                            # A crash mid-render leaves the terminal in an
                            # UNKNOWN state — possibly alt-screen, possibly
                            # raw mode, cursor anywhere. The parachute must
                            # therefore assume nothing about the screen and
                            # emit strictly linear plain text, exactly as it
                            # does for a terminal that cannot be addressed.
                            # Hardware downgrades that still have a good TTY
                            # keep their colour; nothing is broken there.
                            ui.degrade_to_append_only()
                    except Exception:
                        pass
                    await _split_plane_loop(client, console, ui)
            else:
                await _legacy_pump_loop(client)
            if not client.connected:
                console.print(
                    "⎿ organism went away — detached", markup=False,
                    highlight=False,
                )
        except KeyboardInterrupt:
            console.print(
                "⎿ detached (the organism keeps running)", markup=False,
                highlight=False,
            )
        finally:
            await client.close()
        return 0

    try:
        return asyncio.run(_session())
    except KeyboardInterrupt:
        try:
            console.print(
                "⎿ detached (the organism keeps running)", markup=False,
                highlight=False,
            )
        except Exception:
            pass
        return 0
    except Exception as exc:
        try:
            console.print(f"ov attach: failed ({exc})", markup=False)
        except Exception:
            pass
        return 1


# ---------------------------------------------------------------------------
# Thin-client cockpit — the sub-second `ov`
# ---------------------------------------------------------------------------


def run_cockpit_thin(console: Any) -> int:
    """The presentation-shell cockpit: instant crest, zero-trust
    probe, seamless attach — cold-booting a detached organism when
    none is home. The operator NEVER sees a traceback here.

    Borrows the terminal's ALTERNATE SCREEN before the first byte is drawn.
    The cockpit already asked for it (prompt_toolkit's ``full_screen=True``)
    — but it asked at the END, after the crest, the wake logs and the attach
    summary had been printed to the NORMAL buffer. The logo therefore sat in
    the scrollback behind the cockpit and could be scrolled back to, which is
    the one thing a full-screen takeover exists to prevent.

    Entering HERE puts the whole boot inside the borrowed buffer. On exit the
    normal buffer returns exactly as it was found, with the operator's own
    scrollback intact and no logo residue — reversible, rather than erasing
    their history with ED-3.
    """
    with alternate_screen() as _alt:
        return _run_cockpit_thin_inner(console, _alt)


def _run_cockpit_thin_inner(console: Any, alt_screen_active: bool) -> int:
    """The boot itself, running inside whatever screen it was handed."""
    import asyncio

    # The emblem law: the mark ALWAYS greets `ov`. With the Client-Side Boot
    # Animator (default on, real TTY) the mark is the ANIMATED Snake-and-Plus
    # crest — the green head + purple body chase a white `+` around the "V" — and
    # the wake logs stream into its bottom partition (a rich.live.Live managed
    # canvas, so async logs can never tear the emblem). Piped / disabled / tiny
    # terminals get the static mark exactly as before. Kill-switch:
    # JARVIS_CREST_ANIM_DISABLED=1.
    from backend.core.ouroboros.ui.crest_animator import build_animator
    _animator = build_animator(console)
    if _animator is None:
        try:
            from backend.core.ouroboros.ui.crest import print_static_crest
            print_static_crest(console)
        except Exception:
            pass
        console.print(version_line(), markup=False, highlight=False)

    async def _session() -> int:
        import asyncio as _aio
        from backend.core.ouroboros.cli.thin_client import ensure_daemon

        def _status(line: str) -> None:
            if _animator is not None:
                _animator.add_log(line)      # → the Live bottom partition
            else:
                try:
                    console.print(line, markup=False, highlight=False)
                except Exception:
                    pass

        if _animator is not None:
            # Play the chase while the daemon wakes; on daemon-up the Live exits
            # (freezing the final crest frame) and the warm attach surface prints
            # below it — seamless handoff to the interactive prompt.
            _animator.add_log(version_line())
            _stop = _aio.Event()
            _ok = {"v": False}

            async def _boot() -> None:
                try:
                    # THE SCREEN OWNER RENDERS THE GAUGE.
                    #
                    # `_status` appends to the boot transcript, which is right
                    # for events and wrong for progress — appending a gauge is
                    # what produced six stacked bars. `set_progress` is a
                    # single slot inside the same Live renderable, so it
                    # updates in place and cannot be erased by the next crest
                    # frame the way a raw stdout write was.
                    _ok["v"] = await ensure_daemon(
                        on_status=_status,
                        on_progress=_animator.set_progress,
                    )
                finally:
                    _stop.set()

            _boot_task = _aio.ensure_future(_boot())
            try:
                await _animator.play(console, stop_event=_stop)
            except Exception:
                pass
            try:
                await _boot_task
            except Exception:
                pass
            ok = _ok["v"]
        else:
            ok = await ensure_daemon(on_status=_status)

        if not ok:
            # Print below the frozen crest (add_log would land after Live exit).
            try:
                console.print(
                    "⚠ the organism did not come up — `ov daemon` in another "
                    "terminal shows the full boot, or check the daemon log.",
                    markup=False, highlight=False,
                )
            except Exception:
                pass
            return 1
        return 0

    try:
        rc = asyncio.run(_session())
    except KeyboardInterrupt:
        console.print(
            "⎿ cancelled — any background ignition continues; `ov` again "
            "to attach", markup=False, highlight=False,
        )
        return 0
    except Exception:
        return 1
    if rc != 0:
        return rc
    # Warm path from here — identical surface to `ov attach` (DRY:
    # same hydration card, same split-plane, same PresentationRouter-
    # conformed stream, same audio verbs). Still INSIDE the borrowed screen:
    # handing the terminal back between the crest and the cockpit would flash
    # the operator's shell in the middle of a boot.
    return run_attach(console)


# ---------------------------------------------------------------------------
# ov system — the System Observability Panel (Slice G)
# ---------------------------------------------------------------------------


def run_system(console: Any) -> int:
    """Attach the async System Observability cockpit to the running headless
    daemon over the Cockpit Attach UDS. Passive listener; graceful reconnect on
    daemon restart. NEVER raises out to the terminal."""
    import asyncio
    try:
        from backend.core.ouroboros.cli.ov_system_panel import run_system_panel
    except Exception as exc:  # noqa: BLE001
        try:
            console.print(f"ov system unavailable: {exc}", markup=False)
        except Exception:  # noqa: BLE001
            pass
        return 1
    try:
        return asyncio.run(run_system_panel(console=console))
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001
        return 1


def run_hive(console: Any) -> int:
    """Attach the live Agent Hive feed to the running headless daemon over the
    Cockpit Attach UDS — a read-only, chronological projection of the real O+V
    pipeline (Trinity + IDE SSE fabrics, unified by the Hive Aggregator).
    Passive listener; graceful reconnect. NEVER raises out to the terminal."""
    import asyncio
    try:
        from backend.core.ouroboros.cli.ov_hive_panel import run_hive_panel
    except Exception as exc:  # noqa: BLE001
        try:
            console.print(f"ov hive unavailable: {exc}", markup=False)
        except Exception:  # noqa: BLE001
            pass
        return 1
    try:
        return asyncio.run(run_hive_panel(console=console))
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``ov`` entry point (the ``[project.scripts]`` target).

    Returns a process exit code. Boot actions delegate into the shared
    battle-test bootstrap; status/attach/help are handled locally without
    booting the organism.
    """
    inv = resolve(sys.argv[1:] if argv is None else list(argv))
    console = build_console()

    if inv.action == "help":
        console.print(_HELP_TEXT, markup=False, highlight=False)
        return 0
    if inv.action == "version":
        console.print(version_line(), markup=False, highlight=False)
        return 0
    if inv.action == "attach":
        return run_attach(console)
    if inv.action == "system":
        return run_system(console)
    if inv.action == "hive":
        return run_hive(console)
    if inv.action == "restart":
        # Stop, then fall through to the normal cockpit path — which already
        # knows how to ignite, wait for the socket and attach. Reusing it
        # means `ov restart` and `ov` cannot drift apart in how they boot.
        def _say(text: str) -> None:
            try:
                print(text, flush=True)
            except Exception:  # noqa: BLE001
                pass
        rc = _restart_daemon(_say)
        if rc != 0:
            return rc
        inv = Invocation("cockpit", [])

    if inv.action == "demo":
        # Lazy, like `doctor`: a demo that cannot import must not be able to
        # cost the cockpit its boot path.
        try:
            from backend.core.ouroboros.cli.ov_demo import run_demo
        except Exception as exc:  # noqa: BLE001
            console.print(f"ov demo unavailable: {exc}", markup=False)
            return 1
        return run_demo(console, inv.delegate_argv)
    if inv.action == "link":
        # Lazy, like `doctor`: `link` pulls in TLS and the transport stack,
        # and a fault there must not be able to break `ov` itself.
        try:
            from backend.core.ouroboros.cli.ov_link import run_link
        except Exception as exc:  # noqa: BLE001
            console.print(f"ov link unavailable: {exc}", markup=False)
            return 1
        return run_link(console, inv.delegate_argv)
    if inv.action == "doctor":
        try:
            from backend.core.ouroboros.cli.ov_doctor import run_doctor
        except Exception as exc:  # noqa: BLE001
            console.print(f"ov doctor unavailable: {exc}", markup=False)
            return 1
        known = {"--live"}
        for arg in inv.delegate_argv:
            if arg not in known:
                hint = next((k for k in known
                             if k.startswith(arg) or arg.startswith(k)), None)
                console.print(
                    f"unknown flag {arg!r}"
                    + (f" — did you mean {hint!r}?" if hint else
                       f" (known: {', '.join(sorted(known))})"),
                    markup=False, highlight=False)
                return 64          # EX_USAGE — refuse, never silently ignore
        return run_doctor(console, live="--live" in inv.delegate_argv)
    if inv.action == "status":
        console.print(status_digest(), markup=False, highlight=False)
        return 0
    if inv.action in ("daemon_install", "daemon_uninstall"):
        from backend.core.ouroboros.cli.thin_client import (
            install_agent,
            uninstall_agent,
        )
        msg = (
            install_agent() if inv.action == "daemon_install"
            else uninstall_agent()
        )
        console.print(msg, markup=False, highlight=False)
        return 0

    # ── Thin-Client Split (operator-authorized 2026-07-18) ──────────
    # Bare `ov` is a PRESENTATION SHELL: crest + zero-trust probe +
    # attach. The organism runs in a separate execution boundary
    # (detached daemon), so the prompt is sub-second regardless of
    # domain-layer boot cost. `--legacy-boot` (or the env master off)
    # restores the in-process organism below.
    if inv.action == "cockpit" and "--legacy-boot" not in inv.delegate_argv:
        from backend.core.ouroboros.cli.thin_client import thin_client_enabled
        if thin_client_enabled():
            return run_cockpit_thin(console)

    # cockpit / headless -> the one shared bootstrap (DRY). The facade's ONLY
    # added responsibility: declare the presentation skin (spec §3.4).
    from backend.core.ouroboros.ui.presentation_mode import ENV_KEY, PresentationMode

    os.environ[ENV_KEY] = (
        PresentationMode.COCKPIT.value if inv.action == "cockpit"
        else PresentationMode.SOAK.value
    )

    # Cinematic Boot Mux (COCKPIT only): silence the TTY structurally
    # BEFORE the chatty bootstrap import chain runs. The awakening (or
    # the single-flight collision surface) releases it; a fatal boot
    # flushes the hidden buffer (Dead-Man's Switch) so forensics
    # survive the ambition.
    _mux_engaged = False
    if inv.action == "cockpit":
        try:
            from backend.core.ouroboros.ui.boot_mux import engage_boot_mux
            _mux_engaged = engage_boot_mux()
        except Exception:  # noqa: BLE001 — degrade to the noisy legacy boot
            _mux_engaged = False

    try:
        from scripts.ouroboros_battle_test import main as battle_main
        battle_main(inv.delegate_argv)
        return 0
    except SystemExit as exc:
        # 75 (EX_TEMPFAIL) is the single-flight collision — an EXPECTED
        # presentation outcome whose surface already released the mux
        # cleanly; only genuinely-unexpected nonzero exits flush.
        if _mux_engaged and exc.code not in (0, None, 75):
            _deadman_flush()
        raise
    except BaseException as exc:
        if _mux_engaged:
            _deadman_flush()
        console.print(
            f"ov: fatal during boot ({type(exc).__name__}: {exc}) — "
            "buffered logs flushed above",
            markup=False,
        )
        raise


def _deadman_flush() -> None:
    """Dead-Man's Switch — NEVER raises."""
    try:
        from backend.core.ouroboros.ui.boot_mux import release_boot_mux
        release_boot_mux(flush_to_tty=True)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())
