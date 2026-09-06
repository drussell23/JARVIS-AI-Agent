"""backend/core/ouroboros/ui/theme.py -- the unified theme engine.

Single source of truth for O+V's CLI styling. Every component references a
*semantic token* (``accent``, ``muted``, ``danger`` ...) -- never a literal
color -- and the theme resolves that token to a concrete Rich style based on
the terminal's detected capability tier.

Design (spec 2026-07-06 §3):

* **One accent color** (restrained cyan-teal ``#3AAFA9``) degraded across
  four capability tiers: TRUECOLOR -> 256 -> 16 -> stripped.
* **One Console factory** (:func:`build_console`) -- the only place a themed
  :class:`rich.console.Console` is constructed. ``force_tier`` overrides
  detection for tests + the ``JARVIS_UI_THEME_FORCE_TIER`` debug override.
* **Glyph degradation** (:func:`mark`) -- unicode marks fall back to ASCII.
* **Zero escape leakage** at the NONE tier (bulletproof mandate #4): the
  console's ``color_system`` is ``None`` so Rich emits no ANSI at all.

Dependency rule: stdlib + Rich only. No ``governance``/``battle_test`` imports.
"""
from __future__ import annotations

import enum
import logging
import os
import sys
from typing import Callable, Mapping, Optional

from rich.console import Console
from rich.theme import Theme

logger = logging.getLogger("Ouroboros.UI.Theme")

#: Debug override -- force a specific tier regardless of terminal detection.
FORCE_TIER_ENV_VAR: str = "JARVIS_UI_THEME_FORCE_TIER"

#: The one brand accent, as a truecolor hex. Single point of retune.
ACCENT_HEX: str = "#3AAFA9"


# ===========================================================================
# Capability tiers
# ===========================================================================


class ColorTier(enum.IntEnum):
    """Terminal color capability, ordered least -> most capable.

    Ordered (IntEnum) so callers can compare, e.g. ``tier >= ColorTier.C256``.
    """

    NONE = 0       # NO_COLOR / pipe / dumb term -> styles stripped
    STANDARD = 1   # 8/16 color
    C256 = 2       # 256 color
    TRUECOLOR = 3  # 24-bit


_COLOR_SYSTEM_TO_TIER: Mapping[Optional[str], ColorTier] = {
    None: ColorTier.NONE,
    "truecolor": ColorTier.TRUECOLOR,
    "256": ColorTier.C256,
    "standard": ColorTier.STANDARD,
    "windows": ColorTier.STANDARD,
}

_TIER_TO_COLOR_SYSTEM = {
    ColorTier.NONE: None,
    ColorTier.STANDARD: "standard",
    ColorTier.C256: "256",
    ColorTier.TRUECOLOR: "truecolor",
}


def detect_tier(console: Console) -> ColorTier:
    """Map a console's ``color_system`` to a :class:`ColorTier`.

    Defensive: any unexpected value degrades to ``STANDARD`` (colored but
    conservative); a ``None`` color system is the NONE tier. NEVER raises.
    """
    try:
        cs = getattr(console, "color_system", None)
    except Exception:  # noqa: BLE001 -- never raise into a render path
        cs = None
    if cs in _COLOR_SYSTEM_TO_TIER:
        return _COLOR_SYSTEM_TO_TIER[cs]
    return ColorTier.STANDARD


def _forced_tier_from_env() -> Optional[ColorTier]:
    """Resolve ``JARVIS_UI_THEME_FORCE_TIER`` to a tier, or ``None``.

    Accepts the tier name (case-insensitive) or its integer value. Invalid
    values are ignored (auto-detect). NEVER raises.
    """
    raw = os.environ.get(FORCE_TIER_ENV_VAR)
    if not raw:
        return None
    key = raw.strip().upper()
    try:
        if key in ColorTier.__members__:
            return ColorTier[key]
        return ColorTier(int(key))
    except (ValueError, KeyError):
        return None


def _declared_glyph_support() -> Optional[bool]:
    """What the terminal THAT WILL DISPLAY THIS declared, or ``None``.

    Tri-state on purpose. ``supports_wide_glyphs()`` collapses "nothing
    attached" into ``True`` because a renderer needs an answer; here the
    difference is load-bearing — "no cockpit declared" must fall through to
    the locale, while "a cockpit declared narrow" must override a UTF-8
    daemon locale.

    Lazy + fail-soft: `ui/` sits below `battle_test/`, so this is a runtime
    consultation rather than a module-level dependency. Absent the capability
    layer — the client process, a bare import, a test — it returns ``None``
    and `supports_unicode` behaves exactly as it did before.
    """
    try:
        from backend.core.ouroboros.battle_test.terminal_capabilities import (
            current_capabilities,
        )
        caps = current_capabilities()
        return None if caps is None else bool(caps.wide_glyphs)
    except Exception:  # noqa: BLE001 — glyph choice must never raise
        return None


def supports_unicode(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the terminal that will DISPLAY this can render the glyph.

    The daemon renders for a terminal it does not own. Consulting its own
    ``LC_ALL``/``LANG`` answers the wrong question: a daemon launched under
    ``LANG=C`` degrades every glyph to ASCII while the operator watches a
    UTF-8 cockpit, and a UTF-8 daemon emits ``⏺``/``⎿`` into an ASCII
    terminal as mojibake — which is worse, because a misrendered gutter
    misaligns every line beneath it.

    Resolution order, measured before assumed:

    1. **An explicit ``env`` always wins.** Callers passing a mapping are
       asking a deterministic question about a locale, and this stays a pure
       function of that mapping — the existing test contract is unchanged.
    2. **The attached cockpit's declaration**, when one exists. Per-subscriber
       for addressed output; for ambient output `terminal_capabilities`
       already ANDs across live cockpits, so one ASCII terminal degrades the
       shared line — an aligned ASCII gutter beats a misaligned pretty one.
    3. **The local locale**, unchanged. This is the answer in the CLIENT
       process (which owns a real terminal and has no subscribers) and in a
       foreground daemon with nothing attached.

    An env with no UTF-8 hint still yields ``False`` — the conservative
    choice that degrades glyphs to ASCII. NEVER raises.
    """
    if env is None:
        declared = _declared_glyph_support()
        if declared is not None:
            return declared
    e = os.environ if env is None else env
    for key in ("LC_ALL", "LC_CTYPE", "LANG"):
        val = e.get(key, "") or ""
        if "utf" in val.lower():
            return True
    return False


#: DEC private mode 2026 — synchronized output. Between BEGIN and END the
#: terminal buffers everything it receives and paints once, so a region
#: erased and redrawn in one burst is never seen half-drawn.
SYNC_BEGIN = "\x1b[?2026h"
SYNC_END = "\x1b[?2026l"


def supports_synchronized_output(
    env: Optional[Mapping[str, str]] = None, *, is_tty: Optional[bool] = None,
) -> bool:
    """Should a live region bracket each repaint in synchronized output?

    ## Why this exists

    `rich.live.Live` repaints a region by moving the cursor up and rewriting
    every line. Nothing groups the erase and the rewrite into one paint, so a
    terminal can show the erased region before the new frame lands. At the
    crest animation's fourteen frames a second, on a block twenty-odd rows
    tall, that gap is the flicker the operator sees at every boot. DEC mode
    2026 is the terminal feature built for exactly this, and Rich does not
    emit it.

    ## Why this is not a capability table

    A private-mode sequence a terminal does not implement is ignored — that
    is what makes the modes private. So the honest answer is "emit whenever
    a real terminal is on the other end", not "emit for terminals on a list
    someone remembers to update". The only terminals worth excluding are
    the ones that are not VT terminals at all, which `TERM` already names.

    Resolution order, same shape as :func:`supports_unicode`:

    1. **An explicit override wins.** ``JARVIS_SYNC_OUTPUT`` set to a
       falsy value disables it; a truthy value forces it. Absent means
       decide.
    2. **No terminal, no brackets.** Written into a pipe or a log they are
       bytes of noise. ``is_tty`` is injectable so the decision is a pure
       function under test.
    3. ``TERM`` of ``dumb`` (or unset) is not a VT terminal.

    Pure when given ``env`` and ``is_tty``. NEVER raises.
    """
    try:
        e = os.environ if env is None else env
        raw = str(e.get("JARVIS_SYNC_OUTPUT", "") or "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        if is_tty is None:
            try:
                is_tty = bool(sys.stdout.isatty())
            except Exception:  # noqa: BLE001
                is_tty = False
        if not is_tty:
            return False
        term = str(e.get("TERM", "") or "").strip().lower()
        return bool(term) and term != "dumb"
    except Exception:  # noqa: BLE001 — a probe never raises
        return False


# ===========================================================================
# Semantic tokens + per-tier resolution table
# ===========================================================================


class Token(str, enum.Enum):
    """Semantic style names. Components reference these, never colors."""

    ACCENT = "accent"     # interactive verbs, prompt glyph, active op-id
    HEADING = "heading"   # titles -- weight, not color
    BODY = "body"         # primary text (default fg)
    MUTED = "muted"       # subtitles, context line, separators, hints
    SUCCESS = "success"   # OUTCOMES ONLY (apply/verify OK) -- reserved
    WARNING = "warning"   # soft warnings
    DANGER = "danger"     # errors, rollback
    RULE = "rule"         # hairline separators


# Per-tier resolved Rich style strings. NONE maps everything to "" so the
# style is a no-op (and the console's color_system=None strips ANSI anyway).
_TOKEN_TABLE: Mapping[ColorTier, Mapping[Token, str]] = {
    ColorTier.TRUECOLOR: {
        Token.ACCENT: ACCENT_HEX,
        Token.HEADING: "bold",
        Token.BODY: "default",
        Token.MUTED: "grey50",
        Token.SUCCESS: "green",
        Token.WARNING: "yellow",
        Token.DANGER: "red",
        Token.RULE: "grey50",
    },
    ColorTier.C256: {
        Token.ACCENT: "color(73)",   # nearest xterm-256 to #3AAFA9
        Token.HEADING: "bold",
        Token.BODY: "default",
        Token.MUTED: "grey50",
        Token.SUCCESS: "green",
        Token.WARNING: "yellow",
        Token.DANGER: "red",
        Token.RULE: "grey50",
    },
    ColorTier.STANDARD: {
        Token.ACCENT: "cyan",
        Token.HEADING: "bold",
        Token.BODY: "default",
        Token.MUTED: "dim",
        Token.SUCCESS: "green",
        Token.WARNING: "yellow",
        Token.DANGER: "red",
        Token.RULE: "dim",
    },
    ColorTier.NONE: {t: "" for t in Token},
}


def style_for(token: Token, tier: ColorTier) -> str:
    """Resolve a semantic token to a concrete Rich style string for a tier.

    Returns ``""`` at the NONE tier (or for any unmapped combination). NEVER
    raises -- an unknown token degrades to empty rather than erroring a render.
    """
    return _TOKEN_TABLE.get(tier, {}).get(token, "")


def _theme_for(tier: ColorTier) -> Theme:
    """Build a Rich :class:`Theme` mapping every token name to its resolved
    style for the given tier. Empty styles become ``"none"`` (a valid null
    Rich style) so ``Theme`` construction never rejects them.
    """
    styles_map = {t.value: (style_for(t, tier) or "none") for t in Token}
    return Theme(styles_map)


_active_tier_cache: Optional[ColorTier] = None


def active_tier() -> ColorTier:
    """Detect + cache the current terminal's color tier from a probe console.

    Lets renderables that do NOT own a console (e.g. ``diff_preview`` returns a
    Panel printed elsewhere) resolve tokens to concrete styles for the live
    environment via :func:`styles`. Cached for the session. NEVER raises.
    """
    global _active_tier_cache
    if _active_tier_cache is None:
        try:
            _active_tier_cache = detect_tier(Console())
        except Exception:  # noqa: BLE001
            _active_tier_cache = ColorTier.STANDARD
    return _active_tier_cache


def reset_active_tier_cache() -> None:
    """Test isolation for :func:`active_tier`."""
    global _active_tier_cache
    _active_tier_cache = None


def styles(tier: Optional[ColorTier] = None):
    """Return ``{Token: concrete Rich style string}`` for a tier.

    Default tier is :func:`active_tier`. Because the values are *concrete*
    (hex / xterm / base name / empty), a renderable that embeds them renders
    correctly on any console -- themed or not -- while still sourcing every
    style from the one token table (DRY). NEVER raises.
    """
    t = tier if tier is not None else active_tier()
    return {tok: style_for(tok, t) for tok in Token}


# ===========================================================================
# Glyph marks (unicode -> ASCII degradation)
# ===========================================================================


#: (unicode, ascii) pairs. ASCII fallback preserves meaning without codepoints.
_GLYPHS: Mapping[str, tuple] = {
    "dot": ("·", "-"),      # middot separator
    "check": ("✓", "OK"),   # done
    "cross": ("✗", "X"),    # failed
    "arrow": ("›", ">"),    # prompt / pointer
    "rule": ("─", "-"),     # hairline
    # Design-language semantic ration (OV_DESIGN_LANGUAGE.md §2) —
    # the SIX operator-plane glyphs, each with an ASCII degradation
    # so 16-color/none terminals keep identical geometry.
    "action": ("⏺", "*"),   # an actor did/does something
    "detail": ("⎿", "-"),   # continuation under an action
    "voice": ("💭", "K:"),  # the organism speaking
    "human": ("🗣", "you:"),  # the operator's words echoed
    "warn": ("⚠", "!"),     # operator-notable degradation
    "audio": ("🎙", "mic"),  # live audio-plane state
    # Style Guide §04 glyph grammar — the proactive-cockpit vocabulary. Each maps
    # to ONE meaning across the whole organism; ASCII degradation keeps geometry.
    "ignite": ("⚡", "!"),   # a high-energy autonomous fire (AWE launch)
    "tick": ("◈", "+"),     # one unit of map-reduce progress (chunk commit)
    "cycle": ("↻", "~"),    # resume / restart / self-heal
    "poison": ("☠", "X"),   # quarantined / dead-lettered (DLQ)
    "state_on": ("◆", "#"),  # a lifecycle transition took hold (armed)
    "state_off": ("◇", "o"),  # a lifecycle stood down (disarmed)
}


def mark(name: str, *, unicode: Optional[bool] = None) -> str:
    """Return a glyph by semantic name, degraded to ASCII when needed.

    ``unicode=None`` (default) consults :func:`supports_unicode`; pass an
    explicit bool in tests. An unknown mark name returns ``""``.
    """
    pair = _GLYPHS.get(name)
    if pair is None:
        return ""
    use_unicode = supports_unicode() if unicode is None else unicode
    return pair[0] if use_unicode else pair[1]


# ===========================================================================
# Ouroboros spinner — THE organism's identity animation
# ===========================================================================
#
# The snake closing on its own tail (5→0 dots), the bite (🐍◯), reopen.
# CANONICAL here (design-as-code, Style Guide §04): serpent_flow's REPL
# spinner and the attach-heartbeat cockpit pulse both consume this ONE
# definition — the organism animates identically on every surface.
# Clock-stateless: the frame is a pure function of monotonic time, so any
# number of readers agree with zero coordination and zero tick tasks.

OUROBOROS_SPINNER_INTERVAL_S: float = 0.10
#: The organism's glyph, in Claude Code's SLOT.
#:
#: CC's working line is `✽ Synthesizing…` — one cell of glyph, then a verb.
#: This used to be `🐍·····○` closing to `🐍◯`: the snake eating its own tail,
#: eight cells wide, animating the tail rather than the word.
#:
#: The operator's rule is CC's grammar with O+V's content, and here that
#: resolves cleanly — the SHAPE is CC's one-glyph slot, the GLYPH is ours.
#: An emoji cannot morph the way `✻✽✳✶` does, so the motion moved to where CC
#: also puts it: the elapsed seconds, and a verb that changes as the organism
#: works. A spinner that animates a decoration while the words stand still is
#: telling the operator about a timer, not about the work.
#:
#: Kept as a tuple with a single entry rather than collapsing to a constant,
#: so `ouroboros_frame()` keeps its signature and every consumer — the daemon
#: toolbar, the attach pulse, the demo — inherits this with no edit. A second
#: glyph could be added here and animate again with no change to a renderer.
OUROBOROS_SPINNER_FRAMES: tuple = ("🐍",)
#: ASCII degradation (same geometry, same story) for non-unicode terminals.
_OUROBOROS_ASCII_FRAMES: tuple = ("~",)


def ouroboros_frame(
    now: Optional[float] = None, *, unicode: Optional[bool] = None,
) -> str:
    """The current Ouroboros spinner frame, derived from the clock.

    ``now`` overrides monotonic time (tests / synchronized surfaces);
    ``unicode=None`` consults :func:`supports_unicode`. NEVER raises."""
    try:
        import time as _time
        t = _time.monotonic() if now is None else float(now)
        use = supports_unicode() if unicode is None else bool(unicode)
        frames = OUROBOROS_SPINNER_FRAMES if use else _OUROBOROS_ASCII_FRAMES
        return frames[int(t / OUROBOROS_SPINNER_INTERVAL_S) % len(frames)]
    except Exception:  # noqa: BLE001
        return "🐍"


# ===========================================================================
# Console factory + primitives
# ===========================================================================


def build_console(
    *,
    force_tier: Optional[ColorTier] = None,
    **console_kwargs: object,
) -> Console:
    """Construct the one themed :class:`rich.console.Console`.

    This is the single chokepoint for console construction across the CLI --
    every surface should obtain its console here so styling stays uniform
    (DRY mandate #3).

    ``force_tier`` (explicit arg or the ``JARVIS_UI_THEME_FORCE_TIER`` env
    override) pins the tier and the console's ``color_system``; otherwise the
    tier is auto-detected from the constructed console and the matching theme
    is pushed. NEVER raises -- on any failure it returns a plain Console.
    """
    tier = force_tier if force_tier is not None else _forced_tier_from_env()
    try:
        if tier is not None:
            console = Console(
                color_system=_TIER_TO_COLOR_SYSTEM[tier],
                theme=_theme_for(tier),
                **console_kwargs,  # type: ignore[arg-type]
            )
        else:
            console = Console(**console_kwargs)  # type: ignore[arg-type]
            console.push_theme(_theme_for(detect_tier(console)))
        _mark_themed(console)
        return console
    except Exception:  # noqa: BLE001 -- construction must never crash the CLI
        logger.debug("[theme] build_console failed; plain fallback", exc_info=True)
        return Console(**console_kwargs)  # type: ignore[arg-type]


def _mark_themed(console: object) -> None:
    """Tag a console as already carrying the O+V theme (idempotency marker)."""
    try:
        setattr(console, "_ov_themed", True)
    except Exception:  # noqa: BLE001
        pass


def ensure_theme(console: Console) -> Console:
    """Push the tier-appropriate theme onto a console if it lacks one.

    Idempotent -- consoles built via :func:`build_console` are already tagged
    and skipped. Lets any consumer safely use ``[accent]`` markup regardless of
    how its console was constructed (DRY -- the render helpers call this).
    NEVER raises.
    """
    try:
        if getattr(console, "_ov_themed", False):
            return console
        push = getattr(console, "push_theme", None)
        if callable(push):
            push(_theme_for(detect_tier(console)))
            _mark_themed(console)
    except Exception:  # noqa: BLE001
        logger.debug("[theme] ensure_theme failed", exc_info=True)
    return console


def box_for(tier: ColorTier):
    """Return the box style for a tier: rounded when unicode is available,
    ASCII otherwise (bulletproof mandate #4). ``tier`` reserved for future
    per-tier box tuning. NEVER raises."""
    from rich import box
    try:
        return box.ROUNDED if supports_unicode() else box.ASCII
    except Exception:  # noqa: BLE001
        return box.ASCII


def render_panel(
    console: Console,
    body: object,
    *,
    token: Token = Token.MUTED,
    title: Optional[str] = None,
) -> None:
    """Draw a bordered panel styled by a semantic token.

    The single panel-drawing primitive -- consumers must not hand-roll their
    own Panel/box logic (DRY mandate #3). Box degrades to ASCII without
    unicode; border color degrades with the tier. NEVER raises.
    """
    from rich.panel import Panel
    ensure_theme(console)
    try:
        console.print(Panel(
            body,
            border_style=token.value,
            box=box_for(detect_tier(console)),
            title=title,
            expand=False,
            padding=(0, 2),
        ))
    except Exception:  # noqa: BLE001
        logger.debug("[theme] render_panel failed", exc_info=True)
        try:
            console.print(body)
        except Exception:  # noqa: BLE001
            pass


def render_rule(console: Console, label: Optional[str] = None) -> None:
    """Draw a width-measuring hairline separator via the ``rule`` token.

    Replaces every hardcoded ``"-" * N`` in the codebase: Rich measures the
    live console width. Degrades the rule glyph to ASCII when the locale lacks
    UTF-8. NEVER raises.
    """
    char = mark("rule") or "-"
    try:
        console.rule(label or "", characters=char, style="rule")
    except Exception:  # noqa: BLE001
        logger.debug("[theme] render_rule failed", exc_info=True)
        try:
            width = min(int(getattr(console, "width", 80) or 80), 80)
            console.print(char * width)
        except Exception:  # noqa: BLE001
            pass


# ===========================================================================
# Style Guide palette + severity ladder (OV_STYLE_GUIDE.md §02, §05)
#
# The DEFINITIVE palette lives here as truecolor hex — the single source of
# truth the /breadcrumbs registry imports (DRY mandate: zero duplicate color or
# glyph literals anywhere else). Every value degrades across the same four tiers
# as the semantic tokens above.
# ===========================================================================


#: Named brand + semantic hexes (truecolor). Retune the whole CLI from here.
PALETTE: Mapping[str, str] = {
    "ground": "#0A0E0D", "surface": "#111917", "hairline": "#1E2B28",
    "ink": "#DBE6E1", "muted": "#6C7D77", "faint": "#47554F",
    "venom_green": "#5EE06A", "venom_purple": "#A371F7", "cyan": "#43D6D0",
    "crit": "#F85149", "warn": "#E3B341", "info": "#58B0F8", "ok": "#3FB950",
}


# Severity rank → (glyph name, semantic style) — the "derive, don't pick" table.
# Ranks mirror event_breadcrumb_registry SEV_* (0 verbose … 3 critical). The
# registry imports SEVERITY_GLYPH / severity_style from here so the mapping is
# defined exactly ONCE.
SEVERITY_GLYPH: Mapping[int, str] = {3: "✖", 2: "▲", 1: "·", 0: "·"}

_SEVERITY_STYLE_TRUE: Mapping[int, str] = {
    3: "bold #F85149", 2: "#E3B341", 1: "#58B0F8", 0: "#6C7D77",
}
_SEVERITY_STYLE_C256: Mapping[int, str] = {
    3: "bold red", 2: "yellow", 1: "color(75)", 0: "grey50",
}
_SEVERITY_STYLE_STD: Mapping[int, str] = {
    3: "bold red", 2: "yellow", 1: "cyan", 0: "bright_black",
}


def severity_style(rank: int, tier: Optional[ColorTier] = None) -> str:
    """Resolve a severity rank (0–3) to a concrete Rich style for the active (or
    given) tier — truecolor hex → 256 → 16-color → stripped. NEVER raises."""
    t = tier if tier is not None else active_tier()
    try:
        r = int(rank)
    except (TypeError, ValueError):
        r = 1
    if t >= ColorTier.TRUECOLOR:
        return _SEVERITY_STYLE_TRUE.get(r, _SEVERITY_STYLE_TRUE[1])
    if t == ColorTier.C256:
        return _SEVERITY_STYLE_C256.get(r, _SEVERITY_STYLE_C256[1])
    if t == ColorTier.STANDARD:
        return _SEVERITY_STYLE_STD.get(r, _SEVERITY_STYLE_STD[1])
    return ""  # NONE tier — no color


# Named semantic colors the registry references for its tailored per-event
# overrides (e.g. a CRITICAL-urgent AWE launch styled venom-green, not red).
# Values are tier-resolved so a descriptor never carries a raw literal.
_SEMANTIC_TRUE: Mapping[str, str] = {
    # Elevated severity: an emphatic warning wanting the operator's eye
    # NOW — distinct from `warn` (routine caution) and `crit` (failed).
    # Registered HERE because this module owns colour; a token-layer
    # override would be a second palette.
    "alert": "#E3B341", "highlight": "#FFFFFF",
    "crit": "#F85149", "crit_bold": "bold #F85149", "warn": "#E3B341",
    "info": "#58B0F8", "verbose": "#6C7D77", "muted": "#6C7D77",
    "ok": "#3FB950", "ok_bold": "bold #5EE06A", "venom_green": "#5EE06A",
    "venom_purple": "#A371F7", "cyan": "#43D6D0", "cyan_bold": "bold #43D6D0",
    # The §09 ground/ink ladder, which the palette defined but nothing could
    # ASK for. Without a resolvable name, "primary text" had to be spelled as
    # no style at all -- and an unstyled line takes the terminal profile's
    # foreground, so a green-on-black terminal rendered the whole deck in the
    # one colour §08 reserves for outcomes. Naming it is what makes the
    # dim -> ink -> accent hierarchy expressible.
    "ink": PALETTE["ink"], "faint": PALETTE["faint"],
}
_SEMANTIC_STD: Mapping[str, str] = {
    "alert": "bright_yellow", "highlight": "bright_white",
    "crit": "red", "crit_bold": "bold red", "warn": "yellow",
    "info": "cyan", "verbose": "bright_black", "muted": "dim",
    "ok": "green", "ok_bold": "bold green", "venom_green": "green",
    "venom_purple": "magenta", "cyan": "cyan", "cyan_bold": "bold cyan",
    # 16 colours cannot say #DBE6E1. "default" is the honest degradation --
    # the terminal's own foreground -- rather than picking white and fighting
    # a palette we cannot see.
    "ink": "default", "faint": "bright_black",
}


def semantic(name: str, tier: Optional[ColorTier] = None) -> str:
    """Resolve a named semantic color to a concrete Rich style for the tier.
    Truecolor → hex; standard/256 → the 8/16-color ANSI equivalent; NONE → "".
    An unknown name degrades to muted. NEVER raises."""
    t = tier if tier is not None else active_tier()
    if t <= ColorTier.NONE:
        return ""
    table = _SEMANTIC_TRUE if t >= ColorTier.C256 else _SEMANTIC_STD
    return table.get(name, table.get("muted", ""))


# ===========================================================================
# The Reactive Theme Singleton (OV_STYLE_GUIDE.md §06 — State-Reactive border)
#
# A state-aware, in-memory-mutable palette. The organism is PROACTIVE; this glass
# is REACTIVE — a broker state-transition mutates the active accent + fires the
# registered invalidate hooks (a lightweight in-place redraw), NEVER a teardown /
# rebuild of the prompt_toolkit Application. Decoupled from any Application: the
# app REGISTERS its invalidate; the theme calls it. Fully headless-testable.
# ===========================================================================


class UIState(str, enum.Enum):
    """The organism's meta-state, as reflected by the cockpit's accent."""
    DORMANT = "DORMANT"     # idle / disarmed
    ARMED = "ARMED"         # Supervisor armed, watching
    SOAKING = "SOAKING"     # a checkpointed swarm soak is running
    DEGRADED = "DEGRADED"   # a provider's inference lane is down
    HEALTHY = "HEALTHY"     # a provider recovered / live


# UIState → semantic color name (resolved per-tier via semantic()).
_STATE_ACCENT: Mapping[UIState, str] = {
    UIState.DORMANT: "muted",
    UIState.ARMED: "warn",          # amber
    UIState.SOAKING: "venom_green",
    UIState.DEGRADED: "crit",       # red
    UIState.HEALTHY: "cyan",
}


# event_type (+ payload) → UIState. The reactive mapping; feeding ANY of these
# events to on_event mutates the active accent.
def _state_for_event(event_type: str, payload: dict) -> Optional[UIState]:
    et = (event_type or "").strip()
    if et == "provider_state_changed":
        st = str((payload or {}).get("state", "")).upper()
        if st == "DEGRADED":
            return UIState.DEGRADED
        if st == "HEALTHY":
            return UIState.HEALTHY
        return None
    if et in ("supervisor_armed",):
        return UIState.ARMED
    if et in ("supervisor_disarmed",):
        return UIState.DORMANT
    if et in ("awe_soak_launched", "soak_resumed", "soak_chunk_committed",
              "soak_manifest_enqueued"):
        return UIState.SOAKING
    if et in ("awe_soak_complete", "soak_run_complete"):
        return UIState.HEALTHY
    return None


class ReactiveTheme:
    """The mutable, state-aware accent + a set of invalidate hooks. Thread-safe
    for the register/notify path. NEVER raises into a render or event path."""

    def __init__(self, *, initial: UIState = UIState.DORMANT) -> None:
        import threading
        self._state: UIState = initial
        self._lock = threading.Lock()
        self._invalidators: list = []
        self._transitions: int = 0

    # -- state -----------------------------------------------------------

    @property
    def state(self) -> UIState:
        return self._state

    @property
    def transitions(self) -> int:
        return self._transitions

    def active_border_style(self, tier: Optional[ColorTier] = None) -> str:
        """The current border/accent as a concrete Rich style for the tier —
        the property the canvas reads each render. NEVER raises."""
        try:
            name = _STATE_ACCENT.get(self._state, "cyan")
            return semantic(name, tier)
        except Exception:  # noqa: BLE001
            return ""

    def set_state(self, state: UIState) -> bool:
        """Set the meta-state. On a genuine transition, mutate in place and fire
        the invalidate hooks (a lightweight redraw — NO Application rebuild).
        Returns True iff the state actually changed. NEVER raises."""
        with self._lock:
            if state == self._state:
                return False
            self._state = state
            self._transitions += 1
            hooks = list(self._invalidators)
        for fn in hooks:
            try:
                fn()
            except Exception:  # noqa: BLE001 — a bad hook never blocks a transition
                logger.debug("[theme] invalidate hook raised", exc_info=True)
        return True

    def on_event(self, event_type: str, payload: Optional[dict] = None) -> bool:
        """Consume a broker event; if it maps to a meta-state, transition (which
        invalidates). Returns whether the accent changed. NEVER raises."""
        try:
            st = _state_for_event(event_type, payload or {})
        except Exception:  # noqa: BLE001
            st = None
        if st is None:
            return False
        return self.set_state(st)

    # -- invalidate hooks (decoupled from the Application) --------------

    def register_invalidate(self, fn) -> "Callable[[], None]":
        """Register a zero-arg redraw callback (e.g. ``app.invalidate``). Returns
        an unregister thunk. The theme holds NO reference to the Application
        itself — only this callable — so it is fully decoupled. NEVER raises."""
        with self._lock:
            self._invalidators.append(fn)

        def _unregister() -> None:
            with self._lock:
                try:
                    self._invalidators.remove(fn)
                except ValueError:
                    pass
        return _unregister

    def clear_invalidators(self) -> None:
        with self._lock:
            self._invalidators.clear()


_reactive_theme: Optional[ReactiveTheme] = None


def get_reactive_theme() -> ReactiveTheme:
    """Process-local singleton — the ONE reactive accent shared by the broker
    listeners (writers) and the canvas / Application (reader + invalidate)."""
    global _reactive_theme
    if _reactive_theme is None:
        _reactive_theme = ReactiveTheme()
    return _reactive_theme


def reset_reactive_theme() -> None:
    """Test isolation — drop the singleton so a fresh one is built."""
    global _reactive_theme
    _reactive_theme = None


__all__ = [
    "ACCENT_HEX",
    "FORCE_TIER_ENV_VAR",
    "PALETTE",
    "SEVERITY_GLYPH",
    "ColorTier",
    "ReactiveTheme",
    "Token",
    "UIState",
    "active_tier",
    "box_for",
    "build_console",
    "detect_tier",
    "ensure_theme",
    "get_reactive_theme",
    "mark",
    "render_panel",
    "render_rule",
    "reset_active_tier_cache",
    "reset_reactive_theme",
    "semantic",
    "severity_style",
    "style_for",
    "styles",
    "supports_unicode",
]


def cockpit_prompt_style(tier: Optional[ColorTier] = None) -> Any:
    """The cockpit's prompt_toolkit ``Style`` — O+V palette, CC restraint.

    prompt_toolkit ships a completion menu that looks like a Windows listbox:
    a filled light-grey block with reverse-video selection. Claude Code's
    reads as part of the terminal because it has NO fill — the names sit on
    the ground colour and only the selected row is tinted. That is the whole
    difference, and it is styling, not layout.

    Derived from :data:`PALETTE` rather than restating hexes: the brand owns
    those six colours in exactly one place, so a palette change moves the menu
    with it. Degrades by tier — a 16-colour terminal gets named colours rather
    than truecolor hexes that would quantize to mud.

    NEVER raises: an unstyled menu is ugly, a crashed cockpit is unusable."""
    try:
        from prompt_toolkit.styles import Style
    except ImportError:
        return None
    t = tier if tier is not None else active_tier()
    try:
        if t >= ColorTier.TRUECOLOR:
            p = PALETTE
            ground, surface = p["ground"], p["surface"]
            ink, muted, faint = p["ink"], p["muted"], p["faint"]
            green, purple = p["venom_green"], p["venom_purple"]
            rules = [
                # `bg:default` — the TERMINAL's background, stated EXPLICITLY.
                #
                # Two wrong answers preceded this one. `bg:{ground}` paints a
                # slab of our own dark grey. Then OMITTING bg entirely, which
                # looks like it should mean transparent and does not: an
                # unspecified background INHERITS, and prompt_toolkit's own
                # default for `completion-menu` is a light grey fill — so
                # removing our colour handed the palette to pt's, which is
                # louder than what it replaced.
                #
                # Only `bg:default` resolves to the terminal's real
                # background, which is what makes the palette read as part of
                # the page rather than a control sitting on top of it.
                ("completion-menu", f"bg:default {muted}"),
                ("completion-menu.completion", f"bg:default {ink}"),
                # Selection is carried by the TEXT, not by a filled block.
                # A green slab behind one row is the loudest thing on the
                # screen and reads as a widget; Claude marks the current entry
                # by brightening it, so the eye tracks colour and the row
                # stays part of the same page.
                ("completion-menu.completion.current",
                 f"bg:default {green} bold"),
                ("completion-menu.meta.completion", f"bg:default {purple}"),
                ("completion-menu.meta.completion.current",
                 f"bg:default {green}"),
                ("completion-menu.multi-column-meta", f"bg:default {purple}"),
                ("scrollbar.background", "bg:default"),
                ("scrollbar.button", f"bg:default {purple}"),
                # The default bottom-toolbar is reverse-video — a solid bar of
                # colour across the width, which is the single loudest thing
                # on the screen and says nothing.
                ("bottom-toolbar", f"bg:{ground} {faint} noreverse"),
                ("bottom-toolbar.text", f"bg:{ground} {faint}"),
                ("command-deck", f"bg:{ground} {ink}"),
            ]
        else:
            rules = [
                ("completion-menu", "bg:default fg:gray"),
                ("completion-menu.completion", "bg:default"),
                ("completion-menu.completion.current",
                 "bg:default fg:ansibrightgreen bold"),
                ("completion-menu.meta.completion", "bg:default fg:ansimagenta"),
                ("completion-menu.meta.completion.current",
                 "bg:default fg:ansimagenta"),
                ("scrollbar.background", "bg:default"),
                ("scrollbar.button", "bg:ansimagenta"),
                ("bottom-toolbar", "bg:default fg:gray noreverse"),
                ("bottom-toolbar.text", "bg:default fg:gray"),
                ("command-deck", "bg:default"),
            ]
        return Style(rules)
    except Exception:  # noqa: BLE001
        return None
