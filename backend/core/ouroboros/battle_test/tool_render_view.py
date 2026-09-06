"""ToolRenderView — Rich markup composition over the Slice 1-3 substrate.
==========================================================================

Slice 4 of the **Gap #2 closure arc**. This module is the **only** place
in the codebase that composes Rich markup for tool-result rendering.
Everything below it (Slices 1-3) is renderer-agnostic; everything above
it (``serpent_flow``, ``ouroboros_tui``) is a thin caller that hands
strings to ``Console.print``.

Root problem (recap)
--------------------

Two render paths today (``serpent_flow.op_tool_call`` +
``ouroboros_tui.show_tool_call``) hardcode per-tool ``if/elif`` chains
and tool-icon dicts. Slice 4 collapses both paths through a single
adaptive composer.

Slice 4 scope
-------------

* :class:`ComposedToolRender` — frozen output: pre-built Rich-markup
  strings the caller hands directly to ``Console.print`` / ``_op_line``
* :func:`compose` — load-bearing orchestrator: descriptor lookup +
  density resolution + body extraction + body parking + Rich markup
* Master flag :data:`MASTER_FLAG_ENV_VAR` (default ``"false"`` until
  Slice 5 graduation flips it true). When off, callers fall through
  to legacy paths — both paths kept compilable side-by-side.

Authority boundary
------------------

* §1 deterministic — pure markup composition; no LLM, no I/O
* §7 fail-closed — ``compose`` NEVER raises; every input is coerced /
  defaulted; on internal failure returns a minimal render so the
  caller's output is always something displayable
* §8 observable — every :class:`ComposedToolRender` carries the
  resolved :class:`DensityPolicy`; SSE / observability layers in
  Slice 5 can echo "what density was applied here and why"
"""
from __future__ import annotations

from backend.core.ouroboros.ui import theme as _theme
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

from backend.core.ouroboros.battle_test.tool_render_policy import (
    DefaultLayoutModeProvider,
    DefaultPostureProvider,
    DensityPolicy,
    LayoutModeProvider,
    PostureProvider,
    resolve_density_via_providers,
)
from backend.core.ouroboros.battle_test.tool_render_registry import (
    BodyShape,
    ToolStatus,
    get_descriptor,
    render,
)
from backend.core.ouroboros.battle_test.tool_render_store import (
    BoundedBodyStore,
    get_default_store,
)

logger = logging.getLogger("Ouroboros.ToolRenderView")


# ===========================================================================
# Schema + master flag
# ===========================================================================


TOOL_RENDER_VIEW_SCHEMA_VERSION: str = "tool_render_view.v1"


MASTER_FLAG_ENV_VAR: str = "JARVIS_TOOL_RENDER_REGISTRY_ENABLED"

# §41.3 #8 — JSON pretty-print sub-flag. Composes the master
# (off when master is off; rich rendering not in the dispatch
# path at all in that case).
JSON_PRETTY_ENABLED_ENV_VAR: str = (
    "JARVIS_TOOL_OUTPUT_JSON_PRETTY_ENABLED"
)
JSON_PRETTY_MIN_SIZE_ENV_VAR: str = (
    "JARVIS_TOOL_OUTPUT_JSON_PRETTY_MIN_SIZE"
)


def is_master_flag_enabled() -> bool:
    """Read the master flag. **Default ``true``** post Slice 5
    graduation (2026-05-04). Operators flip ``=false`` for instant
    rollback to the legacy hardcoded render paths preserved in
    ``serpent_flow.op_tool_call`` + ``ouroboros_tui.show_tool_call``.

    Re-read on every call — flips take effect immediately for the
    next tool render without restart. NEVER raises."""
    raw = os.environ.get(MASTER_FLAG_ENV_VAR, "true")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def is_json_pretty_enabled() -> bool:
    """``JARVIS_TOOL_OUTPUT_JSON_PRETTY_ENABLED``. §41.3 #8 —
    default ``true``. Gates JSON detection + pretty-printing
    + per-token coloring of tool-output bodies. Implicitly off
    when the registry master is off (no rendering path). NEVER
    raises."""
    if not is_master_flag_enabled():
        return False
    raw = os.environ.get(JSON_PRETTY_ENABLED_ENV_VAR, "true")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def json_pretty_min_size() -> int:
    """Below this byte count, JSON pretty-printing is skipped —
    tiny responses (e.g., ``{"ok": true}``) don't benefit from
    indentation. Default 60. Clamped [10, 100_000]. NEVER raises."""
    raw = os.environ.get(JSON_PRETTY_MIN_SIZE_ENV_VAR, "").strip()
    try:
        n = int(raw) if raw else 60
        if n < 10:
            return 10
        if n > 100_000:
            return 100_000
        return n
    except (TypeError, ValueError):
        return 60


# ---------------------------------------------------------------------------
# §41.3 #8 — JSON detection + pretty-printing substrate
# ---------------------------------------------------------------------------
#
# Two-tier detection per the operator binding:
#   1. Descriptor hint (``descriptor.body_lexer == "json"``) wins
#      first — when a tool's ToolRenderDescriptor declares its
#      body is JSON, trust the declaration without re-parsing.
#   2. Content auto-detect — try ``json.loads`` on the stripped
#      body. Triggered when the descriptor doesn't declare a
#      lexer OR declares something else (e.g., text bodies that
#      happen to contain JSON from a `read_file` of a `.json`).
#
# Per-token coloring runs over the pretty-printed output. Tokens
# resolve to palette keys so operators can theme JSON via the
# same palette they pass to compose() — NO hardcoded hex colors
# inside the wrapper. Falls back to sensible defaults when the
# palette doesn't carry json-specific keys.


# Palette keys for JSON tokens. Operators can override any of
# these via the palette mapping passed to :func:`compose`. The
# defaults below are tuned for dark terminals; the value-tier
# tokens stay distinct from prose body lines.
_JSON_TOKEN_PALETTE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("code_key", "cyan"),       # "key":
    ("code_str", "green"),      # "string value"
    ("code_num", "magenta"),    # 42, 3.14
    ("code_kw", "bright_black bold"),  # null / true / false
    ("code_punct", "dim"),      # { } [ ] , :
)


def _json_palette_value(
    palette: Optional[Mapping[str, str]], key: str,
) -> str:
    """Look up a JSON-token palette colour. Falls back through
    operator-passed palette → :data:`_JSON_TOKEN_PALETTE_KEYS`
    defaults → ``"white"``."""
    if palette and key in palette:
        return palette[key]
    for k, default in _JSON_TOKEN_PALETTE_KEYS:
        if k == key:
            return default
    return "white"


def _detect_json_body(
    text: object,
    *,
    body_lexer_hint: Optional[str] = None,
    min_size: Optional[int] = None,
) -> Optional[Any]:
    """Two-tier JSON detector. NEVER raises.

    Returns the parsed Python object when the body is JSON,
    ``None`` otherwise.

    Tier 1: descriptor's ``body_lexer == "json"`` — explicit hint
            from a tool's render descriptor. We still parse to
            verify the body is *valid* JSON (a tool might
            misreport its content type).
    Tier 2: content auto-detect — strip whitespace; if the body
            starts with ``{`` or ``[`` and meets the min-size
            threshold, attempt ``json.loads``. Successful parse
            → return the object.

    Below the min-size threshold the detector returns ``None``
    even when JSON parses (tiny responses look fine raw)."""
    try:
        body = str(text or "")
    except Exception:  # noqa: BLE001
        return None
    stripped = body.strip()
    if not stripped:
        return None
    min_bytes = (
        min_size if min_size is not None
        else json_pretty_min_size()
    )
    # Tier 1: explicit hint → parse anyway to validate.
    if (
        isinstance(body_lexer_hint, str)
        and body_lexer_hint.strip().lower() == "json"
    ):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    # Tier 2: auto-detect — must look like JSON AND be substantial.
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    if len(stripped) < min_bytes:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def pretty_print_json(
    parsed: object,
    *,
    indent: int = 2,
) -> str:
    """Render a Python object as pretty-printed JSON. NEVER
    raises — degrades to ``str(parsed)`` on any failure."""
    try:
        i = max(0, int(indent))
    except (TypeError, ValueError):
        i = 2
    try:
        return json.dumps(
            parsed,
            indent=i,
            ensure_ascii=False,
            sort_keys=False,
            default=str,
        )
    except (TypeError, ValueError):
        try:
            return str(parsed)
        except Exception:  # noqa: BLE001
            return ""


# Token-level regex over one JSON line. Order matters: keys are
# strings that end with ":", so the key pattern must match BEFORE
# the bare-string pattern. Numbers must match before keywords so
# ``true_count`` isn't mis-tokenized.
_JSON_LINE_TOKENIZER = re.compile(
    r"""
    (?P<key>"(?:\\.|[^"\\])*"\s*:)              # "key":
    | (?P<str>"(?:\\.|[^"\\])*")                # "string"
    | (?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) # number
    | (?P<kw>\b(?:true|false|null)\b)           # keywords
    | (?P<punct>[{}\[\],:])                     # punctuation
    """,
    re.VERBOSE,
)


def _wrap_json_line(
    line: str,
    palette: Optional[Mapping[str, str]],
) -> str:
    """Per-line JSON colorizer. NEVER raises.

    Walks the regex tokenizer left-to-right, emits Rich markup
    around each recognized token; unrecognized spans pass
    through untouched. Operates on PRETTY-PRINTED JSON lines so
    each line carries balanced quotes / punctuation — partial
    quotes from elision aren't possible since elision happens
    AFTER pretty-printing.
    """
    try:
        out: List[str] = []
        last = 0
        for match in _JSON_LINE_TOKENIZER.finditer(line):
            start, end = match.span()
            if start > last:
                out.append(_escape(line[last:start]))
            token_type = match.lastgroup or ""
            text = match.group()
            color_key = {
                "key": "code_key",
                "str": "code_str",
                "num": "code_num",
                "kw": "code_kw",
                "punct": "code_punct",
            }.get(token_type, "")
            if color_key:
                color = _json_palette_value(palette, color_key)
                out.append(f"[{color}]{_escape(text)}[/{color}]")
            else:
                out.append(_escape(text))
            last = end
        if last < len(line):
            out.append(_escape(line[last:]))
        return "".join(out)
    except Exception:  # noqa: BLE001
        # Defensive: any tokenizer pathology degrades to plain
        # text-line wrapping so the operator still sees the body.
        return _escape(line)


# ===========================================================================
# Default palette — operators can override per-call
# ===========================================================================
#
# Mirrors the canonical ``_C`` palette in ``serpent_flow.py`` so that
# headless / sandbox callers get a consistent look without depending on
# serpent_flow's privates. Production wiring (Slice 4 edit to serpent_flow)
# passes its own ``_C`` through ``palette=`` for byte-identical output.


_DEFAULT_PALETTE: Mapping[str, str] = {
    "neural": "cyan",
    "file": "blue underline",
    "dim": "dim",
    "death": "red",
    "code_add": "green",
    "code_del": "red",
    "code_hunk": "cyan",
    "heal": "yellow",
}


def _palette_value(palette: Optional[Mapping[str, str]], key: str) -> str:
    """Look up a palette colour, falling back to the default palette."""
    if palette and key in palette:
        return palette[key]
    return _DEFAULT_PALETTE.get(key, "white")


# ===========================================================================
# Frozen composition output
# ===========================================================================


@dataclass(frozen=True)
class ComposedToolRender:
    """Result of :func:`compose` — pre-built Rich markup the caller
    feeds to ``Console.print``.

    Fields
    ------
    * ``header_markup`` — first display line (e.g. CC-style
      ``[accent]⏺ Read[/accent]([muted]foo.py[/muted])  [muted]42ms[/muted]``).
    * ``summary_markup`` — body-summary line under the header,
      prefixed with the ``⎿`` continuation glyph; empty string
      when the descriptor has nothing useful to summarize.
    * ``body_lines_markup`` — one markup string per body line. The
      caller emits each via ``_op_line`` so the existing per-op
      indenting + side-rail glyphs apply uniformly.
    * ``expansion_hint`` — when the body was elided, a one-line
      ``[muted]…+N more lines elided · /expand t-12[/muted]`` hint
      so the operator knows recovery is one verb away. Empty string
      when no elision happened OR no expansion ref was issued.
    * ``policy`` — the :class:`DensityPolicy` actually applied;
      observability surfaces in Slice 5 echo this back.
    * ``confidence_band`` — §37 Tier 2 #13 Slice 5 (2026-05-07).
      Populated from the singleton ``ToolConfidenceObserver`` for
      this (``op_id``, ``tool_name``) stream when the Slice 1
      master flag is on; ``None`` otherwise. Type is the Slice 1
      ``ToolConfidenceBand`` enum (kept as ``Optional[Any]`` here
      to avoid an eager governance-tier import at module load
      time — pull the enum lazily in callers that branch on it).
      Renderers opt into displaying via
      :func:`confidence_band_markup` (additive — existing
      renderers pass through unchanged).
    """

    header_markup: str
    summary_markup: str
    body_lines_markup: Tuple[str, ...]
    expansion_hint: str
    policy: DensityPolicy
    schema_version: str = TOOL_RENDER_VIEW_SCHEMA_VERSION
    confidence_band: Optional[Any] = None


# ===========================================================================
# Helpers — body-shape-aware markup wrapping
# ===========================================================================


def _wrap_diff_line(
    line: str, palette: Optional[Mapping[str, str]],
) -> str:
    """Style a single diff line per the existing ``_C['code_add']`` /
    ``_C['code_del']`` / ``_C['code_hunk']`` convention."""
    if line.startswith("+++") or line.startswith("---"):
        return f"[{_palette_value(palette, 'dim')}]{_escape(line)}[/{_palette_value(palette, 'dim')}]"
    if line.startswith("+"):
        c = _palette_value(palette, "code_add")
        return f"[{c}]{_escape(line)}[/{c}]"
    if line.startswith("-"):
        c = _palette_value(palette, "code_del")
        return f"[{c}]{_escape(line)}[/{c}]"
    if line.startswith("@@"):
        c = _palette_value(palette, "code_hunk")
        return f"[{c}]{_escape(line)}[/{c}]"
    return _escape(line)


def _diff_wrapper_for(body_lines: Any,
                      fallback_path: Optional[str] = None) -> Any:
    """A per-line diff styler that KNOWS the file and the line numbers.

    `_wrap_diff_line` colours by prefix and nothing else, so an `Update(f.py)`
    block rendered a flat red/green wall while `deck_grammar.diff` — used by one
    scripted beat — rendered the same change with a gutter, syntax highlighting and
    a dark band. Two visual languages for one fact, in one session.

    The fix is not to thread a path down from the caller. A unified diff ALREADY
    carries both things:

        +++ b/backend/.../risk_tier_floor.py     <- the path, so the lexer
        @@ -410,7 +410,22 @@                     <- the numbering origin

    So this walks the body ONCE, derives them, and returns a closure with the same
    ``(line, palette)`` signature every other wrapper has. No new parameter, no
    caller changes, and nothing hardcoded — a diff for a `.toml` highlights as toml
    because the diff says so.

    Numbering follows the two files separately: a removed line belongs to the OLD
    side and a context or added line to the NEW one. Conflating them makes the
    gutter confidently wrong, which is worse than absent.

    Degrades in place: a body with no `@@` (a `git log`, a truncated fragment) gets
    no numbers, and a body with no `+++` gets no highlighting, both landing exactly
    on the previous flat rendering rather than on an error.
    """
    # The diff's own `+++` header is authoritative when present, but the density
    # policy STRIPS the `---`/`+++` pair before a wrapper ever sees the body — it
    # elides them as noise, correctly, since the header line duplicates the block's
    # own `Update(path)` title. So the tool's ARGUMENT is the fallback, and for
    # `edit_file` that argument IS the file. Header first, argument second: a diff
    # that carries its own path should win, because a multi-file patch's argument
    # names only one of them.
    path: Optional[str] = fallback_path or None
    try:
        for raw in list(body_lines or ()):
            text = str(raw)
            if text.startswith("+++"):
                # `+++ b/path` — strip the diff's own `b/` prefix, which is not
                # part of the filename and would defeat extension inference.
                candidate = text[3:].strip().split("\t")[0].strip()
                if candidate.startswith(("a/", "b/")):
                    candidate = candidate[2:]
                if candidate and candidate != "/dev/null":
                    path = candidate
                break
    except Exception:  # noqa: BLE001
        path = None

    state = {"old": 0, "new": 0}

    def _wrap(line: str, palette: Optional[Mapping[str, str]]) -> str:
        try:
            text = str(line)
            stripped = text.lstrip()
            if stripped.startswith("@@"):
                # Reset both counters from the hunk header. `-a,b +c,d` — the two
                # origins are independent and both are needed.
                import re as _re
                m = _re.search(r"-(\d+)(?:,\d+)?\s+\+(\d+)", stripped)
                if m:
                    state["old"] = int(m.group(1))
                    state["new"] = int(m.group(2))
                return _wrap_diff_line(text, palette)
            if stripped.startswith(("+++", "---")):
                return _wrap_diff_line(text, palette)
            from backend.core.ouroboros.ui import deck_grammar as _deck
            # None, deliberately: `deck.diff` asks `resolve_deck_width` itself, and
            # that is the ONE authority both diff surfaces now use. Resolving a
            # width here was the second opinion that gave the same hunk two right
            # edges depending on which path drew it.
            width = None
            if stripped.startswith("+"):
                lineno = state["new"] or None
                state["new"] += 1
                return _deck.diff(lineno, "+", stripped[1:], path=path,
                                  width=width)
            if stripped.startswith("-"):
                lineno = state["old"] or None
                state["old"] += 1
                return _deck.diff(lineno, "-", stripped[1:], path=path,
                                  width=width)
            # Context: advances BOTH sides, and is shown against the new file
            # because that is the one the operator is about to be living with.
            lineno = state["new"] or None
            if state["new"]:
                state["new"] += 1
            if state["old"]:
                state["old"] += 1
            body = stripped[1:] if stripped.startswith(" ") else stripped
            return _deck.diff(lineno, " ", body, path=path, width=width)
        except Exception:  # noqa: BLE001
            # The previous rendering is a perfectly good fallback; losing the diff
            # to a styling failure is not.
            return _wrap_diff_line(str(line), palette)

    return _wrap


def _wrap_log_line(
    line: str, palette: Optional[Mapping[str, str]],
) -> str:
    """Style a single log/bash output line — uniformly dim."""
    c = _palette_value(palette, "dim")
    return f"[{c}]{_escape(line)}[/{c}]"


def _wrap_text_line(
    line: str, palette: Optional[Mapping[str, str]],
) -> str:
    """Plain multi-line body — light dim styling."""
    c = _palette_value(palette, "dim")
    return f"[{c}]{_escape(line)}[/{c}]"


def _wrap_marker_line(
    line: str, palette: Optional[Mapping[str, str]],
) -> str:
    """Truncation marker (``… +N more lines elided …``)."""
    c = _palette_value(palette, "dim")
    return f"[{c} italic]{_escape(line)}[/{c} italic]"


_BODY_WRAPPERS = {
    BodyShape.DIFF: _wrap_diff_line,
    BodyShape.LOG: _wrap_log_line,
    BodyShape.MULTI_LINE: _wrap_text_line,
    BodyShape.CODE: _wrap_text_line,
    BodyShape.SINGLE_LINE: _wrap_text_line,
    # NONE never gets here — body is empty for header-only descriptors
}


def _escape(text: str) -> str:
    """Defensive escape — Rich treats ``[`` / ``]`` as markup; the
    raw tool output may contain them. We escape *everything* the
    caller hands us so we never inject unintended styles."""
    if not text:
        return ""
    return text.replace("[", "\\[")


# ===========================================================================
# Header composition — CC verb path vs. icon path
# ===========================================================================


def _arg_colour(
    arg_kind: str, palette: Optional[Mapping[str, str]],
) -> str:
    """Style for a tool's ARGUMENT, chosen by what the argument IS.

    Every argument used to take the file style, because the only descriptors
    reaching this branch were file tools. Once every tool routed through it,
    `Bash(python3 -m pytest …)` and `Search("except Exception")` came out
    blue-underlined — the convention this deck uses for a path you can open.
    A command is not a location, and underlining one invites a click that
    goes nowhere.

    Falls back to the file style for an unknown kind, matching the behaviour
    of every descriptor written before `arg_kind` existed.
    """
    try:
        return {
            "path": _palette_value(palette, "file"),
            "command": _palette_value(palette, "dim"),
            "text": _palette_value(palette, "dim"),
            "url": _palette_value(palette, "file"),
        }.get(str(arg_kind or ""), _palette_value(palette, "file"))
    except Exception:  # noqa: BLE001
        return _palette_value(palette, "file")


def _compose_header(
    rendered_header: str,
    descriptor_cc_verb: Optional[str],
    duration_ms: float,
    status_enum: ToolStatus,
    palette: Optional[Mapping[str, str]],
    tool_name: str = "",
    arg_kind: str = "text",
) -> str:
    """Build the Rich-markup header line.

    For CC-verb descriptors (Read/Update/Write):

      ``[accent]⏺ Read[/accent]([muted]foo.py[/muted])  [muted]42ms[/muted]``

    For icon descriptors (everything else):

      ``[accent]search_code "pat"[/accent]  [muted]120ms[/muted]``

    Failure status appends a [danger]✗[/danger] mark.
    """
    duration_part = ""
    if duration_ms and duration_ms > 0:
        c = _palette_value(palette, "dim")
        if duration_ms < 1000:
            duration_part = f"  [{c}]{duration_ms:.0f}ms[/{c}]"
        else:
            duration_part = f"  [{c}]{duration_ms / 1000:.1f}s[/{c}]"

    status_part = ""
    if status_enum is not ToolStatus.SUCCESS:
        c = _palette_value(palette, "death")
        status_part = f"  [{c}]✗[/{c}]"

    if descriptor_cc_verb:
        # rendered_header looks like "Read(foo.py)" — split into verb
        # + path-in-parens so we can colour each piece independently.
        verb_color = _palette_value(palette, "neural")
        arg_color = _arg_colour(arg_kind, palette)
        # Find the parens; if not present (defensive), fall back to dim.
        lparen = rendered_header.find("(")
        rparen = rendered_header.rfind(")")
        if 0 < lparen < rparen:
            verb = rendered_header[:lparen]
            inner = rendered_header[lparen + 1 : rparen]
            return (
                f"[{verb_color}]{_theme.mark('action')} {verb}[/{verb_color}]"
                f"([{arg_color}]{_escape(inner)}[/{arg_color}])"
                f"{duration_part}{status_part}"
            )
        # Fall-through: emit as a plain neural-coloured header
        return (
            f"[{verb_color}]{_theme.mark('action')} {_escape(rendered_header)}[/{verb_color}]"
            f"{duration_part}{status_part}"
        )

    # Unknown tool — MCP-forwarded, typically. SAME shape as everything
    # else: a tool this registry has never heard of is still a tool call,
    # and a distinct layout for it tells the operator about our descriptor
    # table rather than about their work.
    verb_color = _palette_value(palette, "neural")
    file_color = _arg_colour(arg_kind, palette)
    lparen = rendered_header.find("(")
    rparen = rendered_header.rfind(")")
    inner = rendered_header[lparen + 1 : rparen] if 0 < lparen < rparen else ""
    try:
        from backend.core.ouroboros.battle_test.tool_render_registry import (
            _titlecase_kind,
        )
        verb = _titlecase_kind(tool_name) if tool_name else rendered_header
    except Exception:  # noqa: BLE001
        verb = rendered_header
    return (
        f"[{verb_color}]{_theme.mark('action')} {_escape(verb)}[/{verb_color}]"
        f"([{file_color}]{_escape(inner)}[/{file_color}])"
        f"{duration_part}{status_part}"
    )


def _detail_prefix() -> str:
    """``  ⎿  `` — the deck's continuation lead, from the ONE definition.

    Composed rather than written so the glyph, the indent and the gap all
    come from `deck_grammar`; a literal here is how this line drifted to
    column 0 in the first place. NEVER raises — an unstyled fallback still
    indents, because the indent is the part that carries meaning.
    """
    try:
        from backend.core.ouroboros.ui.deck_grammar import detail
        from rich.text import Text
        # `detail("")` is the prefix and nothing else.
        return Text.from_markup(detail("")).plain
    except Exception:  # noqa: BLE001
        return "  ⎿  "


def _body_indent() -> str:
    """Leading spaces that put a body line under the summary's text.

    Body lines rendered at column 0, so a pytest traceback under a failing
    `Bash(…)` looked like console output that had escaped the block rather
    than the result OF it.
    """
    try:
        from backend.core.ouroboros.ui.deck_grammar import DETAIL_COLUMN
        return " " * int(DETAIL_COLUMN)
    except Exception:  # noqa: BLE001
        return "     "


def _compose_summary(
    summary_text: str,
    expansion_ref: Optional[str],
    elided: int,
    body_present: bool,
    palette: Optional[Mapping[str, str]],
) -> Tuple[str, str]:
    """Build (summary_markup, expansion_hint).

    ``summary_markup`` uses the CANONICAL continuation glyph, resolved
    from ``ui.theme`` rather than written here.

    It used to be a literal — and the literal was ``⏎`` (U+23CE RETURN
    SYMBOL) while all four docstrings in this module and its registry said
    ``⎿`` (U+23BF). So every tool result in the cockpit continued with a
    different mark from every other continuation line in the deck, and the
    documentation asserted otherwise. A one-character typo is exactly what a
    duplicated glyph literal is FOR: nothing compares the copies, so nothing
    notices. Asking the theme also buys the ASCII degradation that the deck's
    other continuations already have on a non-UTF-8 terminal.

    ``expansion_hint`` is a separate dim line emitted only when the
    body was actually elided (``elided > 0``) AND a stable ref was
    issued — without a ref the operator has no recovery path, so
    surfacing the hint would be misleading.
    """
    if not summary_text:
        return ("", "")

    c_dim = _palette_value(palette, "dim")
    # INDENTED under the action, through the module that owns the deck's
    # column discipline.
    #
    # This line sat at column 0 while `deck_grammar` put every other
    # continuation at column 2 with its body at column 5 — so one deck
    # showed `⏺ Repair(L2)` / `  ⎿ fixture rebuilt` directly above
    # `⏺ Update(f.py)` / `⎿ +5 / -2` with the diff flat beneath it. Two
    # column disciplines, and the block structure is the indentation: remove
    # it and a transcript is a log.
    summary_markup = (
        f"[{c_dim}]{_detail_prefix()}{_escape(summary_text)}[/{c_dim}]"
    )

    expansion_hint = ""
    if elided > 0 and isinstance(expansion_ref, str) and expansion_ref:
        expansion_hint = (
            f"[{c_dim} italic]   … +{elided} more line"
            f"{'s' if elided != 1 else ''} parked · /expand {expansion_ref}"
            f"[/{c_dim} italic]"
        )
    elif body_present and elided == 0:
        # No elision needed; no hint.
        pass

    return (summary_markup, expansion_hint)


# ===========================================================================
# The load-bearing composer
# ===========================================================================


def compose(
    tool_name: str,
    args_str: str,
    result_str: str,
    *,
    status: object = ToolStatus.SUCCESS,
    duration_ms: float = 0.0,
    op_id: str = "",
    round_index: int = 0,
    palette: Optional[Mapping[str, str]] = None,
    posture_provider: Optional[PostureProvider] = None,
    layout_provider: Optional[LayoutModeProvider] = None,
    store: Optional[BoundedBodyStore] = None,
    explicit_density: Optional[DensityPolicy] = None,
) -> ComposedToolRender:
    """Compose Rich markup for one tool call.

    Pipeline:

      1. Resolve the :class:`ToolRenderDescriptor` for ``tool_name``
         (Slice 1; falls back to the default descriptor for unknown
         tools — MCP-forwarded tools land here).
      2. Resolve the :class:`DensityPolicy` from posture × layout ×
         env (Slice 2). Callers that already have a policy (e.g.
         test harnesses) pass ``explicit_density`` to skip resolution.
      3. Park the *full* body in :class:`BoundedBodyStore` (Slice 3)
         when the body shape supports a body block AND the body
         exceeds the density budget. Passing ``store=None`` skips
         parking but everything else still works (graceful degrade).
      4. Run :func:`tool_render_registry.render` to produce the
         bounded :class:`RenderedToolResult`.
      5. Wrap each piece in Rich markup using the supplied palette
         (or the default mirror).

    NEVER raises — every step has a documented degradation.
    """
    # --- 1. Descriptor lookup
    descriptor = get_descriptor(tool_name)

    # --- 1b. §41.3 #8 — JSON detection on the FULL body BEFORE
    # render() applies the line cap. Two-tier: descriptor hint
    # wins first; falls back to content auto-detect. When
    # detected, we replace result_str with the pretty-printed
    # form so render()'s bounding caps the indented output.
    # The `_json_detected` flag downstream switches the per-line
    # wrapper to the token-colorizer. NEVER raises into render().
    _json_detected = False
    if is_json_pretty_enabled() and result_str:
        _parsed = _detect_json_body(
            result_str, body_lexer_hint=descriptor.body_lexer,
        )
        if _parsed is not None:
            _pretty = pretty_print_json(_parsed)
            if _pretty:
                result_str = _pretty
                _json_detected = True

    # --- 2. Density resolution
    policy: DensityPolicy
    if isinstance(explicit_density, DensityPolicy):
        policy = explicit_density
    else:
        policy = resolve_density_via_providers(
            posture_provider or DefaultPostureProvider(),
            layout_provider or DefaultLayoutModeProvider(),
        )

    # --- 3. Park body (issues expansion ref) if eligible
    expansion_ref: Optional[str] = None
    body_eligible = (
        descriptor.body_shape is not BodyShape.NONE
        and policy.show_body
        and bool(result_str)
    )
    if body_eligible and store is not None:
        try:
            stored = store.store(
                op_id=op_id,
                round_index=round_index,
                tool_name=tool_name,
                body=result_str,
                summary="",  # filled-in below; see step 4
                lexer=descriptor.body_lexer,
            )
            expansion_ref = stored.ref
        except Exception:  # noqa: BLE001
            logger.debug(
                "[ToolRenderView] body park failed", exc_info=True,
            )

    # --- 4. Bounded render
    try:
        rendered = render(
            descriptor,
            args_str,
            result_str,
            status,
            max_body_lines=policy.max_body_lines,
            expansion_ref=expansion_ref,
        )
    except Exception:  # noqa: BLE001
        logger.debug("[ToolRenderView] render failed", exc_info=True)
        return _empty_render(policy)

    status_enum = ToolStatus.coerce(status)

    # --- 5. Markup composition
    header_markup = _compose_header(
        rendered.header_line,
        descriptor.cc_verb,
        duration_ms,
        status_enum,
        palette,
        # The tool the CALLER asked for. An unknown tool resolves to the
        # default descriptor, whose own `tool_kind` is the string
        # "default" — so the registry layer cannot name it and an
        # MCP-forwarded call rendered as `Default(#eng)`. This layer has
        # the real name and is the only one that does.
        tool_name=str(tool_name or ""),
        arg_kind=getattr(descriptor, "arg_kind", "text"),
    )

    body_present = bool(rendered.body_lines)
    summary_markup, expansion_hint = _compose_summary(
        rendered.body_summary,
        expansion_ref,
        rendered.elided_line_count,
        body_present,
        palette,
    )

    # Body lines wrapped per shape. §41.3 #8 — when JSON was
    # detected in step 1b, override the shape-dispatched wrapper
    # with the JSON token-colorizer so keys/strings/numbers/
    # keywords/punctuation get distinct markup. Elision markers
    # still route through _wrap_marker_line uniformly.
    if _json_detected:
        wrapper = _wrap_json_line
    else:
        wrapper = _BODY_WRAPPERS.get(
            descriptor.body_shape, _wrap_text_line,
        )
        # A diff body is the one shape whose styling needs CONTEXT — the file it
        # belongs to and where its hunks start — and both are carried in the body
        # itself. Built per compose, so the numbering cannot leak between blocks.
        if wrapper is _wrap_diff_line:
            wrapper = _diff_wrapper_for(rendered.body_lines,
                                        fallback_path=str(args_str or "") or None)
    # ONE seam for the indent, not four wrappers each remembering to.
    #
    # Every body line lands in the deck's body column, so a pytest traceback
    # under a failing `Bash(…)` reads as the result OF that call rather than
    # as console output that escaped the block. The wrappers keep owning
    # COLOUR; this owns POSITION, and neither has to know the other's rules.
    _indent = _body_indent()
    body_lines_markup: Tuple[str, ...] = tuple(
        _indent + (
            _wrap_marker_line(ln, palette)
            if "elided" in ln and ln.lstrip().startswith("…")
            else wrapper(ln, palette)
        )
        for ln in rendered.body_lines
    )

    # §37 Tier 2 #13 Slice 5 (2026-05-07) — composer pulls the
    # last-observed confidence band for this (op_id, tool_name)
    # stream from the canonical singleton observer. Master-flag-
    # gated (zero observer-touch when off). NEVER raises.
    confidence_band = _read_confidence_band_for_compose(
        op_id=op_id, tool_name=tool_name,
    )

    return ComposedToolRender(
        header_markup=header_markup,
        summary_markup=summary_markup,
        body_lines_markup=body_lines_markup,
        expansion_hint=expansion_hint,
        policy=policy,
        confidence_band=confidence_band,
    )


def _read_confidence_band_for_compose(
    *, op_id: str, tool_name: str,
) -> Optional[Any]:
    """§37 Tier 2 #13 Slice 5 — read the last-observed confidence
    band for this (op_id, tool_name) stream from the canonical
    singleton ``ToolConfidenceObserver``.

    Composition discipline (operator binding 2026-05-07):
      * Single source of truth — composes
        :func:`get_default_observer` (no parallel observer
        construction; AST-pinned).
      * Master-flag-gated — when
        ``JARVIS_TOOL_CONFIDENCE_INDICATOR_ENABLED`` is off,
        skips the lookup entirely (zero observer touch).
      * Lazy import — keeps tool_render_view's module-load
        cycle clean (no eager governance import).
      * NEVER raises — defensive at every step. Returns
        ``None`` on any failure.
    """
    if not op_id or not tool_name:
        return None
    try:
        from backend.core.ouroboros.governance.tool_confidence_warning_observer import (  # noqa: E501
            get_default_observer,
            master_enabled,
        )
    except ImportError:
        return None
    try:
        if not master_enabled():
            return None
    except Exception:  # noqa: BLE001 — defensive
        return None
    try:
        return get_default_observer().last_band(
            f"{op_id}::{tool_name}",
        )
    except Exception:  # noqa: BLE001 — defensive
        return None


# ===========================================================================
# Confidence-band markup helper — opt-in for renderers
# ===========================================================================


# Color discipline (§37.9 invariant #3): low-confidence bands map
# to dim/yellow/red — semantically consistent with "warning" /
# "uncertain" without violating the no-bright_green rule. CERTAIN
# and HIGH return empty string (silent on safe pole — chatter-
# suppressed by default).
_BAND_GLYPH_PALETTE_KEY: Mapping[str, str] = {
    "medium": "dim",       # Hedged — barely worth showing
    "low": "code_del",     # Unsafe pole — yellow/red per palette
    "unknown": "code_del",  # Same severity bucket as LOW
}

# Glyph: a single dim question mark expresses "uncertain" without
# crowding the header. Renderers can override by computing their
# own markup from the band field.
_BAND_GLYPH = "?"


def confidence_band_markup(
    band: Optional[Any],
    palette: Optional[Mapping[str, str]] = None,
) -> str:
    """Render a single-glyph confidence indicator for renderers
    that opt into displaying per-tool confidence.

    Returns an empty string when:
      * ``band`` is ``None`` (no signal observed).
      * Band is CERTAIN or HIGH (safe pole — silent by design,
        mirrors the chatter-suppressed first-observation
        discipline in Slice 1).

    Returns a Rich-markup glyph (e.g.
    ``" [code_del]?[/code_del]"``) for MEDIUM / LOW / UNKNOWN,
    using the existing palette tokens (no new color discipline).
    Renderers append the result to their header_markup or
    summary line as they see fit. Pure function — no I/O,
    no env reads. NEVER raises."""
    if band is None:
        return ""
    try:
        band_value = getattr(band, "value", "")
    except Exception:  # noqa: BLE001 — defensive
        return ""
    if not isinstance(band_value, str):
        return ""
    palette_key = _BAND_GLYPH_PALETTE_KEY.get(band_value)
    if palette_key is None:
        # CERTAIN / HIGH / unrecognized → silent.
        return ""
    color = _palette_value(palette, palette_key)
    return f" [{color}]{_BAND_GLYPH}[/{color}]"


def _empty_render(policy: DensityPolicy) -> ComposedToolRender:
    """Defensive minimum so :func:`compose` never raises."""
    return ComposedToolRender(
        header_markup="",
        summary_markup="",
        body_lines_markup=(),
        expansion_hint="",
        policy=policy,
    )


# ===========================================================================
# Convenience: master-flag-aware shim for caller migration
# ===========================================================================


def compose_if_enabled(
    tool_name: str,
    args_str: str,
    result_str: str,
    **kwargs,
) -> Optional[ComposedToolRender]:
    """Return composed markup ONLY if the master flag is on.

    Slice 4 caller migration pattern::

        composed = compose_if_enabled(tool_name, args, result, ...)
        if composed is not None:
            self._op_line(op_id, composed.header_markup)
            if composed.summary_markup:
                self._op_line(op_id, composed.summary_markup)
            for line in composed.body_lines_markup:
                self._op_line(op_id, line)
            if composed.expansion_hint:
                self._op_line(op_id, composed.expansion_hint)
            return  # short-circuit legacy path

    This shim keeps the legacy + new paths cleanly separated; the
    Slice 5 graduation just flips the master flag default.
    """
    if not is_master_flag_enabled():
        return None
    try:
        return compose(tool_name, args_str, result_str, **kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("[ToolRenderView] compose_if_enabled failed", exc_info=True)
        return None


def store_for_view(*, override: Optional[BoundedBodyStore] = None) -> BoundedBodyStore:
    """Resolve the body store: explicit override > default singleton.

    Useful for callers that want to inject a per-session store
    (e.g. cleared on session end) rather than the process singleton.
    """
    if isinstance(override, BoundedBodyStore):
        return override
    return get_default_store()


__all__ = [
    "ComposedToolRender",
    "MASTER_FLAG_ENV_VAR",
    "TOOL_RENDER_VIEW_SCHEMA_VERSION",
    "compose",
    "compose_if_enabled",
    "is_master_flag_enabled",
    "register_flags",
    "register_shipped_invariants",
    "store_for_view",
]


# ===========================================================================
# Slice 5 — FlagRegistry self-registration (auto-discovered via the
# battle_test entry in ``_FLAG_PROVIDER_PACKAGES``)
# ===========================================================================


def register_flags(registry) -> int:
    """Module-owned FlagRegistry registration. Returns count of
    FlagSpecs added. NEVER raises — graduation soak path is fail-open."""
    try:
        from backend.core.ouroboros.governance.flag_registry import (
            Category, FlagSpec, FlagType,
        )
    except ImportError:
        return 0

    specs = [
        FlagSpec(
            name=MASTER_FLAG_ENV_VAR,
            type=FlagType.BOOL,
            default=True,
            description=(
                "Master kill switch for the ToolRenderRegistry "
                "rendering path (Gap #2). When false, both "
                "``serpent_flow.op_tool_call`` and "
                "``ouroboros_tui.show_tool_call`` fall through to "
                "the legacy hardcoded ``if/elif`` chains preserved "
                "below the master-flag guards. Default TRUE post "
                "graduation 2026-05-04. Re-read on every render — "
                "flips take effect immediately for the next tool "
                "call without restart."
            ),
            category=Category.SAFETY,
            source_file=(
                "backend/core/ouroboros/battle_test/tool_render_view.py"
            ),
            example="true",
            since="Gap #2 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_TOOL_RENDER_DENSITY",
            type=FlagType.STR,
            default="",
            description=(
                "Operator override for adaptive density resolution. "
                "When set to ``compact`` / ``balanced`` / ``verbose``, "
                "skips the (Posture × LayoutKind) table lookup and "
                "applies the named level directly. Unrecognized / "
                "blank values are silently ignored (table lookup "
                "applies). Useful for screen-recording sessions "
                "(``verbose``) or stabilization sprints (``compact``)."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/tool_render_policy.py"
            ),
            example="balanced",
            since="Gap #2 Slice 5 (2026-05-04)",
        ),
        FlagSpec(
            name="JARVIS_TOOL_RENDER_STORE_SIZE",
            type=FlagType.INT,
            default=50,
            description=(
                "Capacity of the BoundedBodyStore (Slice 3) — the "
                "session-scoped ring of full tool-result bodies "
                "parked behind ``/expand <ref>`` recovery hints. "
                "Drop-oldest eviction; clamped to [1, 10_000]. "
                "Increase for deep-explore sessions, decrease for "
                "memory-tight environments."
            ),
            category=Category.TUNING,
            source_file=(
                "backend/core/ouroboros/battle_test/tool_render_store.py"
            ),
            example="50",
            since="Gap #2 Slice 5 (2026-05-04)",
        ),
    ]

    count = 0
    for spec in specs:
        try:
            registry.register(spec)
            count += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "[ToolRenderView] flag registration failed for %s",
                getattr(spec, "name", "?"), exc_info=True,
            )
    return count


# ===========================================================================
# Slice 5 — shipped_code_invariants self-registration (auto-discovered
# via the battle_test entry in ``_INVARIANT_PROVIDER_PACKAGES``)
# ===========================================================================


def register_shipped_invariants() -> list:
    """Module-owned shipped-code AST pins. Three structural
    invariants that lock in Gap #2's correctness-critical surfaces:

      1. ``tool_render_view_public_surface`` — the load-bearing
         exports (``compose``, ``compose_if_enabled``,
         ``is_master_flag_enabled``) MUST remain present. Renaming
         any of them silently breaks the call-site wiring.
      2. ``tool_render_registry_descriptor_completeness`` — the
         ``_DESCRIPTORS`` dict MUST contain entries for every Venom
         tool that production callers can render through. This is
         the structural guarantee behind "no hardcoded ``if/elif``
         chains downstream" — missing a descriptor would silently
         route the tool through the default fallback and lose its
         CC-style verb formatting.
      3. ``tool_render_policy_di_cage`` — ``tool_render_policy.py``
         MUST NOT top-level-import the stateful posture surfaces
         (``posture_observer``, ``posture_store``, ``posture_health``).
         Lazy imports inside ``Default*Provider`` methods are
         allowed; top-level imports would defeat the layered
         design and create import-time circular-dependency risk.

    NEVER raises (returns ``[]`` on import failure). Per house style.
    """
    try:
        from backend.core.ouroboros.governance.meta.shipped_code_invariants import (  # noqa: E501
            ShippedCodeInvariant,
        )
    except ImportError:
        return []

    import ast as _ast

    # ---- Pin 1: public surface of tool_render_view -----------------------

    _REQUIRED_VIEW_EXPORTS = (
        "compose", "compose_if_enabled", "is_master_flag_enabled",
    )

    def _validate_view_public_surface(tree, _source) -> tuple:
        del _source
        seen: set = set()
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef):
                seen.add(node.name)
        missing = [
            name for name in _REQUIRED_VIEW_EXPORTS if name not in seen
        ]
        if missing:
            return (
                f"tool_render_view.py missing required public "
                f"functions: {missing} — call sites in serpent_flow "
                "+ ouroboros_tui depend on these exact names",
            )
        return ()

    # ---- Pin 2: descriptor completeness in tool_render_registry ----------

    _REQUIRED_DESCRIPTORS = frozenset({
        "read_file", "list_symbols", "search_code", "run_tests",
        "get_callers", "glob_files", "list_dir", "git_log", "git_diff",
        "git_blame", "bash", "edit_file", "write_file", "delete_file",
        "type_check", "web_fetch", "web_search", "ask_human",
    })

    def _validate_descriptor_completeness(tree, _source) -> tuple:
        del _source
        # Walk for ``_DESCRIPTORS`` assignment (Mapping[str, ...] dict literal).
        descriptor_keys: set = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.AnnAssign) or isinstance(node, _ast.Assign):
                targets = (
                    [node.target]
                    if isinstance(node, _ast.AnnAssign)
                    else node.targets
                )
                for tgt in targets:
                    if (
                        isinstance(tgt, _ast.Name)
                        and tgt.id == "_DESCRIPTORS"
                    ):
                        value = node.value
                        if isinstance(value, _ast.Dict):
                            for key_node in value.keys:
                                if isinstance(key_node, _ast.Constant) and isinstance(
                                    key_node.value, str,
                                ):
                                    descriptor_keys.add(key_node.value)
        missing = _REQUIRED_DESCRIPTORS - descriptor_keys
        if missing:
            return (
                f"_DESCRIPTORS missing entries for: {sorted(missing)} — "
                "every Venom tool kind must have an explicit descriptor "
                "(Gap #2 Slice 1 contract)",
            )
        return ()

    # ---- Pin 3: DI cage on tool_render_policy ----------------------------

    _FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset({
        "backend.core.ouroboros.governance.posture_observer",
        "backend.core.ouroboros.governance.posture_store",
        "backend.core.ouroboros.governance.posture_health",
    })

    def _validate_policy_di_cage(tree, _source) -> tuple:
        del _source
        violations = []
        # Only inspect TOP-LEVEL import statements; ignore those inside
        # function bodies (lazy imports are intentional and required).
        for node in tree.body:
            if isinstance(node, _ast.ImportFrom):
                module = node.module or ""
                if module in _FORBIDDEN_TOP_LEVEL_IMPORTS:
                    violations.append(
                        f"top-level ``from {module} import ...`` "
                        "violates the DI cage — use a lazy import "
                        "inside Default*Provider.current() instead"
                    )
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TOP_LEVEL_IMPORTS:
                        violations.append(
                            f"top-level ``import {alias.name}`` "
                            "violates the DI cage"
                        )
        return tuple(violations)

    return [
        ShippedCodeInvariant(
            invariant_name="tool_render_view_public_surface",
            target_file=(
                "backend/core/ouroboros/battle_test/tool_render_view.py"
            ),
            description=(
                "tool_render_view.py must expose the load-bearing "
                "compose / compose_if_enabled / is_master_flag_enabled "
                "functions. Renaming any of them silently breaks the "
                "lazy-imported call-site wiring in serpent_flow + "
                "ouroboros_tui."
            ),
            validate=_validate_view_public_surface,
        ),
        ShippedCodeInvariant(
            invariant_name="tool_render_registry_descriptor_completeness",
            target_file=(
                "backend/core/ouroboros/battle_test/tool_render_registry.py"
            ),
            description=(
                "_DESCRIPTORS must cover every Venom tool kind. "
                "Missing a descriptor silently routes that tool "
                "through the default fallback and loses CC-verb "
                "formatting — defeats Gap #2's 'no hardcoded "
                "if/elif chains downstream' contract."
            ),
            validate=_validate_descriptor_completeness,
        ),
        ShippedCodeInvariant(
            invariant_name="tool_render_policy_di_cage",
            target_file=(
                "backend/core/ouroboros/battle_test/tool_render_policy.py"
            ),
            description=(
                "tool_render_policy.py must NOT top-level-import "
                "the stateful posture surfaces "
                "(posture_observer / posture_store / posture_health). "
                "Lazy imports inside Default*Provider methods are "
                "allowed; top-level imports defeat the layered "
                "design and create circular-dependency risk."
            ),
            validate=_validate_policy_di_cage,
        ),
    ]
