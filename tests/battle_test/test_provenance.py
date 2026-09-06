"""A transcript that can tell you how it knows what it says.

Every line rendered with equal authority: a test that ran, an AST walk, a
model's assertion, a cap invented when a scan timed out. The organism
computes the difference — `ReasonProvenance`, `Advisory.blast_provenance` —
and drops it one frame short of the eye it was computed for. That is how a
FATAL overlay came to print `origin: ?` in the same green as everything it
had actually measured.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from backend.core.ouroboros.ui import provenance as pv


class TestTheLadder:
    def test_strength_is_ordered(self):
        """Ordinal so strength COMPOSES. Unordered, the weakest-link
        judgement would land in every producer."""
        assert (pv.Provenance.OBSERVED > pv.Provenance.DERIVED
                > pv.Provenance.STATED > pv.Provenance.MODELED
                > pv.Provenance.SYNTHETIC > pv.Provenance.UNKNOWN)

    def test_nesting_takes_the_WEAKEST(self):
        """A model's prose quoted inside a measured block does not become
        measured by being nested."""
        with pv.claiming(pv.Provenance.OBSERVED):
            with pv.claiming(pv.Provenance.MODELED) as inner:
                assert inner is pv.Provenance.MODELED
            # ...and the reverse order gives the same answer.
        with pv.claiming(pv.Provenance.MODELED):
            with pv.claiming(pv.Provenance.OBSERVED) as inner:
                assert inner is pv.Provenance.MODELED

    def test_context_restores_via_TOKEN(self):
        """Restoring a value read at entry would leak one task's footing
        into another under concurrency; the token is the only correct undo.
        Asserted structurally because the race it prevents is not
        reproducible on demand."""
        src = pathlib.Path(pv.__file__).read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "claiming")
        calls = {ast.unparse(n.func) for n in ast.walk(fn)
                 if isinstance(n, ast.Call)}
        assert "_ACTIVE.reset" in calls
        assert "_ACTIVE.set" in calls

    def test_context_is_restored_after_exit(self):
        assert pv.active() is None
        with pv.claiming(pv.Provenance.MODELED):
            assert pv.active() is pv.Provenance.MODELED
        assert pv.active() is None

    def test_it_follows_a_task_across_await(self):
        """contextvars propagate into tasks at creation — the property that
        makes ambient provenance work at all in an async organism."""
        async def _inner():
            await asyncio.sleep(0)
            return pv.active()

        async def _outer():
            with pv.claiming(pv.Provenance.SYNTHETIC):
                return await asyncio.create_task(_inner())

        assert asyncio.run(_outer()) is pv.Provenance.SYNTHETIC

    def test_a_sibling_task_is_NOT_contaminated(self):
        async def _outer():
            with pv.claiming(pv.Provenance.SYNTHETIC):
                pass
            return pv.active()
        assert asyncio.run(_outer()) is None


class TestUnsetIsNotUnknown:
    """The distinction the default rests on."""

    def test_unset_renders_CLEAN(self):
        """Defaulting the unexamined to UNKNOWN would badge the whole
        transcript into noise."""
        assert pv.annotate("plain line") == "plain line"
        assert pv.mark_for(None).marked is False

    def test_unknown_renders_MARKED(self):
        """Defaulting it to OBSERVED would be the fabrication this exists
        to end. Asked-and-failed is a warning; not-asked is not."""
        assert pv.mark_for(pv.Provenance.UNKNOWN).marked is True
        assert "unverified" in pv.annotate("x", pv.Provenance.UNKNOWN)

    def test_UNSET_is_not_a_ladder_member(self):
        assert pv.UNSET is None
        assert None not in list(pv.Provenance)


class TestScarcity:
    @pytest.mark.parametrize("clean", [pv.Provenance.OBSERVED,
                                       pv.Provenance.DERIVED])
    def test_what_the_organism_SAW_is_unmarked(self, clean):
        """A transcript is presumed to be what the organism observed;
        annotating that presumption everywhere spends the reader's
        attention on the ordinary."""
        assert pv.annotate("ran 12 tests", clean) == "ran 12 tests"

    @pytest.mark.parametrize("weak", [pv.Provenance.STATED,
                                      pv.Provenance.MODELED,
                                      pv.Provenance.SYNTHETIC,
                                      pv.Provenance.UNKNOWN])
    def test_everything_WEAKER_earns_a_mark(self, weak):
        assert pv.annotate("a claim", weak) != "a claim"

    def test_marks_use_semantic_ROLES_not_colours(self):
        """A second palette here would drift from `theme` exactly as
        `serpent_flow._C` did."""
        src = pathlib.Path(pv.__file__).read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "mark_for")
        assert "sem" in {ast.unparse(n.func) for n in ast.walk(fn)
                         if isinstance(n, ast.Call)}


class TestTheProjectionCoversTheREALVocabularies:
    """Derived from the live enums, so a new member breaks this test rather
    than silently going unprojected — the drift that made `/narrate` push
    four flags at subsystems reading none of them."""

    def test_every_ReasonProvenance_member_projects(self):
        from backend.core.ouroboros.governance.inline_approval import (
            ReasonProvenance,
        )
        for member in ReasonProvenance:
            assert pv.project(member) is not None, (
                f"ReasonProvenance.{member.name} has no projection"
            )

    def test_unstated_projects_to_SYNTHETIC_not_STATED(self):
        """The subtle one. The DECISION was a human's; the reason STRING is
        a fallback the code supplied. Provenance marks the claim, and the
        claim is the text — presenting it as the operator's word is exactly
        the fabrication `_reject_args` was rewritten to stop."""
        from backend.core.ouroboros.governance.inline_approval import (
            ReasonProvenance,
        )
        assert pv.project(ReasonProvenance.UNSTATED) is pv.Provenance.SYNTHETIC
        assert pv.project(ReasonProvenance.STATED) is pv.Provenance.STATED

    def test_the_advisor_vocabulary_projects(self):
        """`Advisory.blast_provenance`'s documented values."""
        assert pv.project("measured") is pv.Provenance.OBSERVED
        assert pv.project("localized_lower_bound") is pv.Provenance.DERIVED
        assert pv.project("synthetic_cap") is pv.Provenance.SYNTHETIC
        assert pv.project("unknown") is pv.Provenance.UNKNOWN

    def test_the_ladder_round_trips_its_OWN_labels(self):
        """Caught by rendering the legend: `project` accepted every foreign
        vocabulary and rejected its own. Only three of six rungs happen to
        share a spelling with a projected domain value, so `"modeled"`
        resolved to UNSET and rendered CLEAN — a model's word presented as
        an observation, the exact failure this module prevents."""
        for member in pv.Provenance:
            assert pv.project(member.label) is member, member.label

    def test_every_marked_rung_renders_a_visible_mark(self):
        """The legend must not promise a mark the transcript cannot show."""
        for label, glyph, _meaning in pv.legend():
            assert glyph in pv.annotate("x", label), label

    def test_an_unrecognised_vocabulary_is_UNSET_not_guessed(self):
        """A vocabulary this module has not been taught is not evidence of
        anything."""
        assert pv.project("vibes") is None
        assert pv.project("") is None

    def test_of_reads_provenance_off_a_carrier(self):
        class _Advisory:
            blast_provenance = "synthetic_cap"
        assert pv.of(_Advisory()) is pv.Provenance.SYNTHETIC
        assert pv.of({"blast_provenance": "measured"}) is pv.Provenance.OBSERVED
        assert pv.of(object()) is None


class TestOneChokepointOneMark:
    def test_op_line_annotates_ABOVE_every_consumer(self):
        """The mirror, the op buffer replayed by `/expand`, the swarm digest
        and the local console must not disagree about how a line was known.
        Asserted on STATEMENT ORDER: the annotate call must precede the
        first consumer."""
        src = pathlib.Path(
            "backend/core/ouroboros/battle_test/serpent_flow.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "_op_line")
        body = [s for s in fn.body
                if not (isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant))]
        dumped = [ast.dump(s) for s in body]
        annotate_at = next(i for i, d in enumerate(dumped)
                           if "annotate" in d)
        consumer_at = next(i for i, d in enumerate(dumped)
                           if "_record_swarm_event" in d
                           or "_mirror_markup" in d)
        assert annotate_at < consumer_at

    def test_a_mark_is_never_applied_twice(self):
        """A seam reached twice — local console AND cockpit mirror — must
        not double-stamp."""
        once = pv.annotate("line", pv.Provenance.MODELED)
        assert pv.annotate(once, pv.Provenance.MODELED) == once
        assert once.count("‹model›") == 1

    def test_no_call_site_passes_provenance_into_op_line(self):
        """Ambient by design. A per-callsite argument is `/narrate`'s
        hardcoded producer list in a new costume: right on the day it is
        written, wrong the moment a producer is added, undetectably."""
        src = pathlib.Path(
            "backend/core/ouroboros/battle_test/serpent_flow.py").read_text()
        calls = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "_op_line"]
        assert calls, "no _op_line call sites found — test is vacuous"
        offenders = [c for c in calls
                     if any(k.arg == "provenance" for k in c.keywords)]
        assert not offenders


class TestTheWiredProducers:
    def test_narrative_frames_are_marked_as_the_MODEL_speaking(self):
        src = pathlib.Path(
            "backend/core/ouroboros/battle_test/narrative_renderer.py"
        ).read_text()
        # The surfacing seam is `render_to_printer`; `render_to_console`
        # is the console-shaped wrapper over it (the default transport
        # renders through the printer seam with its own sink).
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "render_to_printer")
        assert "claiming" in {ast.unparse(n.func) for n in ast.walk(fn)
                              if isinstance(n, ast.Call)}
        wrapper = next(n for n in ast.walk(ast.parse(src))
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "render_to_console")
        assert "render_to_printer" in {ast.unparse(n.func) for n in ast.walk(wrapper)
                                       if isinstance(n, ast.Call)}

    def test_the_synthesized_preamble_declares_itself(self):
        """The demonstration case: a template the code filled in, currently
        indistinguishable from a preamble the model wrote."""
        src = pathlib.Path(
            "backend/core/ouroboros/battle_test/serpent_flow.py").read_text()
        assert 'preamble_provenance = "synthetic"' in src

    def test_a_synthetic_preamble_renders_differently_than_a_model_one(self):
        model_line = pv.annotate("🗣 I'll read the config first",
                                 pv.Provenance.MODELED)
        template_line = pv.annotate("🗣 I'll read the config first",
                                    "synthetic")
        assert model_line != template_line


class TestItNeverRaises:
    @pytest.mark.parametrize("bad", [None, 0, object(), b"x", [], {}, 3.7])
    def test_project_survives_anything(self, bad):
        pv.project(bad)

    @pytest.mark.parametrize("bad", [None, 0, object(), b"x", []])
    def test_annotate_survives_anything(self, bad):
        assert isinstance(pv.annotate(bad), str)  # type: ignore[arg-type]

    def test_claiming_survives_a_broken_argument(self):
        with pv.claiming(object()) as resolved:
            assert resolved is None

    def test_a_broken_palette_still_marks(self, monkeypatch):
        """Losing the colour is cosmetic; losing the distinction is not."""
        import backend.core.ouroboros.ui.semantic_tokens as st
        monkeypatch.setattr(st, "sem", lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("palette gone")))
        out = pv.annotate("x", pv.Provenance.UNKNOWN)
        assert "unverified" in out

    def test_legend_lists_only_MARKED_rungs(self):
        labels = {row[0] for row in pv.legend()}
        assert "observed" not in labels and "derived" not in labels
        assert {"stated", "modeled", "synthetic", "unknown"} == labels
