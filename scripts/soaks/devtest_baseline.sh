#!/usr/bin/env bash
# BASELINE DEVELOPMENT TEST -- can O+V land a sanctioned change, untrained?
#
# This is NOT a corpus-farming soak. Farming soaks run in quarantine with
# promotion OFF so the operator tree is never mutated and the 41 tasks stay
# re-farmable. This run does the opposite: it lets a verified repair LAND,
# so we can measure whether the sanctioned execution chain actually closes
# end to end -- GENERATE -> VALIDATE -> GATE -> APPLY -> VERIFY -> commit
# -> promotion/PR.
#
# WHY BEFORE TRAINING. O+V has produced 0 autonomous commits to date; its
# historical wall has been GOVERNANCE and APPLY gates, not candidate
# quality. Everything the sibling/noop/parse_error arc fixed improves
# corpus DIVERSITY for training -- none of it makes a candidate more
# CORRECT. If the chain does not close untrained, a fine-tuned model will
# not close it either, and the GPU-hour is better spent after the blocker
# is known. This run is the empirical "before" the "after" is measured
# against, and it costs $0.00.
#
# THE SANCTIONED GOAL (.jarvis/roadmap.yaml, HMAC-signed, one goal):
#   docs-skip-tools-gate-drift -- rewrite ONLY the docstring of
#   _should_use_lean_prompt in backend/core/ouroboros/governance/providers.py
#   so it names should_skip_venom_for_route() instead of quoting the
#   pre-extraction inline route-tuple test. Comment-only, single file, zero
#   runtime effect, verifiable by grep.
#
#   Deliberately the smallest real change that STILL trips the
#   self-modification gate -- a file outside governance/ would not exercise
#   the mechanism at all. Verified still undone at launch time: the
#   docstring quotes `_skip_tools = _route in (...)` and does not mention
#   should_skip_venom_for_route.
#
#   The roadmap says "lines 3284-3358"; the function is now at 3317 because
#   providers.py moved under it. That drift is harmless -- resolution keys
#   on `target_symbol`, not line numbers (#70570 METHOD_DECLARED at
#   confidence 1.0, ahead of both inference passes).
#
# ISOLATION. Run ONLY from a dedicated worktree (see devtest_prepare.sh).
# Promotion lands commits on that worktree's branch; the main checkout is
# never touched. The prepare script also copies the three GITIGNORED
# artifacts a bare worktree cannot have and without which the chain is
# silently inert:
#     .env                     (HMAC secret + every chain flag)
#     .jarvis/roadmap.yaml     (the signature that authorizes the goal)
#     .superpowers/sdd/progress.md  (the work DECLARATION)
# Declaration and authorization are different artifacts; a goal present in
# only one of them does nothing, silently.
#
# WHAT TO MEASURE (devtest_report.sh reads these off the session log):
#   reached GENERATE          provenance verified, not caged at classify
#   terminal_reason           the honest stopping point
#   APPLY / VERIFY            did a byte land, did the tests still pass
#   commit / PR               did it produce an operator-visible artifact
#   governance rejections     which gate, how many, on what
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

_require() {
  [ -e "$1" ] || { echo "REFUSING: missing $1 -- run devtest_prepare.sh" >&2; exit 2; }
}
_require .env
_require .jarvis/roadmap.yaml
_require .superpowers/sdd/progress.md

# Refuse to run in the main checkout: this run MUTATES the tree.
if [ "$(git rev-parse --show-toplevel)" = "/mnt/c/Users/Jarvis/Desktop/TrinityAi/jarvis" ]; then
  echo "REFUSING: this is the main checkout. Run from the devtest worktree." >&2
  exit 2
fi

# --- the chain (all already armed in .env; restated so the run is legible) --
export JARVIS_ROADMAP_READER_ENABLED=true
export JARVIS_DELEGATED_PROVENANCE_ENABLED=true
export JARVIS_WORK_ORDER_SENSOR_ENABLED=true
export JARVIS_SWARM_ROUTING_ENABLED=true
export JARVIS_INTAKE_PRIORITY_SCHEDULER_ENABLED=true

# --- what makes this run different from every farming soak ----------------
export JARVIS_WORKSPACE_PROMOTION_ENABLED=true
export JARVIS_AUTO_COMMIT_ENABLED=true
export JARVIS_ORANGE_PR_ENABLED=false   # land locally; a PR needs gh + a push
                                        # remote and would measure THAT, not
                                        # whether the pipeline closes.

# --- exactly one work item, and no ambient flood --------------------------
export JARVIS_TESTWATCHER_BOOT_HYDRATION_ENABLED=false
export JARVIS_WORK_ORDER_INTERVAL_S=86400
export JARVIS_WORK_ORDER_RECENT_N=1
export JARVIS_WORK_ORDER_MAX_ITEMS=1
export JARVIS_WORK_ORDER_DEFAULT_URGENCY=high
export JARVIS_ALLOW_ROADMAP_REVISIT=true
export JARVIS_DOC_STALENESS_ENABLED=false
export JARVIS_RUNTIME_HEALTH_SENSOR_ENABLED=false
export JARVIS_GITHUB_ISSUE_SENSOR_ENABLED=false
export JARVIS_OPPORTUNITY_MINER_SENSOR_ENABLED=false
export JARVIS_INTENT_TEST_INTERVAL_S=86400
export JARVIS_TODO_SCAN_INTERVAL_S=86400
export JARVIS_EXPLORATION_INTERVAL_S=86400
export JARVIS_INTAKE_BACKLOG_SCAN_INTERVAL_S=86400

# --- generation: same local lane the farming soaks used -------------------
export JARVIS_TRAJECTORY_RECORDER_ENABLED=true
export JARVIS_SIBLING_ENTROPY_ENABLED=true
export JARVIS_LOCAL_SIBLING_CANDIDATES=3
export JARVIS_SIBLING_MAX_RESAMPLE=1
export JARVIS_VALIDATION_RESERVE_ENABLED=true
# 600, not 180: VALIDATE scoped the 2026-09-05 baseline to test_providers.py,
# 65 tests that take ~57 s standalone and blew 180 s under the sandboxed
# candidate tree (pytest-timeout thread dump, JSON report unparseable). The
# timeout is a guard against a HUNG suite, not a budget for a slow one.
# WHICH MODEL THE CONTROL MEASURES, named here rather than inherited.
# This script used to take whatever `.env` pinned, which was fine while the
# pin WAS the base model. The moment the operator repoints that pin at a
# fine-tuned tag so a bare `ov` picks it up, an unpinned control would
# silently start measuring the adapter against itself -- a baseline that
# is its own candidate, and nothing in the output would say so.
#
# `:-` preserves an existing export, so devtest_candidate.sh (which exports
# the candidate tag before invoking this script) still overrides it. Named
# default, overridable, and never ambient.
export JARVIS_LOCAL_MODEL_NAME="${JARVIS_LOCAL_MODEL_NAME:-${OV_BASELINE_MODEL:-qwen3-coder:30b}}"
echo "[devtest] model under test: $JARVIS_LOCAL_MODEL_NAME"

export JARVIS_TEST_TIMEOUT_S=600
# L2 on the local lane cannot re-emit providers.py (521 KB): the 2026-09-05
# baseline got 30-32 KB back, twice, and died full_content_too_short. This
# routes a big-file repair through the same symbol-scoped swarm GENERATE
# uses -- regenerate the failing symbols, stitch them into the original --
# and declines to the single-shot path for everything else. Default OFF in
# code (unproven); ON here because this run is what proves it.
export JARVIS_L2_SYMBOL_SCOPED_ENABLED=true
export JARVIS_THROUGHPUT_GOVERNOR_ENABLED=false
export JARVIS_BG_POOL_SIZE=2

# Shorter than a farming soak: one goal either closes or it does not.
exec /home/jarvis_svc/.venvs/ov/bin/python scripts/ouroboros_battle_test.py \
  --headless --cost-cap 0.50 --idle-timeout 900 --max-wall-seconds 3600 -v
