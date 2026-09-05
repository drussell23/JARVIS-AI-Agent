#!/usr/bin/env bash
# Score a fine-tuned adapter on the SAME harness the control was measured on,
# then ask the promotion gate to decide.
#
# WHY A SEPARATE SCRIPT FROM THE BASELINE RUNNER. The two runs must differ in
# EXACTLY ONE THING: the model tag. Everything else -- the VALIDATE timeout,
# the symbol-scoped L2 path, swarm routing, the goal, the worktree, the
# scoring formula -- has to be identical, or the comparison measures the
# harness instead of the model. That is not a hypothetical: this session
# already watched a stale test and a 180 s timeout dominate a run's outcome
# completely. So this script does NOT re-declare the environment; it invokes
# `devtest_baseline.sh`, which owns every override, with one variable
# exported over it.
#
# `load_env_once` uses `override=False`, so an exported JARVIS_LOCAL_MODEL_NAME
# beats the .env pin. That is the whole mechanism, and it is the same one a
# human uses to drive the cockpit by hand.
#
# THE CANDIDATE SCORE IS NOT WRITTEN AS A BASELINE. The control is the thing
# being compared against; overwriting it with the candidate would destroy the
# comparison and leave a record claiming the adapter is its own control.
#
# Usage:  devtest_candidate.sh [wait] [tag]
#   wait   block until the GPU is free (same holder logic as the baseline)
#   tag    ollama tag to evaluate (default qwen3-coder-ov:30b)
set -uo pipefail

MAIN="${OV_MAIN_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/jarvis}"
WT="${OV_DEVTEST_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/jarvis-devtest}"
REACTOR="${OV_REACTOR_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/reactor}"
RPY="${OV_REACTOR_PYTHON:-$HOME/.venvs/reactor-train/bin/python}"
FREE_MIB="${OV_GPU_FREE_MIB:-2048}"
HOLDERS="${OV_GPU_HOLDER_PATTERN:-run_grpo_training.py|train[0-9]*.sh|lora_to_ollama.sh|ouroboros_battle_test.py}"
OLLAMA_API="${JARVIS_LOCAL_MODEL_BASE_URL:-http://127.0.0.1:11434}"

WAIT=""
TAG="${OV_CANDIDATE_TAG:-qwen3-coder-ov:30b}"
for arg in "$@"; do
  case "$arg" in
    wait) WAIT=1 ;;
    *) TAG="$arg" ;;
  esac
done

die() { echo "REFUSING: $*" >&2; exit 2; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1; }

# --- 1. the control must already exist -------------------------------------
# Running the candidate without one produces a number with nothing to
# compare it to, which is an hour of GPU spent to learn nothing.
BASELINE_PATH="${REACTOR_BASELINE_PATH:-$HOME/.jarvis/baselines/devtest_baseline.json}"
BASELINE_PATH=$(eval echo "$BASELINE_PATH")
[ -f "$BASELINE_PATH" ] \
  || die "no control at $BASELINE_PATH -- run devtest_after_training.sh first;
  a candidate score with nothing to compare against is an hour of GPU for nothing"
log "control on disk: $BASELINE_PATH"

# --- 2. the card, and the tag ----------------------------------------------
# What this guard is actually for: not contending with a TRAINING run, which
# holds ~28 GiB and would push this run's model into a host-RAM spill. It is
# NOT for waiting until the card is empty. ollama keeping a model resident on
# its keep-alive is the normal state after any soak -- and it is the state
# this run NEEDS, since the whole point is to talk to a served model. Gating
# on raw free VRAM waits for a condition that will not arrive until the
# keep-alive expires, which is how this script hung on its first run with
# `holders=0 proc, gpu=22297 MiB used`.
#
# So: wait for competing PROCESSES. Then prove the card works the only way
# that actually means anything -- the tag answers a probe (below). ollama
# swaps its own models; a resident model is not an obstacle to it.
if [ -n "$WAIT" ]; then
  while pgrep -f "$HOLDERS" >/dev/null; do
    log "waiting: holders=$(pgrep -f "$HOLDERS" | wc -l) proc, gpu=$(gpu_used_mib) MiB used"
    sleep 60
  done
fi
pgrep -f "$HOLDERS" >/dev/null && die "a training run or soak still holds the card; pass 'wait'"
USED=$(gpu_used_mib); [ -n "$USED" ] || die "nvidia-smi gave no reading"
log "card: $USED MiB in use (no competing process; ollama may hold a model)"

curl -s -m 5 "$OLLAMA_API/api/tags" | grep -q "\"name\":\"$TAG\"" \
  || die "ollama does not serve '$TAG' -- convert and register the adapter first"
PROBE=$(curl -s -m 180 "$OLLAMA_API/api/generate" \
  -d "{\"model\":\"$TAG\",\"prompt\":\"Reply with the single word: ready\",\"stream\":false}" \
  | grep -o '"response":"[^"]*"' | head -1)
[ -n "$PROBE" ] || die "'$TAG' did not answer a probe"
log "candidate answers: $PROBE"

# --- 3. a fresh worktree, same as the control got ---------------------------
cd "$MAIN" || die "no main tree at $MAIN"
if [ -e "$WT" ]; then
  STAMP=$(date +%Y%m%d-%H%M%S)
  ARCHIVE="${OV_DEVTEST_ARCHIVE:-$HOME/ov-archive}/candidate-$STAMP"
  mkdir -p "$ARCHIVE"
  [ -d "$WT/.ouroboros/sessions" ] && cp -r "$WT/.ouroboros/sessions" "$ARCHIVE/" \
    && log "archived control sessions -> $ARCHIVE"
  OLD_BRANCH=$(git -C "$WT" branch --show-current 2>/dev/null || true)
  git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  git worktree prune
  [ -n "$OLD_BRANCH" ] && git branch -D "$OLD_BRANCH" >/dev/null 2>&1 || true
  git branch -D devtest/baseline-untrained >/dev/null 2>&1 || true
fi
bash scripts/soaks/devtest_prepare.sh || die "devtest_prepare.sh failed"
log "worktree ready from main $(git rev-parse --short main)"

# --- 4. the SAME harness, one variable different ----------------------------
cd "$WT" || die "worktree missing after prepare"
export JARVIS_LOCAL_MODEL_NAME="$TAG"
log "=== CANDIDATE run on $TAG (harness identical to the control) ==="
bash scripts/soaks/devtest_baseline.sh
RC=$?
log "=== candidate exit=$RC ==="

SESSION=$(ls -t .ouroboros/sessions 2>/dev/null | head -1)
[ -n "$SESSION" ] && bash scripts/soaks/devtest_report.sh ".ouroboros/sessions/$SESSION/debug.log" 2>/dev/null

# --- 5. score it the same way, then let the gate decide ---------------------
# Same formula, same module. Scoring the candidate any other way would be
# comparing unlike things, which the gate's `metric` field exists to catch.
[ -n "$SESSION" ] || die "the candidate run produced no session"
[ -x "$RPY" ] || die "no reactor interpreter at $RPY"
cd "$REACTOR" || die "no reactor tree at $REACTOR"
exec "$RPY" -m reactor_core.deployment.evaluate_candidate \
  "$WT/.ouroboros/sessions/$SESSION" \
  --candidate-tag "$TAG" \
  --base-model "${OV_BASE_MODEL:-qwen3-coder:30b}"
