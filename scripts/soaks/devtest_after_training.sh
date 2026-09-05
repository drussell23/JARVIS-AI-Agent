#!/usr/bin/env bash
# Run the baseline development test AFTER a GRPO run has released the card,
# against the UNMODIFIED base model, on a worktree cut from current main.
#
# WHY A SEPARATE STAGE SCRIPT. The baseline needs ollama to load the 30B Q4
# (18.6 GB). A training child holds ~28 GiB of the same 32 GiB card, so a
# baseline launched during training does not fail loudly -- ollama spills
# to host RAM, the lane crawls, VALIDATE times out, and the run reports a
# model problem that is a scheduling problem. This script refuses to start
# until the card is actually free, and it verifies the tag it is about to
# measure is being SERVED, not merely present in the manifest.
#
# WHY A FRESH WORKTREE. The previous devtest worktree was cut before the
# harness fixes (stale diff-schema assertion, frozen-deadline fixture,
# VALIDATE timeout). A baseline on the old tree would re-measure the
# harness walls, not the model. devtest_prepare.sh refuses to reuse a
# worktree, so the old one is archived (its session logs are the evidence
# from the earlier runs) and removed first.
#
# Usage:  devtest_after_training.sh [wait]
#   wait   block until no run_grpo_training process exists and the card is
#          under $OV_GPU_FREE_MIB used (default 2048), polling every 60 s.
set -uo pipefail

MAIN="${OV_MAIN_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/jarvis}"
WT="${OV_DEVTEST_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/jarvis-devtest}"
BASE_TAG="${OV_BASE_MODEL:-qwen3-coder:30b}"
FREE_MIB="${OV_GPU_FREE_MIB:-2048}"
# Everything that may hold the card after the trainer itself exits: the
# launcher script, and the GGUF bridge it runs on a clean exit (ollama
# create + a serving probe both LOAD a model).
HOLDERS="${OV_GPU_HOLDER_PATTERN:-run_grpo_training.py|train[0-9]*.sh|lora_to_ollama.sh}"
ARCHIVE="${OV_DEVTEST_ARCHIVE:-$HOME/ov-archive}"
OLLAMA_API="${JARVIS_LOCAL_MODEL_BASE_URL:-http://127.0.0.1:11434}"

die() { echo "REFUSING: $*" >&2; exit 2; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1; }

# --- 1. the card must be free ---------------------------------------------
if [ "${1:-}" = "wait" ]; then
  while pgrep -f "$HOLDERS" >/dev/null || [ "$(gpu_used_mib)" -gt "$FREE_MIB" ]; do
    log "waiting: holders=$(pgrep -f "$HOLDERS" | wc -l) proc, gpu=$(gpu_used_mib) MiB used"
    sleep 60
  done
fi
pgrep -f "$HOLDERS" >/dev/null && die "a training run or its bridge is still alive; pass 'wait'"
USED=$(gpu_used_mib); [ -n "$USED" ] || die "nvidia-smi gave no reading"
[ "$USED" -le "$FREE_MIB" ] || die "card has $USED MiB in use (> $FREE_MIB); something still holds it"

# --- 2. the base tag must be SERVED, not merely listed ---------------------
curl -s -m 5 "$OLLAMA_API/api/tags" | grep -q "\"name\":\"$BASE_TAG\"" \
  || die "ollama at $OLLAMA_API does not list $BASE_TAG"
PROBE=$(curl -s -m 120 "$OLLAMA_API/api/generate" -d "{\"model\":\"$BASE_TAG\",\"prompt\":\"Reply with the single word: ready\",\"stream\":false}" | grep -o '"response":"[^"]*"' | head -1)
[ -n "$PROBE" ] || die "$BASE_TAG did not answer a probe; the lane is not serving"
log "base model answers: $PROBE"

# --- 3. a worktree cut from CURRENT main ------------------------------------
cd "$MAIN" || die "no main tree at $MAIN"
if [ -e "$WT" ]; then
  STAMP=$(date +%Y%m%d-%H%M%S)
  mkdir -p "$ARCHIVE/devtest-$STAMP"
  if [ -d "$WT/.ouroboros/sessions" ]; then
    cp -r "$WT/.ouroboros/sessions" "$ARCHIVE/devtest-$STAMP/" && log "archived old sessions -> $ARCHIVE/devtest-$STAMP"
  fi
  OLD_BRANCH=$(git -C "$WT" branch --show-current 2>/dev/null || true)
  git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  git worktree prune
  [ -n "$OLD_BRANCH" ] && git branch -D "$OLD_BRANCH" >/dev/null 2>&1 || true
  git branch -D devtest/baseline-untrained >/dev/null 2>&1 || true
  log "removed old worktree ($OLD_BRANCH)"
fi
bash scripts/soaks/devtest_prepare.sh || die "devtest_prepare.sh failed"
log "worktree ready at $WT from main $(git rev-parse --short main)"

# --- 4. the baseline itself, then the report --------------------------------
cd "$WT" || die "worktree missing after prepare"
log "=== BASELINE on $BASE_TAG (untrained) starts ==="
bash scripts/soaks/devtest_baseline.sh
RC=$?
log "=== baseline exit=$RC ==="
SESSION=$(ls -t .ouroboros/sessions 2>/dev/null | head -1)
[ -n "$SESSION" ] && bash scripts/soaks/devtest_report.sh ".ouroboros/sessions/$SESSION/debug.log" 2>/dev/null

# --- 5. record the control the promotion gate reads ------------------------
# Without this the gate has nothing to compare against and refuses every
# adapter with "cannot answer" -- correct, but it makes the baseline run
# pointless. The scoring lives in reactor because the gate does; this
# script supplies only the session and the model it was measured on, so
# there is one definition of what "better" means and it is not here.
REACTOR="${OV_REACTOR_TREE:-/mnt/c/Users/Jarvis/Desktop/TrinityAi/reactor}"
RPY="${OV_REACTOR_PYTHON:-$HOME/.venvs/reactor-train/bin/python}"
if [ -n "$SESSION" ] && [ -x "$RPY" ] && [ -d "$REACTOR" ]; then
  HARNESS="devtest@$(git -C "$MAIN" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  log "recording baseline from $SESSION (model=$BASE_TAG harness=$HARNESS)"
  if ( cd "$REACTOR" && "$RPY" -m reactor_core.deployment.devtest_baseline \
         "$WT/.ouroboros/sessions/$SESSION" \
         --base-model "$BASE_TAG" --harness "$HARNESS" ); then
    log "baseline recorded — the promotion gate can now answer"
  else
    log "baseline NOT recorded (rc=$?) — the gate will refuse every adapter"
  fi
else
  log "baseline NOT recorded: session=${SESSION:-none} python=$RPY reactor=$REACTOR"
fi
exit $RC
