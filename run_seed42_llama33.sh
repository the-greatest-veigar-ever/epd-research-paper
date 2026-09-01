#!/usr/bin/env bash
#
# Launch llama3.3:70b SEED-42 static baseline + watcher (push + stop pod).
#
#   ./run_seed42_llama33.sh
#   ./run_seed42_llama33.sh --dry-run

set -euo pipefail

REPO="/workspace/epd-research-paper"
SESSION="llama33_seed42"
BRANCH="runpod-results-slm"
MODEL_KEY="llama33_70b"
SWEEP_LOG="$REPO/report-output/ghost_agents/run_logs/llama33_70b_seed42.log"
WATCHER="$REPO/watch_seed42_llama33_then_push_and_stop.sh"

DRY=no
[ "${1:-}" = "--dry-run" ] && DRY=yes

cd "$REPO"

cur="$(git rev-parse --abbrev-ref HEAD)"
[ "$cur" = "$BRANCH" ] || { echo "FAIL: on branch '$cur', expected '$BRANCH'"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "FAIL: tmux session '$SESSION' already exists"
    exit 1
fi

[ -x "$WATCHER" ] || chmod +x "$WATCHER"
[ -n "${RUNPOD_POD_ID:-}" ] || echo "WARN: RUNPOD_POD_ID unset -- watcher cannot stop pod"

command -v ollama >/dev/null || { echo "FAIL: ollama not on PATH"; exit 1; }
[ -d /workspace/ollama_models ] || { echo "FAIL: /workspace/ollama_models missing"; exit 1; }

export OLLAMA_MODELS=/workspace/ollama_models
if ! ollama list 2>/dev/null | grep -q 'llama3.3:70b'; then
    echo "FAIL: llama3.3:70b not pulled (OLLAMA_MODELS=$OLLAMA_MODELS)"
    exit 1
fi

python3 -c "import pynvml, requests, numpy, psutil" \
    || { echo "FAIL: pip install -r requirements.txt"; exit 1; }

pgrep -x ollama >/dev/null || {
    tmux new-session -d -s ollama "export OLLAMA_MODELS=/workspace/ollama_models && ollama serve"
    sleep 3
}

RUN_CMD="cd $REPO && export OLLAMA_MODELS=/workspace/ollama_models && \
python3 -u -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
  --approaches llama33_70b_static \
  --seeds 42 \
  --max-per-benchmark 5 \
  --save-every 5 \
  --output report-output/ghost_agents/benchmark_results \
  2>&1 | tee $SWEEP_LOG; \
echo EXIT_CODE=\${PIPESTATUS[0]} | tee -a $SWEEP_LOG"

echo "======================================================================"
echo " llama3.3:70b seed-42 static baseline"
echo "   tmux session : $SESSION"
echo "   approach     : llama33_70b_static"
echo "   scope        : seed 42, 5 samples × 10 benchmarks (50 calls)"
echo "   sweep log    : $SWEEP_LOG"
echo "   watcher log  : /workspace/watch_seed42_llama33.log"
echo "   on finish    : summary -> commit+push ($BRANCH) -> stop pod ${RUNPOD_POD_ID:-<unset>}"
echo "======================================================================"

if [ "$DRY" = yes ]; then
    echo "(dry-run) nothing launched."
    exit 0
fi

mkdir -p "$(dirname "$SWEEP_LOG")"

# Remove partial smoke-test artifacts (2 benchmarks × 2 samples) so the
# full run starts clean.
rm -f "$REPO/report-output/ghost_agents/benchmark_results/${MODEL_KEY}/checkpoint_seed42.json" \
      "$REPO/report-output/ghost_agents/benchmark_results/${MODEL_KEY}/benchmark_"*"_seed42.json" \
      "$REPO/report-output/ghost_agents/benchmark_results/benchmark_"*"_seed42_${MODEL_KEY}"*.json \
      "$REPO/report-output/ghost_agents/benchmark_results/multi_seed_summary_"*"_${MODEL_KEY}"*.json 2>/dev/null || true

tmux new-session -d -s "$SESSION" "$RUN_CMD"
echo "launched run in tmux session '$SESSION'"

nohup bash "$WATCHER" >/dev/null 2>&1 &
echo "launched watcher (pid $!)"

echo
echo "monitor:"
echo "  tmux attach -t $SESSION"
echo "  tail -f $SWEEP_LOG"
echo "  tail -f /workspace/watch_seed42_llama33.log"
