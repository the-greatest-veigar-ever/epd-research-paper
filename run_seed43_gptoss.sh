#!/usr/bin/env bash
#
# Launch the gpt-oss:20b SEED-43 solo run + its watcher. Run this on the pod.
#
# The run goes in tmux session 'gptoss_seed43'. When that session ends (run
# finished, crashed, or hit the watcher's wall-clock ceiling) the watcher
# runs pre-shutdown checks, commits + pushes the results, then stops the pod.
#
#   ./run_seed43_gptoss.sh          # launch
#   ./run_seed43_gptoss.sh --dry-run # print the plan, launch nothing
#
# Regime: gpt-oss:20b runs SOLO and keeps its seed-42 defaults
# (EPD_CALL_RETRIES=1, EPD_REASONING_TIMEOUT_MULT=2.0, num_predict=4096) --
# so NO EPD_* overrides here, unlike the 4-SLM seed-43 command.

set -euo pipefail

REPO="/workspace/epd-research-paper"
SESSION="gptoss_seed43"
BRANCH="runpod-results-slm"
MODEL_KEY="gpt_20b_oss"
SWEEP_LOG="$REPO/report-output/ghost_agents/run_logs/sweep_seed43_gptoss.log"
CKPT="$REPO/report-output/ghost_agents/benchmark_results/${MODEL_KEY}/checkpoint_seed43.json"
WATCHER="$REPO/watch_seed43_gptoss_then_push_and_stop.sh"

DRY=no
[ "${1:-}" = "--dry-run" ] && DRY=yes

cd "$REPO"

# --- preflight -----------------------------------------------------------
cur="$(git rev-parse --abbrev-ref HEAD)"
[ "$cur" = "$BRANCH" ] || { echo "FAIL: on branch '$cur', expected '$BRANCH'"; exit 1; }

if [ -e "$CKPT" ]; then
    echo "FAIL: seed-43 checkpoint already exists:"
    echo "  $CKPT"
    echo "This script is for a FRESH run. To resume, launch benchmark_evaluator by hand."
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "FAIL: tmux session '$SESSION' already exists. Kill it first: tmux kill-session -t $SESSION"
    exit 1
fi

[ -x "$WATCHER" ] || { echo "FAIL: watcher not executable: $WATCHER"; exit 1; }

[ -n "${RUNPOD_POD_ID:-}" ] || echo "WARN: RUNPOD_POD_ID unset -- the watcher will push but will NOT be able to stop the pod."

command -v ollama >/dev/null || { echo "FAIL: ollama not on PATH"; exit 1; }
[ -d /workspace/.ollama/models ] || { echo "FAIL: /workspace/.ollama/models missing"; exit 1; }

python3 -c "import pynvml, requests, numpy, psutil" || { echo "FAIL: python deps missing (pip install -r requirements.txt nvidia-ml-py)"; exit 1; }

# Clean orphan experiment-port ollama servers; keep a diagnostic one on :11434.
for p in $(pgrep -x ollama || true); do
    host="$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^OLLAMA_HOST=' | cut -d= -f2- || true)"
    if [ "$host" != "127.0.0.1:11434" ]; then
        echo "killing orphan ollama pid $p (OLLAMA_HOST='${host:-?}')"
        kill "$p" 2>/dev/null || true
    fi
done
sleep 2

RUN_CMD="cd $REPO && export OLLAMA_MODELS=/workspace/.ollama/models && \
python3 -u run_concurrent_experiment.py \
  --models ${MODEL_KEY} --seeds 43 --pod-hourly-usd 1.39 \
  2>&1 | tee $SWEEP_LOG; \
echo \"EXIT_CODE=\${PIPESTATUS[0]}\" | tee -a $SWEEP_LOG"

echo "======================================================================"
echo " gpt-oss:20b seed-43 solo run"
echo "   tmux session : $SESSION"
echo "   run command  : python3 -u run_concurrent_experiment.py --models ${MODEL_KEY} --seeds 43 --pod-hourly-usd 1.39"
echo "   sweep log    : $SWEEP_LOG"
echo "   watcher      : $WATCHER  -> /workspace/watch_seed43_gptoss.log"
echo "   on session end: checks -> commit+push ($BRANCH) -> runpodctl stop pod ${RUNPOD_POD_ID:-<unset>}"
echo "======================================================================"

if [ "$DRY" = yes ]; then
    echo "(dry-run) nothing launched."
    exit 0
fi

mkdir -p "$(dirname "$SWEEP_LOG")"

tmux new-session -d -s "$SESSION" "$RUN_CMD"
echo "launched run in tmux session '$SESSION'"

nohup bash "$WATCHER" >/dev/null 2>&1 &
echo "launched watcher (pid $!)"

echo
echo "monitor:"
echo "  tmux attach -t $SESSION"
echo "  tail -f $SWEEP_LOG"
echo "  tail -f $REPO/report-output/ghost_agents/run_logs/${MODEL_KEY}.log"
echo "  tail -f /workspace/watch_seed43_gptoss.log"
