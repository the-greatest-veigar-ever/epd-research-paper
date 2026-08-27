#!/usr/bin/env bash
#
# Watch the gpt-oss:20b seed-42 run, then commit + push its results and stop
# the pod. Started alongside the run; detached, so it survives this shell.
#
# Ordering is deliberate: PUSH FIRST, STOP LAST. /workspace is a network volume
# that survives a pod stop, so nothing is lost even if the push fails -- but a
# successful push means the results are off the pod entirely before billing
# ends. The pod is stopped even if the run failed or the push failed: a broken
# run should not keep burning GPU hours, and the marker written below records
# exactly what happened for whoever restarts the pod.
#
# Stop, not terminate: the pod and /workspace are preserved, only the GPU
# allocation ends. Restart it manually when ready.

set -uo pipefail

SESSION="gptoss_seed42"
REPO="/workspace/epd-research-paper"
RESULTS="$REPO/report-output/ghost_agents/benchmark_results/gpt_20b_oss"
LOG="/workspace/watch_gptoss.log"
BRANCH="runpod-results-slm"

exec >>"$LOG" 2>&1
echo "[watcher] started $(date -u) -- waiting for tmux session '$SESSION'"

# Give the run a moment to create the session before deciding it never started.
for _ in $(seq 1 30); do
    tmux has-session -t "$SESSION" 2>/dev/null && break
    sleep 2
done

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watcher] session never appeared -- doing nothing, NOT stopping the pod"
    exit 1
fi

# Wall-clock ceiling. The run is estimated at ~3.5h (range 3-5h), but a hung
# call chain could otherwise burn pod hours indefinitely: gpt-oss is a
# reasoning model, so its client timeout is 600s, and 400 calls at that worst
# case is over 60 hours. 8h bounds the damage at roughly $11 while leaving
# generous headroom over the estimate. Hitting it kills the run and still
# pushes whatever was checkpointed.
MAX_HOURS="${MAX_HOURS:-8}"
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
TIMED_OUT=no

while tmux has-session -t "$SESSION" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        TIMED_OUT=yes
        echo "[watcher] EXCEEDED ${MAX_HOURS}h ceiling -- killing the run"
        tmux kill-session -t "$SESSION" 2>/dev/null
        sleep 10
        break
    fi
    sleep 30
done
FINISHED="$(date -u)"
echo "[watcher] session ended $FINISHED (timed_out=$TIMED_OUT) -- gathering status"

cd "$REPO" || exit 1

# ---- status straight out of the checkpoint, not out of the log ----
STATUS_BODY="$(python3 - <<'PY' 2>&1
import json, os, glob
from collections import Counter
p = "report-output/ghost_agents/benchmark_results/gpt_20b_oss/checkpoint_seed42.json"
if not os.path.exists(p):
    print("NO CHECKPOINT WRITTEN -- the run produced nothing.")
    raise SystemExit
d = json.load(open(p))
cells = calls = 0
st = Counter()
thickness = Counter()
for bench, cell in d.get("benchmark_results", {}).items():
    for name, ap in cell.get("approaches", {}).items():
        tr = ap.get("test_results", [])
        cells += 1
        calls += len(tr)
        ok = 0
        for r in tr:
            st[r.get("call_status")] += 1
            if r.get("call_status") == "success":
                ok += 1
        thickness[ok] += 1
print(f"Benchmarks reached : {len(d.get('benchmark_results', {}))} / 10")
print(f"Cells recorded     : {cells} / 80")
print(f"Calls recorded     : {calls} / 400")
print()
print("Call status:")
for k, v in st.most_common():
    print(f"  {k:15s} {v:4d}")
print()
print("Cells by scoreable calls (n=5 is a full cell):")
for k in sorted(thickness, reverse=True):
    print(f"  n={k}: {thickness[k]:3d} cells")
cfg = {k: v for k, v in d.get("config", {}).items()
       if k not in ("approaches", "benchmarks")}
print()
print("Generation config recorded for this run:")
for k, v in cfg.items():
    print(f"  {k:24s} {v}")
PY
)"

MARKER="/workspace/GPTOSS_RUN_COMPLETE_$(date -u +%Y%m%d_%H%M%S).md"
{
    echo "# gpt-oss:20b seed-42 run finished"
    echo
    echo "Written (UTC): $FINISHED"
    echo
    echo "Solo run of gpt_20b_oss only -- the 4 SLMs' seed-42 data was frozen"
    echo "on 2026-08-27 and was NOT touched by this run."
    echo
    if [ "$TIMED_OUT" = yes ]; then
        echo "**KILLED BY THE ${MAX_HOURS}h WALL-CLOCK CEILING** -- the run did not finish on"
        echo "its own. Everything below is partial. Re-running resumes from the checkpoint."
        echo
    fi
    echo '```'
    echo "$STATUS_BODY"
    echo '```'
} > "$MARKER"
echo "[watcher] marker: $MARKER"

# ---- commit + push BEFORE billing ends ----
PUSH_RESULT="not attempted"
git add report-output/ghost_agents/benchmark_results/gpt_20b_oss/ \
        report-output/ghost_agents/benchmark_results/*gpt_20b_oss*.json \
        report-output/ghost_agents/run_logs/ 2>/dev/null
git add -f report-output/ghost_agents/run_logs/resource_timeseries_gpt_20b_oss.csv 2>/dev/null

if git diff --cached --quiet; then
    echo "[watcher] nothing staged -- skipping commit"
    PUSH_RESULT="nothing to commit"
else
    git commit -q -F - <<EOF
results(seed42): gpt-oss:20b solo run

Run finished $FINISHED. gpt_20b_oss only; the 4 SLMs' frozen seed-42 data
was not touched.

$STATUS_BODY

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
    echo "[watcher] committed $(git rev-parse --short HEAD)"
    if git push origin "$BRANCH"; then
        PUSH_RESULT="pushed $(git rev-parse --short HEAD)"
        echo "[watcher] push OK"
    else
        PUSH_RESULT="PUSH FAILED -- commit is local only, on /workspace which survives the stop"
        echo "[watcher] PUSH FAILED"
    fi
fi
printf '\n## Push\n\n%s\n' "$PUSH_RESULT" >> "$MARKER"

# ---- stop the pod last ----
echo "[watcher] stopping pod ${RUNPOD_POD_ID:-<unset>}"
printf '\n## Pod\n\nStopping pod %s (stop, not terminate -- /workspace preserved).\n' \
    "${RUNPOD_POD_ID:-<unset>}" >> "$MARKER"

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[watcher] RUNPOD_POD_ID unset -- cannot stop, leaving pod running"
    printf 'RUNPOD_POD_ID was unset; pod left RUNNING. Stop it manually.\n' >> "$MARKER"
    exit 1
fi

runpodctl stop pod "$RUNPOD_POD_ID"
echo "[watcher] runpodctl exit: $?"
