#!/usr/bin/env bash
#
# Watch the gpt-oss:20b SEED-43 solo run, then: run pre-shutdown checks,
# commit + push the results, and stop the pod. Detached; survives the shell
# that launched it.
#
# Ordering is deliberate: CHECK -> PUSH -> STOP.
#   * /workspace is a network volume that survives a pod stop, so nothing is
#     lost even if the push fails.
#   * A successful push means the results are off the pod before billing ends.
#   * The pod is stopped even if the run failed, timed out, or a check failed:
#     a broken run must not keep burning GPU hours. The marker written below
#     records exactly what happened for whoever restarts the pod.
#
# Stop, not terminate: the pod and /workspace are preserved, only the GPU
# allocation / compute billing ends. Restart it manually when ready.

set -uo pipefail

SESSION="gptoss_seed43"
REPO="/workspace/epd-research-paper"
SEED=43
MODEL_KEY="gpt_20b_oss"
CKPT="$REPO/report-output/ghost_agents/benchmark_results/${MODEL_KEY}/checkpoint_seed${SEED}.json"
LOG="/workspace/watch_seed43_gptoss.log"
BRANCH="runpod-results-slm"

# Expected generation regime -- MUST match gpt-oss:20b's seed-42 run
# (report-output/.../gpt_20b_oss/checkpoint_seed42.json -> "config"), or seed
# 43 is not the same experiment and the two cannot be averaged.
EXPECT_NUM_PREDICT=4096
EXPECT_NUM_CTX=8192
EXPECT_TEMPERATURE=0.0
EXPECT_WORD_BUDGET_RATIO=0.7
EXPECT_GENERATE_TIMEOUT_S=300
EXPECT_REASONING_TIMEOUT_MULT=2.0
EXPECT_CALL_RETRIES=1

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[watcher] started $(date -u) -- waiting for tmux session '$SESSION'"

# Wall-clock ceiling. Seed-42's gpt-oss solo run took ~5.4h; the estimate for
# seed 43 is ~5.5h. 9h leaves generous headroom and bounds a hung
# reasoning-model call chain (600s client timeout x 400 calls ~ 66h worst
# case) at roughly $12.50. Hitting it kills the run and still pushes whatever
# was checkpointed.
#
# Hardened: seed-42's first watcher run inherited a stray MAX_HOURS=0 from the
# environment and killed the run after ~30s ("0h wall-clock ceiling", no
# checkpoint written). Reject 0 / empty / non-numeric here.
MAX_HOURS="${MAX_HOURS:-9}"
case "$MAX_HOURS" in
    ''|*[!0-9]*|0) echo "[watcher] MAX_HOURS='$MAX_HOURS' invalid -- forcing 9"; MAX_HOURS=9 ;;
esac

# Give the run a moment to create the session before deciding it never started.
for _ in $(seq 1 30); do
    tmux has-session -t "$SESSION" 2>/dev/null && break
    sleep 2
done
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watcher] session never appeared -- doing nothing, NOT stopping the pod"
    exit 1
fi

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
echo "[watcher] session ended $FINISHED (timed_out=$TIMED_OUT) -- running pre-shutdown checks"

cd "$REPO" || exit 1

MARKER="/workspace/SEED43_GPTOSS_RUN_COMPLETE_$(date -u +%Y%m%d_%H%M%S).md"
CHECKS_OK=yes

# ---------------------------------------------------------------------------
# CHECK A (completeness) + CHECK B (config vs seed-42 regime) + status body,
# all straight from the checkpoint -- not from the log.
# ---------------------------------------------------------------------------
STATUS_BODY="$(
  CKPT="$CKPT" \
  EXPECT_NUM_PREDICT="$EXPECT_NUM_PREDICT" EXPECT_NUM_CTX="$EXPECT_NUM_CTX" \
  EXPECT_TEMPERATURE="$EXPECT_TEMPERATURE" EXPECT_WORD_BUDGET_RATIO="$EXPECT_WORD_BUDGET_RATIO" \
  EXPECT_GENERATE_TIMEOUT_S="$EXPECT_GENERATE_TIMEOUT_S" \
  EXPECT_REASONING_TIMEOUT_MULT="$EXPECT_REASONING_TIMEOUT_MULT" \
  EXPECT_CALL_RETRIES="$EXPECT_CALL_RETRIES" \
  python3 - <<'PY' 2>&1
import json, os
from collections import Counter

ckpt = os.environ["CKPT"]
if not os.path.exists(ckpt):
    print("CHECK A [FAIL]  no checkpoint written -- the run produced nothing.")
    print("CHECK B [FAIL]  no checkpoint -- cannot verify config.")
    print("__CHECKS_FAILED__")
    raise SystemExit

d = json.load(open(ckpt))
br = d.get("benchmark_results", {})
cells = calls = 0
st = Counter(); thickness = Counter()
for _bench, cell in br.items():
    for _name, ap in cell.get("approaches", {}).items():
        tr = ap.get("test_results", [])
        cells += 1
        calls += len(tr)
        ok = sum(1 for r in tr if r.get("call_status") == "success")
        for r in tr:
            st[r.get("call_status")] += 1
        thickness[ok] += 1

failed = False

# CHECK A -- completeness
a_ok = (len(br) == 10 and cells == 80 and calls == 400)
print(f"CHECK A [{'PASS' if a_ok else 'FAIL'}]  completeness: "
      f"{len(br)}/10 benchmarks, {cells}/80 cells, {calls}/400 calls")
if not a_ok:
    failed = True

# CHECK B -- generation regime matches gpt-oss:20b's seed-42 run
cfg = d.get("config", {})
expect = {
    "num_predict":            int(os.environ["EXPECT_NUM_PREDICT"]),
    "num_ctx":                int(os.environ["EXPECT_NUM_CTX"]),
    "temperature":            float(os.environ["EXPECT_TEMPERATURE"]),
    "word_budget_ratio":      float(os.environ["EXPECT_WORD_BUDGET_RATIO"]),
    "generate_timeout_s":     int(os.environ["EXPECT_GENERATE_TIMEOUT_S"]),
    "reasoning_timeout_mult": float(os.environ["EXPECT_REASONING_TIMEOUT_MULT"]),
    "call_retries":           int(os.environ["EXPECT_CALL_RETRIES"]),
}
mism = []
for k, want in expect.items():
    got = cfg.get(k)
    try:
        same = float(got) == float(want)
    except (TypeError, ValueError):
        same = got == want
    if not same:
        mism.append(f"{k}={got!r} (want {want!r})")
if cfg.get("seed") != 43:
    mism.append(f"seed={cfg.get('seed')!r} (want 43)")
b_ok = not mism
print(f"CHECK B [{'PASS' if b_ok else 'FAIL'}]  config vs seed-42 regime"
      + ("" if b_ok else ": " + "; ".join(mism)))
if not b_ok:
    failed = True

print()
print(f"Benchmarks reached : {len(br)} / 10")
print(f"Cells recorded     : {cells} / 80")
print(f"Calls recorded     : {calls} / 400")
print()
print("Call status:")
for k, v in st.most_common():
    print(f"  {str(k):15s} {v:4d}")
print()
print("Cells by scoreable calls (n=5 is a full cell):")
for k in sorted(thickness, reverse=True):
    print(f"  n={k}: {thickness[k]:3d} cells")
print()
print("Generation config recorded for this run:")
for k, v in cfg.items():
    if k not in ("approaches", "benchmarks"):
        print(f"  {k:24s} {v}")

if failed:
    print("__CHECKS_FAILED__")
PY
)"
if printf '%s\n' "$STATUS_BODY" | grep -q "__CHECKS_FAILED__"; then
    CHECKS_OK=no
fi
STATUS_BODY="$(printf '%s\n' "$STATUS_BODY" | grep -v '__CHECKS_FAILED__')"

# ---------------------------------------------------------------------------
# CHECK C -- seed-42 vs seed-43 config drift (the tool the ledger names).
# ---------------------------------------------------------------------------
echo "[watcher] CHECK C: analysis/compare_seed_configs.py"
CMP_OUT="$(python3 analysis/compare_seed_configs.py 2>&1)"; CMP_RC=$?
if [ "$CMP_RC" -eq 0 ]; then
    CHECK_C="CHECK C [PASS]  compare_seed_configs.py -- no drift (exit 0)"
else
    CHECK_C="CHECK C [FAIL]  compare_seed_configs.py exit ${CMP_RC}:
$(printf '%s\n' "$CMP_OUT" | sed 's/^/    /')"
    CHECKS_OK=no
fi
printf '%s\n' "$CHECK_C"

# ---------------------------------------------------------------------------
# CHECK D -- code integrity. Report-only: a test failure does not change the
# data already on disk, so it must not block the push or the pod stop.
# ---------------------------------------------------------------------------
echo "[watcher] CHECK D: python3 -m pytest tests/ -q"
PYTEST_OUT="$(python3 -m pytest tests/ -q 2>&1)"; PYTEST_RC=$?
PYTEST_TAIL="$(printf '%s\n' "$PYTEST_OUT" | tail -n 3)"
if [ "$PYTEST_RC" -eq 0 ]; then
    CHECK_D="CHECK D [PASS]  pytest: $(printf '%s' "$PYTEST_TAIL" | tail -n 1)"
else
    CHECK_D="CHECK D [WARN]  pytest exit ${PYTEST_RC} (does NOT affect collected data):
$(printf '%s\n' "$PYTEST_TAIL" | sed 's/^/    /')"
fi
printf '%s\n' "$CHECK_D"

# ---------------------------------------------------------------------------
# Write the marker.
# ---------------------------------------------------------------------------
{
    echo "# gpt-oss:20b SEED-43 solo run finished"
    echo
    echo "Written (UTC): $FINISHED"
    echo
    echo "Solo run of \`${MODEL_KEY}\` only, seed ${SEED}. The 4 SLMs and every"
    echo "seed-42 record were NOT touched by this run."
    echo
    if [ "$TIMED_OUT" = yes ]; then
        echo "**KILLED BY THE ${MAX_HOURS}h WALL-CLOCK CEILING** -- did not finish on its"
        echo "own. Everything below is partial; re-running resumes from the checkpoint."
        echo
    fi
    if [ "$CHECKS_OK" != yes ]; then
        echo "## WARNING -- PRE-SHUTDOWN CHECKS FAILED"
        echo
        echo "One or more checks below did not pass. The data was still pushed and the"
        echo "pod still stopped (so billing ends), but do not trust this run until the"
        echo "failing check is understood."
        echo
    fi
    echo "## Pre-shutdown checks"
    echo
    echo '```'
    printf '%s\n' "$STATUS_BODY" | sed -n '1,2p'   # CHECK A + CHECK B
    printf '%s\n' "$CHECK_C"
    printf '%s\n' "$CHECK_D"
    echo '```'
    echo
    echo "## Run status (from checkpoint)"
    echo
    echo '```'
    printf '%s\n' "$STATUS_BODY" | sed -n '3,$p'
    echo '```'
} > "$MARKER"
echo "[watcher] marker: $MARKER"

# ---------------------------------------------------------------------------
# Commit + push BEFORE billing ends. Seed-agnostic globs -- they pick up the
# seed-43 files the same way the seed-42 watcher picked up seed-42's.
# ---------------------------------------------------------------------------
PUSH_RESULT="not attempted"
git add "report-output/ghost_agents/benchmark_results/${MODEL_KEY}/" \
        "report-output/ghost_agents/benchmark_results/"*"${MODEL_KEY}"*.json \
        "report-output/ghost_agents/run_logs/run_manifest_"*.json 2>/dev/null
git add -f "report-output/ghost_agents/run_logs/resource_timeseries_${MODEL_KEY}.csv" 2>/dev/null

if git diff --cached --quiet; then
    echo "[watcher] nothing staged -- skipping commit"
    PUSH_RESULT="nothing to commit"
else
    git commit -q -F - <<EOF
results(seed43): gpt-oss:20b solo run

Run finished $FINISHED. ${MODEL_KEY} only, seed ${SEED}; the 4 SLMs and all
seed-42 data were not touched. Pre-shutdown checks: ${CHECKS_OK} (see
$(basename "$MARKER") on the volume).

$STATUS_BODY

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
    HEAD_SHORT="$(git rev-parse --short HEAD)"
    echo "[watcher] committed $HEAD_SHORT"
    if git push origin "$BRANCH"; then
        # CHECK E -- confirm the remote actually has this commit.
        git fetch -q origin "$BRANCH" 2>/dev/null
        if [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH" 2>/dev/null)" ]; then
            PUSH_RESULT="pushed $HEAD_SHORT (CHECK E [PASS] -- origin/$BRANCH matches local HEAD)"
            echo "[watcher] push OK, remote verified"
        else
            PUSH_RESULT="pushed $HEAD_SHORT but CHECK E [WARN] -- origin/$BRANCH does not match local HEAD after fetch"
            echo "[watcher] push reported OK but remote/local mismatch"
            CHECKS_OK=no
        fi
    else
        PUSH_RESULT="PUSH FAILED -- commit $HEAD_SHORT is local only, on /workspace which survives the stop"
        echo "[watcher] PUSH FAILED"
        CHECKS_OK=no
    fi
fi
printf '\n## Push\n\n%s\n' "$PUSH_RESULT" >> "$MARKER"

# ---------------------------------------------------------------------------
# Stop the pod last.
# ---------------------------------------------------------------------------
echo "[watcher] stopping pod ${RUNPOD_POD_ID:-<unset>}"
printf '\n## Pod\n\nStopping pod %s (stop, not terminate -- /workspace preserved).\nPre-shutdown checks overall: %s\n' \
    "${RUNPOD_POD_ID:-<unset>}" "$CHECKS_OK" >> "$MARKER"

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[watcher] RUNPOD_POD_ID unset -- cannot stop, leaving pod running"
    printf 'RUNPOD_POD_ID was unset; pod left RUNNING. Stop it manually.\n' >> "$MARKER"
    exit 1
fi

runpodctl stop pod "$RUNPOD_POD_ID"
RC=$?
echo "[watcher] runpodctl exit: $RC"
exit "$RC"
