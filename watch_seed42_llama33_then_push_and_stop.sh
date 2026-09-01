#!/usr/bin/env bash
#
# Watch the llama3.3:70b SEED-42 static baseline run, then: pre-shutdown
# checks, commit + push results, write a full summary marker, and stop the pod.
#
# Ordering: CHECK -> PUSH -> STOP.
# /workspace survives pod stop; push first so results leave the pod before
# billing ends. Pod stops even on failure/timeout so GPU hours don't burn.

set -uo pipefail

SESSION="llama33_seed42"
REPO="/workspace/epd-research-paper"
SEED=42
MODEL_KEY="llama33_70b"
APPROACH="llama33_70b_static"
CKPT="$REPO/report-output/ghost_agents/benchmark_results/${MODEL_KEY}/checkpoint_seed${SEED}.json"
LOG="/workspace/watch_seed42_llama33.log"
BRANCH="runpod-results-slm"
MAX_PER_BENCHMARK=5

EXPECT_NUM_PREDICT=1024
EXPECT_NUM_CTX=8192
EXPECT_TEMPERATURE=0.0
EXPECT_WORD_BUDGET_RATIO=0.7
EXPECT_GENERATE_TIMEOUT_S=300
EXPECT_REASONING_TIMEOUT_MULT=2.0
EXPECT_CALL_RETRIES=1

git config --global credential.helper 'store --file=/workspace/.git-credentials' 2>/dev/null || true

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[watcher] started $(date -u) -- waiting for tmux session '$SESSION'"

MAX_HOURS="${MAX_HOURS:-4}"
case "$MAX_HOURS" in
    ''|*[!0-9]*|0) echo "[watcher] MAX_HOURS='$MAX_HOURS' invalid -- forcing 4"; MAX_HOURS=4 ;;
esac

for _ in $(seq 1 30); do
    tmux has-session -t "$SESSION" 2>/dev/null && break
    sleep 2
done
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watcher] session never appeared -- NOT stopping the pod"
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
echo "[watcher] session ended $FINISHED (timed_out=$TIMED_OUT)"

cd "$REPO" || exit 1

MARKER="/workspace/LLAMA33_SEED42_RUN_COMPLETE_$(date -u +%Y%m%d_%H%M%S).md"
CHECKS_OK=yes

STATUS_BODY="$(
  CKPT="$CKPT" SEED="$SEED" MAX_PER_BENCHMARK="$MAX_PER_BENCHMARK" \
  EXPECT_NUM_PREDICT="$EXPECT_NUM_PREDICT" EXPECT_NUM_CTX="$EXPECT_NUM_CTX" \
  EXPECT_TEMPERATURE="$EXPECT_TEMPERATURE" EXPECT_WORD_BUDGET_RATIO="$EXPECT_WORD_BUDGET_RATIO" \
  EXPECT_GENERATE_TIMEOUT_S="$EXPECT_GENERATE_TIMEOUT_S" \
  EXPECT_REASONING_TIMEOUT_MULT="$EXPECT_REASONING_TIMEOUT_MULT" \
  EXPECT_CALL_RETRIES="$EXPECT_CALL_RETRIES" \
  python3 - <<'PY' 2>&1
import json, os
from collections import Counter

ckpt = os.environ["CKPT"]
seed = int(os.environ["SEED"])
max_pb = int(os.environ["MAX_PER_BENCHMARK"])
if not os.path.exists(ckpt):
    print("CHECK A [FAIL]  no checkpoint written -- the run produced nothing.")
    print("CHECK B [FAIL]  no checkpoint -- cannot verify config.")
    print("__CHECKS_FAILED__")
    raise SystemExit

d = json.load(open(ckpt))
br = d.get("benchmark_results", {})
cells = calls = 0
st = Counter()
bench_done = set()
for bench, cell in br.items():
    bench_done.add(bench)
    for _name, ap in cell.get("approaches", {}).items():
        tr = ap.get("test_results", [])
        cells += 1
        calls += len(tr)
        for r in tr:
            st[r.get("call_status")] += 1

expect_benches = 10
expect_cells = 10
expect_calls = expect_benches * max_pb
a_ok = (len(bench_done) == expect_benches and cells == expect_cells and calls == expect_calls)
print(f"CHECK A [{'PASS' if a_ok else 'FAIL'}]  completeness: "
      f"{len(bench_done)}/{expect_benches} benchmarks, {cells}/{expect_cells} cells, "
      f"{calls}/{expect_calls} calls")
failed = not a_ok

cfg = d.get("config", {})
expect = {
    "num_predict": int(os.environ["EXPECT_NUM_PREDICT"]),
    "num_ctx": int(os.environ["EXPECT_NUM_CTX"]),
    "temperature": float(os.environ["EXPECT_TEMPERATURE"]),
    "word_budget_ratio": float(os.environ["EXPECT_WORD_BUDGET_RATIO"]),
    "generate_timeout_s": int(os.environ["EXPECT_GENERATE_TIMEOUT_S"]),
    "reasoning_timeout_mult": float(os.environ["EXPECT_REASONING_TIMEOUT_MULT"]),
    "call_retries": int(os.environ["EXPECT_CALL_RETRIES"]),
    "max_per_benchmark": max_pb,
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
if cfg.get("seed") != seed:
    mism.append(f"seed={cfg.get('seed')!r} (want {seed})")
b_ok = not mism
print(f"CHECK B [{'PASS' if b_ok else 'FAIL'}]  config"
      + ("" if b_ok else ": " + "; ".join(mism)))
if not b_ok:
    failed = True

print()
print(f"Benchmarks reached : {len(bench_done)} / {expect_benches}")
print(f"Cells recorded     : {cells} / {expect_cells}")
print(f"Calls recorded     : {calls} / {expect_calls}")
print()
print("Call status:")
for k, v in st.most_common():
    print(f"  {str(k):15s} {v:4d}")
print()
print("Generation config recorded for this run:")
for k, v in cfg.items():
    if k not in ("approaches", "benchmarks"):
        print(f"  {k:24s} {v}")

# Per-benchmark ASR/TSR summary
print()
print("Per-benchmark metrics (llama33_70b_static):")
for bench in sorted(br.keys()):
    ap = br[bench]["approaches"].get("llama33_70b_static", {})
    m = ap.get("metrics", {})
    asr = m.get("asr")
    tsr = m.get("tsr")
    n = m.get("completed_tests", len(ap.get("test_results", [])))
    asr_s = f"{asr*100:.1f}%" if asr is not None else "n/a"
    tsr_s = f"{tsr*100:.1f}%" if tsr is not None else "n/a"
    print(f"  {bench:20s} ASR={asr_s:>6s}  TSR={tsr_s:>6s}  n={n}")

if failed:
    print("__CHECKS_FAILED__")
PY
)"
if printf '%s\n' "$STATUS_BODY" | grep -q "__CHECKS_FAILED__"; then
    CHECKS_OK=no
fi
STATUS_BODY="$(printf '%s\n' "$STATUS_BODY" | grep -v '__CHECKS_FAILED__')"

echo "[watcher] CHECK C: skipped (seed 43 llama33 not in scope for this run)"
CHECK_C="CHECK C [SKIP]  seed-43 llama33 not run yet -- compare_seed_configs N/A"

echo "[watcher] CHECK D: python3 -m pytest tests/ -q"
PYTEST_OUT="$(python3 -m pytest tests/ -q 2>&1)"; PYTEST_RC=$?
PYTEST_TAIL="$(printf '%s\n' "$PYTEST_OUT" | tail -n 3)"
if [ "$PYTEST_RC" -eq 0 ]; then
    CHECK_D="CHECK D [PASS]  pytest: $(printf '%s' "$PYTEST_TAIL" | tail -n 1)"
else
    CHECK_D="CHECK D [WARN]  pytest exit ${PYTEST_RC} (does NOT affect collected data):
$(printf '%s\n' "$PYTEST_TAIL" | sed 's/^/    /')"
fi

{
    echo "# llama3.3:70b SEED-42 static baseline run — full summary"
    echo
    echo "Written (UTC): $FINISHED"
    echo
    echo "## Run scope"
    echo
    echo "- **Model:** \`llama3.3:70b\` (Ollama tag, Q4_K_M ~42 GB)"
    echo "- **Approach:** \`${APPROACH}\` (static baseline — persistent model, no persona, safety filter on)"
    echo "- **Seed:** ${SEED} only"
    echo "- **Samples:** ${MAX_PER_BENCHMARK} per benchmark × 10 benchmarks = **50 inference calls**"
    echo "- **Ollama models dir:** \`/workspace/ollama_models\`"
    echo "- **Pod:** \`${RUNPOD_POD_ID:-unset}\`"
    echo
    echo "The 5 SLM ablation data (seeds 42 & 43) were **not** modified by this run."
    echo
    if [ "$TIMED_OUT" = yes ]; then
        echo "**KILLED BY ${MAX_HOURS}h WALL-CLOCK CEILING** — partial results only; re-run resumes from checkpoint."
        echo
    fi
    if [ "$CHECKS_OK" != yes ]; then
        echo "## WARNING — PRE-SHUTDOWN CHECKS FAILED"
        echo
        echo "Review failing checks below before trusting this data for the paper."
        echo
    fi
    echo "## Pre-shutdown checks"
    echo
    echo '```'
    printf '%s\n' "$STATUS_BODY" | sed -n '1,2p'
    printf '%s\n' "$CHECK_C"
    printf '%s\n' "$CHECK_D"
    echo '```'
    echo
    echo "## Run status (from checkpoint)"
    echo
    echo '```'
    printf '%s\n' "$STATUS_BODY" | sed -n '3,$p'
    echo '```'
    echo
    echo "## Output files"
    echo
    echo "- \`report-output/ghost_agents/benchmark_results/${MODEL_KEY}/\`"
    echo "- \`report-output/ghost_agents/run_logs/llama33_70b_seed42.log\`"
    echo
    echo "## Next steps"
    echo
    echo "- Seed 43 for \`${MODEL_KEY}\` is still pending if you want mean±std across two seeds."
    echo "- \`gpt-oss:120b\` LLM baseline not run on this pod."
} > "$MARKER"
echo "[watcher] summary marker: $MARKER"

PUSH_RESULT="not attempted"
git add "report-output/ghost_agents/benchmark_results/${MODEL_KEY}/" \
        "report-output/ghost_agents/benchmark_results/"*"${MODEL_KEY}"*.json \
        "report-output/ghost_agents/run_logs/llama33_70b_seed42.log" \
        "$MARKER" 2>/dev/null
git add -f "$REPO/watch_seed42_llama33_then_push_and_stop.sh" \
        "$REPO/run_seed42_llama33.sh" 2>/dev/null

if git diff --cached --quiet; then
    echo "[watcher] nothing staged -- skipping commit"
    PUSH_RESULT="nothing to commit"
else
    git commit -q -F - <<EOF
results(seed42): llama3.3:70b static baseline

Run finished $FINISHED. ${APPROACH} only, seed ${SEED};
${MAX_PER_BENCHMARK} samples/benchmark × 10 benchmarks.
Pre-shutdown checks: ${CHECKS_OK} (see $(basename "$MARKER")).

$STATUS_BODY

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
    HEAD_SHORT="$(git rev-parse --short HEAD)"
    echo "[watcher] committed $HEAD_SHORT"
    if git push origin "$BRANCH"; then
        git fetch -q origin "$BRANCH" 2>/dev/null
        if [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH" 2>/dev/null)" ]; then
            PUSH_RESULT="pushed $HEAD_SHORT (CHECK E [PASS] -- origin/$BRANCH matches local HEAD)"
        else
            PUSH_RESULT="pushed $HEAD_SHORT but CHECK E [WARN] -- remote/local mismatch after fetch"
            CHECKS_OK=no
        fi
    else
        PUSH_RESULT="PUSH FAILED -- commit $HEAD_SHORT is local only on /workspace"
        CHECKS_OK=no
    fi
fi
printf '\n## Push\n\n%s\n' "$PUSH_RESULT" >> "$MARKER"

echo "[watcher] stopping pod ${RUNPOD_POD_ID:-<unset>}"
printf '\n## Pod\n\nStopping pod %s (stop, not terminate -- /workspace preserved).\nPre-shutdown checks overall: %s\n' \
    "${RUNPOD_POD_ID:-<unset>}" "$CHECKS_OK" >> "$MARKER"

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[watcher] RUNPOD_POD_ID unset -- cannot stop"
    exit 1
fi

runpodctl stop pod "$RUNPOD_POD_ID"
echo "[watcher] runpodctl exit: $?"
exit 0
