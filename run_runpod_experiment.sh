#!/usr/bin/env bash
#
# RunPod scoped experiment: 5 SLMs (full 8-cell ablation, run concurrently)
# followed by the 2 legacy LLM baselines (static-only, run sequentially).
# 3 seeds, 10 samples/benchmark -- matches the scope costed out in the
# RunPod compute-budget memo.
#
# Why SLMs run in parallel and the LLMs don't: the 5 SLMs together need
# only ~21GB of weights, trivial on an 80GB pod, so there's no reason to
# serialize them -- they run as 5 concurrent processes against one Ollama
# server. gpt-oss:120b (~65GB) and llama3.3:70b (~43GB) don't fit together
# (108GB > 80GB), so the LLM phase stays sequential and runs after the SLM
# phase exits and Ollama frees that VRAM.
#
# Usage: ./run_runpod_experiment.sh
# Per-model results land under $OUTPUT_DIR/<model_key>/ as usual; nothing
# about resuming/checkpointing changes, so a killed/interrupted run can
# just be re-run and it picks up from the last checkpoint per model.

set -uo pipefail

SEEDS="42 43 44"
MAX_PER_BENCHMARK=10
OUTPUT_DIR="report-output/ghost_agents/benchmark_results"
LOG_DIR="report-output/ghost_agents/run_logs"
mkdir -p "$LOG_DIR"

SLM_MODELS=(phi3_mini llama32_3b qwen25_3b deepseek_r1_1_5b gpt_20b_oss)
LLM_MODELS=(gpt_120b_oss_static llama33_70b_static)

run_model() {
    local approach="$1"
    local log_file="$LOG_DIR/${approach}.log"
    echo "[$approach] starting -> $log_file"
    python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
        --approaches "$approach" \
        --seeds $SEEDS \
        --max-per-benchmark "$MAX_PER_BENCHMARK" \
        --output "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    echo $? > "$LOG_DIR/${approach}.exitcode"
}

echo "========================================================"
echo " Phase 1/2: SLMs -- ${#SLM_MODELS[@]} models in parallel"
echo "========================================================"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-5}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"

pids=()
for model in "${SLM_MODELS[@]}"; do
    run_model "$model" &
    pids+=("$!")
done

slm_failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || slm_failed=1
done

for model in "${SLM_MODELS[@]}"; do
    code="$(cat "$LOG_DIR/${model}.exitcode" 2>/dev/null || echo "?")"
    if [ "$code" = "0" ]; then
        echo "  [$model] OK"
    else
        echo "  [$model] FAILED (exit $code) -- see $LOG_DIR/${model}.log"
    fi
done

if [ "$slm_failed" -ne 0 ]; then
    echo ""
    echo "One or more SLM runs failed -- check the logs above before continuing to the LLM phase."
    exit 1
fi

echo ""
echo "========================================================"
echo " Phase 2/2: legacy LLM baselines -- sequential"
echo "========================================================"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1

for model in "${LLM_MODELS[@]}"; do
    run_model "$model"
    code="$(cat "$LOG_DIR/${model}.exitcode")"
    if [ "$code" != "0" ]; then
        echo "  [$model] FAILED (exit $code) -- see $LOG_DIR/${model}.log"
        exit 1
    fi
    echo "  [$model] OK"
done

echo ""
echo "All done. Per-model results under $OUTPUT_DIR/<model_key>/."
