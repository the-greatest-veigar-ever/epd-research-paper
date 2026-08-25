#!/usr/bin/env bash
#
# NOTE: run_concurrent_experiment.py now does this same sweep with all the
# models running at once, for roughly a fifth of the pod-hours, WITHOUT
# giving up per-model resource figures -- it attributes CPU/RAM/GPU per OS
# process instead of per machine, so the sequential-execution argument
# below no longer has to be paid for. This script remains the reference
# sequential implementation and the fallback for a host where per-process
# attribution is unavailable. See README Section 8.
#
# RunPod scoped experiment: 5 SLMs (full 8-cell ablation), optionally
# followed by the 2 legacy LLM baselines (static-only, opt-in via
# INCLUDE_LLM_BASELINES=1 -- their weights are ~108GB and are not pulled
# by default, so including them without pulling first just fails preflight).
# Everything selected runs strictly sequentially --
# one model resident at a time, one process at a time -- so that CPU/RAM/
# GPU readings (sampled machine-wide by ResourceMonitor/nvidia-smi) are
# attributable to the model actually being measured, not contaminated by
# sibling processes or a previous model's leftover weights.
#
# This trades speed for measurement validity: the SLMs previously ran as
# 5 concurrent processes (safe from a memory standpoint -- ~21GB together,
# trivial on 80GB), which is faster but makes every resource metric an
# average across whichever models happened to be active at that instant.
# If you need those numbers to be genuinely per-model, concurrency has to
# go, on both phases -- there is no available middle ground: psutil/
# nvidia-smi have no per-process attribution at any layer (see
# resource_monitor.py's own doc comment).
#
# 3 seeds, 5 samples/benchmark -- matches the scope costed out in the
# RunPod compute-budget memo. --seeds is still passed as all 3 seeds to a
# single process per model; seeds run sequentially within that process
# already (no code change needed there).
#
# Usage: ./run_runpod_experiment.sh
# Per-model results land under $OUTPUT_DIR/<model_key>/ as usual; nothing
# about resuming/checkpointing changes, so a killed/interrupted run can
# just be re-run and it picks up from the last checkpoint per model.

set -uo pipefail

SEEDS="42 43 44"
MAX_PER_BENCHMARK=5
OUTPUT_DIR="report-output/ghost_agents/benchmark_results"
LOG_DIR="report-output/ghost_agents/run_logs"
# Checkpoint cadence, in test cases. The default (20) never fires at
# MAX_PER_BENCHMARK=5: the evaluator counts within a single
# (benchmark, ablation-cell) pair, which is only 5 cases long, so the sole
# surviving checkpoint was the one written after each whole benchmark --
# 8 cells x 5 calls of exposure, an hour of gpt-oss:20b work. 5 makes it
# one checkpoint per completed cell; the writes are compact and
# preview-stripped, so the extra cost is negligible next to a 60s call.
SAVE_EVERY=5
mkdir -p "$LOG_DIR"

SLM_MODELS=(phi3_mini llama32_3b qwen25_3b deepseek_r1_1_5b gpt_20b_oss)
LLM_MODELS=(gpt_120b_oss_static llama33_70b_static)

# Ollama tags to check for in the preflight (approach names above are
# EPD's internal keys, not what `ollama list` prints).
SLM_TAGS=(phi3:mini llama3.2:3b qwen2.5:3b deepseek-r1:1.5b gpt-oss:20b)
LLM_TAGS=(llama3.3:70b gpt-oss:120b)

# The LLM baselines are opt-in. They are static-only (1 cell each, no
# ablation sweep), need ~108GB of weights that this script does not pull,
# and -- critically -- the preflight below checks EVERY tag in the selected
# set, so leaving them in unconditionally made the script exit 1 before a
# single call whenever they were absent, which is the normal state of a pod
# scoped to the 5 SLMs.
INCLUDE_LLM_BASELINES="${INCLUDE_LLM_BASELINES:-0}"
if [ "$INCLUDE_LLM_BASELINES" = "1" ]; then
    ALL_APPROACHES=("${SLM_MODELS[@]}" "${LLM_MODELS[@]}")
    ALL_TAGS=("${SLM_TAGS[@]}" "${LLM_TAGS[@]}")
else
    ALL_APPROACHES=("${SLM_MODELS[@]}")
    ALL_TAGS=("${SLM_TAGS[@]}")
fi

run_model() {
    local approach="$1"
    local log_file="$LOG_DIR/${approach}.log"
    echo "[$approach] starting -> $log_file"
    # -u (unbuffered): stdout is a file here, so Python block-buffers it and
    # progress appears in multi-KB bursts -- on a multi-hour model that means
    # tailing a log that looks hung. stderr (tqdm) is unbuffered either way.
    uv run python3 -u -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
        --approaches "$approach" \
        --seeds $SEEDS \
        --max-per-benchmark "$MAX_PER_BENCHMARK" \
        --save-every "$SAVE_EVERY" \
        --output "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    echo $? > "$LOG_DIR/${approach}.exitcode"
}

# Explicitly evicts every model currently resident on the Ollama server.
# Required before EACH model here, not just once: a static approach's
# teardown deliberately does not unload (staying loaded is what "static"
# means), so without this the next model's RAM/CPU/GPU readings would
# include the previous model's leftover weights on top of its own --
# contamination from carryover, not concurrency, but just as real for
# measurement purposes.
purge_all_models() {
    local loaded
    loaded=$(curl -s http://localhost:11434/api/ps | python3 -c \
        "import json,sys; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" \
        2>/dev/null)
    if [ -n "$loaded" ]; then
        for m in $loaded; do
            echo "  purging $m"
            curl -s http://localhost:11434/api/generate \
                -d "{\"model\":\"$m\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":0}" > /dev/null
        done
    fi
}

echo "========================================================"
echo " Sequential run: ${#ALL_APPROACHES[@]} approaches, one model at a time"
echo " Models:     ${ALL_APPROACHES[*]}"
echo " Seeds:      $SEEDS   Samples/benchmark: $MAX_PER_BENCHMARK"
echo " LLM baselines: $([ "$INCLUDE_LLM_BASELINES" = "1" ] && echo included || echo "excluded (INCLUDE_LLM_BASELINES=1 to add)")"
echo "========================================================"
# !! OLLAMA_MAX_LOADED_MODELS / OLLAMA_NUM_PARALLEL are read by the OLLAMA
# !! SERVER at startup, not by this script's Python clients. Exporting them
# !! here does nothing unless `ollama serve` is (re)started afterwards with
# !! them set. Restart the server before running this:
# !!
# !!   pkill ollama
# !!   export OLLAMA_MODELS=/workspace/.ollama/models
# !!   export OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1
# !!   ollama serve > ~/ollama.log 2>&1 &
# !!
# !! Verify with: curl -s localhost:11434/api/ps
#
# Both are 1 here on purpose, unlike the old concurrent Phase 1 (which used
# 5/2): only one process is ever active and it only ever has one request in
# flight, so there is no concurrency demand to serve. Keeping the server
# itself capped at 1 loaded model is also a backstop against stale
# residency if purge_all_models() ever misses something.
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

# Generation/timeout limits consumed by approaches.py + ollama_manager.py.
# EPD_TEMPERATURE 0.0 = greedy decoding, matching the paper's config table.
# NUM_PREDICT is the baseline generation cap; reasoning models get a larger
# per-model cap in approaches.py (MODEL_NUM_PREDICT) because their <think>
# block consumes budget before the answer starts. NUM_CTX pins the 8192
# context the paper reports, instead of letting each model use its own
# default.
export EPD_TEMPERATURE="${EPD_TEMPERATURE:-0.0}"
export EPD_NUM_PREDICT="${EPD_NUM_PREDICT:-1024}"
export EPD_NUM_CTX="${EPD_NUM_CTX:-8192}"
export EPD_GENERATE_TIMEOUT="${EPD_GENERATE_TIMEOUT:-300}"
export EPD_PRELOAD_TIMEOUT="${EPD_PRELOAD_TIMEOUT:-600}"

# Preflight: fail fast if a model isn't pulled, rather than burning an
# entire model's worth of calls against something that isn't there (every
# call would return an Ollama error, get correctly excluded from ASR/TSR,
# and the run would finish with nothing but "NO usable results" warnings).
# The LLM tags are large (~43GB + ~65GB, ~108GB combined) and are not
# pulled by this script -- pull them yourself first if missing.
missing=()
for tag in "${ALL_TAGS[@]}"; do
    if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$tag"; then
        missing+=("$tag")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo ""
    echo "ERROR: required model(s) not present: ${missing[*]}"
    echo "Pull them before running:"
    for tag in "${missing[@]}"; do echo "    ollama pull $tag"; done
    exit 1
fi

failed=0
for approach in "${ALL_APPROACHES[@]}"; do
    purge_all_models
    run_model "$approach"
    code="$(cat "$LOG_DIR/${approach}.exitcode" 2>/dev/null || echo "?")"
    if [ "$code" = "0" ]; then
        echo "  [$approach] OK"
    else
        echo "  [$approach] FAILED (exit $code) -- see $LOG_DIR/${approach}.log"
        failed=1
    fi
done

echo ""
if [ "$failed" -ne 0 ]; then
    echo "One or more runs failed -- check the logs above."
    exit 1
fi

echo "All done. Per-model results under $OUTPUT_DIR/<model_key>/."
