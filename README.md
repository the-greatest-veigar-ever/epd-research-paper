# EPD - Ghost Agent Security Evaluation Framework

## 1. Overview

This project provides a comprehensive evaluation framework for Large Language Model (LLM) security remediation methodologies, specifically testing models designated for automated adversarial defense (EPD). The framework empirically evaluates the full factorial combination of three EPD ingredients across 10 distinct adversarial benchmarks:

1. **Ephemerality**: Static (persistent, model stays loaded) vs. Suicide/EPD (a fresh, isolated model instance is spawned per request and immediately terminated afterward, purging state).
2. **Randomized persona injection**: On vs. off.
3. **Static safety-filter block**: On vs. off.

Every combination of the three (2x2x2 = 8 "cells") is run per model, so the contribution of each ingredient to ASR/TSR can be isolated rather than only comparing the two historical endpoints (plain static vs. full EPD). Each cell also records runtime efficiency metrics (CPU/RAM utilization, throughput, a documented cost estimate) alongside ASR/TSR, and the whole sweep can be repeated across multiple random seeds with results reported as mean +/- std.

The framework is designed for reproducibility, supporting multiple open-weight architectures including Llama, Qwen, DeepSeek, and Phi.

**Hardware note for this revision:** the full ablation sweep runs on 5 SLMs (phi3:mini, llama3.2:3b, qwen2.5:3b, deepseek-r1:1.5b, gpt-oss:20b), which fit in 48GB of unified memory. `gpt-oss:120b` (~65GB) and `llama3.3:70b` (~43GB) do not fit on that reference machine and are excluded from the default sweep; their single static-baseline rows are still defined (see `LEGACY_LLM_BASELINE_MODELS` in `approaches.py`) and are runnable by explicit name (`--approaches gpt_120b_oss_static`) on hardware with sufficient RAM.

## 2. Prerequisites and Installation

### System Requirements
* **Operating System**: Linux or Windows (with PowerShell)
* **Memory**: Minimum 16 GB unified memory or VRAM recommended. For large models (e.g., Llama 3.3 70B, GPT-OSS 120B), multi-GPU setups or 64GB+ VRAM are required.
* **Environment**: Python 3.9 or higher.

### Backend Setup
The framework relies on a local [Ollama](https://ollama.com) endpoint (`http://localhost:11434`) for standardized model execution. Prior to running evaluations, ensure Ollama is active and pull the required models:

```bash
# Required for the default ablation sweep (fit in 48GB unified memory):
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull deepseek-r1:1.5b
ollama pull gpt-oss:20b

# Optional: LLM baselines, excluded from the default sweep on machines with
# less than ~65GB RAM (see Section 1). Only pull these if you have the memory.
ollama pull llama3.3:70b
ollama pull gpt-oss:120b
```

### Dependency Installation
Clone the repository and install the required dependencies using the provided requirements file:

```bash
git clone https://github.com/the-greatest-veigar-ever/epd-research-paper.git
cd epd-research-paper
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Project Structure

The repository is structured to separate benchmark data from evaluation logic and metric output.

```text
epd-research-paper/
├── benchmarks/                         # Standardized security benchmark datasets
├── report-output/ghost_agents/benchmark_results/
│   ├── <model_key>/                    # e.g. phi3_mini/, llama32_3b/ — one folder per model,
│   │   ├── checkpoint_*.json           #   self-contained (own checkpoint/eval/summary files) so
│   │   ├── benchmark_eval_*_seed<N>.json #   different models can be run on different machines
│   │   └── benchmark_summary_*_seed<N>.json
│   ├── benchmark_eval_*_seed<N>_combined.json  # convenience: all model_keys merged, this machine/seed only
│   └── multi_seed_summary_*.json       # mean±std across seeds, this machine's model_keys only
├── analysis/
│   ├── generate_latex_tables.py        # multi_seed_summary.json -> main.tex Table 3/4 rows (mean±std)
│   ├── ablation_report.py              # multi_seed_summary.json -> per-factor main-effects (component contribution)
│   └── merge_model_outputs.py          # consolidate per-model folders copied from multiple machines
├── src/ghost_agents/
│   └── approach_evaluation/
│       ├── approaches.py               # ConfigurableApproach: full ephemeral x persona x safety factorial
│       ├── benchmark_evaluator.py      # The main multi-benchmark, multi-seed evaluation engine
│       ├── benchmark_test_data.py      # Data loaders and formatters for the 10 benchmarks (seedable)
│       ├── ollama_manager.py           # Memory management engine for EPD instance termination
│       ├── resource_monitor.py         # Machine-wide CPU/RAM sampling + cost models
│       └── per_process_monitor.py      # Per-model CPU/RAM/GPU attribution (concurrent runs)
├── run_runpod_experiment.sh            # Sequential runner, one model at a time
├── run_concurrent_experiment.py        # Concurrent runner, ~1/5 the pod-hours (Section 8)
├── requirements.txt
└── README.md
```

## 4. Usage and Reproducibility

The evaluation engine supports highly customizable, repeatable experiments. All experiments must be run from the root of the repository. Ollama must already be running with the required model tags pulled (see Section 2).

### Running the Full Ablation + Multi-Seed Evaluation
To run the full 2x2x2 factorial ablation (ephemeral x persona x safety, 8 cells per model) across the 5 default SLMs, over 3 seeds, at 100 samples/benchmark:

```bash
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --seeds 42 43 44 \
    --max-per-benchmark 200 \
    --verbose
```

This is the default behavior of `--approaches` (no flag needed) — it excludes the two LLM baselines that don't fit in 48GB RAM (see Section 1). It produces one `benchmark_eval_*.json` / `benchmark_summary_*.json` pair per seed, plus a single `multi_seed_summary_*.json` with mean +/- std ASR/TSR aggregated across seeds — feed that file to the `analysis/` scripts (Section 6).

### Running Targeted Subsets
To evaluate a specific subset of approaches, benchmarks, or bound the sample size for a rapid smoke test:

```bash
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches phi3_static llama32_3b_suicide deepseek_r1_1_5b_suicide \
    --benchmarks SecurityEval HarmBench \
    --seeds 42 \
    --max-per-benchmark 10 \
    --verbose
```

Use `--approaches main_table` to run only the 3-cell-per-model subset that reproduces the original paper's Table 3/4 rows (static / static+persona+safety / suicide-EPD), which is far cheaper than the full 8-cell sweep if you only need to refresh those tables. Use `--list-approaches` to print every valid approach name.

### CLI Argument Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--benchmarks` | `all` | Specific datasets to evaluate (space-separated). |
| `--approaches` | `full_ablation` (all 40 cells, 5 SLMs) | Specific approach names, or shortcuts `main_table` / `full_ablation` (space-separated). |
| `--seeds` | `42` | Seeds to evaluate and aggregate over as mean±std, e.g. `--seeds 42 43 44`. |
| `--max-per-benchmark` | `100` | Limits the number of samples drawn per benchmark, per seed. `0` disables the cap and uses the entire dataset (~9,700 total test cases across all 10 benchmarks — see Section 7 before using this on a long-running machine). |
| `--save-every` | `20` | Checkpoint interval (test cases). Checkpoints are also resume points: re-running the same command after an interruption picks up from the last checkpoint instead of starting over (see Section 7). |
| `--output` | `report-output/...` | Custom output path for JSON metrics files. |
| `--verbose` | `True` | Enables detailed logging of prompts and responses. |
| `--list-benchmarks` | — | Print available benchmarks and exit. |
| `--list-approaches` | — | Print every valid approach name and exit. |

## 5. Output and Metric Interpretation

Upon completion, the framework generates detailed JSON reports in the `report-output/ghost_agents/benchmark_results/` directory (per-seed `benchmark_eval_*.json` / `benchmark_summary_*.json`, plus a combined `multi_seed_summary_*.json`) and prints a consolidated terminal matrix per seed and a mean±std table across seeds.

Each test case records:

1. **Attack Success Rate (ASR)**: The percentage of adversarial attempts that successfully bypass the model's defensive constraints. Lower is better.
2. **Task Success Rate (TSR)**: The measure of functional correctness or intended behavior completion, dynamically calculated based on the specific benchmark strategy (e.g., semantic correctness in code generation vs. absolute refusal in HarmBench).
3. **Runtime efficiency metrics**: `avg_cpu_percent`, `avg_ram_gb`, `avg_gpu_percent`, `avg_gpu_mem_used_gb`, `throughput_tasks_per_s`, `p50_inference_latency` / `p95_inference_latency`, `total_cpu_core_seconds`, and `total_cost_usd`.

   Every one of these carries a **`resource_attribution`** field, and reading it is not optional:

   * `per_process` — measured on this model's own Ollama process tree (`per_process_monitor.py`). Attributable to this model even while others run alongside it.
   * `machine_wide` — sampled across the entire host (`resource_monitor.py`). Only attributable to a model when it is the *only* thing running, which is why the sequential runner exists.

   A `machine_wide` number produced during a concurrent run is an average over whichever models happened to be active and belongs to none of them. The two kinds must never be averaged together; a cell containing both is flagged with `resource_attribution: "mixed"` and a warning.

   GPU fields report `null` with an explanatory `gpu_note` when they cannot be measured (no NVIDIA driver, `nvidia-ml-py` not installed, driver too old for per-process utilization, or a container PID namespace that prevents matching GPU processes to local PIDs). They are never estimated or defaulted to zero — "we could not tell" and "it used none" are different claims.

## 6. Ablation and Multi-Seed Analysis

After a multi-seed run, two scripts turn `multi_seed_summary_*.json` into paper-ready output:

```bash
# Component-contribution report: which of {ephemeral, persona, safety_filter}
# moves ASR/TSR the most, per model and overall (reviewer point 1).
python3 analysis/ablation_report.py report-output/ghost_agents/benchmark_results/multi_seed_summary_<id>.json

# LaTeX table rows for main.tex Table 3 (ASR) & Table 4 (TSR), formatted as
# mean$\pm$std (reviewer point 3).
python3 analysis/generate_latex_tables.py report-output/ghost_agents/benchmark_results/multi_seed_summary_<id>.json
```

## 7. Running Different Models on Different Machines

Every approach's results are written to their own folder, `report-output/ghost_agents/benchmark_results/<model_key>/`, containing that model's checkpoint/eval/summary files for whichever seeds were run on that machine. Nothing about those files depends on which other models exist elsewhere, so this is the intended split for running the sweep across several machines in parallel: one machine per model (or a few models), each pulling only what it needs.

A per-model CLI shortcut (`--approaches <model_key>`) expands to that model's full 8-cell ablation flow, so you don't have to hand-list approach names. Run `--list-approaches` to confirm the exact set on your machine.

**Dataset size**: `--max-per-benchmark 200` disables sampling entirely and runs every test case in every benchmark — run `--list-benchmarks` to see current per-benchmark counts; as of this revision the 10 benchmarks total **9,712 test cases** (SECURE alone is 4,069; SecBench 3,000; CyberSOCEval 1,197 — a handful of benchmarks dominate the total). That's the entire dataset per **(benchmark, approach, seed)** cell, and each model has 8 approach cells, so one model's full 3-seed flow is `9,712 x 8 x 3 ≈ 233,000` inference calls — multi-day to multi-week per model on this hardware, not multi-hour. If that's more than you want to commit a machine to, `--max-per-benchmark 200` (the original scoped-down flow) is roughly 10x cheaper per model and still meets the reviewer's 3-seed requirement, just on a capped sample. **Full flow per model, entire dataset, 3 seeds:**

```bash
# phi3_mini
ollama pull phi3:mini
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches phi3_mini --seeds 42 43 44 --max-per-benchmark 5 \
    --output report-output/ghost_agents/benchmark_results

# llama32_3b
ollama pull llama3.2:3b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches llama32_3b --seeds 42 43 44 --max-per-benchmark 5 \
    --output report-output/ghost_agents/benchmark_results

# qwen25_3b
ollama pull qwen2.5:3b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches qwen25_3b --seeds 42 43 44 --max-per-benchmark 5 \
    --output report-output/ghost_agents/benchmark_results

# deepseek_r1_1_5b
ollama pull deepseek-r1:1.5b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches deepseek_r1_1_5b --seeds 42 43 44 --max-per-benchmark 5 \
    --output report-output/ghost_agents/benchmark_results

# gpt_20b_oss (needs ~14GB RAM headroom)
ollama pull gpt-oss:20b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches gpt_20b_oss --seeds 42 43 44 --max-per-benchmark 5 \
    --output report-output/ghost_agents/benchmark_results
```

### Running the LLM baselines: `run_runpod_experiment.sh`

The LLM-tier baselines are `llama3.3:70b` and `gpt-oss:120b`.

`run_runpod_experiment.sh` runs all 7 approaches (5 SLMs + 2 LLM baselines) end to end and is the actual script used on the pod — the 5-SLM commands above are the same models it runs, spelled out individually; the script itself also runs the two LLM baselines, which is what most needs walking through here since it has real memory constraints the SLM commands don't:

```bash
./run_runpod_experiment.sh
```

**Before starting**, on the pod itself:
- **Run it where an SSH drop can't kill it.** The full 7-model sequential flow is a multi-hour-to-multi-day run by design (see above); a plain foreground shell gets SIGHUP'd if the connection drops. Start a `tmux`/`screen` session first (`tmux new -s epd`), or launch both `ollama serve` and this script with `nohup ... &`, so the run survives a dropped connection. Checkpointing means you won't lose data either way, but an unnoticed kill can leave the pod idle and billing with nothing running.
- **Size the volume before pulling models**, not after. Budget for the ~108GB of LLM weights, ~20GB of SLM weights, Ollama's own blob-store overhead, the benchmark data, and result output — comfortably more than the default RunPod volume size. A disk-full failure mid-pull can corrupt Ollama's model store, not just fail cleanly.

What it does, in order:

1. **Preflight** — checks `ollama list` for all 7 model tags before starting (the 5 SLMs plus `llama3.3:70b` and `gpt-oss:120b`); fails fast with the exact `ollama pull` commands if any is missing, rather than burning a whole model's worth of calls against something that isn't there. Pulling the two LLM tags needs **~108GB of disk**.
2. **Sequential run, all 7** — one model at a time, one process at a time, in the order phi3_mini → llama32_3b → qwen25_3b → deepseek_r1_1_5b → gpt_20b_oss → llama33_70b_static → gpt_120b_oss_static. This is deliberate, not just a memory-saving measure: CPU/RAM/GPU are sampled machine-wide (`psutil`/`nvidia-smi` have no per-process attribution), so any concurrency — even among the small SLMs, which fit together in RAM fine — makes those metrics an average across whichever models happened to be active, not a per-model reading. Full sequential execution is the only way to keep them valid.
3. **Purge before every model**, not just before the LLM baselines — the script calls `/api/ps` and unloads (`keep_alive:0`) whatever's resident before each of the 7 runs. This matters even between the SLMs: static approaches never self-unload, so without an explicit purge each next model's RAM/CPU/GPU reading would include the previous model's leftover weights on top of its own.
4. **LLM baselines** (`llama33_70b_static`, `gpt_120b_oss_static`) run last, each static-only — one baseline cell, not the 8-cell ablation. `llama3.3:70b` uses `num_predict=1024`, a reasoned value by analogy to its family-mate `llama3.2:3b` (which showed zero failures with margin at the same cap in the SLM-tier calibration below) — not independently verified, since the 70B model exceeds the local machine's RAM. `gpt-oss:120b` uses `num_predict=3072`, set to match its smaller same-family sibling `gpt-oss:20b`'s *empirically observed* need (see `MODEL_NUM_PREDICT` in `approaches.py`) rather than being an independent guess — still not directly measured on this specific model, so check `truncated`/`length_capped` counts on a real run before trusting it at scale.

`OLLAMA_MAX_LOADED_MODELS`/`OLLAMA_NUM_PARALLEL` exports in the script only take effect if `ollama serve` is (re)started after they're set — see the inline comments in the script for the exact restart sequence.

Each command is self-contained and can run on its own machine (or sequentially on one, if that's all you have — `--output` can stay the same path each time since every model writes to its own subfolder). If a run gets interrupted (crash, reboot, closed terminal), re-running the identical command resumes from the last `--save-every` checkpoint instead of starting over — it skips (benchmark, approach) cells that already finished and picks up mid-benchmark otherwise; only re-run with a different `--max-per-benchmark`/`--seeds` if you actually want to discard progress and start clean. Once every machine has finished, copy or `rsync` each machine's `report-output/ghost_agents/benchmark_results/` directory into one place and merge:

```bash
python3 analysis/merge_model_outputs.py \
    /path/to/machineA/report-output/ghost_agents/benchmark_results \
    /path/to/machineB/report-output/ghost_agents/benchmark_results \
    /path/to/machineC/report-output/ghost_agents/benchmark_results \
    --out report-output/ghost_agents/benchmark_results
```

This scans every `<model_key>/benchmark_eval_*_seed<N>.json` file under the given roots, regroups them by seed across all machines, and writes a single `multi_seed_summary_merged.json` — feed that into `analysis/ablation_report.py` and `analysis/generate_latex_tables.py` exactly as in Section 6. Each machine should run the *same* `--seeds` values so every model has a result for every seed; the merge does not interpolate missing (model, seed) pairs.

## 8. Running Concurrently: `run_concurrent_experiment.py`

`run_runpod_experiment.sh` runs the five SLMs strictly one at a time. That is not a memory constraint — together they need ~21GB, trivial on an A100 — but a measurement one: `psutil` and `nvidia-smi` sample the *machine*, so any concurrency turns every CPU/RAM/GPU figure into an average across whichever models were running. Sequential execution buys valid numbers at roughly **5x the pod-hours**.

`run_concurrent_experiment.py` removes that trade-off instead of picking a side. Resource usage is attributed **per OS process**, so all five models run together and still report separate figures.

```bash
python3 run_concurrent_experiment.py --pod-hourly-usd 1.89
```

### How attribution survives the ablation

Per-process measurement is only as good as the process identity behind it, and the ablation deliberately destroys that identity: every **ephemeral** cell unloads and reloads its model *on every call*, and each reload is a new OS process with a new PID. Anchoring a monitor to a PID would lose the model within one call and — once the kernel recycles that number — silently start recording an unrelated process under the model's name.

So the runner gives each model its own `ollama serve`, on its own port, **for the entire run**, and anchors the monitor on that server. The server is long-lived, so the anchor never churns; the runner children appear and vanish underneath it exactly as ephemerality dictates. Because no other model is ever routed to that server, a new child under it is unambiguously *this* model's reload. Process identity is tracked as `(pid, create_time)`, never PID alone, so a recycled PID can never inherit a dead process's accounting.

A server is never handed from one model to another — even with `--max-parallel` below the model count, a queued model gets a freshly started server on its own port.

### What is and is not recoverable concurrently

| Metric | Concurrent run |
| :--- | :--- |
| **ASR / TSR** | Unaffected. Scored from response content at `temperature=0.0` with a fixed per-seed seed; contention changes timing, not what the model said. |
| **CPU %, RAM, GPU memory** | Exact per model. These are per-PID OS/driver accounting, not a shared total needing division. |
| **GPU utilization %** | Per-process via NVML (Volta+, so the A100 qualifies). Reported as `null` with a note where the driver does not support it. |
| **Latency / throughput** | Measured, but genuinely reflects N-way contention. There is no valid way to reconstruct an isolated-hardware latency from a number measured under load — report it as "under N-way concurrent load" rather than as a dedicated-hardware figure. |
| **Cost** | *More* accurate than before: one real pod bill instead of pretending each model rented its own instance. |

### Cost

Set `--pod-hourly-usd` to the machine's real rate and `estimate_cost_usd` switches from the per-model `HARDWARE_COST_TABLE` (which assumes a dedicated instance per model, and overstates a shared pod several-fold) to one bill split across the models sharing it. Each call also records `cpu_core_seconds` and `gpu_mem_gb_seconds`, so the flat even split applied at call time can be replaced with a usage-weighted one afterwards via `resource_monitor.apportioned_cost()`, using the `pod_wall_seconds` recorded in the run manifest.

### Useful flags

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--models` | all 5 SLMs | Model keys to run. |
| `--max-parallel` | all of them | Models running at once. |
| `--base-port` | `11500` | Each model gets `base-port + its index in the registry`, so a model always lands on the same port and two can never collide. |
| `--monitor-interval` | `0.5` | Sampling cadence, seconds. |
| `--keep-alive` | `-1` | `OLLAMA_KEEP_ALIVE` per server. The default keeps models resident indefinitely so a **static** cell really does stay loaded between calls; Ollama's own 5-minute idle default would quietly convert static cells into ephemeral ones during any gap. Ephemeral cells' explicit `keep_alive: 0` teardown still overrides it per request. |
| `--pod-hourly-usd` | unset | Real hourly rate; enables the shared-pod cost model. |
| `--dry-run` | — | Preflight and print the plan without running. |

Preflight fails fast on a missing `ollama` binary, an un-pulled model tag (matched **exactly** — `phi3:medium` will not be accepted for `phi3:mini`), an occupied port, or a combined VRAM requirement above 90% of the card. Outputs land in the same per-model folders as the sequential runner, so the `analysis/` scripts are unchanged; each model additionally gets `run_logs/resource_timeseries_<model_key>.csv` and the run writes `run_logs/run_manifest_<id>.json`.

Per-process GPU metrics need `nvidia-ml-py` (in `requirements.txt`). Inside Docker, NVML reports **host** PIDs while the container sees namespaced ones; if they cannot be matched the GPU fields report `null` with an explicit note rather than zero. Run the container with host PID visibility (`--pid=host`) to recover GPU attribution — CPU and RAM attribution work either way.

## 9. Generating Performance Charts

If you wish to visualize the performance metrics (such as the SLM vs EPD comparisons or memory usage charts), you can generate PNG charts using the scripts provided in the `performance_archive` directory.

Please refer to the [Performance Archive README](performance_archive/README.md) for detailed instructions on dependencies and how to run the chart generation scripts.
