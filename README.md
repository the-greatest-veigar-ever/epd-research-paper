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
│       └── resource_monitor.py         # CPU/RAM sampling + cost estimate per call
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
3. **Runtime efficiency metrics**: `avg_cpu_percent` and `avg_ram_gb` (sampled via `psutil` during each call — GPU utilization is reported as N/A on this platform, see `resource_monitor.py`), `throughput_tasks_per_s`, `p50_inference_latency` / `p95_inference_latency`, and `total_cost_usd` (a documented, editable on-demand-rate estimate, not a metered bill — see `HARDWARE_COST_TABLE` in `resource_monitor.py`).

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
    --approaches phi3_mini --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results

# llama32_3b
ollama pull llama3.2:3b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches llama32_3b --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results

# qwen25_3b
ollama pull qwen2.5:3b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches qwen25_3b --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results

# deepseek_r1_1_5b
ollama pull deepseek-r1:1.5b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches deepseek_r1_1_5b --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results

# gpt_20b_oss (needs ~14GB RAM headroom)
ollama pull gpt-oss:20b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches gpt_20b_oss --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results
```

### Mid-scale LLM baselines (Qwen/DeepSeek 32B)

`gpt-oss:120b`/`llama3.3:70b` don't fit in 48GB (Section 1); `qwen2.5:32b` and `deepseek-r1:32b` (~20GB each) do, and stay in-family with the `qwen2.5:3b`/`deepseek-r1:1.5b` SLMs already in the study, giving a "same family, bigger" LLM-tier comparison instead. These are registered as **static-only** — one baseline cell each (`ephemeral=False, persona=False, safety_filter=True`), not the 8-cell ablation the 5 SLMs get, since they represent a separate LLM-baseline category rather than another point in the SLM ablation cube. They're excluded from the default sweep for the same reason and run by explicit name. Each call still collects the full metric set (ASR/TSR, latency percentiles, CPU/RAM, throughput, cost estimate) automatically — nothing extra to configure for that.

```bash
# qwen25_32b (static baseline only)
ollama pull qwen2.5:32b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches qwen25_32b_static --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results

# deepseek_r1_32b (static baseline only)
ollama pull deepseek-r1:32b
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches deepseek_r1_32b_static --seeds 42 43 44 --max-per-benchmark 10 \
    --output report-output/ghost_agents/benchmark_results
```

Each command is self-contained and can run on its own machine (or sequentially on one, if that's all you have — `--output` can stay the same path each time since every model writes to its own subfolder). If a run gets interrupted (crash, reboot, closed terminal), re-running the identical command resumes from the last `--save-every` checkpoint instead of starting over — it skips (benchmark, approach) cells that already finished and picks up mid-benchmark otherwise; only re-run with a different `--max-per-benchmark`/`--seeds` if you actually want to discard progress and start clean. Once every machine has finished, copy or `rsync` each machine's `report-output/ghost_agents/benchmark_results/` directory into one place and merge:

```bash
python3 analysis/merge_model_outputs.py \
    /path/to/machineA/report-output/ghost_agents/benchmark_results \
    /path/to/machineB/report-output/ghost_agents/benchmark_results \
    /path/to/machineC/report-output/ghost_agents/benchmark_results \
    --out report-output/ghost_agents/benchmark_results
```

This scans every `<model_key>/benchmark_eval_*_seed<N>.json` file under the given roots, regroups them by seed across all machines, and writes a single `multi_seed_summary_merged.json` — feed that into `analysis/ablation_report.py` and `analysis/generate_latex_tables.py` exactly as in Section 6. Each machine should run the *same* `--seeds` values so every model has a result for every seed; the merge does not interpolate missing (model, seed) pairs.

## 8. Generating Performance Charts

If you wish to visualize the performance metrics (such as the SLM vs EPD comparisons or memory usage charts), you can generate PNG charts using the scripts provided in the `performance_archive` directory.

Please refer to the [Performance Archive README](performance_archive/README.md) for detailed instructions on dependencies and how to run the chart generation scripts.
