# 4-Approach Ghost Agent Evaluation

Compares **4 approaches** for Squad C (Ghost Agents) on initialization time, processing time, ASR, TSR, and PASS@1.

## Approaches

| # | Name | Models | Lifecycle |
|---|------|--------|-----------|
| 1 | `phi_baseline` | phi3:mini | Static — model stays loaded |
| 2 | `phi_suicide` | phi3:mini | Ephemeral — load before, unload after each plan |
| 3 | `multimodal_static` | phi3:mini, llama3.2:3b, gemma2:2b | Static — all pre-loaded, random selection |
| 4 | `multimodal_suicide` | phi3:mini, llama3.2:3b, gemma2:2b | Ephemeral — random select, load, execute, unload |

## Prerequisites

1. **Ollama** must be running locally (`http://localhost:11434`)
2. All required models must be pulled:
   ```bash
   ollama pull phi3:mini
   ollama pull llama3.2:3b
   ollama pull gemma2:2b
   ```
3. Dataset JSON files must exist in `report-output/integration_tests/`

## How to Run

Run from the **project root** (`epd-research-paper/`):

```bash
# Run all 4 approaches on the full dataset (all JSON files in integration_tests/)
python -m src.ghost_agents.approach_evaluation.run_evaluation

# Limit to N plans for quick testing
python -m src.ghost_agents.approach_evaluation.run_evaluation --limit 10

# Run specific approaches only
python -m src.ghost_agents.approach_evaluation.run_evaluation --approaches phi_baseline,phi_suicide

# Use a specific dataset file instead of the directory
python -m src.ghost_agents.approach_evaluation.run_evaluation \
    --dataset report-output/integration_tests/cic_integration_results_20260216_011439.json

# Custom output directory
python -m src.ghost_agents.approach_evaluation.run_evaluation --output-dir my_results/
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `report-output/integration_tests` | Path to a JSON file or directory of JSON files |
| `--limit` | all plans | Max number of plans to evaluate |
| `--approaches` | `all` | Comma-separated approach names or `all` |
| `--output-dir` | `report-output/ghost_agents/approach_comparison` | Output directory for JSON reports |

## Expected Output

### Console Output

Each approach is evaluated sequentially. You'll see:

```
======================================================================
  EVALUATING: phi_baseline
  Models: ['phi3:mini']
  Suicide Mode: False
  Plans: 4026
======================================================================

[phi_baseline] Phase 1: Initialization
[phi_baseline] Preloading phi3:mini...
[phi_baseline] Initialization done: 5.485s
[phi_baseline] Phase 2: Processing 4026 plans
phi_baseline:  100%|██████████████████████████| 4026/4026 [2:40:00<00:00, 2.40s/plan]

--- Results: phi_baseline ---
  Plans Processed:        4026/4026
  Initialization Time:    5.485s
  Avg Per-Plan Init Time: 0.000s
  Avg Processing Time:    2.400s
  Total Processing Time:  9662.400s
  ASR:                    95.00%
  TSR:                    90.00%
  PASS@1:                 95.00%
  Model Distribution:     {'phi3:mini': 4026}
```

After all approaches complete, a comparison table is printed:

```
=============================================================================================================
  APPROACH COMPARISON
=============================================================================================================
Approach                  |  Plans |  One-time Init |  Avg Plan Init |   Avg Proc |     ASR |     TSR |  PASS@1
-------------------------------------------------------------------------------------------------------------
phi_baseline              |   4026 |         5.485s |         0.000s |     2.400s |  95.00% |  90.00% |  95.00%
phi_suicide               |   4026 |         0.000s |         0.430s |     2.500s |  94.00% |  89.00% |  94.00%
multimodal_static         |   4026 |        15.000s |         0.000s |     2.200s |  96.00% |  91.00% |  96.00%
multimodal_suicide        |   4026 |         0.000s |         0.940s |     2.300s |  95.00% |  90.00% |  95.00%
=============================================================================================================
```

> **Note**: The numbers above are illustrative examples. Actual values depend on your hardware and Ollama performance.

### Output Files

Two JSON files are saved to the output directory:

| File | Contents |
|------|----------|
| `comparison_summary_<timestamp>.json` | Compact metrics only — init time, processing time, ASR, TSR, PASS@1 per approach |
| `comparison_detailed_<timestamp>.json` | Full report including per-plan results (action, model used, command generated, timing) |

If the run is interrupted with **Ctrl+C**, partial results are still saved with a `_partial` suffix.

### Summary JSON Structure

```json
{
  "evaluation_type": "approach_comparison_summary",
  "timestamp": "2026-02-16T22:30:00",
  "dataset": "report-output/integration_tests",
  "total_plans_in_dataset": 4026,
  "comparison": {
    "phi_baseline": {
      "models": ["phi3:mini"],
      "suicide_mode": false,
      "plans_processed": 4026,
      "initialization_time_s": 5.485,
      "per_plan_init_time_avg_s": 0.0,
      "total_init_overhead_s": 5.485,
      "avg_processing_time_s": 2.4,
      "total_processing_time_s": 9662.4,
      "asr_pct": 95.0,
      "tsr_pct": 90.0,
      "pass_at_1_pct": 95.0,
      "tool_correctness_pct": 92.0,
      "model_distribution": {"phi3:mini": 4026}
    }
  }
}
```

## Metrics Glossary

| Metric | Definition |
|--------|------------|
| **One-time Init** | Single upfront cost to load models (high for static, 0 for suicide) |
| **Avg Plan Init** | Average time to load models per plan (0 for static, >0 for suicide) |
| **Processing Time** | Inference time per plan (excluding model load/unload) |
| **ASR** (Attack Success Rate) | Successful executions / total plans |
| **TSR** (Task Success Rate) | Successful executions with correct tool / total plans |
| **PASS@1** | Single-pass success rate (equals ASR in this context) |

## Estimated Runtime

With ~4000 plans at ~2.4s/plan per approach:
- **Single approach**: ~2.5 hours
- **All 4 approaches**: ~10+ hours (suicide approaches are slower due to model load/unload overhead)

Use `--limit` for quick validation runs before committing to a full evaluation.
