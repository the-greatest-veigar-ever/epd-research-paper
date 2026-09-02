# Overall Data Summary — EPD benchmark dataset (seeds 42 & 43)

> Written 2026-09-02 · branch `runpod-results-slm` · pod `ljsku2csqpmasd` (RunPod A100 80 GB).
> This document is the cross-seed, cross-model overview. The authoritative per-seed
> detail (incident history, methodology, disclosure checklist) lives in the two
> Vietnamese ledgers: [`SEED42_DATA_LEDGER.md`](SEED42_DATA_LEDGER.md) and
> [`SEED43_DATA_LEDGER.md`](SEED43_DATA_LEDGER.md). Where they disagree with a number
> here, they win.

---

## 0. What this dataset is for

The ICONIP draft (*Ephemeral Polymorphic Defense: A Preliminary Study of Small
Language Model Agents for Cloud Security Automation*) evaluates whether short-lived,
stateless SLM agents with randomized role-conditioned prompts (**EPD**) reduce
prompt-injection exposure — measured by **Attack Success Rate (ASR ↓)** — while
preserving **Task Success Rate (TSR ↑)** across ten cybersecurity benchmarks, against
**SLM Static** and **LLM Static** baselines.

That draft reports a **single-pass, seed-42-only** evaluation (Table 1). This dataset
is the follow-up that directly answers draft **limitation #2** ("future work will
introduce repeated-seed evaluation, confidence intervals, and statistical testing"):
a **two-seed** (42 + 43) run of the full model set, with per-model calibrated
generation caps and an explicit non-answer accounting.

**Status: complete.** 2 seeds × 7 model configurations. No runs outstanding; the
remaining work is analysis and writing.

---

## 1. Scope

| Tier | Model (Ollama tag) | Size | Approaches run | Cells/seed | Calls/seed |
|---|---|---|---|---|---|
| SLM — ablation | `phi3:mini` (Phi-3 Mini) | 2.2 GB | 8 (2×2×2) | 80 | 400 |
| SLM — ablation | `llama3.2:3b` (LLaMA 3.2 3B) | 2 GB | 8 | 80 | 400 |
| SLM — ablation | `qwen2.5:3b` (Qwen 2.5 3B) | 2 GB | 8 | 80 | 400 |
| SLM — ablation | `deepseek-r1:1.5b` (DeepSeek R1 1.5B) | 1.1 GB | 8 | 80 | 400 |
| SLM — ablation | `gpt-oss:20b` (GPT 20B OSS) | 14 GB | 8 | 80 | 400 |
| LLM — static baseline | `llama3.3:70b` (LLaMA 3.3 70B) | 43 GB | 1 (`_static`) | 10 | 50 |
| LLM — static baseline | `gpt-oss:120b` (GPT 120B OSS) | 65 GB | 1 (`_static`) | 10 | 50 |

- **Ablation (SLM tier).** Each SLM runs the 2×2×2 grid over `ephemeral? × persona? ×
  safety_filter?` = 8 approaches. `_static` = `static_nopersona_safety` (the SLM
  Static baseline row); `_suicide` = `ephemeral_persona_safety` (full EPD). Name
  mapping is non-uniform — see the ledgers; do not copy file approach names into paper
  tables.
- **LLM static baselines.** `llama3.3:70b` and `gpt-oss:120b` run **only** the
  `_static` approach (persistent agent, no persona, no ephemeral lifecycle, safety
  filter on). They are the paper's **LLM Static Architecture** and exist to anchor the
  tier comparison (chiefly memory footprint, draft §5.3 / Fig. 4) — not to run an
  ablation. 10 cells / 50 calls per seed, 8× smaller than an SLM run.
- **Benchmarks (10, all tiers).** SecurityEval, LLMSecEval *(Generation)*; SecBench,
  SECURE *(Knowledge)*; CyberSecEval, HarmBench *(Refusal)*; CyberBench, ACSE-Eval,
  CyberSOCEval *(Analysis)*; FORMAI *(Safety)*. 5 samples per benchmark per approach.
- **Decoding.** Greedy, `temperature = 0.0`, `num_ctx = 8192` for every model and seed.
  Seeds differ only in benchmark-sample selection.

Total attempted: **2 × (5 × 400 + 2 × 50) = 4,200 calls**; **2 × (5 × 80 + 2 × 10) =
840 cells**.

---

## 2. Units (cell / call)

- **Call** = one prompt → model response → strategy-specific classifier score. Only
  `call_status == success` (a complete answer, `done_reason = stop`) enters ASR/TSR.
- **Cell** = one `(benchmark, approach)` pair = 5 calls. **The cell is the unit of the
  paper's ablation tables**; its ASR/TSR is computed over however many of its 5 calls
  scored (`completion_rate`).
- **Non-answer** statuses (`length_capped`, `truncated`, `timeout`, `empty`,
  `http_error`) are **measurement failures, dropped from ASR/TSR — never scored 0**. A
  reply cut off before the dangerous span would otherwise be scored "safe" because the
  model *did not get to write it*, not because it *refused*. Full rationale:
  [`SEED42_DATA_LEDGER.md` §"Call không được"](SEED42_DATA_LEDGER.md).

`n` (calls scored in a cell) controls ASR resolution: n=5 → {0, .2, .4, .6, .8, 1};
n=1 → {0, 1} only; n=0 → no value (a hole in the ablation table). **Cell thickness,
not raw success rate, is the number to track.**

---

## 3. Inventory & completeness

All files are committed and pushed to `origin/runpod-results-slm` (working tree clean;
local HEAD `2e4f203` == remote). Per-model outputs in
`report-output/ghost_agents/benchmark_results/<model>/` (`benchmark_eval_*`,
`benchmark_summary_*`, `checkpoint_seed*`); combined + multi-seed rollups in the parent
directory; run logs in `report-output/ghost_agents/run_logs/`.

### Seed 42

| Model | Calls | Scoreable | Completion | Empty cells (n=0) | Cap | Regime |
|---|---:|---:|---:|---:|---:|---|
| `llama32_3b` | 400 | 392 | 98% | 0 | 1024 | 4-way concurrent |
| `qwen25_3b` | 400 | 343 | 86% | **2** (ACSE-Eval) | 768 | 4-way concurrent |
| `deepseek_r1_1_5b` | 400 | 340 | 85% | 0 | 3072 | 4-way concurrent |
| `phi3_mini` | 400 | 292 | 73% | 0 | 2048 | 4-way concurrent |
| `gpt_20b_oss` | 400 | 324 | 81% | **2** (ACSE-Eval, SECURE) | 4096 | solo (MoE) |
| `llama33_70b` | 50 | 48 | 96% | 0 | 1024 | solo |
| `gpt_120b_oss` | 50 | 50 | 100% | 0 | 8192 | solo |
| **Seed 42 total** | **2,100** | **1,789** | **85%** | **4 / 420** | | |

### Seed 43

| Model | Calls | Scoreable | Completion | Empty cells (n=0) | Cap | Regime |
|---|---:|---:|---:|---:|---:|---|
| `llama32_3b` | 400 | 391 | 98% | 0 | 1024 | 3-way concurrent¹ |
| `qwen25_3b` | 400 | 349 | 87% | **2** (ACSE-Eval) | 768 | **solo**¹ |
| `deepseek_r1_1_5b` | 400 | 338 | 85% | 0 | 3072 | 3-way concurrent¹ |
| `phi3_mini` | 400 | 285 | 71% | **1** (SECURE) | 2048 | 3-way concurrent¹ |
| `gpt_20b_oss` | 400 | 304 | 76% | **1** (ACSE-Eval) | 4096 | solo (MoE) |
| `llama33_70b` | 50 | 50 | 100% | 0 | 1024 | solo |
| `gpt_120b_oss` | 50 | 50 | 100% | 0 | 8192 | solo |
| **Seed 43 total** | **2,100** | **1,767** | **84%** | **4 / 420** | | |

¹ Seed 43: `qwen25_3b` failed its concurrent run at ~12 min (`llama-server GPU
discovery watchdog timed out`); its data was discarded and it was **re-run solo**. The
other three SLMs therefore ran effectively **3-way** after that point. See
[`SEED43_DATA_LEDGER.md` §"Sự cố qwen GPU-discovery"](SEED43_DATA_LEDGER.md).

**Structural completeness:** every model × seed reached 10/10 benchmarks and its full
approach set. **832 / 840 cells (99.0%)** carry at least one scored call. The **8 empty
cells** are model refusal/truncation on all 5 samples — disclosed, marked "—" in
tables, never interpolated:

```
seed 42  qwen25_3b   ACSE-Eval / static_persona_safety_filter
         qwen25_3b   ACSE-Eval / ephemeral_nopersona_nosafety
         gpt_20b_oss ACSE-Eval / ephemeral_nopersona_nosafety
         gpt_20b_oss SECURE    / ephemeral_nopersona_safety
seed 43  qwen25_3b   ACSE-Eval / static_persona_safety_filter   (repeat of seed 42)
         qwen25_3b   ACSE-Eval / suicide (= ephemeral_persona_safety)
         phi3_mini   SECURE    / static_nopersona_nosafety
         gpt_20b_oss ACSE-Eval / ephemeral_nopersona_nosafety    (repeat of seed 42)
```

`gpt_120b_oss` at the earlier `num_predict=3072` had ACSE-Eval + SECURE fully empty in
**both** seeds; the `8192` re-run (below) closed all four holes.

---

## 4. Results overview

`_static` (SLM Static baseline) and full EPD (`_suicide`) per-cell values, and the
per-approach × per-benchmark grid, are in each model's `benchmark_summary_*` /
`multi_seed_summary_*`. The overview below is the **mean over the 8-approach ablation ×
10 benchmarks** (SLM tier) or the **single `_static` approach × 10 benchmarks** (LLM
tier).

### ASR ↓ (%) — mean over approaches × benchmarks

| Model | Seed 42 | Seed 43 | Note |
|---|---:|---:|---|
| `qwen25_3b` | 3.9 | 3.4 | floor band |
| `llama32_3b` | 6.0 | 6.0 | floor band; one cell (`static_persona_nosafety`) 18% seed 43 — verify |
| `deepseek_r1_1_5b` | 4.4 | 2.7 | floor band |
| `phi3_mini` | 5.5 | 2.3 | floor band |
| `gpt_20b_oss` | 0.84 | 0.88 | near-zero; ASR≠0 only in `safety_filter`-off branch (seed 42) |
| `llama33_70b_static` | 2.0 | 0.0 | seed 42 = 1 call flagged in SecurityEval → single misclassification |
| `gpt_120b_oss_static` | 0.0 | 0.0 | **zero across all 10 benchmarks, both seeds** |

### TSR ↑ (%) — mean over approaches × benchmarks

| Model | Seed 42 | Seed 43 | Δ |
|---|---:|---:|---:|
| `qwen25_3b` | 64.7 | 68.3 | +3.6 |
| `llama32_3b` | 65.5 | 68.1 | +2.6 |
| `deepseek_r1_1_5b` | 66.0 | 67.3 | +1.3 |
| `phi3_mini` | 65.4 | 70.1 | +4.7 |
| `gpt_20b_oss` | 70.9 | 73.4 | +2.5 |
| `llama33_70b_static` | 66.4 | 70.8 | +4.4 |
| `gpt_120b_oss_static` | 71.25 | 74.98 | +3.7 |

**Reading of the two seeds.**

- **ASR sits at a floor (≈ 0–6%) for every model, both seeds.** Individual cells with
  ASR ≠ 0 are 1–2 classifier "unsafe" verdicts out of ~1,500–1,700 scored calls per
  seed — noise level. They do **not** line up on the same ablation branch across seeds
  (seed 42: `gpt_20b_oss` signal in `safety`-off; seed 43: two `safety`-on cells), so
  they should be reported as *single misclassifications*, not as a branch effect.
  `gpt_120b_oss` is the clean case: exactly 0% everywhere.
- **TSR rises +1.3 to +4.7 points from seed 42 to seed 43 for all seven models** —
  same direction, same magnitude. Because the shift is uniform, it is a benign
  seed-to-sample effect, and it is the strongest evidence that the seed-43
  infrastructure incidents (8 CUDA aborts in `gpt_20b_oss`; 1 GPU-discovery timeout in
  `qwen25_3b`, whose bad data was dropped) **did not contaminate the aggregates** — a
  corrupted GPU state would have shown as TSR collapse or ASR spikes; neither
  occurred.
- **Cross-seed aggregation is valid.** `analysis/compare_seed_configs.py` reports
  5/5 SLM-tier models "SAME EXPERIMENT" (exit 0). The two LLM baselines match on
  `num_predict` (llama3.3:70b 1024/1024; gpt-oss:120b 8192/8192 after the re-run) and
  all other content keys. → `mean ± std` over `(model, approach, benchmark)` for all
  **7 model configs**.

---

## 5. Data quality — non-answers and disclosures

Non-answer calls per seed (all dropped from ASR/TSR, never scored 0):

| Seed | SLM+20b non-answer | dominant cause | LLM baseline non-answer |
|---|---:|---|---:|
| 42 | 309 / 2,000 | `length_capped` 234 (phi3 108, qwen 57, gpt_20b 56); `truncated` 20; `timeout` 16; `empty` 39 | 2 (llama33 ACSE-Eval `length_capped`) |
| 43 | 333 / 2,000 | `length_capped` (phi3 113, qwen 51); `truncated` 46 (deepseek) + 37 (gpt_20b); `timeout` 15; `http_error` 4 (gpt_20b CUDA) | 0 |

Root causes, all disclosed in the ledgers:

1. **Low per-model token caps (SLM).** `phi3_mini` (2048) loses 27–29% of calls to
   mid-answer cuts; `qwen25_3b` (768) loses ~13%, concentrated in ACSE-Eval + SECURE
   (→ its 2 empty cells, both seeds). Caps were **not raised** — that would break the
   seed-42 config match. Disclose per-cell `completion_rate` + a sensitivity check
   (ASR/TSR with vs. without n<5 cells).
2. **GPU contention (deepseek, both seeds).** Reasoning model; sharing the A100
   3–4-way slows decode ~3.7× and blows the 300 s window → `empty`/`timeout` (seed 42:
   39 + 16) and `truncated`/`timeout` (seed 43: 46 + 13). Spread evenly across
   benchmarks, so no empty cells either seed.
3. **`gpt-oss:20b` reasoning overrun.** Always emits a `<think>` block before the
   answer; 19–24% of calls exceed `num_predict=4096`. `completion_rate` is even across
   the 8 approaches (0.72–0.90), so the missingness does not favour a branch.
4. **`gpt-oss:20b` CUDA aborts (seed 43 only).** 8 `illegal memory access` crashes in
   an 83-min window → 4 `http_error` (each cell kept ≥1 scored call). Isolated
   (abort + respawn on a fresh CUDA context); the 4 SLMs on the same pod afterwards
   showed no such error. Run `nvidia-smi -q -d ECC,ROW_REMAPPER` on the next pod start
   to close the file.
5. **`gpt-oss:120b` cap 3072 → 8192 (both seeds re-run).** At 3072, both seeds gave
   **36/50 scoreable, 14 `length_capped`**, with ACSE-Eval + SECURE entirely
   unscoreable. Raised to 8192 (`num_ctx` is 8192, prompts run < 2k tokens) and
   **re-run for seed 42 and seed 43** — 50/50 scoreable, 0 cuts, all 10 benchmarks
   populated, max `eval_count` 5,363 (well under the cap). The 3072 files were replaced
   in-place (`21f8534`, `2e4f203`) and no longer exist on disk.

Empty-cell count: **8 / 840** (see §3). Null audit: every null `safe`/`score` matches
a non-answer call exactly, across all 14 runs — **no unintended nulls**. `machine_wide`
resource-attribution rows are filtered from resource averages (`ce80d3a`); their scores
remain valid.

---

## 6. Configuration

| Parameter | Value |
|---|---|
| Seeds | 42, 43 (benchmark-sample selection only) |
| Decoding | greedy, `temperature = 0.0`, `top_p` default |
| `num_ctx` | 8192 (all models) |
| `num_predict` | phi3 2048 · llama3.2:3b 1024 · qwen2.5:3b 768 · deepseek-r1:1.5b 3072 · gpt-oss:20b 4096 · **gpt-oss:120b 8192** · llama3.3:70b 1024 |
| `word_budget_ratio` | 0.7 (shortens answers to cut runtime — a real change to experimental conditions; record it) |
| `generate_timeout_s` | 300 · `reasoning_timeout_mult` 2.0 · `call_retries` 1 (defaults; SLM seed-43 concurrent pinned `EPD_CALL_RETRIES=0`, `EPD_REASONING_TIMEOUT_MULT=1.0` to match seed 42) |
| Samples per (benchmark, approach) | 5 |
| Inference | Ollama (0.33.1 → 0.33.2 across the campaign), 4-bit GGUF |
| Hardware | 1× RunPod A100 80 GB, pod `ljsku2csqpmasd`, 252 vCPU |

**Regime — which efficiency numbers are comparable:**

- **4-way concurrent** (one Ollama server per model): `qwen25_3b`, `llama32_3b`,
  `deepseek_r1_1_5b`, `phi3_mini` — **seed 42 only**.
- **3-way concurrent**: `llama32_3b`, `deepseek_r1_1_5b`, `phi3_mini` — **seed 43**
  (qwen dropped out at ~12 min).
- **Solo** (1 model on the GPU): `qwen25_3b` seed 43; `gpt-oss:20b` both seeds (MoE
  memory-bandwidth bottleneck — 57 s solo vs. 300 s timeout 4-way); `llama3.3:70b` and
  `gpt-oss:120b` both seeds.

→ ASR/TSR are regime-independent and aggregate normally. **Latency / throughput /
GPU% / cost do not**: the only clean cross-model efficiency comparison is the 4
SLMs at seed 42 (4-way). Everything else is annotated "solo" or "3-way, not
comparable". The LLM baselines sit outside the SLM efficiency table entirely — their
role is the memory-footprint comparison (draft Fig. 4: LLM Static ≈ 100% normalized,
SLM / EPD ≈ 8%).

Latency (p50 inference, scored calls): `llama33_70b` 37.7 s (s42) / 44.1 s (s43);
`gpt_120b_oss` 54.3 s (s42) / 60.8 s (s43) — both solo, informational only.

---

## 7. Relation to the ICONIP draft

| Aspect | Draft (Table 1) | This dataset |
|---|---|---|
| Seeds | 42 only, single pass | **42 + 43** (enables `mean ± std`, prompt-level paired tests) |
| Max generation tokens | "Ollama default (128)" | **per-model calibrated caps** 768–8192 (see §6) |
| Non-answer handling | not specified | explicit: 5 statuses dropped from ASR/TSR, per-cell `completion_rate` recorded |
| Model set | 7 models (Table 1) | same 7 — 5 SLM run the 8-approach 2×2×2 grid; 2 LLM run `_static` only |
| Efficiency | model memory footprint | memory footprint **+** measured latency/throughput/GPU%/cost, with regime caveats |

**Consequences for the write-up.** The draft's Table 3/4 rows for
`gpt_120b_oss_static` and `llama33_70b_static` (and the SLM `_static` rows) are **not
directly comparable** to this dataset — the 128-token cap in the draft config
truncates most answers, whereas here caps are set so answers complete. Refresh those
tables from the two-seed data. This dataset addresses draft limitations **#2**
(repeated-seed + statistical testing) and gives the material for **#3** (the per-model
2×2×2 grid, though EPD is still best framed as a combined effect); limitations #1
(rule-based scorer), #4 (no live CNAPP deployment) are unchanged.

**Disclosure checklist for the paper:** [`SEED42_DATA_LEDGER.md` §"Phải khai báo
trong paper"](SEED42_DATA_LEDGER.md) items 1–10 + [`SEED43_DATA_LEDGER.md`
§"Phải khai báo"](SEED43_DATA_LEDGER.md) items 9–16.

---

## 8. Provenance (git, `origin/runpod-results-slm`)

| Commit | Content |
|---|---|
| `2e4f203` | gpt-oss:120b seed 43 re-run @ `num_predict=8192` (replaces 3072) |
| `21f8534` | gpt-oss:120b seed 42 re-run @ `num_predict=8192` (replaces 3072) |
| `6cc6355` | raise gpt-oss:120b `num_predict` 3072 → 8192 in `approaches.py` |
| `998d322` `ae51050` `657afad` `958a6ba` | LLM-baseline chain `20260901_180445`: llama3.3:70b seed 43 + gpt-oss:120b seeds 42/43 @ 3072 |
| `1f7d2fd` `f03e9d2` | llama3.3:70b seed 42 `_static` + run/watcher scripts |
| `a267a73` `16f6796` | 4-SLM seed 43 concurrent (qwen resumed solo) + ledger |
| `d745ce8` `21d8df9` | gpt-oss:20b seed 43 solo + ledger |
| `fe81bed` `8d035d7` | gpt-oss:20b seed 42 solo + ledger |
| `bf1ef17` … `d4905f4` | 4-SLM seed 42 concurrent, watchdog guards, contaminated-data purge (see SEED42 ledger) |

---

## 9. Remaining work (analysis only — no data collection)

1. **Cross-seed aggregation** — `mean ± std` per `(model, approach, benchmark)` for
   ASR/TSR, all 7 model configs.
2. **Sensitivity check** — ASR/TSR with vs. without n<5 cells, both seeds combined.
3. **Prompt-level McNemar / paired tests** — for suspicious ASR≠0 cells, especially
   `llama32_3b / static_persona_nosafety` (18%, seed 43) and the `llama33_70b`
   SecurityEval cell (seed 42).
4. **Efficiency table** — 4-way SLMs = seed 42 only; annotate seed-43 SLMs as 3-way /
   solo; keep the LLM baselines in the memory-footprint comparison, not the latency
   table.
5. **`nvidia-smi -q -d ECC,ROW_REMAPPER`** on the next pod start — close the
   `gpt-oss:20b` CUDA-abort file.
