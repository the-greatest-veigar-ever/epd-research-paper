# Session status — 2026-08-27, after the gpt-oss:20b solo run

Written at the end of a Claude Code session (running off-pod) that verified the
gpt-oss:20b seed-42 **solo** run, audited its data, and brought
`SEED42_DATA_LEDGER.md` fully up to date. Pod is being stopped by the user's
own choice, not a crash. Read this, then read `SEED42_DATA_LEDGER.md` — that
file is the source of truth and it is now current.

## State right now

- **Seed 42 is complete for all 5 models. 2,000 / 2,000 calls, 396/400 cells
  have data.** The 4 SLMs (concurrent) were frozen on 2026-08-27; gpt-oss:20b
  ran solo and finished 2026-08-27 21:40:08 UTC.
- Everything is committed and pushed to `origin/runpod-results-slm`. Latest
  commits: `fe81bed` (gpt-oss results), then this session's ledger update
  (see below).
- Only work left in the whole 2-seed scope: **the second seed**. Nothing else
  is open.

## What this session did

1. Confirmed the gpt-oss:20b solo run finished cleanly (the 16:13 UTC marker
   was a wall-clock kill that wrote no checkpoint and was discarded; the real
   run finished at 21:40). Watcher committed `fe81bed`, pushed, stopped the pod.
2. Audited all 400 gpt-oss call records directly.
3. Rewrote `SEED42_DATA_LEDGER.md` throughout to reflect 5/5 models — new
   history entry, new `gpt_20b_oss` data subsection, every count and table
   recomputed (2,000 records), issue tracker updated, paper-disclosure
   checklist extended, second-seed run commands added.
4. Added `report-output/ghost_agents/_smoke_test/` to `.gitignore` (it was
   always meant to be uncommitted).

## gpt-oss:20b seed-42 facts (so you don't re-derive them)

Files: `benchmark_eval_20260827_161331_seed42_gpt_20b_oss_pid13898_combined.json`,
matching `benchmark_summary_*`, `multi_seed_summary_20260827_213951_*`,
`gpt_20b_oss/checkpoint_seed42.json`. The old 2026-08-25 partial (1/10
benchmarks) is fully superseded.

- **Run mode: solo** (own Ollama server, no other model on the GPU) because
  gpt-oss:20b is MoE (32 experts, 4 active/token) and is memory-bandwidth-bound
  under contention — 57.3s solo vs 300s timeout at 4-way (measured 2026-08-25).
- **Config** (in the checkpoint): `num_predict 4096` (raised from 2048 in
  `01a409b`), `num_ctx 8192`, `temperature 0.0`, `word_budget_ratio 0.7`,
  `generate_timeout_s 300`, `reasoning_timeout_mult 2.0`, `call_retries 1`.
- **Completeness**: 324/400 success. 76 non-scoreable = 56 `length_capped` +
  20 `truncated`, all from over-long answers, not infra (0 timeout/empty/error).
  Concentrated in ACSE-Eval (25), SECURE (24), CyberBench (17). Cells: 48 at
  n=5, 66/80 at n≥3, 2 empty (`ACSE-Eval/ephemeral_nopersona_nosafety`,
  `SECURE/ephemeral_nopersona_safety`).
- **Missingness is roughly even across the 8 approaches** (completion 0.76–0.90),
  so it does not skew the safety-on vs safety-off comparison.
- **Results**: overall ASR 0.84%, TSR 70.9%. ASR is non-zero only in
  SecurityEval, only in `safety_filter`-off approaches (20–25%). Every
  `safety_filter`-on approach is 0% ASR across all 10 benchmarks.
- **Approach naming is inconsistent**: `gpt_20b_oss_static` =
  `static_nopersona_safety`; `gpt_20b_oss_suicide` = `ephemeral_persona_safety`.
  The other 6 follow `gpt_oss_20b_<ephemeral|static>_<persona|nopersona>_<safety|nosafety>`.
- **Resource metrics (latency, throughput, GPU%, cost) are NOT comparable to
  the 4 SLMs** — different measurement regime (solo vs 4-way). Keep gpt-oss out
  of any cross-model efficiency table. ASR/TSR compare fine.

## Is the seed-42 data paper-ready?

Yes, with disclosure. Full assessment is in `SEED42_DATA_LEDGER.md`
("Phải khai báo trong paper"). The short version:

- **The one real risk is single-seed.** With one seed every ASR/TSR carries
  `std = 0.00` by construction, and comparative claims (ephemeral / persona /
  safety-filter) can't be defended statistically at the seed level. Fix:
  either run the second seed (~$17, ~12.5h), or reframe as a
  measurement/case study and do statistics at the **prompt level**
  (~400 items/approach/model → paired McNemar within one seed).
- Everything else (SLM token-capping, deepseek contention, gpt-oss truncation,
  solo-vs-concurrent resource asymmetry, 4 empty cells) is a
  Limitations / Threats-to-Validity paragraph, not a blocker — provided
  `completion_rate` is reported per cell and a with/without-incomplete-cells
  sensitivity check is shown.

## Next steps, in order

1. On pod restart: follow `RESTART_CHECKLIST.md` steps 1–2 (reinstall Ollama +
   pip, `OLLAMA_MODELS=/workspace/.ollama/models ollama serve`, verify with
   `verify_pod_attribution.py`) before running anything billable.
2. Decide: run seed 43, or lock at single-seed. If running seed 43, use the
   commands in `SEED42_DATA_LEDGER.md` → "Chạy seed 43":
   - 4 SLMs concurrent **must** pin `EPD_CALL_RETRIES=0` and
     `EPD_REASONING_TIMEOUT_MULT=1.0` to match the seed-42 regime.
   - gpt-oss:20b solo **must NOT** pin those (keep retry + reasoning-timeout
     on, cap 4096) — its seed-42 run used them.
   - Verify with `python3 analysis/compare_seed_configs.py` (exit 1 on drift).
3. If locking at single-seed: strip all `mean ± std` notation from result
   tables and add the explicit single-seed statement.
