# Session status — 2026-08-26, post-restart

Written at the end of a Claude Code session that picked up after the
2026-08-25 GPU-revocation incident (`/workspace/RESTART_CHECKLIST.md`, on
the network volume, not this repo). Pod is being stopped after this session
by the user's own choice, not a crash. Read this before doing anything else.

## State right now

- GPU confirmed healthy: `nvidia-smi` clean, `torch.cuda.is_available()` ==
  True, Ollama detects the A100 80GB via CUDA correctly.
- Ollama v0.32.15 reinstalled (overlay was wiped by the restart), pointed at
  `OLLAMA_MODELS=/workspace/.ollama/models` -- models were intact, no re-pull
  needed.
- `verify_pod_attribution.py` passed every check: per-process CPU/RAM/GPU
  attribution genuinely works on this hardware (previously unverified on
  real GPU per `POD_HANDOFF.md`).
- Watchdog guards added this session (see "Code changes" below) -- none of
  this existed before today.

## Seed-42 checkpoint status (`report-output/ghost_agents/benchmark_results/`)

| Model | Clean cells (pre-crash, resumable) | Notes |
|---|---|---|
| llama32_3b | 7 | Untouched, resumable as-is |
| deepseek_r1_1_5b | 5 | Untouched, resumable as-is |
| phi3_mini | 0 | Checkpoint present but all partial/dead; will re-run |
| gpt_20b_oss | 0 | Same, plus a pre-existing low completion rate (see below) |
| qwen25_3b | ~~10~~ **0** | **Checkpoint deleted this session** -- see incident below |

### Incident: qwen25_3b's checkpoint was accidentally destroyed

Mid-session, a live orchestrator smoke test (`--models qwen25_3b
--max-per-benchmark 1`) was run against the *real* output directory instead
of an isolated one, and overwrote `qwen25_3b/checkpoint_seed42.json` (its 10
real clean cells) with 1-sample test data. No backup existed that matched
the real interrupted run's config (only a stale, differently-configured
2026-08-22 snapshot, itself later identified as corrupted data). The
contaminated file has now been deleted outright (rather than left for the
config-mismatch auto-discard to handle) so the directory reflects reality.
**Net effect: qwen25_3b starts seed 42 fully from scratch next run.** Cheap
in practice -- it's by far the fastest model (~10s/call) -- but flagging so
it isn't a surprise.

## Code changes this session (all in `git diff`, not yet on `origin` until pushed)

Three watchdog guards, built from `RESTART_CHECKLIST.md`'s post-mortem:

1. **NVML liveness poll** (`per_process_monitor.nvml_gpu_alive`, wired into
   `run_concurrent_experiment.py`'s main loop) -- makes a live per-device NVML
   call every ~2s cycle, not just a cached init check. Catches a mid-run
   device-cgroup revocation (host driver / `/proc` still say healthy, actual
   opens EPERM -- exactly the 2026-08-25 signature) and stops every model.
2. **Circuit breaker** (`benchmark_evaluator.EvaluatorGuardTripped` via
   `_note_cell_outcome`) -- aborts a model's evaluator after
   `EPD_CIRCUIT_BREAKER_DEAD_CELLS` (default 3) consecutive cells with zero
   completed calls. Checkpoints the dead cell before raising, so a resumed
   run retries it.
3. **Latency fail-fast** (`_check_call_latency`) -- each model learns its own
   baseline from its first `EPD_LATENCY_BASELINE_SAMPLES` (default 3)
   *completed* calls; a later call at `EPD_LATENCY_FAILFAST_MULTIPLIER`x
   (default 6x) that baseline, `EPD_LATENCY_FAILFAST_CONSECUTIVE` (default 2)
   times in a row, aborts. This is what would have caught the actual
   incident within 1-2 calls instead of after 30 dead cells (verified with a
   unit test simulating the exact 7s-to-300s CPU-fallback pattern).
4. **`--stop-pod-on-failure`** flag on `run_concurrent_experiment.py` -- off
   by default. When passed, calls `runpodctl stop pod $RUNPOD_POD_ID` (same
   mechanism `auto_shutdown.sh` already uses) if the GPU watchdog trips, or
   if every model in the run ends up failed. Never triggered by one model
   failing among others still working.

All three were validated: unit tests for the two evaluator-side guards
(confirmed both trip and both reset correctly), a real small end-to-end run
that completed 7/8 cells with zero false trips, and `_stop_pod` tested with
`subprocess.run` mocked (confirmed it builds the right command without
touching the real pod).

## Finding: gpt-oss:20b should NOT run concurrently with the other 4 models

Measured directly this session, same calls, solo vs. under real 4-model GPU
contention:

| Call | Solo | Under 4-model contention |
|---|---|---|
| REFUSAL-style | 33.3s | 88.1s (2.6x) |
| GENERATION-style | 57.3s (hit token cap) | **timeout at 300s -- zero output** |

`ollama ps` confirms gpt-oss:20b is 100% GPU-resident (29/29 layers), so
this isn't a CPU-offload problem -- it's a mixture-of-experts model (32
experts, 4 active/token), which is memory-bandwidth-bound in a way the other
4 (dense, much smaller) models aren't. This most likely explains why
gpt-oss's pre-crash completion rate was only 8/40 (20%) even before the GPU
was lost.

**Recommendation**: run gpt-oss:20b in its own separate
`run_concurrent_experiment.py --models gpt_20b_oss` invocation, not
concurrently with the other 4. Rough estimates at 5 samples x 2 seeds
($1.39/hr):

- 4 SLMs concurrent (qwen25_3b, llama32_3b, phi3_mini, deepseek_r1_1_5b):
  ~27h, ~$37 (bottleneck: deepseek_r1_1_5b)
- gpt-oss:20b solo: ~10h, ~$14 (**n=2 calls only, static-style, no ephemeral
  reload overhead included -- treat as a rough floor, not a tight number**)
- Sequential on one pod: ~37h, ~$51 total
- Parallel on two pods: ~27h wall-clock, ~$51 total (pods bill separately)

## Next steps, in order

1. Reinstall on restart (same as `RESTART_CHECKLIST.md` steps 1-2): `curl
   -fsSL https://ollama.com/install.sh | sh`, `pip install nvidia-ml-py`,
   `pip install -r requirements.txt`, start `ollama serve` with
   `OLLAMA_MODELS=/workspace/.ollama/models`, verify GPU with
   `verify_pod_attribution.py` before running anything real.
2. Consider a slightly bigger gpt-oss solo calibration (include at least one
   *ephemeral* cell) before committing pod-hours to the ~10h estimate above.
3. Launch the 4 SLMs concurrently as before; launch gpt-oss:20b separately.
   Pass `--stop-pod-on-failure` on both if you want the new guard to end
   billing automatically on a broken run.
4. Watch for `data_quality_warning` and the new guard's `[GUARD TRIPPED]` /
   `GPU liveness check failed` messages in the logs.
