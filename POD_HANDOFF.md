# Pod Handoff: Concurrent Sweep with Per-Model Resource Attribution

Instructions for a Claude session running **on the RunPod A100 pod**. Read this
top to bottom, then follow it. Ask before anything destructive.

Reference commit: **`fbe7ca8`** on branch `runpod-results-slm`.

---

## Step 0 — Can this clone reach GitHub?

The pod's clone has been unable to fetch since ~2026-08-22 (`Host key
verification failed`), so it is likely **8+ commits stale**. Check:

```bash
cd /workspace/epd-research-paper
git fetch --all --prune && git cat-file -t fbe7ca8
```

If that prints `commit`, skip to Step 1.

If it fails on the host key, the repo is **public**, so the simplest read-only fix
is to switch to HTTPS:

```bash
git remote set-url origin https://github.com/the-greatest-veigar-ever/epd-research-paper.git
git fetch --all --prune
```

If you also need to **push results back** from the pod (previous runs did), keep
SSH and add the host key — but verify the fingerprint rather than trusting
whatever the network returns:

```bash
ssh-keyscan -t ed25519 github.com > /tmp/gh_key
ssh-keygen -lf /tmp/gh_key
```

Compare the printed `SHA256:...` against GitHub's published fingerprints at
<https://docs.github.com/en/authentication/keeping-your-account-secure/githubs-ssh-key-fingerprints>.
Only if it matches:

```bash
cat /tmp/gh_key >> ~/.ssh/known_hosts
git fetch --all --prune
```

Then pull:

```bash
git checkout runpod-results-slm && git pull
git log --oneline -1     # expect fbe7ca8 (or later)
```

---

## Step 0.5 — Clear out corrupted results from the previous run

**Do this before running anything.** An earlier sweep was silently corrupted:
Ollama was not actually running, every call failed with connection errors, and
the evaluator scored those failures as real (empty, therefore unsafe) answers.
The result was a uniform `asr=1.0 / tsr=0.0` across all 80 (benchmark x cell)
combinations.

Commit `d4905f4` deleted all of it from git — but this pod is stale, so those
files may still be on disk, and `git pull` only removes the **tracked** copies.

```bash
git status --short
ls report-output/ghost_agents/benchmark_results/*/
```

The dangerous leftovers are `checkpoint_seed4*.json`. The evaluator **resumes**
from checkpoints, so a corrupted one causes it to skip those cells entirely
rather than re-run them — silently preserving the bad data in a run that looks
successful. Report what you find and ask before deleting anything.

---

## Background: what changed, and why

The experiment runs 5 SLMs (`phi3:mini`, `llama3.2:3b`, `qwen2.5:3b`,
`deepseek-r1:1.5b`, `gpt-oss:20b`) through a 2x2x2 ablation (ephemeral x persona
x safety_filter = 8 cells per model), across seeds 42/43/44, measuring ASR/TSR
plus CPU/RAM/GPU per model.

The old runner (`run_runpod_experiment.sh`) ran the 5 models strictly
**sequentially**. Not for memory reasons — together they need only ~21GB, trivial
on an A100 — but for measurement ones: `psutil` and `nvidia-smi` sample the whole
**machine**, so any concurrency turned every CPU/RAM/GPU number into an average
across whichever models happened to be running. That made the numbers valid at
roughly **5x the pod-hours**, which is the cost problem this work exists to solve.

A previous session removed the trade-off by attributing resources **per OS
process** rather than per machine. The new topology: each model gets its own
`ollama serve` on its own port for the entire run, its own evaluator process, and
its own monitor anchored on that server's PID.

### The critical design point

The ablation's **ephemeral** cells unload and reload the model on *every call*, so
the process holding the weights is destroyed and recreated constantly, with a new
PID each time. Anchoring a monitor to a PID would lose the model within one call
and — once the kernel recycled that PID number — start attributing an unrelated
process's usage to it. Anchoring on the long-lived **server** survives that churn:
reloads appear and vanish as children underneath it, and because no other model is
ever routed to that server, a new child under it is unambiguously this model's
reload. Process identity is tracked as `(pid, create_time)`, never PID alone, so a
recycled PID cannot inherit a dead process's accounting.

### Also changed

- `benchmark_evaluator.py` imported `fcntl` unconditionally, so it could not be
  imported on Windows at all. Locking is now cross-platform.
- Cost is now **one real pod bill split across the models sharing it**, instead of
  pricing each model as if it had rented its own dedicated instance (which
  overstates a concurrent run several-fold).
- Servers default to `OLLAMA_KEEP_ALIVE=-1`. Ollama's 5-minute idle default would
  quietly unload a **static** cell's model between calls, converting it into an
  ephemeral one and destroying the ablation's central contrast.

### What is already verified, and what is not

Tested end-to-end against real Ollama on a Windows dev box: 2 models concurrently,
both exit 0, genuinely separate per-model figures (520 vs 1433 CPU-seconds, 1.9 vs
3.0 GB peak RSS) where machine-wide sampling would have given both the same number.

**That machine has no NVIDIA GPU, so GPU attribution is UNVERIFIED on real
hardware.** That is the main thing to check here.

---

## Step 1 — Verify attribution works on this host (cheap, do it first)

```bash
pip install nvidia-ml-py
python3 verify_pod_attribution.py
```

This loads one small model, makes **one** generate call, and reports whether CPU,
RAM and GPU attribution actually work here. About a minute. **Do not skip to the
full sweep** — pod time is expensive and a multi-hour run that turns out to have
null GPU numbers is wasted money.

The two failures that matter:

1. **`NVML init failed`** — `nvidia-ml-py` missing or no driver. CPU and RAM
   attribution still work; all GPU fields report null.
2. **`NVML sees compute processes but none match our PIDs`** — this container has
   its own PID namespace while NVML reports **host** PIDs, so the two sets can
   never intersect. The fix is host PID visibility (`--pid=host`). If that is not
   possible on this pod, say so: GPU numbers will be null, CPU/RAM will still be
   correct, and the decision whether to proceed without GPU figures is the user's
   to make, not something to work around.

This is an **A100 PCIe (Ampere)**, so per-process GPU utilisation via
`nvmlDeviceGetProcessUtilization` is expected to work. If it reports unsupported,
mention it — that is informative, not fatal.

> **Report what the script says. Do not "fix" a failure** by falling back to
> machine-wide sampling or substituting device-wide GPU totals. Under a concurrent
> run those belong to no single model, and reporting them as per-model numbers
> would silently corrupt the paper's results. Null with an explanatory note is the
> correct output when something cannot be measured.

---

## Step 2 — The real run (only after Step 1 passes)

```bash
tmux new -s epd
python3 run_concurrent_experiment.py --pod-hourly-usd 1.39
```

`$1.39/hr` is this pod's compute rate (A100 PCIe 1x, 31 vCPU, 117 GB RAM).

Defaults: all 5 models concurrently, seeds 42/43/44, 5 samples per benchmark. It
is multi-hour, so run it under `tmux`/`screen` — a dropped SSH connection would
otherwise kill it. It checkpoints and resumes, so re-running the identical command
after an interruption picks up where it left off.

Useful variations:

```bash
# Smaller validation slice first (~5 min): 2 models, 1 benchmark, 1 seed
python3 run_concurrent_experiment.py --models qwen25_3b llama32_3b \
    --benchmarks HarmBench --max-per-benchmark 1 --seeds 42 --pod-hourly-usd 1.39

# Preflight only, no run
python3 run_concurrent_experiment.py --dry-run
```

---

## What to watch during the run

- **`resource_attribution` must say `per_process`** in the output JSON. If any
  cell says `machine_wide` or `mixed`, something fell back and those numbers are
  not attributable to a model — flag it, and do not average them with the others.
- **`data_quality_warning`**, or high timeout / truncated counts — means a
  generation cap is too low for that model, or Ollama is struggling.
- **Suspiciously uniform numbers.** The old corruption produced identical
  `asr=1.0 / tsr=0.0` everywhere. That specific bug is fixed (failed calls are now
  excluded from ASR/TSR rather than scored as answers), but if you see uniform
  results, treat them as a measurement failure and investigate before trusting
  anything.

## Caveats — please do not engineer these away

- **Latency and throughput genuinely reflect 5-way contention.** There is no valid
  way to reconstruct an isolated-hardware latency from a number measured under
  load. These are reported as "under concurrent load"; do not try to correct them.
- **Per-process GPU utilisation** (as distinct from GPU memory) needs
  `nvmlDeviceGetProcessUtilization`, which some drivers do not support. If
  unsupported it reports null. Expected, not a bug to fix.

## After the run

Results land in the same per-model folders as before, so `analysis/` scripts are
unchanged. Additionally produced:

- `report-output/ghost_agents/run_logs/resource_timeseries_<model_key>.csv`
- `report-output/ghost_agents/run_logs/run_manifest_<id>.json` — contains
  `pod_wall_seconds`, needed by `resource_monitor.apportioned_cost()` to split the
  real bill across models by measured usage.

See README Section 8 for the full reference.
