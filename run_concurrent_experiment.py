#!/usr/bin/env python3
"""
Concurrent EPD sweep with per-model resource attribution.

What this replaces, and why
---------------------------
`run_runpod_experiment.sh` runs the five SLMs strictly one at a time. Not
for memory reasons -- together they need roughly 21GB, trivial on an A100
-- but for measurement ones: `psutil` and `nvidia-smi` sample the machine,
so any concurrency turns every CPU/RAM/GPU figure into an average across
whichever models happened to be running. Sequential execution is the only
way to keep those numbers attributable, and it costs about five times the
pod-hours.

This runner removes that trade-off rather than accepting either side of
it. Resource usage is attributed per OS process (see
per_process_monitor), so the models can run together and still report
separate figures, and one pod-hour does the work of five.

The topology, and the invariant it protects
-------------------------------------------
Each model gets:

  * its own `ollama serve`, on its own port, for that model only, ever;
  * its own evaluator process, pointed at that port;
  * its own monitor, anchored on that server's PID.

The invariant is that a given Ollama server hosts exactly one model for
its entire lifetime. It is what makes the resource figures mean anything,
and the ablation is what puts it under pressure: every *ephemeral* cell
unloads and reloads its model on every call, so the process actually
holding the weights is destroyed and recreated constantly, with a new PID
each time. Anchoring on a PID would therefore lose the model within one
call and, once the kernel recycled that number, start attributing some
other process's usage to it. Anchoring on the server survives that
churn -- the reloads appear and disappear as children underneath it --
and because nothing else is ever routed to that server, a new child under
it is unambiguously this model's reload.

The same invariant is why slots are never recycled. Even with
`--max-parallel` below the model count, a queued model gets a freshly
started server on its own port rather than inheriting a finished model's;
a server is never handed from one model to another.

Usage:
    python3 run_concurrent_experiment.py                    # all 5 SLMs
    python3 run_concurrent_experiment.py --models phi3_mini qwen25_3b
    python3 run_concurrent_experiment.py --max-parallel 2
    python3 run_concurrent_experiment.py --pod-hourly-usd 1.89

Results land in the same per-model folders as the sequential runner, so
the analysis scripts are unchanged. Each model also gets a resource
time-series CSV, and the run writes a manifest recording the pod's billed
wall-clock, which is what `resource_monitor.apportioned_cost` needs to
split one real bill across the models by measured usage.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ghost_agents.approach_evaluation.approaches import (  # noqa: E402
    ABLATION_MODELS,
    MODEL_RAM_GB,
    batch_size_for,
)

DEFAULT_BASE_PORT = 11500
DEFAULT_OUTPUT_DIR = "report-output/ghost_agents/benchmark_results"
DEFAULT_LOG_DIR = "report-output/ghost_agents/run_logs"

# Generation limits, kept identical to run_runpod_experiment.sh so the two
# runners remain comparable -- these are documented experimental parameters
# that must match the paper's configuration table, not runner preferences.
GENERATION_ENV = {
    "EPD_TEMPERATURE": "0.0",
    "EPD_NUM_PREDICT": "1024",
    "EPD_NUM_CTX": "8192",
    "EPD_GENERATE_TIMEOUT": "300",
    "EPD_PRELOAD_TIMEOUT": "600",
}


class ModelRun:
    """One model's dedicated server, evaluator process and monitor CSV."""

    def __init__(self, key: str, tag: str, port: int, log_dir: str):
        self.key = key
        self.tag = tag
        self.port = port
        self.server: Optional[subprocess.Popen] = None
        self.evaluator: Optional[subprocess.Popen] = None
        self.server_log = os.path.join(log_dir, f"ollama_{key}.log")
        self.eval_log = os.path.join(log_dir, f"{key}.log")
        self.csv_path = os.path.join(log_dir, f"resource_timeseries_{key}.csv")
        self.exit_code: Optional[int] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._server_log_fh = None
        self._eval_log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def _terminate_tree(proc: Optional[subprocess.Popen], timeout: float = 15.0) -> None:
    """
    Stop a process and everything under it.

    Children are collected *before* the parent is signalled: once the
    parent dies its children are reparented and can no longer be found
    from it, which on an Ollama server would leave the runner subprocess
    alive and still holding VRAM the next model is about to need.
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    for p in [parent] + children:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs([parent] + children, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _wait_healthy(run: ModelRun, timeout: float = 120.0) -> bool:
    """Poll a freshly started server until it answers, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if run.server is not None and run.server.poll() is not None:
            return False  # died during start-up; the log has why
        try:
            r = requests.get(f"{run.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _server_env(run: ModelRun, models_dir: Optional[str], keep_alive: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{run.port}"
    # One model per server -- this, not OLLAMA_NUM_PARALLEL below, is what
    # keeps this server's process tree equal to this model's process tree:
    # per_process_monitor anchors on the server PID and sums the ENTIRE
    # live descendant tree under it every sample (recursive, re-resolved
    # per tick), so it is correctly attributed to this model regardless of
    # how many requests that one loaded model is serving concurrently. What
    # would break attribution is a SECOND model sharing this server, which
    # OLLAMA_MAX_LOADED_MODELS=1 rules out.
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    # Batched concurrency for this model's calibrated throughput (see
    # approaches.batch_size_for / MODEL_BATCH_SIZE). Must stay in sync with
    # the evaluator's own wave size, which calls the same function -- that
    # is the single source of truth keeping the two from drifting apart.
    env["OLLAMA_NUM_PARALLEL"] = str(batch_size_for(run.tag))
    # Ollama unloads an idle model after 5 minutes by default, which would
    # quietly turn a *static* cell -- defined by its model staying resident
    # -- into an ephemeral one during any gap between calls, corrupting the
    # ablation's central contrast. A per-request keep_alive still overrides
    # this, so the ephemeral cells' explicit keep_alive:0 teardown is
    # unaffected.
    env["OLLAMA_KEEP_ALIVE"] = keep_alive
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir
    return env


def _start_server(run: ModelRun, ollama_bin: str, models_dir: Optional[str],
                  keep_alive: str) -> bool:
    os.makedirs(os.path.dirname(run.server_log) or ".", exist_ok=True)
    run._server_log_fh = open(run.server_log, "w", encoding="utf-8")
    run.server = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=run._server_log_fh,
        stderr=subprocess.STDOUT,
        env=_server_env(run, models_dir, keep_alive),
    )
    print(f"  [{run.key}] ollama serve pid={run.server.pid} port={run.port} "
          f"num_parallel={batch_size_for(run.tag)}")
    if not _wait_healthy(run):
        print(f"  [{run.key}] ERROR: server did not become healthy -- see {run.server_log}")
        return False
    return True


def _tag_present(tag: str, present: set) -> bool:
    """
    Exact tag match only.

    Matching on the family name would accept phi3:medium for phi3:mini --
    a different model, silently evaluated under the other one's name and
    its calibrated generation cap. Every tag in the registry is explicit,
    so there is nothing to be gained by matching loosely and a whole
    corrupted model-row to lose.
    """
    return tag in present


def _model_present(run: ModelRun) -> bool:
    try:
        r = requests.get(f"{run.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        tags = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return False
    return _tag_present(run.tag, tags)


def _start_evaluator(run: ModelRun, args, concurrency: int) -> None:
    env = os.environ.copy()
    env.update(GENERATION_ENV)
    # Route every Ollama call from this process -- generation and the
    # preload/unload lifecycle alike -- to this model's own server.
    env["EPD_OLLAMA_PORT"] = str(run.port)
    # The exact anchor for attribution. Passed explicitly rather than
    # discovered from the port, because we started this server and know its
    # PID for certain; discovery is only the fallback for a hand-started one.
    env["EPD_MONITOR_ROOT_PID"] = str(run.server.pid)
    env["EPD_MONITOR_LABEL"] = run.key
    env["EPD_MONITOR_CSV"] = run.csv_path
    # Sanity ceiling for the per-process GPU memory guard (see
    # per_process_monitor.ProcessTreeRecorder's expected_weight_gb note) --
    # this model's own known weight size, the one signal available here
    # that lets the guard distinguish "implausible" from "legitimately
    # large" without hardcoding a model-specific threshold in that module.
    env["EPD_MONITOR_EXPECTED_GB"] = str(MODEL_RAM_GB.get(run.tag, 0.0))
    env["EPD_MONITOR_INTERVAL"] = str(args.monitor_interval)
    env["EPD_POD_CONCURRENCY"] = str(concurrency)
    if args.pod_hourly_usd is not None:
        env["EPD_POD_HOURLY_USD"] = str(args.pod_hourly_usd)

    cmd = [
        sys.executable, "-u", "-m",
        "src.ghost_agents.approach_evaluation.benchmark_evaluator",
        "--approaches", run.key,
        "--seeds", *[str(s) for s in args.seeds],
        "--max-per-benchmark", str(args.max_per_benchmark),
        "--save-every", str(args.save_every),
        "--output", args.output,
    ]
    if args.benchmarks:
        cmd.extend(["--benchmarks", *args.benchmarks])
    if getattr(args, "verbose", False):
        cmd.append("--verbose")

    os.makedirs(os.path.dirname(run.eval_log) or ".", exist_ok=True)
    run._eval_log_fh = open(run.eval_log, "w", encoding="utf-8")
    run.evaluator = subprocess.Popen(
        cmd, stdout=run._eval_log_fh, stderr=subprocess.STDOUT, env=env
    )
    run.started_at = time.time()
    print(f"  [{run.key}] evaluator pid={run.evaluator.pid} -> {run.eval_log}")


def _finish(run: ModelRun) -> None:
    if run.evaluator is not None:
        try:
            # wait() rather than poll(): on the interrupted path the process
            # was only just signalled, and poll() would report None (i.e.
            # "still running") for a run we are about to record as finished.
            run.exit_code = run.evaluator.wait(timeout=10)
        except subprocess.TimeoutExpired:
            run.exit_code = run.evaluator.poll()
    run.finished_at = time.time()
    _terminate_tree(run.server)
    for fh in (run._eval_log_fh, run._server_log_fh):
        if fh:
            try:
                fh.close()
            except Exception:
                pass
    run._eval_log_fh = run._server_log_fh = None
    status = "OK" if run.exit_code == 0 else f"FAILED (exit {run.exit_code})"
    elapsed = (run.finished_at - (run.started_at or run.finished_at)) / 60.0
    print(f"  [{run.key}] {status} after {elapsed:.1f} min")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _gpu_total_gb() -> Optional[float]:
    try:
        from src.ghost_agents.approach_evaluation.per_process_monitor import (
            nvml_init, _nvml_state, pynvml,
        )
        ok, _ = nvml_init()
        if not ok or pynvml is None:
            return None
        total = 0
        for handle in _nvml_state["handles"]:
            total += pynvml.nvmlDeviceGetMemoryInfo(handle).total
        return total / (1024 ** 3)
    except Exception:
        return None


def _gpu_still_alive() -> Tuple[bool, Optional[str]]:
    try:
        from src.ghost_agents.approach_evaluation.per_process_monitor import nvml_gpu_alive
        return nvml_gpu_alive()
    except Exception as e:
        return False, str(e)


def _stop_pod(reason: str) -> None:
    """
    Stop this RunPod pod via runpodctl -- same tool auto_shutdown.sh already
    uses for the sequential runner. Only called when the caller has decided
    the *whole* run is broken (GPU gone, or every model failed), never for
    one model failing among several that are still doing useful work.

    Best-effort: this only runs after the guard that called it has already
    printed the failure and written the manifest, so a failure here (no
    RUNPOD_POD_ID, runpodctl missing, API hiccup) is reported but must not
    raise -- the run has already ended either way.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print(f"  [pod-stop] RUNPOD_POD_ID not set (not on a RunPod pod?) -- not stopping anything.")
        return
    print(f"  [pod-stop] {reason} -- stopping pod {pod_id} via runpodctl.")
    try:
        subprocess.run(["runpodctl", "stop", "pod", pod_id],
                        check=True, timeout=60,
                        capture_output=True, text=True)
        print(f"  [pod-stop] pod {pod_id} stop requested.")
    except Exception as e:
        print(f"  [pod-stop] FAILED to stop pod {pod_id}: {e}")


def _preflight(runs: List[ModelRun], args) -> bool:
    print("Preflight")

    if shutil.which(args.ollama_bin) is None:
        print(f"  ERROR: '{args.ollama_bin}' not found on PATH.")
        return False
    print(f"  ollama binary: {shutil.which(args.ollama_bin)}")

    # A concurrent run means every model's weights are resident at once, so
    # the VRAM budget is the sum -- not the max, as it was when they ran one
    # at a time. Getting this wrong surfaces as a mid-run CUDA OOM hours in.
    needed = sum(MODEL_RAM_GB.get(r.tag, 0.0) for r in runs)
    total = _gpu_total_gb()
    parallel = min(args.max_parallel, len(runs))
    concurrent_need = needed * parallel / len(runs) if runs else 0.0
    if total is None:
        print(f"  GPU: not detectable from here; models need ~{needed:.1f}GB combined.")
        print("       Per-process GPU attribution will report N/A unless NVML is")
        print("       available on the run host (pip install nvidia-ml-py).")
    else:
        print(f"  GPU: {total:.1f}GB total, ~{concurrent_need:.1f}GB needed "
              f"at {parallel}-way concurrency")
        if concurrent_need > total * 0.9:
            print(f"  ERROR: {concurrent_need:.1f}GB exceeds 90% of {total:.1f}GB. "
                  f"Lower --max-parallel or drop a model.")
            return False

    ports = [r.port for r in runs]
    busy = []
    for run in runs:
        try:
            requests.get(f"{run.base_url}/api/tags", timeout=1)
            busy.append(run.port)
        except requests.RequestException:
            pass
    if busy:
        print(f"  ERROR: something is already listening on port(s) {busy}. "
              f"Pick a different --base-port, or stop it.")
        return False
    print(f"  ports {min(ports)}-{max(ports)} free")
    return True


def _verify_tags(runs: List[ModelRun], args) -> bool:
    """
    Confirm every model tag is pulled, using one throwaway server.

    `ollama list` needs a server to talk to, and none of the per-model ones
    exist yet. Checking up front matters because the alternative is
    discovering a missing tag after the other four models have been running
    for hours -- every call against the absent model would fail, be
    correctly excluded from ASR/TSR, and leave that model with nothing.
    """
    # Reuse the first model's port, which preflight has just confirmed is
    # free. An arbitrary "probably unused" port risks landing on someone
    # else's Ollama, whose health check would pass and whose model list we
    # would then trust.
    probe = ModelRun("preflight", "", runs[0].port, args.log_dir)
    if not _start_server(probe, args.ollama_bin, args.models_dir, args.keep_alive):
        _terminate_tree(probe.server)
        return False

    try:
        r = requests.get(f"{probe.base_url}/api/tags", timeout=15)
        r.raise_for_status()
        present = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception as e:
        print(f"  ERROR: could not list models: {e}")
        return False
    finally:
        _terminate_tree(probe.server)
        if probe._server_log_fh:
            probe._server_log_fh.close()

    missing = [run.tag for run in runs if not _tag_present(run.tag, present)]
    if missing:
        print(f"  ERROR: model(s) not pulled: {', '.join(missing)}")
        print("  Pull them first:")
        for tag in missing:
            print(f"      ollama pull {tag}")
        return False
    print(f"  all {len(runs)} model tag(s) present")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the EPD ablation sweep concurrently with per-model "
                    "resource attribution.",
    )
    parser.add_argument("--models", nargs="*", default=list(ABLATION_MODELS.keys()),
                        help=f"Model keys to run (default: all "
                             f"{len(ABLATION_MODELS)} SLMs).")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43])
    parser.add_argument("--max-per-benchmark", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="Models running at once (default: all of them).")
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT,
                        help=f"First port to use (default: {DEFAULT_BASE_PORT}). "
                             f"Each model gets base-port + its index.")
    parser.add_argument("--ollama-bin", default="ollama")
    parser.add_argument("--models-dir", default=os.environ.get("OLLAMA_MODELS") or None,
                        help="Shared Ollama model store. All servers read the same "
                             "blobs; pull models before starting, not during.")
    parser.add_argument("--keep-alive", default="-1",
                        help="OLLAMA_KEEP_ALIVE for every server (default: -1, "
                             "keep loaded indefinitely, so 'static' cells really "
                             "do stay resident between calls).")
    parser.add_argument("--monitor-interval", type=float, default=0.5,
                        help="Resource sampling cadence in seconds (default: 0.5).")
    parser.add_argument("--pod-hourly-usd", type=float, default=None,
                        help="Real hourly rate of this machine. Set it and costs "
                             "are computed as one shared bill instead of pretending "
                             "each model rented its own instance.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run preflight and print the plan, then stop.")
    parser.add_argument("--verbose", action="store_true",
                        help="Forwarded to every evaluator subprocess -- per-wave dispatch "
                             "lines, guard warnings, and response/error previews.")
    parser.add_argument("--stop-pod-on-failure", action="store_true",
                        help="Stop this RunPod pod (via runpodctl) when the whole run "
                             "is judged broken: the GPU vanishes mid-run, or every "
                             "model ends up failed. Never triggered by a single model "
                             "failing among others that are still running. Off by "
                             "default so a routine or test invocation can't stop the "
                             "pod by surprise.")
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in ABLATION_MODELS]
    if unknown:
        print(f"ERROR: unknown model key(s): {unknown}")
        print(f"Valid keys: {', '.join(ABLATION_MODELS)}")
        return 1

    os.makedirs(args.log_dir, exist_ok=True)
    # Port is derived from each model's index in the *full* registry, not
    # its position in --models, so a given model always lands on the same
    # port across runs and two models can never collide on one.
    all_keys = list(ABLATION_MODELS.keys())
    runs = [
        ModelRun(key, ABLATION_MODELS[key], args.base_port + all_keys.index(key),
                 args.log_dir)
        for key in args.models
    ]
    if args.max_parallel <= 0:
        args.max_parallel = len(runs)
    parallel = min(args.max_parallel, len(runs))

    print("=" * 72)
    print(f" Concurrent EPD sweep -- {len(runs)} model(s), {parallel} at a time")
    print(f" Models:   {', '.join(r.key for r in runs)}")
    print(f" Seeds:    {args.seeds}   Samples/benchmark: {args.max_per_benchmark}")
    print(f" Attribution: per-process (each model gets its own ollama serve)")
    print("=" * 72)

    if not _preflight(runs, args):
        return 1
    if not _verify_tags(runs, args):
        return 1

    if args.dry_run:
        print("\nDry run -- plan:")
        for r in runs:
            print(f"  {r.key:<20} tag={r.tag:<20} port={r.port}  csv={r.csv_path}")
        return 0

    pod_started = time.time()
    queue = list(runs)
    active: List[ModelRun] = []
    completed: List[ModelRun] = []
    interrupted = False
    interrupted_reason: Optional[str] = None
    gpu_vanished = False

    def _shutdown(signum, _frame):
        nonlocal interrupted, interrupted_reason
        interrupted = True
        interrupted_reason = f"signal {signum}"
        print(f"\nSignal {signum} -- stopping everything.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            pass

    # Only watch for a GPU that was here and vanished -- a pod with no GPU or
    # no NVML never had a signal to lose, and _preflight already told the
    # user their attribution will be null.
    gpu_watchdog = _gpu_total_gb() is not None

    try:
        print("\nStarting")
        while (queue or active) and not interrupted:
            while queue and len(active) < parallel:
                run = queue.pop(0)
                if not _start_server(run, args.ollama_bin, args.models_dir,
                                     args.keep_alive):
                    run.exit_code = 1
                    completed.append(run)
                    continue
                if not _model_present(run):
                    print(f"  [{run.key}] ERROR: {run.tag} not available on its server")
                    _terminate_tree(run.server)
                    run.exit_code = 1
                    completed.append(run)
                    continue
                _start_evaluator(run, args, parallel)
                active.append(run)

            # Liveness poll, not just an init check: nvml_gpu_alive() makes a
            # live per-device call every pass, so a mid-run device-cgroup
            # revocation (host driver and /proc still say "healthy", every
            # actual device open returns EPERM) is caught here within one
            # ~2s cycle instead of each model's Ollama server silently
            # falling back to CPU and every call riding out the full
            # generate timeout for no result.
            if gpu_watchdog and active:
                alive, err = _gpu_still_alive()
                if not alive:
                    interrupted = True
                    gpu_vanished = True
                    interrupted_reason = f"GPU vanished: {err}"
                    print(f"\nGPU liveness check failed -- {err}")
                    print("Matches the 2026-08-25 device-cgroup-revocation signature: "
                          "the GPU was present at start and is no longer reachable. "
                          "Stopping every model now rather than letting them run on "
                          "silently for hours on a dead GPU.")
                    break

            time.sleep(2.0)

            for run in list(active):
                if run.evaluator is not None and run.evaluator.poll() is not None:
                    _finish(run)
                    active.remove(run)
                    completed.append(run)
    finally:
        for run in active:
            _terminate_tree(run.evaluator)
            _finish(run)
            completed.append(run)
        pod_finished = time.time()

    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "topology": "concurrent_per_model_server",
        "resource_attribution": "per_process",
        # The pod's billed wall-clock: first model starting to last finishing.
        # NOT the sum of per-model runtimes, which under concurrency counts
        # the same rented seconds once per model. This is the number
        # resource_monitor.apportioned_cost must be given.
        "pod_wall_seconds": round(pod_finished - pod_started, 2),
        "pod_hourly_usd": args.pod_hourly_usd,
        "concurrency": parallel,
        "interrupted": interrupted,
        "interrupted_reason": interrupted_reason,
        "seeds": args.seeds,
        "max_per_benchmark": args.max_per_benchmark,
        "models": [
            {
                "model_key": r.key, "tag": r.tag, "port": r.port,
                "exit_code": r.exit_code,
                "ollama_num_parallel": batch_size_for(r.tag),
                "wall_seconds": (
                    round(r.finished_at - r.started_at, 2)
                    if r.started_at and r.finished_at else None
                ),
                "resource_csv": r.csv_path,
                "eval_log": r.eval_log,
            }
            for r in completed
        ],
    }
    manifest_path = os.path.join(args.log_dir, f"run_manifest_{manifest['run_id']}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    failed = [r for r in completed if r.exit_code != 0]
    hours = manifest["pod_wall_seconds"] / 3600.0
    print("\n" + "=" * 72)
    print(f" Finished in {hours:.2f} pod-hours "
          f"({len(completed) - len(failed)}/{len(completed)} model(s) OK)")
    if args.pod_hourly_usd:
        print(f" Pod cost: ~${args.pod_hourly_usd * hours:.2f} "
              f"(vs ~${args.pod_hourly_usd * hours * parallel:.2f} run sequentially)")
    print(f" Manifest: {manifest_path}")
    if failed:
        print(f" FAILED: {', '.join(r.key for r in failed)} -- see their logs")
    print("=" * 72)

    if args.stop_pod_on_failure:
        if gpu_vanished:
            _stop_pod("GPU vanished mid-run")
        elif completed and len(failed) == len(completed):
            _stop_pod(f"every model failed ({len(failed)}/{len(completed)})")

    return 1 if failed or interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
