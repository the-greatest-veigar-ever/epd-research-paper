#!/usr/bin/env python3
"""
Empirically calibrate OLLAMA_NUM_PARALLEL / evaluator wave size per model.

Decode is memory-bandwidth-bound, so the right batch size for a model isn't
derivable from its size alone -- it depends on where GPU *compute* actually
saturates, which only measurement can answer. This sweeps candidate N values
for one model, starting a fresh `ollama serve` per candidate (OLLAMA_NUM_
PARALLEL is read at server startup), firing exactly N concurrent real
generate calls, and recording wall time, per-call latency spread, peak VRAM,
and whether anything failed.

Cost: a handful of ~10-30s waves per model -- a few pod-minutes total,
negligible against the real sweep.

Usage:
    python3 calibrate_batch_size.py --model phi3:mini
    python3 calibrate_batch_size.py --model gpt-oss:20b --candidates 1 2 4 8
    python3 calibrate_batch_size.py --model deepseek-r1:1.5b --vram-budget-gb 60
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from src.ghost_agents.approach_evaluation.per_process_monitor import (
    nvml_init, pynvml, _nvml_state,
)
from src.ghost_agents.approach_evaluation.approaches import (
    ABLATION_MODELS, MODEL_RAM_GB, GENERATION_TEMPERATURE, GENERATION_NUM_CTX,
    GENERATION_TIMEOUT_S, num_predict_for,
)
from src.ghost_agents.approach_evaluation.benchmark_test_data import load_all_benchmarks

DEFAULT_CANDIDATES = [1, 2, 4, 8]
EFFICIENCY_FLOOR = 0.5  # stop when speedup-per-unit-N drops below this


def _start_server(model: str, port: int, num_parallel: int, ollama_bin: str,
                   models_dir: Optional[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    # Match the real run's server config (see run_concurrent_experiment.py's
    # _server_env) -- left unset, Ollama guesses a default context from the
    # whole card's VRAM, which a calibration run measures under too but a
    # multi-server real run does not get, since it has no idea other
    # servers share the card.
    env["OLLAMA_CONTEXT_LENGTH"] = os.environ.get("EPD_NUM_CTX", "8192")
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir
    return subprocess.Popen(
        [ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )


def _wait_healthy(base: str, server: subprocess.Popen, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            return False
        try:
            if requests.get(f"{base}/api/tags", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _stop_server(server: subprocess.Popen) -> None:
    try:
        import psutil
        parent = psutil.Process(server.pid)
        procs = [parent] + parent.children(recursive=True)
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=15)
    except Exception:
        try:
            server.kill()
        except Exception:
            pass


def _call(base: str, model: str, prompt: str, seed: int) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{base}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GENERATION_TEMPERATURE,
                "num_predict": num_predict_for(model),
                "num_ctx": GENERATION_NUM_CTX,
                "seed": seed,
            },
        }, timeout=GENERATION_TIMEOUT_S)
        # "ok" here means the server round-tripped successfully -- that's
        # the concurrency/throughput/VRAM signal this script cares about.
        # An empty response text (done_reason == "length", the model spent
        # its whole num_predict budget reasoning without concluding) is a
        # content/prompt-difficulty phenomenon the real pipeline already
        # classifies separately as truncated/length_capped -- it says
        # nothing about whether N-way batching is healthy, so it must not
        # trip the calibration stopping rule the way a real HTTP/timeout
        # failure should.
        ok = resp.status_code == 200
        body = resp.json() if ok else {}
        return {"ok": ok, "latency_s": time.perf_counter() - t0,
                "error": None if ok else f"http {resp.status_code}",
                "empty_response": ok and not body.get("response")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": str(e)}


class _VramPoller:
    """Polls NVML total used memory (device-wide -- fine here since this
    script runs one model alone, not the concurrent topology) every
    interval_s while active, tracking the peak."""

    def __init__(self, handles, interval_s: float = 0.2):
        self.handles = handles
        self.interval_s = interval_s
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                used = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles)
                self.peak_gb = max(self.peak_gb, used / (1024 ** 3))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "_VramPoller":
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_gb


def _vram_budget_gb(model: str, total_vram_gb: float, explicit: Optional[float]) -> float:
    if explicit is not None:
        return explicit
    # Topology: the 4 SLMs run concurrently with each other; gpt-oss:20b
    # runs solo. Reserve VRAM for whichever other models will actually
    # share the GPU with this one in the real run, so calibration doesn't
    # pick a batch size that only works in isolation.
    if model == "gpt-oss:20b":
        other_gb = 0.0
    else:
        other_gb = sum(
            MODEL_RAM_GB.get(tag, 0.0) for tag in ABLATION_MODELS.values()
            if tag != model and tag != "gpt-oss:20b"
        )
    return total_vram_gb * 0.9 - other_gb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. phi3:mini")
    ap.add_argument("--candidates", type=int, nargs="*", default=DEFAULT_CANDIDATES)
    ap.add_argument("--port", type=int, default=11599)
    ap.add_argument("--ollama-bin", default="ollama")
    ap.add_argument("--models-dir", default=os.environ.get("OLLAMA_MODELS") or None)
    ap.add_argument("--benchmark", default="HarmBench")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vram-budget-gb", type=float, default=None,
                     help="Override the auto-computed VRAM safety margin.")
    ap.add_argument("--skip-prompt-index", type=int, default=0,
                     help="Loaded-case index to exclude from the prompt pool (default 0, "
                          "which for HarmBench/seed 42 is an open-ended 'continue this "
                          "copyrighted passage' prompt that reliably runs every model, "
                          "reasoning or not, all the way to its token cap -- including it "
                          "would make every candidate's measurement chase that one outlier).")
    args = ap.parse_args()

    nvml_ok, nvml_err = nvml_init()
    if not nvml_ok:
        print(f"NVML unavailable ({nvml_err}) -- cannot measure VRAM, aborting.")
        return 1
    handles = _nvml_state["handles"]
    total_vram_gb = sum(pynvml.nvmlDeviceGetMemoryInfo(h).total for h in handles) / (1024 ** 3)
    budget_gb = _vram_budget_gb(args.model, total_vram_gb, args.vram_budget_gb)
    print(f"Calibrating {args.model} -- GPU {total_vram_gb:.1f}GB total, "
          f"VRAM budget for this model: {budget_gb:.1f}GB "
          f"(reserving headroom for whatever else shares the GPU in the real run)\n")

    n_max = max(args.candidates)
    cases = load_all_benchmarks([args.benchmark], max_per_benchmark=n_max + 1,
                                 seed=args.seed).get(args.benchmark, [])
    cases = [c for i, c in enumerate(cases) if i != args.skip_prompt_index][:n_max]
    if len(cases) < n_max:
        print(f"Only {len(cases)} usable case(s) loaded for benchmark {args.benchmark!r}, "
              f"need {n_max}.")
        return 1
    # DISTINCT real prompts, one per slot -- matches how the real sweep
    # actually dispatches (every test case has its own prompt). An earlier
    # version of this script repeated ONE identical prompt across every
    # slot to control for prompt-difficulty variance, but that triggered
    # what looks like server-side request-coalescing/slot-affinity
    # contention on byte-identical concurrent requests -- an artifact of
    # over-controlling the experiment, not a real production behavior.
    prompts = [c["prompt"] for c in cases]
    print(f"Prompt pool: {n_max} distinct real cases from {args.benchmark} "
          f"(skipped index {args.skip_prompt_index})\n")

    base = f"http://127.0.0.1:{args.port}"

    # Step 1: true serial baseline, once, under a dedicated NUM_PARALLEL=1
    # server -- each prompt's own solo cost, unaffected by any candidate's
    # concurrency setting. Reused as the apples-to-apples comparison point
    # for every candidate below, instead of re-deriving a serial estimate
    # from a single (possibly unrepresentative) N=1 measurement per model.
    print("--- serial baseline (NUM_PARALLEL=1, one call at a time) ---")
    baseline_server = _start_server(args.model, args.port, 1, args.ollama_bin, args.models_dir)
    if not _wait_healthy(base, baseline_server):
        print("  baseline server did not become healthy, aborting.")
        _stop_server(baseline_server)
        return 1
    _call(base, args.model, prompts[0], args.seed)  # warm
    serial_latencies = []
    for i, p in enumerate(prompts):
        r = _call(base, args.model, p, args.seed)
        serial_latencies.append(r["latency_s"])
        print(f"  prompt {i}: {r['latency_s']:.1f}s ok={r['ok']} "
              f"empty_response={r.get('empty_response')}")
    _stop_server(baseline_server)
    print()

    rows = []
    chosen_n = None

    for n in sorted(args.candidates):
        print(f"--- N={n} (concurrent) ---")
        server = _start_server(args.model, args.port, n, args.ollama_bin, args.models_dir)
        if not _wait_healthy(base, server):
            print(f"  server did not become healthy for N={n}, skipping")
            _stop_server(server)
            continue

        wave_prompts = prompts[:n]
        _call(base, args.model, wave_prompts[0], args.seed)  # warm

        poller = _VramPoller(handles).start()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(lambda p: _call(base, args.model, p, args.seed), wave_prompts))
        wall_s = time.perf_counter() - t0
        peak_gb = poller.stop()
        _stop_server(server)

        latencies = [r["latency_s"] for r in results]
        any_failed = any(not r["ok"] for r in results)
        empty_count = sum(1 for r in results if r.get("empty_response"))
        throughput = n / wall_s

        # Apples-to-apples: sum of these SAME n prompts' own solo costs,
        # measured once above, vs. how long they take run concurrently --
        # not a naive N * (one candidate's baseline) estimate, which is
        # noisy whenever prompt difficulty (and therefore solo latency)
        # varies across the pool, as it does for reasoning-model thinking
        # length and for open-ended vs. short-answer prompts alike.
        serial_total_s = sum(serial_latencies[:n])
        speedup = serial_total_s / wall_s
        efficiency = speedup / n

        rows.append({
            "n": n, "wall_s": wall_s, "peak_gb": peak_gb, "any_failed": any_failed,
            "throughput": throughput, "efficiency": efficiency,
            "lat_min": min(latencies), "lat_median": sorted(latencies)[len(latencies) // 2],
            "lat_max": max(latencies),
        })
        print(f"  wall={wall_s:.1f}s (serial equivalent {serial_total_s:.1f}s) "
              f"throughput={throughput:.2f}req/s efficiency={efficiency:.2f} "
              f"peak_vram={peak_gb:.1f}GB "
              f"latency[min/med/max]={min(latencies):.1f}/"
              f"{sorted(latencies)[len(latencies)//2]:.1f}/{max(latencies):.1f}s "
              f"failed={any_failed} empty_response={empty_count}/{n}")

        stop_reason = None
        if any_failed:
            stop_reason = "a call failed/timed out at this N"
        elif peak_gb > budget_gb:
            stop_reason = f"peak VRAM {peak_gb:.1f}GB exceeded budget {budget_gb:.1f}GB"
        elif efficiency < EFFICIENCY_FLOOR:
            stop_reason = f"efficiency {efficiency:.2f} dropped below floor {EFFICIENCY_FLOOR}"

        if stop_reason:
            print(f"  stopping: {stop_reason}")
            # Largest N before this one that was clean; fall back to 1.
            clean = [r["n"] for r in rows[:-1] if not r["any_failed"]]
            chosen_n = max(clean) if clean else 1
            break
        chosen_n = n

    print("\n" + "=" * 72)
    print(f" {args.model}: recommended MODEL_BATCH_SIZE = {chosen_n}")
    if args.model == "gpt-oss:20b":
        print(" NOTE: plan recommends capping gpt-oss:20b conservatively below this "
              " plateau (e.g. 4-8) regardless of what calibration allows -- a wide wave "
              " killed mid-flight by the NVML liveness watchdog is the most expensive "
              " one to lose, since gpt-oss already has the fleet's longest per-call "
              " latency. Apply that judgment call manually before hardcoding this value.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
