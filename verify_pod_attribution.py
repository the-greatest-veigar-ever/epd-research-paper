#!/usr/bin/env python3
"""
Cheap pre-flight for per-model resource attribution on a GPU host.

Answers the one question that cannot be answered off the pod: does
per-process GPU attribution actually work *here*? CPU and RAM attribution
are portable and already proven; GPU attribution depends on the driver and
on the container's PID namespace, both of which are properties of the
machine you are about to rent.

Deliberately tiny. It loads one small model, makes ONE generate call, and
inspects what the monitor saw -- about a minute of pod time, rather than
discovering the problem partway through a paid multi-hour sweep. Run this
before the real sweep, not instead of watching the real sweep.

Usage:
    python3 verify_pod_attribution.py
    python3 verify_pod_attribution.py --model phi3:mini --port 11599
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {PASS: "  ok ", FAIL: "FAIL", WARN: "warn"}[status]
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-r1:1.5b",
                    help="Smallest pulled model is best; this only needs to load.")
    ap.add_argument("--port", type=int, default=11599)
    ap.add_argument("--ollama-bin", default="ollama")
    args = ap.parse_args()

    print("Per-model attribution check\n")

    # --- dependencies ----------------------------------------------------
    print("dependencies")
    try:
        import psutil  # noqa: F401
        record("psutil importable", PASS)
    except ImportError:
        record("psutil importable", FAIL, "pip install psutil")
        return 1

    import requests
    from src.ghost_agents.approach_evaluation.per_process_monitor import (
        ProcessTreeRecorder, nvml_init, pynvml, _nvml_state,
    )

    if pynvml is None:
        record("nvidia-ml-py importable", FAIL,
               "pip install nvidia-ml-py  (GPU fields will be null without it)")
    else:
        record("nvidia-ml-py importable", PASS)

    nvml_ok, nvml_err = nvml_init()
    if nvml_ok:
        total = sum(
            pynvml.nvmlDeviceGetMemoryInfo(h).total for h in _nvml_state["handles"]
        ) / (1024 ** 3)
        record("NVML init", PASS,
               f"{len(_nvml_state['handles'])} device(s), {total:.1f}GB total")
    else:
        record("NVML init", FAIL, str(nvml_err))

    # --- server ----------------------------------------------------------
    print("\nollama server")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{args.port}"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env["OLLAMA_KEEP_ALIVE"] = "-1"

    try:
        server = subprocess.Popen(
            [args.ollama_bin, "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
    except FileNotFoundError:
        record("ollama serve", FAIL, f"'{args.ollama_bin}' not on PATH")
        return 1

    base = f"http://127.0.0.1:{args.port}"
    healthy = False
    for _ in range(120):
        if server.poll() is not None:
            break
        try:
            if requests.get(f"{base}/api/tags", timeout=2).status_code == 200:
                healthy = True
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)

    if not healthy:
        record("server healthy", FAIL, f"nothing answered on port {args.port}")
        _cleanup(server)
        return 1
    record("server healthy", PASS, f"pid {server.pid} on port {args.port}")

    tags = {m.get("name", "") for m in requests.get(f"{base}/api/tags").json().get("models", [])}
    if args.model not in tags:
        record("model pulled", FAIL,
               f"{args.model} not present. Have: {', '.join(sorted(tags)) or '(none)'}")
        _cleanup(server)
        return 1
    record("model pulled", PASS, args.model)

    # --- one real call, watched -----------------------------------------
    print("\nattribution during one generate call")
    rec = ProcessTreeRecorder(server.pid, "verify", interval_s=0.5).start()
    t0 = time.perf_counter()
    try:
        requests.post(f"{base}/api/generate", json={
            "model": args.model,
            "prompt": "Name three common web vulnerabilities.",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        }, timeout=300)
    except Exception as e:
        record("generate call", FAIL, str(e))
        rec.stop()
        _cleanup(server)
        return 1
    t1 = time.perf_counter()
    time.sleep(1.0)  # let a couple more samples land
    stats = rec.window_stats(t0, t1)
    diag = rec.diagnostics()
    rec.stop()

    record("generate call", PASS, f"{t1 - t0:.1f}s")

    if stats.get("sample_count", 0) > 0:
        record("monitor sampled the call", PASS, f"{stats['sample_count']} sample(s)")
    else:
        record("monitor sampled the call", FAIL, "no samples inside the call window")

    if (stats.get("cpu_core_seconds") or 0) > 0:
        record("CPU attributed", PASS,
               f"{stats['cpu_core_seconds']:.2f} cpu-seconds, "
               f"{stats['cpu_percent_cores_avg']:.0f}% of one core")
    else:
        record("CPU attributed", FAIL, "no CPU time attributed to the server tree")

    if (stats.get("ram_used_gb_avg") or 0) > 0:
        record("RAM attributed", PASS, f"{stats['ram_used_gb_avg']:.2f} GB")
    else:
        record("RAM attributed", FAIL, "no RSS attributed to the server tree")

    # The whole reason this script exists.
    if not nvml_ok:
        record("GPU memory attributed", FAIL, "NVML unavailable (see above)")
    elif diag.get("gpu_pid_mismatch"):
        record("GPU memory attributed", FAIL,
               "NVML sees compute processes but none match our PIDs -- container PID "
               "namespace. Re-run the container with --pid=host to fix.")
    elif stats.get("gpu_mem_used_gb_avg"):
        record("GPU memory attributed", PASS,
               f"{stats['gpu_mem_used_gb_avg']:.2f} GB on "
               f"{diag['nvml_compute_procs_matched']} matched process(es)")
    else:
        record("GPU memory attributed", WARN,
               "no GPU memory seen -- is the model running on CPU? check `nvidia-smi`")

    util = diag.get("gpu_per_process_util_supported")
    if util is True and stats.get("gpu_percent_avg") is not None:
        record("GPU utilisation attributed", PASS, f"{stats['gpu_percent_avg']}%")
    elif util is False:
        record("GPU utilisation attributed", WARN,
               "driver does not support per-process utilisation; memory still works, "
               "utilisation will report null")
    else:
        record("GPU utilisation attributed", WARN,
               "no utilisation samples in this short window; usually fine on a real run")

    _cleanup(server)

    # --- verdict ---------------------------------------------------------
    failures = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print("\n" + "=" * 64)
    if failures:
        print(f" {len(failures)} CHECK(S) FAILED -- fix before the paid sweep:")
        for name, _, detail in failures:
            print(f"   - {name}: {detail}")
    else:
        print(" All required checks passed.")
        if warns:
            print(f" {len(warns)} warning(s) -- degraded but usable:")
            for name, _, detail in warns:
                print(f"   - {name}: {detail}")
        print("\n Next: python3 run_concurrent_experiment.py --pod-hourly-usd <rate>")
    print("=" * 64)
    return 1 if failures else 0


def _cleanup(server) -> None:
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


if __name__ == "__main__":
    sys.exit(main())
