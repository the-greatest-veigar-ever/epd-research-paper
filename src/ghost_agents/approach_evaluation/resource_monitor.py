"""
Runtime Resource Monitor.

Addresses reviewer feedback that the efficiency analysis relied on static
memory footprint alone. This module samples CPU, RAM, and (when available)
GPU utilization for the duration of a single model call, and provides a
cost estimate derived from a documented, editable hardware-rate table.

GPU utilization is sampled via `nvidia-smi` when it's present on PATH
(e.g. a CUDA host such as a RunPod pod). On platforms without an NVIDIA
GPU (e.g. Apple Silicon), `gpu_available` is False and the GPU fields are
reported as N/A rather than fabricated.
"""

import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

GPU_MONITORING_AVAILABLE = shutil.which("nvidia-smi") is not None


def _sample_gpu() -> Optional[Dict[str, float]]:
    """
    Snapshots GPU utilization (%) and memory used (GB) via `nvidia-smi`,
    averaged/summed across all visible GPUs. Returns None if unavailable
    or the query fails (e.g. a transient driver hiccup) -- callers must
    not treat a missing sample as 0% utilization.
    """
    if not GPU_MONITORING_AVAILABLE:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        rows = [line.split(",") for line in out.stdout.strip().splitlines() if line.strip()]
        if not rows:
            return None
        utils = [float(r[0]) for r in rows]
        mem_used_mb = [float(r[1]) for r in rows]
        return {
            "percent": sum(utils) / len(utils),
            "mem_used_gb": sum(mem_used_mb) / 1024.0,
        }
    except Exception:
        return None


@dataclass
class ResourceSample:
    t: float
    cpu_percent: float
    ram_used_gb: float
    gpu_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None


@dataclass
class ResourceStats:
    """Aggregated resource usage over the sampled window."""

    duration_s: float = 0.0
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    ram_used_gb_avg: float = 0.0
    ram_used_gb_max: float = 0.0
    gpu_available: bool = GPU_MONITORING_AVAILABLE
    gpu_percent_avg: Optional[float] = None
    gpu_percent_max: Optional[float] = None
    gpu_mem_used_gb_avg: Optional[float] = None
    gpu_mem_used_gb_max: Optional[float] = None
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            # Machine-wide readings: valid only while exactly one model is
            # resident and one process is active. Under the concurrent
            # topology this is replaced by per_process_monitor's
            # "per_process" attribution -- the tag lets any reader tell the
            # two apart without having to know which run produced the file.
            "attribution": "machine_wide",
            "duration_s": round(self.duration_s, 4),
            "cpu_percent_avg": round(self.cpu_percent_avg, 2),
            "cpu_percent_max": round(self.cpu_percent_max, 2),
            "ram_used_gb_avg": round(self.ram_used_gb_avg, 3),
            "ram_used_gb_max": round(self.ram_used_gb_max, 3),
            "gpu_percent_avg": round(self.gpu_percent_avg, 2) if self.gpu_percent_avg is not None else None,
            "gpu_percent_max": round(self.gpu_percent_max, 2) if self.gpu_percent_max is not None else None,
            "gpu_mem_used_gb_avg": (
                round(self.gpu_mem_used_gb_avg, 3) if self.gpu_mem_used_gb_avg is not None else None
            ),
            "gpu_mem_used_gb_max": (
                round(self.gpu_mem_used_gb_max, 3) if self.gpu_mem_used_gb_max is not None else None
            ),
            "gpu_note": (
                None
                if self.gpu_available
                else "GPU utilization unavailable on this platform (no nvidia-smi); "
                "reported as N/A rather than estimated."
            ),
            "platform": self.platform,
            "sample_count": self.sample_count,
        }


class ResourceMonitor:
    """
    Context manager that samples CPU/RAM at a fixed interval on a background
    thread for the duration of the `with` block, then exposes aggregated
    stats via `.stats`.

    Usage:
        with ResourceMonitor() as mon:
            do_work()
        stats = mon.stats
    """

    def __init__(self, interval_s: float = 0.2, gpu_interval_s: float = 1.0):
        # GPU sampling is deliberately slower than CPU/RAM sampling. Each GPU
        # sample shells out to `nvidia-smi` (a process spawn, ~10-50ms); at the
        # 0.2s CPU cadence that was ~450 spawns during a 90s call, and their
        # CPU cost landed in the machine-wide cpu_percent this very monitor is
        # recording -- the instrument was measuring itself. 1s keeps GPU
        # utilization/memory well sampled (dozens of points per call) at a
        # fifth of the overhead. The first tick always samples, so even a
        # sub-second call still gets a GPU reading.
        self.interval_s = interval_s
        self.gpu_interval_s = max(gpu_interval_s, interval_s)
        self._samples: List[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start = 0.0
        self._t_end = 0.0
        self.stats: ResourceStats = ResourceStats()

    def _sample_loop(self):
        # Prime psutil's internal CPU counter so the first real reading isn't 0.0.
        psutil.cpu_percent(interval=None)
        next_gpu_t = 0.0  # 0 => the first iteration always takes a GPU sample
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
            now = time.perf_counter()
            if now >= next_gpu_t:
                gpu = _sample_gpu()
                next_gpu_t = now + self.gpu_interval_s
            else:
                # Not due yet. Recorded as None rather than carrying the last
                # reading forward -- the aggregate averages only real samples,
                # so a repeated value would fake precision it does not have.
                gpu = None
            self._samples.append(
                ResourceSample(
                    t=time.perf_counter(),
                    cpu_percent=cpu,
                    ram_used_gb=ram_used_gb,
                    gpu_percent=gpu["percent"] if gpu else None,
                    gpu_mem_used_gb=gpu["mem_used_gb"] if gpu else None,
                )
            )
            self._stop_event.wait(self.interval_s)

    def __enter__(self) -> "ResourceMonitor":
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._t_end = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

        duration = self._t_end - self._t_start
        if self._samples:
            cpu_values = [s.cpu_percent for s in self._samples]
            ram_values = [s.ram_used_gb for s in self._samples]
            gpu_values = [s.gpu_percent for s in self._samples if s.gpu_percent is not None]
            gpu_mem_values = [s.gpu_mem_used_gb for s in self._samples if s.gpu_mem_used_gb is not None]
            self.stats = ResourceStats(
                duration_s=duration,
                cpu_percent_avg=sum(cpu_values) / len(cpu_values),
                cpu_percent_max=max(cpu_values),
                ram_used_gb_avg=sum(ram_values) / len(ram_values),
                ram_used_gb_max=max(ram_values),
                gpu_percent_avg=sum(gpu_values) / len(gpu_values) if gpu_values else None,
                gpu_percent_max=max(gpu_values) if gpu_values else None,
                gpu_mem_used_gb_avg=sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None,
                gpu_mem_used_gb_max=max(gpu_mem_values) if gpu_mem_values else None,
                sample_count=len(self._samples),
            )
        else:
            # Call was faster than one sampling interval; take a single point reading.
            gpu = _sample_gpu()
            self.stats = ResourceStats(
                duration_s=duration,
                cpu_percent_avg=psutil.cpu_percent(interval=None),
                cpu_percent_max=psutil.cpu_percent(interval=None),
                ram_used_gb_avg=psutil.virtual_memory().used / (1024 ** 3),
                ram_used_gb_max=psutil.virtual_memory().used / (1024 ** 3),
                gpu_percent_avg=gpu["percent"] if gpu else None,
                gpu_percent_max=gpu["percent"] if gpu else None,
                gpu_mem_used_gb_avg=gpu["mem_used_gb"] if gpu else None,
                gpu_mem_used_gb_max=gpu["mem_used_gb"] if gpu else None,
                sample_count=0,
            )
        return False


# ============================================================================
# Cost estimation
# ============================================================================
#
# There is no metered bill for a locally hosted Ollama instance, so "cost" is
# necessarily an estimate. We approximate it as the on-demand hourly rate of
# the smallest cloud instance capable of holding the model in memory,
# multiplied by wall-clock time actually spent in that model's calls. Rates
# are illustrative on-demand list prices (approx., USD/hr) and should be
# re-checked/cited before being reported as a paper claim rather than a
# relative comparison.
HARDWARE_COST_TABLE = [
    # (max_model_ram_gb, hourly_rate_usd, reference_instance)
    (2.5, 0.10, "CPU-class instance (e.g. AWS t3.xlarge on-demand)"),
    (4.0, 0.30, "Small GPU instance (e.g. AWS g5.xlarge on-demand)"),
    (16.0, 1.00, "Mid GPU instance (e.g. AWS g5.2xlarge on-demand)"),
    (24.0, 2.00, "Single A10G/L4-class instance (e.g. AWS g5.4xlarge)"),
    (48.0, 5.00, "Single A100-40GB-class instance (e.g. AWS p4d shared)"),
    (80.0, 8.00, "Single A100-80GB-class instance"),
    (float("inf"), 15.00, "Multi-GPU instance required"),
]


# --- Shared-pod cost model -------------------------------------------------
#
# HARDWARE_COST_TABLE above prices each model as if it had rented its own
# instance. That was a reasonable fiction while the models ran strictly
# one at a time, but it is simply wrong once five of them share a single
# pod: the bill is one machine-hour, not five, and summing per-model
# "dedicated instance" estimates overstates it several-fold.
#
# Set EPD_POD_HOURLY_USD to the real hourly rate of the machine actually
# being rented and costs switch to that single bill, apportioned across
# the models sharing it.
#
#   EPD_POD_HOURLY_USD   real rate of this machine, USD/hr (e.g. 1.89).
#                        Unset => the legacy per-model table is used.
#   EPD_POD_CONCURRENCY  how many models share it (default 1). The runner
#                        sets this to the number of parallel slots.
POD_HOURLY_USD: Optional[float] = None
_raw_pod_rate = os.environ.get("EPD_POD_HOURLY_USD", "").strip()
if _raw_pod_rate:
    try:
        POD_HOURLY_USD = float(_raw_pod_rate)
    except ValueError:
        POD_HOURLY_USD = None

try:
    POD_CONCURRENCY = max(int(os.environ.get("EPD_POD_CONCURRENCY", "1")), 1)
except ValueError:
    POD_CONCURRENCY = 1


def estimate_cost_usd(
    elapsed_seconds: float,
    model_ram_gb: Optional[float] = None,
    cpu_core_seconds: Optional[float] = None,
    gpu_mem_gb: Optional[float] = None,
) -> dict:
    """
    Estimate operational cost for `elapsed_seconds` of this model's work.

    Two models, selected by whether EPD_POD_HOURLY_USD is set:

      shared_pod -- the honest one for a concurrent run. One real machine
        rate, divided by the number of models sharing it. `wall_seconds`,
        `cpu_core_seconds` and `gpu_mem_gb_seconds` are recorded alongside
        so the flat 1/N split can be replaced by a usage-weighted one
        after the fact, once every model's totals are known
        (see apportioned_cost).

      dedicated_instance -- the legacy table, priced per model by RAM
        footprint. Correct only for a strictly sequential run, where each
        model really does have the machine to itself. Summing it across
        concurrently-run models overstates the bill.

    Every field the estimate rests on is returned, so the assumption is
    auditable in the output JSON rather than buried in this function.
    """
    resource_seconds = {
        "wall_seconds": round(elapsed_seconds, 4),
        "cpu_core_seconds": (
            round(cpu_core_seconds, 5) if cpu_core_seconds is not None else None
        ),
        "gpu_mem_gb_seconds": (
            round(gpu_mem_gb * elapsed_seconds, 5) if gpu_mem_gb is not None else None
        ),
    }

    if POD_HOURLY_USD is not None:
        unshared = POD_HOURLY_USD * (elapsed_seconds / 3600.0)
        return {
            "estimated_cost_usd": round(unshared / POD_CONCURRENCY, 8),
            "cost_model": "shared_pod",
            "hourly_rate_usd": POD_HOURLY_USD,
            "concurrency": POD_CONCURRENCY,
            "unshared_cost_usd": round(unshared, 8),
            "resource_seconds": resource_seconds,
            "note": (
                f"One real machine at ${POD_HOURLY_USD:.4f}/hr split evenly across "
                f"{POD_CONCURRENCY} concurrent model(s). Even split because a single "
                f"process cannot see its siblings' usage at call time; "
                f"resource_seconds is recorded so the split can be re-weighted by "
                f"measured usage once every model's run has finished."
            ),
        }

    if model_ram_gb is None:
        return {
            "estimated_cost_usd": None,
            "cost_model": "unavailable",
            "resource_seconds": resource_seconds,
            "note": (
                "No cost estimate: EPD_POD_HOURLY_USD is unset and this model has "
                "no entry in MODEL_RAM_GB. Reported as null rather than guessed."
            ),
        }

    for max_ram, rate, ref in HARDWARE_COST_TABLE:
        if model_ram_gb <= max_ram:
            hourly_rate = rate
            reference = ref
            break
    else:  # pragma: no cover - HARDWARE_COST_TABLE always has an inf sentinel
        hourly_rate, reference = HARDWARE_COST_TABLE[-1][1], HARDWARE_COST_TABLE[-1][2]

    cost = hourly_rate * (elapsed_seconds / 3600.0)
    return {
        "estimated_cost_usd": round(cost, 6),
        "cost_model": "dedicated_instance",
        "hourly_rate_usd": hourly_rate,
        "reference_instance": reference,
        "resource_seconds": resource_seconds,
        "note": (
            "Illustrative on-demand rate estimate, not a metered bill. Assumes this "
            "model had the machine to itself -- set EPD_POD_HOURLY_USD for a "
            "concurrent run, where that assumption does not hold."
        ),
    }


def apportioned_cost(
    pod_hourly_usd: float,
    pod_wall_seconds: float,
    per_model_usage: Dict[str, float],
) -> Dict[str, Any]:
    """
    Split one machine's real bill across models by measured usage.

    `per_model_usage` maps model key to any additive usage total on a
    common basis -- cpu_core_seconds and gpu_mem_gb_seconds are both
    recorded per call by estimate_cost_usd for exactly this purpose.
    Unlike the flat 1/N split applied at call time, this can only be
    computed once every model has finished, because it needs every
    model's total to form the denominator.

    `pod_wall_seconds` is the pod's own billed wall-clock -- from the
    first model starting to the last one finishing, not the sum of the
    models' individual runtimes, which under concurrency would count the
    same rented seconds several times over.
    """
    total_bill = pod_hourly_usd * (pod_wall_seconds / 3600.0)
    total_usage = sum(v for v in per_model_usage.values() if v)

    if total_usage <= 0:
        n = len(per_model_usage) or 1
        return {
            "total_bill_usd": round(total_bill, 6),
            "basis": "even_split_fallback",
            "per_model_usd": {k: round(total_bill / n, 6) for k in per_model_usage},
            "note": "No usable usage totals; fell back to an even split.",
        }

    return {
        "total_bill_usd": round(total_bill, 6),
        "basis": "measured_usage_share",
        "per_model_share": {
            k: round((v or 0.0) / total_usage, 6) for k, v in per_model_usage.items()
        },
        "per_model_usd": {
            k: round(total_bill * (v or 0.0) / total_usage, 6)
            for k, v in per_model_usage.items()
        },
        "note": (
            "One machine's real bill split by each model's measured share of total "
            "usage. pod_wall_seconds is the pod's billed wall-clock, not the sum of "
            "per-model runtimes."
        ),
    }
