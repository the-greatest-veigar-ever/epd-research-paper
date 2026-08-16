"""
Runtime Resource Monitor.

Addresses reviewer feedback that the efficiency analysis relied on static
memory footprint alone. This module samples CPU and RAM utilization for the
duration of a single model call, and provides a cost estimate derived from a
documented, editable hardware-rate table.

GPU utilization is intentionally NOT sampled on this platform: the reference
hardware is Apple Silicon (no NVIDIA GPU / no nvidia-smi), so any reported
"GPU utilization" would be fabricated. `gpu_available` is exposed so callers
and reports can label the field N/A instead of silently omitting it.
"""

import platform
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

GPU_MONITORING_AVAILABLE = shutil.which("nvidia-smi") is not None


@dataclass
class ResourceSample:
    t: float
    cpu_percent: float
    ram_used_gb: float


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
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 4),
            "cpu_percent_avg": round(self.cpu_percent_avg, 2),
            "cpu_percent_max": round(self.cpu_percent_max, 2),
            "ram_used_gb_avg": round(self.ram_used_gb_avg, 3),
            "ram_used_gb_max": round(self.ram_used_gb_max, 3),
            "gpu_percent_avg": self.gpu_percent_avg,
            "gpu_percent_max": self.gpu_percent_max,
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

    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self._samples: List[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start = 0.0
        self._t_end = 0.0
        self.stats: ResourceStats = ResourceStats()

    def _sample_loop(self):
        # Prime psutil's internal CPU counter so the first real reading isn't 0.0.
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
            self._samples.append(ResourceSample(t=time.perf_counter(), cpu_percent=cpu, ram_used_gb=ram_used_gb))
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
            self.stats = ResourceStats(
                duration_s=duration,
                cpu_percent_avg=sum(cpu_values) / len(cpu_values),
                cpu_percent_max=max(cpu_values),
                ram_used_gb_avg=sum(ram_values) / len(ram_values),
                ram_used_gb_max=max(ram_values),
                sample_count=len(self._samples),
            )
        else:
            # Call was faster than one sampling interval; take a single point reading.
            self.stats = ResourceStats(
                duration_s=duration,
                cpu_percent_avg=psutil.cpu_percent(interval=None),
                cpu_percent_max=psutil.cpu_percent(interval=None),
                ram_used_gb_avg=psutil.virtual_memory().used / (1024 ** 3),
                ram_used_gb_max=psutil.virtual_memory().used / (1024 ** 3),
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


def estimate_cost_usd(elapsed_seconds: float, model_ram_gb: float) -> dict:
    """
    Estimate operational cost for `elapsed_seconds` of compute on a model
    requiring `model_ram_gb` of memory. Returns the rate used and the
    resulting estimate, so the assumption is auditable in the output JSON.
    """
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
        "hourly_rate_usd": hourly_rate,
        "reference_instance": reference,
        "note": "Illustrative on-demand rate estimate, not a metered bill.",
    }
