"""
Per-process resource attribution for the concurrent evaluation topology.

Why this module exists
----------------------
`resource_monitor.ResourceMonitor` samples the machine as a whole
(`psutil.cpu_percent()`, `psutil.virtual_memory()`, device-wide
`nvidia-smi` totals). Those readings are only attributable to a model
when exactly one model is resident and one process is active -- which is
precisely why `run_runpod_experiment.sh` runs the five SLMs strictly
sequentially. Anything concurrent turns every resource number into an
average over whichever models happened to be running at that instant.

Sequential execution buys that validity with roughly 5x the pod bill.
This module removes the trade-off by attributing usage per OS process
rather than per machine, so five models can share one pod and still
report separate CPU / RAM / GPU figures.

The identity problem (why we anchor on a server, not on a PID)
--------------------------------------------------------------
Per-process attribution is only as trustworthy as the process identity
behind it, and this experiment makes that identity move: every
*ephemeral* ablation cell unloads and reloads its model on every call,
and each reload is a brand-new OS process with a brand-new PID. A
monitor that resolves one PID up front and then polls it forever gets
two things wrong at once -- it stops seeing the model it is meant to
measure, and once the kernel recycles that PID number (which it will,
with five models churning), it silently starts recording an unrelated
process under the original model's name.

So identity is anchored one level up. Each model gets its own dedicated
`ollama serve`, on its own port, for the entire run, and the recorder
tracks *that server process plus whatever children are alive under it at
each sample*. The server is long-lived, so the anchor never churns; the
runner children appear and disappear exactly as the ephemerality factor
dictates, and are picked up or dropped as they do. Because no other
model is ever routed to that server, a new child under it is always this
model's reload -- never a different model landing on a recycled PID.

`run_concurrent_experiment.py` is what establishes that topology.

Degradation policy
------------------
Every capability here is optional and independently detected: NVML may
be absent, per-process GPU *utilisation* may be unsupported by the
driver, and a container may hide the PID mapping the GPU numbers depend
on. In each case the affected field is reported as None with an
explanatory note. Nothing in this module ever substitutes a plausible
number for a measurement it could not take -- a missing sample must
never be readable as "0% / 0 GB".
"""

import csv
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

# ---------------------------------------------------------------------------
# NVML (per-process GPU) -- optional
# ---------------------------------------------------------------------------
# `nvidia-ml-py` installs as the `pynvml` module. Unlike the `nvidia-smi`
# subprocess used by resource_monitor.py, NVML is an in-process library
# call: no process spawn per sample, so it can be polled sub-second
# without the instrument's own cost landing in the CPU figure it records
# (the reason GPU sampling there had to be throttled to 1s).
try:  # pragma: no cover - import guard
    import pynvml  # type: ignore

    NVML_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - import guard
    pynvml = None  # type: ignore
    NVML_IMPORT_ERROR = str(_e)


_nvml_lock = threading.Lock()
_nvml_state: Dict[str, Any] = {"initialized": False, "handles": [], "error": None}


def nvml_init() -> Tuple[bool, Optional[str]]:
    """
    Initialise NVML once per process. Returns (ok, error_message).

    Safe to call repeatedly and from several threads; the result is
    cached, including the failure, so a machine without a driver does not
    pay a failed init on every sample.
    """
    with _nvml_lock:
        if _nvml_state["initialized"]:
            return bool(_nvml_state["handles"]), _nvml_state["error"]

        _nvml_state["initialized"] = True

        if pynvml is None:
            _nvml_state["error"] = (
                f"pynvml (nvidia-ml-py) not importable: {NVML_IMPORT_ERROR}"
            )
            return False, _nvml_state["error"]

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            _nvml_state["handles"] = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
            ]
            if not _nvml_state["handles"]:
                _nvml_state["error"] = "NVML initialised but reported 0 devices"
                return False, _nvml_state["error"]
            return True, None
        except Exception as e:
            _nvml_state["handles"] = []
            _nvml_state["error"] = f"NVML init failed: {e}"
            return False, _nvml_state["error"]


def _nvml_compute_processes() -> Optional[List[Tuple[int, Optional[int]]]]:
    """
    Every process currently holding a CUDA compute context, as
    [(pid, used_gpu_memory_bytes_or_None), ...] summed across all visible
    devices. Returns None if NVML is unavailable.

    NVML exposes this under three different symbol versions depending on
    the bindings' age; they return the same fields, so whichever exists is
    fine.
    """
    ok, _ = nvml_init()
    if not ok:
        return None

    getters = [
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ]
    out: List[Tuple[int, Optional[int]]] = []
    for handle in _nvml_state["handles"]:
        for name in getters:
            fn = getattr(pynvml, name, None)
            if fn is None:
                continue
            try:
                for proc in fn(handle):
                    mem = getattr(proc, "usedGpuMemory", None)
                    # NVML reports a sentinel rather than 0 when it cannot
                    # read a process's memory (permissions, MIG); keep that
                    # distinguishable from a real zero.
                    if mem is not None and mem >= (1 << 62):
                        mem = None
                    out.append((int(proc.pid), mem))
                break
            except Exception:
                continue
    return out


class _ProcessUtilisationReader:
    """
    Per-process SM utilisation via `nvmlDeviceGetProcessUtilization`.

    Supported from Volta onward (the A100 this experiment targets is
    Ampere, so it is expected to work there), but genuinely absent on
    older or heavily virtualised drivers -- hence the one-shot capability
    probe and the permanent fall back to None. Per-process GPU *memory*
    does not depend on this call and stays available either way.
    """

    def __init__(self) -> None:
        self.supported: Optional[bool] = None
        self.reason: Optional[str] = None
        # NVML returns samples newer than this timestamp (microseconds).
        self._last_seen_us: int = 0

    def read(self) -> Optional[Dict[int, float]]:
        ok, err = nvml_init()
        if not ok:
            self.supported = False
            self.reason = err
            return None

        if self.supported is False:
            return None

        fn = getattr(pynvml, "nvmlDeviceGetProcessUtilization", None)
        if fn is None:
            self.supported = False
            self.reason = "nvmlDeviceGetProcessUtilization missing from bindings"
            return None

        # Look back one second on the first read so the first window is not
        # empty purely because no baseline timestamp existed yet.
        since = self._last_seen_us or int((time.time() - 1.0) * 1_000_000)
        per_pid: Dict[int, float] = {}
        newest = self._last_seen_us
        saw_any_device = False

        for handle in _nvml_state["handles"]:
            try:
                samples = fn(handle, since)
                saw_any_device = True
            except Exception as e:
                msg = str(e)
                # "NotFound" just means no samples in this interval -- that
                # is an idle GPU, not an unsupported driver.
                if "NotFound" in msg or "not found" in msg.lower():
                    saw_any_device = True
                    continue
                if self.supported is None:
                    self.supported = False
                    self.reason = f"nvmlDeviceGetProcessUtilization unsupported: {e}"
                return None

            for s in samples:
                pid = int(s.pid)
                per_pid[pid] = per_pid.get(pid, 0.0) + float(s.smUtil)
                newest = max(newest, int(s.timeStamp))

        if saw_any_device:
            self.supported = True
            if newest:
                self._last_seen_us = newest
        return per_pid


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

@dataclass
class TreeSample:
    """One observation of an anchored process tree."""

    t: float                      # time.perf_counter(), matches call timing
    wall: float                   # time.time(), for the CSV record
    alive: bool
    n_procs: int
    # CPU-seconds the tree consumed since the previous sample, and the wall
    # interval they were consumed over. These, not the percentages below,
    # are the primary measurement: they are additive across samples and
    # across models, so a window total is exact rather than an average of
    # averages, and they are what a shared pod bill gets split by.
    cpu_seconds_delta: Optional[float]
    interval_s: Optional[float]
    # Derived. cpu_percent_cores uses psutil's convention where 100.0 is one
    # core fully busy, so it can exceed 100 on a multi-core box;
    # cpu_percent normalises by core count (0-100) to stay comparable with
    # what the machine-wide monitor reported.
    cpu_percent_cores: Optional[float]
    cpu_percent: Optional[float]
    ram_gb: Optional[float]
    gpu_mem_gb: Optional[float]
    gpu_util_percent: Optional[float]
    gpu_pids_matched: int


def _cpu_count() -> int:
    return psutil.cpu_count(logical=True) or 1


class ProcessTreeRecorder:
    """
    Long-lived background sampler for one anchored process tree.

    Deliberately *not* a per-call context manager, unlike
    `resource_monitor.ResourceMonitor`. `psutil.Process.cpu_percent()`
    reports the average since that same object's previous call, so its
    first reading on a fresh object is always 0.0 -- a monitor
    constructed per call either discards its first sample or, worse,
    records that structural zero as though the process were idle. A
    recorder that runs for the whole evaluation keeps its `Process`
    objects warm, so every reading is a true interval average; per-call
    figures are then taken by slicing the sample buffer to the call's
    time window (`window_stats`).

    Args:
        root_pid: PID of the model's dedicated `ollama serve`. Its
            children are resolved fresh on every sample, so an ephemeral
            cell's unload/reload churn is followed rather than lost.
        label: Model key this tree belongs to, used in the CSV.
        interval_s: Sampling cadence.
        csv_path: Optional path for the streamed time series.
        max_samples: Ring-buffer bound. At 0.5s a multi-day run would
            otherwise accumulate millions of samples in memory; the CSV
            (when enabled) is the complete record, this buffer only has
            to cover the current call.
    """

    def __init__(
        self,
        root_pid: int,
        label: str,
        interval_s: float = 0.5,
        csv_path: Optional[str] = None,
        max_samples: int = 200_000,
    ) -> None:
        self.root_pid = int(root_pid)
        self.label = label
        self.interval_s = max(float(interval_s), 0.05)
        self.csv_path = csv_path
        self.max_samples = max_samples

        self._samples: List[TreeSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Keyed by (pid, create_time), never by pid alone. The creation time
        # is what distinguishes a process from a later one that happened to
        # be handed the same PID number -- with five models constantly
        # tearing down and respawning runners, PID reuse is routine, and a
        # pid-only key would silently carry a dead process's CPU baseline
        # over to an unrelated newcomer.
        self._cpu_baseline: Dict[Tuple[int, float], float] = {}
        self._last_sample_t: Optional[float] = None
        self._util_reader = _ProcessUtilisationReader()
        self._ncpu = _cpu_count()

        self._csv_file = None
        self._csv_writer = None

        # Diagnostics surfaced in every stats payload so a degraded run is
        # visible in the results themselves, not just in a log nobody reads.
        # Initialised before the anchor is resolved below, which may set
        # root_alive False.
        self.root_alive: bool = True
        self.gpu_available: bool = False
        self.gpu_error: Optional[str] = None
        self.gpu_pid_mismatch: bool = False
        self._nvml_procs_seen: int = 0
        self._nvml_procs_matched: int = 0

        # The anchor's own identity, pinned at construction: if the server
        # dies, the monitor must report that rather than silently
        # re-anchoring onto whatever later inherits its PID.
        self._root_created_at: Optional[float] = None
        try:
            self._root_created_at = psutil.Process(self.root_pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.root_alive = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ProcessTreeRecorder":
        ok, err = nvml_init()
        self.gpu_available = ok
        self.gpu_error = err

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp", "activity", "root_pid", "status", "n_procs",
                "cpu_percent", "cpu_percent_cores", "cpu_seconds_delta",
                "interval_s", "ram_mb",
                "gpu_mem_mb", "gpu_util_percent", "gpu_pids_matched",
            ])

        self._thread = threading.Thread(
            target=self._loop, name=f"monitor-{self.label}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
            self._thread = None
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

    # -- sampling ----------------------------------------------------------

    def _resolve_tree(self) -> List[psutil.Process]:
        """
        The anchor process plus every live descendant, re-resolved on each
        sample. Re-resolving (rather than caching the child list) is what
        makes an ephemeral cell's per-call reload visible: the runner
        process it spawns is a new PID that simply appears under the
        anchor on the next tick.
        """
        try:
            root = psutil.Process(self.root_pid)
            # Same PID is not the same process. If the server died and the
            # kernel handed its number to something else, attributing that
            # stranger's CPU and memory to this model would be worse than
            # reporting nothing -- so the creation time is checked too.
            if (
                not root.is_running()
                or (
                    self._root_created_at is not None
                    and root.create_time() != self._root_created_at
                )
            ):
                self.root_alive = False
                return []
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            self.root_alive = False
            return []

        self.root_alive = True
        procs = [root]
        try:
            procs.extend(root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return procs

    def _sample(self) -> TreeSample:
        """
        One observation of the tree.

        CPU is measured as the growth in each process's cumulative CPU time
        since the previous sample, rather than with
        `psutil.Process.cpu_percent()`. Two reasons, both of which bite
        precisely on the ephemeral cells this experiment exists to measure:

        `cpu_percent()` reports the average since the *same object's* last
        call, and `children()` hands back new objects every sweep -- so a
        child would be re-primed on every tick and never contribute at all.
        And a percentage cannot be summed: totalling CPU-seconds gives an
        exact figure for any window, which averaging per-tick percentages
        does not, and it is the additive basis a shared bill is split by.
        """
        now_t = time.perf_counter()
        now_wall = time.time()
        procs = self._resolve_tree()
        interval = (
            now_t - self._last_sample_t if self._last_sample_t is not None else None
        )
        self._last_sample_t = now_t

        if not procs:
            self._prune_cache(set())
            return TreeSample(
                t=now_t, wall=now_wall, alive=False, n_procs=0,
                cpu_seconds_delta=None, interval_s=interval,
                cpu_percent_cores=None, cpu_percent=None, ram_gb=None,
                gpu_mem_gb=None, gpu_util_percent=None, gpu_pids_matched=0,
            )

        live_pids: Set[int] = set()
        live_keys: Set[Tuple[int, float]] = set()
        cpu_seconds_delta = 0.0
        measured_any_cpu = False
        rss_total = 0
        rss_seen = False

        for proc in procs:
            try:
                key = (proc.pid, proc.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            live_pids.add(proc.pid)
            live_keys.add(key)

            try:
                times = proc.cpu_times()
                cpu_now = float(times.user) + float(times.system)
                rss_total += proc.memory_info().rss
                rss_seen = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                # Vanished between enumeration and measurement. Routine for
                # ephemeral cells, not an error.
                continue

            previous = self._cpu_baseline.get(key)
            self._cpu_baseline[key] = cpu_now
            if previous is None:
                # First sighting. Its own CPU total is the baseline rather
                # than zero, so we never credit this model with work done
                # before we started watching. The cost is at most one
                # interval of a freshly spawned runner's startup.
                continue
            if cpu_now >= previous:
                cpu_seconds_delta += cpu_now - previous
                measured_any_cpu = True

        self._prune_cache(live_keys)

        gpu_mem_gb, gpu_util, matched = self._sample_gpu(live_pids)

        cpu_cores = cpu_percent = None
        if measured_any_cpu and interval and interval > 0:
            cpu_cores = cpu_seconds_delta / interval * 100.0
            cpu_percent = min(cpu_cores / self._ncpu, 100.0)

        return TreeSample(
            t=now_t,
            wall=now_wall,
            alive=True,
            n_procs=len(live_pids),
            cpu_seconds_delta=round(cpu_seconds_delta, 6) if measured_any_cpu else None,
            interval_s=round(interval, 4) if interval else None,
            cpu_percent_cores=round(cpu_cores, 3) if cpu_cores is not None else None,
            cpu_percent=round(cpu_percent, 3) if cpu_percent is not None else None,
            ram_gb=round(rss_total / (1024 ** 3), 4) if rss_seen else None,
            gpu_mem_gb=gpu_mem_gb,
            gpu_util_percent=gpu_util,
            gpu_pids_matched=matched,
        )

    def _sample_gpu(
        self, live_pids: Set[int]
    ) -> Tuple[Optional[float], Optional[float], int]:
        if not self.gpu_available:
            return None, None, 0

        procs = _nvml_compute_processes()
        if procs is None:
            return None, None, 0

        self._nvml_procs_seen = len(procs)
        mem_bytes = 0
        matched = 0
        mem_known = False
        for pid, mem in procs:
            if pid in live_pids:
                matched += 1
                if mem is not None:
                    mem_bytes += mem
                    mem_known = True
        self._nvml_procs_matched = matched

        # NVML sees compute processes but none of them are ours. On a pod
        # this almost always means the container has its own PID namespace
        # while NVML reports host PIDs, so the two sets can never
        # intersect. Flagged rather than silently reported as 0 GB, since
        # "this model used no GPU memory" and "we could not tell" are very
        # different claims to put in a paper.
        if procs and matched == 0:
            self.gpu_pid_mismatch = True

        util_map = self._util_reader.read()
        gpu_util: Optional[float] = None
        if util_map is not None and matched:
            total = sum(v for pid, v in util_map.items() if pid in live_pids)
            gpu_util = round(min(total, 100.0), 2)

        gpu_mem_gb = round(mem_bytes / (1024 ** 3), 4) if mem_known else None
        return gpu_mem_gb, gpu_util, matched

    def _prune_cache(self, live_keys: Set[Tuple[int, float]]) -> None:
        """
        Drop processes that are gone. Essential rather than housekeeping:
        an unbounded baseline map would grow with every reload over a
        multi-day run, and keeping a dead entry is what would let a
        recycled PID inherit its predecessor's CPU baseline.
        """
        for key in list(self._cpu_baseline):
            if key not in live_keys:
                self._cpu_baseline.pop(key, None)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._sample()
            except Exception:
                # A sampling failure must never take down the evaluation it
                # is only observing.
                self._stop_event.wait(self.interval_s)
                continue

            with self._lock:
                self._samples.append(sample)
                if len(self._samples) > self.max_samples:
                    del self._samples[: len(self._samples) // 4]

            if self._csv_writer is not None:
                try:
                    self._csv_writer.writerow([
                        f"{sample.wall:.3f}", self.label, self.root_pid,
                        "running" if sample.alive else "finished",
                        sample.n_procs,
                        "" if sample.cpu_percent is None else sample.cpu_percent,
                        "" if sample.cpu_percent_cores is None else sample.cpu_percent_cores,
                        "" if sample.cpu_seconds_delta is None else sample.cpu_seconds_delta,
                        "" if sample.interval_s is None else sample.interval_s,
                        "" if sample.ram_gb is None else round(sample.ram_gb * 1024, 2),
                        "" if sample.gpu_mem_gb is None else round(sample.gpu_mem_gb * 1024, 2),
                        "" if sample.gpu_util_percent is None else sample.gpu_util_percent,
                        sample.gpu_pids_matched,
                    ])
                    self._csv_file.flush()
                except Exception:
                    pass

            self._stop_event.wait(self.interval_s)

    # -- reporting ---------------------------------------------------------

    def window_stats(self, t_start: float, t_end: float) -> Dict[str, Any]:
        """
        Aggregate every sample taken inside [t_start, t_end], both on
        `time.perf_counter()`.

        A window with no samples (a call shorter than one interval) falls
        back to a single synchronous reading and reports sample_count=0,
        matching how `ResourceMonitor` handles the same case.
        """
        # Each sample's CPU delta covers the interval ENDING at its
        # timestamp, so a call is covered by samples up to one interval past
        # its end. Widening the upper bound is what lets a call shorter than
        # the sampling cadence still be measured rather than missed.
        with self._lock:
            window = [
                s for s in self._samples
                if t_start <= s.t <= t_end + self.interval_s
            ]
            nearest: List[TreeSample] = []
            if not window and self._samples:
                nearest = [min(self._samples, key=lambda s: abs(s.t - t_end))]

        sample_count = len(window)
        # Deliberately does NOT fall back to taking a fresh sample here:
        # _sample() mutates the CPU baselines owned by the sampler thread,
        # and calling it from the evaluator thread would corrupt the next
        # interval's delta for both.
        effective = window or nearest

        if not effective:
            return {
                "attribution": "per_process",
                "duration_s": round(max(t_end - t_start, 0.0), 4),
                "sample_count": 0,
                "monitored_root_pid": self.root_pid,
                "monitored_label": self.label,
                "cpu_percent_avg": None, "cpu_percent_max": None,
                "cpu_percent_cores_avg": None,
                "ram_used_gb_avg": None, "ram_used_gb_max": None,
                "gpu_percent_avg": None, "gpu_percent_max": None,
                "gpu_mem_used_gb_avg": None, "gpu_mem_used_gb_max": None,
                "cpu_core_seconds": None,
                "platform": f"{platform.system()} {platform.machine()}",
                "gpu_note": self._gpu_note(),
                "monitor_warning": "No resource samples were taken during this call.",
            }

        def _avg(key: str) -> Optional[float]:
            vals = [getattr(s, key) for s in effective if getattr(s, key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _max(key: str) -> Optional[float]:
            vals = [getattr(s, key) for s in effective if getattr(s, key) is not None]
            return round(max(vals), 4) if vals else None

        duration = max(t_end - t_start, 0.0)

        # Derived from summed CPU-seconds over summed intervals, not by
        # averaging per-sample percentages: samples are not perfectly evenly
        # spaced, so a mean of percentages weights a short interval the same
        # as a long one. This is a true time-weighted utilisation.
        cpu_seconds = sum(
            s.cpu_seconds_delta for s in effective if s.cpu_seconds_delta is not None
        )
        measured_span = sum(
            s.interval_s for s in effective
            if s.cpu_seconds_delta is not None and s.interval_s
        )
        cpu_cores_avg = cpu_percent_avg = None
        if measured_span > 0:
            cpu_cores_avg = round(cpu_seconds / measured_span * 100.0, 3)
            cpu_percent_avg = round(min(cpu_cores_avg / self._ncpu, 100.0), 3)

        stats: Dict[str, Any] = {
            "attribution": "per_process",
            "duration_s": round(duration, 4),
            "cpu_percent_avg": cpu_percent_avg,
            "cpu_percent_max": _max("cpu_percent"),
            "cpu_percent_cores_avg": cpu_cores_avg,
            "ram_used_gb_avg": _avg("ram_gb"),
            "ram_used_gb_max": _max("ram_gb"),
            "gpu_percent_avg": _avg("gpu_util_percent"),
            "gpu_percent_max": _max("gpu_util_percent"),
            "gpu_mem_used_gb_avg": _avg("gpu_mem_gb"),
            "gpu_mem_used_gb_max": _max("gpu_mem_gb"),
            "platform": f"{platform.system()} {platform.machine()}",
            "sample_count": sample_count,
            "monitored_root_pid": self.root_pid,
            "monitored_label": self.label,
            "avg_procs_in_tree": round(
                sum(s.n_procs for s in effective) / len(effective), 2
            ),
            # Measured CPU-seconds, not a percentage rescaled by duration.
            # Additive across calls and across models, which is what makes an
            # honest split of one shared pod bill possible after the fact
            # (see resource_monitor.apportioned_cost).
            "cpu_core_seconds": round(cpu_seconds, 5) if measured_span > 0 else None,
        }
        stats["gpu_note"] = self._gpu_note()
        if not window and nearest:
            stats["monitor_warning"] = (
                "This call was shorter than the sampling interval; figures are the "
                "nearest reading rather than an in-window measurement."
            )
        if not self.root_alive:
            stats["monitor_warning"] = (
                f"The monitored Ollama server (pid {self.root_pid}) was not "
                f"running during this call; resource figures are unattributable."
            )
        return stats

    def _gpu_note(self) -> Optional[str]:
        if not self.gpu_available:
            return (
                f"Per-process GPU metrics unavailable ({self.gpu_error}); "
                f"reported as N/A rather than estimated."
            )
        if self.gpu_pid_mismatch:
            return (
                f"NVML reported {self._nvml_procs_seen} compute process(es) but none "
                f"matched this model's process tree. This is the signature of a "
                f"container PID namespace: NVML reports host PIDs while psutil sees "
                f"namespaced ones. GPU figures are reported as N/A rather than 0. "
                f"Run the container with host PID visibility (docker --pid=host) to "
                f"recover per-process GPU attribution."
            )
        if self._util_reader.supported is False:
            return (
                f"Per-process GPU memory is available, but per-process GPU "
                f"utilisation is not supported by this driver "
                f"({self._util_reader.reason}); utilisation reported as N/A."
            )
        return None

    def diagnostics(self) -> Dict[str, Any]:
        """One-shot health summary, logged once at start-up by the runner."""
        return {
            "label": self.label,
            "root_pid": self.root_pid,
            "root_alive": self.root_alive,
            "interval_s": self.interval_s,
            "cpu_count": self._ncpu,
            "gpu_available": self.gpu_available,
            "gpu_error": self.gpu_error,
            "gpu_per_process_util_supported": self._util_reader.supported,
            "gpu_pid_mismatch": self.gpu_pid_mismatch,
            "nvml_compute_procs_seen": self._nvml_procs_seen,
            "nvml_compute_procs_matched": self._nvml_procs_matched,
            "csv_path": self.csv_path,
        }


# ---------------------------------------------------------------------------
# Anchor discovery
# ---------------------------------------------------------------------------

def discover_listening_pid(port: int) -> Optional[int]:
    """
    PID of the process listening on `port`, or None.

    Only a fallback. `run_concurrent_experiment.py` starts each model's
    server itself and passes the PID down explicitly, which is exact;
    this covers a manually started server. Socket enumeration needs
    elevated privileges on some platforms, hence the broad except.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if (
                conn.laddr
                and conn.laddr.port == port
                and conn.status == psutil.CONN_LISTEN
                and conn.pid
            ):
                return int(conn.pid)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Process-global recorder
# ---------------------------------------------------------------------------
#
# One evaluator process drives exactly one model against exactly one
# dedicated Ollama server, so a single module-level recorder per process
# is the natural scope. It is created on first use from the environment
# the runner sets up:
#
#   EPD_MONITOR_ROOT_PID  PID of this model's `ollama serve` (exact; set by
#                         run_concurrent_experiment.py).
#   EPD_OLLAMA_PORT       Port of that server; used to discover the PID if
#                         EPD_MONITOR_ROOT_PID is absent.
#   EPD_MONITOR_INTERVAL  Sampling cadence in seconds (default 0.5).
#   EPD_MONITOR_CSV       Optional path for the streamed time series.
#   EPD_MONITOR_LABEL     Model key used in the CSV / stats payloads.
#
# With none of these set, `get_recorder()` returns None and callers fall
# back to the machine-wide ResourceMonitor -- which is correct for the
# sequential topology, where the machine *is* the model.

_recorder: Optional[ProcessTreeRecorder] = None
_recorder_resolved = False
_recorder_lock = threading.Lock()


def get_recorder() -> Optional[ProcessTreeRecorder]:
    """The process-global recorder, or None if no anchor is configured."""
    global _recorder, _recorder_resolved
    if _recorder_resolved:
        return _recorder

    with _recorder_lock:
        if _recorder_resolved:
            return _recorder
        _recorder_resolved = True

        root_pid: Optional[int] = None
        raw_pid = os.environ.get("EPD_MONITOR_ROOT_PID", "").strip()
        if raw_pid:
            try:
                root_pid = int(raw_pid)
            except ValueError:
                root_pid = None

        if root_pid is None:
            raw_port = os.environ.get("EPD_OLLAMA_PORT", "").strip()
            if raw_port:
                try:
                    root_pid = discover_listening_pid(int(raw_port))
                except ValueError:
                    root_pid = None

        if root_pid is None:
            return None

        if not psutil.pid_exists(root_pid):
            print(
                f"[monitor] WARNING: configured root pid {root_pid} does not exist; "
                f"falling back to machine-wide resource sampling."
            )
            return None

        label = os.environ.get("EPD_MONITOR_LABEL") or f"pid{root_pid}"
        try:
            interval = float(os.environ.get("EPD_MONITOR_INTERVAL", "0.5"))
        except ValueError:
            interval = 0.5
        csv_path = os.environ.get("EPD_MONITOR_CSV") or None

        _recorder = ProcessTreeRecorder(
            root_pid=root_pid, label=label, interval_s=interval, csv_path=csv_path
        ).start()
        print(f"[monitor] per-process attribution active: {_recorder.diagnostics()}")
        return _recorder


def shutdown_recorder() -> None:
    """Stop the process-global recorder, if one was started."""
    global _recorder
    if _recorder is not None:
        _recorder.stop()
        _recorder = None
