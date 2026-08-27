"""
Free, local, no-GPU correctness check for attribution-aware resource
aggregation (src/ghost_agents/approach_evaluation/benchmark_evaluator.py).
No pytest dependency -- plain asserts, run directly:

    python3 tests/test_resource_attribution.py

Motivation: the 2026-08-27 seed-42 audit found 7 calls (qwen 3, deepseek 2,
phi3 2) whose per-process attribution momentarily failed and fell back to
machine-wide sampling. Those rows measure the whole box -- every model
running on it -- and reported 111-127 GB RAM next to 0.5 GB from the
per-process rows of the same model, ~200x off. _annotate_resource_provenance
already flagged such a cell as "mixed", but _aggregate_metrics still averaged
across the mix, so the flag changed nothing about the numbers.

Checks:
  1. A pure per-process cell keeps every row.
  2. A pure machine-wide cell (the sequential topology) also keeps every row --
     there, machine-wide is the correct and only mode.
  3. A mixed cell drops the machine-wide rows and aggregates per-process only.
  4. The real seed-42 shape (398 per-process + 2 machine-wide at ~200x) does
     not move the mean.
  5. Nulls are skipped per column independently.
  6. The provenance stamp reports per_process (not "mixed") for a mixed cell,
     names how many calls were excluded, and stays silent when none were.

This never touches Ollama, the GPU, or a real pod -- it costs nothing to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation import benchmark_evaluator as be  # noqa: E402


def _row(mode, cpu=2.0, ram=0.5, gpu=3.0, gpu_mem=1.9):
    return {
        "resource_attribution": mode,
        "cpu_percent_avg": cpu,
        "ram_used_gb_avg": ram,
        "gpu_percent_avg": gpu,
        "gpu_mem_used_gb_avg": gpu_mem,
        "call_status": "success",
    }


def test_pure_per_process_keeps_everything():
    rows = [_row("per_process") for _ in range(5)]
    r = be._resource_samples(rows)
    assert len(r["ram"]) == 5, r
    assert r["dropped"] == 0, r
    assert r["mode"] == "per_process"


def test_pure_machine_wide_keeps_everything():
    """Sequential topology: machine-wide is the correct and only mode."""
    rows = [_row("machine_wide", ram=120.0) for _ in range(5)]
    r = be._resource_samples(rows)
    assert len(r["ram"]) == 5, r
    assert r["dropped"] == 0, r
    assert r["mode"] is None, r


def test_mixed_cell_drops_machine_wide():
    rows = [_row("per_process", ram=0.5) for _ in range(4)] + [_row("machine_wide", ram=120.0)]
    r = be._resource_samples(rows)
    assert len(r["ram"]) == 4, r
    assert r["dropped"] == 1, r
    assert 120.0 not in r["ram"], r
    assert all(v == 0.5 for v in r["ram"]), r


def test_real_seed42_shape_does_not_move_the_mean():
    """398 per-process at 0.5 GB + 2 machine-wide at 120 GB."""
    rows = [_row("per_process", ram=0.5) for _ in range(398)]
    rows += [_row("machine_wide", ram=120.0) for _ in range(2)]
    r = be._resource_samples(rows)
    mean = sum(r["ram"]) / len(r["ram"])
    assert abs(mean - 0.5) < 1e-9, f"machine-wide rows leaked into the mean: {mean}"
    polluted = (0.5 * 398 + 120.0 * 2) / 400
    assert polluted > 1.0, "sanity: the unfiltered mean really is inflated"


def test_nulls_skipped_per_column():
    rows = [_row("per_process"), _row("per_process", gpu_mem=None), _row("per_process", cpu=None)]
    r = be._resource_samples(rows)
    assert len(r["ram"]) == 3
    assert len(r["gpu_mem"]) == 2
    assert len(r["cpu"]) == 2


def test_empty_input():
    r = be._resource_samples([])
    assert r["ram"] == [] and r["dropped"] == 0


def test_provenance_reports_per_process_and_counts_exclusions():
    rows = [_row("per_process") for _ in range(4)] + [_row("machine_wide", ram=120.0)]
    ar = {"test_results": rows, "metrics": {}}
    be._annotate_resource_provenance(ar)
    m = ar["metrics"]
    assert m["resource_attribution"] == "per_process", m
    assert m["resource_calls_excluded"] == 1, m
    assert "EXCLUDED" in m["resource_attribution_warning"], m


def test_provenance_silent_when_nothing_excluded():
    ar = {"test_results": [_row("per_process") for _ in range(5)], "metrics": {}}
    be._annotate_resource_provenance(ar)
    m = ar["metrics"]
    assert m["resource_attribution"] == "per_process"
    assert "resource_calls_excluded" not in m, m
    assert "resource_attribution_warning" not in m, m


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
