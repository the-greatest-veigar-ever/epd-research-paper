#!/usr/bin/env python3
"""
Recompute the derived metrics in existing result files, offline.

Every number this touches is DERIVED from `test_results`, which the script
never modifies. Nothing here calls Ollama, loads a model, starts the resource
monitor, or costs a single pod-second -- it is pure arithmetic over JSON that
is already on disk.

Why this exists rather than just re-running the evaluator: re-running the
orchestrator over a completed checkpoint would (a) truncate every cell from
its first failed call and re-run those calls, because `retry_failed` defaults
to True, and (b) reopen `resource_timeseries_<model>.csv` with mode "w",
destroying the real time series from the original run. Both are silent. This
script has neither effect.

Written for the 2026-08-27 attribution fix: 7 calls across qwen/deepseek/phi3
fell back to machine-wide resource sampling and were being averaged in with
per-process rows, inflating some cells' RAM figures by 17-61x. The evaluator
now excludes them (see `_resource_samples`); this brings the already-written
files in line with that.

Usage:
    python3 analysis/recompute_metrics.py --dry-run     # report, write nothing
    python3 analysis/recompute_metrics.py               # rewrite in place
    python3 analysis/recompute_metrics.py --seed 42 --models phi3_mini
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation.benchmark_evaluator import (  # noqa: E402
    _aggregate_metrics,
    _annotate_resource_provenance,
    _atomic_write_json,
    _compute_summary,
    _seed_metric_lists_from_prior,
)

RESULTS_DIR = "report-output/ghost_agents/benchmark_results"
# Metrics worth reporting a delta on; the rest are recomputed silently.
WATCH = ("avg_ram_gb", "avg_cpu_percent", "avg_gpu_percent", "avg_gpu_mem_used_gb")


def recompute_approach(approach_result):
    """Rebuild one cell's `metrics` from its own test_results. Returns the
    fields that changed, as {field: (before, after)}."""
    test_results = approach_result.get("test_results")
    if not test_results:
        return {}

    before = dict(approach_result.get("metrics") or {})

    (scores, safe_count, init_lat, inf_lat, cpu, ram, cost,
     timeout_count, error_count, gpu, gpu_mem) = _seed_metric_lists_from_prior(test_results)

    approach_result["metrics"] = _aggregate_metrics(
        scores, safe_count, init_lat, inf_lat, cpu, ram, cost,
        len(test_results), timeout_count, error_count, gpu, gpu_mem,
    )
    # setup_latency_s is measured, not derived -- carry it across.
    if before.get("setup_latency_s") is not None:
        approach_result["metrics"]["setup_latency_s"] = before["setup_latency_s"]
    _annotate_resource_provenance(approach_result)

    after = approach_result["metrics"]
    return {
        k: (before.get(k), after.get(k))
        for k in WATCH
        if before.get(k) != after.get(k)
    }


def recompute_file(path, dry_run):
    """Recompute a file holding benchmark_results with test_results."""
    with open(path) as f:
        doc = json.load(f)
    if "benchmark_results" not in doc:
        return None

    changes = []
    for bench, cell in doc["benchmark_results"].items():
        for name, approach_result in cell.get("approaches", {}).items():
            delta = recompute_approach(approach_result)
            if delta:
                changes.append((bench, name, delta))

    if "summary" in doc:
        doc["summary"] = _compute_summary(doc)

    if changes and not dry_run:
        _atomic_write_json(path, doc, indent=2, default=str)
        # A standalone benchmark_summary_* sits beside each benchmark_eval_*
        # holding just the summary block -- keep the pair consistent.
        sibling = path.replace("benchmark_eval_", "benchmark_summary_")
        if sibling != path and os.path.exists(sibling) and "summary" in doc:
            _atomic_write_json(sibling, doc["summary"], indent=2, default=str)

    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; write nothing.")
    ap.add_argument("--seed", default="42", help="Seed tag to match (default: 42).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model keys to process (default: every folder found).")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    targets = []
    for model_dir in sorted(glob.glob(os.path.join(args.results_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        if args.models and model not in args.models:
            continue
        targets += sorted(glob.glob(os.path.join(model_dir, f"checkpoint_seed{args.seed}.json")))
        targets += sorted(glob.glob(os.path.join(model_dir, f"benchmark_eval_*seed{args.seed}*.json")))
    # Combined views live at the results-dir root.
    if not args.models:
        targets += sorted(glob.glob(os.path.join(
            args.results_dir, f"benchmark_eval_*seed{args.seed}*_combined.json")))

    if not targets:
        print("No matching result files found.")
        return 1

    print(f"{'DRY RUN -- ' if args.dry_run else ''}recomputing {len(targets)} file(s)\n")
    total_cells = 0
    for path in targets:
        changes = recompute_file(path, args.dry_run)
        if changes is None:
            continue
        rel = os.path.relpath(path)
        if not changes:
            print(f"  {rel}\n      no change")
            continue
        total_cells += len(changes)
        print(f"  {rel}")
        for bench, name, delta in changes:
            bits = []
            for field, (b, a) in delta.items():
                if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a:
                    bits.append(f"{field} {b:g} -> {a:g} ({b / a:.0f}x lower)")
                else:
                    bits.append(f"{field} {b} -> {a}")
            print(f"      {bench}/{name}")
            for bit in bits:
                print(f"          {bit}")

    print(f"\n{total_cells} cell(s) changed"
          + (" (nothing written -- dry run)" if args.dry_run else " and rewritten"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
