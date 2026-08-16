"""
Merge Multi-Machine / Multi-Model Outputs.

Supports the "run different models on different machines" workflow:
benchmark_evaluator.py now writes each model's results into its own
self-contained folder (`output_dir/<model_key>/benchmark_eval_*_seed<N>.json`),
so different machines can each pull and run a different subset of models
(and possibly different seeds) with no filename collisions.

After copying/rsync-ing each machine's `report-output/ghost_agents/
benchmark_results/` tree into one place, run this script against all of
them. It scans every `<model_key>/benchmark_eval_*_seed<N>.json` file it
can find (recursively, across all given roots), regroups results by seed,
merges the per-model results for each seed into one combined per-seed
result (exactly what a single-machine run would have produced), and
reuses the same `aggregate_across_seeds()` / `_compute_summary()` used by
benchmark_evaluator.py to produce the final mean+/-std multi-seed summary.

Usage:
    python3 analysis/merge_model_outputs.py \
        /path/to/machineA/report-output/ghost_agents/benchmark_results \
        /path/to/machineB/report-output/ghost_agents/benchmark_results \
        --out report-output/ghost_agents/benchmark_results
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ghost_agents.approach_evaluation.benchmark_evaluator import (
    _compute_summary,
    aggregate_across_seeds,
)

EVAL_FILE_RE = re.compile(r"benchmark_eval_.*_seed(\d+)\.json$")


def find_eval_files(roots: List[str]) -> List[Path]:
    """
    Find per-model source files only. Deliberately excludes the
    "*_combined.json" convenience files _run_single_seed also writes at the
    output_dir root (merged from that machine's own model groups) -- those
    would double-count approaches already present in the per-model files.
    """
    files: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARNING] Root does not exist, skipping: {root}")
            continue
        files.extend(
            f for f in root_path.rglob("benchmark_eval_*.json")
            if not f.name.endswith("_combined.json")
        )
    return files


def merge(roots: List[str]) -> Dict[int, Dict[str, Any]]:
    """Group every discovered per-model, per-seed result file by seed and
    merge each seed's (benchmark, approach) result cells together."""
    by_seed: Dict[int, List[dict]] = defaultdict(list)
    seen_files = set()

    for f in find_eval_files(roots):
        m = EVAL_FILE_RE.search(f.name)
        if not m:
            continue
        # Avoid double-counting a model's own "_combined" convenience file
        # alongside its per-model source file if both are under the roots.
        if f.resolve() in seen_files:
            continue
        seen_files.add(f.resolve())

        seed = int(m.group(1))
        with open(f) as fh:
            by_seed[seed].append(json.load(fh))

    seed_results: Dict[int, Dict[str, Any]] = {}
    for seed, model_results_list in by_seed.items():
        combined: Dict[str, Any] = {
            "evaluation_id": f"merged_seed{seed}",
            "seed": seed,
            "benchmark_results": {},
        }
        for mr in model_results_list:
            for bench_name, bench_result in mr.get("benchmark_results", {}).items():
                slot = combined["benchmark_results"].setdefault(bench_name, {
                    "benchmark": bench_name,
                    "strategy": bench_result.get("strategy", ""),
                    "citation": bench_result.get("citation", ""),
                    "total_test_cases": bench_result.get("total_test_cases", 0),
                    "approaches": {},
                })
                slot["approaches"].update(bench_result.get("approaches", {}))
        combined["summary"] = _compute_summary(combined)
        seed_results[seed] = combined

    return seed_results


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-model, per-machine outputs into one multi-seed summary"
    )
    parser.add_argument(
        "roots", nargs="+",
        help="One or more report-output/.../benchmark_results roots to scan recursively "
             "(e.g. copies of each machine's output_dir).",
    )
    parser.add_argument("--out", default="report-output/ghost_agents/benchmark_results")
    args = parser.parse_args()

    seed_results = merge(args.roots)
    if not seed_results:
        print("[WARNING] No benchmark_eval_*_seed<N>.json files found under the given roots.")
        return

    print(f"Found results for seeds: {sorted(seed_results.keys())}")
    for seed, combined in sorted(seed_results.items()):
        n_cells = sum(len(b["approaches"]) for b in combined["benchmark_results"].values())
        print(f"  seed {seed}: {len(combined['benchmark_results'])} benchmarks, {n_cells} (benchmark, approach) result cells")

    multi_seed_summary = aggregate_across_seeds(seed_results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multi_seed_summary_merged.json"
    with open(out_file, "w") as f:
        json.dump(multi_seed_summary, f, indent=2, default=str)

    print(f"\nMerged multi-seed summary written to: {out_file}")
    print("Feed this into analysis/ablation_report.py and analysis/generate_latex_tables.py.")


if __name__ == "__main__":
    main()
