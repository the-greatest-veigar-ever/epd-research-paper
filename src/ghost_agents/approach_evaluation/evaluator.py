"""
Approach Evaluator

Core engine that runs the integration test dataset through each approach
and collects comparative metrics: initialization time, processing time,
ASR, TSR, and PASS@1.
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from src.ghost_agents.approach_evaluation.approaches import (
    Approach,
    ALL_APPROACHES,
)


class ApproachEvaluator:
    """
    Evaluates one or more approaches against a dataset of remediation plans
    extracted from integration test results.
    """

    def __init__(
        self,
        dataset_path: str,
        limit: Optional[int] = None,
        output_dir: str = "report-output/ghost_agents/approach_comparison",
    ):
        self.dataset_path = dataset_path
        self.limit = limit
        self.output_dir = output_dir
        self.plans: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> int:
        """
        Load remediation plans from integration test JSON file(s).

        Supports:
        - A single JSON file path
        - A directory containing multiple JSON files (all are merged)

        Each JSON must have a 'stix_objects' list with x_epd_action / x_epd_target.
        Plans are deduplicated by STIX ID across files.

        Returns:
            int: Number of plans loaded.
        """
        json_files = []

        if os.path.isdir(self.dataset_path):
            # Load ALL .json files from the directory
            print(f"[Evaluator] Loading all JSON files from directory: {self.dataset_path}")
            for fname in sorted(os.listdir(self.dataset_path)):
                if fname.endswith(".json") and not fname.startswith("._"):
                    json_files.append(os.path.join(self.dataset_path, fname))
            print(f"[Evaluator] Found {len(json_files)} JSON files.")
        elif os.path.isfile(self.dataset_path):
            json_files = [self.dataset_path]
            print(f"[Evaluator] Loading single dataset file: {self.dataset_path}")
        else:
            print(f"[Evaluator] ERROR: Path not found: {self.dataset_path}")
            return 0

        seen_ids = set()
        for fpath in json_files:
            print(f"  → Loading {os.path.basename(fpath)}...")
            with open(fpath, "r") as f:
                data = json.load(f)

            # Support both the raw CIC output (dict with 'stix_objects') 
            # and our extracted clean/mixed datasets (flat list)
            if isinstance(data, dict):
                stix_objects = data.get("stix_objects", [])
            elif isinstance(data, list):
                stix_objects = data
            else:
                stix_objects = []
                
            added = 0
            for obj in stix_objects:
                # Handle both raw stix format (x_epd_action) and our cleaned format (action)
                action = obj.get("action") or obj.get("x_epd_action")
                target = obj.get("target") or obj.get("x_epd_target")
                stix_id = obj.get("id", "")
                score = obj.get("score") or obj.get("x_epd_score", 0.0)

                if not (action and target):
                    continue

                # Deduplicate by STIX ID across files
                if stix_id and stix_id in seen_ids:
                    continue
                if stix_id:
                    seen_ids.add(stix_id)

                self.plans.append({
                    "action": action,
                    "target": target,
                    "score": score,
                    "stix_id": stix_id,
                    "source_file": os.path.basename(fpath),
                })
                added += 1
            print(f"    Added {added} plans (total so far: {len(self.plans)})")

        if self.limit and self.limit < len(self.plans):
            self.plans = self.plans[: self.limit]

        print(f"[Evaluator] Total plans loaded: {len(self.plans)}")
        return len(self.plans)

    # ------------------------------------------------------------------
    # Single approach evaluation
    # ------------------------------------------------------------------

    def evaluate_approach(self, approach: Approach) -> Dict[str, Any]:
        """
        Run all plans through a single approach and collect metrics.

        Returns:
            dict with summary metrics and detailed per-plan results.
        """
        print(f"\n{'=' * 70}")
        print(f"  EVALUATING: {approach.name}")
        print(f"  Models: {approach.models}")
        print(f"  Suicide Mode: {approach.suicide_mode}")
        print(f"  Plans: {len(self.plans)}")
        print(f"{'=' * 70}")

        # --- Initialization ---
        print(f"\n[{approach.name}] Phase 1: Initialization")
        init_time = approach.initialize()
        print(f"[{approach.name}] Initialization done: {init_time:.3f}s")

        # --- Processing ---
        print(f"[{approach.name}] Phase 2: Processing {len(self.plans)} plans")

        detailed_results = []
        total_success = 0
        total_tool_correct = 0
        total_task_success = 0
        model_usage: Dict[str, int] = {}
        init_times: List[float] = []
        proc_times: List[float] = []

        interrupted = False
        plans_processed = 0
        try:
            for plan in tqdm(self.plans, desc=approach.name, unit="plan"):
                result = approach.execute_plan(plan)
                plans_processed += 1

                # Track timing
                init_times.append(result.get("init_time", 0.0))
                proc_times.append(result.get("processing_time", 0.0))

                # Track model usage
                model_used = result.get("model_used", "unknown")
                model_usage[model_used] = model_usage.get(model_used, 0) + 1

                # Determine success
                is_success = result["status"] in ("success", "simulated_success")

                # Tool correctness — check if a valid CLI tool was generated
                tool_used = result.get("tool_used", "")
                is_tool_correct = bool(tool_used and tool_used != "unknown")

                is_task_success = is_success and is_tool_correct

                if is_success:
                    total_success += 1
                if is_tool_correct:
                    total_tool_correct += 1
                if is_task_success:
                    total_task_success += 1

                detailed_results.append({
                    "plan_action": plan["action"],
                    "plan_target": plan["target"][:40],
                    "plan_score": plan["score"],
                    "model_used": model_used,
                    "status": result["status"],
                    "tool_used": tool_used,
                    "command": (result.get("command") or "")[:100],
                    "init_time": result.get("init_time", 0.0),
                    "processing_time": result.get("processing_time", 0.0),
                    "is_success": is_success,
                    "is_tool_correct": is_tool_correct,
                    "error": result.get("error"),
                })
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[{approach.name}] Interrupted! Saving partial results ({plans_processed}/{len(self.plans)} plans)...")

        # --- Teardown ---
        approach.teardown()

        # --- Compute metrics (use plans_processed for accurate rates) ---
        total_plans = plans_processed or 1

        asr = total_success / total_plans
        tsr = total_task_success / total_plans
        pass_at_1 = asr  # single-pass context

        avg_init = float(np.mean(init_times)) if init_times else 0.0
        avg_proc = float(np.mean(proc_times)) if proc_times else 0.0
        total_proc = float(np.sum(proc_times)) if proc_times else 0.0
        total_init_sum = float(np.sum(init_times)) if init_times else 0.0

        summary = {
            "approach": approach.name,
            "models": approach.models,
            "suicide_mode": approach.suicide_mode,
            "total_plans_in_dataset": len(self.plans),
            "plans_processed": plans_processed,
            "interrupted": interrupted,
            "initialization_time": init_time,
            "per_plan_init_time_avg": avg_init,
            "per_plan_init_time_total": total_init_sum,
            "processing_time_avg": avg_proc,
            "processing_time_total": total_proc,
            "asr": asr,
            "tsr": tsr,
            "pass_at_1": pass_at_1,
            "tool_correctness_rate": total_tool_correct / total_plans,
            "model_distribution": model_usage,
        }

        # Print summary
        status_tag = " (PARTIAL)" if interrupted else ""
        print(f"\n--- Results: {approach.name}{status_tag} ---")
        print(f"  Plans Processed:        {plans_processed}/{len(self.plans)}")
        print(f"  Initialization Time:    {init_time:.3f}s")
        print(f"  Avg Per-Plan Init Time: {avg_init:.3f}s")
        print(f"  Avg Processing Time:    {avg_proc:.3f}s")
        print(f"  Total Processing Time:  {total_proc:.3f}s")
        print(f"  ASR:                    {asr * 100:.2f}%")
        print(f"  TSR:                    {tsr * 100:.2f}%")
        print(f"  PASS@1:                 {pass_at_1 * 100:.2f}%")
        print(f"  Model Distribution:     {model_usage}")

        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "interrupted": interrupted,
        }

    # ------------------------------------------------------------------
    # Full comparative evaluation
    # ------------------------------------------------------------------

    def run_all(
        self, approach_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation for all (or selected) approaches and save results.

        Args:
            approach_names: List of approach names to run, or None for all.

        Returns:
            dict with comparison report.
        """
        if not self.plans:
            self.load_dataset()

        if approach_names is None:
            approach_names = list(ALL_APPROACHES.keys())

        results = {}
        global_interrupted = False
        try:
            for name in approach_names:
                cls = ALL_APPROACHES.get(name)
                if cls is None:
                    print(f"[Evaluator] Unknown approach: {name}, skipping.")
                    continue
                approach = cls()
                results[name] = self.evaluate_approach(approach)
                # If this approach was interrupted, stop remaining approaches
                if results[name].get("interrupted"):
                    global_interrupted = True
                    print(f"\n[Evaluator] Interrupted during {name}. Skipping remaining approaches.")
                    break
        except KeyboardInterrupt:
            global_interrupted = True
            print(f"\n[Evaluator] Interrupted between approaches. Saving collected results...")

        if not results:
            print("[Evaluator] No results collected. Nothing to save.")
            return {}

        # --- Comparison Table ---
        self._print_comparison_table(results)

        # --- Save reports ---
        report = self._save_reports(results, global_interrupted)
        return report

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------

    def _save_reports(self, results: Dict[str, Dict], interrupted: bool = False) -> Dict:
        """Save both detailed and summary-only JSON reports."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_suffix = "_partial" if interrupted else ""

        # 1. Detailed report (full per-plan results)
        detailed_report = {
            "evaluation_type": "approach_comparison",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_path,
            "total_plans_in_dataset": len(self.plans),
            "interrupted": interrupted,
            "limit": self.limit,
            "approaches": {
                name: res["summary"] for name, res in results.items()
            },
            "detailed_results": {
                name: res["detailed_results"] for name, res in results.items()
            },
        }

        detailed_path = os.path.join(
            self.output_dir, f"comparison_detailed_{timestamp}{status_suffix}.json"
        )
        with open(detailed_path, "w") as f:
            json.dump(detailed_report, f, indent=2)
        print(f"\n✅ Detailed report saved to: {detailed_path}")

        # 2. Summary-only report (compact, for quick comparison)
        summary_report = {
            "evaluation_type": "approach_comparison_summary",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_path,
            "total_plans_in_dataset": len(self.plans),
            "interrupted": interrupted,
            "comparison": {}
        }

        for name, res in results.items():
            s = res["summary"]
            total_init_display = s["initialization_time"] + s["per_plan_init_time_total"]
            summary_report["comparison"][name] = {
                "models": s["models"],
                "suicide_mode": s["suicide_mode"],
                "plans_processed": s["plans_processed"],
                "initialization_time_s": round(s["initialization_time"], 3),
                "per_plan_init_time_avg_s": round(s["per_plan_init_time_avg"], 3),
                "total_init_overhead_s": round(total_init_display, 3),
                "avg_processing_time_s": round(s["processing_time_avg"], 3),
                "total_processing_time_s": round(s["processing_time_total"], 3),
                "asr_pct": round(s["asr"] * 100, 2),
                "tsr_pct": round(s["tsr"] * 100, 2),
                "pass_at_1_pct": round(s["pass_at_1"] * 100, 2),
                "tool_correctness_pct": round(s["tool_correctness_rate"] * 100, 2),
                "model_distribution": s["model_distribution"],
            }

        summary_path = os.path.join(
            self.output_dir, f"comparison_summary_{timestamp}{status_suffix}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary_report, f, indent=2)
        print(f"✅ Summary report saved to: {summary_path}")

        return detailed_report

    # ------------------------------------------------------------------
    # Pretty-print comparison table
    # ------------------------------------------------------------------

    def _print_comparison_table(self, results: Dict[str, Dict]):
        """Print a formatted comparison table across all evaluated approaches."""
        print("\n")
        print("=" * 100)
        print("  APPROACH COMPARISON")
        print("=" * 100)

        header = (
            f"{'Approach':<25} | "
            f"{'Plans':>6} | "
            f"{'One-time Init':>14} | "
            f"{'Avg Plan Init':>14} | "
            f"{'Avg Proc':>10} | "
            f"{'ASR':>7} | "
            f"{'TSR':>7} | "
            f"{'PASS@1':>7}"
        )
        print(header)
        print("-" * 120)

        for name, res in results.items():
            s = res["summary"]
            row = (
                f"{s['approach']:<25} | "
                f"{s['plans_processed']:>6} | "
                f"{s['initialization_time']:>13.3f}s | "
                f"{s['per_plan_init_time_avg']:>13.3f}s | "
                f"{s['processing_time_avg']:>9.3f}s | "
                f"{s['asr'] * 100:>6.2f}% | "
                f"{s['tsr'] * 100:>6.2f}% | "
                f"{s['pass_at_1'] * 100:>6.2f}%"
            )
            print(row)

        print("=" * 120)
        print()
