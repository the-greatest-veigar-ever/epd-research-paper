#!/usr/bin/env python3
"""
Full Pipeline Evaluation for EPD Squad C

Runs the COMPLETE pipeline: Squad A → Squad B → Squad C
With comprehensive timing and metrics collection.

Two modes:
- Lightweight: 100-500 flows, quick validation
- Normal: 1000-2000 flows, full evaluation
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

# Import all squads
from src.watchers.agent import DetectionAgent
from src.brain.agent import IntelligenceAgent
from src.ghost_agents.agent import GhostAgentFactory

# CONFIG
DATA_PATH = "ai/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
INJECTION_TEST_PATH = "ai/data/ghost_agents/injection_test_cases.jsonl"
REPORT_OUTPUT_DIR = "report-output/ghost_agents/full_pipeline"

# Preset limits
LIMIT_LIGHTWEIGHT = 500
LIMIT_NORMAL = 2000
BATCH_SIZE = 100


class PipelineMetrics:
    """Collects and aggregates metrics across the full pipeline."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Timing
        self.init_times = {"squad_a": 0, "squad_b": 0, "squad_c": 0}
        self.processing_times = {"squad_a": [], "squad_b": [], "squad_c": []}
        
        # Counts
        self.total_flows = 0
        self.total_anomalies = 0
        self.total_plans = 0
        self.total_executions = 0
        
        # Squad C specific
        self.squad_c_results = []
        self.model_usage = {}
        
        # Injection resistance (if injection tests included)
        self.injection_tests = {"total": 0, "safe": 0, "unsafe": 0}
    
    def log_squad_a(self, duration: float, anomaly_count: int):
        self.processing_times["squad_a"].append(duration)
        self.total_anomalies += anomaly_count
    
    def log_squad_b(self, duration: float, plan_generated: bool):
        self.processing_times["squad_b"].append(duration)
        if plan_generated:
            self.total_plans += 1
    
    def log_squad_c(self, duration: float, result: Dict, model_used: str):
        self.processing_times["squad_c"].append(duration)
        self.squad_c_results.append(result)
        self.model_usage[model_used] = self.model_usage.get(model_used, 0) + 1
        
        if result.get("status") in ["success", "simulated_success"]:
            self.total_executions += 1
    
    def get_summary(self) -> Dict:
        """Generate summary metrics."""
        return {
            "flows_processed": self.total_flows,
            "anomalies_detected": self.total_anomalies,
            "plans_generated": self.total_plans,
            "executions_completed": self.total_executions,
            "timing": {
                "squad_a": {
                    "init": self.init_times["squad_a"],
                    "avg_processing": sum(self.processing_times["squad_a"]) / max(len(self.processing_times["squad_a"]), 1),
                    "total_processing": sum(self.processing_times["squad_a"]),
                },
                "squad_b": {
                    "init": self.init_times["squad_b"],
                    "avg_processing": sum(self.processing_times["squad_b"]) / max(len(self.processing_times["squad_b"]), 1),
                    "total_processing": sum(self.processing_times["squad_b"]),
                },
                "squad_c": {
                    "init": self.init_times["squad_c"],
                    "avg_processing": sum(self.processing_times["squad_c"]) / max(len(self.processing_times["squad_c"]), 1),
                    "total_processing": sum(self.processing_times["squad_c"]),
                },
                "total_init": sum(self.init_times.values()),
                "total_processing": (
                    sum(self.processing_times["squad_a"]) +
                    sum(self.processing_times["squad_b"]) +
                    sum(self.processing_times["squad_c"])
                ),
            },
            "rates": {
                "anomaly_rate": self.total_anomalies / max(self.total_flows, 1),
                "plan_rate": self.total_plans / max(self.total_anomalies, 1),
                "execution_success_rate": self.total_executions / max(self.total_plans, 1),
            },
            "model_distribution": self.model_usage,
        }


def run_full_pipeline(mode: str = "lightweight", baseline: bool = False, custom_limit: int = None):
    """
    Run full pipeline evaluation.
    
    Args:
        mode: "lightweight" (500 flows) or "normal" (2000 flows)
        baseline: If True, disable model rotation (baseline comparison)
        custom_limit: Override default row limit
    """
    # Determine row limit
    if custom_limit:
        limit = custom_limit
    elif mode == "lightweight":
        limit = LIMIT_LIGHTWEIGHT
    else:
        limit = LIMIT_NORMAL
    
    config_name = "baseline" if baseline else "proposed"
    
    print("=" * 80)
    print(f"EPD FULL PIPELINE EVALUATION - {mode.upper()} MODE")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Model Rotation: {not baseline}")
    print(f"Row Limit: {limit}")
    print(f"Date: {datetime.now().isoformat()}")
    print()
    
    metrics = PipelineMetrics()
    
    # ========== INITIALIZATION PHASE ==========
    print("[Phase 1] Initializing Squads...")
    
    # Squad A
    t_start = time.perf_counter()
    watcher = DetectionAgent("Watcher-Eval")
    metrics.init_times["squad_a"] = time.perf_counter() - t_start
    print(f"  Squad A (Watcher): {metrics.init_times['squad_a']:.3f}s")
    
    if not watcher.is_trained:
        print("❌ Error: Watcher model not found. Cannot proceed.")
        return None
    
    # Squad B
    t_start = time.perf_counter()
    brain = IntelligenceAgent("Brain-Eval")
    metrics.init_times["squad_b"] = time.perf_counter() - t_start
    print(f"  Squad B (Brain): {metrics.init_times['squad_b']:.3f}s")
    
    # Squad C (just reset rotation index)
    t_start = time.perf_counter()
    GhostAgentFactory._current_model_idx = 0
    metrics.init_times["squad_c"] = time.perf_counter() - t_start
    print(f"  Squad C (Ghost Factory): {metrics.init_times['squad_c']:.3f}s")
    
    total_init = sum(metrics.init_times.values())
    print(f"  TOTAL T_init: {total_init:.3f}s")
    print()
    
    # ========== PROCESSING PHASE ==========
    print(f"[Phase 2] Processing Traffic ({limit} flows)...")
    print(f"  Data: {DATA_PATH}")
    
    # Load data
    chunk_iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)
    total_batches = limit // BATCH_SIZE
    
    detailed_results = []
    
    try:
        for batch_df in tqdm(chunk_iterator, total=total_batches, unit="batch", desc="Pipeline"):
            if metrics.total_flows >= limit:
                break
            
            batch_df.columns = batch_df.columns.str.strip()
            batch_size = len(batch_df)
            
            # --- SQUAD A: Detection ---
            t_a_start = time.perf_counter()
            alerts = watcher.monitor_traffic_batch(batch_df)
            t_a_duration = time.perf_counter() - t_a_start
            metrics.log_squad_a(t_a_duration, len(alerts))
            
            if alerts:
                for alert in alerts:
                    # --- SQUAD B: Analysis ---
                    t_b_start = time.perf_counter()
                    plan = brain.analyze_alert(alert)
                    t_b_duration = time.perf_counter() - t_b_start
                    metrics.log_squad_b(t_b_duration, plan is not None)
                    
                    if plan:
                        # --- SQUAD C: Execution ---
                        t_c_start = time.perf_counter()
                        
                        base_instruction = f"Perform {plan['action']} on {plan['target']}"
                        ghost = GhostAgentFactory.create_agent(
                            base_instruction, 
                            rotate_model=not baseline
                        )
                        
                        # Capture model BEFORE execution (self_destruct wipes it)
                        model_used = ghost.model
                        
                        exec_result = ghost.execute_remediation(plan)
                        
                        t_c_duration = time.perf_counter() - t_c_start
                        metrics.log_squad_c(t_c_duration, exec_result, model_used)
                        
                        # Record detailed result
                        detailed_results.append({
                            "alert_type": alert.get("type"),
                            "ai_score": alert.get("ai_score"),
                            "plan_action": plan.get("action"),
                            "model_used": model_used,
                            "timing": {
                                "squad_a": t_a_duration,
                                "squad_b": t_b_duration,
                                "squad_c": t_c_duration,
                                "total": t_a_duration + t_b_duration + t_c_duration
                            },
                            "execution_status": exec_result.get("status"),
                            "tool_used": exec_result.get("tool_used")
                        })
            
            metrics.total_flows += batch_size
    
    except KeyboardInterrupt:
        print("\n[STOP] Evaluation interrupted.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline error: {e}")
    
    # ========== RESULTS PHASE ==========
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    summary = metrics.get_summary()
    
    print(f"\n--- Pipeline Statistics ---")
    print(f"  Flows Processed:     {summary['flows_processed']}")
    print(f"  Anomalies Detected:  {summary['anomalies_detected']}")
    print(f"  Plans Generated:     {summary['plans_generated']}")
    print(f"  Executions Complete: {summary['executions_completed']}")
    
    print(f"\n--- Timing Metrics ---")
    print(f"  T_init (total):      {summary['timing']['total_init']:.3f}s")
    print(f"  T_proc (total):      {summary['timing']['total_processing']:.3f}s")
    print(f"  T_total:             {summary['timing']['total_init'] + summary['timing']['total_processing']:.3f}s")
    
    print(f"\n--- Per-Squad Timing ---")
    print(f"  Squad A avg:         {summary['timing']['squad_a']['avg_processing']:.3f}s")
    print(f"  Squad B avg:         {summary['timing']['squad_b']['avg_processing']:.3f}s")
    print(f"  Squad C avg:         {summary['timing']['squad_c']['avg_processing']:.3f}s")
    
    print(f"\n--- Success Rates ---")
    print(f"  Anomaly Rate:        {summary['rates']['anomaly_rate']*100:.2f}%")
    print(f"  Plan Rate:           {summary['rates']['plan_rate']*100:.2f}%")
    print(f"  Execution Success:   {summary['rates']['execution_success_rate']*100:.2f}%")
    
    if summary['model_distribution']:
        print(f"\n--- Model Distribution ---")
        for model, count in summary['model_distribution'].items():
            pct = count / max(summary['executions_completed'], 1) * 100
            print(f"  {model}: {count} ({pct:.1f}%)")
    
    # ========== SAVE RESULTS ==========
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(REPORT_OUTPUT_DIR, f"{config_name}_{mode}_{timestamp}.json")
    
    final_report = {
        "evaluation_type": "full_pipeline",
        "mode": mode,
        "configuration": config_name,
        "model_rotation": not baseline,
        "timestamp": datetime.now().isoformat(),
        "row_limit": limit,
        "summary": summary,
        "detailed_results": detailed_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPD Full Pipeline Evaluation")
    parser.add_argument("--mode", choices=["lightweight", "normal"], default="lightweight",
                        help="Evaluation mode: lightweight (500 flows) or normal (2000 flows)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no model rotation)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Custom row limit (overrides mode default)")
    args = parser.parse_args()
    
    run_full_pipeline(mode=args.mode, baseline=args.baseline, custom_limit=args.limit)
