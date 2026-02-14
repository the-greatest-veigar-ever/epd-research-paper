#!/usr/bin/env python3
"""
Full Pipeline Evaluation for EPD Squad C - V2 IMPROVED
Phase 2: Verifying Reduced Polymorphism + One-Shot Prompting

Runs the COMPLETE pipeline: Squad A → Squad B → Squad C (Ghost Agent V2)
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root) # Ensure relative paths work

import pandas as pd
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import all squads
from src.watchers.agent import DetectionAgent
from src.brain.agent import IntelligenceAgent
# V2 IMPORT: Use the improved factory
from src.ghost_agents.agent_v2 import GhostAgentFactoryV2

# CONFIG
DATA_PATH = "data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
REPORT_OUTPUT_DIR = "experiments/phase_2_improvements/results" # Separate output dir

# Preset limits
LIMIT_LIGHTWEIGHT = 500
LIMIT_NORMAL = 2000
LIMIT_RESEARCH = 200
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
        
        # Advanced Metrics (Squad C)
        self.semantic_scores = []
        self.levenshtein_scores = []
        self.tool_correctness_scores = []
        
        # Squad C specific
        self.squad_c_results = []
        self.model_usage = {}
        # Per-model timing: { "model_name": {"init": [], "action": [], "ephemerality": []} }
        self.model_timings = {}
        
        # Ephemerality Overhead (Time to spin up/down vs execution)
        self.ephemerality_overhead = []
    
    def log_squad_a(self, duration: float, anomaly_count: int):
        self.processing_times["squad_a"].append(duration)
        self.total_anomalies += anomaly_count
    
    def log_squad_b(self, duration: float, plan_generated: bool):
        self.processing_times["squad_b"].append(duration)
        if plan_generated:
            self.total_plans += 1
    
    def log_squad_c(self, duration: float, result: Dict, model_used: str, prompt_metrics: Dict = None, init_duration: float = 0.0, action_duration: float = 0.0):
        self.processing_times["squad_c"].append(duration)
        self.squad_c_results.append(result)
        self.model_usage[model_used] = self.model_usage.get(model_used, 0) + 1
        
        # Per-model timing
        if model_used not in self.model_timings:
            self.model_timings[model_used] = {"init": [], "action": [], "ephemerality": []}
        self.model_timings[model_used]["init"].append(init_duration)
        self.model_timings[model_used]["action"].append(action_duration)
        
        # Ephemerality = Init Time (creation/born)
        self.ephemerality_overhead.append(init_duration)
        self.model_timings[model_used]["ephemerality"].append(init_duration)
        
        if prompt_metrics:
            self.semantic_scores.append(prompt_metrics.get("semantic_similarity", 0))
            self.levenshtein_scores.append(prompt_metrics.get("levenshtein", 0))
            self.tool_correctness_scores.append(1 if prompt_metrics.get("tool_correct") else 0)
        
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
                "tool_correctness_rate": float(np.mean(self.tool_correctness_scores)) if self.tool_correctness_scores else 0.0,
                "avg_semantic_similarity": float(np.mean(self.semantic_scores)) if self.semantic_scores else 0.0,
                "avg_levenshtein_distance": float(np.mean(self.levenshtein_scores)) if self.levenshtein_scores else 0.0,
                "avg_ephemerality_overhead": float(np.mean(self.ephemerality_overhead)) if self.ephemerality_overhead else 0.0
            },
            "model_distribution": self.model_usage,
            "per_model_latency": {
                model: {
                    "avg_init": float(np.mean(times["init"])) if times["init"] else 0.0,
                    "avg_action": float(np.mean(times["action"])) if times["action"] else 0.0,
                    "avg_total": float(np.mean(times["init"]) + np.mean(times["action"])) if times["init"] else 0.0
                }
                for model, times in self.model_timings.items()
            }
        }


def run_full_pipeline_v2(mode: str = "lightweight", baseline: bool = False, custom_limit: int = None):
    # Determine row limit
    if custom_limit:
        limit = custom_limit
    elif mode == "lightweight":
        limit = LIMIT_LIGHTWEIGHT
    elif mode == "research":
        limit = LIMIT_RESEARCH
    else:
        limit = LIMIT_NORMAL
    
    config_name = "baseline_v2" if baseline else "proposed_v2_improved"
    
    print("=" * 80)
    print(f"EPD FULL PIPELINE EVALUATION (V2) - {mode.upper()} MODE")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Row Limit: {limit}")
    print(f"Date: {datetime.now().isoformat()}")
    print()
    
    metrics = PipelineMetrics()
    
    # Initialize SBERT
    print("[Init] Loading Sentence Transformer...")
    sbert_model = None
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 
    except Exception as e:
        print(f"Warning: SBERT load failed ({e})")
    
    # ========== INITIALIZATION PHASE ==========
    print("[Phase 1] Initializing Squads...")
    
    # Squad A
    t_start = time.perf_counter()
    watcher = DetectionAgent("Watcher-Eval-V2")
    metrics.init_times["squad_a"] = time.perf_counter() - t_start
    print(f"  Squad A (Watcher): {metrics.init_times['squad_a']:.3f}s")
    
    # Squad B
    t_start = time.perf_counter()
    brain = IntelligenceAgent("Brain-Eval-V2")
    metrics.init_times["squad_b"] = time.perf_counter() - t_start
    print(f"  Squad B (Brain): {metrics.init_times['squad_b']:.3f}s")
    
    # Squad C
    t_start = time.perf_counter()
    GhostAgentFactoryV2._current_model_idx = 0
    metrics.init_times["squad_c"] = time.perf_counter() - t_start
    print(f"  Squad C (Ghost V2 Factory): {metrics.init_times['squad_c']:.3f}s")
    
    # ========== PROCESSING PHASE ==========
    print(f"[Phase 2] Processing Traffic ({limit} flows)...")
    
    # Load data
    try:
        chunk_iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)
        total_batches = limit // BATCH_SIZE
    except FileNotFoundError:
        print(f"Dataset not found: {DATA_PATH}")
        return

    detailed_results = []
    
    try:
        for batch_df in tqdm(chunk_iterator, total=total_batches, unit="batch", desc="Pipeline V2"):
            if metrics.total_flows >= limit:
                break
            
            batch_df.columns = batch_df.columns.str.strip()
            batch_size = len(batch_df)
            
            # --- SQUAD A ---
            t_a_start = time.perf_counter()
            alerts = watcher.monitor_traffic_batch(batch_df)
            t_a_duration = time.perf_counter() - t_a_start
            metrics.log_squad_a(t_a_duration, len(alerts))
            
            if alerts:
                for alert in alerts:
                    # --- SQUAD B ---
                    t_b_start = time.perf_counter()
                    plan = brain.analyze_alert(alert)
                    t_b_duration = time.perf_counter() - t_b_start
                    metrics.log_squad_b(t_b_duration, plan is not None)
                    
                    if plan:
                        # --- SQUAD C (V2) ---
                        t_c_start = time.perf_counter()
                        
                        # 1. Init
                        t_init_start = time.perf_counter()
                        base_instruction = f"Perform {plan['action']} on {plan['target']}"
                        
                        # V2 Creation
                        ghost = GhostAgentFactoryV2.create_agent(
                            base_instruction, 
                            rotate_model=not baseline
                        )
                        t_init_duration = time.perf_counter() - t_init_start
                        
                        # Capture props BEFORE execution (Fix for V2 cleanup)
                        model_used = ghost.model
                        mutated_prompt = ghost.prompt
                        
                        # 2. Action
                        t_act_start = time.perf_counter()
                        exec_result = ghost.execute_remediation(plan)
                        t_act_duration = time.perf_counter() - t_act_start

                        # Metrics
                        lev_dist = levenshtein_distance(base_instruction, mutated_prompt) if mutated_prompt else 0
                        sem_sim = 0.0
                        if sbert_model and mutated_prompt:
                             emb_base = sbert_model.encode([base_instruction])
                             emb_mutated = sbert_model.encode([mutated_prompt])
                             sem_sim = cosine_similarity(emb_base, emb_mutated)[0][0]
                        
                        tool_used = exec_result.get("tool_used", "")
                        is_tool_correct = "aws" in str(tool_used).lower() if tool_used else False
                        
                        prompt_metrics = {
                            "levenshtein": lev_dist,
                            "semantic_similarity": sem_sim,
                            "tool_correct": is_tool_correct
                        }
                        
                        t_c_duration = time.perf_counter() - t_c_start
                        metrics.log_squad_c(t_c_duration, exec_result, model_used, prompt_metrics, init_duration=t_init_duration, action_duration=t_act_duration)
                        
                        detailed_results.append({
                            "plan_action": plan.get("action"),
                            "model_used": model_used,
                            "execution_status": exec_result.get("status"),
                            "tool_used": exec_result.get("tool_used"),
                            "semantic_sim": float(sem_sim),
                            "latency": float(t_c_duration)
                        })
            
            metrics.total_flows += batch_size
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline error: {e}")
    
    # ========== REPORT ==========
    summary = metrics.get_summary()
    print(f"\n--- V2 Results ---")
    print(f"  Executions: {summary['executions_completed']}")
    print(f"  Tool Correctness: {summary['rates']['tool_correctness_rate']*100:.2f}%")
    print(f"  Avg Semantic Sim: {summary['rates']['avg_semantic_similarity']:.4f}")
    
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(REPORT_OUTPUT_DIR, f"{config_name}_{mode}_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump({"summary": summary, "details": detailed_results}, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lightweight")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_full_pipeline_v2(mode=args.mode, custom_limit=args.limit)
