#!/usr/bin/env python3
import sys
import os
import json
import gc
import pandas as pd
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Try imports
try:
    from src.watchers.agent import DetectionAgent
    from src.brain.agent import IntelligenceAgent
    from src.brain.ollama_agent import OllamaIntelligenceAgent
    from src.brain.consensus import SquadBConsensus
except ImportError:
    # Path fallback
    sys.path.insert(0, os.getcwd())
    from src.watchers.agent import DetectionAgent
    from src.brain.agent import IntelligenceAgent
    from src.brain.ollama_agent import OllamaIntelligenceAgent
    from src.brain.consensus import SquadBConsensus

def run_cic_integration_test():
    print("=== Integration Test: Real CIC Data -> Squad A -> Squad B ===\n")
    
    # 1. Load Real Data
    data_path = "ai/data/processed_watcher_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
        
    print(f"[1] Loading Data from {data_path}...")
    
    # Multi-attack sampling: scan ALL chunks and collect 1 flow per unique attack type
    chunk_size = 100000
    attack_samples = {}  # label -> DataFrame row
    benign_pool = pd.DataFrame()
    
    # Specific chunks where we know attacks exist (from dataset scan)
    # Bot: Ch9, DoS: Ch30, DDoS: Ch40/52, BruteForce: Ch19/63/65, SQL: Ch19
    target_chunks = {9, 19, 30, 40, 52, 63, 65}
    
    print(f"Scanning target chunks {sorted(target_chunks)} for diverse attack types...")
    
    try:
        with pd.read_csv(data_path, chunksize=chunk_size) as reader:
            for i, chunk in enumerate(reader):
                if i not in target_chunks:
                    if i > max(target_chunks):
                        break  # Past all target chunks
                    continue
                
                # Collect attacks we haven't seen yet
                non_benign = chunk[chunk['Label'] != 'Benign']
                for label in non_benign['Label'].unique():
                    if label not in attack_samples:
                        rows = non_benign[non_benign['Label'] == label]
                        attack_samples[label] = rows.sample(n=min(520, len(rows)), random_state=None)
                        print(f"   ✅ Sampled {len(attack_samples[label])} flows of '{label}' from chunk {i}")
                
                # Collect benign flows for mixing
                if len(benign_pool) < 150:
                    benign_in_chunk = chunk[chunk['Label'] == 'Benign']
                    if not benign_in_chunk.empty:
                        benign_pool = pd.concat([benign_pool, benign_in_chunk.sample(n=min(25, len(benign_in_chunk)), random_state=None)])
        
        if not attack_samples:
            print("❌ No attacks found in target chunks.")
            return
        
        # Combine all attack samples + benign
        all_attack_df = pd.concat(attack_samples.values())
        benign_df = benign_pool.head(3)
        batch_df = pd.concat([all_attack_df, benign_df]).reset_index(drop=True)
        
        print(f"\n📊 Final Batch: {len(all_attack_df)} Malicious + {len(benign_df)} Benign = {len(batch_df)} flows")
        print(f"   Attack Types: {list(attack_samples.keys())}")
        all_ports = batch_df['Dst Port'].unique()
        all_protocols = batch_df['Protocol'].unique()
        print(f"   Unique Dst Ports: {len(all_ports)} → {sorted(all_ports)[:10]}")
        print(f"   Unique Protocols: {sorted(all_protocols)}")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback; traceback.print_exc()
        return
    
    # Memory Optimization: Free CSV reader memory before loading LLM
    gc.collect()
    print("[Memory] Freed CSV chunk memory before LLM loading.")

    # 2. Initialize Agents
    print("\n[2] Initializing Multi-Agent Squad B (SentinelNet Consensus)...")
    squad_a = DetectionAgent("Watcher-CIC-Test")
    
    # Squad B: 3 agents with consensus
    agent_1 = IntelligenceAgent("Brain-Phi2-QLoRA")
    agent_2 = OllamaIntelligenceAgent("Brain-Phi3-Mini", model="phi3:mini")
    agent_3 = OllamaIntelligenceAgent("Brain-Llama3.2", model="llama3.2:3b")
    squad_b = SquadBConsensus([agent_1, agent_2, agent_3])
    
    # DEBUG: Check Raw Probabilities
    print("\n[DEBUG] Checking Raw XGBoost Probabilities...")
    try:
        preds, probs = squad_a.watcher.predict_batch(batch_df)
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            label = batch_df.iloc[i]['Label']
            print(f"  Row {i} [{label}]: Pred={pred}, Prob={prob:.4f}")
    except Exception as e:
        print(f"  Debug failed: {e}")

    # 3. Squad A Processing
    print("\n[3] Squad A Scanning Traffic Batch (XGBoost Inference)...")
    # Monitor Traffic Batch expects a DataFrame
    alerts = squad_a.monitor_traffic_batch(batch_df)
    print(f"Squad A generated {len(alerts)} alerts from {len(batch_df)} flows.")
    
    # 4. Squad B Processing
    print("\n[4] Squad B Analyzing Alerts & Deciding Remediation...")
    results = []
    
    # Timestamped output file — unique per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cic_integration_results_{timestamp}.json"
    print(f"Results will be saved to: {output_file}")
    SAVE_EVERY = 20  # Incremental save interval
    
    for idx, alert in enumerate(alerts):
        event = alert['details'].get('event_name', 'Unknown')
        print(f"\n--- Processing Alert {idx+1}/{len(alerts)}: {event} ---")
        
        remediation_plan = squad_b.analyze_alert(alert)
        gc.collect()  # Free intermediate tensors between inferences
        
        if remediation_plan:
             # STIX object is already comprehensive
             results.append(remediation_plan)
             
             # Extract action for printing (from STIX custom field)
             action = remediation_plan.get('x_epd_action', 'UNKNOWN')
             reason = remediation_plan.get('x_epd_reason', 'N/A')
             openc2 = remediation_plan.get('x_epd_openc2', {})
             
             print(f"✅ PLAN: {action}")
             print(f"   Reason: {reason}")
             print(f"   OpenC2 Command: {json.dumps(openc2, indent=2)}")
        else:
             print("❌ No Plan Generated")
        
        # Incremental save every SAVE_EVERY alerts
        if (idx + 1) % SAVE_EVERY == 0 and results:
            _save_results(results, output_file, len(batch_df), len(alerts), squad_b)
            print(f"💾 Incremental save: {len(results)} results → {output_file}")

    # 5. Final Save
    _save_results(results, output_file, len(batch_df), len(alerts), squad_b)
    print(f"\n=== Detailed STIX 2.1 Results saved to {output_file} ===")
    print(f"    Total flows: {len(batch_df)} | Alerts: {len(alerts)} | Plans: {len(results)}")
    
    # Print consensus stats
    stats = squad_b.get_stats()
    print(f"\n=== Consensus Statistics ===")
    print(f"    Decision Methods: {stats['decision_methods']}")
    print(f"    Final Credits: {stats['final_credits']}")


def _save_results(results, output_file, total_flows, total_alerts, squad_b=None):
    """Save results with summary metadata."""
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_flows": total_flows,
            "total_alerts": total_alerts,
            "total_plans": len(results),
        },
        "stix_objects": results
    }
    if squad_b and hasattr(squad_b, 'get_stats'):
        output["consensus_stats"] = squad_b.get_stats()
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    run_cic_integration_test()
