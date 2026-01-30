import pandas as pd
import time
import sys
import os
import datetime
import uuid
import argparse
from tqdm import tqdm
from src.ghost_agents.agent import GhostAgent, GhostAgentFactory
from src.watchers.agent import DetectionAgent
from src.brain.agent import IntelligenceAgent

# CONFIG
DATA_PATH = "ai/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
BATCH_SIZE = 100
MAX_ROWS = 200000 

class ReportGenerator:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.datetime.now()
        self.events = []
        self.stats = {
            "total_flows": 0,
            "anomalies": 0,
            "mitigations": 0,
            "start_time": self.start_time,
            "end_time": None
        }
        
    def log_event(self, event_type, details, score=None, mitigation=None, status=None):
        self.events.append({
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "score": score,
            "details": details,
            "mitigation": mitigation,
            "status": status
        })
        
    def save_report(self):
        self.stats["end_time"] = datetime.datetime.now()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        # Filename: EPD Report - Official - number - time - date.xlsx
        threat_count = len(self.events)
        time_str = self.start_time.strftime('%H-%M')
        date_str = self.start_time.strftime('%Y-%m-%d')
        xls_filename = f"report-output/all/EPD Report - Official - {threat_count} - {time_str} - {date_str}.xlsx"
        
        # 2. Excel Report (Detailed)
        if self.events:
            # Flatten detailed info
            flat_events = []
            for e in self.events:
                # Base info
                item = {
                    "Timestamp": e["timestamp"],
                    "Event Type": e["type"],
                    "AI Score": e["score"],
                    "Mitigation": e["mitigation"],
                    "Status": e["status"]
                }
                # Merge details (Traffic features like IP, Port, etc.)
                if isinstance(e["details"], dict):
                    item.update(e["details"])
                flat_events.append(item)
            
            df_report = pd.DataFrame(flat_events)
            df_report.to_excel(xls_filename, index=False)
            print(f"[REPORT] Detailed EXCEL report saved to: {xls_filename}")
        else:
            print("[REPORT] No events to save to Excel.")

def run_autonomous_mode(test_mode=False, custom_limit=None):
    print("\n=== EPD AUTONOMOUS SENTINEL STARTING ===")
    
    # Arg parser handling if needed, but simple boolean param is fine for internal call
    limit_rows = custom_limit if custom_limit else (1000 if test_mode else MAX_ROWS)
    
    print("Initializing Squads...")
    watcher = DetectionAgent("Watcher-Auto")
    brain = IntelligenceAgent("Brain-Auto")
    reporter = ReportGenerator()
    
    if not watcher.is_trained:
        print("Watcher model missing. Run training first.")
        return

    # 2. Stream Data
    print(f"Loading Traffic Stream: {DATA_PATH}")
    chunk_iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)
    
    start_time = time.time()
    
    # Calculate total expected batches for tqdm
    total_batches = limit_rows // BATCH_SIZE
    
    try:
        for batch_df in tqdm(chunk_iterator, total=total_batches, unit="batch", desc="EPD Sentinel Scan"):
            if reporter.stats["total_flows"] >= limit_rows:
                break
                
            # Clean batch columns
            batch_df.columns = batch_df.columns.str.strip()
            
            # --- SQUAD A: WATCHER SCAN ---
            alerts = watcher.monitor_traffic_batch(batch_df)
            
            if alerts:
                reporter.stats["anomalies"] += len(alerts)
                
                # Log ALL anomalies to Excel (not just first one)
                for alert in alerts:
                    print(f"\n[!] ANOMALY CONFIRMED (Batch {reporter.stats['total_flows'] // BATCH_SIZE})")
                    print(f"    Score: {alert['ai_score']:.4f}")
                    
                    # --- SQUAD B: BRAIN ANALYSIS ---
                    plan = brain.analyze_alert(alert)
                    
                    mitigation_cmd = "Pending Review"
                    mitigation_status = "NOT_EXECUTED"
                    
                    if plan:
                        # --- SQUAD C: GHOST EXECUTION ---
                        reporter.stats["mitigations"] += 1
                        
                        base_instruction = f"Perform {plan['action']} on {plan['target']}"
                        ghost = GhostAgentFactory.create_agent(base_instruction)
                        ghost.execute_remediation(plan)
                        
                        mitigation_cmd = plan['action']
                        mitigation_status = "EXECUTED"
                    
                    # LOG EVERY ANOMALY TO REPORT (with full details)
                    reporter.log_event(
                        "NET_ANOMALY", 
                        alert['details'], 
                        alert['ai_score'], 
                        mitigation_cmd,
                        mitigation_status
                    )
            
            reporter.stats["total_flows"] += len(batch_df)
                
    except KeyboardInterrupt:
        print("\n[STOP] Manual efficient shutdown.")
    except Exception as e:
        print(f"\n[ERROR] Stream crashed: {e}")
        
    print("\n\n=== MISSION REPORT ===")
    print(f"Total Flows Scanned: {reporter.stats['total_flows']}")
    print(f"Total Anomalies:     {reporter.stats['anomalies']}")
    print(f"Total Mitigations:   {reporter.stats['mitigations']}")
    
    reporter.save_report()
    print("System shutting down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run a short test burst")
    parser.add_argument("--rows", type=int, help="Limit number of rows to process", default=None)
    args = parser.parse_args()
    
    # Determine limit
    limit = MAX_ROWS
    if args.test_mode:
        limit = 1000
    if args.rows:
        limit = args.rows
        
    run_autonomous_mode(test_mode=args.test_mode, custom_limit=limit)
