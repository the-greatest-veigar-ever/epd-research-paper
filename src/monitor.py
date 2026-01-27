import time
import random
from typing import Dict, Any, List, Optional
from src.watcher.ml_watcher.watcher_xgboost import XGBoostWatcher

class DetectionAgent:
    """
    Group 1: Persistent Detection Agents (The Watchers)
    Baseline: MALCDF detection framework.
    Status: Persistent.
    
    Task: Continuously monitor Cloud Logs (GuardDuty, CloudTrail) for anomalies.
    """
    def __init__(self, agent_id: str = "watcher-01"):
        self.agent_id = agent_id
        self.status = "ACTIVE"
        try:
            self.watcher = XGBoostWatcher()
        except Exception as e:
            print(f"[{self.agent_id}] Warning: Could not initialize XGBoostWatcher: {e}")
            self.watcher = None
        print(f"[{self.agent_id}] Initialized and monitoring streams.")

    @property
    def is_trained(self):
        return self.watcher is not None

    def monitor_traffic_batch(self, df_batch: Any) -> List[Dict[str, Any]]:
        """
        Scans a batch of traffic (DataFrame) using the XGBoost Model.
        Restored for compatibility with Autonomous Runner.
        """
        alerts = []
        if not self.watcher:
            return []
            
        try:
            current_time = time.time()
            preds, probs = self.watcher.predict_batch(df_batch)
            
            # Find anomalies (assuming 1 = Malicious, 0 = Benign for XGBoost)
            # Or if it's Isolation Forest compatible (-1 vs 1)
            # Let's assume standard binary classification 1 = Attack
            anomalies_indices = [i for i, x in enumerate(preds) if x == 1]
            
            if anomalies_indices:
                 print(f"[{self.agent_id}] BATCH: Found {len(anomalies_indices)} threats in {len(df_batch)} flows.")
                 
                 for idx in anomalies_indices:
                     if len(alerts) < 5: # Throttling for demo console
                         row = df_batch.iloc[idx].to_dict()
                         
                         # Construct Alert for Brain
                         label = row.get("Label", "Network Anomaly")
                         row["event_name"] = label
                         row["target"] = "NetworkInterface"
                         
                         alert = {
                            "source": self.agent_id,
                            "timestamp": current_time,
                            "type": "NET_FLOW_ANOMALY",
                            "ai_score": float(probs[idx]),
                            "details": row
                        }
                         alerts.append(alert)
                     else:
                         break
        except Exception as e:
            print(f"[{self.agent_id}] Batch Scan Error: {e}")
            
        return alerts

    def monitor_logs(self, mock_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulates scanning logs for threats using XGBoost.
        """
        alerts = []
        print(f"[{self.agent_id}] Scanning {len(mock_logs)} log entries with XGBoost...")
        
        for log in mock_logs:
            # Prepare log for XGBoostWatcher if it's a simulated attack log
            # The simulator uses a different format, so we map it or handle both
            
            is_malicious = False
            risk_score = 0.0
            
            if self.watcher and "attack_name" in log:
                # Use XGBoost for logs that fit the expected format
                prediction, risk_score, inference_time = self.watcher.predict(log)
                is_malicious = (prediction == 1)
                if is_malicious:
                     print(f"[{self.agent_id}][ML] Inference Time: {inference_time:.4f} ms")
            else:
                # Fallback to rule-based for internal mock logs
                if log.get("severity") == "HIGH":
                    is_malicious = True
                    risk_score = 0.95 # Mocked high risk
            
            if is_malicious:
                print(f"[{self.agent_id}] !!! THREAT DETECTED (Risk: {risk_score:.2f}): {log.get('event_name') or log.get('attack_name')} !!!")
                alert = {
                    "source": self.agent_id,
                    "timestamp": time.time(),
                    "log_id": log.get("id") or "ml-alert",
                    "risk_score": float(risk_score),
                    "details": log
                }
                alerts.append(alert)
        
        return alerts

def generate_attack_logs(attack_type: str) -> List[Dict[str, Any]]:
    """
    Generates log streams based on the simulated attack scenario.
    """
    # 1. Expanded Noise Pool (Normal Cloud Activity)
    common_noise = [
        {"id": "sys_01", "event_name": "ConsoleLogin", "severity": "LOW"},
        {"id": "sys_02", "event_name": "DescribeInstances", "severity": "LOW"},
        {"id": "sys_03", "event_name": "ListObjects:Bucket-A", "severity": "LOW"},
        {"id": "sys_04", "event_name": "GetMetricStatistics", "severity": "LOW"},
        {"id": "sys_05", "event_name": "HealthCheck:Pass", "severity": "LOW"},
        {"id": "sys_06", "event_name": "UpdateSecurityGroupRuleDescription", "severity": "LOW"},
        {"id": "sys_07", "event_name": "AssumeRole:ReadOnly", "severity": "LOW"},
        {"id": "sys_08", "event_name": "Decrypt:Key-X", "severity": "LOW"},
    ]
    
    # Randomly select a subset of noise to make it dynamic
    selected_noise = random.sample(common_noise, k=random.randint(3, len(common_noise)))
    
    attack_logs = []
    
    if attack_type == "IAM_PRIVILEGE_ESCALATION":
        attack_logs = [
            {"id": "alert_iam_1", "event_name": "CreateAccessKey", "severity": "MEDIUM", "target": "unknown-user"},
            {"id": "alert_iam_2", "event_name": "AttachUserPolicy:AdministratorAccess", "severity": "HIGH", "target": "attacker-user"}
        ]
    elif attack_type == "S3_DATA_EXFILTRATION":
        attack_logs = [
            {"id": "alert_s3_1", "event_name": "ListBuckets", "severity": "LOW"},
            {"id": "alert_s3_2", "event_name": "PutBucketPolicy:PublicRead", "severity": "HIGH", "target": "confidential-data-bucket"}
        ]
    elif attack_type == "EC2_CRYPTO_MINING":
        attack_logs = [
            {"id": "alert_ec2_1", "event_name": "RunInstances:p3.16xlarge", "severity": "MEDIUM"},
            {"id": "alert_ec2_2", "event_name": "HighCpuUtil:MinerSignature", "severity": "HIGH", "target": "miner-instance-x99"}
        ]
    elif attack_type == "DDoS_ATTACK":
        attack_logs = [
             {"id": "alert_net_1", "event_name": "InboundTrafficSpike", "severity": "HIGH", "details": "100x normal load"}
        ]
    
    # --- NEW ATTACK TYPES ---
    elif attack_type == "SQL_INJECTION":
        attack_logs = [
            {"id": "alert_waf_1", "event_name": "WAF:SQLi_Pattern_Detected", "severity": "HIGH", "target": "app-db-primary", "payload": "' OR 1=1 --"},
            {"id": "alert_db_1", "event_name": "RDS:StrangeQueryPattern", "severity": "MEDIUM", "target": "app-db-primary"}
        ]
    elif attack_type == "RANSOMWARE_PRECURSOR":
        attack_logs = [
            {"id": "alert_fs_1", "event_name": "FileSystem:MassRename", "severity": "HIGH", "target": "shared-vol-01", "details": ".enc extension"},
            {"id": "alert_fs_2", "event_name": "HighIOPS:Write", "severity": "MEDIUM", "target": "shared-vol-01"}
        ]
    elif attack_type == "LATERAL_MOVEMENT":
        attack_logs = [
            {"id": "alert_net_2", "event_name": "RDP:BruteForceAttempt", "severity": "MEDIUM", "target": "bastion-host"},
            {"id": "alert_iam_3", "event_name": "SecretsManager:GetSecretValue", "severity": "HIGH", "target": "prod-db-creds"}
        ]
        
    else:
        # Default/Test case
        attack_logs = [{"id": "test_01", "event_name": "TestEvent", "severity": "HIGH"}]

    return selected_noise + attack_logs
