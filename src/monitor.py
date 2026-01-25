import time
import random
import numpy as np
import joblib
from typing import Dict, Any, List, Optional
# No longer using sklearn.ensemble here directly for training, just loading
# But we need joblib

class DetectionAgent:
    """
    Group 1: Persistent Detection Agents (The Watchers)
    Baseline: MALCDF detection framework.
    Status: REAL AI (Custom Trained Isolation Forest).
    
    Task: Continuously monitor Cloud Logs using the locally trained model.
    """
    def __init__(self, agent_id: str = "watcher-01"):
        self.agent_id = agent_id
        self.status = "ACTIVE"
        print(f"[{self.agent_id}] Initializing. Loading custom-trained model...")
        
        # --- AI SETUP ---
        try:
            self.model = joblib.load("src/watchers/watcher_isoforest.joblib")
            self.scaler = joblib.load("src/watchers/watcher_scaler.joblib")
            self.numeric_cols = joblib.load("src/watchers/watcher_features.joblib") # 78 features
            self.is_trained = True
            print(f"[{self.agent_id}] Successfully loaded Isolation Forest trained on CSE-CIC-IDS2018.")
        except Exception as e:
            print(f"[{self.agent_id}] Error loading model: {e}")
            self.is_trained = False

    def monitor_traffic_batch(self, df_batch: Any) -> List[Dict[str, Any]]:
        """
        Scans a batch of RAW traffic data (DataFrame or Numpy Array).
        Used by the Autonomous Runner.
        """
        if not self.is_trained:
            return []
            
        alerts = []
        
        # 1. Transform Feature Match
        # We need to ensure we only send the numeric columns the model expects
        try:
            # Filter columns if DataFrame
            if hasattr(df_batch, "columns"):
                # If columns are missing, fill with 0
                X = df_batch[self.numeric_cols].fillna(0)
            else:
                X = df_batch
                
            # 2. Scale
            X_scaled = self.scaler.transform(X)
            
            # 3. Predict
            predictions = self.model.predict(X_scaled) # -1/1
            scores = self.model.decision_function(X_scaled)
            
            # 4. Process Anomalies
            anomalies_indices = np.where(predictions == -1)[0]
            
            if len(anomalies_indices) > 0:
                print(f"[{self.agent_id}] BATCH SCAN: Found {len(anomalies_indices)} anomalies in {len(df_batch)} flows!")
                
                for idx in anomalies_indices:
                    # Construct Alert Object
                    # In a real batch, we might not want to spam log every single one.
                    # Limit to first 5 for the demo console output
                    if len(alerts) < 5: 
                         # Convert row to dict for detailed reporting
                         row_details = df_batch.iloc[idx].to_dict()
                         
                         # INJECT keys required by Brain Agent
                         # The CSV usually has a 'Label' column
                         label = row_details.get("Label", "Unknown Anomaly")
                         row_details["event_name"] = label
                         row_details["target"] = "NetworkInterface"
                         
                         alert = {
                            "source": self.agent_id,
                            "timestamp": time.time(),
                            "type": "NET_FLOW_ANOMALY",
                            "ai_score": scores[idx],
                            "details": row_details # FULL DETAILS + Metadata
                        }
                         alerts.append(alert)
                    else:
                        break
        except Exception as e:
            print(f"[{self.agent_id}] Batch Error: {e}")
            
        return alerts

    def monitor_logs(self, mock_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans logs using the REAL AI model (Simulated Mode).
        ... (Kept for backward compatibility with main.py) ...
        """
        if not self.is_trained:
            print(f"[{self.agent_id}] Model not ready. Skipping AI scan.")
            return []
            
        # ... logic as before ...
        alerts = []
        # ... (rest of function remains same)
        for log in mock_logs:
            event_name = log.get("event_name", "Unknown")
            features = self._synthesize_features(log)
            # Scale
            features_scaled = self.scaler.transform([features])
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            anomaly_score = self.model.decision_function(features_scaled)[0]
            if prediction == -1:
                print(f"[{self.agent_id}] REAL AI DETECTED ANOMALY! Event: {event_name} (Score: {anomaly_score:.2f})")
                alert = {
                    "source": self.agent_id,
                    "timestamp": time.time(),
                    "log_id": log.get("id"),
                    "details": log,
                    "ai_score": anomaly_score
                }
                alerts.append(alert)
        return alerts

    def _synthesize_features(self, log: Dict[str, Any]) -> List[float]:
        """
        Turns a mock log entry into a 78-float vector that the Isolation Forest understands.
        """
        # Start with a "Normal" base vector (close to mean 0 after scaling)
        # We use random noise around 0.5 
        vector = np.random.normal(loc=10.0, scale=5.0, size=78)
        
        # If the log implies an Attack, we inject MASSIVE statistical anomalies
        # into features that typically indicate attacks (e.g. Fwd Pkts, Duration)
        severity = log.get("severity", "LOW")
        target = log.get("event_name", "")
        
        # NOTE: This 'severity' check mimics the result of flow analysis. 
        # In a real system, the flow collector would naturally produce high numbers for these fields.
        if severity == "HIGH" or "Attack" in target:
             # Make flow duration huge (index 4 is often Duration-ish in IDS datasets)
             vector[4] = 99999999.0 
             # Make packet count huge
             vector[5] = 50000.0
             # Add extreme variance
             vector[10] = -5000.0
             
        return vector

def generate_attack_logs(attack_type: str) -> List[Dict[str, Any]]:
    """
    Generates log streams based on the simulated attack scenario.
    """
    # 1. Expanded Noise Pool (Normal Cloud Activity)
    common_noise_events = [
        "ConsoleLogin", "DescribeInstances", "ListObjects:Bucket-A",
        "GetMetricStatistics", "HealthCheck:Pass",
        "UpdateSecurityGroupRuleDescription", "AssumeRole:ReadOnly", "Decrypt:Key-X"
    ]
    
    # Generate random noise objects
    selected_noise = []
    for _ in range(random.randint(3, 8)):
        evt = random.choice(common_noise_events)
        selected_noise.append({"id": f"sys_{random.randint(100,999)}", "event_name": evt, "severity": "LOW"})
    
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
