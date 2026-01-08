import time
import random
from typing import Dict, Any, List, Optional

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
        print(f"[{self.agent_id}] Initialized and monitoring streams.")

    def monitor_logs(self, mock_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulates scanning logs for threats.
        In a real system, this would ingest GuardDuty/CloudTrail streams.
        """
        alerts = []
        print(f"[{self.agent_id}] Scanning {len(mock_logs)} log entries...")
        
        for log in mock_logs:
            if log.get("severity") == "HIGH":
                print(f"[{self.agent_id}] !!! THREAT DETECTED: {log.get('event_name')} !!!")
                alert = {
                    "source": self.agent_id,
                    "timestamp": time.time(),
                    "log_id": log.get("id"),
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
