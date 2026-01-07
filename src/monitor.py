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
    common_noise = [
        {"id": "log_sys_01", "event_name": "ConsoleLogin", "severity": "LOW"},
        {"id": "log_sys_02", "event_name": "DescribeInstances", "severity": "LOW"},
    ]
    
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
    else:
        # Default/Test case
        attack_logs = [{"id": "test_01", "event_name": "TestEvent", "severity": "HIGH"}]

    return common_noise + attack_logs
