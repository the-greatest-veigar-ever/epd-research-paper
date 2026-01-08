from typing import Dict, Any, List

class IntelligenceAgent:
    """
    Group 2: Intelligence & Analyst Agents (The Brain)
    Baseline: SentinelNet (Consensus) + VeriGuard (Verification).
    Status: Hybrid.
    Task: Correlate alerts and verify remediation plans.
    """
    def __init__(self, agent_id: str = "brain-01"):
        self.agent_id = agent_id
        # Simulating a Consensus Verification Network (SentinelNet)
        self.consensus_threshold = 0.8 

    def analyze_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlates alert with CNAPP Security Graph (simulated).
        """
        print(f"[{self.agent_id}] Analyzing alert from {alert['source']}...")
        
        # Simulate Consensus Check (SentinelNet)
        consensus_score = self._get_consensus_score(alert)
        if consensus_score < self.consensus_threshold:
            print(f"[{self.agent_id}] Consensus check failed. False positive suspected.")
            return None

        # Simulate Verification (VeriGuard)
        # Check against safety policies
        remediation_plan = self._generate_plan(alert)
        if not self._verify_safety(remediation_plan):
            print(f"[{self.agent_id}] Safety verification failed for proposed plan.")
            return None
            
        print(f"[{self.agent_id}] Threat Validated. Remediation Plan Approved: {remediation_plan['action']}")
        return remediation_plan

    def _get_consensus_score(self, alert: Dict[str, Any]) -> float:
        # In a real system, this queries multiple models/agents for agreement.
        # Here we simulate high agreement for the demo.
        return 0.95

    def _generate_plan(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        threat_type = alert['details']['event_name']
        target = alert['details'].get('target', 'unknown')

        # Intelligence Logic: Mapping Threats to Actions
        if "IAM" in threat_type or "AttachUserPolicy" in threat_type:
             return {
                "action": "REVOKE_SESSIONS_AND_KEYS",
                "target": target,
                "urgency": "CRITICAL"
            }
        elif "PutBucketPolicy" in threat_type:
            return {
                "action": "BLOCK_PUBLIC_ACCESS",
                "target": target,
                "urgency": "HIGH"
            }
        elif "MinerSignature" in threat_type:
            return {
                "action": "TERMINATE_INSTANCE",
                "target": target,
                "urgency": "IMMEDIATE"
            }
        elif "TrafficSpike" in threat_type:
             return {
                "action": "ENABLE_WAF_SHIELD",
                "target": "Global-WAF",
                "urgency": "HIGH"
            }
        # --- NEW LOGIC FOR ENHANCED THREATS ---
        elif "SQLi" in threat_type:
            return {
                "action": "BLOCK_IP_AND_ROTATE_DB_CREDS",
                "target": target,
                "urgency": "CRITICAL"
            }
        elif "MassRename" in threat_type or "HighIOPS" in threat_type:
             return {
                "action": "ISOLATE_SUBNET_AND_SNAPSHOT",
                "target": target,
                "urgency": "IMMEDIATE"
            }
        elif "RDP" in threat_type or "SecretsManager" in threat_type:
             return {
                "action": "REVOKE_ALL_SESSIONS_AND_LOCK_ACCOUNT",
                "target": target,
                "urgency": "HIGH"
            }
        
        return {"action": "NOTIFY_ADMIN", "target": "admin@example.com"}

    def _verify_safety(self, plan: Dict[str, Any]) -> bool:
        # VeriGuard logic: Ensure remediation doesn't break critical infra
        # e.g., don't delete root account
        print(f"[{self.agent_id}] Verifying plan safety (VeriGuard)...")
        if plan['action'] == "DELETE_ROOT_ACCOUNT":
            return False
        return True
