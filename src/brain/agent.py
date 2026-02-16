import json
import torch
import time
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class IntelligenceAgent:
    """
    Group 2: Intelligence & Analyst Agents (The Brain)
    Baseline: SentinelNet (Consensus) + VeriGuard (Verification).
    Status: REAL AI (Custom QLoRA Model - Phi-2).
    Task: Correlate alerts and verify remediation plans using Fine-Tuned Local LLM.
    """
    def __init__(self, agent_id: str = "brain-01"):
        self.agent_id = agent_id
        self.consensus_threshold = 0.8
        self._last_reasoning = ""
        
        # --- AI SETUP ---
        print(f"[{self.agent_id}] Initializing. Loading Custom QLoRA Model (Phi-2)...")
        self.device = "cpu" # Force CPU for correct inference on Mac (MPS precision issues)
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.base_model_name = "microsoft/phi-2"
        self.adapter_path = "ai/models/qlora-hugging-face/qlora-secqa" # Adjust path as needed
        
        try:
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load Base Model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.device
            )
            
            # Load Adapters
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            print(f"[{self.agent_id}] Successfully loaded QLoRA Model on {self.device}.")
            self.is_ready = True
        except Exception as e:
            print(f"[{self.agent_id}] Error loading AI model: {e} (Using fallback logic)")
            self.is_ready = False

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

        # --- AI PLANNING ---
        print(f"[{self.agent_id}] Consulting Custom QLoRA Model for remediation plan...")
        remediation_plan = self._generate_plan_with_ai(alert)
        
        if not remediation_plan:
             # Just in case model fails to output clean data
            return self._fallback_plan(alert['details']['event_name'], alert['details'].get('target', 'unknown'))

        # Simulate Verification (VeriGuard)
        if not self._verify_safety(remediation_plan):
            print(f"[{self.agent_id}] Safety verification failed for proposed plan.")
            return None
            
        # Helper to get action from STIX or Legacy
        action = remediation_plan.get('x_epd_action') or remediation_plan.get('action')
        print(f"[{self.agent_id}] Threat Validated. Remediation Plan Approved: {action}")
        return remediation_plan

    def analyze_alert_with_context(self, alert: Dict[str, Any],
                                    peer_proposals: list) -> Dict[str, Any]:
        """
        Debate round: re-analyze with knowledge of other agents' proposals.
        For IntelligenceAgent (QLoRA), we keep the original answer since
        the local model doesn't support multi-turn conversation well.
        """
        # QLoRA Phi-2 is deterministic and doesn't handle debate prompts well,
        # so we return the same analysis. Credit scoring handles the weighting.
        return self.analyze_alert(alert)

    def get_reasoning(self) -> str:
        """Return the reasoning from the last analysis."""
        return self._last_reasoning

    def _get_consensus_score(self, alert: Dict[str, Any]) -> float:
        # Uses Squad A's XGBoost confidence as a proxy for consensus.
        # Higher AI confidence = higher consensus that this is a real threat.
        score = alert.get('ai_score', 0.5)
        return score

    def _generate_plan_with_ai(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses Local QLoRA to decide action using OpenC2 Standard.
        """
        if not self.is_ready:
            return None
            
        threat_type = alert['details']['event_name']
        target = alert['details'].get('target', 'unknown')
        
        # OpenC2 Standard Prompt
        # We instruct the LLM to act as a Security Orchestrator and output valid JSON.
        prompt = f"""### Instruction:
You are an advanced Security Orchestration AI.
A threat has been detected:
- Threat Type: {threat_type}
- Target: {target}

Generate an OpenC2 (Open Command and Control) command to remediate this threat.
The output must be a VALID JSON object with:
- "action": The action verb (e.g., deny, allow, query, scan, contain).
- "target": The target object (e.g., ipv4_connection, file, process).
- "args": Optional arguments (e.g., duration, response_requested).

Example:
{{
  "action": "deny",
  "target": {{ "ipv4_connection": {{ "src_addr": "1.2.3.4" }} }},
  "args": {{ "duration": "24h" }}
}}

### Response:
"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=100,  # Enough for JSON, safe for CPU inference
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False # Deterministic
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response — only parse AFTER "### Response:" marker
            # to avoid matching the example JSON embedded in the prompt.
            raw_output = response
            self._last_reasoning = response.strip()
            json_str = ""
            
            import re
            response_marker = "### Response:"
            response_start = response.rfind(response_marker)
            if response_start != -1:
                response_only = response[response_start + len(response_marker):]
            else:
                response_only = response
            
            # 1. Try to find JSON block in model's response only
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_only, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 2. Heuristic: Look for braces in response only
                start = response_only.find('{')
                end = response_only.rfind('}')
                if start != -1 and end != -1:
                    json_str = response_only[start:end+1]
            
            import json
            try:
                openc2_cmd = json.loads(json_str)
                action = openc2_cmd.get('action', 'unknown')
                target_obj = openc2_cmd.get('target', {})
            except json.JSONDecodeError:
                print(f"[{self.agent_id}] JSON Parse Error. Raw: {json_str[:50]}...")
                openc2_cmd = {}
                action = "unknown"
                target_obj = {}

            print(f"[{self.agent_id}] OpenC2 Action: {action}")

            # Fallback if AI fails to generate valid OpenC2
            if action == "unknown":
                # Threat-type based OpenC2 action selection
                threat_upper = threat_type.upper()
                if any(kw in threat_upper for kw in ["FLOOD", "DDOS", "DOS", "BRUTE", "SCAN", "BOT", "FTP", "SSH"]):
                    openc2_cmd = {
                        "action": "deny",
                        "target": { "ipv4_connection": { "src_addr": target } },
                        "args": { "duration": "1h" }
                    }
                    action = "deny"
                elif any(kw in threat_upper for kw in ["INFILTER", "MALWARE", "COMPROMIS", "SQL", "XSS", "WEB ATTACK"]):
                    openc2_cmd = {
                        "action": "contain",
                        "target": { "device": { "hostname": target } }
                    }
                    action = "contain"
                else:
                    openc2_cmd = {
                         "action": "query",
                         "target": { "features": { "query_type": "status" } }
                    }
                    action = "query"
                self._last_reasoning = f"Fallback rule selected: {action}"
                print(f"[{self.agent_id}] AI output invalid, using Fallback OpenC2: {action}")

            # STIX 2.1 Compliant Output (Course of Action)
            import uuid
            import datetime
            
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            stix_id = f"course-of-action--{str(uuid.uuid4())}"
            
            # Define reasoning
            reason = f"AI selected OpenC2 Action: {action}"
            
            # Construct standard STIX object
            stix_output = {
                "type": "course-of-action",
                "spec_version": "2.1",
                "id": stix_id,
                "created": timestamp,
                "modified": timestamp,
                "name": f"OpenC2 Remediation: {action}",
                "description": f"Automated OpenC2 command triggered by {threat_type} on {target}. Reason: {reason}",
                
                # Custom extensions for EPD internal logic
                "x_epd_action": action,
                "x_epd_target": target,
                "x_epd_openc2": openc2_cmd,
                "x_epd_score": alert.get('ai_score', 0.5),  # Real XGBoost confidence from Squad A
                "x_epd_reason": reason,
                "x_epd_ai_raw_output": raw_output,
                "x_epd_input_alert_id": alert.get('log_id', 'unknown'),
                "x_epd_agent_id": self.agent_id
            }
            
            return stix_output

        except Exception as e:
            print(f"[{self.agent_id}] AI Inference Error: {e}")
            return None

    def _fallback_plan(self, threat_type: str, target: str) -> Dict[str, Any]:
        """
        Placeholder: Legacy Rule Engine (e.g., existing firewall rules).
        Used when AI confidence is low or model is offline.
        """
        print(f"[{self.agent_id}] Using Legacy Rule Engine.")
        if "IAM" in threat_type:
             return {"action": "REVOKE_SESSIONS", "target": target}
        return {"action": "NOTIFY_ADMIN", "target": target}

    def _verify_safety(self, plan: Dict[str, Any]) -> bool:
        """
        VeriGuard S (Simulated) - Safety Check
        Ensures remediation does not violate safety policies.
        """
        print(f"[{self.agent_id}] Verifying plan and ensuring Critical Infrastructure Protection (CIP)...")
        
        # Handle OpenC2, STIX 2.1, and Legacy formats
        action = plan.get('x_epd_action')
        target = plan.get('x_epd_target')
        
        # OpenC2 Fallback access
        if not action or action == "unknown":
            openc2 = plan.get('x_epd_openc2', {})
            action = openc2.get('action')
            # OpenC2 targets are nested (e.g. target: {ipv4_connection: ...})
            # For safety check, we just need to ensure target exists
            if openc2.get('target'):
                target = str(openc2.get('target'))
                target = str(openc2.get('target'))
        
        # Legacy Fallback
        if not action:
            action = plan.get('action')
            target = plan.get('target')
            
        if not action:
            return False

        # 1. Critical Prevention (e.g. don't delete root)
        if action == "DELETE_ROOT_ACCOUNT":
            return False
            
        # 2. Production Safety
        if action == "SHUTDOWN_SERVICE" and target and "production" in target:
             print(f"[{self.agent_id}] REFUSING to shutdown production without human auth.")
             return False
             
        return True
