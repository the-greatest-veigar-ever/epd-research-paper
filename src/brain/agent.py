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
        
        # --- AI SETUP ---
        print(f"[{self.agent_id}] Initializing. Loading Custom QLoRA Model (Phi-2)...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.base_model_name = "microsoft/phi-2"
        self.adapter_path = "src/qlora-hugging-face/output/qlora-secqa" # Adjust path as needed
        
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
            
        print(f"[{self.agent_id}] Threat Validated. Remediation Plan Approved: {remediation_plan['action']}")
        return remediation_plan

    def _get_consensus_score(self, alert: Dict[str, Any]) -> float:
        return 0.95

    def _generate_plan_with_ai(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses Local QLoRA to decide action.
        We frame it as a Q&A because the model was fine-tuned on SecQA.
        """
        if not self.is_ready:
            return None
            
        threat_type = alert['details']['event_name']
        target = alert['details'].get('target', 'unknown')
        
        # Prompt Engineering for SecQA-tuned model
        # We ask it to choose the best action.
        question = f"Context: A security event '{threat_type}' was detected on target '{target}'. What is the appropriate remediation action?"
        choices_text = "A. REVOKE_SESSIONS\nB. TERMINATE_INSTANCE\nC. BLOCK_IP\nD. IGNORE"
        
        prompt = f"""### Question:
{question}

### Choices:
{choices_text}

### Answer:
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=10, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False # Deterministic
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the part after "Answer:"
            # The model usually outputs "Answer: Choice" or "Answer: A. Action"
            answer_part = response.split("Answer:")[-1].strip().upper()
            print(f"[{self.agent_id}] QLoRA Output: '{answer_part}'")
            
            # Map Answer to Action Code
            action = "NOTIFY_ADMIN"
            if "REVOKE" in answer_part or "A." in answer_part:
                action = "REVOKE_SESSIONS"
            elif "TERMINATE" in answer_part or "B." in answer_part:
                action = "TERMINATE_INSTANCE"
            elif "BLOCK" in answer_part or "C." in answer_part:
                action = "BLOCK_IP"
                
            return {
                "action": action,
                "target": target,
                "reason": f"AI Chose: {answer_part}"
            }

        except Exception as e:
            print(f"[{self.agent_id}] AI Inference Error: {e}")
            return None

    def _fallback_plan(self, threat_type: str, target: str) -> Dict[str, Any]:
        """Legacy rule-based fallback."""
        print(f"[{self.agent_id}] Using Legacy Rule Engine.")
        if "IAM" in threat_type:
             return {"action": "REVOKE_SESSIONS", "target": target}
        return {"action": "NOTIFY_ADMIN", "target": target}

    def _verify_safety(self, plan: Dict[str, Any]) -> bool:
        print(f"[{self.agent_id}] Verifying plan and ensuring Critical Infrastructure Protection (CIP)...")
        if plan['action'] == "DELETE_ROOT_ACCOUNT":
            return False
        if plan['action'] == "SHUTDOWN_SERVICE" and "production" in plan['target']:
             print(f"[{self.agent_id}] REFUSING to shutdown production without human auth.")
             return False
        return True
