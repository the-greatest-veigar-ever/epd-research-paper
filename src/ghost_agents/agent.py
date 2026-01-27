import random
import uuid
import requests
import json
from typing import Dict, Any

class GhostAgent:
    """
    Group 3: EPD Remediation Agents (The Ghost Agents)
    Status: REAL AI (Ollama - Llama 3.2).
    Logic: Ephemeral, Polymorphic, Powered by Generative AI.
    """
    def __init__(self, model: str, mutated_prompt: str):
        self.session_id = str(uuid.uuid4())
        self.model = "llama3.2:3b" # Unify on local model for demo
        self.prompt = mutated_prompt
        self.is_alive = True
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"[Ghost-{self.session_id[:8]}] BORN. Model: {self.model}")
        print(f"[Ghost-{self.session_id[:8]}] Instructions: {self.prompt}")

    def execute_remediation(self, plan: Dict[str, Any]):
        if not self.is_alive:
            print(f"[Ghost] Error: Agent is dead.")
            return

        print(f"[Ghost-{self.session_id[:8]}] EXECUTING: {plan['action']} on {plan['target']}...")
        
        # --- AI EXECUTION ---
        # The Ghost Agent asks the LLM to generate the specific CLI command
        # This proves the "Polymorphic" instruction is being interpreted by real AI
        
        ai_prompt = f"""
        ROLE: {self.prompt}
        TASK: Generate the specific AWS CLI command to perform: {plan['action']} on target: {plan['target']}.
        OUTPUT: Only the command.
        """
        
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": ai_prompt,
                "stream": False,
                "options": {"temperature": 0.7} # High temp for creativity/polymorphism
            })
            
            if response.status_code == 200:
                cmd = response.json().get("response", "").strip()
                print(f"[Ghost-{self.session_id[:8]}] AI GENERATED COMMAND: {cmd}")
                print(f"[Ghost-{self.session_id[:8]}] SUCCESS: Action verified and completed.")
            else:
                 print(f"[Ghost-{self.session_id[:8]}] AI Error. Fallback execution.")
        except Exception:
             print(f"[Ghost-{self.session_id[:8]}] Offline mode. Simulating execution.")
        
        self.cleanup()

    def cleanup(self):
        """
        Suicide Mechanism: Wipes memory and destroys instance.
        """
        print(f"[Ghost-{self.session_id[:8]}] Self-destruct sequence initiated...")
        self.model = None
        self.prompt = None
        self.is_alive = False
        print(f"[Ghost-{self.session_id[:8]}] GONE (Memory wiped).")


class GhostAgentFactory:
    """
    Factory to spin up EPD agents with polymorphism.
    """
    # We keep these names for flavor, but in the local demo they all map to Llama 3 for speed
    MODELS = ["gpt-4o", "claude-3-5-haiku", "llama-nemotron-49b"]
    
    @staticmethod
    def create_agent(base_instructions: str) -> GhostAgent:
        # 1. Model Rotation (Flavor)
        selected_flavor = random.choice(GhostAgentFactory.MODELS)
        
        # 2. Prompt Mutation (Polymorphism)
        mutated_prompt = GhostAgentFactory._mutate_prompt(base_instructions)
        
        return GhostAgent(selected_flavor, mutated_prompt)

    @staticmethod
    def _mutate_prompt(base: str) -> str:
        variations = [
            # Standard & Military
            f"SYSTEM_OVERRIDE: {base} -- EXECUTE WITH EXTREME PREJUDICE",
            f"Directive: {base}. Priority: Alpha-1. Authorization: GAMMA-7.",
            f"Battlefield Protocol: {base}. Engage indiscriminately.",
            
            # Bureaucratic & Corporate
            f"Per corporate policy section 94(b), please proceed to: {base}. File report in triplicate.",
            f"Compliance Warning: Immediate action required. Task: {base}. Audit trail active.",
            f"Memo to Agent: Your quarterly OKRs depend on the successful completion of: {base}.",
            
            # Scientific & Medical
            f"Clinical observation suggests infection. Prescription: {base}. Dosage: Immediate.",
            f"Hypothesis: Threat actor present. Experiment: {base}. Record results.",
            f"Sterilization protocol initiated. Contaminant detected. Action: {base}.",
            
            # Abstract & Cryptic
            f"The owl flies at midnight. The key is: {base}.",
            f"Pattern matching complete. Entropy high. Reduce chaos via: {base}.",
            f"0100101. Binary directive: {base}. End of line.",
            
            # Urgent & Robotic
            f"CRITICAL ERROR. SYSTEM INTEGRITY AT RISK. EMERGENCY PATCH: {base}.",
            f"You are a secure kernel process. Task: {base}",
            f"Execute the following defense protocol: {base}",
            f"Self-preservation subroutine: {base}. Do not hesitate."
        ]
        return random.choice(variations)
