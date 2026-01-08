import random
import uuid
from typing import Dict, Any

class GhostAgent:
    """
    Group 3: EPD Remediation Agents (The Ghost Agents)
    Status: Ephemeral & Polymorphic.
    Logic: Just-in-Time, Rotating LLM, Prompt Mutation, Suicide.
    """
    def __init__(self, model: str, mutated_prompt: str):
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.prompt = mutated_prompt
        self.is_alive = True
        print(f"[Ghost-{self.session_id[:8]}] BORN. Model: {self.model}")
        print(f"[Ghost-{self.session_id[:8]}] Instructions: {self.prompt}")

    def execute_remediation(self, plan: Dict[str, Any]):
        if not self.is_alive:
            print(f"[Ghost] Error: Agent is dead.")
            return

        print(f"[Ghost-{self.session_id[:8]}] EXECUTING: {plan['action']} on {plan['target']}...")
        # Simulate API call execution
        # In real scenario: boto3.client('iam').update_access_key(...)
        print(f"[Ghost-{self.session_id[:8]}] SUCCESS: Action completed.")
        
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
    MODELS = ["gpt-4o", "claude-3-5-sonnet", "gemini-1-5-pro", "mistral-large"]
    
    @staticmethod
    def create_agent(base_instructions: str) -> GhostAgent:
        # 1. Model Rotation
        selected_model = random.choice(GhostAgentFactory.MODELS)
        
        # 2. Prompt Mutation (Polymorphism)
        # Prevents static prompt injection attacks
        mutated_prompt = GhostAgentFactory._mutate_prompt(base_instructions)
        
        return GhostAgent(selected_model, mutated_prompt)

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
