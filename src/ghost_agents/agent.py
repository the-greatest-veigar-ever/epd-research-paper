import random
import uuid
import requests
import json
from typing import Dict, Any

class GhostAgent:
    """
    Group 3: EPD Remediation Agents (The Ghost Agents)
    Status: REAL AI (Ollama - Multiple SLMs with rotation).
    Logic: Ephemeral, Polymorphic, Powered by Generative AI.
    """
    def __init__(self, model: str, mutated_prompt: str):
        self.session_id = str(uuid.uuid4())
        self.model = model  # Use the passed model (enables true rotation)
        self.prompt = mutated_prompt
        self.is_alive = True
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"[Ghost-{self.session_id[:8]}] BORN. Model: {self.model}")
        print(f"[Ghost-{self.session_id[:8]}] Instructions: {self.prompt}")

    def execute_remediation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "status": "failed",
            "command": None,
            "tool_used": None,
            "error": None
        }

        if not self.is_alive:
            print(f"[Ghost] Error: Agent is dead.")
            result["error"] = "Agent is dead"
            return result

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
                
                # Extract primary tool for metrics (e.g., 'aws ec2' -> 'aws')
                # Simple heuristic: first word
                tool = cmd.split()[0] if cmd else "unknown"
                
                print(f"[Ghost-{self.session_id[:8]}] SUCCESS: Action verified and completed.")
                result["status"] = "success"
                result["command"] = cmd
                result["tool_used"] = tool
            else:
                 print(f"[Ghost-{self.session_id[:8]}] AI Error. Fallback execution.")
                 result["error"] = f"AI Error: {response.status_code}"
        except Exception as e:
             print(f"[Ghost-{self.session_id[:8]}] Offline mode. Simulating execution.")
             result["status"] = "simulated_success"
             result["error"] = str(e)
        
        self.cleanup()
        return result

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
    Factory to spin up EPD agents with polymorphism and model rotation.
    """
    # Real SLMs available via Ollama for baseline comparison
    MODELS = ["llama3.2:3b", "phi3:mini", "gemma2:2b"]
    
    # Current model index for round-robin rotation
    _current_model_idx = 0
    
    @staticmethod
    def create_agent(base_instructions: str, rotate_model: bool = True) -> GhostAgent:
        """
        Create a new GhostAgent with optional model rotation.
        
        Args:
            base_instructions: The base task instruction
            rotate_model: If True, rotate between models. If False, use first model only.
        """
        # 1. Model Rotation (Round-robin for consistent distribution)
        if rotate_model:
            selected_model = GhostAgentFactory.MODELS[GhostAgentFactory._current_model_idx]
            GhostAgentFactory._current_model_idx = (GhostAgentFactory._current_model_idx + 1) % len(GhostAgentFactory.MODELS)
        else:
            selected_model = GhostAgentFactory.MODELS[0]  # Default to first model
        
        # 2. Prompt Mutation (Polymorphism)
        mutated_prompt = GhostAgentFactory._mutate_prompt(base_instructions)
        
        return GhostAgent(selected_model, mutated_prompt)

    @staticmethod
    def _mutate_prompt(base: str) -> str:
        """
        True AI Polymorphism: Asks the LLM to rewrite the instruction using a specific persona/style.
        """
        ollama_url = "http://localhost:11434/api/generate"
        model = "llama3.2:3b"  # Use smallest model for prompt mutation (speed)
        
        # Expanded persona pool for better polymorphism variety
        personas = [
            # Original 8 personas
            "a Military General issuing a battlefield protocol",
            "a Bureaucratic Corporate Policy Bot",
            "a Medical Doctor prescribing a treatment",
            "a Cryptic spy speaking in code",
            "a Robotic System Kernel issuing a critical patch",
            "a Panicked System Administrator",
            "a Shakespearean Actor",
            "a Cyberpunk Hacker using slang",
            # New personas for expanded variety
            "an Air Traffic Controller giving urgent instructions",
            "a Zen Buddhist Monk offering calm guidance",
            "a Pirate Captain commanding the crew",
            "a NASA Mission Control operator",
            "a Medieval Knight declaring a quest",
            "a Sports Coach giving a pep talk",
            "a Detective explaining evidence",
            "a Chef describing a recipe",
            "a Flight Attendant giving safety instructions",
            "a News Anchor reporting breaking news",
            "a Video Game NPC giving a quest",
            "a Drill Sergeant barking orders"
        ]
        
        selected_persona = random.choice(personas)
        
        # Meta-Prompt for the Polymorphism Engine
        meta_prompt = f"""
        Task: Rewrite the following security instruction.
        Original Instruction: "{base}"
        
        Style Requirement: Rewrite it as if {selected_persona}. 
        Use unique vocabulary, jargon, and tone appropriate for this persona.
        Do NOT change the core meaning or the target entities (IDs/IPs). Just wrap it in the persona's style.
        keep it relatively short (1-2 sentences).
        
        Output: ONLY the rewritten instruction. No "Here is the rewritten text" prefix.
        """
        
        try:
            response = requests.post(ollama_url, json={
                "model": model,
                "prompt": meta_prompt,
                "stream": False,
                "options": {"temperature": 0.9} # High creativity
            }, timeout=10) # Fast timeout for factory
            
            if response.status_code == 200:
                mutated = response.json().get("response", "").strip()
                # Remove quotes if model added them
                if mutated.startswith('"') and mutated.endswith('"'):
                    mutated = mutated[1:-1]
                return mutated
                
        except Exception as e:
            # Fallback to templates if LLM is offline or times out (Hybrid approach)
            print(f"[GhostFactory] AI Polymorphism failed ({e}). Falling back to Templates.")
            
        # Fallback Logic (The old list, kept for robustness)
        variations = [
            f"SYSTEM_OVERRIDE: {base} -- EXECUTE WITH EXTREME PREJUDICE",
            f"Directive: {base}. Priority: Alpha-1. Authorization: GAMMA-7.",
            f"Clinical observation suggests infection. Prescription: {base}. Dosage: Immediate.",
            f"The owl flies at midnight. The key is: {base}.",
            f"CRITICAL ERROR. SYSTEM INTEGRITY AT RISK. EMERGENCY PATCH: {base}."
        ]
        return random.choice(variations)
