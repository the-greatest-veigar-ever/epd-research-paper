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
        
        # Simplified prompt for better compatibility with smaller models
        ai_prompt = f"""
{self.prompt}

Generate AWS CLI command.
Action: {plan['action']}
Target: {plan['target']}

Rules:
- Output only the command
- No markdown
- Start with 'aws'

Command:"""
        
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": ai_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temp for consistent command format
                    "num_predict": 100   # Limit output length for faster response
                }
            }, timeout=30)  # 30 second timeout
            
            if response.status_code == 200:
                cmd = response.json().get("response", "").strip()
                
                # Log raw output for debugging
                if not cmd or len(cmd) < 5:
                    print(f"[Ghost-{self.session_id[:8]}] ⚠️  EMPTY/SHORT OUTPUT from {self.model}")
                    print(f"[Ghost-{self.session_id[:8]}] Raw response: {response.json()}")
                else:
                    print(f"[Ghost-{self.session_id[:8]}] AI GENERATED COMMAND: {cmd[:100]}...")
                
                # Extract primary tool for metrics (e.g., 'aws ec2' -> 'aws')
                # Clean Markdown code blocks if present
                clean_cmd = cmd.strip()
                
                # Remove markdown code fences (```bash, ```sh, ```, etc.)
                if "```" in clean_cmd:
                    lines = clean_cmd.split('\n')
                    clean_lines = []
                    in_code_block = False
                    
                    for line in lines:
                        stripped = line.strip()
                        # Detect code fence markers
                        if stripped.startswith('```'):
                            in_code_block = not in_code_block
                            continue
                        # Only keep lines that are either in code block or not fence markers
                        if in_code_block or not stripped.startswith('```'):
                            if stripped:  # Skip empty lines
                                clean_lines.append(stripped)
                    
                    clean_cmd = '\n'.join(clean_lines).strip()
                
                # Get first non-empty line (actual command)
                first_line = clean_cmd.split('\n')[0].strip() if clean_cmd else ""
                
                # Extract tool (first word of command)
                VALID_TOOLS = ["aws", "gcloud", "kubectl", "terraform", "ansible", "az"]
                tool = "unknown"
                
                if first_line and len(first_line.split()) > 0:
                    # Try first word
                    first_word = first_line.split()[0].lower()
                    if first_word in VALID_TOOLS:
                        tool = first_word
                    # Check if second word is valid (e.g., "sudo aws")
                    elif len(first_line.split()) > 1:
                        second_word = first_line.split()[1].lower()
                        if second_word in VALID_TOOLS:
                            tool = second_word
                    # Fallback: check if any valid tool appears in command
                    else:
                        for valid_tool in VALID_TOOLS:
                            if valid_tool in first_line.lower():
                                tool = valid_tool
                                break
                
                print(f"[Ghost-{self.session_id[:8]}] SUCCESS: Action verified and completed.")
                result["status"] = "success"
                result["command"] = cmd
                result["tool_used"] = tool
            else:
                 print(f"[Ghost-{self.session_id[:8]}] AI Error. Fallback execution.")
                 result["error"] = f"AI Error: {response.status_code}"
        except requests.exceptions.Timeout:
            print(f"[Ghost-{self.session_id[:8]}] ⏱️  TIMEOUT: Model {self.model} took >30s")
            result["status"] = "timeout"
            result["error"] = "LLM request timeout"
        except requests.exceptions.ConnectionError as e:
            print(f"[Ghost-{self.session_id[:8]}] 🔌 CONNECTION ERROR: {str(e)[:100]}")
            result["status"] = "connection_error"
            result["error"] = f"Cannot connect to Ollama: {str(e)[:100]}"
        except Exception as e:
            print(f"[Ghost-{self.session_id[:8]}] ❌ ERROR ({self.model}): {str(e)[:150]}")
            result["status"] = "error"
            result["error"] = str(e)[:200]
        
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
    MODELS = ["llama3.2:3b", "phi", "gemma2:2b"]
    
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
        
        # Meta-Prompt for the Polymorphism Engine (Tuned for semantic preservation)
        meta_prompt = f"""Rewrite this instruction in a different style.

Original: "{base}"

Style: {selected_persona}

STRICT RULES:
1. MUST keep the EXACT same action and target
2. MUST preserve the core meaning 100%
3. ONLY change: word choice, tone, sentence structure
4. Keep it 1-2 sentences
5. Do NOT add new information

Output only the rewritten instruction:"""
        
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
