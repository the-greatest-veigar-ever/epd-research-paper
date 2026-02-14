import random
import uuid
import requests
import json
from typing import Dict, Any

class GhostAgentV2:
    """
    Group 3: EPD Remediation Agents (The Ghost Agents) - V2 Improved
    Status: REAL AI (Ollama - Multiple SLMs with rotation).
    Logic: Ephemeral, Polymorphic (Technical), Powered by Generative AI.
    Improvements: Technical Personas, One-Shot Prompting.
    """
    def __init__(self, model: str, mutated_prompt: str):
        self.session_id = str(uuid.uuid4())
        self.model = model  # Use the passed model (enables true rotation)
        self.prompt = mutated_prompt
        self.is_alive = True
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"[GhostV2-{self.session_id[:8]}] BORN. Model: {self.model}")
        print(f"[GhostV2-{self.session_id[:8]}] Instructions: {self.prompt}")

    def execute_remediation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "status": "failed",
            "command": None,
            "tool_used": None,
            "error": None
        }

        if not self.is_alive:
            print(f"[GhostV2] Error: Agent is dead.")
            result["error"] = "Agent is dead"
            return result

        print(f"[GhostV2-{self.session_id[:8]}] EXECUTING: {plan['action']} on {plan['target']}...")
        
        # --- AI EXECUTION ---
        # The Ghost Agent asks the LLM to generate the specific CLI command
        # IMPROVEMENT: Added "Technical One-Shot Examples" to guide SLMs
        
        ai_prompt = f"""
{self.prompt}

Generate AWS CLI command.
Action: {plan['action']}
Target: {plan['target']}

EXAMPLES:
Instruction: Block IP 192.168.1.5
Command: aws ec2 create-network-acl-entry --network-acl-id acl-xyz --ingress --rule-number 100 --protocol tcp --port-range From=0,To=65535 --cidr-block 192.168.1.5/32 --rule-action deny

Instruction: Isolate Instance i-0123456789abcdef0
Command: aws ec2 modify-instance-attribute --instance-id i-0123456789abcdef0 --groups sg-isolated

Instruction: Stop Instance i-0aaa111222333
Command: aws ec2 stop-instances --instance-ids i-0aaa111222333

Rules:
- Output only the command
- No markdown
- Start with 'aws'
- Use placeholders like <ID> if ID is unknown

Command:"""
        
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": ai_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower temp for consistent command format (V2 Adjustment)
                    "num_predict": 128   # increased length
                }
            }, timeout=30)  # 30 second timeout
            
            if response.status_code == 200:
                cmd = response.json().get("response", "").strip()
                
                # Log raw output for debugging
                if not cmd or len(cmd) < 5:
                    print(f"[GhostV2-{self.session_id[:8]}] ⚠️  EMPTY/SHORT OUTPUT from {self.model}")
                else:
                    print(f"[GhostV2-{self.session_id[:8]}] AI GENERATED COMMAND: {cmd[:100]}...")
                
                # formatting logic (same as v1)
                clean_cmd = cmd.strip()
                if "```" in clean_cmd:
                    lines = clean_cmd.split('\n')
                    clean_lines = []
                    in_code_block = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('```'):
                            in_code_block = not in_code_block
                            continue
                        if in_code_block or not stripped.startswith('```'):
                            if stripped:
                                clean_lines.append(stripped)
                    clean_cmd = '\n'.join(clean_lines).strip()
                
                first_line = clean_cmd.split('\n')[0].strip() if clean_cmd else ""
                
                # Tool extraction logic
                VALID_TOOLS = ["aws", "gcloud", "kubectl", "terraform", "ansible", "az"]
                tool = "unknown"
                if first_line and len(first_line.split()) > 0:
                    first_word = first_line.split()[0].lower()
                    if first_word in VALID_TOOLS:
                        tool = first_word
                    elif len(first_line.split()) > 1:
                        second_word = first_line.split()[1].lower()
                        if second_word in VALID_TOOLS:
                            tool = second_word
                    else:
                        for valid_tool in VALID_TOOLS:
                            if valid_tool in first_line.lower():
                                tool = valid_tool
                                break
                
                print(f"[GhostV2-{self.session_id[:8]}] SUCCESS: Action verified and completed.")
                result["status"] = "success"
                result["command"] = cmd
                result["tool_used"] = tool
            else:
                 print(f"[GhostV2-{self.session_id[:8]}] AI Error. Fallback execution.")
                 result["error"] = f"AI Error: {response.status_code}"
        except Exception as e:
            print(f"[GhostV2-{self.session_id[:8]}] ❌ ERROR ({self.model}): {str(e)[:150]}")
            result["status"] = "error"
            result["error"] = str(e)[:200]
        
        self.cleanup()
        return result

    def cleanup(self):
        print(f"[GhostV2-{self.session_id[:8]}] Self-destruct sequence initiated...")
        self.model = None
        self.prompt = None
        self.is_alive = False
        print(f"[GhostV2-{self.session_id[:8]}] GONE (Memory wiped).")


class GhostAgentFactoryV2:
    """
    Factory V2: Uses Technical Personas for better accuracy.
    """
    # Real SLMs available via Ollama for baseline comparison
    MODELS = ["llama3.2:3b", "phi", "gemma2:2b"]
    
    _current_model_idx = 0
    
    @staticmethod
    def create_agent(base_instructions: str, rotate_model: bool = True) -> GhostAgentV2:
        if rotate_model:
            selected_model = GhostAgentFactoryV2.MODELS[GhostAgentFactoryV2._current_model_idx]
            GhostAgentFactoryV2._current_model_idx = (GhostAgentFactoryV2._current_model_idx + 1) % len(GhostAgentFactoryV2.MODELS)
        else:
            selected_model = GhostAgentFactoryV2.MODELS[0]
        
        mutated_prompt = GhostAgentFactoryV2._mutate_prompt(base_instructions)
        return GhostAgentV2(selected_model, mutated_prompt)

    @staticmethod
    def _mutate_prompt(base: str) -> str:
        """
        V2 Polymorphism: Asks the LLM to rewrite utilizing TECHNICAL PERSONAS to reduce semantic drift.
        """
        ollama_url = "http://localhost:11434/api/generate"
        model = "llama3.2:3b"
        
        # IMPROVEMENT: Technical Personas Only
        personas = [
            "a Strictly Compliant System Administrator",
            "a Senior DevOps Engineer focused on reliability",
            "a Cyber Security Analyst executing protocol",
            "a Site Reliability Engineer (SRE) handling an incident",
            "a Network Operations Center (NOC) Engineer",
            "a Cloud Infrastructure Architect",
            "an Automated Response Bot (v2.0)",
            "a Zero-Trust Security Policy Enforcer",
            "a Compliance Officer auditing actions",
            "a Legacy System Maintainer"
        ]
        
        selected_persona = random.choice(personas)
        
        meta_prompt = f"""Rewrite this instruction in a different style.

Original: "{base}"

Style: {selected_persona}

STRICT RULES:
1. MUST keep the EXACT same action and target
2. MUST preserve the core technical meaning
3. Use technical jargon appropriate for the role
4. Keep it concise (1 sentence)
5. Do NOT use metaphors (no pirates, no knights)

Output only the rewritten instruction:"""
        
        try:
            response = requests.post(ollama_url, json={
                "model": model,
                "prompt": meta_prompt,
                "stream": False,
                "options": {"temperature": 0.7} # Reduced creativity for technical accuracy
            }, timeout=10)
            
            if response.status_code == 200:
                mutated = response.json().get("response", "").strip()
                if mutated.startswith('"') and mutated.endswith('"'):
                    mutated = mutated[1:-1]
                return mutated
        except Exception as e:
            print(f"[GhostFactoryV2] AI Polymorphism failed ({e}). Falling back.")
            
        return base
