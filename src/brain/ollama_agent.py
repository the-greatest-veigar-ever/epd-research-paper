"""
OllamaIntelligenceAgent — Squad B Brain Agent using Ollama local models.
Wraps Phi3:mini and Llama3.2:3b via Ollama REST API.
Produces the same STIX 2.1 output as IntelligenceAgent for compatibility.
"""
import json
import re
import uuid
import datetime
import requests
from typing import Dict, Any, Optional


class OllamaIntelligenceAgent:
    """
    Squad B Brain Agent backed by an Ollama model.
    Compatible interface with IntelligenceAgent (Phi-2 QLoRA).
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, agent_id: str, model: str):
        """
        Args:
            agent_id: Unique identifier for this agent (e.g. "Brain-Phi3-Mini").
            model:    Ollama model tag (e.g. "phi3:mini", "llama3.2:3b").
        """
        self.agent_id = agent_id
        self.model = model
        self.consensus_threshold = 0.8
        self.is_ready = False
        self._last_reasoning = ""

        print(f"[{self.agent_id}] Initializing Ollama agent with model '{model}'...")
        try:
            # Verify model is available
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                available = [m["name"] for m in resp.json().get("models", [])]
                if model in available:
                    self.is_ready = True
                    print(f"[{self.agent_id}] Model '{model}' is available via Ollama.")
                else:
                    print(f"[{self.agent_id}] Model '{model}' NOT found. Available: {available}")
            else:
                print(f"[{self.agent_id}] Ollama API returned status {resp.status_code}")
        except requests.ConnectionError:
            print(f"[{self.agent_id}] Cannot connect to Ollama. Is it running?")
        except Exception as e:
            print(f"[{self.agent_id}] Error checking Ollama: {e}")

    def analyze_alert(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze a Squad A alert and produce a STIX 2.1 course-of-action.
        Compatible interface with IntelligenceAgent.analyze_alert().
        """
        print(f"[{self.agent_id}] Analyzing alert from {alert['source']}...")

        # Consensus check (same logic as IntelligenceAgent)
        consensus_score = alert.get('ai_score', 0.5)
        if consensus_score < self.consensus_threshold:
            print(f"[{self.agent_id}] Consensus check failed. False positive suspected.")
            return None

        print(f"[{self.agent_id}] Consulting Ollama model '{self.model}' for remediation plan...")
        plan = self._generate_plan(alert)

        if not plan:
            return self._fallback_plan(
                alert['details']['event_name'],
                alert['details'].get('target', 'unknown'),
                alert
            )

        # Safety verification
        action = plan.get('x_epd_action')
        if action == "DELETE_ROOT_ACCOUNT":
            print(f"[{self.agent_id}] Safety verification FAILED.")
            return None

        print(f"[{self.agent_id}] Threat Validated. Remediation Plan Approved: {action}")
        return plan

    def analyze_alert_with_context(self, alert: Dict[str, Any],
                                    peer_proposals: list) -> Optional[Dict[str, Any]]:
        """
        Debate round: re-analyze with knowledge of other agents' proposals.
        """
        print(f"[{self.agent_id}] Re-analyzing with peer context (debate round)...")
        plan = self._generate_plan(alert, peer_proposals=peer_proposals)
        if not plan:
            return self._fallback_plan(
                alert['details']['event_name'],
                alert['details'].get('target', 'unknown'),
                alert
            )
        return plan

    def get_reasoning(self) -> str:
        """Return the reasoning from the last analysis."""
        return self._last_reasoning

    def _generate_plan(self, alert: Dict[str, Any],
                       peer_proposals: list = None) -> Optional[Dict[str, Any]]:
        """
        Query Ollama model for an OpenC2 remediation command.
        """
        if not self.is_ready:
            return None

        threat_type = alert['details']['event_name']
        target = alert['details'].get('target', 'unknown')

        # Build prompt — same OpenC2 instruction as IntelligenceAgent
        prompt = f"""You are an advanced Security Orchestration AI.
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
}}"""

        # If debate round, add peer context
        if peer_proposals:
            peer_text = "\n\nOther security agents have proposed the following actions:\n"
            for p in peer_proposals:
                agent = p.get('agent_id', 'Unknown')
                action = p.get('action', 'unknown')
                reason = p.get('reasoning', 'No reasoning provided')
                peer_text += f"- {agent}: action=\"{action}\" — {reason}\n"
            peer_text += "\nConsidering their analysis, provide your final recommendation as a JSON object."
            prompt += peer_text

        prompt += "\n\nRespond with ONLY the JSON object, nothing else."

        try:
            resp = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150
                    }
                },
                timeout=120
            )

            if resp.status_code != 200:
                print(f"[{self.agent_id}] Ollama API error: {resp.status_code}")
                return None

            response_text = resp.json().get("response", "")
            self._last_reasoning = response_text.strip()

            # Extract JSON from response
            openc2_cmd, action = self._parse_openc2(response_text, threat_type, target)

            print(f"[{self.agent_id}] OpenC2 Action: {action}")

            # Build STIX 2.1 output
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            reason = f"AI selected OpenC2 Action: {action}"

            stix_output = {
                "type": "course-of-action",
                "spec_version": "2.1",
                "id": f"course-of-action--{str(uuid.uuid4())}",
                "created": timestamp,
                "modified": timestamp,
                "name": f"OpenC2 Remediation: {action}",
                "description": f"Automated OpenC2 command triggered by {threat_type} on {target}. Reason: {reason}",
                "x_epd_action": action,
                "x_epd_target": target,
                "x_epd_openc2": openc2_cmd,
                "x_epd_score": alert.get('ai_score', 0.5),
                "x_epd_reason": reason,
                "x_epd_ai_raw_output": response_text,
                "x_epd_input_alert_id": alert.get('log_id', 'unknown'),
                "x_epd_agent_id": self.agent_id
            }

            return stix_output

        except requests.Timeout:
            print(f"[{self.agent_id}] Ollama request timed out.")
            return None
        except Exception as e:
            print(f"[{self.agent_id}] Ollama inference error: {e}")
            return None

    def _parse_openc2(self, response: str, threat_type: str, target: str):
        """
        Extract OpenC2 JSON from model response. Falls back to rule-based
        action selection if parsing fails.
        """
        json_str = ""

        # Try to extract JSON block
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end + 1]

        try:
            openc2_cmd = json.loads(json_str)
            action = openc2_cmd.get('action', 'unknown')
            if action != 'unknown':
                return openc2_cmd, action
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback: threat-type-based action
        return self._fallback_openc2(threat_type, target)

    def _fallback_openc2(self, threat_type: str, target: str):
        """Rule-based OpenC2 fallback — same logic as IntelligenceAgent."""
        threat_upper = threat_type.upper()
        if any(kw in threat_upper for kw in ["FLOOD", "DDOS", "DOS", "BRUTE", "SCAN", "BOT", "FTP", "SSH"]):
            cmd = {
                "action": "deny",
                "target": {"ipv4_connection": {"src_addr": target}},
                "args": {"duration": "1h"}
            }
            return cmd, "deny"
        elif any(kw in threat_upper for kw in ["INFILTER", "MALWARE", "COMPROMIS", "SQL", "XSS", "WEB ATTACK"]):
            cmd = {
                "action": "contain",
                "target": {"device": {"hostname": target}}
            }
            return cmd, "contain"
        else:
            cmd = {
                "action": "query",
                "target": {"features": {"query_type": "status"}}
            }
            return cmd, "query"

    def _fallback_plan(self, threat_type: str, target: str,
                       alert: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a full STIX plan using rule-based fallback."""
        openc2_cmd, action = self._fallback_openc2(threat_type, target)
        self._last_reasoning = f"Fallback rule selected: {action}"
        print(f"[{self.agent_id}] AI output invalid, using Fallback OpenC2: {action}")

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "type": "course-of-action",
            "spec_version": "2.1",
            "id": f"course-of-action--{str(uuid.uuid4())}",
            "created": timestamp,
            "modified": timestamp,
            "name": f"OpenC2 Remediation: {action}",
            "description": f"Automated OpenC2 command triggered by {threat_type} on {target}. Reason: Fallback rule",
            "x_epd_action": action,
            "x_epd_target": target,
            "x_epd_openc2": openc2_cmd,
            "x_epd_score": alert.get('ai_score', 0.5),
            "x_epd_reason": f"AI selected OpenC2 Action: {action}",
            "x_epd_ai_raw_output": "fallback",
            "x_epd_input_alert_id": alert.get('log_id', 'unknown'),
            "x_epd_agent_id": self.agent_id
        }
