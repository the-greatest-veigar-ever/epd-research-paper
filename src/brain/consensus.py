"""
SquadBConsensus — SentinelNet-Inspired Multi-Agent Debate & Consensus.

Implements credit-based weighted voting with optional debate rounds
when agents disagree. Adapted from:
  SentinelNet: Safeguarding Multi-Agent Collaboration Through
  Credit-Based Dynamic Threat Detection (The Web Conference 2026).

Credit scores reset per run (not persisted across sessions).
"""
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from collections import Counter


class SquadBConsensus:
    """
    Orchestrates multiple Squad B agents to reach consensus on
    threat remediation using SentinelNet-inspired debate.
    """

    # Credit adjustment per alert
    CREDIT_REWARD = 0.1    # Given to agents agreeing with majority
    CREDIT_PENALTY = 0.1   # Deducted from agents disagreeing
    CREDIT_MIN = 0.1       # Floor to prevent zero-weight agents
    CREDIT_INITIAL = 1.0   # Starting credit for all agents

    def __init__(self, agents: list):
        """
        Args:
            agents: List of Squad B agents (IntelligenceAgent or OllamaIntelligenceAgent).
                    Each must implement analyze_alert(), get_reasoning(), and have agent_id.
        """
        self.agents = agents
        self.credits = {a.agent_id: self.CREDIT_INITIAL for a in agents}
        self._alert_count = 0
        self._consensus_log = []

        print(f"\n[SquadB-Consensus] Initialized with {len(agents)} agents:")
        for a in agents:
            print(f"  - {a.agent_id} (credit: {self.credits[a.agent_id]:.1f})")

    def analyze_alert(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Runs the SentinelNet consensus flow:
        1. Parallel independent analysis
        2. Majority check (fast path)
        3. Debate round (if needed)
        4. Credit-weighted vote (if still no majority)

        Returns a STIX 2.1 course-of-action with consensus metadata.
        """
        self._alert_count += 1

        # ──────────────────────────────────────────
        # Step 1: Independent Analysis (All Agents)
        # ──────────────────────────────────────────
        proposals = []
        for agent in self.agents:
            try:
                result = agent.analyze_alert(alert)
                if result:
                    proposals.append({
                        "agent_id": agent.agent_id,
                        "plan": result,
                        "action": result.get("x_epd_action", "unknown"),
                        "reasoning": agent.get_reasoning() if hasattr(agent, 'get_reasoning') else ""
                    })
            except Exception as e:
                print(f"[SquadB-Consensus] Agent {agent.agent_id} failed: {e}")

        if not proposals:
            print("[SquadB-Consensus] All agents failed. No consensus possible.")
            return None

        # ──────────────────────────────────────
        # Step 2: Majority Vote (Fast Path)
        # ──────────────────────────────────────
        actions = [p["action"] for p in proposals]
        action_counts = Counter(actions)
        majority_action, majority_count = action_counts.most_common(1)[0]

        consensus_record = {
            "alert_index": self._alert_count,
            "round": 1,
            "agent_votes": [
                {"agent": p["agent_id"], "action": p["action"],
                 "credit": self.credits[p["agent_id"]]}
                for p in proposals
            ]
        }

        if majority_count >= 2:
            # Fast path: ≥2 agents agree
            consensus_record["method"] = "majority_vote"
            consensus_record["agreement"] = f"{majority_count}/{len(proposals)}"
            consensus_record["final_action"] = majority_action

            self._update_credits(proposals, majority_action)
            self._consensus_log.append(consensus_record)

            # Use the plan from the highest-credit agent in the majority
            final_plan = self._select_best_plan(proposals, majority_action)
            final_plan["x_epd_consensus"] = consensus_record
            print(f"[SquadB-Consensus] ✅ Majority ({majority_count}/{len(proposals)}): {majority_action}")
            return final_plan

        # ──────────────────────────────────────────
        # Step 3: Debate Round (All disagree)
        # ──────────────────────────────────────────
        print(f"[SquadB-Consensus] ⚡ No majority — initiating debate round...")

        debate_proposals = []
        peer_context = [
            {"agent_id": p["agent_id"], "action": p["action"], "reasoning": p["reasoning"]}
            for p in proposals
        ]

        for agent in self.agents:
            try:
                if hasattr(agent, 'analyze_alert_with_context'):
                    result = agent.analyze_alert_with_context(alert, peer_context)
                else:
                    # For agents without debate support, keep original answer
                    result = next(
                        (p["plan"] for p in proposals if p["agent_id"] == agent.agent_id),
                        None
                    )
                if result:
                    debate_proposals.append({
                        "agent_id": agent.agent_id,
                        "plan": result,
                        "action": result.get("x_epd_action", "unknown"),
                        "reasoning": agent.get_reasoning() if hasattr(agent, 'get_reasoning') else ""
                    })
            except Exception as e:
                print(f"[SquadB-Consensus] Debate failed for {agent.agent_id}: {e}")

        if not debate_proposals:
            debate_proposals = proposals  # Fall back to round 1

        # Check majority after debate
        debate_actions = [p["action"] for p in debate_proposals]
        debate_counts = Counter(debate_actions)
        debate_majority, debate_majority_count = debate_counts.most_common(1)[0]

        consensus_record["round"] = 2
        consensus_record["debate_votes"] = [
            {"agent": p["agent_id"], "action": p["action"],
             "credit": self.credits[p["agent_id"]]}
            for p in debate_proposals
        ]

        if debate_majority_count >= 2:
            consensus_record["method"] = "debate_majority_vote"
            consensus_record["agreement"] = f"{debate_majority_count}/{len(debate_proposals)}"
            consensus_record["final_action"] = debate_majority

            self._update_credits(debate_proposals, debate_majority)
            self._consensus_log.append(consensus_record)

            final_plan = self._select_best_plan(debate_proposals, debate_majority)
            final_plan["x_epd_consensus"] = consensus_record
            print(f"[SquadB-Consensus] ✅ Debate majority ({debate_majority_count}/{len(debate_proposals)}): {debate_majority}")
            return final_plan

        # ──────────────────────────────────────────────────
        # Step 4: Credit-Weighted Vote (Still no majority)
        # ──────────────────────────────────────────────────
        print(f"[SquadB-Consensus] ⚖️  No debate majority — using credit-weighted vote...")

        weighted_scores = {}
        for p in debate_proposals:
            action = p["action"]
            credit = self.credits[p["agent_id"]]
            weighted_scores[action] = weighted_scores.get(action, 0) + credit

        winning_action = max(weighted_scores, key=weighted_scores.get)

        consensus_record["method"] = "credit_weighted_vote"
        consensus_record["weighted_scores"] = weighted_scores
        consensus_record["final_action"] = winning_action
        consensus_record["agreement"] = f"weighted"

        self._update_credits(debate_proposals, winning_action)
        self._consensus_log.append(consensus_record)

        final_plan = self._select_best_plan(debate_proposals, winning_action)
        final_plan["x_epd_consensus"] = consensus_record
        print(f"[SquadB-Consensus] ✅ Weighted vote: {winning_action} (scores: {weighted_scores})")
        return final_plan

    def _update_credits(self, proposals: list, majority_action: str):
        """
        Update agent credits based on agreement with majority.
        SentinelNet credit update rule: reward agreement, penalize disagreement.
        """
        for p in proposals:
            agent_id = p["agent_id"]
            if p["action"] == majority_action:
                self.credits[agent_id] = min(
                    self.credits[agent_id] + self.CREDIT_REWARD, 2.0
                )
            else:
                self.credits[agent_id] = max(
                    self.credits[agent_id] - self.CREDIT_PENALTY,
                    self.CREDIT_MIN
                )

    def _select_best_plan(self, proposals: list, target_action: str) -> Dict[str, Any]:
        """
        Among proposals with the target action, select the one from
        the agent with the highest credit score.
        """
        matching = [p for p in proposals if p["action"] == target_action]
        if not matching:
            return proposals[0]["plan"]

        best = max(matching, key=lambda p: self.credits[p["agent_id"]])
        return best["plan"]

    def get_credits(self) -> Dict[str, float]:
        """Return current credit scores for all agents."""
        return dict(self.credits)

    def get_consensus_log(self) -> list:
        """Return the full consensus log for this run."""
        return self._consensus_log

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of consensus decisions."""
        methods = Counter(r.get("method") for r in self._consensus_log)
        return {
            "total_alerts": self._alert_count,
            "total_decisions": len(self._consensus_log),
            "decision_methods": dict(methods),
            "final_credits": dict(self.credits)
        }
