# SentinelNet: Safeguarding Multi-Agent Collaboration

**Authors**: Yang Feng, Xudong Pan  
**Source**: The Web Conference 2026 (Under Review)

## Summary
**SentinelNet** is a decentralized defense framework designed to protect Multi-Agent Systems (MAS) from malicious agents. It proactively detects and removes compromised agents during collaborative debates without needing a central authority.

## Key Problem
In Multi-Agent Debates (MAD), a single malicious agent can disrupt the entire system by:
*   **Spreading Misinformation**: Steering the group toward wrong answers.
*   **Persuasive Attacks**: Using social engineering to convince other agents of false claims.
*   **Hypnotic Jailbreaks**: Injecting prompts that propagate through the network.
*   **Existing Defenses**: Are either too reactive (only acting after the damage is done) or too centralized (creating bottlenecks).

## Proposed Solution: Sentinel Agents
SentinelNet transforms standard agents into "Sentinel Nodes" equipped with a **Credit-Based Threat Detector**.

1.  **Decentralized Detection**: Each agent autonomously evaluates the credibility of incoming messages from peers.
2.  **Contrastive Learning**: The detector is trained to rank "constructive" responses higher than "adversarial" ones by analyzing debate trajectories.
3.  **Dynamic Ranking**: Agents maintain a credit score for their peers.
4.  **Bottom-k Elimination**: In every round, the lowest-scoring agents (most suspicious) are ignored or quarantined, isolating the threat.

## Key Results
Tested on 6 MAS benchmarks (MMLU, CommonsenseQA, GSM8K) against various attacks:
*   **Detection Accuracy**: Near **100%** within two debate rounds.
*   **System Recovery**: Restored **95%** of system accuracy even when agents were compromised.
*   **Resilience**: Effective against Collaboration Attacks, NetSafe Attacks, and Adversary-in-the-Middle (AITM).
