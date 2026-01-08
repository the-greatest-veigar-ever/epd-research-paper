# MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber Defense

**Authors**: Arth Bhardwaj, Sia Godika, Yuvam Loonker  
**Source**: arXiv:2512.14846v1 [cs.CR] 16 Dec 2025

## Summary
**MALCDF (Multi-Agent LLM Cyber Defense Framework)** is a decentralized security framework designed to detect, analyze, and mitigate cyber threats in real-time. It replaces traditional centralized tools with a cooperative ecosystem of four specialized Large Language Model (LLM) agents that communicate via a secure, ontology-aligned layer.

## Key Problem
Traditional security tools (IDS, Firewalls, statically trained ML models) struggle with:
*   **Adaptive Attacks**: Attackers changing behavior mid-incident.
*   **Context Blindness**: Lack of semantic understanding of logs and multi-stage campaigns.
*   **Centralized Failure Points**: Single points of failure in monolithic architectures.
*   **False Positives**: High noise levels from outdated signature databases.

## Proposed Solution: The Agents
MALCDF deploys four distinct agents working in concert:
1.  **Threat Detection Agent (TDA)**: Monitors raw logs and classifies events (e.g., flagging high-rate UDP traffic).
2.  **Threat Intelligence Agent (TIA)**: Enriches alerts with external context (e.g., linking an IP to a known APT campaign).
3.  **Response Coordination Agent (RCA)**: Proposes actionable mitigation steps (e.g., blocking a port, isolating a host).
4.  **Analyst Agent (AA)**: Generates human-readable incident reports mapped to the MITRE ATT&CK framework.

**Key Technology**:
*   **Secure Communication Layer (SCL)**: Ensures encrypted, structured (ontology-aligned) message passing between agents.
*   **LLM Core**: Built on LLaMA 3.3 70B (Groq API).

## Key Results
Tested on a live stream derived from the **CICIDS2017** dataset:
*   **Accuracy**: **90.0%** (vs 80.0% for ML Baseline).
*   **F1-Score**: **85.7%** (High precision and recall).
*   **False Positive Rate**: **9.1%** (Lower than single-agent setups).
*   **Latency**: **6.8s** per event (Acceptable for real-time human-in-the-loop review).
