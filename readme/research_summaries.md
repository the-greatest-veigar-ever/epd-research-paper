# EPD Research References & Summaries

This document consolidates foundational concepts and academic references used in the Ephemeral Polymorphic Defense (EPD) project.

---

## 1. Foundational Concepts

### 1.1 Cloud-Native Application Protection Platform (CNAPP)

**Cloud-Native Application Protection Platform (CNAPP)** is a unified security solution designed to secure specialized cloud-native applications across their entire lifecycle—from development ("shift left") to production operations ("shield right").

**Core Components:**
1. **Cloud Security Posture Management (CSPM)**: Monitors cloud infrastructure for misconfigurations
2. **Cloud Workload Protection Platform (CWPP)**: Protects VMs, Containers, and Serverless functions
3. **Cloud Infrastructure Entitlement Management (CIEM)**: Manages identities and permissions
4. **Infrastructure as Code (IaC) Scanning**: Scans templates before deployment
5. **Kubernetes Security Posture Management (KSPM)**: Secures container orchestration

**Key Benefits:**
- Unified Visibility: Single "pane of glass" for all cloud risks
- Context-Aware: Prioritizes alerts by understanding relationships
- DevSecOps Integration: Empowers developers to fix issues early

---

### 1.2 Cloud-Native Applications

A **Cloud-Native Application** is software specifically designed to exploit the scalability, elasticity, and distributed nature of modern cloud computing.

**Key Pillars (CNCF definition):**
1. **Microservices**: Small, independent services for specific functions
2. **Containers**: Lightweight, portable units (Docker)
3. **Continuous Delivery (CI/CD)**: Automated pipelines for rapid releases
4. **DevOps & Automation**: Infrastructure as Code, collaboration

---

### 1.3 AI Model Warm-up & Latency

Research on Time to First Token (TTFT) for modern LLMs:

| Model | Provider | Average TTFT | Comparison to 1.5s |
|:---|:---|:---|:---|
| GPT-4o | OpenAI | ~0.56s | 63% Faster |
| Claude 3.5 Haiku | Anthropic | 0.71s | 53% Faster |
| Llama Nemotron 49B | NVIDIA/Groq | ~0.15-0.45s | 80%+ Faster |

> **EPD Recommendation**: Implement randomized latency distribution based on model selection.

---

## 2. Academic Paper Summaries

### 2.1 MALCDF: Multi-Agent LLM Cyber Defense Framework
**Source**: arXiv:2512.14846v1 [cs.CR] Dec 2025

Decentralized security framework with four specialized LLM agents:
1. **Threat Detection Agent (TDA)**: Monitors and classifies events
2. **Threat Intelligence Agent (TIA)**: Enriches alerts with external context
3. **Response Coordination Agent (RCA)**: Proposes mitigation steps
4. **Analyst Agent (AA)**: Generates human-readable reports

**Results (CICIDS2017):**
- Accuracy: **90.0%** (vs 80.0% baseline)
- F1-Score: **85.7%**
- False Positive Rate: **9.1%**

---

### 2.2 SentinelNet: Safeguarding Multi-Agent Collaboration
**Source**: The Web Conference 2026 (Under Review)

Decentralized defense framework protecting Multi-Agent Systems from malicious agents:
- **Credit-Based Threat Detector**: Autonomous credibility evaluation
- **Contrastive Learning**: Ranks constructive vs adversarial responses
- **Bottom-k Elimination**: Quarantines suspicious agents

**Results:**
- Detection Accuracy: Near **100%** within two rounds
- System Recovery: Restored **95%** accuracy when compromised

---

### 2.3 VeriGuard: Enhancing LLM Agent Safety
**Source**: ICLR 2026 (Under Review)

Framework providing formal mathematical guarantees for LLM agent safety:

**Dual-stage Pipeline:**
1. **Offline Policy Generation**: Code + Formal Constraints verified by Nagini/Viper
2. **Online Policy Enforcement**: Runtime action validation with mathematical certainty

**Key Innovation**: "Correct-by-Construction" policies enable deployment in high-stakes environments.

---

## 3. EPD Solution Overview

### 3.1 The Problem: "Sitting Ducks"
Traditional security bots are **persistent** and **static**—hackers can test them repeatedly to find jailbreaks.

### 3.2 The EPD Solution: Ghost Squads
1. **No Permanent Guards**: System empty until threat detected
2. **Born on Demand**: Ghost Agents created per-threat
3. **Polymorphism**: Different models (GPT-4o, Claude, Llama) and mutating prompts each time
4. **Self-Destruct**: Agents delete memory and vanish after task

### 3.3 Technical Architecture
EPD employs **Moving Target Defense (MTD)** at the cognitive layer:

| Concept | Mechanism | Benefit |
|:---|:---|:---|
| **Ephemerality** | Just-in-Time instantiation | Eliminates context accumulation attacks |
| **Cognitive Polymorphism** | Model & Prompt rotation | Prevents model inversion/prompt injection |
| **Distributed Consensus** | SentinelNet/VeriGuard validation | Ensures safe remediation actions |

---

## References

1. MALCDF Paper: `/reference/MALCDF_A_Distributed_Multi-Agent_LLM_Framework_for.pdf`
2. SentinelNet Paper: `/reference/SentinelNet_Safeguarding_Multi-Agent_Collaboration.pdf`
3. VeriGuard Paper: `/reference/VeriGuard - Enhancing LLM Agent Safety.pdf`
4. Vellum.ai LLM Leaderboard: https://www.vellum.ai/llm-leaderboard
5. Artificial Analysis: https://artificialanalysis.ai/leaderboards/models

---

*Consolidated from: CNAPP.md, Cloud_Native.md, AI_Model_Warmup_Research.md, MALCDF.md, SentinelNet.md, VeriGuard.md, Solution_Human.md, Solution_Technical.md*
