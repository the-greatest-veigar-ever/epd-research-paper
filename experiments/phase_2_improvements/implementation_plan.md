# Phase 2: Architecture Improvements Implementation Plan

**Goal:** Improve the accuracy of the Ghost Agents architecture while maintaining its security/evasion benefits.

## 1. Reduce Polymorphism Intensity
**Problem:** Phase 1 results showed a 89% -> 10% drop in accuracy due to overly complex/abstract persona prompts.
**Solution:** Refine `_mutate_prompt` in `agent.py` to produce "Technical Personas" rather than abstract ones.
- **Before:** "You are a Pirate Captain. Block the IP."
- **After:** "You are a Strictly Compliant Sysadmin. Execute IP block protocol."
- **Action:** Create `src/ghost_agents/agent_v2.py` with modified persona list and prompt logic.

## 2. Hybrid Architecture (Route-based)
**Problem:** SLMs fail on complex instruction following.
**Solution:** Implement a router that sends "Critical/Complex" tasks to a verified Monolithic LLM (`llama3`) and "Routine/High-Volume" tasks to the Ghost Swarm.
- **Mechanism:**
  - `Brain-Eval` assigns a "Complexity Score" or "Criticality Level".
  - If `Criticality > Threshold` -> Use `llama3` (Static but Reliable).
  - If `Criticality <= Threshold` -> Use `Ghost Swarm` (Ephemeral & Evasive).
- **Action:** Update `full_pipeline_evaluation.py` to support hybrid routing.

## 3. Improved Prompting for SLMs
**Problem:** SLMs hallucinate commands (e.g. `aws ec2 block-ip`).
**Solution:** Inject strict one-shot examples into the `GHOST_SYSTEM_PROMPT`.
- **Action:** Update `src/ghost_agents/agent_v2.py` to include:
  ```
  Example:
  Instruction: Block IP 1.2.3.4
  Command: aws ec2 create-network-acl-entry ...
  ```

## 4. Evaluation Strategy
- Run `experiments/phase_2_improvements/eval_v2.py` (modified from `full_pipeline_evaluation.py`).
- Compare specific metrics:
  - **Tool Correctness**: Target > 80%.
  - **Semantic Similarity**: Target < 0.65 (maintain evasion).
