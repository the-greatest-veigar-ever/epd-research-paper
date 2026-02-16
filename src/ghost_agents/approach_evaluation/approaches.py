"""
Ghost Agent Approach Definitions

4 approaches for comparative evaluation:
1. Phi Baseline (static single model)
2. Phi Suicide (ephemeral single model)
3. Multimodal Static (random selection, all pre-loaded)
4. Multimodal Suicide (random selection, load on demand)
"""

import random
import time
import uuid
import requests
from typing import Dict, Any, List
from abc import ABC, abstractmethod

from src.ghost_agents.approach_evaluation.ollama_manager import (
    preload_model,
    unload_model,
    unload_all_models,
)

OLLAMA_URL = "http://localhost:11434/api/generate"


class Approach(ABC):
    """Base class for all evaluation approaches."""

    name: str
    models: List[str]
    suicide_mode: bool

    @abstractmethod
    def initialize(self) -> float:
        """
        Pre-flight initialization (e.g. preload models).
        Returns time in seconds.
        """
        ...

    @abstractmethod
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single remediation plan and return results with timing.

        Args:
            plan: dict with 'action' and 'target' keys.

        Returns:
            dict with keys: status, command, tool_used, init_time,
                            processing_time, model_used
        """
        ...

    @abstractmethod
    def teardown(self):
        """Cleanup after evaluation run."""
        ...


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _call_ollama(model: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Send a prompt to an Ollama model and return the parsed result.

    Returns:
        dict with 'status', 'command', 'tool_used'
    """
    result = {
        "status": "failed",
        "command": None,
        "tool_used": None,
        "error": None,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            cmd = response.json().get("response", "").strip()

            # Clean markdown code blocks
            clean_cmd = cmd
            if "```" in clean_cmd:
                clean_lines = [
                    line for line in clean_cmd.split("\n") if "```" not in line
                ]
                clean_cmd = "\n".join(clean_lines).strip()

            tool = clean_cmd.split()[0] if clean_cmd else "unknown"

            result["status"] = "success"
            result["command"] = cmd
            result["tool_used"] = tool
        else:
            result["error"] = f"HTTP {response.status_code}"
    except Exception as e:
        result["status"] = "simulated_success"
        result["error"] = str(e)

    return result


def _build_prompt(action: str, target: str) -> str:
    """Build the remediation prompt for the Ghost Agent."""
    return (
        f"ROLE: You are an EPD Remediation Agent.\n"
        f"TASK: Generate the specific AWS CLI command to perform: "
        f"{action} on target: {target}.\n"
        f"OUTPUT: Only the command.\n"
    )


# ===========================================================================
# Approach 1: Phi Baseline (Static)
# ===========================================================================

class PhiBaselineApproach(Approach):
    """Single phi3:mini model, kept loaded throughout the evaluation."""

    name = "phi_baseline"
    models = ["phi3:mini"]
    suicide_mode = False

    def initialize(self) -> float:
        """Preload phi3:mini so it's warm for all executions."""
        print(f"[{self.name}] Preloading {self.models[0]}...")
        return preload_model(self.models[0])

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        model = self.models[0]
        prompt = _build_prompt(plan["action"], plan["target"])

        t_start = time.perf_counter()
        result = _call_ollama(model, prompt)
        processing_time = time.perf_counter() - t_start

        result["init_time"] = 0.0  # already loaded
        result["processing_time"] = processing_time
        result["model_used"] = model
        return result

    def teardown(self):
        pass  # leave model loaded


# ===========================================================================
# Approach 2: Phi Suicide (Ephemeral)
# ===========================================================================

class PhiSuicideApproach(Approach):
    """Single phi3:mini model, loaded on demand and unloaded after each execution."""

    name = "phi_suicide"
    models = ["phi3:mini"]
    suicide_mode = True

    def initialize(self) -> float:
        """No pre-loading — measure cold-start per execution."""
        print(f"[{self.name}] Suicide mode — no preload.")
        # Ensure clean state
        unload_all_models()
        return 0.0

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        model = self.models[0]
        prompt = _build_prompt(plan["action"], plan["target"])

        # 1. Load model (init_time)
        init_time = preload_model(model)

        # 2. Execute inference (processing_time)
        t_proc_start = time.perf_counter()
        result = _call_ollama(model, prompt)
        processing_time = time.perf_counter() - t_proc_start

        # 3. Suicide — unload model
        unload_model(model)

        result["init_time"] = init_time
        result["processing_time"] = processing_time
        result["model_used"] = model
        return result

    def teardown(self):
        unload_all_models()


# ===========================================================================
# Approach 3: Multimodal Static (Random Selection)
# ===========================================================================

class MultimodalStaticApproach(Approach):
    """3 models pre-loaded, randomly selected per execution."""

    name = "multimodal_static"
    models = ["phi3:mini", "llama3.2:3b", "gemma2:2b"]
    suicide_mode = False

    def initialize(self) -> float:
        """Preload all 3 models."""
        total_time = 0.0
        for model in self.models:
            print(f"[{self.name}] Preloading {model}...")
            total_time += preload_model(model)
        return total_time

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        model = random.choice(self.models)
        prompt = _build_prompt(plan["action"], plan["target"])

        t_start = time.perf_counter()
        result = _call_ollama(model, prompt)
        processing_time = time.perf_counter() - t_start

        result["init_time"] = 0.0  # all models already warm
        result["processing_time"] = processing_time
        result["model_used"] = model
        return result

    def teardown(self):
        pass  # leave models loaded


# ===========================================================================
# Approach 4: Multimodal Suicide (Random Selection + Ephemeral)
# ===========================================================================

class MultimodalSuicideApproach(Approach):
    """3 models, randomly selected per execution, loaded on demand and unloaded after."""

    name = "multimodal_suicide"
    models = ["phi3:mini", "llama3.2:3b", "gemma2:2b"]
    suicide_mode = True

    def initialize(self) -> float:
        """No pre-loading — measure cold-start per execution."""
        print(f"[{self.name}] Suicide mode — no preload.")
        unload_all_models()
        return 0.0

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        model = random.choice(self.models)
        prompt = _build_prompt(plan["action"], plan["target"])

        # 1. Load model (init_time)
        init_time = preload_model(model)

        # 2. Execute inference (processing_time)
        t_proc_start = time.perf_counter()
        result = _call_ollama(model, prompt)
        processing_time = time.perf_counter() - t_proc_start

        # 3. Suicide — unload model
        unload_model(model)

        result["init_time"] = init_time
        result["processing_time"] = processing_time
        result["model_used"] = model
        return result

    def teardown(self):
        unload_all_models()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_APPROACHES = {
    "phi_baseline": PhiBaselineApproach,
    "phi_suicide": PhiSuicideApproach,
    "multimodal_static": MultimodalStaticApproach,
    "multimodal_suicide": MultimodalSuicideApproach,
}
