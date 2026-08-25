"""
Ollama Model Lifecycle Manager.

This module provides the core interface for managing the local Ollama execution
environment. It facilitates the explicit preloading and unloading of model
weights from system memory, which is critical for enforcing the isolation
properties required by the Ephemeral Polymorphic Defense (EPD) strategy.
"""

import os
import requests
import time
from typing import Optional


def _resolve_base_url() -> str:
    """
    Endpoint for this process's Ollama server.

    In the sequential topology every process shared one server on the
    default port. The concurrent topology gives each model its own
    dedicated `ollama serve` on its own port, because that server process
    is what makes per-model resource attribution possible at all: it is
    the stable anchor the monitor tracks, and it survives the unload /
    reload churn an ephemeral cell performs on every call (see
    per_process_monitor for the full argument).

    EPD_OLLAMA_URL wins if set; otherwise EPD_OLLAMA_PORT selects the port
    on localhost. Unset, this is the original single-server default, so
    the sequential scripts keep working untouched.
    """
    explicit = os.environ.get("EPD_OLLAMA_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("EPD_OLLAMA_PORT", "").strip() or "11434"
    return f"http://localhost:{port}"


OLLAMA_BASE_URL = _resolve_base_url()

# Env-tunable lifecycle timeouts (see approaches.py for the matching
# generation limits). Under heavy concurrency a preload can legitimately
# queue behind other in-flight work, so the default has real headroom --
# the previous 120s default was routinely exceeded on the RunPod run,
# producing a flood of "preload failed" warnings.
PRELOAD_TIMEOUT_S = int(os.environ.get("EPD_PRELOAD_TIMEOUT", "600"))
UNLOAD_TIMEOUT_S = int(os.environ.get("EPD_UNLOAD_TIMEOUT", "120"))


def preload_model(model: str, timeout: Optional[int] = None) -> float:
    """
    Initializes a model within the Ollama execution environment.

    This function transmits a minimal payload to the Ollama REST API to force
    the allocation of model weights into system memory.

    Args:
        model (str): The specific model identifier to be loaded.
        timeout (int, optional): The maximum duration, in seconds, to wait for
            the loading operation to complete. Defaults to 120.

    Returns:
        float: The duration of the initialization process in seconds.
        
    Raises:
        requests.exceptions.RequestException: If the communication with the
            Ollama API fails.
    """
    timeout = PRELOAD_TIMEOUT_S if timeout is None else timeout
    t_start = time.perf_counter()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=timeout
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[OllamaManager] Warning: preload failed for {model}: {e}")
    
    duration = time.perf_counter() - t_start
    return duration


def unload_model(model: str, timeout: Optional[int] = None) -> float:
    """
    Terminates a model instance and deallocates its memory footprint.

    This function enforces the ephemeral isolation constraints by issuing a
    keep_alive=0 directive to the Ollama API, purging the specified model.

    Args:
        model (str): The specific model identifier to be terminated.
        timeout (int, optional): The maximum duration, in seconds, to wait for
            the deallocation operation to complete. Defaults to 30.

    Returns:
        float: The duration of the deallocation process in seconds.

    Raises:
        requests.exceptions.RequestException: If the communication with the
            Ollama API fails.
    """
    timeout = UNLOAD_TIMEOUT_S if timeout is None else timeout
    t_start = time.perf_counter()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0
            },
            timeout=timeout
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[OllamaManager] Warning: unload failed for {model}: {e}")
    
    duration = time.perf_counter() - t_start
    return duration


def get_running_models() -> list:
    """
    Retrieves the registry of currently active models in the execution environment.

    Queries the Ollama process state endpoint to determine which models
    are currently occupying system memory.

    Returns:
        list: A collection of string identifiers representing the active models.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"[OllamaManager] Warning: could not list running models: {e}")
        return []


def is_model_loaded(model: str) -> bool:
    """
    Determines whether a specified model is active in system memory.

    Args:
        model (str): The identifier of the model to query.

    Returns:
        bool: True if the model is currently active, False otherwise.
    """
    running = get_running_models()
    # Ollama sometimes appends ":latest" to model names
    return any(model in r or r in model for r in running)


def unload_all_models() -> float:
    """
    Executes a complete purge of the execution environment.

    Iterates through the registry of active models and sequentially terminates
    each instance to guarantee a zero-state environment.

    Returns:
        float: The cumulative duration of the purge operations in seconds.
    """
    total_time = 0.0
    running = get_running_models()
    for model in running:
        total_time += unload_model(model)
    return total_time
