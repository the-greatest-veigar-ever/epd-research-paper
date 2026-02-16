"""
Ollama Model Lifecycle Manager

Handles loading/unloading models for the suicide mechanism evaluation.
Uses the Ollama REST API to control model lifecycle.
"""

import requests
import time
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"


def preload_model(model: str, timeout: int = 120) -> float:
    """
    Warm-load a model into Ollama memory by sending a minimal prompt.
    
    Returns:
        float: Time in seconds to load the model.
    """
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


def unload_model(model: str, timeout: int = 30) -> float:
    """
    Unload a model from Ollama memory using keep_alive=0.
    
    Returns:
        float: Time in seconds to unload the model.
    """
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
    Get list of currently loaded models in Ollama.
    
    Returns:
        list: List of model name strings currently in memory.
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
    """Check if a specific model is currently loaded in memory."""
    running = get_running_models()
    # Ollama sometimes appends ":latest" to model names
    return any(model in r or r in model for r in running)


def unload_all_models() -> float:
    """
    Unload all currently loaded models from memory.
    
    Returns:
        float: Total time to unload all models.
    """
    total_time = 0.0
    running = get_running_models()
    for model in running:
        total_time += unload_model(model)
    return total_time
