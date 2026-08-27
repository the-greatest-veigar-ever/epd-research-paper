"""
Free, local, no-GPU correctness check for the transient-failure retry and
reasoning-model timeout multiplier added to _call_ollama
(src/ghost_agents/approach_evaluation/approaches.py). No pytest dependency --
plain asserts, run directly:

    python3 tests/test_call_retry.py

Motivation: the 2026-08-26 concurrent seed-42 run lost 55/400 deepseek-r1:1.5b
calls -- 16 hard 300s timeouts and 39 "empty" HTTP 200s, the latter returning
at 244-300s next to Ollama "GPU discovery watchdog timed out" log lines, i.e.
the server briefly wedged under 4-way GPU contention rather than the model
choosing silence. These checks pin the two mitigations:

  1. A transient "empty"/"http_error" is retried up to CALL_RETRIES times and
     can recover to "success"; result["attempts"] counts the tries.
  2. A persistent transient failure still ends in that status (not a crash),
     after exactly CALL_RETRIES + 1 attempts.
  3. "truncated" and "length_capped" are deterministic token-cap outcomes and
     are NEVER retried (exactly 1 attempt).
  4. A reasoning model (deepseek-r1, gpt-oss) gets its client timeout scaled
     by REASONING_TIMEOUT_MULT; a dense model does not.
  5. A caller-pinned explicit timeout is passed through untouched, even for a
     reasoning model.

This never touches Ollama, the GPU, or a real pod -- it costs nothing to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation import approaches as A  # noqa: E402


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _install_fake_post(script):
    """script: list of (status_code, payload) served one per call; the last
    entry repeats once the list is exhausted. Returns a dict tracking calls,
    including the `timeout=` kwarg each request was made with."""
    state = {"n": 0, "timeouts": []}

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirror requests.post
        state["timeouts"].append(timeout)
        entry = script[min(state["n"], len(script) - 1)]
        state["n"] += 1
        code, payload = entry
        return _FakeResp(code, payload)

    A.requests.post = fake_post
    return state


def _run(script, model="deepseek-r1:1.5b", **kw):
    state = _install_fake_post(script)
    try:
        A.time.sleep = lambda *_a, **_k: None  # no real backoff wait in tests
        result = A._call_ollama(model, "prompt", seed=42, **kw)
    finally:
        A.requests.post = _real_post
        A.time.sleep = _real_sleep
    return result, state


_real_post = A.requests.post
_real_sleep = A.time.sleep

OK = {"response": "def f():\n    return 1", "done_reason": "stop"}
EMPTY = {"response": "", "done_reason": None}
CAP_NO_ANSWER = {"response": "<think>still thinking", "done_reason": "length"}
CAP_WITH_ANSWER = {"response": "<think>done</think>\ndef f(): pass", "done_reason": "length"}


def test_transient_empty_recovers():
    assert A.CALL_RETRIES >= 1, "test assumes at least one retry configured"
    result, state = _run([(200, EMPTY), (200, OK)])
    assert result["status"] == "success", result["status"]
    assert result["attempts"] == 2, result["attempts"]
    assert state["n"] == 2, state["n"]


def test_persistent_empty_gives_up_cleanly():
    result, state = _run([(200, EMPTY)])
    assert result["status"] == "empty", result["status"]
    assert result["attempts"] == A.CALL_RETRIES + 1, result["attempts"]
    assert state["n"] == A.CALL_RETRIES + 1, state["n"]


def test_http_error_is_retried():
    result, state = _run([(503, {}), (200, OK)])
    assert result["status"] == "success", result["status"]
    assert result["attempts"] == 2, result["attempts"]


def test_truncated_is_not_retried():
    result, state = _run([(200, CAP_NO_ANSWER)])
    assert result["status"] == "truncated", result["status"]
    assert result["attempts"] == 1, result["attempts"]
    assert state["n"] == 1, state["n"]


def test_length_capped_is_not_retried():
    result, state = _run([(200, CAP_WITH_ANSWER)])
    assert result["status"] == "length_capped", result["status"]
    assert result["attempts"] == 1, result["attempts"]


def test_success_first_try_records_single_attempt():
    result, _ = _run([(200, OK)])
    assert result["status"] == "success"
    assert result["attempts"] == 1


def test_reasoning_model_gets_timeout_multiplier():
    _, state = _run([(200, OK)], model="deepseek-r1:1.5b")
    assert state["timeouts"][0] == int(round(A.GENERATION_TIMEOUT_S * A.REASONING_TIMEOUT_MULT)), \
        state["timeouts"]
    _, state2 = _run([(200, OK)], model="gpt-oss:20b")
    assert state2["timeouts"][0] == int(round(A.GENERATION_TIMEOUT_S * A.REASONING_TIMEOUT_MULT))


def test_dense_model_keeps_base_timeout():
    _, state = _run([(200, OK)], model="llama3.2:3b")
    assert state["timeouts"][0] == A.GENERATION_TIMEOUT_S, state["timeouts"]


def test_explicit_timeout_is_never_multiplied():
    _, state = _run([(200, OK)], model="deepseek-r1:1.5b", timeout=99)
    assert state["timeouts"][0] == 99, state["timeouts"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
