"""
Free, local, no-GPU correctness check for reasoning capture and the
truncated/empty split in _call_ollama
(src/ghost_agents/approach_evaluation/approaches.py). No pytest dependency --
plain asserts, run directly:

    python3 tests/test_reasoning_capture.py

Motivation: Ollama returns a reasoning model's chain-of-thought in its own
top-level `thinking` field, not as inline <think> tags inside `response`.
_call_ollama read only `response`, so across all 1,640 seed-42 calls
`reasoning_chars` was 0 and `had_reasoning` False -- even for deepseek-r1 and
gpt-oss, both reasoning models -- and the "truncated" status, which needs
inline tags to detect, fired exactly zero times. A reasoning model that spent
its entire token budget thinking was therefore recorded as "empty",
indistinguishable from an infrastructure failure, and was then pointlessly
retried (same seed, temperature 0 -> same result).

Checks:
  1. The native `thinking` field lands in result["reasoning"].
  2. Inline <think> tags still work, for any model that does inline them.
  3. The native field wins when both are somehow present.
  4. No answer + reasoning + done_reason "length" -> truncated (raise the cap).
  5. No answer + reasoning + done_reason "stop" -> empty (not a cap problem).
  6. No answer + no reasoning at all -> empty.
  7. truncated is NEVER retried; empty still is.
  8. eval_count is carried onto the result -- it spans reasoning and answer
     together, so it is the number num_predict must be calibrated against.

This never touches Ollama, the GPU, or a real pod -- it costs nothing to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation import approaches as A  # noqa: E402

_real_post = A.requests.post
_real_sleep = A.time.sleep


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _call(payload, model="gpt-oss:20b", **kw):
    """Serve `payload` (a dict, or a list served one per attempt)."""
    script = payload if isinstance(payload, list) else [payload]
    state = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        entry = script[min(state["n"], len(script) - 1)]
        state["n"] += 1
        return _FakeResp(entry)

    A.requests.post = fake_post
    A.time.sleep = lambda *a, **k: None
    try:
        return A._call_ollama(model, "prompt", seed=42, **kw), state
    finally:
        A.requests.post = _real_post
        A.time.sleep = _real_sleep


def test_native_thinking_field_is_captured():
    r, _ = _call({"response": "def f(): pass", "thinking": "let me consider...",
                  "done_reason": "stop", "eval_count": 400})
    assert r["status"] == "success", r
    assert r["reasoning"] == "let me consider...", r["reasoning"]
    assert r["command"] == "def f(): pass", r["command"]


def test_inline_think_tags_still_work():
    r, _ = _call({"response": "<think>hmm</think>\ndef f(): pass", "done_reason": "stop"})
    assert r["status"] == "success", r
    assert "hmm" in (r["reasoning"] or ""), r["reasoning"]
    assert r["command"] == "def f(): pass", r["command"]


def test_native_field_wins_over_inline():
    r, _ = _call({"response": "<think>inline</think>\nanswer", "thinking": "native",
                  "done_reason": "stop"})
    assert r["reasoning"] == "native", r["reasoning"]


def test_cap_hit_during_reasoning_is_truncated():
    r, _ = _call({"response": "", "thinking": "x" * 500, "done_reason": "length",
                  "eval_count": 2048})
    assert r["status"] == "truncated", r["status"]
    assert "cap during reasoning" in r["error"], r["error"]
    assert "500 chars" in r["error"], r["error"]


def test_no_answer_but_stopped_naturally_is_empty():
    """Reasoning happened, the model then chose to say nothing. Not a cap
    problem, so raising num_predict would not help -- do not call it truncated."""
    r, _ = _call({"response": "", "thinking": "x" * 500, "done_reason": "stop"})
    assert r["status"] == "empty", r["status"]


def test_nothing_at_all_is_empty():
    r, _ = _call({"response": "", "done_reason": "stop"})
    assert r["status"] == "empty", r["status"]
    assert r["reasoning"] is None, r["reasoning"]


def test_truncated_is_never_retried():
    assert A.CALL_RETRIES >= 1, "test assumes retries are enabled"
    r, state = _call({"response": "", "thinking": "x" * 100, "done_reason": "length"})
    assert r["status"] == "truncated"
    assert r["attempts"] == 1, r["attempts"]
    assert state["n"] == 1, state["n"]


def test_empty_is_still_retried():
    r, state = _call([
        {"response": "", "done_reason": "stop"},
        {"response": "ok", "done_reason": "stop"},
    ])
    assert r["status"] == "success", r["status"]
    assert r["attempts"] == 2, r["attempts"]


def test_eval_count_is_recorded():
    r, _ = _call({"response": "ok", "thinking": "t", "done_reason": "stop", "eval_count": 1734})
    assert r["eval_count"] == 1734, r


def test_length_capped_still_detected_with_native_reasoning():
    """An answer exists but the cap cut it off -- distinct from truncated."""
    r, _ = _call({"response": "def f(): pa", "thinking": "t", "done_reason": "length"})
    assert r["status"] == "length_capped", r["status"]
    assert r["reasoning"] == "t"


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
