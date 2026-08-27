"""
Free, local, no-GPU correctness check for the checkpoint generation-config
fingerprint (src/ghost_agents/approach_evaluation/benchmark_evaluator.py).
No pytest dependency -- plain asserts, run directly:

    python3 tests/test_checkpoint_config.py

Motivation: before this, a checkpoint recorded only `max_per_benchmark`, so
changing `num_predict` and re-running would silently resume -- leaving one
model's dataset holding some cells generated at one token cap and some at
another, with nothing in the output recording the split. The 2026-08-27 data
audit flagged that as a live hazard: phi3_mini (108/400 length_capped) and
qwen25_3b (57/400) both need their caps raised, and both have existing
seed-42 checkpoints on disk.

Checks:
  1. _generation_config captures the per-model num_predict, not the global
     default, and carries the content knobs plus provenance-only fields.
  2. An identical config resumes (no mismatch).
  3. A changed num_predict / num_ctx / temperature / word_budget_ratio is
     reported as a mismatch -> caller discards the checkpoint.
  4. A changed timeout or retry count is NOT a mismatch: those decide whether
     a call succeeds, not what a successful call says.
  5. A pre-fingerprint checkpoint reports "UNVERIFIABLE" -- warn and resume,
     never silently discard a completed run on a guess.

This never touches Ollama, the GPU, or a real pod -- it costs nothing to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation import benchmark_evaluator as be  # noqa: E402
from src.ghost_agents.approach_evaluation import approaches as A  # noqa: E402


class _FakeApproach:
    def __init__(self, model):
        self.models = [model]
        self.name = f"fake_{model}"


def _config(model="phi3:mini"):
    return be._generation_config([_FakeApproach(model)])


def test_captures_per_model_num_predict():
    cfg = _config("phi3:mini")
    assert cfg["num_predict"] == A.num_predict_for("phi3:mini"), cfg
    # Per-model, not the global default -- these two differ in MODEL_NUM_PREDICT.
    assert _config("qwen2.5:3b")["num_predict"] == A.num_predict_for("qwen2.5:3b")
    assert _config("phi3:mini")["num_predict"] != _config("qwen2.5:3b")["num_predict"]


def test_carries_content_and_provenance_fields():
    cfg = _config()
    for k in be.CONTENT_CONFIG_KEYS:
        assert k in cfg, f"missing content key {k}"
    for k in ("generate_timeout_s", "reasoning_timeout_mult", "call_retries"):
        assert k in cfg, f"missing provenance key {k}"


def test_identical_config_resumes():
    cfg = _config()
    assert be._generation_config_mismatch(dict(cfg), cfg) is None


def test_each_content_key_change_is_a_mismatch():
    cfg = _config()
    for key in be.CONTENT_CONFIG_KEYS:
        prior = dict(cfg)
        prior[key] = "SOMETHING-ELSE"
        m = be._generation_config_mismatch(prior, cfg)
        assert m is not None and m != "UNVERIFIABLE", f"{key} change not caught: {m}"
        assert key in m, f"mismatch message should name the key: {m}"


def test_num_predict_raise_is_caught():
    """The exact phi3/qwen cap-raise scenario the audit flagged."""
    cfg = _config("phi3:mini")
    prior = dict(cfg)
    prior["num_predict"] = 2048          # what the seed-42 run used
    cfg = dict(cfg, num_predict=4096)    # raised to fix length_capped
    m = be._generation_config_mismatch(prior, cfg)
    assert m is not None and m != "UNVERIFIABLE", m
    assert "2048" in m and "4096" in m, m


def test_timeout_and_retry_changes_are_not_mismatches():
    cfg = _config()
    for key in ("generate_timeout_s", "reasoning_timeout_mult", "call_retries"):
        prior = dict(cfg)
        prior[key] = 99999
        assert be._generation_config_mismatch(prior, cfg) is None, \
            f"{key} must not force a discard"


def test_pre_fingerprint_checkpoint_is_unverifiable_not_discarded():
    """A seed-42 checkpoint written before this field existed."""
    old = {"seed": 42, "max_per_benchmark": 5, "approaches": [], "benchmarks": []}
    assert be._generation_config_mismatch(old, _config()) == "UNVERIFIABLE"


def test_partial_fingerprint_still_compares_what_it_has():
    cfg = _config()
    prior = {"num_predict": cfg["num_predict"]}          # only one key present
    assert be._generation_config_mismatch(prior, cfg) is None
    prior = {"num_predict": cfg["num_predict"] + 1}
    assert be._generation_config_mismatch(prior, cfg) not in (None, "UNVERIFIABLE")


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
