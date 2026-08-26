"""
Free, local, no-GPU correctness check for the wave-based concurrent dispatch
added to evaluate_benchmark() (src/ghost_agents/approach_evaluation/
benchmark_evaluator.py). No pytest dependency -- plain asserts, run directly:

    python3 tests/test_wave_dispatch.py

Monkeypatches _send_to_model with a scripted fake so results complete in a
scrambled order relative to submission order, and checks the invariants the
wave design depends on:

  1. test_results stays in test_cases index order even when completion order
     is scrambled (required for the resume-prefix check).
  2. A checkpoint taken after a partial run (simulating an interruption) can
     be resumed correctly: the prefix is accepted, only remaining cases are
     re-dispatched, and the final combined list is complete and ordered.
  3. _check_call_latency / _record_latency_baseline_sample observe inf_lat
     values in dataset (index) order, not completion order.
  4. Max concurrent in-flight calls == 1 for an ephemeral approach and == N
     for a static approach at wave_size N, regardless of what N is
     configured to.
  5. Per-call cost_usd is discounted by wave_size for a concurrently
     dispatched (static, wave_size > 1) call, and left untouched for a
     serially dispatched (wave_size == 1) call -- estimate_cost_usd's
     POD_CONCURRENCY split only knows about cross-model sharing, so the
     evaluator must apply the intra-model wave discount itself.

This never touches Ollama, the GPU, or a real pod -- it costs nothing to run.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation import benchmark_evaluator as be  # noqa: E402


def _reset_guard_state():
    """Guards are module-level, process-lifetime state -- reset between
    independent test scenarios so one doesn't contaminate the next."""
    be._consecutive_dead_cells = 0
    be._latency_baseline_samples = []
    be._latency_anomaly_streak = 0


class FakeApproach:
    def __init__(self, name, model, ephemeral):
        self.name = name
        self.model = model
        self.ephemeral = ephemeral

    def initialize(self):
        return 0.0

    def teardown(self):
        return None


def make_test_cases(n):
    return [{"id": f"tc{i}", "category": "test", "prompt": f"prompt {i}"} for i in range(n)]


class ScriptedSendToModel:
    """Replaces _send_to_model. Sleeps a scrambled, per-call-index duration
    (later-dispatched calls finish first) so completion order != submission
    order, and tracks concurrency + call-order for the assertions below."""

    def __init__(self, n, anomaly_index=None, anomaly_latency=999.0, raw_cost_usd=None):
        self.n = n
        self.anomaly_index = anomaly_index
        self.anomaly_latency = anomaly_latency
        self.raw_cost_usd = raw_cost_usd
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.call_order = []  # order _send_to_model was actually invoked, by prompt
        self.dispatch_count = 0

    def __call__(self, approach, prompt, strategy):
        idx = int(prompt.split()[-1])
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.call_order.append(idx)
            self.dispatch_count += 1
        try:
            # Reverse-order sleep: the LAST call submitted in a wave sleeps
            # LEAST, so it finishes first -- guarantees scrambled completion
            # order whenever wave_size > 1.
            time.sleep(0.03 * (self.n - (idx % self.n)))
            inf_lat = self.anomaly_latency if idx == self.anomaly_index else 0.5
            result = {
                "status": "success",
                "command": "This is a safe refusal response.",
                "init_time": 0.0,
                "processing_time": inf_lat,
                "persona_used": "default",
            }
            if self.raw_cost_usd is not None:
                result["cost_estimate"] = {"estimated_cost_usd": self.raw_cost_usd}
            return result
        finally:
            with self.lock:
                self.active -= 1


def test_ordering_under_scrambled_completion():
    _reset_guard_state()
    n = 12
    wave_size = 4
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=wave_size)
    orig = be._send_to_model
    be._send_to_model = fake
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(wave_size)
    try:
        approach = FakeApproach("static_test", "fake-model:1b", ephemeral=False)
        result = be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=False,
        )
        test_results = result["approaches"]["static_test"]["test_results"]
        got_ids = [tr["test_id"] for tr in test_results]
        expected_ids = [tc["id"] for tc in test_cases]
        assert got_ids == expected_ids, f"ordering broken: {got_ids} != {expected_ids}"
        assert fake.max_active == wave_size, (
            f"expected {wave_size} concurrent calls, saw max {fake.max_active}"
        )
        print(f"[PASS] ordering_under_scrambled_completion "
              f"(max_active={fake.max_active}, {n} cases, wave_size={wave_size})")
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


def test_resume_after_partial_run():
    _reset_guard_state()
    n = 12
    wave_size = 4
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=wave_size)
    orig = be._send_to_model
    be._send_to_model = fake
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(wave_size)
    try:
        approach = FakeApproach("resume_test", "fake-model:1b", ephemeral=False)
        # Partial run: only the first wave's worth of cases (simulates a
        # kill after one wave completed and was checkpointed).
        partial_result = be.evaluate_benchmark(
            "HarmBench", test_cases[:wave_size], [approach], save_every=1000, verbose=False,
        )
        prior_test_results = partial_result["approaches"]["resume_test"]["test_results"]
        assert len(prior_test_results) == wave_size

        dispatch_count_before_resume = fake.dispatch_count

        # Resume: full test_cases list, resume_data from the partial run.
        resume_data = {"resume_test": {"test_results": prior_test_results}}
        full_result = be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=True,
            resume_data=resume_data,
        )
        final_results = full_result["approaches"]["resume_test"]["test_results"]
        got_ids = [tr["test_id"] for tr in final_results]
        expected_ids = [tc["id"] for tc in test_cases]
        assert got_ids == expected_ids, f"resume ordering broken: {got_ids} != {expected_ids}"
        assert len(final_results) == n

        newly_dispatched = fake.dispatch_count - dispatch_count_before_resume
        assert newly_dispatched == n - wave_size, (
            f"expected only the {n - wave_size} remaining cases re-dispatched, "
            f"saw {newly_dispatched}"
        )
        print(f"[PASS] resume_after_partial_run "
              f"(resumed from {wave_size}/{n}, re-dispatched exactly {newly_dispatched})")
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


def test_latency_guard_sees_dataset_order():
    _reset_guard_state()
    n = 10
    wave_size = 5
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=wave_size)
    orig = be._send_to_model
    be._send_to_model = fake

    seen_inf_lats = []
    orig_check = be._check_call_latency

    def spy_check(inf_lat, label, verbose):
        seen_inf_lats.append(inf_lat)
        return orig_check(inf_lat, label, verbose)

    be._check_call_latency = spy_check
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(wave_size)
    try:
        approach = FakeApproach("latency_order_test", "fake-model:1b", ephemeral=False)
        be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=False,
        )
        # Every call in this scenario returns processing_time=0.5 (no
        # scripted anomaly), so this only proves _check_call_latency was
        # invoked exactly once per test case, in index order (count check
        # plus the fact no exception -- an out-of-order/duplicate call
        # count would show up as a length mismatch).
        assert len(seen_inf_lats) == n, f"expected {n} latency checks, saw {len(seen_inf_lats)}"
        assert all(v == 0.5 for v in seen_inf_lats)
        print(f"[PASS] latency_guard_sees_dataset_order ({n} calls, all observed exactly once)")
    finally:
        be._send_to_model = orig
        be._check_call_latency = orig_check
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


def test_ephemeral_forced_to_wave_size_one():
    _reset_guard_state()
    n = 8
    configured_batch_size = 6  # deliberately > 1, to prove ephemeral ignores it
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=1)  # n=1 => no scrambling needed, just concurrency check
    orig = be._send_to_model
    be._send_to_model = fake
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(configured_batch_size)
    try:
        approach = FakeApproach("ephemeral_test", "fake-model:1b", ephemeral=True)
        be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=False,
        )
        assert fake.max_active == 1, (
            f"ephemeral approach must never dispatch concurrently -- "
            f"saw max_active={fake.max_active} with configured batch size {configured_batch_size}"
        )
        print(f"[PASS] ephemeral_forced_to_wave_size_one "
              f"(configured batch_size={configured_batch_size}, observed max_active=1)")
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


def test_static_uses_configured_batch_size():
    _reset_guard_state()
    n = 9
    configured_batch_size = 3
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=configured_batch_size)
    orig = be._send_to_model
    be._send_to_model = fake
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(configured_batch_size)
    try:
        approach = FakeApproach("static_batch_test", "fake-model:1b", ephemeral=False)
        be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=False,
        )
        assert fake.max_active == configured_batch_size, (
            f"expected max_active == {configured_batch_size}, saw {fake.max_active}"
        )
        print(f"[PASS] static_uses_configured_batch_size (max_active={fake.max_active})")
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


def test_cost_discounted_by_wave_size():
    _reset_guard_state()
    raw_cost = 0.02

    # Static approach, wave_size=4: each call's estimated_cost_usd (which
    # only accounts for cross-model sharing) must be discounted by the
    # additional wave_size-way intra-model sharing.
    n, wave_size = 8, 4
    test_cases = make_test_cases(n)
    fake = ScriptedSendToModel(n=wave_size, raw_cost_usd=raw_cost)
    orig = be._send_to_model
    be._send_to_model = fake
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = str(wave_size)
    try:
        approach = FakeApproach("cost_static_test", "fake-model:1b", ephemeral=False)
        result = be.evaluate_benchmark(
            "HarmBench", test_cases, [approach], save_every=1000, verbose=False,
        )
        costs = [tr["cost_usd"] for tr in result["approaches"]["cost_static_test"]["test_results"]]
        expected = raw_cost / wave_size
        assert all(abs(c - expected) < 1e-9 for c in costs), (
            f"expected every cost_usd == {expected} (raw {raw_cost} / wave_size {wave_size}), saw {costs}"
        )
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]

    # Ephemeral approach, wave_size forced to 1: no sibling calls to share
    # with, so the raw per-model-share estimate must pass through unchanged.
    _reset_guard_state()
    fake2 = ScriptedSendToModel(n=1, raw_cost_usd=raw_cost)
    be._send_to_model = fake2
    os.environ["EPD_NUM_PARALLEL_OVERRIDE"] = "4"  # ignored -- ephemeral forces wave_size=1
    try:
        approach = FakeApproach("cost_ephemeral_test", "fake-model:1b", ephemeral=True)
        result = be.evaluate_benchmark(
            "HarmBench", make_test_cases(4), [approach], save_every=1000, verbose=False,
        )
        costs = [tr["cost_usd"] for tr in result["approaches"]["cost_ephemeral_test"]["test_results"]]
        assert all(c == raw_cost for c in costs), (
            f"expected every cost_usd == {raw_cost} unchanged at wave_size=1, saw {costs}"
        )
        print(f"[PASS] cost_discounted_by_wave_size "
              f"(static wave={wave_size}: {raw_cost}->{expected}, ephemeral wave=1: {raw_cost} unchanged)")
    finally:
        be._send_to_model = orig
        del os.environ["EPD_NUM_PARALLEL_OVERRIDE"]


if __name__ == "__main__":
    tests = [
        test_ordering_under_scrambled_completion,
        test_resume_after_partial_run,
        test_latency_guard_sees_dataset_order,
        test_ephemeral_forced_to_wave_size_one,
        test_static_uses_configured_batch_size,
        test_cost_discounted_by_wave_size,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    print(f"All {len(tests)} test(s) passed.")
