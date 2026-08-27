#!/usr/bin/env python3
"""
Check that two seeds were run as the same experiment.

This answers a STRICTER question than the evaluator's own resume check.
`_generation_config_mismatch` decides "may I append calls to this checkpoint",
so it only looks at CONTENT_CONFIG_KEYS -- the knobs that change the generated
text. A timeout or retry difference passes that check, correctly: it does not
change what a successful call says.

But a multi-seed `mean +- std` claims the seeds differ ONLY by seed. A timeout
or retry difference changes WHICH calls succeed, and therefore each seed's
completion rate and its latency distribution (a 300s ceiling and a 600s ceiling
censor the distribution at different points). Averaging across that conflates
seed-to-seed variation with a harness change -- which is exactly what the
`+-` is supposed to measure.

So this compares every recorded field and sorts the differences:

  CONTENT      changes the generated text itself. The seeds are not
               comparable at all; do not aggregate them.
  MEASUREMENT  changes which calls survive to be scored. ASR/TSR stay
               estimates of the same quantity, but completion rates and
               latency distributions are not comparable.

Usage:
    python3 analysis/compare_seed_configs.py                 # 42 vs 43
    python3 analysis/compare_seed_configs.py --seeds 42 44

Exit code is 1 if any difference is found, so it can gate a run.
"""
import argparse
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation.benchmark_evaluator import (  # noqa: E402
    CONTENT_CONFIG_KEYS,
)

RESULTS = "report-output/ghost_agents/benchmark_results"
MEASUREMENT_KEYS = ("generate_timeout_s", "reasoning_timeout_mult", "call_retries")
ALL_KEYS = tuple(CONTENT_CONFIG_KEYS) + MEASUREMENT_KEYS + ("max_per_benchmark",)

FIX = {
    "call_retries": "EPD_CALL_RETRIES",
    "reasoning_timeout_mult": "EPD_REASONING_TIMEOUT_MULT",
    "generate_timeout_s": "EPD_GENERATE_TIMEOUT",
    "num_ctx": "EPD_NUM_CTX",
    "temperature": "EPD_TEMPERATURE",
    "word_budget_ratio": "EPD_WORD_BUDGET_RATIO",
}


def load_config(results_dir, model, seed):
    path = os.path.join(results_dir, model, f"checkpoint_seed{seed}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        doc = json.load(f)
    return doc.get("config", {}), doc.get("config_provenance")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs=2, default=["42", "43"], metavar=("A", "B"))
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()
    a, b = args.seeds

    models = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.results_dir, "*"))
        if os.path.isdir(p)
    )

    print(f"Comparing seed {a} vs seed {b}\n")
    any_diff = False
    compared = 0

    for model in models:
        cfg_a, prov_a = load_config(args.results_dir, model, a)
        cfg_b, prov_b = load_config(args.results_dir, model, b)
        if cfg_a is None or cfg_b is None:
            missing = a if cfg_a is None else b
            print(f"  {model:18s} -- no seed {missing} checkpoint yet, skipped")
            continue

        compared += 1
        content_diffs, measure_diffs, unknown = [], [], []
        for key in ALL_KEYS:
            if key not in cfg_a or key not in cfg_b:
                unknown.append(key)
                continue
            if cfg_a[key] != cfg_b[key]:
                entry = (key, cfg_a[key], cfg_b[key])
                (content_diffs if key in CONTENT_CONFIG_KEYS else measure_diffs).append(entry)

        if not content_diffs and not measure_diffs:
            note = ""
            if prov_a or prov_b:
                note = "  (one side reconstructed -- see config_provenance)"
            print(f"  {model:18s} SAME EXPERIMENT{note}")
        else:
            any_diff = True
            print(f"  {model:18s} DIFFERS")
            for key, va, vb in content_diffs:
                print(f"      [CONTENT]     {key}: seed{a}={va!r} seed{b}={vb!r}")
                print(f"                    -> generated text differs; do NOT aggregate these seeds")
            for key, va, vb in measure_diffs:
                print(f"      [MEASUREMENT] {key}: seed{a}={va!r} seed{b}={vb!r}")
                print(f"                    -> completion rates and latency are not comparable")
                if key in FIX:
                    print(f"                    -> pin with {FIX[key]}={va} to match seed {a}")
        if unknown:
            print(f"      note: not recorded on both sides: {', '.join(unknown)}")

    if not compared:
        print("\nNothing to compare yet.")
        return 0

    print("\n" + ("Differences found -- see above." if any_diff
                  else f"All {compared} model(s) match: the seeds are the same experiment."))
    return 1 if any_diff else 0


if __name__ == "__main__":
    sys.exit(main())
