#!/usr/bin/env python3
"""
Backfill the generation-config fingerprint into the seed-42 result files.

The seed-42 run (2026-08-26 17:54 -> 2026-08-27 00:41) predates commit
984ff40, which started recording the generation settings a run was produced
under. Without that block there is no machine-checkable way to confirm a
later seed was run under the same conditions -- and a multi-seed `mean +- std`
is only meaningful if it was.

This script reconstructs that block FROM EVIDENCE ON DISK and writes it back,
alongside a `config_provenance` block that records, per field, whether the
value was measured or inferred. Nothing here is asserted from memory, and the
result is never presented as if it had been recorded at run time.

Evidence per field:

  num_predict          MEASURED  - "Generation hit the N-token cap" strings in
                                   the run's own length_capped records.
  num_ctx              MEASURED  - "n_ctx_seq = N" in each model's Ollama
                                   server log from the run.
  generate_timeout_s   MEASURED  - "Request exceeded Ns timeout" strings.
                                   Only deepseek timed out, but all four
                                   evaluators were one process tree from one
                                   launch, sharing one env.
  call_retries         MEASURED  - the retry feature did not exist at the
                                   run's HEAD (added 2026-08-27 in 5513406),
                                   confirmed by no record carrying `attempts`.
  reasoning_timeout_mult MEASURED - same commit; feature did not exist, so the
                                   effective multiplier was 1.0.
  temperature          INFERRED  - code default at the run's HEAD (8f418e4).
                                   No env record survives; the launch command
                                   is not in any retained transcript.
  word_budget_ratio    INFERRED  - same basis.

The two INFERRED fields are near-certain (both are the documented defaults and
the paper specifies greedy decoding) but they are not proven, and the
provenance block says so. A wrong value there would at worst cause a future
re-run of seed 42 to be conservatively discarded, never to be silently mixed.

Usage:
    python3 analysis/backfill_seed42_config.py --dry-run
    python3 analysis/backfill_seed42_config.py
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ghost_agents.approach_evaluation.benchmark_evaluator import (  # noqa: E402
    CONTENT_CONFIG_KEYS,
    _atomic_write_json,
)

RESULTS = "report-output/ghost_agents/benchmark_results"
RUN_LOGS = "report-output/ghost_agents/run_logs"
RUN_HEAD = "8f418e4"          # HEAD when the run started, 2026-08-26 17:54
RECONSTRUCTED_ON = "2026-08-27"

CAP_RE = re.compile(r"hit the (\d+)-token cap")
TIMEOUT_RE = re.compile(r"exceeded (\d+)s timeout")
NCTX_RE = re.compile(r"n_ctx_seq\s*=\s*(\d+)")


def iter_records(doc):
    for cell in doc.get("benchmark_results", {}).values():
        for approach in cell.get("approaches", {}).values():
            for rec in approach.get("test_results", []):
                yield rec


def evidence_from_records(doc):
    """num_predict, timeout and retry-presence, read out of the run's own records."""
    caps, timeouts, retried = set(), set(), False
    for rec in iter_records(doc):
        text = f"{rec.get('call_error') or ''} {rec.get('detail') or ''}"
        for m in CAP_RE.finditer(text):
            caps.add(int(m.group(1)))
        for m in TIMEOUT_RE.finditer(text):
            timeouts.add(int(m.group(1)))
        if rec.get("attempts"):
            retried = True
    return caps, timeouts, retried


def evidence_num_ctx(model_key):
    """n_ctx_seq from this model's Ollama server log for the run."""
    path = os.path.join(RUN_LOGS, f"ollama_{model_key}.log")
    if not os.path.exists(path):
        return set()
    found = set()
    with open(path, errors="replace") as f:
        for line in f:
            if "n_ctx_seq" in line:
                for m in NCTX_RE.finditer(line):
                    found.add(int(m.group(1)))
    return found


def build_config(model_key, doc):
    """Returns (config_fields, provenance) or (None, reason)."""
    caps, timeouts, retried = evidence_from_records(doc)
    ctxs = evidence_num_ctx(model_key)

    if len(caps) > 1:
        return None, f"conflicting num_predict evidence: {sorted(caps)}"
    if len(ctxs) > 1:
        return None, f"conflicting num_ctx evidence: {sorted(ctxs)}"
    if len(timeouts) > 1:
        return None, f"conflicting timeout evidence: {sorted(timeouts)}"
    if retried:
        return None, "records carry `attempts` -- this run postdates the retry feature"

    cfg = {}
    prov = {}

    if caps:
        cfg["num_predict"] = caps.pop()
        prov["num_predict"] = "measured: length_capped error strings in this run's records"
    else:
        # No call hit the cap, so the run left no direct trace of its value.
        return None, "no length_capped record -- num_predict not recoverable from this file"

    if ctxs:
        cfg["num_ctx"] = ctxs.pop()
        prov["num_ctx"] = f"measured: n_ctx_seq in {RUN_LOGS}/ollama_{model_key}.log"
    else:
        return None, f"no n_ctx_seq found in ollama_{model_key}.log"

    cfg["temperature"] = 0.0
    prov["temperature"] = f"INFERRED: code default at the run's HEAD ({RUN_HEAD}); no env record survives"
    cfg["word_budget_ratio"] = 0.7
    prov["word_budget_ratio"] = f"INFERRED: code default at the run's HEAD ({RUN_HEAD}); no env record survives"

    if timeouts:
        cfg["generate_timeout_s"] = timeouts.pop()
        prov["generate_timeout_s"] = "measured: timeout error strings in this model's own records"
    else:
        # This model never timed out, so it left no direct trace. All four
        # evaluators were one process tree from one launch sharing one env,
        # and deepseek_r1_1_5b's records pin that env's value at 300s.
        cfg["generate_timeout_s"] = 300
        prov["generate_timeout_s"] = (
            "measured indirectly: no call of this model timed out; the value is pinned by "
            f"deepseek_r1_1_5b's records from the same launch and env, and the {RUN_HEAD} "
            "code default agrees"
        )

    cfg["reasoning_timeout_mult"] = 1.0
    prov["reasoning_timeout_mult"] = f"measured: feature did not exist at {RUN_HEAD} (added 2026-08-27 in 5513406)"
    cfg["call_retries"] = 0
    prov["call_retries"] = f"measured: feature did not exist at {RUN_HEAD}; no record carries `attempts`"

    return cfg, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()

    models = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.results_dir, "*"))
        if os.path.isdir(p)
    )

    written = skipped = 0
    for model_key in models:
        checkpoint = os.path.join(args.results_dir, model_key, "checkpoint_seed42.json")
        if not os.path.exists(checkpoint):
            continue
        with open(checkpoint) as f:
            doc = json.load(f)

        if any(k in doc.get("config", {}) for k in CONTENT_CONFIG_KEYS):
            print(f"  {model_key:18s} already fingerprinted -- left alone")
            skipped += 1
            continue

        cfg, prov = build_config(model_key, doc)
        if cfg is None:
            print(f"  {model_key:18s} SKIPPED -- {prov}")
            skipped += 1
            continue

        print(f"  {model_key:18s} " + "  ".join(f"{k}={v}" for k, v in cfg.items()))

        if args.dry_run:
            continue

        # Same block into the checkpoint and every eval file for this seed.
        targets = [checkpoint] + sorted(
            glob.glob(os.path.join(args.results_dir, model_key, "benchmark_eval_*seed42*.json"))
        ) + sorted(
            glob.glob(os.path.join(args.results_dir, f"benchmark_eval_*seed42*{model_key}*_combined.json"))
        )
        for path in targets:
            with open(path) as f:
                target = json.load(f)
            if any(k in target.get("config", {}) for k in CONTENT_CONFIG_KEYS):
                continue
            target.setdefault("config", {}).update(cfg)
            target["config_provenance"] = {
                "reconstructed": True,
                "reconstructed_on": RECONSTRUCTED_ON,
                "reason": (
                    "This run predates commit 984ff40, which started recording the "
                    "generation config at run time. The values here were recovered from "
                    "evidence on disk, not recorded during the run. Fields marked "
                    "INFERRED are the code defaults at the run's HEAD and are not proven."
                ),
                "run_head_commit": RUN_HEAD,
                "fields": prov,
            }
            _atomic_write_json(path, target, indent=2, default=str)
            written += 1

    print(f"\n{written} file(s) written, {skipped} model(s) skipped"
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
