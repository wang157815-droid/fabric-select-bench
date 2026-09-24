"""Repeat the fixed H&M baseline protocol across training random seeds.

The same prepared train/validation/test split and candidate set are reused for
every seed. Each run selects its epoch using validation NDCG@10, then evaluates
the selected state on the test split. No test result influences training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@12")
MODELS = ("BPR-MF", "LightGCN")


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    result = {"seeds": seeds, "runs": len(rows), "models": {}}
    for model in MODELS:
        result["models"][model] = {}
        for split in ("validation", "test"):
            result["models"][model][split] = {}
            for metric in METRICS:
                values = [row["models"][model][split][metric] for row in rows]
                result["models"][model][split][metric] = {
                    "mean": statistics.mean(values),
                    "std_sample": statistics.stdev(values) if len(values) > 1 else None,
                    "values": values,
                }
    result["paired_test_bpr_minus_lightgcn"] = {}
    for metric in METRICS:
        differences = [
            row["models"]["BPR-MF"]["test"][metric]
            - row["models"]["LightGCN"]["test"][metric]
            for row in rows
        ]
        result["paired_test_bpr_minus_lightgcn"][metric] = {
            "mean": statistics.mean(differences),
            "std_sample": statistics.stdev(differences) if len(differences) > 1 else None,
            "values": differences,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260923, 20260924, 20260925])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    sample_dir = ROOT / "sample"
    if not (sample_dir / "interactions.npz").is_file():
        parser.error(f"Missing prepared sample: {sample_dir / 'interactions.npz'}")
    manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_sha = manifest["input_provenance"]["portable_npz_sha256"]
    digest = hashlib.sha256()
    with (sample_dir / "interactions.npz").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError("Prepared sample SHA-256 does not match the manifest")
    baseline_script = ROOT / "baselines.py"
    if not baseline_script.is_file():
        parser.error(f"Missing baseline script: {baseline_script}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        output = args.output_dir / f"seed_{seed}.json"
        row = None
        if output.exists():
            try:
                candidate = json.loads(output.read_text(encoding="utf-8"))
                if set(candidate["models"]) == set(MODELS) and all(
                    candidate["models"][model]["config"]["seed"] == seed for model in MODELS
                ):
                    row = candidate
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if row is None:
            command = [
                sys.executable,
                str(baseline_script),
                "--sample-dir", str(sample_dir),
                "--models", *MODELS,
                "--seed", str(seed),
                "--output", str(output),
            ]
            print(f"Running seed {seed}", flush=True)
            subprocess.run(command, check=True)
            row = json.loads(output.read_text(encoding="utf-8"))
        if set(row["models"]) != set(MODELS):
            raise ValueError(f"Incomplete model results for seed {seed}: {output}")
        for model in MODELS:
            if row["models"][model]["config"]["seed"] != seed:
                raise ValueError(f"Seed mismatch in {output}")
        rows.append(row)

    summary = summarize(rows, args.seeds)
    summary["protocol"] = {
        "sample": "Fixed 50,000-user H&M training-only sample",
        "epoch_selection": "Validation NDCG@10, independently per model and seed",
        "test_use": "One evaluation of each validation-selected model",
        "candidate_set": "76,252 items present in sampled users' training interactions",
        "note": "Test metrics are user macro averages over users with candidate-set labels",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()

