"""Summarize completed validation runs without changing any experiment artifacts."""

import argparse
import json
from pathlib import Path

from liuwir.config import read_config
from liuwir.io import digest
from liuwir.seed_summary import summarize_validation_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError("Summary exists; preserve it and choose a new destination")
    config = read_config(args.config)
    summary = summarize_validation_seeds(config, args.output_root / config["experiment_id"])
    summary["config_sha256"] = digest(args.config)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x") as handle:
        handle.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {summary['n_seed_pairs']} paired-seed validation summary to {args.destination}")


if __name__ == "__main__":
    main()
