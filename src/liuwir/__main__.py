"""One CLI for preparation, training, evaluation and analysis."""

import argparse
import json
from pathlib import Path

from .analysis import diagnose, paired_report, report_runs
from .config import ROOT, read_config
from .experiment import evaluate_checkpoint, prepare, smoke, study_lock, train


def main():
    parser = argparse.ArgumentParser(
        description="Learnable spatial/global input conditioning for underwater restoration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "train", "evaluate", "diagnose", "report"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--output-root", type=Path, default=Path("outputs"))
        if name != "report":
            p.add_argument("--data-root", required=True, type=Path)
            p.add_argument("--manifest", type=Path, default=ROOT / "data/uieb.json")
        if name in ("prepare", "smoke", "train"):
            p.add_argument("--vgg-weights", required=True, type=Path)
        if name in ("smoke", "train", "evaluate", "diagnose"):
            p.add_argument("--condition", required=True)
            p.add_argument("--seed", type=int, required=True)
        if name in ("evaluate", "diagnose"):
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--destination", type=Path, required=True)
            p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if name == "evaluate":
            p.add_argument("--split", choices=("validation", "test"), default="validation")
            p.add_argument("--save-images", action="store_true")
    p = commands.add_parser(
        "compare", help="One paired per-image contrast; never treats images as training seeds"
    )
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        print(json.dumps(paired_report(args.candidate, args.baseline), indent=2))
        return
    config = read_config(args.config)
    output = args.output_root.resolve() / config["experiment_id"]
    if args.command in ("prepare", "smoke", "train", "report"):
        with study_lock(output):
            if args.command == "prepare":
                prepare(config, args.data_root, args.manifest, output, args.vgg_weights)
            elif args.command == "train":
                train(config, args.condition, args.seed, args.data_root, output, args.vgg_weights)
            elif args.command == "smoke":
                smoke(config, args.condition, args.seed, args.data_root, output, args.vgg_weights)
            else:
                report_runs(output)
    elif args.command == "evaluate":
        evaluate_checkpoint(
            config,
            args.condition,
            args.seed,
            args.checkpoint,
            args.data_root,
            args.manifest,
            args.destination,
            split=args.split,
            device=args.device,
            save_images=args.save_images,
        )
    else:
        diagnose(
            config,
            args.condition,
            args.seed,
            args.checkpoint,
            args.data_root,
            args.manifest,
            args.destination,
            args.device,
        )


if __name__ == "__main__":
    main()
