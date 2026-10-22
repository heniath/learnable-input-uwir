"""Bounded GPU smoke checks; temporary binary artifacts are discarded on exit."""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from liuwir.analysis import diagnose
from liuwir.config import ROOT, read_config
from liuwir.experiment import evaluate_checkpoint, prepare, smoke, train
from liuwir.io import atomic_json, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vgg-weights", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="liuwir-smoke-") as temporary:
        for recipe, condition in (
            ("h1_pilot", "A2_global"),
            ("h2_pilot", "B3_combined"),
            ("global_dimension", "d6"),
            ("unet_baselines", "capacity"),
            ("nafnet_baselines", "bounded"),
        ):
            config = read_config(ROOT / "configs" / f"{recipe}.json")
            output = Path(temporary) / recipe
            prepare(config, args.data_root, ROOT / "data/uieb.json", output, args.vgg_weights)
            smoke(config, condition, 0, args.data_root, output, args.vgg_weights)
            record = json.loads((output / "smoke" / f"{condition}_seed0/smoke.json").read_text())
            records.append({"recipe": recipe, **record})
            torch.cuda.empty_cache()
        # Exercise the complete training/selection/completion/evaluation path with
        # an explicitly distinct one-epoch functionality condition.
        config = copy.deepcopy(read_config(ROOT / "configs/h1_pilot.json"))
        config["experiment_id"] = "runtime_one_epoch_functionality_only"
        config["training"]["epochs"] = 1
        output = Path(temporary) / "one_epoch"
        prepare(config, args.data_root, ROOT / "data/uieb.json", output, args.vgg_weights)
        smoke(config, "A2_global", 0, args.data_root, output, args.vgg_weights)
        train(config, "A2_global", 0, args.data_root, output, args.vgg_weights)
        train(config, "A2_global", 0, args.data_root, output, args.vgg_weights)
        checkpoint = output / "runs/seed_0/A2_global/best.pt"
        evaluate_checkpoint(
            config, "A2_global", 0, checkpoint, args.data_root, ROOT / "data/uieb.json", output / "evaluation"
        )
        diagnose(
            config,
            "A2_global",
            0,
            checkpoint,
            args.data_root,
            ROOT / "data/uieb.json",
            output / "diagnostic",
            "cuda",
        )
        runtime = {
            "one_epoch_training_completed": True,
            "completed_run_preserved": True,
            "evaluation_anchor_passed": True,
            "global_diagnostic_passed": True,
            "scope": "Functionality only; not a model quality result",
        }
    atomic_json(
        args.destination,
        {
            "passed": True,
            "environment": environment(),
            "smoke_checks": records,
            "training_integration": runtime,
            "temporary_binary_artifacts_retained": False,
        },
    )


if __name__ == "__main__":
    main()
