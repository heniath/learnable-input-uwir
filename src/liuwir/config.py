"""Explicit experiment recipes; machine paths are supplied through the CLI."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def backbone_defaults():
    return {
        "nafnet": {
            "width": 32,
            "middle_blocks": 4,
            "encoder_blocks": [1, 1, 1, 4],
            "decoder_blocks": [1, 1, 1, 1],
        },
        "capacity_added_parameter_tolerance": 0.02,
    }


def read_config(path):
    from .models.conditioning import SPECS

    config = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["experiment_id"]):
        raise ValueError("experiment_id must be a single safe directory name")
    family = config["model_family"]
    allowed = {
        "baseline": {"rgb", "bounded", "unbounded", "capacity"},
        "conditioning": set(SPECS),
        "global": {"d1", "d3", "d6", "d12"},
    }
    if family not in allowed:
        raise ValueError(f"Unknown model family: {family}")
    conditions, seeds = config["conditions"], config["seeds"]
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - allowed[family]:
        raise ValueError("Invalid conditions")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if config["backbone"] not in {"unet", "nafnet"} or family != "baseline" and config["backbone"] != "unet":
        raise ValueError("Conditioning/global families use the fixed U-Net")
    training = config["training"]
    if not 1 <= training["epochs"] <= training["scheduler_epochs"] == 100:
        raise ValueError("Use a prefix of the historical 100-epoch schedule")
    if training["batch_size"] != 4 or config["crop_size"] != 256:
        raise ValueError("Historical UIEB protocol requires batch 4 and crop 256")
    if type(training["early_stopping"]) is not bool or type(training["reset_training_rng"]) is not bool:
        raise ValueError("Stopping and RNG-reset flags must be boolean")
    if training["lr"] != 0.0002 or training["weight_decay"] != 0.00001 or training["eta_min"] != 0.000001:
        raise ValueError("Changed optimization requires a separately implemented protocol")
    if training["loss_weights"] != {"l1": 1.0, "perceptual": 1.0, "ssim": 0.0}:
        raise ValueError("Historical loss is L1 + VGG, weight 1:1")
    return config
