"""Paired training-seed intervals, conditional on a fixed validation set."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from .io import canonical, digest


def summarize_validation_seeds(config, output, baseline="B0_input"):
    """Require every configured completed run; never pool images with seeds."""
    output = Path(output)
    seeds, conditions = config["seeds"], config["conditions"]
    if len(seeds) < 2 or baseline not in conditions or len(conditions) < 2:
        raise ValueError("Need at least two paired seeds, a baseline and a candidate")
    frozen_path = output / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["config"] != config:
        raise ValueError("Summary config differs from the prepared experiment")
    if digest(output / "manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    frozen_identity = canonical(frozen)
    metrics, sources = {}, []
    for seed in seeds:
        for condition in conditions:
            directory = output / "runs" / f"seed_{seed}" / condition
            complete_path = directory / "complete.json"
            complete = json.loads(complete_path.read_text())
            expected = {"condition": condition, "seed": seed, "frozen_identity": frozen_identity}
            if complete["identity"] != expected:
                raise ValueError(f"Completion identity mismatch: {condition}/seed{seed}")
            if digest(directory / "best.pt") != complete["best_sha256"]:
                raise ValueError(f"Selected checkpoint changed: {condition}/seed{seed}")
            values = np.array([complete["best"][key] for key in ("psnr", "ssim")], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite validation metrics: {condition}/seed{seed}")
            metrics[condition, seed] = values
            sources.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "selected_epoch": complete["best"]["epoch"],
                    "validation": dict(zip(("psnr", "ssim"), values.tolist(), strict=True)),
                    "complete_sha256": digest(complete_path),
                    "checkpoint_sha256": complete["best_sha256"],
                }
            )
    comparisons = []
    critical = float(t.ppf(0.975, df=len(seeds) - 1))
    for condition in conditions:
        if condition == baseline:
            continue
        effects = np.stack([metrics[condition, seed] - metrics[baseline, seed] for seed in seeds])
        endpoints = {}
        for column, key in enumerate(("psnr", "ssim")):
            values = effects[:, column]
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            half_width = critical * sd / np.sqrt(len(seeds))
            endpoints[key] = {
                "paired_seed_effects": values.tolist(),
                "mean": mean,
                "sample_sd": sd,
                "ci95": [float(mean - half_width), float(mean + half_width)],
            }
        comparisons.append({"candidate": condition, "baseline": baseline, "metrics": endpoints})
    return {
        "experiment_id": config["experiment_id"],
        "split": "validation",
        "seeds": seeds,
        "n_seed_pairs": len(seeds),
        "interval_method": "pointwise two-sided Student t interval on paired seed effects; ddof=1",
        "degrees_of_freedom": len(seeds) - 1,
        "scope": "Training variation conditional on fixed validation images and checkpoint selection",
        "limitations": [
            "Independent seed pairs and approximately normal seed effects assumed; small n limits assessment",
            "No image/scene-sampling uncertainty or independent-generalization interval",
            "Validation-selected estimates; not unbiased generalization estimates",
            "Pointwise intervals; no multiplicity adjustment or family-wide significance claim",
        ],
        "frozen_identity": frozen_identity,
        "frozen_file_sha256": digest(frozen_path),
        "manifest_sha256": frozen["manifest_sha256"],
        "analysis_source_sha256": digest(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "comparisons": comparisons,
        "sources": sources,
    }
