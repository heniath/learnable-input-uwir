"""One-time comparison against the original workstation studies.

This audit intentionally needs the original workspace. Normal repo tests and
training do not import it. It does not train or overwrite historical artifacts.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

from liuwir.config import ROOT, read_config
from liuwir.data import UIEBDataset, loader, read_manifest, verify_data
from liuwir.experiment import evaluate_model, seed_original
from liuwir.io import atomic_json, digest, environment
from liuwir.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    parent = workspace / "latent-map-diagnostics"
    sys.path[:0] = [
        str(parent / "src"),
        str(parent / "third_party/upstream/src"),
        str(parent / "third_party/nafnet"),
        str(workspace / "latent-map-conditioning/src"),
        str(workspace / "latent-map-global-study/src"),
    ]
    from conditioning.models import ConditionedUNet as OriginalConditioned
    from diagnostics.models import EnhancementModel as OriginalBaseline
    from diagnostics.protocol import PairedEpochDataset
    from global_study.models import GlobalLatentUNet as OriginalGlobal

    manifest = read_manifest(ROOT / "data/uieb.json")
    verify_data(args.data_root, manifest)
    checks = []
    seed_original(0, 4)
    # Stateless augmentation and ordering are comparability-critical.
    for split in ("train", "validation"):
        old = PairedEpochDataset(split, seed=2, epoch=7)
        new = UIEBDataset(args.data_root, manifest, split, seed=2, epoch=7)
        for index in (0, 1, 9, 31):
            a, b = old[index], new[index]
            for j in (0, 1):
                torch.testing.assert_close(a[j], b[j], rtol=0, atol=0)
            if a[2] != b[2]:
                raise AssertionError("Filename order changed")
    print("Paired preprocessing: exact", flush=True)
    example = torch.rand(1, 3, 256, 256, device=args.device)
    for recipe in ("h1_pilot", "h2_pilot", "global_dimension", "unet_baselines", "nafnet_baselines"):
        config = read_config(ROOT / "configs" / f"{recipe}.json")
        for condition in config["conditions"]:
            new = build_model(config, condition, 0).to(args.device).eval()
            if config["model_family"] == "conditioning":
                old = OriginalConditioned(condition, 0)
            elif config["model_family"] == "global":
                old = OriginalGlobal(int(condition[1:]), 0)
            else:
                old = OriginalBaseline(config["backbone"], condition, 0)
            old = old.to(args.device).eval()
            if old.state_dict().keys() != new.state_dict().keys():
                raise AssertionError("State dict keys changed")
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
            with torch.inference_mode():
                torch.testing.assert_close(old(example), new(example), rtol=0, atol=0)
            checks.append({"recipe": recipe, "condition": condition, "initial_state_and_forward_exact": True})
            print(f"Initial state/forward exact: {recipe}/{condition}", flush=True)
            del old, new
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    checkpoint_checks = []
    cases = []
    for seed in (0, 1, 2):
        label = "conditioning_h1_pilot_v2" if seed == 0 else "conditioning_h1_pilot_seeds12_v2"
        for condition in ("A0_rgb", "A1_spatial", "A2_global", "A3_both"):
            cases.append(
                (
                    "h1_pilot",
                    condition,
                    seed,
                    workspace
                    / f"latent-map-conditioning/outputs/{label}/runs/seed_{seed}/{condition}/best.pt",
                )
            )
    for condition in ("B0_input", "B1_spatial_skip", "B2_global_bottleneck", "B3_combined"):
        cases.append(
            (
                "h2_pilot",
                condition,
                0,
                workspace
                / f"latent-map-conditioning/outputs/conditioning_h2_pilot_v2/runs/seed_0/{condition}/best.pt",
            )
        )
    for dimension in (1, 6, 12):
        cases.append(
            (
                "global_dimension",
                f"d{dimension}",
                0,
                workspace
                / f"latent-map-global-study/outputs/global_branch_followup_v1/runs/seed_0/d_{dimension}/best.pt",
            )
        )
    for seed in (1, 2):
        for dimension in (6, 12):
            cases.append(
                (
                    "global_dimension",
                    f"d{dimension}",
                    seed,
                    workspace
                    / f"latent-map-global-replication/outputs/global_dimension_replication_v1/runs/seed_{seed}/d_{dimension}/best.pt",
                )
            )
    for backbone in ("unet", "nafnet"):
        for condition in ("rgb", "bounded", "unbounded", "capacity"):
            cases.append(
                (
                    f"{backbone}_baselines",
                    condition,
                    0,
                    parent / f"outputs/map_diagnostics_v1/{backbone}/{condition}/seed_0/best.pt",
                )
            )
    for recipe, condition, seed, checkpoint in cases:
        config = read_config(ROOT / "configs" / f"{recipe}.json")
        model = build_model(config, condition, seed).to(args.device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        expected = state["best"]
        metrics, _ = evaluate_model(
            model, loader(args.data_root, manifest, "validation", config, seed=seed, names=True)
        )
        drift = {k: metrics[k] - expected[k] for k in ("psnr", "ssim")}
        if abs(drift["psnr"]) > 0.001 or abs(drift["ssim"]) > 0.00001:
            raise AssertionError(f"Validation drift: {recipe}/{condition}/{seed}: {drift}")
        checkpoint_checks.append(
            {
                "recipe": recipe,
                "condition": condition,
                "seed": seed,
                "checkpoint_sha256": digest(checkpoint),
                "validation": metrics,
                "drift": drift,
            }
        )
        print(f"Validation anchor passed: {recipe}/{condition}/seed{seed}: {drift}", flush=True)
        del model, state
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    atomic_json(
        args.destination,
        {
            "passed": True,
            "environment": environment(),
            "preprocessing_exact": True,
            "initialization_checks": checks,
            "checkpoint_checks": checkpoint_checks,
        },
    )


if __name__ == "__main__":
    main()
