"""Global-vector diagnostic and paired seed summaries."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import loader, read_manifest, verify_data
from .experiment import evaluate_model, seed_original, validate_checkpoint_identity
from .io import atomic_json, code_identity, digest, environment, write_csv
from .models import build_model


@torch.inference_mode()
def diagnose(config, condition, seed, checkpoint, data_root, manifest_path, destination, device):
    if config["model_family"] not in {"conditioning", "global"}:
        raise ValueError("Global diagnostics require a conditioning/global model")
    if destination.exists():
        raise FileExistsError("Choose a new diagnostic destination")
    manifest = read_manifest(manifest_path)
    verify_data(data_root, manifest)
    seed_original(seed, config["training"]["cpu_threads"])
    model = build_model(config, condition, seed).to(device).eval()
    if not model.spec.global_branch:
        raise ValueError("This condition has no global branch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(state, config, condition, seed)
    model.load_state_dict(state["model"], strict=True)
    vectors = []
    for image, _, _ in loader(data_root, manifest, "train", config, seed=seed, names=True, training=False):
        _, values = model.extract_maps(image.to(device))
        vectors.append(values[:, :, 0, 0].double().cpu())
    values = torch.cat(vectors)
    if values.shape[0] != 720 or not torch.isfinite(values).all():
        raise ValueError("Expected 720 finite training vectors")
    matrix = values.numpy()
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * max(1.0, float(np.linalg.norm(matrix)))
    energy = np.where(singular > tolerance, singular**2, 0.0)
    probabilities = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if positive.size else 0.0
    normal, normal_rows = evaluate_model(
        model, loader(data_root, manifest, "validation", config, seed=seed, names=True)
    )
    for key, tolerance in (("psnr", 0.001), ("ssim", 0.00001)):
        if abs(normal[key] - state["best"][key]) > tolerance:
            raise ValueError("Diagnostic validation anchor failed")
    rows = []
    from .metrics import metric_rows, summarize

    for image, target, names in loader(data_root, manifest, "validation", config, seed=seed, names=True):
        image = image.to(device)
        spatial, global_values = model.extract_maps(image)
        replacement = torch.tensor(mean, device=device, dtype=image.dtype)[None, :, None, None].expand_as(
            global_values
        )
        rows.extend(metric_rows(model(image, maps_override=(spatial, replacement)), target, names))
    intervention = summarize(rows)
    destination.mkdir(parents=True)
    write_csv(destination / "normal.csv", normal_rows)
    write_csv(destination / "training_mean.csv", rows)
    atomic_json(
        destination / "diagnostic.json",
        {
            "condition": condition,
            "seed": seed,
            "training_images": 720,
            "training_mean": mean.tolist(),
            "training_population_sd": matrix.std(axis=0).tolist(),
            "singular_values": singular.tolist(),
            "effective_rank": effective_rank,
            "normal": normal,
            "training_mean_intervention": intervention,
            "delta_psnr": intervention["psnr"] - normal["psnr"],
            "checkpoint_sha256": digest(checkpoint),
            "manifest_sha256": digest(manifest_path),
            "environment": environment(),
            "code": code_identity(),
            "interpretation": "Sensitivity to training-mean replacement; no physical semantics established",
        },
    )


def paired_report(candidate_path, baseline_path):
    def read(path):
        with Path(path).open() as handle:
            rows = list(csv.DictReader(handle))
        names = [r["filename"] for r in rows]
        if not rows or len(set(names)) != len(names):
            raise ValueError("Empty/duplicate metrics")
        return names, np.array([[float(r["psnr"]), float(r["ssim"])] for r in rows])

    names, candidate = read(candidate_path)
    reference_names, baseline = read(baseline_path)
    if names != reference_names or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Paired metrics must have identical ordered names and finite values")
    delta = candidate - baseline
    return {
        "images": len(names),
        "delta_psnr": float(delta[:, 0].mean()),
        "delta_ssim": float(delta[:, 1].mean()),
        "psnr_wins": int((delta[:, 0] > 0).sum()),
    }


def report_runs(output):
    rows = []
    for path in sorted((output / "runs").glob("seed_*/*/complete.json")):
        complete = json.loads(path.read_text())
        rows.append(
            {
                "seed": complete["identity"]["seed"],
                "condition": complete["identity"]["condition"],
                **complete["best"],
                "parameters": complete["parameters"],
                "conv_linear_macs": complete["conv_linear_macs"],
            }
        )
    write_csv(output / "validation_summary.csv", rows)
    print(json.dumps(rows, indent=2))
