"""Prepare, train, resume and evaluate one explicit scientific condition."""

import fcntl
import json
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import ROOT, VGG_SHA256
from .data import loader, read_manifest, verify_data
from .io import atomic_json, canonical, code_identity, digest, environment, save_checkpoint, write_csv
from .loops import EarlyStopping, train_epoch, val_loss_epoch
from .metrics import metric_rows, summarize
from .models import build_model
from .models.baselines import convolution_macs
from .rng import capture_rng_state, restore_rng_state
from .scheduler import CosineAnnealingRestartLR


@contextmanager
def study_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another command holds this experiment lock") from exc
        yield


def seed_original(seed, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_weights(path, output):
    path = Path(path).resolve()
    if digest(path) != VGG_SHA256:
        raise ValueError("Expected original VGG16 IMAGENET1K_V1 weights")
    hub = output / "torch-hub"
    cached = hub / "checkpoints/vgg16-397923af.pth"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        cached.symlink_to(path)
    if cached.resolve() != path:
        raise ValueError("VGG cache points to a different file")
    torch.hub.set_dir(str(hub))


def prepare(config, data_root, manifest_path, output, weights):
    manifest = read_manifest(manifest_path)
    verify_data(data_root, manifest)
    configure_weights(weights, output)
    code = code_identity()
    frozen = {
        "config": config,
        "manifest_sha256": digest(manifest_path),
        "weights_sha256": VGG_SHA256,
        "source_hashes": code["source_hashes"],
        "environment": environment(),
    }
    path = output / "frozen.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Prepared experiment changed; choose a new experiment ID")
        return
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "config.json", config)
    atomic_json(output / "origin.json", {"git": code, "data_root": str(Path(data_root).resolve())})
    for relative in code["source_hashes"]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    atomic_json(path, frozen)
    print(f"Prepared {config['experiment_id']}: 720 train / 80 validation / 90 test", flush=True)


def verify(config, output, weights=None):
    frozen = json.loads((output / "frozen.json").read_text())
    if frozen["config"] != config or frozen["source_hashes"] != code_identity()["source_hashes"]:
        raise ValueError("Source/config changed after preparation; use a new experiment ID")
    if frozen["environment"] != environment():
        raise ValueError("Runtime differs from the prepared environment")
    if digest(output / "manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    if weights is not None:
        configure_weights(weights, output)
    return frozen, read_manifest(output / "manifest.json")


def make_objects(config, condition, seed, weights, output, device="cuda"):
    from .losses import CompositeLoss

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training does not silently fall back to CPU")
    c = config["training"]
    configure_weights(weights, output)
    seed_original(seed, c["cpu_threads"])
    model = build_model(config, condition, seed).to(device)
    w = c["loss_weights"]
    loss = CompositeLoss(lambda_l1=w["l1"], lambda_perc=w["perceptual"], lambda_ssim=w["ssim"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = CosineAnnealingRestartLR(
        optimizer, periods=[c["scheduler_epochs"]], restart_weights=[1.0], eta_min=c["eta_min"]
    )
    scaler = torch.amp.GradScaler(device, init_scale=c["amp_initial_scale"], enabled=device == "cuda")
    early = EarlyStopping(c["patience"], c["min_delta"], "max")
    if c["reset_training_rng"]:
        seed_original(seed, c["cpu_threads"])
    return model, loss, optimizer, scheduler, scaler, early


def step_count(optimizer, model):
    step = optimizer.state.get(next(model.parameters()), {}).get("step", 0)
    return int(step.item() if isinstance(step, torch.Tensor) else step)


def pack(objects, **metadata):
    model, _, optimizer, scheduler, scaler, early = objects
    return {
        **metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "early_stopping": vars(early).copy(),
        "rng_state": capture_rng_state(),
    }


def restore(objects, state):
    for obj, name in zip(
        (objects[0], objects[2], objects[3], objects[4]),
        ("model", "optimizer", "scheduler", "scaler"),
        strict=True,
    ):
        obj.load_state_dict(state[name])
    objects[5].__dict__.update(state["early_stopping"])
    restore_rng_state(state["rng_state"])


def validate_checkpoint_identity(state, config, condition, seed):
    identity = state.get("identity", {})
    actual_condition = state.get("condition", identity.get("condition"))
    actual_seed = state.get("seed", identity.get("seed"))
    aliases = {condition}
    if config["model_family"] == "global":
        aliases.add("d_" + condition[1:])
        if condition == "d3":
            aliases.add("A2_global")
    if actual_condition not in aliases or actual_seed != seed:
        raise ValueError("Checkpoint condition/seed differs from the requested comparison")
    actual_backbone = identity.get("backbone")
    if actual_backbone is not None and actual_backbone != config["backbone"]:
        raise ValueError("Checkpoint backbone differs from the requested comparison")


def assert_state_equal(left, right):
    """Compare nested checkpoint state without assuming CPU/CUDA placement."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError("Checkpoint tensor changed after restore")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("Checkpoint keys changed")
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError("Checkpoint sequence length changed")
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    elif left != right:
        raise AssertionError("Checkpoint scalar changed")


def smoke(config, condition, seed, data_root, output, weights):
    frozen, manifest = verify(config, output, weights)
    verify_data(data_root, manifest)
    directory = output / "smoke" / f"{condition}_seed{seed}"
    if directory.exists():
        raise FileExistsError("Smoke directory exists; preserve it and choose a new experiment ID")
    directory.mkdir(parents=True)
    objects = make_objects(config, condition, seed, weights, output)
    batches = loader(data_root, manifest, "train", config, seed=seed, epoch=1, limit=8)
    loss, _ = train_epoch(objects[0], batches, objects[2], objects[1], torch.device("cuda"), objects[4])
    for name, parameter in objects[0].named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing/nonfinite gradient: {name}")
    steps = step_count(objects[2], objects[0])
    if steps != 2:
        raise ValueError(f"Smoke expected two optimizer updates, observed {steps}")
    means, rows = evaluate_model(
        objects[0], loader(data_root, manifest, "validation", config, seed=seed, names=True, limit=4)
    )
    state = pack(objects, smoke=True)
    save_checkpoint(directory / "roundtrip.pt", state)
    saved = torch.load(directory / "roundtrip.pt", map_location="cpu", weights_only=False)
    restore(objects, saved)
    assert_state_equal(saved, pack(objects, smoke=True))
    write_csv(directory / "per_image.csv", rows)
    atomic_json(
        directory / "smoke.json",
        {
            "passed": True,
            "condition": condition,
            "seed": seed,
            "frozen_identity": canonical(frozen),
            "optimizer_updates": steps,
            "loss": loss,
            "validation_subset": means,
            "checkpoint_state_exact": True,
            "scope": "Functionality only; not evidence of restoration quality",
        },
    )
    print(f"Smoke passed: {condition}, seed {seed}; two updates, exact checkpoint restoration", flush=True)


@torch.inference_mode()
def evaluate_model(model, batches, *, output=None):
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for x, y, names in batches:
        prediction = model(x.to(device)).float()
        rows.extend(metric_rows(prediction, y, names))
        if output is not None:
            for name, tensor in zip(names, prediction, strict=True):
                pixels = tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(output / name)
    if len(rows) != len(batches.dataset):
        raise ValueError("Evaluation did not cover every image")
    return summarize(rows), rows


def train(config, condition, seed, data_root, output, weights):
    frozen, manifest = verify(config, output, weights)
    verify_data(data_root, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen training recipe")
    directory = output / "runs" / f"seed_{seed}" / condition
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"frozen_identity": canonical(frozen), "condition": condition, "seed": seed}
    if (directory / "complete.json").exists():
        saved = json.loads((directory / "complete.json").read_text())
        if saved["identity"] != identity or saved["best_sha256"] != digest(directory / "best.pt"):
            raise ValueError("Completed run identity/checkpoint changed")
        print("Completed run preserved", flush=True)
        return
    smoke_path = output / "smoke" / f"{condition}_seed{seed}" / "smoke.json"
    smoke_result = json.loads(smoke_path.read_text())
    if not smoke_result.get("passed") or smoke_result["frozen_identity"] != canonical(frozen):
        raise ValueError("Pass smoke for this condition/seed before training")
    if (directory / "best.pt").exists() and not (directory / "last.pt").exists():
        raise ValueError("Best checkpoint exists without resumable last state; preserve and inspect")
    objects = make_objects(config, condition, seed, weights, output)
    model, criterion, optimizer, scheduler, scaler, early = objects
    history, best, start = [], None, 1
    last = directory / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        restore(objects, state)
        history, best, start = state["history"], state["best"], state["epoch"] + 1
    c = config["training"]
    if last.exists() and state.get("complete"):
        _finish(directory, identity, best, history)
        return
    model.eval()
    macs = convolution_macs(model)
    parameters = sum(p.numel() for p in model.parameters())
    val = loader(data_root, manifest, "validation", config, seed=seed)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, c["epochs"] + 1):
        started = time.perf_counter()
        batches = loader(data_root, manifest, "train", config, seed=seed, epoch=epoch)
        before = step_count(optimizer, model)
        train_loss, parts = train_epoch(model, batches, optimizer, criterion, torch.device("cuda"), scaler)
        validation_loss = val_loss_epoch(model, val, criterion, torch.device("cuda"), amp_enabled=True)
        metrics, rows = evaluate_model(
            model, loader(data_root, manifest, "validation", config, seed=seed, names=True)
        )
        scheduler.step()
        stopped = early(metrics["psnr"])
        updates = step_count(optimizer, model)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "parts": parts,
            "val_loss": validation_loss,
            "validation": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": updates,
            "amp_skipped_updates": len(batches) - (updates - before),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "epoch_seconds": time.perf_counter() - started,
        }
        if not np.isfinite([train_loss, validation_loss]).all():
            raise FloatingPointError("Non-finite epoch loss")
        history.append(row)
        improved = best is None or (metrics["psnr"], metrics["ssim"]) > (best["psnr"], best["ssim"])
        if improved:
            best = {"epoch": epoch, **metrics}
        complete = epoch == c["epochs"] or c["early_stopping"] and stopped
        state = pack(
            objects,
            identity=identity,
            epoch=epoch,
            config=config,
            condition=condition,
            seed=seed,
            history=history,
            best=best,
            complete=complete,
            parameters=parameters,
            conv_linear_macs=macs,
            per_image=rows,
        )
        if improved:
            save_checkpoint(directory / "best.pt", state)
            write_csv(directory / "per_image_validation.csv", rows)
        save_checkpoint(last, state)
        atomic_json(directory / "history.json", history)
        print(
            f"{condition}/seed{seed} epoch {epoch}/{c['epochs']} PSNR={metrics['psnr']:.5f} SSIM={metrics['ssim']:.6f}",
            flush=True,
        )
        if complete:
            _finish(directory, identity, best, history)
            break


def _finish(directory, identity, best, history):
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity or state["best"] != best:
        raise ValueError("Best/last transaction mismatch; inspect before resuming")
    atomic_json(
        directory / "complete.json",
        {
            "identity": identity,
            "best": best,
            "epochs": len(history),
            "optimizer_steps": history[-1]["optimizer_steps"],
            "best_sha256": digest(directory / "best.pt"),
            "parameters": state["parameters"],
            "conv_linear_macs": state["conv_linear_macs"],
        },
    )


def evaluate_checkpoint(
    config,
    condition,
    seed,
    checkpoint,
    data_root,
    manifest_path,
    destination,
    *,
    split="validation",
    device="cuda",
    save_images=False,
):
    if destination.exists():
        raise FileExistsError("Evaluation directory already exists; choose a new destination")
    manifest = read_manifest(manifest_path)
    verify_data(data_root, manifest)
    seed_original(seed, config["training"]["cpu_threads"])
    model = build_model(config, condition, seed).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(state, config, condition, seed)
    model.load_state_dict(state["model"], strict=True)
    destination.mkdir(parents=True)
    val, val_rows = evaluate_model(
        model, loader(data_root, manifest, "validation", config, seed=seed, names=True)
    )
    selected = state.get("best", {})
    if not all(k in selected for k in ("psnr", "ssim")):
        raise ValueError("Checkpoint is missing selected validation metrics")
    for key, tolerance in (("psnr", 0.001), ("ssim", 0.00001)):
        if abs(val[key] - selected[key]) > tolerance:
            raise ValueError(f"Validation anchor failed: {key}: {val[key]} vs {selected[key]}")
    write_csv(destination / "validation_anchor.csv", val_rows)
    image_dir = destination / "predictions"
    if save_images:
        image_dir.mkdir()
    if split == "validation" and not save_images:
        means, rows = val, val_rows
    else:
        means, rows = evaluate_model(
            model,
            loader(data_root, manifest, split, config, seed=seed, names=True),
            output=image_dir if save_images else None,
        )
    write_csv(destination / "per_image.csv", rows)
    atomic_json(
        destination / "result.json",
        {
            "config": config,
            "condition": condition,
            "seed": seed,
            "split": split,
            "metrics": means,
            "validation_anchor": val,
            "checkpoint_sha256": digest(checkpoint),
            "manifest_sha256": digest(manifest_path),
            "environment": environment(),
            "code": code_identity(),
            "device": device,
        },
    )
    print(json.dumps(means), flush=True)
