"""Small artifact helpers shared by training and evaluation."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

from .config import ROOT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty metrics")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def environment():
    packages = ("torch", "torchvision", "numpy", "Pillow", "scipy", "scikit-image", "kornia")
    return {
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def code_identity():
    paths = [ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    paths.extend((ROOT / "src").rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {"commit": commit, "dirty": dirty, "source_hashes": hashes}
