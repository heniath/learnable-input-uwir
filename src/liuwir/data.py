"""Frozen UIEB membership and original stateless paired augmentation."""

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .io import digest


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    indices, pairs = manifest["indices"], manifest["pairs"]
    if [len(indices[s]) for s in ("train", "validation", "test")] != [720, 80, 90]:
        raise ValueError("Expected frozen UIEB train720/validation80/test90")
    flat = [i for s in ("train", "validation", "test") for i in indices[s]]
    if len(set(flat)) != 890 or sorted(flat) != list(range(890)) or len(pairs) != 890:
        raise ValueError("Overlapping or invalid split membership")
    if len({p["filename"] for p in pairs}) != 890:
        raise ValueError("Duplicate filenames")
    for split, members in indices.items():
        for i in members:
            if pairs[i]["index"] != i or pairs[i]["split"] != split:
                raise ValueError("Split membership disagrees with pair metadata")
    return manifest


def verify_data(root, manifest):
    root = Path(root).resolve()
    for pair in manifest["pairs"]:
        for role in ("input", "reference"):
            path = (root / pair[role]["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path escapes the dataset root")
            if digest(path) != pair[role]["sha256"]:
                raise ValueError(f"Dataset content changed: {pair['filename']}/{role}")


def transform_pair(image, target, *, training, size=256):
    """Match upstream resize, crop rounding, flip direction and RNG call order."""
    if image.size != target.size:
        raise ValueError("Input/reference dimensions differ")
    image, target = TF.resize(image, size, antialias=True), TF.resize(target, size, antialias=True)
    width, height = image.size
    if training:
        top, left = random.randint(0, max(0, height - size)), random.randint(0, max(0, width - size))
    else:
        top, left = max(0, (height - size) // 2), max(0, (width - size) // 2)
    image, target = TF.crop(image, top, left, size, size), TF.crop(target, top, left, size, size)
    if training:
        if random.random() > 0.5:
            image, target = TF.hflip(image), TF.hflip(target)
        if random.random() > 0.5:
            image, target = TF.vflip(image), TF.vflip(target)
    return TF.to_tensor(image), TF.to_tensor(target)


class UIEBDataset(Dataset):
    def __init__(self, root, manifest, split, *, seed=0, epoch=0, training=None, limit=None):
        self.root, self.manifest = Path(root), manifest
        self.indices = manifest["indices"][split][:limit]
        self.training = split == "train" if training is None else training
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pair = self.manifest["pairs"][self.indices[index]]
        with Image.open(self.root / pair["input"]["path"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / pair["reference"]["path"]) as handle:
            target = handle.convert("RGB")
        state = random.getstate()
        try:
            key = f"{self.seed}:{self.epoch}:{pair['filename']}"
            random.seed(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
            image, target = transform_pair(image, target, training=self.training)
        finally:
            random.setstate(state)
        return image, target, pair["filename"]


def collate_training(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def loader(root, manifest, split, config, *, seed=0, epoch=0, names=False, training=None, limit=None):
    dataset = UIEBDataset(root, manifest, split, seed=seed, epoch=epoch, training=training, limit=limit)
    train = dataset.training
    key = f"shuffle:{seed}:{epoch}"
    generator = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=train,
        num_workers=config["training"]["workers"],
        pin_memory=True,
        drop_last=train,
        persistent_workers=False,
        generator=generator,
        collate_fn=None if names else collate_training,
    )
