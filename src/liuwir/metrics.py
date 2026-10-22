"""Historical RGB PSNR/SSIM: clipped float predictions and per-image means."""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def metric_rows(prediction, target, names):
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite prediction/target")
    pred = prediction.float().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
    truth = target.float().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
    if pred.shape != truth.shape or len(names) != len(pred):
        raise ValueError("Prediction/target/name coverage mismatch")
    return [
        {
            "filename": name,
            "psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            "ssim": float(structural_similarity(b, a, data_range=1.0, channel_axis=2)),
        }
        for a, b, name in zip(pred, truth, names, strict=True)
    ]


def summarize(rows):
    if not rows or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate image metrics")
    values = np.array([[r["psnr"], r["ssim"]] for r in rows])
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite image metrics")
    return {k: float(v) for k, v in zip(("psnr", "ssim"), values.mean(axis=0), strict=True)}
