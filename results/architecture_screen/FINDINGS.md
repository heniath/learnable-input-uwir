# Fast U-Net architecture screening

Updated: 2026-10-21T07:02:10.665112+00:00

Warm-start, six full-data epochs; fixed train720/validation80. No test evaluation.

| Seed | Condition | Best epoch | Val PSNR | Val SSIM | Final PSNR | Last-three mean PSNR | Δbest vs continuation |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | control | 0 | 22.5000 | 0.91347 | 22.2340 | 22.2120 | +0.0000 |
| 0 | residual_refine | 0 | 22.5000 | 0.91347 | 22.2495 | 22.2305 | +0.0000 |
| 0 | multiscale | 0 | 22.5000 | 0.91347 | 22.2282 | 22.2061 | +0.0000 |
| 0 | skip_gate | 0 | 22.5000 | 0.91347 | 22.2278 | 22.2033 | +0.0000 |

Seed-0 promotion decision: none.

Compare candidates to the continuation control, not just the parent checkpoint. Best scores include the common epoch-0 anchor. A six-epoch ranking is a screening signal, not a converged architecture comparison or SOTA result.

Residual refinement adds a bounded correction after the existing sigmoid. Multi-scale projections are additive at encoder levels 2/3/4. Skip gates multiply skips by a map-conditioned factor in [0.5,1.5]. All extensions are zero-initialized and preserve the pretrained function before adaptation.

Source, environment, split, checkpoint and initialization records are in frozen.json and source_snapshot/. Each run retains full optimizer/scheduler/scaler/RNG in last.pt, selected inference weights in best.pt, per-image best validation metrics and complete per-epoch histories. Sample/augmentation first-batch hashes are paired across conditions. Resume cannot guarantee bitwise CUDA trajectories.

No architecture combinations, dataset changes, test predictions or full retraining are included. Historical test exposure and inherited scene-independence limits remain. Added parameter counts are reported; this screen does not rule out all capacity explanations.
