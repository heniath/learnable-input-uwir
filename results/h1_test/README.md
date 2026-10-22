# Global follow-up: fixed-checkpoint UIEB test inference

User-requested test reporting after the completed validation-only global-branch
follow-up. Training code, configurations, checkpoints and reports remain unchanged.
Use the existing `uieb-latent-repro` environment; no installation is needed.

```bash
/home/heniath/miniforge3/envs/uieb-latent-repro/bin/python -B scripts/evaluate.py
```

Evaluate the 12 H1 validation-selected checkpoints at seeds 0/1/2 and the three
new global-only dimension checkpoints d=1/6/12 at seed 0. Dimension d=3 aliases
H1 A2_global seed 0; it is not evaluated or counted twice. This yields 15 unique
models, 1,350 test predictions and 16 logical study cells.

All choices are frozen before test prediction. The runner verifies the completed
follow-up, parent sources/environment/data, split and checkpoint/validation-file
hashes. Every checkpoint must reproduce all 80 validation-image metrics before
any new test inference (max per-image drift 0.001 dB PSNR, 0.00001 SSIM).

The inherited historical UIEB test90 uses the same ordered images, shorter-edge
resize 256 and aligned center crop 256, RGB [0,1], batch4 float32 inference,
original clipped float per-image PSNR/SSIM and arithmetic image means. No training,
new selection, TTA or ensemble is performed. Prior test exposure and unresolved
scene independence remain limitations; these are not unseen native-resolution
benchmark scores. The 30-epoch budget does not establish full convergence.

The evaluation reuses verified batching/metric-coverage/PNG helpers from the
previous H1/H2 evaluation, recording their source hash. All 90 prediction PNGs
per unique model, per-image validation/test CSVs, seed-wise paired differences,
seed mean/sample SD, dimension contrasts, four fixed-example contact sheets,
protocol, live status and completion hashes are saved under
`outputs/global_followup_test90_v1/`. Metrics precede PNG quantization. Examples
are the first six frozen test filenames, fixed before inference.

H1 gains use the same-seed RGB baseline; the incremental spatial comparison uses
joint versus global-only. Across-seed SD is computed from seed means or paired
seed effects, not pooled image replicates. Dimension comparisons use seed 0
against d=3 and cannot establish cross-seed superiority. No confidence interval
based on unverified scene independence or automatic spatial-branch decision is
introduced. Any next training condition remains validation-driven.

Completed/partial output directories are never overwritten. Another execution
requires a distinct `--run-id`. GPU preflight runs before output creation; the
runner stops if another compute process is active and preserves existing desktop
services. The observed Warframe launcher is permitted only at at most 32 MiB,
GPU utilization at most 10%, and at least 4 GiB free VRAM; this excludes the
running game. This preflight exception and resource readings are recorded.
Persistent logs and failure records make failures inspectable.
