# H1/H2 UIEB test inference

Separate user-requested evaluation of the eight completed seed-0 conditioning
pilot checkpoints. The completed validation-only training studies are read-only
dependencies. No environment changes are needed; use `uieb-latent-repro`.

```bash
/home/heniath/miniforge3/envs/uieb-latent-repro/bin/python -B scripts/evaluate.py
```

Before test inference, the runner verifies both frozen study identities,
environment, parent sources, dataset hashes, manifest and all completed run
checkpoint/metric hashes. It freezes the eight validation-selected `best.pt`
choices in `protocol.json`, checks strict checkpoint loading, then reproduces
all 80 validation-image metrics for each model (maximum per-image drift
0.001 dB PSNR and 0.00001 SSIM). Validation reproduction for all models must
pass before any model predicts test images.

Evaluation follows the inherited historical UIEB test90 partition, shorter-edge
resize 256 and paired center crop 256, float32 inference, clipped float per-image
PSNR/SSIM and arithmetic image means. No TTA, ensemble, training, selection by
test scores or native-resolution substitution is introduced. Prior test exposure
and unresolved scene independence remain limitations. All models have one seed;
no image-level confidence intervals or cross-seed uncertainty are fabricated.

`outputs/conditioning_pilots_test90_v1/` stores per-image test/validation CSVs,
all 90 prediction PNGs for each checkpoint, six prespecified examples in H1/H2
contact sheets, summary/paired CSVs, identities, validation drift, execution
log, job status and completion hashes. Float metrics precede PNG quantization.
H1 gains use `A0_rgb`; H2 gains use its own `B0_input`. Structurally equivalent
H1 `A3_both` and H2 `B0_input` remain separate trained runs.

Output directories are created exclusively. A failed or completed execution is
preserved; another execution requires a distinct `--run-id`. All checkpoint
hashes are rechecked after evaluation. No later experiments launch automatically.
