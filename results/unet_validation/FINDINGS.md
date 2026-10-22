# U-Net learned-map repeatability study

New training completed: 8/8; new diagnostics completed: 8/8.

Seeds 1 and 2 are new runs. Seed 0 is read-only historical evidence from map_diagnostics_v1, with original identity and verified hashes. All scores are validation-selected on the same fixed 80 images. No test predictions are available.

| Condition | Seed | Origin | Status | Selected / total epochs | PSNR | SSIM |
|---|---:|---|---|---|---:|---:|
| rgb | 0 | historical | diagnosed | 77 / 97 | 20.92869 | 0.890754 |
| bounded | 0 | historical | diagnosed | 95 / 100 | 22.50002 | 0.913469 |
| unbounded | 0 | historical | diagnosed | 51 / 71 | 22.45254 | 0.911661 |
| capacity | 0 | historical | diagnosed | 94 / 100 | 21.10759 | 0.892095 |
| rgb | 1 | new | diagnosed | 76 / 96 | 21.21989 | 0.891299 |
| bounded | 1 | new | diagnosed | 93 / 100 | 22.75410 | 0.914007 |
| unbounded | 1 | new | diagnosed | 93 / 100 | 22.81555 | 0.915155 |
| capacity | 1 | new | diagnosed | 76 / 96 | 21.13740 | 0.891723 |
| rgb | 2 | new | diagnosed | 66 / 86 | 21.00496 | 0.888178 |
| bounded | 2 | new | diagnosed | 72 / 92 | 22.79063 | 0.915401 |
| unbounded | 2 | new | diagnosed | 72 / 92 | 22.70523 | 0.914610 |
| capacity | 2 | new | diagnosed | 61 / 81 | 20.98610 | 0.893467 |

## Seed summaries

Available diagnosed seeds only; n_seeds and seeds identify partial summaries. Sample SD measures seed variability, not uncertainty over independent scenes.

| Scope | Condition | Seeds | PSNR mean ± sample SD | SSIM mean ± sample SD |
|---|---|---|---:|---:|
| new_seeds_1_2 | rgb | [1, 2] | 21.11242 ± 0.15198 | 0.889739 ± 0.002207 |
| new_seeds_1_2 | bounded | [1, 2] | 22.77237 ± 0.02583 | 0.914704 ± 0.000986 |
| new_seeds_1_2 | unbounded | [1, 2] | 22.76039 ± 0.07801 | 0.914883 ± 0.000385 |
| new_seeds_1_2 | capacity | [1, 2] | 21.06175 ± 0.10699 | 0.892595 ± 0.001233 |
| historical_0_plus_new_1_2 | rgb | [0, 1, 2] | 21.05118 ± 0.15100 | 0.890077 ± 0.001667 |
| historical_0_plus_new_1_2 | bounded | [0, 1, 2] | 22.68159 ± 0.15830 | 0.914292 ± 0.000997 |
| historical_0_plus_new_1_2 | unbounded | [0, 1, 2] | 22.65777 ± 0.18610 | 0.913809 ± 0.001880 |
| historical_0_plus_new_1_2 | capacity | [0, 1, 2] | 21.07703 ± 0.08015 | 0.892428 ± 0.000919 |

## Predeclared decision

Decision: **advance_bounded_hypothesis**.

Positive PSNR and nonnegative SSIM against both controls in each new seed; descriptive screen, not a significance test.

- bounded, seed 1, vs rgb: ΔPSNR +1.53421 dB, ΔSSIM +0.022707; passes.
- bounded, seed 1, vs capacity: ΔPSNR +1.61670 dB, ΔSSIM +0.022283; passes.
- bounded, seed 2, vs rgb: ΔPSNR +1.78567 dB, ΔSSIM +0.027223; passes.
- bounded, seed 2, vs capacity: ΔPSNR +1.80454 dB, ΔSSIM +0.021934; passes.
- unbounded, seed 1, vs rgb: ΔPSNR +1.59566 dB, ΔSSIM +0.023856; passes.
- unbounded, seed 1, vs capacity: ΔPSNR +1.67815 dB, ΔSSIM +0.023432; passes.
- unbounded, seed 2, vs rgb: ΔPSNR +1.70027 dB, ΔSSIM +0.026432; passes.
- unbounded, seed 2, vs capacity: ΔPSNR +1.71913 dB, ΔSSIM +0.021143; passes.

## Interpretation and next evidence

Bounded is the primary comparison; unbounded is the bounds ablation. A repeatability pass supports further investigation under this recipe. It does not establish novelty, physical map semantics, or generalization. If only unbounded passes, investigate bounds in a separately frozen experiment. If neither passes, investigate variation before adding modules. No further studies launch automatically.

Paired image means, medians, trimmed means, wins and worst/best five cases are in results.json. Curves and shared observed epoch horizons are in convergence.csv and common_horizon_scores.csv. Per-image effects remain descriptive because checkpoints are validation-selected and scene independence is unverified.

Resources are in results.csv: epoch time excludes setup/checkpoint writes, peak VRAM is PyTorch allocated memory, convolution MACs exclude non-convolution operations, and latency is batch-1 float32 GPU forward at 256×256 with 10 warmups and 50 measurements. Parameter matching does not match compute.

Limits: two new seeds plus one historical seed; selection bias; uncertain scene independence; previous exposure to historical UIEB test results; shared U-Net-derived recipe; mixed-precision CUDA continuation may diverge numerically. Learned maps have no physical targets or reconstruction supervision. Published test scores are a separate protocol and are not combined with this validation study.

See ../../../research/PAPER_OUTLINE.md, PRIOR_PAPER_AUDIT.md and NOVELTY.md for the provisional paper and outstanding contribution/generalization evidence.
