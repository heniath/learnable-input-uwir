# Local/global map diagnostic study

Seed-0 validation-only study on a fixed duplicate-grouped 720/80/90 split. Historical test90 reserved; no test predictions were computed.

| Backbone | Condition | Status | Epochs | Selected | Val PSNR | Val SSIM | Epoch hours | Peak alloc GiB | Parameters | Conv GMACs | Median ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unet | rgb | diagnosed | 97 | 77 | 20.9287 | 0.8908 | 0.5730 | 1.5928 | 31384963 | 55.6920 | 8.8692 |
| unet | bounded | diagnosed | 100 | 95 | 22.5000 | 0.9135 | 0.6027 | 1.7137 | 31418409 | 56.6750 | 8.9419 |
| unet | unbounded | diagnosed | 71 | 51 | 22.4525 | 0.9117 | 0.4240 | 1.7111 | 31418409 | 56.6750 | 9.2859 |
| unet | capacity | diagnosed | 100 | 94 | 21.1076 | 0.8921 | 0.6086 | 1.7334 | 31418291 | 57.8649 | 9.2135 |
| nafnet | rgb | diagnosed | 77 | 57 | 24.1176 | 0.9167 | 0.3932 | 1.8339 | 11454819 | 7.3934 | 4.5215 |
| nafnet | bounded | diagnosed | 44 | 24 | 23.5034 | 0.9077 | 0.2339 | 1.9511 | 11486537 | 8.2631 | 4.8518 |
| nafnet | unbounded | diagnosed | 77 | 57 | 23.7065 | 0.9127 | 0.4088 | 1.9506 | 11486537 | 8.2631 | 4.8447 |
| nafnet | capacity | diagnosed | 61 | 41 | 23.9192 | 0.9121 | 0.3269 | 1.9584 | 11486771 | 9.4779 | 5.0640 |

Epoch runtime includes training, validation losses and metrics, but excludes checkpoint writes and setup. Peak allocated VRAM is PyTorch allocation, not driver-wide memory. Conv MACs exclude pooling, activations, normalization and gates. Latency is batch-1 float32 GPU forward with 10 warmups/50 measurements at 256×256.

Screening candidate: **none**. Require >0 dB against RGB and capacity on U-Net, ≥0.2 dB against both on NAFNet, and nonnegative SSIM differences against both on both backbones. This is an engineering screen, not a significance test.

- bounded, unet_vs_rgb: ΔPSNR +1.5713 dB, ΔSSIM +0.02271; screen passes.
- bounded, unet_vs_capacity: ΔPSNR +1.3924 dB, ΔSSIM +0.02137; screen passes.
- bounded, nafnet_vs_rgb: ΔPSNR -0.6142 dB, ΔSSIM -0.00900; screen fails.
- bounded, nafnet_vs_capacity: ΔPSNR -0.4158 dB, ΔSSIM -0.00444; screen fails.
- unbounded, unet_vs_rgb: ΔPSNR +1.5239 dB, ΔSSIM +0.02091; screen passes.
- unbounded, unet_vs_capacity: ΔPSNR +1.3449 dB, ΔSSIM +0.01957; screen passes.
- unbounded, nafnet_vs_rgb: ΔPSNR -0.4111 dB, ΔSSIM -0.00399; screen fails.
- unbounded, nafnet_vs_capacity: ΔPSNR -0.2128 dB, ΔSSIM +0.00057; screen fails.

Bounded-map validation interventions (change relative to normal inference):

- unet, spatial_permutation: ΔPSNR -0.6753 dB, ΔSSIM -0.04938.
- unet, spatial_mean: ΔPSNR -0.4933 dB, ΔSSIM -0.00505.
- unet, cross_image_swap: ΔPSNR -5.0299 dB, ΔSSIM -0.10627.
- nafnet, spatial_permutation: ΔPSNR -2.3415 dB, ΔSSIM -0.14773.
- nafnet, spatial_mean: ΔPSNR -1.8608 dB, ΔSSIM -0.04309.
- nafnet, cross_image_swap: ΔPSNR -3.2900 dB, ΔSSIM -0.12269.

Next hypothesis: the reproduction gain depends on backbone, map output constraints, or optimization interactions. Inspect the bounded/unbounded contrasts, convergence histories and interventions before funding broader training or asserting a general conditioning method.

Limits: validation-selected scores are selection-biased; one seed cannot establish repeatability. Matching parameter counts does not match computation or rule out all architecture differences. Map interventions can be out of distribution and do not prove physical semantics. Detected duplicates are grouped, but scene independence is unverified. The historical UIEB test was previously inspected. This protocol changes split and initialization relative to reproduction, and changes heads/loss/preprocessing relative to the older NAFNet pilot; absolute scores should not be combined.

The same U-Net-derived recipe is used within each backbone. A weak NAFNet result under this recipe does not establish the backbone's best attainable performance. Exact checkpoint restoration is required; mixed-precision CUDA continuation may diverge numerically. No additional seeds, datasets or new modules run automatically.

See [novelty comparison](../../../research/NOVELTY.md) for primary-source method differences and unresolved novelty questions.
