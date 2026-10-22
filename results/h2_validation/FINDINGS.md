# Validation-only conditioning study

Completed runs: 4 / 4.

| Seed | Condition | Epoch | PSNR | SSIM |
|---|---|---|---|---|
| 0 | B0_input | 28 | 21.98475 | 0.905638 |
| 0 | B2_global_bottleneck | 28 | 21.96846 | 0.908194 |
| 0 | B1_spatial_skip | 18 | 21.81950 | 0.901206 |
| 0 | B3_combined | 28 | 22.06596 | 0.906546 |

Missing runs are listed in results.json. Factorial interaction requires all four conditions for the same seed.
Conv/Linear MACs omit interpolation, normalization, pooling, activation and elementwise operations.
No test inference, model promotion or generalization claim is made by this report.
