# One-seed validation pilot

Validation metrics come from validation-selected best checkpoints; this is preliminary, selection-biased evidence rather than a final held-out result.

The model, training recipe, splits and checkpoint-selection rule are unchanged. Seeds 1/2 and LSUI are deferred. No held-out test scores were computed.

| Dataset | Condition | Best epoch | Validation PSNR | Validation SSIM | PSNR difference vs RGB | PSNR difference vs capacity |
|---|---|---:|---:|---:|---:|---:|
| lolv2_real | rgb | 80 | 21.123 | 0.6647 | +0.000 | +0.088 |
| lolv2_real | zero_maps | 80 | 21.123 | 0.6647 | +0.000 | +0.088 |
| lolv2_real | learned_maps | 70 | 20.923 | 0.6579 | -0.200 | -0.112 |
| lolv2_real | capacity | 70 | 21.035 | 0.6589 | -0.088 | +0.000 |
| uieb | rgb | 55 | 22.701 | 0.8751 | +0.000 | -0.087 |
| uieb | zero_maps | 55 | 22.701 | 0.8751 | +0.000 | -0.087 |
| uieb | learned_maps | 79 | 22.835 | 0.8793 | +0.133 | +0.046 |
| uieb | capacity | 79 | 22.789 | 0.8805 | +0.087 | +0.000 |

Compare learned maps against both RGB and the capacity control on each dataset. A one-seed pilot cannot establish repeatability or significance. Record negative and inconsistent results as well as gains.
