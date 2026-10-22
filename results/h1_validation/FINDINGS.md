# Global branch follow-up — validation only

H1 completed: 12/12. Dimension cells: 4/4. Diagnostics: 9/9. Missing entries: 0.

## H1 paired-seed confirmation

| Seed | Condition | Selected epoch | PSNR | SSIM |
|---|---|---:|---:|---:|
| 0 | A0_rgb | 30 | 19.99309 | 0.870316 |
| 0 | A1_spatial | 30 | 20.08367 | 0.874306 |
| 0 | A2_global | 15 | 21.89814 | 0.903820 |
| 0 | A3_both | 15 | 21.94947 | 0.904963 |
| 1 | A0_rgb | 28 | 20.21738 | 0.874197 |
| 1 | A1_spatial | 30 | 20.07783 | 0.877417 |
| 1 | A2_global | 24 | 21.97344 | 0.908145 |
| 1 | A3_both | 23 | 21.79531 | 0.900796 |
| 2 | A0_rgb | 29 | 20.09626 | 0.878622 |
| 2 | A1_spatial | 22 | 19.98542 | 0.870373 |
| 2 | A2_global | 21 | 22.09248 | 0.909015 |
| 2 | A3_both | 24 | 22.04231 | 0.907479 |

| Seed | Condition | Baseline | Delta PSNR | Delta SSIM |
|---|---|---|---:|---:|
| 0 | A1_spatial | A0_rgb | +0.09058 | +0.003990 |
| 0 | A2_global | A0_rgb | +1.90506 | +0.033503 |
| 0 | A3_both | A0_rgb | +1.95638 | +0.034647 |
| 0 | A3_both | A2_global | +0.05132 | +0.001144 |
| 1 | A1_spatial | A0_rgb | -0.13955 | +0.003220 |
| 1 | A2_global | A0_rgb | +1.75606 | +0.033948 |
| 1 | A3_both | A0_rgb | +1.57793 | +0.026599 |
| 1 | A3_both | A2_global | -0.17813 | -0.007349 |
| 2 | A1_spatial | A0_rgb | -0.11084 | -0.008250 |
| 2 | A2_global | A0_rgb | +1.99623 | +0.030392 |
| 2 | A3_both | A0_rgb | +1.94605 | +0.028857 |
| 2 | A3_both | A2_global | -0.05018 | -0.001535 |

Means/sample SD across paired training-seed contrasts are saved in h1_contrast_summary.csv. Image and seed replicates are not pooled; no scene-independent confidence interval is claimed.

## Global-only dimension screen

| Seed | d | Selected epoch | PSNR | SSIM | Delta PSNR vs d=3 | Reused |
|---|---:|---:|---:|---:|---:|---|
| 0 | 1 | 28 | 20.77943 | 0.891454 | -1.11872 | False |
| 0 | 3 | 15 | 21.89814 | 0.903820 | +0.00000 | True |
| 0 | 6 | 28 | 22.11815 | 0.908156 | +0.22000 | False |
| 0 | 12 | 15 | 22.07535 | 0.907285 | +0.17721 | False |

The dimension screen uses seed 0 only; d=3 reuses H1 global-only. It does not establish cross-seed dimension effects. Input/head widths and parameter/MAC counts vary; see dimension_validation.csv.

## Global-vector variation and training-mean intervention

| Scope | Seed | Condition | Training G SD | Normal val PSNR | Training-mean G val PSNR | Mean minus normal |
|---|---:|---|---|---:|---:|---:|
| H1 | 0 | A2_global | 0.110135, 0.138845, 0.142783 | 21.89814 | 17.60449 | -4.29366 |
| H1 | 0 | A3_both | 0.145290, 0.108478, 0.128944 | 21.94947 | 17.84563 | -4.10384 |
| H1 | 1 | A2_global | 0.113697, 0.147359, 0.109066 | 21.97344 | 17.55399 | -4.41945 |
| H1 | 1 | A3_both | 0.110876, 0.148973, 0.108642 | 21.79531 | 17.94979 | -3.84551 |
| H1 | 2 | A2_global | 0.129467, 0.138384, 0.113768 | 22.09248 | 17.79036 | -4.30212 |
| H1 | 2 | A3_both | 0.141914, 0.108489, 0.136633 | 22.04231 | 17.34800 | -4.69431 |
| dimension | 0 | d_1 | 0.170029 | 20.77943 | 18.15419 | -2.62524 |
| dimension | 0 | d_6 | 0.043581, 0.083676, 0.103243, 0.090246, 0.123630, 0.089046 | 22.11815 | 17.49938 | -4.61877 |
| dimension | 0 | d_12 | 0.068035, 0.071568, 0.059471, 0.041512, 0.099869, 0.098972, 0.088097, 0.062378, 0.033136, 0.059664, 0.040074, 0.086453 | 22.07535 | 17.42679 | -4.64856 |

Each mean is computed from that checkpoint's 720 training images only, without augmentation, in eval mode. Normal validation reproduces the saved selected checkpoint before interpretation. Spatial maps are held fixed. Negative mean-minus-normal PSNR indicates loss from this intervention. Continuous SD/ranges, saturation and covariance/SVD/effective rank are saved per diagnostic; near-constant thresholds are descriptive, not proof of collapse.

## Decision boundary

Spatial removal/modification/retention remains deferred pending review of seed-wise evidence. No test predictions, test-based selection, automatic promotion, environment changes or historical artifact replacement. Pilot horizon, historical test exposure and unresolved scene independence remain limitations.
