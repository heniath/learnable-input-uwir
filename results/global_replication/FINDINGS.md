# Global dimension replication — validation only

Validation cells 9/9; diagnostics 9/9; missing entries 0.

Five reference cells are reused in place; exactly four new training runs were authorized.

| Seed | d | Reused | Selected epoch | PSNR | SSIM |
|---:|---:|---|---:|---:|---:|
| 0 | 3 | True | 15 | 21.89814 | 0.903820 |
| 0 | 6 | True | 28 | 22.11815 | 0.908156 |
| 0 | 12 | True | 15 | 22.07535 | 0.907285 |
| 1 | 3 | True | 24 | 21.97344 | 0.908145 |
| 1 | 6 | False | 30 | 22.28948 | 0.907358 |
| 1 | 12 | False | 30 | 22.23044 | 0.906665 |
| 2 | 3 | True | 21 | 22.09248 | 0.909015 |
| 2 | 6 | False | 24 | 22.15578 | 0.905584 |
| 2 | 12 | False | 21 | 22.12888 | 0.908526 |

| Seed | d | Baseline d | Delta PSNR | Delta SSIM | PSNR wins / 80 |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 3 | +0.22000 | +0.004337 | 37 |
| 0 | 12 | 3 | +0.17721 | +0.003466 | 42 |
| 0 | 12 | 6 | -0.04279 | -0.000871 | 41 |
| 1 | 6 | 3 | +0.31604 | -0.000787 | 38 |
| 1 | 12 | 3 | +0.25700 | -0.001480 | 42 |
| 1 | 12 | 6 | -0.05904 | -0.000693 | 39 |
| 2 | 6 | 3 | +0.06329 | -0.003431 | 45 |
| 2 | 12 | 3 | +0.03640 | -0.000489 | 44 |
| 2 | 12 | 6 | -0.02689 | +0.002942 | 38 |

## Across-seed effects

Mean and sample SD use paired training-seed effects, not pooled image replicates. Available seed lists remain explicit.

| d | Baseline d | Seeds | Delta PSNR mean | Sample SD |
|---:|---:|---|---:|---:|
| 6 | 3 | [0, 1, 2] | +0.19978 | 0.12757947909003323 |
| 12 | 3 | [0, 1, 2] | +0.15687 | 0.11169653453243611 |
| 12 | 6 | [0, 1, 2] | -0.04291 | 0.016072791039632536 |

## Prespecified descriptive repeatability

Both new seeds must improve validation PSNR and have nonnegative validation SSIM gain versus same-seed d=3. Missing evidence is pending. This is not a statistical significance test.

- d=6: not_confirmed_in_pilot
- d=12: not_confirmed_in_pilot

## Global-vector diagnostics

| Seed | d | Effective rank | Normal val PSNR | Training-mean G val PSNR | Mean minus normal |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 2.7616 | 21.89814 | 17.60449 | -4.29366 |
| 0 | 6 | 3.9029 | 22.11815 | 17.49938 | -4.61877 |
| 0 | 12 | 3.8125 | 22.07535 | 17.42679 | -4.64856 |
| 1 | 3 | 2.6530 | 21.97344 | 17.55399 | -4.41945 |
| 1 | 6 | 3.9468 | 22.28948 | 18.01978 | -4.26969 |
| 1 | 12 | 4.1975 | 22.23044 | 18.04600 | -4.18444 |
| 2 | 3 | 2.6717 | 22.09248 | 17.79036 | -4.30212 |
| 2 | 6 | 3.6376 | 22.15578 | 17.65974 | -4.49604 |
| 2 | 12 | 3.9938 | 22.12888 | 17.81277 | -4.31612 |

Each mean uses only the 720 training images in eval mode with augmentation disabled. Normal validation must reproduce the selected checkpoint before interpretation. Full vector statistics, saturation and per-image intervention results are retained per diagnostic.

## Limits and next decision

All runs use the inherited 30-epoch pilot, first 30 epochs of the 100-epoch schedule, no early stopping and validation PSNR/SSIM selection. Parameter/MAC differences are recorded; capacity is not matched. CUDA replay need not be bitwise identical.

Inherited scene independence remains unresolved and historical test exposure remains. This study performs no new test inference, test-based choice, automatic 100-epoch training or spatial removal. Reports support later evidence review; they do not automatically promote an architecture.
