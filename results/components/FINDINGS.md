# NAFNet saved-checkpoint latent-map ablation

Seed 0; 80 validation images; float32 inference; no retraining or test prediction.

| Checkpoint | Intervention | PSNR | SSIM | ΔPSNR vs intact | ΔPSNR vs RGB |
|---|---|---:|---:|---:|---:|
| rgb | normal | 24.1176 | 0.91667 | +0.0000 | +0.0000 |
| unbounded_zero_init | normal | 24.0982 | 0.91576 | +0.0000 | -0.0194 |
| unbounded_zero_init | zero_spatial | 22.8801 | 0.89426 | -1.2180 | -1.2375 |
| unbounded_zero_init | zero_global | 23.8926 | 0.91339 | -0.2056 | -0.2250 |
| unbounded_zero_init | zero_both | 23.0589 | 0.89625 | -1.0392 | -1.0587 |
| unbounded_zero_init | spatial_mean | 22.9892 | 0.89584 | -1.1089 | -1.1284 |
| unbounded_zero_init | swap_spatial | 21.8088 | 0.83701 | -2.2893 | -2.3088 |
| unbounded_zero_init | swap_global | 23.7647 | 0.91183 | -0.3334 | -0.3529 |
| bounded_zero_init | normal | 23.8415 | 0.91429 | +0.0000 | -0.2761 |
| bounded_zero_init | zero_spatial | 18.3978 | 0.83414 | -5.4437 | -5.7197 |
| bounded_zero_init | zero_global | 22.5776 | 0.90459 | -1.2639 | -1.5400 |
| bounded_zero_init | zero_both | 16.6941 | 0.77329 | -7.1474 | -7.4234 |
| bounded_zero_init | spatial_mean | 23.2080 | 0.90164 | -0.6335 | -0.9096 |
| bounded_zero_init | swap_spatial | 22.4956 | 0.87099 | -1.3459 | -1.6220 |
| bounded_zero_init | swap_global | 23.4766 | 0.91014 | -0.3649 | -0.6410 |

Removing maps uses the same jointly trained checkpoint. The zero-both condition is not the separately trained RGB baseline.

Zeroing and donor swaps can be out of distribution; a drop measures dependence, not the benefit of including that component during training. Differences between the two checkpoints also include different learned weights and selected epochs.

These results cannot establish SCA redundancy or physical meaning. One seed and validation-selected checkpoints limit generalization. A fresh factorial training experiment is needed for a causal architecture comparison.
