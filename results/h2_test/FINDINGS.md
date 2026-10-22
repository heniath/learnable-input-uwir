# H1/H2 pilot: validation-selected UIEB test90 inference

Eight fixed best.pt checkpoints; seed 0, 30 training epochs per condition.
All validation reproduction checks passed before any test inference.

## H1

Test differences use A0_rgb as the within-study baseline.

| Condition | Selected epoch | Val PSNR | Val SSIM | Test PSNR | Test SSIM | Delta test PSNR |
|---|---:|---:|---:|---:|---:|---:|
| A0_rgb | 30 | 19.99309 | 0.870316 | 19.94769 | 0.876581 | +0.00000 |
| A1_spatial | 30 | 20.08367 | 0.874306 | 19.83176 | 0.872481 | -0.11593 |
| A2_global | 15 | 21.89814 | 0.903820 | 21.58350 | 0.895447 | +1.63581 |
| A3_both | 15 | 21.94947 | 0.904963 | 21.55819 | 0.895206 | +1.61050 |

## H2

Test differences use B0_input as the within-study baseline.

| Condition | Selected epoch | Val PSNR | Val SSIM | Test PSNR | Test SSIM | Delta test PSNR |
|---|---:|---:|---:|---:|---:|---:|
| B0_input | 28 | 21.98475 | 0.905638 | 21.51267 | 0.892396 | +0.00000 |
| B2_global_bottleneck | 28 | 21.96846 | 0.908194 | 21.60412 | 0.894009 | +0.09145 |
| B1_spatial_skip | 18 | 21.81950 | 0.901206 | 21.94703 | 0.897339 | +0.43435 |
| B3_combined | 28 | 22.06596 | 0.906546 | 21.42263 | 0.889905 | -0.09004 |

## Paired test differences

| Phase | Condition | Baseline | Delta PSNR | Delta SSIM | PSNR wins / 90 |
|---|---|---|---:|---:|---:|
| H1 | A1_spatial | A0_rgb | -0.11593 | -0.004100 | 37 |
| H1 | A2_global | A0_rgb | +1.63581 | +0.018866 | 69 |
| H1 | A3_both | A0_rgb | +1.61050 | +0.018625 | 73 |
| H2 | B2_global_bottleneck | B0_input | +0.09145 | +0.001613 | 49 |
| H2 | B1_spatial_skip | B0_input | +0.43435 | +0.004943 | 52 |
| H2 | B3_combined | B0_input | -0.09004 | -0.002491 | 50 |

## Protocol and interpretation limits

Historical UIEB test90 (sorted indices 800:890), inherited shorter-edge resize 256 and aligned center crop 256; RGB [0,1], batch size 4, float32 inference, no AMP/TTA/ensemble. Predictions are clipped for the original per-image PSNR/SSIM implementations and metrics are arithmetic means over the same 90 images.

Checkpoint identities were frozen before evaluation. Maximum per-image validation reproduction tolerance: 0.001 dB PSNR and 0.00001 SSIM. No test result changes model selection or training.

All 90 predictions per checkpoint are saved as PNG; metrics use float predictions before quantization. H1/H2 contact sheets show the first six frozen test filenames, chosen before inference.

One seed per condition: no cross-seed uncertainty or statistical significance is established. H1 and H2 use their own baselines; H1 A3_both and H2 B0_input remain separate training realizations. Active parameter/compute budgets differ. Pilot training does not establish convergence. Historical test exposure and unresolved scene independence remain; these are not previously unseen native-resolution benchmark results.

Training source, configs, checkpoints and reports remain unchanged. See protocol.json, validation_anchor.json files, execution.log and complete.json for provenance, verification, runtime and artifact hashes.
