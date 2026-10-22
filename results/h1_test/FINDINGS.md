# Global follow-up: validation-selected UIEB test90

15 unique checkpoints; 12 H1 cells across seeds 0/1/2 and 3 additional dimension models at seed 0. d=3 aliases H1 global-only seed 0. All 15 full validation anchors passed before test inference.

## Per-seed H1

| Seed | Condition | Selected epoch | Val PSNR | Val SSIM | Test PSNR | Test SSIM |
|---|---|---:|---:|---:|---:|---:|
| 0 | A0_rgb | 30 | 19.99309 | 0.870316 | 19.94769 | 0.876581 |
| 0 | A1_spatial | 30 | 20.08367 | 0.874306 | 19.83176 | 0.872481 |
| 0 | A2_global | 15 | 21.89814 | 0.903820 | 21.58350 | 0.895447 |
| 0 | A3_both | 15 | 21.94947 | 0.904963 | 21.55819 | 0.895206 |
| 1 | A0_rgb | 28 | 20.21738 | 0.874197 | 20.25195 | 0.873791 |
| 1 | A1_spatial | 30 | 20.07783 | 0.877417 | 20.08117 | 0.876567 |
| 1 | A2_global | 24 | 21.97344 | 0.908145 | 21.51169 | 0.896054 |
| 1 | A3_both | 23 | 21.79531 | 0.900796 | 21.54912 | 0.892297 |
| 2 | A0_rgb | 29 | 20.09626 | 0.878622 | 20.28123 | 0.878781 |
| 2 | A1_spatial | 22 | 19.98542 | 0.870373 | 19.74794 | 0.872182 |
| 2 | A2_global | 21 | 22.09248 | 0.909015 | 21.60068 | 0.899788 |
| 2 | A3_both | 24 | 22.04231 | 0.907479 | 21.68656 | 0.897919 |

## H1 three-seed summaries

Means +/- sample SD across training seeds, not image-level confidence intervals.

| Condition | Val PSNR | Test PSNR | Test SSIM |
|---|---:|---:|---:|
| A0_rgb | 20.10224 +/- 0.11227 | 20.16029 +/- 0.18470 | 0.876384 +/- 0.002501 |
| A1_spatial | 20.04897 +/- 0.05512 | 19.88696 +/- 0.17334 | 0.873743 +/- 0.002450 |
| A2_global | 21.98802 +/- 0.09799 | 21.56529 +/- 0.04721 | 0.897096 +/- 0.002350 |
| A3_both | 21.92903 +/- 0.12476 | 21.59795 +/- 0.07686 | 0.895141 +/- 0.002812 |

## H1 paired test effects

| Seed | Condition | Baseline | Delta PSNR | Delta SSIM | PSNR wins / 90 |
|---|---|---|---:|---:|---:|
| 0 | A1_spatial | A0_rgb | -0.11593 | -0.004100 | 37 |
| 0 | A2_global | A0_rgb | +1.63581 | +0.018866 | 69 |
| 0 | A3_both | A0_rgb | +1.61050 | +0.018625 | 73 |
| 0 | A3_both | A2_global | -0.02531 | -0.000241 | 46 |
| 1 | A1_spatial | A0_rgb | -0.17077 | +0.002776 | 40 |
| 1 | A2_global | A0_rgb | +1.25974 | +0.022263 | 63 |
| 1 | A3_both | A0_rgb | +1.29718 | +0.018506 | 67 |
| 1 | A3_both | A2_global | +0.03744 | -0.003757 | 49 |
| 2 | A1_spatial | A0_rgb | -0.53330 | -0.006599 | 34 |
| 2 | A2_global | A0_rgb | +1.31945 | +0.021007 | 60 |
| 2 | A3_both | A0_rgb | +1.40532 | +0.019138 | 63 |
| 2 | A3_both | A2_global | +0.08588 | -0.001869 | 48 |

| Condition | Baseline | Paired delta PSNR mean +/- SD | Paired delta SSIM mean +/- SD |
|---|---|---:|---:|
| A1_spatial | A0_rgb | -0.27333 +/- 0.22680 | -0.002641 +/- 0.004855 |
| A2_global | A0_rgb | +1.40500 +/- 0.20211 | +0.020712 +/- 0.001718 |
| A3_both | A0_rgb | +1.43767 +/- 0.15915 | +0.018756 +/- 0.000336 |
| A3_both | A2_global | +0.03267 +/- 0.05575 | -0.001956 +/- 0.001760 |

New-seeds-only (1/2) summaries are retained separately in the CSV/JSON summaries.

## Global-only dimension screen: seed 0

| d | Selected epoch | Val PSNR | Test PSNR | Test SSIM | Delta test PSNR vs d=3 | PSNR wins / 90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 20.77943 | 20.60163 | 0.882502 | -0.98187 | 27 |
| 3 | 15 | 21.89814 | 21.58350 | 0.895447 | +0.00000 | 0 |
| 6 | 28 | 22.11815 | 21.89807 | 0.899627 | +0.31457 | 45 |
| 12 | 15 | 22.07535 | 21.85112 | 0.899169 | +0.26762 | 51 |

## Protocol and evidence boundaries

User requested test reporting after completion of the validation-only study. All best.pt choices were fixed by validation before inference. No training, test-driven checkpoint choice, TTA or ensemble. This evaluation does not change the spatial decision or launch later experiments.

Inherited historical UIEB test90, shorter-edge resize 256 and aligned center crop 256, float32 batch4. Original per-image PSNR/SSIM on clipped float predictions; arithmetic image means. All 90 prediction PNGs per model are saved, but quantization does not affect metrics. Contact sheets use the first six frozen filenames chosen before inference.

Training seeds and images are distinct units. Across-seed sample SD is descriptive; no significance or scene-independent confidence interval is claimed. Dimension effects have one seed only and varying parameter/input widths. All runs retain a 30-epoch pilot budget. Prior test exposure and unresolved scene independence remain; these are not unseen native-resolution benchmark scores.

See protocol.json, validation_anchor.json files, execution.log and complete.json for sources/environment, identities, exact validation drift and artifact hashes.
