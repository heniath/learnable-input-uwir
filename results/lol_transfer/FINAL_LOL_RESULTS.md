# Final LOL-v2 real U-Net results

Completed 2026-10-16T10:06:17.756647+00:00 (17:06 local time, 16 October 2026).

Authorized LOL-only scope is complete: 12/12 runs trained for 100 epochs and evaluated on the same 100-image test set. LSUI was cancelled before any run started. The original experiment ID and frozen 24-run plan remain as provenance; pending LSUI rows do not mean LOL is incomplete.

## Final test scores

Mean ± sample SD across seeds 0, 1 and 2. Checkpoints selected by validation PSNR, not test performance.

| Condition | Test PSNR (dB) | Test SSIM |
|---|---:|---:|
| rgb | 17.0271 ± 0.0448 | 0.777003 ± 0.000963 |
| zero_maps | 17.0271 ± 0.0448 | 0.777003 ± 0.000963 |
| learned_maps | 18.7629 ± 0.1361 | 0.792944 ± 0.002420 |
| capacity | 16.9447 ± 0.0906 | 0.774697 ± 0.002172 |

Learned maps improve test PSNR and SSIM against RGB and capacity in every seed.

| Seed | Maps − RGB PSNR | Maps − RGB SSIM | Maps − capacity PSNR | Maps − capacity SSIM |
|---|---:|---:|---:|---:|
| 0 | +1.9420 | +0.019706 | +1.9399 | +0.022149 |
| 1 | +1.6106 | +0.014492 | +1.6637 | +0.018208 |
| 2 | +1.6550 | +0.013626 | +1.8511 | +0.014385 |

## Files

- `final_lol_results.csv`: all 12 runs, validation/test scores, checkpoint epochs and hashes.
- `final_lol_summary.csv`: four-condition means and sample SDs.
- `final_lol_paired_differences.csv`: matched-seed differences against both controls.
- `../runs/lolv2_real/<condition>/seed_<0|1|2>/best.pt`: final validation-selected inference checkpoint for each run.
- `../runs/lolv2_real/<condition>/seed_<0|1|2>/last.pt`: final training state for reproducibility/resume.
- `../runs/lolv2_real/<condition>/seed_<0|1|2>/per_image_test.csv`: per-image test metrics.

## Interpretation and provenance

This is the fresh-pilot U-Net recipe (width-16 unconstrained auxiliary heads, float32 batch 8, L1, native-resolution evaluation), a separate scientific condition from the earlier UIEB U-Net diagnostics. It does not isolate a dataset effect relative to UIEB or establish competitive performance against methods evaluated under other protocols. Three seeds are descriptive replication, not a significance test. Scene independence beyond the inherited duplicate audit is unverified.

The last capacity seed-2 run was intentionally interrupted and resumed at epoch 67 when the user cancelled LSUI. Its original source/config and saved optimizer/scheduler/RNG state were retained; no scientific setting changed. The recorded KeyboardInterrupt represents that scope change.

Verified all 12 selected checkpoint hashes, contiguous 100-epoch histories, common 100-image test identities, and per-image metric means. Frozen source/config hashes remain unchanged. No checkpoint was chosen using test scores. There is no single merged model or ensemble artifact.
