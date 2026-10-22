# U-Net UIEB test90 inference

Fixed validation-selected checkpoints; no training, tuning, TTA or ensembling.
Seeds 1/2 are replication runs; seed 0 is historical and is reported separately.

## Per-seed test scores

| Seed | Condition | PSNR | SSIM | Selected epoch |
|---|---|---:|---:|---:|
| 0 | rgb | 20.7229 | 0.88639 | 77 |
| 0 | bounded | 22.4815 | 0.90526 | 95 |
| 0 | unbounded | 22.1827 | 0.90304 | 51 |
| 0 | capacity | 21.0554 | 0.89008 | 94 |
| 1 | rgb | 20.9191 | 0.88624 | 76 |
| 1 | bounded | 22.5425 | 0.90695 | 93 |
| 1 | unbounded | 22.4700 | 0.90694 | 93 |
| 1 | capacity | 20.9690 | 0.88693 | 76 |
| 2 | rgb | 20.7105 | 0.88647 | 66 |
| 2 | bounded | 22.1957 | 0.90576 | 72 |
| 2 | unbounded | 22.2187 | 0.90642 | 72 |
| 2 | capacity | 20.7974 | 0.88781 | 61 |

## new_seeds_1_2

Mean ± sample SD across training seeds, not confidence intervals.

| Condition | PSNR | SSIM |
|---|---:|---:|
| rgb | 20.8148 ± 0.1475 | 0.88636 ± 0.00016 |
| bounded | 22.3691 ± 0.2452 | 0.90635 ± 0.00084 |
| unbounded | 22.3443 ± 0.1777 | 0.90668 ± 0.00037 |
| capacity | 20.8832 ± 0.1213 | 0.88737 ± 0.00063 |

## historical_0_plus_new_1_2

Mean ± sample SD across training seeds, not confidence intervals.

| Condition | PSNR | SSIM |
|---|---:|---:|
| rgb | 20.7842 ± 0.1170 | 0.88637 ± 0.00012 |
| bounded | 22.4066 ± 0.1851 | 0.90599 ± 0.00087 |
| unbounded | 22.2904 ± 0.1565 | 0.90547 ± 0.00211 |
| capacity | 20.9406 ± 0.1313 | 0.88827 ± 0.00163 |

## Paired test differences

| Seed | Condition | Control | ΔPSNR | ΔSSIM | PSNR wins / 90 |
|---|---|---|---:|---:|---:|
| 0 | rgb | capacity | -0.3325 | -0.00369 | 34 |
| 0 | bounded | rgb | +1.7586 | +0.01887 | 66 |
| 0 | bounded | capacity | +1.4260 | +0.01518 | 67 |
| 0 | unbounded | rgb | +1.4598 | +0.01665 | 65 |
| 0 | unbounded | capacity | +1.1273 | +0.01296 | 67 |
| 0 | capacity | rgb | +0.3325 | +0.00369 | 56 |
| 1 | rgb | capacity | -0.0499 | -0.00068 | 43 |
| 1 | bounded | rgb | +1.6234 | +0.02070 | 74 |
| 1 | bounded | capacity | +1.5735 | +0.02002 | 72 |
| 1 | unbounded | rgb | +1.5509 | +0.02070 | 74 |
| 1 | unbounded | capacity | +1.5009 | +0.02001 | 71 |
| 1 | capacity | rgb | +0.0499 | +0.00068 | 47 |
| 2 | rgb | capacity | -0.0869 | -0.00134 | 42 |
| 2 | bounded | rgb | +1.4852 | +0.01929 | 64 |
| 2 | bounded | capacity | +1.3983 | +0.01795 | 61 |
| 2 | unbounded | rgb | +1.5082 | +0.01995 | 63 |
| 2 | unbounded | capacity | +1.4212 | +0.01861 | 63 |
| 2 | capacity | rgb | +0.0869 | +0.00134 | 48 |

## Evaluation boundaries

Historical frozen UIEB test90, sorted indices 800:890; resize shorter edge to 256 then aligned center crop 256. RGB [0,1], float32 inference; clip predictions to [0,1] for the inherited per-image PSNR/SSIM implementations, then arithmetic mean across images.
Validation reproduction is verified separately before every test run (maximum per-image drift: PSNR ≤0.001 dB, SSIM ≤0.00001).
Historical test exposure already existed. These are not previously unseen, native-resolution benchmark scores. No checkpoint selection uses these test results. Scene independence beyond the inherited duplicate audit is unverified.
PNG examples are the first six fixed test filenames for each checkpoint, chosen before inference. Metrics use float predictions before PNG quantization. Contact sheets use the same six samples and all four conditions; they are illustrative, not a separate evaluation.
Original source/configuration/checkpoints/reports remain unchanged. See protocol.json for identities and provenance, execution.log for progress, and complete.json for output hashes and execution resources.
