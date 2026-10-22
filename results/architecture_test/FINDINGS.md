# U-Net architecture screen: validation and test overview

All checkpoint choices were fixed by validation or the predefined epoch-6 endpoint before test inference.

| Seed | Model | Checkpoint | Epoch | Val PSNR | Val SSIM | Test PSNR | Test SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | control | best | 0 | 22.5000 | 0.91347 | 22.4815 | 0.90526 |
| 0 | control | last | 6 | 22.2340 | 0.91010 | 22.4844 | 0.90274 |
| 0 | multiscale | best | 0 | 22.5000 | 0.91347 | 22.4815 | 0.90526 |
| 0 | multiscale | last | 6 | 22.2282 | 0.91001 | 22.4882 | 0.90275 |
| 0 | residual_refine | best | 0 | 22.5000 | 0.91347 | 22.4815 | 0.90526 |
| 0 | residual_refine | last | 6 | 22.2495 | 0.91014 | 22.4920 | 0.90211 |
| 0 | skip_gate | best | 0 | 22.5000 | 0.91347 | 22.4815 | 0.90526 |
| 0 | skip_gate | last | 6 | 22.2278 | 0.91000 | 22.4892 | 0.90276 |
| 0 | original_rgb | best | 77 | 20.9287 | 0.89075 | 20.7229 | 0.88639 |
| 0 | original_bounded | best | 95 | 22.5000 | 0.91347 | 22.4815 | 0.90526 |
| 0 | original_capacity | best | 94 | 21.1076 | 0.89209 | 21.0554 | 0.89008 |
| 0 | nafnet_rgb | best | 57 | 24.1176 | 0.91667 | 23.2638 | 0.90034 |

## Paired test differences against continuation control

| Seed | Model | Checkpoint | Δtest PSNR | Δtest SSIM | PSNR wins / 90 |
|---|---|---|---:|---:|---:|
| 0 | control | best | +0.0000 | +0.00000 | 0 |
| 0 | control | last | +0.0000 | +0.00000 | 0 |
| 0 | multiscale | best | +0.0000 | +0.00000 | 0 |
| 0 | multiscale | last | +0.0038 | +0.00001 | 48 |
| 0 | residual_refine | best | +0.0000 | +0.00000 | 0 |
| 0 | residual_refine | last | +0.0077 | -0.00063 | 52 |
| 0 | skip_gate | best | +0.0000 | +0.00000 | 0 |
| 0 | skip_gate | last | +0.0049 | +0.00002 | 59 |

A best checkpoint can be epoch 0: the new identity-initialized extension then retains the original model and does not demonstrate an improvement. Last checkpoints measure the fixed six-epoch adaptation endpoint and are not selected using test scores.

The user requested test reporting after the validation-only screening plan was frozen. This separate evaluation does not modify training, promotion or checkpoint-selection rules. Test results are descriptive and must not be used to revise those decisions.

Inherited UIEB historical test90; shorter-edge resize 256 then center crop 256; float32; predictions clipped before per-image PSNR/SSIM. These are not native-resolution SOTA benchmark results. Seed-0 external references and any seed-1 confirmation are reported separately. Historical test exposure and scene-independence limits remain.

Per-image scores, selected-checkpoint hashes, validation reproduction checks and fixed example PNGs are retained in this directory. Training histories and promotion decision remain in the original screening output.
