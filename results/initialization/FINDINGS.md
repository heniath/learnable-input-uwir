# NAFNet map-input initialization study

Five fresh seed-0 validation runs; inherited 720/80/90 split. Original early stopping and 100-epoch cosine schedule retained.

| Condition | Status | Epochs | Selected | PSNR | SSIM | AMP skips |
|---|---|---:|---:|---:|---:|---:|
| rgb | diagnosed | 77 | 57 | 24.11759 | 0.91667 | 2 |
| bounded | diagnosed | 44 | 24 | 23.50340 | 0.90766 | 0 |
| bounded_zero_init | diagnosed | 77 | 57 | 23.84150 | 0.91429 | 2 |
| unbounded | diagnosed | 77 | 57 | 23.70646 | 0.91268 | 2 |
| unbounded_zero_init | diagnosed | 87 | 67 | 24.09816 | 0.91576 | 3 |

Initialization benefit requires >=0.2 dB over matching random initialization with nonnegative SSIM. A candidate must also satisfy those gains over fresh RGB. These are descriptive screens.

- bounded_zero_init: initialization benefit=True; candidate improvement=False.
  - vs bounded: PSNR +0.33810 dB; SSIM +0.006625.
  - vs rgb: PSNR -0.27609 dB; SSIM -0.002380.
- unbounded_zero_init: initialization benefit=True; candidate improvement=False.
  - vs unbounded: PSNR +0.39171 dB; SSIM +0.003087.
  - vs rgb: PSNR -0.01943 dB; SSIM -0.000903.

| Paired comparison | Mean PSNR delta | Median | PSNR wins / 80 |
|---|---:|---:|---:|
| bounded_zero_init_vs_bounded | +0.33810 | +0.24505 | 44 |
| unbounded_zero_init_vs_unbounded | +0.39171 | +0.37810 | 46 |
| bounded_vs_rgb | -0.61419 | -0.40332 | 36 |
| bounded_zero_init_vs_rgb | -0.27609 | -0.27508 | 33 |
| unbounded_vs_rgb | -0.41113 | -0.30603 | 32 |
| unbounded_zero_init_vs_rgb | -0.01943 | -0.19380 | 36 |

All-condition common observed horizon: 44 epochs. See common_horizon_scores.csv for best-so-far scores and convergence.csv for complete curves.

Worst five regressions per contrast and historical-control differences are in results.json. Historical scores are context rather than substitutes for fresh controls.

No zero-initialized variant passes the candidate screen. Improvement over random while below RGB is partial recovery rather than a demonstrated advantage from maps.

Limits: one seed; validation-selected checkpoints; unverified scene independence; prior exposure to historical test results. No held-out test predictions. Interventions do not establish physical semantics. Stopping can interact with initialization; a stopping-policy study would be separate.

Epoch time excludes checkpoint writes/setup. VRAM is PyTorch allocation. Latency is GPU-only batch-1 float32 forward at 256x256. AMP skips are counted from durable logs, including interrupted attempts; successful-update estimates may be inaccurate after interruption. Exact state restoration does not guarantee bitwise CUDA continuation.
