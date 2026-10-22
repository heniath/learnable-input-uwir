# Original UIEB latent-map reproduction

Seed 0, pinned original code, historical 720/80/90 split.

| Condition | Status | Epochs | Selected epoch | Val PSNR | Val SSIM | Test PSNR | Test SSIM | Hours | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb | evaluated | 100 | 96 | 21.0217 | 0.8982 | 20.8800 | 0.8858 | 0.6490 | 1.5928 |
| latent | evaluated | 100 | 85 | 23.1206 | 0.9214 | 22.4943 | 0.9060 | 0.7288 | 1.7190 |

Runtime is the sum of recorded epoch wall times, including validation and metrics; setup and checkpoint writes are excluded. VRAM is peak PyTorch allocation, not total driver usage.

Latent minus RGB: validation PSNR +2.099 dB, SSIM +0.0231; held-out test PSNR +1.614 dB, SSIM +0.0202.

The local seed-0 test PSNR gain has the same positive direction as the notebook gain. This is a descriptive single-seed comparison.

Historical Downloads notebook: RGB 20.589 dB; latent 22.336 dB; difference +1.747 dB. Historical source/environment identity is unverified; exact agreement is not required.

Audit: 2 exact duplicate role-pairs and 2 near-duplicate candidate role-pairs cross partitions. Near candidates include exact matches and do not prove scene identity. The historical split is retained; overlap limits claims of independent held-out generalization. Full names and split roles are in `manifest.json`.

Pinned revision: `d9d28626cf32308fd6d57fd0c2f9ab39623cea60`. GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.11.0+cu128; CUDA runtime 12.8.

Models, dataset transforms, loss computations, training/validation inner loops, cosine scheduler, early-stopping class, and metrics are called from the unmodified upstream checkout. Checkpointing additionally retains scheduler, scaler, early stopping and Python/NumPy/CPU/CUDA RNG states. Test prediction uses float32 inference with upstream clipped RGB PSNR and scikit-image SSIM, with an unweighted image mean.

Seeding matches the original notebook. cuDNN deterministic=True and benchmark=False are retained; strict deterministic algorithms and TF32 changes are not imposed. Cross-version/hardware bitwise replay cannot be guaranteed. The smoke replay checks sample/augmentation hashes, model, optimizer, scheduler, scaler, early stopping, and RNG state.

- rgb epoch-boundary resume: restoration bitwise equal=True; continuation bitwise equal=False; maximum tensor difference=0.000887; validation drift={'psnr': 2.6994962187387728e-05, 'ssim': 5.699694156646729e-07}; batch-size-4 GPU smoke passed.
- latent epoch-boundary resume: restoration bitwise equal=True; continuation bitwise equal=False; maximum tensor difference=0.00144; validation drift={'psnr': -9.749599467312464e-06, 'ssim': -6.165355443954468e-07}; batch-size-4 GPU smoke passed.

The initial v1 preflight required strict continuation equality and failed. Its checkpoints, failure and diagnostics remain preserved in outputs/original_uieb_seed0_v1/. Diagnostics verified exact restoration and numerical variation in a fresh uninterrupted repetition too. The revised v2 preflight requires exact restoration, matching augmentation/order hashes and continued RNG/scheduler/scaler state, finite training and small validation drift (<0.01 dB PSNR, <0.0001 SSIM). It measures rather than promises bitwise continuation. No scientific protocol factor changed.

CUDA interpolation may have nondeterministic gradients, and float16 interpolation gradients may be inaccurate; this is a possible explanation rather than a proven attribution of the observed variation. See [PyTorch 2.11 interpolation documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.interpolate.html). Full runs are intended to execute uninterrupted in tmux; restarting cannot promise an identical numerical trajectory.

The cloned environment inherits four pip-check conflicts in the unrelated mambavision package (mamba-ssm, Pillow, requests and transformers). The same conflicts were verified in ml-base; they were not introduced or modified for this reproduction. Kornia 0.7.3 emits an upstream AMP deprecation warning under the local PyTorch version.

Two conditions and one seed do not separate a representation benefit from the latent model's added capacity or establish seed robustness. NAFNet, other datasets, additional seeds and reconstruction variants remain deferred.
