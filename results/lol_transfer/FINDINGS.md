# Fresh auxiliary-map U-Net experiment: LOL-v2 real and LSUI

Hypothesis: input-conditioned auxiliary maps improve restoration beyond comparable extra capacity.

Evaluated runs: 12/24.

Smoke tests establish functionality only. Test results are reported per dataset; three seeds do not establish statistical significance.

## Completed comparisons

| Dataset | Condition | Seeds | RGB PSNR mean ± sample SD | SSIM mean ± sample SD |
|---|---|---:|---:|---:|
| lolv2_real | rgb | 3 | 17.027 ± 0.045 | 0.7770 ± 0.0010 |
| lolv2_real | zero_maps | 3 | 17.027 ± 0.045 | 0.7770 ± 0.0010 |
| lolv2_real | learned_maps | 3 | 18.763 ± 0.136 | 0.7929 ± 0.0024 |
| lolv2_real | capacity | 3 | 16.945 ± 0.091 | 0.7747 ± 0.0022 |

## Validation-selected scores (including partial training)

Available-seed summaries are provisional until all configured runs finish. These are selection-biased validation scores, separate from test results.

| Dataset | Condition | Seeds | PSNR mean | PSNR sample SD | SSIM mean | SSIM sample SD |
|---|---|---|---:|---:|---:|---:|
| lolv2_real | rgb | [0, 1, 2] | 19.1107 | 0.1524 | 0.754430 | 0.002516 |
| lolv2_real | zero_maps | [0, 1, 2] | 19.1107 | 0.1524 | 0.754430 | 0.002516 |
| lolv2_real | learned_maps | [0, 1, 2] | 20.8138 | 0.0323 | 0.770510 | 0.001717 |
| lolv2_real | capacity | [0, 1, 2] | 19.0589 | 0.0469 | 0.752491 | 0.001184 |

## Interpretation

Assess the learned-map model against both RGB and the capacity control using the paired seed differences. Preserve negative and inconsistent findings. Custom, group-aware splits and native-resolution evaluation prevent direct comparison with the historical notebook's T90 scores.

Exact and conservative near-duplicate groups are held together. Automated image similarity does not establish subject/scene independence; inspect the manifests and overlap audit before making broader claims.
