# Predicted auxiliary-map experiment

Hypothesis: input-conditioned auxiliary maps improve restoration beyond comparable extra capacity.

Evaluated runs: 0/36.

Smoke tests establish functionality only. Test results are reported per dataset; three seeds do not establish statistical significance.

## Completed comparisons

| Dataset | Condition | Seeds | RGB PSNR mean ± sample SD | SSIM mean ± sample SD |
|---|---|---:|---:|---:|

No held-out results yet.

## Interpretation

Assess the learned-map model against both RGB and the capacity control using the paired seed differences. Preserve negative and inconsistent findings. Custom, group-aware splits and native-resolution evaluation prevent direct comparison with the historical notebook's T90 scores.

Exact and conservative near-duplicate groups are held together. Automated image similarity does not establish subject/scene independence; inspect the manifests and overlap audit before making broader claims.
