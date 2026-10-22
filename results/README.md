# Recorded results

Tables below are historical reports copied verbatim. Their source/hash mapping
is in `sources.json`. Conditions from different protocols are not pooled.

| Study | Evidence |
|---|---|
| H1 three-seed validation and initial global-width screen | [h1_validation](h1_validation/FINDINGS.md) |
| H1 three-seed test90 | [h1_test](h1_test/FINDINGS.md) |
| H2 one-seed validation | [h2_validation](h2_validation/FINDINGS.md) |
| H2 one-seed test90, alongside original H1 seed 0 | [h2_test](h2_test/FINDINGS.md) |
| Global d=3/6/12 three-seed validation replication | [global_replication](global_replication/FINDINGS.md) |
| U-Net bounded/unbounded/capacity validation replication | [unet_validation](unet_validation/FINDINGS.md) |
| U-Net three-seed test90 | [unet_test](unet_test/FINDINGS.md) |
| U-Net/NAFNet backbone diagnostic | [backbone](backbone/FINDINGS.md) |
| NAFNet initialization diagnostic | [initialization](initialization/FINDINGS.md) |
| NAFNet saved-checkpoint component interventions | [components](components/FINDINGS.md) |
| Original UIEB reproduction | [reproduction](reproduction/FINDINGS.md) |
| Separate fresh-map pilot | [fresh_pilot](fresh_pilot/FINDINGS.md) |
| U-Net LOL-v2 real transfer; LSUI cancelled | [lol_transfer](lol_transfer/FINAL_LOL_RESULTS.md) |
| Six-epoch warm-start architecture screen | [architecture_screen](architecture_screen/FINDINGS.md) |
| Corresponding fixed-checkpoint test overview | [architecture_test](architecture_test/FINDINGS.md) |

H1 global-only improves mean test PSNR over same-seed RGB by 1.405 dB across
three pilot seeds. Adding spatial at input gives a small inconsistent extra
PSNR effect and lower SSIM. H2's static placements have only one seed and
validation/test rankings differ. Global d=6/12 improves validation PSNR across
three seeds, but fails the prespecified joint PSNR/nondecreasing-SSIM criterion.
These observations do not establish convergence or independent generalization.
