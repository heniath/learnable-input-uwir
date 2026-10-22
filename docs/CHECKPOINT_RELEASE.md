# Proposed checkpoint release

Status: local checkpoints exist; no release assets have been uploaded or published.
This initial selection covers a same-seed H1 RGB/global contrast and all four H2
placements. It is a representative pilot release, not the complete three-seed
H1 evidence. Publishing weights will not change the evidence boundaries.

## Initial assets and recorded identities

Proposed asset filenames are stable download names. All are original seed-0
`best.pt` files; do not rename a last-epoch checkpoint as a selected checkpoint.
Hashes below come from the checked-in historical result CSVs and must be checked
against the actual bytes again immediately before upload.

| Proposed asset | Config | Condition | SHA-256 |
|---|---|---|---|
| `h1_A0_rgb_seed0_best.pt` | `h1_pilot.json` | `A0_rgb` | `9ea06ec0dfd69fc00508615698fe66a01bf159f08679236d7547e7c62b5cbbb7` |
| `h1_A2_global_seed0_best.pt` | `h1_pilot.json` | `A2_global` | `b41a7791ebdf87c9606ad2a12e1fd60b1f757f89baa4a3a881a808eecbd44364` |
| `h2_B0_input_seed0_best.pt` | `h2_pilot.json` | `B0_input` | `fe41d41ee59b600ad6b5490ffacdb2cdbd783d7b5419fa2ed62862d1557d931a` |
| `h2_B1_spatial_skip_seed0_best.pt` | `h2_pilot.json` | `B1_spatial_skip` | `488505c0b748ef3de57b0a540998062f3acdfc0d4f200b7d58c1d48878793d21` |
| `h2_B2_global_bottleneck_seed0_best.pt` | `h2_pilot.json` | `B2_global_bottleneck` | `fec6b921e4fedeac115efc57a58fd1f702494327bed7c66f5a05f299bd27d345` |
| `h2_B3_combined_seed0_best.pt` | `h2_pilot.json` | `B3_combined` | `fe6fde357549423fc74e4464b922fd3edbf3964bac08d85845594dee479cf945` |

Original H1 files are under the workstation-relative
`latent-map-conditioning/outputs/conditioning_h1_pilot_v2/runs/seed_0/<condition>/best.pt`;
H2 files use `conditioning_h2_pilot_v2`. These are release-maintainer source
locations, not runtime dependencies for downloaded weights.

## Release preparation

Before publication, generate `SHA256SUMS` and an asset manifest with condition,
seed, selected epoch, original study/config identity, byte size, validation
PSNR/SSIM, dataset-manifest hash and source provenance. Retain original files;
do not duplicate complete run directories or include datasets/VGG weights.
Check checkpoint metadata for private paths or other information unsuitable for
publication. If sanitization or a weights-only export changes any bytes, issue
new asset hashes and validate the exported format; never reuse the original hash.

The migration audit already passed the original checkpoints' full UIEB validation
anchors. A release must additionally repeat the anchor using the final downloadable
bytes and a clean installation of the packaged code, with no historical-project
imports. Verify matching condition/seed and selected-validation PSNR/SSIM tolerances
of 0.001 dB / 0.00001. Record that fresh release validation separately.

Include the applicable upstream/model license and attribution, exact installation
instructions and the existing pilot/exposure caveats. No benchmark claim is created
by making inference reproducible.

## Inference after download

Download weights only from the eventual trusted release, then verify them with
its `SHA256SUMS`. The following command is ready for the proposed global asset
name; the release URL remains pending.

```bash
liuwir evaluate --config configs/h1_pilot.json --condition A2_global --seed 0 --checkpoint /path/to/downloads/h1_A2_global_seed0_best.pt --data-root /path/to/UIEB --split validation --save-images --destination outputs/released_global_seed0_validation
```

Evaluation needs the paired UIEB data and runs the original 80-image validation
anchor before saving predictions. It does not require VGG weights or retraining.
This is validation-anchored dataset inference; arbitrary-image inference is not
implemented by that command. Use a fresh destination for each evaluation.

## Repository metadata draft

Description: `Reproducible pilot studies of spatial/global conditioning placement and allocation for underwater image restoration.`

Topics: `underwater-image-enhancement`, `image-restoration`, `pytorch`, `unet`,
`conditioning`, `reproducible-research`.

These are proposed settings, not verified live GitHub metadata. Updating the
remote description/topics and publishing a weight release remain external
actions; this file does not perform them.
