# Learnable Input for Underwater Image Restoration

Research code for studying **which spatial/global latent conditions help image
restoration, where they should enter a U-Net, and whether their allocation should
adapt to the image**. This repository consolidates the completed workstation
studies into one package while retaining the historical evidence.

The maps are learned restoration features. Their names do not establish physical
transmission or background-light semantics. H1 has three-seed pilot evidence;
H2 has one-seed pilot evidence. H3/H4 remain planned work.

| Hypothesis | Question |
|---|---|
| H1 | Do spatial and global latent branches provide different benefits? |
| H2 | Does their utility depend on conditioning location? |
| H3 | Does the useful location depend on observable image characteristics? |
| H4 | Does image-adaptive allocation improve over fixed allocation? |

## Layout

```text
src/liuwir/       models, data, training, evaluation, analysis
configs/          explicit UIEB experiment recipes
data/uieb.json    frozen split, image order and content hashes (no images)
results/          original reported tables and provenance
tests/            model, data and checkpoint regression checks
docs/             protocol and consolidation notes
archive/          historical source/config/identity snapshot
third_party/      source attribution and upstream licenses
```

## Installation

The recorded research environment uses Python 3.11.16, PyTorch 2.11.0+cu128
and torchvision 0.26.0+cu128. Create a dedicated environment; the commands below
do not modify a shared ML environment.

```bash
conda env create -f environment.yml
conda activate learnable-input-uwir
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m unittest discover -s tests -v
```

For offline use, provision the pinned wheels first. No CLI command downloads data
or weights automatically. CUDA is required for training; CPU evaluation is an
explicit option and must pass the same validation anchor.

## Data and weights

Obtain UIEB through its official distribution and retain this layout:

```text
/path/to/UIEB/
  raw-890/
  reference-890/
```

The checked-in manifest fixes the historical 720/80/90 split and verifies both
image files in every pair with SHA-256. Do not regenerate it to accommodate a
different dataset. Duplicate grouping was audited historically; independence
of all scenes remains unresolved.

Training also requires torchvision's original `vgg16-397923af.pth`
([weight file](https://download.pytorch.org/models/vgg16-397923af.pth)), SHA-256
`397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.
Supply its path explicitly. Datasets and model weights are not redistributed.

## Reproduce a run

Each command acts on one explicit condition and seed. Configuration files define
the available grid; they do not launch a sweep. Run from the repository root.

```bash
liuwir prepare --config configs/h1_pilot.json --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir smoke --config configs/h1_pilot.json --condition A2_global --seed 0 --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir train --config configs/h1_pilot.json --condition A2_global --seed 0 --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir report --config configs/h1_pilot.json
```

Preparation verifies data and weights and records resolved configuration,
environment, source hashes, Git state and a source snapshot. Smoke performs two
real optimizer updates and verifies checkpoint restoration; it is not a quality
result. Training stores best/last state, optimizer, scheduler, scaler, RNG,
per-image validation metrics, epoch history, actual update/AMP-skip counts,
parameters, Conv/Linear MACs, epoch time and peak allocated VRAM.
Repeat the same training command to resume an interrupted run. Completed runs
are preserved. Use a new experiment ID after changing scientific conditions.
Long runs can be launched in `tmux`, with stdout/stderr saved to an output log.

| Config | Conditions | Budget |
|---|---|---|
| `h1_pilot.json` | RGB / spatial / global / joint | 30-epoch schedule prefix |
| `h2_pilot.json` | Input / spatial skip / global FiLM / combined | Same pilot budget |
| `global_dimension.json` | Global-only d=1/3/6/12 | Same pilot budget |
| `unet_baselines.json` | RGB / bounded / unbounded / capacity | Up to 100 epochs, patience 20 |
| `nafnet_baselines.json` | Same baseline conditions, NAFNet | Same full stopping rule |
| `h1_full.json` | H1 factorial | Full confirmation recipe; not a reported completed study |

## Evaluate and diagnose

Evaluation accepts checkpoints produced by this package and original compatible
UIEB checkpoints. It first reproduces the checkpoint's selected validation
metrics, then evaluates the requested split. Only use trusted checkpoint files.

```bash
liuwir evaluate --config configs/h1_pilot.json --condition A2_global --seed 0 --checkpoint outputs/h1_pilot_consolidated_v1/runs/seed_0/A2_global/best.pt --data-root /path/to/UIEB --split validation --destination outputs/evaluation/global_seed0
liuwir diagnose --config configs/h1_pilot.json --condition A2_global --seed 0 --checkpoint outputs/h1_pilot_consolidated_v1/runs/seed_0/A2_global/best.pt --data-root /path/to/UIEB --destination outputs/diagnostics/global_seed0
liuwir compare --candidate outputs/evaluation/global_seed0/per_image.csv --baseline /path/to/rgb/per_image.csv
```

Use `--split test` only after fixing the scientific comparison and checkpoint
selection. `--save-images` optionally writes PNGs; metrics use float predictions
before quantization. The global diagnostic replaces G with its checkpoint's
mean over 720 unaugmented training images, while preserving each spatial map.

## Evidence and reproducibility

Start with [results](results/README.md), the [scientific protocol](docs/PROTOCOL.md)
and [consolidation notes](docs/CONSOLIDATION.md). Historical scores remain separate
from runs made with this package. H1/H2 pilots do not establish convergence,
statistical significance or generalization to independent datasets. The UIEB
test partition has historical exposure and uses 256-pixel center crops.
GPU trajectories are not guaranteed to replay bit for bit.

Checkpoints are retained locally, with their recorded identities/hashes; they
are not yet hosted as downloadable releases. Until weights are distributed,
external users must retrain to reproduce inference. Older LOL-v2, initialization,
component and warm-start studies retain their exact source/config records in the
archive; their distinct protocols are not silently routed through the UIEB CLI.

## Attribution

Derived U-Net, map heads, loss and training equations come from
[`heniath/underwater-image-enhancement`](https://github.com/heniath/underwater-image-enhancement)
at `d9d28626cf32308fd6d57fd0c2f9ab39623cea60` (Apache-2.0).
The standalone NAFNet implementation derives from
[`megvii-research/NAFNet`](https://github.com/megvii-research/NAFNet)
at `2b4af71ebe098a92a75910c233a3965a3e93ede4` (MIT).
See [source mappings](third_party/sources.json), [NAFNet provenance](third_party/nafnet.json)
and [third-party licenses](third_party/licenses/).
