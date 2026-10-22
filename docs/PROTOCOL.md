# Scientific protocol

## Frozen UIEB comparison

- Data: exact manifest in `data/uieb.json`, 720 train / 80 validation / 90 test.
  Filenames, order, membership and original file hashes are retained.
- Preprocessing: RGB [0,1]; aspect-preserving shorter-edge resize to 256,
  aligned 256 crop. Training uses random crops and paired horizontal/vertical
  flips; validation/test use center crops.
- Pair augmentation RNG: SHA-256(`seed:epoch:filename`), first 16 hex digits.
  Epoch shuffle uses an independent generator keyed by `shuffle:seed:epoch`.
  Training batch 4, drop last, two workers; evaluation batch 4, no drop.
- Optimizer: AdamW, lr 2e-4, weight decay 1e-5. Original cosine scheduler,
  horizon 100 epochs, eta_min 1e-6. AMP starts at scale 1024; gradient norm
  clipped at 1.0. Actual updates count skipped AMP steps correctly.
- Loss: L1 + VGG16 perceptual loss, weights 1:1. VGG uses the original first
  4 and following 5 feature children, ImageNet normalization and clipped inputs.
  Historical SSIM-loss computation remains present with zero objective weight.
- Selection: maximum validation `(PSNR, SSIM)` tuple, earliest exact tie.
  Pilot: 30 epochs, no early stopping, first 30 epochs of the 100-epoch schedule.
  Full recipe: maximum 100 epochs, early stopping on PSNR, patience 20,
  min_delta 1e-4. Checkpoint selection and early stopping have separate rules.
- Metrics: original scikit-image RGB PSNR and SSIM defaults (`data_range=1`,
  `channel_axis=2` for SSIM), fp32 inference, clipped float predictions,
  arithmetic means over individual images. No TTA, ensemble or native-resolution
  evaluation is silently introduced.

## Model and initialization

The U-Net has widths 64/128/256/512/1024, bilinear decoder upsampling with
`align_corners=True`, BatchNorm and a direct sigmoid RGB head. Learned spatial
maps are three-channel in [0.1,1]; the three-dimensional global vector is in
[0,1]. The unbounded baseline removes both head bounds.

Shared backbone construction uses `seed`; map heads and expanded input weights
use isolated `seed+10000`; capacity control uses `seed+20000`. The capacity
control matches added parameter count to within 2%, not overall compute.
H1 retains a nine-channel input with zero slots for disabled branches and
removes the disabled extractors. Historical RGB baselines use three channels;
they are separate training conditions from H1 A0 even when their initial
functions coincide. Registered, structurally active and dormant parameters are
distinguished by `ConditionedUNet.parameter_counts()`.

H2 retains both maps at input. Spatial modulation affects decoder skip copies
at E1/E2; the encoder continues with unmodified features. The projection is
random and its scalar strength is initialized to zero. Global FiLM uses G
before broadcast, an MLP 3→64→2C with a zero final layer, after bottleneck and
before upsampling. Spatial resize uses `align_corners=False`. Swapped-placement,
constant-condition and no-concat controls exist as model specs but are not
reported as completed experiments.

The global-width screen starts from the same global-only construction, retains
overlapping d=3 coordinates/input columns, and initializes new coordinates with
isolated `seed+50000`. The first layer has `6+d` columns, including three dormant
spatial slots. Width changes also change capacity and compute.

H1/H2/global reset the training RNG after model/loss construction. The earlier
baseline studies did not reset it at that point. This distinction is explicit
in each recipe and must not be erased when comparing trajectories.

## Recorded evidence and interpretation

Checkpoint restoration includes model/BN buffers, optimizer, scheduler,
scaler, early stopping and Python/NumPy/CPU/CUDA RNG. The dataset generator
is a stateless function of epoch and seed. CUDA determinism settings match
the historical recipe; bitwise GPU continuation is not promised.

The consolidated runner omits auxiliary CIEDE2000/UCIQE/UIQM calculations during
validation, full input-tensor batch hashes and workstation queue-management
scaffolding. Those computations do not change the objective or the PSNR/SSIM
selection rule. Consequently, new wall-clock measurements must not be treated
as historical runtime reproductions. Historical records retain those fields.

Paired seed effects are summarized across training seeds; images are not pooled
with seeds as equivalent replicates. Three seeds do not establish significance.
Test90 is historically exposed; scene independence is unresolved. Image-level
interventions establish sensitivity to that intervention, not physical semantics.
H3 requires independently defined image proxies and independent confirmation;
H4 requires controls for fixed allocation, extra capacity and image–gate pairing.

The [H2 confirmation plan](H2_CONFIRMATION.md) specifies a separate five-fresh-seed
replication at the unchanged pilot budget, paired training-seed intervals and a
subsequent independent-evaluation design. It is planned work, not additional
completed evidence. H2 confirmation does not authorize new test90 model selection
or reuse of challenge images as paired-reference evaluation data.

New output IDs explicitly identify consolidated-code runs. Model and preprocessing
equivalence are audited against original sources/checkpoints; that audit is
not a claim that all complete historical training trajectories were rerun.
