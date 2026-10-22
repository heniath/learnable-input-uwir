# H2 confirmation and independent evaluation

Status: specified on 2026-10-22, before new confirmation training. No new quality
results are claimed. This is a follow-up to exposed pilot results, not a
preregistration or an untouched-dataset confirmation.

## Question and frozen comparison

Does static conditioning placement produce a repeatable benefit under the
existing 30-epoch H2 pilot recipe? The primary contrast is spatial skip
(`B1_spatial_skip`) minus input only (`B0_input`) on validation PSNR.
This contrast is motivated by the existing seed-0 test result; that motivation
is disclosed rather than treated as independent discovery.

[h2_confirmation.json](../configs/h2_confirmation.json) specifies all four H2
conditions, each trained from scratch with seeds 1, 2, 3, 4 and 5: 20 runs,
600 epochs total. Historical H2 seed 0 is shown separately and excluded from
the new five-seed estimates. All four conditions are rerun in the consolidated
implementation; no historical checkpoint is substituted for a new baseline.

Only experiment ID and seed list differ from `h2_pilot.json`. Retain the frozen
720/80/90 manifest, preprocessing, initialization, loss, optimization, effective
batch size, 100-epoch cosine horizon and 30-epoch prefix. Keep early stopping
disabled. Select maximum validation `(PSNR, SSIM)`, earliest exact tie, separately
within every run. The [protocol](PROTOCOL.md) remains authoritative.
This replication addresses training variation at the pilot budget, not convergence.

Run seeds in ascending order and conditions in config order, serially on one GPU.
No automatic sweep, early termination for favorable metrics, extra seeds chosen
after observing results, or new test90 evaluation is part of this stage.
Resume interrupted runs from their last state; record interruptions. For a
nonfinite result, identity mismatch or other scientific failure, stop and inspect;
do not drop or replace the seed. Report incomplete cells and withhold the complete
five-seed summary until they are resolved under the same protocol.

## Reporting and uncertainty

For condition c and seed s, compute the arithmetic mean validation metric
`m[c,s]` over the same 80 frozen images at that run's selected checkpoint.
The paired seed effect is `d[s] = m[c,s] - m[B0_input,s]`.
Report all five effects, their mean, sample SD (`ddof=1`), and the pointwise
two-sided interval `mean(d) ± t(0.975, 4) * SD(d) / sqrt(5)`.
The formula follows the [NIST confidence interval for a mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

The unit for this interval is a paired training seed, n=5. It estimates training
variation conditional on the fixed data and selection procedure, assuming
independent seed pairs and approximately normal seed effects. Five observations
cannot reliably validate that distributional assumption. Show every seed effect
and retain this limitation even if an interval excludes zero. These are
validation-selected scores and do not provide an unbiased generalization estimate.

Primary reporting is B1 minus B0 PSNR. Also report SSIM for that contrast and
PSNR/SSIM for B2 minus B0 and B3 minus B0 as secondary, descriptive comparisons.
All intervals are pointwise, not simultaneous or multiplicity-adjusted. Do not
use them for a family-wide significance claim. No p-value, equivalence or
noninferiority claim is planned. A narrow or positive PSNR interval alone does
not establish acceptable SSIM or practical usefulness; no powered minimum-effect
threshold is claimed.

Images, seeds and scenes are different units. Do not treat 5 × 80 observations
as 400 independent replicates, or interpret image-level win counts as seed
uncertainty. Scene independence is unresolved, so no scene-generalization
confidence interval is planned. The existing `liuwir compare` is descriptive and
does not compute intervals. A future scene-based interval requires verified
scene identifiers and a separately frozen analysis.

The [summary script](../scripts/summarize_h2_confirmation.py) verifies the prepared
config/manifest, completion identities and checkpoint hashes, then implements
the seed-level calculation above. It fails on missing runs rather than silently
reducing n. Its output includes source hashes and software versions.

## Execution and retained artifacts

These commands prepare or execute one explicit run, using user-supplied paths.
Repeat smoke and train for each of the 20 specified condition/seed pairs only
after the compute budget is approved. A smoke result is not a quality result.

```bash
liuwir prepare --config configs/h2_confirmation.json --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir smoke --config configs/h2_confirmation.json --condition B0_input --seed 1 --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir train --config configs/h2_confirmation.json --condition B0_input --seed 1 --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth
liuwir report --config configs/h2_confirmation.json
PYTHONPATH=src python scripts/summarize_h2_confirmation.py --config configs/h2_confirmation.json --destination outputs/h2_confirmation_consolidated_v1/paired_seed_summary.json
```

Retain preparation identity, source snapshot, this plan, resolved config,
environment, manifest, smoke records, best/last checkpoints, per-image validation
metrics, histories, logs, AMP/update counts, runtime and peak VRAM. Save a copy
and SHA-256 of this plan beside the prepared artifacts before the first run;
record later amendments explicitly. Use a new output ID if scientific conditions
change. The script preserves an existing summary destination.

The four historical seed-0 H2 completion records total 2,729.6 recorded epoch
seconds; five repetitions suggest about 3.8 GPU-hours before preparation, smoke
and other overhead. This is a historical budget estimate, not a measurement or
runtime reproduction of the consolidated runner. Full-state best/last checkpoints
also require more storage than the historical best-only files. Check GPU access,
current jobs and free storage before launch. Keep persistent logs in `tmux`.

## Independent evaluation: a separate stage

No independent set is claimed to have been frozen or audited yet. The proposed
first candidate is the paired **EUVP Underwater Scenes** subset. The
[maintainer's dataset page](https://irvlab.cs.umn.edu/resources/euvp-dataset)
describes paired and unpaired collections and the published subsets. Paired
reference images do not by themselves establish physical ground truth.

Before evaluating predictions:

1. Verify access terms, exact downloaded version, pair mapping and reference
   construction. Record source/archive hashes and filenames; do not infer pair
   availability from a split's name alone. Freeze the eligible paired population,
   order and all exclusion reasons without examining model scores.
2. Audit exact and near-duplicate overlap with all UIEB splits and prior local
   research data. Record scene/sequence groups and earlier exposure. Cross-dataset
   naming alone is insufficient evidence of independence. If overlap or exposure
   prevents an independent claim, document it and choose another source before
   scoring.
3. Freeze the full comparison grid: all four H2 conditions and all five seeds,
   with their UIEB-validation-selected checkpoint hashes. No external-data
   finetuning, checkpoint selection, preprocessing selection or seed selection.
4. Use the existing shorter-edge-256/center-crop-256 preprocessing and original
   float PSNR/SSIM as the initial proposed paired protocol. State its restricted
   crop coverage. Native-resolution evaluation would be a distinct condition.
5. Finalize the evaluation manifest and an external-data adapter that retains the
   original UIEB validation anchor, then verify pairing, shapes, coverage and
   metric aggregation on a smoke subset before full scoring. The current UIEB
   CLI enforces 720/80/90 membership; it is not an external-dataset evaluator.
6. Evaluate the frozen grid once, save per-image outputs and identities, and
   report the same primary/secondary contrasts across paired seeds. Conditional
   seed intervals remain separate from any scene-sampling uncertainty. Any tuning
   after this evaluation requires another untouched confirmation set.

UIEB's 60 challenge images lack satisfactory references according to the
[dataset authors](https://li-chongyi.github.io/proj_benchmark.html). They can support
a separately specified qualitative/no-reference assessment, not replacement
paired PSNR/SSIM evidence. No dataset download or external evaluation is launched
by this document. The previously cancelled LSUI study remains cancelled.

## Later H3/H4 work

H3 needs image proxies defined without reference images or outcome-based binning,
then independent confirmation of any discovered association. H4 needs the same
backbone, training/evaluation protocol and comparable parameter/compute budgets
for fixed allocation, adaptive allocation and a simple input-dependent gate.
Record residual budget differences explicitly. Include an image-independent
gate and a shuffled image–gate pairing intervention to separate extra capacity
from useful image-dependent allocation. These controls are required design work;
they are not reported experiments or implemented H4 capabilities.
