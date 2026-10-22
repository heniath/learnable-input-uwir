# Consolidation

The original workstation project directories, datasets and checkpoints were
read-only inputs. They remain in place. This repository has a new Git history;
it does not claim that historical experiments ran at a new commit.

## Active implementation

One `liuwir` package replaces cross-project `sys.path` bootstrapping and implicit
parent-output dependencies. Recipes, dataset root, VGG weights, checkpoint and
output paths are explicit. The active CLI supports UIEB baseline comparisons,
H1/H2, global dimensions, selected-checkpoint evaluation and global diagnostics.
It contains no live dependency on the historical archive or adjacent folders.

`third_party/sources.json` records original hashes and extraction changes.
Model attributes retain checkpoint-compatible keys. Only imports were adapted
in the original model wrappers; equations, bounds, layer order and initialization
were retained. New modules implement portable path handling and a shared runner.

## Historical evidence

`results/sources.json` maps each verbatim report/table to its original relative
path and SHA-256. Historical Markdown links and statements describe their original
context; use `results/README.md` to navigate the consolidated evidence.

`archive/historical-source.tar.gz` retains the source/config/identity files of
the original studies in their original relative hierarchy. `archive/index.json`
records all member hashes and the archive hash. Dataset/checkpoint binaries,
prediction images, notebooks, caches and temporary logs are excluded. Original
notebooks and binary artifacts remain in the workstation directories.

The archive includes superseded runners, environment identities and manifests
for the initial reproduction, fresh pilot, backbone/initialization diagnostics,
component interventions, U-Net replication, LOL transfer, architecture screening,
conditioning, global dimensions and their test evaluators. These old scripts
retain machine-specific paths and historical identity checks; the archive is a
provenance snapshot, not a portable second CLI. Exact forensic replay requires
the original artifacts, environment and pinned upstream checkout. The active
UIEB CLI is the portable entry point; no claim is made that it reproduces the
different LOL/fresh-pilot protocols.

## Validation

`python -m unittest discover -s tests -v` checks paired initialization, identity
modulation, gradient activation, skip/encoder separation, metric aggregation,
manifest overlap guards and optimizer/scheduler/RNG continuation.

`scripts/validate_migration.py` is a one-time migration audit against the original
workspace. It checks exact preprocessing, initialized state/forward equivalence,
and complete 80-image validation anchors from historical checkpoints. It never
updates those checkpoints. The resulting audit and smoke report are recorded
in `results/validation/`; they are functionality/equivalence evidence, not new
model-quality results.

Large model checkpoints remain local. Their hashes are retained in historical
identity files and validation audits. A public model-weight release remains
separate work; until then, external readers can retrain using the frozen recipes.
