# Research-design follow-up validation — 2026-10-22

Scope: README evidence presentation, planned H2 seed replication, paired-seed
reporting and checkpoint-release preparation. No training, new quality analysis,
external dataset scoring, dependency changes, commits, pushes or uploads occurred.

## Checks executed

Using the existing `uieb-latent-repro` environment (Python 3.11.16,
NumPy 2.4.6, SciPy 1.17.1), from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m ruff check src tests scripts
python scripts/verify_records.py
git diff --check
PYTHONPATH=src python scripts/summarize_h2_confirmation.py --help
```

- All 17 tests passed. The five new seed-summary tests use synthetic fixtures,
  not experimental results. They check the paired n=5 t interval against an
  analytic reference, missing-run rejection, checkpoint/manifest hash checks,
  identity/nonfinite checks, and CLI output preservation.
- Ruff and whitespace checks passed; CLI help loaded successfully.
- All 92 historical result files and 626 archive members retained their hashes.
- The confirmation recipe differs from `h2_pilot.json` only in experiment ID
  and seed list. It specifies four conditions × five seeds = 20 runs.
- All local links in the changed README/protocol and the two new plan documents
  resolved. README values were cross-checked against the historical CSVs;
  rounding uses those CSV values rather than already-rounded report cells.
- The six proposed release assets were hashed directly at their original
  locations. Each matched both its historical CSV hash and the release draft.
  No checkpoint was copied, converted, overwritten or uploaded. Fresh validation
  of final downloadable release bytes remains pending.
- Historical H2 completion records summed to 2,729.6232444389607 epoch seconds
  across the four seed-0 runs, yielding a five-repetition estimate of 3.7911
  GPU-hours. This is a planning estimate from the old runner, not new timing.

## Execution boundary

The current restricted session reported `torch.cuda.is_available() == False`;
`nvidia-smi` also could not communicate with the driver. This establishes lack
of GPU access in this session, not the cause or driver health outside it.
No driver or system changes were attempted. GPU access and active jobs need
checking in the execution context before any training launch.

The new recipe has not been prepared with `liuwir prepare`, smoked or trained.
Preparation must freeze the actual GPU execution environment and retain the
confirmation plan before the first run. The independent evaluation set is a
candidate, not a frozen manifest or a verified independent dataset. Remote
repository metadata and checkpoint publication remain unchanged.
