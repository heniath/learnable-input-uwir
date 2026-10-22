# Consolidation validation — 2026-10-22

- Ruff checks passed.
- Twelve CPU regression tests passed.
- Offline wheel build passed without installing or changing dependencies.
- Historical records: all 92 result files and 626 archived members passed SHA-256 verification.
- Migration audit: paired preprocessing was exact; all 20 initial model state/forward comparisons were exact.
- All 31 historical checkpoint validation anchors passed on the complete 80-image validation split. Maximum aggregate PSNR drift: 1.4210854715202004e-14 dB; SSIM drift: zero.
- GPU smoke passed for H1 global, H2 combined, d=6 global, U-Net capacity and NAFNet bounded conditions. Each used two optimizer updates and exact full checkpoint-state restoration.
- A separate one-epoch functionality condition completed training, checkpoint selection, completed-run preservation, validation-anchored evaluation and global diagnostics. Temporary binary artifacts were removed automatically. This is not a research comparison or a reported quality result.

The pinned environment is recorded in [migration.json](migration.json) and [runtime.json](runtime.json). The existing Kornia AMP deprecation warning remained visible; no dependencies or system settings were changed.

Commands (inside the recorded environment):

```bash
python -m ruff check src tests scripts
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/verify_records.py
python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /tmp/liuwir-build .
PYTHONPATH=src python scripts/validate_migration.py --workspace /path/to/original/workspace --data-root /path/to/UIEB --device cuda --destination results/validation/migration.json
PYTHONPATH=src python scripts/validate_runtime.py --data-root /path/to/UIEB --vgg-weights /path/to/vgg16-397923af.pth --destination results/validation/runtime.json
```

Full historical training was not repeated. GPU trajectories need not replay bit for bit. Checkpoints and original study artifacts were not modified.
