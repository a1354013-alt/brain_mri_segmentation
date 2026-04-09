# Delivery Summary (v3.1 stable iteration)

## Scope

- 2D slice-based brain tumor segmentation (BraTS-style) using Attention U-Net
- CLI flows: train / infer / demo
- MC Dropout uncertainty during inference (variance or entropy)
- Optional 3D NIfTI export in inference

## Key Stability Fixes Included

- Train/val cache handling: train/val build independent caches (no cache mismatch crash).
- Train/val split boundary protection: refuses to train when patient count cannot produce non-empty splits.
- Lazy import of training module: `infer` does not import `Trainer` or `tensorboard` at import time.
- Device override correctness: CLI `--device` override is forwarded to MC Dropout inference.
- Demo empty-dataset guard: demo exits safely if scanning yields 0 valid patients.
- Download helper decoupling: `scripts/download_brats.py` uses lazy `kaggle` import so local helpers can be used without Kaggle auth.

## What We Can Automatically Verify (No Dataset Required)

- `python tests/smoke_test.py`
- `python -m unittest -q tests.test_download_brats`
- `python -m unittest -q tests.test_cli_integration`
- `python -m compileall -q .`
- `python scripts/clean_project.py --clean-dist`
- `python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip`

## What Requires Real Data

- `python main.py train`
- `python main.py infer --patient_id <pid>`
- `python main.py infer --save_nifti --save_prob`
- `python main.py demo`

## Clean Packaging

- `python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip`

This excludes local artifacts such as `.git/`, `__pycache__/`, `outputs/`, and `data/`.
The delivery zip should be generated via this script rather than zipping the working directory directly.
The script also excludes common local caches and scratch artifacts (for example `tests/_tmp_*`, `*.pyc`, `*.pth`, `*.png`).
The release zip is always written under `dist/` (for example `dist/brain_mri_segmentation_src.zip`).
For stricter CI-style enforcement, both `clean_project.py` and `make_release_zip.py` support `--strict`.

## Residual Risks / Constraints

- True end-to-end training/inference validation depends on having a real BraTS dataset present.
- Some environments may block deleting certain local artifacts (for example `outputs/` or scratch files). The release zip script excludes them regardless.
