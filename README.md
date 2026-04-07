# Brain MRI Tumor Segmentation (BraTS, 2D Slice, Attention U-Net)

Version: v3.1 stable iteration

This repository is a compact, CLI-driven project for brain tumor segmentation on the BraTS-style dataset.
It trains a 2D slice model (Attention U-Net) and supports MC Dropout uncertainty during inference.

This is not a production pipeline. The goal is: stable CLI flows, predictable data validation, and a clean delivery.

## What This Project Does

- Train: `python main.py train`
- Inference: `python main.py infer [--patient_id <pid>] [--uncertainty var|entropy]`
- Demo (1-epoch quick run on 2 auto-selected patients): `python main.py demo`
- Optional 3D NIfTI export during infer: `--save_nifti` and `--save_prob`

## Dataset Layout

Expected directory structure:

```
data/Brats/<patient_id>/
  <patient_id>_flair.nii.gz
  <patient_id>_t1.nii.gz
  <patient_id>_t1ce.nii.gz
  <patient_id>_t2.nii.gz
  <patient_id>_seg.nii.gz
```

Helper script:

- `python scripts/download_brats.py --auto` optionally downloads via Kaggle API.
- The script validates and aligns extracted structures to `data/Brats/<pid>/...`.

Notes:

- `scripts/download_brats.py` uses lazy import for `kaggle` so local helpers and unit tests do not require Kaggle auth.

## Install

```bash
pip install -r requirements.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

## Dependency Notes (Flexibility)

- `infer` and dataset scanning require `nibabel` (and `numpy`).
- `demo` uses the training stack (`train.py` / `Trainer`) so it requires the same core dependencies as training.
- TensorBoard logging is optional: if `tensorboard` is not installed, training/demo will fall back to a no-op writer.
- Ruff lint is only verifiable in an environment where `ruff` is installable/available.
- Smoke test requires `torch`.

## CLI Usage

Train:

```bash
python main.py train
```

Inference:

```bash
python main.py infer --uncertainty entropy
python main.py infer --patient_id Patient_001 --uncertainty var
```

Inference with 3D NIfTI outputs:

```bash
python main.py infer --patient_id Patient_001 --save_nifti --save_prob
```

Demo:

```bash
python main.py demo
```

### Patient Validation Behavior (infer/demo auto-selection)

Auto-selection uses a two-phase strategy:

1. Fast check: file presence + NIfTI readability + shape consistency + sampled tumor slices
2. Strict check: full seg scan for tumor presence (only for candidates that pass phase 1)

This reduces false negatives while avoiding a full scan over every patient.

## Uncertainty (MC Dropout)

MC Dropout runs multiple stochastic forward passes with dropout enabled.

- `method=var`: pixel-wise variance across samples
- `method=entropy`: predictive entropy computed from the mean probability

## Models

- Primary model used by CLI: `models/attention_unet.py` (`AttentionUNet`)
- Baseline / legacy comparison: `models/unet.py` (`UNet`)

## Tests

This repo uses `unittest` for lightweight integration tests (no dataset required).

| Test / Command | Requires dataset | Requires extra deps | What it covers |
| --- | --- | --- | --- |
| `python tests/smoke_test.py` | No | torch | Model init + one forward pass (fast) |
| `python -m unittest -q tests.test_download_brats` | No | None | `check_data_exists()` correctness on empty/partial/complete layouts |
| `python -m unittest -q tests.test_cli_integration` | No | None (stubbed) | CLI boundaries: train split protections, infer not importing training deps, device override, `--save_nifti` branch, demo empty dataset safe exit |
| `python main.py train` | Yes | torch, nibabel, etc. | End-to-end data scan, split, training loop |
| `python main.py infer --patient_id <pid>` | Yes | torch, nibabel, matplotlib | End-to-end single-patient infer + PNG |
| `python main.py infer --save_nifti` | Yes | torch, nibabel | 3D NIfTI export branch |

## Lint

Ruff configuration lives in `pyproject.toml`.

```bash
ruff check . --fix
ruff format .
```

## Clean Delivery Zip

To build a clean source zip that excludes local artifacts (for example `.git/`, `__pycache__/`, `outputs/`, `data/`):

```bash
python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip
```

## Windows Notes

Some environments block writing `__pycache__` or use legacy terminal encodings.
`sitecustomize.py` is opt-in via env flags:

```powershell
$env:BMS_DONT_WRITE_BYTECODE = "1"
$env:BMS_FORCE_UTF8 = "1"
python main.py infer
```

By default, `sitecustomize.py` does nothing unless these environment variables are set.

## Known Limitations

- Training is slice-based (2D). It is not a full BraTS 3D pipeline with official challenge metrics.
- Dataset length is per-patient (each epoch samples one slice per patient). This is intentional for speed and simplicity.
- True end-to-end verification requires a real BraTS dataset present in `data/Brats/`.
