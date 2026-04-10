# Checklist (Current)

This checklist reflects what is verifiable in the current repository state.
It is intentionally conservative and does not claim production readiness.

## Must-Haves

- CLI does not crash on common boundary cases.
- Dataset validation fails fast with actionable messages.
- Inference can run without training-only deps being installed.
- Repo hygiene: deliverable zip excludes local artifacts.
- Release packaging is fixed: delivery zip is generated only via `python scripts/make_release_zip.py --out ...` (written under `dist/`).
- Base requirements are intentionally separated from optional extras; install `requirements-optional.txt` only when Kaggle or TensorBoard support is needed.

## Verification Commands

No dataset required:

- `python tests/smoke_test.py`
- `python -m unittest -q tests.test_download_brats tests.test_cli_integration tests.test_release_and_clean`
- `python -m compileall -q .`
- `python scripts/clean_project.py --clean-dist`
- `python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip`

Pre-release/CI-style strict mode:

- `python scripts/clean_project.py --clean-dist --strict`
- `python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip --strict --check-git`

Requires dataset under `data/Brats/`:

- `python main.py train`
- `python main.py infer --patient_id <pid> --uncertainty entropy`
- `python main.py infer --patient_id <pid> --save_nifti --save_prob`
- `python main.py demo`

## Notes

- Legacy checklist content was archived for reference and should not be used as current status.
