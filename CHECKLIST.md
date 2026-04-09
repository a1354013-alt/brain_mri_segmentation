# Checklist (Current)

This checklist reflects what is verifiable in the current repository state.
It is intentionally conservative and does not claim production readiness.

## Must-Haves

- CLI does not crash on common boundary cases.
- Dataset validation fails fast with actionable messages.
- Inference can run without training-only deps being installed.
- Repo hygiene: deliverable zip excludes local artifacts.
- Release packaging is fixed: delivery zip is generated only via `python scripts/make_release_zip.py --out ...` (written under `dist/`).

## Verification Commands

No dataset required:

- `python tests/smoke_test.py`
- `python -m unittest -q tests.test_download_brats`
- `python -m unittest -q tests.test_cli_integration`
- `python -m compileall -q .`
- `python scripts/clean_project.py --clean-dist`
- `python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip`

Requires dataset under `data/Brats/`:

- `python main.py train`
- `python main.py infer --patient_id <pid> --uncertainty entropy`
- `python main.py infer --patient_id <pid> --save_nifti --save_prob`
- `python main.py demo`

## Notes

- Legacy checklist content was archived for reference and should not be used as current status.
