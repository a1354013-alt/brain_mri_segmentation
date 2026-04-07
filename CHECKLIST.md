# Checklist (Current)

This checklist reflects what is verifiable in the current repository state.
It is intentionally conservative and does not claim production readiness.

## Must-Haves

- CLI does not crash on common boundary cases.
- Dataset validation fails fast with actionable messages.
- Inference can run without training-only deps being installed.
- Repo hygiene: deliverable zip excludes local artifacts.

## Verification Commands

No dataset required:

- `python tests/smoke_test.py`
- `python -m unittest -q tests.test_download_brats`
- `python -m unittest -q tests.test_cli_integration`

Requires dataset under `data/Brats/`:

- `python main.py train`
- `python main.py infer --patient_id <pid> --uncertainty entropy`
- `python main.py infer --patient_id <pid> --save_nifti --save_prob`
- `python main.py demo`

## Notes

- Legacy checklist content was archived for reference and should not be used as current status.
