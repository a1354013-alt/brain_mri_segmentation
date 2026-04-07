import shutil
import uuid
import unittest
from pathlib import Path


from scripts.download_brats import check_data_exists


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _repo_tmp_dir() -> Path:
    base = Path(__file__).resolve().parent / "_tmp_download_brats"
    base.mkdir(parents=True, exist_ok=True)
    d = base / str(uuid.uuid4())
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_dir(d: Path) -> None:
    try:
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


class TestDownloadBrats(unittest.TestCase):
    def test_check_data_exists_empty_dir_false(self):
        d = _repo_tmp_dir()
        try:
            self.assertFalse(check_data_exists(d))
        finally:
            _cleanup_dir(d)

    def test_check_data_exists_incomplete_patient_false(self):
        base = _repo_tmp_dir()
        try:
            p = base / "Patient_001"
            p.mkdir(parents=True, exist_ok=True)
            _touch(p / "Patient_001_flair.nii.gz")
            self.assertFalse(check_data_exists(base))
        finally:
            _cleanup_dir(base)

    def test_check_data_exists_complete_patient_true(self):
        base = _repo_tmp_dir()
        try:
            p = base / "Patient_001"
            p.mkdir(parents=True, exist_ok=True)
            for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
                _touch(p / f"Patient_001_{mod}.nii.gz")
            self.assertTrue(check_data_exists(base))
        finally:
            _cleanup_dir(base)


if __name__ == "__main__":
    unittest.main()
