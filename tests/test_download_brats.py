import unittest
from dataclasses import dataclass
from typing import Iterable
from unittest.mock import patch


from scripts.download_brats import check_data_exists


@dataclass(frozen=True)
class _FakeEntry:
    is_dir_value: bool = True

    def is_dir(self) -> bool:
        return self.is_dir_value


@dataclass(frozen=True)
class _FakeDataDir:
    exists_value: bool
    entries: Iterable[_FakeEntry]

    def exists(self) -> bool:
        return self.exists_value

    def iterdir(self):
        return iter(self.entries)


class TestDownloadBrats(unittest.TestCase):
    def test_check_data_exists_empty_dir_false(self):
        d = _FakeDataDir(exists_value=True, entries=[])
        self.assertFalse(check_data_exists(d))  # type: ignore[arg-type]

    def test_check_data_exists_incomplete_patient_false(self):
        base = _FakeDataDir(exists_value=True, entries=[_FakeEntry(is_dir_value=True)])
        with patch("scripts.download_brats.is_patient_folder_complete", return_value=(False, "Patient_001")):
            self.assertFalse(check_data_exists(base))  # type: ignore[arg-type]

    def test_check_data_exists_complete_patient_true(self):
        base = _FakeDataDir(exists_value=True, entries=[_FakeEntry(is_dir_value=True)])
        with patch("scripts.download_brats.is_patient_folder_complete", return_value=(True, "Patient_001")):
            self.assertTrue(check_data_exists(base))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
