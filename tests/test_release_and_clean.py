import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class TestReleaseAndClean(unittest.TestCase):
    def _copy_script(self, script_name: str, repo_root: Path) -> Path:
        scripts_dir = repo_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve().parent.parent / "scripts" / script_name
        destination = scripts_dir / script_name
        shutil.copy2(source, destination)
        return destination

    def test_make_release_zip_excludes_non_deliverable_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Create files that should be excluded from the release zip.
            (repo_root / "data" / "Patient_001").mkdir(parents=True, exist_ok=True)
            (repo_root / "data" / "Patient_001" / "Patient_001_flair.nii.gz").write_text("dummy")
            (repo_root / "outputs").mkdir(parents=True, exist_ok=True)
            (repo_root / "outputs" / "ignored.txt").write_text("ignored")
            (repo_root / "archive").mkdir(parents=True, exist_ok=True)
            (repo_root / "archive" / "ignored.txt").write_text("ignored")
            (repo_root / ".git").mkdir(parents=True, exist_ok=True)
            (repo_root / ".git" / "config").write_text("ignored")
            (repo_root / "dist").mkdir(parents=True, exist_ok=True)
            (repo_root / "dist" / "old_release.zip").write_text("old")
            (repo_root / "README.md").write_text("project README")

            script = self._copy_script("make_release_zip.py", repo_root)
            result = subprocess.run(
                [sys.executable, str(script), "--out", "brain_mri_segmentation_src.zip"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            out_zip = repo_root / "dist" / "brain_mri_segmentation_src.zip"
            self.assertTrue(out_zip.exists(), "Expected release zip to be created under dist/")

            with zipfile.ZipFile(out_zip, "r") as zf:
                names = zf.namelist()

            self.assertTrue(any(name.endswith("README.md") for name in names), "Expected README.md to be included in the zip")
            self.assertFalse(any(name.startswith("data/") for name in names), "data/ should be excluded from the release zip")
            self.assertFalse(any(name.startswith("outputs/") for name in names), "outputs/ should be excluded from the release zip")
            self.assertFalse(any(name.startswith("archive/") for name in names), "archive/ should be excluded from the release zip")
            self.assertFalse(any(name.startswith(".git/") for name in names), ".git/ should be excluded from the release zip")
            self.assertFalse(any(name.startswith("dist/") for name in names), "dist/ contents should be excluded from the release zip")
            self.assertFalse(any(name.endswith(".zip") for name in names), "Zip-in-zip files should be excluded from the release zip")

    def test_clean_project_strict_removes_known_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            script = self._copy_script("clean_project.py", repo_root)

            (repo_root / "tests" / "_tmp_artifact").mkdir(parents=True, exist_ok=True)
            (repo_root / "tests" / "_tmp_artifact" / "dummy.txt").write_text("temp")
            (repo_root / "__pycache__").mkdir(parents=True, exist_ok=True)
            (repo_root / "__pycache__" / "dummy.pyc").write_text("")
            (repo_root / "dist").mkdir(parents=True, exist_ok=True)
            (repo_root / "dist" / "old.txt").write_text("old")

            result = subprocess.run(
                [sys.executable, str(script), "--strict"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertFalse((repo_root / "tests" / "_tmp_artifact").exists())
            self.assertFalse((repo_root / "__pycache__").exists())
            self.assertFalse((repo_root / "dist" / "old.txt").exists())


if __name__ == "__main__":
    unittest.main()
