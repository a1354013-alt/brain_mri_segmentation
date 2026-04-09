"""
Download and structure-alignment helper for the BraTS dataset.

This module intentionally avoids importing `kaggle` at import time. Importing kaggle can
trigger authentication checks in some environments, so we lazy-import it only when
`auto_download_kaggle()` is actually used.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional

KAGGLE_DATASET_SLUG = "awsaf49/brats20-dataset-training-validation"
OFFICIAL_DOWNLOAD_URL = "https://www.med.upenn.edu/cbica/brats2020/data.html"


def is_patient_folder_complete(folder_path: Path) -> tuple[bool, Optional[str]]:
    """
    Check whether `folder_path` contains one complete patient layout:
    FLAIR, T1, T1ce, T2, and segmentation.
    """
    flair_files = list(folder_path.glob("*_flair.nii.gz"))
    if not flair_files:
        return False, None

    best_pid = None
    for f in flair_files:
        pid = f.name.replace("_flair.nii.gz", "")
        modalities = ["flair", "t1", "t1ce", "t2"]
        required_exists = True
        for mod in modalities:
            if not (folder_path / f"{pid}_{mod}.nii.gz").exists():
                required_exists = False
                break
        if required_exists and (folder_path / f"{pid}_seg.nii.gz").exists():
            best_pid = pid
            break

    if best_pid:
        return True, best_pid
    return False, None


def validate_and_align_structure(base_dir: Path) -> bool:
    """
    Validate extracted files and move nested patient folders into `base_dir/<pid>`.

    Important safety rule:
    - If a target patient folder already exists, do not rename the incoming folder to
      `pid_1`, `pid_2`, etc. The inner filenames would still use the original pid and the
      dataset loader would later fail. Conflicts are logged instead of silently renamed.
    """
    print("Validating and aligning data structure...")

    all_flair_files = list(base_dir.rglob("*_flair.nii.gz"))
    if not all_flair_files:
        raise RuntimeError(f"Error: No BraTS data found in {base_dir}. Please check your download.")

    valid_patient_count = 0
    conflicts: list[str] = []
    seen_dirs = set()

    for flair_f in all_flair_files:
        p_dir = flair_f.parent
        if p_dir in seen_dirs:
            continue

        is_complete, pid = is_patient_folder_complete(p_dir)
        if is_complete and pid:
            seen_dirs.add(p_dir)
            if p_dir.parent != base_dir:
                target_dir = base_dir / pid
                if target_dir.exists():
                    conflicts.append(f"{pid}\t{p_dir}\t{target_dir}")
                    print(f"Warning: Conflict for {pid}. Target already exists: {target_dir}. Skipping move.")
                else:
                    print(f"Moving valid patient {pid} to {target_dir.name}")
                    try:
                        shutil.move(str(p_dir), str(target_dir))
                    except Exception as e:
                        print(f"Warning: Failed to move {pid}: {e}")
            valid_patient_count += 1

    if conflicts:
        conflict_log = base_dir / "align_conflicts.txt"
        with open(conflict_log, "w", encoding="utf-8") as f:
            f.write("pid\tsource_dir\ttarget_dir\n")
            f.write("\n".join(conflicts))
            f.write("\n")
        print(f"Conflicts saved to {conflict_log}")

    print(f"Validated DATA_DIR structure. Found {valid_patient_count} valid patients.")
    return valid_patient_count > 0


def check_data_exists(data_dir: Path) -> bool:
    """Return True only if at least one complete patient folder is present."""
    if not data_dir.exists():
        return False
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        is_complete, _pid = is_patient_folder_complete(d)
        if is_complete:
            return True
    return False


def print_download_instructions() -> None:
    print("\n" + "=" * 70)
    print("BraTS Dataset Download Instructions (v3.1 stable iteration)")
    print("=" * 70 + "\n")

    print(f"Option 1: Official source ({OFFICIAL_DOWNLOAD_URL})")
    print(f"Option 2: Kaggle API (kaggle datasets download -d {KAGGLE_DATASET_SLUG})")

    print("\nExpected folder layout:")
    print("data/Brats/")
    print("  Patient_001/")
    print("    Patient_001_flair.nii.gz")
    print("    Patient_001_t1.nii.gz")
    print("    Patient_001_t1ce.nii.gz")
    print("    Patient_001_t2.nii.gz")
    print("    Patient_001_seg.nii.gz")

    print("\nDownload finished. Re-run this script to validate or align the structure.")
    print("=" * 70)


def auto_download_kaggle(data_dir: Path) -> bool:
    print("\nStarting automatic download via Kaggle API (v3.1 stable iteration)...\n")

    try:
        import kaggle  # type: ignore
    except Exception as e:
        print("Error: Kaggle package not available or not configured.")
        print("Install: pip install kaggle")
        print(f"Details: {e}")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_dir = data_dir.parent / "_kaggle_download"
        download_dir.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(download_dir), unzip=False)

        slug_name = KAGGLE_DATASET_SLUG.split("/")[-1]
        zip_files = sorted(download_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not zip_files:
            print("Error: No zip file found after download.")
            return False

        preferred = [z for z in zip_files if slug_name in z.name]
        zip_path = preferred[0] if preferred else zip_files[0]

        print(f"Extracting {zip_path.name} using zipfile...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        validate_and_align_structure(data_dir)

        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error during download/extraction: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS Dataset Download Helper (v3.1 stable iteration)")
    parser.add_argument("--auto", action="store_true", help="Automatically download via Kaggle API")
    parser.add_argument("--data_dir", type=str, default="data/Brats", help="Data directory")

    args = parser.parse_args()
    data_path = Path(args.data_dir).resolve()

    if check_data_exists(data_path):
        print(f"\nDataset already exists at: {data_path}")
        try:
            validate_and_align_structure(data_path)
        except RuntimeError as e:
            print(str(e))
        return

    if args.auto:
        auto_download_kaggle(data_path)
    else:
        print_download_instructions()


if __name__ == "__main__":
    main()
