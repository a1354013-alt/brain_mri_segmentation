import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

KAGGLE_DATASET_SLUG = "awsaf49/brats20-dataset-training-validation"
OFFICIAL_DOWNLOAD_URL = "https://www.med.upenn.edu/cbica/brats2020/data.html"

# Try importing kaggle at the top level
try:
    import kaggle
except ImportError:
    kaggle = None  # Set to None if not available


def is_patient_folder_complete(folder_path: Path) -> Tuple[bool, Optional[str]]:
    """
    統一完整性檢查邏輯：必須包含 4 模態 + Seg (v3.1 Final)
    """
    flair_files = list(folder_path.glob("*_flair.nii.gz"))
    if not flair_files:
        return False, None

    # v3.1 Final: 處理多個 flair 檔案的極端情況，優先選擇完整組合
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
    驗證並自動對齊資料結構 (v3.1 Final)：
    1. 解決搬移撞名問題：若目標已存在則追加後綴。
    2. 解決重複計數問題：使用 seen_dirs 記錄。
    3. 錯誤拋出：若未找到任何資料則拋出 RuntimeError。
    """
    print("🔍 Validating and aligning data structure...")

    all_flair_files = list(base_dir.rglob("*_flair.nii.gz"))

    if not all_flair_files:
        raise RuntimeError(f"❌ Error: No BraTS data found in {base_dir}. Please check your download.")

    valid_patient_count = 0
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
                # v3.1 Final 處理搬移衝突
                counter = 1
                while target_dir.exists():
                    target_dir = base_dir / f"{pid}_{counter}"
                    counter += 1

                print(f"📦 Moving valid patient {pid} to {target_dir.name}")
                try:
                    shutil.move(str(p_dir), str(target_dir))
                except Exception as e:
                    print(f"⚠️  Failed to move {pid}: {e}")
            valid_patient_count += 1

    print("🧹 Cleaning up invalid or empty folders...")
    for item in base_dir.iterdir():
        if item.is_dir():
            is_complete, _ = is_patient_folder_complete(item)
            if not is_complete:
                print(f"🗑️  Removing incomplete folder: {item.name}")
                try:
                    shutil.rmtree(item)
                except Exception as e:
                    print(f"⚠️  Could not remove {item}: {e}")

    print(f"✅ Final DATA_DIR structure validated. Found {valid_patient_count} valid patients.")
    return valid_patient_count > 0


def check_data_exists(data_dir: Path) -> bool:
    if not data_dir.exists():
        return False
    patient_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    return len(patient_folders) > 0


def print_download_instructions():
    print("\n" + "=" * 70)
    print("📦 BraTS Dataset Download Instructions (v3.1 Final)")
    print("=" * 70 + "\n")

    print(f"方法 1: 官方網站下載 ({OFFICIAL_DOWNLOAD_URL})")
    print(f"方法 2: Kaggle API 下載 (kaggle datasets download -d {KAGGLE_DATASET_SLUG})")

    print("\n資料集結構應如下:")
    print("data/Brats/")
    print("  ├── Patient_001/")
    print("  │     ├── Patient_001_flair.nii.gz")
    print("  │     ├── Patient_001_t1.nii.gz")
    print("  │     ├── Patient_001_t1ce.nii.gz")
    print("  │     ├── Patient_001_t2.nii.gz")
    print("  │     └── Patient_001_seg.nii.gz")

    print("\n下載完成後重新執行本程式。")
    print("=" * 70)


def auto_download_kaggle(data_dir: Path):
    print("\n🚀 Starting automatic download via Kaggle API (v3.1 Final)...\n")
    if kaggle is None:
        print("❌ Error: Kaggle package not installed. Run: pip install kaggle")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_SLUG, path=str(data_dir.parent), unzip=False
        )

        # v3.1 Final 偵測最新下載的 zip 檔案
        zip_files = sorted(list(data_dir.parent.glob("*.zip")), key=lambda x: x.stat().st_mtime, reverse=True)
        if not zip_files:
            print("❌ Error: No zip file found after download.")
            return False

        zip_path = zip_files[0]
        print(f"⏳ Extracting {zip_path.name} using zipfile...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        validate_and_align_structure(data_dir)

        if zip_path.exists():
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"❌ Error during download/extraction: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BraTS Dataset Download Helper (v3.1 Final)")
    parser.add_argument("--auto", action="store_true", help="Automatically download via Kaggle API")
    parser.add_argument("--data_dir", type=str, default="data/Brats", help="Data directory")

    args = parser.parse_args()
    data_path = Path(args.data_dir).resolve()

    if check_data_exists(data_path):
        print(f"\n✅ Dataset already exists at: {data_path}")
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
