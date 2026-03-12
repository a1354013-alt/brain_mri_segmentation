"""
BraTS Dataset Download Helper Script with Unified Validation (v2.4 Final)
"""
import argparse
import zipfile
import shutil
from pathlib import Path


def is_patient_folder_complete(folder_path: Path) -> bool:
    """
    統一完整性檢查邏輯：必須包含 4 模態 + Seg
    """
    # 尋找資料夾內的 flair 檔案來推導 PID
    flair_files = list(folder_path.glob("*_flair.nii.gz"))
    if not flair_files:
        return False
    
    pid = flair_files[0].name.replace('_flair.nii.gz', '')
    modalities = ['flair', 't1', 't1ce', 't2']
    
    for mod in modalities:
        if not (folder_path / f"{pid}_{mod}.nii.gz").exists():
            return False
    if not (folder_path / f"{pid}_seg.nii.gz").exists():
        return False
        
    return True


def validate_and_align_structure(base_dir: Path) -> None:
    """
    驗證並自動對齊資料結構 (v2.4)：
    1. 統一使用 is_patient_folder_complete 進行判定
    2. 自動從檔名推導 PID，不依賴資料夾名稱
    3. 強化清理邏輯，確保不留殘餘不完整資料
    """
    print("🔍 Validating and aligning data structure...")
    
    # 1. 尋找所有潛在的病人資料夾 (包含子目錄)
    all_flair_files = list(base_dir.rglob("*_flair.nii.gz"))
    
    if not all_flair_files:
        print(f"❌ Error: No BraTS data found in {base_dir}.")
        return

    valid_patient_count = 0
    for flair_f in all_flair_files:
        p_dir = flair_f.parent
        
        if is_patient_folder_complete(p_dir):
            # 如果資料夾不在 base_dir 的直接下一層，則移動它
            if p_dir.parent != base_dir:
                target_dir = base_dir / p_dir.name
                if not target_dir.exists():
                    print(f"📦 Moving valid patient {p_dir.name} to {base_dir}")
                    try:
                        shutil.move(str(p_dir), str(target_dir))
                    except Exception as e:
                        print(f"⚠️  Failed to move {p_dir.name}: {e}")
            valid_patient_count += 1
    
    # 2. 清理邏輯 (v2.4)：統一完整性檢查
    print("🧹 Cleaning up invalid or empty folders...")
    for item in base_dir.iterdir():
        if item.is_dir():
            if not is_patient_folder_complete(item):
                print(f"🗑️  Removing incomplete folder: {item.name}")
                try:
                    shutil.rmtree(item)
                except Exception as e:
                    print(f"⚠️  Could not remove {item}: {e}")

    print(f"✅ Final DATA_DIR structure validated. Found {valid_patient_count} valid patients.")


def check_data_exists(data_dir: Path) -> bool:
    if not data_dir.exists():
        return False
    patient_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    return len(patient_folders) > 0


def print_download_instructions():
    print("\n" + "="*70)
    print("📦 BraTS Dataset Download Instructions")
    print("="*70 + "\n")
    print("方法 1: 官方網站下載 (https://www.med.upenn.edu/cbica/brats2020/data.html)")
    print("方法 2: Kaggle API 下載 (kaggle datasets download -d awsaf49/brats20-dataset-training-validation)")
    print("\n資料集結構應如下:")
    print("data/Brats/Patient_XXX/...")
    print("="*70 + "\n")


def auto_download_kaggle(data_dir: Path):
    print("\n🚀 Starting automatic download via Kaggle API...\n")
    try:
        import kaggle
    except ImportError:
        print("❌ Error: Kaggle package not installed. Run: pip install kaggle")
        return False
    
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir.parent / "brats20-dataset-training-validation.zip"
    
    try:
        kaggle.api.dataset_download_files("awsaf49/brats20-dataset-training-validation", path=str(data_dir.parent), unzip=False)
        
        print(f"⏳ Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        validate_and_align_structure(data_dir)
        
        if zip_path.exists():
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"❌ Error during download/extraction: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='BraTS Dataset Download Helper')
    parser.add_argument('--auto', action='store_true', help='Automatically download via Kaggle API')
    parser.add_argument('--data_dir', type=str, default='data/Brats', help='Data directory')
    
    args = parser.parse_args()
    data_path = Path(args.data_dir).resolve()
    
    if check_data_exists(data_path):
        print(f"\n✅ Dataset already exists at: {data_path}")
        validate_and_align_structure(data_path)
        return
    
    if args.auto:
        auto_download_kaggle(data_path)
    else:
        print_download_instructions()


if __name__ == "__main__":
    main()
