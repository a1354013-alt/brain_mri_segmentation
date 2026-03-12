"""
BraTS Dataset Download Helper Script with Robust Structure Validation (v2.3)
"""
import argparse
import zipfile
import shutil
from pathlib import Path


def validate_and_align_structure(base_dir: Path) -> None:
    """
    驗證並自動對齊資料結構 (v2.3)：
    1. 僅將包含完整 4 模態 + Seg 的資料夾視為 Patient Folder
    2. 自動移動至 base_dir 的直接下一層
    3. 強化錯誤處理與日誌
    """
    print("🔍 Validating and aligning data structure...")
    modalities = ['flair', 't1', 't1ce', 't2']
    
    # 尋找所有包含 _flair.nii.gz 的檔案
    flair_files = list(base_dir.rglob("*_flair.nii.gz"))
    
    if not flair_files:
        print(f"❌ Error: No BraTS data found in {base_dir}.")
        return

    valid_patient_count = 0
    for flair_f in flair_files:
        p_dir = flair_f.parent
        pid = flair_f.name.replace('_flair.nii.gz', '')
        
        # 檢查完整性
        is_complete = True
        for mod in modalities:
            if not (p_dir / f"{pid}_{mod}.nii.gz").exists():
                is_complete = False
                break
        if not (p_dir / f"{pid}_seg.nii.gz").exists():
            is_complete = False
            
        if is_complete:
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
    
    # 清理空資料夾或不完整資料夾
    print("🧹 Cleaning up invalid or empty folders...")
    for item in base_dir.iterdir():
        if item.is_dir():
            # 檢查該資料夾是否為有效病人資料夾 (在 base_dir 下且完整)
            is_valid = False
            pid = item.name
            if (item / f"{pid}_flair.nii.gz").exists() and (item / f"{pid}_seg.nii.gz").exists():
                is_valid = True
            
            if not is_valid:
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
