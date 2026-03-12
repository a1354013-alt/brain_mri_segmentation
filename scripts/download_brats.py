"""
BraTS Dataset Download Helper Script with Structure Validation
"""
import argparse
import zipfile
import shutil
from pathlib import Path


def validate_and_align_structure(base_dir: Path) -> None:
    """
    驗證並自動對齊資料結構：data/Brats/<patient_id>/...
    """
    print("🔍 Validating and aligning data structure...")
    
    # 尋找包含 _flair.nii.gz 的資料夾
    flair_files = list(base_dir.rglob("*_flair.nii.gz"))
    
    if not flair_files:
        raise RuntimeError(f"❌ Error: No BraTS data found in {base_dir}. Expected *_flair.nii.gz files.")
    
    # 取得所有包含資料的實際資料夾
    actual_data_dirs = sorted(list(set([f.parent for f in flair_files])))
    
    for p_dir in actual_data_dirs:
        # 如果資料夾不在 base_dir 的直接下一層，則移動它
        if p_dir.parent != base_dir:
            target_dir = base_dir / p_dir.name
            if not target_dir.exists():
                print(f"📦 Moving {p_dir.name} to {base_dir}")
                shutil.move(str(p_dir), str(target_dir))
    
    # 清理空資料夾
    for item in base_dir.iterdir():
        if item.is_dir() and not any(item.rglob("*_flair.nii.gz")):
            try:
                shutil.rmtree(item)
            except:
                pass

    print("✅ Final DATA_DIR structure validated.")


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
        # 下載
        kaggle.api.dataset_download_files("awsaf49/brats20-dataset-training-validation", path=str(data_dir.parent), unzip=False)
        
        # 使用 Python zipfile 解壓 (跨平台)
        print(f"⏳ Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # 結構對齊
        validate_and_align_structure(data_dir)
        
        # 移除 zip
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
