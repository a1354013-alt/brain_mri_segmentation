"""
BraTS Dataset Download Helper Script
"""
import argparse
import subprocess
from pathlib import Path


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
        print("❌ Error: Kaggle package not installed")
        return False
    
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "awsaf49/brats20-dataset-training-validation", "-p", str(data_dir.parent)], check=True)
        subprocess.run(["unzip", "-q", str(data_dir.parent / "brats20-dataset-training-validation.zip"), "-d", str(data_dir.parent)], check=True)
        print(f"✅ Dataset ready at: {data_dir}")
        return True
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='BraTS Dataset Download Helper')
    parser.add_argument('--auto', action='store_true', help='Automatically download via Kaggle API')
    parser.add_argument('--data_dir', type=str, default='data/Brats', help='Data directory')
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    
    if check_data_exists(data_path):
        print(f"\n✅ Dataset already exists at: {data_path}")
        return
    
    if args.auto:
        auto_download_kaggle(data_path)
    else:
        print_download_instructions()


if __name__ == "__main__":
    main()
