"""
BraTS Dataset Download Helper Script

This script helps users download the BraTS dataset.
"""
import os
import sys
import argparse
from pathlib import Path


def check_data_exists(data_dir: str) -> bool:
    """
    檢查資料集是否已存在
    
    Args:
        data_dir: 資料集目錄
        
    Returns:
        是否存在
    """
    if not os.path.exists(data_dir):
        return False
    
    # 檢查是否有病人資料夾
    patient_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return len(patient_folders) > 0


def print_download_instructions():
    """
    顯示下載指引
    """
    print("\n" + "="*70)
    print("📦 BraTS Dataset Download Instructions")
    print("="*70 + "\n")
    
    print("BraTS (Brain Tumor Segmentation) 資料集需要手動下載或透過 Kaggle API 下載。\n")
    
    print("方法 1: 官方網站下載")
    print("-" * 70)
    print("1. 訪問 BraTS 官方網站:")
    print("   https://www.med.upenn.edu/cbica/brats2020/data.html")
    print("2. 註冊並下載 BraTS2020 Training Data")
    print("3. 解壓縮至專案的 data/Brats/ 目錄\n")
    
    print("方法 2: Kaggle API 下載 (推薦)")
    print("-" * 70)
    print("1. 安裝 Kaggle API:")
    print("   pip install kaggle")
    print()
    print("2. 設定 Kaggle API Token:")
    print("   - 登入 Kaggle: https://www.kaggle.com")
    print("   - 進入 Account Settings")
    print("   - 點擊 'Create New API Token'")
    print("   - 將下載的 kaggle.json 放到 ~/.kaggle/")
    print("   - 設定權限: chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("3. 下載資料集:")
    print("   kaggle datasets download -d awsaf49/brats20-dataset-training-validation")
    print()
    print("4. 解壓縮:")
    print("   unzip brats20-dataset-training-validation.zip -d data/")
    print("   mv data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/* data/Brats/")
    print()
    
    print("方法 3: 使用本腳本自動下載 (需要 Kaggle API)")
    print("-" * 70)
    print("   python scripts/download_brats.py --auto")
    print()
    
    print("="*70)
    print("資料集結構應如下:")
    print("="*70)
    print("""
data/Brats/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
└── ...
    """)
    print("="*70 + "\n")


def auto_download_kaggle(data_dir: str):
    """
    使用 Kaggle API 自動下載
    
    Args:
        data_dir: 資料集目錄
    """
    print("\n🚀 Starting automatic download via Kaggle API...\n")
    
    try:
        import kaggle
    except ImportError:
        print("❌ Error: Kaggle package not installed")
        print("Please install: pip install kaggle")
        return False
    
    # 檢查 Kaggle API token
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("❌ Error: Kaggle API token not found")
        print("Please follow the instructions above to set up Kaggle API")
        return False
    
    print("✓ Kaggle API configured")
    
    # 建立資料目錄
    os.makedirs(data_dir, exist_ok=True)
    
    # 下載資料集
    print("⏳ Downloading BraTS dataset (this may take a while)...")
    
    try:
        # 使用 kaggle API 下載
        os.system("kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p data/")
        
        print("✓ Download completed")
        print("⏳ Extracting files...")
        
        # 解壓縮
        os.system("cd data && unzip -q brats20-dataset-training-validation.zip")
        
        # 移動檔案到正確位置
        if os.path.exists("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"):
            os.system(f"mv data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/* {data_dir}/")
            os.system("rm -rf data/BraTS2020_TrainingData")
            os.system("rm data/brats20-dataset-training-validation.zip")
        
        print("✓ Extraction completed")
        print(f"✅ Dataset ready at: {data_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def main():
    """
    主函數
    """
    parser = argparse.ArgumentParser(description='BraTS Dataset Download Helper')
    parser.add_argument('--auto', action='store_true', help='Automatically download via Kaggle API')
    parser.add_argument('--data_dir', type=str, default='data/Brats', help='Data directory')
    
    args = parser.parse_args()
    
    # 檢查資料是否已存在
    if check_data_exists(args.data_dir):
        print(f"\n✅ Dataset already exists at: {args.data_dir}")
        patient_count = len([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
        print(f"   Found {patient_count} patients")
        return
    
    if args.auto:
        # 自動下載
        success = auto_download_kaggle(args.data_dir)
        if not success:
            print("\n⚠️  Automatic download failed. Please follow manual instructions below.")
            print_download_instructions()
    else:
        # 顯示下載指引
        print_download_instructions()


if __name__ == "__main__":
    main()
