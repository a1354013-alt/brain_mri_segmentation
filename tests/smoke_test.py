"""
Smoke Test for Brain MRI Segmentation Project (v3.0 Final Release Gold Master)
Ensures core components (Dataset, Model, Inference) are working correctly.
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from models import AttentionUNet
from utils import BraTSDataset, mc_dropout_inference


def run_smoke_test():
    print("🧪 Running Smoke Test (v3.0 Final Release Gold Master)...")
    
    # 1. Test Model Initialization
    print("📡 Testing Model Initialization...")
    model = AttentionUNet(n_channels=4, n_classes=1, dropout_p=0.2)
    dummy_input = torch.randn(1, 4, 128, 128)
    output = model(dummy_input)
    assert output.shape == (1, 1, 128, 128), f"Model output shape mismatch: {output.shape}"
    print("✅ Model Initialization Passed.")

    # 2. Test Dataset Logic (Mocking if no data)
    print("📂 Testing Dataset Logic...")
    patient_ids = ["test_patient_001"]
    
    # v3.0 Final: 實作 Mock Dataset 測試，確保在無真實資料時也能驗證核心邏輯
    try:
        # 建立 Mock Cache 模擬已掃描過的資料
        mock_cache = {
            "valid_patient_ids": ["test_patient_001"],
            "patient_cache": {
                "test_patient_001": {
                    "files": {mod: f"mock_{mod}.nii.gz" for mod in ['flair', 't1', 't1ce', 't2', 'seg']},
                    "tumor_slice_indices": [60, 61, 62],
                    "val_best_slice_idx": 61
                }
            },
            "proxy_cache": {}
        }
        
        # 測試共享快取子集化邏輯
        print("🔗 Testing Shared Cache Subsetting...")
        dataset = BraTSDataset(
            data_dir=project_root / "data" / "Brats",
            patient_ids=["test_patient_001"],
            prepared_cache=mock_cache,
            mode='train'
        )
        assert len(dataset) == 1, "Dataset length mismatch with mock cache"
        assert dataset.valid_patient_ids == ["test_patient_001"]
        print("✅ Shared Cache Subsetting Passed.")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        raise e

    # 3. Test MC Dropout Inference
    print("🧠 Testing MC Dropout Inference...")
    model.eval()
    # 使用較少的迭代次數進行冒煙測試
    prediction, uncertainty = mc_dropout_inference(model, dummy_input, n_iterations=5)
    assert prediction.shape == (1, 1, 128, 128)
    assert uncertainty.shape == (1, 1, 128, 128)
    print("✅ MC Dropout Inference Passed.")

    print("\n🎉 All Smoke Tests Passed! (v3.0 Final Release Ready)")


if __name__ == "__main__":
    run_smoke_test()
