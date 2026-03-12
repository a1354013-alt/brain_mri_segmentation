import os
import torch
from torch.utils.data import DataLoader, random_split
from models.attention_unet import AttentionUNet
from utils.dataset import BraTSDataset
from train import train_model
from utils.visualize import mc_dropout_inference, plot_results_with_uncertainty
import nibabel as nib
import numpy as np

# --- 專案設定 ---
DATA_DIR = "data/BraTS2020_TrainingData" # 假設資料已下載並解壓縮至此
MODEL_PATH = "models/attention_unet_brats_model.pth"
IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_patient_ids(data_dir):
    """
    模擬取得 BraTS 資料集中的病人 ID 列表。
    """
    # 實際操作中，您需要根據 BraTS 資料集的結構來取得所有病人資料夾名稱
    # 這裡僅為示意，假設資料夾名稱為 'BraTS20_Training_XXX'
    print("WARNING: Data loading is simulated. Please replace this with actual data loading logic.")
    
    # 由於無法下載實際資料，這裡返回一個空列表，讓程式碼結構完整
    # 實際使用時，請確保 DATA_DIR 內有資料夾
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}. Please download BraTS data.")
        return []
        
    # 實際的 BraTS 資料夾名稱
    patient_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # 為了讓程式碼能運行，即使沒有資料，我們也返回一個模擬的 ID
    if not patient_ids:
        print("No patient folders found. Returning a dummy ID for structure demonstration.")
        return ["dummy_patient_001"] # 實際運行時會因為找不到檔案而失敗，但結構完整

    return patient_ids

def main():
    print(f"--- 腦部 MRI 腫瘤分割專案啟動 (Device: {DEVICE}) ---")
    
    # 1. 資料準備
    patient_ids = get_patient_ids(DATA_DIR)
    if not patient_ids:
        print("無法找到資料集，請先下載 BraTS 資料集並放置於 'data/BraTS2020_TrainingData'。")
        return

    # 由於我們無法實際運行，這裡只展示邏輯
    # 實際操作中，您需要確保 BraTSDataset 能夠成功載入 NIfTI 檔案
    
    # 假設我們只使用前 10 個病人進行演示
    if len(patient_ids) > 10:
        patient_ids = patient_ids[:10]
        
    # 劃分訓練集和驗證集
    train_size = int(0.8 * len(patient_ids))
    val_size = len(patient_ids) - train_size
    train_ids, val_ids = random_split(patient_ids, [train_size, val_size])
    
    # 由於 random_split 返回的是 Subset，我們需要提取原始 ID
    train_ids = [patient_ids[i] for i in train_ids.indices]
    val_ids = [patient_ids[i] for i in val_ids.indices]

    # 建立 Dataset 和 DataLoader
    # train_dataset = BraTSDataset(data_dir=DATA_DIR, patient_ids=train_ids)
    # val_dataset = BraTSDataset(data_dir=DATA_DIR, patient_ids=val_ids)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"訓練集病人數: {len(train_ids)}, 驗證集病人數: {len(val_ids)}")
    
    # 2. 模型初始化
    model = AttentionUNet(n_channels=4, n_classes=1).to(DEVICE)
    print("Attention U-Net 模型初始化完成 (4 通道輸入)。")
    
    # 3. 訓練模型 (由於沒有資料，這裡註釋掉實際訓練)
    # print("開始訓練模型...")
    # train_model(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
    # torch.save(model.state_dict(), MODEL_PATH)
    # print(f"模型已儲存至 {MODEL_PATH}")
    
    # 4. 推論與視覺化 (使用模擬資料進行視覺化展示)
    print("使用模擬資料進行推論與不確定性估計展示...")
    
    # 模擬一個 4 通道的 MRI 影像 (FLAIR, T1, T1ce, T2)
    dummy_image = np.random.rand(1, 4, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    dummy_mask = (np.random.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE) > 0.8).astype(np.float32)
    
    # 模擬 MC Dropout 推論結果
    dummy_prediction = (np.random.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE) > 0.7).astype(np.float32)
    dummy_uncertainty = np.random.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE) * 0.1
    
    # 轉換為 PyTorch Tensor
    image_tensor = torch.from_numpy(dummy_image)
    mask_tensor = torch.from_numpy(dummy_mask)
    
    # 實際推論步驟 (如果模型已訓練)
    # prediction, uncertainty = mc_dropout_inference(model, image_tensor, n_iterations=10, device=DEVICE)
    
    # 視覺化
    # plot_results_with_uncertainty(
    #     image_tensor.squeeze(0).cpu().numpy(), 
    #     mask_tensor.squeeze(0).cpu().numpy(), 
    #     dummy_prediction.squeeze(0),
    #     dummy_uncertainty.squeeze(0),
    #     title="優化後推論結果 (Attention U-Net + MC Dropout)"
    # )
    print("視覺化程式碼已準備，但因無法在沙盒環境中顯示 Matplotlib 視窗，請在本地運行。")
    print("請參考 utils/visualize.py 中的 plot_results 函數。")

if __name__ == "__main__":
    main()
