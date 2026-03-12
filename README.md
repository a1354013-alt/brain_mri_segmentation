# Brain MRI Tumor Segmentation (v2.6 Final Gold Master)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與極致優化的 I/O 與快取機制。

---

## 📋 專案更新亮點 (v2.6 Final)

- **快取共享機制 (Shared Cache)**：`BraTSDataset` 現在支援傳入外部 `prepared_cache`。在訓練啟動時，訓練集與驗證集會共享掃描結果，避免了對同一批病人資料進行重複掃描，大幅縮短啟動時間。
- **輕量化驗證 (Lightweight Validation)**：實作了 `quick_validate_patient` 靜態方法。在推論與 Demo 模式下，系統僅檢查檔案存在性，不再執行耗時的逐切片掃描，實現秒級啟動。
- **Last Checkpoint 雙重保險**：`Trainer` 現在除了儲存 `best_checkpoint.pth` 外，每一輪都會更新 `last_checkpoint.pth`。這確保了即使驗證指標未創新高，使用者仍能獲得最新的訓練狀態。
- **Kaggle 下載魯棒性強化**：`download_brats.py` 現在會自動偵測最新下載的 zip 檔案，不再依賴寫死的檔名。同時，搬移衝突時會自動追加後綴（如 `_1`, `_2`），避免資料遺失。
- **進度條一致性修正**：修正了 `tqdm` 進度條在 Demo 模式下顯示錯誤總輪數的問題，現在會根據實際傳入的 `epochs` 動態調整。
- **Demo 模式一致性**：顯式指定 `num_workers=0` 並同步隨機種子，確保 Demo 流程的行為與正式訓練完全一致。

---

## 🏗️ 技術細節

### 1. 快取共享
透過 `dataset.get_cache()` 提取掃描後的 Metadata（含有效 ID、腫瘤索引、Proxy 快取），並透過 `prepared_cache` 參數傳遞給下一個 Dataset 實例。

### 2. 下載衝突處理
使用 `while target_dir.exists()` 迴圈偵測目標路徑，確保在多來源資料合併時不會發生覆蓋。

---

## 🚀 執行指令

### 1. 資料準備
```bash
# 自動下載、解壓、對齊並處理衝突 (具備最新 zip 偵測)
python scripts/download_brats.py --auto
```

### 2. 訓練與推論
```bash
# 訓練 (支援快取共享，啟動極速)
python main.py train

# 推論 (輕量化驗證，秒級啟動)
python main.py infer --uncertainty entropy

# Demo (行為一致性強化)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 核心配置 (含 LAST_CHECKPOINT_PATH)
├── main.py                # CLI 入口 (含快取共享與輕量驗證)
├── train.py               # 訓練邏輯 (Last Checkpoint 儲存)
├── models/
│   └── attention_unet.py  # 具備斷言保護的模型 (v2.6)
├── utils/
│   ├── dataset.py         # 快取共享與輕量化驗證實作
│   └── visualize.py       # Alpha Blending 視覺化
└── scripts/
    └── download_brats.py  # 強化版下載與衝突處理腳本
```
