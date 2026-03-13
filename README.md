# Brain MRI Tumor Segmentation (v2.8 Final Gold Master Corrected)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與極致優化的 I/O 與快取機制。

---

## 📋 專案更新亮點 (v2.8 Final)

- **v2.8: 極致記憶體優化 (Extreme Memory Optimization)**：實作了「真正逐切片掃描」機制。在資料集準備階段，不再將整個 3D Volume 載入記憶體，而是逐切片讀取並計算腫瘤像素，將 RAM 佔用降至最低。
- **v2.8: 快取安全子集化 (Safe Cache Subsetting)**：強化了快取共享邏輯，加入 PID 存在性檢查與防呆機制，確保訓練集與驗證集在共享快取時能真正分離且不崩潰。
- **v2.8: 日誌路徑優化**：Dataset 的缺失日誌現在會根據執行模式（Demo 或正式訓練）自動寫入對應的輸出資料夾，避免日誌混淆。
- **v2.7: 修正快取共享子集化錯誤**：修復了舊版中驗證集會繼承訓練集所有 ID 的重大邏輯錯誤。
- **快取共享機制 (Shared Cache)**：`BraTSDataset` 支援傳入外部 `prepared_cache`。在訓練啟動時，訓練集與驗證集會共享掃描結果，大幅縮短啟動時間。
- **輕量化驗證 (Lightweight Validation)**：在推論與 Demo 模式下，系統僅檢查檔案存在性，實現秒級啟動。
- **Last Checkpoint 雙重保險**：`Trainer` 現在除了儲存 `best_checkpoint.pth` 外，每一輪都會更新 `last_checkpoint.pth`。
- **Kaggle 下載魯棒性強化**：`download_brats.py` 自動偵測最新下載的 zip 檔案，並處理搬移衝突。
- **視覺化升級**：Overlay 疊加圖支援透明度混合 (`alpha=0.35`)，保留 MRI 紋理細節。

---

## 🏗️ 技術細節

### 1. 極致記憶體優化
透過 `np.asarray(mask_proxy.dataobj[:, :, i])` 逐切片讀取 NIfTI 檔案，避免了 `get_fdata()` 造成的記憶體膨脹，確保在處理大規模資料集時依然穩定。

### 2. 快取共享子集化
透過 `BraTSDataset(..., prepared_cache=shared_cache)` 初始化時，Dataset 會遍歷 `patient_ids` 並僅從 `shared_cache` 中提取存在的 Metadata，實現高效且正確的資料切分。

### 3. 尺寸對齊保護
模型內建 `_align_and_concat` 邏輯，自動處理 Padding 與 Center Crop，確保 Encoder 與 Decoder 特徵圖完美對齊。

---

## 🚀 執行指令

### 1. 資料準備
```bash
# 自動下載、解壓、對齊並處理衝突
python scripts/download_brats.py --auto
```

### 2. 訓練與推論
```bash
# 訓練 (驗證集與訓練集真正分離)
python main.py train

# 推論 (輕量化驗證，秒級啟動)
python main.py infer --uncertainty entropy

# Demo (路徑統一管理)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 核心配置 (v2.8 Final)
├── main.py                # CLI 入口 (含快取共享子集化)
├── train.py               # 訓練邏輯 (Last Checkpoint 儲存)
├── models/
│   └── attention_unet.py  # 具備斷言保護的模型 (v2.8)
├── utils/
│   ├── dataset.py         # 極致記憶體優化與防呆實作
│   └── visualize.py       # Alpha Blending 視覺化
└── scripts/
    └── download_brats.py  # 強化版下載與衝突處理腳本
```
