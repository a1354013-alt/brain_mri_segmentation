# Brain MRI Tumor Segmentation (v3.0 Final Release Gold Master)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計、極致記憶體優化與工業級魯棒性的資料處理流程。

---

## 📋 專案更新亮點 (v3.0 Final)

- **v3.0: 效能與穩定性打磨**：移出了 Hot Path 中的 `skimage.transform.resize` import，強化了 `prepared_cache` 的錯誤處理（改為 `raise ValueError`），並新增了核心流程的 **冒煙測試 (Smoke Test)**。
- **v2.9: Proxy Cache 安全性修復**：修復了 `proxy_cache` 可能因快取資料缺失而塞入 `None` 導致推論崩潰的隱患。
- **v2.8: 極致記憶體優化 (Extreme Memory Optimization)**：實作了「真正逐切片掃描」機制。在資料集準備階段，不再將整個 3D Volume 載入記憶體，而是逐切片讀取並計算腫瘤像素，將 RAM 佔用降至最低。
- **v2.8: 快取安全子集化 (Safe Cache Subsetting)**：強化了快取共享邏輯，加入 PID 存在性檢查與防呆機制，確保訓練集與驗證集在共享快取時能真正分離且不崩潰。
- **v2.7: 修正快取共享子集化錯誤**：修復了舊版中驗證集會繼承訓練集所有 ID 的重大邏輯錯誤。

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

### 2. 冒煙測試
```bash
# 驗證核心流程 (Dataset, Model, Inference) 是否正常
python tests/smoke_test.py
```

### 3. 訓練與推論
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
├── config.py              # 核心配置 (v3.0 Final)
├── main.py                # CLI 入口 (含快取共享子集化)
├── train.py               # 訓練邏輯 (Last Checkpoint 儲存)
├── models/
│   └── attention_unet.py  # 具備斷言保護的模型 (v3.0)
├── utils/
│   ├── dataset.py         # 極致記憶體優化與防呆實作
│   └── visualize.py       # Alpha Blending 視覺化
├── scripts/
│   └── download_brats.py  # 強化版下載與衝突處理腳本
└── tests/
    └── smoke_test.py      # 核心流程冒煙測試 (v3.0)
```
