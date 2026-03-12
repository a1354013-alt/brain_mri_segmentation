# Brain MRI Tumor Segmentation (v2.4 Final Gold)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與高效 I/O。

---

## 📋 專案更新亮點 (v2.4 Final)

- **下載清理邏輯一致化**：`download_brats.py` 現在統一使用 `is_patient_folder_complete` 進行判定。清理階段會嚴格檢查 **4 模態 + Seg**，確保不留任何會導致訓練跳過的殘餘不完整資料。
- **記憶體友善掃描**：`BraTSDataset` 初始化掃描時改用 `np.asarray(proxy)`。這能有效避免 `get_fdata()` 將整個 3D Volume 膨脹成 float64，大幅降低掃描階段的記憶體峰值。
- **推論自動挑選有效病人**：在 `infer` 模式下，若指定的 `patient_id` 無效或未指定，系統會自動迭代搜尋並挑選第一個有效病人進行推論，顯著提升使用者體驗。
- **快取開關控制**：`config.py` 新增 `USE_PROXY_CACHE` 開關。在 I/O 不穩定的環境（如網路磁碟）下可關閉快取，回退至穩定但較慢的逐次讀檔模式。
- **模型斷言保護**：`AttentionUNet` 的尺寸對齊邏輯後方加入了 `assert` 斷言。若對齊後尺寸仍不一致，會立即拋出具備可讀性的錯誤訊息，而非在後續運算中崩潰。
- **CUDA 檢查優化**：`set_seed` 現在會先檢查 `torch.cuda.is_available()`，避免在 CPU-only 環境下呼叫 CUDA 相關指令。

---

## 🏗️ 技術細節

### 1. 尺寸保護與斷言
模型在 Decoder 階段會自動對齊特徵圖尺寸：
- **Padding**：當上採樣尺寸不足時補齊。
- **Center Crop**：當上採樣尺寸多出時裁切。
- **Assertion**：對齊後執行 `assert x_skip.shape[2:] == x_up.shape[2:]`。

### 2. 資料完整性檢查
統一的檢查邏輯確保每個病人資料夾必須完整包含：
- `flair`, `t1`, `t1ce`, `t2` 模態
- `seg` 標籤檔案

---

## 🚀 執行指令

### 1. 資料準備
```bash
# 自動下載、解壓、對齊並清理不完整資料
python scripts/download_brats.py --auto
```

### 2. 訓練與推論
```bash
# 訓練 (支援記憶體友善掃描)
python main.py train

# 推論 (自動挑選有效病人)
python main.py infer --uncertainty entropy

# Demo (自動挑選有效病人進行快速流程測試)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 核心配置 (含 USE_PROXY_CACHE)
├── main.py                # CLI 入口 (含自動挑選有效病人邏輯)
├── train.py               # 訓練邏輯
├── models/
│   └── attention_unet.py  # 具備斷言保護的模型
├── utils/
│   ├── dataset.py         # 記憶體友善掃描與快取控制
│   └── visualize.py       # 視覺化
└── scripts/
    └── download_brats.py  # 統一完整性檢查的下載腳本
```
