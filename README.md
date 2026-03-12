# Brain MRI Tumor Segmentation (v2.3 Final)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與高效 I/O。

---

## 📋 專案更新亮點 (v2.3)

- **模型尺寸雙重保護**：修正了 `_align_and_concat` 中的負 Padding 風險。現在模型能自動判斷特徵圖尺寸，若 `x_up` 較小則 Padding，若較大則執行 **Center Crop**，確保在任何輸入尺寸下都能穩定運行。
- **多 Worker Seed 強化**：`worker_init_fn` 現在同步固定 `torch.manual_seed`，徹底解決多線程資料加載時的隨機性重複問題。
- **I/O 效能優化**：`BraTSDataset` 實作了 **nibabel dataobj proxy 快取**。僅在初始化時讀取檔案指針，`__getitem__` 時直接從 Proxy 讀取特定切片，大幅減少重複開啟檔案的開銷。
- **路徑一致性強化**：`Trainer` 現在完全由外部傳入 `log_file` 與 `tensorboard_dir`，確保 Demo 模式與正式訓練的輸出路徑嚴格隔離且符合 `config.py` 定義。
- **掃描日誌優化**：資料掃描時不再洗版，僅列印前 10 筆錯誤，完整清單自動儲存至 `outputs/skipped_patients.txt`。
- **結構對齊加固**：`download_brats.py` 採用更嚴格的移動策略，僅在確認資料夾包含完整 4 模態 + Seg 後才視為有效病人並進行對齊。

---

## 🏗️ 技術細節

### 1. 不確定性指標說明
- **Variance (變異數)**：衡量多次 MC Dropout 預測值的離散程度。
- **Entropy (預測熵)**：計算 **Predictive Entropy**（基於平均預測機率 $p$）。公式為：$-p \log(p) - (1-p) \log(1-p)$。這能反映模型對分類結果的混亂程度。

### 2. 資料完整性
本專案要求每個病人資料夾必須包含：
- `*_flair.nii.gz`, `*_t1.nii.gz`, `*_t1ce.nii.gz`, `*_t2.nii.gz`
- `*_seg.nii.gz`

### 3. 尺寸保護邏輯
在 Decoder 階段，若 `ConvTranspose2d` 產生的尺寸與 Encoder 的 Skip Connection 尺寸不符（常見於輸入尺寸非 16 的倍數時），模型會自動執行對齊：
- **Padding**：補齊缺失像素。
- **Center Crop**：裁切多餘像素。

---

## 🚀 執行指令

### 1. 資料準備
```bash
python scripts/download_brats.py --auto
```

### 2. 訓練與推論
```bash
# 訓練
python main.py train

# 推論 (支援 config.MC_ITERATIONS)
python main.py infer --uncertainty entropy

# Demo (使用 config.DEMO_MC_ITERATIONS，輸出隔離)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 核心配置 (含 THRESHOLD, MC_ITERATIONS)
├── main.py                # CLI 入口 (含 Worker Seed 修正)
├── train.py               # 訓練邏輯 (路徑完全 Config 化)
├── models/
│   └── attention_unet.py  # 具備 Padding/Crop 雙重保護的模型
├── utils/
│   ├── dataset.py         # Proxy 快取與日誌優化
│   └── visualize.py       # 視覺化與 MC Dropout 邏輯簡化
└── scripts/
    └── download_brats.py  # 加固的結構對齊腳本
```
