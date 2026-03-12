# Brain MRI Tumor Segmentation (v2.5 Final Master)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與極致優化的 I/O。

---

## 📋 專案更新亮點 (v2.5 Final)

- **逐切片掃描 (Slice-by-Slice Scanning)**：`BraTSDataset` 初始化時不再一次性讀取整個 3D Volume，而是逐切片計算腫瘤像素。這將掃描階段的記憶體佔用降至最低，即使在處理數百名病人時也能保持極速且穩定。
- **AMP CPU 安全性修正**：`Trainer` 現在會自動偵測裝置類型。僅在 `cuda` 環境下啟用 **AMP (混合精度訓練)**，避免在 CPU 環境下產生無效的警告或效能損耗。
- **Alpha Blending 視覺化**：推論結果的 Overlay 疊加圖現在支援 **Alpha Blending (透明度混合)**。這讓使用者在看到紅色預測區域的同時，仍能清晰觀察到原圖的 MRI 紋理細節。
- **下載腳本加固**：`download_brats.py` 解決了搬移撞名與重複計數的邊界問題。若未找到任何有效資料，系統會拋出 `RuntimeError`，對自動化流程更友善。
- **依賴清理**：移除了 `requirements.txt` 中未使用的 `torchvision` 依賴，保持環境純淨。
- **曲線繪製品質提升**：訓練曲線圖現在包含完整的座標軸標籤、緊湊佈局 (Tight Layout) 與高解析度 (150 DPI) 輸出。

---

## 🏗️ 技術細節

### 1. 記憶體優化
透過 `np.asarray(proxy.dataobj[:, :, s])` 僅讀取單一 2D 切片，避免了 NIfTI 檔案在轉換為 ndarray 時產生的巨大記憶體開銷。

### 2. 視覺化展示
- **Uncertainty Map**：支援 Variance 與 Entropy 兩種指標。
- **Overlay**：使用 `(1 - alpha) * image + alpha * mask` 進行混合，預設 `alpha=0.35`。

---

## 🚀 執行指令

### 1. 資料準備
```bash
# 自動下載、解壓、對齊並清理不完整資料 (具備錯誤拋出機制)
python scripts/download_brats.py --auto
```

### 2. 訓練與推論
```bash
# 訓練 (自動偵測 AMP 支援)
python main.py train

# 推論 (支援 Alpha Blending 疊加圖)
python main.py infer --uncertainty entropy

# Demo (快速驗證完整流程)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 核心配置 (含 OVERLAY_ALPHA)
├── main.py                # CLI 入口 (含 AMP 自動偵測)
├── train.py               # 訓練邏輯 (AMP CPU 安全修正)
├── models/
│   └── attention_unet.py  # 具備斷言保護的模型
├── utils/
│   ├── dataset.py         # 逐切片掃描優化
│   └── visualize.py       # Alpha Blending 視覺化
└── scripts/
    └── download_brats.py  # 加固的下載與清理腳本
```
