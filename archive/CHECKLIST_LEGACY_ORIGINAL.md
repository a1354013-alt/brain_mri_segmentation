# 專案品質檢查清單

本文件記錄所有規格要求的完成狀態。

---

## ✅ 一、修正現有錯誤與技術債

- [x] 修正 train.py 中 val_dice 計算邏輯（改為 per-sample dice 再 batch 平均）
- [x] 修正 AttentionUNet 中重複定義的 up4 / att4 / up_conv4（第 74-76 行已刪除）
- [x] Dice 計算改為 per-sample dice 再 batch 平均（`dice_coeff_per_sample` 函數）
- [x] MC Dropout 推論改為：
  - model.eval()
  - 只將 Dropout layer 設為 train()（`enable_dropout` 方法）
  - 不污染 BatchNorm
- [x] dataset.py 不再寫死 128，改為 `image_size` 參數
- [x] 移除 main.py 中 dummy data fallback
- [x] 增加 random seed 固定（`config.set_seed()` 函數）

---

## ✅ 二、改為 CLI 架構

- [x] main.py 改為 CLI 架構：
  - `python main.py train` - 訓練模式
  - `python main.py infer` - 推論模式
  - `python main.py demo` - Demo 模式
- [x] train 命令：
  - 載入資料
  - 訓練
  - 儲存 best_model.pth
  - 輸出 log
- [x] infer 命令：
  - 載入模型
  - 對單一病人做推論
  - 存 segmentation.png
  - 存 uncertainty.png（整合在 segmentation.png 中）
- [x] demo 命令：
  - 使用少量資料跑 1 epoch 測試流程

---

## ✅ 三、自動下載資料集

- [x] 新增 `scripts/download_brats.py`
- [x] 功能：
  - 檢查 data/Brats 是否存在
  - 提示使用者到官方 BraTS 網站下載
  - 提供 kaggle API 下載範例
  - 下載後自動解壓縮至 data/Brats/
- [x] README 中加入：
  - 如何取得 BraTS
  - 資料結構說明
  - Kaggle API 下載步驟

---

## ✅ 四、升級 Dataset

- [x] 不再只取中間切片
- [x] 改為自動選擇「含腫瘤的 slice」（`_select_slice` 方法）
- [x] 加入 z-score normalization（`_normalize_image` 方法）
- [x] 加入 percentile clip (1%-99%)（`_normalize_image` 方法）

---

## ✅ 五、訓練強化

- [x] 加入 AMP 混合精度（`GradScaler` 和 `autocast`）
- [x] 加入 ReduceLROnPlateau
- [x] 加入 gradient clipping
- [x] 加入 model checkpoint
- [x] 加入訓練記錄 CSV
- [x] 加入 tensorboard logging
- [x] 儲存：
  - outputs/best_model.pth
  - outputs/training_log.csv
  - outputs/loss_curve.png
  - outputs/tensorboard/

---

## ✅ 六、視覺化升級

- [x] visualize.py 改為輸出：
  - 原圖
  - GT
  - Prediction
  - Uncertainty heatmap
  - Overlay (原圖+預測)
- [x] colormap 改為 viridis（不使用 jet）

---

## ✅ 七、專案結構重整

- [x] 專案結構符合要求：
  ```
  brain_mri_segmentation/
  ├── data/
  ├── models/
  ├── utils/
  ├── scripts/
  ├── outputs/
  ├── main.py
  ├── train.py
  ├── config.py
  ├── requirements.txt
  └── README.md
  ```

---

## ✅ 八、README 必須包含

- [x] 專案簡介
- [x] 模型架構圖
- [x] 訓練方式
- [x] 推論方式
- [x] 範例結果圖
- [x] MC Dropout 不確定性說明
- [x] 未來改進方向

---

## ✅ 九、程式品質要求

- [x] 不可留下未使用 import（已檢查）
- [x] 不可有死碼（已移除）
- [x] 所有 magic number 必須改為 config（已完成）
- [x] 變數命名清晰（已確認）
- [x] 加入 type hint（所有函數都有 type hint）

---

## 📝 額外完成項目

- [x] 加入 .gitignore
- [x] 加入 LICENSE (MIT)
- [x] 加入 __init__.py 到所有模組
- [x] 所有程式碼通過語法檢查
- [x] 完整的 docstring 文件
- [x] 清晰的錯誤訊息與使用提示

---

## ✅ 所有規格已完成！
