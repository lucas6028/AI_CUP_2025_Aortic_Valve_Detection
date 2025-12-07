# AI CUP 2025 秋季賽 - 電腦斷層主動脈瓣物件偵測

**Team:** TEAM\_8472  

**Private Leaderboard:** 0.975520 (Rank 3)

**Public Leaderboard Best:** 0.976331 (Rank 1)

本專案使用 YOLOv8 模型進行主動脈瓣的物件偵測。針對醫學影像資料不平衡的特性，我們實作了 BalancedBatchGenerator、Hard Negative Mining 以及多種 Fine-tuning 策略。

-----

## 環境配置

本專案主要於 Google Colab (T4 GPU) 開發，亦支援 Windows Local 端。

### 依賴 dependencies

請確保安裝以下 Python dependencies：

```
pip install ultralytics pandas matplotlib seaborn scikit-learn gdown pyyaml opencv-python
```

**注意：** 若在 Google Colab 執行，程式碼包含掛載 Google Drive 的步驟，以便儲存 Checkpoints 和訓練結果。

-----

## 資料準備

程式會自動處理原始壓縮檔，執行 Stratified Split 與格式轉換。

1.  **原始輸入**：
      * `training_image.zip`
      * `training_label.zip`
2.  **處理流程**：
      * 解壓縮並遞迴搜尋 `patientXXXX` 資料夾。
      * **正負樣本分類**：檢查是否有 `.txt` 標註，無標註視為背景（負樣本。
      * **Stratified Split**：依照 70% / 15% / 15% 切分為 Train / Val / Test。
      * **Balanced Validation**：額外建立 `val_balanced` (50% 正 / 50% 負)，用於 Threshold Tuning。

-----

## 核心功能

以下說明專案中主要 Notebook 的功能、輸入與輸出，以利除錯與串接。

### 1\. 訓練 notebook (`train.ipynb`)

用於單一模型訓練與微調，可復現 Private Leaderboard 成績。

  * **輸入 (Input)**：原始資料壓縮檔 (`.zip`)。
  * **輸出 (Output)**：
      * `/content/drive/MyDrive/AI_CUP_2025/aortic_valve_checkpoints/finetune_frozen/weights/best.pt`：最佳權重檔。
      * `/content/drive/MyDrive/AI_CUP_2025/aortic_valve_checkpoints/finetune_frozen/weights/last.pt`：最後權重檔 (支援 Resume)。
      * `hard_negatives.json`：挖掘出的困難負樣本資訊。
      * 圖表：PR 曲線、F1 Score 曲線、Confusion Matrix。
  * **關鍵功能**：
      * **Hard Negative Mining**：找出高信心的誤報 (FP) 樣本。
      * **Frozen Layers Fine-Tuning**：凍結骨幹層進行微調。

### 2\. 預測模塊 (`predict.ipynb`)

用於生成最終提交檔案。

  * **輸入 (Input)**：訓練好的 `best.pt` (從 Drive 讀取)。
  * **輸出 (Output)**：`merged.txt` (符合比賽格式的預測結果)。
  * **關鍵參數**：
      * `conf=0.001`：極低閾值以最佳化 AP@0.5。
      * `iou=0.5`。
      * `max_det=20`。

-----

## 訓練與重現結果

### 方案 A：復現 Private Score (Rank 3)

這是最穩定且結構完整的方案，包含完整的資料前處理與 Fine-tuning 流程。

1.  開啟 `train.ipynb`。
2.  確保 Google Drive 已掛載且資料路徑正確。
3.  執行所有 Cells。程式將自動執行：
      * 資料解壓縮與清洗。
      * YOLOv8m 基礎訓練 (60 Epochs)。
      * 產生 `val_balanced` 驗證報告。
      * 執行 Frozen Layer Fine-tuning。
4.  訓練完成後，使用 `predict.ipynb` 載入產生的 `best.pt` 進行預測。

### 方案 B：復現 Public Score (Rank 1)

1.  執行 `train_kfold.ipynb` 取得 YOLOv8 5 folds 模型。
2.  執行 `train_faster_rcnn.ipynb` 取得 Faster R-CNN 模型 。
3.  執行 `predict_hybrid_wbf_optimized.ipynb` 進行 WBF 融合推論。

> **Debug 提示：**
> 若訓練中斷，請檢查 `create_optimized_config()` 函數。若要從 `last.pt` 繼續訓練，請確保 `'resume': True` 欄位已被取消註解 (Uncomment)。

-----

## 超參數設定 (Hyperparameters)

主要模型配置 (`yolov8m.pt`) 如下 ：

| 參數類別 | 設定值 | 說明 |
| :--- | :--- | :--- |
| **Training** | Epochs: 60 | 包含 patience=20 early stop |
| | Batch: 16 |  |
| | Image Size: 640 | |
| **Optimizer** | AdamW | lr0=0.001, lrf=0.01, momentum=0.937 |
| **Loss Gains** | Box: 7.5, Cls: 1.5, Dfl: 1.5 | 強化不平衡下的定位與分類 |
| **Augmentation** | Mosaic: 0.8, Mixup: 0.15  Copy-Paste: 0.3 | 增加多樣性 |
| **Inference** | Conf Threshold: 0.001 | Grid search 後的最佳 AP@0.5 設定 |
| | Max Det: 20 | |

-----

## 引用與致謝

本專案參考了 TotalSegmentator、CFUN 等開源項目以及 Ultralytics YOLOv8 框架。詳細引用列表請參閱原始報告。