# Evaluating Logit-Based GOP 專案分析

## 專案概述

這是一個用於自動發音錯誤檢測的Goodness of Pronunciation (GOP)評分系統實現。該專案實現了基於CTC神經網路的**Phoneme-Perturbed Alignment-Free GOP (PP-AF GOP)**方法，用於評估語音發音質量。

## 專案架構圖

```
Evaluating-Logit-Based-GOP/
├── main.py                 # 主執行腳本 - 程式入口點
├── gop_calculators.py      # 核心GOP計算算法
├── phoneme_map.py          # 音素混淆模式映射
├── data_loader.py          # 數據加載工具
├── evaluation.py           # 性能評估指標
├── requirements.txt        # Python依賴包
└── refData/               # 參考數據目錄
    └── codes/
        └── gop-pykaldi-master (1).zip
```

## 文件功能詳細分析

### 1. main.py - 主執行文件

**功能**: 程式的主要入口點，協調整個GOP計算流程

**主要函數**:
- `main()`: 主執行函數

**執行流程**:
1. 初始化GPU/CPU設備
2. 載入預訓練模型和處理器
3. 載入模擬數據
4. 對每個樣本計算GOP分數（RPS和UPS兩種模式）
5. 輸出結果

**依賴關係**:
- 導入 `gop_calculators.py` 中的 `init_model_and_processor`, `calculate_pp_af_gop`
- 導入 `data_loader.py` 中的 `load_mock_data`

### 2. gop_calculators.py - 核心GOP計算模組

**功能**: 實現PP-AF GOP算法的核心計算邏輯

**主要函數**:

#### `init_model_and_processor(model_name: str)`
- **功能**: 初始化並返回Wav2Vec2模型和處理器
- **參數**: model_name - 預訓練模型名稱
- **返回**: (model, processor) - 模型和處理器對象
- **依賴**: transformers庫

#### `get_ctc_loss(logits, labels, processor, device)`
- **功能**: 計算CTC損失的輔助函數
- **參數**: 
  - logits: 模型輸出的logits
  - labels: 標籤序列
  - processor: 處理器對象
  - device: 計算設備
- **返回**: CTC損失值
- **依賴**: torch.nn.CTCLoss

#### `calculate_pp_af_gop(model, processor, audio_path, canonical_transcript, mode='RPS')`
- **功能**: 計算Phoneme-Perturbed Alignment-Free GOP分數
- **參數**:
  - model: Wav2Vec2 CTC模型
  - processor: Wav2Vec2處理器
  - audio_path: 音頻文件路徑
  - canonical_transcript: 標準文本轉錄
  - mode: 'RPS'（受限音素替換）或'UPS'（非受限音素替換）
- **返回**: (original_phonemes, phoneme_gops) - 原始音素序列和GOP分數
- **算法步驟**:
  1. 載入音頻並獲取logits
  2. 獲取原始音素序列
  3. 計算原始CTC損失
  4. 對每個音素進行擾動（刪除和替換）
  5. 計算最小擾動損失
  6. 計算GOP分數 = min(L_perturbed) - L_original

**依賴關係**:
- 導入 `phoneme_map.py` 中的 `get_restricted_substitutions`
- 使用 librosa 進行音頻處理
- 使用 phonemizer 進行文本到音素轉換

### 3. phoneme_map.py - 音素混淆映射

**功能**: 定義基於語言學知識的音素混淆模式

**主要函數**:

#### `get_restricted_substitutions(phoneme: str) -> list`
- **功能**: 返回給定音素的允許替換列表
- **參數**: phoneme - 目標音素
- **返回**: 替換音素列表
- **依賴**: PHONEME_CONFUSION_MAP 字典

**數據結構**:
- `PHONEME_CONFUSION_MAP`: 包含常見L2學習者錯誤、語音相似性和音韻規則的字典

**混淆模式類型**:
1. **L2學習者常見錯誤**: θ→s/t/f, ð→d/z/v, æ→ɛ/e/a
2. **語音相似性**: p↔b, t↔d, k↔g
3. **元音合併**: eɪ→e/ɛ, oʊ→o/ɔ
4. **其他常見混淆**: l↔ɹ, v↔w

### 4. data_loader.py - 數據加載模組

**功能**: 生成和管理模擬音頻數據

**主要函數**:

#### `load_mock_data(num_samples=2, data_dir=".mock_data")`
- **功能**: 生成合成音頻文件並返回模擬數據樣本
- **參數**:
  - num_samples: 樣本數量
  - data_dir: 數據目錄
- **返回**: 包含音頻路徑、轉錄和標籤的數據列表
- **功能詳情**:
  1. 創建數據目錄
  2. 定義模擬樣本（正確和錯誤發音）
  3. 生成正弦波作為模擬音頻
  4. 保存音頻文件
  5. 返回結構化數據

**依賴關係**:
- 使用 numpy 生成音頻數據
- 使用 soundfile 保存音頻文件

### 5. evaluation.py - 評估模組

**功能**: 提供性能評估指標計算

**主要函數**:

#### `calculate_metrics(y_true, gop_scores, threshold=0)`
- **功能**: 基於GOP分數計算分類指標
- **參數**:
  - y_true: 真實標籤列表（1=正確，0=錯誤發音）
  - gop_scores: 計算的GOP分數
  - threshold: 分類閾值
- **返回**: 包含各種指標的字典
- **計算指標**:
  - accuracy: 準確率
  - precision: 精確率
  - recall: 召回率
  - f1_score: F1分數
  - mcc: Matthews相關係數

**依賴關係**:
- 使用 scikit-learn 的評估指標函數

## 函數調用關係圖

```
main.py
├── main()
    ├── init_model_and_processor() [來自 gop_calculators.py]
    ├── load_mock_data() [來自 data_loader.py]
    └── calculate_pp_af_gop() [來自 gop_calculators.py]
        ├── get_ctc_loss() [來自 gop_calculators.py]
        └── get_restricted_substitutions() [來自 phoneme_map.py]

evaluation.py
└── calculate_metrics() [獨立函數，可被其他模組調用]
```

## 數據流圖

```
音頻文件 → data_loader.py → main.py → gop_calculators.py
                                    ↓
文本轉錄 → phonemizer → gop_calculators.py → GOP分數
                                    ↓
音素映射 → phoneme_map.py → gop_calculators.py
                                    ↓
評估指標 ← evaluation.py ← GOP分數
```

## 關鍵依賴關係

### 外部庫依賴
- **torch**: 深度學習框架
- **transformers**: Hugging Face模型
- **phonemizer**: 文本到音素轉換
- **librosa**: 音頻處理
- **scikit-learn**: 機器學習評估
- **soundfile**: 音頻文件I/O
- **numpy**: 數值計算

### 內部模組依賴
- `main.py` → `gop_calculators.py`, `data_loader.py`
- `gop_calculators.py` → `phoneme_map.py`
- 所有模組都可以使用 `evaluation.py` 進行評估

## 算法核心概念

### PP-AF GOP算法
1. **對齊自由**: 不需要強制對齊音素和音頻幀
2. **音素擾動**: 通過刪除和替換音素創建擾動序列
3. **CTC損失比較**: 比較原始序列和擾動序列的CTC損失
4. **GOP計算**: GOP = min(L_perturbed) - L_original

### 兩種模式
- **RPS (Restricted Phoneme Substitutions)**: 使用預定義的音素混淆映射
- **UPS (Unrestricted Phoneme Substitutions)**: 使用整個詞彙表進行替換

## 使用場景

1. **語言學習應用**: 評估非母語者的發音質量
2. **語音識別系統**: 提高識別準確率
3. **語音治療**: 輔助發音障礙診斷
4. **研究用途**: 語音學和計算語言學研究

## 擴展性

該專案具有良好的模組化設計，可以輕鬆擴展：
- 添加新的音素混淆模式
- 實現不同的GOP計算方法
- 集成其他音頻處理模型
- 添加更多評估指標 