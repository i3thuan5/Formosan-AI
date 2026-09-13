## Context

TTS 系統（`tts/app.py`）目前使用 `gr.Dropdown` 作為族別、配音員、語別的選擇元件。ASR 系統（`asr-kaldi/app.py`）已使用 `gr.Radio`。本次改動將 TTS 的 Dropdown 統一為 Radio，使兩個系統的 UI 一致。

現況中 `tts/app.py` 共有 4 個 `gr.Dropdown` 和 2 處 `.change()` callback 回傳 `gr.Dropdown(...)`：

```
「預設配音員」Tab                    「自己當配音員」Tab
┌───────────────────────┐         ┌───────────────────────┐
│ default_speaker_      │         │ custom_speaker_       │
│   ethnicity           │         │   ethnicity           │
│   (Dropdown→Radio)    │         │   (Dropdown→Radio)    │
│         │ .change()   │         │         │ .change()   │
│         ▼             │         │         ▼             │
│ default_speaker_      │         │ custom_speaker_       │
│   refs                │         │   language            │
│   (Dropdown→Radio)    │         │   (Dropdown→Radio)    │
└───────────────────────┘         └───────────────────────┘
```

## Goals / Non-Goals

**Goals:**

- 將 TTS 的 4 個 `gr.Dropdown` 元件替換為 `gr.Radio`
- 將 2 處 `.change()` callback 回傳值從 `gr.Dropdown(...)` 改為 `gr.Radio(...)`
- 移除 `gr.Radio` 不支援的參數（`filterable`）
- 維持與 ASR 系統一致的 UI 風格

**Non-Goals:**

- 不變更選項的資料來源（ETHNICITIES、refs_config、g2p_object）
- 不調整版面佈局
- 不修改其他系統（ASR、MT）

## Decisions

### 1. 直接替換 `gr.Dropdown` → `gr.Radio`，不引入額外抽象

**選擇**：逐一將 `gr.Dropdown` 替換為 `gr.Radio`，保持現有程式結構。

**理由**：改動僅涉及 Gradio 元件類型替換，邏輯完全相同（choices、value、label）。`gr.Radio` 和 `gr.Dropdown` 的 API 高度相容，不需要重構。

**替代方案**：抽出共用的 selector factory function → 過度工程，此處不需要。

### 2. 移除 `filterable=False` 參數

**選擇**：直接刪除 `filterable=False`。

**理由**：`gr.Radio` 不支援 `filterable` 參數。此參數在 Dropdown 中的用途是停用搜尋過濾，Radio 本身沒有搜尋功能，因此直接移除即可。

### 3. 保留 `show_label=False`

**選擇**：`custom_speaker_language` 的 `show_label=False` 參數保留不變。

**理由**：`gr.Radio` 支援 `show_label` 參數，行為與 Dropdown 一致。

### 4. 保留 `visible` 動態控制邏輯

**選擇**：`custom_speaker_ethnicity.change()` callback 中的 `visible=len(...) > 1` 邏輯保留。

**理由**：`gr.Radio` 支援 `visible` 參數。當某族別只有一個語別時隱藏選擇器的邏輯仍然合理。

## Risks / Trade-offs

**版面空間增加** → Radio 選項全部展開顯示，會比 Dropdown 佔更多垂直空間。族別約 12 個、配音員每族 1-6 個，數量在合理範圍內，使用者可一眼看到所有選項。

**無 migration 需求** → 純前端 UI 元件替換，無資料格式或 API 變更，不需要 rollback 策略。
