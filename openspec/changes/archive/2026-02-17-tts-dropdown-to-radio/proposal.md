## Why

TTS（語音合成）系統的族別、配音員、語別選擇目前使用 `gr.Dropdown`，而 ASR（語音辨識）系統已使用 `gr.Radio`。兩個系統的使用者介面不一致，且 Radio 元件讓使用者一眼即可看到所有選項，操作體驗更好。統一改為 Radio 以維持跨服務的 UI 一致性。

## What Changes

- 將 TTS `tts/app.py` 中 4 處 `gr.Dropdown` 元件改為 `gr.Radio`：
  - `default_speaker_ethnicity`（預設配音員 - 族別選擇）
  - `default_speaker_refs`（預設配音員 - 配音員選擇，依族別動態更新）
  - `custom_speaker_ethnicity`（自訂配音員 - 族別選擇）
  - `custom_speaker_language`（自訂配音員 - 語別選擇，依族別動態更新）
- 將 2 處 `.change()` callback 中回傳的 `gr.Dropdown(...)` 改為 `gr.Radio(...)`
- 移除 `filterable=False` 參數（`gr.Radio` 不支援此屬性）

## Non-goals

- 不更動 ASR 系統（已是 Radio）
- 不更動 MT 系統的 UI 元件
- 不調整選項的資料來源邏輯（ETHNICITIES、refs_config、g2p_object）
- 不變更版面佈局（Row/Column 結構）

## Capabilities

### New Capabilities

- `tts-language-selector`: TTS 語音合成系統的族別、配音員、語別選擇元件規格

### Modified Capabilities

（目前無既有 spec）

## Impact

- **影響模組**：僅 `tts/`
- **影響檔案**：`tts/app.py`（6 處修改）
- **API**：無影響（FastAPI endpoint 不變）
- **Dependencies**：無新增套件（`gr.Radio` 已包含在 Gradio 中）
- **向下相容**：無 breaking change，所有族語方言的選項資料不變
