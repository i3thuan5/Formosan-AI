## Why

族語基礎翻譯系統（MT）「華語 ⮕ 族語」翻譯出族語文字後，使用者不知道怎麼念，必須另開語音合成系統（TTS）網頁、重新選族別配音員、貼上文字才能聽到發音。讓使用者在同一頁按一下就能聽到譯文，可以把「機器翻譯 → 人工修正 → 聽發音」串成一個流程。

探索階段對正式站實測，發現兩個阻礙：

1. TTS 現有 API `/default_speaker_tts` 的 `ref` 參數是 `gr.Radio`，choices 在頁面建立時固定為「阿美」的 6 位配音員。從外部直接呼叫時，其他 36 個語別的配音員都會被擋（`Value ... is not in the list of choices`）。
2. TTS 的文字前處理有 bug：譯文以引號結尾時（例如 `..., "Ano dafak micodad kita."`），會先補句點再刪引號，產生 `..`，導致 `Unknown characters: .`。華語原文有「」時，MT 就會產生這種輸出。以 5 句華語 × 42 語別測試，210 句中有 18 句因此失敗。

## What Changes

- **tts**：新增以「語別」為參數的合成 API `/synthesize(language, text)`，自動使用該語別的第一位配音員，不受 Radio choices 限制。
- **tts**：修正文字前處理順序，先移除引號（`"` `“` `”` `「` `」`）再判斷是否補句點。「預設配音員」與「自己當配音員」兩個 Tab 共用此修正。
- **mt**：「華語 ⮕ 族語」Tab 在翻譯結果下方新增「合成語音」按鈕與音檔播放元件。按下時以翻譯結果 Textbox **當下的內容**（含使用者修改）呼叫 TTS `/synthesize`。
- **mt**：透過 `gradio_client` 呼叫 `https://{SAPOLITA_WEBSITE_HOST}/hnang-kari-ai-asi-sluhay/`。合成等待逾時 20 秒時顯示「現在使用人數眾多，請稍候再試」。
- **mt**：為會被測試腳本呼叫的事件設定固定 `api_name`，避免依賴 Gradio 自動產生的 `/lambda_N` 名稱。
- **新增 `production_tests/`**：上線後的檢查腳本。可用 `BASE_URL` 指定正式機或測試機，檢查 TTS API 約定、42 語別合成、已知 bug 回歸，以及 mt → tts 端到端（5 個語別，含預設阿美）。

## 非目標

- 「族語 ⮕ 華語」Tab 不加語音合成（輸出為華語，TTS 不支援）。
- 翻譯（MT 推論）不設 timeout。
- 不提供配音員選擇，固定使用該語別第一位配音員。
- 不從使用者 request 的 Host 推導 TTS 網址（避免 SSRF 與 mt 子網域無 tts 路徑的問題）。
- 不移除或改變 TTS 既有的 `/default_speaker_tts` 等 API（[grafana-k6/tts.js](../../../grafana-k6/tts.js) 延遲測試仍在使用）。
- 不改 TTS 網頁 UI 版面，也不改 mt 的族別、語別選擇元件。
- 不處理 MT 翻譯排隊的等待時間。

## Capabilities

### New Capabilities

- `tts-synthesize-api`：TTS 以語別為參數的合成 API，以及合成文字的引號前處理規則。
- `mt-speech-playback`：MT「華語 ⮕ 族語」Tab 的合成語音按鈕、呼叫 TTS 的方式、逾時與錯誤訊息。
- `production-tests`：針對已部署主機（正式機或測試機）的上線後檢查腳本。

### Modified Capabilities

（無。既有 `tts-language-selector` 規格的 Radio 行為不變。）

## Impact

- **影響模組**：`tts/`、`mt/`；新增 `production_tests/`。不影響 `asr/`、`asr-kaldi/`、`common/`、`deploy/`。
- **影響檔案**：`tts/app.py`、`mt/app.py`、`mt/requirements.in`、`mt/requirements.txt`，以及新增的 `production_tests/`。
- **API**：
  - tts 新增 `/synthesize`，既有 API 不變。
  - mt 新增合成語音 endpoint，並為族別切換事件設定固定 `api_name`。原本自動產生的 `/lambda`、`/lambda_1` 名稱會改變，但這兩個是 UI 內部事件，沒有已知的外部使用者。
- **Dependencies**：mt 直接 import `gradio_client`（已隨 gradio 安裝），在 `requirements.in` 明列並重新 pip-compile。
- **部署順序**：tts 必須先上線，mt 才能上線，否則 mt 的合成按鈕會失敗。CD 目前停用，需人工依序發佈。
- **網路**：mt 容器需能連到自己的公開網域 `SAPOLITA_WEBSITE_HOST`（hairpin NAT），上線後由 `production_tests/` 的端到端檢查驗證。
- **向下相容**：42 個語別在 `tts/configs/refs.yaml` 都至少有一位配音員，所有 MT 目標語別都可合成。引號修正只移除原本就會被 `text_to_ipa` 刪除的引號字元，不影響其他文字的合成結果。
