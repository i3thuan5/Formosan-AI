# MT 合成語音播放

## Purpose

MT 族語基礎翻譯系統「華語 ⮕ 族語」Tab 在翻譯後，於同一頁呼叫 TTS 合成並播放譯文語音，包含呼叫方式、逾時與錯誤訊息。

## Requirements

### Requirement: 華語 ⮕ 族語 Tab 提供合成語音按鈕與播放元件

MT「華語 ⮕ 族語」Tab SHALL 在「翻譯結果」下方提供「合成語音」按鈕，以及顯示合成結果的 `gr.Audio` 元件（可播放、可下載，不顯示分享鈕）。「族語 ⮕ 華語」Tab MUST NOT 提供合成語音功能。

#### Scenario: 翻譯後合成播放

- **WHEN** 使用者選擇 `阿美_海岸`，輸入「好」並按「翻譯」，接著按「合成語音」
- **THEN** 合成結果元件出現 `ngaʼay ho` 的語音，可以播放

#### Scenario: 族語 ⮕ 華語 Tab 沒有合成功能

- **WHEN** 使用者進入「族語 ⮕ 華語」Tab
- **THEN** 頁面上沒有「合成語音」按鈕與合成結果元件

### Requirement: 以翻譯結果當下的內容合成

按下「合成語音」時，MT SHALL 以「翻譯結果」Textbox 當下的文字（含使用者手動修改），搭配「語別」Radio 當下選擇的語別合成，而不是最近一次翻譯的原始輸出。

#### Scenario: 使用者修改譯文後合成

- **WHEN** 使用者翻譯後把「翻譯結果」改成其他族語文字，再按「合成語音」
- **THEN** 合成的是修改後的文字

#### Scenario: 翻譯結果為空

- **WHEN** 「翻譯結果」為空白時按「合成語音」
- **THEN** 顯示錯誤訊息，請使用者先翻譯或輸入族語文字，不呼叫 TTS

### Requirement: 新翻譯時清空舊的合成結果

按下「翻譯」時，MT SHALL 清空合成結果元件，避免新譯文旁邊顯示上一段譯文的語音。

#### Scenario: 再次翻譯

- **WHEN** 已有合成結果時，使用者輸入新的原文並按「翻譯」
- **THEN** 合成結果元件被清空

### Requirement: 透過固定網址以 gradio_client 呼叫 TTS

MT 後端 SHALL 以 `gradio_client` 呼叫 `https://{SAPOLITA_WEBSITE_HOST}/hnang-kari-ai-asi-sluhay/` 的 `/synthesize`，`language` 參數為語別代碼（如 `ami_Coas`）對應的語別名稱（如 `阿美_海岸`）。MT MUST NOT 由使用者 request 的 Host 等標頭推導 TTS 網址。MT 端的合成函式只做網路呼叫，MUST NOT 占用 GPU（不加 `@spaces.GPU`），在 GPU 或 CPU 環境行為相同。

#### Scenario: 偽造 Host 標頭

- **WHEN** 使用者送出帶有偽造 Host 標頭的請求並觸發合成
- **THEN** MT 仍只連線到 `SAPOLITA_WEBSITE_HOST` 對應的 TTS 網址

#### Scenario: tts 尚未啟動時 mt 可正常啟動

- **WHEN** mt 容器啟動時 TTS 服務無法連線
- **THEN** mt 正常啟動，翻譯功能可用；按合成時才顯示服務暫時無法使用的訊息

### Requirement: 合成逾時與錯誤訊息

MT SHALL 最多等待 TTS 20 秒（含排隊與合成）。各種失敗情況 SHALL 以 Gradio 錯誤訊息呈現，MUST NOT 讓頁面卡住或 crash。

#### Scenario: 超過 20 秒

- **WHEN** TTS 在 20 秒內沒有回傳結果
- **THEN** MT 盡量取消該 TTS job，並顯示「現在使用人數眾多，請稍候再試」

#### Scenario: TTS 回報錯誤

- **WHEN** TTS 回傳錯誤，例如 `Unknown characters: 」`
- **THEN** MT 顯示 TTS 回傳的錯誤訊息

#### Scenario: TTS 無法連線

- **WHEN** 連線 TTS 失敗
- **THEN** MT 顯示「語音合成服務暫時無法使用，請稍候再試」，下次合成時重新建立連線

### Requirement: 供檢查腳本使用的固定 API 名稱

MT SHALL 將合成按鈕事件註冊為 `api_name="synthesize"`，並將「華語 ⮕ 族語」族別切換事件註冊為 `api_name="to_formosan_languages"`。

#### Scenario: 檢查腳本測試非阿美語別

- **WHEN** 外部程式在同一個 session 先呼叫 `/to_formosan_languages`（`ethnicity="泰雅"`），再以 `tay_Wand` 呼叫 `/synthesize`
- **THEN** 回傳音檔
