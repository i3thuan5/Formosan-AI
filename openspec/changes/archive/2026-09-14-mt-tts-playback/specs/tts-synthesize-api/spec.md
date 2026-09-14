## ADDED Requirements

### Requirement: 以語別為參數的合成 API

TTS SHALL 提供名稱固定為 `/synthesize` 的 Gradio API，參數為 `language: str`（語別名稱，與 MT `FORMOSAN_LANGUAGES_MAP` 的 key 相同，例如 `泰雅_萬大`）與 `text: str`，回傳合成音檔。此 API MUST NOT 受任何 `gr.Radio` 或 `gr.Dropdown` 的 choices 檢查限制，也 MUST NOT 依賴同一 session 先前呼叫的其他事件。

#### Scenario: 非阿美語別直接合成成功

- **WHEN** 外部程式以新的 session 呼叫 `/synthesize`，`language="泰雅_四季"`，`text="lokah su"`
- **THEN** 回傳音檔，不出現「not in the list of choices」錯誤

#### Scenario: 42 個 MT 語別皆可合成

- **WHEN** 以 MT `FORMOSAN_LANGUAGES_MAP` 中任一語別呼叫 `/synthesize`，文字為 G2P 可辨識的族語
- **THEN** 回傳音檔

### Requirement: 自動選用該語別第一位配音員

`/synthesize` SHALL 以 `refs.yaml` 中 key 前綴為 `{language}_` 的第一位配音員作為參考音檔。

#### Scenario: 語別有多位配音員

- **WHEN** 呼叫 `/synthesize`，`language="阿美_秀姑巒"`
- **THEN** 使用 `阿美_秀姑巒_女聲1` 合成

#### Scenario: 不支援的語別

- **WHEN** 呼叫 `/synthesize`，`language` 在 `refs.yaml` 中找不到任何配音員
- **THEN** 回傳 Gradio 錯誤，訊息指出不支援的語別，不產生音檔

### Requirement: 合成 API 的前處理與既有網頁合成一致

`/synthesize` SHALL 沿用「預設配音員」的合成流程：空字串檢查、句尾補標點、`text_to_ipa` 轉換、F5-TTS 推論。GPU/CPU 行為 SHALL 與 `default_speaker_tts` 相同：Spaces 環境使用 `spaces.GPU`，其他環境在 `f5_tts` 的 `device` 上執行（有 CUDA 用 CUDA，否則使用 CPU）。

#### Scenario: 空字串

- **WHEN** 呼叫 `/synthesize`，`text` 為空白
- **THEN** 回傳 Gradio 錯誤「請勿輸入空字串。」

#### Scenario: 含 G2P 不認得的字元

- **WHEN** 呼叫 `/synthesize`，`text` 含有該語別 G2P 表不認得的字元
- **THEN** 回傳 Gradio 錯誤，訊息列出不認得的字元

### Requirement: 合成文字先移除引號再補句尾標點

TTS 在「預設配音員」、「自己當配音員」與 `/synthesize` 的合成文字前處理中，SHALL 先移除 `"`、`“`、`”`、`「`、`」` 並去除前後空白，再判斷結尾是否為 `.`、`?`、`!`、`,`、`;`、`:`，若不是才補上 `.`。

#### Scenario: 文字以引號結尾

- **WHEN** 以 `阿美_海岸` 合成 `Sowal sa ko singsi, "Ano dafak micodad kita."`
- **THEN** 合成成功，不出現 `Unknown characters: .`

#### Scenario: 文字含全形引號

- **WHEN** 以 `卑南_知本` 合成 `marengay na sinsi, " temakesi ta nu ʼemanan.」`
- **THEN** 合成成功，不出現 `Unknown characters: 」`

#### Scenario: 不含引號的文字結果不變

- **WHEN** 合成不含上述引號字元的文字，例如 `talacowa kiso?`
- **THEN** 轉出的 IPA 與修正前相同

### Requirement: 既有 API 維持不變

TTS 既有的 `/default_speaker_tts`、`/custom_speaker_tts` 等 API 的名稱與參數 SHALL 維持不變。

#### Scenario: 延遲測試仍可呼叫舊 API

- **WHEN** `grafana-k6/tts.js` 以 `ref="阿美_秀姑巒_女聲1"` 呼叫 `/default_speaker_tts`
- **THEN** 回傳音檔
