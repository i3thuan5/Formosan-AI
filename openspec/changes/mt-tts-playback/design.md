## Context

MT（`mt/app.py`）與 TTS（`tts/app.py`）是同一個 repo 的兩個獨立 Gradio 服務，各自是一個容器，對外透過 `https://ai-labs.ilrdf.org.tw/<路徑>/` 提供網頁與 API：

- MT：`/kari-seejiq-tnpusu-ai-hmjil/`
- TTS：`/hnang-kari-ai-asi-sluhay/`

兩個服務的語別命名一致：MT 的 `FORMOSAN_LANGUAGES_MAP` key（如 `泰雅_萬大`）加上 `_女聲`、`_男聲1` 等後綴，就是 TTS `refs.yaml` 的配音員 key。42 個語別都至少有一位配音員。

探索階段（2026-09-13）對正式站的實測：

```
TTS handshake                         0.8s
TTS 合成  45 字元                     2.3s
TTS 合成 191 字元                     5.5s
TTS 合成 299 字元                     8.7s
MT  翻譯一句                          0.3–0.5s

/default_speaker_tts ref=泰雅_四季_女聲（沒先呼叫 /lambda）   ❌ 不在 choices 裡
/lambda("泰雅") → 同 session 再合成                          ✅
MT 5 句 × 42 語別 → text_to_ipa：191/210 通過，18 句因引號 bug 失敗，1 句含全形 」
```

各網址的 Gradio config：

```
https://ai-labs.ilrdf.org.tw/hnang-kari-ai-asi-sluhay/config             200
https://kari-seejiq-tnpusu-ai-hmjil.ithuan.tw/hnang-kari-ai-asi-sluhay/  404  ← mt 子網域底下沒有 tts
```

## Goals / Non-Goals

**Goals:**

- 「華語 ⮕ 族語」Tab 翻譯後，同一頁可以合成並播放譯文語音。
- TTS 提供不依賴 UI session 狀態、名稱固定的合成 API。
- 修正 TTS 合成文字的引號前處理 bug。
- 提供可指定主機的上線後檢查腳本，及早發現 API 約定損壞或服務之間連不上。

**Non-Goals:**

- 見 proposal「非目標」：族語 ⮕ 華語不加合成、翻譯不設 timeout、不選配音員、不從 request 推導網址、不改既有 TTS API。

## Decisions

### 1. mt 後端以 `gradio_client` 呼叫 tts（不在瀏覽器端呼叫，也不在 mt 載入 TTS 模型）

```
瀏覽器 ──[合成語音]──► mt synthesize() ──gradio_client──► tts /synthesize ──► 音檔
                              │                                              │
                              └────────── gr.Audio ◄── 下載到 mt 暫存 ◄───────┘
```

**理由**：符合 Gradio 一般的 `.click → outputs` 寫法，逾時與錯誤都能在後端處理後轉成 `gr.Error`。`gradio_client` 已隨 gradio 安裝。

**替代方案**：
- 瀏覽器端用 @gradio/client：要在 Gradio 裡塞自訂 JS 與 audio 元素，和現有結構不合。
- mt 容器自己載入 F5-TTS：映像檔與 GPU 記憶體都多一份，還要維護兩份 G2P 與 refs。

### 2. tts 新增 `/synthesize`，以隱藏元件承接輸入（方案 B2）

tts 新增：

- `synthesize_by_language(language: str, text: str)`：用 `get_refs_by_perfix(language + "_")` 取第一位配音員，再交給既有的 `default_speaker_tts` 流程。找不到配音員時 `raise gr.Error`。
- 一組 `visible=False` 的 `gr.Textbox`（language、text）、`gr.Audio`、`gr.Button`，以 `.click(..., api_name="synthesize")` 註冊。

前綴要加 `_`，避免語別名稱互為前綴時選錯配音員。

GPU/CPU 行為與既有的 `default_speaker_tts` 相同：沿用 `gpu_decorator`，在 Spaces 環境用 `spaces.GPU`，其他環境在 `f5_tts` 的 `device` 上執行（有 CUDA 用 CUDA，否則 CPU）。

**理由**：`gr.Textbox` 沒有 choices 檢查，一次呼叫就能完成，而且 API 名稱固定、有明確約定。

**替代方案**：
- B1：mt 每次建新 `Client`，先呼叫 `/lambda` 再呼叫 `/default_speaker_tts`。每次多 handshake 約 0.8 秒，而且依賴自動產生的 `/lambda` 名稱，tts UI 事件一調整就會悄悄壞掉。
- `gr.api()` 純函式 endpoint：回傳音檔要怎麼寫還需要驗證，隱藏元件的做法比較確定。

### 3. 引號前處理：先移除引號，再判斷是否補句點

新增共用函式（例如 `normalize_gen_text`）：先移除 `"` `“` `”` `「` `」`、`strip()`，再檢查結尾是否為 `. ? ! , ; :`，不是就補 `.`。`default_speaker_tts` 與 `custom_speaker_tts` 都改用它，`/synthesize` 因為走 `default_speaker_tts` 也一併套用。

**理由**：`text_to_ipa` 原本就會刪 `"` `“` `”`，把刪除提前不會改變其他文字的結果。加入 `「` `」` 可以涵蓋 MT 輸出混入全形引號的情況。

**替代方案**：只在 mt 端清理文字。這樣 tts 網頁版的使用者還是會遇到這個 bug。

### 4. TTS 網址用 `SAPOLITA_WEBSITE_HOST` 環境變數組成

`TTS_API_URL = f"https://{SAPOLITA_WEBSITE_HOST}/hnang-kari-ai-asi-sluhay/"`，mt 的 docker-compose 已有此變數。

**理由**：Host header 可以偽造，從 request 推導會造成 SSRF。而且使用者若從 `kari-seejiq-tnpusu-ai-hmjil.ithuan.tw` 直接連入，該網域下沒有 tts 路徑。

**替代方案**：直接連容器內部的 `http://tts:7860/`。使用者選擇走公開網址，這樣實際路徑和外部使用者一致，也不必處理 `GRADIO_ROOT_PATH` 對 client root 的影響。

### 5. mt 的 `Client` 延遲建立、全程共用，失敗時重建

模組層級保存一個 `Client`，第一次按合成時才建立，避免 mt 啟動時因 tts 還沒上線而失敗。`/synthesize` 沒有 session 依賴，多位使用者可以共用同一個 client 同時送出 job。呼叫過程發生連線類錯誤時丟棄 client，下次重建，以應付 tts 重啟後 config 過期的情況。

### 6. 逾時與錯誤訊息

```
job = client.submit(language_name, text, api_name="/synthesize")
job.result(timeout=20)          ← 20 秒涵蓋 tts 排隊＋合成
```

| 情況 | 呈現 |
|---|---|
| 翻譯結果為空白 | `gr.Error`：請先翻譯或輸入族語文字 |
| `TimeoutError`（超過 20 秒） | `job.cancel()` 盡量從 tts queue 撤回；`gr.Error`：現在使用人數眾多，請稍候再試 |
| tts 回傳 `gr.Error`（`AppError`，例如不認得的字元） | 將 tts 的錯誤訊息轉成 mt 的 `gr.Error` |
| 連線失敗等其他例外 | 重建 client；`gr.Error`：語音合成服務暫時無法使用，請稍候再試 |

mt 端的 `synthesize` 只負責網路呼叫，不需要 `@spaces.GPU`。

**音檔交給 `gr.Audio` 的方式（實作時決定）**：`Client(download_files=專用暫存目錄)` 下載音檔後讀成 bytes、刪除下載的原檔，再把 bytes 回傳給 `gr.Audio`。

- 不直接回傳下載路徑：Gradio 會複製一份到自己的快取，但 client 下載的原檔不在 `delete_cache` 的追蹤範圍內，會一直累積在容器的 /tmp。
- 不用 `download_files=False` 回傳 tts 的 URL：`gr.Audio` 下載 URL 時會經過 Gradio 的 SSRF 保護（safehttpx）。如果正式機內部 DNS 把網域解析成私有 IP，就會被擋。
- bytes 會由 `gr.Audio` 存進 app 快取，由 `delete_cache` 清理。

### 7. mt 事件設定固定 `api_name`

- 合成按鈕：`api_name="synthesize"`。
- 「華語 ⮕ 族語」族別切換：`api_name="to_formosan_languages"`。

mt 的 `tgt_lang` 是 `gr.Radio`（commit 5228d4a 由 Dropdown 改來），有和 tts Radio 一樣的 choices 限制。端到端檢查要在同一個 session 先呼叫族別切換，才能測非阿美語別。固定名稱可以讓檢查腳本不依賴 `/lambda_1`。

### 8. 新翻譯送出時清空舊音檔

按下「翻譯」時一併把合成結果 `gr.Audio` 設為空，避免新譯文旁邊顯示上一段譯文的語音。

### 9. 上線後檢查腳本放在 `production_tests/`

```
production_tests/
├── README.md               怎麼跑、檢查了什麼
├── requirements.txt        gradio_client、PyYAML
└── check_mt_tts_playback.py
```

`BASE_URL` 環境變數預設 `https://ai-labs.ilrdf.org.tw`，和 `grafana-k6/` 的慣例一致，可以指定為測試機。全部通過 exit 0，任何一項失敗 exit 1，並列出失敗項目。

| 檢查 | 內容 | 資料來源 |
|---|---|---|
| 1. API 約定 | tts `view_api` 含 `/synthesize`，參數為 language、text | — |
| 2. 42 語別合成 | 直接呼叫 tts `/synthesize`，每個語別都要回傳音檔 | 語別：以 `ast` 解析 `mt/app.py` 的 `FORMOSAN_LANGUAGES_MAP`（不 import，避免載入模型）；句子：`tts/configs/refs.yaml` 該語別第一位配音員的 `text` |
| 3. 回歸 | 引號結尾、含「」的文字能合成成功；不支援的語別回傳錯誤 | 腳本內固定字串 |
| 4. 端到端 | 呼叫 mt `/to_formosan_languages` 再呼叫 `/synthesize`，回傳音檔，並記錄耗時 | 5 個語別：`阿美_海岸`（預設）、`泰雅_萬大`（ṟ）、`魯凱_茂林`（ɨ、é）、`卡那卡那富`（ʉ）、`賽夏`（大寫 S、`:`） |

端到端耗時超過 15 秒時印出警告（接近 20 秒逾時），但不算失敗。

**理由**：用和 mt 相同的 `gradio_client` 呼叫，測到的就是實際路徑。k6 定位在延遲壓測，而且需要另外安裝。

## Risks / Trade-offs

- **[mt 容器連不到自己的公開網域（hairpin NAT）]** → 上線後立刻跑 `production_tests/` 端到端檢查。不通時可以再增加 `TTS_API_URL` 覆寫變數改走內部網址（不在本次範圍）。
- **[部署順序：mt 先於 tts 上線]** → tasks 與 README 註明發佈順序；mt 端會顯示「語音合成服務暫時無法使用」，不會 crash。
- **[長文字接近 20 秒逾時]** → 299 字元約 8.7 秒，無排隊時有餘裕；尖峰時段會顯示人數眾多訊息，屬預期行為。
- **[使用者翻譯後切換語別再按合成]** → 會用新語別的配音員與 G2P 念舊譯文，可能出現不認得的字元錯誤。目前接受，錯誤訊息會照實顯示。
- **[檢查腳本讀的是本地 repo 的 refs.yaml 與語別表]** → 若受測主機部署的版本不同，結果可能不一致。README 註明應用與部署版本相同的 commit 執行。
- **[檢查腳本會實際消耗受測主機的 GPU]** → 一次約 42＋5＋數句合成，約 2–3 分鐘；循序執行，不併發。
- **[MT 輸出本身含 G2P 不認得的字元]**（實測 210 句中 1 句）→ 顯示 tts 的錯誤訊息，使用者可以自行修改譯文後再合成。

## Migration Plan

1. 合併後依序建置並推送 `ithuan/formosan-ai:tts`、`ithuan/formosan-ai:mt` image（mt requirements 有變動；`common` image 不需重建）。
2. **先**部署 tts，對測試機或正式機跑 `production_tests/` 的檢查 1–3。
3. **再**部署 mt，跑完整檢查（含端到端）。
4. **回滾**：mt 退回前一版 image 即可移除按鈕；tts 新增的 `/synthesize` 與引號修正都向下相容，不需回滾。

## Open Questions

- hairpin NAT 是否可行，要到正式機部署後由端到端檢查確認。
