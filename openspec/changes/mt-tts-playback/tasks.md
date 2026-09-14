## 1. TTS：引號前處理修正（tts/）

- [x] 1.1 [tts/] 在 `tts/app.py` 新增共用函式 `normalize_gen_text`：先移除 `"` `“` `”` `「` `」` 並 `strip()`，再判斷結尾是否為 `. ? ! , ; :`，不是就補 `.`；空字串時 `raise gr.Error("請勿輸入空字串。")`
- [x] 1.2 [tts/] `default_speaker_tts` 與 `custom_speaker_tts` 改用 `normalize_gen_text` 取代原本的空字串檢查與補句點邏輯
- [x] 1.3 [tts/] 離線驗證：用 stub gradio 載入 `tts/ipa`，確認 `Sowal sa ko singsi, "Ano dafak micodad kita."`（阿美_海岸）與 `marengay na sinsi, " temakesi ta nu ʼemanan.」`（卑南_知本）經 `normalize_gen_text` → `text_to_ipa` 不再報錯，且 `talacowa kiso?` 轉出的 IPA 與修正前相同

## 2. TTS：新增 `/synthesize` API（tts/）

- [x] 2.1 [tts/] 新增 `synthesize_by_language(language, text)`：以 `get_refs_by_perfix(language + "_")` 取第一位配音員，找不到時 `raise gr.Error` 指出不支援的語別，找到則呼叫 `default_speaker_tts`
- [x] 2.2 [tts/] 新增一組 `visible=False` 的 `gr.Textbox`（language、text）、`gr.Audio`、`gr.Button`，以 `.click(synthesize_by_language, ..., api_name="synthesize")` 註冊；確認網頁版面沒有變化
- [x] 2.3 [tts/] 確認既有 `/default_speaker_tts`、`/custom_speaker_tts` 的 api 名稱與參數不變（`grafana-k6/tts.js` 仍可用）
- [x] 2.4 [tts/] Docker：只需重建 `ithuan/formosan-ai:tts` image，**不需**重建 `formosan-ai-common` image；requirements 無變動

## 3. MT：合成語音按鈕與呼叫 TTS（mt/）

- [x] 3.1 [mt/] 在 `mt/requirements.in` 明列 `gradio_client`，執行 `pip-compile mt/requirements.in` 更新 `mt/requirements.txt`，確認版本與 `gradio==5.49.1` 相容（1.13.x）
- [x] 3.2 [mt/] 新增 `TTS_API_URL = f"https://{SAPOLITA_WEBSITE_HOST}/hnang-kari-ai-asi-sluhay/"`（讀環境變數，不使用 request 標頭）與 `CODE_TO_LANGUAGE` 反查表（`ami_Coas` → `阿美_海岸`）
- [x] 3.3 [mt/] 實作延遲建立、全程共用的 `Client`，發生連線錯誤時丟棄並於下次重建（`TtsClient` class 與 `TTS_API_URL` 等常數放在 `mt/tts_client.py`）
- [x] 3.4 [mt/] 實作 `synthesize(text, tgt_lang)`：空白時 `gr.Error`（請先翻譯或輸入族語文字）；`client.submit(..., api_name="/synthesize")` 後 `job.result(timeout=20)`；client 下載到專用暫存目錄，讀成 bytes 後刪除原檔，回傳 bytes 給 `gr.Audio`（見 design 決策 6）；`TimeoutError` 時 `job.cancel()` 並顯示「現在使用人數眾多，請稍候再試」；`AppError` 轉發 tts 訊息；其他例外顯示「語音合成服務暫時無法使用，請稍候再試」；不加 `@spaces.GPU`
- [x] 3.5 [mt/] 「華語 ⮕ 族語」Tab 在 `to_formosan_output` 下方新增「合成語音」按鈕與 `gr.Audio(label="合成結果", show_share_button=False, show_download_button=True)`，`.click(synthesize, inputs=[to_formosan_output, to_formosan_tgt_lang], outputs=audio, api_name="synthesize")`
- [x] 3.6 [mt/] `to_formosan_ethnicity.change` 設定 `api_name="to_formosan_languages"`
- [x] 3.7 [mt/] `to_formosan_btn.click` 翻譯時一併清空合成結果 `gr.Audio`（`translate` 本身維持不設 timeout）
- [x] 3.8 [mt/] 確認「族語 ⮕ 華語」Tab 沒有任何合成相關元件
- [x] 3.9 [mt/] Docker：`mt/Dockerfile` 的 `COPY` 加上 `tts_client.py`；只需重建 `ithuan/formosan-ai:mt` image（requirements 有變動），**不需**重建 `formosan-ai-common` image；`docker-compose.yml` 的 mt 已有 `SAPOLITA_WEBSITE_HOST`，不需修改

## 4. 上線後檢查腳本（production_tests/）

- [x] 4.1 [production_tests/] 建立 `production_tests/requirements.in`（`gradio_client==1.13.3`、`PyYAML`），執行 `pip-compile production_tests/requirements.in` 產生 `requirements.txt`
- [x] 4.2 [production_tests/] `check_mt_tts_playback.py` 基本架構：讀 `BASE_URL`（預設 `https://ai-labs.ilrdf.org.tw`），組出 MT、TTS 網址；收集每項檢查的成功或失敗，結尾列出失敗項目並以 exit code 0/1 結束；循序執行，不併發
- [x] 4.3 [production_tests/] `FORMOSAN_LANGUAGES_MAP` 從 `mt/app.py` 搬到 `mt/formosan_languages.py`（`mt/Dockerfile` 的 `COPY` 一併加上），腳本直接 import 它，以 PyYAML 讀 `tts/configs/refs.yaml`，取每個語別第一位配音員的 `text`；路徑以腳本所在位置為基準
- [x] 4.4 [production_tests/] 檢查 1：TTS `view_api` 含 `/synthesize`，參數為 `language`、`text`
- [x] 4.5 [production_tests/] 檢查 2：42 個語別直接呼叫 TTS `/synthesize`，都要回傳音檔；找不到配音員的語別列為失敗
- [x] 4.6 [production_tests/] 檢查 3：引號結尾文字、含「」文字要合成成功；不存在的語別要回傳錯誤
- [x] 4.7 [production_tests/] 檢查 4：`阿美_海岸`、`泰雅_萬大`、`魯凱_茂林`、`卡那卡那富`、`賽夏` 各自在同一 MT session 先呼叫 `/to_formosan_languages` 再呼叫 `/synthesize`，確認回傳音檔、印出耗時，超過 15 秒印警告
- [x] 4.8 [production_tests/] 撰寫 `production_tests/README.md`：安裝、執行、`BASE_URL`、各項檢查內容、預估耗時與 GPU 消耗、應以與部署相同的 commit 執行、tts 先於 mt 部署

## 5. 驗證

- [x] 5.1 [tts/ mt/ production_tests/] 執行 `tox -e flake8` 通過
- [x] 5.2 [production_tests/] `pymarkdown scan production_tests/README.md` 通過（`tox -e pymarkdown` 的 `scan .` 不會遞迴掃子目錄，要直接指定檔案）
- [x] 5.3 [production_tests/] 對目前正式站跑檢查腳本：確認腳本本身可執行，且在 tts、mt 尚未部署新版時，檢查 1–4 正確回報失敗
- [ ] 5.4 [deploy/] 部署 tts 新版到測試機或正式機後，跑檢查腳本，確認檢查 1–3 通過
- [ ] 5.5 [deploy/] 部署 mt 新版後，跑完整檢查腳本，確認端到端（含 hairpin NAT）通過；在網頁上手動測試「翻譯 → 修改譯文 → 合成語音」流程
