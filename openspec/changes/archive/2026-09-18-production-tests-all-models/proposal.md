## Why

`docker-shared-base-image` 會換掉四個服務的 base image、統一 torch 版本並移除多個 apt 套件，需要一把改前改後都能用的尺。目前 `tests/production_tests/` 只檢查 mt 與 tts，asr 與 asr-kaldi 沒有任何功能檢查；`tests/grafana-k6/` 量的是延遲，不驗功能，且每次要裝 k6 與 ffmpeg。測試機 `https://tshi5v100.ithuankhoki.tw/` 四個服務都已上線，2026-09-18 以 `海岸阿美語-曾玉蘭-個人生命史-短.mp4` 實測，asr、asr-kaldi、mt 輸出皆可重現，tts 音檔每次位元不同但長度穩定，暖機前後耗時差 5 到 8 倍。

## What Changes

- **production_tests 擴充到四個模型**：新增 asr（mp4 → `/generate_srt`）、asr-kaldi（mp3 → `/automatic_speech_recognition`）、mt（`/translate`、`/translate_1`）、tts（`/synthesize` 的 wav 格式與非靜音）與頁面（`/`、`/config`、`common_static`）的檢查；只驗「有回應、格式對、有內容」，不比對輸出內容，模型換版不需更新測試。
- **語別抽樣**：每個服務固定包含阿美（素材語別），再隨機抽 3 個其他語別，`--all-languages` 掃全部。阿美要求非空，其他語別只要求呼叫成功且型別正確，空字串印警告。
- **暖機與耗時門檻**：每個服務第一次呼叫不計時，第二次計時並與門檻比較，超過只印警告。這是拿掉內容比對後唯一能發現 GPU 未被使用的訊號。
- **單一入口 `run_all.py`**：`--services` 選服務、`--all-languages`、`--report <path>` 輸出 JSON（每項結果與耗時），供改前改後比對；沿用 `check_mt_tts_playback.py` 的檢查函式與輸出風格。
- **tox 入口**：`tox -e production_tests -- <參數>`，`BASE_URL` 由 `passenv` 傳入。
- **Claude Code skill**：新增 `.claude/skills/production-tests/SKILL.md`，讓 AI 助理知道何時、如何對指定主機執行檢查與解讀結果。
- **測試相關的東西集中到 `tests/`**：`tests/production_tests/`、`tests/grafana-k6/`、共用素材 `tests/data/`（mp4 與 mp3 各一份，原本的 `grafana-k6/testing_data/` 與未提交的 `production_tests/video/` 移除），以及 `tests/check_lock_consistency.py`。k6 腳本改指向 `../data/`。
- **mt 的「族語 ⮕ 華語」族別切換補上 `api_name="to_zh_languages"`**：原本沒命名，Gradio 自動叫它 `/lambda`，mt 只要多一個沒命名的事件，名字就可能變。檢查腳本改呼叫 `/to_zh_languages`。**BREAKING**：mt 的 `/lambda` API 改名。
- **mt 移除 `spaces`**：`import spaces` 與 `@spaces.GPU` 是照抄 Hugging Face Spaces 留下的，自架環境沒有作用；`mt/requirements.in` 一併移除 `spaces==0.36.0`。README 列的 `ithuan-formosan-translation.hf.space` 是獨立的 Space repo，2026-09-18 查為暫停、CPU 硬體，不受影響。
- **文件**：`tests/production_tests/README.md` 改寫，`docker-shared-base-image` 的 tasks 補上改前改後各跑一次的步驟。

## 非目標

- 不比對 asr、asr-kaldi、mt 的輸出文字與 golden 檔，不比對 tts 音檔內容。
- 不做 Playwright 或 Selenium 的瀏覽器測試。
- 不改 k6 腳本的測試方法與次數，只改素材路徑。
- 不把檢查排進 Travis（需要對測試機的網路存取並消耗其 GPU，維持人工執行）。
- 不新增其他語別的音檔素材。
- 除了 mt 的 `api_name` 與移除 `spaces`，不修改四個服務的程式碼或 API。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `tts-synthesize-api`：「既有 API 維持不變」的情境引用的 k6 腳本路徑改為 `tests/grafana-k6/tts.js`，需求本身不變。
- `production-tests`：檢查範圍從 mt 與 tts 擴大到四個模型與頁面；新增語別抽樣規則、暖機與耗時門檻、單一入口與 JSON 報告、tox 入口、Claude Code skill、素材位置；文件需求隨之更新。

## Impact

- **tests/production_tests/**：新增 `run_all.py`、`check_asr.py`、`check_asr_kaldi.py`、`check_mt.py`、`check_tts.py`、`check_pages.py`、`timing.py`（暖機與門檻）；`check_mt_tts_playback.py` 維持可獨立執行；`README.md` 改寫；`requirements.in` 不需新增套件（wav 與 RMS 用 stdlib）。
- **tests/data/**：新目錄，放 mp4 與 mp3。
- **tests/grafana-k6/**：`asr.js`、`asr-kaldi.js` 的 `open()` 路徑、`README.md`、`METHOD.md` 的素材說明。
- **tox.ini**：新增 `production_tests` env。
- **.claude/skills/**：新增 `production-tests` skill。
- **asr/、asr-kaldi/、mt/、tts/**：不改程式碼；腳本 import `asr/languages.py`、`mt/formosan_languages.py`，讀 `asr-kaldi/configs/models.yaml` 與 `tts/configs/refs.yaml` 取語別表，與現有腳本「以受測主機相同 commit 執行」的前提一致。
- **openspec/changes/docker-shared-base-image/tasks.md**：每個 PR 補「改前改後各跑一次 `tox -e production_tests` 與 k6」。
- **族語方言相容性**：語別表直接讀自各服務的設定，新增語別時測試自動涵蓋；素材只有海岸阿美語，其他語別只驗 API 接受度。
