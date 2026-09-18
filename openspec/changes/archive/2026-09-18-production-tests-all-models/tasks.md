## 1. 測試相關的東西搬到 tests/（tests/data/、tests/grafana-k6/、tests/production_tests/）

- [x] 1.1 [tests/data/] 以 `git mv tests/grafana-k6/testing_data tests/data` 搬移 mp4 與 mp3，刪除未提交的 `tests/production_tests/video/`；新增 `tests/data/README.md`，內容為 METHOD.md 中的素材規格與 ffmpeg 轉檔指令
- [x] 1.2 [tests/grafana-k6/] `asr.js`、`asr-kaldi.js` 的 `open()` 改為 `../data/${FILENAME}`；`README.md` 的檔案表與 `METHOD.md` 第二節改指向 `tests/data/`
- [x] 1.3 [tests/grafana-k6/] 若本機有 k6，執行 `k6 run --iterations 1 asr.js` 確認素材讀得到；沒有則以 `node -e` 或 `ls` 確認相對路徑正確

## 2. 檢查腳本（tests/production_tests/）

- [x] 2.1 [tests/production_tests/] 新增 `timing.py`：門檻常數（asr 15、asr-kaldi 30、mt 10、tts 20 秒）、`timed(fn)` 回傳結果與耗時、`warm_up(client, ...)` 呼叫一次不計時；警告收集到共用清單
- [x] 2.2 [tests/production_tests/] 新增 `check_pages.py`：以 `urllib` 對四個服務的 `/`、`/config` 與 `common_static` 的 favicon、`common.css` 發 GET，回 200 且 `/config` 可解析 JSON；URL 清單依 `common/static/` 實際檔名與 Gradio 靜態路徑實測後定案
- [x] 2.3 [tests/production_tests/] 新增 `check_asr.py`：語別表 import `asr/languages.py` 的 `LANGUAGE_GROUPS`；固定 `ami-x-pswl` 加隨機 3 個或全部；`handle_file` 送 mp4 到 `/generate_srt`；解析 cue 格式（序號、時間戳、`族語：`、`華語：`）；阿美族語行非空，其他語別空則警告
- [x] 2.4 [tests/production_tests/] 新增 `check_asr_kaldi.py`：以 PyYAML 讀 `asr-kaldi/configs/models.yaml` 第一個模型的 `dialect_mapping`；固定 `formosan_ami` 加隨機 3 個或全部；送 mp3 到 `/automatic_speech_recognition`；阿美非空，其他空則警告
- [x] 2.5 [tests/production_tests/] 新增 `check_mt.py`：語別表 import `mt/formosan_languages.py`；固定 `ami_Coas` 加隨機 3 個或全部；`/translate` 送 `sosowalen ako itiya:ayho a ʼorip niyam`，`/translate_1` 先 `/to_formosan_languages` 再送 `我們以前的生活`；阿美非空且不等於輸入，其他空則警告
- [x] 2.6 [tests/production_tests/] 新增 `check_tts.py`：以 `阿美_海岸` 呼叫 `/synthesize`；用 `wave` 與 `array` 檢查 24 kHz、長度大於 0.5 秒、RMS 高於門檻；RMS 門檻以測試機實際合成的 wav 量測後定（預期正常值遠高於靜音）
- [x] 2.7 [tests/production_tests/] 新增 `run_all.py`：`argparse` 提供 `--services`（預設 `pages,asr,asr-kaldi,mt,tts`）、`--all-languages`、`--report`；每個服務各自 `connect`，連不上時該服務所有檢查記為失敗並繼續；tts 與 mt→tts 的檢查直接 import `check_mt_tts_playback` 的 `run_tts_checks`、`run_end_to_end_checks`；結尾彙總失敗與警告；`--report` 寫 JSON（`base_url`、`git rev-parse HEAD`、開始時間、每項 service/name/ok/seconds/note、warnings、failed）
- [x] 2.8 [tests/production_tests/] `.gitignore` 加 `tests/production_tests/results/`；`check_mt_tts_playback.py` 只做必要的函式抽取，確保仍可獨立執行且行為不變
- [x] 2.9 [tests/production_tests/] `requirements.in` 確認不需新增套件（`gradio_client`、`PyYAML` 已有）；若有變動執行 `uv pip compile tests/production_tests/requirements.in --python-version 3.12 -o tests/production_tests/requirements.txt`（或現行的 `pip-compile`）
- [x] 2.10 [tests/production_tests/] `tox -e flake8` 通過

## 3. tox 與 Claude Code skill（tox.ini、.claude/skills/）

- [x] 3.1 [tox.ini] 新增 `[testenv:production_tests]`：`deps = -r tests/production_tests/requirements.txt`、`passenv = BASE_URL`、`commands = python tests/production_tests/run_all.py {posargs}`
- [x] 3.2 [.claude/skills/] 新增 `production-tests/SKILL.md`：frontmatter（`name`、`description`）；內容為何時執行、`BASE_URL=... tox -e production_tests -- <參數>`、三個參數說明、預估耗時與 GPU 消耗、警告與失敗的差別、改前改後 `--report` 比對步驟、以受測主機相同 commit 執行的提醒

## 4. 文件（tests/production_tests/、openspec/）

- [x] 4.1 [tests/production_tests/] 改寫 `README.md`：腳本表加入五個新檢查、安裝與 tox 執行方式、`BASE_URL` 與三個參數、每個服務的判準表、語別抽樣規則、暖機與門檻只警告、預估耗時、`--report` 比對流程、不可 import `app.py`、tts 先於 mt 部署
- [x] 4.2 [openspec/] `openspec/changes/docker-shared-base-image/tasks.md` 在 PR 1 與 PR 2 各補一項：部署到測試機前後各執行 `BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests -- --report tests/production_tests/results/<日期>-<commit>.json` 與 k6 各 10 次，比對報告無新失敗、耗時無數倍劣化
- [x] 4.3 [openspec/] 執行 `openspec validate --all --strict` 通過

## 5. mt 清理（mt/）

- [x] 5.1 [mt/] `to_zh_ethnicity.change` 加 `api_name="to_zh_languages"`；`tests/production_tests/check_mt.py` 改呼叫 `/to_zh_languages`
- [x] 5.2 [mt/] 移除 `import spaces` 與 `translate` 上的 `@spaces.GPU`；`mt/requirements.in` 移除 `spaces==0.36.0`，以 `uv pip compile mt/requirements.in --python-version 3.10 --python-platform linux --no-strip-extras -o mt/requirements.txt` 重編，確認只少了 spaces 一個套件。Docker：只需重建 `ithuan/formosan-ai:mt`，不需重建共用 image
- [x] 5.3 [mt/] 部署 mt 到測試機後跑 `tox -e production_tests -- --services mt`，確認 `/to_zh_languages` 可用；部署前對舊版執行會因為找不到 `/to_zh_languages` 而失敗（2026-09-18 部署後驗證通過，報告 `2026-09-18-after-440d095.json`）

## 6. 對測試機驗證（tests/production_tests/）

- [x] 6.1 [tests/production_tests/] 執行 `BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests -- --report tests/production_tests/results/$(date +%F)-$(git rev-parse --short HEAD).json`，全部通過、無警告，耗時在 3 分鐘內
- [x] 6.2 [tests/production_tests/] 執行 `--all-languages` 一次，確認非阿美語別的空字串只出現警告，記錄總耗時到 README 的預估欄
- [x] 6.3 [tests/production_tests/] 執行 `--services pages` 與 `--services asr,tts`，確認未選的服務沒有被連線
