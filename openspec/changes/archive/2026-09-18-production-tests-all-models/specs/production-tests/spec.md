## MODIFIED Requirements

### Requirement: 可指定受測主機的上線後檢查腳本

`tests/production_tests/` SHALL 提供單一入口 `run_all.py`，以 `BASE_URL` 環境變數指定受測主機，預設為 `https://ai-labs.ilrdf.org.tw`。四個服務的位置為 asr `{BASE_URL}/sapolita/`、asr-kaldi `{BASE_URL}/sapolita-kaldi/`、MT `{BASE_URL}/kari-seejiq-tnpusu-ai-hmjil/`、TTS `{BASE_URL}/hnang-kari-ai-asi-sluhay/`。腳本 SHALL 循序執行所有檢查（不併發），全部通過時 exit code 為 0，任一項失敗時為 1，並在結尾列出每個失敗項目與錯誤訊息。`--services` 參數 SHALL 可限制只跑指定服務（`pages`、`asr`、`asr-kaldi`、`mt`、`tts`），預設全部。既有的 `check_mt_tts_playback.py` SHALL 維持可獨立執行，其檢查由 `run_all.py` 直接 import 使用。

#### Scenario: 對測試機執行

- **WHEN** 執行 `BASE_URL=https://<測試機> python tests/production_tests/run_all.py`
- **THEN** 所有呼叫都送往測試機的四個服務

#### Scenario: 有檢查失敗

- **WHEN** 任一項檢查失敗
- **THEN** 腳本繼續跑完其餘檢查，最後列出失敗項目，exit code 為 1

#### Scenario: 只跑部分服務

- **WHEN** 執行 `python tests/production_tests/run_all.py --services asr,tts`
- **THEN** 只連線並檢查 asr 與 tts，其餘服務不呼叫

#### Scenario: 某個服務連不上

- **WHEN** 其中一個服務的 `/config` 無法連線
- **THEN** 該服務的所有檢查記為失敗並附錯誤訊息，其餘服務照常檢查

### Requirement: 檢查腳本說明文件

`tests/production_tests/README.md` SHALL 說明安裝方式、`tox -e production_tests -- <參數>` 與直接執行兩種方式、`BASE_URL` 用法、`--services`、`--all-languages`、`--report` 的用法、每個服務的檢查內容與判準、語別抽樣規則、暖機與耗時門檻只警告不失敗、預估耗時與會消耗受測主機 GPU、改前改後以 `--report` 比對的流程，並註明應以與受測主機部署版本相同的 commit 執行、不可 import 各服務的 `app.py`，以及 tts 須先於 mt 部署。

#### Scenario: 新成員第一次執行

- **WHEN** 開發者只照 README 操作
- **THEN** 能在 devcontainer 中以 tox 對指定主機完成一次檢查，並知道警告與失敗的差別

## ADDED Requirements

### Requirement: 檢查 asr 辨識輸出格式

腳本 SHALL 以 `tests/data/海岸阿美語-曾玉蘭-個人生命史-短.mp4` 呼叫 asr `/generate_srt`。回傳字串 MUST 含至少一個 cue，每個 cue 為序號、`HH:MM:SS,mmm --> HH:MM:SS,mmm`、`族語：` 行、`華語：` 行；語別為海岸阿美（`ami-x-pswl`）時族語行 MUST 非空。腳本 MUST NOT 比對族語或華語的文字內容。

#### Scenario: 海岸阿美辨識有內容

- **WHEN** 以 `ami-x-pswl` 送出 mp4
- **THEN** 回傳至少一個格式正確的 cue，且每個 cue 的族語行非空

#### Scenario: 其他語別只驗格式

- **WHEN** 以隨機抽到的非阿美語別送出同一支 mp4
- **THEN** 呼叫成功且回傳字串；cue 為零或族語行為空時印出警告，不算失敗

### Requirement: 檢查 asr-kaldi 辨識輸出

腳本 SHALL 以 `tests/data/海岸阿美語-曾玉蘭-個人生命史-短.mp3` 呼叫 asr-kaldi `/automatic_speech_recognition`。族別為阿美（`formosan_ami`）時回傳 MUST 為非空字串；其他族別只要求回傳字串，空字串印警告。族別表 SHALL 讀自 `asr-kaldi/configs/models.yaml` 第一個模型的 `dialect_mapping`。

#### Scenario: 阿美辨識有內容

- **WHEN** 以 `formosan_ami` 送出 mp3
- **THEN** 回傳非空字串

#### Scenario: 族別代碼表壞掉

- **WHEN** 送出 `models.yaml` 中存在但服務不接受的族別代碼
- **THEN** 服務回傳錯誤，該項檢查失敗並列出族別

### Requirement: 檢查 mt 雙向翻譯輸出

腳本 SHALL 以 asr 素材的一句族語（`sosowalen ako itiya:ayho a ʼorip niyam`）呼叫 mt `/translate`（族語到華語），並以一句華語（`我們以前的生活`）呼叫 `/translate_1`（華語到族語）。兩個方向的語別都是 Radio，呼叫前 MUST 在同一個 session 先切換族別：族語到華語呼叫 `/to_zh_languages`，華語到族語呼叫 `/to_formosan_languages`。語別為阿美_海岸（`ami_Coas`）時，兩個方向 MUST 回傳非空字串且不等於輸入；其他語別只要求回傳字串。語別表 SHALL 讀自 `mt/formosan_languages.py`。

#### Scenario: 阿美_海岸雙向翻譯有內容

- **WHEN** 分別呼叫 `/translate` 與 `/translate_1`
- **THEN** 兩者都回傳非空字串，且與輸入不同

#### Scenario: 隨機語別

- **WHEN** 以隨機抽到的其他語別呼叫兩個方向
- **THEN** 呼叫成功且回傳字串，空字串印警告

### Requirement: 檢查 tts 音檔格式與非靜音

腳本 SHALL 以 `阿美_海岸` 與素材句子呼叫 tts `/synthesize`。回傳檔案 MUST 能以 `wave` 開啟、取樣率為 24000、長度大於 0.5 秒、RMS 高於設定門檻。腳本 MUST NOT 比對音檔位元內容。

#### Scenario: 合成音檔有聲音

- **WHEN** 以 `阿美_海岸` 合成素材句子
- **THEN** 檔案為 24 kHz wav、長度大於 0.5 秒、RMS 高於門檻

#### Scenario: 合成出靜音

- **WHEN** 回傳的 wav 長度合格但 RMS 低於門檻
- **THEN** 該項檢查失敗，訊息含實際 RMS 與門檻

### Requirement: 檢查頁面與共用靜態檔

腳本 SHALL 對四個服務的 `/` 與 `/config` 發出 GET，MUST 回 200 且 `/config` 為 JSON；並對每個服務的 `common_static` 下 favicon 與 `common.css` 發出 GET，MUST 回 200。

#### Scenario: image 少複製共用檔案

- **WHEN** 某個服務的 `common_static` favicon 回 404
- **THEN** 該項檢查失敗並列出服務與 URL

### Requirement: 語別抽樣

對 asr、asr-kaldi、mt 三個服務，腳本 SHALL 固定包含阿美對應的語別（asr `ami-x-pswl`、asr-kaldi `formosan_ami`、mt `ami_Coas`），再以 `random.SystemRandom` 隨機抽 3 個其他語別；指定 `--all-languages` 時改為該服務的全部語別。每次執行 SHALL 印出抽到的語別。tts 的語別抽樣維持既有需求「檢查語別都能合成」。

#### Scenario: 預設抽樣

- **WHEN** 不帶 `--all-languages` 執行
- **THEN** asr、asr-kaldi、mt 各測阿美加 3 個隨機語別，並印出語別名稱

#### Scenario: 掃描全部語別

- **WHEN** 加上 `--all-languages`
- **THEN** asr 測 42 個語別、asr-kaldi 測 `dialect_mapping` 全部族別、mt 測 `FORMOSAN_LANGUAGES_MAP` 全部語別

### Requirement: 暖機與耗時門檻

每個服務 SHALL 先以阿美呼叫一次作為暖機且不計時；之後每次呼叫 SHALL 記錄耗時並印出。阿美的第二次呼叫耗時超過該服務門檻（`timing.py` 中定義，初始值 asr 15 秒、asr-kaldi 30 秒、mt 10 秒、tts 20 秒）時 SHALL 印出警告，但不算失敗。

#### Scenario: GPU 未被使用

- **WHEN** asr 的阿美第二次呼叫耗時 40 秒
- **THEN** 印出超過 15 秒門檻的警告，exit code 不因此變為 1

#### Scenario: 正常耗時

- **WHEN** 各服務第二次呼叫都在門檻內
- **THEN** 只印出耗時，沒有警告

### Requirement: JSON 報告

指定 `--report <path>` 時，腳本 SHALL 寫出 JSON，內容含 `base_url`、執行時的 git commit、開始時間、每項檢查的服務、名稱、是否通過、耗時與備註、警告清單、失敗數。未指定時 MUST NOT 寫檔。`tests/production_tests/results/` SHALL 列入 `.gitignore`。

#### Scenario: 改前改後比對

- **WHEN** 部署前後各以 `--report tests/production_tests/results/<日期>-<commit>.json` 執行
- **THEN** 兩份 JSON 的檢查名稱集合相同，可逐項比對通過狀態與耗時

### Requirement: tox 入口

`tox.ini` SHALL 提供 `production_tests` env，安裝 `tests/production_tests/requirements.txt`，以 `passenv` 傳入 `BASE_URL`，並把 `{posargs}` 傳給 `run_all.py`。

#### Scenario: 以 tox 執行並傳參數

- **WHEN** 執行 `BASE_URL=https://<測試機> tox -e production_tests -- --all-languages --report out.json`
- **THEN** `run_all.py` 收到 `--all-languages --report out.json`，且對測試機執行

### Requirement: Claude Code skill

`.claude/skills/production-tests/SKILL.md` SHALL 存在，frontmatter 含 `name: production-tests` 與 `description`，內容說明何時執行、指令、參數、預估耗時與 GPU 消耗、警告與失敗的解讀、改前改後以 `--report` 比對的步驟。skill MUST NOT 內含檢查程式碼。

#### Scenario: AI 助理被要求驗證部署

- **WHEN** 使用者要求對某主機執行上線後檢查
- **THEN** 助理依 skill 以 tox 執行並回報失敗與警告

### Requirement: 測試素材位置

`tests/data/` SHALL 含 `海岸阿美語-曾玉蘭-個人生命史-短.mp4` 與由它轉出的 16 kHz 單聲道 `.mp3` 各一份，並有 README 記載轉檔指令。`tests/production_tests/` 與 `tests/grafana-k6/` MUST 都使用此目錄，repo 內 MUST NOT 有第二份副本。

#### Scenario: k6 仍能讀到素材

- **WHEN** 在 `tests/grafana-k6/` 執行 `k6 run asr.js`
- **THEN** `open("../data/...")` 成功讀取 mp4
