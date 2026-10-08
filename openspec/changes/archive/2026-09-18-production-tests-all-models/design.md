## Context

現有 `tests/production_tests/check_mt_tts_playback.py`（2026-09-14）以 `gradio_client` 對已部署主機呼叫 API，循序執行、彙總失敗、exit code 0 或 1。`tests/grafana-k6/` 以手寫的 Gradio SSE 協定（`gradio-helpers.js`）量四個模型的延遲，素材在 `tests/grafana-k6/testing_data/`。

2026-09-18 對測試機 `https://tshi5v100.ithuankhoki.tw/` 的實測：

```
服務        呼叫                                首次    第二次  兩次輸出
asr         /generate_srt  mp4  ami-x-pswl      8.3s    1.1s   相同（5 個 cue，含時間戳、族語、華語）
asr-kaldi   /automatic_speech_recognition mp3   12.6s   4.2s   相同
mt          /translate  ami_Coas→zho_Hant        4.8s           相同
mt          /translate_1 zho_Hant→ami_Coas        1.3s           相同
tts         /synthesize 阿美_海岸                14.2s          位元不同；2.28s、109612 bytes、24 kHz 相同
mt→tts      /synthesize                          3.1s

四個服務的 /config 都回 200；API 名稱與 ?view=api 一致。
```

語別來源：asr `LANGUAGE_GROUPS` 16 族 42 語別；asr-kaldi `models.yaml` 第一個模型的 `dialect_mapping` 16 族；mt `FORMOSAN_LANGUAGES_MAP` 42 語別；tts `refs.yaml`。

## Goals / Non-Goals

**Goals:**

- 一個指令對指定主機驗完四個模型與頁面，2 到 3 分鐘內完成，任何人與 AI 助理都能跑。
- 模型換版不需要改測試。
- 改前改後可比對：JSON 報告含每項結果與耗時。
- 拿掉內容比對後仍能發現 GPU 未被使用。

**Non-Goals:**

- 見 proposal「非目標」。

## Decisions

### 1. 只驗格式與非空，不比對內容

模型會換版，golden 比對會讓每次模型升級都要改測試，而這套檢查的用途是基礎建設與部署驗收。判準：

| 服務 | 通過條件 |
|---|---|
| asr | 字串含至少一個 cue；每個 cue 有 `HH:MM:SS,mmm --> HH:MM:SS,mmm`、`族語：`、`華語：` 三行；阿美時族語行非空 |
| asr-kaldi | 回傳字串；阿美時非空 |
| mt | 兩個方向都回傳字串；阿美時非空且不等於輸入 |
| tts | 檔案可被 `wave` 開啟、24 kHz、長度大於 0.5 秒、RMS 高於門檻（以 `array` 計算，不加套件） |
| 頁面 | 四個服務的 `/`、`/config` 與 `common_static` 下的 favicon、common.css 回 200 |

**替代方案**：golden 逐字比對。asr、asr-kaldi、mt 都可重現，技術上可行，但與「模型會換」的維護模式衝突，使用者已決定不採用。

### 2. 語別抽樣：阿美必測，其他隨機 3 個，`--all-languages` 掃全部

素材只有海岸阿美語。阿美語別驗證模型真的有辨識或翻譯；其他語別把同一段素材送進去，目的是驗證語別代碼表與模型載入沒壞，所以只要求呼叫成功、回傳型別正確，空字串印警告不算失敗。asr-kaldi 換族別模型辨識 10 秒阿美語很可能回空字串，這是合理行為。

抽樣用 `random.SystemRandom().sample`，每次印出抽到的語別，與現有腳本一致。

### 3. 暖機一次、第二次計時、超過門檻只警告

首次呼叫含模型載入與 cuDNN 自動調校，比穩態慢 5 到 8 倍。每個服務先以阿美呼叫一次不計時，之後所有呼叫都計時；阿美的第二次呼叫與門檻比較。門檻寫在 `timing.py` 頂端，以 2026-09-18 測試機穩態耗時放大約 10 倍：asr 15 秒、asr-kaldi 30 秒、mt 10 秒、tts 20 秒。CPU fallback 時 asr 會從 1 秒變數十秒，警告一定出現；門檻只警告不失敗，避免測試機忙碌時誤報。

### 4. 單一入口 `run_all.py`，沿用現有腳本的函式

```
run_all.py
  ├─ 解析 --services asr,asr-kaldi,mt,tts,pages  --all-languages  --report path
  ├─ pages   check_pages.py       urllib，不需 gradio_client
  ├─ asr     check_asr.py         Client(BASE_URL/sapolita/)
  ├─ asr-kaldi check_asr_kaldi.py Client(BASE_URL/sapolita-kaldi/)
  ├─ mt      check_mt.py          Client(BASE_URL/kari-seejiq-tnpusu-ai-hmjil/)
  ├─ tts     check_tts.py         Client(BASE_URL/hnang-kari-ai-asi-sluhay/)
  ├─ mt→tts  check_mt_tts_playback.run_tts_checks / run_end_to_end_checks（現有）
  └─ 彙總 failures → exit 0/1；--report 寫 JSON
```

每個 `check_*.py` 提供 `run(client, options, failures) -> list[result]`，共用現有的 `check(name, fn, failures)` 與 `connect()`。`check_mt_tts_playback.py` 保持可獨立執行，`run_all.py` import 它而不複製。

**替代方案**：一個服務一支獨立腳本各自執行。多次連線與多份彙總，tox 入口也要串多個指令。

### 5. JSON 報告格式

```json
{"base_url": "...", "commit": "<git rev-parse HEAD>", "started_at": "...",
 "checks": [{"service": "asr", "name": "ami-x-pswl", "ok": true, "seconds": 1.1, "note": ""}],
 "warnings": ["asr 阿美 第二次呼叫 18.2s 超過門檻 15s"], "failed": 0}
```

`--report` 未指定時不寫檔。改前改後各存一份到 `tests/production_tests/results/<日期>-<commit>.json`，該目錄加進 `.gitignore`；要留存就手動 commit。

### 6. tox 入口與 `BASE_URL`

```ini
[testenv:production_tests]
deps = -r tests/production_tests/requirements.txt
passenv = BASE_URL
commands = python tests/production_tests/run_all.py {posargs}
```

`BASE_URL` 維持環境變數（與現有腳本相同），其餘參數走 `{posargs}`。tox 會建立獨立 venv，不污染 `venv/`。

### 7. Claude Code skill

`.claude/skills/production-tests/SKILL.md`，frontmatter 與現有 openspec skill 相同格式。內容：何時執行（部署前後、Docker 或 requirements 變動後）、指令（tox 與 `BASE_URL`）、參數、預估耗時與 GPU 消耗、如何解讀警告與失敗、改前改後如何用 `--report` 比對。skill 只描述流程，不含程式碼。

### 8. 測試相關的東西集中在 `tests/`，素材在 `tests/data/`

```
tests/
├── check_lock_consistency.py   GPU 套件版本一致性（tox -e lockcheck）
├── data/                       共用素材：mp4、mp3、README（規格與轉檔指令）
├── grafana-k6/                 延遲測試
└── production_tests/           上線後功能檢查
```

k6 的 `open()` 以腳本所在目錄為基準，改成 `../data/`。Python 腳本以 `Path(__file__).resolve().parents[2]` 取 repo 根目錄。


## Risks / Trade-offs

- [不比對內容，模型輸出退化不會被抓到] → 這是使用者的決定；模型品質由另一套流程負責。耗時門檻補足 GPU 未使用的情況。
- [測試機忙碌時耗時超過門檻] → 只警告不失敗；報告內有實際秒數可人工判斷。
- [非阿美語別回空字串被忽略，若語別代碼表壞掉但服務仍回空字串] → 語別代碼錯誤時 Gradio 會回 `AppError` 或 KeyError，不是空字串；asr 的 Radio 有 choices 驗證。
- [tox 的 venv 每次重建耗時] → tox 會快取 env，只在 requirements 變動時重建。
- [`run_all.py` import `asr/languages.py` 與 `mt/formosan_languages.py`，不可 import 各服務的 `app.py`] → 與現有腳本的規則相同，README 註明。
- [檢查會消耗測試機 GPU 約 30 次呼叫] → README 與 skill 註明避開尖峰。

## Migration Plan

單一 PR，合併後對測試機跑一次 `tox -e production_tests --report`，結果作為 `docker-shared-base-image` 的「改前」基準。無回滾需求，不影響服務。

## Open Questions

- 頁面檢查的 `common_static` URL 清單：以 `common/static/` 內 favicon 與 css 為準，實作時確認 Gradio 的靜態路徑格式。
- tts 非靜音的 RMS 門檻：以測試機合成的 wav 實測值決定。
