# 上線後檢查

對**已部署的主機**（正式機或測試機）實際呼叫 API，確認服務之間串得起來。
和 [grafana-k6/](../grafana-k6/) 的差別：這裡檢查「功能對不對」，k6 測「延遲多少」。

**只驗「服務活著、格式對、有內容」，不比對辨識或翻譯的文字，也不比對音檔內容。**
模型會換版，比對內容會讓每次模型升級都要改測試。模型品質請用別的流程評估。

| 腳本 | 檢查對象 |
| --- | --- |
| `run_all.py` | 單一入口，四個模型加頁面 |
| `check_mt_tts_playback.py` | MT「華語 ⮕ 族語」合成語音與 TTS `/synthesize` API，也可獨立執行 |

## 一、安裝

用 tox 的話不用自己裝，tox 會建好環境：

```bash
BASE_URL=https://<受測主機> tox -e production_tests
```

要直接執行腳本才需要：

```bash
pip install -r tests/production_tests/requirements.txt
```

套件版本寫在 `requirements.in`，改完後用 `pip-compile tests/production_tests/requirements.in` 更新 `requirements.txt`。
`gradio_client` 的版本要和 `mt/requirements.txt` 一致，測到的呼叫方式才和 mt 相同。

## 二、執行

在 repo 根目錄執行。`--` 後面的參數會原封不動傳給 `run_all.py`：

```bash
# 檢查測試機
BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests

# 檢查正式站（BASE_URL 的預設值）
tox -e production_tests

# 掃全部語別
BASE_URL=... tox -e production_tests -- --all-languages

# 只跑部分服務
BASE_URL=... tox -e production_tests -- --services asr,tts

# 寫報告，供改前改後比對
BASE_URL=... tox -e production_tests -- \
    --report tests/production_tests/results/$(date +%F)-$(git rev-parse --short HEAD).json
```

也可以不透過 tox 直接跑：

```bash
BASE_URL=... python tests/production_tests/run_all.py --services asr
```

| 參數 | 用途 |
| --- | --- |
| `--services` | 逗號分隔，可選 `pages`、`asr`、`asr-kaldi`、`mt`、`tts`，預設全部 |
| `--all-languages` | 每個服務掃全部語別，預設是素材語別加隨機 3 個 |
| `--report <path>` | 把每項結果與耗時寫成 JSON，未指定就不寫檔 |

受測主機要和正式站一樣，以下面的路徑提供服務：

| 服務 | 路徑 |
| --- | --- |
| asr | `{BASE_URL}/sapolita/` |
| asr-kaldi | `{BASE_URL}/sapolita-kaldi/` |
| tts | `{BASE_URL}/hnang-kari-ai-asi-sluhay/` |
| mt | `{BASE_URL}/kari-seejiq-tnpusu-ai-hmjil/` |

## 三、檢查項目

測試素材是 [tests/data/](../data/) 的海岸阿美語影片與音檔。

| 服務 | 檢查內容與判準 |
| --- | --- |
| `pages` | 四個服務的 `/`、`/config`（要是 JSON）、`/favicon.ico`、`/theme.css`、`common_static` 的 logo 都回 200。抓得到 image 少複製 common 檔案、反向代理路徑設錯 |
| `asr` | 送 mp4 到 `/generate_srt`。回傳要能拆成 cue，每個 cue 有序號、時間戳、`族語：`、`華語：` 四行；海岸阿美語的族語行要非空 |
| `asr-kaldi` | 送 mp3 到 `/automatic_speech_recognition`。回傳要是字串；阿美族別要非空 |
| `mt` | `/translate`（族語 ⮕ 華語）與 `/translate_1`（華語 ⮕ 族語）。阿美_海岸兩個方向都要非空且不等於原文 |
| `tts` | `/synthesize` 的音檔要是 24 kHz wav、長於 0.5 秒、RMS 高於門檻（非靜音）；另含既有的 API 約定、語別合成與已知 bug 回歸 |
| `mt` → `tts` | 端到端：先切族別再呼叫 MT `/synthesize`，要回傳音檔。這條路才會經過「mt 容器 → 公開網域 → tts」，可以抓到 hairpin NAT 不通、`SAPOLITA_WEBSITE_HOST` 設錯、tts 還沒部署新版 |

### 語別抽樣

asr、asr-kaldi、mt 三個服務**固定包含素材的阿美語別**，再隨機抽 3 個其他語別；
加 `--all-languages` 就掃全部。每次執行都會印出抽到哪些。

素材只有海岸阿美語，所以：

- **阿美語別**要求真的有辨識或翻譯出內容，沒有就是失敗。
- **其他語別**只要求呼叫成功、回傳型別正確。拿阿美語音檔去問別族的模型，
  回空字串是合理的，只印警告不算失敗。
- 語別代碼表壞掉時服務會回 `AppError`，那才算失敗。

### 暖機與耗時門檻

第一次呼叫含模型載入與 cuDNN 自動調校，比穩態慢 5 到 8 倍，所以每個服務
**先呼叫一次不計時**，之後才計時。阿美語別的耗時超過門檻（定義在 `timing.py`）時印警告。

拿掉內容比對之後，**耗時是唯一能發現「GPU 沒被用到」的訊號**：CPU fallback 會讓 asr
從 1 秒變數十秒，門檻一定會被超過。門檻只警告不算失敗，避免受測主機忙碌時誤報。

## 四、怎麼看結果

- 全部通過 exit code `0`，任一項失敗為 `1`，失敗項目會在最後列出。
- **失敗**是真的有問題。**警告**有兩種：非阿美語別沒內容（正常），或耗時超過門檻（值得追查）。

### 改前改後比對

1. 部署前跑一次，加 `--report tests/production_tests/results/<日期>-<commit>.json`。
2. 部署後用同樣參數再跑一次，存成另一個檔名。
3. 比對兩份 JSON：`checks` 的名稱集合應相同，`ok` 不該從 `true` 變 `false`，
   `seconds` 不該有數倍的劣化。

`tests/production_tests/results/` 已列入 `.gitignore`，要留存就手動 commit。

## 五、注意事項

- **會實際使用受測主機的 GPU**。2026-09-18 對測試機實測：

  | 執行方式 | 檢查項數 | 耗時 |
  | --- | --- | --- |
  | 預設抽樣 | 50 | 約 2 分鐘 |
  | `--all-languages` | 214 | 約 8 分鐘 |

  請避開尖峰時段。檢查循序執行，不併發。
- **請用和受測主機部署版本相同的 commit 執行**：語別表讀自本地的
  `asr/languages.py`、`asr-kaldi/configs/models.yaml`、`mt/formosan_languages.py`，
  測試句讀自本地的 `tts/configs/refs.yaml`，版本不同時結果可能對不上。
- 腳本**不可** import 各服務的 `app.py`，那會在本機載入模型。
- **部署順序**：tts 要先部署新版，mt 再部署。只部署 tts 時，tts 的檢查應該通過、
  mt → tts 端到端失敗。
