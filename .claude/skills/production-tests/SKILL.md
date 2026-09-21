---
name: production-tests
description: 對已部署的主機（正式機或測試機）執行上線後檢查，確認 asr、asr-kaldi、mt、tts 四個服務與共用靜態檔都正常。部署前後、改動 Dockerfile 或 requirements 後、想確認某台主機是否健康時使用。
license: MIT
compatibility: 需要能連到受測主機，並會消耗該主機的 GPU。
metadata:
  author: Formosan-AI
  version: "1.0"
---

對**已部署的主機**實際呼叫 API，確認四個服務串得起來。
細節見 [tests/production_tests/README.md](../../../tests/production_tests/README.md)。

## 什麼時候跑

- 部署到測試機或正式機的前後（改前改後各一次，用來比對）。
- 改動任一 `Dockerfile`、`requirements.txt` 或 base image 之後。
- 有人回報某個服務怪怪的，要快速確認是哪一段壞掉。

## 怎麼跑

在 repo 根目錄：

```bash
# 測試機
BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests

# 正式站（BASE_URL 的預設值）
tox -e production_tests

# 掃全部語別
BASE_URL=... tox -e production_tests -- --all-languages

# 只跑部分服務
BASE_URL=... tox -e production_tests -- --services asr,tts

# 寫報告，供改前改後比對
BASE_URL=... tox -e production_tests -- --report tests/production_tests/results/$(date +%F)-$(git rev-parse --short HEAD).json
```

`--` 後面的參數會原封不動傳給 `run_all.py`。

## 參數

| 參數 | 用途 |
| --- | --- |
| `--services` | 逗號分隔，可選 `pages`、`asr`、`asr-kaldi`、`mt`、`tts`，預設全部 |
| `--all-languages` | 每個服務掃全部語別，預設是素材語別加隨機 3 個 |
| `--report <path>` | 把每項結果與耗時寫成 JSON |

## 耗時與 GPU

- 預設約 2 到 3 分鐘，`--all-languages` 約 10 分鐘。
- **會實際使用受測主機的 GPU**，請避開尖峰時段。
- 檢查循序執行，不併發。

## 怎麼看結果

- **全部通過** exit code 0；**任一項失敗** exit code 1，失敗項目列在最後。
- **失敗**是真的有問題：服務連不上、回傳格式不對、阿美語沒有辨識或翻譯出內容、
  音檔不是 24 kHz wav 或像靜音、頁面或共用靜態檔不是 200。
- **警告**不算失敗，有兩種：
  - 非阿美語別沒有辨識或翻譯出內容。素材只有海岸阿美語，拿去問別族的模型回空是正常的。
  - 暖機後耗時超過門檻。這通常表示 **GPU 沒有被使用**，值得追查；也可能只是受測主機忙碌。
- 檢查**不比對**辨識或翻譯的文字，也不比對音檔內容，因為模型會換版。
  要比對模型品質請用別的流程。

## 改前改後比對

1. 部署前跑一次，加 `--report tests/production_tests/results/<日期>-部署前.json`。
2. 部署後用同樣參數再跑一次，存成另一個檔名（例如 `<日期>-部署後.json`）。
3. 比對兩份 JSON：`checks` 的名稱集合應相同，`ok` 不該從 true 變 false，
   `seconds` 不該有數倍的劣化。

## 注意

- **語別表和測試句讀自本地檔案**：語別表讀自本地的
  `asr/languages.py`、`asr-kaldi/configs/models.yaml`、`mt/formosan_languages.py`，
  測試句讀自本地的 `tts/configs/refs.yaml`，本地版本和受測主機部署的版本不同時，結果可能對不上。
- 腳本**不可** import 各服務的 `app.py`，那會在本機載入模型。
- 部署順序：tts 要先部署新版，mt 再部署。
