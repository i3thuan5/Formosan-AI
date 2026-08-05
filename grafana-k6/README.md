# 族語 AI 四個模型的延遲（latency）測試

用 [Grafana k6](https://k6.io/) 測量 `asr/`、`asr-kaldi/`、`tts/`、`mt/` 四個模型在**線上正式站**的回應延遲，
每個模型測 100 次，供後續改版比較。

| 文件 | 內容 |
| --- | --- |
| **README.md**（本檔） | 怎麼裝、怎麼跑、結果放在哪 |
| [METHOD.md](METHOD.md) | 測試規格：受測的線上 API、測試資料、測試方法 |
| [RESULTS.md](RESULTS.md) | 歷次測試結果與分析 |

## 一、如何執行

### 1-1 安裝需要的工具

本測試在 devcontainer 裡跑，這些工具**不在** [.devcontainer/](../.devcontainer/) 的設定裡，
**容器重建後需要重裝**：

```bash
# ffmpeg（轉測試資料用）、jq（讀 summary JSON 用）
$ sudo apt-get update
$ sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg jq

# k6（本測試用 v2.1.0；官方 apt repo 目前還沒有 v2，所以直接抓 binary）
$ TAG=v2.1.0
$ curl -sL "https://github.com/grafana/k6/releases/download/${TAG}/k6-${TAG}-linux-amd64.tar.gz" \
    -o /tmp/k6.tgz
$ tar xzf /tmp/k6.tgz -C /tmp
$ sudo install -m755 "/tmp/k6-${TAG}-linux-amd64/k6" /usr/local/bin/k6

# 確認
$ ffmpeg -version | head -1
$ jq --version
$ k6 version
```

> 想抓最新版的話，把 `TAG` 換成：
> `TAG=$(curl -s https://api.github.com/repos/grafana/k6/releases/latest | jq -r .tag_name)`

若不想裝 k6，也可以用 docker（見 1-3）。

### 1-2 跑測試

```bash
$ cd grafana-k6/

# 四個模型全跑，每個 100 次（約 20 分鐘）
$ ./run-latency.sh

# 先小跑測試看看
$ ITERATIONS=5 ./run-latency.sh

# 只跑一個模型
$ k6 run -e ITERATIONS=100 ./tts.js

# 把結果整理成比較用的表格
$ ./results-table.sh
```

環境變數：

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `ITERATIONS` | `100` | 每個模型跑幾次 |
| `BASE_URL` | `https://ai-labs.ilrdf.org.tw` | 換成別的站台（例如測試機） |
| `K6` | `k6` | k6 執行檔的路徑 |

### 1-3 用 docker 跑（免裝 k6）

```bash
$ docker run --rm \
	--mount type=bind,src=$(pwd),dst=/scripts -w /scripts \
	grafana/k6:latest run -e ITERATIONS=100 /scripts/tts.js
```

## 二、結果放在哪

`run-latency.sh` 會產生：

- `results/{模型}-summary.json`：k6 完整的 summary，可供後續程式讀取。
- `results/{模型}-run.log`：該次執行的畫面輸出。
- `results/run-all.log`：四個模型合在一起的紀錄。

歷次結果與分析寫在 [RESULTS.md](RESULTS.md)。

## 三、檔案

| 檔案 | 說明 |
| --- | --- |
| `METHOD.md` | 測試規格：受測 API、測試資料、測試方法 |
| `RESULTS.md` | 歷次測試結果與分析 |
| `gradio-helpers.js` | 共用的 Gradio API 客戶端、metric 定義、summary 輸出 |
| `asr.js` | asr（Whisper）測試 |
| `asr-kaldi.js` | asr-kaldi（Kaldi／Vosk）測試 |
| `tts.js` | tts（F5-TTS）測試 |
| `mt.js` | mt（NLLB）測試 |
| `run-latency.sh` | 四個模型依序執行 |
| `results-table.sh` | 把 summary JSON 整理成比較表格 |
| `testing_data/` | 測試資料 |
| `results/` | 測試結果 |
