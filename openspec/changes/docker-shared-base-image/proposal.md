## Why

三個 GPU 服務（asr、mt、tts）的 Dockerfile 以 `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` 為 base，但 PyPI 的 torch wheel 一律自帶 `nvidia-*-cu12` 套件，執行時只會載入 pip 那一份，base image 裡 1.9 GiB（壓縮後）的 CUDA 12.2 與 cuDNN 8 完全沒被使用；asr 的 ctranslate2 4.8 甚至需要 cuDNN 9，現在能跑是靠 whisperx 先 import torch 把 pip 那份 cuDNN 載進來。三個服務的 torch 版本各不相同（asr 2.8.0、mt 2.9.1、tts 2.7.0），每個 image 各自多帶 3 到 4 GiB 的 torch 與 nvidia wheel，layer 無法共用；Travis 每次 push 在兩個 stage 各從零 build 一次四個 image，沒有任何 cache。此外該 base image tag 最後重建於 2023-11，Ubuntu 套件的安全修補停在當時，而四個 Dockerfile 都安裝的 Node.js 20 是從 Hugging Face Spaces build log 抄來的，本專案沒有開啟 Gradio SSR，從未使用。

## What Changes

- **GPU 服務改用 Python 官方 image**：asr、mt、tts 不再以 `nvidia/cuda` 為 base，CUDA 由 pip 的 nvidia wheel 提供，GPU 存取維持由 nvidia-container-toolkit 注入驅動。
- **`common/Dockerfile` 擴充為三個 stage**：`base`（`python:3.10-slim` + apt + nonroot 使用者）、`gpu`（base + torch 系列）、`files`（現有的 `FROM scratch` 共用檔案，放在檔案最下面，維持獨立不疊在 base 上）。分別以 `--target` 打成 `formosan-ai-base`、`formosan-ai-gpu`、`formosan-ai-common`。
- **新增 `common/requirements.in` 與帶 hash 的 `common/requirements.txt`**：只宣告 torch、torchaudio、torchvision，作為 gpu stage 的安裝清單，也是三個 GPU 服務的版本上限來源。
- **torch 版本統一為 2.8.0**：受 whisperx 3.8.x 的 `torch~=2.8.0` 硬 pin 限制。mt 在 `requirements.in` 明確 pin，tts 移除 `torch==2.7.0` 與 `torchcodec==0.5.0` 的 pin，三個服務以 `-c common/requirements.txt` 重新編譯 lock 檔。
- **鎖定檔一致性檢查**：新增 tox env 比對 asr、mt、tts 的 `requirements.txt` 與 `common/requirements.txt` 的共同套件版本；服務 Dockerfile 在 pip install 之前做同樣比對，本機 build 也會擋。
- **移除未使用的 apt 套件**：Node.js 與 nodesource 安裝腳本、curl、cmake、rsync、git、git-lfs、libsm6、libxext6、libgl1-mesa-glx。保留 ffmpeg（whisperx shell out、gradio 與 pydub 轉檔、torchcodec dlopen libav 皆需要）。以 image 內 ldd 掃描與 strace smoke 驗證。
- **Travis 加入 buildx registry cache**：以 Docker Hub 上的 cache tag 做 `--cache-from`，main 分支加 `--cache-to`；build 順序改為 common 三個 target 先、服務後；合併目前重複的兩段 build。
- **hash 鎖定範圍**：asr 維持 `--require-hashes`，gpu stage 新增；mt、tts、asr-kaldi 本次不導入，由團隊試用一段時間後決定。
- **文件更新**：README 的 build 步驟、套件更新流程與 `openspec/config.yaml` 的 Docker 架構描述。

## 非目標

- 不升級 Python 版本（3.10 於 2026-10 結束維護，另開 change 處理，因為要重編四個服務的 lock 檔）。
- 不改用 Docker Hardened Images 或 Chainguard；本次以 `python:slim` 加 digest pin 自行 harden。
- 不為 mt、tts、asr-kaldi 導入 `--require-hashes`；tts 的 `f5-tts @ git+...` 也不改成 PyPI 版本。
- 不以 PyAV 取代系統 ffmpeg，不改 whisperx 的音訊載入方式。
- 不修改任何 `app.py`、模型、族語方言設定、`model_cache_*` volume 與 compose 的網路或環境變數。
- 不重新啟用 CD。

## Capabilities

### New Capabilities

- `docker-image-layering`：共用 base image 的分層規則、服務 image 的 FROM 與 apt 套件清單、build 順序與 CI cache 行為。
- `python-dependency-lock`：`common/requirements.txt` 作為 torch 系列版本來源、各服務 lock 檔與其一致的規則、hash 鎖定範圍、更新流程與一致性檢查。

### Modified Capabilities

（無。既有 `mt-speech-playback`、`production-tests`、`tts-language-selector`、`tts-synthesize-api` 的需求不受影響。）

## Impact

- **common/**：`Dockerfile` 改為三個 stage；新增 `requirements.in`、`requirements.txt`、版本比對腳本。需重建 `formosan-ai-common`，並新增 `formosan-ai-base`、`formosan-ai-gpu` 兩個本機 image。
- **asr/**：`Dockerfile` 改 `FROM formosan-ai-gpu`，移除 apt 段；`requirements.txt` 以 `-c common/requirements.txt` 重編（torch 維持 2.8.0，內容應幾乎不變）。
- **mt/**：`requirements.in` pin `torch==2.8.0`；`Dockerfile` 同 asr；`requirements.txt` 重編，torch 由 2.9.1 降為 2.8.0。
- **tts/**：`requirements.in` 移除 torch 與 torchcodec 的 pin；`Dockerfile` 同 asr；`requirements.txt` 重編，torch 由 2.7.0 升為 2.8.0、torchcodec 升為 0.7.x，需實跑合成驗證。
- **asr-kaldi/**：`Dockerfile` 改 `FROM formosan-ai-base`，移除 apt 段；requirements 不變。
- **deploy/**：`docker-compose-production.yml` 不變；正式機 pull 時 torch 層只需下載一次，之後只在 torch 升版時重拉。
- **CI**：`.travis.yml` 改 build 順序、加 buildx cache、合併重複 build；新增 tox env。
- **文件**：`README.md`、`openspec/config.yaml`。
- **族語方言相容性**：不動應用程式碼與設定檔，所有族語方言的辨識、合成、翻譯行為不變；tts 的 torch 升版是唯一需要以實際合成驗證的項目。
