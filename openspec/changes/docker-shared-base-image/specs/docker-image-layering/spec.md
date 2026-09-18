## ADDED Requirements

### Requirement: 共用 image 由 common/Dockerfile 的三個 stage 產生

`common/Dockerfile` SHALL 包含 `base`、`gpu`、`files` 三個 build stage，分別以 `--target` 打成 `formosan-ai-base`、`formosan-ai-gpu`、`formosan-ai-common`。`base` MUST 以 digest pin 的 `python:3.10-slim` 為 FROM，安裝 apt 套件並建立 uid 1000、gid 1000 的 `nonroot` 使用者；`gpu` MUST `FROM base` 並以 `--require-hashes` 安裝 `common/requirements.txt`，且 MUST 將 `tests/check_lock_consistency.py`（以具名 build context `tests=./tests` 帶入）與 `common/gpu-constraints.txt` 複製到 `/opt/formosan-ai/`；`files` MUST 維持 `FROM scratch`、不依賴 `base`，且位於檔案最下方。

#### Scenario: 以 target 建出三個 image

- **WHEN** 依序執行 `docker build --target base -t formosan-ai-base ./common`、`--target gpu -t formosan-ai-gpu`、`--target files -t formosan-ai-common`
- **THEN** 三個 image 都建立成功，`formosan-ai-common` 的內容與變更前相同（`/app/utils.py`、`/app/colors.py`、`/app/common_static/`）

#### Scenario: 共用檔案變動不影響 gpu image

- **WHEN** 修改 `common/utils.py` 後重新 build `gpu` 與 `files` 兩個 target
- **THEN** `formosan-ai-gpu` 的 image digest 不變，只有 `formosan-ai-common` 改變

### Requirement: 服務 image 以共用 image 為 FROM，且不自行安裝 apt 套件或 torch

asr、mt、tts 的 Dockerfile SHALL `FROM formosan-ai-gpu`；asr-kaldi 的 Dockerfile SHALL `FROM formosan-ai-base`。服務 Dockerfile MUST NOT 執行 `apt-get install`，唯一例外是 tts：f5-tts 以 git URL 安裝，build 時需要 git，MUST 在同一個 `RUN` 內安裝 git、執行 `pip install`、再移除 git，不留在 image 裡。服務 DockerfileMUST NOT 在自己的 `pip install` 步驟重新安裝或變更 torch 系列、`nvidia-*-cu12`、triton 的版本。

#### Scenario: 三個 GPU 服務共用 torch layer

- **WHEN** build 完 asr、mt、tts 三個 image 後執行 `docker history` 或 `docker image inspect` 比對 layer
- **THEN** 三個 image 含有相同 digest 的 torch 安裝 layer，該 layer 只在磁碟上存在一份

#### Scenario: 服務 pip install 不重裝 torch

- **WHEN** build 任一 GPU 服務 image，觀察 `pip install -r requirements.txt` 的輸出
- **THEN** torch、torchaudio、torchvision、`nvidia-*-cu12`、triton 均顯示為已滿足（already satisfied），沒有被下載或重新安裝

### Requirement: GPU 存取由 nvidia-container-toolkit 提供，CPU 環境可正常啟動

共用 image MUST NOT 以 `nvidia/cuda` 為 base。`base` stage SHALL 設定 `NVIDIA_VISIBLE_DEVICES=all` 與 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`，CUDA 函式庫由各服務 requirements 中的 `nvidia-*-cu12` wheel 提供。

#### Scenario: 正式機保留 GPU 時使用 CUDA

- **WHEN** 以 `deploy/docker-compose-production.yml` 的 GPU reservation 啟動 asr、mt、tts
- **THEN** 容器內 `torch.cuda.is_available()` 為 True，`torch.backends.cudnn.version()` 回傳 9 開頭的版本，asr 的 `ctranslate2.get_cuda_device_count()` 大於 0，模型載入到 cuda

#### Scenario: 無 GPU 的環境退回 CPU

- **WHEN** 在沒有 GPU 或未安裝 nvidia-container-toolkit 的主機以 `docker compose up` 啟動同一個 image
- **THEN** 容器正常啟動，`torch.cuda.is_available()` 為 False，服務以 CPU 執行（與變更前行為相同）

### Requirement: apt 套件只安裝有使用者的項目

`base` stage 的 apt 安裝清單 SHALL 只包含 `ffmpeg` 及其遞移依賴。共用與服務 image MUST NOT 安裝 Node.js、nodesource 安裝腳本、curl、cmake、rsync、git、git-lfs、libsm6、libxext6、libgl1-mesa-glx。

#### Scenario: image 內沒有 Node.js

- **WHEN** 在任一服務容器內執行 `which node npm git rsync cmake curl`
- **THEN** 全部找不到，且 Gradio 網頁與 API 行為與變更前相同（SSR 仍為關閉）

#### Scenario: 動態 smoke 沒有呼叫被移除的程式

- **WHEN** 以 `strace -f -e trace=execve,openat` 執行服務並完成：上傳非 wav 音檔、麥克風錄音、一次辨識或合成、冷啟動模型下載
- **THEN** execve 清單只出現 `ffmpeg`（asr、tts、asr-kaldi、mt 依功能而定），沒有 git、rsync、cmake、node；openat 的 `/usr/lib` 共享函式庫都屬於 ffmpeg 或 base image 既有套件

#### Scenario: wheel 的共享函式庫都能解析

- **WHEN** 在 gpu image 內對 site-packages 所有 `.so` 執行 `ldd`
- **THEN** 沒有 `not found` 的項目，且解析到的系統套件集合是 `ffmpeg` 遞移依賴與 base image 既有套件的子集

### Requirement: CI 依序 build 並使用 registry cache

建置 SHALL 由 repo 根目錄的 `docker-bake.hcl` 定義 `base`、`gpu`、`files`、`asr`、`asr-kaldi`、`tts`、`mt` 七個 target。服務 target MUST 以 `contexts` 把 `formosan-ai-gpu`、`formosan-ai-base`、`formosan-ai-common` 對應到 `target:gpu`、`target:base`、`target:files`，MUST NOT 依賴本機 image store（registry cache 需要的 `docker-container` driver 看不到本機 image）。變數 `CACHE` 為 `read` 時每個 target MUST 使用 `cache-from type=registry,ref=ithuan/formosan-ai:cache-<name>`；為 `readwrite` 時 MUST 另加 `cache-to type=registry,mode=max,ref=ithuan/formosan-ai:cache-<name>`；為空字串時不用 cache。`.travis.yml` 的兩個 build job SHALL 執行 `docker buildx bake -f docker-bake.hcl --load`，PR 的 job 設 `CACHE=read`，main 的 job 設 `CACHE=readwrite`。

#### Scenario: requirements 未變時不重新下載 torch

- **WHEN** 只修改 `asr/app.py` 後 push 到 main
- **THEN** asr 的 build 中 `gpu` 層與 asr 的 pip 層均為 CACHED，沒有從 PyPI 下載任何 wheel

#### Scenario: PR build 只讀 cache

- **WHEN** 來自 pull request 的 build 以 `CACHE=read` 執行
- **THEN** build 使用 `--cache-from` 讀取 cache，不執行 `--cache-to`，且不需要 push token

### Requirement: base image 以 digest 固定並由 Dependabot 更新

`common/Dockerfile` 的 `FROM python:3.10-slim` SHALL 附帶 `@sha256:` digest。`.github/dependabot.yml` SHALL 含 `docker` ecosystem 指向 `/common`。

#### Scenario: Debian 或 Python patch 重建後收到 PR

- **WHEN** Docker 官方重建 `python:3.10-slim`
- **THEN** Dependabot 提出更新 digest 的 PR，CI 的 build 與 lockcheck 通過後即可合併，不需改動 requirements

### Requirement: pip 只安裝 wheel，例外須明列

服務 Dockerfile 的 `pip install` SHALL 使用 `--only-binary=:all:`，只在 PyPI 沒有 wheel 的套件以 `--no-binary` 明列例外，避免 build 時執行未預期的 setup 腳本（SonarQube docker:S8541）。目前的例外為：asr `antlr4-python3-runtime`、`julius`；asr-kaldi `antlr4-python3-runtime`、`srt`；mt 無。tts 因 f5-tts 以 git URL 安裝必須 build，另有 4 個套件只有原始碼包，SHALL 不加 `--only-binary`，並在 Dockerfile 註明此為已審查、可接受的例外。

#### Scenario: 新增的依賴沒有 wheel

- **WHEN** mt 的 requirements 新增一個只有原始碼包的套件後 build
- **THEN** `pip install` 失敗並指出該套件，必須明確加入 `--no-binary` 例外才能通過

#### Scenario: tts 的例外有紀錄

- **WHEN** 檢視 `tts/Dockerfile`
- **THEN** pip install 那一段有註解說明為何不加 `--only-binary`，並列出需要從原始碼 build 的套件
