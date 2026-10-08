## Context

四個服務各有一個 Dockerfile，apt 段落幾乎逐字相同但各自 build。三個 GPU 服務以 `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` 為 base，asr-kaldi 已是 `python:3.10-slim`。共用檔案由 `common/Dockerfile`（`FROM scratch`）打成 `formosan-ai-common`，服務以 `COPY --from=formosan-ai-common` 取用。

探索階段（2026-09-18）查證的事實：

```
壓縮後大小
  nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04   1.93 GiB   最後重建 2023-11-21
  python:3.10-slim                                  45 MiB   最後重建 2026-09-02
  asr  torch 2.8.0 + nvidia-*-cu12 (12.8/cuDNN 9.10) + triton   3.70 GiB
  mt   torch 2.9.1 + nvidia-*-cu12 (12.8/cuDNN 9.10) + triton   3.78 GiB
  tts  torch 2.7.0 + nvidia-*-cu12 (12.6/cuDNN 9.5)  + triton   2.83 GiB

版本限制
  whisperx 3.8.7rc1（asr）  torch~=2.8.0, torchaudio~=2.8.0, torchvision~=0.23.0, torchcodec<0.8
  transformers 5.17.0（mt） torch>=2.5
  f5-tts（tts）             torch>=2.0, torchaudio>=2.0, torchcodec 不限
  ctranslate2 4.8.2（asr）  需 cuDNN 9（base image 只有 cuDNN 8）

Node.js
  common/utils.py 的 demo.launch() 未傳 ssr_mode，全 repo 無 GRADIO_SSR_MODE；
  Node 是 2025-06-11「照Hugging Face build log 做」抄進來的，HF Spaces 才會自動開 SSR。

apt 套件使用者（repo 程式碼本身無 subprocess）
  ffmpeg    whisperx.load_audio shell out；gradio/pydub 轉檔；torchcodec dlopen libav*
  其餘      無任何使用者；四個服務的 requirements 都沒有 opencv
```

限制：Travis 每個 job 是全新 VM，`cache:` 只能存目錄；Docker Hub 帳號已有 pull 與 push 兩組 token；正式機以 `docker compose pull` 更新，`pull: always`。

## Goals / Non-Goals

**Goals:**

- 三個 GPU 服務的 image 各少 1.9 GiB 的無用 CUDA layer。
- torch 與 nvidia wheel 那 3.7 GiB 在四個 image 之間只存一份、正式機只拉一次，只在 torch 升版時重建。
- 日常只改 `app.py` 的 CI build 不再重新下載 torch。
- torch 系列的版本只有一個來源，任何服務與它不一致時 CI 與本機 build 都會失敗。
- 拿掉從未使用的 Node.js 與其他 apt 套件，並以可重複的方法證明哪些套件真的需要。

**Non-Goals:**

- 見 proposal「非目標」：不升 Python、不換 hardened image、不擴大 hash 範圍、不改 f5-tts 來源、不動應用程式碼。

## Decisions

### 1. GPU 服務的 base 改為 `python:3.10-slim`，CUDA 全部由 pip 的 nvidia wheel 提供

```
之前                                     之後
┌───────────────────────────┐            ┌───────────────────────────┐
│ app                       │            │ app                       │
│ pip: torch + nvidia 3.7G  │ ← 實際載入 │ pip: torch + nvidia 3.7G  │ ← 實際載入
│ nvidia/cuda 12.2 + cuDNN8 │ ← 沒人用   │ python:3.10-slim  45M     │
│   1.9G                    │            └───────────────────────────┘
└───────────────────────────┘
GPU 存取兩者相同：nvidia-container-toolkit 在 docker run 時注入 libcuda 等驅動函式庫
```

base stage 明確設定 `ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility`，這是 nvidia/cuda image 原本提供、toolkit 用來決定注入內容的兩個變數。

**替代方案：**

- 保留 `nvidia/cuda` 只把 tag 對齊 12.8 / cuDNN 9：兩份 CUDA 仍都在 image 裡，空間沒省，且每次 torch 升版都要同步換 tag。
- NGC `nvcr.io/nvidia/pytorch`：只有一份 CUDA，但 image 約 9 GiB 起跳，torch 是 `2.x.0a0+git` 預發行版號，requirements 裡的 `torch==2.8.0` 會讓 pip 再裝一份；torch、CUDA、Python 版本被月份 tag 綁死。
- 保留 `nvidia/cuda`、torch 用 `--no-deps` 移除 pip nvidia wheel：非官方支援，cuSPARSELt、NVSHMEM 等版本必須與 torch 編譯時完全一致，標準 image 不含這些。

### 2. `common/Dockerfile` 三個 stage，`files` 放最下面且不疊在 `base` 上

```
common/Dockerfile
  stage base   FROM python:3.10-slim@sha256:…        → formosan-ai-base   （asr-kaldi FROM）
               apt ffmpeg；nonroot uid/gid 1000；NVIDIA_* ENV
  stage gpu    FROM base                              → formosan-ai-gpu    （asr、mt、tts FROM）
               pip install --require-hashes -r common/requirements.txt
               COPY requirements.txt /opt/formosan-ai/gpu-requirements.txt   ← 給服務 build 時比對
  stage files  FROM scratch                           → formosan-ai-common （現有內容，不變）
               COPY colors.py utils.py static/
```

`files` 必須保持 `FROM scratch` 且獨立，因為 `gpu` 疊在 `base` 上：若 `utils.py` 進了 `base`，改一個字就會讓 `gpu` 那 3.7 GiB 的 layer digest 改變、整層重建重推。服務仍以 `COPY --from=formosan-ai-common` 取共用檔案，變動只影響服務 image 最上層。

**替代方案：** base 與 gpu 各開一個目錄。跟 common 慣例一致，但 apt 那段本來就是 gpu 層的一部分，拆開要維護兩處；且 `common/` 的意義從「共用檔案」擴大為「所有共用的東西」是合理的。

### 3. torch 系列統一 2.8.0，版本來源是 `common/requirements.in`，各服務以 `-c common/gpu-constraints.txt` 編譯

`common/requirements.in` 寫四行：`torch==2.8.0`、`torchaudio==2.8.0`、`torchvision==0.23.0`、`torchcodec==0.7.0`。nvidia-*、triton 由解析器帶出，落在 `common/requirements.txt`（含 hash，由 gpu stage 安裝）。

**constraints 不能直接用 `common/requirements.txt`**（實作時發現）：那份 lock 檔含 numpy 等小套件，而各服務對它們的需求互相衝突。實測 `uv pip compile` 的錯誤：

```
Because f5-tts==1.1.4 depends on numpy<=1.26.4 and numpy==2.2.6,
we can conclude that f5-tts==1.1.4 cannot be used.
```

asr 的 whisperx 要 `numpy>=2.1.0`，tts 的 f5-tts 要 `numpy<=1.26.4`，兩者無法共用同一個 numpy。所以另外產生 `common/gpu-constraints.txt`，只含真正需要共用 layer 的大型套件（torch 系列、`nvidia-*`、triton，共 19 個），由 `python common/gpu_constraints.py` 從 `common/requirements.txt` 抽出。服務以 `-c common/gpu-constraints.txt` 編譯，torch 系列一律被鎖成同一組；上游若不相容（例如 whisperx 硬 pin `torch~=2.8.0`），編譯直接失敗而不是悄悄裝第二份。

**torchcodec 必須進共用層**（實作時發現）：它的編譯擴充和 libtorch 版本綁定，但 PyPI metadata 沒宣告 torch 依賴，pip 抓不到不相容。tts 原本 pin `torchcodec==0.5.0`（配 torch 2.7），若不管它，uv 會保留 0.5 配上 torch 2.8，import 時才會炸。0.7 配 torch 2.8，與 whisperx 的 `torchcodec>=0.6,<0.8` 一致。

**小套件的版本可以不同**：gpu layer 會裝 numpy 2.2.6，tts 的 layer 再把它降成 1.26.4。numpy 只有十幾 MB，且 numpy 2 的 ABI 設計讓「用 numpy 2 編的模組」能在 numpy 1.x 上執行，所以 torch 2.8 搭 numpy 1.26.4 沒問題（tts 目前就是 torch 2.7 搭 numpy 1.26.4）。一致性檢查只比對兩邊都有的 GPU 套件。

實測各服務重編後的版本差異（2026-09-18）：

```
asr        0 個變動（原本就是 torch 2.8.0）
asr-kaldi  0 個變動（沒有 GPU 套件）
mt         3 個：torch 2.9.1→2.8.0、triton 3.5.1→3.4.0、nccl 2.27.5→2.27.3
           移除 nvidia-nvshmem-cu12（torch 2.9 才需要）
tts       18 個：torch 2.7.0→2.8.0、torchcodec 0.5→0.7.0，加全部 nvidia-*（12.6→12.8）
           transformers 維持 4.53.0、numpy 維持 1.26.4
```

版本規則：**由最嚴格的消費者決定**。目前是 whisperx 的 `torch~=2.8.0`，所以 mt 從 2.9.1 退回、tts 從 2.7.0 升上來。mt 的 transformers 只要求 `>=2.5`，沒有功能差異；tts 的 f5-tts 不限版本，但要實跑驗證。

編譯工具統一為 `uv pip compile --python-version 3.10 --python-platform linux --generate-hashes`（asr 已在用；比 pip-compile 快、且在任何機器都能編出 linux x86_64 的結果）。mt、tts、asr-kaldi 的 lock 檔改由 uv 產生但不加 `--generate-hashes`，維持 pip 可直接安裝。

**替代方案：**

- 統一 2.9.1：asr 編不出來，除非用 uv 的 `--override` 蓋過 whisperx 的 pin，等於替 whisperx 做相容性測試。
- 兩個 gpu image（2.8 與 2.9）：共用效果減半，與目標矛盾。

### 4. hash 範圍：asr 維持，gpu stage 新增，其他三個本次不導入

pip 的 `--require-hashes` 對已安裝的套件不再驗證，所以 asr 對 torch 的 hash 保證實際上由 gpu stage 提供；gpu stage 不帶 hash 的話，最大最值得驗的 3.7 GiB 反而沒驗。`common/requirements.in` 只有三行，帶 hash 的維護成本接近零，也是團隊體驗 hash 更新流程最輕鬆的地方。tts 的 `f5-tts @ git+…` 在 hash 模式下無法安裝，是 tts 暫不導入的另一個原因。

**SonarQube docker:S8544（2026-09-18）**：mt 與 asr-kaldi 的 `pip install` 沒有 `--require-hashes` 被標為資安熱點。實測兩者都能產生完整 hash 且版本不變，但維持原決定，在 SonarQube 接受該 issue，並在兩個 Dockerfile 註明；tts 因 f5-tts 是 git URL 依賴，技術上無法使用 hash 模式。

### 5. 一致性檢查做在兩層

- **產生與檢查分開**：`common/gpu_constraints.py` 產生 `common/gpu-constraints.txt`，是「哪些套件算 GPU 套件」的唯一定義；`--check` 模式確認它沒有過期。`tests/check_lock_consistency.py` 不需要知道這個定義，直接拿 constraints 檔當標準比對。
- **repo 層**：`tests/check_lock_consistency.py` 比對 asr、mt、tts 的 `requirements.txt` 與 `common/gpu-constraints.txt`。規則是「兩邊都有的套件版本必須相同」，服務沒列的不算問題（共用層還是會裝，服務只是沒有直接 import：mt 沒列 torchaudio/torchvision/torchcodec，tts 沒列 torchvision）。以 tox env `lockcheck` 執行，Travis 加一個 job。不需網路，0.1 秒完成。
- **build 層**：服務 Dockerfile 在 `pip install` 之前，用同一支腳本（隨 gpu stage 放在 `/opt/formosan-ai/`）比對 `/opt/formosan-ai/gpu-requirements.txt` 與掛載進來的 `requirements.txt`。本機 build 也會擋，不會出現 CI 沒跑到就 build 出兩份 torch 的情況。

「重跑 uv pip compile 再 git diff」能連忘了重編都抓到，但要網路且每個服務數十秒到數分鐘，列為後續可加項目。

### 6. apt 套件只保留 ffmpeg，以 ldd 與 strace 證明

移除：nodejs 與 nodesource 腳本、curl、cmake、rsync、git、git-lfs、libsm6、libxext6、libgl1-mesa-glx（bookworm 已無此套件）。python3.10 等由 base image 提供。

驗證方法（寫成可重複的指令，放在 `common/README` 或 tox env）：

```bash
# 函式庫層：所有 wheel 的 .so 實際連到哪些系統套件
find /usr/local/lib/python3.*/site-packages -name '*.so*' | xargs -n1 ldd 2>/dev/null \
  | grep -E '=> /(usr/)?lib' | awk '{print $3}' | sort -u | xargs dpkg -S 2>/dev/null | cut -d: -f1 | sort -u
# 動態層：smoke 期間 execve 與 dlopen 了什麼
strace -f -e trace=execve,openat -o /tmp/trace python app.py
```

smoke 必須涵蓋：上傳非 wav 音檔、麥克風錄音、一次辨識或合成、冷啟動模型下載。這四個動作分別走 pydub、gradio、whisperx 與 torchcodec、huggingface-hub。

### 6.1 tts 在 build 時需要 git（實際 build 後發現）

決策 6 的判斷只看了執行時：repo 程式碼沒有 shell out 到 git，模型也走 huggingface-hub 的 HTTP。但 tts 的 `f5-tts @ git+https://github.com/SWivid/F5-TTS.git@695c735...` 是 git URL 依賴，pip 安裝時要 git 才能 clone，第一次 build 就失敗：

```
ERROR: Cannot find command 'git' - do you have 'git' installed and in your PATH?
```

做法是只在 tts 的 Dockerfile，於同一個 `RUN` 內安裝 git、`pip install`、移除 git，git 不進任何 image 的最終內容，也不影響共用的 base 與 gpu 層。

**替代方案：**

- git 放進 base：最簡單，但四個 image 都多一份只有 tts build 時用得到的東西，而且改 base 會讓 gpu 那 3.7 GiB 重建。
- 改用 GitHub 壓縮檔 URL（`.../archive/<commit>.zip`），完全不需要 git：這個 commit 的 `build-system` 需要 `setuptools-scm`，而且有 `src/third_party/BigVGAN` submodule，壓縮檔沒有 git metadata 也不含 submodule，打包出來的內容可能不完整。
- 改用 PyPI 的 `f5-tts==1.1.4`：PyPI 的 1.1.4 發佈於 2025-05-04，這個 commit 是 2025-05-16，內容不同。等升 Python 3.12 時可一併評估換成 PyPI 的新版，就能拿掉這段。

### 7. Travis 用 buildx registry cache，保留兩個 build stage

```
cache ref：ithuan/formosan-ai:cache-<base|gpu|files|asr|asr-kaldi|tts|mt>
所有 build   --cache-from type=registry,ref=<cache ref>
main 分支    --cache-to   type=registry,ref=<cache ref>,mode=max
順序         common(base) → common(gpu) → common(files) → asr → asr-kaldi → tts → mt
```

建置寫在根目錄的 `docker-bake.hcl`，不是依序執行 `docker buildx build`（實作時查官方文件後的調整）：

- registry cache 需要 `docker-container` driver（[Registry cache](https://docs.docker.com/build/cache/backends/registry/)：預設 docker driver 只有在開啟 containerd image store 時才支援）。
- 但 `docker-container` driver 的 BuildKit 看不到本機 image store，`FROM formosan-ai-gpu` 會跑去 Docker Hub 拉而失敗（[docker/buildx#1453](https://github.com/docker/buildx/issues/1453)、[docker/build-push-action#1176](https://github.com/docker/build-push-action/issues/1176)）。原本 `COPY --from=formosan-ai-common` 也有同樣的問題。
- bake 的 `contexts = { formosan-ai-gpu = "target:gpu" }`（[Using a Bake target as build context](https://docs.docker.com/build/bake/contexts/)）讓同一次 build 內的 target 直接當別人的 base，順序由 bake 解。

`CACHE` 變數：空字串不用 cache（本機），`read` 只讀（PR，token 沒有 push 權限），`readwrite` 讀也寫（main）。

**第一次 Travis build 在 `--load` 時磁碟用盡**（2026-09-18）：`docker-container` driver 的 `--load` 會把每個 target 匯出成 tarball 再匯入 Docker，gpu、asr、mt、tts 四個 image 各自帶著同一個 4 GB 的 torch 層平行匯入，加上 builder 自己的 cache，出現 `no space left on device`。改為 PR 不輸出（只驗證能 build），main 用 `--push` 直接推到 Docker Hub：registry 端相同的層只存一份，也不經過 Travis 的 Docker。以 `bake --print --push asr asr-kaldi tts mt` 確認只有四個服務會推送，base、gpu、files 為 `cacheonly`。同一次 log 裡的 `cache-*: not found` 是預期的：main 還沒寫過 cache。一定要 `-f docker-bake.hcl`，否則 bake 會把 `docker-compose.yml` 一起讀進來合併。以 buildx v0.37.1 的 `bake --print` 確認三種模式都解析正確，但尚未實際 build。

requirements 未變時 gpu 與服務的 pip 層直接命中，buildx 不需下載已存在於 registry 的 blob。PR build 只有 pull token，只做 `--cache-from`。目前的兩個 build stage 維持不動，第二次 build 因全部命中而變得很便宜；合併 stage 的改動留到 cache 穩定後。

**替代方案：** `docker pull` 上一版 image 再 `--cache-from`（每個 job 先拉 6 GiB）；Travis `cache: directories` 存 `type=local` cache（多 GiB 上傳下載，更慢）；搬到 GitHub Actions 用 `type=gha`（10 GiB 上限，torch 層一個就佔滿）。

### 8. base image 以 digest pin，交給 Dependabot docker ecosystem

`python:3.10-slim@sha256:…` 可重現，Debian 或 Python patch 重建時由 Dependabot 提 PR，CI build 與 smoke 通過即可合併。OS 修補與 torch 升版從此是兩條獨立的線。

## Risks / Trade-offs

- [tts 的 numpy 停在 1.26.4，會被服務 layer 從 gpu layer 的 2.2.6 降版] → numpy 只有十幾 MB，且 numpy 2 ABI 向下相容；tts 目前就是 numpy 1.26.4。部署後以合成驗證。
- [tts 在 torch 2.8.0 上合成結果或效能改變] → 實作前先 spike：以 2.8.0 編出 lock 檔，本機實跑數個語別的合成並比對音檔長度與可聽性；有問題則 tts 暫留 2.7.0 並回到「兩個 gpu image」方案重新評估。
- [Travis 的 docker 沒有 buildx 或版本過舊] → spike 先在 Travis 跑 `docker buildx version`；沒有就在 `before_install` 安裝 buildx plugin binary。
- [移除 git、git-lfs 後某條路徑仍需 git] → strace smoke 的 execve 清單是驗收條件；漏掉會在容器啟動或第一次操作時以明確錯誤出現，回滾只是加回一個 apt 套件。
- [python:slim 是 Debian，非 Ubuntu 22.04] → manylinux wheel 不受影響。2026-09-18 的 Travis log 顯示 pin 的 digest 是 Debian 13 trixie，ffmpeg 從 Ubuntu 的 4.4 變成 trixie 的版本，torchcodec 0.7 支援 FFmpeg 4 到 7；測試機上線後檢查全部通過。
- [三個 image 共用一層意味 torch 升版必須三個服務一起] → 這正是設計目標；代價是升版由最嚴格的消費者決定，目前是 whisperx。
- [`--cache-to mode=max` 把中間層也推上 Docker Hub，占用倉庫空間] → cache tag 是獨立的 tag，可定期清理；`mode=min` 是退路。
- [服務 lock 檔與 common 一致，但 uv 與 pip 對同一個 lock 的安裝結果應相同] → lock 檔格式是 pip 的 requirements 格式，安裝端仍用 pip，不變。
- [uid 1000 的 nonroot 使用者要自己建] → 與 asr-kaldi 現況相同，`model_cache_*` 目錄 mode 1777，不受影響。

## Migration Plan

拆成四個可獨立上線的 PR，每個都能單獨 revert：

1. **換 base、清 apt**：三個 GPU 服務改 `python:3.10-slim`，拿掉 Node 與其他無用 apt 套件，`libgl1-mesa-glx` 一併處理。不改架構，只省空間。正式機 pull 後以 `torch.cuda.is_available()` 與一次實際操作驗證。
2. **共用 base 與 gpu stage**：`common/Dockerfile` 三 stage、`common/requirements.*`、torch 統一 2.8.0、三個服務 lock 重編、一致性檢查、README 與 openspec config。
3. **Travis cache**：buildx registry cache、build 順序。
4. **digest pin 與 Dependabot docker**。

回滾：每個 PR revert 即可；正式機保留上一版 image tag，`docker compose pull` 回舊 digest。

## Open Questions

- Travis jammy 的 docker 是否內建 buildx？（spike 1）
- tts 在 torch 2.8.0 與 torchcodec 0.7.x 的實跑結果？（spike 2）
- Dependabot 目前是否有設定檔？repo 沒有 `.github/dependabot.yml`，但 git log 有 pip group 的 bump commit，可能是 GitHub 預設安全更新；要加 docker ecosystem 需新增設定檔。
- `formosan-ai-base`、`formosan-ai-gpu` 是否也推到 Docker Hub？正式機不需要（服務 image 已含這些 layer），CI 只需 cache tag；先不推。
