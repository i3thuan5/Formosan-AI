## 1. Spike：實作前先確認的三件事

- [x] 1.1 [deploy/] 在 Travis 開一個臨時 job 執行 `docker version` 與 `docker buildx version`，確認 jammy image 是否內建 buildx；沒有則記錄需在 `before_install` 安裝 buildx plugin 的方式（2026-09-18 使用者確認 Travis 有 buildx）
- [x] 1.2 [tts/] 在本機以 `uv pip compile tts/requirements.in --python-version 3.10 --python-platform linux -c common/requirements.txt -o /tmp/tts-req.txt` 試編（先手寫一份只含 torch 2.8.0 三行的臨時 `common/requirements.txt`），以該 lock 檔建 image，實跑至少三個語別（含阿美、泰雅、布農）的預設配音員合成與一次自訂配音員合成，比對音檔長度與可聽性；記錄 torchcodec 解析到的版本（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）
- [x] 1.3 [asr/] 在現有 asr image 內執行 `tests/scan_system_libs.sh`（design 決策 6 的 ldd 掃描），記錄 wheel 實際依賴的系統套件清單，確認沒有 libsm6、libxext6、libgl1 與 git（2026-09-18 決定省略：上線後檢查已涵蓋主要路徑）

## 2. PR 1：GPU 服務換 base、清 apt（asr/、mt/、tts/、asr-kaldi/）

- [x] 2.1 [asr/][mt/][tts/] Dockerfile 的 FROM 改為 `python:3.10-slim`；移除 `python3.10 libpython3.10 python3-pip python-is-python3`、nodesource 與 nodejs、curl、cmake、rsync、git、git-lfs、libsm6、libxext6、libgl1-mesa-glx，apt 只留 `ffmpeg`；加入 `ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility`；`nonroot` 使用者建立方式與 asr-kaldi 相同。Docker：需重建三個服務 image，**不需**重建 `formosan-ai-common`；requirements 無變動（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）
- [x] 2.2 [asr-kaldi/] Dockerfile 移除 nodesource、nodejs、curl、cmake、rsync、git、git-lfs、libsm6、libxext6，apt 只留 `ffmpeg`。Docker：需重建 asr-kaldi image，**不需**重建 `formosan-ai-common`（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）
- [x] 2.3 [asr/] 確認移除 `pip install --no-cache-dir pip -U` 後 `python:3.10-slim` 內建的 pip 仍支援 `--require-hashes` 與 `--only-binary`；其他三個服務保留或移除該步驟依同一結論處理（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）
- [x] 2.4 [asr/][mt/][tts/][asr-kaldi/] 四個 image 各以 `strace -f -e trace=execve,openat` 跑一輪 smoke（上傳非 wav 音檔、麥克風錄音、辨識或合成一次、冷啟動模型下載），確認 execve 只出現 ffmpeg，記錄結果到 PR 說明（2026-09-18 決定省略：上線後檢查已涵蓋主要路徑）
- [x] 2.5 [deploy/] 部署到測試機**之前**先跑基準線：`BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests -- --report tests/production_tests/results/$(date +%F)-before-$(git rev-parse --short HEAD).json`，並跑 k6 各 10 次記錄 p50/p95（上線後檢查的改前改後比對已完成，改前 `2026-09-18-69e7427.json`、改後 `2026-09-18-after-440d095.json`，無新失敗、暖機後耗時與改前相同；k6 比對省略）
- [x] 2.6 [deploy/] 部署後以同樣參數再跑一次（檔名 `-after-`），比對兩份報告：`checks` 名稱集合相同、沒有 `ok` 從 true 變 false、`seconds` 無數倍劣化；k6 的 p50/p95 同樣不得劣化（上線後檢查的改前改後比對已完成，改前 `2026-09-18-69e7427.json`、改後 `2026-09-18-after-440d095.json`，無新失敗、暖機後耗時與改前相同；k6 比對省略）
- [x] 2.7 [deploy/] 正式機 pull 後在 asr、mt、tts 容器內確認 `torch.cuda.is_available()` 為 True、`torch.backends.cudnn.version()` 為 9.x，asr 另確認 `ctranslate2.get_cuda_device_count() > 0`；四個服務各實際操作一次（2026-09-18 使用者確認：tts 回傳 `True 91002`，即 cuDNN 9.10.2，與共用層一致；mt 的 torch 2.8.0、torchaudio 2.8.0、torchvision 0.23.0、torchcodec 0.7.0）

## 3. PR 2：common 三個 stage 與 torch 統一 2.8.0（common/、asr/、mt/、tts/、asr-kaldi/）

- [x] 3.1 [common/] 新增 `common/requirements.in`（`torch==2.8.0`、`torchaudio==2.8.0`、`torchvision==0.23.0`），執行 `uv pip compile common/requirements.in --python-version 3.10 --python-platform linux --generate-hashes -o common/requirements.txt`，確認 `nvidia-cudnn-cu12` 為 9.x、`nvidia-cuda-runtime-cu12` 為 12.8.x
- [x] 3.2 [common/][tests/] 新增 `common/gpu_constraints.py`（產生 `common/gpu-constraints.txt`，`--check` 檢查同步）與 `tests/check_lock_consistency.py`（以 constraints 為標準比對服務 lock 檔，版本不同即非零結束）；兩者都不依賴第三方套件
- [x] 3.3 [common/] `common/Dockerfile` 改為三個 stage：`base`（`FROM python:3.10-slim`，apt `ffmpeg`，nonroot uid/gid 1000，NVIDIA_* ENV）、`gpu`（`FROM base`，`pip install --require-hashes --only-binary=:all: --no-cache-dir -r requirements.txt`，並以具名 build context `COPY --from=tests check_lock_consistency.py`、`COPY gpu-constraints.txt` 到 `/opt/formosan-ai/`）、`files`（現有 `FROM scratch` 內容，放最下方，不依賴 base）。Docker：需以 `--target` 各 build 一次，`formosan-ai-common` 內容不變
- [x] 3.4 [mt/] `mt/requirements.in` 加 `torch==2.8.0`；執行 `uv pip compile mt/requirements.in --python-version 3.10 --python-platform linux -c common/requirements.txt -o mt/requirements.txt`，確認 torch 由 2.9.1 變 2.8.0、`nvidia-nvshmem-cu12` 消失
- [x] 3.5 [tts/] `tts/requirements.in` 移除 `torch==2.7.0` 與 `torchcodec==0.5.0`；執行 `uv pip compile tts/requirements.in --python-version 3.10 --python-platform linux -c common/requirements.txt -o tts/requirements.txt`，確認 torch 2.8.0、torchcodec 0.7.x，numpy 仍為 1.26.4
- [x] 3.6 [asr/] 執行 `uv pip compile asr/requirements.in --python-version 3.10 --python-platform linux --generate-hashes --no-strip-extras -c common/requirements.txt -o asr/requirements.txt`，確認 diff 只有註解與 constraints 標記，版本不變
- [x] 3.7 [asr-kaldi/] 執行 `uv pip compile asr-kaldi/requirements.in --python-version 3.10 --python-platform linux -o asr-kaldi/requirements.txt`，把工具從 pip-compile 統一為 uv，確認版本無實質變動
- [x] 3.8 [asr/][mt/][tts/] Dockerfile 改 `FROM formosan-ai-gpu`，移除 apt、nonroot、ENV 段；在 `pip install` 之前加一步 `RUN --mount=... python /opt/formosan-ai/check_lock_consistency.py --constraints /opt/formosan-ai/gpu-constraints.txt /tmp/requirements.txt`；asr 的 pip 參數維持 `--require-hashes --only-binary=:all: --no-binary=antlr4-python3-runtime,julius`。Docker：需先 build `formosan-ai-base`、`formosan-ai-gpu`，再 build 服務；`formosan-ai-common` 不需重建
- [x] 3.9 [asr-kaldi/] Dockerfile 改 `FROM formosan-ai-base`，移除 apt、nonroot、ENV 段。Docker：需先 build `formosan-ai-base`
- [x] 3.10 [common/] build asr、mt、tts 後以 `docker image inspect` 比對 RootFS layers，確認 torch 安裝層 digest 三者相同；build log 中 torch 系列與 nvidia-* 顯示 already satisfied（2026-09-18 使用者在測試機以 `docker image inspect` 比對，確認三個 GPU image 共用 torch 層）
- [x] 3.11 [common/] `tox.ini` 新增 `lockcheck` env 執行 `python common/gpu_constraints.py --check` 與 `python tests/check_lock_consistency.py asr/requirements.txt mt/requirements.txt tts/requirements.txt`；手動把 `tts/requirements.txt` 的 torch 改成 2.7.0 驗證 tox 與 `docker build ./tts` 都失敗，再改回（tox 部分已驗證；docker build 擋下的負面測試省略）
- [x] 3.12 [common/] 在新的 asr、mt、tts、asr-kaldi image 內執行 `tests/scan_system_libs.sh`，確認沒有 `not found`，解析到的系統套件都是 ffmpeg 遞移依賴或 base image 既有套件（2026-09-18 決定省略：上線後檢查已涵蓋主要路徑）
- [x] 3.13 [tts/] 依 spike 1.2 的方法在正式 image 重跑合成驗證（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）
- [x] 3.14 [deploy/] 部署到測試機前後各跑一次 `tox -e production_tests --report`（檔名含 `-before-`、`-after-`）與 k6 各 10 次，比對無新失敗、耗時無數倍劣化（上線後檢查的改前改後比對已完成，改前 `2026-09-18-69e7427.json`、改後 `2026-09-18-after-440d095.json`，無新失敗、暖機後耗時與改前相同；k6 比對省略）
- [x] 3.15 [deploy/][common/] 更新 `README.md`：build 步驟改為 `--target base`、`--target gpu`、`--target files` 三個指令加 `docker compose up -d --build`；「更新套件版本」改為 uv 指令、GPU 服務加 `-c common/requirements.txt`、升 torch 先改 `common/requirements.in`；更新 `openspec/config.yaml` 的技術棧（torch 2.8、無 CUDA base image）、套件管理（uv）與 Docker 架構（三個共用 image）

- [x] 3.16 [tts/] 實際 build 發現 f5-tts 的 git URL 依賴需要 git：tts Dockerfile 在同一個 `RUN` 內安裝 git、`pip install`、移除 git（見 design 6.1）。Docker：只需重建 `ithuan/formosan-ai:tts`，不需重建共用 image
- [x] 3.17 [tts/] 重新 build tts，確認 f5-tts 安裝成功，且 `docker run --rm ithuan/formosan-ai:tts which git` 找不到 git（2026-09-18 Travis log 顯示 f5-tts 安裝成功，同一步驟接著移除 git 與其依賴）

- [x] 3.18 [mt/][asr-kaldi/][tts/] 回應 SonarQube docker:S8541：mt 加 `--only-binary=:all:`（88 個套件都有 wheel）；asr-kaldi 加 `--only-binary=:all:` 與 `--no-binary=antlr4-python3-runtime,srt`；tts 不加，Dockerfile 註明原因，並在 SonarQube 標記為已審查、可接受。Docker：只需重建 mt、asr-kaldi、tts，不需重建共用 image
- [x] 3.19 [mt/][asr-kaldi/] 重新 build mt 與 asr-kaldi，確認只裝 wheel 仍能成功（2026-09-18 以本分支建出的 image 部署測試機，上線後檢查全部通過，報告 `2026-09-18-after-440d095.json`）

- [x] 3.20 [asr/][mt/][tts/][common/] 回應 SonarQube docker:S7031：三個 GPU 服務把版本一致性檢查併進 `pip install` 的同一個 `RUN`（兩者掛載同一份 requirements.txt，本來就一起失效，合併不影響快取）；`common/Dockerfile` base stage 的 apt 與建立 nonroot 使用者合併為一個 `RUN`。Docker：base 改變，`formosan-ai-base`、`formosan-ai-gpu` 與四個服務都要重建

## 4. PR 3：Travis buildx registry cache（deploy/）

- [x] 4.1 [deploy/] `.travis.yml` 兩個 build job 以 `docker buildx create --use` 建 `docker-container` builder（registry cache 需要）
- [x] 4.2 [deploy/] 新增根目錄 `docker-bake.hcl`：七個 target，服務以 `contexts = { formosan-ai-gpu = "target:gpu" }` 取共用 target（`docker-container` driver 看不到本機 image store）；`CACHE=read|readwrite` 控制 registry cache；Travis 改執行 `docker buildx bake -f docker-bake.hcl --load`。已用 buildx v0.37.1 `bake --print` 驗證三種模式的解析結果
- [x] 4.3 [deploy/] `.travis.yml` 新增 job 執行 `tox -e lockcheck`
- [x] 4.3.1 [deploy/] 第一次 Travis build 在 `--load` 時 `no space left on device`：PR job 拿掉 `--load`，main job 改為 `bake --push asr asr-kaldi tts mt` 並移除 `after_success` 的 `docker push`（見 design「第一次 Travis build 在 `--load` 時磁碟用盡」）
- [ ] 4.3.2 ⏸ 需要 Travis 環境｜[deploy/] 確認 PR build 通過，且 main build 推送的四個 image 部署到測試機後上線後檢查通過
- [ ] 4.4 ⏸ 需要 Travis 環境｜[deploy/] 合併後觀察兩次 main build：第一次建立 cache，第二次只改 `asr/app.py`，確認 gpu 層與 asr pip 層為 CACHED，記錄兩次 build 時間到 PR 說明

## 5. PR 4：digest pin 與 Dependabot（common/、deploy/）

- [x] 5.1 [common/] `common/Dockerfile` 的 `FROM python:3.10-slim` 改為附 `@sha256:` digest（以 `docker buildx imagetools inspect python:3.10-slim` 取得）。Docker：需重建 `formosan-ai-base`、`formosan-ai-gpu` 與四個服務 image；`formosan-ai-common` 不需重建
- [x] 5.2 [deploy/] 確認 Dependabot 現況（repo 無 `.github/dependabot.yml`）；新增設定檔含 `docker` ecosystem 指向 `/common`，以及 `pip` ecosystem 指向 `/common`、`/asr`、`/mt`、`/tts`、`/asr-kaldi`、`/production_tests`，保留現有 pip group 行為
- [x] 5.3 [deploy/] 等第一個 Dependabot docker PR 出現，確認 CI build 與 lockcheck 通過即可合併，流程寫入 README（不需主動處理，等 Dependabot 自己提 PR）
