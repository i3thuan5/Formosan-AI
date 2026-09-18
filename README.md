# formosan-asr-cd

原語會「族語AI翻譯計畫」語音系統。

- 族語AI成果網站網址：https://ai-labs.ilrdf.org.tw/
- 語音辨識系統網址：https://sapolita-kaldi.ithuan.tw/
- 合成系統網址：https://hnang-kari-ai-asi-sluhay.ithuan.tw/
- 基礎翻譯系統網址：https://ithuan-formosan-translation.hf.space/

## local開發

### 建立 Python virtual environment

```bash
python -m venv venv
```

### 載入 Python virtual environment

Ta̍k-kái攏ài開，才來開發。

```bash
source venv/bin/activate
```

### 安裝tox

tox是tī本機走test用--ê。

```bash
pip install tox
```

### 執行Linter排版器

依需求執行，如：

```bash
tox -e yamllint
tox -e flake8
```

詳細需求可參考`tox.ini`。

## 更新套件版本

`requirements.in`是記專案有直接用ê第三方套件。`requirements.txt`是管kui專案全部第三方套件koh對應版本，保證開發、CI試驗、上線版本一致。

用 [uv](https://github.com/astral-sh/uv) 編譯：`pip install uv`。

### torch 系列的版本只有一個來源

`common/requirements.in` 管 torch、torchaudio、torchvision、torchcodec。
它編出來的 `common/requirements.txt` 由 `formosan-ai-gpu` image 安裝一次，
asr、mt、tts 三個 image 共用同一層，正式機也只 pull 一次。

服務編譯時加 `-c common/gpu-constraints.txt`，torch 與 `nvidia-*` 就一定和共用層一樣。
上游若硬 pin 不同版本（例如 whisperx 的 `torch~=2.8.0`），編譯會直接失敗，
而不是悄悄在服務 layer 再裝一份 torch。

### 改某個服務的套件

1. 手動更新該服務的 `requirements.in`。
2. 重新編譯：

   ```bash
   # mt、tts
   uv pip compile mt/requirements.in \
       --python-version 3.10 --python-platform linux --no-strip-extras \
       -c common/gpu-constraints.txt -o mt/requirements.txt

   # asr（有 hash）
   uv pip compile asr/requirements.in \
       --python-version 3.10 --python-platform linux \
       --generate-hashes --no-strip-extras \
       -c common/gpu-constraints.txt -o asr/requirements.txt

   # asr-kaldi（沒有 GPU 套件，不用 -c）
   uv pip compile asr-kaldi/requirements.in \
       --python-version 3.10 --python-platform linux --no-strip-extras \
       -o asr-kaldi/requirements.txt
   ```

3. `tox -e lockcheck` 確認和共用層一致。
4. 檢查 `requirements.txt` 的 diff。

### 升 torch

1. 改 `common/requirements.in`。**版本由最嚴格的模型決定**，目前是 asr 的
   whisperx（`torch~=2.8.0`）。
2. 重編共用 lock 檔與 constraints：

   ```bash
   uv pip compile common/requirements.in \
       --python-version 3.10 --python-platform linux --generate-hashes \
       -o common/requirements.txt
   python common/gpu_constraints.py
   ```

3. 三個 GPU 服務全部重編（指令同上）。
4. `tox -e lockcheck`。
5. 部署前後跑 [tests/production_tests/](tests/production_tests/) 的檢查比對。

`cu12` 的 wheel 需要主機 NVIDIA 驅動 525.60 以上，升到 `cu13` 前先在正式機確認 `nvidia-smi`。

## local測試

1. 設定reverse proxy server：

```bash
git clone --depth 1 https://github.com/i3thuan5/ZuGi.git
docker compose -f ZuGi/docker-compose.yml up -d --build nginx-proxy
```

2. 準備環境變數檔：`cp deploy/.env.template .env`，並把要測試的服務HOST改成`localhost`。

3. 手動編三個共用image：

   ```bash
   docker buildx bake -f docker-bake.hcl --load base gpu files
   ```

4. 編image後啟動：`docker compose up -d --build`。

## Docker image 架構

```text
python:3.10-slim（digest pin，Dependabot 顧）
└── formosan-ai-base    apt ffmpeg、nonroot 使用者          ← asr-kaldi
    └── formosan-ai-gpu torch + nvidia-*（約 3.7 GiB）      ← asr、mt、tts
formosan-ai-common      FROM scratch，共用檔案             ← 四個服務 COPY --from
```

## 上線後檢查

部署到測試機或正式機之後，跑 [tests/production_tests/](tests/production_tests/) 確認四個服務都正常：

```bash
BASE_URL=https://<受測主機> tox -e production_tests
```

## 發佈

目前未設置自動發佈（CD），請人工發佈。
