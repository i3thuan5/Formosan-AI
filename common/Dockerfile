# syntax=docker/dockerfile:1

# 三個共用 image，由 repo 根目錄的 docker-bake.hcl 依序建：
#
#   docker buildx bake -f docker-bake.hcl --load base gpu files
#
# base  → asr-kaldi 的 FROM
# gpu   → asr、mt、tts 的 FROM，torch 與 nvidia-* 只裝這一次，三個 image 共用同一層
# files → 共用的 utils.py、colors.py 與靜態檔，四個服務都用 COPY --from 取用
#
# gpu 需要具名 build context tests=./tests（bake 會帶），一致性檢查腳本在 tests/。


# ---- base：Python 與系統套件 ---------------------------------------------
# 不用 nvidia/cuda 當 base：PyPI 的 torch wheel 自帶 nvidia-*-cu12，執行時只會載入
# 那一份，base image 裡的 CUDA 從來沒被用到（還比 pip 那份舊，cuDNN 8 vs 9）。
# GPU 存取靠 nvidia-container-toolkit 在 docker run 時注入驅動，和 image 無關。
FROM python:3.10-slim@sha256:8c97ebedc32fd60935cdf5992e935753e2a0f98231830028050e1e04bd3c13c2 AS base

# ffmpeg 有三個使用者：whisperx 的 load_audio 會 shell out、gradio 收到非 wav 時
# 透過 pydub 轉檔、torchcodec 匯入時 dlopen libav*。其餘 apt 套件都沒有使用者。
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN addgroup --gid 1000 nonroot && \
    adduser --uid 1000 --disabled-password --ingroup nonroot --quiet nonroot

# nvidia-container-toolkit 用這兩個變數決定要注入哪些驅動函式庫，
# 原本是 nvidia/cuda image 提供的，改用 python:slim 之後要自己設。
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


# ---- gpu：torch 與 nvidia 套件（約 3.7 GiB，三個服務共用）-----------------
FROM base AS gpu

RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    pip install \
    --require-hashes \
    --only-binary=:all: \
    --no-cache-dir \
    -r /tmp/requirements.txt

# 服務 image build 時用這兩個檔案確認自己的 lock 檔沒有和共用層打架
COPY --from=tests check_lock_consistency.py /opt/formosan-ai/
COPY gpu-constraints.txt /opt/formosan-ai/


# ---- files：共用檔案 ------------------------------------------------------
# 必須是獨立的 FROM scratch，不能疊在 base 上：gpu 疊在 base 上，
# utils.py 改一個字就會讓 gpu 那 3.7 GiB 的 layer 重建重推。
FROM scratch AS files

WORKDIR /app

USER nonroot

COPY colors.py utils.py /app/
COPY static/ /app/common_static/
