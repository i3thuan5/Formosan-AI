FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.12 libpython3.12 python3-pip python-is-python3 \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 && \
    git lfs install && \
    apt-get install --no-install-recommends -y curl && \
    curl --proto "=https" --tlsv1.2 -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install --no-install-recommends -y nodejs && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN groupmod --new-name nonroot ubuntu && \
    usermod --login nonroot --home /home/nonroot --move-home ubuntu

# requirements.txt is fully pinned by uv. WhisperX transitively requires the
# source-only antlr4-python3-runtime and julius packages, so this cannot use
# --only-binary=:all: without breaking the image build.
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    pip install \
    --break-system-packages \
    --no-cache-dir \
    -r /tmp/requirements.txt

COPY --from=formosan-ai-common /app/utils.py /app/colors.py /app/
COPY --from=formosan-ai-common /app/common_static/ /app/common_static/
COPY --link app.py whisper.py subtitles.py languages.py translation.py DEMO.md /app/

USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
