FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.10 python3-pip python-is-python3 \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1-mesa-glx && \
    git lfs install && \
    apt-get install --no-install-recommends -y curl && \
    curl --proto "=https" --tlsv1.2 -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install --no-install-recommends -y nodejs && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN pip install --no-cache-dir pip -U

RUN --mount=target=/tmp/requirements.txt,source=requirements.txt pip install --no-cache-dir -r /tmp/requirements.txt

COPY --link app.py DEMO.md /app/
COPY --link configs/ /app/configs/
COPY --link ipa/ /app/ipa/
COPY --link ref_wav/ /app/ref_wav/
RUN addgroup --gid 1000 nonroot && \
    adduser --uid 1000 --disabled-password --ingroup nonroot --quiet nonroot
USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
