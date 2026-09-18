FROM formosan-ai-gpu

# 先確認這份 lock 檔的 torch 與 nvidia-* 和共用層一致，不一致就不要 build 出
# 多帶一份 torch 的 image。CI 的 tox -e lockcheck 做同樣的事（tests/check_lock_consistency.py）。
# requirements.txt 由 uv 完整鎖版並附 hash。WhisperX 會遞移依賴只有 sdist 的
# antlr4-python3-runtime 與 julius，所以 --only-binary=:all: 要開這兩個例外。
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    python /opt/formosan-ai/check_lock_consistency.py \
    --constraints /opt/formosan-ai/gpu-constraints.txt \
    /tmp/requirements.txt && \
    pip install \
    --require-hashes \
    --only-binary=:all: \
    --no-binary=antlr4-python3-runtime,julius \
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
