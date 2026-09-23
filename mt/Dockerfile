FROM formosan-ai-gpu

# 先確認這份 lock 檔的 torch 與 nvidia-* 和共用層一致，不一致就不要 build 出
# 多帶一份 torch 的 image。CI 的 tox -e lockcheck 做同樣的事（tests/check_lock_consistency.py）。
# 只裝 wheel，不在 build 時執行套件的 setup 腳本。mt 的套件全部都有 wheel，不需要例外。
# 刻意不加 --require-hashes（SonarQube docker:S8544 已審查、可接受）：hash 先在 asr 與
# 共用 gpu 層試用，團隊之後再決定要不要全面導入。
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    python /opt/formosan-ai/check_lock_consistency.py \
    --constraints /opt/formosan-ai/gpu-constraints.txt \
    /tmp/requirements.txt && \
    pip install \
    --only-binary=:all: \
    --no-cache-dir \
    -r /tmp/requirements.txt

COPY --from=formosan-ai-common /app/utils.py /app/colors.py /app/
COPY --from=formosan-ai-common /app/common_static/ /app/common_static/
COPY --link app.py formosan_languages.py tts_client.py DEMO.md /app/

USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
