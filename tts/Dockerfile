FROM formosan-ai-gpu

# 先確認這份 lock 檔的 torch 與 nvidia-* 和共用層一致，不一致就不要 build 出
# 多帶一份 torch 的 image。CI 的 tox -e lockcheck 做同樣的事（tests/check_lock_consistency.py）。
# chatterbox 的模型程式碼已複製在 tts/chatterbox/，不用從 git 安裝，build 時不需要 git。
#
# 這裡刻意不加 --only-binary=:all:（SonarQube docker:S8541 已審查、可接受）：
# omegaconf 依賴的 antlr4-python3-runtime 在 PyPI 上只有原始碼包。版本都鎖在 requirements.txt。
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    python /opt/formosan-ai/check_lock_consistency.py \
    --constraints /opt/formosan-ai/gpu-constraints.txt \
    /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=formosan-ai-common /app/utils.py /app/colors.py /app/languages.py /app/
COPY --from=formosan-ai-common /app/common_static/ /app/common_static/
COPY --link app.py formosan_tts.py DEMO.md /app/
COPY --link configs/models.yaml configs/refs.yaml /app/configs/
COPY --link chatterbox/ /app/chatterbox/
COPY --link assets/ /app/assets/
COPY --link static/ /app/static/

USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
