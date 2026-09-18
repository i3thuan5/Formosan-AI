FROM formosan-ai-base

# 只裝 wheel，不在 build 時執行套件的 setup 腳本。
# antlr4-python3-runtime（omegaconf 的依賴）與 srt（vosk 的依賴）在 PyPI 上只有原始碼包，列為例外。
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    pip install \
    --only-binary=:all: \
    --no-binary=antlr4-python3-runtime,srt \
    --no-cache-dir \
    -r /tmp/requirements.txt

COPY --from=formosan-ai-common /app/utils.py /app/colors.py /app/
COPY --from=formosan-ai-common /app/common_static/ /app/common_static/
COPY --link app.py DEMO.md /app/
COPY --link configs/ /app/configs/
COPY --link examples/ /app/examples/

USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
