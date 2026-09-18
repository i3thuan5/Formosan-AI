FROM formosan-ai-gpu

# 先確認這份 lock 檔的 torch 與 nvidia-* 和共用層一致，不一致就不要 build 出
# 多帶一份 torch 的 image。CI 的 tox -e lockcheck 做同樣的事（tests/check_lock_consistency.py）。
# f5-tts 是從 git URL 安裝的，pip 要 git 才能 clone（還會一併 clone BigVGAN submodule）。
# 不能改用 GitHub 的壓縮檔：這個 commit 的 build 需要 setuptools-scm 與 submodule，壓縮檔都沒有。
# git 只在 build 時需要，同一個 RUN 裝完就移除，不留在 image 裡。
#
# 這裡刻意不加 --only-binary=:all:（SonarQube docker:S8541 已審查、可接受）：
# f5-tts 從 git 安裝一定要 build，另外 antlr4-python3-runtime、encodec、jieba、
# transformers-stream-generator 在 PyPI 上只有原始碼包。版本都鎖在 requirements.txt。
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    python /opt/formosan-ai/check_lock_consistency.py \
    --constraints /opt/formosan-ai/gpu-constraints.txt \
    /tmp/requirements.txt && \
    apt-get update && \
    apt-get install --no-install-recommends -y git && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

COPY --from=formosan-ai-common /app/utils.py /app/colors.py /app/
COPY --from=formosan-ai-common /app/common_static/ /app/common_static/
COPY --link app.py DEMO.md /app/
COPY --link configs/ /app/configs/
COPY --link ipa/ /app/ipa/
COPY --link assets/ /app/assets/
COPY --link static/ /app/static/

USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
