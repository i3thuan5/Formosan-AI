FROM python:3.10

WORKDIR /home/user/app
RUN apt-get update && \
	apt-get install -y 	git 	git-lfs 	ffmpeg 	libsm6 	libxext6 	cmake 	rsync 	libgl1-mesa-glx 	&& \
	rm -rf /var/lib/apt/lists/* 	&& \
	git lfs install

RUN pip install --no-cache-dir pip -U

RUN 	pip install --no-cache-dir 	datasets 	"huggingface-hub>=0.19" "hf_xet>=1.0.0,<2.0.0" "hf-transfer>=0.1.4" "protobuf<4" "click<8.1" "pydantic~=1.0" torch==2.5.1

RUN 	apt-get update && 	apt-get install -y curl && 	curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && 	apt-get install -y nodejs && 	rm -rf /var/lib/apt/lists/* && apt-get clean
RUN --mount=target=/tmp/requirements.txt,source=requirements.txt     pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir 	gradio[oauth,mcp]==5.29.0 	"uvicorn>=0.14.0" 	spaces==0.35.0


RUN addgroup --gid 1000 nonroot && \
  adduser --uid 1000 --disabled-password --ingroup nonroot --quiet nonroot
COPY --link --chown=1000 ./ /app
USER nonroot
WORKDIR /app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
