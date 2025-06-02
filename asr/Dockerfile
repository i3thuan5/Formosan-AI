FROM registry.hf.space/ithuan-formosan-asr:latest

USER root
RUN addgroup --gid 1001 nonroot && \
  adduser --uid 1001 --disabled-password --ingroup nonroot --quiet nonroot

WORKDIR /opt
COPY requirements.txt ./
RUN pip install -r requirements.txt

USER nonroot
WORKDIR /home/nonroot

COPY app.py DEMO.md ./
COPY models.yaml configs/
EXPOSE 7860
CMD ["python", "app.py"]
