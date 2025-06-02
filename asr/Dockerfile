FROM registry.hf.space/ithuan-formosan-asr:latest

WORKDIR /opt
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py DEMO.md ./
COPY models.yaml configs/
EXPOSE 7860
CMD ["python", "app.py"]
