FROM registry.hf.space/ithuan-formosan-f5-tts:latest

COPY requirements.txt ./
# RUN pip install -r requirements.txt

COPY models.yaml configs/models.yaml

EXPOSE 7860
CMD ["python", "app.py"]
