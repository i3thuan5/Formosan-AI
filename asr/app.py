import os
import tempfile
from pathlib import Path

import torch
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from whisper import load_audio, load_model

from utils import render_demo

SAMPLING_RATE = 16000
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

model = load_model(
    "formospeech/whisper-large-v2-formosan-all-ct2",
    device=device,
    compute_type=compute_type,
    asr_options={"word_timestamps": True},
)


def get_title():
    with open("DEMO.md") as tong:
        return tong.readline().strip("# ")


with render_demo(
    title=get_title(),
    js='''
        function remove_gradio5_iframe_issue61() {
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                const parent = iframe.parentNode;
                if (parent) {
                  parent.removeChild(iframe);
                }
            });
        }
    '''
) as demo:

    with open("DEMO.md") as tong:
        gr.Markdown(tong.readline(), elem_id="main")
        gr.Markdown(tong.read())

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="族語影片", sources="upload")
            transcribe_button_video = gr.Button("開始辨識", variant="primary")
        with gr.Column():
            srt_output = gr.Textbox(label="辨識結果", lines=12)
            download_srt_button = gr.Button("下載 SRT 字幕檔", variant="primary")
            download_srt_button_hidden = gr.DownloadButton(
                visible=False, elem_id="download_srt_button_hidden"
            )

    def generate_srt(audio):
        audio = load_audio(audio, sr=SAMPLING_RATE)

        output = model.transcribe(
            audio,
            language="id",
            batch_size=BATCH_SIZE,
        )

        segments = output["segments"]
        print(segments)

        srt_content = ""

        for i, segment in enumerate(segments):
            start_seconds = segment["start"]
            end_seconds = segment["end"]

            srt_content += f"{i + 1}\n"

            start_time_srt = f"{int(start_seconds // 3600):02}:{int((start_seconds % 3600) // 60):02}:{int(start_seconds % 60):02},{int((start_seconds % 1) * 1000):03}"
            end_time_srt = f"{int(end_seconds // 3600):02}:{int((end_seconds % 3600) // 60):02}:{int(end_seconds % 60):02},{int((end_seconds % 1) * 1000):03}"
            srt_content += f"{start_time_srt} --> {end_time_srt}\n"

            srt_content += f"族語：{segment['text'].strip()}\n"
            srt_content += "華語：\n\n"

        return srt_content.strip()

    transcribe_button_video.click(
        fn=generate_srt,
        inputs=[
            video_input,
        ],
        outputs=srt_output,
    )

    def export_srt(srt_content):
        with tempfile.NamedTemporaryFile(
            prefix="族語影片字幕-", suffix=".srt",
            delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(srt_content)
            return f.name

    download_srt_button.click(
        fn=export_srt,
        inputs=srt_output,
        outputs=download_srt_button_hidden,
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => document.querySelector('#download_srt_button_hidden').click()",
    )

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, demo, path="")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
