import os
import tempfile
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from languages import DEFAULT_GROUP, LANGUAGE_GROUPS
from subtitles import render_srt
from translation import Translator
from utils import render_demo
from whisper import load_audio, load_model

SAMPLING_RATE = 16000
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))


device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
model = load_model(
    "ILRDF/whisper-large-v2-formosan-lang-tokens-ct2",
    device=device,
    compute_type=compute_type,
    asr_options={"word_timestamps": True},
)
translator = Translator(device, batch_size=int(os.getenv("TRANSLATION_BATCH_SIZE", 8)))


def export_srt(srt_content):
    """Create a downloadable SRT file whenever the transcription changes."""
    if not srt_content:
        return None

    with tempfile.NamedTemporaryFile(
        prefix="族語影片字幕-",
        suffix=".srt",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as f:
        f.write(srt_content)
        return f.name


with render_demo(
    demo_md_filename="DEMO.md",
    js="""
        function remove_gradio5_iframe_issue61() {
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                const parent = iframe.parentNode;
                if (parent) {
                  parent.removeChild(iframe);
                }
            });
        }
    """,
) as demo:
    with gr.Row():
        with gr.Column():
            group_input = gr.Radio(
                choices=list(LANGUAGE_GROUPS),
                value=DEFAULT_GROUP,
                label="族別",
            )
            language_input = gr.Radio(
                choices=LANGUAGE_GROUPS[DEFAULT_GROUP],
                value=LANGUAGE_GROUPS[DEFAULT_GROUP][0][1],
                type="value",
                label="語別",
            )
            video_input = gr.Video(label="族語影片", sources="upload")
            transcribe_button_video = gr.Button("開始辨識", variant="primary")
        with gr.Column():
            srt_output = gr.Textbox(label="辨識結果", lines=12)
            download_srt_button = gr.DownloadButton(
                label="下載 SRT 字幕檔",
                value=export_srt,
                inputs=srt_output,
                variant="primary",
            )

    def update_languages(group):
        choices = LANGUAGE_GROUPS[group]
        return gr.Radio(
            choices=choices,
            value=choices[0][1],
            type="value",
            label="語別",
        )

    group_input.change(
        fn=update_languages,
        inputs=group_input,
        outputs=language_input,
    )

    def generate_srt(audio, language):
        if not audio:
            raise gr.Error("請先上傳影片。")
        audio = load_audio(audio, sr=SAMPLING_RATE)

        output = model.transcribe(
            audio,
            language=language,
            batch_size=BATCH_SIZE,
        )

        segments = [
            segment
            for segment in output["segments"]
            if segment["text"].strip() and segment["end"] > segment["start"]
        ]
        translations = translator.translate_segments(segments, language)
        return render_srt(segments, translations)

    transcribe_button_video.click(
        fn=generate_srt,
        inputs=[
            video_input,
            language_input,
        ],
        outputs=srt_output,
    )


# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path("./static")

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, demo, path="")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
