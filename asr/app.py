import os
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from whisper import load_audio, load_model

SAMPLING_RATE = 16000
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

model = load_model(
    "formospeech/whisper-large-v2-formosan-all-ct2",
    device="cuda",
    asr_options={"word_timestamps": True},
)


gr.set_static_paths(paths=[Path.cwd().absolute() / "static" / "image"])

with gr.Blocks(
    title="族語語音辨識系統 - 原住民族語言研究發展基金會",
    css_paths=[Path(__file__).parent / 'static' / 'css' / 'app.css', ],
    theme=gr.themes.Default(
        font=(
            "tauhu-oo",
            gr.themes.GoogleFont("Source Sans Pro"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        )
    ),
) as demo:
    gr.HTML("""
        <a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">
            < 返回成果網站首頁
        </a>
        """)
    gr.Markdown(
        """
        # 原住民族語言研究發展基金會族語語音辨識系統
        本辨識系統在讀取檔案後，可自動判斷族語別，請將檔案拖放到下方上傳處，點擊「開始辨識」，流程請見[操作手冊](static/操作手冊｜原住民族語言研究發展基金會族語語音辨識系統.pdf)。\\
        當上傳檔案較大時，請靜待辨識結果產生。
        """
    )

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

    with gr.Row(equal_height=True):
        gr.HTML(
            "<div>"
            "<hr>"
            "<p class='text-center'>Copy &copy; {} "
            "財團法人原住民族語言研究發展基金會 版權所有</p>"
            "</div>".format(datetime.now().year))

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300):
            gr.HTML("""
                <img class='img-fluid' src='/gradio_api/file=static/image/ilrdf-logo.png'
                    alt='財團法人原住民族語言研究發展基金會logo'>
                """)
        with gr.Column(scale=1, min_width=300):
            gr.HTML("""
                <p>電話：(02)2341-8508</p>
                <p>傳真：(02)2341-8256</p>
                <p>信箱：ilrdf@ilrdf.org.tw</p>
                <p>地址：100029台北市中正區羅斯福路一段63號</p>
                """)
        with gr.Column(scale=1, min_width=300):
            gr.HTML("""
                    <p><a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">著作權聲明</a></p>
                    <p><a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">網站使用條款</a></p>
                """)


demo.launch(allowed_paths=['ilrdf-logo.png'])

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, demo, path="", allowed_paths=['ilrdf-logo.png'])

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
