import os
import tempfile
import gradio as gr

from whisper import load_audio, load_model

SAMPLING_RATE = 16000
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

model = load_model(
    "formospeech/whisper-large-v2-formosan-all-ct2",
    device="cuda",
    asr_options={"word_timestamps": True},
)

with gr.Blocks(
    title="族語語音辨識系統 - 原住民族語言研究發展基金會",
) as demo:
    gr.Markdown(
        """
        # 原住民族語言研究發展基金會族語語音辨識系統
        本辨識系統在讀取檔案後，可自動判斷族語別，請將檔案拖放到下方上傳處，點擊「開始辨識」，流程請見操作手冊。\\
        當上傳檔案較大時，請靜待辨識結果產生。
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="族語影片",sources="upload")
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
            suffix=".srt", delete=False, mode="w", encoding="utf-8"
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

demo.launch()
