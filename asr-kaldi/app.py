import json
from pathlib import Path
from datetime import datetime

import gradio as gr
from gradio.themes.utils.colors import Color
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from vosk import KaldiRecognizer, Model


def load_vosk(model_id: str):
    model_dir = snapshot_download(model_id)
    return Model(model_path=model_dir)


OmegaConf.register_new_resolver("load_vosk", load_vosk)

models_config = OmegaConf.load("configs/models.yaml")

DEFAULT_MODEL = OmegaConf.to_object(
    models_config[list(models_config.keys())[0]])


def automatic_speech_recognition(dialect_id: str, audio_data: str):
    if isinstance(DEFAULT_MODEL["model"], dict):
        model = DEFAULT_MODEL["model"][dialect_id]
    else:
        model = DEFAULT_MODEL["model"]

    sample_rate, audio_array = audio_data
    if audio_array.ndim == 2:
        audio_array = audio_array[:, 0]

    audio_bytes = audio_array.tobytes()

    rec = KaldiRecognizer(model, sample_rate)

    rec.SetWords(True)

    results = []

    for start in range(0, len(audio_bytes), 4000):
        end = min(start + 4000, len(audio_bytes))
        data = audio_bytes[start:end]
        if rec.AcceptWaveform(data):
            raw_result = json.loads(rec.Result())
            results.append(raw_result)

    final_result = json.loads(rec.FinalResult())
    results.append(final_result)

    filtered_lines = []

    for result in results:
        if len(result["text"]) > 0:
            if dialect_id == "formosan_ami":
                result["text"] = result["text"].replace("u", "o")
            filtered_lines.append(result["text"])

    return (", ".join(filtered_lines) + ".").capitalize()


def get_title():
    with open("DEMO.md") as tong:
        return tong.readline().strip("# ")


sa_orange_color = Color(
    name="sa_orange",
    c50="#F4855D",  # Lightest shade
    c100="#F7AA8E",
    c200="#F69D7D",
    c300="#F5916D",
    c400="#F4855D",
    c500="#D04410",  # Main color
    c600="#AC370C",
    c700="#BB3D0E",
    c800="#A6360D",
    c900="#92300B",
    c950="#7D290A"   # Darkest shade
)


sa_zinc_color = Color(
    name="sa_grey",
    c50="#EEEEEF",  # Lightest shade
    c100="#CBCBCE",
    c200="#BABABD",
    c300="#97979D",
    c400="#63636B",
    c500="#52525B",  # Main color
    c600="#4A4A52",
    c700="#424249",
    c800="#19191B",
    c900="#101012",
    c950="#080809"   # Darkest shade
)


gr.set_static_paths(paths=[Path.cwd().absolute() / "static" / "image"])

demo = gr.Blocks(
    title=get_title(),
    css_paths=[Path(__file__).parent / 'static' / 'css' / 'app.css', ],
    theme=gr.themes.Default(
        primary_hue=sa_orange_color,
        neutral_hue=sa_zinc_color,
        font=(
            "tauhu-oo",
            gr.themes.GoogleFont("Source Sans Pro"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        )
    )
)

with demo:
    gr.HTML("""
        <a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">
            < 返回成果網站首頁
        </a>
        """)
    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    with gr.Row():
        with gr.Column():
            dialect_drop_down = gr.Radio(
                choices=[(k, v)
                         for k, v in DEFAULT_MODEL["dialect_mapping"].items()],
                value=list(DEFAULT_MODEL["dialect_mapping"].values())[0],
                label="步驟一:選擇族別",
            )

            audio_source = gr.Audio(
                label="步驟二:上傳待辨識音檔或點擊🎙️自行錄音",
                type="numpy",
                format="wav",
                waveform_options=gr.WaveformOptions(
                    sample_rate=16000,
                ),
                sources=["microphone", "upload"],
            )
            submit_button = gr.Button("步驟三:開始辨識", variant="primary")

        with gr.Column():
            output_textbox = gr.TextArea(interactive=True, label="辨識結果")

    submit_button.click(
        automatic_speech_recognition,
        inputs=[dialect_drop_down, audio_source],
        outputs=[output_textbox],
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


demo.launch(
    allowed_paths=['ilrdf-logo.png'],
    favicon_path=Path(__file__).parent / 'static' / 'favicon' / 'logo.svg'
)
