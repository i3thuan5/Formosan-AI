import re

import gradio as gr
import spaces
import torch
from omegaconf import OmegaConf
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def load_pipe(model_id: str):
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch_dtype,
        device=device,
    )


OmegaConf.register_new_resolver("load_pipe", load_pipe)

models_config = OmegaConf.to_object(OmegaConf.load("configs/models.yaml"))


@spaces.GPU
def automatic_speech_recognition(model_id: str, dialect_id: str, audio_file: str):
    model = models_config[model_id]["model"]

    generate_kwargs = {
        "task": "transcribe",
        "language": "id",
        "num_beams": 5,
    }
    if models_config[model_id]["dialect_mapping"] is not None:
        generate_kwargs["prompt_ids"] = torch.from_numpy(
            model.tokenizer.get_prompt_ids(dialect_id)
        ).to(device)

    result = model(audio_file, generate_kwargs=generate_kwargs)["text"].replace(
        f" {dialect_id}", ""
    )

    if result[-1] not in ".!?":
        result = result + "."

    sentences = re.split(r"[.!?] ", result)
    for i in range(len(sentences)):
        sentences[i] = sentences[i][0].upper() + sentences[i][1:]

    return " ".join(sentences)


def when_model_selected(model_id: str):
    model_config = models_config[model_id]

    if model_config["dialect_mapping"] is not None:
        dialect_drop_down_choices = [
            (k, v) for k, v in model_config["dialect_mapping"].items()
        ]

        return gr.update(
            choices=dialect_drop_down_choices,
            value=dialect_drop_down_choices[0][1],
        )
    else:
        return gr.update(visible=False)


def get_title():
    with open("DEMO.md") as tong:
        return tong.readline().strip("# ")


demo = gr.Blocks(
    title=get_title(),
    css="@import url(https://tauhu.tw/tauhu-oo.css);",
    theme=gr.themes.Default(
        font=(
            "tauhu-oo",
            gr.themes.GoogleFont("Source Sans Pro"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        )
    ),
)

with demo:
    default_model_id = list(models_config.keys())[0]
    model_drop_down = gr.Dropdown(
        models_config.keys(),
        value=default_model_id,
        label="模型",
    )

    dialect_drop_down = gr.Radio(
        choices=[
            "test"
        ],
        label="族別",
        visible=False,
    )

    model_drop_down.input(
        when_model_selected,
        inputs=[model_drop_down],
        outputs=[dialect_drop_down],
    )

    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    gr.Interface(
        automatic_speech_recognition,
        inputs=[
            model_drop_down,
            dialect_drop_down,
            gr.Audio(
                label="上傳或錄音",
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    sample_rate=16000,
                ),
            ),
        ],
        outputs=[
            gr.Text(interactive=False, label="辨識結果"),
        ],
        allow_flagging="auto",
    )

demo.launch()
