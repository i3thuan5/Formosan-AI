import json

import gradio as gr
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from vosk import KaldiRecognizer, Model

from utils import render_demo


def load_vosk(model_id: str):
    model_dir = snapshot_download(model_id)
    return Model(model_path=model_dir)


OmegaConf.register_new_resolver("load_vosk", load_vosk)

models_config = OmegaConf.load("configs/models.yaml")

DEFAULT_MODEL = OmegaConf.to_object(
    models_config[list(models_config.keys())[0]])


TITLE = "族語語音辨識系統 - 族語AI成果網站"


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


with render_demo(
    title=TITLE,
    js="""
        function run_asr_kaldi_block(){
            function change_fieldset_span_tag_to_legend(){
                const fieldsets = document.getElementsByTagName('fieldset');
                for(let i=0; i<fieldsets.length; i++){
                    const parentNode=fieldsets[i];
                    const spans=parentNode.querySelectorAll("span[data-testid='block-info']");
                    if(spans.length){
                        const span=spans[0];
                        const legend=document.createElement('legend');
                        for (let i = 0; i < span.attributes.length; i++) {
                            const attribute = span.attributes[i];
                            legend.setAttribute(attribute.name, attribute.value);
                          }
                        legend.innerHTML=span.innerHTML;
                        legend.classList.add('sa-legend');
                        parentNode.insertBefore(legend, parentNode.firstChild);
                        parentNode.removeChild(span);
                        if (parentNode.parentNode){
                            parentNode.parentNode.classList.add('sa-no-bg');
                        }
                    }
                }
            }
            function remove_gradio5_iframe_issue61() {
                const iframes = document.querySelectorAll('iframe');
                iframes.forEach(iframe => {
                    const parent = iframe.parentNode;
                    if (parent) {
                      parent.removeChild(iframe);
                    }
                });
            }
            change_fieldset_span_tag_to_legend();
            remove_gradio5_iframe_issue61();
        }
    """
) as demo:

    gr.HTML(value=f"<h1 id='main'>{TITLE}</h1>")

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
