import json

import gradio as gr
from gradio import processing_utils
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from vosk import KaldiRecognizer, Model


def load_vosk(model_id: str):
    model_dir = snapshot_download(model_id)
    return Model(model_path=model_dir)


OmegaConf.register_new_resolver("load_vosk", load_vosk)

models_config = OmegaConf.to_object(OmegaConf.load("configs/models.yaml"))


def automatic_speech_recognition(model_id: str, dialect_id: str, audio_data: str):
    if isinstance(models_config[model_id]["model"], dict):
        model = models_config[model_id]["model"][dialect_id]
    else:
        model = models_config[model_id]["model"]

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


def when_model_selected(model_id: str):
    model_config = models_config[model_id]

    if "dialect_mapping" not in model_config:
        return gr.update(visible=False)

    dialect_drop_down_choices = [
        (k, v) for k, v in model_config["dialect_mapping"].items()
    ]

    return gr.update(
        choices=dialect_drop_down_choices,
        value=dialect_drop_down_choices[0][1],
        visible=True,
    )

def get_title():
    with open("DEMO.md") as tong:
        return tong.readline().strip('# ')

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
            (k, v)
            for k, v in models_config[default_model_id]["dialect_mapping"].items()
        ],
        value=list(models_config[default_model_id]["dialect_mapping"].values())[0],
        label="族別",
    )

    model_drop_down.input(
        when_model_selected,
        inputs=[model_drop_down],
        outputs=[dialect_drop_down],
    )

    audio_source = gr.Audio(
        label="上傳或錄音",
        type="numpy",
        format="wav",
        waveform_options=gr.WaveformOptions(
            sample_rate=16000,
        ),
    )

    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    gr.Interface(
        automatic_speech_recognition,
        inputs=[model_drop_down, dialect_drop_down, audio_source],
        outputs=[
            gr.Text(interactive=True, label="辨識結果"),
        ],
        allow_flagging="auto",
    )

    gr.Examples(
        [
            [
                "formosan_ami",
                "南勢",
                processing_utils.audio_from_file(
                    "examples/cb52eb9457a0b74abcf02da6253b29e37f44ee6f.wav"
                ),
                "U payniyaru’ nu pangcah i matiya, u ina haw ku miterungay, mikadavu ku vavainay i vavahiyan a luma’.",
                "阿美族的原始社會，是以女人為主的母系社會，男子授室入贅女家。",
            ],
            [
                "formosan_ami",
                "秀姑巒",
                processing_utils.audio_from_file(
                    "examples/9954bc6c934e098dd9900e1f6efc56223903b9ec.wav"
                ),
                "saka mafana’ ko ina ato mama^ no wawa, patayra han i faki anoca^ i akong no wawa^, somad han to no faki^ ko ngangan haw i.",
                "父母一眼就看出有問題，就送到長輩的住處請他查看，當長輩將名字更換了之後。",
            ],
            [
                "formosan_ami",
                "海岸",
                processing_utils.audio_from_file(
                    "examples/c9080c15a60953ee6f2b099a7e3036846583dce6.wav"
                ),
                "Orasaka ora “pataloma’” hananay a sowal, pakalafi han no Pangcah, todongay pakalafi to malinaay, nika oni pataloma’ hananay, manga’ay misaparod han ko sowal.",
                "因此「結婚」一詞，阿美族稱pakalafi，有「請吃晚餐」的意思，但較正式的用法是pataloma’，直譯為「成家」。",
            ],
            [
                "formosan_ami",
                "馬蘭",
                processing_utils.audio_from_file(
                    "examples/eb3364be43c8c133c9bc8cd71f1925aa20a66cc0.wav"
                ),
                "O sata’angayay a pisanga’an to tilong ko Tafalong itiya ho, mapaliwal i kasaniyaroaro’ ko misatilongan to sakacaloway no finawlan i ’orip a lalosidan.",
                "而太巴塱部落則是當時最大的製造陶埸域，供應各部落族人日常生活的陶器用品。",
            ],
            [
                "formosan_sdq",
                "德固達亞",
                processing_utils.audio_from_file(
                    "examples/b02ee31b7dee33bc9195c5b201b2943610b6308f.wav"
                ),
                "Pure macu, ani naq baso ciida we ini snagi beras na, asi hrigi ribo ma psaan rqeda baro, ciida ka seengun posa qsiya.",
                "烹煮小米（粟）、黍時，通常不須清洗就直接入鍋，移置爐灶上再加水。",
            ],
            [
                "formosan_trv",
                "",
                processing_utils.audio_from_file(
                    "examples/d76cb5e64a2ba1bade35edd0d8b12262c27707a7.wav"
                ),
                "Pthangan hangan Truku brah na siida o mniq ska hangan Embgala. Hangan ta siida o Embgala hraan hidaw sun.",
                "太魯閣族正名之前太魯閣族被編入泰雅爾族，當時的名字被稱為東部泰雅爾族。",
            ],
            [
                "formosan_pwn",
                "東",
                processing_utils.audio_from_file(
                    "examples/ef5780bceb44a41368a831513925cc59ebcfe14f.wav"
                ),
                "anema sikavaljualjut na sepaiwan kasicuayan, mavan a semualap ta cemel ta kasiv sa ljamayi sa sanqumayi, kata qemaljup tjepana.",
                "過去排灣族群靠甚麼維生呢？排灣族群的經濟生產是以「山田燒墾」的農耕為主，狩獵和捕魚為副業。",
            ],
        ],
        label="範例",
        inputs=[
            dialect_drop_down,
            gr.Text(label="方言", visible=False),
            audio_source,
            gr.Text(label="族語", visible=False),
            gr.Text(label="中文", visible=False),
        ],
    )


demo.launch()
