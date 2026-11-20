import re
import threading
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from formosan_tts import FormosanTTS
from languages import ETHNICITIES, FORMOSAN_LANGUAGES_MAP, LANGUAGE_CODES
from utils import render_demo

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


models_config = OmegaConf.to_object(OmegaConf.load("configs/models.yaml"))
DEFAULT_MODEL_ID = list(models_config.keys())[0]
model_config = models_config[DEFAULT_MODEL_ID]

model_dir = snapshot_download(model_config["repo_id"], revision=model_config["revision"])
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = FormosanTTS(model_dir, device=device)

# 同時兩個合成會搶同一個模型與 GPU。Gradio 的 concurrency_limit 是每個 event
# listener 各自算，網頁、自己當配音員、mt 呼叫的 synthesize 三個端點會同時執行，
# 所以用全域鎖讓所有合成排同一個隊。
TTS_LOCK = threading.Lock()

refs_config = OmegaConf.to_object(OmegaConf.load("configs/refs.yaml"))

# 語別名稱（如 "阿美_海岸"）→ 模型用的語別代碼（如 "ami-x-pswl"）
LANGUAGE_NAME_TO_CODE = {}
for _name, _nllb_code in FORMOSAN_LANGUAGES_MAP.items():
    LANGUAGE_NAME_TO_CODE[_name] = LANGUAGE_CODES[_nllb_code]


def get_languages_by_ethnicity(ethnicity: str):
    languages = []
    for name in LANGUAGE_NAME_TO_CODE:
        if name.split("_")[0] == ethnicity:
            languages.append(name)
    return languages


@gpu_decorator
def infer(ref_audio, gen_text, language):
    if not ref_audio:
        raise gr.Error("請先錄製或上傳參考音檔。")

    try:
        with TTS_LOCK:
            wav, sample_rate = tts_model.generate(
                gen_text, LANGUAGE_NAME_TO_CODE[language], ref_audio
            )
    except ValueError as e:
        # FormosanTTS 遇到該語別 G2P 表沒有的字元會丟 ValueError
        raise gr.Error(f"文字中有無法轉換的字元：{e}")

    return sample_rate, wav


def normalize_gen_text(gen_text_input: str) -> str:
    # 先移除引號再判斷句尾，避免 `kita."` 補成 `kita.".`，移除引號後變成 `kita..`
    gen_text_input = re.sub(r"[\"“”「」]", "", gen_text_input).strip()
    if len(gen_text_input) == 0:
        raise gr.Error("請勿輸入空字串。")

    if gen_text_input[-1] not in [".", "?", "!", ",", ";", ":"]:
        gen_text_input += "."

    return gen_text_input


with render_demo(
    demo_md_filename="DEMO.md",
    css_paths=[Path(__file__).parent / 'static' / 'css' / 'app.css', ],
    js="""
        function run_tts_block(){
            function addButtonsEvent() {
                const buttons = document.querySelectorAll("#head-html-block button");
                buttons.forEach(button => {
                    button.addEventListener("click", () => {
                        navigator.clipboard.writeText(button.innerText);
                    });
                });
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

            function add_overflow_menu_toggler_innertext() {
                const menus = document.querySelectorAll('.overflow-menu');
                menus.forEach(menu => {
                    const button = menu.querySelector('button');
                    if (button) {
                        const info = document.createElement('span');
                        info.innerText = '其餘頁籤選項';
                        info.classList.add('sa-visually-hidden');
                        button.appendChild(info);
                    }
                });
            }
            addButtonsEvent();
            remove_gradio5_iframe_issue61();
            add_overflow_menu_toggler_innertext();
        }
        """,
) as demo:

    gr.HTML(
        "特殊符號請複製使用（滑鼠點擊即可複製）："
        "<button class='sa-btn'>é</button>"
        "<button class='sa-btn'>ṟ</button>"
        "<button class='sa-btn'>ɨ</button>"
        "<button class='sa-btn'>ʉ</button>",
        elem_id="head-html-block",
        elem_classes=["html-no-start-padding"],
        # Gradio 6 的 padding 預設改成 False；html-no-start-padding 是靠負邊距抵銷 padding，要維持 True
        padding=True,
    )

    with gr.Tab("預設配音員"):
        with gr.Row():
            with gr.Column():
                default_speaker_ethnicity = gr.Radio(
                    choices=ETHNICITIES,
                    label="步驟一：選擇族語別",
                    value="阿美",
                    elem_classes=["sa-radio-equal"],
                )

                def get_refs_by_perfix(prefix: str):
                    return [r for r in refs_config.keys() if r.startswith(prefix)]

                default_speaker_refs = gr.Radio(
                    choices=get_refs_by_perfix(
                        default_speaker_ethnicity.value),
                    label="步驟二：選擇配音員",
                    value=get_refs_by_perfix(
                        default_speaker_ethnicity.value)[0],
                )

                default_speaker_gen_text_input = gr.Textbox(
                    label="步驟三：輸入文字（上限 300 字元）",
                    value="",
                )

                default_speaker_generate_btn = gr.Button(
                    "步驟四：開始合成", variant="primary"
                )

            with gr.Column():
                default_speaker_audio_output = gr.Audio(
                    label="合成結果", buttons=["download"]
                )

    with gr.Tab("自己當配音員"):
        with gr.Row():
            with gr.Column():
                custom_speaker_ethnicity = gr.Radio(
                    choices=ETHNICITIES,
                    label="步驟一：選擇族語別與語言別",
                    value="阿美",
                    elem_classes=["sa-radio-equal"],
                )

                custom_speaker_language = gr.Radio(
                    choices=get_languages_by_ethnicity(
                        custom_speaker_ethnicity.value),
                    value=get_languages_by_ethnicity(
                        custom_speaker_ethnicity.value)[0],
                    show_label=False,
                )

                custom_speaker_ref_text_input = gr.Textbox(
                    value=refs_config[
                        get_refs_by_perfix(custom_speaker_language.value)[0]
                    ]["text"],
                    interactive=False,
                    label="步驟二：點選🎙️錄製一段清楚的聲音（建議 3 秒以上，可念下方句子），或上傳音檔",
                    elem_classes="textonly",
                )

                custom_speaker_audio_input = gr.Audio(
                    type="filepath",
                    sources=["microphone", "upload"],
                    waveform_options=gr.WaveformOptions(
                        sample_rate=24000,
                    ),
                    label="錄製或上傳",
                )

                custom_speaker_gen_text_input = gr.Textbox(
                    label="步驟三：輸入合成文字（上限 300 字元）",
                    value="",
                )

                custom_speaker_generate_btn = gr.Button(
                    "步驟四：開始合成", variant="primary"
                )

            with gr.Column():
                custom_speaker_audio_output = gr.Audio(
                    label="合成結果", buttons=["download"]
                )

    default_speaker_ethnicity.change(
        lambda ethnicity: gr.Radio(
            choices=get_refs_by_perfix(ethnicity),
            value=get_refs_by_perfix(ethnicity)[0],
        ),
        inputs=[default_speaker_ethnicity],
        outputs=[default_speaker_refs],
    )

    def default_speaker_tts(
        ref: str,
        gen_text_input: str,
    ):
        language = re.sub(r"_[男女]聲[12]?", "", ref)
        ref_audio_input = refs_config[ref]["wav"]

        gen_text_input = normalize_gen_text(gen_text_input)

        return infer(ref_audio_input, gen_text_input, language)

    default_speaker_generate_btn.click(
        default_speaker_tts,
        inputs=[
            default_speaker_refs,
            default_speaker_gen_text_input,
        ],
        outputs=[default_speaker_audio_output],
    )

    def synthesize_by_language(
        language: str,
        text: str,
    ):
        # 給其他服務（如 mt）呼叫，用該語言別第一位配音員合成。
        # 用 Textbox 當輸入，才不會受 Radio choices 限制。
        refs = get_refs_by_perfix(language + "_")
        if len(refs) == 0:
            raise gr.Error(f"不支援的語言別：{language}")
        return default_speaker_tts(refs[0], text)

    synthesize_language_input = gr.Textbox(visible=False)
    synthesize_text_input = gr.Textbox(visible=False)
    synthesize_audio_output = gr.Audio(visible=False)
    synthesize_btn = gr.Button(visible=False)
    synthesize_btn.click(
        synthesize_by_language,
        inputs=[
            synthesize_language_input,
            synthesize_text_input,
        ],
        outputs=[synthesize_audio_output],
        api_name="synthesize",
    )

    custom_speaker_ethnicity.change(
        lambda ethnicity: gr.Radio(
            choices=get_languages_by_ethnicity(ethnicity),
            value=get_languages_by_ethnicity(ethnicity)[0],
            visible=len(get_languages_by_ethnicity(ethnicity)) > 1,
        ),
        inputs=[custom_speaker_ethnicity],
        outputs=[custom_speaker_language],
    )

    custom_speaker_language.change(
        lambda lang: gr.Textbox(
            value=refs_config[get_refs_by_perfix(lang)[0]]["text"],
        ),
        inputs=[custom_speaker_language],
        outputs=[custom_speaker_ref_text_input],
    )

    def custom_speaker_tts(
        language: str,
        ref_text_input: str,
        ref_audio_input: str,
        gen_text_input: str,
    ):
        # ref_text_input 只是顯示給使用者念的句子；chatterbox 不需要參考音檔的逐字稿。
        # 參數留著是為了維持 /custom_speaker_tts API 不變。
        gen_text_input = normalize_gen_text(gen_text_input)

        return infer(ref_audio_input, gen_text_input, language)

    custom_speaker_generate_btn.click(
        custom_speaker_tts,
        inputs=[
            custom_speaker_language,
            custom_speaker_ref_text_input,
            custom_speaker_audio_input,
            custom_speaker_gen_text_input,
        ],
        outputs=[custom_speaker_audio_output],
    )
