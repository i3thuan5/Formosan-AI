import concurrent.futures
from pathlib import Path

import gradio as gr
import torch
from gradio_client.exceptions import AppError
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from formosan_languages import FORMOSAN_LANGUAGES_MAP
from tts_client import TTS_TIMEOUT_SECONDS, TtsClient
from utils import render_demo


ETHNICITIES = sorted(set([k.split("_")[0]
                          for k in FORMOSAN_LANGUAGES_MAP.keys()]))

CODE_TO_LANGUAGE = {v: k for k, v in FORMOSAN_LANGUAGES_MAP.items()}

MODEL_NAME = "ithuan/nllb-600m-formosan-all-finetune-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_languages_by_ethnicity(ethnicity: str):
    return [
        (k, v)
        for k, v in FORMOSAN_LANGUAGES_MAP.items()
        if k.split("_")[0] == ethnicity
    ]


def translate(text: str, src_lang: str, tgt_lang: str):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    input_tokens = (
        tokenizer(
            text, return_tensors="pt").input_ids[0].cpu().numpy().tolist()
    )
    translated = model.generate(
        input_ids=torch.tensor([input_tokens]).to(device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=5000,
        num_return_sequences=1,
        num_beams=5,
        # repetition blocking works better if this number is below num_beams
        no_repeat_ngram_size=4,
        renormalize_logits=True,  # recompute token probabilities after banning the repetitions
    )

    translated = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated


tts_client = TtsClient()


def synthesize(text: str, tgt_lang: str):
    text = text.strip()
    if len(text) == 0:
        raise gr.Error("請先翻譯或輸入族語文字。")

    job = None
    try:
        job = tts_client.get().submit(
            CODE_TO_LANGUAGE[tgt_lang],
            text,
            api_name="/synthesize",
        )
        audio_path = Path(job.result(timeout=TTS_TIMEOUT_SECONDS))
        audio = audio_path.read_bytes()
    except concurrent.futures.TimeoutError:
        if job is not None:
            job.cancel()
        raise gr.Error("現在使用人數眾多，請稍候再試")
    except AppError as e:
        raise gr.Error(e.message)
    except Exception:
        tts_client.reset()
        raise gr.Error("語音合成服務暫時無法使用，請稍候再試")

    # 回傳 bytes，讓 gr.Audio 存進 Gradio 快取（delete_cache 會清）；
    # gradio_client 下載的原檔不在快取追蹤範圍內，要自己刪掉
    audio_path.unlink(missing_ok=True)
    try:
        audio_path.parent.rmdir()
    except OSError:
        pass

    return audio


with render_demo(
    demo_md_filename="DEMO.md",
    js="""
        function run_mt_block(){

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
            remove_gradio5_iframe_issue61();
            add_overflow_menu_toggler_innertext();
        }
        """,
) as demo:

    with gr.Tab("族語 ⮕ 華語"):
        to_zh_ethnicity = gr.Radio(
            label="族別",
            choices=ETHNICITIES,
            value="阿美",
        )
        to_zh_src_lang = gr.Radio(
            label="語別",
            choices=get_languages_by_ethnicity(to_zh_ethnicity.value),
            value=get_languages_by_ethnicity(to_zh_ethnicity.value)[0][1],
            interactive=len(get_languages_by_ethnicity(
                to_zh_ethnicity.value)) > 1,
        )
        to_zh_tgt_lang = gr.Text(
            value="zho_Hant", visible=False, interactive=False)
        to_zh_input_text = gr.Textbox(label="原文", lines=6)
        to_zh_btn = gr.Button("翻譯", variant="primary")
        to_zh_output = gr.Textbox(label="翻譯結果", lines=6)

        to_zh_ethnicity.change(
            lambda ethnicity: gr.Radio(
                choices=get_languages_by_ethnicity(ethnicity),
                value=get_languages_by_ethnicity(ethnicity)[0][1],
                interactive=len(get_languages_by_ethnicity(ethnicity)) > 1,
            ),
            inputs=to_zh_ethnicity,
            outputs=to_zh_src_lang,
            api_name="to_zh_languages",
        )

        to_zh_btn.click(
            translate,
            inputs=[to_zh_input_text, to_zh_src_lang, to_zh_tgt_lang],
            outputs=to_zh_output,
        )

    with gr.Tab("華語 ⮕ 族語"):
        to_formosan_src_lang = gr.Text(
            value="zho_Hant", visible=False, interactive=False
        )
        to_formosan_ethnicity = gr.Radio(
            label="族別",
            choices=ETHNICITIES,
            value="阿美",
        )
        to_formosan_tgt_lang = gr.Radio(
            label="語別",
            choices=get_languages_by_ethnicity(to_formosan_ethnicity.value),
            value=get_languages_by_ethnicity(
                to_formosan_ethnicity.value)[0][1],
            interactive=len(get_languages_by_ethnicity(
                to_formosan_ethnicity.value))
            > 1,
        )

        to_formosan_input_text = gr.Textbox(label="原文", lines=6)
        to_formosan_btn = gr.Button("翻譯", variant="primary")
        to_formosan_output = gr.Textbox(label="翻譯結果", lines=6)
        to_formosan_tts_btn = gr.Button("合成語音")
        to_formosan_audio = gr.Audio(
            label="合成結果", show_share_button=False, show_download_button=True
        )

        to_formosan_ethnicity.change(
            lambda ethnicity: gr.Radio(
                choices=get_languages_by_ethnicity(ethnicity),
                value=get_languages_by_ethnicity(ethnicity)[0][1],
                interactive=len(get_languages_by_ethnicity(ethnicity)) > 1,
            ),
            inputs=to_formosan_ethnicity,
            outputs=to_formosan_tgt_lang,
            api_name="to_formosan_languages",
        )

        # 按翻譯時先清掉舊的合成音檔，避免和新譯文對不上
        to_formosan_btn.click(
            lambda: None,
            outputs=to_formosan_audio,
            api_name=False,
        )
        to_formosan_btn.click(
            translate,
            inputs=[to_formosan_input_text,
                    to_formosan_src_lang, to_formosan_tgt_lang],
            outputs=to_formosan_output,
        )

        to_formosan_tts_btn.click(
            synthesize,
            inputs=[to_formosan_output, to_formosan_tgt_lang],
            outputs=to_formosan_audio,
            api_name="synthesize",
        )
