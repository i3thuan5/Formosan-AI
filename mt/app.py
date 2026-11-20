import concurrent.futures
from pathlib import Path

import gradio as gr
import torch
from gradio_client.exceptions import AppError
from transformers import AutoModelForSeq2SeqLM
from feedback import FEEDBACK_MAX_LENGTH, save_feedback
from languages import ETHNICITIES, FORMOSAN_LANGUAGES_MAP, LANGUAGE_CODES
from nllb import load_tokenizer, map_sentence_starts
from orthography import normalize
from tts_client import TTS_TIMEOUT_SECONDS, TtsClient
from utils import render_demo, tsoo_khokpih_button


CODE_TO_LANGUAGE = {v: k for k, v in FORMOSAN_LANGUAGES_MAP.items()}

MODEL_NAME = "ILRDF/nllb-600m-formosan-all-finetune-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = load_tokenizer(MODEL_NAME)


def get_languages_by_ethnicity(ethnicity: str):
    return [
        (k, v)
        for k, v in FORMOSAN_LANGUAGES_MAP.items()
        if k.split("_")[0] == ethnicity
    ]


def translate(text: str, src_lang: str, tgt_lang: str):
    text = normalize(text, LANGUAGE_CODES.get(src_lang))
    text = map_sentence_starts(text, src_lang, str.lower)
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=512,
        num_return_sequences=1,
        num_beams=5,
    )

    translated = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated = map_sentence_starts(translated, tgt_lang, str.upper)

    return normalize(translated, LANGUAGE_CODES.get(tgt_lang))


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


FEEDBACK_SERVICE = "mt"
FEEDBACK_TO_ZH = "族翻華"
FEEDBACK_TO_FORMOSAN = "華翻族"
FEEDBACK_OPTIONS = (FEEDBACK_TO_ZH, FEEDBACK_TO_FORMOSAN)
FEEDBACK_RATES = ("可用", "需修改", "我不確定")


def build_feedback_survey():
    # 翻譯了才顯示；snapshot 存翻譯當時ê語別、原文、譯文，
    # 免得使用者後來改 textbox 內容，存著對袂齊ê資料
    snapshot = gr.State(None)
    # 送出了後 form khàm 起來，干焦賰 thanks 一逝
    with gr.Column(visible=False) as survey:
        with gr.Group() as form:
            rate = gr.Radio(
                label="這個翻譯結果可以用嗎？",
                choices=list(FEEDBACK_RATES),
                value=None,
            )
            feedback = gr.Textbox(
                label=f"修改建議（選填，最多 {FEEDBACK_MAX_LENGTH} 字）",
                lines=3,
                max_length=FEEDBACK_MAX_LENGTH,
            )
            submit_btn = gr.Button("送出回饋")
        thanks = gr.Markdown("感謝您的回饋！", visible=False)

    submit_btn.click(
        submit_feedback,
        inputs=[snapshot, rate, feedback],
        outputs=[form, thanks],
        api_visibility="private",
    )
    return {
        "snapshot": snapshot,
        "survey": survey,
        "form": form,
        "rate": rate,
        "feedback": feedback,
        "thanks": thanks,
    }


def show_feedback_survey(survey, option, input_text, lang_code, output_text):
    # 譯文空白就毋免問
    snapshot = None
    if output_text.strip():
        snapshot = {
            "option": option,
            "language": CODE_TO_LANGUAGE[lang_code],
            "input": input_text,
            "output": output_text,
        }
    return {
        survey["snapshot"]: snapshot,
        survey["survey"]: gr.Column(visible=snapshot is not None),
        survey["form"]: gr.Group(visible=True),
        survey["rate"]: gr.Radio(value=None),
        survey["feedback"]: gr.Textbox(value=""),
        survey["thanks"]: gr.Markdown(visible=False),
    }


def bind_feedback_survey(translate_event, translate_btn, survey, option, input_text, lang_code, output):
    # 按翻譯時先 khàm 起來，翻譯成功才 koh 顯示新ê問卷
    translate_btn.click(
        lambda: gr.Column(visible=False),
        outputs=survey["survey"],
        api_visibility="private",
        queue=False,
    )
    translate_event.success(
        lambda text, code, result: show_feedback_survey(survey, option, text, code, result),
        inputs=[input_text, lang_code, output],
        outputs={
            survey["snapshot"],
            survey["survey"],
            survey["rate"],
            survey["feedback"],
            survey["form"],
            survey["thanks"],
        },
        api_visibility="private",
        queue=False,
    )


def submit_feedback(snapshot, rate, feedback):
    if snapshot is None:
        raise gr.Error("請先翻譯。")
    if snapshot["option"] not in FEEDBACK_OPTIONS:
        raise gr.Error("翻譯方向有誤，請重新翻譯。")
    if rate not in FEEDBACK_RATES:
        raise gr.Error("請選擇翻譯結果是否可用。")
    if len(feedback or "") > FEEDBACK_MAX_LENGTH:
        raise gr.Error(f"修改建議最多 {FEEDBACK_MAX_LENGTH} 字。")
    try:
        save_feedback(
            FEEDBACK_SERVICE,
            snapshot["language"],
            snapshot["input"],
            snapshot["output"],
            rate,
            feedback,
            option=snapshot["option"],
        )
    except OSError:
        raise gr.Error("回饋儲存失敗，請稍候再試。")
    # 送出了就 khàm 問卷，避免仝一个翻譯重複送
    return gr.Group(visible=False), gr.Markdown(visible=True)


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
            label="族語別",
            choices=ETHNICITIES,
            value="阿美",
            elem_classes=["sa-radio-equal"],
        )
        to_zh_src_lang = gr.Radio(
            label="語言別",
            choices=get_languages_by_ethnicity(to_zh_ethnicity.value),
            value=get_languages_by_ethnicity(to_zh_ethnicity.value)[0][1],
            interactive=len(get_languages_by_ethnicity(
                to_zh_ethnicity.value)) > 1,
        )
        to_zh_tgt_lang = gr.Text(
            value="zho_Hant", visible=False, interactive=False)
        to_zh_input_text = gr.Textbox(label="原文", lines=6)
        to_zh_btn = gr.Button("翻譯", variant="primary")
        with gr.Column(elem_classes=["sa-copy-wrap"]):
            to_zh_output = gr.Textbox(label="翻譯結果", lines=6)
            tsoo_khokpih_button(to_zh_output, "複製譯文", "to-zh-output-kopi")
        to_zh_survey = build_feedback_survey()

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

        to_zh_translate_event = to_zh_btn.click(
            translate,
            inputs=[to_zh_input_text, to_zh_src_lang, to_zh_tgt_lang],
            outputs=to_zh_output,
        )
        # 族語 ⮕ 華語：language 記原文ê族語
        bind_feedback_survey(
            to_zh_translate_event,
            to_zh_btn,
            to_zh_survey,
            FEEDBACK_TO_ZH,
            to_zh_input_text,
            to_zh_src_lang,
            to_zh_output,
        )

    with gr.Tab("華語 ⮕ 族語"):
        to_formosan_src_lang = gr.Text(
            value="zho_Hant", visible=False, interactive=False
        )
        to_formosan_ethnicity = gr.Radio(
            label="族語別",
            choices=ETHNICITIES,
            value="阿美",
            elem_classes=["sa-radio-equal"],
        )
        to_formosan_tgt_lang = gr.Radio(
            label="語言別",
            choices=get_languages_by_ethnicity(to_formosan_ethnicity.value),
            value=get_languages_by_ethnicity(
                to_formosan_ethnicity.value)[0][1],
            interactive=len(get_languages_by_ethnicity(
                to_formosan_ethnicity.value))
            > 1,
        )

        to_formosan_input_text = gr.Textbox(label="原文", lines=6)
        to_formosan_btn = gr.Button("翻譯", variant="primary")
        with gr.Column(elem_classes=["sa-copy-wrap"]):
            to_formosan_output = gr.Textbox(label="翻譯結果", lines=6)
            tsoo_khokpih_button(
                to_formosan_output, "複製譯文", "to-formosan-output-kopi")
        to_formosan_survey = build_feedback_survey()
        to_formosan_tts_btn = gr.Button("合成語音")
        to_formosan_audio = gr.Audio(
            label="合成結果", buttons=["download"]
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
            api_visibility="private",
            queue=False,
        )
        to_formosan_translate_event = to_formosan_btn.click(
            translate,
            inputs=[to_formosan_input_text,
                    to_formosan_src_lang, to_formosan_tgt_lang],
            outputs=to_formosan_output,
        )
        # 華語 ⮕ 族語：language 記譯文ê族語
        bind_feedback_survey(
            to_formosan_translate_event,
            to_formosan_btn,
            to_formosan_survey,
            FEEDBACK_TO_FORMOSAN,
            to_formosan_input_text,
            to_formosan_tgt_lang,
            to_formosan_output,
        )

        to_formosan_tts_btn.click(
            synthesize,
            inputs=[to_formosan_output, to_formosan_tgt_lang],
            outputs=to_formosan_audio,
            api_name="synthesize",
        )

    with gr.Accordion("API 變更說明", open=False):
        gr.Markdown(
            """
202609: 下列 endpoint 有變更。

| 功能 | 原本 | 現在 |
| --- | --- | --- |
| 「族語 ⮕ 華語」切換族語別 | `/lambda` | `/to_zh_languages` |
| 「華語 ⮕ 族語」切換族語別 | `/lambda_1` | `/to_formosan_languages` |
| 「華語 ⮕ 族語」合成語音 | 未提供 | `/synthesize` |

`/translate`（族語 ⮕ 華語）與 `/translate_1`（華語 ⮕ 族語）沒有變更。

語言別選項會隨族語別改變，要翻譯或合成阿美以外的語言別時，請在同一個 session 先呼叫切換族語別的 endpoint，例如：

```python
from gradio_client import Client

client = Client("https://ai-labs.ilrdf.org.tw/kari-seejiq-tnpusu-ai-hmjil/")
client.predict("泰雅", api_name="/to_formosan_languages")
text = client.predict("我們以前的生活", "zho_Hant", "tay_Wand", api_name="/translate_1")
audio = client.predict(text, "tay_Wand", api_name="/synthesize")
```
"""
        )
