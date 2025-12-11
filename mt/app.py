import gradio as gr
import spaces
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import render_demo


FORMOSAN_LANGUAGES_MAP = {
    "阿美_海岸": "ami_Coas",
    "阿美_恆春": "ami_Heng",
    "阿美_馬蘭": "ami_Mala",
    "阿美_南勢": "ami_Sout",
    "阿美_秀姑巒": "ami_Xiug",
    "泰雅_四季": "tay_Four",
    "泰雅_賽考利克": "tay_Seko",
    "泰雅_萬大": "tay_Wand",
    "泰雅_汶水": "tay_Wens",
    "泰雅_宜蘭澤敖利": "tay_Yzea",
    "泰雅_澤敖利": "tay_Zeao",
    "布農_郡群": "bnn_Junq",
    "布農_卡群": "bnn_Kaqu",
    "布農_巒群": "bnn_Luan",
    "布農_丹群": "bnn_Tanq",
    "布農_卓群": "bnn_Zhuo",
    "卡那卡那富": "xnb_Kana",
    "噶瑪蘭": "ckv_Kava",
    "排灣_中": "pwn_Cent",
    "排灣_東": "pwn_East",
    "排灣_北": "pwn_Nrth",
    "排灣_南": "pwn_Sout",
    "卑南_建和": "pyu_Jian",
    "卑南_南王": "pyu_Nanw",
    "卑南_西群": "pyu_Xiqu",
    "卑南_知本": "pyu_Zhib",
    "魯凱_大武": "dru_Dawu",
    "魯凱_多納": "dru_Dona",
    "魯凱_東": "dru_East",
    "魯凱_茂林": "dru_Maol",
    "魯凱_萬山": "dru_Wans",
    "魯凱_霧台": "dru_Wuta",
    "拉阿魯哇": "sxr_Saar",
    "賽夏": "xsy_Sais",
    "撒奇萊雅": "szy_Saki",
    "賽德克_德鹿谷": "trv_Delu",
    "賽德克_都達": "trv_Duda",
    "賽德克_德固達雅": "trv_Tegu",
    "邵": "ssf_Thao",
    "太魯閣": "trv_Truk",
    "鄒": "tsu_Tsou",
    "雅美": "tao_Yami",
}

ETHNICITIES = sorted(set([k.split("_")[0] for k in FORMOSAN_LANGUAGES_MAP.keys()]))

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


@spaces.GPU
def translate(text: str, src_lang: str, tgt_lang: str):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    input_tokens = (
        tokenizer(text, return_tensors="pt").input_ids[0].cpu().numpy().tolist()
    )
    translated = model.generate(
        input_ids=torch.tensor([input_tokens]).to(device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=5000,
        num_return_sequences=1,
        num_beams=5,
        no_repeat_ngram_size=4,  # repetition blocking works better if this number is below num_beams
        renormalize_logits=True,  # recompute token probabilities after banning the repetitions
    )

    translated = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated


def get_title():
    with open("DEMO.md", encoding="utf-8") as tong:
        return tong.readline().strip("# ")


with render_demo(
    title=get_title(),
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
    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    with gr.Tab("族語 ⮕ 華語"):
        to_zh_ethnicity = gr.Dropdown(
            label="族別",
            choices=ETHNICITIES,
            value="阿美",
            filterable=False,
        )
        to_zh_src_lang = gr.Dropdown(
            label="語別",
            choices=get_languages_by_ethnicity(to_zh_ethnicity.value),
            value=get_languages_by_ethnicity(to_zh_ethnicity.value)[0][1],
            filterable=False,
            interactive=len(get_languages_by_ethnicity(to_zh_ethnicity.value)) > 1,
        )
        to_zh_tgt_lang = gr.Text(value="zho_Hant", visible=False, interactive=False)
        to_zh_input_text = gr.Textbox(label="原文", lines=6)
        to_zh_btn = gr.Button("翻譯")
        to_zh_output = gr.Textbox(label="翻譯結果", lines=6)

        to_zh_ethnicity.change(
            lambda ethnicity: gr.Dropdown(
                choices=get_languages_by_ethnicity(ethnicity),
                value=get_languages_by_ethnicity(ethnicity)[0][1],
                interactive=len(get_languages_by_ethnicity(ethnicity)) > 1,
            ),
            inputs=to_zh_ethnicity,
            outputs=to_zh_src_lang,
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
        to_formosan_ethnicity = gr.Dropdown(
            label="族別",
            choices=ETHNICITIES,
            value=ETHNICITIES[0],
            filterable=False,
        )
        to_formosan_tgt_lang = gr.Dropdown(
            label="語別",
            choices=get_languages_by_ethnicity(to_formosan_ethnicity.value),
            value=get_languages_by_ethnicity(to_formosan_ethnicity.value)[0][1],
            filterable=False,
            interactive=len(get_languages_by_ethnicity(to_formosan_ethnicity.value))
            > 1,
        )

        to_formosan_input_text = gr.Textbox(label="原文", lines=6)
        to_formosan_btn = gr.Button("翻譯", variant="primary")
        to_formosan_output = gr.Textbox(label="翻譯結果", lines=6)

        to_formosan_ethnicity.change(
            lambda ethnicity: gr.Dropdown(
                choices=get_languages_by_ethnicity(ethnicity),
                value=get_languages_by_ethnicity(ethnicity)[0][1],
                interactive=len(get_languages_by_ethnicity(ethnicity)) > 1,
            ),
            inputs=to_formosan_ethnicity,
            outputs=to_formosan_tgt_lang,
        )

        to_formosan_btn.click(
            translate,
            inputs=[to_formosan_input_text, to_formosan_src_lang, to_formosan_tgt_lang],
            outputs=to_formosan_output,
        )
