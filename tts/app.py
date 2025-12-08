import re
import tempfile
from importlib.resources import files
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
import torchcodec
from cached_path import cached_path
from omegaconf import OmegaConf

from ipa.ipa import g2p_object, text_to_ipa
from utils import render_demo

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

from f5_tts.infer.utils_infer import (
    device,
    hop_length,
    infer_process,
    load_checkpoint,
    load_vocoder,
    mel_spec_type,
    n_fft,
    n_mel_channels,
    ode_method,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    win_length,
)
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


vocoder = load_vocoder()


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
    fp16=False,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" or not fp16 else None
    model = load_checkpoint(model, ckpt_path, device,
                            dtype=dtype, use_ema=use_ema)

    return model


def load_f5tts(ckpt_path, vocab_path, old=False, fp16=False):
    ckpt_path = str(cached_path(ckpt_path))
    F5TTS_model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        text_mask_padding=not old,
        pe_attn_head=1 if old else None,
    )
    vocab_path = str(cached_path(vocab_path))
    return load_model(
        DiT,
        F5TTS_model_cfg,
        ckpt_path,
        vocab_file=vocab_path,
        use_ema=old,
        fp16=fp16,
    )


OmegaConf.register_new_resolver("load_f5tts", load_f5tts)

models_config = OmegaConf.to_object(OmegaConf.load("configs/models.yaml"))
refs_config = OmegaConf.to_object(OmegaConf.load("configs/refs.yaml"))
examples_config = OmegaConf.to_object(OmegaConf.load("configs/examples.yaml"))


DEFAULT_MODEL_ID = list(models_config.keys())[0]

ETHNICITIES = list(set([k.split("_")[0] for k in g2p_object.keys()]))


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence=False,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave = torchcodec.decoders.AudioDecoder(
                f.name).get_all_samples().data
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


def get_title():
    with open("DEMO.md", encoding="utf-8") as tong:
        return tong.readline().strip("# ")


with render_demo(
    title=get_title(),
    css_path=[Path(__file__).parent / 'static' / 'app.css', ],
    js="""
        function addButtonsEvent() {
            const buttons = document.querySelectorAll("#head-html-block button");
            buttons.forEach(button => {
                button.addEventListener("click", () => {
                    navigator.clipboard.writeText(button.innerText);
                });
            });
        }
        """,
) as demo:

    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    gr.HTML(
        "特殊符號請複製使用（滑鼠點擊即可複製）：<button>é</button> <button>ṟ</button> <button>ɨ</button> <button>ʉ</button>",
        padding=False,
        elem_id="head-html-block",
    )

    with gr.Tab("預設配音員"):
        with gr.Row():
            with gr.Column():
                default_speaker_ethnicity = gr.Dropdown(
                    choices=ETHNICITIES,
                    label="步驟一：選擇族別",
                    value="阿美",
                    filterable=False,
                )

                def get_refs_by_perfix(prefix: str):
                    return [r for r in refs_config.keys() if r.startswith(prefix)]

                default_speaker_refs = gr.Dropdown(
                    choices=get_refs_by_perfix(
                        default_speaker_ethnicity.value),
                    label="步驟二：選擇配音員",
                    value=get_refs_by_perfix(
                        default_speaker_ethnicity.value)[0],
                    filterable=False,
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
                    label="合成結果", show_share_button=False, show_download_button=True
                )

    with gr.Tab("自己當配音員"):
        with gr.Row():
            with gr.Column():
                custom_speaker_ethnicity = gr.Dropdown(
                    choices=ETHNICITIES,
                    label="步驟一：選擇族別與語別",
                    value="阿美",
                    filterable=False,
                )

                custom_speaker_language = gr.Dropdown(
                    choices=[
                        k
                        for k in g2p_object.keys()
                        if k.startswith(custom_speaker_ethnicity.value)
                    ],
                    value=[
                        k
                        for k in g2p_object.keys()
                        if k.startswith(custom_speaker_ethnicity.value)
                    ][0],
                    filterable=False,
                    show_label=False,
                )

                custom_speaker_ref_text_input = gr.Textbox(
                    value=refs_config[
                        get_refs_by_perfix(custom_speaker_language.value)[0]
                    ]["text"],
                    interactive=False,
                    label="步驟二：點選🎙️錄製下方句子，或上傳與句子相符的音檔",
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
                    label="合成結果", show_share_button=False, show_download_button=True
                )

    default_speaker_ethnicity.change(
        lambda ethnicity: gr.Dropdown(
            choices=get_refs_by_perfix(ethnicity),
            value=get_refs_by_perfix(ethnicity)[0],
        ),
        inputs=[default_speaker_ethnicity],
        outputs=[default_speaker_refs],
    )

    @gpu_decorator
    def default_speaker_tts(
        ref: str,
        gen_text_input: str,
    ):
        language = re.sub(r"_[男女]聲[12]?", "", ref)
        ref_text_input = refs_config[ref]["text"]
        ref_audio_input = refs_config[ref]["wav"]

        gen_text_input = gen_text_input.strip()
        if len(gen_text_input) == 0:
            raise gr.Error("請勿輸入空字串。")

        if gen_text_input[-1] not in [".", "?", "!", ",", ";", ":"]:
            gen_text_input += "."

        ignore_punctuation = False
        ipa_with_ng = False

        ref_text_input = text_to_ipa(
            ref_text_input, language, ignore_punctuation, ipa_with_ng
        )
        gen_text_input = text_to_ipa(
            gen_text_input, language, ignore_punctuation, ipa_with_ng
        )

        audio_out, _spectrogram_path = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            models_config[DEFAULT_MODEL_ID],
        )
        return audio_out

    default_speaker_generate_btn.click(
        default_speaker_tts,
        inputs=[
            default_speaker_refs,
            default_speaker_gen_text_input,
        ],
        outputs=[default_speaker_audio_output],
    )

    custom_speaker_ethnicity.change(
        lambda ethnicity: gr.Dropdown(
            choices=[k for k in g2p_object.keys() if k.startswith(ethnicity)],
            value=[k for k in g2p_object.keys() if k.startswith(ethnicity)][0],
            visible=len([k for k in g2p_object.keys()
                         if k.startswith(ethnicity)]) > 1,
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

    @gpu_decorator
    def custom_speaker_tts(
        language: str,
        ref_text_input: str,
        ref_audio_input: str,
        gen_text_input: str,
    ):
        ref_text_input = ref_text_input.strip()
        if len(ref_text_input) == 0:
            raise gr.Error("請勿輸入空字串。")

        gen_text_input = gen_text_input.strip()
        if len(gen_text_input) == 0:
            raise gr.Error("請勿輸入空字串。")

        ignore_punctuation = False
        ipa_with_ng = False

        if gen_text_input[-1] not in [".", "?", "!", ",", ";", ":"]:
            gen_text_input += "."

        ref_text_input = text_to_ipa(
            ref_text_input, language, ignore_punctuation, ipa_with_ng
        )
        gen_text_input = text_to_ipa(
            gen_text_input, language, ignore_punctuation, ipa_with_ng
        )

        audio_out, _spectrogram_path = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            models_config[DEFAULT_MODEL_ID],
        )
        return audio_out

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
