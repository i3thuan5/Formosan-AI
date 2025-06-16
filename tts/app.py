import tempfile
from importlib.resources import files

import gradio as gr
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from omegaconf import OmegaConf

from ipa.ipa import g2p_object, text_to_ipa

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
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

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


DEFAULT_MODEL_ID = list(models_config.keys())[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
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
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


def get_title():
    with open("DEMO.md", encoding="utf-8") as tong:
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
    with open("DEMO.md") as tong:
        gr.Markdown(tong.read())

    with gr.Row():
        with gr.Column():
            model_drop_down = gr.Dropdown(
                models_config.keys(),
                value=DEFAULT_MODEL_ID,
                label="模型",
            )

            language = gr.Dropdown(
                choices=g2p_object.keys(),
                label="語言",
                value="阿美_秀姑巒",
            )

            ref_audio_input = gr.Audio(
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    sample_rate=24000,
                ),
                label="Reference Audio",
            )
            ref_text_input = gr.Textbox(
                value="",
                label="Reference Text",
            )

            gen_text_input = gr.Textbox(
                label="Text to Generate",
                value="",
            )

            generate_btn = gr.Button("Synthesize", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    info=(
                        "The model tends to produce silences, especially on longer audio. "
                        "We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. "
                        "This will also increase generation time."
                    ),
                    value=False,
                )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.3,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="語速（越小越慢）",
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=32,
                    step=2,
                    info="Set the number of denoising steps.",
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    info="Set the duration of the cross-fade between audio clips.",
                )
        with gr.Column():
            audio_output = gr.Audio(label="Synthesized Audio")
            spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        model_drop_down: str,
        language: str,
        ref_audio_input: str,
        ref_text_input: str,
        gen_text_input: str,
        remove_silence: bool,
        cross_fade_duration_slider: float,
        nfe_slider: int,
        speed_slider: float,
    ):
        ref_text_input = ref_text_input.strip()
        if len(ref_text_input) == 0:
            raise gr.Error("請勿輸入空字串。")

        gen_text_input = gen_text_input.strip()
        if len(gen_text_input) == 0:
            raise gr.Error("請勿輸入空字串。")

        ignore_punctuation = False
        ipa_with_ng = False

        ref_text_input = text_to_ipa(
            ref_text_input, language, ignore_punctuation, ipa_with_ng
        )
        gen_text_input = text_to_ipa(
            gen_text_input, language, ignore_punctuation, ipa_with_ng
        )

        audio_out, spectrogram_path = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            models_config[model_drop_down],
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path

    generate_btn.click(
        basic_tts,
        inputs=[
            model_drop_down,
            language,
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output],
    )
    gr.Examples(
        [
            [
                "阿美_秀姑巒",
                "./ref_wav/E-PV001-0001.wav",
                "o pakafanaʼ ni akong to pinangan no romiʼad.",
                "Mafanaʼ kiso a misanoPangcah haw?",
            ],
            [
                "阿美_秀姑巒",
                "./ref_wav/E-PV001-0001.wav",
                "o pakafanaʼ ni akong to pinangan no romiʼad.",
                "Kering sa masoni⌃ to ko pipahanhanan a tatokian, o fe:soc no niyam a tayra i piondoan.",
            ],
            [
                "阿美_秀姑巒",
                "./ref_wav/cu_practice-0016849.wav",
                "ano cikasoan to, ano o falangaw to i, malecaday to a matira.",
                "Pafelien cingra to misapoeneray a falocoʼ, nanay madaʼoc matilid i falocoʼ nira konini.",
            ],
        ],
        label="範例",
        inputs=[
            language,
            ref_audio_input,
            ref_text_input,
            gen_text_input,
        ],
    )

demo.launch()
