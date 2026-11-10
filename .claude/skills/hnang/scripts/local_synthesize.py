"""在本機 CPU 用和線上相同的模型與參數合成 WORK/jobs.json（tts 服務掛掉時用）。

    python local_synthesize.py --work WORK

要先跑過 synthesize.py 產生 jobs.json（服務連不上時 synthesize.py 會在合成前就失敗，
jobs.json 已經寫好）。一句約 45–90 秒（16 核 CPU），環境建法見 SKILL.md。
"""
import argparse
import os
import sys
import time
from pathlib import Path

from common import TTS_DIR, load_json

os.chdir(TTS_DIR)
sys.path.insert(0, str(TTS_DIR))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from cached_path import cached_path  # noqa: E402
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text  # noqa: E402
from f5_tts.model import DiT  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from ipa.ipa import text_to_ipa  # noqa: E402

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                 text_mask_padding=True, pe_attn_head=None)


def model_paths():
    """從 tts/configs/models.yaml 第一個模型取 checkpoint 與 vocab，和 app.py 的 DEFAULT_MODEL_ID 一致。"""
    raw = OmegaConf.to_container(OmegaConf.load(TTS_DIR / "configs" / "models.yaml"), resolve=False)
    spec = next(iter(raw.values()))
    args = spec[len("${load_f5tts:"):-1].split(",")
    return args[0], args[1]


def quiet(*args, **kwargs):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    work = Path(args.work).resolve()
    jobs = load_json(work / "jobs.json")
    texts = load_json(work / "texts.json")

    torch.set_num_threads(os.cpu_count())
    ckpt_url, vocab_url = model_paths()
    model = load_model(DiT, MODEL_CFG, str(cached_path(ckpt_url)), vocab_file=str(cached_path(vocab_url)),
                       use_ema=False, device="cpu")
    vocoder = load_vocoder(device="cpu")

    for job in jobs:
        language = job["language"]
        ref_ipa = text_to_ipa(job["ref_text"], language)
        ref_audio, ref_ipa = preprocess_ref_audio_text(job["ref_wav"], ref_ipa, show_info=quiet)
        for i, (text, _zh) in enumerate(texts, start=1):
            out = work / "out" / job["group"] / f"{i:02d}.wav"
            if out.exists():
                continue
            gen = text.strip()
            if gen[-1] not in ".?!,;:":
                gen += "."
            start = time.time()
            wave, rate, _spec = infer_process(
                ref_audio, ref_ipa, text_to_ipa(gen, language), model, vocoder,
                cross_fade_duration=0.15, nfe_step=32, speed=1, show_info=quiet, device="cpu",
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wave, rate)
            print(f"{out} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
