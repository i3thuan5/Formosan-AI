"""在本機 CPU 用和線上相同的模型與參數合成 WORK/jobs.json（tts 服務掛掉時用）。

    python local_synthesize.py --work WORK

要先跑過 synthesize.py 產生 jobs.json（服務連不上時 synthesize.py 會在合成前就失敗，
jobs.json 已經寫好）。環境建法見 SKILL.md。
"""
import argparse
import os
import sys
import time
from pathlib import Path

from common import REPO_ROOT, TTS_DIR, load_json

sys.path.insert(0, str(TTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "common"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from formosan_tts import FormosanTTS  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from languages import FORMOSAN_LANGUAGES_MAP, LANGUAGE_CODES  # noqa: E402


def load_model():
    """從 tts/configs/models.yaml 第一個模型載入，和 app.py 的 DEFAULT_MODEL_ID 一致。"""
    models = yaml.safe_load((TTS_DIR / "configs" / "models.yaml").read_text(encoding="utf-8"))
    config = next(iter(models.values()))
    model_dir = snapshot_download(config["repo_id"], revision=config["revision"])
    return FormosanTTS(model_dir, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    work = Path(args.work).resolve()
    jobs = load_json(work / "jobs.json")
    texts = load_json(work / "texts.json")

    torch.set_num_threads(os.cpu_count())
    model = load_model()

    for job in jobs:
        # job["language"] 是語別名稱（如「阿美_海岸」），模型用的是語別代碼（如 ami-x-pswl）
        lang_code = LANGUAGE_CODES[FORMOSAN_LANGUAGES_MAP[job["language"]]]
        for i, (text, _zh) in enumerate(texts, start=1):
            out = work / "out" / job["group"] / f"{i:02d}.wav"
            if out.exists():
                continue
            start = time.time()
            wave, rate = model.generate(text.strip(), lang_code, job["ref_wav"])
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wave, rate)
            print(f"{out} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
