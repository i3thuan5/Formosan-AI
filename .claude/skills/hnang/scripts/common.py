"""hnang skill 共用的路徑、檔案讀寫與音檔處理。"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TTS_DIR = REPO_ROOT / "tts"
REFS_PATH = TTS_DIR / "configs" / "refs.yaml"
EXAMPLES_PATH = TTS_DIR / "configs" / "examples.yaml"
REF_WAV_DIR = TTS_DIR / "assets" / "ref_wav"
DICT_DIR = REPO_ROOT / "kithann" / "族語辭典"
SKILL_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "tests" / "tts_speakers"))
from pitch import gender_of, median_f0  # noqa: E402,F401

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def normalize_text(text):
    """收斂空白、統一喉塞音符號，和 tts/ipa/ipa.py 的做法一致。"""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[’']", "ʼ", text)


def language_of(speaker):
    """配音員名稱 → 語別，和 tts/app.py 的 default_speaker_tts 相同。"""
    return re.sub(r"_[男女]聲[12]?", "", speaker)


def expected_gender(speaker):
    if "女聲" in speaker:
        return "女"
    if "男聲" in speaker:
        return "男"
    raise SystemExit(f"配音員名稱沒有標男聲或女聲：{speaker}")


def to_wav(src, dst):
    """轉成和 tts/assets/ref_wav 相同的 24 kHz 單聲道 16-bit wav。"""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
         str(dst)],
        check=True,
    )


def to_mp3(src, dst):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-b:a", "64k", str(dst)],
        check=True,
    )


def duration_of(wav):
    import wave
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def analyze(wav):
    f0, voiced = median_f0(wav)
    return {
        "f0": None if f0 is None else round(f0, 1),
        "voiced": round(voiced, 2),
        "dur": round(duration_of(wav), 2),
        "gender": gender_of(f0),
    }
