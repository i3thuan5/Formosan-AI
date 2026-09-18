"""檢查 asr-kaldi（Vosk）的辨識輸出。

族別表讀自 asr-kaldi/configs/models.yaml 第一個模型的 dialect_mapping，
和 asr-kaldi/app.py 取 DEFAULT_MODEL 的方式一致。
"""
from pathlib import Path

import yaml
from gradio_client import handle_file

from reporting import check
from timing import check_threshold, warm_up

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE = "asr-kaldi"
APP_PATH = "sapolita-kaldi"
API_NAME = "/automatic_speech_recognition"
AUDIO = REPO_ROOT / "tests" / "data" / "海岸阿美語-曾玉蘭-個人生命史-短.mp3"
AMIS_DIALECT = "formosan_ami"


def all_dialects():
    """app.py 取 models_config 的第一個模型，這裡照做。"""
    with open(REPO_ROOT / "asr-kaldi" / "configs" / "models.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default_model = config[list(config)[0]]
    return list(default_model["dialect_mapping"].values())


def transcribe(client, dialect):
    return client.predict(dialect, handle_file(str(AUDIO)), api_name=API_NAME)


def check_dialect(client, dialect, warnings):
    def run():
        text = transcribe(client, dialect)
        if not isinstance(text, str):
            raise AssertionError(f"回傳的不是字串，而是 {type(text).__name__}")

        if dialect == AMIS_DIALECT:
            if not text.strip():
                raise AssertionError("阿美語沒有辨識出內容")
            return f"{len(text)} 字元"

        # 拿阿美語音檔去問別族的模型，回空是合理的
        if not text.strip():
            message = f"{SERVICE} {dialect} 沒有辨識出內容（拿阿美語素材測其他族別，屬正常）"
            warnings.append(message)
            return "沒有內容（警告）"
        return f"{len(text)} 字元"
    return run


def run(client, dialects, failures, warnings):
    print(f"\n[asr-kaldi] 辨識 {len(dialects)} 個族別：{'、'.join(dialects)}")
    warm_up(SERVICE, lambda: transcribe(client, AMIS_DIALECT), warnings)

    for dialect in dialects:
        seconds = check(f"{dialect}", check_dialect(client, dialect, warnings), failures, service=SERVICE)
        if seconds is not None and dialect == AMIS_DIALECT:
            check_threshold(SERVICE, seconds, warnings)
