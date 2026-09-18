"""檢查 asr（WhisperX）的辨識輸出格式。

只驗格式與有沒有內容，不比對辨識出來的文字：模型會換版，比對內容會讓每次升級都要改測試。
"""
import re
import sys
from pathlib import Path

from gradio_client import handle_file

from reporting import check
from timing import check_threshold, warm_up

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "asr"))
from languages import LANGUAGE_GROUPS  # noqa: E402

SERVICE = "asr"
APP_PATH = "sapolita"
API_NAME = "/generate_srt"
# asr/app.py 的 group_input.change 沒有指定 api_name，Gradio 以函式名 update_languages 命名
GROUP_API_NAME = "/update_languages"
VIDEO = REPO_ROOT / "tests" / "data" / "海岸阿美語-曾玉蘭-個人生命史-短.mp4"
# 素材本身的語別，一定要測，而且要求有辨識結果
AMIS_LANGUAGE = "ami-x-pswl"
TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")


def all_languages():
    return [code for languages in LANGUAGE_GROUPS.values() for _, code in languages]


def group_of(code):
    for group, languages in LANGUAGE_GROUPS.items():
        if any(c == code for _, c in languages):
            return group
    raise AssertionError(f"asr/languages.py 沒有語別代碼 {code}")


def parse_srt(srt):
    """把 SRT 拆成 cue，順便驗格式。回傳每個 cue 的族語文字。"""
    if not isinstance(srt, str):
        raise AssertionError(f"回傳的不是字串，而是 {type(srt).__name__}")

    formosan_lines = []
    for block in [b for b in srt.strip().split("\n\n") if b.strip()]:
        lines = block.split("\n")
        if len(lines) != 4:
            raise AssertionError(f"cue 應有 4 行（序號、時間、族語、華語），實際 {len(lines)} 行：{block!r}")
        index, timestamp, formosan, chinese = lines
        if not index.strip().isdigit():
            raise AssertionError(f"cue 第一行應為序號，實際為 {index!r}")
        if not TIMESTAMP.match(timestamp.strip()):
            raise AssertionError(f"時間戳格式不對：{timestamp!r}")
        if not formosan.startswith("族語："):
            raise AssertionError(f"第三行應以「族語：」開頭：{formosan!r}")
        if not chinese.startswith("華語："):
            raise AssertionError(f"第四行應以「華語：」開頭：{chinese!r}")
        formosan_lines.append(formosan[len("族語："):].strip())
    return formosan_lines


def transcribe(client, language):
    # 語別是 Radio，要先在同一個 session 切換族別，choices 才會包含該語別
    client.predict(group_of(language), api_name=GROUP_API_NAME)
    return client.predict({"video": handle_file(str(VIDEO))}, language, api_name=API_NAME)


def check_language(client, language, warnings):
    def run():
        srt = transcribe(client, language)
        cues = parse_srt(srt)

        if language == AMIS_LANGUAGE:
            if not cues:
                raise AssertionError("海岸阿美語沒有辨識出任何 cue")
            if not any(cues):
                raise AssertionError("海岸阿美語的族語行全部是空的")
            return f"{len(cues)} 個 cue"

        # 其他語別是拿阿美語音檔去問別的語言模型，回空是合理的，只警告
        if not cues or not any(cues):
            message = f"{SERVICE} {language} 沒有辨識出內容（拿阿美語素材測其他語別，屬正常）"
            warnings.append(message)
            return "沒有內容（警告）"
        return f"{len(cues)} 個 cue"
    return run


def run(client, languages, failures, warnings):
    print(f"\n[asr] 辨識 {len(languages)} 個語別：{'、'.join(languages)}")
    warm_up(SERVICE, lambda: transcribe(client, AMIS_LANGUAGE), warnings)

    for language in languages:
        seconds = check(f"{language}", check_language(client, language, warnings), failures, service=SERVICE)
        if seconds is not None and language == AMIS_LANGUAGE:
            check_threshold(SERVICE, seconds, warnings)
