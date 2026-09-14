"""上線後檢查：MT「華語 ⮕ 族語」合成語音，以及 TTS `/synthesize` API。

    $ BASE_URL=https://ai-labs.ilrdf.org.tw python production_tests/check_mt_tts_playback.py

全部通過 exit 0，任一項失敗 exit 1。說明見 README.md。
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml
from gradio_client import Client
from gradio_client.exceptions import AppError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "mt"))
from formosan_languages import FORMOSAN_LANGUAGES_MAP  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "https://ai-labs.ilrdf.org.tw").rstrip("/")
MT_URL = f"{BASE_URL}/kari-seejiq-tnpusu-ai-hmjil/"
TTS_URL = f"{BASE_URL}/hnang-kari-ai-asi-sluhay/"

# 預設的阿美，加上涵蓋 ṟ、ɨ é、ʉ、大寫 S 與 : 的語別
END_TO_END_LANGUAGES = ["阿美_海岸", "泰雅_萬大", "魯凱_茂林", "卡那卡那富", "賽夏"]
SLOW_WARNING_SECONDS = 15

REGRESSION_TEXTS = [
    ("阿美_海岸", 'Sowal sa ko singsi, "Ano dafak micodad kita."'),
    ("卑南_知本", 'marengay na sinsi, " temakesi ta nu ʼemanan.」'),
]
UNSUPPORTED_LANGUAGE = "不存在的語別"

failures = []


def load_refs():
    with open(REPO_ROOT / "tts" / "configs" / "refs.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_text(refs, language):
    speakers = [k for k in refs if k.startswith(language + "_")]
    if len(speakers) == 0:
        raise AssertionError(f"tts/configs/refs.yaml 沒有「{language}」的配音員")
    return refs[speakers[0]]["text"]


def assert_audio(result):
    path = Path(result)
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"沒有拿到音檔：{result!r}")
    path.unlink()


def check(name, fn):
    start = time.time()
    try:
        note = fn()
    except Exception as e:
        failures.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ✗ {name}（{time.time() - start:.1f}s）{type(e).__name__}: {e}", flush=True)
        return
    print(f"  ✓ {name}（{time.time() - start:.1f}s）{note or ''}", flush=True)


def connect(url, download_dir):
    return Client(url, verbose=False, analytics_enabled=False, download_files=download_dir)


def check_tts_api_contract(tts):
    endpoints = tts.view_api(print_info=False, return_format="dict")["named_endpoints"]
    if "/synthesize" not in endpoints:
        raise AssertionError("TTS 缺少 /synthesize")
    params = [p["parameter_name"] for p in endpoints["/synthesize"]["parameters"]]
    if params != ["language", "text"]:
        raise AssertionError(f"/synthesize 參數應為 ['language', 'text']，實際為 {params}")


def check_tts_language(tts, refs, language):
    def run():
        text = sample_text(refs, language)
        assert_audio(tts.predict(language, text, api_name="/synthesize"))
        return text
    return run


def check_tts_text(tts, language, text):
    def run():
        assert_audio(tts.predict(language, text, api_name="/synthesize"))
    return run


def check_tts_unsupported_language(tts):
    try:
        result = tts.predict(UNSUPPORTED_LANGUAGE, "abc", api_name="/synthesize")
    except AppError as e:
        return f"回傳錯誤：{e}"
    raise AssertionError(f"不支援的語別應回傳錯誤，卻回傳了 {result!r}")


def check_end_to_end(mt, refs, language, code):
    def run():
        text = sample_text(refs, language)
        # mt 的語別是 Radio，要先在同一個 session 切換族別，choices 才會包含該語別
        mt.predict(language.split("_")[0], api_name="/to_formosan_languages")
        start = time.time()
        assert_audio(mt.predict(text, code, api_name="/synthesize"))
        elapsed = time.time() - start
        if elapsed > SLOW_WARNING_SECONDS:
            return f"⚠ 合成花了 {elapsed:.1f}s，接近 mt 的 20 秒逾時"
    return run


def main():
    print(f"BASE_URL = {BASE_URL}")
    refs = load_refs()

    with tempfile.TemporaryDirectory(prefix="production-tests-") as download_dir:
        try:
            tts = connect(TTS_URL, download_dir)
        except Exception as e:
            failures.append(("連線 TTS", f"{type(e).__name__}: {e}"))
            print(f"✗ 連不上 TTS：{TTS_URL}（{e}）")
            tts = None

        if tts is not None:
            print("\n[1] TTS API 約定")
            check("/synthesize(language, text)", lambda: check_tts_api_contract(tts))

            print(f"\n[2] TTS 合成 {len(FORMOSAN_LANGUAGES_MAP)} 個語別")
            for language in FORMOSAN_LANGUAGES_MAP:
                check(language, check_tts_language(tts, refs, language))

            print("\n[3] 已知 bug 回歸")
            for language, text in REGRESSION_TEXTS:
                check(f"{language}：{text}", check_tts_text(tts, language, text))
            check(f"不支援的語別「{UNSUPPORTED_LANGUAGE}」", lambda: check_tts_unsupported_language(tts))

        print("\n[4] mt → tts 端到端")
        try:
            mt = connect(MT_URL, download_dir)
        except Exception as e:
            failures.append(("連線 MT", f"{type(e).__name__}: {e}"))
            print(f"✗ 連不上 MT：{MT_URL}（{e}）")
            mt = None
        if mt is not None:
            for language in END_TO_END_LANGUAGES:
                check(language, check_end_to_end(mt, refs, language, FORMOSAN_LANGUAGES_MAP[language]))

    if failures:
        print(f"\n失敗 {len(failures)} 項：")
        for name, message in failures:
            print(f"  - {name}：{message}")
        return 1

    print("\n全部通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
