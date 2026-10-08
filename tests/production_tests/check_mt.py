"""檢查 mt（NLLB）雙向翻譯。

只驗有沒有翻出東西，不比對譯文：模型會換版。
"""
import sys
from pathlib import Path

from reporting import check
from timing import check_threshold, warm_up

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mt"))
from formosan_languages import FORMOSAN_LANGUAGES_MAP  # noqa: E402

SERVICE = "mt"
APP_PATH = "kari-seejiq-tnpusu-ai-hmjil"
AMIS_LANGUAGE = "阿美_海岸"
CHINESE_CODE = "zho_Hant"
# 取自 tests/data 那段影片的辨識結果
FORMOSAN_TEXT = "sosowalen ako itiya:ayho a ʼorip niyam"
CHINESE_TEXT = "我們以前的生活"
TO_ZH_LANGUAGES_API_NAME = "/to_zh_languages"


def all_languages():
    return list(FORMOSAN_LANGUAGES_MAP)


def to_chinese(client, language, code):
    # 「族語 ⮕ 華語」的語別也是 Radio，同樣要先切換族別
    client.predict(language.split("_")[0], api_name=TO_ZH_LANGUAGES_API_NAME)
    return client.predict(FORMOSAN_TEXT, code, CHINESE_CODE, api_name="/translate")


def to_formosan(client, language, code):
    # 「華語 ⮕ 族語」的語別是 Radio，要先在同一個 session 切換族別，choices 才會包含該語別
    client.predict(language.split("_")[0], api_name="/to_formosan_languages")
    return client.predict(CHINESE_TEXT, CHINESE_CODE, code, api_name="/translate_1")


def assert_translation(result, source, language, direction, warnings):
    if not isinstance(result, str):
        raise AssertionError(f"回傳的不是字串，而是 {type(result).__name__}")

    if language == AMIS_LANGUAGE:
        if not result.strip():
            raise AssertionError(f"{direction} 沒有翻出內容")
        if result.strip() == source.strip():
            raise AssertionError(f"{direction} 的結果和原文一樣：{result!r}")
        return f"{len(result)} 字元"

    if not result.strip():
        message = f"{SERVICE} {language} {direction} 沒有翻出內容"
        warnings.append(message)
        return "沒有內容（警告）"
    return f"{len(result)} 字元"


def check_to_chinese(client, language, code, warnings):
    def run():
        result = to_chinese(client, language, code)
        return assert_translation(result, FORMOSAN_TEXT, language, "族語 ⮕ 華語", warnings)
    return run


def check_to_formosan(client, language, code, warnings):
    def run():
        result = to_formosan(client, language, code)
        return assert_translation(result, CHINESE_TEXT, language, "華語 ⮕ 族語", warnings)
    return run


def run(client, languages, failures, warnings):
    print(f"\n[mt] 雙向翻譯 {len(languages)} 個語別：{'、'.join(languages)}")
    warm_up(
        SERVICE,
        lambda: to_chinese(client, AMIS_LANGUAGE, FORMOSAN_LANGUAGES_MAP[AMIS_LANGUAGE]),
        warnings,
    )

    for language in languages:
        code = FORMOSAN_LANGUAGES_MAP[language]
        seconds = check(
            f"{language} 族語 ⮕ 華語",
            check_to_chinese(client, language, code, warnings),
            failures,
            service=SERVICE,
        )
        if seconds is not None and language == AMIS_LANGUAGE:
            check_threshold(SERVICE, seconds, warnings)

        check(
            f"{language} 華語 ⮕ 族語",
            check_to_formosan(client, language, code, warnings),
            failures,
            service=SERVICE,
        )
