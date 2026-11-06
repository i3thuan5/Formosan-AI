from formosan_languages import FORMOSAN_LANGUAGES_MAP, LANGUAGE_CODES
from languages import TRANSLATION_LANGUAGE_CODES


def test_keys_cover_all_formosan_languages():
    assert set(LANGUAGE_CODES) == set(FORMOSAN_LANGUAGES_MAP.values())
    assert len(LANGUAGE_CODES) == 42


def test_is_reverse_of_asr_translation_language_codes():
    reverse = {}
    for language_code, nllb_code in TRANSLATION_LANGUAGE_CODES.items():
        reverse[nllb_code] = language_code
    assert LANGUAGE_CODES == reverse


def test_south_amis():
    assert LANGUAGE_CODES['ami_Sout'] == 'ami-x-iams'
