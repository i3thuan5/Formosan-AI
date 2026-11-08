from languages import (
    ETHNICITIES,
    FORMOSAN_LANGUAGES_MAP,
    LANGUAGE_CODES,
    LANGUAGE_GROUPS,
    TRANSLATION_LANGUAGE_CODES,
)


def test_language_codes_cover_all_formosan_languages():
    assert set(LANGUAGE_CODES) == set(FORMOSAN_LANGUAGES_MAP.values())
    assert len(LANGUAGE_CODES) == 42


def test_language_codes_is_reverse_of_translation_language_codes():
    reverse = {}
    for language_code, nllb_code in TRANSLATION_LANGUAGE_CODES.items():
        reverse[nllb_code] = language_code
    assert LANGUAGE_CODES == reverse


def test_south_amis():
    assert LANGUAGE_CODES['ami_Sout'] == 'ami-x-iams'


def test_ethnicities_follow_language_groups_order():
    assert ETHNICITIES[:3] == ['阿美', '泰雅', '排灣']
    assert len(ETHNICITIES) == len(LANGUAGE_GROUPS)


def test_ethnicities_are_prefixes_of_formosan_language_names():
    prefixes = set()
    for name in FORMOSAN_LANGUAGES_MAP:
        prefixes.add(name.split('_')[0])
    assert prefixes == set(ETHNICITIES)
