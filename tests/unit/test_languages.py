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


def test_ethnicities_order_and_names():
    assert ETHNICITIES == [
        '阿美', '泰雅', '排灣', '布農', '卑南', '魯凱', '鄒', '賽夏',
        '雅美', '邵', '噶瑪蘭', '太魯閣', '撒奇萊雅', '賽德克', '拉阿魯哇', '卡那卡那富',
    ]


def test_ethnicities_are_prefixes_of_formosan_language_names():
    prefixes = set()
    for name in FORMOSAN_LANGUAGES_MAP:
        prefixes.add(name.split('_')[0])
    assert prefixes == set(ETHNICITIES)


def test_asr_language_options_follow_mt_names_and_order():
    # asr 的語言別選項：顯示名稱與順序同 mt，值是 asr 的語言別代碼
    assert LANGUAGE_GROUPS['阿美'] == [
        ('阿美_海岸', 'ami-x-pswl'),
        ('阿美_恆春', 'ami-x-pld'),
        ('阿美_馬蘭', 'ami-x-frng'),
        ('阿美_南勢', 'ami-x-iams'),
        ('阿美_秀姑巒', 'ami-x-skl'),
    ]
    assert LANGUAGE_GROUPS['魯凱'][-1] == ('魯凱_霧台', 'dru-x-ngdr')
    assert LANGUAGE_GROUPS['噶瑪蘭'] == [('噶瑪蘭', 'ckv')]


def test_asr_language_options_cover_every_asr_code_once():
    codes = []
    for options in LANGUAGE_GROUPS.values():
        assert options
        for _name, code in options:
            codes.append(code)
    assert len(codes) == len(set(codes)) == 42
    assert set(codes) == set(TRANSLATION_LANGUAGE_CODES)
