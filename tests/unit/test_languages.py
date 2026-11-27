from languages import (
    DEFAULT_GROUP,
    ETHNICITIES,
    FORMOSAN_LANGUAGES_MAP,
    LANGUAGE_CODES,
    LANGUAGE_GROUPS,
    TRANSLATION_LANGUAGE_CODES,
)


def test_language_codes_cover_all_formosan_languages():
    assert set(LANGUAGE_CODES) == set(FORMOSAN_LANGUAGES_MAP.values())
    assert len(LANGUAGE_CODES) == 42


def test_south_amis():
    assert LANGUAGE_CODES['ami_Sout'] == 'ami-x-iams'


def test_ethnicities_are_prefixes_of_formosan_language_names():
    prefixes = set()
    for name in FORMOSAN_LANGUAGES_MAP:
        prefixes.add(name.split('_')[0])
    assert prefixes == set(ETHNICITIES)


def test_language_groups_keys_are_ethnicities():
    groups = []
    for ethnicity in ETHNICITIES:
        groups.append(f'{ethnicity}語')
    assert list(LANGUAGE_GROUPS) == groups


def test_language_groups_follow_translation_language_codes_order():
    # LANGUAGE_GROUPS 與 TRANSLATION_LANGUAGE_CODES 都照「族語及語言別名稱 - 語言別名稱.csv」排序
    codes = []
    for languages in LANGUAGE_GROUPS.values():
        for _label, code in languages:
            codes.append(code)
    assert codes == list(TRANSLATION_LANGUAGE_CODES)


def test_language_groups_cover_every_code_once():
    codes = []
    for languages in LANGUAGE_GROUPS.values():
        assert languages
        for _label, code in languages:
            codes.append(code)
    assert len(codes) == len(set(codes)) == 42


def test_language_groups_labels():
    assert LANGUAGE_GROUPS['阿美語'][0] == ('南勢阿美語', 'ami-x-iams')
    assert LANGUAGE_GROUPS['魯凱語'][1] == ('霧臺魯凱語', 'dru-x-ngdr')
    assert LANGUAGE_GROUPS['鄒語'] == [('鄒語', 'tsu')]


def test_default_group():
    assert DEFAULT_GROUP == '阿美語'


def test_formosan_languages_map_names_and_order():
    # 由 LANGUAGE_GROUPS 產生；名稱要和 tts 的 g2p 語言別名稱一致（mt 呼叫 tts 時直接傳）
    assert list(FORMOSAN_LANGUAGES_MAP) == [
        '阿美_南勢', '阿美_秀姑巒', '阿美_海岸', '阿美_馬蘭', '阿美_恆春',
        '泰雅_賽考利克', '泰雅_澤敖利', '泰雅_四季', '泰雅_宜蘭澤敖利', '泰雅_汶水', '泰雅_萬大',
        '排灣_東', '排灣_北', '排灣_中', '排灣_南',
        '布農_卓群', '布農_卡群', '布農_丹群', '布農_巒群', '布農_郡群',
        '卑南_南王', '卑南_知本', '卑南_西群', '卑南_建和',
        '魯凱_東', '魯凱_霧臺', '魯凱_大武', '魯凱_多納', '魯凱_茂林', '魯凱_萬山',
        '鄒',
        '賽夏',
        '雅美',
        '邵',
        '噶瑪蘭',
        '太魯閣',
        '撒奇萊雅',
        '賽德克_都達', '賽德克_德固達雅', '賽德克_德鹿谷',
        '拉阿魯哇',
        '卡那卡那富',
    ]


def test_formosan_languages_map_codes():
    assert FORMOSAN_LANGUAGES_MAP['阿美_南勢'] == 'ami_Sout'
    assert FORMOSAN_LANGUAGES_MAP['阿美_海岸'] == 'ami_Coas'
    assert FORMOSAN_LANGUAGES_MAP['魯凱_霧臺'] == 'dru_Wuta'
    assert FORMOSAN_LANGUAGES_MAP['賽德克_德鹿谷'] == 'trv_Delu'
    assert FORMOSAN_LANGUAGES_MAP['鄒'] == 'tsu_Tsou'
    assert FORMOSAN_LANGUAGES_MAP['太魯閣'] == 'trv_Truk'


def test_formosan_language_names_match_language_groups():
    name_of = {}
    for name, nllb_code in FORMOSAN_LANGUAGES_MAP.items():
        name_of[LANGUAGE_CODES[nllb_code]] = name
    for group, languages in LANGUAGE_GROUPS.items():
        for label, code in languages:
            assert group == f"{name_of[code].split('_')[0]}語", (label, name_of[code])
