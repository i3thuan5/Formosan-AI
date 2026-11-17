"""Formosan UI labels, language codes and Whisper-to-NLLB language codes."""

LANGUAGE_GROUPS = {
    "阿美語": [
        ("南勢阿美語", "ami-x-iams"),
        ("秀姑巒阿美語", "ami-x-skl"),
        ("海岸阿美語", "ami-x-pswl"),
        ("馬蘭阿美語", "ami-x-frng"),
        ("恆春阿美語", "ami-x-pld"),
    ],
    "泰雅語": [
        ("賽考利克泰雅語", "tay-x-sql"),
        ("澤敖利泰雅語", "tay-x-sul"),
        ("四季泰雅語", "tay-x-cql"),
        ("宜蘭澤敖利泰雅語", "tay-x-kls"),
        ("汶水泰雅語", "tay-x-mtuw"),
        ("萬大泰雅語", "tay-x-plngw"),
    ],
    "排灣語": [
        ("東排灣語", "pwn-x-kcdsn"),
        ("北排灣語", "pwn-x-vnrn"),
        ("中排灣語", "pwn-x-pnvn"),
        ("南排灣語", "pwn-x-ynvl"),
    ],
    "布農語": [
        ("卓群布農語", "bnn-x-td"),
        ("卡群布農語", "bnn-x-bkh"),
        ("丹群布農語", "bnn-x-vtn"),
        ("巒群布農語", "bnn-x-bnz"),
        ("郡群布農語", "bnn-x-isbk"),
    ],
    "卑南語": [
        ("南王卑南語", "pyu-x-pym"),
        ("知本卑南語", "pyu-x-ktrp"),
        ("西群卑南語", "pyu-x-mkzy"),
        ("建和卑南語", "pyu-x-ksvk"),
    ],
    "魯凱語": [
        ("東魯凱語", "dru-x-trmk"),
        ("霧臺魯凱語", "dru-x-ngdr"),
        ("大武魯凱語", "dru-x-lbw"),
        ("多納魯凱語", "dru-x-kgdv"),
        ("茂林魯凱語", "dru-x-tldr"),
        ("萬山魯凱語", "dru-x-opnh"),
    ],
    "鄒語": [("鄒語", "tsu")],
    "賽夏語": [("賽夏語", "xsy")],
    "雅美語": [("雅美語", "tao")],
    "邵語": [("邵語", "ssf")],
    "噶瑪蘭語": [("噶瑪蘭語", "ckv")],
    "太魯閣語": [("太魯閣語", "trv-x-truku")],
    "撒奇萊雅語": [("撒奇萊雅語", "szy")],
    "賽德克語": [
        ("都達賽德克語", "trv-x-td"),
        ("德固達雅賽德克語", "trv-x-tgdy"),
        ("德鹿谷賽德克語", "trv-x-trk"),
    ],
    "拉阿魯哇語": [("拉阿魯哇語", "sxr")],
    "卡那卡那富語": [("卡那卡那富語", "xnb")],
}


ETHNICITIES = [
    "阿美", "泰雅", "排灣", "布農", "卑南", "魯凱", "鄒", "賽夏",
    "雅美", "邵", "噶瑪蘭", "太魯閣", "撒奇萊雅", "賽德克", "拉阿魯哇", "卡那卡那富",
]


DEFAULT_GROUP = next(iter(LANGUAGE_GROUPS))


# asr（Whisper）的語言別代碼 → 翻譯模型（NLLB fine-tune）的語言代碼，兩者不同。
TRANSLATION_LANGUAGE_CODES = {
    "ami-x-iams": "ami_Sout",
    "ami-x-skl": "ami_Xiug",
    "ami-x-pswl": "ami_Coas",
    "ami-x-frng": "ami_Mala",
    "ami-x-pld": "ami_Heng",
    "tay-x-sql": "tay_Seko",
    "tay-x-sul": "tay_Zeao",
    "tay-x-cql": "tay_Four",
    "tay-x-kls": "tay_Yzea",
    "tay-x-mtuw": "tay_Wens",
    "tay-x-plngw": "tay_Wand",
    "pwn-x-kcdsn": "pwn_East",
    "pwn-x-vnrn": "pwn_Nrth",
    "pwn-x-pnvn": "pwn_Cent",
    "pwn-x-ynvl": "pwn_Sout",
    "bnn-x-td": "bnn_Zhuo",
    "bnn-x-bkh": "bnn_Kaqu",
    "bnn-x-vtn": "bnn_Tanq",
    "bnn-x-bnz": "bnn_Luan",
    "bnn-x-isbk": "bnn_Junq",
    "pyu-x-pym": "pyu_Nanw",
    "pyu-x-ktrp": "pyu_Zhib",
    "pyu-x-mkzy": "pyu_Xiqu",
    "pyu-x-ksvk": "pyu_Jian",
    "dru-x-trmk": "dru_East",
    "dru-x-ngdr": "dru_Wuta",
    "dru-x-lbw": "dru_Dawu",
    "dru-x-kgdv": "dru_Dona",
    "dru-x-tldr": "dru_Maol",
    "dru-x-opnh": "dru_Wans",
    "tsu": "tsu_Tsou",
    "xsy": "xsy_Sais",
    "tao": "tao_Yami",
    "ssf": "ssf_Thao",
    "ckv": "ckv_Kava",
    "trv-x-truku": "trv_Truk",
    "szy": "szy_Saki",
    "trv-x-td": "trv_Duda",
    "trv-x-tgdy": "trv_Tegu",
    "trv-x-trk": "trv_Delu",
    "sxr": "sxr_Saar",
    "xnb": "xnb_Kana",
}


# 翻譯模型的語言代碼 → 語言別代碼。
LANGUAGE_CODES = {}
for _language_code, _nllb_code in TRANSLATION_LANGUAGE_CODES.items():
    LANGUAGE_CODES[_nllb_code] = _language_code


# mt 與 tts 的語言別名稱（「族語別_語言別」，如「阿美_南勢」）→ 翻譯模型的語言代碼。
# 由 LANGUAGE_GROUPS 產生，順序相同：「南勢阿美語」去掉族語別名稱「阿美語」，前面加上 ETHNICITIES 的「阿美_」。
# 只有一個語言別的族語別，名稱就是 ETHNICITIES 的族語別（如「鄒」）。
# 名稱要和 tts 的 g2p 語言別名稱一致，mt 呼叫 tts 時直接傳這個名稱。
FORMOSAN_LANGUAGES_MAP = {}
for _ethnicity, (_group, _languages) in zip(ETHNICITIES, LANGUAGE_GROUPS.items(), strict=True):
    if _group != f"{_ethnicity}語":
        raise ValueError(f"ETHNICITIES 和 LANGUAGE_GROUPS 的順序或名稱不一致：{_ethnicity}、{_group}")
    for _label, _language_code in _languages:
        if _label == _group:
            _name = _ethnicity
        elif _label.endswith(_group):
            _name = f"{_ethnicity}_{_label.removesuffix(_group)}"
        else:
            raise ValueError(f"語言別名稱要以族語別名稱結尾：{_label}、{_group}")
        FORMOSAN_LANGUAGES_MAP[_name] = TRANSLATION_LANGUAGE_CODES[_language_code]
