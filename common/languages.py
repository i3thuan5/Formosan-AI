"""Formosan UI labels, language codes and Whisper-to-NLLB language codes."""

ETHNICITIES = [
    "阿美", "泰雅", "排灣", "布農", "卑南", "魯凱", "鄒", "賽夏",
    "雅美", "邵", "噶瑪蘭", "太魯閣", "撒奇萊雅", "賽德克", "拉阿魯哇", "卡那卡那富",
]


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


DEFAULT_GROUP = next(iter(LANGUAGE_GROUPS))


OLD_TRANSLATION_LANGUAGE_CODES = {
    "ami-x-iams": ("ami_Sout", "阿美_南勢"),
    "ami-x-skl": ("ami_Xiug", "阿美_秀姑巒"),
    "ami-x-pswl": ("ami_Coas", "阿美_海岸"),
    "ami-x-frng": ("ami_Mala", "阿美_馬蘭"),
    "ami-x-pld": ("ami_Heng", "阿美_恆春"),
    "tay-x-sql": ("tay_Seko", "泰雅_賽考利克"),
    "tay-x-sul": ("tay_Zeao", "泰雅_澤敖利"),
    "tay-x-cql": ("tay_Four", "泰雅_四季"),
    "tay-x-kls": ("tay_Yzea", "泰雅_宜蘭澤敖利"),
    "tay-x-mtuw": ("tay_Wens", "泰雅_汶水"),
    "tay-x-plngw": ("tay_Wand", "泰雅_萬大"),
    "pwn-x-kcdsn": ("pwn_East", "排灣_東"),
    "pwn-x-vnrn": ("pwn_Nrth", "排灣_北"),
    "pwn-x-pnvn": ("pwn_Cent", "排灣_中"),
    "pwn-x-ynvl": ("pwn_Sout", "排灣_南"),
    "bnn-x-td": ("bnn_Zhuo", "布農_卓群"),
    "bnn-x-bkh": ("bnn_Kaqu", "布農_卡群"),
    "bnn-x-vtn": ("bnn_Tanq", "布農_丹群"),
    "bnn-x-bnz": ("bnn_Luan", "布農_巒群"),
    "bnn-x-isbk": ("bnn_Junq", "布農_郡群"),
    "pyu-x-pym": ("pyu_Nanw", "卑南_南王"),
    "pyu-x-ktrp": ("pyu_Zhib", "卑南_知本"),
    "pyu-x-mkzy": ("pyu_Xiqu", "卑南_西群"),
    "pyu-x-ksvk": ("pyu_Jian", "卑南_建和"),
    "dru-x-trmk": ("dru_East", "魯凱_東"),
    "dru-x-ngdr": ("dru_Wuta", "魯凱_霧臺"),
    "dru-x-lbw": ("dru_Dawu", "魯凱_大武"),
    "dru-x-kgdv": ("dru_Dona", "魯凱_多納"),
    "dru-x-tldr": ("dru_Maol", "魯凱_茂林"),
    "dru-x-opnh": ("dru_Wans", "魯凱_萬山"),
    "tsu": ("tsu_Tsou", "鄒"),
    "xsy": ("xsy_Sais", "賽夏"),
    "tao": ("tao_Yami", "雅美"),
    "ssf": ("ssf_Thao", "邵"),
    "ckv": ("ckv_Kava", "噶瑪蘭"),
    "trv-x-truku": ("trv_Truk", "太魯閣"),
    "szy": ("szy_Saki", "撒奇萊雅"),
    "trv-x-td": ("trv_Duda", "賽德克_都達"),
    "trv-x-tgdy": ("trv_Tegu", "賽德克_德固達雅"),
    "trv-x-trk": ("trv_Delu", "賽德克_德鹿谷"),
    "sxr": ("sxr_Saar", "拉阿魯哇"),
    "xnb": ("xnb_Kana", "卡那卡那富"),
}

# TRANSLATION_LANGUAGE_CODES：語言別代碼 → 翻譯模型的語言代碼。
# FORMOSAN_LANGUAGES_MAP：mt 與 tts 的語言別名稱 → 翻譯模型的語言代碼，順序是 mt 介面的顯示順序。
# LANGUAGE_CODES：翻譯模型的語言代碼 → 語言別代碼。
TRANSLATION_LANGUAGE_CODES = {}
FORMOSAN_LANGUAGES_MAP = {}
LANGUAGE_CODES = {}
for _new_code, (_old_code, _old_name) in OLD_TRANSLATION_LANGUAGE_CODES.items():
    TRANSLATION_LANGUAGE_CODES[_new_code] = _old_code
    FORMOSAN_LANGUAGES_MAP[_old_name] = _old_code
    LANGUAGE_CODES[_old_code] = _new_code
