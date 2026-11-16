"""Formosan UI labels, language codes and Whisper-to-NLLB language codes."""

# 族語別名稱：所有模型的 Radio 共用這份，順序就是介面的顯示順序，和 FORMOSAN_LANGUAGES_MAP 名稱的前綴相同。
ETHNICITIES = [
    "阿美", "泰雅", "排灣", "布農", "卑南", "魯凱", "鄒", "賽夏",
    "雅美", "邵", "噶瑪蘭", "太魯閣", "撒奇萊雅", "賽德克", "拉阿魯哇", "卡那卡那富",
]

DEFAULT_GROUP = ETHNICITIES[0]


# asr（Whisper）的語言別代碼 → 翻譯模型（NLLB fine-tune）的語言代碼，兩者不同。
# key 是「族語及語言別名稱 - 語言別名稱.csv」的語言別代碼。
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


# mt 與 tts 的語言別顯示名稱（「族語別_語言別」）→ 翻譯模型的語言代碼。
# 名稱要和 tts 的 g2p 語言別名稱一致（mt 呼叫 tts 時直接傳這個名稱），順序是 mt 介面的顯示順序。
FORMOSAN_LANGUAGES_MAP = {
    "阿美_海岸": "ami_Coas",
    "阿美_恆春": "ami_Heng",
    "阿美_馬蘭": "ami_Mala",
    "阿美_南勢": "ami_Sout",
    "阿美_秀姑巒": "ami_Xiug",
    "泰雅_四季": "tay_Four",
    "泰雅_賽考利克": "tay_Seko",
    "泰雅_萬大": "tay_Wand",
    "泰雅_汶水": "tay_Wens",
    "泰雅_宜蘭澤敖利": "tay_Yzea",
    "泰雅_澤敖利": "tay_Zeao",
    "布農_郡群": "bnn_Junq",
    "布農_卡群": "bnn_Kaqu",
    "布農_巒群": "bnn_Luan",
    "布農_丹群": "bnn_Tanq",
    "布農_卓群": "bnn_Zhuo",
    "卡那卡那富": "xnb_Kana",
    "噶瑪蘭": "ckv_Kava",
    "排灣_中": "pwn_Cent",
    "排灣_東": "pwn_East",
    "排灣_北": "pwn_Nrth",
    "排灣_南": "pwn_Sout",
    "卑南_建和": "pyu_Jian",
    "卑南_南王": "pyu_Nanw",
    "卑南_西群": "pyu_Xiqu",
    "卑南_知本": "pyu_Zhib",
    "魯凱_大武": "dru_Dawu",
    "魯凱_多納": "dru_Dona",
    "魯凱_東": "dru_East",
    "魯凱_茂林": "dru_Maol",
    "魯凱_萬山": "dru_Wans",
    "魯凱_霧台": "dru_Wuta",
    "拉阿魯哇": "sxr_Saar",
    "賽夏": "xsy_Sais",
    "撒奇萊雅": "szy_Saki",
    "賽德克_德鹿谷": "trv_Delu",
    "賽德克_都達": "trv_Duda",
    "賽德克_德固達雅": "trv_Tegu",
    "邵": "ssf_Thao",
    "太魯閣": "trv_Truk",
    "鄒": "tsu_Tsou",
    "雅美": "tao_Yami",
}


# 翻譯模型的語言代碼 → 語言別代碼（「族語及語言別名稱 - 語言別名稱.csv」）。
LANGUAGE_CODES = {}
for _language_code, _nllb_code in TRANSLATION_LANGUAGE_CODES.items():
    LANGUAGE_CODES[_nllb_code] = _language_code


# asr 的語言別選項：族語別 → [(顯示名稱, asr 語言別代碼)]。
# 顯示名稱與順序直接取自 mt 的 FORMOSAN_LANGUAGES_MAP（如「阿美_海岸」），
# 值是 asr 自己的語言別代碼（如 ami-x-pswl），兩邊經由 LANGUAGE_CODES 對應。
LANGUAGE_GROUPS = {}
for _ethnicity in ETHNICITIES:
    LANGUAGE_GROUPS[_ethnicity] = []
for _name, _nllb_code in FORMOSAN_LANGUAGES_MAP.items():
    LANGUAGE_GROUPS[_name.split("_")[0]].append((_name, LANGUAGE_CODES[_nllb_code]))
