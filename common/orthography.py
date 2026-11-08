"""族語書寫符號正規化：每個規則是一個小函式，每個語別設定要套用哪幾個。"""
import re


def _letter_table(pairs):
    """小寫的對應寫一次，大寫的對應自動產生（如 o→u 同時產生 O→U）。"""
    table = {}
    for source, target in pairs.items():
        table[ord(source)] = target
        table[ord(source.upper())] = target.upper()
    return str.maketrans(table)


_O2U = _letter_table({'o': 'u'})


def o2u(text):
    return text.translate(_O2U)


_U2O = _letter_table({'u': 'o'})


def u2o(text):
    return text.translate(_U2O)


_F2V = _letter_table({'f': 'v'})


def f2v(text):
    return text.translate(_F2V)


_B2V = _letter_table({'b': 'v'})


def b2v(text):
    return text.translate(_B2V)


_V2F = _letter_table({'v': 'f'})


def v2f(text):
    return text.translate(_V2F)


_B2F = _letter_table({'b': 'f'})


def b2f(text):
    return text.translate(_B2F)


_V2B = _letter_table({'v': 'b'})


def v2b(text):
    return text.translate(_V2B)


_QUOTE_2_U02BC = str.maketrans({'’': 'ʼ', "'": 'ʼ'})


def quote_2_u02bc(text):
    return text.translate(_QUOTE_2_U02BC)


def lower_Rupper(text):
    return text.lower().replace('r', 'R')


def o2u_except_uo(text):
    # uo 念 [uo]，是兩個元音連續，不是 u 的同位音 [o]。排在 lower_Rupper 之後，所以只處理小寫。
    return re.sub(r'(?<!u)o', 'u', text)


# 詞界只看英文字母，因為 \b 會把 ʼ 和漢字當成詞內字元。虛詞比對到時整個詞跳過，裡面的 o 就不會被第二個分支處理。
_TRUKU_O = re.compile(r'(?P<keep>(?<![a-z])(?:o|do|hido|kido|empeehido)(?![a-z]))|o(?!w)', re.IGNORECASE)


def _keep_particle_or_o2u(match):
    if match.group('keep'):
        return match.group(0)
    return o2u(match.group(0))


def o2u_except_ow_particles(text):
    # 太魯閣語的虛詞固定寫 o，ow 是 /aw/ 的同位音，都不是 u 的另一種寫法。
    return _TRUKU_O.sub(_keep_particle_or_o2u, text)


# key 的順序照「族語及語言別名稱 - 語言別名稱.csv」的顯示排序先後，沒有規則的語別 list 留空。
LANGUAGE_RULES = {
    'ami-x-iams': [o2u, f2v, b2v, quote_2_u02bc],
    'ami-x-skl': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-pswl': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-frng': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-pld': [u2o, v2f, b2f, quote_2_u02bc],
    'tay-x-sql': [],
    'tay-x-sul': [],
    'tay-x-cql': [],
    'tay-x-kls': [],
    'tay-x-mtuw': [],
    'tay-x-plngw': [],
    'pwn-x-kcdsn': [],
    'pwn-x-vnrn': [],
    'pwn-x-pnvn': [],
    'pwn-x-ynvl': [],
    'bnn-x-td': [],
    'bnn-x-bkh': [],
    'bnn-x-vtn': [],
    'bnn-x-bnz': [],
    'bnn-x-isbk': [],
    'pyu-x-pym': [],
    'pyu-x-ktrp': [],
    'pyu-x-mkzy': [],
    'pyu-x-ksvk': [],
    'dru-x-trmk': [],
    'dru-x-ngdr': [],
    'dru-x-lbw': [],
    'dru-x-kgdv': [],
    'dru-x-tldr': [],
    'dru-x-opnh': [],
    'tsu': [],
    'xsy': [],
    'tao': [],
    'ssf': [],
    'ckv': [lower_Rupper, quote_2_u02bc, o2u_except_uo],
    'trv-x-truku': [o2u_except_ow_particles],
    'szy': [o2u, v2b, quote_2_u02bc],
    'trv-x-td': [],
    'trv-x-tgdy': [],
    'trv-x-trk': [],
    'sxr': [],
    'xnb': [],
}


def normalize(text, language):
    for rule in LANGUAGE_RULES.get(language, []):
        text = rule(text)
    return text
