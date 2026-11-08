import csv
from pathlib import Path

from languages import LANGUAGE_GROUPS
from orthography import (
    LANGUAGE_RULES,
    b2f,
    b2v,
    f2v,
    lower_Rupper,
    normalize,
    o2u,
    o2u_except_uo,
    quote_2_u02bc,
    u2o,
    v2b,
    v2f,
)

KLOKAH_CSV = Path(__file__).parents[1] / 'data' / 'klokah_orthography.csv'
CKV_CSV = Path(__file__).parents[1] / 'data' / 'ckv_dictionary_orthography.csv'
# 含全部會被規則碰到的字母（大小寫）、撇號與 ^
PROBE = 'oOuUfFvVbBrRPAB ’\' ʼ ^'


def test_o2u_lower():
    assert o2u('fita’ol') == 'fita’ul'


def test_o2u_upper():
    assert o2u('Ofad OLO') == 'Ufad ULU'


def test_o2u_other_chars_unchanged():
    text = 'uvbf ABC 123 ’\' ^ 你好, .'
    assert o2u(text) == text


def test_u2o_lower():
    assert u2o('vitaʼul') == 'vitaʼol'


def test_u2o_upper():
    assert u2o('Ufad ULU') == 'Ofad OLO'


def test_u2o_other_chars_unchanged():
    text = 'ovbf ABC 123 ’\' ^ 你好, .'
    assert u2o(text) == text


def test_f2v_lower():
    assert f2v('fita’ul') == 'vita’ul'


def test_f2v_upper():
    assert f2v('Fita FAFA') == 'Vita VAVA'


def test_f2v_other_chars_unchanged():
    text = 'oubv ABC 123 ’\' ^ 你好, .'
    assert f2v(text) == text


def test_b2v_lower():
    assert b2v('babahiyan') == 'vavahiyan'


def test_b2v_upper():
    assert b2v('Baba BABA') == 'Vava VAVA'


def test_b2v_other_chars_unchanged():
    text = 'oufv ACD 123 ’\' ^ 你好, .'
    assert b2v(text) == text


def test_v2f_lower():
    assert v2f('vitaʼol') == 'fitaʼol'


def test_v2f_upper():
    assert v2f('Vita VAVA') == 'Fita FAFA'


def test_v2f_other_chars_unchanged():
    text = 'oufb ACD 123 ’\' ^ 你好, .'
    assert v2f(text) == text


def test_b2f_lower():
    assert b2f('babahi') == 'fafahi'


def test_b2f_upper():
    assert b2f('Baba BABA') == 'Fafa FAFA'


def test_b2f_other_chars_unchanged():
    text = 'oufv ACD 123 ’\' ^ 你好, .'
    assert b2f(text) == text


def test_v2b_lower():
    assert v2b('davak') == 'dabak'


def test_v2b_upper():
    assert v2b('Vava VAVA') == 'Baba BABA'


def test_v2b_f_unchanged():
    assert v2b('fataan Fataan') == 'fataan Fataan'


def test_v2b_other_chars_unchanged():
    text = 'oufb ACD 123 ’\' ^ 你好, .'
    assert v2b(text) == text


def test_quote_2_u02bc_curly_and_ascii():
    assert quote_2_u02bc('nga’ay ho, ma\'emin') == 'ngaʼay ho, maʼemin'


def test_quote_2_u02bc_already_u02bc_unchanged():
    assert quote_2_u02bc('ngaʼay') == 'ngaʼay'


def test_quote_2_u02bc_caret_unchanged():
    assert quote_2_u02bc('^orip') == '^orip'


def test_quote_2_u02bc_other_chars_unchanged():
    text = 'oufvb ACD 123 ^ 你好, .'
    assert quote_2_u02bc(text) == text


def test_o2u_except_uo_single_o():
    assert o2u_except_uo('qoR') == 'quR'


def test_o2u_except_uo_keeps_o_after_u():
    assert o2u_except_uo('suykuo suykuoan') == 'suykuo suykuoan'


def test_o2u_except_uo_consecutive_o():
    assert o2u_except_uo('oo') == 'uu'


def test_o2u_except_uo_upper_unchanged():
    assert o2u_except_uo('Oba') == 'Oba'


def test_o2u_except_uo_other_chars_unchanged():
    text = 'uvbf ABC 123 ’\' ^ 你好, .'
    assert o2u_except_uo(text) == text


def test_lower_Rupper_lowers_other_upper():
    assert lower_Rupper('PateRis') == 'pateRis'


def test_lower_Rupper_r_to_R():
    assert lower_Rupper('saribung') == 'saRibung'


def test_lower_Rupper_R_kept():
    assert lower_Rupper('Rabas naRin tebuquR') == 'Rabas naRin tebuquR'


def test_lower_Rupper_other_chars_unchanged():
    text = '123 ’\' ^ 你好, .'
    assert lower_Rupper(text) == text


def test_other_language_unchanged():
    text = 'lalu’ mu o Yumin'
    assert normalize(text, 'tay-x-sql') == text


def test_chinese_unchanged():
    text = '你好 o\''
    assert normalize(text, 'zho_Hant') == text


def test_none_language_unchanged():
    text = 'fita’ol'
    assert normalize(text, None) == text


def test_iams_o_and_f():
    assert normalize('fita’ol', 'ami-x-iams') == 'vitaʼul'


def test_iams_b():
    assert normalize('babahiyan', 'ami-x-iams') == 'vavahiyan'


def test_common_u_and_v():
    assert normalize('vitaʼul', 'ami-x-pswl') == 'fitaʼol'


def test_common_b():
    assert normalize('babahi', 'ami-x-skl') == 'fafahi'


def test_szy_o_and_v():
    assert normalize('davak abo', 'szy') == 'dabak abu'


def test_szy_keeps_f_and_z():
    assert normalize('fataan zazan', 'szy') == 'fataan zazan'


def test_quote_unified():
    assert normalize('nga’ay ho, ma\'emin', 'ami-x-pswl') == 'ngaʼay ho, maʼemin'


def test_glottal_stop_caret_kept():
    assert normalize('^orip', 'ami-x-pswl') == '^orip'


def test_case_kept():
    assert normalize('Ofad Pangcah', 'ami-x-iams') == 'Uvad Pangcah'


def test_normalize_twice_same_as_once():
    once = normalize('Ofad fita’ol babahiyan', 'ami-x-iams')
    assert normalize(once, 'ami-x-iams') == once


def test_truku_unchanged():
    text = 'bowyak Away’'
    assert normalize(text, 'trv-x-truku') == text


def test_ckv_o():
    assert normalize('Oba qoR', 'ckv') == 'uba quR'


def test_ckv_keeps_o_after_u():
    assert normalize('suykuo suykuoan Uo', 'ckv') == 'suykuo suykuoan uo'


def test_ckv_consecutive_o():
    assert normalize('qoo', 'ckv') == 'quu'


def test_ckv_r():
    assert normalize('saribung serai', 'ckv') == 'saRibung seRai'


def test_ckv_keeps_R():
    assert normalize('Rabas naRin tebuquR', 'ckv') == 'Rabas naRin tebuquR'


def test_ckv_proper_noun_not_capitalized():
    assert normalize('PateRis ika ti ukud', 'ckv') == 'pateRis ika ti ukud'


def test_ckv_keeps_b_and_z():
    assert normalize('babuzan', 'ckv') == 'babuzan'


def test_ckv_quote_unified():
    assert normalize('ʼnay ’ta \'qi', 'ckv') == 'ʼnay ʼta ʼqi'


def test_ckv_dictionary_words_rewritten():
    assert normalize('suykuo', 'ckv') == 'suykuo'
    assert normalize('PateRis', 'ckv') == 'pateRis'
    assert normalize('saribung', 'ckv') == 'saRibung'


def test_ckv_dictionary_sentences_are_fixed_points():
    with open(CKV_CSV, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        assert normalize(row['sentence'], row['language']) == row['sentence'], row


def test_klokah_sentences_are_fixed_points():
    with open(KLOKAH_CSV, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        sentence = row['sentence']
        expected = sentence.replace('’', 'ʼ').replace("'", 'ʼ')
        assert normalize(sentence, row['language']) == expected, row


def test_all_languages_normalize_twice_same_as_once():
    for language in LANGUAGE_RULES:
        once = normalize(PROBE, language)
        assert normalize(once, language) == once, language


def test_language_rules_keys_match_language_groups_in_order():
    codes = []
    for group in LANGUAGE_GROUPS.values():
        for _label, code in group:
            codes.append(code)
    assert len(codes) == 42
    assert list(LANGUAGE_RULES) == codes
