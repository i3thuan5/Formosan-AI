"""NLLB v3（ILRDF/nllb-600m-formosan-all-finetune-v3）的載入與大小寫處理，mt 與 asr 共用。"""

import re


def load_tokenizer(name: str):
    # 重量級套件在這裡才 import，tox 的單元測試環境只裝 pytest，
    # map_sentence_starts 才測得到。
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    # v3 的 tokenizer 有 byte fallback。AutoTokenizer 自己解析 tokenizer.json 時會
    # 把 byte fallback 丟掉，詞表外的字（蕉、猴、橘…）會變成 <unk>，
    # 所以先用 tokenizers 載好再交給它。
    path = hf_hub_download(name, "tokenizer.json")
    return AutoTokenizer.from_pretrained(
        name, tokenizer_object=Tokenizer.from_file(path)
    )


# v3 的訓練資料把阿美、布農、賽德克（含太魯閣）的句首改成小寫：
# 輸入先照樣轉小寫，輸出再把句首轉回大寫。
# 噶瑪蘭、賽夏的大小寫代表不同音，不能動。
SENTENCE_CASE_PREFIXES = ("ami_", "bnn_", "trv_")
SENTENCE_BREAK = re.compile(r"[.?!…]+[\"”’)」』]*\s+")


def map_sentence_starts(text: str, lang: str, fn) -> str:
    """lang 是 NLLB 語言代碼（如 ami_Coas）；fn 是 str.lower 或 str.upper。"""
    if not lang.startswith(SENTENCE_CASE_PREFIXES):
        return text
    chars = list(text)
    starts = [0]
    for m in SENTENCE_BREAK.finditer(text):
        starts.append(m.end())
    for i in starts:
        while i < len(chars) and not chars[i].isspace():
            if chars[i].lower() != chars[i].upper():
                chars[i] = fn(chars[i])
                break
            i += 1
    return "".join(chars)
