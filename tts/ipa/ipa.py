import re
from typing import Optional, Tuple

import gradio as gr
from omegaconf import OmegaConf

g2p_config = OmegaConf.load("configs/g2p.yaml")
g2p_object = OmegaConf.to_object(g2p_config)["g2p"]


def lower_formosan_text(raw_text: str, language: str) -> str:
    text = list(raw_text.strip())
    if language == "賽夏":
        for i, char in enumerate(text):
            if char == "S":
                if i == 0:
                    text[i] = char.lower()
            else:
                text[i] = char.lower()
    elif language == "噶瑪蘭":
        for i, char in enumerate(text):
            if char == "R":
                text[i] = char
            else:
                text[i] = char.lower()
    else:
        for i, char in enumerate(text):
            text[i] = char.lower()

    text = "".join(text)

    return text


def replace_to_list(text: str, g2p: dict) -> Tuple[list, set]:
    # 創建標記陣列，記錄哪些位置已被處理
    marked = [False] * len(text)

    # 創建結果列表和臨時緩衝區
    result = []
    buffer = ""
    oovs = set()

    # 處理文本
    i = 0
    while i < len(text):
        # 如果當前位置已經被處理過，跳過
        if marked[i]:
            i += 1
            continue

        # 尋找匹配的 key
        found_key = None
        found_pos = -1

        for key in g2p:
            # 檢查當前位置是否匹配 key
            if i + len(key) <= len(text) and text[i : i + len(key)] == key:
                # 檢查這個範圍是否已有部分被處理過
                if not any(marked[i : i + len(key)]):
                    found_key = key
                    found_pos = i
                    break

        # 如果找到匹配的 key
        if found_key:
            # 先保存緩衝區中的內容（如果有）
            if buffer:
                result.append(buffer)
                buffer = ""

            # 添加替換後的值到結果列表
            result.append(g2p[found_key][0])

            # 標記已處理的位置
            for j in range(found_pos, found_pos + len(found_key)):
                marked[j] = True

            # 移到下一個未處理的位置
            i = found_pos + len(found_key)
        else:
            # 沒有匹配的 key，添加到緩衝區
            buffer += text[i]
            oovs.add(text[i])
            i += 1

    # 不要忘記添加最後的緩衝區內容
    if buffer:
        result.append(buffer)

    return result, oovs


def convert_to_ipa(
    text: str, g2p: dict, end_punctuations: list = ["!", "?", ".", ";", ","]
) -> Tuple[Optional[str], list]:
    result_list = []
    oovs_to_ipa = set()

    for word in text.split():
        ending_punct = ""
        if word and word[-1] in end_punctuations:
            ending_punct = word[-1]
            word = word[:-1]

        ipa_list, oovs = replace_to_list(word, g2p)
        if len(oovs):
            oovs_to_ipa.update(oovs)
            continue

        ipa_string = "".join(ipa_list) + ending_punct
        result_list.append(ipa_string)

    if len(oovs_to_ipa) or len(result_list) == 0:
        return None, sorted(oovs_to_ipa)

    result = " ".join(result_list)

    return result, []


def text_to_ipa(
    text: str, language: str, ignore_punctuation=False, ipa_with_ng=False
) -> str:
    text = lower_formosan_text(text, language)
    # text = text.replace("'", "’")
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = re.sub(r"[\"\-\“\”]", "", text)  # remove punctuation
    text = re.sub(r"[\ʼ\’\']", "ʼ", text)  # normalize ʼ
    text = text.replace("^", "⌃")  # normalize ⌃

    ipa, unknown_chars = convert_to_ipa(text, g2p_object[language])

    if len(unknown_chars) > 0:
        raise gr.Error(
            f"Unknown characters: {', '.join(unknown_chars)}. Please remove them and try again."
        )

    ipa = ipa.replace("ʦ", "t͡s").replace("ʨ", "t͡ɕ").replace("ʤ", "d͡ʒ")

    print(f"ipa: {ipa}")
    return ipa
