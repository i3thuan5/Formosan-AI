"""從族語E樂園的資料產生 sentences.yaml：每個語別 8 句不重複的短句，給 check_tts_speakers.py 用。

    cd tts && python ../tests/tts_speakers/build_sentences.py

資料在 kithann/E樂園/（不在 git 裡，要另外放）：
- 句型篇國中 生活百句（xml/<方言編號>/*.xml 的 sentenceA/B/CAb）：日常短句，優先使用
- 文化篇（6.文化篇/klokah_cuture_sentence.csv）：生活百句不夠時，從句子依標點切出短句補

每句都用 tts/ipa 的 text_to_ipa 驗證，轉不了的字就不收，所以要在 tts/ 目錄、用 tts 的環境執行。
"""
import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tts"))
from ipa.ipa import text_to_ipa  # noqa: E402

DATA_DIR = REPO_ROOT / "kithann" / "E樂園"
SENTENCE_XML_DIR = DATA_DIR / "E樂園 - 句型篇國中" / "xml"
CULTURE_CSV = DATA_DIR / "6.文化篇" / "klokah_cuture_sentence.csv"
DIALECT_LIST = DATA_DIR / "E樂園方言編號.txt"
EXAMPLES_PATH = REPO_ROOT / "tts" / "configs" / "examples.yaml"
OUTPUT = Path(__file__).resolve().parent / "sentences.yaml"

# 方言名稱是「方言＋族名＋語」，這些族要拆成「族_方言」
MULTI_DIALECT = ["阿美", "泰雅", "賽德克", "布農", "排灣", "魯凱", "卑南"]
MIN_WORDS = 3
MAX_WORDS = 8
COUNT = 8


class IndentedDumper(yaml.SafeDumper):
    """list 也縮排，符合 yamllint 的預設規則。"""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def normalize(text):
    text = re.sub(r"\s+", " ", text).strip()
    # 原文常在標點前多一個空白，例如「isu ?」
    text = re.sub(r"\s+([,.;?!])", r"\1", text)
    return re.sub(r"[’']", "ʼ", text)


def language_key(name):
    name = name.removesuffix("語")
    for ethnicity in MULTI_DIALECT:
        if name.endswith(ethnicity) and name != ethnicity:
            return f"{ethnicity}_{name.removesuffix(ethnicity)}"
    return name


def dialects():
    """方言編號 → 語別（和 examples.yaml 的 key 相同）。"""
    result = {}
    for line in DIALECT_LIST.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\s*(\d+):\s*(\S+)", line)
        if match:
            result[match.group(1)] = language_key(match.group(2))
    return result


def short_enough(text):
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


def life_sentences(dialect_id):
    """生活百句：一組 A/B/C 三句，有些用「 / 」列出兩種說法。"""
    rows = []
    for path in sorted((SENTENCE_XML_DIR / dialect_id).glob("*.xml"), key=lambda p: int(p.stem)):
        for item in ET.parse(path).getroot().iter("item"):
            for part in ["A", "B", "C"]:
                ab = item.findtext(f"sentence{part}Ab") or ""
                zh = item.findtext(f"sentence{part}Ch") or ""
                for alternative in ab.split(" / "):
                    text = normalize(alternative)
                    if text and short_enough(text):
                        rows.append({
                            "text": text,
                            "zh": zh.strip(),
                            "source": f"族語E樂園 句型篇國中 生活百句「{item.findtext('classCh')}」"
                                      f"（autoId {item.findtext('autoId')} {part}）",
                        })
    return rows


def culture_clauses(dialect_id, culture_rows):
    rows = []
    for row in culture_rows:
        if row["dialectId"] != dialect_id:
            continue
        for part in re.split(r"(?<=[,.;?!])\s+", row["sentenceAb"].strip()):
            text = normalize(re.sub(r"[,;]$", ".", part))
            if short_enough(text):
                rows.append({
                    "text": text,
                    "zh": "",
                    "source": f"族語E樂園 文化篇（textId {row['textId']}，autoId {row['autoId']}，切出的短句）",
                })
    return rows


def convertible(text, language):
    try:
        text_to_ipa(text, language)
    except Exception:
        return False
    return True


def spread_pick(rows, count, language, picked_texts):
    """平均分散在整份資料裡挑，避免都集中在同一課。"""
    picked = []
    if not rows:
        return picked
    step = max(1, len(rows) // (count * 2))
    for offset in range(step):
        for i in range(offset, len(rows), step):
            row = rows[i]
            if row["text"] in picked_texts or not convertible(row["text"], language):
                continue
            picked.append(row)
            picked_texts.add(row["text"])
            if len(picked) == count:
                return picked
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=COUNT, help=f"每個語別幾句（預設 {COUNT}）")
    args = parser.parse_args()

    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        languages = list(yaml.safe_load(f))
    with open(CULTURE_CSV, encoding="utf-8") as f:
        culture_rows = list(csv.DictReader(f))
    by_language = {}
    for dialect_id, language in dialects().items():
        by_language[language] = dialect_id

    output = {}
    for language in languages:
        if language not in by_language:
            print(f"✗ {language}：E樂園方言編號裡找不到", file=sys.stderr)
            continue
        dialect_id = by_language[language]
        picked_texts = set()
        picked = spread_pick(life_sentences(dialect_id), args.count, language, picked_texts)
        if len(picked) < args.count:
            picked += spread_pick(culture_clauses(dialect_id, culture_rows), args.count - len(picked), language,
                                  picked_texts)
        output[language] = picked
        print(f"{language}\t{len(picked)} 句", file=sys.stderr)

    header = (
        "# 給 tests/tts_speakers/check_tts_speakers.py 用的短句，每個語別不重複。\n"
        "# 由 build_sentences.py 從 kithann/E樂園 產生，要改請改腳本後重新產生。\n"
    )
    body = yaml.dump(output, Dumper=IndentedDumper, allow_unicode=True, sort_keys=False, width=110,
                     explicit_start=True)
    OUTPUT.write_text(header + body, encoding="utf-8")
    print(f"寫到 {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
