"""找候選參考音檔：從族語辭典或族語E樂園抽句子、下載、轉 wav、算 F0。

    python candidates.py --work WORK --speaker 雅美_女聲 --source dict
    python candidates.py --work WORK --speaker 魯凱_萬山_女聲 --source klokah --klokah-dialect 萬山魯凱語
    python candidates.py --work WORK --speaker 魯凱_霧台_女聲 --source dict --dict-dialect 霧台

輸出：
    WORK/candidates.json  候選參考音檔（含來源、F0、長度、自動性別），依「適合當參考」排序
    WORK/texts.json       試聽用的 10 句合成文字（和候選不重複）
"""
import argparse
import random
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from common import (
    DICT_DIR,
    USER_AGENT,
    analyze,
    expected_gender,
    language_of,
    normalize_text,
    save_json,
    to_wav,
)

KLOKAH_DIALECTOR = "https://web.klokah.tw/plugins/klokah/dialector/dialector.js"
# 族語E樂園的資料只能 GET，POST 會 403；短時間請求太多會整個網域連不上一陣子
KLOKAH_CONVERSATION = "https://web.klokah.tw/extension/con_data/xml/{did}/conversation.xml"
KLOKAH_SOUND = "https://klokah.tw/extension/con_data/sound/{did}/sentence/{did}c{lesson}s{k}.mp3"
KLOKAH_PAGE = "https://web.klokah.tw/extension/con_practice/index.php?d={did}&l={lesson}&view=dialogue"
KLOKAH_SLEEP = 3
DICT_SLEEP = 0.3

# 參考音檔 3–8 秒最好：太短聲音特徵不夠，超過 12 秒 F5-TTS 會截掉而對不上文字
MIN_WORDS = 4
MAX_WORDS = 12
GOOD_MIN_SECONDS = 3
GOOD_MAX_SECONDS = 8


def http_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def dict_rows(ethnicity, dialect):
    """讀族語辭典 xlsx 的例句列。例句列的「單字」欄常是空的，要往上找最近的詞條。"""
    import openpyxl

    paths = sorted(DICT_DIR.glob(f"*_[0-9][0-9]{ethnicity}語*.xlsx"))
    if not paths:
        raise SystemExit(f"{DICT_DIR} 找不到{ethnicity}語的 xlsx（kithann/ 不在 git 裡，要另外放）")
    path = paths[0]
    sheet = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = []
    seen = set()
    word, word_id, sense = None, None, None
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[0]:
            word_id, word = row[0], (row[3] or "").strip()
        if row[9]:
            sense = row[9]
        if dialect and (row[2] is None or dialect not in row[2]):
            continue
        text, zh, audio_id, audio_name, url = row[11], row[12], row[13], row[14], row[15]
        if not text or not url or url in seen:
            continue
        seen.add(url)
        rows.append({
            "id": f"ilrdf-dictionary-{audio_id}",
            "text": normalize_text(text),
            "zh": zh or "",
            "url": url,
            "source": {
                "來源": f"族語辭典 {path.name}",
                "語別": row[2],
                "詞條": word,
                "詞條ID": word_id,
                "釋義": sense,
                "例句ID": row[10],
                "例句音檔ID": audio_id,
                "例句音檔檔名": audio_name,
                "音檔": url,
            },
        })
    return rows


def klokah_did(dialect_name):
    js = http_get(KLOKAH_DIALECTOR).decode("utf-8")
    pattern = r'did:\s*"(\d+)",\s*lch:\s*"[^"]*",\s*dch:\s*"' + re.escape(dialect_name) + '"'
    match = re.search(pattern, js)
    if not match:
        raise SystemExit(f"dialector.js 找不到方言「{dialect_name}」，請用 dch 的全名，例如「萬山魯凱語」")
    return match.group(1)


def klokah_rows(dialect_name):
    did = klokah_did(dialect_name)
    root = ET.fromstring(http_get(KLOKAH_CONVERSATION.format(did=did)))
    rows = []
    for content in root.iter("content"):
        lesson = content.findtext("lessonId")
        k = 1
        while (content.findtext(f"con_scene{k}") or "") != "":
            text = (content.findtext(f"con_AB{k}") or "").strip()
            url = KLOKAH_SOUND.format(did=did, lesson=lesson, k=k)
            rows.append({
                "id": f"klokah-con_practice-{did}c{lesson}s{k}",
                "text": normalize_text(text),
                "zh": (content.findtext(f"con_CH{k}") or "").strip(),
                "url": url,
                "source": {
                    "來源": f"族語E樂園 生活會話篇（{dialect_name}）",
                    "頁面": KLOKAH_PAGE.format(did=did, lesson=lesson),
                    "課": int(lesson),
                    "句": k,
                    "對話角色": (content.findtext(f"con_someone{k}") or "").strip(),
                    "音檔": url,
                },
            })
            k += 1
    return rows


def word_count_ok(row):
    # 「giway/wiway」這種列出兩種說法的句子，錄音只念其中一種，文字對不上音檔
    if "/" in row["text"]:
        return False
    words = len(row["text"].split())
    return MIN_WORDS <= words <= MAX_WORDS


def suitability(candidate, expected):
    """越小越適合：性別對、長度在 3–8 秒、有聲比例高。"""
    a = candidate["analysis"]
    score = 0.0
    if a["gender"] != expected:
        score += 10
    if not GOOD_MIN_SECONDS <= a["dur"] <= GOOD_MAX_SECONDS:
        score += 3
    score -= a["voiced"]
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, help="工作資料夾")
    parser.add_argument("--speaker", required=True, help="refs.yaml 的配音員名稱，例如 雅美_女聲")
    parser.add_argument("--source", choices=["dict", "klokah"], required=True)
    parser.add_argument("--dict-dialect", help="族語辭典「語別」欄要包含的字，例如 霧台；雅美語等不分語別的留空")
    parser.add_argument("--klokah-dialect", help="族語E樂園方言全名，例如 萬山魯凱語")
    parser.add_argument("--count", type=int, default=40, help="下載幾個候選（預設 40）")
    parser.add_argument("--texts", type=int, default=10, help="試聽句數（預設 10）")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    work = Path(args.work)
    expected = expected_gender(args.speaker)
    ethnicity = language_of(args.speaker).split("_")[0]

    if args.source == "dict":
        rows = dict_rows(ethnicity, args.dict_dialect)
        sleep = DICT_SLEEP
    else:
        if not args.klokah_dialect:
            parser.error("--source klokah 要加 --klokah-dialect")
        rows = klokah_rows(args.klokah_dialect)
        sleep = KLOKAH_SLEEP
    print(f"找到 {len(rows)} 句有音檔的句子", file=sys.stderr)

    pool = []
    for row in rows:
        if word_count_ok(row):
            pool.append(row)
    if len(pool) < args.count + args.texts:
        raise SystemExit(f"{MIN_WORDS}–{MAX_WORDS} 詞的句子只有 {len(pool)} 句，不夠抽")
    rng = random.Random(args.seed)
    picked = rng.sample(pool, args.count + args.texts)
    texts = []
    for row in picked[args.count:]:
        texts.append([row["text"], row["zh"]])
    save_json(work / "texts.json", texts)

    candidates = []
    for row in picked[:args.count]:
        mp3 = work / "download" / f"{row['id']}.mp3"
        wav = work / "wav" / f"{row['id']}.wav"
        if not mp3.exists():
            try:
                data = http_get(row["url"])
            except Exception as e:
                print(f"下載失敗 {row['id']}：{e}", file=sys.stderr)
                time.sleep(sleep)
                continue
            mp3.parent.mkdir(parents=True, exist_ok=True)
            mp3.write_bytes(data)
            time.sleep(sleep)
        to_wav(mp3, wav)
        row["wav"] = str(wav)
        row["analysis"] = analyze(wav)
        candidates.append(row)

    ranked = sorted(candidates, key=lambda c: suitability(c, expected))
    save_json(work / "candidates.json", ranked)
    for c in ranked:
        a = c["analysis"]
        f0 = "-" if a["f0"] is None else f"{a['f0']:.0f}"
        print(f"{c['id']}\t{a['gender']}\t{f0} Hz\t{a['dur']:.1f}s\tvoiced {a['voiced']}\t{c['text']}")


if __name__ == "__main__":
    main()
