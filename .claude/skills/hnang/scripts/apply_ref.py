"""把選定的候選參考音檔放進 tts/assets/ref_wav/，並改 refs.yaml 該配音員的 text、wav、source。

    python apply_ref.py --work WORK --speaker 雅美_女聲 --ref dict-A5413384-...

只改這位配音員的區塊，其他行保持原樣。不會 git add，wav 要請使用者自己加進版本控制。
"""
import argparse
import re
import shutil
from pathlib import Path

import yaml

from common import REF_WAV_DIR, REFS_PATH, language_of, load_json

SOURCE_KEYS_ORDER = ["來源", "頁面", "語別", "課", "句", "對話角色", "詞條", "詞條ID", "釋義", "例句ID", "例句音檔ID",
                     "例句音檔檔名", "中文", "音檔"]


def yaml_scalar(value):
    """用 yaml.safe_dump 產生單一值的寫法，遇到冒號、引號等會自動加引號。"""
    return yaml.safe_dump(value, allow_unicode=True, default_flow_style=True, width=1000).strip().removesuffix("...").strip()


def speaker_block(speaker, text, wav_rel, source):
    lines = [f"{speaker}:", f"  text: {yaml_scalar(text)}", f"  wav: {wav_rel}", "  source:"]
    for key in SOURCE_KEYS_ORDER:
        if source.get(key) not in (None, ""):
            lines.append(f"    {key}: {yaml_scalar(source[key])}")
    return "\n".join(lines) + "\n"


def block_pattern(speaker):
    return re.compile(rf"^{re.escape(speaker)}:\n(?:[ \t].*\n|\n)*", re.MULTILINE)


def replace_block(content, speaker, block):
    """換掉從「speaker:」到下一個頂層 key 之前的所有行；沒有這位配音員時，插在同一語別最後一位後面。"""
    pattern = block_pattern(speaker)
    if pattern.search(content):
        return pattern.sub(lambda _m: block, content, count=1)
    last = None
    for name in yaml.safe_load(content):
        if language_of(name) == language_of(speaker):
            last = name
    if last is None:
        raise SystemExit(f"refs.yaml 沒有「{speaker}」，也沒有同語別的配音員可以接在後面")
    end = block_pattern(last).search(content).end()
    print(f"新增配音員 {speaker}，接在 {last} 後面")
    return content[:end] + block + content[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--ref", required=True, help="candidates.json 的 id")
    args = parser.parse_args()

    candidate = None
    for c in load_json(Path(args.work) / "candidates.json"):
        if c["id"] == args.ref:
            candidate = c
    if candidate is None:
        raise SystemExit(f"candidates.json 沒有 {args.ref}")

    dst = REF_WAV_DIR / f"{candidate['id']}.wav"
    shutil.copy(candidate["wav"], dst)
    source = dict(candidate["source"])
    if candidate.get("zh"):
        source["中文"] = candidate["zh"]

    content = REFS_PATH.read_text(encoding="utf-8")
    block = speaker_block(args.speaker, candidate["text"], f"./assets/ref_wav/{dst.name}", source)
    REFS_PATH.write_text(replace_block(content, args.speaker, block), encoding="utf-8")

    check = yaml.safe_load(REFS_PATH.read_text(encoding="utf-8"))[args.speaker]
    if check["text"] != candidate["text"]:
        raise SystemExit(f"寫回後讀到的 text 不一致：{check['text']!r}")
    print(f"已更新 {REFS_PATH} 的 {args.speaker}")
    print(f"已複製 {dst}（未 git add）")


if __name__ == "__main__":
    main()
