"""上線前檢查：tts「預設配音員」合成出來的聲音，是否穩定維持參考音檔的聲音。

    $ BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e tts_speakers -- --audio-dir /tmp/tts-speakers
    $ BASE_URL=... python tests/tts_speakers/check_tts_speakers.py --speakers 雅美_女聲,魯凱_萬山_女聲 --samples 3

F5-TTS 每次取樣不同，參考音檔太短或不清楚時，同一位配音員有時會合成出另一個性別，
**短句特別容易**（2026-10 雅美舊參考音檔：長例句 20 次都是女聲，短句 20 次有 18 次是男聲）。
所以每位配音員合成多句、長短句都有，每句各自判斷。
短句來自 sentences.yaml（族語E樂園的生活百句等，由 build_sentences.py 產生），長句是 examples.yaml 的例句。

判斷方式（見 README.md）：
- 每句合成結果的 F0 和該配音員參考音檔的 F0 比，差太多 → 失敗（聲音變了）
- 參考音檔本身的 F0 和名稱標的男聲／女聲相反 → 警告，請人工試聽確認名稱；
  refs.yaml 有 gender_confirmed（人工確認過）就不再警告
- refs.yaml 有 gender_unstable（人工試聽已確認會變聲，只要錯 1 句就算）→ 直接失敗、不再合成，換參考音檔後拿掉
- refs.yaml 有 gender_checked（人工試聽通過過）→ 自動判斷失敗的句子改列警告，不擋上線，但仍請人試聽

全部通過 exit 0，任一項失敗 exit 1。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "tests" / "production_tests"))
from pitch import gender_of, median_f0  # noqa: E402
from reporting import check, connect, report_result  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "https://ai-labs.ilrdf.org.tw").rstrip("/")
# 本機直接跑 tts/app.py 時沒有反向代理的路徑，用 TTS_URL=http://127.0.0.1:7860/ 指定
TTS_URL = os.environ.get("TTS_URL", f"{BASE_URL}/hnang-kari-ai-asi-sluhay/")
API_NAME = "/default_speaker_tts"
# 切換族別後，配音員 Radio 的 choices 才會有該族的配音員；這個事件沒有 api_name，
# 用回傳元件的標籤找出它的端點名稱，避免 gradio 自動編號變動
SPEAKER_RADIO_LABEL = "步驟二：選擇配音員"

REFS_PATH = "tts/configs/refs.yaml"
EXAMPLES_PATH = "tts/configs/examples.yaml"
SENTENCES_PATH = Path(__file__).resolve().parent / "sentences.yaml"
DEFAULT_SAMPLES = 10

# 每 5 句裡有幾句用短句
SHORT_PER_FIVE = 3
SHORT_MIN_WORDS = 3
SHORT_MAX_WORDS = 8

# 合成 F0 ÷ 參考音檔 F0。2026-10-15 對正式機 46 位配音員各 10 句：
# 穩定的配音員多在 0.84–1.12；確定變成另一個性別的句子在 0.39–0.66
FAIL_LOW = 0.70
WARN_LOW = 0.80
WARN_HIGH = 1.25
FAIL_HIGH = 1.40


def read_repo_file(path, rev):
    """讀工作目錄或某個 git commit 的檔案。受測主機部署的版本和工作目錄不同時用 rev。"""
    if rev is None:
        return (REPO_ROOT / path).read_bytes()
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def language_of(speaker):
    # 和 tts/app.py 的 default_speaker_tts 相同
    return re.sub(r"_[男女]聲[12]?", "", speaker)


def expected_gender(speaker):
    if "女聲" in speaker:
        return "女"
    if "男聲" in speaker:
        return "男"
    raise AssertionError(f"配音員名稱沒有標男聲或女聲：{speaker}")


def short_sentences(text):
    """把例句依標點切開，取 3–8 個詞的片段，句尾的逗號換成句號。sentences.yaml 沒有該語別時才用。"""
    result = []
    for part in re.split(r"(?<=[,.;?!])\s+", text.strip()):
        if SHORT_MIN_WORDS <= len(part.split()) <= SHORT_MAX_WORDS:
            result.append(re.sub(r"[,;]$", ".", part))
    return result


def texts_for(speaker, refs, examples, sentences, count):
    """短句用 sentences.yaml，長句用 examples.yaml 的例句；不夠就重複，同一句每次合成結果也不同。"""
    language = language_of(speaker)
    longs = []
    shorts = []
    for item in examples.get(language, []):
        longs.append(item["text"])
    for item in sentences.get(language, []):
        shorts.append(item["text"])
    if not shorts:
        for text in longs:
            shorts.extend(short_sentences(text))
    if not longs:
        longs.append(refs[speaker]["text"])
    if not shorts:
        shorts = longs
    result = []
    short_index = 0
    long_index = 0
    for i in range(count):
        if i % 5 < SHORT_PER_FIVE:
            result.append(("短", shorts[short_index % len(shorts)]))
            short_index += 1
        else:
            result.append(("長", longs[long_index % len(longs)]))
            long_index += 1
    return result


def find_speaker_endpoint(client):
    endpoints = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
    for name, info in endpoints.items():
        for returned in info["returns"]:
            if returned.get("label") == SPEAKER_RADIO_LABEL:
                return name
    raise AssertionError(f"找不到回傳「{SPEAKER_RADIO_LABEL}」的端點")


def ratio_status(ratio):
    if ratio is None:
        return "警告"
    if ratio < FAIL_LOW or ratio > FAIL_HIGH:
        return "失敗"
    if ratio < WARN_LOW or ratio > WARN_HIGH:
        return "警告"
    return "通過"


def keep_audio(path, audio_dir, row):
    """把音檔與判斷結果留在 audio_dir，供人工試聽（/hnang skill 可以做成試聽網頁）。"""
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), audio_dir / row["audio"])
    with open(audio_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_reference(speaker, ref, ref_f0, warnings):
    def run():
        expected = expected_gender(speaker)
        if ref_f0 is None:
            raise AssertionError("參考音檔估不出 F0")
        actual = gender_of(ref_f0)
        confirmed = ref.get("gender_confirmed") or {}
        if confirmed.get("gender") == expected:
            return f"F0 {ref_f0:.0f} Hz，{confirmed.get('date', '')} 人工確認是{expected}聲"
        if actual not in (expected, "?"):
            warnings.append(f"{speaker} 的參考音檔 F0 {ref_f0:.0f} Hz 像{actual}聲，請試聽確認名稱")
            return f"⚠ F0 {ref_f0:.0f} Hz 像{actual}聲，請試聽確認名稱"
        return f"F0 {ref_f0:.0f} Hz"
    return run


def check_sample(client, speaker, index, length, text, ref_f0, checked, warnings, audio_dir):
    def run():
        path = Path(client.predict(speaker, text, api_name=API_NAME))
        f0, _voiced = median_f0(path)
        ratio = None if f0 is None else f0 / ref_f0
        status = ratio_status(ratio)
        downgraded = False
        if status == "失敗" and checked:
            # 人工試聽通過過的配音員，F0 判斷可能是誤判（例如沙啞的男聲），改列警告請人聽
            status = "警告"
            downgraded = True
        if audio_dir:
            keep_audio(path, audio_dir, {
                "speaker": speaker, "n": index, "length": length, "text": text,
                "audio": f"{speaker}_{index:02d}.wav", "expected": expected_gender(speaker),
                "ref_f0": round(ref_f0, 1), "f0": None if f0 is None else round(f0, 1),
                "ratio": None if ratio is None else round(ratio, 2), "gender": gender_of(f0), "status": status,
            })
        else:
            path.unlink(missing_ok=True)
        if f0 is None:
            warnings.append(f"{speaker} 第 {index} 句估不出 F0，請人工試聽")
            return "⚠ 估不出 F0，請人工試聽"
        detail = f"{length}句 F0 {f0:.0f} Hz，參考音檔的 {ratio:.2f} 倍"
        if status == "失敗":
            raise AssertionError(f"{detail}，聲音變了：{text}")
        if downgraded:
            detail += f"，自動判斷會失敗，但 {checked.get('date', '')} 人工試聽通過過"
        if status == "警告":
            warnings.append(f"{speaker} 第 {index} 句{detail}，請人工試聽")
            return f"⚠ {detail}，請人工試聽"
        return detail
    return run


def run(client, speakers, refs, examples, sentences, ref_wav, samples, failures, warnings, audio_dir=None):
    found = {}

    def find():
        found["endpoint"] = find_speaker_endpoint(client)
        return found["endpoint"]

    if check("找切換族別的端點", find, failures, service="tts") is None:
        return

    for speaker in speakers:
        print(f"\n[{speaker}]")
        unstable = refs[speaker].get("gender_unstable")
        if unstable:
            def known(unstable=unstable):
                raise AssertionError(f"{unstable.get('date', '')} 人工試聽已確認會變聲，不再合成：{unstable.get('note', '')}")
            check(f"{speaker} 已知會變聲", known, failures, service="tts")
            continue
        ref_f0, _voiced = median_f0(ref_wav(speaker))
        if check("參考音檔", check_reference(speaker, refs[speaker], ref_f0, warnings), failures,
                 service="tts") is None:
            continue

        # 每位配音員前都切換一次族別，同一個 session 的 Radio choices 才會包含他
        ethnicity = speaker.split("_")[0]

        def switch():
            client.predict(ethnicity, api_name=found["endpoint"])

        if check(f"切換族別 {ethnicity}", switch, failures, service="tts") is None:
            continue
        for i, (length, text) in enumerate(texts_for(speaker, refs, examples, sentences, samples), start=1):
            check(f"{speaker} 第 {i} 句",
                  check_sample(client, speaker, i, length, text, ref_f0, refs[speaker].get("gender_checked"), warnings,
                               audio_dir),
                  failures, service="tts")


def summarize(audio_dir):
    """依配音員列出 通過／警告／失敗 句數與比值範圍，方便一眼看出哪位不穩。"""
    rows = {}
    with open(Path(audio_dir) / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.setdefault(row["speaker"], []).append(row)
    print("\n配音員\t參考 F0\t通過/警告/失敗\t比值範圍")
    for speaker, items in rows.items():
        counts = {"通過": 0, "警告": 0, "失敗": 0}
        ratios = []
        for item in items:
            counts[item["status"]] += 1
            if item["ratio"] is not None:
                ratios.append(item["ratio"])
        span = f"{min(ratios):.2f}–{max(ratios):.2f}" if ratios else "-"
        mark = "  ←" if counts["失敗"] else ""
        print(f"{speaker}\t{items[0]['ref_f0']:.0f} Hz\t{counts['通過']}/{counts['警告']}/{counts['失敗']}\t{span}{mark}")


def parse_args():
    parser = argparse.ArgumentParser(description="上線前檢查預設配音員合成的聲音是否穩定")
    parser.add_argument("--speakers", help="逗號分隔的配音員名稱（預設 refs.yaml 全部）")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"每位配音員合成幾句，每 5 句有 {SHORT_PER_FIVE} 句短句（預設 {DEFAULT_SAMPLES}）")
    parser.add_argument("--audio-dir", help="合成的音檔與 results.jsonl 留在這裡，供人工試聽（建議加）")
    parser.add_argument("--refs-rev", help="refs.yaml 與參考音檔改讀這個 git commit（受測主機部署的版本和工作目錄不同時用）")
    return parser.parse_args()


def main():
    args = parse_args()
    refs = yaml.safe_load(read_repo_file(REFS_PATH, args.refs_rev).decode("utf-8"))
    examples = yaml.safe_load(read_repo_file(EXAMPLES_PATH, args.refs_rev).decode("utf-8"))
    if args.refs_rev:
        # 人工試聽的紀錄是針對參考音檔，舊版本用的是同一個音檔就沿用工作目錄的紀錄
        current = yaml.safe_load(read_repo_file(REFS_PATH, None).decode("utf-8"))
        for speaker, ref in refs.items():
            now = current.get(speaker) or {}
            if now.get("wav") != ref.get("wav"):
                continue
            for key in ["gender_confirmed", "gender_unstable", "gender_checked"]:
                if key in now:
                    ref[key] = now[key]
    with open(SENTENCES_PATH, encoding="utf-8") as f:
        sentences = yaml.safe_load(f)
    if args.speakers:
        speakers = []
        for speaker in args.speakers.split(","):
            speaker = speaker.strip()
            if speaker not in refs:
                raise SystemExit(f"refs.yaml 沒有配音員「{speaker}」")
            speakers.append(speaker)
    else:
        speakers = list(refs)
    print(f"TTS_URL = {TTS_URL}")
    print(f"refs.yaml：{args.refs_rev or '工作目錄'}")
    print(f"檢查 {len(speakers)} 位配音員，每位 {args.samples} 句")

    failures = []
    warnings = []
    with tempfile.TemporaryDirectory(prefix="tts-speakers-") as download_dir:
        def ref_wav(speaker):
            wav = Path(download_dir) / "refs" / Path(refs[speaker]["wav"]).name
            if not wav.exists():
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav.write_bytes(read_repo_file((Path("tts") / refs[speaker]["wav"]).as_posix(), args.refs_rev))
            return wav

        try:
            client = connect(TTS_URL, download_dir)
        except Exception as e:
            print(f"✗ 連不上 TTS：{TTS_URL}（{e}）")
            return 1
        run(client, speakers, refs, examples, sentences, ref_wav, args.samples, failures, warnings, args.audio_dir)

    if args.audio_dir:
        summarize(args.audio_dir)
    return report_result(failures, warnings)


if __name__ == "__main__":
    sys.exit(main())
