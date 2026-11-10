"""產生人工試聽網頁（WORK/site/），可用 Artifact 發布。

兩種輸入：
    python build_site.py --work WORK
        換參考音檔：讀 WORK/jobs.json、texts.json、out/，一個配音員一個頁籤
    python build_site.py --work WORK1 --work WORK2 ... --out DIR
        同時換好幾位配音員：每個 WORK 一個頁籤，網頁輸出到 DIR/site
    python build_site.py --work WORK --speaker-results DIR
        上線前性別檢查：讀 tests/tts_speakers --audio-dir 留下的 DIR/results.jsonl，
        一個族一個頁籤、一位配音員一組

每組（一位配音員或一個參考音檔）的 mp3 打包成 site/audio/<組>.json（data URI），
網頁捲到該組附近才下載，檔案數也不會超過 Artifact 的上限。
最後印出 Artifact 發布要用的 files 清單（JSON）。
"""
import argparse
import base64
import html
import json
import subprocess
from pathlib import Path

import yaml

from common import REFS_PATH, REPO_ROOT, SKILL_DIR, TTS_DIR, analyze, expected_gender, load_json, save_json, to_mp3

# common 會把 tests/tts_speakers 加進 sys.path
from check_tts_speakers import ratio_status  # noqa: E402

METHOD_REPLACE = (
    "每個參考音檔各合成同樣的句子，每句的基頻（F0 中位數）和該參考音檔的 F0 比："
    "低於 0.70 倍或高於 1.40 倍算失敗（聲音變了），0.70–0.80 倍或 1.25–1.40 倍算警告。"
    "性別欄是 F0 的粗略判斷（女 ≥ 175 Hz、男 ≤ 150 Hz）。F0 只是輔助，請以耳朵為準。"
    "F5-TTS 每次取樣都不同，同一句再合成一次性別也可能不同。"
)
METHOD_SPEAKERS = (
    "用 tests/tts_speakers 對每位預設配音員合成 10 句（6 句短句、4 句長句），"
    "每句的基頻（F0 中位數）和該配音員參考音檔的 F0 比：低於 0.70 倍或高於 1.40 倍算失敗（聲音變了），"
    "0.70–0.80 倍或 1.25–1.40 倍算警告。性別欄是 F0 的粗略判斷（女 ≥ 175 Hz、男 ≤ 150 Hz）。"
    "列最左邊有紅條的是失敗的句子，請優先試聽。F0 只是輔助，請以耳朵為準。"
)


def source_label(source):
    if not source:
        return ""
    if "課" in source:
        return f"族語E樂園 生活會話篇 第{source['課']}課第{source['句']}句（{source.get('對話角色', '')}）"
    if "詞條" in source:
        return f"族語辭典 例句（詞條 {source['詞條']}）"
    return source.get("來源", "")


def ref_entry(site, rel, wav, text, zh, source):
    to_mp3(wav, site / rel)
    return {
        "text": text,
        "zh": zh,
        "audio": rel,
        "source": source_label(source),
        "url": (source or {}).get("音檔"),
        **analyze(wav),
    }


def build_replace(work, site):
    jobs = load_json(work / "jobs.json")
    texts = load_json(work / "texts.json")
    speaker = jobs[0]["speaker"]
    groups = []
    for job in jobs:
        slug = job["group"]
        ref = ref_entry(site, f"{slug}/ref.mp3", job["ref_wav"], job["ref_text"], job["ref_zh"], job["source"])
        items = []
        for i, (text, zh) in enumerate(texts, start=1):
            wav = work / "out" / slug / f"{i:02d}.wav"
            if not wav.exists():
                continue
            rel = f"{slug}/{i:02d}.mp3"
            to_mp3(wav, site / rel)
            item = {"n": i, "text": text, "zh": zh, "audio": rel, **analyze(wav)}
            # 和 tests/tts_speakers 相同：和參考音檔的 F0 比
            if item["f0"] is not None and ref["f0"] is not None:
                item["ratio"] = round(item["f0"] / ref["f0"], 2)
                item["status"] = ratio_status(item["ratio"])
            items.append(item)
        label = "目前預設配音員" if job["kind"] == "current" else job.get("label", slug)
        groups.append({
            "id": slug,
            "kind": job["kind"],
            "label": label,
            "ref": ref,
            "items": items,
        })
    return {
        "title": f"{speaker} 參考音檔試聽",
        "intro": f"目前的預設配音員和 {len(jobs) - 1} 個候選參考音檔，各合成同樣的 {len(texts)} 句。"
                 "請試聽後在每句右側標記男女，挑出性別穩定、音質好的參考音檔。",
        "expected": expected_gender(speaker),
        "method": METHOD_REPLACE,
        "tabs": [{"name": speaker, "groups": groups}],
    }


def git_show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def load_refs(rev, work):
    """回傳 (refs, 參考音檔路徑函式)。rev 有給時讀那個 commit 的 refs.yaml 與音檔（受測主機還沒部署新版時用）。"""
    if not rev:
        with open(REFS_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f), lambda wav: TTS_DIR / wav
    refs = yaml.safe_load(git_show(rev, "tts/configs/refs.yaml").decode("utf-8"))

    def wav_at_rev(wav):
        rel = Path("tts") / Path(wav)
        dst = work / "refs-at-rev" / rel.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(git_show(rev, rel.as_posix()))
        return dst
    return refs, wav_at_rev


PRIORITY_TABS = ["失敗", "參考音檔像另一性別", "警告", "全部通過"]


def priority_tab(items, ref, expected, confirmed):
    """沒有指定頁籤時，依「要不要先聽」分：失敗 → 參考音檔像另一性別（沒人工確認過）→ 警告 → 全部通過。"""
    statuses = set()
    for item in items:
        statuses.add(item.get("status"))
    if "失敗" in statuses:
        return "失敗"
    if ref["gender"] not in (expected, "?") and confirmed != expected:
        return "參考音檔像另一性別"
    if "警告" in statuses:
        return "警告"
    return "全部通過"


def build_speakers(results_dir, site, work, refs_rev):
    refs, wav_path = load_refs(refs_rev, work)
    with open(REFS_PATH, encoding="utf-8") as f:
        current_refs = yaml.safe_load(f)
    # 一列可以帶 group（例如「同一句合成 10 次」的實驗），沒有就以配音員分組；
    # 也可以帶 tab（頁籤名稱）與 ref_wav、ref_text（不是 refs.yaml 裡的參考音檔時）
    by_group = {}
    with open(results_dir / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_group.setdefault(row.get("group", row["speaker"]), []).append(row)

    tabs = {}
    for index, (label, rows) in enumerate(by_group.items(), start=1):
        speaker = rows[0]["speaker"]
        slug = f"speaker{index:02d}"
        items = []
        for row in sorted(rows, key=lambda r: r["n"]):
            rel = f"{slug}/{row['n']:02d}.mp3"
            to_mp3(results_dir / row["audio"], site / rel)
            items.append({"n": row["n"], "text": row["text"], "zh": "", "audio": rel, "f0": row["f0"],
                          "gender": row["gender"], "ratio": row.get("ratio"), "status": row.get("status"),
                          "length": row.get("length")})
        if rows[0].get("ref_wav"):
            ref = {"wav": rows[0]["ref_wav"], "text": rows[0]["ref_text"], "source": rows[0].get("ref_source")}
            ref_path = Path(ref["wav"])
        else:
            ref = refs[speaker]
            ref_path = wav_path(ref["wav"])
        entry = ref_entry(site, f"{slug}/ref.mp3", ref_path, ref["text"], "", ref.get("source"))
        # 人工確認看工作目錄的 refs.yaml（同一個參考音檔才算）
        now = current_refs.get(speaker) or {}
        confirmed = (now.get("gender_confirmed") or {}).get("gender") if now.get("wav") == ref.get("wav") else None
        tab_name = rows[0].get("tab") or priority_tab(items, entry, expected_gender(speaker), confirmed)
        tabs.setdefault(tab_name, {"name": tab_name, "groups": []})["groups"].append({
            "id": slug,
            "kind": "speaker",
            "label": label,
            "expected": expected_gender(speaker),
            "ref": entry,
            "items": items,
        })
    # 指定的頁籤照出現順序放前面，再依 PRIORITY_TABS；頁籤名稱加上人數
    ordered = []
    for name, tab in tabs.items():
        if name not in PRIORITY_TABS:
            ordered.append(tab)
    for name in PRIORITY_TABS:
        if name in tabs:
            tabs[name]["name"] = f"{name}（{len(tabs[name]['groups'])} 位）"
            ordered.append(tabs[name])
    return {
        "title": "預設配音員聲音穩定度",
        "intro": "每位預設配音員合成多句，檢查聲音是否穩定維持參考音檔的聲音、和名稱標的男聲／女聲是否一致。",
        "method": METHOD_SPEAKERS,
        "tabs": ordered,
    }


def pack_audio(data, mp3_dir, site):
    """把每組的 mp3 打包成一個 JSON；data 裡的 audio 改成組內的 key。"""
    (site / "audio").mkdir(parents=True, exist_ok=True)
    for tab in data["tabs"]:
        for group in tab["groups"]:
            entries = [group["ref"]] + group["items"]
            pack = {}
            for entry in entries:
                key = Path(entry["audio"]).stem
                encoded = base64.b64encode((mp3_dir / entry["audio"]).read_bytes()).decode("ascii")
                pack[key] = "data:audio/mpeg;base64," + encoded
                entry["audio"] = key
            group["pack"] = f"audio/{group['id']}.json"
            with open(site / group["pack"], "w", encoding="utf-8") as f:
                json.dump(pack, f)


def build_replace_many(works):
    """好幾位配音員各自的 WORK 合成一個網頁；音檔路徑改成絕對路徑，組 id 加上序號避免重複。"""
    tabs = []
    expected = set()
    for index, work in enumerate(works, start=1):
        mp3_dir = work / "mp3"
        data = build_replace(work, mp3_dir)
        expected.add(data["expected"])
        tab = data["tabs"][0]
        for group in tab["groups"]:
            group["id"] = f"w{index}-{group['id']}"
            group["expected"] = data["expected"]
            group["ref"]["audio"] = str(mp3_dir / group["ref"]["audio"])
            for item in group["items"]:
                item["audio"] = str(mp3_dir / item["audio"])
        tabs.append(tab)
    return {
        "title": "配音員參考音檔替換試聽",
        "intro": f"{len(works)} 位配音員各挑幾個候選參考音檔，各合成同樣的句子。"
                 "請試聽後在每句右側標記男女，只要有 1 句變聲就不能用，挑出每位性別穩定、音質好的參考音檔。",
        "method": METHOD_REPLACE,
        "tabs": tabs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, action="append", help="可以給好幾次，每個 WORK 一個頁籤")
    parser.add_argument("--out", help="給好幾個 --work 時，網頁輸出的資料夾")
    parser.add_argument("--speaker-results", help="tests/tts_speakers --audio-dir 的資料夾")
    parser.add_argument("--refs-rev", help="參考音檔改讀這個 git commit 的版本（受測主機部署的版本和工作目錄不同時用）")
    args = parser.parse_args()

    if len(args.work) > 1:
        if not args.out or args.speaker_results:
            parser.error("給好幾個 --work 時要加 --out，而且不能和 --speaker-results 一起用")
        works = []
        for w in args.work:
            works.append(Path(w))
        site = Path(args.out) / "site"
        mp3_dir = Path(args.out) / "mp3"
        data = build_replace_many(works)
    else:
        work = Path(args.work[0])
        site = work / "site"
        mp3_dir = work / "mp3"
        if args.speaker_results:
            data = build_speakers(Path(args.speaker_results), mp3_dir, work, args.refs_rev)
        else:
            data = build_replace(work, mp3_dir)
    pack_audio(data, mp3_dir, site)
    save_json(site / "data.json", data)
    page = (SKILL_DIR / "assets" / "index.html").read_text(encoding="utf-8")
    page = page.replace("{{TITLE}}", html.escape(data["title"])).replace("{{INTRO}}", html.escape(data["intro"]))
    (site / "index.html").write_text(page, encoding="utf-8")

    for tab in data["tabs"]:
        for g in tab["groups"]:
            genders = ""
            for it in g["items"]:
                genders += it["gender"]
            print(f"{tab['name']}\t{g['label']}\tref {g['ref']['f0']} Hz {g['ref']['dur']}s\t{genders}")

    files = []
    for path in sorted(site.rglob("*")):
        if path.is_file() and path.name != "index.html":
            files.append({"path": str(path.relative_to(site))})
    print(f"\n頁面：{site / 'index.html'}（root 設成 {site}）")
    print(f"共 {len(files)} 個附檔；一次發布最多 255 個、64 MB，超過要分批發布到同一個 url")
    print(json.dumps(files, ensure_ascii=False))


if __name__ == "__main__":
    main()
