"""用 tts 服務合成試聽句：目前的預設配音員，加上指定的候選參考音檔。

    python synthesize.py --work WORK --speaker 雅美_女聲 --refs dict-A541...,dict-584D...
    BASE_URL=https://tshi5v100.ithuankhoki.tw python synthesize.py ...

每組合成 WORK/texts.json 的每一句，存成 WORK/out/<組>/NN.wav，並寫 WORK/jobs.json。
已存在的音檔會跳過，中斷後重跑會接著做。
服務掛掉或連不上時改用 local_synthesize.py（本機 CPU，同一份 jobs.json）。
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
from gradio_client import Client, handle_file

from common import REFS_PATH, language_of, load_json, save_json

APP_PATH = "hnang-kari-ai-asi-sluhay"
DEFAULT_API = "/default_speaker_tts"
CUSTOM_API = "/custom_speaker_tts"
SPEAKER_RADIO_LABEL = "步驟二：選擇配音員"
RETRIES = 3


def build_jobs(work, speaker, ref_ids):
    import yaml

    with open(REFS_PATH, encoding="utf-8") as f:
        refs = yaml.safe_load(f)
    if speaker not in refs:
        raise SystemExit(f"refs.yaml 沒有配音員「{speaker}」")
    candidates = {}
    for c in load_json(work / "candidates.json"):
        candidates[c["id"]] = c

    jobs = [{
        "group": "current",
        "kind": "current",
        "speaker": speaker,
        "language": language_of(speaker),
        "ref_wav": str(REFS_PATH.parents[1] / refs[speaker]["wav"]),
        "ref_text": refs[speaker]["text"],
        "ref_zh": "",
        "source": refs[speaker].get("source", {"來源": "目前 refs.yaml"}),
    }]
    for ref_id in ref_ids:
        if ref_id not in candidates:
            raise SystemExit(f"candidates.json 沒有 {ref_id}")
        c = candidates[ref_id]
        jobs.append({
            "group": ref_id,
            "kind": "candidate",
            "speaker": speaker,
            "language": language_of(speaker),
            "ref_wav": c["wav"],
            "ref_text": c["text"],
            "ref_zh": c["zh"],
            "source": c["source"],
        })
    save_json(work / "jobs.json", jobs)
    return jobs


def find_endpoints(client):
    """兩個「切換族別」事件都沒有 api_name，用參數和回傳元件分辨。"""
    endpoints = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
    default_switch, custom_switch = None, None
    for name, info in endpoints.items():
        params = []
        for p in info["parameters"]:
            params.append(p["parameter_name"])
        if params != ["ethnicity"]:
            continue
        labels = []
        for r in info["returns"]:
            labels.append(r.get("label"))
        if SPEAKER_RADIO_LABEL in labels:
            default_switch = name
        else:
            custom_switch = name
    if not default_switch or not custom_switch:
        raise SystemExit(f"找不到切換族別的端點：{list(endpoints)}")
    return default_switch, custom_switch


def predict_with_retry(fn):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  失敗第 {attempt} 次：{type(e).__name__}: {e}", file=sys.stderr)
            if attempt == RETRIES:
                raise
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--refs", default="", help="候選 id，逗號分隔（candidates.json 的 id）")
    args = parser.parse_args()

    work = Path(args.work)
    ref_ids = []
    for ref_id in args.refs.split(","):
        if ref_id.strip():
            ref_ids.append(ref_id.strip())
    jobs = build_jobs(work, args.speaker, ref_ids)
    texts = load_json(work / "texts.json")

    base_url = os.environ.get("BASE_URL", "https://ai-labs.ilrdf.org.tw").rstrip("/")
    client = Client(f"{base_url}/{APP_PATH}/", verbose=False, httpx_kwargs={"timeout": httpx.Timeout(120)})
    default_switch, custom_switch = find_endpoints(client)
    ethnicity = args.speaker.split("_")[0]

    for job in jobs:
        # 每組前都切換族別，同一個 session 的 Radio choices 才會包含這個配音員或語別
        if job["kind"] == "current":
            client.predict(ethnicity, api_name=default_switch)
        else:
            client.predict(ethnicity, api_name=custom_switch)
        for i, (text, _zh) in enumerate(texts, start=1):
            out = work / "out" / job["group"] / f"{i:02d}.wav"
            if out.exists():
                continue
            if job["kind"] == "current":
                result = predict_with_retry(lambda: client.predict(job["speaker"], text, api_name=DEFAULT_API))
            else:
                result = predict_with_retry(lambda: client.predict(
                    job["language"], job["ref_text"], handle_file(job["ref_wav"]), text, api_name=CUSTOM_API))
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(result, out)
            print(out, flush=True)


if __name__ == "__main__":
    main()
