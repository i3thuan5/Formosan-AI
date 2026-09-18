"""上線後檢查：四個模型與頁面。

    $ BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests
    $ BASE_URL=... tox -e production_tests -- --all-languages --report out.json
    $ BASE_URL=... python tests/production_tests/run_all.py --services asr,tts

只驗「服務活著、格式對、有內容」，不比對辨識或翻譯的文字，也不比對音檔內容：
模型會換版，比對內容會讓每次升級都要改測試。
全部通過 exit 0，任一項失敗 exit 1。說明見 README.md。
"""
import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_asr  # noqa: E402
import check_asr_kaldi  # noqa: E402
import check_mt  # noqa: E402
import check_mt_tts_playback as playback  # noqa: E402
import check_pages  # noqa: E402
import check_tts  # noqa: E402
import reporting  # noqa: E402
from reporting import connect, report_result  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "https://ai-labs.ilrdf.org.tw").rstrip("/")
ALL_SERVICES = ["pages", "asr", "asr-kaldi", "mt", "tts"]
SAMPLE_COUNT = 3


def sample_languages(all_items, required, all_languages):
    """固定含素材語別，再隨機抽幾個；--all-languages 時回傳全部。"""
    if all_languages:
        return list(all_items)
    others = [item for item in all_items if item != required]
    count = min(SAMPLE_COUNT, len(others))
    return [required] + random.SystemRandom().sample(others, count)


def with_client(service, app_path, download_dir, body, failures):
    """連線失敗時把整個服務記為失敗，其餘服務照跑。"""
    try:
        client = connect(f"{BASE_URL}/{app_path}/", download_dir)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        failures.append((f"連線 {service}", message))
        reporting.RESULTS.append({
            "service": service,
            "name": f"連線 {service}",
            "ok": False,
            "seconds": 0.0,
            "note": message,
        })
        print(f"\n[{service}] ✗ 連不上 {BASE_URL}/{app_path}/：{message}")
        return
    body(client)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return None


def write_report(path, warnings, failures):
    report = {
        "base_url": BASE_URL,
        "commit": git_commit(),
        "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": reporting.RESULTS,
        "warnings": list(warnings),
        "failed": len(failures),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n報告寫到 {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="上線後檢查四個模型與頁面")
    parser.add_argument(
        "--services",
        default=",".join(ALL_SERVICES),
        help=f"要檢查的服務，逗號分隔。可選：{'、'.join(ALL_SERVICES)}（預設全部）",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help=f"每個服務掃全部語別（預設是素材語別加隨機 {SAMPLE_COUNT} 個）",
    )
    parser.add_argument("--report", help="把結果寫成 JSON 到這個路徑")
    args = parser.parse_args()

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    unknown = [s for s in services if s not in ALL_SERVICES]
    if unknown:
        parser.error(f"不認識的服務：{'、'.join(unknown)}。可選：{'、'.join(ALL_SERVICES)}")
    args.services = services
    return args


def main():
    args = parse_args()
    print(f"BASE_URL = {BASE_URL}")
    print(f"檢查服務：{'、'.join(args.services)}")

    failures = []
    warnings = []

    with tempfile.TemporaryDirectory(prefix="production-tests-") as download_dir:
        if "pages" in args.services:
            check_pages.run(BASE_URL, reporting.check, failures)

        if "asr" in args.services:
            languages = sample_languages(
                check_asr.all_languages(), check_asr.AMIS_LANGUAGE, args.all_languages)
            with_client(
                "asr", check_asr.APP_PATH, download_dir,
                lambda client: check_asr.run(client, languages, failures, warnings),
                failures)

        if "asr-kaldi" in args.services:
            dialects = sample_languages(
                check_asr_kaldi.all_dialects(), check_asr_kaldi.AMIS_DIALECT, args.all_languages)
            with_client(
                "asr-kaldi", check_asr_kaldi.APP_PATH, download_dir,
                lambda client: check_asr_kaldi.run(client, dialects, failures, warnings),
                failures)

        if "mt" in args.services:
            languages = sample_languages(
                check_mt.all_languages(), check_mt.AMIS_LANGUAGE, args.all_languages)
            with_client(
                "mt", check_mt.APP_PATH, download_dir,
                lambda client: run_mt(client, languages, failures, warnings),
                failures)

        if "tts" in args.services:
            tts_languages = sample_languages(
                check_mt.all_languages(), check_mt.AMIS_LANGUAGE, args.all_languages)
            with_client(
                "tts", check_tts.APP_PATH, download_dir,
                lambda client: run_tts(client, tts_languages, args.all_languages, failures, warnings),
                failures)

    exit_code = report_result(failures, warnings)
    if args.report:
        write_report(args.report, warnings, failures)
    return exit_code


def run_mt(client, languages, failures, warnings):
    check_mt.run(client, languages, failures, warnings)
    # 既有的 mt → tts 端到端檢查，走「mt 容器 → 公開網域 → tts」這條路
    playback.run_end_to_end_checks(client, playback.load_refs(), failures)


def run_tts(client, languages, all_languages, failures, warnings):
    check_tts.run(client, failures, warnings)
    # 既有的 TTS API 約定、語別合成與已知 bug 回歸
    playback.run_tts_checks(client, playback.load_refs(), languages, all_languages, failures)


if __name__ == "__main__":
    sys.exit(main())
