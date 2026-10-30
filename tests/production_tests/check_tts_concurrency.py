"""驗證 tts 同時合成時有沒有保護好。

F5-TTS 的 DiT 把 text embedding 快取在 model 上，正常跑完才清。Gradio 的
concurrency_limit 是每個 event listener 各自算，所以 /synthesize（mt 呼叫）和
/default_speaker_tts（網頁）會同時執行 sample()，互相用到對方的快取，出現
"Sizes of tensors must match"。兩個都中途失敗時快取沒人清，之後每個 request 都失敗，
直到重啟 tts（2026-10-28 正式機 log 有發生）。

分三個階段：

1. 分別 request：兩個端點輪流打，確認受測主機本來正常。
2. 同時 request：兩個端點同時打，重複幾輪。
3. 分別 request：再輪流打，確認快取沒有卡住。

有保護時三個階段都成功；沒保護時階段 2 會有 tts 端例外，快取卡住時階段 3 也會。
**沒保護的版本卡住後要重啟 tts 容器才會恢復**，所以預設不對正式站執行。

run_all.py 的 tts 服務會跑這項，也可以單獨執行：

    $ BASE_URL=https://tshi5v100.ithuankhoki.tw python tests/production_tests/check_tts_concurrency.py
    $ BASE_URL=... python tests/production_tests/check_tts_concurrency.py --rounds 10

全部通過 exit 0，失敗 exit 1。
"""
import argparse
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reporting import check, connect, report_result  # noqa: E402

SERVICE = "tts"
APP_PATH = "hnang-kari-ai-asi-sluhay"
PRODUCTION_HOST = "ai-labs.ilrdf.org.tw"
DEFAULT_ROUNDS = 5

# 兩個端點用同一位配音員，前處理時間相近，才會幾乎同時進到 sample()。
# 兩句長度要不同：合成長度相同時，用到對方的快取也不會出錯。
ENDPOINTS = [
    ("/synthesize", "阿美_海岸", "sosowalen ako itiya:ayho a ʼorip niyam."),
    ("/default_speaker_tts", "阿美_海岸_男聲", "Itiya:ay ho a ʼorip niyam."),
]


def call(client, api_name, first, text, origin):
    """打一次 API，回傳結果與相對 origin 的送出、完成時間。"""
    start = time.time()
    try:
        path = Path(client.predict(first, text, api_name=api_name))
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"沒有拿到音檔：{path}")
        path.unlink(missing_ok=True)
        ok, note = True, ""
    except Exception as e:
        ok, note = False, f"{type(e).__name__}: {e}"
        error_type = type(e).__name__
    else:
        error_type = ""
    return {
        "api": api_name,
        "ok": ok,
        "error_type": error_type,
        "start": start - origin,
        "end": time.time() - origin,
        "note": note,
    }


def show(result):
    mark = "✓" if result["ok"] else "✗"
    seconds = result["end"] - result["start"]
    print(
        f"      {mark} {result['api']:<22} 第 {result['start']:5.1f}s 送出，"
        f"第 {result['end']:5.1f}s 完成（{seconds:.1f}s）{result['note']}",
        flush=True,
    )


def run_sequential(client):
    results = []
    for api_name, first, text in ENDPOINTS:
        result = call(client, api_name, first, text, time.time())
        show(result)
        results.append(result)
    return results


def run_together(client, rounds):
    results = []
    for round_index in range(rounds):
        print(f"    第 {round_index + 1}/{rounds} 輪", flush=True)
        barrier = threading.Barrier(len(ENDPOINTS))
        origin = time.time()

        def call_together(api_name, first, text):
            # 等所有 thread 都準備好才一起送，盡量讓 request 同時進到 sample()
            barrier.wait()
            return call(client, api_name, first, text, origin)

        with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
            futures = []
            for api_name, first, text in ENDPOINTS:
                futures.append(pool.submit(call_together, api_name, first, text))
            for future in futures:
                result = future.result()
                show(result)
                results.append(result)
    return results


def count_failures(results):
    failures = 0
    for result in results:
        if not result["ok"]:
            failures += 1
    return failures


def count_app_errors(results):
    # AppError 是 tts 端出例外（快取錯亂就是這種）；其他如 ReadTimeout 是 client 等不到回應
    errors = 0
    for result in results:
        if result["error_type"] == "AppError":
            errors += 1
    return errors


def summary(results):
    return f"{len(results) - count_failures(results)}/{len(results)}"


def check_protection(client, rounds, warnings):
    def run():
        print("    階段 1：分別 request", flush=True)
        first = run_sequential(client)
        if count_failures(first) > 0:
            raise AssertionError(
                f"階段 1 分別 request 只有 {summary(first)} 成功，受測主機本來就不正常。"
                "沒保護的版本可能已經卡住，請先重啟 tts 容器再測"
            )

        print(f"    階段 2：同時 request，{rounds} 輪", flush=True)
        together = run_together(client, rounds)
        print("    階段 3：分別 request", flush=True)
        after = run_sequential(client)

        note = f"階段 1 {summary(first)}、階段 2 {summary(together)}、階段 3 {summary(after)}"
        if count_app_errors(after) > 0:
            raise AssertionError(f"沒保護，快取卡住了，要重啟 tts 容器才會恢復（{note}）")
        if count_app_errors(together) > 0:
            raise AssertionError(f"沒保護，同時合成出現 tts 端例外（{note}）")
        if count_failures(together) > 0 or count_failures(after) > 0:
            # nginx 會把 SSE 整段緩衝到連線結束，連線開超過 timeout 時 client 收不到任何資料，
            # 2026-10-30 試驗機壓力測試約百分之一的 request 會這樣，tts 本身正常
            warnings.append(
                f"tts 同時合成保護：沒有 tts 端例外，但有 request 沒拿到結果（{note}）。"
                "可能是 nginx 緩衝 SSE 造成 client 逾時，請對照 tts 的 log"
            )
        return note
    return run


def run(client, base_url, failures, warnings, rounds=DEFAULT_ROUNDS, allow_production=False):
    print("\n[tts] 同時合成保護")
    if PRODUCTION_HOST in base_url and not allow_production:
        message = (
            f"tts 同時合成保護：正式站 {base_url} 預設不跑，沒保護的版本會卡住到重啟。"
            "確定已部署新版請加 --allow-production"
        )
        warnings.append(message)
        print(f"  略過：{message}", flush=True)
        return
    check(
        f"兩個端點同時合成（{rounds} 輪）",
        check_protection(client, rounds, warnings),
        failures,
        service=SERVICE,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="驗證 tts 同時合成時有沒有保護好")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help=f"同時 request 重複幾輪（預設 {DEFAULT_ROUNDS}）")
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help=f"允許對正式站 {PRODUCTION_HOST} 執行。沒保護的版本會卡住到重啟，請確定已部署新版",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    if not base_url:
        print("請用 BASE_URL 指定受測主機，例如 BASE_URL=https://tshi5v100.ithuankhoki.tw")
        return 1

    url = f"{base_url}/{APP_PATH}/"
    print(f"受測主機 {url}")
    failures = []
    warnings = []
    with tempfile.TemporaryDirectory(prefix="tts-concurrency-") as download_dir:
        client = connect(url, download_dir)
        run(client, base_url, failures, warnings, args.rounds, args.allow_production)
    return report_result(failures, warnings)


if __name__ == "__main__":
    sys.exit(main())
