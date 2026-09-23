"""共用的檢查執行、輸出、結果記錄與連線。

`check()` 由所有檢查腳本共用：印出結果、把失敗收進 failures、
同時把每一項記到 RESULTS，供 run_all.py 的 --report 寫成 JSON。
"""
import time

import httpx
from gradio_client import Client

RESULTS = []

# gradio_client 預設的 httpx timeout 只有 5 秒，比 tts 合成（穩態約 14 秒）還短，
# 會在受測主機稍慢時就 ReadTimeout。放寬到 60 秒，服務真的卡住時仍會失敗。
HTTPX_TIMEOUT = 60


def connect(url, download_dir):
    return Client(
        url,
        verbose=False,
        analytics_enabled=False,
        download_files=download_dir,
        httpx_kwargs={"timeout": httpx.Timeout(HTTPX_TIMEOUT)},
    )


def check(name, fn, failures, service=None):
    """執行一項檢查。通過回傳耗時秒數，失敗回傳 None。"""
    start = time.time()
    try:
        note = fn()
    except Exception as e:
        seconds = time.time() - start
        message = f"{type(e).__name__}: {e}"
        failures.append((name, message))
        RESULTS.append({
            "service": service,
            "name": name,
            "ok": False,
            "seconds": round(seconds, 2),
            "note": message,
        })
        print(f"  ✗ {name}（{seconds:.1f}s）{message}", flush=True)
        return None

    seconds = time.time() - start
    RESULTS.append({
        "service": service,
        "name": name,
        "ok": True,
        "seconds": round(seconds, 2),
        "note": note or "",
    })
    print(f"  ✓ {name}（{seconds:.1f}s）{note or ''}", flush=True)
    return seconds


def report_result(failures, warnings=()):
    """印出結尾彙總，回傳 exit code。"""
    for warning in warnings:
        print(f"\n⚠ {warning}")

    if failures:
        print(f"\n失敗 {len(failures)} 項：")
        for name, message in failures:
            print(f"  - {name}：{message}")
        return 1

    print("\n全部通過")
    return 0
