"""暖機、計時與耗時門檻。

拿掉輸出內容比對之後，耗時是唯一能發現「GPU 沒被用到」的訊號：
CPU fallback 會讓 asr 從 1 秒變數十秒，門檻一定會被超過。

門檻以 2026-09-18 對測試機 tshi5v100 量到的穩態耗時放大約 10 倍，
只印警告不算失敗，避免受測主機忙碌時誤報。
"""
import time

# 服務 -> (門檻秒數, 2026-09-18 測試機穩態耗時)
THRESHOLDS = {
    "asr": (15, 1.1),
    "asr-kaldi": (30, 4.2),
    "mt": (10, 1.3),
    "tts": (20, 14.2),
}


def timed(fn):
    """執行 fn，回傳 (結果, 耗時秒數)。"""
    start = time.time()
    result = fn()
    return result, time.time() - start


def warm_up(service, fn, warnings):
    """第一次呼叫含模型載入與 cuDNN 自動調校，比穩態慢 5 到 8 倍，不計時。"""
    try:
        _, seconds = timed(fn)
        print(f"  暖機 {service}（{seconds:.1f}s，不計入）", flush=True)
    except Exception as e:
        warnings.append(f"{service} 暖機失敗：{type(e).__name__}: {e}")
        print(f"  暖機 {service} 失敗（後續檢查仍會執行）：{type(e).__name__}: {e}", flush=True)


def check_threshold(service, seconds, warnings):
    """暖機後的耗時超過門檻時記警告，不算失敗。"""
    threshold, baseline = THRESHOLDS[service]
    if seconds <= threshold:
        return None
    message = (
        f"{service} 暖機後耗時 {seconds:.1f}s 超過門檻 {threshold}s"
        f"（2026-09-18 測試機為 {baseline}s），請確認 GPU 是否真的有被使用"
    )
    warnings.append(message)
    print(f"  ⚠ {message}", flush=True)
    return message
