"""檢查 tts（F5-TTS）合成的音檔格式與有沒有聲音。

tts 每次合成的音檔位元都不同（2026-09-18 實測，同一句兩次長度相同但內容不同），
所以只驗格式、長度與非靜音，不比對內容。
"""
import array
import math
import wave
from pathlib import Path

from reporting import check
from timing import check_threshold, warm_up

SERVICE = "tts"
APP_PATH = "hnang-kari-ai-asi-sluhay"
API_NAME = "/synthesize"
AMIS_LANGUAGE = "阿美_海岸"
TEXT = "sosowalen ako itiya:ayho a ʼorip niyam."

EXPECTED_SAMPLE_RATE = 24000
MIN_SECONDS = 0.5
# 2026-09-18 測試機實測 RMS/滿刻度約 0.11，門檻取 0.01 有十倍餘裕
MIN_RMS_RATIO = 0.01
SAMPLE_WIDTH_CODE = {1: "b", 2: "h", 4: "i"}


def rms_ratio(path):
    """回傳 (RMS / 滿刻度, 秒數, 取樣率)。只用標準函式庫。"""
    with wave.open(str(path)) as w:
        frames, width, rate = w.getnframes(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(frames)

    if width not in SAMPLE_WIDTH_CODE:
        raise AssertionError(f"沒處理過的取樣寬度 {width} bytes")
    samples = array.array(SAMPLE_WIDTH_CODE[width])
    samples.frombytes(raw)
    if not samples:
        raise AssertionError("音檔沒有任何取樣點")

    rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples))
    return rms / float(2 ** (8 * width - 1)), frames / rate, rate


def synthesize(client, language, text):
    return client.predict(language, text, api_name=API_NAME)


def check_audio(client):
    def run():
        path = Path(synthesize(client, AMIS_LANGUAGE, TEXT))
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"沒有拿到音檔：{path}")

        try:
            ratio, seconds, rate = rms_ratio(path)
        finally:
            path.unlink(missing_ok=True)

        if rate != EXPECTED_SAMPLE_RATE:
            raise AssertionError(f"取樣率是 {rate}，應為 {EXPECTED_SAMPLE_RATE}")
        if seconds <= MIN_SECONDS:
            raise AssertionError(f"音檔只有 {seconds:.2f}s，應長於 {MIN_SECONDS}s")
        if ratio < MIN_RMS_RATIO:
            raise AssertionError(f"音檔像是靜音：RMS/滿刻度 {ratio:.4f} 低於門檻 {MIN_RMS_RATIO}")
        return f"{seconds:.2f}s、{rate} Hz、RMS {ratio:.3f}"
    return run


def run(client, failures, warnings):
    print("\n[tts] 合成音檔格式與非靜音")
    warm_up(SERVICE, lambda: synthesize(client, AMIS_LANGUAGE, TEXT), warnings)

    seconds = check(f"{AMIS_LANGUAGE} 音檔格式", check_audio(client), failures, service=SERVICE)
    if seconds is not None:
        check_threshold(SERVICE, seconds, warnings)
