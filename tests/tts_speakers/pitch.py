"""用基頻（F0）判斷語音是男聲還是女聲。

用 librosa 的 pYIN 估 F0，取有聲音框的中位數。
2026-10-15 比較過只用標準函式庫的 YIN：合成語音兩者一致，但真人錄音（參考音檔）
與部分沙啞的男聲合成結果，YIN 會差到一個八度，所以改用 pYIN。
"""
import librosa
import numpy as np

SAMPLE_RATE = 16000
F0_MIN = 60
F0_MAX = 450

# 成人女聲 F0 多在 165–255 Hz，男聲 85–155 Hz，中間留緩衝
FEMALE_MIN_F0 = 175
MALE_MAX_F0 = 150


def median_f0(path):
    """回傳 (F0 中位數或 None, 有聲音框比例)。"""
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    f0, voiced, _prob = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=1024)
    if not np.any(voiced):
        return None, 0.0
    return float(np.median(f0[voiced])), float(np.mean(voiced))


def gender_of(f0):
    """回傳「女」、「男」或「?」。"""
    if f0 is None:
        return "?"
    if f0 >= FEMALE_MIN_F0:
        return "女"
    if f0 <= MALE_MAX_F0:
        return "男"
    return "?"
