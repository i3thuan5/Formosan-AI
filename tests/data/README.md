# 測試素材

[tests/production_tests/](../production_tests/) 的上線後檢查與 [tests/grafana-k6/](../grafana-k6/) 的延遲測試共用這裡的檔案。
**只放一份**，兩邊都指向這個目錄。

| 檔案 | 用途 |
| --- | --- |
| `海岸阿美語-曾玉蘭-個人生命史-短.mp4` | asr（`gr.Video`，吃影片） |
| `海岸阿美語-曾玉蘭-個人生命史-短.mp3` | asr-kaldi（`gr.Audio`，吃音檔） |

## 原始檔規格

`海岸阿美語-曾玉蘭-個人生命史-短.mp4`：

| 項目 | 值 |
| --- | --- |
| 長度 | 10.18 秒 |
| 大小 | 1,459,372 bytes |
| 影像 | H.264、1280×720 |
| 聲音 | AAC、48kHz、2 聲道 |
| 族語 | 海岸阿美語 |

## mp3 怎麼來的

用 ffmpeg 從 mp4 轉出 16kHz 單聲道（10.12 秒、81,364 bytes）：

```bash
$ cd tests/data/
$ ffmpeg -i 海岸阿美語-曾玉蘭-個人生命史-短.mp4 \
    -vn -ac 1 -ar 16000 -c:a libmp3lame -b:a 64k \
    海岸阿美語-曾玉蘭-個人生命史-短.mp3
```

16kHz 是 `asr-kaldi/app.py` 的 `gr.WaveformOptions(sample_rate=16000)` 所設定，
Vosk 模型也是吃 16kHz。
