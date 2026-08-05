# 測試規格：受測 API、測試資料、測試方法

這份文件記錄延遲測試的**規格**，也就是後續要比較時必須維持一致的部份。
怎麼執行請看 [README.md](README.md)，歷次結果請看 [RESULTS.md](RESULTS.md)。

## 一、受測的線上 API

本測試**不會**在本機啟動服務，直接打線上站台：

| 本專案資料夾 | 系統名稱 | 線上 API | Gradio endpoint |
| --- | --- | --- | --- |
| `asr/` | 族語逐字稿辨識系統（Whisper） | <https://ai-labs.ilrdf.org.tw/sapolita/?view=api> | `/generate_srt` |
| `asr-kaldi/` | 族語語音辨識系統（Kaldi／Vosk） | <https://ai-labs.ilrdf.org.tw/sapolita-kaldi/?view=api> | `/automatic_speech_recognition` |
| `tts/` | 族語語音合成系統（F5-TTS） | <https://ai-labs.ilrdf.org.tw/hnang-kari-ai-asi-sluhay/?view=api> | `/default_speaker_tts` |
| `mt/` | 族語基礎翻譯系統（NLLB-600M） | <https://ai-labs.ilrdf.org.tw/kari-seejiq-tnpusu-ai-hmjil/?view=api> | `/translate`、`/translate_1` |

## 二、測試資料

全部放在 `testing_data/`。

### 2-1 語音辨識（asr、asr-kaldi）

原始檔 `海岸阿美語-曾玉蘭-個人生命史-短.mp4`：

| 項目 | 值 |
| --- | --- |
| 長度 | 10.18 秒 |
| 大小 | 1,459,372 bytes |
| 影像 | H.264、1280×720 |
| 聲音 | AAC、48kHz、2 聲道 |

- **asr**（Whisper）吃的是 `gr.Video`，所以**直接用原始 mp4**。
- **asr-kaldi** 吃的是 `gr.Audio`，所以先用 ffmpeg 轉成 mp3（16kHz 單聲道，10.12 秒、81,364 bytes）：

  ```bash
  $ cd testing_data/
  $ ffmpeg -i 海岸阿美語-曾玉蘭-個人生命史-短.mp4 \
      -vn -ac 1 -ar 16000 -c:a libmp3lame -b:a 64k \
      海岸阿美語-曾玉蘭-個人生命史-短.mp3
  ```

  （16kHz 是 `asr-kaldi/app.py` 的 `gr.WaveformOptions(sample_rate=16000)` 所設定，
  Vosk 模型也是吃 16kHz，所以轉成 16kHz 單聲道。）

族別都選**阿美語**（`formosan_ami`）。

### 2-2 語音合成（tts）

配音員固定用**阿美_秀姑巒_女聲1**，2 句輪流合成，每句各 50 次：

1. `O sasowalen ako`
2. `Itiya:ay ho a ʼorip niyam`

### 2-3 翻譯（mt）

語別固定用**阿美_秀姑巒**（`ami_Xiug`），4 句輪流，每句各 25 次：

| # | 方向 | endpoint | 原文 |
| --- | --- | --- | --- |
| 1 | 族語 ⮕ 華語 | `/translate` | `O sasowalen ako` |
| 2 | 族語 ⮕ 華語 | `/translate` | `Itiya:ay ho a ʼorip niyam` |
| 3 | 華語 ⮕ 族語 | `/translate_1` | `我要說的是` |
| 4 | 華語 ⮕ 族語 | `/translate_1` | `我們以前的生活` |

> 這 6 句都是從上面那個 mp4 的內容來的（`o sasowalen ako` / `itiya:ay ho a ʼorip niyam`），
> 所以三個模型測的是**同一段內容**，可以做橫向比較。

## 三、測試方法

### 3-1 呼叫方式

線上是 Gradio 5，API 是「兩段式」：

1. `POST {BASE_URL}/{APP}/gradio_api/call/{api_name}`，body 是 `{"data": [...]}`，回傳 `{"event_id": "..."}`。
2. `GET {BASE_URL}/{APP}/gradio_api/call/{api_name}/{event_id}`，這一段是 SSE，
   會**阻塞等待模型算完**，才送 `event: complete` 結束。

需要送檔案的（asr、asr-kaldi）還要先 `POST /gradio_api/upload`（multipart），
拿到伺服器上的路徑，才包成 `gradio.FileData` 送進去。

### 3-2 測量三個指標

| metric | 意義 |
| --- | --- |
| `{模型}_upload_ms` | 上傳測試資料的時間（只有 asr、asr-kaldi 有） |
| `{模型}_infer_ms` | **本測試的主要指標**：上面第 1 段＋第 2 段的時間總和，就是模型運算的延遲 |
| `{模型}_total_ms` | `upload + infer`，使用者實際感受到的總延遲 |

`tts`、`mt` 不用上傳檔案，所以 `total` ＝ `infer`。

### 3-3 執行方式

- 每個模型 `vus: 1`、`iterations: 100`，用 `per-vu-iterations` executor。
  **只有 1 個 VU**，連續一次一次跑，測量的是「沒有其他人搶資源」情況下的延遲，不是壓力測試。
- 四個模型**依序執行，不並行**，才不會互相搶後端資源。
- 每次都重新上傳一次檔案（不重複使用上傳結果），這樣 `upload` 也有 100 筆可看。
- 每次都用 `check()` 確認結果正確（例如 asr 要辨識到 `sasowalen`、tts 要有音檔路徑），
  不能只看 status code 就算成功。
