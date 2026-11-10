---
name: hnang
description: 替 tts 的預設配音員找新的參考音檔（族語辭典、族語E樂園），合成試聽句、用 F0 判斷性別，產生給人試聽的網頁，選定後寫回 refs.yaml 並記錄來源。有人回報某位配音員合成出另一個性別、聲音不穩，或上線前要試聽全部配音員（tests/tts_speakers）時使用。
license: MIT
compatibility: 需要 ffmpeg、能連到 tts 服務；族語辭典 xlsx 在 kithann/族語辭典/（不在 git 裡）。
metadata:
  author: Formosan-AI
  version: "1.0"
---

替 `tts/configs/refs.yaml` 的預設配音員換參考音檔，或把上線檢查的合成結果做成試聽網頁。
腳本都在 `scripts/`，網頁模板在 `assets/index.html`。

## 背景

F5-TTS 每次取樣不同。參考音檔太短（約 2 秒）或聲音特徵不明顯時，同一位配音員有時會合成出另一個性別。
2026-10 萬山魯凱、雅美女聲被回報，換參考音檔前 10 句分別有 7、9 句是男聲，換成 3–5 秒、清楚的女聲後 10 句都是女聲。

## 準備環境

這個 devcontainer 的系統 Python 缺 `asyncio`，請用 uv 在 scratchpad 建環境（估 F0 用 `tests/tts_speakers/pitch.py`，要 librosa）：

```bash
uv venv --managed-python --python 3.12 $SCRATCH/hnang-env
uv pip install --python $SCRATCH/hnang-env/bin/python -r tests/tts_speakers/requirements.txt openpyxl
which ffmpeg || (sudo apt-get update && sudo apt-get install -y ffmpeg)
```

`local_synthesize.py`（本機 CPU 合成）另外要一個環境，版本照 `tts/requirements.txt`：

```bash
uv venv --managed-python --python 3.12 $SCRATCH/tts-cpu
grep -vE "^(torch|torchaudio|torchcodec|nvidia|triton)" tts/requirements.txt | grep -E "^[A-Za-z0-9_.-]+==" > $SCRATCH/constraints.txt
uv pip install --python $SCRATCH/tts-cpu/bin/python -c $SCRATCH/constraints.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
    torch==2.8.0 torchaudio==2.8.0 torchcodec==0.7.0 \
    "f5-tts @ git+https://github.com/SWivid/F5-TTS.git@695c7357378c03f25d607cca45f5eb967592ce1a" \
    cached-path omegaconf pyyaml gradio==5.49.1 datasets pyarrow transformers vocos
```

## 流程：換參考音檔

以下 `WORK` 是 scratchpad 裡的工作資料夾，`$PY` 是上面的 hnang-env，指令都在 `scripts/` 裡執行。

### 1. 找候選

```bash
# 族語辭典（雅美語等不分語別）
$PY candidates.py --work WORK --speaker 雅美_女聲 --source dict
# 族語辭典，指定語別（「語別」欄要包含的字）
$PY candidates.py --work WORK --speaker 魯凱_霧台_女聲 --source dict --dict-dialect 霧台
# 族語E樂園 生活會話篇（族語辭典沒有的方言，例如萬山魯凱）
$PY candidates.py --work WORK --speaker 魯凱_萬山_女聲 --source klokah --klokah-dialect 萬山魯凱語
```

- 抽 40 句下載（`--count`）、另抽 10 句當試聽文字（`--texts`），算 F0、長度、有聲比例，
  依「性別對、3–8 秒、有聲比例高」排序印出。
- 族語辭典 xlsx 只有部分方言（魯凱只有霧台），先看 `kithann/族語辭典/` 的「語別」欄。
- **族語E樂園只能 GET**，POST 會 403；短時間請求太多整個網域會連不上一陣子（腳本每 3 秒抓一個）。
  被擋住時等它自己恢復，不要一直重試。

從排序前面挑 5–10 個**性別對、3–8 秒**的候選。同一位錄音者可能錄很多句，盡量挑不同人。

### 2. 合成試聽

```bash
$PY synthesize.py --work WORK --speaker 雅美_女聲 --refs id1,id2,id3
```

- 用 `/default_speaker_tts` 合成「目前預設配音員」，用 `/custom_speaker_tts`（自己當配音員）合成候選，
  每組都合成 `texts.json` 的 10 句。預設打正式機，`BASE_URL=...` 可換主機。
- 「目前預設配音員」是**受測主機已部署的版本**，不一定是工作目錄的 refs.yaml。
- 服務掛掉、或 `/custom_speaker_tts` 回 `AppError` 時（2026-10-15 正式機就是這樣），
  `jobs.json` 已經寫好，改用本機 CPU：`$SCRATCH/tts-cpu/bin/python local_synthesize.py --work WORK`
  （一句約 1 分鐘，70 句約 1 小時，請在背景執行）。
- 已存在的音檔會跳過，中斷後重跑會接著做。
- 會用到受測主機的 GPU，循序呼叫、不併發。

### 3. 產生試聽網頁

```bash
$PY build_site.py --work WORK
```

產生 `WORK/site/`（`index.html`、`data.json`、mp3），最後印出附檔清單。
一次換好幾位配音員時，`--work` 給好幾次並加 `--out DIR`，每位一個頁籤，網頁輸出到 `DIR/site/`。

原本的參考音檔已經確認不行（refs.yaml 有 `gender_unstable`）時，不必再合成它：
把 `jobs.json` 裡 `kind` 是 `current` 的那組拿掉。試聽句可以換成 `tests/tts_speakers/sentences.yaml` 該語別的短句，
和上線前檢查用同一組，最容易聽出變聲。
發布前先載入 `artifact-design` skill，再用 Artifact 發布：`file_path` 是 `WORK/site/index.html`、
`root` 是 `WORK/site`、`files` 用印出的清單，`icon` 用 `audio`。
音檔每組打包成一個 `audio/<組>.json`，附檔數約等於組數；全部配音員各 10 句約 60 個檔、40 MB，一次發布得完。
超過 255 個檔或 64 MB 時，要分批發布到同一個 `url`（第一批發布完拿到 url，之後每批帶 `url` 再發布）。

網頁有自動判斷（F0）、每句試聽、使用者自己標男女，標記可複製成 TSV 貼回來。

### 4. 寫回 refs.yaml

使用者選定後：

```bash
$PY apply_ref.py --work WORK --speaker 雅美_女聲 --ref ilrdf-dictionary-A5413384-...
```

- 音檔複製到 `tts/assets/ref_wav/<候選 id>.wav`（24 kHz 單聲道），
  refs.yaml 該配音員的 `text`、`wav` 換掉，並加 `source`（來源、課句或詞條、ID、原始音檔連結）。
  `app.py` 只讀 `text` 與 `wav`，多的 `source` 不影響服務。
- 依 CLAUDE.md 不能 `git add`，**請提醒使用者自己把新的 wav 加進版本控制**，只 commit refs.yaml 部署後會找不到音檔。
- 跑 `uvx yamllint tts/configs/refs.yaml`。

## 流程：上線前檢查全部配音員的性別

`tests/tts_speakers/`（和 production_tests 分開，判準見該目錄 README）對每位預設配音員合成 10 句
（6 句短句、4 句長句），每句和參考音檔的 F0 比。加 `--audio-dir` 留下音檔，再做成試聽網頁：

```bash
$PY tests/tts_speakers/check_tts_speakers.py --refs-rev <受測主機部署的 commit> --audio-dir WORK/speakers
$PY build_site.py --work WORK --speaker-results WORK/speakers --refs-rev <同一個 commit>
```

這個 devcontainer 的系統 Python 缺 `asyncio`，`tox -e tts_speakers` 建的環境會跑不動，
所以直接用 hnang-env 執行（參數相同）。在 repo 根目錄執行。

一位配音員一組，依「要不要先聽」分頁籤：失敗 → 參考音檔像另一性別（refs.yaml 沒有 `gender_confirmed` 的）→ 警告 → 全部通過。
失敗的句子列最左邊有紅條。`results.jsonl` 的列可以帶 `tab`、`group`、`ref_wav`、`ref_text`，
把實驗（例如同一句合成 10 次）或本機結果放進同一個網頁。

使用者試聽後確認名稱沒錯（音調偏高的男聲、偏低的女聲），在 refs.yaml 該配音員加 `gender_confirmed`
（格式見 `tests/tts_speakers/README.md`）。**男聲變女聲時 F0 常常只差 1.1–1.4 倍，男聲配音員一定要人工試聽。**

使用者試聽聽到變聲（**只要 1 句就算不行**），就在 refs.yaml 該配音員加 `gender_unstable`（日期、說明、聽到變聲的句子），
之後的檢查不再合成這位、直接判失敗。接著用上面的「換參考音檔」流程處理，`apply_ref.py` 換掉整段時會一併拿掉這個欄位。

自動判斷有疑慮、但使用者抽聽的句子都正常，或自動判斷全部通過，就當作通過，加 `gender_checked`（日期、結果、說明）。
之後自動判斷失敗的句子只列警告。還沒聽過有疑慮句子的配音員不要加。
受測主機部署的版本和工作目錄不同時，兩個指令都要加同一個 `--refs-rev`。

## F0 判斷的限制

用 `tests/tts_speakers/pitch.py`（librosa pYIN）估 F0。

- **穩不穩**看「合成 F0 ÷ 參考音檔 F0」（< 0.70 或 > 1.40 失敗），和 `tests/tts_speakers` 相同；
  網頁的「性別」欄只是粗略判斷（女 ≥ 175 Hz、男 ≤ 150 Hz），很多人的聲音本來就在中間。
- **同一句合成 10 次也會變，短句特別容易變**，挑參考音檔時試聽句一定要有短句
  （`candidates.py` 抽的試聽句是 4–12 個詞的辭典／會話句，算短）。
- 音調偏高的男聲、偏低的女聲會誤判（例如 拉阿魯哇_男聲 的參考音檔約 190 Hz），**最後以人耳為準**，
  回報時要請使用者試聽，不要只憑 F0 下結論。
