# 預設配音員聲音穩定度檢查

**上線前**檢查 tts 每位預設配音員合成出來的聲音，是否穩定維持參考音檔的聲音，
以及參考音檔和名稱標的「男聲／女聲」是否一致。
和 [production_tests](../production_tests/) 分開：要合成幾百句、還要人工試聽，很花時間；
配音員也不常改，只在**換參考音檔、換模型、上線前**跑。

## 為什麼要合成多句、而且要有短句

F5-TTS 每次取樣不同。參考音檔太短（約 2 秒）或聲音特徵不明顯時，同一位配音員有時會合成出另一個性別，
**同一句合成 10 次也會變**，而且**短句特別容易變**。2026-10-15 正式機實驗（雅美、萬山魯凱舊參考音檔）：

| 配音員 | 句子 | 同一句合成 10 次：女／男 |
| --- | --- | --- |
| 雅美_女聲 | 長例句 ×2 | 10/0、10/0 |
| 雅美_女聲 | 短句 `o ya am si ina.` | **0/10** |
| 魯凱_萬山_女聲 | 長例句 ×2 | 10/0、7/2 |
| 魯凱_萬山_女聲 | 短句 `cevanana nao pavalrisina molraeli.` | **0/10** |

所以預設每位配音員合成 **10 句：6 句短句、4 句長句**。

- **短句**：[sentences.yaml](sentences.yaml)，每個語別 8 句不重複的 3–8 個詞短句，每句記有來源。
  優先取族語E樂園「句型篇國中・生活百句」，不夠再從「文化篇」的句子依標點切出短句補。
- **長句**：`tts/configs/examples.yaml` 該語別的例句（每個語別約 3 句，會重複）。

### 重新產生 sentences.yaml

資料在 `kithann/E樂園/`（不在 git 裡）。每句都用 `tts/ipa` 的 `text_to_ipa` 驗證，轉不了的字不收，
所以要在 `tts/` 目錄、用 tts 的環境執行：

```bash
cd tts && python ../tests/tts_speakers/build_sentences.py
```

## 執行

在 repo 根目錄：

```bash
# 全部配音員各 10 句，留下音檔供試聽（46 位約 460 句，GPU 上約 40 分鐘）
BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e tts_speakers -- --audio-dir /tmp/tts-speakers

# 受測主機部署的版本和工作目錄不同時，參考音檔改讀部署的 commit
BASE_URL=... tox -e tts_speakers -- --refs-rev <commit> --audio-dir /tmp/tts-speakers

# 只查幾位
BASE_URL=... tox -e tts_speakers -- --speakers 雅美_女聲,魯凱_萬山_女聲 --samples 5

# 本機直接跑 tts/app.py（沒有反向代理的路徑）
TTS_URL=http://127.0.0.1:7860/ tox -e tts_speakers -- --speakers 雅美_女聲
```

| 參數 | 用途 |
| --- | --- |
| `--speakers` | 逗號分隔的配音員名稱，預設 `refs.yaml` 全部 |
| `--samples <n>` | 每位合成幾句，每 5 句有 3 句短句，預設 10 |
| `--audio-dir <dir>` | 音檔與 `results.jsonl`（每句的 F0、比值、判斷）留在這裡，結尾會印每位配音員的通過／警告／失敗句數 |
| `--refs-rev <commit>` | `refs.yaml`、`examples.yaml` 與參考音檔改讀這個 commit |

連線與輸出共用 production_tests 的 `reporting.py`；套件見 `requirements.in`（多了估 F0 用的 librosa），
改完用 `pip-compile --strip-extras requirements.in` 更新 `requirements.txt`。

## 判準

F0 用 librosa 的 pYIN 估（[pitch.py](pitch.py)），取有聲音框的中位數。

**每一句：合成 F0 ÷ 該配音員參考音檔的 F0**

| 比值 | 結果 |
| --- | --- |
| < 0.70 或 > 1.40 | **失敗**：聲音變了（通常是變成另一個性別） |
| 0.70–0.80 或 1.25–1.40 | **警告**：請人工試聽 |
| 0.80–1.25 | 通過 |

**男聲變女聲時，比值常常只有 1.1–1.4**（男聲參考音檔本來就在 150 Hz 上下），2026-10-15 人工試聽發現幾句只被判通過或警告；
反過來，沙啞的男聲有時被估到 200 Hz 而判失敗。所以**男聲配音員一定要人工試聽**，不能只看結果。

2026-10-15 對正式機 46 位配音員各 10 句：穩定的配音員多在 0.84–1.12，確定變成另一個性別的句子在 0.39–0.66。
用「和參考音檔比」而不是固定的男女門檻，是因為很多配音員的聲音本來就在 150–175 Hz 之間，
固定門檻判斷不出來，但他們每次合成都很穩定。

**參考音檔本身**：F0 ≥ 175 Hz 像女聲、≤ 150 Hz 像男聲，和名稱相反時印**警告**，請試聽確認是名稱標錯還是音檔有問題。
試聽確認名稱沒錯（音調偏高的男聲、偏低的女聲）時，在 `refs.yaml` 該配音員加 `gender_confirmed`，之後就不再警告：

```yaml
泰雅_汶水_男聲:
  text: ...
  wav: ...
  gender_confirmed:
    gender: 男
    date: 2026-10-15
    note: 參考音檔 F0 約 226 Hz，音調偏高，試聽確認是男聲
```

**人工試聽確認會變聲**：只要有 1 句變成另一個性別，這位配音員就算不行。在 `refs.yaml` 加 `gender_unstable`，
之後檢查直接判失敗、不再合成（省 GPU 時間）；換了參考音檔後把它拿掉（`/hnang` 的 `apply_ref.py` 會整段換掉）：

```yaml
布農_卓群_女聲:
  text: ...
  wav: ...
  gender_unstable:
    date: 2026-10-15
    note: 正式機 10 句人工試聽，有 2 句變成男聲，要換參考音檔
    heard_as_male:
      - maca matinus huluc a.
      - nanu tu iciat binanau az tu kukuzaun.
```

**人工試聽通過**：自動判斷有疑慮的句子人工抽聽都正常、或自動判斷全部通過，就當作通過，加 `gender_checked`。
之後檢查時，自動判斷失敗的句子改列**警告**（F0 可能誤判，例如沙啞的男聲），不擋上線，但仍請人試聽：

```yaml
阿美_南勢_女聲:
  text: ...
  wav: ...
  gender_checked:
    date: 2026-10-15
    result: 通過
    note: 正式機 10 句，自動判斷有疑慮的句子人工抽聽都正常
```

`app.py` 只讀 `text`、`wav`，多的欄位不影響服務。用 `--refs-rev` 讀舊版本時，參考音檔相同就沿用工作目錄的
`gender_confirmed`、`gender_unstable`、`gender_checked`。

F0 只是輔助，音調偏高的男聲、偏低的女聲會誤判，**最後以人耳為準**。
加 `--audio-dir` 留下音檔，再用 `/hnang` skill 的 `build_site.py --speaker-results` 做成試聽網頁；
真的不穩時，用 `/hnang` skill 找新的參考音檔替換。

## 注意

- **會實際使用受測主機的 GPU**，循序呼叫、不併發，請避開尖峰時段。
- **不要對正式機跑全部配音員**：2026-10-15 對正式機連續合成數十到一百多句後，正式機三次整段回 `AppError`
  （連 `/synthesize` 都失敗），請用測試機。只查一兩位配音員時再考慮正式機。
- 合成用的是受測主機**已部署**的參考音檔；比對用的參考音檔讀自工作目錄或 `--refs-rev`，兩邊要是同一版。
  配音員名稱不在受測主機上時（例如改了名稱還沒部署），該配音員每句都會失敗。
