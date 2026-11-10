# 測試素材

[tests/production_tests/](../production_tests/) 的上線後檢查與 [tests/grafana-k6/](../grafana-k6/) 的延遲測試共用這裡的檔案。
**只放一份**，兩邊都指向這個目錄。

| 檔案 | 用途 |
| --- | --- |
| `海岸阿美語-曾玉蘭-個人生命史-短.mp4` | asr（`gr.Video`，吃影片） |
| `海岸阿美語-曾玉蘭-個人生命史-短.mp3` | asr-kaldi（`gr.Audio`，吃音檔） |
| `kako.wav` | production_tests 的 asr 書寫符號檢查（一個詞 kako） |
| `klokah_orthography.csv` | `tests/unit/` 的書寫符號正規化不動點測試 |
| `ckv_dictionary_orthography.csv` | `tests/unit/` 的噶瑪蘭語書寫符號正規化不動點測試 |
| `truku_dictionary_orthography.csv` | `tests/unit/` 的太魯閣語書寫符號正規化不動點測試 |

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

## klokah_orthography.csv 怎麼來的

來源是族語 E 樂園文化篇的句子（`kithann/E樂園/6.文化篇/klokah_cuture_sentence.csv` 的 `sentenceAb` 欄）。
`kithann/` 被 `.gitignore` 忽略、進不了 repo，所以節錄一份放在這裡，單元測試和 CI 才讀得到。

每個語別各取 20 句，共 120 句，欄位是 `language`（語言別代碼）和 `sentence`（族語句子）。
節錄方式：從該 dialectId 的句子中，排除長度不在 15 到 120 字元的句子，再排除含有會被正規化改寫的字母的句子（見下表），剩下的依原本順序等距取 20 句。

| dialectId | 語言別代碼 | 語別 | 排除含有的字母 |
| --- | --- | --- | --- |
| 1 | `ami-x-iams` | 南勢阿美語 | o、f、b |
| 2 | `ami-x-skl` | 秀姑巒阿美語 | u、v、b |
| 3 | `ami-x-pswl` | 海岸阿美語 | u、v、b |
| 4 | `ami-x-frng` | 馬蘭阿美語 | u、v、b |
| 5 | `ami-x-pld` | 恆春阿美語 | u、v、b |
| 43 | `szy` | 撒奇萊雅語 | o、v |

klokah 本身就符合目標拼法，所以這些句子是正規化的不動點：把句子裡的 `’`、`'` 換成 `ʼ`（U+02BC）當作預期結果，
正規化後必須和預期結果完全相同。撒奇萊雅原本有 3 句含 `o`，已經被上面的排除條件剔除。

## ckv_dictionary_orthography.csv 怎麼來的

來源是族語辭典的噶瑪蘭語例句（`kithann/規範/族語辭典_單詞與例句/20260702_16族前台上線單字_11噶瑪蘭語7476筆(fin)(哈瑪星).xlsx` 的「例句原文」欄）。
`kithann/` 被 `.gitignore` 忽略、進不了 repo，所以節錄一份放在這裡，單元測試和 CI 才讀得到。

共 20 句，欄位是 `language`（固定為 `ckv`）和 `sentence`（族語句子）。節錄方式：

1. 去掉重複句，排除長度不在 15 到 120 字元的句子。
2. 排除會被正規化改寫的句子：含小寫 r、R 以外的大寫字母，或含不在 u 後面的 o。
3. 排除含 tab、換行、全形空白或漢字等雜訊的句子。
4. 剩下的句子中，不含 `suykuo` 的依原本順序等距取 19 句，再加上第一句含 `suykuo` 的句子。

辭典例句符合噶瑪蘭語的書寫規範，所以這些句子是正規化的不動點：正規化後必須和原句完全相同。

## truku_dictionary_orthography.csv 怎麼來的

來源是族語辭典的太魯閣語例句（`kithann/規範/族語辭典_單詞與例句/20260702_16族前台上線單字_12太魯閣語33402筆(fin)(哈瑪星).xlsx` 的「例句原文」欄）。
`kithann/` 被 `.gitignore` 忽略、進不了 repo，所以節錄一份放在這裡，單元測試和 CI 才讀得到。

共 20 句，欄位是 `language`（固定為 `trv-x-truku`）和 `sentence`（族語句子）。節錄方式：

1. 去掉重複句，排除長度不在 15 到 120 字元的句子。
2. 排除會被正規化改寫的句子：含 `Toda` 或 `nhapuydo`。
3. 排除含 tab、換行、連續空白、全形空白或漢字等雜訊的句子。
4. 剩下的句子中，先各取第一句含虛詞 `o`、`do`、`hido`、`kido`、`empeehido` 與含 `ow` 的句子（共 6 句），其餘依原本順序等距取 14 句。

除了第 2 點排除的詞，辭典例句的 o 只出現在虛詞與 `ow`，所以這些句子是正規化的不動點：正規化後必須和原句完全相同。
