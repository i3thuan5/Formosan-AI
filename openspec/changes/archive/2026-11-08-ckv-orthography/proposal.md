# Proposal

## Why

113 年核定的「原住民族語言書寫系統」第 11 項規定噶瑪蘭語（`ckv`）：元音只有 i、e、a、u，o 與 u 是同位音，統一寫 u；小舌擦音用大寫 R，其餘字母都用小寫（R 是這套書寫系統中用大寫字母的唯二例外之一）；喉塞音寫 `’`。`kithann/規範/AI需求.pdf` 也寫明噶瑪蘭語「大小寫為不同發音，專有名詞及句首不大寫」。

上一個 change（`amis-sakizaya-orthography`）只涵蓋阿美與撒奇萊雅，噶瑪蘭的 asr 輸出與 mt 輸入、輸出沒有任何後處理。`common/orthography.py` 已經加上噶瑪蘭的規則（`o2u`、`lower_Rupper`、`quote_2_u02bc`），但沒有測試，而且 `o2u` 會把借詞 `suykuo` 改成 `suykuu`，主 spec `formosan-orthography` 也還寫著「只涵蓋三組語別」「保留大小寫」，和程式碼矛盾。

用 `kithann/規範/族語辭典_單詞與例句` 的噶瑪蘭語資料（7,474 個單字、10,390 句例句）驗證規則：

- 單字與例句沒有 `’`、`'`，撇號全部是 `ʼ`（U+02BC），和正規化的目標一致。
- 例句的句首沒有大寫，只有 R。例外是對話標記 `A:`、`B:`（約 209 句）和一句專有名詞 `PateRis`。
- o 只出現在 2 個單字：`suykuo`、`suykuoan`（水果，華語借詞），而且兩個都是 o 緊接在 u 後面，念 [uo]（兩個元音連續），不是 u 的同位音 [o]。
- 小寫 r 出現在 36 個單字（如 `saribung`、`serai`、`raynguma`），例句裡沒有。沒有任何單字同時含 r 和 R，把 r 換成 R 也不會讓兩個不同的單字變成同一個。這些 r 看起來是辭典裡 R 的誤植，規則會把它們修正。

## What Changes

- 噶瑪蘭（`ckv`）的書寫符號正規化規則，順序為 `[lower_Rupper, quote_2_u02bc, o2u_except_uo]`：
  - 全部轉小寫，再把 r 換成 R（所以 R 保留，其他大寫如 `PateRis` → `pateRis`）。
  - 撇號 `’`、`'` 統一為 `ʼ`。
  - o→u，但緊接在 u 後面的 o 保留，因為 `uo` 念 [uo]，是兩個元音連續，不是 [u] 的同位音 [o]（辭典裡的借詞 `suykuo`、`suykuoan`）。這需要看前一個字元，所以新增規則函式 `o2u_except_uo`；它排在轉小寫之後，只處理小寫 o。南勢與撒奇萊雅繼續用 `o2u`。
- 補上測試：每個規則函式、噶瑪蘭的情境，以及用辭典例句做的不動點測試。
- 更新主 spec `formosan-orthography`：涵蓋範圍增加噶瑪蘭，撇號規則增加噶瑪蘭，大小寫規則增加噶瑪蘭的例外，並新增噶瑪蘭的字母與大小寫需求。
- 單元測試的探測字串（`PROBE`）補上 `r`、`R` 和其他大寫字母，讓既有的冪等測試涵蓋噶瑪蘭。
- 刪除順序無關測試 `test_all_languages_rule_order_does_not_matter`：噶瑪蘭的規則順序有意義，倒過來套用時大寫 `O` 不會變成 `u`。

## 非目標

- 不處理太魯閣：規範列出 o 是獨立元音，不能轉成 u，維持沒有規則。
- 不處理邵語、拉阿魯哇語等其他語別：它們的元音表也沒有 o，但規範沒有明寫 u/o 是同位音，另案查證。
- 不處理賽夏語的大寫 S：賽夏大小寫是不同音，不能套用全部轉小寫。
- 不處理辭典裡的 `A:`、`B:` 對話標記：噶瑪蘭的規則會把它轉成 `a:`、`b:`，目前沒有人會把這種標記送進 asr 或 mt，接受。
- 不處理辭典以外的借詞：緊接在 u 後面的 o 保留，所以辭典裡的 `suykuo`（水果）不變；但其他借詞若含不在 u 後面的 o，仍會被轉成 u。規範第 6 點允許借詞用本族以外的字母，字母層級的規則無法區分，只能就辭典的實際資料設例外。
- 不修改 `lower_Rupper` 的名稱與 gradio API。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `formosan-orthography`：涵蓋語別增加噶瑪蘭（`ckv`）；撇號統一涵蓋噶瑪蘭；大小寫規則為噶瑪蘭設例外；新增噶瑪蘭的字母與大小寫需求。

## Impact

- **common/orthography.py**：新增 `o2u_except_uo()`，`LANGUAGE_RULES['ckv']` 改成 `[lower_Rupper, quote_2_u02bc, o2u_except_uo]`。
- **tests/**：新增辭典例句素材 `tests/data/ckv_dictionary_orthography.csv`、更新 `tests/data/README.md`、補單元測試。
- **asr、mt、tts**：不用改程式。asr 用語言別代碼 `ckv`，mt 用 `ckv_Kava` 對到 `ckv`，兩邊都會自動套用噶瑪蘭規則。
- **向下相容**：噶瑪蘭的 asr 輸出與 mt 輸入、輸出可能和之前不同（o→u、r→R、大寫變小寫、撇號），其他語別完全不變。
- **tts**：tts 不套用正規化，噶瑪蘭的合成文字由 mt 的「翻譯結果」決定，已經是正規化後的版本。
