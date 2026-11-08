# Proposal

## Why

太魯閣語（`trv-x-truku`）目前沒有任何書寫符號正規化規則，asr 與 mt 的文字裡如果把 u 寫成 o，不會被修正。太魯閣語的 [u] 在 q、h 旁邊會唸成 [o]，但 113 年核定的「原住民族語言書寫系統」第 12 項注意事項 4 規定字仍寫 u（如 `huling`、`sapuh`、`quwaq`、`biyuq`），所以模型或使用者照發音寫成 o 的情況需要修正。

上一個 change（`ckv-orthography`）以「規範列出 o 是獨立元音」為由排除太魯閣。o 確實不能全部轉成 u，但 `kithann/規範/族語辭典_單詞與例句` 的太魯閣語資料（33,401 個詞條、30,121 句不重複例句）顯示 o 的出現位置非常有限，可以用條件式規則處理：

- 所有 o 共 10,344 次，其中 83.1%（8,596 次）是虛詞 `o`、`do`、`hido`、`kido` 與衍生詞 `empeehido`，16.8%（1,740 次）是 `ow` 後接子音，其餘只有 8 次。
- 例句中 o 佔 u+o 的 10.7%，扣掉 `o`、`do` 後只剩約 2%。
- `ow` 是 /aw/ 的同位音（注意事項 6，如 `bowyak`），不是 u 的另一種寫法，不能轉成 `uw`。
- q、h 旁邊沒有獨立的 o：`qo` 只出現在 `qow`，`ho` 只出現在 `how`，和注意事項 4 一致。

把規則套回辭典全部 63,522 筆不重複的詞條與例句，只有 3 個詞會被改動（見「非目標」）。

## What Changes

- 太魯閣（`trv-x-truku`）新增書寫符號正規化規則 `o2u_except_ow_particles`，規則依序判斷：
  1. 整個詞是 `o`、`do`、`hido`、`kido`、`empeehido`（不分大小寫）：不改。
  2. `ow`（o 緊接著 w，不分大小寫）：不改。
  3. 其他的 o：轉成 u，`O` 轉成 `U`。
- 其他字母、大小寫、撇號都不動。太魯閣不套用撇號統一。
- 更新主 spec `formosan-orthography`：涵蓋範圍增加太魯閣，新增太魯閣的 o 規則，撇號需求明列適用語別，移除「太魯閣不受影響」情境。
- 補上測試：規則函式、太魯閣情境，以及用辭典例句做的不動點測試。

## 非目標

- 不做反方向修正：`uw` 後接子音不會改成 `ow`，單獨的 `u`、`du` 也不會改成 `o`、`do`。辭典裡沒有 `uw` 後接子音的拼法，目前沒有資料顯示模型會這樣輸出。
- 不處理 `aw` 與 `ow` 的混用（如 `dawriq` 2 次、`dowriq` 186 次），這不是 o→u 的問題。
- 不修正辭典原文。規則會把 3 個詞改掉，這些是已知並接受的例外：`Toda` → `Tuda`（另一句例句本來就寫 `Tuda`）、`Nomaw` → `Numaw`（應為 `Nowmaw` 漏 w）、`nhapuydo` → `nhapuydu`（應為 `nhapuy do` 漏空格）。
- 不處理賽德克語（`trv-x-td`、`trv-x-tgdy`、`trv-x-trk`）：它們的語言別與規範都和太魯閣語不同，要各自用辭典查證。
- 不統一太魯閣的撇號：沒有對應的需求依據，維持原樣。
- 不改 `normalize()` 的結構與 gradio API。

## Capabilities

### New Capabilities

（無）

### Modified Capabilities

- `formosan-orthography`：涵蓋語別增加太魯閣（`trv-x-truku`）；新增太魯閣的 o 規則需求；撇號統一需求改為明列適用語別，避免太魯閣被誤認為適用。

## Impact

- **common/**：`common/orthography.py` 新增 `o2u_except_ow_particles()`，`LANGUAGE_RULES['trv-x-truku']` 改成 `[o2u_except_ow_particles]`。
- **tests/**：新增辭典例句素材 `tests/data/truku_dictionary_orthography.csv`、更新 `tests/data/README.md`、補單元測試。
- **asr、mt**：不用改程式。asr 用語言別代碼 `trv-x-truku`，mt 用 `trv_Truk` 對到 `trv-x-truku`，兩邊都會自動套用規則。
- **tts、asr-kaldi、deploy**：不用改。tts 與 asr-kaldi 沒有呼叫正規化。
- **向下相容**：只有太魯閣的 asr 輸出與 mt 輸入、輸出可能和之前不同（不符合規則的 o 變成 u），其他語別完全不變。mt 的 gradio API 參數不變。
- **辭典以外的專有名詞與借詞**：含 o 且不屬於上述例外的詞會被轉成 u（如 `Taroko` → `Taruku`）。辭典裡只有 `Toda`、`Nomaw` 這類個案，接受。
