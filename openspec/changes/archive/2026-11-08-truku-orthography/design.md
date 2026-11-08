# Design

## Context

規則與動機見 proposal.md，可觀察的行為見 `specs/formosan-orthography/spec.md`。現況：

- `common/orthography.py` 的 `LANGUAGE_RULES['trv-x-truku']` 是空的 list。既有的規則函式都是字元層級：`o2u` 用對照表，`o2u_except_uo` 只看前一個字元。太魯閣的規則要判斷「整個詞」，是第一個需要詞界的規則。
- 主 spec 有「太魯閣不受影響」情境，`tests/unit/test_orthography.py` 有對應的 `test_truku_unchanged`，文字是 `bowyak Away’`。
- 主 spec 的撇號需求和涵蓋範圍需求都用組數描述語別。太魯閣語加入涵蓋範圍，但不統一撇號，兩個需求適用的語別不再相同。
- 太魯閣語辭典在 `kithann/`，被 git 忽略，CI 讀不到，所以測試素材要節錄進 `tests/data/`，做法比照 `ckv_dictionary_orthography.csv`。

## Goals / Non-Goals

**Goals：**

- 不符合規則的 o 轉成 u，同時完整保留虛詞與 `ow`。
- 大小寫保留，重複正規化結果不變。
- 主 spec 和程式碼一致。

**Non-Goals：**

- 不改 `normalize()` 的結構，不處理其他語別，不做 u→o 或 `uw`→`ow` 的反向修正。

## Decisions

### 1. 用一個 regex 同時判斷三條規則

新增 `o2u_except_ow_particles(text)`，`LANGUAGE_RULES['trv-x-truku']` 設為 `[o2u_except_ow_particles]`。

regex 概念如下，加上 `re.IGNORECASE`：

```text
(?P<keep>(?<![a-z])(?:o|do|hido|kido|empeehido)(?![a-z]))|o(?!w)
```

由左到右掃描，兩個分支對應規則：

- `keep` 分支比對到整個虛詞時，原樣回傳，掃描位置跳過這個詞，裡面的 o 就不會被第二個分支處理（規則 1）。
- 第二個分支只比對後面不是 w 的 o（規則 3），`ow` 自然不會被比對到（規則 2）。比對到的 o 用既有的 `_O2U` 對照表轉換，大小寫由對照表處理，不用另寫。

選擇理由：一個 regex 不需要先斷詞再逐詞處理，也不用維護第二份詞界定義。

- **替代方案**：先用 `[a-zA-Z]+` 斷詞，再對每個詞判斷。結果相同，但要多一層函式。

### 2. 詞界只看英文字母

「整個詞」定義為前後都不是英文字母，不用 `\b`。

- Python 的 `\b` 把 `ʼ`（U+02BC，Unicode 修飾字母）和漢字都當成詞內字元，`doʼ`、`do你好` 會被判斷成不是完整的 `do`，進而轉成 `du`。
- 撇號不算詞的一部分，所以 `do’`、`‘do’` 這類加引號的虛詞會保留。辭典裡含撇號又含 o 的詞只有 `bowkʼ`、`Bowsʼ`、`bowtʼ`、`lowpʼ`、`bowngʼ`、`towpʼ`，都因為 `ow` 不變，不受這個選擇影響。
- 連字號同樣是詞界，`o-do` 兩邊都保留。

用這個定義套回辭典 63,522 筆不重複的詞條與例句，只改動 3 個詞（見 proposal 的非目標），重複正規化結果不變。

### 3. 虛詞清單用封閉清單，比對不分大小寫

清單是 `o`、`do`、`hido`、`kido`、`empeehido`。辭典裡有句首的 `O`、`Kido`、`Empeehido`，只比對小寫會把 `Kido` 改成 `Kidu`。

- `empeehido`（「之後要」）只出現 1 次，但它是 `hido` 的衍生詞，留在清單裡，規則就不會產生 `empeehidu`。
- **替代方案**：規則寫成「詞尾的 o 都保留」。這會讓任何以 o 結尾的錯字都不被修正，而辭典裡詞尾的 o 只有清單裡的詞與 1 個漏空格的 `nhapuydo`，封閉清單更精確。

### 4. `ow` 一律不動，不限定後接子音

資料上 `ow` 幾乎都後接子音（1,740 次，後接元音 2 次、詞尾 3 次）。規則不加「後接子音」的條件，因為這條規則只做 o→u：多這個條件只會把 `rowan`、`gow`、`tggow` 這些詞改成 `ruwan`、`guw`、`tgguw`，其中擬聲詞 `gow` 會被改壞。

### 5. 不套用撇號統一，主 spec 改成明列語別

`LANGUAGE_RULES['trv-x-truku']` 不加 `quote_2_u02bc`：這個 change 的依據只有 o 的分布，太魯閣的撇號沒有驗證過。主 spec 的撇號需求改成明列南勢阿美語、秀姑巒阿美語、海岸阿美語、馬蘭阿美語、恆春阿美語、撒奇萊雅語、噶瑪蘭語，並寫明太魯閣語不適用，不再用組數描述。

原本的情境文字 `bowyak Away’` 剛好能驗證 `ow` 與撇號都不變，所以改成太魯閣規則下的情境，繼續使用。

### 6. 測試

- **規則函式**：`o2u_except_ow_particles` 單獨測試，涵蓋每個虛詞、句首大寫、全大寫、`ow` 與 `OW`、一般的 o→u、`O`→`U`、`qo`、`ho`、連續的 o、詞界（`do’`、`doʼ`、`do你好`、`o-do`）、其他字元不變。
- **spec 情境**：每個 scenario 一個測試。`test_truku_unchanged` 改名為符合新情境的名稱，文字維持 `bowyak Away’`。
- **PROBE 不用改**：既有的冪等測試對每個語別都會跑，`PROBE` 已含 `o`、`O`，但沒有虛詞與 `ow`，所以太魯閣另外加一個重複正規化的情境測試。
- **辭典不動點測試**：素材放 `tests/data/truku_dictionary_orthography.csv`，約 20 句，欄位 `language`（`trv-x-truku`）、`sentence`。節錄方式比照噶瑪蘭，排除含 `Toda`、`nhapuydo` 的句子；素材必須包含 `o`、`do`、`hido`、`kido`、`empeehido` 與 `ow` 的例句，其餘等距取樣。
- **辭典的反例**（不動點之外）：直接寫成單元測試：`Toda` → `Tuda`、`Nomaw` → `Numaw`、`nhapuydo` → `nhapuydu`，記錄這些例外是刻意接受的。
- **破壞確認**：這是新功能，走紅綠流程。之後再暫時改成沒有例外的 `o2u`，虛詞與 `ow` 的測試要失敗；暫時改成區分大小寫，`Kido` 的測試要失敗。

### 7. 驗證

- `tox -e test`、`tox -e flake8`、`tox -e pymarkdown`、`openspec validate --all --strict`。
- 部署後在測試機手動檢查 mt 與 asr 的太魯閣輸出：o 只出現在虛詞清單的詞與 `ow`。這不是 production tests 的固定項目，不加進腳本。

## Risks / Trade-offs

- [辭典以外的專有名詞或借詞含 o，例如 `Taroko` 會變成 `Taruku`] → 辭典裡只有 `Toda`、`Nomaw` 這類個案，沒有其他資料可以設例外，接受。之後若發現常見的詞，再加進封閉清單。
- [規則依據只有一份辭典，其他來源的拼寫慣例可能不同，也還不知道模型實際會輸出多少不合規則的 o] → 部署後用 mt 與 asr 的輸出觀察；規則只影響太魯閣，回滾範圍小。
- [`uw` 後接子音不會修正成 `ow`，模型若輸出 `buwyak` 會原樣保留] → 辭典裡沒有這種拼法，先不處理，等實際看到再加。
- [辭典裡的 3 個詞會被改動，其中 `Numaw`、`nhapuydu` 是錯的] → 根源是辭典原文的誤植，而且各只出現 1 次，接受，並用單元測試記錄。
- [虛詞與實詞同形時無法區分] → 辭典裡沒有 `o`、`do` 的 o/u 對立（`du` 未出現，單獨的 `u` 只有 2 次，依句意是 `ku` 的誤植），目前不成問題。

## Migration Plan

1. 照一般流程重建 `formosan-ai-common`、asr、mt 的 image，然後部署。
2. 部署後做決定 7 的手動檢查。
3. 回滾：還原 commit 並重新部署。規則是純文字處理，沒有資料遷移。

## Open Questions

（無）
