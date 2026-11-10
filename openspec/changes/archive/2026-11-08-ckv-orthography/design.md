# Design

## Context

規則與動機見 proposal.md，可觀察的行為見 `specs/formosan-orthography/spec.md`。現況：

- `common/orthography.py` 的 `LANGUAGE_RULES['ckv']` 目前是 `[o2u, lower_Rupper, quote_2_u02bc]`，`lower_Rupper` 是 `text.lower().replace('r', 'R')`。這個 change 會把其中的 `o2u` 換成 `o2u_except_uo`。
- 上一個 change 的冪等測試與順序無關測試（`test_all_languages_rule_order_does_not_matter`），用 `tests/unit/test_orthography.py` 的 `PROBE = 'oOuUfFvVbB ’\' ʼ ^'` 檢查。這個字串沒有 `r`、`R`，也沒有 `o`、`u`、`b` 以外的大寫字母，所以噶瑪蘭的規則通過了測試，卻沒有被檢查到。
- 噶瑪蘭語的辭典資料在 `kithann/規範/族語辭典_單詞與例句/` 的「噶瑪蘭語」xlsx，被 git 忽略，CI 讀不到。上一個 change 的 klokah 素材沒有噶瑪蘭。

## Goals / Non-Goals

**Goals：**

- 噶瑪蘭規則有測試，測試真的會因為規則壞掉而失敗。
- 主 spec 和程式碼一致。

**Non-Goals：**

- 不改 `normalize()` 的結構、不處理其他語別。除了 `o2u_except_uo`，不新增規則函式。

## Decisions

### 1. 規則改成 `[lower_Rupper, quote_2_u02bc, o2u_except_uo]`

- **o2u_except_uo**：規範第 11 項元音表沒有 o，注意事項 5 說 [u]、[o] 是同位音，統一寫 u，所以 o 轉成 u。例外是緊接在 u 後面的 o：`uo` 念 [uo]，是兩個元音連續，不是規範說的同位音（[u] 在 q、R 附近念成 [o]，仍寫 u），所以不能把它當成 [o] 的另一種寫法而轉成 `uu`。辭典裡唯二含 o 的單字 `suykuo`、`suykuoan`（水果，華語借詞）都是這個形態，規範第 6 點也允許借詞用本族以外的字母。
  - 實作是 `re.sub(r'(?<!u)o', 'u', text)`：前一個字元（看這一步之前的文字，不是轉換後的結果）是 u 的 o 保留，其他 o 轉成 u。需要標準函式庫的 `re`，不新增依賴。
  - 放在 `lower_Rupper` 之後，文字已經全部是小寫（R 除外），所以這個函式只處理小寫 o，不用處理大小寫。單獨呼叫時大寫 `O` 不會被轉，這是刻意的。
  - 冪等：`oo` 兩個 o 的前一個字元都不是 u，一次轉成 `uu`，再正規化一次不變；`uo` 保留，再正規化一次不變。
  - 代價：不再是純字母對照，要看前一個字元。
  - **替代方案**：維持單純的 `o2u`，`suykuo` 會變成 `suykuu`。真實資料只有這兩個 o，被改壞的只有這兩個借詞，所以選擇多一個小函式保護它們。
- **lower_Rupper**：R 是小舌擦音，規範用大寫標示；其他字母都用小寫。把 r 換成 R，是因為書寫系統的輔音表沒有小寫 r（台東的 [r] 寫成 z），辭典裡 36 個含小寫 r 的單字，沒有任何一個同時含 R，也沒有任何一個換成 R 之後會和別的單字相同，換成 R 是安全的。
- **quote_2_u02bc**：規範的喉塞音是 `’`，目標是 `ʼ`。辭典資料本來就是 `ʼ`，所以這條規則對辭典是空操作，主要是處理使用者輸入與模型輸出。
- **保留函式名稱 `lower_Rupper`**：使用者已經命名，不另外改。
- **為什麼 `lower()` 可以接受**：AI需求.pdf 寫噶瑪蘭「專有名詞及句首不大寫」，辭典例句也確實沒有大寫（R 除外）。全部轉小寫再把 r 換成 R，等於「除了 R 以外都小寫」。

### 2. 噶瑪蘭的規則順序有意義，移除順序無關測試

噶瑪蘭的 `O` 要先經過 `lower_Rupper` 變成 `o`，`o2u_except_uo` 才會把它轉成 `u`。倒過來套用時 `Oba` 會變成 `oba`，結果不同，所以「list 反過來套用，結果相同」對噶瑪蘭不成立。刪除 `tests/unit/test_orthography.py` 的 `test_all_languages_rule_order_does_not_matter`，spec 也不再要求順序無關。

- 「每個字元最多轉換一次」改成只規定噶瑪蘭以外的語別；重複正規化結果不變則所有語別都要成立，冪等測試保留。
- **代價**：這個測試原本負責抓環狀規則。`f2v` 加 `v2f` 這種組合，正規化一次和兩次的結果相同，冪等測試抓不到（實測確認）。之後在 list 加規則時，要靠每個語別各自的情境測試把關。

### 3. 測試

- **規則函式**：`o2u_except_uo` 單獨測試：單獨的 `o` → `u`、`suykuo` 不變、`suykuoan` 不變、`oo` → `uu`、大寫 `O` 不變、其他字元不變。`lower_Rupper` 單獨測試（`PateRis` → `pateRis`、`saribung` → `saRibung`、`Rabas` 不變、`R` 保留），也要測試已經是 `R` 的字不會被吃掉。
- **spec 情境**：每個 scenario 一個測試（o→u、大寫 O 先轉小寫再轉 u、u 後面的 o 保留、r→R、專有名詞不大寫、R 保留、撇號、其他語別不受影響）。
- **PROBE 補上 `r`、`R` 與其他大寫字母**（例如 `rRPAB`）：既有的冪等測試對 `LANGUAGE_RULES` 的每個語別都會跑，補完後噶瑪蘭才被涵蓋。這個測試寫完就會通過，所以要破壞確認：暫時在 `ckv` 的 list 最前面加 `v2b, f2v`（`f` 第一次變 `v`，第二次變 `b`），冪等測試要失敗；也要暫時拿掉 `o2u_except_uo`，確認單獨 o 的測試失敗；暫時改成沒有例外的 `o2u`，確認 `suykuo` 的測試失敗；暫時把 `o2u_except_uo` 移到 list 最前面，確認 `Oba` 的測試失敗。
- **刪除順序無關測試**：見決定 2。
- **辭典不動點測試**：辭典例句本身符合規範，所以經過正規化要完全不變。素材放 `tests/data/ckv_dictionary_orthography.csv`，約 20 句，欄位 `language`（`ckv`）、`sentence`。從辭典例句原文節錄，排除含 r、R 以外的大寫字母的句子，以及含不在 u 後面的 o 的句子（這些會被規則改寫，不是不動點）；含 `suykuo` 的句子不排除，素材中至少要有一句含它，其餘等距取樣。測試讀取方式比照既有的 klokah 不動點測試，預期結果是句子本身。
- **辭典的反例**（不動點之外）：直接寫成單元測試，不放進素材檔：`suykuo` → `suykuo`（保留）、`PateRis` → `pateRis`、`saribung` → `saRibung`。

### 4. 驗證

- `tox -e test`、`tox -e flake8`。
- 部署後在測試機：mt 選 `噶瑪蘭`，「華語 ⮕ 族語」翻譯幾句，輸出不含不在 u 後面的 o、小寫 r、R 以外的大寫字母，撇號是 U+02BC；asr 選 `噶瑪蘭語` 辨識 `tests/data` 的影片，族語行同樣檢查。這不是 production tests 的固定項目，不加進腳本。

## Risks / Trade-offs

- **辭典以外的借詞若含不在 u 後面的 o，仍會被轉成 u**：`suykuo`、`suykuoan`（水果）因為 o 在 u 後面而保留。例外只依辭典資料設定，沒有辭典以外借詞的資料，接受。
- **`A:`、`B:` 對話標記被轉成 `a:`、`b:`**：只出現在辭典的對話例句，asr 與 mt 不會產生這種標記，接受。
- **r→R 可能把真正的小寫 r 改成 R**：書寫系統沒有小寫 r，辭典的 36 個例子也沒有反例，接受。
- **全部轉小寫對 mt 輸入的影響**：使用者輸入的句首大寫會被改成小寫再送進 NLLB。依 AI需求.pdf，噶瑪蘭句首本來就不大寫，所以不是問題。
- **NLLB 對噶瑪蘭的輸出本來就可能是小寫 r 或含 o**：這就是這條規則存在的原因，之後可以用 mt 的輸出觀察規則實際改了多少。

## Migration Plan

1. 照一般流程重建 `formosan-ai-common`、asr、mt 的 image，然後部署。噶瑪蘭的規則（含 `lower_Rupper`）原本就在 `common/orthography.py`，是否已經部署過，以測試機為準。
2. 部署後做 design 第 4 點的手動檢查。
3. 回滾：還原 commit 並重新部署。
