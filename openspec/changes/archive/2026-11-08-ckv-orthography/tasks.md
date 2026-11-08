# Tasks

**紅綠流程**：先寫測試並確認失敗（讀失敗訊息，確認是功能沒做、不是測試寫錯），再實作，再確認通過。`o2u_except_uo` 是新功能，走完整的紅綠流程（先寫測試、確認失敗、再實作、再確認通過）。`lower_Rupper` 與其他噶瑪蘭規則已經存在於 `common/orthography.py`，這部分的測試寫完就會通過，要暫時弄壞被測的東西，確認它真的會失敗，再還原。

## 1. 測試素材（tests/）

- [x] 1.1 [tests/] 從 `kithann/規範/族語辭典_單詞與例句/20260702_16族前台上線單字_11噶瑪蘭語7476筆(fin)(哈瑪星).xlsx` 的「例句原文」欄節錄約 20 句，存成 `tests/data/ckv_dictionary_orthography.csv`（欄位：`language` 固定為 `ckv`、`sentence`）。去掉重複句；排除含小寫 r、R 以外的大寫字母的句子，以及含不在 u 後面的 o 的句子（這些會被規則改寫，不是不動點）；含 `suykuo` 的句子不排除，素材至少要有一句含它；其餘依原本順序等距取樣。驗證：每句都不含 r，大寫字母只有 R，o 只出現在 u 後面，至少一句含 `suykuo`
- [x] 1.2 [tests/] `tests/data/README.md` 的表格加上 `ckv_dictionary_orthography.csv`，說明來源（族語辭典噶瑪蘭語例句）、節錄方式，以及它是噶瑪蘭正規化不動點測試的素材。驗證：`tox -e pymarkdown` 通過

## 2. 單元測試（tests/）

- [x] 2.0 [common/、tests/] `o2u_except_uo()`：照紅綠流程。先在 `tests/unit/test_orthography.py` 寫測試（單獨 `o` → `u`、`suykuo`、`suykuoan` 不變、`oo` → `uu`、大寫 `O` 不變、其他字元不變），`tox -e test` 確認失敗（函式還不存在）；再在 `common/orthography.py` 實作（`re.sub(r'(?<!u)o', 'u', text)`，只處理小寫），並把 `LANGUAGE_RULES['ckv']` 改成 `[lower_Rupper, quote_2_u02bc, o2u_except_uo]`；`tox -e test` 確認通過
- [x] 2.0.1 [tests/] 刪除 `tests/unit/test_orthography.py` 的 `test_all_languages_rule_order_does_not_matter`：噶瑪蘭的規則順序有意義，這個測試在 2.0 之後一定會失敗。驗證：`tox -e test` 通過
- [x] 2.1 [tests/] 在 `tests/unit/test_orthography.py` 的 `PROBE` 補上 `r`、`R` 與其他大寫字母（例如 `rRPAB`）。驗證：`tox -e test` 通過；暫時在 `ckv` 的 list 最前面加 `v2b, f2v`（`f` 第一次變 `v`，第二次變 `b`），冪等測試要失敗；還原
- [x] 2.2 [tests/] `lower_Rupper()` 單獨測試：`PateRis` → `pateRis`、`saribung` → `saRibung`、`Rabas` 不變、已經是 `R` 的字不會被吃掉、其他字元（數字、標點、空白、漢字）不變。驗證：`tox -e test` 通過；暫時把函式改成只 `text.lower()`，測試要失敗；還原
- [x] 2.3 [tests/] `specs/formosan-orthography/spec.md` 的每個噶瑪蘭 scenario 各一個測試（o→u、保留 u 後面的 o、連續的 o、r→R、保留 R、專有名詞不大寫、保留 b 與 z、撇號統一），加上「太魯閣不受影響」。驗證：`tox -e test` 通過；暫時從 `ckv` 的 list 拿掉 `o2u_except_uo`，o 相關測試要失敗；暫時換成沒有例外的 `o2u`，`suykuo` 的測試要失敗；暫時把 `o2u_except_uo` 移到 list 最前面，`Oba` 的測試要失敗；還原
- [x] 2.4 [tests/] 辭典不動點測試：讀 `tests/data/ckv_dictionary_orthography.csv`，每句 `normalize(sentence, 'ckv')` 必須等於原句。辭典反例用單元測試：`suykuo` → `suykuo`（保留）、`PateRis` → `pateRis`、`saribung` → `saRibung`。測試程式依 CLAUDE.md 用前置 `for`。驗證：`tox -e test` 通過；暫時把素材加一句含單獨 o 的句子，測試要失敗；還原

## 3. 驗證

- [x] 3.1 [tests/] `tox -e test`、`tox -e flake8` 全部通過
- [x] 3.2 [mt/] 測試機（`https://tshi5v100.ithuankhoki.tw`）mt 選 `噶瑪蘭`，「華語 ⮕ 族語」翻譯「你好」「蛋」「水果」。驗證：輸出不含不在 u 後面的 o、小寫 r、R 以外的大寫字母，撇號是 U+02BC
- [x] 3.3 [asr/] 同一台測試機 asr 選 `噶瑪蘭語`，辨識 `tests/data` 的影片。驗證：SRT 的族語行同樣不含不在 u 後面的 o、小寫 r、R 以外的大寫字母，華語行有翻譯
- [x] 3.4 [openspec/] `openspec validate --all --strict` 通過

## Workflow follow-up

- 審查通過後 archive 這個 change，並把 delta sync 進 `formosan-orthography` 主 spec。
- 邵語（元音 i、a、u）與拉阿魯哇語（i、ʉ、a、u）的元音表也沒有 o，規範是否要求 o→u 另案查證，用各自的辭典 xlsx 看 o 的實際使用量。
