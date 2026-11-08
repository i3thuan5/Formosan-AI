# Tasks

**紅綠流程**：先寫測試並確認失敗（讀失敗訊息，確認是功能沒做、不是測試寫錯），再實作，再確認通過。`o2u_except_ow_particles` 是新功能，走完整的紅綠流程。不動點等寫完就會通過的測試，要暫時弄壞被測的東西，確認它真的會失敗，再還原。

## 1. 測試素材（tests/）

- [x] 1.1 [tests/] 從 `kithann/規範/族語辭典_單詞與例句/20260702_16族前台上線單字_12太魯閣語33402筆(fin)(哈瑪星).xlsx` 的「例句原文」欄節錄約 20 句，存成 `tests/data/truku_dictionary_orthography.csv`（欄位：`language` 固定為 `trv-x-truku`、`sentence`）。去掉重複句；排除含 `Toda`、`nhapuydo` 的句子，以及含 tab、換行、全形空白或漢字等雜訊的句子；先各取第一句含 `o`、`do`、`hido`、`kido`、`empeehido`、`ow` 的句子，其餘依原本順序等距取樣。驗證：素材包含上述每一類例句，而且 o 只出現在虛詞清單的詞與 `ow`
- [x] 1.2 [tests/] `tests/data/README.md` 的表格加上 `truku_dictionary_orthography.csv`，並新增一節說明來源與節錄方式，比照 `ckv_dictionary_orthography.csv`。驗證：`tox -e pymarkdown` 通過

## 2. 規則與單元測試（common/、tests/）

- [x] 2.1 [common/、tests/] `o2u_except_ow_particles()`：照紅綠流程。先在 `tests/unit/test_orthography.py` 寫函式測試（每個虛詞、句首大寫、全大寫、`ow` 與 `OW`、一般的 o→u、`O`→`U`、`qo`、`ho`、連續的 o、詞界 `do’`、`doʼ`、`do你好`、`o-do`、其他字元不變），`tox -e test` 確認失敗（函式還不存在）；再在 `common/orthography.py` 依 design 決定 1 實作，放在 `o2u_except_uo` 後面，轉換 o 時重用 `_O2U`；`tox -e test` 確認通過
- [x] 2.2 [common/、tests/] 把 `LANGUAGE_RULES['trv-x-truku']` 改成 `[o2u_except_ow_particles]`。先為 `specs/formosan-orthography/spec.md` 的「太魯閣的 o 規則」每個 scenario 各寫一個測試，並把 `test_truku_unchanged` 改名成對應「太魯閣保留 ow 與撇號」，文字維持 `bowyak Away’`；`tox -e test` 確認轉換類的測試失敗；改 `LANGUAGE_RULES` 後確認通過。測試程式依 CLAUDE.md 用前置 `for`
- [x] 2.3 [tests/] 其他語別情境：補上「賽德克不受影響」（`trv-x-trk`，`holing Toda` 不變）。驗證：`tox -e test` 通過；暫時把 `trv-x-trk` 的 list 設成 `[o2u_except_ow_particles]`，測試要失敗；還原
- [x] 2.4 [tests/] 辭典不動點測試：讀 `tests/data/truku_dictionary_orthography.csv`，每句 `normalize(sentence, 'trv-x-truku')` 必須等於原句。辭典反例用單元測試：`Toda` → `Tuda`、`Nomaw` → `Numaw`、`nhapuydo` → `nhapuydu`。驗證：`tox -e test` 通過；暫時在素材加一句含不符合規則的 o 的句子，測試要失敗；還原
- [x] 2.5 [common/、tests/] 破壞確認。暫時把 `trv-x-truku` 改成沒有例外的 `o2u`，虛詞與 `ow` 的測試要失敗；暫時拿掉 regex 的 `re.IGNORECASE`，`Kido`、`OW` 的測試要失敗；暫時把詞界改成 `\b`，`doʼ`、`do你好` 的測試要失敗；全部還原。驗證：還原後 `tox -e test` 通過

## 3. 驗證

- [x] 3.1 [tests/] `tox -e test`、`tox -e flake8`、`tox -e pymarkdown` 全部通過
- [x] 3.2 [openspec/] `openspec validate --all --strict` 通過
- [x] 3.3 [deploy/] 部署到測試機（`https://tshi5v100.ithuankhoki.tw`）。`common/orthography.py` 有改，要重建 `formosan-ai-common` 共用 image 與 asr、mt 的服務 image；`formosan-ai-base`、`formosan-ai-gpu` 不用重建。驗證：asr、mt 服務正常啟動
- [x] 3.4 [mt/] 測試機 mt 選 `太魯閣`，「華語 ⮕ 族語」翻譯幾句。驗證：輸出的 o 只出現在虛詞清單的詞與 `ow`
- [x] 3.5 [asr/] 同一台測試機 asr 選 `太魯閣語`，辨識 `tests/data` 的影片。驗證：SRT 的族語行同樣符合 3.4 的條件，華語行有翻譯。結果：族語行全部符合；5 段有 4 段有華語翻譯，第 1 段族語只有單獨的 `o`，華語是空的。直接問 mt，`o` 也翻成空字串，是模型對單獨 `o` 的行為，這個 change 不會改動 `o`，接受

## Workflow follow-up

- 審查通過後 archive 這個 change，並把 delta sync 進 `formosan-orthography` 主 spec。
- 賽德克語三個語言別的 o 使用情況，另案用賽德克語辭典查證。
