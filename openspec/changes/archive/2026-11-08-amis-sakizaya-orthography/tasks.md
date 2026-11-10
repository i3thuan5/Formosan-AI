# Tasks

**紅綠流程**：每個功能（一個規則函式、`normalize`、`LANGUAGE_CODES`）各做一輪，不批次寫完才一起跑：

1. 先寫這個功能的測試，執行 `tox -e test` 確認**失敗**，並讀失敗訊息，確認是因為功能還沒做，不是測試本身寫錯。
2. 再實作這個功能。
3. 立刻重跑 `tox -e test`，確認通過，才開始下一個功能。

寫完就會通過的測試（冪等、代碼防呆），要暫時弄壞被測的東西，確認測試會失敗，再還原。asr、mt 的 `app.py` 會 import 重型套件，不能單元測試，改完後立刻做該任務的驗證步驟。

## 1. 正規化模組與單元測試（common/、tests/、tox.ini、.travis.yml）

- [x] 1.1 [tox.ini] 新增 `[testenv:test]`：deps 是 pytest，`setenv` 把 `PYTHONPATH` 設為 `common`、`asr`、`mt`（5.5 縮成只剩 `common`），commands 是 `pytest tests/unit {posargs}`。pytest 只是 tox 的依賴，不改任何服務的 requirements，所以不需要 `uv pip compile` 和 `tox -e lockcheck`。驗證：`tox -e test -- --version` 印出 pytest 版本
- [x] 1.2 [tests/] 從 `kithann/E樂園/6.文化篇/klokah_cuture_sentence.csv` 節錄 dialectId 1、2、3、4、5、43 各約 20 句，存成 `tests/data/klokah_orthography.csv`（欄位：`language` 為語言別代碼、`sentence`）。排除撒奇萊雅中孤立的 `o` 那一句。驗證：檔案中每個語別碼都有句子，而且南勢與撒奇萊雅的句子都不含 o，通用阿美的句子都不含 u、v、b
- [x] 1.3 [tests/] 在 `tests/data/README.md` 的表格加上 `klokah_orthography.csv`，並說明來源（族語 E 樂園文化篇）、節錄方式、各 dialectId 對應的語言別代碼，以及它是正規化不動點測試的素材。驗證：`tox -e pymarkdown` 通過
- [x] 1.4 [common/、tests/] `o2u()`：照紅綠流程。先在 `tests/unit/test_orthography.py` 寫測試（`o`→`u`、`O`→`U`，其他字元不變），`tox -e test` 確認失敗（此時 `common/orthography.py` 還不存在）；再新增 `common/orthography.py` 實作 `o2u()`，用 `str.maketrans` 建表，大寫的對應由程式產生，只依賴標準函式庫，迴圈依 CLAUDE.md 用前置 `for`；`tox -e test` 確認通過
- [x] 1.5 [common/、tests/] `u2o()`：照紅綠流程，測試含大小寫與其他字元不變
- [x] 1.6 [common/、tests/] `f2v()`：照紅綠流程，測試含大小寫與其他字元不變
- [x] 1.7 [common/、tests/] `b2v()`：照紅綠流程，測試含大小寫與其他字元不變
- [x] 1.8 [common/、tests/] `v2f()`：照紅綠流程，測試含大小寫與其他字元不變
- [x] 1.9 [common/、tests/] `b2f()`：照紅綠流程，測試含大小寫與其他字元不變
- [x] 1.10 [common/、tests/] `v2b()`：照紅綠流程，測試含大小寫與其他字元不變；`f` 保留不動
- [x] 1.11 [common/、tests/] `quote_2_u02bc()`：照紅綠流程。測試 `’`（U+2019）、`'`（U+0027）轉成 `ʼ`（U+02BC），已經是 `ʼ` 的維持不變，`^` 保留不動
- [x] 1.12 [common/、tests/] `normalize(text, language)` 與 `LANGUAGE_RULES`：照紅綠流程。先把 `specs/formosan-orthography/spec.md` 每個 scenario 寫成一個測試（其他語別、華語、南勢、通用、撒奇萊雅、撇號、`^` 保留、大小寫、重複正規化結果不變），再加上 klokah 不動點測試：把每句的 `’`、`'` 換成 `ʼ` 當作預期結果，正規化後必須相同；`tox -e test` 確認失敗。再實作 `LANGUAGE_RULES`（`ami-x-iams`、`ami-x-skl`、`ami-x-pswl`、`ami-x-frng`、`ami-x-pld`、`szy` 各自一行有規則的 list，不共用；其餘 36 個語言別代碼也各列一行，list 留空；42 個 key 的順序照 CSV 的「顯示排序先後」）與 `normalize`（依序套用 list，不在表裡的值如 `None`、`zho_Hant` 原樣回傳）；`tox -e test` 確認通過。驗證：`PYTHONPATH=common python -c "from orthography import normalize; print(normalize('fita’ol', 'ami-x-iams'))"` 印出 `vitaʼul`
- [x] 1.13 [tests/] 加上三個防呆測試。冪等測試：對 `LANGUAGE_RULES` 每個語別，用含全部相關字母的字串正規化兩次，結果相同；順序無關測試：list 反過來套用，結果與正著套用相同（環狀規則如 `f2v` 加 `v2f` 會通過冪等測試，要靠這個抓）；代碼防呆：`LANGUAGE_RULES` 的 key 必須等於 `LANGUAGE_GROUPS`（原本在 `asr/languages.py`，現在在 `common/languages.py`）的全部語言別代碼（42 個），順序也要相同（`LANGUAGE_GROUPS` 的順序就是 CSV 的順序）。這三個測試寫完就會通過，所以要各做一次破壞確認：暫時在某個語別的 list 加一個會連鎖的函式（例如南勢加 `v2f`）、暫時把某個 key 改錯、刪掉一個或對調兩個，`tox -e test` 都要失敗；確認後還原，再確認通過
- [x] 1.14 [.travis.yml] 在 `Check dependency lock consistency` 之後新增 job `Unit tests`：`pip install tox`，script 是 `tox -e test`。驗證：`.tox/yamllint/bin/yamllint .travis.yml` 通過（整個 repo 的 `tox -e yamllint` 目前會被本機 `kithann/` 底下的檔案擋住，那些檔案不在 repo 裡）；PR 上的 Travis 結果見 4.4
- [x] 1.15 [common/] `common/Dockerfile` 的 files stage 改成 `COPY colors.py languages.py orthography.py utils.py /app/`（`languages.py` 是 5.1 搬來的）。Docker（依使用者判斷：服務的 production tests 都通過就代表 build 正常，不另外 build）：只需重建 `formosan-ai-common`（`docker buildx bake -f docker-bake.hcl --load files`），**不需**重建 base 和 gpu。驗證：build 成功；檔案是否真的進了 image，由 2.2、3.3 在容器內 import 時一併驗證

## 2. asr 套用正規化（asr/）

- [x] 2.1 [asr/] `asr/app.py` 的 `generate_srt` 在過濾完 segments 之後，直接以 `language`（本來就是語言別代碼）呼叫 `normalize` 改寫每段的 `text`，再交給 `translate_segments` 和 `render_srt`。驗證：`tox -e flake8` 通過
- [x] 2.2 [asr/] `asr/Dockerfile` 的 `COPY --from=formosan-ai-common` 加上 `/app/orthography.py`，並依 5.3 加上 `/app/languages.py`。Docker：需要先完成 1.15 的 `formosan-ai-common`，再重建 asr image。驗證：`docker compose build asr` 成功，而且容器內執行 `python -c "import orthography, languages"` 不出錯
- [x] 2.3 [asr/] 本機或測試機啟動 asr，用 klokah 的南勢句子音檔（`https://web.klokah.tw/text/sound/<textId>/<autoId>.mp3`，轉成影片或直接上傳）選 `南勢阿美語` 辨識。驗證：SRT 的族語行不含 o、f、b、`’`、`'`，華語行有翻譯（2026-11-08 在 tshi5v100 驗證：沒有 klokah 音檔，改用 tests/data 的海岸阿美語影片，分別選南勢、海岸、撒奇萊雅，三個語別的族語行都沒有違規字母，華語行有翻譯）

## 3. mt 套用正規化（mt/）

- [x] 3.1 [mt/、tests/] 新增 `LANGUAGE_CODES`（NLLB 碼對語言別代碼，原本寫在 `mt/formosan_languages.py`，現在在 `common/languages.py`），涵蓋 `FORMOSAN_LANGUAGES_MAP` 的全部 42 個語別；照紅綠流程。先在 `tests/unit/` 加上測試：`LANGUAGE_CODES` 的 key 集合等於 `FORMOSAN_LANGUAGES_MAP` 的值集合，而且整份對照是 asr `TRANSLATION_LANGUAGE_CODES` 的反向，`tox -e test` 確認失敗；再實作；`tox -e test` 確認通過（`LANGUAGE_CODES` 後來改成由 `TRANSLATION_LANGUAGE_CODES` 反轉產生並搬進 `common/languages.py`，見 5.1）
- [x] 3.2 [mt/] `mt/app.py` 的 `translate()` 開頭執行 `text = normalize(text, LANGUAGE_CODES.get(src_lang))`，解碼後回傳 `normalize(translated, LANGUAGE_CODES.get(tgt_lang))`。驗證：`tox -e flake8` 通過
- [x] 3.3 [mt/] `mt/Dockerfile` 的 `COPY --from=formosan-ai-common` 加上 `/app/orthography.py`，並依 5.3 加上 `/app/languages.py`、移除 `formosan_languages.py`。Docker：需要先完成 1.15 的 `formosan-ai-common`，再重建 mt image。驗證：`docker compose build mt` 成功，而且容器內執行 `python -c "import orthography, languages"` 不出錯
- [x] 3.4 [mt/] 啟動 mt，「華語 ⮕ 族語」選 `阿美_南勢` 翻譯「蛋」「你好」，選 `阿美_海岸` 翻譯同樣兩句，選 `撒奇萊雅` 翻譯「你好」；「族語 ⮕ 華語」選 `阿美_海岸` 輸入 `nga’ay ho`。驗證：南勢與撒奇萊雅的輸出不含 o，南勢不含 f、b，海岸不含 u、v、b，撇號都是 `ʼ`（U+02BC）；族→華的輸入框仍顯示 `nga’ay ho`，而且有翻譯結果；按「合成語音」可以播放

## 4. 整合驗證（deploy/）

- [x] 4.1 [deploy/] 部署到測試機之前先跑基準線：`BASE_URL=https://tshi5v100.ithuankhoki.tw tox -e production_tests -- --report tests/production_tests/results/$(date +%F)-before-$(git rev-parse --short HEAD).json`
- [x] 4.2 [deploy/] 重建 `formosan-ai-common`、asr、mt、tts image 並部署到測試機，以同樣參數再跑一次（檔名用 `-after-`）。驗證：兩份報告的 `checks` 名稱集合相同、沒有任何 `ok` 從 true 變 false，再到測試機重做一次 2.3 與 3.4 的手動檢查，並確認 tts 的族別 Radio 有 16 個族別、預設選取「阿美」、切換族別後配音員與語別選項正常
- [x] 4.3 [openspec/] 執行 `openspec validate --all --strict`。驗證：全部通過
- [x] 4.4 [.travis.yml] PR 的 Travis 出現 `Unit tests` job 且通過（已確認：Python 3.10.19，48 個通過，`tox -e test` exit 0）

- [x] 4.5 [tests/production_tests/] asr 新增 `kako.wav` 書寫符號檢查：上傳 `tests/data/kako.wav`，南勢（`ami-x-iams`）與撒奇萊雅（`szy`）要得到 `kaku`，秀姑巒、海岸、馬蘭、恆春（`ami-x-skl`、`ami-x-pswl`、`ami-x-frng`、`ami-x-pld`）要得到 `kako`。實測：舊版正式機六個語別都是 `kaku`，測試機是預期結果，6 項都通過
- [x] 4.6 [deploy/] 以 4.5 之後的腳本，對正式機（舊版）跑一次 `-before-` 報告，對測試機跑一次 `-after-` 報告，比對 `checks`。結果（2026-11-08，`results/2026-11-08-before-ffba2a2.json` 對 `-after-`）：before 是正式機舊版，after 是測試機；共同檢查沒有任何 `ok` 從 true 變 false，只有四個 `kako.wav`（秀姑巒、海岸、馬蘭、恆春）從 false 變 true。兩份報告的檢查名稱不同：預設隨機抽樣的語別不同，而且正式站預設不跑 tts 同時合成。要名稱相同，兩邊都加 `--all-languages`（4.2 未打勾的原因）

## 5. 語別表搬到 common/，asr、mt、tts 共用（common/、asr/、mt/、tts/、tests/）

- [x] 5.1 [common/、asr/、mt/] 把 `asr/languages.py` 搬到 `common/languages.py`；把 `mt/formosan_languages.py` 的 `FORMOSAN_LANGUAGES_MAP`（名稱與順序不變）併進去；`LANGUAGE_CODES` 改由 `TRANSLATION_LANGUAGE_CODES` 反轉產生；新增 `ETHNICITIES`（`LANGUAGE_GROUPS` 的組名去掉「語」）。迴圈依 CLAUDE.md 用前置 `for`。刪除 `mt/formosan_languages.py`。驗證：`python -c` 確認 `FORMOSAN_LANGUAGES_MAP`、`LANGUAGE_CODES` 各 42 個，`ETHNICITIES` 有 16 個且等於 `FORMOSAN_LANGUAGES_MAP` 名稱前綴的集合；tts 的 `refs.yaml` 全部族別前綴都在 `ETHNICITIES` 裡
- [x] 5.2 [mt/、tts/] `mt/app.py` 改成 `from languages import ETHNICITIES, FORMOSAN_LANGUAGES_MAP, LANGUAGE_CODES`，刪掉自己算的 `ETHNICITIES`；`tts/app.py` 改成 `from languages import ETHNICITIES`，刪掉從 `g2p_object` 算的那一行。驗證：`tox -e flake8` 通過
- [x] 5.3 [common/、asr/、mt/、tts/] 四個 Dockerfile：`common/Dockerfile` 的 files stage 複製 `languages.py`（已含在 1.15）；asr、mt 的 `COPY --from=formosan-ai-common` 加上 `/app/languages.py`，並把 `languages.py`（asr）、`formosan_languages.py`（mt）從 `COPY --link` 那一行移除；tts 的 `COPY --from=formosan-ai-common` 加上 `/app/languages.py`。Docker：和 1.15、2.2、3.3 一起驗證，另外 `docker compose build tts` 成功，而且容器內 `python -c "import languages"` 不出錯
- [x] 5.4 [tests/] 單元測試：`tests/unit/test_mt_language_codes.py` 改成 `tests/unit/test_languages.py`（`LANGUAGE_CODES` 涵蓋 42 個語別且是 `TRANSLATION_LANGUAGE_CODES` 的反向、南勢對照、`ETHNICITIES` 順序與前綴一致）；`test_orthography.py` 的代碼防呆測試改名為 `test_language_rules_keys_match_language_groups_in_order`。驗證：`tox -e test` 通過（已確認：48 個通過）
- [x] 5.5 [tox.ini] `[testenv:test]` 的 `PYTHONPATH` 從 `common:asr:mt` 縮成 `common`：單元測試只 import `orthography` 與 `languages`。驗證：`tox -e test` 通過；暫時把 `PYTHONPATH` 清空，測試要因 `ModuleNotFoundError` 失敗，確認它真的有用到
- [x] 5.6 [tests/production_tests/] `check_asr.py`、`check_mt.py`、`check_mt_tts_playback.py` 的 `sys.path` 改指向 `common`，import 改成 `from languages import ...`；`README.md` 的語別表來源改成 `common/languages.py`。驗證：`python -c` 確認三個腳本 import 成功（腳本不可 import 各服務的 `app.py`）；`BASE_URL=... tox -e production_tests` 的結果見 4.1、4.2
- [x] 5.7 [openspec/] `specs/production-tests/spec.md` 的 delta 把兩個需求的語別表來源改成 `common/languages.py`。驗證：`openspec validate --all --strict`（併入 4.3）

## Workflow follow-up

- 正式機部署後，在 mt 用 `阿美_南勢` 翻譯一句，確認輸出用 u、v。
- 合併時注意和 `docker-shared-base-image` 在 `common/Dockerfile`、`asr/Dockerfile`、`mt/Dockerfile`、`tts/Dockerfile` COPY 行的衝突。
- `docker-image-layering` 主 spec 的場景列出 `formosan-ai-common` 的內容是 `/app/utils.py`、`/app/colors.py`、`/app/common_static/`，加了 `orthography.py` 與 `languages.py` 之後和實際不符；archive 時一併決定要不要補一份 delta。
- 審查通過後 archive 這個 change，並把 `formosan-orthography` sync 進主 specs。
