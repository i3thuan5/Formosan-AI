# Proposal

## Why

113 年核定的「原住民族語言書寫系統」對阿美語與撒奇萊雅語的 o/u、f/b/v 有指定書寫符號：阿美語通用版（秀姑巒、海岸、馬蘭、恆春）用 o、f，南勢版用 u、v（或 b，不混用），撒奇萊雅用 u、b。目前 asr（Whisper）與 mt（NLLB）的族語文字都是模型直接吐出的結果，沒有任何後處理，同一個語別可能混用兩套拼法。例如 Kaldi 詞表裡阿美語有 1,570 組詞同時存在 u 與 o 兩種拼法，撒奇萊雅詞表也有 2,766 個含 o 的詞；Whisper 的訓練資料來源相近，輸出拼法同樣不保證。撇號也不一致：規範寫 `’`（U+2019），但 NLLB 詞表沒有這個字元，使用者輸入 `’` 會被切成 `<unk>`，撇號就這樣不見了；NLLB 本身輸出的是 `ʼ`（U+02BC）。

## What Changes

- 新增族語書寫符號正規化規則，以「族語及語言別名稱 - 語言別名稱.csv」的語言別代碼查規則，只涵蓋三組語別：
  - 南勢阿美（`ami-x-iams`）：o→u、f→v、b→v。南勢選 v 是比照族語 E 樂園（klokah）南勢教材的寫法，該教材完全不用 b。
  - 通用阿美（`ami-x-skl`、`ami-x-pswl`、`ami-x-frng`、`ami-x-pld`）：u→o、v→f、b→f。
  - 撒奇萊雅（`szy`）：o→u、v→b；f（借詞用）保留。
  - 三組共同：撇號 `’`、`'` 統一為 `ʼ`（U+02BC）；阿美語的喉塞音 `^` 和會厭塞音是不同的音，保留不動。
  - 大小寫保留（如 `O`→`U`）。
- 語言別代碼：asr 和 mt 之後都會改用 CSV 的語言別代碼。asr 現在的 Whisper 語別碼本來就是這套代碼，可以直接查規則。mt 目前用 NLLB 碼（`ami_Sout` 等），所以先新增 NLLB 碼對語言別代碼的對照，轉成語言別代碼後再查規則；mt 對外 API 的參數這次不改。
- 語別表共用：`asr/languages.py` 搬到 `common/languages.py`，`mt/formosan_languages.py` 取消，asr、mt、tts 都讀 `common/languages.py`。mt 的語別顯示名稱表（`FORMOSAN_LANGUAGES_MAP`）、NLLB 碼對語言別代碼的對照（`LANGUAGE_CODES`），以及 mt 與 tts 共用的族別清單（`ETHNICITIES`，由 `LANGUAGE_GROUPS` 的組名去掉「語」產生）都放在這個檔案，tts 的 `ETHNICITIES` 不再從 g2p 語別名稱算出來。
- asr：辨識結果先正規化，再用來產生 SRT 的族語行並送進 NLLB 翻譯。
- mt「族語 ⮕ 華語」：使用者輸入先正規化，再送進 NLLB。輸入框的顯示內容不變。
- mt「華語 ⮕ 族語」：NLLB 輸出正規化後才顯示在「翻譯結果」，合成語音也是用這份文字。
- 新增單元測試，以 `tox -e test` 執行；測試素材是從 klokah 節錄、放在 `tests/data/` 的南勢、通用、撒奇萊雅句子。

## 非目標

- 不處理 asr-kaldi。它的模型只分族別，沒有語別，現有的 `u→o` 維持原樣。
- 不處理其他 13 族的拼法與撇號，即使 NLLB 對它們的 `’` 一樣會切成 `<unk>`。
- 不處理撒奇萊雅的 z/d：規範要求尊重兩種寫法。
- 不處理 NLLB 輸出時撇號遺漏的問題：撇號一旦遺漏，後處理補不回來。
- 不重新訓練或微調任何模型，不修改 gradio API 的簽名與 `api_name`。
- 不把 mt 的 UI 與 API 全面換成語言別代碼：這次只在 mt 內部加上對照，全面替換另開 change。
- 不改 mt 語別顯示名稱（`阿美_海岸` 等）與各族別下的語別順序：名稱要和 tts 的 g2p 語別名稱一致（mt 呼叫 tts 時直接傳這個名稱），所以在 `common/languages.py` 裡仍是明列的表，不從 CSV 的語別名稱推算（CSV 寫「霧臺」，tts 寫「霧台」）。
- 不修改 tts：tts 本身已經會把各種撇號統一成 `ʼ`。

## Capabilities

### New Capabilities

- `formosan-orthography`：族語文字輸出與輸入的書寫符號正規化，包括以語言別代碼查規則、涵蓋哪些語別、每組的字母對應、撇號與大小寫規則，以及 asr、mt 在哪個環節套用。

### Modified Capabilities

- `production-tests`：兩個需求原本規定語別表讀自 `mt/formosan_languages.py`，檔案搬走後改讀 `common/languages.py`；檢查行為不變。

`mt-speech-playback` 規定合成語音用「翻譯結果」當下的文字，正規化發生在文字顯示之前，所以該需求不受影響。`production-tests` 不比對文字內容。

## Impact

- **common/**：新增正規化模組 `orthography.py`，並收進從 `asr/languages.py` 搬來的 `languages.py`；`common/Dockerfile` 的 files stage 要把兩個檔案都加進複製清單。
- **asr/**：`app.py` 在 `model.transcribe` 之後、`translate_segments` 與 `render_srt` 之前套用正規化；`Dockerfile` 要從 `formosan-ai-common` 複製 `orthography.py` 與 `languages.py`（不再從 asr 目錄複製）。
- **mt/**：刪除 `formosan_languages.py`，改 import `languages`（`FORMOSAN_LANGUAGES_MAP`、`LANGUAGE_CODES`、`ETHNICITIES`）；`app.py` 的 `translate` 依翻譯方向正規化輸入或輸出；`Dockerfile` 要複製 `orthography.py` 與 `languages.py`。
- **tts/**：`app.py` 的 `ETHNICITIES` 改 import 自 `languages`；`Dockerfile` 要複製 `languages.py`。族別 Radio 的順序從 Unicode 排序改成 `LANGUAGE_GROUPS` 的順序（CSV 的顯示排序）。
- **tests/**：`tests/data/` 新增 klokah 節錄句子並更新 README；新增單元測試；`tests/production_tests/` 的腳本與 README 改讀 `common/languages.py`。
- **tox.ini**：新增 `testenv:test`，`PYTHONPATH` 只需要 `common`（語別表搬進 common 之後，單元測試不再需要 `asr`、`mt` 的路徑）。
- **向下相容**：gradio API 參數與回傳型別不變。三組語別的回傳文字可能和之前不同（字母與撇號），其他語別完全不變。
- **和進行中的 change 重疊**：`docker-shared-base-image` 也在改 `common/Dockerfile`、`asr/Dockerfile`、`mt/Dockerfile` 的 COPY 行，合併時要注意衝突。
