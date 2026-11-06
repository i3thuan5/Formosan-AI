# Design

## Context

規則與動機見 proposal.md，可觀察的行為見 `specs/formosan-orthography/spec.md`。這裡只記錄實作上會用到的現況：

- 語言別代碼以 `kithann/規範/族語及語言別名稱 - 語言別名稱.csv` 為準（42 個語別，如 `ami-x-iams`、`szy`），asr 和 mt 之後都會改用這套代碼。
- asr 的語別碼（`asr/languages.py` 的 `LANGUAGE_GROUPS`）和 CSV 的 42 個代碼完全相同。`TRANSLATION_LANGUAGE_CODES` 則把它對應到 NLLB 碼（`ami_Sout` 等）。
- mt 的 `translate(text, src_lang, tgt_lang)` 收到的是 NLLB 碼，華語固定是 `zho_Hant`。兩個翻譯方向共用這個函式。mt 目前沒有語言別代碼。
- asr、mt 的 image 從 `formosan-ai-common` 逐檔複製共用檔案到 `/app`（例如 `utils.py`、`colors.py`），服務以 `/app` 為工作目錄，所以共用模組是用頂層名稱 import 的。
- repo 目前沒有單元測試環境，tox 只有 lint、lockcheck、production_tests。
- klokah 原始資料在 `kithann/` 底下，`.gitignore` 忽略這個目錄，所以進不了 repo。
- NLLB 詞表有 `ʼ`（U+02BC），沒有 `’`（U+2019），`’` 會被切成 `<unk>`。實測 mt 目前輸出的撇號就是 `ʼ`。

## Goals / Non-Goals

**Goals：**

- 一組小規則函式加一張語別設定表、一個 `normalize()` 入口，由 asr 和 mt 共用。
- 模組只依賴 Python 標準函式庫，測試不用載入任何模型或 GPU 套件。

**Non-Goals：**

- 不做詞層級的例外清單（例如專有名詞不轉）。目前接受字母層級轉換可能誤傷少數借詞。

## Decisions

### 1. 新增 `common/orthography.py`，以語言別代碼為 key

模組提供 `normalize(text, language)`：`language` 是語言別代碼，不在語別設定表裡的代碼（包括 `None`、華語）就原樣回傳。規則怎麼拆、語別怎麼設定見決定 3。

- **為什麼用語言別代碼**：這是官方的語言別名稱表，asr 和 mt 之後都會換成這套。規則表先用它，等 mt 全面換碼時，規則表和 asr 都不用改。asr 現在就能直接用。
- **替代方案**：用 NLLB 碼當 key。現在 mt 寫起來比較省事，但 mt 換碼時規則表也得跟著換，而且 asr 要多轉一次碼，所以不採用。

### 2. mt 新增 NLLB 碼對語言別代碼的對照

在 `mt/formosan_languages.py` 新增 `LANGUAGE_CODES`，以 NLLB 碼為 key、語言別代碼為值，涵蓋 `FORMOSAN_LANGUAGES_MAP` 的全部 42 個語別。

- **為什麼 42 個都列**：只列三組語別就夠這次用，但之後 mt 全面換碼時，本來就需要完整的對照。現在列完整，測試也能直接檢查它和 asr 的 `TRANSLATION_LANGUAGE_CODES` 互為反向，有任何漏列或打錯都會被抓到。
- **替代方案**：mt 直接 import asr 的 `TRANSLATION_LANGUAGE_CODES` 再反查。但 mt 和 asr 是兩個獨立的 image，彼此沒有對方的檔案，所以不採用。

### 3. 每個規則拆成一個小函式，每個語別設定要套用哪幾個

`common/orthography.py` 把每個規則寫成獨立的小函式，輸入輸出都是 `str`：

- 字母對應：`o2u()`、`u2o()`、`f2v()`、`b2v()`、`v2f()`、`b2f()`、`v2b()`。
- 撇號：`quote_2_u02bc()`，把 `’`、`'` 轉成 `ʼ`。
- 每個字母函式內部用 `str.maketrans` / `str.translate` 做，小寫的對應寫一次，大寫的對應由程式自動產生，所以大小寫保留（`O`→`U`）不用每個函式各寫一遍。

語別設定是一張 dict，key 是語言別代碼，value 是要依序套用的函式 list：

```python
LANGUAGE_RULES = {
    'ami-x-iams': [o2u, f2v, b2v, quote_2_u02bc],
    'ami-x-skl': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-pswl': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-frng': [u2o, v2f, b2f, quote_2_u02bc],
    'ami-x-pld': [u2o, v2f, b2f, quote_2_u02bc],
    'tay-x-sql': [],
    'tay-x-sul': [],
    # ... 依 CSV 的順序，一路列到 szy 之前
    'szy': [o2u, v2b, quote_2_u02bc],
    'trv-x-td': [],
    # ... 一路列到最後的 xnb
}
```

- 通用阿美的四個語別各自列一份 list，不共用變數，內容目前相同。
- 42 個語言別代碼全部列在表裡，順序照「族語及語言別名稱 - 語言別名稱.csv」的「顯示排序先後」（和 asr `LANGUAGE_GROUPS` 的順序相同），沒有規則的語別 list 留空。這樣「這個語別有沒有規則」是看得見的設定，日後要補規則，直接在那一行填函式，不用新增 key；沒有規則的語別走同一條路徑，套用空 list，文字不變。

`normalize(text, language)` 取出 list，依序套用；查不到（`None`、`zho_Hant` 這類不是語言別代碼的值）就原樣回傳；list 是空的（沒有規則的語別）結果也是原樣。

- **為什麼這樣做**：新增或調整一個語別，只要改它自己那一行，不會影響別的語別（例如日後秀姑巒要多一條規則，不會連帶改到海岸）；每個函式可以單獨測試；之後要支援別的族語（例如只需要統一撇號的語別）直接組合現有函式，缺的規則再補一個小函式。
- **不連鎖、重複正規化結果不變**：spec 要求每個字元最多轉換一次。逐函式依序套用不會自動保證這點，所以改由「同一個 list 裡，任何函式的輸出字元都不是另一個函式的輸入字元」來保證。目前三組都滿足（南勢的來源是 o、f、b，目標是 u、v；通用的來源是 u、v、b，目標是 o、f；撒奇萊雅的來源是 o、v，目標是 u、b），所以 list 內的順序不影響結果。測試要對每個語別驗證兩件事：用含全部字母的字串正規化兩次，結果與一次相同；list 反過來套用，結果與正著套用相同。只看第一項抓不到環狀規則（例如 `f2v` 加 `v2f`，一次和兩次的結果剛好一樣），所以要加上第二項。以後新增函式若造成連鎖，這兩個測試至少有一個會失敗。
- **替代方案**：每組語別各建一張 `str.maketrans` 大表一次轉完。每個字元天然只轉一次，但規則綁在各語別的表裡，通用阿美和南勢的 u/o 規則互相看不到，也不能單獨測試或重用，所以不採用。依序呼叫多次 `str.replace` 同樣會有連鎖替換風險（例如先 b→f 再 f→v），也不採用。正規表示式在這裡沒有額外好處。

### 4. 套用位置

- **asr**：在 `generate_srt` 過濾完 segments 之後，直接用 `language`（已經是語言別代碼）就地改寫每段的 `text`。之後的 `translate_segments` 與 `render_srt` 自然拿到轉換後的文字，不用改這兩個函式。
- **mt**：在 `translate()` 開頭執行 `normalize(text, LANGUAGE_CODES.get(src_lang))`，解碼後再執行 `normalize(result, LANGUAGE_CODES.get(tgt_lang))`。
  - 華語 `zho_Hant` 不在 `LANGUAGE_CODES` 裡，查到 `None` 就不轉換，所以不用判斷翻譯方向：族→華時只有輸入會被轉換，華→族時只有輸出會被轉換。
  - 輸入框的內容是 Gradio 元件的值，`translate()` 只改它自己的區域變數，輸入框顯示不受影響。
- **替代方案**：在 asr 的 `translate_segments` 內部才轉換。這樣 SRT 顯示的文字和翻譯模型收到的文字會不一致，所以不採用。

### 5. 測試：`tests/unit/` 加上 `tox -e test`

- 新增 `[testenv:test]`：依賴 pytest，`PYTHONPATH` 指向 `common`、`asr`、`mt`，指令是 `pytest tests/unit`。
- Travis 新增 `Unit tests` job 執行 `tox -e test`，和 lockcheck 等 job 並列。
- 開發流程採紅綠流程：每個功能先寫測試並確認失敗，再實作，做完立刻確認通過，一次只做一個功能（見 tasks.md 開頭）。寫完就會通過的測試要暫時弄壞被測的東西，確認它真的會失敗。
- 測試內容：
  - spec 的每個 scenario 各對應一個測試。
  - 每個規則函式（`o2u()`、`quote_2_u02bc()` 等）各有單獨的測試，包含大小寫。
  - **冪等測試與順序無關測試**：對每個語別，用含全部相關字母的字串正規化兩次，結果必須相同；list 反過來套用，結果也必須相同。兩者合起來擋住規則連鎖。
  - **klokah 不動點測試**：klokah 本身就符合目標拼法，所以先把句子裡的 `’`、`'` 換成 `ʼ` 當作預期結果，正規化後必須完全一樣。
  - **代碼防呆**：語別設定表的 key 必須等於 asr `LANGUAGE_GROUPS` 的全部語言別代碼（42 個），順序也要相同：少列一個、多列一個、打錯碼或順序不同都會失敗，避免規則靜靜失效。mt 的 `LANGUAGE_CODES` 必須涵蓋 `FORMOSAN_LANGUAGES_MAP` 的全部值，而且是 asr `TRANSLATION_LANGUAGE_CODES` 的反向。`asr/languages.py` 和 `mt/formosan_languages.py` 都沒有依賴其他套件，可以直接 import。CSV 在被 git 忽略的 `kithann/` 底下，CI 讀不到，所以以 asr 的代碼當作 CSV 的替身。
- 測試程式依 CLAUDE.md 的規定，用前置的 `for` 迴圈，不用 list comprehension。

### 6. 測試素材：`tests/data/klokah_orthography.csv`

- 從 `kithann/E樂園/6.文化篇/klokah_cuture_sentence.csv` 節錄。
- 欄位：`language`（語言別代碼）、`sentence`。
- 範圍：南勢（dialectId 1）、秀姑巒、海岸、馬蘭、恆春（2–5）、撒奇萊雅（43），每個 dialectId 各取約 20 句。
- 排除會讓不動點測試失敗的句子。目前已知只有撒奇萊雅中那個孤立的 `o`，用另一句取代。
- `tests/data/README.md` 要註明來源、節錄方式與用途。

### 7. Docker

- `common/Dockerfile` 的 files stage：`COPY colors.py utils.py orthography.py /app/`。
- asr、mt 的 Dockerfile：在 `COPY --from=formosan-ai-common` 那一行加上 `/app/orthography.py`。
- asr-kaldi 和 tts 不用改。

## Risks / Trade-offs

- **字母層級轉換會誤傷借詞與專有名詞**（例如通用阿美裡 b 開頭的外來詞會變成 f）→ 使用者已接受。klokah 的通用阿美資料裡 b、v 都是 0，影響有限。
- **ASCII `'` 當引號用時也會轉成 `ʼ`** → 接受。在這三組語別裡，`'` 幾乎都是喉音符號。
- **模型漏掉撇號時無法補回** → 不在範圍內，proposal 已列為非目標。
- **使用者的翻譯結果會改變**：輸入 `’` 原本會被切成 `<unk>`，現在 NLLB 能看到撇號 → 這是預期的改善。production tests 只檢查非空、也不比對內容，不受影響。
- **和 `docker-shared-base-image` 改到同一行 COPY** → 哪個 change 後合併，就在合併時解決衝突。改動只是在檔案清單加一個檔名。

## Migration Plan

1. 照一般流程重建 `formosan-ai-common`、asr、mt 的 image，然後部署。
2. 部署前後各跑一次 `BASE_URL=... tox -e production_tests -- --report <檔名>`，比對結果。
3. 部署後在 mt 用 `阿美_南勢` 翻譯幾句，確認輸出是 u、v。
4. 回滾：還原 commit 並重新部署。這個 change 沒有資料或模型異動，不需要額外的回滾步驟。
