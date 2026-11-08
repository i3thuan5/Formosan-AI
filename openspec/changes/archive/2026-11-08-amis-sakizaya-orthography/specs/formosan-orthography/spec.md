# Spec Delta

## Purpose

讓 asr 與 mt 輸出或接收的阿美語、撒奇萊雅語文字，符合 113 年「原住民族語言書寫系統」為各語別指定的書寫符號（o/u、f/b/v），並把撇號統一成 `ʼ`。

## ADDED Requirements

### Requirement: 正規化以語言別代碼查規則，只涵蓋三組語別

書寫符號正規化 SHALL 以「族語及語言別名稱」的語言別代碼（如 `ami-x-iams`）查規則，只涵蓋三組：南勢阿美（`ami-x-iams`）、通用阿美（`ami-x-skl`、`ami-x-pswl`、`ami-x-frng`、`ami-x-pld`）、撒奇萊雅（`szy`）。其他語言別代碼，以及不是族語的代碼（如華語），文字 MUST 原樣保留。正規化是純文字處理，在 GPU 和 CPU 環境的結果相同。

#### Scenario: 其他語別不受影響

- **WHEN** 語言別代碼為 `tay-x-sql`，文字為 `lalu’ mu o Yumin`
- **THEN** 正規化後的文字仍為 `lalu’ mu o Yumin`

#### Scenario: 華語不受影響

- **WHEN** 代碼為 `zho_Hant`，文字為 `你好 o'`
- **THEN** 正規化後的文字仍為 `你好 o'`

### Requirement: mt 以語言別代碼查規則

mt 的語別目前用翻譯模型的代碼（如 `ami_Sout`）。mt SHALL 先把它換成對應的語言別代碼（如 `ami-x-iams`），再用語言別代碼查正規化規則。這個對照 MUST 涵蓋 mt 的全部 42 個語別，而且和 asr 的語言別代碼對翻譯模型代碼的對照互為反向。mt 對外的 gradio API 參數維持使用翻譯模型代碼。

#### Scenario: mt 南勢的代碼對照

- **WHEN** mt 的語別為 `ami_Sout`
- **THEN** 以 `ami-x-iams` 套用南勢阿美的規則

### Requirement: 南勢阿美的字母對應

語言別為南勢阿美時，SHALL 把 o 轉成 u，f 和 b 轉成 v。

#### Scenario: 南勢轉換 o 與 f

- **WHEN** 語言別代碼為 `ami-x-iams`，文字為 `fita’ol`
- **THEN** 正規化後為 `vitaʼul`

#### Scenario: 南勢轉換 b

- **WHEN** 語言別代碼為 `ami-x-iams`，文字為 `babahiyan`
- **THEN** 正規化後為 `vavahiyan`

### Requirement: 通用阿美的字母對應

語言別為秀姑巒、海岸、馬蘭、恆春阿美時，SHALL 把 u 轉成 o，v 和 b 轉成 f。

#### Scenario: 通用轉換 u 與 v

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `vitaʼul`
- **THEN** 正規化後為 `fitaʼol`

#### Scenario: 通用轉換 b

- **WHEN** 語言別代碼為 `ami-x-skl`，文字為 `babahi`
- **THEN** 正規化後為 `fafahi`

### Requirement: 撒奇萊雅的字母對應

語言別為撒奇萊雅時，SHALL 把 o 轉成 u，v 轉成 b。f 和 z、d MUST 保留不動。

#### Scenario: 撒奇萊雅轉換 o 與 v

- **WHEN** 語言別代碼為 `szy`，文字為 `davak abo`
- **THEN** 正規化後為 `dabak abu`

#### Scenario: 撒奇萊雅保留 f 與 z

- **WHEN** 語言別代碼為 `szy`，文字為 `fataan zazan`
- **THEN** 正規化後仍為 `fataan zazan`

### Requirement: 撇號統一為 ʼ

三組語別的 `’`（U+2019）和 `'`（U+0027）SHALL 統一轉成 `ʼ`（U+02BC），已經是 `ʼ` 的維持不變。阿美語的喉塞音 `^` 和會厭塞音是不同的音，MUST 保留不動。

#### Scenario: 撇號統一

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `nga’ay ho, ma'emin`
- **THEN** 正規化後為 `ngaʼay ho, maʼemin`

#### Scenario: 喉塞音保留

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `^orip`
- **THEN** 正規化後仍為 `^orip`

### Requirement: 保留大小寫與其他字元

正規化 SHALL 保留字母的大小寫：大寫字母轉成對應的大寫字母（如 `O`→`U`、`F`→`V`）。規則以外的字元（其他字母、數字、標點、空白、漢字） MUST 原樣保留。每個字元最多轉換一次，轉換後的文字不會再被套用其他規則；所以對同一語別重複正規化，結果不變。

#### Scenario: 大寫字母

- **WHEN** 語言別代碼為 `ami-x-iams`，文字為 `Ofad Pangcah`
- **THEN** 正規化後為 `Uvad Pangcah`

#### Scenario: 重複正規化結果不變

- **WHEN** 對 `ami-x-iams` 的文字正規化兩次
- **THEN** 第二次的結果和第一次相同

### Requirement: asr 辨識結果正規化

asr 產生字幕時 SHALL 以使用者選擇的語言別代碼，先正規化每段辨識文字。SRT 的「族語：」行和送進翻譯模型的文字 MUST 都是正規化後的版本。字幕段落數與時間軸 MUST 不受影響。

#### Scenario: 南勢辨識結果

- **WHEN** 語別為 `ami-x-iams`，Whisper 某段輸出為 `fita’ol`
- **THEN** SRT 該段的族語行為 `vitaʼul`，翻譯模型收到的也是 `vitaʼul`

### Requirement: mt 族語 ⮕ 華語輸入正規化

mt「族語 ⮕ 華語」SHALL 先依來源語別正規化使用者輸入的族語文字，再送進翻譯模型。輸入框顯示的內容 MUST 維持使用者原本輸入的文字。

#### Scenario: 使用者輸入規範撇號

- **WHEN** 來源語別為 `阿美_海岸`，使用者輸入 `nga’ay ho`
- **THEN** 翻譯模型收到 `ngaʼay ho`，輸入框仍顯示 `nga’ay ho`

### Requirement: mt 華語 ⮕ 族語輸出正規化

mt「華語 ⮕ 族語」SHALL 依目標語別正規化翻譯模型的輸出，「翻譯結果」顯示的就是正規化後的文字。gradio API 回傳的文字也 MUST 是正規化後的版本。

#### Scenario: 南勢翻譯結果

- **WHEN** 目標語別為 `阿美_南勢`，翻譯模型輸出 `fita’ol`
- **THEN** 「翻譯結果」顯示 `vitaʼul`
