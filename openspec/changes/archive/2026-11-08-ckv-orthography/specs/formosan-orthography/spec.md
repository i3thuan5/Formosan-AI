# Spec Delta

## Purpose

讓 asr 與 mt 輸出或接收的噶瑪蘭語文字，符合 113 年「原住民族語言書寫系統」第 11 項：沒有 o（統一寫 u）、小舌擦音用大寫 R、其餘字母小寫、撇號統一為 `ʼ`。

## RENAMED Requirements

- FROM: `### Requirement: 正規化以語言別代碼查規則，只涵蓋三組語別`
- TO: `### Requirement: 正規化以語言別代碼查規則，只涵蓋有規範依據的語別`

## MODIFIED Requirements

### Requirement: 正規化以語言別代碼查規則，只涵蓋有規範依據的語別

書寫符號正規化 SHALL 以「族語及語言別名稱」的語言別代碼（如 `ami-x-iams`）查規則，只涵蓋四組：南勢阿美（`ami-x-iams`）、通用阿美（`ami-x-skl`、`ami-x-pswl`、`ami-x-frng`、`ami-x-pld`）、撒奇萊雅（`szy`）、噶瑪蘭（`ckv`）。其他語言別代碼，以及不是族語的代碼（如華語），文字 MUST 原樣保留。正規化是純文字處理，在 GPU 和 CPU 環境的結果相同。

#### Scenario: 其他語別不受影響

- **WHEN** 語言別代碼為 `tay-x-sql`，文字為 `lalu’ mu o Yumin`
- **THEN** 正規化後的文字仍為 `lalu’ mu o Yumin`

#### Scenario: 太魯閣不受影響

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `bowyak Away’`
- **THEN** 正規化後的文字仍為 `bowyak Away’`

#### Scenario: 華語不受影響

- **WHEN** 代碼為 `zho_Hant`，文字為 `你好 o'`
- **THEN** 正規化後的文字仍為 `你好 o'`

### Requirement: 撇號統一為 ʼ

四組語別的 `’`（U+2019）和 `'`（U+0027）SHALL 統一轉成 `ʼ`（U+02BC），已經是 `ʼ` 的維持不變。阿美語的喉塞音 `^` 和會厭塞音是不同的音，MUST 保留不動。

#### Scenario: 撇號統一

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `nga’ay ho, ma'emin`
- **THEN** 正規化後為 `ngaʼay ho, maʼemin`

#### Scenario: 噶瑪蘭撇號統一

- **WHEN** 語言別代碼為 `ckv`，文字為 `ʼnay ’ta 'qi`
- **THEN** 正規化後為 `ʼnay ʼta ʼqi`

#### Scenario: 喉塞音保留

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `^orip`
- **THEN** 正規化後仍為 `^orip`

### Requirement: 保留大小寫與其他字元

除了噶瑪蘭，正規化 SHALL 保留字母的大小寫：大寫字母轉成對應的大寫字母（如 `O`→`U`、`F`→`V`）。規則以外的字元（其他字母、數字、標點、空白、漢字） MUST 原樣保留。除了噶瑪蘭，每個字元最多轉換一次，轉換後的文字不會再被套用其他規則。所有語別對同一語別重複正規化，結果都不變。噶瑪蘭的大小寫與套用順序見「噶瑪蘭的字母與大小寫」。

#### Scenario: 大寫字母

- **WHEN** 語言別代碼為 `ami-x-iams`，文字為 `Ofad Pangcah`
- **THEN** 正規化後為 `Uvad Pangcah`

#### Scenario: 重複正規化結果不變

- **WHEN** 對 `ami-x-iams` 的文字正規化兩次
- **THEN** 第二次的結果和第一次相同

## ADDED Requirements

### Requirement: 噶瑪蘭的字母與大小寫

語言別為噶瑪蘭（`ckv`）時，SHALL 依序：先把全部字母轉成小寫，再把 r 轉成 R，所以小舌擦音 R 以大寫保留，其他大寫（句首、專有名詞）都變小寫；統一撇號；最後把 o 轉成 u，但緊接在 u 後面的 o MUST 保留，因為 `uo` 念 [uo]（如借詞 `suykuo`「水果」），不是 u 的同位音。f、b、z、d 與其他字母 MUST 保留不動。重複正規化結果不變。判斷「緊接在 u 後面」時，看的是 o→u 這一步之前的文字，所以連續的 o 都會轉成 u。

#### Scenario: 噶瑪蘭轉換 o

- **WHEN** 語言別代碼為 `ckv`，文字為 `Oba qoR`
- **THEN** 正規化後為 `uba quR`

#### Scenario: 噶瑪蘭保留 u 後面的 o

- **WHEN** 語言別代碼為 `ckv`，文字為 `suykuo suykuoan Uo`
- **THEN** 正規化後為 `suykuo suykuoan uo`

#### Scenario: 噶瑪蘭連續的 o

- **WHEN** 語言別代碼為 `ckv`，文字為 `qoo`
- **THEN** 正規化後為 `quu`

#### Scenario: 噶瑪蘭轉換 r

- **WHEN** 語言別代碼為 `ckv`，文字為 `saribung serai`
- **THEN** 正規化後為 `saRibung seRai`

#### Scenario: 噶瑪蘭保留 R

- **WHEN** 語言別代碼為 `ckv`，文字為 `Rabas naRin tebuquR`
- **THEN** 正規化後為 `Rabas naRin tebuquR`

#### Scenario: 噶瑪蘭專有名詞不大寫

- **WHEN** 語言別代碼為 `ckv`，文字為 `PateRis ika ti ukud`
- **THEN** 正規化後為 `pateRis ika ti ukud`

#### Scenario: 噶瑪蘭保留 b 與 z

- **WHEN** 語言別代碼為 `ckv`，文字為 `babuzan`
- **THEN** 正規化後仍為 `babuzan`
