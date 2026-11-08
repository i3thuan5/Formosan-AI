# Spec Delta

## Purpose

讓 asr 與 mt 輸出或接收的太魯閣語文字，把不是虛詞也不是 `ow` 的 o 改寫成 u，符合 113 年「原住民族語言書寫系統」第 12 項注意事項 4、6 與族語辭典的實際拼寫。

## ADDED Requirements

### Requirement: 太魯閣的 o 規則

語言別為太魯閣（`trv-x-truku`）時，SHALL 依序判斷每個 o：

1. 整個詞是 `o`、`do`、`hido`、`kido`、`empeehido` 時（不分大小寫），詞裡的 o MUST 保留。「整個詞」指前後都不是英文字母，所以撇號、引號、連字號、數字、漢字、空白與標點都算詞界。
2. o 緊接著 w 時（不分大小寫），MUST 保留。
3. 其他的 o SHALL 轉成 u，大寫 `O` 轉成 `U`。

其他字母、大小寫與撇號 MUST 保留不動。這個規則只做 o→u，u、`uw` MUST 不被改成 o、`ow`。重複正規化結果不變。太魯閣的大小寫保留與其他字元的處理見「保留大小寫與其他字元」。

#### Scenario: 太魯閣轉換 o

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `holing sapoh biyoq`
- **THEN** 正規化後為 `huling sapuh biyuq`

#### Scenario: 太魯閣轉換大寫 O

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `Holing TODA`
- **THEN** 正規化後為 `Huling TUDA`

#### Scenario: 太魯閣保留虛詞

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `o do hido kido empeehido`
- **THEN** 正規化後仍為 `o do hido kido empeehido`

#### Scenario: 太魯閣保留大寫的虛詞

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `Kido, mhuya da? DO O Empeehido`
- **THEN** 正規化後仍為 `Kido, mhuya da? DO O Empeehido`

#### Scenario: 太魯閣的虛詞以整個詞比對

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `Toda nhapuydo`
- **THEN** 正規化後為 `Tuda nhapuydu`

#### Scenario: 太魯閣的詞界只看英文字母

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `‘do’ doʼ do你好 o-do`
- **THEN** 正規化後仍為 `‘do’ doʼ do你好 o-do`

#### Scenario: 太魯閣保留 ow 與撇號

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `bowyak Away’`
- **THEN** 正規化後仍為 `bowyak Away’`

#### Scenario: 太魯閣保留大寫的 ow

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `Mowsa OW Ow`
- **THEN** 正規化後仍為 `Mowsa OW Ow`

#### Scenario: 太魯閣不做反向修正

- **WHEN** 語言別代碼為 `trv-x-truku`，文字為 `buwyak u du`
- **THEN** 正規化後仍為 `buwyak u du`

#### Scenario: 太魯閣重複正規化結果不變

- **WHEN** 對 `trv-x-truku` 的文字 `Toda bowyak do Holing` 正規化兩次
- **THEN** 第二次的結果和第一次相同，都是 `Tuda bowyak do Huling`

## MODIFIED Requirements

### Requirement: 撇號統一為 ʼ

南勢阿美語、秀姑巒阿美語、海岸阿美語、馬蘭阿美語、恆春阿美語、撒奇萊雅語、噶瑪蘭語的 `’`（U+2019）和 `'`（U+0027）SHALL 統一轉成 `ʼ`（U+02BC），已經是 `ʼ` 的維持不變。太魯閣語的撇號 MUST 保留不動。阿美語的喉塞音 `^` 和會厭塞音是不同的音，MUST 保留不動。

#### Scenario: 撇號統一

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `nga’ay ho, ma'emin`
- **THEN** 正規化後為 `ngaʼay ho, maʼemin`

#### Scenario: 噶瑪蘭撇號統一

- **WHEN** 語言別代碼為 `ckv`，文字為 `ʼnay ’ta 'qi`
- **THEN** 正規化後為 `ʼnay ʼta ʼqi`

#### Scenario: 喉塞音保留

- **WHEN** 語言別代碼為 `ami-x-pswl`，文字為 `^orip`
- **THEN** 正規化後仍為 `^orip`

### Requirement: 正規化以語言別代碼查規則，只涵蓋有規範依據的語別

書寫符號正規化 SHALL 以「族語及語言別名稱」的語言別代碼（如 `ami-x-iams`）查規則，只涵蓋南勢阿美語（`ami-x-iams`）、秀姑巒阿美語（`ami-x-skl`）、海岸阿美語（`ami-x-pswl`）、馬蘭阿美語（`ami-x-frng`）、恆春阿美語（`ami-x-pld`）、撒奇萊雅語（`szy`）、噶瑪蘭語（`ckv`）、太魯閣語（`trv-x-truku`）。其他語言別代碼，以及不是族語的代碼（如華語），文字 MUST 原樣保留。正規化是純文字處理，在 GPU 和 CPU 環境的結果相同。

#### Scenario: 其他語別不受影響

- **WHEN** 語言別代碼為 `tay-x-sql`，文字為 `lalu’ mu o Yumin`
- **THEN** 正規化後的文字仍為 `lalu’ mu o Yumin`

#### Scenario: 賽德克不受影響

- **WHEN** 語言別代碼為 `trv-x-trk`，文字為 `holing Toda`
- **THEN** 正規化後的文字仍為 `holing Toda`

#### Scenario: 華語不受影響

- **WHEN** 代碼為 `zho_Hant`，文字為 `你好 o'`
- **THEN** 正規化後的文字仍為 `你好 o'`
