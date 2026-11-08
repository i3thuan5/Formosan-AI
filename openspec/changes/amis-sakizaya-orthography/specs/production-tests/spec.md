# Spec Delta

## Purpose

語別表搬到 `common/languages.py` 之後，production tests 的語別表來源路徑跟著改；檢查的行為不變。

## MODIFIED Requirements

### Requirement: 檢查 mt 雙向翻譯輸出

腳本 SHALL 以 asr 素材的一句族語（`sosowalen ako itiya:ayho a ʼorip niyam`）呼叫 mt `/translate`（族語到華語），並以一句華語（`我們以前的生活`）呼叫 `/translate_1`（華語到族語）。兩個方向的語別都是 Radio，呼叫前 MUST 在同一個 session 先切換族別：族語到華語呼叫 `/to_zh_languages`，華語到族語呼叫 `/to_formosan_languages`。語別為阿美_海岸（`ami_Coas`）時，兩個方向 MUST 回傳非空字串且不等於輸入；其他語別只要求回傳字串。語別表 SHALL 讀自 `common/languages.py`。

#### Scenario: 阿美_海岸雙向翻譯有內容

- **WHEN** 分別呼叫 `/translate` 與 `/translate_1`
- **THEN** 兩者都回傳非空字串，且與輸入不同

#### Scenario: 隨機語別

- **WHEN** 以隨機抽到的其他語別呼叫兩個方向
- **THEN** 呼叫成功且回傳字串，空字串印警告

### Requirement: 檢查語別都能合成

腳本 SHALL 預設從 `common/languages.py` 的 `FORMOSAN_LANGUAGES_MAP` 隨機抽樣 5 個語別，指定 `--all-languages` 參數時改為全部語別，對每個受測語別直接呼叫 TTS `/synthesize`，句子取自 `tts/configs/refs.yaml` 中該語別第一位配音員的 `text`，每個受測語別都要回傳音檔。語別表 SHALL 獨立於 `mt/app.py`，腳本 MUST NOT import `mt/app.py`（避免載入翻譯模型）。

#### Scenario: 預設隨機抽樣

- **WHEN** 不帶參數執行腳本
- **THEN** 第 2 項只合成隨機抽出的 5 個語別，並印出抽到哪些語別

#### Scenario: 掃描全部語別

- **WHEN** 執行腳本時加上 `--all-languages`
- **THEN** 第 2 項合成 `FORMOSAN_LANGUAGES_MAP` 的全部語別

#### Scenario: 語別表與配音員設定不一致

- **WHEN** 加上 `--all-languages` 執行，且 `FORMOSAN_LANGUAGES_MAP` 有某個語別在 `refs.yaml` 中找不到配音員
- **THEN** 該語別的檢查失敗，訊息列出語別名稱
