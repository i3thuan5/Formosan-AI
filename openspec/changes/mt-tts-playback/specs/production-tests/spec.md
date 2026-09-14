## ADDED Requirements

### Requirement: 可指定受測主機的上線後檢查腳本

`production_tests/` SHALL 提供一支可直接執行的 Python 檢查腳本，以 `BASE_URL` 環境變數指定受測主機，預設為 `https://ai-labs.ilrdf.org.tw`。MT 位於 `{BASE_URL}/kari-seejiq-tnpusu-ai-hmjil/`，TTS 位於 `{BASE_URL}/hnang-kari-ai-asi-sluhay/`。腳本 SHALL 循序執行所有檢查（不併發），全部通過時 exit code 為 0，任一項失敗時為 1，並在結尾列出每個失敗項目與錯誤訊息。

#### Scenario: 對測試機執行

- **WHEN** 執行 `BASE_URL=https://<測試機> python production_tests/check_mt_tts_playback.py`
- **THEN** 所有呼叫都送往測試機

#### Scenario: 有檢查失敗

- **WHEN** 任一項檢查失敗
- **THEN** 腳本繼續跑完其餘檢查，最後列出失敗項目，exit code 為 1

### Requirement: 檢查 TTS API 約定

腳本 SHALL 確認 TTS 的 API 資訊中含有 `/synthesize`，且參數為 `language` 與 `text`。

#### Scenario: tts 尚未部署新版

- **WHEN** 受測主機的 TTS 沒有 `/synthesize`
- **THEN** 此項檢查失敗，訊息指出缺少 `/synthesize`

### Requirement: 檢查 42 個語別都能合成

腳本 SHALL 對 `mt/app.py` `FORMOSAN_LANGUAGES_MAP` 的每個語別直接呼叫 TTS `/synthesize`，句子取自 `tts/configs/refs.yaml` 中該語別第一位配音員的 `text`，每個語別都要回傳音檔。腳本 MUST 以靜態解析讀取 `mt/app.py`，MUST NOT import 它（避免載入翻譯模型）。

#### Scenario: 語別表與配音員設定不一致

- **WHEN** `FORMOSAN_LANGUAGES_MAP` 有某個語別在 `refs.yaml` 中找不到配音員
- **THEN** 該語別的檢查失敗，訊息列出語別名稱

### Requirement: 檢查已知 bug 回歸

腳本 SHALL 透過 TTS `/synthesize` 確認：以引號結尾的文字與含「」的文字能合成成功，以及不支援的語別會回傳錯誤。

#### Scenario: 引號 bug 回歸

- **WHEN** 以 `阿美_海岸` 合成 `Sowal sa ko singsi, "Ano dafak micodad kita."`
- **THEN** 回傳音檔；若出現 `Unknown characters` 則此項檢查失敗

#### Scenario: 不支援的語別

- **WHEN** 以不存在的語別呼叫 `/synthesize`
- **THEN** TTS 回傳錯誤時檢查通過；若回傳音檔則檢查失敗

### Requirement: 檢查 mt → tts 端到端

腳本 SHALL 對 `阿美_海岸`、`泰雅_萬大`、`魯凱_茂林`、`卡那卡那富`、`賽夏` 共 5 個語別，在同一個 MT session 先呼叫 `/to_formosan_languages` 切換族別，再呼叫 MT `/synthesize`，確認回傳音檔並印出每次耗時。耗時超過 15 秒時 SHALL 印出警告，但不算失敗。

#### Scenario: mt 容器連不到 tts

- **WHEN** MT 回傳「語音合成服務暫時無法使用」等錯誤
- **THEN** 該語別的端到端檢查失敗，並印出 MT 回傳的錯誤訊息

#### Scenario: 接近逾時

- **WHEN** 端到端合成成功，但耗時 16 秒
- **THEN** 檢查通過，並印出接近 20 秒逾時的警告

### Requirement: 檢查腳本說明文件

`production_tests/README.md` SHALL 說明安裝方式、執行指令、`BASE_URL` 用法、每項檢查的內容、預估耗時與會消耗受測主機 GPU，並註明應以與受測主機部署版本相同的 commit 執行，以及 tts 須先於 mt 部署。

#### Scenario: 新成員第一次執行

- **WHEN** 開發者只照 README 操作
- **THEN** 能在 devcontainer 中安裝相依套件並對指定主機完成一次檢查
