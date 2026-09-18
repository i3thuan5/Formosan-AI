## MODIFIED Requirements

### Requirement: 既有 API 維持不變

TTS 既有的 `/default_speaker_tts`、`/custom_speaker_tts` 等 API 的名稱與參數 SHALL 維持不變。

#### Scenario: 延遲測試仍可呼叫舊 API

- **WHEN** `tests/grafana-k6/tts.js` 以 `ref="阿美_秀姑巒_女聲1"` 呼叫 `/default_speaker_tts`
- **THEN** 回傳音檔
