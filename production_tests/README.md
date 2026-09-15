# 上線後檢查

對**已部署的主機**（正式機或測試機）實際呼叫 API，確認服務之間串得起來。
和 [grafana-k6/](../grafana-k6/) 的差別：這裡檢查「功能對不對」，k6 測「延遲多少」。

| 腳本 | 檢查對象 |
| --- | --- |
| `check_mt_tts_playback.py` | MT「華語 ⮕ 族語」合成語音，以及 TTS `/synthesize` API |

## 一、安裝

```bash
pip install -r production_tests/requirements.txt
```

套件版本寫在 `requirements.in`，改完後用 `pip-compile production_tests/requirements.in` 更新 `requirements.txt`。
`gradio_client` 的版本要和 `mt/requirements.txt` 一致，測到的呼叫方式才和 mt 相同。

## 二、執行

在 repo 根目錄執行：

```bash
# 預設檢查正式站 https://ai-labs.ilrdf.org.tw
python production_tests/check_mt_tts_playback.py

# 檢查測試機
BASE_URL=https://<測試機網域> python production_tests/check_mt_tts_playback.py

# 第 2 項掃全部語別（預設隨機抽 5 個）
python production_tests/check_mt_tts_playback.py --all-languages
```

- 受測主機要和正式站一樣，以 `{BASE_URL}/kari-seejiq-tnpusu-ai-hmjil/`（MT）與
  `{BASE_URL}/hnang-kari-ai-asi-sluhay/`（TTS）提供服務。
- 全部通過時 exit code 為 `0`，任一項失敗為 `1`，失敗項目會在最後列出。
- **請用和受測主機部署版本相同的 commit 執行**：語別表讀自本地的 `mt/formosan_languages.py`，
  測試句讀自本地的 `tts/configs/refs.yaml`，版本不同時結果可能對不上。

## 三、檢查項目

| 項目 | 內容 |
| --- | --- |
| 1. TTS API 約定 | TTS 有 `/synthesize`，參數為 `language`、`text` |
| 2. TTS 合成語別 | 預設從 `mt/formosan_languages.py` 隨機抽 5 個語別，加 `--all-languages` 則全部 42 個；直接呼叫 TTS `/synthesize`，句子用 `refs.yaml` 該語別第一位配音員的 `text`，都要回傳音檔 |
| 3. 已知 bug 回歸 | 以引號結尾、含「」的文字要能合成；不存在的語別要回傳錯誤 |
| 4. mt → tts 端到端 | `阿美_海岸`、`泰雅_萬大`、`魯凱_茂林`、`卡那卡那富`、`賽夏` 各自先呼叫 MT `/to_formosan_languages` 切換族別，再呼叫 MT `/synthesize`，要回傳音檔；超過 15 秒會印出警告（MT 的逾時是 20 秒） |

第 4 項才會真正經過「mt 容器 → 公開網域 → tts」這條路，
可以抓到 hairpin NAT 不通、`SAPOLITA_WEBSITE_HOST` 設錯、tts 還沒部署新版等問題。

## 四、注意事項

- **會實際使用受測主機的 GPU**：預設約 12 次合成，循序執行，無人排隊時約 1 分鐘；
  加 `--all-languages` 約 50 次、2–3 分鐘。
  請避開尖峰時段。
- **部署順序**：tts 要先部署新版，mt 再部署。只部署 tts 時，第 1–3 項應該通過、第 4 項失敗。
