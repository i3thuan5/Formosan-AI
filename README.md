# formosan-asr-cd

原語會「族語AI翻譯計畫」語音系統。

- 族語AI成果網站網址：https://ai-labs.ilrdf.org.tw/
- 語音辨識系統網址：https://sapolita-kaldi.ithuan.tw/
- 合成系統網址：https://hnang-kari-ai-asi-sluhay.ithuan.tw/
- 基礎翻譯系統網址：https://ithuan-formosan-translation.hf.space/

## local開發

### 建立 Python virtual environment

```bash
python -m venv venv
```

### 載入 Python virtual environment

Ta̍k-kái攏ài開，才來開發。

```bash
source venv/bin/activate
```

### 安裝tox

tox是tī本機走test用--ê。

```bash
pip install tox
```

### 執行Linter排版器

依需求執行，如：

```bash
tox -e yamllint
tox -e flake8
```

詳細需求可參考`tox.ini`。

## 更新套件版本

`requirements.in`是記專案有直接用ê第三方套件。`requirements.txt`是管kui專案全部第三方套件koh對應版本，保證開發、CI試驗、上線版本一致。

1. 請先 `pip install pip-tools` tàu [pip-tools](https://github.com/jazzband/pip-tools) 自動管理套件版本。
2. 手動更新`requirements.in`。
3. 揀一款指令自動更新套件版本。

      ```bash
      # 有必要--ê才更新
      pip-compile pip-compile asr/requirements.in
      # 盡量更新
      pip-compile --upgrade pip-compile asr/requirements.in
      ```

4. 檢查`requirements.txt`更新狀態。

## local測試

1. 設定reverse proxy server：

```bash
git clone --depth 1 https://github.com/i3thuan5/ZuGi.git
docker compose -f ZuGi/docker-compose.yml up -d --build nginx-proxy
```

2. 準備環境變數檔：`cp deploy/.env.template .env`，並把要測試的服務HOST改成`localhost`。

3. 手動編共用檔案的image：`docker build -t formosan-ai-common ./common` 。

4. 編image後啟動：`docker compose up -d --build`。
