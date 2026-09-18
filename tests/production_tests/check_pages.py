"""檢查四個服務的頁面與共用靜態檔。

抓得到 image 少複製 common 檔案、反向代理路徑設錯、服務沒起來。
不需要 gradio_client，也不消耗 GPU。
"""
import json
import urllib.error
import urllib.request

APPS = {
    "asr": "sapolita",
    "asr-kaldi": "sapolita-kaldi",
    "tts": "hnang-kari-ai-asi-sluhay",
    "mt": "kari-seejiq-tnpusu-ai-hmjil",
}

# common/utils.py 把 common_static 的 favicon 設成 favicon_path、css 併進 theme.css、
# image 與 pdf 交給 gr.set_static_paths，所以這四個 URL 就是共用檔案有沒有進 image 的證據。
PATHS = [
    ("/", None),
    ("/config", "json"),
    ("/favicon.ico", None),
    ("/theme.css", None),
    ("/gradio_api/file=common_static/image/ilrdf-logo.png", None),
]
TIMEOUT = 30


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "formosan-ai-production-tests"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.status, response.read()


def check_path(base_url, app, path, kind):
    def run():
        url = f"{base_url}/{app}{path}"
        try:
            status, body = fetch(url)
        except urllib.error.HTTPError as e:
            raise AssertionError(f"{url} 回 {e.code}")
        except Exception as e:
            raise AssertionError(f"{url} 連不上：{type(e).__name__}: {e}")
        if status != 200:
            raise AssertionError(f"{url} 回 {status}")
        if kind == "json":
            try:
                json.loads(body)
            except ValueError as e:
                raise AssertionError(f"{url} 不是 JSON：{e}")
        if not body:
            raise AssertionError(f"{url} 回傳空內容")
        return f"{len(body)} bytes"
    return run


def run(base_url, check, failures):
    print("\n[頁面與共用靜態檔]")
    for service, app in APPS.items():
        for path, kind in PATHS:
            check(f"{service} {path}", check_path(base_url, app, path, kind), failures, service="pages")
