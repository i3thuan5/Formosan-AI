import tempfile
import threading

from gradio_client import Client
from utils import SAPOLITA_WEBSITE_HOST


# 用環境變數組網址，不可從 request 的 Host 標頭推導（SSRF 風險）
TTS_API_URL = f"https://{SAPOLITA_WEBSITE_HOST}/hnang-kari-ai-asi-sluhay/"
TTS_TIMEOUT_SECONDS = 20
TTS_DOWNLOAD_DIR = tempfile.mkdtemp(prefix="tts-download-")


class TtsClient:
    # 第一次合成時才建立，tts 沒開時 mt 也能正常啟動。
    # 所有使用者共用一個 client：每個 Client 都會常駐一條 heartbeat 連線到 tts，
    # 用完即丟也不會被回收，要 close() 後約 20 秒才會真正釋放。

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._client is None:
                self._client = Client(
                    TTS_API_URL,
                    verbose=False,
                    analytics_enabled=False,
                    download_files=TTS_DOWNLOAD_DIR,
                    httpx_kwargs={"timeout": TTS_TIMEOUT_SECONDS},
                )
            return self._client

    def reset(self):
        # 連線出錯時丟掉，下次 get() 重新建立
        with self._lock:
            self._client = None
