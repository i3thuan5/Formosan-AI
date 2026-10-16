# 建三個共用 image 與四個服務 image。
#
#   docker buildx bake -f docker-bake.hcl --load                  # 本機，建好載入 Docker，不用 cache
#   docker buildx bake -f docker-bake.hcl --load base gpu files   # 只建共用 image（本機 docker compose 開發用）
#   CACHE=read docker buildx bake -f docker-bake.hcl              # CI 的 PR：只驗證能 build，只讀 cache
#   GIT_SHA=abc1234 CACHE=readwrite docker buildx bake -f docker-bake.hcl --push asr asr-kaldi tts mt
#                                                                 # CI 的 main：推送四個服務，讀也寫 cache
#
# 一定要加 -f docker-bake.hcl，否則 bake 會把 docker-compose.yml 也讀進來合併。
#
# 為什麼用 bake 而不是依序 docker buildx build：
# registry cache 需要 docker-container driver，而這個 driver 的 BuildKit 看不到本機
# image store，服務的 FROM formosan-ai-gpu 會跑去 Docker Hub 拉而失敗。
# bake 的 contexts = { x = "target:y" } 讓同一次 build 內的 target 直接當別人的 base，
# 不必先存進本機再讀出來。

variable "CACHE" {
  # 空字串不用 cache，read 只讀，readwrite 讀也寫
  default = ""
}

variable "CACHE_PREFIX" {
  default = "ithuan/formosan-ai:cache"
}

variable "IMAGE" {
  default = "ithuan/formosan-ai"
}

variable "GIT_SHA" {
  # 只有 CI 推送 main 時才設定，例如 d27473f。
  # 有設定：每個服務多一個固定的 tag，例如 tts-d27473f，用來確認是哪個 commit 推的。
  # 沒設定（PR、本機）：只有 tts、asr 這種每次都會被覆蓋的 tag。
  default = ""
}

function "cache_from" {
  params = [name]
  result = CACHE == "" ? [] : ["type=registry,ref=${CACHE_PREFIX}-${name}"]
}

function "cache_to" {
  params = [name]
  result = CACHE == "readwrite" ? ["type=registry,mode=max,ref=${CACHE_PREFIX}-${name}"] : []
}

function "service_tags" {
  params = [name]
  result = GIT_SHA == "" ? ["${IMAGE}:${name}"] : ["${IMAGE}:${name}", "${IMAGE}:${name}-${GIT_SHA}"]
}

group "default" {
  targets = ["base", "gpu", "files", "asr", "asr-kaldi", "tts", "mt"]
}

# ---- common/Dockerfile 的三個 stage ---------------------------------------

target "base" {
  context    = "common"
  target     = "base"
  tags       = ["formosan-ai-base"]
  cache-from = cache_from("base")
  cache-to   = cache_to("base")
}

target "gpu" {
  context = "common"
  target  = "gpu"
  # 一致性檢查腳本在 tests/，不在 common 這個 build context 裡
  contexts = {
    tests = "tests"
  }
  tags       = ["formosan-ai-gpu"]
  cache-from = cache_from("gpu")
  cache-to   = cache_to("gpu")
}

target "files" {
  context    = "common"
  target     = "files"
  tags       = ["formosan-ai-common"]
  cache-from = cache_from("files")
  cache-to   = cache_to("files")
}

# ---- 四個服務 -------------------------------------------------------------

# 四個服務共用的 metadata：在 image 裡記下來源 commit
target "_service" {
  labels = {
    "org.opencontainers.image.source"   = "https://github.com/i3thuan5/Formosan-AI"
    "org.opencontainers.image.revision" = GIT_SHA
  }
}

target "_gpu-service" {
  inherits = ["_service"]
  contexts = {
    "formosan-ai-gpu"    = "target:gpu"
    "formosan-ai-common" = "target:files"
  }
}

target "asr" {
  inherits   = ["_gpu-service"]
  context    = "asr"
  tags       = service_tags("asr")
  cache-from = cache_from("asr")
  cache-to   = cache_to("asr")
}

target "tts" {
  inherits   = ["_gpu-service"]
  context    = "tts"
  tags       = service_tags("tts")
  cache-from = cache_from("tts")
  cache-to   = cache_to("tts")
}

target "mt" {
  inherits   = ["_gpu-service"]
  context    = "mt"
  tags       = service_tags("mt")
  cache-from = cache_from("mt")
  cache-to   = cache_to("mt")
}

target "asr-kaldi" {
  inherits = ["_service"]
  context  = "asr-kaldi"
  contexts = {
    "formosan-ai-base"   = "target:base"
    "formosan-ai-common" = "target:files"
  }
  tags       = service_tags("asr-kaldi")
  cache-from = cache_from("asr-kaldi")
  cache-to   = cache_to("asr-kaldi")
}
