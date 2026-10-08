# 建三個共用 image 與四個服務 image。
#
#   docker buildx bake -f docker-bake.hcl --load                  # 本機，建好載入 Docker，不用 cache
#   docker buildx bake -f docker-bake.hcl --load base gpu files   # 只建共用 image（本機 docker compose 開發用）
#   CACHE=read docker buildx bake -f docker-bake.hcl              # CI 的 PR：只驗證能 build，只讀 cache
#   CACHE=readwrite docker buildx bake -f docker-bake.hcl --push asr asr-kaldi tts mt
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

function "cache_from" {
  params = [name]
  result = CACHE == "" ? [] : ["type=registry,ref=${CACHE_PREFIX}-${name}"]
}

function "cache_to" {
  params = [name]
  result = CACHE == "readwrite" ? ["type=registry,mode=max,ref=${CACHE_PREFIX}-${name}"] : []
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

target "_gpu-service" {
  contexts = {
    "formosan-ai-gpu"    = "target:gpu"
    "formosan-ai-common" = "target:files"
  }
}

target "asr" {
  inherits   = ["_gpu-service"]
  context    = "asr"
  tags       = ["ithuan/formosan-ai:asr"]
  cache-from = cache_from("asr")
  cache-to   = cache_to("asr")
}

target "tts" {
  inherits   = ["_gpu-service"]
  context    = "tts"
  tags       = ["ithuan/formosan-ai:tts"]
  cache-from = cache_from("tts")
  cache-to   = cache_to("tts")
}

target "mt" {
  inherits   = ["_gpu-service"]
  context    = "mt"
  tags       = ["ithuan/formosan-ai:mt"]
  cache-from = cache_from("mt")
  cache-to   = cache_to("mt")
}

target "asr-kaldi" {
  context = "asr-kaldi"
  contexts = {
    "formosan-ai-base"   = "target:base"
    "formosan-ai-common" = "target:files"
  }
  tags       = ["ithuan/formosan-ai:asr-kaldi"]
  cache-from = cache_from("asr-kaldi")
  cache-to   = cache_to("asr-kaldi")
}
