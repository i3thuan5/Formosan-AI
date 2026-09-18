#!/usr/bin/env bash
#
# 4 個模型的延遲測試，一個一個依序執行（不要並行，才不會互相搶資源）。
#
#   $ ./run-latency.sh              # 每個模型 100 次
#   $ ITERATIONS=5 ./run-latency.sh # 先小跑測試看看
#
set -euo pipefail

cd "$(dirname "$0")"

ITERATIONS="${ITERATIONS:-100}"
BASE_URL="${BASE_URL:-https://ai-labs.ilrdf.org.tw}"
K6="${K6:-k6}"

mkdir -p results

for model in asr asr-kaldi tts mt; do
	echo "########## ${model}（${ITERATIONS} 次）##########"
	"${K6}" run \
		--quiet \
		-e "ITERATIONS=${ITERATIONS}" \
		-e "BASE_URL=${BASE_URL}" \
		"./${model}.js" \
		2>&1 | tee "results/${model}-run.log"
done

echo "########## 全部執行完畢，結果在 results/ ##########"
