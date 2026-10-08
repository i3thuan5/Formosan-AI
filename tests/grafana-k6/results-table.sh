#!/usr/bin/env bash
#
# 把 results/*-summary.json 整理成一張 markdown 表格，方便貼進 RESULTS.md 比較。
#
#   $ ./results-table.sh
#
set -euo pipefail

cd "$(dirname "$0")"

echo "| 模型 | 筆數 | 平均 | 中位數 | p90 | p95 | p99 | 最小 | 最大 |"
echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

for model in asr asr-kaldi tts mt; do
	f="results/${model}-summary.json"
	[[ -f "$f" ]] || continue
	metric="$(echo "${model}" | tr '-' '_')_infer_ms"
	jq -r --arg model "$model" --arg metric "$metric" '
		.metrics[$metric].values as $v
		| "| \($model) | \($v.count) "
		+ ([$v.avg, $v.med, $v["p(90)"], $v["p(95)"], $v["p(99)"], $v.min, $v.max]
		   | map("| \(. / 1000 * 100 | round / 100)s ") | join(""))
		+ "|"
	' "$f"
done

echo
echo "（infer ＝ 模型推論延遲，不含上傳）"
