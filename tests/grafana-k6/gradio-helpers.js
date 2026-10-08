/*global __ENV*/

// 共用的 Gradio API 客戶端，供 4 個模型（asr、asr-kaldi、tts、mt）的延遲測試使用。
//
// 線上 API 都是 Gradio 5 的「兩段式」呼叫：
//   1. POST {BASE_URL}/{APP}/gradio_api/call/{api_name}  → {"event_id": "..."}
//   2. GET  {BASE_URL}/{APP}/gradio_api/call/{api_name}/{event_id}
//      這一段是 SSE，會阻塞等待模型算完，才回傳「event: complete」。
// 所以「延遲」＝第 1 段＋第 2 段的時間總和，這就是使用者實際等待的時間。

import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

export const BASE_URL = __ENV.BASE_URL || "https://ai-labs.ilrdf.org.tw";

export const ITERATIONS = Number(__ENV.ITERATIONS || 100);

// summary 裡要印的統計值，count 要自己開啟才會有。
export const SUMMARY_TREND_STATS = [
  "count",
  "avg",
  "min",
  "med",
  "max",
  "p(90)",
  "p(95)",
  "p(99)",
];

// 每個模型都用相同的執行方式：1 個 VU 連續跑 N 次，測量無競爭情況下的延遲。
export function latencyScenario(name) {
  const scenarios = {};
  scenarios[name] = {
    executor: "per-vu-iterations",
    vus: 1,
    iterations: ITERATIONS,
    maxDuration: "3h",
  };
  return scenarios;
}

// k6 要有 threshold 才會把「帶 tag 的 sub-metric」算出來放進 summary，
// 所以這裡開一些永遠不會失敗的 threshold，純粹是為了看各句的統計。
export function tagBreakdown(metricName, tagKey, values) {
  const out = {};
  for (const v of values) {
    out[`${metricName}{${tagKey}:${v}}`] = ["max>=0"];
  }
  return out;
}

// k6 的 metric 名稱只接受字母、數字與底線，所以「asr-kaldi」要先換成「asr_kaldi」。
export function metricPrefix(model) {
  return model.replace(/\W/g, "_");
}

// 各測試檔案自行宣告要用的 Trend，這裡提供產生器，確保命名一致。
export function makeTrends(model) {
  const p = metricPrefix(model);
  return {
    upload: new Trend(`${p}_upload_ms`, true), // 上傳測試資料的時間
    infer: new Trend(`${p}_infer_ms`, true), // 模型推論的時間（本測試的主要指標）
    total: new Trend(`${p}_total_ms`, true), // 上傳＋推論，使用者感受到的總延遲
  };
}

// 上傳檔案到 Gradio，回傳 { path, ms }：伺服器上的檔案路徑與上傳所花的時間。
export function uploadFile(app, model, fileBin, filename, mimeType, trends) {
  const res = http.post(
    `${BASE_URL}/${app}/gradio_api/upload`,
    { files: http.file(fileBin, filename, mimeType) },
    { tags: { model: model, step: "upload" }, timeout: "600s" },
  );

  const ok = check(res, {
    [`${model} 上傳 status is 200`]: (r) => r.status === 200,
    [`${model} 上傳有回傳路徑`]: (r) => r.body?.startsWith("["),
  });

  trends.upload.add(res.timings.duration);

  if (!ok) {
    return { path: null, ms: res.timings.duration };
  }
  return { path: JSON.parse(res.body)[0], ms: res.timings.duration };
}

// 包裝成 Gradio 的 FileData 格式。
export function fileData(path, origName) {
  return {
    path: path,
    orig_name: origName,
    meta: { _type: "gradio.FileData" },
  };
}

// 呼叫 Gradio 的 named endpoint，回傳 { ok, elapsed, payload }。
export function gradioCall(app, model, apiName, data) {
  const start = Date.now();

  const post = http.post(
    `${BASE_URL}/${app}/gradio_api/call/${apiName}`,
    JSON.stringify({ data: data }),
    {
      headers: { "Content-Type": "application/json" },
      tags: { model: model, step: "submit" },
      timeout: "600s",
    },
  );

  const submitted = check(post, {
    [`${model} 送出 status is 200`]: (r) => r.status === 200,
  });
  if (!submitted) {
    return { ok: false, elapsed: Date.now() - start, payload: null };
  }

  const eventId = JSON.parse(post.body).event_id;

  // 這一段 SSE 會阻塞等待，直到模型算完才結束，所以它的 duration 就是模型的計算時間。
  const stream = http.get(
    `${BASE_URL}/${app}/gradio_api/call/${apiName}/${eventId}`,
    { tags: { model: model, step: "result" }, timeout: "600s" },
  );

  const elapsed = Date.now() - start;

  const ok = check(stream, {
    [`${model} 結果 status is 200`]: (r) => r.status === 200,
    [`${model} 結果有 complete 事件`]: (r) =>
      r.body?.includes("event: complete"),
  });

  if (stream.body?.includes("event: error")) {
    console.error(`${model} 回傳 error：${stream.body.slice(0, 500)}`);
  }

  return { ok: ok, elapsed: elapsed, payload: parseSse(stream.body) };
}

// 從 SSE body 取出「event: complete」那一行的 data。
function parseSse(body) {
  if (!body) {
    return null;
  }
  const lines = body.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === "event: complete" && lines[i + 1]) {
      const raw = lines[i + 1].replace(/^data:\s*/, "");
      try {
        return JSON.parse(raw);
      } catch (e) {
        return raw;
      }
    }
  }
  return null;
}

// 一次完整的量測：呼叫模型、記錄 infer 與 total。
//
// 參數用物件傳入：
//   app、model、apiName、data ── 見 gradioCall()
//   trends   ── makeTrends() 產生的 Trend
//   uploadMs ── 這次上傳所花的時間，沒有上傳就省略
//   checkFn  ── 檢查回傳內容的 check 定義
//   tags     ── 附在 metric 上的 tag（例如 mt 各句、tts 各句），方便後續分開比較
export function measure({
  app,
  model,
  apiName,
  data,
  trends,
  uploadMs = 0,
  checkFn,
  tags,
}) {
  const res = gradioCall(app, model, apiName, data);
  trends.infer.add(res.elapsed, tags);
  trends.total.add(res.elapsed + uploadMs, tags);

  if (res.ok && checkFn) {
    check(res.payload, checkFn);
  }
  return res;
}

// 各測試檔案共用的 summary 輸出：stdout 印簡表，另外寫一份 JSON 方便後續比較。
export function summaryHandler(model) {
  return function (data) {
    const out = {};
    out[`results/${model}-summary.json`] = JSON.stringify(data, null, 2);
    out["stdout"] = renderText(model, data);
    return out;
  };
}

function renderText(model, data) {
  const lines = [`\n=== ${model} 延遲測試結果 ===`];

  const prefix = `${metricPrefix(model)}_`;
  const trendNames = Object.keys(data.metrics)
    .filter((n) => n.startsWith(prefix))
    .sort((a, b) => a.localeCompare(b));

  const width = Math.max(...trendNames.map((n) => n.length), 16) + 1;

  for (const name of trendNames) {
    const v = data.metrics[name].values;
    lines.push(
      `${name.padEnd(width)} n=${String(v.count).padStart(4)}` +
        ` avg=${fmt(v.avg)}` +
        ` med=${fmt(v.med)}` +
        ` p90=${fmt(v["p(90)"])}` +
        ` p95=${fmt(v["p(95)"])}` +
        ` p99=${fmt(v["p(99)"])}` +
        ` min=${fmt(v.min)}` +
        ` max=${fmt(v.max)}`,
    );
  }

  const failed = data.metrics.http_req_failed;
  if (failed) {
    lines.push(
      `${"http_req_failed".padEnd(width)} 失敗率=${(failed.values.rate * 100).toFixed(2)}%`,
    );
  }
  const checks = data.metrics.checks;
  if (checks) {
    lines.push(
      `${"checks".padEnd(width)} 成功率=${(checks.values.rate * 100).toFixed(2)}%`,
    );
  }
  lines.push("");
  return lines.join("\n");
}

function fmt(ms) {
  return `${(ms / 1000).toFixed(2)}s`;
}
