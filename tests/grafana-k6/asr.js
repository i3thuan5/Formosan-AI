/*global open*/

// asr／族語逐字稿辨識系統（Whisper）延遲測試
// 線上 API：https://ai-labs.ilrdf.org.tw/sapolita/?view=api
// 測試資料：../data/海岸阿美語-曾玉蘭-個人生命史-短.mp4（原始檔，10.18 秒）

import {
  makeTrends,
  uploadFile,
  fileData,
  measure,
  summaryHandler,
  latencyScenario,
  SUMMARY_TREND_STATS,
} from "./gradio-helpers.js";

const APP = "sapolita";
const MODEL = "asr";
const API_NAME = "generate_srt";
const FILENAME = "海岸阿美語-曾玉蘭-個人生命史-短.mp4";
// 2026-09-14 起 /generate_srt 多了語別參數。海岸阿美語在預設族別的 choices 裡，
// 所以不用先呼叫 /update_languages 切換族別。
const LANGUAGE = "ami-x-pswl";

const trends = makeTrends(MODEL);
const VIDEO_BIN = open(`../data/${FILENAME}`, "b");

export const options = {
  scenarios: latencyScenario("asr"),
  summaryTrendStats: SUMMARY_TREND_STATS,
  thresholds: {
    checks: ["rate>0.99"],
  },
};

export default function asrIteration() {
  const up = uploadFile(APP, MODEL, VIDEO_BIN, FILENAME, "video/mp4", trends);
  if (!up.path) {
    return;
  }

  measure({
    app: APP,
    model: MODEL,
    apiName: API_NAME,
    data: [{ video: fileData(up.path, FILENAME) }, LANGUAGE],
    trends: trends,
    uploadMs: up.ms,
    // 只驗有辨識出字幕，不比對文字：模型換版後辨識結果會變
    // （2026-09-14 前是 sasowalen，之後是 sosowalen）
    checkFn: {
      "asr 有辨識出族語字幕": (p) =>
        Array.isArray(p) && typeof p[0] === "string" && p[0].includes("族語："),
    },
  });
}

export const handleSummary = summaryHandler(MODEL);
