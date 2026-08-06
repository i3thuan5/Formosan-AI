/*global open*/

// asr／族語逐字稿辨識系統（Whisper）延遲測試
// 線上 API：https://ai-labs.ilrdf.org.tw/sapolita/?view=api
// 測試資料：testing_data/海岸阿美語-曾玉蘭-個人生命史-短.mp4（原始檔，10.18 秒）

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

const trends = makeTrends(MODEL);
const VIDEO_BIN = open(`./testing_data/${FILENAME}`, "b");

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
    data: [{ video: fileData(up.path, FILENAME) }],
    trends: trends,
    uploadMs: up.ms,
    checkFn: {
      "asr 有辨識出 sasowalen": (p) =>
        Array.isArray(p) && p[0].includes("sasowalen"),
    },
  });
}

export const handleSummary = summaryHandler(MODEL);
