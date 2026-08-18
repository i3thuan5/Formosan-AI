/*global open*/

// asr-kaldi／族語語音辨識系統（Kaldi／Vosk）延遲測試
// 線上 API：https://ai-labs.ilrdf.org.tw/sapolita-kaldi/?view=api
// 測試資料：testing_data/海岸阿美語-曾玉蘭-個人生命史-短.mp3
//   （由同名 mp4 用 ffmpeg 轉成 16kHz 單聲道 mp3，10.12 秒；此 API 的輸入是音檔，不吃影片）

import {
  makeTrends,
  uploadFile,
  fileData,
  measure,
  summaryHandler,
  latencyScenario,
  SUMMARY_TREND_STATS,
} from "./gradio-helpers.js";

const APP = "sapolita-kaldi";
const MODEL = "asr-kaldi";
const API_NAME = "automatic_speech_recognition";
const FILENAME = "海岸阿美語-曾玉蘭-個人生命史-短.mp3";
const DIALECT_ID = "formosan_ami"; // 阿美語

const trends = makeTrends(MODEL);
const AUDIO_BIN = open(`./testing_data/${FILENAME}`, "b");

export const options = {
  scenarios: latencyScenario("asr_kaldi"),
  summaryTrendStats: SUMMARY_TREND_STATS,
  thresholds: {
    checks: ["rate>0.99"],
  },
};

export default function asrKaldiIteration() {
  const up = uploadFile(APP, MODEL, AUDIO_BIN, FILENAME, "audio/mpeg", trends);
  if (!up.path) {
    return;
  }

  measure({
    app: APP,
    model: MODEL,
    apiName: API_NAME,
    data: [DIALECT_ID, fileData(up.path, FILENAME)],
    trends: trends,
    uploadMs: up.ms,
    checkFn: {
      "asr-kaldi 有辨識出 sasowalen": (p) =>
        Array.isArray(p) && p[0].includes("sasowalen"),
    },
  });
}

export const handleSummary = summaryHandler(MODEL);
