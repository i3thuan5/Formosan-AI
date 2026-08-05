// tts／族語語音合成系統（F5-TTS）延遲測試
// 線上 API：https://ai-labs.ilrdf.org.tw/hnang-kari-ai-asi-sluhay/?view=api
// 測試資料：2 句阿美秀姑巒，輪流合成，配音員固定用「阿美_秀姑巒_女聲1」。

import exec from "k6/execution";

import {
  makeTrends,
  measure,
  summaryHandler,
  latencyScenario,
  tagBreakdown,
  SUMMARY_TREND_STATS,
} from "./gradio-helpers.js";

const APP = "hnang-kari-ai-asi-sluhay";
const MODEL = "tts";
const API_NAME = "default_speaker_tts";
const REF = "阿美_秀姑巒_女聲1";

const SENTENCES = ["O sasowalen ako", "Itiya:ay ho a ʼorip niyam"];

const trends = makeTrends(MODEL);

export const options = {
  // 100 次 ÷ 2 句 ＝ 每句 50 次
  scenarios: latencyScenario("tts"),
  summaryTrendStats: SUMMARY_TREND_STATS,
  thresholds: Object.assign(
    { checks: ["rate>0.99"] },
    tagBreakdown("tts_infer_ms", "sentence", SENTENCES),
  ),
};

export default function () {
  const genText = SENTENCES[exec.scenario.iterationInTest % SENTENCES.length];

  measure(
    APP,
    MODEL,
    API_NAME,
    [REF, genText],
    trends,
    0,
    {
      "tts 有合成出音檔": (p) =>
        Array.isArray(p) && p[0] && typeof p[0].path === "string",
    },
    { sentence: genText },
  );
}

export const handleSummary = summaryHandler(MODEL);
