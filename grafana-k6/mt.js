// mt／族語基礎翻譯系統（NLLB-600M）延遲測試
// 線上 API：https://ai-labs.ilrdf.org.tw/kari-seejiq-tnpusu-ai-hmjil/?view=api
// 測試資料：4 句輪流——2 句阿美秀姑巒翻華語、2 句華語翻阿美秀姑巒。
//   /translate   ＝ 族語 ⮕ 華語（src=ami_Xiug, tgt=zho_Hant）
//   /translate_1 ＝ 華語 ⮕ 族語（src=zho_Hant, tgt=ami_Xiug）

import exec from "k6/execution";

import {
  makeTrends,
  measure,
  summaryHandler,
  latencyScenario,
  tagBreakdown,
  SUMMARY_TREND_STATS,
} from "./gradio-helpers.js";

const APP = "kari-seejiq-tnpusu-ai-hmjil";
const MODEL = "mt";
const AMI = "ami_Xiug"; // 阿美_秀姑巒
const ZH = "zho_Hant";

const CASES = [
  { api: "translate", text: "O sasowalen ako", src: AMI, tgt: ZH },
  { api: "translate", text: "Itiya:ay ho a ʼorip niyam", src: AMI, tgt: ZH },
  { api: "translate_1", text: "我要說的是", src: ZH, tgt: AMI },
  { api: "translate_1", text: "我們以前的生活", src: ZH, tgt: AMI },
];

const trends = makeTrends(MODEL);

export const options = {
  // 100 次 ÷ 4 句 ＝ 每句 25 次
  scenarios: latencyScenario("mt"),
  summaryTrendStats: SUMMARY_TREND_STATS,
  thresholds: Object.assign(
    { checks: ["rate>0.99"] },
    tagBreakdown(
      "mt_infer_ms",
      "direction",
      [`${AMI}2${ZH}`, `${ZH}2${AMI}`],
    ),
    tagBreakdown(
      "mt_infer_ms",
      "sentence",
      CASES.map((c) => c.text),
    ),
  ),
};

export default function () {
  const c = CASES[exec.scenario.iterationInTest % CASES.length];

  measure(
    APP,
    MODEL,
    c.api,
    [c.text, c.src, c.tgt],
    trends,
    0,
    {
      "mt 有翻出非空字串": (p) =>
        Array.isArray(p) && typeof p[0] === "string" && p[0].trim().length > 0,
    },
    { direction: `${c.src}2${c.tgt}`, sentence: c.text },
  );
}

export const handleSummary = summaryHandler(MODEL);
