import { useState, useEffect, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ScatterChart, Scatter, ReferenceLine } from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// API SERVICE LAYER
// Replace API_BASE with your FastAPI endpoint when deploying the Python model
// POST /api/score   → { transactions: [...] } → returns scored results
// GET  /api/stats   → returns model metrics + aggregates
// GET  /api/shap    → returns SHAP feature importance
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"; // ← swap when live

async function fetchModelStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error("API unavailable");
    return await res.json();
  } catch {
    return null; // falls back to mock data below
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MOCK DATA  (mirrors exact output of cashgpt_fraud_detection.py)
// Replace with real API calls once FastAPI backend is deployed
// ─────────────────────────────────────────────────────────────────────────────
function generateMockData() {
  const seed = (n) => Math.abs(Math.sin(n * 9301 + 49297) * 233280) % 1;
  const transactions = Array.from({ length: 56962 }, (_, i) => {
    const isFraud = seed(i) < 0.0017;
    const prob = isFraud
      ? 0.5 + seed(i + 1000) * 0.5
      : seed(i + 2000) * 0.18;
    const risk = prob >= 0.75 ? "Critical" : prob >= 0.5 ? "High" : prob >= 0.35 ? "Medium" : "Low";
    return {
      id: `TXN-${String(i).padStart(5, "0")}`,
      amount: Math.round(Math.exp(seed(i + 3000) * 6) * 10) / 10,
      prob: Math.round(prob * 1000) / 1000,
      risk,
      actual: isFraud ? 1 : 0,
      predicted: prob >= 0.35 ? 1 : 0,
      hour: Math.floor(seed(i + 4000) * 24),
      v14: isFraud ? -(seed(i + 5000) * 8 + 2) : seed(i + 5000) * 2 - 1,
    };
  });

  const fraud = transactions.filter(t => t.actual === 1);
  const flagged = transactions.filter(t => t.predicted === 1);
  const tp = transactions.filter(t => t.actual === 1 && t.predicted === 1).length;
  const fp = flagged.length - tp;
  const fn = fraud.length - tp;
  const tn = transactions.length - tp - fp - fn;

  const rocData = Array.from({ length: 50 }, (_, i) => {
    const fpr = (i / 49) ** 2;
    const tpr = 1 - (1 - fpr) ** 0.08;
    return { fpr: Math.round(fpr * 1000) / 1000, tpr: Math.round(tpr * 1000) / 1000 };
  });

  const prData = Array.from({ length: 50 }, (_, i) => {
    const rec = i / 49;
    const prec = Math.max(0.1, 1 - rec * 0.35 + seed(i) * 0.05);
    return { recall: Math.round(rec * 100) / 100, precision: Math.round(prec * 100) / 100 };
  });

  const shap = [
    { feature: "V14", shap: 0.412 },
    { feature: "V4", shap: 0.287 },
    { feature: "V11", shap: 0.241 },
    { feature: "V12", shap: 0.198 },
    { feature: "V1", shap: 0.176 },
    { feature: "V3", shap: 0.154 },
    { feature: "Amount_log", shap: 0.132 },
    { feature: "V17", shap: 0.119 },
    { feature: "V10", shap: 0.098 },
    { feature: "Amount_zscore", shap: 0.087 },
    { feature: "V1_V2", shap: 0.071 },
    { feature: "V7", shap: 0.065 },
    { feature: "Hour", shap: 0.054 },
    { feature: "V14_V17", shap: 0.047 },
    { feature: "V3_V4", shap: 0.039 },
  ];

  const riskCounts = {
    Critical: transactions.filter(t => t.risk === "Critical").length,
    High: transactions.filter(t => t.risk === "High").length,
    Medium: transactions.filter(t => t.risk === "Medium").length,
    Low: transactions.filter(t => t.risk === "Low").length,
  };

  const modelAucs = { XGBoost: 0.9741, LightGBM: 0.9698, CatBoost: 0.9762, Stacking: 0.9847 };

  const hourDist = Array.from({ length: 24 }, (_, h) => ({
    hour: `${h}:00`,
    fraud: transactions.filter(t => t.hour === h && t.actual === 1).length,
    normal: Math.floor(transactions.filter(t => t.hour === h && t.actual === 0).length / 100),
  }));

  const probHist = Array.from({ length: 20 }, (_, i) => {
    const lo = i * 0.05, hi = (i + 1) * 0.05;
    return {
      bucket: `${(lo * 100).toFixed(0)}%`,
      fraud: transactions.filter(t => t.actual === 1 && t.prob >= lo && t.prob < hi).length,
      normal: Math.floor(transactions.filter(t => t.actual === 0 && t.prob >= lo && t.prob < hi).length / 20),
    };
  });

  const topFlagged = flagged
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 12)
    .map(t => ({ ...t, correct: t.actual === t.predicted }));

  return {
    totalTx: transactions.length,
    fraudCount: fraud.length,
    flaggedCount: flagged.length,
    aucRoc: 0.9847, aucPr: 0.8213,
    f1: 0.8641, precision: 0.8934, recall: 0.8365,
    tp, fp, fn, tn,
    riskCounts, modelAucs, shap, rocData, prData,
    hourDist, probHist, topFlagged,
    valueAtRisk: Math.round(flagged.reduce((s, t) => s + t.amount, 0)),
    limeInsights: [
      { feat: "V14 <= -2.31", weight: +0.312 },
      { feat: "V4 > 1.87", weight: +0.241 },
      { feat: "Amount_log > 5.2", weight: +0.198 },
      { feat: "V11 <= -1.04", weight: +0.176 },
      { feat: "Hour = 02:00", weight: +0.143 },
      { feat: "V1 <= -3.12", weight: +0.129 },
      { feat: "V3_V4 > 2.1", weight: +0.091 },
      { feat: "V17 <= -0.88", weight: +0.074 },
    ],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

const C = {
  bg: "#080B10", surface: "#0E1219", card: "#131820",
  border: "#1E2535", borderHover: "#2E3D52",
  red: "#F03E3E", redDim: "#4A1515", redGlow: "#F03E3E22",
  amber: "#F59E0B", amberDim: "#3D2800",
  green: "#10B981", greenDim: "#0A2E20",
  blue: "#3B82F6", teal: "#06B6D4",
  purple: "#8B5CF6", white: "#F1F5F9",
  muted: "#64748B", mutedLight: "#94A3B8",
};

const pulse = `
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  @keyframes scanline { 0%{top:-4px} 100%{top:100%} }
  @keyframes blink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} 99%{opacity:0} }
`;

function KpiCard({ icon, label, value, sub, accent, delay = 0, glow = false }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), delay); return () => clearTimeout(t); }, [delay]);
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 8, padding: "16px 18px",
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(10px)",
      transition: "opacity 0.5s ease, transform 0.5s ease",
      boxShadow: glow ? `0 0 20px ${accent}22` : "none",
      position: "relative", overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: accent, opacity: 0.8 }} />
      <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: C.muted, letterSpacing: "0.08em", marginBottom: 6, textTransform: "uppercase" }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: accent, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: C.muted, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 16, height: 1, background: C.muted }} />
      {children}
      <div style={{ flex: 1, height: 1, background: C.border }} />
    </div>
  );
}

function RiskBadge({ risk }) {
  const map = { Critical: [C.red, C.redDim], High: [C.amber, C.amberDim], Medium: [C.blue, "#0A1A3D"], Low: [C.green, C.greenDim] };
  const [fg, bg] = map[risk] || [C.muted, C.surface];
  return (
    <span style={{ background: bg, color: fg, border: `1px solid ${fg}44`, borderRadius: 4, padding: "2px 7px", fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, letterSpacing: "0.05em" }}>
      {risk.toUpperCase()}
    </span>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}>
      <div style={{ color: C.mutedLight, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.name}: {typeof p.value === "number" ? p.value.toFixed(3) : p.value}</div>
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// MAIN DASHBOARD
// ─────────────────────────────────────────────────────────────────────────────
export default function CashGPTDashboard() {
  const [data, setData] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("mock");
  const [filterRisk, setFilterRisk] = useState("All");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const live = await fetchModelStats();
      if (live) {
        setData(live);
        setApiStatus("live");
      } else {
        setData(generateMockData());
        setApiStatus("mock");
      }
      setLoading(false);
    }
    load();
  }, []);

  const filteredTx = data?.topFlagged.filter(t =>
    filterRisk === "All" || t.risk === filterRisk
  ) ?? [];

  if (loading) return (
    <div style={{ background: C.bg, minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'IBM Plex Mono', monospace" }}>
      <div style={{ textAlign: "center", color: C.muted }}>
        <div style={{ fontSize: 28, marginBottom: 12, animation: "pulse 1.5s infinite" }}>⬡</div>
        <div style={{ fontSize: 13, color: C.teal }}>INITIALISING CASHGPT</div>
        <div style={{ fontSize: 11, marginTop: 6 }}>Loading stacking ensemble outputs…</div>
      </div>
    </div>
  );

  const d = data;

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'IBM Plex Mono', monospace", color: C.white }}>
      <style>{pulse}</style>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />

      {/* ── HEADER ── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, background: C.surface, padding: "0 24px", display: "flex", alignItems: "center", gap: 20, height: 52, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, background: C.teal, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, color: C.bg, fontWeight: 700 }}>⬡</div>
          <span style={{ fontSize: 15, fontWeight: 700, color: C.teal, letterSpacing: "0.05em" }}>CashGPT</span>
          <span style={{ fontSize: 11, color: C.muted, borderLeft: `1px solid ${C.border}`, paddingLeft: 12 }}>Fraud & Anomaly Detection</span>
        </div>

        <div style={{ flex: 1 }} />

        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: C.muted }}>
          <span>Almalki & Masud (2025)</span>
          <span style={{ color: C.border }}>·</span>
          <span>XGB + LGBM + CatBoost</span>
          <span style={{ color: C.border }}>·</span>
          <span>SHAP + LIME</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 16 }}>
          <div style={{ background: C.redDim, border: `1px solid ${C.red}44`, borderRadius: 4, padding: "3px 10px", fontSize: 10, color: C.red, display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: C.red, animation: "pulse 1.5s infinite" }} />
            {d.riskCounts.Critical} CRITICAL
          </div>
          <div style={{ background: C.greenDim, border: `1px solid ${C.green}44`, borderRadius: 4, padding: "3px 10px", fontSize: 10, color: C.green }}>
            {apiStatus === "live" ? "● LIVE API" : "◎ MOCK"}
          </div>
        </div>
      </div>

      {/* ── NAV TABS ── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "0 24px", display: "flex", gap: 0 }}>
        {["overview", "transactions", "explainability", "model"].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "12px 18px", fontSize: 11, letterSpacing: "0.08em",
            textTransform: "uppercase", color: tab === t ? C.teal : C.muted,
            borderBottom: `2px solid ${tab === t ? C.teal : "transparent"}`,
            transition: "all 0.2s",
          }}>{t}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", alignItems: "center", fontSize: 10, color: C.muted, gap: 6, paddingRight: 4 }}>
          <span style={{ animation: "blink 1s infinite", color: C.green }}>█</span>
          {new Date().toLocaleTimeString()}
        </div>
      </div>

      <div style={{ padding: 24, maxWidth: 1400, margin: "0 auto" }}>

        {/* ══════════════ OVERVIEW TAB ══════════════ */}
        {tab === "overview" && (
          <>
            {/* KPI Row 1 */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 12 }}>
              <KpiCard icon="◈" label="Transactions Scored" value={d.totalTx.toLocaleString()} sub={`Test set · ULB dataset`} accent={C.white} delay={0} />
              <KpiCard icon="⚑" label="Flagged Fraud" value={d.flaggedCount} sub={`${(d.flaggedCount / d.totalTx * 100).toFixed(1)}% flag rate`} accent={C.red} delay={80} glow />
              <KpiCard icon="◎" label="AUC-ROC" value={d.aucRoc.toFixed(4)} sub={`AUC-PR: ${d.aucPr.toFixed(4)}`} accent={C.green} delay={160} />
              <KpiCard icon="◷" label="F1 / Recall" value={`${d.f1.toFixed(3)} / ${d.recall.toFixed(3)}`} sub={`Precision: ${d.precision.toFixed(3)}`} accent={C.blue} delay={240} />
            </div>
            {/* KPI Row 2 */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
              <KpiCard icon="▲" label="Critical Alerts" value={d.riskCounts.Critical} sub={`High: ${d.riskCounts.High} · Med: ${d.riskCounts.Medium}`} accent={C.red} delay={320} glow />
              <KpiCard icon="$" label="Value at Risk" value={`$${d.valueAtRisk.toLocaleString()}`} sub="Flagged transactions" accent={C.amber} delay={400} />
              <KpiCard icon="✓" label="True Positives" value={d.tp} sub={`False Neg: ${d.fn} missed`} accent={C.green} delay={480} />
              <KpiCard icon="✗" label="False Positives" value={d.fp} sub={`True Neg: ${d.tn.toLocaleString()}`} accent={C.amber} delay={560} />
            </div>

            {/* Charts row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* ROC Curve */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>ROC Curve — All Models</SectionTitle>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={d.rocData}>
                    <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                    <XAxis dataKey="fpr" stroke={C.muted} fontSize={10} tickFormatter={v => v.toFixed(1)} label={{ value: "FPR", position: "insideBottom", offset: -4, fill: C.muted, fontSize: 10 }} />
                    <YAxis stroke={C.muted} fontSize={10} label={{ value: "TPR", angle: -90, position: "insideLeft", fill: C.muted, fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine x={0} y={0} stroke={C.muted} strokeDasharray="4 4" strokeWidth={0.5} />
                    <Line type="monotone" dataKey="tpr" stroke={C.red} strokeWidth={2.5} dot={false} name={`Stacking (${d.aucRoc.toFixed(4)})`} />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
                  {Object.entries(d.modelAucs).map(([k, v]) => (
                    <div key={k} style={{ fontSize: 10, color: k === "Stacking" ? C.red : C.muted }}>
                      {k}: <span style={{ color: k === "Stacking" ? C.red : C.mutedLight }}>{v.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Precision-Recall */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>Precision-Recall Curve</SectionTitle>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={d.prData}>
                    <CartesianGrid strokeDasharray="2 4" stroke={C.border} />
                    <XAxis dataKey="recall" stroke={C.muted} fontSize={10} label={{ value: "Recall", position: "insideBottom", offset: -4, fill: C.muted, fontSize: 10 }} />
                    <YAxis stroke={C.muted} fontSize={10} domain={[0, 1]} label={{ value: "Precision", angle: -90, position: "insideLeft", fill: C.muted, fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="precision" stroke={C.red} fill={C.redGlow} strokeWidth={2} dot={false} name={`Stacking (AP=${d.aucPr.toFixed(4)})`} />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 8 }}>
                  Baseline: <span style={{ color: C.mutedLight }}>{(d.fraudCount / d.totalTx).toFixed(4)}</span> · Average Precision: <span style={{ color: C.red }}>{d.aucPr.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* Risk distribution + Confusion matrix */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* Risk dist */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>Risk Score Distribution</SectionTitle>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={[
                    { name: "Critical", count: d.riskCounts.Critical, fill: C.red },
                    { name: "High", count: d.riskCounts.High, fill: C.amber },
                    { name: "Medium", count: d.riskCounts.Medium, fill: C.blue },
                    { name: "Low", count: d.riskCounts.Low, fill: C.green },
                  ]}>
                    <CartesianGrid strokeDasharray="2 4" stroke={C.border} vertical={false} />
                    <XAxis dataKey="name" stroke={C.muted} fontSize={10} />
                    <YAxis stroke={C.muted} fontSize={10} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {["#F03E3E", "#F59E0B", "#3B82F6", "#10B981"].map((c, i) => (
                        <Cell key={i} fill={c} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Confusion matrix */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>Confusion Matrix</SectionTitle>
                <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gridTemplateRows: "auto 1fr 1fr", gap: 4, marginTop: 12, fontFamily: "'IBM Plex Mono', monospace" }}>
                  <div />
                  <div style={{ fontSize: 10, color: C.muted, textAlign: "center", paddingBottom: 4 }}>Pred Normal</div>
                  <div style={{ fontSize: 10, color: C.muted, textAlign: "center", paddingBottom: 4 }}>Pred Fraud</div>
                  <div style={{ fontSize: 10, color: C.muted, writingMode: "vertical-lr", transform: "rotate(180deg)", paddingRight: 4, display: "flex", alignItems: "center" }}>Act Normal</div>
                  <div style={{ background: C.greenDim, border: `1px solid ${C.green}33`, borderRadius: 6, padding: "20px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: C.green }}>{d.tn.toLocaleString()}</div>
                    <div style={{ fontSize: 10, color: C.green, marginTop: 4 }}>TN</div>
                  </div>
                  <div style={{ background: C.amberDim, border: `1px solid ${C.amber}33`, borderRadius: 6, padding: "20px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: C.amber }}>{d.fp}</div>
                    <div style={{ fontSize: 10, color: C.amber, marginTop: 4 }}>FP</div>
                  </div>
                  <div style={{ fontSize: 10, color: C.muted, writingMode: "vertical-lr", transform: "rotate(180deg)", paddingRight: 4, display: "flex", alignItems: "center" }}>Act Fraud</div>
                  <div style={{ background: C.amberDim, border: `1px solid ${C.amber}33`, borderRadius: 6, padding: "20px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: C.amber }}>{d.fn}</div>
                    <div style={{ fontSize: 10, color: C.amber, marginTop: 4 }}>FN</div>
                  </div>
                  <div style={{ background: C.redDim, border: `1px solid ${C.red}33`, borderRadius: 6, padding: "20px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: C.red }}>{d.tp}</div>
                    <div style={{ fontSize: 10, color: C.red, marginTop: 4 }}>TP</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Fraud by hour */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
              <SectionTitle>Fraud Activity by Hour of Day</SectionTitle>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={d.hourDist} barGap={2}>
                  <CartesianGrid strokeDasharray="2 4" stroke={C.border} vertical={false} />
                  <XAxis dataKey="hour" stroke={C.muted} fontSize={9} interval={2} />
                  <YAxis stroke={C.muted} fontSize={9} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="fraud" fill={C.red} radius={[2, 2, 0, 0]} name="Fraud txns" />
                  <Bar dataKey="normal" fill={C.blue} radius={[2, 2, 0, 0]} opacity={0.4} name="Normal (÷100)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* ══════════════ TRANSACTIONS TAB ══════════════ */}
        {tab === "transactions" && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <span style={{ fontSize: 11, color: C.muted }}>FILTER:</span>
              {["All", "Critical", "High", "Medium", "Low"].map(r => (
                <button key={r} onClick={() => setFilterRisk(r)} style={{
                  background: filterRisk === r ? (r === "All" ? C.teal : r === "Critical" ? C.redDim : r === "High" ? C.amberDim : r === "Medium" ? "#0A1A3D" : C.greenDim) : "none",
                  border: `1px solid ${filterRisk === r ? (r === "Critical" ? C.red : r === "High" ? C.amber : r === "Medium" ? C.blue : r === "Low" ? C.green : C.teal) : C.border}`,
                  borderRadius: 4, padding: "5px 14px", fontSize: 10, cursor: "pointer",
                  color: filterRisk === r ? C.white : C.muted, letterSpacing: "0.06em",
                  transition: "all 0.15s",
                }}>{r}</button>
              ))}
              <div style={{ marginLeft: "auto", fontSize: 11, color: C.muted }}>
                Showing {filteredTx.length} flagged transactions
              </div>
            </div>

            {/* Transaction log */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr 1fr 0.8fr 0.8fr 0.8fr 0.8fr", padding: "10px 16px", borderBottom: `1px solid ${C.border}`, fontSize: 10, color: C.muted, letterSpacing: "0.08em" }}>
                <span>TX-ID</span><span>RISK</span><span>FRAUD PROB</span><span>ACTUAL</span><span>PREDICTED</span><span>AMOUNT</span><span>STATUS</span>
              </div>
              {filteredTx.map((tx, i) => (
                <div key={tx.id} style={{
                  display: "grid", gridTemplateColumns: "1.2fr 0.8fr 1fr 0.8fr 0.8fr 0.8fr 0.8fr",
                  padding: "10px 16px", borderBottom: `1px solid ${C.border}`,
                  background: i % 2 === 0 ? "transparent" : `${C.surface}88`,
                  fontSize: 12, transition: "background 0.15s",
                  animation: `fadeUp 0.3s ease ${i * 30}ms both`,
                }}>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", color: C.mutedLight }}>{tx.id}</span>
                  <span><RiskBadge risk={tx.risk} /></span>
                  <span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, height: 4, background: C.surface, borderRadius: 2, overflow: "hidden" }}>
                        <div style={{ width: `${tx.prob * 100}%`, height: "100%", background: tx.prob > 0.75 ? C.red : tx.prob > 0.5 ? C.amber : C.blue, borderRadius: 2 }} />
                      </div>
                      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: tx.prob > 0.75 ? C.red : C.mutedLight, minWidth: 40 }}>{(tx.prob * 100).toFixed(1)}%</span>
                    </div>
                  </span>
                  <span style={{ color: tx.actual === 1 ? C.red : C.muted }}>{tx.actual === 1 ? "FRAUD" : "NORMAL"}</span>
                  <span style={{ color: tx.predicted === 1 ? C.red : C.green }}>{tx.predicted === 1 ? "FRAUD" : "NORMAL"}</span>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>${tx.amount.toFixed(2)}</span>
                  <span style={{ color: tx.correct ? C.green : C.red }}>{tx.correct ? "✓ CORRECT" : "✗ WRONG"}</span>
                </div>
              ))}
            </div>

            {/* Probability distribution */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, marginTop: 16 }}>
              <SectionTitle>Fraud Probability Distribution</SectionTitle>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={d.probHist} barGap={1}>
                  <CartesianGrid strokeDasharray="2 4" stroke={C.border} vertical={false} />
                  <XAxis dataKey="bucket" stroke={C.muted} fontSize={9} />
                  <YAxis stroke={C.muted} fontSize={9} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine x="35%" stroke={C.amber} strokeDasharray="4 4" label={{ value: "Threshold", fill: C.amber, fontSize: 10 }} />
                  <Bar dataKey="normal" fill={C.green} opacity={0.6} name="Normal" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="fraud" fill={C.red} name="Fraud" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* ══════════════ EXPLAINABILITY TAB ══════════════ */}
        {tab === "explainability" && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* SHAP */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>SHAP Feature Importance — XGBoost</SectionTitle>
                <div style={{ fontSize: 10, color: C.muted, marginBottom: 12 }}>Mean |SHAP value| across test set · Higher = more influential</div>
                {d.shap.map((s, i) => (
                  <div key={s.feature} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, animation: `fadeUp 0.3s ease ${i * 40}ms both` }}>
                    <div style={{ width: 90, fontSize: 11, color: i < 3 ? C.red : i < 7 ? C.amber : C.mutedLight, textAlign: "right", flexShrink: 0 }}>{s.feature}</div>
                    <div style={{ flex: 1, height: 14, background: C.surface, borderRadius: 3, overflow: "hidden" }}>
                      <div style={{
                        height: "100%", borderRadius: 3,
                        width: `${(s.shap / d.shap[0].shap) * 100}%`,
                        background: i < 3 ? C.red : i < 7 ? C.amber : C.blue,
                        opacity: 0.85, transition: "width 0.8s ease",
                      }} />
                    </div>
                    <div style={{ width: 48, fontSize: 10, color: C.mutedLight, fontFamily: "'IBM Plex Mono', monospace" }}>{s.shap.toFixed(3)}</div>
                  </div>
                ))}
              </div>

              {/* LIME */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>LIME — Highest-Risk Transaction</SectionTitle>
                <div style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>Per-prediction explanation · Features pushing toward fraud</div>
                <div style={{ background: C.redDim, border: `1px solid ${C.red}33`, borderRadius: 6, padding: "8px 12px", fontSize: 11, color: C.red, marginBottom: 14 }}>
                  Fraud probability: 0.987 — CRITICAL RISK
                </div>
                {d.limeInsights.map((l, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, animation: `fadeUp 0.3s ease ${i * 50}ms both` }}>
                    <div style={{ fontSize: 11, color: C.mutedLight, flex: 1, fontFamily: "'IBM Plex Mono', monospace" }}>{l.feat}</div>
                    <div style={{ width: 80, height: 12, background: C.surface, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", borderRadius: 2, width: `${(l.weight / 0.312) * 100}%`, background: C.red, opacity: 0.8 }} />
                    </div>
                    <div style={{ width: 52, fontSize: 10, color: C.red, textAlign: "right", fontFamily: "'IBM Plex Mono', monospace" }}>{l.weight > 0 ? "+" : ""}{l.weight.toFixed(3)}</div>
                  </div>
                ))}
                <div style={{ marginTop: 16, padding: "10px 12px", background: C.surface, borderRadius: 6, fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
                  <span style={{ color: C.teal }}>LIME</span> fits a local linear model around this transaction. Positive weights increase fraud probability. The sum of all weights approximates the log-odds difference from baseline.
                </div>
              </div>
            </div>

            {/* AI Insights panel */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
              <SectionTitle>AI Anomaly Insights</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[
                  { tag: "MODEL", color: C.teal, text: `XGB + LGBM + CatBoost stacking with RF meta-learner achieves AUC-ROC ${d.aucRoc.toFixed(4)} — ${d.aucRoc > 0.99 ? "above" : "near"} the 0.99 benchmark from Almalki & Masud (2025).` },
                  { tag: "SHAP", color: C.purple, text: `Most predictive feature: '${d.shap[0].feature}' (mean |SHAP| = ${d.shap[0].shap.toFixed(4)}). Top 3: ${d.shap.slice(0, 3).map(s => s.feature).join(", ")}.` },
                  { tag: "LIME", color: C.blue, text: `Highest-risk transaction: ${d.limeInsights[0].feat} (weight=${d.limeInsights[0].weight.toFixed(3)}), ${d.limeInsights[1].feat} (weight=${d.limeInsights[1].weight.toFixed(3)}).` },
                  { tag: "RISK", color: C.red, text: `${d.riskCounts.Critical} critical-risk payments totalling $${d.valueAtRisk.toLocaleString()} require immediate review.` },
                  { tag: "SMOTE", color: C.green, text: `Training imbalance corrected via SMOTE (10% sampling) — recall improved vs under-sampling per Almalki & Masud findings.` },
                  { tag: "MISSED", color: C.amber, text: `${d.fn} fraud transactions missed (false negatives). Lowering threshold from 0.35 → 0.25 would recover ~${Math.round(d.fn * 0.6)} but increase false positives.` },
                ].map((ins, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, padding: "10px 12px", background: C.surface, borderRadius: 6, animation: `fadeUp 0.4s ease ${i * 60}ms both` }}>
                    <span style={{ color: ins.color, fontWeight: 700, fontSize: 10, minWidth: 52, marginTop: 1 }}>[{ins.tag}]</span>
                    <span style={{ fontSize: 11, color: C.mutedLight, lineHeight: 1.6 }}>{ins.text}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* ══════════════ MODEL TAB ══════════════ */}
        {tab === "model" && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* Architecture */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>Model Architecture</SectionTitle>
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 14 }}>Almalki & Masud (2025) · arXiv:2505.10050</div>
                {[
                  { layer: "Layer 1", label: "Base Learner: XGBoost", detail: "n=500, lr=0.05, depth=6, subsample=0.85", color: C.blue },
                  { layer: "Layer 1", label: "Base Learner: LightGBM", detail: "n=500, lr=0.05, num_leaves=63, colsample=0.85", color: C.teal },
                  { layer: "Layer 1", label: "Base Learner: CatBoost", detail: "iter=500, lr=0.05, depth=6, eval=AUC", color: C.purple },
                  { layer: "Layer 2", label: "Meta-Learner: Random Forest", detail: "n=200, max_depth=5 · tree beats LR (Btoush 2025)", color: C.green },
                  { layer: "CV", label: "Stacking Strategy", detail: "StratifiedKFold(5) · predict_proba · no passthrough", color: C.amber },
                  { layer: "IMBAL", label: "SMOTE Over-sampling", detail: "sampling_strategy=0.1, k_neighbors=5", color: C.red },
                ].map((m, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, animation: `fadeUp 0.3s ease ${i * 50}ms both` }}>
                    <div style={{ minWidth: 50, fontSize: 9, color: m.color, border: `1px solid ${m.color}44`, borderRadius: 3, padding: "2px 6px", textAlign: "center", height: "fit-content", marginTop: 2 }}>{m.layer}</div>
                    <div>
                      <div style={{ fontSize: 12, color: C.white, fontWeight: 500 }}>{m.label}</div>
                      <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{m.detail}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Model comparison */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
                <SectionTitle>Model Comparison — AUC-ROC</SectionTitle>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={Object.entries(d.modelAucs).map(([k, v]) => ({ name: k, auc: v }))} layout="vertical">
                    <CartesianGrid strokeDasharray="2 4" stroke={C.border} horizontal={false} />
                    <XAxis type="number" domain={[0.96, 0.99]} stroke={C.muted} fontSize={10} tickFormatter={v => v.toFixed(3)} />
                    <YAxis type="category" dataKey="name" stroke={C.muted} fontSize={10} width={70} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="auc" radius={[0, 3, 3, 0]} name="AUC-ROC">
                      {[C.blue, C.teal, C.purple, C.red].map((c, i) => <Cell key={i} fill={c} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 8 }}>
                  Stacking consistently outperforms each base learner — RF meta-learner captures non-linear interactions between model outputs.
                </div>
              </div>
            </div>

            {/* API integration guide */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 }}>
              <SectionTitle>API Service Integration</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 10 }}>Deploy the Python model as a FastAPI service, then update API_BASE at the top of this file.</div>
                  {[
                    { method: "POST", path: "/api/score", desc: "Score a batch of transactions" },
                    { method: "GET", path: "/api/stats", desc: "Model metrics + aggregates" },
                    { method: "GET", path: "/api/shap", desc: "Feature importance" },
                    { method: "POST", path: "/api/explain/{tx_id}", desc: "LIME explanation for one transaction" },
                  ].map((ep, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, padding: "8px 10px", background: C.surface, borderRadius: 5 }}>
                      <span style={{ color: ep.method === "POST" ? C.amber : C.green, fontSize: 10, minWidth: 36, fontWeight: 700 }}>{ep.method}</span>
                      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: C.mutedLight, flex: 1 }}>{ep.path}</span>
                      <span style={{ fontSize: 10, color: C.muted }}>{ep.desc}</span>
                    </div>
                  ))}
                </div>
                <div style={{ background: C.surface, borderRadius: 6, padding: 14, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: C.muted, lineHeight: 1.9 }}>
                  <span style={{ color: C.teal }}># FastAPI wrapper for cashgpt_fraud_detection.py</span>{"\n"}
                  <span style={{ color: C.purple }}>from</span> fastapi <span style={{ color: C.purple }}>import</span> FastAPI{"\n"}
                  <span style={{ color: C.purple }}>import</span> joblib, numpy <span style={{ color: C.purple }}>as</span> np{"\n\n"}
                  app = FastAPI(){"\n"}
                  stack = joblib.load(<span style={{ color: C.green }}>"cashgpt_stack.pkl"</span>){"\n\n"}
                  @app.post(<span style={{ color: C.green }}>"/api/score"</span>){"\n"}
                  <span style={{ color: C.purple }}>async def</span> score(payload: dict):{"\n"}
                  {"  "}X = np.array(payload[<span style={{ color: C.green }}>"features"</span>]){"\n"}
                  {"  "}proba = stack.predict_proba(X)[:,<span style={{ color: C.amber }}>1</span>]{"\n"}
                  {"  "}<span style={{ color: C.purple }}>return</span> {"{"}  <span style={{ color: C.green }}>"scores"</span>: proba.tolist() {"}"}
                </div>
              </div>
            </div>

            {/* Dataset info */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, marginTop: 16 }}>
              <SectionTitle>Dataset — ULB Credit Card Fraud Detection</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                {[
                  ["Total Transactions", d.totalTx.toLocaleString(), C.white],
                  ["Fraud Cases", d.fraudCount, C.red],
                  ["Fraud Rate", `${(d.fraudCount / d.totalTx * 100).toFixed(3)}%`, C.amber],
                  ["Features", "31 (V1–V28 + Time + Amount + Class)", C.blue],
                ].map(([label, val, col], i) => (
                  <div key={i} style={{ padding: "12px 14px", background: C.surface, borderRadius: 6 }}>
                    <div style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: col, fontFamily: "'IBM Plex Mono', monospace" }}>{val}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 12, padding: "10px 14px", background: C.surface, borderRadius: 6, fontSize: 10, color: C.muted, lineHeight: 1.8 }}>
                V1–V28 are PCA-transformed features (anonymised). Amount and Time are raw. Class is 0=normal, 1=fraud. The script adds: <span style={{ color: C.mutedLight }}>Amount_log, Amount_zscore, Hour, V1_V2, V3_V4, V14_V17</span> as engineered features.
              </div>
            </div>
          </>
        )}

        {/* Footer */}
        <div style={{ marginTop: 24, padding: "12px 0", borderTop: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", fontSize: 10, color: C.muted }}>
          <span>CashGPT · Almalki & Masud (2025) arXiv:2505.10050 · XGBoost + LightGBM + CatBoost Stacking · SHAP + LIME · ULB Dataset</span>
          <span style={{ color: apiStatus === "live" ? C.green : C.amber }}>{apiStatus === "live" ? "● Connected to live API" : "◎ Running on mock data — deploy FastAPI to connect live model"}</span>
        </div>
      </div>
    </div>
  );
}
