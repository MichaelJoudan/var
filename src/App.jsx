import { useState, useCallback, useRef, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, AreaChart, Area,
  PieChart, Pie, Cell, ScatterChart, Scatter, ReferenceLine
} from "recharts";

/* ═══════════════════════════════════════════════════════
   MATH ENGINE
   ═══════════════════════════════════════════════════════ */

// Normal distribution helpers (replaces scipy.stats.norm)
const normalPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

// Rational approximation for inverse normal CDF (Abramowitz & Stegun / Peter Acklam)
const normalPPF = (p) => {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const pLow = 0.02425, pHigh = 1 - pLow;
  let q, r;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (p <= pHigh) {
    q = p - 0.5; r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
};

// Chi-squared CDF (1 df) for Kupiec test
const chi2CDF1 = (x) => {
  if (x <= 0) return 0;
  // erf approximation
  const t = Math.sqrt(x / 2);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const p = 0.3275911;
  const t2 = 1 / (1 + p * t);
  const erf = 1 - (((((a5 * t2 + a4) * t2) + a3) * t2 + a2) * t2 + a1) * t2 * Math.exp(-t * t);
  return erf;
};

const percentile = (arr, p) => {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
};

const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
const std = (arr, ddof = 1) => {
  const m = mean(arr);
  const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - ddof);
  return Math.sqrt(v);
};

// Box-Muller for normal random numbers
const randn = () => {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

// ── VaR methods ──
const historicalVaR = (returns, cl = 0.95) => {
  const p = (1 - cl) * 100;
  return -percentile(returns, p);
};

const parametricVaR = (returns, cl = 0.95) => {
  const sigma = std(returns);
  const z = normalPPF(cl);
  return sigma * z;
};

const monteCarloVaR = (returns, cl = 0.95, nSim = 50000) => {
  const sigma = std(returns);
  const sims = Array.from({ length: nSim }, () => randn() * sigma);
  const p = (1 - cl) * 100;
  return -percentile(sims, p);
};

// ── Expected Shortfall ──
const esHistorical = (returns, cl = 0.95) => {
  const var_ = historicalVaR(returns, cl);
  const tail = returns.filter(r => r < -var_);
  if (tail.length === 0) return var_;
  return -mean(tail);
};

const esParametric = (returns, cl = 0.95) => {
  const sigma = std(returns);
  const z = normalPPF(cl);
  return sigma * normalPDF(z) / (1 - cl);
};

const scaleVaR = (var1d, horizon) => var1d * Math.sqrt(horizon);

// ── Portfolio VaR ──
const covMatrix = (retArrays) => {
  const n = retArrays.length;
  const len = retArrays[0].length;
  const means = retArrays.map(r => mean(r));
  const cov = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < len; k++) {
        s += (retArrays[i][k] - means[i]) * (retArrays[j][k] - means[j]);
      }
      cov[i][j] = s / (len - 1);
      cov[j][i] = cov[i][j];
    }
  }
  return cov;
};

const portfolioVaRParametric = (weights, cov, cl = 0.95, value = 1) => {
  const n = weights.length;
  // Portfolio variance = w' * Σ * w
  let portVar = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      portVar += weights[i] * weights[j] * cov[i][j];
  const portVol = Math.sqrt(portVar);
  const z = normalPPF(cl);
  const diversifiedVaR = portVol * z * value;

  // Undiversified
  const indVols = cov.map((_, i) => Math.sqrt(cov[i][i]));
  const indVaRs = weights.map((w, i) => Math.abs(w) * indVols[i] * z * value);
  const undivVaR = indVaRs.reduce((s, v) => s + v, 0);

  // Component VaR
  const margVaR = weights.map((_, i) => {
    let mv = 0;
    for (let j = 0; j < n; j++) mv += cov[i][j] * weights[j];
    return (mv / portVol) * z * value;
  });
  const compVaR = weights.map((w, i) => w * margVaR[i]);

  return { diversifiedVaR, undivVaR, divBenefit: undivVaR - diversifiedVaR, compVaR, indVaRs };
};

// ── Rolling VaR + Backtest ──
const rollingVaR = (returns, window = 250, cl = 0.95) => {
  const estimates = [];
  for (let i = 0; i < returns.length; i++) {
    if (i < window) { estimates.push(null); continue; }
    const w = returns.slice(i - window, i);
    estimates.push(historicalVaR(w, cl));
  }
  return estimates;
};

const backtestVaR = (returns, varEst, cl = 0.95) => {
  const n = returns.length;
  let exceed = 0;
  for (let i = 0; i < n; i++) if (returns[i] < -varEst[i]) exceed++;
  const expectedRate = 1 - cl;
  const actualRate = exceed / n;
  const expectedExceed = n * expectedRate;

  let kupiecStat = NaN, pValue = NaN;
  if (exceed > 0 && exceed < n) {
    const lr = -2 * (
      (n - exceed) * Math.log(1 - expectedRate) + exceed * Math.log(expectedRate) -
      (n - exceed) * Math.log(1 - actualRate) - exceed * Math.log(actualRate)
    );
    kupiecStat = lr;
    pValue = 1 - chi2CDF1(lr);
  }

  const ratio = actualRate / expectedRate;
  const zone = ratio <= 1.5 ? "Green" : ratio <= 2.0 ? "Yellow" : "Red";

  return { n, exceed, expectedExceed, actualRate, expectedRate, kupiecStat, pValue, zone };
};


/* ═══════════════════════════════════════════════════════
   YAHOO FINANCE DATA FETCH (via free proxy / direct)
   ═══════════════════════════════════════════════════════ */

const periodToSeconds = (period) => {
  const now = Math.floor(Date.now() / 1000);
  const map = { "6mo": 15778800, "1y": 31557600, "2y": 63115200, "5y": 157788000, "10y": 315576000 };
  return { period1: now - (map[period] || 63115200), period2: now };
};

const fetchYahooData = async (ticker, period = "2y") => {
  const { period1, period2 } = periodToSeconds(period);
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?period1=${period1}&period2=${period2}&interval=1d`;

  // Try direct first, then CORS proxies
  const proxies = [
    url, // direct (works if CORS is okay)
    `https://corsproxy.io/?${encodeURIComponent(url)}`,
    `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  ];

  let lastErr;
  for (const proxyUrl of proxies) {
    try {
      const resp = await fetch(proxyUrl, { signal: AbortSignal.timeout(12000) });
      if (!resp.ok) continue;
      const data = await resp.json();
      const result = data?.chart?.result?.[0];
      if (!result) continue;

      const timestamps = result.timestamp;
      const closes = result.indicators.adjclose?.[0]?.adjclose || result.indicators.quote[0].close;

      // Build returns
      const dates = [];
      const closePrices = [];
      const returns = [];
      for (let i = 0; i < timestamps.length; i++) {
        if (closes[i] == null) continue;
        dates.push(new Date(timestamps[i] * 1000).toISOString().slice(0, 10));
        closePrices.push(closes[i]);
      }
      for (let i = 1; i < closePrices.length; i++) {
        returns.push((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]);
      }

      return {
        dates: dates.slice(1),
        closes: closePrices.slice(1),
        returns,
        ticker: ticker.toUpperCase()
      };
    } catch (e) { lastErr = e; }
  }
  throw new Error(`Failed to fetch ${ticker}: ${lastErr?.message || "All proxies failed"}`);
};


/* ═══════════════════════════════════════════════════════
   REACT APPLICATION
   ═══════════════════════════════════════════════════════ */

const ACCENT = "#e63946";
const BG_DARK = "#0b0f19";
const BG_CARD = "#111827";
const BG_CARD2 = "#1a2236";
const BORDER = "#1e293b";
const TEXT = "#e2e8f0";
const TEXT_DIM = "#94a3b8";
const GREEN = "#22c55e";
const AMBER = "#f59e0b";

const PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#264653", "#a8dadc", "#d4a5a5"];

// ── Tiny stat card ──
const StatCard = ({ label, value, sub, accent }) => (
  <div style={{
    background: BG_CARD2, border: `1px solid ${BORDER}`, borderRadius: 10, padding: "14px 18px",
    flex: "1 1 160px", minWidth: 160,
    borderTop: accent ? `3px solid ${accent}` : undefined
  }}>
    <div style={{ fontSize: 11, color: TEXT_DIM, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color: TEXT, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: TEXT_DIM, marginTop: 2 }}>{sub}</div>}
  </div>
);

// ── Section header ──
const SectionTitle = ({ num, title, subtitle }) => (
  <div style={{ marginBottom: 18 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{
        background: ACCENT, color: "#fff", fontWeight: 800, fontSize: 13,
        width: 28, height: 28, borderRadius: "50%", display: "inline-flex",
        alignItems: "center", justifyContent: "center", fontFamily: "'JetBrains Mono', monospace"
      }}>{num}</span>
      <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: TEXT, fontFamily: "'Space Grotesk', sans-serif" }}>{title}</h2>
    </div>
    {subtitle && <p style={{ margin: "4px 0 0 38px", color: TEXT_DIM, fontSize: 13 }}>{subtitle}</p>}
  </div>
);

// Correlation matrix as a table
const CorrTable = ({ matrix, labels }) => (
  <div style={{ overflowX: "auto" }}>
    <table style={{ borderCollapse: "collapse", fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
      <thead>
        <tr>
          <th style={{ padding: "6px 12px", color: TEXT_DIM }}></th>
          {labels.map(l => <th key={l} style={{ padding: "6px 12px", color: TEXT, fontWeight: 600 }}>{l}</th>)}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <td style={{ padding: "6px 12px", color: TEXT, fontWeight: 600 }}>{labels[i]}</td>
            {row.map((v, j) => {
              const abs = Math.abs(v);
              const bg = i === j ? "rgba(255,255,255,0.04)"
                : v > 0 ? `rgba(34,197,94,${abs * 0.35})`
                : `rgba(230,57,70,${abs * 0.35})`;
              return <td key={j} style={{ padding: "6px 12px", textAlign: "center", color: TEXT, background: bg, borderRadius: 4 }}>{v.toFixed(3)}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// ── Main App ──
export default function VaRApp() {
  // Input state
  const [ticker, setTicker] = useState("AAPL");
  const [confidence, setConfidence] = useState("0.95");
  const [period, setPeriod] = useState("2y");
  const [portTickers, setPortTickers] = useState("AAPL,MSFT,GOOGL,JPM");
  const [portWeights, setPortWeights] = useState("0.30,0.30,0.25,0.15");
  const [portValue, setPortValue] = useState("1000000");

  // Results state
  const [singleResult, setSingleResult] = useState(null);
  const [portResult, setPortResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);

  // UI state
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState(null);
  const resultsRef = useRef(null);

  const cl = parseFloat(confidence) || 0.95;
  const pv = parseFloat(portValue.replace(/[,$]/g, "")) || 1000000;

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setPortResult(null);
    setBacktestResult(null);

    try {
      // ── 1. Single stock ──
      setProgress(`Fetching ${ticker.toUpperCase()}…`);
      const stockData = await fetchYahooData(ticker.toUpperCase().trim(), period);
      const ret = stockData.returns;

      const varH = historicalVaR(ret, cl);
      const varP = parametricVaR(ret, cl);
      const varMC = monteCarloVaR(ret, cl);
      const esH = esHistorical(ret, cl);
      const esP = esParametric(ret, cl);
      const v10 = scaleVaR(varH, 10);

      setSingleResult({
        ticker: stockData.ticker,
        nObs: ret.length,
        startDate: stockData.dates[0],
        endDate: stockData.dates[ret.length - 1],
        meanReturn: mean(ret),
        volatility: std(ret, 0),
        minReturn: Math.min(...ret),
        maxReturn: Math.max(...ret),
        varH, varP, varMC, esH, esP, v10,
        returns: ret,
        dates: stockData.dates,
      });

      // ── 2. Portfolio ──
      const tickers = portTickers.split(",").map(t => t.trim().toUpperCase()).filter(Boolean);
      const rawW = portWeights.split(",").map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
      let weights = rawW.length === tickers.length ? rawW : tickers.map(() => 1 / tickers.length);
      const wSum = weights.reduce((a, b) => a + b, 0);
      if (Math.abs(wSum - 1) > 0.01) weights = weights.map(w => w / wSum);

      const CASH = new Set(["__CASH__", "__USD_CASH__", "__CASH_USD__"]);
      const riskyIdx = tickers.map((t, i) => CASH.has(t) ? -1 : i).filter(i => i >= 0);
      const riskyTickers = riskyIdx.map(i => tickers[i]);
      const riskyWeights = riskyIdx.map(i => weights[i]);

      setProgress(`Fetching portfolio data (${riskyTickers.join(", ")})…`);
      const allData = [];
      for (const t of riskyTickers) {
        const d = await fetchYahooData(t, period);
        allData.push(d);
      }

      // Align dates (intersection)
      let commonDates = new Set(allData[0].dates);
      for (const d of allData) commonDates = new Set([...commonDates].filter(x => d.dates.includes(x)));
      const sortedDates = [...commonDates].sort();

      const aligned = allData.map(d => {
        const dateMap = Object.fromEntries(d.dates.map((dt, i) => [dt, d.returns[i]]));
        return sortedDates.map(dt => dateMap[dt]);
      });

      const cov = covMatrix(aligned);
      const nAssets = riskyTickers.length;
      const stdDevs = Array.from({ length: nAssets }, (_, i) => Math.sqrt(cov[i][i]));
      const corr = cov.map((row, i) => row.map((v, j) => v / (stdDevs[i] * stdDevs[j])));

      const pvr = portfolioVaRParametric(riskyWeights, cov, cl, pv);

      // Historical portfolio returns
      const portReturns = sortedDates.map((_, di) => {
        let r = 0;
        for (let a = 0; a < nAssets; a++) r += aligned[a][di] * riskyWeights[a];
        return r;
      });
      const pVarHist = historicalVaR(portReturns, cl) * pv;
      const pEsHist = esHistorical(portReturns, cl) * pv;

      setPortResult({
        tickers, weights, riskyTickers, riskyWeights,
        nObs: sortedDates.length,
        corr, cov,
        diversifiedVaR: pvr.diversifiedVaR,
        undivVaR: pvr.undivVaR,
        divBenefit: pvr.divBenefit,
        compVaR: pvr.compVaR,
        indVaRs: pvr.indVaRs,
        varHist: pVarHist,
        esHist: pEsHist,
        portReturns, dates: sortedDates,
      });

      // ── 3. Backtest ──
      setProgress("Running backtest…");
      const window = Math.min(250, Math.floor(ret.length * 0.5));
      const rollVar = rollingVaR(ret, window, cl);
      const testRet = ret.slice(window);
      const testVar = rollVar.slice(window);
      const bt = backtestVaR(testRet, testVar, cl);

      // Build chart data for backtest
      const btChartData = testRet.map((r, i) => ({
        day: i,
        return: r * 100,
        var: -testVar[i] * 100,
        exceed: r < -testVar[i]
      }));

      setBacktestResult({ ...bt, chartData: btChartData, window });

      setProgress("");
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 200);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setProgress("");
    }
  }, [ticker, confidence, period, portTickers, portWeights, portValue, cl, pv]);

  // Build chart data
  const singleVarCompare = singleResult ? [
    { method: "Historical", var: singleResult.varH * 100, es: singleResult.esH * 100 },
    { method: "Parametric", var: singleResult.varP * 100, es: singleResult.esP * 100 },
    { method: "Monte Carlo", var: singleResult.varMC * 100, es: null },
  ] : [];

  const singleRetHist = singleResult ? (() => {
    const bins = 50;
    const ret = singleResult.returns.map(r => r * 100);
    const minR = Math.min(...ret), maxR = Math.max(...ret);
    const step = (maxR - minR) / bins;
    const counts = Array(bins).fill(0);
    ret.forEach(r => { const b = Math.min(Math.floor((r - minR) / step), bins - 1); counts[b]++; });
    return counts.map((c, i) => ({
      bin: (minR + (i + 0.5) * step).toFixed(2),
      count: c,
      x: minR + (i + 0.5) * step,
    }));
  })() : [];

  const portPieData = portResult ? portResult.tickers.map((t, i) => ({
    name: t, value: Math.abs(portResult.weights[i]) * 100
  })) : [];

  const portVarCompare = portResult ? [
    { label: "Diversified (Parametric)", value: portResult.diversifiedVaR },
    { label: "Diversified (Historical)", value: portResult.varHist },
    { label: "Undiversified", value: portResult.undivVaR },
  ] : [];

  const compVarData = portResult ? portResult.riskyTickers.map((t, i) => ({
    name: t, value: portResult.compVaR[i]
  })) : [];

  // ── Return timeseries for chart (downsample if huge) ──
  const retTimeseries = singleResult ? (() => {
    const r = singleResult.returns;
    const d = singleResult.dates;
    const step = r.length > 500 ? Math.floor(r.length / 500) : 1;
    const data = [];
    for (let i = 0; i < r.length; i += step) {
      data.push({ date: d[i], return: +(r[i] * 100).toFixed(3) });
    }
    return data;
  })() : [];

  return (
    <div style={{
      minHeight: "100vh",
      background: `linear-gradient(180deg, ${BG_DARK} 0%, #0d1320 50%, #0f172a 100%)`,
      color: TEXT,
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
    }}>
      {/* Load fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        input, select { transition: border-color 0.2s; }
        input:focus, select:focus { border-color: ${ACCENT} !important; outline: none; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${BG_DARK}; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        padding: "32px 0 24px", textAlign: "center",
        borderBottom: `1px solid ${BORDER}`,
        background: `radial-gradient(ellipse at 50% 0%, rgba(230,57,70,0.08), transparent 60%)`,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 6 }}>
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <rect x="2" y="18" width="6" height="12" rx="1" fill={ACCENT} opacity="0.6"/>
            <rect x="10" y="12" width="6" height="18" rx="1" fill={ACCENT} opacity="0.8"/>
            <rect x="18" y="6" width="6" height="24" rx="1" fill={ACCENT}/>
            <rect x="26" y="2" width="4" height="28" rx="1" fill="#fff" opacity="0.3"/>
            <line x1="0" y1="30" x2="32" y2="30" stroke={TEXT_DIM} strokeWidth="1"/>
          </svg>
          <h1 style={{
            margin: 0, fontSize: 32, fontWeight: 700,
            fontFamily: "'Space Grotesk', sans-serif",
            background: `linear-gradient(135deg, ${TEXT}, ${ACCENT})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
          }}>VaR Analytics</h1>
        </div>
        <p style={{ margin: 0, color: TEXT_DIM, fontSize: 14 }}>
          Value-at-Risk &amp; Expected Shortfall · Historical, Parametric &amp; Monte Carlo
        </p>
      </header>

      {/* ── Input Panel ── */}
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "28px 20px" }}>
        <div style={{
          background: BG_CARD, border: `1px solid ${BORDER}`, borderRadius: 14,
          padding: 28, marginBottom: 28
        }}>
          <h3 style={{ margin: "0 0 18px", fontSize: 16, fontWeight: 600, color: TEXT, fontFamily: "'Space Grotesk', sans-serif" }}>
            Configuration
          </h3>

          {/* Row 1: Confidence / Period / Single stock */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 16 }}>
            {[
              { label: "Confidence Level", value: confidence, set: setConfidence, ph: "0.95", w: 130 },
              { label: "Data Period", value: period, set: setPeriod, ph: "2y", w: 110, type: "select", opts: ["6mo","1y","2y","5y","10y"] },
              { label: "Single Stock Ticker", value: ticker, set: setTicker, ph: "AAPL", w: 140 },
            ].map(({ label, value, set, ph, w, type, opts }) => (
              <div key={label} style={{ flex: `0 1 ${w}px` }}>
                <label style={{ display: "block", fontSize: 11, color: TEXT_DIM, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.8 }}>{label}</label>
                {type === "select" ? (
                  <select value={value} onChange={e => set(e.target.value)} style={{
                    width: "100%", padding: "8px 10px", background: BG_DARK, color: TEXT,
                    border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 14,
                    fontFamily: "'JetBrains Mono', monospace"
                  }}>
                    {opts.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input value={value} onChange={e => set(e.target.value)} placeholder={ph} style={{
                    width: "100%", padding: "8px 10px", background: BG_DARK, color: TEXT,
                    border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 14,
                    fontFamily: "'JetBrains Mono', monospace"
                  }} />
                )}
              </div>
            ))}
          </div>

          {/* Row 2: Portfolio */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 20 }}>
            <div style={{ flex: "1 1 260px" }}>
              <label style={{ display: "block", fontSize: 11, color: TEXT_DIM, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.8 }}>
                Portfolio Tickers <span style={{ opacity: 0.5 }}>(comma-separated, use __CASH__ for cash)</span>
              </label>
              <input value={portTickers} onChange={e => setPortTickers(e.target.value)} style={{
                width: "100%", padding: "8px 10px", background: BG_DARK, color: TEXT,
                border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 14,
                fontFamily: "'JetBrains Mono', monospace"
              }} />
            </div>
            <div style={{ flex: "1 1 220px" }}>
              <label style={{ display: "block", fontSize: 11, color: TEXT_DIM, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.8 }}>Weights (comma-separated, sum to 1)</label>
              <input value={portWeights} onChange={e => setPortWeights(e.target.value)} style={{
                width: "100%", padding: "8px 10px", background: BG_DARK, color: TEXT,
                border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 14,
                fontFamily: "'JetBrains Mono', monospace"
              }} />
            </div>
            <div style={{ flex: "0 1 160px" }}>
              <label style={{ display: "block", fontSize: 11, color: TEXT_DIM, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.8 }}>Portfolio Value ($)</label>
              <input value={portValue} onChange={e => setPortValue(e.target.value)} style={{
                width: "100%", padding: "8px 10px", background: BG_DARK, color: TEXT,
                border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 14,
                fontFamily: "'JetBrains Mono', monospace"
              }} />
            </div>
          </div>

          {/* Run button */}
          <button onClick={runAnalysis} disabled={loading} style={{
            padding: "12px 36px", fontSize: 15, fontWeight: 700, color: "#fff",
            background: loading ? "#4b5563" : ACCENT, border: "none", borderRadius: 8,
            cursor: loading ? "wait" : "pointer", fontFamily: "'Space Grotesk', sans-serif",
            letterSpacing: 0.5, transition: "background 0.2s",
          }}>
            {loading ? "Running…" : "Run VaR Analysis"}
          </button>

          {progress && (
            <span style={{ marginLeft: 16, color: TEXT_DIM, fontSize: 13, animation: "pulse 1.5s infinite" }}>
              {progress}
            </span>
          )}
          {error && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(230,57,70,0.12)", border: `1px solid ${ACCENT}`, borderRadius: 8, color: ACCENT, fontSize: 13 }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* ═══════ RESULTS ═══════ */}
        <div ref={resultsRef}>

          {/* ── 1. Single Stock ── */}
          {singleResult && (
            <div style={{ background: BG_CARD, border: `1px solid ${BORDER}`, borderRadius: 14, padding: 28, marginBottom: 28 }}>
              <SectionTitle num="1" title={`Single Stock VaR — ${singleResult.ticker}`} subtitle={`${singleResult.startDate} to ${singleResult.endDate} · ${singleResult.nObs} observations`} />

              {/* Stat cards */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                <StatCard label="Mean Daily Return" value={`${(singleResult.meanReturn * 100).toFixed(4)}%`} accent="#457b9d" />
                <StatCard label="Daily Volatility" value={`${(singleResult.volatility * 100).toFixed(4)}%`} accent="#e9c46a" />
                <StatCard label="Worst Day" value={`${(singleResult.minReturn * 100).toFixed(2)}%`} accent={ACCENT} />
                <StatCard label="Best Day" value={`${(singleResult.maxReturn * 100).toFixed(2)}%`} accent={GREEN} />
              </div>

              {/* VaR table */}
              <div style={{ overflowX: "auto", marginBottom: 24 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14, fontFamily: "'JetBrains Mono', monospace" }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${BORDER}` }}>
                      <th style={{ padding: "8px 12px", textAlign: "left", color: TEXT_DIM, fontWeight: 600 }}>Method</th>
                      <th style={{ padding: "8px 12px", textAlign: "right", color: TEXT_DIM }}>1-Day VaR</th>
                      <th style={{ padding: "8px 12px", textAlign: "right", color: TEXT_DIM }}>10-Day VaR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Historical", singleResult.varH],
                      ["Parametric", singleResult.varP],
                      ["Monte Carlo", singleResult.varMC],
                    ].map(([m, v]) => (
                      <tr key={m} style={{ borderBottom: `1px solid ${BORDER}` }}>
                        <td style={{ padding: "8px 12px", color: TEXT }}>{m}</td>
                        <td style={{ padding: "8px 12px", textAlign: "right", color: ACCENT, fontWeight: 600 }}>{(v * 100).toFixed(4)}%</td>
                        <td style={{ padding: "8px 12px", textAlign: "right", color: TEXT }}>{(scaleVaR(v, 10) * 100).toFixed(4)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* ES table */}
              <div style={{ overflowX: "auto", marginBottom: 24 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14, fontFamily: "'JetBrains Mono', monospace" }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${BORDER}` }}>
                      <th style={{ padding: "8px 12px", textAlign: "left", color: TEXT_DIM, fontWeight: 600 }}>Expected Shortfall</th>
                      <th style={{ padding: "8px 12px", textAlign: "right", color: TEXT_DIM }}>ES (%)</th>
                      <th style={{ padding: "8px 12px", textAlign: "right", color: TEXT_DIM }}>ES / VaR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Historical", singleResult.esH, singleResult.varH],
                      ["Parametric", singleResult.esP, singleResult.varP],
                    ].map(([m, es, v]) => (
                      <tr key={m} style={{ borderBottom: `1px solid ${BORDER}` }}>
                        <td style={{ padding: "8px 12px", color: TEXT }}>{m}</td>
                        <td style={{ padding: "8px 12px", textAlign: "right", color: "#f4a261", fontWeight: 600 }}>{(es * 100).toFixed(4)}%</td>
                        <td style={{ padding: "8px 12px", textAlign: "right", color: TEXT }}>{(es / v).toFixed(2)}x</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Charts row */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 20 }}>
                {/* Return distribution */}
                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Return Distribution</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={singleRetHist}>
                      <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                      <XAxis dataKey="bin" tick={{ fontSize: 10, fill: TEXT_DIM }} interval="preserveStartEnd" />
                      <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} />
                      <Tooltip contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                      <Bar dataKey="count" fill="#457b9d" radius={[2, 2, 0, 0]} />
                      <ReferenceLine x={(-singleResult.varH * 100).toFixed(2)} stroke={ACCENT} strokeDasharray="4 4" label={{ value: "VaR", fill: ACCENT, fontSize: 11 }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* VaR comparison */}
                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>VaR vs Expected Shortfall</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={singleVarCompare}>
                      <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                      <XAxis dataKey="method" tick={{ fontSize: 11, fill: TEXT_DIM }} />
                      <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} unit="%" />
                      <Tooltip contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                      <Bar dataKey="var" fill="#457b9d" name="VaR" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="es" fill="#e9c46a" name="ES" radius={[3, 3, 0, 0]} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Return time series */}
              <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16, marginTop: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Returns Over Time (%)</div>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={retTimeseries}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: TEXT_DIM }} interval="preserveStartEnd" minTickGap={60} />
                    <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} unit="%" />
                    <Tooltip contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                    <Area type="monotone" dataKey="return" stroke="#457b9d" fill="rgba(69,123,157,0.15)" strokeWidth={1} dot={false} />
                    <ReferenceLine y={-singleResult.varH * 100} stroke={ACCENT} strokeDasharray="4 4" label={{ value: `VaR ${(singleResult.varH * 100).toFixed(2)}%`, fill: ACCENT, fontSize: 10 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── 2. Portfolio ── */}
          {portResult && (
            <div style={{ background: BG_CARD, border: `1px solid ${BORDER}`, borderRadius: 14, padding: 28, marginBottom: 28 }}>
              <SectionTitle num="2" title="Portfolio VaR Analysis" subtitle={`${portResult.nObs} overlapping trading days · Portfolio value: $${pv.toLocaleString()}`} />

              {/* Allocation list */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
                {portResult.tickers.map((t, i) => (
                  <span key={t} style={{
                    padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                    background: `${PALETTE[i % PALETTE.length]}22`, color: PALETTE[i % PALETTE.length],
                    border: `1px solid ${PALETTE[i % PALETTE.length]}44`, fontFamily: "'JetBrains Mono', monospace"
                  }}>
                    {t} {(portResult.weights[i] * 100).toFixed(1)}%
                  </span>
                ))}
              </div>

              {/* Stat cards */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                <StatCard label="Diversified VaR (Param)" value={`$${portResult.diversifiedVaR.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} accent={ACCENT} />
                <StatCard label="Diversified VaR (Hist)" value={`$${portResult.varHist.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} accent="#457b9d" />
                <StatCard label="Undiversified VaR" value={`$${portResult.undivVaR.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} accent="#f4a261" />
                <StatCard label="Diversification Benefit" value={`$${portResult.divBenefit.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} accent={GREEN} />
                <StatCard label="Expected Shortfall" value={`$${portResult.esHist.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} accent="#e9c46a" />
              </div>

              {/* Charts */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20, marginBottom: 24 }}>
                {/* Pie */}
                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Allocation</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={portPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name} ${value.toFixed(1)}%`} labelLine={{ stroke: TEXT_DIM }} strokeWidth={1} stroke={BG_CARD}>
                        {portPieData.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                      </Pie>
                      <Tooltip contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* VaR comparison bar */}
                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Diversified vs Undiversified</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={portVarCompare}>
                      <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                      <XAxis dataKey="label" tick={{ fontSize: 10, fill: TEXT_DIM }} />
                      <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                      <Tooltip formatter={v => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {portVarCompare.map((_, i) => <Cell key={i} fill={[ACCENT, "#457b9d", "#f4a261"][i]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Component VaR + Correlation */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20 }}>
                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Risk Contribution by Asset</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={compVarData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: TEXT_DIM }} />
                      <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                      <Tooltip formatter={v => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]} >
                        {compVarData.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                      </Bar>
                      <ReferenceLine y={0} stroke={TEXT_DIM} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Correlation Matrix</div>
                  <CorrTable matrix={portResult.corr} labels={portResult.riskyTickers} />
                </div>
              </div>
            </div>
          )}

          {/* ── 3. Backtest ── */}
          {backtestResult && (
            <div style={{ background: BG_CARD, border: `1px solid ${BORDER}`, borderRadius: 14, padding: 28, marginBottom: 28 }}>
              <SectionTitle num="3" title={`Backtesting — ${singleResult?.ticker || ticker}`} subtitle={`Rolling ${backtestResult.window}-day window · ${backtestResult.n} test observations`} />

              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                <StatCard label="Expected Exceedances" value={backtestResult.expectedExceed.toFixed(1)} />
                <StatCard label="Actual Exceedances" value={backtestResult.exceed}
                  accent={backtestResult.zone === "Green" ? GREEN : backtestResult.zone === "Yellow" ? AMBER : ACCENT} />
                <StatCard label="Exceedance Rate" value={`${(backtestResult.actualRate * 100).toFixed(2)}%`}
                  sub={`expected: ${(backtestResult.expectedRate * 100).toFixed(2)}%`} />
                <StatCard label="Kupiec p-value" value={isNaN(backtestResult.pValue) ? "N/A" : backtestResult.pValue.toFixed(4)} />
                <StatCard label="Traffic Light" value={backtestResult.zone}
                  accent={backtestResult.zone === "Green" ? GREEN : backtestResult.zone === "Yellow" ? AMBER : ACCENT} />
              </div>

              <div style={{
                padding: "10px 16px", borderRadius: 8, marginBottom: 20, fontSize: 14,
                background: backtestResult.pValue > 0.05 ? "rgba(34,197,94,0.1)" : "rgba(230,57,70,0.1)",
                border: `1px solid ${backtestResult.pValue > 0.05 ? GREEN : ACCENT}`,
                color: backtestResult.pValue > 0.05 ? GREEN : ACCENT,
              }}>
                {!isNaN(backtestResult.pValue) && backtestResult.pValue > 0.05
                  ? "✓ Model passes backtest (not rejected at 5% significance)"
                  : "✗ Model fails backtest (rejected at 5% significance)"}
              </div>

              {/* Backtest chart */}
              <div style={{ background: BG_CARD2, borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: TEXT_DIM, marginBottom: 10 }}>Returns vs VaR Threshold</div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={backtestResult.chartData.filter((_, i) => i % Math.max(1, Math.floor(backtestResult.chartData.length / 600)) === 0)}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BORDER} />
                    <XAxis dataKey="day" tick={{ fontSize: 9, fill: TEXT_DIM }} />
                    <YAxis tick={{ fontSize: 10, fill: TEXT_DIM }} unit="%" />
                    <Tooltip contentStyle={{ background: BG_DARK, border: `1px solid ${BORDER}`, borderRadius: 6, fontSize: 12 }} />
                    <Line type="monotone" dataKey="return" stroke="#457b9d" strokeWidth={1} dot={false} name="Return" />
                    <Line type="monotone" dataKey="var" stroke={ACCENT} strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="VaR Threshold" />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", padding: "24px 0 40px", color: TEXT_DIM, fontSize: 12 }}>
          VaR Analytics · Powered by Yahoo Finance data · All calculations run client-side in your browser
        </div>
      </div>
    </div>
  );
}
