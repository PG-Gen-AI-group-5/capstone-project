/* Trading with AI: the desk app. Talks to the FastAPI backend in src/api (python -m src.api).
   Based on the initial design (docs/design/trading-with-ai.html): same layout, charts and screens, with the
   simulated intraday feed replaced by the paper-trading desk that replays history one trading day at a time. */
(() => {
"use strict";

/* ================= Helpers ================= */
const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const f1 = (n) => n.toFixed(1);
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } },
};

/* ================= Formatting (₹, dates) ================= */
const nf2 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const nf1 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const dayFmt = new Intl.DateTimeFormat("en-IN", { timeZone: "UTC", day: "numeric", month: "short" });
const dayYearFmt = new Intl.DateTimeFormat("en-IN", { timeZone: "UTC", day: "numeric", month: "short", year: "numeric" });
const dayLongFmt = new Intl.DateTimeFormat("en-IN", { timeZone: "UTC", weekday: "short", day: "numeric", month: "short", year: "numeric" });
const monthFmt = new Intl.DateTimeFormat("en-IN", { timeZone: "UTC", month: "short" });
const isoDate = (iso) => new Date(String(iso).slice(0, 10) + "T00:00:00Z");
const dateLabel = (iso) => (iso ? dayFmt.format(isoDate(iso)) : "—");
const dateYear = (iso) => (iso ? dayYearFmt.format(isoDate(iso)) : "—");
const dayLong = (iso) => (iso ? dayLongFmt.format(isoDate(iso)) : "—");

const signOf = (v) => (v > 0 ? "+" : v < 0 ? "−" : "");
const money = (v) => (v == null ? "—" : "₹" + nf2.format(v));
const signedMoney = (v) => (v == null ? "—" : signOf(v) + "₹" + nf2.format(Math.abs(v)));
const signedMoney0 = (v) => signOf(Math.round(v)) + "₹" + nf0.format(Math.abs(v));
const signedPct = (v, d = 2) => (v == null ? "—" : signOf(+v.toFixed(d)) + Math.abs(v).toFixed(d) + "%");
const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
const toneOf = (v) => (v > 0 ? "up" : v < 0 ? "down" : "flat");
function compactINR(v) {
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e7) return `${s}₹${+(a / 1e7).toFixed(2)}Cr`;
  if (a >= 1e5) return `${s}₹${+(a / 1e5).toFixed(1)}L`;
  if (a >= 1e3) return `${s}₹${+(a / 1e3).toFixed(1)}K`;
  return `${s}₹${nf0.format(a)}`;
}
const signedCompact = (v) => (v > 0 ? "+" : "") + compactINR(v);
const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + "s"}`;
const initials = (name) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);

const SETUP_LABEL = { pullback_uptrend: "Pullback", breakout: "Breakout", oversold_bounce: "Oversold bounce", none: "No setup" };
const SOURCE_LABEL = { agents: "Agents' plan", manual: "Your order" };
const EXIT_LABEL = { stop: "Stop", gap_stop: "Gap through stop", target: "Target", time: "Holding period over", exit_monitor: "Exit Monitor", manual: "Your exit" };
const KIND_LABEL = { entry: "Entry · open", no_fill: "Entry · not filled", exit: "Exit · open", gap_stop: "Gap stop · open", target: "Target", stop: "Stop", time: "Time exit · close" };
const AT_LABEL = { open: "09:15", intraday: "Intraday", close: "15:30" };
const PHASE = { new: ["Planning", "planning"], morning: ["Pre-open", ""], closed: ["Closed", "closed"], finished: ["Finished", "done"] };
const ACTIVE = ["queued", "running", "cancelling"];

/* ================= API ================= */
class ApiError extends Error { constructor(status, detail) { super(detail); this.status = status; } }
async function api(method, path, body) {
  const res = await fetch(path, {
    method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
    throw new ApiError(res.status, msg || `Request failed (${res.status})`);
  }
  return data;
}
const GET = (p) => api("GET", p);
const POST = (p, b) => api("POST", p, b || {});
const DEL = (p) => api("DELETE", p);
const deskPath = (p = "") => `/api/desks/${state.deskId}${p}`;

/* ================= State ================= */
const state = {
  meta: null, desks: [], deskId: null, desk: null, view: "dashboard",
  watch: null, selected: null, detail: null, trace: null,
  feed: [], lastEventId: 0, feedFilter: "all",
  book: null, trades: [], days: [],
  hist: { period: 0, symbol: "all", result: "all", limit: 25 },
  ana: { period: 0, symbol: "all" },
  jobId: null, jobDone: -1, pollTimer: 0, confirm: null, loadToken: 0,
};

/* ================= Tooltip & toast ================= */
const tip = $("tip");
let tipOwner = null;
function showTip(title, rows, x, y, owner) {
  tipOwner = owner || null;
  tip.replaceChildren(el("div", "tip-title", title));
  for (const r of rows) {
    const row = el("div", "tip-row"), key = el("span", "tip-key");
    if (r.key) key.style.background = r.key;
    row.append(key, el("strong", null, r.value), el("span", "tip-label", r.label));
    tip.append(row);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight, pad = 14;
  let left = x + pad, top = y + pad;
  if (left + w > window.innerWidth - 8) left = x - w - pad;
  if (top + h > window.innerHeight - 8) top = y - h - pad;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, top) + "px";
}
function hideTip() { tip.hidden = true; tipOwner = null; }
let toastTimer = 0;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
}

/* ================= Chart primitives ================= */
function niceTicks(lo, hi, count) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step);
  return { ticks, step };
}
function niceDomain(lo, hi, count) {
  if (lo === hi) { hi = lo + 1; }
  const { step } = niceTicks(lo, hi, count);
  const dlo = Math.floor(lo / step + 1e-9) * step, dhi = Math.ceil(hi / step - 1e-9) * step;
  return { lo: dlo, hi: dhi, ...niceTicks(dlo, dhi, count) };
}
function barPath(x, y, w, h, end) {
  if (w <= 0 || h <= 0) return "";
  const r = Math.min(4, w / 2, h / 2);
  const X = (n) => f1(n), Y = X;
  switch (end) {
    case "right": return `M${X(x)} ${Y(y)}H${X(x + w - r)}A${r} ${r} 0 0 1 ${X(x + w)} ${Y(y + r)}V${Y(y + h - r)}A${r} ${r} 0 0 1 ${X(x + w - r)} ${Y(y + h)}H${X(x)}Z`;
    case "left": return `M${X(x + w)} ${Y(y)}H${X(x + r)}A${r} ${r} 0 0 0 ${X(x)} ${Y(y + r)}V${Y(y + h - r)}A${r} ${r} 0 0 0 ${X(x + r)} ${Y(y + h)}H${X(x + w)}Z`;
    case "top": return `M${X(x)} ${Y(y + h)}V${Y(y + r)}A${r} ${r} 0 0 1 ${X(x + r)} ${Y(y)}H${X(x + w - r)}A${r} ${r} 0 0 1 ${X(x + w)} ${Y(y + r)}V${Y(y + h)}Z`;
    default: return `M${X(x)} ${Y(y)}V${Y(y + h - r)}A${r} ${r} 0 0 0 ${X(x + r)} ${Y(y + h)}H${X(x + w - r)}A${r} ${r} 0 0 0 ${X(x + w)} ${Y(y + h - r)}V${Y(y)}Z`;
  }
}
const svgWrap = (W, H, label, inner) => `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${inner}</svg>`;

function sparkline(values, w, h) {
  if (!values || values.length < 2) return "";
  const lo = Math.min(...values), hi = Math.max(...values), pad = 3, span = hi - lo || 1, n = values.length;
  const x = (i) => pad + (i / (n - 1)) * (w - pad * 2);
  const y = (v) => pad + (1 - (v - lo) / span) * (h - pad * 2);
  const tone = values[n - 1] >= values[0] ? "var(--up)" : "var(--down)";
  const pts = values.map((v, i) => `${f1(x(i))},${f1(y(v))}`).join(" ");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">` +
    `<polyline points="${pts}" fill="none" stroke="${tone}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>` +
    `<circle cx="${f1(x(n - 1))}" cy="${f1(y(values[n - 1]))}" r="2.5" fill="${tone}"/></svg>`;
}

/* Crosshair charts (price, equity): pointer + arrow keys */
function bindCrosshair(box) {
  box.addEventListener("pointermove", (e) => {
    const g = box._geo;
    if (!g) return;
    const r = box.getBoundingClientRect();
    box._pointer = { x: e.clientX, y: e.clientY };
    box._hover = g.fromX(e.clientX - r.left);
    drawCrosshair(box);
  });
  box.addEventListener("pointerleave", () => { box._hover = null; box._pointer = null; drawCrosshair(box); });
  box.addEventListener("focus", () => { if (box._hover == null && box._geo) { box._hover = box._geo.defaultH; drawCrosshair(box); } });
  box.addEventListener("blur", () => { box._hover = null; box._pointer = null; drawCrosshair(box); });
  box.addEventListener("keydown", (e) => {
    const g = box._geo;
    if (!g) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      box._pointer = null;
      box._hover = g.clampH((box._hover == null ? g.defaultH : box._hover) + (e.key === "ArrowLeft" ? -1 : 1));
      drawCrosshair(box);
    } else if (e.key === "Escape") {
      box._hover = null;
      drawCrosshair(box);
    }
  });
}
function drawCrosshair(box) {
  const g = box._geo, xh = box.querySelector(".xh");
  if (!g || !xh) return;
  if (box._hover == null) { xh.innerHTML = ""; if (tipOwner === box) hideTip(); return; }
  box._hover = g.clampH(box._hover);
  const res = g.hover(box._hover);
  xh.innerHTML = `<line class="xh-line" x1="${f1(res.x)}" x2="${f1(res.x)}" y1="${g.top}" y2="${g.bottom}"/>` +
    res.dots.map(([y, cls]) => `<circle class="xh-dot ${cls}" cx="${f1(res.x)}" cy="${f1(y)}" r="4"/>`).join("");
  const r = box.getBoundingClientRect();
  const px = box._pointer ? box._pointer.x : r.left + res.x, py = box._pointer ? box._pointer.y : r.top + g.top + 12;
  showTip(res.title, res.rows, px, py, box);
}

/* Bar & column charts: each mark is its own hover/focus target */
function bindMarks(box) {
  const clear = () => {
    box.querySelectorAll(".bar-g.hov").forEach((n) => n.classList.remove("hov"));
    if (tipOwner === box) hideTip();
  };
  const show = (g, e) => {
    if (!box._tip) return;
    const t = box._tip(+g.dataset.i);
    box.querySelectorAll(".bar-g.hov").forEach((n) => n !== g && n.classList.remove("hov"));
    g.classList.add("hov");
    if (e) showTip(t.title, t.rows, e.clientX, e.clientY, box);
    else { const r = g.getBoundingClientRect(); showTip(t.title, t.rows, r.left + r.width / 2, r.top, box); }
  };
  box.addEventListener("pointermove", (e) => { const g = e.target.closest(".bar-g"); if (g) show(g, e); else clear(); });
  box.addEventListener("pointerleave", clear);
  box.addEventListener("focusin", (e) => { const g = e.target.closest(".bar-g"); if (g) show(g, null); });
  box.addEventListener("focusout", clear);
}

/* Daily price chart: close and EMA20, the forecast band from the last close the plan saw, stop/target, fills */
function drawPriceChart(box, d) {
  const bars = d.chart.bars;
  if (!bars.length) { box.innerHTML = '<p class="empty-state">No prices before this day.</p>'; box._geo = null; return; }
  const W = Math.max(280, Math.round(box.clientWidth)), H = W < 520 ? 240 : 300;
  const M = { t: 22, r: 66, b: 26, l: 4 }, pw = W - M.l - M.r, ph = H - M.t - M.b, n = bars.length;
  const fc = d.forecast, pts = fc.points || [];
  let refIdx = n - 1;
  for (let i = n - 1; i >= 0; i--) { if (bars[i].d < d.trade_date) { refIdx = i; break; } }
  const horizon = pts.length ? pts.length - 1 : 0, xMax = Math.max(n - 1, refIdx + horizon, 1);
  const lv = d.position ? { stop: d.position.stop, target: d.position.target } : d.plan ? { stop: d.plan.stop, target: d.plan.target } : null;
  let lo = Infinity, hi = -Infinity;
  for (const b of bars) { lo = Math.min(lo, b.c, b.ema20); hi = Math.max(hi, b.c, b.ema20); }
  for (const p of pts) { lo = Math.min(lo, p.lo80); hi = Math.max(hi, p.hi80); }
  if (lv) { lo = Math.min(lo, lv.stop); hi = Math.max(hi, lv.target); }
  const pad = (hi - lo) * 0.06 || 1;
  lo -= pad; hi += pad;
  const sx = (i) => M.l + (i / xMax) * pw;
  const sy = (v) => M.t + (1 - (v - lo) / (hi - lo)) * ph;
  const { ticks, step } = niceTicks(lo, hi, 5);
  const dec = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  const last = bars[n - 1], tagY = sy(last.c), out = [];

  if (pts.length) {
    const fx = sx(refIdx);
    out.push(`<rect class="future" x="${f1(fx)}" y="${M.t}" width="${f1(M.l + pw - fx)}" height="${ph}"/>`);
    out.push(`<text x="${f1(Math.min(fx + 6, M.l + pw - 60))}" y="${M.t - 8}">Next ${fc.horizon} days</text>`);
  }
  for (const v of ticks) {
    const y = sy(v);
    out.push(`<line class="grid" x1="${M.l}" x2="${M.l + pw}" y1="${f1(y)}" y2="${f1(y)}"/>`);
    if (Math.abs(y - tagY) > 13) out.push(`<text x="${M.l + pw + 8}" y="${f1(y + 4)}">${v.toLocaleString("en-IN", { minimumFractionDigits: dec, maximumFractionDigits: dec })}</text>`);
  }
  out.push(`<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${M.t + ph}" y2="${M.t + ph}"/>`);
  let lastLabelX = -Infinity;
  for (let i = 1; i < n; i++) {
    if (bars[i].d.slice(0, 7) === bars[i - 1].d.slice(0, 7)) continue;
    const x = sx(i);
    if (x - lastLabelX < 44 || x > M.l + pw - 16) continue;
    lastLabelX = x;
    out.push(`<line class="axis" x1="${f1(x)}" x2="${f1(x)}" y1="${M.t + ph}" y2="${M.t + ph + 4}"/>`, `<text x="${f1(x)}" y="${H - 6}" text-anchor="middle">${esc(monthFmt.format(isoDate(bars[i].d)))}</text>`);
  }
  if (pts.length) {
    const band = (a, b) => "M" + pts.map((p) => `${f1(sx(refIdx + p.t))} ${f1(sy(p[b]))}`).join("L") + "L" +
      pts.slice().reverse().map((p) => `${f1(sx(refIdx + p.t))} ${f1(sy(p[a]))}`).join("L") + "Z";
    out.push(`<path class="band80" d="${band("lo80", "hi80")}"/>`, `<path class="band50" d="${band("lo50", "hi50")}"/>`);
    out.push(`<path class="fc-mid" d="M${pts.map((p) => `${f1(sx(refIdx + p.t))} ${f1(sy(p.mid))}`).join("L")}"/>`);
  }
  if (lv) {
    const x0 = sx(Math.max(0, refIdx - 15));
    for (const [k, v] of [["stop", lv.stop], ["target", lv.target]]) {
      const y = sy(v);
      out.push(`<line class="lvl lvl-${k}" x1="${f1(x0)}" x2="${M.l + pw}" y1="${f1(y)}" y2="${f1(y)}"/>`,
        `<text class="lvl-text ${k}" x="${f1(x0)}" y="${f1(k === "stop" ? y + 13 : y - 5)}">${k === "stop" ? "Stop" : "Target"} ${nf2.format(v)}</text>`);
    }
  }
  out.push(`<path class="ln ln-2" d="M${bars.map((b, i) => `${f1(sx(i))} ${f1(sy(b.ema20))}`).join("L")}"/>`);
  out.push(`<path class="ln ln-1" d="M${bars.map((b, i) => `${f1(sx(i))} ${f1(sy(b.c))}`).join("L")}"/>`);
  const idx = new Map(bars.map((b, i) => [b.d, i]));
  for (const m of d.chart.markers || []) {
    const i = idx.get(m.trade_date);
    if (i == null || m.price == null) continue;
    const x = sx(i), y = sy(m.price);
    out.push(m.side === "buy"
      ? `<path class="mk-buy" d="M${f1(x)} ${f1(y + 4)}l5 9h-10z"><title>Bought ${m.quantity} at ${nf2.format(m.price)}</title></path>`
      : `<path class="mk-sell" d="M${f1(x)} ${f1(y - 4)}l5 -9h-10z"><title>Sold ${m.quantity} at ${nf2.format(m.price)}</title></path>`);
  }
  out.push(`<circle class="end-dot" cx="${f1(sx(n - 1))}" cy="${f1(tagY)}" r="4"/>`);
  out.push(`<rect class="tag" x="${M.l + pw + 2}" y="${f1(tagY - 10)}" width="${M.r - 4}" height="20" rx="4"/>`,
    `<text class="tag-text" x="${M.l + pw + 2 + (M.r - 4) / 2}" y="${f1(tagY + 4)}" text-anchor="middle">${nf2.format(last.c)}</text>`);
  out.push(`<g class="xh"></g><rect class="hit" x="${M.l}" y="${M.t}" width="${pw}" height="${ph}"/>`);
  box.innerHTML = svgWrap(W, H, `${d.symbol} daily closes with EMA20 and a ${fc.horizon}-day forecast range`, out.join(""));

  box._geo = {
    top: M.t, bottom: M.t + ph, defaultH: n - 1,
    fromX: (px) => clamp(Math.round(((px - M.l) / pw) * xMax), 0, xMax),
    clampH: (i) => clamp(i, 0, xMax),
    hover: (i) => {
      if (i <= n - 1) {
        const b = bars[i];
        return {
          x: sx(i), dots: [[sy(b.c), "dot-1"], [sy(b.ema20), "dot-2"]], title: dayLong(b.d),
          rows: [
            { key: "var(--series-1)", value: money(b.c), label: "Close" },
            { key: "var(--series-2)", value: money(b.ema20), label: "EMA20" },
            { value: money(b.ema50), label: "EMA50" },
            { value: `${nf2.format(b.o)} / ${nf2.format(b.h)} / ${nf2.format(b.l)}`, label: "Open / high / low" },
            { value: b.v == null ? "—" : nf0.format(b.v), label: "Volume" },
          ],
        };
      }
      const t = clamp(i - refIdx, 0, horizon), p = pts[t];
      return {
        x: sx(refIdx + t), dots: [[sy(p.mid), "dot-3"]], title: `${dayLong(fc.days[t - 1])} · forecast day ${t}`,
        rows: [
          { key: "var(--series-3)", value: money(p.mid), label: "Midpoint" },
          { value: `${money(p.lo50)} – ${money(p.hi50)}`, label: "50% range" },
          { value: `${money(p.lo80)} – ${money(p.hi80)}`, label: "80% range" },
        ],
      };
    },
  };
  drawCrosshair(box);
}

function drawEquity(box, curve) {
  const W = Math.max(280, Math.round(box.clientWidth)), H = 250;
  const M = { t: 22, r: 18, b: 26, l: 58 }, pw = W - M.l - M.r, ph = H - M.t - M.b, n = curve.length;
  let lo = 0, hi = 0;
  for (const p of curve) { lo = Math.min(lo, p.cum); hi = Math.max(hi, p.cum); }
  const span = hi - lo || 1;
  const d = niceDomain(lo < 0 ? lo - span * 0.06 : 0, hi > 0 ? hi + span * 0.1 : 0, 4);
  const sx = (i) => M.l + (i / Math.max(1, n - 1)) * pw;
  const sy = (v) => M.t + (1 - (v - d.lo) / (d.hi - d.lo)) * ph;
  const out = [];
  for (const v of d.ticks) {
    const y = sy(v);
    out.push(`<line class="${Math.abs(v) < 1e-9 ? "axis" : "grid"}" x1="${M.l}" x2="${M.l + pw}" y1="${f1(y)}" y2="${f1(y)}"/>`,
      `<text x="${M.l - 8}" y="${f1(y + 4)}" text-anchor="end">${esc(compactINR(v))}</text>`);
  }
  const labels = Math.min(5, n - 1), used = new Set();
  for (let j = 0; j <= labels; j++) {
    const i = Math.max(1, Math.round((j / labels) * (n - 1)));
    if (used.has(i) || !curve[i]) continue;
    used.add(i);
    const anchor = j === 0 ? "start" : j === labels ? "end" : "middle";
    out.push(`<text x="${f1(sx(i))}" y="${H - 6}" text-anchor="${anchor}">${esc(dateLabel(curve[i].t.exit_date))}</text>`);
  }
  const line = curve.map((p, i) => `${f1(sx(i))} ${f1(sy(p.cum))}`).join("L");
  out.push(`<path class="area-1" d="M${f1(sx(0))} ${f1(sy(0))}L${line}L${f1(sx(n - 1))} ${f1(sy(0))}Z"/>`);
  out.push(`<path class="ln ln-1" d="M${line}"/>`);
  const last = curve[n - 1];
  out.push(`<circle class="end-dot" cx="${f1(sx(n - 1))}" cy="${f1(sy(last.cum))}" r="4"/>`);
  out.push(`<text class="lbl-strong" x="${f1(sx(n - 1) - 8)}" y="${f1(sy(last.cum) + (last.cum >= 0 ? -10 : 18))}" text-anchor="end">${esc(signedMoney0(last.cum))}</text>`);
  out.push(`<g class="xh"></g><rect class="hit" x="${M.l}" y="${M.t}" width="${pw}" height="${ph}"/>`);
  box.innerHTML = svgWrap(W, H, `Equity curve ending at ${signedMoney0(last.cum)}`, out.join(""));
  box._geo = {
    top: M.t, bottom: M.t + ph, defaultH: n - 1,
    fromX: (px) => clamp(Math.round(((px - M.l) / pw) * (n - 1)), 0, n - 1),
    clampH: (i) => clamp(i, 0, n - 1),
    hover: (i) => {
      const p = curve[i];
      if (p.start) return { x: sx(i), dots: [[sy(0), "dot-1"]], title: "Start of period", rows: [{ key: "var(--series-1)", value: "₹0", label: "Cumulative P&L" }] };
      return {
        x: sx(i), dots: [[sy(p.cum), "dot-1"]], title: `${dateLabel(p.t.exit_date)} · ${p.t.symbol}`,
        rows: [{ key: "var(--series-1)", value: signedMoney0(p.cum), label: "Cumulative P&L" }, { value: signedMoney(p.t.net_pnl), label: "This trade" }],
      };
    },
  };
  drawCrosshair(box);
}

function drawHBars(box, items, o) {
  const W = Math.max(280, Math.round(box.clientWidth)), rowH = 34, barH = 14, top = 6;
  const H = top * 2 + items.length * rowH + (o.ref != null ? 16 : 0);
  const vals = items.map((it) => it.value);
  const dLo = o.domain ? o.domain[0] : Math.min(0, ...vals), dHi = o.domain ? o.domain[1] : Math.max(0, ...vals);
  const x0 = o.labelW + (dLo < 0 ? o.valueW : 0), x1 = W - (dHi > 0 ? o.valueW : 8);
  const sx = (v) => x0 + ((v - dLo) / (dHi - dLo || 1)) * (x1 - x0);
  const zx = sx(clamp(0, dLo, dHi)), bottom = top + items.length * rowH, out = [];
  if (o.ref != null) {
    const rx = sx(o.ref);
    out.push(`<line class="ref" x1="${f1(rx)}" x2="${f1(rx)}" y1="${top}" y2="${bottom}"/>`, `<text x="${f1(rx)}" y="${H - 3}" text-anchor="middle">${esc(o.refLabel)}</text>`);
  }
  out.push(`<line class="axis" x1="${f1(zx)}" x2="${f1(zx)}" y1="${top}" y2="${bottom}"/>`);
  items.forEach((it, i) => {
    const yRow = top + i * rowH, y = yRow + (rowH - barH) / 2, v = it.value, pos = v >= 0;
    const bx = pos ? zx : sx(v), bw = Math.abs(sx(v) - zx);
    out.push(`<g class="bar-g" tabindex="0" data-i="${i}" role="img" aria-label="${esc(`${it.label}: ${o.fmt(it)}`)}">`,
      `<rect class="hit" x="0" y="${yRow}" width="${W}" height="${rowH}"/>`,
      `<text class="${o.labelClass}" x="0" y="${f1(yRow + rowH / 2 + 4)}">${esc(it.label)}</text>`,
      bw > 0.5 ? `<path class="bar" d="${barPath(bx, y, bw, barH, pos ? "right" : "left")}" style="fill:${o.color(it)}"/>` : "",
      `<text class="lbl-strong" x="${f1(pos ? bx + bw + 6 : bx - 6)}" y="${f1(y + barH / 2 + 4)}" text-anchor="${pos ? "start" : "end"}">${esc(o.fmt(it))}</text>`,
      `</g>`);
  });
  box.innerHTML = svgWrap(W, H, o.title, out.join(""));
}

function drawColumns(box, items, o) {
  const W = Math.max(280, Math.round(box.clientWidth)), H = 230;
  const M = { t: 22, r: o.edges ? 24 : 8, b: 28, l: 52 }, pw = W - M.l - M.r, ph = H - M.t - M.b;
  const vals = items.map((it) => it.value);
  const lo0 = Math.min(0, ...vals), hi0 = Math.max(0, ...vals), span = hi0 - lo0 || 1;
  const d = niceDomain(lo0 < 0 ? lo0 - span * 0.14 : 0, hi0 > 0 ? hi0 + span * 0.12 : 0, 4);
  const sy = (v) => M.t + (1 - (v - d.lo) / (d.hi - d.lo)) * ph;
  const bandW = pw / items.length, cw = Math.min(24, o.touch ? bandW - 2 : bandW * 0.6), out = [];
  for (const v of d.ticks) {
    if (o.integer && !Number.isInteger(v)) continue;
    const y = sy(v);
    out.push(`<line class="${Math.abs(v) < 1e-9 ? "axis" : "grid"}" x1="${M.l}" x2="${M.l + pw}" y1="${f1(y)}" y2="${f1(y)}"/>`,
      `<text x="${M.l - 8}" y="${f1(y + 4)}" text-anchor="end">${esc(o.tick(v))}</text>`);
  }
  if (o.edges) {
    const every = items.length > 8 ? Math.ceil(items.length / 8) : 1;
    o.edges.forEach((label, k) => {
      if (k % every) return;
      const x = M.l + bandW * k;
      out.push(`<text x="${f1(x)}" y="${H - 8}" text-anchor="middle">${esc(label)}</text>`);
    });
  }
  items.forEach((it, i) => {
    const cx = M.l + bandW * (i + 0.5), v = it.value, pos = v >= 0, y0 = sy(0), y1 = sy(v), h = Math.abs(y1 - y0);
    const cap = o.cap ? o.cap(it) : null, xl = o.xLabel ? o.xLabel(it) : null;
    out.push(`<g class="bar-g" tabindex="0" data-i="${i}" role="img" aria-label="${esc(o.aria(it))}">`,
      `<rect class="hit" x="${f1(M.l + bandW * i)}" y="${M.t}" width="${f1(bandW)}" height="${ph}"/>`,
      h > 0.5 ? `<path class="bar" d="${barPath(cx - cw / 2, pos ? y1 : y0, cw, h, pos ? "top" : "bottom")}" style="fill:${o.color(it)}"/>` : "",
      cap ? `<text class="lbl-strong" x="${f1(cx)}" y="${f1(pos ? y1 - 6 : y1 + 14)}" text-anchor="middle">${esc(cap)}</text>` : "",
      xl ? `<text x="${f1(cx)}" y="${H - 8}" text-anchor="middle">${esc(xl)}</text>` : "",
      `</g>`);
  });
  box.innerHTML = svgWrap(W, H, o.title, out.join(""));
}

function fillTable(container, headers, rows, numeric) {
  const wrap = el("div", "table-wrap"), t = el("table", "table table-compact"), thead = el("thead"), trh = el("tr"), tb = el("tbody");
  headers.forEach((h, i) => { const th = el("th", numeric.includes(i) ? "r" : null, h); th.scope = "col"; trh.append(th); });
  thead.append(trh);
  for (const r of rows) {
    const tr = el("tr");
    r.forEach((c, i) => tr.append(el("td", numeric.includes(i) ? "r num" : null, c)));
    tb.append(tr);
  }
  t.append(thead, tb);
  wrap.append(t);
  container.replaceChildren(wrap);
}

/* ================= Shared bits ================= */
function setKpi(id, value, tone, sub) {
  const v = $(id);
  v.textContent = value;
  v.className = "kpi-value" + (tone && tone !== "flat" ? " " + tone : "");
  $(id + "-sub").textContent = sub;
}
function setupChip(type) {
  return el("span", type && type !== "none" ? "chip chip-preview" : "chip chip-muted", SETUP_LABEL[type] || type || "No setup");
}
function planChip(plan, removed) {
  if (removed) return el("span", "chip chip-outline", "Removed");
  if (!plan) return el("span", "chip chip-muted", "Not analysed");
  if (plan.accepted) return el("span", "chip chip-signal", `Long · ${Math.round(plan.confidence * 100)}%`);
  if (plan.verdict === "accept") return el("span", "chip chip-muted", "No trade");
  if (plan.verdict === "reject") return el("span", "chip chip-down", "Rejected");
  return el("span", "chip chip-warn", "Not settled");
}
function sourceChip(source) {
  return el("span", source === "agents" ? "chip chip-preview" : "chip chip-outline", SOURCE_LABEL[source] || source);
}
function iconButton(label, icon, onClick) {
  const b = el("button", "icon-btn");
  b.type = "button";
  b.setAttribute("aria-label", label);
  b.title = label;
  b.innerHTML = `<svg width="14" height="14" aria-hidden="true"><use href="#i-${icon}"/></svg>`;
  b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
  return b;
}
function button(label, cls, onClick) {
  const b = el("button", "btn " + cls, label);
  b.type = "button";
  b.addEventListener("click", onClick);
  return b;
}
const morning = () => state.desk && state.desk.phase === "morning";
const busy = () => state.jobId != null;
const providerLabel = (d) => (d.provider === "offline" ? "Rule-based agents" : `LLM agents · ${d.model || d.provider}`);

/* ================= Desks and loading ================= */
async function loadDesks() {
  state.desks = await GET("/api/desks");
  for (const id of ["desk-select", "desk-select-m"]) {
    const sel = $(id);
    sel.replaceChildren(...state.desks.map((d) => new Option(d.name, d.id)));
    sel.value = state.deskId == null ? "" : String(state.deskId);
    sel.hidden = state.desks.length === 0;
  }
}
async function selectDesk(id) {
  state.deskId = id;
  store.set("twa-desk", id);
  Object.assign(state, { selected: null, detail: null, trace: null, feed: [], lastEventId: 0, watch: null, book: null, trades: [], days: [], confirm: null });
  state.hist.symbol = "all";
  state.ana.symbol = "all";
  for (const s of ["desk-select", "desk-select-m"]) $(s).value = id == null ? "" : String(id);
  await refreshAll();
}
async function refreshAll() {
  const token = ++state.loadToken;
  if (state.deskId == null) { renderNoDesk(); return; }
  let desk;
  try { desk = await GET(deskPath()); } catch (e) {
    if (e.status === 404) { state.deskId = null; await loadDesks(); renderNoDesk(); return; }
    throw e;
  }
  if (token !== state.loadToken) return;
  state.desk = desk;
  renderShell();
  watchJob(desk.job);
  await loadView(token);
}
async function loadView(token = state.loadToken) {
  if (!state.desk) return;
  const v = state.view;
  if (v === "dashboard") {
    const [watch] = await Promise.all([GET(deskPath("/watchlist")), loadFeed()]);
    if (token !== state.loadToken) return;
    state.watch = watch;
    await loadDetail();
    if (token !== state.loadToken) return;
    renderDashboard();
  } else if (v === "tradebook") {
    const b = await GET(deskPath("/book"));
    if (token !== state.loadToken) return;
    state.book = b;
    renderTradebook();
  } else {
    const [t, days] = await Promise.all([GET(deskPath("/trades")), GET(deskPath("/days"))]);
    if (token !== state.loadToken) return;
    state.trades = t;
    state.days = days;
    fillSymbolSelect("h-symbol", state.hist.symbol);
    fillSymbolSelect("a-symbol", state.ana.symbol);
    if (v === "history") renderHistory(); else renderAnalysis();
  }
}
async function loadFeed() {
  const items = await GET(deskPath("/events?limit=150"));
  const known = state.lastEventId;
  state.feed = items.map((e) => ({ ...e, isNew: known > 0 && e.id > known }));
  if (items.length) state.lastEventId = items[0].id;
}
async function loadDetail() {
  const rows = state.watch ? state.watch.rows : [];
  if (!state.selected || !rows.some((r) => r.symbol === state.selected)) state.selected = rows.length ? rows[0].symbol : null;
  state.trace = null;
  $("trace-details").open = false;
  if (!state.selected) { state.detail = null; return; }
  state.detail = await GET(deskPath(`/stocks/${encodeURIComponent(state.selected)}`));
}

/* ================= Shell: rail, header, job banner ================= */
function renderNoDesk() {
  state.desk = null;
  $("no-desk").hidden = false;
  $("session-card").hidden = true;
  $("delete-desk").hidden = true;
  for (const k in VIEWS) $("view-" + k).hidden = true;
  $("page-title").textContent = "Welcome";
  $("page-sub").textContent = "Create a desk to start replaying history with the agents.";
  $("head-clock").textContent = "No desk";
  $("provider-pill").textContent = "—";
  $("desk-avatar").textContent = "—";
  $("desk-name").textContent = "No desk";
  $("desk-provider").textContent = "";
  $("run-day-m").disabled = true;
}
function renderShell() {
  const d = state.desk;
  $("no-desk").hidden = true;
  $("session-card").hidden = false;
  $("delete-desk").hidden = false;
  for (const k in VIEWS) $("view-" + k).hidden = k !== state.view;
  const [label, cls] = PHASE[d.phase] || [d.phase, ""];
  const st = $("rail-state");
  st.textContent = label;
  st.className = "session-state" + (cls ? " " + cls : "");
  $("rail-day").textContent = dayLong(d.current_date);
  const prog = d.days_total ? (d.days_done / d.days_total) * 100 : 0;
  $("rail-meter-fill").style.width = prog.toFixed(1) + "%";
  $("rail-meter").setAttribute("aria-valuenow", String(Math.round(prog)));
  $("rail-start").textContent = dateYear(d.start_date);
  $("rail-end").textContent = dateYear(d.end_date);
  $("rail-note").textContent = {
    morning: `Day ${d.days_done + 1} of ${d.days_total}. Plans made at 08:30 IST with data to the close of ${dateLabel(d.visible_through)}. Run the day to trade them.`,
    closed: "Session closed. The next morning's plans are pending.",
    new: "The agents are making the first plans.",
    finished: `Replay finished: ${plural(d.days_done, "trading day")}.`,
  }[d.phase] || "";
  $("desk-avatar").textContent = initials(d.name);
  $("desk-name").textContent = d.name;
  $("desk-provider").textContent = providerLabel(d);
  const pill = $("head-pill");
  pill.classList.toggle("is-closed", d.phase === "closed" || d.phase === "finished");
  pill.classList.toggle("is-planning", d.phase === "new");
  $("head-clock").textContent = {
    morning: `Pre-open · ${dateLabel(d.current_date)}, 08:30 IST`, closed: `Closed · ${dateLabel(d.current_date)}`,
    new: "Making plans", finished: "Replay finished",
  }[d.phase] || d.phase;
  $("provider-pill").textContent = providerLabel(d);
  const canRun = !busy() && d.phase !== "finished";
  for (const id of ["run-day", "run-many", "run-day-m"]) $(id).disabled = !canRun;
  $("run-n").disabled = !canRun;
  $("run-day").lastChild.textContent = d.phase === "new" ? "Make the first plans" : "Run the day";
  $("page-title").textContent = VIEWS[state.view][0];
  $("page-sub").textContent = VIEWS[state.view][1]();
}

function watchJob(job) {
  if (job && ACTIVE.includes(job.status)) {
    if (state.jobId !== job.id) { state.jobId = job.id; state.jobDone = job.done; }
    showJob(job);
    renderShell();
    schedulePoll(800);
  } else if (state.desk && state.desk.phase === "new") {
    if (job && job.status === "failed") toast(`The first plans failed: ${job.error}. Press "Make the first plans" to try again.`);
    else { clearTimeout(state.pollTimer); state.pollTimer = setTimeout(() => refreshAll().catch(() => {}), 1500); }
  }
}
function schedulePoll(ms = 1000) { clearTimeout(state.pollTimer); state.pollTimer = setTimeout(pollJob, ms); }
async function pollJob() {
  if (state.jobId == null) return;
  let job;
  try { job = await GET(`/api/jobs/${state.jobId}`); } catch (e) { schedulePoll(2500); return; }
  showJob(job);
  if (!ACTIVE.includes(job.status)) {
    state.jobId = null;
    $("job-banner").hidden = true;
    toast(job.status === "failed" ? `Failed: ${job.error}` : job.message);
    if (job.kind === "analyse" && job.params && job.params.symbol && job.status === "done") state.selected = job.params.symbol;
    await loadDesks();
    await refreshAll();
    return;
  }
  if (job.done !== state.jobDone) {               // a day finished: show it while the rest runs
    state.jobDone = job.done;
    try {
      state.desk = await GET(deskPath());
      renderShell();
      if (state.view === "dashboard") { await loadFeed(); renderDashKpis(); renderFeed(); }
    } catch (e) { /* the next poll retries */ }
  }
  schedulePoll();
}
function showJob(job) {
  $("job-banner").hidden = false;
  const p = job.params || {};
  $("job-title").textContent = job.kind === "create" ? "Making the first plans"
    : job.kind === "analyse" ? `The agents are analysing ${p.symbol}`
    : job.status === "cancelling" ? "Stopping after the current step" : `Running ${plural(job.total, "trading day")}`;
  $("job-message").textContent = job.kind === "advance" ? `${job.done} of ${job.total} done · ${job.message}` : job.message;
  const frac = job.kind === "advance" && job.total ? job.done / job.total : null;
  $("job-meter-fill").style.width = frac == null ? "35%" : Math.max(4, frac * 100).toFixed(1) + "%";
  const c = $("job-cancel");
  c.hidden = job.kind !== "advance";
  c.disabled = job.status === "cancelling";
}
async function runDays(n) {
  try {
    const { job_id } = await POST(deskPath("/advance"), { days: n });
    state.jobId = job_id;
    state.jobDone = 0;
    showJob({ id: job_id, kind: "advance", total: n, done: 0, status: "queued", message: "Waiting to start", params: { days: n } });
    renderShell();
    schedulePoll(500);
  } catch (e) { toast(e.message); }
}

/* ================= Dashboard ================= */
function renderDashboard() { renderDashKpis(); renderWatchlist(); renderFeed(); renderDetail(); renderPipeline(); }

function renderDashKpis() {
  const d = state.desk;
  setKpi("k-day", d.last_session ? signedMoney(d.day_pnl) : "—", toneOf(d.day_pnl),
    d.last_session ? `Session of ${dateLabel(d.last_session)} · equity ${compactINR(d.equity)} (${signedPct(d.total_return * 100)} since the start)` : "No session run yet");
  setKpi("k-pos", String(d.open_positions), "", d.open_positions ? `${compactINR(d.exposure)} at the last close` : "No open positions");
  const s = d.signals, n = s.accepted + s.manual_buys;
  setKpi("k-sig", d.phase === "morning" ? String(n) : "—", "",
    d.phase === "morning" ? `${s.accepted} from the agents, ${s.manual_buys} yours · ${s.analysed} analysed` : "Plans appear each morning");
  const w = d.win_rate_month;
  setKpi("k-win", w.trades ? pct(w.win_rate) : "—", "", `${plural(w.trades, "closed trade")} · profit factor ${w.profit_factor == null ? "—" : w.profit_factor.toFixed(2)}`);
}

function renderWatchlist() {
  const body = $("watch-body"), w = state.watch, rows = w ? w.rows : [];
  body.replaceChildren();
  const live = rows.filter((r) => !r.removed).length;
  $("watch-count").textContent = w && w.visible_through
    ? `${plural(live, "stock")} · picked at 08:30 on ${dateLabel(w.trade_date)}, with data to the close of ${dateLabel(w.visible_through)}` : "";
  const open = morning() && !busy();
  $("add-ticker").disabled = !open;
  $("add-btn").disabled = !open;
  if (!rows.length) {
    const tr = el("tr"), td = el("td", "empty-cell", state.desk.phase === "new" ? "The agents are making the first plans…"
      : "The Planner picked no stocks for today. Add one above to have the agents analyse it.");
    td.colSpan = 9;
    tr.append(td);
    body.append(tr);
    return;
  }
  for (const r of rows) {
    const tr = el("tr", "row-click" + (r.symbol === state.selected ? " is-selected" : ""));
    tr.tabIndex = 0;
    tr.dataset.symbol = r.symbol;
    const c0 = el("td");
    c0.append(el("span", "sym mono", r.symbol), el("span", "sym-sub", r.origin === "manual" ? `${r.name} · yours` : r.name));
    const spark = el("td", "spark-cell");
    spark.innerHTML = sparkline(r.spark, 72, 28);
    const plan = el("td");
    plan.append(planChip(r.plan, r.removed));
    if (r.held) plan.append(document.createTextNode(" "), el("span", "chip chip-outline", "Held"));
    if (r.pending.includes("buy")) plan.append(document.createTextNode(" "), el("span", "chip chip-warn", "Your buy"));
    const act = el("td", "r");
    if (open && !r.removed) act.append(iconButton(`Remove ${r.symbol} from today's list`, "cross", () => removeStock(r.symbol)));
    if (open && r.removed) act.append(iconButton(`Put ${r.symbol} back on today's list`, "plus", () => restoreStock(r.symbol)));
    tr.append(c0, plan, el("td", "r num strong", r.last_close == null ? "—" : nf2.format(r.last_close)),
      el("td", "r num " + toneOf(r.change_pct), r.change_pct == null ? "—" : signedPct(r.change_pct * 100)), spark,
      el("td", "r num", r.rsi14 == null ? "—" : nf1.format(r.rsi14)),
      el("td", "r num " + toneOf(r.ema20_gap), r.ema20_gap == null ? "—" : signedPct(r.ema20_gap * 100)),
      (() => { const td = el("td"); td.append(setupChip(r.setup_type)); return td; })(), act);
    tr.addEventListener("click", () => selectStock(r.symbol));
    tr.addEventListener("keydown", (e) => {
      if (e.target === tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectStock(r.symbol); }
    });
    body.append(tr);
  }
}
function markSelected() {
  document.querySelectorAll("#watch-body tr[data-symbol]").forEach((tr) => tr.classList.toggle("is-selected", tr.dataset.symbol === state.selected));
}
async function selectStock(sym) {
  state.selected = sym;
  markSelected();
  try { await loadDetail(); renderDetail(); } catch (e) { toast(e.message); }
}

const FEED_TYPES = { plan: ["Plan", "chip-signal"], agent: ["Agent", "chip-preview"], trade: ["Trade", "chip-warn"], system: ["Desk", "chip-muted"] };
function renderFeed() {
  const list = $("feed-list");
  list.replaceChildren();
  const items = state.feed.filter((e) => state.feedFilter === "all" || e.kind === state.feedFilter);
  if (!items.length) list.append(el("li", "feed-empty", state.feedFilter === "all" ? "Events appear here as the desk plans and trades." : "Nothing of this type yet."));
  for (const e of items.slice(0, 80)) {
    const li = el("li", "feed-item" + (e.isNew ? " is-new" : "")), body = el("div", "feed-body"), top = el("div", "feed-top");
    const [label, cls] = FEED_TYPES[e.kind] || [e.kind, "chip-muted"];
    top.append(el("span", "sym mono", e.symbol || "Desk"), el("span", "chip " + cls, label));
    body.append(top, el("p", null, e.text));
    const time = el("span", "feed-time mono");
    time.append(el("span", "feed-day", dateLabel(e.trade_date)), el("span", null, e.at === "intraday" ? "day" : e.at));
    li.append(time, body);
    list.append(li);
    e.isNew = false;
  }
}

function valueText(v) {
  if (v.value == null) return "—";
  if (v.kind === "rsi") return nf1.format(v.value);
  if (v.kind === "num") return v.value.toFixed(2);
  if (["atr14_pct", "rvol_20", "vix"].includes(v.key)) return (v.value * 100).toFixed(2) + "%";
  return signedPct(v.value * 100);
}

function renderDetail() {
  const d = state.detail;
  $("detail-empty").hidden = !!d;
  $("detail-main").hidden = !d;
  if (!d) { hideTip(); return; }
  $("d-sym").textContent = d.symbol;
  $("d-name").textContent = `NSE · ${d.name}`;
  $("d-ltp").textContent = money(d.price.last_close);
  const c = $("d-chg");
  c.textContent = d.price.change == null ? "" : `${signedMoney(d.price.change)} (${signedPct(d.price.change_pct * 100)}) on ${dateLabel(d.visible_through)}`;
  c.className = "detail-chg num " + toneOf(d.price.change);
  const removed = !!(d.watch && d.watch.removed);
  $("d-setup").replaceChildren(setupChip(d.setup_type), document.createTextNode(" "), planChip(d.plan, removed));
  $("band-legend").textContent = d.forecast.vol_model === "lstm" ? "Forecast range (LSTM volatility), 50% / 80%" : "Forecast range (20-day volatility), 50% / 80%";
  drawPriceChart($("price-chart"), d);
  if ($("price-details").open) renderPriceTable(d);

  $("d-values-sub").textContent = `At the close before ${dateLabel(d.trade_date)}: the record the agents could cite as ctx:${d.symbol}`;
  const dl = $("d-values");
  dl.replaceChildren();
  for (const v of d.values) {
    const row = el("div"), dt = el("dt", null, v.label);
    dt.append(el("span", "formula", v.formula));
    row.append(dt, el("dd", null, valueText(v)));
    dl.append(row);
  }

  const fired = d.setup_type, show = fired !== "none" ? fired : "pullback_uptrend", checks = d.checks[show] || [];
  $("d-checks-sub").textContent = fired !== "none"
    ? `${SETUP_LABEL[fired]} setup: every rule passes on the last close. Entry, stop and target come from code, not the agents.`
    : `No setup fired today. The pullback rules, for reference: ${checks.filter((x) => x.pass).length} of ${checks.length} pass.`;
  const ul = $("d-checks");
  ul.replaceChildren();
  for (const ch of checks) {
    const li = el("li", "check " + (ch.pass ? "pass" : "fail")), ic = el("span", "check-icon"), label = el("span", null, ch.label);
    ic.innerHTML = `<svg width="12" height="12" aria-hidden="true"><use href="#i-${ch.pass ? "check" : "cross"}"/></svg>`;
    label.append(el("span", "sr-only", ch.pass ? " (passes)" : " (fails)"));
    li.append(ic, label, el("span", "check-detail", ch.detail));
    ul.append(li);
  }
  renderForecast(d);
  renderReport(d);
  renderActions(d);
  $("trace-summary").textContent = `Show the agent trace (${plural(d.trace_steps, "step")})`;
}

function renderPriceTable(d) {
  const bars = d.chart.bars, rows = [];
  for (let i = bars.length - 1; i >= Math.max(0, bars.length - 15); i--) {
    const b = bars[i];
    rows.push([dateYear(b.d), nf2.format(b.o), nf2.format(b.h), nf2.format(b.l), nf2.format(b.c), b.v == null ? "—" : nf0.format(b.v), nf2.format(b.ema20), nf2.format(b.ema50)]);
  }
  fillTable($("price-table"), ["Date", "Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50"], rows, [1, 2, 3, 4, 5, 6, 7]);
}

function renderForecast(d) {
  const fc = d.forecast, box = $("d-forecast");
  box.replaceChildren();
  box.hidden = !fc.points.length;
  if (!fc.points.length) return;
  const end = fc.points[fc.points.length - 1], head = el("div", "forecast-head");
  head.append(el("span", null, `Next ${plural(fc.horizon, "trading day")} · 80% range`),
    el("span", "chip chip-preview", fc.vol_model === "lstm" ? "LSTM volatility" : "20-day volatility"));
  const base = fc.base, p5 = base && base.p_up ? (base.p_up["5"] ?? base.p_up[5]) : null;
  box.append(head, el("div", "forecast-range num", `${money(end.lo80)} – ${money(end.hi80)}`),
    el("div", "forecast-meta", `Midpoint ${money(end.mid)}` + (p5 != null ? ` · ${base.model.toUpperCase()} chance of a rise over 5 days: ${pct(p5)}` : "")),
    el("p", "forecast-note", "The direction models showed no edge in validation, so read the midpoint lightly. The width of the range comes from the volatility forecast, which held up."));
}

function caseBlock(title, c) {
  const wrap = el("div", "case"), h = el("h4", null, title);
  h.append(el("span", "chip chip-muted", `confidence ${pct(c.confidence)}`));
  wrap.append(h, el("p", null, c.thesis));
  if (c.claims && c.claims.length) {
    const ul = el("ul", "claims");
    for (const cl of c.claims) {
      const li = el("li");
      li.append(el("span", null, cl.text));
      if (cl.evidence_ids && cl.evidence_ids.length) {
        const ids = el("span", "ev-ids");
        for (const id of cl.evidence_ids) ids.append(el("span", "ev-id", id));
        li.append(ids);
      }
      ul.append(li);
    }
    wrap.append(ul);
  }
  if (c.rebuttal) wrap.append(el("p", null, `Rebuttal: ${c.rebuttal}`));
  return wrap;
}

function renderReport(d) {
  const box = $("d-report"), p = d.plan;
  box.replaceChildren();
  $("d-report-chip").textContent = !p ? "Waiting" : p.accepted ? "Long" : p.verdict === "accept" ? "No trade" : p.verdict === "reject" ? "Rejected" : "Not settled";
  if (!p) {
    box.append(el("p", null, d.is_today && d.watch ? "The agents are analysing this stock." : "The agents didn't analyse this stock that day."));
    return;
  }
  const plan = el("div", "plan-box"), top = el("div", "plan-top");
  top.append(planChip(p, d.watch && d.watch.removed), el("span", "chip chip-muted", `hold up to ${plural(p.max_hold_days, "day")}`));
  if (p.rounds > 1) top.append(el("span", "chip chip-warn", `revised ${p.rounds - 1}×`));
  const levels = el("div", "levels"), rr = d.setup ? d.setup.risk_reward : null;
  for (const [k, v] of [["Reference close", money(p.entry)], ["Stop", money(p.stop)], ["Target", money(p.target)], ["Reward : risk", rr == null ? "—" : `${rr.toFixed(1)} : 1`]]) {
    const cell = el("div");
    cell.append(el("span", null, k), el("strong", null, v));
    levels.append(cell);
  }
  plan.append(top, levels, el("p", null, p.rationale));
  box.append(plan);
  if (d.position) {
    const pos = d.position;
    box.append(el("p", "notice", `Held: ${nf0.format(pos.quantity)} shares bought ${dateLabel(pos.entry_date)} at ${money(pos.entry_price)}, day ${pos.days} of ${pos.max_hold_days}. Stop ${money(pos.stop)}, target ${money(pos.target)}.`));
  }
  if (d.exit_decision && d.exit_decision.action !== "hold") {
    box.append(el("p", "notice", `Exit Monitor: ${d.exit_decision.action === "exit" ? "exit at the open" : `stop raised to ${money(d.exit_decision.new_stop)}`}. ${d.exit_decision.reason}`));
  }
  if (p.bull) box.append(caseBlock("Bull case", p.bull));
  if (p.bear) box.append(caseBlock("Bear case", p.bear));
  const critic = el("div", "case"), ch = el("h4", null, "Critic");
  ch.append(el("span", "chip " + (p.verdict === "accept" ? "chip-up" : p.verdict === "reject" ? "chip-down" : "chip-warn"), p.verdict));
  critic.append(ch);
  if (p.critic_reasons && p.critic_reasons.length) {
    const ol = el("ul", "reasons");
    for (const r of p.critic_reasons) ol.append(el("li", null, r));
    critic.append(ol);
  }
  if (p.failed_checks && p.failed_checks.length) {
    const ids = el("span", "ev-ids");
    for (const c of p.failed_checks) ids.append(el("span", "ev-id", c));
    critic.append(ids);
  }
  box.append(critic);
  if (p.catalysts && p.catalysts.length) {
    const cat = el("div", "case"), ul = el("ul", "reasons");
    cat.append(el("h4", null, "Events inside the holding period"));
    for (const k of p.catalysts) ul.append(el("li", null, [k.description || k.kind, k.event_date ? `(${dateLabel(k.event_date)})` : ""].join(" ")));
    cat.append(ul);
    box.append(cat);
  }
  if (d.evidence.length) {
    const ev = el("div", "case"), ul = el("ul", "evidence-list");
    ev.append(el("h4", null, `Evidence the agents could cite (${d.evidence.length})`));
    for (const e of d.evidence) {
      const li = el("li"), title = el("span", "ev-title"), url = safeUrl(e.url);
      if (url) { const a = el("a", null, e.title); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; title.append(a); }
      else title.textContent = e.title;
      const meta = el("span", "ev-meta", `${e.important ? "Important · " : ""}${dateLabel(e.published_at)}`);
      li.append(title, meta);
      const text = (e.text || "").trim();
      if (text && e.kind === "announcement") li.append(el("span", "ev-text", text.length > 280 ? text.slice(0, 279) + "…" : text));
      li.append(el("span", "ev-text mono", e.evidence_id));
      ul.append(li);
    }
    ev.append(ul);
    box.append(ev);
  }
}

function renderActions(d) {
  const box = $("d-actions");
  box.replaceChildren();
  $("d-action-error").textContent = "";
  const a = d.actions, sym = d.symbol;
  for (const o of d.pending) {
    const n = el("div", "notice");
    n.append(el("span", null, `${o.action === "buy" ? "Your buy" : "Your exit"} goes in at the open of ${dateLabel(o.trade_date)}.`),
      button("Cancel", "btn-secondary btn-sm", () => cancelOrder(o.id)));
    box.append(n);
  }
  if (busy()) return;
  if (a.buy) box.append(button("Buy at the open", "btn-primary btn-sm", () => placeOrder(sym, "buy")));
  if (a.exit) box.append(button("Exit at the open", "btn-danger btn-sm", () => placeOrder(sym, "exit")));
  if (a.remove) box.append(button(d.plan && d.plan.accepted ? "Skip this trade" : "Remove from the list", "btn-secondary btn-sm", () => removeStock(sym)));
  if (a.restore) box.append(button("Put back on the list", "btn-secondary btn-sm", () => restoreStock(sym)));
  if (d.is_today && !morning() && state.desk.phase !== "finished") box.append(el("p", "card-sub", "Orders are taken in the morning, before the session runs."));
}

async function loadTrace() {
  const d = state.detail;
  if (!d) return;
  try {
    state.trace = await GET(deskPath(`/trace?date=${d.trade_date}&symbol=${encodeURIComponent(d.symbol)}`));
  } catch (e) { toast(e.message); return; }
  const list = $("trace-list");
  list.replaceChildren();
  if (!state.trace.length) list.append(el("li", null, "No trace steps for this stock on this day."));
  for (const s of state.trace) {
    const li = el("li"), top = el("div", "trace-top");
    const kind = s.kind === "llm_call" ? ["LLM", "chip-preview"] : s.kind === "offline" ? ["Rules", "chip-muted"] : [s.kind, "chip-warn"];
    top.append(el("span", "mono", `#${s.step} ${s.agent}`), el("span", "chip " + kind[1], kind[0]));
    const meta = [s.provider, s.prompt_version, s.latency_ms ? `${nf0.format(s.latency_ms)} ms` : null,
      s.input_tokens ? `${nf0.format(s.input_tokens)} + ${nf0.format(s.output_tokens)} tokens` : null, s.cached ? "cached" : null].filter(Boolean).join(" · ");
    top.append(el("span", "trace-meta", meta));
    const json = JSON.stringify(s.output, null, 1);
    li.append(top, el("pre", "trace-out", json.length > 1600 ? json.slice(0, 1599) + "…" : json));
    list.append(li);
  }
}

function renderPipeline() {
  const d = state.desk, m = state.meta, list = $("pipeline");
  const llm = d.provider === "offline" ? ["Rules", "chip-muted"] : ["LLM", "chip-signal"];
  const t = d.llm_today || {};
  const items = [
    ["Prices and filings", ["Lake", "chip-muted"], `NSE prices, corporate actions and announcements, read point-in-time: today the desk sees data to the close of ${dateLabel(d.visible_through)}.`],
    [`Base forecast (${(m.base_model || "tft").toUpperCase()})`, ["Walk-forward", "chip-preview"], "Out-of-sample forecasts the agents see as one piece of evidence. They showed no direction edge in validation."],
    [`Forecast range (${(m.vol_model || "lstm").toUpperCase()})`, ["Walk-forward", "chip-preview"], "The model's volatility forecast draws the range on the chart. It is the forecast that held up in validation."],
    ["Agents", llm, `Planner, Bull, Bear, Forecaster, Critic and Exit Monitor${d.provider === "offline" ? " as rule-based stand-ins" : ` on ${d.model || d.provider}`}. Today: ${plural(t.llm_calls || 0, "LLM call")}, ${plural(t.fallbacks || 0, "fallback")} to rules.`],
    ["Execution", ["Paper", "chip-warn"], "Fills at the open on as-traded prices with STT, exchange, SEBI, stamp duty, GST and DP charges. 1% risk per trade, at most 5 positions."],
  ];
  list.replaceChildren(...items.map(([name, [chip, cls], note]) => {
    const li = el("li");
    li.append(el("span", "pl-name", name), el("span", "chip " + cls, chip), el("span", "pl-note", note));
    return li;
  }));
}

/* ================= Morning actions ================= */
async function addStock(raw) {
  const sym = (raw || "").trim().toUpperCase().replace(/\.NS$/, "");
  $("add-error").textContent = "";
  if (!sym) { $("add-error").textContent = "Enter a stock symbol, for example SBIN."; return; }
  try {
    const r = await POST(deskPath("/watchlist"), { symbol: sym });
    $("add-ticker").value = "";
    if (r.restored) { toast(`${sym} is back on today's list`); state.selected = sym; await loadView(); return; }
    state.jobId = r.job_id;
    state.jobDone = 0;
    showJob({ id: r.job_id, kind: "analyse", total: 1, done: 0, status: "queued", message: "Waiting to start", params: { symbol: sym } });
    renderShell();
    renderWatchlist();
    schedulePoll(500);
  } catch (e) { $("add-error").textContent = e.message; }
}
async function removeStock(sym) {
  try {
    await DEL(deskPath(`/watchlist/${encodeURIComponent(sym)}`));
    toast(`${sym} removed: its plan won't be traded today`);
    await refreshAll();
  } catch (e) { $("add-error").textContent = e.message; }
}
async function restoreStock(sym) { await addStock(sym); }
async function placeOrder(sym, action) {
  try {
    await POST(deskPath("/orders"), { symbol: sym, action });
    toast(action === "buy" ? `Buy order for ${sym} goes in at the open` : `Exit order for ${sym} goes in at the open`);
    await refreshAll();
  } catch (e) {
    const box = state.view === "tradebook" ? $("pos-error") : $("d-action-error");
    box.textContent = e.message;
  }
}
async function cancelOrder(id) {
  try {
    await DEL(deskPath(`/orders/${id}`));
    toast("Order cancelled");
    await refreshAll();
  } catch (e) { toast(e.message); }
}

/* ================= Trade book ================= */
function renderTradebook() {
  const b = state.book, k = b.kpis, d = state.desk, open = morning() && !busy();
  setKpi("t-day", b.session ? signedMoney(k.day_pnl) : "—", toneOf(k.day_pnl), b.session ? `Session of ${dateLabel(b.session)}, after costs` : "No session run yet");
  setKpi("t-real", signedMoney(k.booked), toneOf(k.booked), `${plural(k.closed_today, "trade")} closed that session`);
  setKpi("t-unreal", signedMoney(k.open_pnl), toneOf(k.open_pnl),
    b.positions.length ? `on ${plural(b.positions.length, "open position")} at the close of ${dateLabel(b.visible_through)}` : "No open positions");
  setKpi("t-orders", String(k.orders), "", `${compactINR(k.turnover)} turnover`);
  $("pos-error").textContent = "";

  const body = $("pos-body");
  body.replaceChildren();
  $("pos-empty").hidden = b.positions.length > 0;
  for (const p of b.positions) {
    const tr = el("tr"), c0 = el("td");
    c0.append(el("span", "sym mono", p.symbol), el("span", "sym-sub", `${SOURCE_LABEL[p.source] || p.source} · since ${dateLabel(p.entry_date)}`));
    if (p.exit_decision && p.exit_decision.action !== "hold") {
      c0.append(el("span", "cell-note", p.exit_decision.action === "exit" ? "Exit Monitor: sells at the open" : `Exit Monitor: stop raised to ${nf2.format(p.exit_decision.new_stop)}`));
    }
    const act = el("td", "r");
    if (open && p.pending_exit) {
      const pend = (b.pending.find((o) => o.symbol === p.symbol && o.action === "exit") || {}).id;
      act.append(button("Cancel exit", "btn-secondary btn-sm", () => cancelOrder(pend)));
    } else if (open) {
      const btn = button("Exit at open", "btn-secondary btn-sm", () => onExitClick(p.symbol, btn));
      act.append(btn);
    } else if (p.pending_exit) act.append(el("span", "chip chip-warn", "Exit pending"));
    tr.append(c0, el("td", "r num", nf0.format(p.quantity)), el("td", "r num", nf2.format(p.entry_price)), el("td", "r num", nf2.format(p.last_close)),
      el("td", "r num strong " + toneOf(p.pnl), signedMoney(p.pnl)), el("td", "r num " + toneOf(p.return_pct), signedPct(p.return_pct * 100)),
      el("td", "r num", `${nf2.format(p.stop)} / ${nf2.format(p.target)}`), el("td", "r num", `${p.days} / ${p.max_hold_days}`), act);
    body.append(tr);
  }

  $("pending-sub").textContent = open ? `Filled at the open of ${dateLabel(d.current_date)} when you run the day` : "Orders are taken in the morning, before the session runs";
  const pb = $("pending-body");
  pb.replaceChildren();
  if (!b.pending.length) {
    const tr = el("tr"), td = el("td", "empty-cell", "No orders waiting. Buy an analysed stock from its report, or exit a position above.");
    td.colSpan = 5;
    tr.append(td);
    pb.append(tr);
  }
  for (const o of b.pending) {
    const tr = el("tr"), c0 = el("td"), act = el("td", "r");
    c0.append(el("span", "sym mono", o.symbol));
    if (open) act.append(button("Cancel", "btn-secondary btn-sm", () => cancelOrder(o.id)));
    tr.append(c0, el("td", o.action === "buy" ? "side-buy" : "side-sell", o.action === "buy" ? "BUY at open" : "SELL at open"),
      el("td", "num", dateLabel(o.trade_date)), el("td", "r num", o.stop == null ? "—" : `${nf2.format(o.stop)} / ${nf2.format(o.target)}`), act);
    pb.append(tr);
  }

  $("orders-sub").textContent = b.orders_date ? `Session of ${dayLong(b.orders_date)}, newest first` : "No session run yet";
  const ob = $("orders-body");
  ob.replaceChildren();
  if (!b.orders.length) {
    const tr = el("tr"), td = el("td", "empty-cell", "No orders yet. Run the day to trade the morning's plans.");
    td.colSpan = 9;
    tr.append(td);
    ob.append(tr);
  }
  for (const o of b.orders) {
    const tr = el("tr"), stock = el("td"), status = el("td"), src = el("td");
    stock.append(el("span", "sym mono", o.symbol));
    src.append(sourceChip(o.source));
    const ok = o.status === "filled", chip = el("span", "chip " + (ok ? "chip-up" : "chip-warn"));
    chip.innerHTML = `<svg width="12" height="12" aria-hidden="true"><use href="#i-${ok ? "check" : "cross"}"/></svg>`;
    chip.append(document.createTextNode(ok ? "Filled" : "Not filled"));
    status.append(chip);
    tr.append(el("td", "num mono", AT_LABEL[o.at] || o.at), el("td", "num mono", `D${d.id}-${String(o.id).padStart(6, "0")}`), stock,
      el("td", o.side === "buy" ? "side-buy" : "side-sell", o.side.toUpperCase()), el("td", null, KIND_LABEL[o.kind] || o.kind),
      el("td", "r num", o.quantity ? nf0.format(o.quantity) : "—"), el("td", "r num", o.price == null ? "—" : nf2.format(o.price)), src, status);
    ob.append(tr);
  }
}
function onExitClick(sym, btn) {
  if (state.confirm === sym) { state.confirm = null; placeOrder(sym, "exit"); return; }
  state.confirm = sym;
  btn.textContent = "Confirm exit";
  btn.className = "btn btn-danger btn-sm";
  setTimeout(() => {
    if (state.confirm !== sym) return;
    state.confirm = null;
    if (btn.isConnected) { btn.textContent = "Exit at open"; btn.className = "btn btn-secondary btn-sm"; }
  }, 4000);
}

/* ================= Trade history ================= */
function periodCutoff(period) {
  if (!period) return null;
  const days = state.days.filter((d) => d.closed).map((d) => d.trade_date);
  return days.length ? days[Math.max(0, days.length - period)] : null;
}
function filterTrades(period, symbol) {
  const cut = periodCutoff(period);
  return state.trades.filter((t) => (!cut || t.exit_date >= cut) && (symbol === "all" || t.symbol === symbol));
}
function fillSymbolSelect(id, value) {
  const sel = $(id), syms = [...new Set(state.trades.map((t) => t.symbol))].sort();
  sel.replaceChildren(new Option("All stocks", "all"), ...syms.map((s) => new Option(s, s)));
  sel.value = syms.includes(value) ? value : "all";
}
function renderHistory() {
  const f = state.hist;
  const rows = filterTrades(f.period, f.symbol).filter((t) => f.result === "all" || (f.result === "win") === (t.net_pnl > 0));
  const net = sum(rows.map((t) => t.net_pnl)), wins = rows.filter((t) => t.net_pnl > 0).length;
  $("h-summary").textContent = rows.length ? `${plural(rows.length, "trade")} · net ${signedMoney0(net)} after costs · ${pct(wins / rows.length)} winners` : "";
  $("h-empty").hidden = rows.length > 0;
  $("h-empty").textContent = state.trades.length ? "No trades match these filters. Try a longer period or a different stock." : "No closed trades yet. Run some days on the desk.";
  const body = $("h-body");
  body.replaceChildren();
  for (const t of rows.slice(0, f.limit)) {
    const tr = el("tr"), stock = el("td"), src = el("td"), setup = el("td");
    stock.append(el("span", "sym mono", t.symbol));
    src.append(sourceChip(t.source));
    setup.append(setupChip(t.setup_type));
    tr.append(el("td", "num", dateYear(t.exit_date)), stock, el("td", "r num", nf0.format(t.quantity)),
      el("td", "r num", nf2.format(t.entry_price)), el("td", "r num", nf2.format(t.exit_price)),
      el("td", "r num strong " + toneOf(t.net_pnl), signedMoney(t.net_pnl)), el("td", "r num " + toneOf(t.return_pct), signedPct(t.return_pct * 100)),
      el("td", "r num", plural(t.held_days, "day")), src, setup, el("td", null, EXIT_LABEL[t.exit_reason] || t.exit_reason));
    body.append(tr);
  }
  const left = rows.length - f.limit;
  $("h-more-row").hidden = left <= 0;
  $("h-more").textContent = `Show ${Math.min(25, left)} more`;
}

/* ================= Analysis ================= */
function stats(trades) {
  const chron = trades.slice().sort((a, b) => a.exit_date.localeCompare(b.exit_date) || a.id - b.id);
  let cum = 0, peak = 0, maxDD = 0;
  const curve = [{ start: true, cum: 0, t: chron[0] }];
  for (const t of chron) {
    cum += t.net_pnl;
    peak = Math.max(peak, cum);
    maxDD = Math.max(maxDD, peak - cum);
    curve.push({ cum, t });
  }
  const wins = trades.filter((t) => t.net_pnl > 0), losses = trades.filter((t) => t.net_pnl <= 0);
  const gw = sum(wins.map((t) => t.net_pnl)), gl = -sum(losses.map((t) => t.net_pnl)), n = trades.length;
  return {
    n, net: cum, wins: wins.length, winRate: n ? wins.length / n : 0, pf: gl > 0 ? gw / gl : null,
    avgWin: wins.length ? gw / wins.length : 0, avgLoss: losses.length ? gl / losses.length : 0,
    expectancy: n ? cum / n : 0, maxDD, curve,
  };
}
function groupOf(trades, pred, key) {
  const ts = trades.filter(pred);
  return { key, n: ts.length, pnl: sum(ts.map((t) => t.net_pnl)), winRate: ts.length ? ts.filter((t) => t.net_pnl > 0).length / ts.length : 0 };
}
function groupBy(trades, keyFn) {
  return [...new Set(trades.map(keyFn))].map((k) => groupOf(trades, (t) => keyFn(t) === k, k));
}
function insights(trades, st) {
  if (st.n < 4) return [[["Not enough closed trades in this period to spot patterns. Pick a longer period or run more days."]]];
  const out = [];
  const ag = groupOf(trades, (t) => t.source === "agents", "agents"), man = groupOf(trades, (t) => t.source === "manual", "manual");
  if (ag.n >= 3 && man.n >= 3) out.push([["The agents' plans won "], [pct(ag.winRate), 1], [` of ${ag.n} trades. Your own orders won `], [pct(man.winRate), 1], [` of ${man.n}.`]]);
  if (state.ana.symbol === "all") {
    const bs = groupBy(trades, (t) => t.symbol).sort((a, b) => b.pnl - a.pnl), best = bs[0], worst = bs[bs.length - 1];
    if (best && best.pnl > 0) {
      const seg = [[best.key, 1], [` made the most, ${signedMoney0(best.pnl)} over ${plural(best.n, "trade")}.`]];
      if (worst && worst.pnl < 0 && worst !== best) seg.push([" "], [worst.key, 1], [` lost the most, ${signedMoney0(worst.pnl)}.`]);
      out.push(seg);
    }
  }
  const setups = groupBy(trades, (t) => t.setup_type).filter((g) => g.n >= 3).sort((a, b) => b.winRate - a.winRate);
  if (setups.length >= 2 && setups[0].winRate - setups[setups.length - 1].winRate >= 0.1) {
    const a = setups[0], b = setups[setups.length - 1];
    out.push([[`${SETUP_LABEL[a.key] || a.key} trades won `], [pct(a.winRate), 1], [` of ${a.n}, against `], [pct(b.winRate), 1], [` for ${(SETUP_LABEL[b.key] || b.key).toLowerCase()} trades.`]]);
  }
  const stops = trades.filter((t) => t.exit_reason === "stop" || t.exit_reason === "gap_stop").length;
  const targets = trades.filter((t) => t.exit_reason === "target").length;
  out.push([[`${stops} of ${st.n} trades hit the stop and `], [String(targets), 1], [` reached the target; the rest ended on time or by an exit request.`]]);
  if (st.avgWin > 0 && st.avgLoss > 0) {
    const payoff = st.avgWin / st.avgLoss, be = 1 / (1 + payoff);
    out.push([["The average win is "], [payoff.toFixed(1) + "×", 1], [" the average loss, so any win rate above "], [pct(be), 1], [` breaks even. This period's win rate is ${pct(st.winRate)}.`]]);
  }
  return out;
}
function renderAnalysis() {
  const trades = filterTrades(state.ana.period, state.ana.symbol);
  $("ana-empty").hidden = trades.length > 0;
  $("ana-empty").textContent = state.trades.length ? "No closed trades in this period. Pick a longer period or another stock." : "No closed trades yet. Run some days on the desk.";
  $("ana-content").hidden = trades.length === 0;
  if (!trades.length) { hideTip(); return; }
  const st = stats(trades);
  setKpi("a-net", signedMoney0(st.net), toneOf(st.net), `${plural(st.n, "trade")} · ${signedMoney0(st.expectancy)} per trade on average, after costs`);
  setKpi("a-win", pct(st.winRate), "", `${st.wins} wins · ${st.n - st.wins} losses`);
  setKpi("a-pf", st.pf == null ? "—" : st.pf.toFixed(2), "", "Gross profit ÷ gross loss");
  setKpi("a-payoff", st.avgWin && st.avgLoss ? (st.avgWin / st.avgLoss).toFixed(2) + "×" : "—", "", `Avg win ${compactINR(st.avgWin)} · avg loss ${compactINR(st.avgLoss)}`);
  setKpi("a-dd", st.maxDD ? "−" + compactINR(st.maxDD) : "₹0", st.maxDD ? "down" : "", "Largest drop from a running peak");

  drawEquity($("ch-equity"), st.curve);
  fillTable($("tb-equity"), ["Exit date", "Stock", "P&L", "Cumulative"],
    st.curve.filter((p) => !p.start).map((p) => [dateYear(p.t.exit_date), p.t.symbol, signedMoney(p.t.net_pnl), signedMoney0(p.cum)]), [2, 3]);

  const list = $("a-insights");
  list.replaceChildren();
  for (const parts of insights(trades, st)) {
    const li = el("li"), span = el("span");
    for (const [text, strong] of parts) span.append(strong ? el("strong", null, text) : document.createTextNode(text));
    li.append(span);
    list.append(li);
  }

  const single = state.ana.symbol !== "all";
  const byStock = groupBy(trades, (t) => t.symbol).sort((a, b) => b.pnl - a.pnl);
  $("stock-note").hidden = !single;
  $("ch-stock").hidden = single;
  if (!single) {
    const box = $("ch-stock");
    drawHBars(box, byStock.map((g) => ({ label: g.key, value: g.pnl, g })), {
      title: "Net P&L by stock", labelW: 104, valueW: 80, labelClass: "lbl-mono",
      fmt: (it) => signedMoney0(it.value), color: (it) => (it.value >= 0 ? "var(--up)" : "var(--down)"),
    });
    box._tip = (i) => {
      const g = byStock[i];
      return { title: g.key, rows: [{ value: signedMoney0(g.pnl), label: "Net P&L" }, { value: String(g.n), label: g.n === 1 ? "trade" : "trades" }, { value: pct(g.winRate), label: "win rate" }] };
    };
  }
  fillTable($("tb-stock"), ["Stock", "Trades", "Win rate", "Net P&L"], byStock.map((g) => [g.key, String(g.n), pct(g.winRate), signedMoney0(g.pnl)]), [1, 2, 3]);

  const bySrc = ["agents", "manual"].map((k) => groupOf(trades, (t) => t.source === k, k)).filter((g) => g.n);
  const srcBox = $("ch-source");
  drawHBars(srcBox, bySrc.map((g) => ({ label: SOURCE_LABEL[g.key], value: g.winRate * 100, g })), {
    title: "Win rate by entry source", labelW: 120, valueW: 92, labelClass: "lbl-ui", domain: [0, 100], ref: 50, refLabel: "50%",
    fmt: (it) => `${Math.round(it.value)}% of ${it.g.n}`, color: () => "var(--series-1)",
  });
  srcBox._tip = (i) => {
    const g = bySrc[i];
    return { title: SOURCE_LABEL[g.key], rows: [{ key: "var(--series-1)", value: pct(g.winRate), label: "win rate" }, { value: String(g.n), label: g.n === 1 ? "trade" : "trades" }, { value: signedMoney0(g.pnl), label: "net P&L" }] };
  };
  fillTable($("tb-source"), ["Entry source", "Trades", "Win rate", "Net P&L"], bySrc.map((g) => [SOURCE_LABEL[g.key], String(g.n), pct(g.winRate), signedMoney0(g.pnl)]), [1, 2, 3]);

  const bySetup = ["pullback_uptrend", "breakout", "oversold_bounce", "none"].map((k) => groupOf(trades, (t) => t.setup_type === k, k));
  const setupBox = $("ch-setup");
  drawColumns(setupBox, bySetup.map((g) => ({ value: g.pnl, g })), {
    title: "Net P&L by setup", tick: (v) => compactINR(v), color: (it) => (it.value >= 0 ? "var(--up)" : "var(--down)"),
    cap: (it) => (it.g.n ? signedCompact(it.value) : null), xLabel: (it) => SETUP_LABEL[it.g.key],
    aria: (it) => `${SETUP_LABEL[it.g.key]}: ${signedMoney0(it.value)} over ${it.g.n} trades`,
  });
  setupBox._tip = (i) => {
    const g = bySetup[i];
    return { title: SETUP_LABEL[g.key], rows: [{ value: signedMoney0(g.pnl), label: "net P&L" }, { value: String(g.n), label: g.n === 1 ? "trade" : "trades" }, { value: g.n ? pct(g.winRate) : "—", label: "win rate" }] };
  };
  fillTable($("tb-setup"), ["Setup", "Trades", "Win rate", "Net P&L"], bySetup.map((g) => [SETUP_LABEL[g.key], String(g.n), g.n ? pct(g.winRate) : "—", signedMoney0(g.pnl)]), [1, 2, 3]);

  const w = 1, rets = trades.map((t) => t.return_pct * 100);
  const b0 = Math.floor(Math.min(...rets) / w), b1 = Math.floor(Math.max(...rets) / w) + 1, bins = [];
  for (let k = b0; k < b1; k++) {
    const lo = k * w, hi = (k + 1) * w;
    bins.push({ lo, hi, n: rets.filter((r) => r >= lo && r < hi).length });
  }
  const edgeLabel = (v) => (Math.abs(v) < 1e-9 ? "0%" : `${v < 0 ? "−" : "+"}${+Math.abs(v).toFixed(2)}%`);
  const binLabel = (b) => `${edgeLabel(b.lo)} to ${edgeLabel(b.hi)}`;
  const distBox = $("ch-dist");
  drawColumns(distBox, bins.map((b) => ({ value: b.n, b })), {
    title: "Number of trades per return band", touch: true, integer: true, tick: (v) => String(v),
    color: (it) => (it.b.lo >= 0 ? "var(--up)" : "var(--down)"),
    edges: bins.map((b) => edgeLabel(b.lo)).concat([edgeLabel(bins[bins.length - 1].hi)]),
    aria: (it) => `${binLabel(it.b)}: ${it.b.n} trades`,
  });
  distBox._tip = (i) => ({ title: binLabel(bins[i]), rows: [{ value: String(bins[i].n), label: bins[i].n === 1 ? "trade" : "trades" }] });
  fillTable($("tb-dist"), ["Return band", "Trades"], bins.map((b) => [binLabel(b), String(b.n)]), [1]);
}

/* ================= Navigation ================= */
const VIEWS = {
  dashboard: ["Dashboard", () => {
    const d = state.desk;
    if (!d) return "";
    return `${dayLong(d.current_date)} · ${{ morning: "plans made at 08:30 IST; the market hasn't opened", closed: "session closed", new: "the agents are making the first plans", finished: "replay finished" }[d.phase] || d.phase}`;
  }],
  tradebook: ["Trade book", () => "Open positions, orders waiting for the open and the last session's fills"],
  history: ["Trade history", () => "Closed trades on this desk, after costs"],
  analysis: ["Trade analysis", () => "How this desk's closed trades perform by stock, entry source and setup"],
};
function navigate(view, updateHash = true) {
  if (!VIEWS[view]) view = "dashboard";
  const changed = state.view !== view;
  state.view = view;
  hideTip();
  document.querySelectorAll("[data-view]").forEach((b) => {
    if (b.dataset.view === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  if (updateHash) { try { history.replaceState(null, "", "#" + view); } catch (e) { /* sandboxed */ } }
  if (state.desk) {
    for (const k in VIEWS) $("view-" + k).hidden = k !== view;
    $("page-title").textContent = VIEWS[view][0];
    $("page-sub").textContent = VIEWS[view][1]();
    loadView().catch((e) => toast(e.message));
  }
  if (changed) window.scrollTo(0, 0);
}

/* ================= New desk ================= */
function openDeskDialog() {
  const m = state.meta, dlg = $("desk-dialog");
  const taken = new Set(state.desks.map((d) => d.name));
  let name = "2023 replay", k = 2;
  while (taken.has(name)) name = `2023 replay ${k++}`;
  $("f-name").value = name;
  for (const id of ["f-start", "f-end"]) { $(id).min = m.window.start; $(id).max = m.window.end; }
  $("f-start").value = m.window.start <= "2023-01-02" && "2023-01-02" <= m.window.end ? "2023-01-02" : m.window.start;
  $("f-end").value = "";
  $("f-window").textContent = `Any trading day from ${dayLong(m.window.start)} to ${dayLong(m.window.end)}, where the models have walk-forward forecasts. 2024 onwards is the frozen test period.`;
  $("f-capital").value = m.capital;
  const sel = $("f-provider");
  sel.replaceChildren(...m.providers.map((p) => {
    const o = new Option(p.ready ? p.label : `${p.label} (not configured)`, p.provider);
    o.disabled = !p.ready;
    return o;
  }));
  sel.value = "offline";
  providerHint();
  $("desk-error").textContent = "";
  dlg.showModal();
  $("f-name").focus();
}
function providerHint() {
  $("f-provider-hint").textContent = $("f-provider").value === "offline"
    ? "Instant: each agent's rule-based stand-in. Good for trying the desk."
    : "Each trading day makes about 40 LLM calls (around 6 minutes on a free endpoint). Responses are cached, and the daily call limit in config.yaml applies.";
}
async function submitDesk(e) {
  e.preventDefault();
  const btn = $("desk-submit");
  btn.disabled = true;
  $("desk-error").textContent = "";
  try {
    const r = await POST("/api/desks", {
      name: $("f-name").value, start_date: $("f-start").value, end_date: $("f-end").value || null,
      capital: Number($("f-capital").value) || null, provider: $("f-provider").value,
    });
    $("desk-dialog").close();
    toast(`Desk "${r.desk.name}" created. The agents are making the first plans.`);
    await loadDesks();
    await selectDesk(r.desk.id);
  } catch (err) {
    $("desk-error").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}
let deleteArmed = false;
async function onDeleteDesk() {
  const btn = $("delete-desk");
  if (!deleteArmed) {
    deleteArmed = true;
    btn.textContent = "Delete for good?";
    btn.classList.add("confirm");
    setTimeout(() => { deleteArmed = false; btn.textContent = "Delete"; btn.classList.remove("confirm"); }, 4000);
    return;
  }
  deleteArmed = false;
  btn.textContent = "Delete";
  btn.classList.remove("confirm");
  try {
    const name = state.desk ? state.desk.name : "";
    await DEL(deskPath());
    toast(`Desk "${name}" deleted`);
    await loadDesks();
    await selectDesk(state.desks.length ? state.desks[state.desks.length - 1].id : null);
  } catch (e) { toast(e.message); }
}

/* ================= Wire up ================= */
function bindSeg(id, onChange) {
  const seg = $(id);
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    onChange(b.dataset.value);
  });
}
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => navigate(b.dataset.view)));
window.addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if (VIEWS[h] && h !== state.view) navigate(h, false);
});
$("run-day").addEventListener("click", () => runDays(1));
$("run-day-m").addEventListener("click", () => runDays(1));
$("run-many").addEventListener("click", () => runDays(+$("run-n").value));
$("job-cancel").addEventListener("click", async () => {
  if (state.jobId == null) return;
  try { await POST(`/api/jobs/${state.jobId}/cancel`); $("job-cancel").disabled = true; } catch (e) { toast(e.message); }
});
for (const id of ["desk-select", "desk-select-m"]) $(id).addEventListener("change", (e) => selectDesk(+e.target.value).catch((err) => toast(err.message)));
$("new-desk-btn").addEventListener("click", openDeskDialog);
$("no-desk-create").addEventListener("click", openDeskDialog);
$("delete-desk").addEventListener("click", onDeleteDesk);
$("desk-form").addEventListener("submit", submitDesk);
$("desk-cancel").addEventListener("click", () => $("desk-dialog").close());
$("f-provider").addEventListener("change", providerHint);
$("add-form").addEventListener("submit", (e) => { e.preventDefault(); addStock($("add-ticker").value); });
bindSeg("feed-filter", (v) => { state.feedFilter = v; renderFeed(); });
bindSeg("h-period", (v) => { state.hist.period = +v; state.hist.limit = 25; renderHistory(); });
bindSeg("h-result", (v) => { state.hist.result = v; state.hist.limit = 25; renderHistory(); });
bindSeg("a-period", (v) => { state.ana.period = +v; renderAnalysis(); });
$("h-symbol").addEventListener("change", (e) => { state.hist.symbol = e.target.value; state.hist.limit = 25; renderHistory(); });
$("a-symbol").addEventListener("change", (e) => { state.ana.symbol = e.target.value; renderAnalysis(); });
$("h-more").addEventListener("click", () => { state.hist.limit += 25; renderHistory(); });
$("price-details").addEventListener("toggle", () => { if ($("price-details").open && state.detail) renderPriceTable(state.detail); });
$("trace-details").addEventListener("toggle", () => { if ($("trace-details").open && !state.trace) loadTrace(); });
bindCrosshair($("price-chart"));
bindCrosshair($("ch-equity"));
["ch-stock", "ch-source", "ch-setup", "ch-dist"].forEach((id) => bindMarks($(id)));

let resizeRaf = 0;
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    hideTip();
    if (state.view === "dashboard" && state.detail) drawPriceChart($("price-chart"), state.detail);
    else if (state.view === "analysis" && state.trades.length) renderAnalysis();
  });
});
window.addEventListener("scroll", () => { if (tipOwner) hideTip(); }, { passive: true });
window.addEventListener("unhandledrejection", (e) => { toast(e.reason && e.reason.message ? e.reason.message : "Something went wrong"); });

async function init() {
  $("page-sub").textContent = "Loading market data…";
  try {
    state.meta = await GET("/api/meta");
  } catch (e) {
    $("page-sub").textContent = `The desk server isn't answering (${e.message}). Start it with: python -m src.api`;
    return;
  }
  $("ticker-options").replaceChildren(...state.meta.symbols.map((s) => new Option(s.name, s.symbol)));
  await loadDesks();
  const saved = store.get("twa-desk");
  state.deskId = state.desks.some((d) => d.id === saved) ? saved : state.desks.length ? state.desks[state.desks.length - 1].id : null;
  const initial = location.hash.slice(1);
  state.view = VIEWS[initial] ? initial : "dashboard";
  document.querySelectorAll("[data-view]").forEach((b) => {
    if (b.dataset.view === state.view) b.setAttribute("aria-current", "page");
  });
  await selectDesk(state.deskId);
}
init();
})();
