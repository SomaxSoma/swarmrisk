/* SwarmRisk dashboard */
"use strict";

const $ = (id) => document.getElementById(id);
const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const G = { strong: "#fbfbfb", text: "#ededed", soft: "#c2c2c2", muted: "#8a8a8a", dim: "#6f6f6f", line: "rgba(255,255,255,.08)", bg: "#060606" };
// Monochrome severity: the brighter the bar, the worse the outcome.
const SHADE = { low: "#5c5c5c", moderate: "#8f8f8f", high: "#c6c6c6", critical: "#f2f2f2" };
const level = (p) => (p < 0.1 ? "low" : p < 0.3 ? "moderate" : p < 0.6 ? "high" : "critical");

const state = { schema: null, presets: [], preset: null, modified: false, report: null, sim: null, seq: 0, tab: "outcomes", runs: 2000 };

const pct = (p, d = 0) => `${(100 * p).toFixed(d)}%`;
const pctSmart = (p) => pct(p, p > 0 && p < 0.1 ? 1 : 0);
const money = (m) => (m >= 1000 ? `$${(m / 1000).toFixed(1)}B` : m >= 1 ? `$${m.toFixed(1)}M` : `$${Math.round(m * 1000)}K`);
const days = (d) => (d < 1 ? `${Math.round(d * 24)} h` : d < 60 ? `${d.toFixed(d < 10 ? 1 : 0)} days` : `${(d / 30.4).toFixed(1)} months`);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  // The static GitHub Pages build answers in the browser (Pyodide) instead of a server.
  if (window.SWARMRISK_BACKEND) return window.SWARMRISK_BACKEND(path, body);
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
  return data;
}

function showError(msg) {
  const el = $("error");
  el.hidden = !msg;
  el.textContent = msg || "";
}

/* ---------------------------------------------------------------- tabs ---- */
const TABS = ["outcomes", "drivers", "actions", "threat", "simulation", "model"];

function showTab(name, scroll) {
  if (!TABS.includes(name)) name = "outcomes";
  state.tab = name;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on);
  });
  document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  renderCharts();
  if (scroll) {
    const top = $("tabs-wrap").getBoundingClientRect().top + window.scrollY;
    if (window.scrollY > top) window.scrollTo({ top, behavior: "smooth" });
  }
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.onclick = () => { history.replaceState(null, "", `#${t.dataset.tab}`); showTab(t.dataset.tab, true); };
  });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), true));
  const wrap = $("tabs-wrap");
  const onScroll = () => wrap.classList.toggle("nav-glass", wrap.getBoundingClientRect().top <= 0 && window.scrollY > 20);
  window.addEventListener("scroll", onScroll, { passive: true });
  showTab(location.hash.slice(1) || "outcomes");
}

/* ---------------------------------------------------------------- form ---- */
const isFraction = (f) => f.kind === "float" && f.low === 0 && f.high === 1;
const toPos = (f, v) => (f.log_scale ? Math.log10(Number(v) + 1) : Number(v));
const fromPos = (f, x) => {
  let v = f.log_scale ? Math.pow(10, x) - 1 : x;
  v = Math.min(f.high, Math.max(f.low, v));
  return f.kind === "int" ? Math.round(v) : Math.round(v * 100) / 100;
};

function buildForm(schema) {
  const form = $("profile-form");
  form.innerHTML = "";
  const groups = {};
  for (const f of schema.features) (groups[f.group] ||= []).push(f);
  const narrow = window.matchMedia("(max-width: 980px)").matches;
  Object.entries(groups).forEach(([group, feats]) => {
    const det = document.createElement("details");
    det.className = "group";
    det.open = !narrow && (group === "Threat scenario" || group === "Defences");
    det.innerHTML = `<summary>${esc(group)}</summary><div class="fields"></div>`;
    const box = det.querySelector(".fields");
    for (const f of feats) box.appendChild(fieldEl(f, schema));
    form.appendChild(det);
  });
  form.addEventListener("input", onFormChange);
  form.addEventListener("change", onFormChange);
}

function fieldEl(f, schema) {
  const div = document.createElement("div");
  div.className = "field";
  const help = `<span class="help" title="${esc(f.help)}" aria-label="${esc(f.help)}">?</span>`;
  const id = `f-${f.name}`;
  if (f.kind === "category") {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label></div>
      <select id="${id}" data-name="${f.name}">${schema.industries.map((i) => `<option value="${i}">${i[0].toUpperCase() + i.slice(1)}</option>`).join("")}</select>`;
  } else if (f.kind === "bool") {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label>
      <label class="switch"><input type="checkbox" id="${id}" data-name="${f.name}"><span></span></label></div>`;
  } else if (isFraction(f)) {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label><span class="pct" id="${id}-out"></span></div>
      <input type="range" id="${id}" data-name="${f.name}" min="0" max="1" step="0.01" aria-label="${esc(f.label)}">`;
  } else {
    const lo = toPos(f, f.low), hi = toPos(f, f.high);
    const step = f.log_scale ? 0.001 : f.kind === "int" ? 1 : (f.high - f.low) / 400;
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label>
      <input type="number" id="${id}" data-name="${f.name}" min="${f.low}" max="${f.high}" step="${f.kind === "int" ? 1 : "any"}"></div>
      <input type="range" id="${id}-r" data-slider-for="${f.name}" min="${lo}" max="${hi}" step="${step}" aria-label="${esc(f.label)}">`;
  }
  return div;
}

function featureByName(name) { return state.schema.features.find((f) => f.name === name); }

function setProfile(p) {
  for (const f of state.schema.features) {
    const el = $(`f-${f.name}`);
    const v = p[f.name];
    if (f.kind === "bool") el.checked = !!v;
    else if (f.kind === "category") el.value = v;
    else if (isFraction(f)) { el.value = v; $(`f-${f.name}-out`).textContent = pct(v); }
    else { el.value = v; $(`f-${f.name}-r`).value = toPos(f, v); }
  }
}

function getProfile() {
  const p = {};
  for (const f of state.schema.features) {
    const el = $(`f-${f.name}`);
    if (f.kind === "bool") p[f.name] = el.checked;
    else if (f.kind === "category") p[f.name] = el.value;
    else {
      let v = Number(el.value);
      if (!Number.isFinite(v)) v = f.default;
      p[f.name] = f.kind === "int" ? Math.round(v) : v;
    }
  }
  return p;
}

let debounce;
function onFormChange(e) {
  const t = e.target;
  if (t.dataset.sliderFor) {                       // slider moved -> update number box
    const f = featureByName(t.dataset.sliderFor);
    $(`f-${f.name}`).value = fromPos(f, Number(t.value));
  } else if (t.type === "number") {                // number typed -> move slider
    const f = featureByName(t.dataset.name);
    const v = Number(t.value);
    if (Number.isFinite(v)) $(`f-${f.name}-r`).value = toPos(f, Math.min(f.high, Math.max(f.low, v)));
  } else if (t.type === "range") {
    $(`${t.id}-out`).textContent = pct(Number(t.value));
  }
  state.modified = true;
  if ($("auto").checked) {
    clearTimeout(debounce);
    debounce = setTimeout(assess, 350);
  }
}

/* -------------------------------------------------------------- render ---- */
function render(rep) {
  const pr = rep.prediction, p = rep.profile;
  const name = state.preset?.name || "Default company";
  $("hero-kicker").textContent = `AI-agent swarm attack · ${name}${state.modified ? " (edited)" : ""} · ` +
    `${p.swarm_size.toLocaleString()} agents · sophistication ${p.attacker_sophistication.toFixed(2)}`;
  $("score").textContent = pr.resilience_score.toFixed(0);
  $("hero-sub").innerHTML = `Resilience score, grade <span class="hero-em">${pr.grade}</span>; higher is safer. ` +
    `This swarm gets in <span class="hero-em">${pctSmart(pr.p_breach)}</span> of the time and leaves the company unable to survive in ` +
    `<span class="hero-em">${pctSmart(pr.p_collapse)}</span> of attacks.`;
  $("summary").textContent = pr.summary;

  const hero = [
    ["Breach", pr.p_breach], ["Severe damage", pr.p_severe_damage], ["Collapse", pr.p_collapse],
    ["Detonation", pr.p_detonation_given_breach], ["Damage", pr.expected_damage_given_breach / 100],
  ];
  $("hero-bars").innerHTML = hero.map(([label, v], i) => `<div class="hrow" title="${label}: ${pct(v, 1)}">
      <div class="hero-bar-pct">${pctSmart(v)}</div>
      <div class="hero-bar-fill" style="height:${Math.max(2, 100 * v * 0.78)}%;background:${SHADE[level(v)]};animation-delay:${i * 60}ms"></div></div>`).join("");
  $("hero-bar-labels").innerHTML = ["Breach", "Severe damage", "Collapse", "Detonates if in", "Systems damaged"].map((l) => `<div>${l}</div>`).join("");

  const out = [
    ["Attack breaches the company", pr.p_breach],
    ["Severe infrastructure damage", pr.p_severe_damage],
    ["Company collapses", pr.p_collapse],
    ["Detonates before containment", pr.p_detonation_given_breach],
  ];
  $("outcome-bars").innerHTML = out.map(([label, v]) => hbar(label, v, pctSmart(v), level(v))).join("");

  const r = pr.recovery_days;
  $("rec-p50").textContent = days(r.p50);
  $("rec-range").textContent = `median · 80% range ${days(r.p10)} – ${days(r.p90)}`;
  const maxD = Math.max(r.p90 * 1.15, 30);
  Object.assign($("rec-bar").style, { left: `${(100 * r.p10) / maxD}%`, width: `${Math.max(1, (100 * (r.p90 - r.p10)) / maxD)}%` });
  $("rec-mid").style.left = `${(100 * r.p50) / maxD}%`;
  $("rec-axis-max").textContent = days(maxD);
  $("rec-imp").textContent = days(pr.recovery_days_if_detonated.p50);
  $("rec-con").textContent = days(pr.recovery_days_if_contained.p50);
  $("loss50").textContent = money(pr.loss_musd.p50);
  $("loss90").textContent = money(pr.loss_musd.p90);
  $("deton").textContent = pct(pr.p_detonation_given_breach);
  $("dmg").textContent = `${pr.expected_damage_given_breach.toFixed(1)}%`;
  $("grade").textContent = `${pr.grade} · ${pr.resilience_score.toFixed(0)}/100`;

  renderDrivers(rep.drivers);
  renderRecs(rep.recommendations, pr);
  renderCharts();
}

function hbar(label, v, text, lvl, max = 1) {
  return `<div class="hbar hrow"><div class="hbar-label" title="${esc(label)}">${esc(label)}</div>
    <div class="hbar-track"><div class="hbar-fill" style="width:${(100 * Math.min(v, max)) / max}%;background:${SHADE[lvl] || G.soft}"></div></div>
    <div class="hbar-value">${text}</div>${lvl ? `<div class="hbar-tag ${lvl}">${lvl.toUpperCase()}</div>` : ""}</div>`;
}

function fmtVal(v) {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
  return String(v);
}

function renderDrivers(drivers) {
  if (!drivers.length) { $("driver-bars").innerHTML = `<p class="empty">No input moves the risk noticeably.</p>`; return; }
  const max = Math.max(...drivers.map((d) => Math.abs(d.impact)), 0.1);
  $("driver-bars").innerHTML = drivers.map((d) => {
    const w = (50 * Math.abs(d.impact)) / max;
    const tip = `${d.label}: this company ${fmtVal(d.value)}, typical ${fmtVal(d.typical)}. Breach ${(100 * d.d_breach).toFixed(1)} pp, collapse ${(100 * d.d_collapse).toFixed(2)} pp.`;
    return `<div class="dbar hrow" title="${esc(tip)}">
      <div class="dbar-label">${esc(d.label)}<small>${esc(fmtVal(d.value))} · typical ${esc(fmtVal(d.typical))}</small></div>
      <div class="dbar-track"><div class="dbar-fill ${d.impact > 0 ? "pos" : "neg"}" style="width:${w}%"></div></div>
      <div class="dbar-value">${d.impact > 0 ? "+" : "−"}${Math.abs(d.impact).toFixed(1)}</div></div>`;
  }).join("");
}

function renderRecs(rec, pr) {
  const items = rec.actions;
  if (!items.length) {
    $("recs").innerHTML = `<p class="empty">No single change in the catalogue lowers this company's risk noticeably. It is already well defended for this scenario.</p>`;
  } else {
    const max = Math.max(...items.map((x) => x.risk_reduction));
    $("recs").innerHTML = items.map((x, i) => {
      const parts = [];
      if (Math.abs(x.d_breach) >= 0.005) parts.push(`BREACH ${(100 * x.d_breach).toFixed(1)} PP`);
      if (Math.abs(x.d_collapse) >= 0.001) parts.push(`COLLAPSE ${(100 * x.d_collapse).toFixed(1)} PP`);
      if (Math.abs(x.d_recovery_p50) >= 0.5) parts.push(`RECOVERY ${x.d_recovery_p50 > 0 ? "+" : ""}${x.d_recovery_p50.toFixed(1)} D`);
      parts.push(`COST ${String(x.cost).toUpperCase()}`, `${x.effort_weeks} WEEKS`);
      return `<div class="class-row hrow">
        <div class="class-n">${String(i + 1).padStart(2, "0")}</div>
        <div><div class="class-name">${esc(x.title)}</div><div class="class-desc">${esc(x.detail)}</div>
          <div class="class-meta">${parts.map((s) => `<span>${esc(s)}</span>`).join("")}</div></div>
        <div class="class-gain"><b>−${x.risk_reduction.toFixed(1)}</b><small>risk-index points</small>
          <div class="hbar-track"><div class="hbar-fill" style="width:${(100 * x.risk_reduction) / max}%;background:#f0f0f0"></div></div></div>
      </div>`;
    }).join("");
  }
  const planEl = $("plan");
  if (rec.plan && rec.plan.length && rec.plan_result) {
    const a = rec.plan_result;
    planEl.className = "plan";
    planEl.innerHTML = `<div class="plan-k">COMBINED PLAN</div>
      <div class="plan-title">Do these ${rec.plan.length} together: grade ${pr.grade} → ${a.grade}</div>
      <ol>${rec.plan.map((s) => `<li>${esc(s.title)}</li>`).join("")}</ol>
      <div class="plan-after">
        <div>RESILIENCE<b>${pr.resilience_score.toFixed(0)} → ${a.resilience_score.toFixed(0)}</b></div>
        <div>BREACH<b>${pct(pr.p_breach)} → ${pct(a.p_breach)}</b></div>
        <div>COLLAPSE<b>${pct(pr.p_collapse, 1)} → ${pct(a.p_collapse, 1)}</b></div>
        <div>RECOVERY<b>${days(pr.recovery_days.p50)} → ${days(a.recovery_days.p50)}</b></div>
      </div>
      <button class="pr-face active" id="btn-apply-plan" type="button">APPLY PLAN TO THE PROFILE</button>`;
    $("btn-apply-plan").onclick = () => { setProfile(rec.plan_profile); state.modified = true; assess({ manual: true }); showTab("outcomes", true); };
  } else {
    planEl.className = "";
    planEl.innerHTML = "";
  }
}

/* ------------------------------------------------------------- charts ---- */
const baseLayout = (extra = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: G.muted, family: MONO, size: 11 },
  margin: { l: 50, r: 16, t: 10, b: 44 },
  xaxis: { gridcolor: G.line, zerolinecolor: G.line, linecolor: "rgba(255,255,255,.14)", tickfont: { color: G.dim }, ...extra.xaxis },
  yaxis: { gridcolor: G.line, zerolinecolor: G.line, linecolor: "rgba(255,255,255,.14)", tickfont: { color: G.dim }, ...extra.yaxis },
  legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11, color: G.soft } },
  hoverlabel: { bgcolor: "rgba(8,8,8,.96)", bordercolor: "rgba(255,255,255,.22)", font: { color: G.text, family: MONO, size: 12 } },
  ...Object.fromEntries(Object.entries(extra).filter(([k]) => !["xaxis", "yaxis"].includes(k))),
});
const cfg = { displayModeBar: false, responsive: true };
const axisTitle = (text) => ({ text, font: { size: 10.5, color: G.dim } });

function renderCharts() {
  if (typeof Plotly === "undefined") return;
  if (state.tab === "threat" && state.report) renderThreat(state.report.threat, state.report.profile);
  if (state.tab === "simulation" && state.sim) renderSimCharts(state.sim);
}

function renderThreat(t, profile) {
  const s = t.by_swarm;
  const mk = (y, name, color, width, dash) => ({ x: t.swarm_sizes, y, name, mode: "lines+markers", line: { color, width, dash },
    marker: { size: 5, color }, hovertemplate: `${name}: %{y:.1%}<br>%{x:,} agents<extra></extra>` });
  Plotly.react("chart-swarm", [
    mk(s.p_breach, "Breach", G.muted, 1.5, "dot"), mk(s.p_severe, "Severe damage", G.soft, 1.8), mk(s.p_collapse, "Collapse", G.strong, 2.6),
  ], baseLayout({
    xaxis: { type: "log", title: axisTitle("SWARM SIZE (AGENTS)") },
    yaxis: { tickformat: ".0%", range: [0, 1.02] },
    shapes: [{ type: "line", x0: profile.swarm_size, x1: profile.swarm_size, y0: 0, y1: 1, line: { color: "rgba(255,255,255,.35)", dash: "dot", width: 1 } }],
    annotations: [{ x: Math.log10(profile.swarm_size), y: 1, yanchor: "bottom", text: "CURRENT", showarrow: false, font: { size: 9.5, color: G.dim } }],
  }), cfg);

  const m = t.matrix;
  const xs = m.swarm_sizes.map((v) => v.toLocaleString()), ys = m.sophistication.map((v) => v.toFixed(1));
  const notes = [];
  m.risk_index.forEach((row, i) => row.forEach((v, j) => notes.push({
    x: j, y: i, text: v.toFixed(0),           /* category axes: place by index, not by label */ showarrow: false, font: { size: 10.5, family: MONO, color: v > 55 ? G.bg : G.soft },
  })));
  Plotly.react("chart-matrix", [{
    type: "heatmap", z: m.risk_index, x: xs, y: ys, xgap: 2, ygap: 2,
    colorscale: [[0, "#141414"], [0.5, "#5a5a5a"], [1, "#f2f2f2"]], zmin: 0, zmax: 100, showscale: false,
    hovertemplate: "%{x} agents · sophistication %{y}<br>risk index %{z:.1f}<extra></extra>",
  }], baseLayout({
    xaxis: { type: "category", title: axisTitle("SWARM SIZE (AGENTS)"), gridcolor: "rgba(0,0,0,0)" },
    yaxis: { type: "category", title: axisTitle("SOPHISTICATION"), gridcolor: "rgba(0,0,0,0)" },
    margin: { l: 60, r: 10, t: 10, b: 44 }, annotations: notes,
  }), cfg);
}

/* ---------------------------------------------------------- simulation ---- */
async function runSim() {
  const btn = $("btn-sim");
  btn.disabled = true;
  btn.textContent = "SIMULATING…";
  try {
    const sim = await api("/api/simulate", { profile: getProfile(), runs: state.runs, seed: 0 });
    state.sim = sim;
    renderSim(sim);
    showError("");
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "RUN SIMULATION";
  }
}

function clearSim() {
  state.sim = null;
  $("sim-body").hidden = true;
  $("sim-empty").hidden = false;
}

function renderSim(s) {
  $("sim-empty").hidden = true;
  $("sim-body").hidden = false;
  renderSimCharts(s);

  const names = { exploit: "Exploiting public services", phishing: "AI phishing / social eng.", supply_chain: "Supply chain", ai_agent_hijack: "Hijacking company AI agents" };
  const v = Object.entries(s.vector_share).sort((a, b) => b[1] - a[1]);
  $("vector-bars").innerHTML = v.length ? v.map(([k, val]) => hbar(names[k] || k, val, pct(val), null)).join("")
    : `<p class="empty">No breaches in this simulation.</p>`;

  const pr = state.report.prediction;
  const rows = [
    ["P(breach)", pct(pr.p_breach, 1), pct(s.p_breach, 1)],
    ["P(severe infrastructure damage)", pct(pr.p_severe_damage, 1), pct(s.p_severe_damage, 1)],
    ["P(company collapse)", pct(pr.p_collapse, 1), pct(s.p_collapse, 1)],
    ["Swarm detonates if breached", pct(pr.p_detonation_given_breach, 1), s.p_breach > 0 ? pct(s.p_impact_given_breach, 1) : "–"],
    ["Damage if breached", `${pr.expected_damage_given_breach.toFixed(1)}%`, `${s.mean_damage_given_breach.toFixed(1)}%`],
    ["Median recovery if breached", days(pr.recovery_days.p50), s.p_breach > 0 ? days(s.recovery_days.p50) : "–"],
    ["Recovery 90th percentile", days(pr.recovery_days.p90), s.p_breach > 0 ? days(s.recovery_days.p90) : "–"],
    ["Median recovery after detonation", days(pr.recovery_days_if_detonated.p50), s.p_impact_given_breach > 0 ? days(s.recovery_days_if_impact.p50) : "–"],
    ["Median net loss if breached", money(pr.loss_musd.p50), s.p_breach > 0 ? money(s.loss_musd.p50) : "–"],
  ];
  $("compare-body").innerHTML = `<div class="cmp-row head"><span>METRIC</span><span>ML MODEL</span><span>SIMULATION</span></div>` +
    rows.map((r) => `<div class="cmp-row"><span>${r[0]}</span><span>${r[1]}</span><span>${r[2]}</span></div>`).join("");
  $("sim-notes").textContent = `${s.runs.toLocaleString()} campaigns simulated. If breached: detected ${pct(s.p_detected_given_breach)} of the time ` +
    `(median ${s.median_hours_to_detect.toFixed(0)} h after entry), contained before detonation ${pct(s.p_contained_given_breach)}, ` +
    `backups destroyed in ${pct(s.p_backups_lost)} of all campaigns, deepfake fraud in ${pct(s.p_deepfake_fraud)}. ` +
    `Median time to first breach: ${s.median_hours_to_breach.toFixed(0)} h.`;
}

function renderSimCharts(s) {
  if (state.tab !== "simulation" || $("sim-body").hidden) return;
  const t = s.trace;
  const x = t.hours.map((h) => h / 24);
  const line = (y, name, color, dash, width = 2) => ({ x, y, name, mode: "lines", line: { color, width, dash },
    hovertemplate: `${name}: %{y:.1%}<br>day %{x:.1f}<extra></extra>` });
  Plotly.react("chart-timeline", [
    line(t.breached, "Breached", G.muted, "dot"), line(t.detected, "Detected", G.soft, "dash"), line(t.impact, "Detonated", G.strong, "solid", 2.6),
    line(t.contained, "Contained", "#9a9a9a", "solid", 1.3), line(t.compromise, "Avg. systems compromised", "#5f5f5f", "dashdot"),
  ], baseLayout({ xaxis: { title: axisTitle("DAY OF CAMPAIGN") }, yaxis: { tickformat: ".0%", range: [0, 1.02] },
    legend: { orientation: "h", y: 1.2, x: 0, font: { size: 10.5, color: G.soft } }, margin: { l: 44, r: 10, t: 30, b: 40 } }), cfg);

  const rs = s.recovery_samples;
  Plotly.react("chart-recovery", rs.length ? [{
    type: "histogram", x: rs, nbinsx: 40, marker: { color: "rgba(240,240,240,.7)", line: { color: G.bg, width: 1 } },
    hovertemplate: "%{x} days<br>%{y} campaigns<extra></extra>",
  }] : [], baseLayout({ xaxis: { title: axisTitle("DAYS TO RECOVER") },
    yaxis: { title: axisTitle("CAMPAIGNS") }, bargap: 0.04, showlegend: false, margin: { l: 50, r: 10, t: 6, b: 40 },
    annotations: rs.length ? [] : [{ text: "NO BREACHES IN THIS SIMULATION", showarrow: false, x: 0.5, y: 0.5, xref: "paper", yref: "paper", font: { color: G.dim } }] }), cfg);
}

/* ------------------------------------------------------------- actions ---- */
/* "Generating result" feedback: the Assess button turns busy and a status toast appears.
   Live slider updates only show it if the answer takes longer than a blink, to avoid flicker. */
const gen = { timer: null, hideTimer: null, shown: false };
function setGenerating(on, { immediate = false, failed = false } = {}) {
  clearTimeout(gen.timer);
  clearTimeout(gen.hideTimer);
  const btn = $("btn-assess"), toast = $("gen-status"), text = $("gen-text");
  if (on) {
    const show = () => {
      gen.shown = true;
      btn.classList.add("busy");
      btn.disabled = true;
      btn.textContent = "GENERATING…";
      toast.classList.remove("done");
      text.textContent = "GENERATING RESULT…";
      toast.hidden = false;
    };
    immediate ? show() : (gen.timer = setTimeout(show, 250));
    return;
  }
  btn.classList.remove("busy");
  btn.disabled = false;
  btn.textContent = "ASSESS";
  if (!gen.shown) return;
  gen.shown = false;
  if (failed) { toast.hidden = true; return; }
  toast.classList.add("done");
  text.textContent = "RESULT READY";
  gen.hideTimer = setTimeout(() => { toast.hidden = true; }, 1400);
}

function pressEffect(btn) {
  btn.classList.remove("pressed");
  void btn.offsetWidth;                            // restart the animation on every click
  btn.classList.add("pressed");
}

async function assess({ manual = false } = {}) {
  const seq = ++state.seq;
  const started = performance.now();
  document.body.classList.add("loading");
  setGenerating(true, { immediate: manual });
  let failed = false;
  try {
    const rep = await api("/api/predict", getProfile());
    if (seq !== state.seq) return;                 // a newer request is in flight
    // After a click, keep "generating" on screen long enough to be read.
    const wait = manual ? 600 - (performance.now() - started) : 0;
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    if (seq !== state.seq) return;
    state.report = rep;
    showError("");
    if (state.sim) clearSim();
    render(rep);
  } catch (e) {
    failed = true;
    if (seq === state.seq) showError(e.message);
  } finally {
    if (seq === state.seq) {
      document.body.classList.remove("loading");
      setGenerating(false, { failed });
    }
  }
}

function exportReport() {
  if (!state.report) return;
  const blob = new Blob([JSON.stringify({ ...state.report, simulation: state.sim || undefined }, null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "swarmrisk-report.json" });
  a.click();
  URL.revokeObjectURL(a.href);
}

async function loadModelCard() {
  try {
    const m = await api("/api/model");
    const c = m.metrics.classifiers, f = m.metrics.fidelity, r = m.metrics.regressors;
    $("model-badge").textContent = `${(m.n_samples / 1000).toFixed(0)}K SIMULATED CAMPAIGNS · V${m.version}`;
    $("model-trained").textContent = `trained ${m.trained_at}`;
    $("footer-meta").textContent = `MODEL V${m.version} · ${(m.n_samples / 1000).toFixed(0)}K CAMPAIGNS`;
    const cell = (k, v) => `<div class="metric"><div class="metric-k">${k}</div><div class="metric-v">${v}</div></div>`;
    $("model-card").innerHTML = [
      cell("BREACH AUC", c.breach.auc.toFixed(3)),
      cell("SEVERE-DAMAGE AUC", c.severe.auc.toFixed(3)),
      cell("COLLAPSE AUC", c.collapse.auc.toFixed(3)),
      f ? cell("P(BREACH) ERROR VS SIM.", `±${(100 * f.p_breach.mae).toFixed(1)}pp`) : "",
      f ? cell("P(COLLAPSE) ERROR VS SIM.", `±${(100 * f.p_collapse.mae).toFixed(1)}pp`) : "",
      f ? cell("MEDIAN RECOVERY ERROR", `±${f.recovery_q50.mae.toFixed(1)}d`) : cell("DAMAGE R²", r.damage.r2.toFixed(2)),
    ].join("");
  } catch (e) {
    $("model-badge").textContent = "MODEL NOT TRAINED";
  }
}

async function init() {
  initTabs();
  try {
    const [schema, presets] = await Promise.all([api("/api/schema"), api("/api/presets")]);
    state.schema = schema;
    state.presets = presets;
    buildForm(schema);
    const sel = $("preset");
    sel.innerHTML = presets.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("") + `<option value="-1">Default company</option>`;
    const choose = () => {
      const i = Number(sel.value);
      const preset = i >= 0 ? presets[i] : { name: "Default company", profile: Object.fromEntries(schema.features.map((f) => [f.name, f.default])), description: "Schema defaults." };
      state.preset = preset;
      state.modified = false;
      $("preset-desc").textContent = preset.description || "";
      setProfile(preset.profile);
      assess();
    };
    sel.onchange = choose;
    $("btn-reset").onclick = choose;
    $("btn-assess").onclick = (e) => { pressEffect(e.currentTarget); assess({ manual: true }); };
    $("btn-sim").onclick = runSim;
    $("btn-export").onclick = exportReport;
    document.querySelectorAll("#sim-runs .pr-face").forEach((b) => {
      b.onclick = () => {
        state.runs = Number(b.dataset.runs);
        document.querySelectorAll("#sim-runs .pr-face").forEach((x) => x.classList.toggle("active", x === b));
      };
    });
    choose();
    loadModelCard();
  } catch (e) {
    showError(`Could not start: ${e.message}`);
  }
}

init();
