/* SwarmRisk dashboard */
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {
  text: css("--text"), muted: css("--muted"), border: css("--border"), panel: css("--panel"),
  ok: css("--ok"), warn: css("--warn"), high: css("--high"), crit: css("--crit"), blue: css("--accent-2"),
};
const LEVEL_COLOR = { low: C.ok, moderate: C.warn, high: C.high, critical: C.crit };
const GRADE_COLOR = { A: C.ok, B: "#7cc46a", C: C.warn, D: C.high, F: C.crit };

const state = { schema: null, presets: [], preset: null, report: null, sim: null, seq: 0 };

const pct = (p, d = 0) => `${(100 * p).toFixed(d)}%`;
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
  Object.entries(groups).forEach(([group, feats], gi) => {
    const det = document.createElement("details");
    det.className = "group";
    det.open = gi === 0 || group === "Threat scenario" || group === "Defences";
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
  const help = `<span class="help" title="${esc(f.help)}">?</span>`;
  const id = `f-${f.name}`;
  if (f.kind === "category") {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label></div>
      <select id="${id}" data-name="${f.name}">${schema.industries.map((i) => `<option value="${i}">${i[0].toUpperCase() + i.slice(1)}</option>`).join("")}</select>`;
  } else if (f.kind === "bool") {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label>
      <label class="switch"><input type="checkbox" id="${id}" data-name="${f.name}"><span></span></label></div>`;
  } else if (isFraction(f)) {
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label><span class="pct" id="${id}-out"></span></div>
      <input type="range" id="${id}" data-name="${f.name}" min="0" max="1" step="0.01">`;
  } else {
    const lo = toPos(f, f.low), hi = toPos(f, f.high);
    const step = f.log_scale ? 0.001 : f.kind === "int" ? 1 : (f.high - f.low) / 400;
    div.innerHTML = `<div class="field-head"><label for="${id}">${esc(f.label)}${help}</label>
      <input type="number" id="${id}" data-name="${f.name}" min="${f.low}" max="${f.high}" step="${f.kind === "int" ? 1 : "any"}"></div>
      <input type="range" id="${id}-r" data-slider-for="${f.name}" min="${lo}" max="${hi}" step="${step}">`;
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
  if ($("auto").checked) {
    clearTimeout(debounce);
    debounce = setTimeout(assess, 350);
  }
}

/* -------------------------------------------------------------- render ---- */
function gauge(el, p, lvl) {
  const color = LEVEL_COLOR[lvl] || C.blue;
  const r = 52, cx = 70, cy = 66, a0 = Math.PI, a1 = Math.PI * (1 - Math.min(1, Math.max(0, p)));
  const pt = (a) => [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  const [x0, y0] = pt(a0), [x1, y1] = pt(a1), [xe, ye] = pt(0);
  el.innerHTML = `<svg viewBox="0 0 140 84">
    <path d="M${x0},${y0} A${r},${r} 0 0 1 ${xe},${ye}" stroke="${C.border}" stroke-width="12" fill="none" stroke-linecap="round"/>
    ${p > 0.002 ? `<path d="M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}" stroke="${color}" stroke-width="12" fill="none" stroke-linecap="round"/>` : ""}
    <text x="70" y="62" text-anchor="middle" fill="${C.text}" font-size="22" font-weight="700">${pct(p, p < 0.1 ? 1 : 0)}</text>
    <text x="70" y="80" text-anchor="middle" fill="${color}" font-size="11" font-weight="600">${lvl.toUpperCase()}</text></svg>`;
}

const baseLayout = (extra = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: C.text, family: css("--font"), size: 12 },
  margin: { l: 50, r: 16, t: 10, b: 44 },
  xaxis: { gridcolor: C.border, zerolinecolor: C.border, ...extra.xaxis },
  yaxis: { gridcolor: C.border, zerolinecolor: C.border, ...extra.yaxis },
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
  hoverlabel: { bgcolor: C.panel, bordercolor: C.border, font: { color: C.text } },
  ...Object.fromEntries(Object.entries(extra).filter(([k]) => !["xaxis", "yaxis"].includes(k))),
});
const cfg = { displayModeBar: false, responsive: true };

function render(rep) {
  const pr = rep.prediction;
  $("grade").textContent = pr.grade;
  $("grade").style.color = GRADE_COLOR[pr.grade];
  $("score").textContent = pr.resilience_score.toFixed(0);
  gauge($("g-breach"), pr.p_breach, pr.levels.breach);
  gauge($("g-severe"), pr.p_severe_damage, pr.levels.severe_damage);
  gauge($("g-collapse"), pr.p_collapse, pr.levels.collapse);
  const r = pr.recovery_days;
  $("rec-p50").textContent = days(r.p50);
  $("rec-range").textContent = `80% range: ${days(r.p10)} – ${days(r.p90)}`;
  const maxD = Math.max(r.p90 * 1.15, 30);
  Object.assign($("rec-bar").style, { left: `${(100 * r.p10) / maxD}%`, width: `${Math.max(2, (100 * (r.p90 - r.p10)) / maxD)}%` });
  $("rec-imp").textContent = days(pr.recovery_days_if_detonated.p50);
  $("rec-con").textContent = days(pr.recovery_days_if_contained.p50);
  $("deton").textContent = pct(pr.p_detonation_given_breach);
  $("deton").style.color = LEVEL_COLOR[pr.p_detonation_given_breach < 0.1 ? "low" : pr.p_detonation_given_breach < 0.3 ? "moderate" : pr.p_detonation_given_breach < 0.6 ? "high" : "critical"];
  $("dmg").textContent = `${pr.expected_damage_given_breach.toFixed(1)}%`;
  $("loss50").textContent = money(pr.loss_musd.p50);
  $("loss90").textContent = money(pr.loss_musd.p90);
  $("summary").textContent = pr.summary;

  renderThreat(rep.threat, rep.profile);
  renderDrivers(rep.drivers);
  renderRecs(rep.recommendations, pr);
}

function renderThreat(t, profile) {
  const s = t.by_swarm;
  const mk = (y, name, color) => ({ x: t.swarm_sizes, y, name, mode: "lines+markers", line: { color, width: 2.5 }, marker: { size: 5 },
    hovertemplate: `${name}: %{y:.1%}<br>%{x:,} agents<extra></extra>` });
  Plotly.react("chart-swarm", [
    mk(s.p_breach, "Breach", C.blue), mk(s.p_severe, "Severe damage", C.high), mk(s.p_collapse, "Collapse", C.crit),
  ], baseLayout({
    xaxis: { type: "log", title: { text: "Swarm size (agents)", font: { size: 11 } } },
    yaxis: { tickformat: ".0%", range: [0, 1.02] },
    shapes: [{ type: "line", x0: profile.swarm_size, x1: profile.swarm_size, y0: 0, y1: 1, line: { color: C.muted, dash: "dot", width: 1 } }],
    annotations: [{ x: Math.log10(profile.swarm_size), y: 1, yanchor: "bottom", text: "current", showarrow: false, font: { size: 10, color: C.muted } }],
  }), cfg);

  const m = t.matrix;
  Plotly.react("chart-matrix", [{
    type: "heatmap", z: m.risk_index, x: m.swarm_sizes.map(String), y: m.sophistication.map((v) => v.toFixed(1)),
    colorscale: [[0, C.ok], [0.35, C.warn], [0.65, C.high], [1, C.crit]], zmin: 0, zmax: 100,
    text: m.risk_index.map((row) => row.map((v) => v.toFixed(0))), texttemplate: "%{text}", textfont: { size: 11, color: "#111" },
    hovertemplate: "%{x} agents, sophistication %{y}<br>risk index %{z:.1f}<extra></extra>",
    colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10 } },
  }], baseLayout({
    xaxis: { type: "category", title: { text: "Swarm size (agents)", font: { size: 11 } } },
    yaxis: { type: "category", title: { text: "Sophistication", font: { size: 11 } } },
    margin: { l: 60, r: 10, t: 10, b: 44 },
  }), cfg);
}

function fmtVal(v) {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
  return String(v);
}

function renderDrivers(drivers) {
  const d = [...drivers].reverse();
  Plotly.react("chart-drivers", [{
    type: "bar", orientation: "h", y: d.map((x) => x.label), x: d.map((x) => x.impact),
    marker: { color: d.map((x) => (x.impact > 0 ? C.crit : C.ok)) },
    customdata: d.map((x) => [fmtVal(x.value), fmtVal(x.typical), (100 * x.d_breach).toFixed(1), (100 * x.d_collapse).toFixed(2)]),
    hovertemplate: "<b>%{y}</b><br>this company: %{customdata[0]} · typical: %{customdata[1]}<br>" +
      "risk index %{x:+.1f} pts<br>breach %{customdata[2]} pp · collapse %{customdata[3]} pp<extra></extra>",
  }], baseLayout({
    margin: { l: 170, r: 16, t: 10, b: 40 },
    xaxis: { title: { text: "Risk-index points vs typical company", font: { size: 11 } }, zeroline: true, zerolinecolor: C.muted },
    yaxis: { automargin: true },
    showlegend: false,
  }), cfg);
}

function renderRecs(rec, pr) {
  const items = rec.actions;
  if (!items.length) {
    $("recs").innerHTML = `<p class="muted">No single change in our catalogue lowers this company's risk noticeably. It is already well defended for this scenario.</p>`;
  } else {
    const max = Math.max(...items.map((x) => x.risk_reduction));
    $("recs").innerHTML = items.map((x) => {
      const parts = [];
      if (Math.abs(x.d_breach) >= 0.005) parts.push(`breach <b>${(100 * x.d_breach).toFixed(1)} pp</b>`);
      if (Math.abs(x.d_collapse) >= 0.001) parts.push(`collapse <b>${(100 * x.d_collapse).toFixed(1)} pp</b>`);
      if (Math.abs(x.d_recovery_p50) >= 0.5) parts.push(`recovery <b>${x.d_recovery_p50 > 0 ? "+" : ""}${x.d_recovery_p50.toFixed(1)} d</b>`);
      return `<div class="rec">
        <div class="t">${esc(x.title)}</div>
        <div class="d">${esc(x.detail)}</div>
        <div class="meta">${parts.join(" · ") || "small effect"} · cost ${x.cost} · ${x.effort_weeks} weeks</div>
        <div class="bar"><span class="v">−${x.risk_reduction.toFixed(1)}</span> <span class="muted small">pts</span>
          <div class="track"><div class="fill" style="width:${(100 * x.risk_reduction) / max}%"></div></div></div>
      </div>`;
    }).join("");
  }
  const planEl = $("plan");
  if (rec.plan && rec.plan.length && rec.plan_result) {
    const a = rec.plan_result;
    planEl.innerHTML = `<b>Combined plan: do these ${rec.plan.length} together</b>
      <ol>${rec.plan.map((s) => `<li>${esc(s.title)}</li>`).join("")}</ol>
      <div class="after">
        <span>Grade <b style="color:${GRADE_COLOR[pr.grade]}">${pr.grade}</b> → <b style="color:${GRADE_COLOR[a.grade]}">${a.grade}</b></span>
        <span>Breach <b>${pct(pr.p_breach)}</b> → <b>${pct(a.p_breach)}</b></span>
        <span>Collapse <b>${pct(pr.p_collapse, 1)}</b> → <b>${pct(a.p_collapse, 1)}</b></span>
        <span>Recovery <b>${days(pr.recovery_days.p50)}</b> → <b>${days(a.recovery_days.p50)}</b></span>
      </div>
      <button class="btn" id="btn-apply-plan" type="button">Apply plan to the form</button>`;
    planEl.classList.add("show");
    $("btn-apply-plan").onclick = () => { setProfile(rec.plan_profile); assess(); };
  } else {
    planEl.classList.remove("show");
    planEl.innerHTML = "";
  }
}

/* ------------------------------------------------------------ actions ---- */
async function assess() {
  const seq = ++state.seq;
  document.body.classList.add("loading");
  try {
    const rep = await api("/api/predict", getProfile());
    if (seq !== state.seq) return;                 // a newer request is in flight
    state.report = rep;
    showError("");
    render(rep);
    if (state.sim) { state.sim = null; $("sim-body").hidden = true; $("sim-empty").hidden = false; }
  } catch (e) {
    if (seq === state.seq) showError(e.message);
  } finally {
    if (seq === state.seq) document.body.classList.remove("loading");
  }
}

async function runSim() {
  const btn = $("btn-sim");
  btn.disabled = true;
  btn.textContent = "Simulating…";
  try {
    const runs = Number($("sim-runs").value);
    const sim = await api("/api/simulate", { profile: getProfile(), runs, seed: 0 });
    state.sim = sim;
    renderSim(sim);
    showError("");
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run simulation";
  }
}

function renderSim(s) {
  $("sim-empty").hidden = true;
  $("sim-body").hidden = false;
  const t = s.trace;
  const x = t.hours.map((h) => h / 24);
  const line = (y, name, color, dash) => ({ x, y, name, mode: "lines", line: { color, width: 2, dash },
    hovertemplate: `${name}: %{y:.1%}<br>day %{x:.1f}<extra></extra>` });
  Plotly.react("chart-timeline", [
    line(t.breached, "Breached", C.blue), line(t.detected, "Detected", C.ok), line(t.impact, "Detonated", C.crit),
    line(t.contained, "Contained", "#7cc46a", "dot"), line(t.compromise, "Avg. systems compromised", C.warn, "dash"),
  ], baseLayout({ xaxis: { title: { text: "Day of campaign", font: { size: 11 } } }, yaxis: { tickformat: ".0%", range: [0, 1.02] },
    legend: { orientation: "h", y: 1.22, x: 0, font: { size: 10 } }, margin: { l: 44, r: 10, t: 30, b: 40 } }), cfg);

  const rs = s.recovery_samples;
  Plotly.react("chart-recovery", rs.length ? [{
    type: "histogram", x: rs, nbinsx: 40, marker: { color: C.high, line: { color: C.panel, width: 1 } },
    hovertemplate: "%{x} days<br>%{y} campaigns<extra></extra>",
  }] : [], baseLayout({ xaxis: { title: { text: "Days to recover", font: { size: 11 } } },
    yaxis: { title: { text: "Campaigns", font: { size: 11 } } }, bargap: 0.02, showlegend: false,
    annotations: rs.length ? [] : [{ text: "No breaches in this simulation", showarrow: false, x: 0.5, y: 0.5, xref: "paper", yref: "paper", font: { color: C.muted } }] }), cfg);

  const names = { exploit: "Exploiting public services", phishing: "AI phishing / social eng.", supply_chain: "Supply chain", ai_agent_hijack: "Hijacking company AI agents" };
  const v = Object.entries(s.vector_share).sort((a, b) => a[1] - b[1]);
  Plotly.react("chart-vectors", [{
    type: "bar", orientation: "h", y: v.map(([k]) => names[k] || k), x: v.map(([, val]) => val),
    marker: { color: [C.blue, C.warn, C.high, C.crit] }, hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
  }], baseLayout({ xaxis: { tickformat: ".0%" }, yaxis: { automargin: true }, margin: { l: 10, r: 10, t: 10, b: 30 }, showlegend: false }), cfg);

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
  $("compare-body").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("");
  $("sim-notes").textContent = `${s.runs.toLocaleString()} campaigns simulated. If breached: detected ${pct(s.p_detected_given_breach)} of the time ` +
    `(median ${s.median_hours_to_detect.toFixed(0)} h after entry), contained before detonation ${pct(s.p_contained_given_breach)}, ` +
    `backups destroyed in ${pct(s.p_backups_lost)} of all campaigns, deepfake fraud in ${pct(s.p_deepfake_fraud)}. ` +
    `Median time to first breach: ${s.median_hours_to_breach.toFixed(0)} h.`;
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
    const c = m.metrics.classifiers, f = m.metrics.fidelity;
    $("model-badge").textContent = `model v${m.version} · ${(m.n_samples / 1000).toFixed(0)}k simulated campaigns`;
    const cell = (label, val) => `<div><span class="muted small">${label}</span><b>${val}</b></div>`;
    $("model-card").innerHTML = `<h2 style="margin-bottom:8px">Model card</h2><div class="metrics">
      ${cell("Breach AUC", c.breach.auc.toFixed(3))}${cell("Severe-damage AUC", c.severe.auc.toFixed(3))}${cell("Collapse AUC", c.collapse.auc.toFixed(3))}
      ${f ? cell("P(breach) error vs simulation", `±${(100 * f.p_breach.mae).toFixed(1)} pp`) : ""}
      ${f ? cell("P(collapse) error vs simulation", `±${(100 * f.p_collapse.mae).toFixed(1)} pp`) : ""}
      ${cell("Trained", m.trained_at)}</div>`;
  } catch (e) {
    $("model-badge").textContent = "model not trained";
  }
}

async function init() {
  try {
    const [schema, presets] = await Promise.all([api("/api/schema"), api("/api/presets")]);
    state.schema = schema;
    state.presets = presets;
    buildForm(schema);
    const sel = $("preset");
    sel.innerHTML = presets.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("") + `<option value="-1">Default company</option>`;
    const choose = () => {
      const i = Number(sel.value);
      const preset = i >= 0 ? presets[i] : { profile: Object.fromEntries(schema.features.map((f) => [f.name, f.default])), description: "Schema defaults." };
      state.preset = preset;
      $("preset-desc").textContent = preset.description || "";
      setProfile(preset.profile);
      assess();
    };
    sel.onchange = choose;
    $("btn-reset").onclick = choose;
    $("btn-assess").onclick = assess;
    $("btn-sim").onclick = runSim;
    $("btn-export").onclick = exportReport;
    choose();
    loadModelCard();
  } catch (e) {
    showError(`Could not start: ${e.message}`);
  }
}

init();
