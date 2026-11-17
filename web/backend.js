/* SwarmRisk in-browser backend for the static (GitHub Pages) build.
 *
 * GET endpoints are served from pre-built JSON files, so the form appears at once.
 * POST endpoints (/api/predict, /api/simulate) run the real Python package and trained
 * model inside a Web Worker with Pyodide (CPython compiled to WebAssembly).
 */
"use strict";
(() => {
  const BASE = new URL(".", document.baseURI).href;
  const STATIC_GET = { "/api/schema": "api/schema.json", "/api/presets": "api/presets.json", "/api/model": "api/model.json" };

  /* ---- status banner ---- */
  const banner = document.createElement("div");
  banner.id = "py-status";
  banner.innerHTML = '<span class="spinner"></span><span id="py-status-text">Starting the in-browser model…</span>';
  const mount = () => document.body.prepend(banner);
  document.body ? mount() : document.addEventListener("DOMContentLoaded", mount);
  const setStatus = (text, kind) => {
    banner.className = kind || "";
    banner.querySelector("#py-status-text").textContent = text;
    banner.hidden = kind === "ready";
  };

  /* ---- worker ---- */
  const worker = new Worker(new URL("static/worker.js", BASE));
  let nextId = 1;
  const waiting = new Map();         // id -> {resolve, reject}
  let busy = false;
  let readyResolve, readyReject;
  const ready = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
  ready.catch(() => {});   // callers see the error; avoid an extra "uncaught" log
  const queue = [];                  // requests not yet sent

  let started = false;
  const fail = (msg) => {
    if (started) return;
    started = true;
    clearTimeout(stallTimer);
    setStatus(`The in-browser model failed to start: ${msg}. Reload the page to retry, or run SwarmRisk locally.`, "error");
    readyReject(new Error(msg));
  };
  // A stalled download should become a visible error, not an endless spinner.
  let stallTimer;
  const armStall = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => fail("loading stalled for 3 minutes (slow or interrupted connection?)"), 180_000);
  };
  armStall();

  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "status") { armStall(); return setStatus(m.text); }
    if (m.type === "ready") {
      if (started) return;
      started = true;
      clearTimeout(stallTimer);
      setStatus("", "ready");
      readyResolve();
      return;
    }
    if (m.type === "fatal") return fail(m.error);
    const w = waiting.get(m.id);
    if (!w) return;
    waiting.delete(m.id);
    busy = false;
    m.error ? w.reject(new Error(m.error)) : w.resolve(m.result);
    pump();
  };
  worker.onerror = (e) => {
    if (!started) return fail(e.message || "the worker crashed");
    setStatus(`The in-browser model crashed: ${e.message || "unknown error"}. Reload the page.`, "error");
  };
  worker.postMessage({ type: "boot", base: BASE });

  function pump() {
    if (busy || !queue.length) return;
    const job = queue.shift();
    busy = true;
    waiting.set(job.id, job);
    worker.postMessage({ type: "call", id: job.id, path: job.path, body: JSON.stringify(job.body ?? {}) });
  }

  async function call(path, body) {
    await ready;
    return new Promise((resolve, reject) => {
      // Live sliders fire many predictions: only the newest queued one is worth computing.
      if (path === "/api/predict") {
        for (let i = queue.length - 1; i >= 0; i--) {
          if (queue[i].path === path) queue.splice(i, 1)[0].reject(new Error("superseded"));
        }
      }
      queue.push({ id: nextId++, path, body, resolve, reject });
      pump();
    });
  }

  window.SWARMRISK_BACKEND = async (path, body) => {
    if (body === undefined && STATIC_GET[path]) {
      const res = await fetch(new URL(STATIC_GET[path], BASE));
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
      return res.json();
    }
    if (body === undefined) throw new Error(`${path} is not available in the static build`);
    return call(path, body);
  };
})();
