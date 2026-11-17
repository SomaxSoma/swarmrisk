/* Runs the SwarmRisk Python package + trained model in Pyodide, off the main thread. */
"use strict";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;
let handle = null;

const status = (text) => postMessage({ type: "status", text });

// Pyodide can fail a download inside a promise it never surfaces (e.g. a dropped CDN connection).
self.onunhandledrejection = (ev) => {
  if (!handle) postMessage({ type: "fatal", error: String(ev.reason?.message || ev.reason || "download failed") });
};

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${label} (HTTP ${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

async function boot(base) {
  status("Downloading the Python runtime (first visit only, ~30 MB)…");
  py = await loadPyodide({ indexURL: PYODIDE });
  status("Loading numpy, pandas, SciPy and scikit-learn…");
  const [, code, model] = await Promise.all([
    py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn", "joblib"]),
    fetchBytes(new URL("swarmrisk.zip", base), "the SwarmRisk code"),
    fetchBytes(new URL("model/swarmrisk_model.joblib", base), "the trained model"),
  ]);
  py.unpackArchive(code, "zip", { extractDir: "/home/pyodide/app" });
  py.FS.writeFile("/home/pyodide/model.joblib", model);
  status("Starting the model…");
  await py.runPythonAsync(`
import json, sys
sys.path.insert(0, "/home/pyodide/app")
import numpy as np
from swarmrisk.model import load_bundle
from swarmrisk.predictor import Predictor
from swarmrisk.schema import ProfileError, validate_profile

_P = Predictor(load_bundle("/home/pyodide/model.joblib"))

def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)

def _clean(p):
    if not isinstance(p, dict):
        raise ProfileError("Body must be a JSON object of company fields.")
    return validate_profile({k: v for k, v in p.items() if not str(k).startswith("_")})

def handle(path, body_json):
    body = json.loads(body_json)
    try:
        if path == "/api/predict":
            out = _P.report(_clean(body))
        elif path == "/api/simulate":
            seed = body.get("seed", 0)
            out = _P.simulate(_clean(body.get("profile", {})), runs=int(body.get("runs", 1000)),
                              seed=None if seed is None else int(seed))
        else:
            return json.dumps({"error": "unknown endpoint " + path})
    except (ProfileError, ValueError, TypeError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"result": out}, default=_default)

handle("/api/predict", "{}")   # warm-up: the first call compiles code paths
`);
  handle = py.globals.get("handle");
  postMessage({ type: "ready" });
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === "boot") {
    try { await boot(m.base); } catch (e) { postMessage({ type: "fatal", error: String(e.message || e) }); }
    return;
  }
  if (m.type === "call") {
    try {
      const out = JSON.parse(handle(m.path, m.body));
      postMessage({ id: m.id, result: out.result, error: out.error });
    } catch (e) {
      postMessage({ id: m.id, error: String(e.message || e).split("\n").slice(-2).join(" ") });
    }
  }
};
