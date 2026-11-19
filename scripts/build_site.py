"""Build the static GitHub Pages site into ``_site/``.

The site is the same dashboard as ``python -m swarmrisk serve``, but with no server: the Python
package and the trained model run in the visitor's browser through Pyodide (see ``web/``).

    python scripts/build_site.py [--out _site]

Only the standard library is needed. The model in ``models/`` must have been trained with the
scikit-learn version that Pyodide ships (pinned in requirements.txt), or it cannot be unpickled
in the browser.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarmrisk import __version__  # noqa: E402  (stdlib-only imports)
from swarmrisk.schema import schema_json, validate_profile  # noqa: E402

PKG = ROOT / "swarmrisk"
STATIC = PKG / "static"
WEB = ROOT / "web"
MODELS = ROOT / "models"
EXAMPLES = ROOT / "examples"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
REPO_URL = "https://github.com/SomaxSoma/swarmrisk"
# Modules the browser needs (not the API server or the CLI).
BROWSER_MODULES = ["__init__.py", "schema.py", "sampling.py", "simulator.py", "model.py", "predictor.py"]
PRESET_ORDER = ["regional_hospital", "online_retailer", "ai_startup", "city_government", "power_utility", "global_bank"]


def presets() -> list[dict]:
    files = sorted(EXAMPLES.glob("*.json"),
                   key=lambda f: (PRESET_ORDER.index(f.stem) if f.stem in PRESET_ORDER else len(PRESET_ORDER), f.stem))
    out = []
    for f in files:
        data = json.loads(f.read_text())
        out.append({"id": f.stem, "name": data.pop("_name", f.stem), "description": data.pop("_description", ""),
                    "profile": validate_profile(data)})
    return out


def index_html() -> str:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    replacements = {
        'href="/static/style.css?v=2">': 'href="static/style.css?v=2">\n  <link rel="stylesheet" href="static/site.css?v=2">',
        '<script src="/vendor/plotly.min.js"></script>': f'<script src="{PLOTLY_CDN}"></script>',
        '<script src="/static/app.js?v=2"></script>':
            '<script src="static/backend.js?v=2"></script>\n<script src="static/app.js?v=2"></script>',
        '<span id="engine-label">LIVE MODEL</span>': '<span id="engine-label">RUNS IN YOUR BROWSER</span>',
        'actuarial forecast. <a href="#model">Method and limits</a>.':
            'actuarial forecast. This page runs the trained model in your browser with Pyodide, so nothing you enter leaves '
            f'your machine. <a href="#model">Method and limits</a> · <a href="{REPO_URL}">source, REST API and CLI</a>.',
    }
    for old, new in replacements.items():
        if old not in html:
            raise SystemExit(f"index.html no longer contains {old!r}; update scripts/build_site.py")
        html = html.replace(old, new)
    return html


def build(out: Path) -> None:
    model = MODELS / "swarmrisk_model.joblib"
    metrics = MODELS / "metrics.json"
    if not model.exists() or not metrics.exists():
        raise SystemExit("models/ has no trained model. Run: python -m swarmrisk train")
    if out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True)
    (out / "api").mkdir()
    (out / "model").mkdir()

    (out / "index.html").write_text(index_html(), encoding="utf-8")
    for f in ["style.css", "app.js"]:
        shutil.copy2(STATIC / f, out / "static" / f)
    for f in ["backend.js", "worker.js", "site.css"]:
        shutil.copy2(WEB / f, out / "static" / f)

    (out / "api" / "schema.json").write_text(json.dumps(schema_json()))
    (out / "api" / "presets.json").write_text(json.dumps(presets()))
    shutil.copy2(metrics, out / "api" / "model.json")
    shutil.copy2(model, out / "model" / model.name)

    with zipfile.ZipFile(out / "swarmrisk.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in BROWSER_MODULES:
            z.write(PKG / name, f"swarmrisk/{name}")
    (out / ".nojekyll").write_text("")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"SwarmRisk {__version__}: built {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    build(ap.parse_args().out)
