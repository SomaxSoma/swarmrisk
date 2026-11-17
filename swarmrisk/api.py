"""FastAPI app: REST API + the web dashboard.

Run with ``python -m swarmrisk serve`` and open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .model import METRICS_PATH
from .predictor import Predictor
from .schema import ProfileError, schema_json, validate_profile

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
EXAMPLES = HERE.parent / "examples"

@asynccontextmanager
async def _lifespan(_app):
    if os.environ.get("SWARMRISK_PRELOAD", "1") == "1":
        try:
            predictor()          # load the model once at start-up
        except FileNotFoundError:
            pass
    yield


app = FastAPI(title="SwarmRisk API", version=__version__, lifespan=_lifespan,
              description="Predict breach, infrastructure damage, collapse and recovery time "
                          "for a company attacked by an autonomous AI-agent swarm.")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor()


def _plotly_path() -> Path | None:
    try:
        import plotly
        p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        return p if p.exists() else None
    except ImportError:
        return None


def _clean(profile: Any) -> dict:
    if not isinstance(profile, dict):
        raise HTTPException(422, "Body must be a JSON object of company fields.")
    profile = {k: v for k, v in profile.items() if not k.startswith("_")}
    try:
        return validate_profile(profile)
    except ProfileError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/vendor/plotly.min.js", include_in_schema=False)
def plotly_js():
    p = _plotly_path()
    if p is None:
        raise HTTPException(404, "plotly is not installed")
    return FileResponse(p, media_type="application/javascript")


@app.get("/api/health")
def health():
    try:
        predictor()
        return {"status": "ok", "version": __version__}
    except FileNotFoundError as e:
        return JSONResponse({"status": "no-model", "detail": str(e)}, status_code=503)


@app.get("/api/schema")
def schema():
    return schema_json()


@app.get("/api/presets")
def presets():
    order = ["regional_hospital", "online_retailer", "ai_startup", "city_government", "power_utility", "global_bank"]
    files = sorted(EXAMPLES.glob("*.json"), key=lambda f: (order.index(f.stem) if f.stem in order else len(order), f.stem))
    out = []
    for f in files:
        data = json.loads(f.read_text())
        out.append({"id": f.stem, "name": data.pop("_name", f.stem), "description": data.pop("_description", ""),
                    "profile": validate_profile(data)})
    return out


@app.get("/api/model")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(503, "Model not trained yet. Run: python -m swarmrisk train")
    return json.loads(METRICS_PATH.read_text())


@app.post("/api/predict")
def predict(profile: dict = Body(...)):
    """Full assessment: probabilities, drivers, recommendations and threat curves."""
    return predictor().report(_clean(profile))


@app.post("/api/simulate")
def simulate(payload: dict = Body(...)):
    """Run the agent-based simulator directly: ``{"profile": {...}, "runs": 1000, "seed": 0}``."""
    prof = _clean(payload.get("profile", {}))
    try:
        runs = int(payload.get("runs", 1000))
        seed = payload.get("seed", 0)
        seed = None if seed is None else int(seed)
    except (TypeError, ValueError):
        raise HTTPException(422, "'runs' and 'seed' must be integers") from None
    return predictor().simulate(prof, runs=runs, seed=seed)


@app.post("/api/batch")
def batch(profiles: list = Body(...)):
    """Score many companies at once (no explanations). Max 10,000 profiles."""
    if len(profiles) > 10_000:
        raise HTTPException(413, "At most 10,000 profiles per request.")
    clean = []
    for i, p in enumerate(profiles):
        try:
            clean.append(_clean(p))
        except HTTPException as e:
            raise HTTPException(422, f"profile #{i}: {e.detail}") from None
    pr = predictor()
    return [pr.predict(p) for p in clean] if len(clean) <= 50 else _fast_batch(pr, clean)


def _fast_batch(pr: Predictor, clean: list[dict]) -> list[dict]:
    from .predictor import grade
    df = pr._predict_many(clean)
    return [{"p_breach": r.p_breach, "p_severe_damage": r.p_severe, "p_collapse": r.p_collapse,
             "expected_damage_given_breach": r.damage_given_breach,
             "recovery_days": {"p10": r.recovery_q10, "p50": r.recovery_q50, "p90": r.recovery_q90},
             "p_detonation_given_breach": r.p_impact_given_breach,
             "loss_musd": {"p10": r.loss_q10, "p50": r.loss_q50, "p90": r.loss_q90},
             "risk_index": r.risk_index, "resilience_score": 100 - r.risk_index,
             "grade": grade(100 - r.risk_index)} for r in df.itertuples()]

