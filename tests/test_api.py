import pytest
from fastapi.testclient import TestClient

from swarmrisk import api
from swarmrisk.predictor import Predictor


@pytest.fixture(scope="module")
def client(small_bundle):
    api.predictor.cache_clear()
    original = api.predictor
    api.predictor = lambda: Predictor(small_bundle)   # use the test model, not models/
    with TestClient(api.app) as c:
        yield c
    api.predictor = original


def test_index_and_assets(client):
    assert "SwarmRisk" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_schema_and_presets(client):
    s = client.get("/api/schema").json()
    assert len(s["features"]) > 25
    presets = client.get("/api/presets").json()
    assert {p["id"] for p in presets} >= {"regional_hospital", "global_bank"}


def test_predict(client):
    r = client.post("/api/predict", json={"industry": "retail", "employees": 300})
    assert r.status_code == 200
    body = r.json()
    assert {"prediction", "drivers", "recommendations", "threat", "profile"} <= set(body)


def test_predict_validation_error(client):
    r = client.post("/api/predict", json={"mfa_coverage": 3})
    assert r.status_code == 422
    assert "mfa_coverage" in r.json()["detail"]


def test_simulate(client):
    r = client.post("/api/simulate", json={"profile": {}, "runs": 100})
    assert r.status_code == 200
    assert r.json()["runs"] == 100
    assert client.post("/api/simulate", json={"profile": {}, "runs": "lots"}).status_code == 422


def test_batch(client):
    r = client.post("/api/batch", json=[{}, {"industry": "finance"}])
    assert r.status_code == 200 and len(r.json()) == 2
    big = client.post("/api/batch", json=[{"employees": 100 + i} for i in range(60)])
    assert big.status_code == 200 and len(big.json()) == 60
    assert client.post("/api/batch", json=[{}, {"mfa_coverage": -1}]).status_code == 422
