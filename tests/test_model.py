import numpy as np
import pandas as pd
import pytest

from swarmrisk.model import mixture_quantiles, predict_frame
from swarmrisk.predictor import INTERVENTIONS, Predictor, grade, risk_index
from swarmrisk.schema import validate_profile
from swarmrisk.simulator import monte_carlo


@pytest.fixture(scope="module")
def pred(small_bundle):
    return Predictor(small_bundle)


def test_metrics_are_good(small_bundle):
    c = small_bundle["metrics"]["classifiers"]
    assert c["breach"]["auc"] > 0.85
    assert c["severe"]["auc"] > 0.88
    assert c["collapse"]["auc"] > 0.9
    for prefix in ("recovery", "loss"):
        cov = small_bundle["metrics"]["regressors"][f"{prefix}_mixture"]
        assert abs(cov["q50_coverage"] - 0.5) < 0.06
        assert abs(cov["q90_coverage"] - 0.9) < 0.06


def test_mixture_quantiles_math():
    qa = np.array([[1.0, 2.0, 3.5]])
    qb = np.array([[0.0, 0.5, 1.0]])
    np.testing.assert_allclose(mixture_quantiles(np.array([1.0]), qa, qb), qa, atol=0.01)
    np.testing.assert_allclose(mixture_quantiles(np.array([0.0]), qa, qb), qb, atol=0.01)
    mid = mixture_quantiles(np.array([0.5]), qa, qb)[0]
    assert qb[0, 0] - 0.01 <= mid[0] <= mid[1] <= mid[2] <= qa[0, 2] + 0.01


def test_prediction_ranges(pred, hospital):
    r = pred.predict(hospital)
    for k in ("p_breach", "p_severe_damage", "p_collapse", "p_detonation_given_breach"):
        assert 0 <= r[k] <= 1
    assert r["p_severe_damage"] <= r["p_breach"] + 1e-9
    rec = r["recovery_days"]
    assert 0 < rec["p10"] <= rec["p50"] <= rec["p90"] <= 730
    assert r["loss_musd"]["p10"] <= r["loss_musd"]["p50"] <= r["loss_musd"]["p90"]
    assert r["grade"] in "ABCDF"
    assert 0 <= r["resilience_score"] <= 100
    assert "swarm" in r["summary"]


@pytest.mark.parametrize("name", ["regional_hospital", "global_bank", "online_retailer", "ai_startup",
                                  "power_utility", "city_government"])
def test_production_model_tracks_simulation(name):
    """The shipped model (models/, trained on 300k campaigns) must agree with the simulator."""
    from conftest import load_example
    from swarmrisk.model import MODEL_PATH, load_bundle
    if not MODEL_PATH.exists():
        pytest.skip("production model not trained")
    prof = load_example(name)
    p = predict_frame(load_bundle(), pd.DataFrame([prof])).iloc[0]
    s = monte_carlo(prof, runs=4000, seed=1, trace=False)
    assert abs(p.p_breach - s["p_breach"]) < 0.08
    assert abs(p.p_severe - s["p_severe_damage"]) < 0.08
    assert abs(p.p_collapse - s["p_collapse"]) < 0.08
    if s["p_breach"] > 0.3:
        rel = abs(p.recovery_q50 - s["recovery_days"]["p50"]) / max(s["recovery_days"]["p50"], 2)
        assert rel < 0.35, (p.recovery_q50, s["recovery_days"]["p50"])


def test_controls_never_increase_risk(pred, hospital):
    """Monotonic constraints: every intervention must lower or keep the risk index."""
    base = pred.predict(hospital)["risk_index"]
    for iv in INTERVENTIONS:
        after = pred.predict(iv.apply(hospital))["risk_index"]
        assert after <= base + 0.5, iv.key


def test_bank_safer_than_hospital(pred, hospital, bank):
    assert pred.predict(bank)["resilience_score"] > pred.predict(hospital)["resilience_score"]


def test_recommendations_and_plan(pred, hospital):
    rec = pred.recommend(hospital)
    assert rec["actions"], "a weak hospital should get recommendations"
    reductions = [a["risk_reduction"] for a in rec["actions"]]
    assert reductions == sorted(reductions, reverse=True)
    assert all(r > 0 for r in reductions)
    assert rec["plan_result"]["risk_index"] < pred.predict(hospital)["risk_index"]
    validate_profile(rec["plan_profile"])      # the plan is a valid profile


def test_explain_sorted(pred, hospital):
    d = pred.explain(hospital)
    assert d and len(d) <= 10
    impacts = [abs(x["impact"]) for x in d]
    assert impacts == sorted(impacts, reverse=True)
    assert not {x["feature"] for x in d} & {"swarm_size", "attacker_sophistication", "campaign_days"}


def test_threat_curves_monotone_in_swarm(pred, hospital):
    t = pred.threat_curves(hospital)
    b = t["by_swarm"]["p_breach"]
    assert all(b[i] <= b[i + 1] + 1e-9 for i in range(len(b) - 1))
    assert np.array(t["matrix"]["risk_index"]).shape == (6, 8)


def test_grade_and_index():
    assert grade(90) == "A" and grade(10) == "F"
    assert float(risk_index(0, 0, 0)) == 0
    assert float(risk_index(1, 1, 1)) == 100
