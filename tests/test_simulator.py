"""The simulator is the ground truth, so its behaviour must follow basic security logic."""
import numpy as np
import pandas as pd

from swarmrisk.sampling import sample_companies
from swarmrisk.simulator import monte_carlo, simulate

RUNS = 3000


def mc(profile, **changes):
    return monte_carlo({**profile, **changes}, runs=RUNS, seed=11, trace=False)


def test_deterministic_with_seed():
    df = sample_companies(500, seed=1)
    a = simulate(df, seed=5).outcomes
    b = simulate(df, seed=5).outcomes
    pd.testing.assert_frame_equal(a, b)


def test_outcome_invariants():
    df = sample_companies(5000, seed=2)
    o = simulate(df, seed=3).outcomes
    assert len(o) == len(df)
    nb = ~o.breached
    assert (o.infra_damage[nb] == 0).all() and (o.recovery_days[nb] == 0).all()
    assert not o.impact[nb].any() and not o.backups_lost[nb].any()
    assert o.infra_damage.between(0, 100).all()
    assert (o.recovery_days >= 0).all() and (o.recovery_days <= 730).all()
    assert not (o.impact & o.contained).any()
    assert (o.severe_damage == (o.infra_damage >= 25)).all()
    assert (o.financial_loss_musd >= 0).all()
    assert np.isfinite(o.financial_loss_musd).all()


def test_population_base_rates_are_plausible():
    o = simulate(sample_companies(20_000, seed=4), seed=5).outcomes
    assert 0.5 < o.breached.mean() < 0.9
    assert 0.05 < o.severe_damage.mean() < 0.35
    assert 0.01 < o.collapse.mean() < 0.12


def test_bigger_swarm_more_breaches(hospital):
    assert mc(hospital, swarm_size=5, campaign_days=10)["p_breach"] < mc(hospital, swarm_size=2000, campaign_days=10)["p_breach"]


def test_bigger_swarm_never_less_harmful(hospital):
    """Regression test: larger swarms must not detonate 'too early' and look less dangerous."""
    sizes = [10, 100, 1000, 10000]
    res = [mc(hospital, swarm_size=s) for s in sizes]
    for a, b in zip(res, res[1:]):
        assert b["p_collapse"] >= a["p_collapse"] - 0.02
        assert b["p_severe_damage"] >= a["p_severe_damage"] - 0.02


def test_mfa_and_training_reduce_breach(hospital):
    weak = mc(hospital, swarm_size=20, campaign_days=7)
    strong = mc(hospital, swarm_size=20, campaign_days=7, mfa_coverage=0.99, security_training=0.95)
    assert strong["p_breach"] < weak["p_breach"]


def test_segmentation_reduces_severe_damage(hospital):
    assert mc(hospital, network_segmentation=0.9)["p_severe_damage"] < mc(hospital, network_segmentation=0.1)["p_severe_damage"]


def test_immutable_backups_shorten_recovery_and_save_backups(hospital):
    no = mc(hospital, immutable_backups=False)
    yes = mc(hospital, immutable_backups=True)
    assert yes["p_backups_lost"] < no["p_backups_lost"]
    assert yes["recovery_days"]["p50"] < no["recovery_days"]["p50"]


def test_cash_reserves_reduce_collapse(hospital):
    assert mc(hospital, cash_reserve_months=24)["p_collapse"] < mc(hospital, cash_reserve_months=0.5)["p_collapse"]


def test_ot_never_reduces_damage(hospital):
    assert mc(hospital, ot_ics_present=True)["mean_damage_given_breach"] >= mc(hospital, ot_ics_present=False)["mean_damage_given_breach"]


def test_strong_bank_far_safer_than_hospital(hospital, bank):
    h, b = mc(hospital), mc(bank)
    assert b["p_severe_damage"] < h["p_severe_damage"]
    assert b["p_collapse"] <= h["p_collapse"]


def test_monte_carlo_trace_shapes(hospital):
    s = monte_carlo(hospital, runs=200, seed=0, trace=True)
    t = s["trace"]
    assert len(t["hours"]) == len(t["breached"]) == len(t["compromise"])
    assert all(0 <= v <= 1 for v in t["breached"])
    assert t["breached"] == sorted(t["breached"])          # cumulative
    assert abs(sum(s["vector_share"].values()) - 1) < 1e-6 or s["p_breach"] == 0
