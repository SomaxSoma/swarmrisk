"""Generate realistic synthetic company populations.

Companies are drawn around a latent *security maturity* so that controls are
correlated the way they are in the real world (mature organisations tend to
patch faster, have EDR, test backups, ...). A slice of fully random profiles is
mixed in so the models also see unusual combinations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURE_MAP, FEATURE_NAMES, FEATURES, INDUSTRIES, INDUSTRY_NAMES

_INDUSTRY_WEIGHTS = np.array([0.13, 0.12, 0.08, 0.14, 0.13, 0.15, 0.08, 0.09, 0.08])
_MATURITY_SHIFT = {"finance": 0.15, "technology": 0.10, "government": 0.0, "healthcare": -0.05,
                   "energy": 0.03, "manufacturing": -0.07, "retail": -0.03, "logistics": -0.05,
                   "education": -0.12}
_LEGACY_SHIFT = {"government": 0.15, "healthcare": 0.15, "manufacturing": 0.12, "energy": 0.12,
                 "education": 0.08, "logistics": 0.05}


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _realistic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    ind_idx = rng.choice(len(INDUSTRY_NAMES), size=n, p=_INDUSTRY_WEIGHTS / _INDUSTRY_WEIGHTS.sum())
    industry = np.array(INDUSTRY_NAMES)[ind_idx]
    shift = np.array([_MATURITY_SHIFT[i] for i in industry])
    legacy_shift = np.array([_LEGACY_SHIFT.get(i, 0.0) for i in industry])
    ot_w = np.array([INDUSTRIES[i]["ot_weight"] for i in industry])
    is_tech = (industry == "technology").astype(float)

    log_emp = np.clip(rng.normal(3.0, 0.95, n), 1.0, np.log10(500_000))
    employees = np.round(10 ** log_emp)
    size = (log_emp - 1.0) / (np.log10(500_000) - 1.0)          # 0..1

    rev_per_emp = 10 ** rng.normal(np.log10(0.22), 0.35, n)      # $M per employee
    revenue = np.clip(employees * rev_per_emp, 0.5, 500_000)
    margin = np.clip(rng.normal(0.07, 0.10, n), -0.5, 0.6)
    cash = np.clip(10 ** rng.normal(np.log10(4.0), 0.4, n) * (0.6 + 0.8 * (margin > 0)), 0.0, 60.0)
    distressed = rng.random(n) < 0.07
    cash[distressed] = rng.uniform(0.0, 1.0, distressed.sum())

    m = _clip01(rng.beta(2.2, 2.2, n) + shift + 0.15 * (size - 0.5))

    def ctrl(scale=1.0, noise=0.15, offset=0.0):
        return _clip01(m * scale + offset + rng.normal(0, noise, n))

    df = pd.DataFrame({
        "industry": industry,
        "employees": employees.astype(int),
        "annual_revenue_musd": revenue,
        "profit_margin": margin,
        "cash_reserve_months": cash,
        "cyber_insurance": rng.random(n) < (0.25 + 0.4 * size + 0.25 * m),
        "prior_incidents_3y": rng.poisson(0.3 + 2.0 * (1 - m) + 2.5 * size),
        "internet_facing_services": np.clip(np.round(10 ** (0.5 + 0.5 * log_emp + rng.normal(0, 0.3, n))), 1, 20_000).astype(int),
        "ai_agent_integrations": np.clip(np.round(
            (rng.random(n) > 0.25) * 10 ** (-0.6 + 0.4 * log_emp + 0.6 * is_tech + rng.normal(0, 0.45, n))), 0, 2_000).astype(int),
        "api_exposure": _clip01(rng.beta(2, 3, n) + 0.2 * is_tech),
        "third_party_vendors": np.clip(np.round(10 ** (0.8 + 0.45 * log_emp + rng.normal(0, 0.3, n))), 0, 50_000).astype(int),
        "cloud_share": _clip01(rng.beta(2, 2, n) + 0.2 * is_tech - 0.1 * (ot_w > 0.5)),
        "legacy_system_share": _clip01(rng.beta(2, 4, n) + 0.2 * (1 - m) + legacy_shift - 0.1),
        "ot_ics_present": rng.random(n) < (0.03 + 0.9 * ot_w),
        "security_staff": np.clip(np.round(employees * (0.002 + 0.012 * m) * 10 ** rng.normal(0, 0.2, n)), 0, 10_000).astype(int),
        "security_budget_pct": np.clip(3 + 15 * m + rng.normal(0, 2, n), 0, 40),
        "mfa_coverage": ctrl(1.1, 0.15),
        "patch_latency_days": np.clip(10 ** (2.1 - 1.3 * m + rng.normal(0, 0.25, n)), 1, 365),
        "edr_coverage": ctrl(1.05, 0.15),
        "network_segmentation": ctrl(1.0, 0.18),
        "zero_trust_maturity": ctrl(0.85, 0.15, -0.05),
        "ai_threat_detection": ctrl(0.85, 0.2, -0.1),
        "soc_24x7": rng.random(n) < (0.6 * m + 0.3 * size),
        "security_training": ctrl(1.0, 0.15),
        "deepfake_verification": rng.random(n) < (0.05 + 0.55 * m),
        "immutable_backups": rng.random(n) < (0.1 + 0.65 * m),
        "backup_frequency_hours": np.clip(10 ** (2.2 - 1.5 * m + rng.normal(0, 0.3, n)), 1, 720),
        "restore_test_days": np.clip(10 ** (2.8 - 1.0 * m + rng.normal(0, 0.3, n)), 1, 1000),
        "ir_maturity": ctrl(1.0, 0.15),
        "dr_site": rng.random(n) < (0.05 + 0.4 * m + 0.25 * size),
    })
    return df


def _uniform(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Independent draws over the full range of every feature (log-uniform where marked)."""
    cols = {}
    for f in FEATURES:
        if f.kind == "category":
            cols[f.name] = rng.choice(INDUSTRY_NAMES, size=n)
        elif f.kind == "bool":
            cols[f.name] = rng.random(n) < 0.5
        else:
            lo, hi = f.low, f.high
            if f.log_scale:
                v = 10 ** rng.uniform(np.log10(max(lo, 1)), np.log10(hi), n)
                if lo == 0:
                    v[rng.random(n) < 0.1] = 0
            else:
                v = rng.uniform(lo, hi, n)
            cols[f.name] = np.round(v).astype(int) if f.kind == "int" else v
    return pd.DataFrame(cols)


def _scenario(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "swarm_size": np.round(10 ** rng.uniform(0, 4, n)).astype(int),
        "attacker_sophistication": rng.uniform(0, 1, n),
        "campaign_days": rng.integers(1, 91, n),
    })


def sample_companies(n: int, seed: int = 0, uniform_share: float = 0.15) -> pd.DataFrame:
    """Sample ``n`` company+scenario profiles as a DataFrame with ``FEATURE_NAMES`` columns."""
    rng = np.random.default_rng(seed)
    n_uni = int(round(n * uniform_share))
    parts = [_realistic(n - n_uni, rng)]
    if n_uni:
        parts.append(_uniform(n_uni, rng))
    base = pd.concat(parts, ignore_index=True)
    scen = _scenario(n, rng)
    for c in scen:
        base[c] = scen[c].values
    base = base.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return coerce_frame(base)


def coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to schema ranges and cast to canonical dtypes, in canonical column order."""
    out = pd.DataFrame(index=df.index)
    for name in FEATURE_NAMES:
        f = FEATURE_MAP[name]
        col = df[name]
        if f.kind == "category":
            out[name] = col.astype(str)
        elif f.kind == "bool":
            out[name] = col.astype(bool)
        else:
            col = np.clip(col.astype(float), f.low, f.high)
            out[name] = np.round(col).astype(int) if f.kind == "int" else col
    return out
