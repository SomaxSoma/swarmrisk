"""Training, evaluation and persistence of the swarm-risk models.

The simulator is the "ground truth" generator; the ML models are fast,
smooth surrogates that learn outcome *probabilities* and *distributions*
from hundreds of thousands of simulated campaigns. Each simulated label is a
single noisy draw, so classifiers learn calibrated probabilities and quantile
regressors learn the spread of recovery times and losses.

Models in the bundle
--------------------
- ``breach``        P(swarm gains a foothold)                  - classifier
- ``severe``        P(infrastructure damage >= 25 %)           - classifier
- ``collapse``      P(losses exceed the company's liquidity)   - classifier
- ``impact``        P(swarm detonates: ransomware/wipers/sabotage | breach)
- ``damage``        E[infrastructure damage | breach]          - regressor
- ``recovery_{impact,contained}_{loc,q*}``  location model + 10/50/90 % residual
  quantiles of log recovery days, separately for detonated and contained attacks
- ``loss_{impact,contained}_{loc,q*}``      the same for net financial loss

The ``collapse`` classifier is trained on the simulator's exact per-campaign
collapse probability (soft labels) rather than on one noisy coin flip.
All models get physics-informed indices (``simulator.risk_indices``) as extra
features and monotonic constraints that encode basic security logic.

Recovery and loss are *bimodal*: a contained intrusion costs days, a
detonation costs months. Quantile regression on the mixed distribution is
unstable (the median can fall in the empty gap between the modes), so each
mode is modelled separately and the overall quantiles are computed exactly
from the mixture ``P(impact) * F_impact + (1 - P(impact)) * F_contained``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.special import ndtr
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, r2_score,
                             roc_auc_score)

from . import __version__
from .sampling import coerce_frame, sample_companies
from .schema import FEATURE_NAMES, INDUSTRIES, INDUSTRY_NAMES
from .simulator import monte_carlo, risk_indices, simulate

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "swarmrisk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

BOOL_COLS = ["cyber_insurance", "ot_ics_present", "soc_24x7", "deepfake_verification",
             "immutable_backups", "dr_site"]

# Monotonic constraints (+1 = risk can only go up as the feature grows, -1 = only down).
# They keep the surrogate faithful to basic security logic, which matters
# because recommendations are computed by "what if" changes to these features.
_SURFACE_UP = {"internet_facing_services": 1, "ai_agent_integrations": 1, "api_exposure": 1,
               "third_party_vendors": 1, "legacy_system_share": 1, "patch_latency_days": 1,
               "swarm_size": 1, "campaign_days": 1, "prior_incidents_3y": 1}
_PREVENT_DOWN = {"mfa_coverage": -1, "security_training": -1, "zero_trust_maturity": -1,
                 "ai_threat_detection": -1, "edr_coverage": -1}
_CONTAIN_DOWN = {"network_segmentation": -1, "ir_maturity": -1, "soc_24x7": -1}
_RECOVER = {"immutable_backups": -1, "dr_site": -1, "ir_maturity": -1,
            "restore_test_days": 1, "backup_frequency_hours": 1}

# Physics-informed indices (see simulator.risk_indices) get the matching directions.
_IX_ACCESS = {"ix_log_access_hazard": 1, "ix_expected_breach": 1, "ix_log_alert_rate": -1}
_IX_HARM = {"ix_log_spread_rate": 1, "ix_log_core_rate": 1, "ix_log_detect_rate": -1, "ix_spread_vs_detect": 1,
            "ix_log_contain_hours": 1, "ix_spread_x_contain": 1}
_IX_RECOVER = {"ix_log_recovery_mult": 1, "ix_log_recovery_mult_lost": 1, "ix_backup_loss_risk": 1}

MONOTONIC = {
    "breach": {**_SURFACE_UP, **_PREVENT_DOWN, "attacker_sophistication": 1, **_IX_ACCESS},
    "severe": {**_SURFACE_UP, **_PREVENT_DOWN, **_CONTAIN_DOWN, **_IX_ACCESS, **_IX_HARM},
    "collapse": {**_SURFACE_UP, **_PREVENT_DOWN, **_CONTAIN_DOWN, **_RECOVER,
                 "cash_reserve_months": -1, "cyber_insurance": -1, "deepfake_verification": -1,
                 **_IX_ACCESS, **_IX_HARM, **_IX_RECOVER, "ix_buffer_days_of_ops": -1,
                 "ix_log_buffer_to_revenue": -1, "ix_fraud_prob": 1},
    "damage": {**_CONTAIN_DOWN, "edr_coverage": -1, **_IX_HARM},
    "impact": {**_CONTAIN_DOWN, "edr_coverage": -1, "ai_threat_detection": -1, **_IX_HARM},
    "recovery": {**_RECOVER, **_IX_RECOVER},
    "loss": {**_RECOVER, "cyber_insurance": -1, **_IX_RECOVER},
}

QUANTILES = (0.1, 0.5, 0.9)
LOSS_EPS = 1e-5   # relative floor so log(loss / revenue) stays finite


def to_target(prefix: str, y: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    """Model-space transform: log1p(days) for recovery; log(loss / revenue) for loss (scale-free)."""
    return np.log1p(y) if prefix == "recovery" else np.log(y / revenue + LOSS_EPS)


def from_target(prefix: str, z: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    return np.expm1(z) if prefix == "recovery" else revenue * np.maximum(np.exp(z) - LOSS_EPS, 0.0)
COMPONENTS = ("impact", "contained")
Z90 = 1.2815516


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn (validated) profiles into the model matrix."""
    df = coerce_frame(df.reset_index(drop=True))
    X = df[FEATURE_NAMES].copy()
    for c in BOOL_COLS:
        X[c] = X[c].astype(int)
    X["industry"] = pd.Categorical(X["industry"], categories=INDUSTRY_NAMES)
    emp = df["employees"].clip(lower=1)
    rev = df["annual_revenue_musd"]
    X["security_staff_per_1k"] = df["security_staff"] / emp * 1000
    X["revenue_per_employee"] = rev / emp
    X["liquidity_musd"] = df["cash_reserve_months"] * rev * (1 - df["profit_margin"]) / 12
    X["liquidity_to_revenue"] = X["liquidity_musd"] / rev
    for key in ("attractiveness", "it_dependency", "reg_fine"):
        X[f"industry_{key}"] = df["industry"].map(lambda i, k=key: INDUSTRIES[i][k]).astype(float)
    X["swarm_power"] = df["swarm_size"] ** 0.55 * (0.3 + 1.4 * df["attacker_sophistication"])
    ix = risk_indices(df)
    for c in ix:
        X[c] = ix[c].to_numpy()
    return X


def _clf(constraints: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features="from_dtype", monotonic_cst=constraints, random_state=seed)


def _reg(constraints: dict, seed: int, loss="squared_error", quantile=None) -> HistGradientBoostingRegressor:
    kw = {"quantile": quantile} if quantile is not None else {}
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.1, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        categorical_features="from_dtype", monotonic_cst=constraints, random_state=seed, **kw)


def generate_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    """Sample ``n`` companies and simulate one campaign against each."""
    profiles = sample_companies(n, seed=seed)
    outcomes = simulate(profiles, seed=seed + 1).outcomes
    return pd.concat([profiles, outcomes], axis=1)


def _pinball(y, q_pred, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def train(n_samples: int = 600_000, seed: int = 7, fidelity_companies: int = 150,
          fidelity_runs: int = 400, verbose: bool = True, out_dir: Path | str | None = None) -> dict:
    """Generate data, fit every model, evaluate, and save the bundle (to ``out_dir`` or ``models/``)."""
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    data = generate_dataset(n_samples, seed=seed)
    log(f"[data] simulated {len(data):,} campaigns in {time.time() - t0:.1f}s "
        f"(breach rate {data.breached.mean():.1%}, collapse rate {data.collapse.mean():.1%})")

    rng = np.random.default_rng(seed)
    is_test = rng.random(len(data)) < 0.2
    train_df, test_df = data[~is_test].reset_index(drop=True), data[is_test].reset_index(drop=True)
    Xtr, Xte = build_features(train_df), build_features(test_df)
    models, metrics = {}, {"classifiers": {}, "regressors": {}}

    for name, target in (("breach", "breached"), ("severe", "severe_damage"), ("collapse", "collapse")):
        t = time.time()
        if name == "collapse":
            # Rao-Blackwellised soft labels: train on the simulator's exact P(collapse | campaign)
            # instead of one noisy coin flip per company (each row appears as a weighted 1 and 0).
            pc = train_df["collapse_prob"].to_numpy()
            m = _clf(MONOTONIC[name], seed).set_params(early_stopping=False, max_iter=600).fit(
                pd.concat([Xtr, Xtr], ignore_index=True), np.r_[np.ones(len(Xtr)), np.zeros(len(Xtr))],
                sample_weight=np.r_[pc, 1 - pc])
        else:
            m = _clf(MONOTONIC[name], seed).fit(Xtr, train_df[target].astype(int))
        p = m.predict_proba(Xte)[:, 1]
        y = test_df[target].astype(int).to_numpy()
        base = np.full_like(p, y.mean(), dtype=float)
        metrics["classifiers"][name] = {
            "target": target, "test_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "brier_skill": float(1 - brier_score_loss(y, p) / brier_score_loss(y, base)),
            "log_loss": float(log_loss(y, p)), "iterations": int(m.n_iter_),
        }
        models[name] = m
        log(f"[{name}] AUC={metrics['classifiers'][name]['auc']:.3f} "
            f"Brier skill={metrics['classifiers'][name]['brier_skill']:.3f} ({time.time() - t:.1f}s)")

    btr, bte = train_df.breached.to_numpy(), test_df.breached.to_numpy()
    Xtr_b, Xte_b = Xtr[btr], Xte[bte]
    tr_b, te_b = train_df[btr].reset_index(drop=True), test_df[bte].reset_index(drop=True)

    t = time.time()
    m = _clf(MONOTONIC["impact"], seed).fit(Xtr_b, tr_b.impact.astype(int))
    p = m.predict_proba(Xte_b)[:, 1]
    models["impact"] = m
    metrics["classifiers"]["impact_given_breach"] = {
        "target": "impact | breached", "test_rate": float(te_b.impact.mean()),
        "auc": float(roc_auc_score(te_b.impact, p)), "brier": float(brier_score_loss(te_b.impact, p))}
    log(f"[impact|breach] AUC={metrics['classifiers']['impact_given_breach']['auc']:.3f} ({time.time() - t:.1f}s)")

    t = time.time()
    m = _reg(MONOTONIC["damage"], seed).fit(Xtr_b, tr_b.infra_damage)
    pred = np.clip(m.predict(Xte_b), 0, 100)
    models["damage"] = m
    metrics["regressors"]["damage"] = {"mae": float(mean_absolute_error(te_b.infra_damage, pred)),
                                       "r2": float(r2_score(te_b.infra_damage, pred))}
    log(f"[damage] MAE={metrics['regressors']['damage']['mae']:.2f} R2={metrics['regressors']['damage']['r2']:.3f} "
        f"({time.time() - t:.1f}s)")

    for prefix, col, cons in (("recovery", "recovery_days", MONOTONIC["recovery"]),
                              ("loss", "financial_loss_musd", MONOTONIC["loss"])):
        for comp in COMPONENTS:
            sel_tr = tr_b.impact.to_numpy() if comp == "impact" else ~tr_b.impact.to_numpy()
            sel_te = te_b.impact.to_numpy() if comp == "impact" else ~te_b.impact.to_numpy()
            rev_tr = tr_b.annual_revenue_musd.to_numpy()[sel_tr]
            rev_te = te_b.annual_revenue_musd.to_numpy()[sel_te]
            ytr, yte = to_target(prefix, tr_b[col].to_numpy()[sel_tr], rev_tr), te_b[col].to_numpy()[sel_te]
            Xc_tr, Xc_te = Xtr_b[sel_tr], Xte_b[sel_te]
            # Location-scale: a squared-error model learns the typical (log) level, which it locates
            # far better than sign-only quantile gradients do in sparse tails; quantile models then
            # learn only the spread of the out-of-fold residuals around it.
            t = time.time()
            loc = _reg(cons, seed)
            resid = ytr - cross_val_predict(loc, Xc_tr, ytr, cv=3)
            loc.fit(Xc_tr, ytr)
            models[f"{prefix}_{comp}_loc"] = loc
            loc_te = loc.predict(Xc_te)
            for q in QUANTILES:
                key = f"{prefix}_{comp}_q{int(q * 100)}"
                m = _reg({}, seed, loss="quantile", quantile=q).fit(Xc_tr, resid)
                models[key] = m
                metrics["regressors"][key] = {"quantile": q,
                                              "coverage": float(np.mean(yte <= from_target(prefix, loc_te + m.predict(Xc_te), rev_te)))}
            log(f"[{prefix}_{comp}] coverage " + ", ".join(
                f"q{int(q * 100)}={metrics['regressors'][f'{prefix}_{comp}_q{int(q * 100)}']['coverage']:.3f}"
                for q in QUANTILES) + f" ({time.time() - t:.1f}s)")

    # Coverage of the combined (mixture) quantiles on all breached test rows.
    mix = predict_frame({"models": models, "feature_columns": list(Xtr.columns)}, test_df[bte])
    for prefix, col in (("recovery", "recovery_days"), ("loss", "financial_loss_musd")):
        y_all = te_b[col].to_numpy()
        metrics["regressors"][f"{prefix}_mixture"] = {
            f"q{int(q * 100)}_coverage": float(np.mean(y_all <= mix[f"{prefix}_q{int(q * 100)}"].to_numpy()))
            for q in QUANTILES}
        log(f"[{prefix} mixture] coverage " + ", ".join(
            f"{k}={v:.3f}" for k, v in metrics["regressors"][f"{prefix}_mixture"].items()))

    bundle = {"version": __version__, "models": models, "feature_columns": list(Xtr.columns),
              "population_medians": _population_reference(train_df), "metrics": metrics,
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_samples": int(n_samples)}

    if fidelity_companies:
        t = time.time()
        metrics["fidelity"] = surrogate_fidelity(bundle, fidelity_companies, fidelity_runs, seed=seed + 99)
        log(f"[fidelity] vs Monte-Carlo truth: " + ", ".join(
            f"{k} MAE={v['mae']:.3f}" for k, v in metrics["fidelity"].items() if isinstance(v, dict))
            + f" ({time.time() - t:.1f}s)")

    out = Path(out_dir) if out_dir is not None else MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / MODEL_PATH.name, compress=3)
    (out / METRICS_PATH.name).write_text(json.dumps(
        {k: bundle[k] for k in ("version", "trained_at", "n_samples", "metrics")}, indent=2))
    log(f"[done] saved {out / MODEL_PATH.name} in {time.time() - t0:.1f}s total")
    return bundle


def _population_reference(df: pd.DataFrame) -> dict:
    """Typical values used as the 'average company' baseline in explanations."""
    ref = {}
    for c in FEATURE_NAMES:
        col = df[c]
        if col.dtype == bool:
            ref[c] = bool(col.mean() >= 0.5)
        elif col.dtype == object:
            ref[c] = col.mode().iat[0]
        else:
            v = float(col.median())
            ref[c] = int(round(v)) if np.issubdtype(col.dtype, np.integer) else v
    return ref


def _component_quantiles(models: dict, X: pd.DataFrame, prefix: str, comp: str) -> np.ndarray:
    """(n, 3) predicted 10/50/90 % quantiles in model (target) space, forced to be strictly increasing."""
    loc = models[f"{prefix}_{comp}_loc"].predict(X)
    q = loc[:, None] + np.column_stack([models[f"{prefix}_{comp}_q{int(p * 100)}"].predict(X) for p in QUANTILES])
    q = np.sort(q, axis=1)
    q[:, 1] = np.maximum(q[:, 1], q[:, 0] + 1e-3)
    q[:, 2] = np.maximum(q[:, 2], q[:, 1] + 1e-3)
    return q


def _component_cdf(z: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CDF of a distribution with 10/50/90 % quantiles ``q`` (two-piece normal in log space).

    It reproduces the three given quantiles exactly and has normal tails.
    """
    lo, mid, hi = q[:, :1], q[:, 1:2], q[:, 2:3]
    score = np.where(z < mid, (z - mid) / (mid - lo), (z - mid) / (hi - mid)) * Z90
    return ndtr(score)


def mixture_quantiles(w: np.ndarray, qa: np.ndarray, qb: np.ndarray, probs=QUANTILES,
                      grid: int = 800) -> np.ndarray:
    """Quantiles of the mixture ``w * A + (1 - w) * B`` from each component's 10/50/90 % quantiles."""
    w = np.asarray(w, dtype=float)
    span = np.maximum(qa[:, 2] - qa[:, 0], qb[:, 2] - qb[:, 0])
    lo = np.minimum(qa[:, 0], qb[:, 0]) - 3 * span
    hi = np.maximum(qa[:, 2], qb[:, 2]) + 3 * span
    z = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, grid)[None, :]
    F = w[:, None] * _component_cdf(z, qa) + (1 - w[:, None]) * _component_cdf(z, qb)
    out = np.empty((len(w), len(probs)))
    rows = np.arange(len(w))
    for j, p in enumerate(probs):
        k = np.clip(np.argmax(F >= p, axis=1), 1, grid - 1)
        f0, f1 = F[rows, k - 1], F[rows, k]
        frac = np.clip((p - f0) / np.maximum(f1 - f0, 1e-12), 0, 1)
        out[:, j] = z[rows, k - 1] + frac * (z[rows, k] - z[rows, k - 1])
    return out


def predict_frame(bundle: dict, profiles: pd.DataFrame) -> pd.DataFrame:
    """Vectorised predictions for many profiles."""
    X = build_features(profiles)[bundle["feature_columns"]]
    m = bundle["models"]
    out = pd.DataFrame(index=X.index)
    for name in ("breach", "severe", "collapse"):
        out[f"p_{name}"] = m[name].predict_proba(X)[:, 1]
    w = m["impact"].predict_proba(X)[:, 1]
    out["p_impact_given_breach"] = w
    out["damage_given_breach"] = np.clip(m["damage"].predict(X), 0, 100)
    rev = X["annual_revenue_musd"].to_numpy(dtype=float)[:, None]
    for prefix, lo_cap, hi_cap in (("recovery", 0.25, 730.0), ("loss", 0.0, None)):
        qa = _component_quantiles(m, X, prefix, "impact")
        qb = _component_quantiles(m, X, prefix, "contained")
        mix = mixture_quantiles(w, qa, qb)
        for j, q in enumerate(QUANTILES):
            tag = f"q{int(q * 100)}"
            r = rev[:, 0]
            out[f"{prefix}_{tag}"] = np.clip(from_target(prefix, mix[:, j], r), lo_cap, hi_cap)
            out[f"{prefix}_impact_{tag}"] = np.clip(from_target(prefix, qa[:, j], r), lo_cap, hi_cap)
            out[f"{prefix}_contained_{tag}"] = np.clip(from_target(prefix, qb[:, j], r), lo_cap, hi_cap)
    # Severe damage requires a breach; collapse without a breach (deepfake fraud alone) is rare.
    out["p_severe"] = np.minimum(out["p_severe"], out["p_breach"])
    out["p_collapse"] = np.minimum(out["p_collapse"], out["p_breach"] + 0.02)
    return out


def surrogate_fidelity(bundle: dict, n_companies: int = 150, runs: int = 400, seed: int = 123) -> dict:
    """Compare model predictions with high-precision Monte-Carlo estimates on unseen companies."""
    profiles = sample_companies(n_companies, seed=seed)
    preds = predict_frame(bundle, profiles)
    truth = {"p_breach": [], "p_severe": [], "p_collapse": [], "p_impact_given_breach": [], "recovery_q50": []}
    for i, row in enumerate(profiles.to_dict("records")):
        mc = monte_carlo(row, runs=runs, seed=seed + i, trace=False)
        truth["p_breach"].append(mc["p_breach"])
        truth["p_severe"].append(mc["p_severe_damage"])
        truth["p_collapse"].append(mc["p_collapse"])
        truth["p_impact_given_breach"].append(mc["p_impact_given_breach"] if mc["p_breach"] > 0.05 else np.nan)
        truth["recovery_q50"].append(mc["recovery_days"]["p50"] if mc["p_breach"] > 0.05 else np.nan)
    out = {"companies": n_companies, "runs_per_company": runs}
    for k, v in truth.items():
        v = np.array(v, dtype=float)
        p = preds[k].to_numpy()
        ok = ~np.isnan(v)
        entry = {"mae": float(np.mean(np.abs(p[ok] - v[ok]))),
                 "corr": float(np.corrcoef(p[ok], v[ok])[0, 1]) if ok.sum() > 2 else float("nan")}
        if k == "recovery_q50":
            entry["median_abs_rel_err"] = float(np.median(np.abs(p[ok] - v[ok]) / np.maximum(v[ok], 1)))
        out[k] = entry
    return out


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run:  python -m swarmrisk train")
    return joblib.load(path)
