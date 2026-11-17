"""High-level prediction API: scores, explanations, recommendations, threat curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .model import load_bundle, predict_frame
from .schema import FEATURE_MAP, FEATURES, SCENARIO_FEATURES, validate_profile
from .simulator import monte_carlo

SWARM_SIZES = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
SOPHISTICATION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MATRIX_SWARM_SIZES = [1, 10, 50, 100, 500, 1000, 5000, 10000]


@dataclass(frozen=True)
class Intervention:
    key: str
    title: str
    detail: str
    cost: str          # "$", "$$", "$$$"
    effort_weeks: str
    apply: Callable[[dict], dict]


def _set(**kw):
    """Build an apply() that moves features toward better values, never worse."""
    def f(p: dict) -> dict:
        q = dict(p)
        for k, (mode, v) in kw.items():
            if mode == "max":
                q[k] = max(q[k], v)
            elif mode == "min":
                q[k] = min(q[k], v)
            elif mode == "scale":
                q[k] = type(q[k])(max(FEATURE_MAP[k].low, round(q[k] * v)))
            else:
                q[k] = v
        return q
    return f


INTERVENTIONS: list[Intervention] = [
    Intervention("mfa", "Phishing-resistant MFA everywhere",
                 "Roll out FIDO2/passkeys to 98%+ of accounts so AI-written phishing and credential theft stop working.",
                 "$", "4-8", _set(mfa_coverage=("max", 0.98))),
    Intervention("edr", "EDR on every endpoint and server",
                 "Full endpoint detection & response coverage slows lateral movement and spots agents earlier.",
                 "$$", "4-10", _set(edr_coverage=("max", 0.95))),
    Intervention("patch", "Patch critical vulnerabilities within 7 days",
                 "Swarms scan and weaponise new CVEs at machine speed; shrink the window.",
                 "$", "4-12", _set(patch_latency_days=("min", 7.0))),
    Intervention("segment", "Segment the network and isolate crown jewels",
                 "Micro-segmentation keeps a single foothold from reaching core and OT systems.",
                 "$$$", "12-26", _set(network_segmentation=("max", 0.85))),
    Intervention("zerotrust", "Adopt zero-trust access",
                 "Least privilege plus continuous verification of every identity, device and AI agent.",
                 "$$$", "16-40", _set(zero_trust_maturity=("max", 0.8))),
    Intervention("aidetect", "Deploy AI-driven threat detection",
                 "Behavioural/AI analytics that can keep up with machine-speed, many-agent attacks.",
                 "$$", "6-12", _set(ai_threat_detection=("max", 0.85))),
    Intervention("soc", "24/7 SOC or managed detection & response",
                 "Swarms do not keep office hours; make sure someone answers alerts at 3 a.m.",
                 "$$", "4-8", _set(soc_24x7=("set", True))),
    Intervention("backups", "Immutable, frequent backups",
                 "Offline/immutable copies every 12h or better so ransomware cannot take the recovery path away.",
                 "$", "2-6", _set(immutable_backups=("set", True), backup_frequency_hours=("min", 12.0))),
    Intervention("restore", "Monthly full restore tests",
                 "Untested backups fail when needed; rehearsed restores cut recovery time.",
                 "$", "2-4", _set(restore_test_days=("min", 30.0))),
    Intervention("ir", "Tested incident-response plan and retainer",
                 "Playbooks for AI-swarm scenarios, tabletop exercises and an IR firm on retainer.",
                 "$", "4-8", _set(ir_maturity=("max", 0.85))),
    Intervention("dr", "Disaster-recovery site",
                 "A warm standby environment keeps the business running while production is rebuilt.",
                 "$$$", "12-30", _set(dr_site=("set", True))),
    Intervention("aiagents", "Lock down internal AI agents and APIs",
                 "Sandbox agent tools, least-privilege API keys, prompt-injection filtering, API gateway.",
                 "$$", "6-12", _set(api_exposure=("min", 0.15))),
    Intervention("surface", "Halve the internet-facing attack surface",
                 "Decommission or put behind zero-trust proxies the services that do not need to be public.",
                 "$$", "8-16", _set(internet_facing_services=("scale", 0.5))),
    Intervention("legacy", "Retire or isolate legacy systems",
                 "End-of-life systems are the easiest entry points for automated exploitation.",
                 "$$$", "26-52", _set(legacy_system_share=("min", 0.10))),
    Intervention("people", "Awareness training and deepfake verification",
                 "Train staff against AI-crafted lures and require call-back checks for payments and urgent requests.",
                 "$", "2-6", _set(security_training=("max", 0.85), deepfake_verification=("set", True))),
    Intervention("finance", "Financial resilience: cash buffer and cyber insurance",
                 "Six months of cash reserves and an active cyber policy keep a bad incident from becoming fatal.",
                 "$$", "12-52", _set(cash_reserve_months=("max", 6.0), cyber_insurance=("set", True))),
]


def risk_index(p_breach, p_severe, p_collapse):
    """0 (safe) - 100 (critical) composite; collapse weighs the most."""
    p_breach, p_severe, p_collapse = map(np.asarray, (p_breach, p_severe, p_collapse))
    return 100 * (0.20 * p_breach + 0.35 * p_severe + 0.45 * np.minimum(1.0, 2.0 * p_collapse))


def level(p: float) -> str:
    return "low" if p < 0.10 else "moderate" if p < 0.30 else "high" if p < 0.60 else "critical"


def grade(resilience: float) -> str:
    for g, cut in (("A", 85), ("B", 70), ("C", 55), ("D", 40)):
        if resilience >= cut:
            return g
    return "F"


def _fmt_money(musd: float) -> str:
    if musd >= 1000:
        return f"${musd / 1000:.1f}B"
    if musd >= 1:
        return f"${musd:.1f}M"
    return f"${musd * 1000:.0f}K"


def _fmt_days(d: float) -> str:
    if d < 1:
        h = round(d * 24)
        return f"{h} hour" if h == 1 else f"{h} hours"
    if d < 60:
        return "1 day" if round(d) == 1 else f"{d:.0f} days"
    return f"{d / 30.4:.1f} months"


class Predictor:
    def __init__(self, bundle: dict | None = None):
        self.bundle = bundle if bundle is not None else load_bundle()
        self.reference = self.bundle["population_medians"]

    # -- core --------------------------------------------------------------
    def _predict_many(self, profiles: list[dict]) -> pd.DataFrame:
        out = predict_frame(self.bundle, pd.DataFrame(profiles))
        out["risk_index"] = risk_index(out.p_breach, out.p_severe, out.p_collapse)
        return out

    def predict(self, profile: dict) -> dict[str, Any]:
        p = validate_profile(profile)
        r = self._predict_many([p]).iloc[0]
        resilience = float(100 - r.risk_index)
        res = {
            "p_breach": float(r.p_breach),
            "p_severe_damage": float(r.p_severe),
            "p_collapse": float(r.p_collapse),
            "expected_damage_given_breach": float(r.damage_given_breach),
            "expected_damage": float(r.damage_given_breach * r.p_breach),
            "p_detonation_given_breach": float(r.p_impact_given_breach),
            "p_detonation": float(r.p_impact_given_breach * r.p_breach),
            "recovery_days": {"p10": float(r.recovery_q10), "p50": float(r.recovery_q50), "p90": float(r.recovery_q90)},
            "recovery_days_if_detonated": {"p10": float(r.recovery_impact_q10), "p50": float(r.recovery_impact_q50),
                                           "p90": float(r.recovery_impact_q90)},
            "recovery_days_if_contained": {"p10": float(r.recovery_contained_q10), "p50": float(r.recovery_contained_q50),
                                           "p90": float(r.recovery_contained_q90)},
            "loss_musd": {"p10": float(r.loss_q10), "p50": float(r.loss_q50), "p90": float(r.loss_q90)},
            "risk_index": float(r.risk_index),
            "resilience_score": resilience,
            "grade": grade(resilience),
            "levels": {"breach": level(r.p_breach), "severe_damage": level(r.p_severe),
                       "collapse": level(2 * r.p_collapse)},
        }
        res["summary"] = self._summary(p, res)
        return res

    # -- explanations ----------------------------------------------------------
    def explain(self, profile: dict, top: int = 10) -> list[dict]:
        """Per-feature what-if attribution against a typical company.

        For each company feature we reset it to the population's typical value
        and measure how the risk index moves. Positive impact = this feature
        makes the company *riskier* than typical.
        """
        p = validate_profile(profile)
        names = [f.name for f in FEATURES if f.name not in SCENARIO_FEATURES and p[f.name] != self.reference[f.name]]
        variants = [p] + [{**p, n: self.reference[n]} for n in names]
        preds = self._predict_many(variants)
        base = preds.iloc[0]
        rows = []
        for i, n in enumerate(names, start=1):
            v = preds.iloc[i]
            rows.append({
                "feature": n, "label": FEATURE_MAP[n].label, "group": FEATURE_MAP[n].group,
                "value": p[n], "typical": self.reference[n],
                "impact": float(base.risk_index - v.risk_index),
                "d_breach": float(base.p_breach - v.p_breach),
                "d_collapse": float(base.p_collapse - v.p_collapse),
                "d_recovery_p50": float(base.recovery_q50 - v.recovery_q50),
            })
        rows.sort(key=lambda r: abs(r["impact"]), reverse=True)
        return [r for r in rows if abs(r["impact"]) >= 0.05][:top]

    # -- recommendations -------------------------------------------------------
    def recommend(self, profile: dict, top: int = 8) -> dict:
        p = validate_profile(profile)
        candidates = [(iv, iv.apply(p)) for iv in INTERVENTIONS]
        candidates = [(iv, q) for iv, q in candidates if q != p]
        preds = self._predict_many([p] + [q for _, q in candidates])
        base = preds.iloc[0]
        items = []
        for i, (iv, q) in enumerate(candidates, start=1):
            v = preds.iloc[i]
            items.append({
                "key": iv.key, "title": iv.title, "detail": iv.detail, "cost": iv.cost,
                "effort_weeks": iv.effort_weeks,
                "changes": {k: q[k] for k in q if q[k] != p[k]},
                "risk_reduction": float(base.risk_index - v.risk_index),
                "d_breach": float(v.p_breach - base.p_breach),
                "d_severe": float(v.p_severe - base.p_severe),
                "d_collapse": float(v.p_collapse - base.p_collapse),
                "d_recovery_p50": float(v.recovery_q50 - base.recovery_q50),
            })
        items.sort(key=lambda x: x["risk_reduction"], reverse=True)
        items = [x for x in items if x["risk_reduction"] > 0.05]

        # Greedy combined plan: stack the best interventions one after another.
        plan, current, current_score = [], dict(p), float(base.risk_index)
        remaining = [iv for iv in INTERVENTIONS if iv.key in {x["key"] for x in items}]
        for _ in range(min(5, len(remaining))):
            trials = [(iv, iv.apply(current)) for iv in remaining]
            scores = self._predict_many([q for _, q in trials]).risk_index.to_numpy()
            j = int(np.argmin(scores))
            if current_score - scores[j] < 0.05:
                break
            plan.append({"key": trials[j][0].key, "title": trials[j][0].title,
                         "risk_after": float(scores[j])})
            current, current_score = trials[j][1], float(scores[j])
            remaining.pop(j)
        after = self.predict(current) if plan else None
        return {"actions": items[:top], "plan": plan, "plan_result": after, "plan_profile": current if plan else None}

    # -- threat curves -----------------------------------------------------------
    def threat_curves(self, profile: dict) -> dict:
        p = validate_profile(profile)
        swarm = self._predict_many([{**p, "swarm_size": s} for s in SWARM_SIZES])
        soph = self._predict_many([{**p, "attacker_sophistication": s} for s in SOPHISTICATION_LEVELS])
        grid = [{**p, "swarm_size": s, "attacker_sophistication": a}
                for a in SOPHISTICATION_LEVELS for s in MATRIX_SWARM_SIZES]
        mat = self._predict_many(grid)

        def cols(df):
            return {c: df[c].round(4).tolist() for c in ("p_breach", "p_severe", "p_collapse", "recovery_q50")}
        return {
            "swarm_sizes": SWARM_SIZES, "by_swarm": cols(swarm),
            "sophistication": SOPHISTICATION_LEVELS, "by_sophistication": cols(soph),
            "matrix": {"swarm_sizes": MATRIX_SWARM_SIZES, "sophistication": SOPHISTICATION_LEVELS,
                       "risk_index": mat.risk_index.to_numpy().reshape(len(SOPHISTICATION_LEVELS), -1).round(1).tolist()},
        }

    # -- simulation ------------------------------------------------------------
    def simulate(self, profile: dict, runs: int = 1000, seed: int | None = 0) -> dict:
        runs = int(min(max(runs, 50), 20_000))
        return monte_carlo(validate_profile(profile), runs=runs, seed=seed, trace=True)

    def report(self, profile: dict, simulate_runs: int = 0) -> dict:
        p = validate_profile(profile)
        out = {"profile": p, "prediction": self.predict(p), "drivers": self.explain(p),
               "recommendations": self.recommend(p), "threat": self.threat_curves(p)}
        if simulate_runs:
            out["simulation"] = self.simulate(p, runs=simulate_runs)
        return out

    # -- text --------------------------------------------------------------------
    @staticmethod
    def _summary(p: dict, r: dict) -> str:
        soph = p["attacker_sophistication"]
        soph_txt = "state-level" if soph >= 0.8 else "advanced" if soph >= 0.55 else "moderately skilled" if soph >= 0.3 else "unsophisticated"
        rec = r["recovery_days"]
        lines = [
            f"Against {'an' if soph_txt[0] in 'aeiou' else 'a'} {soph_txt} swarm of {p['swarm_size']:,} AI agents attacking for {p['campaign_days']} days, "
            f"this {p['industry']} company has a {r['p_breach']:.0%} chance of being breached and a "
            f"{r['p_severe_damage']:.0%} chance of severe infrastructure damage (25%+ of systems destroyed or encrypted).",
            f"Once inside, the swarm detonates (ransomware, wipers or sabotage) before being contained "
            f"{r['p_detonation_given_breach']:.0%} of the time. "
            f"If breached, expect about {r['expected_damage_given_breach']:.0f}% of infrastructure to be damaged and recovery to take "
            f"{_fmt_days(rec['p50'])} (likely range {_fmt_days(rec['p10'])} to {_fmt_days(rec['p90'])}; "
            f"{_fmt_days(r['recovery_days_if_detonated']['p50'])} after a detonation, "
            f"{_fmt_days(r['recovery_days_if_contained']['p50'])} if contained), "
            f"with a typical net loss of {_fmt_money(r['loss_musd']['p50'])} (bad case {_fmt_money(r['loss_musd']['p90'])}).",
            f"Probability that the attack pushes the company into collapse (losses exceeding its liquidity): "
            f"{r['p_collapse']:.1%}. Overall resilience grade: {r['grade']} ({r['resilience_score']:.0f}/100).",
        ]
        return " ".join(lines)
