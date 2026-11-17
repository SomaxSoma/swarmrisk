"""Agent-based simulation of an autonomous AI-agent swarm attacking a company.

The model works in discrete time steps (default 2 hours) over the campaign and
is vectorised over many companies at once, so tens of thousands of attacks can
be simulated in seconds.

Attack life-cycle per company
-----------------------------
1. **Reconnaissance** - recon agents build shared knowledge ``k`` (0..1) that
   makes every other agent more effective (the swarm effect). Bigger swarms
   learn faster but are noisier, so AI/behavioural detection may raise an early
   *alert*, which hardens the company.
2. **Initial access** - four parallel vectors, each a per-hour hazard:
   exploiting internet-facing services, AI-generated phishing / social
   engineering, supply-chain compromise via vendors, and hijacking the
   company's own AI agents (prompt injection / tool abuse).
3. **Lateral spread** - compromise fraction ``c`` of IT assets grows
   logistically; segmentation, zero trust and EDR slow it. Crown-jewel / OT
   systems (``core``) are reached through the gaps segmentation leaves.
4. **Detection & containment** - detection hazard rises with the size of the
   footprint and the number of agents; once detected, containment takes a time
   set by incident-response maturity and staffing, throttling the spread.
5. **Impact** - the swarm detonates (ransomware / wipers / sabotage) once it
   holds enough of the estate, or in panic when discovered.
6. **Aftermath** - infrastructure damage, backup survival, recovery time and
   financial loss, and whether that loss exceeds the company's liquidity
   (collapse).

All coefficients are documented assumptions, not measured truths; they encode
widely reported security relationships (MFA stops most credential attacks,
immutable backups shorten ransomware recovery, segmentation limits blast
radius, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sampling import coerce_frame
from .schema import INDUSTRIES

VECTORS = ["exploit", "phishing", "supply_chain", "ai_agent_hijack"]

# Global calibration constants (tuned so a realistic population gives
# plausible base rates; see ``python -m swarmrisk calibrate``).
K_ACCESS = 1.0e-4      # base initial-access hazard per agent-power per hour
K_SPREAD = 0.06        # base lateral-movement rate per hour
K_CORE = 0.02          # base rate of reaching crown-jewel / OT systems
K_DETECT = 0.045       # base post-breach detection hazard per hour
K_FRAUD = 8.0e-6       # deepfake-fraud hazard per agent-power per hour


@dataclass
class SimulationResult:
    outcomes: pd.DataFrame
    trace: dict | None = None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _derive(df: pd.DataFrame) -> dict:
    """All deterministic per-company quantities of the attack model.

    Shared by the simulator and by ``risk_indices`` (the ML features), so the
    two can never drift apart.
    """
    g = lambda c: df[c].to_numpy(dtype=float)  # noqa: E731
    ind = df["industry"].to_numpy()
    v = {
        "attract": np.array([INDUSTRIES[i]["attractiveness"] for i in ind]),
        "it_dep": np.array([INDUSTRIES[i]["it_dependency"] for i in ind]),
        "reg_fine": np.array([INDUSTRIES[i]["reg_fine"] for i in ind]),
    }
    for col in ("employees", "annual_revenue_musd", "profit_margin", "cash_reserve_months", "cyber_insurance",
                "prior_incidents_3y", "internet_facing_services", "ai_agent_integrations", "api_exposure",
                "third_party_vendors", "cloud_share", "legacy_system_share", "ot_ics_present", "security_staff",
                "security_budget_pct", "mfa_coverage", "patch_latency_days", "edr_coverage", "network_segmentation",
                "zero_trust_maturity", "ai_threat_detection", "soc_24x7", "security_training",
                "deepfake_verification", "immutable_backups", "backup_frequency_hours", "restore_test_days",
                "ir_maturity", "dr_site", "swarm_size", "attacker_sophistication", "campaign_days"):
        v[col] = g(col)
    emp, rev, margin, cash = v["employees"], v["annual_revenue_musd"], v["profit_margin"], v["cash_reserve_months"]
    S, soph = v["swarm_size"], v["attacker_sophistication"]
    edr, seg, zt, aid = v["edr_coverage"], v["network_segmentation"], v["zero_trust_maturity"], v["ai_threat_detection"]
    legacy, ir = v["legacy_system_share"], v["ir_maturity"]

    staff_ratio = np.clip(v["security_staff"] / np.maximum(emp, 1) * 1000 / 6.0, 0, 1.5)  # 6 FTE / 1000 staff = good
    capacity = 0.5 * np.minimum(staff_ratio, 1) + 0.5 * np.minimum(v["security_budget_pct"] / 15.0, 1)
    power = S ** 0.55                                       # diminishing returns of more agents
    soph_f = 0.3 + 1.4 * soph
    prior_f = 1 + 0.05 * np.minimum(v["prior_incidents_3y"], 10)
    common = power * soph_f * (0.6 + 0.4 * v["attract"]) * prior_f
    h_vec = np.stack([
        0.7 * np.log1p(v["internet_facing_services"]) / np.log(41)
        * (np.sqrt(v["patch_latency_days"] / 30) + 0.5 * soph ** 2) * (0.5 + legacy) * (1 - 0.4 * zt),
        (emp / 1000) ** 0.25 * (1 - 0.7 * v["security_training"]) * (1 - 0.85 * v["mfa_coverage"]),
        0.3 * np.log1p(v["third_party_vendors"]) / np.log(81) * (1 - 0.5 * zt),
        np.log1p(v["ai_agent_integrations"]) / np.log(6) * (0.2 + v["api_exposure"]) * (1 - 0.7 * aid) * (1 - 0.3 * zt),
    ], axis=1) * (K_ACCESS * common)[:, None]              # (n, 4) per-hour hazards

    v.update(
        capacity=capacity, power=power, soph_f=soph_f, h_vec=h_vec, h_total=h_vec.sum(axis=1),
        k_rate=0.004 * S ** 0.3 * (0.5 + soph),
        h_alert=0.002 * (0.6 * aid + 0.3 * v["soc_24x7"] + 0.3 * edr) * S ** 0.35 * (1 - 0.5 * soph),
        spread=(K_SPREAD * (0.4 + soph) * (S ** 0.25 / 200 ** 0.25) * (1 - 0.7 * seg) * (1 - 0.4 * zt)
                * (1 - 0.4 * edr) * (0.6 + legacy)),
        core_rate=K_CORE * soph_f * (1 - seg) ** 1.5 * (S ** 0.25 / 200 ** 0.25),   # more agents also pivot to OT
        d_static=(K_DETECT * (0.08 + 0.45 * edr + 0.55 * aid + 0.35 * v["soc_24x7"] + 0.2 * capacity)
                  * (1 + 0.35 * np.log10(S)) * (1.15 - 0.75 * soph)),
        contain_base=24 * (0.5 + 3 * (1 - ir)) * (1 + 0.1 * np.log10(emp)) / (0.6 + 0.6 * capacity),
        h_fraud=K_FRAUD * power * soph_f * (1 - 0.85 * v["deepfake_verification"]) * (1 - 0.4 * v["security_training"]),
        # recovery multiplier when backups survive, excluding the OT-damage term
        recovery_mult=((1 + 0.25 * v["backup_frequency_hours"] / 168) * (1 + 0.5 * v["restore_test_days"] / 365)
                       * np.where(v["dr_site"] > 0, 0.55, 1.0) * (1.4 - 0.6 * ir) * (1 + 0.5 * legacy)
                       * (1 - 0.25 * v["cloud_share"]) * (1.25 - 0.4 * capacity) * (1 + 0.07 * np.log10(emp))),
        recovery_mult_backups_lost=(2.2 * np.where(v["dr_site"] > 0, 0.55, 1.0) * (1.4 - 0.6 * ir) * (1 + 0.5 * legacy)
                                    * (1 - 0.25 * v["cloud_share"]) * (1.25 - 0.4 * capacity)
                                    * (1 + 0.07 * np.log10(emp))),
        buffer=cash * rev * (1 - margin) / 12 + np.maximum(margin, 0) * rev * 0.25 + 0.02 * rev,
    )
    return v


def risk_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed summary features of the attack model (deterministic, no randomness).

    Gradient-boosted trees approximate products of many factors poorly in the
    sparse tails of the population; handing them the model's key rates on a log
    scale lets them focus on the stochastic part.
    """
    df = coerce_frame(df.reset_index(drop=True))
    v = _derive(df)
    days = v["campaign_days"]
    daily_ops_rev = v["annual_revenue_musd"] / 365 * v["it_dep"]
    return pd.DataFrame({
        "ix_log_access_hazard": np.log(v["h_total"] + 1e-12),
        "ix_expected_breach": -np.expm1(-v["h_total"] * days * 24 * 1.5),
        "ix_log_spread_rate": np.log(v["spread"] + 1e-12),
        "ix_log_core_rate": np.log(v["core_rate"] + 1e-12),
        "ix_log_detect_rate": np.log(v["d_static"] + 1e-12),
        "ix_log_alert_rate": np.log(v["h_alert"] + 1e-12),
        "ix_spread_vs_detect": np.log((v["spread"] + 1e-12) / (v["d_static"] + 1e-12)),
        "ix_log_contain_hours": np.log(v["contain_base"]),
        "ix_spread_x_contain": v["spread"] * v["contain_base"],
        "ix_log_recovery_mult": np.log(v["recovery_mult"]),
        "ix_log_recovery_mult_lost": np.log(v["recovery_mult_backups_lost"]),
        "ix_backup_loss_risk": np.where(v["immutable_backups"] > 0, 0.03, 0.15 + 0.4 * (1 - 0.5 * v["network_segmentation"])),
        "ix_buffer_days_of_ops": v["buffer"] / np.maximum(daily_ops_rev, 1e-6),
        "ix_log_buffer_to_revenue": np.log(np.maximum(v["buffer"], 1e-6) / v["annual_revenue_musd"]),
        "ix_fraud_prob": -np.expm1(-v["h_fraud"] * days * 24),
    })


def simulate(df: pd.DataFrame, seed: int | None = None, dt: float = 2.0,
             trace: bool = False) -> SimulationResult:
    """Simulate one attack campaign per row of ``df`` (profiles as columns)."""
    rng = np.random.default_rng(seed)
    df = coerce_frame(df.reset_index(drop=True))
    n = len(df)
    v = _derive(df)

    attract, it_dep, reg_fine = v["attract"], v["it_dep"], v["reg_fine"]
    emp, rev, soph, days = v["employees"], v["annual_revenue_musd"], v["attacker_sophistication"], v["campaign_days"]
    ins, ot, edr, seg, aid = v["cyber_insurance"], v["ot_ics_present"], v["edr_coverage"], v["network_segmentation"], v["ai_threat_detection"]
    immut = v["immutable_backups"]
    h_vec, h_total, k_rate, h_alert = v["h_vec"], v["h_total"], v["k_rate"], v["h_alert"]
    spread, core_rate, d_static, h_fraud = v["spread"], v["core_rate"], v["d_static"], v["h_fraud"]
    theta = np.clip(0.25 + 0.45 * soph + rng.normal(0, 0.05, n), 0.15, 0.9)    # detonation threshold
    contain_h = v["contain_base"] * np.exp(rng.normal(0, 0.3, n))

    T = days * 24.0
    steps = int(np.ceil(T.max() / dt))

    # ---- state -------------------------------------------------------------
    k = np.zeros(n)
    alerted = np.zeros(n, bool)
    breached = np.zeros(n, bool)
    t_breach = np.full(n, np.nan)
    vector = np.full(n, -1)
    c = np.zeros(n)
    core = np.zeros(n)
    detected = np.zeros(n, bool)
    t_detect = np.full(n, np.nan)
    impact = np.zeros(n, bool)
    t_impact = np.full(n, np.nan)
    contained = np.zeros(n, bool)
    finished = np.zeros(n, bool)
    c_imp = np.zeros(n)
    core_imp = np.zeros(n)

    if trace:
        tr = {k_: np.zeros(steps) for k_ in ("compromise", "core", "breached", "detected", "impact", "contained")}

    for s in range(steps):
        t = s * dt
        active = t < T
        k = np.where(active, k + dt * k_rate * (1 - k), k)

        new_alert = active & ~alerted & (rng.random(n) < -np.expm1(-h_alert * dt))
        alerted |= new_alert

        pre = active & ~breached
        haz = h_total * (1 + k) * np.where(alerted, 0.45, 1.0)
        nb = pre & (rng.random(n) < -np.expm1(-haz * dt))
        if nb.any():
            breached[nb] = True
            t_breach[nb] = t
            c[nb] = 0.01
            idx = np.flatnonzero(nb)
            u = rng.random(idx.size) * h_total[idx]
            vector[idx] = np.minimum((u[:, None] > np.cumsum(h_vec[idx], axis=1)).sum(axis=1), 3)

        live = active & breached & ~finished
        if live.any():
            prog = np.where(detected, (t - np.nan_to_num(t_detect)) / contain_h, 0.0)
            throttle = np.clip(1 - prog, 0, 1)
            boost = (1 + k) * throttle * live
            c = c + spread * boost * c * (1 - c) * dt
            core = core + core_rate * boost * c * (1 - core) * dt
            c, core = np.clip(c, 0, 1), np.clip(core, 0, 1)

            d = d_static * (0.4 + 1.6 * c) * np.where(alerted, 1.5, 1.0)
            nd = live & ~detected & (rng.random(n) < -np.expm1(-d * dt))
            detected[nd] = True
            t_detect[nd] = t

            panic = nd & (c > 0.05) & (rng.random(n) < 0.2 + 0.6 * soph)
            reached = live & (c >= theta)
            trigger = live & (panic | reached)
            done = live & detected & ~trigger & ((t - np.nan_to_num(t_detect)) >= contain_h)
            if trigger.any():
                impact[trigger] = True
                t_impact[trigger] = t
                c_imp[trigger] = c[trigger]
                core_imp[trigger] = core[trigger]
            contained |= done
            finished |= trigger | done

        if trace:
            tr["compromise"][s] = c.mean()
            tr["core"][s] = core.mean()
            tr["breached"][s] = breached.mean()
            tr["detected"][s] = detected.mean()
            tr["impact"][s] = impact.mean()
            tr["contained"][s] = contained.mean()

    # ---- end of campaign: swarm cashes out what it holds ---------------------
    open_ = breached & ~finished
    final_blow = open_ & (c > 0.05) & (rng.random(n) < 0.7)
    impact[final_blow] = True
    t_impact[final_blow] = T[final_blow]
    c_imp[final_blow] = c[final_blow]
    core_imp[final_blow] = core[final_blow]
    foothold = open_ & ~final_blow

    # ---- infrastructure damage (0-100) --------------------------------------
    # OT/ICS adds physical damage on top of IT damage: it never makes an attack *less* harmful.
    it_destroyed = c_imp * (0.65 + 0.35 * soph) * (1 - 0.35 * edr)
    core_destroyed = core_imp * (0.7 + 0.3 * soph)
    damage = np.where(impact, 100 * (0.7 * it_destroyed + 0.3 * core_destroyed + 0.35 * ot * core_destroyed), 0.0)
    damage = np.where(contained | foothold, 100 * 0.08 * c, damage)
    damage = np.clip(damage, 0, 100)

    p_backup_loss = np.where(immut > 0, 0.03, np.clip(0.15 + 0.8 * c_imp * (1 - 0.5 * seg), 0, 1))
    backups_lost = impact & (rng.random(n) < p_backup_loss)
    exfil = np.where(breached, c * (1 - 0.4 * aid) * (0.5 + 0.5 * soph), 0.0)

    # ---- recovery time (days) -----------------------------------------------
    mult = np.where(backups_lost, v["recovery_mult_backups_lost"], v["recovery_mult"])
    mult = mult * np.where((ot > 0) & (core_destroyed > 0.2), 1.35, 1.0)
    recovery = (1.5 + 0.8 * damage) * mult * np.exp(rng.normal(0, 0.3, n))
    recovery = np.where(breached, np.clip(recovery, 0.25, 730), 0.0)

    # ---- financial impact ($M) -----------------------------------------------
    daily_rev = rev / 365
    op_loss = np.clip(damage / 100 * 1.6, 0, 1) * it_dep
    revenue_loss = daily_rev * recovery * op_loss * 0.8
    remediation = np.where(breached, emp * 0.0015 * damage / 100 + 0.02 * recovery * np.sqrt(emp / 1000), 0.0)
    ransom = np.where(backups_lost & (damage > 30), rev * 0.015 * (0.5 + soph), 0.0)
    fines = reg_fine * rev * exfil
    churn = rev * 0.08 * damage / 100 * (0.5 + 0.5 * attract)

    fraud = rng.random(n) < -np.expm1(-h_fraud * T)
    fraud_loss = np.where(fraud, np.minimum(rev * rng.uniform(0.002, 0.03, n), 250.0), 0.0)

    gross = revenue_loss + remediation + ransom + fines + churn + fraud_loss
    covered = ins * np.minimum(0.6 * gross, np.maximum(1.0, 0.05 * rev))
    net_loss = gross - covered

    buffer = v["buffer"]
    stress = net_loss / np.maximum(buffer, 1e-3)
    stress = stress * np.where((recovery > 120) & (it_dep > 0.7), 1.3, 1.0)
    p_collapse = 1 / (1 + np.maximum(stress, 1e-9) ** -4)
    collapse = rng.random(n) < p_collapse

    outcomes = pd.DataFrame({
        "breached": breached,
        "initial_vector": np.where(vector >= 0, np.array(VECTORS)[np.clip(vector, 0, 3)], "none"),
        "hours_to_breach": t_breach,
        "detected": detected,
        "hours_to_detect": t_detect - t_breach,
        "early_alert": alerted,
        "impact": impact,
        "contained": contained,
        "compromise_pct": 100 * c,
        "core_compromise_pct": 100 * core,
        "infra_damage": damage,
        "severe_damage": damage >= 25,
        "backups_lost": backups_lost,
        "data_exfil_pct": 100 * exfil,
        "recovery_days": recovery,
        "deepfake_fraud": fraud,
        "financial_loss_musd": net_loss,
        "liquidity_stress": stress,
        "collapse": collapse,
        "collapse_prob": p_collapse,     # exact P(collapse | this campaign): low-noise training target
    })
    trace_out = None
    if trace:
        trace_out = {"hours": (np.arange(steps) * dt).tolist(), **{k_: v.tolist() for k_, v in tr.items()}}
    return SimulationResult(outcomes, trace_out)


def monte_carlo(profile: dict, runs: int = 1000, seed: int | None = 0, trace: bool = True) -> dict:
    """Run ``runs`` independent campaigns against one profile and summarise them."""
    df = pd.DataFrame([profile] * runs)
    res = simulate(df, seed=seed, trace=trace)
    o = res.outcomes
    br = o[o.breached]

    def q(series, p):
        return float(np.quantile(series, p)) if len(series) else 0.0

    vectors = br.initial_vector.value_counts(normalize=True).reindex(VECTORS, fill_value=0.0)
    summary = {
        "runs": runs,
        "p_breach": float(o.breached.mean()),
        "p_severe_damage": float(o.severe_damage.mean()),
        "p_collapse": float(o.collapse.mean()),
        "p_backups_lost": float(o.backups_lost.mean()),
        "p_deepfake_fraud": float(o.deepfake_fraud.mean()),
        "p_detected_given_breach": float(br.detected.mean()) if len(br) else 0.0,
        "p_contained_given_breach": float(br.contained.mean()) if len(br) else 0.0,
        "p_impact_given_breach": float(br.impact.mean()) if len(br) else 0.0,
        "mean_damage_given_breach": float(br.infra_damage.mean()) if len(br) else 0.0,
        "median_hours_to_breach": q(br.hours_to_breach, 0.5),
        "median_hours_to_detect": q(br.hours_to_detect.dropna(), 0.5),
        "recovery_days": {"p10": q(br.recovery_days, .1), "p50": q(br.recovery_days, .5), "p90": q(br.recovery_days, .9)},
        "recovery_days_if_impact": {"p10": q(br[br.impact].recovery_days, .1), "p50": q(br[br.impact].recovery_days, .5),
                                    "p90": q(br[br.impact].recovery_days, .9)},
        "loss_musd": {"p10": q(br.financial_loss_musd, .1), "p50": q(br.financial_loss_musd, .5),
                      "p90": q(br.financial_loss_musd, .9)},
        "vector_share": {v: float(s) for v, s in vectors.items()},
        "recovery_samples": br.recovery_days.round(2).tolist(),
        "damage_samples": br.infra_damage.round(2).tolist(),
    }
    if trace:
        summary["trace"] = res.trace
    return summary
