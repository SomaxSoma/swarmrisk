"""Command-line interface.

    python -m swarmrisk train [--samples 300000]
    python -m swarmrisk predict examples/regional_hospital.json [--simulate 2000] [--json]
    python -m swarmrisk serve [--port 8000]
    python -m swarmrisk calibrate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(a):
    from .model import train
    train(n_samples=a.samples, seed=a.seed, fidelity_companies=a.fidelity, fidelity_runs=a.fidelity_runs)


def _cmd_predict(a):
    from .predictor import Predictor
    from .schema import ProfileError
    profile = json.loads(Path(a.profile).read_text()) if a.profile else {}
    profile.pop("_name", None)
    profile.pop("_description", None)
    try:
        rep = Predictor().report(profile, simulate_runs=a.simulate)
    except ProfileError as e:
        sys.exit(f"Invalid profile: {e}")
    if a.json:
        rep.pop("simulation", None) if not a.simulate else None
        if "simulation" in rep:
            rep["simulation"].pop("recovery_samples", None)
            rep["simulation"].pop("damage_samples", None)
            rep["simulation"].pop("trace", None)
        print(json.dumps(rep, indent=2, default=str))
        return
    pr = rep["prediction"]
    bar = "=" * 78
    print(bar)
    print(f" SWARMRISK ASSESSMENT   grade {pr['grade']}   resilience {pr['resilience_score']:.0f}/100")
    print(bar)
    print(f" P(breach)               {pr['p_breach']:6.1%}   [{pr['levels']['breach']}]")
    print(f" P(severe infra damage)  {pr['p_severe_damage']:6.1%}   [{pr['levels']['severe_damage']}]")
    print(f" P(company collapse)     {pr['p_collapse']:6.1%}   [{pr['levels']['collapse']}]")
    print(f" Damage if breached      {pr['expected_damage_given_breach']:6.1f}% of infrastructure")
    r = pr["recovery_days"]
    print(f" Recovery if breached    {r['p50']:.1f} days (80% range {r['p10']:.1f} - {r['p90']:.1f})")
    print(f" Net loss if breached    ${pr['loss_musd']['p50']:.2f}M typical, ${pr['loss_musd']['p90']:.2f}M bad case")
    print()
    print(" " + pr["summary"])
    print("\n Top risk drivers (vs a typical company):")
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    for d in rep["drivers"][:6]:
        sign = "+" if d["impact"] > 0 else "-"
        print(f"   {sign} {d['label']:<30} {fmt(d['value']):>10}  (typical {fmt(d['typical'])})  {d['impact']:+.1f} pts")
    print("\n Recommended actions:")
    for x in rep["recommendations"]["actions"][:6]:
        print(f"   * {x['title']:<52} -{x['risk_reduction']:.1f} pts  cost {x['cost']}")
    pl = rep["recommendations"]
    if pl["plan_result"]:
        print(f"\n Combined plan ({len(pl['plan'])} actions) -> grade {pl['plan_result']['grade']}, "
              f"resilience {pl['plan_result']['resilience_score']:.0f}/100")
    if a.simulate:
        s = rep["simulation"]
        print(f"\n Monte-Carlo check ({s['runs']} simulated campaigns): breach {s['p_breach']:.1%}, "
              f"severe {s['p_severe_damage']:.1%}, collapse {s['p_collapse']:.1%}, "
              f"median recovery {s['recovery_days']['p50']:.1f} days")
    print(bar)


def _cmd_serve(a):
    import uvicorn
    print(f"SwarmRisk dashboard: http://{a.host}:{a.port}")
    uvicorn.run("swarmrisk.api:app", host=a.host, port=a.port, log_level="info")


def _cmd_calibrate(a):
    from .sampling import sample_companies
    from .simulator import simulate
    df = sample_companies(a.samples, seed=a.seed)
    o = simulate(df, seed=a.seed + 1).outcomes
    print(o[["breached", "detected", "impact", "contained", "severe_damage", "backups_lost",
             "deepfake_fraud", "collapse"]].mean().round(3).to_string())
    b = o[o.breached]
    print(b[["infra_damage", "recovery_days", "financial_loss_musd"]].quantile([.1, .5, .9]).round(2).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="swarmrisk", description="AI-swarm attack risk predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="simulate data and train the models")
    t.add_argument("--samples", type=int, default=600_000)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--fidelity", type=int, default=200, help="companies for the Monte-Carlo fidelity check (0=skip)")
    t.add_argument("--fidelity-runs", type=int, default=500)
    t.set_defaults(fn=_cmd_train)

    p = sub.add_parser("predict", help="assess a company profile (JSON file)")
    p.add_argument("profile", nargs="?", help="path to a JSON profile; omitted = default company")
    p.add_argument("--simulate", type=int, default=0, help="also run N Monte-Carlo campaigns")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.set_defaults(fn=_cmd_predict)

    s = sub.add_parser("serve", help="run the web dashboard + REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)

    c = sub.add_parser("calibrate", help="print simulator base rates for a random population")
    c.add_argument("--samples", type=int, default=20_000)
    c.add_argument("--seed", type=int, default=1)
    c.set_defaults(fn=_cmd_calibrate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
