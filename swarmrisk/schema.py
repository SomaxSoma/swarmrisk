"""Company profile schema: every input the predictor understands.

A *profile* is a plain dict mapping feature name -> value. ``validate_profile``
fills defaults, checks types and ranges, and returns a clean copy that every
other module (simulator, model, API, CLI) can rely on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Industry parameters
#   attractiveness : how much effort a hostile swarm is willing to invest
#   it_dependency  : share of revenue that stops when IT is down
#   reg_fine       : max regulatory penalty as a fraction of revenue for a full data leak
#   ot_weight      : how much operational technology (plants, grids, devices) matters
# ---------------------------------------------------------------------------
INDUSTRIES: dict[str, dict[str, float]] = {
    "finance":       {"attractiveness": 1.00, "it_dependency": 0.95, "reg_fine": 0.06, "ot_weight": 0.05},
    "healthcare":    {"attractiveness": 0.90, "it_dependency": 0.80, "reg_fine": 0.05, "ot_weight": 0.35},
    "energy":        {"attractiveness": 0.90, "it_dependency": 0.70, "reg_fine": 0.03, "ot_weight": 0.90},
    "manufacturing": {"attractiveness": 0.75, "it_dependency": 0.75, "reg_fine": 0.02, "ot_weight": 0.75},
    "retail":        {"attractiveness": 0.70, "it_dependency": 0.85, "reg_fine": 0.04, "ot_weight": 0.10},
    "technology":    {"attractiveness": 0.85, "it_dependency": 0.95, "reg_fine": 0.04, "ot_weight": 0.05},
    "government":    {"attractiveness": 0.85, "it_dependency": 0.60, "reg_fine": 0.01, "ot_weight": 0.25},
    "logistics":     {"attractiveness": 0.70, "it_dependency": 0.85, "reg_fine": 0.02, "ot_weight": 0.45},
    "education":     {"attractiveness": 0.55, "it_dependency": 0.50, "reg_fine": 0.02, "ot_weight": 0.05},
}
INDUSTRY_NAMES = list(INDUSTRIES)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str            # "float", "int", "bool", "category"
    default: Any
    low: float | None = None
    high: float | None = None
    group: str = ""
    label: str = ""
    help: str = ""
    log_scale: bool = False


FEATURES: list[Feature] = [
    # -- Organisation ------------------------------------------------------
    Feature("industry", "category", "technology", group="Organisation", label="Industry",
            help="Sector the company operates in."),
    Feature("employees", "int", 1500, 10, 500_000, "Organisation", "Employees",
            "Headcount. Drives the phishing / social-engineering surface.", log_scale=True),
    Feature("annual_revenue_musd", "float", 300.0, 0.5, 500_000, "Organisation", "Annual revenue ($M)",
            "Yearly revenue in millions of USD.", log_scale=True),
    Feature("profit_margin", "float", 0.08, -0.5, 0.6, "Organisation", "Profit margin",
            "Net margin as a fraction (0.08 = 8%)."),
    Feature("cash_reserve_months", "float", 4.0, 0.0, 60.0, "Organisation", "Cash reserves (months)",
            "How many months of operating costs the company can cover from cash."),
    Feature("cyber_insurance", "bool", True, group="Organisation", label="Cyber insurance",
            help="Has an active cyber-insurance policy."),
    Feature("prior_incidents_3y", "int", 1, 0, 100, "Organisation", "Incidents (last 3y)",
            "Significant security incidents in the last three years."),

    # -- Attack surface ------------------------------------------------------
    Feature("internet_facing_services", "int", 40, 1, 20_000, "Attack surface", "Internet-facing services",
            "Public endpoints: web apps, VPNs, mail, exposed APIs.", log_scale=True),
    Feature("ai_agent_integrations", "int", 5, 0, 2_000, "Attack surface", "Internal AI agents",
            "Autonomous AI agents / LLM tools that can act on internal systems "
            "(prompt-injection and agent-hijack surface).", log_scale=True),
    Feature("api_exposure", "float", 0.4, 0.0, 1.0, "Attack surface", "API exposure",
            "How much of the business logic is reachable through APIs (0-1)."),
    Feature("third_party_vendors", "int", 80, 0, 50_000, "Attack surface", "Third-party vendors",
            "Suppliers with network or data access (supply-chain surface).", log_scale=True),
    Feature("cloud_share", "float", 0.5, 0.0, 1.0, "Attack surface", "Cloud share",
            "Fraction of workloads running in the cloud (0-1)."),
    Feature("legacy_system_share", "float", 0.25, 0.0, 1.0, "Attack surface", "Legacy systems",
            "Fraction of systems that are end-of-life or hard to patch (0-1)."),
    Feature("ot_ics_present", "bool", False, group="Attack surface", label="OT / ICS present",
            help="Operates industrial control systems, plant equipment or medical devices."),

    # -- Defences ------------------------------------------------------------
    Feature("security_staff", "int", 6, 0, 10_000, "Defences", "Security staff (FTE)",
            "Full-time security people.", log_scale=True),
    Feature("security_budget_pct", "float", 8.0, 0.0, 40.0, "Defences", "Security budget (% of IT)",
            "Security spend as a percentage of the IT budget."),
    Feature("mfa_coverage", "float", 0.6, 0.0, 1.0, "Defences", "MFA coverage",
            "Fraction of accounts protected by phishing-resistant MFA (0-1)."),
    Feature("patch_latency_days", "float", 30.0, 1.0, 365.0, "Defences", "Patch latency (days)",
            "Typical time to patch a critical vulnerability."),
    Feature("edr_coverage", "float", 0.6, 0.0, 1.0, "Defences", "EDR coverage",
            "Fraction of endpoints and servers with endpoint detection & response (0-1)."),
    Feature("network_segmentation", "float", 0.4, 0.0, 1.0, "Defences", "Network segmentation",
            "How well critical systems are isolated from each other (0-1)."),
    Feature("zero_trust_maturity", "float", 0.3, 0.0, 1.0, "Defences", "Zero-trust maturity",
            "Least-privilege, continuous verification of identities and devices (0-1)."),
    Feature("ai_threat_detection", "float", 0.3, 0.0, 1.0, "Defences", "AI threat detection",
            "Maturity of AI/behavioural anomaly detection that can spot machine-speed attacks (0-1)."),
    Feature("soc_24x7", "bool", False, group="Defences", label="24/7 SOC",
            help="Security operations centre monitoring around the clock."),
    Feature("security_training", "float", 0.5, 0.0, 1.0, "Defences", "Security awareness",
            "Staff resilience to phishing and social engineering (0-1)."),
    Feature("deepfake_verification", "bool", False, group="Defences", label="Deepfake verification",
            help="Out-of-band verification for payments and voice/video requests."),

    # -- Recovery ------------------------------------------------------------
    Feature("immutable_backups", "bool", False, group="Recovery", label="Immutable backups",
            help="Offline / immutable backups that attackers cannot encrypt or delete."),
    Feature("backup_frequency_hours", "float", 24.0, 1.0, 720.0, "Recovery", "Backup interval (hours)",
            "Hours between backups."),
    Feature("restore_test_days", "float", 180.0, 1.0, 1000.0, "Recovery", "Days since restore test",
            "Days since a full restore from backup was last tested."),
    Feature("ir_maturity", "float", 0.4, 0.0, 1.0, "Recovery", "Incident-response maturity",
            "Tested IR plan, playbooks, retained responders (0-1)."),
    Feature("dr_site", "bool", False, group="Recovery", label="Disaster-recovery site",
            help="Warm/hot standby environment that can take over operations."),

    # -- Threat scenario -----------------------------------------------------
    Feature("swarm_size", "int", 200, 1, 10_000, "Threat scenario", "Swarm size (agents)",
            "Number of coordinated autonomous AI attack agents.", log_scale=True),
    Feature("attacker_sophistication", "float", 0.7, 0.0, 1.0, "Threat scenario", "Attacker sophistication",
            "Skill of the swarm: 0 = commodity scripts, 1 = state-level (0-1)."),
    Feature("campaign_days", "int", 30, 1, 90, "Threat scenario", "Campaign length (days)",
            "How long the swarm keeps attacking."),
]

FEATURE_MAP: dict[str, Feature] = {f.name: f for f in FEATURES}
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
SCENARIO_FEATURES = ["swarm_size", "attacker_sophistication", "campaign_days"]


class ProfileError(ValueError):
    """Raised when a profile contains unknown keys or invalid values."""


def default_profile() -> dict[str, Any]:
    return {f.name: f.default for f in FEATURES}


def _to_bool(name: str, v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in {"true", "yes", "1", "y", "on"}:
        return True
    if isinstance(v, str) and v.strip().lower() in {"false", "no", "0", "n", "off"}:
        return False
    raise ProfileError(f"'{name}' must be true/false, got {v!r}")


def validate_profile(raw: dict[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    """Return a complete, type-checked profile.

    Missing fields take their defaults. With ``strict=True`` unknown fields and
    out-of-range values raise ``ProfileError``; with ``strict=False`` unknown
    fields are ignored and numbers are clipped into range.
    """
    raw = dict(raw or {})
    unknown = set(raw) - set(FEATURE_MAP)
    if unknown and strict:
        raise ProfileError(f"Unknown field(s): {', '.join(sorted(unknown))}")

    out: dict[str, Any] = {}
    for f in FEATURES:
        v = raw.get(f.name, f.default)
        if v is None or (isinstance(v, str) and v.strip() == "" and f.kind != "category"):
            v = f.default
        if f.kind == "category":
            v = str(v).strip().lower()
            if v not in INDUSTRIES:
                raise ProfileError(f"'{f.name}' must be one of {INDUSTRY_NAMES}, got {v!r}")
        elif f.kind == "bool":
            v = _to_bool(f.name, v)
        else:
            try:
                num = float(v)
            except (TypeError, ValueError):
                raise ProfileError(f"'{f.name}' must be a number, got {v!r}") from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ProfileError(f"'{f.name}' must be a finite number")
            if num < f.low or num > f.high:
                if strict:
                    raise ProfileError(f"'{f.name}' must be between {f.low:g} and {f.high:g}, got {num:g}")
                num = min(max(num, f.low), f.high)
            v = int(round(num)) if f.kind == "int" else num
        out[f.name] = v
    return out


def schema_json() -> dict[str, Any]:
    """Serializable schema for the web UI."""
    return {
        "industries": INDUSTRY_NAMES,
        "features": [
            {
                "name": f.name, "kind": f.kind, "default": f.default, "low": f.low, "high": f.high,
                "group": f.group, "label": f.label, "help": f.help, "log_scale": f.log_scale,
            }
            for f in FEATURES
        ],
    }
