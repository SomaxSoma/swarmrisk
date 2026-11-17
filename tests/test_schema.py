import pytest

from swarmrisk.schema import FEATURE_NAMES, ProfileError, default_profile, validate_profile


def test_defaults_fill_everything():
    p = validate_profile({})
    assert set(p) == set(FEATURE_NAMES)
    assert p == validate_profile(default_profile())


def test_type_coercion():
    p = validate_profile({"employees": "250", "soc_24x7": "yes", "industry": " Finance ", "mfa_coverage": 1})
    assert p["employees"] == 250 and isinstance(p["employees"], int)
    assert p["soc_24x7"] is True
    assert p["industry"] == "finance"
    assert p["mfa_coverage"] == 1.0


@pytest.mark.parametrize("bad", [
    {"industry": "space-pirates"},
    {"mfa_coverage": 1.5},
    {"employees": -3},
    {"swarm_size": "many"},
    {"dr_site": "maybe"},
    {"not_a_field": 1},
    {"cash_reserve_months": float("nan")},
])
def test_rejects_invalid(bad):
    with pytest.raises(ProfileError):
        validate_profile(bad)


def test_non_strict_clips_and_ignores():
    p = validate_profile({"mfa_coverage": 7, "unknown": 1}, strict=False)
    assert p["mfa_coverage"] == 1.0
    assert "unknown" not in p
