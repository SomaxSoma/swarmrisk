import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swarmrisk.model import train  # noqa: E402
from swarmrisk.schema import validate_profile  # noqa: E402


@pytest.fixture(scope="session")
def small_bundle(tmp_path_factory):
    """A quickly trained model so tests never depend on (or overwrite) models/."""
    out = tmp_path_factory.mktemp("model")
    return train(n_samples=40_000, seed=3, fidelity_companies=0, verbose=False, out_dir=out)


def load_example(name: str) -> dict:
    data = json.loads((ROOT / "examples" / f"{name}.json").read_text())
    return validate_profile({k: v for k, v in data.items() if not k.startswith("_")})


@pytest.fixture
def hospital():
    return load_example("regional_hospital")


@pytest.fixture
def bank():
    return load_example("global_bank")
