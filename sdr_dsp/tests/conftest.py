import sys
from pathlib import Path

# make src/ and tests/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_global_rng():
    """Seed the legacy global np.random before EVERY test.

    A safety net, not a license. Every test now draws from its own
    np.random.default_rng(seed) (the last three legacy users of the global
    state were migrated in patch 0014; `grep np.random. tests/` finds only
    this line). The fixture stays so that if a global draw ever creeps back
    in -- in a test, a helper, or library code under test -- the run is still
    reproducible regardless of test order or selection, instead of flaking
    on whatever state a prior test left behind.
    """
    np.random.seed(1234)
    yield
