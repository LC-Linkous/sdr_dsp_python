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

    A handful of older tests draw from the global np.random without seeding
    (test_util, test_sinks, test_io_and_sources). In isolation they pass, but
    in a full run the global state is whatever prior tests left it, so a
    tolerance-tight assertion could occasionally land on an unlucky draw --
    an order-dependent flake that clears on rerun. Seeding here makes every
    run reproducible regardless of order or selection.

    This is a safety net, not a license: new tests should use an explicit
    np.random.default_rng(seed) with their own generator, and migrating the
    three legacy tests to that pattern is tracked in TODO. Seeding globally
    fixes the flake; explicit generators make the intent local and obvious.
    """
    np.random.seed(1234)
    yield
