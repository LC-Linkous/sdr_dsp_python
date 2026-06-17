"""Tests for util: dB conversion and explicit normalization."""

import numpy as np

from sdr_dsp.core import to_db, from_db, normalize


def test_db_roundtrip_power():
    for v in (1.0, 0.25, 100.0, 1e-6):
        assert abs(from_db(to_db(v)) - v) < 1e-9


def test_db_roundtrip_amplitude():
    for v in (1.0, 0.5, 2.0):
        assert abs(from_db(to_db(v, power=False), power=False) - v) < 1e-9


def test_to_db_handles_zero():
    # zero -> very negative, not -inf or NaN
    d = to_db(0.0)
    assert np.isfinite(d) and d < -150


def test_normalize_peak():
    x = np.array([1 + 0j, 3 + 4j], dtype=np.complex64)  # peak mag 5
    out = normalize(x, mode="peak", target=1.0)
    assert abs(np.abs(out).max() - 1.0) < 1e-6


def test_normalize_rms():
    x = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
    out = normalize(x, mode="rms", target=1.0)
    rms = np.sqrt(np.mean(np.abs(out) ** 2))
    assert abs(rms - 1.0) < 1e-6


def test_normalize_none_is_identity():
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    assert np.array_equal(normalize(x, mode="none"), x)


def test_normalize_empty_and_zero():
    assert len(normalize(np.array([], dtype=np.complex64))) == 0
    z = np.zeros(10, dtype=np.complex64)
    assert np.array_equal(normalize(z), z)   # nothing to scale, unchanged


def test_normalize_does_not_mutate_input():
    x = np.array([2 + 0j, 4 + 0j], dtype=np.complex64)
    orig = x.copy()
    normalize(x, mode="peak")
    assert np.array_equal(x, orig)


def test_normalize_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        normalize(np.ones(5, dtype=np.complex64), mode="bogus")
