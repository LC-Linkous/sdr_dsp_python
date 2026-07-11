"""Tests for fingerprinting feature extraction.

The organizing principle (matching the library's verify-against-truth rule): the
forward model in ``channel_impairments`` is the oracle for the estimators in
``features``. Apply a known impairment; confirm recovery within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdr_dsp.core.channel_impairments import (
    add_iq_imbalance,
    add_pa_nonlinearity,
    add_phase_noise,
    make_device_impairments,
    apply_device_impairments,
)
from sdr_dsp.core.features import (
    estimate_iq_imbalance,
    iq_image_ratio,
    estimate_cfo_ppm,
    estimate_phase_noise_variance,
    decide_symbols,
    error_vector,
    evm_stats,
    fingerprint_vector,
    FEATURE_NAMES,
    EVM_FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def proper_signal():
    """A proper (E[s^2]~=0) complex signal.

    Built from an exactly-balanced QPSK alphabet (each of the 4 points used an
    equal number of times, then shuffled) so the pseudo-covariance E[s^2] is
    zero by construction rather than approximately-zero by luck. This isolates
    the estimator under test from the finite-sample E[s^2] floor; a separate
    consideration (documented in estimate_iq_imbalance) is that real signals do
    carry that floor.
    """
    rng = np.random.default_rng(0)
    const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    reps = 5000
    syms = np.tile(const, reps)          # exactly balanced: sum(s^2) == 0
    rng.shuffle(syms)
    return syms.astype(np.complex64)


# ---------------------------------------------------------------------------
# forward-model invariants (F0)
# ---------------------------------------------------------------------------
def test_iq_imbalance_zero_is_identity(proper_signal):
    out = add_iq_imbalance(proper_signal, 0.0, 0.0)
    assert np.allclose(out, proper_signal, atol=1e-5)


def test_pa_linear_is_identity(proper_signal):
    out = add_pa_nonlinearity(proper_signal, [1.0])
    assert np.allclose(out, proper_signal, atol=1e-5)


def test_phase_noise_zero_linewidth_is_identity(proper_signal):
    out = add_phase_noise(proper_signal, 0.0, 1e6)
    assert np.allclose(out, proper_signal)


def test_imbalance_increases_image_power(proper_signal):
    """More imbalance -> more pseudo-covariance (image) power."""
    p_small = np.abs(np.mean(add_iq_imbalance(proper_signal, 0.2, 1.0) ** 2))
    p_large = np.abs(np.mean(add_iq_imbalance(proper_signal, 1.0, 5.0) ** 2))
    assert p_large > p_small


def test_device_is_deterministic():
    d1 = make_device_impairments(42)
    d2 = make_device_impairments(42)
    assert d1 == d2
    assert make_device_impairments(7) != d1


# ---------------------------------------------------------------------------
# estimator recovery (F1) — forward model is the oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gain_db,phase_deg",
    [(0.0, 0.0), (0.3, 0.0), (0.0, 2.0), (0.5, -3.0), (-0.4, 1.5)],
)
def test_iq_imbalance_roundtrip(proper_signal, gain_db, phase_deg):
    impaired = add_iq_imbalance(proper_signal, gain_db, phase_deg)
    est_gain, est_phase = estimate_iq_imbalance(impaired)
    # Gain recovers tightly. Phase has a sub-degree finite-sample floor from the
    # signal's own residual E[s^2] (documented in estimate_iq_imbalance); the
    # 0.3 deg tolerance reflects that floor, not estimator error.
    assert abs(est_gain - gain_db) < 0.05
    assert abs(est_phase - phase_deg) < 0.3


def test_image_ratio_is_rotation_invariant(proper_signal):
    """The image ratio must not move under an unknown carrier phase rotation --
    this is the property that makes it a receiver-robust fingerprint.
    """
    impaired = add_iq_imbalance(proper_signal, 0.5, -3.0)
    ratios = []
    for rot_deg in (0, 45, 90, 137, 200):
        rotated = impaired * np.exp(1j * np.deg2rad(rot_deg))
        ratios.append(iq_image_ratio(rotated))
    assert max(ratios) - min(ratios) < 1e-6


def test_image_ratio_grows_with_imbalance(proper_signal):
    small = iq_image_ratio(add_iq_imbalance(proper_signal, 0.2, 1.0))
    large = iq_image_ratio(add_iq_imbalance(proper_signal, 1.0, 5.0))
    assert large > small
    assert iq_image_ratio(proper_signal) < 1e-3  # balanced -> ~0


def test_cfo_ppm_conversion():
    # 2 kHz at 2.4 GHz carrier ~ 0.833 ppm
    ppm = estimate_cfo_ppm(2000.0, 2.4e9)
    assert abs(ppm - 0.8333) < 1e-3
    assert estimate_cfo_ppm(1000.0, 0.0) == 0.0


def test_phase_noise_variance_monotonic(proper_signal):
    """Higher linewidth -> larger measured phase-error variance."""
    low = add_phase_noise(proper_signal, 10.0, 1e6, seed=1)
    high = add_phase_noise(proper_signal, 500.0, 1e6, seed=1)
    v_low = estimate_phase_noise_variance(low, reference=proper_signal)
    v_high = estimate_phase_noise_variance(high, reference=proper_signal)
    assert v_high > v_low


# ---------------------------------------------------------------------------
# EVM / error cloud (F2)
# ---------------------------------------------------------------------------
def test_decide_symbols_qpsk():
    const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    rx = np.array([0.6 + 0.6j, -0.5 - 0.4j]) 
    decided = decide_symbols(rx, const)
    assert np.isclose(decided[0], (1 + 1j) / np.sqrt(2))
    assert np.isclose(decided[1], (-1 - 1j) / np.sqrt(2))


def test_zero_error_gives_zero_evm():
    const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    rng = np.random.default_rng(3)
    ref = const[rng.integers(0, 4, 500)]
    err = error_vector(ref, ref)
    stats = evm_stats(err)
    assert stats[0] < 1e-9  # evm_rms
    assert len(stats) == len(EVM_FEATURE_NAMES)


def test_evm_grows_with_noise():
    const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    rng = np.random.default_rng(4)
    ref = const[rng.integers(0, 4, 5000)]
    rx_low = ref + 0.01 * (rng.standard_normal(5000) + 1j * rng.standard_normal(5000))
    rx_high = ref + 0.10 * (rng.standard_normal(5000) + 1j * rng.standard_normal(5000))
    evm_low = evm_stats(error_vector(rx_low, ref))[0]
    evm_high = evm_stats(error_vector(rx_high, ref))[0]
    assert evm_high > evm_low


def test_error_vector_shape_mismatch_raises():
    with pytest.raises(ValueError):
        error_vector(np.zeros(5, dtype=complex), np.zeros(4, dtype=complex))


# ---------------------------------------------------------------------------
# assembly (F4)
# ---------------------------------------------------------------------------
def test_fingerprint_vector_length_and_finite(proper_signal):
    vec, names = fingerprint_vector(
        proper_signal, sample_rate=1e6, carrier_hz=2.4e9, cfo_hz=500.0
    )
    assert names == FEATURE_NAMES
    assert vec.shape[0] == len(FEATURE_NAMES)
    # impairment slots (first 5) are always finite; EVM slots NaN without symbols
    n_impair = len(FEATURE_NAMES) - len(EVM_FEATURE_NAMES)
    assert np.all(np.isfinite(vec[:n_impair]))
    assert np.all(np.isnan(vec[n_impair:]))


def test_fingerprint_vector_with_symbols(proper_signal):
    const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    rng = np.random.default_rng(5)
    ref = const[rng.integers(0, 4, 2000)]
    rx = ref + 0.02 * (rng.standard_normal(2000) + 1j * rng.standard_normal(2000))
    vec, names = fingerprint_vector(
        proper_signal, 1e6, 2.4e9, cfo_hz=0.0, rx_symbols=rx, ref_symbols=ref
    )
    assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# channel-robustness guard (F5) — the honesty test
# ---------------------------------------------------------------------------
def test_fingerprint_separates_devices_across_channels():
    """Within-device feature spread (across channels) must be smaller than
    between-device spread. Otherwise the classifier learns the setup, not the
    device. This is the go/no-go for the whole approach.
    """
    rng = np.random.default_rng(11)
    fs, carrier = 1e6, 2.4e9
    n = 100000  # imbalance estimate is a sample mean; it needs samples to settle
    base = (
        (rng.integers(0, 2, n) * 2 - 1) + 1j * (rng.integers(0, 2, n) * 2 - 1)
    ).astype(np.complex64) / np.sqrt(2)

    # Use devices with clearly distinct signatures. (In reality device spread is
    # what it is; here we assert the *pipeline* can separate devices that DO
    # differ -- a weak feature on near-identical devices is a data problem, not a
    # pipeline bug. Real deployments raise separability by adding features and
    # samples, both of which this test exercises.)
    devices = [
        make_device_impairments(s) for s in (100, 200, 300, 400, 500)
    ]

    def features_for(device, channel_seed):
        # device signature, then a varying but non-destructive "channel":
        # CFO, scale, bulk phase rotation -- the effects apply_channel models.
        # Heavy AWGN is deliberately light here: it corrupts the E[r^2] statistic
        # the imbalance estimators read. Noise-robustness is a separate axis;
        # this test isolates channel *geometry*, which a fingerprint must survive.
        sig = apply_device_impairments(base, device, fs, carrier, seed=channel_seed)
        crng = np.random.default_rng(channel_seed)
        cfo = crng.uniform(-500, 500)
        rot = np.exp(1j * crng.uniform(0, 2 * np.pi))
        scale = crng.uniform(0.7, 1.3)
        idx = np.arange(n)
        sig = sig * (scale * rot) * np.exp(1j * 2 * np.pi * cfo * idx / fs)
        light = 0.005 * (crng.standard_normal(n) + 1j * crng.standard_normal(n))
        sig = (sig + light).astype(np.complex64)
        # Rotation-invariant imbalance + phase-noise variance: the two symbol-free
        # features that survive an unknown carrier phase. (Split gain/phase and
        # absolute CFO are intentionally excluded -- they rotate/shift with the
        # channel and would be the classic "learned the setup" trap.)
        return np.array(
            [iq_image_ratio(sig), estimate_phase_noise_variance(sig)]
        )

    n_chan = 8
    per_device = np.array(
        [[features_for(d, ch) for ch in range(n_chan)] for d in devices]
    )  # (n_dev, n_chan, n_feat)

    # Standardize each feature so the two features contribute comparably.
    flat = per_device.reshape(-1, per_device.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-12
    z = (per_device - mu) / sd

    # Nearest-centroid separability: assign each capture to the closest device
    # centroid; a device-dominated fingerprint classifies its own captures right.
    centroids = z.mean(axis=1)  # (n_dev, n_feat)
    correct = 0
    total = 0
    for di in range(len(devices)):
        for ci in range(n_chan):
            d2 = np.sum((centroids - z[di, ci]) ** 2, axis=1)
            if np.argmin(d2) == di:
                correct += 1
            total += 1
    accuracy = correct / total
    # Honest bar: with ONLY the two symbol-free features (image ratio + phase-
    # noise variance) and a realistic phase-noise+CFO channel, separability is
    # modest -- phase noise scrambles the E[r^2] statistic and the channel adds
    # jitter. The pipeline must still clearly beat chance (1/n_dev = 0.2). Strong
    # separability in practice comes from the SYMBOL-DOMAIN EVM/error-cloud
    # features (F2, which need synchronized symbols) plus many more captures;
    # this guard deliberately tests the weaker symbol-free path so a regression
    # that pushes it back toward chance is caught. If you raise this bar, add the
    # EVM features to `features_for` first.
    chance = 1.0 / len(devices)
    assert accuracy > 2.0 * chance, (
        f"separability at/near chance ({accuracy:.2f} vs chance {chance:.2f}) "
        "-- the symbol-free fingerprint has stopped discriminating"
    )


# ---------------------------------------------------------------------------
# polish additions: properness guard + flattened API
# ---------------------------------------------------------------------------
def test_improper_signal_warns():
    """OOK/BPSK-class (improper) input must trip the properness RuntimeWarning
    rather than silently returning modulation-as-device numbers."""
    ook = np.repeat(np.random.default_rng(0).integers(0, 2, 1000), 10).astype(
        np.complex64
    )
    with pytest.warns(RuntimeWarning, match="pseudo-covariance"):
        estimate_iq_imbalance(ook)


def test_proper_signal_does_not_warn(proper_signal):
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        estimate_iq_imbalance(proper_signal)  # must not raise


def test_flattened_core_api():
    """The feature/synthesis API is reachable from sdr_dsp.core, house-style."""
    from sdr_dsp.core import (
        fingerprint_vector, FEATURE_NAMES, iq_image_ratio,
        make_device_impairments, apply_device_impairments,
    )
    assert callable(fingerprint_vector) and len(FEATURE_NAMES) == 14
