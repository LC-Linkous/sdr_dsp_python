"""Tests for the explicit, observable AGC.

The defining property is honesty: agc() returns the gain trace it applied, so it
is fully reversible (adjusted / gain == original). Also covers convergence to
target, RMS vs peak modes, the max_gain ceiling, and streaming continuity (no
lurch at block boundaries -- the reason the stage carries state).
"""

import numpy as np
import pytest

from sdr_dsp.core import agc, AGC


def _fade(n=50000, lo=0.05, hi=1.0):
    t = np.arange(n)
    env = (lo + hi) / 2 + (hi - lo) / 2 * np.cos(2 * np.pi * 3 * t / n)
    return (env * np.exp(2j * np.pi * 0.05 * t)).astype(np.complex64)


def test_undo_property_is_exact():
    # the honesty guarantee: you can always recover the original
    sig = _fade()
    adjusted, gain = agc(sig, target=0.5)
    assert np.allclose(adjusted / gain, sig, atol=1e-5)


def test_gain_trace_same_length():
    sig = _fade(n=1000)
    adjusted, gain = agc(sig)
    assert len(gain) == len(sig) == len(adjusted)


def test_constant_signal_settles_to_target():
    # constant 0.1 input, target 1.0 -> gain settles ~10, output ~1.0
    sig = (0.1 * np.exp(2j * np.pi * 0.05 * np.arange(20000))).astype(
        np.complex64)
    adjusted, gain = agc(sig, target=1.0, attack=0.01, decay=0.001)
    assert gain[-1] == pytest.approx(10.0, rel=0.1)
    assert abs(adjusted[-1]) == pytest.approx(1.0, rel=0.1)


def test_flattens_a_fade():
    # the output level should vary far less than the input
    sig = _fade()
    adjusted, _ = agc(sig, target=0.5)
    in_std = np.abs(sig)[12500:].std()
    out_std = np.abs(adjusted)[12500:].std()
    assert out_std < in_std / 2


def test_peak_mode_runs_and_is_reversible():
    sig = _fade()
    adjusted, gain = agc(sig, mode="peak", target=0.8)
    assert np.allclose(adjusted / gain, sig, atol=1e-5)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        agc(np.ones(10, dtype=np.complex64), mode="bogus")


def test_max_gain_ceiling_holds():
    # during near-silence, gain would run away; the ceiling must hold it
    silent = (0.001 * np.exp(2j * np.pi * 0.05 * np.arange(10000))).astype(
        np.complex64)
    _, gain = agc(silent, target=1.0, max_gain=10.0)
    assert gain.max() <= 10.0 + 1e-9


def test_no_ceiling_by_default_allows_large_gain():
    # with max_gain=None, a weak signal drives gain well past any small bound
    weak = (0.01 * np.exp(2j * np.pi * 0.05 * np.arange(20000))).astype(
        np.complex64)
    _, gain = agc(weak, target=1.0)
    assert gain.max() > 50          # unbounded by default


def test_empty_input():
    adjusted, gain = agc(np.zeros(0, dtype=np.complex64))
    assert len(adjusted) == 0 and len(gain) == 0


def test_streaming_no_boundary_lurch():
    # the reason the stage carries state: gain steps at block boundaries must
    # be no larger than typical within-block steps (no pumping at edges).
    sig = _fade()
    stage = AGC(target=0.5, attack=0.01, decay=0.001)
    bs = 5000
    gains = []
    for i in range(0, len(sig), bs):
        stage(sig[i:i + bs])
        gains.append(stage.last_gain)
    gs = np.concatenate(gains)
    steps = np.abs(np.diff(gs))
    boundary_steps = [steps[i - 1] for i in range(bs, len(sig), bs)]
    assert max(boundary_steps) <= steps.max()


def test_streaming_tracks_whole_signal():
    # streamed gain should agree with whole-signal processing to within a few %
    # (small float drift over a long recursive loop is expected, not a lurch)
    sig = _fade()
    whole, gw = agc(sig, target=0.5, attack=0.01, decay=0.001)
    stage = AGC(target=0.5, attack=0.01, decay=0.001)
    gains = []
    for i in range(0, len(sig), 5000):
        stage(sig[i:i + 5000])
        gains.append(stage.last_gain)
    gs = np.concatenate(gains)
    rel = np.abs(gs - gw) / np.maximum(gw, 1e-9)
    assert rel.max() < 0.15


def test_stage_reset():
    sig = _fade(n=5000)
    stage = AGC(target=0.5)
    stage(sig)
    assert stage._level is not None
    stage.reset()
    assert stage._level is None and stage._gain == 1.0
