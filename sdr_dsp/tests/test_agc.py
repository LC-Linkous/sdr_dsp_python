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


def test_peak_mode_is_a_different_tracker_than_rms():
    # "peak" and "rms" must not be the same code path (they once were: the
    # mode was validated and then ignored). A peak detector jumps to a spike
    # instantly, so after a short burst its gain is clamped down HARDER than
    # the one-pole rms tracker's, and the two traces differ.
    sig = np.ones(30000, dtype=np.complex64)
    sig[2500:2510] = 5.0
    _, g_rms = agc(sig, mode="rms", target=1.0)
    _, g_peak = agc(sig, mode="peak", target=1.0)
    assert not np.allclose(g_rms, g_peak)
    assert g_peak[2510] < g_rms[2510]
    # the peak detector holds the burst, so it stays clamped longer ...
    assert g_peak[3500] < g_rms[3500]
    # ... but both release and settle back to target on the long constant tail
    assert g_peak[-1] == pytest.approx(1.0, rel=0.05)
    assert g_rms[-1] == pytest.approx(1.0, rel=0.05)


def _first_settled(out, target, after, tol=0.1):
    m = np.abs(out[after:])
    return int(np.argmax(np.abs(m - target) < tol * target))


def test_fast_attack_slow_decay_and_no_collapse_after_a_step_up():
    # The two rates are the AGC's contract: a level step UP is caught at the
    # attack rate (fast), a fade is followed at the decay rate (slow). With
    # attack 10x decay the step-up must settle several times faster than the
    # step-down -- the level tracker's attack/decay assignment was never
    # pinned, and swapping them passed the whole suite.
    # Also pins the fix for output collapse: the output after a +20 dB step
    # must come DOWN to target and stay there, never undershoot. An earlier
    # loop smoothed the OUTPUT level, which lagged the gain it was steering,
    # and drove the output to ~0 for thousands of samples after every step up.
    n = 20000
    step_up = np.concatenate([np.full(n, 0.1), np.full(n, 1.0)]).astype(np.complex64)
    step_dn = np.concatenate([np.full(n, 1.0), np.full(n, 0.1)]).astype(np.complex64)
    out_up, _ = agc(step_up, target=0.5, attack=0.01, decay=0.001)
    out_dn, _ = agc(step_dn, target=0.5, attack=0.01, decay=0.001)
    t_up = _first_settled(out_up, 0.5, n)
    t_dn = _first_settled(out_dn, 0.5, n)
    assert t_up < 1000, f"step up took {t_up} samples to settle"
    assert t_dn > 4 * t_up, f"attack ({t_up}) not clearly faster than decay ({t_dn})"
    # no undershoot: once the output comes back down to target it stays there
    settled = np.abs(out_up[n + t_up:])
    assert settled.min() > 0.45, f"output collapsed to {settled.min():.3f} after step up"
    assert np.abs(out_dn[n:]).max() <= 0.5 + 1e-6      # monotone recovery on fade


def test_digital_silence_holds_gain_and_never_produces_nan():
    # regression: with max_gain=None, a run of exact zeros drove `level` to
    # the eps floor, err to ~1e12, gain to inf, and every later sample to NaN
    # -- including the signal AFTER the gap. The loop must hold through
    # silence and resume cleanly.
    sig = np.concatenate([np.full(20000, 0.5), np.zeros(40000),
                          np.full(2000, 0.5)]).astype(np.complex64)
    with np.errstate(all="raise"):              # any overflow is a failure
        adjusted, gain = agc(sig, target=1.0, attack=0.01, decay=0.001)
    assert np.isfinite(gain).all() and np.isfinite(adjusted).all()
    assert gain[20000] == gain[59999]           # held, not run away
    assert np.all(adjusted[20000:60000] == 0)   # zeros in, zeros out
    assert abs(adjusted[-1]) == pytest.approx(1.0, rel=0.1)   # tracks again


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


def _stream(sig, stage, block_sizes):
    """Push sig through stage in blocks of the given (cycling) sizes."""
    gains, outs = [], []
    i = 0
    k = 0
    while i < len(sig):
        bs = block_sizes[k % len(block_sizes)]
        outs.append(stage(sig[i:i + bs]))
        gains.append(stage.last_gain)
        i += bs
        k += 1
    return np.concatenate(outs), np.concatenate(gains)


def test_streaming_no_boundary_lurch():
    # the reason the stage carries state: the gain step ACROSS a block
    # boundary must be no larger than the largest step the whole-record loop
    # takes anywhere near it. (The previous form of this test compared the
    # boundary steps against the max over ALL streamed steps, a set that
    # contains them -- it could not fail.)
    sig = _fade()
    _, gw = agc(sig, target=0.5, attack=0.01, decay=0.001)
    ref_steps = np.abs(np.diff(gw))
    stage = AGC(target=0.5, attack=0.01, decay=0.001)
    bs = 5000
    _, gs = _stream(sig, stage, [bs])
    for b in range(bs, len(sig), bs):
        boundary = abs(gs[b] - gs[b - 1])
        local_max = ref_steps[b - 50:b + 50].max()
        assert boundary <= local_max * 1.01 + 1e-12, f"lurch at block edge {b}"


@pytest.mark.parametrize("mode", ["rms", "peak"])
@pytest.mark.parametrize("block_sizes", [[5000], [1], [7, 1000, 3, 64, 999]])
def test_streaming_equals_whole_signal_exactly(block_sizes, mode):
    # the stage is "the function plus a memory of the loop state", so streamed
    # processing must reproduce the whole-record run sample for sample -- same
    # arithmetic in the same order, there is no float drift to forgive. A
    # 15% tolerance here once hid a bug where the stage carried |last output|
    # instead of the smoothed level and diverged by ~5% at every block edge.
    sig = _fade(n=20000)
    whole, gw = agc(sig, mode=mode, target=0.5, attack=0.01, decay=0.001)
    stage = AGC(mode=mode, target=0.5, attack=0.01, decay=0.001)
    streamed, gs = _stream(sig, stage, block_sizes)
    np.testing.assert_array_equal(gs, gw)
    np.testing.assert_array_equal(streamed, whole)


def test_streaming_equals_whole_on_a_level_step():
    # a 26 dB step mid-record is where a bad carried state shows up most
    rng = np.random.default_rng(7)
    n = 40000
    env = np.concatenate([np.full(n // 2, 0.1), np.full(n // 2, 2.0)])
    sig = (env * np.exp(1j * rng.uniform(0, 2 * np.pi, n))).astype(np.complex64)
    whole, gw = agc(sig, target=0.5)
    streamed, gs = _stream(sig, AGC(target=0.5), [500])
    np.testing.assert_array_equal(gs, gw)
    np.testing.assert_array_equal(streamed, whole)


def test_stage_reset():
    sig = _fade(n=5000)
    stage = AGC(target=0.5)
    stage(sig)
    assert stage._level is not None
    stage.reset()
    assert stage._level is None and stage._gain == 1.0
