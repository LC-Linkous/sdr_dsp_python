"""Isolation tests for the recovery primitives: carrier and timing loops must
converge on known offsets, all methods, with working diagnostics/CSV.

These are deliberately tested ALONE (before any demod depends on them) so the
recovery layer is proven, not assumed.
"""

import numpy as np
import pytest

from sdr_dsp.core import (carrier_recovery, symbol_sync, LoopDiagnostics,
                          bpsk_modulate, qpsk_modulate, rrc_taps, fir_apply)
from sdr_dsp.core.sync import TED_GAIN


def _bpsk(nsym, sps, seed=0):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nsym)
    return np.repeat(2 * bits - 1, sps).astype(np.complex64), bits


# ---- carrier recovery ----

def test_carrier_costas_removes_offset():
    sym, _ = _bpsk(200, 20, seed=0)
    t = np.arange(len(sym))
    rx = (sym * np.exp(2j * np.pi * 0.002 * t + 1j * 0.7)).astype(np.complex64)
    rng = np.random.default_rng(1)
    rx += 0.05 * (rng.standard_normal(len(rx))
                  + 1j * rng.standard_normal(len(rx)))
    corrected, diag = carrier_recovery(rx, method="costas", order=2,
                                       diagnostics=True)
    assert diag.locked
    # constellation collapses toward the real axis
    ratio = (np.mean(np.abs(corrected[-1000:].imag))
             / np.mean(np.abs(corrected[-1000:].real)))
    assert ratio < 0.3


def test_carrier_decision_directed_locks():
    sym, _ = _bpsk(200, 20, seed=2)
    t = np.arange(len(sym))
    rx = (sym * np.exp(2j * np.pi * 0.001 * t + 1j * 0.4)).astype(np.complex64)
    _, diag = carrier_recovery(rx, method="decision_directed", order=2,
                               diagnostics=True)
    assert diag.locked


def test_carrier_bad_method():
    import pytest
    with pytest.raises(ValueError):
        carrier_recovery(np.ones(100, dtype=np.complex64), method="nope")


def test_carrier_diagnostics_optional():
    rx = np.ones(500, dtype=np.complex64)
    out = carrier_recovery(rx)               # no diagnostics
    assert isinstance(out, np.ndarray)
    out2, diag = carrier_recovery(rx, diagnostics=True)
    assert isinstance(diag, LoopDiagnostics)


def test_carrier_csv(tmp_path):
    sym, _ = _bpsk(100, 20, seed=3)
    p = tmp_path / "diag.csv"
    carrier_recovery(sym, csv_path=str(p))
    lines = p.read_text().splitlines()
    assert lines[0] == "sample,error,estimate,lock"
    assert len(lines) == len(sym) + 1


# ---- symbol timing recovery ----

def test_symbol_sync_all_methods_converge():
    for method, sps in [("gardner", 2), ("early_late", 8),
                        ("mueller_muller", 8)]:
        sym, _ = _bpsk(400, sps, seed=4)
        sig = np.concatenate([np.zeros(max(1, sps // 3), dtype=np.complex64),
                              sym])
        rng = np.random.default_rng(5)
        sig += 0.03 * (rng.standard_normal(len(sig))
                       + 1j * rng.standard_normal(len(sig)))
        syms, diag = symbol_sync(sig, sps, method=method, diagnostics=True)
        assert len(syms) > 50
        dist = np.mean(np.abs(np.abs(syms.real) - 1.0))
        assert dist < 0.15, f"{method}: dist {dist}"
        assert diag.locked, f"{method} did not lock"


def _shaped_bpsk(nsym=400, sps=8, seed=0):
    from sdr_dsp.core import bpsk_modulate
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nsym)
    return bpsk_modulate(bits, sps, pulse_shaping=True), bits


def _bits_from(syms):
    return (np.real(syms) < 0).astype(int)


def test_symbol_sync_is_amplitude_invariant():
    # Every TED here scales with |x|^2. Before the loop normalized by signal
    # power, a 100x stronger input made the correction 10,000x larger: the
    # sampling instant skipped around and the loop emitted ~2x too many
    # symbols. The recovered symbol count and the decisions must not depend
    # on the input's scale.
    sig, bits = _shaped_bpsk()
    ref = symbol_sync(sig, 8)
    for amp in (0.01, 1.0, 100.0, 1e4):
        syms = symbol_sync((sig * amp).astype(np.complex64), 8)
        assert len(syms) == len(ref), f"amp {amp}: {len(syms)} vs {len(ref)} symbols"
        np.testing.assert_array_equal(_bits_from(syms), _bits_from(ref))
    # sanity: the count is the transmitted symbol count (give or take the edges)
    assert abs(len(ref) - len(bits)) <= 2


def test_symbol_sync_damping_is_used():
    # damping sets the integral gain of the second-order loop; it must change
    # the loop's behaviour, not be an accepted-and-ignored argument.
    sig, _ = _shaped_bpsk()
    _, d_light = symbol_sync(sig, 8.05, damping=0.3, diagnostics=True)
    _, d_heavy = symbol_sync(sig, 8.05, damping=2.0, diagnostics=True)
    assert not np.array_equal(d_light.estimate, d_heavy.estimate)


# ---- loop characterization -------------------------------------------------
#
# The tests above prove each loop converges on ONE benign case. The AGC
# collapse (patch 2) taught us that a loop can pass every steady-state test
# and still misbehave dynamically, so this block measures the loops the way a
# radio would meet them: at any amplitude, with a symbol-rate or carrier
# offset to pull in, after a step, in noise. The numbers asserted here are the
# ones quoted in the sync.py docstrings; if an implementation change moves
# them, change both.

METHODS = ("gardner", "early_late", "mueller_muller")


def _mf_bpsk(nsym=400, sps=8, seed=0, snr_db=None):
    """RRC BPSK through its matched filter -- what a timing loop really sees."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nsym)
    x = fir_apply(bpsk_modulate(bits, sps, pulse_shaping=True),
                  rrc_taps(sps, 8, 0.35))
    if snr_db is not None:
        a = 10 ** (-snr_db / 20) / np.sqrt(2)
        x = x + a * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64), bits


def _ber(syms, bits, max_lag=16):
    """BER of hard BPSK decisions against bits, over the best alignment and
    polarity (the RRC chain has a group delay; BPSK has a sign ambiguity)."""
    rec = np.where(np.real(syms) < 0, 1, 0)
    best = 0.5
    for lag in range(-max_lag, max_lag + 1):
        a = rec[max(0, lag):]
        b = bits[max(0, -lag):]
        m = min(len(a), len(b))
        if m < len(bits) // 2:
            continue
        e = float(np.mean(a[:m] != b[:m]))
        best = min(best, e, 1.0 - e)
    return best


def _interp(x, pos):
    i = int(np.floor(pos))
    f = pos - i
    if i + 1 >= len(x):
        return x[-1]
    return x[i] * (1 - f) + x[i + 1] * f


def _ted_open_loop(x, method, sps, offset):
    """Mean TED output when sampling every sps at a fixed offset from the true
    instant (open loop), normalized to unit power. This is one point of the
    detector's S-curve."""
    P = float(np.mean(np.abs(x) ** 2))
    es, prev, prevdec = [], 0j, 1 + 0j
    tau = 3 * sps + offset
    while tau < len(x) - sps:
        s = _interp(x, tau)
        if method == "gardner":
            e = np.real(np.conj(_interp(x, tau - sps / 2)) * (prev - s))
        elif method == "early_late":
            e = np.real(np.conj(s) * (_interp(x, tau + sps / 4) - _interp(x, tau - sps / 4)))
        else:
            dec = np.sign(s.real) + 1j * np.sign(s.imag)
            e = np.real(np.conj(prevdec) * s - np.conj(dec) * prev)
            prevdec = dec
        es.append(float(e))
        prev = s
        tau += sps
    return float(np.mean(es)) / P


@pytest.mark.parametrize("method", METHODS)
def test_ted_s_curve_sign_and_gain(method):
    # Sampling LATE must produce a NEGATIVE error (the loop adds the correction
    # to tau), and the slope near zero must be the TED_GAIN the loop divides
    # by, within 2x. The old early-late detector failed BOTH: its S-curve on
    # random data was ~20x weaker than assumed and inverted, so the loop held
    # nominal timing but could never track -- and one convergence test at
    # nominal timing passed it for months.
    x, _ = _mf_bpsk(nsym=2000)
    sps = 8
    # true sampling instant: the offset with the most symbol energy
    true = max(range(sps), key=lambda o: np.mean(np.abs(x[o::sps]) ** 2))
    late = _ted_open_loop(x, method, sps, true + 1.0)
    early = _ted_open_loop(x, method, sps, true - 1.0)
    on_time = _ted_open_loop(x, method, sps, true)
    assert late < 0 < early, f"{method}: S-curve sign wrong (late={late:.3f}, early={early:.3f})"
    assert abs(on_time) < 0.3 * abs(late), f"{method}: S-curve not zero on time ({on_time:.3f})"
    slope = (early - late) / 2.0                 # per sample of offset
    assert 0.5 < slope / TED_GAIN[method] < 2.0, (
        f"{method}: measured S-curve slope {slope:.3f}/sample vs TED_GAIN "
        f"{TED_GAIN[method]}; update the constant")


# (method, loop_bw) -> rate error in percent acquired on 10/10 bit patterns;
# the table in the symbol_sync docstring, measured over seeds 0-9
PULL_IN = {
    ("gardner", 0.01): 0.5, ("gardner", 0.02): 1.0, ("gardner", 0.05): 2.0,
    ("mueller_muller", 0.01): 0.5, ("mueller_muller", 0.02): 1.5,
    ("mueller_muller", 0.05): 3.0,
    ("early_late", 0.01): 0.25, ("early_late", 0.02): 0.5,
}


@pytest.mark.parametrize("method,loop_bw", sorted(PULL_IN))
def test_symbol_sync_pulls_in_a_rate_error(method, loop_bw):
    # A nominal sps that is wrong by the table's percentage must be TRACKED
    # OUT within a 400-symbol burst -- BER ~0 on the recovered symbols -- on
    # every one of five bit patterns, both signs. The count-only check this
    # replaces could not tell a tracking loop from a free-running one
    # (3200 samples / 7.92 = 404 either way).
    pct = PULL_IN[(method, loop_bw)]
    for seed in range(5):
        x, bits = _mf_bpsk(seed=seed)
        for sign in (+1, -1):
            told = 8.0 * (1 + sign * pct / 100)
            ber = _ber(symbol_sync(x, told, method=method, loop_bw=loop_bw), bits)
            assert ber < 0.02, f"{method} bw {loop_bw} seed {seed}: sps told {told}: BER {ber:.3f}"


def test_symbol_sync_loop_bw_is_not_free():
    # the table has a ceiling: the widest loop does NOT reliably take twice
    # its rated error (so the docstring is not understating), and early-late
    # at 0.05 is unstable even at nominal rate -- both worth knowing
    failures = 0
    for seed in range(5):
        x, bits = _mf_bpsk(seed=seed)
        failures += _ber(symbol_sync(x, 8.0 * 1.04, method="gardner", loop_bw=0.05), bits) > 0.02
    assert failures >= 2, "Gardner at 0.05 takes 4%; raise the table"
    x, bits = _mf_bpsk(seed=0)
    assert _ber(symbol_sync(x, 8.0 * 1.0025, method="early_late", loop_bw=0.05), bits) > 0.1


@pytest.mark.parametrize("method", METHODS)
def test_symbol_sync_in_noise(method):
    # 5 dB: the loop must lock on most bursts; hard-decision BPSK at this SNR
    # contributes ~6e-3 BER itself. A cycle slip on one burst in ten is the
    # measured behaviour, so allow one of five.
    bad = 0
    for seed in range(5):
        x, bits = _mf_bpsk(snr_db=5, seed=seed)
        syms, diag = symbol_sync(x, 8, method=method, diagnostics=True)
        bad += (_ber(syms, bits) > 0.05) or (not diag.locked)
    assert bad <= 1, f"{method}: failed to lock on {bad}/5 noisy bursts"


def test_symbol_sync_recovers_from_a_timing_step():
    # three samples vanish mid-burst (a dropped USB packet does this); the loop
    # must re-acquire and the symbols after the step must be right
    x, bits = _mf_bpsk()
    cut = 1600
    x2 = np.concatenate([x[:cut], x[cut + 3:]])
    syms = symbol_sync(x2, 8, method="gardner")
    tail = syms[len(syms) // 2 + 20:]
    assert _ber(tail, bits[len(bits) // 2:], max_lag=24) < 0.02


def test_symbol_sync_qpsk_with_rate_error():
    # both rails carry timing information for QPSK; the real-only M&M form
    # discarded half of it. Constellation cleanliness after recovery at a 1.25%
    # sps error, each method.
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 800)
    x = fir_apply(qpsk_modulate(bits, 8, pulse_shaping=True), rrc_taps(8, 8, 0.35))
    for method, bound in (("gardner", 0.08), ("mueller_muller", 0.05),
                          ("early_late", 0.2)):
        syms = symbol_sync(x.astype(np.complex64), 8.1, method=method, loop_bw=0.02)
        s = syms[20:] / np.abs(syms[20:])
        quadrant = np.round((np.angle(s) - np.pi / 4) / (np.pi / 2))
        ideal = np.exp(1j * (np.pi / 4 + np.pi / 2 * quadrant))
        err = float(np.mean(np.abs(s - ideal)))
        assert err < bound, f"{method}: QPSK constellation error {err:.3f}"


def test_symbol_sync_lock_means_settled_not_signal_present():
    # `locked` is "the loop's per-symbol correction has settled". It is True
    # for a clean burst, a 5 dB burst, a loop tracking a 1% rate error, and
    # Gardner on rectangular 2-sps pulses (noisy TED, converged loop) ...
    cases = [
        (_mf_bpsk(seed=0)[0], 8, {}),
        (_mf_bpsk(snr_db=5, seed=0)[0], 8, {}),
        (_mf_bpsk(seed=1)[0], 8.08, {"loop_bw": 0.02}),
        (_bpsk(400, 2, seed=4)[0], 2, {}),
    ]
    for x, told, kw in cases:
        _, diag = symbol_sync(x, told, method="gardner", diagnostics=True, **kw)
        assert diag.locked
    # ... and, as the docstring warns, for pure noise: a narrow loop on nothing
    # settles too. Signal presence is capture_health's job. If this assertion
    # ever fails because noise reads unlocked, that is an improvement -- update
    # the docstring, not the loop.
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(3200) + 1j * rng.standard_normal(3200)).astype(np.complex64)
    _, diag = symbol_sync(noise, 8, method="gardner", diagnostics=True)
    assert diag.locked
    # the threshold is the knob: a strict one calls the 5 dB loop unsettled
    _, strict = symbol_sync(_mf_bpsk(snr_db=5, seed=0)[0], 8, method="gardner",
                            diagnostics=True, lock_threshold=0.02)
    assert not strict.locked


# -- carrier loop characterization --

def _qpsk_syms(n, seed=0):
    rng = np.random.default_rng(seed)
    return qpsk_modulate(rng.integers(0, 2, n), 1), None


def _constellation_error(out, order):
    s = out[len(out) // 2:]
    s = s / np.abs(s)
    if order == 2:
        ideal = np.sign(s.real) + 0j
    else:
        ideal = (np.sign(s.real) + 1j * np.sign(s.imag)) / np.sqrt(2)
    return float(np.mean(np.abs(s - ideal)))


@pytest.mark.parametrize("order", (2, 4))
@pytest.mark.parametrize("method", ("costas", "decision_directed"))
def test_carrier_recovery_is_amplitude_invariant(method, order):
    # sign(I)*Q is linear in amplitude; unnormalized, a 100x input ran the loop
    # 100x hotter and lost lock, a 0.01x input never acquired (residual 0.75 /
    # 0.36 vs 0.000 at unit amplitude)
    tx = _bpsk(4000, 1, seed=0)[0] if order == 2 else _qpsk_syms(4000)[0]
    t = np.arange(len(tx))
    imp = tx * np.exp(2j * np.pi * 0.002 * t + 0.7j)
    for amp in (0.01, 1.0, 100.0):
        out = carrier_recovery((imp * amp).astype(np.complex64), method=method, order=order)
        err = _constellation_error(out, order)
        assert err < 0.05, f"{method} order {order} amp {amp}: residual {err:.3f}"


@pytest.mark.parametrize("order,ok,fails", [(2, 0.02, 0.05), (4, 0.01, 0.02)])
def test_costas_pull_in_range(order, ok, fails):
    # at loop_bw 0.01 the loop acquires a CFO up to `ok` cycles/sample unaided
    # and does NOT acquire `fails` -- the docstring's stated range, both ends
    tx = _bpsk(4000, 1, seed=0)[0] if order == 2 else _qpsk_syms(4000)[0]
    t = np.arange(len(tx))
    for cfo in (0.0005, 0.005, ok):
        rx = (tx * np.exp(2j * np.pi * cfo * t)).astype(np.complex64)
        out = carrier_recovery(rx, order=order)
        assert _constellation_error(out, order) < 0.05, f"order {order}: lost CFO {cfo}"
    out = carrier_recovery((tx * np.exp(2j * np.pi * fails * t)).astype(np.complex64), order=order)
    assert _constellation_error(out, order) > 0.2, (
        f"order {order}: acquired CFO {fails}; the docstring understates the range")


@pytest.mark.parametrize("order,within", [(2, 120), (4, 400)])
def test_costas_lock_time_for_a_phase_step(order, within):
    tx = _bpsk(4000, 1, seed=0)[0] if order == 2 else _qpsk_syms(4000)[0]
    out, diag = carrier_recovery((tx * np.exp(0.7j)).astype(np.complex64),
                                 order=order, diagnostics=True)
    assert diag.lock.any() and int(np.argmax(diag.lock)) < within
    assert diag.locked
    assert np.mean(np.abs(diag.error[-500:])) < 0.01


def test_symbol_sync_bad_method():
    import pytest
    sym, _ = _bpsk(100, 4, seed=6)
    with pytest.raises(ValueError):
        symbol_sync(sym, 4, method="nope")


def test_symbol_sync_too_short():
    out = symbol_sync(np.ones(3, dtype=np.complex64), 8)
    assert len(out) == 0


def test_loop_diagnostics_to_csv(tmp_path):
    err = np.array([0.1, 0.05, 0.02])
    est = np.array([1.0, 1.0, 1.0])
    lock = np.array([False, True, True])
    diag = LoopDiagnostics(err, est, lock, True)
    p = tmp_path / "d.csv"
    diag.to_csv(str(p))
    lines = p.read_text().splitlines()
    assert len(lines) == 4   # header + 3
