"""Isolation tests for the recovery primitives: carrier and timing loops must
converge on known offsets, all methods, with working diagnostics/CSV.

These are deliberately tested ALONE (before any demod depends on them) so the
recovery layer is proven, not assumed.
"""

import numpy as np

from sdr_dsp.core import carrier_recovery, symbol_sync, LoopDiagnostics


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


def test_symbol_sync_tracks_a_symbol_rate_offset():
    # the integrator's job: a 1% error in the nominal sps must not accumulate
    # into dropped or doubled symbols over a 400-symbol burst
    sig, bits = _shaped_bpsk()
    for told in (7.92, 8.08):
        syms = symbol_sync(sig, told)
        assert abs(len(syms) - len(bits)) <= 8, f"sps={told}: {len(syms)} symbols"


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
