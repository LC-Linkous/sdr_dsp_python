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
