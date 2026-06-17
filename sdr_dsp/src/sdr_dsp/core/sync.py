"""Recovery loops: carrier (phase/frequency) and symbol timing.

These are the composable primitives that coherent demods build on. They are
kept SEPARATE from the demods (Option B) so each can be tested and inspected on
its own -- you can watch a constellation snap into place, or plot a loop's error
signal converging, independent of any demodulator.

Honesty by design: every loop can return DIAGNOSTICS (per-sample error, the
running estimate, and a per-sample lock trace) plus a summary `locked` flag, and
can write that trace to CSV. Recovery loops do not always converge -- on noisy
real captures they may need parameter tuning or may not lock at all -- so the
library exposes the evidence rather than asserting success.

Loop parameters (loop_bw, damping) are exposed because tuning them IS the lesson
in how these loops behave; they are not hidden.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoopDiagnostics:
    """Per-sample evidence of a recovery loop's behavior.

    error:    the loop's error signal each sample (phase error, or timing
              error). Its settling toward ~0 is convergence.
    estimate: the running quantity the loop tracks (accumulated phase, or the
              fractional sample offset).
    lock:     per-sample boolean trace -- True where the error variance over a
              sliding window is below the lock threshold.
    locked:   summary -- True if the loop was locked over the final portion.
    """
    error: np.ndarray
    estimate: np.ndarray
    lock: np.ndarray
    locked: bool = False

    def to_csv(self, path):
        """Write the per-sample diagnostics to a CSV (sample, error, estimate,
        lock). Useful for plotting convergence outside the library."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "error", "estimate", "lock"])
            for i in range(len(self.error)):
                w.writerow([i, float(self.error[i]), float(self.estimate[i]),
                            int(self.lock[i])])
        return str(path)


def _lock_trace(error, window=200, threshold=0.05):
    """Per-sample lock: error variance over a trailing window below threshold.

    Returns (lock_trace_bool, locked_summary). The summary checks the final
    quarter of the signal so a slow-to-converge loop still counts as locked if
    it settled by the end.
    """
    n = len(error)
    if n == 0:
        return np.zeros(0, dtype=bool), False
    lock = np.zeros(n, dtype=bool)
    # rolling variance via cumulative sums (cheap)
    e = np.asarray(error, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = e[lo:i + 1]
        if len(seg) >= max(10, window // 4):
            lock[i] = float(np.var(seg)) < threshold
    locked = bool(np.mean(lock[max(0, 3 * n // 4):]) > 0.5) if n >= 4 else False
    return lock, locked


# ---------------------------------------------------------------------------
# CARRIER RECOVERY
# ---------------------------------------------------------------------------
def carrier_recovery(iq, method="costas", order=2, loop_bw=0.01, damping=0.707,
                     diagnostics=False, csv_path=None, lock_threshold=0.05):
    """Track and remove residual carrier phase/frequency offset. OUR code.

    Returns a phase-corrected copy of iq (constellation de-rotated). With
    diagnostics=True, returns (corrected, LoopDiagnostics). With csv_path set,
    also writes the diagnostics CSV.

    method:
      "costas"            : a Costas loop -- data-independent phase tracking.
                            order=2 for BPSK, order=4 for QPSK (the phase error
                            detector matches the modulation's symmetry).
      "decision_directed" : uses the nearest-symbol decision to form the error;
                            simpler, better at high SNR, needs order too.
    loop_bw, damping: second-order loop filter parameters. Smaller loop_bw =
      slower but steadier lock. These are exposed on purpose.
    order: 2 (BPSK-like) or 4 (QPSK-like) phase symmetry.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    out = np.empty(n, dtype=np.complex64)
    err = np.zeros(n, dtype=np.float64)
    est = np.zeros(n, dtype=np.float64)
    if n == 0:
        diag = LoopDiagnostics(err, est, np.zeros(0, bool), False)
        return (out, diag) if diagnostics else out

    # second-order loop filter gains from loop_bw + damping (standard form)
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta + theta * theta
    alpha = (4.0 * damping * theta) / denom
    beta = (4.0 * theta * theta) / denom

    phase = 0.0
    freq = 0.0
    for i in range(n):
        s = iq[i] * np.exp(-1j * phase)        # de-rotate by current estimate
        out[i] = s
        # phase error detector
        if method == "costas":
            if order == 2:                      # BPSK: sign(I)*Q
                e = np.real(s) * np.imag(s)
                # normalize-ish for stability
                e = np.sign(np.real(s)) * np.imag(s)
            else:                               # QPSK order-4
                e = (np.sign(np.real(s)) * np.imag(s)
                     - np.sign(np.imag(s)) * np.real(s))
        elif method == "decision_directed":
            # nearest constellation point (unit BPSK/QPSK) then phase diff
            if order == 2:
                dec = np.sign(np.real(s)) + 0j
            else:
                dec = (np.sign(np.real(s)) + 1j * np.sign(np.imag(s)))
            e = np.angle(s * np.conj(dec))
        else:
            raise ValueError(f"unknown carrier method: {method!r}")

        err[i] = e
        freq += beta * e
        phase += freq + alpha * e
        est[i] = phase

    lock, locked = _lock_trace(err, threshold=lock_threshold)
    diag = LoopDiagnostics(err, est, lock, locked)
    if csv_path:
        diag.to_csv(csv_path)
    return (out, diag) if diagnostics else out


# ---------------------------------------------------------------------------
# SYMBOL TIMING RECOVERY
# ---------------------------------------------------------------------------
def symbol_sync(iq, samples_per_symbol, method="gardner", loop_bw=0.01,
                damping=0.707, diagnostics=False, csv_path=None,
                lock_threshold=0.05):
    """Recover symbol timing: pick the best sampling instant per symbol. OUR code.

    Returns the symbol-spaced samples (one complex value per recovered symbol).
    With diagnostics=True, returns (symbols, LoopDiagnostics) where the error is
    the timing-error-detector output and the estimate is the fractional offset.

    method:
      "gardner"        : Gardner TED -- carrier-independent, needs ~2 sps.
      "early_late"     : early-late gate -- simple, intuitive.
      "mueller_muller" : Mueller & Muller -- 1 sps, decision-aided.
    samples_per_symbol: nominal sps (from estimate_symbol_rate or known rate).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    sps = float(samples_per_symbol)
    n = len(iq)
    if n < 2 * sps:
        empty = np.zeros(0, dtype=np.complex64)
        if diagnostics:
            return empty, LoopDiagnostics(np.zeros(0), np.zeros(0),
                                          np.zeros(0, bool), False)
        return empty

    def interp(pos):
        # linear interpolation at fractional sample position pos
        idx = int(np.floor(pos))
        frac = pos - idx
        if idx < 0:
            return iq[0]
        if idx + 1 >= n:
            return iq[min(idx, n - 1)]
        return iq[idx] * (1 - frac) + iq[idx + 1] * frac

    out = []
    errs = []
    ests = []
    prev_sym = 0 + 0j
    prev_decision = 1 + 0j
    half = sps / 2.0

    # tau is the current sampling position (in samples); advances by ~sps each
    # symbol, nudged by the timing error.
    tau = sps
    while tau < n - 1:
        sample = interp(tau)
        if method == "gardner":
            # Gardner TED: midpoint sample between this symbol and the previous
            mid = interp(tau - half)
            e = (np.real(mid) * (np.real(prev_sym) - np.real(sample))
                 + np.imag(mid) * (np.imag(prev_sym) - np.imag(sample)))
        elif method == "early_late":
            d = sps / 4.0
            early = interp(tau - d)
            late = interp(tau + d)
            e = float(np.real((np.abs(late) - np.abs(early))
                              * np.sign(np.real(sample))))
        elif method == "mueller_muller":
            dec = np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))
            e = float(np.real(prev_decision) * np.real(sample)
                      - np.real(dec) * np.real(prev_sym))
            prev_decision = dec
        else:
            raise ValueError(f"unknown timing method: {method!r}")

        out.append(sample)
        errs.append(float(e))
        prev_sym = sample
        # advance one symbol, corrected by the timing error
        tau += sps + loop_bw * float(e)
        ests.append(tau % sps)

    err = np.array(errs, dtype=np.float64)
    est = np.array(ests, dtype=np.float64)
    lock, locked = _lock_trace(err, window=max(20, int(10)),
                               threshold=lock_threshold)
    diag = LoopDiagnostics(err, est, lock, locked)
    if csv_path:
        diag.to_csv(csv_path)
    syms = np.array(out, dtype=np.complex64)
    return (syms, diag) if diagnostics else syms
