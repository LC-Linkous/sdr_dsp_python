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
from dataclasses import dataclass

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


def _timing_lock_trace(steps, limit, window=20):
    """Per-symbol lock for a TIMING loop: the correction it applies has settled.

    `steps` is the timing correction applied at each symbol, in samples (the
    loop's proportional + integral output). Over a trailing window its rms
    must be below `limit`. While the loop is pulling in a timing or rate
    offset the steps are large (up to the sps/4 clamp); once it tracks, they
    shrink to the TED's residual jitter.

    Why not the TED output: a TED is noisy even when the loop is perfectly
    locked (data-pattern "self-noise"), and its noise is what the loop filter
    exists to average out; the STEP is what survives that averaging.
    Why not the sampling phase (tau mod sps): tracking a rate error makes it
    drift against the nominal sps, and against the tracked period the modulus
    is numerically unstable for large tau.

    What this does NOT tell you: whether there is a signal. A narrow loop fed
    pure noise applies small random steps too, and at low SNR the steps on a
    real signal are no smaller. Lock here means "the loop has settled"; use
    capture_health / power detection to establish that something is there.
    """
    n = len(steps)
    if n == 0:
        return np.zeros(0, dtype=bool), False
    s2 = np.asarray(steps, dtype=np.float64) ** 2
    lock = np.zeros(n, dtype=bool)
    need = max(5, window // 4)
    limit2 = float(limit) ** 2
    for i in range(n):
        lo = max(0, i - window + 1)
        if i + 1 - lo >= need:
            lock[i] = float(np.mean(s2[lo:i + 1])) < limit2
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

    Measured behaviour at the defaults (loop_bw 0.01, damping 0.707), on clean
    unit-rate symbols -- these numbers are pinned by tests/test_sync.py:
      pull-in range (CFO the loop acquires unaided): order 2 up to ~0.02
        cycles/sample, order 4 up to ~0.01 cycles/sample. Beyond that, remove
        the bulk of the offset first (estimate_cfo + frequency_shift) and let
        the loop clean up the residual.
      lock time for a 0.7 rad phase step: ~50 samples (order 2), ~200 (order 4).
      amplitude: the Costas detectors are normalized by the input rms, so the
        loop behaves identically at 0.01x and 100x full scale.
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

    # The Costas detectors below are LINEAR in the signal amplitude
    # (sign(I)*Q is |s|*sin(phase error)), so without normalization the loop
    # gains scale with the input level: a 100x stronger input ran 100x hotter
    # and lost lock, a 100x weaker one never acquired. Divide by the rms
    # amplitude so loop_bw means the same thing at any level. The
    # decision-directed detector is an angle and needs no normalization.
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    inv_rms = 1.0 / rms if rms > 0 else 1.0

    phase = 0.0
    freq = 0.0
    for i in range(n):
        s = iq[i] * np.exp(-1j * phase)        # de-rotate by current estimate
        out[i] = s
        # phase error detector
        if method == "costas":
            if order == 2:                      # BPSK: sign(I)*Q
                e = np.sign(np.real(s)) * np.imag(s) * inv_rms
            else:                               # QPSK order-4
                e = (np.sign(np.real(s)) * np.imag(s)
                     - np.sign(np.imag(s)) * np.real(s)) * inv_rms
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

# Nominal S-curve slope of each timing-error detector: mean TED output per
# sample of timing offset, at unit signal power, measured on RRC (beta 0.35,
# span 8) BPSK after a matched filter. Dividing the TED output by this puts the
# loop error in SAMPLES, so one loop_bw gives comparable dynamics across
# methods. These are nominal: a different pulse shape moves them by tens of
# percent, which the loop tolerates (it shifts the effective bandwidth, not the
# sign). tests/test_sync.py re-measures them and fails if they drift by >2x.
TED_GAIN = {"gardner": 0.13, "early_late": 0.13, "mueller_muller": 0.30}


def symbol_sync(iq, samples_per_symbol, method="gardner", loop_bw=0.01,
                damping=0.707, diagnostics=False, csv_path=None,
                lock_threshold=None):
    """Recover symbol timing: pick the best sampling instant per symbol. OUR code.

    Returns the symbol-spaced samples (one complex value per recovered symbol).
    With diagnostics=True, returns (symbols, LoopDiagnostics) where the error is
    the timing-error-detector output in samples and the estimate is the
    fractional sampling offset (tau modulo the nominal sps; it drifts when the
    loop is tracking a symbol-rate error, which is the loop doing its job).
    `lock` / `locked` mean the loop has SETTLED: the rms timing correction it
    applies per symbol, over a 20-symbol window, is below lock_threshold
    samples (default sps/16 -- 0.5 sample at 8 sps, which the loop meets at
    5 dB SNR while still decoding BPSK without error). Lock does not mean a
    signal is present: a narrow loop on pure noise settles too. Establish
    presence separately (capture_health, power detection).

    method:
      "gardner"        : Gardner TED -- carrier-independent, needs ~2 sps.
      "early_late"     : early-late gate -- simple, intuitive.
      "mueller_muller" : Mueller & Muller -- 1 sps, decision-aided.
    samples_per_symbol: nominal sps (from estimate_symbol_rate or known rate).
    loop_bw, damping: normalized loop bandwidth (fraction of the symbol rate)
      and damping ratio of the second-order loop filter -- the same
      parametrization as carrier_recovery, so 0.01 means the same thing in
      both. The loop filter has a proportional path (phase) and an integral
      path (symbol-rate error), so a steady sps mismatch is tracked out, not
      carried as a standing error.

    How the TED output is conditioned before it drives the loop:
      * normalized by the signal's mean power -- every TED here scales with
        |x|^2, so without this a strong input would overshoot and skip symbols;
      * scaled by the TED's nominal S-curve slope (TED_GAIN below, measured on
        RRC beta=0.35 pulses after a matched filter), so the error is in units
        of SAMPLES of timing offset and loop_bw is comparable across methods;
      * each correction clamped to +/- sps/4.

    Measured pull-in -- the symbol-rate error acquired within a 400-symbol
    burst in 10 of 10 random bit patterns (clean RRC BPSK, matched-filtered),
    pinned by tests/test_sync.py:
                        loop_bw 0.01   0.02    0.05
        gardner           +/-0.5%    1.0%    2.0%
        mueller_muller    +/-0.5%    1.5%    3.0%
        early_late        +/-0.25%   0.5%    unstable -- keep it <= 0.02
    Larger errors are acquired on SOME patterns (Gardner takes 1% at 0.01
    about one burst in five); for a dependable lock either widen the loop or
    hand it a better sps from estimate_symbol_rate. In 5 dB noise all three
    lock on most bursts and cycle-slip on roughly one in ten.
    early_late is the classic early-late gate; on random data its +/-T/4 taps
    see uncancelled ISI from neighbouring symbols (high "self-noise"), so it
    tracks the least and should be read as the teaching baseline. Gardner is
    carrier-phase-independent (use it before carrier recovery); Mueller &
    Muller is decision-directed (use it after, or on a clean constellation).
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

    if method not in TED_GAIN:
        raise ValueError(f"unknown timing method: {method!r}")
    # power normalization + S-curve gain (see docstring); an all-zero input has
    # no timing to recover, so leave the scale at 1 and let the loop idle
    power = float(np.mean(np.abs(iq) ** 2))
    norm = (1.0 / power if power > 0 else 1.0) / TED_GAIN[method]
    # second-order loop filter gains from loop_bw + damping (standard form,
    # identical to carrier_recovery)
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta + theta * theta
    alpha = (4.0 * damping * theta) / denom      # proportional (timing phase)
    beta = (4.0 * theta * theta) / denom         # integral (symbol-rate error)
    max_step = sps / 4.0

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
    # symbol, nudged by the timing error. `rate` is the integrator: the loop's
    # running estimate of the symbol-period error, in samples per symbol.
    tau = sps
    rate = 0.0
    steps = []
    while tau < n - 1:
        sample = interp(tau)
        # Sign convention for every TED: sampling LATE (tau past the symbol
        # peak) must give a NEGATIVE error, because the loop applies
        # tau += sps + correction. tests/test_sync.py measures each S-curve.
        if method == "gardner":
            # Gardner TED: midpoint sample between this symbol and the previous
            mid = interp(tau - half)
            e = float(np.real(np.conj(mid) * (prev_sym - sample)))
        elif method == "early_late":
            # early-late gate, projected onto the current symbol so the sign
            # survives random data: late sampling puts the EARLY tap nearer
            # the peak, so late - early is negative. (An earlier version used
            # |late| - |early| times sign(I), whose S-curve on random data was
            # ~20x weaker AND inverted -- the loop could hold nominal timing
            # but never track a rate error.)
            d = sps / 4.0
            early = interp(tau - d)
            late = interp(tau + d)
            e = float(np.real(np.conj(sample) * (late - early)))
        else:  # mueller_muller
            # decision-directed, both rails (the real-only form worked for
            # BPSK but threw away half of QPSK's timing information)
            dec = np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))
            e = float(np.real(np.conj(prev_decision) * sample
                              - np.conj(dec) * prev_sym))
            prev_decision = dec

        e = e * norm                              # -> samples of timing error
        out.append(sample)
        errs.append(e)
        prev_sym = sample
        # advance one symbol, corrected by the loop filter; the integrator is
        # bounded so a burst of bad TED samples can't run the rate away
        rate = float(np.clip(rate + beta * e, -max_step, max_step))
        step = float(np.clip(rate + alpha * e, -max_step, max_step))
        tau += sps + step
        steps.append(step - rate)        # the correction beyond the tracked rate
        ests.append(tau % sps)

    err = np.array(errs, dtype=np.float64)
    est = np.array(ests, dtype=np.float64)
    limit = sps / 16.0 if lock_threshold is None else float(lock_threshold)
    lock, locked = _timing_lock_trace(steps, limit, window=20)
    diag = LoopDiagnostics(err, est, lock, locked)
    if csv_path:
        diag.to_csv(csv_path)
    syms = np.array(out, dtype=np.complex64)
    return (syms, diag) if diagnostics else syms
