"""Frequency-shift-keying demodulation, built on instantaneous frequency."""

from __future__ import annotations

import numpy as np

from .phase import instantaneous_frequency


def fsk_demod(iq, sample_rate, threshold_hz=0.0, smooth_samples=0):
    """Demodulate 2-level frequency-shift keying. OUR code.

    FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
    instantaneous frequency, then a threshold: above threshold_hz -> 1, below
    -> 0. With the default threshold 0, it splits on the sign of the frequency
    deviation (correct when the two tones straddle the center frequency, which
    is the common case after tuning to baseband).

    threshold_hz: the mark/space decision frequency. Two real radios never
        share an oscillator, so a carrier frequency offset (crystal ppm --
        easily +/-10-20 kHz at 433 MHz between two SDRs) shifts BOTH tones and
        biases the fixed 0 Hz split. Pass "auto" to threshold at the
        amplitude^2-weighted mean of the instantaneous frequency instead: with
        roughly balanced mark/space time (any frame with an alternating
        preamble qualifies) that mean IS the offset, so the split self-centers.
        The weighting means silence around a burst contributes ~nothing.
        NOTE: do not use estimate_cfo for this -- it finds the strongest
        spectral tone, which for FSK is +/-deviation, not the offset.
    smooth_samples: if > 1, moving-average the instantaneous frequency over
        this many samples before slicing (a cheap matched-filter stand-in;
        ~samples_per_symbol/2 is a good value). The raw per-sample frequency
        is noisy, and this is the difference between decoding and not at
        moderate SNR. Off by default -- the per-sample output stays exact.

    Returns a uint8 per-sample bit stream (length len(iq)-1); feed to the
    timing helpers (sample_symbols for hardware captures with unknown delay,
    or estimate_symbol_rate / slice_to_symbols) to get symbols. Covers
    GFSK/MSK well enough for typical ISM-band sensors and pagers.
    """
    inst = instantaneous_frequency(iq, sample_rate=sample_rate)
    k = int(smooth_samples)
    if k > 1:
        inst = np.convolve(inst, np.ones(k) / k, mode="same")
    if isinstance(threshold_hz, str):
        if threshold_hz != "auto":
            raise ValueError(
                f"threshold_hz must be a number or 'auto', got {threshold_hz!r}")
        w = (np.abs(np.asarray(iq)) ** 2)[: len(inst)]
        total = float(w.sum())
        thr = float(np.average(inst, weights=w)) if total > 0 else 0.0
    else:
        thr = float(threshold_hz)
    return (inst > thr).astype(np.uint8)




def fsk_demod_nlevel(iq, sample_rate, n_levels=4, thresholds=None):
    """Demodulate N-level FSK (4-FSK, etc.) and CPFSK. OUR code.

    Generalizes 2-FSK: instead of a single 0-threshold on instantaneous
    frequency, it slices the frequency into n_levels bands. Used by 4-FSK
    (DMR, P25, some pagers). CPFSK recovers the same way -- the continuous
    phase is a transmit-side property; the receiver still reads instantaneous
    frequency.

    thresholds: explicit frequency band centers (Hz). If None, the levels are
    spread uniformly across the observed frequency range -- fine for a clean
    capture; pass measured centers for real signals. Returns per-sample symbol
    indices 0..n_levels-1.
    """
    inst = instantaneous_frequency(iq, sample_rate=sample_rate)
    if len(inst) == 0:
        return np.zeros(0, dtype=np.uint8)
    if thresholds is None:
        lo, hi = float(np.percentile(inst, 2)), float(np.percentile(inst, 98))
        centers = np.linspace(lo, hi, n_levels)
    else:
        centers = np.asarray(thresholds, dtype=np.float64)
    idx = np.argmin(np.abs(inst[:, None] - centers[None, :]), axis=1)
    return idx.astype(np.uint8)
