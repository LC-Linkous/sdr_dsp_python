"""Frequency-shift-keying demodulation, built on instantaneous frequency."""

from __future__ import annotations

import numpy as np

from .phase import instantaneous_frequency


def fsk_demod(iq, sample_rate, threshold_hz=0.0):
    """Demodulate 2-level frequency-shift keying. OUR code.

    FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
    instantaneous frequency, then a threshold: above threshold_hz -> 1, below
    -> 0. With the default threshold 0, it splits on the sign of the frequency
    deviation (correct when the two tones straddle the center frequency, which
    is the common case after tuning to baseband).

    Returns a uint8 per-sample bit stream; feed to the timing-recovery helpers
    (estimate_symbol_rate / slice_to_symbols) to get symbols. Covers GFSK/MSK
    well enough for typical ISM-band sensors and pagers.
    """
    inst = instantaneous_frequency(iq, sample_rate=sample_rate)
    return (inst > float(threshold_hz)).astype(np.uint8)




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
