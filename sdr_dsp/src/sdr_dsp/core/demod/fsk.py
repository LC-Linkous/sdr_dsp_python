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


