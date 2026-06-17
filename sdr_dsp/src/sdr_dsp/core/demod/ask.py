"""Amplitude-shift-keying demodulation: OOK/ASK envelope and slicing."""

from __future__ import annotations

import numpy as np


def ook_envelope(iq):
    """On-off-keying / ASK front end: the magnitude envelope. OUR code.

    Returns |iq| (no DC block -- OOK threshold detection wants the absolute
    level). Feed to ``ook_slice`` to recover bits.
    """
    return np.abs(np.asarray(iq)).astype(np.float64)


def ook_slice(envelope, threshold=None):
    """Threshold an OOK envelope into a 0/1 stream. OUR code.

    threshold: level above which a sample is '1'. If None, uses the midpoint
    between the envelope's min and max (a simple, robust default for a clean
    capture). Returns a uint8 array of 0/1.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if threshold is None:
        threshold = (env.min() + env.max()) / 2.0
    return (env > threshold).astype(np.uint8)


