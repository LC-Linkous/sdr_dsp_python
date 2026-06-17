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




def nask_slice(envelope, n_levels=4, levels=None):
    """Slice an amplitude envelope into N levels (M-ASK). OUR code.

    Generalizes 2-level OOK to N amplitude levels (4-ASK, 8-ASK). Returns a
    per-sample symbol index in 0..n_levels-1.

    levels: explicit amplitude thresholds/centers. If None, the levels are
    spread uniformly from the envelope's min to its max -- a reasonable default
    for a clean capture, but YOU can pass measured levels for real signals where
    the spacing isn't uniform.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if len(env) == 0:
        return np.zeros(0, dtype=np.uint8)
    if levels is None:
        lo, hi = float(env.min()), float(env.max())
        # N centers evenly spaced; assign each sample to the nearest center
        centers = np.linspace(lo, hi, n_levels)
    else:
        centers = np.asarray(levels, dtype=np.float64)
    # nearest-center decision
    idx = np.argmin(np.abs(env[:, None] - centers[None, :]), axis=1)
    return idx.astype(np.uint8)
