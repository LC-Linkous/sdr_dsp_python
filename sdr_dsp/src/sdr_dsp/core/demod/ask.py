"""Amplitude-shift-keying demodulation: OOK/ASK envelope and slicing."""

from __future__ import annotations

from typing import Any

import numpy as np
from .._validate import positive_int


def ook_envelope(iq: Any) -> np.ndarray:
    """On-off-keying / ASK front end: the magnitude envelope. OUR code.

    Returns |iq| (no DC block -- OOK threshold detection wants the absolute
    level). Feed to ``ook_slice`` to recover bits.
    """
    return np.abs(np.asarray(iq)).astype(np.float64)


def ook_slice(envelope: Any, threshold: float | None = None) -> np.ndarray:
    """Threshold an OOK envelope into a 0/1 stream. OUR code.

    threshold: level above which a sample is '1'. If None, uses the midpoint
    between the envelope's min and max (a simple, robust default for a clean
    capture). Returns a uint8 array of 0/1.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if env.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if threshold is None:
        threshold = (env.min() + env.max()) / 2.0
    return (env > threshold).astype(np.uint8)




def nask_slice(envelope: Any, n_levels: int = 4, levels: Any | None = None) -> np.ndarray:
    """Slice an amplitude envelope into N levels (M-ASK). OUR code.

    Generalizes 2-level OOK to N amplitude levels (4-ASK, 8-ASK). Returns a
    per-sample symbol index in 0..n_levels-1.

    levels: explicit amplitude centers; when given, they define the level
    count and n_levels is ignored. If None, n_levels centers are
    spread uniformly from the envelope's min to its max -- a reasonable default
    for a clean capture, but YOU can pass measured levels for real signals where
    the spacing isn't uniform.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if len(env) == 0:
        return np.zeros(0, dtype=np.uint8)
    if levels is None:
        n_levels = positive_int("n_levels", n_levels, minimum=2)
        lo, hi = float(env.min()), float(env.max())
        # N centers evenly spaced; assign each sample to the nearest center
        centers = np.linspace(lo, hi, n_levels)
    else:
        # explicit levels DEFINE the level count; n_levels is not consulted
        centers = np.atleast_1d(np.asarray(levels, dtype=np.float64))
        if centers.ndim != 1 or len(centers) < 2:
            raise ValueError(f"levels must be a 1-D sequence of >= 2 amplitudes, got {levels!r}")
    # nearest-center decision
    idx = np.argmin(np.abs(env[:, None] - centers[None, :]), axis=1)
    return idx.astype(np.uint8)
