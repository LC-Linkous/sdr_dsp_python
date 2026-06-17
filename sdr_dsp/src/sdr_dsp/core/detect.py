"""Detection: finding known patterns in a signal. OUR code.

Correlation-based detection -- the matched filter and its relatives. The matched
filter is the optimal detector for a known pattern in white noise; its output
peaks where the pattern best aligns. Used for sync-word detection, ranging, and
pulse detection.
"""

from __future__ import annotations

import numpy as np


def matched_filter(signal, template):
    """Correlate a known template against a signal. OUR code.

    Returns the correlation magnitude; its peak marks where the template best
    aligns with the signal.

    NOTE: np.correlate already conjugates its second argument for complex
    input, so the template is passed directly -- conjugating it ourselves would
    double-conjugate and destroy the match. (Matched filtering for complex
    baseband is correlation with the conjugated template, which is exactly what
    np.correlate computes.)
    """
    signal = np.asarray(signal)
    template = np.asarray(template)
    if len(template) == 0 or len(template) > len(signal):
        raise ValueError("template must be non-empty and no longer than signal")
    corr = np.correlate(signal, template, mode="valid")
    return np.abs(corr)


def detect_peak(signal, template, threshold=None):
    """Run a matched filter and return the best-match index (and its value).

    threshold: if given, returns (index, value) only when the peak exceeds it,
    else (None, value). Without a threshold, always returns the argmax.
    """
    mf = matched_filter(signal, template)
    idx = int(np.argmax(mf))
    val = float(mf[idx])
    if threshold is not None and val < threshold:
        return None, val
    return idx, val


def correlate(a, b, mode="full"):
    """Cross-correlation of two signals, conjugation handled correctly. OUR code.

    np.correlate already conjugates its second argument for complex input -- a
    well-known footgun (conjugating it yourself double-conjugates and breaks the
    result). This wrapper exists so that subtlety lives in ONE place. Returns
    the complex cross-correlation; take np.abs for a magnitude.

    mode: "full", "same", or "valid" (as numpy).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.correlate(a, b, mode=mode)


def convolve(a, b, mode="full"):
    """Convolution of two signals. OUR code (thin, for a uniform API).

    Unlike correlate, convolution does NOT conjugate -- it's the filtering
    operation. Provided alongside correlate so the distinction is explicit and
    both are first-class.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.convolve(a, b, mode=mode)
