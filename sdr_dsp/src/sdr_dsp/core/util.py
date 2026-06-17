"""Small numeric primitives used throughout: dB conversion and normalization.

Centralizes two things that were previously hand-rolled inconsistently across
the codebase: dB/linear conversion (with one canonical epsilon) and amplitude
normalization. Normalization is EXPLICIT and user-controlled -- the library
never silently rescales data, because a hidden gain change can quietly corrupt
an analysis.
"""

from __future__ import annotations

import numpy as np

# one canonical floor for log conversions, so dB readings are consistent
# everywhere instead of each call site picking its own 1e-9/1e-12/1e-20.
DB_EPSILON = 1e-20


def to_db(x, *, power=True, epsilon=DB_EPSILON):
    """Convert linear values to dB.

    power=True  : x is a power quantity      -> 10*log10(x)
    power=False : x is an amplitude/voltage  -> 20*log10(x)
    epsilon floors the input so zeros don't produce -inf.
    """
    x = np.asarray(x, dtype=np.float64)
    factor = 10.0 if power else 20.0
    return factor * np.log10(np.abs(x) + epsilon)


def from_db(db, *, power=True):
    """Inverse of to_db: dB back to a linear value."""
    db = np.asarray(db, dtype=np.float64)
    factor = 10.0 if power else 20.0
    return 10.0 ** (db / factor)


def normalize(iq, mode="peak", target=1.0):
    """Rescale a signal's amplitude. EXPLICIT -- you choose if and how.

    The library never normalizes silently; call this when you want it.

    mode:
      "peak" : scale so max|x| == target (headroom-friendly; good before WAV
               output or display).
      "rms"  : scale so the RMS amplitude == target (good before a demod or
               detector that assumes a consistent level across captures).
      "none" : return unchanged (so a pipeline can be parameterized).
    target: the desired peak or RMS level.

    Returns a new array (does not modify the input). A zero/empty signal is
    returned unchanged (nothing sensible to scale to).
    """
    iq = np.asarray(iq)
    if len(iq) == 0 or mode == "none":
        return iq
    mag = np.abs(iq)
    if mode == "peak":
        ref = float(mag.max())
    elif mode == "rms":
        ref = float(np.sqrt(np.mean(mag ** 2)))
    else:
        raise ValueError(f"unknown normalize mode: {mode!r}")
    if ref == 0.0:
        return iq
    return (iq * (target / ref)).astype(iq.dtype if np.iscomplexobj(iq)
                                        else np.float64)
