"""Resampling: sdr_dsp's own rational resampler + decimation/interpolation.

Changing sample rate is fundamental to SDR work -- decimate a wide capture down
to a channel's bandwidth, or upsample a baseband signal to an audio rate. The
*operation* here is the library's own; we use scipy only to DESIGN the
anti-alias filter taps (firwin), consistent with the implementation rule. The
own resampler is verified against ``scipy.signal.resample_poly`` in the tests
and benchmarked against it in an example.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal as _sig

from .filters import fir_apply


def _antialias_taps(up, down, half_len=10, window="hamming"):
    """Design the anti-alias / interpolation FIR for a rational resample.

    The filter runs at the UP-sampled rate and must cut at the lower of the two
    Nyquists: 1 / max(up, down) of that rate. (scipy designs the taps.)
    """
    max_rate = max(up, down)
    f_c = 1.0 / max_rate                       # normalized to upsampled Nyquist
    n = 2 * half_len * max_rate + 1            # odd length, scaled to rate
    taps = _sig.firwin(n, f_c, window=window)
    return taps * up                           # compensate insert-zeros gain


def resample_poly(iq, up, down, half_len=10, window="hamming"):
    """Rational resample by up/down. OUR implementation (polyphase concept).

    Implements the classic upsample -> lowpass -> downsample, with the zero
    insertion and decimation done explicitly and the lowpass applied by our own
    ``fir_apply``. Verified ~equal to ``scipy.signal.resample_poly`` in tests.

    up, down:  resampling ratio (reduced internally).
    half_len:  controls filter length / quality.
    """
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    g = gcd(up, down)
    up //= g
    down //= g
    if up == 1 and down == 1:
        return np.asarray(iq, dtype=np.complex64 if np.iscomplexobj(iq)
                          else np.float64)

    iq = np.asarray(iq)
    n = len(iq)
    cplx = np.iscomplexobj(iq)

    # 1. upsample: insert (up-1) zeros between samples
    up_n = n * up
    if cplx:
        ups = np.zeros(up_n, dtype=np.complex64)
    else:
        ups = np.zeros(up_n, dtype=np.float64)
    ups[::up] = iq

    # 2. anti-alias lowpass at the upsampled rate (our fir_apply).
    #    Pad the tail by the filter group delay first, so that after we trim
    #    the delay in step 3 the output still has the full expected length
    #    (otherwise the last `delay` output samples are lost to the transient).
    taps = _antialias_taps(up, down, half_len=half_len, window=window)
    delay = (len(taps) - 1) // 2
    if cplx:
        ups = np.concatenate([ups, np.zeros(delay, dtype=np.complex64)])
    else:
        ups = np.concatenate([ups, np.zeros(delay, dtype=np.float64)])
    filt = fir_apply(ups, taps)

    # 3. compensate the filter group delay so output aligns, then decimate
    filt = filt[delay:]
    out = filt[::down]
    # expected output length for a poly resample
    out_len = int(np.ceil(n * up / down))
    out = out[:out_len]
    return out.astype(np.complex64 if cplx else np.float64)


def decimate(iq, factor, half_len=10):
    """Lowpass then keep every ``factor``-th sample. OUR code (via resample)."""
    return resample_poly(iq, 1, factor, half_len=half_len)


def interpolate(iq, factor, half_len=10):
    """Upsample by ``factor`` with interpolation filtering. OUR code."""
    return resample_poly(iq, factor, 1, half_len=half_len)
