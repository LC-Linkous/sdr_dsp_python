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


def _resample_poly_direct(iq, up, down, taps):
    """The textbook form: zero-stuff, lowpass at the high rate, decimate.

    Kept as the readable statement of what a rational resample IS, and as the
    oracle the polyphase form is tested against. Not called by the library:
    it filters ``up`` times more samples than it keeps, most of them zeros.
    """
    iq = np.asarray(iq)
    n = len(iq)
    cplx = np.iscomplexobj(iq)
    dtype = np.complex64 if cplx else np.float64
    ups = np.zeros(n * up, dtype=dtype)
    ups[::up] = iq
    delay = (len(taps) - 1) // 2
    ups = np.concatenate([ups, np.zeros(delay, dtype=dtype)])
    filt = fir_apply(ups, taps)[delay:]
    out = filt[::down][:int(np.ceil(n * up / down))]
    return out.astype(dtype)


def resample_poly(iq, up, down, half_len=10, window="hamming"):
    """Rational resample by up/down. OUR implementation (polyphase).

    A rational resample is: insert (up-1) zeros between samples, lowpass at the
    upsampled rate, keep every down-th sample. Doing it literally filters
    ``up`` times more samples than it keeps and multiplies mostly by zeros
    (``_resample_poly_direct`` -- 300x slower than this at 24/125). The
    polyphase form computes only the outputs that survive: every output sample
    sits at one of ``up`` phases of the zero-stuffed grid, and at phase p only
    the taps h[p], h[p+up], h[p+2*up], ... ever meet a non-zero input, so each
    phase is an ordinary convolution of the INPUT with that short sub-filter,
    evaluated at a stride of ``down``. The result is the same filter and the
    same samples as the direct form (tested to 1e-12), at ``1/up`` of the work.

    The anti-alias filter is designed by scipy (firwin, our implementation
    rule); its group delay is compensated so output aligns with input, and the
    output length is ceil(n * up / down), both matching
    ``scipy.signal.resample_poly`` (verified in tests).

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
    dtype = np.complex64 if cplx else np.float64
    out_len = int(np.ceil(n * up / down))
    taps = _antialias_taps(up, down, half_len=half_len, window=window)
    delay = (len(taps) - 1) // 2

    # Output m is the zero-stuffed/filtered sample at high-rate index
    #     j = m*down + delay      (delay compensation folded in)
    # and that sample is  sum_k taps[k] * ups[j - k], where ups[i] is non-zero
    # only when i % up == 0, with ups[i] = iq[i // up]. Writing j = q*up + p
    # (p = j % up is the output's PHASE), the surviving terms are k = p + i*up:
    #     out[m] = sum_i taps[p + i*up] * iq[q - i]        (0 <= q - i < n)
    # i.e. the full convolution of iq with sub-filter taps[p::up], read at
    # index q. Outputs sharing a phase are m = m0 + r*up (up and down are
    # coprime after reduction), with q advancing by `down` per step.
    x = iq.astype(np.complex64 if cplx else np.float64)
    out = np.empty(out_len, dtype=dtype)
    m = np.arange(out_len)
    j = m * down + delay
    phase = j % up
    q = j // up
    for p in range(up):
        sel = np.nonzero(phase == p)[0]
        if sel.size == 0:
            continue
        sub = taps[p::up]
        if cplx:
            conv = (np.convolve(x.real, sub, mode="full")
                    + 1j * np.convolve(x.imag, sub, mode="full"))
        else:
            conv = np.convolve(x, sub, mode="full")
        qi = q[sel]
        valid = qi < len(conv)          # past the end: the filter sees zeros
        vals = np.zeros(sel.size, dtype=dtype)
        vals[valid] = conv[qi[valid]]
        out[sel] = vals
    return out


def decimate(iq, factor, half_len=10):
    """Lowpass then keep every ``factor``-th sample. OUR code (via resample)."""
    return resample_poly(iq, 1, factor, half_len=half_len)


def interpolate(iq, factor, half_len=10):
    """Upsample by ``factor`` with interpolation filtering. OUR code."""
    return resample_poly(iq, factor, 1, half_len=half_len)
