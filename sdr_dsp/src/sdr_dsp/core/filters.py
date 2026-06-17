"""Filtering: scipy designs the coefficients, sdr_dsp applies them.

This module embodies the library's implementation rule. Designing optimal filter
taps (windowed-sinc, Butterworth, Parks-McClellan) is solved, hard, and stable
math -- so we use ``scipy.signal`` for it. *Applying* the filter -- the
convolution that actually processes the signal -- is sdr_dsp's own code, because
that is the operation we want to own, understand, and keep stable regardless of
scipy's API. The own FIR application is verified against ``scipy.signal.lfilter``
in the tests (scipy as oracle).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _sig


# --------------------------------------------------------------------------
# DESIGN (scipy) -- produce filter coefficients
# --------------------------------------------------------------------------
def design_lowpass(cutoff_hz, sample_rate, num_taps=101, window="hamming"):
    """Design a lowpass FIR. Returns tap coefficients (numpy array).

    cutoff_hz:   passband edge in Hz.
    sample_rate: Hz.
    num_taps:    filter length (odd recommended for a linear-phase Type-I FIR).
    """
    nyq = sample_rate / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff {cutoff_hz} must be in (0, {nyq})")
    return _sig.firwin(int(num_taps), cutoff_hz / nyq, window=window).astype(
        np.float64)


def design_bandpass(low_hz, high_hz, sample_rate, num_taps=101,
                    window="hamming"):
    """Design a bandpass FIR. Returns tap coefficients."""
    nyq = sample_rate / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(f"need 0 < {low_hz} < {high_hz} < {nyq}")
    return _sig.firwin(int(num_taps), [low_hz / nyq, high_hz / nyq],
                       window=window, pass_zero=False).astype(np.float64)


def design_highpass(cutoff_hz, sample_rate, num_taps=101, window="hamming"):
    """Design a highpass FIR. Returns tap coefficients."""
    nyq = sample_rate / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff {cutoff_hz} must be in (0, {nyq})")
    n = int(num_taps)
    if n % 2 == 0:
        n += 1  # highpass needs odd length (Type-I)
    return _sig.firwin(n, cutoff_hz / nyq, window=window,
                       pass_zero=False).astype(np.float64)


# --------------------------------------------------------------------------
# APPLY (sdr_dsp's own) -- the convolution that processes the signal
# --------------------------------------------------------------------------
def fir_apply(iq, taps):
    """Apply an FIR filter to a signal by direct convolution. OUR code.

    Equivalent to ``scipy.signal.lfilter(taps, [1.0], iq)`` but implemented
    here as a full convolution (then truncated to the input length) so the
    filtering operation is the library's own. Works on real or complex input;
    complex IQ is filtered as a whole (numpy.convolve handles complex).

    Returns an array the same length as ``iq`` (the causal 'full' convolution
    truncated to the first len(iq) samples -- matching lfilter's output).
    """
    iq = np.asarray(iq)
    taps = np.asarray(taps, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError("taps must be a non-empty 1-D array")
    # 'full' convolution then keep the first N samples => identical to the
    # causal FIR (lfilter with a=[1]). For complex input, convolve the real
    # and imaginary parts (np.convolve is real-only per call).
    if np.iscomplexobj(iq):
        re = np.convolve(iq.real, taps, mode="full")[:len(iq)]
        im = np.convolve(iq.imag, taps, mode="full")[:len(iq)]
        return (re + 1j * im).astype(np.complex64)
    return np.convolve(iq, taps, mode="full")[:len(iq)].astype(np.float64)


def fir_apply_centered(iq, taps):
    """Apply an FIR with the group delay removed (zero-phase alignment).

    A linear-phase FIR of length L delays the signal by (L-1)/2 samples. For
    analysis where you want the output time-aligned with the input, this
    returns the 'same'-mode convolution (centered), trimming the delay.
    """
    iq = np.asarray(iq)
    taps = np.asarray(taps, dtype=np.float64)
    if np.iscomplexobj(iq):
        re = np.convolve(iq.real, taps, mode="same")
        im = np.convolve(iq.imag, taps, mode="same")
        return (re + 1j * im).astype(np.complex64)
    return np.convolve(iq, taps, mode="same").astype(np.float64)
