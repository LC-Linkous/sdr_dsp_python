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

from typing import Any

import numpy as np

from scipy import signal as _sig

from ._validate import positive, positive_int, one_of

# the scipy.signal.firwin window names this module accepts by string
_WINDOWS = ("hamming", "hann", "blackman", "boxcar", "bartlett", "blackmanharris",
            "kaiser", "flattop", "nuttall")


def _check_design(sample_rate: float, num_taps: int, window: Any) -> tuple[float, int]:
    sample_rate = positive("sample_rate", sample_rate)
    num_taps = positive_int("num_taps", num_taps)
    if isinstance(window, str):
        one_of("window", window, _WINDOWS)
    return sample_rate, num_taps


# --------------------------------------------------------------------------
# DESIGN (scipy) -- produce filter coefficients
# --------------------------------------------------------------------------
def design_lowpass(
    cutoff_hz: float, sample_rate: float, num_taps: int = 101, window: Any = "hamming"
) -> np.ndarray:
    """Design a lowpass FIR. Returns tap coefficients (numpy array).

    cutoff_hz:   passband edge in Hz.
    sample_rate: Hz.
    num_taps:    filter length (odd recommended for a linear-phase Type-I FIR).
    """
    sample_rate, num_taps = _check_design(sample_rate, num_taps, window)
    nyq = sample_rate / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff_hz must be in (0, {nyq}) -- 0 < cutoff < sample_rate/2 -- "
                         f"got {cutoff_hz}")
    return _sig.firwin(num_taps, cutoff_hz / nyq, window=window).astype(
        np.float64)


def design_bandpass(
    low_hz: float,
    high_hz: float,
    sample_rate: float,
    num_taps: int = 101,
    window: Any = "hamming",
) -> np.ndarray:
    """Design a bandpass FIR. Returns tap coefficients."""
    sample_rate, num_taps = _check_design(sample_rate, num_taps, window)
    nyq = sample_rate / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(f"need 0 < low_hz < high_hz < sample_rate/2, got low_hz={low_hz}, "
                         f"high_hz={high_hz}, sample_rate/2={nyq}")
    return _sig.firwin(num_taps, [low_hz / nyq, high_hz / nyq],
                       window=window, pass_zero=False).astype(np.float64)


def design_highpass(
    cutoff_hz: float, sample_rate: float, num_taps: int = 101, window: Any = "hamming"
) -> np.ndarray:
    """Design a highpass FIR. Returns tap coefficients."""
    sample_rate, num_taps = _check_design(sample_rate, num_taps, window)
    nyq = sample_rate / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff_hz must be in (0, {nyq}) -- 0 < cutoff < sample_rate/2 -- "
                         f"got {cutoff_hz}")
    n = num_taps
    if n % 2 == 0:
        n += 1  # highpass needs odd length (Type-I)
    return _sig.firwin(n, cutoff_hz / nyq, window=window,
                       pass_zero=False).astype(np.float64)


# --------------------------------------------------------------------------
# APPLY (sdr_dsp's own) -- the convolution that processes the signal
# --------------------------------------------------------------------------
def fir_apply(iq: Any, taps: Any) -> np.ndarray:
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


def fir_apply_centered(iq: Any, taps: Any) -> np.ndarray:
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
