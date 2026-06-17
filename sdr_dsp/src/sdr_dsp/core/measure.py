"""Measurement: power, SNR, occupied bandwidth. OUR code (simple math on IQ).

These build on the spectral module and tie back to hackrfpy's relative_power_db
for a dB reference. None of this is in scipy -- it is radio-specific.
"""

from __future__ import annotations

import numpy as np

from .spectral import psd


def power_dbfs(iq):
    """Mean power of a complex signal in dBFS (dB relative to |amp|=1)."""
    iq = np.asarray(iq)
    if len(iq) == 0:
        return float("-inf")
    p = float(np.mean(iq.real.astype(np.float64) ** 2
                      + iq.imag.astype(np.float64) ** 2))
    return 10.0 * np.log10(p + 1e-20)


def snr_db(iq, sample_rate, signal_band_hz, nfft=1024):
    """Estimate SNR by comparing in-band power to out-of-band (noise) power.

    signal_band_hz: (low, high) frequency range (relative to center) holding
                    the signal. Everything else in the spectrum is treated as
                    noise. A coarse but useful estimate.
    """
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    psd_lin = 10.0 ** (psd_db / 10.0)
    lo, hi = signal_band_hz
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any() or in_band.all():
        raise ValueError("signal_band must cover part (not all) of the span")
    sig_p = float(np.mean(psd_lin[in_band]))
    noise_p = float(np.mean(psd_lin[~in_band]))
    return 10.0 * np.log10(sig_p / (noise_p + 1e-20))


def occupied_bandwidth(iq, sample_rate, fraction=0.99, nfft=1024):
    """Bandwidth containing ``fraction`` of the total power (e.g. 99%).

    Returns bandwidth in Hz. Integrates the PSD and finds the central band
    holding the requested fraction of total power.
    """
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    p = 10.0 ** (psd_db / 10.0)
    total = float(np.sum(p))
    if total <= 0:
        return 0.0
    # cumulative from the spectrum center outward
    order = np.argsort(np.abs(freqs))  # nearest-to-center first
    cum = np.cumsum(p[order])
    idx = np.searchsorted(cum, fraction * total)
    idx = min(idx, len(order) - 1)
    bw = 2.0 * float(np.abs(freqs[order][idx]))
    return bw
