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


def find_bursts(iq, sample_rate=None, threshold=None, min_gap=0, min_len=1):
    """Find where signal energy is present: burst start/stop indices. OUR code.

    Thresholds the magnitude envelope and returns the spans where it's above the
    threshold -- "where is the signal?" for packet/burst captures. The decoder
    examples did this ad-hoc; this is the reusable version.

    threshold: envelope level for "on". If None, uses the midpoint between the
               envelope's 1st percentile (the noise floor) and its peak. The
               floor is a low PERCENTILE, not the median, deliberately: the
               median is only the noise floor when the record is mostly noise.
               On a capture dominated by one long burst (a triggered packet
               capture), the median IS the signal level, and a median-based
               threshold lands above the signal and shreds one burst into
               fragments. The percentile floor handles both regimes, as long
               as at least ~1% of the record is signal-free. If your record
               has NO quiet samples at all, or bursts sit near the noise
               level, set threshold explicitly -- an automatic threshold is a
               convenience, not a measurement.
    min_gap:   merge bursts separated by fewer than this many samples (bridges
               brief dropouts within one packet).
    min_len:   discard bursts shorter than this (rejects noise blips).

    Returns a list of (start, stop) sample-index pairs (stop exclusive). If
    sample_rate is given, also accepts/returns nothing different -- indices are
    always in samples (convert to time yourself: start/sample_rate).
    """
    env = np.abs(np.asarray(iq))
    if len(env) == 0:
        return []
    if threshold is None:
        floor = float(np.percentile(env, 1))
        pk = float(np.max(env))
        threshold = floor + 0.5 * (pk - floor)
    on = env > threshold
    if not on.any():
        return []
    # find rising/falling edges of the boolean "on" mask
    edges_ = np.diff(on.astype(np.int8))
    starts = list(np.nonzero(edges_ == 1)[0] + 1)
    stops = list(np.nonzero(edges_ == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        stops = stops + [len(on)]
    spans = list(zip(starts, stops))
    # merge close spans
    if min_gap > 0 and spans:
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s - merged[-1][1] <= min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        spans = merged
    # drop short spans
    spans = [(s, e) for s, e in spans if e - s >= min_len]
    return spans


def estimate_cfo(iq, sample_rate, nfft=None):
    """Estimate a signal's carrier frequency offset from band center. OUR code.

    Finds the dominant spectral component -- where the signal actually sits
    relative to 0 Hz. This MEASURES the offset; it does NOT apply any
    correction (correcting would change the data, and that's the user's call --
    pass the result to frequency_shift / tune_to_baseband if you want to
    correct). Returns the offset in Hz.

    For a clean single-carrier signal this is just the FFT peak. For modulated
    signals it estimates the spectral centroid of the strongest region.

    NOT for FSK. An FSK burst's strongest components are the mark/space tones
    at +/-deviation_hz, so this returns roughly +/-deviation, NOT the carrier
    offset -- and "correcting" with it moves the whole signal by a deviation,
    which is worse than no correction. For FSK, threshold at the offset
    directly instead: fsk_demod(iq, fs, threshold_hz="auto") uses the
    amplitude-weighted mean of the instantaneous frequency, which IS the
    offset when mark/space time is roughly balanced.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0:
        return 0.0
    if nfft is None:
        nfft = min(len(iq), 8192)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq[:nfft], nfft)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    return float(freqs[int(np.argmax(spec))])
