"""Spectral analysis: numpy does the FFT, sdr_dsp owns everything around it.

The FFT itself is standardized and deep -- we use ``numpy.fft`` and do not
reinvent it. What sdr_dsp owns is the radio-useful framing: window application,
power scaling to a proper PSD, frame averaging (Welch-style), spectrogram
assembly, and dB conversion. That surrounding logic is where the practical
correctness lives, and it is the library's own code.
"""

from __future__ import annotations

import numpy as np


def _get_window(window, n):
    if window is None:
        return np.ones(n)
    if isinstance(window, np.ndarray):
        return window
    return {
        "hann": np.hanning,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "rect": lambda m: np.ones(m),
        "boxcar": lambda m: np.ones(m),
    }[window](n)


def psd(iq, sample_rate, nfft=1024, window="hann", center_freq=0.0):
    """Power spectral density of a complex signal via Welch averaging. OUR code.

    Splits the signal into nfft-length frames, windows each, FFTs (numpy),
    accumulates |X|^2, averages, and scales. Returns (freqs_hz, psd_db).

    freqs_hz:  frequency axis centered on center_freq, fftshifted (low->high).
    psd_db:    10*log10 of the averaged power spectrum.
    """
    iq = np.asarray(iq)
    nfft = int(nfft)
    win = _get_window(window, nfft).astype(np.float64)
    win_power = np.sum(win ** 2)  # for power normalization

    nframes = max(1, len(iq) // nfft)
    acc = np.zeros(nfft, dtype=np.float64)
    for k in range(nframes):
        seg = iq[k * nfft:(k + 1) * nfft]
        if len(seg) < nfft:
            break
        spec = np.fft.fftshift(np.fft.fft(seg * win))
        acc += (np.abs(spec) ** 2)
    acc /= nframes
    # normalize: window power, fft size, sample rate -> PSD in power/Hz
    acc /= (sample_rate * win_power)
    psd_db = 10.0 * np.log10(acc + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    freqs = freqs + center_freq
    return freqs, psd_db


def spectrogram(iq, sample_rate, nfft=1024, overlap=0.5, window="hann",
                center_freq=0.0):
    """Time-frequency spectrogram. OUR code (numpy FFT per frame).

    Returns (freqs_hz, times_s, sxx_db) where sxx_db has shape
    (n_frames, nfft): one spectrum row per hop.
    """
    iq = np.asarray(iq)
    nfft = int(nfft)
    win = _get_window(window, nfft).astype(np.float64)
    win_power = np.sum(win ** 2)
    hop = max(1, int(nfft * (1.0 - overlap)))

    rows = []
    times = []
    for start in range(0, len(iq) - nfft + 1, hop):
        seg = iq[start:start + nfft]
        spec = np.fft.fftshift(np.fft.fft(seg * win))
        power = (np.abs(spec) ** 2) / (sample_rate * win_power)
        rows.append(10.0 * np.log10(power + 1e-20))
        times.append((start + nfft / 2) / sample_rate)

    sxx_db = np.array(rows) if rows else np.zeros((0, nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate)) + center_freq
    return freqs, np.array(times), sxx_db
