"""Spectral analysis: numpy does the FFT, sdr_dsp owns everything around it.

The FFT itself is standardized and deep -- we use ``numpy.fft`` and do not
reinvent it. What sdr_dsp owns is the radio-useful framing: window application,
power scaling to a proper PSD, frame averaging (Welch-style), spectrogram
assembly, and dB conversion. That surrounding logic is where the practical
correctness lives, and it is the library's own code.
"""

from __future__ import annotations

import numpy as np


def get_window(window, n):
    """Window vector of length n by name. sdr_dsp's own small set (hann,
    hamming, blackman, boxcar/rect), matching what psd()/spectrogram() use.
    Public so examples and callers don't reach for the underscore alias.

    Deliberately numpy-based rather than scipy.signal.get_window: these use
    the symmetric np.* windows the spectral functions are calibrated against,
    not scipy's periodic default, so a window pulled from here matches the
    one psd() applies internally.
    """
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


# internal callers predate the public name; keep the alias
_get_window = get_window


def psd(iq, sample_rate, nfft=1024, window="hann", center_freq=0.0,
        scaling="density"):
    """Power spectral density of a complex signal via Welch averaging. OUR code.

    Splits the signal into nfft-length frames, windows each, FFTs (numpy),
    accumulates |X|^2, averages, and scales. Returns (freqs_hz, psd_db).

    scaling:   "density" (default) -> dB relative to 1 (unit^2)/Hz: the
               averaged |FFT|^2 divided by sample_rate * sum(window^2). The
               level of a tone then depends on nfft and sample_rate (it is a
               density), which is right for comparing noise floors and for
               integrating power over a band.
               "dbfs" -> dB relative to full scale PER BIN: divided by
               sum(window)^2, so a full-scale complex tone (|x| = 1, the level
               a HackRF/RTL-SDR int8 sample of +/-127 lands on after /128)
               sitting on a bin reads 0 dBFS, independent of nfft and
               sample_rate. This is what a spectrum display wants. A tone
               between bins reads lower by the window's scalloping loss (up to
               1.4 dB for Hann).
    freqs_hz:  frequency axis centered on center_freq, fftshifted (low->high).
    psd_db:    10*log10 of the averaged, scaled power spectrum.

    Raises ValueError if the signal is shorter than one frame (len(iq) < nfft):
    there is no full frame to transform. Pass a smaller nfft, or pad the signal
    yourself if that is what you mean. (An earlier version silently returned a
    flat -200 dB floor here, which made every downstream measurement --
    occupied_bandwidth, snr_db -- confidently wrong on short inputs.)
    """
    if scaling not in ("density", "dbfs"):
        raise ValueError(f"scaling must be 'density' or 'dbfs', got {scaling!r}")
    iq = np.asarray(iq)
    nfft = int(nfft)
    if nfft < 1:
        raise ValueError(f"nfft must be >= 1, got {nfft}")
    if len(iq) < nfft:
        raise ValueError(
            f"psd needs at least one full frame: len(iq)={len(iq)} < nfft={nfft}. "
            "Use a smaller nfft or pad the signal.")
    win = _get_window(window, nfft).astype(np.float64)
    win_power = np.sum(win ** 2)  # for power normalization

    nframes = len(iq) // nfft
    acc = np.zeros(nfft, dtype=np.float64)
    for k in range(nframes):
        seg = iq[k * nfft:(k + 1) * nfft]
        spec = np.fft.fftshift(np.fft.fft(seg * win))
        acc += (np.abs(spec) ** 2)
    acc /= nframes
    if scaling == "density":
        # window power and sample rate -> power per Hz
        acc /= (sample_rate * win_power)
    else:
        # coherent gain of the window -> a full-scale on-bin tone is 1.0
        acc /= float(np.sum(win)) ** 2
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
