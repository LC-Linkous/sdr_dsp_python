"""Plot sink: matplotlib helpers for spectra and spectrograms.

Optional -- needs matplotlib (the `plotting` examples extra). These wrap the
common "show a spectrum / show a waterfall" plotting the examples repeat, so an
example can hand off a finished analysis in one call. Importing this module
without matplotlib raises a clear error only when a plot function is called.
"""

from __future__ import annotations

import numpy as np

from ..core.spectral import psd as _psd, spectrogram as _spectrogram


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("plotting needs matplotlib: pip install matplotlib "
                          "(or `uv sync --extra plotting`)") from e


def plot_spectrum(iq, sample_rate, center_freq=0.0, nfft=2048, title=None,
                  show=True):
    """Plot the PSD of a signal. Returns (fig, ax)."""
    plt = _plt()
    freqs, p_db = _psd(iq, sample_rate, nfft=nfft, center_freq=center_freq)
    x = freqs / 1e6 if center_freq else freqs
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, p_db, lw=0.7)
    ax.set_xlabel("frequency (MHz)" if center_freq else "frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(title or "spectrum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_spectrogram(iq, sample_rate, center_freq=0.0, nfft=1024, overlap=0.5,
                     title=None, show=True):
    """Plot a spectrogram waterfall. Returns (fig, ax)."""
    plt = _plt()
    freqs, times, sxx = _spectrogram(iq, sample_rate, nfft=nfft,
                                     overlap=overlap, center_freq=center_freq)
    if sxx.shape[0] == 0:
        raise ValueError("not enough samples for one FFT frame")
    x0 = freqs[0] / 1e6 if center_freq else freqs[0]
    x1 = freqs[-1] / 1e6 if center_freq else freqs[-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(sxx, aspect="auto", origin="lower", cmap="turbo",
                   extent=[x0, x1, times[0] * 1e3, times[-1] * 1e3],
                   vmin=np.percentile(sxx, 20), vmax=np.percentile(sxx, 99.5))
    fig.colorbar(im, ax=ax, label="power (dB)")
    ax.set_xlabel("frequency (MHz)" if center_freq else "frequency (Hz)")
    ax.set_ylabel("time (ms)")
    ax.set_title(title or "spectrogram")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
