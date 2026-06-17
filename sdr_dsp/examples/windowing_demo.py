#! /usr/bin/python3
"""windowing_demo.py -- why FFT windows matter. A teaching visualization.

Takes one signal and computes its spectrum three ways: rectangular (no window),
Hann, and Blackman. Shows how windowing trades main-lobe width for side-lobe
(leakage) suppression -- the reason real spectrum analysis always windows.

The setup deliberately uses a tone whose frequency falls BETWEEN FFT bins, which
is exactly when leakage is worst and windows help most.

Needs: matplotlib (examples extra). No hardware -- synthesizes the signal.

Usage:
    python examples/windowing_demo.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core.spectral import _get_window


def main():
    p = argparse.ArgumentParser(description="Windowing / spectral leakage demo.")
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--rate", type=float, default=1_000_000)
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    n = args.nfft
    fs = args.rate
    # a tone deliberately BETWEEN bins (bin spacing is fs/n; offset by 0.5 bin)
    bin_hz = fs / n
    f_tone = bin_hz * (n // 4 + 0.5)      # half a bin off a bin center
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f_tone * t).astype(np.complex64)

    fig, ax = plt.subplots(figsize=(11, 6))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs)) / 1e3   # kHz

    for name in ("rect", "hann", "blackman"):
        win = _get_window(name, n)
        spec = np.fft.fftshift(np.fft.fft(x * win))
        mag_db = 20 * np.log10(np.abs(spec) / np.abs(spec).max() + 1e-12)
        ax.plot(freqs, mag_db, lw=1.0, label=name)

    ax.axvline(f_tone / 1e3, color="k", ls="--", lw=0.7, alpha=0.5,
               label="true tone")
    ax.set_xlim((f_tone - 40 * bin_hz) / 1e3, (f_tone + 40 * bin_hz) / 1e3)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("magnitude (dB, normalized)")
    ax.set_title("spectral leakage: a tone between bins, three windows\n"
                 "rect = narrow peak but high side-lobes; "
                 "blackman = wide peak but low leakage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
