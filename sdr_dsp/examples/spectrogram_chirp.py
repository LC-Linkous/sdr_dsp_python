#! /usr/bin/python3
"""spectrogram_chirp.py -- a frequency sweep (chirp) in time-frequency.

A chirp is a tone whose frequency slides over time. In a spectrogram it draws a
clean diagonal line -- one of the most satisfying visuals in DSP, and an instant
lesson in why time-frequency analysis matters (a plain FFT would just smear the
sweep across the whole band).

Synthesizes a chirp and shows its spectrogram. Pure synthetic, no hardware.

Needs: matplotlib (examples extra).

Usage:
    python examples/spectrogram_chirp.py
    python examples/spectrogram_chirp.py --f0 -400e3 --f1 400e3 --rate 1e6
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import spectrogram


def main():
    p = argparse.ArgumentParser(description="Chirp spectrogram demo.")
    p.add_argument("--f0", type=float, default=-400e3, help="start freq Hz")
    p.add_argument("--f1", type=float, default=400e3, help="end freq Hz")
    p.add_argument("--rate", type=float, default=1e6)
    p.add_argument("--dur", type=float, default=0.1, help="seconds")
    p.add_argument("--nfft", type=int, default=512)
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    fs = args.rate
    n = int(fs * args.dur)
    t = np.arange(n) / fs
    # linear chirp: frequency goes f0 -> f1; phase is the integral of frequency
    f_inst = args.f0 + (args.f1 - args.f0) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    iq = np.exp(1j * phase).astype(np.complex64)

    freqs, times, sxx = spectrogram(iq, fs, nfft=args.nfft, overlap=0.75)
    print(f"[*] chirp {args.f0/1e3:g} -> {args.f1/1e3:g} kHz over "
          f"{args.dur*1e3:g} ms; spectrogram {sxx.shape}")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(sxx, aspect="auto", origin="lower", cmap="turbo",
                   extent=[freqs[0] / 1e3, freqs[-1] / 1e3,
                           times[0] * 1e3, times[-1] * 1e3])
    fig.colorbar(im, ax=ax, label="power (dB)")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("time (ms)")
    ax.set_title("a chirp is a diagonal line in time-frequency")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
