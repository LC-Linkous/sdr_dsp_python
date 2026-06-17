#! /usr/bin/python3
"""dc_offset_demo.py -- the center DC spike, and how to remove it.

Every direct-conversion SDR (the HackRF included) leaks its local oscillator
into the center of the band, producing a bright spike at 0 Hz that isn't a real
signal. This demo shows the spike on a capture (or synthetic signal), then
removes the DC offset and shows the cleaned spectrum.

Removing it is just subtracting the mean (the DC component) -- one line -- but
seeing the before/after makes the artifact memorable.

Needs: matplotlib (examples extra). Works on a file or a synthetic signal.

Usage:
    python examples/dc_offset_demo.py                 # synthetic
    python examples/dc_offset_demo.py capture.iq
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import psd, remove_dc




def main():
    p = argparse.ArgumentParser(description="DC offset / LO-leakage demo.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=2e6)
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file, count=500_000)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        n = 200_000
        t = np.arange(n) / fs
        # a real tone off-center + a big DC offset (the LO leakage)
        iq = (0.3 * np.exp(2j * np.pi * 300e3 * t) + (0.5 + 0.2j)
              + 0.01 * (np.random.randn(n) + 1j * np.random.randn(n))
              ).astype(np.complex64)

    cleaned = remove_dc(iq)
    print(f"[*] DC offset removed: mean was "
          f"({np.mean(iq).real:+.3f}, {np.mean(iq).imag:+.3f})")

    f1, p_before = psd(iq, fs, nfft=2048)
    f2, p_after = psd(cleaned, fs, nfft=2048)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(f1 / 1e3, p_before, lw=0.8, label="with DC spike", alpha=0.7)
    ax.plot(f2 / 1e3, p_after, lw=0.8, label="DC removed")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("DC offset / LO leakage: the center spike is an artifact, "
                 "not a signal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
