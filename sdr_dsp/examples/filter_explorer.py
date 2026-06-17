#! /usr/bin/python3
"""filter_explorer.py -- "what does a filter actually do?" A teaching tool.

Designs a filter and shows it two ways:
  1. its frequency response (the FFT of the taps) -- the passband, the stopband,
     the roll-off, the -3 dB point.
  2. a real signal's spectrum before and after the filter -- proof it does what
     the response promises.

Lets students see the design/response/effect connection directly. Try changing
the filter type, cutoff, and number of taps to watch the trade-offs (more taps =
sharper roll-off but more delay/cost).

Needs: matplotlib (examples extra). Synthesizes a multi-tone test signal, or
filters a real capture if you pass one.

Usage:
    python examples/filter_explorer.py                       # synthetic
    python examples/filter_explorer.py --type bandpass --low 80e3 --high 120e3
    python examples/filter_explorer.py capture.iq --type lowpass --cutoff 100e3
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (
    design_lowpass, design_bandpass, design_highpass, fir_apply, psd,
)


def main():
    p = argparse.ArgumentParser(description="Explore filter design and effect.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--type", choices=["lowpass", "bandpass", "highpass"],
                   default="lowpass")
    p.add_argument("--cutoff", type=float, default=100e3)
    p.add_argument("--low", type=float, default=80e3)
    p.add_argument("--high", type=float, default=120e3)
    p.add_argument("--taps", type=int, default=127)
    p.add_argument("--rate", type=float, default=1e6, help="synth sample rate")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    # source signal
    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file, count=500_000)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        n = 200_000
        t = np.arange(n) / fs
        # tones spread across the band so you can see what's kept/removed
        iq = np.zeros(n, dtype=np.complex64)
        for f in (-300e3, -150e3, 0, 100e3, 250e3):
            iq += np.exp(2j * np.pi * f * t).astype(np.complex64)

    # design the chosen filter
    if args.type == "lowpass":
        taps = design_lowpass(args.cutoff, fs, num_taps=args.taps)
        desc = f"lowpass @ {args.cutoff/1e3:g} kHz"
    elif args.type == "bandpass":
        taps = design_bandpass(args.low, args.high, fs, num_taps=args.taps)
        desc = f"bandpass {args.low/1e3:g}-{args.high/1e3:g} kHz"
    else:
        taps = design_highpass(args.cutoff, fs, num_taps=args.taps)
        desc = f"highpass @ {args.cutoff/1e3:g} kHz"

    filtered = fir_apply(iq, taps)

    # response of the taps
    H = np.abs(np.fft.rfft(taps, 8192))
    fH = np.fft.rfftfreq(8192, 1.0 / fs) / 1e3
    H_db = 20 * np.log10(H / H.max() + 1e-12)

    # spectra before/after
    f1, p_before = psd(iq, fs, nfft=2048)
    f2, p_after = psd(filtered, fs, nfft=2048)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle(f"filter explorer -- {desc}, {args.taps} taps", fontsize=12)

    ax1.plot(fH, H_db, color="#d62728")
    ax1.axhline(-3, color="gray", ls="--", lw=0.7, label="-3 dB")
    ax1.set_xlim(0, fs / 2 / 1e3)
    ax1.set_ylim(-80, 5)
    ax1.set_xlabel("frequency (kHz)")
    ax1.set_ylabel("response (dB)")
    ax1.set_title("1. filter frequency response (FFT of the taps)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(f1 / 1e3, p_before, lw=0.7, alpha=0.6, label="before")
    ax2.plot(f2 / 1e3, p_after, lw=0.8, label="after")
    ax2.set_xlabel("frequency (kHz)")
    ax2.set_ylabel("PSD (dB)")
    ax2.set_title("2. signal spectrum before vs after")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
