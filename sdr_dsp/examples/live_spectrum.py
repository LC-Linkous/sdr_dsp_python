#! /usr/bin/python3
"""live_spectrum.py -- live spectrum display from a HackRF.

Streams IQ at a fixed frequency and shows the running power spectrum -- "what's
on this frequency right now?" A live, single-frequency analyzer (distinct from
hackrfpy's own sweep waterfall, which scans a wide band).

Requires hackrfpy + hackrf-tools, and matplotlib for the plot:
    pip install hackrfpy matplotlib

Usage:
    python examples/live_spectrum.py --freq 100e6 --rate 2e6
    python examples/live_spectrum.py --freq 2.437e9   # WiFi ch 6
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "examples")
from sdr_dsp.core import psd


def main():
    p = argparse.ArgumentParser(description="Live spectrum display.")
    p.add_argument("--freq", type=float, required=True)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--nfft", type=int, default=2048)
    p.add_argument("--lna", type=int, default=16)
    p.add_argument("--vga", type=int, default=20)
    args = p.parse_args()

    try:
        from hackrf_capture import HackRFCapture
    except ImportError as e:
        print(f"needs hackrfpy: pip install hackrfpy  ({e})", file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    fs = args.rate
    fc = args.freq
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))
    line, = ax.plot(np.zeros(args.nfft))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"live spectrum @ {fc/1e6:g} MHz")
    ax.grid(alpha=0.3)
    ax.set_ylim(-120, -20)

    print(f"[*] live @ {fc/1e6:g} MHz; close the window to stop")
    try:
        with HackRFCapture(fc, fs, lna=args.lna, vga=args.vga,
                           block_size=args.nfft * 16) as src:
            init = False
            for iq in src.blocks():
                if len(iq) < args.nfft:
                    continue
                freqs, p_db = psd(iq, fs, nfft=args.nfft, center_freq=fc)
                if not init:
                    line.set_xdata(freqs / 1e6)
                    ax.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
                    init = True
                line.set_ydata(p_db)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if not plt.fignum_exists(fig.number):
                    break
    except KeyboardInterrupt:
        pass
    print("[*] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
