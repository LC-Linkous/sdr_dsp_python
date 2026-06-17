#! /usr/bin/python3
"""waterfall.py -- offline spectrogram (time-frequency heatmap) of a capture.

Shows how a recording's spectrum evolves over time: frequency on one axis, time
on the other, power as color. Great for spotting bursts, hopping signals, and
chirps that a single averaged spectrum would smear away.

This is the OFFLINE analysis view (a saved file), distinct from a live
streaming waterfall. Built on the library's spectrogram().

Needs: matplotlib (examples extra). Runs on a file (no hardware).

Usage:
    python examples/waterfall.py sample_data/fm_2Msps.iq
    python examples/waterfall.py capture.iq --nfft 1024 --overlap 0.75
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.io import load_iq
from sdr_dsp.core import spectrogram


def main():
    p = argparse.ArgumentParser(description="Offline spectrogram waterfall.")
    p.add_argument("iq_file")
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=4_000_000)
    args = p.parse_args()

    iq, meta = load_iq(args.iq_file, count=args.max_samples)
    g = meta.get("global", {})
    caps = meta.get("captures", [{}])
    fs = float(g.get("core:sample_rate", 1.0))
    fc = float(caps[0].get("core:frequency", 0.0)) if caps else 0.0

    freqs, times, sxx = spectrogram(iq, fs, nfft=args.nfft,
                                    overlap=args.overlap, center_freq=fc)
    print(f"[*] {len(iq):,} samples -> {sxx.shape[0]} time slices x "
          f"{sxx.shape[1]} freq bins")
    if sxx.shape[0] == 0:
        print("[!] not enough samples for one FFT frame")
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    x0 = (freqs[0]) / 1e6 if fc else freqs[0]
    x1 = (freqs[-1]) / 1e6 if fc else freqs[-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(sxx, aspect="auto", origin="lower", cmap="turbo",
                   extent=[x0, x1, times[0] * 1e3, times[-1] * 1e3],
                   vmin=np.percentile(sxx, 20), vmax=np.percentile(sxx, 99.5))
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("power (dB)")
    ax.set_xlabel("frequency (MHz)" if fc else "frequency (Hz)")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"waterfall -- {args.iq_file}")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
