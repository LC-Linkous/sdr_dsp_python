#! /usr/bin/python3
"""spectrum_analyzer.py -- a proper spectrum display of a capture.

Computes the power spectral density with correct dB scaling, estimates the
noise floor, marks the strongest peaks, and plots it. The "do it right" version
of a spectrum view -- the analyzer you'd actually use to see what's in a band.

Needs: matplotlib (examples extra). Runs on a file (no hardware), or pass an
array in if you call run_analyzer() from your own live-capture script.

Usage:
    python examples/spectrum_analyzer.py sample_data/fm_2Msps.iq
    python examples/spectrum_analyzer.py capture.iq --nfft 4096 --peaks 5
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.io import load_iq
from sdr_dsp.core import psd


def find_peaks(freqs, psd_db, n_peaks, min_separation_bins=10):
    """Crude peak picker: the n strongest bins, kept apart by a separation."""
    order = np.argsort(psd_db)[::-1]
    picked = []
    for idx in order:
        if all(abs(idx - q) > min_separation_bins for q in picked):
            picked.append(idx)
        if len(picked) >= n_peaks:
            break
    return [(freqs[i], psd_db[i]) for i in sorted(picked)]


def run_analyzer(iq, sample_rate, center_freq=0.0, nfft=2048, n_peaks=5):
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann",
                        center_freq=center_freq)
    noise_floor = float(np.median(psd_db))
    peaks = find_peaks(freqs, psd_db, n_peaks)
    return freqs, psd_db, noise_floor, peaks


def main():
    p = argparse.ArgumentParser(description="Spectrum analyzer for a capture.")
    p.add_argument("iq_file")
    p.add_argument("--nfft", type=int, default=2048)
    p.add_argument("--peaks", type=int, default=5)
    p.add_argument("--no-plot", action="store_true", help="text output only")
    args = p.parse_args()

    iq, meta = load_iq(args.iq_file)
    g = meta.get("global", {})
    caps = meta.get("captures", [{}])
    fs = float(g.get("core:sample_rate", 1.0))
    fc = float(caps[0].get("core:frequency", 0.0)) if caps else 0.0

    freqs, psd_db, floor, peaks = run_analyzer(iq, fs, fc, args.nfft, args.peaks)

    print(f"[*] {len(iq):,} samples @ {fs/1e6:g} Msps, center {fc/1e6:g} MHz")
    print(f"[*] noise floor ~ {floor:.1f} dB")
    print(f"[*] top {len(peaks)} peaks:")
    for f, db in sorted(peaks, key=lambda x: -x[1]):
        unit = f / 1e6 if fc else f
        print(f"      {db:7.1f} dB @ {unit:.4f} {'MHz' if fc else 'Hz'}  "
              f"({db - floor:+.1f} dB over floor)")

    if args.no_plot:
        return 0
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to plot: pip install matplotlib)")
        return 0

    x = freqs / 1e6 if fc else freqs
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, psd_db, lw=0.7, color="#1f77b4")
    ax.axhline(floor, color="gray", ls="--", lw=0.8, label=f"noise floor "
               f"{floor:.0f} dB")
    for f, db in peaks:
        xf = f / 1e6 if fc else f
        ax.plot(xf, db, "rv", ms=7)
        ax.annotate(f"{xf:.3f}", (xf, db), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("frequency (MHz)" if fc else "frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"spectrum -- {args.iq_file}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
