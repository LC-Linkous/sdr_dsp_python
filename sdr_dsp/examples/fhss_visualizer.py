#! /usr/bin/python3
"""fhss_visualizer.py -- see a frequency-hopping signal hop across the band.

FHSS jumps its carrier among many channels on a schedule -- Bluetooth classic
does this across 2.4 GHz. You can't easily decode it without the hop sequence,
but you can SEE it beautifully: a spectrogram shows the signal jumping from
channel to channel over time, and we can track which channel it's in at each
moment.

Shows: synthesize (or load) a hopping signal, plot its spectrogram (the hops are
obvious diagonal/scattered blips), and overlay the detected hop track.

Needs: matplotlib (examples extra). Synthesizes a hopper if no file given.

Usage:
    python examples/fhss_visualizer.py
    python examples/fhss_visualizer.py capture.iq
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import spectrogram, fhss_detect_hops


def synth_fhss(fs, n_hops=20, dwell=4000, seed=0):
    """A signal that hops among several channels on a random schedule."""
    rng = np.random.default_rng(seed)
    channels = np.linspace(-0.4 * fs, 0.4 * fs, 8)   # 8 hop channels
    segs = []
    hop_seq = []
    for _ in range(n_hops):
        ch = rng.choice(channels)
        hop_seq.append(ch)
        t = np.arange(dwell) / fs
        segs.append(np.exp(2j * np.pi * ch * t))
    iq = np.concatenate(segs).astype(np.complex64)
    iq += 0.05 * (rng.standard_normal(len(iq))
                  + 1j * rng.standard_normal(len(iq)))
    return iq, np.array(hop_seq)


def main():
    p = argparse.ArgumentParser(description="FHSS hop visualizer.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--nfft", type=int, default=256)
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        iq, _ = synth_fhss(fs)
        print(f"[*] synthetic frequency-hopping signal @ {fs/1e6:g} Msps")

    # the spectrogram shows the hops; the detector tracks them
    freqs, times, sxx = spectrogram(iq, fs, nfft=args.nfft, overlap=0.5)
    hop_times, hop_freqs = fhss_detect_hops(iq, fs, nfft=args.nfft, overlap=0.5)
    print(f"[*] {len(hop_times)} time slices; tracking the hopper")

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(sxx, aspect="auto", origin="lower", cmap="turbo",
                   extent=[freqs[0] / 1e6, freqs[-1] / 1e6,
                           times[0] * 1e3, times[-1] * 1e3])
    # overlay the detected hop track
    ax.plot(hop_freqs / 1e6, hop_times * 1e3, "w.", ms=3, alpha=0.6,
            label="detected hop")
    fig.colorbar(im, ax=ax, label="power (dB)")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("time (ms)")
    ax.set_title("frequency hopping: the signal jumps channels over time\n"
                 "(white dots = detected hop track)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
