#! /usr/bin/python3
"""iq_basics.py -- "what *is* IQ data?" A teaching visualization.

Loads (or synthesizes) a simple signal and shows the three ways to look at
complex IQ samples:
  1. the complex plane (I vs Q) -- a tone traces a circle
  2. magnitude over time -- the envelope
  3. phase over time -- the rotating angle whose rate IS frequency

The point: a complex sample is a 2-D vector (I, Q). A pure tone spins that
vector at a constant rate; the spin rate is the frequency, the radius is the
amplitude. Every other DSP operation builds on this picture.

Needs: matplotlib (examples extra).  No hardware -- synthesizes if no file.

Usage:
    python examples/iq_basics.py                       # synthetic tone
    python examples/iq_basics.py capture.iq --samples 2000
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")


def main():
    p = argparse.ArgumentParser(description="Visualize IQ basics.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--freq", type=float, default=2000, help="synth tone Hz")
    p.add_argument("--rate", type=float, default=48000, help="synth rate Hz")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file, count=args.samples)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
        title = args.iq_file
    else:
        n = args.samples
        t = np.arange(n) / args.rate
        iq = np.exp(2j * np.pi * args.freq * t).astype(np.complex64)
        fs = args.rate
        title = f"synthetic {args.freq:g} Hz tone @ {fs/1e3:g} kHz"

    n = len(iq)
    t_ms = np.arange(n) / fs * 1e3

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"IQ basics  --  {title}", fontsize=12)

    # 1. complex plane
    ax1.plot(iq.real, iq.imag, lw=0.8)
    ax1.set_aspect("equal")
    ax1.set_xlabel("I (real)")
    ax1.set_ylabel("Q (imag)")
    ax1.set_title("complex plane: a tone traces a circle")
    ax1.grid(alpha=0.3)

    # 2. magnitude
    ax2.plot(t_ms, np.abs(iq), lw=0.8)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("|amplitude|")
    ax2.set_title("magnitude (envelope)")
    ax2.grid(alpha=0.3)

    # 3. phase
    ax3.plot(t_ms, np.unwrap(np.angle(iq)), lw=0.8)
    ax3.set_xlabel("time (ms)")
    ax3.set_ylabel("phase (rad, unwrapped)")
    ax3.set_title("phase: slope = frequency")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
