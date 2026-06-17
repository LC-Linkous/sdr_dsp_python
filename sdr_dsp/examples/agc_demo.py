#! /usr/bin/python3
"""agc_demo.py -- stabilize a fading signal with AGC, and SEE what it did.

A signal that fades in and out (a moving transmitter, a satellite pass) swings
in level by tens of dB. AGC tracks that and adjusts gain to hold it steady. The
honest part: this AGC hands back the exact gain it applied, so the bottom panel
shows precisely what the loop did -- nothing hidden, and you could divide it back
out to recover the original.

Three panels: the fading input, the AGC-stabilized output, and the gain trace
the AGC applied (the proof of what happened).

Needs: matplotlib (examples extra). Pure synthetic, no hardware.

Usage:
    python examples/agc_demo.py
    python examples/agc_demo.py --mode peak --attack 0.02 --decay 0.0005
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import agc


def main():
    p = argparse.ArgumentParser(description="AGC demo on a fading signal.")
    p.add_argument("--mode", choices=["rms", "peak"], default="rms")
    p.add_argument("--target", type=float, default=0.5)
    p.add_argument("--attack", type=float, default=0.01)
    p.add_argument("--decay", type=float, default=0.001)
    p.add_argument("--max-gain", type=float, default=None)
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    # a fading signal: a tone whose amplitude swings ~26 dB, plus a deep dip
    n = 60000
    t = np.arange(n)
    env = 0.5 + 0.45 * np.cos(2 * np.pi * 2.5 * t / n)
    env *= (1 - 0.7 * np.exp(-((t - 0.65 * n) ** 2) / (2 * (0.03 * n) ** 2)))
    sig = (env * np.exp(2j * np.pi * 0.05 * t)).astype(np.complex64)
    sig += 0.01 * (np.random.randn(n) + 1j * np.random.randn(n))
    sig = sig.astype(np.complex64)

    adjusted, gain = agc(sig, mode=args.mode, target=args.target,
                         attack=args.attack, decay=args.decay,
                         max_gain=args.max_gain)

    # honesty check, printed: the original is recoverable from the trace
    recovered = adjusted / gain
    print(f"[*] AGC ({args.mode}, target={args.target}): "
          f"input level swing {np.abs(sig).max()/max(np.abs(sig).mean(),1e-9):.1f}x")
    print(f"[*] output level std {np.abs(adjusted)[n//4:].std():.3f} "
          f"(input {np.abs(sig)[n//4:].std():.3f})")
    print(f"[*] original recoverable from gain trace: "
          f"{np.allclose(recovered, sig, atol=1e-5)}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(np.abs(sig), lw=0.5)
    ax[0].set_title("1. fading input (|signal|)")
    ax[0].set_ylabel("magnitude")
    ax[1].plot(np.abs(adjusted), lw=0.5, color="tab:green")
    ax[1].axhline(args.target, color="k", ls="--", lw=0.5, label="target")
    ax[1].set_title("2. AGC output -- level held near target")
    ax[1].set_ylabel("magnitude")
    ax[1].legend(loc="upper right")
    ax[2].plot(gain, lw=0.5, color="tab:orange")
    ax[2].set_title("3. gain the AGC applied (the trace -- nothing hidden)")
    ax[2].set_ylabel("gain")
    ax[2].set_xlabel("sample")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("AGC: stabilize a fade, and show exactly what was done",
                 fontsize=12)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
