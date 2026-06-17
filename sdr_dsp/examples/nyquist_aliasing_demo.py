#! /usr/bin/python3
"""nyquist_aliasing_demo.py -- see a signal alias when you undersample it.

The sampling theorem made visual. A tone above half the sample rate (Nyquist)
can't be represented -- it folds back to a LOWER, wrong frequency. This demo
samples the same true tone at decreasing rates and shows where each one "thinks"
the tone is, so students watch aliasing happen.

Needs: matplotlib (examples extra). Pure synthetic, no hardware.

Usage:
    python examples/nyquist_aliasing_demo.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")


def apparent_freq(true_hz, fs):
    """Where a tone at true_hz appears after sampling at fs (folding)."""
    f = true_hz % fs
    return f if f <= fs / 2 else f - fs


def main():
    p = argparse.ArgumentParser(description="Nyquist / aliasing demo.")
    p.add_argument("--tone", type=float, default=1500, help="true tone Hz")
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    true_hz = args.tone
    # sample the same tone at a high rate (reference) and several lower rates
    rates = [8000, 4000, 2500, 2000, 1600]
    fig, axes = plt.subplots(len(rates), 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"true tone = {true_hz:g} Hz, sampled at decreasing rates\n"
                 f"(Nyquist = rate/2; above it, the tone aliases)", fontsize=11)

    dur = 0.01
    t_fine = np.linspace(0, dur, 5000)
    true_wave = np.cos(2 * np.pi * true_hz * t_fine)

    for ax, fs in zip(axes, rates):
        n = int(fs * dur)
        ts = np.arange(n) / fs
        samples = np.cos(2 * np.pi * true_hz * ts)
        app = apparent_freq(true_hz, fs)
        ax.plot(t_fine * 1e3, true_wave, color="gray", lw=0.8, alpha=0.5,
                label="true tone")
        ax.plot(ts * 1e3, samples, "ro", ms=4)
        # the apparent (aliased) reconstruction
        app_wave = np.cos(2 * np.pi * abs(app) * t_fine)
        aliased = abs(app) < true_hz - 1
        ax.plot(t_fine * 1e3, app_wave, color="red", lw=1.0, ls="--",
                alpha=0.7 if aliased else 0.0)
        nyq = fs / 2
        tag = (f"fs={fs} Hz (Nyq {nyq:.0f}) -> appears at {abs(app):.0f} Hz"
               + ("  ** ALIASED **" if aliased else "  (ok)"))
        ax.set_ylabel(tag, fontsize=8, rotation=0, ha="left", labelpad=2)
        ax.yaxis.set_label_position("right")
        ax.set_yticks([])
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
