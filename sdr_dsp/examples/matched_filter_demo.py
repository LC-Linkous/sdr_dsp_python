#! /usr/bin/python3
"""matched_filter_demo.py -- find a known pattern buried in noise.

A matched filter correlates a known reference against a noisy signal; the output
peaks exactly where the pattern occurs. It's the optimal detector for a known
shape in white noise -- the idea behind radar, sync-word detection, and GPS.

This demo hides a short pattern at a random spot in heavy noise, then uses
correlation to find it. Watch the sharp peak appear right where the pattern is.

Needs: matplotlib (examples extra). Pure synthetic, no hardware.

Usage:
    python examples/matched_filter_demo.py --snr-db -10
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")


def matched_filter(signal, template):
    """Correlate template against signal. OUR code (np.correlate).

    Returns the correlation magnitude; its peak marks where the template best
    aligns with the signal. NOTE: np.correlate already conjugates its second
    argument for complex input, so we pass the template directly (conjugating
    it ourselves would double-conjugate and break the match).
    """
    signal = np.asarray(signal)
    template = np.asarray(template)
    corr = np.correlate(signal, template, mode="valid")
    return np.abs(corr)


def main():
    p = argparse.ArgumentParser(description="Matched filter detection demo.")
    p.add_argument("--snr-db", type=float, default=-10,
                   help="signal-to-noise of the buried pattern (dB)")
    p.add_argument("--plen", type=int, default=64, help="pattern length")
    p.add_argument("--total", type=int, default=4000)
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    rng = np.random.default_rng(0)
    # a known unit-energy complex pattern (like a sync word)
    template = (rng.standard_normal(args.plen)
                + 1j * rng.standard_normal(args.plen)).astype(np.complex64)
    template /= np.linalg.norm(template)        # unit energy

    # Bury it at a random position in unit-variance noise. SNR here is the
    # ratio of total pattern energy to the noise power over the pattern's
    # span. Per-sample noise power = 2 (unit variance each in I and Q), so to
    # hit a target SNR we scale the (unit-energy) template by sqrt(SNR * plen *
    # noise_power_per_sample).
    pos = rng.integers(args.plen, args.total - args.plen)
    snr_lin = 10 ** (args.snr_db / 10)
    noise_pp = 2.0
    scale = np.sqrt(snr_lin * args.plen * noise_pp)
    noise = (rng.standard_normal(args.total)
             + 1j * rng.standard_normal(args.total)).astype(np.complex64)
    signal = noise.copy()
    signal[pos:pos + args.plen] += scale * template

    mf = matched_filter(signal, template)
    detected = int(np.argmax(mf))
    print(f"[*] pattern hidden at sample {pos}, SNR {args.snr_db} dB")
    print(f"[*] matched filter peak at sample {detected}  "
          f"({'HIT' if abs(detected - pos) <= 2 else 'miss'})")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    a1.plot(np.abs(signal), lw=0.5, color="gray")
    a1.axvspan(pos, pos + args.plen, color="red", alpha=0.2,
               label="hidden pattern")
    a1.set_title(f"signal (pattern is invisible by eye at {args.snr_db} dB SNR)")
    a1.set_ylabel("|amplitude|")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(mf, lw=0.7, color="#2ca02c")
    a2.axvline(pos, color="red", ls="--", lw=0.8, label="true position")
    a2.set_title("matched filter output: a sharp peak finds the pattern")
    a2.set_xlabel("sample")
    a2.set_ylabel("correlation")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
