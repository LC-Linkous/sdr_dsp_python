#! /usr/bin/python3
"""decimation_stages.py -- why multi-stage decimation beats one big step.

To drop a high sample rate way down, you can decimate in one giant step or in
several smaller stages. Multi-stage is standard practice: each stage needs a
gentler (cheaper) anti-alias filter, so the total work is less for the same
quality. This demo decimates the same signal both ways and compares cost and
the resulting spectrum.

Pure synthetic, no hardware. matplotlib optional (text summary always prints).

Usage:
    python examples/decimation_stages.py --total 64
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import decimate, psd


def factorize(n):
    """Break n into small factors for staged decimation (e.g. 64 -> 4,4,4)."""
    stages = []
    for f in (5, 4, 3, 2):
        while n % f == 0:
            stages.append(f)
            n //= f
    if n > 1:
        stages.append(n)
    return stages


def main():
    p = argparse.ArgumentParser(description="Single vs multi-stage decimation.")
    p.add_argument("--total", type=int, default=64, help="total decimation")
    p.add_argument("--n", type=int, default=400_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    fs = 4_000_000
    t = np.arange(args.n) / fs
    # a low tone we want to keep + a high tone that must be anti-aliased away
    keep_hz = 20_000
    iq = (np.exp(2j * np.pi * keep_hz * t)
          + 0.5 * np.exp(2j * np.pi * 1.5e6 * t)).astype(np.complex64)

    # single stage
    t0 = time.perf_counter()
    one = decimate(iq, args.total)
    t_one = time.perf_counter() - t0

    # multi stage
    stages = factorize(args.total)
    t0 = time.perf_counter()
    multi = iq
    for s in stages:
        multi = decimate(multi, s)
    t_multi = time.perf_counter() - t0

    new_fs = fs / args.total
    print(f"[*] decimate {args.total}x  ({fs/1e6:g} -> {new_fs/1e3:g} ksps)")
    print(f"  single stage : {t_one*1e3:7.2f} ms  -> {len(one):,} samples")
    print(f"  stages {stages} : {t_multi*1e3:7.2f} ms  -> {len(multi):,} samples")
    print(f"  -> staged is {t_one/t_multi:.2f}x the single-step time "
          f"({'faster' if t_multi < t_one else 'slower'})")
    # both should preserve the 20 kHz tone and reject the 1.5 MHz one
    for name, sig_ in (("single", one), ("staged", multi)):
        f, pdb = psd(sig_, new_fs, nfft=512)
        peak = f[np.argmax(pdb)]
        print(f"  {name}: spectral peak @ {peak/1e3:+.1f} kHz "
              f"(expect ~{keep_hz/1e3:g})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, sig_ in (("single-stage", one), ("multi-stage", multi)):
            f, pdb = psd(sig_, new_fs, nfft=512)
            ax.plot(f / 1e3, pdb, lw=0.8, label=name, alpha=0.8)
        ax.set_xlabel("frequency (kHz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{args.total}x decimation: single vs staged")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
