#! /usr/bin/python3
"""resampler_benchmark.py -- our resampler vs scipy: speed and accuracy.

Demonstrates the library's engineering philosophy as a runnable artifact:
sdr_dsp implements its own resampler, and here we PROVE it against
scipy.signal.resample_poly -- both that it agrees numerically (the oracle) and
how the two compare on speed. "Implement it, then verify it against known-good."

Pure synthetic, no hardware.

Usage:
    python examples/resampler_benchmark.py
    python examples/resampler_benchmark.py --up 3 --down 2 --n 200000
"""
import argparse
import sys
import time

import numpy as np
from scipy import signal as sig

sys.path.insert(0, "src")
from sdr_dsp.core import resample_poly as ours


def bench(fn, *a, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*a)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    p = argparse.ArgumentParser(description="Benchmark our resampler vs scipy.")
    p.add_argument("--up", type=int, default=3)
    p.add_argument("--down", type=int, default=2)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    fs = 1_000_000
    t = np.arange(args.n) / fs
    # a multi-tone signal so the comparison exercises real spectral content
    x = (np.exp(2j * np.pi * 50e3 * t) + 0.5 * np.exp(2j * np.pi * 150e3 * t)
         + 0.2 * np.exp(-2j * np.pi * 200e3 * t)).astype(np.complex64)

    print(f"[*] resampling {args.n:,} samples by {args.up}/{args.down}\n")

    out_ours, t_ours = bench(lambda s: ours(s, args.up, args.down), x)
    out_sp, t_sp = bench(lambda s: sig.resample_poly(s, args.up, args.down), x)

    # accuracy: normalized correlation in the steady state (edges differ by
    # design -- different filter lengths / transient handling)
    m = min(len(out_ours), len(out_sp))
    a, b = out_ours[200:m - 200], out_sp[200:m - 200]
    corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"  ours   : {t_ours*1e3:7.2f} ms   -> {len(out_ours):,} samples")
    print(f"  scipy  : {t_sp*1e3:7.2f} ms   -> {len(out_sp):,} samples")
    print(f"  speed  : ours is {t_sp/t_ours:.2f}x scipy "
          f"({'faster' if t_ours < t_sp else 'slower'})")
    print(f"  accuracy: steady-state correlation {corr:.6f} "
          f"({'excellent' if corr > 0.999 else 'check'})")
    print("\n  (scipy is a highly optimized C implementation; the point here is")
    print("   that our pure-Python resampler is CORRECT, verified against it.)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(install matplotlib to plot)")
            return 0
        new_fs = fs * args.up / args.down
        fo = np.fft.fftshift(np.fft.fftfreq(len(out_ours), 1 / new_fs)) / 1e3
        fs_ = np.fft.fftshift(np.fft.fftfreq(len(out_sp), 1 / new_fs)) / 1e3
        So = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(out_ours))) + 1e-9)
        Ss = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(out_sp))) + 1e-9)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(fo, So, lw=0.8, label="ours")
        ax.plot(fs_, Ss, lw=0.8, alpha=0.6, label="scipy")
        ax.set_xlabel("frequency (kHz)")
        ax.set_ylabel("magnitude (dB)")
        ax.set_title(f"resampled spectrum: ours vs scipy ({args.up}/{args.down})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
