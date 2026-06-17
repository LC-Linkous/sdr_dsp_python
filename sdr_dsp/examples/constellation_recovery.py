#! /usr/bin/python3
"""constellation_recovery.py -- watch carrier + timing recovery clean up a
scrambled constellation. The visual payoff of the recovery layer.

A real captured PSK/QAM signal arrives rotated (carrier offset) and sampled at
the wrong instants (timing offset), so its constellation is a smeared mess.
This demo shows it in three panels:
  1. the impaired signal -- a rotating blur
  2. after carrier_recovery -- de-rotated but still timing-smeared
  3. after symbol_sync too -- clean clusters you can read bits from

It also plots the loop error settling toward zero (convergence evidence) and can
write the diagnostics to CSV.

Needs: matplotlib (examples extra). Pure synthetic, no hardware.

Usage:
    python examples/constellation_recovery.py
    python examples/constellation_recovery.py --mod qpsk --csv loops.csv
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import carrier_recovery, symbol_sync


def make_signal(mod, sps, nsym, seed=0):
    rng = np.random.default_rng(seed)
    if mod == "bpsk":
        pts = (2 * rng.integers(0, 2, nsym) - 1).astype(np.complex64)
        order = 2
    else:  # qpsk
        gray = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
        pts = np.array([gray[s] for s in rng.integers(0, 4, nsym)],
                       dtype=np.complex64) / np.sqrt(2)
        order = 4
    tx = np.repeat(pts, sps).astype(np.complex64)
    # impair: carrier frequency offset + phase + timing shift + noise
    t = np.arange(len(tx))
    imp = tx * np.exp(2j * np.pi * 0.001 * t + 1j * 0.6).astype(np.complex64)
    imp = np.concatenate([np.zeros(2, dtype=np.complex64), imp])
    imp += 0.06 * (rng.standard_normal(len(imp))
                   + 1j * rng.standard_normal(len(imp)))
    return imp, order


def main():
    p = argparse.ArgumentParser(description="Constellation recovery demo.")
    p.add_argument("--mod", choices=["bpsk", "qpsk"], default="qpsk")
    p.add_argument("--sps", type=int, default=4)
    p.add_argument("--nsym", type=int, default=600)
    p.add_argument("--csv", default=None, help="write loop diagnostics to CSV")
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    imp, order = make_signal(args.mod, args.sps, args.nsym)

    # stage 1 -> 2: carrier recovery (with diagnostics for the convergence plot)
    corrected, cdiag = carrier_recovery(imp, method="costas", order=order,
                                        loop_bw=0.005, diagnostics=True,
                                        csv_path=args.csv)
    # stage 2 -> 3: symbol timing
    syms, sdiag = symbol_sync(corrected, args.sps, method="gardner",
                              diagnostics=True)

    print(f"[*] {args.mod.upper()}: carrier locked={cdiag.locked}, "
          f"timing locked={sdiag.locked}")
    if args.csv:
        print(f"[*] carrier loop diagnostics -> {args.csv}")

    fig = plt.figure(figsize=(14, 8))
    # three constellations across the top
    def scatter(ax, data, title):
        d = data[len(data)//5:]   # skip convergence transient
        ax.scatter(d.real, d.imag, s=4, alpha=0.4)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        lim = 1.5 * np.percentile(np.abs(d), 95) or 1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    scatter(fig.add_subplot(2, 3, 1), imp[::args.sps],
            "1. impaired (rotating blur)")
    scatter(fig.add_subplot(2, 3, 2), corrected[::args.sps],
            "2. after carrier recovery")
    scatter(fig.add_subplot(2, 3, 3), syms, "3. + symbol timing (clean)")

    # convergence evidence across the bottom
    ax4 = fig.add_subplot(2, 1, 2)
    ax4.plot(cdiag.error, lw=0.5, label="carrier phase error")
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_title("carrier loop error settling toward zero (convergence)")
    ax4.set_xlabel("sample")
    ax4.set_ylabel("phase error")
    ax4.legend()
    ax4.grid(alpha=0.3)

    fig.suptitle(f"{args.mod.upper()} recovery: carrier + timing clean up the "
                 f"constellation", fontsize=12)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
