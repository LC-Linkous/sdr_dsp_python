#! /usr/bin/python3
"""signal_survey.py -- measure what's in one or more captures.

Runs the measurement module over each capture and prints a little report: mean
power, occupied bandwidth, and an in-band SNR estimate. "What's actually here,
and how strong?" Useful for triaging a pile of recordings.

Library deps only (numpy). No hardware -- runs on saved captures.

Usage:
    python examples/signal_survey.py sample_data/fm_2Msps.iq
    python examples/signal_survey.py *.iq --signal-bw 200e3
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.io import load_iq
from sdr_dsp.core import power_dbfs, occupied_bandwidth, snr_db


def main():
    p = argparse.ArgumentParser(description="Survey power/bandwidth/SNR.")
    p.add_argument("iq_files", nargs="+")
    p.add_argument("--signal-bw", type=float, default=None,
                   help="assumed signal bandwidth for SNR (Hz, centered)")
    p.add_argument("--fraction", type=float, default=0.99,
                   help="occupied-bandwidth power fraction")
    args = p.parse_args()

    print(f"{'file':<30} {'power':>10} {'occ-BW':>12} {'SNR':>9}")
    print("-" * 64)
    for path in args.iq_files:
        try:
            iq, meta = load_iq(path, count=1_000_000)
        except Exception as e:
            print(f"{path:<30} ERROR: {e}")
            continue
        fs = float(meta.get("global", {}).get("core:sample_rate", 1.0))
        pwr = power_dbfs(iq)
        obw = occupied_bandwidth(iq, fs, fraction=args.fraction)
        snr_str = "  --"
        if args.signal_bw:
            try:
                s = snr_db(iq, fs, signal_band_hz=(-args.signal_bw / 2,
                                                   args.signal_bw / 2))
                snr_str = f"{s:6.1f} dB"
            except ValueError:
                snr_str = "  n/a"
        name = path.split("/")[-1].split("\\")[-1]
        print(f"{name:<30} {pwr:7.1f} dBFS {obw/1e3:8.1f} kHz {snr_str:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
