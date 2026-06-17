#! /usr/bin/python3
"""annotate_bursts.py -- the detect -> label -> save -> reload workflow.

Finding bursts is only useful if the work survives. This shows the full loop:
detect bursts in a capture, label them as SigMF annotations, save them into the
recording's sidecar, and reload them later as structured objects. The annotated
file is portable -- any SigMF-aware tool (inspectSigMF, GNU Radio) sees the same
labels.

Library deps only (numpy). No hardware -- synthesizes a capture with bursts, or
annotates a real one you pass in.

Usage:
    python examples/annotate_bursts.py
    python examples/annotate_bursts.py capture.iq --label "key fob {i}"
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import find_bursts, power_dbfs
from sdr_dsp.io import (save_iq, load_iq, read_annotations,
                        bursts_to_annotations)


def make_capture(fs):
    rng = np.random.default_rng(0)
    n = 200_000
    sig = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for start, length in [(20_000, 8_000), (90_000, 4_000), (150_000, 12_000)]:
        t = np.arange(length) / fs
        sig[start:start + length] += np.exp(2j * np.pi * 50e3 * t)
    return sig.astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description="Detect, label, and save bursts.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--label", default="burst {i}",
                   help="label template; {i} becomes the burst index")
    p.add_argument("--out", default="annotated.iq",
                   help="where to write the annotated copy")
    p.add_argument("--min-len", type=int, default=1000)
    p.add_argument("--min-gap", type=int, default=500)
    args = p.parse_args()

    if args.iq_file:
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
        center = float(meta.get("captures", [{}])[0].get("core:frequency", 0.0))
    else:
        fs = args.rate
        iq = make_capture(fs)
        center = 433.92e6
        print(f"[*] synthetic capture @ {fs/1e6:g} Msps (3 bursts)")

    # DETECT
    spans = find_bursts(iq, min_gap=args.min_gap, min_len=args.min_len)
    print(f"[*] detected {len(spans)} burst(s)")

    # LABEL (attach the measured power as an extra annotation field)
    anns = bursts_to_annotations(spans, label=args.label)
    for a in anns:
        seg = iq[a.sample_start:a.sample_start + a.sample_count]
        a.extra["sdr_dsp:power_dbfs"] = round(power_dbfs(seg), 1)

    # SAVE (annotations go into the sidecar)
    dp, mp = save_iq(args.out, iq, fs, center_freq=center, annotations=anns)
    print(f"[*] saved annotated recording: {mp}")

    # RELOAD (prove it round-trips)
    loaded = read_annotations(args.out)
    print(f"[*] reloaded {len(loaded)} annotation(s):\n")
    print(f"    {'label':<14} {'start (ms)':>10} {'dur (ms)':>9} {'power':>9}")
    print("    " + "-" * 45)
    for a in loaded:
        s, e = a.time_span(fs)
        pw = a.extra.get("sdr_dsp:power_dbfs", float("nan"))
        print(f"    {a.label:<14} {s*1e3:>10.2f} {(e-s)*1e3:>9.2f} "
              f"{pw:>6} dB")
    print("\n[*] the annotated file is portable to any SigMF-aware tool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
