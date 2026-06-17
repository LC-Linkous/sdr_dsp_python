#! /usr/bin/python3
"""power_calibration.py -- turn dBFS into absolute dBm (opt-in, advanced).

The library reports power in dBFS (relative to full scale) by default, which is
always honest and needs nothing external. To get ABSOLUTE power in dBm you need
a one-time calibration against a known reference -- and it's only valid for the
gain/frequency you measured it at. This shows the whole workflow:

    measure a known reference -> derive a Calibration -> save it ->
    reload it later -> apply it to real captures.

Most users won't have a calibrated source and should stay with dBfs. This is for
advanced users with a signal generator or a known reference.

Library deps only (numpy). No hardware -- synthesizes a "known" reference so the
workflow runs; with real hardware you'd capture an actual calibrated source.

Usage:
    python examples/power_calibration.py
    python examples/power_calibration.py --known-dbm -30 --freq 433.92e6
"""
import argparse
import sys
import tempfile
import os

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import power_dbfs, compute_cal_offset, Calibration


def main():
    p = argparse.ArgumentParser(description="Absolute power calibration demo.")
    p.add_argument("--known-dbm", type=float, default=-30.0,
                   help="true power of your reference source (dBm)")
    p.add_argument("--freq", type=float, default=433.92e6,
                   help="frequency the reference was measured at (Hz)")
    p.add_argument("--save", default=None, help="path to save the calibration")
    args = p.parse_args()

    print("[*] Calibration workflow (dBFS -> dBm)\n")

    # --- 1. capture a KNOWN reference -------------------------------------
    # In reality: connect a signal generator at args.known_dbm, capture it.
    # Here we synthesize a steady tone to stand in for that reference.
    ref = (0.1 * np.exp(2j * np.pi * 0.05 * np.arange(20000))).astype(
        np.complex64)
    print(f"[1] reference captured: {power_dbfs(ref):.2f} dBFS")
    print(f"    (declared true power: {args.known_dbm:g} dBm)")

    # --- 2. derive the calibration ----------------------------------------
    cal = compute_cal_offset(
        ref, known_dbm=args.known_dbm, frequency_hz=args.freq,
        conditions={"sdr": "example", "note": "synthetic reference"},
        notes="demo calibration",
    )
    print(f"[2] derived offset: {cal.offset_db:+.2f} dB")
    print(f"    {cal!r}")

    # --- 3. save it -------------------------------------------------------
    path = args.save or os.path.join(tempfile.mkdtemp(), "demo.cal.json")
    cal.save(path)
    print(f"[3] saved calibration to {path}")

    # --- 4. reload it (a later session) -----------------------------------
    cal2 = Calibration.load(path)
    print(f"[4] reloaded: {cal2!r}")

    # --- 5. apply it to a 'real' capture ----------------------------------
    # a weaker signal at the same frequency
    capture = (0.03 * np.exp(2j * np.pi * 0.05 * np.arange(20000))).astype(
        np.complex64)
    dbfs = power_dbfs(capture)
    dbm = cal2.power_dbm(capture, at_frequency_hz=args.freq)
    print(f"[5] new capture: {dbfs:.2f} dBFS  ->  {dbm:.2f} dBm")

    # --- 6. show the drift warning ----------------------------------------
    print("[6] applying it 100 MHz away (should warn):")
    cal2.power_dbm(capture, at_frequency_hz=args.freq + 100e6)
    print("\n[*] dBFS is always honest; dBm is only as good as your reference "
          "and only valid near the calibrated frequency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
