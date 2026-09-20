#! /usr/bin/python3
"""Diagnose native-capture-rate quality: same station, same gain, several
hardware sample rates, software-decimated to a common 2 Msps for an
apples-to-apples pilot measurement.

Run with the radio attached, pointed at a station the sweep likes:

    uv run python tools/diagnose_rate.py 95.3e6
    uv run python tools/diagnose_rate.py 95.3e6 --lna 32 --vga 20

Interpretation:
  - pilot LOW at native 2 Msps but HIGH at 8+ Msps  ->  the HackRF's
    low-rate capture path is the problem; tools must capture at >= 8 Msps
    and decimate in software (the fix we're testing for).
  - pilot LOW at EVERY rate  ->  not a rate problem: USB drops, overload,
    or the station is genuinely marginal. Try a different USB port/hub
    (the preflight warned about 2 other devices on this bus) and rerun.
  - pilot HIGH everywhere  ->  the earlier failure was transient; rerun
    the preflight.

SAFETY: receive-only; short captures at fixed gain.
"""

import argparse

import numpy as np

from sdr_dsp.core import (capture_health, design_lowpass, fir_apply,
                          fm_pilot_excess_db)

RATES = (2e6, 4e6, 8e6, 10e6, 20e6)
SECONDS = 0.4


def to_2msps(iq, rate):
    """Software channel-decimate any integer-multiple rate down to 2 Msps."""
    decim = int(rate // 2e6)
    if decim <= 1:
        return iq, rate
    taps = design_lowpass(0.45 * 2e6, rate, num_taps=8 * decim + 1)
    return fir_apply(iq, taps)[len(taps)::decim], 2e6


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("freq", type=float, help="station Hz, e.g. 95.3e6")
    p.add_argument("--lna", type=int, default=32,
                   help="fixed LNA dB (default 32, the fm_reference value)")
    p.add_argument("--vga", type=int, default=20,
                   help="fixed VGA dB (default 20, the fm_reference value)")
    p.add_argument("--amp", action="store_true", help="enable the RF amp")
    p.add_argument("--tools-dir", default=None)
    args = p.parse_args()

    from hackrfpy import HackRF
    h = HackRF(tools_dir=args.tools_dir, verbose=False)

    print(f"station {args.freq / 1e6:g} MHz, "
          f"lna={args.lna} vga={args.vga} amp={'on' if args.amp else 'off'}, "
          f"{SECONDS:g}s per capture\n")
    print("native rate | counts | channel   | pilot @native | pilot @2M-decim")
    print("------------+--------+-----------+---------------+----------------")
    for rate in RATES:
        try:
            iq = h.capture_array(args.freq, rate, int(rate * SECONDS),
                                 lna=args.lna, vga=args.vga, amp=args.amp)
        except Exception as e:                                # noqa: BLE001
            print(f"{rate / 1e6:9.0f} M | capture failed: {e}")
            continue
        health = capture_health(iq, rate, channel_bw=100e3)
        p_native = fm_pilot_excess_db(iq, rate)
        dec, dec_rate = to_2msps(iq, rate)
        p_dec = fm_pilot_excess_db(dec, dec_rate)
        fmt = lambda v: "   n/a " if v is None else f"{v:+6.1f}"
        print(f"{rate / 1e6:9.0f} M | {health['adc_counts']:5.1f}  | "
              f"{fmt(health['channel_excess_db'])} dB | {fmt(p_native)} dB     "
              f"| {fmt(p_dec)} dB")
    print("\nIf 8 M+ rows show a clear pilot and the 2 M row does not, the "
          "fix is to\ncapture at >= 8 Msps and decimate in software -- "
          "say so and it will be patched\ninto the preflight and "
          "collection tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
