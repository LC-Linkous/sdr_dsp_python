#! /usr/bin/python3
"""channel_sweep.py -- send a packet through worsening channels (TX Phase C).

Now that we can frame and modulate a packet, the simulated channel lets us ask
the real question: how bad can the link get before the message doesn't make it?
This sends a framed packet through a channel at a range of SNRs (and optionally a
carrier offset) and reports, for each, whether the frame was recovered, flagged
corrupt by the CRC, or lost entirely.

That three-way outcome -- recovered / flagged / lost -- is exactly what an ACK
protocol needs: a good CRC gets an ACK, a bad one triggers a retransmit (Phase D).

Library deps only (numpy). No hardware -- full software loopback.

Usage:
    python examples/channel_sweep.py
    python examples/channel_sweep.py --scheme fsk --cfo 5000
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (build_frame, find_frames, apply_channel,
                          ook_modulate, ook_envelope, ook_slice,
                          fsk_modulate, fsk_demod,
                          bpsk_modulate, bpsk_demod, carrier_recovery)


def modulate(bits, scheme, sps, fs):
    if scheme == "ook":
        return ook_modulate(bits, sps)
    if scheme == "fsk":
        return fsk_modulate(bits, sps, 50e3, fs)
    return bpsk_modulate(bits, sps)


def demodulate(iq, scheme, sps, fs, nbits, recover=False):
    if scheme == "ook":
        return ook_slice(ook_envelope(iq))[::sps][:nbits]
    if scheme == "fsk":
        return fsk_demod(iq, fs)[sps // 2::sps][:nbits]
    # bpsk: optionally recover the carrier first (helps against CFO)
    if recover:
        iq = carrier_recovery(iq, method="costas", order=2)
    rec, _ = bpsk_demod(iq)
    return rec[::sps][:nbits] if sps > 1 else rec[:nbits]


def main():
    p = argparse.ArgumentParser(description="Sweep a packet over worsening SNR.")
    p.add_argument("--msg", default="CQ DE SDR")
    p.add_argument("--scheme", choices=["ook", "fsk", "bpsk"], default="fsk")
    p.add_argument("--sps", type=int, default=20)
    p.add_argument("--rate", type=float, default=1e6)
    p.add_argument("--cfo", type=float, default=0.0, help="carrier offset Hz")
    args = p.parse_args()

    payload = args.msg.encode()
    frame = build_frame(payload)
    iq = modulate(frame, args.scheme, args.sps, args.rate)
    print(f"[*] {payload!r} via {args.scheme.upper()}"
          + (f", CFO {args.cfo/1e3:g} kHz" if args.cfo else "")
          + "\n")
    print(f"    {'SNR (dB)':>9}  outcome")
    print("    " + "-" * 34)

    for snr in [30, 20, 15, 10, 6, 3, 0]:
        rx = apply_channel(iq, sample_rate=args.rate, snr_db=snr,
                           cfo_hz=args.cfo, seed=1)
        bits = np.asarray(demodulate(rx, args.scheme, args.sps, args.rate,
                                     len(frame), recover=bool(args.cfo)),
                          dtype=np.uint8)
        found = find_frames(bits)
        if found and found[0]["crc_ok"] and found[0]["payload"] == payload:
            outcome = "recovered (ACK)"
        elif found:
            outcome = f"CRC flagged corrupt -> retransmit"
        else:
            outcome = "lost (no frame)"
        print(f"    {snr:>9}  {outcome}")

    print("\n[*] recovered/flagged/lost is what the ACK protocol keys off "
          "(Phase D)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
