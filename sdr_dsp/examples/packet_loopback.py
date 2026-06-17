#! /usr/bin/python3
"""packet_loopback.py -- send a verifiable PACKET through the full chain (Phase B).

Phase A could transmit a waveform; now we can transmit a framed PACKET that the
receiver finds and validates. The frame is:
    [preamble][sync word][length][payload][CRC]
and the chain is:
    message -> build_frame -> modulate -> (channel) -> demod -> find_frames

The CRC is the point: the receiver knows whether the payload arrived intact. We
show a clean packet recovered, and a corrupted one caught by the CRC -- which is
exactly what an ACK protocol (Phase D) will key off.

Library deps only (numpy). No hardware -- full software loopback.

Usage:
    python examples/packet_loopback.py
    python examples/packet_loopback.py --msg "CQ CQ DE SDR" --scheme fsk
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (build_frame, find_frames,
                          ook_modulate, ook_envelope, ook_slice,
                          fsk_modulate, fsk_demod,
                          bpsk_modulate, bpsk_demod)


def modulate(bits, scheme, sps, fs):
    if scheme == "ook":
        return ook_modulate(bits, sps)
    if scheme == "fsk":
        return fsk_modulate(bits, sps, 50e3, fs)
    return bpsk_modulate(bits, 1)


def demodulate(iq, scheme, sps, fs, nbits):
    if scheme == "ook":
        return ook_slice(ook_envelope(iq))[::sps][:nbits]
    if scheme == "fsk":
        return fsk_demod(iq, fs)[sps // 2::sps][:nbits]
    rec, _ = bpsk_demod(iq)
    return rec[:nbits]


def main():
    p = argparse.ArgumentParser(description="Packet loopback through the chain.")
    p.add_argument("--msg", default="ACK 42", help="message to send")
    p.add_argument("--scheme", choices=["ook", "fsk", "bpsk"], default="ook")
    p.add_argument("--sps", type=int, default=20)
    p.add_argument("--rate", type=float, default=1e6)
    args = p.parse_args()

    payload = args.msg.encode()
    print(f"[*] sending {payload!r} via {args.scheme.upper()}\n")

    # --- TX: build the frame, modulate -----------------------------------
    frame_bits = build_frame(payload)
    iq = modulate(frame_bits, args.scheme, args.sps, args.rate)
    print(f"[TX] framed {len(payload)} payload bytes -> {len(frame_bits)} bits "
          f"-> {len(iq)} IQ samples")

    # --- RX: demod, find the frame ---------------------------------------
    rec_bits = np.asarray(demodulate(iq, args.scheme, args.sps, args.rate,
                                     len(frame_bits)), dtype=np.uint8)
    found = find_frames(rec_bits)
    if found:
        f = found[0]
        print(f"[RX] frame found at bit {f['bit_offset']}: "
              f"{f['payload']!r}  CRC {'OK' if f['crc_ok'] else 'FAIL'}")
    else:
        print("[RX] no frame found")

    # --- corruption demo: flip a payload bit, watch the CRC catch it -----
    print("\n[*] now corrupt one payload bit in the recovered stream:")
    if found:
        corrupt = rec_bits.copy()
        # payload starts after the sync match + length byte
        payload_start = found[0]["bit_offset"] + 16 + 8
        corrupt[payload_start + 3] ^= 1
        recheck = find_frames(corrupt)
        if recheck:
            print(f"[RX] payload {recheck[0]['payload']!r}  "
                  f"CRC {'OK' if recheck[0]['crc_ok'] else 'FAIL'} "
                  f"<- the CRC caught the bit error")
    print("\n[*] the CRC is what an ACK keys off: ACK a good frame, "
          "retransmit a bad one (Phase D)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
