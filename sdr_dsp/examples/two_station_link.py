#! /usr/bin/python3
"""two_station_link.py -- two SDRs exchanging acked messages (TX Phase D).

The capstone of the software TX arc: station A sends messages to station B over
a link, B acknowledges each, and A retransmits anything that doesn't get acked.
This runs the real chain -- frame -> FSK modulate -> simulated channel -> demod
-> find frame -> ARQ -- entirely in software. A forced frame drop shows a
retransmit; then the whole exchange is saved and REPLAYED with zero transmission,
which is how you demo it repeatedly without keying a radio.

Library deps only (numpy). No hardware.

Usage:
    python examples/two_station_link.py
    python examples/two_station_link.py --window 4 --snr 20
    python examples/two_station_link.py --replay exchange.json
"""
import argparse
import functools
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (build_frame, find_frames, apply_channel,
                          fsk_modulate, fsk_demod)
from sdr_dsp.link import (ARQ, run_sim, run_link, replay, EventLog,
                               make_channel_transport)

FS = 1e6
SPS = 20


def build_transport(snr_db, drop_first=False):
    state = {"used": False}

    def maybe_drop():
        if drop_first and not state["used"]:
            state["used"] = True
            return True
        return False

    def modulate(payload):
        return fsk_modulate(build_frame(payload), SPS, 50e3, FS)

    def demodulate(iq):
        bits = fsk_demod(iq, FS)[SPS // 2::SPS]
        return find_frames(np.asarray(bits, dtype=np.uint8))

    channel = functools.partial(apply_channel, sample_rate=FS, snr_db=snr_db,
                                seed=1)
    return make_channel_transport(modulate, demodulate, channel,
                                  drop_predicate=maybe_drop)


def main():
    p = argparse.ArgumentParser(description="Two-station acked message exchange.")
    p.add_argument("--window", type=int, default=1,
                   help="1 = stop-and-wait, N = sliding window")
    p.add_argument("--snr", type=float, default=25)
    p.add_argument("--save", default=None, help="save the event log to JSON")
    p.add_argument("--replay", default=None,
                   help="replay a saved log instead of transmitting")
    args = p.parse_args()

    # --- replay mode: reproduce a saved exchange, zero TX -----------------
    if args.replay:
        log = EventLog.load(args.replay)
        B = ARQ(window_size=args.window)
        produced = replay(log, B, "B")
        delivered = [i[1] for i in produced if i[0] == "deliver"]
        print(f"[*] replayed {args.replay} ({len(log)} records), zero TX")
        for d in delivered:
            print(f"    B received: {d!r}")
        return 0

    # --- live(ish) sim mode -----------------------------------------------
    messages = [b"CQ CQ", b"DE SDR", b"MSG 3", b"73"]
    mode = "stop-and-wait" if args.window == 1 else f"sliding window N={args.window}"
    print(f"[*] {mode}, FSK over a {args.snr:g} dB channel, one forced drop\n")

    A = ARQ(window_size=args.window, timeout_ticks=3, max_retries=10)
    B = ARQ(window_size=args.window, timeout_ticks=3, max_retries=10)
    for m in messages:
        A.send(m)
    transport = build_transport(args.snr, drop_first=True)
    _, received, log = run_sim(A, B, transport=transport, max_ticks=500)

    print(f"[A] sent:     {messages}")
    print(f"[B] received: {received}")
    print(f"[*] all delivered: {received == messages}")

    # count retransmits from the log (tx records beyond the first per seq)
    tx_recs = [r for r in log.records if r["dir"] == "tx" and r.get("type") == "DATA"]
    retx = len(tx_recs) - len(messages)
    print(f"[*] {len(tx_recs)} DATA transmissions for {len(messages)} messages "
          f"({retx} retransmit{'s' if retx != 1 else ''})")

    if args.save:
        log.save(args.save)
        print(f"\n[*] saved exchange to {args.save}")
        print(f"    replay it with: python examples/two_station_link.py "
              f"--replay {args.save}")
    print("\n[*] this is the software-provable two-SDR exchange; the live "
          "driver (real radio) is Phase E")
    return 0


if __name__ == "__main__":
    sys.exit(main())
