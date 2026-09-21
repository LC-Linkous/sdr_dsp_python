#! /usr/bin/python3
"""transmit_examples.py -- modulate the library's own schemes as framed packets
and key them out through a TXSink (TX Phase C).

For each of OOK / FSK / BPSK / QPSK this builds a real frame
(build_frame -> bits), modulates it, hands the IQ to a TXSink, then reads it
back and decodes it (demod -> find_frames), proving demod(modulate(x)) == x
across the SINK SEAM -- the same protocol boundary a real radio sits behind.

By default the sink is a LoopbackSink (in-memory, no radio), so the WHOLE
generation-and-framing path is exercised and self-checked with no hardware.
The identical driving code runs against a real device by swapping in a HackRF
TXSink (examples/hackrf_sink.py); only the antenna is missing offline.

  *** READ THIS BEFORE TRANSMITTING ***
  Transmitting over the air is legally regulated. YOU are responsible for
  operating within the law: frequency, power, bandwidth, and licensing rules
  apply, and many bands are off-limits. The default center here (433.92 MHz)
  is an ISM band in some regions and NOT license-free in others -- it is a
  placeholder for the offline demo, not advice that it is legal for you.
  The safest bench setup is WIRED: TX -> coax -> attenuator -> RX, which
  exercises the full system without radiating. Do not connect an antenna and
  transmit unless you know the band is permitted and you are licensed where
  required. The HackRF TXSink is guarded (armed=False) and its device call is
  left unimplemented on purpose, so nothing here can radiate by accident.

Library deps only (numpy). No hardware for the default run.

Usage:
    python examples/transmit_examples.py                 # offline, all schemes
    python examples/transmit_examples.py --schemes fsk bpsk
    python examples/transmit_examples.py --payload "HELLO WORLD"
    python examples/transmit_examples.py --sink hackrf   # wire to the device seam
"""
import argparse
import sys

import numpy as np

from sdr_dsp.core import (bpsk_demod, bpsk_modulate, build_frame, find_frames,
                          fsk_demod, fsk_modulate, ook_envelope, ook_modulate,
                          ook_slice, qpsk_demod, qpsk_modulate, sample_symbols)
from sdr_dsp.sinks import LoopbackSink, TXSink

# Link parameters. SPS is deliberately generous (a clean, easily-timed symbol);
# PAD wraps each burst in guard silence so the first/last symbols don't sit at
# the buffer edge, where discriminator/filter edge effects corrupt them (the
# modulators document this -- use >= 4 for anything leaving a same-buffer loop).
FS = 1_000_000.0
SPS = 20
DEVIATION_HZ = 50_000.0
PAD = 4
DEFAULT_CENTER_HZ = 433.92e6


# --------------------------------------------------------------------------
# Each scheme is a (modulate bits -> IQ) paired with (IQ -> recovered bits).
# The RX chains are the hardware-robust ones from the demod docstrings and
# two_station_link.py: a per-sample decision stream decimated to symbols at the
# transition-estimated phase (delay-safe), then find_frames locates the packet.
# --------------------------------------------------------------------------
def _mod_ook(bits):
    return ook_modulate(bits, SPS, pad_symbols=PAD)


def _demod_ook(iq):
    persample = ook_slice(ook_envelope(iq))
    return sample_symbols(persample, SPS)


def _mod_fsk(bits):
    return fsk_modulate(bits, SPS, DEVIATION_HZ, FS, pad_symbols=PAD)


def _demod_fsk(iq):
    raw = fsk_demod(iq, FS, threshold_hz="auto", smooth_samples=SPS // 2)
    env = np.abs(iq)[:len(raw)]
    active = env > 0.25 * env.max() if env.max() > 0 else None
    return sample_symbols(raw, SPS, active=active)


def _mod_bpsk(bits):
    return bpsk_modulate(bits, SPS, pad_symbols=PAD)


def _demod_bpsk(iq):
    bits, _ = bpsk_demod(iq)
    env = np.abs(iq)[:len(bits)]
    active = env > 0.25 * env.max() if env.max() > 0 else None
    return sample_symbols(bits, SPS, active=active)


def _mod_qpsk(bits):
    return qpsk_modulate(bits, SPS, pad_symbols=PAD)


def _demod_qpsk(iq):
    # QPSK carries bits in TWO axes, so its demod needs complex SYMBOLS, not a
    # 0/1 stream. On this clean same-buffer loopback the timing is known (PAD is
    # a whole number of symbols), so we decimate at the symbol centres directly;
    # under an unknown real-channel delay you would recover timing with
    # symbol_sync (and the carrier with carrier_recovery) first -- that is the
    # Phase E two-SDR path, not this offline generation proof.
    symbols = iq[SPS // 2::SPS]
    bits, _ = qpsk_demod(symbols)
    return bits


SCHEMES = {
    "ook": (_mod_ook, _demod_ook),
    "fsk": (_mod_fsk, _demod_fsk),
    "bpsk": (_mod_bpsk, _demod_bpsk),
    "qpsk": (_mod_qpsk, _demod_qpsk),
}


def key_out(scheme, payload, sink):
    """Frame -> modulate -> sink.transmit -> read back -> demod -> find_frames.

    Returns (frames, n_samples). Works with ANY TXSink; the offline default
    passes a LoopbackSink so the buffer is readable, which is how the round
    trip is checked. A real device sink transmits instead of buffering, so it
    has nothing to read back -- validating THAT is the two-SDR bench test.
    """
    modulate, demod = SCHEMES[scheme]
    frame_bits = build_frame(payload)
    iq = modulate(frame_bits)
    sink.transmit(iq)
    buffered = getattr(sink, "buffer", None)
    if buffered is None:
        return [], len(iq)        # a real device sink: transmitted, not buffered
    frames = find_frames(demod(buffered))
    return frames, len(buffered)


def run_offline(payload, schemes):
    """Round-trip every scheme through a LoopbackSink and report. Returns True
    only if every scheme recovered the exact payload with a valid CRC."""
    print(f"[*] payload: {payload!r} ({len(payload)} bytes), "
          f"FS={FS/1e6:g} Msps, SPS={SPS}\n")
    all_ok = True
    for name in schemes:
        sink = LoopbackSink(FS, center_freq=DEFAULT_CENTER_HZ)
        frames, n = key_out(name, payload, sink)
        good = [f for f in frames
                if f["crc_ok"] and f["payload"] == payload]
        ok = len(good) == 1
        all_ok &= ok
        verdict = "PASS" if ok else "FAIL"
        detail = (f"recovered {good[0]['payload']!r}" if good
                  else f"{len(frames)} frame(s), no clean match")
        print(f"    {name.upper():5} {verdict}  "
              f"{n:5d} samples via {sink.transmit_count} transmit  ->  {detail}")
    print(f"\n[*] demod(modulate(x)) == x for all schemes: {all_ok}")
    return all_ok


def wire_to_hackrf(payload, schemes):
    """Show the identical driving code reaching a real device sink.

    Loads the sibling HackRFSink adapter and drives it. The sink is guarded
    (armed=False) and its device call is unimplemented, so this demonstrates
    that the protocol reaches the hardware SEAM and stops there -- the actual
    radiation is bench work, gated behind an explicit, legal, wired setup.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "hackrf_sink.py"
    spec = importlib.util.spec_from_file_location("hackrf_sink", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("[*] driving examples/hackrf_sink.py:HackRFSink (guarded, not armed)\n")
    for name in schemes:
        sink = mod.HackRFSink(DEFAULT_CENTER_HZ, FS)      # safe: armed=False
        print(f"    {name.upper():5} satisfies TXSink protocol: "
              f"{isinstance(sink, TXSink)}")
        try:
            key_out(name, payload, sink)
        except (RuntimeError, NotImplementedError) as e:
            print(f"          reached the device seam and stopped: "
                  f"{type(e).__name__}")
    print("\n[*] the guard is the point: the stack wires end-to-end in software; "
          "keying a\n    real radio is the two-SDR bench step (Phase E), on a "
          "legal, wired setup.")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Modulate the library's schemes as framed packets and key "
                    "them through a TXSink (offline by default).")
    p.add_argument("--payload", default="CQ DE SDR",
                   help="ASCII payload to frame and send")
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES),
                   choices=list(SCHEMES),
                   help="which schemes to run (default: all)")
    p.add_argument("--sink", choices=["loopback", "hackrf"], default="loopback",
                   help="loopback (offline, default) or the HackRF device seam")
    args = p.parse_args()

    payload = args.payload.encode("ascii", errors="replace")
    if args.sink == "hackrf":
        ok = wire_to_hackrf(payload, args.schemes)
    else:
        ok = run_offline(payload, args.schemes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
