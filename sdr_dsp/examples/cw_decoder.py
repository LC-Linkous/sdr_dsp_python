#! /usr/bin/python3
"""cw_decoder.py -- decode Morse code (CW) from a captured tone.

CW (continuous wave) is the oldest digital mode and still heavily used by hams.
A carrier is keyed on and off; dits and dahs and the gaps between them spell out
characters. The chain:
    load IQ -> envelope -> threshold (on/off) -> estimate the dit length ->
    decode dits/dahs/gaps to text.

Library deps only (numpy). No hardware -- runs on a capture, or synthesizes a
message if none is given.

Usage:
    python examples/cw_decoder.py capture.iq
    python examples/cw_decoder.py                       # synthesizes "CQ DE SDR"
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import ook_envelope, ook_slice, estimate_symbol_rate, cw_decode

# encode side (for the synthetic demo only)
_MORSE_ENC = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}


def synth_cw(text, fs, dit_samples, tone_hz=800, noise=0.03):
    """Build a complex tone keyed with the Morse for `text`."""
    on = []   # list of (is_on, units)
    for ci, ch in enumerate(text.upper()):
        if ch == " ":
            on.append((False, 7))
            continue
        code = _MORSE_ENC.get(ch, "")
        for si, sym in enumerate(code):
            on.append((True, 1 if sym == "." else 3))
            if si < len(code) - 1:
                on.append((False, 1))            # intra-char gap
        on.append((False, 3))                     # inter-char gap
    # render to a keyed tone
    segs = []
    for is_on, units in on:
        n = int(units * dit_samples)
        t = np.arange(n) / fs
        if is_on:
            segs.append(np.exp(2j * np.pi * tone_hz * t))
        else:
            segs.append(np.zeros(n, dtype=np.complex64))
    iq = np.concatenate(segs).astype(np.complex64)
    iq += noise * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    return iq


def main():
    p = argparse.ArgumentParser(description="Decode Morse/CW from a capture.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=48000)
    p.add_argument("--wpm", type=float, default=20, help="synth words/min")
    p.add_argument("--smooth", type=int, default=0,
                   help="moving-average the envelope over N samples")
    args = p.parse_args()

    truth = None
    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        # standard: dit length (sec) = 1.2 / wpm
        dit_samples = int(1.2 / args.wpm * fs)
        truth = "CQ DE SDR"
        iq = synth_cw(truth, fs, dit_samples)
        print(f"[*] synthetic CW: {truth!r} @ {args.wpm:g} wpm, {fs/1e3:g} kHz")

    # 1. envelope
    env = ook_envelope(iq)
    if args.smooth > 1:
        env = np.convolve(env, np.ones(args.smooth) / args.smooth, mode="same")

    # 2. on/off slice
    bits = ook_slice(env)
    on_frac = float(np.mean(bits))
    print(f"[*] {len(iq):,} samples; {on_frac*100:.0f}% keyed on")

    # 3. estimate the dit length (the shortest on/off unit)
    spb, _ = estimate_symbol_rate(bits, fs, min_run=3)
    if spb <= 0:
        print("[!] couldn't find keying -- is there a signal?")
        return 1
    wpm = 1.2 / (spb / fs)
    print(f"[*] dit ~ {spb:.0f} samples ({wpm:.0f} wpm)")

    # 4. decode
    text = cw_decode(bits, spb)
    print(f"[*] decoded: {text!r}")
    if truth is not None:
        match = text.replace(" ", "") == truth.replace(" ", "")
        print(f"[*] vs sent {truth!r}: {'MATCH' if match else 'differs'}")
        if not match:
            print("    (CW timing is loose; try --smooth or check --wpm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
