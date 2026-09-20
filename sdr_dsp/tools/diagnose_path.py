#! /usr/bin/python3
"""Isolate WHERE the pilot dies: the station, the capture path, or the code.

Three measurements of the same station at the same gain (8 Msps,
lna=32/vga=20 -- the fm_reference values), through the same measuring code:

  A. the by-ear-verified reference FILE (control: proves the measuring code
     on this machine reads a known-good capture correctly)
  B. a live capture via capture_array (the path preflight/gain-search use)
  C. a live capture to FILE + reload (the path that made the references)

Run pointed at the fm_reference station so the station itself is a known
quantity (it was stereo with a +23 dB pilot two days ago):

    uv run python tools/diagnose_path.py 103.7e6 \\
        --reference ../path/to/hackrfpy/tests/fm_reference/fm_103.7MHz_8Msps.iq

Readout:
  A high, B low, C low   -> live environment/board state changed; station or
                            antenna situation differs from capture day
  A high, B low, C high  -> the capture_array path is the fault: collection
                            tools switch to the file path, hackrfpy gets a
                            bug report
  A high, B high, C high -> everything works; earlier failures were the
                            OTHER stations (e.g. mono broadcasters have no
                            pilot) -- rerun preflight with more candidates
  A low                  -> stop: the measuring code is broken on this
                            machine; nothing downstream is trustworthy
  Also read the carrier column: a live capture sitting hundreds of kHz off
  center is a TUNING fault in that path, which lowers channel excess and
  kills the pilot exactly as observed. And if B' (settle-skipped) is much
  better than B, the fix is simply discarding the first half second.

It also writes B and C demodulated to WAV next to this script -- EARS are
the tiebreaker: glitchy/choppy audio = dropped samples; clean audio with a
low pilot number = measurement problem.

SAFETY: receive-only.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from sdr_dsp.core import (capture_health, deemphasis, design_lowpass,
                          fir_apply, fm_demod, fm_pilot_excess_db,
                          resample_poly)

RATE = 8_000_000
SECONDS = 2.0
LNA, VGA = 32, 20


def carrier_offset_hz(iq, rate, span=600e3, nfft=8192):
    """Where the strongest carrier near DC actually sits, from the PSD.
    A live capture that tunes wrong shows up here immediately."""
    spec = np.zeros(nfft)
    for k in range(min(24, len(iq) // nfft)):
        seg = iq[k * nfft:(k + 1) * nfft] * np.hanning(nfft)
        spec += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
    win = np.abs(f) <= span
    return float(f[win][np.argmax(spec[win])])


def measure(tag, iq, rate):
    h = capture_health(iq, rate, channel_bw=100e3)
    p = fm_pilot_excess_db(iq, rate)
    off = carrier_offset_hz(iq, rate)
    ptxt = "  n/a" if p is None else f"{p:+5.1f}"
    print(f"  {tag:34s} {h['adc_counts']:6.1f}/128  "
          f"channel {h['channel_excess_db']:+5.1f} dB   pilot {ptxt} dB   "
          f"carrier {off / 1e3:+7.1f} kHz off center")
    return p


def demod_to_wav(iq, rate, out_path):
    import wave
    taps = design_lowpass(100e3, rate, num_taps=201)
    iq = fir_apply(iq, taps)[len(taps):]
    audio = fm_demod(iq, deviation_hz=75e3, sample_rate=rate)
    audio = resample_poly(audio, 48_000, int(rate))
    audio = deemphasis(audio, 48_000)
    audio = fir_apply(audio, design_lowpass(15_000, 48_000, num_taps=101))
    audio = audio[2000:]
    peak = np.percentile(np.abs(audio), 99.9) or 1.0
    pcm = np.clip(audio / peak * 0.9, -1, 1)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48_000)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    print(f"      -> wrote {out_path} (listen to it)")


def load_ci8(path, max_seconds=SECONDS):
    raw = np.fromfile(path, dtype=np.int8, count=2 * int(RATE * max_seconds))
    return ((raw[0::2].astype(np.float32)
             + 1j * raw[1::2].astype(np.float32)) / 128.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("freq", type=float, help="station Hz, e.g. 103.7e6")
    ap.add_argument("--reference", default=None,
                    help="path to the verified fm_reference .iq for this "
                         "station (control measurement)")
    ap.add_argument("--tools-dir", default=None)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    n = int(RATE * args.seconds)
    print(f"station {args.freq / 1e6:g} MHz, {RATE / 1e6:g} Msps, "
          f"lna={LNA} vga={VGA} amp=off, {args.seconds:g}s\n")

    # A. control: the verified reference file
    if args.reference:
        ref = load_ci8(args.reference, args.seconds)
        measure("A. reference file (verified)", ref, RATE)
    else:
        print("  A. (no --reference given; control skipped)")

    from hackrfpy import HackRF
    h = HackRF(tools_dir=args.tools_dir, verbose=False)

    # B. live, via capture_array (the path the tools use); measured in full
    # AND with the first 0.5 s dropped, to expose a settle transient
    iq_b = h.capture_array(args.freq, RATE, n, lna=LNA, vga=VGA, amp=False)
    measure("B. live capture_array", iq_b, RATE)
    if len(iq_b) > int(RATE * 0.75):
        measure("B'. same, first 0.5s skipped", iq_b[int(RATE * 0.5):], RATE)
    demod_to_wav(iq_b, RATE, here / "diag_B_capture_array.wav")

    # C. live, to file + reload (the path that made the references)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "diag_capture.iq"
        h.capture(args.freq, RATE, num_samples=n, out=str(out),
                  lna=LNA, vga=VGA, amp=False, sigmf=True)
        iq_c = load_ci8(out, args.seconds)
    measure("C. live capture-to-file", iq_c, RATE)
    demod_to_wav(iq_c, RATE, here / "diag_C_capture_to_file.wav")

    print("\nInterpretation is in the docstring at the top of this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())