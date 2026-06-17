#! /usr/bin/python3
"""dsb_sc_demo.py -- demodulate double-sideband suppressed-carrier (DSB-SC).

DSB-SC sits between AM and SSB: both sidebands are present (like AM) but the
carrier is suppressed (like SSB). With no carrier there's no envelope to follow,
so it needs a coherent reference -- for a baseband capture centered on the
suppressed carrier, the real part recovers the message.

Shows the AM/DSB-SC/SSB relationship by demodulating a synthetic DSB-SC signal
and comparing the spectrum to plain AM (which has the carrier spike DSB-SC lacks).

Library deps only (numpy). No hardware -- synthetic demonstration.

Usage:
    python examples/dsb_sc_demo.py
"""
import argparse
import sys
import wave
from math import gcd

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import dsb_sc_demod, resample_poly, normalize, psd

AUDIO_RATE = 48000


def main():
    p = argparse.ArgumentParser(description="DSB-SC demodulation demo.")
    p.add_argument("--rate", type=float, default=192000)
    p.add_argument("--out", default="dsbsc_audio.wav")
    args = p.parse_args()

    fs = args.rate
    n = 200_000
    t = np.arange(n) / fs
    # message: two audio tones
    msg = np.cos(2 * np.pi * 600 * t) + 0.5 * np.cos(2 * np.pi * 1500 * t)
    # DSB-SC: message modulates a (suppressed) carrier at baseband -> real msg
    # carried on both sidebands, NO carrier component
    dsbsc = msg.astype(np.complex64)
    # for contrast, an AM signal of the same message HAS a carrier (DC) term
    am = (1.0 + 0.5 * msg).astype(np.complex64)

    dsbsc += 0.02 * (np.random.randn(n) + 1j * np.random.randn(n))
    print(f"[*] DSB-SC @ {fs/1e3:g} kHz; demodulating")

    audio = dsb_sc_demod(dsbsc, fs)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    strong = sorted(freqs[np.argsort(spec)[-20:]])
    tones = sorted(set(int(round(f / 100) * 100) for f in strong if f > 100))
    print(f"[*] recovered tones near: {tones[:4]} Hz (expect 600, 1500)")

    # show the carrier difference: AM has a DC/carrier spike, DSB-SC doesn't
    f_am, p_am = psd(am, fs, nfft=2048)
    f_db, p_db = psd(dsbsc, fs, nfft=2048)
    am_dc = p_am[len(p_am) // 2]
    db_dc = p_db[len(p_db) // 2]
    print(f"[*] center (carrier) level: AM {am_dc:.0f} dB vs DSB-SC "
          f"{db_dc:.0f} dB")
    print(f"    -> DSB-SC suppresses the carrier by ~{am_dc - db_dc:.0f} dB")

    g = gcd(AUDIO_RATE, int(fs))
    audio = normalize(resample_poly(audio, AUDIO_RATE // g, int(fs) // g),
                      mode="peak", target=0.9)
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    print(f"[*] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
