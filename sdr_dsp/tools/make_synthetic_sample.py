#! /usr/bin/python3
"""Generate a synthetic broadcast-FM capture for sample_data. Deterministic.

The shipped sample capture must contain a station, and until a good
over-the-air recording exists, this makes one from scratch: a full stereo
multiplex (L+R, 19 kHz pilot, 38 kHz DSB-SC L-R, a small RDS-like carrier at
57 kHz), pre-emphasized, frequency-modulated at 75 kHz peak deviation, with
receiver-like impairments stamped on top (noise, DC offset, a small carrier
frequency offset), quantized to ci8 exactly as a HackRF capture would be.

Because every parameter is known, the file doubles as a measurement fixture:
the pilot is at exactly 19 kHz, the tones in the program are known, the
deviation is known, and the SNR is known. A receiver defect shows up as a
deviation from numbers this docstring states, not as a vague "sounds bad".

    uv run python tools/make_synthetic_sample.py
    uv run python tools/make_synthetic_sample.py --seconds 0.5 --out /tmp/x.iq

Writes sample_data/fm_synthetic_2Msps.iq + .sigmf-meta by default. The
plain fm_2Msps.iq is reserved for a REAL over-the-air capture (see
tools/import_reference_capture.py); this generator will not clobber it.
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy import signal as sig

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "sample_data" / "fm_synthetic_2Msps.iq"

FS = 2_000_000          # capture sample rate
AUDIO_FS = 48_000       # rate the program material is composed at
DEVIATION = 75_000      # peak deviation, Hz
PILOT_HZ = 19_000
PILOT_LEVEL = 0.09      # ~9% of deviation, per broadcast practice
RDS_LEVEL = 0.03
CFO_HZ = 240.0          # small tuning error, like a real capture
DC_COUNTS = 1.2         # LO-leakage DC offset, in ADC counts
PEAK_COUNTS = 68.0      # target peak ADC utilization (healthy, no clipping)
NOISE_SNR_DB = 38.0     # channel SNR of the synthetic "reception"


def _note(freq, n, fs, vibrato=5.0):
    """One synthesized note: a sine with soft harmonics and an envelope."""
    t = np.arange(n) / fs
    f = freq * (1.0 + 0.003 * np.sin(2 * np.pi * vibrato * t))
    ph = 2 * np.pi * np.cumsum(f) / fs
    x = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph))
    env = np.minimum(1.0, t / 0.02) * np.exp(-t * 2.2)
    return x * env


def make_program(n, fs, rng):
    """Stereo program material: a melody, a bass line, and a noise pad.

    Returns (left, right), each peak-normalized-ish. Musical enough that a
    human listening to the demodulated WAV immediately knows the chain works.
    """
    # A little melody (A minor-ish), eighth notes.
    melody = [440.0, 523.25, 659.25, 587.33, 523.25, 659.25, 440.0, 392.0]
    bass = [110.0, 110.0, 130.81, 98.0]
    left = np.zeros(n)
    right = np.zeros(n)
    step = int(fs * 0.25)
    for i in range(0, n, step):
        m = min(step, n - i)
        nt = _note(melody[(i // step) % len(melody)], m, fs)
        left[i:i + m] += 0.9 * nt
        right[i:i + m] += 0.55 * nt
    bstep = int(fs * 0.5)
    for i in range(0, n, bstep):
        m = min(bstep, n - i)
        nb = _note(bass[(i // bstep) % len(bass)], m, fs, vibrato=0.0)
        left[i:i + m] += 0.4 * nb
        right[i:i + m] += 0.6 * nb
    # a soft "air" pad, panned right, so the noise floor of the PROGRAM is
    # nonzero -- real broadcasts are never digitally silent between notes.
    pad = sig.lfilter(*sig.butter(2, 4000 / (fs / 2)), rng.standard_normal(n))
    pad /= np.max(np.abs(pad))
    left += 0.02 * pad
    right += 0.05 * pad

    # broadcast audio is bandlimited to 15 kHz before the multiplex
    b = sig.firwin(201, 15_000 / (fs / 2))
    left = sig.lfilter(b, 1.0, left)
    right = sig.lfilter(b, 1.0, right)
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    return left / peak, right / peak


def preemphasize(x, fs, tau_us=75.0):
    """Broadcast pre-emphasis: the inverse of the receiver's one-pole
    de-emphasis y[n] = a*x[n] + (1-a)*y[n-1], a = dt/(tau+dt)."""
    a = (1.0 / fs) / (tau_us * 1e-6 + 1.0 / fs)
    return sig.lfilter([1.0, -(1.0 - a)], [a], x) * a  # unity at DC


def make_capture(seconds, seed=20260914):
    rng = np.random.default_rng(seed)
    n_audio = int(AUDIO_FS * seconds)
    left, right = make_program(n_audio, AUDIO_FS, rng)
    left = preemphasize(left, AUDIO_FS)
    right = preemphasize(right, AUDIO_FS)

    # assemble the multiplex at the capture rate
    up, down = FS // np.gcd(FS, AUDIO_FS), AUDIO_FS // np.gcd(FS, AUDIO_FS)
    L = sig.resample_poly(left, up, down)
    R = sig.resample_poly(right, up, down)
    n = min(len(L), len(R), int(FS * seconds))
    L, R = L[:n], R[:n]
    t = np.arange(n) / FS

    pilot_ph = 2 * np.pi * PILOT_HZ * t
    mpx = (0.45 * (L + R)
           + 0.45 * (L - R) * np.cos(2 * pilot_ph)     # 38 kHz, pilot-locked
           + PILOT_LEVEL * np.cos(pilot_ph))
    # RDS-like ripple: 57 kHz BPSK at 1187.5 baud, biphase-ish shaping
    baud = 1187.5
    bits = rng.integers(0, 2, int(seconds * baud) + 2) * 2 - 1
    symstream = np.repeat(bits, int(np.ceil(n / len(bits))))[:n].astype(float)
    b57 = sig.firwin(401, 2400 / (FS / 2))
    symstream = sig.lfilter(b57, 1.0, symstream)
    mpx += RDS_LEVEL * symstream * np.cos(3 * pilot_ph)

    mpx /= np.max(np.abs(mpx))                          # peak dev = DEVIATION
    phase = 2 * np.pi * DEVIATION * np.cumsum(mpx) / FS
    phase += 2 * np.pi * CFO_HZ * t                     # small tuning error

    # per-component peak of exp(j*phase) is `amp`; leave a few counts of
    # headroom for the noise and the DC offset added below.
    amp = (PEAK_COUNTS - 3.0) / 128.0
    iq = amp * np.exp(1j * phase)

    # reception impairments: AWGN at a known SNR, shaped by an anti-alias
    # filter like the radio's own baseband filter (~0.75 * fs), plus LO DC.
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    bb = sig.firwin(129, 0.75)                          # of Nyquist
    noise = sig.lfilter(bb, 1.0, noise)
    noise *= amp * 10 ** (-NOISE_SNR_DB / 20) / np.sqrt(np.mean(np.abs(noise) ** 2))
    iq = iq + noise + (DC_COUNTS / 128.0) * (1 + 1j) / np.sqrt(2)

    # quantize exactly as a ci8 capture: scale by 128, clip, round
    i8 = np.empty(2 * n, dtype=np.int8)
    i8[0::2] = np.clip(np.round(iq.real * 128), -128, 127).astype(np.int8)
    i8[1::2] = np.clip(np.round(iq.imag * 128), -128, 127).astype(np.int8)
    return i8, n


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--frequency", type=float, default=98.5e6,
                   help="center frequency written to the sidecar (cosmetic)")
    args = p.parse_args()

    i8, n = make_capture(args.seconds, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    i8.tofile(out)
    meta = {
        "global": {
            "core:datatype": "ci8",
            "core:sample_rate": float(FS),
            "core:hw": "synthetic (tools/make_synthetic_sample.py)",
            "core:version": "1.0.0",
            "core:recorder": "sdr_dsp synthetic generator",
            "core:description": (
                "Synthetic stereo broadcast-FM multiplex: known program, "
                f"19 kHz pilot at {PILOT_LEVEL:g}, RDS-like 57 kHz at "
                f"{RDS_LEVEL:g}, {DEVIATION/1e3:g} kHz peak deviation, "
                f"{NOISE_SNR_DB:g} dB SNR, CFO {CFO_HZ:g} Hz, "
                f"seed {args.seed}."),
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": float(args.frequency),
            "core:datetime": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }],
        "annotations": [],
    }
    out.with_suffix(".sigmf-meta").write_text(
        json.dumps(meta, indent=2), newline="\n")
    print(f"wrote {out} ({2 * n} bytes, {n / FS:g}s at {FS / 1e6:g} Msps) "
          f"+ sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
