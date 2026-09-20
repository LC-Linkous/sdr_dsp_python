#! /usr/bin/python3
"""Import a validated over-the-air capture as sample_data/fm_2Msps.iq.

Takes a reference IQ recording (any rate with a SigMF sidecar -- e.g. the
8 Msps by-ear-verified captures in hackrfpy's tests/fm_reference/), verifies
it is really a broadcast FM station with the same checks the collection
tools use, then channel-decimates a slice of it to the library's 2 Msps
sample rate, re-quantizes to ci8, and writes it with a sidecar that records
where it came from and what was measured -- so the shipped sample is never
again a file nobody can vouch for.

    uv run python tools/import_reference_capture.py \\
        ../hackrfpy/tests/fm_reference/fm_103.7MHz_8Msps.iq

The import refuses captures that fail validation (no pilot, no carrier,
clipping): sample data must be above suspicion, because everything else is
debugged against it.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from sdr_dsp.core import (capture_health, design_lowpass, fir_apply,
                          fm_pilot_excess_db)
from sdr_dsp.io.sigmf import read_meta

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "sample_data" / "fm_2Msps.iq"
OUT_RATE = 2_000_000
PILOT_OK_DB = 6.0


def load_slice(path, rate, skip_s, take_s):
    """Load [skip_s, skip_s + take_s) of a ci8 capture as complex64."""
    n_skip = int(rate * skip_s)
    n_take = int(rate * take_s)
    raw = np.fromfile(path, dtype=np.int8, count=2 * n_take,
                      offset=2 * n_skip)
    if raw.size < 2 * n_take:
        raise SystemExit(f"capture too short: wanted {take_s:g}s after "
                         f"skipping {skip_s:g}s, got {raw.size / 2 / rate:g}s")
    return ((raw[0::2].astype(np.float32)
             + 1j * raw[1::2].astype(np.float32)) / 128.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("source", help="reference .iq with a .sigmf-meta sidecar")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seconds", type=float, default=2.0,
                   help="length of the imported slice (default 2.0)")
    p.add_argument("--skip", type=float, default=1.0,
                   help="seconds to skip at the start (settling, squelch "
                        "tails; default 1.0)")
    p.add_argument("--note", default=None,
                   help="extra provenance note, e.g. 'verified by ear'")
    args = p.parse_args()

    src = Path(args.source)
    meta = read_meta(str(src))
    g = meta["global"]
    rate = float(g["core:sample_rate"])
    if g.get("core:datatype") != "ci8":
        raise SystemExit(f"only ci8 sources supported, got "
                         f"{g.get('core:datatype')}")
    freq = float(meta["captures"][0].get("core:frequency", 0.0))
    if rate % OUT_RATE:
        raise SystemExit(f"source rate {rate:g} is not an integer multiple "
                         f"of {OUT_RATE:g}")
    decim = int(rate // OUT_RATE)

    print(f"[*] {src.name}: {rate / 1e6:g} Msps at {freq / 1e6:.4g} MHz, "
          f"decimating x{decim}")
    iq = load_slice(src, rate, args.skip, args.seconds)

    # ---- validate BEFORE importing: sample data must be above suspicion --
    health = capture_health(iq, rate, channel_bw=100e3)
    pilot = fm_pilot_excess_db(iq, rate)
    ptxt = "n/a" if pilot is None else f"{pilot:+.1f} dB"
    print(f"[*] source slice: {health['adc_counts']:.1f}/128 counts, "
          f"channel {health['channel_excess_db']:+.1f} dB, pilot {ptxt}")
    if not health["ok"]:
        raise SystemExit("REFUSED: source fails capture_health: "
                         + "; ".join(health["reasons"]))
    if pilot is None or pilot < PILOT_OK_DB:
        raise SystemExit(f"REFUSED: no stereo pilot "
                         f"({pilot} dB) -- not a verified FM station")

    # ---- channel-decimate, preserving the recorded level ------------------
    if decim > 1:
        taps = design_lowpass(0.45 * OUT_RATE, rate,
                              num_taps=16 * decim + 1)
        iq = fir_apply(iq, taps)[len(taps):]     # drop the transient
        iq = iq[::decim]
    out_iq = iq[:int(OUT_RATE * args.seconds)]

    # The channel filter strips the out-of-channel energy that set the
    # original peak, which can leave the slice using only a few ADC bits.
    # Rescale to a healthy level before requantizing -- this changes no
    # ratio inside the channel, and the factor is recorded in the sidecar
    # so the original level stays recoverable.
    peak = float(max(np.max(np.abs(out_iq.real)), np.max(np.abs(out_iq.imag))))
    rescale = (96.0 / 128.0) / peak if peak > 0 else 1.0
    out_iq = out_iq * rescale

    # re-validate at the output rate: the import must not have broken it
    health2 = capture_health(out_iq, OUT_RATE, channel_bw=100e3)
    pilot2 = fm_pilot_excess_db(out_iq, OUT_RATE)
    print(f"[*] imported slice: {health2['adc_counts']:.1f}/128 counts, "
          f"channel {health2['channel_excess_db']:+.1f} dB, "
          f"pilot {pilot2:+.1f} dB")
    if not health2["ok"] or pilot2 is None or pilot2 < PILOT_OK_DB:
        raise SystemExit("REFUSED: the decimated slice fails validation -- "
                         "this is an import bug, not a source problem")

    i8 = np.empty(2 * out_iq.size, dtype=np.int8)
    i8[0::2] = np.clip(np.round(out_iq.real * 128), -128, 127).astype(np.int8)
    i8[1::2] = np.clip(np.round(out_iq.imag * 128), -128, 127).astype(np.int8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    i8.tofile(out)

    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    out_meta = {
        "global": {
            "core:datatype": "ci8",
            "core:sample_rate": float(OUT_RATE),
            "core:hw": g.get("core:hw", "HackRF One"),
            "core:version": "1.0.0",
            "core:recorder": "sdr_dsp tools/import_reference_capture.py",
            "core:description": (
                f"Over-the-air broadcast FM, imported from {src.name} "
                f"({rate / 1e6:g} Msps, decimated x{decim}, "
                f"{args.skip:g}s-{args.skip + args.seconds:g}s slice). "
                f"Validated on import: pilot {pilot2:+.1f} dB, channel "
                f"{health2['channel_excess_db']:+.1f} dB, "
                f"{health2['adc_counts']:.0f}/128 counts."
                + (f" {args.note}" if args.note else "")),
            "sdr_dsp:source_file": src.name,
            "sdr_dsp:source_sha256": src_sha,
            "sdr_dsp:source_sample_rate": rate,
            "sdr_dsp:pilot_excess_db": round(pilot2, 2),
            "sdr_dsp:level_rescale_db": round(20 * np.log10(rescale), 2),
            **{k: v for k, v in g.items() if k.startswith("hackrf:")},
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": freq,
            "core:datetime": meta["captures"][0].get(
                "core:datetime",
                datetime.datetime.now(datetime.timezone.utc).isoformat()),
        }],
        "annotations": [],
    }
    out.with_suffix(".sigmf-meta").write_text(
        json.dumps(out_meta, indent=2), newline="\n")
    print(f"[*] wrote {out} ({i8.size} bytes, "
          f"{out_iq.size / OUT_RATE:g}s at {OUT_RATE / 1e6:g} Msps) + sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
