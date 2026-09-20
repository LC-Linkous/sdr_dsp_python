#! /usr/bin/python3
"""Build a varied development corpus from one local FM station. READ-ONLY.

This is the development counterpart to `examples/collect_sample_data.py`.
That script is the simple, shipped one: point it at a band preset, get a
sensible capture. This one is for us, right now -- it records the *same*
station many different ways so the library has something to regression-test
against, and drops each group into its own `data_N/` folder with a manifest.

Why several folders instead of one good capture
-----------------------------------------------
A single clean recording only proves the happy path. What the test suite
actually lacks is the surrounding territory: a station sitting off-center so
the tuning path gets exercised, the same signal at four sample rates so the
resampler's up/down ratios get exercised, gain set deliberately too low and
deliberately too hot, and a genuinely empty capture as a known-negative. Each
`data_N/` folder is one of those axes, varied one at a time so a failure
points somewhere specific.

Gain is calibrated ONCE against the station, then every dataset is derived by
offsetting from that reference. So "weak" means weak relative to a level we
measured on your antenna today, not a guess baked into the file.

SETUP -- edit the CONFIG block below before running. The station is a
variable here on purpose: you'll run this a few times while tuning the
corpus, and retyping a CLI flag each time invites mistakes.

    python tools/preflight_collection.py             # FIRST: verify the setup
    python tools/collect_dev_data.py --station 98.5e6 --quiet 91.3e6
    python tools/collect_dev_data.py                 # collect everything
    python tools/collect_dev_data.py --only 1 3      # just data_1 and data_3
    python tools/collect_dev_data.py --plan          # show the plan, record nothing
    python tools/collect_dev_data.py --calibrate     # find gain and stop

Legality: receive only. This never transmits and never writes firmware.
"""

# ===========================================================================
#  CONFIG -- EDIT THESE
# ===========================================================================

# A strong local FM station. Pick one that is loud and reliably on the air;
# the whole corpus is built from it, so a marginal station makes everything
# downstream marginal too.
STATION_HZ = 98_500_000
STATION_NAME = "98.5 FM"

# A quiet spot with nothing on it, used for the negative controls.
# `tools/preflight_collection.py` finds and VERIFIES one from a band sweep;
# prefer its suggestion (pass it as --quiet) over this default. Note the
# default sits in the guard band below 88 MHz, which only exists in ITU
# Region 2 (the Americas) -- in Region 1/3 the broadcast band starts at
# 87.5 MHz and this frequency may have a station on it.
QUIET_HZ = 87_700_000

# Where the corpus goes. Relative to the sdr_dsp project directory.
OUT_ROOT = "dev_data"

# Path to hackrf-tools binaries, if they are not already on PATH.
# e.g. r"C:\hackrf-tools-windows" on Windows. None = use PATH.
TOOLS_DIR = None

# Set True to skip the interactive prompts (the antenna-disconnect step in
# data_5 will be skipped rather than waited on).
UNATTENDED = False

# Rough ceiling on the whole corpus. The script totals the plan first and
# asks before exceeding this. Captures are int8 I/Q: bytes = rate * secs * 2.
MAX_TOTAL_MB = 60

# ===========================================================================

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from sdr_dsp.core import (capture_health, fm_pilot_excess_db,
                          frequency_shift, search_gain)
from sdr_dsp.sources.probe import probe_capture

try:
    from hackrfpy import HackRF, load_iq
    from hackrfpy.exceptions import HackRFError
except ModuleNotFoundError:
    sys.stderr.write(
        "ERROR: hackrfpy is not installed. It is an optional extra:\n"
        "    uv sync --extra examples-hackrf\n"
        "It also needs the hackrf-tools binaries at the OS level.\n")
    sys.exit(1)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT_DIR = PROJECT / OUT_ROOT

LNA_STEPS = list(range(0, 41, 8))
VGA_STEPS = list(range(0, 63, 2))
TARGET_LO, TARGET_HI = 45.0, 110.0
CLIP_COUNTS = 120.0

FM_CHANNEL_BW = 100_000          # half-width used for carrier checks


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
# Each dataset varies ONE axis. `captures` entries carry:
#   name        file stem inside the folder
#   freq_offset how far the RADIO is tuned from STATION_HZ. Tuning the radio
#               low by 250 kHz puts the station at +250 kHz in baseband.
#   rate        sample rate, sps
#   secs        duration
#   gain_db     dB offset from the calibrated reference gain
#   expect      "signal" | "weak" | "empty" -- what we intend to get, used to
#               judge the health result rather than blanket-failing
#   purpose     why this file exists, written into the manifest

def build_plan():
    return {
        1: {
            "slug": "reference",
            "title": "Clean reference capture",
            "why": ("The golden recording: station centered, calibrated gain, "
                    "the rate the examples default to. Everything else in the "
                    "corpus is a deliberate deviation from this one. This is "
                    "the file that should replace sample_data/fm_2Msps.iq."),
            "captures": [
                dict(name="fm_reference", freq_offset=0, rate=2e6, secs=1.0,
                     gain_db=0, expect="signal",
                     purpose="centered, nominal gain, the baseline"),
                dict(name="fm_reference_long", freq_offset=0, rate=2e6,
                     secs=2.0, gain_db=0, expect="signal",
                     purpose="longer take for audio-quality listening tests"),
            ],
        },
        2: {
            "slug": "tuning",
            "title": "Station offset within the capture",
            "why": ("Exercises frequency_shift and the receiver's --tune path. "
                    "The radio is tuned AWAY from the station, so the station "
                    "sits at a known offset in baseband and has to be mixed "
                    "down before demodulation. A capture whose signal is not "
                    "at DC is the normal case in the field and the one the "
                    "test suite had no example of."),
            "captures": [
                dict(name="fm_offset_p250k", freq_offset=-250_000, rate=2e6,
                     secs=0.5, gain_db=0, expect="signal",
                     purpose="station at +250 kHz; recover with --tune 250e3"),
                dict(name="fm_offset_m400k", freq_offset=+400_000, rate=2e6,
                     secs=0.5, gain_db=0, expect="signal",
                     purpose="station at -400 kHz; recover with --tune -400e3"),
                dict(name="fm_offset_p750k", freq_offset=-750_000, rate=2e6,
                     secs=0.5, gain_db=0, expect="signal",
                     purpose=("station at +750 kHz, near the band edge: "
                              "checks the channel filter's skirt")),
            ],
        },
        3: {
            "slug": "rates",
            "title": "Sample rate ladder",
            "why": ("The same signal at four rates, so resample_poly runs at "
                    "four different up/down ratios down to 48 kHz audio. 2 "
                    "Msps gives 3/125; the others give ratios with different "
                    "filter lengths and decimation factors. Also feeds "
                    "decimation_stages.py and resampler_benchmark.py."),
            "captures": [
                dict(name="fm_2Msps", freq_offset=0, rate=2e6, secs=0.5,
                     gain_db=0, expect="signal", purpose="48k ratio 3/125"),
                dict(name="fm_4Msps", freq_offset=0, rate=4e6, secs=0.5,
                     gain_db=0, expect="signal", purpose="48k ratio 3/250"),
                dict(name="fm_8Msps", freq_offset=0, rate=8e6, secs=0.25,
                     gain_db=0, expect="signal", purpose="48k ratio 3/500"),
                dict(name="fm_10Msps", freq_offset=0, rate=10e6, secs=0.25,
                     gain_db=0, expect="signal", purpose="48k ratio 6/1250"),
            ],
        },
        4: {
            "slug": "gain",
            "title": "Gain ladder",
            "why": ("The same station recorded too quiet, correct, and too "
                    "hot. This is the axis that caused the original bug: a "
                    "capture 24 dB below reference looks fine in the terminal "
                    "and contains nothing usable. Gives capture_health real "
                    "material at both ends, and feeds agc_demo.py, "
                    "power_calibration.py, and dc_offset_demo.py."),
            "captures": [
                dict(name="fm_gain_low", freq_offset=0, rate=2e6, secs=0.5,
                     target_counts=3.0, expect="weak",
                     purpose=("driven down to ~3 of 127 ADC counts, which is "
                              "where the original bad capture sat: should "
                              "FAIL the health check")),
                dict(name="fm_gain_under", freq_offset=0, rate=2e6, secs=0.5,
                     gain_db=-12, expect="signal",
                     purpose="12 dB low: usable but quiet, good for AGC"),
                dict(name="fm_gain_nominal", freq_offset=0, rate=2e6,
                     secs=0.5, gain_db=0, expect="signal",
                     purpose="calibrated reference level"),
                dict(name="fm_gain_hot", freq_offset=0, rate=2e6, secs=0.5,
                     gain_db=+12, expect="signal",
                     purpose=("12 dB hot: likely clipping, so the recovered "
                              "audio should show distortion")),
            ],
        },
        5: {
            "slug": "negative",
            "title": "Known-negative controls",
            "why": ("Captures that deliberately contain nothing. The library "
                    "needs these as fixtures: every no-signal code path, the "
                    "capture_health thresholds, and the noise-floor reference "
                    "for calibration currently have no real data behind them. "
                    "An empty capture is only a bug when it was unintentional."),
            "captures": [
                dict(name="quiet_channel", freq=QUIET_HZ, rate=2e6, secs=0.5,
                     gain_db=0, expect="empty",
                     purpose=("empty frequency at normal gain: noise floor "
                              "with the antenna connected")),
                dict(name="antenna_disconnected", freq_offset=0, rate=2e6,
                     secs=0.5, gain_db=0, expect="empty",
                     prompt=("Disconnect the antenna from the HackRF, then "
                             "press Enter"),
                     purpose=("station frequency, no antenna: isolates "
                              "receiver noise from over-the-air noise")),
                dict(name="min_gain", freq_offset=0, rate=2e6, secs=0.5,
                     gain_db=-99, expect="empty",
                     restore_prompt="Reconnect the antenna, then press Enter",
                     purpose=("strong station at minimum gain: signal present "
                              "but below the ADC's reach")),
            ],
        },
        6: {
            "slug": "wideband",
            "title": "Wideband band context",
            "why": ("A wide slice of the FM band containing several stations "
                    "at once, plus a sweep across the whole band. Feeds "
                    "channelizer.py, signal_survey.py, channel_sweep.py and "
                    "spectrum_analyzer.py, none of which have multi-signal "
                    "data to work with today."),
            "captures": [
                dict(name="fm_band_20Msps", freq_offset=0, rate=20e6,
                     secs=0.25, gain_db=0, expect="signal",
                     purpose=("20 MHz span centered on the station: multiple "
                              "stations visible for channelization")),
            ],
            "sweep": (88_000_000, 108_000_000),
        },
    }


# ---------------------------------------------------------------------------
# gain calibration
# ---------------------------------------------------------------------------
def _snap(lna, vga):
    lna = min(LNA_STEPS, key=lambda s: abs(s - lna))
    vga = min(VGA_STEPS, key=lambda s: abs(s - vga))
    return lna, vga


def calibrate_gain(h, rate):
    """Find a gain that captures the STATION, not just a level. Returns dict.

    Uses sdr_dsp.core.search_gain with the 19 kHz stereo pilot as the
    quality metric: the front end (RF amp + LNA) is chosen by how clearly
    the pilot is measured, then the VGA sets the ADC level. The old
    level-only walk could calibrate the whole corpus against an amplified
    noise floor -- the level window would be met, every dataset would
    inherit it, and nothing would notice until someone listened.

    One calibration for the whole corpus. Everything else is expressed as a
    dB offset from this, so "30 dB low" means 30 dB below a level actually
    measured on this antenna, at this location, today -- which is the only
    way a weak-signal fixture stays meaningful when the corpus is rebuilt
    somewhere else.
    """
    probe_n = int(rate * 0.05)

    def probe(lna, vga, amp):
        # File-path capture, NOT capture_array: the stdout-pipe path drops
        # samples on Windows and buries the pilot (sdr_dsp.sources.probe).
        return probe_capture(h, STATION_HZ, rate, probe_n, lna=lna, vga=vga,
                             amp=amp)

    r = search_gain(probe, lna_steps=LNA_STEPS, vga_steps=VGA_STEPS,
                    quality=lambda iq: fm_pilot_excess_db(iq, rate),
                    target=(TARGET_LO, TARGET_HI), clip=CLIP_COUNTS)
    tried = [{"lna": p[0], "vga": p[1], "amp": p[2], "counts": p[3]}
             for p in r["probes"]]
    for p in r["probes"]:
        print(f"    probe lna={p[0]:2d} vga={p[1]:2d} "
              f"amp={'on ' if p[2] else 'off'} -> peak {p[3]:6.1f} counts")
    if r["quality_db"] is not None:
        print(f"    stereo pilot at best front end: {r['quality_db']:+.1f} dB "
              f"above the demodulated noise floor")
    return {"lna": r["lna"], "vga": r["vga"], "amp": r["amp"],
            "total_db": r["lna"] + r["vga"], "counts": r["counts"],
            "pilot_db": r["quality_db"], "status": r["status"],
            "probes": tried}


def gain_for(ref, offset_db):
    """Apply a dB offset to the reference gain, snapped to the hardware grid.

    VGA absorbs the offset first because its 2 dB steps land closer to the
    requested figure; the LNA's 8 dB steps only come in when VGA runs out of
    range. The dB actually applied is returned alongside, since snapping and
    clamping mean it rarely equals the request exactly.
    """
    if offset_db <= -99:
        return 0, 0, False, -(ref["total_db"] + (14 if ref.get("amp") else 0))
    vga = ref["vga"] + offset_db
    lna = ref["lna"]
    while vga > VGA_STEPS[-1] and lna < LNA_STEPS[-1]:
        vga -= 8
        lna += 8
    while vga < VGA_STEPS[0] and lna > LNA_STEPS[0]:
        vga += 8
        lna -= 8
    lna, vga = _snap(max(0, min(40, lna)), max(0, min(62, vga)))
    return lna, vga, ref.get("amp", False), (lna + vga) - ref["total_db"]


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def plan_bytes(plan, only):
    total = 0
    rows = []
    for n, ds in sorted(plan.items()):
        if only and n not in only:
            continue
        sub = sum(int(c["rate"] * c["secs"] * 2) for c in ds["captures"])
        rows.append((n, ds["slug"], len(ds["captures"]), sub))
        total += sub
    return rows, total


def record_one(h, folder, cap, ref, dataset_no):
    """Record one capture, validate it, and return its manifest entry."""
    freq = cap.get("freq", STATION_HZ + cap.get("freq_offset", 0))
    rate = cap["rate"]
    secs = cap["secs"]
    if "target_counts" in cap:
        # Aim at an absolute ADC level rather than a relative offset. A
        # "weak" fixture defined as -30 dB lands wherever the reference
        # happened to be; defined as ~3 counts it reproduces the original
        # bad capture regardless of how strong the station is today.
        want_db = 20 * np.log10(cap["target_counts"] / max(ref["counts"], 1.0))
        lna, vga, amp, applied_db = gain_for(ref, want_db)
    else:
        lna, vga, amp, applied_db = gain_for(ref, cap.get("gain_db", 0))

    if cap.get("prompt"):
        if UNATTENDED:
            print(f"   skipping {cap['name']} (needs "
                  f"'{cap['prompt']}' and UNATTENDED is set)")
            return None
        input(f"\n   ACTION NEEDED: {cap['prompt']} ")

    path = folder / f"{cap['name']}.iq"
    print(f"   {cap['name']}: {freq/1e6:.3f} MHz, {rate/1e6:g} Msps, "
          f"{secs:g}s, lna={lna} vga={vga} amp={'on' if amp else 'off'} "
          f"({applied_db:+.0f} dB)")
    try:
        h.capture(freq, rate, num_samples=int(rate * secs), out=str(path),
                  lna=lna, vga=vga, amp=amp, sigmf=True)
    except HackRFError as e:
        print(f"      capture failed: {e}", file=sys.stderr)
        return None

    iq = load_iq(str(path))
    # Check the channel where the station actually IS. capture_health looks
    # around DC, so an intentionally-offset capture would otherwise report
    # "no carrier" simply because we tuned away from it. Mixing it down first
    # is exactly what the receiver does with --tune, so this validates the
    # file the way it will be used.
    offset = 0 if "freq" in cap else -cap.get("freq_offset", 0)
    probe = frequency_shift(iq, -offset, rate) if offset else iq
    health = capture_health(probe, rate, channel_bw=FM_CHANNEL_BW)
    excess = health["channel_excess_db"]
    # For captures of the station itself, the decisive check is the 19 kHz
    # stereo pilot in the demodulated signal -- amplified noise can fill the
    # ADC window and even hump the channel, but it cannot manufacture a
    # pilot. Quiet/empty captures skip it (no pilot expected).
    pilot = (fm_pilot_excess_db(probe, rate)
             if cap.get("expect", "signal") != "empty" else None)
    if pilot is not None and cap.get("expect") == "signal" and pilot < 6.0:
        health["ok"] = False
        health["reasons"].append(
            f"no 19 kHz stereo pilot in the demodulated signal "
            f"({pilot:+.1f} dB): this is not a broadcast FM station, "
            f"whatever the level says")
    print(f"      peak {health['adc_counts']:6.1f}/128 counts"
          + (f", channel {excess:+.1f} dB" if excess is not None else "")
          + (f", pilot {pilot:+.1f} dB" if pilot is not None else ""))

    expect = cap.get("expect", "signal")
    at_min_gain = (lna, vga) == (LNA_STEPS[0], VGA_STEPS[0])
    if expect == "signal":
        verdict = "ok" if health["ok"] else "UNEXPECTEDLY EMPTY"
    elif expect == "weak":
        if not health["ok"]:
            verdict = "ok (intentionally weak)"
        elif at_min_gain:
            # Nothing left to attenuate with. On a very strong station the
            # radio simply cannot be turned down far enough to manufacture a
            # noise-floor capture; that needs external attenuation or a
            # disconnected antenna, which data_5 already covers.
            verdict = ("still above the health threshold at minimum gain "
                       "-- use the data_5 controls for a true empty fixture")
        else:
            verdict = "UNEXPECTEDLY STRONG -- lower target_counts"
    else:
        if not health["ok"]:
            verdict = "ok (intentionally empty)"
        elif at_min_gain:
            # A very strong station can stay above the health threshold even
            # at zero gain; that is a property of the station, not a mistake
            # in the run.
            verdict = ("ok (still above threshold at minimum gain -- "
                       "strong station)")
        else:
            verdict = "UNEXPECTEDLY HAS SIGNAL"
    flag = "" if verdict.startswith("ok") or at_min_gain else \
        "   <-- CHECK THIS"
    print(f"      {verdict}{flag}")

    if cap.get("restore_prompt") and not UNATTENDED:
        input(f"\n   ACTION NEEDED: {cap['restore_prompt']} ")

    return {
        "file": path.name,
        "sigmf": path.with_suffix(".sigmf-meta").name,
        "purpose": cap["purpose"],
        "frequency_hz": freq,
        "station_offset_hz": (None if "freq" in cap
                              else -cap.get("freq_offset", 0)),
        "sample_rate": rate,
        "seconds": secs,
        "lna_db": lna, "vga_db": vga, "amp": bool(amp),
        "pilot_excess_db": (None if pilot is None else round(pilot, 2)),
        "gain_offset_db_requested": cap.get("gain_db", 0),
        "gain_offset_db_applied": applied_db,
        "samples": int(len(iq)),
        "size_bytes": path.stat().st_size,
        "expect": expect,
        "health": {
            "ok": health["ok"],
            "adc_counts": round(health["adc_counts"], 2),
            "channel_excess_db": (None if excess is None else round(excess, 2)),
            "peak_dbfs": round(health["peak_dbfs"], 2),
            "reasons": health["reasons"],
        },
        "verdict": verdict,
    }


def write_dataset_readme(folder, n, ds, entries, ref):
    lines = [
        f"# data_{n} -- {ds['title']}", "",
        ds["why"], "",
        f"Station: **{STATION_NAME}** at {STATION_HZ/1e6:.3f} MHz.",
        f"Reference gain: LNA {ref['lna']} dB, VGA {ref['vga']} dB, amp "
        f"{'on' if ref.get('amp') else 'off'} "
        f"(peak {ref['counts']:.0f}/128 counts at capture time).", "",
        "| file | purpose | freq | rate | gain | peak | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e['file']}` | {e['purpose']} | "
            f"{e['frequency_hz']/1e6:.3f} MHz | {e['sample_rate']/1e6:g} Msps | "
            f"{e['gain_offset_db_applied']:+.0f} dB | "
            f"{e['health']['adc_counts']:.0f}/128 | {e['verdict']} |")
    lines += ["", "Load any of these with:", "", "```python",
              "from sdr_dsp.sources import FileSource", "",
              f'src = FileSource("{OUT_ROOT}/data_{n}_{ds["slug"]}/'
              f'{entries[0]["file"]}")' if entries else "", "```", "",
              "`manifest.json` in this folder carries the full recorded "
              "parameters and the capture_health result for each file.", ""]

    offsets = [e for e in entries if e.get("station_offset_hz")]
    if offsets:
        lines += ["## Recovering the offset captures", ""]
        for e in offsets:
            lines.append(
                f"- `{e['file']}`: station sits at "
                f"{e['station_offset_hz']/1e3:+g} kHz. Recover with "
                f"`--tune {e['station_offset_hz']:g}`.")
        lines.append("")
    (folder / "README.md").write_text("\n".join(lines), newline="\n")


def write_index(datasets, ref, det):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    fw = det["boards"][0].get("firmware") if det.get("boards") else "unknown"
    lines = [
        "# dev_data -- development capture corpus", "",
        f"Built by `tools/collect_dev_data.py` on {now}.", "",
        f"- station: **{STATION_NAME}** at {STATION_HZ/1e6:.3f} MHz",
        f"- quiet reference frequency: {QUIET_HZ/1e6:.3f} MHz",
        f"- device firmware: {fw}",
        f"- tools: {det.get('tools_version')}",
        f"- reference gain: LNA {ref['lna']} dB, VGA {ref['vga']} dB, "
        f"amp {'on' if ref.get('amp') else 'off'} "
        f"({ref['counts']:.0f}/128 counts)", "",
        "Every dataset varies one axis against the same station, so a test "
        "failure points at a specific property of the capture rather than at "
        "the recording session as a whole. Gain figures are offsets from the "
        "reference above, which was measured at collection time.", "",
        "| folder | what it varies | files |", "|---|---|---|",
    ]
    for n, ds, entries in datasets:
        lines.append(f"| `data_{n}_{ds['slug']}/` | {ds['title']} | "
                     f"{len(entries)} |")
    total = sum(e["size_bytes"] for _, _, es in datasets for e in es)
    lines += ["", f"Total: {total/1e6:.1f} MB across "
              f"{sum(len(es) for _,_,es in datasets)} captures.", "",
              "Each folder has its own README and a `manifest.json` with the "
              "full parameters and health result per file.", "",
              "## What to commit", "",
              "This whole corpus is larger than a repository wants to carry. "
              "The intent is that it stays local as a working set, and a "
              "chosen subset is copied into `sample_data/` for the examples "
              "and the test suite. Reasonable picks, smallest first:", "",
              "- `data_1/fm_reference.iq` -- replaces the current "
              "`sample_data/fm_2Msps.iq`",
              "- `data_4/fm_gain_low.iq` -- a real bad capture, so the "
              "health-check tests stop relying on synthetic noise",
              "- `data_5/quiet_channel.iq` -- the known-negative",
              "- one of `data_2`'s offset captures -- gives the tuning path "
              "real data", "",
              "The rate ladder and the 20 Msps wideband file are the ones to "
              "leave out of git; they are useful to have on disk while "
              "developing and expensive to carry forever.", ""]
    (OUT_DIR / "README.md").write_text("\n".join(lines), newline="\n")


def main():
    p = argparse.ArgumentParser(
        description="Collect the sdr_dsp development corpus (READ-ONLY).")
    p.add_argument("--only", nargs="*", type=int, default=None,
                   help="dataset numbers to collect, e.g. --only 1 3")
    p.add_argument("--plan", action="store_true",
                   help="print the plan and size budget, record nothing")
    p.add_argument("--calibrate", action="store_true",
                   help="run gain calibration and stop")
    p.add_argument("--clean", action="store_true",
                   help="delete existing dev_data/ first")
    p.add_argument("--station", type=float, default=None,
                   help="station Hz; overrides STATION_HZ in the CONFIG "
                        "block (tools/preflight_collection.py suggests one)")
    p.add_argument("--station-name", default=None,
                   help="label for the station; defaults to e.g. '98.5 FM'")
    p.add_argument("--quiet", type=float, default=None,
                   help="quiet reference Hz; overrides QUIET_HZ (preflight "
                        "verifies one)")
    args = p.parse_args()

    global STATION_HZ, STATION_NAME, QUIET_HZ
    if args.station is not None:
        STATION_HZ = int(args.station)
        STATION_NAME = (args.station_name
                        or f"{STATION_HZ / 1e6:g} FM")
    elif args.station_name is not None:
        STATION_NAME = args.station_name
    if args.quiet is not None:
        QUIET_HZ = int(args.quiet)

    plan = build_plan()
    only = set(args.only) if args.only else None
    rows, total = plan_bytes(plan, only)

    print(f"station : {STATION_NAME} at {STATION_HZ/1e6:.3f} MHz")
    print(f"quiet   : {QUIET_HZ/1e6:.3f} MHz")
    print(f"output  : {OUT_DIR}\n")
    print("plan:")
    for n, slug, count, sub in rows:
        print(f"  data_{n}_{slug:<10s} {count} captures  {sub/1e6:7.1f} MB")
    print(f"  {'total':<21s} {'':13s}{total/1e6:7.1f} MB")

    if args.plan:
        return 0
    if total / 1e6 > MAX_TOTAL_MB:
        print(f"\n! plan exceeds MAX_TOTAL_MB ({MAX_TOTAL_MB} MB).")
        if input("  continue anyway? [y/N] ").strip().lower() != "y":
            return 1

    if args.clean and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h = HackRF(tools_dir=TOOLS_DIR, verbose=False)
    print("\n== confirming a real board ==")
    det = h.detect()
    if not det["ready"]:
        print(f"   NO USABLE HACKRF: {det['problem']}", file=sys.stderr)
        return 1
    print(f"   ready: firmware {det['boards'][0].get('firmware')}")
    for w in det.get("warnings", []):
        print(f"   ! {w}")

    print(f"\n== calibrating gain on {STATION_NAME} ==")
    ref = calibrate_gain(h, 2e6)
    print(f"   reference: lna={ref['lna']} vga={ref['vga']} "
          f"amp={'on' if ref['amp'] else 'off'} "
          f"({ref['counts']:.0f}/128 counts, status={ref['status']})")
    if ref["status"] == "too_weak":
        print("   ! no station was measured: either the level never reached")
        print("     the window at maximum gain, or the level converged but")
        print("     no 19 kHz pilot was found (amplified noise, not signal).")
        print("     Check the antenna, or pick a stronger station -- the whole")
        print("     corpus is built from this signal.", file=sys.stderr)
        if input("   continue anyway? [y/N] ").strip().lower() != "y":
            return 1
    elif ref["status"] == "clipping":
        print("   ! still clipping at minimum gain; add attenuation.",
              file=sys.stderr)
    if args.calibrate:
        return 0

    collected = []
    try:
        for n, ds in sorted(plan.items()):
            if only and n not in only:
                continue
            folder = OUT_DIR / f"data_{n}_{ds['slug']}"
            folder.mkdir(parents=True, exist_ok=True)
            print(f"\n== data_{n}: {ds['title']} ==")
            entries = []
            for cap in ds["captures"]:
                e = record_one(h, folder, cap, ref, n)
                if e:
                    entries.append(e)

            if ds.get("sweep"):
                lo, hi = ds["sweep"]
                print(f"   sweep {lo/1e6:g}-{hi/1e6:g} MHz ...")
                try:
                    rowsx = h.sweep_collect(lo, hi, num_sweeps=2)
                    sp = folder / "fm_band_sweep.csv"
                    with open(sp, "w", newline="\n") as f:
                        f.write("date,time,hz_low,hz_high,bin_width,"
                                "num_samples,db...\n")
                        for r in rowsx:
                            f.write(", ".join(
                                [r["date"], r["time"], str(r["hz_low"]),
                                 str(r["hz_high"]), f"{r['bin_width']:.2f}",
                                 str(r["num_samples"])]
                                + [f"{d:.2f}" for d in r["db"]]) + "\n")
                    print(f"   wrote {sp.name} ({len(rowsx)} rows)")
                except HackRFError as e:
                    print(f"   sweep failed: {e}", file=sys.stderr)

            if entries:
                (folder / "manifest.json").write_text(json.dumps({
                    "dataset": n, "slug": ds["slug"], "title": ds["title"],
                    "why": ds["why"],
                    "station_name": STATION_NAME, "station_hz": STATION_HZ,
                    "quiet_hz": QUIET_HZ,
                    "collected": datetime.datetime.now().isoformat(
                        timespec="seconds"),
                    "reference_gain": ref,
                    "captures": entries,
                }, indent=2), newline="\n")
                write_dataset_readme(folder, n, ds, entries, ref)
                collected.append((n, ds, entries))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    if not collected:
        print("\n== nothing collected ==", file=sys.stderr)
        return 1

    write_index(collected, ref, det)
    size = sum(e["size_bytes"] for _, _, es in collected for e in es)
    n_files = sum(len(es) for _, _, es in collected)
    print(f"\n== done: {n_files} captures, {size/1e6:.1f} MB in "
          f"{OUT_DIR.name}/ ==")

    problems = [(n, e) for n, _, es in collected for e in es
                if not e["verdict"].startswith("ok")]
    if problems:
        print(f"\n{len(problems)} capture(s) did not match expectations:")
        for n, e in problems:
            print(f"   data_{n}/{e['file']}: {e['verdict']}")
        print("   Re-run those datasets with --only after checking the setup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
