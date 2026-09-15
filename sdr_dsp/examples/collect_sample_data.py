#! /usr/bin/python3
"""Collect sdr_dsp sample captures from a HackRF One. STRICTLY READ-ONLY.

Records real IQ into `sample_data/` with SigMF sidecars, so the examples have
something genuine to run against. Receive and sweep only -- this never
transmits and never touches firmware.

Why this exists in this form
----------------------------
The sample capture originally shipped with sdr_dsp was recorded at the noise
floor: it used about 3 of 127 available ADC counts, and every sample in the
2 MB file was one of {-2, -1, 0, 1, 2}. The FM receiver ran on it perfectly
and produced a WAV containing nothing but hiss, because there was no signal
in the file to recover. The recording tool reported a summary at the time,
but nothing checked whether the numbers were any good.

So this script does two things the previous one did not:

1. **Searches for a working gain before recording.** It takes short test
   captures into RAM and adjusts LNA/VGA until the signal occupies a healthy
   fraction of the converter's range -- high enough to be clear of the noise
   floor, low enough not to clip.

2. **Validates every capture after writing it,** via
   `sdr_dsp.core.capture_health`, the same check the FM receiver runs. A
   capture that fails is reported loudly and, unless you pass --keep-failed,
   deleted rather than left to be committed and puzzled over later.

Requires hackrfpy and the hackrf-tools binaries at the OS level:

    uv sync --extra examples-hackrf

Usage:
    uv run python examples/collect_sample_data.py
    uv run python examples/collect_sample_data.py --band fm --band ism433
    uv run python examples/collect_sample_data.py --band all --seconds 1.0
    uv run python examples/collect_sample_data.py --list
    uv run python examples/collect_sample_data.py --band fm --tune-only

Legality: every preset here is receive-only, on bands that are lawful to
listen to in most jurisdictions. Rules differ by country -- you are
responsible for what you tune to. Nothing here transmits.
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

from sdr_dsp.core import capture_health

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
OUT_DIR = HERE.parent / "sample_data"

# HackRF gain steps (hackrfpy snaps to these, but searching on-grid avoids
# silently re-testing a gain we already tried).
LNA_STEPS = list(range(0, 41, 8))     # 0..40 dB in 8 dB steps
VGA_STEPS = list(range(0, 63, 2))     # 0..62 dB in 2 dB steps

# Target window for peak ADC utilization, in counts out of 127. Below the
# floor there is nothing to demodulate; above the ceiling the converter
# clips and the phase information is destroyed.
TARGET_LO, TARGET_HI = 45.0, 110.0
CLIP_COUNTS = 120.0


# Presets chosen to exercise sdr_dsp's demodulators against real signals.
# `channel_bw` is the half-width used for the carrier-presence check, and
# should match roughly what a receiver would filter down to.
BANDS = {
    "fm": {
        "center": 98_000_000, "sweep": (88_000_000, 108_000_000),
        "channel_bw": 100_000, "sample_rate": 2e6, "seconds": 0.5,
        "desc": "FM broadcast -- wideband FM, exercises fm_receiver.py",
        "expect_signal": True,
        "note": "Pick a strong local station; 98.0 MHz is only a placeholder.",
    },
    "wx": {
        "center": 162_450_000, "sweep": (162_375_000, 162_575_000),
        "channel_bw": 12_500, "sample_rate": 2e6, "seconds": 1.0,
        "desc": "NOAA weather radio -- narrowband FM voice, always on",
        "expect_signal": True,
        "note": ("Continuous broadcast, so unlike airband it is reliably "
                 "present. Good narrowband-FM counterpart to the wideband "
                 "capture. US channels: 162.400-162.550 MHz."),
    },
    "airband": {
        "center": 124_000_000, "sweep": (118_000_000, 137_000_000),
        "channel_bw": 8_000, "sample_rate": 2e6, "seconds": 2.0,
        "desc": "VHF airband -- AM voice, exercises am_receiver.py",
        "expect_signal": False,
        "note": ("Bursty: only occupied while an aircraft or tower is "
                 "actually talking, so a capture may legitimately be empty. "
                 "Carrier check is advisory here. Longer --seconds helps."),
    },
    "ism433": {
        "center": 433_920_000, "sweep": (433_000_000, 435_000_000),
        "channel_bw": 200_000, "sample_rate": 2e6, "seconds": 2.0,
        "desc": "433 MHz ISM -- OOK/ASK devices, exercises ook_decoder.py",
        "expect_signal": False,
        "note": ("Short bursts from sensors, remotes, and TPMS. Empty "
                 "captures are normal; trigger a device during the window "
                 "if you have one. Also good for burst_detector.py."),
    },
    "ism915": {
        "center": 915_000_000, "sweep": (902_000_000, 928_000_000),
        "channel_bw": 500_000, "sample_rate": 2e6, "seconds": 2.0,
        "desc": "915 MHz ISM -- frequency hoppers, exercises fhss_visualizer",
        "expect_signal": False,
        "note": "US band. Hopping traffic is intermittent by nature.",
    },
    "noise": {
        "center": 250_000_000, "sweep": None,
        "channel_bw": 100_000, "sample_rate": 2e6, "seconds": 0.5,
        "desc": "deliberate noise-floor reference (no signal expected)",
        "expect_signal": False,
        "skip_gain_search": True,
        "note": ("Intentionally empty: a known-negative for testing "
                 "capture_health, calibration, and the no-signal paths. "
                 "Recorded at fixed mid-range gain with the antenna "
                 "disconnected. This is the one capture that SHOULD fail "
                 "the carrier check."),
    },
}


def _counts(iq):
    """Peak ADC utilization in counts out of 127, from normalized samples."""
    if len(iq) == 0:
        return 0.0
    return float(max(np.max(np.abs(iq.real)), np.max(np.abs(iq.imag)))) * 127.0


def find_gain(h, band, args):
    """Search LNA/VGA for a level in the target window.

    Returns (lna, vga, tried, status). status is one of:
        "ok"        -- landed inside the target window
        "too_weak"  -- still below the window at maximum gain; the signal
                       isn't reaching the radio (antenna, cabling, or there
                       is genuinely nothing on this frequency)
        "clipping"  -- still at or above the clipping point at minimum gain;
                       the signal is too strong and needs external attenuation

    Walks from a conservative starting point using short in-RAM captures.
    VGA moves first because its 2 dB steps are finer than the LNA's 8 dB, so
    it lands inside the window more often without overshooting into clipping.
    """
    probe_n = int(args.sample_rate * 0.05)      # 50 ms is plenty to judge level
    lna, vga = 16, 20
    tried = []

    for _ in range(14):
        iq = h.capture_array(band["center"], args.sample_rate, probe_n,
                             lna=lna, vga=vga, amp=False)
        c = _counts(iq)
        tried.append((lna, vga, c))
        print(f"    probe lna={lna:2d} vga={vga:2d} -> peak {c:6.1f} counts")

        if TARGET_LO <= c <= TARGET_HI:
            return lna, vga, tried, "ok"

        if c >= CLIP_COUNTS:                     # clipping: back off hard
            if vga > VGA_STEPS[0]:
                vga = max(VGA_STEPS[0], vga - 6)
            elif lna > LNA_STEPS[0]:
                lna -= 8
                vga = VGA_STEPS[0]
            else:
                return lna, vga, tried, "clipping"   # already at minimum
            continue

        if c < TARGET_LO:                        # too quiet: add gain
            if vga < VGA_STEPS[-1]:
                # step proportionally to how far below target we are
                deficit_db = 20 * np.log10(max(TARGET_LO, 1.0) / max(c, 0.5))
                vga = min(VGA_STEPS[-1], vga + max(2, int(deficit_db // 2) * 2))
            elif lna < LNA_STEPS[-1]:
                lna += 8
                vga = 20
            else:
                return lna, vga, tried, "too_weak"  # already at maximum
        else:                                    # above window, below clipping
            if vga > VGA_STEPS[0]:
                vga = max(VGA_STEPS[0], vga - 2)
            elif lna > LNA_STEPS[0]:
                lna -= 8
                vga = VGA_STEPS[0]
            else:
                return lna, vga, tried, "clipping"

    # ran out of probes: fall back to the best non-clipping level seen
    usable = [t for t in tried if t[2] < CLIP_COUNTS]
    best = max(usable, key=lambda t: t[2]) if usable else min(
        tried, key=lambda t: t[2])
    return best[0], best[1], tried, "ok" if usable else "clipping"


def collect_band(h, name, args):
    """Record one band. Returns a result dict, or None if nothing was kept."""
    band = BANDS[name]
    sr = args.sample_rate or band["sample_rate"]
    secs = args.seconds or band["seconds"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n== {name}: {band['desc']} ==")
    print(f"   {band['center']/1e6:g} MHz, {sr/1e6:g} Msps, {secs:g}s")
    if band.get("note"):
        print(f"   note: {band['note']}")

    # ---- gain search ----
    if band.get("skip_gain_search"):
        lna, vga, tried = 16, 20, []
        print("   fixed gain (noise reference): lna=16 vga=20")
    else:
        print("   searching for working gain ...")
        try:
            lna, vga, tried, status = find_gain(h, band, argparse.Namespace(
                sample_rate=sr))
        except HackRFError as e:
            print(f"   gain search failed: {e}", file=sys.stderr)
            return None
        print(f"   chose lna={lna} vga={vga}")
        if status == "too_weak":
            print("   ! still below the target level at maximum gain. Either")
            print("     nothing is transmitting here, or the signal isn't")
            print("     reaching the radio -- check the antenna and cabling.")
        elif status == "clipping":
            print("   ! still clipping at minimum gain. The signal is too")
            print("     strong; add external attenuation, or the capture will")
            print("     be distorted and the phase information destroyed.")
        if args.tune_only:
            return {"name": name, "tuned_only": True, "lna": lna, "vga": vga}

    # ---- the real capture ----
    iq_path = OUT_DIR / f"{name}_{int(sr/1e6)}Msps.iq"
    n = int(sr * secs)
    try:
        h.capture(band["center"], sr, num_samples=n, out=str(iq_path),
                  lna=lna, vga=vga, amp=False, sigmf=True)
    except HackRFError as e:
        print(f"   capture failed: {e}", file=sys.stderr)
        return None

    # ---- validate what we actually got ----
    iq = load_iq(str(iq_path))
    health = capture_health(iq, sr, channel_bw=band["channel_bw"])
    size_mb = iq_path.stat().st_size / 1e6
    print(f"   wrote {iq_path.name} ({size_mb:.1f} MB, {len(iq):,} samples)")
    print(f"   peak {health['adc_counts']:.1f}/127 counts, channel "
          f"{health['channel_excess_db']:+.1f} dB above band edges"
          if health["channel_excess_db"] is not None else
          f"   peak {health['adc_counts']:.1f}/127 counts")

    expected = band["expect_signal"]
    if health["ok"]:
        print("   OK -- capture contains signal")
    elif not expected:
        # Bursty or deliberately-empty bands: not a failure, just note it.
        print("   empty, which is expected for this band:")
        for r in health["reasons"]:
            print(f"     - {r}")
    else:
        print("   FAILED -- this capture has nothing usable in it:",
              file=sys.stderr)
        for r in health["reasons"]:
            print(f"     - {r}", file=sys.stderr)
        if args.keep_failed:
            print("   keeping it anyway (--keep-failed)")
        else:
            iq_path.unlink(missing_ok=True)
            Path(str(iq_path).rsplit(".", 1)[0] + ".sigmf-meta").unlink(
                missing_ok=True)
            print("   deleted; re-run with an antenna connected, or pick a "
                  "frequency with a strong local signal")
            return None

    result = {"name": name, "path": iq_path, "lna": lna, "vga": vga,
              "sample_rate": sr, "seconds": secs, "health": health,
              "center": band["center"], "desc": band["desc"],
              "expected": expected, "probes": tried}

    # ---- optional sweep alongside ----
    if not args.no_sweep and band.get("sweep"):
        lo, hi = band["sweep"]
        sweep_path = OUT_DIR / f"{name}_sweep.csv"
        print(f"   sweep {lo/1e6:g}-{hi/1e6:g} MHz ...")
        try:
            rows = h.sweep_collect(lo, hi, num_sweeps=args.sweep_count)
            with open(sweep_path, "w", newline="\n") as f:
                f.write("date,time,hz_low,hz_high,bin_width,num_samples,db...\n")
                for r in rows:
                    f.write(", ".join(
                        [r["date"], r["time"], str(r["hz_low"]),
                         str(r["hz_high"]), f"{r['bin_width']:.2f}",
                         str(r["num_samples"])]
                        + [f"{d:.2f}" for d in r["db"]]) + "\n")
            print(f"   wrote {sweep_path.name} ({len(rows)} rows)")
            result["sweep"] = sweep_path
        except HackRFError as e:
            print(f"   sweep failed: {e}", file=sys.stderr)

    return result


def write_readme(results, det):
    """Regenerate sample_data/README.md so the data documents itself."""
    path = OUT_DIR / "README.md"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    fw = det["boards"][0].get("firmware") if det.get("boards") else "unknown"

    lines = [
        "# sample data", "",
        "Real recordings from a HackRF One, so the examples have something",
        "genuine to run against without owning a board.", "",
        f"- collected: {now}",
        f"- device firmware: {fw}",
        f"- tools: {det.get('tools_version')}",
        "- collected by: `examples/collect_sample_data.py`", "",
        "Each `.iq` is interleaved int8 I/Q (HackRF native) with a",
        "`.sigmf-meta` sidecar describing frequency, rate, and gains.",
        "Load with:", "",
        "```python",
        "from sdr_dsp.io import load_iq, read_meta", "",
        'iq, meta = load_iq("fm_2Msps.iq")   # complex64, normalized to +/-1',
        'meta = read_meta("fm_2Msps.iq")     # sidecar only, no samples read',
        "```", "",
        "Every capture below was validated with `capture_health` at record",
        "time; the measured figures are reported as recorded.", "",
        "## Files", "",
    ]
    for r in results:
        if r.get("tuned_only"):
            continue
        h = r["health"]
        excess = ("n/a" if h["channel_excess_db"] is None
                  else f"{h['channel_excess_db']:+.1f} dB")
        verdict = ("contains signal" if h["ok"]
                   else "empty (expected for this band)" if not r["expected"]
                   else "EMPTY -- needs re-recording")
        lines += [
            f"### `{r['path'].name}`", "",
            f"- {r['desc']}",
            f"- {r['center']/1e6:g} MHz, {r['sample_rate']/1e6:g} Msps, "
            f"{r['seconds']:g}s",
            f"- gain: LNA {r['lna']} dB, VGA {r['vga']} dB, amp off",
            f"- peak level: {h['adc_counts']:.1f} of 127 ADC counts",
            f"- channel vs band edges: {excess}",
            f"- **{verdict}**", "",
        ]
        if r.get("sweep"):
            lines += [f"  Sweep data: `{r['sweep'].name}`", ""]

    path.write_text("\n".join(lines), newline="\n")
    print(f"\n   wrote {path.relative_to(OUT_DIR.parent)}")


def main():
    p = argparse.ArgumentParser(
        description="Collect sdr_dsp sample captures from a HackRF "
                    "(READ-ONLY: never transmits).")
    p.add_argument("--band", action="append", choices=list(BANDS) + ["all"],
                   help="band(s) to collect; repeatable. Default: fm")
    p.add_argument("--tools-dir", default=None,
                   help="path to hackrf-tools binaries if not on PATH")
    p.add_argument("--seconds", type=float, default=None,
                   help="override per-band capture duration")
    p.add_argument("--sample-rate", type=float, default=None,
                   help="override per-band sample rate (sps)")
    p.add_argument("--sweep-count", type=int, default=1)
    p.add_argument("--no-sweep", action="store_true",
                   help="IQ captures only, skip sweep datasets")
    p.add_argument("--keep-failed", action="store_true",
                   help="keep captures that fail the health check")
    p.add_argument("--tune-only", action="store_true",
                   help="run the gain search and report, record nothing")
    p.add_argument("--list", action="store_true",
                   help="describe the band presets and exit")
    args = p.parse_args()

    if args.list:
        print("band presets:\n")
        for name, b in BANDS.items():
            print(f"  {name:9s} {b['center']/1e6:>9.3f} MHz  {b['desc']}")
            print(f"  {'':9s} {'':>9s}       signal expected: "
                  f"{'yes' if b['expect_signal'] else 'no'}")
            if b.get("note"):
                print(f"  {'':9s} {'':>9s}       {b['note']}")
            print()
        return 0

    bands = args.band or ["fm"]
    if "all" in bands:
        bands = list(BANDS)

    h = HackRF(tools_dir=args.tools_dir, verbose=False)
    print("== confirming a real board before collecting ==")
    det = h.detect()
    if not det["ready"]:
        print(f"   NO USABLE HACKRF: {det['problem']}", file=sys.stderr)
        return 1
    print(f"   ready: firmware {det['boards'][0].get('firmware')}")
    for w in det.get("warnings", []):
        print(f"   ! {w}")

    results = []
    try:
        for name in bands:
            r = collect_band(h, name, args)
            if r:
                results.append(r)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    kept = [r for r in results if not r.get("tuned_only")]
    if kept:
        write_readme(kept, det)
        total = sum(r["path"].stat().st_size for r in kept) / 1e6
        print(f"\n== done: {len(kept)} captures, ~{total:.1f} MB in "
              f"{OUT_DIR.name}/ ==")
        bad = [r for r in kept if not r["health"]["ok"] and r["expected"]]
        if bad:
            print(f"   {len(bad)} kept despite failing the health check "
                  f"(--keep-failed): " + ", ".join(r["name"] for r in bad))
    elif not args.tune_only:
        print("\n== nothing kept ==", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
