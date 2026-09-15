#! /usr/bin/python3
"""Promote selected captures from `dev_data/` into `sample_data/`.

The two directories serve different purposes and should not be confused:

    dev_data/     the full working corpus from tools/collect_dev_data.py.
                  Large, varied, gitignored. Every axis we might want to test
                  against, kept on the machine that recorded it.

    sample_data/  the small committed subset the examples and the test suite
                  actually run against. Anyone cloning the repo gets these.

This script moves files from the first to the second, and -- the part that
matters -- carries their provenance with them. Each promoted capture keeps
its SigMF sidecar and gains an entry in `sample_data/manifest.json` recording
the gain it was taken at, the station it came from, and the capture_health
result measured at collection time AND re-measured after the copy.

That last point is the whole reason this is a script rather than a `cp`. The
capture that started all of this shipped because nobody checked what was in
it; a hand-copied file loses the one record that would have caught it. Here,
a promoted file that fails its own declared expectation stops the promotion.

    python tools/promote_to_sample_data.py --list      # what's available
    python tools/promote_to_sample_data.py --dry-run   # what would happen
    python tools/promote_to_sample_data.py             # do it
    python tools/promote_to_sample_data.py --all-from 2

Note that promoting `fm_reference` overwrites `sample_data/fm_2Msps.iq`,
which is the capture the FM receiver example defaults to. That is the point:
that file currently contains nothing but noise.
"""

# ===========================================================================
#  WHAT GETS PROMOTED -- edit this list
# ===========================================================================
# Each entry: (dataset_number, capture_name, destination_stem, why)
#
# Keep this small. sample_data/ is carried by every clone forever, so the
# bar is "an example or a test genuinely needs it", not "it's interesting".
# The rate ladder and the 20 Msps wideband file are deliberately absent:
# useful to have on disk while developing, expensive to carry in git.

PROMOTE = [
    (1, "fm_reference", "fm_2Msps",
     "The default capture for examples/fm_receiver.py. Replaces the "
     "noise-floor file that shipped previously."),

    (2, "fm_offset_p250k", "fm_offset_250k",
     "Station at +250 kHz rather than at DC, so the tuning path has real "
     "data behind it. Recover with --tune 250e3."),

    (4, "fm_gain_low", "fm_weak",
     "A genuinely bad capture, recorded on purpose. Gives the capture_health "
     "tests real material instead of synthetic noise, and reproduces the "
     "original failure as a permanent fixture."),

    (5, "quiet_channel", "quiet_2Msps",
     "Empty frequency at normal gain: the known-negative for no-signal code "
     "paths and the noise reference for calibration."),
]

# ===========================================================================

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from sdr_dsp.core import capture_health, frequency_shift
from sdr_dsp.io import load_iq

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEV_DIR = PROJECT / "dev_data"
SAMPLE_DIR = PROJECT / "sample_data"

FM_CHANNEL_BW = 100_000


def load_manifests(dev_dir=None):
    """Read every data_N/manifest.json. Returns {dataset_no: manifest}."""
    dev_dir = Path(dev_dir) if dev_dir else DEV_DIR
    out = {}
    if not dev_dir.exists():
        return out
    for folder in sorted(dev_dir.glob("data_*")):
        mf = folder / "manifest.json"
        if mf.exists():
            data = json.loads(mf.read_text())
            data["_folder"] = folder
            out[data["dataset"]] = data
    return out


def find_capture(manifests, dataset_no, name):
    """Locate one capture entry. Returns (manifest, entry) or (None, None)."""
    mani = manifests.get(dataset_no)
    if not mani:
        return None, None
    for entry in mani["captures"]:
        if Path(entry["file"]).stem == name:
            return mani, entry
    return mani, None


def revalidate(path, entry):
    """Re-run capture_health on the copied file. Returns (health, verdict).

    Checks the channel where the station actually sits, mirroring the
    collector: an intentionally-offset capture would otherwise look empty
    simply because we tuned away from the station when recording it.
    """
    iq, _ = load_iq(str(path))
    rate = entry["sample_rate"]
    offset = entry.get("station_offset_hz") or 0
    probe = frequency_shift(iq, -offset, rate) if offset else iq
    health = capture_health(probe, rate, channel_bw=FM_CHANNEL_BW)

    expect = entry.get("expect", "signal")
    if expect == "signal":
        ok = health["ok"]
        verdict = "contains signal" if ok else "EMPTY -- should contain signal"
    else:                                   # weak / empty fixtures
        ok = not health["ok"]
        verdict = (f"intentionally {expect}" if ok
                   else f"UNEXPECTEDLY USABLE -- declared {expect}")
    return health, verdict, ok


def write_sample_readme(promoted, manifests):
    """Regenerate sample_data/README.md from the promoted entries."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    any_mani = next(iter(manifests.values()), {})
    station = any_mani.get("station_name", "unknown")
    station_hz = any_mani.get("station_hz")

    lines = [
        "# sample data", "",
        "Real recordings from a HackRF One, so the examples have something",
        "genuine to run against without owning a board.", "",
        f"- promoted from `dev_data/` on {now}",
        f"- source station: {station}"
        + (f" at {station_hz/1e6:.3f} MHz" if station_hz else ""),
        "- collected by `tools/collect_dev_data.py`, promoted by "
        "`tools/promote_to_sample_data.py`", "",
        "Each `.iq` is interleaved int8 I/Q (HackRF native) with a",
        "`.sigmf-meta` sidecar describing frequency, rate, and gains.",
        "Load with:", "",
        "```python",
        "from sdr_dsp.io import load_iq, read_meta", "",
        'iq, meta = load_iq("fm_2Msps.iq")   # complex64, normalized to +/-1',
        'meta = read_meta("fm_2Msps.iq")     # sidecar only, no samples read',
        "```", "",
        "To see what is actually in one:", "",
        "```bash",
        "uv run python examples/inspect_capture.py sample_data/fm_2Msps.iq",
        "```", "",
        "Every file below was validated with `capture_health` when recorded",
        "and again when promoted. `manifest.json` carries the full recorded",
        "parameters and both health results.", "",
        "## Files", "",
    ]
    for p in promoted:
        h = p["health_now"]
        excess = ("n/a" if h["channel_excess_db"] is None
                  else f"{h['channel_excess_db']:+.1f} dB")
        lines += [
            f"### `{p['dest']}.iq`", "",
            f"{p['why']}", "",
            f"- {p['frequency_hz']/1e6:.3f} MHz, "
            f"{p['sample_rate']/1e6:g} Msps, {p['seconds']:g}s",
            f"- gain: LNA {p['lna_db']} dB, VGA {p['vga_db']} dB",
            f"- peak level: {h['adc_counts']:.1f} of 127 ADC counts",
            f"- channel vs band edges: {excess}",
            f"- **{p['verdict']}**",
        ]
        if p.get("station_offset_hz"):
            lines.append(
                f"- station sits at {p['station_offset_hz']/1e3:+g} kHz; "
                f"recover with `--tune {p['station_offset_hz']:g}`")
        lines.append("")

    weak = [p for p in promoted if p["expect"] != "signal"]
    if weak:
        lines += [
            "## A note on the deliberately-empty files", "",
            "Some captures here contain no usable signal **on purpose**. They",
            "are fixtures for the no-signal code paths, and the test suite",
            "asserts that `capture_health` rejects them. If one of these ever",
            "starts passing the health check, that is the bug -- not the",
            "other way round.", "",
        ]
        for p in weak:
            lines.append(f"- `{p['dest']}.iq` -- {p['verdict']}")
        lines.append("")

    (SAMPLE_DIR / "README.md").write_text("\n".join(lines), newline="\n")


def main():
    ap = argparse.ArgumentParser(
        description="Promote dev_data captures into sample_data.")
    ap.add_argument("--dev-dir", default=None,
                    help="override the dev_data location")
    ap.add_argument("--list", action="store_true",
                    help="show what is available to promote and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied, write nothing")
    ap.add_argument("--all-from", type=int, default=None, metavar="N",
                    help="promote every capture in dataset N instead of the "
                         "curated PROMOTE list")
    ap.add_argument("--force", action="store_true",
                    help="promote even if a capture fails re-validation")
    args = ap.parse_args()

    dev_dir = Path(args.dev_dir) if args.dev_dir else DEV_DIR
    manifests = load_manifests(dev_dir)
    if not manifests:
        print(f"no manifests under {dev_dir}. Run tools/collect_dev_data.py "
              f"first.", file=sys.stderr)
        return 1

    if args.list:
        print(f"available in {dev_dir}:\n")
        for n, mani in sorted(manifests.items()):
            print(f"  data_{n} -- {mani['title']}")
            for e in mani["captures"]:
                stem = Path(e["file"]).stem
                h = e["health"]
                print(f"      {stem:<22s} {e['sample_rate']/1e6:>5g} Msps  "
                      f"{h['adc_counts']:>6.1f}/127  {e['verdict']}")
            print()
        chosen = {(d, n) for d, n, _, _ in PROMOTE}
        print("curated PROMOTE list:")
        for d, n, dest, _ in PROMOTE:
            mark = "ok" if (d, n) in chosen and d in manifests else "MISSING"
            print(f"  data_{d}/{n} -> sample_data/{dest}.iq   [{mark}]")
        return 0

    # build the work list
    if args.all_from is not None:
        mani = manifests.get(args.all_from)
        if not mani:
            print(f"no data_{args.all_from} in {dev_dir}", file=sys.stderr)
            return 1
        work = [(args.all_from, Path(e["file"]).stem, Path(e["file"]).stem,
                 e["purpose"]) for e in mani["captures"]]
    else:
        work = PROMOTE

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    promoted, failures = [], []

    for dataset_no, name, dest, why in work:
        mani, entry = find_capture(manifests, dataset_no, name)
        if not mani:
            print(f"!! data_{dataset_no} not collected -- skipping {name}",
                  file=sys.stderr)
            failures.append((name, "dataset missing"))
            continue
        if not entry:
            print(f"!! {name} not found in data_{dataset_no} -- skipping",
                  file=sys.stderr)
            failures.append((name, "capture missing"))
            continue

        src = mani["_folder"] / entry["file"]
        src_meta = mani["_folder"] / entry["sigmf"]
        dst = SAMPLE_DIR / f"{dest}.iq"
        dst_meta = SAMPLE_DIR / f"{dest}.sigmf-meta"

        size_mb = entry["size_bytes"] / 1e6
        print(f"data_{dataset_no}/{name} -> sample_data/{dest}.iq "
              f"({size_mb:.1f} MB)")

        if args.dry_run:
            print("   (dry run, nothing written)")
            continue

        shutil.copy2(src, dst)
        if src_meta.exists():
            shutil.copy2(src_meta, dst_meta)
        else:
            print(f"   ! no SigMF sidecar alongside {src.name}; the capture "
                  f"will not carry its rate or frequency", file=sys.stderr)

        health, verdict, ok = revalidate(dst, entry)
        excess = health["channel_excess_db"]
        print(f"   {health['adc_counts']:.1f}/127 counts"
              + (f", channel {excess:+.1f} dB" if excess is not None else "")
              + f" -- {verdict}")

        if not ok and not args.force:
            dst.unlink(missing_ok=True)
            dst_meta.unlink(missing_ok=True)
            print(f"   REJECTED, not promoted. Re-record data_{dataset_no}, "
                  f"or pass --force if this is intentional.", file=sys.stderr)
            failures.append((name, verdict))
            continue

        promoted.append({
            "dest": dest, "why": why, "source":
                f"dev_data/{mani['_folder'].name}/{entry['file']}",
            "frequency_hz": entry["frequency_hz"],
            "station_offset_hz": entry.get("station_offset_hz"),
            "sample_rate": entry["sample_rate"], "seconds": entry["seconds"],
            "lna_db": entry["lna_db"], "vga_db": entry["vga_db"],
            "expect": entry.get("expect", "signal"),
            "health_at_capture": entry["health"],
            "health_now": {
                "ok": health["ok"],
                "adc_counts": round(health["adc_counts"], 2),
                "channel_excess_db": (None if excess is None
                                      else round(excess, 2)),
                "reasons": health["reasons"],
            },
            "verdict": verdict,
        })

    if args.dry_run:
        print(f"\ndry run: {len(work)} capture(s) would be promoted")
        return 0

    if promoted:
        (SAMPLE_DIR / "manifest.json").write_text(json.dumps({
            "promoted": datetime.datetime.now().isoformat(timespec="seconds"),
            "promoted_by": "tools/promote_to_sample_data.py",
            "captures": promoted,
        }, indent=2), newline="\n")
        write_sample_readme(promoted, manifests)
        total = sum(Path(SAMPLE_DIR / f"{p['dest']}.iq").stat().st_size
                    for p in promoted) / 1e6
        print(f"\n== promoted {len(promoted)} capture(s), {total:.1f} MB "
              f"into sample_data/ ==")
        print("   wrote manifest.json and README.md")

    if failures:
        print(f"\n{len(failures)} not promoted:", file=sys.stderr)
        for name, why in failures:
            print(f"   {name}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
