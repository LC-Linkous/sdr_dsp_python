#! /usr/bin/python3
"""Preflight for a data-collection session: is this setup ready to record?

Run this BEFORE `tools/collect_dev_data.py` or
`examples/collect_sample_data.py`. It answers, with measurements, the
questions that decide whether a session will produce a usable corpus or an
hour of hiss:

  1. Is a real board attached and healthy?
  2. Does the antenna receive ANYTHING? (whole-band sweep -- if this is
     flat, stop and fix the antenna; no software setting will help)
  3. Which station should the corpus be built from? Candidates are taken
     from the sweep, then each is auto-gained and scored by its measured
     19 kHz stereo pilot -- the check amplified noise cannot fake. Sweep
     power alone can mislead (a strong adjacent channel, an image); the
     pilot cannot.
  4. Is the quiet reference frequency actually quiet? The old default
     (87.7 MHz) is a guess, and is inside the broadcast band in ITU
     Region 1. This finds the emptiest spot in the sweep and verifies it
     with a capture that must FAIL the health check.

It ends with a PASS/FAIL verdict and, on PASS, the exact command to run:

    uv run python tools/collect_dev_data.py --station 98.5e6 --quiet 91.34e6

Usage:
    uv run python tools/preflight_collection.py
    uv run python tools/preflight_collection.py --region world   # 87.5-108
    uv run python tools/preflight_collection.py --candidates 5
    uv run python tools/preflight_collection.py --simulate       # no radio

--simulate synthesizes a band with stations in it and runs the whole flow
against it. Use it to confirm the tool works before blaming the hardware.

Requires hackrfpy + hackrf-tools for real use (not for --simulate).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from sdr_dsp.core import capture_health, fm_pilot_excess_db, search_gain

SAMPLE_RATE = 2_000_000
CHANNEL_BW = 100_000          # half-width used for the carrier check
PILOT_OK_DB = 6.0             # pilot excess above this = broadcast FM present
DETECT_THRESHOLD_DB = 6.0     # sweep prominence for a station candidate
MIN_STATION_SPACING_HZ = 300_000
QUIET_WIDTH_HZ = 400_000      # a quiet spot must be this wide
QUIET_MARGIN_DB = 3.0         # ...and within this of the band's noise floor

REGIONS = {
    "americas": (88.0e6, 108.0e6),    # ITU Region 2
    "world": (87.5e6, 108.0e6),       # ITU Regions 1 and 3 (most of)
}


# ---------------------------------------------------------------------------
# pure spectrum analysis (unit-testable; no hardware involved)
# ---------------------------------------------------------------------------
def spectrum_from_sweep(rows):
    """Flatten hackrfpy sweep_collect rows into (freqs_hz, power_db) arrays.

    Each row covers [hz_low, hz_high) in bins of bin_width with a dB value
    per bin. Rows may repeat frequencies across sweep passes; repeated bins
    are averaged (in dB -- these are already log-domain display values, and
    a display average is what a ranking needs).
    """
    acc = {}
    for r in rows:
        f0 = float(r["hz_low"])
        bw = float(r["bin_width"])
        for i, db in enumerate(r["db"]):
            f = f0 + (i + 0.5) * bw
            if f in acc:
                s, c = acc[f]
                acc[f] = (s + float(db), c + 1)
            else:
                acc[f] = (float(db), 1)
    freqs = np.array(sorted(acc))
    power = np.array([acc[f][0] / acc[f][1] for f in freqs])
    return freqs, power


def find_stations(freqs, power_db, threshold_db=DETECT_THRESHOLD_DB,
                  min_spacing_hz=MIN_STATION_SPACING_HZ):
    """Station candidates: local maxima standing above the noise floor.

    The floor is the band's median (most of an FM band is between stations).
    Returns a list of (freq_hz, prominence_db), strongest first. Peaks
    closer than min_spacing to a stronger peak are absorbed into it.
    """
    if len(freqs) == 0:
        return []
    floor = float(np.median(power_db))
    order = np.argsort(power_db)[::-1]
    picks = []
    for i in order:
        prom = float(power_db[i]) - floor
        if prom < threshold_db:
            break
        f = float(freqs[i])
        if any(abs(f - pf) < min_spacing_hz for pf, _ in picks):
            continue
        picks.append((f, prom))
    return picks


def find_quiet(freqs, power_db, stations, width_hz=QUIET_WIDTH_HZ,
               margin_db=QUIET_MARGIN_DB,
               min_spacing_hz=MIN_STATION_SPACING_HZ):
    """The emptiest spot in the sweep: a window of width_hz whose HIGHEST
    bin sits within margin_db of the band's noise floor, as far from every
    detected station as available. Returns center frequency Hz, or None.

    Judging the window by its maximum matters: an average would let a
    narrow carrier hide inside an otherwise quiet window, and the whole
    point of the quiet reference is that nothing is there.
    """
    if len(freqs) < 3:
        return None
    floor = float(np.median(power_db))
    step = float(np.median(np.diff(freqs)))
    half = max(1, int(width_hz / 2 / step))
    best = None                                   # (dist_to_station, -peak, f)
    for i in range(half, len(freqs) - half):
        peak = float(np.max(power_db[i - half:i + half + 1]))
        if peak > floor + margin_db:
            continue
        f = float(freqs[i])
        dist = min((abs(f - sf) for sf, _ in stations), default=np.inf)
        if dist < min_spacing_hz + width_hz / 2:
            continue
        key = (min(dist, 2e6), -peak)             # cap: "far enough" is enough
        if best is None or key > best[:2]:
            best = (*key, f)
    return None if best is None else best[2]


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
def score_station(h, freq_hz):
    """Auto-gain at freq_hz and measure what is really there.

    Returns a dict with the chosen gain, peak counts, channel excess, and
    the pilot excess -- the number that decides. None on capture failure.
    """
    def probe(lna, vga, amp):
        return h.capture_array(freq_hz, SAMPLE_RATE,
                               int(SAMPLE_RATE * 0.05),
                               lna=lna, vga=vga, amp=amp)

    r = search_gain(probe,
                    quality=lambda iq: fm_pilot_excess_db(iq, SAMPLE_RATE))
    iq = h.capture_array(freq_hz, SAMPLE_RATE, int(SAMPLE_RATE * 0.2),
                         lna=r["lna"], vga=r["vga"], amp=r["amp"])
    health = capture_health(iq, SAMPLE_RATE, channel_bw=CHANNEL_BW)
    pilot = fm_pilot_excess_db(iq, SAMPLE_RATE)
    # Refine the frequency from the capture itself: sweep bins rarely sit on
    # a channel, so find where the carrier's energy actually is (within
    # +/-300 kHz) and snap that onto the 100 kHz channel grid. The pilot
    # survives a sub-channel tuning offset (the discriminator turns a
    # carrier offset into a DC term, not into pilot loss), so the score
    # above stays valid for the refined frequency.
    nfft = 8192
    spec = np.zeros(nfft)
    for k in range(min(16, len(iq) // nfft)):
        spec += np.abs(np.fft.fftshift(
            np.fft.fft(iq[k * nfft:(k + 1) * nfft] * np.hanning(nfft)))) ** 2
    fbins = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / SAMPLE_RATE))
    win = np.abs(fbins) <= 300e3
    off = float(fbins[win][np.argmax(spec[win])])
    freq_hz = round((freq_hz + off) / 100e3) * 100e3
    return {"freq": freq_hz, "lna": r["lna"], "vga": r["vga"],
            "amp": r["amp"], "status": r["status"],
            "counts": health["adc_counts"],
            "excess_db": health["channel_excess_db"],
            "pilot_db": pilot}


def verify_quiet(h, freq_hz):
    """A quiet frequency must FAIL the health check at moderate gain."""
    iq = h.capture_array(freq_hz, SAMPLE_RATE, int(SAMPLE_RATE * 0.2),
                         lna=24, vga=30, amp=False)
    health = capture_health(iq, SAMPLE_RATE, channel_bw=CHANNEL_BW)
    pilot = fm_pilot_excess_db(iq, SAMPLE_RATE)
    quiet = (not health["ok"]) and (pilot is None or pilot < PILOT_OK_DB)
    return quiet, health, pilot


def run_preflight(h, band, candidates=3, out=sys.stdout):
    """The whole flow against a radio-like object. Returns (ok, report).

    ``h`` needs detect(), sweep_collect(lo, hi, num_sweeps=), and
    capture_array(freq, rate, n, lna=, vga=, amp=) -- hackrfpy's HackRF, or
    a simulator. Separated from main() so tests can drive it directly.
    """
    p = lambda *a: print(*a, file=out)
    report = {"board": False, "band_alive": False, "stations": [],
              "quiet_hz": None, "quiet_verified": False}

    p("== 1. board ==")
    det = h.detect()
    if not det.get("ready"):
        p(f"   FAIL: no usable HackRF: {det.get('problem')}")
        return False, report
    fw = det["boards"][0].get("firmware") if det.get("boards") else "?"
    p(f"   ok: firmware {fw}")
    for w in det.get("warnings", []):
        p(f"   ! {w}")
    report["board"] = True

    lo, hi = band
    p(f"\n== 2. band sweep {lo / 1e6:g}-{hi / 1e6:g} MHz ==")
    rows = h.sweep_collect(lo, hi, num_sweeps=4)
    freqs, power = spectrum_from_sweep(rows)
    if len(freqs) == 0:
        p("   FAIL: sweep returned nothing")
        return False, report
    floor = float(np.median(power))
    spread = float(np.max(power)) - floor
    p(f"   noise floor {floor:.1f} dB, strongest point +{spread:.1f} dB "
      f"above it")
    if spread < DETECT_THRESHOLD_DB:
        p("   FAIL: the band is FLAT. Nothing is reaching the radio; this "
          "is the antenna,")
        p("   the cabling, or the environment -- no gain setting or "
          "station choice will fix it.")
        p("   Fully extend the antenna, move to a window, or use ~72 cm "
          "of wire.")
        return False, report
    report["band_alive"] = True

    stations = find_stations(freqs, power)
    p(f"   {len(stations)} station candidate(s) above "
      f"+{DETECT_THRESHOLD_DB:g} dB")

    p(f"\n== 3. scoring the top {min(candidates, len(stations))} by "
      f"measured pilot ==")
    scored = []
    for f, prom in stations[:candidates]:
        # snap the sweep-bin center onto the 100 kHz FM channel grid before
        # tuning to it; sweep bins rarely land exactly on a channel
        f = round(f / 100e3) * 100e3
        s = score_station(h, f)
        s["sweep_prom"] = prom
        scored.append(s)
        pil = "n/a" if s["pilot_db"] is None else f"{s['pilot_db']:+5.1f} dB"
        p(f"   {s['freq'] / 1e6:7.2f} MHz  sweep +{prom:4.1f} dB  ->  "
          f"lna={s['lna']:2d} vga={s['vga']:2d} "
          f"amp={'on ' if s['amp'] else 'off'}  "
          f"{s['counts']:5.1f}/128 counts  pilot {pil}")
    report["stations"] = scored
    good = [s for s in scored
            if s["pilot_db"] is not None and s["pilot_db"] >= PILOT_OK_DB
            and s["status"] == "ok"]
    # The pilot decides WHETHER a station is usable; once several are, its
    # excess saturates, so among candidates within 6 dB of the best pilot
    # prefer the one the sweep measured strongest.
    good.sort(key=lambda s: s["pilot_db"], reverse=True)
    if len(good) > 1:
        top = good[0]["pilot_db"]
        near = [s for s in good if s["pilot_db"] >= top - 6.0]
        near.sort(key=lambda s: s.get("sweep_prom", 0), reverse=True)
        good = near + [s for s in good if s not in near]

    p("\n== 4. quiet reference ==")
    quiet_hz = find_quiet(freqs, power, stations)
    if quiet_hz is None:
        p("   ! no sufficiently empty spot inside the sweep span; the "
          "negative-control")
        p("   captures (data_5) will need a frequency chosen by hand.")
    else:
        ok_q, qh, qp = verify_quiet(h, quiet_hz)
        report["quiet_hz"] = quiet_hz
        report["quiet_verified"] = ok_q
        pil = "n/a" if qp is None else f"{qp:+.1f} dB"
        p(f"   candidate {quiet_hz / 1e6:.2f} MHz: "
          f"{qh['adc_counts']:.1f}/128 counts, pilot {pil} -> "
          f"{'verified quiet' if ok_q else 'NOT quiet, rejected'}")
        if not ok_q:
            report["quiet_hz"] = None

    p("\n== verdict ==")
    if not good:
        p("   FAIL: no candidate produced a stereo pilot. The band has "
          "energy but no")
        p("   station was cleanly received -- improve the antenna before "
          "collecting;")
        p("   a corpus built from a marginal station is marginal "
          "everywhere.")
        return False, report
    best = good[0]
    p(f"   PASS: build the corpus from {best['freq'] / 1e6:.2f} MHz "
      f"(pilot {best['pilot_db']:+.1f} dB,")
    p(f"   gain lna={best['lna']} vga={best['vga']} "
      f"amp={'on' if best['amp'] else 'off'}).")
    cmd = (f"uv run python tools/collect_dev_data.py "
           f"--station {best['freq'] / 1e6:g}e6")
    if report["quiet_hz"]:
        cmd += f" --quiet {round(report['quiet_hz'] / 100e3) * 100e3 / 1e6:g}e6"
    p(f"\n   {cmd}\n")
    return True, report


# ---------------------------------------------------------------------------
# simulation (also what the tests drive)
# ---------------------------------------------------------------------------
class SimulatedRadio:
    """A band with real stations in it, behind the hackrfpy interface.

    Stations: dict {freq_hz: antenna_counts} -- signal level BEFORE gain.
    quiet gaps are just absence. The receive chain matches the model the
    gain-search tests use: front end (amp+LNA) amplifies the antenna,
    receiver noise is injected after it and rides the VGA alone.
    """

    def __init__(self, stations, noise_counts=1.2, seed=7):
        self.stations = dict(stations)
        self.noise_counts = float(noise_counts)
        self.seed = seed

    def detect(self):
        return {"ready": True, "boards": [{"firmware": "simulated"}],
                "warnings": []}

    def sweep_collect(self, lo, hi, num_sweeps=1, bin_width=100e3):
        rows = []
        rng = np.random.default_rng(self.seed)
        for _ in range(num_sweeps):
            nbins = int((hi - lo) / bin_width)
            db = np.full(nbins, -75.0) + rng.normal(0, 0.7, nbins)
            for f, counts in self.stations.items():
                if lo <= f < hi:
                    i = int((f - lo) / bin_width)
                    for j, bleed in ((i - 1, 6), (i, 0), (i + 1, 6)):
                        if 0 <= j < nbins:
                            db[j] = max(db[j],
                                        -75 + 20 * np.log10(max(counts, .1))
                                        + 18 - bleed)
            rows.append({"hz_low": lo, "hz_high": hi, "bin_width": bin_width,
                         "num_samples": 8192, "date": "", "time": "",
                         "db": db.tolist()})
        return rows

    def capture_array(self, freq, rate, n, lna=16, vga=20, amp=False, **k):
        rng = np.random.default_rng(self.seed + int(freq) % 100003)
        n = int(n)
        t = np.arange(n) / rate
        g_fe = 10 ** ((lna + (14 if amp else 0)) / 20)
        g_vga = 10 ** (vga / 20)
        iq = np.zeros(n, dtype=np.complex128)
        for f, counts in self.stations.items():
            off = f - freq
            if abs(off) > rate / 2:
                continue
            msg = (0.6 * np.sin(2 * np.pi * 600 * t)
                   + 0.09 * np.sin(2 * np.pi * 19_000 * t))
            ph = 2 * np.pi * (75e3 * np.cumsum(msg) / rate + off * t)
            iq += counts * g_fe * g_vga * np.exp(1j * ph)
        noise_c = self.noise_counts * g_vga + 0.7
        iq += noise_c * (rng.standard_normal(n)
                         + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq /= 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1, lim)
                + 1j * np.clip(iq.imag, -1, lim)).astype(np.complex64)


def main():
    p = argparse.ArgumentParser(
        description="Preflight a collection session (READ-ONLY).")
    p.add_argument("--region", choices=sorted(REGIONS), default="americas",
                   help="FM band span to sweep (americas: 88-108, "
                        "world: 87.5-108)")
    p.add_argument("--candidates", type=int, default=3,
                   help="how many sweep candidates to auto-gain and score")
    p.add_argument("--simulate", action="store_true",
                   help="run against a synthesized band, no radio needed")
    p.add_argument("--tools-dir", default=None,
                   help="path to hackrf-tools if not on PATH")
    args = p.parse_args()

    if args.simulate:
        h = SimulatedRadio({94.1e6: 8.0, 98.5e6: 25.0, 104.3e6: 0.9})
        print("[simulated band: stations at 94.1, 98.5, 104.3 MHz]\n")
    else:
        try:
            from hackrfpy import HackRF
        except ModuleNotFoundError:
            print("ERROR: hackrfpy is not installed (or use --simulate):\n"
                  "    uv sync --extra examples-hackrf", file=sys.stderr)
            return 1
        h = HackRF(tools_dir=args.tools_dir, verbose=False)

    ok, _report = run_preflight(h, REGIONS[args.region],
                                candidates=args.candidates)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
