#! /usr/bin/python3
"""fm_band_explorer.py -- scan the FM band, click a station, tune it live.

An interactive matplotlib window for answering "why am I not hearing
anything?". Three things on screen at once:

  TOP     a sweep of the whole FM band. Stations show as humps well above
          the noise floor, and detected peaks are marked and labelled. If
          this panel is FLAT, nothing is reaching the radio -- the problem
          is the antenna, the gain, or the connection, not your tuning. That
          single observation is what this example is for.

  BOTTOM  a live narrowband view of the tuned channel, refreshed a few times
          a second, so you can see the channel you are actually sitting on.

  READOUT ADC counts and how far the channel stands above the band edges,
          which are the two numbers that distinguish "no signal", "gain too
          low", and "gain too high" from each other.

Click anywhere on the top panel to tune there, use the slider, or step
between detected stations with the arrow buttons.

    python examples/fm_band_explorer.py
    python examples/fm_band_explorer.py --simulate    # no radio needed

--simulate synthesizes a band with a few stations in it. Use it to confirm
the window itself works before blaming the hardware: if the simulated band
shows stations and the real one does not, the DSP and the UI are fine.

Requires matplotlib, plus hackrfpy + hackrf-tools for real use:
    uv sync --extra plotting --extra examples-hackrf
"""

# ===========================================================================
#  CONFIG -- EDIT THIS
# ===========================================================================

BAND_MIN_HZ = 88.0e6      # FM broadcast band, region 2 (Americas)
BAND_MAX_HZ = 108.0e6     # in most of the rest of the world: 87.5 - 108
START_HZ = 98.5e6         # where the tuner starts

SAMPLE_RATE = 2_000_000   # span of the live narrowband panel
CHANNEL_BW = 100_000      # half-width of one FM channel
LNA_DB, VGA_DB = 16, 20   # starting gain; adjustable in the window
TOOLS_DIR = None          # path to hackrf-tools if not on PATH

# A station has to stand this far above the band's noise floor to be marked.
# Lower it if you know a weak station is there and it isn't being detected.
DETECT_THRESHOLD_DB = 6.0
MIN_STATION_SPACING_HZ = 300_000

# ===========================================================================

import argparse
import importlib.util
import sys

import numpy as np

from sdr_dsp.core import psd, capture_health


# ---------------------------------------------------------------------------
# pure logic -- no hardware, no plotting, so it can be tested
# ---------------------------------------------------------------------------
def assemble_sweep(rows):
    """Turn hackrf_sweep rows into (freqs_hz, db) sorted by frequency.

    Each row covers one slice of the span and carries a list of bin powers.
    Rows arrive in whatever order the sweep visited them, and a pass may be
    split across `time` values, so they are keyed by start frequency and
    concatenated in frequency order rather than arrival order.
    """
    if not rows:
        return np.zeros(0), np.zeros(0)
    chunks = {}
    for r in rows:
        lo, hi = float(r["hz_low"]), float(r["hz_high"])
        db = np.asarray(r["db"], dtype=float)
        if db.size == 0:
            continue
        step = (hi - lo) / db.size
        chunks[lo] = (lo + step * (np.arange(db.size) + 0.5), db)
    if not chunks:
        return np.zeros(0), np.zeros(0)
    freqs = np.concatenate([chunks[k][0] for k in sorted(chunks)])
    db = np.concatenate([chunks[k][1] for k in sorted(chunks)])
    order = np.argsort(freqs)
    return freqs[order], db[order]


def find_stations(freqs, db, threshold_db=DETECT_THRESHOLD_DB,
                  min_spacing_hz=MIN_STATION_SPACING_HZ, limit=40):
    """Peaks standing clear of the noise floor. Returns [(freq_hz, db), ...].

    The floor is the median rather than the mean: a band with strong
    stations in it drags a mean upward until the stations themselves stop
    looking exceptional, whereas the median tracks the empty majority of the
    span. Peaks are then taken greedily strongest-first with a minimum
    spacing, so one broad station is reported once instead of once per bin.
    """
    if freqs.size == 0 or db.size != freqs.size:
        return []
    floor = float(np.median(db))
    cutoff = floor + threshold_db
    candidates = np.argsort(db)[::-1]
    picked = []
    for i in candidates:
        if db[i] < cutoff:
            break
        f = float(freqs[i])
        if any(abs(f - pf) < min_spacing_hz for pf, _ in picked):
            continue
        picked.append((f, float(db[i])))
        if len(picked) >= limit:
            break
    return sorted(picked)


def split_gain(total):
    """Split a total dB figure into (lna, vga) on the hardware's steps.

    The LNA sits in front of the mixer, so gain taken there amplifies
    everything in the band and is the first thing to overload on a strong
    signal; the VGA is after the filter and only sees the channel. So fill
    the LNA to a moderate 24 dB, give the rest to the VGA, and only push the
    LNA higher once the VGA has run out of range. Steps are 8 dB for the LNA
    and 2 dB for the VGA, and the caller's figure is rounded down onto them.
    """
    total = max(0, min(102, int(total)))
    lna = min(24, (total // 8) * 8)
    vga = min(62, ((total - lna) // 2) * 2)
    while lna < 40 and lna + vga < total - 1:
        lna += 8
        vga = min(62, ((total - lna) // 2) * 2)
    return lna, vga


def simulate_sweep(f_min, f_max, stations=(88.5e6, 94.1e6, 98.5e6, 103.7e6),
                   bin_hz=100_000, seed=0):
    """A fake band with a few stations, for --simulate."""
    rng = np.random.default_rng(seed)
    n = int((f_max - f_min) / bin_hz)
    freqs = f_min + bin_hz * (np.arange(n) + 0.5)
    db = -95.0 + rng.standard_normal(n) * 1.5
    for k, fc in enumerate(stations):
        width = 100e3
        # cycle the heights rather than stepping monotonically down, so a
        # long station list still produces real signals instead of peaks
        # that fade to nothing and then go negative
        height = 28.0 - 4.0 * (k % 5)
        db += height * np.exp(-0.5 * ((freqs - fc) / width) ** 2)
    return freqs, db


# ---------------------------------------------------------------------------
# radio access
# ---------------------------------------------------------------------------
class Radio:
    """Thin wrapper so the UI can run identically against real or fake data."""

    def __init__(self, simulate=False, tools_dir=None):
        self.simulate = simulate
        self._h = None
        if not simulate:
            from hackrfpy import HackRF
            self._h = HackRF(tools_dir=tools_dir, verbose=False)
            det = self._h.detect()
            if not det["ready"]:
                raise RuntimeError(f"no usable HackRF: {det['problem']}")

    def sweep(self, f_min, f_max):
        if self.simulate:
            return simulate_sweep(f_min, f_max)
        rows = self._h.sweep_collect(f_min, f_max, num_sweeps=1)
        return assemble_sweep(rows)

    def snapshot(self, freq, rate, n, lna, vga):
        """A short IQ capture at one frequency."""
        if self.simulate:
            t = np.arange(n) / rate
            rng = np.random.default_rng(int(freq) % 1000)
            near = min(abs(freq - f) for f in
                       (88.5e6, 94.1e6, 98.5e6, 103.7e6))
            amp = 90.0 * np.exp(-0.5 * (near / 120e3) ** 2)
            msg = np.sin(2 * np.pi * 440 * t)
            z = np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / rate) * amp
            z = z + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1.5
            return ((np.round(z.real) + 1j * np.round(z.imag))
                    / 127.0).astype(np.complex64)
        return self._h.capture_array(freq, rate, n, lna=lna, vga=vga,
                                     amp=False)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
def run_ui(radio, start_hz):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    state = {"freq": start_hz, "lna": LNA_DB, "vga": VGA_DB, "stations": []}

    fig = plt.figure(figsize=(11, 7))
    fig.canvas.manager.set_window_title("FM band explorer")
    ax_band = fig.add_axes([0.08, 0.60, 0.88, 0.32])
    ax_chan = fig.add_axes([0.08, 0.28, 0.88, 0.24])
    ax_freq = fig.add_axes([0.08, 0.16, 0.66, 0.03])
    ax_gain = fig.add_axes([0.08, 0.11, 0.66, 0.03])
    ax_scan = fig.add_axes([0.80, 0.145, 0.16, 0.045])
    ax_prev = fig.add_axes([0.80, 0.095, 0.075, 0.04])
    ax_next = fig.add_axes([0.885, 0.095, 0.075, 0.04])

    band_line, = ax_band.plot([], [], lw=0.9)
    tune_line = ax_band.axvline(start_hz / 1e6, color="crimson", lw=1.2)
    ax_band.set_xlim(BAND_MIN_HZ / 1e6, BAND_MAX_HZ / 1e6)
    ax_band.set_ylabel("dB")
    ax_band.set_title("band sweep -- click to tune")
    ax_band.grid(alpha=0.3)

    chan_line, = ax_chan.plot([], [], lw=0.9)
    ax_chan.set_xlabel("offset from tuned frequency (kHz)")
    ax_chan.set_ylabel("dB")
    ax_chan.grid(alpha=0.3)
    for edge in (-CHANNEL_BW / 1e3, CHANNEL_BW / 1e3):
        ax_chan.axvline(edge, color="gray", ls="--", lw=0.8)

    readout = fig.text(0.08, 0.035, "", family="monospace", fontsize=9)

    s_freq = Slider(ax_freq, "MHz", BAND_MIN_HZ / 1e6, BAND_MAX_HZ / 1e6,
                    valinit=start_hz / 1e6, valstep=0.1)
    s_gain = Slider(ax_gain, "gain dB", 0, 102,
                    valinit=LNA_DB + VGA_DB, valstep=2)
    b_scan = Button(ax_scan, "Rescan band")
    b_prev = Button(ax_prev, "< prev")
    b_next = Button(ax_next, "next >")

    def set_gain(total):
        state["lna"], state["vga"] = split_gain(total)

    def do_scan(_=None):
        ax_band.set_title("band sweep -- scanning ...")
        fig.canvas.draw_idle()
        freqs, db = radio.sweep(BAND_MIN_HZ, BAND_MAX_HZ)
        if freqs.size == 0:
            ax_band.set_title("band sweep -- sweep returned nothing")
            return
        band_line.set_data(freqs / 1e6, db)
        ax_band.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
        pad = max(3.0, 0.1 * (db.max() - db.min()))
        ax_band.set_ylim(db.min() - pad, db.max() + pad)
        for t in list(ax_band.texts):
            t.remove()
        found = find_stations(freqs, db)
        state["stations"] = [f for f, _ in found]
        for f, d in found:
            ax_band.annotate(f"{f/1e6:.1f}", (f / 1e6, d),
                             textcoords="offset points", xytext=(0, 5),
                             ha="center", fontsize=7, color="crimson")
        spread = float(db.max() - np.median(db))
        if not found:
            ax_band.set_title(
                f"band sweep -- NO STATIONS FOUND (strongest point is only "
                f"{spread:.1f} dB above the floor). Check antenna and gain.")
        else:
            ax_band.set_title(f"band sweep -- {len(found)} station(s); "
                              f"click to tune")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is ax_band and event.xdata:
            s_freq.set_val(round(event.xdata, 1))

    def hop(delta):
        def handler(_):
            if not state["stations"]:
                return
            cur = state["freq"]
            ordered = sorted(state["stations"])
            if delta > 0:
                nxt = next((f for f in ordered if f > cur + 5e4), ordered[0])
            else:
                nxt = next((f for f in reversed(ordered) if f < cur - 5e4),
                           ordered[-1])
            s_freq.set_val(round(nxt / 1e6, 1))
        return handler

    def on_freq(val):
        state["freq"] = val * 1e6
        tune_line.set_xdata([val, val])

    def on_gain(val):
        set_gain(val)

    s_freq.on_changed(on_freq)
    s_gain.on_changed(on_gain)
    b_scan.on_clicked(do_scan)
    b_prev.on_clicked(hop(-1))
    b_next.on_clicked(hop(+1))
    fig.canvas.mpl_connect("button_press_event", on_click)
    set_gain(LNA_DB + VGA_DB)

    def refresh(_frame):
        freq = state["freq"]
        try:
            iq = radio.snapshot(freq, SAMPLE_RATE, int(SAMPLE_RATE * 0.05),
                                state["lna"], state["vga"])
        except Exception as exc:                      # keep the window alive
            readout.set_text(f"capture failed: {exc}")
            return chan_line,
        f, p = psd(iq, SAMPLE_RATE, nfft=2048)
        chan_line.set_data(f / 1e3, p)
        ax_chan.set_xlim(-SAMPLE_RATE / 2e3, SAMPLE_RATE / 2e3)
        ax_chan.set_ylim(p.min() - 3, p.max() + 6)

        h = capture_health(iq, SAMPLE_RATE, channel_bw=CHANNEL_BW)
        excess = h["channel_excess_db"]
        verdict = ("station present" if h["ok"] else
                   " | ".join(r.split(",")[0] for r in h["reasons"]))
        readout.set_text(
            f"{freq/1e6:7.1f} MHz   lna {state['lna']:2d} vga {state['vga']:2d}"
            f"   peak {h['adc_counts']:5.1f}/127"
            f"   channel {excess:+6.1f} dB   {verdict}"
            if excess is not None else
            f"{freq/1e6:7.1f} MHz   peak {h['adc_counts']:5.1f}/127")
        return chan_line,

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, refresh, interval=250, blit=False,
                         cache_frame_data=False)
    fig._explorer_anim = anim          # keep a reference or it is collected
    do_scan()
    plt.show()
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Interactive FM band scanner and tuner.")
    p.add_argument("--simulate", action="store_true",
                   help="synthesize a band; no radio required")
    p.add_argument("--start", type=float, default=START_HZ,
                   help="initial tuned frequency in Hz")
    args = p.parse_args()

    if importlib.util.find_spec("matplotlib") is None:
        print("needs matplotlib:  uv sync --extra plotting", file=sys.stderr)
        return 1
    if not args.simulate and importlib.util.find_spec("hackrfpy") is None:
        print("needs hackrfpy:  uv sync --extra examples-hackrf\n"
              "(or run with --simulate to try the window without a radio)",
              file=sys.stderr)
        return 1
    try:
        radio = Radio(simulate=args.simulate, tools_dir=TOOLS_DIR)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    run_ui(radio, args.start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
