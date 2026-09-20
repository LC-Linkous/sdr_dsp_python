"""tools/preflight_collection.py must catch a bad session before recording.

The preflight exists so a collection session cannot begin from a setup that
would only produce hiss: a dead antenna must fail loudly, a station must be
chosen by its measured pilot (not just sweep power), and the quiet reference
must be verified empty rather than assumed. All of it is driven here through
the tool's own SimulatedRadio, which shares the receive-chain model of the
gain-search tests: the front end decides SNR, the VGA only decides level.
"""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = (Path(__file__).resolve().parent.parent / "tools"
        / "preflight_collection.py")


def _load():
    spec = importlib.util.spec_from_file_location("_preflight", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_preflight"] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load()


# ---------------------------------------------------------------------------
# pure spectrum analysis
# ---------------------------------------------------------------------------
def _band(stations, lo=88e6, hi=108e6, seed=7):
    radio = pf.SimulatedRadio(stations, seed=seed)
    rows = radio.sweep_collect(lo, hi, num_sweeps=4)
    return pf.spectrum_from_sweep(rows)


def test_spectrum_flattens_and_averages_sweep_rows():
    freqs, power = _band({98.5e6: 20.0})
    assert len(freqs) == len(power) > 0
    assert np.all(np.diff(freqs) > 0)
    # averaging across passes keeps the floor tight
    floor = np.median(power)
    assert np.percentile(power, 90) < floor + 3


def test_find_stations_ranks_by_strength():
    freqs, power = _band({94.1e6: 8.0, 98.5e6: 25.0, 104.3e6: 0.9})
    picks = pf.find_stations(freqs, power)
    assert len(picks) == 3
    got = [round(f / 1e5) * 1e5 for f, _ in picks]
    assert got[0] == pytest.approx(98.5e6, abs=1e5)
    proms = [p for _, p in picks]
    assert proms == sorted(proms, reverse=True)


def test_find_stations_absorbs_near_duplicates():
    """Bleed into adjacent bins must not become extra stations."""
    freqs, power = _band({98.5e6: 25.0})
    picks = pf.find_stations(freqs, power)
    assert len(picks) == 1


def test_find_quiet_avoids_stations_and_verifies_by_peak():
    freqs, power = _band({94.1e6: 8.0, 98.5e6: 25.0})
    q = pf.find_quiet(freqs, power, pf.find_stations(freqs, power))
    assert q is not None
    assert min(abs(q - 94.1e6), abs(q - 98.5e6)) > 500e3


def test_find_quiet_gives_up_on_a_crowded_band():
    stations = {f: 10.0 for f in np.arange(88.2e6, 108e6, 0.6e6)}
    freqs, power = _band(stations)
    picks = pf.find_stations(freqs, power)
    assert pf.find_quiet(freqs, power, picks) is None


# ---------------------------------------------------------------------------
# the whole flow
# ---------------------------------------------------------------------------
def _run(stations, **kw):
    radio = pf.SimulatedRadio(stations, **kw)
    out = io.StringIO()
    ok, report = pf.run_preflight(radio, (88e6, 108e6), out=out)
    return ok, report, out.getvalue()


def test_preflight_passes_on_a_healthy_band():
    ok, report, text = _run({94.1e6: 8.0, 98.5e6: 25.0, 104.3e6: 0.9})
    assert ok
    assert report["band_alive"]
    best = report["stations"]
    assert any(s["pilot_db"] and s["pilot_db"] > pf.PILOT_OK_DB
               for s in best)
    assert report["quiet_hz"] is not None and report["quiet_verified"]
    assert "--station" in text and "--quiet" in text


def test_preflight_refines_onto_the_channel_grid():
    """Sweep bins sit off-channel; the suggestion must land on it."""
    ok, report, text = _run({98.5e6: 25.0})
    assert ok
    best = max(report["stations"], key=lambda s: s["pilot_db"] or -1e9)
    assert best["freq"] == pytest.approx(98.5e6, abs=1)
    assert "98.5e6" in text


def test_preflight_fails_loud_on_a_dead_antenna():
    """A flat band is an antenna problem and must stop the session before
    any station scoring happens -- this is the check whose absence let the
    original dead capture get recorded."""
    ok, report, text = _run({})
    assert not ok
    assert report["board"] and not report["band_alive"]
    assert "antenna" in text.lower()
    assert report["stations"] == []


def test_preflight_fails_when_energy_has_no_pilot():
    """Band energy without a receivable station must not pass. An antenna
    picking up broadband hash lights the sweep up, but no candidate will
    produce a pilot, and building a corpus from it would be the old bug
    with extra steps."""
    radio = pf.SimulatedRadio({98.5e6: 25.0})
    real_capture = radio.capture_array

    def no_pilot_capture(freq, rate, n, **kw):
        rng = np.random.default_rng(5)
        # energy at the tuned channel, but unpiloted (a carrier, not FM)
        t = np.arange(int(n)) / rate
        g = 10 ** ((kw.get("lna", 16) + kw.get("vga", 20)
                    + (14 if kw.get("amp") else 0)) / 20)
        iq = 20.0 * g * np.exp(2j * np.pi * 5e3 * t) / 128.0
        iq = iq + real_capture(freq, rate, n, **{**kw}) * 0  # keep shape
        noise = (rng.standard_normal(int(n))
                 + 1j * rng.standard_normal(int(n))) / np.sqrt(2)
        iq = iq + (1.2 * 10 ** (kw.get("vga", 20) / 20) + 0.7
                   ) * noise / 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1, lim)
                + 1j * np.clip(iq.imag, -1, lim)).astype(np.complex64)

    radio.capture_array = no_pilot_capture
    out = io.StringIO()
    ok, report = pf.run_preflight(radio, (88e6, 108e6), out=out)
    assert not ok
    assert report["band_alive"]
    assert "pilot" in out.getvalue().lower()


def test_quiet_candidate_with_a_station_on_it_is_rejected():
    """If the 'quiet' spot turns out occupied when actually captured, the
    preflight must reject it rather than bless it."""
    radio = pf.SimulatedRadio({98.5e6: 25.0})
    real_capture = radio.capture_array

    def haunted_capture(freq, rate, n, **kw):
        if abs(freq - 98.5e6) > 1e6:      # every "quiet" spot has a station
            return real_capture(98.5e6 + (freq - 98.5e6) * 0, rate, n, **kw)
        return real_capture(freq, rate, n, **kw)

    radio.capture_array = haunted_capture
    out = io.StringIO()
    ok, report = pf.run_preflight(radio, (88e6, 108e6), out=out)
    assert report["quiet_hz"] is None
    assert not report["quiet_verified"]
    assert ok  # a missing quiet reference degrades the run, not the verdict


def test_simulated_radio_matches_the_gain_model():
    """The sim must reward the front end and not the VGA, or the preflight
    would test a different physics than search_gain optimizes."""
    radio = pf.SimulatedRadio({98.5e6: 0.5})
    from sdr_dsp.core import fm_pilot_excess_db
    weak = radio.capture_array(98.5e6, 2e6, 120_000, lna=0, vga=40, amp=False)
    strong = radio.capture_array(98.5e6, 2e6, 120_000, lna=40, vga=0,
                                 amp=True)
    pw = fm_pilot_excess_db(weak, 2e6, nfft=8192)
    ps = fm_pilot_excess_db(strong, 2e6, nfft=8192)
    assert ps > pw + 10, (pw, ps)
