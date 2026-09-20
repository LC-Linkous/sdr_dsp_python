"""Logic behind the interactive band explorer, tested without a radio or a
window: sweep assembly, station detection, and the gain split.

The UI itself isn't unit tested -- but everything it decides with is here,
and the detection threshold in particular is what tells someone whether the
band is empty or their antenna is disconnected. Getting that wrong sends
people looking in the wrong place.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "fm_band_explorer.py"


def _load():
    sys.path.insert(0, str(EXAMPLE.parent))
    spec = importlib.util.spec_from_file_location("_fm_band_explorer", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fm_band_explorer"] = mod
    spec.loader.exec_module(mod)
    return mod


fbe = _load()


# --------------------------------------------------------------------------
# sweep assembly
# --------------------------------------------------------------------------
def _row(lo, hi, db):
    return {"hz_low": lo, "hz_high": hi, "db": list(db)}


def test_assemble_sweep_orders_by_frequency_not_arrival():
    """hackrf_sweep visits slices in its own order; output must be sorted."""
    rows = [
        _row(100e6, 101e6, [-90, -89]),
        _row(98e6, 99e6, [-95, -94]),
        _row(99e6, 100e6, [-92, -91]),
    ]
    freqs, db = fbe.assemble_sweep(rows)
    assert freqs.size == 6
    assert np.all(np.diff(freqs) > 0), "frequencies are not monotonic"
    assert db[0] == -95 and db[-1] == -89


def test_assemble_sweep_places_bins_inside_their_slice():
    freqs, _ = fbe.assemble_sweep([_row(88e6, 89e6, [-90] * 4)])
    assert freqs.min() > 88e6 and freqs.max() < 89e6
    assert np.allclose(np.diff(freqs), 250e3)


def test_assemble_sweep_handles_empty_input():
    for rows in ([], [_row(88e6, 89e6, [])]):
        freqs, db = fbe.assemble_sweep(rows)
        assert freqs.size == 0 and db.size == 0


# --------------------------------------------------------------------------
# station detection
# --------------------------------------------------------------------------
def test_finds_the_simulated_stations():
    stations = (88.5e6, 94.1e6, 98.5e6, 103.7e6)
    freqs, db = fbe.simulate_sweep(88e6, 108e6, stations=stations)
    found = [f for f, _ in fbe.find_stations(freqs, db)]
    assert len(found) == len(stations), f"expected {len(stations)}, got {found}"
    for want in stations:
        assert any(abs(f - want) < 150e3 for f in found), f"missed {want/1e6}"


def test_empty_band_reports_nothing():
    """The case that matters: an antenna problem must not look like stations."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(88e6, 108e6, 400)
    db = -95 + rng.standard_normal(400) * 1.5
    assert fbe.find_stations(freqs, db) == []


def test_one_station_is_reported_once_not_per_bin():
    """A broad carrier spans many bins; minimum spacing collapses them."""
    freqs, db = fbe.simulate_sweep(88e6, 108e6, stations=(98.5e6,))
    found = fbe.find_stations(freqs, db)
    assert len(found) == 1, f"one station reported as {len(found)} peaks"


def test_uses_a_median_floor_not_a_mean():
    """A band crowded with strong signals must still detect them.

    A mean floor rises with the stations themselves until nothing clears the
    threshold any more; the median tracks the empty majority of the span.
    """
    many = tuple(88.5e6 + k * 1.0e6 for k in range(12))
    freqs, db = fbe.simulate_sweep(88e6, 108e6, stations=many)
    found = fbe.find_stations(freqs, db)
    assert len(found) >= 10, (
        f"only {len(found)} of {len(many)} crowded stations detected; the "
        "noise floor estimate is being dragged up by the signals")


def test_detection_threshold_is_honoured():
    freqs, db = fbe.simulate_sweep(88e6, 108e6, stations=(98.5e6,))
    assert fbe.find_stations(freqs, db, threshold_db=100.0) == []
    assert len(fbe.find_stations(freqs, db, threshold_db=3.0)) >= 1


def test_results_are_sorted_by_frequency():
    freqs, db = fbe.simulate_sweep(88e6, 108e6)
    found = [f for f, _ in fbe.find_stations(freqs, db)]
    assert found == sorted(found)


def test_find_stations_handles_degenerate_input():
    assert fbe.find_stations(np.zeros(0), np.zeros(0)) == []
    assert fbe.find_stations(np.arange(10.0), np.zeros(3)) == []


# --------------------------------------------------------------------------
# gain split
# --------------------------------------------------------------------------
@pytest.mark.parametrize("total", [0, 8, 20, 36, 50, 60, 86, 102])
def test_gain_split_lands_on_the_hardware_grid(total):
    lna, vga = fbe.split_gain(total)
    assert lna % 8 == 0 and 0 <= lna <= 40, f"lna={lna} off grid"
    assert vga % 2 == 0 and 0 <= vga <= 62, f"vga={vga} off grid"
    assert lna + vga <= total, "split exceeds the requested total"
    assert total - (lna + vga) <= 2, f"split wastes {total-(lna+vga)} dB"


def test_gain_split_prefers_baseband_over_front_end():
    """LNA gain is ahead of the mixer and overloads first, so keep it modest.

    36 dB split as LNA 32 / VGA 4 amplifies the whole band into the mixer
    before any filtering; LNA 24 / VGA 12 reaches the same total with far
    more headroom against a strong neighbouring station.
    """
    lna, vga = fbe.split_gain(36)
    assert lna <= 24, f"lna={lna} too high for a mid-range total"
    assert vga >= 8


def test_gain_split_raises_lna_only_when_vga_saturates():
    assert fbe.split_gain(86)[0] == 24        # VGA still has room
    assert fbe.split_gain(102)[0] == 40       # VGA maxed, LNA must take it
    assert fbe.split_gain(102) == (40, 62)


def test_gain_split_clamps_out_of_range_input():
    assert fbe.split_gain(-50) == (0, 0)
    lna, vga = fbe.split_gain(500)
    assert lna == 40 and vga == 62
