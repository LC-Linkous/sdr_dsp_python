"""The gain search in examples/collect_sample_data.py must actually converge.

This is the check that would have prevented the original sample capture from
shipping. That file was recorded at roughly 3 of 127 ADC counts -- every
sample in 2 MB was one of {-2, -1, 0, 1, 2} -- and nothing in the recording
path noticed. The search here exists to land the level in a usable window
before anything is written to disk, so it needs to demonstrably do that
across the range of signal strengths a real antenna presents.

hackrfpy is an optional extra, and the example imports it at module scope,
so these skip when it isn't installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hackrfpy",
                    reason="optional extra: uv sync --extra examples-hackrf")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "collect_sample_data.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("_collect_sample_data",
                                                  EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_collect_sample_data"] = mod
    spec.loader.exec_module(mod)
    return mod


collect = _load_example()


class FakeRadio:
    """A radio whose output level scales with applied gain, and clips.

    antenna_counts is what a peak sample would measure at 0 dB of combined
    gain. Real hardware is messier than this -- gain stages are not perfectly
    linear and the LNA has its own noise figure -- but the search only needs
    monotonicity in gain to converge, which this captures.
    """

    def __init__(self, antenna_counts, noise_counts=0.5):
        self.antenna_counts = float(antenna_counts)
        self.noise_counts = float(noise_counts)
        self.calls = []

    def capture_array(self, freq, sample_rate, num_samples, *, lna=16, vga=20,
                      amp=False, **k):
        self.calls.append((lna, vga))
        gain_lin = 10.0 ** ((lna + vga) / 20.0)
        peak = min(127.0, (self.antenna_counts + self.noise_counts) * gain_lin)
        n = max(16, int(num_samples))
        phase = np.linspace(0, 40 * np.pi, n)
        iq = (peak / 127.0) * np.exp(1j * phase)
        return iq.astype(np.complex64)


def _run(antenna_counts):
    radio = FakeRadio(antenna_counts)
    band = {"center": 98e6}
    args = type("A", (), {"sample_rate": 2e6})()
    lna, vga, tried, status = collect.find_gain(radio, band, args)
    gain_lin = 10.0 ** ((lna + vga) / 20.0)
    final = min(127.0, (antenna_counts + 0.5) * gain_lin)
    return lna, vga, final, tried, status


# Spans a very weak antenna signal through one strong enough to need
# attenuation. The originally-shipped capture sat around the 1e-3 end.
@pytest.mark.parametrize("antenna_counts", [
    0.002, 0.01, 0.05, 0.2, 1.0, 3.0, 10.0, 40.0,
])
def test_gain_search_lands_in_target_window(antenna_counts):
    lna, vga, final, tried, status = _run(antenna_counts)
    assert status == "ok", f"status={status}"
    assert collect.TARGET_LO <= final <= collect.TARGET_HI, (
        f"antenna={antenna_counts} counts converged to lna={lna} vga={vga} "
        f"-> {final:.1f} counts, outside "
        f"[{collect.TARGET_LO}, {collect.TARGET_HI}] after {len(tried)} probes"
    )


def test_gain_search_never_silently_accepts_clipping():
    """If the level can't be brought below clipping, say so explicitly."""
    for antenna_counts in [0.002, 0.5, 5.0, 60.0]:
        _, _, final, _, status = _run(antenna_counts)
        assert final < collect.CLIP_COUNTS and status == "ok", (
            f"settled at {final:.1f} counts with status={status}")


def test_overloaded_input_is_reported_not_swallowed():
    """A signal too strong even at minimum gain needs external attenuation."""
    lna, vga, final, _, status = _run(400.0)
    assert status == "clipping", f"expected clipping, got {status}"
    assert (lna, vga) == (collect.LNA_STEPS[0], collect.VGA_STEPS[0]), (
        "should have wound gain all the way down before giving up")


def test_dead_antenna_is_reported_as_too_weak():
    radio = FakeRadio(0.0, noise_counts=0.0)
    band = {"center": 98e6}
    args = type("A", (), {"sample_rate": 2e6})()
    lna, vga, tried, status = collect.find_gain(radio, band, args)
    assert status == "too_weak", f"expected too_weak, got {status}"
    assert (lna, vga) == (collect.LNA_STEPS[-1], collect.VGA_STEPS[-1]), (
        "should have wound gain all the way up before giving up")


def test_gain_search_stays_on_the_hardware_grid():
    """LNA moves in 8 dB steps, VGA in 2 dB; off-grid values get snapped."""
    for antenna_counts in [0.01, 0.2, 3.0, 40.0]:
        lna, vga, _, tried, _s = _run(antenna_counts)
        assert lna in collect.LNA_STEPS, f"lna={lna} off grid"
        assert vga in collect.VGA_STEPS, f"vga={vga} off grid"
        for probed_lna, probed_vga in tried and [(t[0], t[1]) for t in tried]:
            assert probed_lna in collect.LNA_STEPS
            assert probed_vga in collect.VGA_STEPS


def test_gain_search_is_economical():
    """Each probe is a real USB capture, so the search should be short."""
    for antenna_counts in [0.002, 0.05, 1.0, 40.0]:
        _, _, _, tried, _s = _run(antenna_counts)
        assert len(tried) <= 8, f"took {len(tried)} probes to converge"


def test_band_presets_are_well_formed():
    required = {"center", "channel_bw", "sample_rate", "seconds", "desc",
                "expect_signal"}
    for name, band in collect.BANDS.items():
        missing = required - set(band)
        assert not missing, f"{name} missing {missing}"
        assert 1e6 <= band["center"] <= 6e9, f"{name} center out of range"
        assert 0 < band["channel_bw"] < band["sample_rate"] / 2, (
            f"{name} channel_bw must fit inside the captured span")


def test_noise_preset_is_marked_as_expecting_nothing():
    """The deliberate noise reference must not be treated as a failure."""
    noise = collect.BANDS["noise"]
    assert noise["expect_signal"] is False
    assert noise.get("skip_gain_search") is True
