"""GainModel: auto-gain is not locked to the HackRF.

search_gain (the HackRF wrapper) is covered by test_gain_and_health.py. These
tests cover the abstraction it now delegates to -- GainStage/GainModel and the
generic search_gain_model engine -- and, crucially, prove the SAME engine drives
a NON-HackRF radio: a single-knob (RTL-SDR-like) tuner with no separate level
stage. The SNR-first guarantee (don't bless amplified noise) must hold there
too.
"""

import numpy as np
import pytest

from sdr_dsp.core import (GainStage, fm_pilot_excess_db,
                          hackrf_gain_model, rtlsdr_gain_model,
                          search_gain_model)

FS = 2_000_000


# ---------------------------------------------------------------------------
# 1. the abstraction
# ---------------------------------------------------------------------------
def test_hackrf_model_structure():
    m = hackrf_gain_model()
    assert {s.name for s in m.frontend} == {"lna", "amp"}
    assert [s.name for s in m.level] == ["vga"]
    assert m.level_stage().name == "vga"            # separate level knob


def test_hackrf_model_without_amp_drops_the_amp_stage():
    m = hackrf_gain_model(use_amp=False)
    assert {s.name for s in m.frontend} == {"lna"}


def test_rtlsdr_model_is_single_knob_no_level_stage():
    m = rtlsdr_gain_model()
    assert [s.name for s in m.frontend] == ["gain"]
    assert m.level == ()                            # no separate level stage
    assert m.level_stage().name == "gain"          # one knob does both


def test_gainstage_validates():
    with pytest.raises(ValueError):
        GainStage("x", (0, 1), "bogus")            # bad role
    with pytest.raises(ValueError):
        GainStage("x", (), "frontend")             # no steps


def test_amp_db_weight_used_for_tiebreak():
    m = hackrf_gain_model()
    amp = m.by_name("amp")
    assert amp.db(True) == 14.0 and amp.db(False) == 0.0


# ---------------------------------------------------------------------------
# 2. a NON-HackRF radio: one pre-ADC gain knob (RTL-SDR-like)
# ---------------------------------------------------------------------------
class MockTuner:
    """A single pre-ADC gain: signal and antenna noise scale with gain, while a
    fixed downstream (quantization) floor does not -- so SNR rises with gain
    until the antenna noise dominates, and too much gain clips. One knob sets
    both SNR and level, exactly the case the HackRF's separate VGA avoids."""

    def __init__(self, antenna_counts, ant_noise=0.02, floor=0.6, seed=3):
        self.ant = float(antenna_counts)
        self.an = float(ant_noise)
        self.floor = float(floor)
        rng = np.random.default_rng(seed)
        n = int(FS * 0.06)
        t = np.arange(n) / FS
        msg = 0.6 * np.sin(2 * np.pi * 500 * t) + 0.09 * np.sin(
            2 * np.pi * 19_000 * t)
        self._clean = np.exp(2j * np.pi * 75e3 * np.cumsum(msg) / FS)
        self._noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
                       ) / np.sqrt(2)
        self._floor = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
                       ) / np.sqrt(2)

    def probe(self, gain):
        g = 10 ** (gain / 20)
        iq = (self.ant * g * self._clean + self.an * g * self._noise
              + self.floor * self._floor) / 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1.0, lim)
                + 1j * np.clip(iq.imag, -1.0, lim)).astype(np.complex64)


def _quality(iq):
    return fm_pilot_excess_db(iq, FS, nfft=8192)


def _rtl_probe(radio):
    return lambda s: radio.probe(s["gain"])


def test_single_knob_auto_gain_converges():
    r = search_gain_model(_rtl_probe(MockTuner(1.0)), rtlsdr_gain_model(),
                          quality=_quality)
    assert r["status"] == "ok", r
    assert 45 <= r["counts"] <= 110, r
    assert r["quality_db"] is not None and r["quality_db"] > 6, r
    # chosen gain is a legal step of the model
    assert r["settings"]["gain"] in rtlsdr_gain_model().by_name("gain").steps


def test_single_knob_dead_antenna_is_not_ok():
    """The SNR-first guarantee on a second radio: level may fill the window
    with amplified noise, but the search must NOT bless it."""
    r = search_gain_model(_rtl_probe(MockTuner(0.0)), rtlsdr_gain_model(),
                          quality=_quality)
    assert r["status"] == "too_weak", r
    assert r["quality_db"] is None or r["quality_db"] < 6


def test_single_knob_weak_antenna_reports_too_weak():
    r = search_gain_model(_rtl_probe(MockTuner(0.05)), rtlsdr_gain_model(),
                          quality=_quality)
    assert r["status"] == "too_weak", r


def test_single_knob_overload_reports_clipping():
    r = search_gain_model(_rtl_probe(MockTuner(400.0)), rtlsdr_gain_model(),
                          quality=_quality)
    assert r["status"] == "clipping", r


def test_single_knob_probes_stay_on_grid():
    steps = set(rtlsdr_gain_model().by_name("gain").steps)
    r = search_gain_model(_rtl_probe(MockTuner(1.0)), rtlsdr_gain_model(),
                          quality=_quality)
    for s, _c in r["probes"]:
        assert s["gain"] in steps


# ---------------------------------------------------------------------------
# 3. the SAME engine drives the multi-stage HackRF model directly
# ---------------------------------------------------------------------------
class _HackRFish:
    """SNR depends on the front end (amp+LNA); the VGA amplifies signal and
    receiver noise together -- the property search_gain exists to respect."""

    def __init__(self, antenna_counts, seed=2):
        self.ant = float(antenna_counts)
        rng = np.random.default_rng(seed)
        n = int(FS * 0.06)
        t = np.arange(n) / FS
        msg = 0.6 * np.sin(2 * np.pi * 500 * t) + 0.09 * np.sin(
            2 * np.pi * 19_000 * t)
        self._clean = np.exp(2j * np.pi * 75e3 * np.cumsum(msg) / FS)
        self._noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
                       ) / np.sqrt(2)

    def probe(self, lna, vga, amp):
        g_fe = 10 ** ((lna + (14 if amp else 0)) / 20)
        g_vga = 10 ** (vga / 20)
        sig_c = self.ant * g_fe * g_vga
        noise_c = 1.5 * g_vga + 0.7
        iq = (sig_c * self._clean + noise_c * self._noise) / 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1.0, lim)
                + 1j * np.clip(iq.imag, -1.0, lim)).astype(np.complex64)


def test_engine_drives_hackrf_model_weak_antenna_enables_amp():
    radio = _HackRFish(0.05)
    r = search_gain_model(
        lambda s: radio.probe(s["lna"], s["vga"], s.get("amp", False)),
        hackrf_gain_model(), quality=_quality)
    assert r["status"] == "ok", r
    assert r["settings"]["amp"] is True, r
    assert 45 <= r["counts"] <= 110, r
