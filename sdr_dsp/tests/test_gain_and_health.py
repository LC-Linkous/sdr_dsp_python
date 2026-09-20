"""The failure mode that shipped the dead sample capture, fixed and pinned.

Three layers, each of which previously reported success on pure noise:

1. ``capture_health``'s carrier check compared the channel against the OUTER
   band edges -- which is exactly where the SDR's own baseband anti-alias
   filter rolls off, so an empty capture at healthy gain measured tens of dB
   of "excess" and passed. The check now references a mid-band annulus.
2. Nothing could say "that energy is not an FM station". The 19 kHz pilot
   check can: amplified noise fills an ADC window, but cannot manufacture a
   pilot standing above the FM noise triangle.
3. The gain searches walked the VGA -- which amplifies signal and receiver
   noise together -- until the LEVEL looked right, and never touched the RF
   amp, the one control that most improves what is actually there to record.
   ``search_gain`` chooses the front end by measured signal quality first.
"""

import numpy as np
import pytest
from scipy import signal as sig

from sdr_dsp.core import (capture_health, fm_pilot_excess_db, search_gain,
                          peak_counts)

FS = 2_000_000


def _shaped_noise(n, level=60 / 128, seed=0):
    """Receiver noise as an SDR records it: white noise through a baseband
    anti-alias filter (~0.75 * fs on a HackRF), edges rolled off."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = sig.lfilter(sig.firwin(129, 0.75), 1.0, x)
    return (x * level / np.max(np.abs(x.real))).astype(np.complex64)


def _fm_station(n, fs=FS, dev=75_000, pilot=0.09, amp=0.5, snr_db=35,
                seed=1):
    """A minimal broadcast-FM signal: one audio tone + the 19 kHz pilot."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    msg = 0.7 * np.sin(2 * np.pi * 700 * t) + pilot * np.sin(
        2 * np.pi * 19_000 * t)
    iq = amp * np.exp(2j * np.pi * dev * np.cumsum(msg) / fs)
    noise = _shaped_noise(n, level=1.0, seed=seed + 1)
    noise *= amp * 10 ** (-snr_db / 20) / np.sqrt(
        np.mean(np.abs(noise) ** 2))
    return (iq + noise).astype(np.complex64)


# ---------------------------------------------------------------------------
# 1. capture_health must not pass on anti-alias-shaped receiver noise
# ---------------------------------------------------------------------------
def test_health_rejects_antialias_shaped_noise():
    """The regression that mattered: an empty capture at healthy gain.

    With the old outer-edge reference this measured > +70 dB of "channel
    excess" and passed; the mid-band reference reads it flat.
    """
    h = capture_health(_shaped_noise(int(FS * 0.5)), FS, channel_bw=100e3)
    assert not h["ok"], h
    assert h["channel_excess_db"] is not None
    assert h["channel_excess_db"] < 3.0, (
        f"shaped noise measured {h['channel_excess_db']:+.1f} dB of channel "
        f"excess -- the anti-alias rolloff is being read as a carrier again")


def test_health_still_accepts_a_real_station():
    h = capture_health(_fm_station(int(FS * 0.5)), FS, channel_bw=100e3)
    assert h["ok"], h["reasons"]
    assert h["channel_excess_db"] > 10


def test_health_counts_match_loader_scale():
    """load_iq normalizes int8 by 128, so a full-scale sample must read as
    128 counts, not 126.99."""
    x = np.array([1.0 + 0j], dtype=np.complex64).repeat(16)
    h = capture_health(x, FS)
    assert h["adc_counts"] == pytest.approx(128.0)


# ---------------------------------------------------------------------------
# 2. the pilot check separates "energy" from "an FM broadcast station"
# ---------------------------------------------------------------------------
def test_pilot_found_on_a_station():
    p = fm_pilot_excess_db(_fm_station(int(FS * 0.5)), FS)
    assert p is not None and p > 20, p


def test_pilot_absent_on_noise_at_healthy_level():
    p = fm_pilot_excess_db(_shaped_noise(int(FS * 0.5)), FS)
    assert p is not None and p < 6, (
        f"noise measured a {p:+.1f} dB pilot")


def test_pilot_absent_on_unpiloted_fm():
    """FM energy without a pilot (e.g. NBFM) is energy, not broadcast FM."""
    p = fm_pilot_excess_db(_fm_station(int(FS * 0.5), pilot=0.0), FS)
    assert p is not None and p < 6, p


def test_pilot_returns_none_when_undecidable():
    assert fm_pilot_excess_db(_fm_station(1000), FS) is None      # too short
    assert fm_pilot_excess_db(_fm_station(200_000), 40_000) is None  # rate


# ---------------------------------------------------------------------------
# 3. search_gain: front end by SNR, VGA for level
# ---------------------------------------------------------------------------
class RadioModel:
    """A receive chain where SNR depends on the FRONT END, not the VGA.

    The antenna signal is amplified by amp+LNA+VGA; the receiver's own noise
    is injected after the front end, so it is amplified by the VGA alone
    (plus a floor). Raising the VGA therefore raises the level without
    improving what is there -- the property the old searches were blind to.
    """

    def __init__(self, antenna_counts, rx_noise_counts=1.5, seed=2):
        self.ant = float(antenna_counts)
        self.noise = float(rx_noise_counts)
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
        noise_c = self.noise * g_vga + 0.7
        iq = (sig_c * self._clean + noise_c * self._noise) / 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1.0, lim)
                + 1j * np.clip(iq.imag, -1.0, lim)).astype(np.complex64)


def _quality(iq):
    return fm_pilot_excess_db(iq, FS, nfft=8192)


def test_weak_antenna_enables_the_amp():
    """The whole point: on a weak antenna the +14 dB amp and a high LNA are
    what improve reception, and the search must find that on its own."""
    r = search_gain(RadioModel(0.05).probe, quality=_quality)
    assert r["status"] == "ok", r
    assert r["amp"] is True, r
    assert r["lna"] >= 32, r
    assert 45 <= r["counts"] <= 110, r
    assert r["quality_db"] > 20, r


def test_strong_antenna_leaves_the_amp_off():
    r = search_gain(RadioModel(30.0).probe, quality=_quality)
    assert r["status"] == "ok", r
    assert r["amp"] is False, r
    assert 45 <= r["counts"] <= 110, r


def test_noise_filling_the_window_is_not_ok():
    """Level convergence on a dead antenna must NOT report success: this is
    exactly how the original dead capture got recorded and blessed."""
    r = search_gain(RadioModel(0.0).probe, quality=_quality)
    assert r["status"] == "too_weak", r
    assert r["quality_db"] is None or r["quality_db"] < 6


def test_level_only_fallback_still_converges():
    """Without a quality metric (nothing known to be on the air), the level
    walk still works -- and may now escalate through LNA and amp."""
    r = search_gain(RadioModel(0.5).probe, quality=None)
    assert r["status"] == "ok"
    assert 45 <= r["counts"] <= 110


def test_search_stays_on_the_hardware_grid():
    r = search_gain(RadioModel(0.4).probe, quality=_quality)
    from sdr_dsp.core.gain_search import (DEFAULT_LNA_STEPS,
                                          DEFAULT_VGA_STEPS)
    for lna, vga, _amp, _c in r["probes"]:
        assert lna in DEFAULT_LNA_STEPS
        assert vga in DEFAULT_VGA_STEPS
    assert r["lna"] in DEFAULT_LNA_STEPS and r["vga"] in DEFAULT_VGA_STEPS


def test_peak_counts_is_per_component():
    x = np.array([0.5 + 0.5j] * 8, dtype=np.complex64)
    assert peak_counts(x) == pytest.approx(64.0)   # not 0.707 * 128
