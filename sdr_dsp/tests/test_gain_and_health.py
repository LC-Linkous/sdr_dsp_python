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
    h = capture_health(_shaped_noise(int(FS * 0.5)), FS, channel_bw_hz=100e3)
    assert not h["ok"], h
    assert h["channel_excess_db"] is not None
    assert h["channel_excess_db"] < 3.0, (
        f"shaped noise measured {h['channel_excess_db']:+.1f} dB of channel "
        f"excess -- the anti-alias rolloff is being read as a carrier again")


def test_health_still_accepts_a_real_station():
    h = capture_health(_fm_station(int(FS * 0.5)), FS, channel_bw_hz=100e3)
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


# ---------------------------------------------------------------------------
# capture_health verdicts: the thresholds live here, consumers key off `verdict`
# ---------------------------------------------------------------------------

def _wgn(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _clipped(scale=0.8):
    x = _wgn() * scale
    return (np.clip(x.real, -1, 1) + 1j * np.clip(x.imag, -1, 1)).astype(np.complex64)


def test_capture_health_verdict_for_each_failure():
    from sdr_dsp.core import capture_health, HEALTH_VERDICTS
    assert HEALTH_VERDICTS == ("clipping", "under_driven", "stuck_dc", "no_carrier", "ok")
    cases = {
        "clipping": capture_health(_clipped(), 2e6),
        "under_driven": capture_health(_wgn() * 0.002, 2e6),
        "stuck_dc": capture_health(_wgn() * 0.02 + 0.3, 2e6),
        "no_carrier": capture_health(_wgn() * 0.1, 2e6, channel_bw_hz=50e3),
        "ok": capture_health(_wgn() * 0.1, 2e6),
    }
    for want, h in cases.items():
        assert h["verdict"] == want, f"{want}: got {h['verdict']} {h['reasons']}"
        assert h["ok"] == (want == "ok")
        assert bool(h["reasons"]) == (want != "ok")
        for key in ("clip_fraction", "adc_counts", "peak_dbfs", "dc_offset", "dc_ratio"):
            assert key in h


def test_capture_health_verdict_is_the_worst_failure():
    from sdr_dsp.core import capture_health
    # clipping AND a huge DC: clipping is reported (it cannot be undone), and
    # both reasons are listed
    h = capture_health(_clipped() + 1.5, 2e6)
    assert h["verdict"] == "clipping"
    assert len(h["reasons"]) == 2


def test_capture_health_metrics_and_thresholds():
    from sdr_dsp.core import capture_health
    x = _clipped()
    h = capture_health(x, 2e6)
    # clip_fraction counts samples at the converter's last code, either rail
    rail = 127 / 128
    expected = np.mean((x.real >= rail) | (x.real <= -1.0) | (x.imag >= rail) | (x.imag <= -1.0))
    assert h["clip_fraction"] == pytest.approx(expected)
    # the threshold is a parameter, so a consumer can be stricter or looser
    assert capture_health(x, 2e6, max_clip_fraction=0.5)["verdict"] == "ok"
    # a healthy direct-conversion capture carries a little DC and passes
    h = capture_health(_wgn() * 0.1 + 0.01, 2e6)
    assert h["verdict"] == "ok" and 0.05 < h["dc_ratio"] < 0.1
    assert capture_health(_wgn() * 0.1 + 0.01, 2e6, max_dc_ratio=0.05)["verdict"] == "stuck_dc"
    # a short capture with channel_bw_hz: spectral check "not computed", not failed
    h = capture_health(_wgn(1000) * 0.1, 2e6, channel_bw_hz=50e3)
    assert h["channel_excess_db"] is None and h["verdict"] == "ok"
    # empty
    h = capture_health(np.zeros(0, dtype=np.complex64), 2e6)
    assert h["verdict"] == "under_driven" and not h["ok"]


def test_shipped_capture_health_matches_the_hackrfpy_report():
    # the independent clip / DC numbers hackrfpy wrote at capture time are in
    # the sidecar's .report.json; our measurement of the same bytes must agree
    import json
    from pathlib import Path
    from sdr_dsp.core import capture_health
    from sdr_dsp.io import load_iq
    base = Path(__file__).resolve().parent.parent / "sample_data" / "hackrf_one"
    payload = base / "fm_98.1MHz_8Msps.iq"
    if not payload.exists():
        pytest.skip("raw capture not downloaded (tools/fetch_sample_data.py)")
    report = json.loads((base / "fm_98.1MHz_8Msps.report.json").read_text())["iq"]
    iq, _ = load_iq(payload, count=4_000_000)
    h = capture_health(iq, 8e6, channel_bw_hz=100e3)
    assert h["verdict"] == "ok"
    assert h["clip_fraction"] == pytest.approx(report["clip_fraction"], abs=1e-4)
    assert h["dc_offset"] == pytest.approx(report["dc_offset"], abs=5e-4)


def test_capture_health_clip_sees_both_int8_rails():
    # Two's complement is asymmetric: the rails are -128 (-> -1.0) and +127
    # (-> 0.9921875). Both must count as clipped, +126 must not. A first
    # version only caught the negative rail.
    from sdr_dsp.core import capture_health

    def as_iq(code, n=1000):
        return (np.full(n, code, dtype=np.int8).astype(np.float32) / 128 + 0j).astype(np.complex64)

    assert capture_health(as_iq(127), 2e6)["verdict"] == "clipping"
    assert capture_health(as_iq(-128), 2e6)["verdict"] == "clipping"
    assert capture_health(as_iq(126), 2e6)["clip_fraction"] == 0.0
    assert capture_health(as_iq(-127), 2e6)["clip_fraction"] == 0.0     # one code short
    # every code once: exactly the two rail codes are clipped
    codes = np.arange(-128, 128, dtype=np.int8)
    iq = (codes.astype(np.float32) / 128 + 1j * 0).astype(np.complex64)
    assert capture_health(iq, 2e6)["clip_fraction"] == pytest.approx(2 / 256)
    # and on the Q rail too
    iq = (0 + 1j * codes.astype(np.float32) / 128).astype(np.complex64)
    assert capture_health(iq, 2e6)["clip_fraction"] == pytest.approx(2 / 256)
