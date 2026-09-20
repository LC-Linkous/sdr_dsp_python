"""The shipped sample and the pilot check, pinned against real-world physics.

The pilot check originally demodulated at the full capture rate. On clean
synthetic 2 Msps fixtures that worked; on a real, by-ear-verified 8 Msps
capture the out-of-channel noise and adjacent stations corrupted the
discriminator and the pilot measured +1.8 dB -- below threshold, so the
check would have REJECTED genuinely good captures. The fix channelizes
before discriminating. These tests reconstruct that failure shape
synthetically (wide capture, modest SNR, an adjacent station) so the
regression stays pinned without shipping 80 MB of reference IQ, and they
hold the shipped sample_data capture to the standards the collection tools
enforce.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sig

from sdr_dsp.core import capture_health, fm_pilot_excess_db
from sdr_dsp.io.sigmf import load_iq, read_meta

SAMPLE = (Path(__file__).resolve().parent.parent / "sample_data"
          / "fm_2Msps.iq")


def _fm(n, fs, offset_hz=0.0, amp=0.4, pilot=0.09, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    msg = (0.6 * np.sin(2 * np.pi * (500 + 70 * seed) * t)
           + pilot * np.sin(2 * np.pi * 19_000 * t))
    ph = 2 * np.pi * (75e3 * np.cumsum(msg) / fs + offset_hz * t)
    return amp * np.exp(1j * ph), rng


def _wide_capture(fs=8e6, snr_db=18, adjacent=True, seconds=0.35):
    """The real-capture shape: wide rate, modest SNR, a neighbor at 800 kHz.

    At 8 Msps the discriminator sees 40x the channel's noise bandwidth, and
    an adjacent station beats against the tuned one -- the conditions that
    buried the pilot for the unchannelized check.
    """
    n = int(fs * seconds)
    iq, rng = _fm(n, fs)
    if adjacent:
        neighbor, _ = _fm(n, fs, offset_hz=800e3, amp=0.5, seed=3)
        iq = iq + neighbor
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    noise = sig.lfilter(sig.firwin(129, 0.75), 1.0, noise)
    noise *= 0.4 * 10 ** (-snr_db / 20) / np.sqrt(np.mean(np.abs(noise) ** 2))
    return (iq + noise).astype(np.complex64)


def test_pilot_survives_a_wide_noisy_capture_with_neighbors():
    """The regression: a receivable station in a realistic wide capture must
    measure a clear pilot. Unchannelized, this measured ~+2 dB and failed."""
    p = fm_pilot_excess_db(_wide_capture(), 8e6)
    assert p is not None and p > 10, (
        f"pilot read {p} dB on a good wide capture -- the check is being "
        f"fed out-of-channel energy again")


def test_pilot_still_absent_on_wide_noise():
    """Channelizing must not manufacture pilots from noise."""
    rng = np.random.default_rng(1)
    n = int(8e6 * 0.35)
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    noise = sig.lfilter(sig.firwin(129, 0.75), 1.0, noise).astype(np.complex64)
    p = fm_pilot_excess_db(noise, 8e6)
    assert p is not None and p < 6, p


def test_pilot_unmoved_by_an_adjacent_station_alone():
    """A neighbor at 800 kHz with NO station on-channel must not read as a
    pilot on the tuned channel."""
    n = int(8e6 * 0.35)
    neighbor, rng = _fm(n, 8e6, offset_hz=800e3, amp=0.5, seed=3)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.02
    p = fm_pilot_excess_db((neighbor + noise).astype(np.complex64), 8e6)
    assert p is not None and p < 6, p


def test_pilot_agrees_across_rates_for_the_same_station():
    """The measurement is a property of the station, not the capture rate:
    the same signal at 2 and 8 Msps must score within a few dB."""
    fs_hi = 8e6
    n = int(fs_hi * 0.35)
    iq, rng = _fm(n, fs_hi, amp=0.4)
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    noise *= 0.4 * 10 ** (-25 / 20) / np.sqrt(np.mean(np.abs(noise) ** 2))
    wide = (iq + noise).astype(np.complex64)
    taps = sig.firwin(65, 0.2)
    narrow = sig.lfilter(taps, 1.0, wide)[::4].astype(np.complex64)
    p_hi = fm_pilot_excess_db(wide, fs_hi)
    p_lo = fm_pilot_excess_db(narrow, fs_hi / 4)
    assert abs(p_hi - p_lo) < 6, (p_hi, p_lo)


# ---------------------------------------------------------------------------
# the shipped sample must be above suspicion
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample_data not present")
def test_shipped_sample_is_a_validated_fm_capture():
    """Everything downstream is debugged against this file, and it has been
    a silent noise capture once already. It must pass the same checks the
    collection tools enforce, forever."""
    iq, _meta = load_iq(str(SAMPLE))
    h = capture_health(iq, 2e6, channel_bw=100e3)
    assert h["ok"], h["reasons"]
    p = fm_pilot_excess_db(iq, 2e6)
    assert p is not None and p > 6, (
        f"shipped sample has no stereo pilot ({p} dB)")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample_data not present")
def test_shipped_sample_declares_its_provenance():
    """The sidecar must say where the capture came from and what was
    measured, so 'trust me' never has to be said about sample data again."""
    meta = read_meta(str(SAMPLE))
    g = meta["global"]
    assert g.get("core:description"), "sidecar has no description"
    freq = meta["captures"][0].get("core:frequency", 0)
    assert 87e6 <= freq <= 108e6, f"frequency {freq} is not in the FM band"


# ---------------------------------------------------------------------------
# public get_window (cleanup pass): examples no longer import the _ alias
# ---------------------------------------------------------------------------
def test_public_get_window_matches_numpy_conventions():
    from sdr_dsp.core import get_window
    assert np.allclose(get_window("hann", 32), np.hanning(32))
    assert np.allclose(get_window("hamming", 16), np.hamming(16))
    assert np.allclose(get_window(None, 8), np.ones(8))
    passed = np.array([1.0, 2.0, 3.0])
    assert get_window(passed, 3) is passed        # arrays pass through
