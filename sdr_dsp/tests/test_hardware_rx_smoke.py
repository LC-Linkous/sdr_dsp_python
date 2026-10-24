"""Hardware RX smoke: a real HackRF, a real station, through the library chain.

Receive only -- nothing here transmits. Auto-skips without a board or the
hackrf-tools binaries, so it is harmless in CI; on the bench it turns "did
the receive path actually work today" into one command:

    uv run pytest -m hardware tests/test_hardware_rx_smoke.py -v

Pick the station with SDR_DSP_TEST_FM_HZ (defaults to the 103.7 MHz capture
the shipped sample came from); set SDR_DSP_TEST_LNA / _VGA / _AMP=1 to
override the gains (default 32 / 28 / off, the hackrfpy 1.1.0 collection
defaults). A failing pilot test with healthy counts means the frequency is
wrong for your location, not that the chain is broken.

What is asserted, in order of what each proves:

1. hackrfpy detects a board (tools present, USB works).
2. The reference IQSource adapter (examples/hackrf_capture.py) opens,
   streams, closes, and the samples are complex64 in [-1, 1).
3. capture_health passes at the chosen gains: the ADC is driven (not a 3-bit
   capture) and the channel stands out of the noise -- the two failure modes
   that produced dead sample data in September.
4. fm_pilot_excess_db > 6 dB: the 19 kHz pilot is there, so this is a real
   FM broadcast and not amplified noise. Amplified noise can hump a channel;
   it cannot manufacture a pilot.
5. The streaming chain (StreamingFIR -> decimate -> StreamingFMDemod) on the
   live blocks produces audio with the broadcast low-frequency tilt, the
   same property test_real_demod_e2e proves on the recorded sample.
6. The pipe path and the file path agree: pilot excess from blocks() over
   the pipe is within 6 dB of a file-path probe at the same settings. This
   is the A/B from the 2026-09-20 log, automated. A large gap here is the
   Windows pipe corruption; an I/Q swap would show as a mirrored channel and
   a vanished pilot.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sdr_dsp.core import (capture_health, fm_pilot_excess_db, design_lowpass,
                          psd)
from sdr_dsp.sources.probe import probe_capture
from sdr_dsp.stream import StreamingFIR, StreamingDecimator, StreamingFMDemod

pytestmark = pytest.mark.hardware

hackrfpy = pytest.importorskip("hackrfpy")

FREQ = float(os.environ.get("SDR_DSP_TEST_FM_HZ", 103.7e6))
RATE = 2_000_000
LNA = int(os.environ.get("SDR_DSP_TEST_LNA", 32))
VGA = int(os.environ.get("SDR_DSP_TEST_VGA", 28))
AMP = os.environ.get("SDR_DSP_TEST_AMP", "0") == "1"
SECONDS = 1.0
CHANNEL_BW = 200e3


@pytest.fixture(scope="module")
def radio():
    from hackrfpy import HackRF
    h = HackRF(verbose=False)
    det = h.detect()
    if not det["ready"]:
        pytest.skip(f"no usable HackRF: {det['problem']}")
    return h


@pytest.fixture(scope="module")
def HackRFCapture():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    try:
        from hackrf_capture import HackRFCapture
    finally:
        sys.path.pop(0)
    return HackRFCapture


@pytest.fixture(scope="module")
def live_iq(radio, HackRFCapture):
    """One second of IQ from the adapter's blocks() -- the pipe path."""
    got, have = [], 0
    with HackRFCapture(FREQ, RATE, lna=LNA, vga=VGA, amp=AMP) as src:
        assert src.sample_rate == RATE and src.center_freq == FREQ
        for blk in src.blocks():
            got.append(blk)
            have += len(blk)
            if have >= SECONDS * RATE:
                break
    return np.concatenate(got)[:int(SECONDS * RATE)]


@pytest.fixture(scope="module")
def file_iq(radio):
    """The same second via the file path (sources/probe.py), the trusted one."""
    return probe_capture(radio, FREQ, RATE, int(SECONDS * RATE),
                         lna=LNA, vga=VGA, amp=AMP)


def test_board_detected(radio):
    det = radio.detect()
    assert det["ready"] and det["boards"], det


def test_adapter_streams_normalized_complex64(live_iq):
    assert live_iq.dtype == np.complex64
    assert len(live_iq) == int(SECONDS * RATE)
    assert np.max(np.abs(live_iq.real)) <= 1.0 and np.max(np.abs(live_iq.imag)) <= 1.0
    assert np.all(np.isfinite(live_iq))


def test_capture_is_healthy_at_these_gains(file_iq):
    h = capture_health(file_iq, RATE, channel_bw=CHANNEL_BW)
    assert h["ok"], (f"capture_health failed at lna={LNA} vga={VGA} amp={AMP}: {h}. "
                     "Under-driven ADC or no channel energy -- adjust gains or "
                     "SDR_DSP_TEST_FM_HZ.")


def test_station_has_a_stereo_pilot(file_iq):
    db = fm_pilot_excess_db(file_iq, RATE)
    assert db > 6.0, (f"pilot excess {db:+.1f} dB at {FREQ/1e6:.1f} MHz: no FM "
                      "broadcast here. Set SDR_DSP_TEST_FM_HZ to a local station.")


def test_streaming_chain_recovers_broadcast_audio(live_iq):
    dec = 8                                 # 2 Msps -> 250 kHz
    taps = design_lowpass(CHANNEL_BW / 2, RATE, num_taps=201)
    fir, decim, demod = StreamingFIR(taps), StreamingDecimator(dec), \
        StreamingFMDemod(75e3, RATE / dec)
    audio = np.concatenate([demod(decim(fir(b)))
                            for b in np.array_split(live_iq, 16)])
    assert len(audio) > 0 and np.all(np.isfinite(audio))
    assert np.std(audio) > 1e-3, "demodulated silence"
    # broadcast program audio tilts toward low frequencies; noise is flat
    f, p = psd(audio, RATE / dec, nfft=4096)
    low = np.mean(p[(f > 100) & (f < 3_000)])
    mid = np.mean(p[(f > 8_000) & (f < 15_000)])
    assert low - mid > 4.0, f"no low-frequency tilt (low-mid = {low-mid:.1f} dB)"


def test_pipe_path_matches_file_path(live_iq, file_iq):
    """The 2026-09-20 A/B, automated. Both captures are of the same station
    seconds apart at identical settings; their pilot excess should agree."""
    pipe_db = fm_pilot_excess_db(live_iq, RATE)
    file_db = fm_pilot_excess_db(file_iq, RATE)
    assert abs(pipe_db - file_db) < 6.0, (
        f"pipe path pilot {pipe_db:+.1f} dB vs file path {file_db:+.1f} dB. "
        "A large gap means the stdout-pipe path is corrupting samples "
        "(see sources/probe.py and docs/TODO.md).")
