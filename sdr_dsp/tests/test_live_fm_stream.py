"""The live receive chain must produce the same audio block-by-block as it
does in one pass, and must run faster than real time.

Both properties are easy to lose and neither is visible from a code read.
A block boundary is not a signal boundary: every stage in the chain carries
state, and restarting any of them once per block puts a periodic artifact in
the audio at the block rate. Meanwhile a chain that is correct but slower
than real time stutters regardless.

These run against synthetic IQ. No radio, no sound card.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "live_fm_listen.py"


def _load():
    # the example imports its HackRF helper from the examples/ directory
    sys.path.insert(0, str(EXAMPLE.parent))
    spec = importlib.util.spec_from_file_location("_live_fm_listen", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_live_fm_listen"] = mod
    spec.loader.exec_module(mod)
    return mod


lfl = _load()


def _fm_iq(seconds=2.0, tones=(440.0, 1000.0), amp=100.0, seed=0):
    """Synthetic broadcast-FM IQ, quantized to 8 bits like a real capture."""
    fs = lfl.SAMPLE_RATE
    n = int(fs * seconds)
    t = np.arange(n) / fs
    msg = 0.7 * np.sin(2 * np.pi * tones[0] * t)
    if len(tones) > 1:
        msg = msg + 0.3 * np.sin(2 * np.pi * tones[1] * t)
    msg /= np.abs(msg).max()
    z = np.exp(1j * 2 * np.pi * lfl.FM_DEVIATION * np.cumsum(msg) / fs) * amp
    rng = np.random.default_rng(seed)
    z = z + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.4
    return ((np.round(z.real) + 1j * np.round(z.imag)) / 127.0
            ).astype(np.complex64)


def _blockwise(iq, block=None):
    chain = lfl.FMStream()
    block = block or lfl.BLOCK_SAMPLES
    parts = [chain.process(iq[i:i + block]) for i in range(0, len(iq), block)]
    return np.concatenate([p for p in parts if p.size])


# --------------------------------------------------------------------------
def test_blockwise_matches_single_pass():
    """The core property: streaming must not change the audio."""
    iq = _fm_iq()
    ref = lfl.FMStream().process(iq)
    blk = _blockwise(iq)
    m = min(len(ref), len(blk))
    assert m > 40_000, "not enough audio produced to judge"
    err = np.abs(ref[:m] - blk[:m])
    rms_sig = np.sqrt(np.mean(ref[:m] ** 2))
    assert err.max() < 1e-5, (
        f"max deviation {err.max():.2e} between streamed and single-pass "
        "audio; a stage is not carrying its state across blocks")
    assert np.sqrt(np.mean(err ** 2)) / rms_sig < 1e-6


def test_no_artifact_concentrated_at_block_boundaries():
    """A periodic tick at the block rate is the failure this guards against.

    Averages hide it: a one-sample error every block can leave the overall
    RMS looking fine while being clearly audible at eight ticks a second. So
    compare error near the seams against error everywhere else.
    """
    iq = _fm_iq()
    ref = lfl.FMStream().process(iq)
    blk = _blockwise(iq)
    m = min(len(ref), len(blk))
    err = np.abs(ref[:m] - blk[:m])

    per_block = lfl.BLOCK_SAMPLES * lfl.AUDIO_RATE // lfl.SAMPLE_RATE
    near = np.zeros(m, bool)
    for k in range(m // per_block + 2):
        b = per_block * k
        near[max(0, b - 8):min(m, b + 8)] = True
    if not near.any() or not (~near).any():
        pytest.skip("signal too short to separate boundaries")

    at_seam = np.sqrt(np.mean(err[near] ** 2))
    elsewhere = np.sqrt(np.mean(err[~near] ** 2)) + 1e-30
    assert at_seam / elsewhere < 10.0, (
        f"error at block boundaries is {at_seam/elsewhere:.0f}x the error "
        "elsewhere -- a stage restarts each block and will tick audibly")


@pytest.mark.parametrize("block", [50_000, 100_000, 250_000, 500_000])
def test_result_is_independent_of_block_size(block):
    """Block size is an I/O choice and must not change the audio."""
    iq = _fm_iq(seconds=1.5)
    ref = lfl.FMStream().process(iq)
    blk = _blockwise(iq, block=block)
    m = min(len(ref), len(blk))
    assert np.abs(ref[:m] - blk[:m]).max() < 1e-5


def test_recovers_the_modulating_tones():
    """End to end: the audio must actually contain the transmitted tones."""
    audio = _blockwise(_fm_iq(tones=(440.0, 1000.0)))
    n = 8192
    seg = audio[len(audio) // 2:len(audio) // 2 + n]
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / lfl.AUDIO_RATE)
    peaks = {round(freqs[i] / 10) * 10 for i in np.argsort(spec)[-8:]}
    assert any(abs(p - 440) <= 20 for p in peaks), f"no 440 Hz tone: {peaks}"
    assert any(abs(p - 1000) <= 20 for p in peaks), f"no 1 kHz tone: {peaks}"


def test_deemphasis_is_applied():
    """Broadcast FM is pre-emphasized; without de-emphasis it sounds harsh.

    A 75 us time constant is a one-pole rolloff with its corner near 2.1 kHz,
    so a 6 kHz tone should come out clearly below a 500 Hz one of equal
    transmitted amplitude. Comparing two tones through the same chain tests
    the shaping without depending on absolute level.
    """
    def tone_level(hz):
        audio = _blockwise(_fm_iq(seconds=1.0, tones=(hz,)))
        seg = audio[len(audio) // 2:len(audio) // 2 + 8192]
        spec = np.abs(np.fft.rfft(seg * np.hanning(8192)))
        freqs = np.fft.rfftfreq(8192, 1 / lfl.AUDIO_RATE)
        return spec[np.argmin(np.abs(freqs - hz))]

    low, high = tone_level(500.0), tone_level(6000.0)
    rolloff_db = 20 * np.log10(low / (high + 1e-30))
    assert rolloff_db > 6.0, (
        f"6 kHz is only {rolloff_db:.1f} dB below 500 Hz; de-emphasis looks "
        "absent, and the audio will sound harsh")


def test_runs_faster_than_real_time():
    """A correct chain that cannot keep up still stutters.

    Decimating before the fractional resample is what buys the headroom:
    going straight from 2 Msps to 48 kHz needs up=3, down=125, which pushes
    a 2501-tap filter at 6 Msps and lands well over 1x.
    """
    seconds = 2.0
    iq = _fm_iq(seconds=seconds)
    chain = lfl.FMStream()
    b = lfl.BLOCK_SAMPLES
    t0 = time.perf_counter()
    for i in range(0, len(iq), b):
        chain.process(iq[i:i + b])
    ratio = (time.perf_counter() - t0) / seconds
    # Generous bound: CI machines are slow and shared. The design measures
    # around 0.2x, and the arrangement it replaced was about 2.4x.
    assert ratio < 0.8, f"{ratio:.2f}x real time -- audio would stutter"


def test_decimation_keeps_the_channel_inside_nyquist():
    chain = lfl.FMStream()
    assert lfl.CHANNEL_BW < chain.inter_fs / 2, (
        "the channel filter passes energy above the decimated Nyquist rate, "
        "which will alias into the audio")
    assert lfl.BLOCK_SAMPLES % lfl.DECIMATION == 0, (
        "block length must divide by the decimation factor or the sampling "
        "phase shifts between blocks")


def test_handles_short_and_empty_blocks():
    chain = lfl.FMStream()
    assert chain.process(np.zeros(0, dtype=np.complex64)).size == 0
    assert chain.process(np.zeros(3, dtype=np.complex64)).size == 0
    iq = _fm_iq(seconds=0.5)
    out = [chain.process(iq[i:i + 7_001]) for i in range(0, len(iq), 7_001)]
    assert sum(o.size for o in out) > 0, "ragged block sizes produced nothing"
