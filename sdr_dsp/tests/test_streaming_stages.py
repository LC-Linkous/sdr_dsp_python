"""Streaming stages must reproduce the whole-record core functions.

The contract for every class in ``sdr_dsp.stream.stages``: feed a signal
through it in blocks -- awkward, uneven block sizes included -- concatenate
the outputs, and get the same samples ``sdr_dsp.core`` produces on the whole
record. The core function is the oracle; the stage only carries state.
"""

import numpy as np
import pytest

from sdr_dsp.core import (design_lowpass, fir_apply, frequency_shift, fm_demod,
                          resample_poly, deemphasis)
from sdr_dsp.stream.stages import (StreamingFIR, StreamingDecimator,
                                   StreamingMixer, StreamingFMDemod,
                                   StreamingResampler, StreamingDeemphasis)
from sdr_dsp.stream import Pipeline
from sdr_dsp.sources import ArraySource

FS = 200_000.0

# uneven, non-divisible block sizes, including a tiny one, to exercise the
# bookkeeping rather than the happy path of equal blocks
SIZES = [1000, 777, 3, 2048, 1, 4999, 1500]


def _blocks(x, sizes=SIZES):
    out, pos = [], 0
    while pos < len(x):
        for s in sizes:
            out.append(x[pos:pos + s])
            pos += s
            if pos >= len(x):
                break
    return [b for b in out if b.size]


def _run(stage, x, sizes=SIZES, flush=False):
    parts = [stage(b) for b in _blocks(x, sizes)]
    if flush:
        parts.append(stage.flush())
    return np.concatenate(parts)


@pytest.fixture
def fm_iq():
    rng = np.random.default_rng(7)
    n = 20_000
    t = np.arange(n) / FS
    msg = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 3700 * t)
    iq = np.exp(1j * 2 * np.pi * 20e3 * np.cumsum(msg) / FS)
    iq = iq + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return iq.astype(np.complex64)


# --- FIR ------------------------------------------------------------------

def test_streaming_fir_matches_whole_record_complex(fm_iq):
    taps = design_lowpass(30e3, FS, num_taps=129)
    ref = fir_apply(fm_iq, taps)
    got = _run(StreamingFIR(taps), fm_iq)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6)


def test_streaming_fir_matches_whole_record_real():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(12_345)
    taps = design_lowpass(5e3, 48e3, num_taps=101)
    np.testing.assert_allclose(_run(StreamingFIR(taps), x), fir_apply(x, taps),
                               atol=1e-12)


def test_streaming_fir_blocks_shorter_than_filter(fm_iq):
    taps = design_lowpass(30e3, FS, num_taps=257)
    got = _run(StreamingFIR(taps), fm_iq, sizes=[5, 17, 40])
    np.testing.assert_allclose(got, fir_apply(fm_iq, taps), atol=2e-6)


def test_streaming_fir_carries_history_across_a_dtype_change(fm_iq):
    # a source that hands over complex64 then complex128 is a caller-side
    # event, not a new stream; the delay line used to be silently ZEROED on
    # the dtype change, which put a filter-length click at the switch
    taps = design_lowpass(30e3, FS, num_taps=129)
    ref = fir_apply(fm_iq, taps)
    stage = StreamingFIR(taps)
    a = stage(fm_iq[:5000].astype(np.complex64))
    b = stage(fm_iq[5000:].astype(np.complex128))
    got = np.concatenate([a, b])
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6)
    # complex -> real would have to discard the imaginary history: refuse
    with pytest.raises(TypeError, match="complex to real"):
        stage(np.ones(10, dtype=np.float64))
    stage.reset()
    stage(np.ones(10, dtype=np.float64))     # fine after an explicit reset


def test_stateless_fir_per_block_is_actually_wrong(fm_iq):
    """The reason the stage exists: the naive pattern fails this same check."""
    taps = design_lowpass(30e3, FS, num_taps=129)
    naive = np.concatenate([fir_apply(b, taps) for b in _blocks(fm_iq)])
    assert not np.allclose(naive, fir_apply(fm_iq, taps), atol=1e-3)


# --- decimator ------------------------------------------------------------

@pytest.mark.parametrize("factor", [2, 5, 8, 13])
def test_streaming_decimator_keeps_the_grid(factor):
    x = np.arange(30_001, dtype=np.float64)
    got = _run(StreamingDecimator(factor), x)
    np.testing.assert_array_equal(got, x[::factor])


def test_streaming_decimator_survives_blocks_smaller_than_factor():
    x = np.arange(500, dtype=np.float64)
    got = _run(StreamingDecimator(8), x, sizes=[1, 2, 3, 5, 7])
    np.testing.assert_array_equal(got, x[::8])


# --- mixer ----------------------------------------------------------------

def test_streaming_mixer_is_phase_continuous(fm_iq):
    ref = frequency_shift(fm_iq, -37_500.0, FS)
    got = _run(StreamingMixer(-37_500.0, FS), fm_iq)
    np.testing.assert_allclose(got, ref, atol=5e-6)


def test_naive_mixer_per_block_has_phase_steps(fm_iq):
    ref = frequency_shift(fm_iq, -37_500.0, FS)
    naive = np.concatenate([frequency_shift(b, -37_500.0, FS)
                            for b in _blocks(fm_iq)])
    assert not np.allclose(naive, ref, atol=1e-3)


# --- FM demod -------------------------------------------------------------

def test_streaming_fm_demod_matches_whole_record(fm_iq):
    ref = fm_demod(fm_iq, 20e3, FS)
    got = _run(StreamingFMDemod(20e3, FS), fm_iq)
    assert got.shape == ref.shape        # whole record is one shorter than input
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_streaming_fm_demod_raw_radians(fm_iq):
    np.testing.assert_allclose(_run(StreamingFMDemod(), fm_iq), fm_demod(fm_iq),
                               atol=1e-9)


def test_streaming_fm_demod_no_seam_spike(fm_iq):
    """A missing carry shows up as one wild sample per block boundary."""
    got = _run(StreamingFMDemod(20e3, FS), fm_iq)
    assert np.max(np.abs(got)) < 2.0     # message peaks ~1.3 of deviation


# --- resampler ------------------------------------------------------------

@pytest.mark.parametrize("up,down", [(48_000, 200_000), (3, 2), (1, 4), (24, 125)])
def test_streaming_resampler_matches_whole_record(up, down):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(40_000)
    ref = resample_poly(x, up, down)
    st = StreamingResampler(up, down)
    got = _run(st, x, flush=True)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_streaming_resampler_complex_input():
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)).astype(np.complex64)
    ref = resample_poly(x, 1, 5)
    got = _run(StreamingResampler(1, 5), x, flush=True)
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_streaming_resampler_latency_is_small_and_reported():
    st = StreamingResampler(48_000, 400_000)
    assert 0 < st.latency_samples < 400_000 * 0.01      # under 10 ms of input


def test_streaming_resampler_unity_ratio_passthrough():
    x = np.arange(10.0)
    st = StreamingResampler(4, 4)
    np.testing.assert_array_equal(st(x), x)


# --- de-emphasis ----------------------------------------------------------

def test_streaming_deemphasis_matches_whole_record():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(30_000)
    ref = deemphasis(x, 48_000, tau_us=75.0)
    got = _run(StreamingDeemphasis(48_000, tau_us=75.0), x)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_streaming_deemphasis_other_tau():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(5_000)
    np.testing.assert_allclose(_run(StreamingDeemphasis(44_100, tau_us=50.0), x),
                               deemphasis(x, 44_100, tau_us=50.0), atol=1e-12)


# --- reset + pipeline integration -----------------------------------------

def test_reset_forgets_history(fm_iq):
    taps = design_lowpass(30e3, FS, num_taps=65)
    st = StreamingFIR(taps)
    st(fm_iq[:5000])
    st.reset()
    np.testing.assert_allclose(st(fm_iq[:3000]), fir_apply(fm_iq[:3000], taps),
                               atol=2e-6)


def test_full_fm_chain_in_a_pipeline_matches_whole_record(fm_iq):
    """The recommended live pattern, end to end: filter -> decimate -> demod
    -> resample, as Pipeline stages, equals the same chain on the whole record."""
    dec = 4
    taps = design_lowpass(30e3, FS, num_taps=101)
    inter = FS / dec

    whole = fir_apply(fm_iq, taps)[::dec]
    whole = fm_demod(whole, 20e3, inter)
    whole = resample_poly(whole, 48_000, int(inter))

    rs = StreamingResampler(48_000, int(inter))
    pipe = (Pipeline(ArraySource(fm_iq, FS, block_size=3333))
            .add(StreamingFIR(taps), "channel")
            .add(StreamingDecimator(dec), "decimate")
            .add(StreamingFMDemod(20e3, inter), "fm_demod")
            .add(rs, "audio"))
    out = [b for b in pipe.run() if b.size]
    out.append(rs.flush())
    got = np.concatenate(out)
    assert got.shape == whole.shape
    np.testing.assert_allclose(got, whole, atol=1e-5)


# ---------------------------------------------------------------------------
# StreamingPSD
# ---------------------------------------------------------------------------

def _noisy_tone(n, fs=1e6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = np.exp(2j * np.pi * 123_456.0 * t) + 0.05 * noise
    return x.astype(np.complex64)


@pytest.mark.parametrize("block_sizes", [[777], [1], [4096], [3, 1000, 64, 9999]])
def test_streaming_psd_equals_psd_on_the_same_samples(block_sizes):
    from sdr_dsp.core import psd
    from sdr_dsp.stream import StreamingPSD
    fs, nfft, avg = 1e6, 1024, 8
    x = _noisy_tone(nfft * avg * 3 + 500, fs)
    stage = StreamingPSD(nfft, avg=avg, sample_rate=fs)
    outs = []
    i = k = 0
    while i < len(x):
        bs = block_sizes[k % len(block_sizes)]
        r = stage(x[i:i + bs])
        if r is not None:
            outs.append(r)
        i += bs
        k += 1
    assert len(outs) == 3 and stage.frames_emitted == 3          # 500 leftover buffered
    for j, out in enumerate(outs):
        _, ref = psd(x[j * nfft * avg:(j + 1) * nfft * avg], fs, nfft=nfft, scaling="dbfs")
        np.testing.assert_array_equal(out, ref)                   # bit-equal, not close
    np.testing.assert_array_equal(stage.freqs, psd(x[:nfft], fs, nfft=nfft)[0])


def test_streaming_psd_reset_scaling_and_validation():
    from sdr_dsp.stream import StreamingPSD
    from sdr_dsp.core import psd
    fs, nfft = 1e6, 256
    x = _noisy_tone(nfft * 2, fs)
    stage = StreamingPSD(nfft, avg=1, sample_rate=fs)
    assert stage(x[:100]) is None
    stage.reset()                                                 # retune: drop the partial frame
    assert stage(x[:nfft - 1]) is None                            # 255 buffered, not 355
    out = stage(x[nfft - 1:nfft])
    np.testing.assert_array_equal(out, psd(x[:nfft], fs, nfft=nfft, scaling="dbfs")[1])
    # density scaling needs a sample rate; dbfs does not
    dens = StreamingPSD(nfft, sample_rate=fs, scaling="density")
    np.testing.assert_array_equal(dens(x[:nfft]), psd(x[:nfft], fs, nfft=nfft)[1])
    with pytest.raises(ValueError, match="sample_rate"):
        StreamingPSD(nfft, scaling="density")
    with pytest.raises(ValueError, match="avg"):
        StreamingPSD(nfft, avg=0)
    rateless = StreamingPSD(nfft)                                 # axis in cycles/sample
    assert rateless.freqs.min() == pytest.approx(-0.5)
    assert -1.5 < rateless(x[:nfft]).max() < 0.1          # 0 dBFS less scalloping


def test_streaming_psd_retune_relabels_axis_and_keeps_buffer():
    from sdr_dsp.stream import StreamingPSD
    fs, nfft = 1e6, 256
    x = _noisy_tone(nfft * 2, fs)
    stage = StreamingPSD(nfft, avg=2, sample_rate=fs, center_freq=100e6)
    assert stage(x[:300]) is None
    old_axis = stage.freqs.copy()
    stage.retune(101e6)                                   # display retune: keep samples
    np.testing.assert_allclose(stage.freqs, old_axis + 1e6)
    assert stage(x[300:]) is not None                     # 300 + 212 = 512 = avg*nfft
    stage.retune(102e6, drop_buffer=True)                 # RF retune: start clean
    assert stage(x[:nfft * 2 - 1]) is None and stage.center_freq == 102e6
