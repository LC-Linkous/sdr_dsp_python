"""Edge-case and alternative-branch coverage for sdr_dsp.core.

Each test here targets a line the main suite never reached: an empty-input
guard, an argument-validation raise, or a real alternative branch (LSB
modulation, the odd-length Hilbert kernel, the RRC singularity tap, explicit
decision levels) that had no test of its own. The contract under test is the
documented behaviour at the boundary -- an empty array in means an empty array
out of the right dtype, a bad argument raises ValueError with a readable
message -- not the DSP itself, which the per-module tests cover.
"""

import numpy as np
import pytest

from sdr_dsp.core.filters import (design_lowpass, design_bandpass,
                                  design_highpass, fir_apply_centered)
from sdr_dsp.core.channel import add_noise, add_cfo
from sdr_dsp.core.channelize import channelize_bank, _oversample_time
from sdr_dsp.core.calibrate import Calibration
from sdr_dsp.core.demod.analog import (ssb_demod, dsb_sc_demod, cw_decode,
                                       _analytic_phase)
from sdr_dsp.core.demod.ask import nask_slice
from sdr_dsp.core.demod.fsk import fsk_demod_nlevel
from sdr_dsp.core.demod.psk import dqpsk_demod, psk8_demod
from sdr_dsp.core.demod.spread import dsss_despread
from sdr_dsp.core.demod.timing import estimate_symbol_rate, slice_to_symbols
from sdr_dsp.core.features.evm import evm_stats, EVM_FEATURE_NAMES
from sdr_dsp.core.features.impairments import (iq_image_ratio,
                                               estimate_iq_imbalance)
from sdr_dsp.core.framing import build_frame, find_frames, DEFAULT_SYNC
from sdr_dsp.core.modulate.analog import ssb_modulate
from sdr_dsp.core.modulate.shaping import rrc_taps
from sdr_dsp.core.spectral import spectrogram
from sdr_dsp.core.sync import _lock_trace, carrier_recovery, symbol_sync
from sdr_dsp.core.measure import (snr_db, fm_pilot_excess_db,
                                  find_bursts,
                                  estimate_cfo, estimate_fm_cfo)
from sdr_dsp.core.gain_search import peak_counts, GainModel, GainStage

FS = 1_000_000.0
EMPTY = np.zeros(0, dtype=np.complex64)


def _tone(f_hz, n=4096, fs=FS):
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f_hz * t).astype(np.complex64)


# -- filters -----------------------------------------------------------------

@pytest.mark.parametrize("lo,hi", [(0.0, 1e3), (2e3, 1e3), (1e3, FS)])
def test_design_bandpass_rejects_bad_band(lo, hi):
    with pytest.raises(ValueError, match="need 0 <"):
        design_bandpass(lo, hi, FS)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, FS / 2, FS])
def test_design_highpass_rejects_cutoff_outside_nyquist(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        design_highpass(cutoff, FS)


def test_fir_apply_centered_complex_filters_each_rail():
    # A zero-phase ("same") convolution of a tone through a lowpass that
    # passes it must return the tone itself, as complex64, same length.
    taps = design_lowpass(100e3, FS, num_taps=31)
    x = _tone(10e3, n=512)
    y = fir_apply_centered(x, taps)
    assert y.dtype == np.complex64 and y.shape == x.shape
    core = slice(32, -32)                        # away from the edge transients
    assert np.allclose(y[core], x[core], atol=0.02)


# -- channel -----------------------------------------------------------------

def test_add_noise_passes_empty_and_silent_input_through():
    assert add_noise(EMPTY, 10.0).size == 0
    silent = np.zeros(64, dtype=np.complex64)
    out = add_noise(silent, 10.0)
    assert out is not None and np.all(out == 0)   # no power -> nothing to scale


def test_add_cfo_empty_input():
    out = add_cfo(EMPTY, 1e3, FS)
    assert out.size == 0 and out.dtype == np.complex64


# -- channelize --------------------------------------------------------------

def test_channelize_bank_without_freqs_returns_pair():
    iq = _tone(50e3, n=8192)
    result = channelize_bank(iq, FS, n_channels=4, return_freqs=False)
    assert len(result) == 2
    channels, new_rate = result
    assert channels.shape[0] == 4 and new_rate == pytest.approx(FS / 4)


def test_oversample_time_identity_cases():
    ch = np.ones((2, 8), dtype=np.complex64)
    assert _oversample_time(ch, 1) is ch                     # factor 1: no-op
    short = np.ones((2, 1), dtype=np.complex64)
    assert _oversample_time(short, 2) is short               # nothing to interpolate


# -- calibrate ---------------------------------------------------------------

def test_calibration_repr_with_and_without_frequency():
    with_f = Calibration(offset_db=-3.25, frequency_hz=100e6)
    assert "100MHz" in repr(with_f) and "-3.2dB" in repr(with_f)
    no_f = Calibration(offset_db=1.0, frequency_hz=0.0)
    assert "no-freq" in repr(no_f)


# -- analog demod ------------------------------------------------------------

def test_ssb_demod_empty_input_returns_empty_float():
    out = ssb_demod(EMPTY, FS)
    assert out.size == 0 and out.dtype == np.float64


def test_ssb_demod_bfo_shifts_the_recovered_tone():
    # A 2 kHz baseband tone demodulated with a +1 kHz BFO lands at 3 kHz.
    iq = _tone(2e3, n=16384)
    audio = ssb_demod(iq, FS, sideband="usb", bfo_hz=1e3)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    f_peak = np.argmax(spec) * FS / len(audio)
    assert f_peak == pytest.approx(3e3, abs=FS / len(audio) * 2)


def test_dsb_sc_demod_bfo_branch_runs_and_preserves_length():
    iq = _tone(2e3, n=4096)
    out = dsb_sc_demod(iq, FS, bfo_hz=500.0)
    assert out.shape == iq.shape and out.dtype == np.float64
    assert np.all(np.isfinite(out))


def test_analytic_phase_odd_length_kernel():
    # The odd-N branch of the Hilbert kernel: the analytic signal of a real
    # cosine must have (nearly) one-sided spectrum, i.e. its phase advances
    # linearly at the tone frequency.
    n = 1001                                     # odd
    t = np.arange(n)
    x = np.cos(2 * np.pi * 0.05 * t)
    phase = _analytic_phase(x)
    assert phase.shape == (n,)
    dphi = np.diff(np.unwrap(phase))[50:-50]     # ignore the circular edges
    assert np.allclose(dphi, 2 * np.pi * 0.05, atol=0.02)


def test_cw_decode_word_gap_inserts_space():
    # "E E" = dit, letter gap (3 units), dit, WORD gap (7 units), dit -> "EE E".
    sps = 4
    units = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    bits = np.repeat(np.array(units, dtype=np.uint8), sps)
    bits = np.concatenate([np.zeros(sps * 3, np.uint8), bits,
                           np.zeros(sps * 3, np.uint8)])
    text = cw_decode(bits, sps)
    assert " " in text
    assert text.replace(" ", "") == "EEE"


# -- digital demod -----------------------------------------------------------

def test_nask_slice_explicit_levels():
    env = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    sym = nask_slice(env, levels=[0.0, 0.5, 1.0])
    assert list(sym) == [0, 0, 1, 2, 2]


def test_fsk_demod_nlevel_empty_and_explicit_thresholds():
    assert fsk_demod_nlevel(EMPTY, FS).size == 0
    # explicit centers: a tone at -30 kHz maps to index 0, +30 kHz to the last
    centers = [-30e3, -10e3, 10e3, 30e3]
    lo = fsk_demod_nlevel(_tone(-30e3, n=512), FS, thresholds=centers)
    hi = fsk_demod_nlevel(_tone(30e3, n=512), FS, thresholds=centers)
    assert np.all(lo[8:-8] == 0) and np.all(hi[8:-8] == 3)


def test_dqpsk_demod_needs_two_symbols():
    bits, phases = dqpsk_demod(np.array([1 + 0j], dtype=np.complex64))
    assert bits.size == 0 and phases.size == 0
    bits, phases = dqpsk_demod(EMPTY)
    assert bits.size == 0 and phases.size == 0


def test_psk8_demod_empty_input():
    bits, syms = psk8_demod(EMPTY)
    assert bits.size == 0 and syms.size == 0
    assert bits.dtype == np.uint8 and syms.dtype == np.uint8


def test_dsss_despread_samples_per_chip_repeats_code():
    code = np.array([1, -1, 1, -1], dtype=np.complex64)
    spc = 3
    chips = np.repeat(code, spc)
    iq = np.tile(chips, 5).astype(np.complex64)   # 5 spread symbols, all "+1"
    out = dsss_despread(iq, code, samples_per_chip=spc)
    assert len(out) == 5
    assert np.allclose(np.real(out) / np.real(out[0]), 1.0)


def test_estimate_symbol_rate_degenerate_inputs():
    # no bits -> no runs -> (0, 0) rather than a division by zero
    assert estimate_symbol_rate(np.zeros(0, dtype=np.uint8), FS) == (0.0, 0.0)
    # every interior run shorter than min_run -> falls back to the raw runs,
    # still returns a finite positive estimate rather than raising
    bits = np.array([0] * 10 + [1, 0, 1, 0, 1, 0] + [0] * 10, dtype=np.uint8)
    rate, spb = estimate_symbol_rate(bits, FS, min_run=5)
    assert rate > 0 and spb > 0


def test_slice_to_symbols_nonpositive_period_returns_bits():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    out = slice_to_symbols(bits, 0)
    assert out.dtype == np.uint8 and list(out) == [1, 0, 1, 1]


# -- features ----------------------------------------------------------------

def test_evm_stats_normalize_requires_reference():
    rx = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
    with pytest.raises(ValueError, match="normalize=True requires ref_symbols"):
        evm_stats(rx, normalize=True)


def test_evm_stats_normalize_divides_by_reference_rms():
    ref = np.array([2 + 0j, -2 + 0j, 0 + 2j, 0 - 2j], dtype=np.complex64)
    err = ref * 0.1                                # 10 % gain error, |ref| = 2
    normed = evm_stats(err, ref_symbols=ref, normalize=True)
    raw = evm_stats(err, ref_symbols=ref, normalize=False)
    i_rms = EVM_FEATURE_NAMES.index("evm_rms")
    assert raw[i_rms] == pytest.approx(0.2, rel=1e-6)
    assert normed[i_rms] == pytest.approx(0.1, rel=1e-6)   # divided by ref RMS 2


def test_impairment_estimators_on_silence():
    silent = np.zeros(128, dtype=np.complex64)
    assert iq_image_ratio(silent) == 0.0
    assert estimate_iq_imbalance(silent) == (0.0, 0.0)


# -- framing -----------------------------------------------------------------

def test_find_frames_stops_at_truncated_length_byte():
    # Sync word present, but the stream ends before the length byte is
    # complete: find_frames must return nothing rather than index past the end.
    frame = build_frame(b"ok")
    sync_len = len(DEFAULT_SYNC) * 8
    preamble_len = len(frame) - sync_len - 8 - 2 * 8 - 16   # see build_frame
    cut = frame[:preamble_len + sync_len + 3]               # 3 bits of length
    assert find_frames(cut) == []


# -- modulate ----------------------------------------------------------------

def test_ssb_modulate_lsb_is_conjugate_of_usb():
    msg = np.sin(2 * np.pi * 0.01 * np.arange(512))
    usb = ssb_modulate(msg, sideband="usb")
    lsb = ssb_modulate(msg, sideband="lsb")
    assert lsb.dtype == np.complex64
    assert np.allclose(lsb, np.conj(usb), atol=1e-6)
    # the energy really moved to negative frequencies
    spec = np.fft.fftshift(np.abs(np.fft.fft(lsb)))
    assert spec[:256].sum() > 10 * spec[256:].sum()


def test_rrc_taps_singularity_handled():
    # With sps=4, beta=0.25 the time index t = 1/(4*beta) = 1.0 symbol lands
    # exactly on a sample, exercising the limit-value branch of the formula.
    taps = rrc_taps(sps=4, span_symbols=6, beta=0.25)
    assert np.all(np.isfinite(taps))
    assert taps.size == 4 * 6 + 1
    assert np.allclose(taps, taps[::-1])          # still symmetric
    assert taps[len(taps) // 2] == taps.max()      # peak at the center


# -- spectral ----------------------------------------------------------------

def test_spectrogram_nfft_larger_than_input_yields_no_frames():
    f, t, sxx = spectrogram(_tone(1e3, n=100), FS, nfft=256)
    assert sxx.shape[0] == 0


# -- sync --------------------------------------------------------------------

def test_lock_trace_empty():
    lock, locked = _lock_trace(np.zeros(0))
    assert lock.size == 0 and locked is False


def test_carrier_recovery_empty_with_and_without_diagnostics():
    out = carrier_recovery(EMPTY, order=2)
    assert out.size == 0
    out, diag = carrier_recovery(EMPTY, order=2, diagnostics=True)
    assert out.size == 0 and diag.locked is False


def test_carrier_recovery_order4_decision_branch():
    # QPSK constellation with a small static phase offset: order-4 Costas
    # must drive the residual error down over the record.
    rng = np.random.default_rng(7)
    syms = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, 4000)))
    rx = (syms * np.exp(1j * 0.2)).astype(np.complex64)
    out, diag = carrier_recovery(rx, order=4, loop_bw=0.02, diagnostics=True)
    assert out.shape == rx.shape
    assert np.mean(np.abs(diag.error[-500:])) < np.mean(np.abs(diag.error[:500]))


def test_symbol_sync_too_short_returns_empty_diagnostics():
    short = np.ones(5, dtype=np.complex64)
    syms, diag = symbol_sync(short, samples_per_symbol=4, diagnostics=True)
    assert syms.size == 0 and diag.locked is False
    assert symbol_sync(short, samples_per_symbol=4).size == 0


def test_symbol_sync_writes_csv_when_asked(tmp_path):
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 200) * 2 - 1
    iq = np.repeat(bits, 4).astype(np.complex64)
    path = tmp_path / "loop.csv"
    syms = symbol_sync(iq, samples_per_symbol=4, csv_path=str(path))
    assert syms.size > 0
    text = path.read_text().splitlines()
    assert len(text) > 1                           # header + rows


# -- measure -----------------------------------------------------------------

@pytest.mark.parametrize("band", [(-FS, FS), (2 * FS, 3 * FS)])
def test_snr_db_rejects_band_covering_all_or_none(band):
    with pytest.raises(ValueError, match="signal_band"):
        snr_db(_tone(1e3), FS, band)


def test_fm_pilot_excess_db_none_when_record_too_short():
    assert fm_pilot_excess_db(_tone(19e3, n=64), FS) is None


def test_fm_pilot_excess_db_none_when_pilot_outside_span():
    # 1 kHz sample rate: a 19 kHz pilot bin cannot exist in the spectrum.
    iq = _tone(100.0, n=1 << 17, fs=1e3)
    assert fm_pilot_excess_db(iq, 1e3, nfft=1024) is None


def test_find_bursts_empty_and_edge_touching():
    assert find_bursts(EMPTY) == []
    # burst "on" at both ends: spans must start at 0 and end at len
    env = np.ones(64, dtype=np.complex64)
    env[20:40] = 0
    spans = find_bursts(env, threshold=0.5)
    assert spans[0][0] == 0 and spans[-1][1] == 64


def test_estimate_cfo_empty_is_zero():
    assert estimate_cfo(EMPTY, FS) == 0.0


def test_estimate_fm_cfo_none_when_decimation_leaves_one_sample():
    # Below the 4096-sample floor the function declines up front; at an
    # absurd sample rate the channelizing decimation itself leaves fewer
    # than two samples, which is the later guard.
    assert estimate_fm_cfo(_tone(1e3, n=3), FS) is None
    assert estimate_fm_cfo(_tone(1e3, n=4096, fs=2.5e9), 2.5e9) is None


# -- gain model --------------------------------------------------------------

def test_peak_counts_empty():
    assert peak_counts(EMPTY) == 0.0


def test_gain_model_validation_and_lookup():
    with pytest.raises(ValueError, match="at least one stage"):
        GainModel([])
    model = GainModel([GainStage("vga", (0, 2, 4), "level")])
    with pytest.raises(KeyError):
        model.by_name("lna")
    # no frontend stages: the grid is the single empty combination
    assert list(model.frontend_grid()) == [{}]
    assert model.frontend_db({}) == 0.0


# -- gain search: strategy branches ------------------------------------------

def _const_probe(counts_fn):
    """A probe returning a real-valued record whose peak is counts_fn(settings)
    ADC counts (full scale 128), so the search's level logic can be steered."""
    def probe(settings):
        c = float(counts_fn(settings))
        return np.full(256, c / 128.0, dtype=np.complex64)
    return probe


def test_search_gain_skips_frontend_combos_quality_declines_to_score():
    from sdr_dsp.core.gain_search import hackrf_gain_model, search_gain_model
    model = hackrf_gain_model()
    r = search_gain_model(_const_probe(lambda s: 60.0), model,
                          quality=lambda iq: None)
    # every combo was scored "unknown": no front-end chosen by quality, and
    # the result reports that rather than inventing a score
    assert r["quality_db"] is None
    assert r["status"] == "ok"


def test_search_gain_dead_antenna_exhausts_frontend_then_reports_too_weak():
    from sdr_dsp.core.gain_search import hackrf_gain_model, search_gain_model
    model = hackrf_gain_model()
    r = search_gain_model(_const_probe(lambda s: 0.0), model,
                          max_level_probes=60)
    assert r["status"] == "too_weak"
    # it did try: every front-end stage ended at its top step
    for st in model.frontend:
        assert r["settings"][st.name] == st.steps[-1]


def test_search_gain_hot_but_not_clipping_backs_frontend_off():
    from sdr_dsp.core.gain_search import hackrf_gain_model, search_gain_model
    model = hackrf_gain_model()
    lna = model.by_name("lna")

    def counts(s):
        # with any front-end gain the record sits above the target window
        # (but below clip); at minimum front end it is too weak. There is no
        # setting that satisfies the window, so the search must back the
        # front end off and eventually give up with a definite status.
        return 115.0 if s["lna"] > lna.steps[0] or s["amp"] else 20.0

    r = search_gain_model(_const_probe(counts), model, max_level_probes=80)
    assert r["status"] in ("too_weak", "clipping")
    lna_trace = [p[0]["lna"] for p in r["probes"]]
    assert max(lna_trace) > lna.steps[0]           # it escalated the LNA ...
    i_up = lna_trace.index(max(lna_trace))
    assert lna.steps[0] in lna_trace[i_up:]        # ... and later backed it off


def test_search_gain_above_window_at_minimum_everything_is_clipping():
    from sdr_dsp.core.gain_search import hackrf_gain_model, search_gain_model
    # Even the quietest settings are hot: nothing left to turn down.
    r = search_gain_model(_const_probe(lambda s: 115.0), hackrf_gain_model())
    assert r["status"] == "clipping"
    assert r["settings"] == hackrf_gain_model().min_settings()


# -- sync: decision-directed order 4 -----------------------------------------

def test_carrier_recovery_decision_directed_qpsk():
    rng = np.random.default_rng(11)
    syms = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, 3000)))
    rx = (syms * np.exp(1j * 0.15)).astype(np.complex64)
    out, diag = carrier_recovery(rx, method="decision_directed", order=4,
                                 loop_bw=0.02, diagnostics=True)
    assert out.shape == rx.shape
    assert np.mean(np.abs(diag.error[-500:])) < np.mean(np.abs(diag.error[:200]))


# -- measure: pilot metric declines when the channelized record is too short --

def test_fm_pilot_excess_db_none_when_decimation_starves_the_fft():
    # Long enough on input (> 4*nfft), but at 100 MS/s the channelizing
    # decimation leaves a few hundred samples: not enough for even the
    # shrunken FFT, so the metric declines rather than returning noise.
    iq = _tone(19e3, n=70_000, fs=100e6)
    assert fm_pilot_excess_db(iq, 100e6) is None


def test_fm_pilot_excess_db_none_when_pilot_above_channelized_nyquist():
    # After decimation to ~500 kS/s the spectrum spans +-250 kHz; a "pilot"
    # asked for at 400 kHz has no bin to be measured in.
    iq = _tone(1e3, n=1 << 17)
    assert fm_pilot_excess_db(iq, FS, pilot_hz=400e3) is None
