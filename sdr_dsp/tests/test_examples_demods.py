"""Tests for the demod example logic (synthetic signal generators + decode
chains). These guard the EXAMPLES, complementing the core library tests -- they
catch when an example's signal generation or glue logic breaks.
"""

import sys
from pathlib import Path

import numpy as np

# examples aren't a package; add the dir so we can import their helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


def test_cw_decoder_roundtrip():
    import cw_decoder as cw
    fs = 48000
    dit = int(1.2 / 20 * fs)             # 20 wpm
    iq = cw.synth_cw("SOS", fs, dit, noise=0.02)
    from sdr_dsp.core import ook_envelope, ook_slice, estimate_symbol_rate, \
        cw_decode
    bits = ook_slice(ook_envelope(iq))
    spb, _ = estimate_symbol_rate(bits, fs, min_run=3)
    assert cw_decode(bits, spb) == "SOS"


def test_ssb_synth_is_proper_sideband():
    import ssb_receiver as ssb
    from sdr_dsp.core import ssb_demod
    fs = 192000
    usb = ssb.synth_ssb(fs, "usb")
    audio = ssb_demod(usb, fs, "usb")
    wrong = ssb_demod(usb, fs, "lsb")
    # correct sideband recovers more energy than the wrong one
    assert np.std(audio) > 1.5 * np.std(wrong)


def test_ssb_recovers_tones():
    import ssb_receiver as ssb
    from sdr_dsp.core import ssb_demod
    fs = 192000
    audio = ssb_demod(ssb.synth_ssb(fs, "usb"), fs, "usb")
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    # the 800 Hz tone should be among the strong peaks
    strong = freqs[np.argsort(spec)[-30:]]
    assert any(abs(f - 800) < 100 for f in strong)


def test_dsss_demo_recovers_at_negative_snr():
    import dsss_demo as d
    from sdr_dsp.core import dsss_despread
    code = d.pn_code(63)
    rng = np.random.default_rng(0)
    data = (2 * rng.integers(0, 2, 20) - 1).astype(np.complex64)
    spread = np.concatenate([x * code for x in data]).astype(np.complex64)
    # -10 dB SNR
    snr_lin = 10 ** (-10 / 10)
    npow = np.mean(np.abs(spread) ** 2) / snr_lin
    noise = np.sqrt(npow / 2) * (rng.standard_normal(len(spread))
                                 + 1j * rng.standard_normal(len(spread)))
    rec = dsss_despread((spread + noise).astype(np.complex64), code)
    bits_out = (np.real(rec) > 0).astype(int)
    bits_in = (np.real(data) > 0).astype(int)
    n = min(len(bits_out), len(bits_in))
    assert np.sum(bits_out[:n] != bits_in[:n]) == 0   # processing gain wins


def test_fhss_synth_and_detect():
    import fhss_visualizer as f
    from sdr_dsp.core import fhss_detect_hops
    iq, seq = f.synth_fhss(2e6, n_hops=15)
    times, hops = fhss_detect_hops(iq, 2e6, nfft=256)
    assert len(times) > 0
    # the detected hop frequencies should span multiple channels
    assert len(set(np.round(hops / 1e5))) >= 4


def test_fsk_decoder_synth():
    import fsk_decoder as fd
    iq, truth = fd.make_demo(2e6, 2)
    from sdr_dsp.core import instantaneous_frequency, estimate_symbol_rate, \
        slice_to_symbols
    inst = instantaneous_frequency(iq, sample_rate=2e6)
    sm = np.convolve(inst, np.ones(20) / 20, mode="same")
    bits = (sm > 0).astype(np.uint8)
    spb, _ = estimate_symbol_rate(bits, 2e6, min_run=3)
    syms = slice_to_symbols(bits, spb)
    n = min(len(syms), len(truth))
    errs = sum(int(a) != int(b) for a, b in zip(syms[:n], truth[:n]))
    assert errs == 0   # clean recovery on the demo signal


def test_dsb_sc_recovers_tones():
    import numpy as np
    from sdr_dsp.core import dsb_sc_demod
    fs = 192000
    t = np.arange(200000) / fs
    msg = np.cos(2 * np.pi * 600 * t) + 0.5 * np.cos(2 * np.pi * 1500 * t)
    audio = dsb_sc_demod(msg.astype(np.complex64), fs)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    strong = freqs[np.argsort(spec)[-20:]]
    assert any(abs(f - 600) < 100 for f in strong)


def test_nask_decoder_synth():
    import nask_decoder as nd
    import numpy as np
    from sdr_dsp.core import ook_envelope, nask_slice
    fs = 1e6
    spb = 200
    sig, truth, amps = nd.make_demo(fs, 4, 40, spb, snr_db=30)
    env = np.convolve(ook_envelope(sig), np.ones(20) / 20, mode="same")
    per_sample = nask_slice(env, n_levels=4)
    syms = np.array([int(np.round(np.median(
        per_sample[int((i + 0.2) * spb):int((i + 0.8) * spb)])))
        for i in range(40)])
    errs = sum(int(a) != int(b) for a, b in zip(syms, truth))
    assert errs == 0


def test_burst_detector_finds_three():
    import burst_detector as bd
    from sdr_dsp.core import find_bursts
    iq = bd.make_capture(2e6)
    bursts = find_bursts(iq, min_gap=500, min_len=1000)
    assert len(bursts) == 3


def test_cfo_measures_without_changing_signal():
    import numpy as np
    from sdr_dsp.core import estimate_cfo
    fs = 1e6
    iq = np.exp(2j * np.pi * 27000 * np.arange(100000) / fs).astype(
        np.complex64)
    before = iq.copy()
    cfo = estimate_cfo(iq, fs)
    assert abs(cfo - 27000) < fs / 8192        # measured correctly
    assert np.array_equal(iq, before)          # signal untouched


def test_cfo_correction_centers_signal():
    import numpy as np
    from sdr_dsp.core import estimate_cfo, frequency_shift
    fs = 1e6
    iq = np.exp(2j * np.pi * 35000 * np.arange(100000) / fs).astype(
        np.complex64)
    cfo = estimate_cfo(iq, fs)
    corrected = frequency_shift(iq, -cfo, fs)
    assert abs(estimate_cfo(corrected, fs)) < fs / 8192


def test_differential_psk_demo_ignores_offset():
    import differential_psk_demo as dp
    from sdr_dsp.core import dbpsk_demod, dqpsk_demod
    # DBPSK: a big phase offset must not matter
    syms, tx = dp.make_dbpsk(200, offset=2.5, noise=0.05)
    bits, _ = dbpsk_demod(syms)
    n = min(len(bits), len(tx) - 1)
    assert np.sum(np.array(bits[:n]) != np.array(tx[1:1+n])) == 0
    # DQPSK: same
    syms2, tx2 = dp.make_dqpsk(150, offset=2.0, noise=0.05)
    bits2, _ = dqpsk_demod(syms2)
    n2 = min(len(bits2), len(tx2))
    assert np.sum(np.array(bits2[:n2]) != np.array(tx2[:n2])) == 0
