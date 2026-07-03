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


def test_annotate_bursts_workflow():
    import annotate_bursts as ab
    import tempfile, os
    from sdr_dsp.io import read_annotations
    fs = 2e6
    iq = ab.make_capture(fs)
    from sdr_dsp.core import find_bursts
    from sdr_dsp.io import save_iq, bursts_to_annotations
    spans = find_bursts(iq, min_gap=500, min_len=1000)
    anns = bursts_to_annotations(spans, label="burst {i}")
    d = tempfile.mkdtemp()
    save_iq(os.path.join(d, "a.iq"), iq, fs, annotations=anns)
    loaded = read_annotations(os.path.join(d, "a.iq"))
    assert len(loaded) == len(spans)


def test_power_calibration_workflow():
    import numpy as np
    from sdr_dsp.core import compute_cal_offset, Calibration, power_dbfs
    import tempfile, os
    ref = (0.1 * np.exp(2j * np.pi * 0.05 * np.arange(20000))).astype(
        np.complex64)
    cal = compute_cal_offset(ref, known_dbm=-30.0, frequency_hz=433.92e6)
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "demo.cal.json")
    cal.save(fp)
    cal2 = Calibration.load(fp)
    # the reference reads back as its known power
    assert abs(cal2.power_dbm(ref) - (-30.0)) < 1e-6


def test_agc_demo_recoverable():
    import numpy as np
    from sdr_dsp.core import agc
    n = 60000
    t = np.arange(n)
    env = 0.5 + 0.45 * np.cos(2 * np.pi * 2.5 * t / n)
    sig = (env * np.exp(2j * np.pi * 0.05 * t)).astype(np.complex64)
    adjusted, gain = agc(sig, target=0.5)
    # the example's headline honesty claim
    assert np.allclose(adjusted / gain, sig, atol=1e-5)
    # and it flattens the fade
    assert np.abs(adjusted)[n // 4:].std() < np.abs(sig)[n // 4:].std()


def test_channelizer_example_both_modes():
    import channelizer as ch
    import numpy as np
    from sdr_dsp.core import channelize, channelize_bank
    fs = 2e6
    iq = ch.synth_band(fs)
    # single
    out, rate = channelize(iq, fs, 300e3, 100e3)
    assert rate < fs and len(out) > 0
    # bank
    chans, brate, freqs = channelize_bank(iq, fs, 8)
    assert chans.shape[0] == 8


def test_modulate_demo_closes_loop():
    import numpy as np
    from sdr_dsp.core import (ook_modulate, ook_envelope, ook_slice,
                              qpsk_modulate, qpsk_demod)
    bits = np.random.default_rng(0).integers(0, 2, 100)
    # OOK loop
    rec = ook_slice(ook_envelope(ook_modulate(bits, 20)))[::20][:len(bits)]
    assert np.mean(rec != bits) == 0
    # QPSK loop
    qrec, _ = qpsk_demod(qpsk_modulate(bits, 1))
    assert np.mean(qrec[:len(bits)] != bits) == 0


def test_packet_loopback_example():
    import numpy as np
    from sdr_dsp.core import (build_frame, find_frames,
                              ook_modulate, ook_envelope, ook_slice)
    frame_bits = build_frame(b"ACK 42")
    sps = 20
    iq = ook_modulate(frame_bits, sps)
    rec = np.asarray(ook_slice(ook_envelope(iq))[::sps][:len(frame_bits)],
                     dtype=np.uint8)
    found = find_frames(rec)
    assert found and found[0]["payload"] == b"ACK 42" and found[0]["crc_ok"]


def test_channel_sweep_example():
    import numpy as np
    from sdr_dsp.core import (build_frame, find_frames, apply_channel,
                              fsk_modulate, fsk_demod)
    frame = build_frame(b"CQ DE SDR")
    sps = 20
    fs = 1e6
    iq = fsk_modulate(frame, sps, 50e3, fs)
    # clean channel -> recovered
    rx = apply_channel(iq, sample_rate=fs, snr_db=30, seed=1)
    bits = np.asarray(fsk_demod(rx, fs)[sps // 2::sps][:len(frame)],
                      dtype=np.uint8)
    found = find_frames(bits)
    assert found and found[0]["crc_ok"]


def test_two_station_link_example():
    import two_station_link as tsl
    from sdr_dsp.link import ARQ, run_sim
    A = ARQ(window_size=1, timeout_ticks=3, max_retries=10)
    B = ARQ(window_size=1, timeout_ticks=3, max_retries=10)
    msgs = [b"CQ CQ", b"DE SDR"]
    for m in msgs:
        A.send(m)
    transport = tsl.build_transport(25, drop_first=True)
    _, received, log = run_sim(A, B, transport=transport, max_ticks=500)
    assert received == msgs               # delivered despite the forced drop
