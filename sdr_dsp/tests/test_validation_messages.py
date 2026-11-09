"""Bad arguments produce a ValueError/TypeError that names the PARAMETER and
the bad value -- never a numpy/scipy internal, a ZeroDivisionError, or a
plausible-looking result. One case per function the 0.3.0 API review found
accepting garbage or leaking a foreign message.
"""
import numpy as np
import pytest

import sdr_dsp as S
from sdr_dsp.core import (nask_slice, fsk_demod_nlevel, dsss_despread, cw_decode,
                          slice_to_symbols, sample_symbols, upsample, build_frame,
                          crc16, power_dbm, get_window, snr_db, AGC, edges)

IQ = np.exp(2j * np.pi * 0.01 * np.arange(4096)).astype(np.complex64)
TAPS = S.design_lowpass(50e3, 1e6, num_taps=31)

CASES = [
    # (callable, exception, text that must appear in the message)
    (lambda: S.design_lowpass(1e3, 1e6, num_taps=0), ValueError, "num_taps"),
    (lambda: S.design_lowpass(1e3, 1e6, window="foo"), ValueError, "window"),
    (lambda: S.design_lowpass(1e3, -1e6), ValueError, "sample_rate"),
    (lambda: S.design_bandpass(5e3, 1e3, 1e6), ValueError, "low_hz"),
    (lambda: S.decimate(IQ, 0), ValueError, "factor"),
    (lambda: S.interpolate(IQ, -2), ValueError, "factor"),
    (lambda: S.resample_poly(IQ, 2.5, 1), ValueError, "up"),
    (lambda: S.psd(IQ, 1e6, window="kaiser"), ValueError, "window"),
    (lambda: get_window("kaiser", 64), ValueError, "window"),
    (lambda: S.occupied_bandwidth(IQ[:100], 1e6), ValueError, "nfft"),
    (lambda: S.occupied_bandwidth(IQ, 1e6, fraction=1.5), ValueError, "fraction"),
    (lambda: snr_db(IQ, 1e6, 100e3), TypeError, "signal_band_hz"),
    (lambda: power_dbm(IQ, None), TypeError, "cal_offset_db"),
    (lambda: S.tune_to_baseband(IQ, 0, 1e3), ValueError, "sample_rate"),
    (lambda: S.frequency_shift(IQ, -1e6, 1e3), ValueError, "sample_rate"),
    (lambda: S.fm_demod(IQ, deviation_hz=75e3), ValueError, "BOTH"),
    (lambda: S.symbol_sync(IQ, 0), ValueError, "samples_per_symbol"),
    (lambda: S.symbol_sync(IQ, 8, method="zero_crossing"), ValueError, "method"),
    (lambda: S.carrier_recovery(IQ, order=3), ValueError, "order"),
    (lambda: S.carrier_recovery(IQ, method="pll"), ValueError, "method"),
    (lambda: S.agc(IQ, mode="loud"), ValueError, "mode"),
    (lambda: AGC(mode="x"), ValueError, "mode"),
    (lambda: S.add_noise(IQ, "10"), TypeError, "snr_db"),
    (lambda: S.add_noise(IQ, 10, rng=3), TypeError, "rng"),
    (lambda: S.bpsk_modulate([0, 2, 1]), ValueError, "bits"),
    (lambda: S.qpsk_modulate([1, 0, 1]), ValueError, "even number of bits"),
    (lambda: S.ook_modulate([1, 0], 0), ValueError, "samples_per_symbol"),
    (lambda: S.fsk_modulate([1, 0], 0.5, 1e3, 1e6), TypeError, "samples_per_symbol"),
    (lambda: upsample([1, -1], 0), ValueError, "sps"),
    (lambda: slice_to_symbols([1, 0, 1], 0), ValueError, "samples_per_symbol"),
    (lambda: sample_symbols([1, 0, 1], 2.5), TypeError, "samples_per_symbol"),
    (lambda: cw_decode([1, 0, 1], 0), ValueError, "samples_per_symbol"),
    (lambda: dsss_despread(IQ, [1, -1, 1], samples_per_chip=0), ValueError, "samples_per_chip"),
    (lambda: nask_slice([0.1, 0.5], levels=[0.5]), ValueError, "levels"),
    (lambda: fsk_demod_nlevel(IQ, 1e6, thresholds=[0.0]), ValueError, "thresholds"),
    (lambda: S.fsk_demod(IQ, 1e6, threshold_hz="middle"), ValueError, "threshold_hz"),
    (lambda: build_frame("text"), TypeError, "encode"),
    (lambda: crc16("text"), TypeError, "encode"),
    (lambda: build_frame(b"x", preamble_bits=-2), ValueError, "preamble_bits"),
    (lambda: edges(np.array([-1, 1, -1])), ValueError, "bits"),
    (lambda: S.ArraySource(IQ, 1e6, block_size=0), ValueError, "block_size"),
    (lambda: S.find_frames(np.array([0, 1, 2])), ValueError, "bits"),
    (lambda: S.channelize_bank(IQ, 1e6, 4, return_freqs=False), TypeError, "return_freqs"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[2] + "/" + str(i) for i, c in enumerate(CASES)])
def test_bad_argument_names_the_parameter(case):
    fn, exc, needle = case
    with pytest.raises(exc) as info:
        fn()
    msg = str(info.value)
    assert needle in msg, f"message does not mention {needle!r}: {msg}"
    # and never a numpy/scipy internal leaking through
    for leak in ("Nx=", "slice step", "modulo by zero", "negative dimensions",
                 "non-iterable", "unsupported operand", "without an encoding"):
        assert leak not in msg, f"foreign message leaked: {msg}"


def test_explicit_levels_define_the_count():
    env = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    # three explicit levels: n_levels (default 4) is ignored
    assert list(nask_slice(env, levels=[0.0, 0.5, 1.0])) == [0, 0, 1, 2, 2]
    assert list(nask_slice(env, n_levels=2)) == [0, 0, 0, 1, 1]


def test_none_means_auto_everywhere():
    # fsk_demod joins ook_slice / find_bursts: None is "decide from the data";
    # the old "auto" string is still accepted
    from sdr_dsp.core import fsk_modulate, fsk_demod
    x = fsk_modulate([1, 0, 1, 1, 0, 0, 1, 0], 20, 50e3, 1e6)
    np.testing.assert_array_equal(fsk_demod(x, 1e6, threshold_hz=None),
                                  fsk_demod(x, 1e6, threshold_hz="auto"))
