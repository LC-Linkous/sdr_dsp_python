"""Tests for Tier 1 demods (no recovery needed): DSB-SC, CW, N-ASK, N-FSK,
differential PSK. Each verified on synthetic ground truth.
"""

import numpy as np

from sdr_dsp.core import (dsb_sc_demod, cw_decode, nask_slice,
                          fsk_demod_nlevel, dbpsk_demod, dqpsk_demod)


def test_dsb_sc_recovers_message():
    fs = 1e6
    t = np.arange(20000) / fs
    msg = np.cos(2 * np.pi * 3000 * t)
    dsb = msg.astype(np.complex64)
    rec = dsb_sc_demod(dsb, fs)
    spec = np.abs(np.fft.rfft(rec * np.hanning(len(rec))))
    peak = np.fft.rfftfreq(len(rec), 1 / fs)[np.argmax(spec)]
    assert abs(peak - 3000) < 50


def _morse_bits(pattern, spb):
    out = []
    chars = pattern.split(" ")
    for ci, ch in enumerate(chars):
        for si, sym in enumerate(ch):
            out += [1] * (1 if sym == "." else 3)
            if si < len(ch) - 1:
                out += [0]
        if ci < len(chars) - 1:
            out += [0, 0, 0]
    return np.repeat(out, spb).astype(np.uint8)


def test_cw_decodes_sos():
    bits = _morse_bits("... --- ...", 100)
    assert cw_decode(bits, 100) == "SOS"


def test_cw_decodes_word():
    # "HI" = .... ..
    bits = _morse_bits(".... ..", 80)
    assert cw_decode(bits, 80) == "HI"


def test_cw_empty():
    assert cw_decode(np.zeros(0, dtype=np.uint8), 100) == ""


def test_nask_four_levels():
    levels_in = [0, 1, 2, 3, 3, 2, 1, 0]
    spb = 50
    env = np.repeat([l / 3.0 for l in levels_in], spb)
    syms = nask_slice(env, n_levels=4)
    rec = [int(round(np.mean(syms[i*spb:(i+1)*spb])))
           for i in range(len(levels_in))]
    assert rec == levels_in


def test_nask_empty():
    assert len(nask_slice(np.zeros(0))) == 0


def test_fsk_nlevel_four():
    fs = 1e6
    freqs = [-75e3, -25e3, 25e3, 75e3]
    syms_in = [0, 1, 2, 3, 2, 0, 3, 1]
    spb = 200
    parts = [np.exp(2j * np.pi * freqs[s] * np.arange(spb) / fs)
             for s in syms_in]
    iq = np.concatenate(parts).astype(np.complex64)
    out = fsk_demod_nlevel(iq, fs, n_levels=4)
    rec = [int(round(np.median(out[i*spb:(i+1)*spb-1])))
           for i in range(len(syms_in))]
    assert rec == syms_in


def test_dbpsk_ignores_phase_offset():
    bits_in = [1, 0, 1, 1, 0, 0, 1, 0]
    phase = 0
    sym = []
    for b in bits_in:
        if b:
            phase += np.pi
        sym.append(np.exp(1j * phase))
    sym = np.array(sym, dtype=np.complex64) * np.exp(1j * 0.9)  # offset
    bits, _ = dbpsk_demod(sym)
    assert list(bits) == bits_in[1:]   # differential -> n-1 bits


def test_dbpsk_too_short():
    bits, soft = dbpsk_demod(np.array([1 + 0j], dtype=np.complex64))
    assert len(bits) == 0


def test_dqpsk_ignores_phase_offset():
    sym_bits = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 1)]
    step = {(0, 0): 0, (0, 1): np.pi/2, (1, 1): np.pi, (1, 0): -np.pi/2}
    phase = 0
    syms = [1 + 0j]
    for b in sym_bits:
        phase += step[b]
        syms.append(np.exp(1j * phase))
    syms = np.array(syms, dtype=np.complex64) * np.exp(1j * 0.6)
    bits, _ = dqpsk_demod(syms)
    expected = [bit for pair in sym_bits for bit in pair]
    assert list(bits) == expected


def test_fsk_nlevel_smoothing_recovers_noisy_symbols():
    """smooth_samples must rescue 4-FSK at moderate SNR: N-level slicing packs
    the bands close, so per-sample discriminator noise scatters symbols across
    adjacent bands. Smoothing over ~half a symbol is the fix (the same one
    fsk_demod gives the 2-level path). Added when the fsk_decoder example
    stopped hand-rolling this and called the library instead."""
    rng = np.random.default_rng(0)
    fs = 2e6
    nsym, spb, levels = 40, 200, 4
    syms_in = rng.integers(0, levels, nsym)
    freqs = np.linspace(-75e3, 75e3, levels)
    iq = np.concatenate([np.exp(2j * np.pi * freqs[s] * np.arange(spb) / fs)
                         for s in syms_in]).astype(np.complex64)
    iq += 0.05 * (rng.standard_normal(len(iq))
                  + 1j * rng.standard_normal(len(iq)))

    def recover(smooth):
        ps = fsk_demod_nlevel(iq, fs, n_levels=levels, smooth_samples=smooth)
        return np.array([int(ps[int((i + 0.5) * spb)]) for i in range(nsym)])

    raw_ok = int(np.sum(recover(0) == syms_in))
    sm_ok = int(np.sum(recover(spb // 2) == syms_in))
    assert sm_ok == nsym, f"smoothed recovered only {sm_ok}/{nsym}"
    assert sm_ok > raw_ok, (
        f"smoothing did not help (raw {raw_ok}, smoothed {sm_ok})")


def test_fsk_nlevel_smoothing_default_is_exact_passthrough():
    """Default smooth_samples=0 must leave the per-sample output byte-for-byte
    what it was before the parameter existed."""
    fs = 1e6
    freqs = [-75e3, -25e3, 25e3, 75e3]
    syms_in = [0, 1, 2, 3, 2, 0, 3, 1]
    spb = 200
    iq = np.concatenate([np.exp(2j * np.pi * freqs[s] * np.arange(spb) / fs)
                         for s in syms_in]).astype(np.complex64)
    a = fsk_demod_nlevel(iq, fs, n_levels=4)
    b = fsk_demod_nlevel(iq, fs, n_levels=4, smooth_samples=0)
    c = fsk_demod_nlevel(iq, fs, n_levels=4, smooth_samples=1)  # <=1 = off
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
