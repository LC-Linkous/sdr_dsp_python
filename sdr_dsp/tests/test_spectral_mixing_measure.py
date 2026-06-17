"""Spectral, mixing, and measure tests: a known tone lands in the right FFT
bin, frequency shifts move it predictably, and measurements are sane.
"""

import numpy as np

from sdr_dsp.core import spectral, mixing, measure
from helpers.signals import tone, noise


def test_psd_peak_at_tone_frequency():
    fs = 1_000_000
    x = tone(100_000, fs, 16384)
    freqs, psd_db = spectral.psd(x, fs, nfft=1024)
    peak_freq = freqs[np.argmax(psd_db)]
    assert abs(peak_freq - 100_000) < fs / 1024   # within one bin


def test_psd_center_freq_offset():
    fs = 1_000_000
    x = tone(0, fs, 8192)             # DC tone
    freqs, psd_db = spectral.psd(x, fs, nfft=1024, center_freq=2.4e9)
    peak_freq = freqs[np.argmax(psd_db)]
    assert abs(peak_freq - 2.4e9) < fs / 1024


def test_spectrogram_shape():
    fs = 1_000_000
    x = tone(50_000, fs, 8192)
    freqs, times, sxx = spectral.spectrogram(x, fs, nfft=512, overlap=0.5)
    assert sxx.shape[1] == 512
    assert sxx.shape[0] == len(times)
    assert len(freqs) == 512


def test_frequency_shift_moves_tone():
    fs = 1_000_000
    x = tone(100_000, fs, 8192)
    shifted = mixing.frequency_shift(x, 50_000, fs)   # 100k -> 150k
    freqs, psd_db = spectral.psd(shifted, fs, nfft=1024)
    peak = freqs[np.argmax(psd_db)]
    assert abs(peak - 150_000) < fs / 1024


def test_tune_to_baseband():
    fs = 1_000_000
    x = tone(100_000, fs, 8192)
    based = mixing.tune_to_baseband(x, 100_000, fs)   # 100k -> 0
    freqs, psd_db = spectral.psd(based, fs, nfft=1024)
    peak = freqs[np.argmax(psd_db)]
    assert abs(peak) < fs / 1024


def test_power_dbfs_known():
    # amplitude 0.5 tone -> power 0.25 -> ~ -6 dBFS
    x = tone(1000, 48_000, 1000, amp=0.5)
    assert abs(measure.power_dbfs(x) - (-6.02)) < 0.1


def test_power_dbfs_empty():
    assert measure.power_dbfs(np.array([], dtype=np.complex64)) == float("-inf")


def test_snr_high_for_clean_tone():
    fs = 1_000_000
    x = tone(0, fs, 16384) + 0.001 * noise(16384, seed=4)
    snr = measure.snr_db(x, fs, signal_band_hz=(-20_000, 20_000))
    assert snr > 20      # clean tone -> high SNR


def test_occupied_bandwidth_positive():
    fs = 1_000_000
    # a band-limited noise-ish signal
    x = tone(0, fs, 16384) + 0.1 * noise(16384, seed=5)
    bw = measure.occupied_bandwidth(x, fs, fraction=0.99)
    assert bw > 0


def test_mixing_roundtrip_identity():
    # shift up by f then down by f recovers the original within precision
    fs = 1_000_000
    x = tone(100_000, fs, 8192)
    up = mixing.frequency_shift(x, 200_000, fs)
    back = mixing.frequency_shift(up, -200_000, fs)
    assert np.allclose(back, x, atol=1e-5)


def test_psd_parseval_scaling():
    # the integrated PSD should relate to time-domain power. For a unit tone,
    # total power is ~1; the PSD integrated over frequency (linear) should
    # recover a comparable value to mean |x|^2.
    fs = 1_000_000
    x = tone(100_000, fs, 16384, amp=1.0)
    time_power = float(np.mean(np.abs(x) ** 2))      # ~1.0
    freqs, psd_db = spectral.psd(x, fs, nfft=1024, window="rect")
    psd_lin = 10 ** (psd_db / 10.0)
    df = freqs[1] - freqs[0]
    integrated = float(np.sum(psd_lin) * df)          # integrate over Hz
    # within a few dB -- windowing/scaling conventions vary, but same order
    ratio = integrated / time_power
    assert 0.3 < ratio < 3.0


def test_spectrogram_time_alignment():
    # a burst that starts partway through should appear in the right time rows,
    # not at the start.
    fs = 1_000_000
    n = 20480
    x = np.zeros(n, dtype=np.complex64)
    start = n // 2
    x[start:start + 4096] = tone(100_000, fs, 4096)   # burst in the 2nd half
    freqs, times, sxx = spectral.spectrogram(x, fs, nfft=512, overlap=0.5)
    # find the row with the most energy; its time should be in the 2nd half
    row_energy = np.sum(10 ** (sxx / 10), axis=1)
    peak_row = int(np.argmax(row_energy))
    peak_time = times[peak_row]
    assert peak_time > (start / fs) * 0.8   # roughly in the burst region


def test_mixing_empty_and_single():
    # edge cases shouldn't crash
    assert len(mixing.frequency_shift(np.array([], dtype=np.complex64), 1e3,
                                      1e6)) == 0
    out = mixing.frequency_shift(np.array([1 + 0j], dtype=np.complex64), 1e3, 1e6)
    assert len(out) == 1


def test_measure_occupied_bandwidth_wider_for_wider_signal():
    # a wider-band signal should report a larger occupied bandwidth
    fs = 1_000_000
    narrow = tone(0, fs, 16384) + 0.01 * noise(16384, seed=10)
    wide = noise(16384, seed=11)        # full-band noise
    bw_narrow = measure.occupied_bandwidth(narrow, fs)
    bw_wide = measure.occupied_bandwidth(wide, fs)
    assert bw_wide > bw_narrow
