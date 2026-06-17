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
