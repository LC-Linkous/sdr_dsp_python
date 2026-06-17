"""sdr_dsp: a personal, fully-functional DSP library for SDR IQ.

The radio DSP is the library's own code; scipy designs filter taps and numpy
provides the FFT. The core operates on complex64 arrays and knows nothing about
where they came from: IQ arrives through a source that satisfies the IQSource
protocol (ArraySource / FileSource ship here; device sources live in your own
application code) and analysis comes back out. The library provides the hooks;
you provide the hardware.
"""

from . import core
from .core import (
    to_db, from_db, normalize, DB_EPSILON,
    instantaneous_phase, instantaneous_frequency,
    fsk_demod, ssb_demod, bpsk_demod,
    correlate, convolve,
    design_lowpass, design_bandpass, design_highpass,
    fir_apply, fir_apply_centered,
    resample_poly, decimate, interpolate,
    psd, spectrogram,
    frequency_shift, tune_to_baseband, remove_dc,
    power_dbfs, snr_db, occupied_bandwidth, find_bursts, estimate_cfo,
    fm_demod, am_demod, ook_envelope, ook_slice,
    edges, estimate_symbol_rate, slice_to_symbols, deemphasis,
    matched_filter, detect_peak,
    carrier_recovery, symbol_sync, LoopDiagnostics,
)
from .sources import IQSource, ArraySource, FileSource
from . import io
from . import sinks

__version__ = "0.1.0"

__all__ = [
    "core", "io", "sinks", "IQSource", "ArraySource", "FileSource",
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "to_db", "from_db", "normalize", "DB_EPSILON",
    "instantaneous_phase", "instantaneous_frequency",
    "fsk_demod", "ssb_demod", "bpsk_demod",
    "correlate", "convolve",
    "resample_poly", "decimate", "interpolate",
    "psd", "spectrogram",
    "frequency_shift", "tune_to_baseband", "remove_dc",
    "power_dbfs", "snr_db", "occupied_bandwidth",
    "find_bursts", "estimate_cfo",
    "fm_demod", "am_demod", "ook_envelope", "ook_slice",
    "edges", "estimate_symbol_rate", "slice_to_symbols", "deemphasis",
    "matched_filter", "detect_peak",
    "carrier_recovery", "symbol_sync", "LoopDiagnostics",
]
