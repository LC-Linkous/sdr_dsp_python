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
    design_lowpass, design_bandpass, design_highpass,
    fir_apply, fir_apply_centered,
    resample_poly, decimate, interpolate,
    psd, spectrogram,
    frequency_shift, tune_to_baseband,
    power_dbfs, snr_db, occupied_bandwidth,
    fm_demod, am_demod, ook_envelope, ook_slice,
    edges, estimate_symbol_rate, slice_to_symbols,
)
from .sources import IQSource, ArraySource, FileSource
from . import io

__version__ = "0.1.0"

__all__ = [
    "core", "io", "IQSource", "ArraySource", "FileSource",
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "resample_poly", "decimate", "interpolate",
    "psd", "spectrogram",
    "frequency_shift", "tune_to_baseband",
    "power_dbfs", "snr_db", "occupied_bandwidth",
    "fm_demod", "am_demod", "ook_envelope", "ook_slice",
    "edges", "estimate_symbol_rate", "slice_to_symbols",
]
