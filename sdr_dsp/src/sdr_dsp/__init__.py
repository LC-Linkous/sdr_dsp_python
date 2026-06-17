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
    dsb_sc_demod, cw_decode, nask_slice, fsk_demod_nlevel,
    dbpsk_demod, dqpsk_demod, qpsk_demod, psk8_demod,
    qam16_demod, dsss_despread, fhss_detect_hops,
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

__version__ = "0.1.0"

# io, sinks, stream and Pipeline are imported LAZILY (PEP 562) so that importing
# the package -- or a submodule like `from sdr_dsp.core import demod` -- never
# force-loads them during package initialization. Eagerly importing them here
# created an import-ordering cycle on some platforms (the package was still
# partially initialized when `stream` was pulled in). Lazy access keeps
# `sdr_dsp.io`, `sdr_dsp.sinks`, `sdr_dsp.stream`, and `sdr_dsp.Pipeline` working
# identically while removing the fragility.
def __getattr__(name):
    if name in ("io", "sinks", "stream"):
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "Pipeline":
        from .stream import Pipeline
        globals()["Pipeline"] = Pipeline
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "core", "io", "sinks", "stream", "Pipeline",
    "IQSource", "ArraySource", "FileSource",
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "to_db", "from_db", "normalize", "DB_EPSILON",
    "instantaneous_phase", "instantaneous_frequency",
    "fsk_demod", "ssb_demod", "bpsk_demod",
    "dsb_sc_demod", "cw_decode", "nask_slice", "fsk_demod_nlevel",
    "dbpsk_demod", "dqpsk_demod", "qpsk_demod", "psk8_demod",
    "qam16_demod", "dsss_despread", "fhss_detect_hops",
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