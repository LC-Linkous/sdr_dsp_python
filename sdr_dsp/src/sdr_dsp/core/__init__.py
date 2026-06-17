"""Pure DSP core: arrays in, arrays out. Never imports sources or sinks."""

from .util import to_db, from_db, normalize, DB_EPSILON
from .filters import (
    design_lowpass,
    design_bandpass,
    design_highpass,
    fir_apply,
    fir_apply_centered,
)
from .resample import resample_poly, decimate, interpolate
from .spectral import psd, spectrogram
from .mixing import frequency_shift, tune_to_baseband, remove_dc
from .measure import (power_dbfs, snr_db, occupied_bandwidth,
                     find_bursts, estimate_cfo)
from .detect import matched_filter, detect_peak, correlate, convolve
from .demod import (fm_demod, am_demod, ook_envelope, ook_slice,
                    edges, estimate_symbol_rate, slice_to_symbols, deemphasis,
                    instantaneous_phase, instantaneous_frequency,
                    fsk_demod, ssb_demod, bpsk_demod)

__all__ = [
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "to_db", "from_db", "normalize", "DB_EPSILON",
    "resample_poly", "decimate", "interpolate",
    "psd", "spectrogram",
    "frequency_shift", "tune_to_baseband", "remove_dc",
    "power_dbfs", "snr_db", "occupied_bandwidth",
    "find_bursts", "estimate_cfo",
    "fm_demod", "am_demod", "ook_envelope", "ook_slice",
    "instantaneous_phase", "instantaneous_frequency",
    "fsk_demod", "ssb_demod", "bpsk_demod",
    "edges", "estimate_symbol_rate", "slice_to_symbols", "deemphasis",
    "matched_filter", "detect_peak", "correlate", "convolve",
]
