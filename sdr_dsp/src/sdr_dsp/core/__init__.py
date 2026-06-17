"""Pure DSP core: arrays in, arrays out. Never imports sources or sinks."""

from .filters import (
    design_lowpass,
    design_bandpass,
    design_highpass,
    fir_apply,
    fir_apply_centered,
)
from .resample import resample_poly, decimate, interpolate
from .spectral import psd, spectrogram
from .mixing import frequency_shift, tune_to_baseband
from .measure import power_dbfs, snr_db, occupied_bandwidth
from .demod import fm_demod, am_demod, ook_envelope, ook_slice

__all__ = [
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "resample_poly", "decimate", "interpolate",
    "psd", "spectrogram",
    "frequency_shift", "tune_to_baseband",
    "power_dbfs", "snr_db", "occupied_bandwidth",
    "fm_demod", "am_demod", "ook_envelope", "ook_slice",
]
