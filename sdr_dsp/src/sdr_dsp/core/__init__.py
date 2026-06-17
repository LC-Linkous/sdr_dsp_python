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
from .sync import carrier_recovery, symbol_sync, LoopDiagnostics
from .demod import (fm_demod, am_demod, ssb_demod, dsb_sc_demod, cw_decode,
                    ook_envelope, ook_slice, nask_slice,
                    fsk_demod, fsk_demod_nlevel,
                    bpsk_demod, dbpsk_demod, dqpsk_demod, qpsk_demod, psk8_demod,
                    qam16_demod, dsss_despread, fhss_detect_hops,
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
    "fsk_demod", "fsk_demod_nlevel", "ssb_demod", "dsb_sc_demod", "cw_decode",
    "bpsk_demod", "dbpsk_demod", "dqpsk_demod", "nask_slice",
    "qpsk_demod", "psk8_demod", "qam16_demod",
    "dsss_despread", "fhss_detect_hops",
    "edges", "estimate_symbol_rate", "slice_to_symbols", "deemphasis",
    "matched_filter", "detect_peak", "correlate", "convolve",
    "carrier_recovery", "symbol_sync", "LoopDiagnostics",
]
