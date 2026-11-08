"""Pure DSP core: arrays in, arrays out. Never imports sources or sinks.

Two export tiers, on purpose:

* ``sdr_dsp.<name>`` (the top-level package) re-exports the SIGNAL-PROCESSING
  surface: filters, resampling, spectral, mixing, measurement, mod/demod,
  framing, channel simulation, recovery loops, AGC, calibration.
* ``sdr_dsp.core.<name>`` additionally exports the RADIO-CHARACTERIZATION
  layer: device-impairment synthesis (``add_iq_imbalance``, ...,
  ``DeviceImpairments``), the fingerprint estimators and feature vector
  (``core.features``), capture-health / pilot gating (``capture_health``,
  ``fm_pilot_excess_db``), the gain-search engine and models
  (``GainModel``, ``search_gain_model``, ...), and ``get_window``.

Dtype policy (pinned by tests/test_public_surface.py::test_dtype_policy):

* IQ in, IQ out -> ``complex64``. Every function that returns baseband
  (filters, resamplers, mixers, channelizers, AGC, channel models, recovery
  loops) returns complex64 whatever complex dtype came in; a complex128 input
  is processed as complex64. That is the dtype every source produces (8- and
  16-bit ADCs land in complex64 with headroom to spare) and the one the
  streaming stages are bit-exact in; keeping one IQ dtype is what lets a
  streamed run equal a whole-record run sample for sample.
* Real-valued results -> ``float64``: demodulated audio, envelopes,
  correlation magnitudes, gain traces, measurements, and any filter or
  resampler applied to a REAL signal. These are smaller arrays, downstream of
  the throughput-critical path, and float64 keeps the arithmetic (de-emphasis
  IIR state, cumulative phase) honest.
* There is deliberately no "preserve the input dtype" mode: the two rules
  above are the contract, and a function that needs something else says so
  in its docstring.

Everything in the second tier is reachable as ``sdr_dsp.core.X``; it is not
flattened to ``sdr_dsp.X`` because it is about radios and captures rather than
about signals, and the top-level namespace is kept to what every user of a DSP
library expects to find there. ``tests/test_public_surface.py`` pins the split
so a new name lands in a tier by decision, not by accident.
"""

from .util import to_db, from_db, normalize, DB_EPSILON
from .filters import (
    design_lowpass,
    design_bandpass,
    design_highpass,
    fir_apply,
    fir_apply_centered,
)
from .resample import resample_poly, decimate, interpolate
from .spectral import get_window, psd, spectrogram
from .mixing import frequency_shift, tune_to_baseband, remove_dc
from .modulate import (fm_modulate, am_modulate, ssb_modulate,
                       ook_modulate, fsk_modulate, bpsk_modulate,
                       qpsk_modulate, rrc_taps, upsample, pulse_shape)
from .channel import apply_channel, add_noise, add_cfo, add_delay
from .channel_impairments import (add_iq_imbalance, add_pa_nonlinearity,
                                  add_phase_noise, DeviceImpairments,
                                  make_device_impairments,
                                  apply_device_impairments)
from .features import (estimate_iq_imbalance, iq_image_ratio,
                       estimate_cfo_ppm, estimate_phase_noise_variance,
                       decide_symbols, error_vector, evm_stats,
                       EVM_FEATURE_NAMES, fingerprint_vector, FEATURE_NAMES)
from .framing import build_frame, find_frames, crc16
from .channelize import channelize, channelize_bank
from .agc import agc, AGC
from .calibrate import power_dbm, Calibration, compute_cal_offset
from .gain_search import (search_gain, peak_counts, search_gain_model,
                          GainStage, GainModel, hackrf_gain_model,
                          rtlsdr_gain_model)
from .measure import (power_dbfs, snr_db, occupied_bandwidth, capture_health,
                      fm_pilot_excess_db,
                     find_bursts, estimate_cfo, estimate_fm_cfo,
                     correct_fm_cfo)
from .detect import matched_filter, detect_peak, correlate, convolve
from .sync import carrier_recovery, symbol_sync, LoopDiagnostics
from .demod import (fm_demod, am_demod, dsb_sc_demod, cw_decode,
                    fm_stereo_decode,
                    ook_envelope, ook_slice, nask_slice,
                    fsk_demod_nlevel,
                    dbpsk_demod, dqpsk_demod, qpsk_demod, psk8_demod,
                    qam16_demod, dsss_despread, fhss_detect_hops,
                    edges, estimate_symbol_rate, slice_to_symbols, sample_symbols, deemphasis,
                    instantaneous_phase, instantaneous_frequency,
                    fsk_demod, ssb_demod, bpsk_demod)

__all__ = [
    "design_lowpass", "design_bandpass", "design_highpass",
    "fir_apply", "fir_apply_centered",
    "to_db", "from_db", "normalize", "DB_EPSILON",
    "resample_poly", "decimate", "interpolate",
    "get_window", "psd", "spectrogram",
    "frequency_shift", "tune_to_baseband", "remove_dc",
    "power_dbfs", "power_dbm", "Calibration", "compute_cal_offset", "agc", "AGC",
    "channelize", "channelize_bank",
    "build_frame", "find_frames", "crc16",
    "apply_channel", "add_noise", "add_cfo", "add_delay",
    "add_iq_imbalance", "add_pa_nonlinearity", "add_phase_noise",
    "DeviceImpairments", "make_device_impairments", "apply_device_impairments",
    "estimate_iq_imbalance", "iq_image_ratio", "estimate_cfo_ppm",
    "estimate_phase_noise_variance", "decide_symbols", "error_vector",
    "evm_stats", "EVM_FEATURE_NAMES", "fingerprint_vector", "FEATURE_NAMES",
    "fm_modulate", "am_modulate", "ssb_modulate",
    "ook_modulate", "fsk_modulate", "bpsk_modulate", "qpsk_modulate",
    "rrc_taps", "upsample", "pulse_shape", "snr_db", "occupied_bandwidth", "capture_health",
    "fm_pilot_excess_db", "search_gain", "peak_counts",
    "search_gain_model", "GainStage", "GainModel",
    "hackrf_gain_model", "rtlsdr_gain_model",
    "find_bursts", "estimate_cfo",
    "estimate_fm_cfo", "correct_fm_cfo",
    "fm_demod", "am_demod", "ook_envelope", "ook_slice",
    "instantaneous_phase", "instantaneous_frequency",
    "fsk_demod", "fsk_demod_nlevel", "ssb_demod", "dsb_sc_demod", "cw_decode",
    "fm_stereo_decode",
    "bpsk_demod", "dbpsk_demod", "dqpsk_demod", "nask_slice",
    "qpsk_demod", "psk8_demod", "qam16_demod",
    "dsss_despread", "fhss_detect_hops",
    "edges", "estimate_symbol_rate", "slice_to_symbols", "sample_symbols", "deemphasis",
    "matched_filter", "detect_peak", "correlate", "convolve",
    "carrier_recovery", "symbol_sync", "LoopDiagnostics",
]
