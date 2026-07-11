"""Feature extraction for RF device fingerprinting.

Pure-DSP extractors that turn IQ (and synchronized symbols) into device-linked
feature vectors. Classification lives *outside* the library — this package only
produces the numbers a classifier consumes.

See ``core.channel_impairments`` for the matching forward (synthesis) model that
serves as these extractors' test oracle.
"""

from __future__ import annotations

from .impairments import (
    estimate_cfo_ppm,
    estimate_iq_imbalance,
    estimate_phase_noise_variance,
    iq_image_ratio,
)
from .evm import (
    EVM_FEATURE_NAMES,
    decide_symbols,
    error_vector,
    evm_stats,
)
from .fingerprint import FEATURE_NAMES, fingerprint_vector

__all__ = [
    # impairments
    "estimate_iq_imbalance",
    "iq_image_ratio",
    "estimate_cfo_ppm",
    "estimate_phase_noise_variance",
    # evm
    "decide_symbols",
    "error_vector",
    "evm_stats",
    "EVM_FEATURE_NAMES",
    # assembly
    "fingerprint_vector",
    "FEATURE_NAMES",
]
