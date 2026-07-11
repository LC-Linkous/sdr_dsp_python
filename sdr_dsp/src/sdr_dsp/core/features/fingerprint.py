"""Fingerprint assembly — concatenate extractors into one feature vector.

Thin by design: every sub-extractor is independently tested, so this module only
orchestrates them and guarantees a stable, documented vector layout. The
returned vector is the object an *external* classifier consumes; no classifier
lives here (that is application logic, outside the library).

Design contract (matches core §9.2): pure, deterministic, numpy only, no hidden
correction. What corrections a caller wants (CFO removal, carrier recovery)
happen *before* this is called, explicitly, using the library's recovery blocks.
"""

from __future__ import annotations

import numpy as np

from .impairments import (
    estimate_cfo_ppm,
    estimate_iq_imbalance,
    estimate_phase_noise_variance,
    iq_image_ratio,
)
from .evm import EVM_FEATURE_NAMES, error_vector, evm_stats

__all__ = ["FEATURE_NAMES", "fingerprint_vector"]

#: Canonical, ordered names of the full fingerprint vector. The classifier side
#: relies on this ordering; append new features at the END, never insert.
#:
#: ``iq_image_ratio`` is the rotation-invariant (receiver-robust) imbalance
#: feature and is the one to trust for classification; ``iq_gain_db`` /
#: ``iq_phase_deg`` are retained for diagnostics but rotate with the carrier
#: phase, so a classifier should treat them cautiously.
FEATURE_NAMES: tuple[str, ...] = (
    "iq_image_ratio",
    "iq_gain_db",
    "iq_phase_deg",
    "cfo_ppm",
    "phase_noise_var",
    *EVM_FEATURE_NAMES,
)


def fingerprint_vector(
    iq: np.ndarray,
    sample_rate: float,
    carrier_hz: float,
    cfo_hz: float = 0.0,
    rx_symbols: np.ndarray | None = None,
    ref_symbols: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assemble the full fingerprint feature vector for one capture.

    The impairment features (imbalance, CFO ppm, phase-noise variance) are always
    computed from the IQ. The EVM/error-cloud block is included only when both
    ``rx_symbols`` and ``ref_symbols`` are supplied (i.e. the caller has run a
    demod far enough to have synchronized symbols and their decisions); otherwise
    those slots are filled with NaN so the vector length stays fixed.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband for this capture (ideally post-recovery). Not mutated.
    sample_rate : float
        Sample rate (Hz).
    carrier_hz : float
        Nominal carrier (Hz), for the ppm conversion.
    cfo_hz : float, default 0.0
        Absolute CFO from ``measure.estimate_cfo``, passed in rather than
        recomputed here (keeps this module a thin assembler). 0 if unknown.
    rx_symbols, ref_symbols : np.ndarray, optional
        Synchronized received symbols and their ideal references, for the EVM
        block. If either is None, EVM features are NaN.

    Returns
    -------
    (vector, names) : tuple[np.ndarray, tuple[str, ...]]
        ``vector`` is float64 of length ``len(FEATURE_NAMES)``; ``names`` is
        :data:`FEATURE_NAMES`.
    """
    iq = np.asarray(iq, dtype=np.complex64)

    img_ratio = iq_image_ratio(iq)
    gain_db, phase_deg = estimate_iq_imbalance(iq)
    cfo_ppm = estimate_cfo_ppm(cfo_hz, carrier_hz)
    pn_var = estimate_phase_noise_variance(iq)

    if rx_symbols is not None and ref_symbols is not None:
        err = error_vector(rx_symbols, ref_symbols)
        evm = evm_stats(err)
    else:
        evm = np.full(len(EVM_FEATURE_NAMES), np.nan, dtype=np.float64)

    vector = np.concatenate(
        [
            np.array(
                [img_ratio, gain_db, phase_deg, cfo_ppm, pn_var],
                dtype=np.float64,
            ),
            evm,
        ]
    )
    assert vector.shape[0] == len(FEATURE_NAMES)
    return vector, FEATURE_NAMES
