"""EVM and error-cloud features — the modulation-domain fingerprint.

Once symbols are synchronized and decided, the *error* between each received
symbol and its ideal constellation point carries a device signature that the
decoded bits throw away. This module extracts that error and summarizes its
structure.

Design contract (matches core §9.2):
- pure functions, no input mutation, deterministic;
- numpy only at runtime;
- no hidden normalization — the caller controls scaling. ``evm_stats`` reports
  raw error moments; a caller wanting reference-normalized EVM divides by the
  reference RMS explicitly (a flag is provided but off by default).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "decide_symbols",
    "error_vector",
    "evm_stats",
    "EVM_FEATURE_NAMES",
]

#: Canonical ordering of the vector returned by :func:`evm_stats`.
EVM_FEATURE_NAMES: tuple[str, ...] = (
    "evm_rms",
    "evm_mean_mag",
    "err_var_i",
    "err_var_q",
    "err_skew_i",
    "err_skew_q",
    "err_kurt_i",
    "err_kurt_q",
    "err_corr_iq",
)


def decide_symbols(rx_symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Nearest-constellation-point decision for each received symbol.

    A pure, fully vectorized minimum-distance slicer. Provided so callers who
    have received symbols but not the ideal decisions can obtain ``s_hat``
    without hiding the decision inside another function.

    Parameters
    ----------
    rx_symbols : np.ndarray
        Complex received symbols (one sample per symbol). Not mutated.
    constellation : np.ndarray
        Complex array of the ideal constellation points.

    Returns
    -------
    np.ndarray
        Complex array, same length as ``rx_symbols``, each entry one of the
        constellation points.
    """
    r = np.asarray(rx_symbols, dtype=np.complex128)
    c = np.asarray(constellation, dtype=np.complex128)
    # distance matrix |r_i - c_j|, argmin over j
    dists = np.abs(r[:, None] - c[None, :])
    idx = np.argmin(dists, axis=1)
    return c[idx]


def error_vector(rx_symbols: np.ndarray, ref_symbols: np.ndarray) -> np.ndarray:
    """Per-symbol error ``e_k = r_k - s_hat_k``.

    Parameters
    ----------
    rx_symbols : np.ndarray
        Complex received symbols. Not mutated.
    ref_symbols : np.ndarray
        Ideal reference symbols of equal length (the decided or known-true
        constellation points).

    Returns
    -------
    np.ndarray
        Complex error per symbol.
    """
    r = np.asarray(rx_symbols, dtype=np.complex128)
    s = np.asarray(ref_symbols, dtype=np.complex128)
    if r.shape != s.shape:
        raise ValueError(
            f"rx_symbols {r.shape} and ref_symbols {s.shape} must match"
        )
    return r - s


def _moment_skew(x: np.ndarray) -> float:
    m = np.mean(x)
    sd = np.std(x)
    if sd == 0.0:
        return 0.0
    return float(np.mean(((x - m) / sd) ** 3))


def _moment_kurt(x: np.ndarray) -> float:
    m = np.mean(x)
    sd = np.std(x)
    if sd == 0.0:
        return 0.0
    # excess kurtosis (Gaussian -> 0)
    return float(np.mean(((x - m) / sd) ** 4) - 3.0)


def evm_stats(
    errors: np.ndarray,
    ref_symbols: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Summarize the error cloud into a fixed feature vector.

    Returns the moments described by :data:`EVM_FEATURE_NAMES`, in that order.
    The higher moments (skew, kurtosis, I/Q correlation) are what separate a
    receiver-noise-dominated cloud (Gaussian, symmetric, uncorrelated) from a
    device-impairment-dominated one (structured, skewed, correlated).

    .. note::
        For fingerprinting, weight the *shape* moments (skew, kurtosis,
        ``err_corr_iq``) over the *magnitude* features: ``evm_rms`` and
        ``evm_mean_mag`` scale directly with receiver noise, so at moderate SNR
        they are mostly a distance-to-antenna thermometer. A classifier fed the
        raw vector will happily learn the capture geometry from them unless
        SNR is controlled or the magnitude features are dropped/normalized.

    Parameters
    ----------
    errors : np.ndarray
        Complex per-symbol error from :func:`error_vector`. Not mutated.
    ref_symbols : np.ndarray, optional
        Needed only if ``normalize=True``, to divide by the reference RMS.
    normalize : bool, default False
        If True, scale the magnitude features by the reference RMS so EVM is
        expressed as a fraction (the conventional %EVM/100). Off by default to
        honor the no-hidden-normalization rule.

    Returns
    -------
    np.ndarray
        Real-valued feature vector, length ``len(EVM_FEATURE_NAMES)``.
    """
    e = np.asarray(errors, dtype=np.complex128)
    ei = e.real
    eq = e.imag

    scale = 1.0
    if normalize:
        if ref_symbols is None:
            raise ValueError("normalize=True requires ref_symbols")
        ref_rms = np.sqrt(np.mean(np.abs(np.asarray(ref_symbols)) ** 2))
        scale = ref_rms if ref_rms != 0.0 else 1.0

    evm_rms = np.sqrt(np.mean(np.abs(e) ** 2)) / scale
    evm_mean_mag = np.mean(np.abs(e)) / scale
    var_i = np.var(ei) / (scale ** 2)
    var_q = np.var(eq) / (scale ** 2)
    skew_i = _moment_skew(ei)
    skew_q = _moment_skew(eq)
    kurt_i = _moment_kurt(ei)
    kurt_q = _moment_kurt(eq)

    if np.std(ei) == 0.0 or np.std(eq) == 0.0:
        corr_iq = 0.0
    else:
        corr_iq = float(np.corrcoef(ei, eq)[0, 1])

    return np.array(
        [
            evm_rms,
            evm_mean_mag,
            var_i,
            var_q,
            skew_i,
            skew_q,
            kurt_i,
            kurt_q,
            corr_iq,
        ],
        dtype=np.float64,
    )
