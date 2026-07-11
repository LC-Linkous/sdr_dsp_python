"""Impairment estimators — the per-device feature extractors.

Each function inverts one impairment from ``core.channel_impairments`` and
returns a scalar (or small vector) that characterizes it. These are the raw
material of a device fingerprint.

Design contract (matches core §9.2):
- pure functions, no input mutation, deterministic;
- do NOT apply corrections — they *measure* and return numbers; the caller
  decides what to do (this is the library's honest-DSP rule);
- numpy only at runtime.

The test oracle for every estimator here is the forward model in
``channel_impairments``: apply a known impairment, confirm recovery.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "estimate_iq_imbalance",
    "iq_image_ratio",
    "estimate_cfo_ppm",
    "estimate_phase_noise_variance",
]


def iq_image_ratio(iq: np.ndarray) -> float:
    """Rotation-invariant I/Q-imbalance magnitude: ``|E[r^2]| / E[|r|^2]``.

    Unlike the separate ``(gain_db, phase_deg)`` from
    :func:`estimate_iq_imbalance`, this scalar is invariant to an unknown bulk
    carrier phase rotation (a rotation by ``theta`` multiplies ``E[r^2]`` by
    ``exp(2j*theta)`` but leaves its magnitude unchanged). That makes it the
    receiver-robust imbalance fingerprint: it depends on the transmitter's
    image-rejection, not on where the receiver happened to sample the carrier
    phase. Prefer this feature for classification; keep the split gain/phase for
    diagnostics where the absolute axis orientation is known.

    .. warning::
        This statistic assumes a **proper** modulation (``E[s^2] ~= 0``:
        QPSK, M-PSK for M>2, square QAM, GFSK). On an improper modulation it
        measures the *modulation*, not the device: a perfect, unimpaired OOK
        or BPSK signal reads ~1.0 (the maximum) because the signal's own
        pseudo-covariance is nonzero by construction. Gate on modulation
        class before trusting this as a fingerprint feature.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband (roughly zero-mean). Not mutated.

    Returns
    -------
    float
        Non-negative image ratio. ~0 for a balanced signal; grows with imbalance.
    """
    r = np.asarray(iq, dtype=np.complex128)
    power = np.mean(np.abs(r) ** 2)
    if power == 0.0:
        return 0.0
    return float(np.abs(np.mean(r * r)) / power)


def estimate_iq_imbalance(iq: np.ndarray) -> tuple[float, float]:
    """Estimate I/Q gain/phase imbalance from second-order statistics.

    For a proper (rotationally invariant) baseband signal the pseudo-covariance
    ``E[s^2]`` is zero. I/Q imbalance mixes in ``beta*conj(s)``, which makes
    ``E[r^2]`` nonzero; the ratio to the ordinary power ``E[|r|^2]`` recovers the
    imbalance parameters.

    .. warning::
        Valid only for **proper** modulations (``E[s^2] ~= 0``). Real hardware
        imbalance produces ``|c| << 1`` (an IRR of -25 dB corresponds to
        ``|c| ~= 0.1``); if ``|c| > 0.5`` the properness assumption has almost
        certainly failed (OOK, BPSK, a strong DC term, or a tone) and the
        returned numbers describe the modulation, not the device. A
        ``RuntimeWarning`` is emitted in that regime rather than silently
        returning garbage — measurement is still returned (the library never
        hides), but do not feed it to a classifier.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband. Should be roughly zero-mean; a DC term biases the
        pseudo-covariance, so remove DC first if present (the library's
        ``remove_dc`` does this). Not mutated.

    Returns
    -------
    (gain_db, phase_deg) : tuple of float
        Estimated gain mismatch (dB) and phase mismatch (degrees). A balanced
        signal returns approximately (0.0, 0.0).

    Notes
    -----
    Let ``c = E[r^2] / E[|r|^2]`` (normalized pseudo-covariance). Writing
    ``r = alpha*s + beta*conj(s)`` with proper ``s`` (``E[s^2]=0``)::

        E[|r|^2] = (|alpha|^2 + |beta|^2) * sigma^2
        E[r^2]   = 2 * alpha * beta * sigma^2
        => c     = 2*alpha*beta / (|alpha|^2 + |beta|^2)

    The model also satisfies ``alpha = 1 - conj(beta)`` (from the definitions of
    ``alpha`` and ``beta``), which closes the system. A first-order guess
    ``beta ~ c/2`` is accurate only for gain-dominated imbalance; for
    phase-dominated imbalance ``alpha`` is not real and the naive guess biases the
    phase estimate. We therefore solve the fixed point::

        alpha = 1 - conj(beta)
        beta  = c * (|alpha|^2 + |beta|^2) / (2*alpha)

    a few iterations of which converge for the small imbalances real hardware
    exhibits. Then ``g*exp(-j*phi) = 2*alpha - 1`` recovers the parameters::

        gain_db = 20*log10(|2*alpha - 1|)
        phase   = -angle(2*alpha - 1)

    Accuracy floor: this reads the signal's *own* residual ``E[s^2]`` as
    imbalance. For finite random data ``E[s^2]`` is small but nonzero, setting a
    sub-degree phase floor. Longer captures and pilot/known symbols reduce it;
    for fingerprinting the floor is itself part of the (receiver-side) signature
    to be controlled for.
    """
    r = np.asarray(iq, dtype=np.complex128)
    power = np.mean(np.abs(r) ** 2)
    if power == 0.0:
        return 0.0, 0.0
    c = np.mean(r * r) / power
    if np.abs(c) > 0.5:
        warnings.warn(
            f"pseudo-covariance ratio |c|={np.abs(c):.2f} is far outside the "
            "hardware-imbalance regime; the signal is likely improper "
            "(OOK/BPSK/DC/tone) and this estimate reflects the modulation, "
            "not the device.",
            RuntimeWarning,
            stacklevel=2,
        )
    beta = c / 2.0
    for _ in range(20):
        alpha = 1.0 - np.conj(beta)
        beta = c * (np.abs(alpha) ** 2 + np.abs(beta) ** 2) / (2.0 * alpha)
    alpha = 1.0 - np.conj(beta)
    w = 2.0 * alpha - 1.0          # = g * exp(-j*phi)
    # On garbage (improper) input the fixed point can drive |w| -> 0; the
    # RuntimeWarning above is the caller's signal, so keep numpy quiet and let
    # the degenerate value (-inf dB) pass through rather than masking it.
    with np.errstate(divide="ignore"):
        gain_db = 20.0 * np.log10(np.abs(w))
    phase_deg = -np.rad2deg(np.angle(w))
    return float(gain_db), float(phase_deg)


def estimate_cfo_ppm(
    cfo_hz: float,
    carrier_hz: float,
) -> float:
    """Convert an absolute CFO (Hz) into carrier-relative ppm.

    ppm is the device-comparable unit: it normalizes out the carrier so that the
    *same crystal* reads the same ppm regardless of which channel it's tuned to.
    The absolute ``cfo_hz`` is expected to come from the library's existing
    ``measure.estimate_cfo`` (kept separate so this module stays a thin, testable
    unit-conversion rather than duplicating the estimator).

    Parameters
    ----------
    cfo_hz : float
        Absolute carrier-frequency offset (Hz), e.g. from ``estimate_cfo``.
    carrier_hz : float
        Nominal carrier frequency (Hz).

    Returns
    -------
    float
        Offset in parts-per-million of the carrier.
    """
    if carrier_hz == 0.0:
        return 0.0
    return float(cfo_hz / carrier_hz * 1e6)


def estimate_phase_noise_variance(
    iq: np.ndarray,
    reference: np.ndarray | None = None,
) -> float:
    """Estimate residual phase-error variance (a phase-noise proxy).

    After the intended modulation and any bulk CFO are removed, what remains in
    the phase is largely oscillator phase noise. Its variance is a compact,
    device-linked summary. If a clean ``reference`` is supplied (e.g. the ideal
    modulated signal in a closed-loop test), the phase error is measured against
    it directly; otherwise the sample-to-sample phase increment variance is used,
    which is insensitive to a constant residual CFO.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband (post-recovery). Not mutated.
    reference : np.ndarray, optional
        Ideal signal of equal length. If given, variance of
        ``angle(iq * conj(reference))`` is returned.

    Returns
    -------
    float
        Phase-error variance in rad^2.

    Notes
    -----
    The differenced form removes a constant frequency offset (a linear phase
    ramp differences to a constant, which the variance ignores), isolating the
    random-walk component that phase noise contributes.

    .. warning::
        The no-reference path measures the variance of ALL phase increments.
        On a still-modulated signal with phase/frequency content (FSK, PSK,
        GFSK) the modulation dominates that variance by orders of magnitude —
        an FSK deviation of a few kHz swamps a few-hundred-Hz linewidth. Use
        this either (a) against a ``reference`` in a closed-loop test, or
        (b) on the *residual* after demodulation/recovery has removed the
        intended modulation. On raw modulated captures it is a modulation
        feature, not a device feature.
    """
    r = np.asarray(iq, dtype=np.complex128)
    if reference is not None:
        ref = np.asarray(reference, dtype=np.complex128)
        err = np.angle(r * np.conj(ref))
        return float(np.var(err))
    # No reference: use phase increments, robust to constant CFO.
    inst_phase = np.unwrap(np.angle(r))
    dphase = np.diff(inst_phase)
    return float(np.var(dphase))
