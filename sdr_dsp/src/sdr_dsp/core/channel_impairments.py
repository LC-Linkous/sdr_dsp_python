"""Device-impairment synthesis — the *forward* model for fingerprinting.

These functions stamp transmitter hardware imperfections onto a clean signal.
They are the mathematical inverse of the estimators in ``features/impairments``
and therefore serve as those estimators' test oracle: apply a known impairment
here, confirm the estimator recovers it.

Design contract (matches core §9.2):
- pure functions, no input mutation, deterministic given a seed;
- complex64 in / complex64 out;
- numpy only at runtime (no scipy needed for these);
- corrections are never silent — this module *adds* impairment explicitly, and
  every function's parameters fully describe what it did (so it is reversible in
  principle from the returned spec).

This module lives alongside ``core/channel.py`` deliberately: ``channel.py``
models *propagation* (AWGN, CFO, delay — what happens between radios), while
this module models the *transmitter itself* (what a specific device stamps on
everything it sends). A full synthetic capture is
``apply_channel(apply_device_impairments(clean, device, ...))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "add_iq_imbalance",
    "add_pa_nonlinearity",
    "add_phase_noise",
    "DeviceImpairments",
    "make_device_impairments",
    "apply_device_impairments",
]


def add_iq_imbalance(iq: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    """Apply I/Q gain/phase imbalance: ``r = alpha*s + beta*conj(s)``.

    A perfectly balanced up/down-converter has ``gain_db == 0`` and
    ``phase_deg == 0`` (identity). Real hardware mismatches the I and Q paths
    slightly; the resulting ``beta`` term injects a scaled conjugate ("image")
    of the signal, and the image-rejection ratio ``|beta/alpha|**2`` is a
    device-specific fingerprint.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input. Not mutated.
    gain_db : float
        Gain mismatch between I and Q paths, in dB. 0 = balanced.
    phase_deg : float
        Phase mismatch (quadrature error), in degrees. 0 = balanced.

    Returns
    -------
    np.ndarray
        complex64, same length as input.

    Notes
    -----
    With linear gain ``g = 10**(gain_db/20)`` and phase ``phi`` in radians::

        alpha = (1 + g*exp(-j*phi)) / 2
        beta  = (1 - g*exp(+j*phi)) / 2
        r     = alpha*s + beta*conj(s)

    This is the standard second-order I/Q imbalance model used by the estimator.
    """
    s = np.asarray(iq, dtype=np.complex64)
    g = 10.0 ** (gain_db / 20.0)
    phi = np.deg2rad(phase_deg)
    alpha = (1.0 + g * np.exp(-1j * phi)) / 2.0
    beta = (1.0 - g * np.exp(1j * phi)) / 2.0
    out = alpha * s + beta * np.conj(s)
    return out.astype(np.complex64)


def add_pa_nonlinearity(iq: np.ndarray, coeffs: np.ndarray | list[complex]) -> np.ndarray:
    """Apply a memoryless power-amplifier nonlinearity (odd-order polynomial).

    Models AM/AM and AM/PM distortion as::

        y = sum_k  c_k * x * |x|**(2k)   for k = 0, 1, 2, ...

    The linear term ``c_0`` is the small-signal gain; higher terms are the
    device-specific compression characteristic. Complex ``c_k`` capture AM/PM
    (phase distortion) as well as AM/AM (amplitude distortion).

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input. Not mutated.
    coeffs : sequence of complex
        ``[c_0, c_1, c_2, ...]``. ``[1.0]`` is a pure linear passthrough
        (identity gain, no distortion).

    Returns
    -------
    np.ndarray
        complex64, same length as input.

    Notes
    -----
    This is the odd-order-only Taylor form (each term carries an extra
    ``|x|**2``), which is the standard baseband PA model — even-order products
    fall out of band and are not represented here.

    .. warning::
        On a **constant-envelope** signal (``|x| = A`` everywhere: GFSK, FSK,
        MSK, unshaped PSK) this entire model collapses to multiplication by
        the single complex constant ``sum_k c_k * A**(2k)`` — indistinguishable
        from a bulk gain/phase, carrying zero fingerprint information. This is
        exact, not an approximation: PA nonlinearity is only observable through
        amplitude *variation* (QAM, OFDM, shaped/filtered transitions, OOK
        edges). Don't expect a PA-linked feature to move on constant-envelope
        captures.
    """
    x = np.asarray(iq, dtype=np.complex64)
    c = np.asarray(coeffs, dtype=np.complex128)
    mag2 = (np.abs(x) ** 2).astype(np.float64)
    y = np.zeros_like(x, dtype=np.complex128)
    power = np.ones_like(mag2)  # |x|**(2k), starts at k=0 -> 1
    for ck in c:
        y += ck * x * power
        power = power * mag2
    return y.astype(np.complex64)


def add_phase_noise(
    iq: np.ndarray,
    linewidth_hz: float,
    sample_rate: float,
    seed: int | None = None,
) -> np.ndarray:
    """Apply oscillator phase noise as a Wiener (random-walk) phase process.

    A free-running oscillator's phase does a random walk; the per-sample phase
    increment is zero-mean Gaussian with variance ``2*pi*linewidth/fs``. This
    produces a Lorentzian-shaped phase-noise spectrum characterized by the
    linewidth, which is device-specific.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband input. Not mutated.
    linewidth_hz : float
        Oscillator linewidth (Hz). 0 = ideal oscillator (no phase noise).
    sample_rate : float
        Sample rate (Hz).
    seed : int, optional
        Seed for the random walk; fixing it makes the result deterministic (so
        one seed == one repeatable "device").

    Returns
    -------
    np.ndarray
        complex64, same length as input.
    """
    s = np.asarray(iq, dtype=np.complex64)
    if linewidth_hz <= 0.0:
        return s.copy()
    rng = np.random.default_rng(seed)
    step_var = 2.0 * np.pi * linewidth_hz / float(sample_rate)
    increments = rng.normal(0.0, np.sqrt(step_var), size=s.shape[0])
    theta = np.cumsum(increments)
    return (s * np.exp(1j * theta)).astype(np.complex64)


@dataclass(frozen=True)
class DeviceImpairments:
    """A fixed bundle of impairments = one virtual device's signature.

    Frozen and fully inspectable: the same instance applied to any signal
    produces that "device's" fingerprint, and the fields *are* the ground truth
    the estimators are tested against.
    """

    iq_gain_db: float
    iq_phase_deg: float
    pa_coeffs: tuple[complex, ...]
    phase_noise_hz: float
    cfo_ppm: float


def make_device_impairments(seed: int) -> DeviceImpairments:
    """Draw a random-but-fixed impairment set — one repeatable virtual device.

    The distributions are deliberately in *realistic* ranges (small imbalances,
    mild compression, modest ppm offsets) so synthesized devices resemble real
    hardware spread rather than exaggerated toy values.

    Parameters
    ----------
    seed : int
        Device identity. Same seed -> identical device, every time.

    Returns
    -------
    DeviceImpairments
    """
    rng = np.random.default_rng(seed)
    gain_db = float(rng.normal(0.0, 0.3))        # ~0.3 dB path mismatch
    phase_deg = float(rng.normal(0.0, 2.0))       # ~2 deg quadrature error
    # linear gain 1.0, small 3rd-order compression, tiny complex 5th-order AM/PM
    c1 = 1.0 + 0j
    c3 = complex(rng.normal(-0.15, 0.05), rng.normal(0.0, 0.02))
    c5 = complex(rng.normal(0.03, 0.01), rng.normal(0.0, 0.01))
    phase_noise = float(abs(rng.normal(50.0, 20.0)))   # Hz linewidth
    cfo_ppm = float(rng.normal(0.0, 2.0))              # ppm crystal offset
    return DeviceImpairments(
        iq_gain_db=gain_db,
        iq_phase_deg=phase_deg,
        pa_coeffs=(c1, c3, c5),
        phase_noise_hz=phase_noise,
        cfo_ppm=cfo_ppm,
    )


def apply_device_impairments(
    iq: np.ndarray,
    device: DeviceImpairments,
    sample_rate: float,
    carrier_hz: float,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a full device signature to a clean signal.

    Order matches the physical chain: PA nonlinearity and I/Q imbalance at the
    converter, then oscillator phase noise, then the carrier-frequency offset.
    CFO is applied here (rather than deferring to ``add_cfo``) so a device
    signature is self-contained; a caller wanting propagation effects layers
    ``apply_channel`` on top afterwards.

    Parameters
    ----------
    iq : np.ndarray
        Clean complex baseband. Not mutated.
    device : DeviceImpairments
        The signature to stamp on.
    sample_rate : float
        Sample rate (Hz).
    carrier_hz : float
        Nominal carrier (Hz), needed to turn ppm into an absolute CFO.
    seed : int, optional
        Seed for the phase-noise random walk within this application.

    Returns
    -------
    np.ndarray
        complex64.
    """
    x = np.asarray(iq, dtype=np.complex64)
    x = add_pa_nonlinearity(x, device.pa_coeffs)
    x = add_iq_imbalance(x, device.iq_gain_db, device.iq_phase_deg)
    x = add_phase_noise(x, device.phase_noise_hz, sample_rate, seed=seed)
    cfo_hz = device.cfo_ppm * 1e-6 * carrier_hz
    n = np.arange(x.shape[0], dtype=np.float64)
    x = x * np.exp(1j * 2.0 * np.pi * cfo_hz * n / float(sample_rate))
    return x.astype(np.complex64)
