"""Synthetic signal generators: ground truth we control, for testing demod
and DSP correctness without hardware. Each returns complex64 baseband IQ.
"""

from __future__ import annotations

import numpy as np


def tone(freq_hz, sample_rate, n, amp=1.0):
    """A pure complex exponential at freq_hz. The simplest test signal."""
    t = np.arange(n) / float(sample_rate)
    return (amp * np.exp(2j * np.pi * freq_hz * t)).astype(np.complex64)


def noise(n, sigma=1.0, seed=0):
    """Complex white Gaussian noise."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n)
            + 1j * rng.normal(0, sigma, n)).astype(np.complex64)


def fm_signal(message_hz, deviation_hz, sample_rate, n, amp=1.0):
    """FM-modulate a single-tone message onto a baseband carrier.

    Returns (iq, message) so a demod test can compare recovered audio against
    the known message. The instantaneous phase is the integral of the
    frequency, which for a cosine message of frequency fm is:
        phase(t) = 2*pi * (deviation/fm) * sin(2*pi*fm*t)
    """
    t = np.arange(n) / float(sample_rate)
    message = np.cos(2 * np.pi * message_hz * t)
    # integrate message -> phase. For cos(2*pi*fm*t), integral is
    # sin(2*pi*fm*t)/(2*pi*fm); scaled by 2*pi*deviation.
    phase = 2 * np.pi * deviation_hz * np.sin(2 * np.pi * message_hz * t) / (
        2 * np.pi * message_hz)
    iq = (amp * np.exp(1j * phase)).astype(np.complex64)
    return iq, message.astype(np.float64)


def ook_burst(bits, samples_per_bit, sample_rate, amp=1.0, noise_sigma=0.0,
              seed=0):
    """On-off-keyed burst: each bit held for samples_per_bit samples.

    Returns (iq, bits) where iq is amp (carrier-on) or ~0 (carrier-off) per bit.
    A demod test recovers the bits from the envelope.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    env = np.repeat(bits.astype(np.float64), samples_per_bit) * amp
    n = len(env)
    # put it on a small carrier offset so it's realistic baseband IQ
    t = np.arange(n) / float(sample_rate)
    carrier = np.exp(2j * np.pi * 0.0 * t)
    iq = (env * carrier).astype(np.complex64)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        iq = iq + (rng.normal(0, noise_sigma, n)
                   + 1j * rng.normal(0, noise_sigma, n)).astype(np.complex64)
    return iq, bits
