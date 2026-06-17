"""A simulated propagation channel: degrade a signal the way a real link does.

Between a transmitter and a receiver, a signal picks up impairments -- thermal
noise, a carrier frequency offset (the two radios' oscillators never match
exactly), a propagation delay, amplitude scaling, and sometimes fading. Testing
the TX->RX chain honestly means putting those impairments in deliberately, in
meaningful units, rather than the magic noise amplitudes the demod examples each
hand-rolled.

`apply_channel` is that one reusable channel. You ask for impairments in the
units you actually reason about -- SNR in dB, CFO in Hz, delay in samples -- and
it applies them. It is honest about what it does: every impairment is explicit
and off by default, so a channel with no arguments returns the signal unchanged.
This is the consolidation of the scattered `0.02*(randn+1j*randn)` lines into
one place that means something.

It pairs with the measurement side: set snr_db here, recover it with
core.snr_db; set cfo_hz here, measure it with core.estimate_cfo.
"""

from __future__ import annotations

import numpy as np


def add_noise(iq, snr_db, rng=None):
    """Add complex AWGN at a specified SNR (dB) relative to the signal. OUR code.

    Measures the signal's mean power, computes the noise power for the requested
    SNR, and adds complex Gaussian noise at that level. This is the honest way to
    set noise -- by the SNR you want, not an arbitrary amplitude.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0:
        return iq
    rng = rng or np.random.default_rng()
    sig_power = float(np.mean(np.abs(iq) ** 2))
    if sig_power <= 0:
        return iq
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(iq))
                                        + 1j * rng.standard_normal(len(iq)))
    return (iq + noise).astype(np.complex64)


def add_cfo(iq, cfo_hz, sample_rate):
    """Apply a carrier frequency offset (Hz). OUR code.

    Multiplies by a complex exponential at cfo_hz -- the rotation a real link
    imposes when the TX and RX oscillators differ. Recoverable on the RX side by
    carrier recovery, or measurable with estimate_cfo.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        return iq
    t = np.arange(n) / float(sample_rate)
    return (iq * np.exp(2j * np.pi * cfo_hz * t)).astype(np.complex64)


def add_delay(iq, delay_samples):
    """Delay the signal by an integer number of samples (zero-pad the front).

    Models propagation delay / a late frame start. The output is the same length
    as the input (the tail is truncated); negative delay advances. OUR code.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    d = int(delay_samples)
    if d == 0 or len(iq) == 0:
        return iq
    if d > 0:
        return np.concatenate([np.zeros(d, dtype=np.complex64), iq[:-d]]) \
            if d < len(iq) else np.zeros(len(iq), dtype=np.complex64)
    d = -d
    return np.concatenate([iq[d:], np.zeros(d, dtype=np.complex64)]) \
        if d < len(iq) else np.zeros(len(iq), dtype=np.complex64)


def apply_channel(iq, sample_rate=None, snr_db=None, cfo_hz=None,
                  delay_samples=0, scale=1.0, phase=0.0, seed=None):
    """Pass a signal through a simulated channel. OUR code.

    Applies, in order: delay -> scale/phase -> CFO -> noise. Every impairment is
    explicit and optional; with no impairments set, returns the signal unchanged
    (a no-op channel). Units are the ones you reason in:

        snr_db:        target SNR in dB (None = noiseless). Needs nothing else.
        cfo_hz:        carrier frequency offset in Hz (requires sample_rate).
        delay_samples: integer sample delay (propagation / late start).
        scale:         amplitude multiplier (path loss / gain).
        phase:         constant phase rotation in radians.
        seed:          seed the noise RNG for reproducible channels.

    Returns the degraded complex64 signal, same length as the input. Pair it with
    the chain: modulate -> build_frame -> apply_channel -> demod -> find_frames,
    to test how the link holds up before any hardware.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0:
        return iq
    rng = np.random.default_rng(seed)

    out = add_delay(iq, delay_samples)
    if scale != 1.0 or phase != 0.0:
        out = (out * (scale * np.exp(1j * phase))).astype(np.complex64)
    if cfo_hz:
        if sample_rate is None:
            raise ValueError("cfo_hz requires sample_rate")
        out = add_cfo(out, cfo_hz, sample_rate)
    if snr_db is not None:
        out = add_noise(out, snr_db, rng=rng)
    return out.astype(np.complex64)
