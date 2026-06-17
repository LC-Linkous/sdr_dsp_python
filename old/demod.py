"""Demodulation: FM, AM, OOK/ASK. Entirely sdr_dsp's own code.

scipy has none of this -- demodulation is the radio logic the library exists to
provide. Each function takes baseband complex64 IQ and returns the recovered
signal (audio for FM/AM, a bit/envelope stream for OOK).
"""

from __future__ import annotations

import numpy as np


def instantaneous_phase(iq, unwrap=True):
    """The phase angle of each complex sample. OUR code.

    Returns the per-sample phase in radians. With unwrap=True the 2*pi jumps are
    removed so the phase is continuous (useful for seeing accumulated phase /
    measuring frequency as its slope).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    phase = np.angle(iq).astype(np.float64)
    return np.unwrap(phase) if unwrap else phase


def instantaneous_frequency(iq, sample_rate=None):
    """The instantaneous frequency of a complex signal. OUR code.

    Computed by the phase discriminator: the phase change between consecutive
    samples, angle(x[n] * conj(x[n-1])). This is THE primitive under FM and FSK
    demodulation, exposed so both build on it (and so you can analyze frequency
    directly -- Doppler, drift, chirps).

    Returns radians/sample, or Hz if sample_rate is given. Output length is
    len(iq) - 1 (one difference per adjacent pair).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.zeros(0, dtype=np.float64)
    prod = iq[1:] * np.conj(iq[:-1])
    rad_per_sample = np.angle(prod).astype(np.float64)
    if sample_rate is not None:
        return rad_per_sample * float(sample_rate) / (2.0 * np.pi)
    return rad_per_sample


def fm_demod(iq, deviation_hz=None, sample_rate=None):
    """Demodulate frequency modulation via the phase discriminator. OUR code.

    FM carries the message in instantaneous frequency, so demod IS the
    instantaneous frequency (see instantaneous_frequency). Returns a real array.

    If deviation_hz and sample_rate are given, the output is scaled by the peak
    deviation to give roughly normalized audio; otherwise it returns raw
    radians/sample.
    """
    if len(np.asarray(iq)) < 2:
        return np.zeros(0, dtype=np.float64)
    if deviation_hz and sample_rate:
        inst_hz = instantaneous_frequency(iq, sample_rate=sample_rate)
        return inst_hz / float(deviation_hz)
    return instantaneous_frequency(iq)


def am_demod(iq, dc_block=True):
    """Demodulate amplitude modulation: the envelope (magnitude). OUR code.

    Returns the real envelope |iq|. With dc_block, the mean (carrier DC) is
    removed so the output swings around zero like audio.
    """
    iq = np.asarray(iq)
    env = np.abs(iq).astype(np.float64)
    if dc_block:
        env = env - np.mean(env)
    return env


def ook_envelope(iq):
    """On-off-keying / ASK front end: the magnitude envelope. OUR code.

    Returns |iq| (no DC block -- OOK threshold detection wants the absolute
    level). Feed to ``ook_slice`` to recover bits.
    """
    return np.abs(np.asarray(iq)).astype(np.float64)


def ook_slice(envelope, threshold=None):
    """Threshold an OOK envelope into a 0/1 stream. OUR code.

    threshold: level above which a sample is '1'. If None, uses the midpoint
    between the envelope's min and max (a simple, robust default for a clean
    capture). Returns a uint8 array of 0/1.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if threshold is None:
        threshold = (env.min() + env.max()) / 2.0
    return (env > threshold).astype(np.uint8)


def edges(bits):
    """Indices where a 0/1 stream transitions, and the run lengths. OUR code.

    Returns (transition_indices, run_lengths, run_values): the sample index of
    each transition, how many samples each run lasted, and whether that run was
    0 or 1. The building block for recovering symbol timing from a sliced
    on/off stream.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), \
            np.array([], dtype=np.uint8)
    change = np.nonzero(np.diff(bits))[0] + 1
    bounds = np.concatenate([[0], change, [len(bits)]])
    run_lengths = np.diff(bounds)
    run_values = bits[bounds[:-1]]
    return change, run_lengths, run_values


def estimate_symbol_rate(bits, sample_rate):
    """Estimate samples-per-symbol from the shortest run in a sliced stream.

    The shortest on/off run is (usually) one symbol period. OUR code -- a
    robust-enough heuristic for clean OOK/ASK bursts. Returns
    (samples_per_symbol, symbol_rate_hz). Use the shortest run as the unit and
    round longer runs to multiples of it.
    """
    _, run_lengths, _ = edges(bits)
    # ignore the leading/trailing idle runs (often huge) by taking the modal
    # short run: the minimum of the interior runs is the symbol period.
    interior = run_lengths[1:-1] if len(run_lengths) > 2 else run_lengths
    if len(interior) == 0:
        return 0.0, 0.0
    spb = float(np.min(interior))
    return spb, (sample_rate / spb if spb else 0.0)


def slice_to_symbols(bits, samples_per_symbol):
    """Collapse an over-sampled 0/1 stream into one bit per symbol. OUR code.

    Given the samples-per-symbol, walk each run and emit its value repeated
    round(run_length / spb) times. Returns a uint8 array of symbol bits.
    """
    _, run_lengths, run_values = edges(bits)
    spb = float(samples_per_symbol)
    if spb <= 0:
        return np.asarray(bits, dtype=np.uint8)
    out = []
    for length, val in zip(run_lengths, run_values):
        count = max(1, int(round(length / spb)))
        out.extend([int(val)] * count)
    return np.array(out, dtype=np.uint8)


def deemphasis(audio, sample_rate, tau_us=75.0):
    """Single-pole de-emphasis filter for broadcast FM audio. OUR code.

    Broadcast FM pre-emphasizes high frequencies before transmission; the
    receiver must de-emphasize them back. A one-pole IIR does it:
        y[n] = a*x[n] + (1-a)*y[n-1],   a = dt / (tau + dt)
    tau_us: time constant (75 us in the Americas/Korea, 50 us elsewhere).
    """
    audio = np.asarray(audio, dtype=np.float64)
    tau = tau_us * 1e-6
    dt = 1.0 / float(sample_rate)
    a = dt / (tau + dt)
    out = np.empty_like(audio)
    acc = 0.0
    for i, x in enumerate(audio):
        acc = a * x + (1.0 - a) * acc
        out[i] = acc
    return out


def fsk_demod(iq, sample_rate, threshold_hz=0.0):
    """Demodulate 2-level frequency-shift keying. OUR code.

    FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
    instantaneous frequency, then a threshold: above threshold_hz -> 1, below
    -> 0. With the default threshold 0, it splits on the sign of the frequency
    deviation (correct when the two tones straddle the center frequency, which
    is the common case after tuning to baseband).

    Returns a uint8 per-sample bit stream; feed to the timing-recovery helpers
    (estimate_symbol_rate / slice_to_symbols) to get symbols. Covers GFSK/MSK
    well enough for typical ISM-band sensors and pagers.
    """
    inst = instantaneous_frequency(iq, sample_rate=sample_rate)
    return (inst > float(threshold_hz)).astype(np.uint8)


def ssb_demod(iq, sample_rate, sideband="usb", bfo_hz=0.0):
    """Demodulate single-sideband (USB or LSB). OUR code.

    SSB transmits one sideband of an AM signal with the carrier suppressed. In a
    complex baseband capture the two sidebands are ALREADY separated: positive
    frequencies are the upper sideband, negative frequencies the lower. So we
    select a sideband by keeping only positive (USB) or only negative (LSB)
    frequency content, then take the real part as audio.

    (Note: simply conjugating and taking the real part does NOT work --
    real(z) == real(conj(z)) -- so sideband selection must happen in the
    frequency domain, which is what we do here.)

    bfo_hz applies a beat-frequency-oscillator shift to fine-tune pitch, as a
    real radio's BFO does (user-controlled).

    Returns the real demodulated audio.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if bfo_hz:
        t = np.arange(n) / float(sample_rate)
        iq = iq * np.exp(2j * np.pi * float(bfo_hz) * t).astype(np.complex64)

    sb = sideband.lower()
    if sb not in ("usb", "lsb"):
        raise ValueError("sideband must be 'usb' or 'lsb'")

    # select the sideband in the frequency domain: zero out the half we don't
    # want, then return the real part of the inverse transform.
    spec = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n)
    if sb == "usb":
        spec[freqs < 0] = 0.0      # keep positive frequencies
    else:
        spec[freqs > 0] = 0.0      # keep negative frequencies
    selected = np.fft.ifft(spec)
    return np.real(selected).astype(np.float64)


def bpsk_demod(iq, normalize_phase=True):
    """Demodulate binary phase-shift keying (coherent-ish). OUR code.

    BPSK encodes bits as 0 or pi phase. With the carrier already at baseband and
    roughly phase-aligned, the sign of the real part recovers the bits. This is
    a SIMPLE demod: it assumes the signal is already carrier-aligned (no Costas
    loop / carrier recovery). For captures with a residual carrier offset,
    correct it first (see estimate_cfo / frequency_shift) -- the library does
    not auto-recover the carrier.

    Returns (bits, soft) where bits is uint8 (0/1) and soft is the real-part
    decision statistic (useful for confidence / plotting a constellation).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if normalize_phase and len(iq):
        # remove a constant phase offset by aligning the dominant axis to real:
        # rotate so the mean squared phase lands on the real axis.
        rot = np.exp(-1j * 0.5 * np.angle(np.mean(iq ** 2)))
        iq = iq * rot
    soft = np.real(iq).astype(np.float64)
    bits = (soft > 0).astype(np.uint8)
    return bits, soft
