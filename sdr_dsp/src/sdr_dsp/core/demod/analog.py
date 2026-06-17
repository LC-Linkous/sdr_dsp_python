"""Analog demodulation: FM, AM, SSB, and FM de-emphasis.

The radio DSP is our own code; these build on the phase primitives and numpy's
FFT (for SSB sideband selection).
"""

from __future__ import annotations

import numpy as np

from .phase import instantaneous_frequency


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


def dsb_sc_demod(iq, sample_rate, bfo_hz=0.0):
    """Demodulate double-sideband suppressed-carrier (DSB-SC). OUR code.

    DSB-SC is AM with the carrier removed -- both sidebands, no carrier spike.
    With the carrier suppressed there's no envelope to follow, so recovery needs
    a coherent reference. For a complex baseband capture centered on the
    (suppressed) carrier, the real part IS the message (the two sidebands beat
    back together). A bfo_hz shift fine-tunes if the center is slightly off.

    This is the conceptual midpoint between AM (carrier present, envelope) and
    SSB (one sideband): DSB-SC keeps both sidebands but drops the carrier.
    Returns the real demodulated message.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if bfo_hz:
        n = len(iq)
        t = np.arange(n) / float(sample_rate)
        iq = iq * np.exp(2j * np.pi * float(bfo_hz) * t).astype(np.complex64)
    return np.real(iq).astype(np.float64)


# Morse code lookup (International Morse) for the CW decoder.
_MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", ".-.-.-": ".", "--..--": ",", "..--..": "?",
    "-..-.": "/", "-.-.--": "!", ".-.-.": "+", "-...-": "=",
}


def cw_decode(bits, samples_per_symbol):
    """Decode Morse (CW) from a sliced on/off stream. OUR code.

    CW is on-off keying at audio rates: a "dit" is one unit on, a "dah" is three
    units on, with one-unit gaps inside a character, three-unit gaps between
    characters, and seven-unit gaps between words. Given a 0/1 stream and the
    unit length (samples_per_symbol = one dit), this groups the on/off runs into
    dits/dahs and gaps, then looks up the characters.

    Front end: get the on/off stream from ook_envelope + ook_slice on a
    tone-filtered capture, and estimate samples_per_symbol from the shortest
    "on" run (one dit). Returns the decoded text string.

    Honest note: CW timing is famously loose (hand-keyed sending varies), so the
    unit estimate and the dit/dah threshold may need tuning on real signals.
    """
    from .timing import edges
    _, run_lengths, run_values = edges(bits)
    spb = float(samples_per_symbol)
    if spb <= 0 or len(run_lengths) == 0:
        return ""
    text = []
    symbol = []
    for length, val in zip(run_lengths, run_values):
        units = length / spb
        if val == 1:                       # tone ON: dit (~1) or dah (~3)
            symbol.append("." if units < 2 else "-")
        else:                              # tone OFF: gap
            if units < 2:                  # intra-character gap: continue
                continue
            # character or word gap: flush the current symbol
            if symbol:
                text.append(_MORSE.get("".join(symbol), "?"))
                symbol = []
            if units >= 5:                 # word gap
                text.append(" ")
    if symbol:                             # flush trailing symbol
        text.append(_MORSE.get("".join(symbol), "?"))
    return "".join(text)
