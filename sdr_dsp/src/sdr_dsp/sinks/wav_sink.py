"""WAV sink: write a real audio array to a playable .wav file.

Used by the receiver examples (FM/AM) instead of inlining wave-file plumbing.
Takes a real float array (demodulated audio), normalizes, and writes int16 PCM.
"""

from __future__ import annotations

import wave

import numpy as np


def write_wav(path, audio, sample_rate, normalize=True, headroom=0.9,
              percentile=99.9):
    """Write a real audio array to a mono 16-bit WAV.

    audio:      real-valued samples (e.g. demodulated output).
    normalize:  scale to use the int16 range (with a little headroom). If
                False, audio is assumed already in [-1, 1].
    headroom:   peak level when normalizing (0.9 = -1 dBFS-ish, avoids clipping).
    percentile: the |audio| percentile treated as "full scale" when
                normalizing. 100 reproduces plain peak normalization.

    Why a percentile instead of the peak: demodulated audio routinely carries
    short transients far above the program material -- filter delay lines
    settling, a phase discriminator fed a near-zero-magnitude sample, an IIR
    starting from a zero accumulator. Dividing by the raw maximum lets one
    such sample set the scale for the whole file, pushing everything else
    toward zero: the file plays as a click followed by near-silence. Scaling
    to a high percentile and clipping the few samples above it keeps the
    program material at a sensible level. Trim known settling regions at the
    source too; this is a backstop, not a substitute.

    Returns the path written.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        raise ValueError("audio is empty; nothing to write")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or inf")
    if not 0 < headroom <= 1.0:
        raise ValueError(f"headroom must be in (0, 1], got {headroom}")
    if not 0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    # Mono is a 1-D array; stereo is (N, 2) -- e.g. from fm_stereo_decode
    # stacked as np.column_stack([left, right]). More than 2 columns is not a
    # WAV this writer makes.
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2 and audio.shape[1] == 2:
        channels = 2
    else:
        raise ValueError(
            f"audio must be 1-D (mono) or (N, 2) (stereo), got shape "
            f"{audio.shape}")

    if normalize:
        # ONE shared scale across all channels, not per-channel: normalizing
        # left and right independently would change their relative level and
        # collapse the stereo image (a hard-left sound would drift center).
        mag = np.abs(audio)
        ref = float(np.percentile(mag, percentile))
        if ref <= 0.0:                      # near-silent or heavily sparse
            ref = float(np.max(mag))
        if ref <= 0.0:                      # all zeros
            ref = 1.0
        audio = audio / ref * headroom

    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        # wave expects interleaved frames; a (N, 2) array is already
        # L,R,L,R... in row-major order, so ravel gives the right layout.
        w.writeframes(pcm.tobytes())
    return str(path)
