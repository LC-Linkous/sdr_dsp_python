"""WAV sink: write a real audio array to a playable .wav file.

Used by the receiver examples (FM/AM) instead of inlining wave-file plumbing.
Takes a real float array (demodulated audio), normalizes, and writes int16 PCM.
"""

from __future__ import annotations

import wave

import numpy as np


def write_wav(path, audio, sample_rate, normalize=True, headroom=0.9):
    """Write a real audio array to a mono 16-bit WAV.

    audio:      real-valued samples (e.g. demodulated output).
    normalize:  scale to use the int16 range (with a little headroom). If
                False, audio is assumed already in [-1, 1].
    headroom:   peak level when normalizing (0.9 = -1 dBFS-ish, avoids clipping).
    Returns the path written.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if normalize:
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio / peak * headroom
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return str(path)
