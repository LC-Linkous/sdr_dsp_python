"""Spread-spectrum: DSSS despreading (known code) and FHSS hop visualization.

Spread-spectrum signals deliberately occupy more bandwidth than the data needs
-- direct-sequence (DSSS) multiplies the data by a fast code; frequency-hopping
(FHSS) jumps the carrier around. Both are only partially in scope, honestly:

- DSSS: despreading works WELL if you KNOW the spreading code (correlate against
  it). Blind code recovery is a research problem and out of scope.
- FHSS: VISUALIZING the hops (a spectrogram) is easy and gorgeous. DECODING
  requires knowing/tracking the hop sequence; we offer hop DETECTION (find where
  energy is, per time slice) but not blind decode.

See MODULATIONS.md for the honest status table.
"""

from __future__ import annotations

import numpy as np


def dsss_despread(iq, code, samples_per_chip=1):
    """Despread a DSSS signal using a KNOWN spreading code. OUR code.

    Multiplies the signal by the (time-aligned) spreading code and integrates
    over each code period to recover the underlying data symbols. This is the
    correlation-with-known-code approach -- it works because the code correlates
    with itself and averages noise/interference down.

    code: the spreading sequence (e.g. a PN sequence), as +/-1 or complex chips.
    samples_per_chip: how many signal samples per code chip (if oversampled).

    Returns the despread data symbols (one per full code period). You must
    provide the code and rough alignment -- the library does not search for an
    unknown code (that's out of scope). For alignment search, slide the code
    with core.correlate and pick the peak.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    code = np.asarray(code, dtype=np.complex64)
    if samples_per_chip > 1:
        code = np.repeat(code, samples_per_chip)
    clen = len(code)
    if clen == 0 or len(iq) < clen:
        return np.zeros(0, dtype=np.complex64)
    n_periods = len(iq) // clen
    out = np.empty(n_periods, dtype=np.complex64)
    cc = np.conj(code)
    for k in range(n_periods):
        seg = iq[k * clen:(k + 1) * clen]
        out[k] = np.sum(seg * cc) / clen      # correlate + integrate
    return out


def fhss_detect_hops(iq, sample_rate, nfft=256, overlap=0.5, center_freq=0.0):
    """Detect frequency hops: the dominant frequency per time slice. OUR code.

    For an FHSS signal, computes a spectrogram and reports, for each time slice,
    where the energy is -- i.e. which channel the hopper is in at that moment.
    This TRACKS hops you can see; it does NOT decode the data or know the hop
    sequence (out of scope). Pair it with core.spectrogram to SEE the hops.

    Returns (times, hop_freqs) where hop_freqs[i] is the peak frequency (Hz,
    offset by center_freq) during time slice times[i].
    """
    from ..spectral import spectrogram
    freqs, times, sxx = spectrogram(iq, sample_rate, nfft=nfft,
                                    overlap=overlap, center_freq=center_freq)
    if sxx.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    hop_freqs = freqs[np.argmax(sxx, axis=1)]
    return times, hop_freqs
