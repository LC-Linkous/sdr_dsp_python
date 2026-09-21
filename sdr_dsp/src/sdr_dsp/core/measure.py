"""Measurement: power, SNR, occupied bandwidth. OUR code (simple math on IQ).

These build on the spectral module and tie back to hackrfpy's relative_power_db
for a dB reference. None of this is in scipy -- it is radio-specific.
"""

from __future__ import annotations

import numpy as np

from .spectral import psd


def power_dbfs(iq):
    """Mean power of a complex signal in dBFS (dB relative to |amp|=1)."""
    iq = np.asarray(iq)
    if len(iq) == 0:
        return float("-inf")
    p = float(np.mean(iq.real.astype(np.float64) ** 2
                      + iq.imag.astype(np.float64) ** 2))
    return 10.0 * np.log10(p + 1e-20)


def snr_db(iq, sample_rate, signal_band_hz, nfft=1024):
    """Estimate SNR by comparing in-band power to out-of-band (noise) power.

    signal_band_hz: (low, high) frequency range (relative to center) holding
                    the signal. Everything else in the spectrum is treated as
                    noise. A coarse but useful estimate.
    """
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    psd_lin = 10.0 ** (psd_db / 10.0)
    lo, hi = signal_band_hz
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any() or in_band.all():
        raise ValueError("signal_band must cover part (not all) of the span")
    sig_p = float(np.mean(psd_lin[in_band]))
    noise_p = float(np.mean(psd_lin[~in_band]))
    return 10.0 * np.log10(sig_p / (noise_p + 1e-20))


def capture_health(iq, sample_rate, channel_bw=None, adc_bits=8,
                   min_counts=4.0, min_excess_db=3.0, nfft=8192):
    """Assess whether a recording actually contains a signal. OUR code.

    Answers the question that precedes all DSP: is there anything in this
    file? Nearly every "it ran without errors but the output is silent or
    sounds like hiss" report is a property of the recording -- an antenna
    that wasn't connected, gain set too low, or a capture tuned somewhere
    with nothing on it -- and no amount of processing recovers a signal that
    was never recorded.

    Two independent checks:

    ``adc_counts``
        How much of the digitizer's range the recording used, recovered by
        scaling the normalized peak back up by the converter's full scale.
        A capture peaking at a couple of counts out of 127 is quantization
        noise; there is no signal in it at any gain setting downstream.

        Measured per component, as max(|I|, |Q|), because I and Q are
        quantized by separate converters and clipping happens to each of
        them independently. The complex magnitude |I + jQ| would read up to
        sqrt(2) higher and can exceed full scale outright -- a sample at
        I=127, Q=127 is 179 by that measure, which is not a number any 8-bit
        converter can produce.

    ``channel_excess_db``
        Power inside ``channel_bw`` of DC, relative to the outer 20% of the
        span. A real carrier is a hump standing above its surroundings; a
        flat spectrum means nothing is there. Skipped when channel_bw is
        None or the capture is too short to average.

    Args:
        iq:          complex samples normalized to +/-1 (as sources return).
        sample_rate: Hz, used to place the channel band.
        channel_bw:  half-width in Hz of the channel of interest. None skips
                     the spectral check.
        adc_bits:    converter width; 8 for HackRF's ci8. Counts scale is
                 2**(adc_bits-1) = 128, matching the loader's /128
                 normalization, so a full-scale sample reads as +/-128.
        min_counts:  ADC counts below which the capture is called empty.
        min_excess_db: in-band excess below which no carrier is called.
        nfft:        FFT size for the averaged spectrum.

    Returns a dict with ``ok`` (bool), ``adc_counts``, ``channel_excess_db``
    (None if not computed), ``peak_dbfs``, and ``reasons`` (list of strings
    describing each failure, empty when ok).

    This is a screening tool, not a detector. A signal far weaker than the
    noise floor in the channel -- spread spectrum below the noise, say --
    will be reported as absent, which is the right answer for "can I demod
    this directly" and the wrong one for "is there anything here at all."
    """
    iq = np.asarray(iq)
    full_scale = float(2 ** (adc_bits - 1))
    reasons = []

    if iq.size == 0:
        return {"ok": False, "adc_counts": 0.0, "channel_excess_db": None,
                "peak_dbfs": float("-inf"), "reasons": ["capture is empty"]}

    peak = float(max(np.max(np.abs(iq.real)), np.max(np.abs(iq.imag))))
    adc_counts = peak * full_scale
    if adc_counts < min_counts:
        reasons.append(
            f"capture peaks at ~{adc_counts:.1f} of {full_scale:.0f} ADC "
            f"counts, which is the noise floor: check the antenna, raise "
            f"gain, and confirm the tuned frequency")

    excess = None
    if channel_bw is not None and sample_rate > 0:
        nseg = min(64, iq.size // nfft)
        if nseg >= 4:
            win = np.hanning(nfft)
            acc = np.zeros(nfft)
            for k in range(nseg):
                seg = iq[k * nfft:(k + 1) * nfft] * win
                acc += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
            psd_lin = acc / nseg
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
            # Noise reference: a MID-BAND annulus, not the outer band edges.
            # The SDR's own baseband anti-alias filter (~0.75 * fs on a
            # HackRF) rolls the edges off, so "in-band vs edges" reads tens
            # of dB of excess on a capture containing nothing but receiver
            # noise -- an empty capture at healthy gain would pass. The
            # annulus between 1.5x the channel and 0.35 * fs sits inside the
            # anti-alias passband, where noise is flat. The MEDIAN makes the
            # reference robust to an adjacent station landing in the annulus,
            # and averaging happens in linear power (a dB-domain mean is a
            # geometric mean, biased low for spiky spectra).
            in_band = np.abs(freqs) <= channel_bw
            ref_band = ((np.abs(freqs) >= 1.5 * channel_bw)
                        & (np.abs(freqs) <= 0.35 * sample_rate))
            if in_band.any() and ref_band.any():
                excess = float(
                    10.0 * np.log10(psd_lin[in_band].mean() + 1e-30)
                    - 10.0 * np.log10(np.median(psd_lin[ref_band]) + 1e-30))
                if excess < min_excess_db:
                    reasons.append(
                        f"channel is only {excess:+.1f} dB above the "
                        f"surrounding noise floor, so no carrier is present: "
                        f"demodulating this yields noise, not signal")

    return {"ok": not reasons, "adc_counts": adc_counts,
            "channel_excess_db": excess,
            "peak_dbfs": 20.0 * np.log10(peak + 1e-20),
            "reasons": reasons}


def fm_pilot_excess_db(iq, sample_rate, pilot_hz=19_000.0, nfft=16384):
    """How far the 19 kHz stereo pilot stands above the demodulated noise
    floor, in dB. The decisive "is this an FM broadcast station" check.

    The channel-power test in ``capture_health`` answers "is there energy
    here"; this answers the sharper question "is that energy a broadcast FM
    station". Nearly every FM broadcast transmits a 19 kHz pilot at ~9% of
    deviation, and the FM noise floor rises with frequency (the noise
    triangle), so a pilot standing above its own neighborhood is essentially
    impossible to produce from noise. Use it in gain searches and capture
    validation when the target is known to be broadcast FM; skip it for
    narrowband FM (NOAA weather etc.), which has no pilot.

    Works directly on baseband IQ: channelizes to the FM channel first,
    then demodulates (phase discriminator) and compares the PSD at pilot_hz
    against the median PSD in the surrounding 15-23 kHz region, pilot bins
    excluded. The channel filter is not optional: a discriminator fed the
    full capture bandwidth is corrupted by out-of-channel noise and adjacent
    stations, and on real wide captures (e.g. 8 Msps) the pilot all but
    disappears from its output -- a real, by-ear-verified broadcast capture
    measured +1.8 dB unfiltered and +20 dB channelized.

    Returns dB (positive = pilot present; > ~6 dB is a confident yes), or
    None when the capture is too short (< 4 * nfft demodulated samples) or
    the rate is too low to see the pilot.

    OUR code.
    """
    iq = np.asarray(iq)
    if sample_rate <= 2.5 * pilot_hz or iq.size < 4 * nfft + 1:
        return None
    # Channelize: lowpass to the FM channel and decimate to ~500 kSps
    # before discriminating. Taps scale with the decimation so the
    # transition band stays proportionate at any input rate.
    decim = max(1, int(sample_rate // 480_000))
    if decim > 1:
        from .filters import design_lowpass, fir_apply
        taps = design_lowpass(120_000.0, sample_rate,
                              num_taps=8 * decim + 1)
        iq = fir_apply(iq, taps)[::decim]
        sample_rate = sample_rate / decim
        if iq.size < 4 * nfft + 1:
            nfft = 1 << max(10, int(np.log2(max(16, iq.size // 4))))
            if iq.size < 4 * nfft + 1:
                return None
    # phase discriminator: angle(x[n] * conj(x[n-1])), radians/sample
    demod = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float64)
    nseg = min(32, demod.size // nfft)
    win = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    for k in range(nseg):
        seg = demod[k * nfft:(k + 1) * nfft] * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    half_bin = sample_rate / nfft
    pilot = np.abs(freqs - pilot_hz) <= max(200.0, 2 * half_bin)
    hood = ((freqs >= pilot_hz - 4000.0) & (freqs <= pilot_hz + 4000.0)
            & ~pilot)
    if not pilot.any() or not hood.any():
        return None
    return float(10.0 * np.log10(acc[pilot].max() + 1e-30)
                 - 10.0 * np.log10(np.median(acc[hood]) + 1e-30))


def occupied_bandwidth(iq, sample_rate, fraction=0.99, nfft=1024):
    """Bandwidth containing ``fraction`` of the total power (e.g. 99%).

    Returns bandwidth in Hz. Integrates the PSD and finds the central band
    holding the requested fraction of total power.
    """
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    p = 10.0 ** (psd_db / 10.0)
    total = float(np.sum(p))
    if total <= 0:
        return 0.0
    # cumulative from the spectrum center outward
    order = np.argsort(np.abs(freqs))  # nearest-to-center first
    cum = np.cumsum(p[order])
    idx = np.searchsorted(cum, fraction * total)
    idx = min(idx, len(order) - 1)
    bw = 2.0 * float(np.abs(freqs[order][idx]))
    return bw


def find_bursts(iq, sample_rate=None, threshold=None, min_gap=0, min_len=1):
    """Find where signal energy is present: burst start/stop indices. OUR code.

    Thresholds the magnitude envelope and returns the spans where it's above the
    threshold -- "where is the signal?" for packet/burst captures. The decoder
    examples did this ad-hoc; this is the reusable version.

    threshold: envelope level for "on". If None, uses the midpoint between the
               envelope's 1st percentile (the noise floor) and its peak. The
               floor is a low PERCENTILE, not the median, deliberately: the
               median is only the noise floor when the record is mostly noise.
               On a capture dominated by one long burst (a triggered packet
               capture), the median IS the signal level, and a median-based
               threshold lands above the signal and shreds one burst into
               fragments. The percentile floor handles both regimes, as long
               as at least ~1% of the record is signal-free. If your record
               has NO quiet samples at all, or bursts sit near the noise
               level, set threshold explicitly -- an automatic threshold is a
               convenience, not a measurement.
    min_gap:   merge bursts separated by fewer than this many samples (bridges
               brief dropouts within one packet).
    min_len:   discard bursts shorter than this (rejects noise blips).

    Returns a list of (start, stop) sample-index pairs (stop exclusive). If
    sample_rate is given, also accepts/returns nothing different -- indices are
    always in samples (convert to time yourself: start/sample_rate).
    """
    env = np.abs(np.asarray(iq))
    if len(env) == 0:
        return []
    if threshold is None:
        floor = float(np.percentile(env, 1))
        pk = float(np.max(env))
        threshold = floor + 0.5 * (pk - floor)
    on = env > threshold
    if not on.any():
        return []
    # find rising/falling edges of the boolean "on" mask
    edges_ = np.diff(on.astype(np.int8))
    starts = [int(i) for i in np.nonzero(edges_ == 1)[0] + 1]
    stops = [int(i) for i in np.nonzero(edges_ == -1)[0] + 1]
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        stops = stops + [len(on)]
    spans = list(zip(starts, stops))
    # merge close spans
    if min_gap > 0 and spans:
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s - merged[-1][1] <= min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        spans = merged
    # drop short spans
    spans = [(s, e) for s, e in spans if e - s >= min_len]
    return spans


def estimate_cfo(iq, sample_rate, nfft=None):
    """Estimate a signal's carrier frequency offset from band center. OUR code.

    Finds the dominant spectral component -- where the signal actually sits
    relative to 0 Hz. This MEASURES the offset; it does NOT apply any
    correction (correcting would change the data, and that's the user's call --
    pass the result to frequency_shift / tune_to_baseband if you want to
    correct). Returns the offset in Hz.

    For a clean single-carrier signal this is just the FFT peak. For modulated
    signals it estimates the spectral centroid of the strongest region.

    NOT for FSK. An FSK burst's strongest components are the mark/space tones
    at +/-deviation_hz, so this returns roughly +/-deviation, NOT the carrier
    offset -- and "correcting" with it moves the whole signal by a deviation,
    which is worse than no correction. For FSK, threshold at the offset
    directly instead: fsk_demod(iq, fs, threshold_hz="auto") uses the
    amplitude-weighted mean of the instantaneous frequency, which IS the
    offset when mark/space time is roughly balanced.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) == 0:
        return 0.0
    if nfft is None:
        nfft = min(len(iq), 8192)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq[:nfft], nfft)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
    return float(freqs[int(np.argmax(spec))])


def estimate_fm_cfo(iq, sample_rate, *, channel_bw=100_000.0, trim_hz=None):
    """Estimate a broadcast-FM capture's carrier frequency offset, in Hz.

    Method: the phase discriminator's DC term. Under a constant carrier
    offset df, ``angle(x[n] * conj(x[n-1]))`` picks up that same df at every
    sample, while the program audio (the wanted FM message) averages to zero
    over a long window. So the MEAN of the demodulated instantaneous frequency
    IS the carrier offset. This is the fine, accurate estimator for FM, and is
    distinct from ``estimate_cfo``: that one finds where energy sits in the
    passband (a coarse, spread-out measurement for a wideband FM signal),
    whereas this recovers the sub-kHz crystal offset a real receiver must
    correct.

    The capture is channelized (lowpass + decimate to the FM channel) before
    discriminating, for exactly the reason ``fm_pilot_excess_db`` documents: a
    discriminator fed the full capture bandwidth is dominated by out-of-channel
    noise and adjacent stations, and the mean of that is meaningless. On the
    channelized signal the mean is the offset.

    Why NOT the 19 kHz pilot: with a differential discriminator a carrier
    offset adds a DC term but does NOT move the pilot -- differentiation is
    invariant to a constant frequency shift -- so the pilot's position carries
    no CFO information (this is the trap the TODO's "or the pilot frequency"
    wording invites). The pilot's real role here is a presence gate: confirm a
    station is actually there with ``fm_pilot_excess_db`` before trusting this
    number, because the mean of noise is not a carrier offset.

    Args:
        channel_bw: FM channel half-not-needed; used only to size the click
            trim. Broadcast FM is ~100 kHz half-bandwidth of program content.
        trim_hz: discriminator "clicks" (phase wraps in deep fades / low SNR)
            are impulsive outliers that bias a plain mean. Samples whose
            demodulated frequency exceeds trim_hz in magnitude are dropped
            before averaging. Default 1.5 * channel_bw keeps the full +/-75 kHz
            deviation swing while rejecting the near-+/-Nyquist click spikes.
            Pass None to disable trimming.

    Returns the offset in Hz (positive = the signal sits above the tuned
    center), or None when the capture is too short or the rate too low to
    channelize to the FM channel.

    OUR code.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if sample_rate <= 4.0 * channel_bw or iq.size < 4096:
        return None
    # Channelize to the FM channel, mirroring fm_pilot_excess_db: lowpass and
    # decimate to ~500 kSps before discriminating so out-of-channel energy
    # does not pollute the mean. Taps scale with the decimation.
    decim = max(1, int(sample_rate // 480_000))
    if decim > 1:
        from .filters import design_lowpass, fir_apply
        taps = design_lowpass(120_000.0, sample_rate, num_taps=8 * decim + 1)
        iq = fir_apply(iq, taps)[::decim]
        sample_rate = sample_rate / decim
    if iq.size < 2:
        return None
    # phase discriminator -> instantaneous frequency in Hz
    inst_hz = (np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float64)
               * sample_rate / (2.0 * np.pi))
    trim = 1.5 * float(channel_bw) if trim_hz is None else float(trim_hz)
    if trim > 0:
        keep = np.abs(inst_hz) <= trim
        if keep.any():
            inst_hz = inst_hz[keep]
    if inst_hz.size == 0:
        return None
    return float(np.mean(inst_hz))


def correct_fm_cfo(iq, sample_rate, cfo_hz=None, **kwargs):
    """Remove a broadcast-FM carrier frequency offset. OUR code.

    If ``cfo_hz`` is None it is measured with ``estimate_fm_cfo`` (any keyword
    arguments are forwarded to it). The correction is a frequency shift by
    ``-cfo_hz`` -- ``frequency_shift(iq, -cfo_hz, sample_rate)`` -- which brings
    the carrier back to center. Estimating and correcting are separated on
    purpose (``estimate_fm_cfo`` never mutates data), and this is the
    convenience that does both in one call for the common case.

    Returns ``(corrected_iq, cfo_hz)`` so the applied correction is visible to
    the caller. When the offset cannot be estimated (capture too short or rate
    too low), returns the input unchanged with ``cfo_hz = 0.0``.
    """
    if cfo_hz is None:
        cfo_hz = estimate_fm_cfo(iq, sample_rate, **kwargs)
    if cfo_hz is None:
        return np.asarray(iq, dtype=np.complex64), 0.0
    from .mixing import frequency_shift
    return frequency_shift(iq, -float(cfo_hz), sample_rate), float(cfo_hz)
