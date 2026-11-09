"""Measurement: power, SNR, occupied bandwidth. OUR code (simple math on IQ).

These build on the spectral module and tie back to hackrfpy's relative_power_db
for a dB reference. None of this is in scipy -- it is radio-specific.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .spectral import psd
from ._validate import pair


def power_dbfs(iq: Any) -> float:
    """Mean power of a complex signal in dBFS (dB relative to |amp|=1)."""
    iq = np.asarray(iq)
    if len(iq) == 0:
        return float("-inf")
    p = float(np.mean(iq.real.astype(np.float64) ** 2
                      + iq.imag.astype(np.float64) ** 2))
    return 10.0 * np.log10(p + 1e-20)


def snr_db(iq: Any, sample_rate: float, signal_band_hz: Any, nfft: int = 1024) -> float:
    """Estimate SNR by comparing in-band power to out-of-band (noise) power.

    signal_band_hz: (low, high) frequency range (relative to center) holding
                    the signal. Everything else in the spectrum is treated as
                    noise. A coarse but useful estimate.
    """
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    psd_lin = 10.0 ** (psd_db / 10.0)
    lo, hi = pair("signal_band_hz", signal_band_hz)
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any() or in_band.all():
        raise ValueError("signal_band must cover part (not all) of the span")
    sig_p = float(np.mean(psd_lin[in_band]))
    noise_p = float(np.mean(psd_lin[~in_band]))
    return 10.0 * np.log10(sig_p / (noise_p + 1e-20))


# Machine-readable verdicts, worst first. The first failing check in this
# order is the verdict; `reasons` still lists every failure.
HEALTH_VERDICTS = ("clipping", "under_driven", "stuck_dc", "no_carrier", "ok")


def capture_health(
    iq: Any,
    sample_rate: float,
    channel_bw_hz: float | None = None,
    adc_bits: int = 8,
    min_counts: float = 4.0,
    min_excess_db: float = 3.0,
    nfft: int = 8192,
    max_clip_fraction: float = 1e-3,
    max_dc_ratio: float = 1.0,
) -> dict[str, Any]:
    """Assess whether a recording actually contains a signal. OUR code.

    Answers the question that precedes all DSP: is there anything in this
    file? Nearly every "it ran without errors but the output is silent or
    sounds like hiss" report is a property of the recording -- an antenna
    that wasn't connected, gain set too low, or a capture tuned somewhere
    with nothing on it -- and no amount of processing recovers a signal that
    was never recorded.

    Four independent checks, each with its threshold HERE so that every
    consumer (the gain search, the collection tools, a bridge gating a
    recording) applies the same one rather than re-deriving it:

    ``clip_fraction``
        Fraction of samples with |I| or |Q| at the converter's last code on
        either rail (+127 or -128 for 8 bits; the threshold is
        (full_scale - 1) / full_scale in normalized units). Clipping destroys information the way
        under-driving does not -- it cannot be undone downstream -- so it
        is the worst verdict. ``max_clip_fraction`` 1e-3 (0.1%) tolerates
        the occasional spike a strong adjacent signal produces.

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
        Power inside ``channel_bw_hz`` of DC, relative to the outer 20% of the
        span. A real carrier is a hump standing above its surroundings; a
        flat spectrum means nothing is there. Skipped when channel_bw_hz is
        None or the capture is too short to average.

    ``dc_ratio``
        |mean| of the complex samples over the rms of what is left after the
        mean is removed: DC power against everything-else power. A
        direct-conversion receiver always has SOME DC (the HackRF's spike;
        see docs/DC_SPIKE.md, typically a few % of rms), so a small ratio is
        normal. A ratio above ``max_dc_ratio`` (1.0: more DC than signal) is
        a stuck or saturated front end, a muted stream, or a capture of
        nothing but the LO -- the "stuck_dc" verdict.

    Args:
        iq:          complex samples normalized to +/-1 (as sources return).
        sample_rate: Hz, used to place the channel band.
        channel_bw_hz:  half-width in Hz of the channel of interest. None skips
                     the spectral check.
        adc_bits:    converter width; 8 for HackRF's ci8. Counts scale is
                 2**(adc_bits-1) = 128, matching the loader's /128
                 normalization, so a full-scale sample reads as +/-128.
        min_counts:  ADC counts below which the capture is called empty.
        min_excess_db: in-band excess below which no carrier is called.
        nfft:        FFT size for the averaged spectrum.

    Returns a dict with:
      ``verdict``  one of HEALTH_VERDICTS -- "clipping", "under_driven",
                   "stuck_dc", "no_carrier" or "ok" -- the WORST failing
                   check, in that order. Key off this, not off the numbers.
      ``ok``       ``verdict == "ok"``.
      ``clip_fraction``, ``adc_counts``, ``peak_dbfs``, ``dc_offset`` (|mean|,
                   normalized units), ``dc_ratio``, ``channel_excess_db``
                   (None when channel_bw_hz is None or the capture is shorter
                   than 4 * nfft -- "not computed", never "no carrier").
      ``reasons``  one sentence per failing check, empty when ok.

    This is a screening tool, not a detector. A signal far weaker than the
    noise floor in the channel -- spread spectrum below the noise, say --
    will be reported as absent, which is the right answer for "can I demod
    this directly" and the wrong one for "is there anything here at all."
    """
    iq = np.asarray(iq)
    full_scale = float(2 ** (adc_bits - 1))
    reasons: list[str] = []
    failed: list[str] = []                      # verdict names, appended in check order

    if iq.size == 0:
        return {"verdict": "under_driven", "ok": False, "clip_fraction": 0.0,
                "adc_counts": 0.0, "peak_dbfs": float("-inf"),
                "dc_offset": 0.0, "dc_ratio": 0.0, "channel_excess_db": None,
                "reasons": ["capture is empty"]}

    re_abs = np.abs(iq.real)
    im_abs = np.abs(iq.imag)
    # The converter's last codes, in normalized units. Two's complement is
    # asymmetric: for 8 bits the rails are -128 (-> -1.0 exactly) and +127
    # (-> 0.9921875), so each rail gets its own threshold: |x| >= 127/128
    # would also count -127, one code short of the negative rail. A small
    # epsilon absorbs float32 rounding of the /128. (A first version used a
    # single full_scale - 0.5 level, which only ever saw the negative rail --
    # a capture pinned at +127 on every sample read as stuck DC, not clipping.)
    pos_rail = (full_scale - 1.0) / full_scale - 1e-6      # +127/128
    neg_rail = -1.0 + 1e-6                                  # -128/128
    re_, im_ = iq.real, iq.imag
    clipped = np.count_nonzero((re_ >= pos_rail) | (re_ <= neg_rail)
                               | (im_ >= pos_rail) | (im_ <= neg_rail))
    clip_fraction = clipped / iq.size
    if clip_fraction > max_clip_fraction:
        failed.append("clipping")
        reasons.append(
            f"{100 * clip_fraction:.2f}% of samples sit at the converter's "
            f"full-scale code: the front end is clipping, lower the gain "
            f"(this cannot be repaired downstream)")

    peak = float(max(np.max(re_abs), np.max(im_abs)))
    adc_counts = peak * full_scale
    if adc_counts < min_counts:
        failed.append("under_driven")
        reasons.append(
            f"capture peaks at ~{adc_counts:.1f} of {full_scale:.0f} ADC "
            f"counts, which is the noise floor: check the antenna, raise "
            f"gain, and confirm the tuned frequency")

    mean = complex(np.mean(iq))
    dc_offset = abs(mean)
    ac_rms = float(np.sqrt(np.mean(np.abs(iq - mean) ** 2)))
    dc_ratio = dc_offset / ac_rms if ac_rms > 0 else (float("inf") if dc_offset > 0 else 0.0)
    if dc_ratio > max_dc_ratio:
        failed.append("stuck_dc")
        reasons.append(
            f"DC is {dc_ratio:.1f}x the rms of everything else: a stuck or "
            f"saturated front end, a muted stream, or a capture of the LO alone")

    excess = None
    if channel_bw_hz is not None and sample_rate > 0:
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
            in_band = np.abs(freqs) <= channel_bw_hz
            ref_band = ((np.abs(freqs) >= 1.5 * channel_bw_hz)
                        & (np.abs(freqs) <= 0.35 * sample_rate))
            if in_band.any() and ref_band.any():
                excess = float(
                    10.0 * np.log10(psd_lin[in_band].mean() + 1e-30)
                    - 10.0 * np.log10(np.median(psd_lin[ref_band]) + 1e-30))
                if excess < min_excess_db:
                    failed.append("no_carrier")
                    reasons.append(
                        f"channel is only {excess:+.1f} dB above the "
                        f"surrounding noise floor, so no carrier is present: "
                        f"demodulating this yields noise, not signal")

    verdict = next((v for v in HEALTH_VERDICTS if v in failed), "ok")
    return {"verdict": verdict, "ok": verdict == "ok",
            "clip_fraction": clip_fraction,
            "adc_counts": adc_counts,
            "peak_dbfs": 20.0 * np.log10(peak + 1e-20),
            "dc_offset": dc_offset, "dc_ratio": dc_ratio,
            "channel_excess_db": excess,
            "reasons": reasons}


def fm_pilot_excess_db(
    iq: Any, sample_rate: float, pilot_hz: float = 19_000.0, nfft: int = 16384
) -> float | None:
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
    # numpy's stubs lose the no-argument .max() -> scalar overload after a
    # boolean-mask index; the call is fine at runtime
    return float(10.0 * np.log10(acc[pilot].max() + 1e-30)  # type: ignore[call-overload]
                 - 10.0 * np.log10(np.median(acc[hood]) + 1e-30))


def occupied_bandwidth(
    iq: Any, sample_rate: float, fraction: float = 0.99, nfft: int = 1024
) -> float:
    """Bandwidth containing ``fraction`` of the total power (e.g. 99%). OUR code.

    The standard occupied-bandwidth definition: from the averaged PSD, the
    frequency interval between the point where the cumulative power (low to
    high frequency) reaches (1 - fraction)/2 of the total and the point where
    it reaches 1 - (1 - fraction)/2. It is computed WHEREVER the signal sits in
    the span -- a tone at +100 kHz has the bandwidth of a tone, not of a
    200 kHz band around DC. (An earlier version integrated outward from 0 Hz,
    which assumed the signal was centred and reported 203 kHz for that tone,
    and 1.47 MHz for a 200 kHz FM station tuned off centre.)

    Returns the bandwidth in Hz, quantized to the PSD's bin spacing
    (sample_rate / nfft). Raises ValueError for fraction outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    if len(np.asarray(iq)) < nfft:
        raise ValueError(f"occupied_bandwidth needs at least nfft={nfft} samples, got "
                         f"{len(np.asarray(iq))}; pass a smaller nfft")
    freqs, psd_db = psd(iq, sample_rate, nfft=nfft, window="hann")
    p = 10.0 ** (psd_db / 10.0)
    total = float(np.sum(p))
    if total <= 0:
        return 0.0
    cum = np.cumsum(p) / total
    tail = (1.0 - fraction) / 2.0
    lo = int(np.searchsorted(cum, tail))                 # first bin past the lower tail
    hi = int(np.searchsorted(cum, 1.0 - tail))           # first bin reaching the upper edge
    lo = min(max(lo, 0), len(freqs) - 1)
    hi = min(max(hi, lo), len(freqs) - 1)
    bin_hz = float(sample_rate) / nfft
    return float(freqs[hi] - freqs[lo]) + bin_hz


def find_bursts(
    iq: Any,
    sample_rate: float | None = None,
    threshold: float | None = None,
    min_gap: int = 0,
    min_len: int = 1,
) -> list[tuple[int, int]]:
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
    spans = list(zip(starts, stops, strict=True))
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


def estimate_cfo(iq: Any, sample_rate: float, nfft: int | None = None) -> float:
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


def estimate_fm_cfo(
    iq: Any,
    sample_rate: float,
    *,
    channel_bw_hz: float = 100_000.0,
    trim_hz: float | None = None,
) -> float | None:
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
    no CFO information (a tempting shortcut -- "just read the CFO off the
    pilot" -- that does not work). The pilot's real role here is a presence gate: confirm a
    station is actually there with ``fm_pilot_excess_db`` before trusting this
    number, because the mean of noise is not a carrier offset.

    Args:
        channel_bw_hz: FM channel half-not-needed; used only to size the click
            trim. Broadcast FM is ~100 kHz half-bandwidth of program content.
        trim_hz: discriminator "clicks" (phase wraps in deep fades / low SNR)
            are impulsive outliers that bias a plain mean. Samples whose
            demodulated frequency exceeds trim_hz in magnitude are dropped
            before averaging. Default 1.5 * channel_bw_hz keeps the full +/-75 kHz
            deviation swing while rejecting the near-+/-Nyquist click spikes.
            Pass None to disable trimming.

    Returns the offset in Hz (positive = the signal sits above the tuned
    center), or None when the capture is too short or the rate too low to
    channelize to the FM channel.

    OUR code.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if sample_rate <= 4.0 * channel_bw_hz or iq.size < 4096:
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
    trim = 1.5 * float(channel_bw_hz) if trim_hz is None else float(trim_hz)
    if trim > 0:
        keep = np.abs(inst_hz) <= trim
        if keep.any():
            inst_hz = inst_hz[keep]
    if inst_hz.size == 0:
        return None
    return float(np.mean(inst_hz))


def correct_fm_cfo(
    iq: Any, sample_rate: float, cfo_hz: float | None = None, **kwargs: Any
) -> tuple[np.ndarray, float]:
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
    return frequency_shift(iq, sample_rate, -float(cfo_hz)), float(cfo_hz)
