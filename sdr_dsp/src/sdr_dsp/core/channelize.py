"""Channelization: pull channels out of a wide capture.

Two functions for two genuinely different jobs:

- `channelize` extracts ONE channel at an arbitrary offset and bandwidth -- the
  surgical case (tune to one FM station, one ISM device). tune -> filter ->
  decimate.

- `channelize_bank` splits the WHOLE band into N equal, evenly-spaced channels
  at once -- the survey case (monitor every channel in a slice). It uses a
  polyphase filterbank: instead of N separate mix-filter-decimate chains, it
  reshapes the input through the polyphase branches of a single prototype filter
  and takes an FFT across them, producing all N channels for roughly the cost of
  one filter plus an FFT. For large N that's dramatically cheaper than the naive
  loop -- which we keep as the correctness oracle in the tests.

The split is honest: uniform-and-fast (bank) vs. arbitrary-and-flexible
(single). They are not the same function scaled; the bank is a different, more
efficient algorithm that only applies to a uniform channel grid.
"""

from __future__ import annotations

import numpy as np

from .mixing import tune_to_baseband
from .filters import design_lowpass, fir_apply
from .resample import decimate


def channelize(iq, sample_rate, offset_hz, channel_bw, decim=None):
    """Extract the single channel at offset_hz with bandwidth channel_bw. OUR code.

    tune -> lowpass -> decimate. Returns (channel_iq, new_sample_rate). decim
    defaults to the largest integer that keeps the channel comfortably inside
    the new Nyquist (new rate >= ~2.5x the channel bandwidth).

    Use this when you want one specific channel at an arbitrary offset/width.
    For splitting the whole band into a uniform grid, use channelize_bank.
    """
    base = tune_to_baseband(iq, offset_hz, sample_rate)
    taps = design_lowpass(channel_bw / 2, sample_rate, num_taps=201)
    filt = fir_apply(base, taps)
    if decim is None:
        decim = max(1, int(sample_rate / (channel_bw * 2.5)))
    out = decimate(filt, decim)
    return out, sample_rate / decim


def channelize_bank(iq, sample_rate, n_channels, decim=None, taps_per_channel=12,
                    return_freqs=True):
    """Split the band into n_channels equal channels via a polyphase filterbank.

    Produces N evenly-spaced channels each sample_rate/N wide, in one efficient
    pass. Returns (channels, new_sample_rate[, center_freqs]):
        channels:     complex64 array of shape (n_channels, n_out_samples).
                      Row k is the channel centered at center_freqs[k].
        new_rate:     the per-channel output sample rate.
        center_freqs: (if return_freqs) the center frequency of each channel, in
                      Hz relative to the capture center, ordered low->high.

    decim sets the sampling scheme (D = decim, must divide N):
        decim = n_channels (default)  -> critically sampled: each channel output
            at rate/N. Standard and most efficient. The channel's transition
            band (where the prototype rolls off, just outside +/- rate/2N)
            aliases back into the channel: a signal straddling a boundary is
            folded, not just attenuated.
        decim = n_channels // 2       -> oversampled by 2: output at rate/(N/2)
            per channel, so the transition band is carried unaliased and a
            channel can be filtered or re-centered downstream without the
            fold. Costs 2x the output samples. This is a true polyphase
            oversampled bank (the commutator advances D samples per frame and
            channel k is rotated by exp(-j*2*pi*k*m*D/N) per frame), not an
            interpolation of the critically-sampled output -- an earlier
            version did the latter, which added no alias protection.
    Any divisor of N works; N and N//2 are the usual choices.

    The prototype is a lowpass with cutoff at the channel half-width; taps_per_channel
    sets its length (N * taps_per_channel taps total) -- more = sharper channel
    edges, more compute.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    N = int(n_channels)
    if N < 1:
        raise ValueError("n_channels must be >= 1")
    if decim is None:
        decim = N
    decim = int(decim)
    if decim < 1 or N % decim != 0:
        # keep the math simple and correct: require decim to divide N so the
        # commutator/phase bookkeeping stays exact. (N and N//2 both satisfy this.)
        raise ValueError("decim must be a positive divisor of n_channels "
                         f"(got decim={decim}, n_channels={N})")

    # prototype lowpass: cutoff at half a channel width, length = N * taps_per_channel
    ntaps = N * int(taps_per_channel)
    proto = design_lowpass(sample_rate / (2 * N), sample_rate, num_taps=ntaps)
    proto = np.asarray(proto, dtype=np.float64)

    L = int(taps_per_channel)
    D = decim
    # Frame m is referenced to input sample T_m = m*D + N - 1 (for D = N that
    # is the last sample of input block m, the convention the critically
    # sampled bank has always used). Frames run while T_m is a real sample.
    n_frames = (len(iq) - N) // D + 1 if len(iq) >= N else 0
    if n_frames <= 0:
        empty = np.zeros((N, 0), dtype=np.complex64)
        cf = _center_freqs(sample_rate, N)
        return (empty, sample_rate / D, cf) if return_freqs else \
            (empty, sample_rate / D)

    # --- polyphase decomposition -------------------------------------------
    # Branch n holds prototype taps h[n], h[n+N], h[n+2N], ... (L of them) and
    # is fed x[T_m - n - l*N], l = 0..L-1. An inverse FFT across the branches
    # then yields the N channels: channel k of frame m is
    #     exp(-j*2*pi*k*T_m/N) * sum_n exp(+j*2*pi*k*n/N) * y_n[m],
    # which is exactly "mix x down by channel k's center, lowpass with the
    # prototype, take every D-th sample". The constant part of the phase
    # factor, exp(-j*2*pi*k*(N-1)/N), is dropped (a fixed per-channel phase);
    # the frame-dependent part exp(-j*2*pi*k*m*D/N) is 1 when D = N and is
    # applied otherwise.
    pb = proto.reshape(L, N)                      # row = tap index l, col = branch n
    xpad = np.concatenate([np.zeros(L * N, dtype=np.complex64), iq])
    ref = np.arange(n_frames) * D + (N - 1) + L * N          # T_m, shifted by the pad
    idx = ref[:, None] - np.arange(N)[None, :]               # (n_frames, N): x index for tap l=0
    filtered = np.zeros((n_frames, N), dtype=np.complex64)
    for tap in range(L):
        filtered += xpad[idx - tap * N] * pb[tap, :][None, :]

    chan = np.fft.ifft(filtered, n=N, axis=1) * N            # (n_frames, N), channel k in column k
    if D != N:
        k = np.arange(N)[None, :]
        m = np.arange(n_frames)[:, None]
        chan = chan * np.exp(-2j * np.pi * k * m * D / N)
    channels = np.ascontiguousarray(chan.T).astype(np.complex64)   # (N, n_frames)
    new_rate = sample_rate / D

    # reorder channels so they run low frequency -> high (fftshift the channel axis)
    channels = np.fft.fftshift(channels, axes=0)
    if return_freqs:
        return channels, new_rate, _center_freqs(sample_rate, N)
    return channels, new_rate


def _center_freqs(sample_rate, N):
    """Channel center frequencies (Hz from capture center), low -> high."""
    # channels tile [-rate/2, rate/2) in N steps, fftshifted to ascending order
    k = np.arange(N)
    freqs = (k - N // 2) * (sample_rate / N)
    return freqs.astype(np.float64)
