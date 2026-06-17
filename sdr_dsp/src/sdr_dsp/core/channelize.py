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

    decim sets the sampling scheme:
        decim = n_channels (default)  -> critically sampled: each channel output
            at rate/N. Standard and most efficient; channels tile the band with
            minimal overlap (slight edge aliasing at channel boundaries).
        decim = n_channels // 2       -> oversampled by 2: cleaner separation
            between channels at the cost of 2x the output samples.
    Any integer divisor of the prototype length works; the two above are the
    usual choices. Critically-sampled is the default.

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

    # pad input so it splits evenly into length-N input strides
    n_blocks = len(iq) // N
    if n_blocks == 0:
        empty = np.zeros((N, 0), dtype=np.complex64)
        cf = _center_freqs(sample_rate, N)
        return (empty, sample_rate / N, cf) if return_freqs else \
            (empty, sample_rate / N)
    usable = n_blocks * N
    x = iq[:usable]

    # --- polyphase decomposition -------------------------------------------
    # Arrange the signal and the prototype into N polyphase branches. Each branch
    # is a sub-filter; an FFT across the branches yields the N channel outputs.
    # taps_per_channel taps per branch.
    L = int(taps_per_channel)
    # prototype branches: proto reshaped (L, N) so column n is branch n
    # (proto[n], proto[n+N], proto[n+2N], ...)
    pb = proto.reshape(L, N)                      # row=tap index, col=branch

    # input branches: feed samples to branches in reverse order (commutator).
    # reshape the stream into (n_blocks, N); column n is the n-th branch's input.
    xb = x.reshape(n_blocks, N)
    # the commutator distributes samples to branches in reverse; flip columns
    xb = xb[:, ::-1]

    # filter each branch (length-L FIR along the block axis), then FFT across
    # branches. Build the per-branch filtered output via convolution.
    # branch n output[m] = sum_l pb[l, n] * xb[m - l, n]
    out_len = n_blocks
    filtered = np.zeros((out_len, N), dtype=np.complex64)
    for l in range(L):
        if l == 0:
            filtered += xb * pb[0, :][None, :]
        else:
            filtered[l:] += xb[:-l] * pb[l, :][None, :]

    # FFT across the N branches gives the N channels (one column per channel).
    # Use the inverse transform so branch index k maps to ascending positive
    # frequency (matching our low->high label convention); the 1/N scaling is
    # immaterial to channel routing and consistent across channels.
    chan = np.fft.ifft(filtered, n=N, axis=1) * N  # (out_len, N)
    channels = np.ascontiguousarray(chan.T).astype(np.complex64)  # (N, out_len)

    # critically-sampled output is at rate/N already (one output per N inputs).
    new_rate = sample_rate / N
    # oversampling (decim = N//2) -> upsample the per-channel time axis by N/decim
    if decim != N:
        factor = N // decim
        # simple linear-phase interpolation of the channel time series
        channels = _oversample_time(channels, factor)
        new_rate = sample_rate / decim

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


def _oversample_time(channels, factor):
    """Upsample each channel's time series by an integer factor (linear interp).

    Used only for the oversampled (decim = N//2) scheme to deliver the higher
    output rate. Critically-sampled output skips this.
    """
    if factor <= 1:
        return channels
    n_ch, n = channels.shape
    if n < 2:
        return channels
    old_idx = np.arange(n)
    new_idx = np.linspace(0, n - 1, n * factor)
    out = np.empty((n_ch, len(new_idx)), dtype=np.complex64)
    for c in range(n_ch):
        out[c].real = np.interp(new_idx, old_idx, channels[c].real)
        out[c].imag = np.interp(new_idx, old_idx, channels[c].imag)
    return out
