# sdr_dsp — Definitive API Handbook

> **Auto-generated** from the source by `docs/dev_handbook/generate_api_handbook.py`. Every signature, parameter, default, and docstring is extracted directly from the code, so this handbook cannot drift from the library. Re-run the generator after any API change.

> For architecture, philosophy, extension guides, hardware notes, and diagrams, see `sdr_dsp_REFERENCE.md`. This document is the exhaustive call-level reference.

> **Reading the tables:** where the *Type* column is blank, the library documents argument types in the docstring rather than via annotations — read the function's docstring for the expected dtype and shape (typically `numpy.complex64` IQ arrays, `float` rates in Hz, and `bytes`/bit-arrays for the protocol layer). A *Default* of `*required*` means the argument is positional and mandatory.

> **Note on ARQ:** `window_size=1` is stop-and-wait (the default); `window_size=N` is sliding-window Selective Repeat. The opt-in `ARQ(cumulative_ack=True)` acknowledges the contiguous high-water mark instead of each frame. Both ACK modes are validated correct under heavy random and burst loss — see `sdr_dsp_REFERENCE.md` §11.1.

## Modules

- [Top-level package](#top-level-package)
- [core — pure DSP](#core--pure-dsp)
- [core.demod — demodulators](#core.demod--demodulators)
- [core.modulate — modulators](#core.modulate--modulators)
- [sources — receive seam](#sources--receive-seam)
- [sinks — output & transmit seam](#sinks--output-&-transmit-seam)
- [io — file formats](#io--file-formats)
- [stream — orchestration](#stream--orchestration)
- [link — ARQ protocol](#link--arq-protocol)

---

## Top-level package

Lazily-loaded submodules and the flattened DSP API.

Import: `import sdr_dsp`

**Submodules:** `core`, `io`, `link`, `sinks`, `stream`

### Classes

### class `AGC`

Streaming AGC: the agc() loop with memory across blocks.

A per-block AGC that reset each block would lurch at every boundary. This
holds the gain and level between blocks so the loop is continuous, and calls
the same agc() core -- it adds memory, not new DSP. Use it as a Pipeline
stage; the last gain trace is kept on `.last_gain` so a .tap() can watch it.

    stage = AGC(mode="rms", target=1.0, attack=0.01, decay=0.001)
    pipe.add(stage, "agc").tap(lambda b: meter.update(stage.last_gain))

Like the function, it never hides what it did: every processed block has its
gain trace available, and the same caveats apply (do calibration before AGC;
set max_gain to bound silent-gap runaway).


**Constructor:** `AGC(self, mode: 'str' = 'rms', target: 'float' = 1.0, attack: 'float' = 0.01, decay: 'float' = 0.001, max_gain: 'float | None' = None) -> None`

| Parameter | Type | Default |
|---|---|---|
| `mode` | `str` | `'rms'` |
| `target` | `float` | `1.0` |
| `attack` | `float` | `0.01` |
| `decay` | `float` | `0.001` |
| `max_gain` | `float | None` | `None` |


**Methods:**

#### `reset(self)`

Forget the carried state (start the next block fresh).



**Attributes:**

- `mode`: `str`
- `target`: `float`
- `attack`: `float`
- `decay`: `float`
- `max_gain`: `float | None`



### class `ArraySource`

The simplest source: wrap an in-memory complex64 array.

Useful for tests, synthetic signals, and feeding already-loaded data into
the same pipeline code a file or device would drive.


**Constructor:** `ArraySource(self, iq: 'np.ndarray', sample_rate: 'float', center_freq: 'float' = 0.0, block_size: 'int' = 65536)`

| Parameter | Type | Default |
|---|---|---|
| `iq` | `np.ndarray` | *required* |
| `sample_rate` | `float` | *required* |
| `center_freq` | `float` | `0.0` |
| `block_size` | `int` | `65536` |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

_No docstring._


#### `read(self, n_samples: 'int') -> 'np.ndarray'`

Return the whole array (or the first n_samples).

| Parameter | Type | Default |
|---|---|---|
| `n_samples` | `int` | *required* |




### class `Calibration`

A dBFS->dBm offset plus the conditions it was measured under.

The offset alone is a footgun -- it's only valid for the receive-chain setup
it was measured at. So a Calibration carries that context: `frequency_hz`
(first-class, for the drift warning) and a free-form `conditions` dict for
everything else (gains, antenna, SDR model, temperature -- whatever you want
to record). `notes` and `measured_at` are for your own bookkeeping.

Apply it with `.power_dbm(iq)`. If you apply it far from `frequency_hz`, it
warns (the offset drifts with frequency); pass `warn=False` to silence, or
widen/disable the threshold with `drift_warn_hz`.


**Constructor:** `Calibration(self, offset_db: 'float', frequency_hz: 'float | None' = None, conditions: 'dict' = <factory>, notes: 'str' = '', measured_at: 'str' = '', drift_warn_hz: 'float' = 5000000.0) -> None`

| Parameter | Type | Default |
|---|---|---|
| `offset_db` | `float` | *required* |
| `frequency_hz` | `float | None` | `None` |
| `conditions` | `dict` | `<factory>` |
| `notes` | `str` | `''` |
| `measured_at` | `str` | `''` |
| `drift_warn_hz` | `float` | `5000000.0` |


**Methods:**

#### `load(path)`

Load a calibration saved by `save`.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |


#### `power_dbm(self, iq, at_frequency_hz=None, warn=True)`

Absolute power of `iq` in dBm, using this calibration's offset.

at_frequency_hz: the frequency `iq` was captured at, if known. When both
    this and the calibration's frequency_hz are set, applying the
    calibration more than drift_warn_hz away raises a warning (the offset
    is only valid near where it was measured).
warn: set False to silence the drift warning for this call.

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `at_frequency_hz` |  | `None` |
| `warn` |  | `True` |


#### `save(self, path)`

Write the calibration to a human-readable JSON file.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



**Attributes:**

- `offset_db`: `float`
- `frequency_hz`: `float | None`
- `conditions`: `dict`
- `notes`: `str`
- `measured_at`: `str`
- `drift_warn_hz`: `float`



### class `FileSource`

An IQSource backed by a SigMF recording on disk.

sample_rate and center_freq come from the sidecar (read cheaply, without
loading the samples). blocks() streams the file in block_size chunks read
incrementally from disk; .iq loads the whole recording lazily if you ask.

Args:
    path:           the .iq / .sigmf-data / .sigmf-meta path.
    block_size:     samples per block from blocks().
    count:          limit reading to this many samples (None = whole file).
    offset_samples: skip this many samples from the start.


**Constructor:** `FileSource(self, path, block_size=65536, count=None, offset_samples=0)`

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `block_size` |  | `65536` |
| `count` |  | `None` |
| `offset_samples` |  | `0` |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

Yield the recording in block_size chunks, read from disk on demand.

Each block is read with its own seek+read, so memory use stays at one
block regardless of file size. If .iq has already been loaded (small-file
path), slice that instead of re-reading.


#### `read(self, n_samples: 'int') -> 'np.ndarray'`

Read up to n_samples from the start (respecting offset). Streams from
disk without loading the whole file.

| Parameter | Type | Default |
|---|---|---|
| `n_samples` | `int` | *required* |




### class `IQSource`

Anything that can provide IQ samples plus the metadata to interpret them.

Attributes:
    sample_rate: samples per second (Hz).
    center_freq: RF center frequency the samples were captured at (Hz).

Implementations provide ``blocks()`` to stream decoded complex64 arrays.
A bounded source may also support ``read(n)``; unbounded/live sources need
only ``blocks()``.


**Constructor:** `IQSource(self, *args, **kwargs)`

| Parameter | Type | Default |
|---|---|---|
| `args` (*args) |  | *required* |
| `kwargs` (**kwargs) |  | *required* |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

Yield complex64 blocks until the source is exhausted or stopped.



**Attributes:**

- `sample_rate`: `float`
- `center_freq`: `float`



### class `LoopDiagnostics`

Per-sample evidence of a recovery loop's behavior.

error:    the loop's error signal each sample (phase error, or timing
          error). Its settling toward ~0 is convergence.
estimate: the running quantity the loop tracks (accumulated phase, or the
          fractional sample offset).
lock:     per-sample boolean trace -- True where the error variance over a
          sliding window is below the lock threshold.
locked:   summary -- True if the loop was locked over the final portion.


**Constructor:** `LoopDiagnostics(self, error: 'np.ndarray', estimate: 'np.ndarray', lock: 'np.ndarray', locked: 'bool' = False) -> None`

| Parameter | Type | Default |
|---|---|---|
| `error` | `np.ndarray` | *required* |
| `estimate` | `np.ndarray` | *required* |
| `lock` | `np.ndarray` | *required* |
| `locked` | `bool` | `False` |


**Methods:**

#### `to_csv(self, path)`

Write the per-sample diagnostics to a CSV (sample, error, estimate,
lock). Useful for plotting convergence outside the library.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



**Attributes:**

- `error`: `np.ndarray`
- `estimate`: `np.ndarray`
- `lock`: `np.ndarray`
- `locked`: `bool`



### class `Pipeline`

A source + an ordered chain of block->block stages (and taps).

Build declaratively and run:

    pipe = (Pipeline(source)
            .add(lambda b: fir_apply(b, taps), "filter")
            .add(lambda b: fm_demod(b, 75000, fs), "demod")
            .tap(lambda b: meter.update(b)))      # live peek, flow unchanged
    audio = pipe.run()                            # or run(sink=write_audio)

Stages transform the block; taps observe it and return nothing (the original
block continues). Order matters and is preserved.


**Constructor:** `Pipeline(self, source)`

| Parameter | Type | Default |
|---|---|---|
| `source` |  | *required* |


**Methods:**

#### `add(self, fn, name=None)`

Append a transforming stage (block -> block).

| Parameter | Type | Default |
|---|---|---|
| `fn` |  | *required* |
| `name` |  | `None` |


#### `describe(self)`

Return the chain as inspectable text (the pipeline is data).


#### `process_block(self, block)`

Thread a single block through every stage. Useful for testing and
for driving the pipeline from an external loop (e.g. a GUI timer).

| Parameter | Type | Default |
|---|---|---|
| `block` |  | *required* |


#### `run(self, sink=None, profile=False, max_blocks=None)`

Pull blocks from the source, process each, deliver to sink.

sink:       callable(result_block) -> None. If None, results are
            collected and returned as a list.
profile:    if True, also return PipelineStats (per-stage timing).
max_blocks: stop after this many blocks (useful for live sources).

Returns the result list (or None if a sink was given); if profile,
returns (results_or_None, PipelineStats).

| Parameter | Type | Default |
|---|---|---|
| `sink` |  | `None` |
| `profile` |  | `False` |
| `max_blocks` |  | `None` |


#### `stream(self, max_blocks=None) -> 'Iterator[np.ndarray]'`

Run as a generator, yielding each processed block lazily.

This is the bridge to the generator-chain style: a Pipeline can be
consumed lazily, so it composes with other generators and stays
memory-friendly for long/continuous streams.

| Parameter | Type | Default |
|---|---|---|
| `max_blocks` |  | `None` |


#### `tap(self, fn, name=None)`

Append an observing stage (block -> ignored). Flow is unchanged.

A tap is how live display attaches: fn receives the current block and
does whatever it likes (update a plot, accumulate a message) without
affecting what the next stage sees.

| Parameter | Type | Default |
|---|---|---|
| `fn` |  | *required* |
| `name` |  | `None` |




### Functions

### `add_cfo(iq, cfo_hz, sample_rate)`

Apply a carrier frequency offset (Hz). OUR code.

Multiplies by a complex exponential at cfo_hz -- the rotation a real link
imposes when the TX and RX oscillators differ. Recoverable on the RX side by
carrier recovery, or measurable with estimate_cfo.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `cfo_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `add_delay(iq, delay_samples)`

Delay the signal by an integer number of samples (zero-pad the front).

Models propagation delay / a late frame start. The output is the same length
as the input (the tail is truncated); negative delay advances. OUR code.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `delay_samples` |  | *required* |



### `add_noise(iq, snr_db, rng=None)`

Add complex AWGN at a specified SNR (dB) relative to the signal. OUR code.

Measures the signal's mean power, computes the noise power for the requested
SNR, and adds complex Gaussian noise at that level. This is the honest way to
set noise -- by the SNR you want, not an arbitrary amplitude.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `snr_db` |  | *required* |
| `rng` |  | `None` |



### `agc(iq, mode='rms', target=1.0, attack=0.01, decay=0.001, max_gain=None, _initial_gain=1.0, _initial_level=None)`

Apply automatic gain control. Returns (adjusted_iq, gain_trace). OUR code.

Drives the signal level toward `target` with a one-pole tracking loop. The
gain rises slowly when the signal is weak (decay rate) and clamps down
quickly when it's strong (attack rate) -- fast attack avoids clipping, slow
decay avoids pumping on noise. Both rates are in (0, 1]; larger = faster.

mode:     "rms" tracks average power (smoother; good for analog/voice),
          "peak" tracks the running peak (twitchier; better anti-clipping).
target:   the level the loop steers the signal toward.
attack:   tracking rate when the measured level is ABOVE target (gain down).
decay:    tracking rate when the measured level is BELOW target (gain up).
max_gain: optional ceiling on the gain (None = no ceiling; see the module
          note about gain runaway during silence).

Returns:
    adjusted_iq:  iq * gain_trace
    gain_trace:   the per-sample gain that was applied (same length as iq).
                  This is the whole point -- it makes the AGC observable and
                  reversible: iq == adjusted_iq / gain_trace.

The _initial_* args let the streaming AGC stage continue a loop across blocks
and aren't normally set by hand.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `mode` |  | `'rms'` |
| `target` |  | `1.0` |
| `attack` |  | `0.01` |
| `decay` |  | `0.001` |
| `max_gain` |  | `None` |
| `_initial_gain` |  | `1.0` |
| `_initial_level` |  | `None` |



### `am_demod(iq, dc_block=True)`

Demodulate amplitude modulation: the envelope (magnitude). OUR code.

Returns the real envelope |iq|. With dc_block, the mean (carrier DC) is
removed so the output swings around zero like audio.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `dc_block` |  | `True` |



### `am_modulate(message, modulation_index=0.5)`

AM-modulate a real message into IQ. Inverse of am_demod. OUR code.

Amplitude modulation rides the message on the carrier envelope:
(1 + k*m) carrier. am_demod recovers it with an envelope detector. Keep
modulation_index <= 1 to avoid over-modulation (envelope going negative,
which the envelope detector can't undo).

message:          real message, roughly [-1, 1].
modulation_index: depth k of modulation (0..1). >1 over-modulates.

Returns complex64 IQ with the message in its magnitude.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `modulation_index` |  | `0.5` |



### `apply_channel(iq, sample_rate=None, snr_db=None, cfo_hz=None, delay_samples=0, scale=1.0, phase=0.0, seed=None)`

Pass a signal through a simulated channel. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |
| `snr_db` |  | `None` |
| `cfo_hz` |  | `None` |
| `delay_samples` |  | `0` |
| `scale` |  | `1.0` |
| `phase` |  | `0.0` |
| `seed` |  | `None` |



### `bpsk_demod(iq, normalize_phase=True)`

Demodulate binary phase-shift keying (coherent-ish). OUR code.

BPSK encodes bits as 0 or pi phase. With the carrier already at baseband and
roughly phase-aligned, the sign of the real part recovers the bits. This is
a SIMPLE demod: it assumes the signal is already carrier-aligned (no Costas
loop / carrier recovery). For captures with a residual carrier offset,
correct it first (see estimate_cfo / frequency_shift) -- the library does
not auto-recover the carrier.

Returns (bits, soft) where bits is uint8 (0/1) and soft is the real-part
decision statistic (useful for confidence / plotting a constellation).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `normalize_phase` |  | `True` |



### `bpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, pad_symbols=0)`

BPSK: bit 0 -> +1, bit 1 -> -1 (phase 0 or pi). Inverse of bpsk_demod.

Carries one bit per symbol in the carrier phase. With pulse_shaping=False the
symbols are held rectangular (sps samples each); with pulse_shaping=True they
are RRC-shaped for a bandlimited spectrum (use a matched RRC filter on
receive). bpsk_demod recovers bits from the real part's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per symbol (1 = one sample per symbol).
pulse_shaping:      RRC-shape the symbols if True.

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `pad_symbols` |  | `0` |



### `build_frame(payload, sync=None, preamble_bits=32)`

Build a complete frame from a payload. Returns a bit array (0/1). OUR code.

Layout: [preamble][sync][length:1 byte][payload][crc16:2 bytes].
The length byte limits a single frame to 255 payload bytes; split larger
messages across frames yourself.

payload:        bytes-like (bytes, bytearray, or a list of ints 0..255).
sync:           the sync word bits (default DEFAULT_SYNC).
preamble_bits:  number of alternating preamble bits.

The CRC covers the length byte and the payload, so a receiver validates both.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `payload` |  | *required* |
| `sync` |  | `None` |
| `preamble_bits` |  | `32` |



### `carrier_recovery(iq, method='costas', order=2, loop_bw=0.01, damping=0.707, diagnostics=False, csv_path=None, lock_threshold=0.05)`

Track and remove residual carrier phase/frequency offset. OUR code.

Returns a phase-corrected copy of iq (constellation de-rotated). With
diagnostics=True, returns (corrected, LoopDiagnostics). With csv_path set,
also writes the diagnostics CSV.

method:
  "costas"            : a Costas loop -- data-independent phase tracking.
                        order=2 for BPSK, order=4 for QPSK (the phase error
                        detector matches the modulation's symmetry).
  "decision_directed" : uses the nearest-symbol decision to form the error;
                        simpler, better at high SNR, needs order too.
loop_bw, damping: second-order loop filter parameters. Smaller loop_bw =
  slower but steadier lock. These are exposed on purpose.
order: 2 (BPSK-like) or 4 (QPSK-like) phase symmetry.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `method` |  | `'costas'` |
| `order` |  | `2` |
| `loop_bw` |  | `0.01` |
| `damping` |  | `0.707` |
| `diagnostics` |  | `False` |
| `csv_path` |  | `None` |
| `lock_threshold` |  | `0.05` |



### `channelize(iq, sample_rate, offset_hz, channel_bw, decim=None)`

Extract the single channel at offset_hz with bandwidth channel_bw. OUR code.

tune -> lowpass -> decimate. Returns (channel_iq, new_sample_rate). decim
defaults to the largest integer that keeps the channel comfortably inside
the new Nyquist (new rate >= ~2.5x the channel bandwidth).

Use this when you want one specific channel at an arbitrary offset/width.
For splitting the whole band into a uniform grid, use channelize_bank.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `offset_hz` |  | *required* |
| `channel_bw` |  | *required* |
| `decim` |  | `None` |



### `channelize_bank(iq, sample_rate, n_channels, decim=None, taps_per_channel=12, return_freqs=True)`

Split the band into n_channels equal channels via a polyphase filterbank.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `n_channels` |  | *required* |
| `decim` |  | `None` |
| `taps_per_channel` |  | `12` |
| `return_freqs` |  | `True` |



### `compute_cal_offset(iq, known_dbm, frequency_hz=None, conditions=None, notes='', drift_warn_hz=5000000.0)`

Derive a calibration from a measurement of a KNOWN-power reference.

Feed in `iq` captured from a calibrated source whose true power is
`known_dbm` (e.g. a signal generator set to -30 dBm), and this returns a
ready-to-use Calibration stamped with the conditions you pass:

    offset = known_dbm - power_dbfs(reference_iq)

Record the conditions honestly -- the offset is only valid at the gain and
frequency this reference was captured at. Example:

    cal = compute_cal_offset(ref_iq, known_dbm=-30.0,
                             frequency_hz=433.92e6,
                             conditions={"lna": 16, "vga": 20, "amp": False})
    cal.save("hackrf_433.cal.json")
    ...
    cal = Calibration.load("hackrf_433.cal.json")
    dbm = cal.power_dbm(capture, at_frequency_hz=433.92e6)

Returns a Calibration. The measurement should be on a steady reference tone;
a noisy or fluctuating source gives a noisy offset.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `known_dbm` |  | *required* |
| `frequency_hz` |  | `None` |
| `conditions` |  | `None` |
| `notes` |  | `''` |
| `drift_warn_hz` |  | `5000000.0` |



### `convolve(a, b, mode='full')`

Convolution of two signals. OUR code (thin, for a uniform API).

Unlike correlate, convolution does NOT conjugate -- it's the filtering
operation. Provided alongside correlate so the distinction is explicit and
both are first-class.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `a` |  | *required* |
| `b` |  | *required* |
| `mode` |  | `'full'` |



### `correlate(a, b, mode='full')`

Cross-correlation of two signals, conjugation handled correctly. OUR code.

np.correlate already conjugates its second argument for complex input -- a
well-known footgun (conjugating it yourself double-conjugates and breaks the
result). This wrapper exists so that subtlety lives in ONE place. Returns
the complex cross-correlation; take np.abs for a magnitude.

mode: "full", "same", or "valid" (as numpy).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `a` |  | *required* |
| `b` |  | *required* |
| `mode` |  | `'full'` |



### `crc16(data)`

CRC-16/CCITT-FALSE over a bytes-like input. OUR code.

A standard 16-bit CRC (poly 0x1021, init 0xFFFF). Used to detect whether a
received payload is intact. Not cryptographic -- just error detection.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `data` |  | *required* |



### `cw_decode(bits, samples_per_symbol)`

Decode Morse (CW) from a sliced on/off stream. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `dbpsk_demod(symbols)`

Demodulate differential BPSK. OUR code.

Differential PSK encodes bits in phase CHANGES between consecutive symbols,
not absolute phase. That's the whole point: it needs NO carrier recovery,
because a constant phase offset cancels when you compare adjacent symbols.
This makes it robust and a good fit for block processing.

Takes symbol-spaced samples (one per symbol -- use symbol_sync first if you
have oversampled data). A bit is 0 if the phase barely changed, 1 if it
flipped by ~pi. Returns (bits, soft) where soft is the real part of the
differential product (sign gives the bit, magnitude gives confidence).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `decimate(iq, factor, half_len=10)`

Lowpass then keep every ``factor``-th sample. OUR code (via resample).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `factor` |  | *required* |
| `half_len` |  | `10` |



### `deemphasis(audio, sample_rate, tau_us=75.0)`

Single-pole de-emphasis filter for broadcast FM audio. OUR code.

Broadcast FM pre-emphasizes high frequencies before transmission; the
receiver must de-emphasize them back. A one-pole IIR does it:
    y[n] = a*x[n] + (1-a)*y[n-1],   a = dt / (tau + dt)
tau_us: time constant (75 us in the Americas/Korea, 50 us elsewhere).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `audio` |  | *required* |
| `sample_rate` |  | *required* |
| `tau_us` |  | `75.0` |



### `design_bandpass(low_hz, high_hz, sample_rate, num_taps=101, window='hamming')`

Design a bandpass FIR. Returns tap coefficients.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `low_hz` |  | *required* |
| `high_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `design_highpass(cutoff_hz, sample_rate, num_taps=101, window='hamming')`

Design a highpass FIR. Returns tap coefficients.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `cutoff_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `design_lowpass(cutoff_hz, sample_rate, num_taps=101, window='hamming')`

Design a lowpass FIR. Returns tap coefficients (numpy array).

cutoff_hz:   passband edge in Hz.
sample_rate: Hz.
num_taps:    filter length (odd recommended for a linear-phase Type-I FIR).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `cutoff_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `detect_peak(signal, template, threshold=None)`

Run a matched filter and return the best-match index (and its value).

threshold: if given, returns (index, value) only when the peak exceeds it,
else (None, value). Without a threshold, always returns the argmax.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `signal` |  | *required* |
| `template` |  | *required* |
| `threshold` |  | `None` |



### `dqpsk_demod(symbols)`

Demodulate differential QPSK. OUR code.

The QPSK analogue of DBPSK: 2 bits per symbol encoded in the phase CHANGE
(one of four ~90-degree steps), so it also needs no carrier recovery. Takes
symbol-spaced samples; returns (bits, phase_diffs) where bits is a uint8
array (2 per symbol, MSB first) and phase_diffs are the raw differential
angles for inspection.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `dsb_sc_demod(iq, sample_rate, bfo_hz=0.0)`

Demodulate double-sideband suppressed-carrier (DSB-SC). OUR code.

DSB-SC is AM with the carrier removed -- both sidebands, no carrier spike.
With the carrier suppressed there's no envelope to follow, so recovery needs
a coherent reference. For a complex baseband capture centered on the
(suppressed) carrier, the real part IS the message (the two sidebands beat
back together). A bfo_hz shift fine-tunes if the center is slightly off.

This is the conceptual midpoint between AM (carrier present, envelope) and
SSB (one sideband): DSB-SC keeps both sidebands but drops the carrier.
Returns the real demodulated message.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `bfo_hz` |  | `0.0` |



### `dsss_despread(iq, code, samples_per_chip=1)`

Despread a DSSS signal using a KNOWN spreading code. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `code` |  | *required* |
| `samples_per_chip` |  | `1` |



### `edges(bits)`

Indices where a 0/1 stream transitions, and the run lengths. OUR code.

Returns (transition_indices, run_lengths, run_values): the sample index of
each transition, how many samples each run lasted, and whether that run was
0 or 1. The building block for recovering symbol timing from a sliced
on/off stream.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |



### `estimate_cfo(iq, sample_rate, nfft=None)`

Estimate a signal's carrier frequency offset from band center. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `None` |



### `estimate_symbol_rate(bits, sample_rate, min_run=2)`

Estimate samples-per-symbol from the run lengths in a sliced stream.

The shortest on/off run is (usually) one symbol period. OUR code. Returns
(samples_per_symbol, symbol_rate_hz).

Robustness: a single glitch sample (from noise or a demod transient at a
symbol boundary) creates a spurious 1-sample run that would fool a naive
"minimum run" estimate. So we DISCARD runs shorter than min_run, then take a
low percentile of what remains as the symbol period -- stable against a few
outliers while still finding the shortest real symbol. Raise min_run if your
capture is very noisy; lower it (to 1) only for pristine synthetic data.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `sample_rate` |  | *required* |
| `min_run` |  | `2` |



### `fhss_detect_hops(iq, sample_rate, nfft=256, overlap=0.5, center_freq=0.0)`

Detect frequency hops: the dominant frequency per time slice. OUR code.

For an FHSS signal, computes a spectrogram and reports, for each time slice,
where the energy is -- i.e. which channel the hopper is in at that moment.
This TRACKS hops you can see; it does NOT decode the data or know the hop
sequence (out of scope). Pair it with core.spectrogram to SEE the hops.

Returns (times, hop_freqs) where hop_freqs[i] is the peak frequency (Hz,
offset by center_freq) during time slice times[i].


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `256` |
| `overlap` |  | `0.5` |
| `center_freq` |  | `0.0` |



### `find_bursts(iq, sample_rate=None, threshold=None, min_gap=0, min_len=1)`

Find where signal energy is present: burst start/stop indices. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |
| `threshold` |  | `None` |
| `min_gap` |  | `0` |
| `min_len` |  | `1` |



### `find_frames(bits, sync=None, max_sync_errors=2)`

Find and validate frames in a recovered bit stream. OUR code.

Searches for the sync word (allowing up to max_sync_errors bit mismatches,
since recovered bits may have errors), then reads the length byte, payload,
and CRC after each match. Returns a list of dicts, one per frame found:

    {"payload": bytes, "crc_ok": bool, "bit_offset": int}

crc_ok tells you whether the payload survived intact -- the basis for an ACK.
Frames with a bad length read or running off the end of the buffer are
skipped. Overlapping/false sync matches inside a validated frame are stepped
past so one packet isn't reported twice.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `sync` |  | `None` |
| `max_sync_errors` |  | `2` |



### `fir_apply(iq, taps)`

Apply an FIR filter to a signal by direct convolution. OUR code.

Equivalent to ``scipy.signal.lfilter(taps, [1.0], iq)`` but implemented
here as a full convolution (then truncated to the input length) so the
filtering operation is the library's own. Works on real or complex input;
complex IQ is filtered as a whole (numpy.convolve handles complex).

Returns an array the same length as ``iq`` (the causal 'full' convolution
truncated to the first len(iq) samples -- matching lfilter's output).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `taps` |  | *required* |



### `fir_apply_centered(iq, taps)`

Apply an FIR with the group delay removed (zero-phase alignment).

A linear-phase FIR of length L delays the signal by (L-1)/2 samples. For
analysis where you want the output time-aligned with the input, this
returns the 'same'-mode convolution (centered), trimming the delay.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `taps` |  | *required* |



### `fm_demod(iq, deviation_hz=None, sample_rate=None)`

Demodulate frequency modulation via the phase discriminator. OUR code.

FM carries the message in instantaneous frequency, so demod IS the
instantaneous frequency (see instantaneous_frequency). Returns a real array.

If deviation_hz and sample_rate are given, the output is scaled by the peak
deviation to give roughly normalized audio; otherwise it returns raw
radians/sample.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `deviation_hz` |  | `None` |
| `sample_rate` |  | `None` |



### `fm_modulate(message, deviation_hz, sample_rate)`

FM-modulate a real message into IQ. Inverse of fm_demod. OUR code.

Frequency modulation encodes the message in the carrier's instantaneous
frequency: the IQ phase is the running integral of the message, scaled by
the deviation. fm_demod recovers it by differentiating the phase.

message:      real-valued message, expected roughly in [-1, 1].
deviation_hz: peak frequency deviation (must match the demod's deviation_hz).
sample_rate:  sample rate in Hz.

Returns unit-magnitude complex64 IQ (FM is constant-envelope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `frequency_shift(iq, shift_hz, sample_rate)`

Shift a complex signal up (positive) or down (negative) in frequency.

Multiplies by exp(j*2*pi*shift*t). To bring a signal at offset f to
baseband (0 Hz), pass shift_hz = -f.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `shift_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `from_db(db, *, power=True)`

Inverse of to_db: dB back to a linear value.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `db` |  | *required* |
| `power` |  | `True` |



### `fsk_demod(iq, sample_rate, threshold_hz=0.0, smooth_samples=0)`

Demodulate 2-level frequency-shift keying. OUR code.

FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
instantaneous frequency, then a threshold: above threshold_hz -> 1, below
-> 0. With the default threshold 0, it splits on the sign of the frequency
deviation (correct when the two tones straddle the center frequency, which
is the common case after tuning to baseband).

threshold_hz: the mark/space decision frequency. Two real radios never
    share an oscillator, so a carrier frequency offset (crystal ppm --
    easily +/-10-20 kHz at 433 MHz between two SDRs) shifts BOTH tones and
    biases the fixed 0 Hz split. Pass "auto" to threshold at the
    amplitude^2-weighted mean of the instantaneous frequency instead: with
    roughly balanced mark/space time (any frame with an alternating
    preamble qualifies) that mean IS the offset, so the split self-centers.
    The weighting means silence around a burst contributes ~nothing.
    NOTE: do not use estimate_cfo for this -- it finds the strongest
    spectral tone, which for FSK is +/-deviation, not the offset.
smooth_samples: if > 1, moving-average the instantaneous frequency over
    this many samples before slicing (a cheap matched-filter stand-in;
    ~samples_per_symbol/2 is a good value). The raw per-sample frequency
    is noisy, and this is the difference between decoding and not at
    moderate SNR. Off by default -- the per-sample output stays exact.

Returns a uint8 per-sample bit stream (length len(iq)-1); feed to the
timing helpers (sample_symbols for hardware captures with unknown delay,
or estimate_symbol_rate / slice_to_symbols) to get symbols. Covers
GFSK/MSK well enough for typical ISM-band sensors and pagers.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `threshold_hz` |  | `0.0` |
| `smooth_samples` |  | `0` |



### `fsk_demod_nlevel(iq, sample_rate, n_levels=4, thresholds=None)`

Demodulate N-level FSK (4-FSK, etc.) and CPFSK. OUR code.

Generalizes 2-FSK: instead of a single 0-threshold on instantaneous
frequency, it slices the frequency into n_levels bands. Used by 4-FSK
(DMR, P25, some pagers). CPFSK recovers the same way -- the continuous
phase is a transmit-side property; the receiver still reads instantaneous
frequency.

thresholds: explicit frequency band centers (Hz). If None, the levels are
spread uniformly across the observed frequency range -- fine for a clean
capture; pass measured centers for real signals. Returns per-sample symbol
indices 0..n_levels-1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `n_levels` |  | `4` |
| `thresholds` |  | `None` |



### `fsk_modulate(bits, samples_per_symbol, deviation_hz, sample_rate, pad_symbols=0)`

Binary FSK: bit selects one of two frequencies. Inverse of fsk_demod.

Bit 1 -> +deviation_hz, bit 0 -> -deviation_hz, encoded as a continuous-phase
frequency shift (CPFSK -- the phase is integrated so there are no jumps,
which keeps the spectrum clean). fsk_demod recovers bits from the
instantaneous frequency's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
deviation_hz:       frequency shift magnitude (match the demod's threshold).
sample_rate:        sample rate in Hz.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 IQ, unit magnitude within the burst (FSK is
constant-envelope); the pad regions are silence.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `pad_symbols` |  | `0` |



### `instantaneous_frequency(iq, sample_rate=None)`

The instantaneous frequency of a complex signal. OUR code.

Computed by the phase discriminator: the phase change between consecutive
samples, angle(x[n] * conj(x[n-1])). This is THE primitive under FM and FSK
demodulation, exposed so both build on it (and so you can analyze frequency
directly -- Doppler, drift, chirps).

Returns radians/sample, or Hz if sample_rate is given. Output length is
len(iq) - 1 (one difference per adjacent pair).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |



### `instantaneous_phase(iq, unwrap=True)`

The phase angle of each complex sample. OUR code.

Returns the per-sample phase in radians. With unwrap=True the 2*pi jumps are
removed so the phase is continuous (useful for seeing accumulated phase /
measuring frequency as its slope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `unwrap` |  | `True` |



### `interpolate(iq, factor, half_len=10)`

Upsample by ``factor`` with interpolation filtering. OUR code.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `factor` |  | *required* |
| `half_len` |  | `10` |



### `matched_filter(signal, template)`

Correlate a known template against a signal. OUR code.

Returns the correlation magnitude; its peak marks where the template best
aligns with the signal.

NOTE: np.correlate already conjugates its second argument for complex
input, so the template is passed directly -- conjugating it ourselves would
double-conjugate and destroy the match. (Matched filtering for complex
baseband is correlation with the conjugated template, which is exactly what
np.correlate computes.)


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `signal` |  | *required* |
| `template` |  | *required* |



### `nask_slice(envelope, n_levels=4, levels=None)`

Slice an amplitude envelope into N levels (M-ASK). OUR code.

Generalizes 2-level OOK to N amplitude levels (4-ASK, 8-ASK). Returns a
per-sample symbol index in 0..n_levels-1.

levels: explicit amplitude thresholds/centers. If None, the levels are
spread uniformly from the envelope's min to its max -- a reasonable default
for a clean capture, but YOU can pass measured levels for real signals where
the spacing isn't uniform.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `n_levels` |  | `4` |
| `levels` |  | `None` |



### `normalize(iq, mode='peak', target=1.0)`

Rescale a signal's amplitude. EXPLICIT -- you choose if and how.

The library never normalizes silently; call this when you want it.

mode:
  "peak" : scale so max|x| == target (headroom-friendly; good before WAV
           output or display).
  "rms"  : scale so the RMS amplitude == target (good before a demod or
           detector that assumes a consistent level across captures).
  "none" : return unchanged (so a pipeline can be parameterized).
target: the desired peak or RMS level.

Returns a new array (does not modify the input). A zero/empty signal is
returned unchanged (nothing sensible to scale to).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `mode` |  | `'peak'` |
| `target` |  | `1.0` |



### `occupied_bandwidth(iq, sample_rate, fraction=0.99, nfft=1024)`

Bandwidth containing ``fraction`` of the total power (e.g. 99%).

Returns bandwidth in Hz. Integrates the PSD and finds the central band
holding the requested fraction of total power.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `fraction` |  | `0.99` |
| `nfft` |  | `1024` |



### `ook_envelope(iq)`

On-off-keying / ASK front end: the magnitude envelope. OUR code.

Returns |iq| (no DC block -- OOK threshold detection wants the absolute
level). Feed to ``ook_slice`` to recover bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `ook_modulate(bits, samples_per_symbol, high=1.0, low=0.0, pad_symbols=0)`

On-off keying: bit 1 -> carrier on, bit 0 -> off. Inverse of ook_slice.

The simplest digital modulation. Each bit becomes samples_per_symbol samples
at amplitude `high` (for 1) or `low` (for 0). ook_envelope + ook_slice
recover the bits by thresholding the magnitude.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 baseband (real-valued amplitude, zero phase).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `high` |  | `1.0` |
| `low` |  | `0.0` |
| `pad_symbols` |  | `0` |



### `ook_slice(envelope, threshold=None)`

Threshold an OOK envelope into a 0/1 stream. OUR code.

threshold: level above which a sample is '1'. If None, uses the midpoint
between the envelope's min and max (a simple, robust default for a clean
capture). Returns a uint8 array of 0/1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `threshold` |  | `None` |



### `power_dbfs(iq)`

Mean power of a complex signal in dBFS (dB relative to |amp|=1).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `power_dbm(iq, cal_offset_db)`

Absolute power in dBm = power_dbfs(iq) + cal_offset_db. OUR code.

The one-off, stateless form: you supply the calibration offset directly.
For repeated work, or to keep the offset with the conditions it's valid for,
use a `Calibration` object instead.

The result is only meaningful if cal_offset_db came from a real measurement
of a known reference at the SAME gain/frequency as `iq`. Garbage offset in,
confidently-wrong dBm out -- there is no way for this function to check that,
so the responsibility is yours.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `cal_offset_db` |  | *required* |



### `psd(iq, sample_rate, nfft=1024, window='hann', center_freq=0.0)`

Power spectral density of a complex signal via Welch averaging. OUR code.

Splits the signal into nfft-length frames, windows each, FFTs (numpy),
accumulates |X|^2, averages, and scales. Returns (freqs_hz, psd_db).

freqs_hz:  frequency axis centered on center_freq, fftshifted (low->high).
psd_db:    10*log10 of the averaged power spectrum.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `1024` |
| `window` |  | `'hann'` |
| `center_freq` |  | `0.0` |



### `psk8_demod(symbols)`

Demodulate 8-PSK from recovered symbols. OUR code.

8-PSK carries 3 bits/symbol in eight phase points (45-degree spacing).
COHERENT: assumes carrier-aligned, symbol-timed input (recover first, as in
qpsk_demod). Higher-order PSK demands more SNR -- the eight points are
closer together -- so on an 8-bit SDR like the HackRF this needs a strong,
clean signal. Returns (bits, sector) where bits is uint8 (3 per symbol) and
sector is the chosen 0..7 phase sector.

Honest note: at 45-degree spacing, a small residual carrier error rotates
points across decision boundaries, so good carrier recovery matters more
here than for QPSK.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `pulse_shape(symbols, sps, span_symbols=8, beta=0.35)`

Upsample symbols and shape them with an RRC pulse. OUR code.

The standard transmit-side digital chain: take complex symbols (e.g. PSK
constellation points), upsample by sps, and convolve with a root-raised-cosine
pulse so the result is bandlimited and ISI-free when matched-filtered on
receive. Returns the shaped complex64 baseband signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `qam16_demod(symbols, normalize=True)`

Demodulate QAM-16 from recovered symbols. OUR code.

COHERENT and amplitude-sensitive: assumes carrier-aligned, symbol-timed
input AND a known amplitude scale (QAM decisions depend on absolute level,
unlike PSK). Recover first, then normalize:

    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sym(corr, sps)
    bits, pts = qam16_demod(syms)   # normalize=True scales by RMS

The 16 points sit on a 4x4 grid at I,Q in {-3,-1,+1,+3} (scaled). Each axis
carries 2 Gray-coded bits. Returns (bits, points) -- 4 bits/symbol and the
chosen grid points (for plotting the constellation).

normalize=True scales the input so its RMS matches the standard grid; this
is the one place QAM needs an amplitude assumption, and it's explicit. Pass
normalize=False if you've already scaled the signal yourself.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `normalize` |  | `True` |



### `qpsk_demod(symbols, gray=True)`

Demodulate QPSK from recovered symbols. OUR code.

QPSK carries 2 bits/symbol in four phase points (the four quadrants of the
complex plane). This is a COHERENT demod: it assumes the symbols are already
carrier-aligned and symbol-timed. For a raw capture, recover first:

    from sdr_dsp.core import carrier_recovery, symbol_sync
    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sync(corr, sps)
    bits, _ = qpsk_demod(syms)

The library does NOT auto-recover -- you compose the recovery you want, so
nothing is hidden. Returns (bits, decisions) where bits is uint8 (2 per
symbol) and decisions are the constellation points chosen (for plotting).

gray=True uses Gray coding (adjacent quadrants differ by one bit), the
standard choice that minimizes bit errors.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `gray` |  | `True` |



### `qpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, gray=True, pad_symbols=0)`

QPSK: 2 bits per symbol, Gray-coded quadrants. Inverse of qpsk_demod.

Pairs of bits map to the four points (1+1j, -1+1j, -1-1j, 1-1j)/sqrt(2) via
the same Gray convention qpsk_demod uses, so the round-trip is exact. An odd
trailing bit is dropped (QPSK consumes bits in pairs).

bits:               sequence of 0/1 (length should be even).
samples_per_symbol: samples per symbol.
pulse_shaping:      RRC-shape the symbols if True.
gray:               use Gray coding (match the demod).

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `gray` |  | `True` |
| `pad_symbols` |  | `0` |



### `remove_dc(iq)`

Remove the DC offset / LO leakage: subtract the complex mean. OUR code.

Direct-conversion SDRs leak their local oscillator into the band center,
producing a spurious spike at 0 Hz that isn't a real signal. Subtracting the
mean removes it. Returns complex64.

CAVEAT: the whole-record mean is only the DC offset when the record is
mostly noise. If a strong burst dominates the record (a triggered packet
capture), the burst's own mean contaminates the estimate and subtracting
it bends the signal. In that case estimate DC from a signal-free segment
and subtract that -- or better, capture offset-tuned (tune the hardware
100-200 kHz off-target and tune_to_baseband in software) so the LO spike
is never near your signal in the first place. See docs/DC_SPIKE.md.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `resample_poly(iq, up, down, half_len=10, window='hamming')`

Rational resample by up/down. OUR implementation (polyphase concept).

Implements the classic upsample -> lowpass -> downsample, with the zero
insertion and decimation done explicitly and the lowpass applied by our own
``fir_apply``. Verified ~equal to ``scipy.signal.resample_poly`` in tests.

up, down:  resampling ratio (reduced internally).
half_len:  controls filter length / quality.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `up` |  | *required* |
| `down` |  | *required* |
| `half_len` |  | `10` |
| `window` |  | `'hamming'` |



### `rrc_taps(sps, span_symbols=8, beta=0.35)`

Root-raised-cosine filter taps. OUR code.

sps:          samples per symbol (the upsampling factor).
span_symbols: how many symbols wide the pulse is (longer = sharper spectrum).
beta:         roll-off factor in [0, 1]; larger = more bandwidth, gentler.

Returns the normalized tap array. Used on BOTH ends: shape on transmit,
matched-filter with the same taps on receive (root * root = raised cosine,
the zero-ISI pulse).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `sample_symbols(bits, samples_per_symbol, active=None)`

Decimate an over-sampled 0/1 stream to symbols at the RIGHT phase. OUR code.

A per-sample bit stream (from fsk_demod / ook_slice) carries each symbol
samples_per_symbol times, but a real capture arrives with an arbitrary
delay -- so a fixed stride like bits[sps//2::sps] samples at an arbitrary
point in each symbol, sometimes right on the transitions. This estimates
the symbol phase FROM the stream itself: transitions can only occur at
symbol boundaries, so the circular mean of (transition_index mod sps) is
the boundary phase, and boundary + sps/2 is the symbol center.

Contrast with slice_to_symbols, which is run-length based: robust to
unknown phase but a single glitch sample inserts/deletes a bit and shifts
everything after it. This keeps the fixed-stride robustness (a glitch
corrupts one bit, not the alignment) while removing the phase assumption.
Use this one on hardware captures.

bits:               per-sample 0/1 stream (uint8).
samples_per_symbol: the (integer) oversampling factor.
active:             optional boolean mask, same length as bits (or 1 less,
                    e.g. from a len-1 instantaneous-frequency chain).
                    Transitions outside the mask are ignored when
                    estimating the phase -- pass envelope > threshold so
                    garbage flicker in silence between bursts doesn't
                    pollute the estimate.

Returns a uint8 array with one entry per symbol period.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `active` |  | `None` |



### `slice_to_symbols(bits, samples_per_symbol)`

Collapse an over-sampled 0/1 stream into one bit per symbol. OUR code.

Given the samples-per-symbol, walk each run and emit its value repeated
round(run_length / spb) times. Returns a uint8 array of symbol bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `snr_db(iq, sample_rate, signal_band_hz, nfft=1024)`

Estimate SNR by comparing in-band power to out-of-band (noise) power.

signal_band_hz: (low, high) frequency range (relative to center) holding
                the signal. Everything else in the spectrum is treated as
                noise. A coarse but useful estimate.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `signal_band_hz` |  | *required* |
| `nfft` |  | `1024` |



### `spectrogram(iq, sample_rate, nfft=1024, overlap=0.5, window='hann', center_freq=0.0)`

Time-frequency spectrogram. OUR code (numpy FFT per frame).

Returns (freqs_hz, times_s, sxx_db) where sxx_db has shape
(n_frames, nfft): one spectrum row per hop.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `1024` |
| `overlap` |  | `0.5` |
| `window` |  | `'hann'` |
| `center_freq` |  | `0.0` |



### `ssb_demod(iq, sample_rate, sideband='usb', bfo_hz=0.0)`

Demodulate single-sideband (USB or LSB). OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `sideband` |  | `'usb'` |
| `bfo_hz` |  | `0.0` |



### `ssb_modulate(message, sideband='usb')`

SSB-modulate a real message into IQ. Inverse of ssb_demod. OUR code.

Single-sideband keeps one sideband of the message's analytic signal and
suppresses the carrier and the other sideband. We build the analytic signal
(positive frequencies only) for USB; conjugate for LSB. ssb_demod recovers
the real message by selecting the matching sideband.

message:  real message.
sideband: "usb" (upper) or "lsb" (lower).

Returns complex64 IQ -- the single-sideband analytic signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `sideband` |  | `'usb'` |



### `symbol_sync(iq, samples_per_symbol, method='gardner', loop_bw=0.01, damping=0.707, diagnostics=False, csv_path=None, lock_threshold=0.05)`

Recover symbol timing: pick the best sampling instant per symbol. OUR code.

Returns the symbol-spaced samples (one complex value per recovered symbol).
With diagnostics=True, returns (symbols, LoopDiagnostics) where the error is
the timing-error-detector output and the estimate is the fractional offset.

method:
  "gardner"        : Gardner TED -- carrier-independent, needs ~2 sps.
  "early_late"     : early-late gate -- simple, intuitive.
  "mueller_muller" : Mueller & Muller -- 1 sps, decision-aided.
samples_per_symbol: nominal sps (from estimate_symbol_rate or known rate).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `method` |  | `'gardner'` |
| `loop_bw` |  | `0.01` |
| `damping` |  | `0.707` |
| `diagnostics` |  | `False` |
| `csv_path` |  | `None` |
| `lock_threshold` |  | `0.05` |



### `to_db(x, *, power=True, epsilon=1e-20)`

Convert linear values to dB.

power=True  : x is a power quantity      -> 10*log10(x)
power=False : x is an amplitude/voltage  -> 20*log10(x)
epsilon floors the input so zeros don't produce -inf.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `x` |  | *required* |
| `power` |  | `True` |
| `epsilon` |  | `1e-20` |



### `tune_to_baseband(iq, offset_hz, sample_rate)`

Bring a signal sitting at +offset_hz down to 0 Hz (DC).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `offset_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `upsample(symbols, sps)`

Insert sps-1 zeros between symbols (zero-stuffing). OUR code.

The first step of pulse shaping: place each symbol on the output grid, then
filter to spread it into a pulse. Returns a complex64 array sps times longer.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |



---

## core — pure DSP

Every signal-processing primitive. numpy + scipy only.

Import: `import sdr_dsp.core`

### Classes

### class `AGC`

Streaming AGC: the agc() loop with memory across blocks.

A per-block AGC that reset each block would lurch at every boundary. This
holds the gain and level between blocks so the loop is continuous, and calls
the same agc() core -- it adds memory, not new DSP. Use it as a Pipeline
stage; the last gain trace is kept on `.last_gain` so a .tap() can watch it.

    stage = AGC(mode="rms", target=1.0, attack=0.01, decay=0.001)
    pipe.add(stage, "agc").tap(lambda b: meter.update(stage.last_gain))

Like the function, it never hides what it did: every processed block has its
gain trace available, and the same caveats apply (do calibration before AGC;
set max_gain to bound silent-gap runaway).


**Constructor:** `AGC(self, mode: 'str' = 'rms', target: 'float' = 1.0, attack: 'float' = 0.01, decay: 'float' = 0.001, max_gain: 'float | None' = None) -> None`

| Parameter | Type | Default |
|---|---|---|
| `mode` | `str` | `'rms'` |
| `target` | `float` | `1.0` |
| `attack` | `float` | `0.01` |
| `decay` | `float` | `0.001` |
| `max_gain` | `float | None` | `None` |


**Methods:**

#### `reset(self)`

Forget the carried state (start the next block fresh).



**Attributes:**

- `mode`: `str`
- `target`: `float`
- `attack`: `float`
- `decay`: `float`
- `max_gain`: `float | None`



### class `Calibration`

A dBFS->dBm offset plus the conditions it was measured under.

The offset alone is a footgun -- it's only valid for the receive-chain setup
it was measured at. So a Calibration carries that context: `frequency_hz`
(first-class, for the drift warning) and a free-form `conditions` dict for
everything else (gains, antenna, SDR model, temperature -- whatever you want
to record). `notes` and `measured_at` are for your own bookkeeping.

Apply it with `.power_dbm(iq)`. If you apply it far from `frequency_hz`, it
warns (the offset drifts with frequency); pass `warn=False` to silence, or
widen/disable the threshold with `drift_warn_hz`.


**Constructor:** `Calibration(self, offset_db: 'float', frequency_hz: 'float | None' = None, conditions: 'dict' = <factory>, notes: 'str' = '', measured_at: 'str' = '', drift_warn_hz: 'float' = 5000000.0) -> None`

| Parameter | Type | Default |
|---|---|---|
| `offset_db` | `float` | *required* |
| `frequency_hz` | `float | None` | `None` |
| `conditions` | `dict` | `<factory>` |
| `notes` | `str` | `''` |
| `measured_at` | `str` | `''` |
| `drift_warn_hz` | `float` | `5000000.0` |


**Methods:**

#### `load(path)`

Load a calibration saved by `save`.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |


#### `power_dbm(self, iq, at_frequency_hz=None, warn=True)`

Absolute power of `iq` in dBm, using this calibration's offset.

at_frequency_hz: the frequency `iq` was captured at, if known. When both
    this and the calibration's frequency_hz are set, applying the
    calibration more than drift_warn_hz away raises a warning (the offset
    is only valid near where it was measured).
warn: set False to silence the drift warning for this call.

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `at_frequency_hz` |  | `None` |
| `warn` |  | `True` |


#### `save(self, path)`

Write the calibration to a human-readable JSON file.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



**Attributes:**

- `offset_db`: `float`
- `frequency_hz`: `float | None`
- `conditions`: `dict`
- `notes`: `str`
- `measured_at`: `str`
- `drift_warn_hz`: `float`



### class `LoopDiagnostics`

Per-sample evidence of a recovery loop's behavior.

error:    the loop's error signal each sample (phase error, or timing
          error). Its settling toward ~0 is convergence.
estimate: the running quantity the loop tracks (accumulated phase, or the
          fractional sample offset).
lock:     per-sample boolean trace -- True where the error variance over a
          sliding window is below the lock threshold.
locked:   summary -- True if the loop was locked over the final portion.


**Constructor:** `LoopDiagnostics(self, error: 'np.ndarray', estimate: 'np.ndarray', lock: 'np.ndarray', locked: 'bool' = False) -> None`

| Parameter | Type | Default |
|---|---|---|
| `error` | `np.ndarray` | *required* |
| `estimate` | `np.ndarray` | *required* |
| `lock` | `np.ndarray` | *required* |
| `locked` | `bool` | `False` |


**Methods:**

#### `to_csv(self, path)`

Write the per-sample diagnostics to a CSV (sample, error, estimate,
lock). Useful for plotting convergence outside the library.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



**Attributes:**

- `error`: `np.ndarray`
- `estimate`: `np.ndarray`
- `lock`: `np.ndarray`
- `locked`: `bool`



### Functions

### `add_cfo(iq, cfo_hz, sample_rate)`

Apply a carrier frequency offset (Hz). OUR code.

Multiplies by a complex exponential at cfo_hz -- the rotation a real link
imposes when the TX and RX oscillators differ. Recoverable on the RX side by
carrier recovery, or measurable with estimate_cfo.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `cfo_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `add_delay(iq, delay_samples)`

Delay the signal by an integer number of samples (zero-pad the front).

Models propagation delay / a late frame start. The output is the same length
as the input (the tail is truncated); negative delay advances. OUR code.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `delay_samples` |  | *required* |



### `add_noise(iq, snr_db, rng=None)`

Add complex AWGN at a specified SNR (dB) relative to the signal. OUR code.

Measures the signal's mean power, computes the noise power for the requested
SNR, and adds complex Gaussian noise at that level. This is the honest way to
set noise -- by the SNR you want, not an arbitrary amplitude.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `snr_db` |  | *required* |
| `rng` |  | `None` |



### `agc(iq, mode='rms', target=1.0, attack=0.01, decay=0.001, max_gain=None, _initial_gain=1.0, _initial_level=None)`

Apply automatic gain control. Returns (adjusted_iq, gain_trace). OUR code.

Drives the signal level toward `target` with a one-pole tracking loop. The
gain rises slowly when the signal is weak (decay rate) and clamps down
quickly when it's strong (attack rate) -- fast attack avoids clipping, slow
decay avoids pumping on noise. Both rates are in (0, 1]; larger = faster.

mode:     "rms" tracks average power (smoother; good for analog/voice),
          "peak" tracks the running peak (twitchier; better anti-clipping).
target:   the level the loop steers the signal toward.
attack:   tracking rate when the measured level is ABOVE target (gain down).
decay:    tracking rate when the measured level is BELOW target (gain up).
max_gain: optional ceiling on the gain (None = no ceiling; see the module
          note about gain runaway during silence).

Returns:
    adjusted_iq:  iq * gain_trace
    gain_trace:   the per-sample gain that was applied (same length as iq).
                  This is the whole point -- it makes the AGC observable and
                  reversible: iq == adjusted_iq / gain_trace.

The _initial_* args let the streaming AGC stage continue a loop across blocks
and aren't normally set by hand.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `mode` |  | `'rms'` |
| `target` |  | `1.0` |
| `attack` |  | `0.01` |
| `decay` |  | `0.001` |
| `max_gain` |  | `None` |
| `_initial_gain` |  | `1.0` |
| `_initial_level` |  | `None` |



### `am_demod(iq, dc_block=True)`

Demodulate amplitude modulation: the envelope (magnitude). OUR code.

Returns the real envelope |iq|. With dc_block, the mean (carrier DC) is
removed so the output swings around zero like audio.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `dc_block` |  | `True` |



### `am_modulate(message, modulation_index=0.5)`

AM-modulate a real message into IQ. Inverse of am_demod. OUR code.

Amplitude modulation rides the message on the carrier envelope:
(1 + k*m) carrier. am_demod recovers it with an envelope detector. Keep
modulation_index <= 1 to avoid over-modulation (envelope going negative,
which the envelope detector can't undo).

message:          real message, roughly [-1, 1].
modulation_index: depth k of modulation (0..1). >1 over-modulates.

Returns complex64 IQ with the message in its magnitude.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `modulation_index` |  | `0.5` |



### `apply_channel(iq, sample_rate=None, snr_db=None, cfo_hz=None, delay_samples=0, scale=1.0, phase=0.0, seed=None)`

Pass a signal through a simulated channel. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |
| `snr_db` |  | `None` |
| `cfo_hz` |  | `None` |
| `delay_samples` |  | `0` |
| `scale` |  | `1.0` |
| `phase` |  | `0.0` |
| `seed` |  | `None` |



### `bpsk_demod(iq, normalize_phase=True)`

Demodulate binary phase-shift keying (coherent-ish). OUR code.

BPSK encodes bits as 0 or pi phase. With the carrier already at baseband and
roughly phase-aligned, the sign of the real part recovers the bits. This is
a SIMPLE demod: it assumes the signal is already carrier-aligned (no Costas
loop / carrier recovery). For captures with a residual carrier offset,
correct it first (see estimate_cfo / frequency_shift) -- the library does
not auto-recover the carrier.

Returns (bits, soft) where bits is uint8 (0/1) and soft is the real-part
decision statistic (useful for confidence / plotting a constellation).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `normalize_phase` |  | `True` |



### `bpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, pad_symbols=0)`

BPSK: bit 0 -> +1, bit 1 -> -1 (phase 0 or pi). Inverse of bpsk_demod.

Carries one bit per symbol in the carrier phase. With pulse_shaping=False the
symbols are held rectangular (sps samples each); with pulse_shaping=True they
are RRC-shaped for a bandlimited spectrum (use a matched RRC filter on
receive). bpsk_demod recovers bits from the real part's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per symbol (1 = one sample per symbol).
pulse_shaping:      RRC-shape the symbols if True.

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `pad_symbols` |  | `0` |



### `build_frame(payload, sync=None, preamble_bits=32)`

Build a complete frame from a payload. Returns a bit array (0/1). OUR code.

Layout: [preamble][sync][length:1 byte][payload][crc16:2 bytes].
The length byte limits a single frame to 255 payload bytes; split larger
messages across frames yourself.

payload:        bytes-like (bytes, bytearray, or a list of ints 0..255).
sync:           the sync word bits (default DEFAULT_SYNC).
preamble_bits:  number of alternating preamble bits.

The CRC covers the length byte and the payload, so a receiver validates both.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `payload` |  | *required* |
| `sync` |  | `None` |
| `preamble_bits` |  | `32` |



### `carrier_recovery(iq, method='costas', order=2, loop_bw=0.01, damping=0.707, diagnostics=False, csv_path=None, lock_threshold=0.05)`

Track and remove residual carrier phase/frequency offset. OUR code.

Returns a phase-corrected copy of iq (constellation de-rotated). With
diagnostics=True, returns (corrected, LoopDiagnostics). With csv_path set,
also writes the diagnostics CSV.

method:
  "costas"            : a Costas loop -- data-independent phase tracking.
                        order=2 for BPSK, order=4 for QPSK (the phase error
                        detector matches the modulation's symmetry).
  "decision_directed" : uses the nearest-symbol decision to form the error;
                        simpler, better at high SNR, needs order too.
loop_bw, damping: second-order loop filter parameters. Smaller loop_bw =
  slower but steadier lock. These are exposed on purpose.
order: 2 (BPSK-like) or 4 (QPSK-like) phase symmetry.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `method` |  | `'costas'` |
| `order` |  | `2` |
| `loop_bw` |  | `0.01` |
| `damping` |  | `0.707` |
| `diagnostics` |  | `False` |
| `csv_path` |  | `None` |
| `lock_threshold` |  | `0.05` |



### `channelize(iq, sample_rate, offset_hz, channel_bw, decim=None)`

Extract the single channel at offset_hz with bandwidth channel_bw. OUR code.

tune -> lowpass -> decimate. Returns (channel_iq, new_sample_rate). decim
defaults to the largest integer that keeps the channel comfortably inside
the new Nyquist (new rate >= ~2.5x the channel bandwidth).

Use this when you want one specific channel at an arbitrary offset/width.
For splitting the whole band into a uniform grid, use channelize_bank.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `offset_hz` |  | *required* |
| `channel_bw` |  | *required* |
| `decim` |  | `None` |



### `channelize_bank(iq, sample_rate, n_channels, decim=None, taps_per_channel=12, return_freqs=True)`

Split the band into n_channels equal channels via a polyphase filterbank.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `n_channels` |  | *required* |
| `decim` |  | `None` |
| `taps_per_channel` |  | `12` |
| `return_freqs` |  | `True` |



### `compute_cal_offset(iq, known_dbm, frequency_hz=None, conditions=None, notes='', drift_warn_hz=5000000.0)`

Derive a calibration from a measurement of a KNOWN-power reference.

Feed in `iq` captured from a calibrated source whose true power is
`known_dbm` (e.g. a signal generator set to -30 dBm), and this returns a
ready-to-use Calibration stamped with the conditions you pass:

    offset = known_dbm - power_dbfs(reference_iq)

Record the conditions honestly -- the offset is only valid at the gain and
frequency this reference was captured at. Example:

    cal = compute_cal_offset(ref_iq, known_dbm=-30.0,
                             frequency_hz=433.92e6,
                             conditions={"lna": 16, "vga": 20, "amp": False})
    cal.save("hackrf_433.cal.json")
    ...
    cal = Calibration.load("hackrf_433.cal.json")
    dbm = cal.power_dbm(capture, at_frequency_hz=433.92e6)

Returns a Calibration. The measurement should be on a steady reference tone;
a noisy or fluctuating source gives a noisy offset.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `known_dbm` |  | *required* |
| `frequency_hz` |  | `None` |
| `conditions` |  | `None` |
| `notes` |  | `''` |
| `drift_warn_hz` |  | `5000000.0` |



### `convolve(a, b, mode='full')`

Convolution of two signals. OUR code (thin, for a uniform API).

Unlike correlate, convolution does NOT conjugate -- it's the filtering
operation. Provided alongside correlate so the distinction is explicit and
both are first-class.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `a` |  | *required* |
| `b` |  | *required* |
| `mode` |  | `'full'` |



### `correlate(a, b, mode='full')`

Cross-correlation of two signals, conjugation handled correctly. OUR code.

np.correlate already conjugates its second argument for complex input -- a
well-known footgun (conjugating it yourself double-conjugates and breaks the
result). This wrapper exists so that subtlety lives in ONE place. Returns
the complex cross-correlation; take np.abs for a magnitude.

mode: "full", "same", or "valid" (as numpy).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `a` |  | *required* |
| `b` |  | *required* |
| `mode` |  | `'full'` |



### `crc16(data)`

CRC-16/CCITT-FALSE over a bytes-like input. OUR code.

A standard 16-bit CRC (poly 0x1021, init 0xFFFF). Used to detect whether a
received payload is intact. Not cryptographic -- just error detection.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `data` |  | *required* |



### `cw_decode(bits, samples_per_symbol)`

Decode Morse (CW) from a sliced on/off stream. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `dbpsk_demod(symbols)`

Demodulate differential BPSK. OUR code.

Differential PSK encodes bits in phase CHANGES between consecutive symbols,
not absolute phase. That's the whole point: it needs NO carrier recovery,
because a constant phase offset cancels when you compare adjacent symbols.
This makes it robust and a good fit for block processing.

Takes symbol-spaced samples (one per symbol -- use symbol_sync first if you
have oversampled data). A bit is 0 if the phase barely changed, 1 if it
flipped by ~pi. Returns (bits, soft) where soft is the real part of the
differential product (sign gives the bit, magnitude gives confidence).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `decimate(iq, factor, half_len=10)`

Lowpass then keep every ``factor``-th sample. OUR code (via resample).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `factor` |  | *required* |
| `half_len` |  | `10` |



### `deemphasis(audio, sample_rate, tau_us=75.0)`

Single-pole de-emphasis filter for broadcast FM audio. OUR code.

Broadcast FM pre-emphasizes high frequencies before transmission; the
receiver must de-emphasize them back. A one-pole IIR does it:
    y[n] = a*x[n] + (1-a)*y[n-1],   a = dt / (tau + dt)
tau_us: time constant (75 us in the Americas/Korea, 50 us elsewhere).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `audio` |  | *required* |
| `sample_rate` |  | *required* |
| `tau_us` |  | `75.0` |



### `design_bandpass(low_hz, high_hz, sample_rate, num_taps=101, window='hamming')`

Design a bandpass FIR. Returns tap coefficients.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `low_hz` |  | *required* |
| `high_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `design_highpass(cutoff_hz, sample_rate, num_taps=101, window='hamming')`

Design a highpass FIR. Returns tap coefficients.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `cutoff_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `design_lowpass(cutoff_hz, sample_rate, num_taps=101, window='hamming')`

Design a lowpass FIR. Returns tap coefficients (numpy array).

cutoff_hz:   passband edge in Hz.
sample_rate: Hz.
num_taps:    filter length (odd recommended for a linear-phase Type-I FIR).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `cutoff_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `num_taps` |  | `101` |
| `window` |  | `'hamming'` |



### `detect_peak(signal, template, threshold=None)`

Run a matched filter and return the best-match index (and its value).

threshold: if given, returns (index, value) only when the peak exceeds it,
else (None, value). Without a threshold, always returns the argmax.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `signal` |  | *required* |
| `template` |  | *required* |
| `threshold` |  | `None` |



### `dqpsk_demod(symbols)`

Demodulate differential QPSK. OUR code.

The QPSK analogue of DBPSK: 2 bits per symbol encoded in the phase CHANGE
(one of four ~90-degree steps), so it also needs no carrier recovery. Takes
symbol-spaced samples; returns (bits, phase_diffs) where bits is a uint8
array (2 per symbol, MSB first) and phase_diffs are the raw differential
angles for inspection.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `dsb_sc_demod(iq, sample_rate, bfo_hz=0.0)`

Demodulate double-sideband suppressed-carrier (DSB-SC). OUR code.

DSB-SC is AM with the carrier removed -- both sidebands, no carrier spike.
With the carrier suppressed there's no envelope to follow, so recovery needs
a coherent reference. For a complex baseband capture centered on the
(suppressed) carrier, the real part IS the message (the two sidebands beat
back together). A bfo_hz shift fine-tunes if the center is slightly off.

This is the conceptual midpoint between AM (carrier present, envelope) and
SSB (one sideband): DSB-SC keeps both sidebands but drops the carrier.
Returns the real demodulated message.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `bfo_hz` |  | `0.0` |



### `dsss_despread(iq, code, samples_per_chip=1)`

Despread a DSSS signal using a KNOWN spreading code. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `code` |  | *required* |
| `samples_per_chip` |  | `1` |



### `edges(bits)`

Indices where a 0/1 stream transitions, and the run lengths. OUR code.

Returns (transition_indices, run_lengths, run_values): the sample index of
each transition, how many samples each run lasted, and whether that run was
0 or 1. The building block for recovering symbol timing from a sliced
on/off stream.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |



### `estimate_cfo(iq, sample_rate, nfft=None)`

Estimate a signal's carrier frequency offset from band center. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `None` |



### `estimate_symbol_rate(bits, sample_rate, min_run=2)`

Estimate samples-per-symbol from the run lengths in a sliced stream.

The shortest on/off run is (usually) one symbol period. OUR code. Returns
(samples_per_symbol, symbol_rate_hz).

Robustness: a single glitch sample (from noise or a demod transient at a
symbol boundary) creates a spurious 1-sample run that would fool a naive
"minimum run" estimate. So we DISCARD runs shorter than min_run, then take a
low percentile of what remains as the symbol period -- stable against a few
outliers while still finding the shortest real symbol. Raise min_run if your
capture is very noisy; lower it (to 1) only for pristine synthetic data.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `sample_rate` |  | *required* |
| `min_run` |  | `2` |



### `fhss_detect_hops(iq, sample_rate, nfft=256, overlap=0.5, center_freq=0.0)`

Detect frequency hops: the dominant frequency per time slice. OUR code.

For an FHSS signal, computes a spectrogram and reports, for each time slice,
where the energy is -- i.e. which channel the hopper is in at that moment.
This TRACKS hops you can see; it does NOT decode the data or know the hop
sequence (out of scope). Pair it with core.spectrogram to SEE the hops.

Returns (times, hop_freqs) where hop_freqs[i] is the peak frequency (Hz,
offset by center_freq) during time slice times[i].


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `256` |
| `overlap` |  | `0.5` |
| `center_freq` |  | `0.0` |



### `find_bursts(iq, sample_rate=None, threshold=None, min_gap=0, min_len=1)`

Find where signal energy is present: burst start/stop indices. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |
| `threshold` |  | `None` |
| `min_gap` |  | `0` |
| `min_len` |  | `1` |



### `find_frames(bits, sync=None, max_sync_errors=2)`

Find and validate frames in a recovered bit stream. OUR code.

Searches for the sync word (allowing up to max_sync_errors bit mismatches,
since recovered bits may have errors), then reads the length byte, payload,
and CRC after each match. Returns a list of dicts, one per frame found:

    {"payload": bytes, "crc_ok": bool, "bit_offset": int}

crc_ok tells you whether the payload survived intact -- the basis for an ACK.
Frames with a bad length read or running off the end of the buffer are
skipped. Overlapping/false sync matches inside a validated frame are stepped
past so one packet isn't reported twice.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `sync` |  | `None` |
| `max_sync_errors` |  | `2` |



### `fir_apply(iq, taps)`

Apply an FIR filter to a signal by direct convolution. OUR code.

Equivalent to ``scipy.signal.lfilter(taps, [1.0], iq)`` but implemented
here as a full convolution (then truncated to the input length) so the
filtering operation is the library's own. Works on real or complex input;
complex IQ is filtered as a whole (numpy.convolve handles complex).

Returns an array the same length as ``iq`` (the causal 'full' convolution
truncated to the first len(iq) samples -- matching lfilter's output).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `taps` |  | *required* |



### `fir_apply_centered(iq, taps)`

Apply an FIR with the group delay removed (zero-phase alignment).

A linear-phase FIR of length L delays the signal by (L-1)/2 samples. For
analysis where you want the output time-aligned with the input, this
returns the 'same'-mode convolution (centered), trimming the delay.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `taps` |  | *required* |



### `fm_demod(iq, deviation_hz=None, sample_rate=None)`

Demodulate frequency modulation via the phase discriminator. OUR code.

FM carries the message in instantaneous frequency, so demod IS the
instantaneous frequency (see instantaneous_frequency). Returns a real array.

If deviation_hz and sample_rate are given, the output is scaled by the peak
deviation to give roughly normalized audio; otherwise it returns raw
radians/sample.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `deviation_hz` |  | `None` |
| `sample_rate` |  | `None` |



### `fm_modulate(message, deviation_hz, sample_rate)`

FM-modulate a real message into IQ. Inverse of fm_demod. OUR code.

Frequency modulation encodes the message in the carrier's instantaneous
frequency: the IQ phase is the running integral of the message, scaled by
the deviation. fm_demod recovers it by differentiating the phase.

message:      real-valued message, expected roughly in [-1, 1].
deviation_hz: peak frequency deviation (must match the demod's deviation_hz).
sample_rate:  sample rate in Hz.

Returns unit-magnitude complex64 IQ (FM is constant-envelope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `frequency_shift(iq, shift_hz, sample_rate)`

Shift a complex signal up (positive) or down (negative) in frequency.

Multiplies by exp(j*2*pi*shift*t). To bring a signal at offset f to
baseband (0 Hz), pass shift_hz = -f.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `shift_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `from_db(db, *, power=True)`

Inverse of to_db: dB back to a linear value.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `db` |  | *required* |
| `power` |  | `True` |



### `fsk_demod(iq, sample_rate, threshold_hz=0.0, smooth_samples=0)`

Demodulate 2-level frequency-shift keying. OUR code.

FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
instantaneous frequency, then a threshold: above threshold_hz -> 1, below
-> 0. With the default threshold 0, it splits on the sign of the frequency
deviation (correct when the two tones straddle the center frequency, which
is the common case after tuning to baseband).

threshold_hz: the mark/space decision frequency. Two real radios never
    share an oscillator, so a carrier frequency offset (crystal ppm --
    easily +/-10-20 kHz at 433 MHz between two SDRs) shifts BOTH tones and
    biases the fixed 0 Hz split. Pass "auto" to threshold at the
    amplitude^2-weighted mean of the instantaneous frequency instead: with
    roughly balanced mark/space time (any frame with an alternating
    preamble qualifies) that mean IS the offset, so the split self-centers.
    The weighting means silence around a burst contributes ~nothing.
    NOTE: do not use estimate_cfo for this -- it finds the strongest
    spectral tone, which for FSK is +/-deviation, not the offset.
smooth_samples: if > 1, moving-average the instantaneous frequency over
    this many samples before slicing (a cheap matched-filter stand-in;
    ~samples_per_symbol/2 is a good value). The raw per-sample frequency
    is noisy, and this is the difference between decoding and not at
    moderate SNR. Off by default -- the per-sample output stays exact.

Returns a uint8 per-sample bit stream (length len(iq)-1); feed to the
timing helpers (sample_symbols for hardware captures with unknown delay,
or estimate_symbol_rate / slice_to_symbols) to get symbols. Covers
GFSK/MSK well enough for typical ISM-band sensors and pagers.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `threshold_hz` |  | `0.0` |
| `smooth_samples` |  | `0` |



### `fsk_demod_nlevel(iq, sample_rate, n_levels=4, thresholds=None)`

Demodulate N-level FSK (4-FSK, etc.) and CPFSK. OUR code.

Generalizes 2-FSK: instead of a single 0-threshold on instantaneous
frequency, it slices the frequency into n_levels bands. Used by 4-FSK
(DMR, P25, some pagers). CPFSK recovers the same way -- the continuous
phase is a transmit-side property; the receiver still reads instantaneous
frequency.

thresholds: explicit frequency band centers (Hz). If None, the levels are
spread uniformly across the observed frequency range -- fine for a clean
capture; pass measured centers for real signals. Returns per-sample symbol
indices 0..n_levels-1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `n_levels` |  | `4` |
| `thresholds` |  | `None` |



### `fsk_modulate(bits, samples_per_symbol, deviation_hz, sample_rate, pad_symbols=0)`

Binary FSK: bit selects one of two frequencies. Inverse of fsk_demod.

Bit 1 -> +deviation_hz, bit 0 -> -deviation_hz, encoded as a continuous-phase
frequency shift (CPFSK -- the phase is integrated so there are no jumps,
which keeps the spectrum clean). fsk_demod recovers bits from the
instantaneous frequency's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
deviation_hz:       frequency shift magnitude (match the demod's threshold).
sample_rate:        sample rate in Hz.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 IQ, unit magnitude within the burst (FSK is
constant-envelope); the pad regions are silence.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `pad_symbols` |  | `0` |



### `instantaneous_frequency(iq, sample_rate=None)`

The instantaneous frequency of a complex signal. OUR code.

Computed by the phase discriminator: the phase change between consecutive
samples, angle(x[n] * conj(x[n-1])). This is THE primitive under FM and FSK
demodulation, exposed so both build on it (and so you can analyze frequency
directly -- Doppler, drift, chirps).

Returns radians/sample, or Hz if sample_rate is given. Output length is
len(iq) - 1 (one difference per adjacent pair).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |



### `instantaneous_phase(iq, unwrap=True)`

The phase angle of each complex sample. OUR code.

Returns the per-sample phase in radians. With unwrap=True the 2*pi jumps are
removed so the phase is continuous (useful for seeing accumulated phase /
measuring frequency as its slope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `unwrap` |  | `True` |



### `interpolate(iq, factor, half_len=10)`

Upsample by ``factor`` with interpolation filtering. OUR code.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `factor` |  | *required* |
| `half_len` |  | `10` |



### `matched_filter(signal, template)`

Correlate a known template against a signal. OUR code.

Returns the correlation magnitude; its peak marks where the template best
aligns with the signal.

NOTE: np.correlate already conjugates its second argument for complex
input, so the template is passed directly -- conjugating it ourselves would
double-conjugate and destroy the match. (Matched filtering for complex
baseband is correlation with the conjugated template, which is exactly what
np.correlate computes.)


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `signal` |  | *required* |
| `template` |  | *required* |



### `nask_slice(envelope, n_levels=4, levels=None)`

Slice an amplitude envelope into N levels (M-ASK). OUR code.

Generalizes 2-level OOK to N amplitude levels (4-ASK, 8-ASK). Returns a
per-sample symbol index in 0..n_levels-1.

levels: explicit amplitude thresholds/centers. If None, the levels are
spread uniformly from the envelope's min to its max -- a reasonable default
for a clean capture, but YOU can pass measured levels for real signals where
the spacing isn't uniform.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `n_levels` |  | `4` |
| `levels` |  | `None` |



### `normalize(iq, mode='peak', target=1.0)`

Rescale a signal's amplitude. EXPLICIT -- you choose if and how.

The library never normalizes silently; call this when you want it.

mode:
  "peak" : scale so max|x| == target (headroom-friendly; good before WAV
           output or display).
  "rms"  : scale so the RMS amplitude == target (good before a demod or
           detector that assumes a consistent level across captures).
  "none" : return unchanged (so a pipeline can be parameterized).
target: the desired peak or RMS level.

Returns a new array (does not modify the input). A zero/empty signal is
returned unchanged (nothing sensible to scale to).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `mode` |  | `'peak'` |
| `target` |  | `1.0` |



### `occupied_bandwidth(iq, sample_rate, fraction=0.99, nfft=1024)`

Bandwidth containing ``fraction`` of the total power (e.g. 99%).

Returns bandwidth in Hz. Integrates the PSD and finds the central band
holding the requested fraction of total power.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `fraction` |  | `0.99` |
| `nfft` |  | `1024` |



### `ook_envelope(iq)`

On-off-keying / ASK front end: the magnitude envelope. OUR code.

Returns |iq| (no DC block -- OOK threshold detection wants the absolute
level). Feed to ``ook_slice`` to recover bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `ook_modulate(bits, samples_per_symbol, high=1.0, low=0.0, pad_symbols=0)`

On-off keying: bit 1 -> carrier on, bit 0 -> off. Inverse of ook_slice.

The simplest digital modulation. Each bit becomes samples_per_symbol samples
at amplitude `high` (for 1) or `low` (for 0). ook_envelope + ook_slice
recover the bits by thresholding the magnitude.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 baseband (real-valued amplitude, zero phase).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `high` |  | `1.0` |
| `low` |  | `0.0` |
| `pad_symbols` |  | `0` |



### `ook_slice(envelope, threshold=None)`

Threshold an OOK envelope into a 0/1 stream. OUR code.

threshold: level above which a sample is '1'. If None, uses the midpoint
between the envelope's min and max (a simple, robust default for a clean
capture). Returns a uint8 array of 0/1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `threshold` |  | `None` |



### `power_dbfs(iq)`

Mean power of a complex signal in dBFS (dB relative to |amp|=1).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `power_dbm(iq, cal_offset_db)`

Absolute power in dBm = power_dbfs(iq) + cal_offset_db. OUR code.

The one-off, stateless form: you supply the calibration offset directly.
For repeated work, or to keep the offset with the conditions it's valid for,
use a `Calibration` object instead.

The result is only meaningful if cal_offset_db came from a real measurement
of a known reference at the SAME gain/frequency as `iq`. Garbage offset in,
confidently-wrong dBm out -- there is no way for this function to check that,
so the responsibility is yours.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `cal_offset_db` |  | *required* |



### `psd(iq, sample_rate, nfft=1024, window='hann', center_freq=0.0)`

Power spectral density of a complex signal via Welch averaging. OUR code.

Splits the signal into nfft-length frames, windows each, FFTs (numpy),
accumulates |X|^2, averages, and scales. Returns (freqs_hz, psd_db).

freqs_hz:  frequency axis centered on center_freq, fftshifted (low->high).
psd_db:    10*log10 of the averaged power spectrum.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `1024` |
| `window` |  | `'hann'` |
| `center_freq` |  | `0.0` |



### `psk8_demod(symbols)`

Demodulate 8-PSK from recovered symbols. OUR code.

8-PSK carries 3 bits/symbol in eight phase points (45-degree spacing).
COHERENT: assumes carrier-aligned, symbol-timed input (recover first, as in
qpsk_demod). Higher-order PSK demands more SNR -- the eight points are
closer together -- so on an 8-bit SDR like the HackRF this needs a strong,
clean signal. Returns (bits, sector) where bits is uint8 (3 per symbol) and
sector is the chosen 0..7 phase sector.

Honest note: at 45-degree spacing, a small residual carrier error rotates
points across decision boundaries, so good carrier recovery matters more
here than for QPSK.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `pulse_shape(symbols, sps, span_symbols=8, beta=0.35)`

Upsample symbols and shape them with an RRC pulse. OUR code.

The standard transmit-side digital chain: take complex symbols (e.g. PSK
constellation points), upsample by sps, and convolve with a root-raised-cosine
pulse so the result is bandlimited and ISI-free when matched-filtered on
receive. Returns the shaped complex64 baseband signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `qam16_demod(symbols, normalize=True)`

Demodulate QAM-16 from recovered symbols. OUR code.

COHERENT and amplitude-sensitive: assumes carrier-aligned, symbol-timed
input AND a known amplitude scale (QAM decisions depend on absolute level,
unlike PSK). Recover first, then normalize:

    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sym(corr, sps)
    bits, pts = qam16_demod(syms)   # normalize=True scales by RMS

The 16 points sit on a 4x4 grid at I,Q in {-3,-1,+1,+3} (scaled). Each axis
carries 2 Gray-coded bits. Returns (bits, points) -- 4 bits/symbol and the
chosen grid points (for plotting the constellation).

normalize=True scales the input so its RMS matches the standard grid; this
is the one place QAM needs an amplitude assumption, and it's explicit. Pass
normalize=False if you've already scaled the signal yourself.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `normalize` |  | `True` |



### `qpsk_demod(symbols, gray=True)`

Demodulate QPSK from recovered symbols. OUR code.

QPSK carries 2 bits/symbol in four phase points (the four quadrants of the
complex plane). This is a COHERENT demod: it assumes the symbols are already
carrier-aligned and symbol-timed. For a raw capture, recover first:

    from sdr_dsp.core import carrier_recovery, symbol_sync
    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sync(corr, sps)
    bits, _ = qpsk_demod(syms)

The library does NOT auto-recover -- you compose the recovery you want, so
nothing is hidden. Returns (bits, decisions) where bits is uint8 (2 per
symbol) and decisions are the constellation points chosen (for plotting).

gray=True uses Gray coding (adjacent quadrants differ by one bit), the
standard choice that minimizes bit errors.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `gray` |  | `True` |



### `qpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, gray=True, pad_symbols=0)`

QPSK: 2 bits per symbol, Gray-coded quadrants. Inverse of qpsk_demod.

Pairs of bits map to the four points (1+1j, -1+1j, -1-1j, 1-1j)/sqrt(2) via
the same Gray convention qpsk_demod uses, so the round-trip is exact. An odd
trailing bit is dropped (QPSK consumes bits in pairs).

bits:               sequence of 0/1 (length should be even).
samples_per_symbol: samples per symbol.
pulse_shaping:      RRC-shape the symbols if True.
gray:               use Gray coding (match the demod).

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `gray` |  | `True` |
| `pad_symbols` |  | `0` |



### `remove_dc(iq)`

Remove the DC offset / LO leakage: subtract the complex mean. OUR code.

Direct-conversion SDRs leak their local oscillator into the band center,
producing a spurious spike at 0 Hz that isn't a real signal. Subtracting the
mean removes it. Returns complex64.

CAVEAT: the whole-record mean is only the DC offset when the record is
mostly noise. If a strong burst dominates the record (a triggered packet
capture), the burst's own mean contaminates the estimate and subtracting
it bends the signal. In that case estimate DC from a signal-free segment
and subtract that -- or better, capture offset-tuned (tune the hardware
100-200 kHz off-target and tune_to_baseband in software) so the LO spike
is never near your signal in the first place. See docs/DC_SPIKE.md.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `resample_poly(iq, up, down, half_len=10, window='hamming')`

Rational resample by up/down. OUR implementation (polyphase concept).

Implements the classic upsample -> lowpass -> downsample, with the zero
insertion and decimation done explicitly and the lowpass applied by our own
``fir_apply``. Verified ~equal to ``scipy.signal.resample_poly`` in tests.

up, down:  resampling ratio (reduced internally).
half_len:  controls filter length / quality.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `up` |  | *required* |
| `down` |  | *required* |
| `half_len` |  | `10` |
| `window` |  | `'hamming'` |



### `rrc_taps(sps, span_symbols=8, beta=0.35)`

Root-raised-cosine filter taps. OUR code.

sps:          samples per symbol (the upsampling factor).
span_symbols: how many symbols wide the pulse is (longer = sharper spectrum).
beta:         roll-off factor in [0, 1]; larger = more bandwidth, gentler.

Returns the normalized tap array. Used on BOTH ends: shape on transmit,
matched-filter with the same taps on receive (root * root = raised cosine,
the zero-ISI pulse).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `sample_symbols(bits, samples_per_symbol, active=None)`

Decimate an over-sampled 0/1 stream to symbols at the RIGHT phase. OUR code.

A per-sample bit stream (from fsk_demod / ook_slice) carries each symbol
samples_per_symbol times, but a real capture arrives with an arbitrary
delay -- so a fixed stride like bits[sps//2::sps] samples at an arbitrary
point in each symbol, sometimes right on the transitions. This estimates
the symbol phase FROM the stream itself: transitions can only occur at
symbol boundaries, so the circular mean of (transition_index mod sps) is
the boundary phase, and boundary + sps/2 is the symbol center.

Contrast with slice_to_symbols, which is run-length based: robust to
unknown phase but a single glitch sample inserts/deletes a bit and shifts
everything after it. This keeps the fixed-stride robustness (a glitch
corrupts one bit, not the alignment) while removing the phase assumption.
Use this one on hardware captures.

bits:               per-sample 0/1 stream (uint8).
samples_per_symbol: the (integer) oversampling factor.
active:             optional boolean mask, same length as bits (or 1 less,
                    e.g. from a len-1 instantaneous-frequency chain).
                    Transitions outside the mask are ignored when
                    estimating the phase -- pass envelope > threshold so
                    garbage flicker in silence between bursts doesn't
                    pollute the estimate.

Returns a uint8 array with one entry per symbol period.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `active` |  | `None` |



### `slice_to_symbols(bits, samples_per_symbol)`

Collapse an over-sampled 0/1 stream into one bit per symbol. OUR code.

Given the samples-per-symbol, walk each run and emit its value repeated
round(run_length / spb) times. Returns a uint8 array of symbol bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `snr_db(iq, sample_rate, signal_band_hz, nfft=1024)`

Estimate SNR by comparing in-band power to out-of-band (noise) power.

signal_band_hz: (low, high) frequency range (relative to center) holding
                the signal. Everything else in the spectrum is treated as
                noise. A coarse but useful estimate.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `signal_band_hz` |  | *required* |
| `nfft` |  | `1024` |



### `spectrogram(iq, sample_rate, nfft=1024, overlap=0.5, window='hann', center_freq=0.0)`

Time-frequency spectrogram. OUR code (numpy FFT per frame).

Returns (freqs_hz, times_s, sxx_db) where sxx_db has shape
(n_frames, nfft): one spectrum row per hop.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `1024` |
| `overlap` |  | `0.5` |
| `window` |  | `'hann'` |
| `center_freq` |  | `0.0` |



### `ssb_demod(iq, sample_rate, sideband='usb', bfo_hz=0.0)`

Demodulate single-sideband (USB or LSB). OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `sideband` |  | `'usb'` |
| `bfo_hz` |  | `0.0` |



### `ssb_modulate(message, sideband='usb')`

SSB-modulate a real message into IQ. Inverse of ssb_demod. OUR code.

Single-sideband keeps one sideband of the message's analytic signal and
suppresses the carrier and the other sideband. We build the analytic signal
(positive frequencies only) for USB; conjugate for LSB. ssb_demod recovers
the real message by selecting the matching sideband.

message:  real message.
sideband: "usb" (upper) or "lsb" (lower).

Returns complex64 IQ -- the single-sideband analytic signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `sideband` |  | `'usb'` |



### `symbol_sync(iq, samples_per_symbol, method='gardner', loop_bw=0.01, damping=0.707, diagnostics=False, csv_path=None, lock_threshold=0.05)`

Recover symbol timing: pick the best sampling instant per symbol. OUR code.

Returns the symbol-spaced samples (one complex value per recovered symbol).
With diagnostics=True, returns (symbols, LoopDiagnostics) where the error is
the timing-error-detector output and the estimate is the fractional offset.

method:
  "gardner"        : Gardner TED -- carrier-independent, needs ~2 sps.
  "early_late"     : early-late gate -- simple, intuitive.
  "mueller_muller" : Mueller & Muller -- 1 sps, decision-aided.
samples_per_symbol: nominal sps (from estimate_symbol_rate or known rate).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `method` |  | `'gardner'` |
| `loop_bw` |  | `0.01` |
| `damping` |  | `0.707` |
| `diagnostics` |  | `False` |
| `csv_path` |  | `None` |
| `lock_threshold` |  | `0.05` |



### `to_db(x, *, power=True, epsilon=1e-20)`

Convert linear values to dB.

power=True  : x is a power quantity      -> 10*log10(x)
power=False : x is an amplitude/voltage  -> 20*log10(x)
epsilon floors the input so zeros don't produce -inf.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `x` |  | *required* |
| `power` |  | `True` |
| `epsilon` |  | `1e-20` |



### `tune_to_baseband(iq, offset_hz, sample_rate)`

Bring a signal sitting at +offset_hz down to 0 Hz (DC).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `offset_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `upsample(symbols, sps)`

Insert sps-1 zeros between symbols (zero-stuffing). OUR code.

The first step of pulse shaping: place each symbol on the output grid, then
filter to spread it into a pulse. Returns a complex64 array sps times longer.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |



---

## core.demod — demodulators

Recover bits/audio/symbols from IQ.

Import: `import sdr_dsp.core.demod`

### Functions

### `am_demod(iq, dc_block=True)`

Demodulate amplitude modulation: the envelope (magnitude). OUR code.

Returns the real envelope |iq|. With dc_block, the mean (carrier DC) is
removed so the output swings around zero like audio.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `dc_block` |  | `True` |



### `bpsk_demod(iq, normalize_phase=True)`

Demodulate binary phase-shift keying (coherent-ish). OUR code.

BPSK encodes bits as 0 or pi phase. With the carrier already at baseband and
roughly phase-aligned, the sign of the real part recovers the bits. This is
a SIMPLE demod: it assumes the signal is already carrier-aligned (no Costas
loop / carrier recovery). For captures with a residual carrier offset,
correct it first (see estimate_cfo / frequency_shift) -- the library does
not auto-recover the carrier.

Returns (bits, soft) where bits is uint8 (0/1) and soft is the real-part
decision statistic (useful for confidence / plotting a constellation).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `normalize_phase` |  | `True` |



### `cw_decode(bits, samples_per_symbol)`

Decode Morse (CW) from a sliced on/off stream. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `dbpsk_demod(symbols)`

Demodulate differential BPSK. OUR code.

Differential PSK encodes bits in phase CHANGES between consecutive symbols,
not absolute phase. That's the whole point: it needs NO carrier recovery,
because a constant phase offset cancels when you compare adjacent symbols.
This makes it robust and a good fit for block processing.

Takes symbol-spaced samples (one per symbol -- use symbol_sync first if you
have oversampled data). A bit is 0 if the phase barely changed, 1 if it
flipped by ~pi. Returns (bits, soft) where soft is the real part of the
differential product (sign gives the bit, magnitude gives confidence).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `deemphasis(audio, sample_rate, tau_us=75.0)`

Single-pole de-emphasis filter for broadcast FM audio. OUR code.

Broadcast FM pre-emphasizes high frequencies before transmission; the
receiver must de-emphasize them back. A one-pole IIR does it:
    y[n] = a*x[n] + (1-a)*y[n-1],   a = dt / (tau + dt)
tau_us: time constant (75 us in the Americas/Korea, 50 us elsewhere).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `audio` |  | *required* |
| `sample_rate` |  | *required* |
| `tau_us` |  | `75.0` |



### `dqpsk_demod(symbols)`

Demodulate differential QPSK. OUR code.

The QPSK analogue of DBPSK: 2 bits per symbol encoded in the phase CHANGE
(one of four ~90-degree steps), so it also needs no carrier recovery. Takes
symbol-spaced samples; returns (bits, phase_diffs) where bits is a uint8
array (2 per symbol, MSB first) and phase_diffs are the raw differential
angles for inspection.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `dsb_sc_demod(iq, sample_rate, bfo_hz=0.0)`

Demodulate double-sideband suppressed-carrier (DSB-SC). OUR code.

DSB-SC is AM with the carrier removed -- both sidebands, no carrier spike.
With the carrier suppressed there's no envelope to follow, so recovery needs
a coherent reference. For a complex baseband capture centered on the
(suppressed) carrier, the real part IS the message (the two sidebands beat
back together). A bfo_hz shift fine-tunes if the center is slightly off.

This is the conceptual midpoint between AM (carrier present, envelope) and
SSB (one sideband): DSB-SC keeps both sidebands but drops the carrier.
Returns the real demodulated message.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `bfo_hz` |  | `0.0` |



### `dsss_despread(iq, code, samples_per_chip=1)`

Despread a DSSS signal using a KNOWN spreading code. OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `code` |  | *required* |
| `samples_per_chip` |  | `1` |



### `edges(bits)`

Indices where a 0/1 stream transitions, and the run lengths. OUR code.

Returns (transition_indices, run_lengths, run_values): the sample index of
each transition, how many samples each run lasted, and whether that run was
0 or 1. The building block for recovering symbol timing from a sliced
on/off stream.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |



### `estimate_symbol_rate(bits, sample_rate, min_run=2)`

Estimate samples-per-symbol from the run lengths in a sliced stream.

The shortest on/off run is (usually) one symbol period. OUR code. Returns
(samples_per_symbol, symbol_rate_hz).

Robustness: a single glitch sample (from noise or a demod transient at a
symbol boundary) creates a spurious 1-sample run that would fool a naive
"minimum run" estimate. So we DISCARD runs shorter than min_run, then take a
low percentile of what remains as the symbol period -- stable against a few
outliers while still finding the shortest real symbol. Raise min_run if your
capture is very noisy; lower it (to 1) only for pristine synthetic data.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `sample_rate` |  | *required* |
| `min_run` |  | `2` |



### `fhss_detect_hops(iq, sample_rate, nfft=256, overlap=0.5, center_freq=0.0)`

Detect frequency hops: the dominant frequency per time slice. OUR code.

For an FHSS signal, computes a spectrogram and reports, for each time slice,
where the energy is -- i.e. which channel the hopper is in at that moment.
This TRACKS hops you can see; it does NOT decode the data or know the hop
sequence (out of scope). Pair it with core.spectrogram to SEE the hops.

Returns (times, hop_freqs) where hop_freqs[i] is the peak frequency (Hz,
offset by center_freq) during time slice times[i].


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `nfft` |  | `256` |
| `overlap` |  | `0.5` |
| `center_freq` |  | `0.0` |



### `fm_demod(iq, deviation_hz=None, sample_rate=None)`

Demodulate frequency modulation via the phase discriminator. OUR code.

FM carries the message in instantaneous frequency, so demod IS the
instantaneous frequency (see instantaneous_frequency). Returns a real array.

If deviation_hz and sample_rate are given, the output is scaled by the peak
deviation to give roughly normalized audio; otherwise it returns raw
radians/sample.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `deviation_hz` |  | `None` |
| `sample_rate` |  | `None` |



### `fsk_demod(iq, sample_rate, threshold_hz=0.0, smooth_samples=0)`

Demodulate 2-level frequency-shift keying. OUR code.

FSK encodes bits as two frequencies (a "mark" and a "space"). Demod is the
instantaneous frequency, then a threshold: above threshold_hz -> 1, below
-> 0. With the default threshold 0, it splits on the sign of the frequency
deviation (correct when the two tones straddle the center frequency, which
is the common case after tuning to baseband).

threshold_hz: the mark/space decision frequency. Two real radios never
    share an oscillator, so a carrier frequency offset (crystal ppm --
    easily +/-10-20 kHz at 433 MHz between two SDRs) shifts BOTH tones and
    biases the fixed 0 Hz split. Pass "auto" to threshold at the
    amplitude^2-weighted mean of the instantaneous frequency instead: with
    roughly balanced mark/space time (any frame with an alternating
    preamble qualifies) that mean IS the offset, so the split self-centers.
    The weighting means silence around a burst contributes ~nothing.
    NOTE: do not use estimate_cfo for this -- it finds the strongest
    spectral tone, which for FSK is +/-deviation, not the offset.
smooth_samples: if > 1, moving-average the instantaneous frequency over
    this many samples before slicing (a cheap matched-filter stand-in;
    ~samples_per_symbol/2 is a good value). The raw per-sample frequency
    is noisy, and this is the difference between decoding and not at
    moderate SNR. Off by default -- the per-sample output stays exact.

Returns a uint8 per-sample bit stream (length len(iq)-1); feed to the
timing helpers (sample_symbols for hardware captures with unknown delay,
or estimate_symbol_rate / slice_to_symbols) to get symbols. Covers
GFSK/MSK well enough for typical ISM-band sensors and pagers.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `threshold_hz` |  | `0.0` |
| `smooth_samples` |  | `0` |



### `fsk_demod_nlevel(iq, sample_rate, n_levels=4, thresholds=None)`

Demodulate N-level FSK (4-FSK, etc.) and CPFSK. OUR code.

Generalizes 2-FSK: instead of a single 0-threshold on instantaneous
frequency, it slices the frequency into n_levels bands. Used by 4-FSK
(DMR, P25, some pagers). CPFSK recovers the same way -- the continuous
phase is a transmit-side property; the receiver still reads instantaneous
frequency.

thresholds: explicit frequency band centers (Hz). If None, the levels are
spread uniformly across the observed frequency range -- fine for a clean
capture; pass measured centers for real signals. Returns per-sample symbol
indices 0..n_levels-1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `n_levels` |  | `4` |
| `thresholds` |  | `None` |



### `instantaneous_frequency(iq, sample_rate=None)`

The instantaneous frequency of a complex signal. OUR code.

Computed by the phase discriminator: the phase change between consecutive
samples, angle(x[n] * conj(x[n-1])). This is THE primitive under FM and FSK
demodulation, exposed so both build on it (and so you can analyze frequency
directly -- Doppler, drift, chirps).

Returns radians/sample, or Hz if sample_rate is given. Output length is
len(iq) - 1 (one difference per adjacent pair).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | `None` |



### `instantaneous_phase(iq, unwrap=True)`

The phase angle of each complex sample. OUR code.

Returns the per-sample phase in radians. With unwrap=True the 2*pi jumps are
removed so the phase is continuous (useful for seeing accumulated phase /
measuring frequency as its slope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `unwrap` |  | `True` |



### `nask_slice(envelope, n_levels=4, levels=None)`

Slice an amplitude envelope into N levels (M-ASK). OUR code.

Generalizes 2-level OOK to N amplitude levels (4-ASK, 8-ASK). Returns a
per-sample symbol index in 0..n_levels-1.

levels: explicit amplitude thresholds/centers. If None, the levels are
spread uniformly from the envelope's min to its max -- a reasonable default
for a clean capture, but YOU can pass measured levels for real signals where
the spacing isn't uniform.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `n_levels` |  | `4` |
| `levels` |  | `None` |



### `ook_envelope(iq)`

On-off-keying / ASK front end: the magnitude envelope. OUR code.

Returns |iq| (no DC block -- OOK threshold detection wants the absolute
level). Feed to ``ook_slice`` to recover bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |



### `ook_slice(envelope, threshold=None)`

Threshold an OOK envelope into a 0/1 stream. OUR code.

threshold: level above which a sample is '1'. If None, uses the midpoint
between the envelope's min and max (a simple, robust default for a clean
capture). Returns a uint8 array of 0/1.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `envelope` |  | *required* |
| `threshold` |  | `None` |



### `psk8_demod(symbols)`

Demodulate 8-PSK from recovered symbols. OUR code.

8-PSK carries 3 bits/symbol in eight phase points (45-degree spacing).
COHERENT: assumes carrier-aligned, symbol-timed input (recover first, as in
qpsk_demod). Higher-order PSK demands more SNR -- the eight points are
closer together -- so on an 8-bit SDR like the HackRF this needs a strong,
clean signal. Returns (bits, sector) where bits is uint8 (3 per symbol) and
sector is the chosen 0..7 phase sector.

Honest note: at 45-degree spacing, a small residual carrier error rotates
points across decision boundaries, so good carrier recovery matters more
here than for QPSK.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |



### `qam16_demod(symbols, normalize=True)`

Demodulate QAM-16 from recovered symbols. OUR code.

COHERENT and amplitude-sensitive: assumes carrier-aligned, symbol-timed
input AND a known amplitude scale (QAM decisions depend on absolute level,
unlike PSK). Recover first, then normalize:

    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sym(corr, sps)
    bits, pts = qam16_demod(syms)   # normalize=True scales by RMS

The 16 points sit on a 4x4 grid at I,Q in {-3,-1,+1,+3} (scaled). Each axis
carries 2 Gray-coded bits. Returns (bits, points) -- 4 bits/symbol and the
chosen grid points (for plotting the constellation).

normalize=True scales the input so its RMS matches the standard grid; this
is the one place QAM needs an amplitude assumption, and it's explicit. Pass
normalize=False if you've already scaled the signal yourself.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `normalize` |  | `True` |



### `qpsk_demod(symbols, gray=True)`

Demodulate QPSK from recovered symbols. OUR code.

QPSK carries 2 bits/symbol in four phase points (the four quadrants of the
complex plane). This is a COHERENT demod: it assumes the symbols are already
carrier-aligned and symbol-timed. For a raw capture, recover first:

    from sdr_dsp.core import carrier_recovery, symbol_sync
    corr = carrier_recovery(iq, method="costas", order=4)
    syms = symbol_sync(corr, sps)
    bits, _ = qpsk_demod(syms)

The library does NOT auto-recover -- you compose the recovery you want, so
nothing is hidden. Returns (bits, decisions) where bits is uint8 (2 per
symbol) and decisions are the constellation points chosen (for plotting).

gray=True uses Gray coding (adjacent quadrants differ by one bit), the
standard choice that minimizes bit errors.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `gray` |  | `True` |



### `sample_symbols(bits, samples_per_symbol, active=None)`

Decimate an over-sampled 0/1 stream to symbols at the RIGHT phase. OUR code.

A per-sample bit stream (from fsk_demod / ook_slice) carries each symbol
samples_per_symbol times, but a real capture arrives with an arbitrary
delay -- so a fixed stride like bits[sps//2::sps] samples at an arbitrary
point in each symbol, sometimes right on the transitions. This estimates
the symbol phase FROM the stream itself: transitions can only occur at
symbol boundaries, so the circular mean of (transition_index mod sps) is
the boundary phase, and boundary + sps/2 is the symbol center.

Contrast with slice_to_symbols, which is run-length based: robust to
unknown phase but a single glitch sample inserts/deletes a bit and shifts
everything after it. This keeps the fixed-stride robustness (a glitch
corrupts one bit, not the alignment) while removing the phase assumption.
Use this one on hardware captures.

bits:               per-sample 0/1 stream (uint8).
samples_per_symbol: the (integer) oversampling factor.
active:             optional boolean mask, same length as bits (or 1 less,
                    e.g. from a len-1 instantaneous-frequency chain).
                    Transitions outside the mask are ignored when
                    estimating the phase -- pass envelope > threshold so
                    garbage flicker in silence between bursts doesn't
                    pollute the estimate.

Returns a uint8 array with one entry per symbol period.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `active` |  | `None` |



### `slice_to_symbols(bits, samples_per_symbol)`

Collapse an over-sampled 0/1 stream into one bit per symbol. OUR code.

Given the samples-per-symbol, walk each run and emit its value repeated
round(run_length / spb) times. Returns a uint8 array of symbol bits.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |



### `ssb_demod(iq, sample_rate, sideband='usb', bfo_hz=0.0)`

Demodulate single-sideband (USB or LSB). OUR code.

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


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `sideband` |  | `'usb'` |
| `bfo_hz` |  | `0.0` |



---

## core.modulate — modulators

Inverse of demod: bits/symbols -> IQ.

Import: `import sdr_dsp.core.modulate`

### Functions

### `am_modulate(message, modulation_index=0.5)`

AM-modulate a real message into IQ. Inverse of am_demod. OUR code.

Amplitude modulation rides the message on the carrier envelope:
(1 + k*m) carrier. am_demod recovers it with an envelope detector. Keep
modulation_index <= 1 to avoid over-modulation (envelope going negative,
which the envelope detector can't undo).

message:          real message, roughly [-1, 1].
modulation_index: depth k of modulation (0..1). >1 over-modulates.

Returns complex64 IQ with the message in its magnitude.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `modulation_index` |  | `0.5` |



### `bpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, pad_symbols=0)`

BPSK: bit 0 -> +1, bit 1 -> -1 (phase 0 or pi). Inverse of bpsk_demod.

Carries one bit per symbol in the carrier phase. With pulse_shaping=False the
symbols are held rectangular (sps samples each); with pulse_shaping=True they
are RRC-shaped for a bandlimited spectrum (use a matched RRC filter on
receive). bpsk_demod recovers bits from the real part's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per symbol (1 = one sample per symbol).
pulse_shaping:      RRC-shape the symbols if True.

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `pad_symbols` |  | `0` |



### `fm_modulate(message, deviation_hz, sample_rate)`

FM-modulate a real message into IQ. Inverse of fm_demod. OUR code.

Frequency modulation encodes the message in the carrier's instantaneous
frequency: the IQ phase is the running integral of the message, scaled by
the deviation. fm_demod recovers it by differentiating the phase.

message:      real-valued message, expected roughly in [-1, 1].
deviation_hz: peak frequency deviation (must match the demod's deviation_hz).
sample_rate:  sample rate in Hz.

Returns unit-magnitude complex64 IQ (FM is constant-envelope).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |



### `fsk_modulate(bits, samples_per_symbol, deviation_hz, sample_rate, pad_symbols=0)`

Binary FSK: bit selects one of two frequencies. Inverse of fsk_demod.

Bit 1 -> +deviation_hz, bit 0 -> -deviation_hz, encoded as a continuous-phase
frequency shift (CPFSK -- the phase is integrated so there are no jumps,
which keeps the spectrum clean). fsk_demod recovers bits from the
instantaneous frequency's sign.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
deviation_hz:       frequency shift magnitude (match the demod's threshold).
sample_rate:        sample rate in Hz.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 IQ, unit magnitude within the burst (FSK is
constant-envelope); the pad regions are silence.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `deviation_hz` |  | *required* |
| `sample_rate` |  | *required* |
| `pad_symbols` |  | `0` |



### `ook_modulate(bits, samples_per_symbol, high=1.0, low=0.0, pad_symbols=0)`

On-off keying: bit 1 -> carrier on, bit 0 -> off. Inverse of ook_slice.

The simplest digital modulation. Each bit becomes samples_per_symbol samples
at amplitude `high` (for 1) or `low` (for 0). ook_envelope + ook_slice
recover the bits by thresholding the magnitude.

bits:               sequence of 0/1.
samples_per_symbol: samples per bit.
pad_symbols:        symbols of silence appended to EACH end of the burst
                    (default 0: exact legacy output). The first/last
                    symbols of an unpadded burst sit flush against the
                    buffer edges, where instantaneous-frequency and filter
                    edge effects corrupt them -- a loopback at zero delay
                    hides this; any real capture (arbitrary delay, burst
                    embedded in noise) exposes it. Use >= 4 for anything
                    leaving a same-buffer loopback.

Returns complex64 baseband (real-valued amplitude, zero phase).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | *required* |
| `high` |  | `1.0` |
| `low` |  | `0.0` |
| `pad_symbols` |  | `0` |



### `pulse_shape(symbols, sps, span_symbols=8, beta=0.35)`

Upsample symbols and shape them with an RRC pulse. OUR code.

The standard transmit-side digital chain: take complex symbols (e.g. PSK
constellation points), upsample by sps, and convolve with a root-raised-cosine
pulse so the result is bandlimited and ISI-free when matched-filtered on
receive. Returns the shaped complex64 baseband signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `qpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False, beta=0.35, span_symbols=8, gray=True, pad_symbols=0)`

QPSK: 2 bits per symbol, Gray-coded quadrants. Inverse of qpsk_demod.

Pairs of bits map to the four points (1+1j, -1+1j, -1-1j, 1-1j)/sqrt(2) via
the same Gray convention qpsk_demod uses, so the round-trip is exact. An odd
trailing bit is dropped (QPSK consumes bits in pairs).

bits:               sequence of 0/1 (length should be even).
samples_per_symbol: samples per symbol.
pulse_shaping:      RRC-shape the symbols if True.
gray:               use Gray coding (match the demod).

Returns complex64 baseband.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `bits` |  | *required* |
| `samples_per_symbol` |  | `1` |
| `pulse_shaping` |  | `False` |
| `beta` |  | `0.35` |
| `span_symbols` |  | `8` |
| `gray` |  | `True` |
| `pad_symbols` |  | `0` |



### `rrc_taps(sps, span_symbols=8, beta=0.35)`

Root-raised-cosine filter taps. OUR code.

sps:          samples per symbol (the upsampling factor).
span_symbols: how many symbols wide the pulse is (longer = sharper spectrum).
beta:         roll-off factor in [0, 1]; larger = more bandwidth, gentler.

Returns the normalized tap array. Used on BOTH ends: shape on transmit,
matched-filter with the same taps on receive (root * root = raised cosine,
the zero-ISI pulse).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `sps` |  | *required* |
| `span_symbols` |  | `8` |
| `beta` |  | `0.35` |



### `ssb_modulate(message, sideband='usb')`

SSB-modulate a real message into IQ. Inverse of ssb_demod. OUR code.

Single-sideband keeps one sideband of the message's analytic signal and
suppresses the carrier and the other sideband. We build the analytic signal
(positive frequencies only) for USB; conjugate for LSB. ssb_demod recovers
the real message by selecting the matching sideband.

message:  real message.
sideband: "usb" (upper) or "lsb" (lower).

Returns complex64 IQ -- the single-sideband analytic signal.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `message` |  | *required* |
| `sideband` |  | `'usb'` |



### `upsample(symbols, sps)`

Insert sps-1 zeros between symbols (zero-stuffing). OUR code.

The first step of pulse shaping: place each symbol on the output grid, then
filter to spread it into a pulse. Returns a complex64 array sps times longer.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `symbols` |  | *required* |
| `sps` |  | *required* |



---

## sources — receive seam

IQSource protocol and concrete sources.

Import: `import sdr_dsp.sources`

### Classes

### class `ArraySource`

The simplest source: wrap an in-memory complex64 array.

Useful for tests, synthetic signals, and feeding already-loaded data into
the same pipeline code a file or device would drive.


**Constructor:** `ArraySource(self, iq: 'np.ndarray', sample_rate: 'float', center_freq: 'float' = 0.0, block_size: 'int' = 65536)`

| Parameter | Type | Default |
|---|---|---|
| `iq` | `np.ndarray` | *required* |
| `sample_rate` | `float` | *required* |
| `center_freq` | `float` | `0.0` |
| `block_size` | `int` | `65536` |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

_No docstring._


#### `read(self, n_samples: 'int') -> 'np.ndarray'`

Return the whole array (or the first n_samples).

| Parameter | Type | Default |
|---|---|---|
| `n_samples` | `int` | *required* |




### class `FileSource`

An IQSource backed by a SigMF recording on disk.

sample_rate and center_freq come from the sidecar (read cheaply, without
loading the samples). blocks() streams the file in block_size chunks read
incrementally from disk; .iq loads the whole recording lazily if you ask.

Args:
    path:           the .iq / .sigmf-data / .sigmf-meta path.
    block_size:     samples per block from blocks().
    count:          limit reading to this many samples (None = whole file).
    offset_samples: skip this many samples from the start.


**Constructor:** `FileSource(self, path, block_size=65536, count=None, offset_samples=0)`

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `block_size` |  | `65536` |
| `count` |  | `None` |
| `offset_samples` |  | `0` |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

Yield the recording in block_size chunks, read from disk on demand.

Each block is read with its own seek+read, so memory use stays at one
block regardless of file size. If .iq has already been loaded (small-file
path), slice that instead of re-reading.


#### `read(self, n_samples: 'int') -> 'np.ndarray'`

Read up to n_samples from the start (respecting offset). Streams from
disk without loading the whole file.

| Parameter | Type | Default |
|---|---|---|
| `n_samples` | `int` | *required* |




### class `IQSource`

Anything that can provide IQ samples plus the metadata to interpret them.

Attributes:
    sample_rate: samples per second (Hz).
    center_freq: RF center frequency the samples were captured at (Hz).

Implementations provide ``blocks()`` to stream decoded complex64 arrays.
A bounded source may also support ``read(n)``; unbounded/live sources need
only ``blocks()``.


**Constructor:** `IQSource(self, *args, **kwargs)`

| Parameter | Type | Default |
|---|---|---|
| `args` (*args) |  | *required* |
| `kwargs` (**kwargs) |  | *required* |


**Methods:**

#### `blocks(self) -> 'Iterator[np.ndarray]'`

Yield complex64 blocks until the source is exhausted or stopped.



**Attributes:**

- `sample_rate`: `float`
- `center_freq`: `float`



---

## sinks — output & transmit seam

TXSink protocol, file and plot sinks.

Import: `import sdr_dsp.sinks`

### Classes

### class `LoopbackSink`

A TXSink that 'transmits' into an in-memory buffer instead of a radio.

The transmit-side counterpart of ArraySource: it satisfies the TXSink
protocol but keeps every transmitted sample in `.buffer`, so the full stack
(ARQ -> modulate -> sink) can be wired and tested without hardware. Pair it
with an ArraySource on the receive side to close a software loop, optionally
through the simulated channel.

This is how the live-driver wiring is verified in software: the protocol
really drives a sink, the sink really receives the IQ -- only the antenna is
missing.


**Constructor:** `LoopbackSink(self, sample_rate, center_freq=0.0)`

| Parameter | Type | Default |
|---|---|---|
| `sample_rate` |  | *required* |
| `center_freq` |  | `0.0` |


**Methods:**

#### `clear(self)`

Reset the buffer and counter.


#### `transmit(self, iq)`

Append the IQ to the in-memory buffer (no radio).

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |




### class `TXSink`

Anything that can transmit complex64 IQ at a center frequency.

Attributes:
    sample_rate: samples per second (Hz) the IQ is sampled at.
    center_freq: RF center frequency to transmit at (Hz).

Implementations provide ``transmit(iq)`` to send one buffer of complex64
samples. A device adapter handles gain, timing, and the half-duplex
TX/RX turnaround; the library only hands it baseband IQ.


**Constructor:** `TXSink(self, *args, **kwargs)`

| Parameter | Type | Default |
|---|---|---|
| `args` (*args) |  | *required* |
| `kwargs` (**kwargs) |  | *required* |


**Methods:**

#### `transmit(self, iq: 'np.ndarray') -> 'None'`

Transmit one buffer of complex64 IQ samples.

| Parameter | Type | Default |
|---|---|---|
| `iq` | `np.ndarray` | *required* |



**Attributes:**

- `sample_rate`: `float`
- `center_freq`: `float`



### Functions

### `plot_spectrogram(iq, sample_rate, center_freq=0.0, nfft=1024, overlap=0.5, title=None, show=True)`

Plot a spectrogram waterfall. Returns (fig, ax).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `center_freq` |  | `0.0` |
| `nfft` |  | `1024` |
| `overlap` |  | `0.5` |
| `title` |  | `None` |
| `show` |  | `True` |



### `plot_spectrum(iq, sample_rate, center_freq=0.0, nfft=2048, title=None, show=True)`

Plot the PSD of a signal. Returns (fig, ax).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `center_freq` |  | `0.0` |
| `nfft` |  | `2048` |
| `title` |  | `None` |
| `show` |  | `True` |



### `write_iq(path, iq, sample_rate, center_freq=0.0, **extra_global)`

Save complex64 IQ + a SigMF sidecar (cf32_le). Returns (data, meta) paths.

Pass-through to io.sigmf.save_iq; kept here so sinks are a uniform place to
look for "where results go". extra_global lands in the SigMF global object.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `center_freq` |  | `0.0` |
| `extra_global` (**kwargs) |  | *required* |



### `write_wav(path, audio, sample_rate, normalize=True, headroom=0.9)`

Write a real audio array to a mono 16-bit WAV.

audio:      real-valued samples (e.g. demodulated output).
normalize:  scale to use the int16 range (with a little headroom). If
            False, audio is assumed already in [-1, 1].
headroom:   peak level when normalizing (0.9 = -1 dBFS-ish, avoids clipping).
Returns the path written.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `audio` |  | *required* |
| `sample_rate` |  | *required* |
| `normalize` |  | `True` |
| `headroom` |  | `0.9` |



---

## io — file formats

SigMF load/save, annotations, metadata.

Import: `import sdr_dsp.io`

### Classes

### class `Annotation`

A labeled region of a recording, round-tripped to/from SigMF.

Maps to a SigMF annotation object. The two required fields locate the region
in time (sample_start + sample_count); the optional frequency edges locate it
in frequency, and label/comment/extra carry the human meaning.

Fields:
    sample_start:     first sample of the region.
    sample_count:     length of the region in samples.
    freq_lower_edge:  lower frequency bound (Hz), or None.
    freq_upper_edge:  upper frequency bound (Hz), or None.
    label:            short label (SigMF core:label), e.g. "key fob burst".
    comment:          longer note (SigMF core:comment).
    extra:            any additional namespaced keys to round-trip verbatim.


**Constructor:** `Annotation(self, sample_start: 'int', sample_count: 'int', freq_lower_edge: 'float | None' = None, freq_upper_edge: 'float | None' = None, label: 'str | None' = None, comment: 'str | None' = None, extra: 'dict' = <factory>) -> None`

| Parameter | Type | Default |
|---|---|---|
| `sample_start` | `int` | *required* |
| `sample_count` | `int` | *required* |
| `freq_lower_edge` | `float | None` | `None` |
| `freq_upper_edge` | `float | None` | `None` |
| `label` | `str | None` | `None` |
| `comment` | `str | None` | `None` |
| `extra` | `dict` | `<factory>` |


**Methods:**

#### `from_sigmf(d: 'dict') -> "'Annotation'"`

Parse a SigMF annotation dict back into an Annotation.

Recognized core: keys map to fields; anything else is preserved in
`extra` so a save -> load round-trip is lossless.

| Parameter | Type | Default |
|---|---|---|
| `d` | `dict` | *required* |


#### `time_span(self, sample_rate: 'float') -> 'tuple[float, float]'`

Convenience: (start_seconds, end_seconds) for this region.

| Parameter | Type | Default |
|---|---|---|
| `sample_rate` | `float` | *required* |


#### `to_sigmf(self) -> 'dict'`

Serialize to a SigMF annotation dict (core:-namespaced keys).



**Attributes:**

- `sample_start`: `int`
- `sample_count`: `int`
- `freq_lower_edge`: `float | None`
- `freq_upper_edge`: `float | None`
- `label`: `str | None`
- `comment`: `str | None`
- `extra`: `dict`



### Functions

### `bursts_to_annotations(spans, label=None, freq_lower_edge=None, freq_upper_edge=None)`

Convert find_bursts() output into Annotation objects. The detect->label
step in one call.

spans: a list of (start, stop) sample-index pairs (what find_bursts returns).
label: applied to every burst, optionally with an index suffix if it
       contains "{i}" (e.g. "burst {i}" -> "burst 0", "burst 1", ...).
freq_lower_edge / freq_upper_edge: optional frequency bounds applied to all.

Returns a list of Annotation ready to pass to save_iq(annotations=...).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `spans` |  | *required* |
| `label` |  | `None` |
| `freq_lower_edge` |  | `None` |
| `freq_upper_edge` |  | `None` |



### `iq_info(path)`

Inspect a recording WITHOUT loading the IQ data into memory.

Returns a dict with: meta, datatype, np_dtype, is_complex, itemsize,
bytes_per_sample, total_samples, sample_rate, center_freq. This is what lets
a streaming reader know how big the file is and how to seek into it without
reading the samples. Cheap: it stats the data file and parses the sidecar.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



### `load_iq(path, count=None, offset_samples=0)`

Load a SigMF recording into complex64, using its sidecar to interpret.

path:           the .iq/.sigmf-data or .sigmf-meta path.
count:          max samples to read (None = all).
offset_samples: skip this many complex samples from the start.

Returns (iq_complex64, meta_dict). The datatype is read from the sidecar so
hackrfpy's ci8 captures decode correctly; everything is normalized to
complex64 in roughly [-1, 1).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `count` |  | `None` |
| `offset_samples` |  | `0` |



### `read_annotations(path) -> 'list'`

Read the annotations from a recording's sidecar as Annotation objects.

Accepts any path form (.iq / .sigmf-data / .sigmf-meta). Returns a list of
Annotation, sorted by sample_start. Empty list if there are none.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |


**Returns:** `list`



### `read_meta(path)`

Read a .sigmf-meta sidecar into a dict. Accepts the meta or data path.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



### `save_iq(path, iq, sample_rate, center_freq=0.0, extra_global=None, annotations=None)`

Write complex64 IQ + a SigMF sidecar as cf32_le (processed-output type).

Writes <stem>.sigmf-data and <stem>.sigmf-meta. cf32_le is SigMF's canonical
complex-float type; we use it for processed output (vs hackrfpy's raw ci8).

annotations: an optional list of Annotation objects (or raw SigMF annotation
dicts) to record labeled regions of the capture. They round-trip: load_iq /
read_annotations read them back as Annotation objects.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |
| `iq` |  | *required* |
| `sample_rate` |  | *required* |
| `center_freq` |  | `0.0` |
| `extra_global` |  | `None` |
| `annotations` |  | `None` |



---

## stream — orchestration

Pipeline and block-streaming.

Import: `import sdr_dsp.stream`

### Classes

### class `Pipeline`

A source + an ordered chain of block->block stages (and taps).

Build declaratively and run:

    pipe = (Pipeline(source)
            .add(lambda b: fir_apply(b, taps), "filter")
            .add(lambda b: fm_demod(b, 75000, fs), "demod")
            .tap(lambda b: meter.update(b)))      # live peek, flow unchanged
    audio = pipe.run()                            # or run(sink=write_audio)

Stages transform the block; taps observe it and return nothing (the original
block continues). Order matters and is preserved.


**Constructor:** `Pipeline(self, source)`

| Parameter | Type | Default |
|---|---|---|
| `source` |  | *required* |


**Methods:**

#### `add(self, fn, name=None)`

Append a transforming stage (block -> block).

| Parameter | Type | Default |
|---|---|---|
| `fn` |  | *required* |
| `name` |  | `None` |


#### `describe(self)`

Return the chain as inspectable text (the pipeline is data).


#### `process_block(self, block)`

Thread a single block through every stage. Useful for testing and
for driving the pipeline from an external loop (e.g. a GUI timer).

| Parameter | Type | Default |
|---|---|---|
| `block` |  | *required* |


#### `run(self, sink=None, profile=False, max_blocks=None)`

Pull blocks from the source, process each, deliver to sink.

sink:       callable(result_block) -> None. If None, results are
            collected and returned as a list.
profile:    if True, also return PipelineStats (per-stage timing).
max_blocks: stop after this many blocks (useful for live sources).

Returns the result list (or None if a sink was given); if profile,
returns (results_or_None, PipelineStats).

| Parameter | Type | Default |
|---|---|---|
| `sink` |  | `None` |
| `profile` |  | `False` |
| `max_blocks` |  | `None` |


#### `stream(self, max_blocks=None) -> 'Iterator[np.ndarray]'`

Run as a generator, yielding each processed block lazily.

This is the bridge to the generator-chain style: a Pipeline can be
consumed lazily, so it composes with other generators and stays
memory-friendly for long/continuous streams.

| Parameter | Type | Default |
|---|---|---|
| `max_blocks` |  | `None` |


#### `tap(self, fn, name=None)`

Append an observing stage (block -> ignored). Flow is unchanged.

A tap is how live display attaches: fn receives the current block and
does whatever it likes (update a plot, accumulate a message) without
affecting what the next stage sees.

| Parameter | Type | Default |
|---|---|---|
| `fn` |  | *required* |
| `name` |  | `None` |




### class `PipelineStats`

Per-stage timing/throughput, collected when run(profile=True).


**Constructor:** `PipelineStats(self, blocks: 'int' = 0, samples_in: 'int' = 0, per_stage_seconds: 'dict' = <factory>) -> None`

| Parameter | Type | Default |
|---|---|---|
| `blocks` | `int` | `0` |
| `samples_in` | `int` | `0` |
| `per_stage_seconds` | `dict` | `<factory>` |


**Attributes:**

- `blocks`: `int`
- `samples_in`: `int`
- `per_stage_seconds`: `dict`



### class `Stage`

One step in a pipeline: a named block->block operation, or a tap.


**Constructor:** `Stage(self, name: 'str', fn: 'Callable', is_tap: 'bool' = False) -> None`

| Parameter | Type | Default |
|---|---|---|
| `name` | `str` | *required* |
| `fn` | `Callable` | *required* |
| `is_tap` | `bool` | `False` |


**Attributes:**

- `name`: `str`
- `fn`: `Callable`
- `is_tap`: `bool`



---

## link — ARQ protocol

Reliable acknowledged messaging.

Import: `import sdr_dsp.link`

### Classes

### class `ARQ`

Event-driven ARQ engine. window_size=1 is stop-and-wait; N is windowed.

Args:
    window_size:  max frames in flight (1 = stop-and-wait).
    timeout_ticks: ticks to wait for an ACK before retransmitting.
    max_retries:  retransmit attempts before giving up on a frame.
    seq_mod:      sequence-number modulus (must be >= 2*window_size so the
                  window never aliases; defaults to 2*window_size).
    cumulative_ack: if True, the receiver acknowledges the highest
                  contiguously-received seq (one ACK confirms everything up
                  to it) instead of each frame individually. Correct under
                  loss; the traffic saving only appears with batched arrival.
                  Default False (Selective Repeat, per-frame ACK).


**Constructor:** `ARQ(self, window_size=1, timeout_ticks=10, max_retries=5, seq_mod=None, cumulative_ack=False)`

| Parameter | Type | Default |
|---|---|---|
| `window_size` |  | `1` |
| `timeout_ticks` |  | `10` |
| `max_retries` |  | `5` |
| `seq_mod` |  | `None` |
| `cumulative_ack` |  | `False` |


**Methods:**

#### `on_event(self, event)`

Feed one event to the machine. Updates state, may queue intentions.

| Parameter | Type | Default |
|---|---|---|
| `event` |  | *required* |


#### `poll(self)`

Return and clear the pending intentions emitted so far.


#### `send(self, data)`

Queue a message for reliable delivery (an application event).

| Parameter | Type | Default |
|---|---|---|
| `data` |  | *required* |




### class `EventLog`

A structured, replayable record of a protocol exchange.

Each record is a flat dict with named fields (tick, station, dir, type, seq,
crc_ok, payload_hex, note) -- deliberately tool-friendly, so the JSON can be
transformed into pcap/Wireshark, dropped into pandas, etc., rather than being
an insular format. Replay only needs the inbound events; the full log
(including emitted intentions) is kept for inspection.


**Constructor:** `EventLog(self, records: 'list' = <factory>) -> None`

| Parameter | Type | Default |
|---|---|---|
| `records` | `list` | `<factory>` |


**Methods:**

#### `load(path)`

Load a log saved by save().

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |


#### `record(self, tick, station, direction, payload=None, crc_ok=None, note='')`

Append one structured event record. payload is the protocol payload
bytes (decoded into type/seq for readability).

| Parameter | Type | Default |
|---|---|---|
| `tick` |  | *required* |
| `station` |  | *required* |
| `direction` |  | *required* |
| `payload` |  | `None` |
| `crc_ok` |  | `None` |
| `note` |  | `''` |


#### `save(self, path)`

Write the log as indented JSON.

| Parameter | Type | Default |
|---|---|---|
| `path` |  | *required* |



**Attributes:**

- `records`: `list`



### class `LiveLink`

Drive one ARQ engine over real radio: a TXSink to transmit, an IQSource
to receive. The Phase E seam.

This is the structural bridge from the pure protocol to hardware. It turns
the engine's ("tx", payload) intentions into modulate -> sink.transmit(), and
turns received IQ into ("rx", payload, crc_ok) events fed back to the engine.
It does NOT implement a radio -- it drives whatever TXSink/IQSource you give
it, so a LoopbackSink tests the wiring and a real device adapter (Phase E on
the bench) does the real thing.

HONESTY: half-duplex turnaround timing, real device latency, and whether the
tick-based timeouts map onto hardware are exactly the things that can only be
settled on a bench. This class makes the wiring correct and testable; it does
not and cannot prove over-the-air behavior in software.

Args:
    engine:     an ARQ instance.
    sink:       a TXSink (transmit side).
    modulate:   payload_bytes -> iq  (build_frame + a modulator).
    demodulate: iq -> [frames]       (a demod + find_frames), used on RX.
    log:        optional EventLog to record the exchange (for later replay).
    station:    label for the log ("A"/"B").
    carry_samples: how many trailing IQ samples to carry from each
        on_rx_iq() call into the next. A streaming receiver delivers
        blocks at arbitrary boundaries, and a frame split across two
        blocks is invisible to both halves' demod -- silently lost. Set
        this to at least one full frame's length in samples (frame bits x
        samples_per_symbol, plus padding) and the overlap guarantees every
        frame lands whole in some window. Default 0 preserves the old
        per-block behavior, which is only correct when the caller
        delivers burst-aligned segments (e.g. via find_bursts, or the
        whole-buffer LoopbackSink loop). A frame that falls entirely
        inside the overlap can be found twice; that is safe here -- ARQ
        duplicate detection (sequence numbers) exists for exactly this.


**Constructor:** `LiveLink(self, engine, sink, modulate, demodulate, log=None, station='A', carry_samples=0)`

| Parameter | Type | Default |
|---|---|---|
| `engine` |  | *required* |
| `sink` |  | *required* |
| `modulate` |  | *required* |
| `demodulate` |  | *required* |
| `log` |  | `None` |
| `station` |  | `'A'` |
| `carry_samples` |  | `0` |


**Methods:**

#### `on_rx_iq(self, iq)`

Feed received IQ: demodulate, find frames, deliver each as an rx
event. Returns the app outputs produced.

With carry_samples > 0, the tail of the previous call's IQ is
prepended so frames straddling a block boundary are still found.

| Parameter | Type | Default |
|---|---|---|
| `iq` |  | *required* |


#### `pump(self)`

Flush the engine's pending intentions: transmit any tx, return the
app-level outputs (deliver/done/failed). Call after feeding events.


#### `send(self, data)`

Queue an application message and flush.

| Parameter | Type | Default |
|---|---|---|
| `data` |  | *required* |


#### `tick(self)`

Advance one logical tick (drives timeouts/retransmits).




### Functions

### `make_channel_transport(modulate, demodulate, channel, drop_predicate=None)`

Build a transport that runs a payload through modulate -> channel ->
demodulate -> framing, returning (crc_ok, recovered_payload | None).

modulate:   payload_bytes -> iq      (build_frame + a modulator)
demodulate: iq -> [frames]           (a demod + find_frames)
channel:    iq -> iq                 (apply_channel, partial-applied)
drop_predicate: optional callable() -> bool to force a drop (for demos).

This is the bridge from the abstract protocol to the real DSP chain. It's how
the sim driver exercises the protocol over the Phase C channel.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `modulate` |  | *required* |
| `demodulate` |  | *required* |
| `channel` |  | *required* |
| `drop_predicate` |  | `None` |



### `pack_payload(frame_type, seq, data=b'')`

Build a protocol payload: [type][seq][data]. Returns bytes.

This goes inside build_frame() as the payload. seq is taken mod 256.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `frame_type` |  | *required* |
| `seq` |  | *required* |
| `data` |  | `b''` |



### `perfect_transport(payload)`

A lossless transport: every frame arrives, CRC ok.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `payload` |  | *required* |



### `replay(log, engine, station)`

Feed a recorded log's inbound events for `station` to a fresh engine and
return the intentions it produces -- reproducing the original run with no
transmission.

Only the rx events (and ticks) drive the engine; tx records in the log were
the *output* of the original run and are recomputed here. This is the
zero-TX demo path: load a saved exchange and watch the protocol re-derive it.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `log` |  | *required* |
| `engine` |  | *required* |
| `station` |  | *required* |



### `run_link(messages, window_size=1, transport=None, timeout_ticks=10, max_retries=5, max_ticks=500)`

Send a list of messages from station A to station B over a sim link.

The concrete, readable entry point: hides the event plumbing for the common
"run it in simulation" case. Returns (received_by_b, log).

    received, log = run_link([b"hello", b"world"])
    log.save("exchange.json")          # then replay later with zero TX

transport defaults to a perfect link; pass a channel transport (see
make_channel_transport) to run over the Phase C DSP chain.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `messages` |  | *required* |
| `window_size` |  | `1` |
| `transport` |  | `None` |
| `timeout_ticks` |  | `10` |
| `max_retries` |  | `5` |
| `max_ticks` |  | `500` |



### `run_sim(station_a, station_b, max_ticks=200, transport=None, log=None)`

Drive two ARQ engines until both are idle or max_ticks is reached.

station_a, station_b: ARQ engines (already given their send() messages).
transport: a transport callable (default: perfect_transport).
log: an EventLog to record into (created if None).

Returns (delivered_to_a, delivered_to_b, log) where delivered_* are the
payloads each station's app received (in order).


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `station_a` |  | *required* |
| `station_b` |  | *required* |
| `max_ticks` |  | `200` |
| `transport` |  | `None` |
| `log` |  | `None` |



### `type_name(t)`

Human name for a frame type byte.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `t` |  | *required* |



### `unpack_payload(payload)`

Parse a protocol payload. Returns (frame_type, seq, data).

Raises ValueError if too short to hold the 2-byte header.


**Parameters:**

| Parameter | Type | Default |
|---|---|---|
| `payload` |  | *required* |



---