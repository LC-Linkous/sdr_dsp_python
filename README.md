# sdr_dsp_python

A personal, fully-functional DSP library for software-defined radio, written in Python. It consumes IQ samples from a file or a live SDR and turns them into meaning: filtered channels, spectra, demodulated audio, decoded signals. It can also go the other way — framing a message, modulating it into IQ, and handing it to a transmit sink — so a message can make the whole round trip in software.

This is part of a personal ecosystem of libraries for experimental and educational purposes, so there are already tools out there that do some of this.

`sdr_dsp` is a library, not a framework. When using, you have to import functions and classes and orchestrate the pipeline yourself in plain Python. There is no runtime, scheduler, flowgraph engine, or GUI. It is not a GNU Radio competitor or replacement. GNU Radio remains the right tool for large real-time flowgraphs; `sdr_dsp` is for direct, scriptable DSP in Python.

NOTE: to make the documentation cover all parts of this library, AI is being used to summarize the codebase and the development notes. The largest improvement has been the spelling correction, and format updates to make the development easier to follow. All mistakes are human, and will likely take a revision or two to fix experimentally.

## Design

**The radio DSP is the library's own code.** Filtering, demodulation, modulation, resampling, mixing, and measurement are all implemented here. scipy is a design-time tool, not a runtime crutch: `scipy.signal` only *designs* filter coefficients, `numpy.fft` provides the FFT, and scipy doubles as a **test oracle** that the library's own implementations are verified against.

**Device-agnostic by structure.** The core DSP operates on `complex64` arrays and knows nothing about any device. IQ arrives through a *source* satisfying the `IQSource` protocol; `ArraySource` and `FileSource` ship with the library and carry no device dependency. On the transmit side, IQ leaves through a *sink* — `WavSink`, `IQSink`, and the `TXSink` protocol ship with the library. Device adapters live in *your* code (see `examples/hackrf_capture.py` and `examples/hackrf_sink.py` for HackRF references on each side), so pointing a different SDR at it is a small adaptation. Note: there may be another SDR hook in, but that's a later phase of design.

## Project Layout

The repository has two levels: docs at the root, the installable project one level down in `sdr_dsp/` (that's where `pyproject.toml` lives).

```
sdr_dsp_python/
├── README.md                    # this file
├── LICENSE
├── docs/
│   ├── EXAMPLES.md              # catalog of every example script
│   ├── LOG.md                   # running development log
│   ├── sdr_dsp_REFERENCE.md     # the deep technical reference
│   └── dev_handbook/            # doc-generation helper scripts
└── sdr_dsp/                     # the project — run uv from here
    ├── pyproject.toml
    ├── README.md
    ├── src/sdr_dsp/
    │   ├── __init__.py
    │   ├── py.typed
    │   ├── core/                # PURE DSP — arrays in, arrays out
    │   │   ├── filters.py       # scipy designs taps; we apply them
    │   │   ├── resample.py      # our polyphase resampler
    │   │   ├── spectral.py      # PSD, spectrogram (numpy FFT, our scaling)
    │   │   ├── mixing.py        # frequency translation
    │   │   ├── measure.py       # power, SNR, occupied bandwidth, bursts, CFO
    │   │   ├── agc.py           # automatic gain control
    │   │   ├── calibrate.py     # dBFS → dBm calibration
    │   │   ├── channel.py       # simulated channel (AWGN, CFO, delay)
    │   │   ├── channelize.py    # channel extraction / channelizer bank
    │   │   ├── detect.py        # matched filter, correlation, peak detection
    │   │   ├── framing.py       # frame build/find, CRC
    │   │   ├── sync.py          # carrier recovery, symbol timing
    │   │   ├── demod/           # FM, AM, SSB, OOK, ASK, FSK, (D)PSK, QAM, DSSS/FHSS — all ours
    │   │   └── modulate/        # the transmit mirrors + pulse shaping
    │   ├── sources/             # adapters: where IQ comes from
    │   │   ├── base.py          # IQSource protocol + ArraySource
    │   │   └── file_source.py   # read SigMF recordings (the dev workhorse)
    │   ├── sinks/               # adapters: where results go (WAV, IQ, plot, TX)
    │   ├── stream/              # Pipeline for chunked/streaming processing
    │   ├── link/                # framed link protocol with ARQ + drivers
    │   └── io/
    │       └── sigmf.py         # read ci8 captures, write cf32_le output, annotations
    ├── examples/                # 40 runnable scripts — see docs/EXAMPLES.md
    ├── tests/
    │   ├── conftest.py
    │   ├── helpers/signals.py   # synthetic signal generators (ground truth)
    │   └── test_*.py            # 356 tests; scipy-oracle + synthetic verification
    └── sample_data/             # SigMF recordings for the examples
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/). From the **`sdr_dsp/` project directory** (the one containing `pyproject.toml`, one level below the repo root):

```bash
cd sdr_dsp
uv sync                  # create the venv, install numpy + scipy + dev tools
uv run pytest -q         # run the test suite (expect: 355 passed, 1 skipped)
uv build                 # build the wheel + sdist into dist/
```

The one skipped test requires a connected HackRF and skips itself automatically.

The library installs on **numpy + scipy alone** — no SDR hardware or device library required. Everything works on files.

### Quick start: hear a station

```bash
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
```

This loads a wideband-FM recording, filters to the station, FM-demodulates, resamples to 48 kHz, and writes a WAV you can play. If the station isn't at the capture's center frequency, tune to it with `--tune <offset_hz>`.

### Using the library: receive

```python
from sdr_dsp.sources import FileSource
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod, resample_poly

src = FileSource("sample_data/fm_2Msps.iq")      # reads rate/freq from SigMF
taps = design_lowpass(100_000, src.sample_rate)  # scipy designs the taps
iq = fir_apply(src.iq, taps)                     # our code applies them
audio = fm_demod(iq, deviation_hz=75_000, sample_rate=src.sample_rate)
audio = resample_poly(audio, 48_000 // 16, int(src.sample_rate) // 16)
```

### Using the library: transmit (and get it back)

The transmit side is the mirror image, and the whole loop closes in software — frame, modulate, pass through a simulated channel, demodulate, recover the frame:

```python
import numpy as np
from sdr_dsp.core import (build_frame, find_frames,
                          fsk_modulate, fsk_demod, apply_channel)

fs, sps = 48_000, 8
frame_bits = build_frame(b"hello")               # preamble + sync + len + payload + CRC
iq = fsk_modulate(frame_bits, sps, 5_000, fs, pad_symbols=16)
iq = apply_channel(iq, sample_rate=fs, snr_db=20, seed=1)   # AWGN / CFO / delay

soft = fsk_demod(iq, fs)
rx_bits = (np.asarray(soft)[sps // 2::sps] > 0).astype(np.uint8)
frames = find_frames(rx_bits)
# → [{'payload': b'hello', 'crc_ok': True, ...}]
```

`examples/two_station_link.py` extends this to two stations exchanging acknowledged messages over the `link` ARQ protocol.

## Dependencies

**The library depends only on numpy + scipy.** It installs and runs with no SDR hardware or device library; everything in `src/sdr_dsp/` works on arrays and files alone.

The **example scripts** may need more than the library does. Optional extras (defined in `pyproject.toml`, all for examples rather than the library core):

```bash
uv sync --extra plotting        # matplotlib, for analyzer/spectrogram/teaching examples
uv sync --extra audio           # sounddevice, for live audio playback
uv sync --extra examples-hackrf # hackrfpy, for live capture/transmit
uv sync --extra examples        # everything above in one install
```

There are 40 examples, from single-concept teaching demos (aliasing, windowing, matched filtering) through full receivers (FM, AM, SSB, CW), digital decoders (OOK, ASK, FSK, DPSK, DSSS, FHSS), the transmit arc (modulate → packet → channel sweep → two-station ARQ link), and live-hardware scripts. See [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for the full catalog with per-example requirements.

Live capture and transmit also need the `hackrf-tools` binaries at the OS level (see the hackrfpy docs); file-based examples need none of that.

## Testing approach

Two disciplines, both carried over from the sibling `hackrfpy` project:

- **scipy as oracle** — the library's own implementations (FIR application, resampler) are asserted equal to scipy's within numerical tolerance.
- **synthetic ground truth** — signals are generated with known properties (a tone at a known frequency, an FM-modulated message, an OOK bit pattern, a framed packet through a known channel) and the DSP is checked against what it should recover.

The suite is currently 356 tests. Most tests need no hardware. Hardware-dependent tests (live capture) are marked `@pytest.mark.hardware` and skip automatically when no board is present.

## Documentation

- [`docs/EXAMPLES.md`](docs/EXAMPLES.md) — every example script, grouped by topic, with what each needs to run.
- [`docs/sdr_dsp_REFERENCE.md`](docs/sdr_dsp_REFERENCE.md) — the deep reference: architecture, full module map, usage for receive and transmit, extending the library, known weaknesses, and what still needs real hardware to validate.
- [`docs/LOG.md`](docs/LOG.md) — the running development log.

## Status/Updates

The receive path is complete and tested: filtering, resampling, spectral analysis, measurement, and demodulators from analog (FM/AM/SSB/CW) through coherent digital (PSK/QAM with carrier and timing recovery) and spread spectrum (DSSS/FHSS). The transmit path is complete **in software**: modulators mirror the demodulators, framing and CRC round-trip through a simulated channel, and the ARQ link protocol runs station-to-station in loopback. Some more hardware testing needs to happen before moving on to experimental functions — wired one-way bench tests to characterize fractional delay, drift, and gain staging that the simulated channel can't capture.