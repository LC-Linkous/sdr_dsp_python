# sdr_dsp

A personal, fully-functional DSP library for software-defined radio, written in Python. It consumes IQ samples from a file or a live SDR and turns them into meaning: filtered channels, spectra, demodulated audio, decoded signals.

This is part of a personal ecosystem of libraries for experimenal and educational purposes, so there are already tools out there that do some of this.

`sdr_dsp` is a library, not a framework. When using, you have to import functions and classes and orchestrate the pipeline yourself in plain Python. There is no runtime, scheduler, flowgraph engine, or GUI. It is not a GNU Radio competitor or replacement. GNU Radio remains the right tool for large real-time flowgraphs; `sdr_dsp` is for direct, scriptable DSP in Python.

## Design

**The radio DSP is the library's own code.** Filtering, demodulation, resampling, mixing, and measurement are all implemented here. scipy is a design-time tool, not a runtime crutch: `scipy.signal` only *designs* filter coefficients, `numpy.fft` provides the FFT, and scipy doubles as a **test oracle** that the library's own implementations are verified against.

**Device-agnostic by structure.** The core DSP operates on `complex64` arrays and knows nothing about any device. IQ arrives through a *source* satisfying the `IQSource` protocol; `ArraySource` and `FileSource` ship with the library and carry no device dependency. A device source lives in *your* code (see `examples/hackrf_capture.py` for a HackRF reference), so pointing a different SDR at it is a small adaption. Note: there may be another SDR hook in, but that's a later phase of design.

## Project Layout

```
sdr_dsp/
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
│   │   ├── demod.py         # FM, AM, OOK — all ours
│   │   └── measure.py       # power, SNR, occupied bandwidth
│   ├── sources/             # adapters: where IQ comes from
│   │   ├── base.py          # IQSource protocol + ArraySource
│   │   └── file_source.py   # read SigMF recordings (the dev workhorse)
│   ├── sinks/               # adapters: where results go (WIP)
│   └── io/
│       └── sigmf.py         # read ci8 captures, write cf32_le output
├── examples/
│   ├── fm_receiver.py       # IQ file -> filtered -> demod -> WAV
│   └── hackrf_capture.py    # live-capture helper (NOT library; uses hackrfpy)
├── tests/
│   ├── conftest.py
│   ├── helpers/signals.py   # synthetic signal generators (ground truth)
│   └── test_*.py            # 34 tests; scipy-oracle + synthetic verification
└── sample_data/             # SigMF recordings for the examples
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/). From the project directory (the one containing `pyproject.toml`):

```bash
uv sync                  # create the venv, install numpy + scipy + dev tools
uv run pytest -q         # run the test suite (expect: 34 passed)
uv build                 # build the wheel + sdist into dist/
```

The library installs on **numpy + scipy alone** — no SDR hardware or device library required. Everything works on files.

### Quick start: hear a station

```bash
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
```

This loads a wideband-FM recording, filters to the station, FM-demodulates, resamples to 48 kHz, and writes a WAV you can play. If the station isn't at the capture's center frequency, tune to it with `--tune <offset_hz>`.

### Using the library

```python
from sdr_dsp.sources import FileSource
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod, resample_poly

src = FileSource("sample_data/fm_2Msps.iq")      # reads rate/freq from SigMF
taps = design_lowpass(100_000, src.sample_rate)  # scipy designs the taps
iq = fir_apply(src.iq, taps)                      # our code applies them
audio = fm_demod(iq, deviation_hz=75_000, sample_rate=src.sample_rate)
audio = resample_poly(audio, 48_000 // 16, int(src.sample_rate) // 16)
```

## Dependencies

**The library depends only on numpy + scipy.** It installs and runs with no SDR hardware or device library; everything in `src/sdr_dsp/` works on arrays and files alone.

The **example scripts** may need more than the library does. Optional extras (defined in `pyproject.toml`, all for examples rather than the library core):

```bash
uv sync --extra plotting        # matplotlib, for analyzer/spectrogram examples
uv sync --extra audio           # sounddevice, for live audio playback
uv sync --extra examples-hackrf # hackrfpy, for live capture
uv sync --extra examples        # everything above in one install
```

Per-example requirements:

| Example | Needs (beyond the library) | Install |
|---|---|---|
| `fm_receiver.py` | nothing extra | — |
| `hackrf_capture.py` | `hackrfpy` + `hackrf-tools` binaries (OS level) | `uv sync --extra examples-hackrf` |
| *(spectrum analyzer — planned)* | `matplotlib` | `uv sync --extra plotting` |
| *(live examples — planned)* | `hackrfpy`, optionally `sounddevice` | `uv sync --extra examples-hackrf` |

Live capture also needs the `hackrf-tools` binaries at the OS level (see the hackrfpy docs); file-based examples need none of that.

## Testing approach

Two disciplines, both carried over from the sibling `hackrfpy` project:

- **scipy as oracle** — the library's own implementations (FIR application, resampler) are asserted equal to scipy's within numerical tolerance.
- **synthetic ground truth** — signals are generated with known properties (a tone at a known frequency, an FM-modulated message, an OOK bit pattern) and the DSP is checked against what it should recover.

Most tests need no hardware. Hardware-dependent tests (live capture) are marked `@pytest.mark.hardware` and skip automatically when no board is present.

## Status/Updates

Phase 1 (core DSP primitives) and the first end-to-end example (FM receiver) are complete and tested. Planned next: a spectrum analyzer, a resampler benchmark against scipy, an OOK decoder, and a channelizer. Some more hardware testing needs to happen before moving on to experimental functions.