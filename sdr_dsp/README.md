# sdr_dsp

A personal, fully-functional DSP library for software-defined radio IQ, written in Python. It consumes IQ samples from a file or a live SDR and turns them into meaning — filtered channels, spectra, demodulated audio, decoded signals — and it can go the other way, framing and modulating messages into IQ for transmit. The full receive/transmit loop closes in software.

This is the installable project directory; the full README, example catalog, and technical reference live one level up in the repository:
https://github.com/LC-Linkous/sdr_dsp_python

## What it is (and isn't)

`sdr_dsp` is a library, not a framework: you import functions and classes and orchestrate the pipeline yourself in plain Python. There is no runtime, scheduler, flowgraph engine, or GUI — GNU Radio remains the right tool for large real-time flowgraphs.

The radio DSP is the library's own code. scipy only *designs* filter coefficients and serves as a test oracle; numpy provides the FFT. The core operates on `complex64` arrays and knows nothing about any device — IQ arrives through the `IQSource` protocol and leaves through sinks, and device adapters live in your own code.

## Install & test

The library depends only on **numpy + scipy**. From this directory:

```
uv sync                  # create the venv, install numpy + scipy + dev tools
uv run pytest -q         # expect: 355 passed, 1 skipped (hardware)
```

Optional extras for the example scripts (not the library core): `--extra plotting` (matplotlib), `--extra audio` (sounddevice), `--extra examples-hackrf` (hackrfpy), or `--extra examples` for all three.

## Quick start

```python
from sdr_dsp.sources import FileSource
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod, resample_poly

src = FileSource("sample_data/fm_2Msps.iq")      # reads rate/freq from SigMF
taps = design_lowpass(100_000, src.sample_rate)
iq = fir_apply(src.iq, taps)
audio = fm_demod(iq, deviation_hz=75_000, sample_rate=src.sample_rate)
audio = resample_poly(audio, 48_000 // 16, int(src.sample_rate) // 16)
```

See `examples/` for 40 runnable scripts — receivers, decoders, teaching demos, the transmit/link arc, and live-hardware helpers — cataloged in `docs/EXAMPLES.md` at the repository root.