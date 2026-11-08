# sdr_dsp_python

A personal, fully-functional DSP library for software-defined radio, written in
Python. It consumes IQ samples from a file or a live SDR and turns them into
filtered channels, spectra, demodulated audio, and decoded frames -- and runs
the other way too, so a full loop from RX through processing to TX is possible.
It is a library, not a framework: you import functions and compose the pipeline
yourself in plain Python. There is no runtime, scheduler, flowgraph engine, or
GUI, and it is not a GNU Radio replacement.

The installable project lives one level down in [`sdr_dsp/`](sdr_dsp/) (that is
where `pyproject.toml` is). Run `uv` from there:

```bash
cd sdr_dsp && uv sync        # (pip install sdr-dsp once the first PyPI release is cut)
```

Hear a station from the shipped capture:

```python
from sdr_dsp import tune_to_baseband, design_lowpass, fir_apply, fm_demod
from sdr_dsp.io import load_iq

iq, meta = load_iq("sample_data/fm_2Msps.iq")
fs = meta["global"]["core:sample_rate"]
iq = tune_to_baseband(iq, offset_hz=0.0, sample_rate=fs)   # capture is centered
audio = fm_demod(fir_apply(iq, design_lowpass(100e3, fs)),
                 deviation_hz=75e3, sample_rate=fs)
```

## Where to go next

- [`sdr_dsp/README.md`](sdr_dsp/README.md) -- install, extras, and the quickstart.
- [`docs/README.md`](docs/README.md) -- index of all the documentation.
- [`docs/MODULATIONS.md`](docs/MODULATIONS.md) -- which schemes are covered and what "supported" means for each.
- [`docs/EXAMPLES.md`](docs/EXAMPLES.md) -- catalog of the 47 example scripts and what each needs.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- bug reports, captures, and what kind of PR is welcome.
- [`CHANGELOG.md`](CHANGELOG.md) -- user-facing changes by version.

## Status

The receive path is complete and tested in software: filtering, resampling,
spectral analysis, measurement, and demodulators from analog (FM/AM/SSB/CW)
through coherent digital (PSK/QAM) and spread spectrum (DSSS/FHSS). FM is the
only scheme validated on a real over-the-air capture; the digital schemes are
validated against the channel simulator only. The transmit path and the ARQ
link protocol are complete in software and close the loop in loopback, but the
wired one-way bench tests that would characterize fractional delay, drift, and
gain staging have not been run yet.

> [!NOTE]
> AI is being used to summarize the codebase and development notes into these
> docs. All mistakes are human, and will likely take a revision or two to fix.

Participation in the project's spaces is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). Licensed under GPL-2.0; see
[`LICENSE`](LICENSE).
