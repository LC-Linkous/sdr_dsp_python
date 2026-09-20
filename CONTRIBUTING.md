# Contributing to sdr_dsp_python

Thanks for your interest. This is a **personal project**, built for
experimental and educational purposes, and it's still pre-release — not on
PyPI, with the API still moving.

## Pull requests: not right now

**I'm not accepting pull requests at the moment.** That isn't a comment on the
quality of anyone's work — it's that this is a personal project I'm actively
reshaping, and merging outside changes while the structure is still in motion
creates more coordination than I can keep up with. If you open one I'll read it
and I'll thank you for it, but I'll most likely close it and fold the idea in
myself if it fits.

That may change once the API settles and the library goes to PyPI. Until then,
please fork freely — the license permits it, and you don't need my permission.

## What is genuinely useful

**Bug reports.** These are the most valuable thing you can send, and testing
against a capture I've never seen is exactly the kind of coverage I can't
generate alone. Use the bug template.

**Reproductions.** If something misbehaves on your own recording, a short
script plus a description of the capture (sample rate, datatype, center
frequency) is worth far more than a patch. If the recording is small and you're
able to share it, better still.

**Documentation problems.** Wrong, stale, or confusing docs — including
anything in an example's docstring — are bugs. Report them the same way.

**Feature ideas.** Open an issue with the feature template. Note the scope
boundaries there before filing: no GUI, no flowgraph runtime, and no device
dependencies in the core.

## Before you file a bug

Run the capture check first:

```bash
cd sdr_dsp
uv run python examples/inspect_capture.py your_capture.iq
```

Most "it ran without errors but the output is silent or looks like noise"
reports are a property of the recording rather than the DSP — an antenna that
wasn't connected, gain too low, or a capture tuned to a dead frequency. An
8-bit capture using only a couple of ADC counts of range has nothing in it, and
no amount of processing will find a signal that was never recorded. The check
will tell you.

Then try to reproduce against `sample_data/`. If it reproduces there, I can fix
it without your hardware, which is much faster for both of us.

## Capture data: two tiers

Recordings live in one of two places, and the distinction matters:

- **`sample_data/`** is committed. Every clone carries it, the examples
  default to it, and the test suite runs against it. Small and curated.
- **`dev_data/`** is gitignored. It is the full working corpus produced by
  `tools/collect_dev_data.py` — the same station recorded at several sample
  rates, several gains, tuned on and off center, plus deliberate
  known-negatives. Tens of megabytes, useful to have on the machine doing
  the development, not something to carry in git forever.

Captures move from the second to the first through
`tools/promote_to_sample_data.py`, never by hand. The script re-runs
`capture_health` on each file *after* copying it and refuses to promote a
capture that fails its own declared expectation, then records the source,
gain settings, and both health results in `sample_data/manifest.json`. A
hand-copied file loses exactly the record that would catch a bad recording,
which is how a noise-floor capture came to ship as the FM example's default
input.

Some files in `sample_data/` are empty **on purpose** — fixtures for the
no-signal code paths, which the tests assert `capture_health` rejects. Those
are declared in the manifest and explained in the directory's README. If one
of them ever starts passing the health check, that is the bug.

## Running things yourself

This project uses [uv](https://docs.astral.sh/uv/). The installable project
lives in the `sdr_dsp/` subdirectory — the one with `pyproject.toml`, one level
below the repo root.

```bash
cd sdr_dsp
uv sync                       # numpy + scipy + dev tools, editable install
uv run pytest -q              # the full suite
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out out.wav
```

Run everything through `uv run` so the synced environment is used. A bare
`python ...` can silently pick up a different environment.

The examples import `sdr_dsp` as an installed package — there are no
`sys.path` shims — so `uv sync` has to have run first. Once it has, examples
work from any directory.

Optional extras, only needed by some examples (the library core needs neither):

```bash
uv sync --extra examples      # matplotlib + sounddevice + hackrfpy
uv sync --extra plotting      # just matplotlib
```

Tests that need a connected HackRF are marked `@pytest.mark.hardware` and skip
themselves automatically when no device is present.

## How the code is organized

Two rules explain most of the structure:

**The radio DSP is the library's own code.** Filtering, demodulation,
modulation, resampling, mixing, and measurement are implemented here. `scipy`
is a design-time tool, not a runtime crutch: `scipy.signal` only *designs*
filter coefficients, `numpy.fft` provides the FFT, and scipy doubles as a test
oracle that the library's own implementations are verified against. If you're
reporting a numerical discrepancy, that oracle relationship is usually where to
look.

**The core is device-agnostic.** Core DSP operates on `complex64` arrays and
knows nothing about any radio. IQ arrives through a *source* satisfying the
`IQSource` protocol and leaves through a *sink*. Device adapters live in user
code — `examples/hackrf_capture.py` and `examples/hackrf_sink.py` are
references for each side.

Everything in the core is verified either against scipy as an oracle or against
a synthetic signal with known ground truth. `tests/helpers/signals.py` holds the
generators.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under **GPL-2.0**. Anything you share in an issue —
snippets, reproductions, captures — is offered under those terms.
