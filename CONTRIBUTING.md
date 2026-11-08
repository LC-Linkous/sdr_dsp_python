# Contributing to sdr_dsp_python

Thanks for your interest. This is a **personal project**, built for
experimental and educational purposes, and it's still pre-release — not yet on
PyPI, with the API still moving between minor versions.

## Issues yes, pull requests not yet

**I'm not accepting pull requests at the moment.** The structure is still
moving quickly enough that a PR written against this week's layout is often
stale by the time I get to it, and reviewing it costs more than the change
would save. This will change once the API settles; until then, PRs will be
closed with a pointer here, not reviewed.

**Issues are very welcome** — for bugs, reproductions, captures, documentation
problems, hardware reports, and feature ideas. If you already have a fix, put
the diff or the snippet in the issue; a failing test that pins a bug is just as
useful pasted as pushed. Use the issue templates; each one asks for what I
actually need.

**Please fork freely.** The license permits it and you don't need my
permission. If your fork grows something worth bringing back, open an issue
describing it and link the fork.

Scope boundaries, so nobody spends time on something that won't land: no GUI,
no flowgraph runtime, and no device dependencies in the core.

## What is genuinely useful

**Bug reports.** These are the most valuable thing you can send, and testing
against a capture I've never seen is exactly the kind of coverage I can't
generate alone. Use the bug template.

**Reproductions.** If something misbehaves on your own recording, a short
script plus a description of the capture (sample rate, datatype, center
frequency) is worth far more than a patch. If the recording is small and you're
able to share it, better still.

**Documentation problems.** Wrong, stale, or confusing docs — including
anything in an example's docstring — are bugs. Report them the same way; quote
the passage and say what it should have said.

**Hardware reports.** If you ran something on a board (yours, or a radio this
library has never met), use the hardware/bench report template, whether it
worked or not. See [`docs/ADDING_AN_SDR.md`](docs/ADDING_AN_SDR.md) for how
a port is done and recorded.

**Feature ideas.** Use the feature template. Say what you'd use it for; a
concrete signal or workflow is far more persuasive than a general capability.

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
`tools/promote_to_sample_data.py` (or, for a validated reference recording,
`tools/import_reference_capture.py`), never by hand. Both re-run
`capture_health` on the file *after* copying it and refuse a capture that
fails its own declared expectation; the promote script records the source,
gain settings, and both health results in `sample_data/manifest.json`, the
import script writes the provenance into the SigMF sidecar. A hand-copied
file loses exactly the record that would catch a bad recording, which is how
a noise-floor capture once came to ship as the FM example's default input.
See [`sdr_dsp/sample_data/README.md`](sdr_dsp/sample_data/README.md) for
what is there now.

## Running things yourself

This project uses [uv](https://docs.astral.sh/uv/). The installable project
lives in the `sdr_dsp/` subdirectory — the one with `pyproject.toml`, one level
below the repo root.

```bash
cd sdr_dsp
uv sync                       # numpy + scipy + dev tools, editable install
uv run pytest -q              # the suite (hardware tests deselected by default)
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out out.wav
```

Run everything through `uv run` so the synced environment is used. A bare
`python ...` can silently pick up a different environment.

The examples import `sdr_dsp` as an installed package — there are no
`sys.path` shims — so `uv sync` has to have run first. Once it has, examples
work from any directory.

Optional extras (`plotting`, `audio`, `examples-hackrf`, `examples`) are only
needed by some examples; the library core needs none of them. They are listed
in [`sdr_dsp/README.md`](sdr_dsp/README.md).

Tests that need a connected HackRF are marked `@pytest.mark.hardware`, are
deselected from a bare `pytest`, and skip themselves when run explicitly
(`pytest -m hardware`) with no device present.

If you're working in a fork, the gates CI applies are: `uv run pytest`,
`uv run ruff check . ../docs/dev_handbook`, `uv run mypy`, `uv lock --check`,
LF line endings in the index, and the API handbook regenerated
(`uv run python ../docs/dev_handbook/generate_api_handbook.py --check
../docs/dev_handbook/sdr_dsp_API_HANDBOOK.md`). The workflow file is
`.github/workflows/tests.yml`.

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