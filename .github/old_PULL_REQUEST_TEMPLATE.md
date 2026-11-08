<!-- Thanks for the PR. Keep it short; the checklist below is what CI runs.

     Bug fixes, reproductions, captures and doc fixes: open the PR directly.
     Feature work: discuss it in an issue first (see CONTRIBUTING.md) and
     link that issue below. -->

## What this changes

<!-- One or two sentences. Link the issue if there is one: "Fixes #12". -->

## Why

<!-- The problem or motivation. For a DSP change: what was wrong with the
     output before, and how you know it is right now -- a scipy oracle,
     synthetic ground truth, a real capture. "The tests pass" is not an
     answer on its own; several of the bugs fixed in 0.3.0 passed a green
     suite. -->

## Checklist

Run from `sdr_dsp/` (the directory with `pyproject.toml`).

- [ ] `uv run pytest` passes (hardware tests deselect themselves)
- [ ] `uv run ruff check . ../docs/dev_handbook` and `uv run mypy` are clean
- [ ] `uv lock --check` passes if `pyproject.toml` changed
- [ ] New public functions and classes have docstrings
      (`test_public_surface.py` fails without them)
- [ ] If the public API changed, the handbook is regenerated and committed:
      `uv run python ../docs/dev_handbook/generate_api_handbook.py -o ../docs/dev_handbook/sdr_dsp_API_HANDBOOK.md`
      (CI runs `--check` and fails on a stale copy)
- [ ] New tests draw randomness from `np.random.default_rng(seed)`; nothing
      depends on the order tests run in or on `PYTHONHASHSEED`
- [ ] Files are LF line-ended (CI checks the index; `.gitattributes` handles
      this for you unless you force CRLF)
- [ ] `CHANGELOG.md` has an entry under **Unreleased** if a user would notice
- [ ] Docs updated if behaviour or the API changed: `docs/sdr_dsp_REFERENCE.md`,
      `docs/MODULATIONS.md`, `docs/EXAMPLES.md`, as relevant

## Hardware

<!-- Delete this section if it does not apply.

     If this touches a device adapter, a live example, or the TX path: was it
     run on a board? Which one, which band, what happened? "Not run on
     hardware" is a fine answer -- it just needs saying, because most of the
     digital path is verified in software only, and that is the single most
     useful thing for a reviewer to know. -->