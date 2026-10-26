<!-- Thanks. Short is fine; the checklist is what CI will ask anyway. -->

## What this changes

<!-- One or two sentences. Link the issue if there is one: "Fixes #12". -->

## Why

<!-- The problem or motivation. For a DSP change, what was wrong with the
     output before, and how you know it is right now (scipy oracle,
     synthetic ground truth, a real capture). -->

## Checklist

- [ ] `uv run pytest -m "not hardware"` passes locally
- [ ] `uv run ruff check .` and `uv run mypy` are clean
- [ ] New public functions and methods have docstrings (`test_public_surface.py` enforces this)
- [ ] New tests seed randomness explicitly (`np.random.default_rng(seed)`), never the builtin `hash()`
- [ ] New files are LF line-ended (CI checks the index)
- [ ] `CHANGELOG.md` has an entry under **Unreleased** if a user would notice the change
- [ ] Docs updated if behaviour or the public API changed (`docs/sdr_dsp_REFERENCE.md`, `docs/MODULATIONS.md`, `docs/EXAMPLES.md` as relevant)

## Hardware

<!-- Delete if not applicable. If this touches a device adapter, a live
     example, or the TX path: was it run on a board? Which one, which band,
     what happened? "Not run on hardware" is an acceptable answer -- it just
     needs saying, since most of the digital path is software-verified only. -->
