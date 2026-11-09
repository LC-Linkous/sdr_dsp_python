# Releasing sdr_dsp

How a version gets from `main` to PyPI. The mechanics live in
`.github/workflows/release.yml`; this is the human checklist around it. The
package is published as `sdr-dsp` (PyPI normalises the underscore).

## One-time setup (before the first release)

1. **Register the PyPI project and its trusted publisher.** On pypi.org, logged
   in as the owner: *Your projects → Publishing → Add a new pending publisher*:
   - PyPI project name: `sdr-dsp`
   - Owner: `LC-Linkous`  Repository: `sdr_dsp_python`
   - Workflow name: `release.yml`  Environment name: `pypi`

   This is OIDC "trusted publishing": GitHub proves to PyPI which workflow is
   running, and PyPI issues a short-lived token for that run. **No API token
   is created, stored, or rotated.** The pending publisher becomes the real
   one on the first successful publish.
2. **Create the `pypi` environment** in the repo: *Settings → Environments →
   New environment → `pypi`*. Optionally add yourself as a required reviewer:
   then every publish waits for one click, which is cheap insurance against a
   mistyped tag.
3. (Optional) Do the same on **test.pypi.org** and run the workflow once with
   `repository-url: https://test.pypi.org/legacy/` added to the publish step,
   to see the whole path work before it is public. Remove the line after.

## Per release

1. **Pick the version.** While the major is 0, a minor bump (`0.x.0`) is any
   release where a public function's behaviour or signature changed -- a new
   `raise` on previously accepted input counts. A patch bump (`0.3.x`) is
   fixes that no correct caller can observe.
2. **Edit `sdr_dsp/pyproject.toml`**: `version = "X.Y.Z"`. This is the only
   place the version lives; `sdr_dsp.__version__` reads it from the installed
   metadata, and the release workflow refuses a tag that does not match it.
   If the development status changed, update the classifier.
3. **Roll `CHANGELOG.md`**: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`
   with a short summary paragraph on top (what kind of release this is, in
   three or four sentences), and open a fresh empty `## [Unreleased]` above it.
4. **Run the gates locally** from `sdr_dsp/` -- the same ones CI runs:
   ```sh
   uv lock --check
   uv run pytest -q
   uv run ruff check . ../docs/dev_handbook
   uv run mypy
   uv run python ../docs/dev_handbook/generate_api_handbook.py --check ../docs/dev_handbook/sdr_dsp_API_HANDBOOK.md
   uv build && uv run --with twine twine check dist/*
   ```
5. **Commit and push to `main`; wait for `tests.yml` to go green** on the
   full matrix. The release workflow re-runs the suite on one cell, not the
   matrix, so the matrix run on `main` is the real multi-platform gate.
6. **Rehearse the release on `main`** (optional but cheap): *Actions →
   release → Run workflow*, ref `main`, `publish` **unchecked**. The run
   performs every guard the real release will (suite, ruff, mypy, handbook
   `--check`, build, `twine check`, sdist contents) on that commit and
   uploads the wheel + sdist as a workflow artifact named `dist`; nothing is
   published and no tag is needed. Download it and `pip install` the wheel
   into a fresh venv to see the exact bytes that would ship. The only check
   it cannot do without a tag is tag == version, which step 7 does.
7. **Tag and push the tag:**
   ```sh
   git tag -a vX.Y.Z -m "sdr_dsp X.Y.Z"
   git push origin vX.Y.Z
   ```
   `release.yml` then: checks the tag against `pyproject`, re-runs the
   guards on the tagged commit, publishes via trusted publishing, and
   attaches the wheel and sdist to a GitHub release for the tag. If the
   `pypi` environment has a reviewer, approve it in the Actions tab when it
   pauses. To publish by hand instead (or to retry after a failed publish
   job), *Run workflow* with ref `vX.Y.Z` and `publish` **checked**; the
   workflow refuses `publish` on anything that is not a `v*` tag.
8. **Confirm:** `pip install sdr-dsp==X.Y.Z` in a fresh venv;
   `python -c "import sdr_dsp; print(sdr_dsp.__version__)"`.
9. **After the first release only:** remove the "once the first PyPI release
   is cut" hedges in `README.md` and `sdr_dsp/README.md`.

## Things that are deliberately NOT part of a code release

- **The raw reference captures** (`sample_data/hackrf_one/*.iq`, 160 MB) are
  release assets on their own tag (`v0.3.0-captures`, the `DEFAULT_RELEASE`
  in `tools/fetch_sample_data.py`). They change when a capture is re-taken,
  not when code changes, so they are not re-uploaded per version. To publish
  a new set: `gh release create vX.Y.Z-captures sdr_dsp/sample_data/hackrf_one/*.iq`,
  update `DEFAULT_RELEASE`, and make sure each payload's `sdr_dsp:sha256` in
  its sidecar matches (`tools/fetch_sample_data.py --check`).
- **The API handbook** is regenerated and committed as part of the change
  that alters the API, not at release time; CI fails if it is stale.

## If a release goes wrong

- A tag pushed by mistake before `main` was ready: delete it
  (`git push --delete origin vX.Y.Z`) **before** the publish job runs or is
  approved. Once a version is on PyPI it cannot be re-uploaded; the fix is
  the next patch version, and `pip`'s view of the bad one can be hidden with
  *yank* on pypi.org (it stays installable by exact pin, which is what you
  want for anyone who already locked it).
- The workflow refused the tag: the version in `pyproject.toml` and the tag
  disagree. Fix `pyproject`, commit, re-tag.
