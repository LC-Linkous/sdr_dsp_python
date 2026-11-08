# docs/

Start at the quickstart: [`../sdr_dsp/README.md`](../sdr_dsp/README.md)
(install, extras, hear a station). Everything here is one layer deeper.

## For users

| File | Who it is for |
|---|---|
| [`../sdr_dsp/README.md`](../sdr_dsp/README.md) | Anyone starting out: install and the FM quickstart. |
| [`MODULATIONS.md`](MODULATIONS.md) | Deciding whether your signal is covered, and what "supported" means for it. |
| [`EXAMPLES.md`](EXAMPLES.md) | Finding the example script that does what you want, and what it needs to run. |
| [`HARDWARE.md`](HARDWARE.md) | HackRF One owners: what the board can and can't do, and why it shapes the demods. |
| [`DC_SPIKE.md`](DC_SPIKE.md) | Anyone with a direct-conversion capture: where the DC spike comes from and the offset-tuning recipe. |
| [`sdr_dsp_REFERENCE.md`](sdr_dsp_REFERENCE.md) | Users who want the whole picture: architecture, module map, usage, limitations. |

## For contributors

| File | Who it is for |
|---|---|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Anyone filing a bug, sharing a capture, or wondering why PRs are closed for now. |
| [`ADDING_AN_SDR.md`](ADDING_AN_SDR.md) | Anyone pointing the library at a radio other than the HackRF: the `IQSource`/`TXSink` seam, the porting process and record. |
| [`dev_handbook/`](dev_handbook/) | Generated API handbook plus the scripts that produce it (CI checks it is current). |

## Maintainer notes

| File | Who it is for |
|---|---|
| [`TODO.md`](TODO.md) | Open work in priority order, plus known limitations. The LOG's deferrals live here. |
| [`RELEASING.md`](RELEASING.md) | The maintainer, cutting a version: one-time PyPI setup, the per-release checklist, what is not part of a release. |
| [`LOG.md`](LOG.md) | The dated development log, newest first: what was tried, measured, and why. |
| [`notes/`](notes/) | One-off investigation notes (currently the packet-capture converter note). |

*Documentation is AI-assisted for spelling and structure; all mistakes are
human and will take a revision or two to fix experimentally.*
