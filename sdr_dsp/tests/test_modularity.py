"""Modularity guards: prove sdr_dsp is not locked to any one SDR.

Two complementary checks:

1. The core imports NO device library. This is the structural guarantee behind
   the whole design -- if it holds, any SDR can be driven by writing an adapter
   OUTSIDE src/, and the DSP never needs to change. A single stray
   `import hackrfpy` in a core module would quietly break that promise; this
   test makes it a red build instead.
2. A NON-HackRF source (the adapter template, over a fake cu8 device) drives
   the real library end to end. This exercises the seam the way a new SDR
   would: normalize a foreign sample format to complex64, hand it to the core,
   get a real measurement back -- with nothing device-specific in between.

See docs/ADDING_AN_SDR.md and docs/MODULARITY.md.
"""

import ast
import sys
from pathlib import Path

import numpy as np

from sdr_dsp.sinks import TXSink
from sdr_dsp.sources import ArraySource, IQSource

# examples aren't a package; add the dir so we can import the template adapter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import sdr_adapter_template as tmpl        # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src" / "sdr_dsp"

# Device SDKs the CORE must never import. Adapters live in examples/ and may
# import these freely; the core may not. Add new SDKs here as adapters appear.
DEVICE_LIBS = {
    "hackrfpy", "rtlsdr", "pyrtlsdr", "SoapySDR", "soapysdr", "uhd",
    "bladerf", "airspy", "limesdr", "sounddevice",
}


def _imported_top_levels(pyfile):
    """Top-level package names imported by a Python file (via AST, no exec)."""
    tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:      # skip relative imports
                names.add(node.module.split(".")[0])
    return names


# --------------------------------------------------------------------------
# 1. the structural guarantee: the core is device-free
# --------------------------------------------------------------------------
def test_core_imports_no_device_library():
    offenders = {}
    for pyfile in sorted(SRC.rglob("*.py")):
        hits = _imported_top_levels(pyfile) & DEVICE_LIBS
        if hits:
            offenders[str(pyfile.relative_to(SRC))] = sorted(hits)
    assert not offenders, (
        "core module(s) import a device library -- the DSP must stay "
        f"device-agnostic; move device code to examples/: {offenders}")


def test_core_files_were_actually_scanned():
    """Guard the guard: make sure the walk found the core, not an empty dir."""
    scanned = list(SRC.rglob("*.py"))
    assert len(scanned) > 20, f"only {len(scanned)} core files scanned"


# --------------------------------------------------------------------------
# 2. a non-HackRF adapter satisfies the protocols and drives the pipeline
# --------------------------------------------------------------------------
def test_template_source_satisfies_iqsource():
    src = tmpl.TemplateSDRSource(2e6, 103.7e6)
    assert isinstance(src, IQSource)
    assert src.sample_rate == 2e6 and src.center_freq == 103.7e6


def test_template_sink_satisfies_txsink_and_is_guarded():
    sink = tmpl.TemplateSDRSink(103.7e6, 2e6)
    assert isinstance(sink, TXSink)
    assert sink.armed is False
    try:
        sink.transmit(np.zeros(8, np.complex64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("unarmed sink should refuse to transmit")


def test_template_normalizes_cu8_to_unit_complex64():
    """The sample-format spot: cu8 midpoint 127.5 -> ~0, full scale -> ~+/-1."""
    raw = bytes([127, 127, 255, 255, 0, 0])      # ~zero, +full, -full (I/Q)
    z = tmpl.TemplateSDRSource._to_complex64(raw)
    assert z.dtype == np.complex64
    assert abs(z[0]) < 0.02                       # midpoint -> ~0
    assert 0.9 < z[1].real <= 1.1 and 0.9 < z[1].imag <= 1.1
    assert -1.1 <= z[2].real < -0.9 and -1.1 <= z[2].imag < -0.9


def test_non_hackrf_source_drives_the_core():
    """End to end over a foreign (cu8) device: adapter -> core DSP -> result."""
    from sdr_dsp.core import fm_pilot_excess_db, psd

    src = tmpl.TemplateSDRSource(2e6, 103.7e6, block_size=200_000)
    blocks = list(src.blocks())
    assert blocks and blocks[0].dtype == np.complex64

    iq = np.concatenate(blocks)
    freqs, power = psd(iq, 2e6, nfft=4096)
    assert power.shape[0] == 4096 and np.all(np.isfinite(power))
    # the fake device carries a 19 kHz pilot, so the pilot check should fire
    pilot = fm_pilot_excess_db(iq, 2e6)
    assert pilot is not None and pilot > 6.0


def test_shipped_arraysource_is_also_device_free_and_drives_core():
    """Sanity: the library's own ArraySource is just another IQSource."""
    from sdr_dsp.core import psd
    iq = np.exp(1j * np.linspace(0, 100, 4096)).astype(np.complex64)
    src = ArraySource(iq, 2e6, center_freq=100e6)
    assert isinstance(src, IQSource)
    got = np.concatenate(list(src.blocks()))
    freqs, power = psd(got, 2e6, nfft=1024)
    assert np.all(np.isfinite(power))
