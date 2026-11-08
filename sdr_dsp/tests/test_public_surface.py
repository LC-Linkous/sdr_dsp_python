"""The public surface is consistent, complete, documented, and versioned.

Three things nothing else checked:

* every name in every ``__all__`` actually resolves (a typo or a removed
  function in ``__all__`` only fails at ``from pkg import *`` time, which no
  test did), and the lazy ``__getattr__`` loaders in ``sdr_dsp`` and
  ``sdr_dsp.sinks`` deliver every name they promise;
* every public function and class has a docstring -- the same gate hackrfpy
  runs (``test_docstrings.py``), so the surface cannot drift back to bare
  ``help()`` output;
* ``sdr_dsp.__version__`` agrees with ``pyproject.toml`` (it had drifted to
  0.1.0 against a 0.2.0 project).
"""

import importlib
import inspect
import tomllib
from pathlib import Path

import pytest

import sdr_dsp

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# every module that declares a public surface
MODULES = [
    "sdr_dsp", "sdr_dsp.core", "sdr_dsp.core.demod", "sdr_dsp.core.modulate",
    "sdr_dsp.core.features", "sdr_dsp.io", "sdr_dsp.sinks", "sdr_dsp.sources",
    "sdr_dsp.stream", "sdr_dsp.stream.stages", "sdr_dsp.link",
]


@pytest.mark.parametrize("modname", MODULES)
def test_all_names_resolve(modname):
    mod = importlib.import_module(modname)
    assert hasattr(mod, "__all__"), f"{modname} has no __all__"
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"{modname}.__all__ names that do not resolve: {missing}"


@pytest.mark.parametrize("modname", MODULES)
def test_all_has_no_duplicates(modname):
    names = importlib.import_module(modname).__all__
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"{modname}.__all__ lists twice: {dupes}"


# The radio-characterization tier: exported from sdr_dsp.core, deliberately
# NOT flattened to the top-level package (see the core/__init__ docstring).
CORE_ONLY = {
    # device-impairment synthesis
    "add_iq_imbalance", "add_pa_nonlinearity", "add_phase_noise",
    "DeviceImpairments", "make_device_impairments", "apply_device_impairments",
    # fingerprint estimators / feature vector
    "estimate_iq_imbalance", "iq_image_ratio", "estimate_cfo_ppm",
    "estimate_phase_noise_variance", "decide_symbols", "error_vector",
    "evm_stats", "EVM_FEATURE_NAMES", "fingerprint_vector", "FEATURE_NAMES",
    # capture health / pilot gating
    "capture_health", "fm_pilot_excess_db",
    # gain search engine and models
    "search_gain", "peak_counts", "search_gain_model", "GainStage",
    "GainModel", "hackrf_gain_model", "rtlsdr_gain_model",
    # window helper
    "get_window",
}


def test_export_tiers_are_the_documented_split():
    # Every core name is either flattened to the top level or in the
    # documented core-only set -- nothing lands in a tier by accident. A new
    # core export must be added to sdr_dsp.__all__ OR to CORE_ONLY (and the
    # core/__init__ docstring), as a decision.
    import sdr_dsp
    core_names = set(sdr_dsp.core.__all__)
    top_names = set(sdr_dsp.__all__)
    core_only = core_names - top_names
    assert core_only == CORE_ONLY, (
        f"unclassified core exports: {sorted(core_only - CORE_ONLY)}; "
        f"listed as core-only but flattened or gone: {sorted(CORE_ONLY - core_only)}")
    # and the top level never exports a core name that core itself does not
    submods = {"core", "io", "sinks", "stream", "link", "Pipeline",
               "IQSource", "ArraySource", "FileSource"}
    assert (top_names - submods) <= core_names


def test_lazy_submodules_load_and_are_cached():
    for name in ("io", "sinks", "stream", "link"):
        mod = getattr(sdr_dsp, name)
        assert inspect.ismodule(mod)
        assert getattr(sdr_dsp, name) is mod            # cached after first use
    assert sdr_dsp.Pipeline is sdr_dsp.stream.Pipeline


def test_lazy_attribute_errors_are_real_attribute_errors():
    with pytest.raises(AttributeError):
        _ = sdr_dsp.no_such_thing
    with pytest.raises(AttributeError):
        _ = sdr_dsp.sinks.no_such_thing


def test_sinks_lazy_plot_helpers_are_the_plot_sink_ones():
    pytest.importorskip("matplotlib")
    from sdr_dsp.sinks import plot_sink, plot_spectrum, plot_spectrogram
    assert plot_spectrum is plot_sink.plot_spectrum
    assert plot_spectrogram is plot_sink.plot_spectrogram


def _public_callables(modname):
    mod = importlib.import_module(modname)
    out = []
    for name in mod.__all__:
        obj = getattr(mod, name)
        if inspect.ismodule(obj):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            # attribute the object to the module that owns it so a name
            # re-exported from several places is reported once
            out.append((name, obj))
    return out


def _all_public():
    seen = {}
    for m in MODULES:
        for _name, obj in _public_callables(m):
            seen.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
    return sorted(seen.items())


@pytest.mark.parametrize("qualname,obj", _all_public(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_every_public_callable_has_a_docstring(qualname, obj):
    doc = inspect.getdoc(obj)
    assert doc and len(doc.strip()) >= 20, f"{qualname} has no (or a trivial) docstring"


def test_public_classes_document_their_public_methods():
    missing = []
    for qualname, obj in _all_public():
        if not inspect.isclass(obj):
            continue
        for mname, meth in inspect.getmembers(obj, inspect.isfunction):
            if mname.startswith("_") and mname not in ("__call__",):
                continue
            if meth.__qualname__.split(".")[0] != obj.__name__:
                continue                       # inherited, documented upstream
            if not inspect.getdoc(meth):
                missing.append(f"{qualname}.{mname}")
    assert not missing, "public methods without docstrings:\n  " + "\n  ".join(missing)


def test_version_matches_pyproject():
    with open(PYPROJECT, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert sdr_dsp.__version__ == declared, (
        f"sdr_dsp.__version__={sdr_dsp.__version__!r} but pyproject says {declared!r}")


def test_every_module_has_a_docstring():
    """Functions, classes and methods are gated above; this closes the last
    docstring surface. A module docstring is what `help(sdr_dsp.core.sync)`
    and the API handbook generator show first."""
    import pkgutil
    missing = []
    for info in pkgutil.walk_packages(sdr_dsp.__path__, "sdr_dsp."):
        mod = importlib.import_module(info.name)
        if not (mod.__doc__ or "").strip():
            missing.append(info.name)
    assert not missing, "modules without a docstring:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------
# dtype policy (see the core/__init__ docstring): IQ out is complex64, real
# results are float64, for every public function and every input dtype
# ---------------------------------------------------------------------------

def _dtype_cases():
    import numpy as np
    from sdr_dsp.core import design_lowpass
    taps = design_lowpass(50e3, 1e6, num_taps=31)
    fs = 1e6
    S = importlib.import_module("sdr_dsp")
    iq_out = {   # name -> callable; result must be complex64 for complex input
        "fir_apply": lambda x: S.fir_apply(x, taps),
        "fir_apply_centered": lambda x: S.fir_apply_centered(x, taps),
        "resample_poly": lambda x: S.resample_poly(x, 3, 2),
        "decimate": lambda x: S.decimate(x, 4),
        "interpolate": lambda x: S.interpolate(x, 2),
        "frequency_shift": lambda x: S.frequency_shift(x, 1e3, fs),
        "remove_dc": S.remove_dc,
        "normalize": S.normalize,
        "agc": lambda x: S.agc(x)[0],
        "channelize": lambda x: S.channelize(x, fs, 100e3, 50e3)[0],
        "add_noise": lambda x: S.add_noise(x, 20),
        "apply_channel": lambda x: S.apply_channel(x, sample_rate=fs, snr_db=20),
        "carrier_recovery": S.carrier_recovery,
        "symbol_sync": lambda x: S.symbol_sync(x, 8),
    }
    real_out = {  # name -> callable; result must be float64 for any input
        "fm_demod": S.fm_demod,
        "am_demod": S.am_demod,
        "deemphasis": lambda x: S.deemphasis(x, fs),
        "agc gain trace": lambda x: S.agc(x)[1],
        "matched_filter": lambda x: S.matched_filter(x, taps.astype(np.complex64)),
        "fir_apply (real in)": lambda x: S.fir_apply(x.real, taps),
        "resample_poly (real in)": lambda x: S.resample_poly(x.real, 3, 2),
        "normalize (real in)": lambda x: S.normalize(x.real),
    }
    return iq_out, real_out


def test_dtype_policy():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    iq_out, real_out = _dtype_cases()
    bad = []
    for name, f in iq_out.items():
        for dt in (np.complex64, np.complex128):
            got = np.asarray(f(c.astype(dt))).dtype
            if got != np.complex64:
                bad.append(f"{name}({dt.__name__}) -> {got}, want complex64")
    for name, f in real_out.items():
        for dt in (np.complex64, np.complex128):
            got = np.asarray(f(c.astype(dt))).dtype
            if got != np.float64:
                bad.append(f"{name}({dt.__name__}) -> {got}, want float64")
    for name in ("fir_apply (real in)", "resample_poly (real in)", "normalize (real in)"):
        for dt in (np.float32, np.float64):
            got = np.asarray(real_out[name](c.real.astype(dt) + 0j)).dtype
            if got != np.float64:
                bad.append(f"{name}({dt.__name__}) -> {got}, want float64")
    assert not bad, "dtype policy violations:\n  " + "\n  ".join(bad)
