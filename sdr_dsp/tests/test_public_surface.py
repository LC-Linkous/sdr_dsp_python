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
