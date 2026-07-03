#! /usr/bin/python3
"""generate_api_handbook.py -- introspect sdr_dsp and emit the definitive API handbook.

Walks every public module, function, and class, pulling the REAL signature
(parameters, defaults, type annotations) and docstring straight from the source,
so the handbook can never drift from the code. Re-run after any API change.

Usage:
    PYTHONPATH=src python3 tools/generate_api_handbook.py > docs/sdr_dsp_API_HANDBOOK.md
"""
import importlib
import inspect
import sys
import textwrap

# the public surface, in the order we want it documented
MODULES = [
    ("sdr_dsp", "Top-level package", "Lazily-loaded submodules and the flattened DSP API."),
    ("sdr_dsp.core", "core — pure DSP", "Every signal-processing primitive. numpy + scipy only."),
    ("sdr_dsp.core.demod", "core.demod — demodulators", "Recover bits/audio/symbols from IQ."),
    ("sdr_dsp.core.modulate", "core.modulate — modulators", "Inverse of demod: bits/symbols -> IQ."),
    ("sdr_dsp.sources", "sources — receive seam", "IQSource protocol and concrete sources."),
    ("sdr_dsp.sinks", "sinks — output & transmit seam", "TXSink protocol, file and plot sinks."),
    ("sdr_dsp.io", "io — file formats", "SigMF load/save, annotations, metadata."),
    ("sdr_dsp.stream", "stream — orchestration", "Pipeline and block-streaming."),
    ("sdr_dsp.link", "link — ARQ protocol", "Reliable acknowledged messaging."),
]


def fmt_signature(obj):
    """Return the signature string with annotations and defaults intact."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "(...)"


def param_table(obj):
    """Build a parameter table from the signature: name, default, annotation."""
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return ""
    rows = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ann = "" if p.annotation is inspect.Parameter.empty else _ann(p.annotation)
        if p.default is inspect.Parameter.empty:
            default = "*required*"
        else:
            default = f"`{p.default!r}`"
        kind = ""
        if p.kind == p.VAR_POSITIONAL:
            kind = " (*args)"
        elif p.kind == p.VAR_KEYWORD:
            kind = " (**kwargs)"
        rows.append(f"| `{name}`{kind} | {ann} | {default} |")
    if not rows:
        return "_No parameters._\n"
    return ("| Parameter | Type | Default |\n|---|---|---|\n"
            + "\n".join(rows) + "\n")


def _ann(annotation):
    """Render a type annotation compactly."""
    if isinstance(annotation, str):
        return f"`{annotation}`"
    name = getattr(annotation, "__name__", None)
    if name:
        return f"`{name}`"
    return f"`{str(annotation).replace('typing.', '')}`"


def doc_block(obj):
    """Return the cleaned docstring, or a placeholder."""
    d = inspect.getdoc(obj)
    if not d:
        return "_No docstring._\n"
    return d + "\n"


def document_function(name, obj, level="###"):
    out = [f"{level} `{name}{fmt_signature(obj)}`\n"]
    out.append(doc_block(obj))
    out.append("\n**Parameters:**\n")
    out.append(param_table(obj))
    # return annotation if present
    try:
        sig = inspect.signature(obj)
        if sig.return_annotation is not inspect.Signature.empty:
            out.append(f"\n**Returns:** {_ann(sig.return_annotation)}\n")
    except (ValueError, TypeError):
        pass
    out.append("\n")
    return "\n".join(out)


def document_class(name, obj):
    out = [f"### class `{name}`\n"]
    out.append(doc_block(obj))
    # constructor
    init = getattr(obj, "__init__", None)
    if init and init is not object.__init__:
        out.append(f"\n**Constructor:** `{name}{fmt_signature(init)}`\n")
        out.append(param_table(init))
    # public methods
    methods = [(n, m) for n, m in inspect.getmembers(obj, inspect.isfunction)
               if not n.startswith("_")]
    # also catch methods defined directly (not just inherited)
    own_methods = [(n, getattr(obj, n)) for n in vars(obj)
                   if callable(getattr(obj, n, None)) and not n.startswith("_")]
    seen = set()
    all_methods = []
    for n, m in own_methods + methods:
        if n not in seen:
            seen.add(n)
            all_methods.append((n, m))
    if all_methods:
        out.append("\n**Methods:**\n")
        for n, m in sorted(all_methods):
            out.append(f"#### `{n}{fmt_signature(m)}`\n")
            out.append(doc_block(m))
            pt = param_table(m)
            if "No parameters" not in pt:
                out.append(pt)
            out.append("")
    # public attributes (class-level, annotated)
    annotations = getattr(obj, "__annotations__", {})
    if annotations:
        out.append("\n**Attributes:**\n")
        for an, at in annotations.items():
            out.append(f"- `{an}`: {_ann(at)}")
        out.append("")
    out.append("\n")
    return "\n".join(out)


def main():
    print("# sdr_dsp — Definitive API Handbook\n")
    print("> **Auto-generated** from the source by "
          "`tools/generate_api_handbook.py`. Every signature, parameter, "
          "default, and docstring is extracted directly from the code, so this "
          "handbook cannot drift from the library. Re-run the generator after "
          "any API change.\n")
    print("> For architecture, philosophy, extension guides, hardware notes, "
          "and diagrams, see `sdr_dsp_REFERENCE.md`. This document is the "
          "exhaustive call-level reference.\n")
    print("> **Reading the tables:** where the *Type* column is blank, the "
          "library documents argument types in the docstring rather than via "
          "annotations — read the function's docstring for the expected dtype "
          "and shape (typically `numpy.complex64` IQ arrays, `float` rates in "
          "Hz, and `bytes`/bit-arrays for the protocol layer). A *Default* of "
          "`*required*` means the argument is positional and mandatory.\n")
    print("> **Note on ARQ:** `window_size=1` is stop-and-wait (the default); "
          "`window_size=N` is sliding-window Selective Repeat. The opt-in "
          "`ARQ(cumulative_ack=True)` acknowledges the contiguous high-water "
          "mark instead of each frame. Both ACK modes are validated correct "
          "under heavy random and burst loss — see `sdr_dsp_REFERENCE.md` "
          "§11.1.\n")

    # table of contents
    print("## Modules\n")
    for modname, title, _ in MODULES:
        anchor = title.lower().replace(" — ", "--").replace(" ", "-").replace("—", "")
        print(f"- [{title}](#{anchor})")
    print("\n---\n")

    for modname, title, blurb in MODULES:
        mod = importlib.import_module(modname)
        print(f"## {title}\n")
        print(f"{blurb}\n")
        print(f"Import: `import {modname}`\n")

        names = getattr(mod, "__all__", None)
        if not names:
            names = [n for n in dir(mod) if not n.startswith("_")]
        # split into functions and classes
        funcs, classes, others = [], [], []
        for n in sorted(names):
            obj = getattr(mod, n, None)
            if obj is None:
                continue
            if inspect.isclass(obj):
                classes.append((n, obj))
            elif inspect.isfunction(obj):
                funcs.append((n, obj))
            elif inspect.ismodule(obj):
                others.append((n, obj))

        if others:
            print("**Submodules:** "
                  + ", ".join(f"`{n}`" for n, _ in others) + "\n")

        if classes:
            print("### Classes\n")
            for n, obj in classes:
                print(document_class(n, obj))

        if funcs:
            print("### Functions\n")
            for n, obj in funcs:
                print(document_function(n, obj))

        print("---\n")


if __name__ == "__main__":
    main()
