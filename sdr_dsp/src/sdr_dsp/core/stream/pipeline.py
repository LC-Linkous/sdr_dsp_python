"""Pipeline: streaming orchestration over the pure DSP core.

A Pipeline threads blocks from a source through an ordered list of operations to
a sink. It ORCHESTRATES the pure-function core -- every stage is just a callable
that takes a block and returns a block (the same library functions you'd call by
hand) -- so streaming adds no new DSP, only flow control.

The chain is data: you can inspect it, reorder it, and insert taps. A tap is a
peek-at-the-data stage that fires a callback without altering the flow, which is
how live display falls out of the same abstraction (a power meter, a
constellation view, a decoded-message readout are all taps). This keeps the
generator-chain style's composability and adds inspectability and live feedback
in one model.

Design note: stages are plain callables, so the lighter generator-chain style is
not locked out -- you can still nest generators by hand. Pipeline is the
recommended default; it is not the only option.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np


@dataclass
class Stage:
    """One step in a pipeline: a named block->block operation, or a tap."""
    name: str
    fn: Callable
    is_tap: bool = False


@dataclass
class PipelineStats:
    """Per-stage timing/throughput, collected when run(profile=True)."""
    blocks: int = 0
    samples_in: int = 0
    per_stage_seconds: dict = field(default_factory=dict)

    def __str__(self):
        lines = [f"{self.blocks} blocks, {self.samples_in:,} input samples"]
        for name, secs in self.per_stage_seconds.items():
            lines.append(f"  {name:20s} {secs*1e3:8.2f} ms total")
        return "\n".join(lines)


class Pipeline:
    """A source + an ordered chain of block->block stages (and taps).

    Build declaratively and run:

        pipe = (Pipeline(source)
                .add(lambda b: fir_apply(b, taps), "filter")
                .add(lambda b: fm_demod(b, 75000, fs), "demod")
                .tap(lambda b: meter.update(b)))      # live peek, flow unchanged
        audio = pipe.run()                            # or run(sink=write_audio)

    Stages transform the block; taps observe it and return nothing (the original
    block continues). Order matters and is preserved.
    """

    def __init__(self, source):
        self.source = source
        self.stages: list[Stage] = []

    # -- building -----------------------------------------------------------
    def add(self, fn, name=None):
        """Append a transforming stage (block -> block)."""
        self.stages.append(Stage(name or getattr(fn, "__name__", "stage"), fn))
        return self

    def tap(self, fn, name=None):
        """Append an observing stage (block -> ignored). Flow is unchanged.

        A tap is how live display attaches: fn receives the current block and
        does whatever it likes (update a plot, accumulate a message) without
        affecting what the next stage sees.
        """
        self.stages.append(Stage(name or getattr(fn, "__name__", "tap"),
                                 fn, is_tap=True))
        return self

    def describe(self):
        """Return the chain as inspectable text (the pipeline is data)."""
        rows = [f"source: {self.source!r}"]
        for i, st in enumerate(self.stages):
            kind = "tap " if st.is_tap else "step"
            rows.append(f"  [{i}] {kind} {st.name}")
        return "\n".join(rows)

    # -- running ------------------------------------------------------------
    def process_block(self, block):
        """Thread a single block through every stage. Useful for testing and
        for driving the pipeline from an external loop (e.g. a GUI timer)."""
        x = block
        for st in self.stages:
            if st.is_tap:
                st.fn(x)              # observe, don't replace
            else:
                x = st.fn(x)
        return x

    def run(self, sink=None, profile=False, max_blocks=None):
        """Pull blocks from the source, process each, deliver to sink.

        sink:       callable(result_block) -> None. If None, results are
                    collected and returned as a list.
        profile:    if True, also return PipelineStats (per-stage timing).
        max_blocks: stop after this many blocks (useful for live sources).

        Returns the result list (or None if a sink was given); if profile,
        returns (results_or_None, PipelineStats).
        """
        results = [] if sink is None else None
        stats = PipelineStats()
        for st in self.stages:
            stats.per_stage_seconds.setdefault(st.name, 0.0)

        for bi, block in enumerate(self.source.blocks()):
            if max_blocks is not None and bi >= max_blocks:
                break
            stats.blocks += 1
            stats.samples_in += len(block)
            x = block
            for st in self.stages:
                if profile:
                    t0 = time.perf_counter()
                if st.is_tap:
                    st.fn(x)
                else:
                    x = st.fn(x)
                if profile:
                    stats.per_stage_seconds[st.name] += time.perf_counter() - t0
            if sink is not None:
                sink(x)
            else:
                results.append(x)

        if profile:
            return results, stats
        return results

    def stream(self, max_blocks=None) -> Iterator[np.ndarray]:
        """Run as a generator, yielding each processed block lazily.

        This is the bridge to the generator-chain style: a Pipeline can be
        consumed lazily, so it composes with other generators and stays
        memory-friendly for long/continuous streams.
        """
        for bi, block in enumerate(self.source.blocks()):
            if max_blocks is not None and bi >= max_blocks:
                break
            yield self.process_block(block)

    def __repr__(self):
        return f"Pipeline({len(self.stages)} stages, source={self.source!r})"
