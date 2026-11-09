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
from typing import TYPE_CHECKING, Any, Callable, Iterator, Self

import numpy as np

if TYPE_CHECKING:
    from ..sources.base import IQSource


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

    def __str__(self) -> str:
        lines = [f"{self.blocks} blocks, {self.samples_in:,} input samples"]
        for name, secs in self.per_stage_seconds.items():
            lines.append(f"  {name:20s} {secs*1e3:8.2f} ms total")
        return "\n".join(lines)


class Pipeline:
    """A source + an ordered chain of block->block stages (and taps).

    Build declaratively and run:

        pipe = (Pipeline(source)
                .add(StreamingFIR(taps), "filter")
                .add(StreamingFMDemod(75000, fs), "demod")
                .tap(lambda b: meter.update(b)))      # live peek, flow unchanged
        audio = pipe.run()                            # or run(sink=write_audio)

    Stages transform the block; taps observe it and return nothing (the original
    block continues). Order matters and is preserved.

    Use the ``Streaming*`` stages (``sdr_dsp.stream.stages``) for anything with
    memory -- filters, mixers, the FM discriminator, resamplers. A bare core
    function such as ``lambda b: fir_apply(b, taps)`` restarts its delay line
    at every block and clicks at every seam; it is fine only for stages that
    are genuinely per-sample (``to_db``, ``am_demod`` without dc_block, a
    slicer) or for taps.
    """

    def __init__(self, source: IQSource) -> None:
        self.source = source
        self.stages: list[Stage] = []

    # -- building -----------------------------------------------------------
    def add(self, fn: Callable[[np.ndarray], np.ndarray], name: str | None = None) -> Self:
        """Append a transforming stage (block -> block)."""
        # mypy keeps `None` in the type of `name or <Any>`; the fallback
        # default guarantees a str at runtime.
        self.stages.append(
            Stage(name or getattr(fn, "__name__", "stage"), fn))  # type: ignore[arg-type]
        return self

    def tap(self, fn: Callable[[np.ndarray], Any], name: str | None = None) -> Self:
        """Append an observing stage (block -> ignored). Flow is unchanged.

        A tap is how live display attaches: fn receives the current block and
        does whatever it likes (update a plot, accumulate a message) without
        affecting what the next stage sees.
        """
        # see add(): the `or` with an Any right-hand side confuses mypy
        self.stages.append(Stage(name or getattr(fn, "__name__", "tap"),  # type: ignore[arg-type]
                                 fn, is_tap=True))
        return self

    def describe(self) -> str:
        """Return the chain as inspectable text (the pipeline is data)."""
        rows = [f"source: {self.source!r}"]
        for i, st in enumerate(self.stages):
            kind = "tap " if st.is_tap else "step"
            rows.append(f"  [{i}] {kind} {st.name}")
        return "\n".join(rows)

    # -- running ------------------------------------------------------------
    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Thread a single block through every stage. Useful for testing and
        for driving the pipeline from an external loop (e.g. a GUI timer)."""
        x = block
        for st in self.stages:
            if st.is_tap:
                st.fn(x)              # observe, don't replace
            else:
                x = st.fn(x)
        return x

    def flush(self) -> list[np.ndarray]:
        """Drain the look-ahead held inside stages once the stream has ended.

        Stages with internal look-ahead (``StreamingResampler``) hold back the
        last few output samples until they know the stream is over; without a
        flush that tail is silently lost. This walks the chain in order: each
        stage that exposes ``flush()`` is asked for its tail, and that tail is
        threaded through every LATER stage (taps included) exactly like a
        normal block -- so a downstream stage with its own look-ahead receives
        the upstream tail before being flushed itself.

        Returns the list of non-empty processed tail blocks (usually 0 or 1).
        ``run()`` and ``stream()`` call this when the source is exhausted;
        call it yourself only if you drive ``process_block`` by hand.
        Flushing twice is harmless: a drained stage returns nothing.
        """
        tails: list[np.ndarray] = []
        for i, st in enumerate(self.stages):
            fl = getattr(st.fn, "flush", None)
            if st.is_tap or fl is None:
                continue
            x = fl()
            if x is None or len(x) == 0:
                continue
            for later in self.stages[i + 1:]:
                if later.is_tap:
                    later.fn(x)
                else:
                    x = later.fn(x)
            if len(x):
                tails.append(x)
        return tails

    def run(
        self,
        sink: Callable[[np.ndarray], Any] | None = None,
        profile: bool = False,
        max_blocks: int | None = None,
    ) -> list[np.ndarray] | None | tuple[list[np.ndarray] | None, PipelineStats]:
        """Pull blocks from the source, process each, deliver to sink.

        sink:       callable(result_block) -> None. If None, results are
                    collected and returned as a list.
        profile:    if True, also return PipelineStats (per-stage timing).
        max_blocks: stop after this many blocks (useful for live sources).

        When the source runs dry the stages are flushed (see ``flush``) and
        any tail is delivered like a final block, so a Pipeline run produces
        the same sample count as the whole-record function. Stopping early via
        ``max_blocks`` does NOT flush: the stream was cut, not ended.

        Returns the result list (or None if a sink was given); if profile,
        returns (results_or_None, PipelineStats).
        """
        results: list[np.ndarray] = []
        stats = PipelineStats()
        for st in self.stages:
            stats.per_stage_seconds.setdefault(st.name, 0.0)

        exhausted = True
        for bi, block in enumerate(self.source.blocks()):
            if max_blocks is not None and bi >= max_blocks:
                exhausted = False
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

        if exhausted:
            for x in self.flush():
                if sink is not None:
                    sink(x)
                else:
                    results.append(x)

        out = None if sink is not None else results
        if profile:
            return out, stats
        return out

    def stream(self, max_blocks: int | None = None) -> Iterator[np.ndarray]:
        """Run as a generator, yielding each processed block lazily.

        This is the bridge to the generator-chain style: a Pipeline can be
        consumed lazily, so it composes with other generators and stays
        memory-friendly for long/continuous streams. Like ``run``, it flushes
        the stages' look-ahead when the source is exhausted (not when cut by
        ``max_blocks``) and yields any tail as a final block.
        """
        for bi, block in enumerate(self.source.blocks()):
            if max_blocks is not None and bi >= max_blocks:
                return
            yield self.process_block(block)
        yield from self.flush()

    def __repr__(self) -> str:
        return f"Pipeline({len(self.stages)} stages, source={self.source!r})"
