"""Streaming orchestration over the pure DSP core.

The Pipeline threads blocks from a source through block->block stages to a
sink, with taps for live observation. It adds flow control, not DSP -- every
stage is a library function you could call by hand.

A block boundary is not a signal boundary, though: the core functions are
stateless, so calling them once per block restarts every delay line at every
block. The ``Streaming*`` stages wrap the core functions and carry exactly the
state each needs between calls, so a live chain built from them produces the
same samples as the whole record processed at once. Use them for anything that
runs on a live source.
"""

from .pipeline import Pipeline, Stage, PipelineStats
from .stages import (StreamingFIR, StreamingDecimator, StreamingMixer,
                     StreamingFMDemod, StreamingResampler, StreamingDeemphasis)

__all__ = ["Pipeline", "Stage", "PipelineStats",
           "StreamingFIR", "StreamingDecimator", "StreamingMixer",
           "StreamingFMDemod", "StreamingResampler", "StreamingDeemphasis"]
