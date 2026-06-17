"""Streaming orchestration over the pure DSP core.

The Pipeline threads blocks from a source through pure block->block stages to a
sink, with taps for live observation. It adds flow control, not DSP -- every
stage is a library function you could call by hand.
"""

from .pipeline import Pipeline, Stage, PipelineStats

__all__ = ["Pipeline", "Stage", "PipelineStats"]
