"""IQ sources: adapters that feed complex64 into the DSP core.

The library ships only device-free sources: ArraySource (wrap an array) and
FileSource (read a SigMF recording). Both depend on nothing but numpy. There is
deliberately NO device source here -- acquiring samples from specific hardware
is an application concern. To drive the DSP from an SDR, implement the IQSource
protocol in your own code (see examples/hackrf_capture.py for a reference that
wraps a HackRF). The library provides the hooks; you provide the hardware.
"""

from .base import IQSource, ArraySource
from .file_source import FileSource

__all__ = ["IQSource", "ArraySource", "FileSource"]
