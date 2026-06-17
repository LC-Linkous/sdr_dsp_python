"""Sinks: where results go. WAV audio, processed-IQ files, and plots.

write_wav and write_iq have no extra deps. The plot helpers need matplotlib
(the `plotting` extra) and are imported lazily so importing this package never
requires matplotlib.
"""

from .wav_sink import write_wav
from .iq_sink import write_iq
from .tx_sink import TXSink, LoopbackSink

__all__ = ["write_wav", "write_iq", "TXSink", "LoopbackSink", "plot_spectrum", "plot_spectrogram"]


def __getattr__(name):
    if name in ("plot_spectrum", "plot_spectrogram"):
        from . import plot_sink
        return getattr(plot_sink, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
