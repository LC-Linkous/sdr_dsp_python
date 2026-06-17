"""File I/O: SigMF read/write bridging files to complex64 arrays, with
annotation support (detect -> label -> save -> reload)."""

from .sigmf import (read_meta, load_iq, save_iq, iq_info,
                    Annotation, read_annotations, bursts_to_annotations)

__all__ = ["read_meta", "load_iq", "save_iq", "iq_info",
           "Annotation", "read_annotations", "bursts_to_annotations"]
