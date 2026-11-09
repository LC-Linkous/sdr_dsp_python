"""Argument checks with uniform messages: parameter name, the value given, and
what would have been valid. Internal; every public function that validates
input uses these so a caller sees the same shape of message everywhere and
never a numpy or scipy internal ("Parameter Nx=0 is not a positive integer",
"slice step cannot be zero", "float modulo by zero").
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    """A finite number > 0 (>= 0 with allow_zero). Returns it as float."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a number, got the string {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return v


def positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    """An integer >= minimum (integral floats and numpy ints accepted)."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        if value != int(value):
            raise ValueError
        v = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return v


def one_of(name: str, value: str, choices: Iterable[str]) -> str:
    """Membership in a fixed set of strings, with the set in the message."""
    if value not in choices:
        opts = ", ".join(repr(c) for c in choices)
        raise ValueError(f"{name} must be one of {opts}, got {value!r}")
    return value


def bits(name: str, value: Any) -> np.ndarray:
    """A 1-D array of 0/1 (bool, int, list all accepted). Returns uint8."""
    arr = np.asarray(value)
    if arr.dtype == bool:
        return arr.astype(np.uint8).ravel()
    if arr.ndim != 1 and arr.size:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size and (not np.issubdtype(arr.dtype, np.number)
                     or not np.all((arr == 0) | (arr == 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return arr.astype(np.uint8)


def pair(name: str, value: Any) -> tuple[float, float]:
    """A (lo, hi) pair of numbers with lo <= hi."""
    try:
        lo, hi = value
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a (low, high) pair of numbers, got {value!r}") from None
    if lo > hi:
        raise ValueError(f"{name} must satisfy low <= high, got {value!r}")
    return lo, hi


def byteslike(name: str, value: Any) -> bytes:
    """bytes/bytearray/memoryview; str is refused with a hint."""
    if isinstance(value, str):
        raise TypeError(f"{name} must be bytes, not str -- encode it first "
                        f"({name}.encode('utf-8'))")
    try:
        return bytes(value)
    except TypeError:
        raise TypeError(f"{name} must be bytes-like, got {type(value).__name__}") from None
