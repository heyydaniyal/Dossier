"""Deterministic keyed hashing: opaque IDs and per-entity pseudo-random numbers.

Why keyed hashes instead of a seeded RNG stream: a stream assigns draws in row order, so the
result depends on how the data happen to be sorted, and sorting by ID is banned (D5: ID order
leaks the label). A keyed hash gives every (seed, purpose, key) its own value, independent of
row order and of ID order, and is byte-reproducible across machines and library versions
(blake2b is in the standard library; the mixing below is pure integer arithmetic).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _key(purpose: str, seed: int) -> bytes:
    return hashlib.blake2b(f"{purpose}|{seed}".encode(), digest_size=32).digest()


def keyed_u64(values: Iterable[str], purpose: str, seed: int) -> np.ndarray:
    """64-bit keyed hash of each string (order-free; same input -> same output)."""
    k = _key(purpose, seed)
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(v.encode(), digest_size=8, key=k).digest(), "little")
            for v in values
        ),
        dtype=np.uint64,
    )


def keyed_hex(value: str, purpose: str, seed: int, n: int = 16) -> str:
    k = _key(purpose, seed)
    return hashlib.blake2b(value.encode(), digest_size=n // 2, key=k).hexdigest()


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK
        return z ^ (z >> np.uint64(31))


def uniforms(h: np.ndarray, stream: str) -> np.ndarray:
    """Independent U[0,1) per element for a named stream, derived from keyed hashes h."""
    s = np.uint64(
        int.from_bytes(hashlib.blake2b(stream.encode(), digest_size=8).digest(), "little")
    )
    z = _splitmix64(h.astype(np.uint64) ^ s)
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def choose(u: np.ndarray, options: list[str], weights: list[float]) -> np.ndarray:
    """Inverse-CDF categorical draw from uniforms (weights normalised)."""
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    cdf = np.cumsum(w / w.sum())
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, u, side="right")
    return np.asarray(options, dtype=object)[np.minimum(idx, len(options) - 1)]
