"""PyMatching bridge placeholder for residual syndrome experiments."""

from __future__ import annotations


def require_pymatching():
    try:
        import pymatching  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pymatching is not installed. Install the repo inference requirements "
            "before running the global-decoder bridge."
        ) from exc
    return pymatching


def decode_with_pymatching_stub(*args, **kwargs):
    """Reserved hook for the erasure-aware PyMatching integration."""
    require_pymatching()
    raise NotImplementedError(
        "PyMatching bridge needs the finalized residual syndrome graph and erasure weighting contract."
    )

