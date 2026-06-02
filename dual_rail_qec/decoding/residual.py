"""Residual syndrome construction for local pre-decoder predictions."""

from __future__ import annotations

import torch


def threshold_corrections(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert 4-channel logits into binary local correction candidates."""
    return (torch.sigmoid(logits) >= float(threshold)).to(torch.uint8)


def residual_syndrome_inputs(inputs: torch.Tensor, correction_candidates: torch.Tensor) -> torch.Tensor:
    """Build a simple residual input tensor after local correction candidates.

    This is a local XOR-style placeholder for the first training loop. Full
    surface-code residual construction should replace it once the exact target
    semantics are finalized.
    """
    residual = inputs.clone()
    residual[:, 0] = torch.logical_xor(
        residual[:, 0].to(torch.bool),
        correction_candidates[:, 1].to(torch.bool),
    ).to(residual.dtype)
    residual[:, 1] = torch.logical_xor(
        residual[:, 1].to(torch.bool),
        correction_candidates[:, 0].to(torch.bool),
    ).to(residual.dtype)
    return residual


def logical_prediction_from_corrections(correction_candidates: torch.Tensor) -> torch.Tensor:
    """Return one parity-style logical prediction per sample."""
    parity = correction_candidates[:, 0].sum(dim=(1, 2, 3)) + correction_candidates[:, 1].sum(dim=(1, 2, 3))
    return (parity % 2).to(torch.uint8).view(-1, 1)

