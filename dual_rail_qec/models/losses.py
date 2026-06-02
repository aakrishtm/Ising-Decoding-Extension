"""Loss functions for dual-rail local fault identification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def local_fault_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """Binary cross entropy over local 4-channel fault/correction targets."""
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = torch.full(
            (logits.shape[1],),
            float(pos_weight),
            dtype=logits.dtype,
            device=logits.device,
        ).view(1, logits.shape[1], 1, 1, 1)
    return F.binary_cross_entropy_with_logits(logits, targets, **kwargs)

