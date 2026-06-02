"""Metrics for the first dual-rail decoder pipeline."""

from __future__ import annotations

import torch


def binary_precision_recall(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= float(threshold)).to(torch.bool)
    truth = targets.to(torch.bool)
    tp = torch.logical_and(preds, truth).sum().item()
    fp = torch.logical_and(preds, ~truth).sum().item()
    fn = torch.logical_and(~preds, truth).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": float(precision), "recall": float(recall)}


def syndrome_density(tensor: torch.Tensor, channels: tuple[int, ...] = (0, 1)) -> float:
    selected = tensor[:, list(channels)]
    return float((selected > 0).float().mean().item())


def logical_error_rate(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predicted.to(torch.uint8) != labels.to(torch.uint8)).float().mean().item())

