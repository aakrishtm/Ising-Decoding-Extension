"""Metrics for the first dual-rail decoder pipeline."""

from __future__ import annotations

import torch


def binary_confusion_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, int]:
    preds = (torch.sigmoid(logits) >= float(threshold)).to(torch.bool)
    truth = targets.to(torch.bool)
    return {
        "tp": int(torch.logical_and(preds, truth).sum().item()),
        "fp": int(torch.logical_and(preds, ~truth).sum().item()),
        "fn": int(torch.logical_and(~preds, truth).sum().item()),
        "tn": int(torch.logical_and(~preds, ~truth).sum().item()),
    }


def binary_summary_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = max(int(tp) + int(fp) + int(fn) + int(tn), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "target_density": float((tp + fn) / total),
        "prediction_density": float((tp + fp) / total),
    }


def binary_precision_recall(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    counts = binary_confusion_counts(logits, targets, threshold=threshold)
    summary = binary_summary_from_counts(**counts)
    return {"precision": summary["precision"], "recall": summary["recall"]}


def syndrome_density(tensor: torch.Tensor, channels: tuple[int, ...] = (0, 1)) -> float:
    selected = tensor[:, list(channels)]
    return float((selected > 0).float().mean().item())


def logical_error_rate(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predicted.to(torch.uint8) != labels.to(torch.uint8)).float().mean().item())
