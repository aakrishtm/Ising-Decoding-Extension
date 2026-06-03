"""Evaluate a trained dual-rail pre-decoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dual_rail_qec.decoding.residual import (
    logical_prediction_from_corrections,
    residual_syndrome_inputs,
    threshold_corrections,
)
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder
from dual_rail_qec.training.metrics import (
    binary_confusion_counts,
    binary_summary_from_counts,
    logical_error_rate,
    syndrome_density,
)
from dual_rail_qec.training.train import DualRailTorchDataset


def evaluate(
    *,
    dataset_dir: Path,
    checkpoint: Path,
    batch_size: int,
    threshold: float,
    device: str | None = None,
) -> dict[str, float]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint, map_location=dev, weights_only=False)
    model_cfg = payload.get("model", {})
    model = DualRailCNN3DPreDecoder(
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        depth=int(model_cfg.get("depth", 3)),
        in_channels=int(model_cfg.get("in_channels", 7)),
        out_channels=int(model_cfg.get("out_channels", 4)),
    ).to(dev)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    ds = DualRailTorchDataset(dataset_dir)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_ler = 0.0
    total_before = 0.0
    total_after = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(dev)
            targets = batch["targets"].to(dev)
            labels = batch["logical_labels"].to(dev)
            logits = model(inputs)
            candidates = threshold_corrections(logits, threshold=threshold)
            residual = residual_syndrome_inputs(inputs, candidates)
            predicted_logicals = logical_prediction_from_corrections(candidates)

            batch_counts = binary_confusion_counts(logits, targets, threshold=threshold)
            for key, value in batch_counts.items():
                counts[key] += int(value)
            total_ler += logical_error_rate(predicted_logicals, labels)
            total_before += syndrome_density(inputs)
            total_after += syndrome_density(residual)
            batches += 1

    pr = binary_summary_from_counts(**counts)
    metrics = {
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "target_density": pr["target_density"],
        "prediction_density": pr["prediction_density"],
        "logical_error_rate": total_ler / max(batches, 1),
        "syndrome_density_before": total_before / max(batches, 1),
        "syndrome_density_after": total_after / max(batches, 1),
        "threshold": float(threshold),
    }
    return {k: float(v) for k, v in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-rail 3D CNN pre-decoder.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
