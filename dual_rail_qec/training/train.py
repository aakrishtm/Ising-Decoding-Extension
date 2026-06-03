"""Train the dual-rail 3D CNN on pre-generated shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dual_rail_qec.data.datasets import DualRailShardDataset
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder
from dual_rail_qec.models.losses import local_fault_bce_loss
from dual_rail_qec.training.metrics import binary_confusion_counts, binary_summary_from_counts


class DualRailTorchDataset(Dataset):
    """Torch Dataset backed by lazy NPZ shard reads."""

    def __init__(self, dataset_dir: str | Path):
        self.shard_dataset = DualRailShardDataset(dataset_dir)
        self.metadata = self.shard_dataset.metadata

    def __len__(self) -> int:
        return len(self.shard_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.shard_dataset.get_sample(idx)
        return {
            "inputs": torch.from_numpy(sample["inputs"]).float(),
            "targets": torch.from_numpy(sample["targets"]).float(),
            "logical_labels": torch.from_numpy(sample["logical_labels"]).to(torch.uint8),
        }


def train(
    *,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_channels: int,
    depth: int,
    num_workers: int = 0,
    shuffle: bool = False,
    precision: str = "auto",
    device: str | None = None,
    pos_weight: float | None = None,
    threshold: float = 0.5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = DualRailTorchDataset(dataset_dir)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
    )
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = DualRailCNN3DPreDecoder(hidden_channels=hidden_channels, depth=depth).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    precision_norm = str(precision).strip().lower()
    if precision_norm == "auto":
        precision_norm = "bf16" if dev.type == "cuda" else "fp32"
    amp_enabled = dev.type == "cuda" and precision_norm in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision_norm == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and precision_norm == "fp16")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and precision_norm == "fp16")

    history = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_batches = 0
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for batch in loader:
            inputs = batch["inputs"].to(dev)
            targets = batch["targets"].to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(inputs)
                loss = local_fault_bce_loss(logits, targets, pos_weight=pos_weight)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
            with torch.no_grad():
                batch_counts = binary_confusion_counts(logits.detach(), targets.detach(), threshold=threshold)
            for key, value in batch_counts.items():
                counts[key] += int(value)

        metrics = binary_summary_from_counts(**counts)
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_batches, 1),
            "threshold": float(threshold),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    checkpoint = output_dir / "dual_rail_cnn3d_predecoder.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": ds.metadata,
        "model": {
            "in_channels": 7,
            "out_channels": 4,
            "hidden_channels": int(hidden_channels),
            "depth": int(depth),
            "kernel_size": 3,
            "precision": precision_norm,
            "pos_weight": None if pos_weight is None else float(pos_weight),
            "threshold": float(threshold),
        },
        "history": history,
    }
    torch.save(payload, checkpoint)
    latest = output_dir / "latest.pt"
    torch.save(payload, latest)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-rail 3D CNN pre-decoder.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dual_rail_train"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Global sample shuffle; slower for very large NPZ shard sets.")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive-class BCE weight for sparse local targets.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for reported precision/recall.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        precision=args.precision,
        device=args.device,
        pos_weight=args.pos_weight,
        threshold=args.threshold,
    )
    print(f"Wrote checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
