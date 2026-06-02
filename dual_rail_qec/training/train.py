"""Train the dual-rail 3D CNN on pre-generated shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dual_rail_qec.data.datasets import DualRailShardDataset
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder
from dual_rail_qec.models.losses import local_fault_bce_loss
from dual_rail_qec.training.metrics import binary_precision_recall


class DualRailTorchDataset(Dataset):
    """Materialize shard arrays into a simple torch Dataset."""

    def __init__(self, dataset_dir: str | Path):
        shard_dataset = DualRailShardDataset(dataset_dir)
        self.metadata = shard_dataset.metadata
        self.inputs = []
        self.targets = []
        self.logical_labels = []
        for shard in shard_dataset.iter_shards():
            self.inputs.append(shard["inputs"])
            self.targets.append(shard["targets"])
            self.logical_labels.append(shard["logical_labels"])
        self.inputs_np = np.concatenate(self.inputs, axis=0)
        self.targets_np = np.concatenate(self.targets, axis=0)
        self.labels_np = np.concatenate(self.logical_labels, axis=0)

    def __len__(self) -> int:
        return int(self.inputs_np.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs_np[idx]).float(),
            "targets": torch.from_numpy(self.targets_np[idx]).float(),
            "logical_labels": torch.from_numpy(self.labels_np[idx]).to(torch.uint8),
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
    device: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = DualRailTorchDataset(dataset_dir)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = DualRailCNN3DPreDecoder(hidden_channels=hidden_channels, depth=depth).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    history = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_batches = 0
        last_logits = None
        last_targets = None
        for batch in loader:
            inputs = batch["inputs"].to(dev)
            targets = batch["targets"].to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = local_fault_bce_loss(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
            last_logits = logits.detach()
            last_targets = targets.detach()

        metrics = binary_precision_recall(last_logits, last_targets) if last_logits is not None else {}
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_batches, 1),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    checkpoint = output_dir / "dual_rail_cnn3d_predecoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": ds.metadata,
            "model": {
                "in_channels": 7,
                "out_channels": 4,
                "hidden_channels": int(hidden_channels),
                "depth": int(depth),
                "kernel_size": 3,
            },
            "history": history,
        },
        checkpoint,
    )
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
    parser.add_argument("--device", type=str, default=None)
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
        device=args.device,
    )
    print(f"Wrote checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

