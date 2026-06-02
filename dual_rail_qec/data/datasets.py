"""NPZ shard reader for dual-rail QEC datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


class DualRailShardDataset:
    """Small numpy-backed reader for pre-generated dual-rail shards."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        with (self.dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.shard_paths = sorted((self.dataset_dir / "shards").glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.npz files found in {self.dataset_dir / 'shards'}")

    def __len__(self) -> int:
        return int(self.metadata["num_samples"])

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        for path in self.shard_paths:
            with np.load(path) as data:
                yield {
                    "inputs": np.asarray(data["inputs"]),
                    "targets": np.asarray(data["targets"]),
                    "logical_labels": np.asarray(data["logical_labels"]),
                }

