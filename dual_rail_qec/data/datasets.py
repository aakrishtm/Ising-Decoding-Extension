"""NPZ shard reader for dual-rail QEC datasets."""

from __future__ import annotations

import json
from bisect import bisect_right
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
        self._shard_lengths = [self._validate_shard(path) for path in self.shard_paths]
        self._offsets = np.cumsum([0, *self._shard_lengths]).astype(np.int64)
        self._cache_index: int | None = None
        self._cache_data: dict[str, np.ndarray] | None = None
        expected = int(self.metadata.get("num_samples", sum(self._shard_lengths)))
        actual = int(sum(self._shard_lengths))
        if expected != actual:
            raise ValueError(f"metadata num_samples={expected} but shards contain {actual} samples")

    def _validate_shard(self, path: Path) -> int:
        expected_input = tuple(self.metadata.get("input_shape", ()))
        expected_target = tuple(self.metadata.get("target_shape", ()))
        with np.load(path) as data:
            for key in ("inputs", "targets", "logical_labels"):
                if key not in data:
                    raise ValueError(f"{path} is missing required array {key!r}")
            inputs = data["inputs"]
            targets = data["targets"]
            labels = data["logical_labels"]
            if expected_input and tuple(inputs.shape[1:]) != expected_input:
                raise ValueError(f"{path} inputs shape {inputs.shape[1:]} != metadata {expected_input}")
            if expected_target and tuple(targets.shape[1:]) != expected_target:
                raise ValueError(f"{path} targets shape {targets.shape[1:]} != metadata {expected_target}")
            if inputs.shape[0] != targets.shape[0] or inputs.shape[0] != labels.shape[0]:
                raise ValueError(f"{path} sample counts differ across inputs/targets/logical_labels")
            if labels.ndim != 2 or labels.shape[1] != 1:
                raise ValueError(f"{path} logical_labels must have shape (N, 1), got {labels.shape}")
            return int(inputs.shape[0])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        for path in self.shard_paths:
            with np.load(path) as data:
                yield {
                    "inputs": np.asarray(data["inputs"]),
                    "targets": np.asarray(data["targets"]),
                    "logical_labels": np.asarray(data["logical_labels"]),
                }

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == shard_index and self._cache_data is not None:
            return self._cache_data
        path = self.shard_paths[int(shard_index)]
        with np.load(path) as data:
            shard = {
                "inputs": np.asarray(data["inputs"]),
                "targets": np.asarray(data["targets"]),
                "logical_labels": np.asarray(data["logical_labels"]),
            }
        self._cache_index = int(shard_index)
        self._cache_data = shard
        return shard

    def get_sample(self, idx: int) -> dict[str, np.ndarray]:
        if int(idx) < 0:
            idx = len(self) + int(idx)
        if int(idx) < 0 or int(idx) >= len(self):
            raise IndexError(idx)
        shard_index = bisect_right(self._offsets, int(idx)) - 1
        local_index = int(idx) - int(self._offsets[shard_index])
        shard = self._load_shard(shard_index)
        return {
            "inputs": shard["inputs"][local_index],
            "targets": shard["targets"][local_index],
            "logical_labels": shard["logical_labels"][local_index],
        }
