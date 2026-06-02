"""Validate dual-rail dataset artifacts before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dual_rail_qec.data.datasets import DualRailShardDataset


def _density_from_dataset(dataset_dir: Path) -> dict[str, float]:
    ds = DualRailShardDataset(dataset_dir)
    total = 0
    syndrome_sum = 0.0
    target_sum = 0.0
    erasure_sum = 0.0
    for shard in ds.iter_shards():
        inputs = shard["inputs"]
        targets = shard["targets"]
        n = int(inputs.shape[0])
        total += n
        syndrome_sum += float(np.mean(inputs[:, 0] + inputs[:, 1])) * n
        erasure_sum += float(np.mean(inputs[:, 2] + inputs[:, 3])) * n
        target_sum += float(np.mean(targets)) * n
    denom = max(total, 1)
    return {
        "syndrome_density": syndrome_sum / denom,
        "erasure_density": erasure_sum / denom,
        "target_density": target_sum / denom,
    }


def _check_stim_sidecars(dataset_dir: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    bases = [str(b).upper() for b in metadata.get("bases", ("X", "Z"))]
    for basis in bases:
        for name in (f"samples_{basis}.dets", f"erasures_{basis}.npz"):
            if not (dataset_dir / name).exists():
                errors.append(f"missing required sidecar: {name}")
        manifest_path = dataset_dir / f"erasures_{basis}.npz"
        if manifest_path.exists():
            with np.load(manifest_path, allow_pickle=True) as manifest:
                if "shard_files" not in manifest:
                    errors.append(f"{manifest_path.name} missing shard_files")
                    continue
                for raw in manifest["shard_files"]:
                    rel = str(raw)
                    if not (dataset_dir / rel).exists():
                        errors.append(f"{manifest_path.name} references missing shard sidecar {rel}")


def validate_dataset(
    dataset_dir: str | Path,
    *,
    require_stim_sidecars: bool = True,
    require_nonzero_erasure: bool = False,
    require_nonzero_syndrome: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    ds = DualRailShardDataset(dataset_dir)
    metadata = ds.metadata
    errors: list[str] = []

    if not (dataset_dir / "metadata.json").exists():
        errors.append("missing metadata.json")
    if not (dataset_dir / "shards" / "shard_00000.npz").exists():
        errors.append("missing shards/shard_00000.npz")

    if require_stim_sidecars and str(metadata.get("data_source", "")).lower() == "stim":
        _check_stim_sidecars(dataset_dir, metadata, errors)

    densities = _density_from_dataset(dataset_dir)
    if require_nonzero_erasure and densities["erasure_density"] <= 0.0:
        errors.append("erasure channels are empty")
    if require_nonzero_syndrome and densities["syndrome_density"] <= 0.0:
        errors.append("syndrome channels are empty")

    result = {
        "dataset_dir": str(dataset_dir),
        "metadata": {
            "artifact": metadata.get("artifact"),
            "data_source": metadata.get("data_source"),
            "distance": metadata.get("distance"),
            "rounds": metadata.get("rounds"),
            "num_samples": metadata.get("num_samples"),
        },
        "densities": densities,
        "errors": errors,
        "ok": not errors,
    }
    if errors:
        raise ValueError(json.dumps(result, indent=2, sort_keys=True))
    return result


def compare_coupling(
    baseline_dir: str | Path,
    erasure_dir: str | Path,
    *,
    min_delta: float = 1e-6,
) -> dict[str, Any]:
    baseline = _density_from_dataset(Path(baseline_dir))
    erasure = _density_from_dataset(Path(erasure_dir))
    syndrome_delta = erasure["syndrome_density"] - baseline["syndrome_density"]
    target_delta = erasure["target_density"] - baseline["target_density"]
    ok = abs(syndrome_delta) > float(min_delta) or abs(target_delta) > float(min_delta)
    result = {
        "baseline": baseline,
        "erasure": erasure,
        "syndrome_density_delta": float(syndrome_delta),
        "target_density_delta": float(target_delta),
        "ok": ok,
    }
    if not ok:
        raise ValueError(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dual-rail dataset artifacts.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--require-nonzero-erasure", action="store_true")
    parser.add_argument("--require-nonzero-syndrome", action="store_true")
    parser.add_argument("--no-stim-sidecars", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_dataset(
        args.dataset_dir,
        require_stim_sidecars=not args.no_stim_sidecars,
        require_nonzero_erasure=args.require_nonzero_erasure,
        require_nonzero_syndrome=args.require_nonzero_syndrome,
    )
    if args.baseline_dir is not None:
        result["coupling"] = compare_coupling(
            args.baseline_dir,
            args.dataset_dir,
            min_delta=args.min_delta,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
