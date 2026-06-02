"""Write dual-rail pre-generated dataset shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dual_rail_qec.data.simulator import (
    generate_synthetic_events,
    logical_label_from_targets,
)
from dual_rail_qec.telemetry.tensorize import NUM_INPUT_CHANNELS, make_local_targets, tensorize_events


def build_dataset_metadata(
    *,
    distance: int,
    rounds: int,
    num_shards: int,
    samples_per_shard: int,
    p_erasure: float,
    p_pauli: float,
    p_ambiguity: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "dual_rail_qec_dataset",
        "distance": int(distance),
        "rounds": int(rounds),
        "input_shape": [NUM_INPUT_CHANNELS, int(rounds), int(distance), int(distance)],
        "target_shape": [4, int(rounds), int(distance), int(distance)],
        "num_shards": int(num_shards),
        "samples_per_shard": int(samples_per_shard),
        "num_samples": int(num_shards) * int(samples_per_shard),
        "noise": {
            "p_erasure": float(p_erasure),
            "p_pauli": float(p_pauli),
            "p_ambiguity": float(p_ambiguity),
        },
        "seed": int(seed),
        "format": {
            "inputs": "float32[N,7,T,H,W]",
            "targets": "float32[N,4,T,H,W]",
            "logical_labels": "uint8[N,1]",
        },
    }


def write_dataset(
    *,
    output_root: Path,
    distance: int,
    rounds: int,
    num_shards: int,
    samples_per_shard: int,
    p_erasure: float,
    p_pauli: float,
    p_ambiguity: float = 0.0,
    seed: int = 0,
) -> Path:
    """Write a synthetic dual-rail dataset and return its dataset directory."""
    dataset_dir = Path(output_root) / f"dual_rail_d{int(distance)}"
    shard_dir = dataset_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_dataset_metadata(
        distance=distance,
        rounds=rounds,
        num_shards=num_shards,
        samples_per_shard=samples_per_shard,
        p_erasure=p_erasure,
        p_pauli=p_pauli,
        p_ambiguity=p_ambiguity,
        seed=seed,
    )
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    rng = np.random.default_rng(int(seed))
    for shard_idx in range(int(num_shards)):
        inputs = np.zeros(
            (int(samples_per_shard), NUM_INPUT_CHANNELS, int(rounds), int(distance), int(distance)),
            dtype=np.float32,
        )
        targets = np.zeros(
            (int(samples_per_shard), 4, int(rounds), int(distance), int(distance)),
            dtype=np.float32,
        )
        logical_labels = np.zeros((int(samples_per_shard), 1), dtype=np.uint8)

        for sample_idx in range(int(samples_per_shard)):
            events = generate_synthetic_events(
                distance=distance,
                rounds=rounds,
                rng=rng,
                p_erasure=p_erasure,
                p_pauli=p_pauli,
                p_ambiguity=p_ambiguity,
            )
            sample_inputs = tensorize_events(events, distance=distance, rounds=rounds)
            sample_targets = make_local_targets(sample_inputs)
            inputs[sample_idx] = sample_inputs
            targets[sample_idx] = sample_targets
            logical_labels[sample_idx] = logical_label_from_targets(sample_targets)

        np.savez_compressed(
            shard_dir / f"shard_{shard_idx:05d}.npz",
            inputs=inputs,
            targets=targets,
            logical_labels=logical_labels,
        )

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic dual-rail QEC dataset shards.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--distance", "-d", type=int, required=True)
    parser.add_argument("--rounds", "--n-rounds", dest="rounds", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--p-erasure", type=float, default=0.01)
    parser.add_argument("--p-pauli", type=float, default=0.001)
    parser.add_argument("--p-ambiguity", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = write_dataset(
        output_root=args.output_root,
        distance=args.distance,
        rounds=args.rounds,
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        p_erasure=args.p_erasure,
        p_pauli=args.p_pauli,
        p_ambiguity=args.p_ambiguity,
        seed=args.seed,
    )
    print(f"Wrote dual-rail dataset: {dataset_dir}")


if __name__ == "__main__":
    main()

