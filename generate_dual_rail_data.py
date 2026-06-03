# SPDX-License-Identifier: Apache-2.0
"""Generate dual-rail erasure dataset artifacts.

The default path writes the project's pre-generated NPZ shard dataset:

    datasets/dual_rail_d{distance}/metadata.json
    datasets/dual_rail_d{distance}/shards/shard_00000.npz

When Stim is available, the generator also writes sparse detector samples:

    datasets/dual_rail_d{distance}/samples_X.dets
    datasets/dual_rail_d{distance}/samples_Z.dets
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import stim
except ImportError:
    stim = None

from dual_rail_qec.data.export import write_dataset
from dual_rail_qec.data.simulator import build_base_surface_code_circuit


def build_dual_rail_circuit(d, rounds, p_erasure, p_pauli):
    """Build the basis-specific base Stim circuits used by the dual-rail writer.

    Erasures are sampled per shot and converted into probability-one Pauli error
    instructions by ``write_dataset``. They are not inserted as measurement
    instructions here, so Stim ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` rec
    offsets stay valid.
    """
    if stim is None:
        raise RuntimeError("stim is required for dual-rail circuit generation.")
    _ = p_erasure
    return {
        "X": build_base_surface_code_circuit(d, rounds, "X", p_pauli),
        "Z": build_base_surface_code_circuit(d, rounds, "Z", p_pauli),
    }


def write_offline_samples(
    *,
    output_dir: Path,
    distance: int,
    rounds: int,
    num_shots: int,
    p_erasure: float,
    p_pauli: float,
    p_measure: float,
    p_false_positive: float,
    p_false_negative: float,
    p_ambiguity: float,
    seed: int,
    compress: bool,
) -> Path:
    """Write one Stim-backed dataset shard with NVIDIA-style ``.dets`` files."""
    return write_dataset(
        output_root=output_dir,
        distance=distance,
        rounds=rounds,
        num_shards=1,
        samples_per_shard=num_shots,
        p_erasure=p_erasure,
        p_pauli=p_pauli,
        p_measure=p_measure,
        p_false_positive=p_false_positive,
        p_false_negative=p_false_negative,
        p_ambiguity=p_ambiguity,
        seed=seed,
        data_source="stim",
        bases=("X", "Z"),
        compress=compress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dual-rail erasure data for the local 7-channel decoder pipeline."
    )
    parser.add_argument("--distance", "-d", type=int, required=True, help="Surface-code distance.")
    parser.add_argument("--rounds", "--n-rounds", dest="rounds", type=int, required=True, help="Syndrome rounds.")
    parser.add_argument("--p-erasure", type=float, default=0.0, help="Dual-rail erasure probability.")
    parser.add_argument("--p-pauli", type=float, default=0.0, help="Residual Pauli error probability.")
    parser.add_argument("--p-measure", type=float, default=0.0, help="Stim measurement-flip probability.")
    parser.add_argument("--p-false-positive", type=float, default=0.0, help="Erasure flag fires without physical erasure.")
    parser.add_argument("--p-false-negative", type=float, default=0.0, help="Physical erasure is missed by telemetry.")
    parser.add_argument("--p-ambiguity", type=float, default=0.0, help="Ambiguous readout probability.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of NPZ shards to write.")
    parser.add_argument("--samples-per-shard", type=int, default=1024, help="Samples in each NPZ shard.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic generation.")
    parser.add_argument("--num-shots", type=int, default=262144, help="Number of detector-sample shots for stim-samples mode.")
    parser.add_argument("--data-source", choices=("auto", "stim", "synthetic"), default="auto")
    parser.add_argument("--basis", action="append", choices=("X", "Z"), default=None)
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ output.")
    parser.add_argument(
        "--artifact",
        choices=("dataset", "stim-samples", "stim-stub"),
        default="dataset",
        help="Write NPZ dataset shards, or write one Stim-backed shard with samples_X/Z.dets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Dataset output root, or Stim sample output root in stim-samples mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.artifact in {"stim-samples", "stim-stub"}:
        dataset_dir = write_offline_samples(
            output_dir=args.output_dir,
            distance=args.distance,
            rounds=args.rounds,
            num_shots=args.num_shots,
            p_erasure=args.p_erasure,
            p_pauli=args.p_pauli,
            p_measure=args.p_measure,
            p_false_positive=args.p_false_positive,
            p_false_negative=args.p_false_negative,
            p_ambiguity=args.p_ambiguity,
            seed=args.seed,
            compress=args.compress,
        )
        print(f"Wrote dual-rail Stim samples: {dataset_dir}")
        return

    dataset_dir = write_dataset(
        output_root=args.output_dir,
        distance=args.distance,
        rounds=args.rounds,
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        p_erasure=args.p_erasure,
        p_pauli=args.p_pauli,
        p_measure=args.p_measure,
        p_false_positive=args.p_false_positive,
        p_false_negative=args.p_false_negative,
        p_ambiguity=args.p_ambiguity,
        seed=args.seed,
        data_source=args.data_source,
        bases=tuple(args.basis or ("X", "Z")),
        compress=args.compress,
    )
    print(f"Wrote dual-rail dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
