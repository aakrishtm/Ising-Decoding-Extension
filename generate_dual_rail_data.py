# SPDX-License-Identifier: Apache-2.0
"""Generate dual-rail erasure dataset artifacts.

The default path writes the project's pre-generated NPZ shard dataset:

    datasets/dual_rail_d{distance}/metadata.json
    datasets/dual_rail_d{distance}/shards/shard_00000.npz

The Stim detector-sample path is intentionally still a stub until the exact
dual-rail circuit model is approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import stim
except ImportError:
    stim = None

from dual_rail_qec.data.export import write_dataset


def build_dual_rail_circuit(d, rounds, p_erasure, p_pauli):
    """Build a surface-code memory circuit with dual-rail erasure boundaries.

    The eventual implementation should construct a rotated surface-code memory
    experiment for the requested code distance and number of syndrome rounds,
    inject explicit dual-rail erasure events at rate ``p_erasure``, inject any
    residual Pauli noise at rate ``p_pauli``, and expose erasure locations in a
    form that can be converted into the pre-decoder's extra input channel.

    The body is deliberately left unimplemented during the reconnaissance phase.
    """
    if stim is None:
        raise RuntimeError("stim is required for dual-rail circuit generation.")
    raise NotImplementedError("Dual-rail Stim circuit generation is not implemented yet.")


def _metadata_for_basis(
    *,
    basis: str,
    distance: int,
    rounds: int,
    num_shots: int,
    p_erasure: float,
    p_pauli: float,
    circuit: stim.Circuit | None,
) -> Dict[str, Any]:
    """Return metadata compatible with NVIDIA's offline Stim sample contract."""
    circuit_text = "" if circuit is None else str(circuit)
    circuit_sha256 = hashlib.sha256(circuit_text.encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "artifact": "stim_detector_samples",
        "format": "dets",
        "append_observables": True,
        "distance": distance,
        "n_rounds": rounds,
        "basis": basis,
        "code_rotation": "O1",
        "num_detectors": None,
        "num_observables": None,
        "num_shots": num_shots,
        "p_error": p_pauli,
        "noise_model": "dual-rail-erasure-stub",
        "noise_model_sha256": circuit_sha256,
        "noise_model_params": {
            "p_erasure": p_erasure,
            "p_pauli": p_pauli,
        },
    }


def write_offline_samples(
    *,
    output_dir: Path,
    distance: int,
    rounds: int,
    num_shots: int,
    p_erasure: float,
    p_pauli: float,
) -> None:
    """Outline the offline Stim sample outputs expected by the NVIDIA pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Build separate memory circuits for X and Z bases if the final dual-
    # rail construction requires basis-specific detector layouts.
    circuit = None

    expected_outputs = {
        "X": output_dir / "samples_X.dets",
        "Z": output_dir / "samples_Z.dets",
    }
    metadata = {
        basis: _metadata_for_basis(
            basis=basis,
            distance=distance,
            rounds=rounds,
            num_shots=num_shots,
            p_erasure=p_erasure,
            p_pauli=p_pauli,
            circuit=circuit,
        )
        for basis in expected_outputs
    }

    # TODO: Replace this placeholder with Stim sampler output in sparse .dets
    # format with logical observables appended.
    for path in expected_outputs.values():
        path.touch(exist_ok=True)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write(os.linesep)

    raise NotImplementedError(
        "Output paths and metadata scaffold were created, but sample generation is not implemented."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dual-rail erasure data for the local 7-channel decoder pipeline."
    )
    parser.add_argument("--distance", "-d", type=int, required=True, help="Surface-code distance.")
    parser.add_argument("--rounds", "--n-rounds", dest="rounds", type=int, required=True, help="Syndrome rounds.")
    parser.add_argument("--p-erasure", type=float, default=0.0, help="Dual-rail erasure probability.")
    parser.add_argument("--p-pauli", type=float, default=0.0, help="Residual Pauli error probability.")
    parser.add_argument("--p-ambiguity", type=float, default=0.0, help="Ambiguous readout probability.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of NPZ shards to write.")
    parser.add_argument("--samples-per-shard", type=int, default=1024, help="Samples in each NPZ shard.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic generation.")
    parser.add_argument("--num-shots", type=int, default=262144, help="Number of detector-sample shots for stim-stub mode.")
    parser.add_argument("--data-source", choices=("auto", "stim", "synthetic"), default="auto")
    parser.add_argument("--basis", action="append", choices=("X", "Z"), default=None)
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ output.")
    parser.add_argument(
        "--artifact",
        choices=("dataset", "stim-stub"),
        default="dataset",
        help="Write NPZ dataset shards, or create the legacy Stim sample stub outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Dataset output root, or Stim sample directory in stim-stub mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.artifact == "stim-stub":
        write_offline_samples(
            output_dir=args.output_dir,
            distance=args.distance,
            rounds=args.rounds,
            num_shots=args.num_shots,
            p_erasure=args.p_erasure,
            p_pauli=args.p_pauli,
        )
        return

    dataset_dir = write_dataset(
        output_root=args.output_dir,
        distance=args.distance,
        rounds=args.rounds,
        num_shards=args.num_shards,
        samples_per_shard=args.samples_per_shard,
        p_erasure=args.p_erasure,
        p_pauli=args.p_pauli,
        p_ambiguity=args.p_ambiguity,
        seed=args.seed,
        data_source=args.data_source,
        bases=tuple(args.basis or ("X", "Z")),
        compress=args.compress,
    )
    print(f"Wrote dual-rail dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
