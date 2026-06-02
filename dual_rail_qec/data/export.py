"""Write dual-rail pre-generated dataset shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dual_rail_qec.data.simulator import (
    COUPLING_RULE,
    INJECTION_RULE,
    generate_erasure_sidecar,
    generate_synthetic_events,
    generate_stim_assisted_events,
    logical_erasure_parity,
    logical_label_from_targets,
    pack_erasure_sidecar,
    sample_stim_basis,
    stim,
)
from dual_rail_qec.telemetry.tensorize import NUM_INPUT_CHANNELS, make_local_targets, tensorize_events


DEFAULT_BASES = ("X", "Z")


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
    data_source: str,
    bases: tuple[str, ...],
    compress: bool,
) -> dict[str, Any]:
    source = str(data_source).strip().lower()
    stim_version = None if stim is None else getattr(stim, "__version__", "unknown")
    return {
        "schema_version": 1,
        "artifact": "dual_rail_qec_stim_coupled_dataset" if source == "stim" else "dual_rail_qec_dataset",
        "data_source": source,
        "stim_version": stim_version,
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
        "bases": list(bases),
        "injection_rule": INJECTION_RULE if source == "stim" else None,
        "coupling_rule": COUPLING_RULE if source == "stim" else None,
        "logical_label_rule": (
            "Stim observable parity XOR central-strip erasure parity proxy"
            if source == "stim"
            else "synthetic target parity proxy"
        ),
        "warning": (
            "first-pass Stim-assisted dual-rail approximation; erasures are explicit "
            "sidecar telemetry and are causally coupled at the phenomenological "
            "detector/target layer, not an exact cavity hardware model"
            if source == "stim"
            else "synthetic schema exerciser; not Stim surface-code dynamics"
        ),
        "sidecars": (
            {
                "detector_samples": [f"samples_{basis}.dets" for basis in bases],
                "erasures": [f"erasures_{basis}.npz" for basis in bases],
                "erasure_layout": (
                    "root NPZ manifests point to per-shard erasure sidecars; "
                    "each sidecar is aligned as shot, round, grid-x, grid-y"
                ),
            }
            if source == "stim"
            else {}
        ),
        "format": {
            "inputs": "float32[N,7,T,H,W]",
            "targets": "float32[N,4,T,H,W]",
            "logical_labels": "uint8[N,1]",
            "compressed_npz": bool(compress),
        },
    }


def resolve_data_source(data_source: str) -> str:
    value = str(data_source).strip().lower()
    if value == "auto":
        return "stim" if stim is not None else "synthetic"
    if value not in {"stim", "synthetic"}:
        raise ValueError(f"data_source must be 'auto', 'stim', or 'synthetic', got {data_source!r}")
    if value == "stim" and stim is None:
        raise RuntimeError("data_source='stim' requires the stim Python package.")
    return value


def _save_npz(path: Path, *, compress: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _append_dets(path: Path, detector_samples: np.ndarray, observables: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dets = np.asarray(detector_samples, dtype=np.uint8)
    obs = np.asarray(observables, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs.reshape(obs.shape[0], 1)
    with path.open("a", encoding="utf-8") as f:
        for shot_idx in range(dets.shape[0]):
            terms = [
                f"D{idx}"
                for idx, bit in enumerate(dets[shot_idx])
                if bool(bit)
            ]
            if obs.size:
                terms.extend(
                    f"L{idx}"
                    for idx, bit in enumerate(obs[shot_idx])
                    if bool(bit)
                )
            f.write("shot")
            if terms:
                f.write(" " + " ".join(terms))
            f.write("\n")


def _logical_label_from_observables(
    batches: dict[str, Any],
    sidecars: dict[str, Any],
    sample_idx: int,
) -> np.ndarray:
    label = 0
    used_observable = False
    for batch in batches.values():
        obs = np.asarray(batch.observables, dtype=np.uint8)
        if obs.size == 0:
            continue
        used_observable = True
        label ^= int(np.sum(obs[int(sample_idx)].reshape(-1)) % 2)
    for sidecar in sidecars.values():
        label ^= logical_erasure_parity(sidecar, sample_idx)
    if not used_observable:
        return np.asarray([label], dtype=np.uint8)
    return np.asarray([label], dtype=np.uint8)


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
    data_source: str = "auto",
    bases: tuple[str, ...] = DEFAULT_BASES,
    compress: bool = False,
) -> Path:
    """Write a dual-rail dataset and return its dataset directory."""
    source = resolve_data_source(data_source)
    basis_tuple = tuple(str(basis).strip().upper() for basis in bases)
    if source == "stim" and not basis_tuple:
        raise ValueError("Stim data generation requires at least one basis.")

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
        data_source=source,
        bases=basis_tuple,
        compress=compress,
    )
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    rng = np.random.default_rng(int(seed))
    erasure_manifest: dict[str, dict[str, Any]] = {}
    if source == "stim":
        for basis in basis_tuple:
            samples_path = dataset_dir / f"samples_{basis}.dets"
            samples_path.write_text("", encoding="utf-8")
            erasure_manifest[basis] = {"shard_files": [], "site_records": None}

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

        stim_batches = {}
        erasure_sidecars = {}
        if source == "stim":
            for basis in basis_tuple:
                batch = sample_stim_basis(
                    distance=distance,
                    rounds=rounds,
                    basis=basis,
                    num_shots=samples_per_shard,
                    p_pauli=p_pauli,
                    seed=int(seed) + shard_idx * 1009 + (17 if basis == "X" else 29),
                )
                stim_batches[basis] = batch
                erasure_sidecar = generate_erasure_sidecar(
                    distance=distance,
                    rounds=rounds,
                    num_shots=samples_per_shard,
                    p_erasure=p_erasure,
                    p_ambiguity=p_ambiguity,
                    basis=basis,
                    rng=np.random.default_rng(int(seed) + shard_idx * 1009 + (101 if basis == "X" else 211)),
                )
                erasure_sidecars[basis] = erasure_sidecar
                _append_dets(
                    dataset_dir / f"samples_{basis}.dets",
                    batch.detector_samples,
                    batch.observables,
                )
                erasure_shard_name = f"erasures_{basis}_shard_{shard_idx:05d}.npz"
                packed_sidecar = pack_erasure_sidecar(erasure_sidecar)
                _save_npz(
                    shard_dir / erasure_shard_name,
                    compress=compress,
                    **packed_sidecar,
                )
                erasure_manifest[basis]["shard_files"].append(f"shards/{erasure_shard_name}")
                if erasure_manifest[basis]["site_records"] is None:
                    erasure_manifest[basis]["site_records"] = [
                        record.to_dict() for record in erasure_sidecar.site_records
                    ]

        for sample_idx in range(int(samples_per_shard)):
            if source == "stim":
                events = []
                for basis, batch in stim_batches.items():
                    events.extend(
                        generate_stim_assisted_events(
                            distance=distance,
                            rounds=rounds,
                            shot_index=sample_idx,
                            detector_samples=batch.detector_samples,
                            basis=basis,
                            detector_coordinates=batch.detector_coordinates,
                            erasure_sidecar=erasure_sidecars[basis],
                        )
                    )
            else:
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
            logical_labels[sample_idx] = (
                _logical_label_from_observables(stim_batches, erasure_sidecars, sample_idx)
                if source == "stim"
                else logical_label_from_targets(sample_targets)
            )

        _save_npz(
            shard_dir / f"shard_{shard_idx:05d}.npz",
            compress=compress,
            inputs=inputs,
            targets=targets,
            logical_labels=logical_labels,
        )

    if source == "stim":
        for basis, manifest in erasure_manifest.items():
            _save_npz(
                dataset_dir / f"erasures_{basis}.npz",
                compress=compress,
                shard_files=np.asarray(manifest["shard_files"], dtype=object),
                site_records_json=np.asarray(
                    [json.dumps(manifest["site_records"] or [], sort_keys=True)],
                    dtype=object,
                ),
                layout=np.asarray(
                    ["per-shard arrays align as shot, round, grid-x, grid-y; binary arrays are packbits-flattened"],
                    dtype=object,
                ),
            )

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dual-rail QEC dataset shards.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--distance", "-d", type=int, required=True)
    parser.add_argument("--rounds", "--n-rounds", dest="rounds", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--p-erasure", type=float, default=0.01)
    parser.add_argument("--p-pauli", type=float, default=0.001)
    parser.add_argument("--p-ambiguity", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-source", choices=("auto", "stim", "synthetic"), default="auto")
    parser.add_argument("--basis", action="append", choices=("X", "Z"), default=None)
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ output.")
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
        data_source=args.data_source,
        bases=tuple(args.basis or DEFAULT_BASES),
        compress=args.compress,
    )
    print(f"Wrote dual-rail dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
