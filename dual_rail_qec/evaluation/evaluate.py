"""Evaluate erasure-aware PyMatching regimes for dual-rail datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pymatching
import stim
import torch

from dual_rail_qec.data.datasets import DualRailShardDataset
from dual_rail_qec.data.simulator import (
    ErasureSidecarBatch,
    build_base_surface_code_circuit,
    extract_stim_qubit_layout,
    generate_stim_assisted_events,
    _detector_role_for_grid,
    _nearest_site,
    _round_to_tick,
    _tick_count,
)
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder
from dual_rail_qec.telemetry.geometry import SurfacePatchGeometry
from dual_rail_qec.telemetry.schema import QubitRole
from dual_rail_qec.telemetry.tensorize import tensorize_events


Signature = tuple[stim.DemTarget, ...]
OracleMap = dict[tuple[int, int], list[Signature]]


@dataclass(frozen=True)
class BasisSidecar:
    """Basis-specific erasure masks unpacked from sidecar shards."""

    basis: str
    data_erasures: np.ndarray
    measure_erasures: np.ndarray
    physical_data_erasures: np.ndarray
    physical_measure_erasures: np.ndarray
    readout_ambiguity: np.ndarray

    def observed_mask(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data_erasures, self.measure_erasures

    def physical_mask(self) -> tuple[np.ndarray, np.ndarray]:
        return self.physical_data_erasures, self.physical_measure_erasures


@dataclass
class InjectionDiagnostics:
    """Counters describing how erasure masks mapped into oracle signatures."""

    mask_locations: int = 0
    missing_erased_locations: int = 0
    oracle_lookup_hits: int = 0
    oracle_lookup_misses: int = 0
    injected_signatures: int = 0

    def add(self, other: "InjectionDiagnostics") -> None:
        self.mask_locations += other.mask_locations
        self.missing_erased_locations += other.missing_erased_locations
        self.oracle_lookup_hits += other.oracle_lookup_hits
        self.oracle_lookup_misses += other.oracle_lookup_misses
        self.injected_signatures += other.injected_signatures

    def to_metrics(self, shots: int) -> dict[str, float]:
        denom = max(int(shots), 1)
        return {
            "mask_locations": float(self.mask_locations),
            "missing_erased_locations": float(self.missing_erased_locations),
            "oracle_lookup_hits": float(self.oracle_lookup_hits),
            "oracle_lookup_misses": float(self.oracle_lookup_misses),
            "injected_signatures": float(self.injected_signatures),
            "mean_mask_locations_per_shot": float(self.mask_locations / denom),
            "mean_injected_signatures_per_shot": float(self.injected_signatures / denom),
        }


def _noise_value(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    noise = metadata.get("noise", {})
    if isinstance(noise, dict) and key in noise:
        return float(noise[key])
    return float(metadata.get(key, default))


def unpack_packed_mask(packed: np.ndarray, shape: Iterable[int]) -> np.ndarray:
    """Unpack a sidecar mask stored as per-shot ``np.packbits`` rows."""
    shape_tuple = tuple(int(v) for v in shape)
    flat_width = int(np.prod(shape_tuple[1:]))
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, count=flat_width)
    return unpacked.reshape(shape_tuple).astype(np.uint8)


def load_basis_sidecar(dataset_dir: Path, basis: str) -> BasisSidecar:
    """Load all per-shard sidecar masks for one basis."""
    basis_norm = str(basis).strip().upper()
    manifest_path = Path(dataset_dir) / f"erasures_{basis_norm}.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing basis sidecar manifest: {manifest_path}")
    with np.load(manifest_path, allow_pickle=True) as manifest:
        shard_files = [str(v) for v in manifest["shard_files"]]

    arrays: dict[str, list[np.ndarray]] = {
        "data_erasures": [],
        "measure_erasures": [],
        "physical_data_erasures": [],
        "physical_measure_erasures": [],
        "readout_ambiguity": [],
    }
    for rel_path in shard_files:
        shard_path = Path(dataset_dir) / rel_path
        with np.load(shard_path, allow_pickle=True) as shard:
            shape = tuple(int(v) for v in shard["shape"])
            arrays["data_erasures"].append(unpack_packed_mask(shard["data_erasures"], shape))
            arrays["measure_erasures"].append(unpack_packed_mask(shard["measure_erasures"], shape))
            arrays["physical_data_erasures"].append(unpack_packed_mask(shard["physical_data_erasures"], shape))
            arrays["physical_measure_erasures"].append(unpack_packed_mask(shard["physical_measure_erasures"], shape))
            arrays["readout_ambiguity"].append(np.asarray(shard["readout_ambiguity"], dtype=np.float32))

    return BasisSidecar(
        basis=basis_norm,
        data_erasures=np.concatenate(arrays["data_erasures"], axis=0),
        measure_erasures=np.concatenate(arrays["measure_erasures"], axis=0),
        physical_data_erasures=np.concatenate(arrays["physical_data_erasures"], axis=0),
        physical_measure_erasures=np.concatenate(arrays["physical_measure_erasures"], axis=0),
        readout_ambiguity=np.concatenate(arrays["readout_ambiguity"], axis=0),
    )


def _parse_sparse_dets(path: Path, *, num_detectors: int, num_observables: int) -> tuple[np.ndarray, np.ndarray]:
    det_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] != "shot":
                raise ValueError(f"{path}:{line_number} expected line to start with 'shot'")
            det = np.zeros(int(num_detectors), dtype=np.uint8)
            obs = np.zeros(int(num_observables), dtype=np.uint8)
            for token in parts[1:]:
                prefix = token[:1]
                try:
                    index = int(token[1:])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} invalid token {token!r}") from exc
                if prefix == "D":
                    if not 0 <= index < int(num_detectors):
                        raise ValueError(f"{path}:{line_number} detector index out of range: {index}")
                    det[index] = 1
                elif prefix == "L":
                    if not 0 <= index < int(num_observables):
                        raise ValueError(f"{path}:{line_number} observable index out of range: {index}")
                    obs[index] = 1
                else:
                    raise ValueError(f"{path}:{line_number} invalid token prefix: {token!r}")
            det_rows.append(det)
            obs_rows.append(obs)
    if not det_rows:
        return (
            np.zeros((0, int(num_detectors)), dtype=np.uint8),
            np.zeros((0, int(num_observables)), dtype=np.uint8),
        )
    return np.vstack(det_rows), np.vstack(obs_rows)


def _prediction_bit(prediction: np.ndarray) -> int:
    arr = np.asarray(prediction, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.sum(arr) % 2)


def _target_qubits_from_pauli_product(pauli_product) -> set[int]:
    qubits: set[int] = set()
    for term in pauli_product:
        gate_target = getattr(term, "gate_target", term)
        predicates = []
        for attr in ("is_qubit_target", "is_x_target", "is_y_target", "is_z_target"):
            value = getattr(gate_target, attr, False)
            predicates.append(value() if callable(value) else bool(value))
        if any(predicates):
            qubit_value = getattr(gate_target, "qubit_value")
            if callable(qubit_value):
                qubit_value = qubit_value()
            qubits.add(int(qubit_value))
    return qubits


def build_location_oracle(circuit, dem) -> OracleMap:
    """Map ``(tick, stim_qubit)`` locations to exact DEM target signatures."""
    oracle: OracleMap = {}
    explained_errors = circuit.explain_detector_error_model_errors(
        dem_filter=dem,
        reduce_to_one_representative_error=False,
    )
    for explained_error in explained_errors:
        signature = tuple(term.dem_target for term in explained_error.dem_error_terms)
        if not signature:
            continue
        for loc in explained_error.circuit_error_locations:
            tick = int(getattr(loc, "tick_offset", 0))
            qubits = _target_qubits_from_pauli_product(loc.flipped_pauli_product)
            for qubit in qubits:
                oracle.setdefault((tick, int(qubit)), []).append(signature)

    for key, signatures in list(oracle.items()):
        seen = set()
        unique: list[Signature] = []
        for signature in signatures:
            sig_key = tuple(str(target) for target in signature)
            if sig_key in seen:
                continue
            seen.add(sig_key)
            unique.append(signature)
        oracle[key] = unique
    return oracle


def _basis_sidecar_batch(
    sidecar: BasisSidecar,
    *,
    data_mask: np.ndarray,
    measure_mask: np.ndarray,
) -> ErasureSidecarBatch:
    return ErasureSidecarBatch(
        basis=sidecar.basis,
        data_erasures=np.asarray(data_mask, dtype=np.uint8),
        measure_erasures=np.asarray(measure_mask, dtype=np.uint8),
        readout_ambiguity=sidecar.readout_ambiguity.astype(np.float32, copy=False),
        site_records=[],
        physical_data_erasures=sidecar.physical_data_erasures.astype(np.uint8, copy=False),
        physical_measure_erasures=sidecar.physical_measure_erasures.astype(np.uint8, copy=False),
    )


def build_basis_tensors(
    *,
    distance: int,
    rounds: int,
    basis: str,
    detector_samples: np.ndarray,
    detector_coordinates: dict[int, tuple[float, ...]],
    sidecar: BasisSidecar,
    max_shots: int | None,
) -> np.ndarray:
    """Reconstruct basis-specific dense tensors for CNN inference."""
    num_shots = detector_samples.shape[0] if max_shots is None else min(detector_samples.shape[0], int(max_shots))
    observed_data, observed_measure = sidecar.observed_mask()
    batch_sidecar = _basis_sidecar_batch(
        sidecar,
        data_mask=observed_data[:num_shots],
        measure_mask=observed_measure[:num_shots],
    )
    tensors = []
    for shot_index in range(num_shots):
        events = generate_stim_assisted_events(
            distance=distance,
            rounds=rounds,
            shot_index=shot_index,
            detector_samples=detector_samples,
            basis=basis,
            detector_coordinates=detector_coordinates,
            erasure_sidecar=batch_sidecar,
        )
        tensors.append(tensorize_events(events, distance=distance, rounds=rounds))
    if not tensors:
        return np.zeros((0, 7, int(rounds), int(distance), int(distance)), dtype=np.float32)
    return np.stack(tensors).astype(np.float32)


def _load_model(checkpoint: Path, device: torch.device) -> DualRailCNN3DPreDecoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = payload.get("model", {})
    model = DualRailCNN3DPreDecoder(
        in_channels=int(model_cfg.get("in_channels", 7)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        depth=int(model_cfg.get("depth", 3)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def cnn_erasure_masks_for_basis(
    *,
    model: DualRailCNN3DPreDecoder,
    device: torch.device,
    basis_inputs: np.ndarray,
    batch_size: int,
    erasure_channel: int,
    threshold: float,
) -> tuple[np.ndarray, float]:
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, basis_inputs.shape[0], int(batch_size)):
            batch = torch.as_tensor(basis_inputs[start:start + int(batch_size)], dtype=torch.float32, device=device)
            logits = model(batch)
            if not 0 <= int(erasure_channel) < logits.shape[1]:
                raise ValueError(f"erasure_channel={erasure_channel} invalid for logits shape {tuple(logits.shape)}")
            mask = (torch.sigmoid(logits[:, int(erasure_channel)]) >= float(threshold)).to(torch.uint8)
            masks.append(mask.cpu().numpy())
    if not masks:
        shape = tuple(int(v) for v in basis_inputs.shape[2:])
        return np.zeros((0, *shape), dtype=np.uint8), 0.0
    out = np.concatenate(masks, axis=0)
    return out, float(np.mean(out)) if out.size else 0.0


def _mask_to_locations(
    *,
    circuit,
    basis: str,
    distance: int,
    rounds: int,
    data_mask: np.ndarray,
    measure_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], int]:
    geometry = SurfacePatchGeometry(distance=int(distance))
    sites = extract_stim_qubit_layout(circuit, distance=distance)
    total_ticks = _tick_count(circuit)
    locations: list[tuple[int, int]] = []
    missing = 0
    for t in range(int(rounds)):
        tick = _round_to_tick(t, rounds=rounds, total_ticks=total_ticks)
        for x in range(int(distance)):
            for y in range(int(distance)):
                if bool(data_mask[t, x, y]):
                    site = _nearest_site(sites, grid_x=x, grid_y=y, preferred_role=QubitRole.DATA)
                    if site is None:
                        missing += 1
                    else:
                        locations.append((tick, int(site.qubit)))
                if bool(measure_mask[t, x, y]):
                    role = geometry.role_at(x, y)
                    preferred = role if role != QubitRole.DATA else _detector_role_for_grid(geometry, x, y, basis)
                    site = _nearest_site(sites, grid_x=x, grid_y=y, preferred_role=preferred)
                    if site is None:
                        missing += 1
                    else:
                        locations.append((tick, int(site.qubit)))
    return locations, missing


def _signatures_for_mask(
    *,
    oracle: OracleMap,
    circuit,
    basis: str,
    distance: int,
    rounds: int,
    data_mask: np.ndarray,
    measure_mask: np.ndarray,
) -> tuple[list[Signature], InjectionDiagnostics]:
    locations, missing = _mask_to_locations(
        circuit=circuit,
        basis=basis,
        distance=distance,
        rounds=rounds,
        data_mask=data_mask,
        measure_mask=measure_mask,
    )
    diagnostics = InjectionDiagnostics(mask_locations=len(locations), missing_erased_locations=missing)
    signatures: list[Signature] = []
    seen_signatures = set()
    for key in locations:
        matches = oracle.get(key, [])
        if matches:
            diagnostics.oracle_lookup_hits += 1
        else:
            diagnostics.oracle_lookup_misses += 1
        for signature in matches:
            sig_key = tuple(str(target) for target in signature)
            if sig_key in seen_signatures:
                continue
            seen_signatures.add(sig_key)
            signatures.append(signature)
    diagnostics.injected_signatures = len(signatures)
    return signatures, diagnostics


def _dem_with_injections(base_dem, signatures: list[Signature], probability: float):
    dem = stim.DetectorErrorModel(str(base_dem))
    for signature in signatures:
        if signature:
            dem.append("error", float(probability), list(signature))
    return dem


def _decode_with_injected_dem(
    *,
    base_dem,
    detector_row: np.ndarray,
    signatures: list[Signature],
    oracle_probability: float,
) -> np.ndarray:
    injected_dem = _dem_with_injections(base_dem, signatures, oracle_probability)
    matching = pymatching.Matching.from_detector_error_model(injected_dem)
    return matching.decode(detector_row)


def _evaluate_injection_regime(
    *,
    name: str,
    base_dem,
    detector_samples: np.ndarray,
    truth_bits: np.ndarray,
    oracle: OracleMap,
    circuit,
    basis: str,
    distance: int,
    rounds: int,
    data_masks: np.ndarray,
    measure_masks: np.ndarray,
    oracle_probability: float,
) -> dict[str, float]:
    errors = 0
    diagnostics = InjectionDiagnostics()
    for shot_index in range(detector_samples.shape[0]):
        signatures, shot_diag = _signatures_for_mask(
            oracle=oracle,
            circuit=circuit,
            basis=basis,
            distance=distance,
            rounds=rounds,
            data_mask=data_masks[shot_index],
            measure_mask=measure_masks[shot_index],
        )
        diagnostics.add(shot_diag)
        prediction = _decode_with_injected_dem(
            base_dem=base_dem,
            detector_row=detector_samples[shot_index],
            signatures=signatures,
            oracle_probability=oracle_probability,
        )
        errors += int(_prediction_bit(prediction) != int(truth_bits[shot_index]))

    shots = max(int(detector_samples.shape[0]), 1)
    metrics = {
        "shots": float(detector_samples.shape[0]),
        "logical_errors": float(errors),
        "ler": float(errors / shots),
        "mask_density": float(np.mean(np.logical_or(data_masks, measure_masks))) if data_masks.size else 0.0,
        "oracle_probability": float(oracle_probability),
    }
    metrics.update(diagnostics.to_metrics(shots))
    metrics["regime"] = name
    return metrics


def _baseline_metrics(matching: pymatching.Matching, detector_samples: np.ndarray, truth_bits: np.ndarray) -> dict[str, float]:
    predictions = matching.decode_batch(detector_samples)
    pred_bits = np.asarray([_prediction_bit(row) for row in predictions], dtype=np.uint8)
    errors = int(np.sum(pred_bits != truth_bits))
    shots = max(int(detector_samples.shape[0]), 1)
    return {
        "shots": float(detector_samples.shape[0]),
        "logical_errors": float(errors),
        "ler": float(errors / shots),
        "regime": "vanilla_pymatching",
    }


def evaluate_basis(
    *,
    dataset_dir: Path,
    metadata: dict[str, Any],
    basis: str,
    sidecar: BasisSidecar,
    checkpoint: Path,
    model: DualRailCNN3DPreDecoder | None,
    device: torch.device,
    batch_size: int,
    erasure_channel: int,
    erasure_threshold: float,
    max_shots: int | None,
    oracle_probability: float,
) -> dict[str, Any]:
    basis_norm = str(basis).strip().upper()
    distance = int(metadata["distance"])
    rounds = int(metadata["rounds"])
    p_pauli = _noise_value(metadata, "p_pauli")
    p_measure = _noise_value(metadata, "p_measure")
    circuit = build_base_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        basis=basis_norm,
        p_pauli=p_pauli,
        p_measure=p_measure,
    )
    base_dem = circuit.detector_error_model(decompose_errors=True)
    vanilla_matching = pymatching.Matching.from_detector_error_model(base_dem)
    detector_samples, observables = _parse_sparse_dets(
        dataset_dir / f"samples_{basis_norm}.dets",
        num_detectors=int(circuit.num_detectors),
        num_observables=int(circuit.num_observables),
    )
    if max_shots is not None:
        detector_samples = detector_samples[:int(max_shots)]
        observables = observables[:int(max_shots)]
    truth_bits = np.asarray([_prediction_bit(row) for row in observables], dtype=np.uint8)
    num_shots = detector_samples.shape[0]

    oracle = build_location_oracle(circuit, base_dem)
    physical_data, physical_measure = sidecar.physical_mask()
    observed_data, observed_measure = sidecar.observed_mask()
    physical_data = physical_data[:num_shots]
    physical_measure = physical_measure[:num_shots]
    observed_data = observed_data[:num_shots]
    observed_measure = observed_measure[:num_shots]

    detector_coordinates = {
        int(k): tuple(float(v_i) for v_i in v)
        for k, v in circuit.get_detector_coordinates().items()
    }
    cnn_data = np.zeros_like(observed_data)
    cnn_measure = np.zeros_like(observed_measure)
    cnn_density = 0.0
    if model is not None:
        basis_inputs = build_basis_tensors(
            distance=distance,
            rounds=rounds,
            basis=basis_norm,
            detector_samples=detector_samples,
            detector_coordinates=detector_coordinates,
            sidecar=sidecar,
            max_shots=num_shots,
        )
        cnn_mask, cnn_density = cnn_erasure_masks_for_basis(
            model=model,
            device=device,
            basis_inputs=basis_inputs,
            batch_size=batch_size,
            erasure_channel=erasure_channel,
            threshold=erasure_threshold,
        )
        roles = np.zeros((distance, distance), dtype=np.uint8)
        geometry = SurfacePatchGeometry(distance=distance)
        for x in range(distance):
            for y in range(distance):
                roles[x, y] = 1 if geometry.role_at(x, y) == QubitRole.DATA else 2
        cnn_data = (cnn_mask & (roles[None, None, :, :] == 1)).astype(np.uint8)
        cnn_measure = (cnn_mask & (roles[None, None, :, :] != 1)).astype(np.uint8)

    regimes = {
        "vanilla_pymatching": _baseline_metrics(vanilla_matching, detector_samples, truth_bits),
        "physical_location_oracle": _evaluate_injection_regime(
            name="physical_location_oracle",
            base_dem=base_dem,
            detector_samples=detector_samples,
            truth_bits=truth_bits,
            oracle=oracle,
            circuit=circuit,
            basis=basis_norm,
            distance=distance,
            rounds=rounds,
            data_masks=physical_data,
            measure_masks=physical_measure,
            oracle_probability=oracle_probability,
        ),
        "observed_sidecar_dem_injection": _evaluate_injection_regime(
            name="observed_sidecar_dem_injection",
            base_dem=base_dem,
            detector_samples=detector_samples,
            truth_bits=truth_bits,
            oracle=oracle,
            circuit=circuit,
            basis=basis_norm,
            distance=distance,
            rounds=rounds,
            data_masks=observed_data,
            measure_masks=observed_measure,
            oracle_probability=oracle_probability,
        ),
        "cnn_dem_injection": _evaluate_injection_regime(
            name="cnn_dem_injection",
            base_dem=base_dem,
            detector_samples=detector_samples,
            truth_bits=truth_bits,
            oracle=oracle,
            circuit=circuit,
            basis=basis_norm,
            distance=distance,
            rounds=rounds,
            data_masks=cnn_data,
            measure_masks=cnn_measure,
            oracle_probability=oracle_probability,
        ),
    }
    vanilla_ler = regimes["vanilla_pymatching"]["ler"]
    for metrics in regimes.values():
        metrics["delta_vs_vanilla"] = float(metrics["ler"] - vanilla_ler)

    return {
        "basis": basis_norm,
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
        "oracle_locations": int(len(oracle)),
        "oracle_signatures": int(sum(len(v) for v in oracle.values())),
        "cnn_basis_mask_density": float(cnn_density),
        "regimes": regimes,
    }


def _aggregate_results(basis_results: dict[str, Any]) -> dict[str, Any]:
    regimes: dict[str, dict[str, float]] = {}
    for result in basis_results.values():
        for name, metrics in result["regimes"].items():
            agg = regimes.setdefault(
                name,
                {
                    "shots": 0.0,
                    "logical_errors": 0.0,
                    "mask_locations": 0.0,
                    "missing_erased_locations": 0.0,
                    "oracle_lookup_hits": 0.0,
                    "oracle_lookup_misses": 0.0,
                    "injected_signatures": 0.0,
                },
            )
            agg["shots"] += float(metrics.get("shots", 0.0))
            agg["logical_errors"] += float(metrics.get("logical_errors", 0.0))
            for key in (
                "mask_locations",
                "missing_erased_locations",
                "oracle_lookup_hits",
                "oracle_lookup_misses",
                "injected_signatures",
            ):
                agg[key] += float(metrics.get(key, 0.0))

    vanilla_ler = None
    for name, agg in regimes.items():
        shots = max(float(agg["shots"]), 1.0)
        agg["ler"] = float(agg["logical_errors"] / shots)
        agg["mean_mask_locations_per_shot"] = float(agg["mask_locations"] / shots)
        agg["mean_injected_signatures_per_shot"] = float(agg["injected_signatures"] / shots)
        if name == "vanilla_pymatching":
            vanilla_ler = agg["ler"]
    vanilla_ler = 0.0 if vanilla_ler is None else float(vanilla_ler)
    for agg in regimes.values():
        agg["delta_vs_vanilla"] = float(agg["ler"] - vanilla_ler)
    return regimes


def evaluate_pipeline(
    *,
    dataset_dir: Path,
    checkpoint: Path | None,
    batch_size: int = 64,
    erasure_channel: int = 3,
    erasure_threshold: float = 0.5,
    bases: tuple[str, ...] = ("X", "Z"),
    max_shots: int | None = None,
    oracle_probability: float = 0.499999,
    device: str | None = None,
) -> dict[str, Any]:
    dataset = DualRailShardDataset(dataset_dir)
    metadata = dataset.metadata
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(checkpoint, dev) if checkpoint is not None else None

    basis_results = {}
    for basis in bases:
        basis_norm = str(basis).strip().upper()
        sidecar = load_basis_sidecar(dataset_dir, basis_norm)
        basis_results[basis_norm] = evaluate_basis(
            dataset_dir=dataset_dir,
            metadata=metadata,
            basis=basis_norm,
            sidecar=sidecar,
            checkpoint=checkpoint or Path(""),
            model=model,
            device=dev,
            batch_size=batch_size,
            erasure_channel=erasure_channel,
            erasure_threshold=erasure_threshold,
            max_shots=max_shots,
            oracle_probability=oracle_probability,
        )

    return {
        "dataset_dir": str(dataset_dir),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "device": str(dev),
        "distance": int(metadata["distance"]),
        "rounds": int(metadata["rounds"]),
        "max_shots": None if max_shots is None else int(max_shots),
        "oracle_probability": float(oracle_probability),
        "erasure_channel": int(erasure_channel),
        "erasure_threshold": float(erasure_threshold),
        "basis": basis_results,
        "aggregate": _aggregate_results(basis_results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate erasure-aware dual-rail decoder regimes.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--erasure-channel", type=int, default=3)
    parser.add_argument("--erasure-threshold", type=float, default=0.5)
    parser.add_argument("--basis", action="append", choices=("X", "Z"), default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--oracle-probability", type=float, default=0.499999)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_pipeline(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        erasure_channel=args.erasure_channel,
        erasure_threshold=args.erasure_threshold,
        bases=tuple(args.basis or ("X", "Z")),
        max_shots=args.max_shots,
        oracle_probability=args.oracle_probability,
        device=args.device,
    )
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
