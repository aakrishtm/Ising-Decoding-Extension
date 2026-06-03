"""Evaluate PyMatching with optional CNN-guided erasure edge weighting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymatching
import torch

from dual_rail_qec.data.datasets import DualRailShardDataset
from dual_rail_qec.data.simulator import (
    build_base_surface_code_circuit,
    _coord_bounds_from_map,
    _grid_coord_from_stim,
)
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder


@dataclass(frozen=True)
class DetectorSite:
    """Dense-grid location for one Stim detector."""

    detector_id: int
    round_id: int
    x: int
    y: int


def _noise_value(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    noise = metadata.get("noise", {})
    if isinstance(noise, dict) and key in noise:
        return float(noise[key])
    return float(metadata.get(key, default))


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


def _detector_sites(circuit, *, distance: int, rounds: int) -> list[DetectorSite]:
    detector_coordinates = {
        int(k): tuple(float(v_i) for v_i in v)
        for k, v in circuit.get_detector_coordinates().items()
    }
    detector_xy = {
        idx: (coords[0], coords[1])
        for idx, coords in detector_coordinates.items()
        if len(coords) >= 2
    }
    bounds = _coord_bounds_from_map(detector_xy)
    sites: list[DetectorSite] = []
    for detector_id in range(int(circuit.num_detectors)):
        coords = detector_coordinates.get(detector_id, ())
        stim_x = coords[0] if len(coords) >= 1 else None
        stim_y = coords[1] if len(coords) >= 2 else None
        stim_t = coords[2] if len(coords) >= 3 else 0
        x, y = _grid_coord_from_stim(stim_x, stim_y, distance=distance, bounds=bounds)
        t = max(0, min(int(rounds) - 1, int(round(float(stim_t)))))
        sites.append(DetectorSite(detector_id=detector_id, round_id=t, x=x, y=y))
    return sites


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


def _cnn_erasure_masks(
    dataset: DualRailShardDataset,
    *,
    model: DualRailCNN3DPreDecoder,
    device: torch.device,
    batch_size: int,
    erasure_channel: int,
    threshold: float,
    max_shots: int | None,
) -> np.ndarray:
    masks: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for shard in dataset.iter_shards():
            inputs = shard["inputs"]
            if max_shots is not None:
                remaining = int(max_shots) - seen
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
            for start in range(0, inputs.shape[0], int(batch_size)):
                batch = torch.as_tensor(inputs[start:start + int(batch_size)], dtype=torch.float32, device=device)
                logits = model(batch)
                if not 0 <= int(erasure_channel) < logits.shape[1]:
                    raise ValueError(f"erasure_channel={erasure_channel} is invalid for logits shape {tuple(logits.shape)}")
                mask = (torch.sigmoid(logits[:, int(erasure_channel)]) >= float(threshold)).to(torch.uint8)
                masks.append(mask.cpu().numpy())
            seen += int(inputs.shape[0])
    if not masks:
        return np.zeros((0, *dataset.metadata["input_shape"][1:]), dtype=np.uint8)
    return np.concatenate(masks, axis=0)


def _input_erasure_masks(
    dataset: DualRailShardDataset,
    *,
    max_shots: int | None,
) -> np.ndarray:
    masks: list[np.ndarray] = []
    seen = 0
    for shard in dataset.iter_shards():
        inputs = shard["inputs"]
        if max_shots is not None:
            remaining = int(max_shots) - seen
            if remaining <= 0:
                break
            inputs = inputs[:remaining]
        masks.append(((inputs[:, 2] > 0.0) | (inputs[:, 3] > 0.0)).astype(np.uint8))
        seen += int(inputs.shape[0])
    if not masks:
        return np.zeros((0, *dataset.metadata["input_shape"][1:]), dtype=np.uint8)
    return np.concatenate(masks, axis=0)


def _edge_touches_erasure(
    node1: int,
    node2: int | None,
    *,
    detector_sites: list[DetectorSite],
    erasure_mask: np.ndarray,
    space_radius: int,
    time_radius: int,
) -> bool:
    for node in (node1, node2):
        if node is None:
            continue
        site = detector_sites[int(node)]
        t0 = max(0, site.round_id - int(time_radius))
        t1 = min(erasure_mask.shape[0], site.round_id + int(time_radius) + 1)
        x0 = max(0, site.x - int(space_radius))
        x1 = min(erasure_mask.shape[1], site.x + int(space_radius) + 1)
        y0 = max(0, site.y - int(space_radius))
        y1 = min(erasure_mask.shape[2], site.y + int(space_radius) + 1)
        if bool(np.any(erasure_mask[t0:t1, x0:x1, y0:y1])):
            return True
    return False


def _dynamic_matching_for_shot(
    baseline: pymatching.Matching,
    *,
    detector_sites: list[DetectorSite],
    erasure_mask: np.ndarray,
    space_radius: int,
    time_radius: int,
    erasure_weight_scale: float,
    min_erasure_weight: float,
) -> tuple[pymatching.Matching, int]:
    dynamic = pymatching.Matching()
    touched_edges = 0
    for node1, node2, data in baseline.edges():
        weight = float(data.get("weight", 1.0))
        if _edge_touches_erasure(
            int(node1),
            None if node2 is None else int(node2),
            detector_sites=detector_sites,
            erasure_mask=erasure_mask,
            space_radius=space_radius,
            time_radius=time_radius,
        ):
            weight = max(float(min_erasure_weight), weight * float(erasure_weight_scale))
            touched_edges += 1
        fault_ids = set(data.get("fault_ids", set()))
        error_probability = data.get("error_probability")
        if node2 is None:
            dynamic.add_boundary_edge(
                int(node1),
                fault_ids=fault_ids,
                weight=weight,
                error_probability=error_probability,
            )
        else:
            dynamic.add_edge(
                int(node1),
                int(node2),
                fault_ids=fault_ids,
                weight=weight,
                error_probability=error_probability,
            )
    return dynamic, touched_edges


def _prediction_bit(prediction: np.ndarray) -> int:
    arr = np.asarray(prediction, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.sum(arr) % 2)


def _evaluate_basis(
    *,
    basis: str,
    metadata: dict[str, Any],
    dataset_dir: Path,
    erasure_masks: np.ndarray,
    max_shots: int | None,
    space_radius: int,
    time_radius: int,
    erasure_weight_scale: float,
    min_erasure_weight: float,
) -> dict[str, float]:
    distance = int(metadata["distance"])
    rounds = int(metadata["rounds"])
    p_pauli = _noise_value(metadata, "p_pauli")
    p_measure = _noise_value(metadata, "p_measure")
    circuit = build_base_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        basis=basis,
        p_pauli=p_pauli,
        p_measure=p_measure,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    baseline = pymatching.Matching.from_detector_error_model(dem)
    dets, observables = _parse_sparse_dets(
        dataset_dir / f"samples_{basis}.dets",
        num_detectors=int(circuit.num_detectors),
        num_observables=int(circuit.num_observables),
    )
    if max_shots is not None:
        dets = dets[:int(max_shots)]
        observables = observables[:int(max_shots)]
    if erasure_masks.shape[0] != dets.shape[0]:
        raise ValueError(
            f"CNN mask count {erasure_masks.shape[0]} does not match {basis} detector shots {dets.shape[0]}"
        )

    baseline_predictions = baseline.decode_batch(dets)
    baseline_bits = np.asarray([_prediction_bit(row) for row in baseline_predictions], dtype=np.uint8)
    truth_bits = np.asarray([_prediction_bit(row) for row in observables], dtype=np.uint8)
    baseline_errors = int(np.sum(baseline_bits != truth_bits))

    detector_sites = _detector_sites(circuit, distance=distance, rounds=rounds)
    assisted_errors = 0
    touched_edge_total = 0
    for shot_index in range(dets.shape[0]):
        dynamic, touched_edges = _dynamic_matching_for_shot(
            baseline,
            detector_sites=detector_sites,
            erasure_mask=erasure_masks[shot_index],
            space_radius=space_radius,
            time_radius=time_radius,
            erasure_weight_scale=erasure_weight_scale,
            min_erasure_weight=min_erasure_weight,
        )
        prediction = dynamic.decode(dets[shot_index])
        assisted_errors += int(_prediction_bit(prediction) != int(truth_bits[shot_index]))
        touched_edge_total += int(touched_edges)

    shots = max(int(dets.shape[0]), 1)
    return {
        "shots": float(dets.shape[0]),
        "baseline_logical_errors": float(baseline_errors),
        "baseline_ler": float(baseline_errors / shots),
        "cnn_assisted_logical_errors": float(assisted_errors),
        "cnn_assisted_ler": float(assisted_errors / shots),
        "mean_dynamic_edges_per_shot": float(touched_edge_total / shots),
        "num_detectors": float(circuit.num_detectors),
        "num_observables": float(circuit.num_observables),
    }


def evaluate_pipeline(
    *,
    dataset_dir: Path,
    checkpoint: Path,
    batch_size: int = 64,
    mask_source: str = "cnn",
    erasure_channel: int = 3,
    erasure_threshold: float = 0.5,
    bases: tuple[str, ...] = ("X", "Z"),
    max_shots: int | None = None,
    space_radius: int = 1,
    time_radius: int = 1,
    erasure_weight_scale: float = 0.05,
    min_erasure_weight: float = 0.001,
    device: str | None = None,
) -> dict[str, Any]:
    dataset = DualRailShardDataset(dataset_dir)
    metadata = dataset.metadata
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mask_source_norm = str(mask_source).strip().lower()
    if mask_source_norm == "cnn":
        model = _load_model(checkpoint, dev)
        erasure_masks = _cnn_erasure_masks(
            dataset,
            model=model,
            device=dev,
            batch_size=batch_size,
            erasure_channel=erasure_channel,
            threshold=erasure_threshold,
            max_shots=max_shots,
        )
    elif mask_source_norm == "input":
        erasure_masks = _input_erasure_masks(dataset, max_shots=max_shots)
    else:
        raise ValueError(f"mask_source must be 'cnn' or 'input', got {mask_source!r}")
    results = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint),
        "device": str(dev),
        "shots": int(erasure_masks.shape[0]),
        "mask_source": mask_source_norm,
        "mask_erasure_density": float(np.mean(erasure_masks)) if erasure_masks.size else 0.0,
        "erasure_channel": int(erasure_channel),
        "erasure_threshold": float(erasure_threshold),
        "space_radius": int(space_radius),
        "time_radius": int(time_radius),
        "erasure_weight_scale": float(erasure_weight_scale),
        "min_erasure_weight": float(min_erasure_weight),
        "basis": {},
    }
    total_shots = 0
    baseline_errors = 0
    assisted_errors = 0
    for basis in bases:
        basis_norm = str(basis).strip().upper()
        basis_metrics = _evaluate_basis(
            basis=basis_norm,
            metadata=metadata,
            dataset_dir=dataset_dir,
            erasure_masks=erasure_masks,
            max_shots=max_shots,
            space_radius=space_radius,
            time_radius=time_radius,
            erasure_weight_scale=erasure_weight_scale,
            min_erasure_weight=min_erasure_weight,
        )
        results["basis"][basis_norm] = basis_metrics
        total_shots += int(basis_metrics["shots"])
        baseline_errors += int(basis_metrics["baseline_logical_errors"])
        assisted_errors += int(basis_metrics["cnn_assisted_logical_errors"])

    total_shots = max(total_shots, 1)
    results["aggregate"] = {
        "shots": float(total_shots),
        "baseline_logical_errors": float(baseline_errors),
        "baseline_ler": float(baseline_errors / total_shots),
        "cnn_assisted_logical_errors": float(assisted_errors),
        "cnn_assisted_ler": float(assisted_errors / total_shots),
        "ler_delta": float((assisted_errors - baseline_errors) / total_shots),
    }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN-guided dynamic erasure weighting for PyMatching.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mask-source", choices=("cnn", "input"), default="cnn")
    parser.add_argument("--erasure-channel", type=int, default=3)
    parser.add_argument("--erasure-threshold", type=float, default=0.5)
    parser.add_argument("--basis", action="append", choices=("X", "Z"), default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--space-radius", type=int, default=1)
    parser.add_argument("--time-radius", type=int, default=1)
    parser.add_argument("--erasure-weight-scale", type=float, default=0.05)
    parser.add_argument("--min-erasure-weight", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_pipeline(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        mask_source=args.mask_source,
        erasure_channel=args.erasure_channel,
        erasure_threshold=args.erasure_threshold,
        bases=tuple(args.basis or ("X", "Z")),
        max_shots=args.max_shots,
        space_radius=args.space_radius,
        time_radius=args.time_radius,
        erasure_weight_scale=args.erasure_weight_scale,
        min_erasure_weight=args.min_erasure_weight,
        device=args.device,
    )
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
