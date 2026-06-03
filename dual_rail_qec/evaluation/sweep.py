"""Sweep entry point for large erasure-aware decoder evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dual_rail_qec.data.export import write_dataset
from dual_rail_qec.evaluation.evaluate import evaluate_pipeline


DEFAULT_P_PAULI_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def _run_id(distance: int, p_pauli: float, p_erasure: float) -> str:
    return f"d{int(distance)}_pP{float(p_pauli):.1e}_pE{float(p_erasure):.1e}".replace("+", "")


def _resolve_checkpoint(args: argparse.Namespace, distance: int) -> Path | None:
    if args.checkpoint_template:
        return Path(str(args.checkpoint_template).format(distance=int(distance)))
    if args.checkpoint:
        return Path(args.checkpoint)
    return None


def _flatten_results(
    *,
    result: dict[str, Any],
    distance: int,
    p_pauli: float,
    p_erasure: float,
    p_measure: float,
    p_false_positive: float,
    p_false_negative: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = {
        "distance": int(distance),
        "rounds": int(result["rounds"]),
        "p_pauli": float(p_pauli),
        "p_erasure": float(p_erasure),
        "p_measure": float(p_measure),
        "p_false_positive": float(p_false_positive),
        "p_false_negative": float(p_false_negative),
        "dataset_dir": result["dataset_dir"],
        "checkpoint": result["checkpoint"],
        "oracle_probability": float(result["oracle_probability"]),
    }
    for regime, metrics in result["aggregate"].items():
        row = dict(common)
        row.update({"basis": "aggregate", "regime": regime})
        row.update(metrics)
        rows.append(row)
    for basis, basis_result in result["basis"].items():
        for regime, metrics in basis_result["regimes"].items():
            row = dict(common)
            row.update(
                {
                    "basis": basis,
                    "regime": regime,
                    "oracle_locations": int(basis_result["oracle_locations"]),
                    "oracle_signatures": int(basis_result["oracle_signatures"]),
                    "cnn_basis_mask_density": float(basis_result["cnn_basis_mask_density"]),
                }
            )
            row.update(metrics)
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return
    if output_format == "csv":
        file_exists = path.exists() and path.stat().st_size > 0
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError(f"Unsupported output format: {output_format!r}")


def run_sweep(args: argparse.Namespace) -> None:
    all_rows: list[dict[str, Any]] = []
    for distance in args.distances:
        checkpoint = _resolve_checkpoint(args, distance)
        for p_pauli in args.p_pauli_values:
            run_id = _run_id(distance, p_pauli, args.p_erasure)
            run_root = Path(args.output_root) / run_id
            dataset_dir = run_root / f"dual_rail_d{int(distance)}"
            if not args.skip_generate:
                dataset_dir = write_dataset(
                    output_root=run_root,
                    distance=int(distance),
                    rounds=int(args.rounds or distance),
                    num_shards=int(args.num_shards),
                    samples_per_shard=int(args.samples_per_shard),
                    p_erasure=float(args.p_erasure),
                    p_pauli=float(p_pauli),
                    p_measure=float(args.p_measure),
                    p_false_positive=float(args.p_false_positive),
                    p_false_negative=float(args.p_false_negative),
                    p_ambiguity=float(args.p_ambiguity),
                    seed=int(args.seed),
                    data_source="stim",
                    compress=bool(args.compress),
                )
            result = evaluate_pipeline(
                dataset_dir=dataset_dir,
                checkpoint=checkpoint,
                batch_size=int(args.batch_size),
                max_shots=args.max_shots,
                oracle_probability=float(args.oracle_probability),
                erasure_threshold=float(args.erasure_threshold),
                erasure_channel=int(args.erasure_channel),
                device=args.device,
            )
            rows = _flatten_results(
                result=result,
                distance=int(distance),
                p_pauli=float(p_pauli),
                p_erasure=float(args.p_erasure),
                p_measure=float(args.p_measure),
                p_false_positive=float(args.p_false_positive),
                p_false_negative=float(args.p_false_negative),
            )
            _write_rows(Path(args.results_path), rows, args.output_format)
            all_rows.extend(rows)
            print(json.dumps({"run_id": run_id, "rows": len(rows), "dataset_dir": str(dataset_dir)}, sort_keys=True))
    print(json.dumps({"status": "complete", "total_rows": len(all_rows), "results_path": str(args.results_path)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H100-ready dual-rail decoder evaluation sweeps.")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--rounds", type=int, default=None, help="Defaults to distance for each run.")
    parser.add_argument("--p-pauli-values", type=float, nargs="+", default=list(DEFAULT_P_PAULI_VALUES))
    parser.add_argument("--p-erasure", type=float, default=0.01)
    parser.add_argument("--p-measure", type=float, default=0.001)
    parser.add_argument("--p-false-positive", type=float, default=0.0001)
    parser.add_argument("--p-false-negative", type=float, default=0.001)
    parser.add_argument("--p-ambiguity", type=float, default=0.0)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--samples-per-shard", type=int, default=10000)
    parser.add_argument("--output-root", type=Path, default=Path("sweeps/data"))
    parser.add_argument("--results-path", type=Path, default=Path("sweeps/results/decoder_sweep.jsonl"))
    parser.add_argument("--output-format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-template", type=str, default=None, help="Example: outputs/dual_rail_d{distance}/latest.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--oracle-probability", type=float, default=0.499999)
    parser.add_argument("--erasure-channel", type=int, default=3)
    parser.add_argument("--erasure-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-generate", action="store_true", help="Reuse already-generated sweep datasets.")
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
