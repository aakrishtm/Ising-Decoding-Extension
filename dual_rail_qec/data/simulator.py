"""Synthetic dual-rail telemetry generator.

This is not a Stim circuit model. It creates deterministic hardware-style event
records that exercise the dataset schema and tensor contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dual_rail_qec.telemetry.geometry import SurfacePatchGeometry
from dual_rail_qec.telemetry.schema import DualRailState, HardwareEvent, QubitRole


def _sample_state(rng: np.random.Generator, p_erasure: float, p_ambiguity: float) -> DualRailState:
    u = float(rng.random())
    if u < p_ambiguity:
        return DualRailState.AMBIGUOUS
    if u < p_ambiguity + p_erasure / 2.0:
        return DualRailState.LEAKAGE_00
    if u < p_ambiguity + p_erasure:
        return DualRailState.LEAKAGE_11
    return DualRailState.LOGICAL_01 if bool(rng.integers(0, 2)) else DualRailState.LOGICAL_10


def generate_synthetic_events(
    *,
    distance: int,
    rounds: int,
    rng: np.random.Generator,
    p_erasure: float,
    p_pauli: float,
    p_ambiguity: float = 0.0,
) -> list[HardwareEvent]:
    """Generate one sample of synthetic dual-rail hardware telemetry."""
    geometry = SurfacePatchGeometry(distance=int(distance))
    events: list[HardwareEvent] = []

    for t in range(int(rounds)):
        for x in range(geometry.shape[0]):
            for y in range(geometry.shape[1]):
                role = geometry.role_at(x, y)
                state = _sample_state(rng, p_erasure=float(p_erasure), p_ambiguity=float(p_ambiguity))
                confidence = 1.0
                if state == DualRailState.AMBIGUOUS:
                    confidence = float(rng.uniform(0.0, 0.75))
                elif state.is_erasure:
                    confidence = float(rng.uniform(0.75, 1.0))

                syndrome_parity = None
                if role in (QubitRole.X_MEASURE, QubitRole.Z_MEASURE) and not state.is_erasure:
                    syndrome_parity = bool(rng.random() < float(p_pauli))

                events.append(
                    HardwareEvent(
                        round_id=t,
                        qubit_id=geometry.qubit_id(x, y),
                        x=x,
                        y=y,
                        role=role,
                        dual_rail_state=state,
                        readout_confidence=confidence,
                        syndrome_parity=syndrome_parity,
                    )
                )

    return events


def logical_label_from_targets(targets: np.ndarray) -> np.ndarray:
    """Return a simple one-observable logical label for scaffold datasets."""
    return np.asarray([int(np.sum(targets[0] + targets[1]) % 2)], dtype=np.uint8)


def pack_events(events: Sequence[HardwareEvent]) -> str:
    """Compact JSONL-ish representation for optional shard provenance."""
    return "\n".join(str(event.to_dict()) for event in events)

