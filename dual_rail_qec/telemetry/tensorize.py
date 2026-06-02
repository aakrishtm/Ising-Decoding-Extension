"""Convert dual-rail hardware events into dense CNN tensors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from dual_rail_qec.telemetry.geometry import SurfacePatchGeometry
from dual_rail_qec.telemetry.schema import HardwareEvent, QubitRole

CHANNELS = {
    "syndrome_x": 0,
    "syndrome_z": 1,
    "data_erasure": 2,
    "measure_erasure": 3,
    "valid_geometry": 4,
    "boundary_conditions": 5,
    "readout_ambiguity": 6,
}

NUM_INPUT_CHANNELS = 7


def tensorize_events(
    events: Iterable[HardwareEvent],
    *,
    distance: int,
    rounds: int,
) -> np.ndarray:
    """Embed hardware events into one ``(7, T, H, W)`` float32 input tensor."""
    if int(rounds) <= 0:
        raise ValueError(f"rounds must be positive, got {rounds!r}")

    geometry = SurfacePatchGeometry(distance=int(distance))
    h, w = geometry.shape
    tensor = np.zeros((NUM_INPUT_CHANNELS, int(rounds), h, w), dtype=np.float32)
    tensor[CHANNELS["valid_geometry"], :, :, :] = geometry.valid_geometry()[None, :, :]
    tensor[CHANNELS["boundary_conditions"], :, :, :] = geometry.boundary_conditions()[None, :, :]

    for event in events:
        t = int(event.round_id)
        if not 0 <= t < int(rounds):
            raise ValueError(f"event round_id out of range: {event.round_id!r} for rounds={rounds}")
        if not geometry.in_bounds(event.x, event.y):
            raise ValueError(f"event coordinate out of bounds: {(event.x, event.y)!r}")

        if event.role == QubitRole.X_MEASURE and event.syndrome_parity:
            tensor[CHANNELS["syndrome_x"], t, event.x, event.y] = 1.0
        elif event.role == QubitRole.Z_MEASURE and event.syndrome_parity:
            tensor[CHANNELS["syndrome_z"], t, event.x, event.y] = 1.0

        if event.is_erasure:
            if event.role == QubitRole.DATA:
                tensor[CHANNELS["data_erasure"], t, event.x, event.y] = 1.0
            else:
                tensor[CHANNELS["measure_erasure"], t, event.x, event.y] = 1.0

        tensor[CHANNELS["readout_ambiguity"], t, event.x, event.y] = max(
            tensor[CHANNELS["readout_ambiguity"], t, event.x, event.y],
            float(event.ambiguity),
        )

    return tensor


def make_local_targets(input_tensor: np.ndarray) -> np.ndarray:
    """Create initial local fault targets from the 7-channel input tensor.

    These targets are intentionally simple and local. They are a supervised
    placeholder until the exact physical correction target is finalized.
    """
    if input_tensor.shape[0] != NUM_INPUT_CHANNELS:
        raise ValueError(f"expected {NUM_INPUT_CHANNELS} input channels, got {input_tensor.shape[0]}")

    targets = np.zeros((4, *input_tensor.shape[1:]), dtype=np.float32)
    targets[0] = np.maximum(
        input_tensor[CHANNELS["syndrome_z"]],
        input_tensor[CHANNELS["data_erasure"]],
    )
    targets[1] = np.maximum(
        input_tensor[CHANNELS["syndrome_x"]],
        input_tensor[CHANNELS["data_erasure"]],
    )
    targets[2] = input_tensor[CHANNELS["measure_erasure"]]
    targets[3] = np.maximum(
        input_tensor[CHANNELS["data_erasure"]],
        input_tensor[CHANNELS["measure_erasure"]],
    )
    return targets

