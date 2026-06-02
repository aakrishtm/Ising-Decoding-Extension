"""Simple grid geometry for the first dual-rail dataset contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dual_rail_qec.telemetry.schema import QubitRole


@dataclass(frozen=True)
class SurfacePatchGeometry:
    """Dense d-by-d embedding used by the initial 7-channel tensor contract."""

    distance: int

    def __post_init__(self) -> None:
        if int(self.distance) <= 0:
            raise ValueError(f"distance must be positive, got {self.distance!r}")
        if int(self.distance) % 2 == 0:
            raise ValueError(f"distance should be odd for this surface-code scaffold, got {self.distance!r}")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.distance), int(self.distance))

    def in_bounds(self, x: int, y: int) -> bool:
        h, w = self.shape
        return 0 <= int(x) < h and 0 <= int(y) < w

    def role_at(self, x: int, y: int) -> QubitRole:
        if (int(x) + int(y)) % 2 == 0:
            return QubitRole.DATA
        return QubitRole.X_MEASURE if int(x) % 2 == 0 else QubitRole.Z_MEASURE

    def qubit_id(self, x: int, y: int) -> str:
        return f"{self.role_at(x, y).value}:{int(x)}:{int(y)}"

    def valid_geometry(self) -> np.ndarray:
        return np.ones(self.shape, dtype=np.float32)

    def boundary_conditions(self) -> np.ndarray:
        boundary = np.zeros(self.shape, dtype=np.float32)
        boundary[0, :] = 1.0
        boundary[-1, :] = 1.0
        boundary[:, 0] = -1.0
        boundary[:, -1] = -1.0
        return boundary

