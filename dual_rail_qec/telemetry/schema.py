"""Typed hardware-event schema for dual-rail decoder datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class QubitRole(str, Enum):
    """Physical role of a hardware site in the surface-code patch."""

    DATA = "DATA"
    X_MEASURE = "X_MEASURE"
    Z_MEASURE = "Z_MEASURE"


class DualRailState(str, Enum):
    """Dual-rail readout state emitted by the hardware layer."""

    LOGICAL_01 = "LOGICAL_01"
    LOGICAL_10 = "LOGICAL_10"
    LEAKAGE_00 = "LEAKAGE_00"
    LEAKAGE_11 = "LEAKAGE_11"
    AMBIGUOUS = "AMBIGUOUS"

    @property
    def is_erasure(self) -> bool:
        return self in {
            DualRailState.LEAKAGE_00,
            DualRailState.LEAKAGE_11,
            DualRailState.AMBIGUOUS,
        }


@dataclass(frozen=True)
class HardwareEvent:
    """One dual-rail hardware telemetry event at a grid site and round."""

    round_id: int
    qubit_id: str
    x: int
    y: int
    role: QubitRole
    dual_rail_state: DualRailState
    readout_confidence: float
    syndrome_parity: bool | None = None

    @property
    def is_erasure(self) -> bool:
        return self.dual_rail_state.is_erasure

    @property
    def ambiguity(self) -> float:
        if self.dual_rail_state != DualRailState.AMBIGUOUS:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(self.readout_confidence)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        data["dual_rail_state"] = self.dual_rail_state.value
        data["is_erasure"] = self.is_erasure
        data["ambiguity"] = self.ambiguity
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareEvent":
        return cls(
            round_id=int(data["round_id"]),
            qubit_id=str(data["qubit_id"]),
            x=int(data["x"]),
            y=int(data["y"]),
            role=QubitRole(str(data["role"])),
            dual_rail_state=DualRailState(str(data["dual_rail_state"])),
            readout_confidence=float(data["readout_confidence"]),
            syndrome_parity=(
                None if data.get("syndrome_parity") is None else bool(data["syndrome_parity"])
            ),
        )

