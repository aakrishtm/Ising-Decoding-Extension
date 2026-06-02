"""Dual-rail telemetry generators.

The synthetic path is a deterministic schema exerciser. The Stim-assisted path
uses unmodified Stim surface-code memory circuits for syndrome/logical dynamics
and generates explicit erasure telemetry as a shot/round/site-aligned sidecar.
This is still a first-pass dual-rail approximation, not an exact cavity-hardware
timing model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from dual_rail_qec.telemetry.geometry import SurfacePatchGeometry
from dual_rail_qec.telemetry.schema import DualRailState, HardwareEvent, QubitRole

try:
    import stim
except ImportError:
    stim = None


INJECTION_RULE = (
    "No erasure instruction is injected into the Stim circuit. Stim provides "
    "unmodified detector/logical samples; explicit dual-rail erasure telemetry "
    "is generated as sidecar arrays aligned by shot, round, and grid site."
)

COUPLING_RULE = (
    "Coupled first-pass dual-rail approximation: each explicit erasure sidecar "
    "event is also converted into deterministic local detector events before "
    "tensorization. Data-qubit erasures light up neighboring measure sites in "
    "the same and next round; measure-qubit erasures light up their own site in "
    "the same and next round. Logical labels are Stim observable parity XOR a "
    "central-strip erasure parity proxy."
)


@dataclass(frozen=True)
class ErasureSiteRecord:
    """Metadata for one sidecar erasure site."""

    site_index: int
    round_id: int
    x: int
    y: int
    role: QubitRole
    basis: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["role"] = self.role.value
        return data


@dataclass(frozen=True)
class StimSampleBatch:
    """Sampled Stim streams and metadata for one basis."""

    basis: str
    circuit: object
    detector_samples: np.ndarray
    observables: np.ndarray
    detector_coordinates: dict[int, tuple[float, ...]]


@dataclass(frozen=True)
class ErasureSidecarBatch:
    """Explicit erasure telemetry aligned by shot, round, and grid site."""

    basis: str
    data_erasures: np.ndarray
    measure_erasures: np.ndarray
    readout_ambiguity: np.ndarray
    site_records: list[ErasureSiteRecord]


def _require_stim():
    if stim is None:
        raise RuntimeError(
            "Stim is required for data_source='stim'. Install stim or use data_source='synthetic'."
        )
    return stim


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


def build_base_surface_code_circuit(
    distance: int,
    rounds: int,
    basis: str,
    p_pauli: float,
):
    """Build Stim's generated rotated surface-code memory circuit."""
    stim_module = _require_stim()
    basis_norm = str(basis).strip().upper()
    if basis_norm == "X":
        task = "surface_code:rotated_memory_x"
    elif basis_norm == "Z":
        task = "surface_code:rotated_memory_z"
    else:
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

    return stim_module.Circuit.generated(
        task,
        distance=int(distance),
        rounds=int(rounds),
        after_clifford_depolarization=float(p_pauli),
    )


def _operation_name(line: str) -> str:
    first = line.strip().split(maxsplit=1)[0]
    return first.split("(", 1)[0].split("[", 1)[0].upper()


def _parse_gate_args(line: str) -> tuple[float, ...]:
    line = line.strip()
    if "(" not in line or ")" not in line:
        return ()
    inside = line.split("(", 1)[1].split(")", 1)[0].strip()
    if not inside:
        return ()
    out = []
    for part in inside.split(","):
        try:
            out.append(float(part.strip()))
        except ValueError:
            continue
    return tuple(out)


def _parse_qubit_targets(line: str) -> list[int]:
    text = line.strip()
    if not text or text.startswith("#"):
        return []
    if " " not in text:
        return []
    target_text = text.split(maxsplit=1)[1]
    qubits = []
    for raw in target_text.replace("*", " ").split():
        token = raw.strip().lstrip("!").upper()
        if not token:
            continue
        if token[0] in "XYZ":
            token = token[1:]
        if token.isdigit():
            qubits.append(int(token))
    return qubits


def _parse_qubit_coords(line: str) -> tuple[int, float, float] | None:
    if _operation_name(line) != "QUBIT_COORDS":
        return None
    args = _parse_gate_args(line)
    targets = _parse_qubit_targets(line)
    if len(args) < 2 or not targets:
        return None
    return targets[0], float(args[0]), float(args[1])


def _role_from_stim_coords(stim_x: float | None, stim_y: float | None) -> QubitRole:
    if stim_x is None or stim_y is None:
        return QubitRole.DATA
    x_i = int(round(stim_x))
    y_i = int(round(stim_y))
    if x_i % 2 == 1 and y_i % 2 == 1:
        return QubitRole.DATA
    return QubitRole.X_MEASURE if y_i % 4 == 0 else QubitRole.Z_MEASURE


def _grid_coord_from_stim(
    stim_x: float | None,
    stim_y: float | None,
    *,
    distance: int,
    bounds: tuple[float, float, float, float] | None,
) -> tuple[int, int]:
    if stim_x is None or stim_y is None or bounds is None:
        return 0, 0
    min_x, max_x, min_y, max_y = bounds
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    gx = int(round((float(stim_x) - min_x) * (int(distance) - 1) / x_span))
    gy = int(round((float(stim_y) - min_y) * (int(distance) - 1) / y_span))
    return max(0, min(int(distance) - 1, gx)), max(0, min(int(distance) - 1, gy))


def _coord_bounds_from_map(coord_map: dict[int, tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not coord_map:
        return None
    xs = [xy[0] for xy in coord_map.values()]
    ys = [xy[1] for xy in coord_map.values()]
    return min(xs), max(xs), min(ys), max(ys)


def _compile_sampler(circuit, *, seed: int | None, detector: bool):
    if detector:
        try:
            return circuit.compile_detector_sampler(seed=seed)
        except TypeError:
            return circuit.compile_detector_sampler()
    try:
        return circuit.compile_sampler(seed=seed)
    except TypeError:
        return circuit.compile_sampler()


def sample_stim_basis(
    *,
    distance: int,
    rounds: int,
    basis: str,
    num_shots: int,
    p_pauli: float,
    seed: int | None = None,
) -> StimSampleBatch:
    """Sample detector and observable data from an unmodified Stim circuit."""
    circuit = build_base_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        basis=basis,
        p_pauli=p_pauli,
    )

    detector_sampler = _compile_sampler(circuit, seed=seed, detector=True)
    try:
        detector_samples, observables = detector_sampler.sample(
            shots=int(num_shots),
            separate_observables=True,
        )
    except TypeError:
        detector_samples = detector_sampler.sample(shots=int(num_shots))
        observables = np.zeros((int(num_shots), 1), dtype=np.uint8)

    detector_coordinates = {
        int(k): tuple(float(v_i) for v_i in v)
        for k, v in circuit.get_detector_coordinates().items()
    }
    return StimSampleBatch(
        basis=str(basis).strip().upper(),
        circuit=circuit,
        detector_samples=np.asarray(detector_samples, dtype=np.uint8),
        observables=np.asarray(observables, dtype=np.uint8),
        detector_coordinates=detector_coordinates,
    )


def generate_erasure_sidecar(
    *,
    distance: int,
    rounds: int,
    num_shots: int,
    p_erasure: float,
    p_ambiguity: float,
    basis: str,
    rng: np.random.Generator,
) -> ErasureSidecarBatch:
    """Generate explicit sidecar erasure telemetry aligned as ``(N, T, H, W)``."""
    geometry = SurfacePatchGeometry(distance=int(distance))
    h, w = geometry.shape
    site_records: list[ErasureSiteRecord] = []
    data_mask = np.zeros((h, w), dtype=bool)
    measure_mask = np.zeros((h, w), dtype=bool)
    for x in range(h):
        for y in range(w):
            role = geometry.role_at(x, y)
            if role == QubitRole.DATA:
                data_mask[x, y] = True
            else:
                measure_mask[x, y] = True
            for t in range(int(rounds)):
                site_records.append(
                    ErasureSiteRecord(
                        site_index=len(site_records),
                        round_id=t,
                        x=x,
                        y=y,
                        role=role,
                        basis=str(basis).strip().upper(),
                    )
                )

    erasure_draws = rng.random((int(num_shots), int(rounds), h, w)) < float(p_erasure)
    ambiguity_draws = rng.random((int(num_shots), int(rounds), h, w)) < float(p_ambiguity)
    ambiguity_strength = rng.uniform(0.0, 1.0, size=(int(num_shots), int(rounds), h, w)).astype(np.float32)
    ambiguity = np.where(ambiguity_draws, ambiguity_strength, 0.0).astype(np.float32)

    data_sites = data_mask[None, None, :, :]
    measure_sites = measure_mask[None, None, :, :]
    data_erasures = np.logical_or(erasure_draws, ambiguity_draws) & data_sites
    measure_erasures = np.logical_or(erasure_draws, ambiguity_draws) & measure_sites

    return ErasureSidecarBatch(
        basis=str(basis).strip().upper(),
        data_erasures=data_erasures.astype(np.uint8),
        measure_erasures=measure_erasures.astype(np.uint8),
        readout_ambiguity=ambiguity,
        site_records=site_records,
    )


def _detector_role_for_grid(geometry: SurfacePatchGeometry, x: int, y: int, basis: str) -> QubitRole:
    role = geometry.role_at(x, y)
    if role != QubitRole.DATA:
        return role
    return QubitRole.X_MEASURE if str(basis).strip().upper() == "X" else QubitRole.Z_MEASURE


def _neighbor_measure_sites(
    geometry: SurfacePatchGeometry,
    x: int,
    y: int,
    basis: str,
) -> list[tuple[int, int, QubitRole]]:
    sites = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = int(x) + dx, int(y) + dy
        if not geometry.in_bounds(nx, ny):
            continue
        role = geometry.role_at(nx, ny)
        if role != QubitRole.DATA:
            sites.append((nx, ny, role))
    if sites:
        return sites

    # Boundary/corner fallback for the compressed d-by-d scaffold.
    role = QubitRole.X_MEASURE if str(basis).strip().upper() == "X" else QubitRole.Z_MEASURE
    return [(int(x), int(y), role)]


def _add_syndrome_event(
    events: list[HardwareEvent],
    *,
    basis: str,
    round_id: int,
    x: int,
    y: int,
    role: QubitRole,
    reason: str,
) -> None:
    events.append(
        HardwareEvent(
            round_id=int(round_id),
            qubit_id=f"COUPLED:{basis}:{reason}:{round_id}:{x}:{y}",
            x=int(x),
            y=int(y),
            role=role,
            dual_rail_state=DualRailState.LOGICAL_01,
            readout_confidence=1.0,
            syndrome_parity=True,
        )
    )


def generate_stim_assisted_events(
    distance: int,
    rounds: int,
    shot_index: int,
    detector_samples: np.ndarray,
    basis: str,
    detector_coordinates: dict[int, tuple[float, ...]] | None = None,
    erasure_sidecar: ErasureSidecarBatch | None = None,
) -> list[HardwareEvent]:
    """Convert one sampled Stim shot into hardware-style telemetry events."""
    geometry = SurfacePatchGeometry(distance=int(distance))
    events: list[HardwareEvent] = []
    detector_coordinates = detector_coordinates or {}
    detector_xy = {
        idx: (coords[0], coords[1])
        for idx, coords in detector_coordinates.items()
        if len(coords) >= 2
    }
    det_bounds = _coord_bounds_from_map(detector_xy)

    det_row = np.asarray(detector_samples[int(shot_index)], dtype=np.uint8)
    for detector_idx, bit in enumerate(det_row):
        if not bool(bit):
            continue
        coords = detector_coordinates.get(detector_idx, ())
        stim_x = coords[0] if len(coords) >= 1 else None
        stim_y = coords[1] if len(coords) >= 2 else None
        stim_t = coords[2] if len(coords) >= 3 else 0
        x, y = _grid_coord_from_stim(stim_x, stim_y, distance=distance, bounds=det_bounds)
        role = _detector_role_for_grid(geometry, x, y, basis)
        events.append(
            HardwareEvent(
                round_id=max(0, min(int(rounds) - 1, int(round(float(stim_t))))),
                qubit_id=f"DETECTOR:{basis}:{detector_idx}",
                x=x,
                y=y,
                role=role,
                dual_rail_state=DualRailState.LOGICAL_01,
                readout_confidence=1.0,
                syndrome_parity=True,
            )
        )

    if erasure_sidecar is not None:
        data = np.asarray(erasure_sidecar.data_erasures[int(shot_index)], dtype=np.uint8)
        meas = np.asarray(erasure_sidecar.measure_erasures[int(shot_index)], dtype=np.uint8)
        ambiguity = np.asarray(erasure_sidecar.readout_ambiguity[int(shot_index)], dtype=np.float32)
        for t in range(int(rounds)):
            for x in range(int(distance)):
                for y in range(int(distance)):
                    if not bool(data[t, x, y] or meas[t, x, y]):
                        continue
                    role = geometry.role_at(x, y)
                    ambiguous = float(ambiguity[t, x, y]) > 0.0
                    state = (
                        DualRailState.AMBIGUOUS
                        if ambiguous
                        else (DualRailState.LEAKAGE_00 if (x + y + t) % 2 == 0 else DualRailState.LEAKAGE_11)
                    )
                    events.append(
                        HardwareEvent(
                            round_id=t,
                            qubit_id=f"ERASURE:{basis}:{t}:{x}:{y}",
                            x=x,
                            y=y,
                            role=role,
                            dual_rail_state=state,
                            readout_confidence=1.0 - float(ambiguity[t, x, y]),
                            syndrome_parity=None,
                        )
                    )
                    coupled_rounds = [t]
                    if t + 1 < int(rounds):
                        coupled_rounds.append(t + 1)

                    if bool(data[t, x, y]):
                        for cx, cy, c_role in _neighbor_measure_sites(geometry, x, y, basis):
                            for c_round in coupled_rounds:
                                _add_syndrome_event(
                                    events,
                                    basis=basis,
                                    round_id=c_round,
                                    x=cx,
                                    y=cy,
                                    role=c_role,
                                    reason="data_erasure",
                                )
                    if bool(meas[t, x, y]):
                        c_role = role if role != QubitRole.DATA else _detector_role_for_grid(geometry, x, y, basis)
                        for c_round in coupled_rounds:
                            _add_syndrome_event(
                                events,
                                basis=basis,
                                round_id=c_round,
                                x=x,
                                y=y,
                                role=c_role,
                                reason="measure_erasure",
                            )

    return events


def logical_erasure_parity(
    sidecar: ErasureSidecarBatch,
    shot_index: int,
) -> int:
    """Central-strip parity proxy for erasure-induced logical changes."""
    data = np.asarray(sidecar.data_erasures[int(shot_index)], dtype=np.uint8)
    meas = np.asarray(sidecar.measure_erasures[int(shot_index)], dtype=np.uint8)
    _, h, w = data.shape
    mid_x = h // 2
    mid_y = w // 2
    if str(sidecar.basis).strip().upper() == "X":
        parity_source = data[:, :, mid_y] + meas[:, :, mid_y]
    else:
        parity_source = data[:, mid_x, :] + meas[:, mid_x, :]
    return int(np.sum(parity_source) % 2)


def pack_erasure_sidecar(sidecar: ErasureSidecarBatch) -> dict[str, np.ndarray]:
    """Pack sidecar arrays for scalable shard storage."""
    return {
        "data_erasures": np.packbits(sidecar.data_erasures.reshape(sidecar.data_erasures.shape[0], -1), axis=1),
        "measure_erasures": np.packbits(sidecar.measure_erasures.reshape(sidecar.measure_erasures.shape[0], -1), axis=1),
        "readout_ambiguity": sidecar.readout_ambiguity.astype(np.float32, copy=False),
        "shape": np.asarray(sidecar.data_erasures.shape, dtype=np.int64),
    }


def logical_label_from_targets(targets: np.ndarray) -> np.ndarray:
    """Return a simple one-observable logical label for scaffold datasets."""
    return np.asarray([int(np.sum(targets[0] + targets[1]) % 2)], dtype=np.uint8)


def pack_events(events: Sequence[HardwareEvent]) -> str:
    """Compact JSONL-ish representation for optional shard provenance."""
    return "\n".join(str(event.to_dict()) for event in events)
