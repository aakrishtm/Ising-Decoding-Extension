"""Dual-rail telemetry generators.

The synthetic path is a deterministic schema exerciser. The Stim-assisted path
samples explicit erasure patterns, builds per-shot Stim circuits with
deterministic Pauli error instructions at the corresponding spacetime sites, and
stores the same erasure patterns as telemetry sidecars.
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
    "No measurement-producing erasure instruction is injected into the Stim circuit. "
    "For each sampled erasure pattern, deterministic Pauli error instructions "
    "X_ERROR(1), Y_ERROR(1), or Z_ERROR(1) are inserted at non-measurement-producing "
    "spacetime locations. This preserves DETECTOR/OBSERVABLE rec offsets."
)

COUPLING_RULE = (
    "Per-erasure Stim circuit conversion: true erasure events are sampled first, "
    "then converted into deterministic Pauli error instructions in a per-shot "
    "Stim circuit. Detector and logical samples are taken from that modified "
    "circuit, while the observed erasure detections are preserved as telemetry. "
    "False positives appear only in telemetry; false negatives affect the circuit "
    "without appearing in telemetry."
)


@dataclass(frozen=True)
class ErasureNoiseModel:
    """Noise parameters for coupled dual-rail erasure simulations."""

    p_erasure: float
    p_pauli: float
    p_measure: float = 0.0
    p_false_positive: float = 0.0
    p_false_negative: float = 0.0
    p_ambiguity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "p_erasure": float(self.p_erasure),
            "p_pauli": float(self.p_pauli),
            "p_measure": float(self.p_measure),
            "p_false_positive": float(self.p_false_positive),
            "p_false_negative": float(self.p_false_negative),
            "p_ambiguity": float(self.p_ambiguity),
        }


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
    physical_data_erasures: np.ndarray | None = None
    physical_measure_erasures: np.ndarray | None = None
    false_positive_data: np.ndarray | None = None
    false_positive_measure: np.ndarray | None = None
    false_negative_data: np.ndarray | None = None
    false_negative_measure: np.ndarray | None = None


@dataclass(frozen=True)
class StimQubitSite:
    """Mapping from one Stim qubit to the dense decoder grid."""

    qubit: int
    stim_x: float
    stim_y: float
    grid_x: int
    grid_y: int
    role: QubitRole


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
    p_measure: float = 0.0,
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
        before_measure_flip_probability=float(p_measure),
        after_reset_flip_probability=float(p_pauli),
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


def _as_flat_circuit_text(circuit) -> str:
    flat = circuit.flattened() if hasattr(circuit, "flattened") else circuit
    return str(flat)


def extract_stim_qubit_layout(circuit, *, distance: int) -> list[StimQubitSite]:
    """Extract Stim qubit coordinates and map them onto the dense d-by-d grid."""
    coord_map: dict[int, tuple[float, float]] = {}
    for line in _as_flat_circuit_text(circuit).splitlines():
        parsed = _parse_qubit_coords(line.strip())
        if parsed is not None:
            qubit, stim_x, stim_y = parsed
            coord_map[int(qubit)] = (float(stim_x), float(stim_y))
    bounds = _coord_bounds_from_map(coord_map)
    sites: list[StimQubitSite] = []
    for qubit, (stim_x, stim_y) in sorted(coord_map.items()):
        grid_x, grid_y = _grid_coord_from_stim(stim_x, stim_y, distance=distance, bounds=bounds)
        role = _role_from_stim_coords(stim_x, stim_y)
        sites.append(
            StimQubitSite(
                qubit=int(qubit),
                stim_x=float(stim_x),
                stim_y=float(stim_y),
                grid_x=grid_x,
                grid_y=grid_y,
                role=role,
            )
        )
    return sites


def _nearest_site(
    sites: Sequence[StimQubitSite],
    *,
    grid_x: int,
    grid_y: int,
    preferred_role: QubitRole | None,
) -> StimQubitSite | None:
    candidates = list(sites)
    if preferred_role is not None:
        role_candidates = [site for site in candidates if site.role == preferred_role]
        if role_candidates:
            candidates = role_candidates
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda site: (site.grid_x - int(grid_x)) ** 2 + (site.grid_y - int(grid_y)) ** 2,
    )


def _tick_count(circuit) -> int:
    return sum(1 for line in _as_flat_circuit_text(circuit).splitlines() if _operation_name(line.strip()) == "TICK")


def _round_to_tick(round_id: int, *, rounds: int, total_ticks: int) -> int:
    if int(total_ticks) <= 0:
        return 0
    return max(0, min(int(total_ticks) - 1, int(round(int(round_id) * int(total_ticks) / max(int(rounds), 1)))))


def _pauli_for_data_erasure(rng: np.random.Generator) -> str:
    return str(rng.choice(np.asarray(["X", "Y", "Z"], dtype=object)))


def _pauli_for_measure_erasure(role: QubitRole, basis: str, rng: np.random.Generator) -> str:
    if role == QubitRole.X_MEASURE:
        return "Z"
    if role == QubitRole.Z_MEASURE:
        return "X"
    return _pauli_for_data_erasure(rng)


def _append_pauli_error(insertions: dict[int, dict[str, list[int]]], tick: int, pauli: str, qubit: int) -> None:
    if pauli not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported Pauli {pauli!r}")
    insertions.setdefault(int(tick), {}).setdefault(pauli, []).append(int(qubit))


def _physical_mask(sidecar: ErasureSidecarBatch, *, data: bool, shot_index: int) -> np.ndarray:
    if data:
        source = sidecar.physical_data_erasures if sidecar.physical_data_erasures is not None else sidecar.data_erasures
    else:
        source = sidecar.physical_measure_erasures if sidecar.physical_measure_erasures is not None else sidecar.measure_erasures
    return np.asarray(source[int(shot_index)], dtype=np.uint8)


def build_per_erasure_stim_circuit(
    base_circuit,
    *,
    sidecar: ErasureSidecarBatch,
    shot_index: int,
    distance: int,
    rounds: int,
    rng: np.random.Generator,
):
    """Return a per-shot circuit with erasures converted into Pauli errors.

    The inserted instructions are noise instructions with probability one, e.g.
    ``X_ERROR(1) q``. They do not create measurement records, so existing
    ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` references remain valid.
    """
    stim_module = _require_stim()
    sites = extract_stim_qubit_layout(base_circuit, distance=distance)
    total_ticks = _tick_count(base_circuit)
    insertions: dict[int, dict[str, list[int]]] = {}
    data_mask = _physical_mask(sidecar, data=True, shot_index=shot_index)
    measure_mask = _physical_mask(sidecar, data=False, shot_index=shot_index)

    for t in range(int(rounds)):
        tick = _round_to_tick(t, rounds=rounds, total_ticks=total_ticks)
        for x in range(int(distance)):
            for y in range(int(distance)):
                if bool(data_mask[t, x, y]):
                    site = _nearest_site(sites, grid_x=x, grid_y=y, preferred_role=QubitRole.DATA)
                    if site is not None:
                        _append_pauli_error(insertions, tick, _pauli_for_data_erasure(rng), site.qubit)
                if bool(measure_mask[t, x, y]):
                    role = SurfacePatchGeometry(distance=int(distance)).role_at(x, y)
                    preferred = role if role != QubitRole.DATA else _detector_role_for_grid(
                        SurfacePatchGeometry(distance=int(distance)), x, y, sidecar.basis
                    )
                    site = _nearest_site(sites, grid_x=x, grid_y=y, preferred_role=preferred)
                    if site is not None:
                        _append_pauli_error(
                            insertions,
                            tick,
                            _pauli_for_measure_erasure(preferred, sidecar.basis, rng),
                            site.qubit,
                        )

    output_lines: list[str] = []
    seen_tick = -1
    inserted_zero_tick = False
    for line in _as_flat_circuit_text(base_circuit).splitlines():
        stripped = line.strip()
        if _operation_name(stripped) == "TICK":
            seen_tick += 1
            output_lines.append(line)
            for pauli, qubits in sorted(insertions.get(seen_tick, {}).items()):
                if qubits:
                    output_lines.append(f"{pauli}_ERROR(1) " + " ".join(str(q) for q in sorted(set(qubits))))
            inserted_zero_tick = True
        else:
            output_lines.append(line)

    if not inserted_zero_tick and insertions:
        for by_pauli in insertions.values():
            for pauli, qubits in sorted(by_pauli.items()):
                if qubits:
                    output_lines.append(f"{pauli}_ERROR(1) " + " ".join(str(q) for q in sorted(set(qubits))))

    return stim_module.Circuit("\n".join(output_lines) + "\n")


def sample_per_erasure_stim_shot(
    base_circuit,
    *,
    sidecar: ErasureSidecarBatch,
    shot_index: int,
    distance: int,
    rounds: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one detector/observable row from a per-erasure Stim circuit."""
    rng = np.random.default_rng(seed)
    circuit = build_per_erasure_stim_circuit(
        base_circuit,
        sidecar=sidecar,
        shot_index=shot_index,
        distance=distance,
        rounds=rounds,
        rng=rng,
    )
    sampler = _compile_sampler(circuit, seed=seed, detector=True)
    try:
        dets, obs = sampler.sample(shots=1, separate_observables=True)
    except TypeError:
        dets = sampler.sample(shots=1)
        obs = np.zeros((1, 1), dtype=np.uint8)
    return np.asarray(dets[0], dtype=np.uint8), np.asarray(obs[0], dtype=np.uint8)


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
    p_measure: float = 0.0,
    seed: int | None = None,
) -> StimSampleBatch:
    """Sample detector and observable data from an unmodified Stim circuit."""
    circuit = build_base_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        basis=basis,
        p_pauli=p_pauli,
        p_measure=p_measure,
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
    p_false_positive: float = 0.0,
    p_false_negative: float = 0.0,
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

    true_erasures = rng.random((int(num_shots), int(rounds), h, w)) < float(p_erasure)
    false_positives = rng.random((int(num_shots), int(rounds), h, w)) < float(p_false_positive)
    false_negatives = rng.random((int(num_shots), int(rounds), h, w)) < float(p_false_negative)
    ambiguity_draws = rng.random((int(num_shots), int(rounds), h, w)) < float(p_ambiguity)
    ambiguity_strength = rng.uniform(0.0, 1.0, size=(int(num_shots), int(rounds), h, w)).astype(np.float32)
    ambiguity = np.where(ambiguity_draws, ambiguity_strength, 0.0).astype(np.float32)

    data_sites = data_mask[None, None, :, :]
    measure_sites = measure_mask[None, None, :, :]
    physical_data = true_erasures & data_sites
    physical_measure = true_erasures & measure_sites
    false_positive_events = false_positives & ~true_erasures
    false_negative_events = false_negatives & true_erasures
    observed_flags = np.logical_or(true_erasures & ~false_negative_events, false_positive_events)
    observed_flags = np.logical_or(observed_flags, ambiguity_draws)
    data_erasures = observed_flags & data_sites
    measure_erasures = observed_flags & measure_sites

    return ErasureSidecarBatch(
        basis=str(basis).strip().upper(),
        data_erasures=data_erasures.astype(np.uint8),
        measure_erasures=measure_erasures.astype(np.uint8),
        readout_ambiguity=ambiguity,
        site_records=site_records,
        physical_data_erasures=physical_data.astype(np.uint8),
        physical_measure_erasures=physical_measure.astype(np.uint8),
        false_positive_data=(false_positive_events & data_sites).astype(np.uint8),
        false_positive_measure=(false_positive_events & measure_sites).astype(np.uint8),
        false_negative_data=(false_negative_events & data_sites).astype(np.uint8),
        false_negative_measure=(false_negative_events & measure_sites).astype(np.uint8),
    )


def _detector_role_for_grid(geometry: SurfacePatchGeometry, x: int, y: int, basis: str) -> QubitRole:
    role = geometry.role_at(x, y)
    if role != QubitRole.DATA:
        return role
    return QubitRole.X_MEASURE if str(basis).strip().upper() == "X" else QubitRole.Z_MEASURE


def generate_stim_assisted_events(
    distance: int,
    rounds: int,
    shot_index: int,
    detector_samples: np.ndarray,
    basis: str,
    detector_coordinates: dict[int, tuple[float, ...]] | None = None,
    erasure_sidecar: ErasureSidecarBatch | None = None,
) -> list[HardwareEvent]:
    """Convert one sampled Stim shot into hardware-style telemetry events.

    The detector row is assumed to already include any erasure-induced physical
    effects when generated through ``sample_per_erasure_stim_shot``. The sidecar
    therefore adds observed erasure telemetry only; it does not fabricate extra
    syndrome events.
    """
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

    return events


def pack_erasure_sidecar(sidecar: ErasureSidecarBatch) -> dict[str, np.ndarray]:
    """Pack sidecar arrays for scalable shard storage."""
    physical_data = (
        sidecar.physical_data_erasures
        if sidecar.physical_data_erasures is not None
        else sidecar.data_erasures
    )
    physical_measure = (
        sidecar.physical_measure_erasures
        if sidecar.physical_measure_erasures is not None
        else sidecar.measure_erasures
    )
    false_positive_data = (
        sidecar.false_positive_data
        if sidecar.false_positive_data is not None
        else np.zeros_like(sidecar.data_erasures)
    )
    false_positive_measure = (
        sidecar.false_positive_measure
        if sidecar.false_positive_measure is not None
        else np.zeros_like(sidecar.measure_erasures)
    )
    false_negative_data = (
        sidecar.false_negative_data
        if sidecar.false_negative_data is not None
        else np.zeros_like(sidecar.data_erasures)
    )
    false_negative_measure = (
        sidecar.false_negative_measure
        if sidecar.false_negative_measure is not None
        else np.zeros_like(sidecar.measure_erasures)
    )

    def pack(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.uint8)
        return np.packbits(mask.reshape(mask.shape[0], -1), axis=1)

    return {
        "data_erasures": pack(sidecar.data_erasures),
        "measure_erasures": pack(sidecar.measure_erasures),
        "physical_data_erasures": pack(physical_data),
        "physical_measure_erasures": pack(physical_measure),
        "false_positive_data": pack(false_positive_data),
        "false_positive_measure": pack(false_positive_measure),
        "false_negative_data": pack(false_negative_data),
        "false_negative_measure": pack(false_negative_measure),
        "readout_ambiguity": sidecar.readout_ambiguity.astype(np.float32, copy=False),
        "shape": np.asarray(sidecar.data_erasures.shape, dtype=np.int64),
    }


def logical_label_from_targets(targets: np.ndarray) -> np.ndarray:
    """Return a simple one-observable logical label for scaffold datasets."""
    return np.asarray([int(np.sum(targets[0] + targets[1]) % 2)], dtype=np.uint8)


def pack_events(events: Sequence[HardwareEvent]) -> str:
    """Compact JSONL-ish representation for optional shard provenance."""
    return "\n".join(str(event.to_dict()) for event in events)
