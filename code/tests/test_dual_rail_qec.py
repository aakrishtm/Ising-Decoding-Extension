# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from dual_rail_qec.data.datasets import DualRailShardDataset
from dual_rail_qec.data.export import resolve_data_source, write_dataset
from dual_rail_qec.data.simulator import (
    build_base_surface_code_circuit,
    generate_erasure_sidecar,
    generate_stim_assisted_events,
    pack_erasure_sidecar,
    sample_per_erasure_stim_shot,
    stim,
)
from dual_rail_qec.telemetry.schema import DualRailState, HardwareEvent, QubitRole
from dual_rail_qec.telemetry.tensorize import make_local_targets, tensorize_events

if torch is not None:
    from dual_rail_qec.decoding.residual import (
        logical_prediction_from_corrections,
        residual_syndrome_inputs,
        threshold_corrections,
    )
    from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder


class TestDualRailQEC(unittest.TestCase):

    def test_tensorize_events_shape_and_channels(self):
        events = [
            HardwareEvent(
                round_id=0,
                qubit_id="X_MEASURE:0:1",
                x=0,
                y=1,
                role=QubitRole.X_MEASURE,
                dual_rail_state=DualRailState.LOGICAL_01,
                readout_confidence=1.0,
                syndrome_parity=True,
            ),
            HardwareEvent(
                round_id=1,
                qubit_id="DATA:0:0",
                x=0,
                y=0,
                role=QubitRole.DATA,
                dual_rail_state=DualRailState.LEAKAGE_00,
                readout_confidence=0.95,
            ),
            HardwareEvent(
                round_id=1,
                qubit_id="Z_MEASURE:1:0",
                x=1,
                y=0,
                role=QubitRole.Z_MEASURE,
                dual_rail_state=DualRailState.AMBIGUOUS,
                readout_confidence=0.25,
                syndrome_parity=None,
            ),
        ]
        tensor = tensorize_events(events, distance=3, rounds=2)
        self.assertEqual(tensor.shape, (7, 2, 3, 3))
        self.assertEqual(float(tensor[0, 0, 0, 1]), 1.0)
        self.assertEqual(float(tensor[2, 1, 0, 0]), 1.0)
        self.assertEqual(float(tensor[3, 1, 1, 0]), 1.0)
        self.assertGreater(float(tensor[6, 1, 1, 0]), 0.0)

        targets = make_local_targets(tensor)
        self.assertEqual(targets.shape, (4, 2, 3, 3))

    def test_dataset_writer_and_reader_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = write_dataset(
                output_root=Path(tmp),
                distance=3,
                rounds=3,
                num_shards=2,
                samples_per_shard=4,
                p_erasure=0.05,
                p_pauli=0.01,
                p_ambiguity=0.02,
                seed=123,
                data_source="synthetic",
            )
            ds = DualRailShardDataset(dataset_dir)
            self.assertEqual(len(ds), 8)
            shards = list(ds.iter_shards())
            self.assertEqual(len(shards), 2)
            self.assertEqual(shards[0]["inputs"].shape, (4, 7, 3, 3, 3))
            self.assertEqual(shards[0]["targets"].shape, (4, 4, 3, 3, 3))
            self.assertEqual(shards[0]["logical_labels"].shape, (4, 1))
            self.assertEqual(shards[0]["inputs"].dtype, np.float32)
            sample = ds.get_sample(7)
            self.assertEqual(sample["inputs"].shape, (7, 3, 3, 3))

    def test_data_source_resolution(self):
        self.assertEqual(resolve_data_source("synthetic"), "synthetic")
        if stim is None:
            with self.assertRaises(RuntimeError):
                resolve_data_source("stim")
            self.assertEqual(resolve_data_source("auto"), "synthetic")

    def test_erasure_sidecar_is_shot_round_site_aligned(self):
        rng = np.random.default_rng(5)
        sidecar = generate_erasure_sidecar(
            distance=3,
            rounds=3,
            num_shots=4,
            p_erasure=0.1,
            p_ambiguity=0.05,
            basis="X",
            rng=rng,
        )
        self.assertEqual(sidecar.data_erasures.shape, (4, 3, 3, 3))
        self.assertEqual(sidecar.measure_erasures.shape, (4, 3, 3, 3))
        self.assertEqual(sidecar.readout_ambiguity.shape, (4, 3, 3, 3))
        packed = pack_erasure_sidecar(sidecar)
        self.assertEqual(tuple(packed["shape"]), (4, 3, 3, 3))
        self.assertIn("data_erasures", packed)
        self.assertIn("measure_erasures", packed)

    def test_erasure_sidecar_adds_telemetry_not_synthetic_syndrome(self):
        rng = np.random.default_rng(9)
        sidecar = generate_erasure_sidecar(
            distance=3,
            rounds=3,
            num_shots=1,
            p_erasure=1.0,
            p_ambiguity=0.0,
            basis="X",
            rng=rng,
        )
        events = generate_stim_assisted_events(
            distance=3,
            rounds=3,
            shot_index=0,
            detector_samples=np.zeros((1, 0), dtype=np.uint8),
            basis="X",
            detector_coordinates={},
            erasure_sidecar=sidecar,
        )
        tensor = tensorize_events(events, distance=3, rounds=3)
        self.assertGreater(float(tensor[2].sum() + tensor[3].sum()), 0.0)
        self.assertEqual(float(tensor[0].sum() + tensor[1].sum()), 0.0)

    def test_per_erasure_stim_sampler_smoke(self):
        if stim is None:
            self.skipTest("stim is not installed")
        base_circuit = build_base_surface_code_circuit(
            distance=3,
            rounds=3,
            basis="X",
            p_pauli=0.0,
            p_measure=0.0,
        )
        sidecar = generate_erasure_sidecar(
            distance=3,
            rounds=3,
            num_shots=1,
            p_erasure=0.25,
            p_ambiguity=0.0,
            p_false_positive=0.01,
            p_false_negative=0.01,
            basis="X",
            rng=np.random.default_rng(42),
        )
        det_row, obs_row = sample_per_erasure_stim_shot(
            base_circuit,
            sidecar=sidecar,
            shot_index=0,
            distance=3,
            rounds=3,
            seed=42,
        )
        self.assertEqual(det_row.ndim, 1)
        self.assertEqual(obs_row.ndim, 1)
        self.assertGreater(det_row.size, 0)

    def test_model_and_residual_smoke(self):
        if torch is None:
            self.skipTest("torch is not installed")
        model = DualRailCNN3DPreDecoder(hidden_channels=8, depth=2)
        inputs = torch.zeros((2, 7, 3, 3, 3), dtype=torch.float32)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 4, 3, 3, 3))
        candidates = threshold_corrections(logits, threshold=0.5)
        residual = residual_syndrome_inputs(inputs, candidates)
        labels = logical_prediction_from_corrections(candidates)
        self.assertEqual(tuple(residual.shape), tuple(inputs.shape))
        self.assertEqual(tuple(labels.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
