# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymatching
import stim
import torch

from dual_rail_qec.data.export import write_dataset
from dual_rail_qec.data.simulator import build_base_surface_code_circuit, generate_erasure_sidecar, pack_erasure_sidecar
from dual_rail_qec.evaluation.evaluate import (
    build_location_oracle,
    evaluate_pipeline,
    load_basis_sidecar,
    unpack_packed_mask,
    _dem_with_injections,
)
from dual_rail_qec.models.cnn3d_predecoder import DualRailCNN3DPreDecoder


class TestDualRailEvaluation(unittest.TestCase):

    def test_unpack_packed_sidecar_masks(self):
        sidecar = generate_erasure_sidecar(
            distance=3,
            rounds=3,
            num_shots=2,
            p_erasure=0.5,
            p_ambiguity=0.0,
            basis="X",
            rng=np.random.default_rng(1),
        )
        packed = pack_erasure_sidecar(sidecar)
        unpacked = unpack_packed_mask(packed["data_erasures"], packed["shape"])
        self.assertEqual(unpacked.shape, sidecar.data_erasures.shape)
        np.testing.assert_array_equal(unpacked, sidecar.data_erasures)

    def test_load_basis_specific_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = write_dataset(
                output_root=Path(tmp),
                distance=3,
                rounds=3,
                num_shards=1,
                samples_per_shard=2,
                p_erasure=0.2,
                p_pauli=0.01,
                p_measure=0.001,
                p_false_positive=0.01,
                p_false_negative=0.01,
                seed=7,
                data_source="stim",
            )
            sidecar_x = load_basis_sidecar(dataset_dir, "X")
            sidecar_z = load_basis_sidecar(dataset_dir, "Z")
            self.assertEqual(sidecar_x.data_erasures.shape, (2, 3, 3, 3))
            self.assertEqual(sidecar_z.measure_erasures.shape, (2, 3, 3, 3))
            self.assertEqual(sidecar_x.basis, "X")
            self.assertEqual(sidecar_z.basis, "Z")

    def test_location_oracle_construction_on_tiny_stim_circuit(self):
        circuit = build_base_surface_code_circuit(
            distance=3,
            rounds=3,
            basis="X",
            p_pauli=0.01,
            p_measure=0.001,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        oracle = build_location_oracle(circuit, dem)
        self.assertGreater(len(oracle), 0)
        first_signatures = next(iter(oracle.values()))
        self.assertGreater(len(first_signatures), 0)
        self.assertIsInstance(first_signatures[0][0], stim.DemTarget)

    def test_dem_injection_accepts_dem_targets_and_decodes(self):
        circuit = build_base_surface_code_circuit(
            distance=3,
            rounds=3,
            basis="X",
            p_pauli=0.01,
            p_measure=0.001,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        oracle = build_location_oracle(circuit, dem)
        signature = next(sig for signatures in oracle.values() for sig in signatures if sig)
        injected = _dem_with_injections(dem, [signature], 0.499999)
        matching = pymatching.Matching.from_detector_error_model(injected)
        prediction = matching.decode(np.zeros(injected.num_detectors, dtype=np.uint8))
        self.assertEqual(np.asarray(prediction).ndim, 1)

    def test_tiny_evaluator_smoke_all_dem_injection_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = write_dataset(
                output_root=root / "data",
                distance=3,
                rounds=3,
                num_shards=1,
                samples_per_shard=2,
                p_erasure=0.2,
                p_pauli=0.01,
                p_measure=0.001,
                seed=11,
                data_source="stim",
            )
            model = DualRailCNN3DPreDecoder(hidden_channels=4, depth=2)
            checkpoint = root / "model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": {
                        "in_channels": 7,
                        "out_channels": 4,
                        "hidden_channels": 4,
                        "depth": 2,
                        "kernel_size": 3,
                    },
                },
                checkpoint,
            )
            results = evaluate_pipeline(
                dataset_dir=dataset_dir,
                checkpoint=checkpoint,
                batch_size=2,
                max_shots=2,
                bases=("X",),
                device="cpu",
            )
            regimes = results["basis"]["X"]["regimes"]
            self.assertIn("vanilla_pymatching", regimes)
            self.assertIn("physical_location_oracle", regimes)
            self.assertIn("observed_sidecar_dem_injection", regimes)
            self.assertIn("cnn_dem_injection", regimes)
            self.assertGreaterEqual(regimes["physical_location_oracle"]["oracle_lookup_hits"], 0.0)


if __name__ == "__main__":
    unittest.main()
