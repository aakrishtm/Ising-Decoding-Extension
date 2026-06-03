# Dual-Rail Evaluation Bridge

This directory evaluates erasure-aware decoding regimes for the dual-rail QEC
pipeline.

## Why Nearby Weighting Failed

The first bridge lowered PyMatching edge weights near dense-grid erasure pixels.
That was too blunt: one grid site can touch many detector-error-model edges, and
nearby detector coordinates are not the same thing as the physical circuit fault
that produced a DEM signature. In small d=3 smoke runs, nearby weighting often
lowered most of the graph and could make LER worse than vanilla PyMatching.

## DEM Injection

The current bridge uses a Stim location oracle:

1. Build the basis circuit.
2. Build `circuit.detector_error_model(decompose_errors=True)`.
3. Call `circuit.explain_detector_error_model_errors(...)`.
4. Map `(tick, stim_qubit)` fault locations to exact DEM target signatures.
5. For each shot, convert erasure masks into `(tick, stim_qubit)` locations.
6. Append matching DEM signatures with probability `p=0.499999`.

The probability is intentionally `0.499999`, not `0.9999`. Matching weights are
log-odds, so a probability near 0.5 makes the edge nearly free without crossing
into invalid or numerically misleading certainty.

## Regimes

`evaluate.py` reports:

- `vanilla_pymatching`: standard PyMatching on the base DEM.
- `physical_location_oracle`: uses physical sidecar erasures, including hidden
  false negatives. This is an oracle upper-bound diagnostic.
- `observed_sidecar_dem_injection`: uses observed erasure flags only.
- `cnn_dem_injection`: reconstructs basis-specific tensors, runs the CNN, and
  injects exact DEM signatures at predicted erasure locations.

Each result includes per-basis and aggregate shots, logical errors, LER, delta
vs vanilla, mask density, injected signatures per shot, oracle hit/miss counts,
and missing erased-location counts.

## Local Evaluation

```bash
source .venv/bin/activate
export PYTHONPATH=.
export MPLCONFIGDIR=/private/tmp/mplconfig

python -m dual_rail_qec.evaluation.evaluate \
  --dataset-dir datasets/dual_rail_d3 \
  --checkpoint outputs/dual_rail_d3_weighted/dual_rail_cnn3d_predecoder.pt \
  --batch-size 64 \
  --max-shots 100 \
  --output-json outputs/dual_rail_d3_weighted/eval_dem_bridge.json
```

If `physical_location_oracle` does not beat vanilla, inspect the diagnostics.
High miss counts mean the sidecar-to-Stim location map is incomplete. High hit
counts with no LER improvement mean the injected DEM signatures are not yet
calibrated to the generated erasure mechanism.

## H100 Sweep Scaffold

Do not run large sweeps locally. On an H100 box, use:

```bash
python -m dual_rail_qec.evaluation.sweep \
  --distances 3 5 7 \
  --p-pauli-values 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --p-erasure 0.01 \
  --p-measure 0.001 \
  --p-false-positive 0.0001 \
  --p-false-negative 0.001 \
  --num-shards 100 \
  --samples-per-shard 10000 \
  --checkpoint-template 'outputs/dual_rail_d{distance}/latest.pt' \
  --output-root sweeps/data \
  --results-path sweeps/results/decoder_sweep.jsonl
```

The sweep emits one JSONL or CSV row per `(distance, noise point, basis, regime)`
plus aggregate rows.
