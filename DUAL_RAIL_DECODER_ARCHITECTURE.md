# Neural QEC for Dual-Rail Architectures

## 1. Project Thesis

This project builds a neural quantum error-correction decoder for surface-code
experiments running on top of dual-rail qubit hardware architectures.

The central claim is that dual-rail hardware changes the decoding problem. In
standard superconducting qubits, leakage or photon-loss-like events are often
silent faults that must be inferred indirectly from stabilizer syndromes. In a
dual-rail cavity architecture, the logical qubit is encoded across two physical
modes. Valid logical states occupy the one-photon subspace, while leakage states
such as `00` and `11` can be flagged directly by the readout stack.

The decoder should therefore not be asked to infer what the hardware already
knows. It should ingest explicit erasure telemetry alongside the usual syndrome
history and use that telemetry as a physical boundary condition for decoding.

## 2. Hardware Assumption

The target hardware architecture uses dual-rail qubits:

- Logical 0 is represented by one photon in one rail and zero in the other.
- Logical 1 is represented by zero photons in one rail and one in the other.
- Leakage states such as `00`, `11`, or ambiguous readout outcomes are detected
  by the hardware measurement stack.
- The dominant physical fault is treated as a located erasure, not as an
  unlocated Pauli error.

This project is not merely simulating erasure flags as an extra toy channel. It
assumes the decoder is operating above a dual-rail hardware layer that emits
native erasure telemetry.

## 3. Hardware Event Schema

The raw input to the decoder should be a structured telemetry stream over
rounds, qubit sites, and readout outcomes.

Each hardware event has the following fields:

```text
round_id: Integer time step.
qubit_id: Stable physical qubit identifier.
x: Integer physical grid coordinate.
y: Integer physical grid coordinate.
role: DATA, X_MEASURE, or Z_MEASURE.
dual_rail_state: LOGICAL_01, LOGICAL_10, LEAKAGE_00, LEAKAGE_11, or AMBIGUOUS.
readout_confidence: Float in [0.0, 1.0].
is_erasure: Boolean derived from the dual-rail state and confidence.
syndrome_parity: Boolean, valid only for non-erased measure qubits.
```

The schema deliberately separates physical telemetry from neural-network tensor
representation. The model should consume a tensorized view of the hardware
record, but the dataset should preserve the raw event contract.

## 4. Decoder Role

The first decoder should be a fault identifier / pre-decoder, not a full
topological decoder.

The pipeline is:

```text
dual-rail hardware telemetry
        -> spacetime tensorization
        -> local neural fault identifier
        -> residual syndrome construction
        -> classical global decoder
        -> logical prediction / correction
```

The neural network is responsible for high-throughput local reasoning over
syndrome and erasure telemetry. It should reduce obvious local faults and lower
the density of the remaining syndrome. A global solver such as MWPM/PyMatching
then handles topological constraints and long-range ambiguity.

This design keeps the neural model aligned with what CNNs are good at:
spatiotemporal local pattern recognition. It avoids asking the CNN to solve the
entire global decoding problem alone.

## 5. Input Tensor Contract

Hardware events are embedded into a dense spacetime tensor:

```text
shape = (B, 7, T, H, W)
```

where:

- `B` is batch size.
- `T` is the number of measurement rounds.
- `H, W` are the dimensions of the embedded surface-code patch.

The seven channels are:

```text
0: Syndrome X
1: Syndrome Z
2: Data Qubit Erasures
3: Measure Qubit Erasures
4: Valid Geometry
5: Boundary Conditions
6: Readout Ambiguity
```

Channel meanings:

- Syndrome channels store raw stabilizer parity differences through time.
- Data-erasure and measure-erasure channels store explicit hardware erasure
  telemetry.
- Valid geometry marks real hardware sites versus empty grid cells.
- Boundary conditions encode the patch edges and surface-code boundary type.
- Readout ambiguity is continuous and should preserve soft readout confidence.

The erasure channels are not labels for the model to guess. They are measured
hardware facts supplied to the model.

## 6. Output Target

The first model should predict local fault or correction candidates rather than
directly predicting the final logical observable.

Candidate output tensor:

```text
shape = (B, 4, T, H, W)
```

Proposed output channels:

```text
0: X-like local correction likelihood
1: Z-like local correction likelihood
2: Measurement-fault likelihood
3: Erasure-associated correction likelihood
```

This output can be converted into a residual syndrome and passed to a global
decoder. The exact output target may be revised once the data generator and
surface-code geometry are locked.

## 7. Distances And Scope

The initial project should support:

```text
d = 3
d = 5
```

Distance 3 is for debugging, tensor-contract validation, and rapid training.
Distance 5 is the main experimental target. Distance 7 is intentionally deferred
until the architecture and evaluation loop are stable.

## 8. Baselines

The project should evaluate at least the following baselines:

```text
1. Standard PyMatching / MWPM with syndrome only.
2. CNN pre-decoder with syndrome and geometry only.
3. CNN pre-decoder with syndrome, geometry, erasure, and ambiguity channels.
4. Optional later baseline: erasure-aware PyMatching edge weighting.
```

The critical ablation is whether explicit dual-rail telemetry improves logical
error rate, syndrome-density reduction, and/or residual solver runtime compared
with syndrome-only decoding.

## 9. Metrics

Primary metrics:

```text
logical error rate
syndrome density before pre-decoding
syndrome density after pre-decoding
residual PyMatching runtime
CNN inference latency
error suppression from d=3 to d=5
```

Secondary metrics:

```text
local fault precision
local fault recall
erasure-associated correction accuracy
readout-ambiguity calibration behavior
throughput in samples per second
```

## 10. Data Artifacts

The generated dataset should preserve both raw hardware-style telemetry and
model-ready tensors.

Recommended artifact layout:

```text
datasets/
  dual_rail_d3/
    metadata.json
    shards/
      shard_00000.npz
      shard_00001.npz
  dual_rail_d5/
    metadata.json
    shards/
      shard_00000.npz
      shard_00001.npz
```

Each shard should contain:

```text
inputs: Tensor-like array with shape (N, 7, T, H, W).
targets: Tensor-like array with local correction/fault targets.
logical_labels: Logical outcome labels for LER evaluation.
raw_event_refs or packed_events: Optional raw telemetry reference.
```

Raw event logs may be stored separately as JSONL or a compact binary format if
they become too large.

## 11. H100 / Colab Training Plan

The training system should be designed for Google Colab H100 execution.

Recommended choices:

```text
precision: bf16 mixed precision
data format: pre-generated shards
dataloader: persistent workers when available
checkpointing: frequent, resume-safe checkpoints
logging: CSV first, optional Weights & Biases later
model compilation: optional torch.compile after correctness is stable
```

Suggested training scale:

```text
d=3 debug: 100k to 1M samples
d=5 initial: 1M to 10M samples
d=5 serious: 50M+ samples if generation and training throughput allow
```

H100 access allows larger data volume and wider CNNs, but the first priority is
locking the telemetry contract and evaluation loop.

## 12. Implementation Spine

The first implementation should be organized around these modules:

```text
dual_rail_qec/
  telemetry/
    schema.py
    geometry.py
    tensorize.py
  data/
    simulator.py
    datasets.py
    export.py
  models/
    cnn3d_predecoder.py
    losses.py
  decoding/
    residual.py
    pymatching_bridge.py
  training/
    train.py
    evaluate.py
    metrics.py
  configs/
    d3.yaml
    d5.yaml
```

Implementation order:

```text
1. telemetry schema
2. surface-code geometry
3. tensorization into (B, 7, T, H, W)
4. synthetic dual-rail telemetry generator
5. dataset shard writer/reader
6. 3D CNN pre-decoder
7. local training loss
8. residual syndrome construction
9. PyMatching bridge
10. d=3 smoke training
11. d=5 main training
12. ablations and plots
```

## 13. Open Design Questions

These should be resolved before writing the core model-training path:

```text
1. Are erasures stored on physical data/measure sites, detector cells, or both?
2. How are data-qubit erasures projected onto the CNN grid?
3. What is the exact local correction target for supervised training?
4. Does the first PyMatching baseline receive erasure-aware weights, or is that
   deferred to a second version?
5. How should ambiguous readout confidence be calibrated into channel 6?
6. Should d=3 and d=5 share one model, or use separate models first?
```

Recommended first answers:

```text
1. Store physical erasures, then tensorize them into grid channels.
2. Keep separate data-erasure and measure-erasure channels.
3. Start with local correction/fault likelihood targets.
4. Defer erasure-aware PyMatching until the CNN ablation is clear.
5. Encode ambiguity as 1.0 - readout_confidence for suspicious events.
6. Train separate d=3 and d=5 models first, then try mixed-distance training.
```
