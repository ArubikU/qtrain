# qtrain — Plan

## Vision

qubit (the engine) proved that a 6 GB consumer GPU can hold quantum
states past its dense ceiling under an enforced fidelity budget. qtrain
applies that engine to the workload where it matters most and where no
incumbent competes: the **variational training loop** on consumer
hardware. The unique technical contribution is **differentiation
through lossy-compressed simulation with a formal gradient-error
bound** — an extension of qubit's state-fidelity theorem to gradients.

Positioning (deliberate):
- NOT competing with cuQuantum/Lightning on dense kernel speed
  (data-center fight, unwinnable, uninteresting).
- Competing on a capability axis: largest *trainable* variational
  circuit per GB of consumer VRAM, with error guarantees.
- Audience: QML/algorithm developers and students without cluster
  access. Delivery vehicle: a PennyLane device plugin, so adoption
  requires no behavior change from users.

## Why this can be SOTA (and what "SOTA" means here)

1. **Measurable record**: "largest VQE/QAOA trainable on a 6-8 GB
   consumer GPU." Dense simulators cap at 29q on 6 GB; with tiered
   compression + adjoint, 30-32q is plausible. Nobody optimizes or even
   reports this axis.
2. **Novel result**: bounded-error gradients under lossy simulation.
   Prior art compresses *states* (Wu, BMQSim, qubit paper 1); nobody
   differentiates *through* the compression with guarantees.
3. **Adoption surface**: `pip install pennylane-qubit` puts the engine
   in front of the exact audience with the pain.

## Phases

### Phase 0 — Risk spike (2-3 sessions) [GATE]
Validate the hard assumption before investing in ecosystem:
- Implement textbook adjoint differentiation on the dense CPU backend
  (small circuits, exact states).
- Verify adjoint gradients == parameter-shift gradients (differential
  test, same seeds).
- Then the real question: run the SAME adjoint pass where the state is
  round-tripped through int16 tier compression at each step boundary.
  Measure gradient error vs compression budget empirically.
- GATE: if gradient error scales sanely with budget (roughly linearly,
  as the theory sketch predicts), proceed. If it explodes or biases
  training, rethink (fallback: parameter-shift on compressed states,
  which inherits the state bound trivially but costs 2P executions).

### Phase 1 — Python bindings + minimal PennyLane device (M1-2)
- [x] pybind11 wrapper over qubit's public API (Circuit, run, Result) —
      `bindings/qubit_py.cpp` -> `qubit_native` module, built via
      `bindings/build.bat` (direct cl; setuptools misses the 2019
      BuildTools install). Bell/expectation smoke-tested.
- [x] `pennylane-qubit` device (`qubit.simulator`) on PennyLane's modern
      Device API — analytic mode: expval (Pauli/tensor/Hamiltonian via
      pauli_sentence + basis rotation), state, probs. Gate decomposition
      to the native set through `devices.preprocess.decompose`.
- [x] Correctness gate: `tests/test_device.py` — 6/6 match default.qubit
      to ~1e-8 (incl. Rot-decomposition path).
- [x] Trainability gate: `examples/vqe_train.py` — a GradientDescent
      optimizer drives a TFIM VQE on `qubit.simulator` to the SAME energy
      as default.qubit (param-shift through the device). End-to-end proof.
- [x] Finite-shot mode: expval (per-Pauli-term estimate), probs, sample,
      counts — computational-basis samples from the engine mapped through
      PennyLane's process_samples. 4/4 match default.qubit within
      statistical tolerance (200k shots, err ~1e-3).
- [ ] CI wheel build (Windows + Linux), PyPI entry point. DEFERRED to
      Phase 4: nothing worth publishing until adjoint+compression land and
      the API stabilizes; the build is also not yet portable (hard-coded
      paths, setuptools misses VS). Entry point declared in setup.py;
      used via direct import for now.
- Deliverable (met, local): run a PennyLane VQE on the qubit engine, in
  both analytic and finite-shot modes.

### Phase 2 — Adjoint differentiation, dense (M3-4)
- Adjoint method (Jones & Gacon 2020 style) in the C++ engine:
  backward pass applying U_k^dagger while propagating the observable.
- Wire into the PennyLane device as `diff_method="adjoint"`.
- Benchmark gradients/second vs PennyLane-Lightning CPU (and GPU where
  it fits) at 20-26q. Target: competitive dense, not necessarily
  faster — dense adjoint is table stakes, not the contribution.

### Phase 3 — Gradients through compression + theory (M4-7) [CORE]
- Adjoint through tiered states: forward pass stores compressed
  checkpoints (int16 tiers) instead of full states; backward pass
  decompresses per segment. This is exactly gradient checkpointing
  where the checkpoint memory is 2x smaller and error-bounded.
- Theory (paper 2 core): extend the L2 composition lemma to the
  adjoint pass — each decompression injects a bounded perturbation
  into both bra and ket trajectories; triangle inequality gives
  |grad_error| <= f(D) with D the total injected norm. See
  theory/gradient-bound.md for the sketch and open questions.
- Empirical validation: gradient error vs budget curves; training
  convergence (VQE ground-state energy) under budgets.

### Phase 4 — Killer demo + paper 2 (M7-11)
- Train VQE (e.g. transverse-field Ising ground state) and QAOA
  (MaxCut) at 28-32 qubits on the RTX 3060; replicate on free-cloud T4.
- Baselines: Lightning/Aer wherever they fit; document where they OOM.
- Paper 2: "Error-Bounded Gradients Through Lossy Quantum Simulation"
  (working title) -> QCE full paper or ICS; reuse the peer-review
  pipeline from paper 1.
- Release qtrain 1.0 + plugin on PyPI.

### Phase 5 — Stretch (M12+)
- MPS backend behind the same plugin for low-entanglement ansatze.
- Batched parameter sweeps (many circuits, one VRAM residency).
- Hot representation migration during training as memory pressure
  shifts (the "nobody has this" feature from qubit's roadmap).

## Architecture

```
qtrain/
├── src/        C++ additions to the engine: adjoint pass, compressed
│               checkpointing (builds against ../include/qubit)
├── python/     pybind11 bindings (qubit_py module)
├── plugin/     pennylane-qubit device (pure Python, depends on python/)
├── theory/     gradient-bound notes -> paper 2 material
├── bench/      gradient benchmarks vs Lightning/Aer; training curves
└── spike/      Phase 0 risk spike (throwaway allowed)
```

Engine changes land in qubit (parent repo) only when stable; qtrain
pins a qubit version. This subrepo is its own git repository.

## Success metrics (honest)

- Phase 0 gate passed with gradient error ~linear in budget.
- `pip install pennylane-qubit` works on a clean machine.
- Adjoint dense within 2x of Lightning-CPU gradients at 24q.
- A 30q+ VQE trains to chemical-accuracy-style convergence on the 3060
  where Lightning/Aer OOM. This single plot is the project.
- Paper 2 submitted to QCE (minimum) with the gradient bound proved.
- 10 external users / issues from strangers (adoption signal).

## Risks (no sugar)

| Risk | Reality | Mitigation |
|---|---|---|
| Adjoint+compression math doesn't close | Possible; it's research | Phase 0 gate BEFORE ecosystem investment; fallback to parameter-shift-on-compressed (weaker but trivially inherits state bound) |
| Compression error biases training direction | Quantization is deterministic, not zero-mean per step | Measure bias empirically in spike; dithered quantization as fallback |
| PennyLane API churn | Moderate | Pin versions; device API is stable-ish |
| Adoption is slow | Certain | Plugin + 3 runnable tutorials + honest benchmark page; paper drives discovery |
| 30q+ trainable claim fails (paging, like echo-30) | Possible at 31-32q | Anchor the claim at 30q (4 GB compressed + adjoint checkpoints); 31-32 is stretch |
| Solo maintainer burnout | Real over 12 months | Paper 1 ships first (career value banked); phases each end in a usable artifact |

## Immediate next action

Phase 2: adjoint differentiation in the C++ engine, wired into the device
as `diff_method="adjoint"`. The spike already proved the adjoint math
(spike/adjoint_spike.cpp); Phase 2 moves it from throwaway into qubit's
backend and exposes it through `qubit_native` + the device, replacing the
param-shift fallback the device uses today. Then Phase 3 layers the
compressed checkpoints (the core contribution) on top.
