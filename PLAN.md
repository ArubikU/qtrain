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
- [x] Adjoint method (Jones & Gacon 2020) as `qtrain/src/adjoint.h`
      (dense complex128, header-only, namespace qtrain): forward +
      backward pass applying U_k^dagger to both bra/ket trajectories,
      gradient from the generator factor. Built around apply / apply_inv
      / generator / dot so Phase 3 can swap in compressed checkpoints.
- [x] Bound into `qubit_native` as `ACircuit.value_and_grad(H)`; native
      test (`tests/test_adjoint_native.py`) matches parameter-shift to
      7.8e-16 over random ansaetze.
- [x] Wired into the device as `diff_method="adjoint"` via the modern
      device-derivative hooks (setup_execution_config +
      preprocess_transforms + compute_derivatives). `tests/
      test_adjoint_device.py`: gradient matches default.qubit param-shift
      to 5e-16; VQE trains identically (gap 2e-7).
- [x] Scaling benchmark (`bench/grad_scaling.py`): adjoint is 2 passes vs
      parameter-shift's 2P; measured 21x (16 params) -> 133x (96 params)
      speedup, advantage growing with P as predicted.
- [x] Head-to-head vs PennyLane-Lightning adjoint (same qml.grad path,
      complex128). Result: same order of magnitude — roughly parity at
      <=10q, ~3-6x slower at 12-14q (measurement noisy). No performance
      cliff; dense adjoint is table stakes and we clear it within a small
      constant factor. This matters because the winning axis is Phase 4:
      Lightning OOMs at ~30q dense, qubit trains there via compression, so
      "a few x slower per gradient but runs where Lightning can't" is the
      deliberate trade. The 20-133x vs our own param-shift stands as the
      algorithmic (2 vs 2P) result, not a kernel claim.

### Phase 3 — Gradients through compression + theory (M4-7) [CORE]
- [x] Adjoint through compression: `ACircuit.value_and_grad_q(H, levels)`
      round-trips both carried trajectories (phi, lambda) through the
      int16 block-scaled transform at each gate boundary, accumulating the
      injected budget D. Returns (value, grad, D). levels<=0 == exact.
- [x] Theory (paper 2 core): |grad_error| <= D, linear in the injected
      norm (theory/gradient-bound.md, Lemma + Phase 3 confirmation).
- [x] Empirical (bench/phase3_compression.py + tests/test_compression.py):
      err/D constant ~0.01-0.05 (linear), worst-case bound holds as a hard
      assertion, VQE converges under budgets (levels=256 gap 1.2e-4).
- [x] Memory/capability model: uniform int16 clears Lightning's dense 6 GB
      ceiling by +1 qubit (29 vs 28) WITH the error guarantee. 30-32q
      needs tiered blocks on the GPU engine (Phase 4).
- [ ] Tiered (ZERO/COMPRESSED/FULL) storage rather than uniform int16 —
      the sparse-block win that reaches 30-32q. Belongs with the GPU
      engine integration (Phase 4).

### Phase 4 — Killer demo + paper 2 (M7-11)
- [x] Training demo (CPU, `bench/phase4_demo.py`): compressed-gradient VQE
      reaches the same energy as exact-gradient training (difference 6e-5
      at a fine budget, n=10 vs exact ground); a 16-qubit VQE trains end to
      end (~11 s, energy decreasing monotonically). Proves the loop and the
      fidelity at hardware-available scale.
- [x] Capability crossover + memory model: int16 error-bounded storage
      trains one qubit past Lightning's dense 6 GB ceiling (29 vs 28), with
      the gradient bound.
- [x] Paper 2 outline (`paper2/outline.md`): claims mapped to artifacts,
      structure, figure status, and the honest GPU gap. Reuses paper 1's
      peer-review pipeline.
- [x] GPU adjoint (`src/adjoint_gpu.cu`, CUDA complex64, sm_86): matches
      CPU adjoint to ~2e-7 (tests/test_gpu_adjoint.py). GPU vs CPU per
      gradient: 3.3x @16q, 5x @18q, 15x @20q (grows with n). Lambda built
      in place (Pauli involution restore) so only 2 trajectories reside.
      bench/phase4_gpu.py.
- [x] Qubit ceiling on the 6 GB RTX 3060 Laptop: a **28-qubit** adjoint
      gradient computes (4.3 GB, 268M amplitudes) — past the dense
      Lightning/Aer ceiling on the same card. A **22-qubit** VQE trains
      end to end on the GPU with the compressed adjoint.
- [ ] 30-32q: needs int16 DEVICE storage (halve bytes -> 30q at ~4.3 GB).
      value_and_grad_q already models the compression error on-device but
      still allocates complex64; swapping the resident arrays to int16 +
      per-block scale is the remaining step. Then QAOA + T4 replicate.
- [ ] Release qtrain 1.0 + plugin on PyPI (with Phase 1's deferred CI).

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

Phase 3 [CORE]: gradients through compression. adjoint.h's backward pass
already isolates the four state ops (apply / apply_inv / generator / dot)
and the two carried trajectories (phi, lambda). Phase 3 stores those
trajectories as int16-tier compressed checkpoints and round-trips them per
segment, injecting a bounded perturbation whose effect on the gradient is
the paper-2 bound (theory/gradient-bound.md, already spike-validated).
Build: a compressed StateStore behind the same adjoint loop, budget knob D,
gradient-error-vs-D and training-convergence curves. This is the
contribution; everything so far was the substrate to reach it.
