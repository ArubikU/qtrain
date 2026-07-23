# Phase 0 spike — does adjoint-through-compression work?

The gate before any ecosystem investment. Three steps, in order:

## 1. Adjoint on dense CPU (correctness)
Implement textbook adjoint differentiation against the qubit engine
(small n, exact states). Verify against parameter-shift gradients on
random circuits under shared seeds: max |adjoint - pshift| < 1e-6.

## 2. Compression round-trips in the forward pass
Same adjoint pass, but the forward state is round-tripped through
int16 block quantization (the COMPRESSED tier's exact transform) at
each layer boundary, with a budget knob D. Measure:
  max_k |grad_k(compressed) - grad_k(exact)|   vs   D
Expect: ~linear in D (theory/gradient-bound.md). If superlinear or
erratic -> investigate before proceeding.

## 3. Does training survive? (bias check)
Train a small VQE (e.g. 8q transverse-field Ising) with compressed
gradients at several budgets. Compare converged energy vs exact-
gradient training. Checks open question #1 (systematic bias).

## Gate criteria
- PASS: step 1 exact; step 2 linear-ish; step 3 converges to within
  ~D of the exact-training energy.
- FAIL contingency: fall back to parameter-shift on compressed states
  (inherits the state bound trivially, costs 2P evals) and re-scope
  the plan around batched execution instead of adjoint.
