# Gradient error bound under lossy simulation (sketch)

Goal: extend qubit's state-fidelity theorem to gradients computed by
the adjoint method, when the forward state is stored through lossy
(int16-tier) checkpoints.

## Setup

Circuit U = U_L ... U_1 with parameters theta_k inside some U_k.
Observable O (Pauli-Z string, ||O|| = 1). Cost C(theta) = <psi|O|psi>,
|psi> = U|0>.

Adjoint method computes all dC/dtheta_k in one forward + one backward
pass, holding two states:
  |phi>   : forward state, progressively un-evolved backward
  |lambda>: O|psi>, un-evolved in lockstep
Gradient element: dC/dtheta_k = 2 Im <lambda_k | dU_k/dtheta_k | phi_k>
(for rotation gates, dU/dtheta = -(i/2) G U with generator G, ||G||=1).

## Perturbation model

Compression injects, at each checkpoint boundary i, an additive error
delta_i with ||delta_i|| <= eps_i into whichever trajectory reads the
checkpoint. Unitaries preserve norms, so by the same triangle-inequality
argument as qubit's Lemma 1, at the moment the gradient element k is
evaluated:

  || |phi_k~>    - |phi_k>    || <= D_phi    = sum of eps_i on the phi path
  || |lambda_k~> - |lambda_k> || <= D_lambda = sum on the lambda path

## Bound

The gradient element is a bilinear form g = 2 Im <lambda|A|phi> with
||A|| <= 1/2 (A = G U_k / 2 up to phase). For perturbed vectors:

  |g~ - g| <= 2 ||A|| ( ||d_lambda|| ||phi|| + ||lambda|| ||d_phi||
                        + ||d_lambda|| ||d_phi|| )
           <= D_lambda + D_phi + D_lambda D_phi        (unit vectors)

So to first order:

  |grad_k error| <= D_phi + D_lambda   for every k.

With a shared budget D = D_phi + D_lambda enforced by the runtime
(same accounting machinery as qubit's state bound), every gradient
component is off by at most ~D. LINEAR in the budget — this is the
property the Phase 0 spike must confirm empirically.

## Consequences for training

- Gradient descent with epsilon-bounded gradient error converges to a
  neighborhood of a stationary point of radius O(epsilon / mu) under
  standard smoothness assumptions; for small budgets (D ~ 1e-3) this
  is negligible against shot noise people already tolerate on real QPUs.
- The budget knob becomes a *training* knob: loose budget early
  (cheap, coarse gradients), tighten as the optimizer converges —
  an annealed-precision training schedule with guarantees. Possibly a
  paper-2 subsection of its own.

## Spike verification (Phase 0, compress_spike.cpp)

The predicted linear bound holds empirically. Measured max gradient
error vs total injected norm D on a 6-qubit, 3-layer TFIM circuit:

| levels | D       | max grad err | err/D |
|--------|---------|--------------|-------|
| 4      | 5.1e+01 | 1.81e+00     | 0.035 |
| 16     | 9.2e+00 | 2.44e-01     | 0.026 |
| 64     | 2.3e+00 | 5.97e-02     | 0.025 |
| 256    | 5.8e-01 | 1.54e-02     | 0.027 |
| 1024   | 1.4e-01 | 5.65e-03     | 0.039 |

err/D is ~constant (~0.03) over three orders of magnitude: the error
is linear in D, and the worst-case bound |grad err| <= D holds with
~30x margin. Training (Step 3) converges with a graceful, budget-tunable
gap, so the deterministic quantization bias (open question 1) does not
break optimization in practice at these budgets.

## Open questions (spike must answer)

1. Quantization error is deterministic given the state, not zero-mean:
   does it introduce a systematic BIAS in the gradient direction across
   steps, or does parameter movement decorrelate it in practice?
   (If biased: dithered quantization as fix.)
2. The lambda trajectory starts from O|psi~| (already-perturbed psi):
   does the composition stay within the D_lambda accounting or does it
   pick up a factor ||O||-dependent term? (Pauli O: isometry on the
   relevant subspace, should be fine — verify.)
3. Checkpoint placement: per-layer vs per-segment trade-off between
   memory (number of checkpoints) and injected error (recompressions).

## References to build on

- qubit paper 1: Lemma (linear L2 composition), Theorem (fidelity
  bound), inversion into enforced budgets.
- Jones & Gacon 2020 (adjoint differentiation for simulators).
- Chen et al. 2016 (gradient checkpointing, classical analogue).
