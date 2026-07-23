# qtrain

The variational-training-loop simulator, built on the
[qubit](https://github.com/ArubikU/qubit) engine.

Variational quantum algorithms (VQE, QAOA, QML) do not run one circuit;
they run thousands of nearly identical circuits inside an optimizer
loop. Existing simulators treat every execution as independent. qtrain
treats the *training loop* as the first-class object:

- **Adjoint differentiation**: all parameter gradients in ~2 state
  passes instead of 2P circuit executions.
- **Error-bounded gradients under compression**: train on tiered,
  lossy-compressed states (qubit's VRAM-resident tiers) with a formal
  bound on the gradient error — the capability no other simulator has.
- **PennyLane device plugin** (`pennylane-qubit`): plug directly into
  the tools the target audience already uses.

Headline goal: **train a 30-32 qubit variational circuit on a 6 GB
consumer GPU**, which no existing simulator can do.

Status: planning + risk spike. See [PLAN.md](PLAN.md).

Author: Piero Jose Alarcon Dueñas (ORCID 0009-0008-7075-5501). MIT.
