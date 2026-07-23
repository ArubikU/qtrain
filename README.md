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

## Status

Phases 0-3 done, Phase 4 in progress (CPU). See [PLAN.md](PLAN.md).

- **Phase 0** — risk spike: adjoint-through-compression validated (gate
  passed).
- **Phase 1** — `qubit_native` pybind11 bindings + the `qubit.simulator`
  PennyLane device (analytic + finite-shot). 10/10 vs `default.qubit`.
- **Phase 2** — adjoint differentiation (`src/adjoint.h`, OpenMP),
  `diff_method="adjoint"`, matches parameter-shift to 5e-16; kernel
  competitive with PennyLane-Lightning.
- **Phase 3 [CORE]** — error-bounded gradients through int16 compression:
  |grad error| <= budget D, linear, training-convergent (compression cost
  6e-5 in energy at a fine budget). This is the paper-2 contribution.
- **Phase 4** — CPU training demo + capability crossover done; the GPU
  30-32q headline figure is the remaining engineering
  ([paper2/outline.md](paper2/outline.md)).

### Quick start

```
cmd /c bindings\build.bat          # build qubit_native.pyd (VS2019 + Python 3.12)
py -3.12 tests\test_device.py      # device vs default.qubit (10/10)
py -3.12 tests\test_adjoint_device.py
py -3.12 tests\test_compression.py # Phase 3 gate
py -3.12 bench\phase4_demo.py      # compressed-gradient VQE, 16 qubits
```

```python
import pennylane as qml
from pennylane_qubit import QubitDevice
dev = QubitDevice(wires=4)

@qml.qnode(dev, diff_method="adjoint")
def cost(theta):
    ...
    return qml.expval(H)
```

Author: Piero Jose Alarcon Dueñas (ORCID 0009-0008-7075-5501). MIT.
