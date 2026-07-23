# Python bindings — `qubit_native`

pybind11 wrapper over qubit's header-only CPU engine
(`../include/qubit/qubit.h`). Exposes `Circuit`, `RunOptions`, `Result`,
`run()`, and the `Device`/`Precision` enums to Python.

## Build

setuptools' MSVC auto-detection misses the VS2019 BuildTools install on
this machine, so the module is compiled by invoking `cl` directly inside
the VS build environment:

```
cd qtrain
cmd /c bindings\build.bat
```

Produces `qubit_native.pyd` in `qtrain/`. CPU-only (Phase 1);
`QUBIT_CUDA` is left undefined, so `Device.GPU` falls back to CPU.

Paths in `build.bat` (Python include/libs, pybind11 include) are hard-
coded for this environment; adjust if Python moves.

## Use

```python
import qubit_native as qn
c = qn.Circuit(2)
c.h(0); c.cnot(0, 1)
r = qn.run(c)
print(dict(r.counts))            # {'00': 512, '11': 512}
print(r.expectation_z([0, 1]))   # 1.0
```

The PennyLane device (`../pennylane_qubit`) drives this module. See
`../tests/test_device.py` and `../examples/vqe_train.py`.
