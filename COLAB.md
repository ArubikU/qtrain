# Running the GPU demo on a Colab T4

The engine (adjoint kernels) is part of the qubit library
(`qubit/include/qubit/adjoint*.{h,cuh}`); qtrain is the implementation
(bindings + PennyLane device). So the T4 build clones **qubit with its
qtrain submodule** — one recursive clone brings both.

Runtime must be **GPU (T4)**: Runtime → Change runtime type → T4.

## Turnkey — paste one cell, it does everything

Public repos, no token. Clones qubit (the library) + qtrain (the impl,
its submodule) and builds the GPU module against qubit's headers.

```bash
%%bash
git clone --quiet --recurse-submodules https://github.com/ArubikU/qubit.git
cd qubit/qtrain
pip -q install pybind11 numpy
nvcc -O2 -std=c++17 -arch=sm_75 --shared -DQUBIT_CUDA -Xcompiler "-fPIC -fopenmp -DNDEBUG" \
    $(python3 -m pybind11 --includes) -I ../include \
    bindings/qubit_gpu.cu ../src/backend_gpu.cu \
    -o qubit_gpu_native$(python3-config --extension-suffix)
python bench/phase4_t4.py
```

The dense adjoint runs on qubit's own GPU backend (`DenseGPU`, its
`k_apply` kernel), so the build compiles + links `src/backend_gpu.cu`.

`-arch=sm_75` is the T4 (Turing). Other cards: A100 `sm_80`, L4 `sm_89`,
RTX 30xx `sm_86`.

## Full build (CPU + GPU modules)

From `qubit/qtrain` after the recursive clone:

```bash
!bash bindings/build_gpu.sh sm_75   # builds qubit_native + qubit_gpu_native
```

Other cards: A100 `sm_80`, L4 `sm_89`, RTX 30xx `sm_86`.

## What it shows

- dense complex64 adjoint ceiling on 16 GB (n up to ~29),
- int16-storage ceiling one qubit higher (n up to ~30),
- a 28-qubit VQE training with the compressed adjoint.

To also build the CPU module and run the full test/bench suite, use
`bash bindings/build_gpu.sh sm_75` (needs the parent `../include/qubit`).
