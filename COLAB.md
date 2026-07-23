# Running the GPU demo on a Colab T4

The GPU module (`qubit_gpu_native`) is self-contained: it only needs
`src/adjoint.h` + `src/adjoint_gpu.cu`, `vqe_helpers.py`, and
`bench/phase4_t4.py` — not the parent `qubit` engine header. So you can run
the headline on a free Colab T4 (16 GB) without cloning the whole repo.

## 1. Get the files into Colab

Easiest: put the `qtrain/` folder in your Google Drive (e.g.
`MyDrive/qtrain`), then in a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/qtrain
```

(If you later push `qtrain` to its own GitHub repo, replace the two lines
above with `!git clone <url> && %cd qtrain`.)

## 2. Build + run — one cell

Colab runtime must be **GPU (T4)**: Runtime → Change runtime type → T4.

```bash
!pip -q install pybind11 numpy
!nvcc -O2 -std=c++17 -arch=sm_75 --shared -Xcompiler "-fPIC -fopenmp -DNDEBUG" \
    $(python3 -m pybind11 --includes) -I src \
    src/adjoint_gpu.cu -o qubit_gpu_native$(python3-config --extension-suffix)
!python bench/phase4_t4.py
```

`-arch=sm_75` is the T4 (Turing). For other cards: A100 `sm_80`,
L4 `sm_89`, RTX 30xx `sm_86`.

## What it shows

- dense complex64 adjoint ceiling on 16 GB (n up to ~29),
- int16-storage ceiling one qubit higher (n up to ~30),
- a 28-qubit VQE training with the compressed adjoint.

To also build the CPU module and run the full test/bench suite, use
`bash bindings/build_gpu.sh sm_75` (needs the parent `../include/qubit`).
