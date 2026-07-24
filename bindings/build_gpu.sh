#!/usr/bin/env bash
# Build the CPU (qubit_native) and GPU (qubit_gpu_native) extension modules
# on Linux — for Colab / T4. Run from the qtrain/ directory:
#     bash bindings/build_gpu.sh [sm_arch]
# sm_arch defaults to sm_75 (T4 / Turing). Use sm_80 for A100, sm_86 for 3060.
set -e
ARCH="${1:-sm_75}"
PYINC=$(python3 -m pybind11 --includes)
# python3-config is missing on some setups (e.g. Colab); sysconfig always works
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo ">> CPU module (qubit_native)"
c++ -O3 -std=c++17 -fopenmp -shared -fPIC $PYINC \
    -I ../include \
    bindings/qubit_py.cpp -o "qubit_native${EXT}"

echo ">> GPU module (qubit_gpu_native), arch=${ARCH}"
nvcc -O2 -std=c++17 -arch="${ARCH}" --shared -DQUBIT_CUDA \
    -Xcompiler "-fPIC -fopenmp -DNDEBUG" \
    $PYINC -I ../include \
    bindings/qubit_gpu.cu ../src/backend_gpu.cu -o "qubit_gpu_native${EXT}"

echo ">> done: $(ls qubit_native${EXT} qubit_gpu_native${EXT})"
