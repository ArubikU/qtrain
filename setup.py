"""
Build the qubit_native pybind11 extension over the header-only CPU engine.

    py -3.12 setup.py build_ext --inplace

Produces qubit_native.*.pyd next to this file. The engine is header-only
(../include/qubit/qubit.h); no separate compile/link of the library needed.
CPU-only for Phase 1 (QUBIT_CUDA left undefined).
"""
import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

here = os.path.dirname(os.path.abspath(__file__))
include = os.path.normpath(os.path.join(here, "..", "include"))

ext_modules = [
    Pybind11Extension(
        "qubit_native",
        ["bindings/qubit_py.cpp"],
        include_dirs=[include],
        cxx_std=17,
    )
]

setup(
    name="pennylane-qubit",
    version="0.1.0",
    description="PennyLane device backed by the qubit state-vector simulator",
    packages=["pennylane_qubit"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pennylane>=0.40", "numpy"],
    entry_points={
        "pennylane.plugins": [
            "qubit.simulator = pennylane_qubit:QubitDevice",
        ],
    },
)
