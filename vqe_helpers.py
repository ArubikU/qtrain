"""
Shared VQE building blocks for the tests and benchmarks.

One definition of the ansatz, the TFIM Hamiltonian, and exact
diagonalization — the native circuit builders (ACircuit, GPUCircuit,
GPUCircuitQ) share the same rot/cfixed API, so `hea_ansatz` drives any of
them. Keeps the test/bench files free of copy-pasted circuit code.
"""
import numpy as np

# Pauli generator codes used by the native modules.
X, Y, Z = 1, 2, 3


def hea_ansatz(builder_cls, n, layers, theta):
    """Hardware-efficient ansatz on a native builder: per layer, RY+RZ on
    each qubit (all trainable) then a CNOT ring. `builder_cls` is any of the
    native circuit classes; returns the built circuit."""
    c = builder_cls(n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            c.rot(Y, q, float(theta[p]), True, p); p += 1
            c.rot(Z, q, float(theta[p]), True, p); p += 1
        for q in range(n):
            c.cfixed([q], (q + 1) % n, 0, 1, 1, 0)   # CNOT
    return c


def tfim(n, J=1.0, h=1.0):
    """Transverse-field Ising Hamiltonian: -J sum ZZ - h sum X, as the
    native (coeff, [(wire, pauli)]) term list."""
    H = [(-J, [(i, Z), (i + 1, Z)]) for i in range(n - 1)]
    return H + [(-h, [(i, X)]) for i in range(n)]


def exact_ground(n, H):
    """Smallest eigenvalue of a native Hamiltonian term list (dense; small n)."""
    P = {X: np.array([[0, 1], [1, 0]], complex), Z: np.array([[1, 0], [0, -1]], complex)}
    M = np.zeros((1 << n, 1 << n), dtype=complex)
    for coeff, ops in H:
        mats = {w: P[p] for w, p in ops}
        term = np.array([[1]], complex)
        for q in range(n):
            term = np.kron(mats.get(q, np.eye(2)), term)
        M += coeff * term
    return float(np.min(np.linalg.eigvalsh(M)))
