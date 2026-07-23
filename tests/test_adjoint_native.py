"""
Native adjoint gradient (qubit_native.ACircuit) vs parameter-shift.

Self-contained: builds a hardware-efficient ansatz + TFIM in the native
ACircuit, compares value_and_grad's adjoint gradient against a
parameter-shift reference computed on the same builder. No PennyLane.

Run: py -3.12 tests/test_adjoint_native.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
from vqe_helpers import hea_ansatz, tfim


def pshift_grad(n, layers, theta, H):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += math.pi / 2
        tm = theta.copy(); tm[i] -= math.pi / 2
        vp, _ = hea_ansatz(qn.ACircuit, n, layers, tp).value_and_grad(H)
        vm, _ = hea_ansatz(qn.ACircuit, n, layers, tm).value_and_grad(H)
        g[i] = 0.5 * (vp - vm)
    return g


def main():
    rng = np.random.default_rng(2024)
    worst_g = 0.0
    for trial in range(8):
        n = 3 + int(rng.integers(0, 3))
        layers = 2 + int(rng.integers(0, 2))
        theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
        H = tfim(n)
        val, grad = hea_ansatz(qn.ACircuit, n, layers, theta).value_and_grad(H)
        gerr = np.max(np.abs(np.array(grad) - pshift_grad(n, layers, theta, H)))
        worst_g = max(worst_g, gerr)
        if trial < 3:
            print(f"trial {trial}: n={n} L={layers} params={len(theta)}  "
                  f"value={val:.5f}  max|adj-pshift|={gerr:.2e}")
    print(f"\nworst max|adjoint - parameter-shift| over 8 trials: {worst_g:.3e}")
    ok = worst_g < 1e-8
    print("PASS: native adjoint gradients are correct." if ok else "FAIL: adjoint mismatch.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
