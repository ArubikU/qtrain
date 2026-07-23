"""
Native adjoint gradient (qubit_native.ACircuit) vs parameter-shift.

Self-contained: builds a hardware-efficient ansatz + TFIM Hamiltonian in
the native ACircuit, compares value_and_grad's adjoint gradient against a
parameter-shift reference computed on the same builder. No PennyLane.

Run: py -3.12 tests/test_adjoint_native.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn

X, Y, Z = 1, 2, 3
INV_SQRT2 = 1.0 / math.sqrt(2.0)


def tfim(n, J=1.0, h=1.0):
    H = []
    for i in range(n - 1):
        H.append((-J, [(i, Z), (i + 1, Z)]))
    for i in range(n):
        H.append((-h, [(i, X)]))
    return H


def build(n, layers, theta):
    """Same ansatz as the device tests: per layer RY,RZ per qubit + CNOT ring.
       All rotations trainable; slot = order encountered."""
    c = qn.ACircuit(n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            c.rot(Y, q, float(theta[p]), True, p); p += 1
            c.rot(Z, q, float(theta[p]), True, p); p += 1
        for q in range(n):
            c.cfixed([q], (q + 1) % n, 0, 1, 1, 0)   # CNOT
    return c


def pshift_grad(n, layers, theta, H):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += math.pi / 2
        tm = theta.copy(); tm[i] -= math.pi / 2
        vp, _ = build(n, layers, tp).value_and_grad(H)
        vm, _ = build(n, layers, tm).value_and_grad(H)
        g[i] = 0.5 * (vp - vm)
    return g


def main():
    rng = np.random.default_rng(2024)
    worst_v, worst_g = 0.0, 0.0
    for trial in range(8):
        n = 3 + int(rng.integers(0, 3))     # 3..5 qubits
        layers = 2 + int(rng.integers(0, 2))
        theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
        H = tfim(n)
        c = build(n, layers, theta)
        val, grad = c.value_and_grad(H)
        grad = np.array(grad)
        gref = pshift_grad(n, layers, theta, H)
        gerr = np.max(np.abs(grad - gref))
        worst_g = max(worst_g, gerr)
        if trial < 3:
            print(f"trial {trial}: n={n} L={layers} params={len(theta)}  "
                  f"value={val:.5f}  max|adj-pshift|={gerr:.2e}")
    print(f"\nworst max|adjoint - parameter-shift| over 8 trials: {worst_g:.3e}")
    ok = worst_g < 1e-8
    print("PASS: native adjoint gradients are correct." if ok
          else "FAIL: adjoint mismatch.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
