"""
GPU adjoint (qubit_gpu_native, complex64) vs CPU adjoint (complex128).

Builds the same ansatz + TFIM on both, compares value and gradient. GPU is
single-precision so tolerance is ~1e-4. Run: py -3.12 tests/test_gpu_adjoint.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
import qubit_gpu_native as qg

X, Y, Z = 1, 2, 3
fails = 0


def build(mod, n, layers, theta):
    c = mod(n); p = 0
    for _ in range(layers):
        for q in range(n):
            c.rot(Y, q, float(theta[p]), True, p); p += 1
            c.rot(Z, q, float(theta[p]), True, p); p += 1
        for q in range(n):
            c.cfixed([q], (q + 1) % n, 0, 1, 1, 0)
    return c


def tfim(n):
    H = [(-1.0, [(i, Z), (i + 1, Z)]) for i in range(n - 1)]
    return H + [(-1.0, [(i, X)]) for i in range(n)]


rng = np.random.default_rng(3)
for n, layers in [(4, 2), (6, 3), (8, 3)]:
    theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
    H = tfim(n)
    vc, gc = build(qn.ACircuit, n, layers, theta).value_and_grad(H)
    vg, gg = build(qg.GPUCircuit, n, layers, theta).value_and_grad(H)
    verr = abs(vc - vg)
    gerr = np.max(np.abs(np.array(gc) - np.array(gg)))
    ok = verr < 1e-3 and gerr < 1e-3
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] n={n} L={layers}  value err={verr:.2e}  grad err={gerr:.2e}")

print("\n" + ("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
sys.exit(1 if fails else 0)
