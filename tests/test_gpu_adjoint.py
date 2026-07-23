"""
GPU adjoint (qubit_gpu_native, complex64) vs CPU adjoint (complex128).

Builds the same ansatz + TFIM on both, compares value and gradient. GPU is
single-precision so tolerance is ~1e-3. Run: py -3.12 tests/test_gpu_adjoint.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
import qubit_gpu_native as qg
from vqe_helpers import hea_ansatz, tfim

fails = 0
rng = np.random.default_rng(3)
for n, layers in [(4, 2), (6, 3), (8, 3)]:
    theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
    H = tfim(n)
    vc, gc = hea_ansatz(qn.ACircuit, n, layers, theta).value_and_grad(H)
    vg, gg = hea_ansatz(qg.GPUCircuit, n, layers, theta).value_and_grad(H)
    verr = abs(vc - vg)
    gerr = np.max(np.abs(np.array(gc) - np.array(gg)))
    ok = verr < 1e-3 and gerr < 1e-3
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] n={n} L={layers}  value err={verr:.2e}  grad err={gerr:.2e}")

print("\n" + ("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
sys.exit(1 if fails else 0)
