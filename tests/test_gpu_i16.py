"""
int16-storage GPU adjoint (qubit_gpu_native.GPUCircuitQ) sanity.

The int16 path stores each trajectory in 4 B/amp (half of complex64), so a
6 GB card reaches ~29 qubits where the dense path OOMs at ~28. It is coarse
(lambda uses an a-priori scale), so the check is gradient DIRECTION, not
magnitude: cosine similarity to the exact CPU adjoint must be high.

Run: py -3.12 tests/test_gpu_i16.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
import qubit_gpu_native as qg
from vqe_helpers import hea_ansatz, tfim

fails = 0
rng = np.random.default_rng(3)
for n, L in [(4, 2), (6, 3), (8, 3), (10, 3)]:
    th = rng.uniform(-math.pi, math.pi, L * n * 2)
    H = tfim(n)
    _, gc = hea_ansatz(qn.ACircuit, n, L, th).value_and_grad(H)
    _, gq, D = hea_ansatz(qg.GPUCircuitQ, n, L, th).value_and_grad(H)
    gc, gq = np.array(gc), np.array(gq)
    cos = float(np.dot(gc, gq) / (np.linalg.norm(gc) * np.linalg.norm(gq) + 1e-12))
    ok = cos > 0.99
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] n={n} L={L}  grad cos-sim={cos:.5f}  D={D:.2e}")

print(f"\nmemory: int16 2-traj = 4 B/amp/traj -> ~29 qubits on 6 GB "
      f"({2*4*(1<<29)/1024**3:.1f} GB) vs dense complex64 ~28 "
      f"({2*8*(1<<28)/1024**3:.1f} GB).")
print("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED")
sys.exit(1 if fails else 0)
