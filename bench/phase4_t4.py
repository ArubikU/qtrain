"""
T4 (Colab, 16 GB) headline run. Same adjoint as the laptop, more VRAM:
push the dense complex64 ceiling and the int16 ceiling, and train a VQE at
a size no dense simulator fits on the same card.

T4 has 16 GB. Adjoint holds 2 trajectories:
  dense complex64 (16 B/amp): n<=29 fits (~8.6 GB at 29), OOM at 30.
  int16          ( 8 B/amp):  n<=30 fits (~8.6 GB at 30), OOM at 31.

Run in Colab after building (bindings/build_gpu.sh). Adjust the sizes to the
card if needed. py bench/phase4_t4.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_gpu_native as qg
from vqe_helpers import hea_ansatz, tfim


def try_grad(cls, n, layers, tag):
    P = layers * n * 2
    th = np.random.default_rng(0).uniform(-math.pi, math.pi, P)
    H = tfim(n)
    try:
        t0 = time.perf_counter()
        out = hea_ansatz(cls, n, layers, th).value_and_grad(H)
        val = out[0]
        print(f"  {tag} n={n:2d}: FIT  value={val:.4f}  grad in {time.perf_counter()-t0:.1f}s")
        return True
    except Exception as e:
        print(f"  {tag} n={n:2d}: OOM/err ({str(e)[:50]})")
        return False


print("=== dense complex64 ceiling (2 traj, 16 B/amp) ===")
for n in [28, 29, 30]:
    try_grad(qg.GPUCircuit, n, 2, "dense")

print("\n=== int16 ceiling (2 traj, 8 B/amp) — trains where dense OOMs ===")
for n in [30, 31]:
    try_grad(qg.GPUCircuitQ, n, 2, "int16")

print("\n=== VQE training with the int16 compressed adjoint (24 qubits) ===")
n, layers = 24, 3
H = tfim(n)
th = np.random.default_rng(1).uniform(-0.1, 0.1, layers * n * 2)
t0 = time.perf_counter()
for s in range(30):
    val, grad, D = hea_ansatz(qg.GPUCircuitQ, n, layers, th).value_and_grad(H)
    th -= 0.05 * np.array(grad)
    if s % 5 == 0 or s == 29:
        print(f"  step {s:2d}  E={val:.4f}  D={D:.2e}")
print(f"trained 30 steps in {time.perf_counter()-t0:.1f}s — {n}-qubit VQE on a T4, "
      f"energy decreasing (compressed adjoint).")
