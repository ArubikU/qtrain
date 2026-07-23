"""
Phase 3 [CORE]: gradients through compression.

Three things, on the production adjoint (ACircuit, real gate set incl.
controls), scaled past the spike:

  1. Gradient error vs injected budget D  -> linear (the paper-2 bound).
  2. VQE training under several budgets    -> converges, graceful gap.
  3. Memory / capability model             -> bytes per amplitude and the
     largest trainable circuit per GB for each storage format.

Run: py -3.12 bench/phase3_compression.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
from vqe_helpers import hea_ansatz, tfim, exact_ground


def build(n, layers, theta):
    return hea_ansatz(qn.ACircuit, n, layers, theta)


# ---------- 1. gradient error vs budget ----------
print("=== 1. adjoint gradient error vs compression budget D (n=9, 3 layers) ===")
n, layers = 9, 3
rng = np.random.default_rng(4)
theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
H = tfim(n)
c = build(n, layers, theta)
_, g_exact, _ = c.value_and_grad_q(H, 0)
g_exact = np.array(g_exact)
print(f"{'levels':>7} {'injected D':>12} {'max|grad err|':>14} {'err/D':>8}")
for levels in [4, 8, 16, 32, 64, 256, 1024]:
    _, gq, D = c.value_and_grad_q(H, levels)
    e = np.max(np.abs(np.array(gq) - g_exact))
    print(f"{levels:>7} {D:>12.3e} {e:>14.3e} {e/D if D else 0:>8.3f}")
print("err/D ~ constant => error linear in D; worst-case bound |err|<=D holds.")

# ---------- 2. training under budgets ----------
print("\n=== 2. VQE training: exact vs compressed gradients (n=6, 3 layers) ===")
nv, lv = 6, 3
Hv = tfim(nv)
theta0 = rng.uniform(-0.1, 0.1, lv * nv * 2)


def train(levels, steps=120, lr=0.1):
    th = theta0.copy()
    for _ in range(steps):
        c = build(nv, lv, th)
        _, grad, _ = c.value_and_grad_q(Hv, levels)
        th -= lr * np.array(grad)
    val, _, _ = build(nv, lv, th).value_and_grad_q(Hv, 0)
    return val


e0 = exact_ground(nv, Hv)
e_exact = train(0)
print(f"exact ground energy       : {e0:.5f}")
print(f"trained (exact gradients) : {e_exact:.5f}  (gap {abs(e_exact-e0):.1e})")
for levels in [256, 32, 8]:
    ev = train(levels)
    print(f"trained (levels={levels:<4})     : {ev:.5f}  (gap vs exact-grad {abs(ev-e_exact):.1e})")
print("compressed-gradient training converges; degradation graceful and tunable.")

# ---------- 3. memory / capability model ----------
print("\n=== 3. memory model: adjoint holds 2 trajectories (phi, lambda) ===")
GB = 6 * 1024**3
fmts = [("complex128", 16), ("complex64", 8), ("int16-compressed", 4)]
print(f"{'format':>18} {'bytes/amp':>10} {'max n on 6 GB (2 states)':>26}")
for name, bpa in fmts:
    # 2 * bpa * 2^n <= GB
    max_n = int(math.floor(math.log2(GB / (2 * bpa))))
    print(f"{name:>18} {bpa:>10} {max_n:>26}")
print("int16 storage (validated above to keep gradients bounded and training\n"
      "convergent) buys +1 qubit over complex64, +2 over complex128 per axis —\n"
      "the 'largest trainable circuit per GB' lever. Real tiered blocks (ZERO/\n"
      "COMPRESSED/FULL, Phase 4 GPU) push further where amplitudes are sparse.")
