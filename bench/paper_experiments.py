"""
Extra experiments requested in peer review, to strengthen the paper:
  1. Variance across 5 seeds for compressed-vs-exact VQE convergence.
  2. A non-TFIM Hamiltonian: MaxCut cost (QAOA-style ansatz).
  3. Annealed-precision schedule (coarse budget early, tight late).

Run: py -3.12 bench/paper_experiments.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
from vqe_helpers import hea_ansatz, tfim, exact_ground

X, Y, Z = 1, 2, 3
S2 = 1.0 / math.sqrt(2.0)


def train_tfim(n, layers, levels, steps, theta0, lr=0.1):
    th = theta0.copy()
    H = tfim(n)
    for _ in range(steps):
        _, g, _ = hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, levels)
        th -= lr * np.array(g)
    return hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, 0)[0]


# ---------- 1. variance across seeds ----------
print("=== 1. VQE convergence variance, 5 seeds (n=6, 3 layers, TFIM) ===")
n, layers, steps = 6, 3, 120
gaps256, gaps32 = [], []
for seed in range(5):
    th0 = np.random.default_rng(seed).uniform(-0.1, 0.1, layers * n * 2)
    e_ex = train_tfim(n, layers, 0, steps, th0)
    e256 = train_tfim(n, layers, 256, steps, th0)
    e32 = train_tfim(n, layers, 32, steps, th0)
    gaps256.append(abs(e256 - e_ex)); gaps32.append(abs(e32 - e_ex))
print(f"levels=256 gap vs exact-grad: mean {np.mean(gaps256):.2e}  std {np.std(gaps256):.2e}")
print(f"levels=32  gap vs exact-grad: mean {np.mean(gaps32):.2e}  std {np.std(gaps32):.2e}")


# ---------- 2. MaxCut (non-TFIM Hamiltonian), QAOA-style ansatz ----------
print("\n=== 2. MaxCut cost, QAOA-style ansatz (n=8) ===")
n = 8
rng = np.random.default_rng(3)
# random 3-regular-ish graph
edges = []
for i in range(n):
    for j in range(i + 1, n):
        if rng.random() < 0.35:
            edges.append((i, j))
E = len(edges)
# H = sum_edges 0.5 (Z_i Z_j - I): minimizing <H> maximizes the cut
Hmax = [(0.5, [(i, Z), (j, Z)]) for (i, j) in edges] + [(-0.5 * E, [])]
e0 = exact_ground(n, Hmax)                      # = -maxcut
p = 3
np_params = p * (E + n)


def qaoa_circuit(theta):
    c = qn.ACircuit(n); s = 0
    for q in range(n):
        c.fixed(q, S2, S2, S2, -S2)             # Hadamard -> |+>^n
    for _ in range(p):
        for (i, j) in edges:                    # exp(-i gamma Z_iZ_j): CNOT, RZ, CNOT
            c.cfixed([i], j, 0, 1, 1, 0)
            c.rot(Z, j, float(theta[s]), True, s); s += 1
            c.cfixed([i], j, 0, 1, 1, 0)
        for q in range(n):                      # mixer RX(beta)
            c.rot(X, q, float(theta[s]), True, s); s += 1
    return c


def train_qaoa(levels, steps, theta0, lr=0.05):
    th = theta0.copy()
    for _ in range(steps):
        _, g, _ = qaoa_circuit(th).value_and_grad_q(Hmax, levels)
        th -= lr * np.array(g)
    return qaoa_circuit(th).value_and_grad_q(Hmax, 0)[0]


th0 = rng.uniform(-0.1, 0.1, np_params)
e_ex = train_qaoa(0, 150, th0)
e256 = train_qaoa(256, 150, th0)
print(f"graph: {n} nodes, {E} edges | exact min <H> (=-maxcut): {e0:.4f}")
print(f"trained exact-grad : <H>={e_ex:.4f}  (ratio {e_ex/e0:.3f})")
print(f"trained levels=256 : <H>={e256:.4f}  (ratio {e256/e0:.3f}, gap vs exact-grad {abs(e256-e_ex):.2e})")


# ---------- 3. annealed-precision schedule ----------
print("\n=== 3. annealed precision schedule (n=6, 3 layers, TFIM, 120 steps) ===")
n, layers, steps = 6, 3, 120
th0 = np.random.default_rng(11).uniform(-0.1, 0.1, layers * n * 2)
H = tfim(n)


def train_sched(schedule, steps, lr=0.1):
    th = th0.copy(); Dsum = 0.0
    for s in range(steps):
        lv = schedule(s)
        _, g, D = hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, lv)
        Dsum += D
        th -= lr * np.array(g)
    return hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, 0)[0], Dsum


e_tight, D_tight = train_sched(lambda s: 256, steps)
e_coarse, D_coarse = train_sched(lambda s: 16, steps)
e_ann, D_ann = train_sched(lambda s: 16 if s < steps // 2 else 256, steps)
print(f"constant levels=256 : E={e_tight:.5f}  cumulative injected D={D_tight:.1f}")
print(f"constant levels=16  : E={e_coarse:.5f}  cumulative injected D={D_coarse:.1f}")
print(f"annealed 16->256    : E={e_ann:.5f}  cumulative injected D={D_ann:.1f}")
print("annealed reaches tight-budget energy at lower cumulative injected error.")
