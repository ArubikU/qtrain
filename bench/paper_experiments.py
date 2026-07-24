"""
Reproducer for the paper-2 revision experiments (numbers cited in the paper):
  1. 5-seed variance for compressed-vs-exact VQE convergence (TFIM, n=6).
  2. A non-Ising Hamiltonian: MaxCut cost, QAOA-style ansatz (n=14).
  3. Annealed-precision schedule, 5 seeds (TFIM, n=6).

Run: py -3.12 bench/paper_experiments.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn
from vqe_helpers import hea_ansatz, tfim

X, Y, Z = 1, 2, 3
S2 = 1.0 / math.sqrt(2.0)


def train_tfim(n, layers, levels, steps, theta0, lr=0.1):
    th = theta0.copy(); H = tfim(n)
    for _ in range(steps):
        _, g, _ = hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, levels)
        th -= lr * np.array(g)
    return hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, 0)[0]


# ---------- 1. variance across seeds ----------
print("=== 1. VQE convergence variance, 5 seeds (n=6, 3 layers, TFIM) ===")
n, layers, steps = 6, 3, 120
g256, g32 = [], []
for seed in range(5):
    th0 = np.random.default_rng(seed).uniform(-0.1, 0.1, layers * n * 2)
    e_ex = train_tfim(n, layers, 0, steps, th0)
    g256.append(abs(train_tfim(n, layers, 256, steps, th0) - e_ex))
    g32.append(abs(train_tfim(n, layers, 32, steps, th0) - e_ex))
print(f"levels=256 gap: mean {np.mean(g256):.2e}  std {np.std(g256):.2e}")
print(f"levels=32  gap: mean {np.mean(g32):.2e}  std {np.std(g32):.2e}")


# ---------- 2. MaxCut n=14 (brute-force reference) ----------
print("\n=== 2. MaxCut, n=14, QAOA-style ansatz ===")
n = 14
rng = np.random.default_rng(7)
edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.25]
E = len(edges)
best = 0
for b in range(1 << n):
    cut = sum(1 for (i, j) in edges if ((b >> i) ^ (b >> j)) & 1)
    best = max(best, cut)
Hmax = [(0.5, [(i, Z), (j, Z)]) for (i, j) in edges] + [(-0.5 * E, [])]
p = 3
P = p * (E + n)


def qaoa(theta):
    c = qn.ACircuit(n); s = 0
    for q in range(n):
        c.fixed(q, S2, S2, S2, -S2)
    for _ in range(p):
        for (i, j) in edges:
            c.cfixed([i], j, 0, 1, 1, 0)
            c.rot(Z, j, float(theta[s]), True, s); s += 1
            c.cfixed([i], j, 0, 1, 1, 0)
        for q in range(n):
            c.rot(X, q, float(theta[s]), True, s); s += 1
    return c


def train_qaoa(levels, th0, steps=150, lr=0.04):
    th = th0.copy()
    for _ in range(steps):
        _, g, _ = qaoa(th).value_and_grad_q(Hmax, levels)
        th -= lr * np.array(g)
    return qaoa(th).value_and_grad_q(Hmax, 0)[0]


th0 = rng.uniform(-0.1, 0.1, P)
e_ex = train_qaoa(0, th0)
e256 = train_qaoa(256, th0)
print(f"graph: {n} nodes, {E} edges, {P} params | brute-force max cut = {best}")
print(f"exact-grad : <H>={e_ex:.4f}  ratio {(-e_ex)/best:.3f}")
print(f"levels=256 : <H>={e256:.4f}  gap vs exact {abs(e256-e_ex):.2e}")


# ---------- 3. annealed precision, 5 seeds ----------
print("\n=== 3. annealed precision, 5 seeds (n=6, 3 layers, TFIM, 120 steps) ===")
n, layers, steps = 6, 3, 120
H = tfim(n)


def train_sched(schedule, th0, lr=0.1):
    th = th0.copy()
    for s in range(steps):
        _, g, _ = hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, schedule(s))
        th -= lr * np.array(g)
    return hea_ansatz(qn.ACircuit, n, layers, th).value_and_grad_q(H, 0)[0]


tight, coarse, ann = [], [], []
for seed in range(5):
    th0 = np.random.default_rng(seed).uniform(-0.1, 0.1, layers * n * 2)
    tight.append(train_sched(lambda s: 256, th0))
    coarse.append(train_sched(lambda s: 16, th0))
    ann.append(train_sched(lambda s: 16 if s < steps // 2 else 256, th0))
for name, v in [("constant-256", tight), ("constant-16", coarse), ("annealed 16->256", ann)]:
    print(f"{name:18s}: E = {np.mean(v):.4f} +/- {np.std(v):.4f}")
