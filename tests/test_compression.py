"""
Phase 3 gate: gradients through compression behave as the theory predicts.

  a) levels=0 reproduces the exact adjoint gradient bit-for-bit.
  b) worst-case bound holds: max|grad err| <= D across a budget sweep.
  c) error is ~linear in D (err/D stays within a small band).
  d) training at a fine budget (levels=256) matches exact-gradient training.

Run: py -3.12 tests/test_compression.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qubit_native as qn

X, Y, Z = 1, 2, 3
fails = 0


def rep(ok, msg):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")


def build(n, layers, theta):
    c = qn.ACircuit(n); p = 0
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


n, layers = 8, 3
rng = np.random.default_rng(7)
theta = rng.uniform(-math.pi, math.pi, layers * n * 2)
H = tfim(n)
c = build(n, layers, theta)

v0, g0, D0 = c.value_and_grad_q(H, 0)
_, gref = c.value_and_grad(H)
g0, gref = np.array(g0), np.array(gref)

# a) levels=0 == exact adjoint
rep(np.max(np.abs(g0 - gref)) < 1e-14 and D0 == 0.0, "levels=0 reproduces exact adjoint gradient")

# b) + c) bound and linearity
ratios = []
bound_ok = True
for levels in [4, 8, 16, 32, 64, 256, 1024]:
    _, gq, D = c.value_and_grad_q(H, levels)
    err = np.max(np.abs(np.array(gq) - g0))
    if err > D:
        bound_ok = False
    ratios.append(err / D)
rep(bound_ok, "worst-case bound max|grad err| <= D holds across sweep")
rep(max(ratios) / min(ratios) < 20, f"error ~linear in D (err/D band {min(ratios):.3f}..{max(ratios):.3f})")

# d) training match at fine budget
nv, lv = 6, 3
Hv = tfim(nv)
t0 = rng.uniform(-0.1, 0.1, lv * nv * 2)


def train(levels, steps=100, lr=0.1):
    th = t0.copy()
    for _ in range(steps):
        _, grad, _ = build(nv, lv, th).value_and_grad_q(Hv, levels)
        th -= lr * np.array(grad)
    return build(nv, lv, th).value_and_grad_q(Hv, 0)[0]


e_ex = train(0)
e_c = train(256)
rep(abs(e_c - e_ex) < 1e-2, f"levels=256 training matches exact-grad (E={e_c:.4f} vs {e_ex:.4f})")

print("\n" + ("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
sys.exit(1 if fails else 0)
