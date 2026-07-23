"""
diff_method="adjoint" through the pennylane-qubit device.

1. Gradient of a VQE-style QNode matches default.qubit's parameter-shift.
2. A full training run with diff_method="adjoint" reaches the same energy
   as the same loop under parameter-shift.

Run: py -3.12 tests/test_adjoint_device.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_qubit import QubitDevice

N, LAYERS = 4, 3
fails = 0


def hamiltonian():
    coeffs, ops = [], []
    for i in range(N - 1):
        coeffs.append(-1.0); ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(N):
        coeffs.append(-1.0); ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


H = hamiltonian()


def ansatz(theta):
    t = theta.reshape(LAYERS, N, 2)
    for L in range(LAYERS):
        for q in range(N):
            qml.RY(t[L, q, 0], wires=q)
            qml.RZ(t[L, q, 1], wires=q)
        for q in range(N):
            qml.CNOT([q, (q + 1) % N])


def cost_factory(dev, diff):
    @qml.qnode(dev, diff_method=diff)
    def cost(theta):
        ansatz(theta)
        return qml.expval(H)
    return cost


# --- 1. gradient match ---
rng = np.random.default_rng(11)
theta0 = pnp.array(rng.uniform(-np.pi, np.pi, LAYERS * N * 2), requires_grad=True)

dev_q = QubitDevice(wires=N)
dev_d = qml.device("default.qubit", wires=N)
g_adj = qml.grad(cost_factory(dev_q, "adjoint"))(theta0)
g_ref = qml.grad(cost_factory(dev_d, "parameter-shift"))(theta0)
gerr = np.max(np.abs(np.array(g_adj) - np.array(g_ref)))
ok = gerr < 1e-6
fails += 0 if ok else 1
print(f"[{'PASS' if ok else 'FAIL'}] adjoint grad vs default.qubit param-shift   max|err|={gerr:.2e}")


# --- 2. training equivalence ---
theta_init = pnp.array(np.random.default_rng(99).uniform(-0.1, 0.1, LAYERS * N * 2),
                       requires_grad=True)


def train(dev, diff, steps=60, lr=0.1):
    cost = cost_factory(dev, diff)
    theta = pnp.array(theta_init, requires_grad=True)   # same start for both
    opt = qml.GradientDescentOptimizer(lr)
    for _ in range(steps):
        theta = opt.step(cost, theta)
    return float(cost(theta))


e_adj = train(QubitDevice(wires=N), "adjoint")
e_ref = train(QubitDevice(wires=N), "parameter-shift")
terr = abs(e_adj - e_ref)
ok2 = terr < 1e-4
fails += 0 if ok2 else 1
print(f"[{'PASS' if ok2 else 'FAIL'}] VQE trained: adjoint={e_adj:.5f} pshift={e_ref:.5f}  gap={terr:.1e}")

print("\n" + ("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
sys.exit(1 if fails else 0)
