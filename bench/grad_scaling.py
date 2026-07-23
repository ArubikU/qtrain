"""
Gradient-time scaling: adjoint vs parameter-shift.

Adjoint computes all P gradients in one forward + one backward pass;
parameter-shift needs 2P circuit evaluations. So adjoint's advantage grows
with the parameter count. This times both on the qubit.simulator device
(and default.qubit as a reference) across circuit sizes.

Run: py -3.12 bench/grad_scaling.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_qubit import QubitDevice


def tfim(n):
    coeffs, ops = [], []
    for i in range(n - 1):
        coeffs.append(-1.0); ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n):
        coeffs.append(-1.0); ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def time_grad(dev, diff, n, layers, reps=3):
    H = tfim(n)

    def ansatz(theta):
        t = theta.reshape(layers, n, 2)
        for L in range(layers):
            for q in range(n):
                qml.RY(t[L, q, 0], wires=q); qml.RZ(t[L, q, 1], wires=q)
            for q in range(n):
                qml.CNOT([q, (q + 1) % n])

    @qml.qnode(dev, diff_method=diff)
    def cost(theta):
        ansatz(theta)
        return qml.expval(H)

    theta = pnp.array(np.random.default_rng(0).uniform(-np.pi, np.pi, layers * n * 2),
                      requires_grad=True)
    g = qml.grad(cost)
    g(theta)                                   # warmup / trace
    t0 = time.perf_counter()
    for _ in range(reps):
        g(theta)
    return (time.perf_counter() - t0) / reps


def maybe_lightning(n):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return None


print(f"{'n':>3} {'L':>3} {'P':>5}  {'qubit-adj(ms)':>14} {'qubit-pshift(ms)':>16} "
      f"{'light-adj(ms)':>14} {'vs pshift':>10} {'vs light':>9}")
for n, layers in [(4, 2), (6, 3), (8, 3), (10, 4), (12, 4), (14, 4)]:
    P = layers * n * 2
    ta = time_grad(QubitDevice(wires=n), "adjoint", n, layers) * 1e3
    tp = time_grad(QubitDevice(wires=n), "parameter-shift", n, layers) * 1e3
    ld = maybe_lightning(n)
    tl = time_grad(ld, "adjoint", n, layers) * 1e3 if ld is not None else float("nan")
    vs_l = f"{tl/ta:.2f}x" if tl == tl else "n/a"    # >1 => we are faster
    print(f"{n:>3} {layers:>3} {P:>5}  {ta:>14.1f} {tp:>16.1f} {tl:>14.1f} "
          f"{tp/ta:>9.1f}x {vs_l:>9}")

print("\nqubit-adj: 2 passes regardless of P; parameter-shift: 2P evaluations.")
print("'vs light' > 1 means qubit adjoint is faster than lightning adjoint;")
print("< 1 means lightning is faster (expected — its dense kernels are tuned).")
