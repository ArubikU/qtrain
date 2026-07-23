"""
End-to-end VQE training THROUGH the pennylane-qubit device.

Point of Phase 1: prove the device is a real PennyLane citizen — a standard
optimizer differentiates a QNode running on the qubit engine and drives it to
the ground state. Same loop on default.qubit is the reference.

TFIM (transverse-field Ising) on 4 qubits, H = -sum Z_i Z_{i+1} - sum X_i.
Run: py -3.12 examples/vqe_train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_qubit import QubitDevice

N, LAYERS, STEPS, LR = 4, 3, 60, 0.1


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


def exact_ground():
    mat = qml.matrix(H, wire_order=range(N))
    return float(np.min(np.linalg.eigvalsh(mat)))


def train(device_name):
    dev = QubitDevice(wires=N) if device_name == "qubit" else qml.device("default.qubit", wires=N)

    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(theta):
        ansatz(theta)
        return qml.expval(H)

    rng = np.random.default_rng(7)
    theta = pnp.array(rng.uniform(-0.1, 0.1, LAYERS * N * 2), requires_grad=True)
    opt = qml.GradientDescentOptimizer(LR)
    for _ in range(STEPS):
        theta, _ = opt.step_and_cost(cost, theta)
    return float(cost(theta))


if __name__ == "__main__":
    e_exact = exact_ground()
    e_qubit = train("qubit")
    e_ref = train("default")
    print(f"exact ground energy      : {e_exact:.5f}")
    print(f"VQE on qubit.simulator   : {e_qubit:.5f}  (gap {abs(e_qubit - e_exact):.2e})")
    print(f"VQE on default.qubit     : {e_ref:.5f}  (gap {abs(e_ref - e_exact):.2e})")
    ok = abs(e_qubit - e_ref) < 1e-3
    print("\n" + ("PASS: qubit device trains identically to default.qubit."
                  if ok else "FAIL: training diverged from reference."))
    sys.exit(0 if ok else 1)
