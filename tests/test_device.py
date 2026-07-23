"""
Correctness of the pennylane-qubit device vs PennyLane's default.qubit.

Analytic-only (shots=None). Covers expval (Z/X/Y and tensor products),
a Hamiltonian, state, and probs. Run: py -3.12 tests/test_device.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pennylane as qml
from pennylane_qubit import QubitDevice

TOL = 1e-5
fails = 0


def check(name, got, ref):
    global fails
    got, ref = np.asarray(got), np.asarray(ref)
    err = np.max(np.abs(got - ref)) if got.size else 0.0
    ok = err < TOL
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} max|err|={err:.2e}")


def run_both(fn, wires):
    dev_q = QubitDevice(wires=wires)
    dev_d = qml.device("default.qubit", wires=wires)
    return qml.QNode(fn, dev_q)(), qml.QNode(fn, dev_d)()


# 1. expval PauliZ tensor on a Bell-ish state
def c1():
    qml.Hadamard(0); qml.CNOT([0, 1]); qml.RY(0.7, wires=2)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
check("expval Z0Z1 (Bell)", *run_both(c1, 3))


# 2. expval mixed X/Y/Z basis rotation
def c2():
    qml.RX(0.3, 0); qml.RY(1.1, 1); qml.RZ(0.5, 0); qml.CNOT([0, 1])
    return qml.expval(qml.PauliX(0) @ qml.PauliY(1))
check("expval X0Y1", *run_both(c2, 2))


# 3. Hamiltonian expval (multi-term, non-commuting)
def c3():
    qml.RY(0.9, 0); qml.RX(0.4, 1); qml.CNOT([0, 1])
    H = qml.Hamiltonian([0.5, -1.2, 0.3],
                        [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(1)])
    return qml.expval(H)
check("expval Hamiltonian", *run_both(c3, 2))


# 4. state
def c4():
    qml.Hadamard(0); qml.T(0); qml.CNOT([0, 1]); qml.RX(0.6, 2)
    return qml.state()
check("statevector", *run_both(c4, 3))


# 5. probs (subset of wires)
def c5():
    qml.Hadamard(0); qml.CNOT([0, 1]); qml.RY(1.3, 2); qml.CNOT([1, 2])
    return qml.probs(wires=[0, 2])
check("probs [0,2]", *run_both(c5, 3))


# 6. decomposition path: a gate not in the native set (Rot -> RZ RY RZ)
def c6():
    qml.Rot(0.2, 0.5, -0.3, wires=0); qml.CNOT([0, 1])
    return qml.expval(qml.PauliZ(1))
check("expval after Rot decomp", *run_both(c6, 2))


print("\n" + ("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
sys.exit(1 if fails else 0)
