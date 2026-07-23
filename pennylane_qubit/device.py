"""
pennylane-qubit: a PennyLane device backed by the qubit state-vector engine.

Phase 1 scope: analytic (shots=None) execution of expectation values, state,
and probabilities. This is what variational training (VQE / QML) needs; the
device is the entry point through which qtrain's error-bounded, compressed
gradients will later be exposed (adjoint diff lands in a subsequent phase).

Register name: "qubit.simulator".
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript

import qubit_native as qn

_DEFAULT_CONFIG = ExecutionConfig()

# PennyLane op name -> function building the gate on a qubit_native.Circuit.
# `w` is the mapped integer line, `p` the op parameters.
_GATES = {
    "Identity":   lambda c, w, p: None,
    "Hadamard":   lambda c, w, p: c.h(w[0]),
    "PauliX":     lambda c, w, p: c.x(w[0]),
    "PauliY":     lambda c, w, p: c.y(w[0]),
    "PauliZ":     lambda c, w, p: c.z(w[0]),
    "S":          lambda c, w, p: c.s(w[0]),
    "T":          lambda c, w, p: c.t(w[0]),
    "RX":         lambda c, w, p: c.rx(w[0], float(p[0])),
    "RY":         lambda c, w, p: c.ry(w[0], float(p[0])),
    "RZ":         lambda c, w, p: c.rz(w[0], float(p[0])),
    "PhaseShift": lambda c, w, p: c.phase(w[0], float(p[0])),
    "CNOT":       lambda c, w, p: c.cnot(w[0], w[1]),
    "CZ":         lambda c, w, p: c.cz(w[0], w[1]),
    "SWAP":       lambda c, w, p: c.swap(w[0], w[1]),
    "Toffoli":    lambda c, w, p: c.toffoli(w[0], w[1], w[2]),
}


def _supported(op) -> bool:
    return op.name in _GATES


class QubitDevice(Device):
    """PennyLane device wrapping the qubit CPU engine."""

    name = "qubit.simulator"

    def __init__(self, wires=None, shots=None, fidelity: float = 1.0, seed: int = 0xC0FFEE):
        if shots is not None:
            raise qml.DeviceError("qubit.simulator is analytic-only in Phase 1 (use shots=None).")
        super().__init__(wires=wires, shots=shots)
        self._fidelity = float(fidelity)
        self._seed = int(seed)

    # --- preprocessing: decompose to the supported gate set ---
    def preprocess(self, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        program = TransformProgram()
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=_supported,
            name=self.name,
        )
        return program, execution_config

    # --- execution ---
    def execute(self, circuits, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        single = isinstance(circuits, QuantumScript)
        tapes = [circuits] if single else list(circuits)
        results = tuple(self._execute_tape(t) for t in tapes)
        return results[0] if single else results

    # ---- internals ----
    def _wire_order(self, tape):
        wires = self.wires if self.wires is not None else tape.wires
        return {w: i for i, w in enumerate(wires)}, len(wires)

    def _build_circuit(self, tape, wmap, n, extra_ops=()):
        c = qn.Circuit(n)
        for op in list(tape.operations) + list(extra_ops):
            w = [wmap[x] for x in op.wires]
            _GATES[op.name](c, w, op.data)
        return c

    def _run(self, circuit):
        opts = qn.RunOptions()
        opts.device = qn.Device.Auto
        opts.fidelity = self._fidelity
        opts.seed = self._seed
        return qn.run(circuit, opts)

    def _statevector(self, tape, wmap, n):
        r = self._run(self._build_circuit(tape, wmap, n))
        dim = 1 << n
        vec = np.empty(dim, dtype=np.complex128)
        for i in range(dim):
            # PennyLane index: wire 0 is the most-significant bit.
            # qubit index: line k is bit k (LSB). Reverse the n-bit order.
            qi = 0
            for k in range(n):
                if (i >> (n - 1 - k)) & 1:
                    qi |= 1 << k
            vec[i] = r.amplitude(qi)
        return vec

    def _execute_tape(self, tape):
        wmap, n = self._wire_order(tape)
        out = []
        for m in tape.measurements:
            mp = type(m).__name__
            if mp == "ExpectationMP":
                out.append(self._expval(tape, wmap, n, m.obs))
            elif mp == "StateMP":
                out.append(self._statevector(tape, wmap, n))
            elif mp == "ProbabilityMP":
                out.append(self._probs(tape, wmap, n, m.wires))
            else:
                raise qml.DeviceError(f"qubit.simulator does not support {mp} yet.")
        if len(out) == 1:
            return out[0]
        return tuple(out)

    def _probs(self, tape, wmap, n, wires):
        vec = self._statevector(tape, wmap, n)
        full = np.abs(vec) ** 2
        if wires is None or len(wires) == 0 or len(wires) == n:
            return full
        # marginalize onto the requested wires (PennyLane MSB-first order)
        keep = [wmap[w] for w in wires]
        marg = np.zeros(1 << len(keep))
        for i in range(1 << n):
            idx = 0
            for j, k in enumerate(keep):
                bit = (i >> (n - 1 - k)) & 1
                idx |= bit << (len(keep) - 1 - j)
            marg[idx] += full[i]
        return marg

    def _expval(self, tape, wmap, n, obs):
        # decompose observable into a Pauli sentence: {PauliWord: coeff}
        ps = qml.pauli.pauli_sentence(obs)
        total = 0.0
        for word, coeff in ps.items():
            if len(word) == 0:            # identity term
                total += float(np.real(coeff))
                continue
            # Rotate each Pauli letter into the Z eigenbasis, then measure Z.
            #   X:  H            (H Z H = X)
            #   Y:  S-dagger, H  (standard Y-basis measurement)
            #   Z:  nothing
            extra, zwires = [], []
            for wire, letter in word.items():
                if letter == "X":
                    extra.append(qml.Hadamard(wire))
                elif letter == "Y":
                    extra.append(qml.PhaseShift(-np.pi / 2, wire))  # S-dagger
                    extra.append(qml.Hadamard(wire))
                zwires.append(wmap[wire])
            r = self._run(self._build_circuit(tape, wmap, n, extra))
            total += float(np.real(coeff)) * r.expectation_z(zwires)
        return np.array(total)
