"""
pennylane-qubit: a PennyLane device backed by the qubit state-vector engine.

Phase 1 scope:
  - analytic (shots=None): expval, state, probs.
  - finite-shot: expval, probs, sample, counts (computational-basis samples
    from the engine, mapped through PennyLane's process_samples).

This is what variational training (VQE / QML) needs; the device is the entry
point through which qtrain's error-bounded, compressed gradients will later
be exposed (adjoint diff lands in a subsequent phase).

Register name: "qubit.simulator".
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

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


_S2 = 1.0 / np.sqrt(2.0)
_TPH = np.exp(1j * np.pi / 4)
# fixed-gate 2x2 matrices (row-major) for the native adjoint ACircuit
_FIXED = {
    "Hadamard": (_S2, _S2, _S2, -_S2),
    "PauliX":   (0, 1, 1, 0),
    "PauliY":   (0, -1j, 1j, 0),
    "PauliZ":   (1, 0, 0, -1),
    "S":        (1, 0, 0, 1j),
    "T":        (1, 0, 0, _TPH),
}
_ROT_GEN = {"RX": 1, "RY": 2, "RZ": 3, "PhaseShift": 3}  # PhaseShift ~ RZ for expval
_PAULI_CODE = {"X": 1, "Y": 2, "Z": 3}


def _pauli_rotation(word):
    """Ops that rotate a Pauli word into the Z eigenbasis, plus its Z wires.
       X: H   Y: S-dagger, H   Z: nothing."""
    extra, wires = [], []
    for wire, letter in word.items():
        if letter == "X":
            extra.append(qml.Hadamard(wire))
        elif letter == "Y":
            extra.append(qml.PhaseShift(-np.pi / 2, wire))  # S-dagger
            extra.append(qml.Hadamard(wire))
        wires.append(wire)
    return extra, wires


class QubitDevice(Device):
    """PennyLane device wrapping the qubit CPU engine."""

    name = "qubit.simulator"

    def __init__(self, wires=None, shots=None, fidelity: float = 1.0, seed: int = 0xC0FFEE):
        super().__init__(wires=wires, shots=shots)
        self._fidelity = float(fidelity)
        self._seed = int(seed)

    # --- preprocessing: decompose to the supported gate set ---
    def preprocess_transforms(self, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        program = TransformProgram()
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=_supported,
            name=self.name,
        )
        return program

    # --- execution ---
    def execute(self, circuits, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        single = isinstance(circuits, QuantumScript)
        tapes = [circuits] if single else list(circuits)
        results = tuple(self._execute_tape(t) for t in tapes)
        return results[0] if single else results

    # --- device-provided derivatives (adjoint) ---
    def setup_execution_config(self, config=None, circuit=None):
        from dataclasses import replace
        config = config or ExecutionConfig()
        upd = {}
        if config.gradient_method in ("best", "device"):
            upd["gradient_method"] = "adjoint"
        gm = upd.get("gradient_method", config.gradient_method)
        if config.use_device_gradient is None:
            upd["use_device_gradient"] = gm == "adjoint"
        if config.grad_on_execution is None:
            upd["grad_on_execution"] = gm == "adjoint"
        return replace(config, **upd)

    def supports_derivatives(self, execution_config=None, circuit=None):
        if execution_config is None:
            return True
        return execution_config.gradient_method in (None, "adjoint", "device", "best")

    def compute_derivatives(self, circuits, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        single = isinstance(circuits, QuantumScript)
        tapes = [circuits] if single else list(circuits)
        res = tuple(self._jacobian(t) for t in tapes)
        return res[0] if single else res

    def execute_and_compute_derivatives(self, circuits, execution_config: ExecutionConfig = _DEFAULT_CONFIG):
        return (self.execute(circuits, execution_config),
                self.compute_derivatives(circuits, execution_config))

    # ---- circuit / run plumbing ----
    def _wire_order(self, tape):
        wires = list(self.wires) if self.wires is not None else list(tape.wires)
        return wires, {w: i for i, w in enumerate(wires)}, len(wires)

    def _build_circuit(self, tape, wmap, n, extra_ops=()):
        c = qn.Circuit(n)
        for op in list(tape.operations) + list(extra_ops):
            w = [wmap[x] for x in op.wires]
            _GATES[op.name](c, w, op.data)
        return c

    def _run(self, circuit, shots=None):
        opts = qn.RunOptions()
        opts.device = qn.Device.Auto
        opts.fidelity = self._fidelity
        opts.seed = self._seed
        if shots:
            opts.shots = int(shots)
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

    def _comp_samples(self, tape, wmap, n, extra_ops, shots):
        """(shots, n) array of 0/1 computational-basis samples, column k = line k."""
        r = self._run(self._build_circuit(tape, wmap, n, extra_ops), shots=shots)
        out = np.empty((shots, n), dtype=np.int64)
        row = 0
        for key, cnt in r.counts.items():
            # engine key char[i] = qubit line (n-1-i); column k = line k = char[n-1-k]
            bits = [int(key[n - 1 - k]) for k in range(n)]
            for _ in range(cnt):
                out[row] = bits
                row += 1
        return out[:row]

    # ---- dispatch ----
    def _tape_shots(self, tape):
        s = tape.shots
        return int(s.total_shots) if (s and s.total_shots) else None

    def _execute_tape(self, tape):
        wires, wmap, n = self._wire_order(tape)
        shots = self._tape_shots(tape)
        out = [self._measure(tape, wires, wmap, n, m, shots) for m in tape.measurements]
        return out[0] if len(out) == 1 else tuple(out)

    def _measure(self, tape, wires, wmap, n, m, shots):
        mp = type(m).__name__
        if shots is None:
            if mp == "ExpectationMP":
                return self._expval_analytic(tape, wmap, n, m.obs)
            if mp == "StateMP":
                return self._statevector(tape, wmap, n)
            if mp == "ProbabilityMP":
                return self._probs_analytic(tape, wmap, n, m.wires)
            raise qml.DeviceError(f"qubit.simulator: analytic {mp} not supported.")
        # ---- finite-shot ----
        if mp == "ExpectationMP":
            return self._expval_shots(tape, wmap, n, m.obs, shots)
        if mp in ("SampleMP", "CountsMP", "ProbabilityMP"):
            obs = getattr(m, "obs", None)
            if obs is not None:
                ps = qml.pauli.pauli_sentence(obs)
                if len(ps) != 1:
                    raise qml.DeviceError(f"qubit.simulator: shot {mp} needs a single Pauli observable.")
                word = next(iter(ps))
                extra, _ = _pauli_rotation(word)
            else:
                extra = ()
            samples = self._comp_samples(tape, wmap, n, extra, shots)
            return m.process_samples(samples, wire_order=Wires(wires))
        raise qml.DeviceError(f"qubit.simulator: shot {mp} not supported.")

    # ---- analytic implementations ----
    def _probs_analytic(self, tape, wmap, n, wires):
        vec = self._statevector(tape, wmap, n)
        full = np.abs(vec) ** 2
        if wires is None or len(wires) == 0 or len(wires) == n:
            return full
        keep = [wmap[w] for w in wires]
        marg = np.zeros(1 << len(keep))
        for i in range(1 << n):
            idx = 0
            for j, k in enumerate(keep):
                bit = (i >> (n - 1 - k)) & 1
                idx |= bit << (len(keep) - 1 - j)
            marg[idx] += full[i]
        return marg

    def _expval_analytic(self, tape, wmap, n, obs):
        ps = qml.pauli.pauli_sentence(obs)
        total = 0.0
        for word, coeff in ps.items():
            if len(word) == 0:
                total += float(np.real(coeff))
                continue
            extra, wlist = _pauli_rotation(word)
            r = self._run(self._build_circuit(tape, wmap, n, extra))
            total += float(np.real(coeff)) * r.expectation_z([wmap[w] for w in wlist])
        return np.array(total)

    # ---- shot implementations ----
    def _expval_shots(self, tape, wmap, n, obs, shots):
        """Estimate <sum c_i P_i> from computational-basis samples, per Pauli term."""
        ps = qml.pauli.pauli_sentence(obs)
        total = 0.0
        for word, coeff in ps.items():
            c = float(np.real(coeff))
            if len(word) == 0:
                total += c
                continue
            extra, wlist = _pauli_rotation(word)
            samples = self._comp_samples(tape, wmap, n, extra, shots)
            cols = [wmap[w] for w in wlist]
            # eigenvalue per shot = product over the word's wires of (+1 for bit0, -1 for bit1)
            eig = np.prod(1 - 2 * samples[:, cols], axis=1)
            total += c * float(np.mean(eig))
        return np.array(total)

    # ---- adjoint jacobian ----
    def _ham_terms(self, obs, wmap):
        ps = qml.pauli.pauli_sentence(obs)
        terms = []
        for word, coeff in ps.items():
            ops = [(wmap[w], _PAULI_CODE[letter]) for w, letter in word.items()]
            terms.append((float(np.real(coeff)), ops))
        return terms

    def _build_acircuit(self, tape, wmap, n):
        trainable = set(tape.trainable_params)
        slot = {gp: i for i, gp in enumerate(sorted(trainable))}
        ac = qn.ACircuit(n)
        pc = 0                                   # running parametric-op index
        for op in tape.operations:
            name, w = op.name, [wmap[x] for x in op.wires]
            if name in _ROT_GEN:
                tr = pc in trainable
                ac.rot(_ROT_GEN[name], w[0], float(op.data[0]), tr, slot.get(pc, -1))
                pc += 1
            elif name in _FIXED:
                ac.fixed(w[0], *_FIXED[name])
            elif name == "CNOT":
                ac.cfixed([w[0]], w[1], 0, 1, 1, 0)
            elif name == "CZ":
                ac.cfixed([w[0]], w[1], 1, 0, 0, -1)
            elif name == "Toffoli":
                ac.cfixed([w[0], w[1]], w[2], 0, 1, 1, 0)
            elif name == "SWAP":
                ac.cfixed([w[0]], w[1], 0, 1, 1, 0)
                ac.cfixed([w[1]], w[0], 0, 1, 1, 0)
                ac.cfixed([w[0]], w[1], 0, 1, 1, 0)
            elif name == "Identity":
                pass
            else:
                raise qml.DeviceError(f"qubit.simulator adjoint: unsupported op {name}.")
        return ac

    def _jacobian(self, tape):
        if len(tape.measurements) != 1 or type(tape.measurements[0]).__name__ != "ExpectationMP":
            raise qml.DeviceError("qubit.simulator adjoint supports a single expval measurement.")
        if self._tape_shots(tape) is not None:
            raise qml.DeviceError("qubit.simulator adjoint is analytic (use shots=None).")
        _, wmap, n = self._wire_order(tape)
        ac = self._build_acircuit(tape, wmap, n)
        ham = self._ham_terms(tape.measurements[0].obs, wmap)
        _, grad = ac.value_and_grad(ham)
        # PennyLane's Jacobian layout for one expectation:
        #   1 param  -> scalar;  >1 params -> tuple of per-param scalars.
        if len(grad) == 1:
            return np.array(grad[0])
        return tuple(np.array(g) for g in grad)
