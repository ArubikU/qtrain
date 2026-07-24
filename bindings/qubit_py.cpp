/*
 * qubit_py.cpp — pybind11 bindings over qubit's public C++ API.
 *
 * Exposes the header-only CPU engine (include/qubit/qubit.h) to Python as
 * the `qubit_native` module: Circuit, RunOptions, Result, run(), plus the
 * Device/Precision enums. This is the substrate the pennylane-qubit device
 * plugin drives; it is also usable standalone.
 *
 * CPU-only for Phase 1 (no -DQUBIT_CUDA): the GPU backend needs nvcc and is
 * wired in a later phase. Device::GPU simply falls back to CPU here.
 *
 * Built via setup.py (pybind11.setup_helpers, /std:c++17).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "qubit/qubit.h"
#include "qubit/adjoint.h"

namespace py = pybind11;
using namespace qubit;

PYBIND11_MODULE(qubit_native, m) {
	m.doc() = "Python bindings for the qubit state-vector simulator (CPU engine).";

	py::register_exception<Error>(m, "QubitError");

	py::enum_<Device>(m, "Device")
		.value("Auto", Device::Auto)
		.value("CPU", Device::CPU)
		.value("GPU", Device::GPU);

	py::enum_<Precision>(m, "Precision")
		.value("F32", Precision::F32)
		.value("F64", Precision::F64);

	py::class_<Circuit>(m, "Circuit")
		.def(py::init<int>(), py::arg("num_qubits"))
		.def_property_readonly("num_qubits", &Circuit::num_qubits)
		.def_property_readonly("num_cbits", &Circuit::num_cbits)
		.def("h", &Circuit::h, py::arg("q"))
		.def("x", &Circuit::x, py::arg("q"))
		.def("y", &Circuit::y, py::arg("q"))
		.def("z", &Circuit::z, py::arg("q"))
		.def("s", &Circuit::s, py::arg("q"))
		.def("t", &Circuit::t, py::arg("q"))
		.def("rx", &Circuit::rx, py::arg("q"), py::arg("theta"))
		.def("ry", &Circuit::ry, py::arg("q"), py::arg("theta"))
		.def("rz", &Circuit::rz, py::arg("q"), py::arg("theta"))
		.def("phase", &Circuit::phase, py::arg("q"), py::arg("phi"))
		.def("unitary", &Circuit::unitary,
		     py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("cnot", &Circuit::cnot, py::arg("ctrl"), py::arg("tgt"))
		.def("cx", &Circuit::cnot, py::arg("ctrl"), py::arg("tgt"))
		.def("cz", &Circuit::cz, py::arg("ctrl"), py::arg("tgt"))
		.def("toffoli", &Circuit::toffoli, py::arg("c1"), py::arg("c2"), py::arg("tgt"))
		.def("controlled", &Circuit::controlled,
		     py::arg("ctrl"), py::arg("q"),
		     py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("swap", &Circuit::swap, py::arg("a"), py::arg("b"))
		.def("measure", &Circuit::measure, py::arg("q"))
		.def("reset", &Circuit::reset, py::arg("q"))
		.def("has_measurements", &Circuit::has_measurements);

	py::class_<RunStats>(m, "RunStats")
		.def_readonly("backend", &RunStats::backend)
		.def_readonly("memory_peak_bytes", &RunStats::memory_peak_bytes)
		.def_readonly("time_ms", &RunStats::time_ms)
		.def_readonly("qubits_total", &RunStats::qubits_total)
		.def_readonly("qubits_live", &RunStats::qubits_live);

	py::class_<RunOptions>(m, "RunOptions")
		.def(py::init<>())
		.def_readwrite("device", &RunOptions::device)
		.def_readwrite("fidelity", &RunOptions::fidelity)
		.def_readwrite("shots", &RunOptions::shots)
		.def_readwrite("seed", &RunOptions::seed)
		.def_readwrite("precision", &RunOptions::precision);

	py::class_<Result>(m, "Result")
		.def_property_readonly("counts", &Result::counts)
		.def("prob", &Result::prob, py::arg("idx"))
		.def("amplitude",
		     [](const Result& r, uint64_t idx) {
			     cf a = r.amplitude(idx);
			     return std::complex<double>(a.real(), a.imag());
		     }, py::arg("idx"))
		.def("expectation_z", &Result::expectation_z, py::arg("qubits"))
		.def_readonly("stats", &Result::stats);

	m.def("run", &run, py::arg("circuit"), py::arg("options") = RunOptions(),
	      "Analyze the circuit, pick a backend, execute, return a Result.");

	/* ---- adjoint differentiation (qubit::ACircuit) ---- */
	using qubit::ACircuit;
	using cdd = std::complex<double>;
	py::class_<ACircuit>(m, "ACircuit")
		.def(py::init<int>(), py::arg("num_qubits"))
		.def_property_readonly("num_qubits", &ACircuit::num_qubits)
		.def_property_readonly("num_params", &ACircuit::num_params)
		.def("rot", &ACircuit::rot,
		     py::arg("gen"), py::arg("q"), py::arg("theta"),
		     py::arg("trainable"), py::arg("slot"),
		     "Parametric rotation exp(-i theta G/2); gen 1=X 2=Y 3=Z.")
		.def("fixed", &ACircuit::fixed,
		     py::arg("q"), py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("cfixed", &ACircuit::cfixed,
		     py::arg("ctrl"), py::arg("q"),
		     py::arg("m00"), py::arg("m01"), py::arg("m10"), py::arg("m11"))
		.def("value_and_grad",
		     [](const ACircuit& c,
		        const std::vector<std::pair<double, std::vector<std::pair<int, int>>>>& terms) {
			     qubit::Ham H;
			     for (auto& t : terms) H.push_back({t.first, t.second});
			     return c.value_and_grad(H);
		     },
		     py::arg("hamiltonian"),
		     "Return (<psi|H|psi>, gradient over trainable params) via adjoint.\n"
		     "hamiltonian: list of (coeff, [(wire, pauli)]); pauli 1=X 2=Y 3=Z.")
		.def("value_and_grad_q",
		     [](const ACircuit& c,
		        const std::vector<std::pair<double, std::vector<std::pair<int, int>>>>& terms,
		        int levels) {
			     qubit::Ham H;
			     for (auto& t : terms) H.push_back({t.first, t.second});
			     return c.value_and_grad_q(H, levels);
		     },
		     py::arg("hamiltonian"), py::arg("levels"),
		     "Adjoint through int16 compression: returns (value, grad, D) where\n"
		     "D is the total injected L2 norm (compression budget). levels<=0\n"
		     "gives the exact result.");
}
